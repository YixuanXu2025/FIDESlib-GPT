import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    cmt_algorithm2_rowwise,
)


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def compact_inspect(info):
    parts = info.get("parts", []) if info else []
    return {
        "valid": info.get("valid") if info else None,
        "openfhe_level": info.get("openfhe_level") if info else None,
        "openfhe_noise_scale_deg": info.get("openfhe_noise_scale_deg") if info else None,
        "openfhe_scaling_factor": info.get("openfhe_scaling_factor") if info else None,
        "openfhe_slots": info.get("openfhe_slots") if info else None,
        "openfhe_encoding_type": info.get("openfhe_encoding_type") if info else None,
        "num_parts": info.get("num_parts") if info else None,
        "parts_summary": [
            {
                "part_index": p.get("part_index"),
                "num_towers": p.get("num_towers"),
                "tower0_ring_dim": p.get("towers", [{}])[0].get("ring_dim") if p.get("towers") else None,
                "tower0_coeff_head": p.get("towers", [{}])[0].get("coeff_head") if p.get("towers") else None,
            }
            for p in parts
        ],
    }


def inspect_rt(rt, ct):
    return compact_inspect(rt.inspect_rlwe_components_cpu(ct, coeff_sample=8))


def shape_ok_parts(v, expected_parts):
    parts = v.get("parts_summary", [])
    return (
        bool(v.get("valid"))
        and v.get("openfhe_encoding_type") == 1
        and v.get("num_parts") == expected_parts
        and len(parts) == expected_parts
        and all(p.get("num_towers") == 4 for p in parts)
        and all(p.get("tower0_ring_dim") == 16384 for p in parts)
    )


def main():
    rng = np.random.default_rng(207300)
    U = rng.integers(-2, 3, size=(4, 4)).astype(np.int64)
    V = rng.integers(-2, 3, size=(4, 4)).astype(np.int64)

    cfg = HEConfig(
        ring_dim=1 << 14,
        multiplicative_depth=2,
        scaling_mod_size=50,
        first_mod_size=60,
        num_large_digits=2,
        batch_size=8,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    report = {
        "experiment": "wp4w3a_assemble_algorithm3_components",
        "purpose": (
            "Assemble a paper-aligned Algorithm 3 three-component ciphertext from "
            "component primitive outputs: C0=L0R0, C1=L0R1+L1R0, C2=L1R1."
        ),
        "status": "single_pair_component_formula_assembly_probe_no_full_ppmm_no_final_decrypt",
        "input_U_i64": U.tolist(),
        "input_V_i64": V.tolist(),
        "reference_U_matmul_V_i64": (U @ V).tolist(),
        "formula": {
            "C0": "component_mul(L0, R0)",
            "C1": "component_add(component_mul(L0, R1), component_mul(L1, R0))",
            "C2": "component_mul(L1, R1)",
            "assemble": "(C0, C1, C2)",
            "zh": "这里 L0/L1 来自 U 的 c0/c1，R0/R1 来自 C-MT(V) 的 c0/c1。",
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
        V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())
        V_cmt, V_cmt_trace = cmt_algorithm2_rowwise(rt, V_bundle)

        # Single-pair structural probe:
        # left row 0, right transformed index 0.
        left = U_bundle.rows[0]
        right = V_cmt[0]

        C0 = rt.component_mul_part_coeff_ct(left, 0, right, 0)

        C1_left = rt.component_mul_part_coeff_ct(left, 0, right, 1)
        C1_right = rt.component_mul_part_coeff_ct(left, 1, right, 0)
        C1 = rt.component_add_1part_coeff_ct(C1_left, C1_right)

        C2 = rt.component_mul_part_coeff_ct(left, 1, right, 1)

        assembled = rt.assemble_3part_from_1parts_coeff_ct(C0, C1, C2)

        # For sanity, compare structure with old no-relin whole-ciphertext multiply.
        # This is not used as the final algorithm, only as a structural reference.
        whole_no_relin = rt.mul_coeff_ct_no_relin(left, right)

        report["component_outputs"] = {
            "C0": safe_call("inspect_C0", lambda: inspect_rt(rt, C0)),
            "C1_left": safe_call("inspect_C1_left", lambda: inspect_rt(rt, C1_left)),
            "C1_right": safe_call("inspect_C1_right", lambda: inspect_rt(rt, C1_right)),
            "C1_sum": safe_call("inspect_C1_sum", lambda: inspect_rt(rt, C1)),
            "C2": safe_call("inspect_C2", lambda: inspect_rt(rt, C2)),
        }

        report["assembled_output"] = safe_call(
            "inspect_assembled_3part",
            lambda: inspect_rt(rt, assembled),
        )

        report["whole_no_relin_reference"] = safe_call(
            "inspect_whole_no_relin_reference",
            lambda: inspect_rt(rt, whole_no_relin),
        )

        component_ok = all(
            item["ok"] and shape_ok_parts(item["value"], 1)
            for item in report["component_outputs"].values()
        )

        assembled_ok = (
            report["assembled_output"]["ok"]
            and shape_ok_parts(report["assembled_output"]["value"], 3)
        )

        whole_ref_ok = (
            report["whole_no_relin_reference"]["ok"]
            and shape_ok_parts(report["whole_no_relin_reference"]["value"], 3)
        )

        report["checks"] = {
            "all_C0_C1_C2_terms_are_1part（C0/C1/C2 相关 component 项都是 1-component）": component_ok,
            "assembled_output_is_3part（组装后输出是 c0/c1/c2 三分量 ciphertext）": assembled_ok,
            "assembled_output_uses_coef_encoding（组装后仍是 coefficient encoding）": (
                report["assembled_output"]["ok"]
                and report["assembled_output"]["value"].get("openfhe_encoding_type") == 1
            ),
            "whole_no_relin_reference_is_3part（旧 no-relin 参考输出也是三分量）": whole_ref_ok,
            "V_cmt_auto_alphas_expected（C-MT(V) Auto alpha 是 [1,3,5,7]）": [
                x["alpha"] for x in V_cmt_trace["auto"]
            ] == [1, 3, 5, 7],
            "ready_for_W3b_component_ppmm_matrix（已具备组装单个三分量输出的能力）": (
                component_ok and assembled_ok
            ),
        }

        report["summary"] = {
            "next_step": "WP4-W3b",
            "next_step_goal": (
                "Lift this single-pair C0/C1/C2 component assembly into a matrix-level "
                "component PP-MM skeleton, then diagnose whether PP-MM placement must follow "
                "the paper's Algorithm 3 rather than old positionwise multiplication."
            ),
            "zh": "下一步把单个三分量组装推广到 component matrix PP-MM 层。",
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4w3a_assemble_algorithm3_components_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-W3a assemble Algorithm 3 components")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
