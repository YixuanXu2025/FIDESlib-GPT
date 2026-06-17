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


def onepart_shape_ok(v):
    parts = v.get("parts_summary", [])
    return (
        bool(v.get("valid"))
        and v.get("openfhe_encoding_type") == 1
        and v.get("num_parts") == 1
        and len(parts) == 1
        and parts[0].get("num_towers") == 4
        and parts[0].get("tower0_ring_dim") == 16384
    )


def main():
    rng = np.random.default_rng(207200)
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
        "experiment": "wp4w2_component_primitive_probe",
        "purpose": (
            "Probe DCRTPoly component-level primitives needed for paper-aligned "
            "Algorithm 3 PP-MM: selected component multiply and one-component add."
        ),
        "status": "component_primitive_probe_only_no_full_ppmm_no_final_decrypt",
        "input_U_i64": U.tolist(),
        "input_V_i64": V.tolist(),
        "reference_U_matmul_V_i64": (U @ V).tolist(),
        "component_mapping": {
            "L0": "U.rows[*].GetElements()[0] / c0",
            "L1": "U.rows[*].GetElements()[1] / c1",
            "R0": "CMT(V)[*].GetElements()[0] / c0",
            "R1": "CMT(V)[*].GetElements()[1] / c1",
            "paper_formula": {
                "C0": "PP-MM(L0, R0)",
                "C1": "PP-MM(L0, R1) + PP-MM(L1, R0)",
                "C2": "PP-MM(L1, R1)",
            },
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
        V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())
        V_cmt, V_cmt_trace = cmt_algorithm2_rowwise(rt, V_bundle)

        # Use row/index 0 as smoke-test input for component primitives.
        u0 = U_bundle.rows[0]
        r0 = V_cmt[0]

        cases = [
            {
                "name": "L0_R0_for_C0",
                "left_part": 0,
                "right_part": 0,
                "formula": "component_mul(L0, R0)",
                "zh": "对应 C0 = PP-MM(L0,R0) 的单项 component multiply",
            },
            {
                "name": "L0_R1_for_C1_left",
                "left_part": 0,
                "right_part": 1,
                "formula": "component_mul(L0, R1)",
                "zh": "对应 C1 左半项 PP-MM(L0,R1)",
            },
            {
                "name": "L1_R0_for_C1_right",
                "left_part": 1,
                "right_part": 0,
                "formula": "component_mul(L1, R0)",
                "zh": "对应 C1 右半项 PP-MM(L1,R0)",
            },
            {
                "name": "L1_R1_for_C2",
                "left_part": 1,
                "right_part": 1,
                "formula": "component_mul(L1, R1)",
                "zh": "对应 C2 = PP-MM(L1,R1) 的单项 component multiply",
            },
        ]

        product_outputs = {}
        product_reports = []

        for c in cases:
            mul = safe_call(
                c["name"],
                lambda c=c: rt.component_mul_part_coeff_ct(
                    u0,
                    c["left_part"],
                    r0,
                    c["right_part"],
                ),
            )

            entry = {
                "name": c["name"],
                "left_part": c["left_part"],
                "right_part": c["right_part"],
                "formula": c["formula"],
                "zh": c["zh"],
                "call": {
                    "ok": mul["ok"],
                    "error": mul.get("error"),
                },
            }

            if mul["ok"]:
                product_outputs[c["name"]] = mul["value"]
                entry["inspect"] = safe_call(
                    "inspect_" + c["name"],
                    lambda mul=mul: inspect_rt(rt, mul["value"]),
                )

            product_reports.append(entry)

        report["component_mul_cases"] = product_reports

        # Probe C1-style one-component add:
        #   component_mul(L0,R1) + component_mul(L1,R0)
        add_report = {
            "formula": "component_add(component_mul(L0,R1), component_mul(L1,R0))",
            "zh": "对应 Algorithm 3 里 C1 的两个 PP-MM 项相加的最小 primitive probe",
        }

        if "L0_R1_for_C1_left" in product_outputs and "L1_R0_for_C1_right" in product_outputs:
            add = safe_call(
                "component_add_C1_terms",
                lambda: rt.component_add_1part_coeff_ct(
                    product_outputs["L0_R1_for_C1_left"],
                    product_outputs["L1_R0_for_C1_right"],
                ),
            )
            add_report["call"] = {
                "ok": add["ok"],
                "error": add.get("error"),
            }
            if add["ok"]:
                add_report["inspect"] = safe_call(
                    "inspect_component_add_C1_terms",
                    lambda: inspect_rt(rt, add["value"]),
                )
        else:
            add_report["call"] = {
                "ok": False,
                "error": "missing product terms",
            }

        report["component_add_probe"] = add_report

        product_shape_checks = []
        for entry in product_reports:
            ok = (
                entry.get("inspect", {}).get("ok") is True
                and onepart_shape_ok(entry["inspect"]["value"])
            )
            product_shape_checks.append(ok)

        add_shape_ok = (
            add_report.get("inspect", {}).get("ok") is True
            and onepart_shape_ok(add_report["inspect"]["value"])
        )

        report["checks"] = {
            "all_component_mul_calls_ok（四种 L0/L1 × R0/R1 component multiply 全部调用成功）": all(
                e["call"]["ok"] for e in product_reports
            ),
            "all_component_mul_outputs_are_1part（所有 component multiply 输出都是 1-component handle）": all(product_shape_checks),
            "component_add_call_ok（C1 两个 one-component 项相加调用成功）": add_report["call"]["ok"],
            "component_add_output_is_1part（component add 输出仍是 1-component handle）": add_shape_ok,
            "V_cmt_auto_alphas_expected（C-MT(V) Auto alpha 是 [1,3,5,7]）": [
                x["alpha"] for x in V_cmt_trace["auto"]
            ] == [1, 3, 5, 7],
            "ready_for_W3_ppmm_skeleton（已具备实现 PP-MM skeleton 的 component primitive）": (
                all(e["call"]["ok"] for e in product_reports)
                and all(product_shape_checks)
                and add_report["call"]["ok"]
                and add_shape_ok
            ),
        }

        report["summary"] = {
            "next_step": "WP4-W3",
            "next_step_goal": (
                "Use component_mul_part_coeff_ct and component_add_1part_coeff_ct "
                "to build paper-aligned C0/C1/C2 one-component matrices, then assemble "
                "3-component ciphertexts without using whole-ciphertext multiplication."
            ),
            "zh": "下一步用 component primitive 组装论文 Algorithm 3 的 C0/C1/C2，而不是整密文相乘。",
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4w2_component_primitive_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-W2 component primitive probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
