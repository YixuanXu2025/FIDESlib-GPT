import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


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


def main():
    row_a = np.array([1, -2, 3, -1], dtype=np.int64)
    row_b = np.array([2, 1, -1, 0], dtype=np.int64)

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
        "experiment": "wp4s1_mul_no_relin_probe",
        "purpose": (
            "Smoke test true component-level ciphertext-ciphertext multiply: "
            "(c0,c1)*(d0,d1)->(e0,e1,e2), before relinearization."
        ),
        "status": "primitive_probe_only_no_relin_no_rescale_no_full_ccmm",
        "input_row_a_i64": row_a.tolist(),
        "input_row_b_i64": row_b.tolist(),
        "formula": {
            "e0": {
                "value": "a0*b0",
                "zh": "三分量乘法结果的第 0 个 component",
            },
            "e1": {
                "value": "a0*b1 + a1*b0",
                "zh": "三分量乘法结果的第 1 个 component",
            },
            "e2": {
                "value": "a1*b1",
                "zh": "三分量乘法结果的第 2 个 component；后续 relinearization 会处理这一项",
            },
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        ct_a = rt.encrypt_coeff_row_i64(row_a.tolist())
        ct_b = rt.encrypt_coeff_row_i64(row_b.tolist())

        report["input_a"] = safe_call("inspect_a", lambda: inspect_rt(rt, ct_a))
        report["input_b"] = safe_call("inspect_b", lambda: inspect_rt(rt, ct_b))

        mul = safe_call("mul_coeff_ct_no_relin", lambda: rt.mul_coeff_ct_no_relin(ct_a, ct_b))
        report["mul_call"] = {
            "ok": mul["ok"],
            "error": mul.get("error"),
        }

        if mul["ok"]:
            ct_mul = mul["value"]
            report["mul_output"] = safe_call("inspect_mul", lambda: inspect_rt(rt, ct_mul))

            if report["mul_output"]["ok"]:
                val = report["mul_output"]["value"]
                parts = val.get("parts_summary", [])

                report["checks"] = {
                    "mul_call_ok（2-component×2-component 乘法调用成功）": True,
                    "mul_output_valid（乘法输出密文有效）": bool(val.get("valid")),
                    "mul_output_uses_coef_encoding（乘法输出仍是 coefficient encoding）": val.get("openfhe_encoding_type") == 1,
                    "mul_output_has_three_rlwe_parts（乘法输出有 c0/c1/c2 三个 RLWE component）": val.get("num_parts") == 3,
                    "mul_output_all_parts_have_4_towers（输出每个 component 都有 4 个 RNS towers）": all(p.get("num_towers") == 4 for p in parts),
                    "mul_output_all_parts_ring_dim_16384（输出每个 component 的 ring_dim 都是 16384）": all(p.get("tower0_ring_dim") == 16384 for p in parts),
                }
        else:
            report["checks"] = {
                "mul_call_ok（2-component×2-component 乘法调用成功）": False,
            }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4s1_mul_no_relin_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-S1 no-relin component ciphertext multiplication probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
