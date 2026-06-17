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
        "experiment": "wp4t1_relinearize_probe",
        "purpose": (
            "Probe native relinearization for no-relin 3-component coefficient ciphertext: "
            "(c0,c1,c2) -> (c0,c1)."
        ),
        "status": "relinearization_probe_only_no_rescale_no_final_decrypt",
        "input_row_a_i64": row_a.tolist(),
        "input_row_b_i64": row_b.tolist(),
        "formula": {
            "before": {
                "value": "mul_no_relin(ct_a, ct_b) -> c0/c1/c2",
                "zh": "先用 component-level no-relin 乘法生成三分量密文",
            },
            "after": {
                "value": "Relinearize(c0,c1,c2) -> c0/c1",
                "zh": "使用 relinearization key 把 c2 项压回二分量密文",
            },
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        ct_a = rt.encrypt_coeff_row_i64(row_a.tolist())
        ct_b = rt.encrypt_coeff_row_i64(row_b.tolist())

        prod = rt.mul_coeff_ct_no_relin(ct_a, ct_b)
        report["before_relin"] = safe_call("inspect_before_relin", lambda: inspect_rt(rt, prod))

        relin = safe_call("relinearize_coeff_ct", lambda: rt.relinearize_coeff_ct(prod))
        report["relin_call"] = {
            "ok": relin["ok"],
            "error": relin.get("error"),
        }

        if relin["ok"]:
            ct_relin = relin["value"]
            report["after_relin"] = safe_call("inspect_after_relin", lambda: inspect_rt(rt, ct_relin))

        before_ok = False
        if report["before_relin"]["ok"]:
            val = report["before_relin"]["value"]
            before_ok = (
                bool(val.get("valid"))
                and val.get("openfhe_encoding_type") == 1
                and val.get("num_parts") == 3
            )

        after_ok = False
        if report.get("after_relin", {}).get("ok"):
            val = report["after_relin"]["value"]
            parts = val.get("parts_summary", [])
            after_ok = (
                bool(val.get("valid"))
                and val.get("openfhe_encoding_type") == 1
                and val.get("num_parts") == 2
                and all(p.get("num_towers") == 4 for p in parts)
                and all(p.get("tower0_ring_dim") == 16384 for p in parts)
            )

        report["checks"] = {
            "before_relin_is_3part（重线性化前是 c0/c1/c2 三分量密文）": before_ok,
            "relin_call_ok（Relinearize 调用成功）": report["relin_call"]["ok"],
            "after_relin_valid（重线性化后密文有效）": after_ok,
            "after_relin_has_two_rlwe_parts（重线性化后压回 c0/c1 两个 RLWE component）": (
                report.get("after_relin", {}).get("ok")
                and report["after_relin"]["value"].get("num_parts") == 2
            ),
            "after_relin_uses_coef_encoding（重线性化后仍是 coefficient encoding）": (
                report.get("after_relin", {}).get("ok")
                and report["after_relin"]["value"].get("openfhe_encoding_type") == 1
            ),
            "after_relin_ring_dim_16384（重线性化后 ring_dim 仍是 16384）": (
                report.get("after_relin", {}).get("ok")
                and all(
                    p.get("tower0_ring_dim") == 16384
                    for p in report["after_relin"]["value"].get("parts_summary", [])
                )
            ),
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4t1_relinearize_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-T1 relinearization probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
