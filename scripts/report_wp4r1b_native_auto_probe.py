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
    if not info:
        return None

    parts = info.get("parts", [])
    return {
        "valid": info.get("valid"),
        "fides_loaded": info.get("fides_loaded"),
        "openfhe_level": info.get("openfhe_level"),
        "openfhe_noise_scale_deg": info.get("openfhe_noise_scale_deg"),
        "openfhe_scaling_factor": info.get("openfhe_scaling_factor"),
        "openfhe_slots": info.get("openfhe_slots"),
        "openfhe_encoding_type": info.get("openfhe_encoding_type"),
        "num_parts": info.get("num_parts"),
        "num_towers_part0": parts[0].get("num_towers") if parts else None,
        "ring_dim_part0_tower0": (
            parts[0].get("towers", [{}])[0].get("ring_dim") if parts else None
        ),
        "tower0_coeff_head_part0": (
            parts[0].get("towers", [{}])[0].get("coeff_head") if parts else None
        ),
    }


def main():
    row = np.array([1, -2, 3, -1], dtype=np.int64)

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

    # For N=4 paper toy C-MT, Algorithm 2 uses alpha = 2j + 1 -> 1,3,5,7.
    # For real ring dimension 16384, these are valid odd automorphism indices modulo 32768.
    alphas = [1, 3, 5, 7]

    report = {
        "experiment": "wp4r1b_native_auto_probe",
        "purpose": "Probe native OpenFHE EvalAutomorphism(ct, alpha) on coefficient ciphertext.",
        "status": "single_auto_probe_no_full_cmt",
        "input_row_i64": row.tolist(),
        "alphas": alphas,
        "current_state": {
            "completed": [
                "WP4-Q2 real row-wise coefficient ciphertext matrix bundle",
                "WP4-R1a API availability probe",
            ],
            "this_step": "native Auto(ct; alpha) smoke test",
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        ct = rt.encrypt_coeff_row_i64(row.tolist())

        before = safe_call(
            "inspect_before",
            lambda: compact_inspect(rt.inspect_rlwe_components_cpu(ct, coeff_sample=8)),
        )
        report["before"] = before

        cases = []
        for alpha in alphas:
            case = {"alpha": alpha}

            autoed = safe_call(
                f"eval_automorphism_alpha_{alpha}",
                lambda alpha=alpha: rt.eval_automorphism_coeff_ct(ct, alpha),
            )

            case["auto_call"] = {
                "ok": autoed["ok"],
                "error": autoed.get("error"),
            }

            if autoed["ok"]:
                ct_auto = autoed["value"]
                insp = safe_call(
                    f"inspect_auto_alpha_{alpha}",
                    lambda ct_auto=ct_auto: compact_inspect(
                        rt.inspect_rlwe_components_cpu(ct_auto, coeff_sample=8)
                    ),
                )
                case["after"] = insp

                if insp["ok"]:
                    val = insp["value"]
                    case["checks"] = {
                        "auto_output_is_valid（Auto 输出密文有效）": bool(val.get("valid")),
                        "auto_output_is_coef_encoding_type_1（Auto 输出仍是 coefficient encoding 类型 1）": val.get("openfhe_encoding_type") == 1,
                        "auto_output_has_two_rlwe_parts（Auto 输出仍有 c0/c1 两个 RLWE component）": val.get("num_parts") == 2,
                        "auto_output_ring_dim_16384（Auto 输出 ring_dim 仍是 16384）": val.get("ring_dim_part0_tower0") == 16384,
                    }

            cases.append(case)

        report["cases"] = cases
        report["summary"] = {
            "all_auto_calls_ok（所有 Auto 调用都成功）": all(c["auto_call"]["ok"] for c in cases),
            "successful_alphas（成功执行 Auto 的 alpha 列表）": [
                c["alpha"] for c in cases if c["auto_call"]["ok"]
            ],
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4r1b_native_auto_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-R1b native automorphism coefficient ciphertext probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
