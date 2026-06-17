import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def one_case(depth, first_mod_size=60, scaling_mod_size=50):
    row = np.array([-3, -2, -1, 0, 1, 2, 3, 4], dtype=np.int64)

    cfg = HEConfig(
        ring_dim=1 << 14,
        multiplicative_depth=depth,
        scaling_mod_size=scaling_mod_size,
        first_mod_size=first_mod_size,
        num_large_digits=2,
        batch_size=8,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    case = {
        "multiplicative_depth": depth,
        "first_mod_size": first_mod_size,
        "scaling_mod_size": scaling_mod_size,
        "input_row": row.tolist(),
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        enc = safe_call("encrypt_coeff_row_i64", lambda: rt.encrypt_coeff_row_i64(row.tolist()))
        case["coef_encrypt"] = {"ok": enc["ok"], "error": enc.get("error")}

        if enc["ok"]:
            ct = enc["value"]

            case["coef_inspect"] = safe_call(
                "inspect",
                lambda: rt.inspect_rlwe_components_cpu(ct, coeff_sample=8),
            )

            # Keep inspect report compact.
            if case["coef_inspect"]["ok"]:
                info = case["coef_inspect"]["value"]
                case["coef_inspect_summary"] = {
                    "openfhe_level": info.get("openfhe_level"),
                    "openfhe_noise_scale_deg": info.get("openfhe_noise_scale_deg"),
                    "openfhe_scaling_factor": info.get("openfhe_scaling_factor"),
                    "openfhe_slots": info.get("openfhe_slots"),
                    "openfhe_encoding_type": info.get("openfhe_encoding_type"),
                    "num_parts": info.get("num_parts"),
                    "num_towers_part0": (
                        info.get("parts", [{}])[0].get("num_towers")
                        if info.get("parts") else None
                    ),
                }
                del case["coef_inspect"]

            dec = safe_call(
                "decrypt_coeff_row_i64",
                lambda: rt.decrypt_coeff_row_i64(ct, logical_length=len(row)),
            )
            case["coef_decrypt"] = dec

            if dec["ok"]:
                got = np.array(dec["value"], dtype=np.int64)
                case["errors"] = {
                    "roundtrip_exact": bool(np.array_equal(got, row)),
                    "roundtrip_max_abs_err": int(np.max(np.abs(got - row))),
                }
            else:
                case["errors"] = {
                    "roundtrip_exact": False,
                    "roundtrip_max_abs_err": None,
                }

    return case


def main():
    report = {
        "experiment": "wp4p4a_single_limb_coef_decrypt_probe",
        "purpose": (
            "Test whether native OpenFHE coefficient-row decrypt works when CKKS "
            "ciphertext has one RNS limb / sizeQl == 1."
        ),
        "status": "probe_only",
        "cases": [],
    }

    # Try a small sweep. The previous failure was sizeQl 4 != 1.
    for depth in [0, 1, 2]:
        report["cases"].append(one_case(depth))

    report["summary"] = {
        "any_roundtrip_exact": any(
            c.get("errors", {}).get("roundtrip_exact") is True
            for c in report["cases"]
        ),
        "successful_depths": [
            c["multiplicative_depth"]
            for c in report["cases"]
            if c.get("errors", {}).get("roundtrip_exact") is True
        ],
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4p4a_single_limb_coef_decrypt_probe_report.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 100)
    print("WP4-P4a single-limb coefficient decrypt probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
