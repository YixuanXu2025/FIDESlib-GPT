import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def main():
    row = np.array([-3, -2, -1, 0, 1, 2, 3, 4], dtype=np.int64)

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
        "experiment": "wp4p4b_manual_coef_decrypt_probe",
        "purpose": (
            "Manually decrypt native coefficient-row ciphertext using RLWE phase "
            "phase = c0 + c1 * sk, bypassing OpenFHE high-level coefficient decrypt."
        ),
        "status": "debug_probe_only",
        "input_row_i64": row.tolist(),
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        enc = safe_call(
            "encrypt_coeff_row_i64",
            lambda: rt.encrypt_coeff_row_i64(row.tolist()),
        )

        report["coef_encrypt"] = {
            "ok": enc["ok"],
            "error": enc.get("error"),
        }

        if enc["ok"]:
            ct = enc["value"]

            report["openfhe_high_level_decrypt"] = safe_call(
                "decrypt_coeff_row_i64",
                lambda: rt.decrypt_coeff_row_i64(ct, logical_length=len(row)),
            )

            report["manual_decrypt"] = safe_call(
                "manual_decrypt_coeff_row_i64",
                lambda: rt.manual_decrypt_coeff_row_i64(ct, logical_length=len(row)),
            )

            if report["manual_decrypt"]["ok"]:
                dec = np.array(report["manual_decrypt"]["value"], dtype=np.int64)
                diff = dec - row
                report["manual_decrypt_first_values"] = dec.tolist()
                report["errors"] = {
                    "manual_roundtrip_exact": bool(np.array_equal(dec, row)),
                    "manual_roundtrip_max_abs_err": int(np.max(np.abs(diff))),
                    "manual_diff": diff.tolist(),
                }
            else:
                report["errors"] = {
                    "manual_roundtrip_exact": False,
                    "manual_roundtrip_max_abs_err": None,
                    "manual_diff": None,
                }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4p4b_manual_coef_decrypt_probe_report.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 100)
    print("WP4-P4b manual coefficient decrypt probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
