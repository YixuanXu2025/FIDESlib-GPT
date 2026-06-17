import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def simplify_component_inspect(info):
    out = {}
    for k, v in dict(info).items():
        if k != "parts":
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = str(v)

    parts_summary = []
    for part in info.get("parts", []):
        p = dict(part)
        towers = p.get("towers", [])
        parts_summary.append({
            "part_index": p.get("part_index"),
            "openfhe_name": p.get("openfhe_name"),
            "fides_raw_name": p.get("fides_raw_name"),
            "format": p.get("format"),
            "num_towers": p.get("num_towers"),
            "tower0_modulus_u64": towers[0].get("modulus_u64") if towers else None,
            "tower0_ring_dim": towers[0].get("ring_dim") if towers else None,
            "tower0_coeff_head": towers[0].get("coeff_head") if towers else None,
            "tower0_fnv1a64": towers[0].get("fnv1a64") if towers else None,
        })

    out["parts_summary"] = parts_summary
    return out


def main():
    slots = 8

    cfg = HEConfig(
        ring_dim=1 << 14,
        multiplicative_depth=2,
        scaling_mod_size=50,
        first_mod_size=60,
        num_large_digits=2,
        batch_size=slots,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    row = np.array([-3, -2, -1, 0, 1, 2, 3, 4], dtype=np.int64)

    report = {
        "experiment": "wp4p_coef_row_encrypt_probe",
        "purpose": (
            "Probe native OpenFHE CoefPackedEncoding encrypt/decrypt bridge "
            "through FIDESlib-GPT bindings."
        ),
        "status": "probe_only",
        "input_row_i64": row.tolist(),
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        report["runtime_info"] = safe_call("info", lambda: rt.info())

        # Baseline slot-packed path remains available.
        ct_slot = rt.encrypt([float(x) for x in row.tolist()])
        report["slot_baseline_decrypt"] = safe_call(
            "slot_decrypt",
            lambda: rt.decrypt(ct_slot, logical_length=len(row)),
        )
        report["slot_baseline_inspect"] = safe_call(
            "slot_inspect",
            lambda: simplify_component_inspect(
                rt.inspect_rlwe_components_cpu(ct_slot, coeff_sample=8)
            ),
        )

        # New coefficient-row path.
        coef_encrypt = safe_call(
            "encrypt_coeff_row_i64",
            lambda: rt.encrypt_coeff_row_i64(row.tolist()),
        )

        report["coef_encrypt"] = {
            "ok": coef_encrypt["ok"],
            "error": coef_encrypt.get("error"),
        }

        if coef_encrypt["ok"]:
            ct_coef = coef_encrypt["value"]

            report["coef_storage_state"] = safe_call(
                "coef_storage_state",
                lambda: dict(rt.ciphertext_storage_state(ct_coef)),
            )

            report["coef_decrypt"] = safe_call(
                "decrypt_coeff_row_i64",
                lambda: rt.decrypt_coeff_row_i64(ct_coef, logical_length=len(row)),
            )

            report["coef_inspect"] = safe_call(
                "coef_inspect",
                lambda: simplify_component_inspect(
                    rt.inspect_rlwe_components_cpu(ct_coef, coeff_sample=8)
                ),
            )

            if report["coef_decrypt"]["ok"]:
                dec = np.array(report["coef_decrypt"]["value"], dtype=np.int64)
                report["errors"] = {
                    "coef_roundtrip_exact": bool(np.array_equal(dec, row)),
                    "coef_roundtrip_max_abs_err": int(np.max(np.abs(dec - row))),
                }
            else:
                report["errors"] = {
                    "coef_roundtrip_exact": False,
                    "coef_roundtrip_max_abs_err": None,
                }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4p_coef_row_encrypt_probe_report.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 100)
    print("WP4-P coefficient row encrypt/decrypt probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
