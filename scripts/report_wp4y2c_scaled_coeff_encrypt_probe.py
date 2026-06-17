import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def round_divide(vals, scale):
    return [int(np.rint(int(v) / int(scale))) for v in vals]


def residual_after_unscale(vals, row, scale):
    if vals is None:
        return None

    scaled_ref = np.array(row, dtype=np.int64) * int(scale)
    got = np.array(vals, dtype=np.int64)
    residual = got - scaled_ref

    return {
        "residual": residual.tolist(),
        "residual_max_abs": int(np.max(np.abs(residual))),
        "residual_l1": int(np.sum(np.abs(residual))),
    }


def compare_row(got, ref):
    if got is None:
        return {
            "exact": False,
            "max_abs_err": None,
            "l1_err": None,
            "diff": None,
        }

    G = np.array(got, dtype=np.int64)
    R = np.array(ref, dtype=np.int64)
    D = G - R

    return {
        "exact": bool(np.array_equal(G, R)),
        "max_abs_err": int(np.max(np.abs(D))),
        "l1_err": int(np.sum(np.abs(D))),
        "diff": D.tolist(),
    }


def run_scaled_case(rt, row_name, row, scale):
    scaled_row = [int(v) * int(scale) for v in row]

    enc = safe_call(
        "encrypt_coeff_row_i64_scaled",
        lambda: rt.encrypt_coeff_row_i64(scaled_row),
    )

    out = {
        "row_name": row_name,
        "input_row": row,
        "scale": int(scale),
        "scaled_input_row": scaled_row,
        "encrypt": {
            "ok": enc["ok"],
            "error": enc.get("error"),
        },
    }

    if not enc["ok"]:
        return out

    comp = safe_call(
        "compress_towers_left_1",
        lambda: rt.compress_coeff_ct(enc["value"], towers_left=1),
    )

    out["compress"] = {
        "ok": comp["ok"],
        "error": comp.get("error"),
    }

    if not comp["ok"]:
        return out

    dec = safe_call(
        "decrypt_coeff_row_i64_after_compress",
        lambda: rt.decrypt_coeff_row_i64(comp["value"], logical_length=len(row)),
    )

    raw_decoded = dec.get("value") if dec["ok"] else None
    rounded = round_divide(raw_decoded, scale) if dec["ok"] else None

    out["decrypt_after_compress"] = {
        "ok": dec["ok"],
        "error": dec.get("error"),
        "raw_decoded_scaled_domain": raw_decoded,
        "rounded_after_dividing_by_scale": rounded,
    }

    out["compare_rounded_to_input"] = compare_row(rounded, row)
    out["scaled_domain_residual"] = residual_after_unscale(raw_decoded, row, scale)

    return out


def main():
    rows = {
        "zero8": [0, 0, 0, 0, 0, 0, 0, 0],
        "basis0": [1, 0, 0, 0, 0, 0, 0, 0],
        "basis1": [0, 1, 0, 0, 0, 0, 0, 0],
        "basis3": [0, 0, 0, 1, 0, 0, 0, 0],
        "small_mixed": [-3, -2, -1, 0, 1, 2, 3, 4],
        "alternating": [1, -1, 1, -1, 1, -1, 1, -1],
        "large_small_mix": [7, 0, -5, 2, -1, 0, 3, -4],
    }

    # Avoid scales too close to the 50-bit tower after compression.
    # Max input magnitude is 7, so 7*2^36 is still safely below 2^50.
    scales = [
        1,
        2**4,
        2**8,
        2**12,
        2**16,
        2**20,
        2**24,
        2**28,
        2**32,
        2**36,
    ]

    cfg = HEConfig(
        ring_dim=1 << 14,
        multiplicative_depth=2,
        first_mod_size=60,
        scaling_mod_size=50,
        num_large_digits=2,
        batch_size=8,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    report = {
        "experiment": "wp4y2c_scaled_coeff_encrypt_probe",
        "purpose": (
            "Test whether real encrypt_coeff_row_i64 becomes semantically correct when coefficient plaintexts "
            "are scaled before encryption and rounded after coefficient decrypt. Y2a showed unscaled messages "
            "are dominated by coefficient-domain encryption noise."
        ),
        "status": "scaled_coeff_encryption_probe",
        "scale_policy": {
            "value": "encrypt scale*row, decrypt, then round(decoded/scale)",
            "zh": "加密前把 coefficient 明文乘以 scale，解密后除以 scale 并四舍五入。",
        },
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for scale in scales:
            scale_report = {
                "scale": int(scale),
                "rows": [],
            }

            for row_name, row in rows.items():
                scale_report["rows"].append(
                    run_scaled_case(rt, row_name, row, scale)
                )

            report["cases"].append(scale_report)

    for scale_report in report["cases"]:
        rows_report = scale_report["rows"]

        scale_report["summary"] = {
            "all_encrypt_ok": all(r.get("encrypt", {}).get("ok") is True for r in rows_report),
            "all_compress_ok": all(r.get("compress", {}).get("ok") is True for r in rows_report),
            "all_decrypt_after_compress_ok": all(
                r.get("decrypt_after_compress", {}).get("ok") is True
                for r in rows_report
            ),
            "all_rounded_exact": all(
                r.get("compare_rounded_to_input", {}).get("exact") is True
                for r in rows_report
            ),
            "exact_count": sum(
                1 for r in rows_report
                if r.get("compare_rounded_to_input", {}).get("exact") is True
            ),
            "max_residual_abs_over_rows": max(
                [
                    r.get("scaled_domain_residual", {}).get("residual_max_abs", 0)
                    for r in rows_report
                    if r.get("scaled_domain_residual") is not None
                ] or [None]
            ),
        }

    successful_scales = [
        s["scale"]
        for s in report["cases"]
        if s["summary"]["all_rounded_exact"] is True
    ]

    best_scale_by_exact_count = sorted(
        [
            {
                "scale": s["scale"],
                "exact_count": s["summary"]["exact_count"],
                "max_residual_abs_over_rows": s["summary"]["max_residual_abs_over_rows"],
                "all_rounded_exact": s["summary"]["all_rounded_exact"],
            }
            for s in report["cases"]
        ],
        key=lambda x: (-x["exact_count"], x["max_residual_abs_over_rows"] if x["max_residual_abs_over_rows"] is not None else 10**18, x["scale"]),
    )

    report["checks"] = {
        "any_scale_roundtrip_all_exact（是否存在某个 scale 使所有 row roundtrip 精确）": len(successful_scales) > 0,
        "successful_scales（所有通过的 scale）": successful_scales,
        "best_scale_by_exact_count（按 exact_count 排序的 scale）": best_scale_by_exact_count,
    }

    if successful_scales:
        next_decision = {
            "value": "scaled_coeff_row_encryption_works",
            "recommended_scale": successful_scales[0],
            "next_step": "WP4-Y2d",
            "next_goal": "Add scaled encrypt/decrypt wrappers and re-run rowwise bundle + logical-n oracle with real scaled coefficient encryption.",
            "zh": "真实 coefficient-row encrypt path 可用，但必须使用 scale；下一步封装 scaled API。",
        }
    else:
        next_decision = {
            "value": "scaled_coeff_row_encryption_still_fails",
            "recommended_scale": None,
            "next_step": "WP4-Y2d",
            "next_goal": "Bypass current encrypt_coeff_row_i64 and implement a controlled RLWE/SK coefficient-row encryption probe.",
            "zh": "即使 scale 放大也不能 roundtrip，下一步才绕开当前 high-level coefficient encrypt。",
        }

    report["global_decision"] = next_decision

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4y2c_scaled_coeff_encrypt_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Y2c scaled coefficient-row encryption probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
