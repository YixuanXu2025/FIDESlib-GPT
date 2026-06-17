import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def nearest_div_int(x, scale):
    x = int(x)
    scale = int(scale)
    if x >= 0:
        return int((x + scale // 2) // scale)
    return int(-((-x + scale // 2) // scale))


def round_np(vals, scale):
    if vals is None:
        return None
    return [int(np.rint(int(v) / int(scale))) for v in vals]


def round_py(vals, scale):
    if vals is None:
        return None
    return [int(round(int(v) / int(scale))) for v in vals]


def round_int(vals, scale):
    if vals is None:
        return None
    return [nearest_div_int(v, scale) for v in vals]


def compare_row(got, ref):
    if got is None:
        return {
            "exact": False,
            "max_abs_err": None,
            "l1_err": None,
            "diff": None,
        }

    G = np.array(got, dtype=object)
    R = np.array(ref, dtype=object)
    D = G - R

    return {
        "exact": bool(all(int(x) == 0 for x in D.tolist())),
        "max_abs_err": int(max(abs(int(x)) for x in D.tolist())),
        "l1_err": int(sum(abs(int(x)) for x in D.tolist())),
        "diff": [int(x) for x in D.tolist()],
    }


def residual_report(decoded, scaled_ref, scale):
    if decoded is None:
        return None

    residual = [int(a) - int(b) for a, b in zip(decoded, scaled_ref)]
    max_abs = max(abs(x) for x in residual)
    threshold = int(scale) // 2

    return {
        "residual": residual,
        "residual_max_abs": int(max_abs),
        "residual_l1": int(sum(abs(x) for x in residual)),
        "rounding_threshold_scale_over_2": int(threshold),
        "residual_strictly_below_rounding_threshold": bool(max_abs < threshold),
    }


def run_one(rt, row_name, row, scale):
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

    decoded = dec.get("value") if dec["ok"] else None

    rounded_np = round_np(decoded, scale)
    rounded_py = round_py(decoded, scale)
    rounded_int = round_int(decoded, scale)

    out["decrypt_after_compress"] = {
        "ok": dec["ok"],
        "error": dec.get("error"),
        "raw_decoded_scaled_domain": decoded,
    }

    out["residual_scaled_domain"] = residual_report(decoded, scaled_row, scale)

    out["rounding"] = {
        "np_rint": {
            "value": rounded_np,
            "compare_to_input": compare_row(rounded_np, row),
        },
        "python_round": {
            "value": rounded_py,
            "compare_to_input": compare_row(rounded_py, row),
        },
        "integer_nearest": {
            "value": rounded_int,
            "compare_to_input": compare_row(rounded_int, row),
        },
    }

    out["audit_consistency"] = {
        "residual_below_threshold_implies_integer_exact": (
            out["residual_scaled_domain"] is not None
            and out["residual_scaled_domain"]["residual_strictly_below_rounding_threshold"]
            and out["rounding"]["integer_nearest"]["compare_to_input"]["exact"]
        ),
        "all_rounding_methods_agree": rounded_np == rounded_py == rounded_int,
    }

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

    scales = [
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
        "experiment": "wp4y2d_scaled_rounding_audit",
        "purpose": (
            "Audit the apparent contradiction in Y2c: large scales reported small residuals "
            "but did not report all rows as exact. This script prints per-row raw decoded, residual, "
            "and multiple rounding methods."
        ),
        "status": "rounding_audit_before_bypassing_encrypt_path",
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for scale in scales:
            scale_case = {
                "scale": int(scale),
                "rows": [],
            }

            for row_name, row in rows.items():
                scale_case["rows"].append(run_one(rt, row_name, row, scale))

            scale_case["summary"] = {
                "all_encrypt_ok": all(r.get("encrypt", {}).get("ok") is True for r in scale_case["rows"]),
                "all_compress_ok": all(r.get("compress", {}).get("ok") is True for r in scale_case["rows"]),
                "all_decrypt_ok": all(r.get("decrypt_after_compress", {}).get("ok") is True for r in scale_case["rows"]),
                "all_integer_nearest_exact": all(
                    r.get("rounding", {}).get("integer_nearest", {}).get("compare_to_input", {}).get("exact") is True
                    for r in scale_case["rows"]
                ),
                "integer_nearest_exact_count": sum(
                    1 for r in scale_case["rows"]
                    if r.get("rounding", {}).get("integer_nearest", {}).get("compare_to_input", {}).get("exact") is True
                ),
                "all_residuals_below_threshold": all(
                    r.get("residual_scaled_domain", {}).get("residual_strictly_below_rounding_threshold") is True
                    for r in scale_case["rows"]
                ),
                "all_rounding_methods_agree": all(
                    r.get("audit_consistency", {}).get("all_rounding_methods_agree") is True
                    for r in scale_case["rows"]
                ),
                "max_residual_abs": max(
                    [
                        r.get("residual_scaled_domain", {}).get("residual_max_abs", 0)
                        for r in scale_case["rows"]
                        if r.get("residual_scaled_domain") is not None
                    ] or [None]
                ),
            }

            report["cases"].append(scale_case)

    successful_scales = [
        c["scale"]
        for c in report["cases"]
        if c["summary"]["all_integer_nearest_exact"]
    ]

    report["checks"] = {
        "any_scale_all_integer_nearest_exact（是否存在 scale 使 integer nearest 全部精确）": len(successful_scales) > 0,
        "successful_scales（integer nearest 全部精确的 scale）": successful_scales,
        "scale_summaries（每个 scale 的汇总）": [
            {
                "scale": c["scale"],
                **c["summary"],
            }
            for c in report["cases"]
        ],
    }

    if successful_scales:
        report["global_decision"] = {
            "value": "scaled_coeff_row_encryption_works_with_integer_rounding",
            "recommended_scale": successful_scales[0],
            "next_step": "WP4-Y2e",
            "next_goal": "Patch or wrap scaled coefficient encrypt/decrypt using integer nearest rounding.",
            "zh": "scaled coefficient-row encryption 实际可用；Y2c 的失败来自 rounding/统计路径，需要封装 integer nearest rounding。",
        }
    else:
        report["global_decision"] = {
            "value": "scaled_coeff_row_encryption_really_fails",
            "recommended_scale": None,
            "next_step": "WP4-Y2e",
            "next_goal": "Bypass current encrypt_coeff_row_i64 and implement controlled RLWE/SK coefficient-row encryption probe.",
            "zh": "复核后仍失败，下一步绕开当前 high-level coefficient encrypt。",
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4y2d_scaled_rounding_audit_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Y2d scaled rounding audit")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
