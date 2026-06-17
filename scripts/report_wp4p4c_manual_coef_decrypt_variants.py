import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def score(vec, target):
    arr = np.array(vec, dtype=np.int64)
    tgt = np.array(target, dtype=np.int64)
    diff = arr - tgt
    return {
        "values": arr.tolist(),
        "exact": bool(np.array_equal(arr, tgt)),
        "max_abs_err": int(np.max(np.abs(diff))),
        "l2_sq": int(np.sum(diff * diff)),
        "diff": diff.tolist(),
    }


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
        "experiment": "wp4p4c_manual_coef_decrypt_variants",
        "purpose": "Test RLWE phase formula/component-order/sign variants for coefficient-row ciphertext.",
        "status": "debug_probe_only",
        "input_row_i64": row.tolist(),
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        enc = safe_call("encrypt_coeff_row_i64", lambda: rt.encrypt_coeff_row_i64(row.tolist()))
        report["coef_encrypt"] = {"ok": enc["ok"], "error": enc.get("error")}

        if enc["ok"]:
            ct = enc["value"]

            variants = safe_call(
                "manual_decrypt_coeff_row_i64_variants",
                lambda: rt.manual_decrypt_coeff_row_i64_variants(ct, logical_length=len(row)),
            )

            report["variants_call"] = {
                "ok": variants["ok"],
                "error": variants.get("error"),
            }

            if variants["ok"]:
                scored = {
                    name: score(values, row)
                    for name, values in variants["value"].items()
                }

                best_name = min(scored, key=lambda k: scored[k]["l2_sq"])

                report["variants"] = scored
                report["best"] = {
                    "name": best_name,
                    **scored[best_name],
                }
                report["summary"] = {
                    "any_exact": any(v["exact"] for v in scored.values()),
                    "exact_variants": [k for k, v in scored.items() if v["exact"]],
                    "best_name": best_name,
                    "best_max_abs_err": scored[best_name]["max_abs_err"],
                    "best_l2_sq": scored[best_name]["l2_sq"],
                }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4p4c_manual_coef_decrypt_variants_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-P4c manual coefficient decrypt variants")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
