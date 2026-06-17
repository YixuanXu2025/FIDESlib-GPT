import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def make_config(**kwargs):
    if is_dataclass(HEConfig):
        allowed = {f.name for f in fields(HEConfig)}
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return HEConfig(**kwargs)


def max_abs_err(a, b):
    if a is None or b is None:
        return None
    A = np.array(a, dtype=np.int64)
    B = np.array(b, dtype=np.int64)
    return int(np.max(np.abs(A - B)))


def l1_err(a, b):
    if a is None or b is None:
        return None
    A = np.array(a, dtype=np.int64)
    B = np.array(b, dtype=np.int64)
    return int(np.sum(np.abs(A - B)))


def rotate_left(xs, k):
    k %= len(xs)
    return xs[k:] + xs[:k]


def anti_cyclic_shift(xs, k):
    """
    Coefficient ring convention for X^N = -1.
    For logical small prefix this is only a diagnostic variant.
    """
    n = len(xs)
    out = [0 for _ in range(n)]
    for i, v in enumerate(xs):
        j = i + k
        if j < n:
            out[j] += v
        else:
            out[j - n] -= v
    return out


def candidate_transforms(row):
    n = len(row)
    candidates = {
        "identity": row,
        "neg_identity": [-v for v in row],
        "reverse": list(reversed(row)),
        "neg_reverse": [-v for v in reversed(row)],
    }

    for k in range(n):
        candidates[f"rot_left_{k}"] = rotate_left(row, k)
        candidates[f"neg_rot_left_{k}"] = [-v for v in rotate_left(row, k)]
        candidates[f"anti_cyclic_shift_{k}"] = anti_cyclic_shift(row, k)
        candidates[f"neg_anti_cyclic_shift_{k}"] = [-v for v in anti_cyclic_shift(row, k)]

    return candidates


def classify(decoded, row):
    if decoded is None:
        return {
            "best_match": None,
            "exact_identity": False,
            "best_max_abs_err": None,
            "best_l1_err": None,
        }

    candidates = candidate_transforms(row)
    scored = []

    for name, cand in candidates.items():
        scored.append({
            "name": name,
            "max_abs_err": max_abs_err(decoded, cand),
            "l1_err": l1_err(decoded, cand),
            "candidate": cand,
        })

    scored.sort(key=lambda x: (x["max_abs_err"], x["l1_err"], x["name"]))

    return {
        "exact_identity": decoded == row,
        "best_match": scored[0],
        "top5": scored[:5],
    }


def decrypt_direct_and_compressed(rt, ct, logical_length):
    direct = safe_call(
        "decrypt_direct",
        lambda: rt.decrypt_coeff_row_i64(ct, logical_length=logical_length),
    )

    comp = safe_call(
        "compress_towers_left_1",
        lambda: rt.compress_coeff_ct(ct, towers_left=1),
    )

    if comp["ok"]:
        after_comp = safe_call(
            "decrypt_after_compress",
            lambda: rt.decrypt_coeff_row_i64(comp["value"], logical_length=logical_length),
        )
    else:
        after_comp = {
            "ok": False,
            "error": "compress failed: " + comp.get("error", ""),
        }

    return {
        "direct": {
            "ok": direct["ok"],
            "value": direct.get("value"),
            "error": direct.get("error"),
        },
        "compress": {
            "ok": comp["ok"],
            "error": comp.get("error"),
        },
        "after_compress": {
            "ok": after_comp["ok"],
            "value": after_comp.get("value"),
            "error": after_comp.get("error"),
        },
    }


def run_case(rt, name, row):
    enc = safe_call(
        "encrypt_coeff_row_i64",
        lambda: rt.encrypt_coeff_row_i64(row),
    )

    result = {
        "name": name,
        "input_row": row,
        "encrypt": {
            "ok": enc["ok"],
            "error": enc.get("error"),
        },
    }

    if not enc["ok"]:
        result["decrypt"] = None
        result["classification"] = None
        return result

    dec = decrypt_direct_and_compressed(rt, enc["value"], logical_length=len(row))
    result["decrypt"] = dec

    direct_val = dec["direct"].get("value") if dec["direct"]["ok"] else None
    comp_val = dec["after_compress"].get("value") if dec["after_compress"]["ok"] else None

    result["classification"] = {
        "direct": classify(direct_val, row),
        "after_compress": classify(comp_val, row),
    }

    result["errors"] = {
        "direct_max_abs_err_to_input": max_abs_err(direct_val, row),
        "direct_l1_err_to_input": l1_err(direct_val, row),
        "after_compress_max_abs_err_to_input": max_abs_err(comp_val, row),
        "after_compress_l1_err_to_input": l1_err(comp_val, row),
    }

    return result


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

    param_sets = [
        {
            "name": "default_depth2_first60_scale50",
            "cfg_kwargs": {
                "ring_dim": 1 << 14,
                "multiplicative_depth": 2,
                "first_mod_size": 60,
                "scaling_mod_size": 50,
                "num_large_digits": 2,
                "batch_size": 8,
                "devices": (0,),
                "plaintext_autoload": True,
                "ciphertext_autoload": True,
                "with_mult_key": True,
            },
        },
        {
            "name": "depth0_first60_scale50",
            "cfg_kwargs": {
                "ring_dim": 1 << 14,
                "multiplicative_depth": 0,
                "first_mod_size": 60,
                "scaling_mod_size": 50,
                "num_large_digits": 2,
                "batch_size": 8,
                "devices": (0,),
                "plaintext_autoload": True,
                "ciphertext_autoload": True,
                "with_mult_key": True,
            },
        },
        {
            "name": "depth1_first60_scale50",
            "cfg_kwargs": {
                "ring_dim": 1 << 14,
                "multiplicative_depth": 1,
                "first_mod_size": 60,
                "scaling_mod_size": 50,
                "num_large_digits": 2,
                "batch_size": 8,
                "devices": (0,),
                "plaintext_autoload": True,
                "ciphertext_autoload": True,
                "with_mult_key": True,
            },
        },
        {
            "name": "default_depth2_first55_scale50",
            "cfg_kwargs": {
                "ring_dim": 1 << 14,
                "multiplicative_depth": 2,
                "first_mod_size": 55,
                "scaling_mod_size": 50,
                "num_large_digits": 2,
                "batch_size": 8,
                "devices": (0,),
                "plaintext_autoload": True,
                "ciphertext_autoload": True,
                "with_mult_key": True,
            },
        },
    ]

    report = {
        "experiment": "wp4y2a_coeff_encrypt_failure_classifier",
        "purpose": (
            "Classify why real encrypt_coeff_row_i64 -> decrypt_coeff_row_i64 does not roundtrip. "
            "This separates sign/order/rotation convention bugs from scale/noise/encryption-path bugs."
        ),
        "status": "diagnostic_before_fixing_real_coeff_row_encryption",
        "cases": [],
    }

    for ps in param_sets:
        cfg = make_config(**ps["cfg_kwargs"])
        ps_report = {
            "param_set": ps["name"],
            "cfg_kwargs_requested": ps["cfg_kwargs"],
            "rows": [],
        }

        try:
            with HERuntime(cfg, rotation_steps=[]) as rt:
                ps_report["context_init"] = {
                    "ok": True,
                    "error": None,
                }
                for name, row in rows.items():
                    ps_report["rows"].append(run_case(rt, name, row))
        except Exception as e:
            ps_report["context_init"] = {
                "ok": False,
                "error": repr(e),
                "zh": "该参数组无法初始化 OpenFHE/FIDESlib context；跳过该组，不中断整个诊断脚本。",
            }

        report["cases"].append(ps_report)

    all_encrypt_ok = all(
        r["encrypt"]["ok"]
        for ps in report["cases"]
        for r in ps["rows"]
    )

    after_comp_ok = all(
        r["decrypt"]["after_compress"]["ok"]
        for ps in report["cases"]
        for r in ps["rows"]
        if r["encrypt"]["ok"]
    )

    after_comp_exact = all(
        r["classification"]["after_compress"]["exact_identity"]
        for ps in report["cases"]
        for r in ps["rows"]
        if r["encrypt"]["ok"] and r["decrypt"]["after_compress"]["ok"]
    )

    best_matches = [
        {
            "param_set": ps["param_set"],
            "name": r["name"],
            "best_after_compress": None
            if not r.get("classification")
            else r["classification"]["after_compress"]["best_match"],
            "decoded_after_compress": None
            if not r.get("decrypt")
            else r["decrypt"]["after_compress"].get("value"),
            "input_row": r["input_row"],
        }
        for ps in report["cases"]
        for r in ps["rows"]
    ]

    report["checks"] = {
        "all_encrypt_calls_ok（所有 encrypt_coeff_row_i64 调用成功）": all_encrypt_ok,
        "all_after_compress_decrypt_calls_ok（所有 compress 后 decrypt 调用成功）": after_comp_ok,
        "all_after_compress_roundtrip_exact（所有 compress 后 roundtrip 精确）": after_comp_exact,
    }

    report["diagnostic_summary"] = {
        "best_matches_after_compress": best_matches,
        "interpretation": {
            "if_best_match_identity_small_error": "Likely rounding/noise/scale issue; inspect coefficient plaintext scale and modulus compression.",
            "if_best_match_rotation_or_reverse": "Likely coefficient order / automorphism / anti-cyclic placement issue.",
            "if_best_match_large_error_for_zero": "Encryption/decryption path is semantically broken even for zero.",
            "if_zero_exact_but_basis_wrong": "Likely placement/encoding convention bug.",
            "if_all_exact": "Real coefficient-row encryption baseline is fixed; proceed to logical-n C-MT real-design work.",
        },
        "next_step": "WP4-Y2b",
        "zh": "Y2a 先分类失败形态，再决定是修 encode/decode、修 compress/scale，还是绕开 OpenFHE high-level coefficient encrypt。"
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4y2a_coeff_encrypt_failure_classifier_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Y2a coefficient-row encryption failure classifier")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
