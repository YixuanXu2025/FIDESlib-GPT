import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def inspect_compact(rt, ct):
    info = rt.inspect_rlwe_components_cpu(ct, coeff_sample=8)
    parts = info.get("parts", [])
    return {
        "valid": info.get("valid"),
        "openfhe_level": info.get("openfhe_level"),
        "openfhe_noise_scale_deg": info.get("openfhe_noise_scale_deg"),
        "openfhe_scaling_factor": info.get("openfhe_scaling_factor"),
        "openfhe_slots": info.get("openfhe_slots"),
        "openfhe_encoding_type": info.get("openfhe_encoding_type"),
        "num_parts": info.get("num_parts"),
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


def signed_row(vals):
    return [int(v) for v in vals]


def compare_row(got, ref):
    if got is None:
        return {
            "exact": False,
            "max_abs_err": None,
            "diff": None,
        }

    g = np.array(got, dtype=np.int64)
    r = np.array(ref, dtype=np.int64)
    d = g - r

    return {
        "exact": bool(np.array_equal(g, r)),
        "max_abs_err": int(np.max(np.abs(d))),
        "diff": d.tolist(),
    }


def try_current_coeff_roundtrip(rt, row):
    out = {
        "input_row": row,
        "path": "encrypt_coeff_row_i64 -> optional compress -> decrypt_coeff_row_i64",
    }

    enc = safe_call(
        "encrypt_coeff_row_i64",
        lambda: rt.encrypt_coeff_row_i64(row),
    )

    out["encrypt"] = {
        "ok": enc["ok"],
        "error": enc.get("error"),
    }

    if not enc["ok"]:
        return out

    ct = enc["value"]

    out["inspect_after_encrypt"] = safe_call(
        "inspect_after_encrypt",
        lambda: inspect_compact(rt, ct),
    )

    dec_direct = safe_call(
        "decrypt_coeff_row_i64_direct",
        lambda: rt.decrypt_coeff_row_i64(ct, logical_length=len(row)),
    )

    out["decrypt_direct"] = {
        "ok": dec_direct["ok"],
        "error": dec_direct.get("error"),
        "value": signed_row(dec_direct["value"]) if dec_direct["ok"] else None,
    }
    out["decrypt_direct_compare"] = compare_row(
        out["decrypt_direct"]["value"],
        row,
    )

    comp = safe_call(
        "compress_coeff_ct_towers_left_1",
        lambda: rt.compress_coeff_ct(ct, towers_left=1),
    )

    out["compress"] = {
        "ok": comp["ok"],
        "error": comp.get("error"),
    }

    if comp["ok"]:
        ct_comp = comp["value"]

        out["inspect_after_compress"] = safe_call(
            "inspect_after_compress",
            lambda: inspect_compact(rt, ct_comp),
        )

        dec_comp = safe_call(
            "decrypt_coeff_row_i64_after_compress",
            lambda: rt.decrypt_coeff_row_i64(ct_comp, logical_length=len(row)),
        )

        out["decrypt_after_compress"] = {
            "ok": dec_comp["ok"],
            "error": dec_comp.get("error"),
            "value": signed_row(dec_comp["value"]) if dec_comp["ok"] else None,
        }
        out["decrypt_after_compress_compare"] = compare_row(
            out["decrypt_after_compress"]["value"],
            row,
        )

    return out


def try_standard_ckks_roundtrip(rt, row):
    out = {
        "input_row": row,
        "path": "standard CKKS slot encrypt/decrypt control path",
        "zh": "用于确认普通 CKKS slot 加解密是否正常；它不是 paper coefficient encoding。",
    }

    enc = safe_call(
        "standard_encrypt",
        lambda: rt.encrypt([float(x) for x in row]),
    )

    out["encrypt"] = {
        "ok": enc["ok"],
        "error": enc.get("error"),
    }

    if not enc["ok"]:
        return out

    ct = enc["value"]

    dec = safe_call(
        "standard_decrypt",
        lambda: rt.decrypt(ct, logical_length=len(row)),
    )

    out["decrypt"] = {
        "ok": dec["ok"],
        "error": dec.get("error"),
        "value": [float(x) for x in dec["value"]] if dec["ok"] else None,
    }

    if dec["ok"]:
        got = np.array(dec["value"], dtype=np.float64)
        ref = np.array(row, dtype=np.float64)
        diff = got - ref
        out["compare"] = {
            "max_abs_err": float(np.max(np.abs(diff))),
            "approx_exact_int_after_round": bool(np.array_equal(np.rint(got).astype(np.int64), np.array(row, dtype=np.int64))),
            "rounded_value": np.rint(got).astype(np.int64).tolist(),
        }

    return out


def main():
    n = 4
    rows = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-2, -1, 0, 1],
        [2, -2, 1, 0],
    ]

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
        "experiment": "wp4x7d_coeff_row_roundtrip_baseline",
        "purpose": (
            "Verify the semantic baseline for coefficient-row encryption/decryption. "
            "X7c showed that even input row-wise ciphertexts do not decrypt back exactly, "
            "so CCMM/C-MT debugging must pause until this path is fixed."
        ),
        "status": "coefficient_row_baseline_probe_no_ccmm",
        "n": n,
        "rows": rows,
        "paths": {
            "current_coeff_path": "encrypt_coeff_row_i64 -> compress(towers_left=1) -> decrypt_coeff_row_i64",
            "standard_ckks_control": "rt.encrypt -> rt.decrypt",
        },
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for row in rows:
            case = {
                "input_row": row,
                "current_coeff_path": try_current_coeff_roundtrip(rt, row),
                "standard_ckks_control": try_standard_ckks_roundtrip(rt, row),
            }
            report["cases"].append(case)

    report["checks"] = {
        "coeff_encrypt_all_ok（encrypt_coeff_row_i64 全部调用成功）": all(
            c["current_coeff_path"]["encrypt"]["ok"] for c in report["cases"]
        ),
        "coeff_direct_decrypt_all_ok（不 compress 直接 decrypt_coeff_row_i64 全部成功）": all(
            c["current_coeff_path"].get("decrypt_direct", {}).get("ok") is True
            for c in report["cases"]
        ),
        "coeff_direct_roundtrip_all_exact（不 compress 直接 roundtrip 全部精确）": all(
            c["current_coeff_path"].get("decrypt_direct_compare", {}).get("exact") is True
            for c in report["cases"]
        ),
        "coeff_compress_all_ok（compress(towers_left=1) 全部成功）": all(
            c["current_coeff_path"].get("compress", {}).get("ok") is True
            for c in report["cases"]
        ),
        "coeff_after_compress_decrypt_all_ok（compress 后 decrypt_coeff_row_i64 全部成功）": all(
            c["current_coeff_path"].get("decrypt_after_compress", {}).get("ok") is True
            for c in report["cases"]
        ),
        "coeff_after_compress_roundtrip_all_exact（compress 后 roundtrip 全部精确）": all(
            c["current_coeff_path"].get("decrypt_after_compress_compare", {}).get("exact") is True
            for c in report["cases"]
        ),
        "standard_ckks_control_all_ok（普通 CKKS slot 加解密对照全部成功）": all(
            c["standard_ckks_control"].get("encrypt", {}).get("ok") is True
            and c["standard_ckks_control"].get("decrypt", {}).get("ok") is True
            for c in report["cases"]
        ),
        "standard_ckks_control_rounds_to_input（普通 CKKS slot 解密四舍五入后等于输入）": all(
            c["standard_ckks_control"].get("compare", {}).get("approx_exact_int_after_round") is True
            for c in report["cases"]
        ),
    }

    report["summary"] = {
        "if_coeff_roundtrip_fails": (
            "Stop CCMM/C-MT work. Fix coefficient-row encode/decode first. "
            "The current C-MT and PP-MM diagnostics are invalid if inputs cannot decrypt to their rows."
        ),
        "if_standard_ckks_passes_but_coeff_fails": (
            "The HE runtime is fine, but the custom coefficient-row path is semantically wrong."
        ),
        "next_step": "WP4-X7e",
        "next_step_goal": (
            "Depending on X7d result, either implement a correct coefficient plaintext encode/decode path, "
            "or switch the toy semantic tests to a plaintext/open-key debug mode before returning to CCMM."
        ),
        "zh": "X7d 是分水岭：如果 coefficient row 自身不能 roundtrip，就必须先修加解码，不能继续调 CCMM。",
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x7d_coeff_row_roundtrip_baseline_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X7d coefficient-row roundtrip baseline")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
