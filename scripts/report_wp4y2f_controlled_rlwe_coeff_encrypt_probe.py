import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.ccmm_oracle_debug import normalize_export


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def signed_to_mod(v, q):
    return int(v) % int(q)


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


def decrypt_after_compress(rt, ct, logical_length):
    comp = safe_call(
        "compress_towers_left_1",
        lambda: rt.compress_coeff_ct(ct, towers_left=1),
    )

    if not comp["ok"]:
        return {
            "compress": {
                "ok": False,
                "error": comp.get("error"),
            },
            "decrypt": {
                "ok": False,
                "error": "compress failed",
                "value": None,
            },
        }

    dec = safe_call(
        "decrypt_coeff_row_i64_after_compress",
        lambda: rt.decrypt_coeff_row_i64(comp["value"], logical_length=logical_length),
    )

    return {
        "compress": {
            "ok": True,
            "error": None,
        },
        "decrypt": {
            "ok": dec["ok"],
            "error": dec.get("error"),
            "value": dec.get("value") if dec["ok"] else None,
        },
    }


def make_constant_c1_towers(a_scalar, n, moduli):
    towers = []
    for q in moduli:
        q = int(q)
        coeffs = [0 for _ in range(n)]
        coeffs[0] = int(a_scalar) % q
        towers.append(coeffs)
    return towers


def make_controlled_rlwe_coeff_row_ct(rt, template_ct, row, sk_towers, moduli, a_scalar, formula):
    """
    Build a controlled debug RLWE ciphertext.

    OpenFHE/FIDESlib convention:
      c0 = component 0
      c1 = component 1

    Expected decrypt phase:
      c0 + c1 * sk

    Variant formula:
      plus_phase:
        c1 = a
        c0 = m - a*sk
        expected c0 + c1*sk = m

      minus_phase:
        c1 = a
        c0 = m + a*sk
        expected c0 - c1*sk = m
        This is only a sign-convention diagnostic.
    """
    n = len(row)

    c1_towers = make_constant_c1_towers(a_scalar, n, moduli)

    c0_towers = []
    for t, q in enumerate(moduli):
        q = int(q)
        sk_coeffs = sk_towers[t][:n]

        c0_coeffs = []
        for i in range(n):
            m_i = signed_to_mod(row[i], q)
            ask_i = (int(a_scalar) * int(sk_coeffs[i])) % q

            if formula == "plus_phase_c0_eq_m_minus_a_sk":
                c0_i = (m_i - ask_i) % q
            elif formula == "minus_phase_c0_eq_m_plus_a_sk":
                c0_i = (m_i + ask_i) % q
            else:
                raise ValueError(f"unknown formula: {formula}")

            c0_coeffs.append(c0_i)

        c0_towers.append(c0_coeffs)

    c0 = rt.import_1part_coeff_u64(template_ct, c0_towers)
    c1 = rt.import_1part_coeff_u64(template_ct, c1_towers)

    return rt.assemble_2part_from_1parts_coeff_ct(c0, c1), {
        "formula": formula,
        "a_scalar": int(a_scalar),
        "c0_tower0_coeffs": c0_towers[0],
        "c1_tower0_coeffs": c1_towers[0],
        "sk_tower0_coeffs": sk_towers[0][:n],
    }


def run_case(rt, template_ct, sk_towers, moduli, row_name, row, a_scalar, formula):
    built = safe_call(
        "make_controlled_rlwe_coeff_row_ct",
        lambda: make_controlled_rlwe_coeff_row_ct(
            rt=rt,
            template_ct=template_ct,
            row=row,
            sk_towers=sk_towers,
            moduli=moduli,
            a_scalar=a_scalar,
            formula=formula,
        ),
    )

    out = {
        "row_name": row_name,
        "input_row": row,
        "a_scalar": int(a_scalar),
        "formula": formula,
        "build": {
            "ok": built["ok"],
            "error": built.get("error"),
        },
    }

    if not built["ok"]:
        return out

    ct, build_trace = built["value"]

    dec = decrypt_after_compress(rt, ct, logical_length=len(row))
    value = dec["decrypt"].get("value") if dec["decrypt"]["ok"] else None

    out.update({
        "build_trace": build_trace,
        "decrypt_after_compress": dec,
        "compare_to_input": compare_row(value, row),
    })

    return out


def main():
    n = 8

    rows = {
        "zero8": [0, 0, 0, 0, 0, 0, 0, 0],
        "basis0": [1, 0, 0, 0, 0, 0, 0, 0],
        "basis1": [0, 1, 0, 0, 0, 0, 0, 0],
        "basis3": [0, 0, 0, 1, 0, 0, 0, 0],
        "small_mixed": [-3, -2, -1, 0, 1, 2, 3, 4],
        "alternating": [1, -1, 1, -1, 1, -1, 1, -1],
        "large_small_mix": [7, 0, -5, 2, -1, 0, 3, -4],
    }

    formulas = [
        "plus_phase_c0_eq_m_minus_a_sk",
        "minus_phase_c0_eq_m_plus_a_sk",
    ]

    a_scalars = [
        0,
        1,
        3,
        7,
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
        "experiment": "wp4y2f_controlled_rlwe_coeff_encrypt_probe",
        "purpose": (
            "Bypass high-level encrypt_coeff_row_i64 and construct controlled RLWE coefficient-row ciphertexts "
            "using exported secret-key coefficients. This validates whether c0/c1 raw-imported coefficient ciphertexts "
            "can satisfy c0 + c1*sk = m under decrypt_coeff_row_i64."
        ),
        "status": "debug_controlled_sk_rlwe_probe_not_secure",
        "warning": {
            "value": "This uses the secret key to construct ciphertexts and is not secure encryption.",
            "zh": "该脚本使用 secret key 构造 ciphertext，只用于验证语义，不是最终安全加密。",
        },
        "n": n,
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        template_ct = rt.encrypt_coeff_row_i64([0 for _ in range(n)])

        template_export = normalize_export(
            rt.export_component_coeff_matrix_u64([template_ct], part=0, coeff_count=1)
        )

        moduli = [int(q) for q in template_export["moduli_u64"]]
        sk_towers = rt.export_secret_key_coeff_u64(n)

        report["context_summary"] = {
            "num_moduli": len(moduli),
            "moduli_u64": moduli,
            "template_ring_dim": template_export["ring_dim"],
            "template_num_towers": template_export["num_towers"],
            "sk_num_towers": len(sk_towers),
            "sk_coeff_count": len(sk_towers[0]) if sk_towers else 0,
            "sk_tower0_coeffs": sk_towers[0] if sk_towers else [],
        }

        for formula in formulas:
            for a_scalar in a_scalars:
                group = {
                    "formula": formula,
                    "a_scalar": int(a_scalar),
                    "rows": [],
                }

                for row_name, row in rows.items():
                    group["rows"].append(
                        run_case(
                            rt=rt,
                            template_ct=template_ct,
                            sk_towers=sk_towers,
                            moduli=moduli,
                            row_name=row_name,
                            row=row,
                            a_scalar=a_scalar,
                            formula=formula,
                        )
                    )

                group["summary"] = {
                    "all_build_ok": all(r.get("build", {}).get("ok") is True for r in group["rows"]),
                    "all_decrypt_ok": all(
                        r.get("decrypt_after_compress", {}).get("decrypt", {}).get("ok") is True
                        for r in group["rows"]
                    ),
                    "all_exact": all(
                        r.get("compare_to_input", {}).get("exact") is True
                        for r in group["rows"]
                    ),
                    "exact_count": sum(
                        1 for r in group["rows"]
                        if r.get("compare_to_input", {}).get("exact") is True
                    ),
                    "max_abs_err_over_rows": max(
                        [
                            r.get("compare_to_input", {}).get("max_abs_err", 10**18)
                            for r in group["rows"]
                            if r.get("compare_to_input") is not None
                        ] or [None]
                    ),
                }

                report["cases"].append(group)

    successful_groups = [
        {
            "formula": c["formula"],
            "a_scalar": c["a_scalar"],
            "summary": c["summary"],
        }
        for c in report["cases"]
        if c["summary"]["all_exact"]
    ]

    nonzero_a_successful_groups = [
        g for g in successful_groups
        if g["a_scalar"] != 0
    ]

    ranked_groups = sorted(
        [
            {
                "formula": c["formula"],
                "a_scalar": c["a_scalar"],
                "summary": c["summary"],
            }
            for c in report["cases"]
        ],
        key=lambda x: (
            -x["summary"]["exact_count"],
            x["summary"]["max_abs_err_over_rows"] if x["summary"]["max_abs_err_over_rows"] is not None else 10**18,
            x["formula"],
            x["a_scalar"],
        ),
    )

    report["checks"] = {
        "any_group_all_exact（是否存在任一公式/a 组合全部精确）": len(successful_groups) > 0,
        "any_nonzero_a_group_all_exact（是否存在非零 c1 的组合全部精确）": len(nonzero_a_successful_groups) > 0,
        "successful_groups（全部精确的组合）": successful_groups,
        "ranked_groups（按 exact_count 排序）": ranked_groups,
    }

    if nonzero_a_successful_groups:
        report["global_decision"] = {
            "value": "controlled_rlwe_coeff_row_encryption_works",
            "recommended_formula": nonzero_a_successful_groups[0]["formula"],
            "recommended_a_scalar": nonzero_a_successful_groups[0]["a_scalar"],
            "next_step": "WP4-Y2g",
            "next_goal": "Wrap controlled RLWE coefficient-row construction and re-run rowwise bundle/C-MT oracle with non-transparent c1.",
            "zh": "controlled RLWE coefficient-row 构造可用；下一步封装并替换 transparent 输入。",
        }
    elif successful_groups:
        report["global_decision"] = {
            "value": "only_transparent_a0_works",
            "recommended_formula": successful_groups[0]["formula"],
            "recommended_a_scalar": successful_groups[0]["a_scalar"],
            "next_step": "WP4-Y2g",
            "next_goal": "Investigate c1/sk multiplication convention; only a=0 transparent path works.",
            "zh": "只有 c1=0 的 transparent path 通过，说明 sk 乘法/sign/format 约定仍未对齐。",
        }
    else:
        report["global_decision"] = {
            "value": "controlled_rlwe_probe_failed",
            "recommended_formula": None,
            "recommended_a_scalar": None,
            "next_step": "WP4-Y2g",
            "next_goal": "Diagnose secret-key coefficient export, c0/c1 sign convention, or raw component import format.",
            "zh": "controlled RLWE 构造仍失败，需要检查 sk 导出、相位符号或 raw import format。",
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4y2f_controlled_rlwe_coeff_encrypt_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Y2f controlled RLWE coefficient-row encryption probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
