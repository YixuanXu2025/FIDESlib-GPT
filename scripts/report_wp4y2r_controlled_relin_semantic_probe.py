import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.ccmm_oracle_debug import normalize_export, make_zero_1part


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def signed_to_mod(v, q):
    return int(v) % int(q)


def center_mod(x, q):
    x = int(x) % int(q)
    if x > int(q) // 2:
        x -= int(q)
    return int(x)


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
            "compress": {"ok": False, "error": comp.get("error")},
            "decrypt": {"ok": False, "error": "compress failed", "value": None},
        }

    dec = safe_call(
        "decrypt_coeff_row_i64_after_compress",
        lambda: rt.decrypt_coeff_row_i64(comp["value"], logical_length=logical_length),
    )

    return {
        "compress": {"ok": True, "error": None},
        "decrypt": {
            "ok": dec["ok"],
            "error": dec.get("error"),
            "value": dec.get("value") if dec["ok"] else None,
        },
    }


def negacyclic_mul_prefix(a, b, q, out_count):
    """
    Compute prefix of a*b mod (X^N+1), mod q.
    Optimized for sparse-ish secret key coefficients.
    """
    q = int(q)
    N = len(a)
    b_sparse = [(j, int(v) % q) for j, v in enumerate(b) if int(v) % q != 0]

    out = []
    for k in range(out_count):
        acc = 0
        for j, bj in b_sparse:
            if j <= k:
                i = k - j
                acc += int(a[i]) * bj
            else:
                i = k - j + N
                acc -= int(a[i]) * bj
        out.append(acc % q)
    return out


def sk_square_prefix_by_tower(sk_towers, moduli, out_count):
    out = []
    for t, q in enumerate(moduli):
        sk = [int(x) for x in sk_towers[t]]
        out.append(negacyclic_mul_prefix(sk, sk, int(q), out_count))
    return out


def make_constant_poly_towers(scalar, n, moduli):
    towers = []
    for q in moduli:
        coeffs = [0 for _ in range(n)]
        coeffs[0] = int(scalar) % int(q)
        towers.append(coeffs)
    return towers


def make_controlled_3part_relin_input(rt, template_ct, row, sk2_towers, moduli, c2_scalar, formula):
    """
    Construct a 3-component ciphertext-like handle.

    Formula options:
      plus_phase:
        c2 = a
        c1 = 0
        c0 = m - a*sk^2
        phase = c0 + c1*s + c2*s^2 = m

      minus_phase:
        c2 = a
        c1 = 0
        c0 = m + a*sk^2
        phase = c0 - c2*s^2 = m
        This is only a sign diagnostic.
    """
    n = len(row)

    c2_towers = make_constant_poly_towers(c2_scalar, n, moduli)
    c1_towers = [[0 for _ in range(n)] for _ in moduli]

    c0_towers = []
    for t, q in enumerate(moduli):
        q = int(q)
        c0_coeffs = []
        for i in range(n):
            m_i = signed_to_mod(row[i], q)
            ask2_i = (int(c2_scalar) * int(sk2_towers[t][i])) % q

            if formula == "plus_phase_c0_eq_m_minus_a_sk2":
                c0_i = (m_i - ask2_i) % q
            elif formula == "minus_phase_c0_eq_m_plus_a_sk2":
                c0_i = (m_i + ask2_i) % q
            else:
                raise ValueError(f"unknown formula: {formula}")

            c0_coeffs.append(c0_i)

        c0_towers.append(c0_coeffs)

    c0 = rt.import_1part_coeff_u64(template_ct, c0_towers)
    c1 = rt.import_1part_coeff_u64(template_ct, c1_towers)
    c2 = rt.import_1part_coeff_u64(template_ct, c2_towers)

    ct3 = rt.assemble_3part_from_1parts_coeff_ct(c0, c1, c2)

    return ct3, {
        "formula": formula,
        "c2_scalar": int(c2_scalar),
        "c0_tower0_coeffs": c0_towers[0],
        "c1_tower0_coeffs": c1_towers[0],
        "c2_tower0_coeffs": c2_towers[0],
        "sk2_tower0_prefix": sk2_towers[0][:n],
    }


def run_case(rt, template_ct, row_name, row, sk2_towers, moduli, c2_scalar, formula):
    built = safe_call(
        "make_controlled_3part_relin_input",
        lambda: make_controlled_3part_relin_input(
            rt=rt,
            template_ct=template_ct,
            row=row,
            sk2_towers=sk2_towers,
            moduli=moduli,
            c2_scalar=c2_scalar,
            formula=formula,
        ),
    )

    out = {
        "row_name": row_name,
        "input_row": row,
        "c2_scalar": int(c2_scalar),
        "formula": formula,
        "build": {"ok": built["ok"], "error": built.get("error")},
    }

    if not built["ok"]:
        return out

    ct3, build_trace = built["value"]
    out["build_trace"] = build_trace

    relin = safe_call("relinearize_coeff_ct", lambda: rt.relinearize_coeff_ct(ct3))
    out["relinearize"] = {"ok": relin["ok"], "error": relin.get("error")}

    if not relin["ok"]:
        return out

    dec = decrypt_after_compress(rt, relin["value"], logical_length=len(row))
    value = dec["decrypt"].get("value") if dec["decrypt"]["ok"] else None

    out["decrypt_after_relin_compress"] = dec
    out["compare_to_input"] = compare_row(value, row)

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
        "plus_phase_c0_eq_m_minus_a_sk2",
        "minus_phase_c0_eq_m_plus_a_sk2",
    ]

    c2_scalars = [0, 1, 3, 7]

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
        "experiment": "wp4y2r_controlled_relin_semantic_probe",
        "purpose": (
            "Validate semantic correctness of OpenFHE/FIDESlib relinearization on controlled 3-part coefficient ciphertexts. "
            "This isolates Algorithm 3 line5 KS_{s^2->s}."
        ),
        "status": "debug_controlled_relin_probe_not_secure",
        "warning": {
            "value": "Uses secret key to construct c0 = m - c2*s^2. Not secure.",
            "zh": "该脚本用 secret key 构造 3-part 输入，只用于验证 relinearization 语义。",
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
        sk_towers = rt.export_secret_key_coeff_u64(0)
        sk2_towers = sk_square_prefix_by_tower(sk_towers, moduli, n)

        report["context_summary"] = {
            "num_moduli": len(moduli),
            "moduli_u64": moduli,
            "template_ring_dim": template_export["ring_dim"],
            "template_num_towers": template_export["num_towers"],
            "sk_num_towers": len(sk_towers),
            "sk_coeff_count": len(sk_towers[0]) if sk_towers else 0,
            "sk_tower0_prefix": sk_towers[0][:n] if sk_towers else [],
            "sk2_tower0_prefix": sk2_towers[0] if sk2_towers else [],
        }

        for formula in formulas:
            for c2_scalar in c2_scalars:
                group = {
                    "formula": formula,
                    "c2_scalar": int(c2_scalar),
                    "rows": [],
                }

                for row_name, row in rows.items():
                    group["rows"].append(
                        run_case(
                            rt=rt,
                            template_ct=template_ct,
                            row_name=row_name,
                            row=row,
                            sk2_towers=sk2_towers,
                            moduli=moduli,
                            c2_scalar=c2_scalar,
                            formula=formula,
                        )
                    )

                group["summary"] = {
                    "all_build_ok": all(r.get("build", {}).get("ok") is True for r in group["rows"]),
                    "all_relinearize_ok": all(r.get("relinearize", {}).get("ok") is True for r in group["rows"]),
                    "all_decrypt_ok": all(
                        r.get("decrypt_after_relin_compress", {}).get("decrypt", {}).get("ok") is True
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
            "c2_scalar": c["c2_scalar"],
            "summary": c["summary"],
        }
        for c in report["cases"]
        if c["summary"]["all_exact"]
    ]

    nonzero_c2_success = [
        g for g in successful_groups
        if g["c2_scalar"] != 0
    ]

    ranked_groups = sorted(
        [
            {
                "formula": c["formula"],
                "c2_scalar": c["c2_scalar"],
                "summary": c["summary"],
            }
            for c in report["cases"]
        ],
        key=lambda x: (
            -x["summary"]["exact_count"],
            x["summary"]["max_abs_err_over_rows"] if x["summary"]["max_abs_err_over_rows"] is not None else 10**18,
            x["formula"],
            x["c2_scalar"],
        ),
    )

    report["checks"] = {
        "any_group_all_exact（是否存在任一公式/c2 组合全部精确）": len(successful_groups) > 0,
        "any_nonzero_c2_group_all_exact（是否存在非零 c2 的组合全部精确）": len(nonzero_c2_success) > 0,
        "successful_groups（全部精确的组合）": successful_groups,
        "ranked_groups（按 exact_count 排序）": ranked_groups,
    }

    if nonzero_c2_success:
        decision = {
            "value": "relinearize_semantics_work_for_controlled_3part",
            "recommended_formula": nonzero_c2_success[0]["formula"],
            "recommended_c2_scalar": nonzero_c2_success[0]["c2_scalar"],
            "next_step": "WP4-Y2s",
            "next_goal": "Return to Algorithm 3 symbolic/formula debugging; relinearization is not the blocker.",
            "zh": "非零 c2 的 controlled 3-part relinearization 语义正确，line5 blocker 排除。",
        }
    elif successful_groups:
        decision = {
            "value": "only_c2_zero_works",
            "next_step": "WP4-Y2s",
            "next_goal": "Relinearization does not preserve nonzero c2 semantics; inspect relinearize_coeff_ct / key switching.",
            "zh": "只有 c2=0 通过，说明 relinearize 对 raw-imported c2 语义不正确。",
        }
    else:
        decision = {
            "value": "controlled_relinearize_probe_failed",
            "next_step": "WP4-Y2s",
            "next_goal": "Inspect 3-part construction, sk^2 prefix, or relinearization key semantics.",
            "zh": "controlled 3-part relinearization probe 全失败，需要检查构造或 key switching。",
        }

    report["global_decision"] = decision

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4y2r_controlled_relin_semantic_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Y2r controlled relinearization semantic probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
