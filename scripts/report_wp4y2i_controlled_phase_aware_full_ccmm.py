import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.ccmm_oracle_debug import (
    TempRowwiseBundle,
    normalize_export,
    raw_rns_ppmm,
    import_ppmm_rows,
    make_zero_1part_rows,
    make_temp_pair_bundle,
    extract_1part_rows,
    raw_add_2part_rows,
)


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def signed_to_mod(v, q):
    return int(v) % int(q)


def compare_matrix(decoded, ref):
    if decoded is None or any(row is None for row in decoded):
        return {
            "all_rows_decrypted": False,
            "exact": False,
            "max_abs_err": None,
            "l1_err": None,
            "diff": None,
        }

    G = np.array(decoded, dtype=np.int64)
    R = np.array(ref, dtype=np.int64)
    D = G - R

    return {
        "all_rows_decrypted": True,
        "exact": bool(np.array_equal(G, R)),
        "max_abs_err": int(np.max(np.abs(D))),
        "l1_err": int(np.sum(np.abs(D))),
        "diff": D.tolist(),
    }


def decrypt_rows_after_compress(rt, rows, logical_length):
    compressed = []
    compress_reports = []

    for i, ct in enumerate(rows):
        comp = safe_call(
            f"compress_row_{i}",
            lambda ct=ct: rt.compress_coeff_ct(ct, towers_left=1),
        )
        compress_reports.append({
            "row_index": i,
            "ok": comp["ok"],
            "error": comp.get("error"),
        })
        compressed.append(comp["value"] if comp["ok"] else None)

    decoded = []
    decrypt_reports = []

    for i, ct in enumerate(compressed):
        if ct is None:
            decoded.append(None)
            decrypt_reports.append({
                "row_index": i,
                "ok": False,
                "error": "compress failed",
            })
            continue

        dec = safe_call(
            f"decrypt_row_{i}",
            lambda ct=ct: rt.decrypt_coeff_row_i64(ct, logical_length=logical_length),
        )
        decrypt_reports.append({
            "row_index": i,
            "ok": dec["ok"],
            "error": dec.get("error"),
        })
        decoded.append(list(map(int, dec["value"])) if dec["ok"] else None)

    return {
        "compress_reports": compress_reports,
        "decrypt_reports": decrypt_reports,
        "decoded": decoded,
    }


def inspect_rows(rt, rows, coeff_sample=4):
    info = rt.inspect_coeff_row_component_matrix(rows, coeff_sample=coeff_sample)
    return {
        "num_rows": info.get("num_rows"),
        "consistent_shape": info.get("consistent_shape"),
        "expected_parts": info.get("expected_parts"),
        "expected_towers": info.get("expected_towers"),
        "expected_ring_dim": info.get("expected_ring_dim"),
        "rows": [
            {
                "row_index": r.get("row_index"),
                "encoding_type": r.get("encoding_type"),
                "num_parts": r.get("num_parts"),
                "level": r.get("level"),
                "slots": r.get("slots"),
            }
            for r in info.get("rows", [])
        ],
    }


def rows_are_npart(summary, n, parts):
    return (
        summary.get("num_rows") == n
        and summary.get("expected_parts") == parts
        and all(r.get("num_parts") == parts for r in summary.get("rows", []))
    )


def make_constant_c1_towers(a_scalar, n, moduli):
    towers = []
    for q in moduli:
        q = int(q)
        coeffs = [0 for _ in range(n)]
        coeffs[0] = int(a_scalar) % q
        towers.append(coeffs)
    return towers


def make_controlled_rlwe_coeff_row_ct(rt, template_ct, row, sk_towers, moduli, a_scalar):
    """
    Controlled debug RLWE coefficient row.

    Verified by Y2f:
      phase = c0 + c1*sk
      c1 = a
      c0 = m - a*sk
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
            c0_i = (m_i - ask_i) % q
            c0_coeffs.append(c0_i)

        c0_towers.append(c0_coeffs)

    c0 = rt.import_1part_coeff_u64(template_ct, c0_towers)
    c1 = rt.import_1part_coeff_u64(template_ct, c1_towers)

    return rt.assemble_2part_from_1parts_coeff_ct(c0, c1)


def make_controlled_rowwise_bundle(rt, matrix, a_scalar=1, ring_dim=1 << 14, batch_size=8):
    n = len(matrix)

    template_ct = rt.encrypt_coeff_row_i64([0 for _ in range(n)])
    template_export = normalize_export(
        rt.export_component_coeff_matrix_u64([template_ct], part=0, coeff_count=1)
    )

    moduli = [int(q) for q in template_export["moduli_u64"]]
    sk_towers = rt.export_secret_key_coeff_u64(n)

    rows = []
    for row in matrix:
        rows.append(
            make_controlled_rlwe_coeff_row_ct(
                rt=rt,
                template_ct=template_ct,
                row=[int(v) for v in row],
                sk_towers=sk_towers,
                moduli=moduli,
                a_scalar=a_scalar,
            )
        )

    bundle = TempRowwiseBundle(
        rows=rows,
        shape=(n, n),
        label=f"controlled_rlwe_rowwise_a{a_scalar}",
        ring_dim=ring_dim,
        batch_size=batch_size,
    )

    info = {
        "a_scalar": int(a_scalar),
        "template_ring_dim": template_export["ring_dim"],
        "template_num_towers": template_export["num_towers"],
        "moduli_u64": moduli,
        "sk_num_towers": len(sk_towers),
        "sk_coeff_count": len(sk_towers[0]) if sk_towers else 0,
        "sk_tower0_coeffs": sk_towers[0] if sk_towers else [],
    }

    return bundle, info


def phase_aware_logical_n_cmt_oracle(rt, bundle, n, a_scalar, label):
    """
    Debug-only phase-aware C-MT oracle.

    It decrypts phase, transposes plaintext, and rebuilds controlled RLWE rows.
    This is not secure/homomorphic.
    """
    dec = decrypt_rows_after_compress(rt, bundle.rows, logical_length=n)

    if any(row is None for row in dec["decoded"]):
        raise RuntimeError("phase_aware_logical_n_cmt_oracle(): input decrypt failed")

    M = np.array(dec["decoded"], dtype=np.int64)
    M_T = M.T.copy()

    template_ct = bundle.rows[0]
    template_export = normalize_export(
        rt.export_component_coeff_matrix_u64([template_ct], part=0, coeff_count=1)
    )

    moduli = [int(q) for q in template_export["moduli_u64"]]
    sk_towers = rt.export_secret_key_coeff_u64(n)

    out_rows = []
    for row in M_T.tolist():
        out_rows.append(
            make_controlled_rlwe_coeff_row_ct(
                rt=rt,
                template_ct=template_ct,
                row=[int(v) for v in row],
                sk_towers=sk_towers,
                moduli=moduli,
                a_scalar=a_scalar,
            )
        )

    out_bundle = TempRowwiseBundle(
        rows=out_rows,
        shape=(n, n),
        label=label,
        ring_dim=bundle.ring_dim,
        batch_size=bundle.batch_size,
    )

    return out_bundle, {
        "input_plaintext_matrix": M.tolist(),
        "output_plaintext_transpose": M_T.tolist(),
        "a_scalar": int(a_scalar),
        "warning": "debug-only decrypt-transpose-reencrypt oracle",
    }


def export_component_sources(rt, left_bundle, right_bundle, n):
    return {
        "A_L": normalize_export(rt.export_component_coeff_matrix_u64(left_bundle.rows, part=1, coeff_count=n)),
        "B_L": normalize_export(rt.export_component_coeff_matrix_u64(left_bundle.rows, part=0, coeff_count=n)),
        "A_R": normalize_export(rt.export_component_coeff_matrix_u64(right_bundle.rows, part=1, coeff_count=n)),
        "B_R": normalize_export(rt.export_component_coeff_matrix_u64(right_bundle.rows, part=0, coeff_count=n)),
    }


def run_phase_aware_full_pipeline(rt, U, V, a_scalar):
    n = U.shape[0]

    U_bundle, U_info = make_controlled_rowwise_bundle(rt, U.tolist(), a_scalar=a_scalar)
    V_bundle, V_info = make_controlled_rowwise_bundle(rt, V.tolist(), a_scalar=a_scalar)

    U_dec = decrypt_rows_after_compress(rt, U_bundle.rows, logical_length=n)
    V_dec = decrypt_rows_after_compress(rt, V_bundle.rows, logical_length=n)

    # Algorithm 3 line 1: phase-aware C-MT(U)
    U_cmt_bundle, U_cmt_trace = phase_aware_logical_n_cmt_oracle(
        rt,
        U_bundle,
        n=n,
        a_scalar=a_scalar,
        label=f"phase_aware_CMT_U_a{a_scalar}",
    )

    # Algorithm 3 line 2: raw RNS PP-MM on components.
    comps = export_component_sources(rt, U_cmt_bundle, V_bundle, n)

    M00 = raw_rns_ppmm(comps["A_L"], comps["A_R"], n)
    M01 = raw_rns_ppmm(comps["A_L"], comps["B_R"], n)
    M10 = raw_rns_ppmm(comps["B_L"], comps["A_R"], n)
    M11 = raw_rns_ppmm(comps["B_L"], comps["B_R"], n)

    M00_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M00)
    M01_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M01)
    M10_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M10)
    M11_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M11)

    zero_rows = make_zero_1part_rows(rt, U_cmt_bundle.rows, num_towers=M00["num_towers"])

    # Algorithm 3 line 3: phase-aware C-MT((M01,0))
    T01_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M01_rows,
        B_rows=zero_rows,
        n=n,
        label="T01=(M01,0)",
        template_bundle=U_bundle,
    )
    T01_cmt_bundle, T01_cmt_trace = phase_aware_logical_n_cmt_oracle(
        rt,
        T01_bundle,
        n=n,
        a_scalar=a_scalar,
        label=f"phase_aware_CMT_T01_a{a_scalar}",
    )

    # Algorithm 3 line 4: phase-aware C-MT((M00,0))
    T00_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M00_rows,
        B_rows=zero_rows,
        n=n,
        label="T00=(M00,0)",
        template_bundle=U_bundle,
    )
    T00_cmt_bundle, T00_cmt_trace = phase_aware_logical_n_cmt_oracle(
        rt,
        T00_bundle,
        n=n,
        a_scalar=a_scalar,
        label=f"phase_aware_CMT_T00_a{a_scalar}",
    )

    # Extract paper pair (A_hat,B_hat), OpenFHE order c0=B_hat,c1=A_hat.
    Bhat_rows, Bhat_export = extract_1part_rows(rt, T00_cmt_bundle.rows, part=0, coeff_count=0)
    Ahat_rows, Ahat_export = extract_1part_rows(rt, T00_cmt_bundle.rows, part=1, coeff_count=0)

    # Algorithm 3 line 5.
    line5_rows = []
    line5_ks_inputs = []
    line5_ks_outputs = []
    line5_bhat_terms = []

    for i in range(n):
        ks_input = rt.assemble_3part_from_1parts_coeff_ct(
            zero_rows[i],
            zero_rows[i],
            Ahat_rows[i],
        )
        line5_ks_inputs.append(ks_input)

        ks_output = rt.relinearize_coeff_ct(ks_input)
        line5_ks_outputs.append(ks_output)

        # paper pair (B_hat,0) -> OpenFHE c0=0,c1=B_hat.
        bhat_term = rt.assemble_2part_from_1parts_coeff_ct(
            zero_rows[i],
            Bhat_rows[i],
        )
        line5_bhat_terms.append(bhat_term)

        line5_rows.append(rt.add_coeff_ct(ks_output, bhat_term))

    # Algorithm 3 line 6: (M10,M11), OpenFHE c0=M11,c1=M10.
    M10_M11_pair_rows = [
        rt.assemble_2part_from_1parts_coeff_ct(M11_rows[i], M10_rows[i])
        for i in range(n)
    ]

    # Use raw RNS add for line6 debug path, per X7k.
    tmp = raw_add_2part_rows(rt, line5_rows, T01_cmt_bundle.rows, n)
    final_rows = raw_add_2part_rows(rt, tmp, M10_M11_pair_rows, n)

    final = decrypt_rows_after_compress(rt, final_rows, logical_length=n)

    trace = {
        "U_info": U_info,
        "V_info": V_info,
        "U_decrypt_exact": compare_matrix(U_dec["decoded"], U),
        "V_decrypt_exact": compare_matrix(V_dec["decoded"], V),
        "U_cmt_trace": U_cmt_trace,
        "T01_cmt_trace": T01_cmt_trace,
        "T00_cmt_trace": T00_cmt_trace,
        "Ahat_export": Ahat_export,
        "Bhat_export": Bhat_export,
        "line5_ks_inputs_summary": inspect_rows(rt, line5_ks_inputs, coeff_sample=4),
        "line5_ks_outputs_summary": inspect_rows(rt, line5_ks_outputs, coeff_sample=4),
        "line5_rows_summary": inspect_rows(rt, line5_rows, coeff_sample=4),
        "T01_cmt_rows_summary": inspect_rows(rt, T01_cmt_bundle.rows, coeff_sample=4),
        "M10_M11_pair_rows_summary": inspect_rows(rt, M10_M11_pair_rows, coeff_sample=4),
        "final_rows_summary": inspect_rows(rt, final_rows, coeff_sample=4),
    }

    return final, trace


def one_hot(n, i, j, value=1):
    M = np.zeros((n, n), dtype=np.int64)
    M[i, j] = value
    return M


def summarize_against_references(got, refs):
    return {
        name: compare_matrix(got, ref)
        for name, ref in refs.items()
    }


def main():
    n = 4
    rng = np.random.default_rng(209300)

    cases = [
        {
            "name": "basis_E00_times_E00",
            "U": one_hot(n, 0, 0),
            "V": one_hot(n, 0, 0),
        },
        {
            "name": "basis_E01_times_E10",
            "U": one_hot(n, 0, 1),
            "V": one_hot(n, 1, 0),
        },
        {
            "name": "basis_E01_times_E12",
            "U": one_hot(n, 0, 1),
            "V": one_hot(n, 1, 2),
        },
        {
            "name": "basis_E23_times_E31",
            "U": one_hot(n, 2, 3),
            "V": one_hot(n, 3, 1),
        },
        {
            "name": "identity_times_random_small",
            "U": np.eye(n, dtype=np.int64),
            "V": rng.integers(-2, 3, size=(n, n)).astype(np.int64),
        },
        {
            "name": "random_small_times_identity",
            "U": rng.integers(-2, 3, size=(n, n)).astype(np.int64),
            "V": np.eye(n, dtype=np.int64),
        },
    ]

    a_scalars = [1, 3, 7]

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
        "experiment": "wp4y2i_controlled_phase_aware_full_ccmm",
        "purpose": (
            "Run full Algorithm 3 debug/oracle pipeline using controlled RLWE inputs and phase-aware logical-n C-MT oracle. "
            "Line6 uses raw RNS add per X7k."
        ),
        "status": "debug_full_ccmm_oracle_not_secure",
        "n": n,
        "warning": {
            "value": "This uses secret-key phase-aware C-MT oracle and controlled RLWE construction. It is not secure/homomorphic.",
            "zh": "该脚本用于语义定位，不是最终安全 CCMM。",
        },
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for a_scalar in a_scalars:
            for case in cases:
                U = case["U"]
                V = case["V"]

                final, trace = run_phase_aware_full_pipeline(rt, U, V, a_scalar=a_scalar)

                refs = {
                    "U_matmul_V": U @ V,
                    "U_T_matmul_V": U.T @ V,
                    "U_matmul_V_T": U @ V.T,
                    "V_matmul_U": V @ U,
                    "V_T_matmul_U": V.T @ U,
                    "U_T_matmul_V_T": U.T @ V.T,
                }

                comparisons = summarize_against_references(final["decoded"], refs)

                report["cases"].append({
                    "name": case["name"],
                    "a_scalar": int(a_scalar),
                    "input_U": U.tolist(),
                    "input_V": V.tolist(),
                    "references": {k: v.tolist() for k, v in refs.items()},
                    "final": final,
                    "trace": trace,
                    "comparisons": comparisons,
                })

    ref_names = [
        "U_matmul_V",
        "U_T_matmul_V",
        "U_matmul_V_T",
        "V_matmul_U",
        "V_T_matmul_U",
        "U_T_matmul_V_T",
    ]

    match_counts = {
        ref: sum(
            1 for c in report["cases"]
            if c["comparisons"][ref]["exact"]
        )
        for ref in ref_names
    }

    by_a_scalar = {}
    for a_scalar in a_scalars:
        cs = [c for c in report["cases"] if c["a_scalar"] == a_scalar]
        by_a_scalar[str(a_scalar)] = {
            ref: sum(1 for c in cs if c["comparisons"][ref]["exact"])
            for ref in ref_names
        }

    total_cases = len(report["cases"])

    report["checks"] = {
        "all_cases_decrypted（所有 full oracle case 都成功解密）": all(
            c["comparisons"]["U_matmul_V"]["all_rows_decrypted"]
            for c in report["cases"]
        ),
        "match_counts_by_reference（各参考矩阵匹配 case 数）": match_counts,
        "match_counts_by_reference_and_a_scalar（按 a_scalar 分组的匹配数）": by_a_scalar,
        "all_cases_match_U_matmul_V（是否全部等于 U@V）": match_counts["U_matmul_V"] == total_cases,
        "all_cases_match_U_T_matmul_V（是否全部等于 U.T@V）": match_counts["U_T_matmul_V"] == total_cases,
        "all_line5_ks_inputs_are_3part（line5 KS inputs 均为三分量）": all(
            rows_are_npart(c["trace"]["line5_ks_inputs_summary"], n, 3)
            for c in report["cases"]
        ),
        "all_final_rows_are_2part（最终输出 rows 均为二分量）": all(
            rows_are_npart(c["trace"]["final_rows_summary"], n, 2)
            for c in report["cases"]
        ),
    }

    best_ref = max(match_counts.items(), key=lambda kv: kv[1])

    if report["checks"]["all_cases_match_U_matmul_V（是否全部等于 U@V）"]:
        decision = {
            "value": "full_phase_aware_oracle_matches_U_matmul_V",
            "next_step": "WP4-Y3",
            "next_goal": "Promote controlled/phase-aware oracle result into a real logical-n C-MT design plan.",
            "zh": "完整 oracle pipeline 已匹配 U@V；下一步设计真实 logical-n C-MT。",
        }
    elif report["checks"]["all_cases_match_U_T_matmul_V（是否全部等于 U.T@V）"]:
        decision = {
            "value": "full_phase_aware_oracle_matches_U_T_matmul_V",
            "next_step": "WP4-Y2j",
            "next_goal": "Fix orientation contract or PP-MM operand orientation; current pipeline computes U.T@V.",
            "zh": "完整 oracle pipeline 稳定计算 U.T@V，需要修正 orientation/operand contract。",
        }
    else:
        decision = {
            "value": "full_phase_aware_oracle_has_remaining_orientation_or_formula_gap",
            "best_reference": {
                "name": best_ref[0],
                "match_count": best_ref[1],
                "total_cases": total_cases,
            },
            "next_step": "WP4-Y2j",
            "next_goal": "Run orientation/formula sweep under phase-aware C-MT oracle.",
            "zh": "phase-aware full pipeline 仍有方向或公式差异，下一步做定向扫描。",
        }

    report["global_decision"] = decision

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4y2i_controlled_phase_aware_full_ccmm_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Y2i controlled phase-aware full CCMM oracle")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
