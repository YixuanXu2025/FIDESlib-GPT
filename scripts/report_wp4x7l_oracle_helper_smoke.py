import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.ccmm_oracle_debug import (
    make_transparent_rowwise_bundle,
    logical_n_cmt_oracle,
    raw_rns_ppmm,
    import_ppmm_rows,
    make_zero_1part_rows,
    make_temp_pair_bundle,
    extract_1part_rows,
    raw_add_2part_rows,
    normalize_export,
)


def one_hot(n, i, j, value=1):
    M = np.zeros((n, n), dtype=np.int64)
    M[i, j] = value
    return M


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


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


def compare_matrix(got, ref):
    if any(r is None for r in got):
        return {
            "all_rows_decrypted": False,
            "exact": False,
            "max_abs_err": None,
            "l1_err": None,
            "diff": None,
        }

    G = np.array(got, dtype=np.int64)
    R = np.array(ref, dtype=np.int64)
    D = G - R

    return {
        "all_rows_decrypted": True,
        "exact": bool(np.array_equal(G, R)),
        "max_abs_err": int(np.max(np.abs(D))),
        "l1_err": int(np.sum(np.abs(D))),
        "diff": D.tolist(),
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


def run_oracle_raw_add_pipeline(rt, U, V):
    n = U.shape[0]

    U_bundle, _ = make_transparent_rowwise_bundle(rt, U.tolist())
    V_bundle, _ = make_transparent_rowwise_bundle(rt, V.tolist())

    U_cmt_bundle = logical_n_cmt_oracle(rt, U_bundle, n, "oracle_CMT_U")

    A_U = normalize_export(rt.export_component_coeff_matrix_u64(U_cmt_bundle.rows, part=1, coeff_count=n))
    B_U = normalize_export(rt.export_component_coeff_matrix_u64(U_cmt_bundle.rows, part=0, coeff_count=n))
    A_V = normalize_export(rt.export_component_coeff_matrix_u64(V_bundle.rows, part=1, coeff_count=n))
    B_V = normalize_export(rt.export_component_coeff_matrix_u64(V_bundle.rows, part=0, coeff_count=n))

    M00 = raw_rns_ppmm(A_U, A_V, n)
    M01 = raw_rns_ppmm(A_U, B_V, n)
    M10 = raw_rns_ppmm(B_U, A_V, n)
    M11 = raw_rns_ppmm(B_U, B_V, n)

    M00_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M00)
    M01_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M01)
    M10_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M10)
    M11_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M11)

    zero_rows = make_zero_1part_rows(rt, U_cmt_bundle.rows, num_towers=M11["num_towers"])

    # paper (M10,M11) -> OpenFHE c0=M11,c1=M10.
    bonly_rows = [
        rt.assemble_2part_from_1parts_coeff_ct(M11_rows[i], M10_rows[i])
        for i in range(n)
    ]

    # T01 = oracle Transpose((M01,0)).
    T01_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M01_rows,
        B_rows=zero_rows,
        n=n,
        label="T01=(M01,0)",
        template_bundle=U_bundle,
    )
    T01_cmt_bundle = logical_n_cmt_oracle(rt, T01_bundle, n, "oracle_CMT_T01")
    T01_rows = T01_cmt_bundle.rows

    # T00 = oracle Transpose((M00,0)).
    T00_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M00_rows,
        B_rows=zero_rows,
        n=n,
        label="T00=(M00,0)",
        template_bundle=U_bundle,
    )
    T00_cmt_bundle = logical_n_cmt_oracle(rt, T00_bundle, n, "oracle_CMT_T00")

    Bhat_rows, _ = extract_1part_rows(rt, T00_cmt_bundle.rows, part=0, coeff_count=0)
    Ahat_rows, _ = extract_1part_rows(rt, T00_cmt_bundle.rows, part=1, coeff_count=0)

    line5_rows = []
    line5_ks_inputs = []

    for i in range(n):
        ks_input = rt.assemble_3part_from_1parts_coeff_ct(
            zero_rows[i],
            zero_rows[i],
            Ahat_rows[i],
        )
        line5_ks_inputs.append(ks_input)

        ks_output = rt.relinearize_coeff_ct(ks_input)

        # paper (B_hat,0) -> OpenFHE c0=0,c1=B_hat.
        bhat_term = rt.assemble_2part_from_1parts_coeff_ct(
            zero_rows[i],
            Bhat_rows[i],
        )

        # line5 internal add remains OpenFHE add; X7j/X7k issue was line6 order.
        line5_rows.append(rt.add_coeff_ct(ks_output, bhat_term))

    # line6 uses raw RNS add.
    zero_terms = raw_add_2part_rows(rt, line5_rows, T01_rows, n)
    raw_full_rows = raw_add_2part_rows(rt, zero_terms, bonly_rows, n)

    final = decrypt_rows_after_compress(rt, raw_full_rows, logical_length=n)

    return {
        "expected_U_T_matmul_V": (U.T @ V).tolist(),
        "final": final,
        "comparison_to_U_T_matmul_V": compare_matrix(final["decoded"], U.T @ V),
        "shape_summaries": {
            "line5_ks_inputs": inspect_rows(rt, line5_ks_inputs, coeff_sample=4),
            "line5_rows": inspect_rows(rt, line5_rows, coeff_sample=4),
            "T01_rows": inspect_rows(rt, T01_rows, coeff_sample=4),
            "raw_full_rows": inspect_rows(rt, raw_full_rows, coeff_sample=4),
        },
    }


def main():
    n = 4
    rng = np.random.default_rng(209000)

    cases_spec = [
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
        "experiment": "wp4x7l_oracle_helper_smoke",
        "purpose": (
            "Verify reusable ccmm_oracle_debug helpers after X7k. "
            "The oracle/raw-add pipeline should equal U.T @ V for transparent c0=row,c1=0 inputs."
        ),
        "status": "debug_helper_smoke_no_security_no_homomorphic_cmt",
        "n": n,
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for spec in cases_spec:
            result = run_oracle_raw_add_pipeline(rt, spec["U"], spec["V"])
            report["cases"].append({
                "name": spec["name"],
                "input_U": spec["U"].tolist(),
                "input_V": spec["V"].tolist(),
                "result": result,
            })

    report["checks"] = {
        "all_cases_decrypted（所有 helper smoke case 都成功解密）": all(
            c["result"]["comparison_to_U_T_matmul_V"]["all_rows_decrypted"]
            for c in report["cases"]
        ),
        "all_cases_equal_U_T_matmul_V（所有 helper smoke case 都等于 U.T@V）": all(
            c["result"]["comparison_to_U_T_matmul_V"]["exact"]
            for c in report["cases"]
        ),
        "all_line5_ks_inputs_are_3part（line5 KS inputs 均为三分量）": all(
            rows_are_npart(c["result"]["shape_summaries"]["line5_ks_inputs"], n, 3)
            for c in report["cases"]
        ),
        "all_raw_full_rows_are_2part（raw-add full rows 均为二分量）": all(
            rows_are_npart(c["result"]["shape_summaries"]["raw_full_rows"], n, 2)
            for c in report["cases"]
        ),
    }

    report["summary"] = {
        "current_status": (
            "Reusable debug/oracle helpers are installed if all checks pass. "
            "This does not implement secure CCMM; it validates raw-RNS Algorithm 3 algebra under logical-n C-MT oracle."
        ),
        "remaining_real_ccmm_tasks": {
            "task1": "Implement or design real logical-n homomorphic C-MT for n << ring_dim, or run dimension-compatible experiments.",
            "task2": "Fix real coefficient-row encryption semantics; transparent c0=row,c1=0 is debug-only.",
            "task3": "Replace debug raw line6 add with a secure/homomorphic equivalent once real ciphertext semantics are restored.",
        },
        "next_step": "WP4-Y1",
        "next_step_goal": (
            "Decide implementation track: logical-n C-MT design for small matrices, or dimension-compatible benchmark path. "
            "PCMM remains preserved and separate."
        ),
        "zh": "X7l 通过后，debug/oracle 代数链条已经收敛；下一阶段回到真实安全实现路线选择。",
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x7l_oracle_helper_smoke_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X7l oracle helper smoke")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
