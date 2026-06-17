import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_wp4x7i_full_vs_bonly_term_isolation import (
    one_hot,
    normalize_export,
    make_transparent_rowwise_bundle,
    logical_n_cmt_oracle,
    raw_rns_ppmm,
    import_ppmm_rows,
    make_zero_1part_rows,
    make_temp_pair_bundle,
    extract_1part_rows,
    decrypt_rows_after_compress,
    compare_matrix,
    inspect_rows,
    rows_are_npart,
)

from hegpt import HEConfig, HERuntime


def raw_add_exported_parts(part_a, part_b, n):
    if part_a["num_rows"] != part_b["num_rows"]:
        raise ValueError("raw_add_exported_parts: row count mismatch")
    if part_a["num_towers"] != part_b["num_towers"]:
        raise ValueError("raw_add_exported_parts: tower count mismatch")
    if part_a["moduli_u64"] != part_b["moduli_u64"]:
        raise ValueError("raw_add_exported_parts: moduli mismatch")

    rows = []
    moduli = [int(q) for q in part_a["moduli_u64"]]

    for i in range(part_a["num_rows"]):
        towers_coeffs = []

        for t, q in enumerate(moduli):
            ca = part_a["rows"][i]["towers"][t]["coeffs_u64"][:n]
            cb = part_b["rows"][i]["towers"][t]["coeffs_u64"][:n]
            towers_coeffs.append([
                (int(x) + int(y)) % q
                for x, y in zip(ca, cb)
            ])

        rows.append({
            "row_index": i,
            "towers_coeffs": towers_coeffs,
        })

    return {
        "num_rows": part_a["num_rows"],
        "num_towers": part_a["num_towers"],
        "moduli_u64": moduli,
        "rows": rows,
    }


def raw_add_2part_rows(rt, rows_a, rows_b, n):
    """
    Raw RNS add for two-component coefficient rows.

    This avoids OpenFHE EvalAdd/add_coeff_ct metadata/order behavior.
    It only preserves the first n coefficients, which is enough for this logical-n oracle diagnostic.
    """
    a0 = normalize_export(rt.export_component_coeff_matrix_u64(rows_a, part=0, coeff_count=n))
    a1 = normalize_export(rt.export_component_coeff_matrix_u64(rows_a, part=1, coeff_count=n))
    b0 = normalize_export(rt.export_component_coeff_matrix_u64(rows_b, part=0, coeff_count=n))
    b1 = normalize_export(rt.export_component_coeff_matrix_u64(rows_b, part=1, coeff_count=n))

    sum0 = raw_add_exported_parts(a0, b0, n)
    sum1 = raw_add_exported_parts(a1, b1, n)

    c0_rows = import_ppmm_rows(rt, rows_a, sum0)
    c1_rows = import_ppmm_rows(rt, rows_a, sum1)

    return [
        rt.assemble_2part_from_1parts_coeff_ct(c0_rows[i], c1_rows[i])
        for i in range(n)
    ]


def run_case(rt, U, V):
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

    # B-only / (M10,M11):
    # paper pair (M10,M11), OpenFHE c0=M11,c1=M10.
    bonly_rows = [
        rt.assemble_2part_from_1parts_coeff_ct(M11_rows[i], M10_rows[i])
        for i in range(n)
    ]

    # T01 = Transpose((M01,0))
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

    # T00 = Transpose((M00,0))
    T00_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M00_rows,
        B_rows=zero_rows,
        n=n,
        label="T00=(M00,0)",
        template_bundle=U_bundle,
    )
    T00_cmt_bundle = logical_n_cmt_oracle(rt, T00_bundle, n, "oracle_CMT_T00")

    # T00_cmt paper pair (A_hat,B_hat):
    # OpenFHE c0=B_hat, c1=A_hat.
    Bhat_rows, _ = extract_1part_rows(rt, T00_cmt_bundle.rows, part=0, coeff_count=0)
    Ahat_rows, _ = extract_1part_rows(rt, T00_cmt_bundle.rows, part=1, coeff_count=0)

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

        # paper (B_hat,0) -> OpenFHE c0=0,c1=B_hat.
        bhat_term = rt.assemble_2part_from_1parts_coeff_ct(
            zero_rows[i],
            Bhat_rows[i],
        )
        line5_bhat_terms.append(bhat_term)

        # Keep OpenFHE add here only inside line5, matching earlier probes.
        # In transparent/B-only cases line5 has already been verified to decrypt to zero.
        line5_rows.append(rt.add_coeff_ct(ks_output, bhat_term))

    # Raw-add variants.
    raw_bonly_plus_line5 = raw_add_2part_rows(rt, bonly_rows, line5_rows, n)
    raw_line5_plus_bonly = raw_add_2part_rows(rt, line5_rows, bonly_rows, n)
    raw_bonly_plus_T01 = raw_add_2part_rows(rt, bonly_rows, T01_rows, n)
    raw_T01_plus_bonly = raw_add_2part_rows(rt, T01_rows, bonly_rows, n)
    raw_line5_plus_T01 = raw_add_2part_rows(rt, line5_rows, T01_rows, n)
    raw_T01_plus_line5 = raw_add_2part_rows(rt, T01_rows, line5_rows, n)

    raw_full_left_assoc = raw_add_2part_rows(rt, raw_add_2part_rows(rt, bonly_rows, line5_rows, n), T01_rows, n)
    raw_full_zero_first = raw_add_2part_rows(rt, raw_add_2part_rows(rt, line5_rows, T01_rows, n), bonly_rows, n)
    raw_full_paper_order = raw_add_2part_rows(rt, raw_add_2part_rows(rt, line5_rows, T01_rows, n), bonly_rows, n)

    decoded = {
        "bonly": decrypt_rows_after_compress(rt, bonly_rows, logical_length=n),
        "line5": decrypt_rows_after_compress(rt, line5_rows, logical_length=n),
        "T01": decrypt_rows_after_compress(rt, T01_rows, logical_length=n),
        "raw_bonly_plus_line5": decrypt_rows_after_compress(rt, raw_bonly_plus_line5, logical_length=n),
        "raw_line5_plus_bonly": decrypt_rows_after_compress(rt, raw_line5_plus_bonly, logical_length=n),
        "raw_bonly_plus_T01": decrypt_rows_after_compress(rt, raw_bonly_plus_T01, logical_length=n),
        "raw_T01_plus_bonly": decrypt_rows_after_compress(rt, raw_T01_plus_bonly, logical_length=n),
        "raw_line5_plus_T01": decrypt_rows_after_compress(rt, raw_line5_plus_T01, logical_length=n),
        "raw_T01_plus_line5": decrypt_rows_after_compress(rt, raw_T01_plus_line5, logical_length=n),
        "raw_full_left_assoc": decrypt_rows_after_compress(rt, raw_full_left_assoc, logical_length=n),
        "raw_full_zero_first": decrypt_rows_after_compress(rt, raw_full_zero_first, logical_length=n),
        "raw_full_paper_order": decrypt_rows_after_compress(rt, raw_full_paper_order, logical_length=n),
    }

    zero_ref = np.zeros((n, n), dtype=np.int64)
    bonly_ref = U.T @ V
    bonly_dec = np.array(decoded["bonly"]["decoded"], dtype=np.int64)

    comparisons = {
        "bonly_equals_U_T_matmul_V": compare_matrix(decoded["bonly"]["decoded"], bonly_ref),
        "line5_is_zero": compare_matrix(decoded["line5"]["decoded"], zero_ref),
        "T01_is_zero": compare_matrix(decoded["T01"]["decoded"], zero_ref),

        "raw_bonly_plus_line5_equals_bonly": compare_matrix(decoded["raw_bonly_plus_line5"]["decoded"], bonly_dec),
        "raw_line5_plus_bonly_equals_bonly": compare_matrix(decoded["raw_line5_plus_bonly"]["decoded"], bonly_dec),
        "raw_bonly_plus_T01_equals_bonly": compare_matrix(decoded["raw_bonly_plus_T01"]["decoded"], bonly_dec),
        "raw_T01_plus_bonly_equals_bonly": compare_matrix(decoded["raw_T01_plus_bonly"]["decoded"], bonly_dec),
        "raw_line5_plus_T01_is_zero": compare_matrix(decoded["raw_line5_plus_T01"]["decoded"], zero_ref),
        "raw_T01_plus_line5_is_zero": compare_matrix(decoded["raw_T01_plus_line5"]["decoded"], zero_ref),

        "raw_full_left_assoc_equals_bonly": compare_matrix(decoded["raw_full_left_assoc"]["decoded"], bonly_dec),
        "raw_full_zero_first_equals_bonly": compare_matrix(decoded["raw_full_zero_first"]["decoded"], bonly_dec),
        "raw_full_paper_order_equals_bonly": compare_matrix(decoded["raw_full_paper_order"]["decoded"], bonly_dec),
        "raw_full_paper_order_equals_U_T_matmul_V": compare_matrix(decoded["raw_full_paper_order"]["decoded"], bonly_ref),
    }

    return {
        "expected_bonly_U_T_matmul_V": bonly_ref.tolist(),
        "decoded": decoded,
        "comparisons": comparisons,
        "shape_summaries": {
            "bonly": inspect_rows(rt, bonly_rows, coeff_sample=4),
            "line5": inspect_rows(rt, line5_rows, coeff_sample=4),
            "T01": inspect_rows(rt, T01_rows, coeff_sample=4),
            "raw_full_paper_order": inspect_rows(rt, raw_full_paper_order, coeff_sample=4),
            "line5_ks_inputs": inspect_rows(rt, line5_ks_inputs, coeff_sample=4),
            "line5_ks_outputs": inspect_rows(rt, line5_ks_outputs, coeff_sample=4),
        },
    }


def main():
    n = 4
    rng = np.random.default_rng(208900)

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
        "experiment": "wp4x7k_raw_rns_add_line6_probe",
        "purpose": (
            "Replace line6 add_coeff_ct with raw RNS component addition for logical-n oracle pipeline. "
            "This tests whether line6 becomes additive-order independent on raw-imported coefficient handles."
        ),
        "status": "debug_raw_rns_add_probe_no_security_no_homomorphic_cmt",
        "n": n,
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for spec in cases_spec:
            result = run_case(rt, spec["U"], spec["V"])
            report["cases"].append({
                "name": spec["name"],
                "input_U": spec["U"].tolist(),
                "input_V": spec["V"].tolist(),
                "result": result,
            })

    comparison_names = [
        "bonly_equals_U_T_matmul_V",
        "line5_is_zero",
        "T01_is_zero",
        "raw_bonly_plus_line5_equals_bonly",
        "raw_line5_plus_bonly_equals_bonly",
        "raw_bonly_plus_T01_equals_bonly",
        "raw_T01_plus_bonly_equals_bonly",
        "raw_line5_plus_T01_is_zero",
        "raw_T01_plus_line5_is_zero",
        "raw_full_left_assoc_equals_bonly",
        "raw_full_zero_first_equals_bonly",
        "raw_full_paper_order_equals_bonly",
        "raw_full_paper_order_equals_U_T_matmul_V",
    ]

    report["checks"] = {
        name + "（" + name + "）": all(
            c["result"]["comparisons"][name]["exact"]
            for c in report["cases"]
        )
        for name in comparison_names
    }

    report["checks"].update({
        "all_raw_full_rows_are_2part（raw-add full rows 均为二分量）": all(
            rows_are_npart(c["result"]["shape_summaries"]["raw_full_paper_order"], n, 2)
            for c in report["cases"]
        ),
        "all_line5_ks_inputs_are_3part（line5 KS inputs 均为三分量）": all(
            rows_are_npart(c["result"]["shape_summaries"]["line5_ks_inputs"], n, 3)
            for c in report["cases"]
        ),
    })

    failed_checks = [
        k for k, v in report["checks"].items()
        if v is not True
    ]

    report["summary"] = {
        "failed_checks": failed_checks,
        "if_all_raw_add_checks_pass": (
            "The line6 mismatch was caused by add_coeff_ct metadata/order behavior. "
            "Raw RNS component add is the correct debug/oracle add path for paper Algorithm 3 line6."
        ),
        "if_raw_add_still_fails": (
            "The issue is deeper than add_coeff_ct; inspect raw export/import of line5/T01 after relinearization."
        ),
        "next_step": "WP4-X7l",
        "next_step_goal": (
            "Promote raw RNS add into a reusable debug/oracle Algorithm 3 implementation, then decide between "
            "logical-n C-MT design and dimension-compatible benchmark for real homomorphic CCMM."
        ),
        "zh": "如果 X7k 通过，line6 组合应改用 raw RNS component add；真实实现后续再换成安全同态等价操作。",
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x7k_raw_rns_add_line6_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X7k raw RNS add line6 probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
