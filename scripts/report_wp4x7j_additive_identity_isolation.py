import json
import sys
from pathlib import Path

import numpy as np

# Allow importing helpers from X7i.
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


def export_2part_prefix(rt, rows, n):
    c0 = normalize_export(rt.export_component_coeff_matrix_u64(rows, part=0, coeff_count=n))
    c1 = normalize_export(rt.export_component_coeff_matrix_u64(rows, part=1, coeff_count=n))
    return {
        "c0_row0_tower0": c0["rows"][0]["towers"][0]["coeffs_u64"],
        "c1_row0_tower0": c1["rows"][0]["towers"][0]["coeffs_u64"],
        "c0_moduli": c0["moduli_u64"],
        "c1_moduli": c1["moduli_u64"],
    }


def run_case_with_add_isolation(rt, U, V):
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

    # B-only / M10_M11 pair:
    # paper (M10,M11), OpenFHE c0=M11,c1=M10.
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

        line5_rows.append(rt.add_coeff_ct(ks_output, bhat_term))

    # Additive identity probes.
    bonly_plus_line5 = [
        rt.add_coeff_ct(bonly_rows[i], line5_rows[i])
        for i in range(n)
    ]

    line5_plus_bonly = [
        rt.add_coeff_ct(line5_rows[i], bonly_rows[i])
        for i in range(n)
    ]

    bonly_plus_T01 = [
        rt.add_coeff_ct(bonly_rows[i], T01_rows[i])
        for i in range(n)
    ]

    T01_plus_bonly = [
        rt.add_coeff_ct(T01_rows[i], bonly_rows[i])
        for i in range(n)
    ]

    line5_plus_T01 = [
        rt.add_coeff_ct(line5_rows[i], T01_rows[i])
        for i in range(n)
    ]

    T01_plus_line5 = [
        rt.add_coeff_ct(T01_rows[i], line5_rows[i])
        for i in range(n)
    ]

    bonly_plus_line5_plus_T01 = [
        rt.add_coeff_ct(bonly_plus_line5[i], T01_rows[i])
        for i in range(n)
    ]

    line5_plus_T01_plus_bonly = [
        rt.add_coeff_ct(line5_plus_T01[i], bonly_rows[i])
        for i in range(n)
    ]

    # Decode everything.
    zero_ref = np.zeros((n, n), dtype=np.int64)
    bonly_ref = U.T @ V

    decoded = {
        "bonly": decrypt_rows_after_compress(rt, bonly_rows, logical_length=n),
        "line5": decrypt_rows_after_compress(rt, line5_rows, logical_length=n),
        "T01": decrypt_rows_after_compress(rt, T01_rows, logical_length=n),
        "bonly_plus_line5": decrypt_rows_after_compress(rt, bonly_plus_line5, logical_length=n),
        "line5_plus_bonly": decrypt_rows_after_compress(rt, line5_plus_bonly, logical_length=n),
        "bonly_plus_T01": decrypt_rows_after_compress(rt, bonly_plus_T01, logical_length=n),
        "T01_plus_bonly": decrypt_rows_after_compress(rt, T01_plus_bonly, logical_length=n),
        "line5_plus_T01": decrypt_rows_after_compress(rt, line5_plus_T01, logical_length=n),
        "T01_plus_line5": decrypt_rows_after_compress(rt, T01_plus_line5, logical_length=n),
        "bonly_plus_line5_plus_T01": decrypt_rows_after_compress(rt, bonly_plus_line5_plus_T01, logical_length=n),
        "line5_plus_T01_plus_bonly": decrypt_rows_after_compress(rt, line5_plus_T01_plus_bonly, logical_length=n),
    }

    comparisons = {
        "bonly_equals_ref": compare_matrix(decoded["bonly"]["decoded"], bonly_ref),
        "line5_is_zero": compare_matrix(decoded["line5"]["decoded"], zero_ref),
        "T01_is_zero": compare_matrix(decoded["T01"]["decoded"], zero_ref),

        "bonly_plus_line5_equals_bonly": compare_matrix(
            decoded["bonly_plus_line5"]["decoded"],
            np.array(decoded["bonly"]["decoded"], dtype=np.int64),
        ),
        "line5_plus_bonly_equals_bonly": compare_matrix(
            decoded["line5_plus_bonly"]["decoded"],
            np.array(decoded["bonly"]["decoded"], dtype=np.int64),
        ),
        "bonly_plus_T01_equals_bonly": compare_matrix(
            decoded["bonly_plus_T01"]["decoded"],
            np.array(decoded["bonly"]["decoded"], dtype=np.int64),
        ),
        "T01_plus_bonly_equals_bonly": compare_matrix(
            decoded["T01_plus_bonly"]["decoded"],
            np.array(decoded["bonly"]["decoded"], dtype=np.int64),
        ),
        "line5_plus_T01_is_zero": compare_matrix(
            decoded["line5_plus_T01"]["decoded"],
            zero_ref,
        ),
        "T01_plus_line5_is_zero": compare_matrix(
            decoded["T01_plus_line5"]["decoded"],
            zero_ref,
        ),
        "bonly_plus_line5_plus_T01_equals_bonly": compare_matrix(
            decoded["bonly_plus_line5_plus_T01"]["decoded"],
            np.array(decoded["bonly"]["decoded"], dtype=np.int64),
        ),
        "line5_plus_T01_plus_bonly_equals_bonly": compare_matrix(
            decoded["line5_plus_T01_plus_bonly"]["decoded"],
            np.array(decoded["bonly"]["decoded"], dtype=np.int64),
        ),
    }

    return {
        "expected_bonly_U_T_matmul_V": bonly_ref.tolist(),
        "decoded": decoded,
        "comparisons": comparisons,
        "shape_summaries": {
            "bonly": inspect_rows(rt, bonly_rows, coeff_sample=4),
            "line5": inspect_rows(rt, line5_rows, coeff_sample=4),
            "T01": inspect_rows(rt, T01_rows, coeff_sample=4),
            "bonly_plus_line5": inspect_rows(rt, bonly_plus_line5, coeff_sample=4),
            "bonly_plus_T01": inspect_rows(rt, bonly_plus_T01, coeff_sample=4),
            "line5_plus_T01": inspect_rows(rt, line5_plus_T01, coeff_sample=4),
            "line5_ks_inputs": inspect_rows(rt, line5_ks_inputs, coeff_sample=4),
            "line5_ks_outputs": inspect_rows(rt, line5_ks_outputs, coeff_sample=4),
        },
        "raw_prefix": {
            "bonly": export_2part_prefix(rt, bonly_rows, n),
            "line5": export_2part_prefix(rt, line5_rows, n),
            "T01": export_2part_prefix(rt, T01_rows, n),
            "bonly_plus_line5": export_2part_prefix(rt, bonly_plus_line5, n),
            "bonly_plus_T01": export_2part_prefix(rt, bonly_plus_T01, n),
            "line5_plus_T01": export_2part_prefix(rt, line5_plus_T01, n),
            "bonly_plus_line5_plus_T01": export_2part_prefix(rt, bonly_plus_line5_plus_T01, n),
            "line5_plus_T01_plus_bonly": export_2part_prefix(rt, line5_plus_T01_plus_bonly, n),
        },
    }


def main():
    n = 4
    rng = np.random.default_rng(208800)

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
        "experiment": "wp4x7j_additive_identity_isolation",
        "purpose": (
            "Isolate why full = line5 + T01 + B-only does not equal B-only even though "
            "line5 and T01 decrypt to zero individually. This probes add_coeff_ct identity behavior "
            "on raw-imported coefficient ciphertexts."
        ),
        "status": "debug_additive_identity_probe",
        "n": n,
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for spec in cases_spec:
            U = spec["U"]
            V = spec["V"]
            result = run_case_with_add_isolation(rt, U, V)

            report["cases"].append({
                "name": spec["name"],
                "input_U": U.tolist(),
                "input_V": V.tolist(),
                "result": result,
            })

    comparison_names = [
        "bonly_equals_ref",
        "line5_is_zero",
        "T01_is_zero",
        "bonly_plus_line5_equals_bonly",
        "line5_plus_bonly_equals_bonly",
        "bonly_plus_T01_equals_bonly",
        "T01_plus_bonly_equals_bonly",
        "line5_plus_T01_is_zero",
        "T01_plus_line5_is_zero",
        "bonly_plus_line5_plus_T01_equals_bonly",
        "line5_plus_T01_plus_bonly_equals_bonly",
    ]

    report["checks"] = {
        name + "（" + name + "）": all(
            c["result"]["comparisons"][name]["exact"]
            for c in report["cases"]
        )
        for name in comparison_names
    }

    report["checks"].update({
        "all_rows_are_2part_after_adds（加法后的 rows 都仍是二分量）": all(
            rows_are_npart(c["result"]["shape_summaries"]["bonly_plus_line5"], n, 2)
            and rows_are_npart(c["result"]["shape_summaries"]["bonly_plus_T01"], n, 2)
            and rows_are_npart(c["result"]["shape_summaries"]["line5_plus_T01"], n, 2)
            for c in report["cases"]
        ),
    })

    failed_checks = [
        k for k, v in report["checks"].items()
        if v is not True
    ]

    report["summary"] = {
        "failed_checks": failed_checks,
        "if_add_zero_fails": (
            "A term that decrypts to zero is not an algebraic zero for add_coeff_ct. "
            "Use raw RNS coefficient addition for Algorithm 3 line6 instead of OpenFHE EvalAdd on raw-imported debug handles."
        ),
        "if_order_differs": (
            "If B+zero differs from zero+B, add path has metadata/order dependence."
        ),
        "if_all_pass": (
            "Then X7i full mismatch came from script assembly, not add_coeff_ct; inspect X7i full construction."
        ),
        "next_step": "WP4-X7k",
        "zh": "X7j 用于判断 line6 是否必须改成 raw coefficient add，而不能使用 add_coeff_ct。",
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x7j_additive_identity_isolation_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X7j additive identity isolation")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
