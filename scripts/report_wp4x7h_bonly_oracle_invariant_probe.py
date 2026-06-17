import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


class TempRowwiseBundle:
    def __init__(self, rows, shape, label, ring_dim=1 << 14, batch_size=8):
        self.rows = rows
        self.shape = shape
        self.orientation = "rowwise"
        self.label = label
        self.ring_dim = int(ring_dim)
        self.batch_size = int(batch_size)


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def normalize_export(x):
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def one_hot(n, i, j, value=1):
    M = np.zeros((n, n), dtype=np.int64)
    M[i, j] = value
    return M


def signed_to_mod(v, q):
    return int(v) % int(q)


def make_zero_1part(rt, template_ct, num_towers):
    return rt.import_1part_coeff_u64(template_ct, [[] for _ in range(num_towers)])


def make_zero_1part_rows(rt, template_rows, num_towers):
    return [make_zero_1part(rt, tmpl, num_towers) for tmpl in template_rows]


def make_transparent_coeff_row_ct(rt, template_ct, row, moduli):
    towers_coeffs = []
    for q in moduli:
        towers_coeffs.append([signed_to_mod(v, q) for v in row])

    c0 = rt.import_1part_coeff_u64(template_ct, towers_coeffs)
    c1 = make_zero_1part(rt, template_ct, len(moduli))

    # transparent debug ciphertext:
    #   c0 = row
    #   c1 = 0
    return rt.assemble_2part_from_1parts_coeff_ct(c0, c1)


def make_transparent_rowwise_bundle(rt, matrix, ring_dim=1 << 14, batch_size=8):
    n = len(matrix)

    template = rt.encrypt_coeff_row_i64([0 for _ in range(n)])
    template_export = normalize_export(
        rt.export_component_coeff_matrix_u64([template], part=0, coeff_count=1)
    )
    moduli = [int(q) for q in template_export["moduli_u64"]]

    rows = [
        make_transparent_coeff_row_ct(rt, template, list(map(int, row)), moduli)
        for row in matrix
    ]

    return TempRowwiseBundle(
        rows=rows,
        shape=(n, n),
        label="transparent_coeff_rowwise_debug_bundle",
        ring_dim=ring_dim,
        batch_size=batch_size,
    ), {
        "template_num_towers": template_export["num_towers"],
        "template_ring_dim": template_export["ring_dim"],
        "moduli_u64": moduli,
    }


def matrix_for_tower(export_obj, tower_index, n):
    out = []
    for row in export_obj["rows"]:
        out.append([int(v) for v in row["towers"][tower_index]["coeffs_u64"][:n]])
    return out


def transpose_matrix(M):
    n = len(M)
    return [[M[i][j] for i in range(n)] for j in range(n)]


def logical_n_transpose_export(export_obj, n):
    num_towers = int(export_obj["num_towers"])
    moduli = [int(q) for q in export_obj["moduli_u64"]]

    tower_transposes = []
    for t in range(num_towers):
        M = matrix_for_tower(export_obj, t, n)
        tower_transposes.append(transpose_matrix(M))

    rows = []
    for i in range(n):
        towers_coeffs = []
        for t in range(num_towers):
            towers_coeffs.append([int(v) for v in tower_transposes[t][i]])
        rows.append({
            "row_index": i,
            "towers_coeffs": towers_coeffs,
        })

    return {
        "num_rows": n,
        "num_towers": num_towers,
        "moduli_u64": moduli,
        "rows": rows,
    }


def import_1part_rows(rt, template_rows, exported_rows):
    out = []
    for i, row in enumerate(exported_rows["rows"]):
        out.append(rt.import_1part_coeff_u64(template_rows[i], row["towers_coeffs"]))
    return out


def logical_n_cmt_oracle(rt, bundle, n, label):
    c0_export = normalize_export(
        rt.export_component_coeff_matrix_u64(bundle.rows, part=0, coeff_count=n)
    )
    c1_export = normalize_export(
        rt.export_component_coeff_matrix_u64(bundle.rows, part=1, coeff_count=n)
    )

    c0_T = logical_n_transpose_export(c0_export, n)
    c1_T = logical_n_transpose_export(c1_export, n)

    c0_rows = import_1part_rows(rt, bundle.rows, c0_T)
    c1_rows = import_1part_rows(rt, bundle.rows, c1_T)

    out_rows = [
        rt.assemble_2part_from_1parts_coeff_ct(c0_rows[i], c1_rows[i])
        for i in range(n)
    ]

    return TempRowwiseBundle(
        rows=out_rows,
        shape=(n, n),
        label=label,
        ring_dim=bundle.ring_dim,
        batch_size=bundle.batch_size,
    )


def matmul_mod(A, B, q):
    n = len(A)
    m = len(B[0])
    kdim = len(B)
    q = int(q)

    out = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            acc = 0
            for k in range(kdim):
                acc = (acc + int(A[i][k]) * int(B[k][j])) % q
            out[i][j] = acc

    return out


def raw_rns_ppmm(left_export, right_export, n):
    if left_export["num_rows"] != n or right_export["num_rows"] != n:
        raise ValueError("raw_rns_ppmm expects n rows in both inputs")
    if left_export["num_towers"] != right_export["num_towers"]:
        raise ValueError("tower count mismatch")
    if left_export["moduli_u64"] != right_export["moduli_u64"]:
        raise ValueError("RNS moduli mismatch")

    num_towers = int(left_export["num_towers"])
    moduli = [int(q) for q in left_export["moduli_u64"]]

    tower_mats = []
    for t in range(num_towers):
        A = matrix_for_tower(left_export, t, n)
        B = matrix_for_tower(right_export, t, n)
        tower_mats.append(matmul_mod(A, B, moduli[t]))

    rows = []
    for i in range(n):
        towers_coeffs = []
        for t in range(num_towers):
            towers_coeffs.append([int(v) for v in tower_mats[t][i]])
        rows.append({
            "row_index": i,
            "towers_coeffs": towers_coeffs,
        })

    return {
        "num_rows": n,
        "num_towers": num_towers,
        "moduli_u64": moduli,
        "rows": rows,
        "tower_mats_prefix": tower_mats,
    }


def import_ppmm_rows(rt, template_rows, ppmm_result):
    out = []
    for i, row in enumerate(ppmm_result["rows"]):
        out.append(rt.import_1part_coeff_u64(template_rows[i], row["towers_coeffs"]))
    return out


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


def nonzero_positions(M):
    if M is None:
        return None
    out = []
    for i, row in enumerate(M):
        if row is None:
            return None
        for j, v in enumerate(row):
            if int(v) != 0:
                out.append([int(i), int(j), int(v)])
    return out


def run_bonly_case(rt, U, V):
    n = U.shape[0]

    U_bundle, _ = make_transparent_rowwise_bundle(rt, U.tolist())
    V_bundle, _ = make_transparent_rowwise_bundle(rt, V.tolist())

    U_cmt_bundle = logical_n_cmt_oracle(rt, U_bundle, n, "oracle_CMT_U")

    # For transparent input:
    #   A_U = 0, B_U = U.T
    #   A_V = 0, B_V = V
    #
    # Therefore:
    #   only M11 = B_U @ B_V = U.T @ V should be nonzero.
    B_U = normalize_export(rt.export_component_coeff_matrix_u64(U_cmt_bundle.rows, part=0, coeff_count=n))
    B_V = normalize_export(rt.export_component_coeff_matrix_u64(V_bundle.rows, part=0, coeff_count=n))

    M11 = raw_rns_ppmm(B_U, B_V, n)
    M11_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M11)

    # Reduced expected final pair:
    # paper (A=0, B=M11) -> OpenFHE c0=M11, c1=0.
    zero_rows = make_zero_1part_rows(rt, U_cmt_bundle.rows, num_towers=M11["num_towers"])

    reduced_rows = [
        rt.assemble_2part_from_1parts_coeff_ct(M11_rows[i], zero_rows[i])
        for i in range(n)
    ]

    final = decrypt_rows_after_compress(rt, reduced_rows, logical_length=n)

    return {
        "B_U_summary": {
            "row0_tower0_coeffs": B_U["rows"][0]["towers"][0]["coeffs_u64"],
            "moduli_u64": B_U["moduli_u64"],
        },
        "B_V_summary": {
            "row0_tower0_coeffs": B_V["rows"][0]["towers"][0]["coeffs_u64"],
            "moduli_u64": B_V["moduli_u64"],
        },
        "M11_summary": {
            "row0_tower0_coeffs": M11["rows"][0]["towers_coeffs"][0],
            "row1_tower0_coeffs": M11["rows"][1]["towers_coeffs"][0],
        },
        "reduced_rows_summary": inspect_rows(rt, reduced_rows, coeff_sample=4),
        "final": final,
    }


def run_zero_invariant_case(rt, n):
    template_bundle, template_info = make_transparent_rowwise_bundle(rt, np.zeros((n, n), dtype=np.int64).tolist())

    template_export = normalize_export(
        rt.export_component_coeff_matrix_u64(template_bundle.rows, part=0, coeff_count=1)
    )
    num_towers = template_export["num_towers"]

    zero_rows = make_zero_1part_rows(rt, template_bundle.rows, num_towers=num_towers)

    zero_pair_rows = [
        rt.assemble_2part_from_1parts_coeff_ct(zero_rows[i], zero_rows[i])
        for i in range(n)
    ]

    zero_pair_bundle = TempRowwiseBundle(
        rows=zero_pair_rows,
        shape=(n, n),
        label="zero_pair_bundle",
        ring_dim=template_bundle.ring_dim,
        batch_size=template_bundle.batch_size,
    )

    zero_cmt_bundle = logical_n_cmt_oracle(rt, zero_pair_bundle, n, "oracle_CMT_zero")

    zero_3part_rows = [
        rt.assemble_3part_from_1parts_coeff_ct(zero_rows[i], zero_rows[i], zero_rows[i])
        for i in range(n)
    ]

    zero_relin_rows = [
        rt.relinearize_coeff_ct(zero_3part_rows[i])
        for i in range(n)
    ]

    zero_sum_rows = [
        rt.add_coeff_ct(zero_cmt_bundle.rows[i], zero_relin_rows[i])
        for i in range(n)
    ]

    dec_zero_pair = decrypt_rows_after_compress(rt, zero_pair_rows, logical_length=n)
    dec_zero_cmt = decrypt_rows_after_compress(rt, zero_cmt_bundle.rows, logical_length=n)
    dec_zero_relin = decrypt_rows_after_compress(rt, zero_relin_rows, logical_length=n)
    dec_zero_sum = decrypt_rows_after_compress(rt, zero_sum_rows, logical_length=n)

    zero_ref = np.zeros((n, n), dtype=np.int64)

    return {
        "template_info": template_info,
        "zero_pair_summary": inspect_rows(rt, zero_pair_rows, coeff_sample=4),
        "zero_cmt_summary": inspect_rows(rt, zero_cmt_bundle.rows, coeff_sample=4),
        "zero_3part_summary": inspect_rows(rt, zero_3part_rows, coeff_sample=4),
        "zero_relin_summary": inspect_rows(rt, zero_relin_rows, coeff_sample=4),
        "zero_sum_summary": inspect_rows(rt, zero_sum_rows, coeff_sample=4),
        "decrypt_zero_pair": dec_zero_pair,
        "decrypt_zero_cmt": dec_zero_cmt,
        "decrypt_zero_relin": dec_zero_relin,
        "decrypt_zero_sum": dec_zero_sum,
        "comparisons": {
            "zero_pair_exact": compare_matrix(dec_zero_pair["decoded"], zero_ref),
            "zero_cmt_exact": compare_matrix(dec_zero_cmt["decoded"], zero_ref),
            "zero_relin_exact": compare_matrix(dec_zero_relin["decoded"], zero_ref),
            "zero_sum_exact": compare_matrix(dec_zero_sum["decoded"], zero_ref),
        },
    }


def main():
    n = 4
    rng = np.random.default_rng(208600)

    cases_spec = [
        {
            "name": "basis_E00_times_E00",
            "U": one_hot(n, 0, 0),
            "V": one_hot(n, 0, 0),
            "expected_B_only": one_hot(n, 0, 0).T @ one_hot(n, 0, 0),
            "zh": "B-only transparent expected = U.T @ V",
        },
        {
            "name": "basis_E01_times_E10",
            "U": one_hot(n, 0, 1),
            "V": one_hot(n, 1, 0),
            "expected_B_only": one_hot(n, 0, 1).T @ one_hot(n, 1, 0),
            "zh": "B-only transparent expected = U.T @ V",
        },
        {
            "name": "basis_E01_times_E12",
            "U": one_hot(n, 0, 1),
            "V": one_hot(n, 1, 2),
            "expected_B_only": one_hot(n, 0, 1).T @ one_hot(n, 1, 2),
            "zh": "B-only transparent expected = U.T @ V",
        },
        {
            "name": "basis_E23_times_E31",
            "U": one_hot(n, 2, 3),
            "V": one_hot(n, 3, 1),
            "expected_B_only": one_hot(n, 2, 3).T @ one_hot(n, 3, 1),
            "zh": "B-only transparent expected = U.T @ V",
        },
        {
            "name": "identity_times_random_small",
            "U": np.eye(n, dtype=np.int64),
            "V": rng.integers(-2, 3, size=(n, n)).astype(np.int64),
            "expected_B_only": None,
            "zh": "B-only transparent expected = I.T @ V = V",
        },
        {
            "name": "random_small_times_identity",
            "U": rng.integers(-2, 3, size=(n, n)).astype(np.int64),
            "V": np.eye(n, dtype=np.int64),
            "expected_B_only": None,
            "zh": "B-only transparent expected = U.T @ I = U.T",
        },
    ]

    # Fill expected_B_only for random cases.
    for c in cases_spec:
        if c["expected_B_only"] is None:
            c["expected_B_only"] = c["U"].T @ c["V"]

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
        "experiment": "wp4x7h_bonly_oracle_invariant_probe",
        "purpose": (
            "Isolate transparent-input B-only algebra under logical-n C-MT oracle. "
            "For c0=row,c1=0 inputs, Algorithm 3 should reduce to M11=B_U@B_V=U.T@V. "
            "Also verify zero C-MT/relinearize/add invariants."
        ),
        "status": "debug_oracle_bonly_and_zero_invariant_probe",
        "n": n,
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        report["zero_invariants"] = run_zero_invariant_case(rt, n)

        for spec in cases_spec:
            U = spec["U"]
            V = spec["V"]
            expected = spec["expected_B_only"]

            result = run_bonly_case(rt, U, V)
            comparison = compare_matrix(result["final"]["decoded"], expected)

            report["cases"].append({
                "name": spec["name"],
                "zh": spec["zh"],
                "input_U": U.tolist(),
                "input_V": V.tolist(),
                "expected_B_only_U_transpose_matmul_V": expected.tolist(),
                "result": result,
                "comparison_to_U_transpose_matmul_V": comparison,
                "got_nonzero_positions": nonzero_positions(result["final"]["decoded"]),
                "expected_nonzero_positions": nonzero_positions(expected.tolist()),
            })

    zero_cmp = report["zero_invariants"]["comparisons"]

    report["checks"] = {
        "zero_pair_decrypts_to_zero（zero pair 解密为 0）": zero_cmp["zero_pair_exact"]["exact"],
        "zero_cmt_decrypts_to_zero（oracle C-MT(zero) 解密为 0）": zero_cmp["zero_cmt_exact"]["exact"],
        "zero_relin_decrypts_to_zero（relinearize(zero 3part) 解密为 0）": zero_cmp["zero_relin_exact"]["exact"],
        "zero_add_decrypts_to_zero（zero CMT + zero relin 解密为 0）": zero_cmp["zero_sum_exact"]["exact"],
        "bonly_all_cases_decrypted（B-only reduced pipeline 全部成功解密）": all(
            c["comparison_to_U_transpose_matmul_V"]["all_rows_decrypted"]
            for c in report["cases"]
        ),
        "bonly_all_cases_match_U_transpose_matmul_V（B-only reduced pipeline 全部等于 U.T@V）": all(
            c["comparison_to_U_transpose_matmul_V"]["exact"]
            for c in report["cases"]
        ),
        "bonly_exact_case_count（B-only reduced pipeline 等于 U.T@V 的 case 数）": sum(
            1 for c in report["cases"]
            if c["comparison_to_U_transpose_matmul_V"]["exact"]
        ),
    }

    report["summary"] = {
        "if_bonly_passes_and_zero_passes": (
            "Raw PP-MM/import/decode and zero invariants are correct. The remaining full-pipeline mismatch is likely line5/line6 placement for nonzero A components, or the transparent-input expected reference was wrong."
        ),
        "if_bonly_fails": (
            "The raw PP-MM orientation/import/decode is still wrong even before line5/line6. Fix M11 path first."
        ),
        "if_zero_fails": (
            "Zero C-MT/relinearize/add contaminates full pipeline. Fix zero construction or relinearize/add path before continuing."
        ),
        "next_step": "WP4-X7i",
        "zh": "X7h 先把问题切成 M11 B-only 与 zero contamination 两块。",
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x7h_bonly_oracle_invariant_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X7h B-only oracle invariant probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
