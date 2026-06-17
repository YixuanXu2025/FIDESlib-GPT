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


def ppmm_is_zero(ppmm_result):
    for row in ppmm_result["rows"]:
        for tower_coeffs in row["towers_coeffs"]:
            for v in tower_coeffs:
                if int(v) != 0:
                    return False
    return True


def import_ppmm_rows(rt, template_rows, ppmm_result):
    out = []
    for i, row in enumerate(ppmm_result["rows"]):
        out.append(rt.import_1part_coeff_u64(template_rows[i], row["towers_coeffs"]))
    return out


def make_temp_pair_bundle(rt, A_rows, B_rows, n, label, template_bundle):
    # Paper pair is (A,B). OpenFHE component order is c0=B, c1=A.
    rows = [
        rt.assemble_2part_from_1parts_coeff_ct(B_rows[i], A_rows[i])
        for i in range(n)
    ]

    return TempRowwiseBundle(
        rows=rows,
        shape=(n, n),
        label=label,
        ring_dim=template_bundle.ring_dim,
        batch_size=template_bundle.batch_size,
    )


def extract_1part_rows(rt, source_rows, part, coeff_count=0):
    exported = normalize_export(
        rt.export_component_coeff_matrix_u64(source_rows, part=part, coeff_count=coeff_count)
    )

    out = []
    for i, row in enumerate(exported["rows"]):
        towers_coeffs = [tower["coeffs_u64"] for tower in row["towers"]]
        out.append(rt.import_1part_coeff_u64(source_rows[i], towers_coeffs))

    return out, {
        "part": part,
        "num_rows": exported["num_rows"],
        "num_towers": exported["num_towers"],
        "ring_dim": exported["ring_dim"],
        "coeff_count": exported["coeff_count"],
        "row0_tower0_coeff_head": exported["rows"][0]["towers"][0]["coeffs_u64"][:8],
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

    # B-only output: paper (A=0,B=M11) -> OpenFHE c0=M11,c1=0.
    bonly_rows = [
        rt.assemble_2part_from_1parts_coeff_ct(M11_rows[i], zero_rows[i])
        for i in range(n)
    ]

    # Full pipeline terms.
    T01_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M01_rows,
        B_rows=zero_rows,
        n=n,
        label="T01=(M01,0)",
        template_bundle=U_bundle,
    )
    T01_cmt_bundle = logical_n_cmt_oracle(rt, T01_bundle, n, "oracle_CMT_T01")

    T00_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M00_rows,
        B_rows=zero_rows,
        n=n,
        label="T00=(M00,0)",
        template_bundle=U_bundle,
    )
    T00_cmt_bundle = logical_n_cmt_oracle(rt, T00_bundle, n, "oracle_CMT_T00")

    # T00_cmt paper pair (A_hat,B_hat): OpenFHE c0=B_hat, c1=A_hat.
    Bhat_rows, Bhat_export = extract_1part_rows(rt, T00_cmt_bundle.rows, part=0, coeff_count=0)
    Ahat_rows, Ahat_export = extract_1part_rows(rt, T00_cmt_bundle.rows, part=1, coeff_count=0)

    line5_ks_inputs = []
    line5_ks_outputs = []
    line5_bhat_terms = []
    line5_rows = []

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

    # paper (M10,M11) -> OpenFHE c0=M11,c1=M10.
    M10_M11_pair_rows = [
        rt.assemble_2part_from_1parts_coeff_ct(M11_rows[i], M10_rows[i])
        for i in range(n)
    ]

    full_rows = []
    for i in range(n):
        s1 = rt.add_coeff_ct(line5_rows[i], T01_cmt_bundle.rows[i])
        full_rows.append(rt.add_coeff_ct(s1, M10_M11_pair_rows[i]))

    dec_bonly = decrypt_rows_after_compress(rt, bonly_rows, logical_length=n)
    dec_T01 = decrypt_rows_after_compress(rt, T01_cmt_bundle.rows, logical_length=n)
    dec_T00 = decrypt_rows_after_compress(rt, T00_cmt_bundle.rows, logical_length=n)
    dec_line5 = decrypt_rows_after_compress(rt, line5_rows, logical_length=n)
    dec_M10_M11 = decrypt_rows_after_compress(rt, M10_M11_pair_rows, logical_length=n)
    dec_full = decrypt_rows_after_compress(rt, full_rows, logical_length=n)

    expected_bonly = U.T @ V
    zero_ref = np.zeros((n, n), dtype=np.int64)

    return {
        "ppmm_zero_checks": {
            "M00_is_raw_zero": ppmm_is_zero(M00),
            "M01_is_raw_zero": ppmm_is_zero(M01),
            "M10_is_raw_zero": ppmm_is_zero(M10),
            "M11_is_raw_zero": ppmm_is_zero(M11),
        },
        "exports": {
            "A_U_row0_tower0": A_U["rows"][0]["towers"][0]["coeffs_u64"],
            "B_U_row0_tower0": B_U["rows"][0]["towers"][0]["coeffs_u64"],
            "A_V_row0_tower0": A_V["rows"][0]["towers"][0]["coeffs_u64"],
            "B_V_row0_tower0": B_V["rows"][0]["towers"][0]["coeffs_u64"],
            "Ahat_export": Ahat_export,
            "Bhat_export": Bhat_export,
        },
        "decoded_terms": {
            "bonly": dec_bonly,
            "T01_cmt": dec_T01,
            "T00_cmt": dec_T00,
            "line5": dec_line5,
            "M10_M11_pair": dec_M10_M11,
            "full": dec_full,
        },
        "term_comparisons": {
            "bonly_equals_U_T_matmul_V": compare_matrix(dec_bonly["decoded"], expected_bonly),
            "T01_cmt_is_zero": compare_matrix(dec_T01["decoded"], zero_ref),
            "T00_cmt_is_zero": compare_matrix(dec_T00["decoded"], zero_ref),
            "line5_is_zero": compare_matrix(dec_line5["decoded"], zero_ref),
            "M10_M11_pair_equals_bonly": compare_matrix(dec_M10_M11["decoded"], expected_bonly),
            "full_equals_bonly": compare_matrix(dec_full["decoded"], np.array(dec_bonly["decoded"], dtype=np.int64)),
            "full_equals_U_T_matmul_V": compare_matrix(dec_full["decoded"], expected_bonly),
        },
        "shape_summaries": {
            "line5_ks_inputs": inspect_rows(rt, line5_ks_inputs, coeff_sample=4),
            "line5_ks_outputs": inspect_rows(rt, line5_ks_outputs, coeff_sample=4),
            "line5_rows": inspect_rows(rt, line5_rows, coeff_sample=4),
            "full_rows": inspect_rows(rt, full_rows, coeff_sample=4),
        },
    }


def main():
    n = 4
    rng = np.random.default_rng(208700)

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
        "experiment": "wp4x7i_full_vs_bonly_term_isolation",
        "purpose": (
            "For transparent inputs under logical-n C-MT oracle, isolate why the full Algorithm 3 pipeline "
            "does not reduce to the B-only M11 path. M00/M01/M10 should be zero, line5/T01 should be zero, "
            "and full output should equal M11 = U.T @ V."
        ),
        "status": "debug_oracle_term_isolation_no_security_no_homomorphic_cmt",
        "n": n,
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for spec in cases_spec:
            U = spec["U"]
            V = spec["V"]
            result = run_case(rt, U, V)

            report["cases"].append({
                "name": spec["name"],
                "input_U": U.tolist(),
                "input_V": V.tolist(),
                "expected_bonly_U_T_matmul_V": (U.T @ V).tolist(),
                "result": result,
            })

    def all_case(path):
        cur = report["cases"]
        for key in path:
            cur = [c["result"][key] if key in c["result"] else c[key] for c in cur]
        return all(cur)

    report["checks"] = {
        "M00_all_raw_zero（transparent 输入下 M00 全部为 raw zero）": all(
            c["result"]["ppmm_zero_checks"]["M00_is_raw_zero"] for c in report["cases"]
        ),
        "M01_all_raw_zero（transparent 输入下 M01 全部为 raw zero）": all(
            c["result"]["ppmm_zero_checks"]["M01_is_raw_zero"] for c in report["cases"]
        ),
        "M10_all_raw_zero（transparent 输入下 M10 全部为 raw zero）": all(
            c["result"]["ppmm_zero_checks"]["M10_is_raw_zero"] for c in report["cases"]
        ),
        "M11_not_all_raw_zero（至少部分 M11 非零）": any(
            not c["result"]["ppmm_zero_checks"]["M11_is_raw_zero"] for c in report["cases"]
        ),

        "bonly_all_equals_U_T_matmul_V（B-only 全部等于 U.T@V）": all(
            c["result"]["term_comparisons"]["bonly_equals_U_T_matmul_V"]["exact"]
            for c in report["cases"]
        ),
        "T01_cmt_all_zero（Transpose((M01,0)) 全部解密为 0）": all(
            c["result"]["term_comparisons"]["T01_cmt_is_zero"]["exact"]
            for c in report["cases"]
        ),
        "T00_cmt_all_zero（Transpose((M00,0)) 全部解密为 0）": all(
            c["result"]["term_comparisons"]["T00_cmt_is_zero"]["exact"]
            for c in report["cases"]
        ),
        "line5_all_zero（line5 全部解密为 0）": all(
            c["result"]["term_comparisons"]["line5_is_zero"]["exact"]
            for c in report["cases"]
        ),
        "M10_M11_pair_all_equals_bonly（line6 的 (M10,M11) 项全部等于 B-only）": all(
            c["result"]["term_comparisons"]["M10_M11_pair_equals_bonly"]["exact"]
            for c in report["cases"]
        ),
        "full_all_equals_bonly（full pipeline 全部等于 B-only）": all(
            c["result"]["term_comparisons"]["full_equals_bonly"]["exact"]
            for c in report["cases"]
        ),
        "full_all_equals_U_T_matmul_V（full pipeline 全部等于 U.T@V）": all(
            c["result"]["term_comparisons"]["full_equals_U_T_matmul_V"]["exact"]
            for c in report["cases"]
        ),
        "all_line5_ks_inputs_are_3part（line5 KS inputs 均为三分量）": all(
            rows_are_npart(c["result"]["shape_summaries"]["line5_ks_inputs"], n, 3)
            for c in report["cases"]
        ),
        "all_line5_rows_are_2part（line5 rows 均为二分量）": all(
            rows_are_npart(c["result"]["shape_summaries"]["line5_rows"], n, 2)
            for c in report["cases"]
        ),
        "all_full_rows_are_2part（full rows 均为二分量）": all(
            rows_are_npart(c["result"]["shape_summaries"]["full_rows"], n, 2)
            for c in report["cases"]
        ),
    }

    report["summary"] = {
        "if_all_true": (
            "The oracle full pipeline is internally consistent for transparent inputs and equals U.T@V. "
            "X7g mismatch was due to reference expectation, not implementation."
        ),
        "if_zero_ppmm_false": "A_U/A_V are not actually zero; inspect transparent component extraction.",
        "if_T01_or_T00_false": "Oracle C-MT or zero temporary bundle construction is not preserving zero in the actual full path.",
        "if_line5_false": "Relinearize or Bhat placement introduces nonzero from zero Ahat/Bhat.",
        "if_full_not_bonly_but_terms_zero": "Add path or M10/M11 pair assembly is wrong.",
        "next_step": "WP4-X7j",
        "zh": "X7i 会定位 X7g full pipeline 与 B-only 的差异来自哪个项。",
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x7i_full_vs_bonly_term_isolation_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X7i full-vs-B-only term isolation")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
