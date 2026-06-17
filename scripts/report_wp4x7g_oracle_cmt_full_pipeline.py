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


def make_transparent_coeff_row_ct(rt, template_ct, row, moduli):
    towers_coeffs = []
    for q in moduli:
        towers_coeffs.append([signed_to_mod(v, q) for v in row])

    c0 = rt.import_1part_coeff_u64(template_ct, towers_coeffs)
    c1 = make_zero_1part(rt, template_ct, len(moduli))

    # OpenFHE component order:
    #   c0 = B
    #   c1 = A
    #
    # Transparent debug ciphertext:
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
    ), {
        "input_c0_row0_tower0": c0_export["rows"][0]["towers"][0]["coeffs_u64"],
        "input_c1_row0_tower0": c1_export["rows"][0]["towers"][0]["coeffs_u64"],
        "output_c0_row0_tower0": c0_T["rows"][0]["towers_coeffs"][0],
        "output_c1_row0_tower0": c1_T["rows"][0]["towers_coeffs"][0],
    }


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


def make_zero_1part_rows(rt, template_rows, num_towers):
    zeros = []
    zero_towers = [[] for _ in range(num_towers)]
    for tmpl in template_rows:
        zeros.append(rt.import_1part_coeff_u64(tmpl, zero_towers))
    return zeros


def make_temp_pair_bundle(rt, A_rows, B_rows, n, label, template_bundle):
    # Paper pair is (A,B).
    # OpenFHE component order is c0=B, c1=A.
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


def assemble_M10_M11_pair(rt, M10_rows, M11_rows, n):
    # Paper pair (M10,M11) maps to OpenFHE c0=B=M11, c1=A=M10.
    return [
        rt.assemble_2part_from_1parts_coeff_ct(M11_rows[i], M10_rows[i])
        for i in range(n)
    ]


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


def run_oracle_full_pipeline(rt, U, V):
    n = U.shape[0]

    U_bundle, U_template_info = make_transparent_rowwise_bundle(rt, U.tolist())
    V_bundle, V_template_info = make_transparent_rowwise_bundle(rt, V.tolist())

    # Line 1: oracle C-MT(U)
    U_cmt_bundle, U_cmt_trace = logical_n_cmt_oracle(rt, U_bundle, n, "oracle_CMT_U")

    # Line 2: raw RNS PP-MM
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

    zero_rows = make_zero_1part_rows(rt, U_cmt_bundle.rows, num_towers=M00["num_towers"])

    # Line 3: oracle C-MT((M01,0))
    T01_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M01_rows,
        B_rows=zero_rows,
        n=n,
        label="T01=(M01,0)",
        template_bundle=U_bundle,
    )
    T01_cmt_bundle, T01_cmt_trace = logical_n_cmt_oracle(rt, T01_bundle, n, "oracle_CMT_T01")

    # Line 4: oracle C-MT((M00,0))
    T00_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M00_rows,
        B_rows=zero_rows,
        n=n,
        label="T00=(M00,0)",
        template_bundle=U_bundle,
    )
    T00_cmt_bundle, T00_cmt_trace = logical_n_cmt_oracle(rt, T00_bundle, n, "oracle_CMT_T00")

    # Extract paper pair (A_hat,B_hat):
    # OpenFHE c0=B_hat, c1=A_hat.
    Bhat_rows, Bhat_export = extract_1part_rows(rt, T00_cmt_bundle.rows, part=0, coeff_count=0)
    Ahat_rows, Ahat_export = extract_1part_rows(rt, T00_cmt_bundle.rows, part=1, coeff_count=0)

    # Line 5
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

        # Paper pair (B_hat,0) -> OpenFHE c0=0, c1=B_hat.
        bhat_term = rt.assemble_2part_from_1parts_coeff_ct(
            zero_rows[i],
            Bhat_rows[i],
        )
        line5_bhat_terms.append(bhat_term)

        line5_rows.append(rt.add_coeff_ct(ks_output, bhat_term))

    # Line 6 pre-rescale
    M10_M11_pair_rows = assemble_M10_M11_pair(rt, M10_rows, M11_rows, n)

    line6_pre_rows = []
    for i in range(n):
        s1 = rt.add_coeff_ct(line5_rows[i], T01_cmt_bundle.rows[i])
        line6_pre_rows.append(rt.add_coeff_ct(s1, M10_M11_pair_rows[i]))

    final = decrypt_rows_after_compress(rt, line6_pre_rows, logical_length=n)

    trace = {
        "U_template_info": U_template_info,
        "V_template_info": V_template_info,
        "U_cmt_trace": U_cmt_trace,
        "T01_cmt_trace": T01_cmt_trace,
        "T00_cmt_trace": T00_cmt_trace,
        "Ahat_export": Ahat_export,
        "Bhat_export": Bhat_export,
        "line5_ks_inputs": inspect_rows(rt, line5_ks_inputs, coeff_sample=4),
        "line5_ks_outputs": inspect_rows(rt, line5_ks_outputs, coeff_sample=4),
        "line5_rows": inspect_rows(rt, line5_rows, coeff_sample=4),
        "line6_pre_rows": inspect_rows(rt, line6_pre_rows, coeff_sample=4),
        "raw_ppmm_moduli": M00["moduli_u64"],
    }

    return final, trace


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


def summarize_matrix_error(got, refs):
    if any(r is None for r in got):
        return {
            "all_rows_decrypted": False,
            "matches": {k: False for k in refs},
            "errors": {},
            "nonzero_positions_got": None,
        }

    G = np.array(got, dtype=np.int64)

    matches = {}
    errors = {}

    for name, ref in refs.items():
        D = G - ref
        matches[name] = bool(np.array_equal(G, ref))
        errors[name] = {
            "max_abs_err": int(np.max(np.abs(D))),
            "l1_err": int(np.sum(np.abs(D))),
            "diff": D.tolist(),
            "ref_nonzero_positions": nonzero_positions(ref.tolist()),
        }

    return {
        "all_rows_decrypted": True,
        "matches": matches,
        "errors": errors,
        "nonzero_positions_got": nonzero_positions(G.tolist()),
    }


def main():
    n = 4
    rng = np.random.default_rng(208500)

    cases_spec = [
        {
            "name": "basis_E00_times_E00",
            "U": one_hot(n, 0, 0),
            "V": one_hot(n, 0, 0),
            "zh": "理论 U@V = E00",
        },
        {
            "name": "basis_E01_times_E10",
            "U": one_hot(n, 0, 1),
            "V": one_hot(n, 1, 0),
            "zh": "理论 U@V = E00",
        },
        {
            "name": "basis_E01_times_E12",
            "U": one_hot(n, 0, 1),
            "V": one_hot(n, 1, 2),
            "zh": "理论 U@V = E02",
        },
        {
            "name": "basis_E23_times_E31",
            "U": one_hot(n, 2, 3),
            "V": one_hot(n, 3, 1),
            "zh": "理论 U@V = E21",
        },
        {
            "name": "identity_times_random_small",
            "U": np.eye(n, dtype=np.int64),
            "V": rng.integers(-2, 3, size=(n, n)).astype(np.int64),
            "zh": "I·V 应等于 V",
        },
        {
            "name": "random_small_times_identity",
            "U": rng.integers(-2, 3, size=(n, n)).astype(np.int64),
            "V": np.eye(n, dtype=np.int64),
            "zh": "U·I 应等于 U",
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
        "experiment": "wp4x7g_oracle_cmt_full_pipeline",
        "purpose": (
            "Use debug-only logical-n C-MT oracle inside the full Algorithm 3 pipeline. "
            "This isolates PP-MM/line5/line6 algebra from the known homomorphic C-MT dimension-contract failure."
        ),
        "status": "debug_oracle_pipeline_no_security_no_homomorphic_cmt",
        "n": n,
        "warning": {
            "value": "This uses transparent inputs and logical-n raw C-MT oracle. It is not secure and not a final CCMM implementation.",
            "zh": "本脚本使用 transparent ciphertext 与 raw oracle C-MT，只用于定位代数问题。",
        },
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for spec in cases_spec:
            U = spec["U"]
            V = spec["V"]

            final, trace = run_oracle_full_pipeline(rt, U, V)

            refs = {
                "U_matmul_V": U @ V,
                "U_transpose_matmul_V": U.T @ V,
                "U_matmul_V_transpose": U @ V.T,
                "V_matmul_U": V @ U,
                "V_transpose_matmul_U": V.T @ U,
            }

            analysis = summarize_matrix_error(final["decoded"], refs)

            case = {
                "name": spec["name"],
                "zh": spec["zh"],
                "input_U": U.tolist(),
                "input_V": V.tolist(),
                "references": {k: v.tolist() for k, v in refs.items()},
                "final": final,
                "trace": trace,
                "analysis": analysis,
            }

            report["cases"].append(case)

    match_counts = {}
    for ref_name in [
        "U_matmul_V",
        "U_transpose_matmul_V",
        "U_matmul_V_transpose",
        "V_matmul_U",
        "V_transpose_matmul_U",
    ]:
        match_counts[ref_name] = sum(
            1 for c in report["cases"]
            if c["analysis"]["matches"].get(ref_name) is True
        )

    all_decrypted = all(
        c["analysis"]["all_rows_decrypted"]
        for c in report["cases"]
    )

    all_line5_ks_inputs_3part = all(
        rows_are_npart(c["trace"]["line5_ks_inputs"], n, 3)
        for c in report["cases"]
    )
    all_line5_rows_2part = all(
        rows_are_npart(c["trace"]["line5_rows"], n, 2)
        for c in report["cases"]
    )
    all_line6_pre_rows_2part = all(
        rows_are_npart(c["trace"]["line6_pre_rows"], n, 2)
        for c in report["cases"]
    )

    report["checks"] = {
        "all_cases_decrypted（所有 oracle pipeline case 都成功解密）": all_decrypted,
        "all_line5_ks_inputs_are_3part（所有 line5 KS 输入都是三分量）": all_line5_ks_inputs_3part,
        "all_line5_rows_are_2part（所有 line5 输出都是二分量）": all_line5_rows_2part,
        "all_line6_pre_rows_are_2part（所有 line6 pre-rescale 输出都是二分量）": all_line6_pre_rows_2part,
        "match_counts_by_reference（与不同参考矩阵匹配的 case 数）": match_counts,
        "oracle_pipeline_matches_U_matmul_V_all_cases（oracle pipeline 是否全部等于 U@V）": match_counts["U_matmul_V"] == len(report["cases"]),
    }

    report["summary"] = {
        "if_matches_U_matmul_V": (
            "Downstream Algorithm 3 algebra is correct under logical-n C-MT. The remaining task is a real logical-n homomorphic C-MT or dimension-compatible benchmark."
        ),
        "if_matches_another_reference": (
            "The pipeline is algebraically consistent but has orientation convention mismatch. Promote the matching reference to guide fixes."
        ),
        "if_matches_none": (
            "Even with logical-n C-MT oracle, PP-MM/line5/line6 algebra is still wrong. Next step: symbolic plaintext Algorithm 3 simulation for line5/line6."
        ),
        "next_step": "WP4-X7h",
        "zh": "X7g 判断：排除 C-MT 维度问题后，后续 Algorithm 3 代数是否正确。",
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x7g_oracle_cmt_full_pipeline_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X7g oracle C-MT full Algorithm 3 pipeline")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
