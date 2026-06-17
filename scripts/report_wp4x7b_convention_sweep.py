import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    cmt_algorithm2_rowwise,
)


class TempRowwiseBundle:
    def __init__(self, rows, shape, label, template_bundle=None, ring_dim=None):
        self.rows = rows
        self.shape = shape
        self.orientation = "rowwise"
        self.label = label

        if ring_dim is not None:
            self.ring_dim = int(ring_dim)
        elif template_bundle is not None and hasattr(template_bundle, "ring_dim"):
            self.ring_dim = int(template_bundle.ring_dim)
        else:
            self.ring_dim = 1 << 14

        if template_bundle is not None and hasattr(template_bundle, "batch_size"):
            self.batch_size = int(template_bundle.batch_size)
        else:
            self.batch_size = int(shape[1])


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


def maybe_transpose(M, do_transpose):
    if not do_transpose:
        return M
    n = len(M)
    return [[M[j][i] for j in range(n)] for i in range(n)]


def matrix_for_tower(export_obj, tower_index, n):
    rows = []
    for row in export_obj["rows"]:
        coeffs = row["towers"][tower_index]["coeffs_u64"][:n]
        rows.append([int(v) for v in coeffs])
    return rows


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


def raw_rns_ppmm_variant(left_export, right_export, n, left_T=False, right_T=False):
    if left_export["num_rows"] != n or right_export["num_rows"] != n:
        raise ValueError("raw_rns_ppmm_variant expects n rows in both inputs")
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
        A = maybe_transpose(A, left_T)
        B = maybe_transpose(B, right_T)
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


def assemble_paper_pair(rt, A_rows, B_rows, mapping):
    """
    Paper pair is (A, B).

    normal:
      OpenFHE c0=B, c1=A

    swapped:
      OpenFHE c0=A, c1=B
    """
    rows = []
    for A, B in zip(A_rows, B_rows):
        if mapping == "normal_c0B_c1A":
            rows.append(rt.assemble_2part_from_1parts_coeff_ct(B, A))
        elif mapping == "swapped_c0A_c1B":
            rows.append(rt.assemble_2part_from_1parts_coeff_ct(A, B))
        else:
            raise ValueError(f"unknown pair mapping: {mapping}")
    return rows


def make_temp_pair_bundle(rt, A_rows, B_rows, n, label, mapping, template_bundle):
    rows = assemble_paper_pair(rt, A_rows=A_rows, B_rows=B_rows, mapping=mapping)
    return TempRowwiseBundle(
        rows=rows,
        shape=(n, n),
        label=label,
        template_bundle=template_bundle,
    )


def extract_paper_pair_components(rt, source_rows, mapping, coeff_count=0):
    """
    Returns (A_rows, B_rows) from OpenFHE rows according to the mapping convention.
    """
    if mapping == "normal_c0B_c1A":
        # c0=B, c1=A
        B_part = 0
        A_part = 1
    elif mapping == "swapped_c0A_c1B":
        # c0=A, c1=B
        A_part = 0
        B_part = 1
    else:
        raise ValueError(f"unknown pair mapping: {mapping}")

    A_rows = extract_1part_rows(rt, source_rows, part=A_part, coeff_count=coeff_count)
    B_rows = extract_1part_rows(rt, source_rows, part=B_part, coeff_count=coeff_count)

    return A_rows, B_rows, {
        "mapping": mapping,
        "A_part": A_part,
        "B_part": B_part,
    }


def extract_1part_rows(rt, source_rows, part, coeff_count=0):
    exported = normalize_export(
        rt.export_component_coeff_matrix_u64(
            source_rows,
            part=part,
            coeff_count=coeff_count,
        )
    )

    out = []
    for i, row in enumerate(exported["rows"]):
        towers_coeffs = [tower["coeffs_u64"] for tower in row["towers"]]
        out.append(rt.import_1part_coeff_u64(source_rows[i], towers_coeffs))

    return out


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


def run_variant_pipeline(rt, U, V, variant):
    n = U.shape[0]
    coeff_count = n

    U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
    V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

    # Line 1: C-MT(U)
    U_cmt, U_cmt_trace = cmt_algorithm2_rowwise(rt, U_bundle)

    # Export paper sources using the base convention from the real ciphertexts:
    # OpenFHE ciphertexts are known to have c0=B, c1=A for actual encrypted rows.
    A_U = normalize_export(rt.export_component_coeff_matrix_u64(U_cmt, part=1, coeff_count=coeff_count))
    B_U = normalize_export(rt.export_component_coeff_matrix_u64(U_cmt, part=0, coeff_count=coeff_count))
    A_V = normalize_export(rt.export_component_coeff_matrix_u64(V_bundle.rows, part=1, coeff_count=coeff_count))
    B_V = normalize_export(rt.export_component_coeff_matrix_u64(V_bundle.rows, part=0, coeff_count=coeff_count))

    left_T = bool(variant["ppmm_left_T"])
    right_T = bool(variant["ppmm_right_T"])

    # Line 2: PP-MM with orientation variant
    M00 = raw_rns_ppmm_variant(A_U, A_V, n, left_T=left_T, right_T=right_T)
    M01 = raw_rns_ppmm_variant(A_U, B_V, n, left_T=left_T, right_T=right_T)
    M10 = raw_rns_ppmm_variant(B_U, A_V, n, left_T=left_T, right_T=right_T)
    M11 = raw_rns_ppmm_variant(B_U, B_V, n, left_T=left_T, right_T=right_T)

    M00_rows = import_ppmm_rows(rt, U_cmt, M00)
    M01_rows = import_ppmm_rows(rt, U_cmt, M01)
    M10_rows = import_ppmm_rows(rt, U_cmt, M10)
    M11_rows = import_ppmm_rows(rt, U_cmt, M11)

    zero_rows = make_zero_1part_rows(rt, U_cmt, num_towers=M00["num_towers"])

    pair_mapping = variant["temp_pair_mapping"]
    line6_mapping = variant["line6_pair_mapping"]

    # Line 3: Transpose((M01,0))
    T01_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M01_rows,
        B_rows=zero_rows,
        n=n,
        label="T01=(M01,0)",
        mapping=pair_mapping,
        template_bundle=U_bundle,
    )
    T01_cmt_rows, T01_cmt_trace = cmt_algorithm2_rowwise(rt, T01_bundle)

    # Line 4: Transpose((M00,0))
    T00_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M00_rows,
        B_rows=zero_rows,
        n=n,
        label="T00=(M00,0)",
        mapping=pair_mapping,
        template_bundle=U_bundle,
    )
    T00_cmt_rows, T00_cmt_trace = cmt_algorithm2_rowwise(rt, T00_bundle)

    # Extract paper (A_hat,B_hat) according to temp pair mapping
    Ahat_rows, Bhat_rows, extract_trace = extract_paper_pair_components(
        rt,
        T00_cmt_rows,
        mapping=pair_mapping,
        coeff_count=0,
    )

    # Line 5:
    # KS_{s²->s}((A_hat,0)) + (B_hat,0)
    line5_rows = []
    for i in range(n):
        ks_input = rt.assemble_3part_from_1parts_coeff_ct(
            zero_rows[i],
            zero_rows[i],
            Ahat_rows[i],
        )
        ks_output = rt.relinearize_coeff_ct(ks_input)

        # paper pair (B_hat,0)
        bhat_term_rows = assemble_paper_pair(
            rt,
            A_rows=[Bhat_rows[i]],
            B_rows=[zero_rows[i]],
            mapping=pair_mapping,
        )
        bhat_term = bhat_term_rows[0]

        line5_rows.append(rt.add_coeff_ct(ks_output, bhat_term))

    # Line 6:
    # W_pre = line5 + Transpose((M01,0)) + (M10,M11)
    M10_M11_pair_rows = assemble_paper_pair(
        rt,
        A_rows=M10_rows,
        B_rows=M11_rows,
        mapping=line6_mapping,
    )

    line6_pre_rows = []
    for i in range(n):
        s1 = rt.add_coeff_ct(line5_rows[i], T01_cmt_rows[i])
        line6_pre_rows.append(rt.add_coeff_ct(s1, M10_M11_pair_rows[i]))

    final = decrypt_after_compress(rt, line6_pre_rows, logical_length=n)

    trace = {
        "U_cmt_auto_alphas": [x["alpha"] for x in U_cmt_trace["auto"]],
        "T01_cmt_auto_alphas": [x["alpha"] for x in T01_cmt_trace["auto"]],
        "T00_cmt_auto_alphas": [x["alpha"] for x in T00_cmt_trace["auto"]],
        "extract_Ahat_Bhat": extract_trace,
        "line6_pre_summary": inspect_rows(rt, line6_pre_rows, coeff_sample=4),
    }

    return final, trace


def decrypt_after_compress(rt, rows, logical_length):
    compressed_rows = []
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
        compressed_rows.append(comp["value"] if comp["ok"] else None)

    decoded = []
    decrypt_rows = []

    for i, ct in enumerate(compressed_rows):
        if ct is None:
            decoded.append(None)
            decrypt_rows.append({
                "row_index": i,
                "decrypt": {"ok": False, "error": "compress failed"},
                "value": None,
            })
            continue

        dec = safe_call(
            f"decrypt_row_{i}",
            lambda ct=ct: rt.decrypt_coeff_row_i64(ct, logical_length=logical_length),
        )

        decrypt_rows.append({
            "row_index": i,
            "decrypt": {
                "ok": dec["ok"],
                "error": dec.get("error"),
            },
            "value": dec.get("value") if dec["ok"] else None,
        })

        decoded.append(list(map(int, dec["value"])) if dec["ok"] else None)

    return {
        "compress_reports": compress_reports,
        "decrypt_rows": decrypt_rows,
        "decoded_matrix": decoded,
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


def summarize_matrix_error(got, ref):
    if any(r is None for r in got):
        return {
            "all_rows_decrypted": False,
            "exact": False,
            "max_abs_err": None,
            "l1_err": None,
            "nonzero_positions_got": None,
            "nonzero_positions_ref": nonzero_positions(ref.tolist()),
        }

    G = np.array(got, dtype=np.int64)
    D = G - ref

    return {
        "all_rows_decrypted": True,
        "exact": bool(np.array_equal(G, ref)),
        "max_abs_err": int(np.max(np.abs(D))),
        "l1_err": int(np.sum(np.abs(D))),
        "nonzero_positions_got": nonzero_positions(G.tolist()),
        "nonzero_positions_ref": nonzero_positions(ref.tolist()),
    }


def variant_id(v):
    return (
        f"L{'T' if v['ppmm_left_T'] else 'I'}"
        f"_R{'T' if v['ppmm_right_T'] else 'I'}"
        f"_pair-{v['temp_pair_mapping']}"
        f"_line6-{v['line6_pair_mapping']}"
    )


def main():
    n = 4

    cases_spec = [
        {
            "name": "basis_E00_times_E00",
            "U": one_hot(n, 0, 0),
            "V": one_hot(n, 0, 0),
            "zh": "理论输出 E00",
        },
        {
            "name": "basis_E01_times_E10",
            "U": one_hot(n, 0, 1),
            "V": one_hot(n, 1, 0),
            "zh": "理论输出 E00",
        },
        {
            "name": "basis_E01_times_E12",
            "U": one_hot(n, 0, 1),
            "V": one_hot(n, 1, 2),
            "zh": "理论输出 E02",
        },
        {
            "name": "basis_E23_times_E31",
            "U": one_hot(n, 2, 3),
            "V": one_hot(n, 3, 1),
            "zh": "理论输出 E21",
        },
    ]

    variants = []
    for left_T in [False, True]:
        for right_T in [False, True]:
            for pair_mapping in ["normal_c0B_c1A", "swapped_c0A_c1B"]:
                for line6_mapping in ["normal_c0B_c1A", "swapped_c0A_c1B"]:
                    v = {
                        "ppmm_left_T": left_T,
                        "ppmm_right_T": right_T,
                        "temp_pair_mapping": pair_mapping,
                        "line6_pair_mapping": line6_mapping,
                    }
                    v["id"] = variant_id(v)
                    variants.append(v)

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
        "experiment": "wp4x7b_convention_sweep",
        "purpose": (
            "Sweep PP-MM orientation and component order conventions for the full paper-shaped "
            "Algorithm 3 pipeline on one-hot cases."
        ),
        "status": "diagnostic_convention_sweep_no_code_promotion",
        "n": n,
        "sweep_dimensions": {
            "ppmm_left_T": [False, True],
            "ppmm_right_T": [False, True],
            "temp_pair_mapping": ["normal_c0B_c1A", "swapped_c0A_c1B"],
            "line6_pair_mapping": ["normal_c0B_c1A", "swapped_c0A_c1B"],
        },
        "cases": [
            {
                "name": c["name"],
                "zh": c["zh"],
                "input_U": c["U"].tolist(),
                "input_V": c["V"].tolist(),
                "reference_U_matmul_V": (c["U"] @ c["V"]).tolist(),
            }
            for c in cases_spec
        ],
        "variant_results": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for variant in variants:
            case_results = []

            exact_count = 0
            all_decrypted_count = 0
            total_l1_err = 0
            one_hot_got_nnz_total = 0

            for spec in cases_spec:
                U = spec["U"]
                V = spec["V"]
                W_ref = U @ V

                final, trace = run_variant_pipeline(rt, U, V, variant)
                analysis = summarize_matrix_error(final["decoded_matrix"], W_ref)

                if analysis["all_rows_decrypted"]:
                    all_decrypted_count += 1
                if analysis["exact"]:
                    exact_count += 1
                if analysis["l1_err"] is not None:
                    total_l1_err += int(analysis["l1_err"])

                got_nz = analysis["nonzero_positions_got"]
                if got_nz is not None:
                    one_hot_got_nnz_total += len(got_nz)
                else:
                    one_hot_got_nnz_total += 10**9

                case_results.append({
                    "name": spec["name"],
                    "reference": W_ref.tolist(),
                    "decoded": final["decoded_matrix"],
                    "analysis": analysis,
                    "trace_summary": {
                        "U_cmt_auto_alphas": trace["U_cmt_auto_alphas"],
                        "T01_cmt_auto_alphas": trace["T01_cmt_auto_alphas"],
                        "T00_cmt_auto_alphas": trace["T00_cmt_auto_alphas"],
                        "extract_Ahat_Bhat": trace["extract_Ahat_Bhat"],
                        "line6_pre_parts": trace["line6_pre_summary"]["expected_parts"],
                    },
                })

            score = {
                "exact_count": exact_count,
                "all_decrypted_count": all_decrypted_count,
                "total_l1_err": total_l1_err,
                "one_hot_got_nnz_total": one_hot_got_nnz_total,
            }

            report["variant_results"].append({
                "variant": variant,
                "score": score,
                "case_results": case_results,
            })

    ranked = sorted(
        report["variant_results"],
        key=lambda x: (
            -x["score"]["exact_count"],
            x["score"]["total_l1_err"],
            x["score"]["one_hot_got_nnz_total"],
        ),
    )

    report["ranking"] = [
        {
            "rank": i + 1,
            "variant": r["variant"],
            "score": r["score"],
        }
        for i, r in enumerate(ranked)
    ]

    best = ranked[0]

    report["checks"] = {
        "variants_tested（扫描的约定组合数量）": len(variants),
        "cases_per_variant（每个约定组合测试的 one-hot case 数量）": len(cases_spec),
        "best_exact_count（最佳组合完全正确的 case 数量）": best["score"]["exact_count"],
        "best_all_cases_decrypted_count（最佳组合成功解密 case 数量）": best["score"]["all_decrypted_count"],
        "any_variant_solves_all_one_hot（是否存在完全通过 4 个 one-hot 的组合）": best["score"]["exact_count"] == len(cases_spec),
        "best_variant_id（当前最佳组合 id）": best["variant"]["id"],
    }

    report["summary"] = {
        "best_variant": best["variant"],
        "best_score": best["score"],
        "next_if_best_exact_4": "Promote this convention into X8 full identity/random diagnostics.",
        "next_if_best_exact_0": (
            "The error is likely not a simple transpose/component-order convention. "
            "Move to X7c plaintext-symbolic Algorithm 3 simulation with the paper's exact C-MT definitions."
        ),
        "zh": "如果扫描没有任何 one-hot 通过，说明问题不是这三类简单约定，需要回到论文 C-MT/PP-MM 的明文符号模拟。",
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x7b_convention_sweep_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X7b convention sweep")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
