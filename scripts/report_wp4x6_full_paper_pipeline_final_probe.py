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
        ct = rt.import_1part_coeff_u64(template_rows[i], row["towers_coeffs"])
        out.append(ct)
    return out


def make_zero_1part_rows(rt, template_rows, num_towers):
    zeros = []
    zero_towers = [[] for _ in range(num_towers)]
    for tmpl in template_rows:
        zeros.append(rt.import_1part_coeff_u64(tmpl, zero_towers))
    return zeros


def make_temp_pair_bundle(rt, onepart_A_rows, zero_B_rows, n, label, template_bundle=None):
    # Paper pair is (A,B), OpenFHE component order is c0=B, c1=A.
    rows = []
    for A, Bzero in zip(onepart_A_rows, zero_B_rows):
        rows.append(rt.assemble_2part_from_1parts_coeff_ct(Bzero, A))

    return TempRowwiseBundle(
        rows=rows,
        shape=(n, n),
        label=label,
        template_bundle=template_bundle,
    )


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

    return out, {
        "num_rows": exported["num_rows"],
        "part": exported["part"],
        "num_towers": exported["num_towers"],
        "ring_dim": exported["ring_dim"],
        "coeff_count": exported["coeff_count"],
        "moduli_u64": exported["moduli_u64"],
        "row0_tower0_coeff_head": exported["rows"][0]["towers"][0]["coeffs_u64"][:8],
    }


def assemble_M10_M11_pair(rt, M10_rows, M11_rows, n):
    # Paper pair (M10,M11) maps to OpenFHE c0=B=M11, c1=A=M10.
    return [
        rt.assemble_2part_from_1parts_coeff_ct(M11_rows[i], M10_rows[i])
        for i in range(n)
    ]


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


def run_paper_pipeline_pre_rows(rt, U, V):
    n = U.shape[0]
    coeff_count = n

    U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
    V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

    # Line 1: C-MT(U)
    U_cmt, U_cmt_trace = cmt_algorithm2_rowwise(rt, U_bundle)

    # Line 2: raw RNS PP-MM
    A_U = normalize_export(rt.export_component_coeff_matrix_u64(U_cmt, part=1, coeff_count=coeff_count))
    B_U = normalize_export(rt.export_component_coeff_matrix_u64(U_cmt, part=0, coeff_count=coeff_count))
    A_V = normalize_export(rt.export_component_coeff_matrix_u64(V_bundle.rows, part=1, coeff_count=coeff_count))
    B_V = normalize_export(rt.export_component_coeff_matrix_u64(V_bundle.rows, part=0, coeff_count=coeff_count))

    M00 = raw_rns_ppmm(A_U, A_V, n)
    M01 = raw_rns_ppmm(A_U, B_V, n)
    M10 = raw_rns_ppmm(B_U, A_V, n)
    M11 = raw_rns_ppmm(B_U, B_V, n)

    M00_rows = import_ppmm_rows(rt, U_cmt, M00)
    M01_rows = import_ppmm_rows(rt, U_cmt, M01)
    M10_rows = import_ppmm_rows(rt, U_cmt, M10)
    M11_rows = import_ppmm_rows(rt, U_cmt, M11)

    zero_rows = make_zero_1part_rows(rt, U_cmt, num_towers=M00["num_towers"])

    # Line 3: Transpose((M01,0))
    T01_bundle = make_temp_pair_bundle(
        rt,
        onepart_A_rows=M01_rows,
        zero_B_rows=zero_rows,
        n=n,
        label="T01=(M01,0)",
        template_bundle=U_bundle,
    )
    T01_cmt_rows, T01_cmt_trace = cmt_algorithm2_rowwise(rt, T01_bundle)

    # Line 4: Transpose((M00,0))
    T00_bundle = make_temp_pair_bundle(
        rt,
        onepart_A_rows=M00_rows,
        zero_B_rows=zero_rows,
        n=n,
        label="T00=(M00,0)",
        template_bundle=U_bundle,
    )
    T00_cmt_rows, T00_cmt_trace = cmt_algorithm2_rowwise(rt, T00_bundle)

    # Extract line4 result:
    # T00_cmt paper pair (A_hat,B_hat), OpenFHE c0=B_hat, c1=A_hat.
    Bhat_rows, Bhat_export_summary = extract_1part_rows(rt, T00_cmt_rows, part=0, coeff_count=0)
    Ahat_rows, Ahat_export_summary = extract_1part_rows(rt, T00_cmt_rows, part=1, coeff_count=0)

    # Line 5:
    line5_ks_inputs = []
    line5_ks_outputs = []
    line5_bhat_terms = []
    line5_rows = []

    for i in range(n):
        # c0=0, c1=0, c2=A_hat
        ks_input = rt.assemble_3part_from_1parts_coeff_ct(
            zero_rows[i],
            zero_rows[i],
            Ahat_rows[i],
        )
        line5_ks_inputs.append(ks_input)

        ks_output = rt.relinearize_coeff_ct(ks_input)
        line5_ks_outputs.append(ks_output)

        # paper (B_hat,0) -> OpenFHE c0=0, c1=B_hat
        bhat_term = rt.assemble_2part_from_1parts_coeff_ct(
            zero_rows[i],
            Bhat_rows[i],
        )
        line5_bhat_terms.append(bhat_term)

        line5_rows.append(rt.add_coeff_ct(ks_output, bhat_term))

    # Line 6 pre-rescale:
    M10_M11_pair_rows = assemble_M10_M11_pair(rt, M10_rows, M11_rows, n)

    line6_sum1_rows = []
    line6_pre_rows = []

    for i in range(n):
        s1 = rt.add_coeff_ct(line5_rows[i], T01_cmt_rows[i])
        line6_sum1_rows.append(s1)

        final_pre = rt.add_coeff_ct(s1, M10_M11_pair_rows[i])
        line6_pre_rows.append(final_pre)

    trace = {
        "U_cmt_auto_alphas": [x["alpha"] for x in U_cmt_trace["auto"]],
        "T01_cmt_auto_alphas": [x["alpha"] for x in T01_cmt_trace["auto"]],
        "T00_cmt_auto_alphas": [x["alpha"] for x in T00_cmt_trace["auto"]],
        "Ahat_export_summary": Ahat_export_summary,
        "Bhat_export_summary": Bhat_export_summary,
        "line5_ks_inputs": inspect_rows(rt, line5_ks_inputs, coeff_sample=4),
        "line5_ks_outputs": inspect_rows(rt, line5_ks_outputs, coeff_sample=4),
        "line5_rows": inspect_rows(rt, line5_rows, coeff_sample=4),
        "line6_pre_rows": inspect_rows(rt, line6_pre_rows, coeff_sample=4),
        "raw_ppmm_moduli": M00["moduli_u64"],
    }

    return line6_pre_rows, trace


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

        if comp["ok"]:
            compressed_rows.append(comp["value"])
        else:
            compressed_rows.append(None)

    decrypt_rows = []
    decoded = []

    for i, ct in enumerate(compressed_rows):
        if ct is None:
            decrypt_rows.append({
                "row_index": i,
                "decrypt": {"ok": False, "error": "compress failed"},
            })
            decoded.append(None)
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

        if dec["ok"]:
            decoded.append(list(map(int, dec["value"])))
        else:
            decoded.append(None)

    compressed_summary = None
    if all(ct is not None for ct in compressed_rows):
        compressed_summary = inspect_rows(rt, compressed_rows, coeff_sample=4)

    return {
        "compress_reports": compress_reports,
        "compressed_summary": compressed_summary,
        "decrypt_rows": decrypt_rows,
        "decoded_matrix": decoded,
    }


def summarize_matrix_error(got, ref):
    if any(r is None for r in got):
        return {
            "all_rows_decrypted": False,
            "exact": False,
            "max_abs_err": None,
            "diff": None,
            "nonzero_positions_got": None,
            "nonzero_positions_ref": [
                [int(i), int(j), int(ref[i, j])]
                for i in range(ref.shape[0])
                for j in range(ref.shape[1])
                if ref[i, j] != 0
            ],
        }

    G = np.array(got, dtype=np.int64)
    D = G - ref

    return {
        "all_rows_decrypted": True,
        "exact": bool(np.array_equal(G, ref)),
        "max_abs_err": int(np.max(np.abs(D))),
        "diff": D.tolist(),
        "nonzero_positions_got": [
            [int(i), int(j), int(G[i, j])]
            for i in range(G.shape[0])
            for j in range(G.shape[1])
            if G[i, j] != 0
        ],
        "nonzero_positions_ref": [
            [int(i), int(j), int(ref[i, j])]
            for i in range(ref.shape[0])
            for j in range(ref.shape[1])
            if ref[i, j] != 0
        ],
    }


def main():
    n = 4
    rng = np.random.default_rng(208100)

    cases_spec = [
        {
            "name": "basis_E00_times_E00",
            "U": one_hot(n, 0, 0),
            "V": one_hot(n, 0, 0),
            "zh": "理论输出 E00；最基本 one-hot",
        },
        {
            "name": "basis_E01_times_E10",
            "U": one_hot(n, 0, 1),
            "V": one_hot(n, 1, 0),
            "zh": "理论输出 E00；检查内积索引",
        },
        {
            "name": "basis_E01_times_E12",
            "U": one_hot(n, 0, 1),
            "V": one_hot(n, 1, 2),
            "zh": "理论输出 E02；检查列位置",
        },
        {
            "name": "basis_E23_times_E31",
            "U": one_hot(n, 2, 3),
            "V": one_hot(n, 3, 1),
            "zh": "理论输出 E21；检查行列方向",
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
        "experiment": "wp4x6_full_paper_pipeline_final_probe",
        "purpose": (
            "Run final compress/decrypt diagnostics on the full paper-shaped Algorithm 3 pipeline "
            "after line1-line6 pre-rescale structure has been implemented."
        ),
        "status": "full_paper_shaped_pipeline_compress_decrypt_probe_no_explicit_rescale_api",
        "n": n,
        "pipeline": {
            "line1": "(A_U,B_U)=Transpose(ct_U)",
            "line2": "raw RNS PP-MM: M00/M01/M10/M11",
            "line3": "Transpose((M01,0))",
            "line4": "Transpose((M00,0))",
            "line5": "KS_{s²->s}((A_hat,0))+(B_hat,0)",
            "line6_pre_rescale": "line5 + Transpose((M01,0)) + (M10,M11)",
            "final_probe": "Compress(towers_left=1)+decrypt_coeff_row_i64",
        },
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for spec in cases_spec:
            U = spec["U"]
            V = spec["V"]
            W_ref = U @ V

            line6_pre_rows, trace = run_paper_pipeline_pre_rows(rt, U, V)
            final = decrypt_after_compress(rt, line6_pre_rows, logical_length=n)
            analysis = summarize_matrix_error(final["decoded_matrix"], W_ref)

            case = {
                "name": spec["name"],
                "zh": spec["zh"],
                "input_U": U.tolist(),
                "input_V": V.tolist(),
                "reference_U_matmul_V": W_ref.tolist(),
                "trace": trace,
                "final": final,
                "analysis": analysis,
            }

            report["cases"].append(case)

    exact_cases = [
        c["name"] for c in report["cases"]
        if c["analysis"]["exact"]
    ]
    failed_cases = [
        c["name"] for c in report["cases"]
        if not c["analysis"]["exact"]
    ]

    one_hot_failure_summary = []
    for c in report["cases"]:
        if c["name"].startswith("basis_") and not c["analysis"]["exact"]:
            got_nnz = c["analysis"]["nonzero_positions_got"]
            ref_nnz = c["analysis"]["nonzero_positions_ref"]
            one_hot_failure_summary.append({
                "name": c["name"],
                "got_nonzero_count": None if got_nnz is None else len(got_nnz),
                "ref_nonzero_count": len(ref_nnz),
                "got_nonzero_positions": got_nnz,
                "ref_nonzero_positions": ref_nnz,
            })

    all_decrypted = all(
        c["analysis"]["all_rows_decrypted"]
        for c in report["cases"]
    )

    all_alphas_ok = all(
        c["trace"]["U_cmt_auto_alphas"] == [1, 3, 5, 7]
        and c["trace"]["T01_cmt_auto_alphas"] == [1, 3, 5, 7]
        and c["trace"]["T00_cmt_auto_alphas"] == [1, 3, 5, 7]
        for c in report["cases"]
    )

    all_line5_ks_3part = all(
        rows_are_npart(c["trace"]["line5_ks_inputs"], n, 3)
        for c in report["cases"]
    )
    all_line5_2part = all(
        rows_are_npart(c["trace"]["line5_rows"], n, 2)
        for c in report["cases"]
    )
    all_line6_2part = all(
        rows_are_npart(c["trace"]["line6_pre_rows"], n, 2)
        for c in report["cases"]
    )

    report["checks"] = {
        "all_cases_decrypted（所有 case 都成功 compress/decrypt）": all_decrypted,
        "all_cmt_alphas_expected（所有 C-MT alpha 都是 [1,3,5,7]）": all_alphas_ok,
        "all_line5_ks_inputs_are_3part（所有 line5 KS 输入都是三分量）": all_line5_ks_3part,
        "all_line5_rows_are_2part（所有 line5 输出都是二分量）": all_line5_2part,
        "all_line6_pre_rows_are_2part（所有 line6 pre-rescale 输出都是二分量）": all_line6_2part,
        "exact_case_count（完全等于 U@V 的 case 数量）": len(exact_cases),
        "failed_case_count（不等于 U@V 的 case 数量）": len(failed_cases),
        "one_hot_failure_summary（one-hot 失败诊断）": one_hot_failure_summary,
    }

    report["summary"] = {
        "exact_cases": exact_cases,
        "failed_cases": failed_cases,
        "interpretation_if_all_exact": "Paper-shaped Algorithm 3 toy-size E2E succeeds; next promote as experimental API.",
        "interpretation_if_decrypt_ok_but_wrong": (
            "Pipeline structure and final decode work, but semantics still need correction: likely scale/rescale, "
            "component order in line5/line6, or C-MT/PP-MM orientation."
        ),
        "next_step_if_failed": "WP4-X7: compare against a pure plaintext simulation of Algorithm 3 with the exact same C-MT/PP-MM/component-order conventions.",
        "zh": "如果全部能解密但不等于 U@V，下一步不再盲调代码，而是做同构明文模拟来定位是 line5/line6 组合还是 C-MT/PP-MM orientation。",
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x6_full_paper_pipeline_final_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X6 full paper-shaped pipeline final probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
