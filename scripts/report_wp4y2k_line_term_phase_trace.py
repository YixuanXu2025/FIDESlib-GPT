import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hegpt import HEConfig, HERuntime
from hegpt.ccmm_oracle_debug import (
    logical_n_cmt_oracle,
    raw_rns_ppmm,
    import_ppmm_rows,
    make_zero_1part_rows,
    make_temp_pair_bundle,
    extract_1part_rows,
    raw_add_2part_rows,
    normalize_export,
)

from report_wp4y2i_controlled_phase_aware_full_ccmm import (
    make_controlled_rowwise_bundle,
    phase_aware_logical_n_cmt_oracle,
    decrypt_rows_after_compress,
    compare_matrix,
    inspect_rows,
    rows_are_npart,
)


def one_hot(n, i, j, value=1):
    M = np.zeros((n, n), dtype=np.int64)
    M[i, j] = value
    return M


def nonzero_positions(M):
    if M is None or any(row is None for row in M):
        return None

    out = []
    for i, row in enumerate(M):
        for j, v in enumerate(row):
            if int(v) != 0:
                out.append([int(i), int(j), int(v)])
    return out


def matrix_add(*mats):
    arrs = [np.array(m, dtype=np.int64) for m in mats]
    out = arrs[0].copy()
    for x in arrs[1:]:
        out += x
    return out.tolist()


def matrix_diff(A, B):
    if A is None or B is None:
        return None
    D = np.array(A, dtype=np.int64) - np.array(B, dtype=np.int64)
    return {
        "max_abs_err": int(np.max(np.abs(D))),
        "l1_err": int(np.sum(np.abs(D))),
        "diff": D.tolist(),
    }


def assemble_pair_rows(rt, A_rows, B_rows, n):
    """
    Paper pair (A,B) maps to OpenFHE c0=B,c1=A.
    """
    return [
        rt.assemble_2part_from_1parts_coeff_ct(B_rows[i], A_rows[i])
        for i in range(n)
    ]


def decrypt_pair(rt, A_rows, B_rows, n):
    rows = assemble_pair_rows(rt, A_rows, B_rows, n)
    return decrypt_rows_after_compress(rt, rows, logical_length=n), rows


def export_component_sources(rt, left_bundle, right_bundle, n):
    return {
        "A_L": normalize_export(rt.export_component_coeff_matrix_u64(left_bundle.rows, part=1, coeff_count=n)),
        "B_L": normalize_export(rt.export_component_coeff_matrix_u64(left_bundle.rows, part=0, coeff_count=n)),
        "A_R": normalize_export(rt.export_component_coeff_matrix_u64(right_bundle.rows, part=1, coeff_count=n)),
        "B_R": normalize_export(rt.export_component_coeff_matrix_u64(right_bundle.rows, part=0, coeff_count=n)),
    }


def temp_pair_cmt(rt, bundle, n, a_scalar, mode, label):
    if mode == "raw_component":
        return logical_n_cmt_oracle(rt, bundle, n, label=label), {
            "mode": "raw_component",
            "zh": "raw component transpose",
        }

    if mode == "phase_aware":
        return phase_aware_logical_n_cmt_oracle(
            rt=rt,
            bundle=bundle,
            n=n,
            a_scalar=a_scalar,
            label=label,
        )

    raise ValueError(f"unknown mode: {mode}")


def run_trace(rt, case_name, U, V, a_scalar, temp_cmt_mode):
    n = U.shape[0]

    U_bundle, U_info = make_controlled_rowwise_bundle(rt, U.tolist(), a_scalar=a_scalar)
    V_bundle, V_info = make_controlled_rowwise_bundle(rt, V.tolist(), a_scalar=a_scalar)

    U_dec = decrypt_rows_after_compress(rt, U_bundle.rows, logical_length=n)
    V_dec = decrypt_rows_after_compress(rt, V_bundle.rows, logical_length=n)

    # line1: C-MT(U), fixed phase-aware because U is ordinary RLWE ciphertext matrix.
    U_cmt_bundle, U_cmt_trace = phase_aware_logical_n_cmt_oracle(
        rt=rt,
        bundle=U_bundle,
        n=n,
        a_scalar=a_scalar,
        label=f"phase_aware_CMT_U_a{a_scalar}",
    )
    U_cmt_dec = decrypt_rows_after_compress(rt, U_cmt_bundle.rows, logical_length=n)

    comps = export_component_sources(rt, U_cmt_bundle, V_bundle, n)

    # line2 raw RNS PP-MM component products.
    M00 = raw_rns_ppmm(comps["A_L"], comps["A_R"], n)
    M01 = raw_rns_ppmm(comps["A_L"], comps["B_R"], n)
    M10 = raw_rns_ppmm(comps["B_L"], comps["A_R"], n)
    M11 = raw_rns_ppmm(comps["B_L"], comps["B_R"], n)

    M00_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M00)
    M01_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M01)
    M10_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M10)
    M11_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M11)

    zero_rows = make_zero_1part_rows(rt, U_cmt_bundle.rows, num_towers=M00["num_towers"])

    # Interpret each Mij as B-only and A-only diagnostic pairs.
    M00_B_dec, M00_B_rows = decrypt_pair(rt, zero_rows, M00_rows, n)
    M01_B_dec, M01_B_rows = decrypt_pair(rt, zero_rows, M01_rows, n)
    M10_B_dec, M10_B_rows = decrypt_pair(rt, zero_rows, M10_rows, n)
    M11_B_dec, M11_B_rows = decrypt_pair(rt, zero_rows, M11_rows, n)

    M00_A_dec, M00_A_rows = decrypt_pair(rt, M00_rows, zero_rows, n)
    M01_A_dec, M01_A_rows = decrypt_pair(rt, M01_rows, zero_rows, n)
    M10_A_dec, M10_A_rows = decrypt_pair(rt, M10_rows, zero_rows, n)
    M11_A_dec, M11_A_rows = decrypt_pair(rt, M11_rows, zero_rows, n)

    # line3: C-MT((M01,0)).
    T01_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M01_rows,
        B_rows=zero_rows,
        n=n,
        label="T01=(M01,0)",
        template_bundle=U_bundle,
    )
    T01_before_dec = decrypt_rows_after_compress(rt, T01_bundle.rows, logical_length=n)

    T01_cmt_bundle, T01_cmt_trace = temp_pair_cmt(
        rt,
        T01_bundle,
        n=n,
        a_scalar=a_scalar,
        mode=temp_cmt_mode,
        label=f"{temp_cmt_mode}_CMT_T01_a{a_scalar}",
    )
    T01_after_dec = decrypt_rows_after_compress(rt, T01_cmt_bundle.rows, logical_length=n)

    # line4: C-MT((M00,0)).
    T00_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M00_rows,
        B_rows=zero_rows,
        n=n,
        label="T00=(M00,0)",
        template_bundle=U_bundle,
    )
    T00_before_dec = decrypt_rows_after_compress(rt, T00_bundle.rows, logical_length=n)

    T00_cmt_bundle, T00_cmt_trace = temp_pair_cmt(
        rt,
        T00_bundle,
        n=n,
        a_scalar=a_scalar,
        mode=temp_cmt_mode,
        label=f"{temp_cmt_mode}_CMT_T00_a{a_scalar}",
    )
    T00_after_dec = decrypt_rows_after_compress(rt, T00_cmt_bundle.rows, logical_length=n)

    # Extract paper pair (A_hat,B_hat), OpenFHE order c0=B_hat,c1=A_hat.
    Bhat_rows, Bhat_export = extract_1part_rows(rt, T00_cmt_bundle.rows, part=0, coeff_count=0)
    Ahat_rows, Ahat_export = extract_1part_rows(rt, T00_cmt_bundle.rows, part=1, coeff_count=0)

    # line5.
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

        bhat_term = rt.assemble_2part_from_1parts_coeff_ct(
            zero_rows[i],
            Bhat_rows[i],
        )
        line5_bhat_terms.append(bhat_term)

        line5_rows.append(rt.add_coeff_ct(ks_output, bhat_term))

    line5_dec = decrypt_rows_after_compress(rt, line5_rows, logical_length=n)
    line5_ks_output_dec = decrypt_rows_after_compress(rt, line5_ks_outputs, logical_length=n)
    line5_bhat_dec = decrypt_rows_after_compress(rt, line5_bhat_terms, logical_length=n)

    # line6: (M10,M11), OpenFHE c0=M11,c1=M10.
    M10_M11_pair_rows = assemble_pair_rows(rt, M10_rows, M11_rows, n)
    M10_M11_dec = decrypt_rows_after_compress(rt, M10_M11_pair_rows, logical_length=n)

    # final: raw add per X7k.
    tmp = raw_add_2part_rows(rt, line5_rows, T01_cmt_bundle.rows, n)
    tmp_dec = decrypt_rows_after_compress(rt, tmp, logical_length=n)

    final_rows = raw_add_2part_rows(rt, tmp, M10_M11_pair_rows, n)
    final_dec = decrypt_rows_after_compress(rt, final_rows, logical_length=n)

    # Decoded-term add consistency check.
    decoded_term_sum = None
    decoded_term_sum_vs_final = None
    try:
        decoded_term_sum = matrix_add(
            line5_dec["decoded"],
            T01_after_dec["decoded"],
            M10_M11_dec["decoded"],
        )
        decoded_term_sum_vs_final = matrix_diff(decoded_term_sum, final_dec["decoded"])
    except Exception as e:
        decoded_term_sum_vs_final = {"error": repr(e)}

    refs = {
        "U_matmul_V": U @ V,
        "U_T_matmul_V": U.T @ V,
        "U_matmul_V_T": U @ V.T,
        "V_matmul_U": V @ U,
        "V_T_matmul_U": V.T @ U,
        "U_T_matmul_V_T": U.T @ V.T,
    }

    comparisons = {
        name: compare_matrix(final_dec["decoded"], ref)
        for name, ref in refs.items()
    }

    term_decoded = {
        "U": U_dec["decoded"],
        "V": V_dec["decoded"],
        "U_cmt": U_cmt_dec["decoded"],

        "M00_as_B_only": M00_B_dec["decoded"],
        "M01_as_B_only": M01_B_dec["decoded"],
        "M10_as_B_only": M10_B_dec["decoded"],
        "M11_as_B_only": M11_B_dec["decoded"],

        "M00_as_A_only_phase": M00_A_dec["decoded"],
        "M01_as_A_only_phase": M01_A_dec["decoded"],
        "M10_as_A_only_phase": M10_A_dec["decoded"],
        "M11_as_A_only_phase": M11_A_dec["decoded"],

        "T01_before_cmt": T01_before_dec["decoded"],
        "T01_after_cmt": T01_after_dec["decoded"],
        "T00_before_cmt": T00_before_dec["decoded"],
        "T00_after_cmt": T00_after_dec["decoded"],

        "line5_ks_output": line5_ks_output_dec["decoded"],
        "line5_bhat_term": line5_bhat_dec["decoded"],
        "line5": line5_dec["decoded"],

        "M10_M11_pair": M10_M11_dec["decoded"],
        "tmp_line5_plus_T01": tmp_dec["decoded"],
        "final": final_dec["decoded"],
        "decoded_term_sum_line5_T01_M10M11": decoded_term_sum,
    }

    term_nonzeros = {
        k: nonzero_positions(v)
        for k, v in term_decoded.items()
    }

    return {
        "case_name": case_name,
        "a_scalar": int(a_scalar),
        "temp_cmt_mode": temp_cmt_mode,
        "input_U": U.tolist(),
        "input_V": V.tolist(),
        "references": {k: v.tolist() for k, v in refs.items()},
        "comparisons": comparisons,
        "term_decoded": term_decoded,
        "term_nonzero_positions": term_nonzeros,
        "decoded_term_sum_vs_final": decoded_term_sum_vs_final,
        "shape_checks": {
            "line5_ks_inputs_are_3part": rows_are_npart(inspect_rows(rt, line5_ks_inputs, coeff_sample=4), n, 3),
            "final_rows_are_2part": rows_are_npart(inspect_rows(rt, final_rows, coeff_sample=4), n, 2),
        },
        "exports": {
            "Ahat_export": Ahat_export,
            "Bhat_export": Bhat_export,
        },
        "traces": {
            "U_info": U_info,
            "V_info": V_info,
            "U_cmt_trace": U_cmt_trace,
            "T01_cmt_trace": T01_cmt_trace,
            "T00_cmt_trace": T00_cmt_trace,
        },
    }


def compact_trace(case_trace):
    """
    Keep the report readable: preserve full matrices for a small number of selected cases,
    and summarize others by nonzero positions / comparisons.
    """
    return {
        "case_name": case_trace["case_name"],
        "a_scalar": case_trace["a_scalar"],
        "temp_cmt_mode": case_trace["temp_cmt_mode"],
        "comparisons": case_trace["comparisons"],
        "term_nonzero_positions": case_trace["term_nonzero_positions"],
        "decoded_term_sum_vs_final": case_trace["decoded_term_sum_vs_final"],
        "shape_checks": case_trace["shape_checks"],
    }


def main():
    n = 4
    rng = np.random.default_rng(209500)

    cases = [
        {
            "name": "basis_E00_times_E00",
            "U": one_hot(n, 0, 0),
            "V": one_hot(n, 0, 0),
            "keep_full": True,
        },
        {
            "name": "basis_E01_times_E12",
            "U": one_hot(n, 0, 1),
            "V": one_hot(n, 1, 2),
            "keep_full": True,
        },
        {
            "name": "identity_times_random_small",
            "U": np.eye(n, dtype=np.int64),
            "V": rng.integers(-2, 3, size=(n, n)).astype(np.int64),
            "keep_full": False,
        },
        {
            "name": "random_small_times_identity",
            "U": rng.integers(-2, 3, size=(n, n)).astype(np.int64),
            "V": np.eye(n, dtype=np.int64),
            "keep_full": False,
        },
    ]

    temp_cmt_modes = ["raw_component", "phase_aware"]
    a_scalar = 1

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
        "experiment": "wp4y2k_line_term_phase_trace",
        "purpose": (
            "Decode line2-line6 intermediate terms of controlled RLWE + phase-aware C-MT full CCMM. "
            "This locates whether mismatch starts at M00/M01/M10/M11, line5, T01, or line6 addition."
        ),
        "status": "debug_term_trace_not_secure",
        "n": n,
        "a_scalar": a_scalar,
        "cases_full": [],
        "cases_compact": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for mode in temp_cmt_modes:
            for c in cases:
                trace = run_trace(
                    rt=rt,
                    case_name=c["name"],
                    U=c["U"],
                    V=c["V"],
                    a_scalar=a_scalar,
                    temp_cmt_mode=mode,
                )

                if c["keep_full"]:
                    report["cases_full"].append(trace)
                else:
                    report["cases_compact"].append(compact_trace(trace))

    all_cases = report["cases_full"] + report["cases_compact"]

    ref_names = [
        "U_matmul_V",
        "U_T_matmul_V",
        "U_matmul_V_T",
        "V_matmul_U",
        "V_T_matmul_U",
        "U_T_matmul_V_T",
    ]

    by_mode = {}
    for mode in temp_cmt_modes:
        cs = [c for c in all_cases if c["temp_cmt_mode"] == mode]
        by_mode[mode] = {
            "num_cases": len(cs),
            "match_counts": {
                ref: sum(1 for c in cs if c["comparisons"][ref]["exact"])
                for ref in ref_names
            },
            "decoded_term_sum_equals_final": all(
                (
                    c.get("decoded_term_sum_vs_final", {}).get("max_abs_err") == 0
                    if isinstance(c.get("decoded_term_sum_vs_final"), dict)
                    else False
                )
                for c in cs
            ),
            "all_line5_ks_inputs_are_3part": all(c["shape_checks"]["line5_ks_inputs_are_3part"] for c in cs),
            "all_final_rows_are_2part": all(c["shape_checks"]["final_rows_are_2part"] for c in cs),
        }

    report["checks"] = {
        "summary_by_temp_cmt_mode（按 temp C-MT mode 汇总）": by_mode,
        "any_decoded_term_sum_differs_from_final（decoded term sum 是否与 raw-add final 不一致）": any(
            not by_mode[m]["decoded_term_sum_equals_final"]
            for m in by_mode
        ),
        "first_full_case_raw_component_terms（第一个 raw_component full case 的逐项矩阵）": next(
            c for c in report["cases_full"] if c["temp_cmt_mode"] == "raw_component"
        ),
        "first_full_case_phase_aware_terms（第一个 phase_aware full case 的逐项矩阵）": next(
            c for c in report["cases_full"] if c["temp_cmt_mode"] == "phase_aware"
        ),
    }

    report["global_decision"] = {
        "value": "line_term_trace_generated",
        "next_step": "WP4-Y2l",
        "next_goal": "Use Y2k term matrices to identify exact formula mismatch and patch Algorithm 3 component composition.",
        "zh": "Y2k 已生成逐项 plaintext phase trace；下一步根据矩阵定位公式差异。",
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4y2k_line_term_phase_trace_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Y2k line-term plaintext phase trace")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
