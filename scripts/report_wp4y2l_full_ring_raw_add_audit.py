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
)


def one_hot(n, i, j, value=1):
    M = np.zeros((n, n), dtype=np.int64)
    M[i, j] = value
    return M


def matrix_add(*mats):
    arrs = [np.array(m, dtype=np.int64) for m in mats]
    out = arrs[0].copy()
    for x in arrs[1:]:
        out += x
    return out.tolist()


def matrix_diff(A, B):
    D = np.array(A, dtype=np.int64) - np.array(B, dtype=np.int64)
    return {
        "max_abs_err": int(np.max(np.abs(D))),
        "l1_err": int(np.sum(np.abs(D))),
        "diff": D.tolist(),
    }


def export_component_sources(rt, left_bundle, right_bundle, n):
    return {
        "A_L": normalize_export(rt.export_component_coeff_matrix_u64(left_bundle.rows, part=1, coeff_count=n)),
        "B_L": normalize_export(rt.export_component_coeff_matrix_u64(left_bundle.rows, part=0, coeff_count=n)),
        "A_R": normalize_export(rt.export_component_coeff_matrix_u64(right_bundle.rows, part=1, coeff_count=n)),
        "B_R": normalize_export(rt.export_component_coeff_matrix_u64(right_bundle.rows, part=0, coeff_count=n)),
    }


def assemble_pair_rows(rt, A_rows, B_rows, n):
    """
    Paper pair (A,B) maps to OpenFHE c0=B,c1=A.
    """
    return [
        rt.assemble_2part_from_1parts_coeff_ct(B_rows[i], A_rows[i])
        for i in range(n)
    ]


def temp_pair_cmt(rt, bundle, n, a_scalar, mode, label):
    if mode == "raw_component":
        return logical_n_cmt_oracle(rt, bundle, n, label=label), {
            "mode": "raw_component",
        }

    if mode == "phase_aware":
        return phase_aware_logical_n_cmt_oracle(
            rt=rt,
            bundle=bundle,
            n=n,
            a_scalar=a_scalar,
            label=label,
        )

    raise ValueError(f"unknown temp C-MT mode: {mode}")


def get_moduli_from_export(ex):
    if "moduli_u64" in ex:
        return [int(x) for x in ex["moduli_u64"]]
    if "moduli" in ex:
        return [int(x) for x in ex["moduli"]]
    raise KeyError("export has no moduli_u64/moduli")


def get_coeffs(ex, row_i, tower_i):
    return [int(x) for x in ex["rows"][row_i]["towers"][tower_i]["coeffs_u64"]]


def full_ring_raw_add_2part_rows(rt, left_rows, right_rows):
    """
    Full-ring raw RNS add for 2-part ciphertext rows.

    Unlike raw_add_2part_rows(..., n), this exports coeff_count=0,
    which should mean full ring_dim, then imports the full coefficient vectors.
    This avoids prefix-only import/tail contamination.
    """
    nrows = len(left_rows)

    L0 = normalize_export(rt.export_component_coeff_matrix_u64(left_rows, part=0, coeff_count=0))
    L1 = normalize_export(rt.export_component_coeff_matrix_u64(left_rows, part=1, coeff_count=0))
    R0 = normalize_export(rt.export_component_coeff_matrix_u64(right_rows, part=0, coeff_count=0))
    R1 = normalize_export(rt.export_component_coeff_matrix_u64(right_rows, part=1, coeff_count=0))

    moduli = get_moduli_from_export(L0)
    num_towers = int(L0["num_towers"])
    ring_dim = int(L0["ring_dim"])

    out_rows = []

    for i in range(nrows):
        parts_1part = []

        for part_ex_L, part_ex_R in [(L0, R0), (L1, R1)]:
            towers = []
            for t in range(num_towers):
                q = int(moduli[t])
                a = get_coeffs(part_ex_L, i, t)
                b = get_coeffs(part_ex_R, i, t)

                if len(a) != ring_dim or len(b) != ring_dim:
                    raise RuntimeError(
                        f"full-ring export length mismatch: row={i}, tower={t}, len(a)={len(a)}, len(b)={len(b)}, ring_dim={ring_dim}"
                    )

                towers.append([(int(x) + int(y)) % q for x, y in zip(a, b)])

            parts_1part.append(rt.import_1part_coeff_u64(left_rows[i], towers))

        out_rows.append(rt.assemble_2part_from_1parts_coeff_ct(parts_1part[0], parts_1part[1]))

    return out_rows, {
        "num_rows": nrows,
        "num_towers": num_towers,
        "ring_dim": ring_dim,
        "coeff_count_used": ring_dim,
        "moduli_u64": moduli,
    }


def add_coeff_ct_rows(rt, left_rows, right_rows):
    return [
        rt.add_coeff_ct(left_rows[i], right_rows[i])
        for i in range(len(left_rows))
    ]


def count_tail_nonzero(rt, rows, part, logical_n):
    ex = normalize_export(rt.export_component_coeff_matrix_u64(rows, part=part, coeff_count=0))
    count = 0
    first_positions = []

    for i, row in enumerate(ex["rows"]):
        for t, tower in enumerate(row["towers"]):
            coeffs = [int(x) for x in tower["coeffs_u64"]]
            for j, v in enumerate(coeffs[logical_n:], start=logical_n):
                if v != 0:
                    count += 1
                    if len(first_positions) < 20:
                        first_positions.append([i, t, j, v])

    return {
        "part": part,
        "tail_nonzero_count_after_logical_n": count,
        "first_tail_nonzero_positions": first_positions,
        "ring_dim": ex["ring_dim"],
        "num_rows": ex["num_rows"],
        "num_towers": ex["num_towers"],
    }


def run_build_terms(rt, temp_cmt_mode="raw_component", a_scalar=1):
    n = 4
    U = one_hot(n, 0, 0)
    V = one_hot(n, 0, 0)

    U_bundle, _ = make_controlled_rowwise_bundle(rt, U.tolist(), a_scalar=a_scalar)
    V_bundle, _ = make_controlled_rowwise_bundle(rt, V.tolist(), a_scalar=a_scalar)

    U_cmt_bundle, _ = phase_aware_logical_n_cmt_oracle(
        rt=rt,
        bundle=U_bundle,
        n=n,
        a_scalar=a_scalar,
        label=f"phase_aware_CMT_U_a{a_scalar}",
    )

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

    T01_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M01_rows,
        B_rows=zero_rows,
        n=n,
        label="T01=(M01,0)",
        template_bundle=U_bundle,
    )
    T01_cmt_bundle, _ = temp_pair_cmt(
        rt,
        T01_bundle,
        n=n,
        a_scalar=a_scalar,
        mode=temp_cmt_mode,
        label=f"{temp_cmt_mode}_CMT_T01_a{a_scalar}",
    )

    T00_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M00_rows,
        B_rows=zero_rows,
        n=n,
        label="T00=(M00,0)",
        template_bundle=U_bundle,
    )
    T00_cmt_bundle, _ = temp_pair_cmt(
        rt,
        T00_bundle,
        n=n,
        a_scalar=a_scalar,
        mode=temp_cmt_mode,
        label=f"{temp_cmt_mode}_CMT_T00_a{a_scalar}",
    )

    # Extract paper pair (A_hat,B_hat), OpenFHE order c0=B_hat,c1=A_hat.
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

        bhat_term = rt.assemble_2part_from_1parts_coeff_ct(
            zero_rows[i],
            Bhat_rows[i],
        )
        line5_bhat_terms.append(bhat_term)

        line5_rows.append(rt.add_coeff_ct(ks_output, bhat_term))

    M10_M11_pair_rows = assemble_pair_rows(rt, M10_rows, M11_rows, n)

    return {
        "n": n,
        "U": U,
        "V": V,
        "line5_rows": line5_rows,
        "T01_rows": T01_cmt_bundle.rows,
        "M10_M11_rows": M10_M11_pair_rows,
        "line5_ks_outputs": line5_ks_outputs,
        "line5_bhat_terms": line5_bhat_terms,
    }


def decrypt_term(rt, rows, n):
    return decrypt_rows_after_compress(rt, rows, logical_length=n)["decoded"]


def main():
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
        "experiment": "wp4y2l_full_ring_raw_add_audit",
        "purpose": (
            "Audit whether Y2k mismatch between decoded term sum and final output is caused by "
            "prefix-only raw_add_2part_rows/import tail contamination. Compare prefix raw add, full-ring raw add, and add_coeff_ct."
        ),
        "status": "debug_add_audit_not_secure",
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for temp_mode in ["raw_component", "phase_aware"]:
            terms = run_build_terms(rt, temp_cmt_mode=temp_mode, a_scalar=1)
            n = terms["n"]

            line5 = terms["line5_rows"]
            T01 = terms["T01_rows"]
            M10M11 = terms["M10_M11_rows"]

            # Individual decoded term sum.
            dec_line5 = decrypt_term(rt, line5, n)
            dec_T01 = decrypt_term(rt, T01, n)
            dec_M10M11 = decrypt_term(rt, M10M11, n)
            decoded_sum = matrix_add(dec_line5, dec_T01, dec_M10M11)

            # Prefix raw add path, same as Y2k.
            prefix_tmp = raw_add_2part_rows(rt, line5, T01, n)
            prefix_final = raw_add_2part_rows(rt, prefix_tmp, M10M11, n)
            dec_prefix_final = decrypt_term(rt, prefix_final, n)

            # Full-ring raw add path.
            full_tmp, full_tmp_info = full_ring_raw_add_2part_rows(rt, line5, T01)
            full_final, full_final_info = full_ring_raw_add_2part_rows(rt, full_tmp, M10M11)
            dec_full_final = decrypt_term(rt, full_final, n)

            # Native add path.
            native_tmp = add_coeff_ct_rows(rt, line5, T01)
            native_final = add_coeff_ct_rows(rt, native_tmp, M10M11)
            dec_native_final = decrypt_term(rt, native_final, n)

            refs = {
                "U_matmul_V": terms["U"] @ terms["V"],
            }

            case = {
                "temp_cmt_mode": temp_mode,
                "individual_terms_decoded": {
                    "line5": dec_line5,
                    "T01": dec_T01,
                    "M10_M11": dec_M10M11,
                    "decoded_sum_line5_T01_M10M11": decoded_sum,
                },
                "prefix_raw_add": {
                    "decoded": dec_prefix_final,
                    "diff_vs_decoded_sum": matrix_diff(dec_prefix_final, decoded_sum),
                    "compare_to_U_matmul_V": compare_matrix(dec_prefix_final, refs["U_matmul_V"]),
                },
                "full_ring_raw_add": {
                    "decoded": dec_full_final,
                    "diff_vs_decoded_sum": matrix_diff(dec_full_final, decoded_sum),
                    "compare_to_U_matmul_V": compare_matrix(dec_full_final, refs["U_matmul_V"]),
                    "tmp_info": full_tmp_info,
                    "final_info": full_final_info,
                },
                "native_add_coeff_ct": {
                    "decoded": dec_native_final,
                    "diff_vs_decoded_sum": matrix_diff(dec_native_final, decoded_sum),
                    "compare_to_U_matmul_V": compare_matrix(dec_native_final, refs["U_matmul_V"]),
                },
                "tail_nonzero_diagnostics": {
                    "line5_part0": count_tail_nonzero(rt, line5, part=0, logical_n=n),
                    "line5_part1": count_tail_nonzero(rt, line5, part=1, logical_n=n),
                    "T01_part0": count_tail_nonzero(rt, T01, part=0, logical_n=n),
                    "T01_part1": count_tail_nonzero(rt, T01, part=1, logical_n=n),
                    "M10M11_part0": count_tail_nonzero(rt, M10M11, part=0, logical_n=n),
                    "M10M11_part1": count_tail_nonzero(rt, M10M11, part=1, logical_n=n),
                },
            }

            report["cases"].append(case)

    report["checks"] = {
        "full_ring_raw_add_matches_decoded_sum_all_modes（full-ring raw add 是否全部等于逐项解密和）": all(
            c["full_ring_raw_add"]["diff_vs_decoded_sum"]["max_abs_err"] == 0
            for c in report["cases"]
        ),
        "prefix_raw_add_matches_decoded_sum_all_modes（prefix raw add 是否全部等于逐项解密和）": all(
            c["prefix_raw_add"]["diff_vs_decoded_sum"]["max_abs_err"] == 0
            for c in report["cases"]
        ),
        "native_add_matches_decoded_sum_all_modes（add_coeff_ct 是否全部等于逐项解密和）": all(
            c["native_add_coeff_ct"]["diff_vs_decoded_sum"]["max_abs_err"] == 0
            for c in report["cases"]
        ),
        "mode_summaries（各模式差异汇总）": [
            {
                "temp_cmt_mode": c["temp_cmt_mode"],
                "prefix_max_abs_err_vs_sum": c["prefix_raw_add"]["diff_vs_decoded_sum"]["max_abs_err"],
                "full_ring_max_abs_err_vs_sum": c["full_ring_raw_add"]["diff_vs_decoded_sum"]["max_abs_err"],
                "native_max_abs_err_vs_sum": c["native_add_coeff_ct"]["diff_vs_decoded_sum"]["max_abs_err"],
                "full_ring_exact_U_matmul_V": c["full_ring_raw_add"]["compare_to_U_matmul_V"]["exact"],
            }
            for c in report["cases"]
        ],
    }

    if report["checks"]["full_ring_raw_add_matches_decoded_sum_all_modes（full-ring raw add 是否全部等于逐项解密和）"]:
        decision = {
            "value": "prefix_raw_add_tail_contamination_confirmed",
            "next_step": "WP4-Y2m",
            "next_goal": "Patch ccmm_oracle_debug.raw_add_2part_rows to full-ring raw add, then re-run Y2i/Y2j.",
            "zh": "确认 prefix-only raw add/import tail 污染；下一步把 raw_add_2part_rows 改成 full-ring 版本。",
        }
    else:
        decision = {
            "value": "full_ring_raw_add_still_not_phase_linear",
            "next_step": "WP4-Y2m",
            "next_goal": "Inspect full-ring export/import or compress/decrypt metadata; raw add is still not phase-linear.",
            "zh": "full-ring raw add 仍不等于逐项 phase 和，需要检查完整导出/import 或 compress/decrypt 元数据。",
        }

    report["global_decision"] = decision

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4y2l_full_ring_raw_add_audit_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Y2l full-ring raw-add audit")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
