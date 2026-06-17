import json
import sys
from pathlib import Path
from itertools import product

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hegpt import HEConfig, HERuntime
from hegpt.ccmm_oracle_debug import (
    TempRowwiseBundle,
    logical_n_cmt_oracle,
    raw_rns_ppmm,
    import_ppmm_rows,
    make_zero_1part_rows,
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

from report_wp4y2p_line5_full_ring_raw_add_formula_sweep import (
    make_temp_pair_custom,
    temp_pair_cmt,
    make_line5_rows,
    make_line6_pair,
)


def one_hot(n, i, j, value=1):
    M = np.zeros((n, n), dtype=np.int64)
    M[i, j] = value
    return M


def transpose_export_matrix(ex, n):
    """
    Transpose the first n x n coefficient block while preserving the
    normalized export schema expected by raw_rns_ppmm():

      rows[row_i]["towers"][tower_i]["coeffs_u64"]

    Previous Y2q returned rows[*]["towers_coeffs"], which caused
    KeyError('towers') for every left_T/right_T variant.
    """
    num_towers = int(ex["num_towers"])
    moduli = ex.get("moduli_u64", ex.get("moduli"))

    out = {
        "num_rows": n,
        "num_towers": num_towers,
        "ring_dim": ex.get("ring_dim", n),
        "moduli_u64": [int(q) for q in moduli],
        "rows": [],
    }

    for i in range(n):
        towers = []

        for t in range(num_towers):
            coeffs = []

            for r in range(n):
                tower = ex["rows"][r]["towers"][t]
                if "coeffs_u64" in tower:
                    coeffs.append(int(tower["coeffs_u64"][i]))
                elif "coeffs" in tower:
                    coeffs.append(int(tower["coeffs"][i]))
                else:
                    raise KeyError("coeffs_u64")

            towers.append({
                "tower_index": t,
                "coeffs_u64": coeffs,
            })

        out["rows"].append({
            "row_index": i,
            "towers": towers,
        })

    return out


def maybe_transpose_export(ex, n, flag):
    return transpose_export_matrix(ex, n) if flag else ex


def negate_ppmm_result(pp):
    moduli = [int(q) for q in pp["moduli_u64"]]
    out = {
        "num_rows": pp["num_rows"],
        "num_towers": pp["num_towers"],
        "moduli_u64": moduli,
        "rows": [],
    }

    for row in pp["rows"]:
        towers_coeffs = []

        for t, coeffs in enumerate(row["towers_coeffs"]):
            q = moduli[t]
            towers_coeffs.append([(-int(x)) % q for x in coeffs])

        out["rows"].append({
            "row_index": row["row_index"],
            "towers_coeffs": towers_coeffs,
        })

    return out


def signed_ppmm(left, right, n, sign):
    pp = raw_rns_ppmm(left, right, n)
    if int(sign) == 1:
        return pp
    if int(sign) == -1:
        return negate_ppmm_result(pp)
    raise ValueError(f"invalid sign: {sign}")


def export_component_sources(rt, left_bundle, right_bundle, n, source_map, ppmm_orientation):
    raw = {
        "A_L": normalize_export(rt.export_component_coeff_matrix_u64(left_bundle.rows, part=1, coeff_count=n)),
        "B_L": normalize_export(rt.export_component_coeff_matrix_u64(left_bundle.rows, part=0, coeff_count=n)),
        "A_R": normalize_export(rt.export_component_coeff_matrix_u64(right_bundle.rows, part=1, coeff_count=n)),
        "B_R": normalize_export(rt.export_component_coeff_matrix_u64(right_bundle.rows, part=0, coeff_count=n)),
    }

    if source_map["left"] == "normal":
        L_A, L_B = raw["A_L"], raw["B_L"]
    elif source_map["left"] == "swapped":
        L_A, L_B = raw["B_L"], raw["A_L"]
    else:
        raise ValueError(source_map["left"])

    if source_map["right"] == "normal":
        R_A, R_B = raw["A_R"], raw["B_R"]
    elif source_map["right"] == "swapped":
        R_A, R_B = raw["B_R"], raw["A_R"]
    else:
        raise ValueError(source_map["right"])

    L_T = ppmm_orientation["left_T"]
    R_T = ppmm_orientation["right_T"]

    return {
        "A_L": maybe_transpose_export(L_A, n, L_T),
        "B_L": maybe_transpose_export(L_B, n, L_T),
        "A_R": maybe_transpose_export(R_A, n, R_T),
        "B_R": maybe_transpose_export(R_B, n, R_T),
    }


def run_variant_case(rt, U, V, variant):
    n = U.shape[0]
    a_scalar = 1

    U_bundle, _ = make_controlled_rowwise_bundle(rt, U.tolist(), a_scalar=a_scalar)
    V_bundle, _ = make_controlled_rowwise_bundle(rt, V.tolist(), a_scalar=a_scalar)

    # line1 fixed: U is real controlled RLWE, so phase-aware C-MT.
    U_cmt_bundle, _ = phase_aware_logical_n_cmt_oracle(
        rt=rt,
        bundle=U_bundle,
        n=n,
        a_scalar=a_scalar,
        label="phase_aware_CMT_U",
    )

    comps = export_component_sources(
        rt,
        left_bundle=U_cmt_bundle,
        right_bundle=V_bundle,
        n=n,
        source_map=variant["source_map"],
        ppmm_orientation=variant["ppmm_orientation"],
    )

    signs = variant["term_signs"]

    M00 = signed_ppmm(comps["A_L"], comps["A_R"], n, signs["M00"])
    M01 = signed_ppmm(comps["A_L"], comps["B_R"], n, signs["M01"])
    M10 = signed_ppmm(comps["B_L"], comps["A_R"], n, signs["M10"])
    M11 = signed_ppmm(comps["B_L"], comps["B_R"], n, signs["M11"])

    M00_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M00)
    M01_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M01)
    M10_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M10)
    M11_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M11)

    zero_rows = make_zero_1part_rows(rt, U_cmt_bundle.rows, num_towers=M00["num_towers"])

    # Fixed to Y2p best variant:
    # temp_cmt_mode=raw_component, T01=Bonly, T00=Aonly,
    # line5=ks_p0_lin_p1_as_c0, line6=normal_M10_A_M11_B
    T01_bundle = make_temp_pair_custom(
        rt,
        rows_x=M01_rows,
        zero_rows=zero_rows,
        n=n,
        placement="Bonly",
        template_bundle=U_bundle,
        label="T01_Bonly",
    )
    T01_cmt_bundle, _ = temp_pair_cmt(
        rt=rt,
        bundle=T01_bundle,
        n=n,
        a_scalar=a_scalar,
        mode="raw_component",
        label="raw_component_T01_Bonly",
    )

    T00_bundle = make_temp_pair_custom(
        rt,
        rows_x=M00_rows,
        zero_rows=zero_rows,
        n=n,
        placement="Aonly",
        template_bundle=U_bundle,
        label="T00_Aonly",
    )
    T00_cmt_bundle, _ = temp_pair_cmt(
        rt=rt,
        bundle=T00_bundle,
        n=n,
        a_scalar=a_scalar,
        mode="raw_component",
        label="raw_component_T00_Aonly",
    )

    line5_variant = {
        "id": "ks_p0_lin_p1_as_c0",
        "ks_part": 0,
        "lin_part": 1,
        "lin_place": "c0",
    }

    line5_rows, line5_trace = make_line5_rows(
        rt=rt,
        zero_rows=zero_rows,
        T00_cmt_rows=T00_cmt_bundle.rows,
        n=n,
        line5_variant=line5_variant,
    )

    line6_pair_rows = make_line6_pair(
        rt=rt,
        M10_rows=M10_rows,
        M11_rows=M11_rows,
        n=n,
        line6_mapping="normal_M10_A_M11_B",
    )

    tmp = raw_add_2part_rows(rt, line5_rows, T01_cmt_bundle.rows, n)
    final_rows = raw_add_2part_rows(rt, tmp, line6_pair_rows, n)

    final = decrypt_rows_after_compress(rt, final_rows, logical_length=n)

    return {
        "final": final,
        "shape": {
            "line5_ks_inputs_are_3part": rows_are_npart(
                inspect_rows(rt, line5_trace["ks_inputs"], coeff_sample=4),
                n,
                3,
            ),
            "final_rows_are_2part": rows_are_npart(
                inspect_rows(rt, final_rows, coeff_sample=4),
                n,
                2,
            ),
        },
    }


def compare_all_refs(got, U, V):
    refs = {
        "U_matmul_V": U @ V,
        "U_T_matmul_V": U.T @ V,
        "U_matmul_V_T": U @ V.T,
        "V_matmul_U": V @ U,
        "V_T_matmul_U": V.T @ U,
        "U_T_matmul_V_T": U.T @ V.T,
    }

    return {
        name: compare_matrix(got, ref)
        for name, ref in refs.items()
    }


def build_variants():
    variants = []
    vid = 0

    for left_map in ["normal", "swapped"]:
        for right_map in ["normal", "swapped"]:
            for left_T in [False, True]:
                for right_T in [False, True]:
                    for signs in product([1, -1], repeat=4):
                        vid += 1
                        term_signs = {
                            "M00": signs[0],
                            "M01": signs[1],
                            "M10": signs[2],
                            "M11": signs[3],
                        }

                        variants.append({
                            "variant_id": (
                                f"q{vid:04d}_L-{left_map}_R-{right_map}"
                                f"_LT-{int(left_T)}_RT-{int(right_T)}"
                                f"_s{term_signs['M00']}{term_signs['M01']}{term_signs['M10']}{term_signs['M11']}"
                            ),
                            "source_map": {
                                "left": left_map,
                                "right": right_map,
                            },
                            "ppmm_orientation": {
                                "left_T": bool(left_T),
                                "right_T": bool(right_T),
                            },
                            "term_signs": term_signs,
                        })

    return variants


def main():
    n = 4

    cases = [
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
    ]

    ref_names = [
        "U_matmul_V",
        "U_T_matmul_V",
        "U_matmul_V_T",
        "V_matmul_U",
        "V_T_matmul_U",
        "U_T_matmul_V_T",
    ]

    variants = build_variants()

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
        "experiment": "wp4y2q_ppmm_source_sign_sweep",
        "purpose": (
            "Sweep PP-MM source mapping, operand transpose, and M00/M01/M10/M11 signs, "
            "using Y2p best line3-line6 framework."
        ),
        "status": "debug_ppmm_source_sign_sweep",
        "n": n,
        "base_line3_line6_variant": {
            "from": "Y2p best",
            "temp_cmt_mode": "raw_component",
            "T01_placement": "Bonly",
            "T00_placement": "Aonly",
            "line5_variant": "ks_p0_lin_p1_as_c0",
            "line6_mapping": "normal_M10_A_M11_B",
            "line5_add": "full_ring_raw_add_2part_rows",
        },
        "num_variants": len(variants),
        "num_cases": len(cases),
        "variant_results": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for variant in variants:
            vr = {
                "variant": variant,
                "cases": [],
                "errors": [],
            }

            for case in cases:
                U = case["U"]
                V = case["V"]

                try:
                    value = run_variant_case(rt, U, V, variant)
                except Exception as e:
                    vr["errors"].append({
                        "case": case["name"],
                        "error": repr(e),
                    })
                    continue

                comparisons = compare_all_refs(value["final"]["decoded"], U, V)

                vr["cases"].append({
                    "name": case["name"],
                    "all_decrypted": comparisons["U_matmul_V"]["all_rows_decrypted"],
                    "shape": value["shape"],
                    "comparisons": comparisons,
                    "decoded": value["final"]["decoded"],
                })

            match_counts = {
                ref: sum(
                    1 for c in vr["cases"]
                    if c["comparisons"][ref]["exact"]
                )
                for ref in ref_names
            }

            total_l1 = {
                ref: sum(
                    c["comparisons"][ref]["l1_err"]
                    for c in vr["cases"]
                    if c["comparisons"][ref]["l1_err"] is not None
                )
                for ref in ref_names
            }

            vr["summary"] = {
                "num_cases_completed": len(vr["cases"]),
                "num_errors": len(vr["errors"]),
                "all_decrypted": all(c["all_decrypted"] for c in vr["cases"]) if vr["cases"] else False,
                "all_line5_ks_inputs_are_3part": all(
                    c["shape"]["line5_ks_inputs_are_3part"] for c in vr["cases"]
                ) if vr["cases"] else False,
                "all_final_rows_are_2part": all(
                    c["shape"]["final_rows_are_2part"] for c in vr["cases"]
                ) if vr["cases"] else False,
                "match_counts": match_counts,
                "total_l1": total_l1,
            }

            report["variant_results"].append(vr)

    ranked = sorted(
        report["variant_results"],
        key=lambda vr: (
            vr["summary"]["num_errors"] > 0,
            -vr["summary"]["num_cases_completed"],
            -max(vr["summary"]["match_counts"].values()),
            vr["summary"]["total_l1"]["U_matmul_V"],
            vr["variant"]["variant_id"],
        ),
    )

    best = ranked[0]

    solved_variants = []
    for vr in report["variant_results"]:
        for ref in ref_names:
            if vr["summary"]["match_counts"][ref] == len(cases):
                solved_variants.append({
                    "variant_id": vr["variant"]["variant_id"],
                    "reference": ref,
                    "variant": vr["variant"],
                    "summary": vr["summary"],
                })

    report["checks"] = {
        "variants_tested（扫描 variant 数量）": len(variants),
        "cases_per_variant（每个 variant 的 case 数量）": len(cases),
        "any_variant_solves_U_matmul_V（是否有 variant 全部匹配 U@V）": any(
            x["reference"] == "U_matmul_V" for x in solved_variants
        ),
        "any_variant_solves_any_reference（是否有 variant 全部匹配任一参考）": len(solved_variants) > 0,
        "solved_variants（全部匹配的 variants）": solved_variants[:20],
        "best_variant": {
            "variant_id": best["variant"]["variant_id"],
            "variant": best["variant"],
            "summary": best["summary"],
        },
        "top10_variants": [
            {
                "variant_id": vr["variant"]["variant_id"],
                "variant": vr["variant"],
                "summary": vr["summary"],
            }
            for vr in ranked[:10]
        ],
    }

    if any(x["reference"] == "U_matmul_V" for x in solved_variants):
        decision = {
            "value": "ppmm_source_sign_variant_found_for_U_matmul_V",
            "next_step": "WP4-Y2r",
            "next_goal": "Promote winning PP-MM variant to identity/random diagnostics.",
            "zh": "找到匹配 U@V 的 PP-MM source/sign variant；下一步扩大测试。",
        }
    elif solved_variants:
        decision = {
            "value": "ppmm_source_sign_variant_found_but_reference_differs",
            "next_step": "WP4-Y2r",
            "next_goal": "Analyze orientation contract of winning variant.",
            "zh": "找到稳定 variant 但参考方向不是 U@V；下一步修 orientation contract。",
        }
    else:
        decision = {
            "value": "ppmm_source_sign_sweep_no_exact_variant",
            "next_step": "WP4-Y2r",
            "next_goal": "Switch to plaintext symbolic Algorithm 3 simulation and compare exact intermediate formulas.",
            "zh": "PP-MM source/sign 扫描仍未找到 exact variant；下一步做明文符号模拟，而不是继续盲扫。",
        }

    report["global_decision"] = decision

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4y2q_ppmm_source_sign_sweep_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    printable = {
        "experiment": report["experiment"],
        "status": report["status"],
        "checks": report["checks"],
        "global_decision": report["global_decision"],
        "saved": str(out_path),
    }

    print("=" * 100)
    print("WP4-Y2q PP-MM source/sign/orientation sweep")
    print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
