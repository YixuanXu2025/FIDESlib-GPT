import json
import sys
from pathlib import Path

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


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def export_component_sources(rt, left_bundle, right_bundle, n):
    return {
        "A_L": normalize_export(rt.export_component_coeff_matrix_u64(left_bundle.rows, part=1, coeff_count=n)),
        "B_L": normalize_export(rt.export_component_coeff_matrix_u64(left_bundle.rows, part=0, coeff_count=n)),
        "A_R": normalize_export(rt.export_component_coeff_matrix_u64(right_bundle.rows, part=1, coeff_count=n)),
        "B_R": normalize_export(rt.export_component_coeff_matrix_u64(right_bundle.rows, part=0, coeff_count=n)),
    }


def assemble_pair_rows(rt, A_rows, B_rows, n):
    """
    Paper pair (A,B) maps to OpenFHE component order:
      c0 = B
      c1 = A
    """
    return [
        rt.assemble_2part_from_1parts_coeff_ct(B_rows[i], A_rows[i])
        for i in range(n)
    ]


def make_temp_pair_custom(rt, rows_x, zero_rows, n, placement, template_bundle, label):
    """
    placement:
      Aonly: paper pair (X,0) => c0=0,c1=X
      Bonly: paper pair (0,X) => c0=X,c1=0
    """
    if placement == "Aonly":
        rows = assemble_pair_rows(rt, A_rows=rows_x, B_rows=zero_rows, n=n)
    elif placement == "Bonly":
        rows = assemble_pair_rows(rt, A_rows=zero_rows, B_rows=rows_x, n=n)
    else:
        raise ValueError(f"unknown temp placement: {placement}")

    return TempRowwiseBundle(
        rows=rows,
        shape=(n, n),
        label=label,
        ring_dim=template_bundle.ring_dim,
        batch_size=template_bundle.batch_size,
    )


def temp_pair_cmt(rt, bundle, n, a_scalar, mode, label):
    if mode == "raw_component":
        out = logical_n_cmt_oracle(rt, bundle, n, label=label)
        return out, {"mode": mode}

    if mode == "phase_aware":
        out, trace = phase_aware_logical_n_cmt_oracle(
            rt=rt,
            bundle=bundle,
            n=n,
            a_scalar=a_scalar,
            label=label,
        )
        trace["mode"] = mode
        return out, trace

    raise ValueError(f"unknown temp C-MT mode: {mode}")


def make_line5_rows(rt, zero_rows, T00_cmt_rows, n, line5_variant):
    """
    line5_variant fields:
      ks_part: 0 or 1        # which component becomes c2 for relinearize
      lin_part: 0 or 1       # which component becomes linear Bhat-like term
      lin_place: c0 or c1    # place lin_part as c0 or c1 in 2-part term
    """
    part0_rows, _ = extract_1part_rows(rt, T00_cmt_rows, part=0, coeff_count=0)
    part1_rows, _ = extract_1part_rows(rt, T00_cmt_rows, part=1, coeff_count=0)

    parts = {
        0: part0_rows,
        1: part1_rows,
    }

    ks_rows = parts[int(line5_variant["ks_part"])]
    lin_rows = parts[int(line5_variant["lin_part"])]

    line5_rows = []
    ks_inputs = []
    ks_outputs = []
    lin_terms = []

    for i in range(n):
        ks_input = rt.assemble_3part_from_1parts_coeff_ct(
            zero_rows[i],
            zero_rows[i],
            ks_rows[i],
        )
        ks_inputs.append(ks_input)

        ks_output = rt.relinearize_coeff_ct(ks_input)
        ks_outputs.append(ks_output)

        if line5_variant["lin_place"] == "c0":
            lin_term = rt.assemble_2part_from_1parts_coeff_ct(
                lin_rows[i],
                zero_rows[i],
            )
        elif line5_variant["lin_place"] == "c1":
            lin_term = rt.assemble_2part_from_1parts_coeff_ct(
                zero_rows[i],
                lin_rows[i],
            )
        else:
            raise ValueError(f"unknown lin_place: {line5_variant['lin_place']}")

        lin_terms.append(lin_term)

    # Y2m showed native add_coeff_ct is not raw-phase-linear on these
    # raw-imported / relinearized debug handles. Therefore line5 must also
    # use full-ring raw add, not add_coeff_ct.
    line5_rows = raw_add_2part_rows(rt, ks_outputs, lin_terms, n)

    return line5_rows, {
        "ks_inputs": ks_inputs,
        "ks_outputs": ks_outputs,
        "lin_terms": lin_terms,
        "line5_add": "full_ring_raw_add_2part_rows",
    }


def make_line6_pair(rt, M10_rows, M11_rows, n, line6_mapping):
    """
    normal:
      paper pair (M10,M11) => c0=M11,c1=M10

    swapped:
      paper pair (M11,M10) => c0=M10,c1=M11
    """
    if line6_mapping == "normal_M10_A_M11_B":
        return assemble_pair_rows(rt, A_rows=M10_rows, B_rows=M11_rows, n=n)

    if line6_mapping == "swapped_M11_A_M10_B":
        return assemble_pair_rows(rt, A_rows=M11_rows, B_rows=M10_rows, n=n)

    raise ValueError(f"unknown line6 mapping: {line6_mapping}")


def run_variant_case(rt, U, V, variant):
    n = U.shape[0]
    a_scalar = int(variant["a_scalar"])

    U_bundle, _ = make_controlled_rowwise_bundle(rt, U.tolist(), a_scalar=a_scalar)
    V_bundle, _ = make_controlled_rowwise_bundle(rt, V.tolist(), a_scalar=a_scalar)

    # line1: ordinary controlled RLWE matrix, so fixed phase-aware C-MT.
    U_cmt_bundle, _ = phase_aware_logical_n_cmt_oracle(
        rt=rt,
        bundle=U_bundle,
        n=n,
        a_scalar=a_scalar,
        label=f"phase_aware_CMT_U_a{a_scalar}",
    )

    comps = export_component_sources(rt, U_cmt_bundle, V_bundle, n)

    # Current paper-style component products.
    M00 = raw_rns_ppmm(comps["A_L"], comps["A_R"], n)
    M01 = raw_rns_ppmm(comps["A_L"], comps["B_R"], n)
    M10 = raw_rns_ppmm(comps["B_L"], comps["A_R"], n)
    M11 = raw_rns_ppmm(comps["B_L"], comps["B_R"], n)

    M00_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M00)
    M01_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M01)
    M10_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M10)
    M11_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M11)

    zero_rows = make_zero_1part_rows(rt, U_cmt_bundle.rows, num_towers=M00["num_towers"])

    # line3: temp pair for M01.
    T01_bundle = make_temp_pair_custom(
        rt,
        rows_x=M01_rows,
        zero_rows=zero_rows,
        n=n,
        placement=variant["T01_placement"],
        template_bundle=U_bundle,
        label=f"T01_{variant['T01_placement']}",
    )

    T01_cmt_bundle, _ = temp_pair_cmt(
        rt=rt,
        bundle=T01_bundle,
        n=n,
        a_scalar=a_scalar,
        mode=variant["temp_cmt_mode"],
        label=f"{variant['temp_cmt_mode']}_T01_{variant['T01_placement']}",
    )

    # line4: temp pair for M00.
    T00_bundle = make_temp_pair_custom(
        rt,
        rows_x=M00_rows,
        zero_rows=zero_rows,
        n=n,
        placement=variant["T00_placement"],
        template_bundle=U_bundle,
        label=f"T00_{variant['T00_placement']}",
    )

    T00_cmt_bundle, _ = temp_pair_cmt(
        rt=rt,
        bundle=T00_bundle,
        n=n,
        a_scalar=a_scalar,
        mode=variant["temp_cmt_mode"],
        label=f"{variant['temp_cmt_mode']}_T00_{variant['T00_placement']}",
    )

    # line5 variants.
    line5_rows, line5_trace = make_line5_rows(
        rt=rt,
        zero_rows=zero_rows,
        T00_cmt_rows=T00_cmt_bundle.rows,
        n=n,
        line5_variant=variant["line5_variant"],
    )

    # line6 pair variants.
    line6_pair_rows = make_line6_pair(
        rt=rt,
        M10_rows=M10_rows,
        M11_rows=M11_rows,
        n=n,
        line6_mapping=variant["line6_mapping"],
    )

    # final: full-ring raw add, after Y2n patch.
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
    line5_variants = [
        {
            "id": "ks_p1_lin_p0_as_c1",
            "ks_part": 1,
            "lin_part": 0,
            "lin_place": "c1",
        },
        {
            "id": "ks_p1_lin_p0_as_c0",
            "ks_part": 1,
            "lin_part": 0,
            "lin_place": "c0",
        },
        {
            "id": "ks_p0_lin_p1_as_c1",
            "ks_part": 0,
            "lin_part": 1,
            "lin_place": "c1",
        },
        {
            "id": "ks_p0_lin_p1_as_c0",
            "ks_part": 0,
            "lin_part": 1,
            "lin_place": "c0",
        },
    ]

    variants = []
    vid = 0

    for temp_mode in ["raw_component", "phase_aware"]:
        for T01_place in ["Aonly", "Bonly"]:
            for T00_place in ["Aonly", "Bonly"]:
                for line5 in line5_variants:
                    for line6 in ["normal_M10_A_M11_B", "swapped_M11_A_M10_B"]:
                        vid += 1
                        variants.append({
                            "variant_id": f"v{vid:03d}_{temp_mode}_T01-{T01_place}_T00-{T00_place}_{line5['id']}_{line6}",
                            "a_scalar": 1,
                            "temp_cmt_mode": temp_mode,
                            "T01_placement": T01_place,
                            "T00_placement": T00_place,
                            "line5_variant": line5,
                            "line6_mapping": line6,
                        })

    return variants


def main():
    n = 4
    rng = np.random.default_rng(209600)

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
        "experiment": "wp4y2p_line5_full_ring_raw_add_formula_sweep",
        "purpose": (
            "Sweep component formula conventions after Y2n full-ring raw add. "
            "This scans temp-pair placement, line5 part placement, and line6 pair mapping."
        ),
        "status": "debug_formula_sweep_line5_raw_add",
        "n": n,
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

                result = safe_call(
                    "run_variant_case",
                    lambda U=U, V=V, variant=variant: run_variant_case(rt, U, V, variant),
                )

                if not result["ok"]:
                    vr["errors"].append({
                        "case": case["name"],
                        "error": result.get("error"),
                    })
                    continue

                value = result["value"]
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

    # Rank by best match count, then L1 to U@V, then total errors.
    ranked = sorted(
        report["variant_results"],
        key=lambda vr: (
            -max(vr["summary"]["match_counts"].values()),
            vr["summary"]["total_l1"]["U_matmul_V"],
            vr["summary"]["num_errors"],
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
            "value": "component_formula_variant_found_for_U_matmul_V",
            "next_step": "WP4-Y2p",
            "next_goal": "Promote winning variant to full identity/random diagnostics.",
            "zh": "找到匹配 U@V 的 component formula variant；下一步扩大测试。",
        }
    elif solved_variants:
        decision = {
            "value": "component_formula_variant_found_but_reference_differs",
            "next_step": "WP4-Y2p",
            "next_goal": "Analyze winning reference orientation and fix external contract.",
            "zh": "找到稳定 variant 但参考方向不是 U@V；下一步修 orientation contract。",
        }
    else:
        decision = {
            "value": "component_formula_sweep_no_exact_variant",
            "next_step": "WP4-Y2p",
            "next_goal": "Expand sweep to PP-MM source mapping and signs, or compare against plaintext symbolic Algorithm 3.",
            "zh": "当前扫描未找到全匹配 variant；下一步扩大到 PP-MM source mapping 和符号。",
        }

    report["global_decision"] = decision

    # Keep JSON manageable: write full report, but print compact result.
    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4y2p_line5_full_ring_raw_add_formula_sweep_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    printable = {
        "experiment": report["experiment"],
        "status": report["status"],
        "checks": report["checks"],
        "global_decision": report["global_decision"],
        "saved": str(out_path),
    }

    print("=" * 100)
    print("WP4-Y2p line5 full-ring raw-add formula sweep")
    print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
