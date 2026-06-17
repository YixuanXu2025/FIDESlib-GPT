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


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def one_hot(n, i, j, value=1):
    M = np.zeros((n, n), dtype=np.int64)
    M[i, j] = value
    return M


def temp_pair_cmt(rt, bundle, n, a_scalar, mode, label):
    if mode == "raw_component":
        out = logical_n_cmt_oracle(rt, bundle, n, label=label)
        return out, {
            "mode": mode,
            "zh": "对 temporary component-pair 做 raw component transpose，不按 RLWE phase 解密。",
        }

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


def export_component_sources(rt, left_bundle, right_bundle, n):
    return {
        "A_L": normalize_export(rt.export_component_coeff_matrix_u64(left_bundle.rows, part=1, coeff_count=n)),
        "B_L": normalize_export(rt.export_component_coeff_matrix_u64(left_bundle.rows, part=0, coeff_count=n)),
        "A_R": normalize_export(rt.export_component_coeff_matrix_u64(right_bundle.rows, part=1, coeff_count=n)),
        "B_R": normalize_export(rt.export_component_coeff_matrix_u64(right_bundle.rows, part=0, coeff_count=n)),
    }


def run_pipeline(rt, U, V, a_scalar, temp_cmt_mode):
    n = U.shape[0]

    U_bundle, U_info = make_controlled_rowwise_bundle(rt, U.tolist(), a_scalar=a_scalar)
    V_bundle, V_info = make_controlled_rowwise_bundle(rt, V.tolist(), a_scalar=a_scalar)

    U_dec = decrypt_rows_after_compress(rt, U_bundle.rows, logical_length=n)
    V_dec = decrypt_rows_after_compress(rt, V_bundle.rows, logical_length=n)

    # line1: real RLWE ciphertext matrix, so use phase-aware C-MT oracle.
    U_cmt_bundle, U_cmt_trace = phase_aware_logical_n_cmt_oracle(
        rt,
        U_bundle,
        n=n,
        a_scalar=a_scalar,
        label=f"phase_aware_CMT_U_a{a_scalar}",
    )

    # line2: raw RNS PP-MM on components.
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

    # line3: C-MT((M01,0)) with swept temp mode.
    T01_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M01_rows,
        B_rows=zero_rows,
        n=n,
        label="T01=(M01,0)",
        template_bundle=U_bundle,
    )
    T01_cmt_bundle, T01_cmt_trace = temp_pair_cmt(
        rt,
        T01_bundle,
        n=n,
        a_scalar=a_scalar,
        mode=temp_cmt_mode,
        label=f"{temp_cmt_mode}_CMT_T01_a{a_scalar}",
    )

    # line4: C-MT((M00,0)) with swept temp mode.
    T00_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M00_rows,
        B_rows=zero_rows,
        n=n,
        label="T00=(M00,0)",
        template_bundle=U_bundle,
    )
    T00_cmt_bundle, T00_cmt_trace = temp_pair_cmt(
        rt,
        T00_bundle,
        n=n,
        a_scalar=a_scalar,
        mode=temp_cmt_mode,
        label=f"{temp_cmt_mode}_CMT_T00_a{a_scalar}",
    )

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

        # paper (B_hat,0) -> OpenFHE c0=0,c1=B_hat.
        bhat_term = rt.assemble_2part_from_1parts_coeff_ct(
            zero_rows[i],
            Bhat_rows[i],
        )
        line5_bhat_terms.append(bhat_term)

        line5_rows.append(rt.add_coeff_ct(ks_output, bhat_term))

    # line6: (M10,M11), OpenFHE c0=M11,c1=M10.
    M10_M11_pair_rows = [
        rt.assemble_2part_from_1parts_coeff_ct(M11_rows[i], M10_rows[i])
        for i in range(n)
    ]

    tmp = raw_add_2part_rows(rt, line5_rows, T01_cmt_bundle.rows, n)
    final_rows = raw_add_2part_rows(rt, tmp, M10_M11_pair_rows, n)

    final = decrypt_rows_after_compress(rt, final_rows, logical_length=n)

    trace = {
        "U_info": U_info,
        "V_info": V_info,
        "U_decrypt_exact": compare_matrix(U_dec["decoded"], U),
        "V_decrypt_exact": compare_matrix(V_dec["decoded"], V),
        "temp_cmt_mode": temp_cmt_mode,
        "U_cmt_trace": U_cmt_trace,
        "T01_cmt_trace": T01_cmt_trace,
        "T00_cmt_trace": T00_cmt_trace,
        "Ahat_export": Ahat_export,
        "Bhat_export": Bhat_export,
        "line5_ks_inputs_summary": inspect_rows(rt, line5_ks_inputs, coeff_sample=4),
        "line5_ks_outputs_summary": inspect_rows(rt, line5_ks_outputs, coeff_sample=4),
        "line5_rows_summary": inspect_rows(rt, line5_rows, coeff_sample=4),
        "T01_cmt_rows_summary": inspect_rows(rt, T01_cmt_bundle.rows, coeff_sample=4),
        "M10_M11_pair_rows_summary": inspect_rows(rt, M10_M11_pair_rows, coeff_sample=4),
        "final_rows_summary": inspect_rows(rt, final_rows, coeff_sample=4),
    }

    return final, trace


def summarize_against_references(got, refs):
    return {
        name: compare_matrix(got, ref)
        for name, ref in refs.items()
    }


def main():
    n = 4
    rng = np.random.default_rng(209400)

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

    a_scalars = [1]
    temp_cmt_modes = [
        "raw_component",
        "phase_aware",
    ]

    ref_names = [
        "U_matmul_V",
        "U_T_matmul_V",
        "U_matmul_V_T",
        "V_matmul_U",
        "V_T_matmul_U",
        "U_T_matmul_V_T",
    ]

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
        "experiment": "wp4y2j_temp_pair_cmt_mode_sweep",
        "purpose": (
            "Sweep C-MT semantics for Algorithm 3 temporary pairs (M01,0) and (M00,0). "
            "Line1 C-MT(U) remains phase-aware because U is a real RLWE ciphertext matrix. "
            "Temp pairs are tested with raw component C-MT vs phase-aware C-MT."
        ),
        "status": "debug_temp_pair_cmt_mode_sweep",
        "n": n,
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for a_scalar in a_scalars:
            for temp_mode in temp_cmt_modes:
                for case in cases:
                    U = case["U"]
                    V = case["V"]

                    final, trace = run_pipeline(
                        rt,
                        U,
                        V,
                        a_scalar=a_scalar,
                        temp_cmt_mode=temp_mode,
                    )

                    refs = {
                        "U_matmul_V": U @ V,
                        "U_T_matmul_V": U.T @ V,
                        "U_matmul_V_T": U @ V.T,
                        "V_matmul_U": V @ U,
                        "V_T_matmul_U": V.T @ U,
                        "U_T_matmul_V_T": U.T @ V.T,
                    }

                    comparisons = summarize_against_references(final["decoded"], refs)

                    report["cases"].append({
                        "name": case["name"],
                        "a_scalar": int(a_scalar),
                        "temp_cmt_mode": temp_mode,
                        "input_U": U.tolist(),
                        "input_V": V.tolist(),
                        "references": {k: v.tolist() for k, v in refs.items()},
                        "final": final,
                        "trace": trace,
                        "comparisons": comparisons,
                    })

    # Aggregate.
    total_by_mode = {}
    for temp_mode in temp_cmt_modes:
        cs = [c for c in report["cases"] if c["temp_cmt_mode"] == temp_mode]
        total_by_mode[temp_mode] = {
            "num_cases": len(cs),
            "match_counts": {
                ref: sum(1 for c in cs if c["comparisons"][ref]["exact"])
                for ref in ref_names
            },
            "all_decrypted": all(c["comparisons"]["U_matmul_V"]["all_rows_decrypted"] for c in cs),
            "all_line5_ks_inputs_are_3part": all(
                rows_are_npart(c["trace"]["line5_ks_inputs_summary"], n, 3)
                for c in cs
            ),
            "all_final_rows_are_2part": all(
                rows_are_npart(c["trace"]["final_rows_summary"], n, 2)
                for c in cs
            ),
        }

    best = None
    for mode, s in total_by_mode.items():
        for ref, cnt in s["match_counts"].items():
            item = {
                "temp_cmt_mode": mode,
                "reference": ref,
                "match_count": cnt,
                "num_cases": s["num_cases"],
            }
            if best is None or item["match_count"] > best["match_count"]:
                best = item

    report["checks"] = {
        "summary_by_temp_cmt_mode（按 temporary pair C-MT 模式汇总）": total_by_mode,
        "best_mode_and_reference（最佳模式和参考）": best,
        "any_mode_solves_U_matmul_V（是否有模式全部匹配 U@V）": any(
            s["match_counts"]["U_matmul_V"] == s["num_cases"]
            for s in total_by_mode.values()
        ),
        "any_mode_solves_any_reference（是否有模式全部匹配某个参考）": any(
            any(cnt == s["num_cases"] for cnt in s["match_counts"].values())
            for s in total_by_mode.values()
        ),
    }

    if report["checks"]["any_mode_solves_U_matmul_V（是否有模式全部匹配 U@V）"]:
        decision = {
            "value": "temp_cmt_mode_found_for_U_matmul_V",
            "best": best,
            "next_step": "WP4-Y3",
            "zh": "找到 temporary pair C-MT 语义，使 full pipeline 匹配 U@V。",
        }
    elif report["checks"]["any_mode_solves_any_reference（是否有模式全部匹配某个参考）"]:
        decision = {
            "value": "temp_cmt_mode_found_but_orientation_differs",
            "best": best,
            "next_step": "WP4-Y2k",
            "zh": "找到稳定语义但方向不是 U@V；下一步修 orientation contract。",
        }
    else:
        decision = {
            "value": "temp_cmt_mode_sweep_not_sufficient",
            "best": best,
            "next_step": "WP4-Y2k",
            "next_goal": "Trace plaintext phase of M00/M01/M10/M11, line5, T01, and final output to locate formula mismatch.",
            "zh": "raw vs phase-aware temp C-MT 仍不足以修复；下一步逐项解密 line2-line6 中间项定位公式差异。",
        }

    report["global_decision"] = decision

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4y2j_temp_pair_cmt_mode_sweep_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Y2j temp-pair C-MT mode sweep")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
