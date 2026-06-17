import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    ccmm_algorithm3_component_formula_skeleton,
    relinearize_ciphertext_matrix,
    pack_relinearized_position_matrix_rowwise,
    describe_rowwise_packed_output,
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


def run_component_formula_pipeline(rt, U, V):
    U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
    V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

    W_component, component_trace = ccmm_algorithm3_component_formula_skeleton(
        rt,
        U_bundle,
        V_bundle,
    )

    W_relin, relin_trace = relinearize_ciphertext_matrix(rt, W_component)
    W_rows, pack_trace = pack_relinearized_position_matrix_rowwise(rt, W_relin)

    W_compressed = [
        rt.compress_coeff_ct(ct, towers_left=1)
        for ct in W_rows
    ]

    packed_summary = describe_rowwise_packed_output(rt, W_compressed, coeff_sample=4)

    dec_rows = []
    row_reports = []

    for i, ct in enumerate(W_compressed):
        dec = safe_call(
            f"decrypt_row_{i}",
            lambda ct=ct: rt.decrypt_coeff_row_i64(ct, logical_length=4),
        )

        row_reports.append({
            "row_index": i,
            "decrypt_ok": dec["ok"],
            "error": dec.get("error"),
            "value": dec.get("value") if dec["ok"] else None,
        })

        if dec["ok"]:
            dec_rows.append(list(map(int, dec["value"])))
        else:
            dec_rows.append(None)

    return {
        "decrypted_rows": dec_rows,
        "row_reports": row_reports,
        "packed_summary": packed_summary,
        "component_trace_summary": {
            "uses_whole_ciphertext_multiply": component_trace.get("uses_whole_ciphertext_multiply"),
            "right_cmt_auto_alphas": [
                x["alpha"] for x in component_trace["right_cmt"]["auto"]
            ],
        },
        "pack_trace_exps": [
            [term["exp"] for term in row["terms"]]
            for row in pack_trace
        ],
    }


def main():
    n = 4
    rng = np.random.default_rng(207500)

    cases_spec = [
        {
            "name": "basis_E00_times_E00",
            "U": one_hot(n, 0, 0),
            "V": one_hot(n, 0, 0),
            "zh": "理论输出 E00；检查最简单 one-hot 是否扩散",
        },
        {
            "name": "basis_E01_times_E10",
            "U": one_hot(n, 0, 1),
            "V": one_hot(n, 1, 0),
            "zh": "理论输出 E00；检查内积索引是否对齐",
        },
        {
            "name": "basis_E01_times_E12",
            "U": one_hot(n, 0, 1),
            "V": one_hot(n, 1, 2),
            "zh": "理论输出 E02；检查列 placement",
        },
        {
            "name": "basis_E23_times_E31",
            "U": one_hot(n, 2, 3),
            "V": one_hot(n, 3, 1),
            "zh": "理论输出 E21；检查行列 orientation",
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
        "experiment": "wp4w3c_component_formula_final_probe",
        "purpose": (
            "Run final packing/compress/decrypt diagnostics on the component-formula "
            "Algorithm 3 skeleton. This verifies whether explicit C0/C1/C2 assembly fixes "
            "the V3 dense one-hot failure."
        ),
        "status": "component_formula_final_diagnostic_not_true_ppmm_yet",
        "cases": [],
        "pipeline": {
            "step1": {
                "value": "W_component = ccmm_algorithm3_component_formula_skeleton(U,V)",
                "zh": "使用显式 C0/C1/C2 component 公式，不调用整密文乘法",
            },
            "step2": {
                "value": "W_relin = relinearize_ciphertext_matrix(W_component)",
                "zh": "每个位置从 c0/c1/c2 压回 c0/c1",
            },
            "step3": {
                "value": "W_rows = pack_relinearized_position_matrix_rowwise(W_relin)",
                "zh": "把 16 个位置级 ciphertext pack 成 4 个 row-wise ciphertext",
            },
            "step4": {
                "value": "Compress(..., towers_left=1) then decrypt_coeff_row_i64",
                "zh": "压缩到 1 个 RNS tower 后最终解密",
            },
        },
        "expected_interpretation": {
            "if_exact": {
                "value": "component formula skeleton fixed the previous issue",
                "zh": "如果 one-hot 和 identity 用例通过，则 component-formula skeleton 至少小尺寸可用",
            },
            "if_dense_wrong": {
                "value": "pairwise skeleton is still not true PP-MM placement",
                "zh": "如果 one-hot 仍扩散，说明必须实现论文真正 PP-MM placement/indexing",
            },
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for spec in cases_spec:
            U = spec["U"]
            V = spec["V"]
            W_ref = U @ V

            result = run_component_formula_pipeline(rt, U, V)
            analysis = summarize_matrix_error(result["decrypted_rows"], W_ref)

            case = {
                "name": spec["name"],
                "zh": spec["zh"],
                "input_U": U.tolist(),
                "input_V": V.tolist(),
                "reference_U_matmul_V": W_ref.tolist(),
                "decrypted_output": result["decrypted_rows"],
                "row_reports": result["row_reports"],
                "packed_summary": result["packed_summary"],
                "component_trace_summary": result["component_trace_summary"],
                "pack_trace_exps": result["pack_trace_exps"],
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

    all_decrypted = all(
        c["analysis"]["all_rows_decrypted"]
        for c in report["cases"]
    )

    one_hot_dense_failures = []
    for c in report["cases"]:
        if c["name"].startswith("basis_") and not c["analysis"]["exact"]:
            got_nnz = c["analysis"]["nonzero_positions_got"]
            ref_nnz = c["analysis"]["nonzero_positions_ref"]
            one_hot_dense_failures.append({
                "name": c["name"],
                "got_nonzero_count": None if got_nnz is None else len(got_nnz),
                "ref_nonzero_count": len(ref_nnz),
            })

    report["checks"] = {
        "all_cases_decrypted（所有测试用例都成功解密）": all_decrypted,
        "component_trace_uses_no_whole_ciphertext_multiply（component skeleton 未使用整密文乘法）": all(
            c["component_trace_summary"]["uses_whole_ciphertext_multiply"] is False
            for c in report["cases"]
        ),
        "all_right_cmt_auto_alphas_expected（所有 case 的 C-MT alpha 都是 [1,3,5,7]）": all(
            c["component_trace_summary"]["right_cmt_auto_alphas"] == [1, 3, 5, 7]
            for c in report["cases"]
        ),
        "exact_case_count（完全等于 U@V 的 case 数量）": len(exact_cases),
        "failed_case_count（不等于 U@V 的 case 数量）": len(failed_cases),
        "one_hot_dense_failure_summary（one-hot 失败时输出非零数量诊断）": one_hot_dense_failures,
    }

    report["summary"] = {
        "exact_cases": exact_cases,
        "failed_cases": failed_cases,
        "next_step_if_all_exact": "Promote component-formula skeleton to experimental CCMM API.",
        "next_step_if_dense_wrong": "WP4-X1: implement true PP-MM placement/indexing from the paper, not pairwise position skeleton.",
        "zh": (
            "如果 one-hot 仍然密集扩散，则 C0/C1/C2 公式可运行但 pairwise skeleton 仍不是论文真正 PP-MM。"
        ),
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4w3c_component_formula_final_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-W3c component-formula final probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
