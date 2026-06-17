import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    describe_rowwise_coeff_ciphertext_matrix,
    cmt_algorithm2_rowwise,
    describe_cmt_output,
)


def main():
    rng = np.random.default_rng(207600)
    U = rng.integers(-2, 3, size=(4, 4)).astype(np.int64)
    V = rng.integers(-2, 3, size=(4, 4)).astype(np.int64)

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
        "experiment": "wp4x1_paper_algorithm3_corrected_contract",
        "purpose": (
            "Correct the real CCMM implementation contract to match Park-2025 Algorithm 3. "
            "This explicitly rejects the previous right-CMT/pairwise-position skeleton."
        ),
        "status": "corrected_contract_only_no_true_ppmm_yet",
        "input_U_i64": U.tolist(),
        "input_V_i64": V.tolist(),
        "reference_U_matmul_V_i64": (U @ V).tolist(),
        "why_this_step": {
            "W3c_result": {
                "value": "component-formula skeleton decrypted but all one-hot/identity tests failed with dense outputs",
                "zh": "W3c 证明 C0/C1/C2 公式可运行，但 pairwise skeleton 仍不是论文 PP-MM",
            },
            "main_correction": {
                "value": "Algorithm 3 transposes U first, not V",
                "zh": "论文 Algorithm 3 第一步是 C-MT(U)，不是 C-MT(V)",
            },
        },
        "paper_algorithm3_contract": {
            "line1": {
                "value": "(A_U, B_U) = Transpose(ct_U)",
                "zh": "对左矩阵 U 的 row-wise ciphertext bundle 做 C-MT，得到 column-wise/transformed U components",
            },
            "line2": {
                "value": "[[M00, M01], [M10, M11]] = [[A_U], [B_U]] @ [A_V, B_V]",
                "zh": "做一次大的 modular PP-MM，产生四个 component matrix",
            },
            "line2_expanded": {
                "M00": "A_U @ A_V",
                "M01": "A_U @ B_V",
                "M10": "B_U @ A_V",
                "M11": "B_U @ B_V",
                "zh": "这里 @ 是 coefficient/RNS matrix multiplication，不是 DCRTPoly polynomial multiplication，也不是 EvalMult",
            },
            "line3": {
                "value": "(A_check, B_check) = Transpose((M01, 0))",
                "zh": "对临时 component pair (M01, 0) 做 C-MT",
            },
            "line4": {
                "value": "(A_hat, B_hat) = Transpose((M00, 0))",
                "zh": "对临时 component pair (M00, 0) 做 C-MT",
            },
            "line5": {
                "value": "(A_hat, B_hat) = KS_{s^2 -> s}((A_hat, 0)) + (B_hat, 0)",
                "zh": "把 sk^2 项 key-switch/relinearize 回 sk 项",
            },
            "line6": {
                "value": "(A_W, B_W) = Rescale((A_hat,B_hat) + (A_check,B_check) + (M10,M11))",
                "zh": "合并三部分并 rescale，得到 row-wise output encryption",
            },
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
        V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

        # Correct Algorithm 3 direction:
        # Transpose U, not V.
        U_cmt_rows, U_cmt_trace = cmt_algorithm2_rowwise(rt, U_bundle)

        report["U_bundle_rowwise"] = describe_rowwise_coeff_ciphertext_matrix(U_bundle)
        report["V_bundle_rowwise"] = describe_rowwise_coeff_ciphertext_matrix(V_bundle)
        report["U_after_cmt"] = describe_cmt_output(rt, U_cmt_rows, coeff_sample=4)

        report["component_sources_for_ppmm"] = {
            "A_U": {
                "value": "U_after_cmt[*].GetElements()[1]",
                "zh": "C-MT(U) 后的 A component matrix；代码里 c1/GetElements()[1]",
            },
            "B_U": {
                "value": "U_after_cmt[*].GetElements()[0]",
                "zh": "C-MT(U) 后的 B component matrix；代码里 c0/GetElements()[0]",
            },
            "A_V": {
                "value": "V_bundle.rows[*].GetElements()[1]",
                "zh": "V row-wise 的 A component matrix；代码里 c1/GetElements()[1]",
            },
            "B_V": {
                "value": "V_bundle.rows[*].GetElements()[0]",
                "zh": "V row-wise 的 B component matrix；代码里 c0/GetElements()[0]",
            },
        }

        got_alphas = [x["alpha"] for x in U_cmt_trace["auto"]]

        report["checks"] = {
            "U_bundle_has_4_rows（U 输入是 4 个 row-wise ciphertext）": U_bundle.shape == (4, 4),
            "V_bundle_has_4_rows（V 输入是 4 个 row-wise ciphertext）": V_bundle.shape == (4, 4),
            "algorithm3_transposes_U_not_V（Algorithm 3 当前合同已改为 C-MT(U)）": True,
            "U_cmt_has_4_rows（C-MT(U) 输出 4 个 transformed ciphertext）": report["U_after_cmt"]["num_rows"] == 4,
            "U_cmt_has_two_parts（C-MT(U) 每个 ciphertext 仍有 c0/c1）": all(
                r["num_parts"]["value"] == 2 for r in report["U_after_cmt"]["rows"]
            ),
            "U_cmt_auto_alphas_expected（C-MT(U) Auto alpha 是 [1,3,5,7]）": got_alphas == [1, 3, 5, 7],
            "ppmm_must_be_raw_component_matrix_multiply（下一步必须做 raw coefficient/RNS PP-MM）": True,
            "old_right_cmt_pairwise_skeleton_rejected（旧 right-CMT pairwise skeleton 已排除）": True,
        }

        report["missing_primitives_for_true_algorithm3"] = {
            "raw_component_export": {
                "value": "export selected c0/c1 DCRTPoly components as coefficient/RNS matrices",
                "zh": "需要把 ciphertext component 导出成真正可做矩阵乘的 coefficient/RNS matrix",
            },
            "raw_component_ppmm": {
                "value": "Mij = modular matrix multiplication over raw RNS coefficient matrices",
                "zh": "需要对每个 RNS tower 做模 q 的矩阵乘法；这才是论文 PP-MM",
            },
            "raw_component_import": {
                "value": "import PP-MM result matrices back into coefficient ciphertext-like component bundles",
                "zh": "需要把 PP-MM 结果重新封装为 (M01,0)、(M00,0)、(M10,M11) 这类临时 bundle",
            },
            "temporary_pair_cmt": {
                "value": "run C-MT on temporary component pair bundles (M01,0) and (M00,0)",
                "zh": "不能只对普通 encrypted matrix bundle 做 C-MT，还要能对临时 component pair 做 C-MT",
            },
        }

        report["summary"] = {
            "current_result": "Algorithm 3 contract corrected to paper direction.",
            "next_step": "WP4-X2",
            "next_step_goal": (
                "Implement raw component export/import probes so PP-MM can operate on "
                "coefficient/RNS matrices instead of DCRTPoly polynomial products."
            ),
            "zh": (
                "当前修正的是算法合同；下一步要暴露 raw component matrix，真正实现论文 PP-MM。"
            ),
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x1_paper_algorithm3_corrected_contract_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X1 paper Algorithm 3 corrected contract")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
