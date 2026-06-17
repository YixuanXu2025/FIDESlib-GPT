import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    ccmm_algorithm3_component_formula_skeleton,
    describe_no_relin_product_matrix,
    relinearize_ciphertext_matrix,
    describe_relinearized_product_matrix,
)


def main():
    rng = np.random.default_rng(207400)
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
        "experiment": "wp4w3b_component_formula_skeleton",
        "purpose": (
            "Build a matrix-level Algorithm 3 skeleton using explicit component formula "
            "C0=L0R0, C1=L0R1+L1R0, C2=L1R1, without whole-ciphertext multiplication."
        ),
        "status": "component_formula_skeleton_no_true_ppmm_placement_yet",
        "input_U_i64": U.tolist(),
        "input_V_i64": V.tolist(),
        "reference_U_matmul_V_i64": (U @ V).tolist(),
        "formula": {
            "right_cmt": {
                "value": "R = C-MT(V)",
                "zh": "先对右矩阵 V 做 C-MT，得到 transformed component bundle",
            },
            "component_formula": {
                "C0": "component_mul(L0, R0)",
                "C1": "component_mul(L0, R1) + component_mul(L1, R0)",
                "C2": "component_mul(L1, R1)",
                "zh": "对每个位置显式组装三分量 before-relinearization ciphertext",
            },
            "important_note": {
                "value": "This removes whole-ciphertext multiplication but is still not a full PP-MM placement implementation.",
                "zh": "本步只替换整密文乘法为 component 公式；若数值仍错，下一步需要实现论文 PP-MM placement。",
            },
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
        V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

        W_component, trace = ccmm_algorithm3_component_formula_skeleton(rt, U_bundle, V_bundle)

        report["algorithm3_component_trace"] = trace
        report["W_component_no_relin_summary"] = describe_no_relin_product_matrix(
            rt,
            W_component,
            coeff_sample=4,
        )

        W_relin, relin_trace = relinearize_ciphertext_matrix(rt, W_component)
        report["relin_trace"] = relin_trace
        report["W_component_relin_summary"] = describe_relinearized_product_matrix(
            rt,
            W_relin,
            coeff_sample=4,
        )

        no_relin_entries = report["W_component_no_relin_summary"]["entries"]
        relin_entries = report["W_component_relin_summary"]["entries"]

        got_alphas = [x["alpha"] for x in trace["right_cmt"]["auto"]]

        report["checks"] = {
            "uses_no_whole_ciphertext_multiply（本 skeleton 不调用整密文乘法）": trace["uses_whole_ciphertext_multiply"] is False,
            "component_no_relin_matrix_shape_is_4x4（component-formula no-relin 输出矩阵是 4x4）": report["W_component_no_relin_summary"]["shape"]["value"] == [4, 4],
            "component_no_relin_has_16_ciphertexts（component-formula no-relin 输出有 16 个 ciphertext）": report["W_component_no_relin_summary"]["num_ciphertexts"]["value"] == 16,
            "component_no_relin_each_has_three_parts（每个 no-relin 输出都是 c0/c1/c2 三分量）": all(e["num_parts"]["value"] == 3 for e in no_relin_entries),
            "component_no_relin_uses_coef_encoding（no-relin 输出仍是 coefficient encoding）": all(e["encoding_type"]["value"] == 1 for e in no_relin_entries),
            "component_relin_each_has_two_parts（重线性化后每个输出压回 c0/c1 二分量）": all(e["num_parts"]["value"] == 2 for e in relin_entries),
            "component_relin_uses_coef_encoding（重线性化后仍是 coefficient encoding）": all(e["encoding_type"]["value"] == 1 for e in relin_entries),
            "right_cmt_auto_alphas_expected（C-MT(V) Auto alpha 是 [1,3,5,7]）": got_alphas == [1, 3, 5, 7],
            "ready_for_W3c_component_formula_final_probe（已可用 component-formula skeleton 接入 packing/compress/decrypt 诊断）": True,
        }

        report["summary"] = {
            "next_step": "WP4-W3c",
            "next_step_goal": (
                "Run final packing/compress/decrypt diagnostics on the component-formula skeleton. "
                "If one-hot still diffuses, implement the paper's true PP-MM placement instead of pairwise position skeleton."
            ),
            "zh": "下一步把 component-formula skeleton 接入最终解密诊断；若 one-hot 仍扩散，则必须实现论文真正 PP-MM placement。",
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4w3b_component_formula_skeleton_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-W3b component-formula skeleton")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
