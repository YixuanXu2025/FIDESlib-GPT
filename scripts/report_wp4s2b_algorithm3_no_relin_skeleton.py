import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    describe_rowwise_coeff_ciphertext_matrix,
    ccmm_algorithm3_no_relin_skeleton,
    describe_no_relin_product_matrix,
)


def main():
    rng = np.random.default_rng(206500)
    U = rng.integers(-3, 4, size=(4, 4)).astype(np.int64)
    V = rng.integers(-3, 4, size=(4, 4)).astype(np.int64)

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
        "experiment": "wp4s2b_algorithm3_no_relin_skeleton",
        "purpose": (
            "Construct real Park-2025 Algorithm 3 no-relinearization skeleton: "
            "right C-MT plus pairwise 2-component×2-component ciphertext multiplication."
        ),
        "status": "algorithm3_skeleton_only_no_relin_no_rescale_no_final_decrypt",
        "input_U_i64": U.tolist(),
        "input_V_i64": V.tolist(),
        "reference_U_matmul_V_plain": (U @ V).tolist(),
        "formula": {
            "right_cmt": {
                "value": "V_cmt = C-MT(V)",
                "zh": "先把右矩阵 V 从 row-wise bundle 变换成 C-MT 后的 column-like/transformed bundle",
            },
            "no_relin_product": {
                "value": "W_no_relin[i][j] = mul_no_relin(U.rows[i], V_cmt[j])",
                "zh": "每个输出位置先生成一个三分量 before-relinearization ciphertext",
            },
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
        V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

        report["U_bundle"] = describe_rowwise_coeff_ciphertext_matrix(U_bundle)
        report["V_bundle"] = describe_rowwise_coeff_ciphertext_matrix(V_bundle)

        W_no_relin, trace = ccmm_algorithm3_no_relin_skeleton(rt, U_bundle, V_bundle)

        report["algorithm3_trace"] = trace
        report["W_no_relin_summary"] = describe_no_relin_product_matrix(rt, W_no_relin, coeff_sample=4)

        entries = report["W_no_relin_summary"]["entries"]

        report["checks"] = {
            "output_matrix_shape_is_4x4（no-relin 输出矩阵形状是 4x4）": report["W_no_relin_summary"]["shape"]["value"] == [4, 4],
            "output_has_16_ciphertexts（no-relin 输出共有 16 个 ciphertext）": report["W_no_relin_summary"]["num_ciphertexts"]["value"] == 16,
            "output_consistent_shape（所有 no-relin 输出 component/tower/ring_dim 形状一致）": report["W_no_relin_summary"]["consistent_shape"]["value"] is True,
            "output_has_three_rlwe_parts（每个 no-relin 输出有 c0/c1/c2 三个 RLWE component）": all(e["num_parts"]["value"] == 3 for e in entries),
            "output_uses_coef_encoding（每个 no-relin 输出仍是 coefficient encoding）": all(e["encoding_type"]["value"] == 1 for e in entries),
            "output_ring_dim_16384（no-relin 输出 ring_dim 仍是 16384）": report["W_no_relin_summary"]["expected_ring_dim"]["value"] == 16384,
            "right_cmt_auto_alphas_are_expected（右矩阵 C-MT 的 Auto alpha 符合 2*j+1）": [
                item["alpha"] for item in trace["right_cmt"]["auto"]
            ] == [1, 3, 5, 7],
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4s2b_algorithm3_no_relin_skeleton_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-S2b Algorithm 3 no-relin skeleton report")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
