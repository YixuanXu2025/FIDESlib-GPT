import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    ccmm_algorithm3_no_relin_skeleton,
    describe_no_relin_product_matrix,
    relinearize_ciphertext_matrix,
    describe_relinearized_product_matrix,
)


def main():
    rng = np.random.default_rng(206600)
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
        "experiment": "wp4t2_relinearize_algorithm3_matrix",
        "purpose": (
            "Relinearize all 16 no-relin ciphertexts produced by the real Algorithm 3 skeleton: "
            "each c0/c1/c2 ciphertext becomes c0/c1."
        ),
        "status": "algorithm3_positionwise_relinearization_only_no_row_packing_no_final_decrypt",
        "input_U_i64": U.tolist(),
        "input_V_i64": V.tolist(),
        "reference_U_matmul_V_plain": (U @ V).tolist(),
        "pipeline": {
            "step1": {
                "value": "W_no_relin = Algorithm3Skeleton(U, V)",
                "zh": "先运行 Algorithm 3 skeleton，得到 4x4 个三分量位置级 ciphertext",
            },
            "step2": {
                "value": "W_relin[i][j] = Relinearize(W_no_relin[i][j])",
                "zh": "对每个位置级三分量 ciphertext 做重线性化，压回二分量",
            },
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
        V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

        W_no_relin, skeleton_trace = ccmm_algorithm3_no_relin_skeleton(rt, U_bundle, V_bundle)
        report["no_relin_summary"] = describe_no_relin_product_matrix(rt, W_no_relin, coeff_sample=4)

        W_relin, relin_trace = relinearize_ciphertext_matrix(rt, W_no_relin)
        report["relin_trace"] = relin_trace
        report["relin_summary"] = describe_relinearized_product_matrix(rt, W_relin, coeff_sample=4)

        no_relin_entries = report["no_relin_summary"]["entries"]
        relin_entries = report["relin_summary"]["entries"]

        report["checks"] = {
            "no_relin_matrix_shape_is_4x4（重线性化前矩阵形状是 4x4）": report["no_relin_summary"]["shape"]["value"] == [4, 4],
            "no_relin_has_16_ciphertexts（重线性化前共有 16 个 ciphertext）": report["no_relin_summary"]["num_ciphertexts"]["value"] == 16,
            "no_relin_each_has_three_parts（重线性化前每个 ciphertext 都是 c0/c1/c2 三分量）": all(e["num_parts"]["value"] == 3 for e in no_relin_entries),

            "relin_matrix_shape_is_4x4（重线性化后矩阵形状仍是 4x4）": report["relin_summary"]["shape"]["value"] == [4, 4],
            "relin_has_16_ciphertexts（重线性化后仍共有 16 个位置级 ciphertext）": report["relin_summary"]["num_ciphertexts"]["value"] == 16,
            "relin_consistent_shape（重线性化后所有 ciphertext 的 component/tower/ring_dim 形状一致）": report["relin_summary"]["consistent_shape"]["value"] is True,
            "relin_each_has_two_parts（重线性化后每个 ciphertext 都压回 c0/c1 二分量）": all(e["num_parts"]["value"] == 2 for e in relin_entries),
            "relin_uses_coef_encoding（重线性化后每个 ciphertext 仍是 coefficient encoding）": all(e["encoding_type"]["value"] == 1 for e in relin_entries),
            "relin_ring_dim_16384（重线性化后 ring_dim 仍是 16384）": report["relin_summary"]["expected_ring_dim"]["value"] == 16384,
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4t2_relinearize_algorithm3_matrix_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-T2 relinearize Algorithm 3 matrix report")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
