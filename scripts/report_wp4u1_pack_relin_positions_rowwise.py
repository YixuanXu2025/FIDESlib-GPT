import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    ccmm_algorithm3_no_relin_skeleton,
    relinearize_ciphertext_matrix,
    describe_relinearized_product_matrix,
    pack_relinearized_position_matrix_rowwise,
    describe_rowwise_packed_output,
)


def main():
    rng = np.random.default_rng(206700)
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
        "experiment": "wp4u1_pack_relin_positions_rowwise",
        "purpose": (
            "Pack 16 position-level relinearized coefficient ciphertexts back into "
            "4 row-wise coefficient ciphertexts using X^j placement and ciphertext addition."
        ),
        "status": "rowwise_packing_probe_only_no_final_decrypt",
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
                "zh": "每个位置级 ciphertext 从三分量压回二分量",
            },
            "step3": {
                "value": "W_row[i] = sum_j X^j * W_relin[i][j]",
                "zh": "每一行的 4 个位置级二分量 ciphertext 通过 X^j placement 后相加，合成 1 个 row-wise ciphertext",
            },
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
        V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

        W_no_relin, skeleton_trace = ccmm_algorithm3_no_relin_skeleton(rt, U_bundle, V_bundle)
        W_relin, relin_trace = relinearize_ciphertext_matrix(rt, W_no_relin)

        report["relin_summary"] = describe_relinearized_product_matrix(rt, W_relin, coeff_sample=4)

        W_rows, pack_trace = pack_relinearized_position_matrix_rowwise(rt, W_relin)
        report["pack_trace"] = pack_trace
        report["rowwise_output"] = describe_rowwise_packed_output(rt, W_rows, coeff_sample=4)

        relin_entries = report["relin_summary"]["entries"]
        row_entries = report["rowwise_output"]["rows"]

        expected_pack_exps = [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]
        got_pack_exps = [
            [term["exp"] for term in row["terms"]]
            for row in pack_trace
        ]

        report["checks"] = {
            "input_relin_matrix_shape_is_4x4（输入的重线性化位置级矩阵是 4x4）": report["relin_summary"]["shape"]["value"] == [4, 4],
            "input_relin_has_16_ciphertexts（输入有 16 个位置级二分量 ciphertext）": report["relin_summary"]["num_ciphertexts"]["value"] == 16,
            "input_relin_each_has_two_parts（输入每个位置级 ciphertext 都是 c0/c1 二分量）": all(e["num_parts"]["value"] == 2 for e in relin_entries),

            "packed_output_has_4_rows（packing 后输出 4 个 row-wise ciphertext）": report["rowwise_output"]["num_rows"]["value"] == 4,
            "packed_output_consistent_shape（packing 后所有 row ciphertext 形状一致）": report["rowwise_output"]["consistent_shape"]["value"] is True,
            "packed_output_each_has_two_parts（packing 后每个 row ciphertext 仍是 c0/c1 二分量）": all(e["num_parts"]["value"] == 2 for e in row_entries),
            "packed_output_uses_coef_encoding（packing 后每个 row ciphertext 仍是 coefficient encoding）": all(e["encoding_type"]["value"] == 1 for e in row_entries),
            "packed_output_ring_dim_16384（packing 后 ring_dim 仍是 16384）": report["rowwise_output"]["expected_ring_dim"]["value"] == 16384,
            "pack_exponents_are_expected（packing placement 指数符合每行 [0,1,2,3]）": got_pack_exps == expected_pack_exps,
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4u1_pack_relin_positions_rowwise_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-U1 pack relinearized positions into row-wise ciphertexts")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
