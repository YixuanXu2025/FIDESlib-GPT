import gc
import json
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from report_wp3e2_matrix_role_tile_64x768_he_pcmm import (
    make_matrices,
    component_he_matmul_return_rows,
    he_add_rows,
    materialize_and_decrypt,
    max_abs_err,
)


def now_us():
    return time.perf_counter() * 1e6


def print_sep(title):
    print("=" * 100)
    print(title)


def run_role_specific_streaming(rt, X, W):
    """
    Streaming role-specific HE PCMM.

    原始 role-specific 版本会把一个 output block 的所有 partial rows 先保存起来：
      partial_sets = [partial0, partial1, ..., partial11]

    这会占用大量 GPU memory。

    本版本改成：
      acc = first partial
      acc = acc + next partial
      acc = acc + next partial
      ...
    边算边加，减少同时存活的 GPU ciphertext rows。
    """
    block = 128
    sub = 64

    num_mid_blocks = X.shape[1] // block
    num_out_blocks = W.shape[1] // block

    plain_t0 = now_us()
    Y_ref = X @ W
    plain_t1 = now_us()

    Y_blocks = []

    total_encrypt = 0.0
    total_compute = 0.0
    total_add = 0.0
    total_materialize = 0.0
    total_decrypt = 0.0

    component_products = 0
    output_block_reports = []

    for j in range(num_out_blocks):
        print_sep(f"role-specific output block {j}/{num_out_blocks - 1}")

        acc_rows = None
        num_partials = 0

        for k in range(num_mid_blocks):
            X_block = X[:, k * block:(k + 1) * block]
            W_block = W[k * block:(k + 1) * block, j * block:(j + 1) * block]

            parts = [
                (X_block[:, :sub], W_block[:sub, :]),
                (X_block[:, sub:block], W_block[sub:block, :]),
            ]

            for part_idx, (X_part, W_part) in enumerate(parts):
                print(f"[partial] output_block={j}, mid_block={k}, part={part_idx}", flush=True)

                result = component_he_matmul_return_rows(rt, X_part, W_part)

                total_encrypt += result["encrypt_us"]
                total_compute += result["component_compute_us"]
                component_products += 1
                num_partials += 1

                if acc_rows is None:
                    acc_rows = result["rows"]
                else:
                    acc_rows, add_us = he_add_rows(rt, acc_rows, result["rows"])
                    total_add += add_us

                # 释放 Python 引用，尽量帮助 CUDA/FIDESlib 及时回收
                del result
                gc.collect()

        Y_block_dec, md = materialize_and_decrypt(rt, acc_rows, logical_length=X.shape[0])
        total_materialize += md["materialize_us"]
        total_decrypt += md["decrypt_us"]

        Y_blocks.append(Y_block_dec)

        Y_block_ref = Y_ref[:, j * block:(j + 1) * block]
        block_err = max_abs_err(Y_block_dec, Y_block_ref)

        output_block_reports.append({
            "output_block_index（输出块编号）": j,
            "num_partial_products（partial乘法数量）": num_partials,
            "Y_block_shape（输出块形状）": list(Y_block_dec.shape),
            "max_abs_err（最大绝对误差）": block_err,
        })

        print(f"[done block] output_block={j}, err={block_err:.6e}", flush=True)

        del acc_rows
        gc.collect()

    Y_dec = np.hstack(Y_blocks)
    err = max_abs_err(Y_dec, Y_ref)

    report = {
        "experiment（实验）": "role_specific_block_tiling_64x768_he_pcmm_streaming",
        "method（方法）": "streaming role-specific component HE backend; activation tiles 64x64 effective, weight tiles 64x128",
        "target（目标）": "Y = X(64x768) @ W(768x768)",
        "X_shape（X形状）": list(X.shape),
        "W_shape（W形状）": list(W.shape),
        "Y_shape（Y形状）": list(Y_dec.shape),
        "logical_block_shape（逻辑块形状）": [128, 128],
        "activation_tile_shape_effective（实际输入/激活tile形状）": [64, 64],
        "weight_tile_shape（权重tile形状）": [64, 128],
        "num_mid_blocks（中间维度块数）": num_mid_blocks,
        "num_out_blocks（输出维度块数）": num_out_blocks,
        "component_products（component HE partial乘法数量）": component_products,
        "plain_reference_us（明文参考用时）": plain_t1 - plain_t0,
        "encrypt_us_total（总加密用时）": total_encrypt,
        "component_compute_us_total（总HE核心计算用时）": total_compute,
        "he_add_partial_us_total（partial结果HE加法总用时）": total_add,
        "materialize_us_total（总materialize用时）": total_materialize,
        "decrypt_us_total（总解密用时）": total_decrypt,
        "total_he_us（HE总用时）": total_encrypt + total_compute + total_add + total_materialize + total_decrypt,
        "max_abs_err（最大绝对误差）": err,
        "output_block_reports（各输出块报告）": output_block_reports,
        "Y_ref_first_row_first8（明文首行前8个）": Y_ref[0, :8].tolist(),
        "Y_dec_first_row_first8（解密首行前8个）": Y_dec[0, :8].tolist(),
    }

    return report


def main():
    X, W = make_matrices(seed=206768)

    cfg = HEConfig(
        ring_dim=16384,
        multiplicative_depth=2,
        scaling_mod_size=50,
        batch_size=8192,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    with HERuntime(cfg, rotation_steps=()) as rt:
        print("runtime info（运行时信息）:", rt.info())
        report = run_role_specific_streaming(rt, X, W)

    print_sep("WP3-E2 role-specific streaming report")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_path = Path("/workspace/FIDESlib-GPT/reports/wp3e2_role_specific_64x768_streaming_report.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
