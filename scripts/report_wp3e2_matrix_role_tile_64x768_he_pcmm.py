import json
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def now_us():
    return time.perf_counter() * 1e6


def print_sep(title):
    print("=" * 100)
    print(title)


def max_abs_err(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def as_int_matrix(x):
    return np.asarray(x, dtype=np.int64)


def decrypt_rows_as_matrix(rt, rows, logical_length):
    """
    rows represent Y.T rows.
    Decrypt stack gives Y.T, then transpose to Y.
    """
    dec_rows = []
    for r in rows:
        dec = rt.decrypt(r, logical_length=logical_length)
        dec_rows.append(np.asarray(dec, dtype=np.float64))
    y_t = np.stack(dec_rows, axis=0)
    return y_t.T.copy()


def call_component_linear_transform(rt, rows, U):
    if hasattr(rt, "component_linear_transform_gpu"):
        return rt.component_linear_transform_gpu(rows, U)

    ctx = rt.require_context()
    if hasattr(ctx, "component_linear_transform_gpu"):
        return ctx.component_linear_transform_gpu(rows, U)

    if hasattr(ctx, "_ctx") and hasattr(ctx._ctx, "component_linear_matmul_gpu_fused_raw"):
        return ctx._ctx.component_linear_matmul_gpu_fused_raw(rows, U, False)

    raise RuntimeError("No component linear transform API found")


def call_materialize(rt, rows):
    if hasattr(rt, "materialize_gpu_ciphertexts"):
        return rt.materialize_gpu_ciphertexts(rows)

    ctx = rt.require_context()
    if hasattr(ctx, "materialize_gpu_ciphertexts"):
        return ctx.materialize_gpu_ciphertexts(rows)

    if hasattr(ctx, "_ctx") and hasattr(ctx._ctx, "materialize_gpu_ciphertexts"):
        return ctx._ctx.materialize_gpu_ciphertexts(rows)

    return rows


def encrypt_matrix_columns(rt, X):
    """
    For component transform:
      Y = X @ W
      Y.T = W.T @ X.T
    Therefore encrypt rows of X.T, i.e. columns of X.
    """
    X = np.asarray(X, dtype=np.float64)
    rows = []
    for k in range(X.shape[1]):
        rows.append(rt.encrypt(X[:, k].tolist()))
    return rows


def component_he_matmul_return_rows(rt, X, W):
    """
    Compute encrypted rows for (X @ W).T.

    X: m × k
    W: k × n, integer weights

    returns:
      rows: n ciphertext rows, each length m
    """
    X = np.asarray(X, dtype=np.float64)
    W = as_int_matrix(W)

    if X.shape[1] != W.shape[0]:
        raise ValueError(f"shape mismatch: X={X.shape}, W={W.shape}")

    t0 = now_us()
    enc_rows = encrypt_matrix_columns(rt, X)
    t1 = now_us()

    U = W.T.astype(int).tolist()

    t2 = now_us()
    gpu_rows = call_component_linear_transform(rt, enc_rows, U)
    t3 = now_us()

    return {
        "rows": gpu_rows,
        "encrypted_input_rows": len(enc_rows),
        "encrypted_output_rows": len(gpu_rows),
        "encrypt_us": t1 - t0,
        "component_compute_us": t3 - t2,
    }


def he_add_rows(rt, left_rows, right_rows):
    if len(left_rows) != len(right_rows):
        raise ValueError(f"row count mismatch: {len(left_rows)} vs {len(right_rows)}")

    out = []
    t0 = now_us()
    for a, b in zip(left_rows, right_rows):
        out.append(rt.add_ct(a, b))
    t1 = now_us()

    return out, (t1 - t0)


def add_many_row_sets(rt, row_sets):
    if not row_sets:
        raise ValueError("row_sets must be non-empty")
    acc = row_sets[0]
    add_total = 0.0
    for rows in row_sets[1:]:
        acc, t = he_add_rows(rt, acc, rows)
        add_total += t
    return acc, add_total


def materialize_and_decrypt(rt, rows, logical_length):
    t0 = now_us()
    mat_rows = call_materialize(rt, rows)
    t1 = now_us()

    t2 = now_us()
    Y_dec = decrypt_rows_as_matrix(rt, mat_rows, logical_length=logical_length)
    t3 = now_us()

    return Y_dec, {
        "materialize_us": t1 - t0,
        "decrypt_us": t3 - t2,
        "num_output_rows": len(mat_rows),
    }


def make_matrices(seed=206768):
    rng = np.random.default_rng(seed)

    # X 使用小整数转 float，W 使用 int64，因为 fused raw component backend 当前只支持整数权重。
    X = rng.integers(-2, 3, size=(64, 768)).astype(np.float64)
    W = rng.integers(-2, 3, size=(768, 768)).astype(np.int64)

    return X, W


def run_matrix_physical_tile_case(rt, X, W):
    """
    Legacy matrix-block / physical tile route.

    X: 64 × 768
    W: 768 × 768

    Block sizes:
      hidden block = 128
      output block = 128
      physical row tile = 64 × 128

    Since X has only 64 rows, the lower 64×128 tile in each 128×128 logical X block is zero and skipped.
    """
    print_sep("WP3-E2 / Experiment 1: X=64x768 matrix-block / physical tile HE PCMM")

    block = 128
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

    nonzero_physical_tile_products = 0
    skipped_zero_physical_tile_products = 0

    output_block_reports = []

    for j in range(num_out_blocks):
        partial_sets = []

        for k in range(num_mid_blocks):
            X_tile = X[:, k * block:(k + 1) * block]
            W_block = W[k * block:(k + 1) * block, j * block:(j + 1) * block]

            # In legacy physical tiling, each 128×128 logical X block would have two 64×128 tiles.
            # For X=64×768, the lower tile is all zero and skipped.
            skipped_zero_physical_tile_products += 1

            if np.allclose(X_tile, 0.0):
                continue

            result = component_he_matmul_return_rows(rt, X_tile, W_block)
            partial_sets.append(result["rows"])

            total_encrypt += result["encrypt_us"]
            total_compute += result["component_compute_us"]
            nonzero_physical_tile_products += 1

        summed_rows, add_us = add_many_row_sets(rt, partial_sets)
        total_add += add_us

        Y_block_dec, md = materialize_and_decrypt(rt, summed_rows, logical_length=X.shape[0])
        total_materialize += md["materialize_us"]
        total_decrypt += md["decrypt_us"]

        Y_blocks.append(Y_block_dec)

        Y_block_ref = Y_ref[:, j * block:(j + 1) * block]
        output_block_reports.append({
            "output_block_index（输出块编号）": j,
            "num_partial_products（partial乘法数量）": len(partial_sets),
            "Y_block_shape（输出块形状）": list(Y_block_dec.shape),
            "max_abs_err（最大绝对误差）": max_abs_err(Y_block_dec, Y_block_ref),
        })

    Y_dec = np.hstack(Y_blocks)
    err = max_abs_err(Y_dec, Y_ref)

    report = {
        "experiment（实验）": "matrix_block_physical_tile_64x768_he_pcmm",
        "method（方法）": "legacy physical row tile route using component HE backend",
        "target（目标）": "Y = X(64x768) @ W(768x768)",
        "X_shape（X形状）": list(X.shape),
        "W_shape（W形状）": list(W.shape),
        "Y_shape（Y形状）": list(Y_dec.shape),
        "logical_block_shape（逻辑块形状）": [128, 128],
        "physical_tile_shape（物理tile形状）": [64, 128],
        "num_mid_blocks（中间维度块数）": num_mid_blocks,
        "num_out_blocks（输出维度块数）": num_out_blocks,
        "nonzero_physical_tile_products（非零物理tile乘法数）": nonzero_physical_tile_products,
        "skipped_zero_physical_tile_products（跳过的全零物理tile乘法数）": skipped_zero_physical_tile_products,
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

    zh_print_report(report)
    return report


def run_role_specific_case(rt, X, W):
    """
    Role-specific route.

    X: 64 × 768, padded conceptually to 128 rows.
    For each 128 hidden block:
      X_part_0: 64 × 64
      X_part_1: 64 × 64
      W_part_0: 64 × 128
      W_part_1: 64 × 128

    Computes:
      Y_block_j = sum_k (X_k0 @ W_k0_j + X_k1 @ W_k1_j)
    """
    print_sep("WP3-E2 / Experiment 2: X=64x768 role-specific block tiling HE PCMM")

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
        partial_sets = []

        for k in range(num_mid_blocks):
            X_block = X[:, k * block:(k + 1) * block]
            W_block = W[k * block:(k + 1) * block, j * block:(j + 1) * block]

            X0 = X_block[:, :sub]
            X1 = X_block[:, sub:block]
            W0 = W_block[:sub, :]
            W1 = W_block[sub:block, :]

            for X_part, W_part in [(X0, W0), (X1, W1)]:
                result = component_he_matmul_return_rows(rt, X_part, W_part)
                partial_sets.append(result["rows"])
                total_encrypt += result["encrypt_us"]
                total_compute += result["component_compute_us"]
                component_products += 1

        summed_rows, add_us = add_many_row_sets(rt, partial_sets)
        total_add += add_us

        Y_block_dec, md = materialize_and_decrypt(rt, summed_rows, logical_length=X.shape[0])
        total_materialize += md["materialize_us"]
        total_decrypt += md["decrypt_us"]

        Y_blocks.append(Y_block_dec)

        Y_block_ref = Y_ref[:, j * block:(j + 1) * block]
        output_block_reports.append({
            "output_block_index（输出块编号）": j,
            "num_partial_products（partial乘法数量）": len(partial_sets),
            "Y_block_shape（输出块形状）": list(Y_block_dec.shape),
            "max_abs_err（最大绝对误差）": max_abs_err(Y_block_dec, Y_block_ref),
        })

    Y_dec = np.hstack(Y_blocks)
    err = max_abs_err(Y_dec, Y_ref)

    report = {
        "experiment（实验）": "role_specific_block_tiling_64x768_he_pcmm",
        "method（方法）": "activation tiles 64x64 within conceptual 128x64, weight tiles 64x128, component HE backend",
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

    zh_print_report(report)
    return report


def zh_print_report(report):
    for k, v in report.items():
        if isinstance(v, list) and k.endswith("报告）"):
            print(f"{k}:")
            for item in v:
                print(f"  - {item}")
        else:
            print(f"{k}: {v}")


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
        print_sep("Runtime info")
        print("runtime info（运行时信息）:", rt.info())

        matrix_report = run_matrix_physical_tile_case(rt, X, W)
        role_report = run_role_specific_case(rt, X, W)

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3e2_matrix_role_tile_64x768_he_pcmm_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "matrix_block_physical_tile_64x768_he_pcmm": matrix_report,
                "role_specific_block_tiling_64x768_he_pcmm": role_report,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_sep("Report saved")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
