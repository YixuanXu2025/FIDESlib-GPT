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
    """
    当前 component-linear-transform fused raw backend 支持 integer weights。
    所以本实验权重 W 使用整数矩阵，输入 X 可以是整数或浮点。
    """
    return np.asarray(x, dtype=np.int64)


def decrypt_rows_as_matrix(rt, rows, logical_length):
    """
    component transform 输出的是 Y.T 的每一行：
      output_rows[j] = Y[:, j]
    所以解密后需要 stack 成 Y.T，再 transpose 回 Y。
    """
    dec_rows = []
    for r in rows:
        dec = rt.decrypt(r, logical_length=logical_length)
        dec_rows.append(np.asarray(dec, dtype=np.float64))
    y_t = np.stack(dec_rows, axis=0)
    return y_t.T.copy()


def call_component_linear_transform(rt, rows, U):
    """
    兼容不同 runtime 封装层：
      - 优先 rt.component_linear_transform_gpu
      - 其次 rt.require_context().component_linear_transform_gpu
      - 再其次 ctx._ctx.component_linear_matmul_gpu_fused_raw
    """
    if hasattr(rt, "component_linear_transform_gpu"):
        return rt.component_linear_transform_gpu(rows, U)

    ctx = rt.require_context()
    if hasattr(ctx, "component_linear_transform_gpu"):
        return ctx.component_linear_transform_gpu(rows, U)

    if hasattr(ctx, "_ctx") and hasattr(ctx._ctx, "component_linear_matmul_gpu_fused_raw"):
        return ctx._ctx.component_linear_matmul_gpu_fused_raw(rows, U, False)

    raise RuntimeError(
        "No component linear transform API found. "
        "Expected component_linear_transform_gpu or component_linear_matmul_gpu_fused_raw."
    )


def call_materialize(rt, rows):
    """
    将 GPU-only / component 状态的 ciphertext 显式 materialize 成可解密对象。
    如果当前 runtime 没有 materialize API，则直接返回 rows。
    """
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
    将矩阵 X 按列加密。

    数学目标：
      Y = X @ W

    component-linear-transform 的形式：
      Y.T = W.T @ X.T

    所以这里加密的是 X.T 的每一行，也就是 X 的每一列。
    """
    X = np.asarray(X, dtype=np.float64)
    rows = []
    for k in range(X.shape[1]):
        rows.append(rt.encrypt(X[:, k].tolist()))
    return rows


def component_he_matmul(rt, X, W, *, materialize=True):
    """
    用 component-linear-transform 做真实 HE PCMM。

    输入：
      X: shape = (m, k)
      W: shape = (k, n)

    计算：
      Y = X @ W

    HE 内部形式：
      encrypt rows of X.T
      compute W.T @ X.T
      decrypt rows of Y.T
    """
    X = np.asarray(X, dtype=np.float64)
    W = as_int_matrix(W)

    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("X and W must be 2D matrices")
    if X.shape[1] != W.shape[0]:
        raise ValueError(f"shape mismatch: X={X.shape}, W={W.shape}")

    t0 = now_us()
    enc_rows = encrypt_matrix_columns(rt, X)
    t1 = now_us()

    U = W.T.astype(int).tolist()

    t2 = now_us()
    gpu_rows = call_component_linear_transform(rt, enc_rows, U)
    t3 = now_us()

    if materialize:
        t4 = now_us()
        mat_rows = call_materialize(rt, gpu_rows)
        t5 = now_us()
    else:
        t4 = t5 = now_us()
        mat_rows = gpu_rows

    t6 = now_us()
    Y_dec = decrypt_rows_as_matrix(rt, mat_rows, logical_length=X.shape[0])
    t7 = now_us()

    return {
        "Y_dec": Y_dec,
        "encrypted_input_rows": len(enc_rows),
        "encrypted_output_rows": len(mat_rows),
        "encrypt_us": t1 - t0,
        "component_compute_us": t3 - t2,
        "materialize_us": t5 - t4,
        "decrypt_us": t7 - t6,
        "total_us": (t1 - t0) + (t3 - t2) + (t5 - t4) + (t7 - t6),
    }


def he_add_rows(rt, left_rows, right_rows):
    """
    对两组 ciphertext rows 做逐行 HE add。
    每个 row 表示 Y.T 的一行，也就是 Y 的一列。
    """
    if len(left_rows) != len(right_rows):
        raise ValueError(f"row count mismatch: {len(left_rows)} vs {len(right_rows)}")

    out = []
    t0 = now_us()
    for a, b in zip(left_rows, right_rows):
        out.append(rt.add_ct(a, b))
    t1 = now_us()

    return out, (t1 - t0)


def component_he_matmul_return_rows(rt, X, W):
    """
    和 component_he_matmul 类似，但返回密文 rows，不立刻解密。
    用于 role-specific route 中做：
      partial0 + partial1
    """
    X = np.asarray(X, dtype=np.float64)
    W = as_int_matrix(W)

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


def make_test_matrices(seed=206301, logical_rows=128, logical_k=128, logical_out=128, active_rows=64):
    """
    构造一个 matrix-block 测试输入。

    X_logical:
      shape = 128 × 128
      前 active_rows 行为真实数据
      后面行为 0，用于模拟非方阵输入 padding

    W_logical:
      shape = 128 × 128
      整数权重，匹配 fused raw component backend 约束
    """
    rng = np.random.default_rng(seed)

    X = np.zeros((logical_rows, logical_k), dtype=np.float64)
    X[:active_rows, :] = rng.integers(-3, 4, size=(active_rows, logical_k)).astype(np.float64)

    W = rng.integers(-3, 4, size=(logical_k, logical_out)).astype(np.int64)

    return X, W


def run_matrix_physical_tile_case(rt, X_logical, W_logical):
    """
    实验 1：
    matrix-block / physical tile HE PCMM

    legacy physical tile:
      X logical block = 128 × 128
      physical tiles  = 两个 64 × 128 row tiles

    对每个非零 physical tile 做：
      X_phys(64×128) @ W_block(128×128)
    """
    print_sep("WP3-E / Experiment 1: matrix-block / physical tile HE PCMM")

    tile_rows = 64
    out_tiles = []
    tile_reports = []

    plain_t0 = now_us()
    Y_ref = X_logical @ W_logical
    plain_t1 = now_us()

    for tile_id, r0 in enumerate(range(0, X_logical.shape[0], tile_rows)):
        r1 = r0 + tile_rows
        X_tile = X_logical[r0:r1, :]

        is_zero_tile = bool(np.allclose(X_tile, 0.0))
        if is_zero_tile:
            tile_reports.append({
                "tile_id": tile_id,
                "row_range": [r0, r1],
                "skipped_zero_tile": True,
            })
            out_tiles.append(np.zeros((tile_rows, W_logical.shape[1]), dtype=np.float64))
            continue

        result = component_he_matmul(rt, X_tile, W_logical, materialize=True)
        Y_tile_dec = result.pop("Y_dec")
        out_tiles.append(Y_tile_dec)

        Y_tile_ref = Y_ref[r0:r1, :]
        result.update({
            "tile_id": tile_id,
            "row_range": [r0, r1],
            "skipped_zero_tile": False,
            "X_physical_tile_shape（X物理tile形状）": list(X_tile.shape),
            "W_logical_block_shape（W逻辑块形状）": list(W_logical.shape),
            "Y_physical_tile_shape（Y物理tile形状）": list(Y_tile_dec.shape),
            "max_abs_err（最大绝对误差）": max_abs_err(Y_tile_dec, Y_tile_ref),
        })
        tile_reports.append(result)

    Y_dec = np.vstack(out_tiles)
    err = max_abs_err(Y_dec, Y_ref)

    encrypt_us = sum(x.get("encrypt_us", 0.0) for x in tile_reports)
    compute_us = sum(x.get("component_compute_us", 0.0) for x in tile_reports)
    materialize_us = sum(x.get("materialize_us", 0.0) for x in tile_reports)
    decrypt_us = sum(x.get("decrypt_us", 0.0) for x in tile_reports)

    report = {
        "experiment（实验）": "matrix_block_physical_tile_he_pcmm",
        "method（方法）": "legacy physical row tile: X 64x128 physical tiles, component HE backend",
        "target（目标）": "Y = X_logical @ W_logical",
        "X_logical_shape（X逻辑块形状）": list(X_logical.shape),
        "W_logical_shape（W逻辑块形状）": list(W_logical.shape),
        "Y_shape（Y形状）": list(Y_dec.shape),
        "physical_tile_shape（物理tile形状）": [64, 128],
        "plain_reference_us（明文参考用时）": plain_t1 - plain_t0,
        "encrypt_us_total（总加密用时）": encrypt_us,
        "component_compute_us_total（总HE核心计算用时）": compute_us,
        "materialize_us_total（总materialize用时）": materialize_us,
        "decrypt_us_total（总解密用时）": decrypt_us,
        "total_he_us（HE总用时）": encrypt_us + compute_us + materialize_us + decrypt_us,
        "max_abs_err（最大绝对误差）": err,
        "num_physical_tiles（物理tile总数）": len(tile_reports),
        "num_nonzero_physical_tiles（非零物理tile数）": sum(1 for x in tile_reports if not x.get("skipped_zero_tile")),
        "tile_reports（各tile报告）": tile_reports,
        "Y_ref_first_row_first8（明文首行前8个）": Y_ref[0, :8].tolist(),
        "Y_dec_first_row_first8（解密首行前8个）": Y_dec[0, :8].tolist(),
    }

    zh_print_report(report)
    return report


def run_role_specific_case(rt, X_logical, W_logical):
    """
    实验 2：
    role-specific block tiling HE PCMM

    activation/input:
      X0 = X[:, 0:64]   shape = 128 × 64
      X1 = X[:, 64:128] shape = 128 × 64

    weight:
      W0 = W[0:64, :]   shape = 64 × 128
      W1 = W[64:128, :] shape = 64 × 128

    HE 中计算：
      partial0 = X0 @ W0
      partial1 = X1 @ W1
      Y = partial0 + partial1   # 这里是逐 ciphertext row 的 HE add
    """
    print_sep("WP3-E / Experiment 2: role-specific block tiling HE PCMM")

    plain_t0 = now_us()
    Y_ref = X_logical @ W_logical
    plain_t1 = now_us()

    X0 = X_logical[:, :64]
    X1 = X_logical[:, 64:128]
    W0 = W_logical[:64, :]
    W1 = W_logical[64:128, :]

    part0 = component_he_matmul_return_rows(rt, X0, W0)
    part1 = component_he_matmul_return_rows(rt, X1, W1)

    add_rows, add_us = he_add_rows(rt, part0["rows"], part1["rows"])

    t4 = now_us()
    mat_rows = call_materialize(rt, add_rows)
    t5 = now_us()

    t6 = now_us()
    Y_dec = decrypt_rows_as_matrix(rt, mat_rows, logical_length=X_logical.shape[0])
    t7 = now_us()

    err = max_abs_err(Y_dec, Y_ref)

    encrypt_us = part0["encrypt_us"] + part1["encrypt_us"]
    compute_us = part0["component_compute_us"] + part1["component_compute_us"]
    materialize_us = t5 - t4
    decrypt_us = t7 - t6

    report = {
        "experiment（实验）": "role_specific_block_tiling_he_pcmm",
        "method（方法）": "activation tiles 128x64, weight tiles 64x128, component HE backend, HE-add partials",
        "target（目标）": "Y = X0@W0 + X1@W1 = X_logical @ W_logical",
        "X_logical_shape（X逻辑块形状）": list(X_logical.shape),
        "W_logical_shape（W逻辑块形状）": list(W_logical.shape),
        "activation_tile_shapes（输入/激活tile形状）": [list(X0.shape), list(X1.shape)],
        "weight_tile_shapes（权重tile形状）": [list(W0.shape), list(W1.shape)],
        "Y_shape（Y形状）": list(Y_dec.shape),
        "plain_reference_us（明文参考用时）": plain_t1 - plain_t0,
        "encrypt_us_total（总加密用时）": encrypt_us,
        "component_compute_us_total（总HE核心计算用时）": compute_us,
        "he_add_partial_us（partial结果HE加法用时）": add_us,
        "materialize_us_total（总materialize用时）": materialize_us,
        "decrypt_us_total（总解密用时）": decrypt_us,
        "total_he_us（HE总用时）": encrypt_us + compute_us + add_us + materialize_us + decrypt_us,
        "max_abs_err（最大绝对误差）": err,
        "encrypted_input_rows_total（输入密文row总数）": part0["encrypted_input_rows"] + part1["encrypted_input_rows"],
        "encrypted_output_rows_before_add（加法前输出密文row总数）": part0["encrypted_output_rows"] + part1["encrypted_output_rows"],
        "encrypted_output_rows_after_add（加法后输出密文row总数）": len(add_rows),
        "Y_ref_first_row_first8（明文首行前8个）": Y_ref[0, :8].tolist(),
        "Y_dec_first_row_first8（解密首行前8个）": Y_dec[0, :8].tolist(),
    }

    zh_print_report(report)
    return report


def zh_print_report(report):
    for k, v in report.items():
        if k.endswith("reports（各tile报告）"):
            print(f"{k}:")
            for item in v:
                print(f"  - {item}")
        else:
            print(f"{k}: {v}")


def main():
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

    # 这里 active_rows=64，用来模拟真实输入 X 为 64×128，
    # padding 成 logical block 128×128。
    X_logical, W_logical = make_test_matrices(
        seed=206301,
        logical_rows=128,
        logical_k=128,
        logical_out=128,
        active_rows=64,
    )

    with HERuntime(cfg, rotation_steps=()) as rt:
        print_sep("Runtime info")
        print("runtime info（运行时信息）:", rt.info())

        matrix_report = run_matrix_physical_tile_case(rt, X_logical, W_logical)
        role_report = run_role_specific_case(rt, X_logical, W_logical)

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3e_matrix_role_tile_he_pcmm_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "matrix_block_physical_tile_he_pcmm": matrix_report,
                "role_specific_block_tiling_he_pcmm": role_report,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_sep("Report saved")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
