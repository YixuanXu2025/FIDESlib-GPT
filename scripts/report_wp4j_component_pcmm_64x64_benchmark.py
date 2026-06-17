import json
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def now_us():
    return time.perf_counter() * 1e6


def as_float_matrix(mat):
    return [[float(v) for v in row] for row in np.asarray(mat, dtype=np.float64)]


def get_call_targets(rt):
    """
    兼容不同封装层：
      - rt.xxx
      - rt.ctx.xxx
      - rt.ctx._ctx.xxx
      - rt.require_context().xxx
      - rt.require_context()._ctx.xxx
    """
    targets = [rt]

    if hasattr(rt, "ctx") and rt.ctx is not None:
        targets.append(rt.ctx)
        if hasattr(rt.ctx, "_ctx"):
            targets.append(rt.ctx._ctx)

    if hasattr(rt, "require_context"):
        try:
            ctx = rt.require_context()
            targets.append(ctx)
            if hasattr(ctx, "_ctx"):
                targets.append(ctx._ctx)
        except Exception:
            pass

    # 去重但保序
    out = []
    seen = set()
    for t in targets:
        key = id(t)
        if key not in seen:
            out.append(t)
            seen.add(key)
    return out


def call_component_linear_transform(rt, rows, U, copyback=False):
    """
    调用 component-level matrix transform。

    优先使用正式 API:
      component_linear_transform_gpu(rows, U)

    如果当前仓库只有底层 fused raw API，则回退到:
      component_linear_matmul_gpu_fused_raw(rows, U, copyback=False)
    """
    U_list = as_float_matrix(U)

    method_specs = [
        ("component_linear_transform_gpu", "formal"),
        ("component_linear_matmul_gpu_fused_raw", "fused_raw"),
    ]

    last_errors = []

    for target in get_call_targets(rt):
        for name, kind in method_specs:
            fn = getattr(target, name, None)
            if fn is None:
                continue

            # 尽量兼容不同函数签名
            attempts = []
            if kind == "formal":
                attempts = [
                    lambda: fn(rows, U_list),
                    lambda: fn(rows=rows, U=U_list),
                    lambda: fn(rows=rows, weights=U_list),
                    lambda: fn(rows, U_list, copyback),
                ]
            else:
                attempts = [
                    lambda: fn(rows, U_list, copyback),
                    lambda: fn(rows, U_list),
                    lambda: fn(rows=rows, U=U_list, copyback=copyback),
                ]

            for attempt in attempts:
                try:
                    return attempt(), {
                        "method": name,
                        "target_type": type(target).__name__,
                        "copyback": copyback,
                    }
                except TypeError as e:
                    last_errors.append(f"{type(target).__name__}.{name}: {repr(e)}")
                    continue

    raise RuntimeError(
        "No usable component linear transform API found. Last signature errors:\n"
        + "\n".join(last_errors[-12:])
    )


def call_materialize(rt, rows):
    """
    把 GPU-only component ciphertext rows materialize 回可解密 ciphertext rows。
    如果当前调用路径已经 copyback/materialized，则返回原 rows。
    """
    method_names = [
        "materialize_gpu_ciphertexts",
        "materialize_gpu_ciphertext_rows",
        "materialize_ciphertexts",
    ]

    last_errors = []

    for target in get_call_targets(rt):
        for name in method_names:
            fn = getattr(target, name, None)
            if fn is None:
                continue

            attempts = [
                lambda: fn(rows),
                lambda: fn(rows=rows),
            ]

            for attempt in attempts:
                try:
                    return attempt(), {
                        "method": name,
                        "target_type": type(target).__name__,
                    }
                except TypeError as e:
                    last_errors.append(f"{type(target).__name__}.{name}: {repr(e)}")
                    continue

    # 没有 materialize API 时，尝试直接返回。
    # 如果 rows 已经是 CPU/materialized handle，后续 decrypt 会成功；
    # 如果不是，decrypt 会显式失败。
    return rows, {
        "method": "none_direct_rows_returned",
        "target_type": None,
        "note": "No materialize API found; assuming rows are already decryptable.",
        "last_errors": last_errors[-12:],
    }


def decrypt_rows(rt, rows, logical_length):
    outs = []
    for h in rows:
        outs.append(np.array(rt.decrypt(h, logical_length=logical_length), dtype=np.float64))
    return np.vstack(outs)


def main():
    n = 64
    seed = 205764

    rng = np.random.default_rng(seed)

    # component_linear_matmul_gpu_fused_raw 当前只支持整数明文权重。
    # 因此 W 必须使用 int64 / 小整数矩阵。
    # X 可以是浮点密文输入；为了结果稳定和便于解释，这里也使用小整数再转 float。
    X = rng.integers(-2, 3, size=(n, n)).astype(np.float64)
    W = rng.integers(-2, 3, size=(n, n)).astype(np.int64)

    Y_ref = X @ W

    # component transform 更自然做 U @ rows。
    # 为了得到 X @ W，使用：
    #   (X @ W)^T = W^T @ X^T
    X_transposed_rows = X.T.copy()
    U = W.T.copy()

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

    print("=" * 100)
    print("WP4-J component PCMM 64x64 benchmark")
    print(f"X shape（输入矩阵形状）: {X.shape}")
    print(f"W shape（明文权重形状）: {W.shape}")
    print("target（目标）: Y = X @ W")
    print("implementation（实现）: compute (X @ W)^T = W.T @ X.T via component-linear-transform")
    print("=" * 100)

    with HERuntime(cfg, rotation_steps=[]) as rt:
        print("[phase] encrypt rows: encrypt X.T rows")
        t0 = now_us()
        encrypted_rows = [
            rt.encrypt([float(v) for v in X_transposed_rows[i]])
            for i in range(n)
        ]
        t1 = now_us()
        encrypt_us = t1 - t0

        print("[phase] GPU-only component linear transform")
        t0 = now_us()
        gpu_rows, transform_info = call_component_linear_transform(
            rt,
            encrypted_rows,
            U,
            copyback=False,
        )
        t1 = now_us()
        compute_us = t1 - t0

        print("[phase] explicit materialize")
        t0 = now_us()
        materialized_rows, materialize_info = call_materialize(rt, gpu_rows)
        t1 = now_us()
        materialize_us = t1 - t0

        print("[phase] decrypt materialized rows")
        t0 = now_us()
        Y_t_dec = decrypt_rows(rt, materialized_rows, logical_length=n)
        t1 = now_us()
        decrypt_us = t1 - t0

    # Y_t_dec should be (X @ W).T
    Y_dec = Y_t_dec.T.copy()

    max_abs_err = float(np.max(np.abs(Y_dec - Y_ref)))
    mean_abs_err = float(np.mean(np.abs(Y_dec - Y_ref)))

    # 额外检查 orientation，防止 API 实际语义和预期相反
    candidates = {
        "target_X_matmul_W": X @ W,
        "X_matmul_W_T": X @ W.T,
        "W_matmul_X": W @ X,
        "W_T_matmul_X_T_transposed": (W.T @ X.T).T,
    }
    orientation_errors = {
        name: float(np.max(np.abs(Y_dec - ref)))
        for name, ref in candidates.items()
    }

    report = {
        "experiment": "wp4j_component_pcmm_64x64_benchmark",
        "purpose": "Benchmark component-level PCMM for X(64x64) @ W(64x64) with integer plaintext weights, matching the current fused raw component backend constraint.",
        "n": n,
        "seed": seed,
        "shapes": {
            "X": list(X.shape),
            "W": list(W.shape),
            "Y": list(Y_ref.shape),
            "encrypted_input_rows": list(X_transposed_rows.shape),
            "component_transform_U": list(U.shape),
        },
        "method": {
            "semantic_target": "Y = X @ W",
            "component_transform_form": "(X @ W).T = W.T @ X.T",
            "transform_info": transform_info,
            "materialize_info": materialize_info,
        },
        "timing_us": {
            "encrypt_rows": encrypt_us,
            "component_compute": compute_us,
            "explicit_materialize_all_rows": materialize_us,
            "decrypt_materialized_rows": decrypt_us,
            "total_including_encrypt_materialize_decrypt": encrypt_us + compute_us + materialize_us + decrypt_us,
        },
        "errors": {
            "max_abs_err": max_abs_err,
            "mean_abs_err": mean_abs_err,
            "orientation_errors": orientation_errors,
        },
        "Y_ref_first_row_first8": Y_ref[0, :8].tolist(),
        "Y_dec_first_row_first8": Y_dec[0, :8].tolist(),
    }

    print("=" * 100)
    print("WP4-J component PCMM 64x64 benchmark report")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 100)

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4j_component_pcmm_64x64_benchmark_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"报告已保存: {out_path}")


if __name__ == "__main__":
    main()
