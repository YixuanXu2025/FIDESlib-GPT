from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np

from .runtime import HERuntime
from .tensor import CipherTensor
from .ops import (
    he_add,
    he_add_plain,
    he_mult_plain,
    he_place_slot0_to,
    he_sum_slots_to_slot0,
)


def _now_us() -> float:
    return time.perf_counter() * 1e6


def _to_float_list(x):
    return [float(v) for v in x]


def he_encrypt_vector_padded(
    rt: HERuntime,
    x,
    *,
    physical_slots: int,
    name: str = "",
    layout: str = "pcmm_row_vector_padded",
    device=None,
) -> CipherTensor:
    """
    将一个逻辑长度为 len(x) 的向量，加零 padding 到 physical_slots 后再加密。

    这样做的目的：
      - 逻辑维度可以是 split_dim（例如 2 / 4 / 8 / 64）
      - 物理上可以占用更多 slots，从而允许 out_dim <= slots_used
      - 这使得我们可以在 Python 中先做一个“PCMM-structured baseline”

    返回：
      CipherTensor:
        shape = (logical_dim,)
        slots_used = physical_slots
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    logical_dim = int(x.shape[0])

    if physical_slots < logical_dim:
        raise ValueError(
            f"physical_slots={physical_slots} < logical_dim={logical_dim}"
        )

    padded = np.zeros((physical_slots,), dtype=np.float64)
    padded[:logical_dim] = x

    ct = rt.encrypt(padded.tolist())

    if device is None and hasattr(rt, "cfg") and getattr(rt.cfg, "devices", None):
        device = rt.cfg.devices[0]

    return CipherTensor(
        ct=ct,
        shape=(logical_dim,),
        layout=layout,
        slots_used=physical_slots,
        level=None,
        scale=None,
        device=device,
        name=name,
    )


def he_dot_plain_to_slot0_padded(
    rt: HERuntime,
    x: CipherTensor,
    plain_vec,
    bias_scalar=None,
    *,
    name=None,
) -> CipherTensor:
    """
    与 ops.he_dot_plain_to_slot0(...) 类似，但这里允许：
      - x.shape[0] = 逻辑维度
      - x.slots_used = 物理槽位数（可能更大）

    因此 plain_vec 只需要提供“逻辑长度”的系数，
    函数内部会自动补 0 到 x.slots_used。
    """
    x.ensure_cipher()

    plain_vec = np.asarray(plain_vec, dtype=np.float64).reshape(-1)
    logical_dim = int(x.shape[0])

    if plain_vec.shape[0] != logical_dim:
        raise ValueError(
            f"plain_vec length {plain_vec.shape[0]} != logical_dim {logical_dim}"
        )

    padded = np.zeros((x.slots_used,), dtype=np.float64)
    padded[:logical_dim] = plain_vec

    prod = he_mult_plain(rt, x, padded.tolist(), name=f"{x.name}_mul_plain")

    # 只对逻辑长度范围做 reduce
    acc0 = he_sum_slots_to_slot0(
        rt,
        prod,
        length=logical_dim,
        name=(x.name if name is None else name),
    )

    if bias_scalar is not None and float(bias_scalar) != 0.0:
        bias0 = np.zeros((x.slots_used,), dtype=np.float64)
        bias0[0] = float(bias_scalar)
        acc0 = he_add_plain(
            rt,
            acc0,
            bias0.tolist(),
            name=(x.name if name is None else name),
        )

    return acc0


def he_linear_plain_padded(
    rt: HERuntime,
    x: CipherTensor,
    weight,
    bias=None,
    *,
    name=None,
) -> CipherTensor:
    """
    Python 版“带物理 padding 的线性层”。

    和现有 ops.he_linear_plain(...) 的区别：
      - 允许 x.shape[0] = in_dim
      - 允许 x.slots_used > in_dim
      - 只要 out_dim <= x.slots_used 即可

    适合拿来做 WP3-A 的 PCMM row-wise baseline。
    """
    x.ensure_cipher()

    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 2:
        raise ValueError("weight must be 2D")

    in_dim, out_dim = weight.shape

    if x.ndim != 1:
        raise ValueError("he_linear_plain_padded currently only supports 1D CipherTensor input")

    if x.shape[0] != in_dim:
        raise ValueError(
            f"input dim mismatch: x.shape={x.shape}, weight.shape={weight.shape}"
        )

    if out_dim > x.slots_used:
        raise ValueError(
            f"out_dim={out_dim} must be <= x.slots_used={x.slots_used}"
        )

    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        if bias.shape != (out_dim,):
            raise ValueError(f"bias must have shape ({out_dim},), got {bias.shape}")

    out = None
    for j in range(out_dim):
        col_j = weight[:, j]
        bj = None if bias is None else float(bias[j])

        scalar_j = he_dot_plain_to_slot0_padded(
            rt,
            x,
            col_j,
            bias_scalar=bj,
            name=f"{x.name}_dot_{j}",
        )

        placed_j = he_place_slot0_to(
            rt,
            scalar_j,
            j,
            name=f"{x.name}_place_{j}",
        )

        out = placed_j if out is None else he_add(
            rt,
            out,
            placed_j,
            name=(x.name if name is None else name),
        )

    return x.clone_with(
        ct=out.ct,
        shape=(out_dim,),
        slots_used=out_dim,
        name=(x.name if name is None else name),
    )


def pcmm_square_plain_reference(
    x_block,
    w_block,
    *,
    split_dim: int,
):
    """
    单个 logical block 的 plain 参考实现。

    数学上：
      X_block (rows, mid_dim)
      W_block (mid_dim, out_dim)

      若 mid_dim 按 split_dim 拆成 g 段：
        Y = sum_t X_t @ W_t

    这里是 WP3-A 的“plain reference”。
    """
    x_block = np.asarray(x_block, dtype=np.float64)
    w_block = np.asarray(w_block, dtype=np.float64)

    rows, mid_dim = x_block.shape
    mid_dim_w, out_dim = w_block.shape

    if mid_dim_w != mid_dim:
        raise ValueError("shape mismatch in pcmm_square_plain_reference")
    if mid_dim % split_dim != 0:
        raise ValueError("mid_dim must be divisible by split_dim")

    y = np.zeros((rows, out_dim), dtype=np.float64)

    num_splits = mid_dim // split_dim
    for s in range(num_splits):
        c0 = s * split_dim
        c1 = (s + 1) * split_dim
        x_seg = x_block[:, c0:c1]
        w_seg = w_block[c0:c1, :]
        y += x_seg @ w_seg

    return y


def pcmm_square_python_baseline(
    rt: HERuntime,
    x_block,
    w_block,
    *,
    split_dim: int,
    physical_slots: int,
    repeats: int = 1,
    seed: int = 0,
) -> Dict:
    """
    WP3-A: Python 版 PCMM baseline（单个 logical block）

    当前策略：
      - 不做真正的 packed PCMM 内核
      - 而是采用“row-wise + padded slots”的 Python baseline
      - 每一行按 split_dim 分段
      - 每段都走一遍 he_linear_plain_padded(...)
      - 再把各段结果累加

    这是一个：
      - 正确性优先
      - 结构与接口优先
      - 性能不追求最优
    的 baseline

    返回：
      {
        "y_ref": ...,
        "y_he": ...,
        "max_abs_err": ...,
        "encrypt_us": ...,
        "core_us": ...,
        "decrypt_us": ...,
        "total_us": ...,
      }
    """
    rng = np.random.default_rng(seed)  # 这里只是为了接口统一，目前不直接使用
    _ = rng

    x_block = np.asarray(x_block, dtype=np.float64)
    w_block = np.asarray(w_block, dtype=np.float64)

    rows, mid_dim = x_block.shape
    mid_dim_w, out_dim = w_block.shape

    if mid_dim_w != mid_dim:
        raise ValueError("shape mismatch in pcmm_square_python_baseline")
    if mid_dim % split_dim != 0:
        raise ValueError("mid_dim must be divisible by split_dim")
    if out_dim > physical_slots:
        raise ValueError(
            f"out_dim={out_dim} must be <= physical_slots={physical_slots}"
        )

    y_ref = pcmm_square_plain_reference(x_block, w_block, split_dim=split_dim)

    encrypt_us_list = []
    core_us_list = []
    decrypt_us_list = []
    total_us_list = []
    err_list = []

    y_he_last = None

    for _rep in range(repeats):
        row_outputs = []

        t_total_0 = _now_us()
        encrypt_us = 0.0
        core_us = 0.0
        decrypt_us = 0.0

        num_splits = mid_dim // split_dim

        for r in range(rows):
            acc_ct = None

            for s in range(num_splits):
                c0 = s * split_dim
                c1 = (s + 1) * split_dim

                x_seg = x_block[r, c0:c1]
                w_seg = w_block[c0:c1, :]

                t0 = _now_us()
                ct_x = he_encrypt_vector_padded(
                    rt,
                    x_seg,
                    physical_slots=physical_slots,
                    name=f"x_r{r}_s{s}",
                )
                t1 = _now_us()
                encrypt_us += (t1 - t0)

                t2 = _now_us()
                ct_part = he_linear_plain_padded(
                    rt,
                    ct_x,
                    w_seg,
                    bias=None,
                    name=f"y_r{r}_s{s}",
                )
                acc_ct = ct_part if acc_ct is None else he_add(
                    rt,
                    acc_ct,
                    ct_part,
                    name=f"y_r{r}_acc",
                )
                t3 = _now_us()
                core_us += (t3 - t2)

            t4 = _now_us()
            row_out = np.array(
                rt.decrypt(acc_ct.ct, logical_length=out_dim),
                dtype=np.float64,
            )[:out_dim]
            t5 = _now_us()
            decrypt_us += (t5 - t4)

            row_outputs.append(row_out)

        t_total_1 = _now_us()

        y_he = np.stack(row_outputs, axis=0)
        y_he_last = y_he

        total_us = t_total_1 - t_total_0
        max_abs_err = float(np.max(np.abs(y_ref - y_he)))

        encrypt_us_list.append(encrypt_us)
        core_us_list.append(core_us)
        decrypt_us_list.append(decrypt_us)
        total_us_list.append(total_us)
        err_list.append(max_abs_err)

    result = {
        "rows": rows,
        "mid_dim": mid_dim,
        "out_dim": out_dim,
        "split_dim": split_dim,
        "physical_slots": physical_slots,
        "repeats": repeats,
        "y_ref": y_ref,
        "y_he": y_he_last,
        "max_abs_err": max(err_list) if err_list else None,
        "encrypt_us_median": float(np.median(encrypt_us_list)) if encrypt_us_list else None,
        "core_us_median": float(np.median(core_us_list)) if core_us_list else None,
        "decrypt_us_median": float(np.median(decrypt_us_list)) if decrypt_us_list else None,
        "total_us_median": float(np.median(total_us_list)) if total_us_list else None,
    }
    return result
