from __future__ import annotations

import numpy as np

from .runtime import HERuntime
from .tensor import CipherTensor


def _to_float_list(x):
    return [float(v) for v in x]


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _ensure_same_shape(x: CipherTensor, y: CipherTensor):
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")


def _ensure_slots_match_plain(x: CipherTensor, plain):
    plain = _to_float_list(plain)
    if x.slots_used > 0 and len(plain) != x.slots_used:
        raise ValueError(
            f"plaintext length {len(plain)} != CipherTensor.slots_used {x.slots_used}"
        )
    return plain


# ----------------------------------------------------------------------
# 兼容之前的最小测试
# ----------------------------------------------------------------------

def he_roundtrip_vector(rt: HERuntime, x):
    return rt.roundtrip(_to_float_list(x))


def he_add_plain_input_vectors(rt: HERuntime, x, y):
    return rt.add(_to_float_list(x), _to_float_list(y))


def he_mult_scalar_plain_input(rt: HERuntime, x, scalar: float):
    return rt.mult_scalar(_to_float_list(x), float(scalar))


# ----------------------------------------------------------------------
# 真实可用的 CipherTensor 低层接口
# ----------------------------------------------------------------------

def he_encrypt_tensor(
    rt: HERuntime,
    x,
    *,
    shape=None,
    layout="vector_contiguous",
    name="",
    device=None,
) -> CipherTensor:
    values = _to_float_list(x)

    if shape is None:
        shape = (len(values),)
    else:
        shape = tuple(shape)

    if _numel(shape) != len(values):
        raise ValueError(
            f"shape {shape} implies numel {_numel(shape)} but input length is {len(values)}"
        )

    ct = rt.encrypt(values)

    if device is None and hasattr(rt, "cfg") and getattr(rt.cfg, "devices", None):
        device = rt.cfg.devices[0]

    return CipherTensor(
        ct=ct,
        shape=shape,
        layout=layout,
        slots_used=len(values),
        level=None,
        scale=None,
        device=device,
        name=name,
    )


def he_decrypt_tensor(rt: HERuntime, x: CipherTensor, logical_length=None):
    x.ensure_cipher()
    if logical_length is None:
        logical_length = x.slots_used if x.slots_used > 0 else x.numel
    return rt.decrypt(x.ct, logical_length=logical_length)


def he_add(rt: HERuntime, x: CipherTensor, y: CipherTensor, *, name=None) -> CipherTensor:
    x.ensure_cipher()
    y.ensure_cipher()
    _ensure_same_shape(x, y)

    ct = rt.add_ct(x.ct, y.ct)
    return x.clone_with(ct=ct, name=(x.name if name is None else name))


def he_add_plain(rt: HERuntime, x: CipherTensor, plain, *, name=None) -> CipherTensor:
    x.ensure_cipher()
    plain = _ensure_slots_match_plain(x, plain)

    ct = rt.add_plain_ct(x.ct, plain)
    return x.clone_with(ct=ct, name=(x.name if name is None else name))


def he_mult_scalar(rt: HERuntime, x: CipherTensor, scalar: float, *, name=None) -> CipherTensor:
    x.ensure_cipher()

    ct = rt.mult_scalar_ct(x.ct, float(scalar))
    return x.clone_with(ct=ct, name=(x.name if name is None else name))


def he_mult_plain(rt: HERuntime, x: CipherTensor, plain, *, name=None) -> CipherTensor:
    x.ensure_cipher()
    plain = _ensure_slots_match_plain(x, plain)

    ct = rt.mult_plain_ct(x.ct, plain)
    return x.clone_with(ct=ct, name=(x.name if name is None else name))


def he_rotate(rt: HERuntime, x: CipherTensor, steps: int, *, name=None) -> CipherTensor:
    x.ensure_cipher()

    ct = rt.rotate_ct(x.ct, int(steps))
    return x.clone_with(ct=ct, name=(x.name if name is None else name))


# ----------------------------------------------------------------------
# baseline helpers for he_linear_plain
# ----------------------------------------------------------------------

def he_sum_slots_to_slot0(rt: HERuntime, x: CipherTensor, length=None, *, name=None) -> CipherTensor:
    x.ensure_cipher()

    if length is None:
        length = x.slots_used
    if length <= 0:
        raise ValueError("length must be positive")

    acc = x
    for step in range(1, length):
        rotated = he_rotate(rt, x, step, name=f"{x.name}_rot{step}")
        acc = he_add(rt, acc, rotated, name=(x.name if name is None else name))

    mask0 = [1.0] + [0.0] * (x.slots_used - 1)
    return he_mult_plain(rt, acc, mask0, name=(x.name if name is None else name))


def he_dot_plain_to_slot0(rt: HERuntime, x: CipherTensor, plain_vec, bias_scalar=None, *, name=None) -> CipherTensor:
    x.ensure_cipher()
    plain_vec = _ensure_slots_match_plain(x, plain_vec)

    prod = he_mult_plain(rt, x, plain_vec, name=f"{x.name}_mul_plain")
    acc0 = he_sum_slots_to_slot0(rt, prod, length=x.slots_used, name=(x.name if name is None else name))

    if bias_scalar is not None and float(bias_scalar) != 0.0:
        bias0 = [float(bias_scalar)] + [0.0] * (x.slots_used - 1)
        acc0 = he_add_plain(rt, acc0, bias0, name=(x.name if name is None else name))

    return acc0


def he_place_slot0_to(rt: HERuntime, x: CipherTensor, target_slot: int, *, name=None) -> CipherTensor:
    x.ensure_cipher()

    if target_slot < 0:
        raise ValueError("target_slot must be non-negative")
    if target_slot >= x.slots_used:
        raise ValueError(f"target_slot {target_slot} out of range for slots_used={x.slots_used}")

    if target_slot == 0:
        return x

    return he_rotate(rt, x, -target_slot, name=(x.name if name is None else name))


def he_linear_plain(rt: HERuntime, x: CipherTensor, weight, bias=None, *, name=None) -> CipherTensor:
    x.ensure_cipher()

    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 2:
        raise ValueError("weight must be 2D")

    in_dim, out_dim = weight.shape

    if x.ndim != 1:
        raise ValueError("baseline he_linear_plain currently only supports 1D CipherTensor input")
    if x.shape[0] != in_dim:
        raise ValueError(f"input dim mismatch: x.shape={x.shape}, weight.shape={weight.shape}")
    if out_dim > x.slots_used:
        raise ValueError(
            f"baseline he_linear_plain requires out_dim <= x.slots_used, "
            f"got out_dim={out_dim}, slots_used={x.slots_used}"
        )

    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        if bias.shape != (out_dim,):
            raise ValueError(f"bias must have shape ({out_dim},), got {bias.shape}")

    out = None
    for j in range(out_dim):
        col_j = weight[:, j].tolist()
        bj = None if bias is None else float(bias[j])

        scalar_j = he_dot_plain_to_slot0(rt, x, col_j, bias_scalar=bj, name=f"{x.name}_dot_{j}")
        placed_j = he_place_slot0_to(rt, scalar_j, j, name=f"{x.name}_place_{j}")

        out = placed_j if out is None else he_add(rt, out, placed_j, name=(x.name if name is None else name))

    return x.clone_with(
        ct=out.ct,
        shape=(out_dim,),
        slots_used=out_dim,
        name=(x.name if name is None else name),
    )


# ----------------------------------------------------------------------
# 明文参考
# ----------------------------------------------------------------------

def plaintext_linear(x, weight, bias=None):
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)

    y = x @ weight
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float64)
    return y
