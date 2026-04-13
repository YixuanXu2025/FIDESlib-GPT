import json
import math
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.tensor import CipherTensor
from hegpt.ops import (
    he_add,
    he_mult_plain,
    he_rotate,
    he_sum_slots_to_slot0,
    he_place_slot0_to,
)


def now_us():
    return time.perf_counter() * 1e6


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def print_sep(title: str):
    print("=" * 100)
    print(title)


def zh_print(result: dict):
    print(f"  案例名称: {result['case_name']}")
    print(f"  行数: {result['rows']}")
    print(f"  中间维度: {result['mid_dim']}")
    print(f"  输出维度: {result['out_dim']}")
    print(f"  split_dim: {result['split_dim']}")
    print(f"  明文参考中位时间(us): {result['plain_reference_us_median']}")
    print(f"  BSGS输入密文数: {result['bsgs_input_ct_count_per_run']}")
    print(f"  naive输入密文数: {result['naive_input_ct_count_per_run']}")
    print(f"  输出密文数: {result['output_ct_count_per_run']}")
    print(f"  BSGS加密中位时间(us): {result['bsgs_encrypt_us_median']}")
    print(f"  BSGS核心计算中位时间(us): {result['bsgs_core_us_median']}")
    print(f"  BSGS解密中位时间(us): {result['bsgs_decrypt_us_median']}")
    print(f"  naive加密中位时间(us): {result['naive_encrypt_us_median']}")
    print(f"  naive核心计算中位时间(us): {result['naive_core_us_median']}")
    print(f"  naive解密中位时间(us): {result['naive_decrypt_us_median']}")
    print(f"  BSGS相对明文核心倍数: {result['bsgs_core_vs_plain_ratio']}")
    print(f"  naive相对明文核心倍数: {result['naive_core_vs_plain_ratio']}")
    print(f"  BSGS相对naive核心加速比: {result['bsgs_speedup_over_naive_core']}")
    print(f"  BSGS正确性误差(max_abs_err): {result['bsgs_plain_vs_he_max_abs_err']}")
    print(f"  naive正确性误差(max_abs_err): {result['naive_plain_vs_he_max_abs_err']}")
    print(f"  BSGS首行前8个输出: {result['bsgs_y_first_row'][:8]}")
    print(f"  naive首行前8个输出: {result['naive_y_first_row'][:8]}")


def call_he_add(rt, a, b, name):
    try:
        return he_add(rt, a, b, name=name)
    except TypeError:
        return he_add(rt, a, b)


def call_he_mult_plain(rt, x, plain_vec, name):
    try:
        return he_mult_plain(rt, x, plain_vec, name=name)
    except TypeError:
        return he_mult_plain(rt, x, plain_vec)


def call_he_rotate(rt, x, steps, name):
    try:
        return he_rotate(rt, x, steps, name=name)
    except TypeError:
        return he_rotate(rt, x, steps)


def call_he_sum_slots_to_slot0(rt, x, length, name):
    try:
        return he_sum_slots_to_slot0(rt, x, length=length, name=name)
    except TypeError:
        return he_sum_slots_to_slot0(rt, x, length=length)


def call_he_place_slot0_to(rt, x, idx, name):
    try:
        return he_place_slot0_to(rt, x, idx, name=name)
    except TypeError:
        return he_place_slot0_to(rt, x, idx)


def make_cipher_tensor(rt, packed_vec, logical_dim, name):
    ct = rt.encrypt([float(v) for v in packed_vec])
    device = None
    if hasattr(rt, "cfg") and getattr(rt.cfg, "devices", None):
        device = rt.cfg.devices[0]
    return CipherTensor(
        ct=ct,
        shape=(logical_dim,),
        layout="custom_packed_row",
        slots_used=len(packed_vec),
        level=None,
        scale=None,
        device=device,
        name=name,
    )


def plain_rotate(vec, steps, sigma):
    vec = np.asarray(vec, dtype=np.float64)
    return np.roll(vec, -sigma * steps)


def detect_rotation_sigma(rt):
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ct = make_cipher_tensor(rt, x, 4, "rot_detect")
    ct_r = call_he_rotate(rt, ct, 1, "rot_detect_r1")
    out = np.array(rt.decrypt(ct_r.ct, logical_length=4), dtype=np.float64)

    left = np.array([1.0, 2.0, 3.0, 0.0], dtype=np.float64)
    right = np.array([3.0, 0.0, 1.0, 2.0], dtype=np.float64)

    if np.max(np.abs(out - left)) < 1e-6:
        return +1
    if np.max(np.abs(out - right)) < 1e-6:
        return -1
    raise RuntimeError(f"Unable to detect rotation direction, got {out.tolist()}")


def build_padded_square_weight(weight, L):
    weight = np.asarray(weight, dtype=np.float64)
    in_dim, out_dim = weight.shape
    Wp = np.zeros((L, L), dtype=np.float64)
    Wp[:in_dim, :out_dim] = weight
    return Wp


def build_diagonals_for_linear(weight, sigma):
    weight = np.asarray(weight, dtype=np.float64)
    in_dim, out_dim = weight.shape
    L = max(in_dim, out_dim)
    Wp = build_padded_square_weight(weight, L)

    diags = []
    for s in range(L):
        d = np.zeros((L,), dtype=np.float64)
        for j in range(L):
            i = (j + sigma * s) % L
            d[j] = Wp[i, j]
        diags.append(d)
    return diags, L


def periodic_pack(vec, physical_slots):
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    L = len(vec)
    if physical_slots % L != 0:
        raise ValueError(f"physical_slots={physical_slots} must be divisible by L={L}")
    repeat = physical_slots // L
    return np.tile(vec, repeat)


def build_periodic_diagonals(weight, sigma, physical_slots):
    diags, L = build_diagonals_for_linear(weight, sigma)
    periodic_diags = [periodic_pack(d, physical_slots) for d in diags]
    return periodic_diags, L


def plain_dense_reference(x_block, w_block):
    return np.asarray(x_block, dtype=np.float64) @ np.asarray(w_block, dtype=np.float64)


def bench_plain_dense_reference(x_block, w_block, repeats=5):
    times = []
    y_last = None
    for _ in range(repeats):
        t0 = now_us()
        y = plain_dense_reference(x_block, w_block)
        t1 = now_us()
        times.append(t1 - t0)
        y_last = y
    return {
        "plain_reference_us_median": median(times),
        "y_plain_reference": y_last,
    }


def encrypt_periodic_row(rt, x_row, L, physical_slots, name):
    x_row = np.asarray(x_row, dtype=np.float64).reshape(-1)
    base = np.zeros((L,), dtype=np.float64)
    base[: len(x_row)] = x_row
    packed = periodic_pack(base, physical_slots)
    return make_cipher_tensor(rt, packed, L, name)


def encrypt_vector_padded(rt, x, physical_slots, name):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    logical_dim = len(x)
    padded = np.zeros((physical_slots,), dtype=np.float64)
    padded[:logical_dim] = x
    return make_cipher_tensor(rt, padded, logical_dim, name)


def he_dot_plain_to_slot0_padded(rt, x, plain_vec, name):
    plain_vec = np.asarray(plain_vec, dtype=np.float64).reshape(-1)
    logical_dim = x.shape[0]
    padded = np.zeros((x.slots_used,), dtype=np.float64)
    padded[:logical_dim] = plain_vec
    prod = call_he_mult_plain(rt, x, padded.tolist(), f"{name}_mul")
    acc0 = call_he_sum_slots_to_slot0(rt, prod, logical_dim, f"{name}_sum")
    return acc0


def he_linear_plain_padded(rt, x, weight, name):
    weight = np.asarray(weight, dtype=np.float64)
    in_dim, out_dim = weight.shape

    if x.shape[0] != in_dim:
        raise ValueError(f"input dim mismatch: {x.shape[0]} vs {in_dim}")
    if out_dim > x.slots_used:
        raise ValueError(f"out_dim={out_dim} must be <= slots_used={x.slots_used}")

    out = None
    for j in range(out_dim):
        col_j = weight[:, j]
        scalar_j = he_dot_plain_to_slot0_padded(rt, x, col_j, f"{name}_dot_{j}")
        placed_j = call_he_place_slot0_to(rt, scalar_j, j, f"{name}_place_{j}")
        out = placed_j if out is None else call_he_add(rt, out, placed_j, f"{name}_acc_{j}")

    return CipherTensor(
        ct=out.ct,
        shape=(out_dim,),
        layout="naive_linear_out",
        slots_used=out_dim,
        level=None,
        scale=None,
        device=x.device,
        name=name,
    )


def he_naive_single_row_split_timed(rt, x_row, weight, split_dim, name_prefix):
    x_row = np.asarray(x_row, dtype=np.float64).reshape(-1)
    weight = np.asarray(weight, dtype=np.float64)
    mid_dim, out_dim = weight.shape

    if len(x_row) != mid_dim:
        raise ValueError("x_row / weight mismatch")
    if mid_dim % split_dim != 0:
        raise ValueError("mid_dim must be divisible by split_dim")

    num_splits = mid_dim // split_dim
    acc_ct = None
    encrypt_us = 0.0
    core_us = 0.0

    for s in range(num_splits):
        c0 = s * split_dim
        c1 = (s + 1) * split_dim
        x_seg = x_row[c0:c1]
        w_seg = weight[c0:c1, :]

        t0 = now_us()
        ct_x = encrypt_vector_padded(rt, x_seg, out_dim, f"{name_prefix}_x_s{s}")
        t1 = now_us()
        encrypt_us += (t1 - t0)

        t2 = now_us()
        ct_part = he_linear_plain_padded(rt, ct_x, w_seg, f"{name_prefix}_part_s{s}")
        acc_ct = ct_part if acc_ct is None else call_he_add(rt, acc_ct, ct_part, f"{name_prefix}_acc_s{s}")
        t3 = now_us()
        core_us += (t3 - t2)

    return acc_ct, encrypt_us, core_us


def he_diagonal_bsgs_single_row_periodic(rt, x_ct, periodic_diags, sigma):
    L = len(periodic_diags)
    physical_slots = len(periodic_diags[0])

    g = int(math.ceil(math.sqrt(L)))
    h = int(math.ceil(L / g))

    baby = []
    for i in range(g):
        if i == 0:
            baby.append(x_ct)
        else:
            baby.append(call_he_rotate(rt, x_ct, i, f"{x_ct.name}_rot_{i}"))

    acc = None
    for j in range(h):
        shift = j * g
        inner = None

        for i in range(g):
            s = shift + i
            if s >= L:
                break

            shifted_diag = plain_rotate(periodic_diags[s], -shift, sigma)
            term = call_he_mult_plain(rt, baby[i], shifted_diag.tolist(), f"{x_ct.name}_mul_s{s}")
            inner = term if inner is None else call_he_add(rt, inner, term, f"{x_ct.name}_inner_j{j}")

        if inner is None:
            continue

        shifted = inner if shift == 0 else call_he_rotate(rt, inner, shift, f"{x_ct.name}_giant_{shift}")
        acc = shifted if acc is None else call_he_add(rt, acc, shifted, f"{x_ct.name}_acc_j{j}")

    return acc


def bench_bsgs_same_case(rt, x_block, w_block, sigma):
    rows, _ = x_block.shape
    slots_per_ct = rt.info()["ring_dim"] // 2
    periodic_diags, L = build_periodic_diagonals(w_block, sigma=sigma, physical_slots=slots_per_ct)

    t0 = now_us()
    row_cts_in = []
    for r in range(rows):
        row_cts_in.append(
            encrypt_periodic_row(rt, x_block[r], L, slots_per_ct, f"bsgs_x_r{r}")
        )
    t1 = now_us()

    row_cts_out = []
    t2 = now_us()
    for r in range(rows):
        row_cts_out.append(
            he_diagonal_bsgs_single_row_periodic(rt, row_cts_in[r], periodic_diags, sigma=sigma)
        )
    t3 = now_us()

    y_rows = []
    t4 = now_us()
    out_dim = w_block.shape[1]
    for r in range(rows):
        full_dec = np.array(
            rt.decrypt(row_cts_out[r].ct, logical_length=slots_per_ct),
            dtype=np.float64,
        )
        y_rows.append(full_dec[:out_dim])
    t5 = now_us()

    return {
        "y_he": np.stack(y_rows, axis=0),
        "encrypt_us": t1 - t0,
        "core_us": t3 - t2,
        "decrypt_us": t5 - t4,
        "L": L,
        "physical_slots": slots_per_ct,
    }


def bench_naive_same_case(rt, x_block, w_block, split_dim):
    rows, mid_dim = x_block.shape
    _, out_dim = w_block.shape

    encrypt_us = 0.0
    row_cts_out = []

    t0 = now_us()
    for r in range(rows):
        ct_y_r, enc_us_r, core_us_r = he_naive_single_row_split_timed(
            rt, x_block[r], w_block, split_dim, f"naive_r{r}"
        )
        row_cts_out.append((ct_y_r, core_us_r))
        encrypt_us += enc_us_r
    t1 = now_us()

    # core_us_r 已经单独累计，这里不用 t1-t0 再算 core
    core_us = sum(v for _, v in row_cts_out)

    y_rows = []
    t2 = now_us()
    for r in range(rows):
        y_r = np.array(
            rt.decrypt(row_cts_out[r][0].ct, logical_length=out_dim),
            dtype=np.float64,
        )[:out_dim]
        y_rows.append(y_r)
    t3 = now_us()

    return {
        "y_he": np.stack(y_rows, axis=0),
        "encrypt_us": encrypt_us,
        "core_us": core_us,
        "decrypt_us": t3 - t2,
    }


def run_case(case_name, rows, mid_dim, out_dim, split_dim, plain_repeats=5, he_repeats=1, seed=0):
    rng = np.random.default_rng(seed)
    x_block = rng.normal(0.0, 1.0, size=(rows, mid_dim)).astype(np.float64)
    w_block = rng.normal(0.0, 0.2, size=(mid_dim, out_dim)).astype(np.float64)

    plain_result = bench_plain_dense_reference(x_block, w_block, repeats=plain_repeats)
    y_plain = plain_result.pop("y_plain_reference")

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

    sigma_last = None

    bsgs_encrypt_list = []
    bsgs_core_list = []
    bsgs_decrypt_list = []
    bsgs_err_list = []
    bsgs_y_last = None

    naive_encrypt_list = []
    naive_core_list = []
    naive_decrypt_list = []
    naive_err_list = []
    naive_y_last = None

    with HERuntime(cfg, rotation_steps=list(range(-1024, 1025))) as rt:
        sigma = detect_rotation_sigma(rt)
        sigma_last = sigma

        for _ in range(he_repeats):
            bsgs_res = bench_bsgs_same_case(rt, x_block, w_block, sigma)
            bsgs_y_last = bsgs_res["y_he"]
            bsgs_encrypt_list.append(bsgs_res["encrypt_us"])
            bsgs_core_list.append(bsgs_res["core_us"])
            bsgs_decrypt_list.append(bsgs_res["decrypt_us"])
            bsgs_err_list.append(float(np.max(np.abs(y_plain - bsgs_res["y_he"]))))

            naive_res = bench_naive_same_case(rt, x_block, w_block, split_dim)
            naive_y_last = naive_res["y_he"]
            naive_encrypt_list.append(naive_res["encrypt_us"])
            naive_core_list.append(naive_res["core_us"])
            naive_decrypt_list.append(naive_res["decrypt_us"])
            naive_err_list.append(float(np.max(np.abs(y_plain - naive_res["y_he"]))))

    result = {
        "case_name": case_name,
        "rows": rows,
        "mid_dim": mid_dim,
        "out_dim": out_dim,
        "split_dim": split_dim,
        "plain_repeats": plain_repeats,
        "he_repeats": he_repeats,
        "bsgs_input_ct_count_per_run": rows,
        "naive_input_ct_count_per_run": rows * (mid_dim // split_dim),
        "output_ct_count_per_run": rows,
        **plain_result,
        "rotation_sigma": sigma_last,

        "bsgs_encrypt_us_median": median(bsgs_encrypt_list),
        "bsgs_core_us_median": median(bsgs_core_list),
        "bsgs_decrypt_us_median": median(bsgs_decrypt_list),
        "bsgs_plain_vs_he_max_abs_err": max(bsgs_err_list) if bsgs_err_list else None,
        "bsgs_y_first_row": bsgs_y_last[0].tolist(),

        "naive_encrypt_us_median": median(naive_encrypt_list),
        "naive_core_us_median": median(naive_core_list),
        "naive_decrypt_us_median": median(naive_decrypt_list),
        "naive_plain_vs_he_max_abs_err": max(naive_err_list) if naive_err_list else None,
        "naive_y_first_row": naive_y_last[0].tolist(),
    }

    if result["plain_reference_us_median"] and result["bsgs_core_us_median"]:
        result["bsgs_core_vs_plain_ratio"] = (
            result["bsgs_core_us_median"] / result["plain_reference_us_median"]
        )
    else:
        result["bsgs_core_vs_plain_ratio"] = None

    if result["plain_reference_us_median"] and result["naive_core_us_median"]:
        result["naive_core_vs_plain_ratio"] = (
            result["naive_core_us_median"] / result["plain_reference_us_median"]
        )
    else:
        result["naive_core_vs_plain_ratio"] = None

    if result["naive_core_us_median"] and result["bsgs_core_us_median"]:
        result["bsgs_speedup_over_naive_core"] = (
            result["naive_core_us_median"] / result["bsgs_core_us_median"]
        )
    else:
        result["bsgs_speedup_over_naive_core"] = None

    return result


def main():
    print_sep("大尺度实验 / 案例1")
    case1 = run_case(
        case_name="large_multi_ct_16x128__128x128",
        rows=16,
        mid_dim=128,
        out_dim=128,
        split_dim=16,
        plain_repeats=5,
        he_repeats=1,
        seed=203101,
    )
    zh_print(case1)

    print_sep("大尺度实验 / 案例2")
    case2 = run_case(
        case_name="large_multi_ct_32x128__128x128",
        rows=32,
        mid_dim=128,
        out_dim=128,
        split_dim=16,
        plain_repeats=5,
        he_repeats=1,
        seed=203102,
    )
    zh_print(case2)

    print_sep("大尺度实验 / 案例3")
    case3 = run_case(
        case_name="large_multi_ct_16x128__128x256",
        rows=16,
        mid_dim=128,
        out_dim=256,
        split_dim=16,
        plain_repeats=5,
        he_repeats=1,
        seed=203103,
    )
    zh_print(case3)

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3b_large_scale_compare_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "case_1_large_multi_ct_square": case1,
                "case_2_large_multi_ct_more_rows": case2,
                "case_3_large_multi_ct_non_square_output": case3,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_sep("报告已保存")
    print(f"  JSON路径: {out_path}")


if __name__ == "__main__":
    main()
