import json
import math
import time
import hashlib
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.tensor import CipherTensor
from hegpt.ops import he_add, he_mult_plain, he_rotate


# ============================================================
# 全局 Python/RAM cache
# ============================================================
BSGS_PLAIN_CACHE = {}


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


def safe_avg(total, count):
    return total / count if count > 0 else None


def zh_print_direct(direct_stats: dict):
    print("  ---- 同尺度底层 primitive 单次基准 ----")
    print(f"    方块维度 d: {direct_stats['square_dim']}")
    print(f"    物理槽数: {direct_stats['physical_slots']}")
    print(f"    direct pMult 单次中位(us): {direct_stats['direct_pmult_single_us_median']}")
    print(f"    direct add   单次中位(us): {direct_stats['direct_add_single_us_median']}")
    print(f"    direct rotate 单次中位(us): {direct_stats['direct_rotate_single_us_median']}")
    print(f"    rotate步长集合: {direct_stats['rotate_steps_used']}")


def zh_print_kernel_stats(tag: str, stats: dict):
    print(f"  ---- {tag} 内核内 primitive 统计 ----")
    print(f"    pMult 次数: {stats['pMult_count']}")
    print(f"    pMult 总时间(us): {stats['pMult_total_us']}")
    print(f"    pMult 平均单次(us): {stats['pMult_avg_us']}")
    print(f"    add 次数: {stats['add_count']}")
    print(f"    add 总时间(us): {stats['add_total_us']}")
    print(f"    add 平均单次(us): {stats['add_avg_us']}")
    print(f"    rotate 次数: {stats['rotate_count']}")
    print(f"    rotate 总时间(us): {stats['rotate_total_us']}")
    print(f"    rotate 平均单次(us): {stats['rotate_avg_us']}")


def zh_print_case(result: dict):
    print(f"  案例名称: {result['case_name']}")
    print(f"  方块维度 d: {result['square_dim']}")
    print(f"  原始输入形状: {result['orig_input_shape']}")
    print(f"  原始权重形状: {result['orig_weight_shape']}")
    print(f"  明文参考中位时间(us): {result['plain_reference_us_median']}")
    print(f"  加密中位时间(us): {result['encrypt_us_median']}")
    print(f"  BSGS cache构建时间(us): {result['bsgs_cache_build_us']}")
    print(f"  BSGS cache命中块数: {result['bsgs_cache_hits']}")
    print(f"  BSGS cache新建块数: {result['bsgs_cache_misses']}")
    print(f"  BSGS核心计算中位时间(us): {result['bsgs_core_us_median']}")
    print(f"  JKLS核心计算中位时间(us): {result['jkls_core_us_median']}")
    print(f"  解密中位时间(us): {result['decrypt_us_median']}")
    print(f"  BSGS正确性误差(max_abs_err): {result['bsgs_plain_vs_he_max_abs_err']}")
    print(f"  JKLS正确性误差(max_abs_err): {result['jkls_plain_vs_he_max_abs_err']}")
    print(f"  BSGS首行前8个输出: {result['bsgs_y_he_first_row'][:8]}")
    print(f"  JKLS首行前8个输出: {result['jkls_y_he_first_row'][:8]}")
    print(f"  BSGS核心/明文倍数: {result['bsgs_core_vs_plain_ratio']}")
    print(f"  JKLS核心/明文倍数: {result['jkls_core_vs_plain_ratio']}")
    print(f"  JKLS相对BSGS核心倍率(JKLS/BSGS): {result['jkls_core_over_bsgs_core_ratio']}")
    print(f"  direct估算 BSGS核心(us): {result['bsgs_estimated_core_from_direct_us']}")
    print(f"  direct估算 JKLS核心(us): {result['jkls_estimated_core_from_direct_us']}")
    zh_print_kernel_stats("BSGS", result["bsgs_kernel_primitive_stats"])
    zh_print_kernel_stats("JKLS", result["jkls_kernel_primitive_stats"])


def estimate_kernel_from_direct(stats, direct_stats):
    return (
        stats["pMult_count"] * direct_stats["direct_pmult_single_us_median"]
        + stats["add_count"] * direct_stats["direct_add_single_us_median"]
        + stats["rotate_count"] * direct_stats["direct_rotate_single_us_median"]
    )


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


def timed_he_add(rt, a, b, name, stats):
    t0 = now_us()
    out = call_he_add(rt, a, b, name)
    t1 = now_us()
    stats["add_count"] += 1
    stats["add_total_us"] += (t1 - t0)
    return out


def timed_he_mult_plain(rt, x, plain_vec, name, stats):
    t0 = now_us()
    out = call_he_mult_plain(rt, x, plain_vec, name)
    t1 = now_us()
    stats["pMult_count"] += 1
    stats["pMult_total_us"] += (t1 - t0)
    return out


def timed_he_rotate(rt, x, steps, name, stats):
    t0 = now_us()
    out = call_he_rotate(rt, x, steps, name)
    t1 = now_us()
    stats["rotate_count"] += 1
    stats["rotate_total_us"] += (t1 - t0)
    return out


def finalize_stats(stats):
    stats["pMult_avg_us"] = safe_avg(stats["pMult_total_us"], stats["pMult_count"])
    stats["add_avg_us"] = safe_avg(stats["add_total_us"], stats["add_count"])
    stats["rotate_avg_us"] = safe_avg(stats["rotate_total_us"], stats["rotate_count"])
    return stats


def make_cipher_tensor(rt, packed_vec, logical_dim, name):
    ct = rt.encrypt([float(v) for v in packed_vec])
    device = None
    if hasattr(rt, "cfg") and getattr(rt.cfg, "devices", None):
        device = rt.cfg.devices[0]
    return CipherTensor(
        ct=ct,
        shape=(logical_dim,),
        layout="square_block_compare",
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
    raise RuntimeError(f"无法检测旋转方向: {out.tolist()}")


def flatten_row_major(mat):
    return np.asarray(mat, dtype=np.float64).reshape(-1)


def periodic_pack(vec, physical_slots):
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    L = len(vec)
    if physical_slots % L != 0:
        raise ValueError(f"physical_slots={physical_slots} 必须能被 L={L} 整除")
    repeat = physical_slots // L
    return np.tile(vec, repeat)


def pad_to_square_block(x, d):
    x = np.asarray(x, dtype=np.float64)
    r, c = x.shape
    out = np.zeros((d, d), dtype=np.float64)
    out[:r, :c] = x
    return out


def matrix_to_periodic_cipher(rt, mat_square, name):
    d = mat_square.shape[0]
    slots_per_ct = rt.info()["ring_dim"] // 2
    flat = flatten_row_major(mat_square)
    packed = periodic_pack(flat, slots_per_ct)
    return make_cipher_tensor(rt, packed, d * d, name)


def decode_periodic_square_output(rt, ct, d, out_rows, out_cols):
    slots_per_ct = rt.info()["ring_dim"] // 2
    full = np.array(rt.decrypt(ct.ct, logical_length=slots_per_ct), dtype=np.float64)
    base = full[: d * d].reshape(d, d)
    return base[:out_rows, :out_cols]


# ============================================================
# BSGS cache helpers
# ============================================================
def hash_weight_block(B_square):
    arr = np.ascontiguousarray(B_square, dtype=np.float64)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def build_signed_diagonal_periodic(B_square, d, signed_shift, sigma, physical_slots):
    N = d * d
    diag = np.zeros((N,), dtype=np.float64)

    for c in range(N):
        out_row = c // d
        out_col = c % d

        r = (c + sigma * signed_shift) % N
        in_row = r // d
        in_col = r % d

        if in_row == out_row:
            diag[c] = B_square[in_col, out_col]

    return periodic_pack(diag, physical_slots)


def build_signed_diagonal_map(B_square, d, sigma, physical_slots):
    diag_map = {}
    for s in range(-(d - 1), d):
        diag_map[s] = build_signed_diagonal_periodic(B_square, d, s, sigma, physical_slots)
    return diag_map


def build_bsgs_schedule(d):
    # 正半部分: base_shift=0, m=d
    g_pos = int(math.ceil(math.sqrt(d)))
    h_pos = int(math.ceil(d / g_pos))
    pos_terms = []
    for j in range(h_pos):
        shift = j * g_pos
        for i in range(g_pos):
            t = shift + i
            if t >= d:
                break
            signed_shift = t
            pos_terms.append((signed_shift, shift))

    # 负半部分: base_shift=-(d-1), m=d-1
    m_neg = d - 1
    g_neg = int(math.ceil(math.sqrt(m_neg)))
    h_neg = int(math.ceil(m_neg / g_neg))
    neg_terms = []
    for j in range(h_neg):
        shift = j * g_neg
        for i in range(g_neg):
            t = shift + i
            if t >= m_neg:
                break
            signed_shift = -(d - 1) + t
            neg_terms.append((signed_shift, shift))

    return {
        "pos": {
            "base_shift": 0,
            "m": d,
            "g": g_pos,
            "h": h_pos,
            "terms": pos_terms,
        },
        "neg": {
            "base_shift": -(d - 1),
            "m": d - 1,
            "g": g_neg,
            "h": h_neg,
            "terms": neg_terms,
        },
    }


def get_or_build_bsgs_cached_plain_tables(B_square, d, sigma, physical_slots):
    key = (hash_weight_block(B_square), d, sigma, physical_slots)

    if key in BSGS_PLAIN_CACHE:
        return BSGS_PLAIN_CACHE[key], True

    schedule = build_bsgs_schedule(d)
    diag_map = build_signed_diagonal_map(B_square, d, sigma, physical_slots)

    shifted_tables = {}
    for phase_name in ("pos", "neg"):
        shifted_tables[phase_name] = {}
        for signed_shift, shift in schedule[phase_name]["terms"]:
            shifted_tables[phase_name][(signed_shift, shift)] = plain_rotate(
                diag_map[signed_shift], -shift, sigma
            ).tolist()

    obj = {
        "schedule": schedule,
        "shifted_tables": shifted_tables,
    }
    BSGS_PLAIN_CACHE[key] = obj
    return obj, False


def collect_rotate_steps_for_square_block_bsgs(d):
    g_pos = int(math.ceil(math.sqrt(d)))
    h_pos = int(math.ceil(d / g_pos))

    g_neg = int(math.ceil(math.sqrt(d - 1)))
    h_neg = int(math.ceil((d - 1) / g_neg))

    steps = []
    for i in range(1, g_pos):
        steps.append(i)
    for j in range(1, h_pos):
        steps.append(j * g_pos)

    steps.append(-(d - 1))
    for i in range(1, g_neg):
        steps.append(i)
    for j in range(1, h_neg):
        steps.append(j * g_neg)

    dedup = []
    seen = set()
    for s in steps:
        if s not in seen:
            dedup.append(s)
            seen.add(s)
    return dedup


def bsgs_contiguous_signed_timed_cached(rt, x_ct, cached_plain, phase_name, name_prefix, stats):
    phase = cached_plain["schedule"][phase_name]
    shifted_table = cached_plain["shifted_tables"][phase_name]

    base_shift = phase["base_shift"]
    g = phase["g"]
    h = phase["h"]
    m = phase["m"]

    x_base = x_ct if base_shift == 0 else timed_he_rotate(rt, x_ct, base_shift, f"{name_prefix}_base", stats)

    baby = []
    for i in range(g):
        if i == 0:
            baby.append(x_base)
        else:
            baby.append(timed_he_rotate(rt, x_base, i, f"{name_prefix}_baby_{i}", stats))

    acc = None
    for j in range(h):
        shift = j * g
        inner = None

        for i in range(g):
            t = shift + i
            if t >= m:
                break

            if phase_name == "pos":
                signed_shift = t
            else:
                signed_shift = base_shift + t

            shifted_diag_list = shifted_table[(signed_shift, shift)]
            term = timed_he_mult_plain(rt, baby[i], shifted_diag_list, f"{name_prefix}_mul_{signed_shift}", stats)

            inner = term if inner is None else timed_he_add(rt, inner, term, f"{name_prefix}_inner_{j}", stats)

        if inner is None:
            continue

        shifted = inner if shift == 0 else timed_he_rotate(rt, inner, shift, f"{name_prefix}_giant_{shift}", stats)
        acc = shifted if acc is None else timed_he_add(rt, acc, shifted, f"{name_prefix}_acc_{j}", stats)

    return acc


def square_block_bsgs_kernel_timed_cached(rt, ct_A, cached_plain, d, sigma, name_prefix):
    stats = {
        "pMult_count": 0,
        "pMult_total_us": 0.0,
        "add_count": 0,
        "add_total_us": 0.0,
        "rotate_count": 0,
        "rotate_total_us": 0.0,
    }

    pos = bsgs_contiguous_signed_timed_cached(
        rt, ct_A, cached_plain, "pos", f"{name_prefix}_pos", stats
    )
    neg = bsgs_contiguous_signed_timed_cached(
        rt, ct_A, cached_plain, "neg", f"{name_prefix}_neg", stats
    )

    out = timed_he_add(rt, pos, neg, f"{name_prefix}_sum_pos_neg", stats)
    return out, finalize_stats(stats)


# ============================================================
# JKLS
# ============================================================
def column_mask_plain_periodic(d, k, physical_slots):
    mask = np.zeros((d * d,), dtype=np.float64)
    for i in range(d):
        mask[i * d + k] = 1.0
    return periodic_pack(mask, physical_slots)


def repeat_plain_row_periodic(row_vec, d, physical_slots):
    row_vec = np.asarray(row_vec, dtype=np.float64).reshape(-1)
    mat = np.tile(row_vec.reshape(1, d), (d, 1))
    return periodic_pack(flatten_row_major(mat), physical_slots)


def replicate_rowwise_from_col0_timed(rt, ct_col0, d, sigma, name_prefix, stats):
    acc = ct_col0
    step = 1
    while step < d:
        rot_step = -sigma * step
        shifted = timed_he_rotate(rt, acc, rot_step, f"{name_prefix}_rep_{step}", stats)
        acc = timed_he_add(rt, acc, shifted, f"{name_prefix}_rep_add_{step}", stats)
        step *= 2
    return acc


def jkls_square_kernel_pcmm_timed(rt, ct_A, B_square, d, sigma, name_prefix):
    slots_per_ct = rt.info()["ring_dim"] // 2
    B_square = np.asarray(B_square, dtype=np.float64)

    stats = {
        "pMult_count": 0,
        "pMult_total_us": 0.0,
        "add_count": 0,
        "add_total_us": 0.0,
        "rotate_count": 0,
        "rotate_total_us": 0.0,
    }

    acc = None
    for k in range(d):
        col_mask = column_mask_plain_periodic(d, k, slots_per_ct)
        ct_col = timed_he_mult_plain(rt, ct_A, col_mask.tolist(), f"{name_prefix}_maskcol_{k}", stats)

        move_step = sigma * k
        ct_col0 = ct_col if k == 0 else timed_he_rotate(rt, ct_col, move_step, f"{name_prefix}_movecol_{k}", stats)

        ct_rowrep = replicate_rowwise_from_col0_timed(rt, ct_col0, d, sigma, f"{name_prefix}_repl_{k}", stats)

        plain_rowrep = repeat_plain_row_periodic(B_square[k, :], d, slots_per_ct)
        term = timed_he_mult_plain(rt, ct_rowrep, plain_rowrep.tolist(), f"{name_prefix}_mulrow_{k}", stats)

        acc = term if acc is None else timed_he_add(rt, acc, term, f"{name_prefix}_acc_{k}", stats)

    return acc, finalize_stats(stats)


def collect_rotate_steps_for_jkls(d, sigma):
    steps = []
    for k in range(1, d):
        steps.append(sigma * k)
    step = 1
    while step < d:
        steps.append(-sigma * step)
        step *= 2

    dedup = []
    seen = set()
    for s in steps:
        if s not in seen:
            dedup.append(s)
            seen.add(s)
    return dedup


# ============================================================
# Direct primitives
# ============================================================
def bench_direct_primitives_same_scale(rt, d, sigma, repeats=30):
    slots_per_ct = rt.info()["ring_dim"] // 2

    rng = np.random.default_rng(203399)
    A = rng.normal(0.0, 1.0, size=(d, d)).astype(np.float64)
    B = rng.normal(0.0, 1.0, size=(d, d)).astype(np.float64)
    W = rng.normal(0.0, 0.2, size=(d, d)).astype(np.float64)

    ct_A = matrix_to_periodic_cipher(rt, A, "direct_ct_A")
    ct_B = matrix_to_periodic_cipher(rt, B, "direct_ct_B")

    diag0 = build_signed_diagonal_periodic(W, d, 0, sigma, slots_per_ct).tolist()
    rotate_steps = sorted(set(collect_rotate_steps_for_square_block_bsgs(d) + collect_rotate_steps_for_jkls(d, sigma)))

    pmult_times = []
    for i in range(repeats):
        t0 = now_us()
        _ = call_he_mult_plain(rt, ct_A, diag0, f"direct_pmult_{i}")
        t1 = now_us()
        pmult_times.append(t1 - t0)

    add_times = []
    for i in range(repeats):
        t0 = now_us()
        _ = call_he_add(rt, ct_A, ct_B, f"direct_add_{i}")
        t1 = now_us()
        add_times.append(t1 - t0)

    rotate_times = []
    per_step_repeats = max(2, repeats // max(len(rotate_steps), 1) + 1)
    for step in rotate_steps:
        for i in range(per_step_repeats):
            t0 = now_us()
            _ = call_he_rotate(rt, ct_A, step, f"direct_rot_{step}_{i}")
            t1 = now_us()
            rotate_times.append(t1 - t0)

    return {
        "square_dim": d,
        "physical_slots": slots_per_ct,
        "direct_pmult_single_us_median": median(pmult_times),
        "direct_add_single_us_median": median(add_times),
        "direct_rotate_single_us_median": median(rotate_times),
        "rotate_steps_used": rotate_steps,
    }


# ============================================================
# Per-case runner
# ============================================================
def bench_plain_reference(A_small, B_full):
    return A_small @ B_full


def run_case(rt, sigma, direct_stats, case_name, A_shape, B_shape, d=64, repeats=1, seed=0):
    rng = np.random.default_rng(seed)

    A_small = rng.normal(0.0, 1.0, size=A_shape).astype(np.float64)
    B_full = rng.normal(0.0, 0.2, size=B_shape).astype(np.float64)

    plain_times = []
    y_plain_last = None
    for _ in range(5):
        t0 = now_us()
        y_plain = bench_plain_reference(A_small, B_full)
        t1 = now_us()
        plain_times.append(t1 - t0)
        y_plain_last = y_plain

    out_cols = B_shape[1]
    num_out_blocks = math.ceil(out_cols / d)

    # 先做 BSGS cache 预计算，不放进 core timing
    slots_per_ct = rt.info()["ring_dim"] // 2
    bsgs_cached_blocks = []
    cache_build_us = 0.0
    cache_hits = 0
    cache_misses = 0

    for j in range(num_out_blocks):
        c0 = j * d
        c1 = min((j + 1) * d, out_cols)

        B_block = np.zeros((d, d), dtype=np.float64)
        B_block[:, : (c1 - c0)] = B_full[:, c0:c1]

        t0 = now_us()
        cached_obj, hit = get_or_build_bsgs_cached_plain_tables(B_block, d, sigma, slots_per_ct)
        t1 = now_us()

        cache_build_us += (t1 - t0)
        cache_hits += int(hit)
        cache_misses += int(not hit)
        bsgs_cached_blocks.append((cached_obj, c1 - c0, B_block))

    encrypt_list = []
    decrypt_list = []

    bsgs_core_list = []
    bsgs_err_list = []
    bsgs_y_last = None
    bsgs_kernel_stats_last = None

    jkls_core_list = []
    jkls_err_list = []
    jkls_y_last = None
    jkls_kernel_stats_last = None

    for _ in range(repeats):
        A_square = pad_to_square_block(A_small, d)

        t0 = now_us()
        ct_A = matrix_to_periodic_cipher(rt, A_square, "A_square_shared")
        t1 = now_us()
        encrypt_list.append(t1 - t0)

        # -------- BSGS cached --------
        bsgs_out_cts = []
        merged_bsgs = {
            "pMult_count": 0, "pMult_total_us": 0.0,
            "add_count": 0, "add_total_us": 0.0,
            "rotate_count": 0, "rotate_total_us": 0.0,
        }

        t2 = now_us()
        for j, (cached_plain, valid_cols, _) in enumerate(bsgs_cached_blocks):
            ct_Cj, stats_j = square_block_bsgs_kernel_timed_cached(
                rt, ct_A, cached_plain, d, sigma, f"SBBSGS_blk_{j}"
            )
            bsgs_out_cts.append((ct_Cj, valid_cols))
            for k in merged_bsgs.keys():
                merged_bsgs[k] += stats_j[k]
        t3 = now_us()
        bsgs_core_list.append(t3 - t2)
        bsgs_kernel_stats_last = finalize_stats(merged_bsgs)

        # -------- JKLS --------
        jkls_out_cts = []
        merged_jkls = {
            "pMult_count": 0, "pMult_total_us": 0.0,
            "add_count": 0, "add_total_us": 0.0,
            "rotate_count": 0, "rotate_total_us": 0.0,
        }

        t4 = now_us()
        for j, (_, valid_cols, B_block) in enumerate(bsgs_cached_blocks):
            ct_Cj, stats_j = jkls_square_kernel_pcmm_timed(
                rt, ct_A, B_block, d, sigma, f"JKLS_blk_{j}"
            )
            jkls_out_cts.append((ct_Cj, valid_cols))
            for k in merged_jkls.keys():
                merged_jkls[k] += stats_j[k]
        t5 = now_us()
        jkls_core_list.append(t5 - t4)
        jkls_kernel_stats_last = finalize_stats(merged_jkls)

        # 统一解密
        t6 = now_us()
        bsgs_parts = []
        for ct_Cj, valid_cols in bsgs_out_cts:
            bsgs_parts.append(decode_periodic_square_output(rt, ct_Cj, d, A_shape[0], valid_cols))
        bsgs_y = np.concatenate(bsgs_parts, axis=1)

        jkls_parts = []
        for ct_Cj, valid_cols in jkls_out_cts:
            jkls_parts.append(decode_periodic_square_output(rt, ct_Cj, d, A_shape[0], valid_cols))
        jkls_y = np.concatenate(jkls_parts, axis=1)
        t7 = now_us()
        decrypt_list.append(t7 - t6)

        bsgs_y_last = bsgs_y
        jkls_y_last = jkls_y
        bsgs_err_list.append(float(np.max(np.abs(y_plain_last - bsgs_y))))
        jkls_err_list.append(float(np.max(np.abs(y_plain_last - jkls_y))))

    result = {
        "case_name": case_name,
        "square_dim": d,
        "orig_input_shape": A_shape,
        "orig_weight_shape": B_shape,
        "plain_reference_us_median": median(plain_times),
        "encrypt_us_median": median(encrypt_list),
        "decrypt_us_median": median(decrypt_list),

        "bsgs_cache_build_us": cache_build_us,
        "bsgs_cache_hits": cache_hits,
        "bsgs_cache_misses": cache_misses,

        "bsgs_core_us_median": median(bsgs_core_list),
        "bsgs_plain_vs_he_max_abs_err": max(bsgs_err_list) if bsgs_err_list else None,
        "bsgs_y_he_first_row": bsgs_y_last[0].tolist(),
        "bsgs_kernel_primitive_stats": bsgs_kernel_stats_last,

        "jkls_core_us_median": median(jkls_core_list),
        "jkls_plain_vs_he_max_abs_err": max(jkls_err_list) if jkls_err_list else None,
        "jkls_y_he_first_row": jkls_y_last[0].tolist(),
        "jkls_kernel_primitive_stats": jkls_kernel_stats_last,
    }

    if result["plain_reference_us_median"] and result["bsgs_core_us_median"]:
        result["bsgs_core_vs_plain_ratio"] = result["bsgs_core_us_median"] / result["plain_reference_us_median"]
    else:
        result["bsgs_core_vs_plain_ratio"] = None

    if result["plain_reference_us_median"] and result["jkls_core_us_median"]:
        result["jkls_core_vs_plain_ratio"] = result["jkls_core_us_median"] / result["plain_reference_us_median"]
    else:
        result["jkls_core_vs_plain_ratio"] = None

    if result["bsgs_core_us_median"] and result["jkls_core_us_median"]:
        result["jkls_core_over_bsgs_core_ratio"] = result["jkls_core_us_median"] / result["bsgs_core_us_median"]
    else:
        result["jkls_core_over_bsgs_core_ratio"] = None

    result["bsgs_estimated_core_from_direct_us"] = estimate_kernel_from_direct(
        result["bsgs_kernel_primitive_stats"], direct_stats
    )
    result["jkls_estimated_core_from_direct_us"] = estimate_kernel_from_direct(
        result["jkls_kernel_primitive_stats"], direct_stats
    )

    return result


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

    with HERuntime(cfg, rotation_steps=list(range(-512, 513))) as rt:
        sigma = detect_rotation_sigma(rt)

        print_sep("同尺度底层 primitive 基准")
        direct_stats = bench_direct_primitives_same_scale(rt, d=64, sigma=sigma, repeats=40)
        zh_print_direct(direct_stats)

        print_sep("BSGS(cache) vs JKLS / 案例1")
        case1 = run_case(
            rt, sigma, direct_stats,
            case_name="compare_cached_64x64__64x64",
            A_shape=(64, 64),
            B_shape=(64, 64),
            d=64,
            repeats=1,
            seed=203401,
        )
        zh_print_case(case1)

        print_sep("BSGS(cache) vs JKLS / 案例2")
        case2 = run_case(
            rt, sigma, direct_stats,
            case_name="compare_cached_8x64__64x64",
            A_shape=(8, 64),
            B_shape=(64, 64),
            d=64,
            repeats=1,
            seed=203402,
        )
        zh_print_case(case2)

        print_sep("BSGS(cache) vs JKLS / 案例3")
        case3 = run_case(
            rt, sigma, direct_stats,
            case_name="compare_cached_8x64__64x128",
            A_shape=(8, 64),
            B_shape=(64, 128),
            d=64,
            repeats=1,
            seed=203403,
        )
        zh_print_case(case3)

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3c_square_block_bsgs_primitive_timing_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "direct_primitive_same_scale": direct_stats,
                "case_1_square": case1,
                "case_2_rect_input": case2,
                "case_3_rect_output": case3,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_sep("报告已保存")
    print(f"  JSON路径: {out_path}")


if __name__ == "__main__":
    main()
