import math
import time
import hashlib
import json
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


def safe_avg(total, count):
    return total / count if count > 0 else None


def print_sep(title: str):
    print("=" * 100)
    print(title)


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
        layout="square_block_bsgs_5round_single_ct",
        slots_used=len(packed_vec),
        level=None,
        scale=None,
        device=device,
        name=name,
    )


def plain_rotate(vec, steps, sigma):
    vec = np.asarray(vec, dtype=np.float64)
    return np.roll(vec, -sigma * steps)


def detect_rotation_sigma():
    """
    用一个轻量 context 探测 rotate 方向，避免在重参数上下文里额外生成很多 key。
    sigma 只反映语义方向，不依赖你最终的大参数设置。
    """
    cfg = HEConfig(
        ring_dim=16384,
        multiplicative_depth=1,
        scaling_mod_size=40,
        batch_size=8,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=False,
    )

    with HERuntime(cfg, rotation_steps=[-1, 1]) as rt:
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


def get_physical_slots(rt):
    cfg = getattr(rt, "cfg", None)
    bs = getattr(cfg, "batch_size", None) if cfg is not None else None
    if bs is not None:
        return int(bs)
    return int(rt.info()["ring_dim"] // 2)


def periodic_pack(vec, physical_slots):
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    L = len(vec)
    if physical_slots % L != 0:
        raise ValueError(f"physical_slots={physical_slots} 必须能被 L={L} 整除")
    repeat = physical_slots // L
    return np.tile(vec, repeat)


def matrix_to_periodic_cipher(rt, mat_square, name):
    d = mat_square.shape[0]
    slots_per_ct = get_physical_slots(rt)
    flat = flatten_row_major(mat_square)
    packed = periodic_pack(flat, slots_per_ct)
    return make_cipher_tensor(rt, packed, d * d, name)


def decode_periodic_square_output(rt, ct, d):
    slots_per_ct = get_physical_slots(rt)
    full = np.array(rt.decrypt(ct.ct, logical_length=slots_per_ct), dtype=np.float64)
    return full[: d * d].reshape(d, d)


# ============================================================
# rotation steps
# ============================================================
def collect_bsgs_rotation_steps(d):
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

    out = []
    seen = set()
    for s in steps:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def collect_jkls_rotation_steps(d, sigma):
    steps = []
    for k in range(1, d):
        steps.append(sigma * k)
    step = 1
    while step < d:
        steps.append(-sigma * step)
        step *= 2

    out = []
    seen = set()
    for s in steps:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


# ============================================================
# BSGS cache
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
    slots_per_ct = get_physical_slots(rt)
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


# ============================================================
# direct primitive timing
# ============================================================
def bench_direct_primitives_same_scale(rt, d, sigma, needed_rotation_steps, repeats=12):
    slots_per_ct = get_physical_slots(rt)
    rng = np.random.default_rng(203799)

    A = rng.normal(0.0, 1.0, size=(d, d)).astype(np.float64)
    B = rng.normal(0.0, 1.0, size=(d, d)).astype(np.float64)
    W = rng.normal(0.0, 0.2, size=(d, d)).astype(np.float64)

    ct_A = matrix_to_periodic_cipher(rt, A, "direct_ct_A")
    ct_B = matrix_to_periodic_cipher(rt, B, "direct_ct_B")

    diag0 = build_signed_diagonal_periodic(W, d, 0, sigma, slots_per_ct).tolist()

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
    per_step_repeats = 2
    for step in needed_rotation_steps:
        for i in range(per_step_repeats):
            t0 = now_us()
            _ = call_he_rotate(rt, ct_A, step, f"direct_rot_{step}_{i}")
            t1 = now_us()
            rotate_times.append(t1 - t0)

    return {
        "direct_pmult_single_us_median": median(pmult_times),
        "direct_add_single_us_median": median(add_times),
        "direct_rotate_single_us_median": median(rotate_times),
        "rotate_steps_used": needed_rotation_steps,
    }


# ============================================================
# main run
# ============================================================
def print_direct_stats(stats):
    print_sep("131072 / 29 / 59 同参数 single-primitive timing")
    print(f"direct pMult 单次中位(us): {stats['direct_pmult_single_us_median']:.3f}")
    print(f"direct add   单次中位(us): {stats['direct_add_single_us_median']:.3f}")
    print(f"direct rotate单次中位(us): {stats['direct_rotate_single_us_median']:.3f}")
    print(f"rotation steps: {stats['rotate_steps_used']}")


def print_round_compare(rounds_bsgs, rounds_jkls, final_err_bsgs, final_err_jkls):
    print_sep("单输入密文 + 连续5轮同一W：BSGS vs JKLS")
    header = (
        f"{'轮次':<6}"
        f"{'B_cache(us)':>14}{'B_hit':>8}{'B_new':>8}{'B_core(us)':>16}"
        f"{'J_core(us)':>16}"
        f"{'B_pM(us)':>12}{'J_pM(us)':>12}"
        f"{'B_add(us)':>12}{'J_add(us)':>12}"
        f"{'B_rot(us)':>12}{'J_rot(us)':>12}"
    )
    print(header)
    print("-" * len(header))

    for i in range(len(rounds_bsgs)):
        rb = rounds_bsgs[i]
        rj = rounds_jkls[i]
        print(
            f"{i+1:<6}"
            f"{rb['cache_build_us']:>14.3f}{rb['cache_hits']:>8}{rb['cache_misses']:>8}{rb['core_us']:>16.3f}"
            f"{rj['core_us']:>16.3f}"
            f"{rb['primitive_stats']['pMult_avg_us']:>12.3f}{rj['primitive_stats']['pMult_avg_us']:>12.3f}"
            f"{rb['primitive_stats']['add_avg_us']:>12.3f}{rj['primitive_stats']['add_avg_us']:>12.3f}"
            f"{rb['primitive_stats']['rotate_avg_us']:>12.3f}{rj['primitive_stats']['rotate_avg_us']:>12.3f}"
        )

    tail_b = rounds_bsgs[1:]
    tail_j = rounds_jkls[1:]

    print("-" * len(header))
    print("第2轮及以后中位数:")
    print(f"  BSGS cache构建(us): {median([r['cache_build_us'] for r in tail_b]):.3f}")
    print(f"  BSGS core(us):      {median([r['core_us'] for r in tail_b]):.3f}")
    print(f"  JKLS core(us):      {median([r['core_us'] for r in tail_j]):.3f}")
    print(f"  BSGS pMult均值(us): {median([r['primitive_stats']['pMult_avg_us'] for r in tail_b]):.3f}")
    print(f"  JKLS pMult均值(us): {median([r['primitive_stats']['pMult_avg_us'] for r in tail_j]):.3f}")
    print(f"  BSGS add均值(us):   {median([r['primitive_stats']['add_avg_us'] for r in tail_b]):.3f}")
    print(f"  JKLS add均值(us):   {median([r['primitive_stats']['add_avg_us'] for r in tail_j]):.3f}")
    print(f"  BSGS rot均值(us):   {median([r['primitive_stats']['rotate_avg_us'] for r in tail_b]):.3f}")
    print(f"  JKLS rot均值(us):   {median([r['primitive_stats']['rotate_avg_us'] for r in tail_j]):.3f}")
    print(f"最终一次解密后的 BSGS 误差(max_abs_err): {final_err_bsgs:.6e}")
    print(f"最终一次解密后的 JKLS 误差(max_abs_err): {final_err_jkls:.6e}")


def main():
    D = 64
    NUM_ROUNDS = 5
    SEED = 203601

    rng = np.random.default_rng(SEED)
    X = rng.normal(0.0, 1.0, size=(D, D)).astype(np.float64)
    W = rng.normal(0.0, 0.2, size=(D, D)).astype(np.float64)

    # 明文参考：连续乘5次同一个 W
    Y_plain = X.copy()
    for _ in range(NUM_ROUNDS):
        Y_plain = Y_plain @ W

    sigma = detect_rotation_sigma()

    bsgs_steps = collect_bsgs_rotation_steps(D)
    jkls_steps = collect_jkls_rotation_steps(D, sigma)
    needed_rotation_steps = sorted(set(bsgs_steps + jkls_steps))

    cfg = HEConfig(
        # 沿用你这次已经跑通的大参数
        ring_dim=131072,
        multiplicative_depth=29,
        scaling_mod_size=59,
        batch_size=4096,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    BSGS_PLAIN_CACHE.clear()

    with HERuntime(cfg, rotation_steps=needed_rotation_steps) as rt:
        # 同参数下的 single-primitive timing
        direct_stats = bench_direct_primitives_same_scale(
            rt, D, sigma, needed_rotation_steps, repeats=12
        )

        # 只各加密一次
        t0 = now_us()
        ct_init_bsgs = matrix_to_periodic_cipher(rt, X, "X_init_bsgs")
        t1 = now_us()
        encrypt_once_bsgs_us = t1 - t0

        t2 = now_us()
        ct_init_jkls = matrix_to_periodic_cipher(rt, X, "X_init_jkls")
        t3 = now_us()
        encrypt_once_jkls_us = t3 - t2

        slots_per_ct = get_physical_slots(rt)

        rounds_bsgs = []
        ct_current_bsgs = ct_init_bsgs

        for k in range(NUM_ROUNDS):
            # BSGS cache 查询/构建
            t4 = now_us()
            cached_obj, hit = get_or_build_bsgs_cached_plain_tables(W, D, sigma, slots_per_ct)
            t5 = now_us()
            cache_build_us = t5 - t4

            t6 = now_us()
            ct_next, stats = square_block_bsgs_kernel_timed_cached(
                rt, ct_current_bsgs, cached_obj, D, sigma, f"bsgs_round{k+1}"
            )
            t7 = now_us()

            rounds_bsgs.append({
                "round_idx": k + 1,
                "cache_build_us": cache_build_us,
                "cache_hits": int(hit),
                "cache_misses": int(not hit),
                "core_us": t7 - t6,
                "primitive_stats": stats,
            })
            ct_current_bsgs = ct_next

        rounds_jkls = []
        ct_current_jkls = ct_init_jkls

        for k in range(NUM_ROUNDS):
            t8 = now_us()
            ct_next, stats = jkls_square_kernel_pcmm_timed(
                rt, ct_current_jkls, W, D, sigma, f"jkls_round{k+1}"
            )
            t9 = now_us()

            rounds_jkls.append({
                "round_idx": k + 1,
                "core_us": t9 - t8,
                "primitive_stats": stats,
            })
            ct_current_jkls = ct_next

        # 最后各解密一次
        t10 = now_us()
        Y_bsgs = decode_periodic_square_output(rt, ct_current_bsgs, D)
        t11 = now_us()
        decrypt_once_bsgs_us = t11 - t10

        t12 = now_us()
        Y_jkls = decode_periodic_square_output(rt, ct_current_jkls, D)
        t13 = now_us()
        decrypt_once_jkls_us = t13 - t12

    final_err_bsgs = float(np.max(np.abs(Y_plain - Y_bsgs)))
    final_err_jkls = float(np.max(np.abs(Y_plain - Y_jkls)))

    print_sep("总体信息")
    print(f"输入大小: ({D}, {D})")
    print(f"W大小: ({D}, {D})")
    print(f"连续轮数: {NUM_ROUNDS}")
    print(f"rotation sigma: {sigma}")
    print(f"实际生成的 rotation key 数量: {len(needed_rotation_steps)}")
    print(f"仅首次加密一次(BSGS) (us): {encrypt_once_bsgs_us:.3f}")
    print(f"仅首次加密一次(JKLS) (us): {encrypt_once_jkls_us:.3f}")
    print(f"仅最后解密一次(BSGS) (us): {decrypt_once_bsgs_us:.3f}")
    print(f"仅最后解密一次(JKLS) (us): {decrypt_once_jkls_us:.3f}")

    print_direct_stats(direct_stats)
    print_round_compare(rounds_bsgs, rounds_jkls, final_err_bsgs, final_err_jkls)

    out = {
        "input_shape": (D, D),
        "weight_shape": (D, D),
        "num_rounds": NUM_ROUNDS,
        "sigma": sigma,
        "needed_rotation_steps": needed_rotation_steps,
        "encrypt_once_bsgs_us": encrypt_once_bsgs_us,
        "encrypt_once_jkls_us": encrypt_once_jkls_us,
        "decrypt_once_bsgs_us": decrypt_once_bsgs_us,
        "decrypt_once_jkls_us": decrypt_once_jkls_us,
        "direct_primitive_stats": direct_stats,
        "final_max_abs_err_bsgs": final_err_bsgs,
        "final_max_abs_err_jkls": final_err_jkls,
        "rounds_bsgs": rounds_bsgs,
        "rounds_jkls": rounds_jkls,
        "post_round2_summary": {
            "bsgs_cache_build_us_median": median([r["cache_build_us"] for r in rounds_bsgs[1:]]),
            "bsgs_core_us_median": median([r["core_us"] for r in rounds_bsgs[1:]]),
            "jkls_core_us_median": median([r["core_us"] for r in rounds_jkls[1:]]),
            "bsgs_pmult_avg_us_median": median([r["primitive_stats"]["pMult_avg_us"] for r in rounds_bsgs[1:]]),
            "jkls_pmult_avg_us_median": median([r["primitive_stats"]["pMult_avg_us"] for r in rounds_jkls[1:]]),
            "bsgs_add_avg_us_median": median([r["primitive_stats"]["add_avg_us"] for r in rounds_bsgs[1:]]),
            "jkls_add_avg_us_median": median([r["primitive_stats"]["add_avg_us"] for r in rounds_jkls[1:]]),
            "bsgs_rotate_avg_us_median": median([r["primitive_stats"]["rotate_avg_us"] for r in rounds_bsgs[1:]]),
            "jkls_rotate_avg_us_median": median([r["primitive_stats"]["rotate_avg_us"] for r in rounds_jkls[1:]]),
        },
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3c_square_block_bsgs_5round_single_ct_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print_sep("报告已保存")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
