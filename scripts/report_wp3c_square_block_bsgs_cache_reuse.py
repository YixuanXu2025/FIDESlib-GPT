import math
import time
import hashlib
from pathlib import Path
import json

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
        layout="square_block_bsgs_cache_reuse",
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


def bench_plain_reference(A_small, B_full):
    return A_small @ B_full


def run_one_round(rt, sigma, A_small, B_full, d, round_name):
    out_cols = B_full.shape[1]
    num_out_blocks = math.ceil(out_cols / d)
    slots_per_ct = rt.info()["ring_dim"] // 2

    cache_build_us = 0.0
    cache_hits = 0
    cache_misses = 0
    cached_blocks = []

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
        cached_blocks.append((cached_obj, c1 - c0))

    A_square = pad_to_square_block(A_small, d)
    t0 = now_us()
    ct_A = matrix_to_periodic_cipher(rt, A_square, f"A_square_{round_name}")
    t1 = now_us()
    encrypt_us = t1 - t0

    merged_stats = {
        "pMult_count": 0, "pMult_total_us": 0.0,
        "add_count": 0, "add_total_us": 0.0,
        "rotate_count": 0, "rotate_total_us": 0.0,
    }

    out_cts = []
    t2 = now_us()
    for j, (cached_plain, valid_cols) in enumerate(cached_blocks):
        ct_Cj, stats_j = square_block_bsgs_kernel_timed_cached(
            rt, ct_A, cached_plain, d, sigma, f"{round_name}_blk_{j}"
        )
        out_cts.append((ct_Cj, valid_cols))
        for k in merged_stats.keys():
            merged_stats[k] += stats_j[k]
    t3 = now_us()
    core_us = t3 - t2
    merged_stats = finalize_stats(merged_stats)

    t4 = now_us()
    parts = []
    for ct_Cj, valid_cols in out_cts:
        part = decode_periodic_square_output(rt, ct_Cj, d, A_small.shape[0], valid_cols)
        parts.append(part)
    y_he = np.concatenate(parts, axis=1)
    t5 = now_us()
    decrypt_us = t5 - t4

    return {
        "round_name": round_name,
        "cache_build_us": cache_build_us,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "encrypt_us": encrypt_us,
        "core_us": core_us,
        "decrypt_us": decrypt_us,
        "primitive_stats": merged_stats,
        "y_he": y_he,
    }


def summarize_post_round2(rounds):
    tail = rounds[1:]  # 第2轮及以后
    return {
        "cache_build_us_median": median([r["cache_build_us"] for r in tail]),
        "cache_hits_median": median([r["cache_hits"] for r in tail]),
        "cache_misses_median": median([r["cache_misses"] for r in tail]),
        "encrypt_us_median": median([r["encrypt_us"] for r in tail]),
        "core_us_median": median([r["core_us"] for r in tail]),
        "decrypt_us_median": median([r["decrypt_us"] for r in tail]),
        "pMult_avg_us_median": median([r["primitive_stats"]["pMult_avg_us"] for r in tail]),
        "add_avg_us_median": median([r["primitive_stats"]["add_avg_us"] for r in tail]),
        "rotate_avg_us_median": median([r["primitive_stats"]["rotate_avg_us"] for r in tail]),
        "max_abs_err_median": median([r["max_abs_err"] for r in tail]),
    }


def print_round_table(case_name, plain_ref_us, rounds, summary):
    print_sep(f"同一case多轮复用测试 / {case_name}")
    header = f"{'轮次':<8}{'cache构建(us)':>16}{'命中':>8}{'新建':>8}{'加密(us)':>14}{'核心(us)':>14}{'解密(us)':>14}{'误差':>16}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rounds, start=1):
        print(
            f"{i:<8}"
            f"{r['cache_build_us']:>16.3f}"
            f"{r['cache_hits']:>8}"
            f"{r['cache_misses']:>8}"
            f"{r['encrypt_us']:>14.3f}"
            f"{r['core_us']:>14.3f}"
            f"{r['decrypt_us']:>14.3f}"
            f"{r['max_abs_err']:>16.6e}"
        )

    print("-" * len(header))
    print(f"明文参考时间(us): {plain_ref_us:.3f}")
    print("第2轮及以后中位数:")
    print(f"  cache构建时间(us): {summary['cache_build_us_median']:.3f}")
    print(f"  cache命中块数: {summary['cache_hits_median']}")
    print(f"  cache新建块数: {summary['cache_misses_median']}")
    print(f"  加密时间(us): {summary['encrypt_us_median']:.3f}")
    print(f"  核心计算时间(us): {summary['core_us_median']:.3f}")
    print(f"  解密时间(us): {summary['decrypt_us_median']:.3f}")
    print(f"  正确性误差(max_abs_err): {summary['max_abs_err_median']:.6e}")
    print(f"  pMult平均单次(us): {summary['pMult_avg_us_median']:.3f}")
    print(f"  add平均单次(us): {summary['add_avg_us_median']:.3f}")
    print(f"  rotate平均单次(us): {summary['rotate_avg_us_median']:.3f}")


def main():
    CASE_NAME = "cache_reuse_8x64__64x64"
    A_SHAPE = (8, 64)
    B_SHAPE = (64, 64)
    D = 64
    NUM_ROUNDS = 5
    SEED = 203501

    rng = np.random.default_rng(SEED)
    A_small = rng.normal(0.0, 1.0, size=A_SHAPE).astype(np.float64)
    B_full = rng.normal(0.0, 0.2, size=B_SHAPE).astype(np.float64)

    plain_times = []
    y_plain_last = None
    for _ in range(5):
        t0 = now_us()
        y_plain = bench_plain_reference(A_small, B_full)
        t1 = now_us()
        plain_times.append(t1 - t0)
        y_plain_last = y_plain
    plain_ref_us = median(plain_times)

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

    rounds = []

    with HERuntime(cfg, rotation_steps=list(range(-512, 513))) as rt:
        sigma = detect_rotation_sigma(rt)

        for k in range(NUM_ROUNDS):
            result = run_one_round(rt, sigma, A_small, B_full, D, f"round{k+1}")
            result["max_abs_err"] = float(np.max(np.abs(y_plain_last - result["y_he"])))
            rounds.append(result)

    summary = summarize_post_round2(rounds)
    print_round_table(CASE_NAME, plain_ref_us, rounds, summary)

    out = {
        "case_name": CASE_NAME,
        "A_shape": A_SHAPE,
        "B_shape": B_SHAPE,
        "square_dim": D,
        "num_rounds": NUM_ROUNDS,
        "plain_reference_us_median": plain_ref_us,
        "rounds": [
            {
                "round_name": r["round_name"],
                "cache_build_us": r["cache_build_us"],
                "cache_hits": r["cache_hits"],
                "cache_misses": r["cache_misses"],
                "encrypt_us": r["encrypt_us"],
                "core_us": r["core_us"],
                "decrypt_us": r["decrypt_us"],
                "max_abs_err": r["max_abs_err"],
                "primitive_stats": r["primitive_stats"],
            }
            for r in rounds
        ],
        "post_round2_summary": summary,
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3c_square_block_bsgs_cache_reuse_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print_sep("报告已保存")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
