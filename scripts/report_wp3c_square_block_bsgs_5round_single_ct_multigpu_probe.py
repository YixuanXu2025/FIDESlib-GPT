import math
import time
import hashlib
import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.tensor import CipherTensor
from hegpt.ops import he_add, he_mult_plain, he_rotate


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
        layout="square_block_bsgs_5round_single_ct_multigpu_probe",
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


def print_rounds(rounds_bsgs, final_err_bsgs):
    print_sep("单输入密文 + 连续5轮同一W + 最后解密（BSGS-only multi-GPU probe）")
    header = (
        f"{'轮次':<6}"
        f"{'cache构建(us)':>16}{'命中':>8}{'新建':>8}{'核心(us)':>16}"
        f"{'pMult均值(us)':>16}{'add均值(us)':>14}{'rotate均值(us)':>16}"
    )
    print(header)
    print("-" * len(header))

    for r in rounds_bsgs:
        print(
            f"{r['round_idx']:<6}"
            f"{r['cache_build_us']:>16.3f}{r['cache_hits']:>8}{r['cache_misses']:>8}{r['core_us']:>16.3f}"
            f"{r['primitive_stats']['pMult_avg_us']:>16.3f}"
            f"{r['primitive_stats']['add_avg_us']:>14.3f}"
            f"{r['primitive_stats']['rotate_avg_us']:>16.3f}"
        )

    tail = rounds_bsgs[1:]
    print("-" * len(header))
    print("第2轮及以后中位数:")
    print(f"  cache构建(us): {median([r['cache_build_us'] for r in tail]):.3f}")
    print(f"  核心计算(us):  {median([r['core_us'] for r in tail]):.3f}")
    print(f"  pMult均值(us): {median([r['primitive_stats']['pMult_avg_us'] for r in tail]):.3f}")
    print(f"  add均值(us):   {median([r['primitive_stats']['add_avg_us'] for r in tail]):.3f}")
    print(f"  rotate均值(us): {median([r['primitive_stats']['rotate_avg_us'] for r in tail]):.3f}")
    print(f"最终一次解密后的误差(max_abs_err): {final_err_bsgs:.6e}")


def main():
    D = 64
    NUM_ROUNDS = 5
    SEED = 203601

    rng = np.random.default_rng(SEED)
    X = rng.normal(0.0, 1.0, size=(D, D)).astype(np.float64)
    W = rng.normal(0.0, 0.2, size=(D, D)).astype(np.float64)

    Y_plain = X.copy()
    for _ in range(NUM_ROUNDS):
        Y_plain = Y_plain @ W

    sigma = detect_rotation_sigma()
    needed_rotation_steps = collect_bsgs_rotation_steps(D)

    cfg = HEConfig(
        ring_dim=1 << 17,
        multiplicative_depth=29,
        scaling_mod_size=59,
        first_mod_size=60,
        num_large_digits=3,
        batch_size=4096,
        devices=(0, 1, 2, 3, 4, 5, 6, 7),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    BSGS_PLAIN_CACHE.clear()

    print_sep("多卡验证配置")
    print(f"devices = {cfg.devices}")
    print(f"rotation_steps_count = {len(needed_rotation_steps)}")

    with HERuntime(cfg, rotation_steps=needed_rotation_steps) as rt:
        print("[runtime.info()]", rt.info())

        t0 = now_us()
        ct_current = matrix_to_periodic_cipher(rt, X, "X_init_bsgs")
        t1 = now_us()
        encrypt_once_us = t1 - t0

        slots_per_ct = get_physical_slots(rt)
        rounds_bsgs = []

        for k in range(NUM_ROUNDS):
            t2 = now_us()
            cached_obj, hit = get_or_build_bsgs_cached_plain_tables(W, D, sigma, slots_per_ct)
            t3 = now_us()

            t4 = now_us()
            ct_next, stats = square_block_bsgs_kernel_timed_cached(
                rt, ct_current, cached_obj, D, sigma, f"bsgs_round{k+1}"
            )
            t5 = now_us()

            rounds_bsgs.append({
                "round_idx": k + 1,
                "cache_build_us": t3 - t2,
                "cache_hits": int(hit),
                "cache_misses": int(not hit),
                "core_us": t5 - t4,
                "primitive_stats": stats,
            })
            ct_current = ct_next

        t6 = now_us()
        Y_bsgs = decode_periodic_square_output(rt, ct_current, D)
        t7 = now_us()
        decrypt_once_us = t7 - t6

    final_err_bsgs = float(np.max(np.abs(Y_plain - Y_bsgs)))

    print_sep("总体信息")
    print(f"输入大小: ({D}, {D})")
    print(f"W大小: ({D}, {D})")
    print(f"连续轮数: {NUM_ROUNDS}")
    print(f"rotation sigma: {sigma}")
    print(f"实际生成的 rotation key 数量: {len(needed_rotation_steps)}")
    print(f"仅首次加密一次(us): {encrypt_once_us:.3f}")
    print(f"仅最后解密一次(us): {decrypt_once_us:.3f}")

    print_rounds(rounds_bsgs, final_err_bsgs)

    out = {
        "devices": list(cfg.devices),
        "input_shape": (D, D),
        "weight_shape": (D, D),
        "num_rounds": NUM_ROUNDS,
        "sigma": sigma,
        "needed_rotation_steps": needed_rotation_steps,
        "encrypt_once_us": encrypt_once_us,
        "decrypt_once_us": decrypt_once_us,
        "final_max_abs_err_bsgs": final_err_bsgs,
        "rounds_bsgs": rounds_bsgs,
        "post_round2_summary": {
            "cache_build_us_median": median([r["cache_build_us"] for r in rounds_bsgs[1:]]),
            "core_us_median": median([r["core_us"] for r in rounds_bsgs[1:]]),
            "pmult_avg_us_median": median([r["primitive_stats"]["pMult_avg_us"] for r in rounds_bsgs[1:]]),
            "add_avg_us_median": median([r["primitive_stats"]["add_avg_us"] for r in rounds_bsgs[1:]]),
            "rotate_avg_us_median": median([r["primitive_stats"]["rotate_avg_us"] for r in rounds_bsgs[1:]]),
        },
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3c_square_block_bsgs_5round_single_ct_multigpu_probe_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print_sep("报告已保存")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
