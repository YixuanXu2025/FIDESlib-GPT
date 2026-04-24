import time
import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.tensor import CipherTensor
from hegpt.ops import he_add, he_mult_plain, he_rotate


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
        layout="square_block_jkls_5round_single_ct",
        slots_used=len(packed_vec),
        level=None,
        scale=None,
        device=device,
        name=name,
    )


def detect_rotation_sigma():
    # 这里只是测 rotate 方向，不跑正式实验，所以继续用轻量参数
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


def print_rounds(rounds_jkls, final_err_jkls):
    print_sep("单输入密文 + 连续5轮同一W + 最后解密（JKLS-only）")
    header = (
        f"{'轮次':<6}"
        f"{'核心(us)':>16}"
        f"{'pMult均值(us)':>16}"
        f"{'add均值(us)':>14}"
        f"{'rotate均值(us)':>16}"
    )
    print(header)
    print("-" * len(header))

    for r in rounds_jkls:
        print(
            f"{r['round_idx']:<6}"
            f"{r['core_us']:>16.3f}"
            f"{r['primitive_stats']['pMult_avg_us']:>16.3f}"
            f"{r['primitive_stats']['add_avg_us']:>14.3f}"
            f"{r['primitive_stats']['rotate_avg_us']:>16.3f}"
        )

    tail = rounds_jkls[1:]
    print("-" * len(header))
    print("第2轮及以后中位数:")
    print(f"  核心计算(us):  {median([r['core_us'] for r in tail]):.3f}")
    print(f"  pMult均值(us): {median([r['primitive_stats']['pMult_avg_us'] for r in tail]):.3f}")
    print(f"  add均值(us):   {median([r['primitive_stats']['add_avg_us'] for r in tail]):.3f}")
    print(f"  rotate均值(us): {median([r['primitive_stats']['rotate_avg_us'] for r in tail]):.3f}")
    print(f"最终一次解密后的误差(max_abs_err): {final_err_jkls:.6e}")


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
    needed_rotation_steps = collect_jkls_rotation_steps(D, sigma)

    cfg = HEConfig(
        ring_dim=1 << 17,
        multiplicative_depth=29,
        scaling_mod_size=59,
        first_mod_size=60,
        num_large_digits=3,
        batch_size=4096,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    with HERuntime(cfg, rotation_steps=needed_rotation_steps) as rt:
        t0 = now_us()
        ct_current = matrix_to_periodic_cipher(rt, X, "X_init_jkls")
        t1 = now_us()
        encrypt_once_us = t1 - t0

        rounds_jkls = []

        for k in range(NUM_ROUNDS):
            t2 = now_us()
            ct_next, stats = jkls_square_kernel_pcmm_timed(
                rt, ct_current, W, D, sigma, f"jkls_round{k+1}"
            )
            t3 = now_us()

            rounds_jkls.append({
                "round_idx": k + 1,
                "core_us": t3 - t2,
                "primitive_stats": stats,
            })
            ct_current = ct_next

        t4 = now_us()
        Y_jkls = decode_periodic_square_output(rt, ct_current, D)
        t5 = now_us()
        decrypt_once_us = t5 - t4

    final_err_jkls = float(np.max(np.abs(Y_plain - Y_jkls)))

    print_sep("总体信息")
    print(f"输入大小: ({D}, {D})")
    print(f"W大小: ({D}, {D})")
    print(f"连续轮数: {NUM_ROUNDS}")
    print(f"rotation sigma: {sigma}")
    print(f"实际生成的 rotation key 数量: {len(needed_rotation_steps)}")
    print(f"仅首次加密一次(us): {encrypt_once_us:.3f}")
    print(f"仅最后解密一次(us): {decrypt_once_us:.3f}")

    print_rounds(rounds_jkls, final_err_jkls)

    out = {
        "input_shape": (D, D),
        "weight_shape": (D, D),
        "num_rounds": NUM_ROUNDS,
        "sigma": sigma,
        "needed_rotation_steps": needed_rotation_steps,
        "encrypt_once_us": encrypt_once_us,
        "decrypt_once_us": decrypt_once_us,
        "final_max_abs_err_jkls": final_err_jkls,
        "rounds_jkls": rounds_jkls,
        "post_round2_summary": {
            "core_us_median": median([r["core_us"] for r in rounds_jkls[1:]]),
            "pmult_avg_us_median": median([r["primitive_stats"]["pMult_avg_us"] for r in rounds_jkls[1:]]),
            "add_avg_us_median": median([r["primitive_stats"]["add_avg_us"] for r in rounds_jkls[1:]]),
            "rotate_avg_us_median": median([r["primitive_stats"]["rotate_avg_us"] for r in rounds_jkls[1:]]),
        },
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3c_square_block_jkls_5round_single_ct_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print_sep("报告已保存")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
