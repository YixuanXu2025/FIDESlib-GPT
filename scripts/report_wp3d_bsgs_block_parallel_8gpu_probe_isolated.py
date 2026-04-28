import math
import time
import hashlib
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


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


def plain_rotate(vec, steps, sigma):
    vec = np.asarray(vec, dtype=np.float64)
    return np.roll(vec, -sigma * steps)


def flatten_row_major(mat):
    return np.asarray(mat, dtype=np.float64).reshape(-1)


def periodic_pack(vec, physical_slots):
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    L = len(vec)
    if physical_slots % L != 0:
        raise ValueError(f"physical_slots={physical_slots} must be divisible by L={L}")
    repeat = physical_slots // L
    return np.tile(vec, repeat)


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
            pos_terms.append((t, shift))

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


def get_or_build_bsgs_cached_plain_tables(cache, B_square, d, sigma, physical_slots):
    key = (hash_weight_block(B_square), d, sigma, physical_slots)

    if key in cache:
        return cache[key], True

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
    cache[key] = obj
    return obj, False


def get_physical_slots(rt):
    cfg = getattr(rt, "cfg", None)
    bs = getattr(cfg, "batch_size", None) if cfg is not None else None
    if bs is not None:
        return int(bs)
    return int(rt.info()["ring_dim"] // 2)


def call_he_add(rt, a, b, name):
    from hegpt.ops import he_add
    try:
        return he_add(rt, a, b, name=name)
    except TypeError:
        return he_add(rt, a, b)


def call_he_mult_plain(rt, x, plain_vec, name):
    from hegpt.ops import he_mult_plain
    try:
        return he_mult_plain(rt, x, plain_vec, name=name)
    except TypeError:
        return he_mult_plain(rt, x, plain_vec)


def call_he_rotate(rt, x, steps, name):
    from hegpt.ops import he_rotate
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
    from hegpt.tensor import CipherTensor

    ct = rt.encrypt([float(v) for v in packed_vec])
    device = None
    if hasattr(rt, "cfg") and getattr(rt.cfg, "devices", None):
        device = rt.cfg.devices[0]

    return CipherTensor(
        ct=ct,
        shape=(logical_dim,),
        layout="wp3d_bsgs_block_parallel_8gpu_probe",
        slots_used=len(packed_vec),
        level=None,
        scale=None,
        device=device,
        name=name,
    )


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


def bsgs_contiguous_signed_timed_cached(rt, x_ct, cached_plain, phase_name, name_prefix, stats):
    phase = cached_plain["schedule"][phase_name]
    shifted_table = cached_plain["shifted_tables"][phase_name]

    base_shift = phase["base_shift"]
    g = phase["g"]
    h = phase["h"]
    m = phase["m"]

    x_base = x_ct if base_shift == 0 else timed_he_rotate(
        rt, x_ct, base_shift, f"{name_prefix}_base", stats
    )

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
            term = timed_he_mult_plain(
                rt, baby[i], shifted_diag_list, f"{name_prefix}_mul_{signed_shift}", stats
            )

            inner = term if inner is None else timed_he_add(
                rt, inner, term, f"{name_prefix}_inner_{j}", stats
            )

        if inner is None:
            continue

        shifted = inner if shift == 0 else timed_he_rotate(
            rt, inner, shift, f"{name_prefix}_giant_{shift}", stats
        )
        acc = shifted if acc is None else timed_he_add(
            rt, acc, shifted, f"{name_prefix}_acc_{j}", stats
        )

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


def run_one_worker(task_id, physical_devices, seed):
    # 关键：每个 worker 进程只暴露自己的物理 GPU 对。
    # 暴露后，进程内部看到的 GPU 会被重编号为 local 0, local 1。
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in physical_devices)
    local_devices = (0, 1)

    from hegpt import HEConfig, HERuntime

    D = 64
    NUM_ROUNDS = 5
    sigma = 1

    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(D, D)).astype(np.float64)
    W = rng.normal(0.0, 0.2, size=(D, D)).astype(np.float64)

    Y_plain = X.copy()
    for _ in range(NUM_ROUNDS):
        Y_plain = Y_plain @ W

    rotation_steps = collect_bsgs_rotation_steps(D)

    cfg = HEConfig(
        ring_dim=1 << 17,
        multiplicative_depth=29,
        scaling_mod_size=59,
        first_mod_size=60,
        num_large_digits=3,
        batch_size=4096,
        devices=local_devices,
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    local_cache = {}

    t_worker0 = now_us()

    with HERuntime(cfg, rotation_steps=rotation_steps) as rt:
        runtime_info = rt.info()

        t0 = now_us()
        ct_current = matrix_to_periodic_cipher(rt, X, f"task{task_id}_X_init")
        t1 = now_us()
        encrypt_once_us = t1 - t0

        slots_per_ct = get_physical_slots(rt)
        rounds = []

        for k in range(NUM_ROUNDS):
            t2 = now_us()
            cached_obj, hit = get_or_build_bsgs_cached_plain_tables(
                local_cache, W, D, sigma, slots_per_ct
            )
            t3 = now_us()

            t4 = now_us()
            ct_next, stats = square_block_bsgs_kernel_timed_cached(
                rt, ct_current, cached_obj, D, sigma, f"task{task_id}_bsgs_round{k+1}"
            )
            t5 = now_us()

            rounds.append({
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

    t_worker1 = now_us()

    final_err = float(np.max(np.abs(Y_plain - Y_bsgs)))

    tail = rounds[1:]
    result = {
        "task_id": task_id,
        "pid": os.getpid(),
        "physical_devices": list(physical_devices),
        "local_devices": list(local_devices),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": seed,
        "runtime_info": runtime_info,
        "input_shape": [D, D],
        "weight_shape": [D, D],
        "num_rounds": NUM_ROUNDS,
        "rotation_steps_count": len(rotation_steps),
        "encrypt_once_us": encrypt_once_us,
        "decrypt_once_us": decrypt_once_us,
        "worker_wall_us": t_worker1 - t_worker0,
        "final_max_abs_err": final_err,
        "rounds": rounds,
        "post_round2_summary": {
            "cache_build_us_median": median([r["cache_build_us"] for r in tail]),
            "core_us_median": median([r["core_us"] for r in tail]),
            "pmult_avg_us_median": median([r["primitive_stats"]["pMult_avg_us"] for r in tail]),
            "add_avg_us_median": median([r["primitive_stats"]["add_avg_us"] for r in tail]),
            "rotate_avg_us_median": median([r["primitive_stats"]["rotate_avg_us"] for r in tail]),
        },
    }
    return result


def print_summary(results, parallel_wall_us):
    print("=" * 110)
    print("WP3-D BSGS block-parallel 8GPU probe")
    print("=" * 110)
    print(f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE')}")
    print(f"CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}")
    print(f"parallel_wall_us={parallel_wall_us:.3f}")
    print()

    header = (
        f"{'task':<6}"
        f"{'physical':<18}"
        f"{'local':<12}"
        f"{'err':>14}"
        f"{'worker_wall(us)':>18}"
        f"{'core_med(us)':>16}"
        f"{'pMult(us)':>12}"
        f"{'add(us)':>12}"
        f"{'rot(us)':>12}"
        f"{'enc(us)':>12}"
        f"{'dec(us)':>12}"
    )
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: x["task_id"]):
        s = r["post_round2_summary"]
        print(
            f"{r['task_id']:<6}"
            f"{str(tuple(r['physical_devices'])):<18}"
            f"{str(tuple(r['local_devices'])):<12}"
            f"{r['final_max_abs_err']:>14.6e}"
            f"{r['worker_wall_us']:>18.3f}"
            f"{s['core_us_median']:>16.3f}"
            f"{s['pmult_avg_us_median']:>12.3f}"
            f"{s['add_avg_us_median']:>12.3f}"
            f"{s['rotate_avg_us_median']:>12.3f}"
            f"{r['encrypt_once_us']:>12.3f}"
            f"{r['decrypt_once_us']:>12.3f}"
        )

    seq_sum = sum(r["worker_wall_us"] for r in results)
    print("-" * len(header))
    print(f"sum_worker_wall_us={seq_sum:.3f}")
    print(f"parallel_wall_us={parallel_wall_us:.3f}")
    print(f"estimated_throughput_speedup=sum_worker_wall/parallel_wall={seq_sum / parallel_wall_us:.3f}x")
    print(f"max_err={max(r['final_max_abs_err'] for r in results):.6e}")


def main():
    gpu_groups = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]

    base_seed = 204000
    tasks = [
        {
            "task_id": i,
            "devices": gpu_groups[i],
            "seed": base_seed + i,
        }
        for i in range(len(gpu_groups))
    ]

    print("=" * 110)
    print("Launching block-parallel BSGS workers")
    print(f"gpu_groups={gpu_groups}")
    print(f"num_workers={len(tasks)}")
    print("=" * 110)

    ctx = mp.get_context("spawn")
    t0 = now_us()

    results = []
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=ctx) as ex:
        futs = [
            ex.submit(run_one_worker, t["task_id"], t["devices"], t["seed"])
            for t in tasks
        ]

        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(
                f"[done] task={r['task_id']} physical={tuple(r['physical_devices'])} "
                f"local={tuple(r['local_devices'])} "
                f"CUDA_VISIBLE_DEVICES={r['cuda_visible_devices']} "
                f"err={r['final_max_abs_err']:.6e} "
                f"worker_wall_us={r['worker_wall_us']:.3f}"
            )

    t1 = now_us()
    parallel_wall_us = t1 - t0

    print_summary(results, parallel_wall_us)

    out = {
        "experiment": "wp3d_bsgs_block_parallel_8gpu_probe_isolated",
        "gpu_groups": [list(g) for g in gpu_groups],
        "parallel_wall_us": parallel_wall_us,
        "sum_worker_wall_us": sum(r["worker_wall_us"] for r in results),
        "estimated_throughput_speedup": sum(r["worker_wall_us"] for r in results) / parallel_wall_us,
        "max_err": max(r["final_max_abs_err"] for r in results),
        "results": sorted(results, key=lambda x: x["task_id"]),
        "env": {
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG"),
            "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE"),
            "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3d_bsgs_block_parallel_8gpu_probe_isolated_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=" * 110)
    print("报告已保存")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
