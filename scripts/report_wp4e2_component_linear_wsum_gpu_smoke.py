import json
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def now_us():
    return time.perf_counter() * 1e6


def add_many_repeated(rt, ct, k: int):
    if k <= 0:
        return None
    acc = ct
    for _ in range(k - 1):
        acc = rt.add_ct(acc, ct)
    return acc


def highlevel_nonnegative_int_wsum(rt, ct_rows, weights):
    acc = None
    for ct, k in zip(ct_rows, weights):
        k = int(k)
        if k == 0:
            continue
        term = add_many_repeated(rt, ct, k)
        if acc is None:
            acc = term
        else:
            acc = rt.add_ct(acc, term)
    if acc is None:
        raise RuntimeError("all-zero weights are not supported in this smoke script")
    return acc


def main():
    n_in = 4
    slots = 8
    seed = 205402

    rng = np.random.default_rng(seed)
    M = rng.normal(0.0, 0.25, size=(n_in, slots)).astype(np.float64)

    weights = np.array([1, 0, 1, 2], dtype=np.int64)
    y_ref = weights.astype(np.float64) @ M

    cfg = HEConfig(
        ring_dim=1 << 14,
        multiplicative_depth=2,
        scaling_mod_size=50,
        first_mod_size=60,
        num_large_digits=2,
        batch_size=slots,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    with HERuntime(cfg, rotation_steps=[]) as rt:
        t0 = now_us()
        ct_rows = [rt.encrypt(M[i].tolist()) for i in range(n_in)]
        t1 = now_us()

        in_states = [rt.ciphertext_storage_state(ct) for ct in ct_rows]

        dec_in = np.array([
            rt.decrypt(ct_rows[i], logical_length=slots)
            for i in range(n_in)
        ], dtype=np.float64)

        t2 = now_us()
        ct_gpu = rt.component_linear_wsum_gpu(ct_rows, weights.astype(np.float64).tolist())
        t3 = now_us()

        gpu_state = rt.ciphertext_storage_state(ct_gpu)

        t4 = now_us()
        ct_high = highlevel_nonnegative_int_wsum(rt, ct_rows, weights.tolist())
        t5 = now_us()

        high_state = rt.ciphertext_storage_state(ct_high)

        dec_gpu = np.array(rt.decrypt(ct_gpu, logical_length=slots), dtype=np.float64)
        dec_high = np.array(rt.decrypt(ct_high, logical_length=slots), dtype=np.float64)
        t6 = now_us()

    err_input = float(np.max(np.abs(dec_in - M)))
    err_gpu_to_ref = float(np.max(np.abs(dec_gpu - y_ref)))
    err_high_to_ref = float(np.max(np.abs(dec_high - y_ref)))
    err_gpu_to_high = float(np.max(np.abs(dec_gpu - dec_high)))

    report = {
        "experiment": "wp4e2_component_linear_wsum_gpu_smoke",
        "n_in": n_in,
        "slots": slots,
        "seed": seed,
        "weights": weights.tolist(),
        "err_input": err_input,
        "err_gpu_to_ref": err_gpu_to_ref,
        "err_high_to_ref": err_high_to_ref,
        "err_gpu_to_high": err_gpu_to_high,
        "encrypt_rows_us": t1 - t0,
        "gpu_wsum_core_us": t3 - t2,
        "highlevel_add_core_us": t5 - t4,
        "decrypt_outputs_us": t6 - t5,
        "input_states": in_states,
        "gpu_output_state": gpu_state,
        "highlevel_output_state": high_state,
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4e2_component_linear_wsum_gpu_smoke_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("WP4-E2 GPU-native component linear weighted-sum smoke")
    print("=" * 100)
    print(f"n_in={n_in}, slots={slots}")
    print(f"weights={weights.tolist()}")
    print("-" * 100)
    print(f"encrypt_rows_us={t1 - t0:.3f}")
    print(f"gpu_wsum_core_us={t3 - t2:.3f}")
    print(f"highlevel_add_core_us={t5 - t4:.3f}")
    print(f"decrypt_outputs_us={t6 - t5:.3f}")
    print("-" * 100)
    print(f"err_input={err_input:.6e}")
    print(f"err_gpu_to_ref={err_gpu_to_ref:.6e}")
    print(f"err_high_to_ref={err_high_to_ref:.6e}")
    print(f"err_gpu_to_high={err_gpu_to_high:.6e}")
    print("-" * 100)
    print("input0 state:")
    print(in_states[0])
    print("gpu component wsum output state:")
    print(gpu_state)
    print("high-level repeated-add output state:")
    print(high_state)
    print("=" * 100)
    print("报告已保存")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
