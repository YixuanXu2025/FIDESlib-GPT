import json
import statistics
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def now_us():
    return time.perf_counter() * 1e6


def make_nonzero_int_matrix(rng, n_out, n_in, low=-2, high=3):
    U = rng.integers(low, high, size=(n_out, n_in), dtype=np.int64)
    for r in range(n_out):
        if not np.any(U[r]):
            U[r, r % n_in] = 1
    return U


def decrypt_many(rt, cts, slots):
    return np.array([rt.decrypt(ct, logical_length=slots) for ct in cts], dtype=np.float64)


def median_call_us(fn, reps=3):
    vals = []
    last = None
    for _ in range(reps):
        t0 = now_us()
        last = fn()
        t1 = now_us()
        vals.append(t1 - t0)
    return float(statistics.median(vals)), vals, last


def run_case(n_in, n_out, slots, seed, measure_baselines):
    rng = np.random.default_rng(seed)
    M = rng.normal(0.0, 0.25, size=(n_in, slots)).astype(np.float64)
    U = make_nonzero_int_matrix(rng, n_out, n_in)
    Y_ref = U.astype(np.float64) @ M

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

    result = {
        "n_in": n_in,
        "n_out": n_out,
        "slots": slots,
        "seed": seed,
        "U": U.tolist(),
        "timing_us": {},
        "errors": {},
        "states": {},
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        t0 = now_us()
        ct_rows = [rt.encrypt(M[i].tolist()) for i in range(n_in)]
        t1 = now_us()
        result["timing_us"]["encrypt_rows"] = t1 - t0

        input_state = rt.ciphertext_storage_state(ct_rows[0])
        result["states"]["input0"] = input_state

        # Pure GPU matrix fused raw. No CPU copyback. This is the key performance number.
        med_nocpu, vals_nocpu, ct_matrix_nocpu = median_call_us(
            lambda: rt.component_linear_matmul_gpu_fused_raw(
                ct_rows,
                U.astype(np.float64).tolist(),
                copyback=False,
            ),
            reps=3,
        )
        result["timing_us"]["matrix_fused_raw_gpu_only_copyback_false_median"] = med_nocpu
        result["timing_us"]["matrix_fused_raw_gpu_only_copyback_false_samples"] = vals_nocpu
        result["states"]["matrix_nocpu_output0"] = rt.ciphertext_storage_state(ct_matrix_nocpu[0])

        # Matrix fused raw with CPU copyback for correctness.
        t2 = now_us()
        ct_matrix = rt.component_linear_matmul_gpu_fused_raw(
            ct_rows,
            U.astype(np.float64).tolist(),
            copyback=True,
        )
        t3 = now_us()
        result["timing_us"]["matrix_fused_raw_with_copyback"] = t3 - t2
        result["states"]["matrix_output0"] = rt.ciphertext_storage_state(ct_matrix[0])

        t4 = now_us()
        dec_matrix = decrypt_many(rt, ct_matrix, slots)
        t5 = now_us()
        result["timing_us"]["decrypt_matrix_outputs"] = t5 - t4
        result["errors"]["matrix_to_ref"] = float(np.max(np.abs(dec_matrix - Y_ref)))

        if measure_baselines:
            # Per-row fused raw baseline.
            t6 = now_us()
            ct_perrow = [
                rt.component_linear_wsum_gpu_fused_raw(
                    ct_rows,
                    U[r].astype(np.float64).tolist(),
                )
                for r in range(n_out)
            ]
            t7 = now_us()
            result["timing_us"]["perrow_fused_raw_with_copyback"] = t7 - t6
            result["states"]["perrow_output0"] = rt.ciphertext_storage_state(ct_perrow[0])

            # Repeated add/sub baseline.
            t8 = now_us()
            ct_repeated = [
                rt.component_linear_wsum_gpu(
                    ct_rows,
                    U[r].astype(np.float64).tolist(),
                )
                for r in range(n_out)
            ]
            t9 = now_us()
            result["timing_us"]["repeated_addsub_with_copyback"] = t9 - t8
            result["states"]["repeated_output0"] = rt.ciphertext_storage_state(ct_repeated[0])

            # CPU component baseline.
            t10 = now_us()
            ct_cpu = rt.component_int_linear_combination_cpu(
                ct_rows,
                U.astype(np.int64).tolist(),
            )
            t11 = now_us()
            result["timing_us"]["cpu_component"] = t11 - t10
            result["states"]["cpu_output0"] = rt.ciphertext_storage_state(ct_cpu[0])

            dec_perrow = decrypt_many(rt, ct_perrow, slots)
            dec_repeated = decrypt_many(rt, ct_repeated, slots)
            dec_cpu = decrypt_many(rt, ct_cpu, slots)

            result["errors"]["perrow_to_ref"] = float(np.max(np.abs(dec_perrow - Y_ref)))
            result["errors"]["repeated_to_ref"] = float(np.max(np.abs(dec_repeated - Y_ref)))
            result["errors"]["cpu_to_ref"] = float(np.max(np.abs(dec_cpu - Y_ref)))
            result["errors"]["matrix_to_perrow"] = float(np.max(np.abs(dec_matrix - dec_perrow)))
            result["errors"]["matrix_to_repeated"] = float(np.max(np.abs(dec_matrix - dec_repeated)))
            result["errors"]["matrix_to_cpu"] = float(np.max(np.abs(dec_matrix - dec_cpu)))

    return result


def fmt(x):
    if x is None:
        return "NA"
    return f"{x:.3f}"


def main():
    slots = 8

    cases = [
        (4, 4, True),
        (8, 8, True),
        (16, 16, False),
        (32, 32, False),
    ]

    all_results = []

    print("=" * 120)
    print("WP4-H3 matrix-level fused raw scaling sweep")
    print("=" * 120)
    print(
        f"{'n_in':>6} {'n_out':>6} "
        f"{'gpu_only_med(us)':>18} "
        f"{'copyback(us)':>14} "
        f"{'decrypt(us)':>12} "
        f"{'perrow(us)':>12} "
        f"{'repeat(us)':>12} "
        f"{'cpu(us)':>12} "
        f"{'err_matrix':>14}"
    )
    print("-" * 120)

    for idx, (n_in, n_out, measure_baselines) in enumerate(cases):
        result = run_case(
            n_in=n_in,
            n_out=n_out,
            slots=slots,
            seed=205500 + idx,
            measure_baselines=measure_baselines,
        )
        all_results.append(result)

        timing = result["timing_us"]
        errors = result["errors"]

        print(
            f"{n_in:6d} {n_out:6d} "
            f"{fmt(timing.get('matrix_fused_raw_gpu_only_copyback_false_median')):>18} "
            f"{fmt(timing.get('matrix_fused_raw_with_copyback')):>14} "
            f"{fmt(timing.get('decrypt_matrix_outputs')):>12} "
            f"{fmt(timing.get('perrow_fused_raw_with_copyback')):>12} "
            f"{fmt(timing.get('repeated_addsub_with_copyback')):>12} "
            f"{fmt(timing.get('cpu_component')):>12} "
            f"{errors.get('matrix_to_ref', float('nan')):14.6e}"
        )

    report = {
        "experiment": "wp4h3_matrix_api_scaling_sweep",
        "slots": slots,
        "cases": all_results,
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4h3_matrix_api_scaling_sweep_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 120)
    print("报告已保存")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
