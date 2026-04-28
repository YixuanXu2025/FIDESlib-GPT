import json
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def now_us():
    return time.perf_counter() * 1e6


def decrypt_many(rt, cts, slots):
    return np.array([rt.decrypt(ct, logical_length=slots) for ct in cts], dtype=np.float64)


def main():
    n_in = 4
    n_out = 4
    slots = 8
    seed = 205407

    rng = np.random.default_rng(seed)
    M = rng.normal(0.0, 0.25, size=(n_in, slots)).astype(np.float64)

    U = np.array([
        [ 1,  0, -1,  2],
        [ 0,  1,  1, -1],
        [ 2, -1,  0,  1],
        [-1,  2,  0,  1],
    ], dtype=np.int64)

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

    with HERuntime(cfg, rotation_steps=[]) as rt:
        t0 = now_us()
        ct_rows = [rt.encrypt(M[i].tolist()) for i in range(n_in)]
        t1 = now_us()

        input_states = [rt.ciphertext_storage_state(ct) for ct in ct_rows]

        dec_in = decrypt_many(rt, ct_rows, slots)

        # H2 matrix API, GPU-only: measures fused kernels + one synchronize + handle registration.
        t2 = now_us()
        ct_matrix_nocpu = rt.component_linear_matmul_gpu_fused_raw(
            ct_rows,
            U.astype(np.float64).tolist(),
            copyback=False,
        )
        t3 = now_us()
        matrix_nocpu_states = [rt.ciphertext_storage_state(ct) for ct in ct_matrix_nocpu]

        # H2 matrix API with CPU copyback: correctness path.
        t4 = now_us()
        ct_matrix = rt.component_linear_matmul_gpu_fused_raw(
            ct_rows,
            U.astype(np.float64).tolist(),
            copyback=True,
        )
        t5 = now_us()
        matrix_states = [rt.ciphertext_storage_state(ct) for ct in ct_matrix]

        # H1 old path: Python loop, one fused_raw call per output row.
        t6 = now_us()
        ct_perrow = [
            rt.component_linear_wsum_gpu_fused_raw(ct_rows, U[r].astype(np.float64).tolist())
            for r in range(n_out)
        ]
        t7 = now_us()
        perrow_states = [rt.ciphertext_storage_state(ct) for ct in ct_perrow]

        # Repeated add/sub path.
        t8 = now_us()
        ct_repeated = [
            rt.component_linear_wsum_gpu(ct_rows, U[r].astype(np.float64).tolist())
            for r in range(n_out)
        ]
        t9 = now_us()
        repeated_states = [rt.ciphertext_storage_state(ct) for ct in ct_repeated]

        # CPU component baseline.
        t10 = now_us()
        ct_cpu = rt.component_int_linear_combination_cpu(
            ct_rows,
            U.astype(np.int64).tolist(),
        )
        t11 = now_us()
        cpu_states = [rt.ciphertext_storage_state(ct) for ct in ct_cpu]

        t12 = now_us()
        dec_matrix = decrypt_many(rt, ct_matrix, slots)
        dec_perrow = decrypt_many(rt, ct_perrow, slots)
        dec_repeated = decrypt_many(rt, ct_repeated, slots)
        dec_cpu = decrypt_many(rt, ct_cpu, slots)
        t13 = now_us()

    report = {
        "experiment": "wp4h2_matrix_api_perf_breakdown",
        "n_in": n_in,
        "n_out": n_out,
        "slots": slots,
        "seed": seed,
        "U": U.tolist(),
        "M": M.tolist(),
        "Y_ref": Y_ref.tolist(),
        "timing_us": {
            "encrypt_rows": t1 - t0,
            "matrix_fused_raw_gpu_only_copyback_false": t3 - t2,
            "matrix_fused_raw_with_copyback": t5 - t4,
            "perrow_fused_raw_with_copyback": t7 - t6,
            "repeated_addsub_with_copyback": t9 - t8,
            "cpu_component": t11 - t10,
            "decrypt_all_outputs": t13 - t12,
        },
        "errors": {
            "input": float(np.max(np.abs(dec_in - M))),
            "matrix_to_ref": float(np.max(np.abs(dec_matrix - Y_ref))),
            "perrow_to_ref": float(np.max(np.abs(dec_perrow - Y_ref))),
            "repeated_to_ref": float(np.max(np.abs(dec_repeated - Y_ref))),
            "cpu_to_ref": float(np.max(np.abs(dec_cpu - Y_ref))),
            "matrix_to_perrow": float(np.max(np.abs(dec_matrix - dec_perrow))),
            "matrix_to_repeated": float(np.max(np.abs(dec_matrix - dec_repeated))),
            "matrix_to_cpu": float(np.max(np.abs(dec_matrix - dec_cpu))),
        },
        "states": {
            "input0": input_states[0],
            "matrix_nocpu_output0": matrix_nocpu_states[0],
            "matrix_output0": matrix_states[0],
            "perrow_output0": perrow_states[0],
            "repeated_output0": repeated_states[0],
            "cpu_output0": cpu_states[0],
        },
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4h2_matrix_api_perf_breakdown_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("WP4-H2 matrix-level fused raw API performance breakdown")
    print("=" * 100)
    print(f"n_in={n_in}, n_out={n_out}, slots={slots}")
    print("U=")
    print(U)
    print("-" * 100)
    for k, v in report["timing_us"].items():
        print(f"{k}: {v:.3f} us")
    print("-" * 100)
    for k, v in report["errors"].items():
        print(f"{k}: {v:.6e}")
    print("-" * 100)
    print("input0 state:")
    print(input_states[0])
    print("matrix nocpu output0 state:")
    print(matrix_nocpu_states[0])
    print("matrix copyback output0 state:")
    print(matrix_states[0])
    print("perrow fused output0 state:")
    print(perrow_states[0])
    print("repeated output0 state:")
    print(repeated_states[0])
    print("cpu output0 state:")
    print(cpu_states[0])
    print("=" * 100)
    print("报告已保存")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
