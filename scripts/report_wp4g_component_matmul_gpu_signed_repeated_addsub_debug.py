import json
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def now_us():
    return time.perf_counter() * 1e6


def main():
    n_in = 4
    n_out = 4
    slots = 8
    seed = 205405

    rng = np.random.default_rng(seed)
    M = rng.normal(0.0, 0.25, size=(n_in, slots)).astype(np.float64)

    # 含负权重；最后一行专门测试“第一个非零权重为负”的分支。
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

        dec_in = np.array([
            rt.decrypt(ct_rows[i], logical_length=slots)
            for i in range(n_in)
        ], dtype=np.float64)

        t2 = now_us()
        ct_gpu_outputs = [
            rt.component_linear_wsum_gpu(ct_rows, U[r].astype(np.float64).tolist())
            for r in range(n_out)
        ]
        t3 = now_us()

        gpu_states = [rt.ciphertext_storage_state(ct) for ct in ct_gpu_outputs]

        # CPU component debug baseline，之前 WP4-C 已验证过。
        t4 = now_us()
        ct_cpu_outputs = rt.component_int_linear_combination_cpu(
            ct_rows,
            U.astype(np.int64).tolist(),
        )
        t5 = now_us()

        cpu_states = [rt.ciphertext_storage_state(ct) for ct in ct_cpu_outputs]

        dec_gpu = np.array([
            rt.decrypt(ct_gpu_outputs[r], logical_length=slots)
            for r in range(n_out)
        ], dtype=np.float64)

        dec_cpu = np.array([
            rt.decrypt(ct_cpu_outputs[r], logical_length=slots)
            for r in range(n_out)
        ], dtype=np.float64)
        t6 = now_us()

    err_input = float(np.max(np.abs(dec_in - M)))
    err_gpu_to_ref = float(np.max(np.abs(dec_gpu - Y_ref)))
    err_cpu_to_ref = float(np.max(np.abs(dec_cpu - Y_ref)))
    err_gpu_to_cpu = float(np.max(np.abs(dec_gpu - dec_cpu)))

    report = {
        "experiment": "wp4g_component_matmul_gpu_signed_repeated_addsub_debug",
        "n_in": n_in,
        "n_out": n_out,
        "slots": slots,
        "seed": seed,
        "U": U.tolist(),
        "M": M.tolist(),
        "Y_ref": Y_ref.tolist(),
        "dec_in": dec_in.tolist(),
        "dec_gpu": dec_gpu.tolist(),
        "dec_cpu": dec_cpu.tolist(),
        "err_input": err_input,
        "err_gpu_to_ref": err_gpu_to_ref,
        "err_cpu_to_ref": err_cpu_to_ref,
        "err_gpu_to_cpu": err_gpu_to_cpu,
        "encrypt_rows_us": t1 - t0,
        "gpu_signed_component_matmul_core_us": t3 - t2,
        "cpu_component_matmul_core_us": t5 - t4,
        "decrypt_outputs_us": t6 - t5,
        "input_states": input_states,
        "gpu_output_states": gpu_states,
        "cpu_output_states": cpu_states,
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4g_component_matmul_gpu_signed_repeated_addsub_debug_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("WP4-G GPU-native signed component matmul repeated-add/sub debug")
    print("=" * 100)
    print(f"n_in={n_in}, n_out={n_out}, slots={slots}")
    print("U=")
    print(U)
    print("-" * 100)
    print(f"encrypt_rows_us={t1 - t0:.3f}")
    print(f"gpu_signed_component_matmul_core_us={t3 - t2:.3f}")
    print(f"cpu_component_matmul_core_us={t5 - t4:.3f}")
    print(f"decrypt_outputs_us={t6 - t5:.3f}")
    print("-" * 100)
    print(f"err_input={err_input:.6e}")
    print(f"err_gpu_to_ref={err_gpu_to_ref:.6e}")
    print(f"err_cpu_to_ref={err_cpu_to_ref:.6e}")
    print(f"err_gpu_to_cpu={err_gpu_to_cpu:.6e}")
    print("-" * 100)
    print("input0 state:")
    print(input_states[0])
    print("gpu output0 state:")
    print(gpu_states[0])
    print("cpu output0 state:")
    print(cpu_states[0])
    print("=" * 100)
    print("报告已保存")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
