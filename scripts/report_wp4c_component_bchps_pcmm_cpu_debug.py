import json
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def now_us():
    return time.perf_counter() * 1e6


def main():
    # Row-wise toy matrix:
    #   M has n_in encrypted rows, each row is a CKKS packed vector of length slots.
    #   U is a small integer plaintext matrix.
    # Component-level operation:
    #   ct_out[r] = sum_i U[r,i] * ct_in[i]
    #
    # This is the BCHPS-style A/B linear-combination skeleton:
    #   c0_out = U @ c0
    #   c1_out = U @ c1
    n_in = 4
    n_out = 3
    slots = 8
    seed = 205400

    rng = np.random.default_rng(seed)

    M = rng.normal(0.0, 0.25, size=(n_in, slots)).astype(np.float64)

    U = np.array([
        [1,  0, -1,  2],
        [0,  1,  1, -1],
        [2, -1,  0,  1],
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
        plaintext_autoload=False,
        ciphertext_autoload=False,
        with_mult_key=True,
    )

    with HERuntime(cfg, rotation_steps=[]) as rt:
        t0 = now_us()
        ct_rows = [rt.encrypt(M[i].tolist()) for i in range(n_in)]
        t1 = now_us()

        dec_in = np.array([
            rt.decrypt(ct_rows[i], logical_length=slots)
            for i in range(n_in)
        ], dtype=np.float64)

        t2 = now_us()
        ct_out = rt.component_int_linear_combination_cpu(
            ct_rows,
            U.astype(np.int64).tolist(),
        )
        t3 = now_us()

        dec_out = np.array([
            rt.decrypt(ct_out[r], logical_length=slots)
            for r in range(n_out)
        ], dtype=np.float64)
        t4 = now_us()

        info_in0 = rt.inspect_rlwe_components_cpu(ct_rows[0], coeff_sample=2)
        info_out0 = rt.inspect_rlwe_components_cpu(ct_out[0], coeff_sample=2)

    err_input = float(np.max(np.abs(dec_in - M)))
    err_output_to_ref = float(np.max(np.abs(dec_out - Y_ref)))
    err_output_to_dec_ref = float(np.max(np.abs(dec_out - (U.astype(np.float64) @ dec_in))))

    report = {
        "experiment": "wp4c_component_bchps_pcmm_cpu_debug",
        "n_in": n_in,
        "n_out": n_out,
        "slots": slots,
        "seed": seed,
        "U": U.tolist(),
        "M": M.tolist(),
        "Y_ref": Y_ref.tolist(),
        "dec_in": dec_in.tolist(),
        "dec_out": dec_out.tolist(),
        "err_input": err_input,
        "err_output_to_ref": err_output_to_ref,
        "err_output_to_dec_ref": err_output_to_dec_ref,
        "encrypt_rows_us": t1 - t0,
        "component_pcmm_core_us": t3 - t2,
        "decrypt_outputs_us": t4 - t3,
        "component_info_input0": info_in0,
        "component_info_output0": info_out0,
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4c_component_bchps_pcmm_cpu_debug_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("WP4-C BCHPS-style component PCMM CPU debug")
    print("=" * 100)
    print(f"n_in={n_in}, n_out={n_out}, slots={slots}")
    print(f"U=")
    print(U)
    print("-" * 100)
    print(f"encrypt_rows_us={t1 - t0:.3f}")
    print(f"component_pcmm_core_us={t3 - t2:.3f}")
    print(f"decrypt_outputs_us={t4 - t3:.3f}")
    print("-" * 100)
    print(f"err_input={err_input:.6e}")
    print(f"err_output_to_ref={err_output_to_ref:.6e}")
    print(f"err_output_to_dec_ref={err_output_to_dec_ref:.6e}")
    print("-" * 100)
    print("input0:")
    print(f"  loaded={info_in0.get('fides_loaded')} gpu={info_in0.get('fides_gpu_handle')} parts={info_in0.get('num_parts')}")
    print(f"  level={info_in0.get('openfhe_level')} noise_deg={info_in0.get('openfhe_noise_scale_deg')} slots={info_in0.get('openfhe_slots')}")
    print("output0:")
    print(f"  loaded={info_out0.get('fides_loaded')} gpu={info_out0.get('fides_gpu_handle')} parts={info_out0.get('num_parts')}")
    print(f"  level={info_out0.get('openfhe_level')} noise_deg={info_out0.get('openfhe_noise_scale_deg')} slots={info_out0.get('openfhe_slots')}")
    print("=" * 100)
    print("报告已保存")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
