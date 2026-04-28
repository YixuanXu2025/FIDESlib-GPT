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


def highlevel_nonnegative_int_linear_combo(rt, ct_rows, U):
    out = []
    for r in range(U.shape[0]):
        acc = None
        for i in range(U.shape[1]):
            k = int(U[r, i])
            if k == 0:
                continue
            term = add_many_repeated(rt, ct_rows[i], k)
            if acc is None:
                acc = term
            else:
                acc = rt.add_ct(acc, term)
        if acc is None:
            raise RuntimeError("This simple debug script does not support all-zero output rows.")
        out.append(acc)
    return out


def main():
    n_in = 4
    n_out = 3
    slots = 8
    seed = 205401

    rng = np.random.default_rng(seed)
    M = rng.normal(0.0, 0.25, size=(n_in, slots)).astype(np.float64)

    # 非负整数矩阵，方便 high-level 路线只用 add_ct，不引入 mult_plain 的 scale 变化。
    U = np.array([
        [1, 0, 1, 2],
        [0, 1, 1, 1],
        [2, 1, 0, 1],
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
        ct_component = rt.component_int_linear_combination_cpu(
            ct_rows,
            U.astype(np.int64).tolist(),
        )
        t3 = now_us()

        t4 = now_us()
        ct_highlevel = highlevel_nonnegative_int_linear_combo(rt, ct_rows, U)
        t5 = now_us()

        dec_component = np.array([
            rt.decrypt(ct_component[r], logical_length=slots)
            for r in range(n_out)
        ], dtype=np.float64)

        dec_highlevel = np.array([
            rt.decrypt(ct_highlevel[r], logical_length=slots)
            for r in range(n_out)
        ], dtype=np.float64)
        t6 = now_us()

        info_component0 = rt.inspect_rlwe_components_cpu(ct_component[0], coeff_sample=2)
        info_highlevel0 = rt.inspect_rlwe_components_cpu(ct_highlevel[0], coeff_sample=2)

    err_input = float(np.max(np.abs(dec_in - M)))
    err_component_to_ref = float(np.max(np.abs(dec_component - Y_ref)))
    err_highlevel_to_ref = float(np.max(np.abs(dec_highlevel - Y_ref)))
    err_component_to_highlevel = float(np.max(np.abs(dec_component - dec_highlevel)))

    report = {
        "experiment": "wp4d_component_vs_highlevel_add_pcmm_cpu_debug",
        "n_in": n_in,
        "n_out": n_out,
        "slots": slots,
        "seed": seed,
        "U": U.tolist(),
        "M": M.tolist(),
        "Y_ref": Y_ref.tolist(),
        "dec_in": dec_in.tolist(),
        "dec_component": dec_component.tolist(),
        "dec_highlevel": dec_highlevel.tolist(),
        "err_input": err_input,
        "err_component_to_ref": err_component_to_ref,
        "err_highlevel_to_ref": err_highlevel_to_ref,
        "err_component_to_highlevel": err_component_to_highlevel,
        "encrypt_rows_us": t1 - t0,
        "component_pcmm_core_us": t3 - t2,
        "highlevel_add_core_us": t5 - t4,
        "decrypt_outputs_us": t6 - t5,
        "component_info_output0": info_component0,
        "highlevel_info_output0": info_highlevel0,
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4d_component_vs_highlevel_add_pcmm_cpu_debug_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("WP4-D component PCMM vs high-level HE add PCMM CPU debug")
    print("=" * 100)
    print(f"n_in={n_in}, n_out={n_out}, slots={slots}")
    print("U=")
    print(U)
    print("-" * 100)
    print(f"encrypt_rows_us={t1 - t0:.3f}")
    print(f"component_pcmm_core_us={t3 - t2:.3f}")
    print(f"highlevel_add_core_us={t5 - t4:.3f}")
    print(f"decrypt_outputs_us={t6 - t5:.3f}")
    print("-" * 100)
    print(f"err_input={err_input:.6e}")
    print(f"err_component_to_ref={err_component_to_ref:.6e}")
    print(f"err_highlevel_to_ref={err_highlevel_to_ref:.6e}")
    print(f"err_component_to_highlevel={err_component_to_highlevel:.6e}")
    print("-" * 100)
    print("component output0:")
    print(
        f"  loaded={info_component0.get('fides_loaded')} "
        f"gpu={info_component0.get('fides_gpu_handle')} "
        f"parts={info_component0.get('num_parts')} "
        f"level={info_component0.get('openfhe_level')} "
        f"noise_deg={info_component0.get('openfhe_noise_scale_deg')} "
        f"slots={info_component0.get('openfhe_slots')}"
    )
    print("high-level output0:")
    print(
        f"  loaded={info_highlevel0.get('fides_loaded')} "
        f"gpu={info_highlevel0.get('fides_gpu_handle')} "
        f"parts={info_highlevel0.get('num_parts')} "
        f"level={info_highlevel0.get('openfhe_level')} "
        f"noise_deg={info_highlevel0.get('openfhe_noise_scale_deg')} "
        f"slots={info_highlevel0.get('openfhe_slots')}"
    )
    print("=" * 100)
    print("报告已保存")
    print(f"JSON路径: {out_path}")


if __name__ == "__main__":
    main()
