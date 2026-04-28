import json
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def now_us():
    return time.perf_counter() * 1e6


def make_u(rng, n):
    U = rng.integers(-1, 2, size=(n, n), dtype=np.int64)
    for r in range(n):
        if not np.any(U[r]):
            U[r, r] = 1
    return U


def decrypt_many(rt, cts, slots):
    return np.array(
        [rt.decrypt(ct, logical_length=slots) for ct in cts],
        dtype=np.float64,
    )


def safe_state(rt, ct):
    try:
        return rt.ciphertext_storage_state(ct)
    except Exception as e:
        return {"state_error": repr(e)}


def main():
    n = 8
    layers = 4
    slots = 8
    seed = 205600

    rng = np.random.default_rng(seed)
    M = rng.normal(0.0, 0.01, size=(n, slots)).astype(np.float64)

    Us = [make_u(rng, n) for _ in range(layers)]

    Y_ref = M.copy()
    for U in Us:
        Y_ref = U.astype(np.float64) @ Y_ref

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

    report = {
        "experiment": "wp4i_gpu_only_ccmm_chain_reordered",
        "purpose": "Run copyback=True baseline before GPU-only chain to isolate bad_any_cast source.",
        "n": n,
        "layers": layers,
        "slots": slots,
        "seed": seed,
        "Us": [U.tolist() for U in Us],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        print("[phase] encrypt rows for copyback baseline", flush=True)
        t0 = now_us()
        ct_cb = [rt.encrypt(M[i].tolist()) for i in range(n)]
        t1 = now_us()

        print("[phase] copyback=True every layer baseline", flush=True)
        copyback_every_layer_times = []
        copyback_every_layer_states = []

        for ell in range(layers):
            print(f"[copyback baseline] layer={ell} copyback=True", flush=True)
            t_layer0 = now_us()
            ct_cb = rt.component_linear_matmul_gpu_fused_raw(
                ct_cb,
                Us[ell].astype(np.float64).tolist(),
                copyback=True,
            )
            t_layer1 = now_us()

            copyback_every_layer_times.append(t_layer1 - t_layer0)
            copyback_every_layer_states.append(safe_state(rt, ct_cb[0]))

        print("[phase] decrypt copyback baseline", flush=True)
        t2 = now_us()
        dec_cb_final = decrypt_many(rt, ct_cb, slots)
        t3 = now_us()

        print("[phase] encrypt rows for gpu-only chain", flush=True)
        t4 = now_us()
        ct_gpu = [rt.encrypt(M[i].tolist()) for i in range(n)]
        t5 = now_us()

        print("[phase] gpu-only chain: intermediate copyback=False, final copyback=True", flush=True)
        gpu_only_layer_times = []
        gpu_only_layer_states = []

        for ell in range(layers):
            copyback = (ell == layers - 1)
            print(f"[gpu-only chain] layer={ell} copyback={copyback}", flush=True)
            t_layer0 = now_us()
            ct_gpu = rt.component_linear_matmul_gpu_fused_raw(
                ct_gpu,
                Us[ell].astype(np.float64).tolist(),
                copyback=copyback,
            )
            t_layer1 = now_us()

            gpu_only_layer_times.append(t_layer1 - t_layer0)
            gpu_only_layer_states.append(safe_state(rt, ct_gpu[0]))

        print("[phase] decrypt gpu-only chain final", flush=True)
        t6 = now_us()
        dec_gpu_final = decrypt_many(rt, ct_gpu, slots)
        t7 = now_us()

    err_copyback_chain_to_ref = float(np.max(np.abs(dec_cb_final - Y_ref)))
    err_gpu_chain_to_ref = float(np.max(np.abs(dec_gpu_final - Y_ref)))
    err_gpu_chain_to_copyback_chain = float(np.max(np.abs(dec_gpu_final - dec_cb_final)))

    report.update({
        "timing_us": {
            "encrypt_rows_copyback_baseline": t1 - t0,
            "copyback_every_layer_times": copyback_every_layer_times,
            "decrypt_copyback_baseline": t3 - t2,
            "encrypt_rows_gpu_chain": t5 - t4,
            "gpu_only_layer_times": gpu_only_layer_times,
            "decrypt_gpu_chain_final": t7 - t6,
        },
        "states": {
            "copyback_every_layer_states": copyback_every_layer_states,
            "gpu_only_layer_states": gpu_only_layer_states,
        },
        "errors": {
            "copyback_chain_to_ref_max_abs_err": err_copyback_chain_to_ref,
            "gpu_chain_to_ref_max_abs_err": err_gpu_chain_to_ref,
            "gpu_chain_to_copyback_chain_max_abs_err": err_gpu_chain_to_copyback_chain,
        },
        "Y_ref_first_row": Y_ref[0].tolist(),
        "dec_copyback_first_row": dec_cb_final[0].tolist(),
        "dec_gpu_chain_first_row": dec_gpu_final[0].tolist(),
    })

    print("=" * 100)
    print("WP4-I reordered report")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4i_gpu_only_ccmm_chain_reordered_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
