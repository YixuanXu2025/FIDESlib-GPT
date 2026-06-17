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


def state_many(rt, cts, limit=2):
    out = []
    for ct in cts[:limit]:
        try:
            out.append(dict(rt.ciphertext_storage_state(ct)))
        except Exception as e:
            out.append({"state_error": repr(e)})
    return out


def main():
    n = 64
    layers = 4
    slots = 64
    seed = 205700

    rng = np.random.default_rng(seed)
    M = rng.normal(0.0, 0.01, size=(n, slots)).astype(np.float64)
    Us = [make_u(rng, n) for _ in range(layers)]

    Y_ref = M.copy()
    for U in Us:
        Y_ref = U.astype(np.float64) @ Y_ref

    cfg = HEConfig(
        ring_dim=1 << 15,
        multiplicative_depth=4,
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
        "experiment": "wp4j_PCMM_large",
        "purpose": "Validate formal GPU-only component-linear-transform / materialize / decrypt API layering.",
        "matrix_shape": f"{n}x{n} @ {n}x{slots}",
        "n": n,
        "layers": layers,
        "slots": slots,
        "seed": seed,
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        print("[phase] encrypt rows", flush=True)
        t0 = now_us()
        rows = [rt.encrypt(M[i].tolist()) for i in range(n)]
        t1 = now_us()

        print("[phase] GPU-only component linear transform chain via rt.component_linear_transform_gpu", flush=True)
        compute_times = []
        compute_states = []

        for ell, U in enumerate(Us):
            print(f"[component_linear_transform_gpu] layer={ell}", flush=True)
            t_layer0 = now_us()
            rows = rt.component_linear_transform_gpu(rows, U.astype(np.float64).tolist())
            t_layer1 = now_us()
            compute_times.append(t_layer1 - t_layer0)
            compute_states.append(state_many(rt, rows, limit=2))

        print("[phase] explicit materialize via rt.materialize_gpu_ciphertexts", flush=True)
        t2 = now_us()
        rows_cpu = rt.materialize_gpu_ciphertexts(rows)
        t3 = now_us()

        materialized_states = state_many(rt, rows_cpu, limit=2)

        print("[phase] decrypt materialized rows", flush=True)
        t4 = now_us()
        dec = decrypt_many(rt, rows_cpu, slots)
        t5 = now_us()

    err = float(np.max(np.abs(dec - Y_ref)))

    report.update({
        "timing_us": {
            "encrypt_rows": t1 - t0,
            "gpu_only_compute_layer_times": compute_times,
            "explicit_materialize_all_rows": t3 - t2,
            "decrypt_materialized_rows": t5 - t4,
            "gpu_only_compute_total": float(sum(compute_times)),
        },
        "states": {
            "gpu_only_compute_states_first2_each_layer": compute_states,
            "materialized_states_first2": materialized_states,
        },
        "errors": {
            "component_linear_transform_to_ref_max_abs_err": err,
        },
        "Y_ref_first_row": Y_ref[0].tolist(),
        "dec_first_row": dec[0].tolist(),
    })

    print("=" * 100)
    print("WP4-J PCMM large report")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4j_PCMM_large.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
