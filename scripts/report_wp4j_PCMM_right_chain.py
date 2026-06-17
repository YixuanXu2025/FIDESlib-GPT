import json
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def now_us():
    return time.perf_counter() * 1e6


def make_u(rng, dim):
    U = rng.integers(-1, 2, size=(dim, dim), dtype=np.int64)
    for r in range(dim):
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
    # ------------------------------------------------------------
    # Case A: 64x64 @ 64x64 chain
    # ------------------------------------------------------------
    # rows_count = 64
    # dim = 64

    # ------------------------------------------------------------
    # Case B: 64x768 @ 768x768 chain
    # ------------------------------------------------------------
    rows_count = 64
    dim = 768

    layers = 2
    slots = rows_count
    seed = 205701

    U_names = ["U3", "U2", "U1", "U0"]

    rng = np.random.default_rng(seed)

    M = rng.normal(0.0, 0.01, size=(rows_count, dim)).astype(np.float64)
    Us = [make_u(rng, dim) for _ in range(layers)]

    # 明文参考：M @ U3 @ U2 @ U1 @ U0
    Y_ref = M.copy()
    for U in Us:
        Y_ref = Y_ref @ U.astype(np.float64)

    cfg = HEConfig(
        # 4层时 1<<14 可能不满足 OpenFHE 安全检查；
        # 你前面已经看到 1<<15 可以过。
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
        "experiment": "wp4j_PCMM_right_chain",
        "purpose": "Validate M @ U3 @ U2 @ U1 @ U0 using column-encrypted component-linear-transform GPU backend.",
        "matrix_shape": f"{rows_count}x{dim} @ {dim}x{dim} repeated {layers} layers",
        "layout": "column_encrypted",
        "rows_count": rows_count,
        "dim": dim,
        "layers": layers,
        "slots": slots,
        "seed": seed,
        "U_order": U_names,
        "semantics": "M @ U3 @ U2 @ U1 @ U0",
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        print("[phase] encrypt columns", flush=True)
        t0 = now_us()

        # 按列加密：每个 ciphertext 是 M 的一列，长度 rows_count=64
        cols = [rt.encrypt(M[:, j].tolist()) for j in range(dim)]

        t1 = now_us()

        print("[phase] GPU-only right-multiply chain via rt.component_linear_transform_gpu(cols, U.T)", flush=True)
        compute_times = []
        compute_states = []

        for ell, (name, U) in enumerate(zip(U_names, Us)):
            print(f"[component_linear_transform_gpu] layer={ell} name={name}", flush=True)
            t_layer0 = now_us()

            # 右乘 M @ U 在列加密布局下等价于：
            # cols_new = U.T @ cols
            cols = rt.component_linear_transform_gpu(
                cols,
                U.T.astype(np.float64).tolist(),
            )

            t_layer1 = now_us()
            compute_times.append(t_layer1 - t_layer0)
            compute_states.append(state_many(rt, cols, limit=2))

        print("[phase] explicit materialize via rt.materialize_gpu_ciphertexts", flush=True)
        t2 = now_us()
        cols_cpu = rt.materialize_gpu_ciphertexts(cols)
        t3 = now_us()

        materialized_states = state_many(rt, cols_cpu, limit=2)

        print("[phase] decrypt materialized columns", flush=True)
        t4 = now_us()

        # dec_cols shape = (dim, rows_count)
        dec_cols = decrypt_many(rt, cols_cpu, slots)

        # 转回普通矩阵 shape = (rows_count, dim)
        dec = dec_cols.T

        t5 = now_us()

    err = float(np.max(np.abs(dec - Y_ref)))

    report.update({
        "timing_us": {
            "encrypt_columns": t1 - t0,
            "gpu_only_compute_layer_times": compute_times,
            "explicit_materialize_all_columns": t3 - t2,
            "decrypt_materialized_columns": t5 - t4,
            "gpu_only_compute_total": float(sum(compute_times)),
        },
        "states": {
            "gpu_only_compute_states_first2_each_layer": compute_states,
            "materialized_states_first2": materialized_states,
        },
        "errors": {
            "component_linear_transform_to_ref_max_abs_err": err,
        },
        "Y_ref_first_row_first16": Y_ref[0, :16].tolist(),
        "dec_first_row_first16": dec[0, :16].tolist(),
        "Y_ref_shape": list(Y_ref.shape),
        "dec_shape": list(dec.shape),
    })

    print("=" * 100)
    print("WP4-J PCMM right-chain report")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "wp4j_PCMM_right_chain.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
