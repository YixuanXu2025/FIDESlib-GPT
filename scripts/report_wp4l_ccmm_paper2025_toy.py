import json
from pathlib import Path

import numpy as np

from hegpt.ccmm_paper2025 import (
    ccmm_paper2025_toy,
    make_rowwise_bundle_from_plain,
    negacyclic_inverse_poly,
    negacyclic_toeplitz,
    reconstruct_bundle,
)


def max_abs(x):
    return float(np.max(np.abs(x)))


def main():
    rng = np.random.default_rng(205800)

    # Toy ring degree / matrix dimension.
    # Paper Algorithm 3 first focuses on N x N matrices.
    N = 4

    # Sparse-ish toy secret, only for algebra verification.
    # Real server-side CC-MM must not use sk.
    sk = np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float64)

    U = rng.integers(-2, 3, size=(N, N)).astype(np.float64)
    V = rng.integers(-2, 3, size=(N, N)).astype(np.float64)

    # Non-trivial component A matrices so all Algorithm 3 terms are exercised.
    A_U = rng.integers(-2, 3, size=(N, N)).astype(np.float64)
    A_V = rng.integers(-2, 3, size=(N, N)).astype(np.float64)

    ctU = make_rowwise_bundle_from_plain(U, sk, A=A_U)
    ctV = make_rowwise_bundle_from_plain(V, sk, A=A_V)

    U_rec = reconstruct_bundle(ctU, sk)
    V_rec = reconstruct_bundle(ctV, sk)

    ctW, trace = ccmm_paper2025_toy(ctU, ctV, sk, return_trace=True)

    W_ref = U @ V
    W_rec = reconstruct_bundle(ctW, sk)

    T = negacyclic_toeplitz(sk)
    Tf = negacyclic_toeplitz(negacyclic_inverse_poly(sk))

    report = {
        "experiment": "wp4l_ccmm_paper2025_toy",
        "purpose": (
            "Validate paper-aligned Algorithm 3 algebra for ciphertext-ciphertext "
            "matrix multiplication using toy/oracle C-MT and oracle relinearization."
        ),
        "status": "toy_algebra_only_not_real_he",
        "N": N,
        "sk": sk.tolist(),
        "Toep_sk": T.tolist(),
        "Toep_skf": Tf.tolist(),
        "U": U.tolist(),
        "V": V.tolist(),
        "U_rec": U_rec.tolist(),
        "V_rec": V_rec.tolist(),
        "W_ref_UV": W_ref.tolist(),
        "W_rec_from_toy_ccmm": W_rec.tolist(),
        "errors": {
            "rowwise_U_reconstruct_max_abs_err": max_abs(U_rec - U),
            "rowwise_V_reconstruct_max_abs_err": max_abs(V_rec - V),
            "toy_ccmm_to_UV_max_abs_err": max_abs(W_rec - W_ref),
        },
        "algorithm3_shapes": {
            "M00": list(trace.M00.shape),
            "M01": list(trace.M01.shape),
            "M10": list(trace.M10.shape),
            "M11": list(trace.M11.shape),
            "C2_before_relin": list(trace.C2_before_relin.shape),
            "C1_before_relin": list(trace.C1_before_relin.shape),
            "C0_before_relin": list(trace.C0_before_relin.shape),
            "output_AW": list(ctW.A.shape),
            "output_BW": list(ctW.B.shape),
        },
        "notes": [
            "This implements Algorithm 3 matrix-form algebra only.",
            "C-MT is an oracle toy conversion, not Algorithm 2 yet.",
            "Relinearization and rescale are oracle/identity toy steps.",
            "The old GPU component_linear_transform primitive is not used as CC-MM here.",
        ],
    }

    print("=" * 100)
    print("WP4-L Park-2025 CC-MM toy algebra report")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4l_ccmm_paper2025_toy_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
