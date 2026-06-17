import json
from pathlib import Path

import numpy as np

from hegpt.ccmm_paper2025 import (
    ccmm_paper2025_component_toy,
    cmt_algorithm2_component_toy,
    make_rowwise_bundle_from_plain,
    reconstruct_bundle,
)


def max_abs(x):
    return float(np.max(np.abs(x)))


def main():
    rng = np.random.default_rng(206000)

    N = 4
    sk = np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float64)

    U = rng.integers(-2, 3, size=(N, N)).astype(np.float64)
    V = rng.integers(-2, 3, size=(N, N)).astype(np.float64)

    A_U = rng.integers(-2, 3, size=(N, N)).astype(np.float64)
    A_V = rng.integers(-2, 3, size=(N, N)).astype(np.float64)

    ctU = make_rowwise_bundle_from_plain(U, sk, A=A_U)
    ctV = make_rowwise_bundle_from_plain(V, sk, A=A_V)

    U_rec = reconstruct_bundle(ctU, sk)
    V_rec = reconstruct_bundle(ctV, sk)

    # Directly validate component-level Algorithm 2 C-MT on ctU.
    ctU_col = cmt_algorithm2_component_toy(ctU, sk)
    U_after_cmt = reconstruct_bundle(ctU_col, sk)

    # Full Algorithm 3 with component-level Algorithm 2 C-MT.
    ctW, trace = ccmm_paper2025_component_toy(ctU, ctV, sk, return_trace=True)
    W_rec = reconstruct_bundle(ctW, sk)
    W_ref = U @ V

    report = {
        "experiment": "wp4n_ccmm_component_cmt2_toy",
        "purpose": (
            "Validate Park-2025 Algorithm 3 using component-level toy Algorithm 2 C-MT "
            "with exact toy Auto/key-switch and exact toy relin."
        ),
        "status": "component_level_toy_not_real_he",
        "N": N,
        "sk": sk.tolist(),
        "U": U.tolist(),
        "V": V.tolist(),
        "W_ref_UV": W_ref.tolist(),
        "W_rec": W_rec.tolist(),
        "errors": {
            "U_rowwise_reconstruct_max_abs_err": max_abs(U_rec - U),
            "V_rowwise_reconstruct_max_abs_err": max_abs(V_rec - V),
            "CMT_ctU_preserves_plaintext_max_abs_err": max_abs(U_after_cmt - U),
            "CCMM_component_toy_to_UV_max_abs_err": max_abs(W_rec - W_ref),
        },
        "orientations": {
            "ctU": ctU.orientation,
            "ctU_after_cmt": ctU_col.orientation,
            "ctW": ctW.orientation,
        },
        "algorithm3_trace": trace,
        "notes": [
            "This is no longer the reconstruct-based C-MT oracle from WP4-L.",
            "C-MT uses Algorithm 2 structure on toy RLWE components.",
            "Auto uses exact toy switching keys from sigma(sk) to sk.",
            "Relinearization uses exact toy switching key from sk^2 to sk.",
            "Still not real FIDESlib/OpenFHE HE: no RNS, no gadget decomposition, no rescale modulus management.",
        ],
    }

    print("=" * 100)
    print("WP4-N Park-2025 CC-MM component C-MT Algorithm 2 toy report")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4n_ccmm_component_cmt2_toy_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print(f"saved: {out_path}")

    if report["errors"]["CCMM_component_toy_to_UV_max_abs_err"] != 0.0:
        raise SystemExit("WP4-N failed: component toy CC-MM mismatch")


if __name__ == "__main__":
    main()
