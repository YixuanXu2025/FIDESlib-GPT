import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    describe_rowwise_coeff_ciphertext_matrix,
)


def main():
    rng = np.random.default_rng(206200)
    M = rng.integers(-3, 4, size=(4, 4)).astype(np.int64)

    cfg = HEConfig(
        ring_dim=1 << 14,
        multiplicative_depth=2,
        scaling_mod_size=50,
        first_mod_size=60,
        num_large_digits=2,
        batch_size=8,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    report = {
        "experiment": "wp4q2_real_rowwise_bundle",
        "purpose": "Create a real Park-2025 row-wise coefficient ciphertext matrix bundle.",
        "status": "bundle_api_only_no_cmt_no_ccmm",
        "input_matrix_M_i64": M.tolist(),
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        bundle = make_rowwise_coeff_ciphertext_matrix(rt, M.tolist())
        report["bundle"] = describe_rowwise_coeff_ciphertext_matrix(bundle)

        report["checks"] = {
            "bundle_is_rowwise（bundle 是按行加密的 row-wise 结构）": bundle.orientation == "rowwise",
            "bundle_has_4_rows（bundle 有 4 个 row ciphertext）": len(bundle.rows) == 4,
            "bundle_shape_is_4x4（逻辑矩阵形状是 4x4）": bundle.shape == (4, 4),
            "bundle_uses_coef_encoding（使用 coefficient encoding）": bundle.encoding_type == 1,
            "bundle_has_two_rlwe_parts（每个密文有 c0/c1 两个 RLWE component）": bundle.num_parts == 2,
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4q2_real_rowwise_bundle_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Q2 real row-wise coefficient ciphertext matrix bundle report")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
