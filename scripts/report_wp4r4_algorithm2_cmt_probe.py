import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    describe_rowwise_coeff_ciphertext_matrix,
    cmt_algorithm2_rowwise,
    describe_cmt_output,
)


def main():
    rng = np.random.default_rng(206400)
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
        "experiment": "wp4r4_algorithm2_cmt_probe",
        "purpose": "Implement and smoke-test real Park-2025 Algorithm 2 C-MT using Tweak + Auto.",
        "status": "algorithm2_cmt_probe_only_no_ppmm_no_ccmm",
        "input_matrix_M_i64": M.tolist(),
        "formula": {
            "algorithm1_tweak": {
                "value": "ct_prime[j] = sum_i X^(2*i*j*N/n) * ct_i",
                "zh": "Algorithm 1 Tweak：第 j 个中间密文由所有输入行密文乘单项式后相加得到",
            },
            "algorithm2_auto": {
                "value": "out[j] = Auto(ct_prime[j]; 2*j + 1)",
                "zh": "Algorithm 2：对 Tweak 输出执行 automorphism/key-switch",
            },
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        bundle = make_rowwise_coeff_ciphertext_matrix(rt, M.tolist())
        report["input_bundle"] = describe_rowwise_coeff_ciphertext_matrix(bundle)

        cmt_out, trace = cmt_algorithm2_rowwise(rt, bundle)

        report["cmt_trace"] = trace
        report["cmt_output"] = describe_cmt_output(rt, cmt_out, coeff_sample=4)

        rows = report["cmt_output"]["rows"]

        expected_tweak_exps = [
            [0, 0, 0, 0],
            [0, 8192, 16384, 24576],
            [0, 16384, 32768, 49152],
            [0, 24576, 49152, 73728],
        ]
        got_tweak_exps = [
            [term["exp"] for term in item["terms"]]
            for item in trace["tweak"]
        ]

        expected_alphas = [1, 3, 5, 7]
        got_alphas = [item["alpha"] for item in trace["auto"]]

        report["checks"] = {
            "cmt_output_has_4_rows（C-MT 输出有 4 个 ciphertext）": len(cmt_out) == 4,
            "cmt_output_consistent_shape（C-MT 输出 component/tower/ring_dim 形状一致）": report["cmt_output"]["consistent_shape"]["value"] is True,
            "cmt_output_has_two_rlwe_parts（每个 C-MT 输出仍有 c0/c1 两个 RLWE component）": all(r["num_parts"]["value"] == 2 for r in rows),
            "cmt_output_uses_coef_encoding（每个 C-MT 输出仍是 coefficient encoding）": all(r["encoding_type"]["value"] == 1 for r in rows),
            "cmt_output_ring_dim_16384（C-MT 输出 ring_dim 仍是 16384）": report["cmt_output"]["expected_ring_dim"]["value"] == 16384,
            "cmt_tweak_exponents_are_expected（C-MT 内部 Tweak 指数符合 2*i*j*N/n）": got_tweak_exps == expected_tweak_exps,
            "cmt_auto_alphas_are_expected（C-MT 内部 Auto alpha 符合 2*j+1）": got_alphas == expected_alphas,
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4r4_algorithm2_cmt_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-R4 Algorithm 2 C-MT probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
