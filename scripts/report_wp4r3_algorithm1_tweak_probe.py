import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    describe_rowwise_coeff_ciphertext_matrix,
    tweak_algorithm1_rowwise,
    describe_tweak_output,
)


def main():
    rng = np.random.default_rng(206300)
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
        "experiment": "wp4r3_algorithm1_tweak_probe",
        "purpose": "Implement and smoke-test real Park-2025 Algorithm 1 Tweak using coefficient ciphertext primitives.",
        "status": "algorithm1_tweak_probe_only_no_cmt_no_ccmm",
        "input_matrix_M_i64": M.tolist(),
        "formula": {
            "value": "ct_prime[j] = sum_i X^(2*i*j*N/n) * ct_i",
            "zh": "Algorithm 1 Tweak：第 j 个输出由所有输入行密文乘单项式后相加得到",
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        bundle = make_rowwise_coeff_ciphertext_matrix(rt, M.tolist())
        report["input_bundle"] = describe_rowwise_coeff_ciphertext_matrix(bundle)

        tweaked, trace = tweak_algorithm1_rowwise(rt, bundle)
        report["tweak_trace"] = trace

        report["tweak_output"] = describe_tweak_output(rt, tweaked, coeff_sample=4)

        rows = report["tweak_output"]["rows"]

        report["checks"] = {
            "tweak_output_has_4_rows（Tweak 输出有 4 个 ciphertext）": len(tweaked) == 4,
            "tweak_output_consistent_shape（Tweak 输出 component/tower/ring_dim 形状一致）": report["tweak_output"]["consistent_shape"]["value"] is True,
            "tweak_output_has_two_rlwe_parts（每个 Tweak 输出仍有 c0/c1 两个 RLWE component）": all(r["num_parts"]["value"] == 2 for r in rows),
            "tweak_output_uses_coef_encoding（每个 Tweak 输出仍是 coefficient encoding）": all(r["encoding_type"]["value"] == 1 for r in rows),
            "tweak_output_ring_dim_16384（Tweak 输出 ring_dim 仍是 16384）": report["tweak_output"]["expected_ring_dim"]["value"] == 16384,
            "tweak_trace_exponents_are_expected（Tweak 指数符合 2*i*j*N/n 公式）": [
                [term["exp"] for term in item["terms"]]
                for item in trace
            ] == [
                [0, 0, 0, 0],
                [0, 8192, 16384, 24576],
                [0, 16384, 32768, 49152],
                [0, 24576, 49152, 73728],
            ],
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4r3_algorithm1_tweak_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-R3 Algorithm 1 Tweak probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
