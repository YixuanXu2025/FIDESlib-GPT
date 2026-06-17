import json
from pathlib import Path

import numpy as np

from hegpt.cmt_paper2025 import (
    automorphism_negacyclic,
    matrix_rows_to_polys,
    monomial_mul_negacyclic,
    polys_to_matrix_rows,
    transpose_algorithm2_plain_rows_to_cols,
    tweak_plain,
)


def max_abs(x):
    return float(np.max(np.abs(x)))


def run_case(N: int, seed: int):
    rng = np.random.default_rng(seed)
    M = rng.integers(-3, 4, size=(N, N)).astype(np.float64)

    rows = matrix_rows_to_polys(M)
    cols, trace = transpose_algorithm2_plain_rows_to_cols(rows, return_trace=True)
    out = polys_to_matrix_rows(cols)

    err = max_abs(out - M.T)

    return {
        "N": N,
        "seed": seed,
        "input_M": M.tolist(),
        "output_cols_as_rows": out.tolist(),
        "expected_M_transpose": M.T.tolist(),
        "max_abs_err": err,
        "trace_shapes": {
            "input_rows": list(trace.input_rows.shape),
            "x_i_rows": list(trace.x_i_rows.shape),
            "aux_after_first_tweak": list(trace.aux_after_first_tweak.shape),
            "aux_prime_after_auto": list(trace.aux_prime_after_auto.shape),
            "ct_double_prime_after_second_tweak": list(trace.ct_double_prime_after_second_tweak.shape),
            "output_cols": list(trace.output_cols.shape),
        },
        "trace_head": {
            "aux_after_first_tweak_first_row": trace.aux_after_first_tweak[0].tolist(),
            "aux_prime_after_auto_first_row": trace.aux_prime_after_auto[0].tolist(),
            "ct_double_prime_after_second_tweak_first_row": trace.ct_double_prime_after_second_tweak[0].tolist(),
        },
    }


def primitive_self_tests():
    # Basic ring relation: X^N = -1.
    p = np.array([1.0, 2.0, -1.0, 3.0])
    xN_p = monomial_mul_negacyclic(p, len(p))

    # Auto identity.
    auto_id = automorphism_negacyclic(p, 1)

    # Tweak n=1 returns itself.
    tweak_one = tweak_plain([p])[0]

    return {
        "X_N_equals_minus_one_err": max_abs(xN_p + p),
        "auto_identity_err": max_abs(auto_id - p),
        "tweak_n1_identity_err": max_abs(tweak_one - p),
    }


def main():
    report = {
        "experiment": "wp4m_cmt_algorithm2_plain_toy",
        "purpose": (
            "Validate paper Algorithm 1 Tweak and Algorithm 2 Transpose "
            "at plaintext coefficient-polynomial level."
        ),
        "status": "plain_polynomial_toy_not_he",
        "primitive_self_tests": primitive_self_tests(),
        "cases": [],
    }

    for idx, N in enumerate([2, 4, 8]):
        report["cases"].append(run_case(N, seed=205900 + idx))

    report["summary"] = {
        "all_cases_pass": all(c["max_abs_err"] == 0.0 for c in report["cases"]),
        "max_case_err": max(c["max_abs_err"] for c in report["cases"]),
    }

    print("=" * 100)
    print("WP4-M C-MT Algorithm 2 plaintext toy report")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4m_cmt_algorithm2_plain_toy_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print(f"saved: {out_path}")

    if not report["summary"]["all_cases_pass"]:
        raise SystemExit("WP4-M failed: Algorithm 2 plaintext transpose mismatch")


if __name__ == "__main__":
    main()
