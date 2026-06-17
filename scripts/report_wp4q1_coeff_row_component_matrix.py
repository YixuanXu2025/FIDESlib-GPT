import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


def compact_report(info):
    rows = info.get("rows", [])
    compact_rows = []

    for rr in rows:
        cr = {
            "row_index": rr.get("row_index"),
            "num_parts": rr.get("num_parts"),
            "level": rr.get("level"),
            "noise_scale_deg": rr.get("noise_scale_deg"),
            "scaling_factor": rr.get("scaling_factor"),
            "slots": rr.get("slots"),
            "encoding_type": rr.get("encoding_type"),
            "parts": [],
        }

        for pr in rr.get("parts", []):
            cp = {
                "part_index": pr.get("part_index"),
                "num_towers": pr.get("num_towers"),
                "ring_dim": pr.get("ring_dim"),
                "tower0_modulus_u64": None,
                "tower0_coeff_head_u64": None,
            }

            towers = pr.get("towers", [])
            if towers:
                cp["tower0_modulus_u64"] = towers[0].get("modulus_u64")
                cp["tower0_coeff_head_u64"] = towers[0].get("coeff_head_u64")

            cr["parts"].append(cp)

        compact_rows.append(cr)

    return {
        "num_rows": info.get("num_rows"),
        "consistent_shape": info.get("consistent_shape"),
        "expected_parts": info.get("expected_parts"),
        "expected_towers": info.get("expected_towers"),
        "expected_ring_dim": info.get("expected_ring_dim"),
        "paper_mapping": info.get("paper_mapping"),
        "rows": compact_rows,
    }


def main():
    N = 4
    coeff_sample = 8

    rng = np.random.default_rng(206100)
    M = rng.integers(-3, 4, size=(N, N)).astype(np.int64)

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
        "experiment": "wp4q1_coeff_row_component_matrix",
        "purpose": (
            "Extract/inspect true coefficient-row ciphertext rows as paper-style "
            "row-wise A/B component matrix metadata."
        ),
        "status": "component_extraction_inspect_only",
        "N": N,
        "input_matrix_M_i64": M.tolist(),
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        rows = [
            rt.encrypt_coeff_row_i64(M[i].tolist())
            for i in range(N)
        ]

        info = rt.inspect_coeff_row_component_matrix(rows, coeff_sample=coeff_sample)

        report["component_matrix_summary"] = compact_report(info)
        report["checks"] = {
            "all_rows_are_coef_encoding_type_1": all(
                rr.get("encoding_type") == 1
                for rr in report["component_matrix_summary"]["rows"]
            ),
            "all_rows_have_two_parts": all(
                rr.get("num_parts") == 2
                for rr in report["component_matrix_summary"]["rows"]
            ),
            "consistent_shape": bool(report["component_matrix_summary"]["consistent_shape"]),
            "expected_ring_dim_is_16384": report["component_matrix_summary"]["expected_ring_dim"] == 16384,
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4q1_coeff_row_component_matrix_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Q1 coefficient-row component matrix report")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
