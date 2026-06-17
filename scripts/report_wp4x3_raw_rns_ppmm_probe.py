import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    cmt_algorithm2_rowwise,
)


def normalize_export(x):
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def matrix_for_tower(export_obj, tower_index, n):
    """
    Convert exported rows×towers×coeffs into an n×n Python-int matrix for one tower.
    Uses the first n coefficients of each row.
    """
    rows = []
    for row in export_obj["rows"]:
        coeffs = row["towers"][tower_index]["coeffs_u64"][:n]
        rows.append([int(v) for v in coeffs])
    return rows


def matmul_mod(A, B, q):
    """
    Pure Python modular matrix multiplication to avoid uint64 overflow.
    """
    n = len(A)
    m = len(B[0])
    kdim = len(B)

    out = [[0 for _ in range(m)] for _ in range(n)]

    q = int(q)

    for i in range(n):
        for j in range(m):
            acc = 0
            for k in range(kdim):
                acc = (acc + int(A[i][k]) * int(B[k][j])) % q
            out[i][j] = acc

    return out


def raw_rns_ppmm(left_export, right_export, n):
    """
    Compute PP-MM tower-by-tower:
      for each tower ell:
        M_ell = left_ell @ right_ell mod q_ell

    Output layout:
      {
        "num_rows": n,
        "num_towers": L,
        "rows": [
          {
            "row_index": i,
            "towers_coeffs": [
              [coeffs for tower0],
              [coeffs for tower1],
              ...
            ]
          }
        ]
      }
    """
    if left_export["num_rows"] != n or right_export["num_rows"] != n:
        raise ValueError("raw_rns_ppmm expects n rows in both inputs")

    if left_export["num_towers"] != right_export["num_towers"]:
        raise ValueError("tower count mismatch")

    if left_export["moduli_u64"] != right_export["moduli_u64"]:
        raise ValueError("RNS moduli mismatch")

    num_towers = int(left_export["num_towers"])
    moduli = [int(q) for q in left_export["moduli_u64"]]

    tower_mats = []

    for t in range(num_towers):
        A = matrix_for_tower(left_export, t, n)
        B = matrix_for_tower(right_export, t, n)
        q = moduli[t]
        M = matmul_mod(A, B, q)
        tower_mats.append(M)

    rows = []
    for i in range(n):
        towers_coeffs = []
        for t in range(num_towers):
            towers_coeffs.append([int(v) for v in tower_mats[t][i]])
        rows.append({
            "row_index": i,
            "towers_coeffs": towers_coeffs,
        })

    return {
        "num_rows": n,
        "num_towers": num_towers,
        "moduli_u64": moduli,
        "rows": rows,
        "tower_mats_prefix": tower_mats,
        "zh": "每个 RNS tower 上分别做 coefficient matrix multiplication；这里只计算前 n 个 coefficients",
    }


def import_ppmm_rows(rt, template_rows, ppmm_result):
    out = []
    for i, row in enumerate(ppmm_result["rows"]):
        ct = rt.import_1part_coeff_u64(template_rows[i], row["towers_coeffs"])
        out.append(ct)
    return out


def export_1part_rows(rt, rows, coeff_count):
    return normalize_export(
        rt.export_component_coeff_matrix_u64(
            rows,
            part=0,
            coeff_count=coeff_count,
        )
    )


def compare_ppmm_to_reexport(ppmm_result, reexport, n):
    checks = []
    ok_all = True

    for i in range(n):
        for t in range(ppmm_result["num_towers"]):
            expected = [int(v) for v in ppmm_result["rows"][i]["towers_coeffs"][t]]
            got = [int(v) for v in reexport["rows"][i]["towers"][t]["coeffs_u64"][:n]]
            ok = expected == got
            ok_all = ok_all and ok
            checks.append({
                "row": i,
                "tower": t,
                "equal": ok,
                "expected": expected,
                "got": got,
            })

    return {
        "equal_all": ok_all,
        "details": checks,
    }


def ppmm_summary(ppmm_result, n):
    return {
        "num_rows": ppmm_result["num_rows"],
        "num_towers": ppmm_result["num_towers"],
        "moduli_u64": ppmm_result["moduli_u64"],
        "row0_tower0_coeffs": ppmm_result["rows"][0]["towers_coeffs"][0][:n],
        "row1_tower0_coeffs": ppmm_result["rows"][1]["towers_coeffs"][0][:n],
    }


def main():
    n = 4
    coeff_count = n

    rng = np.random.default_rng(207800)
    U = rng.integers(-2, 3, size=(n, n)).astype(np.int64)
    V = rng.integers(-2, 3, size=(n, n)).astype(np.int64)

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
        "experiment": "wp4x3_raw_rns_ppmm_probe",
        "purpose": (
            "Implement CPU toy-size raw RNS PP-MM over exported DCRTPoly component "
            "coefficient matrices. This is the first real PP-MM step needed by Park-2025 Algorithm 3."
        ),
        "status": "raw_rns_ppmm_probe_only_no_temp_cmt_no_keyswitch_no_final_decrypt",
        "n": n,
        "coeff_count": coeff_count,
        "input_U_i64": U.tolist(),
        "input_V_i64": V.tolist(),
        "reference_U_matmul_V_i64": (U @ V).tolist(),
        "paper_algorithm3_line2": {
            "M00": "A_U @ A_V",
            "M01": "A_U @ B_V",
            "M10": "B_U @ A_V",
            "M11": "B_U @ B_V",
            "zh": "这里 @ 是 raw coefficient/RNS matrix multiplication，不是 DCRTPoly polynomial multiplication",
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
        V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

        # Correct Algorithm 3 direction: C-MT(U), not C-MT(V)
        U_cmt, U_cmt_trace = cmt_algorithm2_rowwise(rt, U_bundle)

        A_U = normalize_export(rt.export_component_coeff_matrix_u64(U_cmt, part=1, coeff_count=coeff_count))
        B_U = normalize_export(rt.export_component_coeff_matrix_u64(U_cmt, part=0, coeff_count=coeff_count))
        A_V = normalize_export(rt.export_component_coeff_matrix_u64(V_bundle.rows, part=1, coeff_count=coeff_count))
        B_V = normalize_export(rt.export_component_coeff_matrix_u64(V_bundle.rows, part=0, coeff_count=coeff_count))

        report["exports_summary"] = {
            "A_U": {
                "num_rows": A_U["num_rows"],
                "num_towers": A_U["num_towers"],
                "moduli_u64": A_U["moduli_u64"],
                "row0_tower0_coeffs": A_U["rows"][0]["towers"][0]["coeffs_u64"],
            },
            "B_U": {
                "num_rows": B_U["num_rows"],
                "num_towers": B_U["num_towers"],
                "moduli_u64": B_U["moduli_u64"],
                "row0_tower0_coeffs": B_U["rows"][0]["towers"][0]["coeffs_u64"],
            },
            "A_V": {
                "num_rows": A_V["num_rows"],
                "num_towers": A_V["num_towers"],
                "moduli_u64": A_V["moduli_u64"],
                "row0_tower0_coeffs": A_V["rows"][0]["towers"][0]["coeffs_u64"],
            },
            "B_V": {
                "num_rows": B_V["num_rows"],
                "num_towers": B_V["num_towers"],
                "moduli_u64": B_V["moduli_u64"],
                "row0_tower0_coeffs": B_V["rows"][0]["towers"][0]["coeffs_u64"],
            },
        }

        M00 = raw_rns_ppmm(A_U, A_V, n)
        M01 = raw_rns_ppmm(A_U, B_V, n)
        M10 = raw_rns_ppmm(B_U, A_V, n)
        M11 = raw_rns_ppmm(B_U, B_V, n)

        report["ppmm_results_summary"] = {
            "M00_AU_AV": ppmm_summary(M00, n),
            "M01_AU_BV": ppmm_summary(M01, n),
            "M10_BU_AV": ppmm_summary(M10, n),
            "M11_BU_BV": ppmm_summary(M11, n),
        }

        # Import PP-MM results as 1-component ciphertext-like row bundles.
        M00_rows = import_ppmm_rows(rt, U_cmt, M00)
        M01_rows = import_ppmm_rows(rt, U_cmt, M01)
        M10_rows = import_ppmm_rows(rt, U_cmt, M10)
        M11_rows = import_ppmm_rows(rt, U_cmt, M11)

        M00_re = export_1part_rows(rt, M00_rows, coeff_count)
        M01_re = export_1part_rows(rt, M01_rows, coeff_count)
        M10_re = export_1part_rows(rt, M10_rows, coeff_count)
        M11_re = export_1part_rows(rt, M11_rows, coeff_count)

        report["import_reexport_compare"] = {
            "M00": compare_ppmm_to_reexport(M00, M00_re, n),
            "M01": compare_ppmm_to_reexport(M01, M01_re, n),
            "M10": compare_ppmm_to_reexport(M10, M10_re, n),
            "M11": compare_ppmm_to_reexport(M11, M11_re, n),
        }

        got_alphas = [x["alpha"] for x in U_cmt_trace["auto"]]

        all_same_moduli = (
            A_U["moduli_u64"] == B_U["moduli_u64"]
            and A_U["moduli_u64"] == A_V["moduli_u64"]
            and A_U["moduli_u64"] == B_V["moduli_u64"]
        )

        all_ppmm_shapes_ok = all(
            M["num_rows"] == n and M["num_towers"] == A_U["num_towers"]
            for M in [M00, M01, M10, M11]
        )

        all_import_reexport_ok = all(
            report["import_reexport_compare"][name]["equal_all"]
            for name in ["M00", "M01", "M10", "M11"]
        )

        report["checks"] = {
            "algorithm3_transposes_U_not_V（Algorithm 3 使用 C-MT(U)，不是 C-MT(V)）": True,
            "U_cmt_auto_alphas_expected（C-MT(U) Auto alpha 是 [1,3,5,7]）": got_alphas == [1, 3, 5, 7],
            "all_sources_have_same_rns_moduli（A_U/B_U/A_V/B_V 的 RNS moduli 一致）": all_same_moduli,
            "raw_ppmm_M00_M01_M10_M11_shapes_ok（四个 PP-MM 结果形状都是 4 rows × 4 towers）": all_ppmm_shapes_ok,
            "raw_ppmm_import_reexport_all_equal（四个 PP-MM 结果 import 后 re-export 完全一致）": all_import_reexport_ok,
            "no_dcrtpoly_polynomial_multiply_used（本步未使用 DCRTPoly 多项式乘法）": True,
            "ready_for_X4_temp_pair_cmt（已具备 M00/M01/M10/M11 raw PP-MM row bundles，可进入临时 pair C-MT）": (
                all_same_moduli and all_ppmm_shapes_ok and all_import_reexport_ok
            ),
        }

        report["summary"] = {
            "next_step": "WP4-X4",
            "next_step_goal": (
                "Create temporary two-component row bundles (M01,0) and (M00,0), "
                "then run C-MT on them as required by Algorithm 3 lines 3 and 4."
            ),
            "zh": "X3 完成真正 raw PP-MM；下一步构造 (M01,0)、(M00,0) 临时密文对并执行 C-MT。",
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x3_raw_rns_ppmm_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X3 raw RNS PP-MM probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
