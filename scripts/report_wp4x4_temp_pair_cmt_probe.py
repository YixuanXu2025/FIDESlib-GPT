import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    cmt_algorithm2_rowwise,
    describe_cmt_output,
)


class TempRowwiseBundle:
    def __init__(self, rows, shape, label, template_bundle=None, ring_dim=None):
        self.rows = rows
        self.shape = shape
        self.orientation = "rowwise"
        self.label = label

        if ring_dim is not None:
            self.ring_dim = int(ring_dim)
        elif template_bundle is not None and hasattr(template_bundle, "ring_dim"):
            self.ring_dim = int(template_bundle.ring_dim)
        else:
            self.ring_dim = 1 << 14

        if template_bundle is not None and hasattr(template_bundle, "batch_size"):
            self.batch_size = int(template_bundle.batch_size)
        else:
            self.batch_size = int(shape[1])


def normalize_export(x):
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def matrix_for_tower(export_obj, tower_index, n):
    rows = []
    for row in export_obj["rows"]:
        coeffs = row["towers"][tower_index]["coeffs_u64"][:n]
        rows.append([int(v) for v in coeffs])
    return rows


def matmul_mod(A, B, q):
    n = len(A)
    m = len(B[0])
    kdim = len(B)
    q = int(q)

    out = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            acc = 0
            for k in range(kdim):
                acc = (acc + int(A[i][k]) * int(B[k][j])) % q
            out[i][j] = acc

    return out


def raw_rns_ppmm(left_export, right_export, n):
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
        M = matmul_mod(A, B, moduli[t])
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
    }


def import_ppmm_rows(rt, template_rows, ppmm_result):
    out = []
    for i, row in enumerate(ppmm_result["rows"]):
        ct = rt.import_1part_coeff_u64(template_rows[i], row["towers_coeffs"])
        out.append(ct)
    return out


def make_zero_1part_rows(rt, template_rows, num_towers, n):
    zeros = []
    zero_towers = [[0 for _ in range(n)] for _ in range(num_towers)]
    for tmpl in template_rows:
        zeros.append(rt.import_1part_coeff_u64(tmpl, zero_towers))
    return zeros


def make_temp_pair_bundle(rt, onepart_A_rows, zero_B_rows, n, label, template_bundle=None):
    # Paper temporary pair is written as (A, B).
    # OpenFHE component order is (c0=B, c1=A).
    #
    # Therefore:
    #   (M01, 0) -> c0=zero, c1=M01
    #   (M00, 0) -> c0=zero, c1=M00
    rows = []
    for A, Bzero in zip(onepart_A_rows, zero_B_rows):
        rows.append(rt.assemble_2part_from_1parts_coeff_ct(Bzero, A))

    return TempRowwiseBundle(
        rows=rows,
        shape=(n, n),
        label=label,
        template_bundle=template_bundle,
    )


def inspect_2part_rows(rt, rows, coeff_sample=4):
    info = rt.inspect_coeff_row_component_matrix(rows, coeff_sample=coeff_sample)
    return {
        "num_rows": info.get("num_rows"),
        "consistent_shape": info.get("consistent_shape"),
        "expected_parts": info.get("expected_parts"),
        "expected_towers": info.get("expected_towers"),
        "expected_ring_dim": info.get("expected_ring_dim"),
        "rows": [
            {
                "row_index": r.get("row_index"),
                "encoding_type": r.get("encoding_type"),
                "num_parts": r.get("num_parts"),
                "level": r.get("level"),
                "slots": r.get("slots"),
            }
            for r in info.get("rows", [])
        ],
    }


def rows_are_2part(summary, n):
    return (
        summary.get("num_rows") == n
        and summary.get("expected_parts") == 2
        and all(r.get("num_parts") == 2 for r in summary.get("rows", []))
    )


def main():
    n = 4
    coeff_count = n

    rng = np.random.default_rng(207900)
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
        "experiment": "wp4x4_temp_pair_cmt_probe",
        "purpose": (
            "Construct temporary two-component bundles (M01,0) and (M00,0), then run C-MT on them "
            "as required by Park-2025 Algorithm 3 lines 3 and 4."
        ),
        "status": "temp_pair_cmt_probe_no_keyswitch_no_final_decrypt",
        "n": n,
        "coeff_count": coeff_count,
        "input_U_i64": U.tolist(),
        "input_V_i64": V.tolist(),
        "reference_U_matmul_V_i64": (U @ V).tolist(),
        "component_order": {
            "paper_pair": "(A, B)",
            "openfhe_elements": "GetElements()[0]=c0=B, GetElements()[1]=c1=A",
            "T01": "(M01,0) is encoded as c0=0, c1=M01",
            "T00": "(M00,0) is encoded as c0=0, c1=M00",
        },
        "paper_algorithm3_lines": {
            "line1": "(A_U,B_U)=Transpose(ct_U)",
            "line2": "M00=A_U@A_V, M01=A_U@B_V, M10=B_U@A_V, M11=B_U@B_V",
            "line3": "(A_check,B_check)=Transpose((M01,0))",
            "line4": "(A_hat,B_hat)=Transpose((M00,0))",
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
        V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

        U_cmt, U_cmt_trace = cmt_algorithm2_rowwise(rt, U_bundle)

        A_U = normalize_export(rt.export_component_coeff_matrix_u64(U_cmt, part=1, coeff_count=coeff_count))
        B_U = normalize_export(rt.export_component_coeff_matrix_u64(U_cmt, part=0, coeff_count=coeff_count))
        A_V = normalize_export(rt.export_component_coeff_matrix_u64(V_bundle.rows, part=1, coeff_count=coeff_count))
        B_V = normalize_export(rt.export_component_coeff_matrix_u64(V_bundle.rows, part=0, coeff_count=coeff_count))

        M00 = raw_rns_ppmm(A_U, A_V, n)
        M01 = raw_rns_ppmm(A_U, B_V, n)
        M10 = raw_rns_ppmm(B_U, A_V, n)
        M11 = raw_rns_ppmm(B_U, B_V, n)

        M00_rows = import_ppmm_rows(rt, U_cmt, M00)
        M01_rows = import_ppmm_rows(rt, U_cmt, M01)
        M10_rows = import_ppmm_rows(rt, U_cmt, M10)
        M11_rows = import_ppmm_rows(rt, U_cmt, M11)

        zero_rows = make_zero_1part_rows(
            rt,
            template_rows=U_cmt,
            num_towers=M00["num_towers"],
            n=n,
        )

        T01_bundle = make_temp_pair_bundle(
            rt,
            onepart_A_rows=M01_rows,
            zero_B_rows=zero_rows,
            n=n,
            label="T01=(M01,0)",
            template_bundle=U_bundle,
        )

        T00_bundle = make_temp_pair_bundle(
            rt,
            onepart_A_rows=M00_rows,
            zero_B_rows=zero_rows,
            n=n,
            label="T00=(M00,0)",
            template_bundle=U_bundle,
        )

        report["temp_pair_bundles"] = {
            "T01_M01_0": inspect_2part_rows(rt, T01_bundle.rows, coeff_sample=4),
            "T00_M00_0": inspect_2part_rows(rt, T00_bundle.rows, coeff_sample=4),
        }

        T01_cmt_rows, T01_cmt_trace = cmt_algorithm2_rowwise(rt, T01_bundle)
        T00_cmt_rows, T00_cmt_trace = cmt_algorithm2_rowwise(rt, T00_bundle)

        report["T01_after_cmt"] = describe_cmt_output(rt, T01_cmt_rows, coeff_sample=4)
        report["T00_after_cmt"] = describe_cmt_output(rt, T00_cmt_rows, coeff_sample=4)

        # M10 and M11 are already one-component row bundles from raw PP-MM.
        # For line6 they will later be combined as paper pair (M10, M11),
        # which maps to OpenFHE c0=B=M11, c1=A=M10.
        M10_M11_pair_rows = [
            rt.assemble_2part_from_1parts_coeff_ct(M11_rows[i], M10_rows[i])
            for i in range(n)
        ]

        report["M10_M11_available_for_line6"] = {
            "M10_rows_count": len(M10_rows),
            "M11_rows_count": len(M11_rows),
            "M10_M11_pair_inspect": inspect_2part_rows(rt, M10_M11_pair_rows, coeff_sample=4),
            "zh": "line6 需要加入 paper pair (M10,M11)，OpenFHE component order 为 c0=M11, c1=M10。",
        }

        U_cmt_alphas = [x["alpha"] for x in U_cmt_trace["auto"]]
        T01_alphas = [x["alpha"] for x in T01_cmt_trace["auto"]]
        T00_alphas = [x["alpha"] for x in T00_cmt_trace["auto"]]

        report["checks"] = {
            "algorithm3_transposes_U_not_V（Algorithm 3 先 C-MT(U)，不是 C-MT(V)）": True,
            "U_cmt_auto_alphas_expected（C-MT(U) alpha 是 [1,3,5,7]）": U_cmt_alphas == [1, 3, 5, 7],

            "T01_temp_pair_is_2part（临时 bundle T01=(M01,0) 是 c0/c1 二分量）": rows_are_2part(report["temp_pair_bundles"]["T01_M01_0"], n),
            "T00_temp_pair_is_2part（临时 bundle T00=(M00,0) 是 c0/c1 二分量）": rows_are_2part(report["temp_pair_bundles"]["T00_M00_0"], n),

            "T01_cmt_has_4_rows（Transpose((M01,0)) 输出 4 行）": report["T01_after_cmt"]["num_rows"] == n,
            "T00_cmt_has_4_rows（Transpose((M00,0)) 输出 4 行）": report["T00_after_cmt"]["num_rows"] == n,
            "T01_cmt_rows_are_2part（Transpose((M01,0)) 输出仍是二分量）": all(
                r["num_parts"]["value"] == 2 for r in report["T01_after_cmt"]["rows"]
            ),
            "T00_cmt_rows_are_2part（Transpose((M00,0)) 输出仍是二分量）": all(
                r["num_parts"]["value"] == 2 for r in report["T00_after_cmt"]["rows"]
            ),
            "T01_cmt_auto_alphas_expected（Transpose((M01,0)) alpha 是 [1,3,5,7]）": T01_alphas == [1, 3, 5, 7],
            "T00_cmt_auto_alphas_expected（Transpose((M00,0)) alpha 是 [1,3,5,7]）": T00_alphas == [1, 3, 5, 7],

            "M10_M11_raw_ppmm_pair_available（line6 的 (M10,M11) pair 已可构造）": rows_are_2part(report["M10_M11_available_for_line6"]["M10_M11_pair_inspect"], n),

            "ready_for_X5_keyswitch_line5（已完成 line3/line4 临时 pair C-MT，可进入 line5 KS_{s²->s} 组装探针）": (
                report["T01_after_cmt"]["num_rows"] == n
                and report["T00_after_cmt"]["num_rows"] == n
                and T01_alphas == [1, 3, 5, 7]
                and T00_alphas == [1, 3, 5, 7]
            ),
        }

        report["summary"] = {
            "next_step": "WP4-X5",
            "next_step_goal": (
                "Implement Algorithm 3 line5/line6 structure: use Transpose((M00,0)) result as "
                "the sk^2-related term, key-switch/relinearize it back to sk, then combine with "
                "Transpose((M01,0)) and (M10,M11)."
            ),
            "zh": "X4 完成临时 pair C-MT；下一步处理 line5 的 KS_{s²→s} 与 line6 的三项组合。",
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x4_temp_pair_cmt_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X4 temporary pair C-MT probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
