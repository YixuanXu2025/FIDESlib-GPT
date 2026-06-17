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
        tower_mats.append(matmul_mod(A, B, moduli[t]))

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


def make_zero_1part_rows(rt, template_rows, num_towers):
    zeros = []
    zero_towers = [[] for _ in range(num_towers)]
    for tmpl in template_rows:
        zeros.append(rt.import_1part_coeff_u64(tmpl, zero_towers))
    return zeros


def make_temp_pair_bundle(rt, onepart_A_rows, zero_B_rows, n, label, template_bundle=None):
    # Paper pair is (A,B), OpenFHE component order is c0=B, c1=A.
    rows = []
    for A, Bzero in zip(onepart_A_rows, zero_B_rows):
        rows.append(rt.assemble_2part_from_1parts_coeff_ct(Bzero, A))

    return TempRowwiseBundle(
        rows=rows,
        shape=(n, n),
        label=label,
        template_bundle=template_bundle,
    )


def inspect_rows(rt, rows, coeff_sample=4):
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


def rows_are_npart(summary, n, parts):
    return (
        summary.get("num_rows") == n
        and summary.get("expected_parts") == parts
        and all(r.get("num_parts") == parts for r in summary.get("rows", []))
    )


def extract_1part_rows(rt, source_rows, part, coeff_count=0):
    """
    Extract one component from 2-part rows by full raw export/import.

    part:
      0 -> c0/B
      1 -> c1/A

    coeff_count=0 exports full ring dimension.
    """
    exported = normalize_export(
        rt.export_component_coeff_matrix_u64(
            source_rows,
            part=part,
            coeff_count=coeff_count,
        )
    )

    out = []
    for i, row in enumerate(exported["rows"]):
        towers_coeffs = [tower["coeffs_u64"] for tower in row["towers"]]
        out.append(rt.import_1part_coeff_u64(source_rows[i], towers_coeffs))

    return out, {
        "num_rows": exported["num_rows"],
        "part": exported["part"],
        "num_towers": exported["num_towers"],
        "ring_dim": exported["ring_dim"],
        "coeff_count": exported["coeff_count"],
        "moduli_u64": exported["moduli_u64"],
        "row0_tower0_coeff_head": exported["rows"][0]["towers"][0]["coeffs_u64"][:8],
    }


def assemble_M10_M11_pair(rt, M10_rows, M11_rows, n):
    # Paper pair (M10,M11) maps to OpenFHE c0=B=M11, c1=A=M10.
    return [
        rt.assemble_2part_from_1parts_coeff_ct(M11_rows[i], M10_rows[i])
        for i in range(n)
    ]


def main():
    n = 4
    coeff_count = n

    rng = np.random.default_rng(208000)
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
        "experiment": "wp4x5_line5_line6_structure_probe",
        "purpose": (
            "Implement Park-2025 Algorithm 3 line5 and line6 pre-rescale structure: "
            "KS_{s²->s}((A_hat,0)) + (B_hat,0), then add Transpose((M01,0)) and (M10,M11)."
        ),
        "status": "line5_line6_pre_rescale_structure_probe_no_final_rescale_no_decrypt",
        "n": n,
        "coeff_count_for_raw_ppmm": coeff_count,
        "input_U_i64": U.tolist(),
        "input_V_i64": V.tolist(),
        "reference_U_matmul_V_i64": (U @ V).tolist(),
        "component_order": {
            "paper_pair": "(A,B)",
            "openfhe_components": "c0=B, c1=A",
            "line5_ks_input": "A_hat*s^2 is encoded as c0=0,c1=0,c2=A_hat",
            "line5_Bhat_term": "(B_hat,0) is encoded as c0=0,c1=B_hat",
            "line6_M10_M11": "(M10,M11) is encoded as c0=M11,c1=M10",
        },
        "paper_algorithm3_progress": {
            "line1": "(A_U,B_U)=Transpose(ct_U)",
            "line2": "M00=A_U@A_V, M01=A_U@B_V, M10=B_U@A_V, M11=B_U@B_V",
            "line3": "(A_check,B_check)=Transpose((M01,0))",
            "line4": "(A_hat,B_hat)=Transpose((M00,0))",
            "line5": "(A_hat,B_hat)=KS_{s²->s}((A_hat,0))+(B_hat,0)",
            "line6_pre_rescale": "W_pre=line5_result+(A_check,B_check)+(M10,M11)",
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
        V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

        # Line 1
        U_cmt, U_cmt_trace = cmt_algorithm2_rowwise(rt, U_bundle)

        # Line 2 raw RNS PP-MM
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

        zero_rows = make_zero_1part_rows(rt, U_cmt, num_towers=M00["num_towers"])

        # Lines 3 and 4 temporary pair C-MT.
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

        T01_cmt_rows, T01_cmt_trace = cmt_algorithm2_rowwise(rt, T01_bundle)
        T00_cmt_rows, T00_cmt_trace = cmt_algorithm2_rowwise(rt, T00_bundle)

        report["line3_T01_after_cmt"] = describe_cmt_output(rt, T01_cmt_rows, coeff_sample=4)
        report["line4_T00_after_cmt"] = describe_cmt_output(rt, T00_cmt_rows, coeff_sample=4)

        # Extract line4 result:
        # T00_cmt gives paper pair (A_hat, B_hat):
        #   c0=B_hat
        #   c1=A_hat
        Bhat_rows, Bhat_export_summary = extract_1part_rows(rt, T00_cmt_rows, part=0, coeff_count=0)
        Ahat_rows, Ahat_export_summary = extract_1part_rows(rt, T00_cmt_rows, part=1, coeff_count=0)

        report["line4_extracted_components"] = {
            "Ahat_from_part1": Ahat_export_summary,
            "Bhat_from_part0": Bhat_export_summary,
        }

        # Line 5:
        # KS_{s²->s}((A_hat,0)) + (B_hat,0)
        line5_ks_inputs = []
        line5_ks_outputs = []
        line5_bhat_terms = []
        line5_rows = []

        for i in range(n):
            # c0=0, c1=0, c2=A_hat
            ks_input = rt.assemble_3part_from_1parts_coeff_ct(
                zero_rows[i],
                zero_rows[i],
                Ahat_rows[i],
            )
            line5_ks_inputs.append(ks_input)

            ks_output = rt.relinearize_coeff_ct(ks_input)
            line5_ks_outputs.append(ks_output)

            # (B_hat,0) in paper pair maps to c0=0, c1=B_hat.
            bhat_term = rt.assemble_2part_from_1parts_coeff_ct(
                zero_rows[i],
                Bhat_rows[i],
            )
            line5_bhat_terms.append(bhat_term)

            line5_rows.append(rt.add_coeff_ct(ks_output, bhat_term))

        report["line5"] = {
            "ks_inputs_3part": inspect_rows(rt, line5_ks_inputs, coeff_sample=4),
            "ks_outputs_2part": inspect_rows(rt, line5_ks_outputs, coeff_sample=4),
            "bhat_terms_2part": inspect_rows(rt, line5_bhat_terms, coeff_sample=4),
            "line5_rows_2part": inspect_rows(rt, line5_rows, coeff_sample=4),
            "zh": "line5_rows = Relinearize(c0=0,c1=0,c2=A_hat) + (c0=0,c1=B_hat)",
        }

        # Line 6 pre-rescale:
        # W_pre = line5 + Transpose((M01,0)) + (M10,M11)
        M10_M11_pair_rows = assemble_M10_M11_pair(rt, M10_rows, M11_rows, n)

        line6_sum1_rows = []
        line6_pre_rows = []

        for i in range(n):
            s1 = rt.add_coeff_ct(line5_rows[i], T01_cmt_rows[i])
            line6_sum1_rows.append(s1)

            final_pre = rt.add_coeff_ct(s1, M10_M11_pair_rows[i])
            line6_pre_rows.append(final_pre)

        report["line6_pre_rescale"] = {
            "T01_cmt_rows_2part": inspect_rows(rt, T01_cmt_rows, coeff_sample=4),
            "M10_M11_pair_rows_2part": inspect_rows(rt, M10_M11_pair_rows, coeff_sample=4),
            "line5_plus_T01_2part": inspect_rows(rt, line6_sum1_rows, coeff_sample=4),
            "line6_pre_rows_2part": inspect_rows(rt, line6_pre_rows, coeff_sample=4),
            "zh": "line6_pre_rows = line5_rows + Transpose((M01,0)) + (M10,M11)，尚未 Rescale",
        }

        U_cmt_alphas = [x["alpha"] for x in U_cmt_trace["auto"]]
        T01_alphas = [x["alpha"] for x in T01_cmt_trace["auto"]]
        T00_alphas = [x["alpha"] for x in T00_cmt_trace["auto"]]

        report["checks"] = {
            "algorithm3_transposes_U_not_V（Algorithm 3 先 C-MT(U)，不是 C-MT(V)）": True,
            "U_cmt_auto_alphas_expected（C-MT(U) alpha 是 [1,3,5,7]）": U_cmt_alphas == [1, 3, 5, 7],
            "T01_cmt_auto_alphas_expected（Transpose((M01,0)) alpha 是 [1,3,5,7]）": T01_alphas == [1, 3, 5, 7],
            "T00_cmt_auto_alphas_expected（Transpose((M00,0)) alpha 是 [1,3,5,7]）": T00_alphas == [1, 3, 5, 7],

            "line5_ks_inputs_are_3part（line5 的 KS 输入是 c0/c1/c2 三分量，c2=A_hat）": rows_are_npart(report["line5"]["ks_inputs_3part"], n, 3),
            "line5_ks_outputs_are_2part（line5 的 KS/relinearize 输出是 c0/c1 二分量）": rows_are_npart(report["line5"]["ks_outputs_2part"], n, 2),
            "line5_bhat_terms_are_2part（line5 的 (B_hat,0) 项是 c0/c1 二分量）": rows_are_npart(report["line5"]["bhat_terms_2part"], n, 2),
            "line5_rows_are_2part（line5 相加结果是 c0/c1 二分量）": rows_are_npart(report["line5"]["line5_rows_2part"], n, 2),

            "line6_T01_rows_are_2part（line6 的 Transpose((M01,0)) 项是二分量）": rows_are_npart(report["line6_pre_rescale"]["T01_cmt_rows_2part"], n, 2),
            "line6_M10_M11_pair_rows_are_2part（line6 的 (M10,M11) 项是二分量）": rows_are_npart(report["line6_pre_rescale"]["M10_M11_pair_rows_2part"], n, 2),
            "line6_pre_rows_are_2part（line6 pre-rescale 三项相加后仍是二分量）": rows_are_npart(report["line6_pre_rescale"]["line6_pre_rows_2part"], n, 2),

            "ready_for_X6_rescale_or_final_decode_probe（line5/line6 pre-rescale 结构完成，可进入 rescale/final decode 探针）": rows_are_npart(report["line6_pre_rescale"]["line6_pre_rows_2part"], n, 2),
        }

        report["summary"] = {
            "next_step": "WP4-X6",
            "next_step_goal": (
                "Probe whether line6_pre_rows need explicit Rescale/ModReduce before final "
                "compress/decrypt, then run one-hot diagnostics on the full paper-shaped pipeline."
            ),
            "zh": "X5 完成 line5/line6 pre-rescale 结构；下一步处理 Rescale/Compress/Decrypt 并做 one-hot 诊断。",
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x5_line5_line6_structure_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X5 line5 / line6 structure probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
