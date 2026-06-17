import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    cmt_algorithm2_rowwise,
)


def compact_json(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def part_head(row_report, part_index):
    parts = row_report.get("parts", [])
    if part_index >= len(parts):
        return None
    towers = parts[part_index].get("towers", [])
    if not towers:
        return None
    return {
        "part_index": part_index,
        "num_towers": parts[part_index].get("num_towers"),
        "ring_dim": parts[part_index].get("ring_dim"),
        "tower0_modulus_u64": towers[0].get("modulus_u64"),
        "tower0_coeff_head_u64": towers[0].get("coeff_head_u64"),
    }


def summarize_component_bundle(info, name):
    rows = []
    for rr in info.get("rows", []):
        rows.append({
            "row_index": rr.get("row_index"),
            "encoding_type": rr.get("encoding_type"),
            "num_parts": rr.get("num_parts"),
            "level": rr.get("level"),
            "slots": rr.get("slots"),
            "B_c0_GetElements0": part_head(rr, 0),
            "A_c1_GetElements1": part_head(rr, 1),
        })

    return {
        "name": name,
        "num_rows": info.get("num_rows"),
        "consistent_shape": info.get("consistent_shape"),
        "expected_parts": info.get("expected_parts"),
        "expected_towers": info.get("expected_towers"),
        "expected_ring_dim": info.get("expected_ring_dim"),
        "mapping": {
            "B_c0": "GetElements()[0]",
            "A_c1": "GetElements()[1]",
            "zh": "每个 row ciphertext 的 c0/c1 component 映射；不是明文矩阵值，只是 DCRTPoly component 摘要",
        },
        "rows": rows,
    }


def main():
    rng = np.random.default_rng(207100)
    U = rng.integers(-2, 3, size=(4, 4)).astype(np.int64)
    V = rng.integers(-2, 3, size=(4, 4)).astype(np.int64)

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
        "experiment": "wp4w1_component_mapping_alignment",
        "purpose": (
            "Align real coefficient ciphertext component matrices with Park-2025 "
            "Algorithm 3 mapping. This explicitly avoids the wrong whole-ciphertext "
            "positionwise skeleton diagnosed by WP4-V3."
        ),
        "status": "component_mapping_alignment_only_no_ppmm_yet",
        "input_U_i64": U.tolist(),
        "input_V_i64": V.tolist(),
        "reference_U_matmul_V_i64": (U @ V).tolist(),
        "diagnosis_from_v3": {
            "value": "whole ciphertext positionwise skeleton caused dense wrong outputs for one-hot tests",
            "zh": "V3 已证明错误不是简单转置/placement，而是 Algorithm 3 映射层错误",
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
        V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

        V_cmt_rows, V_cmt_trace = cmt_algorithm2_rowwise(rt, V_bundle)

        U_info = rt.inspect_coeff_row_component_matrix(U_bundle.rows, coeff_sample=4)
        V_info = rt.inspect_coeff_row_component_matrix(V_bundle.rows, coeff_sample=4)
        V_cmt_info = rt.inspect_coeff_row_component_matrix(V_cmt_rows, coeff_sample=4)

        report["component_bundles"] = {
            "U_rowwise": summarize_component_bundle(U_info, "U_rowwise"),
            "V_rowwise_before_cmt": summarize_component_bundle(V_info, "V_rowwise_before_cmt"),
            "V_after_cmt_transformed": summarize_component_bundle(V_cmt_info, "V_after_cmt_transformed"),
        }

        report["algorithm3_component_mapping"] = {
            "left_components": {
                "L0": {
                    "value": "U_rowwise.B_c0 = U.rows[*].GetElements()[0]",
                    "zh": "左矩阵的 c0 component matrix",
                },
                "L1": {
                    "value": "U_rowwise.A_c1 = U.rows[*].GetElements()[1]",
                    "zh": "左矩阵的 c1 component matrix",
                },
            },
            "right_components_after_cmt": {
                "R0": {
                    "value": "CMT(V).B_c0 = V_cmt[*].GetElements()[0]",
                    "zh": "右矩阵 C-MT 后的 c0 component matrix",
                },
                "R1": {
                    "value": "CMT(V).A_c1 = V_cmt[*].GetElements()[1]",
                    "zh": "右矩阵 C-MT 后的 c1 component matrix",
                },
            },
            "paper_formula": {
                "C0": {
                    "value": "PP-MM(L0, R0)",
                    "zh": "输出 before-relin ciphertext 的 c0 component",
                },
                "C1": {
                    "value": "PP-MM(L0, R1) + PP-MM(L1, R0)",
                    "zh": "输出 before-relin ciphertext 的 c1 component",
                },
                "C2": {
                    "value": "PP-MM(L1, R1)",
                    "zh": "输出 before-relin ciphertext 的 c2 component",
                },
            },
            "important_note": {
                "value": "PP-MM is not whole ciphertext multiplication. It must multiply DCRTPoly component matrices.",
                "zh": "PP-MM 不是整密文相乘，而是 DCRTPoly component matrix 层的结构化乘法",
            },
        }

        got_alphas = [x["alpha"] for x in V_cmt_trace["auto"]]

        report["checks"] = {
            "U_bundle_has_4_rows（U 有 4 个 row-wise ciphertext）": U_info.get("num_rows") == 4,
            "V_bundle_has_4_rows（V 有 4 个 row-wise ciphertext）": V_info.get("num_rows") == 4,
            "V_cmt_has_4_rows（C-MT(V) 有 4 个 transformed ciphertext）": V_cmt_info.get("num_rows") == 4,
            "U_rows_have_two_parts（U 每行都有 c0/c1 两个 component）": U_info.get("expected_parts") == 2,
            "V_cmt_rows_have_two_parts（C-MT(V) 每行都有 c0/c1 两个 component）": V_cmt_info.get("expected_parts") == 2,
            "V_cmt_auto_alphas_expected（C-MT 的 Auto alpha 是 [1,3,5,7]）": got_alphas == [1, 3, 5, 7],
            "paper_mapping_declared（已明确 Algorithm 3 的 L0/L1/R0/R1 到 C0/C1/C2 映射）": True,
            "whole_ct_skeleton_rejected（不再使用 whole ciphertext positionwise skeleton）": True,
        }

        report["summary"] = {
            "next_step": "WP4-W2",
            "next_step_goal": (
                "Implement/probe PP-MM over DCRTPoly component matrices, starting with "
                "component-level zero/clone/add/mul primitives rather than whole ciphertext multiplication."
            ),
            "zh": "下一步实现 DCRTPoly component matrix 乘法 primitive，之后再按 C0/C1/C2 公式组装三分量 ciphertext。",
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4w1_component_mapping_alignment_report.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("WP4-W1 component mapping alignment done")
    print("checks=" + compact_json(report["checks"]))
    print("summary=" + compact_json(report["summary"]))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
