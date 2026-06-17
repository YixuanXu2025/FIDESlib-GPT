import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    ccmm_algorithm3_no_relin_skeleton,
    relinearize_ciphertext_matrix,
    pack_relinearized_position_matrix_rowwise,
    describe_rowwise_packed_output,
)


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def run_pipeline(rt, U, V):
    U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
    V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

    W_no_relin, _ = ccmm_algorithm3_no_relin_skeleton(rt, U_bundle, V_bundle)
    W_relin, _ = relinearize_ciphertext_matrix(rt, W_no_relin)
    W_rows, _ = pack_relinearized_position_matrix_rowwise(rt, W_relin)

    W_comp = [rt.compress_coeff_ct(ct, towers_left=1) for ct in W_rows]

    dec_rows = []
    row_reports = []

    for i, ct in enumerate(W_comp):
        dec = safe_call(
            f"decrypt_row_{i}",
            lambda ct=ct: rt.decrypt_coeff_row_i64(ct, logical_length=4),
        )

        row_reports.append({
            "row_index": i,
            "decrypt_ok": dec["ok"],
            "error": dec.get("error"),
            "value": dec.get("value") if dec["ok"] else None,
        })

        if dec["ok"]:
            dec_rows.append(list(map(int, dec["value"])))
        else:
            dec_rows.append(None)

    return dec_rows, row_reports, describe_rowwise_packed_output(rt, W_comp, coeff_sample=4)


def one_hot(n, i, j, value=1):
    M = np.zeros((n, n), dtype=np.int64)
    M[i, j] = value
    return M


def summarize_matrix_error(got, ref):
    if any(r is None for r in got):
        return {
            "all_rows_decrypted": False,
            "exact": False,
            "max_abs_err": None,
            "diff": None,
            "nonzero_positions_got": None,
            "nonzero_positions_ref": [
                [int(i), int(j), int(ref[i, j])]
                for i in range(ref.shape[0])
                for j in range(ref.shape[1])
                if ref[i, j] != 0
            ],
        }

    G = np.array(got, dtype=np.int64)
    D = G - ref

    return {
        "all_rows_decrypted": True,
        "exact": bool(np.array_equal(G, ref)),
        "max_abs_err": int(np.max(np.abs(D))),
        "diff": D.tolist(),
        "nonzero_positions_got": [
            [int(i), int(j), int(G[i, j])]
            for i in range(G.shape[0])
            for j in range(G.shape[1])
            if G[i, j] != 0
        ],
        "nonzero_positions_ref": [
            [int(i), int(j), int(ref[i, j])]
            for i in range(ref.shape[0])
            for j in range(ref.shape[1])
            if ref[i, j] != 0
        ],
    }


def main():
    n = 4

    cases_spec = [
        {
            "name": "basis_E00_times_E00",
            "U_pos": [0, 0],
            "V_pos": [0, 0],
            "zh": "最简单基向量：理论输出 E00",
        },
        {
            "name": "basis_E01_times_E10",
            "U_pos": [0, 1],
            "V_pos": [1, 0],
            "zh": "内积索引匹配：理论输出 E00",
        },
        {
            "name": "basis_E01_times_E12",
            "U_pos": [0, 1],
            "V_pos": [1, 2],
            "zh": "理论输出 E02；用于检查列 placement",
        },
        {
            "name": "basis_E23_times_E31",
            "U_pos": [2, 3],
            "V_pos": [3, 1],
            "zh": "理论输出 E21；用于检查行/列 orientation",
        },
        {
            "name": "identity_times_random_small",
            "kind": "identity_random",
            "zh": "I·V 应该等于 V；用于检查整体方向",
        },
        {
            "name": "random_small_times_identity",
            "kind": "random_identity",
            "zh": "U·I 应该等于 U；用于检查右侧 C-MT/packing 方向",
        },
    ]

    rng = np.random.default_rng(207000)

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
        "experiment": "wp4v3_basis_impulse_probe",
        "purpose": (
            "Diagnose algebra/orientation/placement mismatch after final decrypt succeeds but "
            "does not equal U@V. Use one-hot and identity cases to locate where results land."
        ),
        "status": "diagnostic_probe_after_v2_decrypt_path_success",
        "cases": [],
        "current_state": {
            "V2": "Compress to one tower and final decrypt works, but values differ from U@V.",
            "hypotheses": [
                "Algorithm 3 skeleton orientation is wrong",
                "row-wise packing placement X^j is wrong",
                "C-MT output interpretation is not the simple V_cmt[j] column ciphertext assumed",
                "coefficient multiplication is polynomial convolution, not scalar position multiplication",
            ],
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for spec in cases_spec:
            if spec.get("kind") == "identity_random":
                U = np.eye(n, dtype=np.int64)
                V = rng.integers(-2, 3, size=(n, n)).astype(np.int64)
            elif spec.get("kind") == "random_identity":
                U = rng.integers(-2, 3, size=(n, n)).astype(np.int64)
                V = np.eye(n, dtype=np.int64)
            else:
                U = one_hot(n, spec["U_pos"][0], spec["U_pos"][1])
                V = one_hot(n, spec["V_pos"][0], spec["V_pos"][1])

            W_ref = U @ V

            dec_rows, row_reports, packed_summary = run_pipeline(rt, U, V)

            case = {
                "name": spec["name"],
                "zh": spec["zh"],
                "input_U": U.tolist(),
                "input_V": V.tolist(),
                "reference_U_matmul_V": W_ref.tolist(),
                "decrypted_output": dec_rows,
                "row_reports": row_reports,
                "packed_summary": packed_summary,
                "analysis": summarize_matrix_error(dec_rows, W_ref),
            }

            report["cases"].append(case)

    report["summary"] = {
        "all_cases_decrypted（所有测试用例都成功解密）": all(
            c["analysis"]["all_rows_decrypted"] for c in report["cases"]
        ),
        "exact_cases（完全等于 U@V 的用例名）": [
            c["name"] for c in report["cases"] if c["analysis"]["exact"]
        ],
        "failed_cases（不等于 U@V 的用例名）": [
            c["name"] for c in report["cases"] if not c["analysis"]["exact"]
        ],
        "zh": "如果 one-hot 用例输出位置系统性偏移/转置，就修 orientation/packing；如果输出扩散成多个非零，则当前 Algorithm 3 skeleton 乘法语义不对。",
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4v3_basis_impulse_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-V3 basis impulse diagnostic probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
