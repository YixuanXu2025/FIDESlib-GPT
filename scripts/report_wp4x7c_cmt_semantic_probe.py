import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import (
    make_rowwise_coeff_ciphertext_matrix,
    cmt_algorithm2_rowwise,
    describe_cmt_output,
)


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def one_hot(n, i, j, value=1):
    M = np.zeros((n, n), dtype=np.int64)
    M[i, j] = value
    return M


def decrypt_rows_after_compress(rt, rows, logical_length):
    compressed = []
    compress_reports = []

    for i, ct in enumerate(rows):
        comp = safe_call(
            f"compress_row_{i}",
            lambda ct=ct: rt.compress_coeff_ct(ct, towers_left=1),
        )
        compress_reports.append({
            "row_index": i,
            "ok": comp["ok"],
            "error": comp.get("error"),
        })
        compressed.append(comp["value"] if comp["ok"] else None)

    decoded = []
    decrypt_reports = []

    for i, ct in enumerate(compressed):
        if ct is None:
            decoded.append(None)
            decrypt_reports.append({
                "row_index": i,
                "ok": False,
                "error": "compress failed",
            })
            continue

        dec = safe_call(
            f"decrypt_row_{i}",
            lambda ct=ct: rt.decrypt_coeff_row_i64(ct, logical_length=logical_length),
        )
        decrypt_reports.append({
            "row_index": i,
            "ok": dec["ok"],
            "error": dec.get("error"),
        })

        if dec["ok"]:
            decoded.append(list(map(int, dec["value"])))
        else:
            decoded.append(None)

    return {
        "compress_reports": compress_reports,
        "decrypt_reports": decrypt_reports,
        "decoded": decoded,
    }


def matrix_equal(decoded, ref):
    if any(row is None for row in decoded):
        return False
    return np.array_equal(np.array(decoded, dtype=np.int64), ref)


def diff_matrix(decoded, ref):
    if any(row is None for row in decoded):
        return None
    return (np.array(decoded, dtype=np.int64) - ref).tolist()


def nonzero_positions(M):
    if M is None:
        return None
    out = []
    for i, row in enumerate(M):
        if row is None:
            return None
        for j, v in enumerate(row):
            if int(v) != 0:
                out.append([int(i), int(j), int(v)])
    return out


def run_case(rt, U, name, zh):
    n = U.shape[0]
    bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())

    input_dec = decrypt_rows_after_compress(rt, bundle.rows, logical_length=n)

    cmt_rows, cmt_trace = cmt_algorithm2_rowwise(rt, bundle)
    cmt_dec = decrypt_rows_after_compress(rt, cmt_rows, logical_length=n)

    U_T = U.T.copy()

    return {
        "name": name,
        "zh": zh,
        "input_U": U.tolist(),
        "expected_input_decrypt": U.tolist(),
        "expected_CMT_output_U_transpose": U_T.tolist(),
        "input_decrypt": input_dec,
        "cmt_output_summary": describe_cmt_output(rt, cmt_rows, coeff_sample=4),
        "cmt_auto_alphas": [x["alpha"] for x in cmt_trace["auto"]],
        "cmt_decrypt": cmt_dec,
        "analysis": {
            "input_decrypt_exact": matrix_equal(input_dec["decoded"], U),
            "cmt_decrypt_equals_U_transpose": matrix_equal(cmt_dec["decoded"], U_T),
            "cmt_diff_to_U_transpose": diff_matrix(cmt_dec["decoded"], U_T),
            "expected_nonzero_positions": nonzero_positions(U_T.tolist()),
            "got_nonzero_positions": nonzero_positions(cmt_dec["decoded"]),
            "got_nonzero_count": None if nonzero_positions(cmt_dec["decoded"]) is None else len(nonzero_positions(cmt_dec["decoded"])),
            "expected_nonzero_count": len(nonzero_positions(U_T.tolist())),
        },
    }


def main():
    n = 4
    ring_dim = 1 << 14

    rng = np.random.default_rng(208200)

    cases = [
        {
            "name": "basis_E00",
            "U": one_hot(n, 0, 0),
            "zh": "C-MT 后理论仍为 E00",
        },
        {
            "name": "basis_E01",
            "U": one_hot(n, 0, 1),
            "zh": "C-MT 后理论为 E10",
        },
        {
            "name": "basis_E23",
            "U": one_hot(n, 2, 3),
            "zh": "C-MT 后理论为 E32",
        },
        {
            "name": "random_small",
            "U": rng.integers(-2, 3, size=(n, n)).astype(np.int64),
            "zh": "随机小矩阵，C-MT 后理论为转置",
        },
    ]

    cfg = HEConfig(
        ring_dim=ring_dim,
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
        "experiment": "wp4x7c_cmt_semantic_probe",
        "purpose": (
            "Verify whether the real C-MT implementation is semantically valid for the current "
            "toy setting n=4 while ring_dim=16384. If C-MT(U) does not decrypt to U^T, "
            "then all downstream PP-MM/line5/line6 diagnostics are invalid under this dimension contract."
        ),
        "status": "semantic_probe_no_code_change",
        "n": n,
        "ring_dim": ring_dim,
        "dimension_contract_hypothesis": {
            "value": "Park C-MT assumes N row ciphertexts for an N×N matrix where N is the ring degree; current toy uses only 4 rows with ring_dim=16384.",
            "zh": "当前 4×4 toy 可能不满足论文 C-MT 的维度合同，因此 one-hot 扩散可能来自 C-MT 使用条件不成立。",
        },
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for c in cases:
            report["cases"].append(
                run_case(rt, c["U"], c["name"], c["zh"])
            )

    report["checks"] = {
        "all_input_rows_decrypt_exact（所有输入 row-wise 加密都能正确解密回 U）": all(
            c["analysis"]["input_decrypt_exact"] for c in report["cases"]
        ),
        "all_cmt_outputs_equal_transpose（所有 C-MT 输出都等于 U^T）": all(
            c["analysis"]["cmt_decrypt_equals_U_transpose"] for c in report["cases"]
        ),
        "all_cmt_auto_alphas_expected（所有 C-MT alpha 都是 [1,3,5,7]）": all(
            c["cmt_auto_alphas"] == [1, 3, 5, 7] for c in report["cases"]
        ),
        "any_cmt_case_passes（是否至少有一个 C-MT case 语义正确）": any(
            c["analysis"]["cmt_decrypt_equals_U_transpose"] for c in report["cases"]
        ),
        "cmt_failure_nonzero_summary（C-MT 失败时的非零数量诊断）": [
            {
                "name": c["name"],
                "expected_nonzero_count": c["analysis"]["expected_nonzero_count"],
                "got_nonzero_count": c["analysis"]["got_nonzero_count"],
                "expected_nonzero_positions": c["analysis"]["expected_nonzero_positions"],
                "got_nonzero_positions": c["analysis"]["got_nonzero_positions"],
            }
            for c in report["cases"]
            if not c["analysis"]["cmt_decrypt_equals_U_transpose"]
        ],
    }

    report["summary"] = {
        "next_if_cmt_passes": (
            "C-MT is semantically correct for n=4; continue with plaintext-symbolic line2-line6 simulation."
        ),
        "next_if_cmt_fails": (
            "Stop debugging line5/line6. Fix dimension contract: either test with row_count == ring_dim, "
            "or implement a block/padded C-MT variant whose mathematical dimension matches the logical matrix."
        ),
        "zh": (
            "如果 C-MT 本身不等于转置，则 X6/X7b 的失败根因在维度合同/C-MT 层，不应继续调 PP-MM 或 line5 顺序。"
        ),
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x7c_cmt_semantic_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X7c C-MT semantic probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
