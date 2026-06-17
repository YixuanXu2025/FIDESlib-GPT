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


def main():
    rng = np.random.default_rng(206800)
    U = rng.integers(-2, 3, size=(4, 4)).astype(np.int64)
    V = rng.integers(-2, 3, size=(4, 4)).astype(np.int64)
    W_ref = U @ V

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
        "experiment": "wp4v1_final_decrypt_probe",
        "purpose": (
            "Attempt final decryption of the packed row-wise output of the real CCMM pipeline "
            "and compare against U @ V."
        ),
        "status": "final_decrypt_probe_may_need_modreduce_or_rescale",
        "input_U_i64": U.tolist(),
        "input_V_i64": V.tolist(),
        "reference_W_U_matmul_V_i64": W_ref.tolist(),
        "pipeline": {
            "step1": {
                "value": "U_bundle, V_bundle = row-wise coefficient ciphertext matrices",
                "zh": "按行 coefficient encoding 加密输入矩阵 U 和 V",
            },
            "step2": {
                "value": "W_no_relin = Algorithm3Skeleton(U, V)",
                "zh": "运行 C-MT + no-relin component multiplication，得到位置级三分量 ciphertext",
            },
            "step3": {
                "value": "W_relin = Relinearize(W_no_relin)",
                "zh": "每个位置级 ciphertext 从 c0/c1/c2 压回 c0/c1",
            },
            "step4": {
                "value": "W_rows = pack rows with sum_j X^j * W_relin[i][j]",
                "zh": "把每行 4 个位置级 ciphertext 合成为 1 个 row-wise ciphertext",
            },
            "step5": {
                "value": "decrypt_coeff_row_i64(W_rows[i])",
                "zh": "尝试解密每个输出行，并与 U@V 的对应行比较",
            },
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
        V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

        W_no_relin, skeleton_trace = ccmm_algorithm3_no_relin_skeleton(rt, U_bundle, V_bundle)
        W_relin, relin_trace = relinearize_ciphertext_matrix(rt, W_no_relin)
        W_rows, pack_trace = pack_relinearized_position_matrix_rowwise(rt, W_relin)

        report["packed_output_summary"] = describe_rowwise_packed_output(rt, W_rows, coeff_sample=4)

        decrypt_rows = []
        decoded = []

        for i, ct_row in enumerate(W_rows):
            dec = safe_call(
                f"decrypt_output_row_{i}",
                lambda ct_row=ct_row: rt.decrypt_coeff_row_i64(ct_row, logical_length=4),
            )

            row_report = {
                "row_index": i,
                "decrypt": {
                    "ok": dec["ok"],
                    "error": dec.get("error"),
                },
                "reference_row": W_ref[i].tolist(),
            }

            if dec["ok"]:
                got = np.array(dec["value"], dtype=np.int64)
                diff = got - W_ref[i]
                row_report["decrypted_row"] = got.tolist()
                row_report["diff_to_reference"] = diff.tolist()
                row_report["exact"] = bool(np.array_equal(got, W_ref[i]))
                row_report["max_abs_err"] = int(np.max(np.abs(diff)))
                decoded.append(got.tolist())
            else:
                decoded.append(None)

            decrypt_rows.append(row_report)

        report["decrypt_rows"] = decrypt_rows

        all_decrypt_ok = all(r["decrypt"]["ok"] for r in decrypt_rows)
        all_exact = all(r.get("exact") is True for r in decrypt_rows)

        max_abs_err = None
        if all_decrypt_ok:
            max_abs_err = int(max(r["max_abs_err"] for r in decrypt_rows))

        report["checks"] = {
            "packed_output_has_4_rows（最终 packing 输出 4 个 row-wise ciphertext）": report["packed_output_summary"]["num_rows"]["value"] == 4,
            "packed_output_each_has_two_parts（最终每个 row ciphertext 是 c0/c1 二分量）": all(
                r["num_parts"]["value"] == 2
                for r in report["packed_output_summary"]["rows"]
            ),
            "final_decrypt_all_rows_ok（最终 4 行全部成功解密）": all_decrypt_ok,
            "final_decrypt_matches_U_matmul_V_exact（最终解密结果精确等于 U@V）": all_exact,
            "final_decrypt_max_abs_err（最终解密结果相对 U@V 的最大绝对误差）": max_abs_err,
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4v1_final_decrypt_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-V1 final decrypt probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
