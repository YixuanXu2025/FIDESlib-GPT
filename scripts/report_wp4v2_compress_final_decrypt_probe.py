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
    rng = np.random.default_rng(206900)
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
        "experiment": "wp4v2_compress_final_decrypt_probe",
        "purpose": (
            "Compress final packed row-wise coefficient ciphertexts to one RNS tower "
            "and retry coefficient decrypt."
        ),
        "status": "compress_then_final_decrypt_probe",
        "input_U_i64": U.tolist(),
        "input_V_i64": V.tolist(),
        "reference_W_U_matmul_V_i64": W_ref.tolist(),
        "pipeline": {
            "step1": {
                "value": "W_rows = packed row-wise output from real CCMM pipeline",
                "zh": "先运行当前真实 CCMM pipeline，得到 4 个 row-wise 二分量 ciphertext",
            },
            "step2": {
                "value": "W_rows_compressed[i] = Compress(W_rows[i], towers_left=1)",
                "zh": "把每个最终输出行 ciphertext 压缩到 1 个 RNS tower，目标是满足 coefficient decrypt 的 sizeQl==1 限制",
            },
            "step3": {
                "value": "decrypt_coeff_row_i64(W_rows_compressed[i])",
                "zh": "尝试解密压缩后的输出行，并与 U@V 比较",
            },
        },
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        U_bundle = make_rowwise_coeff_ciphertext_matrix(rt, U.tolist())
        V_bundle = make_rowwise_coeff_ciphertext_matrix(rt, V.tolist())

        W_no_relin, _ = ccmm_algorithm3_no_relin_skeleton(rt, U_bundle, V_bundle)
        W_relin, _ = relinearize_ciphertext_matrix(rt, W_no_relin)
        W_rows, _ = pack_relinearized_position_matrix_rowwise(rt, W_relin)

        report["before_compress_summary"] = describe_rowwise_packed_output(rt, W_rows, coeff_sample=4)

        W_rows_compressed = []
        compress_rows = []

        for i, ct_row in enumerate(W_rows):
            comp = safe_call(
                f"compress_output_row_{i}",
                lambda ct_row=ct_row: rt.compress_coeff_ct(ct_row, towers_left=1),
            )

            row_report = {
                "row_index": i,
                "compress": {
                    "ok": comp["ok"],
                    "error": comp.get("error"),
                },
            }

            if comp["ok"]:
                W_rows_compressed.append(comp["value"])
            else:
                W_rows_compressed.append(None)

            compress_rows.append(row_report)

        report["compress_rows"] = compress_rows

        compressed_valid_rows = [ct for ct in W_rows_compressed if ct is not None]

        if len(compressed_valid_rows) == len(W_rows):
            report["after_compress_summary"] = describe_rowwise_packed_output(
                rt,
                compressed_valid_rows,
                coeff_sample=4,
            )

            decrypt_rows = []
            for i, ct_row in enumerate(compressed_valid_rows):
                dec = safe_call(
                    f"decrypt_compressed_output_row_{i}",
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

                decrypt_rows.append(row_report)

            report["decrypt_rows"] = decrypt_rows

        else:
            report["after_compress_summary"] = None
            report["decrypt_rows"] = []

        all_compress_ok = all(r["compress"]["ok"] for r in compress_rows)
        all_decrypt_ok = all(r["decrypt"]["ok"] for r in report["decrypt_rows"]) if report["decrypt_rows"] else False
        all_exact = all(r.get("exact") is True for r in report["decrypt_rows"]) if report["decrypt_rows"] else False

        max_abs_err = None
        if all_decrypt_ok:
            max_abs_err = int(max(r["max_abs_err"] for r in report["decrypt_rows"]))

        after_expected_towers = None
        if report.get("after_compress_summary"):
            after_expected_towers = report["after_compress_summary"]["expected_towers"]["value"]

        report["checks"] = {
            "before_compress_has_4_rows（压缩前最终输出有 4 个 row-wise ciphertext）": report["before_compress_summary"]["num_rows"]["value"] == 4,
            "before_compress_expected_towers_4（压缩前每个 component 有 4 个 RNS towers）": report["before_compress_summary"]["expected_towers"]["value"] == 4,

            "compress_all_rows_ok（4 个最终 row ciphertext 全部 Compress 成功）": all_compress_ok,
            "after_compress_expected_towers_1（压缩后每个 component 只剩 1 个 RNS tower）": after_expected_towers == 1,
            "after_compress_each_has_two_parts（压缩后每个 row ciphertext 仍是 c0/c1 二分量）": (
                report.get("after_compress_summary") is not None
                and all(r["num_parts"]["value"] == 2 for r in report["after_compress_summary"]["rows"])
            ),
            "after_compress_uses_coef_encoding（压缩后仍是 coefficient encoding）": (
                report.get("after_compress_summary") is not None
                and all(r["encoding_type"]["value"] == 1 for r in report["after_compress_summary"]["rows"])
            ),

            "final_decrypt_all_rows_ok（压缩后 4 行全部成功解密）": all_decrypt_ok,
            "final_decrypt_matches_U_matmul_V_exact（压缩后最终解密结果精确等于 U@V）": all_exact,
            "final_decrypt_max_abs_err（压缩后最终解密相对 U@V 的最大绝对误差）": max_abs_err,
        }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4v2_compress_final_decrypt_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-V2 compress final decrypt probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
