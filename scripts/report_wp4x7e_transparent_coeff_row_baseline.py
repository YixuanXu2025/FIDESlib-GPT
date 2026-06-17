import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.real_ccmm import cmt_algorithm2_rowwise, describe_cmt_output


class TempRowwiseBundle:
    def __init__(self, rows, shape, label, ring_dim=1 << 14, batch_size=8):
        self.rows = rows
        self.shape = shape
        self.orientation = "rowwise"
        self.label = label
        self.ring_dim = int(ring_dim)
        self.batch_size = int(batch_size)


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def normalize_export(x):
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def one_hot(n, i, j, value=1):
    M = np.zeros((n, n), dtype=np.int64)
    M[i, j] = value
    return M


def signed_to_mod(v, q):
    return int(v) % int(q)


def make_zero_1part(rt, template_ct, num_towers):
    return rt.import_1part_coeff_u64(
        template_ct,
        [[] for _ in range(num_towers)],
    )


def make_transparent_coeff_row_ct(rt, template_ct, row, moduli):
    """
    Debug-only transparent coefficient ciphertext.

    OpenFHE component order:
      c0 = B
      c1 = A

    We construct:
      c0 = row coefficients
      c1 = 0

    Therefore:
      decrypt = c0 + c1*s = row

    This is not secure encryption. It is only a semantic baseline.
    """
    towers_coeffs = []
    for q in moduli:
        towers_coeffs.append([signed_to_mod(v, q) for v in row])

    c0 = rt.import_1part_coeff_u64(template_ct, towers_coeffs)
    c1 = make_zero_1part(rt, template_ct, len(moduli))

    return rt.assemble_2part_from_1parts_coeff_ct(c0, c1)


def make_transparent_rowwise_bundle(rt, matrix, ring_dim=1 << 14, batch_size=8):
    n = len(matrix)

    # Template only provides valid OpenFHE metadata / towers / context shape.
    template = rt.encrypt_coeff_row_i64([0 for _ in range(n)])
    template_export = normalize_export(
        rt.export_component_coeff_matrix_u64([template], part=0, coeff_count=1)
    )

    moduli = [int(q) for q in template_export["moduli_u64"]]

    rows = [
        make_transparent_coeff_row_ct(rt, template, list(map(int, row)), moduli)
        for row in matrix
    ]

    return TempRowwiseBundle(
        rows=rows,
        shape=(n, n),
        label="transparent_coeff_rowwise_debug_bundle",
        ring_dim=ring_dim,
        batch_size=batch_size,
    ), {
        "template_num_towers": template_export["num_towers"],
        "template_ring_dim": template_export["ring_dim"],
        "moduli_u64": moduli,
    }


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

        decoded.append(list(map(int, dec["value"])) if dec["ok"] else None)

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

    bundle, template_info = make_transparent_rowwise_bundle(rt, U.tolist())

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
        "template_info": template_info,
        "input_decrypt": input_dec,
        "cmt_output_summary": describe_cmt_output(rt, cmt_rows, coeff_sample=4),
        "cmt_auto_alphas": [x["alpha"] for x in cmt_trace["auto"]],
        "cmt_decrypt": cmt_dec,
        "analysis": {
            "input_decrypt_exact": matrix_equal(input_dec["decoded"], U),
            "input_diff": diff_matrix(input_dec["decoded"], U),
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

    rng = np.random.default_rng(208300)

    cases = [
        {
            "name": "basis_E00",
            "U": one_hot(n, 0, 0),
            "zh": "transparent 输入 E00；C-MT 后理论仍为 E00",
        },
        {
            "name": "basis_E01",
            "U": one_hot(n, 0, 1),
            "zh": "transparent 输入 E01；C-MT 后理论为 E10",
        },
        {
            "name": "basis_E23",
            "U": one_hot(n, 2, 3),
            "zh": "transparent 输入 E23；C-MT 后理论为 E32",
        },
        {
            "name": "random_small",
            "U": rng.integers(-2, 3, size=(n, n)).astype(np.int64),
            "zh": "transparent 随机小矩阵；C-MT 后理论为转置",
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
        "experiment": "wp4x7e_transparent_coeff_row_baseline",
        "purpose": (
            "Build debug-only transparent coefficient ciphertexts with c0=row and c1=0. "
            "This separates coefficient encode/decode semantics from the broken encrypt_coeff_row_i64 path."
        ),
        "status": "debug_transparent_coeff_baseline_no_security_no_ccmm",
        "n": n,
        "ring_dim": ring_dim,
        "warning": {
            "value": "Transparent ciphertext c0=row,c1=0 is not encryption and is not secure.",
            "zh": "这是 debug-only 语义基线，不是最终安全 CCMM 实现。",
        },
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for c in cases:
            report["cases"].append(
                run_case(rt, c["U"], c["name"], c["zh"])
            )

    report["checks"] = {
        "transparent_input_rows_decrypt_exact（transparent c0=row,c1=0 输入能正确解密回 U）": all(
            c["analysis"]["input_decrypt_exact"] for c in report["cases"]
        ),
        "transparent_cmt_outputs_equal_transpose（transparent 输入下 C-MT 输出等于 U^T）": all(
            c["analysis"]["cmt_decrypt_equals_U_transpose"] for c in report["cases"]
        ),
        "all_cmt_auto_alphas_expected（所有 C-MT alpha 都是 [1,3,5,7]）": all(
            c["cmt_auto_alphas"] == [1, 3, 5, 7] for c in report["cases"]
        ),
        "any_transparent_cmt_case_passes（是否至少有一个 transparent C-MT case 语义正确）": any(
            c["analysis"]["cmt_decrypt_equals_U_transpose"] for c in report["cases"]
        ),
        "transparent_cmt_failure_nonzero_summary（transparent C-MT 失败时的非零数量诊断）": [
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
        "if_transparent_input_fails": (
            "The raw import/decrypt path itself is not a usable coefficient baseline; inspect import_1part/decrypt_coeff_row_i64."
        ),
        "if_transparent_input_passes_but_cmt_fails": (
            "Coefficient baseline works, but C-MT semantics fail for n=4/ring_dim=16384. "
            "Fix dimension contract or implement logical-n C-MT."
        ),
        "if_transparent_cmt_passes": (
            "C-MT is semantically valid once coefficient rows are correct. "
            "Then fix encrypt_coeff_row_i64 to produce real ciphertexts with the same coefficient semantics."
        ),
        "next_step": "WP4-X7f",
        "zh": "X7e 用 transparent ciphertext 分离问题：如果输入能 roundtrip 但 C-MT 失败，根因就是 C-MT 维度合同；如果 C-MT 通过，根因就是 encrypt_coeff_row_i64。",
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x7e_transparent_coeff_row_baseline_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X7e transparent coefficient-row baseline")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
