import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime


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
    towers_coeffs = []
    for q in moduli:
        towers_coeffs.append([signed_to_mod(v, q) for v in row])

    c0 = rt.import_1part_coeff_u64(template_ct, towers_coeffs)
    c1 = make_zero_1part(rt, template_ct, len(moduli))

    # OpenFHE component order:
    #   c0 = B
    #   c1 = A
    #
    # transparent plaintext row:
    #   c0=row, c1=0
    return rt.assemble_2part_from_1parts_coeff_ct(c0, c1)


def make_transparent_rowwise_bundle(rt, matrix, ring_dim=1 << 14, batch_size=8):
    n = len(matrix)

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


def matrix_for_tower(export_obj, tower_index, n):
    out = []
    for row in export_obj["rows"]:
        out.append([int(v) for v in row["towers"][tower_index]["coeffs_u64"][:n]])
    return out


def transpose_matrix(M):
    n = len(M)
    return [[M[i][j] for i in range(n)] for j in range(n)]


def logical_n_transpose_export(export_obj, n):
    """
    Oracle transpose for the first n×n coefficient block.

    Input:
      export_obj layout: rows × towers × coeffs

    Output:
      rows[j].towers[t].coeffs = transpose of original tower matrix.
    """
    num_towers = int(export_obj["num_towers"])
    moduli = [int(q) for q in export_obj["moduli_u64"]]

    tower_transposes = []
    for t in range(num_towers):
        M = matrix_for_tower(export_obj, t, n)
        tower_transposes.append(transpose_matrix(M))

    rows = []
    for i in range(n):
        towers_coeffs = []
        for t in range(num_towers):
            towers_coeffs.append([int(v) for v in tower_transposes[t][i]])
        rows.append({
            "row_index": i,
            "towers_coeffs": towers_coeffs,
        })

    return {
        "num_rows": n,
        "num_towers": num_towers,
        "moduli_u64": moduli,
        "rows": rows,
        "zh": "debug-only logical-n transpose over the first n×n coefficient block",
    }


def import_1part_rows(rt, template_rows, transposed_export):
    out = []
    for i, row in enumerate(transposed_export["rows"]):
        out.append(rt.import_1part_coeff_u64(template_rows[i], row["towers_coeffs"]))
    return out


def logical_n_cmt_oracle(rt, bundle, n):
    """
    Debug-only logical-n C-MT oracle.

    It transposes c0 and c1 raw coefficient matrices independently
    over the first n×n coefficient block.

    This is not homomorphic and not secure.
    It is only a semantic oracle to validate downstream Algorithm 3 structure.
    """
    c0_export = normalize_export(
        rt.export_component_coeff_matrix_u64(bundle.rows, part=0, coeff_count=n)
    )
    c1_export = normalize_export(
        rt.export_component_coeff_matrix_u64(bundle.rows, part=1, coeff_count=n)
    )

    c0_T = logical_n_transpose_export(c0_export, n)
    c1_T = logical_n_transpose_export(c1_export, n)

    c0_rows = import_1part_rows(rt, bundle.rows, c0_T)
    c1_rows = import_1part_rows(rt, bundle.rows, c1_T)

    out_rows = [
        rt.assemble_2part_from_1parts_coeff_ct(c0_rows[i], c1_rows[i])
        for i in range(n)
    ]

    return TempRowwiseBundle(
        rows=out_rows,
        shape=(n, n),
        label="logical_n_cmt_oracle_output",
        ring_dim=bundle.ring_dim,
        batch_size=bundle.batch_size,
    ), {
        "c0_export_summary": {
            "num_rows": c0_export["num_rows"],
            "num_towers": c0_export["num_towers"],
            "ring_dim": c0_export["ring_dim"],
            "coeff_count": c0_export["coeff_count"],
            "row0_tower0_coeffs": c0_export["rows"][0]["towers"][0]["coeffs_u64"],
        },
        "c1_export_summary": {
            "num_rows": c1_export["num_rows"],
            "num_towers": c1_export["num_towers"],
            "ring_dim": c1_export["ring_dim"],
            "coeff_count": c1_export["coeff_count"],
            "row0_tower0_coeffs": c1_export["rows"][0]["towers"][0]["coeffs_u64"],
        },
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

    oracle_bundle, oracle_trace = logical_n_cmt_oracle(rt, bundle, n)
    oracle_dec = decrypt_rows_after_compress(rt, oracle_bundle.rows, logical_length=n)

    U_T = U.T.copy()

    return {
        "name": name,
        "zh": zh,
        "input_U": U.tolist(),
        "expected_input_decrypt": U.tolist(),
        "expected_logical_CMT_output_U_transpose": U_T.tolist(),
        "template_info": template_info,
        "input_decrypt": input_dec,
        "logical_cmt_oracle_trace": oracle_trace,
        "logical_cmt_oracle_decrypt": oracle_dec,
        "analysis": {
            "input_decrypt_exact": matrix_equal(input_dec["decoded"], U),
            "input_diff": diff_matrix(input_dec["decoded"], U),
            "logical_cmt_oracle_equals_U_transpose": matrix_equal(oracle_dec["decoded"], U_T),
            "logical_cmt_oracle_diff_to_U_transpose": diff_matrix(oracle_dec["decoded"], U_T),
            "expected_nonzero_positions": nonzero_positions(U_T.tolist()),
            "got_nonzero_positions": nonzero_positions(oracle_dec["decoded"]),
            "got_nonzero_count": None if nonzero_positions(oracle_dec["decoded"]) is None else len(nonzero_positions(oracle_dec["decoded"])),
            "expected_nonzero_count": len(nonzero_positions(U_T.tolist())),
        },
    }


def main():
    n = 4
    ring_dim = 1 << 14

    rng = np.random.default_rng(208400)

    cases = [
        {
            "name": "basis_E00",
            "U": one_hot(n, 0, 0),
            "zh": "logical-n C-MT oracle：E00 -> E00",
        },
        {
            "name": "basis_E01",
            "U": one_hot(n, 0, 1),
            "zh": "logical-n C-MT oracle：E01 -> E10",
        },
        {
            "name": "basis_E23",
            "U": one_hot(n, 2, 3),
            "zh": "logical-n C-MT oracle：E23 -> E32",
        },
        {
            "name": "random_small",
            "U": rng.integers(-2, 3, size=(n, n)).astype(np.int64),
            "zh": "logical-n C-MT oracle：随机小矩阵 -> 转置",
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
        "experiment": "wp4x7f_logical_n_cmt_oracle",
        "purpose": (
            "Implement a debug-only logical-n C-MT oracle by raw transposing the first n×n coefficient block. "
            "This validates that the downstream semantic target is reachable once C-MT matches the logical matrix dimension."
        ),
        "status": "debug_oracle_no_security_no_homomorphic_cmt",
        "n": n,
        "ring_dim": ring_dim,
        "warning": {
            "value": "This oracle reads/writes raw components and is not a secure homomorphic C-MT implementation.",
            "zh": "本脚本是 debug oracle，不是最终安全 C-MT。",
        },
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for c in cases:
            report["cases"].append(
                run_case(rt, c["U"], c["name"], c["zh"])
            )

    report["checks"] = {
        "transparent_input_rows_decrypt_exact（transparent 输入能正确解密回 U）": all(
            c["analysis"]["input_decrypt_exact"] for c in report["cases"]
        ),
        "logical_n_cmt_oracle_outputs_equal_transpose（logical-n C-MT oracle 输出等于 U^T）": all(
            c["analysis"]["logical_cmt_oracle_equals_U_transpose"] for c in report["cases"]
        ),
        "any_logical_n_cmt_case_passes（是否至少有一个 logical-n C-MT case 正确）": any(
            c["analysis"]["logical_cmt_oracle_equals_U_transpose"] for c in report["cases"]
        ),
        "failure_summary（失败时的非零数量诊断）": [
            {
                "name": c["name"],
                "expected_nonzero_count": c["analysis"]["expected_nonzero_count"],
                "got_nonzero_count": c["analysis"]["got_nonzero_count"],
                "expected_nonzero_positions": c["analysis"]["expected_nonzero_positions"],
                "got_nonzero_positions": c["analysis"]["got_nonzero_positions"],
            }
            for c in report["cases"]
            if not c["analysis"]["logical_cmt_oracle_equals_U_transpose"]
        ],
    }

    report["summary"] = {
        "if_oracle_passes": (
            "The logical-n transpose target is sound. The current homomorphic C-MT fails because its dimension contract does not match n=4. "
            "Next, use this oracle to validate the remaining Algorithm 3 algebra, then design a real logical-n C-MT or run tests with row_count matching ring dimension."
        ),
        "if_oracle_fails": (
            "Even raw coefficient transpose does not match decrypt semantics; inspect raw import/export/decrypt sign handling."
        ),
        "next_step": "WP4-X7g",
        "next_step_goal": (
            "Use logical-n C-MT oracle inside the full Algorithm 3 debug pipeline to verify PP-MM/line5/line6 algebra independently from homomorphic C-MT."
        ),
        "zh": "如果 X7f 通过，就证明失败根因是当前 C-MT 的维度合同；接下来用 oracle C-MT 验证后续 Algorithm 3 代数。",
    }

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4x7f_logical_n_cmt_oracle_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-X7f logical-n C-MT oracle")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
