import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.ccmm_oracle_debug import (
    TempRowwiseBundle,
    normalize_export,
    logical_n_cmt_oracle,
)


def safe_call(name, fn):
    try:
        return {"ok": True, "value": fn()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def signed_to_mod(v, q):
    return int(v) % int(q)


def compare_matrix(decoded, ref):
    if decoded is None or any(row is None for row in decoded):
        return {
            "all_rows_decrypted": False,
            "exact": False,
            "max_abs_err": None,
            "l1_err": None,
            "diff": None,
        }

    G = np.array(decoded, dtype=np.int64)
    R = np.array(ref, dtype=np.int64)
    D = G - R

    return {
        "all_rows_decrypted": True,
        "exact": bool(np.array_equal(G, R)),
        "max_abs_err": int(np.max(np.abs(D))),
        "l1_err": int(np.sum(np.abs(D))),
        "diff": D.tolist(),
    }


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


def make_constant_c1_towers(a_scalar, n, moduli):
    towers = []
    for q in moduli:
        q = int(q)
        coeffs = [0 for _ in range(n)]
        coeffs[0] = int(a_scalar) % q
        towers.append(coeffs)
    return towers


def make_controlled_rlwe_coeff_row_ct(rt, template_ct, row, sk_towers, moduli, a_scalar):
    """
    Controlled debug RLWE coefficient row.

    Verified by Y2f:
      phase = c0 + c1*sk
      c1 = a
      c0 = m - a*sk
    """
    n = len(row)

    c1_towers = make_constant_c1_towers(a_scalar, n, moduli)

    c0_towers = []
    for t, q in enumerate(moduli):
        q = int(q)
        sk_coeffs = sk_towers[t][:n]

        c0_coeffs = []
        for i in range(n):
            m_i = signed_to_mod(row[i], q)
            ask_i = (int(a_scalar) * int(sk_coeffs[i])) % q
            c0_i = (m_i - ask_i) % q
            c0_coeffs.append(c0_i)

        c0_towers.append(c0_coeffs)

    c0 = rt.import_1part_coeff_u64(template_ct, c0_towers)
    c1 = rt.import_1part_coeff_u64(template_ct, c1_towers)

    return rt.assemble_2part_from_1parts_coeff_ct(c0, c1), {
        "a_scalar": int(a_scalar),
        "c0_tower0_coeffs": c0_towers[0],
        "c1_tower0_coeffs": c1_towers[0],
        "sk_tower0_coeffs": sk_towers[0][:n],
    }


def make_controlled_rowwise_bundle(rt, matrix, a_scalar=1, ring_dim=1 << 14, batch_size=8):
    n = len(matrix)

    template_ct = rt.encrypt_coeff_row_i64([0 for _ in range(n)])
    template_export = normalize_export(
        rt.export_component_coeff_matrix_u64([template_ct], part=0, coeff_count=1)
    )

    moduli = [int(q) for q in template_export["moduli_u64"]]
    sk_towers = rt.export_secret_key_coeff_u64(n)

    rows = []
    traces = []

    for row in matrix:
        ct, trace = make_controlled_rlwe_coeff_row_ct(
            rt=rt,
            template_ct=template_ct,
            row=[int(v) for v in row],
            sk_towers=sk_towers,
            moduli=moduli,
            a_scalar=a_scalar,
        )
        rows.append(ct)
        traces.append(trace)

    bundle = TempRowwiseBundle(
        rows=rows,
        shape=(n, n),
        label=f"controlled_rlwe_rowwise_a{a_scalar}",
        ring_dim=ring_dim,
        batch_size=batch_size,
    )

    info = {
        "a_scalar": int(a_scalar),
        "template_ring_dim": template_export["ring_dim"],
        "template_num_towers": template_export["num_towers"],
        "moduli_u64": moduli,
        "sk_num_towers": len(sk_towers),
        "sk_coeff_count": len(sk_towers[0]) if sk_towers else 0,
        "sk_tower0_coeffs": sk_towers[0] if sk_towers else [],
        "row0_build_trace": traces[0] if traces else None,
    }

    return bundle, info


def export_c0_c1_prefix(rt, rows, n):
    c0 = normalize_export(rt.export_component_coeff_matrix_u64(rows, part=0, coeff_count=n))
    c1 = normalize_export(rt.export_component_coeff_matrix_u64(rows, part=1, coeff_count=n))

    return {
        "c0_row0_tower0": c0["rows"][0]["towers"][0]["coeffs_u64"],
        "c1_row0_tower0": c1["rows"][0]["towers"][0]["coeffs_u64"],
        "c0_num_towers": c0["num_towers"],
        "c1_num_towers": c1["num_towers"],
    }


def one_hot(n, i, j, value=1):
    M = np.zeros((n, n), dtype=np.int64)
    M[i, j] = value
    return M


def run_case(rt, name, U, a_scalar):
    n = U.shape[0]

    bundle, build_info = make_controlled_rowwise_bundle(rt, U.tolist(), a_scalar=a_scalar)

    input_dec = decrypt_rows_after_compress(rt, bundle.rows, logical_length=n)

    cmt_bundle = logical_n_cmt_oracle(rt, bundle, n, label=f"logical_n_CMT_controlled_a{a_scalar}")
    cmt_dec = decrypt_rows_after_compress(rt, cmt_bundle.rows, logical_length=n)

    U_T = U.T.copy()

    return {
        "name": name,
        "a_scalar": int(a_scalar),
        "input_U": U.tolist(),
        "expected_input_decrypt": U.tolist(),
        "expected_cmt_output_U_transpose": U_T.tolist(),
        "build_info": build_info,
        "input_component_prefix": export_c0_c1_prefix(rt, bundle.rows, n),
        "cmt_component_prefix": export_c0_c1_prefix(rt, cmt_bundle.rows, n),
        "input_decrypt": input_dec,
        "cmt_decrypt": cmt_dec,
        "analysis": {
            "input_decrypt_exact": compare_matrix(input_dec["decoded"], U),
            "cmt_decrypt_equals_U_transpose": compare_matrix(cmt_dec["decoded"], U_T),
            "expected_nonzero_positions": nonzero_positions(U_T.tolist()),
            "got_nonzero_positions": nonzero_positions(cmt_dec["decoded"]),
        },
    }


def main():
    n = 4
    rng = np.random.default_rng(209100)

    cases = [
        {
            "name": "basis_E00",
            "U": one_hot(n, 0, 0),
        },
        {
            "name": "basis_E01",
            "U": one_hot(n, 0, 1),
        },
        {
            "name": "basis_E23",
            "U": one_hot(n, 2, 3),
        },
        {
            "name": "random_small",
            "U": rng.integers(-2, 3, size=(n, n)).astype(np.int64),
        },
    ]

    a_scalars = [1, 3, 7]

    cfg = HEConfig(
        ring_dim=1 << 14,
        multiplicative_depth=2,
        first_mod_size=60,
        scaling_mod_size=50,
        num_large_digits=2,
        batch_size=8,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    report = {
        "experiment": "wp4y2g_controlled_rlwe_rowwise_cmt_probe",
        "purpose": (
            "Wrap Y2f controlled RLWE coefficient rows into rowwise bundles and test whether "
            "logical-n raw C-MT oracle still preserves matrix transpose for non-transparent c1!=0 inputs."
        ),
        "status": "controlled_rlwe_rowwise_probe",
        "n": n,
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for a_scalar in a_scalars:
            for case in cases:
                report["cases"].append(
                    run_case(rt, case["name"], case["U"], a_scalar)
                )

    report["checks"] = {
        "all_controlled_inputs_decrypt_exact（所有 controlled RLWE 输入 bundle 都能解密回 U）": all(
            c["analysis"]["input_decrypt_exact"]["exact"]
            for c in report["cases"]
        ),
        "all_logical_n_cmt_outputs_equal_transpose（controlled 输入下 logical-n C-MT oracle 全部等于 U^T）": all(
            c["analysis"]["cmt_decrypt_equals_U_transpose"]["exact"]
            for c in report["cases"]
        ),
        "any_logical_n_cmt_controlled_case_passes（至少一个 controlled C-MT case 正确）": any(
            c["analysis"]["cmt_decrypt_equals_U_transpose"]["exact"]
            for c in report["cases"]
        ),
        "failed_cmt_cases（C-MT 失败 case 摘要）": [
            {
                "name": c["name"],
                "a_scalar": c["a_scalar"],
                "max_abs_err": c["analysis"]["cmt_decrypt_equals_U_transpose"]["max_abs_err"],
                "l1_err": c["analysis"]["cmt_decrypt_equals_U_transpose"]["l1_err"],
                "expected_nonzero_positions": c["analysis"]["expected_nonzero_positions"],
                "got_nonzero_positions": c["analysis"]["got_nonzero_positions"],
            }
            for c in report["cases"]
            if not c["analysis"]["cmt_decrypt_equals_U_transpose"]["exact"]
        ],
    }

    if report["checks"]["all_logical_n_cmt_outputs_equal_transpose（controlled 输入下 logical-n C-MT oracle 全部等于 U^T）"]:
        decision = {
            "value": "controlled_rlwe_logical_n_cmt_oracle_works",
            "next_step": "WP4-Y2h",
            "next_goal": "Use controlled RLWE rowwise bundles in full oracle CCMM pipeline.",
            "zh": "controlled RLWE 输入也能通过 logical-n C-MT oracle，可进入完整 oracle CCMM。",
        }
    else:
        decision = {
            "value": "controlled_rlwe_input_works_but_logical_n_raw_cmt_oracle_fails",
            "next_step": "WP4-Y2h",
            "next_goal": (
                "Design real logical-n C-MT/key-switch semantics for non-transparent RLWE c1!=0 inputs. "
                "The raw component transpose oracle is only valid for transparent c1=0 diagnostics."
            ),
            "zh": "controlled 输入自身正确，但 raw 逐 component 转置 oracle 对非透明 RLWE 不成立；下一步处理 C-MT/key-switch 语义。",
        }

    report["global_decision"] = decision

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4y2g_controlled_rlwe_rowwise_cmt_probe_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Y2g controlled RLWE rowwise C-MT probe")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
