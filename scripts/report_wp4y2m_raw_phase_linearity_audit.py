import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hegpt import HEConfig, HERuntime
from hegpt.ccmm_oracle_debug import (
    logical_n_cmt_oracle,
    raw_rns_ppmm,
    import_ppmm_rows,
    make_zero_1part_rows,
    make_temp_pair_bundle,
    extract_1part_rows,
    raw_add_2part_rows,
    normalize_export,
)

from report_wp4y2i_controlled_phase_aware_full_ccmm import (
    make_controlled_rowwise_bundle,
    phase_aware_logical_n_cmt_oracle,
    decrypt_rows_after_compress,
)


def one_hot(n, i, j, value=1):
    M = np.zeros((n, n), dtype=np.int64)
    M[i, j] = value
    return M


def center_mod(x, q):
    x = int(x) % int(q)
    if x > int(q) // 2:
        x -= int(q)
    return int(x)


def negacyclic_mul_prefix(a, b, q, out_count):
    """
    Compute prefix coefficients of a*b mod (X^N + 1), mod q.

    Optimized for sparse secret key b.

    Coefficient formula:
      (a*b)[k] = sum_j sign(k,j) * a[(k-j) mod N] * b[j]
      sign = +1 if j <= k, else -1

    This avoids O(N^2) full polynomial multiplication.
    """
    q = int(q)
    N = len(a)

    # Sparse representation of b/sk.
    b_sparse = [(j, int(v) % q) for j, v in enumerate(b) if int(v) % q != 0]

    out = []

    for k in range(out_count):
        acc = 0

        for j, bj in b_sparse:
            if j <= k:
                i = k - j
                acc += int(a[i]) * bj
            else:
                i = k - j + N
                acc -= int(a[i]) * bj

        out.append(acc % q)

    return out


def get_moduli_from_export(ex):
    if "moduli_u64" in ex:
        return [int(x) for x in ex["moduli_u64"]]
    if "moduli" in ex:
        return [int(x) for x in ex["moduli"]]
    raise KeyError("export has no moduli_u64/moduli")


def get_coeffs(ex, row_i, tower_i):
    return [int(x) for x in ex["rows"][row_i]["towers"][tower_i]["coeffs_u64"]]


def export_2part_full(rt, rows):
    c0 = normalize_export(rt.export_component_coeff_matrix_u64(rows, part=0, coeff_count=0))
    c1 = normalize_export(rt.export_component_coeff_matrix_u64(rows, part=1, coeff_count=0))
    return c0, c1


def raw_phase_prefix(rt, rows, logical_n, tower_index=0):
    """
    Compute phase prefix c0 + c1*sk directly from exported full-ring RNS coefficients.
    No compress/decrypt/rounding is used.
    """
    c0, c1 = export_2part_full(rt, rows)
    moduli = get_moduli_from_export(c0)
    q = int(moduli[tower_index])
    ring_dim = int(c0["ring_dim"])

    sk_towers = rt.export_secret_key_coeff_u64(ring_dim)
    sk = [int(x) for x in sk_towers[tower_index]]

    phases = []

    for row_i in range(len(rows)):
        c0_coeffs = get_coeffs(c0, row_i, tower_index)
        c1_coeffs = get_coeffs(c1, row_i, tower_index)

        prod = negacyclic_mul_prefix(c1_coeffs, sk, q, logical_n)

        phase = [
            (int(c0_coeffs[k]) + int(prod[k])) % q
            for k in range(logical_n)
        ]

        phases.append(phase)

    return {
        "tower_index": tower_index,
        "modulus": q,
        "ring_dim": ring_dim,
        "phase_prefix_u64": phases,
        "phase_prefix_centered_i64": [
            [center_mod(v, q) for v in row]
            for row in phases
        ],
    }


def phase_add_mod(phases, q):
    rows = len(phases[0])
    cols = len(phases[0][0])

    out = [[0 for _ in range(cols)] for _ in range(rows)]
    for P in phases:
        for i in range(rows):
            for j in range(cols):
                out[i][j] = (out[i][j] + int(P[i][j])) % int(q)
    return out


def phase_diff_mod(A, B, q):
    rows = len(A)
    cols = len(A[0])

    diff_u64 = []
    diff_centered = []

    for i in range(rows):
        ru = []
        rc = []
        for j in range(cols):
            d = (int(A[i][j]) - int(B[i][j])) % int(q)
            ru.append(d)
            rc.append(center_mod(d, q))
        diff_u64.append(ru)
        diff_centered.append(rc)

    max_abs = max(abs(v) for row in diff_centered for v in row)
    l1 = sum(abs(v) for row in diff_centered for v in row)

    return {
        "max_abs_centered": int(max_abs),
        "l1_centered": int(l1),
        "diff_centered": diff_centered,
    }


def matrix_diff_i64(A, B):
    D = np.array(A, dtype=np.int64) - np.array(B, dtype=np.int64)
    return {
        "max_abs_err": int(np.max(np.abs(D))),
        "l1_err": int(np.sum(np.abs(D))),
        "diff": D.tolist(),
    }


def assemble_pair_rows(rt, A_rows, B_rows, n):
    # paper pair (A,B) maps to OpenFHE c0=B,c1=A
    return [
        rt.assemble_2part_from_1parts_coeff_ct(B_rows[i], A_rows[i])
        for i in range(n)
    ]


def add_coeff_ct_rows(rt, left_rows, right_rows):
    return [
        rt.add_coeff_ct(left_rows[i], right_rows[i])
        for i in range(len(left_rows))
    ]


def full_ring_raw_add_2part_rows(rt, left_rows, right_rows):
    nrows = len(left_rows)

    L0 = normalize_export(rt.export_component_coeff_matrix_u64(left_rows, part=0, coeff_count=0))
    L1 = normalize_export(rt.export_component_coeff_matrix_u64(left_rows, part=1, coeff_count=0))
    R0 = normalize_export(rt.export_component_coeff_matrix_u64(right_rows, part=0, coeff_count=0))
    R1 = normalize_export(rt.export_component_coeff_matrix_u64(right_rows, part=1, coeff_count=0))

    moduli = get_moduli_from_export(L0)
    num_towers = int(L0["num_towers"])
    ring_dim = int(L0["ring_dim"])

    out_rows = []

    for i in range(nrows):
        imported_parts = []

        for part_L, part_R in [(L0, R0), (L1, R1)]:
            towers = []
            for t in range(num_towers):
                q = int(moduli[t])
                a = get_coeffs(part_L, i, t)
                b = get_coeffs(part_R, i, t)
                towers.append([(int(x) + int(y)) % q for x, y in zip(a, b)])

            imported_parts.append(rt.import_1part_coeff_u64(left_rows[i], towers))

        out_rows.append(rt.assemble_2part_from_1parts_coeff_ct(imported_parts[0], imported_parts[1]))

    return out_rows


def export_component_sources(rt, left_bundle, right_bundle, n):
    return {
        "A_L": normalize_export(rt.export_component_coeff_matrix_u64(left_bundle.rows, part=1, coeff_count=n)),
        "B_L": normalize_export(rt.export_component_coeff_matrix_u64(left_bundle.rows, part=0, coeff_count=n)),
        "A_R": normalize_export(rt.export_component_coeff_matrix_u64(right_bundle.rows, part=1, coeff_count=n)),
        "B_R": normalize_export(rt.export_component_coeff_matrix_u64(right_bundle.rows, part=0, coeff_count=n)),
    }


def temp_pair_cmt(rt, bundle, n, a_scalar, mode, label):
    if mode == "raw_component":
        return logical_n_cmt_oracle(rt, bundle, n, label=label)

    if mode == "phase_aware":
        return phase_aware_logical_n_cmt_oracle(
            rt=rt,
            bundle=bundle,
            n=n,
            a_scalar=a_scalar,
            label=label,
        )[0]

    raise ValueError(f"unknown temp C-MT mode: {mode}")


def build_terms(rt, temp_cmt_mode):
    n = 4
    a_scalar = 1

    U = one_hot(n, 0, 0)
    V = one_hot(n, 0, 0)

    U_bundle, _ = make_controlled_rowwise_bundle(rt, U.tolist(), a_scalar=a_scalar)
    V_bundle, _ = make_controlled_rowwise_bundle(rt, V.tolist(), a_scalar=a_scalar)

    U_cmt_bundle, _ = phase_aware_logical_n_cmt_oracle(
        rt=rt,
        bundle=U_bundle,
        n=n,
        a_scalar=a_scalar,
        label=f"phase_aware_CMT_U",
    )

    comps = export_component_sources(rt, U_cmt_bundle, V_bundle, n)

    M00 = raw_rns_ppmm(comps["A_L"], comps["A_R"], n)
    M01 = raw_rns_ppmm(comps["A_L"], comps["B_R"], n)
    M10 = raw_rns_ppmm(comps["B_L"], comps["A_R"], n)
    M11 = raw_rns_ppmm(comps["B_L"], comps["B_R"], n)

    M00_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M00)
    M01_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M01)
    M10_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M10)
    M11_rows = import_ppmm_rows(rt, U_cmt_bundle.rows, M11)

    zero_rows = make_zero_1part_rows(rt, U_cmt_bundle.rows, num_towers=M00["num_towers"])

    T01_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M01_rows,
        B_rows=zero_rows,
        n=n,
        label="T01=(M01,0)",
        template_bundle=U_bundle,
    )
    T01_cmt = temp_pair_cmt(
        rt,
        T01_bundle,
        n=n,
        a_scalar=a_scalar,
        mode=temp_cmt_mode,
        label=f"{temp_cmt_mode}_CMT_T01",
    )

    T00_bundle = make_temp_pair_bundle(
        rt,
        A_rows=M00_rows,
        B_rows=zero_rows,
        n=n,
        label="T00=(M00,0)",
        template_bundle=U_bundle,
    )
    T00_cmt = temp_pair_cmt(
        rt,
        T00_bundle,
        n=n,
        a_scalar=a_scalar,
        mode=temp_cmt_mode,
        label=f"{temp_cmt_mode}_CMT_T00",
    )

    Bhat_rows, _ = extract_1part_rows(rt, T00_cmt.rows, part=0, coeff_count=0)
    Ahat_rows, _ = extract_1part_rows(rt, T00_cmt.rows, part=1, coeff_count=0)

    line5_rows = []
    for i in range(n):
        ks_input = rt.assemble_3part_from_1parts_coeff_ct(
            zero_rows[i],
            zero_rows[i],
            Ahat_rows[i],
        )

        ks_output = rt.relinearize_coeff_ct(ks_input)

        bhat_term = rt.assemble_2part_from_1parts_coeff_ct(
            zero_rows[i],
            Bhat_rows[i],
        )

        line5_rows.append(rt.add_coeff_ct(ks_output, bhat_term))

    M10_M11_rows = assemble_pair_rows(rt, M10_rows, M11_rows, n)

    return {
        "n": n,
        "U": U,
        "V": V,
        "line5": line5_rows,
        "T01": T01_cmt.rows,
        "M10_M11": M10_M11_rows,
    }


def decrypt_i64(rt, rows, n):
    return decrypt_rows_after_compress(rt, rows, logical_length=n)["decoded"]


def main():
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
        "experiment": "wp4y2m_raw_phase_linearity_audit",
        "purpose": (
            "Audit ciphertext addition at raw RLWE phase level: phase(c0,c1)=c0+c1*sk mod q. "
            "This avoids the invalid assumption that separate compress/decrypt results are additively linear."
        ),
        "status": "debug_raw_phase_audit",
        "cases": [],
    }

    with HERuntime(cfg, rotation_steps=[]) as rt:
        for mode in ["raw_component", "phase_aware"]:
            terms = build_terms(rt, temp_cmt_mode=mode)
            n = terms["n"]

            line5 = terms["line5"]
            T01 = terms["T01"]
            M10_M11 = terms["M10_M11"]

            prefix_tmp = raw_add_2part_rows(rt, line5, T01, n)
            prefix_final = raw_add_2part_rows(rt, prefix_tmp, M10_M11, n)

            full_tmp = full_ring_raw_add_2part_rows(rt, line5, T01)
            full_final = full_ring_raw_add_2part_rows(rt, full_tmp, M10_M11)

            native_tmp = add_coeff_ct_rows(rt, line5, T01)
            native_final = add_coeff_ct_rows(rt, native_tmp, M10_M11)

            print(f"[Y2m] computing raw phases for mode={mode}", flush=True)
            P_line5 = raw_phase_prefix(rt, line5, n)
            P_T01 = raw_phase_prefix(rt, T01, n)
            P_M10M11 = raw_phase_prefix(rt, M10_M11, n)

            q = int(P_line5["modulus"])

            P_sum_u64 = phase_add_mod(
                [
                    P_line5["phase_prefix_u64"],
                    P_T01["phase_prefix_u64"],
                    P_M10M11["phase_prefix_u64"],
                ],
                q=q,
            )

            P_prefix_final = raw_phase_prefix(rt, prefix_final, n)
            P_full_final = raw_phase_prefix(rt, full_final, n)
            P_native_final = raw_phase_prefix(rt, native_final, n)

            dec_line5 = decrypt_i64(rt, line5, n)
            dec_T01 = decrypt_i64(rt, T01, n)
            dec_M10M11 = decrypt_i64(rt, M10_M11, n)
            dec_prefix_final = decrypt_i64(rt, prefix_final, n)
            dec_full_final = decrypt_i64(rt, full_final, n)
            dec_native_final = decrypt_i64(rt, native_final, n)

            decoded_sum = (
                np.array(dec_line5, dtype=np.int64)
                + np.array(dec_T01, dtype=np.int64)
                + np.array(dec_M10M11, dtype=np.int64)
            ).tolist()

            case = {
                "temp_cmt_mode": mode,
                "phase_tower": P_line5["tower_index"],
                "phase_modulus": q,
                "phase_prefix_centered": {
                    "line5": P_line5["phase_prefix_centered_i64"],
                    "T01": P_T01["phase_prefix_centered_i64"],
                    "M10_M11": P_M10M11["phase_prefix_centered_i64"],
                    "sum": [[center_mod(v, q) for v in row] for row in P_sum_u64],
                    "prefix_final": P_prefix_final["phase_prefix_centered_i64"],
                    "full_final": P_full_final["phase_prefix_centered_i64"],
                    "native_final": P_native_final["phase_prefix_centered_i64"],
                },
                "phase_linearity": {
                    "prefix_final_vs_sum": phase_diff_mod(P_prefix_final["phase_prefix_u64"], P_sum_u64, q),
                    "full_final_vs_sum": phase_diff_mod(P_full_final["phase_prefix_u64"], P_sum_u64, q),
                    "native_final_vs_sum": phase_diff_mod(P_native_final["phase_prefix_u64"], P_sum_u64, q),
                },
                "decrypt_after_compress_i64": {
                    "line5": dec_line5,
                    "T01": dec_T01,
                    "M10_M11": dec_M10M11,
                    "decoded_sum": decoded_sum,
                    "prefix_final": dec_prefix_final,
                    "full_final": dec_full_final,
                    "native_final": dec_native_final,
                },
                "decrypt_vs_raw_phase_centered": {
                    "line5": matrix_diff_i64(dec_line5, P_line5["phase_prefix_centered_i64"]),
                    "T01": matrix_diff_i64(dec_T01, P_T01["phase_prefix_centered_i64"]),
                    "M10_M11": matrix_diff_i64(dec_M10M11, P_M10M11["phase_prefix_centered_i64"]),
                    "prefix_final": matrix_diff_i64(dec_prefix_final, P_prefix_final["phase_prefix_centered_i64"]),
                    "full_final": matrix_diff_i64(dec_full_final, P_full_final["phase_prefix_centered_i64"]),
                    "native_final": matrix_diff_i64(dec_native_final, P_native_final["phase_prefix_centered_i64"]),
                },
            }

            report["cases"].append(case)

    report["checks"] = {
        "prefix_raw_add_phase_linear_all_modes（prefix raw add 的 raw phase 是否线性）": all(
            c["phase_linearity"]["prefix_final_vs_sum"]["max_abs_centered"] == 0
            for c in report["cases"]
        ),
        "full_ring_raw_add_phase_linear_all_modes（full-ring raw add 的 raw phase 是否线性）": all(
            c["phase_linearity"]["full_final_vs_sum"]["max_abs_centered"] == 0
            for c in report["cases"]
        ),
        "native_add_phase_linear_all_modes（native add_coeff_ct 的 raw phase 是否线性）": all(
            c["phase_linearity"]["native_final_vs_sum"]["max_abs_centered"] == 0
            for c in report["cases"]
        ),
        "decrypt_matches_raw_phase_all_terms（compress/decrypt 输出是否等于 raw phase centered prefix）": all(
            all(v["max_abs_err"] == 0 for v in c["decrypt_vs_raw_phase_centered"].values())
            for c in report["cases"]
        ),
        "mode_summaries（各模式 raw phase 线性摘要）": [
            {
                "temp_cmt_mode": c["temp_cmt_mode"],
                "prefix_phase_max_abs_vs_sum": c["phase_linearity"]["prefix_final_vs_sum"]["max_abs_centered"],
                "full_ring_phase_max_abs_vs_sum": c["phase_linearity"]["full_final_vs_sum"]["max_abs_centered"],
                "native_phase_max_abs_vs_sum": c["phase_linearity"]["native_final_vs_sum"]["max_abs_centered"],
                "max_decrypt_vs_raw_phase": max(
                    v["max_abs_err"] for v in c["decrypt_vs_raw_phase_centered"].values()
                ),
            }
            for c in report["cases"]
        ],
    }

    if (
        report["checks"]["full_ring_raw_add_phase_linear_all_modes（full-ring raw add 的 raw phase 是否线性）"]
        or report["checks"]["native_add_phase_linear_all_modes（native add_coeff_ct 的 raw phase 是否线性）"]
    ):
        decision = {
            "value": "raw_phase_add_is_linear_decode_sum_was_invalid_diagnostic",
            "next_step": "WP4-Y2n",
            "next_goal": "Stop comparing separately decoded terms. Use raw phase trace to diagnose Algorithm 3 formula/orientation.",
            "zh": "raw phase 层加法线性成立；之前逐项 decrypt 后再相加的诊断不成立。下一步用 raw phase 追踪公式。",
        }
    else:
        decision = {
            "value": "raw_phase_add_not_linear",
            "next_step": "WP4-Y2n",
            "next_goal": "Inspect export/import/add at component coefficient level; ciphertext addition is not preserving raw phase.",
            "zh": "raw phase 层加法仍不线性，需要查 component export/import/add。",
        }

    report["global_decision"] = decision

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp4y2m_raw_phase_linearity_audit_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("WP4-Y2m raw RLWE phase linearity audit")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 100)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
