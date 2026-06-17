"""
Debug/oracle helpers for Park-2025 CC-MM experiments.

Important:
  This module is NOT a secure homomorphic implementation.

It provides reusable logical-n / raw-RNS helpers used after WP4-X7k:
  - raw RNS component addition for 2-part coefficient ciphertext rows
  - logical-n C-MT oracle over the first n×n coefficient block
  - transparent coefficient row construction c0=row,c1=0

Known limitations:
  - transparent ciphertexts are not encrypted
  - logical_n_cmt_oracle reads/writes raw components
  - raw_add_2part_rows only preserves the first n coefficients
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class TempRowwiseBundle:
    rows: list
    shape: tuple
    label: str
    ring_dim: int = 1 << 14
    batch_size: int = 8
    orientation: str = "rowwise"


def normalize_export(x: Any) -> Dict[str, Any]:
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def signed_to_mod(v: int, q: int) -> int:
    return int(v) % int(q)


def make_zero_1part(rt, template_ct, num_towers: int):
    return rt.import_1part_coeff_u64(template_ct, [[] for _ in range(num_towers)])


def make_zero_1part_rows(rt, template_rows: Sequence, num_towers: int) -> list:
    return [make_zero_1part(rt, tmpl, num_towers) for tmpl in template_rows]


def matrix_for_tower(export_obj: Dict[str, Any], tower_index: int, n: int) -> List[List[int]]:
    out = []
    for row in export_obj["rows"]:
        out.append([int(v) for v in row["towers"][tower_index]["coeffs_u64"][:n]])
    return out


def transpose_matrix(M: List[List[int]]) -> List[List[int]]:
    n = len(M)
    return [[M[i][j] for i in range(n)] for j in range(n)]


def logical_n_transpose_export(export_obj: Dict[str, Any], n: int) -> Dict[str, Any]:
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
        rows.append({"row_index": i, "towers_coeffs": towers_coeffs})

    return {
        "num_rows": n,
        "num_towers": num_towers,
        "moduli_u64": moduli,
        "rows": rows,
    }


def import_1part_rows(rt, template_rows: Sequence, exported_rows: Dict[str, Any]) -> list:
    out = []
    for i, row in enumerate(exported_rows["rows"]):
        out.append(rt.import_1part_coeff_u64(template_rows[i], row["towers_coeffs"]))
    return out


def logical_n_cmt_oracle(rt, bundle: TempRowwiseBundle, n: int, label: str = "logical_n_cmt_oracle"):
    """
    Debug-only logical-n C-MT oracle.

    It transposes c0 and c1 raw coefficient matrices independently over
    the first n×n coefficient block.
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
        label=label,
        ring_dim=bundle.ring_dim,
        batch_size=bundle.batch_size,
    )


def make_transparent_coeff_row_ct(rt, template_ct, row: Sequence[int], moduli: Sequence[int]):
    """
    Debug-only transparent coefficient ciphertext:
      c0 = row
      c1 = 0

    This is not secure encryption.
    """
    towers_coeffs = []
    for q in moduli:
        towers_coeffs.append([signed_to_mod(v, q) for v in row])

    c0 = rt.import_1part_coeff_u64(template_ct, towers_coeffs)
    c1 = make_zero_1part(rt, template_ct, len(moduli))

    return rt.assemble_2part_from_1parts_coeff_ct(c0, c1)


def make_transparent_rowwise_bundle(rt, matrix: Sequence[Sequence[int]], ring_dim: int = 1 << 14, batch_size: int = 8):
    n = len(matrix)

    template = rt.encrypt_coeff_row_i64([0 for _ in range(n)])
    template_export = normalize_export(
        rt.export_component_coeff_matrix_u64([template], part=0, coeff_count=1)
    )
    moduli = [int(q) for q in template_export["moduli_u64"]]

    rows = [
        make_transparent_coeff_row_ct(rt, template, [int(v) for v in row], moduli)
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


def matmul_mod(A: List[List[int]], B: List[List[int]], q: int) -> List[List[int]]:
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


def raw_rns_ppmm(left_export: Dict[str, Any], right_export: Dict[str, Any], n: int) -> Dict[str, Any]:
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
        rows.append({"row_index": i, "towers_coeffs": towers_coeffs})

    return {
        "num_rows": n,
        "num_towers": num_towers,
        "moduli_u64": moduli,
        "rows": rows,
        "tower_mats_prefix": tower_mats,
    }


def raw_add_exported_parts(part_a: Dict[str, Any], part_b: Dict[str, Any], n: int) -> Dict[str, Any]:
    if part_a["num_rows"] != part_b["num_rows"]:
        raise ValueError("raw_add_exported_parts: row count mismatch")
    if part_a["num_towers"] != part_b["num_towers"]:
        raise ValueError("raw_add_exported_parts: tower count mismatch")
    if part_a["moduli_u64"] != part_b["moduli_u64"]:
        raise ValueError("raw_add_exported_parts: moduli mismatch")

    rows = []
    moduli = [int(q) for q in part_a["moduli_u64"]]

    for i in range(part_a["num_rows"]):
        towers_coeffs = []
        for t, q in enumerate(moduli):
            ca = part_a["rows"][i]["towers"][t]["coeffs_u64"][:n]
            cb = part_b["rows"][i]["towers"][t]["coeffs_u64"][:n]
            towers_coeffs.append([(int(x) + int(y)) % q for x, y in zip(ca, cb)])
        rows.append({"row_index": i, "towers_coeffs": towers_coeffs})

    return {
        "num_rows": part_a["num_rows"],
        "num_towers": part_a["num_towers"],
        "moduli_u64": moduli,
        "rows": rows,
    }


def _export_moduli_u64(export_obj: Dict[str, Any]) -> List[int]:
    if "moduli_u64" in export_obj:
        return [int(q) for q in export_obj["moduli_u64"]]
    if "moduli" in export_obj:
        return [int(q) for q in export_obj["moduli"]]
    raise KeyError("export object has no moduli_u64/moduli")


def _tower_coeffs(export_obj: Dict[str, Any], row_i: int, tower_i: int) -> List[int]:
    return [int(x) for x in export_obj["rows"][row_i]["towers"][tower_i]["coeffs_u64"]]


def raw_add_2part_rows(rt, rows_a: Sequence, rows_b: Sequence, n: int | None = None) -> list:
    """
    Full-ring raw RNS add for two-component coefficient rows.

    Important:
      Older WP4 debug code only added the first logical n coefficients.
      Y2m proved that prefix-only add is not phase-linear when c1/sk convolution
      sees the full ring. Therefore this function now always exports/imports the
      full ring_dim for both c0 and c1.

    The n argument is kept for backward compatibility but ignored.
    """
    if len(rows_a) != len(rows_b):
        raise ValueError("raw_add_2part_rows: row count mismatch")

    a0 = normalize_export(rt.export_component_coeff_matrix_u64(rows_a, part=0, coeff_count=0))
    a1 = normalize_export(rt.export_component_coeff_matrix_u64(rows_a, part=1, coeff_count=0))
    b0 = normalize_export(rt.export_component_coeff_matrix_u64(rows_b, part=0, coeff_count=0))
    b1 = normalize_export(rt.export_component_coeff_matrix_u64(rows_b, part=1, coeff_count=0))

    moduli = _export_moduli_u64(a0)

    for obj_name, obj in [("a1", a1), ("b0", b0), ("b1", b1)]:
        if _export_moduli_u64(obj) != moduli:
            raise ValueError(f"raw_add_2part_rows: modulus mismatch for {obj_name}")
        if int(obj["num_towers"]) != int(a0["num_towers"]):
            raise ValueError(f"raw_add_2part_rows: tower count mismatch for {obj_name}")
        if int(obj["ring_dim"]) != int(a0["ring_dim"]):
            raise ValueError(f"raw_add_2part_rows: ring_dim mismatch for {obj_name}")

    num_rows = len(rows_a)
    num_towers = int(a0["num_towers"])
    ring_dim = int(a0["ring_dim"])

    out_rows = []

    for i in range(num_rows):
        imported_parts = []

        for part_left, part_right in [(a0, b0), (a1, b1)]:
            towers = []

            for t in range(num_towers):
                q = int(moduli[t])
                left = _tower_coeffs(part_left, i, t)
                right = _tower_coeffs(part_right, i, t)

                if len(left) != ring_dim or len(right) != ring_dim:
                    raise RuntimeError(
                        "raw_add_2part_rows: full-ring export length mismatch "
                        f"row={i}, tower={t}, len(left)={len(left)}, len(right)={len(right)}, ring_dim={ring_dim}"
                    )

                towers.append([(int(x) + int(y)) % q for x, y in zip(left, right)])

            imported_parts.append(rt.import_1part_coeff_u64(rows_a[i], towers))

        out_rows.append(
            rt.assemble_2part_from_1parts_coeff_ct(imported_parts[0], imported_parts[1])
        )

    return out_rows



def import_ppmm_rows(rt, template_rows: Sequence, ppmm_result: Dict[str, Any]) -> list:
    return import_1part_rows(rt, template_rows, ppmm_result)


def make_temp_pair_bundle(rt, A_rows: Sequence, B_rows: Sequence, n: int, label: str, template_bundle: TempRowwiseBundle):
    """
    Paper pair is (A,B).
    OpenFHE component order is c0=B,c1=A.
    """
    rows = [
        rt.assemble_2part_from_1parts_coeff_ct(B_rows[i], A_rows[i])
        for i in range(n)
    ]

    return TempRowwiseBundle(
        rows=rows,
        shape=(n, n),
        label=label,
        ring_dim=template_bundle.ring_dim,
        batch_size=template_bundle.batch_size,
    )


def extract_1part_rows(rt, source_rows: Sequence, part: int, coeff_count: int = 0):
    exported = normalize_export(
        rt.export_component_coeff_matrix_u64(source_rows, part=part, coeff_count=coeff_count)
    )

    out = []
    for i, row in enumerate(exported["rows"]):
        towers_coeffs = [tower["coeffs_u64"] for tower in row["towers"]]
        out.append(rt.import_1part_coeff_u64(source_rows[i], towers_coeffs))

    return out, {
        "part": part,
        "num_rows": exported["num_rows"],
        "num_towers": exported["num_towers"],
        "ring_dim": exported["ring_dim"],
        "coeff_count": exported["coeff_count"],
        "row0_tower0_coeff_head": exported["rows"][0]["towers"][0]["coeffs_u64"][:8],
    }
