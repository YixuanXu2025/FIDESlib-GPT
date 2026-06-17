from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass
class RowwiseCoeffCiphertextMatrix:
    """
    Real Park-2025 row-wise coefficient ciphertext matrix bundle.

    Semantic mapping:
      rows[i] is a real OpenFHE/FIDESlib ciphertext encrypting matrix row i.

      B[i] = rows[i].GetElements()[0] = c0
      A[i] = rows[i].GetElements()[1] = c1

    Paper-style row-wise form:
      A · sk + B ≈ M

    This object stores ciphertext handles, not plaintext-readable coefficients.
    """

    rows: List[Any]
    shape: Tuple[int, int]
    ring_dim: int
    encoding_type: int
    num_parts: int
    num_towers: int
    orientation: str = "rowwise"
    mapping: str = "B=c0/GetElements()[0], A=c1/GetElements()[1]"


def make_rowwise_coeff_ciphertext_matrix(rt, M_i64) -> RowwiseCoeffCiphertextMatrix:
    """
    Encrypt an integer matrix row-by-row using coefficient encoding
    and return a real row-wise ciphertext matrix bundle.
    """
    rows_plain = [list(map(int, row)) for row in M_i64]
    nrows = len(rows_plain)
    if nrows == 0:
        raise ValueError("empty matrix")

    ncols = len(rows_plain[0])
    if any(len(r) != ncols for r in rows_plain):
        raise ValueError("ragged matrix")

    rows = [rt.encrypt_coeff_row_i64(r) for r in rows_plain]
    info = rt.inspect_coeff_row_component_matrix(rows, coeff_sample=4)

    checks = {
        "all_rows_are_coef_encoding_type_1": all(
            rr.get("encoding_type") == 1 for rr in info.get("rows", [])
        ),
        "all_rows_have_two_parts": all(
            rr.get("num_parts") == 2 for rr in info.get("rows", [])
        ),
        "consistent_shape": bool(info.get("consistent_shape")),
    }

    if not all(checks.values()):
        raise RuntimeError(f"invalid row-wise coefficient ciphertext matrix: {checks}")

    return RowwiseCoeffCiphertextMatrix(
        rows=rows,
        shape=(nrows, ncols),
        ring_dim=int(info.get("expected_ring_dim")),
        encoding_type=1,
        num_parts=int(info.get("expected_parts")),
        num_towers=int(info.get("expected_towers")),
    )


def describe_rowwise_coeff_ciphertext_matrix(bundle: RowwiseCoeffCiphertextMatrix):
    return {
        "shape": list(bundle.shape),
        "ring_dim": bundle.ring_dim,
        "encoding_type": {
            "value": bundle.encoding_type,
            "zh": "OpenFHE coefficient encoding 类型；1 表示 coefficient-style encoding",
        },
        "num_parts": {
            "value": bundle.num_parts,
            "zh": "每个 RLWE ciphertext 的 component 数量；应为 2，即 c0 和 c1",
        },
        "num_towers": {
            "value": bundle.num_towers,
            "zh": "每个 DCRTPoly component 的 RNS tower 数量",
        },
        "orientation": {
            "value": bundle.orientation,
            "zh": "当前 bundle 是 row-wise，每个 ciphertext 加密矩阵的一行",
        },
        "mapping": {
            "value": bundle.mapping,
            "zh": "论文映射：B 是 c0，A 是 c1，因此 A·sk+B 近似表示明文矩阵",
        },
    }


def tweak_algorithm1_rowwise(rt, bundle: RowwiseCoeffCiphertextMatrix):
    """
    Real Park-2025 Algorithm 1 Tweak over coefficient ciphertext rows.

    Input:
      bundle.rows = [ct_0, ..., ct_{n-1}]
      each ct_i encrypts one coefficient row.

    Output:
      tweaked[j] = sum_i X^(2*i*j*N/n) * ct_i

    Here:
      N = ring_dim
      n = logical matrix dimension / number of rows

    This function only combines ciphertext components using:
      - monomial_mul_coeff_ct
      - add_coeff_ct

    It does not decrypt and does not perform C-MT by itself.
    """
    nrows, ncols = bundle.shape
    if nrows != ncols:
        raise ValueError(f"Algorithm 1 Tweak currently expects square matrix, got {bundle.shape}")

    n = nrows
    ring_dim = int(bundle.ring_dim)

    if ring_dim % n != 0:
        raise ValueError(f"ring_dim={ring_dim} must be divisible by n={n}")

    out = []
    trace = []

    for j in range(n):
        acc = None
        terms = []

        for i, ct_i in enumerate(bundle.rows):
            exp = (2 * i * j * ring_dim) // n

            term = rt.monomial_mul_coeff_ct(ct_i, exp)

            if acc is None:
                acc = term
            else:
                acc = rt.add_coeff_ct(acc, term)

            terms.append({
                "i": i,
                "j": j,
                "exp": exp,
                "formula": "2*i*j*ring_dim/n",
                "zh": f"第 j={j} 个输出中，第 i={i} 行乘以 X^{exp}",
            })

        out.append(acc)
        trace.append({
            "j": j,
            "terms": terms,
            "zh": f"输出 ct_prime[{j}] 是所有 i 项 X^(2*i*j*N/n)·ct_i 的和",
        })

    return out, trace


def describe_tweak_output(rt, tweaked_rows, coeff_sample: int = 4):
    """
    Compact inspection for Tweak output rows.
    """
    info = rt.inspect_coeff_row_component_matrix(tweaked_rows, coeff_sample=coeff_sample)

    return {
        "num_rows": info.get("num_rows"),
        "consistent_shape": {
            "value": bool(info.get("consistent_shape")),
            "zh": "所有 Tweak 输出 ciphertext 的 component/tower/ring_dim 形状一致",
        },
        "expected_parts": {
            "value": info.get("expected_parts"),
            "zh": "每个输出 ciphertext 的 RLWE component 数量；应为 2，即 c0/c1",
        },
        "expected_towers": {
            "value": info.get("expected_towers"),
            "zh": "每个 DCRTPoly component 的 RNS tower 数量",
        },
        "expected_ring_dim": {
            "value": info.get("expected_ring_dim"),
            "zh": "输出 ciphertext 的 ring dimension",
        },
        "rows": [
            {
                "row_index": rr.get("row_index"),
                "encoding_type": {
                    "value": rr.get("encoding_type"),
                    "zh": "OpenFHE encoding type；1 表示 coefficient encoding",
                },
                "num_parts": {
                    "value": rr.get("num_parts"),
                    "zh": "RLWE component 数量；应为 c0/c1 两部分",
                },
                "level": rr.get("level"),
                "noise_scale_deg": rr.get("noise_scale_deg"),
                "scaling_factor": rr.get("scaling_factor"),
                "slots": rr.get("slots"),
            }
            for rr in info.get("rows", [])
        ],
    }


def cmt_algorithm2_rowwise(rt, bundle: RowwiseCoeffCiphertextMatrix):
    """
    Real Park-2025 Algorithm 2 C-MT first implementation.

    Input:
      row-wise coefficient ciphertext matrix bundle.

    Steps:
      1. Algorithm 1 Tweak:
           ct_prime[j] = sum_i X^(2*i*j*N/n) * ct_i

      2. Automorphism:
           out[j] = Auto(ct_prime[j]; 2*j + 1)

    Output:
      transformed ciphertext list.
      Semantically this is the C-MT-transformed bundle, expected to be column-oriented
      according to the toy WP4-N result.
    """
    nrows, ncols = bundle.shape
    if nrows != ncols:
        raise ValueError(f"Algorithm 2 C-MT currently expects square matrix, got {bundle.shape}")

    n = nrows

    tweaked, tweak_trace = tweak_algorithm1_rowwise(rt, bundle)

    out = []
    auto_trace = []

    for j, ct_j in enumerate(tweaked):
        alpha = 2 * j + 1

        if alpha == 1:
            # Auto(ct; 1) is the identity automorphism.
            # Use X^0 multiplication as a clone-like no-op to avoid OpenFHE EvalAutomorphism
            # requiring a non-empty key map for the identity case.
            ct_auto = rt.monomial_mul_coeff_ct(ct_j, 0)
            auto_kind = "identity_bypass"
            zh = f"alpha=1 时 Auto(ct; 1) 是恒等变换；使用 X^0·ct 作为 clone/no-op"
        else:
            ct_auto = rt.eval_automorphism_coeff_ct(ct_j, alpha)
            auto_kind = "eval_automorphism"
            zh = f"对 Tweak 输出 ct_prime[{j}] 执行 Auto(ct; {alpha})"

        out.append(ct_auto)

        auto_trace.append({
            "j": j,
            "alpha": alpha,
            "formula": "2*j + 1",
            "auto_kind": auto_kind,
            "zh": zh,
        })

    trace = {
        "tweak": tweak_trace,
        "auto": auto_trace,
        "zh": "Algorithm 2 C-MT = Algorithm 1 Tweak 后，对每个 j 执行 Auto(ct_prime[j]; 2j+1)",
    }

    return out, trace


def describe_cmt_output(rt, cmt_rows, coeff_sample: int = 4):
    """
    Compact inspection for Algorithm 2 C-MT output.
    """
    info = rt.inspect_coeff_row_component_matrix(cmt_rows, coeff_sample=coeff_sample)

    return {
        "num_rows": info.get("num_rows"),
        "consistent_shape": {
            "value": bool(info.get("consistent_shape")),
            "zh": "所有 C-MT 输出 ciphertext 的 component/tower/ring_dim 形状一致",
        },
        "expected_parts": {
            "value": info.get("expected_parts"),
            "zh": "每个 C-MT 输出 ciphertext 的 RLWE component 数量；应为 2，即 c0/c1",
        },
        "expected_towers": {
            "value": info.get("expected_towers"),
            "zh": "每个 DCRTPoly component 的 RNS tower 数量",
        },
        "expected_ring_dim": {
            "value": info.get("expected_ring_dim"),
            "zh": "C-MT 输出 ciphertext 的 ring dimension",
        },
        "orientation": {
            "value": "cmt_transformed_expected_columnwise",
            "zh": "根据 WP4-N toy 结果，C-MT 后语义上应转为 column-wise / transformed bundle",
        },
        "rows": [
            {
                "row_index": rr.get("row_index"),
                "encoding_type": {
                    "value": rr.get("encoding_type"),
                    "zh": "OpenFHE encoding type；1 表示 coefficient encoding",
                },
                "num_parts": {
                    "value": rr.get("num_parts"),
                    "zh": "RLWE component 数量；应为 c0/c1 两部分",
                },
                "level": rr.get("level"),
                "noise_scale_deg": rr.get("noise_scale_deg"),
                "scaling_factor": rr.get("scaling_factor"),
                "slots": rr.get("slots"),
            }
            for rr in info.get("rows", [])
        ],
    }


def ccmm_algorithm3_no_relin_skeleton(rt, left_bundle: RowwiseCoeffCiphertextMatrix, right_bundle: RowwiseCoeffCiphertextMatrix):
    """
    Real Park-2025 Algorithm 3 no-relinearization skeleton.

    Input:
      left_bundle:
        row-wise coefficient ciphertext matrix for the left matrix U.

      right_bundle:
        row-wise coefficient ciphertext matrix for the right matrix V.

    Steps:
      1. Apply Algorithm 2 C-MT to right_bundle:
           right_cmt[j] is the transformed/column-oriented ciphertext for column-like index j.

      2. Compute pairwise no-relin ciphertext products:
           W_no_relin[i][j] = mul_coeff_ct_no_relin(left_bundle.rows[i], right_cmt[j])

    Output:
      matrix of 3-component ciphertexts, before relinearization/rescale.

    Notes:
      This is a structural skeleton for Algorithm 3. It verifies that the real OpenFHE/FIDESlib
      component primitives compose into an n×n matrix of 3-component ciphertexts.
    """
    if left_bundle.shape[1] != right_bundle.shape[0]:
        raise ValueError(
            f"incompatible matrix shapes for multiplication: {left_bundle.shape} and {right_bundle.shape}"
        )

    if left_bundle.shape[0] != left_bundle.shape[1] or right_bundle.shape[0] != right_bundle.shape[1]:
        raise ValueError(
            f"current skeleton expects square matrices, got {left_bundle.shape} and {right_bundle.shape}"
        )

    n = left_bundle.shape[0]

    right_cmt, right_cmt_trace = cmt_algorithm2_rowwise(rt, right_bundle)

    out = []
    product_trace = []

    for i in range(n):
        out_row = []
        trace_row = []

        for j in range(n):
            ct_prod = rt.mul_coeff_ct_no_relin(left_bundle.rows[i], right_cmt[j])
            out_row.append(ct_prod)

            trace_row.append({
                "i": i,
                "j": j,
                "left_row": i,
                "right_cmt_index": j,
                "operation": "mul_coeff_ct_no_relin(left_bundle.rows[i], right_cmt[j])",
                "zh": f"输出 W_no_relin[{i}][{j}] = 左矩阵第 {i} 行密文 × 右矩阵 C-MT 后第 {j} 个密文，结果为三分量密文",
            })

        out.append(out_row)
        product_trace.append(trace_row)

    trace = {
        "right_cmt": right_cmt_trace,
        "products": product_trace,
        "zh": "Algorithm 3 no-relin skeleton：先对右矩阵做 C-MT，再逐行逐列做 no-relin 密文乘法",
    }

    return out, trace


def describe_no_relin_product_matrix(rt, matrix, coeff_sample: int = 4):
    """
    Compact inspection for an n×n matrix of 3-component no-relin ciphertexts.
    """
    flat = [ct for row in matrix for ct in row]
    info = rt.inspect_coeff_row_component_matrix(flat, coeff_sample=coeff_sample)

    nrows = len(matrix)
    ncols = len(matrix[0]) if matrix else 0

    rows_summary = []
    for idx, rr in enumerate(info.get("rows", [])):
        i = idx // ncols if ncols else 0
        j = idx % ncols if ncols else 0

        rows_summary.append({
            "matrix_index": [i, j],
            "flat_index": idx,
            "encoding_type": {
                "value": rr.get("encoding_type"),
                "zh": "OpenFHE encoding type；1 表示 coefficient encoding",
            },
            "num_parts": {
                "value": rr.get("num_parts"),
                "zh": "no-relin 乘法输出应为 3 个 RLWE component，即 c0/c1/c2",
            },
            "level": rr.get("level"),
            "noise_scale_deg": rr.get("noise_scale_deg"),
            "scaling_factor": rr.get("scaling_factor"),
            "slots": rr.get("slots"),
        })

    return {
        "shape": {
            "value": [nrows, ncols],
            "zh": "no-relin 结果矩阵形状",
        },
        "num_ciphertexts": {
            "value": len(flat),
            "zh": "结果矩阵中的 ciphertext 总数",
        },
        "consistent_shape": {
            "value": bool(info.get("consistent_shape")),
            "zh": "所有 no-relin 输出 ciphertext 的 component/tower/ring_dim 形状一致",
        },
        "expected_parts": {
            "value": info.get("expected_parts"),
            "zh": "每个 no-relin 输出 ciphertext 的 RLWE component 数量；应为 3",
        },
        "expected_towers": {
            "value": info.get("expected_towers"),
            "zh": "每个 DCRTPoly component 的 RNS tower 数量",
        },
        "expected_ring_dim": {
            "value": info.get("expected_ring_dim"),
            "zh": "输出 ciphertext 的 ring dimension",
        },
        "entries": rows_summary,
    }


def relinearize_ciphertext_matrix(rt, matrix):
    """
    Relinearize an n×n matrix of no-relin 3-component ciphertexts.

    Input:
      matrix[i][j] has c0/c1/c2 components.

    Output:
      relin_matrix[i][j] has c0/c1 components.
    """
    out = []
    trace = []

    for i, row in enumerate(matrix):
        out_row = []
        trace_row = []

        for j, ct in enumerate(row):
            ct_relin = rt.relinearize_coeff_ct(ct)
            out_row.append(ct_relin)

            trace_row.append({
                "i": i,
                "j": j,
                "operation": "relinearize_coeff_ct(W_no_relin[i][j])",
                "zh": f"将 W_no_relin[{i}][{j}] 从三分量 c0/c1/c2 压回二分量 c0/c1",
            })

        out.append(out_row)
        trace.append(trace_row)

    return out, trace


def describe_relinearized_product_matrix(rt, matrix, coeff_sample: int = 4):
    """
    Compact inspection for an n×n matrix of relinearized 2-component ciphertexts.
    """
    flat = [ct for row in matrix for ct in row]
    info = rt.inspect_coeff_row_component_matrix(flat, coeff_sample=coeff_sample)

    nrows = len(matrix)
    ncols = len(matrix[0]) if matrix else 0

    entries = []
    for idx, rr in enumerate(info.get("rows", [])):
        i = idx // ncols if ncols else 0
        j = idx % ncols if ncols else 0

        entries.append({
            "matrix_index": [i, j],
            "flat_index": idx,
            "encoding_type": {
                "value": rr.get("encoding_type"),
                "zh": "OpenFHE encoding type；1 表示 coefficient encoding",
            },
            "num_parts": {
                "value": rr.get("num_parts"),
                "zh": "relinearize 后应为 2 个 RLWE component，即 c0/c1",
            },
            "level": rr.get("level"),
            "noise_scale_deg": rr.get("noise_scale_deg"),
            "scaling_factor": rr.get("scaling_factor"),
            "slots": rr.get("slots"),
        })

    return {
        "shape": {
            "value": [nrows, ncols],
            "zh": "relinearized 结果矩阵形状",
        },
        "num_ciphertexts": {
            "value": len(flat),
            "zh": "relinearized 结果矩阵中的 ciphertext 总数",
        },
        "consistent_shape": {
            "value": bool(info.get("consistent_shape")),
            "zh": "所有 relinearized 输出 ciphertext 的 component/tower/ring_dim 形状一致",
        },
        "expected_parts": {
            "value": info.get("expected_parts"),
            "zh": "每个 relinearized 输出 ciphertext 的 RLWE component 数量；应为 2",
        },
        "expected_towers": {
            "value": info.get("expected_towers"),
            "zh": "每个 DCRTPoly component 的 RNS tower 数量",
        },
        "expected_ring_dim": {
            "value": info.get("expected_ring_dim"),
            "zh": "输出 ciphertext 的 ring dimension",
        },
        "entries": entries,
    }



def pack_relinearized_position_matrix_rowwise(rt, matrix):
    """
    Pack an n×n matrix of position-level relinearized ciphertexts into n row-wise ciphertexts.

    Input:
      matrix[i][j] is a 2-component coefficient ciphertext for output position (i, j).

    Packing hypothesis:
      W_row[i] = sum_j X^j * matrix[i][j]

    Output:
      rowwise_rows[i] is a 2-component coefficient ciphertext representing one output row.

    Notes:
      This is a structural packing/reduction step. It verifies that position-level
      ciphertexts can be combined back into row-wise ciphertexts. Numerical validation
      is intentionally deferred.
    """
    out = []
    trace = []

    for i, row in enumerate(matrix):
        acc = None
        row_trace = []

        for j, ct in enumerate(row):
            exp = j
            placed = rt.monomial_mul_coeff_ct(ct, exp)

            if acc is None:
                acc = placed
            else:
                acc = rt.add_coeff_ct(acc, placed)

            row_trace.append({
                "i": i,
                "j": j,
                "exp": exp,
                "operation": "X^j * W_relin[i][j]",
                "zh": f"将位置级 ciphertext W_relin[{i}][{j}] 乘以 X^{j}，放入输出行的第 {j} 个 coefficient 位置",
            })

        out.append(acc)
        trace.append({
            "row": i,
            "terms": row_trace,
            "zh": f"输出 row-wise ciphertext W_row[{i}] 由第 {i} 行的所有位置级 ciphertext placement 后相加得到",
        })

    return out, trace


def describe_rowwise_packed_output(rt, rows, coeff_sample: int = 4):
    """
    Inspect packed row-wise output ciphertexts.
    """
    info = rt.inspect_coeff_row_component_matrix(rows, coeff_sample=coeff_sample)

    row_entries = []
    for rr in info.get("rows", []):
        row_entries.append({
            "row_index": rr.get("row_index"),
            "encoding_type": {
                "value": rr.get("encoding_type"),
                "zh": "OpenFHE encoding type；1 表示 coefficient encoding",
            },
            "num_parts": {
                "value": rr.get("num_parts"),
                "zh": "row-wise packed 输出应为 2 个 RLWE component，即 c0/c1",
            },
            "level": rr.get("level"),
            "noise_scale_deg": rr.get("noise_scale_deg"),
            "scaling_factor": rr.get("scaling_factor"),
            "slots": rr.get("slots"),
        })

    return {
        "num_rows": {
            "value": info.get("num_rows"),
            "zh": "row-wise packed 输出 ciphertext 数量；4x4 矩阵应为 4 个 row ciphertext",
        },
        "consistent_shape": {
            "value": bool(info.get("consistent_shape")),
            "zh": "所有 row-wise packed 输出 ciphertext 的 component/tower/ring_dim 形状一致",
        },
        "expected_parts": {
            "value": info.get("expected_parts"),
            "zh": "每个 row-wise packed 输出 ciphertext 的 RLWE component 数量；应为 2",
        },
        "expected_towers": {
            "value": info.get("expected_towers"),
            "zh": "每个 DCRTPoly component 的 RNS tower 数量",
        },
        "expected_ring_dim": {
            "value": info.get("expected_ring_dim"),
            "zh": "row-wise packed 输出 ciphertext 的 ring dimension",
        },
        "rows": row_entries,
    }


def assemble_algorithm3_component_pair(rt, left_ct, right_ct):
    """
    Assemble one paper-formula Algorithm 3 product ciphertext from selected components.

    Given:
      left_ct  = (L0, L1)
      right_ct = (R0, R1)

    Produce:
      C0 = L0 * R0
      C1 = L0 * R1 + L1 * R0
      C2 = L1 * R1

    Output:
      assembled 3-component ciphertext-like handle.
    """
    C0 = rt.component_mul_part_coeff_ct(left_ct, 0, right_ct, 0)

    C1_left = rt.component_mul_part_coeff_ct(left_ct, 0, right_ct, 1)
    C1_right = rt.component_mul_part_coeff_ct(left_ct, 1, right_ct, 0)
    C1 = rt.component_add_1part_coeff_ct(C1_left, C1_right)

    C2 = rt.component_mul_part_coeff_ct(left_ct, 1, right_ct, 1)

    assembled = rt.assemble_3part_from_1parts_coeff_ct(C0, C1, C2)

    trace = {
        "C0": "component_mul(left.c0, right.c0)",
        "C1": "component_mul(left.c0, right.c1) + component_mul(left.c1, right.c0)",
        "C2": "component_mul(left.c1, right.c1)",
        "zh": "显式使用论文三分量公式组装一个 before-relinearization ciphertext",
    }

    return assembled, trace


def ccmm_algorithm3_component_formula_skeleton(rt, left_bundle: RowwiseCoeffCiphertextMatrix, right_bundle: RowwiseCoeffCiphertextMatrix):
    """
    Matrix-level component-formula Algorithm 3 skeleton.

    This version does NOT use whole-ciphertext multiplication.

    Steps:
      1. right_cmt = C-MT(right_bundle)
      2. for each i,j:
           assemble_algorithm3_component_pair(left_bundle.rows[i], right_cmt[j])

    Output:
      n x n matrix of 3-component ciphertext-like handles.

    Important:
      This is still a skeleton. It verifies that the whole-ciphertext multiplication has
      been replaced by explicit C0/C1/C2 component assembly. It does not yet implement
      any additional PP-MM placement/indexing beyond the current pairwise matrix skeleton.
    """
    if left_bundle.shape[1] != right_bundle.shape[0]:
        raise ValueError(
            f"incompatible matrix shapes for multiplication: {left_bundle.shape} and {right_bundle.shape}"
        )

    if left_bundle.shape[0] != left_bundle.shape[1] or right_bundle.shape[0] != right_bundle.shape[1]:
        raise ValueError(
            f"current skeleton expects square matrices, got {left_bundle.shape} and {right_bundle.shape}"
        )

    n = left_bundle.shape[0]
    right_cmt, right_cmt_trace = cmt_algorithm2_rowwise(rt, right_bundle)

    out = []
    product_trace = []

    for i in range(n):
        out_row = []
        trace_row = []

        for j in range(n):
            ct_prod, pair_trace = assemble_algorithm3_component_pair(
                rt,
                left_bundle.rows[i],
                right_cmt[j],
            )
            out_row.append(ct_prod)

            trace_row.append({
                "i": i,
                "j": j,
                "left_row": i,
                "right_cmt_index": j,
                "operation": "assemble_algorithm3_component_pair(left.rows[i], right_cmt[j])",
                "pair_trace": pair_trace,
                "zh": f"W_component_formula[{i}][{j}] 使用 C0/C1/C2 component 公式组装，不调用整密文乘法",
            })

        out.append(out_row)
        product_trace.append(trace_row)

    trace = {
        "right_cmt": right_cmt_trace,
        "products": product_trace,
        "uses_whole_ciphertext_multiply": False,
        "zh": "Algorithm 3 component-formula skeleton：右矩阵先 C-MT，然后每个位置显式组装 C0/C1/C2",
    }

    return out, trace
