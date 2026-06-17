from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


Orientation = Literal["rowwise", "columnwise"]


def negacyclic_toeplitz(poly) -> np.ndarray:
    """
    Toeplitz matrix for multiplication in R = Z[X] / (X^N + 1).

    For poly = s0 + s1 X + ... + s_{N-1} X^{N-1},

        row vector a_coeffs @ Toep(poly)

    equals the coefficient vector of a(X) * poly(X) mod X^N + 1.

    Matches the paper's Toep(sk) convention.
    """
    s = np.asarray(poly, dtype=np.float64).reshape(-1)
    n = s.shape[0]
    T = np.zeros((n, n), dtype=np.float64)

    for r in range(n):
        for c in range(n):
            k = c - r
            if k >= 0:
                T[r, c] = s[k]
            else:
                T[r, c] = -s[n + k]

    return T


def negacyclic_inverse_poly(poly) -> np.ndarray:
    """
    Coefficients of poly(X^{-1}) in R = Z[X] / (X^N + 1).

    Since X^{-i} = -X^{N-i} for i > 0:
        f[0] = poly[0]
        f[N-i] = -poly[i]
    """
    s = np.asarray(poly, dtype=np.float64).reshape(-1)
    n = s.shape[0]
    out = np.zeros_like(s)
    out[0] = s[0]
    for i in range(1, n):
        out[n - i] = -s[i]
    return out


@dataclass
class EncryptedMatrixRLWEBundle:
    """
    Paper-aligned matrix-form RLWE ciphertext bundle.

    rowwise:
        A @ Toep(sk) + B ~= M

    columnwise:
        Toep(skf) @ A + B ~= M

    This toy object stores component matrices directly. In the real HE backend,
    A and B come from RLWE ciphertext components, not from plaintext arrays.
    """

    A: np.ndarray
    B: np.ndarray
    shape: Tuple[int, int]
    orientation: Orientation
    encoding: str = "coefficient"
    meta: Optional[dict] = None

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=np.float64)
        self.B = np.asarray(self.B, dtype=np.float64)

        if self.A.shape != self.shape:
            raise ValueError(f"A.shape {self.A.shape} != shape {self.shape}")
        if self.B.shape != self.shape:
            raise ValueError(f"B.shape {self.B.shape} != shape {self.shape}")
        if self.orientation not in ("rowwise", "columnwise"):
            raise ValueError(f"bad orientation: {self.orientation}")


@dataclass
class CCMMToyTrace:
    """
    Debug trace for the paper-aligned toy CC-MM.
    """

    M00: np.ndarray
    M01: np.ndarray
    M10: np.ndarray
    M11: np.ndarray
    A_check: np.ndarray
    B_check: np.ndarray
    A_hat: np.ndarray
    B_hat: np.ndarray
    C2_before_relin: np.ndarray
    C1_before_relin: np.ndarray
    C0_before_relin: np.ndarray


def reconstruct_bundle(bundle: EncryptedMatrixRLWEBundle, sk) -> np.ndarray:
    """
    Reconstruct plaintext matrix from the toy matrix-form bundle.

    This uses sk only for toy verification. A real server-side CC-MM
    implementation must not require secret key access.
    """
    sk = np.asarray(sk, dtype=np.float64).reshape(-1)
    T = negacyclic_toeplitz(sk)
    Tf = negacyclic_toeplitz(negacyclic_inverse_poly(sk))

    if bundle.orientation == "rowwise":
        return bundle.A @ T + bundle.B

    return Tf @ bundle.A + bundle.B


def make_rowwise_bundle_from_plain(
    M,
    sk,
    A: Optional[np.ndarray] = None,
) -> EncryptedMatrixRLWEBundle:
    """
    Toy constructor for row-wise matrix-form encryption:

        A @ Toep(sk) + B = M

    This is NOT encryption. It only constructs a component-pair satisfying
    the paper's matrix-form equation for algorithm testing.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"expected square matrix, got {M.shape}")

    n = M.shape[0]
    T = negacyclic_toeplitz(sk)

    if A is None:
        A = np.zeros((n, n), dtype=np.float64)
    else:
        A = np.asarray(A, dtype=np.float64)

    B = M - A @ T

    return EncryptedMatrixRLWEBundle(
        A=A,
        B=B,
        shape=M.shape,
        orientation="rowwise",
        meta={"toy_constructor": "rowwise_from_plain"},
    )


def make_columnwise_bundle_from_plain(
    M,
    sk,
    A: Optional[np.ndarray] = None,
) -> EncryptedMatrixRLWEBundle:
    """
    Toy constructor for column-wise matrix-form encryption:

        Toep(skf) @ A + B = M

    This is NOT encryption. It only constructs a component-pair satisfying
    the paper's matrix-form equation for algorithm testing.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"expected square matrix, got {M.shape}")

    n = M.shape[0]
    Tf = negacyclic_toeplitz(negacyclic_inverse_poly(sk))

    if A is None:
        A = np.zeros((n, n), dtype=np.float64)
    else:
        A = np.asarray(A, dtype=np.float64)

    B = M - Tf @ A

    return EncryptedMatrixRLWEBundle(
        A=A,
        B=B,
        shape=M.shape,
        orientation="columnwise",
        meta={"toy_constructor": "columnwise_from_plain"},
    )


def cmt_oracle_toy(
    bundle: EncryptedMatrixRLWEBundle,
    sk,
    target_orientation: Optional[Orientation] = None,
) -> EncryptedMatrixRLWEBundle:
    """
    Toy oracle for ciphertext matrix transpose.

    It preserves the matrix encrypted by the input bundle, but switches
    rowwise <-> columnwise representation.

    This is NOT Algorithm 2. It deliberately uses reconstruct_bundle(...)
    and therefore belongs only to toy correctness tests.

    Real implementation target:
        Algorithm 2 Transpose = Tweak -> Auto -> Tweak -> monomial correction.
    """
    if target_orientation is None:
        target_orientation = "columnwise" if bundle.orientation == "rowwise" else "rowwise"

    M = reconstruct_bundle(bundle, sk)

    # Choose deterministic non-zero A to exercise all four PP-MM terms.
    # This is arbitrary toy metadata, not encryption.
    A_seed = bundle.B.T if bundle.B.shape == M.shape else np.zeros_like(M)

    if target_orientation == "rowwise":
        return make_rowwise_bundle_from_plain(M, sk, A=A_seed)

    return make_columnwise_bundle_from_plain(M, sk, A=A_seed)


def ppmm_four_products(
    left_columnwise: EncryptedMatrixRLWEBundle,
    right_rowwise: EncryptedMatrixRLWEBundle,
):
    """
    Algorithm 3 line 2:

        [M00 M01]
        [M10 M11] = [A_U_bar] [A_V B_V]
                    [B_U_bar]
    """
    if left_columnwise.orientation != "columnwise":
        raise ValueError("left input must be columnwise after first C-MT")
    if right_rowwise.orientation != "rowwise":
        raise ValueError("right input must be rowwise")

    AU = left_columnwise.A
    BU = left_columnwise.B
    AV = right_rowwise.A
    BV = right_rowwise.B

    M00 = AU @ AV
    M01 = AU @ BV
    M10 = BU @ AV
    M11 = BU @ BV

    return M00, M01, M10, M11


def relinearize_and_rescale_oracle_toy(C2, C1, C0, sk) -> EncryptedMatrixRLWEBundle:
    """
    Toy oracle for Algorithm 3 lines 5-6.

    In the real algorithm:
        C2 corresponds to sk^2 and must be key-switched from sk^2 to sk.
        Then the result is rescaled.

    Here we reconstruct the represented plaintext:
        C2 @ Toep(sk)^2 + C1 @ Toep(sk) + C0

    and return a rowwise bundle encrypting the same matrix.
    """
    T = negacyclic_toeplitz(sk)
    plaintext = C2 @ (T @ T) + C1 @ T + C0

    # Deterministic A choice so output is not the trivial A=0 representation.
    A_out = C1.copy()

    return make_rowwise_bundle_from_plain(plaintext, sk, A=A_out)


def ccmm_paper2025_toy(
    left: EncryptedMatrixRLWEBundle,
    right: EncryptedMatrixRLWEBundle,
    sk,
    *,
    return_trace: bool = False,
):
    """
    Paper-aligned toy implementation of Algorithm 3.

    Input:
        left  = row-wise bundle encrypting U
        right = row-wise bundle encrypting V

    Output:
        row-wise bundle encrypting U @ V

    This is NOT a real HE implementation yet:
        - C-MT is cmt_oracle_toy
        - key switching is relinearize_and_rescale_oracle_toy
        - rescale is identity inside the toy oracle

    The purpose is to lock down the exact Algorithm 3 algebra before connecting
    FIDESlib/OpenFHE component kernels.
    """
    if left.orientation != "rowwise":
        raise ValueError("left must be rowwise")
    if right.orientation != "rowwise":
        raise ValueError("right must be rowwise")
    if left.shape[1] != right.shape[0]:
        raise ValueError(f"shape mismatch: {left.shape} @ {right.shape}")

    n = left.shape[0]
    if left.shape != (n, n) or right.shape != (n, n):
        raise ValueError("toy implementation currently supports square N x N only")

    # Algorithm 3 line 1.
    left_col = cmt_oracle_toy(left, sk, target_orientation="columnwise")

    # Algorithm 3 line 2: four PP-MMs.
    M00, M01, M10, M11 = ppmm_four_products(left_col, right)

    zero = np.zeros_like(M00)

    # Algorithm 3 lines 3-4:
    # Transpose((M01, 0)) and Transpose((M00, 0)).
    # These are column-wise encryptions of Toep(skf) @ M01 and Toep(skf) @ M00.
    m01_col_bundle = EncryptedMatrixRLWEBundle(
        A=M01,
        B=zero.copy(),
        shape=M01.shape,
        orientation="columnwise",
        meta={"toy_source": "(M01, 0)"},
    )
    m00_col_bundle = EncryptedMatrixRLWEBundle(
        A=M00,
        B=zero.copy(),
        shape=M00.shape,
        orientation="columnwise",
        meta={"toy_source": "(M00, 0)"},
    )

    check = cmt_oracle_toy(m01_col_bundle, sk, target_orientation="rowwise")
    hat = cmt_oracle_toy(m00_col_bundle, sk, target_orientation="rowwise")

    A_check, B_check = check.A, check.B
    A_hat, B_hat = hat.A, hat.B

    # Algorithm 3 lines 5-6 before relinearization:
    # A_hat * sk^2 + (B_hat + A_check + M10) * sk + (B_check + M11)
    C2 = A_hat
    C1 = B_hat + A_check + M10
    C0 = B_check + M11

    out = relinearize_and_rescale_oracle_toy(C2, C1, C0, sk)

    if not return_trace:
        return out

    trace = CCMMToyTrace(
        M00=M00,
        M01=M01,
        M10=M10,
        M11=M11,
        A_check=A_check,
        B_check=B_check,
        A_hat=A_hat,
        B_hat=B_hat,
        C2_before_relin=C2,
        C1_before_relin=C1,
        C0_before_relin=C0,
    )

    return out, trace


def ccmm_paper2025(left: EncryptedMatrixRLWEBundle, right: EncryptedMatrixRLWEBundle):
    """
    Real Park-2025 CC-MM entry point.

    This is intentionally not implemented until we replace:
      - cmt_oracle_toy with Algorithm 2 C-MT
      - relinearize_and_rescale_oracle_toy with real KS_{s^2->s} + Rescale
      - NumPy PP-MM with modular/component PP-MM backend
    """
    raise NotImplementedError(
        "Real Park-2025 CC-MM is not implemented yet. "
        "Use ccmm_paper2025_toy(...) only for paper-aligned algebra validation."
    )


# ======================================================================================
# WP4-N: Algorithm 3 with Algorithm 2 exact toy component C-MT
# ======================================================================================

from .cmt_paper2025 import (
    ToyRLWECiphertext,
    make_exact_auto_switching_keys,
    make_exact_relin_switching_key,
    ring_mul_negacyclic,
    toy_key_switch_exact,
    transpose_algorithm2_ciphertexts_toy,
)


def _bundle_to_toy_ciphertexts(bundle: EncryptedMatrixRLWEBundle):
    """
    Convert matrix-form bundle into a list of toy ciphertexts.

    rowwise:
        row i of A/B is ciphertext i.

    columnwise:
        column j of A/B is ciphertext j.
    """
    n = bundle.shape[0]

    if bundle.orientation == "rowwise":
        return [
            ToyRLWECiphertext(bundle.A[i, :], bundle.B[i, :])
            for i in range(n)
        ]

    return [
        ToyRLWECiphertext(bundle.A[:, j], bundle.B[:, j])
        for j in range(n)
    ]


def _toy_ciphertexts_to_bundle(cts, orientation: Orientation) -> EncryptedMatrixRLWEBundle:
    """
    Convert list of toy ciphertexts back to matrix-form bundle.
    """
    cts = list(cts)
    n = len(cts)

    if orientation == "rowwise":
        A = np.vstack([ct.a for ct in cts])
        B = np.vstack([ct.b for ct in cts])
    elif orientation == "columnwise":
        A = np.column_stack([ct.a for ct in cts])
        B = np.column_stack([ct.b for ct in cts])
    else:
        raise ValueError(f"bad orientation: {orientation}")

    return EncryptedMatrixRLWEBundle(
        A=A,
        B=B,
        shape=(n, n),
        orientation=orientation,
        meta={"toy_cmt": "algorithm2_exact_component_model"},
    )


def cmt_algorithm2_component_toy(
    bundle: EncryptedMatrixRLWEBundle,
    sk,
    auto_keys=None,
) -> EncryptedMatrixRLWEBundle:
    """
    Component-level toy C-MT using paper Algorithm 2.

    This converts:
        rowwise    -> columnwise
        columnwise -> rowwise

    It does not reconstruct plaintext and directly operate on toy ciphertext
    components with exact toy Auto/key-switching.
    """
    if bundle.shape[0] != bundle.shape[1]:
        raise ValueError("component C-MT toy supports square N x N only")

    if auto_keys is None:
        auto_keys = make_exact_auto_switching_keys(sk)

    input_cts = _bundle_to_toy_ciphertexts(bundle)
    output_cts = transpose_algorithm2_ciphertexts_toy(input_cts, auto_keys)

    target = "columnwise" if bundle.orientation == "rowwise" else "rowwise"
    return _toy_ciphertexts_to_bundle(output_cts, target)


def relinearize_and_rescale_component_toy(C2, C1, C0, sk, relin_key=None) -> EncryptedMatrixRLWEBundle:
    """
    Exact toy replacement for Algorithm 3 lines 5-6.

    Input algebra:
        C2 * sk^2 + C1 * sk + C0

    Exact toy relinearization key:
        relin_key.a * sk + relin_key.b = sk^2

    Output rowwise:
        (C2 * relin_key.a + C1) * sk + (C2 * relin_key.b + C0)
    """
    C2 = np.asarray(C2, dtype=np.float64)
    C1 = np.asarray(C1, dtype=np.float64)
    C0 = np.asarray(C0, dtype=np.float64)

    if C2.shape != C1.shape or C2.shape != C0.shape:
        raise ValueError("C2/C1/C0 shape mismatch")

    if relin_key is None:
        relin_key = make_exact_relin_switching_key(sk)

    n = C2.shape[0]
    A_out = np.zeros_like(C2)
    B_out = np.zeros_like(C2)

    for i in range(n):
        A_out[i, :] = ring_mul_negacyclic(C2[i, :], relin_key.a) + C1[i, :]
        B_out[i, :] = ring_mul_negacyclic(C2[i, :], relin_key.b) + C0[i, :]

    return EncryptedMatrixRLWEBundle(
        A=A_out,
        B=B_out,
        shape=C2.shape,
        orientation="rowwise",
        meta={"toy_relin": "exact_sk2_to_sk", "toy_rescale": "identity"},
    )


def ccmm_paper2025_component_toy(
    left: EncryptedMatrixRLWEBundle,
    right: EncryptedMatrixRLWEBundle,
    sk,
    *,
    auto_keys=None,
    relin_key=None,
    return_trace: bool = False,
):
    """
    Paper Algorithm 3 with component-level Algorithm 2 toy C-MT.

    This is closer than ccmm_paper2025_toy:
      - C-MT is no longer reconstruct-based oracle.
      - Auto/key-switching is modeled on toy RLWE components.
      - Relinearization is exact toy sk^2 -> sk key-switching.

    Still not real HE:
      - no RNS
      - no gadget decomposition
      - no CKKS scaling/rescale
      - no OpenFHE/FIDESlib ciphertext handles
    """
    if left.orientation != "rowwise":
        raise ValueError("left must be rowwise")
    if right.orientation != "rowwise":
        raise ValueError("right must be rowwise")
    if left.shape != right.shape or left.shape[0] != left.shape[1]:
        raise ValueError("component toy currently supports same square N x N inputs")

    if auto_keys is None:
        auto_keys = make_exact_auto_switching_keys(sk)
    if relin_key is None:
        relin_key = make_exact_relin_switching_key(sk)

    # Algorithm 3 line 1.
    left_col = cmt_algorithm2_component_toy(left, sk, auto_keys=auto_keys)

    # Algorithm 3 line 2.
    M00, M01, M10, M11 = ppmm_four_products(left_col, right)

    zero = np.zeros_like(M00)

    # Algorithm 3 line 3: Transpose((M01, 0)).
    m01_col_bundle = EncryptedMatrixRLWEBundle(
        A=M01,
        B=zero.copy(),
        shape=M01.shape,
        orientation="columnwise",
        meta={"toy_source": "(M01, 0)"},
    )
    check = cmt_algorithm2_component_toy(m01_col_bundle, sk, auto_keys=auto_keys)

    # Algorithm 3 line 4: Transpose((M00, 0)).
    m00_col_bundle = EncryptedMatrixRLWEBundle(
        A=M00,
        B=zero.copy(),
        shape=M00.shape,
        orientation="columnwise",
        meta={"toy_source": "(M00, 0)"},
    )
    hat = cmt_algorithm2_component_toy(m00_col_bundle, sk, auto_keys=auto_keys)

    # Before Algorithm 3 line 5.
    C2 = hat.A
    C1 = hat.B + check.A + M10
    C0 = check.B + M11

    out = relinearize_and_rescale_component_toy(
        C2,
        C1,
        C0,
        sk,
        relin_key=relin_key,
    )

    if not return_trace:
        return out

    trace = {
        "left_col_orientation": left_col.orientation,
        "M00_shape": list(M00.shape),
        "M01_shape": list(M01.shape),
        "M10_shape": list(M10.shape),
        "M11_shape": list(M11.shape),
        "check_orientation": check.orientation,
        "hat_orientation": hat.orientation,
        "C2_shape": list(C2.shape),
        "C1_shape": list(C1.shape),
        "C0_shape": list(C0.shape),
    }

    return out, trace
