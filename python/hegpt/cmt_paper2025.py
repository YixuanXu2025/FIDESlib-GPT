from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class CMTPlainTrace:
    """
    Debug trace for paper Algorithm 2 in plaintext polynomial toy mode.
    """

    input_rows: np.ndarray
    x_i_rows: np.ndarray
    aux_after_first_tweak: np.ndarray
    aux_prime_after_auto: np.ndarray
    ct_double_prime_after_second_tweak: np.ndarray
    output_cols: np.ndarray


def _check_power_of_two(n: int):
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"expected power-of-two positive integer, got {n}")


def monomial_mul_negacyclic(poly, exp: int) -> np.ndarray:
    """
    Multiply a polynomial by X^exp in R = Z[X] / (X^N + 1).

    Coefficients are represented as:
        poly[k] = coefficient of X^k

    This works for arbitrary integer exp.
    """
    p = np.asarray(poly, dtype=np.float64).reshape(-1)
    n = p.shape[0]
    out = np.zeros_like(p)

    mod = 2 * n
    for i, c in enumerate(p):
        if c == 0:
            continue

        e = (i + int(exp)) % mod
        if e < n:
            out[e] += c
        else:
            out[e - n] -= c

    return out


def automorphism_negacyclic(poly, alpha: int) -> np.ndarray:
    """
    Apply the ring automorphism sigma_alpha: X -> X^alpha.

    alpha must be odd modulo 2N.
    """
    p = np.asarray(poly, dtype=np.float64).reshape(-1)
    n = p.shape[0]

    alpha = int(alpha) % (2 * n)
    if alpha % 2 != 1:
        raise ValueError(f"alpha must be odd modulo 2N, got {alpha}")

    out = np.zeros_like(p)

    for i, c in enumerate(p):
        if c == 0:
            continue

        e = (alpha * i) % (2 * n)
        if e < n:
            out[e] += c
        else:
            out[e - n] -= c

    return out


def tweak_plain(polys: Sequence[Sequence[float]]) -> List[np.ndarray]:
    """
    Plain polynomial version of paper Algorithm 1 Tweak.

    For n input polynomials ct_i of ring degree N, output:

        ct'_j = sum_i X^(2 i j N/n) * ct_i

    for j in [n].

    In Algorithm 2 we use n = N, so the exponent becomes 2ij.
    """
    arr = [np.asarray(p, dtype=np.float64).reshape(-1) for p in polys]
    n = len(arr)
    if n == 0:
        raise ValueError("tweak_plain(): empty input")

    _check_power_of_two(n)

    ring_n = arr[0].shape[0]
    for i, p in enumerate(arr):
        if p.shape[0] != ring_n:
            raise ValueError(f"polys[{i}] length {p.shape[0]} != {ring_n}")

    if ring_n % n != 0:
        raise ValueError(f"ring degree N={ring_n} must be divisible by tweak n={n}")

    out = []
    factor = ring_n // n

    for j in range(n):
        acc = np.zeros(ring_n, dtype=np.float64)

        for i, p in enumerate(arr):
            exp = 2 * i * j * factor
            acc += monomial_mul_negacyclic(p, exp)

        out.append(acc)

    return out


def _odd_inverse_index_for_algorithm2(j: int, N: int) -> int:
    """
    Algorithm 2 line 3 index:

        ((2j+1)^(-1) mod 2N - 1) / 2

    The inverse of an odd number modulo 2N is also odd, so this is integer.
    """
    odd = 2 * j + 1
    inv = pow(odd, -1, 2 * N)
    if inv % 2 != 1:
        raise RuntimeError("internal error: inverse of odd modulo 2N is not odd")
    return (inv - 1) // 2


def transpose_algorithm2_plain_rows_to_cols(
    rows: Sequence[Sequence[float]],
    *,
    return_trace: bool = False,
):
    """
    Plain coefficient-polynomial toy implementation of paper Algorithm 2.

    Input:
        rows[i][j] = M[i, j]

    Output:
        cols[j][i] = M[i, j]

    That is, output polynomials encode columns of the input matrix.

    Important:
        This is NOT homomorphic C-MT yet.
        It only validates the coefficient/ring algebra of Algorithm 2.
    """
    row_polys = [np.asarray(r, dtype=np.float64).reshape(-1) for r in rows]
    N = len(row_polys)

    if N == 0:
        raise ValueError("transpose_algorithm2_plain_rows_to_cols(): empty input")
    _check_power_of_two(N)

    for i, p in enumerate(row_polys):
        if p.shape[0] != N:
            raise ValueError(
                f"expected an N x N coefficient matrix; row {i} has length {p.shape[0]}, N={N}"
            )

    # Algorithm 2 line 1:
    # aux <- Tweak(N, {X^i * ct_i})
    x_i_rows = [
        monomial_mul_negacyclic(row_polys[i], i)
        for i in range(N)
    ]
    aux = tweak_plain(x_i_rows)

    # Algorithm 2 lines 2-5:
    # aux'_j <- N^{-1} * aux_idx
    # aux'_j <- Auto(aux'_j; 2j+1)
    #
    # In this plaintext toy over R, N^{-1} is represented by real scaling 1/N.
    aux_prime = []
    for j in range(N):
        idx = _odd_inverse_index_for_algorithm2(j, N)
        odd = 2 * j + 1
        aux_prime_j = (1.0 / float(N)) * automorphism_negacyclic(aux[idx], odd)
        aux_prime.append(aux_prime_j)

    # Algorithm 2 line 6:
    # ct'' <- Tweak(N, aux')
    ct_double_prime = tweak_plain(aux_prime)

    # Algorithm 2 lines 7-9:
    #
    # The paper writes:
    #   for j <- 1 to N:
    #       ct'_{j mod N} <- -X^{N-j} * ct''_{(N-j) mod N}
    #
    # In direct coefficient tests, the j=N case must be interpreted as the
    # wraparound monomial correction -X^N = +1 in R = Z[X]/(X^N+1), otherwise
    # column 0 gets an incorrect global sign. For j=1..N-1, the printed formula
    # is used directly.
    out = [None for _ in range(N)]

    for j in range(1, N + 1):
        src = (N - j) % N
        dst = j % N

        if j == N:
            corrected = ct_double_prime[src].copy()
        else:
            corrected = -monomial_mul_negacyclic(ct_double_prime[src], N - j)

        out[dst] = corrected

    if not return_trace:
        return out

    trace = CMTPlainTrace(
        input_rows=np.vstack(row_polys),
        x_i_rows=np.vstack(x_i_rows),
        aux_after_first_tweak=np.vstack(aux),
        aux_prime_after_auto=np.vstack(aux_prime),
        ct_double_prime_after_second_tweak=np.vstack(ct_double_prime),
        output_cols=np.vstack(out),
    )
    return out, trace


def matrix_rows_to_polys(M) -> List[np.ndarray]:
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"expected square matrix, got {M.shape}")
    return [M[i, :].copy() for i in range(M.shape[0])]


def polys_to_matrix_rows(polys: Sequence[Sequence[float]]) -> np.ndarray:
    return np.vstack([np.asarray(p, dtype=np.float64).reshape(-1) for p in polys])


# ======================================================================================
# Exact toy RLWE component model for Algorithm 2 C-MT
# ======================================================================================

@dataclass
class ToyRLWECiphertext:
    """
    Exact toy RLWE ciphertext over R = Z[X]/(X^N+1).

    Semantics:
        a * sk + b = message

    This is for paper-aligned algorithm validation only.
    It is not secure encryption and not an RNS/CKKS implementation.
    """

    a: np.ndarray
    b: np.ndarray

    def __post_init__(self):
        self.a = np.asarray(self.a, dtype=np.float64).reshape(-1)
        self.b = np.asarray(self.b, dtype=np.float64).reshape(-1)
        if self.a.shape != self.b.shape:
            raise ValueError(f"a.shape {self.a.shape} != b.shape {self.b.shape}")


@dataclass
class ExactToySwitchingKey:
    """
    Exact toy switching key.

    Semantics:
        key.a * target_sk + key.b = source_sk

    This lets us switch a ciphertext under source_sk back to target_sk exactly:

        c0 * source_sk + c1
      = c0 * (key.a * target_sk + key.b) + c1
      = (c0 * key.a) * target_sk + (c0 * key.b + c1)
    """

    source_name: str
    a: np.ndarray
    b: np.ndarray

    def __post_init__(self):
        self.a = np.asarray(self.a, dtype=np.float64).reshape(-1)
        self.b = np.asarray(self.b, dtype=np.float64).reshape(-1)
        if self.a.shape != self.b.shape:
            raise ValueError(f"a.shape {self.a.shape} != b.shape {self.b.shape}")


def ring_mul_negacyclic(x, y) -> np.ndarray:
    """
    Multiply two polynomials in R = Z[X]/(X^N + 1).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"ring_mul_negacyclic shape mismatch: {x.shape} vs {y.shape}")

    n = x.shape[0]
    out = np.zeros(n, dtype=np.float64)

    for i, xi in enumerate(x):
        if xi == 0:
            continue
        for j, yj in enumerate(y):
            if yj == 0:
                continue
            out += monomial_mul_negacyclic(np.eye(1, n, 0).reshape(-1) * (xi * yj), i + j)

    return out


def ring_mul_negacyclic_fast(x, y) -> np.ndarray:
    """
    Same as ring_mul_negacyclic, but written directly to avoid temporary basis vectors.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"ring_mul_negacyclic_fast shape mismatch: {x.shape} vs {y.shape}")

    n = x.shape[0]
    out = np.zeros(n, dtype=np.float64)

    for i, xi in enumerate(x):
        if xi == 0:
            continue
        for j, yj in enumerate(y):
            if yj == 0:
                continue
            e = i + j
            if e < n:
                out[e] += xi * yj
            else:
                out[e - n] -= xi * yj

    return out


# Use the direct implementation by default.
ring_mul_negacyclic = ring_mul_negacyclic_fast


def toy_ct_add(x: ToyRLWECiphertext, y: ToyRLWECiphertext) -> ToyRLWECiphertext:
    return ToyRLWECiphertext(x.a + y.a, x.b + y.b)


def toy_ct_sub(x: ToyRLWECiphertext, y: ToyRLWECiphertext) -> ToyRLWECiphertext:
    return ToyRLWECiphertext(x.a - y.a, x.b - y.b)


def toy_ct_scalar_mul(x: ToyRLWECiphertext, scalar: float) -> ToyRLWECiphertext:
    return ToyRLWECiphertext(float(scalar) * x.a, float(scalar) * x.b)


def toy_ct_monomial_mul(x: ToyRLWECiphertext, exp: int) -> ToyRLWECiphertext:
    return ToyRLWECiphertext(
        monomial_mul_negacyclic(x.a, exp),
        monomial_mul_negacyclic(x.b, exp),
    )


def toy_ct_decrypt(x: ToyRLWECiphertext, sk) -> np.ndarray:
    sk = np.asarray(sk, dtype=np.float64).reshape(-1)
    return ring_mul_negacyclic(x.a, sk) + x.b


def _deterministic_toy_key_a(n: int, tag: int) -> np.ndarray:
    """
    Deterministic non-zero toy key a-part, just to avoid degenerate a=0 keys.
    """
    a = np.zeros(n, dtype=np.float64)
    a[0] = 1.0
    if n > 1:
        a[(tag % (n - 1)) + 1] = -1.0 if tag % 2 else 1.0
    return a


def make_exact_auto_switching_keys(sk):
    """
    Build exact toy keys for all automorphisms alpha = 2j+1 mod 2N.

    For each alpha:
        key.a * sk + key.b = sigma_alpha(sk)

    This is a toy replacement for real automorphism keys.
    """
    sk = np.asarray(sk, dtype=np.float64).reshape(-1)
    n = sk.shape[0]

    keys = {}
    for j in range(n):
        alpha = 2 * j + 1
        source_sk = automorphism_negacyclic(sk, alpha)
        key_a = _deterministic_toy_key_a(n, alpha)
        key_b = source_sk - ring_mul_negacyclic(key_a, sk)
        keys[alpha % (2 * n)] = ExactToySwitchingKey(
            source_name=f"sigma_{alpha}(sk)",
            a=key_a,
            b=key_b,
        )

    return keys


def make_exact_relin_switching_key(sk) -> ExactToySwitchingKey:
    """
    Exact toy relinearization key from sk^2 to sk:

        key.a * sk + key.b = sk^2
    """
    sk = np.asarray(sk, dtype=np.float64).reshape(-1)
    n = sk.shape[0]
    source_sk2 = ring_mul_negacyclic(sk, sk)
    key_a = _deterministic_toy_key_a(n, 999)
    key_b = source_sk2 - ring_mul_negacyclic(key_a, sk)

    return ExactToySwitchingKey(
        source_name="sk^2",
        a=key_a,
        b=key_b,
    )


def toy_key_switch_exact(
    ct_under_source: ToyRLWECiphertext,
    key: ExactToySwitchingKey,
) -> ToyRLWECiphertext:
    """
    Exact toy key switching:

        ct_under_source.a * source_sk + ct_under_source.b

    with:
        source_sk = key.a * target_sk + key.b

    returns:
        (ct_under_source.a * key.a, ct_under_source.a * key.b + ct_under_source.b)
    """
    a_new = ring_mul_negacyclic(ct_under_source.a, key.a)
    b_new = ring_mul_negacyclic(ct_under_source.a, key.b) + ct_under_source.b
    return ToyRLWECiphertext(a_new, b_new)


def toy_ct_auto_exact(
    ct: ToyRLWECiphertext,
    alpha: int,
    auto_keys,
) -> ToyRLWECiphertext:
    """
    Apply Auto(ct; alpha) in the exact toy model:
      1. automorphism on both components
      2. exact key switch from sigma_alpha(sk) back to sk
    """
    n = ct.a.shape[0]
    alpha_mod = int(alpha) % (2 * n)

    if alpha_mod not in auto_keys:
        raise KeyError(f"missing exact toy auto key for alpha={alpha_mod}")

    ct_sigma = ToyRLWECiphertext(
        automorphism_negacyclic(ct.a, alpha_mod),
        automorphism_negacyclic(ct.b, alpha_mod),
    )

    return toy_key_switch_exact(ct_sigma, auto_keys[alpha_mod])


def tweak_ciphertexts_toy(cts):
    """
    Ciphertext-pair version of Algorithm 1 Tweak.

    Output:
        ct'_j = sum_i X^(2ijN/n) * ct_i
    """
    cts = list(cts)
    n = len(cts)
    if n == 0:
        raise ValueError("tweak_ciphertexts_toy(): empty input")

    _check_power_of_two(n)

    ring_n = cts[0].a.shape[0]
    if ring_n % n != 0:
        raise ValueError(f"ring degree N={ring_n} must be divisible by tweak n={n}")

    for i, ct in enumerate(cts):
        if ct.a.shape[0] != ring_n:
            raise ValueError(f"cts[{i}] ring degree mismatch")

    factor = ring_n // n
    out = []

    for j in range(n):
        acc = ToyRLWECiphertext(
            np.zeros(ring_n, dtype=np.float64),
            np.zeros(ring_n, dtype=np.float64),
        )

        for i, ct in enumerate(cts):
            exp = 2 * i * j * factor
            acc = toy_ct_add(acc, toy_ct_monomial_mul(ct, exp))

        out.append(acc)

    return out


def transpose_algorithm2_ciphertexts_toy(
    cts,
    auto_keys,
    *,
    return_trace: bool = False,
):
    """
    Exact toy RLWE component implementation of paper Algorithm 2.

    Input:
        cts[i] decrypts the i-th row, or the i-th column, depending on caller.

    Output:
        transposed ciphertext list.

    This is still a toy:
        - exact switching keys
        - no RNS
        - no gadget decomposition
        - no CKKS scale/modulus management
    """
    cts = list(cts)
    N = len(cts)
    if N == 0:
        raise ValueError("transpose_algorithm2_ciphertexts_toy(): empty input")
    _check_power_of_two(N)

    for i, ct in enumerate(cts):
        if ct.a.shape[0] != N:
            raise ValueError(f"ct[{i}] has ring degree {ct.a.shape[0]}, expected {N}")

    # Algorithm 2 line 1.
    x_i_cts = [
        toy_ct_monomial_mul(cts[i], i)
        for i in range(N)
    ]
    aux = tweak_ciphertexts_toy(x_i_cts)

    # Algorithm 2 lines 2-5.
    aux_prime = []
    for j in range(N):
        idx = _odd_inverse_index_for_algorithm2(j, N)
        odd = 2 * j + 1

        scaled = toy_ct_scalar_mul(aux[idx], 1.0 / float(N))
        autoed = toy_ct_auto_exact(scaled, odd, auto_keys)
        aux_prime.append(autoed)

    # Algorithm 2 line 6.
    ct_double_prime = tweak_ciphertexts_toy(aux_prime)

    # Algorithm 2 lines 7-9, with the same wraparound convention as plaintext toy.
    out = [None for _ in range(N)]

    for j in range(1, N + 1):
        src = (N - j) % N
        dst = j % N

        if j == N:
            corrected = ct_double_prime[src]
        else:
            corrected = toy_ct_scalar_mul(
                toy_ct_monomial_mul(ct_double_prime[src], N - j),
                -1.0,
            )

        out[dst] = corrected

    if not return_trace:
        return out

    trace = {
        "input_messages": np.vstack([toy_ct_decrypt(ct, np.zeros(N)) for ct in cts]).tolist()
        if False else "omitted_without_sk",
        "num_inputs": N,
        "ring_degree": N,
    }
    return out, trace
