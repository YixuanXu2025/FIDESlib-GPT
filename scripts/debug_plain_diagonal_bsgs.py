import math
import numpy as np


def print_section(title: str):
    print("=" * 100)
    print(title)


def plain_rotate(vec, steps, sigma):
    vec = np.asarray(vec, dtype=np.float64)
    return np.roll(vec, -sigma * steps)


def build_padded_square_weight(weight, L):
    weight = np.asarray(weight, dtype=np.float64)
    in_dim, out_dim = weight.shape
    Wp = np.zeros((L, L), dtype=np.float64)
    Wp[:in_dim, :out_dim] = weight
    return Wp


def build_diagonals_for_linear(weight, sigma):
    """
    与你当前 HE 脚本保持一致的 diagonal 构造：
      diag_s[j] = W_pad[(j + sigma*s) mod L, j]
    """
    weight = np.asarray(weight, dtype=np.float64)
    in_dim, out_dim = weight.shape
    L = max(in_dim, out_dim)

    Wp = build_padded_square_weight(weight, L)
    diags = []

    for s in range(L):
        d = np.zeros((L,), dtype=np.float64)
        for j in range(L):
            i = (j + sigma * s) % L
            d[j] = Wp[i, j]
        diags.append(d)

    return diags, L, Wp


def dense_reference_single_row(x_row, weight):
    x_row = np.asarray(x_row, dtype=np.float64).reshape(-1)
    weight = np.asarray(weight, dtype=np.float64)
    return x_row @ weight


def full_diagonal_plain_single_row(x_row, weight, sigma):
    x_row = np.asarray(x_row, dtype=np.float64).reshape(-1)
    weight = np.asarray(weight, dtype=np.float64)
    in_dim, out_dim = weight.shape

    diags, L, _ = build_diagonals_for_linear(weight, sigma)

    x_pad = np.zeros((L,), dtype=np.float64)
    x_pad[:in_dim] = x_row

    y_pad = np.zeros((L,), dtype=np.float64)
    for s, d in enumerate(diags):
        y_pad += plain_rotate(x_pad, s, sigma) * d

    return y_pad[:out_dim], {
        "diagonals": diags,
        "L": L,
    }


def bsgs_regrouped_plain_single_row(x_row, weight, sigma):
    """
    与当前 HE 脚本保持一致的 BSGS regrouping：
      y = sum_j rotate(
                sum_i rotate(x, i) * rotate(diag_{j*g+i}, -j*g),
              j*g)
    """
    x_row = np.asarray(x_row, dtype=np.float64).reshape(-1)
    weight = np.asarray(weight, dtype=np.float64)
    in_dim, out_dim = weight.shape

    diags, L, _ = build_diagonals_for_linear(weight, sigma)

    x_pad = np.zeros((L,), dtype=np.float64)
    x_pad[:in_dim] = x_row

    g = int(math.ceil(math.sqrt(L)))
    h = int(math.ceil(L / g))

    baby = []
    for i in range(g):
        baby.append(plain_rotate(x_pad, i, sigma))

    acc = np.zeros((L,), dtype=np.float64)

    for j in range(h):
        shift = j * g
        inner = np.zeros((L,), dtype=np.float64)

        for i in range(g):
            s = shift + i
            if s >= L:
                break

            rotated_diag = plain_rotate(diags[s], -shift, sigma)
            inner += baby[i] * rotated_diag

        shifted = plain_rotate(inner, shift, sigma) if shift != 0 else inner
        acc += shifted

    return acc[:out_dim], {
        "L": L,
        "baby_step_size": g,
        "giant_step_count": h,
        "diagonals": diags,
    }


def first_mismatch(a, b, atol=1e-10):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = np.abs(a - b)
    idx = np.where(diff > atol)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    return {
        "index": i,
        "a": float(a[i]),
        "b": float(b[i]),
        "abs_diff": float(diff[i]),
    }


def run_case(case_name, rows, mid_dim, out_dim, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(rows, mid_dim)).astype(np.float64)
    w = rng.normal(0.0, 0.2, size=(mid_dim, out_dim)).astype(np.float64)

    print_section(case_name)
    print(f"shape: X=({rows},{mid_dim}), W=({mid_dim},{out_dim})")

    # 只看第一行，先把 pure-plain 关系定位清楚
    x_row = x[0]
    y_dense = dense_reference_single_row(x_row, w)

    for sigma in (+1, -1):
        print(f"-- sigma = {sigma} --")

        y_full, meta_full = full_diagonal_plain_single_row(x_row, w, sigma=sigma)
        y_bsgs, meta_bsgs = bsgs_regrouped_plain_single_row(x_row, w, sigma=sigma)

        err_full_vs_dense = float(np.max(np.abs(y_full - y_dense)))
        err_bsgs_vs_dense = float(np.max(np.abs(y_bsgs - y_dense)))
        err_bsgs_vs_full = float(np.max(np.abs(y_bsgs - y_full)))

        print(f"  L = {meta_full['L']}")
        print(f"  BSGS meta = {{baby_step_size: {meta_bsgs['baby_step_size']}, giant_step_count: {meta_bsgs['giant_step_count']}}}")
        print(f"  full_vs_dense_max_abs_err = {err_full_vs_dense}")
        print(f"  bsgs_vs_dense_max_abs_err = {err_bsgs_vs_dense}")
        print(f"  bsgs_vs_full_max_abs_err = {err_bsgs_vs_full}")

        mm1 = first_mismatch(y_full, y_dense)
        mm2 = first_mismatch(y_bsgs, y_dense)
        mm3 = first_mismatch(y_bsgs, y_full)

        print(f"  first_mismatch(full, dense) = {mm1}")
        print(f"  first_mismatch(bsgs, dense) = {mm2}")
        print(f"  first_mismatch(bsgs, full)  = {mm3}")

        print(f"  y_dense first row = {y_dense.tolist()}")
        print(f"  y_full  first row = {y_full.tolist()}")
        print(f"  y_bsgs  first row = {y_bsgs.tolist()}")


def main():
    run_case(
        case_name="Case 1: square_input_8x8__8x8",
        rows=8,
        mid_dim=8,
        out_dim=8,
        seed=202901,
    )

    run_case(
        case_name="Case 2: rect_input_4x8__8x8",
        rows=4,
        mid_dim=8,
        out_dim=8,
        seed=202902,
    )

    run_case(
        case_name="Case 3: square_input_8x8__8x16",
        rows=8,
        mid_dim=8,
        out_dim=16,
        seed=202903,
    )


if __name__ == "__main__":
    main()
