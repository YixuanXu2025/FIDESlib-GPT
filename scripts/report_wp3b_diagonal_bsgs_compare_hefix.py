import json
import math
import time
from pathlib import Path

import numpy as np
from _fideslib import FidesCKKSContext


def now_us():
    return time.perf_counter() * 1e6


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def print_section(title: str):
    print("=" * 100)
    print(title)


def print_dict(d: dict, prefix: str = ""):
    for k, v in d.items():
        print(f"{prefix}{k}: {v}")


def load_naive_report():
    report_path = Path("/workspace/FIDESlib-GPT/reports/wp3a_pcmm_compare_report.json")
    if not report_path.exists():
        return {}
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_case = {}
    for _, item in data.items():
        case_name = item.get("case_name")
        if case_name is not None:
            by_case[case_name] = item
    return by_case


def plain_rotate(vec, steps, sigma):
    vec = np.asarray(vec, dtype=np.float64)
    return np.roll(vec, -sigma * steps)


def detect_rotation_sigma(ctx):
    x = [0.0, 1.0, 2.0, 3.0]
    ct = ctx.encrypt(x)
    ct_r = ctx.eval_rotate_ct(ct, 1)
    out = np.array(ctx.decrypt(ct_r, logical_length=4), dtype=np.float64)

    left = np.array([1.0, 2.0, 3.0, 0.0], dtype=np.float64)
    right = np.array([3.0, 0.0, 1.0, 2.0], dtype=np.float64)

    if np.max(np.abs(out - left)) < 1e-6:
        return +1
    if np.max(np.abs(out - right)) < 1e-6:
        return -1
    raise RuntimeError(f"Unable to detect rotation direction, got {out.tolist()}")


def build_padded_square_weight(weight, L):
    weight = np.asarray(weight, dtype=np.float64)
    in_dim, out_dim = weight.shape
    Wp = np.zeros((L, L), dtype=np.float64)
    Wp[:in_dim, :out_dim] = weight
    return Wp


def build_diagonals_for_linear(weight, sigma):
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
    return diags, L


def periodic_pack(vec, physical_slots):
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    L = len(vec)
    if physical_slots % L != 0:
        raise ValueError(f"physical_slots={physical_slots} must be divisible by L={L}")
    repeat = physical_slots // L
    return np.tile(vec, repeat)


def build_periodic_diagonals(weight, sigma, physical_slots):
    diags, L = build_diagonals_for_linear(weight, sigma)
    periodic_diags = [periodic_pack(d, physical_slots) for d in diags]
    return periodic_diags, L


def plain_dense_reference(x_block, w_block):
    return np.asarray(x_block, dtype=np.float64) @ np.asarray(w_block, dtype=np.float64)


def bench_plain_dense_reference(x_block, w_block, repeats=5):
    times = []
    y_last = None
    for _ in range(repeats):
        t0 = now_us()
        y = plain_dense_reference(x_block, w_block)
        t1 = now_us()
        times.append(t1 - t0)
        y_last = y
    return {
        "plain_reference_us_median": median(times),
        "y_plain_reference": y_last,
    }


def diagonal_bsgs_op_counts(L, rows):
    g = int(math.ceil(math.sqrt(L)))
    h = int(math.ceil(L / g))

    baby_rotates = max(g - 1, 0)
    mult_plain = L

    inner_adds = 0
    for j in range(h):
        shift = j * g
        terms = max(0, min(g, L - shift))
        inner_adds += max(terms - 1, 0)

    giant_rotates = max(h - 1, 0)
    outer_adds = max(h - 1, 0)

    per_row = {
        "mult_plain": mult_plain,
        "rotate": baby_rotates + giant_rotates,
        "add": inner_adds + outer_adds,
        "add_plain": 0,
    }
    total = {k: rows * v for k, v in per_row.items()}
    return {
        "L": L,
        "baby_step_size": g,
        "giant_step_count": h,
        "per_row_op_counts": per_row,
        "total_op_counts": total,
    }


def encrypt_periodic_row(ctx, x_row, L, physical_slots):
    x_row = np.asarray(x_row, dtype=np.float64).reshape(-1)
    if len(x_row) != L and len(x_row) > L:
        raise ValueError("row length exceeds L")
    base = np.zeros((L,), dtype=np.float64)
    base[: len(x_row)] = x_row
    packed = periodic_pack(base, physical_slots)
    return ctx.encrypt(packed.tolist())


def he_diagonal_bsgs_single_row_periodic(ctx, x_ct, periodic_diags, sigma):
    """
    periodic packing 版：
      - x 按长度 L 周期复制到全部 physical_slots
      - diagonal 也按长度 L 周期复制
      - HE rotate 虽然是全 slots 旋转，但等价于长度 L 的循环旋转
    """
    L = len(periodic_diags)
    physical_slots = len(periodic_diags[0])

    g = int(math.ceil(math.sqrt(L)))
    h = int(math.ceil(L / g))

    baby = []
    for i in range(g):
        if i == 0:
            baby.append(x_ct)
        else:
            baby.append(ctx.eval_rotate_ct(x_ct, i))

    acc = None
    for j in range(h):
        shift = j * g
        inner = None

        for i in range(g):
            s = shift + i
            if s >= L:
                break

            shifted_diag = plain_rotate(periodic_diags[s], -shift, sigma).tolist()
            term = ctx.eval_mult_plain_ct(baby[i], shifted_diag)
            inner = term if inner is None else ctx.eval_add_ct(inner, term)

        if inner is None:
            continue

        shifted = inner if shift == 0 else ctx.eval_rotate_ct(inner, shift)
        acc = shifted if acc is None else ctx.eval_add_ct(acc, shifted)

    return acc, {
        "L": L,
        "physical_slots": physical_slots,
        "baby_step_size": g,
        "giant_step_count": h,
    }


def run_case(case_name: str,
             rows: int,
             mid_dim: int,
             out_dim: int,
             plain_repeats: int = 5,
             he_repeats: int = 1,
             seed: int = 0,
             naive_by_case=None):
    if naive_by_case is None:
        naive_by_case = {}

    rng = np.random.default_rng(seed)
    x_block = rng.normal(0.0, 1.0, size=(rows, mid_dim)).astype(np.float64)
    w_block = rng.normal(0.0, 0.2, size=(mid_dim, out_dim)).astype(np.float64)

    plain_result = bench_plain_dense_reference(x_block, w_block, repeats=plain_repeats)
    y_plain = plain_result.pop("y_plain_reference")

    ctx = FidesCKKSContext()
    ctx.init(
        multiplicative_depth=2,
        scaling_mod_size=50,
        batch_size=8192,
        ring_dim=(1 << 14),
        devices=[0],
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
        rotation_steps=list(range(-64, 65)),
    )

    encrypt_list = []
    core_list = []
    decrypt_list = []
    total_list = []
    err_list = []
    sigma_last = None
    bsgs_meta_last = None
    y_he_last = None
    op_meta = None

    try:
        sigma = detect_rotation_sigma(ctx)
        sigma_last = sigma

        slots_per_ct = ctx.info()["ring_dim"] // 2
        periodic_diags, L = build_periodic_diagonals(w_block, sigma=sigma, physical_slots=slots_per_ct)
        op_meta = diagonal_bsgs_op_counts(L=L, rows=rows)

        for _ in range(he_repeats):
            t0 = now_us()
            row_cts_in = []
            for r in range(rows):
                row_cts_in.append(encrypt_periodic_row(ctx, x_block[r], L, slots_per_ct))
            t1 = now_us()

            row_cts_out = []
            t2 = now_us()
            for r in range(rows):
                ct_y_r, bsgs_meta = he_diagonal_bsgs_single_row_periodic(
                    ctx,
                    row_cts_in[r],
                    periodic_diags,
                    sigma=sigma,
                )
                row_cts_out.append(ct_y_r)
                bsgs_meta_last = bsgs_meta
            t3 = now_us()

            y_rows = []
            t4 = now_us()
            for r in range(rows):
                full_dec = np.array(
                    ctx.decrypt(row_cts_out[r], logical_length=slots_per_ct),
                    dtype=np.float64,
                )
                y_rows.append(full_dec[:out_dim])
            t5 = now_us()

            y_he = np.stack(y_rows, axis=0)
            y_he_last = y_he

            encrypt_list.append(t1 - t0)
            core_list.append(t3 - t2)
            decrypt_list.append(t5 - t4)
            total_list.append((t1 - t0) + (t3 - t2) + (t5 - t4))
            err_list.append(float(np.max(np.abs(y_plain - y_he))))

    finally:
        try:
            ctx.close()
        except Exception:
            pass

    result = {
        "case_name": case_name,
        "rows": rows,
        "mid_dim": mid_dim,
        "out_dim": out_dim,
        "plain_repeats": plain_repeats,
        "he_repeats": he_repeats,
        **plain_result,
        "rotation_sigma": sigma_last,
        "encrypt_us_median": median(encrypt_list),
        "core_us_median": median(core_list),
        "decrypt_us_median": median(decrypt_list),
        "total_us_median": median(total_list),
        "plain_vs_he_max_abs_err": max(err_list) if err_list else None,
        "y_plain_first_row": y_plain[0].tolist(),
        "y_he_first_row": y_he_last[0].tolist(),
        "bsgs_meta": bsgs_meta_last,
        "op_meta": op_meta,
    }

    if result["plain_reference_us_median"] and result["core_us_median"]:
        result["he_core_vs_plain_ratio"] = (
            result["core_us_median"] / result["plain_reference_us_median"]
        )
    else:
        result["he_core_vs_plain_ratio"] = None

    if result["plain_reference_us_median"] and result["total_us_median"]:
        result["he_total_vs_plain_ratio"] = (
            result["total_us_median"] / result["plain_reference_us_median"]
        )
    else:
        result["he_total_vs_plain_ratio"] = None

    naive = naive_by_case.get(case_name)
    if naive is not None:
        result["naive_core_us_from_report"] = naive.get("core_us_median")
        result["naive_total_us_from_report"] = naive.get("total_us_median")

        if naive.get("core_us_median") and result["core_us_median"]:
            result["bsgs_speedup_over_naive_core"] = (
                naive["core_us_median"] / result["core_us_median"]
            )
        else:
            result["bsgs_speedup_over_naive_core"] = None

        if naive.get("total_us_median") and result["total_us_median"]:
            result["bsgs_speedup_over_naive_total"] = (
                naive["total_us_median"] / result["total_us_median"]
            )
        else:
            result["bsgs_speedup_over_naive_total"] = None

    return result


def main():
    naive_by_case = load_naive_report()

    print_section("HE-FIX / Case 1: square input")
    case1 = run_case(
        case_name="square_input_8x8__8x8",
        rows=8,
        mid_dim=8,
        out_dim=8,
        plain_repeats=5,
        he_repeats=1,
        seed=202901,
        naive_by_case=naive_by_case,
    )
    print_dict(case1, prefix="  ")

    print_section("HE-FIX / Case 2: non-square input")
    case2 = run_case(
        case_name="rect_input_4x8__8x8",
        rows=4,
        mid_dim=8,
        out_dim=8,
        plain_repeats=5,
        he_repeats=1,
        seed=202902,
        naive_by_case=naive_by_case,
    )
    print_dict(case2, prefix="  ")

    print_section("HE-FIX / Case 3: non-square output")
    case3 = run_case(
        case_name="square_input_8x8__8x16",
        rows=8,
        mid_dim=8,
        out_dim=16,
        plain_repeats=5,
        he_repeats=1,
        seed=202903,
        naive_by_case=naive_by_case,
    )
    print_dict(case3, prefix="  ")

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3b_diagonal_bsgs_compare_hefix_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "case_1_square_input": case1,
                "case_2_non_square_input": case2,
                "case_3_non_square_output": case3,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_section("Report Saved")
    print(f"JSON report saved to: {out_path}")


if __name__ == "__main__":
    main()
