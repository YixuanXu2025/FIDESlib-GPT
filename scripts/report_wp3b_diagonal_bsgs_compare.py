import json
import math
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.pcmm_baseline import he_encrypt_vector_padded
from hegpt.ops import he_add, he_mult_plain, he_rotate


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


def plain_dense_reference(x_block, w_block):
    x_block = np.asarray(x_block, dtype=np.float64)
    w_block = np.asarray(w_block, dtype=np.float64)
    return x_block @ w_block


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


def load_naive_report():
    """
    读取你前面已经跑过的 naive baseline 报告。
    这样当前脚本可以直接给出：
      diagonal+BSGS vs naive 的实测速度对比
    """
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
    """
    用 sigma 描述 rotate 方向：
      sigma = +1  : 正 steps 表示左旋
      sigma = -1  : 正 steps 表示右旋
    """
    vec = np.asarray(vec, dtype=np.float64)
    return np.roll(vec, -sigma * steps)


def detect_rotation_sigma(rt):
    """
    自动检测 he_rotate(rt, x, +1) 的方向。
    返回：
      sigma = +1  => 正步数是左旋
      sigma = -1  => 正步数是右旋
    """
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ct_x = he_encrypt_vector_padded(
        rt,
        x,
        physical_slots=4,
        name="rot_detect",
    )
    ct_r = he_rotate(rt, ct_x, 1)
    out = np.array(rt.decrypt(ct_r.ct, logical_length=4), dtype=np.float64)

    left = np.array([1.0, 2.0, 3.0, 0.0], dtype=np.float64)
    right = np.array([3.0, 0.0, 1.0, 2.0], dtype=np.float64)

    if np.max(np.abs(out - left)) < 1e-6:
        return +1
    if np.max(np.abs(out - right)) < 1e-6:
        return -1

    raise RuntimeError(
        f"Unable to detect rotation direction. rotate(+1) output = {out.tolist()}"
    )


def build_padded_square_weight(weight, L):
    weight = np.asarray(weight, dtype=np.float64)
    in_dim, out_dim = weight.shape

    Wp = np.zeros((L, L), dtype=np.float64)
    Wp[:in_dim, :out_dim] = weight
    return Wp


def build_diagonals_for_linear(weight, sigma):
    """
    为 row-vector x @ W 构造 diagonal method 所需的 diagonals。

    设：
      x in R^{in_dim}
      W in R^{in_dim x out_dim}
      L = max(in_dim, out_dim)
      x / W 都 padding 到 L

    若 he_rotate 的正方向由 sigma 描述，则第 s 条 diagonal 定义为：
      diag_s[j] = W_pad[(j + sigma*s) mod L, j]

    这样：
      y = sum_s rotate(x, s) .* diag_s
    在前 out_dim 个槽位上等于 x @ W
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
    return diags, L


def plain_diagonal_reference_single_row(x_row, weight, sigma):
    """
    用“相同 diagonals + 相同 rotate 方向”的纯明文实现验证 diagonal method 正确性。
    """
    x_row = np.asarray(x_row, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    in_dim, out_dim = weight.shape
    diags, L = build_diagonals_for_linear(weight, sigma)

    x_pad = np.zeros((L,), dtype=np.float64)
    x_pad[:in_dim] = x_row

    y_pad = np.zeros((L,), dtype=np.float64)
    for s, d in enumerate(diags):
        y_pad += plain_rotate(x_pad, s, sigma) * d

    return y_pad[:out_dim]


def he_diagonal_bsgs_single_row(
    rt,
    x_ct,
    diagonals,
    *,
    out_dim: int,
    sigma: int,
):
    """
    单行输入的 diagonal method + BSGS baseline。

    目标：
      y = sum_s rotate(x, s) .* diag_s

    BSGS 分解：
      s = j*g + i
      y = sum_j rotate(
                sum_i rotate(x, i) .* rotate(diag_{j*g+i}, -j*g),
              j*g)

    注意：
      这是 Python baseline：
      - correctness first
      - structure first
      - not final high-performance kernel
    """
    L = len(diagonals)
    g = int(math.ceil(math.sqrt(L)))
    h = int(math.ceil(L / g))

    # baby-step rotations: rotate(x, i)
    baby = []
    for i in range(g):
        if i == 0:
            baby.append(x_ct)
        else:
            baby.append(he_rotate(rt, x_ct, i))

    acc = None

    for j in range(h):
        shift = j * g
        inner = None

        for i in range(g):
            s = shift + i
            if s >= L:
                break

            # 为了让外层 giant-step rotate 正确复原，
            # 需要先把 diagonal 反向平移 -shift
            plain_diag = plain_rotate(diagonals[s], -shift, sigma)
            term = he_mult_plain(rt, baby[i], plain_diag.tolist())

            inner = term if inner is None else he_add(rt, inner, term)

        if inner is None:
            continue

        shifted = inner if shift == 0 else he_rotate(rt, inner, shift)
        acc = shifted if acc is None else he_add(rt, acc, shifted)

    return acc, {
        "L": L,
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

    # 明文 dense reference
    plain_result = bench_plain_dense_reference(
        x_block,
        w_block,
        repeats=plain_repeats,
    )
    y_plain = plain_result.pop("y_plain_reference")

    cfg = HEConfig(
        ring_dim=16384,
        multiplicative_depth=2,
        scaling_mod_size=50,
        batch_size=8192,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    encrypt_list = []
    core_list = []
    decrypt_list = []
    total_list = []
    err_list = []
    y_he_last = None
    sigma_last = None
    bsgs_meta_last = None

    with HERuntime(cfg, rotation_steps=list(range(-64, 65))) as rt:
        sigma = detect_rotation_sigma(rt)
        sigma_last = sigma

        # 先验证 diagonal 构造本身在明文世界里是正确的
        y_diag_plain_check = plain_diagonal_reference_single_row(
            x_block[0],
            w_block,
            sigma=sigma,
        )
        check_err = float(np.max(np.abs(y_diag_plain_check - y_plain[0])))

        if check_err > 1e-8:
            raise RuntimeError(
                f"Diagonal reference construction seems wrong, max err = {check_err}"
            )

        diagonals, L = build_diagonals_for_linear(w_block, sigma=sigma)

        for _ in range(he_repeats):
            # encrypt rows once
            t0 = now_us()
            encrypted_rows = []
            for r in range(rows):
                ct_x = he_encrypt_vector_padded(
                    rt,
                    x_block[r],
                    physical_slots=L,
                    name=f"x_r{r}",
                )
                encrypted_rows.append(ct_x)
            t1 = now_us()

            # core
            row_cts = []
            t2 = now_us()
            for r in range(rows):
                ct_y_r, meta = he_diagonal_bsgs_single_row(
                    rt,
                    encrypted_rows[r],
                    diagonals,
                    out_dim=out_dim,
                    sigma=sigma,
                )
                row_cts.append(ct_y_r)
                bsgs_meta_last = meta
            t3 = now_us()

            # decrypt
            y_rows = []
            t4 = now_us()
            for r in range(rows):
                y_r = np.array(
                    rt.decrypt(row_cts[r].ct, logical_length=out_dim),
                    dtype=np.float64,
                )[:out_dim]
                y_rows.append(y_r)
            t5 = now_us()

            y_he = np.stack(y_rows, axis=0)
            y_he_last = y_he

            encrypt_list.append(t1 - t0)
            core_list.append(t3 - t2)
            decrypt_list.append(t5 - t4)
            total_list.append((t1 - t0) + (t3 - t2) + (t5 - t4))
            err_list.append(float(np.max(np.abs(y_plain - y_he))))

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

    # 若已有 naive 报告，则自动给出对比
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

    print_section("WP3-B Diagonal+BSGS / Case 1: square input")
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

    print_section("WP3-B Diagonal+BSGS / Case 2: non-square input")
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

    print_section("WP3-B Diagonal+BSGS / Case 3: non-square output")
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
    out_path = out_dir / "wp3b_diagonal_bsgs_compare_report.json"

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
