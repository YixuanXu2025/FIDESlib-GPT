import json
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.pcmm_baseline import (
    pcmm_square_plain_reference,
    pcmm_square_python_baseline,
)


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


def bench_plain_reference(x_block, w_block, *, split_dim: int, repeats: int = 5):
    """
    计时 plain reference：
      Y = sum_s X_s @ W_s
    这里不是 PCMM，只是 block matmul 的明文参考实现。
    """
    times = []
    y_last = None

    for _ in range(repeats):
        t0 = now_us()
        y = pcmm_square_plain_reference(
            x_block,
            w_block,
            split_dim=split_dim,
        )
        t1 = now_us()
        times.append(t1 - t0)
        y_last = y

    return {
        "plain_reference_us_median": median(times),
        "y_plain_reference": y_last,
    }


def run_case(case_name: str,
             rows: int,
             mid_dim: int,
             out_dim: int,
             split_dim: int,
             he_repeats: int = 1,
             plain_repeats: int = 5,
             seed: int = 0):
    rng = np.random.default_rng(seed)

    x_block = rng.normal(0.0, 1.0, size=(rows, mid_dim)).astype(np.float64)
    w_block = rng.normal(0.0, 0.2, size=(mid_dim, out_dim)).astype(np.float64)

    # plain reference timing
    plain_result = bench_plain_reference(
        x_block,
        w_block,
        split_dim=split_dim,
        repeats=plain_repeats,
    )

    # HE baseline timing
    # 这里 physical_slots 先取 out_dim，保证输出能放入 slots
    physical_slots = out_dim

    # 旋转 key：
    #   reduce: 1 .. split_dim-1
    #   place : -(out_dim-1) .. 0
    rotation_steps = list(range(1, max(split_dim, 1))) + list(range(-(max(out_dim, 1) - 1), 0))

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

    with HERuntime(cfg, rotation_steps=rotation_steps) as rt:
        he_result = pcmm_square_python_baseline(
            rt,
            x_block,
            w_block,
            split_dim=split_dim,
            physical_slots=physical_slots,
            repeats=he_repeats,
            seed=seed,
        )

    # 对齐引用名：这里明确叫 plain reference，而不是 PCMM
    y_plain = plain_result.pop("y_plain_reference")
    y_he = he_result.pop("y_he")
    y_ref = he_result.pop("y_ref")

    result = {
        "case_name": case_name,
        "rows": rows,
        "mid_dim": mid_dim,
        "out_dim": out_dim,
        "split_dim": split_dim,
        "physical_slots": physical_slots,
        "plain_repeats": plain_repeats,
        "he_repeats": he_repeats,
        **plain_result,
        **he_result,
    }

    # 两类误差都保留：
    # 1) plain reference vs y_ref（应为 0）
    # 2) plain reference vs y_he
    result["plain_vs_ref_max_abs_err"] = float(np.max(np.abs(y_plain - y_ref)))
    result["plain_vs_he_max_abs_err"] = float(np.max(np.abs(y_plain - y_he)))

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

    # 只保存第一行，避免 JSON 太大
    result["y_plain_first_row"] = y_plain[0].tolist()
    result["y_he_first_row"] = y_he[0].tolist()

    return result


def main():
    print_section("WP3-A Compare / Case 1: square input")
    # 方阵输入：X = 8x8, W = 8x8
    case1 = run_case(
        case_name="square_input_8x8__8x8",
        rows=8,
        mid_dim=8,
        out_dim=8,
        split_dim=4,
        he_repeats=1,
        plain_repeats=5,
        seed=202701,
    )
    print_dict(case1, prefix="  ")

    print_section("WP3-A Compare / Case 2: non-square input")
    # 非方阵输入：X = 4x8, W = 8x8
    case2 = run_case(
        case_name="rect_input_4x8__8x8",
        rows=4,
        mid_dim=8,
        out_dim=8,
        split_dim=4,
        he_repeats=1,
        plain_repeats=5,
        seed=202702,
    )
    print_dict(case2, prefix="  ")

    print_section("WP3-A Compare / Case 3: non-square output")
    # 输出非方阵：X = 8x8, W = 8x16
    case3 = run_case(
        case_name="square_input_8x8__8x16",
        rows=8,
        mid_dim=8,
        out_dim=16,
        split_dim=4,
        he_repeats=1,
        plain_repeats=5,
        seed=202703,
    )
    print_dict(case3, prefix="  ")

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3a_pcmm_compare_report.json"

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
