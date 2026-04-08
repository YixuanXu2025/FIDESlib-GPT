import json
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.ops import he_add
from hegpt.pcmm_baseline import (
    he_encrypt_vector_padded,
    he_linear_plain_padded,
    pcmm_square_plain_reference,
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


def preencrypt_row_segments(rt, x_block, *, split_dim: int, physical_slots: int):
    """
    一次性把所有 row / split 的输入段都加密好。
    返回：
      encrypted[r][s] = CipherTensor
    """
    x_block = np.asarray(x_block, dtype=np.float64)
    rows, mid_dim = x_block.shape

    if mid_dim % split_dim != 0:
        raise ValueError("mid_dim must be divisible by split_dim")

    num_splits = mid_dim // split_dim
    encrypted = []

    t0 = now_us()
    for r in range(rows):
        row_list = []
        for s in range(num_splits):
            c0 = s * split_dim
            c1 = (s + 1) * split_dim
            x_seg = x_block[r, c0:c1]

            ct_x = he_encrypt_vector_padded(
                rt,
                x_seg,
                physical_slots=physical_slots,
                name=f"x_r{r}_s{s}",
            )
            row_list.append(ct_x)
        encrypted.append(row_list)
    t1 = now_us()

    return encrypted, (t1 - t0)


def pcmm_square_python_cached_input(
    rt: HERuntime,
    encrypted_rows,
    w_block,
    *,
    split_dim: int,
    out_dim: int,
):
    """
    使用“已缓存的输入密文段”来执行 Python 版 PCMM baseline。

    这一步只测：
      - core（所有行、所有 split 的 HE 线性层与密文累加）
      - decrypt
    不再重复计入 encrypt。
    """
    w_block = np.asarray(w_block, dtype=np.float64)
    mid_dim, out_dim_w = w_block.shape
    if out_dim_w != out_dim:
        raise ValueError("out_dim mismatch")

    num_splits = mid_dim // split_dim
    rows = len(encrypted_rows)

    row_outputs = []

    t_core_0 = now_us()
    for r in range(rows):
        acc_ct = None

        for s in range(num_splits):
            c0 = s * split_dim
            c1 = (s + 1) * split_dim
            w_seg = w_block[c0:c1, :]

            ct_x = encrypted_rows[r][s]
            ct_part = he_linear_plain_padded(
                rt,
                ct_x,
                w_seg,
                bias=None,
                name=f"y_r{r}_s{s}",
            )
            acc_ct = ct_part if acc_ct is None else he_add(
                rt,
                acc_ct,
                ct_part,
                name=f"y_r{r}_acc",
            )

        row_outputs.append(acc_ct)
    t_core_1 = now_us()

    t_dec_0 = now_us()
    y_rows = []
    for r in range(rows):
        y_r = np.array(
            rt.decrypt(row_outputs[r].ct, logical_length=out_dim),
            dtype=np.float64,
        )[:out_dim]
        y_rows.append(y_r)
    t_dec_1 = now_us()

    y_he = np.stack(y_rows, axis=0)

    return {
        "y_he": y_he,
        "core_us": t_core_1 - t_core_0,
        "decrypt_us": t_dec_1 - t_dec_0,
    }


def run_case(case_name: str,
             rows: int,
             mid_dim: int,
             out_dim: int,
             split_dim: int,
             plain_repeats: int = 5,
             cached_repeats: int = 3,
             seed: int = 0):
    rng = np.random.default_rng(seed)

    x_block = rng.normal(0.0, 1.0, size=(rows, mid_dim)).astype(np.float64)
    w_block = rng.normal(0.0, 0.2, size=(mid_dim, out_dim)).astype(np.float64)

    # 明文参考
    plain_result = bench_plain_reference(
        x_block,
        w_block,
        split_dim=split_dim,
        repeats=plain_repeats,
    )
    y_plain = plain_result.pop("y_plain_reference")

    physical_slots = out_dim
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

    encrypt_once_list = []
    core_list = []
    decrypt_list = []
    total_list = []
    err_list = []
    y_he_last = None

    with HERuntime(cfg, rotation_steps=rotation_steps) as rt:
        for _ in range(cached_repeats):
            encrypted_rows, encrypt_once_us = preencrypt_row_segments(
                rt,
                x_block,
                split_dim=split_dim,
                physical_slots=physical_slots,
            )

            t_total_0 = now_us()
            run_result = pcmm_square_python_cached_input(
                rt,
                encrypted_rows,
                w_block,
                split_dim=split_dim,
                out_dim=out_dim,
            )
            t_total_1 = now_us()

            y_he = run_result["y_he"]
            y_he_last = y_he

            encrypt_once_list.append(encrypt_once_us)
            core_list.append(run_result["core_us"])
            decrypt_list.append(run_result["decrypt_us"])
            total_list.append((t_total_1 - t_total_0) + encrypt_once_us)
            err_list.append(float(np.max(np.abs(y_plain - y_he))))

    result = {
        "case_name": case_name,
        "rows": rows,
        "mid_dim": mid_dim,
        "out_dim": out_dim,
        "split_dim": split_dim,
        "physical_slots": physical_slots,
        "plain_repeats": plain_repeats,
        "cached_repeats": cached_repeats,
        **plain_result,
        "encrypt_once_us_median": median(encrypt_once_list),
        "core_us_median": median(core_list),
        "decrypt_us_median": median(decrypt_list),
        "total_us_median": median(total_list),
        "plain_vs_he_max_abs_err": max(err_list) if err_list else None,
        "y_plain_first_row": y_plain[0].tolist(),
        "y_he_first_row": y_he_last[0].tolist(),
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

    if result["plain_reference_us_median"] and result["encrypt_once_us_median"]:
        result["encrypt_once_vs_plain_ratio"] = (
            result["encrypt_once_us_median"] / result["plain_reference_us_median"]
        )
    else:
        result["encrypt_once_vs_plain_ratio"] = None

    return result


def main():
    print_section("WP3-A Cached / Case 1: square input")
    case1 = run_case(
        case_name="square_input_8x8__8x8_cached",
        rows=8,
        mid_dim=8,
        out_dim=8,
        split_dim=4,
        plain_repeats=5,
        cached_repeats=3,
        seed=202801,
    )
    print_dict(case1, prefix="  ")

    print_section("WP3-A Cached / Case 2: non-square input")
    case2 = run_case(
        case_name="rect_input_4x8__8x8_cached",
        rows=4,
        mid_dim=8,
        out_dim=8,
        split_dim=4,
        plain_repeats=5,
        cached_repeats=3,
        seed=202802,
    )
    print_dict(case2, prefix="  ")

    print_section("WP3-A Cached / Case 3: non-square output")
    case3 = run_case(
        case_name="square_input_8x8__8x16_cached",
        rows=8,
        mid_dim=8,
        out_dim=16,
        split_dim=4,
        plain_repeats=5,
        cached_repeats=3,
        seed=202803,
    )
    print_dict(case3, prefix="  ")

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3a_pcmm_cached_compare_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "case_1_square_input_cached": case1,
                "case_2_non_square_input_cached": case2,
                "case_3_non_square_output_cached": case3,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_section("Report Saved")
    print(f"JSON report saved to: {out_path}")


if __name__ == "__main__":
    main()
