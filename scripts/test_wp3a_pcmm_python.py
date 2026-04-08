import json
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.pcmm_baseline import (
    pcmm_square_plain_reference,
    pcmm_square_python_baseline,
)


def print_section(title: str):
    print("=" * 100)
    print(title)


def print_dict(d: dict, prefix: str = ""):
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(f"{prefix}{k}: ndarray(shape={v.shape})")
        else:
            print(f"{prefix}{k}: {v}")


def run_case(block_rows: int, mid_dim: int, out_dim: int, split_dim: int, repeats: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)

    x_block = rng.normal(0.0, 1.0, size=(block_rows, mid_dim)).astype(np.float64)
    w_block = rng.normal(0.0, 0.2, size=(mid_dim, out_dim)).astype(np.float64)

    # 为了让 he_linear_plain_padded 能把 out_dim 个输出都放进槽位里，
    # 这里先把 physical_slots 设成 out_dim。
    physical_slots = out_dim

    # 旋转 key 需求：
    #   - reduce: 1 .. split_dim-1
    #   - place : -(out_dim-1) .. 0
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
        result = pcmm_square_python_baseline(
            rt,
            x_block,
            w_block,
            split_dim=split_dim,
            physical_slots=physical_slots,
            repeats=repeats,
            seed=seed,
        )

    # 为了报告更紧凑，不把完整矩阵打印到屏幕
    screen_result = dict(result)
    y_ref = screen_result.pop("y_ref")
    y_he = screen_result.pop("y_he")

    print_dict(screen_result, prefix="  ")
    print("  y_ref first row first 8:", y_ref[0, : min(8, y_ref.shape[1])])
    print("  y_he  first row first 8:", y_he[0, : min(8, y_he.shape[1])])

    # 保存 JSON 报告
    save_result = dict(screen_result)
    save_result["y_ref_first_row"] = y_ref[0].tolist()
    save_result["y_he_first_row"] = y_he[0].tolist()

    return save_result


def main():
    print_section("WP3-A / Case 1: tiny PCMM logical block")
    # 小块：4x4 @ 4x4，按 split_dim=2 分两段
    result_1 = run_case(
        block_rows=4,
        mid_dim=4,
        out_dim=4,
        split_dim=2,
        repeats=1,
        seed=202601,
    )

    print_section("WP3-A / Case 2: small PCMM logical block")
    # 稍大一点：8x8 @ 8x8，按 split_dim=4 分两段
    result_2 = run_case(
        block_rows=8,
        mid_dim=8,
        out_dim=8,
        split_dim=4,
        repeats=1,
        seed=202602,
    )

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3a_pcmm_python_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "case_1": result_1,
                "case_2": result_2,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_section("Report Saved")
    print(f"JSON report saved to: {out_path}")


if __name__ == "__main__":
    main()
