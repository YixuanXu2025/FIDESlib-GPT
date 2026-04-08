from __future__ import annotations

import json
from pathlib import Path


def print_section(title: str):
    print("=" * 100)
    print(title)


def print_dict(d: dict, prefix: str = ""):
    for k, v in d.items():
        print(f"{prefix}{k}: {v}")


def he_linear_plain_padded_op_counts(split_dim: int, out_dim: int, has_bias: bool = False):
    """
    统计当前 Python baseline 中：
      he_linear_plain_padded(...)
    一次调用所对应的大致原语数量。

    当前实现逻辑（按每个输出列 j）：
      1) mult_plain
      2) sum_slots_to_slot0(length=split_dim)
         -> (split_dim - 1) 次 rotate
         -> (split_dim - 1) 次 add
      3) place_slot0_to(j)
         -> j=0 不需要 rotate
         -> 其余 out_dim-1 个输出各需要 1 次 rotate
      4) 将 placed_j 累加到输出
         -> out_dim-1 次 add
      5) 若有 bias，则每个输出 1 次 add_plain

    这里的统计目标不是 cycle-accurate，而是作为：
      - 当前 Python PCMM baseline 的核心操作基线
      - 后续比较 BSGS / JKLS / 高性能协议时的参照
    """
    if split_dim <= 0 or out_dim <= 0:
        raise ValueError("split_dim and out_dim must be positive")

    mult_plain = out_dim
    reduce_rotates = out_dim * max(split_dim - 1, 0)
    reduce_adds = out_dim * max(split_dim - 1, 0)

    # 把 slot0 放到输出槽位 j
    place_rotates = max(out_dim - 1, 0)

    # 将每个 placed_j 与已有输出累加
    output_adds = max(out_dim - 1, 0)

    add_plain = out_dim if has_bias else 0

    return {
        "mult_plain": mult_plain,
        "rotate": reduce_rotates + place_rotates,
        "add": reduce_adds + output_adds,
        "add_plain": add_plain,
    }


def pcmm_python_baseline_core_op_counts(
    rows: int,
    mid_dim: int,
    out_dim: int,
    split_dim: int,
    has_bias: bool = False,
):
    """
    统计当前 Python 版 PCMM baseline（单个 logical block）的核心操作数。

    当前 baseline 逻辑：
      对每一行 r：
        对每个 split s：
          ct_part = he_linear_plain_padded(x[r,s], W[s], ...)
        再把所有 split 的 ct_part 累加成该行输出

    因此需要统计两层：
      A) 每个 row-split 内部的 he_linear_plain_padded 操作数
      B) 每行把多个 split 结果相加的 ct_add 次数
    """
    if mid_dim % split_dim != 0:
        raise ValueError("mid_dim must be divisible by split_dim")

    num_splits = mid_dim // split_dim
    per_row_split = he_linear_plain_padded_op_counts(
        split_dim=split_dim,
        out_dim=out_dim,
        has_bias=has_bias,
    )

    # 每一行内部，要把 num_splits 个 ct_part 累加起来
    # 需要 (num_splits - 1) 次密文加法
    per_row_split_accum_adds = max(num_splits - 1, 0)

    per_row_core = {
        "mult_plain": num_splits * per_row_split["mult_plain"],
        "rotate": num_splits * per_row_split["rotate"],
        "add": num_splits * per_row_split["add"] + per_row_split_accum_adds,
        "add_plain": num_splits * per_row_split["add_plain"],
    }

    total_core = {
        "mult_plain": rows * per_row_core["mult_plain"],
        "rotate": rows * per_row_core["rotate"],
        "add": rows * per_row_core["add"],
        "add_plain": rows * per_row_core["add_plain"],
    }

    # 当前 Python baseline 的加/解密次数（按一次完整运行）
    encrypt_ct_count = rows * num_splits     # 每个 row、每个 split 一个输入段 ciphertext
    decrypt_ct_count = rows                  # 每行一个最终输出 ciphertext

    return {
        "rows": rows,
        "mid_dim": mid_dim,
        "out_dim": out_dim,
        "split_dim": split_dim,
        "num_splits": num_splits,
        "per_row_split_op_counts": per_row_split,
        "per_row_core_op_counts": per_row_core,
        "total_core_op_counts": total_core,
        "encrypt_ct_count_per_run": encrypt_ct_count,
        "decrypt_ct_count_per_run": decrypt_ct_count,
        "split_accum_add_per_row": per_row_split_accum_adds,
        "total_split_accum_add": rows * per_row_split_accum_adds,
    }


def build_case_report(case_name: str, rows: int, mid_dim: int, out_dim: int, split_dim: int):
    result = pcmm_python_baseline_core_op_counts(
        rows=rows,
        mid_dim=mid_dim,
        out_dim=out_dim,
        split_dim=split_dim,
        has_bias=False,
    )
    result["case_name"] = case_name
    return result


def main():
    reports = {}

    print_section("WP3-B / Case 1: square input")
    case1 = build_case_report(
        case_name="square_input_8x8__8x8",
        rows=8,
        mid_dim=8,
        out_dim=8,
        split_dim=4,
    )
    print_dict(case1, prefix="  ")
    reports["case_1_square_input"] = case1

    print_section("WP3-B / Case 2: non-square input")
    case2 = build_case_report(
        case_name="rect_input_4x8__8x8",
        rows=4,
        mid_dim=8,
        out_dim=8,
        split_dim=4,
    )
    print_dict(case2, prefix="  ")
    reports["case_2_non_square_input"] = case2

    print_section("WP3-B / Case 3: non-square output")
    case3 = build_case_report(
        case_name="square_input_8x8__8x16",
        rows=8,
        mid_dim=8,
        out_dim=16,
        split_dim=4,
    )
    print_dict(case3, prefix="  ")
    reports["case_3_non_square_output"] = case3

    # 额外给一个“更贴近未来 block 的统计例子”
    print_section("WP3-B / Case 4: projected logical-block-like example")
    case4 = build_case_report(
        case_name="projected_block_like_8x64__64x64",
        rows=8,
        mid_dim=64,
        out_dim=64,
        split_dim=16,
    )
    print_dict(case4, prefix="  ")
    reports["case_4_projected_block_like"] = case4

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3b_pcmm_core_ops_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print_section("Report Saved")
    print(f"JSON report saved to: {out_path}")


if __name__ == "__main__":
    main()
