import numpy as np

from hegpt.spec import ProjectShapeSpec
from hegpt.layouts import MatrixBlockLayout
from hegpt.ops import (
    pad_matrix_to_block_multiple,
    split_matrix_into_logical_blocks,
    split_matrix_blocks_into_tiles,
    count_zero_and_nonzero_tiles,
)


def print_section(title: str):
    print("=" * 100)
    print(title)


def print_dict(d: dict, prefix: str = ""):
    for k, v in d.items():
        print(f"{prefix}{k}: {v}")


def analyze_input_zero_tiles(rows: int, cols: int, block_rows: int = 128, block_cols: int = 128,
                             tile_rows: int = 64, tile_cols: int = 128):
    """
    统计输入矩阵在“当前 row-split physical tile 统计口径”下的：
      - total tiles
      - zero tiles
      - nonzero tiles

    这里沿用你前面已经验证过的那条统计逻辑：
      logical block = 128 x 128
      physical tile = 64 x 128
    这样可以给 matrix-block 方法报告补上“输入侧 zero-skip 可减少多少 tile”。
    """
    x = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)

    padded, meta = pad_matrix_to_block_multiple(
        x,
        block_rows=block_rows,
        block_cols=block_cols,
    )

    blocks = split_matrix_into_logical_blocks(
        padded,
        block_rows=block_rows,
        block_cols=block_cols,
    )

    tiled_blocks = split_matrix_blocks_into_tiles(
        blocks,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )

    stats = count_zero_and_nonzero_tiles(tiled_blocks)
    return {
        "orig_shape": (rows, cols),
        "padded_shape": meta["padded_shape"],
        "tile_stats_row_split_for_zero_skip": stats,
    }


def build_case_report(batch_size: int, seq_len: int, hidden_dim: int, out_dim: int, has_bias: bool = True):
    spec = ProjectShapeSpec(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
    )

    st = spec.make_single_token_layout(has_bias=has_bias)
    mb = spec.make_matrix_block_layout(
        ring_dim=16384,
        block_rows=128,
        block_cols=128,
        act_tile_rows=128,
        act_tile_cols=64,
        weight_tile_rows=64,
        weight_tile_cols=128,
    )

    # 这里补一个“输入侧 zero-skip”统计
    # 注意：
    #   当前 matrix-block 主方案的执行侧我们更偏向 role-specific tiling，
    #   但 zero-skip 的直观收益之前是在 row-split (64x128) 输入 tile 下观测到的。
    #   这里把它单独列出来，作为“输入 padding 带来的潜在节省统计”。
    zero_skip = analyze_input_zero_tiles(
        rows=batch_size * seq_len,
        cols=hidden_dim,
        block_rows=128,
        block_cols=128,
        tile_rows=64,
        tile_cols=128,
    )

    report = {
        "project_shape_spec": spec.summary(),
        "single_token_baseline": st.summary(),
        "matrix_block_layout": mb.summary(),
        "input_zero_skip_stats": zero_skip,
    }
    return report


def main():
    # ------------------------------------------------------------------
    # Case A: 普通线性层 / 输出投影 / MLP 单路投影
    #   X: (B,T,N) = (4,16,768)
    #   W: (768,768)
    # ------------------------------------------------------------------
    case_a = build_case_report(
        batch_size=4,
        seq_len=16,
        hidden_dim=768,
        out_dim=768,
        has_bias=True,
    )

    print_section("Case A: 普通线性层 / 输出投影 / MLP 单路投影")
    print("A0. project shape spec")
    print_dict(case_a["project_shape_spec"], prefix="  ")

    print_section("A1. single-token baseline")
    print_dict(case_a["single_token_baseline"], prefix="  ")

    print_section("A2. matrix-block layout")
    print_dict(case_a["matrix_block_layout"], prefix="  ")

    print_section("A3. input zero-skip stats (row-split reference)")
    print_dict(case_a["input_zero_skip_stats"], prefix="  ")

    # ------------------------------------------------------------------
    # Case B: QKV 合并投影
    #   X: (B,T,N) = (4,16,768)
    #   W_QKV: (768,2304)
    # ------------------------------------------------------------------
    case_b = build_case_report(
        batch_size=4,
        seq_len=16,
        hidden_dim=768,
        out_dim=2304,
        has_bias=True,
    )

    print_section("Case B: QKV 合并投影")
    print("B0. project shape spec")
    print_dict(case_b["project_shape_spec"], prefix="  ")

    print_section("B1. single-token baseline")
    print_dict(case_b["single_token_baseline"], prefix="  ")

    print_section("B2. matrix-block layout")
    print_dict(case_b["matrix_block_layout"], prefix="  ")

    print_section("B3. input zero-skip stats (row-split reference)")
    print_dict(case_b["input_zero_skip_stats"], prefix="  ")


if __name__ == "__main__":
    main()
