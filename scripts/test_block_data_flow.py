import numpy as np

from hegpt.ops import (
    pad_matrix_to_block_multiple,
    trim_padded_matrix,
    split_matrix_into_logical_blocks,
    merge_logical_blocks_back_to_matrix,
    split_matrix_blocks_into_tiles,
    merge_tiled_blocks_back_to_logical_blocks,
    count_zero_and_nonzero_tiles,
)


def print_case(name, x, block_rows=128, block_cols=128, tile_rows=64, tile_cols=128):
    print("=" * 100)
    print(name)
    print("input shape:", x.shape)

    padded, meta = pad_matrix_to_block_multiple(
        x,
        block_rows=block_rows,
        block_cols=block_cols,
    )
    print("meta:", meta)

    blocks = split_matrix_into_logical_blocks(
        padded,
        block_rows=block_rows,
        block_cols=block_cols,
    )
    print("num logical blocks:", len(blocks))

    tiled_blocks = split_matrix_blocks_into_tiles(
        blocks,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )

    tile_stats = count_zero_and_nonzero_tiles(tiled_blocks)
    print("tile stats:", tile_stats)

    # 选一个 block 看看 shape
    first_coord = sorted(blocks.keys())[0]
    first_block = blocks[first_coord]
    first_tiles = tiled_blocks[first_coord]

    print("first logical block coord:", first_coord)
    print("first logical block shape:", first_block.shape)
    print("first logical block num tiles:", len(first_tiles))
    print("first tile shapes:", [t.shape for t in first_tiles])

    # 做一轮 roundtrip 还原
    merged_blocks = merge_tiled_blocks_back_to_logical_blocks(tiled_blocks)
    merged_padded = merge_logical_blocks_back_to_matrix(
        merged_blocks,
        padded_shape=meta["padded_shape"],
        block_rows=block_rows,
        block_cols=block_cols,
    )
    recovered = trim_padded_matrix(merged_padded, meta["orig_shape"])

    max_abs_err = float(np.max(np.abs(recovered - x)))
    print("recovered shape:", recovered.shape)
    print("max_abs_err:", max_abs_err)

    return {
        "meta": meta,
        "tile_stats": tile_stats,
        "max_abs_err": max_abs_err,
    }


def main():
    # Case 1: 输入矩阵 X_flat = (64, 768)
    # 前 64 行真实，padding 后会多出 64 行 0，因此预期会出现大量 zero tiles
    x = np.arange(64 * 768, dtype=np.float64).reshape(64, 768)
    print_case("Input X_flat = (64, 768)", x)

    # Case 2: 权重矩阵 W = (768, 768)
    # 这里不涉及行方向 padding，通常不应出现 zero tiles
    w = np.arange(768 * 768, dtype=np.float64).reshape(768, 768)
    print_case("Weight W = (768, 768)", w)

    # Case 3: QKV 权重 W_QKV = (768, 2304)
    w_qkv = np.arange(768 * 2304, dtype=np.float64).reshape(768, 2304)
    print_case("Weight W_QKV = (768, 2304)", w_qkv)


if __name__ == "__main__":
    main()
