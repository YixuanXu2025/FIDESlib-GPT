import numpy as np

from hegpt.ops import (
    build_input_cipher_placeholders_from_matrix,
    build_plain_tiled_blocks_from_matrix,
    summarize_first_items,
)


def main():
    # ------------------------------------------------------------
    # Case 1: 输入矩阵 X_flat = (64, 768)
    # 目标：
    #   - 构造成输入侧 CipherTensor placeholders
    #   - 验证 zero-tile skip 后，只保留 6 个非零 tile
    # ------------------------------------------------------------
    x = np.arange(64 * 768, dtype=np.float64).reshape(64, 768)

    x_result = build_input_cipher_placeholders_from_matrix(
        x,
        ring_dim=16384,
        block_rows=128,
        block_cols=128,
        device=0,
        name_prefix="x",
        skip_zero_tiles=True,
    )

    print("=" * 100)
    print("Input X_flat = (64, 768)")
    print("layout summary:", x_result["layout"])
    print("meta:", x_result["meta"])
    print("tile stats:", x_result["tile_stats"])
    print("num cipher placeholders:", len(x_result["cipher_placeholders"]))

    print("first placeholder summaries:")
    for key, ct in summarize_first_items(x_result["cipher_placeholders"], k=3):
        print("  key =", key)
        print("  summary =", ct.summary())

    # ------------------------------------------------------------
    # Case 2: 权重矩阵 W = (768, 768)
    # 目标：
    #   - 构造成 plaintext tiled blocks
    #   - 验证 tile 数符合预期
    # ------------------------------------------------------------
    w = np.arange(768 * 768, dtype=np.float64).reshape(768, 768)

    w_result = build_plain_tiled_blocks_from_matrix(
        w,
        ring_dim=16384,
        block_rows=128,
        block_cols=128,
        name_prefix="w",
    )

    print("=" * 100)
    print("Weight W = (768, 768)")
    print("layout summary:", w_result["layout"])
    print("meta:", w_result["meta"])
    print("tile stats:", w_result["tile_stats"])

    print("first tiled block entry:")
    first_block_key, first_tiles = summarize_first_items(w_result["tiled_blocks"], k=1)[0]
    print("  block key:", first_block_key)
    print("  num tiles:", len(first_tiles))
    print("  tile shapes:", [t.shape for t in first_tiles])

    # ------------------------------------------------------------
    # Case 3: QKV 权重矩阵 W_QKV = (768, 2304)
    # ------------------------------------------------------------
    w_qkv = np.arange(768 * 2304, dtype=np.float64).reshape(768, 2304)

    qkv_result = build_plain_tiled_blocks_from_matrix(
        w_qkv,
        ring_dim=16384,
        block_rows=128,
        block_cols=128,
        name_prefix="wqkv",
    )

    print("=" * 100)
    print("Weight W_QKV = (768, 2304)")
    print("layout summary:", qkv_result["layout"])
    print("meta:", qkv_result["meta"])
    print("tile stats:", qkv_result["tile_stats"])

    first_block_key, first_tiles = summarize_first_items(qkv_result["tiled_blocks"], k=1)[0]
    print("  first block key:", first_block_key)
    print("  num tiles:", len(first_tiles))
    print("  tile shapes:", [t.shape for t in first_tiles])


if __name__ == "__main__":
    main()
