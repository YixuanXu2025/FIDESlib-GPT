import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.ops import (
    build_input_cipher_placeholders_from_matrix,
    encrypt_input_cipher_tiles,
    decrypt_encrypted_input_tiles_back_to_matrix,
    summarize_cipher_tile_mapping,
)


def main():
    # ------------------------------------------------------------
    # 例子：
    #   X_flat = (64, 768)
    #   在 ring_dim=16384 下：
    #     logical block = 128x128
    #     physical tile = 64x128
    # ------------------------------------------------------------
    x = np.arange(64 * 768, dtype=np.float64).reshape(64, 768)

    x_mapping = build_input_cipher_placeholders_from_matrix(
        x,
        ring_dim=16384,
        block_rows=128,
        block_cols=128,
        device=0,
        name_prefix="x",
        skip_zero_tiles=True,   # 输入侧启用 zero-tile skip
    )

    print("=" * 100)
    print("Input X_flat = (64, 768)")
    print("layout summary:", x_mapping["layout"])
    print("meta:", x_mapping["meta"])
    print("tile stats:", x_mapping["tile_stats"])
    print("num cipher placeholders:", len(x_mapping["cipher_placeholders"]))

    # ------------------------------------------------------------
    # 关键点：
    #   这里显式把 batch_size 设成 8192
    #   因为 physical tile = 64x128 = 8192 个数
    # ------------------------------------------------------------
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

    with HERuntime(cfg) as rt:
        print("runtime info:", rt.info())

        encrypted = encrypt_input_cipher_tiles(rt, x_mapping)

        print("num encrypted ciphertext tiles:", len(encrypted["encrypted_cipher_tiles"]))
        print("first encrypted tile summaries:")
        for key, summary in summarize_cipher_tile_mapping(encrypted, k=3):
            print("  key =", key)
            print("  summary =", summary)

        recovered = decrypt_encrypted_input_tiles_back_to_matrix(rt, encrypted)

    max_abs_err = float(np.max(np.abs(recovered - x)))

    print("=" * 100)
    print("recovered shape:", recovered.shape)
    print("max_abs_err:", max_abs_err)

    if recovered.shape == x.shape:
        print("shape check: OK")
    else:
        print("shape check: FAILED")

    print("first row first 8 values (orig):", x[0, :8])
    print("first row first 8 values (recv):", recovered[0, :8])


if __name__ == "__main__":
    main()
