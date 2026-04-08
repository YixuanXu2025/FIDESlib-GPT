from _fideslib import FidesCKKSContext


def main():
    ctx = FidesCKKSContext()
    ctx.init(
        multiplicative_depth=2,
        scaling_mod_size=50,
        batch_size=8192,
        ring_dim=1 << 14,
        devices=[0],
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
        rotation_steps=[],
    )

    print("runtime info:", dict(ctx.info()))
    print("block summary:", dict(ctx.block_layout_summary(64, 768, 1 << 14, 128, 128)))

    # 构造一个 128x128 的 activation block 和 weight block（row-major flatten）
    x_block = [float(i) for i in range(128 * 128)]
    w_block = [float(i) * 0.001 for i in range(128 * 128)]

    x_tiles = ctx.split_activation_block_tiles(x_block, 128, 128, 64)
    w_tiles = ctx.split_weight_block_tiles(w_block, 128, 128, 64)

    print("num x_tiles:", len(x_tiles), "tile lens:", [len(t) for t in x_tiles])
    print("num w_tiles:", len(w_tiles), "tile lens:", [len(t) for t in w_tiles])

    y_pcmm = ctx.pcmm_square_plain_reference(x_block, w_block, 128, 128, 64)
    y_ccmm = ctx.ccmm_square_plain_reference(x_block, w_block, 128, 128)

    print("pcmm ref len:", len(y_pcmm), "first 8:", y_pcmm[:8])
    print("ccmm ref len:", len(y_ccmm), "first 8:", y_ccmm[:8])

    # 预留接口当前应该抛异常
    try:
        ct = ctx.encrypt([1.0] * 8192)
        ctx.eval_pcmm_square_ct_plain_stub(ct, w_block, 128, 128)
    except Exception as e:
        print("pcmm stub expected error:", e)

    ctx.close()


if __name__ == "__main__":
    main()
