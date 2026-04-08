from hegpt.spec import ProjectShapeSpec


def print_section(title: str):
    print("=" * 100)
    print(title)


def print_dict(d: dict, prefix: str = ""):
    for k, v in d.items():
        print(f"{prefix}{k}: {v}")


def main():
    # ------------------------------------------------------------
    # 你当前研究中最典型的一组规模：
    #   (B, T, N) = (4, 16, 768)
    #   先看普通投影 W: (768, 768)
    #   再看 QKV 合并投影 W_QKV: (768, 2304)
    # ------------------------------------------------------------

    print_section("Case A: 普通线性层 / 输出投影 / MLP 单路投影")
    spec_a = ProjectShapeSpec(
        batch_size=4,
        seq_len=16,
        hidden_dim=768,
        out_dim=768,
    )

    print("project shape spec:")
    print_dict(spec_a.summary(), prefix="  ")

    st_a = spec_a.make_single_token_layout(has_bias=True)
    mb_a = spec_a.make_matrix_block_layout(
        ring_dim=16384,
        block_rows=128,
        block_cols=128,
        act_tile_rows=128,
        act_tile_cols=64,
        weight_tile_rows=64,
        weight_tile_cols=128,
    )

    print_section("A1. single-token baseline summary")
    print_dict(st_a.summary(), prefix="  ")

    print_section("A2. matrix-block layout summary")
    print_dict(mb_a.summary(), prefix="  ")

    print_section("Case B: QKV 合并投影")
    spec_b = ProjectShapeSpec(
        batch_size=4,
        seq_len=16,
        hidden_dim=768,
        out_dim=2304,
    )

    print("project shape spec:")
    print_dict(spec_b.summary(), prefix="  ")

    st_b = spec_b.make_single_token_layout(has_bias=True)
    mb_b = spec_b.make_matrix_block_layout(
        ring_dim=16384,
        block_rows=128,
        block_cols=128,
        act_tile_rows=128,
        act_tile_cols=64,
        weight_tile_rows=64,
        weight_tile_cols=128,
    )

    print_section("B1. single-token baseline summary")
    print_dict(st_b.summary(), prefix="  ")

    print_section("B2. matrix-block layout summary")
    print_dict(mb_b.summary(), prefix="  ")


if __name__ == "__main__":
    main()
