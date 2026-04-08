from hegpt.layouts import MatrixBlockLayout
from hegpt.spec import make_bt_hidden_logical_spec, flatten_bt_hidden_to_2d, MatrixBlockExecSpec


def print_case(title, rows, cols, ring_dim=16384, block_rows=128, block_cols=128):
    layout = MatrixBlockLayout(
        rows=rows,
        cols=cols,
        ring_dim=ring_dim,
        block_rows=block_rows,
        block_cols=block_cols,
    )
    print("=" * 80)
    print(title)
    print(layout.summary())
    print("logical block coords:")
    for bc in layout.iter_logical_block_coords():
        print("  block", bc, "tiles:", list(layout.iter_physical_tile_coords(bc)))


def main():
    logical = make_bt_hidden_logical_spec(batch_size=4, seq_len=16, hidden_dim=768)
    flat = flatten_bt_hidden_to_2d(batch_size=4, seq_len=16, hidden_dim=768)
    exec_spec = MatrixBlockExecSpec(rows=64, cols=768, ring_dim=16384, block_rows=128, block_cols=128)

    print("logical BT hidden spec:", logical.summary())
    print("flattened 2D spec:", flat.summary())
    print("exec spec:", exec_spec.summary())

    print_case("Input X_flat = (64, 768)", rows=64, cols=768)
    print_case("Weight W = (768, 768)", rows=768, cols=768)
    print_case("Weight W_QKV = (768, 2304)", rows=768, cols=2304)


if __name__ == "__main__":
    main()
