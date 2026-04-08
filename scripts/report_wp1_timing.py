import json
import time
from pathlib import Path

import numpy as np

from hegpt.config import HEConfig
from hegpt.runtime import HERuntime
from hegpt.ops import (
    plaintext_linear,
    he_encrypt_tensor,
    he_decrypt_tensor,
    he_linear_plain,
)
from hegpt.spec import ProjectShapeSpec


# ============================================================
# 工具函数
# ============================================================

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


# ============================================================
# Part A: single-token 小规模真实 HE timing
# ============================================================

def bench_single_token_he_case(hidden_dim: int, out_dim: int, repeats: int = 3, seed: int = 0):
    """
    对 single-token baseline 做“小规模真实 HE 时间”测试。

    说明：
      - 这里的目的是得到趋势，不是跑 full-size 大规模。
      - 当前 he_linear_plain(...) 的复杂度很高，所以只建议测小规模。
    """
    rng = np.random.default_rng(seed)

    x = rng.normal(0.0, 1.0, size=(hidden_dim,)).astype(np.float64)
    w = rng.normal(0.0, 0.2, size=(hidden_dim, out_dim)).astype(np.float64)
    b = rng.normal(0.0, 0.1, size=(out_dim,)).astype(np.float64)

    # 为了保证槽位和旋转 key 足够，直接把 batch_size 开大到 8192
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

    # he_linear_plain 当前会用到：
    #   1..hidden_dim-1 的 reduce rotations
    #   -(out_dim-1)..0 的 place rotations
    rotation_steps = list(range(1, max(hidden_dim, 1))) + list(range(-(max(out_dim, 1) - 1), 0))

    plaintext_linear_us_list = []
    encrypt_us_list = []
    he_linear_plain_us_list = []
    decrypt_us_list = []
    he_total_us_list = []
    max_abs_err_list = []

    with HERuntime(cfg, rotation_steps=rotation_steps) as rt:
        for _ in range(repeats):
            # plaintext
            t0 = now_us()
            y_ref = plaintext_linear(x, w, b)
            t1 = now_us()

            # encrypt
            t2 = now_us()
            ct_x = he_encrypt_tensor(
                rt,
                x,
                shape=(hidden_dim,),
                layout="token_hidden_contiguous",
                name="x",
                device=0,
            )
            t3 = now_us()

            # HE linear
            t4 = now_us()
            ct_y = he_linear_plain(rt, ct_x, w, b, name="y")
            t5 = now_us()

            # decrypt
            t6 = now_us()
            y_he = np.array(he_decrypt_tensor(rt, ct_y), dtype=np.float64)
            t7 = now_us()

            plaintext_linear_us_list.append(t1 - t0)
            encrypt_us_list.append(t3 - t2)
            he_linear_plain_us_list.append(t5 - t4)
            decrypt_us_list.append(t7 - t6)
            he_total_us_list.append((t3 - t2) + (t5 - t4) + (t7 - t6))
            max_abs_err_list.append(float(np.max(np.abs(y_ref - y_he[:out_dim]))))

    result = {
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "repeats": repeats,
        "plaintext_linear_us_median": median(plaintext_linear_us_list),
        "encrypt_us_median": median(encrypt_us_list),
        "he_linear_plain_us_median": median(he_linear_plain_us_list),
        "decrypt_us_median": median(decrypt_us_list),
        "he_total_us_median": median(he_total_us_list),
        "max_abs_err_max": max(max_abs_err_list) if max_abs_err_list else None,
    }

    if result["plaintext_linear_us_median"] and result["he_linear_plain_us_median"]:
        result["he_core_vs_plain_ratio"] = (
            result["he_linear_plain_us_median"] / result["plaintext_linear_us_median"]
        )
    else:
        result["he_core_vs_plain_ratio"] = None

    return result


# ============================================================
# Part B: matrix-block plain/block timing
# ============================================================

def pad_matrix_to_block_multiple(matrix, block_rows=128, block_cols=128):
    matrix = np.asarray(matrix, dtype=np.float64)
    rows, cols = matrix.shape
    padded_rows = ((rows + block_rows - 1) // block_rows) * block_rows
    padded_cols = ((cols + block_cols - 1) // block_cols) * block_cols

    padded = np.zeros((padded_rows, padded_cols), dtype=np.float64)
    padded[:rows, :cols] = matrix
    return padded, {
        "orig_shape": (rows, cols),
        "padded_shape": (padded_rows, padded_cols),
        "block_shape": (block_rows, block_cols),
    }


def trim_padded_matrix(padded_matrix, orig_shape):
    rows, cols = orig_shape
    return padded_matrix[:rows, :cols]


def split_matrix_into_logical_blocks(matrix, block_rows=128, block_cols=128):
    rows, cols = matrix.shape
    blocks = {}
    for br in range(rows // block_rows):
        for bc in range(cols // block_cols):
            r0 = br * block_rows
            r1 = (br + 1) * block_rows
            c0 = bc * block_cols
            c1 = (bc + 1) * block_cols
            blocks[(br, bc)] = matrix[r0:r1, c0:c1].copy()
    return blocks


def merge_logical_blocks_back_to_matrix(blocks, padded_shape, block_rows=128, block_cols=128):
    rows, cols = padded_shape
    merged = np.zeros((rows, cols), dtype=np.float64)
    for (br, bc), block in blocks.items():
        r0 = br * block_rows
        r1 = (br + 1) * block_rows
        c0 = bc * block_cols
        c1 = (bc + 1) * block_cols
        merged[r0:r1, c0:c1] = block
    return merged


def split_activation_logical_block_into_tiles(block, tile_rows=128, tile_cols=64):
    # 输入/激活块按列切：128x128 -> [128x64, 128x64]
    block = np.asarray(block, dtype=np.float64)
    rows, cols = block.shape
    assert rows == tile_rows
    assert cols % tile_cols == 0

    tiles = []
    for t in range(cols // tile_cols):
        c0 = t * tile_cols
        c1 = (t + 1) * tile_cols
        tiles.append(block[:, c0:c1].copy())
    return tiles


def split_weight_logical_block_into_tiles(block, tile_rows=64, tile_cols=128):
    # 权重块按行切：128x128 -> [64x128, 64x128]
    block = np.asarray(block, dtype=np.float64)
    rows, cols = block.shape
    assert cols == tile_cols
    assert rows % tile_rows == 0

    tiles = []
    for t in range(rows // tile_rows):
        r0 = t * tile_rows
        r1 = (t + 1) * tile_rows
        tiles.append(block[r0:r1, :].copy())
    return tiles


def logical_block_matmul_from_role_tiles(x_tiles, w_tiles):
    out = None
    for xt, wt in zip(x_tiles, w_tiles):
        partial = xt @ wt
        out = partial if out is None else (out + partial)
    return out


def block_matmul_numpy_from_role_tiles(x_matrix, w_matrix, block_rows=128, block_cols=128):
    x_matrix = np.asarray(x_matrix, dtype=np.float64)
    w_matrix = np.asarray(w_matrix, dtype=np.float64)

    x_pad, x_meta = pad_matrix_to_block_multiple(x_matrix, block_rows, block_cols)
    w_pad, w_meta = pad_matrix_to_block_multiple(w_matrix, block_rows, block_cols)

    x_blocks = split_matrix_into_logical_blocks(x_pad, block_rows, block_cols)
    w_blocks = split_matrix_into_logical_blocks(w_pad, block_rows, block_cols)

    y_padded_shape = (x_pad.shape[0], w_pad.shape[1])
    y_blocks = {}

    num_row_blocks = x_pad.shape[0] // block_rows
    num_mid_blocks = x_pad.shape[1] // block_cols
    num_col_blocks = w_pad.shape[1] // block_cols

    for bi in range(num_row_blocks):
        for bj in range(num_col_blocks):
            acc = None
            for bk in range(num_mid_blocks):
                x_block = x_blocks[(bi, bk)]
                w_block = w_blocks[(bk, bj)]

                x_tiles = split_activation_logical_block_into_tiles(
                    x_block, tile_rows=block_rows, tile_cols=block_cols // 2
                )
                w_tiles = split_weight_logical_block_into_tiles(
                    w_block, tile_rows=block_rows // 2, tile_cols=block_cols
                )

                y_part = logical_block_matmul_from_role_tiles(x_tiles, w_tiles)
                acc = y_part if acc is None else (acc + y_part)

            y_blocks[(bi, bj)] = acc

    y_padded = merge_logical_blocks_back_to_matrix(
        y_blocks,
        padded_shape=y_padded_shape,
        block_rows=block_rows,
        block_cols=block_cols,
    )
    y = trim_padded_matrix(y_padded, (x_matrix.shape[0], w_matrix.shape[1]))

    dbg = {
        "x_padded_shape": x_meta["padded_shape"],
        "w_padded_shape": w_meta["padded_shape"],
        "y_padded_shape": y_padded_shape,
        "num_row_blocks": num_row_blocks,
        "num_mid_blocks": num_mid_blocks,
        "num_col_blocks": num_col_blocks,
        "logical_block_products": num_row_blocks * num_mid_blocks * num_col_blocks,
    }
    return y, dbg


def bench_matrix_block_plain_case(rows: int, hidden_dim: int, out_dim: int, repeats: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)

    x = rng.normal(0.0, 1.0, size=(rows, hidden_dim)).astype(np.float64)
    w = rng.normal(0.0, 0.2, size=(hidden_dim, out_dim)).astype(np.float64)

    dense_us_list = []
    block_us_list = []
    max_abs_err_list = []
    dbg_last = None

    for _ in range(repeats):
        t0 = now_us()
        y_ref = x @ w
        t1 = now_us()

        t2 = now_us()
        y_blk, dbg = block_matmul_numpy_from_role_tiles(
            x,
            w,
            block_rows=128,
            block_cols=128,
        )
        t3 = now_us()

        dense_us_list.append(t1 - t0)
        block_us_list.append(t3 - t2)
        max_abs_err_list.append(float(np.max(np.abs(y_blk - y_ref))))
        dbg_last = dbg

    result = {
        "rows": rows,
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "repeats": repeats,
        "dense_numpy_us_median": median(dense_us_list),
        "block_numpy_us_median": median(block_us_list),
        "max_abs_err_max": max(max_abs_err_list) if max_abs_err_list else None,
        "debug": dbg_last,
    }

    if result["dense_numpy_us_median"] and result["block_numpy_us_median"]:
        result["block_vs_dense_ratio"] = (
            result["block_numpy_us_median"] / result["dense_numpy_us_median"]
        )
    else:
        result["block_vs_dense_ratio"] = None

    return result


# ============================================================
# 主报告
# ============================================================

def main():
    report = {}

    # ------------------------------------------------------------
    # Part A: single-token 小规模真实 HE timing
    # ------------------------------------------------------------
    print_section("Part A: single-token 小规模真实 HE timing")
    single_token_cases = [
        (4, 3),     # 已经验证过的极小例子
        (16, 16),   # 小规模方阵
        (32, 32),   # 稍大一点，看趋势
    ]

    part_a = []
    for idx, (h, o) in enumerate(single_token_cases):
        result = bench_single_token_he_case(
            hidden_dim=h,
            out_dim=o,
            repeats=3,
            seed=100 + idx,
        )
        print(f"[single-token HE] hidden_dim={h}, out_dim={o}")
        print_dict(result, prefix="  ")
        part_a.append(result)

    report["single_token_he_small_scale"] = part_a

    # ------------------------------------------------------------
    # Part B: matrix-block plain timing（真实目标尺寸）
    # ------------------------------------------------------------
    print_section("Part B: matrix-block plain/block timing")

    # Case A: 64x768 @ 768x768
    result_a = bench_matrix_block_plain_case(
        rows=64,
        hidden_dim=768,
        out_dim=768,
        repeats=5,
        seed=2026,
    )
    print("[matrix-block plain] X=(64,768), W=(768,768)")
    print_dict(result_a, prefix="  ")

    # Case B: 64x768 @ 768x2304
    result_b = bench_matrix_block_plain_case(
        rows=64,
        hidden_dim=768,
        out_dim=2304,
        repeats=5,
        seed=2027,
    )
    print("[matrix-block plain] X=(64,768), W_QKV=(768,2304)")
    print_dict(result_b, prefix="  ")

    report["matrix_block_plain_case_a"] = result_a
    report["matrix_block_plain_case_b"] = result_b

    # ------------------------------------------------------------
    # Part C: 结构统计（来自工作包1的方法层）
    # ------------------------------------------------------------
    print_section("Part C: 方法层结构统计摘要")

    spec_a = ProjectShapeSpec(batch_size=4, seq_len=16, hidden_dim=768, out_dim=768)
    spec_b = ProjectShapeSpec(batch_size=4, seq_len=16, hidden_dim=768, out_dim=2304)

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

    structure_summary = {
        "case_a_single_token_total_ops": st_a.total_op_counts(),
        "case_a_matrix_block_summary": mb_a.summary(),
        "case_b_single_token_total_ops": st_b.total_op_counts(),
        "case_b_matrix_block_summary": mb_b.summary(),
    }
    print_dict(structure_summary, prefix="  ")
    report["structure_summary"] = structure_summary

    # ------------------------------------------------------------
    # 保存 JSON 报告
    # ------------------------------------------------------------
    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp1_timing_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_section("Report Saved")
    print(f"JSON report saved to: {out_path}")


if __name__ == "__main__":
    main()
