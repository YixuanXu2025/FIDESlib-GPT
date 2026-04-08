from __future__ import annotations

import numpy as np

from .runtime import HERuntime
from .tensor import CipherTensor


def _to_float_list(x):
    return [float(v) for v in x]


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _ensure_same_shape(x: CipherTensor, y: CipherTensor):
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")


def _ensure_slots_match_plain(x: CipherTensor, plain):
    plain = _to_float_list(plain)
    if x.slots_used > 0 and len(plain) != x.slots_used:
        raise ValueError(
            f"plaintext length {len(plain)} != CipherTensor.slots_used {x.slots_used}"
        )
    return plain


# ----------------------------------------------------------------------
# 兼容之前的最小测试
# ----------------------------------------------------------------------

def he_roundtrip_vector(rt: HERuntime, x):
    return rt.roundtrip(_to_float_list(x))


def he_add_plain_input_vectors(rt: HERuntime, x, y):
    return rt.add(_to_float_list(x), _to_float_list(y))


def he_mult_scalar_plain_input(rt: HERuntime, x, scalar: float):
    return rt.mult_scalar(_to_float_list(x), float(scalar))


# ----------------------------------------------------------------------
# 明文参考
# ----------------------------------------------------------------------

def plaintext_linear(x, weight, bias=None):
    """
    x: (..., in_dim)
    weight: (in_dim, out_dim)
    bias: (out_dim,)
    """
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)

    y = x @ weight
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float64)
    return y


# ----------------------------------------------------------------------
# 真正可用的 CipherTensor 低层接口
# ----------------------------------------------------------------------

def he_encrypt_tensor(
    rt: HERuntime,
    x,
    *,
    shape=None,
    layout="vector_contiguous",
    name="",
    device=None,
) -> CipherTensor:
    values = _to_float_list(x)

    if shape is None:
        shape = (len(values),)
    else:
        shape = tuple(shape)

    if _numel(shape) != len(values):
        raise ValueError(
            f"shape {shape} implies numel {_numel(shape)} but input length is {len(values)}"
        )

    ct = rt.encrypt(values)

    if device is None and hasattr(rt, "cfg") and getattr(rt.cfg, "devices", None):
        if len(rt.cfg.devices) > 0:
            device = rt.cfg.devices[0]

    return CipherTensor(
        ct=ct,
        shape=shape,
        layout=layout,
        slots_used=len(values),
        level=None,
        scale=None,
        device=device,
        name=name,
    )


def he_decrypt_tensor(rt: HERuntime, x: CipherTensor, logical_length=None):
    x.ensure_cipher()
    if logical_length is None:
        logical_length = x.slots_used if x.slots_used > 0 else x.numel
    return rt.decrypt(x.ct, logical_length=logical_length)


def he_add(rt: HERuntime, x: CipherTensor, y: CipherTensor, *, name=None) -> CipherTensor:
    x.ensure_cipher()
    y.ensure_cipher()
    _ensure_same_shape(x, y)

    ct = rt.add_ct(x.ct, y.ct)
    return x.clone_with(ct=ct, name=(x.name if name is None else name))


def he_add_plain(rt: HERuntime, x: CipherTensor, plain, *, name=None) -> CipherTensor:
    x.ensure_cipher()
    plain = _ensure_slots_match_plain(x, plain)

    ct = rt.add_plain_ct(x.ct, plain)
    return x.clone_with(ct=ct, name=(x.name if name is None else name))


def he_mult_scalar(rt: HERuntime, x: CipherTensor, scalar: float, *, name=None) -> CipherTensor:
    x.ensure_cipher()

    ct = rt.mult_scalar_ct(x.ct, float(scalar))
    return x.clone_with(ct=ct, name=(x.name if name is None else name))


def he_mult_plain(rt: HERuntime, x: CipherTensor, plain, *, name=None) -> CipherTensor:
    x.ensure_cipher()
    plain = _ensure_slots_match_plain(x, plain)

    ct = rt.mult_plain_ct(x.ct, plain)
    return x.clone_with(ct=ct, name=(x.name if name is None else name))


def he_rotate(rt: HERuntime, x: CipherTensor, steps: int, *, name=None) -> CipherTensor:
    x.ensure_cipher()

    ct = rt.rotate_ct(x.ct, int(steps))
    return x.clone_with(ct=ct, name=(x.name if name is None else name))


# ----------------------------------------------------------------------
# baseline helpers for he_linear_plain
# ----------------------------------------------------------------------

def he_sum_slots_to_slot0(rt: HERuntime, x: CipherTensor, length=None, *, name=None) -> CipherTensor:
    x.ensure_cipher()

    if length is None:
        length = x.slots_used
    if length <= 0:
        raise ValueError("length must be positive")

    acc = x
    for step in range(1, length):
        rotated = he_rotate(rt, x, step, name=f"{x.name}_rot{step}")
        acc = he_add(rt, acc, rotated, name=(x.name if name is None else name))

    mask0 = [1.0] + [0.0] * (x.slots_used - 1)
    return he_mult_plain(rt, acc, mask0, name=(x.name if name is None else name))


def he_dot_plain_to_slot0(rt: HERuntime, x: CipherTensor, plain_vec, bias_scalar=None, *, name=None) -> CipherTensor:
    x.ensure_cipher()
    plain_vec = _ensure_slots_match_plain(x, plain_vec)

    prod = he_mult_plain(rt, x, plain_vec, name=f"{x.name}_mul_plain")
    acc0 = he_sum_slots_to_slot0(rt, prod, length=x.slots_used, name=(x.name if name is None else name))

    if bias_scalar is not None and float(bias_scalar) != 0.0:
        bias0 = [float(bias_scalar)] + [0.0] * (x.slots_used - 1)
        acc0 = he_add_plain(rt, acc0, bias0, name=(x.name if name is None else name))

    return acc0


def he_place_slot0_to(rt: HERuntime, x: CipherTensor, target_slot: int, *, name=None) -> CipherTensor:
    x.ensure_cipher()

    if target_slot < 0:
        raise ValueError("target_slot must be non-negative")
    if target_slot >= x.slots_used:
        raise ValueError(f"target_slot {target_slot} out of range for slots_used={x.slots_used}")

    if target_slot == 0:
        return x

    # 目前根据你已验证的行为：rotate(+1) 左移，所以这里用负号把 slot0 放到右边目标槽位
    return he_rotate(rt, x, -target_slot, name=(x.name if name is None else name))


def he_linear_plain(rt: HERuntime, x: CipherTensor, weight, bias=None, *, name=None) -> CipherTensor:
    """
    baseline 版本：
      - 仅支持 1D CipherTensor 输入
      - 仅支持 out_dim <= x.slots_used
      - 正确性优先，不追求性能
    """
    x.ensure_cipher()

    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 2:
        raise ValueError("weight must be 2D")

    in_dim, out_dim = weight.shape

    if x.ndim != 1:
        raise ValueError("baseline he_linear_plain currently only supports 1D CipherTensor input")
    if x.shape[0] != in_dim:
        raise ValueError(f"input dim mismatch: x.shape={x.shape}, weight.shape={weight.shape}")
    if out_dim > x.slots_used:
        raise ValueError(
            f"baseline he_linear_plain requires out_dim <= x.slots_used, "
            f"got out_dim={out_dim}, slots_used={x.slots_used}"
        )

    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        if bias.shape != (out_dim,):
            raise ValueError(f"bias must have shape ({out_dim},), got {bias.shape}")

    out = None
    for j in range(out_dim):
        col_j = weight[:, j].tolist()
        bj = None if bias is None else float(bias[j])

        scalar_j = he_dot_plain_to_slot0(rt, x, col_j, bias_scalar=bj, name=f"{x.name}_dot_{j}")
        placed_j = he_place_slot0_to(rt, scalar_j, j, name=f"{x.name}_place_{j}")

        out = placed_j if out is None else he_add(rt, out, placed_j, name=(x.name if name is None else name))

    return x.clone_with(
        ct=out.ct,
        shape=(out_dim,),
        slots_used=out_dim,
        name=(x.name if name is None else name),
    )


# ----------------------------------------------------------------------
# Matrix-block / physical-tile 纯 NumPy 工具函数
# 说明：
#   这一组函数只做“数据布局与还原”的验证，不做真正 HE 计算。
#   它们的作用是把下面这条链先跑通：
#
#       原矩阵
#         -> padding 到 block 整数倍
#         -> 切成 logical blocks
#         -> 每个 logical block 再切成 physical tiles
#         -> 再把 tiles 合并回去
#
#   这样我们就能确认 matrix-block 主方案在“数据组织层”是正确的。
# ----------------------------------------------------------------------

def pad_matrix_to_block_multiple(matrix, block_rows=128, block_cols=128):
    """
    将 2D 矩阵 padding 到 block_rows / block_cols 的整数倍。

    参数：
        matrix: 原始 2D numpy 数组，shape = (rows, cols)
        block_rows, block_cols: logical block 的大小

    返回：
        padded: padding 后矩阵
        meta:   一个字典，记录原始形状和 padding 后形状
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got ndim={matrix.ndim}")

    rows, cols = matrix.shape
    padded_rows = ((rows + block_rows - 1) // block_rows) * block_rows
    padded_cols = ((cols + block_cols - 1) // block_cols) * block_cols

    padded = np.zeros((padded_rows, padded_cols), dtype=np.float64)
    padded[:rows, :cols] = matrix

    meta = {
        "orig_shape": (rows, cols),
        "padded_shape": (padded_rows, padded_cols),
        "block_shape": (block_rows, block_cols),
    }
    return padded, meta


def trim_padded_matrix(padded_matrix, orig_shape):
    """
    将 padding 后的大矩阵裁回原始大小。
    """
    padded_matrix = np.asarray(padded_matrix, dtype=np.float64)
    if padded_matrix.ndim != 2:
        raise ValueError(f"padded_matrix must be 2D, got ndim={padded_matrix.ndim}")

    rows, cols = orig_shape
    return padded_matrix[:rows, :cols]


def split_matrix_into_logical_blocks(matrix, block_rows=128, block_cols=128):
    """
    将一个已经 padding 好的 2D 矩阵切成 logical blocks。

    返回：
        blocks: dict[(block_row, block_col)] = block_numpy_array
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got ndim={matrix.ndim}")

    rows, cols = matrix.shape
    if rows % block_rows != 0 or cols % block_cols != 0:
        raise ValueError(
            f"matrix shape {matrix.shape} is not divisible by block shape ({block_rows}, {block_cols})"
        )

    blocks = {}
    num_row_blocks = rows // block_rows
    num_col_blocks = cols // block_cols

    for br in range(num_row_blocks):
        r0 = br * block_rows
        r1 = (br + 1) * block_rows
        for bc in range(num_col_blocks):
            c0 = bc * block_cols
            c1 = (bc + 1) * block_cols
            blocks[(br, bc)] = matrix[r0:r1, c0:c1].copy()

    return blocks


def merge_logical_blocks_back_to_matrix(blocks, padded_shape, block_rows=128, block_cols=128):
    """
    将 logical blocks 按原网格顺序拼回 padding 后的大矩阵。

    参数：
        blocks: dict[(br, bc)] = block
        padded_shape: (padded_rows, padded_cols)

    返回：
        merged: 合并后的大矩阵
    """
    padded_rows, padded_cols = padded_shape
    if padded_rows % block_rows != 0 or padded_cols % block_cols != 0:
        raise ValueError(
            f"padded_shape {padded_shape} is not divisible by block shape ({block_rows}, {block_cols})"
        )

    merged = np.zeros((padded_rows, padded_cols), dtype=np.float64)
    num_row_blocks = padded_rows // block_rows
    num_col_blocks = padded_cols // block_cols

    for br in range(num_row_blocks):
        r0 = br * block_rows
        r1 = (br + 1) * block_rows
        for bc in range(num_col_blocks):
            c0 = bc * block_cols
            c1 = (bc + 1) * block_cols
            block = np.asarray(blocks[(br, bc)], dtype=np.float64)
            if block.shape != (block_rows, block_cols):
                raise ValueError(
                    f"block {(br, bc)} has shape {block.shape}, expected {(block_rows, block_cols)}"
                )
            merged[r0:r1, c0:c1] = block

    return merged


def split_logical_block_into_tiles(block, tile_rows=64, tile_cols=128):
    """
    将一个 logical block 再切成若干 physical tiles。

    注意：
        当前第一版只考虑“按行方向切 tile”。
        所以典型情况是：
            logical block = 128 x 128
            physical tile = 64 x 128
        那么一个 logical block 会变成两个 tiles。

    返回：
        tiles: list[np.ndarray]
    """
    block = np.asarray(block, dtype=np.float64)
    if block.ndim != 2:
        raise ValueError(f"block must be 2D, got ndim={block.ndim}")

    rows, cols = block.shape
    if cols != tile_cols:
        raise ValueError(
            f"block cols {cols} must equal tile_cols {tile_cols} in current row-split design"
        )
    if rows % tile_rows != 0:
        raise ValueError(
            f"block rows {rows} must be divisible by tile_rows {tile_rows}"
        )

    tiles = []
    num_tiles = rows // tile_rows
    for t in range(num_tiles):
        r0 = t * tile_rows
        r1 = (t + 1) * tile_rows
        tiles.append(block[r0:r1, :].copy())
    return tiles


def merge_tiles_back_to_logical_block(tiles):
    """
    将若干按行切开的 physical tiles 重新拼成一个 logical block。

    当前假设：
        - 所有 tile 的列数一致
        - 所有 tile 按从上到下的顺序传入
    """
    if len(tiles) == 0:
        raise ValueError("tiles must be non-empty")

    tiles = [np.asarray(t, dtype=np.float64) for t in tiles]
    cols = tiles[0].shape[1]

    for idx, t in enumerate(tiles):
        if t.ndim != 2:
            raise ValueError(f"tile {idx} must be 2D, got ndim={t.ndim}")
        if t.shape[1] != cols:
            raise ValueError(
                f"tile {idx} has cols {t.shape[1]}, expected {cols}"
            )

    return np.vstack(tiles)


def split_matrix_blocks_into_tiles(blocks, tile_rows=64, tile_cols=128):
    """
    将一整张矩阵的 logical blocks 全部拆成 physical tiles。

    参数：
        blocks: dict[(br, bc)] = logical_block

    返回：
        tiled_blocks: dict[(br, bc)] = [tile0, tile1, ...]
    """
    tiled_blocks = {}
    for coord, block in blocks.items():
        tiled_blocks[coord] = split_logical_block_into_tiles(
            block,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
        )
    return tiled_blocks


def merge_tiled_blocks_back_to_logical_blocks(tiled_blocks):
    """
    将一整张矩阵的 tiled blocks 恢复为 logical blocks。
    """
    blocks = {}
    for coord, tiles in tiled_blocks.items():
        blocks[coord] = merge_tiles_back_to_logical_block(tiles)
    return blocks


def is_zero_tile(tile, atol=0.0):
    """
    判断一个 tile 是否全为 0。
    用于后续 zero-tile skip 优化统计。
    """
    tile = np.asarray(tile, dtype=np.float64)
    return np.all(np.abs(tile) <= atol)


def count_zero_and_nonzero_tiles(tiled_blocks, atol=0.0):
    """
    统计 tiled blocks 中：
        - 总 tile 数
        - 全零 tile 数
        - 非零 tile 数
    """
    total = 0
    zero = 0

    for _, tiles in tiled_blocks.items():
        for tile in tiles:
            total += 1
            if is_zero_tile(tile, atol=atol):
                zero += 1

    return {
        "total_tiles": total,
        "zero_tiles": zero,
        "nonzero_tiles": total - zero,
    }


# ----------------------------------------------------------------------
# Matrix-block -> CipherTensor / plaintext-tile 映射工具
# 说明：
#   这一组函数的目标不是做真正 HE 计算，而是把：
#
#       大矩阵
#         -> logical blocks
#         -> physical tiles
#
#   映射成后续 block-HE 计算真正会用到的两类对象：
#
#     1) 输入侧：CipherTensor 占位对象（未来可替换成真实加密后的 tile）
#     2) 权重侧：明文 tile 字典（因为当前主路线是明文权重）
# ----------------------------------------------------------------------

from .tensor import make_block_tile_cipher_placeholder
from .layouts import MatrixBlockLayout


def make_matrix_block_layout(rows, cols, ring_dim=16384, block_rows=128, block_cols=128):
    """
    生成一个 matrix-block 布局对象。
    这里主要用于统一拿到：
      - padded_shape
      - logical block 大小
      - physical tile 大小
      - block / tile 数量统计
    """
    return MatrixBlockLayout(
        rows=rows,
        cols=cols,
        ring_dim=ring_dim,
        block_rows=block_rows,
        block_cols=block_cols,
    )


def build_input_cipher_placeholders_from_matrix(
    matrix,
    *,
    ring_dim=16384,
    block_rows=128,
    block_cols=128,
    device=0,
    name_prefix="x",
    skip_zero_tiles=True,
    zero_atol=0.0,
):
    """
    将输入大矩阵（例如 X_flat）映射成“待加密的 block-tile 占位对象”。

    返回：
        result = {
            "layout": layout.summary(),
            "meta": padding/meta 信息,
            "logical_blocks": {(br,bc): block_numpy},
            "tiled_blocks": {(br,bc): [tile0, tile1, ...]},
            "cipher_placeholders": {(br,bc,t): CipherTensor},
            "tile_stats": {...},
        }

    设计意图：
        - 输入侧未来在 HE 中会对应 ciphertext tiles
        - 所以这里先把“哪些 tile 需要存在”整理好
        - 如果 skip_zero_tiles=True，则全零 tile 不生成占位对象
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got ndim={matrix.ndim}")

    layout = make_matrix_block_layout(
        rows=matrix.shape[0],
        cols=matrix.shape[1],
        ring_dim=ring_dim,
        block_rows=block_rows,
        block_cols=block_cols,
    )

    padded, meta = pad_matrix_to_block_multiple(
        matrix,
        block_rows=block_rows,
        block_cols=block_cols,
    )

    logical_blocks = split_matrix_into_logical_blocks(
        padded,
        block_rows=block_rows,
        block_cols=block_cols,
    )

    tiled_blocks = split_matrix_blocks_into_tiles(
        logical_blocks,
        tile_rows=layout.tile_rows,
        tile_cols=layout.tile_cols,
    )

    cipher_placeholders = {}
    total_tiles = 0
    zero_tiles = 0

    for (br, bc), tiles in tiled_blocks.items():
        for tile_index, tile in enumerate(tiles):
            total_tiles += 1
            if is_zero_tile(tile, atol=zero_atol):
                zero_tiles += 1
                if skip_zero_tiles:
                    continue

            ct = make_block_tile_cipher_placeholder(
                tile_rows=tile.shape[0],
                tile_cols=tile.shape[1],
                block_coord=(br, bc),
                tile_index=tile_index,
                padded_shape=meta["padded_shape"],
                device=device,
                name=f"{name_prefix}_b{br}_{bc}_t{tile_index}",
            )
            cipher_placeholders[(br, bc, tile_index)] = ct

    tile_stats = {
        "total_tiles": total_tiles,
        "zero_tiles": zero_tiles,
        "nonzero_tiles": total_tiles - zero_tiles,
        "materialized_cipher_tiles": len(cipher_placeholders),
        "skip_zero_tiles": skip_zero_tiles,
    }

    return {
        "layout": layout.summary(),
        "meta": meta,
        "logical_blocks": logical_blocks,
        "tiled_blocks": tiled_blocks,
        "cipher_placeholders": cipher_placeholders,
        "tile_stats": tile_stats,
    }


def build_plain_tiled_blocks_from_matrix(
    matrix,
    *,
    ring_dim=16384,
    block_rows=128,
    block_cols=128,
    name_prefix="w",
):
    """
    将权重矩阵映射成“明文 block/tile 字典”。

    当前主路线里，预训练权重保持明文，所以权重侧不需要 CipherTensor，
    而是直接保留为 plaintext numpy tiles，供未来 he_block_linear_plain 使用。

    返回：
        result = {
            "layout": ...,
            "meta": ...,
            "logical_blocks": ...,
            "tiled_blocks": ...,
            "tile_stats": ...,
        }
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got ndim={matrix.ndim}")

    layout = make_matrix_block_layout(
        rows=matrix.shape[0],
        cols=matrix.shape[1],
        ring_dim=ring_dim,
        block_rows=block_rows,
        block_cols=block_cols,
    )

    padded, meta = pad_matrix_to_block_multiple(
        matrix,
        block_rows=block_rows,
        block_cols=block_cols,
    )

    logical_blocks = split_matrix_into_logical_blocks(
        padded,
        block_rows=block_rows,
        block_cols=block_cols,
    )

    tiled_blocks = split_matrix_blocks_into_tiles(
        logical_blocks,
        tile_rows=layout.tile_rows,
        tile_cols=layout.tile_cols,
    )

    tile_stats = count_zero_and_nonzero_tiles(tiled_blocks)

    return {
        "layout": layout.summary(),
        "meta": meta,
        "logical_blocks": logical_blocks,
        "tiled_blocks": tiled_blocks,
        "tile_stats": tile_stats,
    }


def summarize_first_items(d, k=3):
    """
    打印/查看前几个条目时用的小工具。
    """
    items = list(d.items())
    return items[:k]


# ----------------------------------------------------------------------
# Matrix-block physical tile 的 HE roundtrip 工具
# 说明：
#   这一组函数真正调用 HERuntime 的 encrypt/decrypt，
#   用来验证：
#
#       输入矩阵
#         -> 切成 physical tiles
#         -> 只加密非零 tile
#         -> 再解密回来
#         -> 补回零 tile
#         -> 合并恢复原矩阵
#
#   当前仍然只做“输入侧 roundtrip”，不做 block 乘法。
# ----------------------------------------------------------------------

def encrypt_input_cipher_tiles(rt: HERuntime, input_mapping):
    """
    将 build_input_cipher_placeholders_from_matrix(...) 生成的输入侧占位对象，
    替换成真正加密后的 CipherTensor。

    参数：
        rt: HERuntime，必须已 initialize
        input_mapping: build_input_cipher_placeholders_from_matrix(...) 的返回结果

    返回：
        encrypted_result = {
            ...原字段...
            "encrypted_cipher_tiles": {(br,bc,t): CipherTensor(真正带 ct)},
        }
    """
    tiled_blocks = input_mapping["tiled_blocks"]
    cipher_placeholders = input_mapping["cipher_placeholders"]

    encrypted_cipher_tiles = {}

    for key, placeholder in cipher_placeholders.items():
        br, bc, tile_index = key
        tile = tiled_blocks[(br, bc)][tile_index]

        # 注意：这里按 row-major flatten，未来 block 乘法路径也会沿用这个顺序
        flat = tile.reshape(-1).astype(np.float64).tolist()

        ct_handle = rt.encrypt(flat)

        encrypted_cipher_tiles[key] = placeholder.clone_with(
            ct=ct_handle,
            name=placeholder.name,
        )

    result = dict(input_mapping)
    result["encrypted_cipher_tiles"] = encrypted_cipher_tiles
    return result


def decrypt_encrypted_input_tiles_back_to_matrix(rt: HERuntime, encrypted_result):
    """
    将加密后的输入 tiles 解密回来，并重建原始矩阵。

    注意：
        - 对于 skip_zero_tiles=True 时未物化的零 tile，这里自动补全为 0
        - 所以最终能够完整还原 padding 后的大矩阵，再裁剪回原矩阵
    """
    meta = encrypted_result["meta"]
    tiled_blocks_ref = encrypted_result["tiled_blocks"]
    encrypted_cipher_tiles = encrypted_result["encrypted_cipher_tiles"]

    recovered_tiled_blocks = {}

    for (br, bc), tiles in tiled_blocks_ref.items():
        recovered_tiles = []

        for tile_index, tile_ref in enumerate(tiles):
            key = (br, bc, tile_index)

            if key in encrypted_cipher_tiles:
                ct_tensor = encrypted_cipher_tiles[key]
                ct_tensor.ensure_cipher()

                flat = np.array(
                    rt.decrypt(ct_tensor.ct, logical_length=ct_tensor.slots_used),
                    dtype=np.float64,
                )
                tile = flat.reshape(ct_tensor.shape)
            else:
                # 说明这是 skip_zero_tiles 时被跳过的全零 tile
                tile = np.zeros_like(tile_ref)

            recovered_tiles.append(tile)

        recovered_tiled_blocks[(br, bc)] = recovered_tiles

    recovered_blocks = merge_tiled_blocks_back_to_logical_blocks(recovered_tiled_blocks)

    recovered_padded = merge_logical_blocks_back_to_matrix(
        recovered_blocks,
        padded_shape=meta["padded_shape"],
        block_rows=meta["block_shape"][0],
        block_cols=meta["block_shape"][1],
    )

    recovered = trim_padded_matrix(recovered_padded, meta["orig_shape"])
    return recovered


def summarize_cipher_tile_mapping(encrypted_result, k=3):
    """
    用于调试打印前几个 ciphertext tiles 的 summary。
    """
    items = list(encrypted_result["encrypted_cipher_tiles"].items())
    out = []
    for key, ct in items[:k]:
        out.append((key, ct.summary()))
    return out
