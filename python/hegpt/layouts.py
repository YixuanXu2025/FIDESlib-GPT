from __future__ import annotations

from dataclasses import dataclass


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("b must be positive")
    return (a + b - 1) // b


def pad_to_multiple(x: int, base: int) -> int:
    return ceil_div(x, base) * base


@dataclass(frozen=True)
class SingleTokenBaselineLayout:
    """
    single-token baseline 的统计模型。

    这里描述的是我们当前已经跑通的那条 baseline：
      - 一个 token 的 hidden 向量 -> 一个 ciphertext
      - 线性层 y = xW + b 通过 he_linear_plain(...) 做

    这个类的作用不是执行计算，而是：
      1) 给出 single-token 路线的结构化统计
      2) 作为 matrix-block 主方案的对照组
    """
    batch_size: int
    seq_len: int
    hidden_dim: int
    out_dim: int
    has_bias: bool = True

    @property
    def num_tokens(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def input_ciphertexts(self) -> int:
        # baseline 下：一个 token 一个密文
        return self.num_tokens

    @property
    def output_ciphertexts(self) -> int:
        # baseline 下：输出仍按 token 组织
        return self.num_tokens

    def per_token_op_counts(self) -> dict:
        """
        基于当前 he_linear_plain(...) 的实现方式，估算单个 token 的原语数量。

        当前 baseline 逻辑大致是：
          对每个输出维 j：
            1) 先做一遍 mult_plain
            2) 再把长度 hidden_dim 的槽位累加到 slot0
               - 需要 (hidden_dim - 1) 次 rotate
               - 需要 (hidden_dim - 1) 次 add
            3) 如果有 bias，做 1 次 add_plain
            4) 把 slot0 放到输出槽位 j（j=0 时可视为不需要额外 rotate）
            5) 再和已有输出累加

        这里给的是“和当前 baseline 实现一致的粗略计数”。
        """
        mult_plain = self.out_dim
        reduce_rotates = self.out_dim * max(self.hidden_dim - 1, 0)
        reduce_adds = self.out_dim * max(self.hidden_dim - 1, 0)

        # 输出放置：第 0 个输出不用 rotate，其余 out_dim-1 个输出需要放到对应槽位
        place_rotates = max(self.out_dim - 1, 0)

        # 多个输出向量累加成最终结果
        output_adds = max(self.out_dim - 1, 0)

        add_plain = self.out_dim if self.has_bias else 0

        total_rotates = reduce_rotates + place_rotates
        total_adds = reduce_adds + output_adds

        return {
            "mult_plain": mult_plain,
            "rotate": total_rotates,
            "add": total_adds,
            "add_plain": add_plain,
        }

    def total_op_counts(self) -> dict:
        per_token = self.per_token_op_counts()
        return {k: v * self.num_tokens for k, v in per_token.items()}

    def summary(self) -> dict:
        return {
            "layout_name": "single_token_baseline",
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "num_tokens": self.num_tokens,
            "input_ciphertexts": self.input_ciphertexts,
            "output_ciphertexts": self.output_ciphertexts,
            "per_token_op_counts": self.per_token_op_counts(),
            "total_op_counts": self.total_op_counts(),
        }


@dataclass(frozen=True)
class MatrixBlockLayout:
    """
    matrix-block 主方案的统计模型。

    逻辑层：
      - 输入先看成 2D 矩阵 X_flat = (B*T, hidden_dim)
      - 权重看成 W = (hidden_dim, out_dim)
      - 再按 logical block = block_rows x block_cols 切块

    当前为了方法收束，先把“逻辑 block”和“物理 tile”区分开：

      logical block:
        默认固定为 128 x 128

      role-specific tile:
        activation tile = 128 x 64
        weight tile     = 64 x 128

    注意：
      这一步仍然是“方法统计层”，不是最终执行内核。
    """
    batch_size: int
    seq_len: int
    hidden_dim: int
    out_dim: int

    ring_dim: int = 16384
    block_rows: int = 128
    block_cols: int = 128

    # 当前 method side 先固定角色化 tile
    act_tile_rows: int = 128
    act_tile_cols: int = 64
    weight_tile_rows: int = 64
    weight_tile_cols: int = 128

    @property
    def rows(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self.rows, self.hidden_dim)

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.hidden_dim, self.out_dim)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.rows, self.out_dim)

    @property
    def slots_per_ct(self) -> int:
        # 这里沿用当前项目中的经验设定：CKKS 可用槽位量级约为 ring_dim / 2
        return self.ring_dim // 2

    @property
    def padded_input_shape(self) -> tuple[int, int]:
        return (
            pad_to_multiple(self.rows, self.block_rows),
            pad_to_multiple(self.hidden_dim, self.block_cols),
        )

    @property
    def padded_weight_shape(self) -> tuple[int, int]:
        return (
            pad_to_multiple(self.hidden_dim, self.block_rows),
            pad_to_multiple(self.out_dim, self.block_cols),
        )

    @property
    def padded_output_shape(self) -> tuple[int, int]:
        return (
            self.padded_input_shape[0],
            self.padded_weight_shape[1],
        )

    @property
    def num_row_blocks(self) -> int:
        return self.padded_input_shape[0] // self.block_rows

    @property
    def num_mid_blocks(self) -> int:
        return self.padded_input_shape[1] // self.block_cols

    @property
    def num_col_blocks(self) -> int:
        return self.padded_weight_shape[1] // self.block_cols

    @property
    def input_logical_blocks(self) -> int:
        return self.num_row_blocks * self.num_mid_blocks

    @property
    def weight_logical_blocks(self) -> int:
        return self.num_mid_blocks * self.num_col_blocks

    @property
    def output_logical_blocks(self) -> int:
        return self.num_row_blocks * self.num_col_blocks

    @property
    def activation_tiles_per_logical_block(self) -> int:
        if self.block_cols % self.act_tile_cols != 0:
            raise ValueError("block_cols must be divisible by act_tile_cols")
        return self.block_cols // self.act_tile_cols

    @property
    def weight_tiles_per_logical_block(self) -> int:
        if self.block_rows % self.weight_tile_rows != 0:
            raise ValueError("block_rows must be divisible by weight_tile_rows")
        return self.block_rows // self.weight_tile_rows

    @property
    def input_physical_tiles(self) -> int:
        return self.input_logical_blocks * self.activation_tiles_per_logical_block

    @property
    def weight_physical_tiles(self) -> int:
        return self.weight_logical_blocks * self.weight_tiles_per_logical_block

    @property
    def output_physical_tiles(self) -> int:
        # 输出按 activation 风格继续传递，方便后续接下一层
        return self.output_logical_blocks * self.activation_tiles_per_logical_block

    @property
    def logical_block_products(self) -> int:
        """
        外层 block GEMM 里，逻辑 block 乘法项的总数：
            num_row_blocks * num_col_blocks * num_mid_blocks
        """
        return self.num_row_blocks * self.num_col_blocks * self.num_mid_blocks

    def summary(self) -> dict:
        return {
            "layout_name": "matrix_block_layout",
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "ring_dim": self.ring_dim,
            "slots_per_ct": self.slots_per_ct,
            "input_shape": self.input_shape,
            "weight_shape": self.weight_shape,
            "output_shape": self.output_shape,
            "logical_block_shape": (self.block_rows, self.block_cols),
            "activation_tile_shape": (self.act_tile_rows, self.act_tile_cols),
            "weight_tile_shape": (self.weight_tile_rows, self.weight_tile_cols),
            "padded_input_shape": self.padded_input_shape,
            "padded_weight_shape": self.padded_weight_shape,
            "padded_output_shape": self.padded_output_shape,
            "num_row_blocks": self.num_row_blocks,
            "num_mid_blocks": self.num_mid_blocks,
            "num_col_blocks": self.num_col_blocks,
            "input_logical_blocks": self.input_logical_blocks,
            "weight_logical_blocks": self.weight_logical_blocks,
            "output_logical_blocks": self.output_logical_blocks,
            "activation_tiles_per_logical_block": self.activation_tiles_per_logical_block,
            "weight_tiles_per_logical_block": self.weight_tiles_per_logical_block,
            "input_physical_tiles": self.input_physical_tiles,
            "weight_physical_tiles": self.weight_physical_tiles,
            "output_physical_tiles": self.output_physical_tiles,
            "logical_block_products": self.logical_block_products,
        }
