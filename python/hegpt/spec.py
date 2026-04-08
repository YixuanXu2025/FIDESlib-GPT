from __future__ import annotations

from dataclasses import dataclass

from .layouts import MatrixBlockLayout, SingleTokenBaselineLayout


@dataclass(frozen=True)
class ProjectShapeSpec:
    """
    工作包 1 用的统一形状规格。

    逻辑语义：
      - 输入形状默认看成 (B, T, N)
      - 线性层权重形状看成 (N, M)

    这个类的作用是：
      1) 把论文方法层里最常见的张量形状固定下来
      2) 为 single-token baseline 与 matrix-block 主方案提供统一入口
    """
    batch_size: int
    seq_len: int
    hidden_dim: int
    out_dim: int

    @property
    def flat_rows(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def input_shape_3d(self) -> tuple[int, int, int]:
        return (self.batch_size, self.seq_len, self.hidden_dim)

    @property
    def input_shape_2d(self) -> tuple[int, int]:
        return (self.flat_rows, self.hidden_dim)

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.hidden_dim, self.out_dim)

    @property
    def output_shape_2d(self) -> tuple[int, int]:
        return (self.flat_rows, self.out_dim)

    def make_single_token_layout(self, has_bias: bool = True) -> SingleTokenBaselineLayout:
        return SingleTokenBaselineLayout(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            has_bias=has_bias,
        )

    def make_matrix_block_layout(
        self,
        ring_dim: int = 16384,
        block_rows: int = 128,
        block_cols: int = 128,
        act_tile_rows: int = 128,
        act_tile_cols: int = 64,
        weight_tile_rows: int = 64,
        weight_tile_cols: int = 128,
    ) -> MatrixBlockLayout:
        return MatrixBlockLayout(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            ring_dim=ring_dim,
            block_rows=block_rows,
            block_cols=block_cols,
            act_tile_rows=act_tile_rows,
            act_tile_cols=act_tile_cols,
            weight_tile_rows=weight_tile_rows,
            weight_tile_cols=weight_tile_cols,
        )

    def summary(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "input_shape_3d": self.input_shape_3d,
            "input_shape_2d": self.input_shape_2d,
            "weight_shape": self.weight_shape,
            "output_shape_2d": self.output_shape_2d,
        }
