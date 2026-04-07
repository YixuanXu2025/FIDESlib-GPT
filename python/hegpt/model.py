from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .layers import HEMLP, HEQKV
from .params import BlockParams, ModelParams
from .runtime import HERuntime
from .tensor import CipherTensor


@dataclass
class HETransformerBlock:
    params: BlockParams

    def __post_init__(self):
        self.qkv = HEQKV(self.params.qkv) if self.params.qkv is not None else None
        self.mlp = HEMLP(self.params.mlp) if self.params.mlp is not None else None

    def forward(self, rt: HERuntime, x: CipherTensor):
        raise NotImplementedError(
            "Full HE block is not ready yet. "
            "Need HE attention path + ciphertext-handle ops first."
        )

    def forward_plain_debug(self, x):
        """
        这里只作为结构占位，不代表完整 GPT2 block。
        因为 attention 还没落地，所以这里只保留接口，不做假实现。
        """
        raise NotImplementedError(
            "Plain debug path for full block is not implemented yet. "
            "We will add it after QKV + attention path design is finalized."
        )


@dataclass
class HEGPT2Model:
    params: ModelParams

    def __post_init__(self):
        self.blocks = [HETransformerBlock(bp) for bp in self.params.blocks]

    def forward(self, rt: HERuntime, x: CipherTensor):
        h = x
        for block in self.blocks:
            h = block.forward(rt, h)
        return h
