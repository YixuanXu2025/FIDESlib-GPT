from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class EmbeddingParams:
    token_embedding: Any
    position_embedding: Optional[Any] = None


@dataclass
class NormParams:
    g: Any
    b: Any
    epsilon: float = 1e-5


@dataclass
class LinearPlainParams:
    weight: Any
    bias: Optional[Any] = None
    name: str = ""


@dataclass
class QKVParams:
    w_q: Any
    b_q: Optional[Any]
    w_k: Any
    b_k: Optional[Any]
    w_v: Any
    b_v: Optional[Any]
    name: str = "qkv"


@dataclass
class MLPParams:
    fc_in: LinearPlainParams
    fc_out: LinearPlainParams
    name: str = "mlp"


@dataclass
class BlockParams:
    ln_1: Optional[NormParams] = None
    qkv: Optional[QKVParams] = None
    attn_proj: Optional[LinearPlainParams] = None
    ln_2: Optional[NormParams] = None
    mlp: Optional[MLPParams] = None
    name: str = "block"


@dataclass
class ModelParams:
    embeddings: Optional[EmbeddingParams] = None
    blocks: List[BlockParams] = field(default_factory=list)
    ln_f: Optional[NormParams] = None
    lm_head: Optional[LinearPlainParams] = None
    name: str = "model"

    def num_blocks(self) -> int:
        return len(self.blocks)
