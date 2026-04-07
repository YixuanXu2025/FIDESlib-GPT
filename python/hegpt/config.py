from dataclasses import dataclass, field
from typing import Sequence, Tuple


@dataclass
class GPT2Config:
    n_vocab: int = 0
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        return self.n_embd // self.n_head

    @property
    def mlp_hidden_dim(self) -> int:
        return 4 * self.n_embd


@dataclass
class HEConfig:
    ring_dim: int = 1 << 14
    multiplicative_depth: int = 2
    scaling_mod_size: int = 50
    batch_size: int = 8

    devices: Tuple[int, ...] = (0,)
    plaintext_autoload: bool = True
    ciphertext_autoload: bool = True
    with_mult_key: bool = True


@dataclass
class ApproxConfig:
    gelu_fit_degree: int = 6
    gelu_fit_interval: Tuple[float, float] = (-3.0, 3.0)


@dataclass
class ProjectConfig:
    gpt2: GPT2Config = field(default_factory=GPT2Config)
    he: HEConfig = field(default_factory=HEConfig)
    approx: ApproxConfig = field(default_factory=ApproxConfig)

    infer_batch_size: int = 4
    infer_seq_len: int = 16

    def summary(self) -> dict:
        return {
            "n_vocab": self.gpt2.n_vocab,
            "n_ctx": self.gpt2.n_ctx,
            "n_embd": self.gpt2.n_embd,
            "n_head": self.gpt2.n_head,
            "n_layer": self.gpt2.n_layer,
            "head_dim": self.gpt2.head_dim,
            "mlp_hidden_dim": self.gpt2.mlp_hidden_dim,
            "infer_batch_size": self.infer_batch_size,
            "infer_seq_len": self.infer_seq_len,
            "ring_dim": self.he.ring_dim,
            "multiplicative_depth": self.he.multiplicative_depth,
            "scaling_mod_size": self.he.scaling_mod_size,
            "he_batch_size": self.he.batch_size,
            "devices": list(self.he.devices),
            "gelu_fit_degree": self.approx.gelu_fit_degree,
            "gelu_fit_interval": self.approx.gelu_fit_interval,
        }
