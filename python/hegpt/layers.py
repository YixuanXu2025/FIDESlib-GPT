from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .approx import gelu_poly6_plain
from .ops import he_linear_plain, plaintext_linear
from .params import LinearPlainParams, MLPParams, QKVParams
from .runtime import HERuntime
from .tensor import CipherTensor


@dataclass
class HELinearPlain:
    params: LinearPlainParams

    def forward(self, rt: HERuntime, x: CipherTensor) -> CipherTensor:
        return he_linear_plain(rt, x, self.params.weight, self.params.bias)

    def forward_plain_debug(self, x):
        return plaintext_linear(x, self.params.weight, self.params.bias)


@dataclass
class HEGELU:
    """
    当前先把明文 debug 路径写通。
    真正 HE 版本后续会落成：
        he_poly_eval_horner(...)
    """

    def forward(self, rt: HERuntime, x: CipherTensor) -> CipherTensor:
        raise NotImplementedError(
            "True HE GELU needs ciphertext polynomial evaluation bindings first."
        )

    def forward_plain_debug(self, x):
        return gelu_poly6_plain(x)


@dataclass
class HEQKV:
    params: QKVParams

    def __post_init__(self):
        self.q_proj = HELinearPlain(
            LinearPlainParams(self.params.w_q, self.params.b_q, name=f"{self.params.name}.q")
        )
        self.k_proj = HELinearPlain(
            LinearPlainParams(self.params.w_k, self.params.b_k, name=f"{self.params.name}.k")
        )
        self.v_proj = HELinearPlain(
            LinearPlainParams(self.params.w_v, self.params.b_v, name=f"{self.params.name}.v")
        )

    def forward(self, rt: HERuntime, x: CipherTensor):
        q = self.q_proj.forward(rt, x)
        k = self.k_proj.forward(rt, x)
        v = self.v_proj.forward(rt, x)
        return q, k, v

    def forward_plain_debug(self, x):
        q = self.q_proj.forward_plain_debug(x)
        k = self.k_proj.forward_plain_debug(x)
        v = self.v_proj.forward_plain_debug(x)
        return q, k, v


@dataclass
class HEMLP:
    params: MLPParams

    def __post_init__(self):
        self.fc_in = HELinearPlain(self.params.fc_in)
        self.gelu = HEGELU()
        self.fc_out = HELinearPlain(self.params.fc_out)

    def forward(self, rt: HERuntime, x: CipherTensor) -> CipherTensor:
        h = self.fc_in.forward(rt, x)
        h = self.gelu.forward(rt, h)
        h = self.fc_out.forward(rt, h)
        return h

    def forward_plain_debug(self, x):
        h = self.fc_in.forward_plain_debug(x)
        h = self.gelu.forward_plain_debug(h)
        h = self.fc_out.forward_plain_debug(h)
        return h
