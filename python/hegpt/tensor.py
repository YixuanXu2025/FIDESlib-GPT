from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple


@dataclass
class CipherTensor:
    """
    对“密文对象 + 张量语义”的最小封装。

    注意：
    - ct: 目前先放底层密文句柄（未来由 _fideslib 暴露）
    - shape/layout/slots_used: 由 Python 上层维护
    """
    ct: Any = None
    shape: Tuple[int, ...] = ()
    layout: str = "unknown"
    slots_used: int = 0
    level: Optional[int] = None
    scale: Optional[float] = None
    device: Optional[int] = None
    name: str = ""

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        if not self.shape:
            return 0
        n = 1
        for d in self.shape:
            n *= d
        return n

    def has_cipher(self) -> bool:
        return self.ct is not None

    def ensure_cipher(self):
        if self.ct is None:
            raise RuntimeError(
                f"CipherTensor(name={self.name!r}, shape={self.shape}) does not hold a ciphertext handle yet"
            )

    def clone_with(
        self,
        *,
        ct: Any = None,
        shape: Optional[Tuple[int, ...]] = None,
        layout: Optional[str] = None,
        slots_used: Optional[int] = None,
        level: Optional[int] = None,
        scale: Optional[float] = None,
        device: Optional[int] = None,
        name: Optional[str] = None,
    ) -> "CipherTensor":
        return CipherTensor(
            ct=self.ct if ct is None else ct,
            shape=self.shape if shape is None else shape,
            layout=self.layout if layout is None else layout,
            slots_used=self.slots_used if slots_used is None else slots_used,
            level=self.level if level is None else level,
            scale=self.scale if scale is None else scale,
            device=self.device if device is None else device,
            name=self.name if name is None else name,
        )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "shape": self.shape,
            "layout": self.layout,
            "slots_used": self.slots_used,
            "level": self.level,
            "scale": self.scale,
            "device": self.device,
            "has_cipher": self.has_cipher(),
        }


def make_token_cipher_placeholder(
    *,
    hidden_dim: int = 768,
    device: Optional[int] = None,
    name: str = "",
) -> CipherTensor:
    """
    第一版最常用布局：
    一个密文 = 一个 token 的 hidden 向量。
    """
    return CipherTensor(
        ct=None,
        shape=(hidden_dim,),
        layout="token_hidden_contiguous",
        slots_used=hidden_dim,
        level=None,
        scale=None,
        device=device,
        name=name,
    )
