from __future__ import annotations

from typing import Optional, Sequence, Tuple

from _fideslib import (
    CiphertextHandle as _NativeCiphertextHandle,
    FidesCKKSContext as _NativeFidesCKKSContext,
)

from .config import HEConfig


def _to_float_list(x):
    return [float(v) for v in x]


class FidesContext:
    """
    对 _fideslib.FidesCKKSContext 的 Python 友好包装。

    当前支持两类接口：
    1. 旧接口（兼容你之前的最小测试）：
       - info()
       - roundtrip()
       - add(x, y)
       - mult_scalar(x, s)

    2. 新接口（真正的密文句柄式接口）：
       - encrypt(x) -> CiphertextHandle
       - decrypt(ct)
       - add_ct(...)
       - add_plain_ct(...)
       - mult_scalar_ct(...)
       - mult_plain_ct(...)
       - rotate_ct(...)
    """

    def __init__(
        self,
        multiplicative_depth=2,
        scaling_mod_size=50,
        batch_size=8,
        ring_dim=1 << 14,
        devices=None,
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
        rotation_steps=None,
    ):
        if devices is None:
            devices = [0]
        if rotation_steps is None:
            rotation_steps = []

        self._ctx = _NativeFidesCKKSContext()
        self._ctx.init(
            multiplicative_depth=multiplicative_depth,
            scaling_mod_size=scaling_mod_size,
            batch_size=batch_size,
            ring_dim=ring_dim,
            devices=list(devices),
            plaintext_autoload=plaintext_autoload,
            ciphertext_autoload=ciphertext_autoload,
            with_mult_key=with_mult_key,
            rotation_steps=list(rotation_steps),
        )

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def info(self):
        return dict(self._ctx.info())

    # ------------------------------------------------------------------
    # 新：密文句柄式接口
    # ------------------------------------------------------------------

    def encrypt(self, x):
        return self._ctx.encrypt(_to_float_list(x))

    def decrypt(self, ct, logical_length=0):
        return self._ctx.decrypt(ct, int(logical_length))

    def add_ct(self, a, b):
        return self._ctx.eval_add_ct(a, b)

    def add_plain_ct(self, a, plain):
        return self._ctx.eval_add_plain_ct(a, _to_float_list(plain))

    def mult_scalar_ct(self, a, scalar):
        return self._ctx.eval_mult_scalar_ct(a, float(scalar))

    def mult_plain_ct(self, a, plain):
        return self._ctx.eval_mult_plain_ct(a, _to_float_list(plain))

    def rotate_ct(self, a, steps: int):
        return self._ctx.eval_rotate_ct(a, int(steps))

    # ------------------------------------------------------------------
    # 旧：兼容最小测试
    # ------------------------------------------------------------------

    def roundtrip(self, x):
        return self._ctx.roundtrip(_to_float_list(x))

    def add(self, x, y):
        return self._ctx.eval_add(
            _to_float_list(x),
            _to_float_list(y),
        )

    def mult_scalar(self, x, scalar):
        return self._ctx.eval_mult_scalar(
            _to_float_list(x),
            float(scalar),
        )

    def close(self):
        if getattr(self, "_ctx", None) is not None:
            self._ctx.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class HERuntime:
    """
    工程级运行时入口。

    这层负责：
    - 统一持有 FidesContext
    - 初始化 HE 参数
    - 对上层提供稳定接口

    注意：
    - rotation_steps 需要在 init 时就提供，因为底层要求
      EvalRotateKeyGen 必须在 LoadContext 之前执行。
    """

    def __init__(self, cfg: HEConfig, *, rotation_steps=()):
        self.cfg = cfg
        self.rotation_steps = tuple(int(s) for s in rotation_steps)
        self.ctx: Optional[FidesContext] = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        self.ctx = FidesContext(
            multiplicative_depth=self.cfg.multiplicative_depth,
            scaling_mod_size=self.cfg.scaling_mod_size,
            batch_size=self.cfg.batch_size,
            ring_dim=self.cfg.ring_dim,
            devices=list(self.cfg.devices),
            plaintext_autoload=self.cfg.plaintext_autoload,
            ciphertext_autoload=self.cfg.ciphertext_autoload,
            with_mult_key=self.cfg.with_mult_key,
            rotation_steps=list(self.rotation_steps),
        )
        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized and self.ctx is not None

    def require_context(self) -> FidesContext:
        if not self.is_initialized():
            raise RuntimeError("HERuntime is not initialized")
        return self.ctx

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def info(self):
        return self.require_context().info()

    # ------------------------------------------------------------------
    # 旧：兼容最小接口
    # ------------------------------------------------------------------

    def roundtrip(self, x):
        return self.require_context().roundtrip(x)

    def add(self, x, y):
        return self.require_context().add(x, y)

    def mult_scalar(self, x, scalar):
        return self.require_context().mult_scalar(x, scalar)

    # ------------------------------------------------------------------
    # 新：真正的密文句柄式接口
    # ------------------------------------------------------------------

    def encrypt(self, x):
        return self.require_context().encrypt(x)

    def decrypt(self, ct, logical_length=0):
        return self.require_context().decrypt(ct, logical_length)

    def add_ct(self, a, b):
        return self.require_context().add_ct(a, b)

    def add_plain_ct(self, a, plain):
        return self.require_context().add_plain_ct(a, plain)

    def mult_scalar_ct(self, a, scalar):
        return self.require_context().mult_scalar_ct(a, scalar)

    def mult_plain_ct(self, a, plain):
        return self.require_context().mult_plain_ct(a, plain)

    def rotate_ct(self, a, steps: int):
        return self.require_context().rotate_ct(a, steps)

    def close(self):
        if self.ctx is not None:
            self.ctx.close()
            self.ctx = None
        self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
