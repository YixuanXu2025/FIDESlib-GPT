from __future__ import annotations

from typing import Iterable, Optional

from _fideslib import FidesCKKSContext as _NativeFidesCKKSContext

from .config import HEConfig


class FidesContext:
    """
    对 _fideslib.FidesCKKSContext 的轻包装。
    这一版先只透传 first_mod_size / num_large_digits。
    """

    def __init__(
        self,
        multiplicative_depth: int = 2,
        scaling_mod_size: int = 50,
        batch_size: int = 8,
        ring_dim: int = 1 << 14,
        first_mod_size: Optional[int] = None,
        num_large_digits: Optional[int] = None,
        devices=None,
        plaintext_autoload: bool = True,
        ciphertext_autoload: bool = True,
        with_mult_key: bool = True,
        rotation_steps: Iterable[int] = (),
    ):
        if devices is None:
            devices = [0]

        self._ctx = _NativeFidesCKKSContext()
        self._ctx.init(
            multiplicative_depth=int(multiplicative_depth),
            scaling_mod_size=int(scaling_mod_size),
            batch_size=int(batch_size),
            ring_dim=int(ring_dim),
            first_mod_size=-1 if first_mod_size is None else int(first_mod_size),
            num_large_digits=-1 if num_large_digits is None else int(num_large_digits),
            devices=list(devices),
            plaintext_autoload=bool(plaintext_autoload),
            ciphertext_autoload=bool(ciphertext_autoload),
            with_mult_key=bool(with_mult_key),
            rotation_steps=[int(s) for s in rotation_steps],
        )

    # ----------------------------------------------------------
    # 旧最小接口
    # ----------------------------------------------------------

    def info(self):
        return dict(self._ctx.info())

    def roundtrip(self, x):
        return self._ctx.roundtrip([float(v) for v in x])

    def add(self, x, y):
        return self._ctx.eval_add(
            [float(v) for v in x],
            [float(v) for v in y],
        )

    def mult_scalar(self, x, scalar):
        return self._ctx.eval_mult_scalar(
            [float(v) for v in x],
            float(scalar),
        )

    # ----------------------------------------------------------
    # ciphertext-handle 接口
    # ----------------------------------------------------------

    def encrypt(self, x):
        return self._ctx.encrypt([float(v) for v in x])

    def decrypt(self, ciphertext, logical_length: int = 0):
        return self._ctx.decrypt(ciphertext, int(logical_length))


    def inspect_rlwe_components_cpu(self, ciphertext, coeff_sample: int = 8):
        return self._ctx.inspect_rlwe_components_cpu(ciphertext, int(coeff_sample))


    def roundtrip_rlwe_components_cpu(self, ciphertext):
        return self._ctx.roundtrip_rlwe_components_cpu(ciphertext)


    def component_int_linear_combination_cpu(self, rows, weights):
        return self._ctx.component_int_linear_combination_cpu(rows, weights)



    def ciphertext_storage_state(self, ciphertext):
        return self._ctx.ciphertext_storage_state(ciphertext)

    def component_linear_wsum_gpu(self, rows, weights):
        return self._ctx.component_linear_wsum_gpu(rows, [float(v) for v in weights])


    def component_linear_wsum_gpu_fused_raw(self, rows, weights):
        return self._ctx.component_linear_wsum_gpu_fused_raw(rows, [float(v) for v in weights])


    def component_linear_matmul_gpu_fused_raw(self, rows, U, copyback: bool = True):
        U2 = [[float(v) for v in row] for row in U]
        return self._ctx.component_linear_matmul_gpu_fused_raw(rows, U2, bool(copyback))


    def gpu_copyback_cpu_debug(self, ciphertext, rev: int = 0):
        return self._ctx.gpu_copyback_cpu_debug(ciphertext, int(rev))


    def add_ct(self, a, b):
        return self._ctx.eval_add_ct(a, b)

    def add_plain_ct(self, a, plain):
        return self._ctx.eval_add_plain_ct(a, [float(v) for v in plain])

    def mult_scalar_ct(self, a, scalar: float):
        return self._ctx.eval_mult_scalar_ct(a, float(scalar))

    def mult_plain_ct(self, a, plain):
        return self._ctx.eval_mult_plain_ct(a, [float(v) for v in plain])

    def rotate_ct(self, a, steps: int):
        return self._ctx.eval_rotate_ct(a, int(steps))

    def close(self):
        if getattr(self, "_ctx", None) is not None:
            self._ctx.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class HERuntime:
    """
    工程正式运行时入口。
    """

    def __init__(self, cfg: HEConfig, rotation_steps: Iterable[int] = ()):
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
            first_mod_size=self.cfg.first_mod_size,
            num_large_digits=self.cfg.num_large_digits,
            devices=list(self.cfg.devices),
            plaintext_autoload=self.cfg.plaintext_autoload,
            ciphertext_autoload=self.cfg.ciphertext_autoload,
            with_mult_key=self.cfg.with_mult_key,
            rotation_steps=self.rotation_steps,
        )
        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized and self.ctx is not None

    def require_context(self) -> FidesContext:
        if not self.is_initialized():
            raise RuntimeError("HERuntime is not initialized")
        return self.ctx

    # ----------------------------------------------------------
    # 旧最小接口
    # ----------------------------------------------------------

    def info(self):
        return self.require_context().info()

    def roundtrip(self, x):
        return self.require_context().roundtrip(x)

    def add(self, x, y):
        return self.require_context().add(x, y)

    def mult_scalar(self, x, scalar):
        return self.require_context().mult_scalar(x, scalar)

    # ----------------------------------------------------------
    # ciphertext-handle 接口
    # ----------------------------------------------------------

    def encrypt(self, x):
        return self.require_context().encrypt(x)

    def decrypt(self, ciphertext, logical_length: int = 0):
        return self.require_context().decrypt(ciphertext, logical_length=logical_length)


    def inspect_rlwe_components_cpu(self, ciphertext, coeff_sample: int = 8):
        return self.require_context().inspect_rlwe_components_cpu(ciphertext, coeff_sample=coeff_sample)


    def roundtrip_rlwe_components_cpu(self, ciphertext):
        return self.require_context().roundtrip_rlwe_components_cpu(ciphertext)


    def component_int_linear_combination_cpu(self, rows, weights):
        return self.require_context().component_int_linear_combination_cpu(rows, weights)



    def ciphertext_storage_state(self, ciphertext):
        return self.require_context().ciphertext_storage_state(ciphertext)

    def component_linear_wsum_gpu(self, rows, weights):
        return self.require_context().component_linear_wsum_gpu(rows, weights)


    def component_linear_wsum_gpu_fused_raw(self, rows, weights):
        return self.require_context().component_linear_wsum_gpu_fused_raw(rows, weights)


    def component_linear_matmul_gpu_fused_raw(self, rows, U, copyback: bool = True):
        return self.require_context().component_linear_matmul_gpu_fused_raw(rows, U, copyback=copyback)


    def gpu_copyback_cpu_debug(self, ciphertext, rev: int = 0):
        return self.require_context().gpu_copyback_cpu_debug(ciphertext, rev=rev)


    def add_ct(self, a, b):
        return self.require_context().add_ct(a, b)

    def add_plain_ct(self, a, plain):
        return self.require_context().add_plain_ct(a, plain)

    def mult_scalar_ct(self, a, scalar: float):
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




