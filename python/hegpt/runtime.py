from __future__ import annotations

from typing import Iterable, Optional

from _fideslib import FidesCKKSContext as _NativeFidesCKKSContext

from .config import HEConfig


class FidesContext:
    """
    对 _fideslib.FidesCKKSContext 的轻包装。
    这一版先只透传 first_mod_size / num_large_digits。
    """

    def export_secret_key_coeff_u64(self, coeff_count: int = 0):
        """
        Debug-only helper for controlled RLWE coefficient-row probes.

        Returns:
            list[num_towers][coeff_count] of secret-key coefficients as unsigned residues.
        """
        return self._ctx.export_secret_key_coeff_u64(int(coeff_count))


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


    def component_linear_transform_gpu(self, rows, U):
        """
        GPU-only fused raw component linear transform.

        This performs CCMM-like component linear matmul and returns GPU-resident
        ciphertext handles. It intentionally does not copy numerical ciphertext
        data back to CPU/OpenFHE.
        """
        U2 = [[float(v) for v in row] for row in U]
        return self._ctx.component_linear_matmul_gpu_fused_raw(rows, U2, False)


    def component_linear_transform_gpu_materialized(self, rows, U):
        """
        Fused raw component linear transform with immediate GPU->CPU/OpenFHE materialization.

        This is the formal name for the old copyback=True behavior.
        """
        U2 = [[float(v) for v in row] for row in U]
        return self._ctx.component_linear_matmul_gpu_fused_raw(rows, U2, True)


    def ccmm_gpu_fused_raw(self, rows, U):
        raise NotImplementedError(
            "ccmm_gpu_fused_raw was a misleading name. "
            "The existing implementation is only a GPU component linear transform "
            "(plaintext matrix times ciphertext rows), not Park-2025 ciphertext-ciphertext matrix multiplication. "
            "Use component_linear_transform_gpu(...) for the existing primitive, "
            "and implement true CCMM through hegpt.ccmm_paper2025."
        )


    def materialize_gpu_ciphertext(self, ciphertext):
        return self._ctx.materialize_gpu_ciphertext(ciphertext)


    def materialize_gpu_ciphertexts(self, ciphertexts):
        return self._ctx.materialize_gpu_ciphertexts(ciphertexts)


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


    def encrypt_coeff_row_i64(self, coeffs):
        return self._ctx.encrypt_coeff_row_i64([int(v) for v in coeffs])


    def decrypt_coeff_row_i64(self, ciphertext, logical_length: int = 0):
        return self._ctx.decrypt_coeff_row_i64(ciphertext, int(logical_length))

    def manual_decrypt_coeff_row_i64(self, ciphertext, logical_length: int = 0):
        return self._ctx.manual_decrypt_coeff_row_i64(ciphertext, int(logical_length))

    def manual_decrypt_coeff_row_i64_variants(self, ciphertext, logical_length: int = 0):
        return self._ctx.manual_decrypt_coeff_row_i64_variants(ciphertext, int(logical_length))

    def inspect_coeff_row_component_matrix(self, rows, coeff_sample: int = 8):
        return self._ctx.inspect_coeff_row_component_matrix(rows, int(coeff_sample))



    def eval_automorphism_coeff_ct(self, ciphertext, alpha: int):
        return self._ctx.eval_automorphism_coeff_ct(ciphertext, int(alpha))

    def add_coeff_ct(self, a, b):
        return self._ctx.add_coeff_ct(a, b)


    def monomial_mul_coeff_ct(self, ciphertext, exp: int):
        return self._ctx.monomial_mul_coeff_ct(ciphertext, int(exp))

    def mul_coeff_ct_no_relin(self, a, b):
        return self._ctx.mul_coeff_ct_no_relin(a, b)

    def relinearize_coeff_ct(self, ciphertext):
        return self._ctx.relinearize_coeff_ct(ciphertext)

    def compress_coeff_ct(self, ciphertext, towers_left: int = 1):
        return self._ctx.compress_coeff_ct(ciphertext, int(towers_left))

    def component_mul_part_coeff_ct(self, a, part_a: int, b, part_b: int):
        return self._ctx.component_mul_part_coeff_ct(a, int(part_a), b, int(part_b))


    def component_add_1part_coeff_ct(self, a, b):
        return self._ctx.component_add_1part_coeff_ct(a, b)

    def assemble_3part_from_1parts_coeff_ct(self, c0, c1, c2):
        return self._ctx.assemble_3part_from_1parts_coeff_ct(c0, c1, c2)

    def export_component_coeff_matrix_u64(self, rows, part: int, coeff_count: int = 0):
        return self._ctx.export_component_coeff_matrix_u64(rows, int(part), int(coeff_count))


    def import_1part_coeff_u64(self, template_ct, towers_coeffs):
        return self._ctx.import_1part_coeff_u64(template_ct, towers_coeffs)

    def assemble_2part_from_1parts_coeff_ct(self, c0, c1):
        return self._ctx.assemble_2part_from_1parts_coeff_ct(c0, c1)

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

    def export_secret_key_coeff_u64(self, coeff_count: int = 0):
        """
        Debug-only helper for controlled RLWE coefficient-row probes.

        Returns:
            list[num_towers][coeff_count] of secret-key coefficients as unsigned residues.
        """
        return self.require_context().export_secret_key_coeff_u64(int(coeff_count))


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


    def component_linear_transform_gpu(self, rows, U):
        """
        GPU-only fused raw component linear transform.

        Use this for intermediate CCMM layers. It returns ciphertext handles that
        remain GPU-resident but still carry CPU-side metadata templates.
        """
        return self.require_context().component_linear_transform_gpu(rows, U)


    def component_linear_transform_gpu_materialized(self, rows, U):
        """
        Fused raw component linear transform followed by immediate materialization.

        Use this for baselines or final layers when you want CPU/OpenFHE-ready
        ciphertexts immediately.
        """
        return self.require_context().component_linear_transform_gpu_materialized(rows, U)


    def ccmm_gpu_fused_raw(self, rows, U):
        raise NotImplementedError(
            "ccmm_gpu_fused_raw was a misleading name. "
            "The existing implementation is only a GPU component linear transform "
            "(plaintext matrix times ciphertext rows), not Park-2025 ciphertext-ciphertext matrix multiplication. "
            "Use component_linear_transform_gpu(...) for the existing primitive, "
            "and implement true CCMM through hegpt.ccmm_paper2025."
        )


    def materialize_gpu_ciphertext(self, ciphertext):
        """
        Explicitly materialize one GPU-resident ciphertext into CPU/OpenFHE form.
        """
        return self.require_context().materialize_gpu_ciphertext(ciphertext)


    def materialize_gpu_ciphertexts(self, ciphertexts):
        """
        Explicitly materialize a list of GPU-resident ciphertexts into CPU/OpenFHE form.
        """
        return self.require_context().materialize_gpu_ciphertexts(ciphertexts)


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

    def encrypt_coeff_row_i64(self, coeffs):
        """
        Encrypt a coefficient-encoded integer row using native OpenFHE CoefPackedEncoding.

        This is for Park-2025 CC-MM coefficient-row probing.
        """
        return self.require_context().encrypt_coeff_row_i64(coeffs)


    def decrypt_coeff_row_i64(self, ciphertext, logical_length: int = 0):
        """
        Decrypt a coefficient-encoded integer row.
        """
        return self.require_context().decrypt_coeff_row_i64(
            ciphertext,
            logical_length=logical_length,
        )

    def manual_decrypt_coeff_row_i64(self, ciphertext, logical_length: int = 0):
        """
        Debug-only manual coefficient decrypt from RLWE components.
        """
        return self.require_context().manual_decrypt_coeff_row_i64(
            ciphertext,
            logical_length=logical_length,
        )

    def manual_decrypt_coeff_row_i64_variants(self, ciphertext, logical_length: int = 0):
        """
        Debug-only manual coefficient decrypt variants from RLWE components.
        """
        return self.require_context().manual_decrypt_coeff_row_i64_variants(
            ciphertext,
            logical_length=logical_length,
        )

    def inspect_coeff_row_component_matrix(self, rows, coeff_sample: int = 8):
        """
        Inspect row-wise coefficient ciphertext bundle as paper-style A/B component matrices.
        """
        return self.require_context().inspect_coeff_row_component_matrix(
            rows,
            coeff_sample=coeff_sample,
        )

    def eval_automorphism_coeff_ct(self, ciphertext, alpha: int):
        """
        Native OpenFHE Auto(ct; alpha) for coefficient ciphertext.
        """
        return self.require_context().eval_automorphism_coeff_ct(
            ciphertext,
            alpha=int(alpha),
        )

    def add_coeff_ct(self, a, b):
        """
        Native OpenFHE coefficient ciphertext add.
        """
        return self.require_context().add_coeff_ct(a, b)


    def monomial_mul_coeff_ct(self, ciphertext, exp: int):
        """
        Component-wise X^exp · ct for coefficient ciphertext.
        """
        return self.require_context().monomial_mul_coeff_ct(
            ciphertext,
            exp=int(exp),
        )

    def mul_coeff_ct_no_relin(self, a, b):
        """
        Native component-level ciphertext-ciphertext multiplication without relinearization.

        Input:
          two coefficient ciphertexts with c0/c1 components.

        Output:
          one ciphertext with c0/c1/c2 components.
        """
        return self.require_context().mul_coeff_ct_no_relin(a, b)

    def relinearize_coeff_ct(self, ciphertext):
        """
        Native OpenFHE relinearization for coefficient ciphertext.

        Input:
          ciphertext with c0/c1/c2 components.

        Output:
          ciphertext with c0/c1 components.
        """
        return self.require_context().relinearize_coeff_ct(ciphertext)

    def compress_coeff_ct(self, ciphertext, towers_left: int = 1):
        """
        Compress coefficient ciphertext to a target number of RNS towers.

        Intended use:
          final coefficient ciphertext with sizeQl > 1
            -> towers_left=1
            -> decrypt_coeff_row_i64
        """
        return self.require_context().compress_coeff_ct(
            ciphertext,
            towers_left=int(towers_left),
        )

    def component_mul_part_coeff_ct(self, a, part_a: int, b, part_b: int):
        """
        Multiply selected DCRTPoly components from two coefficient ciphertexts.

        Output:
          a ciphertext-like handle with exactly one DCRTPoly component.
        """
        return self.require_context().component_mul_part_coeff_ct(
            a,
            int(part_a),
            b,
            int(part_b),
        )


    def component_add_1part_coeff_ct(self, a, b):
        """
        Add two one-component DCRTPoly ciphertext-like handles.
        """
        return self.require_context().component_add_1part_coeff_ct(a, b)

    def assemble_3part_from_1parts_coeff_ct(self, c0, c1, c2):
        """
        Assemble three one-component DCRTPoly handles into one three-component ciphertext-like handle.

        Used for paper-aligned Algorithm 3:
          C0 = PP-MM(L0,R0)
          C1 = PP-MM(L0,R1) + PP-MM(L1,R0)
          C2 = PP-MM(L1,R1)
        """
        return self.require_context().assemble_3part_from_1parts_coeff_ct(c0, c1, c2)

    def export_component_coeff_matrix_u64(self, rows, part: int, coeff_count: int = 0):
        """
        Export selected c0/c1 component from a row bundle as raw RNS coefficient data.

        Output layout:
          rows × towers × coeffs_u64
        """
        return self.require_context().export_component_coeff_matrix_u64(
            rows,
            int(part),
            int(coeff_count),
        )


    def import_1part_coeff_u64(self, template_ct, towers_coeffs):
        """
        Import raw RNS coefficient tower arrays into a one-component ciphertext-like handle.
        """
        return self.require_context().import_1part_coeff_u64(template_ct, towers_coeffs)

    def assemble_2part_from_1parts_coeff_ct(self, c0, c1):
        """
        Assemble two one-component DCRTPoly handles into one two-component ciphertext-like handle.

        Component order:
          c0 = B
          c1 = A
        """
        return self.require_context().assemble_2part_from_1parts_coeff_ct(c0, c1)

