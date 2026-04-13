import json
import math
import time
from pathlib import Path

import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.tensor import CipherTensor
from hegpt.ops import he_add, he_mult_plain, he_rotate


def now_us():
    return time.perf_counter() * 1e6


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def print_sep(title: str):
    print("=" * 100)
    print(title)


def zh_print(result: dict):
    print(f"  案例名称: {result['case_name']}")
    print(f"  方核维度 d: {result['square_dim']}")
    print(f"  原始输入形状: {result['orig_input_shape']}")
    print(f"  原始权重形状: {result['orig_weight_shape']}")
    print(f"  明文参考中位时间(us): {result['plain_reference_us_median']}")
    print(f"  输入密文数: {result['input_ct_count_per_run']}")
    print(f"  输出密文数: {result['output_ct_count_per_run']}")
    print(f"  加密中位时间(us): {result['encrypt_us_median']}")
    print(f"  核心计算中位时间(us): {result['core_us_median']}")
    print(f"  解密中位时间(us): {result['decrypt_us_median']}")
    print(f"  核心相对明文倍数: {result['core_vs_plain_ratio']}")
    print(f"  正确性误差(max_abs_err): {result['plain_vs_he_max_abs_err']}")
    print(f"  首行前8个输出: {result['y_he_first_row'][:8]}")
    print(f"  操作统计: {result['op_meta']}")


def call_he_add(rt, a, b, name):
    try:
        return he_add(rt, a, b, name=name)
    except TypeError:
        return he_add(rt, a, b)


def call_he_mult_plain(rt, x, plain_vec, name):
    try:
        return he_mult_plain(rt, x, plain_vec, name=name)
    except TypeError:
        return he_mult_plain(rt, x, plain_vec)


def call_he_rotate(rt, x, steps, name):
    try:
        return he_rotate(rt, x, steps, name=name)
    except TypeError:
        return he_rotate(rt, x, steps)


def make_cipher_tensor(rt, packed_vec, logical_dim, name):
    ct = rt.encrypt([float(v) for v in packed_vec])
    device = None
    if hasattr(rt, "cfg") and getattr(rt.cfg, "devices", None):
        device = rt.cfg.devices[0]
    return CipherTensor(
        ct=ct,
        shape=(logical_dim,),
        layout="jkls_square_kernel",
        slots_used=len(packed_vec),
        level=None,
        scale=None,
        device=device,
        name=name,
    )


def plain_rotate(vec, steps, sigma):
    vec = np.asarray(vec, dtype=np.float64)
    return np.roll(vec, -sigma * steps)


def detect_rotation_sigma(rt):
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ct = make_cipher_tensor(rt, x, 4, "rot_detect")
    ct_r = call_he_rotate(rt, ct, 1, "rot_detect_r1")
    out = np.array(rt.decrypt(ct_r.ct, logical_length=4), dtype=np.float64)

    left = np.array([1.0, 2.0, 3.0, 0.0], dtype=np.float64)
    right = np.array([3.0, 0.0, 1.0, 2.0], dtype=np.float64)

    if np.max(np.abs(out - left)) < 1e-6:
        return +1
    if np.max(np.abs(out - right)) < 1e-6:
        return -1
    raise RuntimeError(f"无法检测旋转方向: {out.tolist()}")


def flatten_row_major(mat):
    return np.asarray(mat, dtype=np.float64).reshape(-1)


def periodic_pack(vec, physical_slots):
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    L = len(vec)
    if physical_slots % L != 0:
        raise ValueError(f"physical_slots={physical_slots} 必须能被 L={L} 整除")
    repeat = physical_slots // L
    return np.tile(vec, repeat)


def pad_to_square_block(x, d):
    x = np.asarray(x, dtype=np.float64)
    r, c = x.shape
    out = np.zeros((d, d), dtype=np.float64)
    out[:r, :c] = x
    return out


def matrix_to_periodic_cipher(rt, mat_square, name):
    d = mat_square.shape[0]
    slots_per_ct = rt.info()["ring_dim"] // 2
    flat = flatten_row_major(mat_square)
    packed = periodic_pack(flat, slots_per_ct)
    return make_cipher_tensor(rt, packed, d * d, name)


def column_mask_plain_periodic(d, k, physical_slots):
    mask = np.zeros((d * d,), dtype=np.float64)
    for i in range(d):
        mask[i * d + k] = 1.0
    return periodic_pack(mask, physical_slots)


def repeat_plain_row_periodic(row_vec, d, physical_slots):
    row_vec = np.asarray(row_vec, dtype=np.float64).reshape(-1)
    assert len(row_vec) == d
    mat = np.tile(row_vec.reshape(1, d), (d, 1))
    return periodic_pack(flatten_row_major(mat), physical_slots)


def decode_periodic_square_output(rt, ct, d, out_rows, out_cols):
    slots_per_ct = rt.info()["ring_dim"] // 2
    full = np.array(rt.decrypt(ct.ct, logical_length=slots_per_ct), dtype=np.float64)
    base = full[: d * d].reshape(d, d)
    return base[:out_rows, :out_cols]


def replicate_rowwise_from_col0(rt, ct_col0, d, sigma, name_prefix):
    """
    输入:
      ct_col0 在每一行的第 0 列有值，其余列为 0
    输出:
      每一行都把第 0 列复制到整行
    """
    acc = ct_col0
    step = 1
    while step < d:
        rot_step = -sigma * step
        shifted = call_he_rotate(rt, acc, rot_step, f"{name_prefix}_rep_{step}")
        acc = call_he_add(rt, acc, shifted, f"{name_prefix}_rep_add_{step}")
        step *= 2
    return acc


def jkls_square_kernel_pcmm(rt, ct_A, B_square, d, sigma, name_prefix):
    """
    一个 first-cut JKLS square-kernel baseline:
      - A 是加密的 dxd 方阵（row-major 打包到一个 ciphertext）
      - B 是明文 dxd 方阵
      - 通过 outer-product 风格：
          C = sum_k col(A,k) * row(B,k)
        来构造
    """
    slots_per_ct = rt.info()["ring_dim"] // 2
    B_square = np.asarray(B_square, dtype=np.float64)

    acc = None

    for k in range(d):
        # 1) 取出 A 的第 k 列
        col_mask = column_mask_plain_periodic(d, k, slots_per_ct)
        ct_col = call_he_mult_plain(rt, ct_A, col_mask.tolist(), f"{name_prefix}_maskcol_{k}")

        # 2) 把第 k 列移到每行第 0 列
        move_step = sigma * k
        ct_col0 = ct_col if k == 0 else call_he_rotate(rt, ct_col, move_step, f"{name_prefix}_movecol_{k}")

        # 3) 在每一行内部把 col0 复制成整行
        ct_rowrep = replicate_rowwise_from_col0(rt, ct_col0, d, sigma, f"{name_prefix}_repl_{k}")

        # 4) 明文 B 的第 k 行复制成 dxd 矩阵
        plain_rowrep = repeat_plain_row_periodic(B_square[k, :], d, slots_per_ct)

        # 5) 对应槽逐项乘，再累加
        term = call_he_mult_plain(rt, ct_rowrep, plain_rowrep.tolist(), f"{name_prefix}_mulrow_{k}")
        acc = term if acc is None else call_he_add(rt, acc, term, f"{name_prefix}_acc_{k}")

    return acc


def jkls_square_kernel_op_counts(d, input_ct_count, output_ct_count):
    logd = int(round(math.log2(d)))
    # 每个 k:
    #   1 次列mask pMult
    #   最多 1 次 rotate 把列挪到 col0
    #   log2(d) 次 rotate + log2(d) 次 add 复制整行
    #   1 次 row-wise pMult
    #   1 次 add 累加（首项除外）
    per_kernel = {
        "pMult": 2 * d,
        "rotate": d + d * logd,
        "add": d * logd + (d - 1),
        "kernel_calls": 1,
    }
    total = {
        "pMult": per_kernel["pMult"] * output_ct_count,
        "rotate": per_kernel["rotate"] * output_ct_count,
        "add": per_kernel["add"] * output_ct_count,
        "kernel_calls": output_ct_count,
        "输入密文数": input_ct_count,
        "输出密文数": output_ct_count,
    }
    return {
        "per_kernel": per_kernel,
        "total": total,
    }


def bench_plain_reference(A_small, B_full):
    y = A_small @ B_full
    return y


def run_case(case_name, A_shape, B_shape, d=64, repeats=1, seed=0):
    rng = np.random.default_rng(seed)

    A_small = rng.normal(0.0, 1.0, size=A_shape).astype(np.float64)
    B_full = rng.normal(0.0, 0.2, size=B_shape).astype(np.float64)

    # 明文参考
    plain_times = []
    y_plain_last = None
    for _ in range(5):
        t0 = now_us()
        y_plain = bench_plain_reference(A_small, B_full)
        t1 = now_us()
        plain_times.append(t1 - t0)
        y_plain_last = y_plain

    cfg = HEConfig(
        ring_dim=16384,
        multiplicative_depth=2,
        scaling_mod_size=50,
        batch_size=8192,
        devices=(0,),
        plaintext_autoload=True,
        ciphertext_autoload=True,
        with_mult_key=True,
    )

    encrypt_list = []
    core_list = []
    decrypt_list = []
    err_list = []
    y_he_last = None
    sigma_last = None

    with HERuntime(cfg, rotation_steps=list(range(-512, 513))) as rt:
        sigma = detect_rotation_sigma(rt)
        sigma_last = sigma

        out_cols = B_shape[1]
        num_out_blocks = math.ceil(out_cols / d)

        for _ in range(repeats):
            # 左矩阵 pad 到 dxd，并只加密一次
            A_square = pad_to_square_block(A_small, d)

            t0 = now_us()
            ct_A = matrix_to_periodic_cipher(rt, A_square, "A_square")
            t1 = now_us()

            out_cts = []
            t2 = now_us()
            for j in range(num_out_blocks):
                c0 = j * d
                c1 = min((j + 1) * d, out_cols)

                B_block = np.zeros((d, d), dtype=np.float64)
                B_block[:, : (c1 - c0)] = B_full[:, c0:c1]

                ct_Cj = jkls_square_kernel_pcmm(rt, ct_A, B_block, d, sigma, f"Cblk_{j}")
                out_cts.append((ct_Cj, c1 - c0))
            t3 = now_us()

            y_parts = []
            t4 = now_us()
            for ct_Cj, valid_cols in out_cts:
                part = decode_periodic_square_output(rt, ct_Cj, d, A_shape[0], valid_cols)
                y_parts.append(part)
            t5 = now_us()

            y_he = np.concatenate(y_parts, axis=1)
            y_he_last = y_he

            encrypt_list.append(t1 - t0)
            core_list.append(t3 - t2)
            decrypt_list.append(t5 - t4)
            err_list.append(float(np.max(np.abs(y_plain_last - y_he))))

    result = {
        "case_name": case_name,
        "square_dim": d,
        "orig_input_shape": A_shape,
        "orig_weight_shape": B_shape,
        "plain_reference_us_median": median(plain_times),
        "input_ct_count_per_run": 1,
        "output_ct_count_per_run": math.ceil(B_shape[1] / d),
        "encrypt_us_median": median(encrypt_list),
        "core_us_median": median(core_list),
        "decrypt_us_median": median(decrypt_list),
        "plain_vs_he_max_abs_err": max(err_list) if err_list else None,
        "y_he_first_row": y_he_last[0].tolist(),
        "rotation_sigma": sigma_last,
        "op_meta": jkls_square_kernel_op_counts(
            d=d,
            input_ct_count=1,
            output_ct_count=math.ceil(B_shape[1] / d),
        ),
    }

    if result["plain_reference_us_median"] and result["core_us_median"]:
        result["core_vs_plain_ratio"] = result["core_us_median"] / result["plain_reference_us_median"]
    else:
        result["core_vs_plain_ratio"] = None

    return result


def main():
    print_sep("JKLS square-kernel / 案例1")
    case1 = run_case(
        case_name="jkls_square_64x64__64x64",
        A_shape=(64, 64),
        B_shape=(64, 64),
        d=64,
        repeats=1,
        seed=203201,
    )
    zh_print(case1)

    print_sep("JKLS square-kernel / 案例2")
    case2 = run_case(
        case_name="jkls_padded_8x64__64x64",
        A_shape=(8, 64),
        B_shape=(64, 64),
        d=64,
        repeats=1,
        seed=203202,
    )
    zh_print(case2)

    print_sep("JKLS square-kernel / 案例3")
    case3 = run_case(
        case_name="jkls_padded_split_8x64__64x128",
        A_shape=(8, 64),
        B_shape=(64, 128),
        d=64,
        repeats=1,
        seed=203203,
    )
    zh_print(case3)

    out_dir = Path("/workspace/FIDESlib-GPT/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wp3c_jkls_square_baseline_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "case_1_square": case1,
                "case_2_padded_rect_input": case2,
                "case_3_padded_split_rect_output": case3,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_sep("报告已保存")
    print(f"  JSON路径: {out_path}")


if __name__ == "__main__":
    main()
