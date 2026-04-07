import time
import numpy as np

from hegpt import HEConfig, HERuntime
from hegpt.ops import he_decrypt_tensor, he_encrypt_tensor, he_linear_plain, plaintext_linear


def main():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    weight = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 0.5],
            [1.0, 1.0, 0.0],
            [0.5, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # 新增：非零 bias 验证
    bias = np.array([0.5, -1.0, 2.0], dtype=np.float64)

    t0 = time.perf_counter()
    y_ref = plaintext_linear(x, weight, bias)
    t1 = time.perf_counter()

    # 4 维输入，3 维输出
    # baseline 需要：
    #   rotate +1,+2,+3 做求和
    #   rotate -1,-2 做把 slot0 放到 slot1/2
    with HERuntime(HEConfig(devices=(0,)), rotation_steps=(1, 2, 3, -1, -2)) as rt:
        ct_x = he_encrypt_tensor(
            rt,
            x.tolist(),
            shape=(4,),
            layout="token_hidden_contiguous",
            name="x",
        )

        print("ct_x summary:", ct_x.summary())
        print("ct_x handle valid:", ct_x.ct.valid())

        t2 = time.perf_counter()
        ct_y = he_linear_plain(rt, ct_x, weight, bias=bias, name="y")
        t3 = time.perf_counter()

        y_he = np.array(he_decrypt_tensor(rt, ct_y, logical_length=3), dtype=np.float64)
        t4 = time.perf_counter()

    max_abs_err = float(np.max(np.abs(y_ref - y_he)))

    plaintext_linear_us = (t1 - t0) * 1e6
    he_linear_plain_us = (t3 - t2) * 1e6
    decrypt_us = (t4 - t3) * 1e6
    he_total_us = (t4 - t2) * 1e6

    print("x:", x)
    print("weight:\n", weight)
    print("bias:", bias)
    print("y_ref:", y_ref)
    print("y_he :", y_he)
    print("max_abs_err:", max_abs_err)
    print("plaintext_linear_us:", plaintext_linear_us)
    print("he_linear_plain_us:", he_linear_plain_us)
    print("decrypt_us:", decrypt_us)
    print("he_total_us:", he_total_us)


if __name__ == "__main__":
    main()
