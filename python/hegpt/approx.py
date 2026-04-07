import math
from typing import Dict

import numpy as np


# 参考 GPT-2/你提供的 model.py 中的 GELU 近似公式
def gelu_reference(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


# 这是一个可用的 starter 版 6 次多项式系数。
# 后续你可以用 scripts/fit_gelu.py 重新拟合并替换这里。
#
# 多项式形式：
# p(x) = a6*x^6 + a5*x^5 + a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0
GELU_POLY6_COEFFS = [
    0.00181160442,
    0.0,
   -0.0377386852,
    0.0,
    0.360453272,
    0.5,
    0.00848495893,
]


def poly_eval(x, coeffs):
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x, dtype=np.float64)
    for c in coeffs:
        y = y * x + c
    return y


def gelu_poly6_plain(x):
    return poly_eval(x, GELU_POLY6_COEFFS)


def evaluate_gelu_fit(
    interval=(-3.0, 3.0),
    num_points=10001,
) -> Dict[str, float]:
    lo, hi = interval
    xs = np.linspace(lo, hi, num_points, dtype=np.float64)

    y_ref = gelu_reference(xs)
    y_approx = gelu_poly6_plain(xs)

    abs_err = np.abs(y_ref - y_approx)

    return {
        "interval_lo": float(lo),
        "interval_hi": float(hi),
        "num_points": int(num_points),
        "max_abs_err": float(np.max(abs_err)),
        "mean_abs_err": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean((y_ref - y_approx) ** 2))),
    }


def print_gelu_fit_report():
    report = evaluate_gelu_fit()
    print("GELU poly6 fit report")
    for k, v in report.items():
        print(f"{k}: {v}")
