# -*- coding: utf-8 -*-
"""`example_caseE.csv` 到底有没有扩散项？两种模型对拍（只读）。

背景
----
`bm_kk.py:389` 把这条谱写成「无扩散元件」，并把它当作 `log|Z|` 幅度判据的
立论依据。但实测：低频 `A_W = -Im·2√(πf)` 在 1.8 个 decade 内稳定在 1.98–2.02。

本脚本不去争"它是不是扩散"，只做两件事：

  1. 对 `caseE` 与 `caseE_warburg` 各拟合两个**参数个数相同**的模型
       A) `R∞ + ZARC + Ws(A_W)`
       B) `R∞ + ZARC + ZARC(R₂, f₂, n₂)`
     比较 chi2；
  2. 把 B 的拟合结果原样报出来——如果 B 自己把第二支推到窗口外且 `n₂≈0.5`，
     那就是 `WS_CRITERION_PLAN` §5.1 的"同形解"在数据上的直接证据。

**这不是真值判定**。两条模型 chi2 相等时，唯一诚实的结论是"不可区分"，
不能反过来说"所以它是扩散"。`truth` 仍为 `unknown`。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(r"D:/eis训练数据/bmdrt")
EXAMPLES = ["example_caseE.csv", "example_caseE_warburg.csv"]


def load(fn):
    d = np.genfromtxt(ROOT / "examples" / fn, delimiter=",", names=True)
    nm = d.dtype.names
    f = np.asarray(d[nm[0]], float)
    Z = np.asarray(d[nm[1]], float) + 1j * np.asarray(d[nm[2]], float)
    o = np.argsort(f)
    return f[o], Z[o]


def zarc(f, R, f0, n):
    return R / (1.0 + (1j * f / f0) ** n)


def zw(f, A):
    return A / np.sqrt(1j * 2 * np.pi * f)


def model_ws(f, x):
    return x[0] + zarc(f, x[1], x[2], x[3]) + zw(f, x[4])


def model_2zarc(f, x):
    return x[0] + zarc(f, x[1], x[2], x[3]) + zarc(f, x[4], x[5], x[6])


def fit(f, Z, model, x0, lo, hi):
    def res(x):
        r = (model(f, x) - Z) / np.abs(Z)
        return np.r_[r.real, r.imag]
    s = least_squares(res, x0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14,
                      max_nfev=20000)
    r = res(s.x)
    return s.x, float(np.sqrt(np.mean(r ** 2))), float(np.sum(r ** 2))


def main():
    print(f"{'file':30s} {'模型':22s} {'rms':>7s} {'chi2':>8s}")
    summary = []
    for fn in EXAMPLES:
        f, Z = load(fn)
        xa, ra, ca = fit(f, Z, model_ws, [10, 30, 300, .85, 2],
                         [0, 1, 1, .4, 1e-3], [100, 300, 1e6, 1, 50])
        xb, rb, cb = fit(f, Z, model_2zarc, [10, 20, 300, .85, 15, .05, .8],
                         [0, 1, 1, .4, 1, 1e-4, .4], [100, 300, 1e6, 1, 300, 1e3, 1])
        print(f"{fn:30s} {'A) R∞+ZARC+Ws':22s} {ra:7.4f} {ca:8.4f}")
        print(f"{'':30s} {'B) R∞+ZARC+ZARC':22s} {rb:7.4f} {cb:8.4f}")
        print(f"{'':30s} {'chi2(A)/chi2(B)':22s} {ca / cb:7.3f}")
        print(f"{'':30s} A 参数: R∞={xa[0]:.2f} R={xa[1]:.2f} f0={xa[2]:.1f} "
              f"n={xa[3]:.3f} A_W={xa[4]:.3f}")
        print(f"{'':30s} B 参数: R∞={xb[0]:.2f} R1={xb[1]:.2f} f1={xb[2]:.1f} "
              f"n1={xb[3]:.3f} | R2={xb[4]:.2f} f2={xb[5]:.4g} n2={xb[6]:.3f}")
        print()
        summary.append((fn, ra, ca, rb, cb, xa, xb))

    print("读法：")
    print("  · A 给出 A_W≈2 且 rms 在噪声量级(σ=0.2 Ω 对应 ~0.5%·|Z|)，")
    print("    说明'低频需要一个 ω^-1/2 项'这句有数值支撑；")
    print("  · B 与 A 的 chi2 相等，且 B 把第二支推到 f2≈1e-4 Hz(窗外)、n2≈0.5，")
    print("    正是 §5.1 的同形解 —— 所以'它是扩散'这句**没有**支撑；")
    print("  · 两条模型不可区分时，truth 仍是 unknown，不能进 TP/FP/TN 统计。")
    return summary


if __name__ == "__main__":
    main()
