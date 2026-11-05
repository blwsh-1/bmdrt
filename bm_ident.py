# -*- coding: utf-8 -*-
"""峰位偏差的归因：是代码错，还是估计量的性质？

`bm_check.py` 的 B2 项在 σ=0.2 的合成数据上给出 f0 = 340/23.4 Hz，
真值 357/20 Hz——峰位误差 17%，而面积误差只有 1.4%。这个脚本把原因查清：

  C1 无噪数据：从 5 个不同起点做确定性最小二乘，看是否都回到真值
     → 检验**正演模型**与**可辨识性**（有简并脊的话会停在别处）
  C2 有噪数据：同样做多起点最小二乘，取 rms 最小的解
     → 这是似然的全局最优（MAP 的位置），与后验中位数比较
  C3 有噪数据 + ζ 钉死为 1
     → 检验"ζ 自由度是否把峰位带偏"

注意：C1/C2/C3 用的是**确定性最小二乘**，不是采样。所以它回答的是
"似然长什么样"，不回答"后验混合得好不好"（那是 bm_experiments 的事）。

    python bm_ident.py
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import least_squares

import bm_core as B

warnings.filterwarnings("ignore")

TRUTH = dict(f0=np.array([357.0, 20.0]), w=np.array([33.0, 15.8]),
             phi=np.array([0.85, 0.90]), R_inf=10.0)


def _unpack(x, zfix):
    """x → model_z 的参数字典。

    zfix 为 None 时 x 有 9 个元素，否则 7 个（省掉两个 ζ）。
    两个分量的面积用 exp 参数化并归一，保证 Σw = W 且 w>0。
    """
    if zfix is None:
        e1, e2 = np.exp(x[1]), np.exp(x[5])
        tau = np.exp(np.array([x[2], x[6]]))
        phi = np.array([x[3], x[7]])
        zeta = np.array([x[4], x[8]])
    else:
        e1, e2 = np.exp(x[1]), np.exp(x[4])
        tau = np.exp(np.array([x[2], x[5]]))
        phi = np.array([x[3], x[6]])
        zeta = np.array([zfix, zfix])
    return dict(R_inf=x[0], W=e1 + e2, pi=np.array([e1, e2]) / (e1 + e2),
                tau=tau, phi=phi, zeta=zeta)


def _x0(zfix):
    if zfix is None:
        return np.array([10.0, np.log(33), np.log(1 / (2 * np.pi * 357)), 0.85, 1.0,
                         np.log(15.8), np.log(1 / (2 * np.pi * 20)), 0.90, 1.0])
    return np.array([10.0, np.log(33), np.log(1 / (2 * np.pi * 357)), 0.85,
                     np.log(15.8), np.log(1 / (2 * np.pi * 20)), 0.90])


def map_fit(f, Z, zfix=None, n_start: int = 15, seed: int = 0):
    """多起点最小二乘，返回 rms 最小的那组参数（近似 MAP）。"""
    x0 = _x0(zfix)
    n = x0.size
    rng = np.random.default_rng(seed)

    def resid(x):
        r = Z - B.model_z(f, _unpack(x, zfix), include_L=False)
        return np.concatenate([r.real, r.imag])

    # 起点扰动：面积(ln) ±0.5、τ(ln) ±0.6、φ ±0.15、ζ ±0.3，都只动一个
    jit = [(1, 0.5), (2, 0.6), (3, 0.15)] + ([] if zfix is not None else [(4, 0.3)]) \
        + [(4 if zfix is not None else 5, 0.5), (5 if zfix is not None else 6, 0.6),
           (6 if zfix is not None else 7, 0.15)] \
        + ([] if zfix is not None else [(8, 0.3)])
    best = None
    for t in range(n_start):
        dx = np.zeros(n) if t == 0 else np.zeros(n)
        if t > 0:
            for j, sd in jit:
                if j < n:
                    dx[j] = rng.normal(0, sd)
        r = least_squares(resid, x0 + dx, method="lm", max_nfev=40000,
                          xtol=1e-15, ftol=1e-15)
        v = float(np.sqrt(np.mean(resid(r.x) ** 2)))
        if best is None or v < best[0]:
            best = (v, r.x)
    v, x = best
    p = _unpack(x, zfix)
    f0 = 1.0 / (2 * np.pi * p["tau"])
    w = p["W"] * p["pi"]
    o = np.argsort(-f0)
    return {"rms": v, "f0": f0[o], "w": w[o], "phi": p["phi"][o],
            "zeta": p["zeta"][o], "R_inf": float(x[0])}


def _show(tag, r):
    ef = np.abs(r["f0"] - TRUTH["f0"]) / TRUTH["f0"]
    ew = np.abs(r["w"] - TRUTH["w"]) / TRUTH["w"]
    print(f"{tag:<22}{r['rms']:>10.2e}  "
          f"{np.round(r['f0'], 1)!s:<16}{np.round(r['w'], 2)!s:<16}"
          f"{np.round(r['phi'], 3)!s:<16}{np.round(r['zeta'], 2)!s:<14}"
          f"{r['R_inf']:>6.2f}   "
          f"f0误差 {ef.max()*100:5.1f}%  w误差 {ew.max()*100:4.1f}%")


def main():
    f, Z0 = B.true_spectrum()
    _, Zn = B.noisy()

    print("真值            f0=[357. 20.]  w=[33. 15.8]  phi=[0.85 0.9 ]  zeta=[1. 1.]  Rinf=10")
    print(f"\n{'情形':<22}{'rms(Ω)':>10}  {'f0':<16}{'w':<16}{'phi':<16}{'zeta':<14}{'Rinf':>6}")

    r1 = map_fit(f, Z0, zfix=None)
    _show("C1 无噪, ζ free", r1)

    r2 = map_fit(f, Zn, zfix=None)
    _show("C2 有噪, ζ free", r2)

    r3 = map_fit(f, Zn, zfix=1.0)
    _show("C3 有噪, ζ=1 固定", r3)

    r4 = map_fit(f, Z0, zfix=1.0)
    _show("C4 无噪, ζ=1 固定", r4)

    print("\n读法：")
    print("  C1 若回到真值 → 正演模型与可辨识性没问题，不是代码错。")
    print("  C2 若峰位仍偏 → 是**这个噪声实现下似然的真实最优**就在那里，")
    print("     不是采样没收敛，也不是公式错。")
    print("  C3 若峰位回到真值 → 偏差来自 ζ 自由度：ζ≠1 时 DRT 峰不在 τ0 上，")
    print("     (τ0, ζ) 之间存在弱约束方向，峰位被拉走而阻抗几乎不变。")


if __name__ == "__main__":
    main()
