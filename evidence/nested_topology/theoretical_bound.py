# -*- coding: utf-8 -*-
"""嵌套并联拓扑立项的理论上限证据（M0）。

回答一个不依赖采样器的问题：**同一个模型族，理论上最好能拟合到多好？**
做法是绕开 NUTS 与先验，用 scipy 多起点全局最小二乘分别拟两种拓扑：

  A 串联 Voigt + 串联 Z_ad   —— bmdrt 现有模型族（bm_core.model_z）
  B 嵌套并联 + 内层 R∥L      —— Shahri et al. 2024, JAC 1003, 175572 的电路

实测结论（真实 AZ31-4Li_100 谱）：A 最佳相对残差 30.5%，B 6.7%，差 4.6 倍。
差距与 K、先验、采样器无关（A 已给 3 个 HN 分量 + 全局搜索），因此是
**模型族结构错配**，不是调参问题。

原始数据按仓库约定不入库。脚本有两种模式：

  python evidence/nested_topology/theoretical_bound.py                 # 内置合成谱，自包含
  python evidence/nested_topology/theoretical_bound.py --data <path>  # 真实谱（三列 f Zre Zim）
  python evidence/nested_topology/theoretical_bound.py --data <path> --plot out.png

合成模式用 B 的参数生成一条"AZ31 型"谱再加噪，用来验证：
即使数据确实由 B 生成，A 也表达不了 —— 这是标准的模型错配检验。
"""
from __future__ import annotations

import argparse
import os
import numpy as np
from scipy.optimize import least_squares


# --------------------------------------------------------------------------
# 元件
# --------------------------------------------------------------------------

def z_ad(omega, Rad, tau):
    """R_ad ∥ L_ad：Z = R·jωτ/(1+jωτ)，恒等于 R − R/(1+jωτ)。"""
    x = omega * tau
    return Rad * (1j * x) / (1.0 + 1j * x)


def z_cpe(omega, Y, n):
    """纯 CPE：Z = 1/(Y·(jω)^n)。"""
    return 1.0 / (Y * (1j * omega) ** n)


# --------------------------------------------------------------------------
# 两种拓扑（omega 从外部绑定，便于 least_squares 只传参数）
# --------------------------------------------------------------------------

def make_models(omega):
    def model_a(p):
        """A: R_inf + Σ_{1..3} ZARC + Z_ad，全部串联（bmdrt 现有基底）。"""
        z = np.full(omega.shape, np.exp(p[0]), dtype=complex)
        for k in range(3):
            R, tau, phi = np.exp(p[1 + 3 * k]), np.exp(p[2 + 3 * k]), p[3 + 3 * k]
            z = z + R / (1.0 + (1j * omega * tau) ** phi)
        return z + z_ad(omega, np.exp(p[10]), np.exp(p[11]))

    def model_b(p):
        """B: Rs + CPE1 ∥ (R1 + CPE2 ∥ (R2 + (R_ad∥L_ad)))，文献嵌套并联。"""
        Rs, Y1, n1, R1, Y2, n2, R2, Rad, tau = (
            np.exp(p[0]), np.exp(p[1]), p[2], np.exp(p[3]),
            np.exp(p[4]), p[5], np.exp(p[6]), np.exp(p[7]), np.exp(p[8]))
        inner = R2 + z_ad(omega, Rad, tau)
        par2 = 1.0 / (Y2 * (1j * omega) ** n2 + 1.0 / inner)
        par1 = 1.0 / (Y1 * (1j * omega) ** n1 + 1.0 / (R1 + par2))
        return Rs + par1

    return model_a, model_b


# --------------------------------------------------------------------------
# 合成一条 AZ31 型谱（不依赖外部数据）
# --------------------------------------------------------------------------

def synthetic():
    f = np.logspace(-2, 5, 50)
    omega = 2 * np.pi * f
    _, mb = make_models(omega)
    # 真实谱上 B 的最小二乘最优参数（见立项书 §4）
    # Rs 取 49 Ω（真实谱高频截距），避免高频 |Z|≈0 被全局噪声淹没；
    # 噪声按逐点 |Z| 的 2% 加，保持全频段信噪比大致一致。
    p = np.array([np.log(49.0), np.log(1.1e-6), 0.64, np.log(55.7),
                  np.log(3.2e-5), 0.79, np.log(600.3), np.log(3085.3),
                  np.log(1.624)])          # n 是线性参数，其余是 log
    Z = mb(p)
    rng = np.random.default_rng(7)
    Z = Z + 0.02 * np.abs(Z) * (rng.normal(size=f.size)
                                + 1j * rng.normal(size=f.size))
    return f, Z, "内置合成 AZ31 型谱（B 生成 + 逐点 2% 噪声）"


def from_file(path):
    a = np.loadtxt(path)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError("需要三列数值：f, Zre, Zim")
    return a[:, 0], a[:, 1] + 1j * a[:, 2], os.path.abspath(path)


# --------------------------------------------------------------------------
# 全局最小二乘
# --------------------------------------------------------------------------

def best_fit(Z, wabs, omega, model, lo, hi, n_start=36, seed=3):
    def resid(p):
        d = (model(p) - Z) / wabs
        return np.concatenate([d.real, d.imag])

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_start):
        p0 = lo + (hi - lo) * rng.random(len(lo))
        try:
            r = least_squares(resid, p0, bounds=(lo, hi),
                              max_nfev=2500, ftol=1e-10, xtol=1e-10)
            cost = float(np.sqrt(np.mean(r.fun ** 2)))
            if best is None or cost < best[0]:
                best = (cost, r.x)
        except Exception:                    # noqa: BLE001 - 单个起点病态不影响全局
            continue
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="真实谱三列文件（f Zre Zim）")
    ap.add_argument("--plot", default=None, help="可选：输出 Nyquist 对比图")
    args = ap.parse_args()

    if args.data:
        f, Z, src = from_file(args.data)
    else:
        f, Z, src = synthetic()
    omega = 2 * np.pi * f
    wabs = np.abs(Z)
    ma, mb = make_models(omega)

    LO_R, HI_R = np.log(1e-2), np.log(1e5)
    LO_T, HI_T = np.log(1e-5), np.log(1e4)
    lo_a = np.array([LO_R] + [LO_R, LO_T, 0.30] * 3 + [LO_R, LO_T])
    hi_a = np.array([HI_R] + [HI_R, HI_T, 1.05] * 3 + [HI_R, HI_T])
    lo_b = np.array([LO_R, np.log(1e-9), 0.30, LO_R, np.log(1e-9), 0.30,
                     LO_R, LO_R, LO_T])
    hi_b = np.array([HI_R, np.log(1e-1), 1.05, HI_R, np.log(1e-1), 1.05,
                     HI_R, HI_R, HI_T])

    print("=" * 72)
    print("拓扑理论上限对照（scipy 多起点全局最小二乘，绕开 NUTS / 先验）")
    print("数据:", src)
    print("  n=%d, f=%.3g..%.3g Hz, |Z|max=%.1f" % (f.size, f.min(), f.max(), wabs.max()))
    curves = {}
    for tag, model, lo, hi in (("A", ma, lo_a, hi_a), ("B", mb, lo_b, hi_b)):
        best = best_fit(Z, wabs, omega, model, lo, hi)
        if best is None:
            print(tag, "全部起点失败")
            continue
        cost, p = best
        zf = model(p)
        rel = np.abs(zf - Z) / wabs
        curves[tag] = zf
        name = ("A 串联 Voigt + 串联 Z_ad（bmdrt 现有）" if tag == "A"
                else "B 嵌套并联 + 内层 R∥L（文献）")
        print("-" * 72)
        print("%s：最佳相对残差 RMS = %.4f (%.2f%%), max = %.4f"
              % (name, cost, cost * 100, rel.max()))
        for lf, hf, lab in ((0.01, 1, "低频 0.01–1 Hz"), (1, 100, "中频 1–100 Hz"),
                            (100, 1e5, "高频 100–1e5 Hz")):
            m = (f >= lf) & (f < hf)
            print("    %-20s n=%2d  rms=%.4f" % (lab, m.sum(), np.sqrt(np.mean(rel[m] ** 2))))
        if tag == "B":
            print("    B 参数: Rs=%.2f Y1=%.2g n1=%.2f R1=%.1f Y2=%.2g n2=%.2f "
                  "R2=%.1f Rad=%.1f tau=%.4g s (f_c=%.3g Hz)" %
                  (np.exp(p[0]), np.exp(p[1]), p[2], np.exp(p[3]), np.exp(p[4]), p[5],
                   np.exp(p[6]), np.exp(p[7]), np.exp(p[8]),
                   1 / (2 * np.pi * np.exp(p[8]))))
    if "A" in curves and "B" in curves:
        ra = np.sqrt(np.mean((np.abs(curves["A"] - Z) / wabs) ** 2))
        rb = np.sqrt(np.mean((np.abs(curves["B"] - Z) / wabs) ** 2))
        print("-" * 72)
        print("A / B = %.2f 倍差距。差距与 K、先验、采样器无关 → 模型族结构错配。" % (ra / rb))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
        for a_, (lf, hf, title) in zip(ax, ((0, 1e9, "Full Nyquist"), (0, 1, "Low-f zoom"))):
            m = (f >= lf) & (f < hf)
            a_.plot(Z.real, -Z.imag, "ko", ms=5, mfc="none", label="data")
            for tag, color in (("A", "#d94b4b"), ("B", "#2c6fbb")):
                a_.plot(curves[tag][m].real, -curves[tag][m].imag, "-", lw=2,
                        color=color, label="%s (rms %.1f%%)" %
                        (tag, 100 * np.sqrt(np.mean((np.abs(curves[tag] - Z) / wabs) ** 2))))
            a_.axhline(0, color="grey", lw=0.8)
            a_.set_xlabel("Re Z (Ohm.cm2)")
            a_.set_ylabel("-Im Z (Ohm.cm2)")
            a_.set_title(title)
            a_.grid(alpha=0.3)
            a_.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print("图:", os.path.abspath(args.plot))


if __name__ == "__main__":
    main()
