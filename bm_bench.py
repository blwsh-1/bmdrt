# -*- coding: utf-8 -*-
"""合成数据 benchmark：一组难度递增的案例，量化"拟合度到底怎么样"。

    python bm_bench.py

为什么用确定性多起点最小二乘（MAP）而不是 MCMC：
  "拟合度"问的是"这组模型参数能不能把数据贴住"，这是似然面的事，
  和链混合得好不好是两件事。MAP 几秒就能算，可以一次扫 8 个案例；
  后验/区间的事由 bm_check.py 与 bm_experiments.py 负责（那要十几分钟一组）。

每个案例报 5 个量：
  rms      拟合残差 rms（Ω）
  rms/σ√2  与噪声水平的比值，**理想值 1.0**（>2 说明模型贴不住数据）
  rel      相对残差（占 |Z| 的百分比）
  acf_max  残差自相关（滞后 1–5）的最大绝对值，阈值 1.96/√n≈0.35
           **rms 小但 acf 大 = 残差有结构 = 模型缺项**，这是最有信息的一条
  参数     面积/峰位相对真值的误差（只统计权重 >1% 的分量）

案例设计（A→H 逐步加难度）：
  A 两个 ZARC，峰间距 18 倍，绝对噪声 σ=0.2      ← 基线，即 bm_check 那组
  B 峰间距 3 倍                                  ← 分离度下降
  C 峰间距 1.5 倍                                ← 几乎重合，看会不会并成一个
  D 三个 ZARC                                    ← K=3
  E ZARC + Warburg（模型里没有 Warburg）          ← 缺项，看 rms/acf 能不能报警
  F 两个 ZARC + 电感（拟合时 include_L=False）    ← 缺项
  G 相对噪声（0.5%·|Z|）而不是绝对噪声            ← 高频端信噪比骤降
  H 窄窗口 1e-1..1e4 Hz（少一个数量级低频）        ← 低频信息不足
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import least_squares

import bm_core as B
from bm_experiments import acf_max   # 保留 BCH.acf_max 兼容

warnings.filterwarnings("ignore")

F_FULL = np.logspace(-2, 5, 31)      # 默认窗口 0.01–1e5 Hz，31 点
F_NARROW = np.logspace(-1, 4, 31)    # 案例 H


# --------------------------------------------------------------------------
# 数据生成
# --------------------------------------------------------------------------

def _zarc(f, R, f0, phi):
    return B.zarc(f, R, f0, phi)


def _zwarburg(f, Aw):
    """半无限 Warburg：Z = Aw / sqrt(jω)，ω = 2πf。"""
    return Aw / np.sqrt(1j * 2 * np.pi * f)


def _zl(f, L):
    return 1j * 2 * np.pi * f * L


TAG = {
    "A": "A  2xZARC sep 18x",
    "B": "B  2xZARC sep 3x",
    "C": "C  2xZARC sep 1.5x",
    "D": "D  3xZARC",
    "E": "E  ZARC+Warburg (missing term)",
    "F": "F  2xZARC+L (missing L)",
    "G": "G  2xZARC rel-noise 0.5%",
    "H": "H  2xZARC narrow window",
}


def make_cases():
    """返回 [(名字, f, Zobs, Ztrue, 真值列表[(R,f0,phi)], sigma_vec)]。"""
    cases = []

    def add(name, f, Ztrue, comps, sigma_vec, seed=0, extra=None):
        rng = np.random.default_rng(seed)
        n = f.size
        if np.isscalar(sigma_vec):
            sr = si = float(sigma_vec)
            eps = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            Zobs = Ztrue + sr * eps.real + 1j * si * eps.imag
        else:
            eps = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            Zobs = Ztrue + sigma_vec * eps
        cases.append(dict(name=name, tag=TAG[name[0]], f=f, Z=Zobs, Z0=Ztrue,
                          comps=comps, sigma=sigma_vec, extra=extra))

    def zsum(f, Rinf, comps):
        z = np.full(f.size, Rinf, complex)
        for R, f0, phi in comps:
            z = z + _zarc(f, R, f0, phi)
        return z

    # A 基线：峰间距 18 倍
    add("A 2×ZARC 间距18×", F_FULL, zsum(F_FULL, 10.0, [(33.0, 357.0, 0.85), (15.8, 20.0, 0.90)]),
        [(33.0, 357.0, 0.85), (15.8, 20.0, 0.90)], 0.2)

    # B 间距 3 倍
    add("B 2×ZARC 间距3×", F_FULL, zsum(F_FULL, 10.0, [(30.0, 600.0, 0.85), (18.0, 200.0, 0.90)]),
        [(30.0, 600.0, 0.85), (18.0, 200.0, 0.90)], 0.2)

    # C 间距 1.5 倍
    add("C 2×ZARC 间距1.5×", F_FULL, zsum(F_FULL, 10.0, [(25.0, 150.0, 0.85), (20.0, 100.0, 0.90)]),
        [(25.0, 150.0, 0.85), (20.0, 100.0, 0.90)], 0.2)

    # D 三个
    add("D 3×ZARC", F_FULL, zsum(F_FULL, 10.0, [(25.0, 1000.0, 0.85), (15.0, 100.0, 0.90), (8.0, 10.0, 0.80)]),
        [(25.0, 1000.0, 0.85), (15.0, 100.0, 0.90), (8.0, 10.0, 0.80)], 0.2)

    # E Warburg（模型缺项）
    zE = zsum(F_FULL, 10.0, [(30.0, 300.0, 0.85)]) + _zwarburg(F_FULL, 2.0)
    add("E ZARC+Warburg", F_FULL, zE, [(30.0, 300.0, 0.85)], 0.2, extra="Warburg Aw=2")

    # F 电感（拟合时不含 L）
    zF = zsum(F_FULL, 10.0, [(33.0, 357.0, 0.85), (15.8, 20.0, 0.90)]) + _zl(F_FULL, 1e-5)
    add("F 2×ZARC+L", F_FULL, zF, [(33.0, 357.0, 0.85), (15.8, 20.0, 0.90)], 0.2,
        extra="L=1e-5 H")

    # G 相对噪声 0.5%
    zG = zsum(F_FULL, 10.0, [(33.0, 357.0, 0.85), (15.8, 20.0, 0.90)])
    add("G 相对噪声0.5%", F_FULL, zG, [(33.0, 357.0, 0.85), (15.8, 20.0, 0.90)],
        0.005 * np.abs(zG))

    # H 窄窗口
    zH = zsum(F_NARROW, 10.0, [(33.0, 357.0, 0.85), (15.8, 20.0, 0.90)])
    add("H 窄窗口1e-1..1e4", F_NARROW, zH, [(33.0, 357.0, 0.85), (15.8, 20.0, 0.90)], 0.2)

    return cases


# --------------------------------------------------------------------------
# 多起点有界最小二乘（K 个分量，通用）
# --------------------------------------------------------------------------

def map_fit(f, Z, K, zfix=None, n_start=16, seed=0, pad_dec=1.5):
    """返回 dict(rms, w, f0, phi, zeta, R_inf)。

    参数化：x = [R_inf, lnw_1..K, lntau_1..K, phi_1..K, (zeta_1..K)]
    边界**与 make_model 的支撑对齐**（否则 MAP 会跑到模型根本不允许的区域，
    比如 τ→1e-9 让 f0 冲到 1e8 Hz，得到一堆无意义的"最优解"）：
      τ ∈ [1/(2πfmax)/10^pad, 1/(2πfmin)·10^pad]
      φ ∈ [0.5, 1.0]，ζ ∈ [0.3, 2.0]
    """
    nz = 0 if zfix is not None else K
    n = 1 + K + K + K + nz

    def unpack(x):
        w = np.exp(x[1:1 + K])
        tau = np.exp(x[1 + K:1 + 2 * K])
        phi = x[1 + 2 * K:1 + 3 * K]
        zeta = np.full(K, zfix) if zfix is not None else x[1 + 3 * K:]
        return dict(R_inf=x[0], W=w.sum(), pi=w / w.sum(),
                    tau=tau, phi=phi, zeta=zeta)

    def resid(x):
        r = Z - B.model_z(f, unpack(x), include_L=False)
        return np.concatenate([r.real, r.imag])

    tau_lo = 1.0 / (2 * np.pi * f.max()) / 10 ** pad_dec
    tau_hi = 1.0 / (2 * np.pi * f.min()) * 10 ** pad_dec
    f0s = np.logspace(np.log10(f.min() * 3), np.log10(f.max() / 3), K)
    x0 = np.concatenate([[max(np.real(Z).min(), 1e-2)],
                         np.full(K, np.log(np.abs(Z).max() / K)),
                         np.log(1 / (2 * np.pi * f0s)),
                         np.full(K, 0.85)] + ([np.ones(K)] if nz else []))

    lo = np.concatenate([[1e-3], np.full(K, -12), np.full(K, np.log(tau_lo)),
                         np.full(K, 0.50)] + ([np.full(K, 0.30)] if nz else []))
    hi = np.concatenate([[1e4], np.full(K, 12), np.full(K, np.log(tau_hi)),
                         np.full(K, 1.0)] + ([np.full(K, 2.0)] if nz else []))

    rng = np.random.default_rng(seed)
    best = None
    for t in range(n_start):
        x_start = x0.copy()
        if t > 0:
            x_start = x_start + np.concatenate(
                [[0], rng.normal(0, 1.0, K), rng.normal(0, 1.0, K),
                 rng.normal(0, 0.10, K)] + ([rng.normal(0, 0.3, K)] if nz else []))
            x_start = np.clip(x_start, lo + 1e-9, hi - 1e-9)
        try:
            r = least_squares(resid, x_start, bounds=(lo, hi), method="trf",
                              max_nfev=6000, xtol=1e-13, ftol=1e-13)
        except Exception:
            continue
        v = float(np.sqrt(np.mean(resid(r.x) ** 2)))
        if best is None or v < best[0]:
            best = (v, r.x)
    if best is None:
        return None
    v, x = best
    p = unpack(x)
    f0 = 1.0 / (2 * np.pi * p["tau"])
    w = p["W"] * p["pi"]
    o = np.argsort(-f0)
    return dict(rms=v, f0=f0[o], w=w[o], phi=p["phi"][o], zeta=p["zeta"][o],
                R_inf=float(x[0]), lo=lo, hi=hi)


# --------------------------------------------------------------------------
# 诊断量
# --------------------------------------------------------------------------

def evaluate(c, fit):
    """把一个 fit 结果与真值比，返回一行指标。

    关键参照不是 σ√2，而是 **rms_true = rms(Z_obs − Z_true)**：
    这就是"这组数据里噪声本身有多大"，也就是任何模型能达到的地板。
    ratio = rms_fit / rms_true，1.0 表示"已经贴到噪声底"，
    >2 表示"模型再怎么调也贴不住"。
    这样归一化对相对噪声（案例 G）和缺项模型（E/F）都成立。
    """
    f, Z, Z0 = c["f"], c["Z"], c["Z0"]
    p = dict(R_inf=fit["R_inf"], W=float(np.sum(fit["w"])),
             pi=fit["w"] / np.sum(fit["w"]), tau=1 / (2 * np.pi * fit["f0"]),
             phi=fit["phi"], zeta=fit["zeta"])
    Zm = B.model_z(f, p, include_L=False)
    r = Z - Zm
    rms = float(np.sqrt(np.mean(np.abs(r) ** 2)))
    rms_true = float(np.sqrt(np.mean(np.abs(Z - Z0) ** 2)))
    rel = rms / float(np.sqrt(np.mean(np.abs(Z) ** 2)))
    am = max(acf_max(r.real), acf_max(r.imag))

    # 只统计"权重 >2% 且峰位落在**测量频段内**"的分量。
    # 判据必须是严格的 [fmin, fmax]：峰跑到测量频段外就没有数据支撑它，
    # 不管权重多大都不可信。实测案例 C 会把一个分量放在 175 kHz（窗口上限 100 kHz）
    # 而权重只有 1.05%——它只是在拟合高频端的噪声，不是第二个峰。
    w_tot = fit["w"].sum()
    big = fit["w"] > 0.02 * w_tot
    inwin = (fit["f0"] >= f.min()) & (fit["f0"] <= f.max())
    keep = big & inwin
    degen = int(big.sum() - keep.sum())
    fe, we = fit["f0"][keep], fit["w"][keep]
    tr = sorted(c["comps"], key=lambda t: -t[1])
    ft = np.array([t[1] for t in tr])
    wt = np.array([t[0] for t in tr])
    ef = ew = np.nan
    if len(fe) == len(ft):
        ef = float(np.max(np.abs(fe - ft) / ft))
        ew = float(np.max(np.abs(we - wt) / wt))
    return dict(rms=rms, ratio=rms / rms_true, rel=rel, acf=am,
                ef=ef, ew=ew, ncomp=int(keep.sum()), degen=degen,
                K=len(fit["f0"]), rms_true=rms_true)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def _row(name, K, tag, m, ntrue):
    ef = "—" if np.isnan(m["ef"]) else f"{m['ef']*100:.1f}%"
    ew = "—" if np.isnan(m["ew"]) else f"{m['ew']*100:.1f}%"
    dg = "!" if m["degen"] else ""
    return (f"{name:<20}{K:>2}{tag:>6}{m['rms']:>9.3f}{m['ratio']:>8.2f}"
            f"{m['rel']*100:>7.2f}{m['acf']:>7.2f}"
            f"{str(m['ncomp'])+'/'+str(ntrue)+dg:>8}{ef:>9}{ew:>9}")


def main():
    cases = make_cases()
    hdr = (f"{'案例':<20}{'K':>2}{'ζ':>6}{'rms(Ω)':>9}{'rms/地板':>8}"
           f"{'rel%':>7}{'acf':>7}{'分量':>8}{'f0误差':>9}{'面积误差':>9}")
    print("合成 benchmark：拟合度（确定性多起点 MAP，边界与模型支撑对齐）")
    print("rms/地板 = rms(拟合) / rms(真值曲线 vs 观测)，**1.0 = 已经贴到噪声底**")
    print("acf = 残差自相关(滞后1-5)最大值，阈值 1.96/√n≈0.35；分量列带 ! = 有分量跑到窗口外\n")
    print(hdr)
    print("-" * 76)

    rows = []
    for c in cases:
        K = len(c["comps"])
        for zfix, tag in ((1.0, "固定1"), (None, "自由")):
            fit = map_fit(c["f"], c["Z"], K=K, zfix=zfix)
            if fit is None:
                print(f"{c['name']:<20}{K:>2}{tag:>6}   拟合失败")
                continue
            m = evaluate(c, fit)
            print(_row(c["name"], K, tag, m, K))
            rows.append((c["name"], K, tag, m))
        print()

    # ---------------- 多给一个分量 ----------------
    print("\n多给一个分量（K = 真值+1），看多余分量会不会自己消失")
    print(hdr)
    print("-" * 76)
    for c in cases:
        if len(c["comps"]) >= 3:
            continue
        for zfix, tag in ((1.0, "固定1"), (None, "自由")):
            fit = map_fit(c["f"], c["Z"], K=len(c["comps"]) + 1, zfix=zfix)
            if fit is None:
                continue
            m = evaluate(c, fit)
            print(_row(c["name"], len(c["comps"]) + 1, tag, m, len(c["comps"])))
            rows.append((c["name"], len(c["comps"]) + 1, tag, m))
        print()

    # ---------------- 判读 ----------------
    print("=" * 76)
    print("判读")
    print("=" * 76)
    good_fit = [r for r in rows if r[3]["ratio"] < 1.30]
    bad_fit = [r for r in rows if r[3]["ratio"] >= 2.0]
    mid_fit = [r for r in rows if 1.30 <= r[3]["ratio"] < 2.0]
    bad_acf = [r for r in rows if r[3]["ratio"] < 2.0 and r[3]["acf"] >= 0.35]
    good_param = [r for r in rows if not np.isnan(r[3]["ew"]) and r[3]["ew"] < 0.05]
    print(f"  贴到噪声底（rms/地板<1.3）：{len(good_fit)}/{len(rows)} 组")
    if mid_fit:
        print(f"  略差（1.3–2.0）：{len(mid_fit)} 组 → " +
              ", ".join(f"{r[0]} K={r[1]} ζ{r[2]}" for r in mid_fit))
    if bad_fit:
        print(f"  贴不住（≥2.0）：{len(bad_fit)} 组 → " +
              ", ".join(f"{r[0]} K={r[1]} ζ{r[2]}({r[3]['ratio']:.1f})" for r in bad_fit))
    if bad_acf:
        print(f"  **rms 够好但残差有结构**（acf≥0.35）：{len(bad_acf)} 组")
        for nm, K, tag, m in bad_acf:
            print(f"    · {nm} K={K} ζ{tag}: acf={m['acf']:.2f}, rms/地板={m['ratio']:.2f}")
    print(f"  面积误差 <5%：{len(good_param)}/{len(rows)} 组")
    print("\n  注意：rms 到噪声底 ≠ 参数对。看 f0/面积误差列与'分量'列。")


if __name__ == "__main__":
    main()
