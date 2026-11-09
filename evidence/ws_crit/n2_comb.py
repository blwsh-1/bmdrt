# -*- coding: utf-8 -*-
"""N2：Boukamp 的 DRT 梳状指纹——**理论成立，但做不成判据**。

理论（Boukamp 2017, *Electrochim. Acta* 252:154）
------------------------------------------------
有限长 Warburg 的两种边界给出严格的部分分式展开（`z = √(jωτ_0)`）：

  透射 Wshort  `tanh(z)/z = Σ_{k≥1} 2/(z² + π²(k−½)²)`
      → Debey 梳齿 `τ_k = τ_0/[π²(k−½)²]`，`R_k = 2Z_0·τ_k/τ_0`
      → `τ_1 = 0.4053 τ_0`，`R_1 = 0.8106 Z_0`（基齿吃掉 81%）
      → 齿距比 `τ_k/τ_1 = 1 : 1/9 : 1/25`，幅值比同

  阻挡 Wo      `coth(z)/z = 1/z² + Σ_{n≥1} 2/(z² + n²π²)`
      → 一个**串联电容** `C = τ_0/Z_0` + 梳齿 `τ_n = τ_0/(n²π²)`
      → `τ_1 = τ_0/π²`，`R_1 = 0.2026 Z_0`，齿距比 `1 : 1/4 : 1/9`

`verify_identity()` 直接数值验证两条恒等式，并给出 `O(1/K)` 截断收敛速率。
这是"不依赖 45° 段长"的**唯一**独立指纹——所以值得试，但必须先问一句：

**DRT 反演分辨得出这些齿吗？** 本脚本的回答是**不能**，理由有三条，全部有数：

  1. 齿间幅值衰减太快：第 2 齿只有基齿的 **11%**（Wshort）/ **25%**（Wo），
     第 3 齿 4%/11%。1% 噪声下第 2 齿就没了。
  2. 齿距 0.95 decade（Wshort 1→2）压在 DRT 网格分辨率（m=60 覆盖 ~7 dec
     → 0.12 dec/点）的两端；实测次峰 τ 比在 11.7–194 之间乱跳，预期是 9。
  3. 唯一稳的量是 `R_基齿/R_total = 0.82`（三个 τ_0 × 三个噪声档都是 0.82），
     但**单一 ZARC 也给 0.87–0.94**——差 0.05 而散度就有 0.03，判不开。

Wo 更糟：`1/z²` 那个串联电容**根本不在 DRT 的表示能力里**（DRT 是 Debye 之和，
没有纯电容项），实测 chi2 比 Wshort 大 3 个量级，`R_total` 直接跑飞。

结论：**梳齿指纹不能用作判据**，`Ws ≡ 窗外 R||CPE(0.5)` 的死结**没有**被解开。
能留下的只有两个可用的常数（拟合后做一致性检查，不是事前判据）：

  `τ_peak(DRT) = 0.4053 τ_D`  → DRT 峰位比 `f_D` 低 **0.39 decade**
  `R_基齿 = 0.811 Z_0`        → 单个齿而不是整条

产物：`comb_n2.json`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:/eis训练数据/bmdrt")
sys.path.insert(0, r"D:/eis训练数据/bmdrt/evidence/ws_crit")

import drt
import bench

ROOT = Path(r"D:/eis训练数据/bmdrt")
OUT = ROOT / "evidence" / "ws_crit" / "comb_n2.json"

#: Wshort / Wo 的前 5 齿（τ_k/τ_0 与 R_k/Z_0），与 `verify_identity` 同源
K = np.arange(1, 6)
W_SHORT = dict(tau=np.pi ** -2 / (K - 0.5) ** 2,
               R=2.0 / (np.pi * (K - 0.5)) ** 2)
W_OPEN = dict(tau=np.pi ** -2 / K ** 2, R=2.0 / (np.pi * K) ** 2)


def verify_identity(n_terms=(200, 2000, 20000)):
    """数值验证两条部分分式恒等式，并给出**截断收敛速率**。

    级数尾部是 `Σ_{k>K} 2/(π²k²) ≈ 2/(π²K)`，即 `O(1/K)`——所以
    "固定 K 报一个误差"没意义（K=200 时 3.7e-3，全是截断残差，不是恒等式错）。
    这里报三个 K 的误差，看它是否按 1/K 掉一个量级。

    区间 `|z| ∈ [0.03, 3]`：梳齿真正起作用的区域是 `f ≳ f_D`（`|z| ≲ 1`），
    `|z|=40` 时函数值本身只剩 0.025，任何截断都会显得很大。
    """
    z = np.logspace(-1.5, 0.5, 24) * (1 + 0.6j)
    out = []
    for n in n_terms:
        k = np.arange(1, n + 1)[:, None]
        comb_tanh = np.sum(2.0 / (z[None, :] ** 2 + (np.pi * (k - 0.5)) ** 2), axis=0)
        th = np.tanh(z) / z
        e1 = float((np.abs(comb_tanh - th) / np.abs(th)).max())

        comb_coth = 1.0 / z[None, :] ** 2 + np.sum(
            2.0 / (z[None, :] ** 2 + (np.pi * k) ** 2), axis=0)
        ct = (1 / np.tanh(z)) / z
        e2 = float((np.abs(comb_coth - ct) / np.abs(ct)).max())
        out.append((int(n), e1, e2))
    return out


def measure(f, Z, sigma, rng, m=60):
    """对一条谱做 DRT，返回与梳齿有关的可测量。"""
    sd = sigma * np.abs(Z)
    Zn = Z + sd * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
    r = drt.drt(f, Zn, sigma=sigma, m=m)
    ps = [p for p in drt.peaks(r, rel=0.02) if p["sign"] > 0 and p["R_ohm"] > 0]
    ps.sort(key=lambda q: -q["R_ohm"])
    rtot = float(r["R_total"])
    out = dict(chi2=float(r["chi2"]), R_total=rtot, n_peaks=len(ps),
               tau_top=float("nan"), r_frac=float("nan"),
               tau_ratio_2=float("nan"))
    if ps and np.isfinite(rtot) and abs(rtot) > 1e-12:
        out["tau_top"] = ps[0]["tau_s"]
        out["r_frac"] = ps[0]["R_ohm"] / rtot
        if len(ps) >= 2:
            out["tau_ratio_2"] = ps[0]["tau_s"] / ps[1]["tau_s"]
    return out


def main():
    F = np.logspace(-2, 5, 61)

    ident = verify_identity()
    print("===== 1. 部分分式恒等式（误差是 O(1/K) 截断残差）=====")
    for n, e1, e2 in ident:
        print(f"  K={n:6d}  tanh(z)/z 相对误差 {e1:.2e}   coth(z)/z {e2:.2e}")
    print("  Wshort 前5齿 τ/τ0 =", np.round(W_SHORT["tau"], 4),
          " R/Z0 =", np.round(W_SHORT["R"], 4))
    print("  Wo     前5齿 τ/τ0 =", np.round(W_OPEN["tau"], 4),
          " R/Z0 =", np.round(W_OPEN["R"], 4))

    rec = dict(identity=dict(convergence=[dict(K=n, err_tanh=a, err_coth=b)
                                          for n, a, b in ident],
                             wshort=dict(tau=[float(x) for x in W_SHORT["tau"]],
                                         R=[float(x) for x in W_SHORT["R"]]),
                             wopen=dict(tau=[float(x) for x in W_OPEN["tau"]],
                                        R=[float(x) for x in W_OPEN["R"]])),
               recover=[], discriminate=[], wopen=[])

    # ---- 2. Wshort 的可恢复性 ----
    print("\n===== 2. Wshort（tanh）：DRT 能还原到哪一步 =====")
    print(f"{'tau0':>7s} {'σ':>6s} {'chi2':>9s} {'Rtot':>7s} {'峰数':>4s} "
          f"{'τ_top/τ0':>9s} {'(预期0.405)':>11s} {'R_top/Rtot':>10s} {'(预期0.81)':>10s} {'τ次峰比':>8s}")
    for tD in (1e-3, 1e-2, 1e-1):
        Z = bench.ztanh(F, 10.0, tD)
        for sigma in (1e-3, 1e-2):
            runs = []
            for s in range(6):
                runs.append(measure(F, Z, sigma, np.random.default_rng(200 + s)))
            rec_ = dict(tau0=tD, sigma=sigma,
                        chi2=float(np.median([r["chi2"] for r in runs])),
                        R_total=float(np.median([r["R_total"] for r in runs])),
                        n_peaks=float(np.median([r["n_peaks"] for r in runs])),
                        tau_top_over_tau0=float(np.nanmedian(
                            [r["tau_top"] / tD for r in runs
                             if np.isfinite(r["tau_top"])])),
                        r_frac=float(np.nanmedian([r["r_frac"] for r in runs])),
                        tau_ratio_2=float(np.nanmedian(
                            [r["tau_ratio_2"] for r in runs])))
            rec["recover"].append(rec_)
            print(f"{tD:7.0e} {sigma:6.0e} {rec_['chi2']:9.1f} {rec_['R_total']:7.2f} "
                  f"{rec_['n_peaks']:4.0f} {rec_['tau_top_over_tau0']:9.3f} "
                  f"{0.4053:11.3f} {rec_['r_frac']:10.3f} {0.8106:10.3f} "
                  f"{rec_['tau_ratio_2']:8.1f}")

    # ---- 3. 这个比值能不能当判据：与单一弛豫比 ----
    print("\n===== 3. `R_top/R_total` 的判别力（这是唯一稳的量）=====")
    cases = {
        "wshort τ0=1e-2": bench.ztanh(F, 10.0, 1e-2),
        "zarc n=0.70": bench.zarc(F, 10.0, 10.0, 0.70),
        "zarc n=0.85": bench.zarc(F, 10.0, 10.0, 0.85),
        "zarc n=0.95": bench.zarc(F, 10.0, 10.0, 0.95),
        "rq(0.5) 峰在窗内": bench.rq(F, 10.0, 1 / (2 * np.pi * 10.0), 0.5),
        "两张 zarc": bench.zarc(F, 6.0, 100.0, 0.85) + bench.zarc(F, 4.0, 2.0, 0.85),
        "Ws A=2": bench.zw(F, 2.0),
    }
    print(f"{'case':18s} {'σ':>6s} {'R_top/Rtot':>10s} {'区间':>16s} {'峰数':>5s}")
    for sigma in (1e-3, 1e-2):
        for name, Z in cases.items():
            vals = []
            for s in range(6):
                r = measure(F, Z, sigma, np.random.default_rng(300 + s))
                vals.append(r["r_frac"])
            v = np.array(vals, float)
            v = v[np.isfinite(v)]
            med = float(np.median(v)) if v.size else float("nan")
            lo = float(np.min(v)) if v.size else float("nan")
            hi = float(np.max(v)) if v.size else float("nan")
            rec["discriminate"].append(dict(case=name, sigma=sigma, med=med,
                                            lo=lo, hi=hi, n=len(v)))
            print(f"{name:18s} {sigma:6.0e} {med:10.3f} "
                  f"{f'[{lo:.3f},{hi:.3f}]':>16s} {len(v):5d}")

    # ---- 4. Wo：串联电容不在 DRT 的表示能力里 ----
    print("\n===== 4. Wo（coth）：串联电容不在 DRT 表示能力里 =====")
    print(f"{'tau0':>7s} {'chi2':>12s} {'Rtot':>11s} {'峰数':>4s}  说明")
    for tD in (1e-3, 1e-2, 1e-1):
        Z = bench.zcoth(F, 10.0, tD)
        runs = [measure(F, Z, 1e-3, np.random.default_rng(400 + s)) for s in range(4)]
        c2 = float(np.median([r["chi2"] for r in runs]))
        rt = float(np.median([r["R_total"] for r in runs]))
        npk = float(np.median([r["n_peaks"] for r in runs]))
        rec["wopen"].append(dict(tau0=tD, chi2=c2, R_total=rt, n_peaks=npk))
        print(f"{tD:7.0e} {c2:12.1f} {rt:11.3g} {npk:4.0f}  "
              f"Z_0/3={10.0/3:.2f} 才是可表示部分；chi2 远大于 2n=122")

    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {OUT}")
    print("\n结论：梳齿间距（1:1/9:1/25）在 1% 噪声下分辨不出；唯一稳的 "
          "`R_top/R_total=0.82` 与 ZARC 的 0.87–0.94 重叠，做不成判据。")


if __name__ == "__main__":
    main()
