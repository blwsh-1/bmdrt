# -*- coding: utf-8 -*-
"""3E 谱与外部工具截图的**只读对照**：差距来自噪声口径还是结构自由度？

用户截图：外部工具 `3E-1_OER_exp1_0005mA.csv`（67 点 0.1–1e5 Hz）用
`R1 + PhaHF + 3×(R|CPE) + CPEtail` 得 `Rwp 0.65416%`，候选 28 选 1，
我们的电路排第 5。本项目同谱输出在 `out/3E-1_OER_exp1_0005mA/`
（voigt K=2 + 高频 L，自由绝对 σ，acf 0.743）。

外部 `Rwp = sqrt(mean((|ΔZ|/|Z|)^2))`（§9.4.29 已证，未除 √2）；
本项目 `rms_rel = Rwp/√2`。重算项目后验中位曲线得 Rwp 1.4726%。

本脚本不改任何生产代码、不重跑验收、不改判定，只做 **LS 上界** 2×2 对照：
    噪声口径：绝对 σ（项目默认） / 逐点 2%·|Z|（外部疑似口径）
    结构    ：voigt K=2（我们） / K=3（多一个 HN，模拟多一个 (R|CPE)）
LS 是给定模型下的拟合上界：若某配置连 LS 最优都压不到 0.65%，
说明该因素不是差距来源。所有结果标 `exploratory`。

    python evidence/nested_topology/m4_3e_cmp.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
import bm_experiments as E          # noqa: E402

NPZ = os.path.join("out", "3E-1_OER_exp1_0005mA",
                   "real_3E-1_OER_exp1_0005mA_samples.npz")
OUT = os.path.join("out", "m4_3e_cmp.json")
EXT_RWP = 0.65416                  # 截图，%
N_STARTS = 12
SEED = 20261111
PHI_LO, PHI_HI = 0.5, 1.0


def model_z(f, x, K):
    """`R_inf + jωL + Σ w_i/(1+(jωτ_i)^φ_i)`（ζ=1 的 HN 串联 voigt 族）。"""
    w2 = 2.0 * np.pi * np.asarray(f, float)
    lw = x[:K]
    lt = x[K:2 * K]
    ph = x[2 * K:3 * K]
    rinf, ll = 10.0 ** x[3 * K], 10.0 ** x[3 * K + 1]
    z = np.full(f.size, rinf, complex) + 1j * w2 * ll
    for i in range(K):
        z = z + (10.0 ** lw[i]) / (1.0 + (1j * w2 * (10.0 ** lt[i])) ** ph[i])
    return z


def bounds(K):
    lo = np.concatenate([np.full(K, -6.0), np.full(K, -9.0),
                         np.full(K, PHI_LO), [-6.0], [-12.0]])
    hi = np.concatenate([np.full(K, 3.0), np.full(K, 4.0),
                         np.full(K, PHI_HI), [3.0], [-3.0]])
    return lo, hi


def starts(K, z, ref, n_starts):
    """第 0 个起点用项目 MCMC 的中位参数，其余按 0.6 dec 抖动。"""
    rng = np.random.default_rng(SEED)
    lo, hi = bounds(K)
    w_ref = np.asarray([np.log10(max(v, 1e-6)) for v in ref["w"]])
    t_ref = np.asarray([np.log10(max(v, 1e-9)) for v in ref["tau"]])
    p_ref = np.asarray(ref["phi"], float)
    base2 = np.concatenate([w_ref[:2], t_ref[:2], p_ref[:2],
                            [np.log10(max(ref["R_inf"], 1e-6)),
                             np.log10(max(ref["L"], 1e-12))]])
    if K == 3:
        # 第三个分量放在中频，初值取前两分量的几何平均
        base = np.concatenate([
            list(w_ref[:2]) + [0.5 * (w_ref[0] + w_ref[1]) - 0.5],
            list(t_ref[:2]) + [0.5 * (t_ref[0] + t_ref[1])],
            list(p_ref[:2]) + [0.8],
            [base2[-2], base2[-1]]])
    else:
        base = base2
    out = [np.clip(base, lo + 1e-6, hi - 1e-6)]
    for _ in range(n_starts - 1):
        j = rng.normal(0, 0.6, size=base.size)
        j[2 * K:3 * K] *= 0.15
        out.append(np.clip(base + j, lo + 1e-6, hi - 1e-6))
    return out


def fit(K, sigma_mode, f, Z, ref):
    sig = (0.02 * np.abs(Z) if sigma_mode == "perpoint"
           else np.full(f.size, 0.02 * float(np.abs(Z).max())))
    lo, hi = bounds(K)

    def resid(x):
        d = model_z(f, x, K) - Z
        return np.concatenate([d.real / sig, d.imag / sig])

    best = None
    for x0 in starts(K, Z, ref, N_STARTS):
        r = least_squares(resid, x0, bounds=(lo, hi), max_nfev=3000,
                          ftol=1e-11, xtol=1e-11, gtol=1e-11)
        cost = float(np.sqrt(np.mean(r.fun ** 2)))
        if best is None or cost < best[0]:
            best = (cost, r.x)
    x = best[1]
    Zm = model_z(f, x, K)
    d = (Zm - Z) / np.abs(Z)
    r = Zm - Z
    return dict(K=K, sigma_mode=sigma_mode,
                rwp_ext_pct=float(np.sqrt(np.mean(np.abs(d) ** 2))) * 100,
                rms_proj_pct=float(np.sqrt(np.mean(
                    np.concatenate([d.real, d.imag]) ** 2))) * 100,
                acf=float(max(E.acf_max(r.real), E.acf_max(r.imag))),
                top_rel_pct=float(np.abs(d[np.argmax(f)])) * 100,
                bands_pct=[float(np.sqrt(np.mean(np.abs(d[m]) ** 2))) * 100
                           for m in ((f >= 0.1) & (f < 10),
                                     (f >= 10) & (f < 1000),
                                     (f >= 1000) & (f <= 1e5))],
                cost=best[0])


def main() -> int:
    z = np.load(NPZ)
    f, Z = z["f"], z["Zre"] + 1j * z["Zim"]
    med = lambda k: np.median(np.asarray(z[k], float), axis=0)
    ref = dict(R_inf=float(med("R_inf")), W=float(med("W")),
               w=float(med("W")) * np.asarray(med("pi"), float),
               tau=np.asarray(med("tau"), float),
               phi=np.asarray(med("phi"), float), L=float(med("L")))
    # 项目 MCMC 后验中位曲线的同口径 Rwp（基准）
    import bm_core as B
    p = dict(R_inf=ref["R_inf"], W=ref["W"], pi=med("pi"), tau=ref["tau"],
             phi=ref["phi"], zeta=np.ones(2), L=ref["L"])
    Zm0 = B.model_z(f, p, include_L=True)
    d0 = (Zm0 - Z) / np.abs(Z)
    base = dict(rwp_ext_pct=float(np.sqrt(np.mean(np.abs(d0) ** 2))) * 100,
                K=2, sigma_mode="free_absolute_mcmc")

    rows = []
    for K in (2, 3):
        for mode in ("absolute", "perpoint"):
            r = fit(K, mode, f, Z, ref)
            rows.append(r)
            print(f"  K={K} {mode:<9} Rwp {r['rwp_ext_pct']:7.4f}%  "
                  f"acf {r['acf']:.3f}  低/中/高 "
                  + "/".join(f"{v:.2f}%" for v in r["bands_pct"]), flush=True)
    print(f"\n  项目 MCMC 基准 Rwp {base['rwp_ext_pct']:.4f}%  "
          f"|  外部截图 {EXT_RWP:.5f}%  "
          f"|  基准/外部 = {base['rwp_ext_pct'] / EXT_RWP:.2f}x")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(dict(status="exploratory", spec="3E-1_OER_exp1_0005mA",
                       external_rwp_pct=EXT_RWP, mcmc_baseline=base,
                       note="LS 上界；不改任何验收判定，不写生产代码",
                       rows=rows), fh, ensure_ascii=False, indent=1)
    print(f"已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
