#!/usr/bin/env python
"""K 臂诊断：K=2 的降级到底坏在哪个参数、坏到什么程度。

只用一条谱（`2EMC1-1_0050mA.csv`，K 臂里 K=2 五 seed 全降的那条），在**生产路径**
（`bm_select.fit_voigt`，ζ 固定 1.0、quick 预算、target_accept=0.98）上跑 4 个配置：

    (K=1, noL) (K=2, noL) (K=2, +L, seed 0) (K=2, +L, seed 4)

**必须走 `fit_voigt` 而不是 `bm_core.fit`**：K 臂冻结的是 ζ=1.0，`bm_core.fit` 默认
ζ 自由（0.3–2.0）。第一版诊断就是直接调 `bm_core.fit`，同一格得到 R̂=1.101，
而 K 臂记录的是 1.848——不是采样不稳定，是**模型不同**（多一个 ζ 维）。

每个配置报四样：
  ① `rhat_max`（`fit_one` 口径 = rank-normalized，逐元素取最大，与 K 臂同源）；
  ② 逐参数 R̂，只列 >1.05 的和 L/R_inf/W；
  ③ `L` 后验 5/50/95（nH）与跨几个 decade；
  ④ 排序统计量下的分量幅值与 τ 跨度（`bm_select.unresolved_components`）+ guard 状态。

跑法：
    python evidence/nested_topology/m4_armLK_diag.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "evidence", "nested_topology")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bm_experiments as E                             # noqa: E402
import bm_select as S                                 # noqa: E402

NAME = "2EMC1-1_0050mA.csv"
TARGET_ACCEPT = 0.98                                  # 与 §9.4.16.3 一致
CFG = ((1, False, 0), (2, False, 0), (2, True, 0), (2, True, 4))


def show(k: int, il: bool, seed: int) -> dict:
    import bm_real as R
    f, Z = R.load_spectrum(os.path.join("archive", NAME))
    kw = dict(S.BUDGETS["quick"])
    kw.update(target_accept=TARGET_ACCEPT)
    c = S.fit_voigt(f, Z, k, seed=seed, include_L=il, kw=kw)
    raw = c["raw"]
    s = raw["s"]
    print(f"\n===== K={k}  include_L={il}  seed={seed} =====")
    print(f"  rhat_max(fit_one) = {c['rhat']:.3f}   div={c['div']}   "
          f"elpd={c['elpd']:.2f}")

    mc = raw.get("mc")
    if mc is not None:
        ch = mc.get_samples(group_by_chain=True)
        for r in E.full_param_diag(ch):
            mark = "  ←坏" if r["rhat"] > S.RHAT_MAX else ""
            if r["rhat"] > 1.02 or r["name"] in ("L", "R_inf", "W"):
                print(f"    {r['name']:<8}R̂={r['rhat']:.3f}{mark}")
    if "L" in s:
        L = np.asarray(s["L"], float).ravel()
        p5, p50, p95 = np.percentile(L, [5, 50, 95])
        print(f"    L 后验 5/50/95 = {p5*1e9:.1f} / {p50*1e9:.1f} / {p95*1e9:.1f} nH"
              f"（跨 {np.log10(p95/p5):.2f} decade）")
    e = S.unresolved_components(s)
    if e:
        span = " / ".join(f"{v:.2f}" for v in e["span_dec"])
        amp = " / ".join(f"{v:.3f}" for v in e["amp_frac"])
        gs = S.guard_state(dict(family="voigt", raw=dict(s=s)),
                           int(S.BUDGETS["quick"]["num_chains"]
                               * S.BUDGETS["quick"]["n_samples"]))["comp_unresolved"]
        print(f"    τ 跨度(排序后, dec) = {span}   幅值 w/W = {amp}"
              f"   → comp_unresolved={gs}")
        # 窗口外的分量：τ 上限对应频率是否低于最底频率
        tau_hi = np.exp(np.percentile(np.sort(np.log(np.asarray(s["tau"], float)),
                                              axis=1)[:, -1], 95))
        print(f"    最大分量 τ 的 95% = {tau_hi:.3g} s → {1/(2*np.pi*tau_hi):.3g} Hz"
              f"（窗口下界 {float(f.min()):.3g} Hz）")
    if k > 1:
        tau = np.asarray(s["tau"], float)
        print(f"    tau 列排序一致率 = {float(np.mean(tau[:, 0] < tau[:, 1])):.3f}"
              "（0.5 = 标签互换）")
    return c


def main(argv=None):
    print(f"谱 {NAME}，走 fit_voigt（ζ=1.0），预算 quick，"
          f"target_accept={TARGET_ACCEPT}")
    for k, il, sd in CFG:
        show(k, il, sd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
