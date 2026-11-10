#!/usr/bin/env python
"""K 臂诊断：K=2 + L 的 R̂≈1.84 是哪一维没定住。

只用一条谱（`2EMC1-1_0050mA.csv`，K 臂里 K=2 全 5 seed 降级的那条），
同一预算/同一 seed 下把 4 个配置各跑一次，逐参数打 R̂ 与后验分位：

    (K=1, noL) (K=1, +L) (K=2, noL) (K=2, +L)  + (K=2, +L, seed=4) 复现

判读规则（写回 §9.4.17.2）：
  · 坏的是 `L` 自己（R̂ 高、后验跨几个十年级）→ L 在 K=2 下**不被数据约束**；
    若同时 elpd 仍大涨，说明涨的是**拟合**而非 L，K 臂的 Δ 不能读成「需要电感」。
  · 坏的是某个 `tau` / `phi` / `zeta` / `v`（L 自己 R̂ 正常）→ K=2+L 下有交换
    简并，属**采样几何**问题，K 臂结论应降级为「不可判」而非「L 无用」。

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

import bm_core as C                                   # noqa: E402
import bm_real as R                                   # noqa: E402
import bm_select as S                                 # noqa: E402

NAME = "2EMC1-1_0050mA.csv"
TARGET_ACCEPT = 0.98                                  # 与 §9.4.16.3 一致
CFG = ((1, False, 0), (1, True, 0), (2, False, 0), (2, True, 0), (2, True, 4))


def _rhat2(x):
    """两条等长链的 R̂（split-less；与 numpyro 的 rank 版不同但同量级）。"""
    n = len(x) // 2
    if n < 4:
        return float("nan")
    ch = np.stack([x[:n], x[n:2 * n]])
    m, s2 = ch.mean(axis=1), ch.var(axis=1, ddof=1)
    w, b = s2.mean(), n * m.var(ddof=1)
    if not np.isfinite(w) or w <= 0:
        return float("nan")
    return float(np.sqrt(max(((n - 1) / n * w + b / n) / w, 1e-12)))


def per_param(s: dict):
    """摊平 plate 维，逐元素算 R̂ / 中位 / 5% / 95%。"""
    out = []
    for nm, arr in s.items():
        a = np.asarray(arr, float)
        if a.ndim == 2:
            for j in range(a.shape[1]):
                out.append((f"{nm}[{j}]", a[:, j]))
        else:
            out.append((nm, a))
    return [(nm, _rhat2(v), float(np.median(v)),
             float(np.percentile(v, 5)), float(np.percentile(v, 95)))
            for nm, v in out]


def show(k: int, il: bool, seed: int):
    f, Z = R.load_spectrum(os.path.join("archive", NAME))
    kw = dict(S.BUDGETS["quick"])
    kw.update(target_accept=TARGET_ACCEPT, progress=False)
    print(f"\n===== K={k}  include_L={il}  seed={seed} =====")
    s, mc = C.fit(f, Z, K=k, seed=seed, include_L=il, **kw)
    tab = sorted(per_param(s), key=lambda t: -t[1])
    print(f"  {'参数':<12}{'R̂':>7}{'中位':>13}{'5%':>13}{'95%':>13}")
    for nm, r, med, lo, hi in tab:
        if r > 1.02 or nm in ("W", "R_inf", "alpha", "L"):
            mark = "  ←坏" if r > 1.05 else ""
            print(f"  {nm:<12}{r:>7.3f}{med:>13.5g}{lo:>13.5g}{hi:>13.5g}{mark}")
    if "L" in s:
        L = np.asarray(s["L"], float).ravel()
        p5, p50, p95 = np.percentile(L, [5, 50, 95])
        print(f"  L 后验 5/50/95 = {p5*1e9:.1f} / {p50*1e9:.1f} / {p95*1e9:.1f} nH"
              f"（跨度 {np.log10(p95/p5):.2f} decade）")
    if k > 1:
        tau = np.asarray(s["tau"], float)
        print(f"  tau 排序一致率 = {float(np.mean(tau[:, 0] < tau[:, 1])):.3f}"
              "（0.5 = 标签互换）")
    return s, mc


def main(argv=None):
    print(f"谱 {NAME}，预算 quick，target_accept={TARGET_ACCEPT}，"
          f"num_chains={S.BUDGETS['quick']['num_chains']}")
    for k, il, seed in CFG:
        show(k, il, seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
