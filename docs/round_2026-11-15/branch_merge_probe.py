# -*- coding: utf-8 -*-
"""只读探针：把 `series` / `parallel` / `capacitive` 三条支路**合进一个模型**行不行。

用户问的是「三条支路能不能合在一起」。这件事有两个完全不同的层次，本脚本只回答第二个：

1. **合在候选库里并列比较** —— 已经支持（`circuit_registry` 的 `branch` 槽位
   `min_count=1, max_count=1`，`circuit_runner --branches`），三条各自独立拟合、各自出观测。
2. **合进同一个候选（一个模型同时挂三条支路）** —— registry 的 `_nested_consistency`
   会直接拒（同一槽位只能挂一个元件）。本脚本量化「如果硬合，会得到什么」。

做法：在 `AZ31-12Li_0` 这条感抗最强的谱上，用**确定性多起点最小二乘**（不做 MCMC，
避免采样噪声混淆）把四种模型拟合到数据：

| 变体 | 支路 |
|---|---|
| `series` | `R_L + jωL`（2 参数） |
| `parallel` | `R_ad·jωτ/(1+jωτ)`（2 参数） |
| `capacitive` | `Q_b·(ω_ref/ω)^n_b·e^{-j n_b π/2}`（2 参数） |
| `merged` | `series + parallel` **串联叠加**（4 参数） |

然后报三件事：
- `rms_rel`（相对残差，§14 口径）——多了 2 个参数到底买到了什么；
- 在 LS 最优点上参数空间的**条件数**与最大两两相关——是不是靠一个极扁的方向在降残差；
- `merged` 的解能否被 `series` 或 `parallel` 单支路在噪声内复现（曲线最大相对差）。

边界：只读、不 import 生产拟合入口做拟合（只借 `bm_top.z_cpe` 的**元件**公式）、
不改 `bm_*.py`、不改阈值、不写准入结论。结果一律 `exploratory`。

用法：
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/branch_merge_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import bm_real as R          # noqa: E402
import bm_top as T           # noqa: E402

SPEC_DIR = os.path.join("out", "_m6_samples", "z_csv")
OUT_JSON = os.path.join("out", "branch_merge_probe.json")
SIGMA_PCT = 0.02             # 与 §14/§17 同口径的逐点相对权重
N_STARTS = 12
SEED = 20261116


def z_branch(f, kind: str, p, w_ref: float):
    """支路阻抗。参数口径与 `bm_top` 一致（τ 与 L 的换算见 series）。

    `none` = 支路短接（Z_b=0），用来给「支路被解成不存在」提供对照基线。
    """
    if kind == "none":
        return np.zeros_like(np.asarray(f, float), dtype=complex)
    if kind == "series":
        R_L, tau = p
        return R_L + 1j * 2 * np.pi * f * (R_L * tau)
    if kind == "parallel":
        R_ad, tau = p
        wt = 2 * np.pi * f * tau
        return R_ad * 1j * wt / (1 + 1j * wt)
    if kind == "capacitive":
        Q_b, n_b = p
        return T.z_cpe(f, Q_b, n_b, f_ref=w_ref / (2 * np.pi))
    if kind == "merged":
        zs = z_branch(f, "series", p[:2], w_ref)
        zp = z_branch(f, "parallel", p[2:4], w_ref)
        return zs + zp                      # 串联叠加，与 bm_top 的网络拓扑一致
    if kind == "merged3":
        zs = z_branch(f, "series", p[:2], w_ref)
        zp = z_branch(f, "parallel", p[2:4], w_ref)
        zc = z_branch(f, "capacitive", p[4:6], w_ref)
        return zs + zp + zc                 # 三条全挂
    raise ValueError(kind)


def z_model(f, base, kind: str, bp, w_ref: float):
    """Rs + Q1 ∥ (R1 + [Q2 ∥ (R2 + Z_b)])，`layers=2` 骨架，与 `z_nested` 同构。"""
    Rs, R1, Q1, n1, R2, Q2, n2 = base
    z1 = T.z_cpe(f, Q1, n1, f_ref=w_ref / (2 * np.pi))
    z2 = T.z_cpe(f, Q2, n2, f_ref=w_ref / (2 * np.pi))
    zb = z_branch(f, kind, bp, w_ref)
    zinner = 1.0 / (1.0 / z2 + 1.0 / (R2 + zb))
    return Rs + 1.0 / (1.0 / z1 + 1.0 / (R1 + zinner))


def unpack(x, kind: str):
    """log 参数 → 物理参数。base 7 个 + 支路 n 个。"""
    base = tuple(np.exp(x[:7]))
    nb = {"merged": 4, "merged3": 6}.get(kind, 2)
    bp = tuple(np.exp(x[7:7 + nb]))
    return base, bp


def residual(x, f, Z, kind, w_ref):
    base, bp = unpack(x, kind)
    Zm = z_model(f, base, kind, bp, w_ref)
    d = (Zm - Z) / np.abs(Z)
    return np.concatenate([d.real, d.imag])


def rms_rel(f, Z, Zm):
    d = (Zm - Z) / np.abs(Z)
    return float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2))) * 100


def fit_one(f, Z, kind, w_ref, rng, base0, bp0):
    """多起点 LS。起点在 log 空间抖动 ±1 个量级。"""
    nb = {"merged": 4, "merged3": 6, "none": 0}.get(kind, 2)
    x0 = np.log(np.array(list(base0) + list(bp0), float))
    best = None
    for k in range(N_STARTS):
        x = x0.copy() if k == 0 else x0 + rng.normal(0.0, 1.0, size=x0.size)
        try:
            r = least_squares(residual, x, args=(f, Z, kind, w_ref),
                              method="trf", max_nfev=4000)
        except Exception:                             # noqa: BLE001
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    base, bp = unpack(best.x, kind)
    Zm = z_model(f, base, kind, bp, w_ref)
    return dict(x=best.x, base=base, bp=bp, Zm=Zm, rms=rms_rel(f, Z, Zm),
                cost=float(best.cost))


def identifiability(x, f, Z, kind, w_ref):
    """在 LS 最优点上算 log 参数 Jacobian 的条件数与最大两两相关。"""
    J = np.empty((2 * f.size, x.size))
    for j in range(x.size):
        h = 1e-4 * max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (residual(xp, f, Z, kind, w_ref)
                   - residual(xm, f, Z, kind, w_ref)) / (2 * h)
    s = np.linalg.svd(J, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    G = J.T @ J
    dg = np.sqrt(np.diag(G))
    C = G / np.outer(dg, dg)
    C = np.clip(C, -1, 1)
    np.fill_diagonal(C, 0.0)
    i, j = np.unravel_index(np.argmax(np.abs(C)), C.shape)
    names = ["Rs", "R1", "Q1", "n1", "R2", "Q2", "n2"] + list(
        {"series": ["R_L", "tau_L"], "parallel": ["R_ad", "tau_ad"],
         "capacitive": ["Q_b", "n_b"],
         "merged": ["R_L", "tau_L", "R_ad", "tau_ad"],
         "merged3": ["R_L", "tau_L", "R_ad", "tau_ad", "Q_b", "n_b"],
         "none": []}[kind])
    return dict(cond=cond, smin=float(s[-1]), smax=float(s[0]),
                max_abs_corr=float(abs(C[i, j])),
                max_corr_pair=[names[i], names[j]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="AZ31-12Li_0")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args(argv)

    f, Z = R.load_spectrum(os.path.join(SPEC_DIR, a.spec + ".csv"))
    w_ref = 2 * np.pi * float(np.sqrt(f.min() * f.max()))
    zmax = float(np.max(np.abs(Z)))
    rng = np.random.default_rng(a.seed)

    # 中性起点：全部按 zmax 缩放，不看真值
    base0 = (0.05 * zmax, 0.10 * zmax, zmax, 0.9, 0.10 * zmax, zmax, 0.9)
    bp0 = {"series": (0.10 * zmax, 1.0 / (2 * np.pi * 10.0)),
           "parallel": (0.10 * zmax, 1.0 / (2 * np.pi * 10.0)),
           "capacitive": (zmax, 0.9),
           "merged": (0.10 * zmax, 1.0 / (2 * np.pi * 10.0),
                      0.10 * zmax, 1.0 / (2 * np.pi * 10.0)),
           "merged3": (0.10 * zmax, 1.0 / (2 * np.pi * 10.0),
                       0.10 * zmax, 1.0 / (2 * np.pi * 10.0),
                       zmax, 0.9),
           "none": ()}

    rows = {}
    print(f"谱 {a.spec}  n={f.size}  f {f.min():.3g}–{f.max():.3g} Hz  "
          f"|Z| {np.abs(Z).min():.3g}–{zmax:.3g} Ω\n")
    print(f"{'变体':<12}{'参数数':>6}{'rms%':>8}{'cond(J)':>12}"
          f"{'最大|相关|':>11}  对应参数对")
    for kind in ("none", "series", "parallel", "capacitive", "merged", "merged3"):
        r = fit_one(f, Z, kind, w_ref, rng, base0, bp0[kind])
        if r is None:
            rows[kind] = dict(failed=True)
            print(f"{kind:<12}  拟合失败")
            continue
        idf = identifiability(r["x"], f, Z, kind, w_ref)
        rows[kind] = dict(rms=r["rms"], n_params=int(r["x"].size),
                          base=[float(v) for v in r["base"]],
                          branch=[float(v) for v in r["bp"]], **idf)
        print(f"{kind:<12}{r['x'].size:>6}{r['rms']:>8.2f}{idf['cond']:>12.3g}"
              f"{idf['max_abs_corr']:>11.3f}  {idf['max_corr_pair']}")

    # merged 的解能否被单支路在噪声内复现
    if "merged" in rows and "rms" in rows["merged"]:
        Zm = fit_one(f, Z, "merged", w_ref, rng, base0, bp0["merged"])["Zm"]
        for kind in ("series", "parallel"):
            if "rms" not in rows.get(kind, {}):
                continue
            r2 = fit_one(f, Z, kind, w_ref, rng, base0, bp0[kind])
            d = np.abs(Zm - r2["Zm"]) / np.abs(Z)
            rows[f"merged_vs_{kind}"] = dict(max_rel_diff=float(np.max(d)),
                                             hf_max_rel_diff=float(np.max(d[f >= 1e4])))
            print(f"\nmerged 曲线 vs {kind} 最优曲线：最大相对差 "
                  f"{np.max(d) * 100:.2f}%（1e4–1e5 段 {np.max(d[f >= 1e4]) * 100:.2f}%）")
        print(f"\nmerged 的 rms = {rows['merged']['rms']:.2f}%，"
              f"比 {min(rows['series']['rms'], rows['parallel']['rms']):.2f}% 只低 "
              f"{min(rows['series']['rms'], rows['parallel']['rms']) - rows['merged']['rms']:.2f} 个百分点，"
              f"代价是多 2 个参数、cond {rows['merged']['cond']:.3g} vs "
              f"{min(rows['series']['cond'], rows['parallel']['cond']):.3g}")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(spec=a.spec, seed=a.seed, sigma_pct=SIGMA_PCT,
                       n_starts=N_STARTS, note="只读探针；LS 上界；exploratory",
                       rows=rows), fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"\n产物：{a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
