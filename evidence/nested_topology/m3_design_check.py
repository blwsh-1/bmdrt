# -*- coding: utf-8 -*-
"""档 2 真值组的事前可辨识性筛查（**跑 MCMC 之前**就能算清楚的一组数）。

问题来源：档 2 第一组 5 个 seed 里 `R_ad/tau_ad/L_ad` 恢复到几个百分点，
而 `R1/R2/Q1/Q2/n1/n2` 的后验 90% 区间宽到几百 %。这些参数的"覆盖率 5/5"
不构成恢复证据——区间宽到能把任何真值包进去。

本脚本不猜原因，做两件事：

1. **边际敏感度** `S_i = ‖∂Z/∂θ_i‖/σ`：该参数单独能把 Z 推动多少个 σ。
   注意 S 大**不代表**可辨识——若两列方向几乎平行，同时放开就分不开。
2. **条件化预测**：`F = JᵀJ/σ²`（J 为 ∂Z/∂θ），`sd_i = sqrt([F⁻¹]_ii)`。
   这是"其它参数全自由时，数据单独能把这个参数定到多准"（Cramér–Rao，平坦先验）。
   与先验 sd 比：`sd_i ≫ 先验 sd` → 后验≈先验，该参数在本配置下不可辨识。

`sd` 对对数参数是 log 尺度（=相对误差），对 `n` 是绝对量。

用法：
    python m3_design_check.py                    # 三个真值组对照
    python m3_design_check.py --obs <samples.npz>  # 再加一列实测 log-sd
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bm_top_gen as G                                   # noqa: E402

F_GRID = (1e-2, 1e5, 61)
NOISE = 0.02
LOG_PARAMS = ("Rs", "R1", "R2", "Q1", "Q2", "R_ad", "tau_ad", "R_L", "tau_L")


def spectrum(p: dict, f: np.ndarray) -> np.ndarray:
    return G.gen_parallel_L2(f, p)


def jacobian(p: dict, f: np.ndarray, h: float = 1e-3):
    """∂Z/∂θ，θ 为对数参数（相对扰动）或 n1/n2（绝对扰动）。"""
    cols = {}
    for k in p:
        if k in ("n1", "n2"):
            pp, pm = dict(p), dict(p)
            pp[k], pm[k] = p[k] + h, p[k] - h
            d = (spectrum(pp, f) - spectrum(pm, f)) / (2 * h)
        else:
            pp, pm = dict(p), dict(p)
            pp[k], pm[k] = p[k] * np.exp(h), p[k] * np.exp(-h)
            d = (spectrum(pp, f) - spectrum(pm, f)) / (2 * h)
        cols[k] = np.concatenate([d.real, d.imag])
    return cols


def report(name: str, p: dict, obs: dict | None = None) -> dict:
    f = np.logspace(np.log10(F_GRID[0]), np.log10(F_GRID[1]), F_GRID[2])
    z0 = spectrum(p, f)
    zmax = float(np.max(np.abs(z0)))
    sigma = NOISE * zmax
    cols = jacobian(p, f)
    ks = list(cols)

    J = np.column_stack([cols[k] for k in ks]) / sigma
    F = J.T @ J
    sd = np.sqrt(np.clip(np.diag(np.linalg.pinv(F, rcond=1e-14)), 0, None))
    sens = {k: float(np.linalg.norm(cols[k]) / sigma) for k in ks}

    print(f"\n{'='*94}")
    print(f"{name}")
    print(f"  zmax={zmax:.1f} Ω  σ={sigma:.2f} Ω  |Z| {np.abs(z0).min():.2f}..{np.abs(z0).max():.1f} Ω"
          f"  σ/|Z| 中位 {np.median(sigma/np.abs(z0)):.0%}")
    print(f"  {'参数':<7}{'边际 S':>9}{'预测 sd':>11}{'先验 sd':>10}{'sd/先验':>10}"
          f"{'实测 log-sd':>13}  判定")
    out = {"zmax": zmax, "sigma": sigma, "params": {}}
    for k in ks:
        prior_sd = 0.5 / np.sqrt(12) if k in ("n1", "n2") else 2.0   # LogNormal(s=2) / Uniform(.5,1)
        ratio = sd[ks.index(k)] / prior_sd
        verdict = ("先验主导" if ratio > 1 else
                   "数据有信息" if ratio < 0.5 else "两者相当")
        o = ""
        if obs and k in obs:
            o = f"{obs[k]:>13.2f}"
        print(f"  {k:<7}{sens[k]:>9.1f}{sd[ks.index(k)]:>11.3f}{prior_sd:>10.3f}"
              f"{ratio:>10.2f}{o:>13}  {verdict}")
        out["params"][k] = {"S": sens[k], "sd_pred": float(sd[ks.index(k)]),
                            "prior_sd": prior_sd, "ratio": float(ratio),
                            "verdict": verdict}

    M = np.column_stack([cols[k] / np.linalg.norm(cols[k]) for k in ks])
    C = M.T @ M
    pairs = [(ks[i], ks[j], float(C[i, j]))
             for i in range(len(ks)) for j in range(i + 1, len(ks))]
    pairs.sort(key=lambda t: -abs(t[2]))
    print("  最共线对（|cos|≈1 → 只能定组合）："
          + "、".join(f"{a}/{b} {c:+.3f}" for a, b, c in pairs[:3]))
    out["collinear"] = [{"a": a, "b": b, "cos": c} for a, b, c in pairs[:3]]

    # 组合参数的预测精度：∇ᵀ cov ∇（对数尺度），与实测对照
    cov = np.linalg.pinv(F, rcond=1e-14)
    idx = {k: i for i, k in enumerate(ks)}
    combos = {}
    if {"R1", "R2"} <= set(ks):
        w = np.zeros(len(ks))
        tot = p["R1"] + p["R2"]
        w[idx["R1"]], w[idx["R2"]] = p["R1"] / tot, p["R2"] / tot
        combos["log(R1+R2)"] = w
    if {"R_ad", "tau_ad"} <= set(ks):
        w = np.zeros(len(ks))
        w[idx["R_ad"]] = w[idx["tau_ad"]] = 1.0
        combos["log L_ad"] = w
    print(f"  {'组合':<13}{'预测 log-sd':>14}{'实测':>10}")
    out["combos"] = {}
    for name, w in combos.items():
        s = float(np.sqrt(max(w @ cov @ w, 0.0)))
        key = "R1+R2" if name.startswith("log(R1") else "L_ad"
        o = f"{obs[key]:>10.3f}" if obs and key in obs else ""
        print(f"  {name:<13}{s:>14.3f}{o:>10}")
        out["combos"][name] = {"sd_pred": s,
                              "sd_obs": (obs.get(key) if obs else None)}
    return out


BASE = dict(Rs=10.0, R1=55.0, R2=600.0, Q1=200.0, n1=0.75,
            Q2=900.0, n2=0.85, R_ad=3000.0, tau_ad=1.6)


def load_obs(path: str) -> dict:
    z = np.load(path)
    out = {}
    for k in z.files:
        v = np.asarray(z[k], float)
        out[k] = float(np.std(np.log(v))) if k not in ("n1", "n2") else float(np.std(v))
    if {"R1", "R2"} <= set(z.files):
        out["R1+R2"] = float(np.std(np.log(z["R1"] + z["R2"])))
    if {"R_ad", "tau_ad"} <= set(z.files):
        out["L_ad"] = float(np.std(np.log(z["R_ad"] * z["tau_ad"])))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default=None, help="posterior 样本 npz，加一列实测 sd")
    args = ap.parse_args(argv)

    if args.obs:
        print(__doc__)
    obs = load_obs(args.obs) if args.obs else None

    report("A 基准：R_ad=3000, τ=1.6（档 2 现用真值）", BASE, obs)
    report("B 感抗环减弱：R_ad=300, τ=1.6", dict(BASE, R_ad=300.0), obs)
    report("C 容性进窗：Q1=R1=55, Q2=R2=600（ω_ref 处 |Z_CPE|=Q）",
           dict(BASE, Q1=float(BASE["R1"]), Q2=float(BASE["R2"])), obs)

    print(f"\n{'='*94}")
    print("读法：预测 sd 是『其它参数全自由时，这个参数被数据定到的精度』（log 尺度）。")
    print("      预测 sd ≫ 先验 sd → 后验基本等于先验，此时『90% 区间覆盖真值』无意义。")
    print("      预测 sd 小 → 数据真的有信息；再看 cos 判断是定单个还是只定组合。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
