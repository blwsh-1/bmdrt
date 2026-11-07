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
import json
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
        prior_sd = PRIOR_SD[k]
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

# 先验 sd：LogNormal(s=2) 在 log 尺度上是 2；Uniform(0.5,1) 是 0.5/√12。
# 注意这两个数必须与 bm_top._prior_table() 保持一致，否则"先验主导"的判定会错。
PRIOR_SD = {k: 2.0 for k in ("Rs", "R1", "R2", "Q1", "Q2", "R_ad", "tau_ad",
                             "R_L", "tau_L")}
PRIOR_SD.update(n1=0.5 / np.sqrt(12), n2=0.5 / np.sqrt(12))

# A 组是现用真值；B1/B2/B3 是感抗环逐级减弱的候选。只动 R_ad，
# 其它真值固定 —— 一次只改一个因子，否则分不清是谁改善了信息。
CANDIDATES = [("A", 3000.0), ("B1", 1000.0), ("B2", 500.0), ("B3", 300.0)]


def band_stats(z0: np.ndarray, sigma: float) -> dict:
    """三频段 σ/|Z| 中位与信噪比>1 的点数。频栅是对数等距，按索引均分即按 log f 均分。"""
    a = np.abs(z0)
    thirds = np.array_split(np.arange(a.size), 3)
    med = [float(np.median(sigma / a[idx])) for idx in thirds]
    return {"sigma_over_z_bands": med,
            "n_snr_gt1": int(np.sum(sigma < a)),
            "sigma_over_z_median": float(np.median(sigma / a))}


def combo_sd(p: dict, ks: list[str], cov: np.ndarray) -> dict:
    """组合参数在 log 尺度上的预测 sd：∇ᵀ cov ∇。"""
    idx = {k: i for i, k in enumerate(ks)}
    out = {}
    if {"R1", "R2"} <= set(ks):
        w = np.zeros(len(ks))
        tot = p["R1"] + p["R2"]
        w[idx["R1"]], w[idx["R2"]] = p["R1"] / tot, p["R2"] / tot
        out["log(R1+R2)"] = float(np.sqrt(max(w @ cov @ w, 0.0)))
    if {"R_ad", "tau_ad"} <= set(ks):
        w = np.zeros(len(ks))
        w[idx["R_ad"]] = w[idx["tau_ad"]] = 1.0
        out["log L_ad"] = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return out


def scan_candidate(name: str, r_ad: float) -> dict:
    """单个候选真值组的信息量画像（不打印，只返回结构）。"""
    return predict(dict(BASE, R_ad=float(r_ad)), name=name)


def predict(p: dict, name: str = "", mode: str = "abs") -> dict:
    """给定真值与噪声口径，算 zmax/σ、条件化预测 sd、组合参数与频段信噪比。

    `mode="abs"`：固定 σ = 2%·zmax（A/B 组用的口径）；
    `mode="rel"`：逐点 σ_j = 2%·|Z_j|（**另一种口径**，只能作为参考上限，
                  不可与 abs 的结果直接比较）。
    """
    f = np.logspace(np.log10(F_GRID[0]), np.log10(F_GRID[1]), F_GRID[2])
    z0 = spectrum(p, f)
    zmax = float(np.max(np.abs(z0)))
    sigma = NOISE * zmax
    cols = jacobian(p, f)
    ks = list(cols)
    J = np.column_stack([cols[k] for k in ks])
    if mode == "abs":
        J = J / sigma
    else:
        w = np.concatenate([NOISE * np.abs(z0), NOISE * np.abs(z0)])
        J = J / w[:, None]
    cov = np.linalg.pinv(J.T @ J, rcond=1e-14)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    return {"name": name, "R_ad": float(p["R_ad"]), "zmax": zmax, "sigma": sigma,
            "mode": mode,
            "params": {k: {"sd_pred": float(sd[i]),
                           "ratio": float(sd[i] / PRIOR_SD[k]),
                           "prior_sd": PRIOR_SD[k]} for i, k in enumerate(ks)},
            "combos": combo_sd(p, ks, cov), **band_stats(z0, sigma)}


def print_levers() -> list[dict]:
    """为什么"只把 R_ad 降下来"不够：换另外两个杠杆看同一张表。

    变体一：形状幅值整体放大（R1/R2/Q1/Q2 ×k）——看起来应该提高信噪比，
            实际上 zmax 与 σ 同步变大，比值几乎不动。
    变体二：把噪声口径换成逐点相对（σ_j = 2%|Z_j|）——所有形状参数立刻进"有信息"区。
            **这是另一个实验条件，不与 A/B 组比较**，只用来说明限制来自口径而非电路。
    """
    shape = ("R1", "R2", "Q1", "n1", "Q2", "n2")
    variants = [("B3 R_ad=300", dict(BASE, R_ad=300.0), "abs"),
                ("B3 + 形状×3", dict(BASE, R_ad=300., R1=165., R2=1800.,
                                     Q1=600., Q2=2700.), "abs"),
                ("B3 + 形状×10", dict(BASE, R_ad=300., R1=550., R2=6000.,
                                      Q1=2000., Q2=9000.), "abs"),
                ("B3 逐点相对 2%|Z| *", dict(BASE, R_ad=300.0), "rel")]
    print(f"\n{'='*104}")
    print("旁证：换杠杆（形状幅值 vs 噪声口径）。带 * 的一行是另一种噪声条件，"
          "只作参考，不与 A/B 组比较。")
    print(f"{'='*104}")
    print(f"{'变体':<24}{'zmax':>9}{'σ':>10}" + "".join(f"{k:>8}" for k in shape))
    rows = []
    for name, p, mode in variants:
        r = predict(p, name=name, mode=mode)
        print(f"{name:<24}{r['zmax']:>9.0f}{r['sigma']:>10.1f}"
              + "".join(f"{r['params'][k]['ratio']:>8.2f}" for k in shape))
        rows.append(r)
    print("-" * 104)
    print("读法：放大形状幅值不改善比值（σ 随 zmax 同步涨）；换逐点相对口径后")
    print("      六个形状参数全部落到 <1 —— 说明瓶颈是『固定 σ 相对整条谱标定』这个口径，")
    print("      不是电路结构，也不是采样器。")
    return rows


def print_scan(scans: list[dict]) -> None:
    """候选对照表：一行一组，格子是 sd/先验 sd 的倍数（<1 才叫"数据有信息"）。"""
    shape = ("R1", "R2", "Q1", "n1", "Q2", "n2")
    print(f"\n{'='*104}")
    print("候选真值扫描（只改 R_ad，其余真值固定；噪声口径固定 2%·zmax，未人为降低）")
    print(f"{'='*104}")
    print(f"{'组':<4}{'R_ad':>7}{'zmax':>9}{'σ':>8}"
          f"{'σ/|Z| 低':>10}{'中':>8}{'高':>8}{'SNR>1':>7}"
          + "".join(f"{k:>8}" for k in shape)
          + f"{'L_ad':>9}{'R1+R2':>9}")
    for s in scans:
        bands = s["sigma_over_z_bands"]
        print(f"{s['name']:<4}{s['R_ad']:>7.0f}{s['zmax']:>9.1f}{s['sigma']:>8.2f}"
              f"{bands[0]:>10.0%}{bands[1]:>8.0%}{bands[2]:>8.0%}"
              f"{s['n_snr_gt1']:>7d}"
              + "".join(f"{s['params'][k]['ratio']:>8.2f}" for k in shape)
              + f"{s['combos']['log L_ad']:>9.3f}"
              f"{s['combos']['log(R1+R2)']:>9.3f}")
    print("-" * 104)
    print("格子含义：预测后验 sd ÷ 先验 sd（log 尺度）。"
          ">1 = 后验被先验撑住（该参数在本组不可辨识）；<0.5 = 数据真的有信息。")
    print("末两列为组合参数的绝对预测 sd（如 0.026 = 2.6%），它们通常比单参数小一个量级。")


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
    ap.add_argument("--detail", default=None,
                    help="对指定候选（如 B3）打印完整逐参数表，可逗号分隔")
    ap.add_argument("--json", default=None, help="把扫描结果写成 json")
    args = ap.parse_args(argv)

    if args.obs:
        print(__doc__)
    obs = load_obs(args.obs) if args.obs else None

    scans = [scan_candidate(n, x) for n, x in CANDIDATES]
    print_scan(scans)
    levers = print_levers()

    detail = set((args.detail or "").split(",")) if args.detail else set()
    for s in scans:
        if s["name"] in detail or args.obs:
            p = dict(BASE, R_ad=s["R_ad"])
            label = f"{s['name']}：R_ad={s['R_ad']:.0f}, τ=1.6" + \
                    ("（档 2 现用真值）" if s["name"] == "A" else "")
            report(label, p, obs if s["name"] == "A" else None)

    print(f"\n{'='*104}")
    print("读法：预测 sd 是『其它参数全自由时，这个参数被数据定到的精度』（log 尺度）。")
    print("      预测 sd ≫ 先验 sd → 后验基本等于先验，此时『90% 区间覆盖真值』无意义。")
    print("      预测 sd 小 → 数据真的有信息；再看 cos 判断是定单个还是只定组合。")
    print("      选组只看『信息量是否够』，不看『结果是否好看』——噪声口径不得为选组而改。")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"f_grid": F_GRID, "noise_frac": NOISE, "base": BASE,
                        "candidates": scans, "levers": levers},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"扫描结果写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
