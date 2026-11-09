# -*- coding: utf-8 -*-
"""Ws 判据基准：带真值的合成谱库 + 混淆矩阵。

每类 20 条，参数随机，加 **1% 相对噪声**（`sd = 1%·|Z_i|`，与实测 EIS 的
相对误差口径一致）。早期版本用常数 σ=1%·|Z|max，结果 `|Z|` 跨 3 个量级的
低频点被绝对噪声淹没（`Re` 出现 -89 Ω），那是构造错误，不是判据结论。

**标签是构造真值**（这条谱里到底有没有 `A_W(jω)^(-1/2)` 项），所以这里算的
是判据本身的灵敏度/特异度，不是物理结论。

类别与期望
  ws_vis     真含 Ws，|Z_W(f_min)| ≈ 1–2×R_p     → candidate / Ws_or_wide_CPE
  ws_buried  真含 Ws，幅值只有 R_p 的 30%         → candidate（看能不能捡起来）
  wo_in      真含 Wo，f_D 在窗内                  → candidate / Wo_like
  wo_below   真含 Wo，f_D 在窗下                  → 与 Ws 同形，应给 Ws_or_wide_CPE
  wshort_in  真含 Wshort，f_D 在窗内              → candidate / Wshort_like
  none_2zarc 两个真弛豫，无扩散                   → reject
  rq05_low   R||CPE(0.5)，峰远在窗下              → **与 Ws 同形，允许 candidate**
  rq05_in    R||CPE(0.5)，峰在窗内                → reject
  sorp_rl    R||L 并联吸附环（低频感抗）          → reject
"""
import json
import sys
import numpy as np

sys.path.insert(0, r"D:/eis训练数据/bmdrt")
sys.path.insert(0, r"D:/eis训练数据/bmdrt/evidence/ws_crit")

import bm_wscrit as W


# --------------------------------------------------------------------------
# 元件（数值稳定版；bench 自己的实现，避免 import 生产模块的私有函数）
# --------------------------------------------------------------------------

def zarc(f, R, f0, phi):
    return R / (1.0 + (1j * 2 * np.pi * f / (2 * np.pi * f0)) ** phi)


def rq(f, R, tau, n):
    Q = tau ** n / R
    return R / (1.0 + R * Q * (1j * 2 * np.pi * f) ** n)


def zw(f, A):
    return A / np.sqrt(1j * 2 * np.pi * f)


def _cothz(z):
    az = np.abs(z)
    out = np.empty_like(z)
    sm, lg = az < 1e-3, az > 30.0
    mid = ~(sm | lg)
    zs = z[sm]
    out[sm] = 1.0 / zs ** 2 + 1.0 / 3.0
    e2 = np.exp(2.0 * z[mid])
    out[mid] = (e2 + 1.0) / (e2 - 1.0) / z[mid]
    out[lg] = 1.0 / z[lg]
    return out


def zcoth(f, Rw, tD):
    return Rw * _cothz(np.sqrt(1j * 2 * np.pi * np.asarray(f, float) * tD))


def ztanh(f, Rw, tD):
    """`R_W·tanh(z)/z`（有限长**透射**，Wshort）。

    早期版本写成 `Rw*(1/_cothz(z))/z`：`_cothz` 返回的是 `coth(z)/z`，
    所以那样算出来是 `Rw·tanh(z)`——**漏了一次 `/z`，高低频行为整体反了**
    （低频趋于 0 而不是 `R_W`）。`make()` 一直用的是正确的 `_ztanh`，
    所以基准结果不受影响；只有外部直接调 `ztanh` 的脚本会中招
    （2026-11-09 做 N2 梳齿时踩到）。这里直接转调稳定实现。
    """
    return _ztanh(f, Rw, tD)


def z_rl_par(f, Rw, L):
    w = 2 * np.pi * f
    zl = 1j * w * L
    return Rw * zl / (Rw + zl)


# --------------------------------------------------------------------------
# 谱生成
# --------------------------------------------------------------------------

def make(f, cls, rng):
    fmin, fmax = f.min(), f.max()
    Rinf = rng.uniform(2.0, 30.0)
    Rp = rng.uniform(15.0, 120.0)
    f0 = 10 ** rng.uniform(0.3, 3.5)
    phi = rng.uniform(0.70, 0.95)
    base = Rinf + zarc(f, Rp, f0, phi)
    ratio = rng.uniform(1.0, 2.0)

    if cls == "ws_vis":
        A = ratio * Rp * np.sqrt(2 * np.pi * fmin)
        Z = base + zw(f, A)
    elif cls == "ws_buried":
        A = 0.3 * Rp * np.sqrt(2 * np.pi * fmin)
        Z = base + zw(f, A)
    elif cls == "wo_in":
        fD = 10 ** rng.uniform(np.log10(fmin * 3), np.log10(fmax / 30))
        tD = 1.0 / (2 * np.pi * fD)
        Z = base + zcoth(f, ratio * Rp, tD)
    elif cls == "wo_below":
        k = rng.uniform(1.0, 3.0)
        tD = (fmin / 10 ** k) ** (-1) / (2 * np.pi)
        Rw = ratio * Rp * np.sqrt(2 * np.pi * fmin * tD)
        Z = base + zcoth(f, Rw, tD)
    elif cls == "wshort_in":
        fD = 10 ** rng.uniform(np.log10(fmin * 3), np.log10(fmax / 30))
        tD = 1.0 / (2 * np.pi * fD)
        Z = base + _ztanh(f, ratio * Rp, tD)
    elif cls == "none_2zarc":
        R2 = rng.uniform(0.3, 1.0) * Rp
        f2 = 10 ** rng.uniform(np.log10(fmin * 3), np.log10(fmax / 30))
        Z = base + zarc(f, R2, f2, rng.uniform(0.70, 0.95))
    elif cls == "rq05_low":
        k = rng.uniform(2.0, 4.0)
        fpk = fmin / 10 ** k
        R = ratio * Rp * 10 ** (k / 2)          # 使 |Z_CPE(f_min)| = ratio·Rp
        Z = base + rq(f, R, 1.0 / (2 * np.pi * fpk), 0.5)
    elif cls == "rq05_in":
        fpk = 10 ** rng.uniform(np.log10(fmin * 3), np.log10(fmax / 30))
        Z = base + rq(f, ratio * Rp, 1.0 / (2 * np.pi * fpk), 0.5)
    elif cls == "sorp_rl":
        fL = 10 ** rng.uniform(np.log10(fmin * 2), np.log10(fmin * 20))
        L = ratio * Rp / (2 * np.pi * fL)
        Z = base + z_rl_par(f, ratio * Rp, L)
    else:
        raise ValueError(cls)
    return Z


def _ztanh(f, Rw, tD):
    """tanh(z)/z，数值稳定版。"""
    z = np.sqrt(1j * 2 * np.pi * np.asarray(f, float) * tD)
    az = np.abs(z)
    out = np.empty_like(z)
    sm, lg = az < 1e-3, az > 30.0
    mid = ~(sm | lg)
    zs = z[sm]
    out[sm] = 1.0 - zs ** 2 / 3.0
    e2 = np.exp(-2.0 * z[mid])
    out[mid] = (1.0 - e2) / (1.0 + e2) / z[mid]
    out[lg] = 1.0 / z[lg]
    return Rw * out


CLASSES = ["ws_vis", "ws_buried", "wo_in", "wo_below", "wshort_in",
           "none_2zarc", "rq05_low", "rq05_in", "sorp_rl"]

FAMILY_EXPECT = {
    "ws_vis": "Ws_or_wide_CPE",
    "ws_buried": "Ws_or_wide_CPE",
    "wo_in": "Wo_like",
    "wo_below": "Ws_or_wide_CPE",
    "wshort_in": "Wshort_like",
    "none_2zarc": None,
    "rq05_low": "Ws_or_wide_CPE",
    "rq05_in": None,
    "sorp_rl": None,
}


def run(n_per=20, npts=31, fmin=0.1, fmax=1e3, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    f = np.logspace(np.log10(fmin), np.log10(fmax), npts)
    rows = []
    for cls in CLASSES:
        for i in range(n_per):
            Z = make(f, cls, rng)
            sd = noise * np.abs(Z)
            Zn = Z + sd * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
            d = W.ws_diagnostics(f, Zn)
            rows.append(dict(cls=cls, i=i, verdict=d["verdict"], family=d["family"],
                             term=d["term"], term_se=d["term_se"],
                             term_im_ratio=d["term_im_ratio"],
                             band=d["band_dec"],
                             slope_im=d["slope_im"], amp45=d["amp45"], amp90=d["amp90"]))
    return rows


def report(rows, title=""):
    if title:
        print(f"\n{title}")
    print(f"{'class':11s} {'n':>3s} {'cand':>5s} {'rej':>4s} {'undec':>6s} | "
          f"{'Ws_or_wide':>10s} {'Wo_like':>8s} {'Wshort':>7s} {'induc':>6s} "
          f"{'unknown':>8s} | 判读")
    for cls in CLASSES:
        sub = [r for r in rows if r["cls"] == cls]
        n = len(sub)
        v = [r["verdict"] for r in sub]
        fam = [r["family"] for r in sub]
        cand = sum(x == "candidate" for x in v)
        rej = sum(x == "reject" for x in v)
        und = sum(x == "undecidable" for x in v)
        exp = FAMILY_EXPECT[cls]
        if exp is None:
            note = f"应 reject，实得 {cand}/{n} 假阳"
        else:
            hit = sum(x == exp for x in fam)
            note = f"应 {exp}，家族命中 {hit}/{n}"
        print(f"{cls:11s} {n:3d} {cand:5d} {rej:4d} {und:6d} | "
              f"{fam.count('Ws_or_wide_CPE'):10d} {fam.count('Wo_like'):8d} "
              f"{fam.count('Wshort_like'):7d} {fam.count('inductive_tail'):6d} "
              f"{fam.count('unknown'):8d} | {note}")


def main():
    all_rows = []
    for tag, (fmin, fmax) in {"0.1–1e3 Hz": (0.1, 1e3),
                              "0.01–1e5 Hz": (0.01, 1e5)}.items():
        rows = run(fmin=fmin, fmax=fmax, seed=0)
        report(rows, f"===== 窗口 {tag}，31 点，1% 相对噪声 =====")
        for r in rows:
            r["window"] = tag
        all_rows += rows
    out = r"D:/eis训练数据/bmdrt/evidence/ws_crit/bench_rows.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(all_rows, fh, ensure_ascii=False, indent=1)
    print(f"\n逐条结果写入 {out}")


if __name__ == "__main__":
    main()
