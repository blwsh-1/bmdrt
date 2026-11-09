# -*- coding: utf-8 -*-
"""N4：`Ws` 的可检测下限曲面——按**信噪比**给答案，而不是按经验幅值比。

问题
----
`bench.py` 里 `ws_buried`（幅值 `0.3·R_p`）只有 10–11/20 通过，`ws_vis`
（`1–2·R_p`）是 20/20。但"0.3·R_p"本身不是原因——**同一个幅值比，噪声大一倍
就判不出来**。真正控制可检测性的是**低频端扩散项的信噪比**

    SNR_im = |Z_W(f_min)| / ( σ_rel · |Z(f_min)| )

因为判据看的是 `-Im` 的幂律段，而噪声是按 `σ_rel·|Z|` 加的（相对噪声口径，
与实测一致）。本脚本扫 `幅值比 × σ`，同时记录 `SNR_im`，回答两件事：

  1. `P(candidate)` 在 `SNR_im` 上是不是单变量决定（如果是，就能直接当门槛用）；
  2. 0.5 交点在哪——低于它的谱应当回 `undecidable`，而不是硬判。

**这是判据的能力边界，不是物理结论。** `truth` 由构造给出（属于合成谱），
不能替代真实谱验收，不改 `G8b` 的 `pending`。

产物：`limit_surface.json`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:/eis训练数据/bmdrt")
sys.path.insert(0, r"D:/eis训练数据/bmdrt/evidence/ws_crit")

import bm_wscrit as W
from bench import zarc, zw

ROOT = Path(r"D:/eis训练数据/bmdrt")
OUT = ROOT / "evidence" / "ws_crit" / "limit_surface.json"

RATIOS = [0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
SIGMAS = [0.0025, 0.005, 0.01, 0.02]
N_PER = 20
WINDOWS = {"0.1–1e3 Hz": (0.1, 1e3), "0.01–1e5 Hz": (0.01, 1e5)}


def one(f, ratio, rng):
    """`R_inf + ZARC + Ws`，`|Z_W(f_min)| = ratio·R_p`。返回 (Z, rp, A)。"""
    fmin = f.min()
    Rinf = rng.uniform(2.0, 30.0)
    Rp = rng.uniform(15.0, 120.0)
    f0 = 10 ** rng.uniform(0.3, 3.5)
    phi = rng.uniform(0.70, 0.95)
    A = ratio * Rp * np.sqrt(2 * np.pi * fmin)      # |Z_W| = A/√(2πf)
    return Rinf + zarc(f, Rp, f0, phi) + zw(f, A), Rp, A, Rinf


def main():
    rows = []
    cell = 0
    for tag, (fmin, fmax) in WINDOWS.items():
        f = np.logspace(np.log10(fmin), np.log10(fmax), 31)
        for ratio in RATIOS:
            for sigma in SIGMAS:
                # 种子按格子编号给（不能用 hash(str)：Python 字符串哈希每进程
                # 都变，结果不可复现）
                rng = np.random.default_rng(20261109 + cell)
                cell += 1
                for _ in range(N_PER):
                    Z, Rp, A, Rinf = one(f, ratio, rng)
                    sd = sigma * np.abs(Z)
                    Zn = Z + sd * (rng.standard_normal(f.size)
                                   + 1j * rng.standard_normal(f.size))
                    d = W.ws_diagnostics(f, Zn)
                    re_w = A / np.sqrt(4 * np.pi * fmin)     # 窗内可测的实部
                    rows.append(dict(
                        window=tag, ratio=ratio, sigma=sigma,
                        Rp=float(Rp), A=float(A),
                        # 低频端扩散项的"信号"与其噪声地板
                        snr_im=float((A / np.sqrt(2 * np.pi * fmin))
                                     / (sigma * abs(Z[0]))),
                        snr_re=float(re_w / (sigma * abs(Z[0]))),
                        verdict=d["verdict"], family=d["family"], term=d["term"],
                        slope_im=d["slope_im"], band=d["band_dec"],
                    ))

    # ---------- 表 1：P(candidate) 随幅值比 × σ ----------
    for tag in WINDOWS:
        print(f"\n===== 窗口 {tag}，每格 {N_PER} 条，`|Z_W(f_min)| = ratio·R_p` =====")
        head = f"{'ratio':>6s} |" + "".join(
            f"  σ={s*100:.2f}%".rjust(13) for s in SIGMAS)
        print(head)
        for ratio in RATIOS:
            cells = []
            for sigma in SIGMAS:
                sub = [r for r in rows
                       if r["window"] == tag and r["ratio"] == ratio
                       and r["sigma"] == sigma]
                pc = np.mean([r["verdict"] == "candidate" for r in sub])
                pu = np.mean([r["verdict"] == "undecidable" for r in sub])
                cells.append(f"{pc:.2f}/U{pu:.2f}".rjust(13))
            print(f"{ratio:6.2f} |" + "".join(cells))
        print("（单元 = P(candidate)/U = P(undecidable)；其余为 reject）")

    # ---------- 表 2：P(candidate) 对 SNR_im 分箱 ----------
    print("\n===== P(candidate) vs 低频 -Im 信噪比 SNR_im（两窗口合并）=====")
    bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20),
            (20, 40), (40, 1e9)]
    print(f"{'SNR_im':>14s} {'n':>5s} {'P(cand)':>8s} {'P(undec)':>9s} {'P(rej)':>7s}")
    curve = []
    for lo, hi in bins:
        sub = [r for r in rows if lo <= r["snr_im"] < hi]
        if not sub:
            continue
        pc = np.mean([r["verdict"] == "candidate" for r in sub])
        pu = np.mean([r["verdict"] == "undecidable" for r in sub])
        pr = np.mean([r["verdict"] == "reject" for r in sub])
        lab = f"[{lo:g},{hi:g})" if hi < 1e8 else f"[{lo:g},∞)"
        print(f"{lab:>14s} {len(sub):5d} {pc:8.2f} {pu:9.2f} {pr:7.2f}")
        curve.append(dict(lo=lo, hi=hi, n=len(sub), p_cand=float(pc),
                          p_undec=float(pu), p_rej=float(pr)))

    # ---------- 检测限：在 SNR 分箱的**条件概率**上插值 ----------
    # 不能用 P(candidate | SNR ≥ t)：那是累积量，被高 SNR 的长尾拉高
    # （实测 t 取到最小值时它仍有 0.65，等于什么都没说）。
    # 用分箱内的 P(candidate) 才对应"这个 SNR 的谱判得出来吗"。
    print("\n===== 检测限（按 SNR 分箱条件概率插值）=====")
    cand = np.array([r["verdict"] == "candidate" for r in rows])
    s_all = np.array([r["snr_im"] for r in rows], float)
    # 用 curve 自带的边界（空箱会被跳过，按 bins 取会错位一格）
    ctr = [(c["lo"] * c["hi"]) ** 0.5 if c["hi"] < 1e8 else c["lo"] * 1.5
           for c in curve]
    pv = np.array([c["p_cand"] for c in curve], float)
    lx = np.log10(np.array(ctr, float))

    def _cross(target):
        for i in range(len(pv) - 1):
            if pv[i] < target <= pv[i + 1]:
                w = (target - pv[i]) / (pv[i + 1] - pv[i])
                return float(10 ** (lx[i] + w * (lx[i + 1] - lx[i])))
        return None

    lim50 = _cross(0.5)
    lim90 = _cross(0.9)
    print(f"  P(candidate)=50% → SNR_im ≈ {lim50:.0f}" if lim50 else
          "  P(candidate)=50%：插值范围内不跨越")
    print(f"  P(candidate)=90% → SNR_im ≈ {lim90:.0f}" if lim90 else
          f"  P(candidate)=90%：**达不到**（实测天花板 {pv.max():.2f}，"
          f"因为族内还有 to_cap/noisy 误档）")
    print(f"  candidate 的 SNR_im 中位数 {np.median(s_all[cand]):.1f}，"
          f"非 candidate 的 {np.median(s_all[~cand]):.1f}")

    limits = dict(p50=lim50, p90=lim90, ceiling=float(pv.max()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(dict(rows=rows, curve=curve, limits=limits), fh,
                  ensure_ascii=False, indent=1)
    print(f"\n逐条写入 {OUT}（{len(rows)} 条）")


if __name__ == "__main__":
    main()
