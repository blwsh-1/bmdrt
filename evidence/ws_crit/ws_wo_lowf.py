"""Ws 与 Wo 的差别分布在哪里？能靠"重点看低频偏差"判它们吗？

数学：ωτ_D >> 1 时 coth(z)/z → 1/z，Wo ≡ Ws（同一高频尾，A_W=R_W/√τ_D）。
      只有 ωτ_D ≲ 1（f ≲ f_D）两者才分开 → 差别**确实只在低频**。

本脚本：
  ① 用本谱两组实测参数，算 |Z_Ws − Z_Wo| 的频段分布
  ② 扫 (R_W, τ_D) 画"可分辨相图"：在 σ=0.098、0.1–10⁴ Hz 下，
     max|ΔZ|/σ ≥ 2 的区域 = 理论上能分清 Ws 与 Wo 的参数区
  ③ 把本谱与 caseE 的实际点投到相图上
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bm_diff as D

ROOT, STEM = "out/_wp2_blank", "Blank_0.1KBr_pH10_EIS_PtCu_1.2V"
SIG = 0.098


def z_ws(f, aw):
    return aw * (1j * 2 * np.pi * f) ** -0.5


def z_wo(f, rw, td):
    return D.z_wopen(f, rw, td)


def main():
    js = json.load(open(f"{ROOT}/k2_Wo/{STEM}/real_{STEM}.json", encoding="utf-8"))
    jk1 = json.load(open(f"{ROOT}/k1_Wo/{STEM}/real_{STEM}.json", encoding="utf-8"))
    f = np.load(f"{ROOT}/k2_Wo/{STEM}/real_{STEM}_samples.npz", allow_pickle=True)["f"]
    o = np.argsort(f)
    fs = f[o]

    print("=" * 76)
    print("① 同一高频尾下 |Z_Ws − Z_Wo| 的频段分布（按 f/f_D 归一）")
    for tag, d in (("K=1 +Wo", jk1["diffusion_params"]),
                   ("K=2 +Wo", js["diffusion_params"])):
        rw, td = d["Rwo"], d["tauD"]
        fD = 1.0 / (2 * np.pi * td)
        aw = rw / np.sqrt(td)
        dz = z_ws(fs, aw) - z_wo(fs, rw, td)
        ad = np.abs(dz)
        print(f"\n  {tag}: R_W={rw:.3f} Ω  τ_D={td:.4f} s  f_D={fD:.4f} Hz  "
              f"A_W(等效)={aw:.3f}")
        print(f"    max|ΔZ| = {ad.max():.4f} Ω = {ad.max()/SIG:.2f}σ，"
              f"出现在 f={fs[ad.argmax()]:.3f} Hz（f/f_D={(fs/ad.argmax() if False else fs[ad.argmax()]/fD):.2f}）")
        for lo, hi in ((0.1, 0.3), (0.3, 1), (1, 10), (10, 100), (100, 1e4)):
            m = (fs >= lo) & (fs < hi)
            if m.sum():
                print(f"    {lo:>7.2f}–{hi:<7.0f} Hz  n={m.sum():2d}  "
                      f"mean|ΔZ|={ad[m].mean():7.4f}  max={ad[m].max():7.4f} Ω "
                      f"({ad[m].max()/SIG:4.2f}σ)")
        print(f"    ΔZ 的相位：@{fs[0]:.2f} Hz {np.degrees(np.angle(dz[0])):6.1f}°，"
              f"@{fs[-1]:.0f} Hz {np.degrees(np.angle(dz[-1])):6.1f}°")

    print("\n" + "=" * 76)
    print("② 可分辨相图：给定 (R_W, τ_D)，max|ΔZ|/σ 在 0.1–10⁴ Hz 内")
    td_grid = np.logspace(-2.5, 2.5, 46)
    rw_grid = np.logspace(-1.5, 2.0, 46)
    ratio = np.empty((len(rw_grid), len(td_grid)))
    for i, rw in enumerate(rw_grid):
        for j, td in enumerate(td_grid):
            aw = rw / np.sqrt(td)
            ratio[i, j] = np.abs(z_ws(fs, aw) - z_wo(fs, rw, td)).max() / SIG

    print(f"    σ={SIG} Ω；判据 max|ΔZ| ≥ 2σ = {2*SIG:.3f} Ω")
    print("    τ_D 上限（f_D>0.1 Hz）= 1.59 s；以下为「刚好 2σ」所需 R_W：")
    for td in (0.01, 0.03, 0.1, 0.3, 1.0, 1.59, 3.0, 10.0):
        aw = None
        j = int(np.argmin(np.abs(td_grid - td)))
        col = ratio[:, j]
        k = np.argmax(col >= 2.0)
        if col[k] >= 2.0:
            print(f"      τ_D={td:6.2f} s  →  R_W ≥ {rw_grid[k]:7.3f} Ω")
        else:
            print(f"      τ_D={td:6.2f} s  →  扫描上限内始终 <2σ（分不开）")

    # 图
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    cs = ax.contourf(td_grid, rw_grid, np.log10(ratio), levels=30, cmap="viridis")
    c0 = ax.contour(td_grid, rw_grid, ratio, levels=[2.0], colors="w", linewidths=2.0)
    ax.clabel(c0, fmt="2σ", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("τ_D (s)   →   f_D = 1/(2πτ_D) decreases to the right")
    ax.set_ylabel("R_W (Ω)")
    ax.axvline(1 / (2 * np.pi * 0.1), color="w", ls="--", lw=1.2)
    ax.text(1 / (2 * np.pi * 0.1) * 1.08, 30, "f_D = 0.1 Hz\n(window edge)", color="w",
            fontsize=8, va="center")
    pts = [("K=1 +Wo\n(12.4 Ω, 9.2 s)", 9.17, 12.36, "o"),
           ("K=2 +Wo\n(0.061 Ω, 0.50 s)", 0.495, 0.061, "s"),
           ("req. for 2σ\nat τ_D=0.5 s", 0.495, None, "^")]
    for lbl, td, rw, mk in pts:
        if rw is None:
            j = int(np.argmin(np.abs(td_grid - td)))
            col = ratio[:, j]
            k = np.argmax(col >= 2.0)
            rw = rw_grid[k]
            lbl = f"min R_W for 2σ\n= {rw:.2f} Ω"
        ax.plot(td, rw, mk, ms=10, mfc="none", mec="r", mew=2.0)
        ax.annotate(lbl, (td, rw), textcoords="offset points", xytext=(10, 8),
                    color="r", fontsize=8.5)
    fig.colorbar(cs, ax=ax, label="log10( max|Z_Ws−Z_Wo| / σ )")
    ax.set_title(f"Can Ws and Wo be told apart?  (σ={SIG} Ω, 0.1–10⁴ Hz)\n"
                 "white line = 2σ; below it they are indistinguishable", fontsize=10)
    fig.tight_layout()
    out = f"{ROOT}/ws_wo_phase_map.png"
    fig.savefig(out, dpi=150)
    print("\n图已存：", out)


if __name__ == "__main__":
    main()
