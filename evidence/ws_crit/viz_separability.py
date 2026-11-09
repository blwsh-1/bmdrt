"""可视化"不可区分"：六臂拟合曲线与数据噪声带的相对大小。

问：拟合后的图能不能把 Ws / Wo / K+1 分开？
做法：用各臂后验中位数重建 Z(f)，算
  ① 曲线两两最大差（Ω 与 σ_im 两种刻度）
  ② 差值落在噪声带内的比例
  ③ 画 Nyquist（全窗/低频放大）、残差、相位四联图

输出：out/_wp2_blank/separability.png + 终端数值表
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bm_core as B

ROOT = "out/_wp2_blank"
STEM = "Blank_0.1KBr_pH10_EIS_PtCu_1.2V"
ARMS = ["k1_none", "k1_Ws", "k1_Wo", "k2_none", "k2_Ws", "k2_Wo"]
COL = {"k1_none": "#888888", "k1_Ws": "#1f77b4", "k1_Wo": "#17becf",
       "k2_none": "#d62728", "k2_Ws": "#2ca02c", "k2_Wo": "#ff7f0e"}
LBL = {"k1_none": "K=1 none", "k1_Ws": "K=1 + Ws", "k1_Wo": "K=1 + Wo",
       "k2_none": "K=2 none", "k2_Ws": "K=2 + Ws", "k2_Wo": "K=2 + Wo"}


def load(tag):
    b = f"{ROOT}/{tag}/{STEM}/real_{STEM}"
    js = json.load(open(b + ".json", encoding="utf-8"))
    z = np.load(b + "_samples.npz", allow_pickle=True)
    p = {
        "R_inf": float(np.median(z["R_inf"])),
        "W": float(np.median(z["W"])),
        "pi": np.median(z["pi"], axis=0),
        "tau": np.median(z["tau"], axis=0),
        "phi": np.median(z["phi"], axis=0),
        "zeta": np.median(z["zeta"], axis=0),
    }
    if js["include_L"]:
        p["L"] = float(np.median(z["L"]))
    dp = js.get("diffusion_params") or {}
    if "Aw" in dp:
        p["Aw"] = float(dp["Aw"])
    if "Rwo" in dp:
        p["Rwo"] = float(dp["Rwo"])
        p["tauD"] = float(dp["tauD"])
    f = z["f"]
    Zobs = z["Zre"] + 1j * z["Zim"]
    Zmod = B.model_z(f, p, include_L=bool(js["include_L"]))
    return dict(tag=tag, f=f, Zobs=Zobs, Zmod=Zmod, js=js,
                sig_im=js["sigma_im"], sig_re=js["sigma_re"], p=p)


def main():
    A = {t: load(t) for t in ARMS}
    f = A["k2_none"]["f"]
    Zo = A["k2_none"]["Zobs"]

    print("=" * 78)
    print("① 各臂拟合残差（自报 σ 与 rms，单位 Ω）")
    print(f"{'arm':10s} {'σ_im':>8s} {'rms_im':>8s} {'σ_re':>8s} {'rms_re':>8s} "
          f"{'elpd':>9s} {'grade':>6s}")
    for t in ARMS:
        d = A[t]
        print(f"{LBL[t]:10s} {d['sig_im']:8.3f} {d['js']['rms_im']:8.3f} "
              f"{d['sig_re']:8.3f} {d['js']['rms_re']:8.3f} "
              f"{d['js']['elpd_loo']:9.2f} {d['js']['sample_grade']:6d}")

    print("\n" + "=" * 78)
    print("② 曲线两两最大差：Δmax = max|Z_a−Z_b|（Ω），及折成 K=2 噪声的倍数")
    print("   σ_im(K=2) ≈ %.3f Ω；>1σ 才可能在图上被看出来" % A["k2_none"]["sig_im"])
    sig = A["k2_none"]["sig_im"]
    pairs = [("k2_none", "k2_Ws"), ("k2_none", "k2_Wo"), ("k2_Ws", "k2_Wo"),
             ("k1_Ws", "k1_Wo"), ("k1_Ws", "k2_none"), ("k1_Wo", "k2_none")]
    print(f"{'pair':22s} {'Δmax_im':>9s} {'Δmax_re':>9s} {'Δmax_|Z|':>10s} {'/σ_im':>7s}")
    for a, b in pairs:
        dz = A[a]["Zmod"] - A[b]["Zmod"]
        mi = np.abs(dz.imag).max()
        mr = np.abs(dz.real).max()
        mz = np.abs(dz).max()
        print(f"{LBL[a]+' vs '+LBL[b]:22s} {mi:9.3f} {mr:9.3f} {mz:10.3f} {mz/sig:7.1f}")

    print("\n" + "=" * 78)
    print("③ 低频端（最低 10 点，0.1–2 Hz）曲线差 与 残差差")
    lo = f <= 2.0
    print(f"{'pair':22s} {'Δmax_im(lo)':>12s} {'Δ_maxresid':>11s} {'/σ':>6s}")
    for a, b in pairs:
        dz = (A[a]["Zmod"] - A[b]["Zmod"])[lo]
        ra = (A[a]["Zmod"] - Zo).imag[lo]
        rb = (A[b]["Zmod"] - Zo).imag[lo]
        dres = np.abs(ra - rb).max()
        print(f"{LBL[a]+' vs '+LBL[b]:22s} {np.abs(dz.imag).max():12.3f} "
              f"{dres:11.3f} {dres/sig:6.1f}")

    # ---------------- 图 ----------------
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 9.4))
    fig.suptitle(f"Can the fitted curves tell them apart?  ({STEM[:28]}…)\n"
                 f"posterior-median curves, {len(f)} points, σ_im(K=2)={sig:.3f} Ω",
                 fontsize=11)

    # (a) Nyquist 全窗
    a0 = ax[0, 0]
    a0.plot(Zo.real, -Zo.imag, "o", ms=4.2, mfc="none", mec="k", mew=0.9,
            label="data", zorder=5)
    for t in ARMS:
        d = A[t]
        a0.plot(d["Zmod"].real, -d["Zmod"].imag, "-", lw=1.5, color=COL[t],
                label=LBL[t], zorder=3)
    a0.set_xlabel("Re Z (Ω)")
    a0.set_ylabel("−Im Z (Ω)")
    a0.set_title("(a) Nyquist, full window", fontsize=10)
    a0.legend(fontsize=7.5, loc="upper right")
    a0.set_aspect("equal", adjustable="datalim")
    a0.grid(alpha=0.25)

    # (b) Nyquist 低频放大
    a1 = ax[0, 1]
    a1.plot(Zo.real[lo], -Zo.imag[lo], "o", ms=6, mfc="none", mec="k", mew=1.0,
            label="data", zorder=5)
    for t in ["k2_none", "k2_Ws", "k2_Wo"]:
        d = A[t]
        a1.plot(d["Zmod"].real[lo], -d["Zmod"].imag[lo], ".-", lw=1.6, ms=5,
                color=COL[t], label=LBL[t])
    a1.set_xlabel("Re Z (Ω)")
    a1.set_ylabel("−Im Z (Ω)")
    a1.set_title("(b) Nyquist, low-f zoom (0.1–2 Hz) — K=2 arms", fontsize=10)
    a1.legend(fontsize=8)
    a1.grid(alpha=0.25)

    # (c) 残差 Im vs f，K=2 三臂 + ±2σ 带
    a2 = ax[1, 0]
    a2.axhspan(-2 * sig, 2 * sig, color="0.85", zorder=0, label="±2σ_im band")
    for t in ARMS:
        d = A[t]
        r = (d["Zmod"] - Zo).imag
        a2.semilogx(d["f"], r, "o-", ms=3, lw=0.9, color=COL[t], label=LBL[t],
                    alpha=0.95 if t.startswith("k2") else 0.55)
    a2.axhline(0, color="k", lw=0.6)
    a2.set_xlabel("f (Hz)")
    a2.set_ylabel("Im residual (Ω)")
    a2.set_title(f"(c) Im residual vs f  (band = ±2×{sig:.3f} Ω)", fontsize=10)
    a2.legend(fontsize=7, ncol=2)
    a2.grid(alpha=0.25)

    # (d) 相位
    a3 = ax[1, 1]
    a3.semilogx(f, np.degrees(np.angle(Zo)), "o", ms=4.5, mfc="none", mec="k",
                mew=0.9, label="data", zorder=5)
    for t in ARMS:
        d = A[t]
        a3.semilogx(d["f"], np.degrees(np.angle(d["Zmod"])), "-", lw=1.3,
                    color=COL[t], label=LBL[t], alpha=0.95 if t.startswith("k2") else 0.55)
    a3.set_xlabel("f (Hz)")
    a3.set_ylabel("phase (deg)")
    a3.set_title("(d) Bode phase", fontsize=10)
    a3.legend(fontsize=7, ncol=2)
    a3.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f"{ROOT}/separability.png"
    fig.savefig(out, dpi=150)
    print("\n图已存：", out)

    # ---- 补充：配对累积残差差（相对 k2_none）——Δelpd 的几何 ----
    o = np.argsort(f)
    fs = f[o]
    ref = "k2_none"
    cum_ref = np.cumsum((A[ref]["Zmod"] - Zo).imag[o])
    fig2, ax2 = plt.subplots(figsize=(8.2, 5.0))
    k = np.arange(1, len(fs) + 1)
    env = 2 * np.sqrt(2) * sig * np.sqrt(k)          # 配对差 → √2
    ax2.plot(fs, env, "--", color="0.55", lw=1.1,
             label="±2√2·σ·√k envelope")
    ax2.plot(fs, -env, "--", color="0.55", lw=1.1)
    for t in ARMS:
        if t == ref:
            continue
        cum = np.cumsum((A[t]["Zmod"] - Zo).imag[o]) - cum_ref
        ax2.semilogx(fs, cum, "o-", ms=3, lw=1.5, color=COL[t],
                     label=f"{LBL[t]} − {LBL[ref]}")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel("f (Hz)")
    ax2.set_ylabel(f"cumulative Im residual difference vs {LBL[ref]} (Ω)")
    ax2.set_title("Paired cumulative residual — most sensitive plot one can draw\n"
                  "(inside dashed envelope ⇒ indistinguishable at 2σ)", fontsize=10)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    out2 = f"{ROOT}/separability_cum.png"
    fig2.savefig(out2, dpi=150)
    print("图已存：", out2)

    print("\n" + "=" * 78)
    print(f"④ 配对累积差终点（相对 {LBL[ref]}）与 2√2σ√n 包络")
    n = len(fs)
    envn = 2 * np.sqrt(2) * sig * np.sqrt(n)
    for t in ARMS:
        if t == ref:
            continue
        cum = np.cumsum((A[t]["Zmod"] - Zo).imag[o])[-1] - cum_ref[-1]
        print(f"  {LBL[t]:10s} Σdiff={cum:+8.3f} Ω   包络 ±{envn:5.2f} Ω   "
              f"{'包络内 → 不可区分' if abs(cum) <= envn else '包络外 → 可区分'}")

    # ---- 补充 2：差值放大图（把三臂的差拉到 ±0.1 Ω 刻度上）----
    fig3, axd = plt.subplots(1, 2, figsize=(11.6, 4.2))
    for j, (part, nm) in enumerate((("imag", "Im"), ("real", "Re"))):
        a = axd[j]
        a.axhspan(-2 * sig, 2 * sig, color="0.88", zorder=0, label="±2σ")
        a.axhspan(-sig, sig, color="0.72", zorder=0, label="±1σ")
        a.axhline(0, color="k", lw=0.8, zorder=2)
        base = getattr(A[ref]["Zmod"], part)
        for t in ["k2_Ws", "k2_Wo"]:
            dz = getattr(A[t]["Zmod"], part) - base
            a.semilogx(f, dz, "o-", ms=4, lw=1.6, color=COL[t],
                       label=f"{LBL[t]} − {LBL[ref]}", zorder=3)
        a.set_xlabel("f (Hz)")
        a.set_ylabel(f"Δ{nm} Z (Ω)")
        a.set_title(f"(a) Δ{nm} vs f — zoomed to ±0.12 Ω" if j == 0
                    else f"(b) Δ{nm} vs f — zoomed to ±0.12 Ω", fontsize=10)
        a.set_ylim(-0.12, 0.12)
        a.legend(fontsize=7.5, loc="upper left", ncol=2)
        a.grid(alpha=0.25)
    fig3.suptitle("Zoom the difference by ~10× : the curves separate, but the "
                  "separation is inside the ±1σ noise band", fontsize=10)
    fig3.tight_layout(rect=(0, 0, 1, 0.93))
    out3 = f"{ROOT}/separability_zoom.png"
    fig3.savefig(out3, dpi=150)
    print("图已存：", out3)
    print("\n" + "=" * 78)
    print("⑤ 差值放大后仍落在噪声带的点数占比（K=2 三臂两两）")
    for a_, b_ in (("k2_none", "k2_Ws"), ("k2_none", "k2_Wo"), ("k2_Ws", "k2_Wo")):
        dz = np.hypot((A[a_]["Zmod"] - A[b_]["Zmod"]).real,
                      (A[a_]["Zmod"] - A[b_]["Zmod"]).imag)
        print(f"  {LBL[a_]:10s} vs {LBL[b_]:10s}  |ΔZ|<1σ: "
              f"{(dz < sig).sum():2d}/{len(dz)} 点  |ΔZ|<2σ: {(dz < 2*sig).sum():2d}/{len(dz)} 点")



if __name__ == "__main__":
    main()
