"""噪声尺度探针的图（只读）。四张图对应交付 1-4 项。

1 fig1_repeats   : 4 条重复谱的 Re / Im 叠加（看系统漂移）
2 fig2_sigma_f   : 逐点 s_Re / s_Im 随频率，叠加 M1/M2 拟合
3 fig3_sigma_Z   : s 对 |Z| 的关系，raw vs decom（常数->相对的翻转）
4 fig4_drift_art : 12 个 n>=4 组的相对离散度 vs Rs 漂移量（证明离散度不是噪声）

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/probe_figs.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import noise_scale_probe as P  # noqa: E402

OUT = os.path.join("out", "noise_scale_probe")
C = ["#1D9E75", "#378ADD", "#D85A30", "#534AB7"]
MK = ["o", "s", "^", "D"]


def style(ax, xlabel, ylabel, title=None):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, which="both", alpha=0.18, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, fontweight="500", loc="left")


def load(key):
    f, Zs, exps = P.load_group(P.discover()[key])
    return f, Zs, exps


def _p(box, kind, key):
    for c in box["cands"]:
        if c["kind"] == kind:
            return c[key]
    raise KeyError(kind)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "probe.json"), encoding="utf-8") as fh:
        probe = json.load(fh)

    key = P.PRIMARY
    f, Zs, exps = load(key)
    A2, rs = P.decompose(Zs)
    Zbar, s_re, s_im = P.disp(Zs)
    Zbar2, s_re2, _ = P.disp(list(A2))
    absZ, absZ2 = np.abs(Zbar), np.abs(Zbar2)
    det = probe["detail"]["/".join(key)]
    best_re_raw = det["forms"]["raw"]["re"]
    best_re_dec = det["forms"]["decom"]["re"]

    # ---- fig1 重复谱
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i, Z in enumerate(Zs):
        axs[0].plot(f, Z.real, MK[i], color=C[i], ms=3.2, lw=1.0,
                    label="exp%d" % exps[i])
        axs[1].plot(f, Z.imag, MK[i], color=C[i], ms=3.2, lw=1.0,
                    label="exp%d" % exps[i])
    style(axs[0], "f / Hz", "Re Z / ohm", "Repeated Re Z")
    style(axs[1], "f / Hz", "Im Z (as stored) / ohm", "Repeated Im Z")
    axs[0].legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle("3E-3 full cell 0100mA, 4 cross-experiment repeats", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_repeats.png"), dpi=160)
    plt.close(fig)

    # ---- fig2 s(f)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ax = axs[0]
    ax.plot(f, s_re, "o", color="#D85A30", ms=3.2, label="raw s_Re")
    ax.plot(f, s_re2, "s", mfc="none", color="#D85A30", ms=4.2, mew=1.0,
            label="after Rs-offset removal")
    b_raw = [c for c in best_re_raw["cands"] if c["kind"] == "M1"][0]["b"]
    ax.axhline(b_raw, color="#D85A30", lw=1.0, ls="--", alpha=0.7,
               label="M1 const b=%.3g" % b_raw)
    m2 = [c for c in best_re_dec["cands"] if c["kind"] == "M2" and c["x"] == "|Z|"][0]
    ax.plot(f, m2["a"] * absZ2, color="#7F77DD", lw=1.4,
            label="M2 a|Z|, a=%.2f%%" % (100 * m2["a"]))
    style(ax, "f / Hz", "s_Re / ohm", "Re channel dispersion")
    ax.legend(fontsize=8, frameon=False)

    ax = axs[1]
    ax.plot(f, s_im, "o", color="#378ADD", ms=3.2, label="s_Im")
    b_im = [c for c in det["forms"]["decom"]["im"]["cands"] if c["kind"] == "M1"][0]["b"]
    ax.axhline(b_im, color="#378ADD", lw=1.0, ls="--", alpha=0.7,
               label="M1 const b=%.3g" % b_im)
    ax.plot(f, 0.02 * absZ, color="#888780", lw=1.2, ls=":",
            label="perpoint 2%|Z|")
    style(ax, "f / Hz", "s_Im / ohm", "Im channel dispersion")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_sigma_f.png"), dpi=160)
    plt.close(fig)

    # ---- fig3 s vs |Z| 三档
    A3, coef = P.affine_align(Zs)
    Zbar3, s_re3, _ = P.disp(list(A3))
    absZ3 = np.abs(Zbar3)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    panels = ((absZ, s_re, det["forms"]["raw"]["re"],
               "raw: b=%.3g, a-proportional rejected" % _p(det["forms"]["raw"]["re"], "M1", "b")),
              (absZ2, s_re2, det["forms"]["decom"]["re"],
               "Rs-offset removed: proportional wins"),
              (absZ3, s_re3, det["forms"]["affine"]["re"],
               "complex affine removed: constant wins again"))
    for ax, (zz, ss, box, ttl) in zip(axs, panels):
        ax.plot(zz, ss, "o", color="#D85A30", ms=3.4, alpha=0.85)
        c1 = [c for c in box["cands"] if c["kind"] == "M1"][0]
        c2 = [c for c in box["cands"] if c["kind"] == "M2" and c["x"] == "|Z|"][0]
        xx = np.logspace(np.log10(zz.min()), np.log10(zz.max()), 100)
        ax.plot(xx, np.full_like(xx, c1["b"]), color="#378ADD", lw=1.6,
                label="M1 const  dAIC %+.1f" % c1["daic"])
        ax.plot(xx, c2["a"] * xx, color="#534AB7", lw=1.6,
                label="M2 a|Z|  dAIC %+.1f" % c2["daic"])
        style(ax, "|Z| / ohm", "s_Re / ohm", ttl)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    axs[2].set_xscale("linear")
    axs[2].set_xticks([0.5, 0.7, 0.9, 1.1])
    fig.suptitle("How much systematics you remove decides which noise model wins",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_sigma_Z.png"), dpi=160)
    plt.close(fig)

    # ---- fig4 a vs drift
    sw = [r for r in probe["sweep"] if r["ch"] == "re"]
    x = 100 * np.array([r["rs_cv"] for r in sw])
    y = 100 * np.array([r["rel_comp"] for r in sw])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(x, y, "o", color="#D85A30", ms=6)
    for r, xi, yi in zip(sw, x, y):
        if yi > 9 or xi < 16:
            ax.annotate(r["group"].split("/")[-1] + "mA", (xi, yi),
                        textcoords="offset points", xytext=(5, -10), fontsize=7.5)
    k, b0 = np.polyfit(x, y, 1)
    xx = np.linspace(0, x.max() * 1.05, 50)
    ax.plot(xx, k * xx + b0, color="#888780", lw=1.2, ls="--",
            label="linear fit, r=%.2f" % np.corrcoef(x, y)[0, 1])
    ax.axhline(2.0, color="#378ADD", lw=1.2, ls=":",
               label="perpoint 2% (current setting)")
    ax.set_xlabel("Rs coefficient of variation across repeats / %", fontsize=10)
    ax.set_ylabel("Re relative dispersion / %", fontsize=10)
    ax.set_title("Dispersion tracks sample drift, not measurement noise",
                 fontsize=11, fontweight="500", loc="left")
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.18, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_drift_artifact.png"), dpi=160)
    plt.close(fig)

    print("4 figures ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
