# -*- coding: utf-8 -*-
"""§18 结果图：合成谱 vs 真实谱的 `acf`（只读已有 JSON，不重跑拟合）。

左：按组分布的 acf（G/N/H 合成 + §14 真实谱未过/对照）。
右：三条「孪生真值」逐条对照 —— 同一条曲线、同一套代码，只有噪声/端部
    偏差不同，真实谱的 acf 与合成谱差多少。

用法：
    python evidence/nested_topology/m4_synth_plot.py
产物：out/fig_synth_vs_real.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib import font_manager                    # noqa: E402

SYN = os.path.join("out", "m4_synth_probe.json")
HOLDOUT = os.path.join("out", "m4_holdout2.json")
OUT = os.path.join("out", "fig_synth_vs_real.png")
T2 = {"T2a": "AZ31-8Li_50", "T2b": "AZ31-8Li_5", "T2c": "AZ31-4Li_0"}

_cn = any(x.name in ("Microsoft YaHei", "SimHei")
          for x in font_manager.fontManager.ttflist)
if _cn:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
L = (lambda a, b: a) if _cn else (lambda a, b: b)


def main() -> int:
    s = json.load(open(SYN, encoding="utf-8"))
    h = json.load(open(HOLDOUT, encoding="utf-8"))
    rows = [r for r in s["rows"] if "acf" in r]
    real = {r["spec"]: r for r in h["rows"] if "voigt_base" in r}
    fail = [real[k]["rows"][0]["acf"] for k in real
            if real[k]["state"] == "未过"]
    ctrl = [real[k]["rows"][0]["acf"] for k in real
            if real[k]["state"] == "过"]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    c_g, c_n, c_h, c_r = "#2e86c1", "#e67e22", "#c0392b", "#1f2d3d"

    # ① 按组分布
    a = ax[0]
    groups = [
        (L("真实·未过(10)", "real / failed"), fail, c_r, "o"),
        (L("真实·对照(4)", "real / control"), ctrl, "#7f8c8d", "o"),
        (L("合成 G 干净(12)", "synth G clean"), [r["acf"] for r in rows if r["group"] == "G"], c_g, "s"),
        (L("合成 N 噪声×2(3)", "synth N noise x2"), [r["acf"] for r in rows if r["group"] == "N"], c_n, "D"),
        (L("合成 H 端部偏差(3)", "synth H HF bias"), [r["acf"] for r in rows if r["group"] == "H"], c_h, "^"),
    ]
    for i, (name, vals, col, mk) in enumerate(groups):
        x = np.random.default_rng(100 + i).normal(i + 1, 0.075, len(vals))
        a.scatter(x, vals, s=48, marker=mk, color=col, alpha=0.85,
                  edgecolor="w", linewidth=0.6, zorder=3, label=name)
        a.plot([i + 0.75, i + 1.25], [np.median(vals)] * 2, "-",
               color=col, lw=2.2, zorder=2)
    a.axhline(0.50, color="#7f8c8d", ls="--", lw=1.3, zorder=1)
    a.text(0.52, 0.512, L("M-b 上限 0.50", "M-b limit 0.50"),
           color="#7f8c8d", fontsize=8.5)
    a.set_xticks(range(1, 6))
    a.set_xticklabels([g[0] for g in groups], rotation=18, ha="right", fontsize=8.5)
    a.set_ylabel(L("残差自相关 acf", "residual acf"))
    a.set_title(L("① 同一套判据：合成谱 vs 真实谱", "(1) same criterion: synth vs real"))
    a.set_ylim(0, 1.0)
    a.grid(alpha=0.25, axis="y")
    a.legend(fontsize=8, loc="upper left")

    # ② 孪生真值逐条对照
    a = ax[1]
    for xi, (tag, spec) in enumerate(T2.items(), start=1):
        rv = real[spec]["rows"][0]["acf"] if spec in real else None
        g = [r["acf"] for r in rows if r["group"] == "G" and r["truth"] == tag]
        n = [r["acf"] for r in rows if r["group"] == "N" and r["truth"] == tag]
        hh = [r["acf"] for r in rows if r["group"] == "H" and r["truth"] == tag]
        for j, (vals, col, mk) in enumerate(((g, c_g, "s"), (n, c_n, "D"),
                                             (hh, c_h, "^"))):
            if vals:
                a.scatter([xi + (j - 1) * 0.22] * len(vals), vals, s=44,
                          marker=mk, color=col, alpha=0.85,
                          edgecolor="w", linewidth=0.6, zorder=3)
        if rv is not None:
            a.scatter([xi], [rv], s=170, marker="*", color=c_r, zorder=4)
            a.annotate(f"{rv:.3f}", (xi, rv), textcoords="offset points",
                       xytext=(10, -3), fontsize=8.5, color=c_r)
        a.plot([xi - 0.36, xi + 0.36], [np.median(g)] * 2 if g else [0] * 2,
               "-", color=c_g, lw=2.0, zorder=2)
    a.axhline(0.50, color="#7f8c8d", ls="--", lw=1.3, zorder=1)
    a.set_xticks([1, 2, 3])
    a.set_xticklabels([f"{sp.replace('AZ31-', '')}\n({tag})" for tag, sp in T2.items()],
                      fontsize=9)
    a.set_ylabel(L("残差自相关 acf", "residual acf"))
    a.set_title(L("② 孪生谱：同一曲线只换噪声", "(2) twin spectra: only noise differs"))
    a.set_ylim(0, 1.0)
    a.grid(alpha=0.25, axis="y")
    from matplotlib.lines import Line2D
    a.legend(handles=[
        Line2D([], [], linestyle="", marker="*", color=c_r, markersize=12,
               label=L("真实谱", "real spectrum")),
        Line2D([], [], linestyle="", marker="s", color=c_g, markersize=7,
               label="G 2%"),
        Line2D([], [], linestyle="", marker="D", color=c_n, markersize=7,
               label="N 4%"),
        Line2D([], [], linestyle="", marker="^", color=c_h, markersize=7,
               label=L("H 端部偏差", "H HF bias"))],
        fontsize=8.5, loc="upper left")

    fig.suptitle(L("§18 合成谱判别：卡 acf 的是数据不是代码（exploratory）",
                   "Sec.18: acf failure comes from data, not code (exploratory)"),
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150)
    print(f"图已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
