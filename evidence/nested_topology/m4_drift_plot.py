# -*- coding: utf-8 -*-
"""§19 结果图：什么形态/幅度的系统性偏差能把 acf 推进真实未过谱区间。

只读已有 JSON，不重跑拟合。产物 `out/fig_drift_scan.png`。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import font_manager                   # noqa: E402

SCAN = os.path.join("out", "m4_drift_scan.json")
SYN = os.path.join("out", "m4_synth_probe.json")
HOLD = os.path.join("out", "m4_holdout2.json")
OUT = os.path.join("out", "fig_drift_scan.png")

_cn = any(x.name in ("Microsoft YaHei", "SimHei")
          for x in font_manager.fontManager.ttflist)
if _cn:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
L = (lambda a, b: a) if _cn else (lambda a, b: b)
COL = {"S1": "#2e86c1", "S2": "#e67e22", "S3": "#8e44ad", "S4": "#c0392b"}


def main() -> int:
    d = json.load(open(SCAN, encoding="utf-8"))
    rows = [r for r in d["rows"] if "acf" in r]
    syn = [r for r in json.load(open(SYN, encoding="utf-8"))["rows"]
           if r.get("group") == "G" and "acf" in r]
    real = [r for r in json.load(open(HOLD, encoding="utf-8"))["rows"]
            if "rows" in r and r["rows"]]

    fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.4))

    # ① acf vs 注入幅度
    a = ax[0]
    for shape in ("S1", "S2", "S3"):
        rr = sorted([r for r in rows if r["shape"] == shape],
                    key=lambda r: r["amp"])
        a.plot([r["amp"] for r in rr], [r["acf"] for r in rr], "-o",
               color=COL[shape], lw=2, ms=7, label=shape)
    rr = sorted([r for r in rows if r["shape"] == "S4"],
                key=lambda r: r["amp"])
    if rr:
        a.plot([r["amp"] for r in rr], [r["acf"] for r in rr], "--*",
               color=COL["S4"], lw=2, ms=12, label="S4 参数漂移")
    a.axhline(0.50, color="#7f8c8d", ls="--", lw=1.3)
    a.axhspan(0.505, 0.896, color="#c0392b", alpha=0.08, zorder=0)
    a.text(0.005, 0.86, L("真实未过组 0.505–0.896", "real failed 0.505-0.896"),
           fontsize=8.5, color="#c0392b")
    a.plot([0], [0.363], marker="o", ms=9, color="#1f2d3d", zorder=5)
    a.annotate(L("干净基线 0.363", "clean 0.363"), (0, 0.363),
               textcoords="offset points", xytext=(8, -12), fontsize=8.5)
    a.set_xlabel(L("注入幅度 a / 参数漂移 d", "injection amplitude a / drift d"))
    a.set_ylabel(L("残差自相关 acf", "residual acf"))
    a.set_title(L("① 偏差要多宽、多大才推得动 acf", "(1) how much bias is needed"))
    a.set_ylim(0, 1.0)
    a.grid(alpha=0.25)
    a.legend(fontsize=9)

    # ② rms vs acf：四象限
    a = ax[1]
    a.axvline(10.0, color="#7f8c8d", ls=":", lw=1.2)
    a.axhline(0.50, color="#7f8c8d", ls=":", lw=1.2)
    a.text(10.3, 0.96, L("M-a 上限 10%", "M-a limit 10%"), fontsize=8,
           color="#7f8c8d", rotation=90, va="top")
    a.text(0.02, 0.515, L("M-b 上限 0.50", "M-b limit 0.50"), fontsize=8,
           color="#7f8c8d")
    a.scatter([r["rms_rel"] for r in syn], [r["acf"] for r in syn],
              s=40, marker="s", color="#2e86c1", alpha=0.7,
              edgecolor="w", linewidth=0.5, label=L("§18 干净合成 (12)", "clean synth (12)"))
    a.scatter([r["rows"][0]["rms_rel"] for r in real],
              [r["rows"][0]["acf"] for r in real],
              s=48, marker="o", color="#1f2d3d", alpha=0.8,
              edgecolor="w", linewidth=0.5, label=L("§14 真实谱 (14)", "real (14)"))
    for shape in ("S1", "S2", "S3", "S4"):
        rr = [r for r in rows if r["shape"] == shape]
        if rr:
            a.scatter([r["rms_rel"] for r in rr], [r["acf"] for r in rr],
                      s=58, marker="^" if shape != "S4" else "D",
                      color=COL[shape], alpha=0.85, edgecolor="w",
                      linewidth=0.5, label=f"§19 {shape}")
    a.set_xlabel(L("rms_rel / %（M-a）", "rms_rel / % (M-a)"))
    a.set_ylabel(L("残差自相关 acf（M-b）", "residual acf (M-b)"))
    a.set_title(L("② 「精度够、结构不够」能不能被复现",
                  "(2) reproducing 'good fit, structured residual'"))
    a.set_xlim(0, max(35, max(r["rms_rel"] for r in rows) * 1.15))
    a.set_ylim(0, 1.0)
    a.grid(alpha=0.25)
    a.legend(fontsize=8, loc="lower right")

    fig.suptitle(L("§19 全频段偏差扫描：真实谱 acf 需要跨频段系统性结构（exploratory）",
                   "Sec.19: real acf needs broadband systematic structure (exploratory)"),
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150)
    print(f"图已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
