"""§9.4.29.5 附图：Bode |Z| 上「外部工具 vs 我们两种噪声口径」。

目的：把「外部工具高频 Bode 更贴」这个观察量化成口径差异（R16/R17），
而不是拓扑差异。三条预测曲线都来自**已存在的拟合结果**，本脚本不重新采样：

  · 外部工具   evidence/nested_topology/m4_external_cmp.py::z_external（按截图参数）
  · 我们逐点2%σ out/m4_conv_2000.json seed0（standard 2000/1000×2, ta=0.98）
  · 我们绝对σ  out/m4_real_nested_az31.json 的 parallel/L=2（quick）

输出 out/fig_bode_hf_compare.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib import font_manager                 # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "evidence", "nested_topology"))

import bm_real as R                                 # noqa: E402
import bm_top as T                                  # noqa: E402
from m4_external_cmp import z_external              # noqa: E402

_CN = any("yahei" in x.name.lower() or "simhei" in x.name.lower()
          for x in font_manager.fontManager.ttflist)
if _CN:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
L = (lambda cn, en: cn) if _CN else (lambda cn, en: en)

SPEC = os.path.join("out", "_m6_samples", "z_csv", "AZ31-4Li_100.csv")
OUT = os.path.join("out", "fig_bode_hf_compare.png")

f, Z = R.load_spectrum(SPEC)
w = np.abs(Z)
Ze = z_external(f)

d = json.load(open(os.path.join("out", "m4_conv_2000.json"), encoding="utf-8"))
Zp = T.z_nested(f, d["rows"][0]["params"], branch=d["branch"],
                layers=int(d["layers"]))
dn = json.load(open(os.path.join("out", "m4_real_nested_az31.json"), encoding="utf-8"))
r = [x for x in dn["rows"] if x["label"] == "parallel/L=2"][0]
Za = T.z_nested(f, r["params"], branch="parallel", layers=2)

curves = (("外部工具（28 选 1）", Ze, "#d62728", "-", 2.0),
          ("我们 逐点 2% σ（standard）", Zp, "#2ca02c", "--", 2.1),
          ("我们 绝对 σ（旧图用的）", Za, "#888888", ":", 2.2))


def band_rms(Zm, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.sqrt(np.mean(((np.abs(Zm[m]) - w[m]) / w[m]) ** 2)) * 100)


fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2),
                       gridspec_kw=dict(width_ratios=[1.35, 1]))

a = ax[0]
a.loglog(f, w, "o", ms=5.5, mfc="none", mec="#1f77b4", mew=1.3,
         label=L("实测（50 点）", "data (50 pts)"))
for tag, Zm, c, ls, lw in curves:
    a.loglog(f, np.abs(Zm), ls, lw=lw, color=c, label=tag)
a.set_xlabel(L("频率 / Hz", "f / Hz"))
a.set_ylabel(L("|Z| (kOhm)", "|Z| (kOhm)"))
a.set_title(L("① 全谱 Bode |Z|：三条预测在中低频几乎重合",
              "(1) Full Bode |Z|: curves overlap at low/mid f"))
a.grid(alpha=0.25, which="both")
a.legend(fontsize=9, loc="lower left")

b = ax[1]
hi = f >= 1e3
b.semilogx(f, w, "o", ms=6, mfc="none", mec="#1f77b4", mew=1.4,
           label=L("实测", "data"))
for tag, Zm, c, ls, lw in curves:
    b.semilogx(f, np.abs(Zm), ls, lw=lw, color=c, marker=".", ms=4, label=tag)
b.set_xlim(1e3, 1.2e5)
b.set_ylim(35, 95)
b.set_xlabel(L("频率 / Hz", "f / Hz"))
b.set_ylabel(L("|Z| (Ohm)", "|Z| (Ohm)"))
b.set_title(L("② 高频放大：绝对 σ 档偏离，逐点 σ 档与外部重合",
              "(2) HF zoom: absolute sigma deviates, pointwise overlaps"))
b.grid(alpha=0.3, which="both")
b.legend(fontsize=8.5, loc="upper right")
b.annotate(f"{abs(Za[-1]):.1f}", (f[-1], abs(Za[-1])), color="#888888",
           fontsize=8.5, xytext=(-34, 7), textcoords="offset points")
b.annotate(f"{w[-1]:.1f}", (f[-1], w[-1]), color="#1f77b4", fontsize=8.5,
           xytext=(-34, 9), textcoords="offset points")
# 外部 45.4 与逐点 2% 45.8 在图上重合，合并成一个标注
b.annotate(f"{abs(Ze[-1]):.1f} / {abs(Zp[-1]):.1f}",
           (f[-1], abs(Zp[-1])), color="#2ca02c", fontsize=8.5,
           xytext=(-78, -14), textcoords="offset points")

vals = [band_rms(Zm, 1e4, 1e5) for _, Zm, _, _, _ in curves]
txt = (L("10k–100k Hz 的 |Z| 相对偏差 RMS：", "|Z| rel. RMS, 10k-100k Hz: ")
       + L("外部 ", "external ") + f"{vals[0]:.2f}%  |  "
       + L("我们逐点 2%σ ", "pointwise ") + f"{vals[1]:.2f}%  |  "
       + L("我们绝对σ ", "absolute ") + f"{vals[2]:.2f}%\n"
       + L("1e5 Hz 实测 51.12 Ω；低频 0.01–0.1 Hz 三家都在 12% 左右",
           "1e5 Hz data 51.12 Ohm; low-f band ~12% for all three"))
fig.text(0.5, 0.015, txt, ha="center", va="bottom", fontsize=9,
         bbox=dict(fc="white", ec="0.75", alpha=0.9, boxstyle="round,pad=0.4"))

fig.suptitle(L("AZ31-4Li_100 · 高频 Bode 差异来自噪声权重，不是拓扑（R16/R17）",
               "AZ31-4Li_100: HF Bode gap is a weighting choice, not topology"),
             fontsize=12)
fig.tight_layout(rect=(0, 0.13, 1, 0.95))
fig.savefig(OUT, dpi=160, facecolor="white")
print("saved", OUT)
for tag, Zm, _, _, _ in curves:
    print(f"  {tag:<30} 1k-10k {band_rms(Zm,1e3,1e4):6.2f}%   "
          f"10k-100k {band_rms(Zm,1e4,1e5):6.2f}%   "
          f"1e5端点 {(abs(Zm[-1])-w[-1])/w[-1]*100:+6.2f}%")
