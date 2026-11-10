#!/usr/bin/env python
"""出一张对照图：同一条真实谱上 nested（最优） vs voigt（bmdrt.bat 默认族）。

三个面板：
  ① Nyquist（`Z'` vs `−Z''`）—— 看低频感抗尾在第四象限，以及两族谁跟得住；
  ② Bode 幅值（log-log）；
  ③ 相对残差 vs 频率（半对数）—— 直接看误差集中在哪一段。

跑法：
    python evidence/nested_topology/m4_plot_fit.py
产物：out/fig_az31_4Li_100.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "evidence", "nested_topology")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib import font_manager                    # noqa: E402

import bm_core as C                                   # noqa: E402
import bm_experiments as E                             # noqa: E402
import bm_real as R                                    # noqa: E402
import bm_select as S                                 # noqa: E402
import bm_top as T                                    # noqa: E402

SPEC = "out/_m6_samples/z_csv/AZ31-4Li_100.csv"
OUT = "out/fig_az31_4Li_100.png"

# 中文字体：有就用中文，没有就退回英文（不要在图上出现豆腐块）
_CN = any(f.name in ("Microsoft YaHei", "SimHei")
          for f in font_manager.fontManager.ttflist)
if _CN:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
L = (lambda cn, en: cn) if _CN else (lambda cn, en: en)


def med_params(s: dict, keys) -> dict:
    return {k: float(np.median(np.asarray(s[k], float))) for k in keys if k in s}


def rel_rms(Zm, Z):
    d = (Zm - Z) / np.abs(Z)
    return float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2)))


def main():
    f, Z = R.load_spectrum(SPEC)
    kw = dict(S.BUDGETS["quick"])
    zmax = float(np.abs(Z).max())
    print(f"谱 {os.path.basename(SPEC)}：{f.size} 点，|Z|max={zmax:.0f} Ω")

    c = S.fit_nested(f, Z, 2, branch="parallel", kw=kw)
    pn = med_params(c["raw"]["s"], T.sampled_keys("parallel", 2))
    Zn = T.z_nested(f, pn, branch="parallel", layers=2)
    cn = S.fit_voigt(f, Z, 2, kw=kw)
    sv = cn["raw"]["s"]
    pv = dict(R_inf=float(np.median(sv["R_inf"])), W=float(np.median(sv["W"])),
              pi=np.median(sv["pi"], axis=0), tau=np.median(sv["tau"], axis=0),
              phi=np.median(sv["phi"], axis=0), zeta=np.median(sv["zeta"], axis=0))
    Zv = C.model_z(f, pv, include_L=False)

    stats = []
    for tag, Zm in (("nested parallel/L=2", Zn), ("voigt K=2（bat 默认族）", Zv)):
        r = Z - Zm
        stats.append(dict(
            tag=tag, rms_rel=rel_rms(Zm, Z),
            rms_re=float(np.sqrt(np.mean(r.real ** 2))),
            rms_im=float(np.sqrt(np.mean(r.imag ** 2))),
            acf=float(max(E.acf_max(r.real), E.acf_max(r.imag)))))
    for s in stats:
        print(f"  {s['tag']:<24} rms_rel={s['rms_rel']:6.2%}  "
              f"rms={s['rms_re']:5.1f}/{s['rms_im']:5.1f} Ω  acf={s['acf']:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
    c_data, c_n, c_v = "#1f2d3d", "#c0392b", "#2e86c1"

    # ① Nyquist
    a = ax[0]
    a.plot(Z.real, -Z.imag, "o", ms=5.5, mfc="none", mec=c_data, mew=1.3,
           label=L("实测", "data"))
    a.plot(Zn.real, -Zn.imag, "-", lw=1.9, color=c_n,
           label=f"nested  L=2  ({stats[0]['rms_rel']:.0%})")
    a.plot(Zv.real, -Zv.imag, "--", lw=1.9, color=c_v,
           label=f"voigt  K=2  ({stats[1]['rms_rel']:.0%})")
    a.axhline(0, color="0.6", lw=0.8, zorder=0)
    # 标出几个频点，避免读者分不清环的走向（低频在右下、高频在左上）
    for fr in (0.01, 0.1, 1.0, 100.0, 1e5):
        i = int(np.argmin(np.abs(np.log10(f) - np.log10(fr))))
        a.annotate(f"{fr:g} Hz", (Z.real[i], -Z.imag[i]), fontsize=7.5,
                   color="0.25", textcoords="offset points", xytext=(6, 6))
    a.text(0.02, 0.02,
           f"nested: rms {stats[0]['rms_re']:.0f}/{stats[0]['rms_im']:.0f} Ω, "
           f"acf {stats[0]['acf']:.2f}\n"
           f"voigt : rms {stats[1]['rms_re']:.0f}/{stats[1]['rms_im']:.0f} Ω, "
           f"acf {stats[1]['acf']:.2f}",
           transform=a.transAxes, fontsize=8, family="monospace",
           va="bottom", ha="left",
           bbox=dict(fc="w", ec="0.7", alpha=0.85, boxstyle="round,pad=0.35"))
    a.set_xlabel(L("Z′ / Ω", "Z' / Ohm"))
    a.set_ylabel(L("−Z″ / Ω", "-Z'' / Ohm"))
    a.set_title(L("① Nyquist：蓝色（voigt）缺失整个低频环",
                  "(1) Nyquist: voigt misses the whole LF loop"))
    a.legend(loc="upper left", fontsize=9)
    a.grid(alpha=0.25)

    # ② Bode |Z|
    a = ax[1]
    a.loglog(f, np.abs(Z), "o", ms=5.5, mfc="none", mec=c_data, mew=1.3, label=L("实测", "data"))
    a.loglog(f, np.abs(Zn), "-", lw=1.9, color=c_n, label="nested")
    a.loglog(f, np.abs(Zv), "--", lw=1.9, color=c_v, label="voigt")
    a.set_xlabel(L("频率 / Hz", "f / Hz"))
    a.set_ylabel(L("|Z| / Ω", "|Z| / Ohm"))
    a.set_title(L("② Bode 幅值", "(2) Bode magnitude"))
    a.legend(fontsize=9)
    a.grid(alpha=0.25, which="both")

    # ③ 相对残差（语义：|ΔZ|/|Z|）
    a = ax[2]
    a.semilogx(f, np.abs(Z - Zn) / np.abs(Z), "-o", ms=3.5, lw=1.5, color=c_n,
               label=f"nested  rms {stats[0]['rms_rel']:.1%}")
    a.semilogx(f, np.abs(Z - Zv) / np.abs(Z), "--o", ms=3.5, lw=1.5, color=c_v,
               label=f"voigt   rms {stats[1]['rms_rel']:.1%}")
    a.axhline(0.10, color="#7f8c8d", ls=":", lw=1.2)
    a.text(f.min() * 1.15, 0.108, L("验收线 10%", "acceptance 10%"),
           color="#7f8c8d", fontsize=8.5)
    a.set_xlabel(L("频率 / Hz", "f / Hz"))
    a.set_ylabel(L("|ΔZ| / |Z|", "|dZ| / |Z|"))
    a.set_yscale("log")
    a.set_title(L("③ 相对残差：低频两端最大",
                  "(3) relative residual: worst at both ends"))
    a.legend(fontsize=9)
    a.grid(alpha=0.25, which="both")

    fig.suptitle(
        L("AZ31-4Li_100：nested（嵌套并联 R∥L） vs voigt（bmdrt.bat 默认族）",
          "AZ31-4Li_100: nested vs voigt (bmdrt.bat default)"), fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=150)
    print(f"图已写入 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
