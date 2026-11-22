"""诊断：腐蚀谱高频段崩的真正原因，以及 R_inf 定标修法的效果。

已查明（同一条谱、同一条曲线）：
  `SM.metrics` 的 `rms_rel` = **逐点相对**口径  → 110.392%
  全局归一化 `sqrt(mean|Δ|²)/sqrt(mean|Z|²)`    →   1.964%
所以 110% 不是度量假象：高频端 |Z|≈1.4e3 处 `R_inf` 被拟到 9.2e3，
相对误差 ~550%，低频端 2.5e6 把它稀释掉了，全局看才 2%。

本脚本量化两件事：
1. `R_inf` 真值 / 先验中心的偏离（σ 数）；
2. 用高频端 |Z(f_max)| 给 `R_inf` 定标后，同预算下 `rms_rel` 能降到多少。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import bm_core as B
import bm_real as BR
import bm_select as SEL
import circuit_runner as RU
import spectrum_metrics as SM

P = ("D:/eis训练数据/测试数据/腐蚀/Uniform_Generated data and Nyquist plots/"
     "Generated data and Nyquist plots/"
     "R1-[R2,P3]-[1000.0, 2512134.2009090395, 9012389.983760463, "
     "0.7761076465819079].csv")

f, Z = BR.load_spectrum(P)
KW = dict(RU.BUDGETS["quick"])
rs = float(np.abs(Z[np.argmax(f)]))
zmax = float(np.abs(Z).max())


def run(tag, **kw_extra):
    o = BR.fit_one(f, Z, 1, False, "1.0", seed=0, kw=dict(KW), **kw_extra)
    s = o["s"]
    p = dict(R_inf=float(np.median(s["R_inf"])), W=float(np.median(s["W"])),
             pi=np.median(s["pi"], axis=0), tau=np.median(s["tau"], axis=0),
             phi=np.median(s["phi"], axis=0), zeta=np.median(s["zeta"], axis=0))
    Zc = B.model_z(f, p, include_L=False)
    m = SM.metrics(f, Z, Zc)
    g = 100 * np.sqrt(np.mean(np.abs(Zc - Z) ** 2)) / np.sqrt(
        np.mean(np.abs(Z) ** 2))
    b = m.get("bands") or []
    btxt = " ".join("—" if x is None else f"{x:.1f}" for x in b)
    print(f"{tag:30s} rms_rel={m['rms_rel']:8.2f}%  全局={g:6.2f}%  "
          f"R_inf={p['R_inf']:9.3g}  bands=[{btxt}]  R̂={o['rhat_max']:.3f}")


print(f"高频端 |Z(f_max)| = {rs:.4g} Ω；真值 R_s=1000 Ω；"
      f"zmax = {zmax:.4g} Ω")
print(f"R_inf 先验中心 0.2·zmax = {0.2 * zmax:.4g} Ω  "
      f"⇒ 偏离真值 {0.2 * zmax / rs:.0f} 倍\n")
run("默认（0.2·zmax）")
run("zmax_override=5·|Z(fmax)|", zmax_override=5 * rs)
run("zmax_override=2·|Z(fmax)|", zmax_override=2 * rs)
run("zmax_override=10·|Z(fmax)|", zmax_override=10 * rs)
