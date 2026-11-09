# -*- coding: utf-8 -*-
"""把新判据套到 `examples/` 的真实文件上，与 `bm_kk.diffusion_tail_candidate`
（现行 L1）逐条对照。

已知真值（`DIFFUSION_PLAN.md` §9 表 + `WS_DECISION_PROJECT.md` §4）：
  example_caseE.csv         R∞=10 + ZARC(30,300Hz,.85) + **Ws(A_W=2)**, σ=0.2
  example_caseE_warburg.csv R∞=10 + ZARC(30,300Hz,.85) + **Ws(A_W=2)**, σ=0.2
  example_caseF_wo.csv      R∞=10 + ZARC + **Wo(R_W=30, τ_D=0.3)**, σ=0.835
  example_caseF.csv         （拓扑见文件生成脚本；此处只做形态登记）
  example_caseG_wo_only.csv R∞=10 + **Wo(R_W=30, τ_D=0.3)**, σ=0.803
"""
import sys
import numpy as np

sys.path.insert(0, r"D:/eis训练数据/bmdrt")
sys.path.insert(0, r"D:/eis训练数据/bmdrt/evidence/ws_crit")

import bm_kk
import bm_wscrit as W

TRUTH = {
    "example_caseE.csv": "真含 Ws(A_W=2)",
    "example_caseE_warburg.csv": "真含 Ws(A_W=2)",
    "example_caseF_wo.csv": "真含 Wo(τ_D=0.3, f_D=0.53Hz 在窗内)",
    "example_caseF.csv": "（未登记）",
    "example_caseG_wo_only.csv": "纯 Wo(τ_D=0.3, f_D=0.53Hz 在窗内)",
}


def load(fn):
    d = np.genfromtxt(r"D:/eis训练数据/bmdrt/examples/" + fn, delimiter=",", names=True)
    return d[d.dtype.names[0]], d[d.dtype.names[1]] + 1j * d[d.dtype.names[2]]


print(f"{'file':28s} {'旧:amp':>6s} {'旧:cand':>7s} | {'新:slope_im':>11s} "
      f"{'band':>5s} {'term':>11s} {'family':>14s} {'verdict':>11s}")
for fn, truth in TRUTH.items():
    f, Z = load(fn)
    old = bm_kk.diffusion_tail_candidate(f, Z)
    d = W.ws_diagnostics(f, Z)
    print(f"{fn:28s} {str(old['amp_ok']):>6s} {str(old['is_candidate']):>7s} | "
          f"{d['slope_im']:11.2f} {d['band_dec']:5.2f} {d['term']:>11s} "
          f"{d['family']:>14s} {d['verdict']:>11s}   {truth}")
    for w in d["why"]:
        print(f"    · {w}")
