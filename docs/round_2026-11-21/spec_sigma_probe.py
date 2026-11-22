"""验证：合成谱用**真值 σ** 固定后，后验是否还炸（只读）。

背景
----
`ecm_05687` 在 `free` 口径下 R̂=176.7、`触顶=91%`。已查明原因是 σ 先验
（`0.01·zmax=1.3255 Ω`）比真值噪声（`2.37e-5 Ω`）大 **4.75 dec**，似然被抹平。
这条谱的真值 σ 是**已知的**（构造时就给定），所以根本不该去"猜"。

本脚本对同一条谱、同一个候选跑三种噪声口径：

  · free      —— σ 采样，先验中心 0.01·zmax（= 21 条与复核用的口径）
  · perpoint  —— σ_j = 2%·|Z_j|（固定）
  · true      —— σ = 真值 (sigma_re, sigma_im)，固定

判据（预注册）：若 `true` 档 R̂<=1.05、div=0、且 `L` 恢复接近 24.9 nH，
则说明"合成基准必须用真值 σ"，问题不在模型表达力。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/spec_sigma_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import bm_real as R  # noqa: E402
import bm_select as SEL  # noqa: E402
import circuit_runner as RU  # noqa: E402
import spectrum_metrics as SM  # noqa: E402

CSV = os.path.join("out", "ecm_test", "csv", "ecm_05687_L_R_RCPE.csv")
TRUTH = os.path.join("out", "ecm_test", "res", "ecm_05687_L_R_RCPE",
                     "real_ecm_05687_L_R_RCPE.json")
OUT = os.path.join("out", "noise_scale_probe", "spec_sigma_probe.json")
CFG = dict(layers=2, branch="parallel", include_L=True, diffusion=None)


def main() -> int:
    tj = json.load(open(TRUTH, encoding="utf-8"))
    f, Z = R.load_spectrum(CSV)
    kw = dict(RU.BUDGETS["standard"])
    L_true = float(tj["L_uH"]) * 1e-6
    s_re, s_im = float(tj["sigma_re"]), float(tj["sigma_im"])
    print("真值: R_inf=%.4f  L=%.4f nH  sigma=(%.3g, %.3g)"
          % (tj["R_inf"], tj["L_uH"], s_re, s_im))
    print("free 先验中心 0.01*%.3f = %.4f  ⇒ 比真值 σ 大 %.1f 倍\n"
          % (tj["zmax_obs"], 0.01 * tj["zmax_obs"],
             0.01 * tj["zmax_obs"] / s_re))

    cases = [("free", None),
             ("perpoint", SM.sigma_perpoint(Z, SM.SIGMA_PCT)),
             ("true", (s_re, s_im))]
    rows = []
    print("%-9s %10s %6s %8s %10s %10s %8s" % (
        "noise", "rhat", "div", "rms%", "L/nH", "L偏真值", "secs"))
    print("-" * 68)
    for tag, sig in cases:
        t0 = time.time()
        c = SEL.fit_nested(f, Z, 2, branch=CFG["branch"], seed=0, kw=kw,
                           sigma=sig, include_L=True)
        raw = c["raw"]
        p = RU._median_params_nested(raw, CFG["branch"], 2, include_L=True)
        Zm = RU._pointwise_median_z(raw, "nested", CFG, f)
        m = RU.metrics(f, Z, Zm)
        L_hat = float(p.get("L", np.nan))
        row = dict(noise=tag, rhat=float(raw["rhat_max"]),
                   div=int(raw["divergences"] or 0),
                   rms=float(m["rms_rel"]), L_hat_nH=L_hat * 1e9,
                   L_bias=float(L_hat / L_true - 1.0) if L_true else None,
                   seconds=time.time() - t0,
                   params={k: float(v) for k, v in p.items()})
        rows.append(row)
        print("%-9s %10.3f %6d %8.2f %10.2f %9.1f%% %8.1f" % (
            tag, row["rhat"], row["div"], row["rms"], row["L_hat_nH"],
            100 * row["L_bias"], row["seconds"]))

    print("-" * 68)
    tr = [r for r in rows if r["noise"] == "true"][0]
    ok = tr["rhat"] <= 1.05 and tr["div"] == 0
    print("true 档 R̂<=1.05 且 div=0 : %s" % ok)
    print("true 档 L 恢复偏差 %.1f%%（真值 %.3f nH）" % (
        100 * tr["L_bias"], L_true * 1e9))
    print("判据成立 ⇒ 合成基准必须用真值 σ；如果不成立，说明还有别的不可辨识来源。")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(dict(csv=CSV, truth_path=TRUTH, cfg=CFG,
                       truth=dict(R_inf=tj["R_inf"], L_uH=tj["L_uH"],
                                  sigma_re=s_re, sigma_im=s_im),
                       free_prior_center=0.01 * tj["zmax_obs"],
                       rows=rows), fh, ensure_ascii=False, indent=1,
                  allow_nan=False)
    print("->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
