"""验证：把合成谱的噪声加到**仪器量级（2%）**后，参数能否正常恢复（只读）。

动机
----
`ecm_05687` 的真值噪声是 `sigma=2.37e-5 Ω`，相对 |Z|≈132 Ω 只有 **1.8e-7**。
这个"零噪声"让三种 σ 口径全部失效：
  · `free`      σ 大 5 dec ⇒ 似然抹平 ⇒ R̂ 170（假收敛）
  · `true`      σ 恰为真值 ⇒ 似然过尖 ⇒ NUTS 崩（R̂ 1.3e10）
  · `perpoint`  σ=2%·|Z| ⇒ 采样健康但参数偏 583%

所以问题可能不在"用哪个 σ"，而在**基准的噪声水平本身与 MCMC 不兼容**。
本脚本在内存里给同一条谱加 2%·|Z| 的复噪声（不落盘、不改原文件），
再用与之匹配的 perpoint σ 拟合，看 L 能否恢复。

判据：L 恢复偏差 <10% 且 R̂<=1.05 且 div=0 ⇒ "噪声加到仪器量级"是正确做法。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/synthetic_noise_design.py
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
OUT = os.path.join("out", "noise_scale_probe", "synthetic_noise_design.json")
CFG = dict(layers=2, branch="parallel", include_L=True, diffusion=None)


def main() -> int:
    tj = json.load(open(TRUTH, encoding="utf-8"))
    f, Z = R.load_spectrum(CSV)
    kw = dict(RU.BUDGETS["standard"])
    L_true = float(tj["L_uH"]) * 1e-6
    rel_inherent = float(tj["sigma_re"]) / float(tj["zmax_obs"])
    print("原谱噪声水平: sigma_re/zmax = %.2e（%.1f 个 ppm 的十分之一量级）"
          % (rel_inherent, rel_inherent * 1e6))
    print("真值 L = %.3f nH\n" % (L_true * 1e9))

    rows = []
    print("%-18s %10s %6s %8s %10s %10s %8s" % (
        "case", "rhat", "div", "rms%", "L/nH", "L偏真值", "secs"))
    print("-" * 76)
    for pct in (0.5, 1.0, 2.0):
        rng = np.random.default_rng(20261121)
        sig_abs = (pct / 100.0) * np.abs(Z)
        eps = (rng.standard_normal(Z.size) + 1j * rng.standard_normal(Z.size))
        Zi = Z + sig_abs * eps / np.sqrt(2.0)      # 实虚各 pct 标准差
        t0 = time.time()
        c = SEL.fit_nested(f, Zi, 2, branch=CFG["branch"], seed=0, kw=kw,
                           sigma=SM.sigma_perpoint(Zi, pct),
                           include_L=True)
        raw = c["raw"]
        p = RU._median_params_nested(raw, CFG["branch"], 2, include_L=True)
        Zm = RU._pointwise_median_z(raw, "nested", CFG, f)
        m = RU.metrics(f, Zi, Zm)
        L_hat = float(p.get("L", np.nan))
        row = dict(noise_pct=pct, rhat=float(raw["rhat_max"]),
                   div=int(raw["divergences"] or 0),
                   rms=float(m["rms_rel"]), L_hat_nH=L_hat * 1e9,
                   L_bias=float(L_hat / L_true - 1.0),
                   seconds=time.time() - t0)
        rows.append(row)
        print("%-18s %10.3f %6d %8.2f %10.2f %9.1f%% %8.1f" % (
            "noise %.1f%%" % pct, row["rhat"], row["div"], row["rms"],
            row["L_hat_nH"], 100 * row["L_bias"], row["seconds"]))

    print("-" * 76)
    best = [r for r in rows if r["rhat"] <= 1.05 and r["div"] == 0
            and abs(r["L_bias"]) < 0.10]
    print("满足 R̂<=1.05 / div=0 / |L偏|<10%% 的噪声档：%s" % (
        [r["noise_pct"] for r in best] or "无"))

    # ---- 族对照：真值是 L_R_RCPE（voigt 型），nested 装不下 ----
    print("\n族对照（2% 噪声，同一份加噪数据）")
    print("%-18s %10s %6s %8s %10s %10s" % ("candidate", "rhat", "div", "rms%",
                                            "L/nH", "L偏真值"))
    print("-" * 68)
    rng = np.random.default_rng(20261121)
    pct = 2.0
    sig_abs = (pct / 100.0) * np.abs(Z)
    eps = (rng.standard_normal(Z.size) + 1j * rng.standard_normal(Z.size))
    Zi = Z + sig_abs * eps / np.sqrt(2.0)
    fam = [("nested/L=2", "nested", 2), ("voigt/K=1", "voigt", 1),
           ("voigt/K=2", "voigt", 2)]
    for tag, topo, k in fam:
        t0 = time.time()
        if topo == "voigt":
            c = SEL.fit_voigt(f, Zi, k, seed=0, include_L=True,
                              sigma=SM.sigma_perpoint(Zi, pct), kw=kw)
            raw = c["raw"]
            p = RU._median_params_voigt(raw)
        else:
            c = SEL.fit_nested(f, Zi, k, branch=CFG["branch"], seed=0, kw=kw,
                               sigma=SM.sigma_perpoint(Zi, pct), include_L=True)
            raw = c["raw"]
            p = RU._median_params_nested(raw, CFG["branch"], k, include_L=True)
        L_hat = float(p.get("L", np.nan))
        rows.append(dict(family=tag, noise_pct=pct, rhat=float(raw["rhat_max"]),
                         div=int((raw.get("divergences") or raw.get("ndiv") or 0)),
                         L_hat_nH=L_hat * 1e9,
                         L_bias=float(L_hat / L_true - 1.0)))
        print("%-18s %10.3f %6d %8s %10.2f %9.1f%%" % (
            tag, rows[-1]["rhat"], rows[-1]["div"], "—", L_hat * 1e9,
            100 * rows[-1]["L_bias"]))

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(dict(csv=CSV, truth_L_nH=L_true * 1e9,
                       inherent_rel_noise=rel_inherent, rows=rows), fh,
                  ensure_ascii=False, indent=1, allow_nan=False)
    print("->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
