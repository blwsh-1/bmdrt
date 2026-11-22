"""真实谱对比：adr（并联吸附）vs 串联改法（nested/L=1），同一批数据同一 sigma。

问题
----
`B2_REPORT.md` 证明串联改法能恢复自己的真值（7/7），但两条谱在低频差 200σ
（`TOPOLOGY_AUDIT.md` §3.5）⇒ 它们不是同一个物理模型。所以"改到合格"能不能
落到真实数据上，必须看真实谱：串联式会不会把低频那一段拟合坏。

做法
----
AZ31-12Li_5（§14 留出集里 acf 最差的一条），同 sigma（perpoint 2%）、
同预算（standard 2000+1000）、同 seed，只换结构：

    adr          Rs + [CPE ∥ Rct ∥ Z_ad] + jωL
    series       Rs + [CPE ∥ (R1  + Z_ad)] + jωL      （nested/L=1）

报 `rms_rel`、`acf`（与 §14 同口径：`max(|acf(Re)|,|acf(Im)|)`）、`R̂`、`div`。
判据只看"串联式是否更差"：若 rms 与 acf 都不劣于 adr，则换结构不付拟合代价；
若明显更差，说明这批数据的低频确实是"泄流型"，只能走固定 Rct 那条路。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/adr_vs_series_az31.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import bm_experiments as E      # noqa: E402
import bm_real as R             # noqa: E402
import bm_select as SEL         # noqa: E402
import circuit_runner as RU     # noqa: E402
import spectrum_metrics as SM   # noqa: E402

OUT = os.path.join(REPO, "out", "adr_series_az31")
SRC = os.path.join(REPO, "out", "_m6_samples", "z_csv", "AZ31-12Li_5.csv")
PCT = 2.0
BUDGET = "standard"
SEED = 0


def resid_metrics(f, z, zf):
    """与 §14 同口径：相对残差、acf、三个频段的 rms。"""
    d = (z - zf) / np.abs(z)
    bands = []
    for lo, hi in ((0, 10), (10, 1e4), (1e4, 1e7)):
        m = (f >= lo) & (f < hi)
        if np.any(m):
            bands.append(float(100 * np.sqrt(np.mean(
                np.concatenate([d.real[m], d.imag[m]]) ** 2))))
        else:
            bands.append(None)
    acf = float(max(abs(E.acf_max(d.real)), abs(E.acf_max(d.imag))))
    return dict(rms_rel=float(SM.metrics(f, z, zf)["rms_rel"]),
                bands=bands, acf=acf,
                acf_re=float(abs(E.acf_max(d.real))),
                acf_im=float(abs(E.acf_max(d.imag))))


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    f, z = R.load_spectrum(SRC)
    sg = SM.sigma_perpoint(z, PCT)
    kw = dict(RU.BUDGETS[BUDGET])
    rows = []

    for tag, topo, call in (
            ("adr", "adr", lambda: SEL.fit_adr(f, z, seed=SEED, sigma_floor=0.0, kw=kw,
                                        sigma=sg, sigma_from_data=True,
                                        include_L=True)),
            ("series_L1", "nested", lambda: SEL.fit_nested(f, z, 1, branch="parallel",
                                                 seed=SEED, sigma_floor=0.0,
                                                 kw=kw, sigma=sg,
                                                 sigma_from_data=True,
                                                 include_L=True)),
            ("nested_L2", "nested", lambda: SEL.fit_nested(f, z, 2, branch="parallel",
                                                 seed=SEED, sigma_floor=0.0,
                                                 kw=kw, sigma=sg,
                                                 sigma_from_data=True,
                                                 include_L=True))):
        cfg = (dict(include_L=True, diffusion=None) if tag == "adr"
               else dict(layers=int(tag[-1]), branch="parallel", include_L=True,
                         diffusion=None))
        c = call()
        raw = c["raw"]
        zf = RU._pointwise_median_z(raw, topo, cfg, f)
        m = resid_metrics(f, z, zf)
        div = int(raw.get("divergences", raw.get("ndiv")) or 0)
        rows.append(dict(tag=tag, n=int(f.size), pct=PCT, budget=BUDGET, seed=SEED,
                         rhat=float(raw["rhat_max"]), div=div,
                         **m,
                         z_fit=dict(re=[float(v) for v in zf.real],
                                    im=[float(v) for v in zf.imag])))
        print("%-10s rms=%7.2f%%  acf=%.3f (re %.3f / im %.3f)  bands=%s  rhat=%.3f div=%d"
              % (tag, m["rms_rel"], m["acf"], m["acf_re"], m["acf_im"],
                 ["%.1f" % b if b is not None else "-" for b in m["bands"]],
                 rows[-1]["rhat"], div))

    a, s = rows[0], rows[1]
    res = dict(src=SRC, rows=rows,
               verdict=dict(d_rms=s["rms_rel"] - a["rms_rel"],
                            d_acf=s["acf"] - a["acf"],
                            series_better=bool(s["rms_rel"] <= a["rms_rel"]
                                               and s["acf"] <= a["acf"])))
    with open(os.path.join(OUT, "cmp.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, allow_nan=False)
    print("-" * 74)
    print("串联 - 并联：d_rms=%+.2f pp, d_acf=%+.3f ⇒ %s"
          % (res["verdict"]["d_rms"], res["verdict"]["d_acf"],
             "串联不劣" if res["verdict"]["series_better"] else "串联更差"))
    print("->", os.path.join(OUT, "cmp.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
