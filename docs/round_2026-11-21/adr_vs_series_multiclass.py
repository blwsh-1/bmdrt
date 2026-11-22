"""adr 到底有没有用：跨类同尺对比（adr vs 串联式 = nested/L=1）。

口径
----
两边都 **6 个参数、都不带顶层 L**，只差吸附支路的接点：

    adr     Rs + [CPE ∥ Rct ∥ Z_ad]
    series  Rs + [CPE ∥ (R1 + Z_ad)]

同一 sigma（perpoint 2%）、同一优化器（DE 找盆地 + least_squares 抛光）、
同一起点策略。报单点最优 rms 与解处 Jacobian 条件数。

为什么不用 MCMC：这里只问"谁更能拟合"，单点最优就够了；MCMC 已在前一轮
交叉校验过（7.10/7.11、6.06/6.04）。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/adr_vs_series_multiclass.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import differential_evolution, least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import bm_real as R        # noqa: E402
import bm_top as T         # noqa: E402
import spectrum_metrics as SM  # noqa: E402

OUT = os.path.join(REPO, "out", "adr_multiclass")
PCT = 2.0

RANGE = dict(Rs=(1e-2, 1e5), R1=(1e-2, 1e6), Q1=(1e-2, 1e7), n1=(0.3, 1.0),
             Rct=(1e-2, 1e6), Q=(1e-2, 1e7), n=(0.3, 1.0),
             R_ad=(1e-2, 1e7), tau_ad=(1e-9, 1e3), L=(1e-12, 1e-2))

#: `--L` 开关：2E/3E 是整胞谱，带导线电感；AZ31 那批数据把 L 压到下界。
USE_L = "--L" in sys.argv

MODELS = {
    "adr": dict(keys=("Rs", "Q", "n", "Rct", "R_ad", "tau_ad", "L"),
                paren=("n",)),
    "series": dict(keys=("Rs", "R1", "Q1", "n1", "R_ad", "tau_ad", "L"),
                   paren=("n1",)),
}


def build(kind):
    keys = MODELS[kind]["keys"]
    paren = MODELS[kind]["paren"]
    if not USE_L:
        keys = tuple(k for k in keys if k != "L")

    def unpack(x):
        return {k: (float(v) if k in paren else float(np.exp(v)))
                for v, k in zip(x, keys)}

    def zf(f, p):
        return (T.z_adr(f, p, include_L=USE_L) if kind == "adr"
                else T.z_nested(f, p, branch="parallel", layers=1,
                                include_L=USE_L))

    def res(x, f, z, sg):
        d = (z - zf(f, unpack(x))) / sg
        return np.concatenate([d.real, d.imag])

    lo = [RANGE[k][0] if k in paren else float(np.log(RANGE[k][0])) for k in keys]
    hi = [RANGE[k][1] if k in paren else float(np.log(RANGE[k][1])) for k in keys]
    return keys, unpack, zf, res, lo, hi


def fit(kind, f, z, sg, seed=0):
    keys, unpack, zf, res, lo, hi = build(kind)
    de = differential_evolution(
        lambda x: float(np.sum(res(x, f, z, sg) ** 2)), list(zip(lo, hi)),
        seed=seed, tol=1e-6, maxiter=250, popsize=16, polish=True, init="sobol")
    q = least_squares(res, de.x, bounds=(lo, hi), x_scale="jac",
                      args=(f, z, sg), max_nfev=6000)
    p = unpack(q.x)
    sv = np.linalg.svd(q.jac, compute_uv=False)
    return dict(params={k: float(v) for k, v in p.items()},
                rms=float(SM.metrics(f, z, zf(f, p))["rms_rel"]),
                cond=float(sv[0] / sv[-1]) if sv[-1] > 0 else None)


def cases():
    out = []
    for p in sorted(glob.glob(os.path.join(REPO, "out", "_m6_samples", "z_csv",
                                           "AZ31-*.csv")))[::4]:
        out.append(("AZ31", p))
    for p in sorted(glob.glob(os.path.join(REPO, "EISmeasurements", "csv_2e",
                                           "2EMC1-1_0*.csv")))[::2]:
        out.append(("2E", p))
    for p in sorted(glob.glob(os.path.join(REPO, "EISmeasurements", "csv_3e",
                                           "3E-1_HER_exp1_0*.csv")))[::2]:
        out.append(("3E", p))
    return out


def main() -> int:
    limit = next((int(a) for a in sys.argv[1:] if a.isdigit()), 0)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for cls, path in cases():
        if limit and len(rows) >= limit:
            break
        try:
            f, z = R.load_spectrum(path)
        except Exception as e:
            print("skip", os.path.basename(path), repr(e)[:60])
            continue
        if f.size < 12:
            continue
        sg = SM.sigma_perpoint(z, PCT)[0]
        a = fit("adr", f, z, sg)
        s = fit("series", f, z, sg)
        win = "series" if s["rms"] < a["rms"] else "adr"
        rows.append(dict(cls=cls, spec=os.path.basename(path), n=int(f.size),
                         adr=a, series=s, win=win,
                         d_rms=float(s["rms"] - a["rms"])))
        print("%-5s %-26s n=%3d  adr %7.2f%% (cond %.2g)  series %7.2f%% "
              "(cond %.2g)  -> %s"
              % (cls, os.path.basename(path)[:26], f.size, a["rms"],
                 a["cond"] or -1, s["rms"], s["cond"] or -1, win))
    n_adr = sum(1 for r in rows if r["win"] == "adr")
    print("-" * 96)
    print("共 %d 条：adr 赢 %d 条，串联式赢 %d 条" % (len(rows), n_adr,
                                                len(rows) - n_adr))
    if n_adr:
        print("adr 赢的谱：", ", ".join(r["spec"] for r in rows if r["win"] == "adr"))
    with open(os.path.join(OUT, "cmp_L.json" if USE_L else "cmp.json"),
              "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(dict(pct=PCT, include_L=USE_L, rows=rows,
                       n_adr_win=n_adr), fh,
                  ensure_ascii=False, indent=1, allow_nan=False)
    print("->", os.path.join(OUT, "cmp_L.json" if USE_L else "cmp.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
