"""`adr_fixRct` 的数据预验：固定 Rct 是"让拟合变好"还是"把不确定性换名字"。

为什么先做这一步
----------------
`TOPOLOGY_AUDIT.md` §3.6 说固定 `Rct` 后 cond 从 2.7e9 降到 17.8（合成真值点）。
但那是**正演局部量**，而且合成真值点是我自己选的。真实数据上还没验两件事：

  Q1 拟合质量对 `Rct` 敏不敏感？若不敏感 ⇒ 数据本身给不出 Rct，必须外部给。
  Q2 固定之后 `R_ad` 会不会只是"补偿"到另一个值？若会 ⇒ 参数解释依赖假设的 Rct。

做法：profile 扫描（**不改生产代码**）
------------------------------------
对每个 `Rct` 在网格上取定值，用 `scipy.least_squares` 拟合其余 6 个参数
（`Rs, Q, n, R_ad, tau_ad, L`），记最优 rms。同时在解处算 Jacobian 的条件数
（log 坐标，与审计口径一致）。free（7 参数）也跑一遍作对照。

若 `rms(Rct)` 在一段宽区间内几乎不变，就说明：Rct 不可由这批数据定出，
固定它必须靠外部测量；而"固定后条件数变好"是真的（数学上），但**不产生信息**。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/adr_fixrct_probe.py
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

OUT = os.path.join(REPO, "out", "adr_fixrct")
PCT = 2.0
#: 平坦判据：rms 相对最优恶化 <= FLAT_PP 个百分点的 Rct 区间宽度（decade）。
FLAT_PP = 0.5
#: Rct 网格（Ω），覆盖 AZ31 谱的 Rs/Rp 量级。
#: 上界取到 1e5 Ω：首版只到 1e3，四条全部"最优点落在网格上界"，
#: 看不出是饱和还是仍在下降。
RCT_GRID = np.logspace(0.0, 5.0, 17)

KEYS = ("Rs", "Q", "n", "R_ad", "tau_ad", "L")


def pack(flat: dict, keys) -> np.ndarray:
    """统一坐标：正量取 log，指数/分数取原值。"""
    return np.array([np.log(flat[k]) if k not in ("n",) else flat[k] for k in keys])


def unpack(x, keys, rct) -> dict:
    """`rct=None` 时 `Rct` 本身就在 `keys` 里（自由拟合），不要覆盖。"""
    p = {}
    for v, k in zip(x, keys):
        p[k] = float(np.exp(v)) if k != "n" else float(v)
    if rct is not None:
        p["Rct"] = float(rct)
    return p


def resid(x, keys, f, z, sg, rct):
    p = unpack(x, keys, rct)
    if not (0.3 <= p["n"] <= 1.0):
        return np.full(2 * f.size, 1e3)
    try:
        zm = T.z_adr(f, p, include_L=True)
    except Exception:
        return np.full(2 * f.size, 1e3)
    d = (z - zm) / sg
    return np.concatenate([d.real, d.imag])


#: 逐参数搜索界。**不能统一用 1e-3..1e6**：AZ31 谱上 `L` 的后验中位在 1e-8 H
#: 量级，统一下界 1e-3 会把初值 clip 成 1 mH，rms 从 7% 变 68%（首版就栽在这）。
PARAM_RANGE = {"Rs": (1e-2, 1e5), "Q": (1e-2, 1e7), "Rct": (1e-2, 1e6),
               "R_ad": (1e-2, 1e7), "tau_ad": (1e-9, 1e3),
               "L": (1e-12, 1e-3), "n": (0.3, 1.0)}


def bounds_of(keys):
    lo, hi = [], []
    for k in keys:
        a, b = PARAM_RANGE[k]
        if k == "n":
            lo.append(a)
            hi.append(b)
        else:
            lo.append(float(np.log(a)))
            hi.append(float(np.log(b)))
    return lo, hi


def de_basin(f, z, sg, seed=0):
    """全局搜索自由 7 参数。

    为什么需要：这个族在真实谱上是强退化的（cond ~1e8），单纯 `least_squares`
    从任何合理初值都会卡在 68%–133% 的局部极小（首版就这样，rms 全错）。
    先用 `differential_evolution` 找到盆地，再交给 `least_squares` 抛光。
    """
    keys = ("Rs", "Q", "n", "Rct", "R_ad", "tau_ad", "L")
    lo, hi = bounds_of(keys)
    bnds = list(zip(lo, hi))

    def obj(x):
        r = resid(x, keys, f, z, sg, None)
        return float(np.sum(r ** 2))

    r = differential_evolution(obj, bnds, seed=seed, tol=1e-8, maxiter=500,
                               popsize=24, polish=True, init="sobol")
    return r.x, float(r.fun)


def fit(f, z, sg, rct=None, x0=None, n_jitter=3, seed=0):
    """`rct=None` 自由拟 7 参数；否则固定该值拟 6 参数。"""
    keys = ("Rs", "Q", "n", "Rct", "R_ad", "tau_ad", "L") if rct is None else KEYS
    lo, hi = bounds_of(keys)
    rng = np.random.default_rng(seed)
    if x0 is None:
        x0 = np.clip(np.full(len(keys), np.log(50.0)), lo, hi)
    starts = [np.clip(x0, lo, hi)]
    for _ in range(n_jitter):
        starts.append(np.clip(x0 + rng.normal(0, 1.0, len(keys)), lo, hi))
    best = None
    for x_start in starts:
        try:
            r = least_squares(resid, x_start, bounds=(lo, hi), x_scale="jac",
                              args=(keys, f, z, sg, rct), max_nfev=6000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    p = unpack(best.x, keys, rct)
    zm = T.z_adr(f, p, include_L=True)
    rms = float(SM.metrics(f, z, zm)["rms_rel"])
    sv = np.linalg.svd(best.jac, compute_uv=False)
    # 逐参数灵敏度 = Jacobian 列的范数（log 坐标）。很小的那几个就是
    # "数据看不见的参数"，cond 通常由它们主导。
    col = np.linalg.norm(best.jac, axis=0)
    sens = {k: float(c) for k, c in zip(keys, col)}
    return dict(params=p, rms=rms,
                cond=float(sv[0] / sv[-1]) if sv[-1] > 0 else None,
                s_min=float(sv[-1]), sv=[float(v) for v in sv],
                sens=sens,
                weakest=sorted(sens, key=lambda k: sens[k])[:2])


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(REPO, "out", "_m6_samples", "z_csv",
                                          "AZ31-12Li_5.csv")))
    files += sorted(glob.glob(os.path.join(REPO, "out", "_m6_samples", "z_csv",
                                           "AZ31-8Li_5.csv")))
    files += sorted(glob.glob(os.path.join(REPO, "out", "_m6_samples", "z_csv",
                                           "AZ31-4Li_0.csv")))
    files += sorted(glob.glob(os.path.join(REPO, "out", "_m6_samples", "z_csv",
                                           "AZ31-12Li_0.csv")))
    res = dict(pct=PCT, flat_pp=FLAT_PP, rct_grid=[float(v) for v in RCT_GRID],
               rows=[])
    for path in files:
        f, z = R.load_spectrum(path)
        # perpoint 下 Re/Im 共用同一 sigma，取一维即可（二维会让残差变成 (4,n)）
        sg = SM.sigma_perpoint(z, PCT)[0]
        xs, _ = de_basin(f, z, sg)
        free = fit(f, z, sg, None, x0=xs)
        x_best = pack(free["params"], ("Rs", "Q", "n", "Rct", "R_ad",
                                       "tau_ad", "L"))
        prof = []
        for rc in RCT_GRID:
            b = dict(free["params"])
            b["Rct"] = float(rc)
            o = fit(f, z, sg, float(rc), x0=pack(b, KEYS))
            if o is not None:
                prof.append(dict(rct=float(rc), rms=o["rms"], cond=o["cond"],
                                 s_min=o["s_min"],
                                 R_ad=o["params"]["R_ad"],
                                 tau_ad=o["params"]["tau_ad"]))
        best = min(prof, key=lambda x: x["rms"])
        # 平坦区间：rms <= best + FLAT_PP 的最长连续 decade 宽度
        ok = [x["rct"] for x in prof if x["rms"] <= best["rms"] + FLAT_PP]
        width = (np.log10(max(ok)) - np.log10(min(ok))) if ok else 0.0
        res["rows"].append(dict(
            spec=os.path.basename(path), n=int(f.size),
            free=dict(rms=free["rms"], cond=free["cond"], s_min=free["s_min"],
                      Rct=free["params"]["Rct"], R_ad=free["params"]["R_ad"],
                      L=free["params"]["L"], sens=free["sens"],
                      weakest=free["weakest"], params=free["params"]),
            profile=prof, best=best, flat_decades=float(width),
            rms_span=float(max(x["rms"] for x in prof)
                           - min(x["rms"] for x in prof))))
        print("%-16s free: rms=%6.2f%% cond=%9.3g Rct=%8.3g L=%.2g 最弱=%s"
              % (os.path.basename(path), free["rms"], free["cond"] or -1,
                 free["params"]["Rct"], free["params"]["L"],
                 ",".join(free["weakest"])))
        print("%-16s 固定 Rct: rms %5.2f–%5.2f%%  平坦 %.2f dec  cond@best=%7.3g"
              % ("", min(x["rms"] for x in prof),
                 max(x["rms"] for x in prof), width, best["cond"] or -1))
    with open(os.path.join(OUT, "probe.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, allow_nan=False)
    print("->", os.path.join(OUT, "probe.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
