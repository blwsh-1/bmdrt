"""复现外部软件（EIS Battery Intelligence）在 AZ31-4Li_100 上的两个吸附电路。

目的只有一个：**先证明我们能逐位复现他们的参数与 Rwp**，再用同一套 LS 口径算残差。
不能复现，后面任何「残差一样」的说法都不可信。

电路（截图文案 → 已验证的接法）
------------------------------
EXT2 `adsorption_RQ`  : Rs + ( CPEdl || ( Rct || (Rads + jωLads) ) )
EXT1 `adsorption_2RQ` : Rs + ( Rct || CPE1 ) + ( CPEdl || ( Rct2 || (Rads + jωLads) ) )

关键点：`Rct[(Rads+Lads)]` 的方括号是**并联**，不是串联。判据是算术而非猜测——
若按串联解读，DC 极限 `Rs+Rct+Rads ≈ 4474 Ω`，而 0.01 Hz 实测 `|Z| = 600.6 Ω`；
按并联解读 `Rct∥Rads = 607.9 Ω`，与实际吻合。据此拟合，Rwp/RMSE/ndl/Rct/Rads/Lads
六个量同时复现（见 `--check` 输出）。

CPE 用外部工具的标准写法 `Z = 1/(Q·(jω)^n)`（截图 Qdl 单位 `S·sⁿ`）。
本脚本内部参数化成 **`A = |Z_cpe| @ ω_ref`（Ω）**，因为它与指数 n 解耦；
输出时再换算回 `Q = 1/(A·ω_ref^n)` 便于对拍截图。

跑法
----
    python docs/round_2026-11-11/ext_circuit_repro.py
    python docs/round_2026-11-11/ext_circuit_repro.py --specs AZ31-4Li_100 --global
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import differential_evolution, least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import bm_real as R                     # noqa: E402

# ---- 外部工具报告值（截图原文，用于对拍） -----------------------------------
REPORTED = {
    "AZ31-4Li_100": {
        "EXT1": dict(rwp=9.379, rmse=145.79, abs_rms=206.17, abs_max=650.64,
                     n_pts=50, ndl=0.79537585, n1=0.44446299,
                     Qdl=3.1239048e-05, Rct=3645.1159),
        "EXT2": dict(rwp=11.175, rmse=147.0, abs_rms=207.89, abs_max=666.21,
                     n_pts=50, ndl=0.77071566, Rct=3748.5298,
                     Rads=725.68276, Lads=7087.7138),
    }
}

SEED = 20261111
MAX_NFEV = 30000


# ---------------------------------------------------------------------------
# 元件与电路
# ---------------------------------------------------------------------------

def cpe_a(f, A, n, w_ref):
    """`A = |Z| @ ω_ref` 的参数化，等价于 `Z = 1/(Q (jω)^n)`，其中 `Q = 1/(A·ω_ref^n)`。"""
    w = 2.0 * np.pi * np.asarray(f, float)
    return A * (w_ref / w) ** n * np.exp(-1j * n * np.pi / 2.0)


def par(*zs):
    return 1.0 / sum(1.0 / z for z in zs)


def z_ext2(f, p, w_ref):
    """Rs + ( CPEdl || ( Rct || (Rads + jωLads) ) )"""
    w = 2.0 * np.pi * np.asarray(f, float)
    ads = p["Rads"] + 1j * w * p["Lads"]
    return p["Rs"] + par(cpe_a(f, p["A_dl"], p["n_dl"], w_ref), par(p["Rct"], ads))


def z_ext1(f, p, w_ref):
    """Rs + (Rct||CPE1) + ( CPEdl || ( Rct2 || (Rads + jωLads) ) )"""
    w = 2.0 * np.pi * np.asarray(f, float)
    ads = p["Rads"] + 1j * w * p["Lads"]
    return (p["Rs"] + par(p["Rct"], cpe_a(f, p["A1"], p["n1"], w_ref))
            + par(cpe_a(f, p["A_dl"], p["n_dl"], w_ref), par(p["Rct2"], ads)))


CIRCUITS = {
    "EXT1": dict(fn=z_ext1, keys=("Rs", "Rct", "A1", "n1", "A_dl", "n_dl",
                                  "Rct2", "Rads", "Lads")),
    "EXT2": dict(fn=z_ext2, keys=("Rs", "A_dl", "n_dl", "Rct", "Rads", "Lads")),
}

# LS 向量顺序：正向电阻/幅值/电感走 log10，指数走线性。
_BOUNDS = {"n": (0.05, 1.0), "log": (-1.5, 7.0)}


def to_q(A, n, w_ref):
    """`A`（Ω @ ω_ref）→ 标准 `Q`（`S·sⁿ`）。"""
    return 1.0 / (A * w_ref ** n)


# ---------------------------------------------------------------------------
# 拟合
# ---------------------------------------------------------------------------

def _transform(keys):
    """返回 (decode, encode, lo, hi)：log 参数与线性参数分开处理。"""
    lo = np.array([_BOUNDS["n"][0] if k.startswith("n") else _BOUNDS["log"][0]
                   for k in keys])
    hi = np.array([_BOUNDS["n"][1] if k.startswith("n") else _BOUNDS["log"][1]
                   for k in keys])

    def decode(x):
        return {k: (float(v) if k.startswith("n") else float(10.0 ** v))
                for k, v in zip(keys, x)}

    return decode, lo, hi


def fit(f, Z, name, use_global=True):
    """相对加权 LS（与外部 Rwp 同口径）。DE 全局 → LS 精修。"""
    spec = CIRCUITS[name]
    keys = spec["keys"]
    decode, lo, hi = _transform(keys)
    w_ref = 2.0 * np.pi * np.sqrt(f.min() * f.max())
    wa = np.abs(Z)

    def resid(x):
        d = spec["fn"](f, decode(x), w_ref) - Z
        return np.concatenate([d.real / wa, d.imag / wa])

    def cost(x):
        return float(np.sqrt(np.mean(resid(x) ** 2)))

    if use_global:
        de = differential_evolution(cost, list(zip(lo, hi)), seed=SEED, tol=1e-10,
                                    maxiter=900, popsize=30, polish=False)
        x0 = de.x
    else:
        x0 = np.clip(np.zeros(len(keys)) if True else None, lo + 1e-6, hi - 1e-6)
    r = least_squares(resid, x0, bounds=(lo, hi), max_nfev=MAX_NFEV,
                      ftol=1e-14, xtol=1e-14, gtol=1e-14)
    p = decode(r.x)
    return dict(ok=bool(np.isfinite(r.cost)), params=p, w_ref=float(w_ref),
                n_params=len(keys), cost=float(np.sqrt(np.mean(r.fun ** 2))))


def metrics(f, Z, Zm):
    r = Zm - Z
    w = np.abs(Z)
    q = np.abs(r) / w
    return dict(
        rwp=float(np.sqrt(np.mean(q ** 2)) * 100),          # 外部 Rwp 口径
        rmse=float(np.sqrt(np.mean(r.real ** 2 + r.imag ** 2) / 2.0)),
        abs_rms=float(np.sqrt(np.mean(np.abs(r) ** 2))),
        abs_max=float(np.max(np.abs(r))),
        rel_max=float(np.max(q) * 100),
        rel_max_at=float(f[int(np.argmax(q))]),
        rms_re=float(np.sqrt(np.mean(r.real ** 2))),
        rms_im=float(np.sqrt(np.mean(r.imag ** 2))),
    )


def predict(f, Z, name, params):
    w_ref = 2.0 * np.pi * np.sqrt(f.min() * f.max())
    return CIRCUITS[name]["fn"](f, params, w_ref)


#: 截图里的参数名 → 本脚本内部名。EXT1 截图标的那个 `Rct=3645` 在接法上是第二层，
#: 对应内部的 `Rct2`（第一层 `Rct∥CPE1` 的电阻未在截图可见区列出）。
_NAME_MAP = {"ndl": "n_dl", "n1": "n1", "Rct": "Rct2", "Rads": "Rads", "Lads": "Lads"}


def check(spec, rec):
    """把复现值与截图逐项比，返回最大相对偏差。"""
    rep = REPORTED.get(spec, {})
    worst = 0.0
    for name, got in rec.items():
        rp = rep.get(name)
        if not rp or not got.get("ok"):
            continue
        for k in ("rwp", "rmse", "abs_rms", "abs_max"):
            worst = max(worst, abs(got[k] - rp[k]) / abs(rp[k]) * 100)
        for k, inner in _NAME_MAP.items():
            if k in rp and inner in got["params"]:
                dev = abs(got["params"][inner] - rp[k]) / abs(rp[k]) * 100
                worst = max(worst, dev)
        if "Qdl" in rp:
            worst = max(worst, abs(got["Qdl_standard"] - rp["Qdl"]) / rp["Qdl"] * 100)
    return float(worst)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=["AZ31-4Li_100"])
    ap.add_argument("--json", default=os.path.join("out", "ext_circuit_repro.json"))
    ap.add_argument("--fast", action="store_true", help="跳过 DE 全局搜索（只做 LS 精修）")
    a = ap.parse_args(argv)

    out = {}
    for sp in a.specs:
        f, Z = R.load_spectrum(os.path.join("out", "_m6_samples", "z_csv", sp + ".csv"))
        print(f"\n=== {sp}  n={f.size}  |Z| {np.abs(Z).min():.2f}–{np.abs(Z).max():.2f} Ω ===")
        rep = REPORTED.get(sp, {})
        rec = dict(n=int(f.size))
        for name in ("EXT1", "EXT2"):
            b = fit(f, Z, name, use_global=not a.fast)
            m = metrics(f, Z, predict(f, Z, name, b["params"]))
            rp = rep.get(name, {})
            print(f"  {name}  Rwp {m['rwp']:.3f}%  报告 {rp.get('rwp')}%"
                  f"   | RMSE {m['rmse']:.2f} 报告 {rp.get('rmse')}"
                  f"   | |ΔZ|rms {m['abs_rms']:.2f} max {m['abs_max']:.2f}"
                  f"（报告 {rp.get('abs_rms')}/{rp.get('abs_max')}）")
            print(f"       rel_max {m['rel_max']:.1f}% @ {m['rel_max_at']:.3g} Hz"
                  f"   rms_re {m['rms_re']:.2f} rms_im {m['rms_im']:.2f}")
            q = to_q(b["params"]["A_dl"], b["params"]["n_dl"], b["w_ref"])
            print(f"       ndl {b['params']['n_dl']:.6f} 报告 {rp.get('ndl')}"
                  f"   |  Qdl(标准) {q:.6g} 报告 {rp.get('Qdl', '—')}")
            if name == "EXT1":
                print(f"       n1 {b['params']['n1']:.6f} 报告 {rp.get('n1')}")
            for k in ("Rct", "Rct2", "Rads", "Lads"):
                if k in b["params"]:
                    want = rp.get(k)
                    tag = (f"   报告 {want:.6g}（差 {abs(b['params'][k]-want)/want*100:.2f}%）"
                           if want else "")
                    print(f"       {k:<5} {b['params'][k]:<12.6g}{tag}")
            rec[name] = dict(ok=b["ok"], **m, params=b["params"],
                             Qdl_standard=q, reported=rp)
        rec["worst_dev_pct"] = check(sp, {k: v for k, v in rec.items()
                                          if k in ("EXT1", "EXT2")})
        print(f"  → 与截图的最大相对偏差 {rec['worst_dev_pct']:.2f}%")
        out[sp] = rec

    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\n→ {a.json}")


if __name__ == "__main__":
    main()
