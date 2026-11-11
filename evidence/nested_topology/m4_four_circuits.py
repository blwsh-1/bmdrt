# -*- coding: utf-8 -*-
"""§17 执行器：Shahri 2024 Table 1 四个电路的反事实并列 LS 上界探针。

口径全部来自 `M6_VALIDATION_PLAN.md` §17（跑前预注册），脚本里不另立常数。

**边界**：只读探针。不改生产代码、不改族定义、不改 §13/§14 判据与阈值、
不重跑 §13/§14、不接 `bmdrt.py`；结果一律 `exploratory`。

四个电路的阻抗函数按 Table 1 原图与方程**独立手写**（`_z_c1`..`_z_c4`），
不调用 `bm_top.z_nested`；只在自检里拿 C4 与 `z_nested(parallel,2)` 对拍一次，
防止把文献方程抄错。C1 的 L 是 `R3 + L` 串联、C4 的 L 是 `R3 ∥ L` 并联
（§9.4.27.5 已按原图复核），两者不可混。

拟合是确定性多起点加权最小二乘（σ_j = 0.02·|Z_j|），**不做 MCMC**：
所以本节没有 R̂ / div，M-c 不评；LS 是给定模型下的拟合上界。

用法（用项目 venv `D:\\autoeis_env\\Scripts\\python.exe`）:
    python evidence/nested_topology/m4_four_circuits.py --selftest
    python evidence/nested_topology/m4_four_circuits.py --dry
    python evidence/nested_topology/m4_four_circuits.py --specs AZ31-8Li_5
    python evidence/nested_topology/m4_four_circuits.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bm_experiments as E       # noqa: E402
import bm_real as R              # noqa: E402
import bm_top as T               # noqa: E402
from m4_holdout2 import (SET14, SPEC_DIR, SIGMA_PCT, RMS_MAX, ACF_MAX,  # noqa: E402
                         ACF_DROP, cut_top)

# ---- §17 冻结常数 ------------------------------------------------------------
HOLDOUT_JSON = os.path.join("out", "m4_holdout2.json")
OUT_JSON = os.path.join("out", "m4_four_circuits.json")
N_STARTS = 8                    # §17.2
MAX_NFEV = 1500
LOG_BOUND = 6.0                 # 正参数允许 zmax·10^±6
N_LO, N_HI = 0.5, 1.0          # 沿用项目先验支撑
HF_LO, HF_HI = 1e4, 1e5        # 高频 Bode 对照段（只报，不参与判决）
SEED_STARTS = 20261111
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 元件与四个电路（独立实现，不调用项目拓扑代码）
# ---------------------------------------------------------------------------

def _cpe(f, Q, n, w_ref):
    """`Z = Q·(ω_ref/ω)^n·exp(-j·nπ/2)`，与 `bm_top.z_cpe` 同一参数化。"""
    w = 2.0 * np.pi * np.asarray(f, float)
    return Q * (w_ref / w) ** n * np.exp(-1j * n * np.pi / 2.0)


def _p2(a, b):
    return 1.0 / (1.0 / a + 1.0 / b)


def _p3(a, b, c):
    return 1.0 / (1.0 / a + 1.0 / b + 1.0 / c)


def _common(f, p, w_ref, nq):
    w = 2.0 * np.pi * np.asarray(f, float)
    zl = 1j * w * p["L"]
    q = [None] + [_cpe(f, p[f"Q{i}"], p[f"n{i}"], w_ref) for i in range(1, nq + 1)]
    return zl, q


def _z_c1(f, p, w_ref):
    """Table 1 Model 1：`R(Q(R(QR(LR))))`，L 与 R3 串联，再与 R2 / QPE2 并联。"""
    zl, q = _common(f, p, w_ref, 2)
    z_par2 = _p3(q[2], p["R2"], p["R3"] + zl)
    return p["Rs"] + _p2(q[1], p["R1"] + z_par2)


def _z_c2(f, p, w_ref):
    """Table 1 Model 2：三层嵌套，最内层 `R4 ∥ L1`。"""
    zl, q = _common(f, p, w_ref, 3)
    zb = _p2(p["R4"], zl)
    z_p3 = _p2(q[3], p["R3"] + zb)
    z_p2 = _p2(q[2], p["R2"] + z_p3)
    return p["Rs"] + _p2(q[1], p["R1"] + z_p2)


def _z_c3(f, p, w_ref):
    """Table 1 Model 3：`(R1∥QPE1)` 串联 + 两层嵌套，最内层 `R4 ∥ L1`。"""
    zl, q = _common(f, p, w_ref, 3)
    zb = _p2(p["R4"], zl)
    z_p3 = _p2(q[3], p["R3"] + zb)
    z_p2 = _p2(q[2], p["R2"] + z_p3)
    return p["Rs"] + _p2(p["R1"], q[1]) + z_p2


def _z_c4(f, p, w_ref):
    """Table 1 Model 4：两层嵌套，最内层 `R3 ∥ L1`（= 项目 `parallel/L=2`）。"""
    zl, q = _common(f, p, w_ref, 2)
    zb = _p2(p["R3"], zl)
    z_p2 = _p2(q[2], p["R2"] + zb)
    return p["Rs"] + _p2(q[1], p["R1"] + z_p2)


# 参数顺序即 LS 向量顺序；`n*` 走线性，其余走 log10。
CIRCUITS = {
    "C1": dict(fn=_z_c1, keys=("Rs", "R1", "R2", "R3", "Q1", "n1", "Q2", "n2", "L")),
    "C2": dict(fn=_z_c2, keys=("Rs", "R1", "R2", "R3", "R4", "Q1", "n1",
                               "Q2", "n2", "Q3", "n3", "L")),
    "C3": dict(fn=_z_c3, keys=("Rs", "R1", "R2", "R3", "R4", "Q1", "n1",
                               "Q2", "n2", "Q3", "n3", "L")),
    "C4": dict(fn=_z_c4, keys=("Rs", "R1", "R2", "R3", "Q1", "n1", "Q2", "n2", "L")),
}

# 起点用的典型量级（相对 zmax；L 相对 zmax/w_ref）。
_TYP = {"Rs": 0.2, "R1": 0.1, "R2": 0.3, "R3": 0.5, "R4": 0.2,
        "Q1": 1.0, "Q2": 1.0, "Q3": 1.0, "L": 10.0}


def c1_from_c4(p):
    """C4 → C1 的**解析参数映射**（R-L 星三角变换，与数据无关）。

    C4 内层（去掉 Q2）：`a + (b ∥ jωL)`
    C1 内层（去掉 Q2）：`c ∥ (d + jωL')`

    令两者恒等，解出：
        c = a+b,   d = a(a+b)/b,   L' = ((a+b)/b)² · L

    其余参数（Rs/R1/Q1/n1/Q2/n2）逐位不变。推导见
    `NESTED_TOPOLOGY_PROJECT.md` §9.4.30：**Table 1 的 C1 与 C4 是同一阻抗
    函数的两种等效电路画法**，不是两个不同拓扑。§9.4.27.5 用「同一套
    参数数值」对拍得到的 65 倍 / 227 倍差异，来自参数化不同，不能用来
    断言拓扑不同（同文献 §8 / `REFS_DIGEST.md` §2.I 的恒等案例）。
    """
    a, b, L = p["R2"], p["R3"], p["L"]
    q = dict(p)
    q["R2"] = a + b
    q["R3"] = a * (a + b) / b
    q["L"] = ((a + b) / b) ** 2 * L
    return q


def selftest():
    """C4 与 `bm_top.z_nested(parallel,2)` 对拍 + C1/C4 不等价检查。"""
    f = np.logspace(-2, 5, 61)
    w_ref = 2.0 * np.pi * np.sqrt(f.min() * f.max())
    p = dict(Rs=10.0, R1=55.0, R2=600.0, R3=3000.0,
             Q1=200.0, n1=0.75, Q2=900.0, n2=0.85,
             R4=800.0, Q3=1200.0, n3=0.9, L=4800.0)
    z4 = _z_c4(f, p, w_ref)
    p_top = dict(Rs=p["Rs"], R1=p["R1"], R2=p["R2"], Q1=p["Q1"], n1=p["n1"],
                 Q2=p["Q2"], n2=p["n2"], R_ad=p["R3"], tau_ad=p["L"] / p["R3"])
    z_top = T.z_nested(f, p_top, branch="parallel", layers=2)
    e4 = float(np.max(np.abs(z4 - z_top)) / np.max(np.abs(z_top)))
    z1 = _z_c1(f, p, w_ref)
    e1 = float(np.max(np.abs(z1 - z4)) / np.max(np.abs(z4)))
    z1s = T.z_nested(f, dict(p_top, R_L=p["R3"], L_ad=p["L"]),
                     branch="series", layers=2)
    e1s = float(np.max(np.abs(z1 - z1s)) / np.max(np.abs(z1s)))
    p1 = c1_from_c4(p)
    z1m = _z_c1(f, p1, w_ref)
    e_map = float(np.max(np.abs(z1m - z4)) / np.max(np.abs(z4)))
    print(f"[selftest] C4 vs z_nested(parallel,2)：max rel {e4:.3e}（须 <1e-10）")
    print(f"[selftest] C1/C4 同参数直接对拍：max rel {e1:.2f}（参数化不同，须 >0.1）")
    print(f"[selftest] C1/C4 按星三角映射后：max rel {e_map:.3e}"
          f"（须 <1e-10 → 两者严格恒等）")
    # 差异倍数随参数集变化（§9.4.27.5 在其随机参数下记 227 倍），这里只要求
    # 「远大于数值误差」，不复现某个具体倍数。
    print(f"[selftest] C1 vs z_nested(series,2)：max rel {e1s:.2f}（须 >0.1；"
          f"§9.4.27.5 在另一套参数下记 227 倍量级）")
    ok = (e4 < 1e-10) and (e_map < 1e-10) and (e1 > 0.1) and (e1s > 0.1)
    print("[selftest] " + ("通过" if ok else "失败"))
    return dict(c4_vs_z_nested=e4, c1_vs_c4_same_params=e1,
                c1_vs_c4_mapped=e_map, c1_vs_series=e1s,
                c1_c4_isomorphic=bool(e_map < 1e-10), ok=bool(ok))


# ---------------------------------------------------------------------------
# 拟合
# ---------------------------------------------------------------------------

def _scales(keys, zmax, w_ref):
    return np.array([zmax / w_ref if k == "L" else zmax for k in keys], float)


def _bounds(keys):
    lo = np.array([N_LO if k.startswith("n") else -LOG_BOUND for k in keys], float)
    hi = np.array([N_HI if k.startswith("n") else LOG_BOUND for k in keys], float)
    return lo, hi


def _starts(keys, zmax, w_ref, n_starts):
    """固定 seed 的确定性多起点：第 0 个是基准，其余按 0.7 dec 抖动。"""
    rng = np.random.default_rng(SEED_STARTS)
    sc = _scales(keys, zmax, w_ref)
    base = np.array([0.8 if k.startswith("n") else np.log10(_TYP[k])
                     for k in keys], float)
    lo, hi = _bounds(keys)
    out = [np.clip(base, lo + 1e-6, hi - 1e-6)]
    for _ in range(n_starts - 1):
        jit = rng.normal(0.0, 0.7, size=base.size)
        jit = np.array([j * 0.15 if k.startswith("n") else j
                        for k, j in zip(keys, jit)])
        out.append(np.clip(base + jit, lo + 1e-6, hi - 1e-6))
    return out


def _decode(x, keys, scales):
    p = {}
    for k, v, sc in zip(keys, x, scales):
        p[k] = float(v) if k.startswith("n") else float(sc * 10.0 ** v)
    return p


def fit_one(name, f, Z, sig, w_ref, n_starts=N_STARTS):
    """单电路多起点 LS，返回最优参数与加权代价。"""
    spec = CIRCUITS[name]
    keys, sc = spec["keys"], _scales(spec["keys"], float(np.max(np.abs(Z))), w_ref)
    lo, hi = _bounds(keys)

    def resid(x):
        z = spec["fn"](f, _decode(x, keys, sc), w_ref)
        d = z - Z
        return np.concatenate([d.real / sig, d.imag / sig])

    best, failed = None, 0
    for x0 in _starts(keys, float(np.max(np.abs(Z))), w_ref, n_starts):
        try:
            r = least_squares(resid, x0, bounds=(lo, hi), max_nfev=MAX_NFEV,
                              ftol=1e-10, xtol=1e-10, gtol=1e-10)
            cost = float(np.sqrt(np.mean(r.fun ** 2)))
            if not np.isfinite(cost):
                raise FloatingPointError("non-finite cost")
            if best is None or cost < best["cost"]:
                best = dict(cost=cost, params=_decode(r.x, keys, sc),
                            x=r.x.tolist(), nfev=int(r.nfev),
                            success=bool(r.success))
        except Exception:
            failed += 1
    if best is None:
        return dict(ok=False, error=f"全部 {n_starts} 个起点失败")
    best["ok"] = True
    best["failed_starts"] = failed
    best["n_params"] = len(keys)
    return best


def _metrics(f, Z, Zm):
    """§14 `_metrics` 同口径；额外报 10k–1e5 Hz 高频段。"""
    w = np.abs(Z)
    res = Zm - Z
    d = res / w
    rel = np.abs(res) / w
    bands = []
    for a, b in ((0.01, 1), (1, 100), (100, 1e5)):
        m = (f >= a) & (f < b)
        bands.append(float(np.sqrt(np.mean(np.abs(d[m]) ** 2))) * 100 if m.any() else None)
    hf = (f >= HF_LO) & (f <= HF_HI)
    return dict(
        rms_rel=float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2))) * 100,
        bands=bands,
        acf=float(max(E.acf_max(res.real), E.acf_max(res.imag))),
        acf_re=float(E.acf_max(res.real)), acf_im=float(E.acf_max(res.imag)),
        acf_mag=float(E.acf_max(rel)),
        hf_rms=float(np.sqrt(np.mean(np.abs(d[hf]) ** 2))) * 100 if hf.any() else None,
        hf_n=int(hf.sum()))


def one(spec, voigt_acf, n_starts):
    f_all, Z_all = R.load_spectrum(os.path.join(SPEC_DIR, spec + ".csv"))
    c = cut_top(f_all, Z_all)
    f, Z = f_all[c["keep"]], Z_all[c["keep"]]
    sig = SIGMA_PCT / 100.0 * np.abs(Z)
    w_ref = 2.0 * np.pi * np.sqrt(f.min() * f.max())
    out = {}
    for name in CIRCUITS:
        t0 = time.time()
        b = fit_one(name, f, Z, sig, w_ref, n_starts)
        if not b["ok"]:
            out[name] = b
            continue
        Zm = CIRCUITS[name]["fn"](f, b["params"], w_ref)
        met = _metrics(f, Z, Zm)
        met.update(Ma=bool(met["rms_rel"] <= RMS_MAX),
                   Mb=bool(met["acf"] <= ACF_MAX and voigt_acf - met["acf"] >= ACF_DROP),
                   drop=float(voigt_acf - met["acf"]),
                   cost=b["cost"], n_params=b["n_params"],
                   failed_starts=b["failed_starts"],
                   seconds=round(time.time() - t0, 2), params=b["params"])
        out[name] = met
    return dict(spec=spec, n_pts=int(f_all.size), n_kept=int(f.size),
                dropped_hz=c["dropped"], voigt_acf=float(voigt_acf),
                circuits=out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=list(SET14))
    ap.add_argument("--starts", type=int, default=N_STARTS)
    ap.add_argument("--json", default=OUT_JSON)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    st = selftest()
    if a.selftest:
        return 0 if st["ok"] else 1
    if not st["ok"]:
        print("自检失败，拒绝跑拟合（防止把文献方程抄错）", file=sys.stderr)
        return 1

    with open(HOLDOUT_JSON, encoding="utf-8") as fh:
        hold = json.load(fh)
    base = {r["spec"]: r for r in hold["rows"] if "voigt_base" in r}

    print(f"§17 口径：σ={SIGMA_PCT:g}%·|Z|  LS 上界（{a.starts} 起点）  "
          f"M-a rms≤{RMS_MAX:g}%  M-b acf≤{ACF_MAX:g} 且比 voigt 低≥{ACF_DROP:g}")
    if a.dry:
        for s in a.specs:
            f, Z = R.load_spectrum(os.path.join(SPEC_DIR, s + ".csv"))
            c = cut_top(f, Z)
            print(f"  {s:<15} 剔 {len(c['dropped'])} 点  保留 {int(c['keep'].sum())}/{f.size}"
                  f"  §14 state={base[s]['state']}")
        return 0

    rows, t_all = [], time.time()
    for sp in a.specs:
        if sp not in base:
            print(f"  {sp} 不在 §14 JSON 里，跳过", flush=True)
            continue
        t0 = time.time()
        r = one(sp, base[sp]["voigt_base"]["acf"], a.starts)
        r["state"] = base[sp]["state"]
        m4 = base[sp]["rows"][0]
        r["m4_c4"] = dict(rms_rel=m4["rms_rel"], acf=m4["acf"],
                          hf_rms=m4["bands"][2])
        rows.append(r)
        print(f"\n=== {sp}  ({r['state']})  剔 {len(r['dropped_hz'])} 点 ===", flush=True)
        print(f"  §14 C4 MCMC: rms {m4['rms_rel']:.2f}%  acf {m4['acf']:.3f}  "
              f"voigt acf {r['voigt_acf']:.3f}")
        for name, m in r["circuits"].items():
            if not m.get("ok", True):
                print(f"  {name}: {m.get('error')}", flush=True)
                continue
            print(f"  {name} ({m['n_params']}参): rms {m['rms_rel']:6.2f}%  "
                  f"acf {m['acf']:.3f} (mag {m['acf_mag']:.3f})  "
                  f"drop {m['drop']:+.3f}  10k-100k {m['hf_rms']:.2f}%  "
                  f"M-a {'✓' if m['Ma'] else '✗'} M-b {'✓' if m['Mb'] else '✗'}  "
                  f"[{m['seconds']}s]", flush=True)
        print(f"  小计 {time.time() - t0:.1f}s", flush=True)

    summary = {}
    for name in CIRCUITS:
        rr = [r for r in rows if r["circuits"].get(name, {}).get("ok", True)
              and "acf" in r["circuits"][name]]
        fail = [r for r in rr if r["state"] == "未过"]
        ctrl = [r for r in rr if r["state"] == "过"]
        summary[name] = dict(
            n_params=len(CIRCUITS[name]["keys"]), n=len(rr),
            Ma=int(sum(r["circuits"][name]["Ma"] for r in rr)),
            Mb=int(sum(r["circuits"][name]["Mb"] for r in rr)),
            both=int(sum(r["circuits"][name]["Ma"] and r["circuits"][name]["Mb"]
                         for r in rr)),
            fail_both=int(sum(r["circuits"][name]["Ma"] and r["circuits"][name]["Mb"]
                              for r in fail)),
            mean_acf=float(np.mean([r["circuits"][name]["acf"] for r in rr])) if rr else None,
            mean_rms=float(np.mean([r["circuits"][name]["rms_rel"] for r in rr])) if rr else None,
            fail_mean_acf=float(np.mean([r["circuits"][name]["acf"] for r in fail])) if fail else None,
            ctrl_mean_acf=float(np.mean([r["circuits"][name]["acf"] for r in ctrl])) if ctrl else None,
            ctrl_Mb=int(sum(r["circuits"][name]["Mb"] for r in ctrl)))

    def _f(v):
        return "n/a" if v is None else f"{v:.3f}"

    print("\n=== 汇总（LS 上界，M-c 不评） ===")
    for name, s in summary.items():
        print(f"  {name} ({s['n_params']}参): M-a {s['Ma']}/{s['n']}  "
              f"M-b {s['Mb']}/{s['n']}  同时 {s['both']}/{s['n']}  "
              f"其中未过组 {s['fail_both']}  平均 acf {_f(s['mean_acf'])}  "
              f"未过/对照 acf {_f(s['fail_mean_acf'])}/{_f(s['ctrl_mean_acf'])}  "
              f"对照 M-b {s['ctrl_Mb']}/4")
    print(f"总耗时 {time.time() - t_all:.1f}s")

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(prereg="M6_VALIDATION_PLAN.md §17",
                           status="exploratory", selftest=st,
                           constants=dict(sigma_pct=SIGMA_PCT, n_starts=a.starts,
                                          rms_max=RMS_MAX, acf_max=ACF_MAX,
                                          acf_drop=ACF_DROP),
                           rows=rows, summary=summary), fh,
                      ensure_ascii=False, indent=1)
        print(f"已写 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
