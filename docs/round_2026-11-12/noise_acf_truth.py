#!/usr/bin/env python3
"""真值噪声 / acf 标定（只读证据脚本，2026-11-12）。

**不跑 NUTS、不改生产代码、不参与任何判定、不改 §13/§14 结论。**

为什么做这一轮
--------------
`M6_VALIDATION_PLAN.md` §13.3 已经给了 `acf≤0.50` 的零分布依据（白噪声
5% 分位上界 0.28–0.40），§19.5 也做过 5% 平滑振荡注入（S3/a=0.05 →
acf 0.616）。所以"给 0.50 找依据"是重复劳动，本脚本**不重做**。

新数据 `D:/eis训练数据/测试数据/` 独占的是：文件名里带**拓扑真值 +
参数真值**，因此 `Z_obs − Z_true` 是可以直接算出来的噪声 realization。
有了它才能第一次把拟合残差 acf 拆成「噪声」＋「模型失配」，并检验
§18/§19「卡 acf 的是数据不是 ECM」在**族外拓扑**上还成不成立。

另外钉死一个容易被说错的数学事实：`acf_max` 对残差是**尺度不变**的
（分子分母同乘常数），所以它只响应噪声的**相关结构**，不响应幅度。
上一轮把它叫「噪声幅度 → acf 标定」不准确，这里用 ×4 缩放与频率重排
两个对照把它写死。

四块内容
--------
P1 拓扑归属：用 `bm_top.z_cpe` 的窗口幅值参数化重写三类真值谱，
   逐位断言与解析真值一致 → 证明涂层/钝化是**嵌套骨架**（最内层是容性
   CPE，而我们 `z_nested` 的 branch 是感性 RL，故严格族外）、腐蚀是
   `voigt K=1` 可表示。
P2 零分布 + 真值噪声：复现 §13.3；统计这批**重尾**噪声在 0.50 线上的
   实际假阳性率；重排/×4 缩放对照。
P3 §19 S3 注入跨数据源复现（同一公式 `+1j·a·|Z_true|·sin(2πt)`）。
P4 分族拟合对照：确定性加权最小二乘（**不是**生产 NUTS 结果），
   分别拟合 voigt K=1..3 与 `z_nested(parallel/series, L=1/2)`，
   与真值上界对比，量化「跑错模型」把 acf 推高多少。

用法（在仓库根跑）
------------------
    python docs/round_2026-11-12/noise_acf_truth.py              # 全跑
    python docs/round_2026-11-12/noise_acf_truth.py --n-per-class 20 --fit-n 3
    python docs/round_2026-11-12/noise_acf_truth.py --p1-only      # 只核拓扑
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import bm_top  # noqa: E402
from bm_experiments import acf_max  # noqa: E402
from bm_ref_rwp import rwp  # noqa: E402

DATA = Path("D:/eis训练数据/测试数据")
OUT = ROOT / "out" / "noise_acf_truth"
ACF_BAR = 0.50
LAGS = (1, 2, 3, 4, 5)
CATS = ("涂层", "钝化", "腐蚀")

# 拓扑归属（由 P1 的数值断言支撑，不是嘴上分类）
TOPO_CLASS = {
    "涂层": dict(skeleton="nested L=2", innermost="R5 + CPE(P6) 容性",
                 our_family="族外：骨架对，但 z_nested 最内层 branch 是感性 RL",
                 voigt="不可表示"),
    "钝化": dict(skeleton="nested L=1", innermost="CPE(P2) + R3 容性",
                 our_family="族外：骨架对，最内层 branch 类型不同",
                 voigt="不可表示"),
    "腐蚀": dict(skeleton="Rs + (R ∥ CPE)", innermost="—",
                 our_family="族内：标准 ZARC = voigt K=1",
                 voigt="K=1 可表示"),
}


# --------------------------------------------------------------------------
# 文件名解析与真值谱
# --------------------------------------------------------------------------
def parse_name(path: Path):
    """`拓扑-[参数].csv` → (spec, params)。用 rfind('-[') 因为拓扑串里也有 '-['。"""
    name = path.name
    i = name.rfind("-[")
    if i < 0:
        raise ValueError(f"文件名不合约定: {name}")
    spec = name[:i]
    params = ast.literal_eval(name[i + 1:-4])  # 含方括号，去掉 .csv
    return spec, [float(x) for x in params]


def list_csv(cat: str):
    return sorted((DATA / cat).rglob("*.csv"))


def cpe_Q(f, Q, n):
    """合成数据的 CPE：`Z = Q/(j·2πf)^n`，Q 是角频率幅值尺度。"""
    return Q / (1j * 2.0 * np.pi * f) ** n


def par(a, b):
    return a * b / (a + b)


def truth_z(spec, f, p):
    if spec.startswith("R1-[P2"):                      # 涂层
        return p[0] + par(cpe_Q(f, p[1], p[2]),
                          p[3] + par(cpe_Q(f, p[4], p[5]),
                                     p[6] + cpe_Q(f, p[7], p[8])))
    if spec.startswith("[P1"):                         # 钝化
        return par(cpe_Q(f, p[0], p[1]), cpe_Q(f, p[2], p[3]) + p[4]) + p[5]
    if spec.startswith("R1-[R2"):                     # 腐蚀
        return p[0] + par(p[1], cpe_Q(f, p[2], p[3]))
    raise ValueError(f"未知拓扑: {spec}")


def skeleton_z(cat, f, p):
    """用 `bm_top.z_cpe` 的**窗口幅值**参数化重写嵌套骨架，最内层放容性 CPE。

    与 truth_z 逐位一致即证明：Q_ours = Q_data / w_ref^n，且这批谱的
    嵌套结构就是我们 z_nested 的骨架（只差 branch 类型）。
    """
    f_ref = float(np.sqrt(f.min() * f.max()))
    w_ref = 2.0 * np.pi * f_ref

    def zc(Q_data, n):
        return bm_top.z_cpe(f, Q_data / w_ref ** n, n, f_ref=f_ref)

    if cat == "涂层":
        z_par2 = par(zc(p[4], p[5]), p[6] + zc(p[7], p[8]))
        return p[0] + par(zc(p[1], p[2]), p[3] + z_par2)
    if cat == "钝化":
        return p[5] + par(zc(p[0], p[1]), p[4] + zc(p[2], p[3]))
    if cat == "腐蚀":
        return p[0] + par(p[1], zc(p[2], p[3]))
    raise ValueError(cat)


def acf(res):
    """项目口径：原始残差实/虚各取 acf_max 再取 max（bm_real.py:1086）。"""
    return max(acf_max(np.asarray(res.real, float)),
               acf_max(np.asarray(res.imag, float)))


def load_case(path: Path, cat: str):
    t = pd.read_csv(path)
    f = t.freq.to_numpy(float)
    z = t.Zreal.to_numpy(float) + 1j * t.Zimag.to_numpy(float)
    o = np.argsort(f)                    # 合成谱是降序，统一升序
    f, z = f[o], z[o]
    spec, p = parse_name(path)
    zt = truth_z(spec, f, p)
    return dict(path=str(path), cat=cat, spec=spec, params=p,
                f=f, z=z, z_true=zt, noise=z - zt)


def p1_topology(all_files):
    rows = []
    for cat in CATS:
        path = all_files[cat][0]
        c = load_case(path, cat)
        zs = skeleton_z(cat, c["f"], c["params"])
        rel = float(np.max(np.abs(zs - c["z_true"]) / np.abs(c["z_true"])))
        rows.append(dict(cat=cat, spec=c["spec"], n_params=len(c["params"]),
                         n_freq=int(c["f"].size),
                         f_min=float(c["f"].min()), f_max=float(c["f"].max()),
                         skeleton_max_rel_err=rel,
                         skeleton_identical=bool(rel < 1e-9),
                         **TOPO_CLASS[cat]))
    return rows


# --------------------------------------------------------------------------
# P2 零分布 / 真值噪声
# --------------------------------------------------------------------------
def null_distribution(n, nrep=12000, seed=20261112):
    rng = np.random.default_rng(seed)
    v = np.empty(nrep)
    for i in range(nrep):
        v[i] = max(acf_max(rng.standard_normal(n)),
                   acf_max(rng.standard_normal(n)))
    return dict(n=int(n), nrep=int(nrep),
               p50=float(np.percentile(v, 50)),
               p95=float(np.percentile(v, 95)),
               p99=float(np.percentile(v, 99)),
               p_max=float(v.max()),
               fpr_at_050=float(np.mean(v >= ACF_BAR)))


def p2_noise(all_files, n_per_class, nrep):
    null = {n: null_distribution(n, nrep) for n in (50, 60, 121)}
    per, detail = {}, []
    rng = np.random.default_rng(7)
    for cat in CATS:
        picks = rng.choice(all_files[cat], min(n_per_class, len(all_files[cat])),
                           replace=False)
        vals, shuf, scaled, rel = [], [], [], []
        for path in picks:
            c = load_case(path, cat)
            d = c["noise"]
            a = acf(d)
            vals.append(a)
            shuf.append(acf(rng.permutation(d)))     # 破坏频率相关
            scaled.append(acf(4.0 * d))             # 尺度不变性
            rel.append(float(np.median(np.abs(d) / np.abs(c["z_true"])) * 100))
        v = np.asarray(vals)
        per[cat] = dict(n=int(v.size),
                        acf_median=float(np.median(v)),
                        acf_p95=float(np.percentile(v, 95)),
                        acf_max=float(v.max()),
                        fpr_at_050=float(np.mean(v >= ACF_BAR)),
                        shuf_median=float(np.median(shuf)),
                        shuf_fpr=float(np.mean(np.asarray(shuf) >= ACF_BAR)),
                        scale_max_abs_diff=float(np.max(np.abs(
                            np.asarray(scaled) - v))),
                        noise_rel_median_pct=float(np.median(rel)))
        for path, a, s, sc in zip(picks, vals, shuf, scaled):
            detail.append(dict(cat=cat, file=Path(path).name[:60],
                               acf_noise=float(a), acf_shuffled=float(s),
                               acf_x4=float(sc)))
    return dict(null=null, per_class=per, detail=detail)


# --------------------------------------------------------------------------
# P3 §19 S3 注入（同一公式，跨数据源复现）
# --------------------------------------------------------------------------
def p3_injection(all_files, amps=(0.05, 0.15, 0.30), n_per_class=12):
    rng = np.random.default_rng(19)
    rows = []
    for cat in CATS:
        picks = rng.choice(all_files[cat], min(n_per_class, len(all_files[cat])),
                           replace=False)
        by_amp = {a: [] for a in amps}
        for path in picks:
            c = load_case(path, cat)
            f, zt, d = c["f"], c["z_true"], c["noise"]
            t = np.log10(f / f.min()) / np.log10(f.max() / f.min())
            for a in amps:
                # §19 原文：Z = z_noisy + 1j*a*|z_true|*sin(2πt)
                Z = zt + d + 1j * a * np.abs(zt) * np.sin(2 * np.pi * t)
                by_amp[a].append(acf(Z - zt))
        rows.append(dict(cat=cat, n=len(picks),
                         by_amp={f"{a:g}": dict(
                             acf_median=float(np.median(by_amp[a])),
                             acf_min=float(np.min(by_amp[a])),
                             acf_max=float(np.max(by_amp[a])),
                             over_050=int(np.sum(np.asarray(by_amp[a]) >= ACF_BAR)),
                             n=int(len(by_amp[a]))) for a in amps},
                         section19_reference=dict(S3_005=0.616, S3_015=0.919,
                                                  S3_030=0.960)))
    return rows


# --------------------------------------------------------------------------
# P4 分族最小二乘对照（确定性代理，不是生产 NUTS）
# --------------------------------------------------------------------------
def _z_voigt(f, x, K):
    """串联 K 个对称 ZARC：`R_inf + Σ R_k/(1+(jωτ_k)^φ_k)`。"""
    z = np.full(f.size, 10.0 ** x[0], dtype=complex)
    w = 2.0 * np.pi * f
    for k in range(K):
        R, tau, phi = 10.0 ** x[1 + 3 * k], 10.0 ** x[2 + 3 * k], x[3 + 3 * k]
        z = z + R / (1.0 + (1j * w * tau) ** phi)
    return z


def _fit_voigt(f, z, zmax, K, n_start=3):
    lo_f, hi_f = np.log10(1 / (2 * np.pi * f.max())) - 2, \
                 np.log10(1 / (2 * np.pi * f.min())) + 2
    # 下界必须足够低：低频 |Z| 峰值可比 Rs 大好几个量级，
    # 用 0.001*zmax 会把 R_inf 顶到真值之上，整条谱拟合崩掉。
    lo = [np.log10(1e-8 * zmax)] + [np.log10(1e-8 * zmax), lo_f, 0.30] * K
    hi = [np.log10(10 * zmax)] + [np.log10(100 * zmax), hi_f, 1.00] * K
    rng = np.random.default_rng(100 + K)
    best = None
    for s in range(n_start):
        x0 = np.array([np.log10(0.01 * zmax)] +
                     [np.log10(zmax / K), 0.5 * (lo_f + hi_f), 0.8] * K, float)
        if s:
            x0 = np.clip(x0 + rng.normal(0, 0.6, x0.size), lo, hi)
        r = least_squares(lambda x: _resid(_z_voigt(f, x, K), z),
                          x0, bounds=(lo, hi), max_nfev=2500)
        if best is None or r.cost < best.cost:
            best = r
    zf = _z_voigt(f, best.x, K)
    return zf, bool(best.success)


def _z_nested_from_x(f, x, keys):
    p = dict(zip(keys, x))
    # log 参数还原；n1/n2 线性
    q = {}
    for k, v in p.items():
        q[k] = v if k in ("n1", "n2") else 10.0 ** v
    branch = "series" if "R_L" in keys else "parallel"
    layers = 2 if "R2" in keys else 1
    if branch == "series":
        # fit_top 里由 numpyro.deterministic 派生：L_ad = R_L · tau_L
        q["L_ad"] = q["R_L"] * q["tau_L"]
    return bm_top.z_nested(f, q, branch=branch, layers=layers)


def _fit_nested(f, z, zmax, branch, layers, n_start=3):
    keys = bm_top.sampled_keys(branch, layers)
    lo, hi, x0 = [], [], []
    for k in keys:
        if k in ("n1", "n2"):
            lo.append(0.30); hi.append(1.00); x0.append(0.8)
        elif k.startswith("tau"):
            lo.append(-12.0); hi.append(4.0); x0.append(-4.0)
        elif k == "L_ad":
            lo.append(-14.0); hi.append(2.0); x0.append(-6.0)
        else:
            lo.append(np.log10(1e-8 * zmax)); hi.append(np.log10(100 * zmax))
            x0.append(np.log10(0.1 * zmax))
    lo, hi, x0 = np.array(lo), np.array(hi), np.array(x0)
    rng = np.random.default_rng(200 + 10 * layers + (1 if branch == "series" else 0))
    best = None
    for s in range(n_start):
        xx = np.clip(x0 if s == 0 else x0 + rng.normal(0, 0.7, x0.size), lo, hi)
        r = least_squares(lambda x: _resid(_z_nested_from_x(f, x, keys), z),
                          xx, bounds=(lo, hi), max_nfev=2500)
        if best is None or r.cost < best.cost:
            best = r
    zf = _z_nested_from_x(f, best.x, keys)
    return zf, bool(best.success)


def _resid(zf, z):
    az = np.abs(z)
    return np.r_[((zf - z).real) / az, ((zf - z).imag) / az]


def p4_family_fit(all_files, fit_n):
    rng = np.random.default_rng(404)
    rows = []
    for cat in CATS:
        picks = rng.choice(all_files[cat], min(fit_n, len(all_files[cat])),
                           replace=False)
        for path in picks:
            c = load_case(path, cat)
            f, z, zt = c["f"], c["z"], c["z_true"]
            zmax = float(np.max(np.abs(z)))
            base = dict(cat=cat, file=Path(path).name[:60],
                        truth=dict(rwp_pct=float(rwp(f, z, zt)),
                                   acf=float(acf(z - zt))))
            models = {}
            for K in (1, 2, 3):
                zf, ok = _fit_voigt(f, z, zmax, K)
                models[f"voigt_K{K}"] = dict(
                    converged=ok, rwp_pct=float(rwp(f, z, zf)),
                    acf=float(acf(zf - z)))
            for branch in ("parallel", "series"):
                for L in (1, 2):
                    zf, ok = _fit_nested(f, z, zmax, branch, L)
                    models[f"nested_{branch}_L{L}"] = dict(
                        converged=ok, rwp_pct=float(rwp(f, z, zf)),
                        acf=float(acf(zf - z)))
            rows.append(dict(**base, models=models))
    # 汇总：每类各模型中位
    summary = {}
    for cat in CATS:
        rs = [r for r in rows if r["cat"] == cat]
        summary[cat] = dict(
            n=len(rs),
            truth_rwp_median=float(np.median([r["truth"]["rwp_pct"] for r in rs])),
            truth_acf_median=float(np.median([r["truth"]["acf"] for r in rs])),
            models={m: dict(
                rwp_median=float(np.median([r["models"][m]["rwp_pct"] for r in rs])),
                acf_median=float(np.median([r["models"][m]["acf"] for r in rs])),
                over_050=int(np.sum([r["models"][m]["acf"] >= ACF_BAR for r in rs])),
                converged=int(np.sum([r["models"][m]["converged"] for r in rs])))
                for m in rs[0]["models"]})
    return dict(cases=rows, summary=summary)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--fit-n", type=int, default=6)
    ap.add_argument("--nrep", type=int, default=12000)
    ap.add_argument("--p1-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    all_files = {c: list_csv(c) for c in CATS}
    print("文件数:", {c: len(v) for c, v in all_files.items()}, flush=True)
    if any(not v for c, v in all_files.items()):
        raise SystemExit("找不到合成谱，检查 DATA 路径")

    t0 = time.time()
    p1 = p1_topology(all_files)
    print("\n=== P1 拓扑归属 ===")
    for r in p1:
        print(f"  {r['cat']}: {r['spec']}  {r['skeleton']}  "
              f"骨架重建相对误差={r['skeleton_max_rel_err']:.2e}  "
              f"identical={r['skeleton_identical']}")
    assert all(r["skeleton_identical"] for r in p1), "骨架数值断言失败"
    json.dump(p1, open(OUT / "p1_topology.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    if a.p1_only:
        print("P1-only 结束"); return

    print("\n=== P2 零分布 / 真值噪声 ===", flush=True)
    p2 = p2_noise(all_files, a.n_per_class, a.nrep)
    for n, d in p2["null"].items():
        print(f"  白噪声 n={n}: p95={d['p95']:.3f} p99={d['p99']:.3f} "
              f"FPR@0.50={d['fpr_at_050']:.4f}")
    for c, d in p2["per_class"].items():
        print(f"  {c}: 真值噪声 acf 中位 {d['acf_median']:.3f} "
              f"max {d['acf_max']:.3f}  FPR@0.50 {d['fpr_at_050']:.3f}  "
              f"重排中位 {d['shuf_median']:.3f}  ×4 最大差 {d['scale_max_abs_diff']:.2e}")
    json.dump(p2, open(OUT / "p2_noise.json", "w", encoding="utf-8"),
             ensure_ascii=False, indent=2)

    print("\n=== P3 §19 S3 注入跨数据源 ===", flush=True)
    p3 = p3_injection(all_files)
    for r in p3:
        s = "  ".join(f"a={k}: {v['acf_median']:.3f} ({v['over_050']}/{v['n']}过线)"
                      for k, v in r["by_amp"].items())
        print(f"  {r['cat']}: {s}")
    print("  §19.5 参照 S3: a=.05→0.616  a=.15→0.919  a=.30→0.960")
    json.dump(p3, open(OUT / "p3_injection.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n=== P4 分族最小二乘对照（确定性代理，非 NUTS）===", flush=True)
    p4 = p4_family_fit(all_files, a.fit_n)
    for c, d in p4["summary"].items():
        print(f"  {c} (n={d['n']}, 真值上界 acf={d['truth_acf_median']:.3f})")
        for m, v in d["models"].items():
            print(f"     {m:22s} Rwp={v['rwp_median']:7.2f}%  "
                  f"acf={v['acf_median']:.3f}  过线 {v['over_050']}/{d['n']}  "
                  f"收敛 {v['converged']}/{d['n']}")
    json.dump(p4, open(OUT / "p4_family_fit.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    json.dump(dict(acf_bar=ACF_BAR, lags=list(LAGS),
                   elapsed_s=round(time.time() - t0, 1),
                   p1=p1, p2=p2, p3=p3, p4=p4),
              open(OUT / "summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\n全部完成 {time.time()-t0:.1f}s → {OUT}")


if __name__ == "__main__":
    main()
