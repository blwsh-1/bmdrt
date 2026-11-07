# -*- coding: utf-8 -*-
"""M3 档 2：单组参数恢复（`BM_TOP_M3_PLAN.md` §6.2 / §12）。

一次只跑一个条件，分两个子步骤：

    A  固定噪声 `sigma = (0.02·zmax, 0.02·zmax)` —— 隔离后验几何，只看
       `R_ad` / `tau_ad` 能不能从数据里回来；
    B  自由噪声（`sigma=None`，走 `LogNormal(log 0.01·zmax, 1)`）—— 看噪声
       参数与电路参数共同采样是否引入新的简并。

**两步必须分开记录**。A 通过不等于 B 通过：A 里 σ 是已知的，B 多两个自由度，
后验几何完全不同（`bm_core` 记录过无噪声 + 自由 σ 的漏斗）。

合成谱由 `bm_top_gen` 独立生成（不调 `bm_top.z_nested`），噪声按 §4.1 的
固定标准差、实虚独立口径。

用法
----
    # 单组正式跑（4 链 × 1200 样本）
    python evidence/nested_topology/m3_stage2.py --stage both

    # 只跑 A、换 seed、写 JSON
    python evidence/nested_topology/m3_stage2.py --stage a --seed 1 --json out.json

    # 链路自检（1 链 150 样本，只证明能跑通，结果不可引用）
    python evidence/nested_topology/m3_stage2.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import bm_top as T                      # noqa: E402  （拟合器）
import bm_top_gen as G                  # noqa: E402  （独立生成器）

# 审查批准的基准真值（沿用 §11.2 冒烟的那组，便于对照）
TRUTH = dict(Rs=10.0, R1=55.0, R2=600.0, Q1=200.0, n1=0.75,
             Q2=900.0, n2=0.85, R_ad=3000.0, tau_ad=1.6)

# A 组：感抗环主导，回答"感抗支路能否恢复"；
# B 组：只把 R_ad 降到 300（zmax 3049→891、σ 61→17.8），回答"形状参数能否改善"。
# 两组共用同一噪声口径（固定 2%·zmax），差异只有一个因子，便于归因。
TRUTH_SETS = {"a": dict(TRUTH), "b": dict(TRUTH, R_ad=300.0)}

F_GRID = (1e-2, 1e5, 61)


SERIES_TRUTH = dict(R_L=800.0, tau_L=1.625)     # L_ad = 1300，与 parallel 支路可比


def truth_params(branch: str, layers: int, override: dict | None = None) -> dict:
    """当前拓扑的真值（只含该拓扑真正采样的参数）。`override` 覆盖个别值。"""
    p = dict(TRUTH)
    if override:
        p.update(override)
    if branch == "series":
        p.pop("R_ad", None)
        p.pop("tau_ad", None)
        p.update(SERIES_TRUTH)
    return {k: p[k] for k in T.sampled_keys(branch, layers) if k in p}


def make_spectrum(noise: float = 0.02, seed: int = 0, branch: str = "parallel",
                  layers: int = 2, truth: dict | None = None):
    """独立生成 + 加噪。返回 (f, Z, zmax)。"""
    f = np.logspace(np.log10(F_GRID[0]), np.log10(F_GRID[1]), F_GRID[2])
    p = truth if truth is not None else truth_params(branch, layers)
    Z0 = G.gen(f, p, branch=branch, layers=layers)
    zmax = float(np.max(np.abs(Z0)))
    Z = G.add_noise(Z0, noise, mode="fixed", seed=seed, zmax=zmax)
    return f, Z, zmax


def _truth_of_comb(name: str, truth: dict):
    """组合参数对应的真值（真值来自同一组 `truth_params`）。"""
    if name == "R1+R2":
        return truth["R1"] + truth["R2"]
    if name.startswith("L_ad"):
        if "R_ad" in truth:
            return truth["R_ad"] * truth["tau_ad"]
        return truth["R_L"] * truth["tau_L"]
    if name.startswith("R2·Q2"):
        return truth["R2"] * truth["Q2"] ** truth["n2"]
    return None


def summarize(res: dict, truth: dict, branch: str, layers: int,
              sigma_note: str) -> dict:
    """把一个 `fit_top` 结果压成可记录的指标。"""
    s = res["s"]
    keys = list(T.sampled_keys(branch, layers))
    if branch == "series":
        keys.append("L_ad")                     # 派生量也看一下
    out = {"sigma_note": sigma_note, "params": {}, "diagnostics": {}}

    for k in keys:
        if k not in s:
            continue
        v = np.asarray(s[k], float).ravel()
        lo, med, hi = np.percentile(v, [5, 50, 95])
        rec = {"median": float(med), "p05": float(lo), "p95": float(hi),
               "width90": float(hi - lo), "n": int(v.size)}
        if k in truth:
            tv = float(truth[k])
            rec.update(truth=tv,
                       rel_err=float((med - tv) / tv),
                       covered=bool(lo <= tv <= hi))
        out["params"][k] = rec

    # 成对相关：只取采样参数，避免派生量重复计入
    mat_keys = [k for k in T.sampled_keys(branch, layers) if k in s and
                np.asarray(s[k]).size > 1]
    M = np.column_stack([np.asarray(s[k], float).ravel() for k in mat_keys])
    C = np.corrcoef(M, rowvar=False)
    idx = {k: i for i, k in enumerate(mat_keys)}
    pairs = []
    for i in range(len(mat_keys)):
        for j in range(i + 1, len(mat_keys)):
            pairs.append((mat_keys[i], mat_keys[j], float(C[i, j])))
    pairs.sort(key=lambda t: -abs(t[2]))

    # 审查点名要看：优先恢复参数 vs 形状参数（§6.2 的"结构简并"来源）
    pri_keys = [k for k in ("R_ad", "tau_ad", "R_L", "tau_L") if k in idx]
    shape_keys = [k for k in ("Rs", "R1", "R2", "Q1", "n1", "Q2", "n2") if k in idx]
    pri_shape = [{"a": a, "b": b, "rho": float(C[idx[a], idx[b]])}
                 for a in pri_keys for b in shape_keys]

    # 多 seed 要盯的几对固定组合（"权衡是否稳定"看的就是这几个 ρ）
    def _rho(a, b):
        return float(C[idx[a], idx[b]]) if (a in idx and b in idx) else None

    named = {"R_ad,tau_ad": _rho("R_ad", "tau_ad"),
             "R_L,tau_L": _rho("R_L", "tau_L"),
             "R1,R2": _rho("R1", "R2"),
             "Rs,n1": _rho("Rs", "n1")}

    # 组合参数（§6.2："无法分别辨识者，报告可辨识的组合"）
    comb = {}
    if {"R1", "R2"} <= set(s):
        comb["R1+R2"] = np.asarray(s["R1"], float).ravel() + np.asarray(s["R2"], float).ravel()
    if branch == "parallel":
        comb["L_ad(=R_ad·tau_ad)"] = (np.asarray(s["R_ad"], float).ravel()
                                      * np.asarray(s["tau_ad"], float).ravel())
    else:
        comb["L_ad(=R_L·tau_L)"] = (np.asarray(s["R_L"], float).ravel()
                                    * np.asarray(s["tau_L"], float).ravel())
    if branch == "parallel" and "R2" in s:      # CPE 的"等效容抗"组合，Q 与 n 强耦合时看
        comb["R2·Q2^n2"] = (np.asarray(s["R2"], float).ravel()
                            * np.asarray(s["Q2"], float).ravel()
                            ** np.median(np.asarray(s["n2"], float).ravel()))
    for k, v in comb.items():
        lo, med, hi = np.percentile(v, [5, 50, 95])
        rec = {"median": float(med), "p05": float(lo), "p95": float(hi),
               "width90": float(hi - lo), "n": int(v.size)}
        tv = _truth_of_comb(k, truth)
        if tv is not None:
            rec.update(truth=float(tv), rel_err=float((med - tv) / tv),
                       covered=bool(lo <= tv <= hi))
        out.setdefault("combinations", {})[k] = rec

    pw = res.get("pointwise_elpd")
    pk = np.asarray(res.get("pareto_k", []), float)
    div = res.get("divergences")
    n_draws = int(np.asarray(s[mat_keys[0]]).size)
    out["diagnostics"] = {
        "rhat_max": res.get("rhat_max"),
        "ess_min": res.get("ess_min"),
        "rhat_by_param": res.get("rhat_by_param") or {},
        "ess_by_param": res.get("ess_by_param") or {},
        "divergences": div,
        "div_frac": (None if div is None else float(div) / max(n_draws, 1)),
        "n_draws": n_draws,
        "elpd": res.get("elpd"), "elpd_se": res.get("elpd_se"),
        "pointwise_len": (None if pw is None else int(np.asarray(pw).size)),
        "pareto_k_max": (float(np.max(pk)) if pk.size else None),
        "pareto_k_gt07_frac": (float(np.mean(pk > 0.7)) if pk.size else None),
        "corr_max_abs": (abs(pairs[0][2]) if pairs else None),
        "corr_top": [{"a": a, "b": b, "rho": r} for a, b, r in pairs[:5]],
        "corr_pri_vs_shape": pri_shape,
        "corr_pri_vs_shape_max": (max(abs(c["rho"]) for c in pri_shape)
                                  if pri_shape else None),
        "corr_named": named,
    }
    return out


def acceptance(rec: dict, n_freq: int,
               err_targets=(0.01, 0.04, 0.08)) -> dict:
    """§6.3 的机器判定。只判**这一次**记录，不当覆盖率结论。"""
    d = rec["diagnostics"]
    p = rec["params"]
    pri = ["R_ad", "tau_ad"] if "R_ad" in p else ["R_L", "tau_L"]
    worst_err = max(abs(p[k].get("rel_err", np.inf)) for k in pri if k in p)
    checks = {
        "rhat_lt_1.01": (d["rhat_max"] is not None and d["rhat_max"] < 1.01),
        "ess_gt_200": (d["ess_min"] is not None and d["ess_min"] > 200),
        "div_lt_0.1pct": (d["div_frac"] is not None and d["div_frac"] < 1e-3),
        "pointwise_len_ok": (d["pointwise_len"] == n_freq),
        "pareto_k_ok": (d["pareto_k_max"] is not None and d["pareto_k_max"] <= 0.7),
        "covered": all(bool(p[k].get("covered")) for k in pri if k in p),
        "pri_err_le_4pct": bool(worst_err <= err_targets[1]),
    }
    return {"checks": checks, "pri_err_worst": float(worst_err),
            "pass": all(checks.values())}


def run_stage(stage: str, *, seed: int, noise: float, branch: str, layers: int,
              n_warmup: int, n_samples: int, n_chains: int, progress: bool,
              init_mode: str = "prior", target_accept: float = 0.9,
              max_tree_depth: int = 10, samples_path: str | None = None,
              truth_override: dict | None = None) -> dict:
    truth = truth_params(branch, layers, override=truth_override)
    f, Z, zmax = make_spectrum(noise=noise, seed=seed, branch=branch, layers=layers,
                               truth=truth)

    sigma = (noise * zmax, noise * zmax) if stage == "a" else None
    if init_mode == "oracle":
        # 真值经物理 → 无约束转换。只用于"先验/初值是否卡住采样器"的诊断，
        # **不能**当正式结果：链从真值出发时，"收敛得好"可能只是初值的回声。
        # 必须把 sigma 一起传进去：固定噪声时模型没有 s_re/s_im 位点，
        # 多给会被 NumPyro 当成多余的树节点。
        init = T.physical_to_unconstrained(
            truth, zmax=zmax, w_ref=float(2 * np.pi * np.sqrt(f.min() * f.max())),
            branch=branch, layers=layers, sigma=sigma, num_chains=n_chains)
    else:
        init = None                            # 走 numpyro 默认（先验支撑内的随机初值）

    t0 = time.time()
    res = T.fit_top(f, Z, branch=branch, layers=layers,
                    n_warmup=n_warmup, n_samples=n_samples, num_chains=n_chains,
                    seed=seed, sigma=sigma, init_values=init, progress=progress,
                    target_accept=target_accept, max_tree_depth=max_tree_depth)
    dt = time.time() - t0
    if samples_path:
        np.savez_compressed(samples_path, **{k: np.asarray(v) for k, v in res["s"].items()})

    rec = summarize(res, truth, branch, layers,
                    "fixed 2%·zmax" if stage == "a" else "free LogNormal(0.01·zmax,1)")
    rec.update(stage=stage, seed=seed, noise=noise, branch=branch, layers=layers,
               n_chains=n_chains, n_samples=n_samples, n_warmup=n_warmup,
               n_freq=int(f.size), zmax=zmax, seconds=round(dt, 1),
               init_mode=init_mode, target_accept=target_accept,
               max_tree_depth=max_tree_depth, truth=truth)
    rec["acceptance"] = acceptance(rec, int(f.size))
    return rec


def print_record(rec: dict) -> None:
    d, p = rec["diagnostics"], rec["params"]
    print(f"\n{'='*74}")
    print(f"档 2  [{rec['stage']}] {rec['branch']}_L{rec['layers']}  "
          f"seed={rec['seed']}  噪声={rec['noise']:.0%}  噪声口径={rec['sigma_note']}")
    print(f"       {rec['n_chains']} 链 × {rec['n_samples']} 样本 / warmup {rec['n_warmup']}"
          f"  初值={rec['init_mode']}  → {rec['seconds']}s")
    print(f"{'='*74}")
    print(f"{'参数':<8}{'真值':>10}{'中位':>12}{'90% 区间':>26}{'相对误差':>10}{'覆盖':>6}")
    for k, v in p.items():
        ci = "[{:.4g}, {:.4g}]".format(v["p05"], v["p95"])
        if "truth" in v:
            cover = "✓" if v["covered"] else "✗"
            print(f"{k:<8}{v['truth']:>10.4g}{v['median']:>12.4g}{ci:>26}"
                  f"{v['rel_err']:>+9.2%}{cover:>6}")
        else:
            print(f"{k:<8}{'—':>10}{v['median']:>12.4g}{ci:>26}")
    comb = rec.get("combinations") or {}
    if comb:
        print(f"\n{'组合参数':<20}{'真值':>12}{'中位':>12}{'90% 区间':>26}{'相对误差':>10}{'覆盖':>6}")
        for k, v in comb.items():
            ci = "[{:.4g}, {:.4g}]".format(v["p05"], v["p95"])
            if "truth" in v:
                cover = "✓" if v["covered"] else "✗"
                print(f"{k:<20}{v['truth']:>12.4g}{v['median']:>12.4g}{ci:>26}"
                      f"{v['rel_err']:>+9.2%}{cover:>6}")
            else:
                print(f"{k:<20}{'—':>12}{v['median']:>12.4g}{ci:>26}")
    print(f"\n诊断  R̂max={d['rhat_max']:.4f}  ESSmin={d['ess_min']:.0f}  "
          f"div={d['divergences']} ({d['div_frac']:.4%})")
    print(f"      elpd={d['elpd']:.2f}±{d['elpd_se']:.2f}  "
          f"pointwise_len={d['pointwise_len']} (N={rec['n_freq']})  "
          f"pareto_k_max={d['pareto_k_max']:.3f}  k>0.7 占比={d['pareto_k_gt07_frac']:.1%}")
    print("      最强相关：")
    for c in d["corr_top"][:3]:
        print(f"        corr({c['a']}, {c['b']}) = {c['rho']:+.3f}")
    rb = d.get("rhat_by_param") or {}
    if rb:
        bad = sorted(rb.items(), key=lambda kv: -kv[1])[:3]
        print("      R̂ 最差参数：" + "，".join(f"{k}={v:.4f}" for k, v in bad))
    if d.get("corr_pri_vs_shape_max") is not None:
        worst = max(d["corr_pri_vs_shape"], key=lambda c: abs(c["rho"]))
        print(f"      优先恢复 × 形状参数 最大 |ρ| = {abs(worst['rho']):.3f}"
              f"（{worst['a']} vs {worst['b']}）")
    a = rec["acceptance"]
    print(f"\n判定（仅本组）：{'通过' if a['pass'] else '未通过'}"
          f"  优先恢复参数最大误差 {a['pri_err_worst']:.2%}")
    for k, v in a["checks"].items():
        print(f"      {'✓' if v else '✗'} {k}")


def _fmt_pct(x, w: int = 11) -> str:
    return f"{x:>+{w}.1%}" if x is not None else f"{'—':>{w}}"


def _fmt_num(x, w: int = 8, s: str = "g") -> str:
    s = s.lstrip(".")                     # 允许传 ".3%" 或 "3%"，统一成 "3%"
    return f"{x:>{w}.{s}}" if x is not None else f"{'—':>{w}}"


def print_seed_table(recs: list[dict]) -> dict:
    """逐 seed 一行，再给汇总。**不汇总成单一数字**——审查要求逐 seed 保留。"""
    import statistics as stats

    r0 = recs[0]
    branch = r0["branch"]
    p1, p2 = ("R_ad", "tau_ad") if branch == "parallel" else ("R_L", "tau_L")
    lk = (f"L_ad(={p1}·{p2})")
    print(f"\n{'='*96}")
    print(f"多 seed 汇总：{branch}_L{r0['layers']}  {r0['noise']:.0%} 噪声（{r0['sigma_note']}）  "
          f"{r0['n_chains']} 链 × {r0['n_samples']} / warmup {r0['n_warmup']}  "
          f"ta={r0['target_accept']}  depth={r0['max_tree_depth']}  初值={r0['init_mode']}")
    print(f"{'='*96}")
    print(f"{'seed':>5}{'R_ad 误差':>12}{'tau_ad 误差':>13}{'L_ad 误差':>12}"
          f"{'覆盖':>6}{'ρ(ad,τ)':>10}{'ρ(R1,R2)':>11}"
          f"{'R̂max':>8}{'ESSmin':>8}{'div':>9}{'判定':>6}")
    for r in recs:
        p, d, c = r["params"], r["diagnostics"], r.get("combinations", {})
        e1 = p.get(p1, {}).get("rel_err")
        e2 = p.get(p2, {}).get("rel_err")
        e3 = c.get(lk, {}).get("rel_err")
        cov = sum(1 for k in (p1, p2) if p.get(k, {}).get("covered"))
        named = d.get("corr_named") or {}
        r1, r2 = named.get(f"{p1},{p2}"), named.get("R1,R2")
        print(f"{r['seed']:>5}"
              f"{_fmt_pct(e1, 12)}{_fmt_pct(e2, 13)}{_fmt_pct(e3, 12)}"
              f"{f'{cov}/2':>6}"
              f"{(f'{r1:+.3f}' if r1 is not None else '—'):>10}"
              f"{(f'{r2:+.3f}' if r2 is not None else '—'):>11}"
              f"{_fmt_num(d['rhat_max'], 8, '4f')}{_fmt_num(d['ess_min'], 8, '0f')}"
              f"{_fmt_num(d['div_frac'], 8, '.3%')}"
              f"{'✓' if r['acceptance']['pass'] else '✗':>6}")

    e1s = [r["params"].get(p1, {}).get("rel_err") for r in recs]
    e2s = [r["params"].get(p2, {}).get("rel_err") for r in recs]
    e1s = [x for x in e1s if x is not None]
    e2s = [x for x in e2s if x is not None]
    cov1 = sum(1 for r in recs if r["params"].get(p1, {}).get("covered"))
    cov2 = sum(1 for r in recs if r["params"].get(p2, {}).get("covered"))
    npass = sum(1 for r in recs if r["acceptance"]["pass"])
    n = len(recs)
    print("-" * 96)
    print(f"汇总  {p1} 误差 中位 {stats.median(e1s):+.2%}  范围 "
          f"[{min(e1s):+.2%}, {max(e1s):+.2%}]")
    print(f"      {p2} 误差 中位 {stats.median(e2s):+.2%}  "
          f"均值 {stats.mean(e2s):+.2%} ± {stats.stdev(e2s):.2%}  "
          f"范围 [{min(e2s):+.2%}, {max(e2s):+.2%}]  同号 "
          f"{sum(1 for x in e2s if x > 0)}/{len(e2s)} 正")
    print(f"      90% 区间覆盖：{p1} {cov1}/{n}、{p2} {cov2}/{n}；"
          f"收敛判定 {npass}/{n}；|{p2}|≤4% 的 seed "
          f"{sum(1 for x in e2s if abs(x) <= 0.04)}/{len(e2s)}")
    print("      注：seed 数是探索性重复，不是 90% 覆盖率的统计检验。")
    return {"n_seeds": n, "n_converged": npass,
            "covered": {p1: cov1, p2: cov2},
            "err": {p1: {"median": stats.median(e1s), "min": min(e1s),
                         "max": max(e1s)},
                    p2: {"median": stats.median(e2s), "mean": stats.mean(e2s),
                         "sd": stats.stdev(e2s) if len(e2s) > 1 else 0.0,
                         "min": min(e2s), "max": max(e2s),
                         "n_positive": sum(1 for x in e2s if x > 0)}}}


def aggregate_seeds(recs: list[dict], params: list[str] | None = None) -> dict:
    """多 seed 的**稳健性**汇总。只读，不改 `acceptance()` 的语义。

    三个层次分开记（审查要求）：
      1. 单次精度：每次拟合的 |中位误差| 是否 ≤4%（`acceptance()` 里的判据，逐 seed）；
      2. 多 seed 稳健性：误差均值 / SD / 最大绝对误差 —— 反映采样随机性；
      3. 后验不确定性：90% 区间宽度与覆盖次数。

    这里**不给**"5 个 seed 的均值 ≤4% 就算达标"的结论：这 5 次共用同一组真值、
    同一频栅、同一份合成谱，只是采样随机性的重复，不是 5 次独立实验。
    均值 0% 也推不出"精度 4%"，正负抵消的单次误差同样没达精度。
    """
    import statistics as stats

    r0 = recs[0]
    branch = r0["branch"]
    if params is None:
        pri = ["R_ad", "tau_ad"] if branch == "parallel" else ["R_L", "tau_L"]
        params = [k for k in pri if k in r0["params"]] + \
                 [k for k in ("Rs", "R1", "R2", "Q1", "n1", "Q2", "n2")
                  if k in r0["params"]]

    def pack(errs, widths, cov, n_le4):
        return {"n": len(errs),
                "mean": float(stats.mean(errs)) if errs else None,
                "sd": float(stats.stdev(errs)) if len(errs) > 1 else None,
                "median": float(stats.median(errs)) if errs else None,
                "max_abs": float(max(abs(x) for x in errs)) if errs else None,
                "n_le_4pct": int(n_le4), "covered": int(cov),
                "width90_median": float(stats.median(widths)) if widths else None}

    out = {"n_seeds": len(recs),
           "n_pass_acceptance": sum(1 for r in recs if r["acceptance"]["pass"]),
           "params": {}, "combinations": {}}

    for k in params:
        errs, widths, cov, n4 = [], [], 0, 0
        for r in recs:
            v = r["params"].get(k)
            if not v or "rel_err" not in v:
                continue
            errs.append(float(v["rel_err"]))
            widths.append(float(v["width90"] / v["truth"]))
            cov += bool(v["covered"])
            n4 += abs(v["rel_err"]) <= 0.04
        if errs:
            out["params"][k] = pack(errs, widths, cov, n4)

    for k in ("L_ad(=R_ad·tau_ad)", "L_ad(=R_L·tau_L)", "R1+R2"):
        errs, widths, cov, n4 = [], [], 0, 0
        for r in recs:
            v = (r.get("combinations") or {}).get(k)
            if not v or "rel_err" not in v:
                continue
            errs.append(float(v["rel_err"]))
            widths.append(float(v["width90"] / v["truth"]))
            cov += bool(v["covered"])
            n4 += abs(v["rel_err"]) <= 0.04
        if errs:
            out["combinations"][k] = pack(errs, widths, cov, n4)
    return out


def print_robustness(agg: dict) -> None:
    """三层指标并排打印。列名写死，避免把'均值'读成'精度'。"""
    n = agg["n_seeds"]
    print(f"\n{'='*104}")
    print("多 seed 稳健性（**不改变单次 4% 判据**；本表只描述对采样随机性的敏感度）")
    print(f"{'='*104}")
    print(f"{'参数':<20}{'均值':>10}{'SD':>9}{'中位':>9}{'max|err|':>10}"
          f"{'|err|≤4%':>10}{'覆盖':>8}{'区间宽中位':>12}")
    for grp, title in (("params", "单参数"), ("combinations", "组合参数")):
        if not agg[grp]:
            continue
        print(f"-- {title} " + "-" * 88)
        for k, v in agg[grp].items():
            frac4 = "{}/{}".format(v["n_le_4pct"], n)
            cov = "{}/{}".format(v["covered"], n)
            print(f"{k:<20}{_fmt_pct(v['mean'], 10)}{_fmt_pct(v['sd'], 9)}"
                  f"{_fmt_pct(v['median'], 9)}{_fmt_pct(v['max_abs'], 10)}"
                  f"{frac4:>10}{cov:>8}{v['width90_median']:>11.1%}")
    print("-" * 104)
    print(f"收敛（acceptance 全过）：{agg['n_pass_acceptance']}/{n}。")
    print("注意 ①：均值接近 0 **不等于**达到 4% 精度——正负误差可以互相抵消。")
    print("注意 ②：区间宽远超 100% 的参数，其'覆盖真值'没有信息量（宽区间包得住任何值）。")
    print("注意 ③：SD 与 4% 目标同量级时，单次拟合判达标/不达标接近抛硬币；")
    print("        结论只能按分布说，不能拿单个 seed 说'达标'或'不达标'。")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M3 档 2 单组参数恢复")
    ap.add_argument("--stage", choices=["a", "b", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None,
                    help="逗号分隔的 seed 列表（如 0,1,2,3,4）；给了就逐 seed 全记录")
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--truth-set", choices=sorted(TRUTH_SETS), default="a",
                    help="真值组：a=感抗环主导(R_ad=3000)，b=感抗环减弱(R_ad=300)")
    ap.add_argument("--truth", default=None,
                    help="个别真值覆盖，如 R_ad=300,tau_ad=1.6（优先于 --truth-set）")
    ap.add_argument("--branch", choices=["parallel", "series"], default="parallel")
    ap.add_argument("--layers", type=int, choices=[1, 2], default=2)
    ap.add_argument("--smoke", action="store_true", help="1 链 150 样本，只验链路")
    ap.add_argument("--init", choices=["prior", "oracle"], default="prior",
                    help="prior=默认随机初值（正式用）；oracle=真值初值（只做诊断）")
    ap.add_argument("--warmup", type=int, default=None, help="覆盖 warmup 步数")
    ap.add_argument("--samples", type=int, default=None, help="覆盖每链样本数")
    ap.add_argument("--chains", type=int, default=None, help="覆盖链数")
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--save-samples", action="store_true", help="把后验样本存成 npz 备查")
    ap.add_argument("--progress", action="store_true", help="打印 MCMC 进度条")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    kw = dict(n_warmup=150, n_samples=150, n_chains=1) if args.smoke else \
        dict(n_warmup=600, n_samples=1200, n_chains=4)
    if args.warmup:
        kw["n_warmup"] = args.warmup
    if args.samples:
        kw["n_samples"] = args.samples
    if args.chains:
        kw["n_chains"] = args.chains

    stages = ["a", "b"] if args.stage == "both" else [args.stage]
    seeds = ([args.seed] if not args.seeds
             else [int(s) for s in args.seeds.split(",") if s.strip()])
    truth_override = dict(TRUTH_SETS[args.truth_set])
    if args.truth:
        for kv in args.truth.split(","):
            k, _, v = kv.partition("=")
            truth_override[k.strip()] = float(v)
    print(f"真值组：{ {k: v for k, v in truth_override.items()} }")
    t_all = time.time()
    recs = []
    for st in stages:
        for sd in seeds:
            sp = None
            if args.save_samples and args.json:
                sp = str(Path(args.json).with_suffix(f".{st}.s{sd}.samples.npz"))
            rec = run_stage(st, seed=sd, noise=args.noise, branch=args.branch,
                            layers=args.layers, progress=args.progress,
                            init_mode=args.init, target_accept=args.target_accept,
                            max_tree_depth=args.max_tree_depth, samples_path=sp,
                            truth_override=truth_override, **kw)
            print_record(rec)
            recs.append(rec)

    summary = robustness = None
    if len(seeds) > 1 and len(stages) == 1:
        summary = print_seed_table(recs)
        robustness = aggregate_seeds(recs)
        print_robustness(robustness)
    elif len(seeds) > 1:
        for st in stages:
            sub = [r for r in recs if r["stage"] == st]
            summary = print_seed_table(sub)
            robustness = aggregate_seeds(sub)
            print_robustness(robustness)

    ok = all(r["acceptance"]["pass"] for r in recs) if not args.smoke else True
    print(f"\n{'='*74}")
    if args.smoke:
        print("冒烟结束：只证明链路通，结果不可引用（1 链 150 样本不收敛）")
    else:
        print("两组都通过" if ok else "有未通过项——按 §8 记录，失败不删不改 seed")
    if len(recs) > 1:
        print(f"{len(recs)} 次拟合总耗时 {time.time()-t_all:.0f}s")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "smoke": bool(args.smoke),
                        "config": {"branch": args.branch, "layers": args.layers,
                                   "noise": args.noise, "seeds": seeds,
                                   "truth_set": args.truth_set,
                                   "truth": truth_override,
                                   "n_chains": kw["n_chains"],
                                   "n_samples": kw["n_samples"],
                                   "n_warmup": kw["n_warmup"],
                                   "target_accept": args.target_accept,
                                   "max_tree_depth": args.max_tree_depth,
                                   "init": args.init},
                        "summary": summary, "robustness": robustness,
                        "records": recs},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"记录写入 {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
