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

F_GRID = (1e-2, 1e5, 61)


SERIES_TRUTH = dict(R_L=800.0, tau_L=1.625)     # L_ad = 1300，与 parallel 支路可比


def truth_params(branch: str, layers: int) -> dict:
    """当前拓扑的真值（只含该拓扑真正采样的参数）。"""
    p = dict(TRUTH)
    if branch == "series":
        p.pop("R_ad", None)
        p.pop("tau_ad", None)
        p.update(SERIES_TRUTH)
    return {k: p[k] for k in T.sampled_keys(branch, layers) if k in p}


def make_spectrum(noise: float = 0.02, seed: int = 0, branch: str = "parallel",
                  layers: int = 2):
    """独立生成 + 加噪。返回 (f, Z, zmax)。"""
    f = np.logspace(np.log10(F_GRID[0]), np.log10(F_GRID[1]), F_GRID[2])
    Z0 = G.gen(f, truth_params(branch, layers), branch=branch, layers=layers)
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
              max_tree_depth: int = 10, samples_path: str | None = None) -> dict:
    f, Z, zmax = make_spectrum(noise=noise, seed=seed, branch=branch, layers=layers)
    truth = truth_params(branch, layers)

    sigma = (noise * zmax, noise * zmax) if stage == "a" else None
    if init_mode == "oracle":
        # 真值经物理 → 无约束转换。只用于"先验/初值是否卡住采样器"的诊断，
        # **不能**当正式结果：链从真值出发时，"收敛得好"可能只是初值的回声。
        init = T.physical_to_unconstrained(
            truth, zmax=zmax, w_ref=float(2 * np.pi * np.sqrt(f.min() * f.max())),
            branch=branch, layers=layers)
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M3 档 2 单组参数恢复")
    ap.add_argument("--stage", choices=["a", "b", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.02)
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
    recs = []
    for st in stages:
        sp = None
        if args.save_samples and args.json:
            sp = str(Path(args.json).with_suffix(f".{st}.samples.npz"))
        rec = run_stage(st, seed=args.seed, noise=args.noise, branch=args.branch,
                        layers=args.layers, progress=args.progress,
                        init_mode=args.init, target_accept=args.target_accept,
                        max_tree_depth=args.max_tree_depth, samples_path=sp, **kw)
        print_record(rec)
        recs.append(rec)

    ok = all(r["acceptance"]["pass"] for r in recs) if not args.smoke else True
    print(f"\n{'='*74}")
    if args.smoke:
        print("冒烟结束：只证明链路通，结果不可引用（1 链 150 样本不收敛）")
    else:
        print("两组都通过" if ok else "有未通过项——按 §8 记录，失败不删不改 seed")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "smoke": bool(args.smoke), "records": recs},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"记录写入 {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
