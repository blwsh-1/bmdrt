# -*- coding: utf-8 -*-
"""§18 执行器：合成谱判别实验（测量/数据 vs 代码）。

口径全部来自 `M6_VALIDATION_PLAN.md` §18（跑前预注册）与 §14 执行器，
脚本里不另立常数；改任何判据都等于改预注册。

**边界**：只读探针。不改生产代码、不改族定义/阈值/验收集，不重跑 §13/§14/§17，
不接 `bmdrt.py`；结果一律 `exploratory`，合成谱是 `inferred` 级构造数据。

三组（§18.2）：
  G 干净          ：2% 逐点噪声，T1×3 seed + T2a/b/c 各×3 seed = 12 条
  N 随机噪声加倍  ：4% 逐点噪声，T2a/b/c 各 seed0 = 3 条
  H 端部系统偏差  ：G(seed0) + 最高 3 点注入 +Im 系统偏差 = 3 条

生成用独立实现 `bm_top_gen.gen`，拟合用 `bm_top.fit_top`；
跑前先逐点对拍，口径不一致整节作废（§18.3 规则 4）。

用法（项目 venv D:\\autoeis_env\\Scripts\\python.exe）：
    python evidence/nested_topology/m4_synth_probe.py --selftest
    python evidence/nested_topology/m4_synth_probe.py --dry
    python evidence/nested_topology/m4_synth_probe.py --smoke
    python evidence/nested_topology/m4_synth_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bm_core as C                     # noqa: E402
import bm_top as T                      # noqa: E402
import bm_top_gen as G                  # noqa: E402
from m4_holdout2 import (BUDGET, SIGMA_PCT, RMS_MAX, ACF_MAX,  # noqa: E402
                         ACF_DROP, RHAT_MAX, cut_top, _metrics)

# ---- §18 冻结常数 ------------------------------------------------------------
FOUR_JSON = os.path.join("out", "m4_four_circuits.json")
OUT_JSON = os.path.join("out", "m4_synth_probe.json")
T2_SPECS = ("AZ31-8Li_50", "AZ31-8Li_5", "AZ31-4Li_0")   # 已过 / 临界 / 未过
SEEDS = (0, 1, 2)
NOISE_G = 0.02
NOISE_N = 0.04
HF_INJECT = 0.15
HF_TOP_N = 3
BRANCH, LAYERS = "parallel", 2
# -----------------------------------------------------------------------------


def selftest() -> dict:
    """§18.3 规则 4：独立生成器 vs 拟合器逐点对拍（防止把口径差读成代码 bug）。"""
    f = G.real_grid("AZ31")
    p = dict(G.BASE)
    zg = G.gen(f, p, branch=BRANCH, layers=LAYERS)
    zt = T.z_nested(f, p, branch=BRANCH, layers=LAYERS)
    e = float(np.max(np.abs(zg - zt)) / np.max(np.abs(zt)))
    print(f"[selftest] bm_top_gen.gen vs bm_top.z_nested：max rel {e:.3e}"
          f"（须 <1e-10）；{f.size} 点 {f.min():g}–{f.max():g} Hz")
    ok = e < 1e-10
    print("[selftest] " + ("通过" if ok else "失败：口径不一致，整节作废"))
    return dict(max_rel=e, n_points=int(f.size), ok=bool(ok))


def truth_sets() -> dict:
    """T1 = M3 冻结真值；T2a/b/c = §17 真实谱孪生的 C4 LS 中位参数。"""
    d = json.load(open(FOUR_JSON, encoding="utf-8"))
    rows = {r["spec"]: r for r in d["rows"]}
    out = {"T1": dict(G.BASE)}
    for i, sp in enumerate(T2_SPECS):
        if sp not in rows:
            raise KeyError(f"{FOUR_JSON} 里没有 {sp}")
        p = dict(rows[sp]["circuits"]["C4"]["params"])
        # §17 的 C4 按文献 Table 1 命名为 (R3, L)；`bm_top_gen` / `bm_top`
        # 用 (R_ad, tau_ad=L/R_ad)。§17 自检已证两者逐点等价（3.7e-16），
        # 这里只做命名映射，不改任何数值。
        p["R_ad"] = p.pop("R3")
        p["tau_ad"] = p.pop("L") / p["R_ad"]
        out[f"T2{'abc'[i]}"] = p
    return out


def build_cases() -> list[dict]:
    """先于结果把三组 18 条全部列出来（`--dry` 可直接看清单）。"""
    ts = truth_sets()
    f = G.real_grid("AZ31")
    cases = []
    for tag, p in ts.items():
        z_true = G.gen(f, p, branch=BRANCH, layers=LAYERS)
        for sd in SEEDS:
            cases.append(dict(group="G", truth=tag, seed=sd, frac=NOISE_G,
                              inject=False, f=f, z_true=z_true, p=p))
        if tag != "T1":
            cases.append(dict(group="N", truth=tag, seed=0, frac=NOISE_N,
                              inject=False, f=f, z_true=z_true, p=p))
            cases.append(dict(group="H", truth=tag, seed=0, frac=NOISE_G,
                              inject=True, f=f, z_true=z_true, p=p))
    return cases


def make_observed(c: dict) -> tuple[np.ndarray, np.ndarray]:
    """噪声 + 可选的端部系统偏差（§18.2 的构造式，原样实现）。"""
    Z = G.add_noise(c["z_true"], c["frac"], mode="relative", seed=c["seed"])
    if c["inject"]:
        f, z_true = c["f"], c["z_true"]
        top = np.argsort(f)[-HF_TOP_N:]           # 频率升序
        Z = Z.astype(complex).copy()
        for k, j in enumerate(top, start=1):
            Z[j] = Z[j] + 1j * HF_INJECT * np.abs(z_true[j]) * (k / HF_TOP_N)
    return c["f"], Z


def fit_one(c: dict, Z_override: np.ndarray | None = None) -> dict:
    """单条合成谱：§14 同口径 nested + voigt 基线，报 M-a/M-b/M-c。

    `Z_override`（§19 新增，硬边界 2）：外部构造好的观测谱。给定时跳过
    `make_observed`，其余流程逐位不变；**默认 None 时与 §18 完全同一路径**，
    由 §19 的 `--smoke` 回归验证守着，不许"差不多就行"。
    """
    if Z_override is None:
        f_all, Z_all = make_observed(c)
    else:
        f_all, Z_all = c["f"], np.asarray(Z_override, complex)
    cut = cut_top(f_all, Z_all)
    f, Z = f_all[cut["keep"]], Z_all[cut["keep"]]
    sg = SIGMA_PCT / 100.0 * np.abs(Z)            # 拟合 σ 用观测谱（N 组故意不匹配）
    sigma_top = np.stack([sg, sg])
    t0 = time.time()

    o = T.fit_top(f, Z, branch=BRANCH, layers=LAYERS, seed=0,
                  sigma=sigma_top, sigma_from_data=True,
                  progress=False, **BUDGET)
    keys = list(T.sampled_keys(BRANCH, LAYERS))
    p = {k: float(np.median(np.asarray(o["s"][k], float).reshape(-1))) for k in keys}
    Zm = T.z_nested(f, p, branch=BRANCH, layers=LAYERS)
    m = _metrics(f, Z, Zm)

    sv, _mc = C.fit(f, Z, K=1, alpha=0.5, sigma=(sg, sg), zeta_fixed=1.0,
                    seed=0, progress=False, **BUDGET)
    pv = dict(R_inf=float(np.median(np.asarray(sv["R_inf"], float))),
              W=float(np.median(np.asarray(sv["W"], float))),
              pi=np.median(np.asarray(sv["pi"], float), axis=0),
              tau=np.median(np.asarray(sv["tau"], float), axis=0),
              phi=np.median(np.asarray(sv["phi"], float), axis=0),
              zeta=np.median(np.asarray(sv["zeta"], float), axis=0))
    base = _metrics(f, Z, C.model_z(f, pv, include_L=False))

    rhat, div = o.get("rhat_max"), int(o.get("divergences") or 0)
    Ma = bool(m["rms_rel"] <= RMS_MAX)
    Mb = bool((m["acf"] <= ACF_MAX) and (base["acf"] - m["acf"] >= ACF_DROP))
    Mc = bool(rhat is not None and rhat <= RHAT_MAX and div == 0)
    return dict(group=c["group"], truth=c["truth"], seed=c["seed"],
                noise_frac=c["frac"], inject=bool(c["inject"]),
                n_pts=int(f_all.size), n_kept=int(f.size),
                dropped_hz=cut["dropped"],
                rms_rel=m["rms_rel"], acf=m["acf"], acf_re=m["acf_re"],
                acf_im=m["acf_im"], acf_mag=m["acf_mag"],
                hf_band=m["bands"][2], voigt_acf=base["acf"],
                drop=float(base["acf"] - m["acf"]),
                rhat_max=rhat, divergences=div,
                Ma=Ma, Mb=Mb, Mc=Mc, passed=bool(Ma and Mb and Mc),
                seconds=round(time.time() - t0, 1),
                truth_params={k: float(v) for k, v in c["p"].items()
                              if k in keys})


def summarize(rows: list[dict]) -> dict:
    """§18.3 规则 1/2/3 的机械判读（只报，不改任何验收判定）。"""
    out = {}
    for grp in ("G", "N", "H"):
        rr = [r for r in rows if r["group"] == grp]
        if not rr:
            continue
        acfs = [r["acf"] for r in rr]
        nb = sum(r["Mb"] for r in rr)
        rec = dict(n=len(rr), Ma=sum(r["Ma"] for r in rr), Mb=nb,
                   Mc=sum(r["Mc"] for r in rr),
                   passed=sum(r["passed"] for r in rr),
                   acf_median=float(np.median(acfs)),
                   acf_min=float(np.min(acfs)), acf_max=float(np.max(acfs)),
                   rms_median=float(np.median([r["rms_rel"] for r in rr])),
                   mean_hf=float(np.mean([r["hf_band"] for r in rr])))
        if grp == "G":
            if len(rr) < 12:
                rec["rule"] = "n<12：规则不适用（smoke / 部分跑），不判读"
            elif nb >= 10 and rec["acf_median"] <= ACF_MAX:
                rec["rule"] = "A：代码/判据在合规合成数据上可通过 → 真实谱 acf 指向数据层"
            elif nb <= 6:
                rec["rule"] = "B：干净合成谱也大面积不过 → 须查代码/判据"
            else:
                rec["rule"] = "C：indeterminate（7–9/12），需追加 seed，本节不改判定"
        out[grp] = rec
    # 同真值同 seed 的配对差（N/H 相对 G）
    gmap = {(r["truth"], r["seed"]): r["acf"] for r in rows if r["group"] == "G"}
    for grp in ("N", "H"):
        pairs = [(r["truth"], r["acf"] - gmap.get((r["truth"], r["seed"]), float("nan")))
                 for r in rows if r["group"] == grp]
        if grp in out and pairs:
            out[grp]["paired_dacf"] = [[t, float(d)] for t, d in pairs]
            out[grp]["paired_mean"] = float(np.mean([d for _, d in pairs]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="§18 合成谱判别实验（exploratory）")
    ap.add_argument("--groups", nargs="+", default=["G", "N", "H"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--json", default=OUT_JSON)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="只跑 G/T1/seed0 一条验证链路")
    a = ap.parse_args(argv)

    st = selftest()
    if a.selftest:
        return 0 if st["ok"] else 1
    if not st["ok"]:
        print("§18.3 规则 4：对拍失败，整节作废", file=sys.stderr)
        return 1

    cases = build_cases()
    if a.smoke:
        cases = [c for c in cases if c["group"] == "G" and c["truth"] == "T1"
                 and c["seed"] == 0]
    cases = [c for c in cases if c["group"] in a.groups and c["seed"] in a.seeds]

    print(f"§18 口径：{BUDGET}  σ_fit={SIGMA_PCT:g}%·|Z_obs|  "
          f"M-a rms≤{RMS_MAX:g}%  M-b acf≤{ACF_MAX:g}且比 voigt 低≥{ACF_DROP:g}")
    print(f"共 {len(cases)} 条：",
          {g: sum(c["group"] == g for c in cases) for g in ("G", "N", "H")},
          flush=True)
    if a.dry:
        for c in cases:
            print(f"  {c['group']}/{c['truth']}/seed{c['seed']}  "
                  f"noise={c['frac']:.0%}  inject={c['inject']}")
        return 0

    rows, t_all = [], time.time()
    for i, c in enumerate(cases, 1):
        tag = f"{c['group']}/{c['truth']}/s{c['seed']}"
        try:
            r = fit_one(c)
            rows.append(r)
            print(f"[{i:2d}/{len(cases)}] {tag:<18} rms {r['rms_rel']:6.2f}%  "
                  f"acf {r['acf']:.3f}  voigt {r['voigt_acf']:.3f}  "
                  f"drop {r['drop']:+.3f}  R̂ {r['rhat_max']:.3f}  div {r['divergences']}  "
                  f"M-a{'✓' if r['Ma'] else '✗'} M-b{'✓' if r['Mb'] else '✗'} "
                  f"M-c{'✓' if r['Mc'] else '✗'}  [{r['seconds']}s]", flush=True)
        except Exception as exc:
            print(f"[{i:2d}/{len(cases)}] {tag} 失败：{type(exc).__name__}: {exc}",
                  flush=True)
            rows.append(dict(group=c["group"], truth=c["truth"], seed=c["seed"],
                             error=f"{type(exc).__name__}: {exc}"))

    good = [r for r in rows if "acf" in r]
    summary = summarize(good)
    print("\n=== §18 汇总 ===")
    for g, s in summary.items():
        print(f"  {g}: M-a {s['Ma']}/{s['n']}  M-b {s['Mb']}/{s['n']}  "
              f"M-c {s['Mc']}/{s['n']}  三条全过 {s['passed']}/{s['n']}  "
              f"acf 中位 {s['acf_median']:.3f} [{s['acf_min']:.3f},{s['acf_max']:.3f}]  "
              f"rms 中位 {s['rms_median']:.2f}%")
        if "rule" in s:
            print(f"     规则判定：{s['rule']}")
        if "paired_mean" in s:
            print(f"     同真值同 seed 相对 G 的 Δacf 均值 {s['paired_mean']:+.3f}")
    print(f"总耗时 {time.time() - t_all:.1f}s")

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(prereg="M6_VALIDATION_PLAN.md §18",
                           status="exploratory", selftest=st,
                           constants=dict(budget=BUDGET, sigma_pct=SIGMA_PCT,
                                          noise_g=NOISE_G, noise_n=NOISE_N,
                                          hf_inject=HF_INJECT, hf_top_n=HF_TOP_N,
                                          t2_specs=list(T2_SPECS)),
                           rows=rows, summary=summary), fh,
                      ensure_ascii=False, indent=1)
        print(f"已写 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
