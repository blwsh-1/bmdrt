"""留出谱验收（`M6_VALIDATION_PLAN.md` §13 的执行器）。

**口径与阈值全部读 §13，脚本里不另立**：σ_j = 2%·|Z_j|、`standard` 档
（2000/1000×2）+ ta=0.98、判据 N-a..N-d、留出谱三条写死。
改这里任何一个常数都等于改预注册，必须在 §13.7 记一条。

用法
----
    python evidence/nested_topology/m4_holdout.py                # 三条全部
    python evidence/nested_topology/m4_holdout.py --specs AZ31-8Li_100
    python evidence/nested_topology/m4_holdout.py --dry           # 只打印口径

结果写 `out/m4_holdout.json`（配置进 JSON，§9 第 4 条）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bm_core as B          # noqa: E402
import bm_experiments as E   # noqa: E402
import bm_real as R          # noqa: E402
import bm_top as T           # noqa: E402

# ---- §13 冻结常数：改动 = 改预注册 -------------------------------------------
SPEC_DIR = os.path.join("out", "_m6_samples", "z_csv")
HOLDOUT = ("AZ31-4Li_150", "AZ31-8Li_100", "AZ31-12Li_100")
SIGMA_PCT = 2.0                 # §13.1 N1
BUDGET = dict(n_warmup=2000, n_samples=1000, num_chains=2, target_accept=0.98)  # §13.2 N2
RMS_MAX = 10.0                  # N-a（%）
ACF_MAX = 0.50                  # N-b
ACF_DROP = 0.20                 # N-b（相对同口径 voigt）
RHAT_MAX = 1.05                 # N-c
BRANCH, LAYERS = "parallel", 2  # 4Li_100 上 series 差 4 倍，series 不参赛
# -----------------------------------------------------------------------------


def _metrics(f, Z, Zm):
    w = np.abs(Z)
    rel = np.abs(Zm - Z) / w
    d = (Zm - Z) / w
    cost = float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2))) * 100
    bands = []
    for lo, hi in ((0.01, 1), (1, 100), (100, 1e5)):
        m = (f >= lo) & (f < hi)
        bands.append(float(np.sqrt(np.mean(np.abs(d)[m] ** 2))) * 100 if m.any() else None)
    return dict(rms_rel=cost, bands=bands, acf_max=float(E.acf_max(rel)),
                rms_re=float(np.sqrt(np.mean((Zm - Z).real ** 2))),
                rms_im=float(np.sqrt(np.mean((Zm - Z).imag ** 2))))


def _corr_pairs(ch: dict, names: list[str], thr: float = 0.90) -> list[dict]:
    # 必须先摊成 (n_draw_total,) 再取列：`(chain, draw)` 直接 `column_stack`
    # 会得到 (chain, draw*k) 的二维拼贴，`corrcoef` 当成 k 个变量 × 2 个观测，
    # 必然给出 ±1.000（第一版就这么错了，见 §9.4.23 的更正）。
    cols = [np.log(np.asarray(ch[k], float).reshape(-1)) for k in names]
    M = np.column_stack(cols)
    C = np.corrcoef(M, rowvar=False)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(C[i, j]) > thr:
                out.append(dict(pair=[names[i], names[j]], rho=float(C[i, j])))
    return sorted(out, key=lambda d: -abs(d["rho"]))


def one(spec: str, seeds: list[int]) -> dict:
    path = os.path.join(SPEC_DIR, spec + ".csv")
    f, Z = R.load_spectrum(path)
    sg = (SIGMA_PCT / 100.0 * np.abs(Z), SIGMA_PCT / 100.0 * np.abs(Z))
    w = np.abs(Z)
    rows = []
    for sd in seeds:
        o = T.fit_top(f, Z, branch=BRANCH, layers=LAYERS, seed=sd, sigma=sg,
                      progress=False, **BUDGET)
        names = [k for k in T.sampled_keys(BRANCH, LAYERS)]
        pm = {k: float(np.median(o["s"][k])) for k in names}
        Zm = T.z_nested(f, pm, branch=BRANCH, layers=LAYERS)
        met = _metrics(f, Z, Zm)
        ci = {k: [float(np.percentile(o["s"][k], 5)), float(np.percentile(o["s"][k], 95))]
              for k in names}
        po = o["idata"].posterior
        ch = {k: np.asarray(po[k].values) for k in names}
        rows.append(dict(seed=sd, **met, params=pm, ci90=ci,
                         rhat_max=float(o["rhat_max"]), div=int(o["divergences"] or 0),
                         rs_plus_r1=[float(np.percentile(o["s"]["Rs"] + o["s"]["R1"], q))
                                     for q in (5, 50, 95)],
                         degenerate=_corr_pairs(ch, names)))

    # 同口径 voigt 基线（§13.3 N-b 只作相对参照，不单独设线）
    s, mc = B.fit(f, Z, K=1, alpha=0.5, sigma=sg, zeta_fixed=1.0, seed=0,
                  progress=False, **BUDGET)
    p = dict(R_inf=float(np.median(s["R_inf"])), W=float(np.median(s["W"])),
             pi=np.median(s["pi"], axis=0), tau=np.median(s["tau"], axis=0),
             phi=np.median(s["phi"], axis=0), zeta=np.median(s["zeta"], axis=0))
    base = _metrics(f, Z, B.model_z(f, p, include_L=False))
    from numpyro.diagnostics import summary
    sm = summary(mc.get_samples(group_by_chain=True), prob=0.9)
    base["rhat_max"] = max(float(np.max(np.asarray(v["r_hat"]))) for v in sm.values())
    base["div"] = int(np.sum(mc.get_extra_fields()["diverging"]))
    base["K"] = 1

    # 判定：逐 seed 全过才算过（N-c 是逐 seed 的）
    verdict = []
    for r in rows:
        na = r["rms_rel"] <= RMS_MAX
        nb = (r["acf_max"] <= ACF_MAX) and (base["acf_max"] - r["acf_max"] >= ACF_DROP)
        nc = (r["rhat_max"] <= RHAT_MAX) and (r["div"] == 0)
        nd = True   # 报告义务：区间与简并对已落 JSON，见 §13.3 N-d
        verdict.append(dict(seed=r["seed"], Na=bool(na), Nb=bool(nb),
                            Nc=bool(nc), Nd=nd, passed=bool(na and nb and nc and nd)))
    return dict(spec=spec, n_pts=int(f.size), f_range=[float(f.min()), float(f.max())],
                sigma_pct=SIGMA_PCT, budget=BUDGET, branch=BRANCH, layers=LAYERS,
                seeds=seeds, rows=rows, voigt_base=base,
                verdict=verdict, passed_all=all(v["passed"] for v in verdict))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=list(HOLDOUT))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--json", default=os.path.join("out", "m4_holdout.json"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)

    print(f"§13 口径：σ={SIGMA_PCT:g}%·|Z| 逐点  {BUDGET}  "
          f"N-a rms≤{RMS_MAX:g}%  N-b acf≤{ACF_MAX:g} 且比 voigt 低≥{ACF_DROP:g}  "
          f"N-c R̂≤{RHAT_MAX:g} 分支 {BRANCH}/L={LAYERS}")
    if a.dry:
        print("留出谱：", ", ".join(a.specs))
        return 0

    res = []
    for sp in a.specs:
        print(f"\n=== {sp} ===")
        try:
            r = one(sp, a.seeds)
        except Exception as e:                      # 谱缺失/点数不足要显式报，不跳过
            print(f"  失败：{type(e).__name__}: {e}")
            res.append(dict(spec=sp, error=f"{type(e).__name__}: {e}"))
            continue
        b = r["voigt_base"]
        print(f"  voigt K=1 同口径：rms {b['rms_rel']:6.2f}%  acf {b['acf_max']:.3f}  "
              f"R̂ {b['rhat_max']:.3f}  div {b['div']}")
        for row, v in zip(r["rows"], r["verdict"]):
            print(f"  seed{row['seed']} nested：rms {row['rms_rel']:6.2f}%  "
                  f"低/中/高 {row['bands'][0]:.1f}/{row['bands'][1]:.1f}/{row['bands'][2]:.1f}  "
                  f"acf {row['acf_max']:.3f}  R̂ {row['rhat_max']:.3f}  div {row['div']}  "
                  f"Rs+R1 {row['rs_plus_r1'][1]:.1f}")
            print(f"         N-a {'✓' if v['Na'] else '✗'}  N-b {'✓' if v['Nb'] else '✗'}  "
                  f"N-c {'✓' if v['Nc'] else '✗'}  → {'过' if v['passed'] else '未过'}"
                  + ("   简并对：" + ", ".join(f"{d['pair'][0]}-{d['pair'][1]} {d['rho']:.2f}"
                                              for d in row["degenerate"][:2])
                     if row["degenerate"] else ""))
        print(f"  本谱：{'**过**' if r['passed_all'] else '未过'}")
        res.append(r)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(prereg="M6_VALIDATION_PLAN.md §13", rows=res), fh,
                      ensure_ascii=False, indent=1)
        ok = [r.get("spec") for r in res if r.get("passed_all")]
        print(f"\n通过 {len(ok)}/{len(res)}：{ok}")
        print(f"已写 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
