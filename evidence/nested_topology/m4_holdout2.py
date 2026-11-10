"""§14 执行器：M1 高频剔点 + 14 条全量留出验收。

口径全部来自 `M6_VALIDATION_PLAN.md` §14，脚本里不另立常数；
改这里任何一个数都等于改预注册，必须在 §14.6 记一条。

与 §13 执行器（`m4_holdout.py`）的区别只有两处，都是 §14 明确规定的：
1. 先按 M1（纯数据判据 `n>2.0`，最多剔 2 点）剔掉窗口上端的不可靠点，再拟合；
2. 验收集改为**全部 14 条未跑过 nested 的谱**（不挑不换）。

用法
----
    python evidence/nested_topology/m4_holdout2.py --dry
    python evidence/nested_topology/m4_holdout2.py
    python evidence/nested_topology/m4_holdout2.py --specs AZ31-4Li_5 --seeds 0
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

# ---- §14 冻结常数 -----------------------------------------------------------
SPEC_DIR = os.path.join("out", "_m6_samples", "z_csv")
SET14 = ("AZ31-4Li_0", "AZ31-4Li_5", "AZ31-4Li_10", "AZ31-4Li_50",
         "AZ31-8Li_0", "AZ31-8Li_5", "AZ31-8Li_10", "AZ31-8Li_50", "AZ31-8Li_150",
         "AZ31-12Li_0", "AZ31-12Li_5", "AZ31-12Li_10", "AZ31-12Li_50", "AZ31-12Li_150")
N_CUT = 2.0            # §14.1
MAX_DROP = 2           # §14.1
SIGMA_PCT = 2.0        # §14.3（同 §13.1）
BUDGET = dict(n_warmup=2000, n_samples=1000, num_chains=2, target_accept=0.98)  # §14.3
RMS_MAX, ACF_MAX, ACF_DROP, RHAT_MAX = 10.0, 0.50, 0.20, 1.05   # §14.3
BRANCH, LAYERS = "parallel", 2
# -----------------------------------------------------------------------------


def cut_top(f: np.ndarray, Z: np.ndarray, n_cut: float = N_CUT,
            max_drop: int = MAX_DROP) -> dict:
    """§14.1：从最高频开始，按局部幂指数剔点。纯数据判据，不碰模型。

    返回 `keep`（布尔掩码）、`dropped`（被剔的频点）、`n_trace`（每轮判据值）。
    `-Im ≤ 0`（感抗段）时对数无定义 → 该轮不可判，**不剔**，记 None。
    """
    order = np.argsort(f)
    keep = np.ones(f.size, bool)
    dropped, trace = [], []
    for _ in range(int(max_drop)):
        idx = order[keep[order]]          # 升序保留点
        if idx.size < 4:
            break
        top = idx[-4:]                    # 最高 4 点 → 3 对
        im = -Z.imag[top]
        n_val = None
        if im[-2] > 0 and im[-1] > 0:     # 只用最高那一对判断
            n_val = float(np.log(im[-1] / im[-2]) / np.log(f[top[-1]] / f[top[-2]]))
        trace.append(n_val)
        if n_val is None or n_val <= n_cut:
            break
        keep[top[-1]] = False
        dropped.append(float(f[top[-1]]))
    return dict(keep=keep, dropped=dropped, n_trace=trace)


def _metrics(f, Z, Zm):
    w = np.abs(Z)
    res = Zm - Z
    rel = np.abs(res) / w
    d = res / w
    bands = []
    for lo, hi in ((0.01, 1), (1, 100), (100, 1e5)):
        m = (f >= lo) & (f < hi)
        bands.append(float(np.sqrt(np.mean(np.abs(d)[m] ** 2))) * 100 if m.any() else None)
    return dict(
        rms_rel=float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2))) * 100,
        bands=bands,
        # §14.3 M-b：项目口径，带符号残差取较坏值
        acf=float(max(E.acf_max(res.real), E.acf_max(res.imag))),
        acf_re=float(E.acf_max(res.real)), acf_im=float(E.acf_max(res.imag)),
        acf_mag=float(E.acf_max(rel)))


def _corr_pairs(ch: dict, names: list[str], thr: float = 0.90) -> list[dict]:
    cols = [np.log(np.asarray(ch[k], float).reshape(-1)) for k in names]
    C = np.corrcoef(np.column_stack(cols), rowvar=False)
    out = [dict(pair=[names[i], names[j]], rho=float(C[i, j]))
           for i in range(len(names)) for j in range(i + 1, len(names))
           if abs(C[i, j]) > thr]
    return sorted(out, key=lambda d: -abs(d["rho"]))


def _fit_set(f, Z, seeds: list[int]) -> dict:
    """在一组点上跑 nested + 同口径 voigt，返回逐 seed 指标与 M 判据。"""
    sg = (SIGMA_PCT / 100.0 * np.abs(Z), SIGMA_PCT / 100.0 * np.abs(Z))
    rows = []
    for sd in seeds:
        o = T.fit_top(f, Z, branch=BRANCH, layers=LAYERS, seed=sd, sigma=sg,
                      progress=False, **BUDGET)
        names = list(T.sampled_keys(BRANCH, LAYERS))
        pm = {k: float(np.median(o["s"][k])) for k in names}
        met = _metrics(f, Z, T.z_nested(f, pm, branch=BRANCH, layers=LAYERS))
        po = o["idata"].posterior
        ch = {k: np.asarray(po[k].values) for k in names}
        rows.append(dict(seed=sd, **met, params=pm,
                         ci90={k: [float(np.percentile(o["s"][k], 5)),
                                   float(np.percentile(o["s"][k], 95))] for k in names},
                         rhat_max=float(o["rhat_max"]), div=int(o["divergences"] or 0),
                         degenerate=_corr_pairs(ch, names)))

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

    verdict = []
    for r in rows:
        na = r["rms_rel"] <= RMS_MAX
        nb = (r["acf"] <= ACF_MAX) and (base["acf"] - r["acf"] >= ACF_DROP)
        nc = (r["rhat_max"] <= RHAT_MAX) and (r["div"] == 0)
        verdict.append(dict(seed=r["seed"], Ma=bool(na), Mb=bool(nb), Mc=bool(nc),
                            Md=True, passed=bool(na and nb and nc)))
    return dict(rows=rows, voigt_base=base, verdict=verdict)


def one(spec: str, seeds: list[int]) -> dict:
    f_all, Z_all = R.load_spectrum(os.path.join(SPEC_DIR, spec + ".csv"))
    c = cut_top(f_all, Z_all)
    keep = c["keep"]
    cut = _fit_set(f_all[keep], Z_all[keep], seeds)

    # §14.4b M4b：并报「未剔点」结果。若 M1 一个点都没剔，两套的 (f,Z,seed,模型)
    # 完全相同 → 同 seed 拟合逐位可复现（本项目已实测），无需重跑，标记 identical。
    if c["dropped"]:
        nocut = _fit_set(f_all, Z_all, seeds)
        nocut["identical"] = False
    else:
        nocut = dict(rows=cut["rows"], voigt_base=cut["voigt_base"],
                     verdict=cut["verdict"], identical=True)

    # 两套都过才算过；一套过一套不过 = 口径敏感，不计为过（§14.4b）
    cut_ok = all(v["passed"] for v in cut["verdict"])
    no_ok = all(v["passed"] for v in nocut["verdict"])
    state = "过" if (cut_ok and no_ok) else ("未过" if not (cut_ok or no_ok)
                                            else "口径敏感(cut_sensitive)")
    return dict(spec=spec, n_pts=int(f_all.size), n_kept=int(keep.sum()),
                dropped_hz=c["dropped"], n_trace=c["n_trace"],
                sigma_pct=SIGMA_PCT, budget=BUDGET, branch=BRANCH, layers=LAYERS,
                seeds=seeds, rows=cut["rows"], voigt_base=cut["voigt_base"],
                verdict=cut["verdict"], nocut=dict(
                    rows=[] if nocut["identical"] else nocut["rows"],
                    voigt_base=nocut["voigt_base"], verdict=nocut["verdict"],
                    identical=nocut["identical"]),
                cut_ok=bool(cut_ok), nocut_ok=bool(no_ok),
                cut_sensitive=bool(cut_ok != no_ok), state=state,
                passed_all=bool(cut_ok and no_ok))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=list(SET14))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--json", default=os.path.join("out", "m4_holdout2.json"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)

    print(f"§14 口径：σ={SIGMA_PCT:g}%·|Z|  {BUDGET}")
    print(f"  M1 剔点 n_cut={N_CUT:g} 最多{MAX_DROP}点 | M-a rms≤{RMS_MAX:g}% "
          f"M-b acf≤{ACF_MAX:g}且比voigt低≥{ACF_DROP:g} M-c R̂≤{RHAT_MAX:g} | {BRANCH}/L={LAYERS}")
    if a.dry:
        for s in a.specs:
            f, Z = R.load_spectrum(os.path.join(SPEC_DIR, s + ".csv"))
            c = cut_top(f, Z)
            print(f"  {s:<14} 剔 {len(c['dropped'])} 点 {c['dropped']}  n_trace={c['n_trace']}")
        return 0

    res = []
    for sp in a.specs:
        print(f"\n=== {sp} ===", flush=True)
        try:
            r = one(sp, a.seeds)
        except Exception as e:
            print(f"  失败：{type(e).__name__}: {e}", flush=True)
            res.append(dict(spec=sp, error=f"{type(e).__name__}: {e}"))
            continue
        b = r["voigt_base"]
        print(f"  剔点 {len(r['dropped_hz'])} 个 {r['dropped_hz']}  n_trace={r['n_trace']}  "
              f"保留 {r['n_kept']}/{r['n_pts']}")
        print(f"  voigt K=1：rms {b['rms_rel']:6.2f}%  acf {b['acf']:.3f}  R̂ {b['rhat_max']:.3f}")
        for row, v in zip(r["rows"], r["verdict"]):
            print(f"  seed{row['seed']} nested：rms {row['rms_rel']:6.2f}%  "
                  f"低/中/高 {row['bands'][0]:.1f}/{row['bands'][1]:.1f}/{row['bands'][2]:.1f}  "
                  f"acf {row['acf']:.3f}  R̂ {row['rhat_max']:.3f}  div {row['div']}  "
                  f"RsR1 {row['params']['Rs'] + row['params']['R1']:.1f}"
                  f"  M-a {'✓' if v['Ma'] else '✗'} M-b {'✓' if v['Mb'] else '✗'} "
                  f"M-c {'✓' if v['Mc'] else '✗'} → {'过' if v['passed'] else '未过'}")
        print(f"  M4b 未剔点：{'与剔点版逐位相同（M1 未剔点）' if r['nocut']['identical'] else r['nocut']['verdict']}"
              f"  两套都过={r['cut_ok'] and r['nocut_ok']}  口径敏感={r['cut_sensitive']}")
        print(f"  本谱：{r['state']}", flush=True)
        res.append(r)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(prereg="M6_VALIDATION_PLAN.md §14", rows=res), fh,
                      ensure_ascii=False, indent=1)
        ok = [r.get("spec") for r in res if r.get("passed_all")]
        print(f"\n通过 {len(ok)}/{len(res)}：{ok}")
        print(f"已写 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
