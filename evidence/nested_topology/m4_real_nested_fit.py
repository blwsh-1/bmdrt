#!/usr/bin/env python
"""nested 族在真实谱上的首次拟合（验收 §9.3 的第三条）。

只做**诊断**：不改生产路径、不接注册表、不参与跨族选择。目标是把
`NESTED_TOPOLOGY_PROJECT.md` §9.3 那条从「未测」变成「有数」：

    `AZ31-4Li_100` 在 nested 族的相对残差 ≤10%（理论上限 6.7%），
    `acf` 从 0.909 明显下降。

口径（全部与既有代码同源，避免自造指标）：
  · `acf` = `max(acf_max(Re r), acf_max(Im r))`（`bm_real` 的算法），
    基准 0.909 来自 B3b 报告里 voigt 的那次拟合；
  · `rms_re` / `rms_im` 为后验**中位参数**复现曲线的残差 RMS（Ω），
    与 `bm_real.fit_one` 的 rms 同定义；
  · **相对残差**逐点算 `|Z_mod−Z_obs| / |Z_obs|`，报 max 与中位；
    验收线的 10% 对应 max；
  · guard 走 `bm_select.guard_state(family="nested")`，`not_applicable` 项照实列。

跑法：
    python evidence/nested_topology/m4_real_nested_fit.py \
        --json out/m4_real_nested_az31.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "evidence", "nested_topology")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bm_experiments as E                             # noqa: E402
import bm_select as S                                 # noqa: E402
import bm_top as T                                    # noqa: E402

SPEC = "out/_m6_samples/z_csv/AZ31-4Li_100.csv"        # 实验真值读 csv（内存纪律）
ACF_BASE = 0.909                                       # B3b：voigt 的残差 acf
REL_BAR = 0.10                                         # §9.3 验收线
REL_BOUND = 0.067                                      # 理论上限 6.7%


def median_params(s: dict, branch: str, layers: int) -> dict:
    """后验中位参数 → `z_nested` 要的 dict。

    `L_ad` 是派生量（`R_L·tau_L`），`s` 里没有，必须在这里补——否则
    `z_nested` 会在 `p["L_ad"]` 上 KeyError，而这正是 series 支路唯一的电感通路。
    """
    p = {k: float(np.median(np.asarray(v, float))) for k, v in s.items()
         if k in T.sampled_keys(branch, layers)}
    if branch == "series":
        p["L_ad"] = p["R_L"] * p["tau_L"]
    return p


def evaluate(f, Z, c: dict, branch: str, layers: int, n_draw: int) -> dict:
    raw = c["raw"]
    s = raw["s"]
    p = median_params(s, branch, layers)
    Zm = T.z_nested(f, p, branch=branch, layers=layers)
    r = Z - Zm
    rel = np.abs(r) / np.abs(Z)
    # 必须把 rhat/div/ok 一起传进去：`guard_state` 从 `c` 读这三个字段，
    # 只传 raw 会让 A 类三项一律记 `not_evaluable`（第一版就是这样误报的）。
    g = S.guard_state(dict(family="nested", raw=raw, rhat=c["rhat"],
                           div=c["div"], ok=c["ok"]), n_draw, float(f.max()))
    out = dict(
        label=c["label"], branch=branch, layers=int(layers),
        elpd=float(c["elpd"]), se=float(c["se"]), rhat=c["rhat"], div=c["div"],
        rms_re=float(np.sqrt(np.mean(r.real ** 2))),
        rms_im=float(np.sqrt(np.mean(r.imag ** 2))),
        acf=float(max(E.acf_max(r.real), E.acf_max(r.imag))),
        rel_max=float(rel.max()), rel_med=float(np.median(rel)),
        rel_max_at_Hz=float(f[int(np.argmax(rel))]),
        passes_rel=bool(rel.max() <= REL_BAR),
        acf_drop=float(ACF_BASE - max(E.acf_max(r.real), E.acf_max(r.imag))),
        params={k: round(v, 6) for k, v in p.items()},
        sigma_re=raw.get("sigma_re"), sigma_im=raw.get("sigma_im"),
        guards=g,
    )
    if branch == "parallel":
        Rad, tau = p["R_ad"], p["tau_ad"]
        out["L_ad_nH"] = float(Rad * tau * 1e9)
        out["f_ad_Hz"] = float(1.0 / (2 * np.pi * tau))
    else:
        out["L_ad_nH"] = float(p["L_ad"] * 1e9)
        out["f_ad_Hz"] = float(1.0 / (2 * np.pi * (p["L_ad"] / p["R_L"])))
    return out


def voigt_row(f, Z, K: int, kw: dict, n_draw: int) -> dict:
    """voigt 基线，指标与 nested **完全同一套**（否则「过没过线」没法比）。

    只看相对残差的最大值会误导：`|Z|` 跨 7 decade，高频那几点的 `|Z|` 只有几十 Ω，
    同样的绝对误差在那里放大成百分比。所以 `rel_max` 一定要**报出所在频点**，
    并且必须与 voigt 在**同一 f 网格**上比。
    """
    import bm_core as C
    c = S.fit_voigt(f, Z, K, kw=kw)
    s = c["raw"]["s"]
    med = lambda k: np.asarray(s[k], float)
    p = dict(R_inf=float(np.median(med("R_inf"))),
             W=float(np.median(med("W"))),
             pi=np.median(med("pi"), axis=0),
             tau=np.median(med("tau"), axis=0),
             phi=np.median(med("phi"), axis=0),
             zeta=np.median(med("zeta"), axis=0))
    Zm = C.model_z(f, p, include_L=False)
    r = Z - Zm
    rel = np.abs(r) / np.abs(Z)
    acf = float(max(E.acf_max(r.real), E.acf_max(r.imag)))
    return dict(label=f"voigt K={K}", branch=None, layers=K,
                elpd=float(c["elpd"]), se=float(c["se"]),
                rhat=c["rhat"], div=c["div"],
                rms_re=float(np.sqrt(np.mean(r.real ** 2))),
                rms_im=float(np.sqrt(np.mean(r.imag ** 2))),
                acf=acf, rel_max=float(rel.max()),
                rel_med=float(np.median(rel)),
                rel_max_at_Hz=float(f[int(np.argmax(rel))]),
                passes_rel=bool(rel.max() <= REL_BAR), acf_drop=float(ACF_BASE - acf),
                params={}, sigma_re=None, sigma_im=None,
                guards=S.guard_state(c, n_draw, float(f.max())))


def main(argv=None):
    ap = argparse.ArgumentParser(description="nested 真实谱诊断（只报，不接生产）")
    ap.add_argument("--spec", default=SPEC)
    ap.add_argument("--branches", default="parallel,series")
    ap.add_argument("--layers", default="1,2")
    ap.add_argument("--budget", choices=sorted(S.BUDGETS), default="quick")
    ap.add_argument("--with-voigt", action="store_true", default=True,
                    help="同时跑 voigt 基线（同一套指标，默认开）")
    ap.add_argument("--no-voigt", dest="with_voigt", action="store_false")
    ap.add_argument("--voigt-K", default="1,2", help="voigt 基线跑哪些 K")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    from bm_real import load_spectrum
    f, Z = load_spectrum(a.spec)
    kw = dict(S.BUDGETS[a.budget])
    n_draw = int(kw["num_chains"]) * int(kw["n_samples"])
    brs = [x.strip() for x in a.branches.split(",") if x.strip()]
    lys = [int(x) for x in a.layers.split(",") if x.strip()]

    print(f"谱 {os.path.basename(a.spec)}：{f.size} 点 {f.min():.3g}–{f.max():.3g} Hz，"
          f"|Z|max={np.abs(Z).max():.3g} Ω，Im(f_min)={Z.imag[0]:+.1f} Ω")
    print(f"预算 {a.budget}（{kw['n_warmup']}/{kw['n_samples']}/{kw['num_chains']}）"
          f"，候选 {len(brs) * len(lys)} 个；基准 acf={ACF_BASE}，验收 rel_max≤{REL_BAR:.0%}"
          f"（理论上限 {REL_BOUND:.1%}）\n")
    print(f"{'候选':<16}{'elpd':>9}{'R̂':>7}{'div':>5}{'rms_re':>9}{'rms_im':>9}"
          f"{'acf':>7}{'rel_max':>9}{'@Hz':>9}{'rel_med':>9}  验收")
    rows, t0 = [], time.time()
    for br in brs:
        for L in lys:
            c = S.fit_nested(f, Z, L, branch=br, kw=kw)
            r = evaluate(f, Z, c, br, L, n_draw)
            rows.append(r)
            print(f"{r['label']:<16}{r['elpd']:>9.1f}{r['rhat']:>7.3f}{r['div']:>5}"
                  f"{r['rms_re']:>9.1f}{r['rms_im']:>9.1f}{r['acf']:>7.3f}"
                  f"{r['rel_max']:>9.2%}{r['rel_max_at_Hz']:>9.3g}{r['rel_med']:>9.2%}"
                  f"  {'过线' if r['passes_rel'] else '未过'}  {time.time() - t0:.0f}s")

    if a.with_voigt:
        print()
        for K in [int(x) for x in a.voigt_K.split(",") if x.strip()]:
            r = voigt_row(f, Z, K, kw, n_draw)
            rows.append(r)
            print(f"{r['label']:<16}{r['elpd']:>9.1f}{r['rhat']:>7.3f}{r['div']:>5}"
                  f"{r['rms_re']:>9.1f}{r['rms_im']:>9.1f}{r['acf']:>7.3f}"
                  f"{r['rel_max']:>9.2%}{r['rel_max_at_Hz']:>9.3g}{r['rel_med']:>9.2%}"
                  f"  {'过线' if r['passes_rel'] else '未过'}  ←基线")

    ok = [r for r in rows if r["passes_rel"]]
    print(f"\n验收 §9.3：{'有候选过线' if ok else '无候选过线'}（{len(ok)}/{len(rows)}）")
    if ok:
        b = max(ok, key=lambda r: r["elpd"])
        print(f"  最优过线者 {b['label']}：rel_max={b['rel_max']:.2%}、"
              f"acf={b['acf']:.3f}（基准 {ACF_BASE}，降 {b['acf_drop']:+.3f}）")
    best = max(rows, key=lambda r: r["elpd"])
    print(f"  elpd 最优：{best['label']}（acf={best['acf']:.3f}，"
          f"rel_max={best['rel_max']:.2%}）")
    print("\n边界：本脚本只报数，不推断「界面上有几个过程」；nested 的形状参数"
          "（R1/R2）按 M3.5 结论仍是先验主导，别把它们的后验当物理量读。")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fp:
            json.dump(dict(spec=a.spec, n_points=int(f.size),
                           f_min=float(f.min()), f_max=float(f.max()),
                           budget=a.budget, acf_base=ACF_BASE,
                           rel_bar=REL_BAR, rel_bound=REL_BOUND,
                           rows=rows, elapsed_s=round(time.time() - t0, 1)),
                      fp, ensure_ascii=False, indent=1)
        print(f"\n结果已写入 {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
