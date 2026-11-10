#!/usr/bin/env python
"""真实谱上的跨族选择（第一次）：AZ31-4Li_100 到底会不会被判到 nested。

与 §9.4.19 的区别：那里是**强制**跑 nested 各候选再看残差；这里跑的是
`bm_select.select_across_families` —— 完整的选择器（两族候选池、配对 LOO、
guard 门禁、多 seed 聚合）。默认关 `--gate-comp`。

口径提醒（§9.4.1）：
  · 真实谱**只报**判定 + Δ/dse/z + 两族 guard 状态，**不设通过线**；
  · `indistinguishable` / `voigt_inadmissible` / `unstable` 都不等于「选到 nested」；
  · 判到 nested ≠ 证明界面就是嵌套拓扑（§3 边界）。

跑法：
    python evidence/nested_topology/m4_real_select.py --json out/m4_real_select_az31.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "evidence", "nested_topology")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bm_select as S                                 # noqa: E402

SPEC = "out/_m6_samples/z_csv/AZ31-4Li_100.csv"


def main(argv=None):
    ap = argparse.ArgumentParser(description="真实谱跨族选择（只报）")
    ap.add_argument("--spec", default=SPEC)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", choices=sorted(S.BUDGETS), default="quick")
    ap.add_argument("--pair-mode", choices=S.PAIR_MODES, default="pooled")
    ap.add_argument("--gate-comp", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    from bm_real import load_spectrum
    f, Z = load_spectrum(a.spec)
    print(f"谱 {os.path.basename(a.spec)}：{f.size} 点 {f.min():.3g}–{f.max():.3g} Hz，"
          f"|Z|max={np.abs(Z).max():.3g} Ω")
    ok, why = S.voigt_can_represent(f, Z)
    print(f"voigt 可表示性：{'可表示' if ok else '不可表示'}"
          + ("" if ok else f"（{why}）"))
    print(f"选择器：seeds={a.seeds}  budget={a.budget}  pair_mode={a.pair_mode}  "
          f"gate_comp={a.gate_comp}\n")

    r = S.select_across_families(f, Z, seed=1, n_seeds=a.seeds,
                                 branches=("parallel", "series"),
                                 pair_mode=a.pair_mode, strict=False,
                                 include_L=False, adsorption=None,
                                 kw=dict(S.BUDGETS[a.budget]),
                                 retry_kw=dict(S.BUDGETS["standard"]),
                                 gate_b=bool(a.gate_comp))

    print(f"{'seed':>5}{'decision':<20}{'Δ(n−v)':>10}{'dse':>9}{'z':>8}  "
          f"{'voigt':<14}{'nested':<14}降级")
    for x in r["runs"]:
        v = x["voigt"]["label"] if x["voigt"] else "-"
        n = x["nested"]["label"] if x["nested"] else "-"
        d = "-" if x["d_elpd"] is None else f"{x['d_elpd']:+.1f}"
        se = "-" if x["dse"] is None else f"{x['dse']:.1f}"
        z = "-" if x["z"] is None else f"{x['z']:+.1f}"
        print(f"{x['seed']:>5}{x['decision']:<20}{d:>10}{se:>9}{z:>8}  "
              f"{v:<14}{n:<14}{'是' if x['degraded'] else '否'}")
        if x["degraded"]:
            print(f"{'':>5}↳ {x['degraded_why']}")
        # K / k_eff 与 guard：说明「选到的这一族」是否真的被数据定住
        for c in (x["voigt"], x["nested"]):
            if c is None:
                continue
            extra = ""
            if c["family"] == "voigt":
                extra = f" k_eff={c.get('k_eff')}/{c.get('dof')}" \
                        f" comp={c.get('comp_state')}"
            print(f"{'':>5}  {c['label']:<14} elpd={c['elpd']:+.1f} "
                  f"R̂={c['rhat']:.3f} div={c['div']}{extra}")

    print(f"\n聚合判定：**{r['decision']}**  逐 seed={r['decisions']}")
    if r["d_elpd"] is not None:
        print(f"  Δelpd(nested−voigt)={r['d_elpd']:+.2f}  dse={r['dse']:.3f}  "
              f"z={r['z']:+.2f}（阈值 Δ≥{S.SWITCH_DELTA:g} 且 z≥{S.SWITCH_Z:g}）")
    if r["degraded_seeds"]:
        print(f"  降级 seed：{r['degraded_seeds']} → 命中的不是「结论」，是空白")
    print("\n边界：真实谱只报判定，不设通过线；判到 nested 也不构成物理机制证据"
          "（§9.4.1、§3 表述边界）。")

    if a.json:
        keep = dict(spec=a.spec, seeds=a.seeds, budget=a.budget,
                    pair_mode=a.pair_mode, gate_comp=bool(a.gate_comp),
                    voigt_can_represent=bool(ok), voigt_why=why,
                    decision=r["decision"], decisions=r["decisions"],
                    d_elpd=r["d_elpd"], dse=r["dse"], z=r["z"],
                    degraded_seeds=r["degraded_seeds"],
                    runs=[dict(seed=x["seed"], decision=x["decision"],
                               d_elpd=x["d_elpd"], dse=x["dse"], z=x["z"],
                               degraded=x["degraded"],
                               degraded_why=x["degraded_why"],
                               voigt=None if x["voigt"] is None else dict(
                                   label=x["voigt"]["label"], elpd=x["voigt"]["elpd"],
                                   rhat=x["voigt"]["rhat"], div=x["voigt"]["div"],
                                   k_eff=x["voigt"].get("k_eff"),
                                   comp_state=x["voigt"].get("comp_state")),
                               nested=None if x["nested"] is None else dict(
                                   label=x["nested"]["label"], elpd=x["nested"]["elpd"],
                                   rhat=x["nested"]["rhat"], div=x["nested"]["div"]))
                          for x in r["runs"]])
        with open(a.json, "w", encoding="utf-8") as fp:
            json.dump(keep, fp, ensure_ascii=False, indent=1)
        print(f"\n结果已写入 {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
