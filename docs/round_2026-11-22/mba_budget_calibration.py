"""mba_budget_calibration.py —— M-b 外部验证前的预算标定（6 条 2E/3E）。

问题
----
用户问：quick 档够不够，要不要直接 standard。

已有证据（batch21 产物）：
- 6 条 2E/3E 的 nested layers=2 里，**3 条 R̂ > 1.05**（1.289 / 1.269 / 4.433）；
- 全局 15 条 layers=2 里 9 条 div>0 或 R̂>1.05，而 `ok` 字段竟然全是 True——
  它不检查采样质量；
- 采样失败谱的 acf 不可信（§14 的 `12Li_5`：未收敛谱 legacy acf 0.896 vs F1 0.713）。

⇒ quick 档的 acf 不能直接用。本脚本在 6 条 2E/3E 上把 quick 与 standard 并排跑，
确认 standard 是否真能把 R̂ 修回来，再决定正式验证的预算与规模。

口径边界
--------
`batch21` 的候选 acf 全部来自 `zmax_override = 5·|Z(f_max)|` 的 fixed 口径，
而 §14 用**默认**先验（不传 override）。两口径不可混比（先验定标已证 override
会同时缩放 W/Q/R1）。本脚本**两档都不传 override**，与 §14 一致。

用法
----
    python docs/round_2026-11-22/mba_budget_calibration.py
    python docs/round_2026-11-22/mba_budget_calibration.py --json out/mba_budget_calib.json
"""

import argparse
import contextlib
import io
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "docs", "round_2026-11-21"))

import numpy as np                    # noqa: E402
import bm_real as BR                  # noqa: E402
import circuit_runner as RU           # noqa: E402
import cmd_compare as C               # noqa: E402
from batch_fit_21 import SEED         # noqa: E402

BUDGETS = ("quick", "standard")


def pick_six():
    """batch21 里的 6 条 2E/3E 谱路径。"""
    d = json.load(open(os.path.join(REPO, "out", "batch21", "results.json"),
                       encoding="utf-8"))
    rows = d["rows"] if isinstance(d, dict) else d
    out = []
    for r in rows:
        if r["class"] in ("2E", "3E"):
            out.append((r["class"], r["path"]))
    return out


def run_one(path, budget):
    with contextlib.redirect_stdout(io.StringIO()):
        f, Z = BR.load_spectrum(path)
        inc_l = bool(C.need_L(f, Z))
        cands = RU.build_candidates(ks=(1,), layers=(2,), include_adr=False,
                                    include_L=inc_l, budget_profile=budget)
        # 默认先验口径：不传 zmax_override（与 §14 一致）
        voigt = [c for c in cands if c.topology_id == "voigt"][0]
        nest = [c for c in cands if c.topology_id == "nested"][0]
        res = {}
        for tag, cand in (("voigt", voigt), ("nested", nest)):
            o = RU.fit_candidate(cand, f, Z, seed=SEED,
                                 budget_profile=budget, likelihood="free")
            m = o["metrics"]
            res[tag] = dict(acf=float(m["acf"]),
                            rms=float(m["rms_rel"]),
                            rhat=float(o["rhat_max"]),
                            div=int(o["divergences"] or 0),
                            ess=float(o.get("ess_min") or 0))
    res["include_L"] = inc_l
    res["n_pts"] = int(f.size)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    cases = pick_six()
    print(f"预算标定：{len(cases)} 条 2E/3E × 2 档 × 2 族，"
          f"默认先验口径（与 §14 一致）", flush=True)
    rows = []
    for cls, path in cases:
        name = os.path.basename(path)
        rec = dict(cls=cls, name=name)
        for budget in BUDGETS:
            r = run_one(path, budget)
            rec[budget] = r
            v, n = r["voigt"], r["nested"]
            print(f"  {cls:<3} {name[:26]:<27} {budget:<9} "
                  f"voigt {v['acf']:.3f}/R̂{v['rhat']:.3f}/d{v['div']:<3}  "
                  f"nested {n['acf']:.3f}/R̂{n['rhat']:.3f}/d{n['div']:<3}",
                  flush=True)
        rows.append(rec)

    print()
    print("=" * 92)
    print(f"{'谱':<28}{'quick R̂(n/v)':>16}{'standard R̂(n/v)':>19}"
          f"{'修好?':>8}{'acf q→s':>16}")
    fixed = 0
    for r in rows:
        q, s = r["quick"], r["standard"]
        qr, qv = q["nested"]["rhat"], q["voigt"]["rhat"]
        sr, sv = s["nested"]["rhat"], s["voigt"]["rhat"]
        qbad = qr > 1.05 or qv > 1.05
        sgood = sr <= 1.05 and sv <= 1.05
        if qbad and sgood:
            fixed += 1
        print(f"{r['name'][:27]:<28}{qr:>7.3f}/{qv:<8.3f}"
              f"{sr:>10.3f}/{sv:<8.3f}"
              f"{('YES' if qbad and sgood else ('-' if not qbad else 'NO')):>8}"
              f"{q['nested']['acf']:>7.3f}→{s['nested']['acf']:<7.3f}")
    print("=" * 92)
    n_qbad = sum(1 for r in rows
                 if r["quick"]["nested"]["rhat"] > 1.05
                 or r["quick"]["voigt"]["rhat"] > 1.05)
    n_sgood = sum(1 for r in rows
                  if r["standard"]["nested"]["rhat"] <= 1.05
                  and r["standard"]["voigt"]["rhat"] <= 1.05)
    print(f"quick 采样不过 {n_qbad}/{len(rows)}；standard 采样达标 {n_sgood}/{len(rows)}"
          f"；quick 不过被 standard 修好 {fixed}/{n_qbad}")
    print()
    print("预算：quick =", RU.BUDGETS["quick"])
    print("      standard =", RU.BUDGETS["standard"])

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"budgets": {k: {kk: vv for kk, vv in v.items()
                                       if not isinstance(vv, tuple)}
                                   for k, v in RU.BUDGETS.items() if k in BUDGETS},
                       "prior": "default (no zmax_override)", "rows": rows},
                      fh, ensure_ascii=False, indent=1)
        print(f"已写 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
