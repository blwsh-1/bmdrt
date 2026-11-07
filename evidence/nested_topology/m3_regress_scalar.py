# -*- coding: utf-8 -*-
"""标量噪声路径的回归测试（M3.5 改 `fit_top` 之前的冻结基线）。

为什么单独成一个脚本
--------------------
`bm_top` 自检 [12] 只能在**算术层面**证明"常数数组 ≡ 标量"（loglik 与 logp
逐位相同）。它证明不了"整条 NUTS 路径没被动过"——后者要求同环境、同依赖、
同 seed 重跑，再与改动前的记录对比。

冻结基线
--------
`g1_seeds_a.json` 的 seed 0 记录（M3 档 2 A 组，改动前写入）：
4 链 × 1200 样本、warmup 2000、`target_accept=0.95`、`max_tree_depth=12`、
初值 `prior`、固定噪声 `σ=2%·zmax`。

两种判据分开报（审查要求）
--------------------------
1. **逐位一致**：`rhat_max`、`ess_min` 与各参数后验分位数与本机基线完全相同。
   同机同版本同 seed 下 NUTS 应该是确定性的；不一致就说明计算图被改了。
2. **容差一致**：相对差 < 1e-9。跨平台/升级依赖后浮点会有末位差异，
   此时只有容差判据有解释力——**把浮点差异当成模型路径改变是误判**。

结论分三档：`exact`（逐位一致）/ `tolerance`（只在容差内一致）/ `fail`。

用法
----
    python evidence/nested_topology/m3_regress_scalar.py            # 跑一次对比
    python evidence/nested_topology/m3_regress_scalar.py --json x.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import m3_stage2 as S                                     # noqa: E402

# 冻结配置：与 g1_seeds_a.json 的 config 块逐项一致，改这里等于改基线
FROZEN = dict(seed=0, stage="a", noise=0.02, branch="parallel", layers=2,
              n_warmup=2000, n_samples=1200, n_chains=4,
              target_accept=0.95, max_tree_depth=12, init_mode="prior")
BASE_JSON = HERE / "g1_seeds_a.json"


def load_baseline(path: Path, seed: int) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    for r in d["records"]:
        if r["seed"] == seed and r["stage"] == FROZEN["stage"]:
            return r
    raise KeyError(f"{path} 里没有 seed={seed} stage={FROZEN['stage']} 的记录")


def compare(ref: dict, cur: dict, tol: float = 1e-9) -> dict:
    """逐参数比较中位数与 90% 分位；分开记'逐位'与'容差'两种一致。"""
    rows, worst_exact, worst_tol = [], True, 0.0
    for k, v in ref["params"].items():
        if k not in cur["params"]:
            rows.append({"param": k, "status": "missing"})
            worst_exact = False
            continue
        w = cur["params"][k]
        ds = {}
        for fld in ("median", "p05", "p95"):
            a, b = float(v[fld]), float(w[fld])
            rel = abs(a - b) / max(abs(a), 1e-300)
            ds[fld] = {"ref": a, "cur": b, "exact": a == b, "rel": rel}
            worst_exact &= (a == b)
            worst_tol = max(worst_tol, rel)
        rows.append({"param": k, "fields": ds})

    scalars = {}
    for key in ("rhat_max", "ess_min"):
        a, b = ref["diagnostics"][key], cur["diagnostics"][key]
        rel = abs(float(a) - float(b)) / max(abs(float(a)), 1e-300)
        scalars[key] = {"ref": float(a), "cur": float(b),
                        "exact": float(a) == float(b), "rel": rel}
        worst_exact &= (float(a) == float(b))
        worst_tol = max(worst_tol, rel)

    status = "exact" if worst_exact and worst_tol == 0.0 else (
        "tolerance" if worst_tol < tol else "fail")
    return {"status": status, "max_rel_diff": worst_tol, "tol": tol,
            "params": rows, "scalars": scalars}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(BASE_JSON))
    ap.add_argument("--json", default=None)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args(argv)

    ref = load_baseline(Path(args.baseline), FROZEN["seed"])
    print(f"基线：{args.baseline} seed={FROZEN['seed']} stage={FROZEN['stage']}")
    print(f"      配置 {FROZEN['n_chains']} 链 × {FROZEN['n_samples']} / "
          f"warmup {FROZEN['n_warmup']}  ta={FROZEN['target_accept']}  "
          f"depth={FROZEN['max_tree_depth']}")
    print("      重跑中（约 1–2 分钟）…", flush=True)

    cur = S.run_stage(FROZEN["stage"], seed=FROZEN["seed"], noise=FROZEN["noise"],
                      branch=FROZEN["branch"], layers=FROZEN["layers"],
                      n_warmup=FROZEN["n_warmup"], n_samples=FROZEN["n_samples"],
                      n_chains=FROZEN["n_chains"], progress=False,
                      init_mode=FROZEN["init_mode"],
                      target_accept=FROZEN["target_accept"],
                      max_tree_depth=FROZEN["max_tree_depth"])
    cmp = compare(ref, cur, tol=args.tol)

    print(f"\n{'='*76}")
    print(f"{'参数':<8}{'基线中位':>14}{'重跑中位':>14}{'相对差':>12}  逐位")
    print(f"{'='*76}")
    for row in cmp["params"]:
        if row.get("status") == "missing":
            print(f"{row['param']:<8}{'—':>14}{'缺失':>14}")
            continue
        m = row["fields"]["median"]
        print(f"{row['param']:<8}{m['ref']:>14.6g}{m['cur']:>14.6g}"
              f"{m['rel']:>12.1e}  {'✓' if m['exact'] else '✗'}")
    print("-" * 76)
    for k, v in cmp["scalars"].items():
        print(f"{k:<8}{v['ref']:>14.6g}{v['cur']:>14.6g}{v['rel']:>12.1e}  "
              f"{'✓' if v['exact'] else '✗'}")
    print("=" * 76)
    label = {"exact": "逐位一致（标量路径未被改动）",
             "tolerance": f"仅在 {args.tol:.0e} 容差内一致（浮点末位差异，"
                          "不能据此断言路径改变）",
             "fail": "**超出容差——标量路径可能被改动，先查再继续**"}[cmp["status"]]
    print(f"结论：{label}   max rel diff = {cmp['max_rel_diff']:.2e}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"frozen": FROZEN, "baseline": str(args.baseline),
                        "current": {k: cur[k] for k in
                                    ("stage", "seed", "zmax", "seconds")},
                        "compare": cmp}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"写入 {args.json}")
    return 0 if cmp["status"] in ("exact", "tolerance") else 1


if __name__ == "__main__":
    sys.exit(main())
