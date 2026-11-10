#!/usr/bin/env python
"""M4 L 臂：2EMC 高频段加串联 L 值不值。

预注册见 `NESTED_TOPOLOGY_PROJECT.md` §9.4.14（v1.13）。本脚本是它的**唯一**执行器，
下列常数与该节逐字对应，**改这里等于改预注册**，必须同时改文档并说明前因。

    7 条独立谱 × 5 seeds × 2 臂（include_L False/True），voigt K=1，
    quick 预算（800/400/2），sigma=None + sigma_floor，zeta=1.0，
    SWITCH_DELTA=4、SWITCH_Z=2，单谱内**逐点**配对 LOO。

跑法：
    python evidence/nested_topology/m4_armL_2emc.py --json out/m4_armL_2emc.json

判读四档（§9.4.14.4）：L 有用 / 无变化 / L 有害 / 不可判。
禁止挑 seed、禁止把 7 条谱的 Δ 平均成一个数、禁止把 ±dse 内的差称为升或降。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import bm_real as R                      # noqa: E402
import bm_select as S                    # noqa: E402

#: §9.4.14.2 的 7 条。`2EMC1-3_exp1_0050mA_1.csv` 与 #4 的 sha256 相同，已删。
SPECTRA = (
    "2EMC1-1_0005mA.csv",
    "2EMC1-1_0050mA.csv",
    "2EMC1-3_exp1_0005mA.csv",
    "2EMC1-3_exp1_0050mA.csv",
    "2EMC1-3_exp1_0100mA.csv",
    "2EMC2-1_0200mA.csv",
    "2EP-2_exp2_0500mA.csv",
)

K = 1                     # §9.4.14.3：该批谱的现行口径
SEEDS = (0, 1, 2, 3, 4)   # §9.4.14.3
DELTA = S.SWITCH_DELTA    # 4
ZMIN = S.SWITCH_Z         # 2
RHAT_MAX = 1.05           # §9.4.14.7 K3


def paired(pt_noL, pt_L):
    """单谱内逐点配对：Δ、dse、z。与 `bm_select.paired_diff` 同口径。"""
    d = np.asarray(pt_L, float) - np.asarray(pt_noL, float)
    n = d.size
    delta = float(d.sum())
    dse = float(np.sqrt(n * np.var(d, ddof=1))) if n > 1 else float("nan")
    z = delta / dse if np.isfinite(dse) and dse > 0 else float("nan")
    return delta, dse, z


def classify(rows):
    """按 §9.4.14.4 的四档判读。`rows` 是同一谱的若干 seed（已剔除降级格）。

    阈值按**比例**折算而不是写死 4 和 3：§9.4.14.4 写的是「≥4/5」「≥3/5」，
    n=5 时折算出 4 和 3，与原文逐字相同；`--seeds` 取别的值时才需要按比例缩放
    （写死 4 会让 n=1 的冒烟永远落「不可判」，那是统计口径 bug，不是结果）。
    """
    ok = [r for r in rows if not r["degraded"]]
    n = len(ok)
    if n == 0:
        return "不可判（全部降级）", 0, 0, 0

    need_useful = max(1, int(np.ceil(0.8 * n)))     # n=5 → 4
    need_flat = max(1, int(np.ceil(0.6 * n)))       # n=5 → 3

    useful = sum(1 for r in ok if r["delta"] >= DELTA and r["z"] >= ZMIN)
    flat = sum(1 for r in ok if abs(r["delta"]) < r["dse"])
    harm = sum(1 for r in ok if r["delta"] <= -DELTA and r["z"] <= -ZMIN)

    if useful >= need_useful:
        return "L 有用", useful, flat, harm
    if harm >= need_useful:
        return "L 有害", useful, flat, harm
    if flat >= need_flat:
        return "无变化", useful, flat, harm
    return "不可判", useful, flat, harm


def run_one(f, Z, seed: int, kw: dict) -> dict:
    out = {}
    for tag, il in (("noL", False), ("L", True)):
        c = S.fit_voigt(f, Z, K, seed=seed, include_L=il, kw=dict(kw))
        rhat = c.get("rhat")
        out[tag] = dict(elpd=c["elpd"], se=c["se"], rhat=rhat,
                        div=c["div"], ok=c["ok"],
                        pointwise=None if c["pointwise"] is None
                        else np.asarray(c["pointwise"], float).tolist())
    degraded = (out["noL"]["pointwise"] is None or out["L"]["pointwise"] is None
                or out["noL"]["div"] > 0 or out["L"]["div"] > 0
                or (out["noL"]["rhat"] or 0) > RHAT_MAX
                or (out["L"]["rhat"] or 0) > RHAT_MAX)
    if not degraded:
        delta, dse, z = paired(out["noL"]["pointwise"], out["L"]["pointwise"])
    else:
        delta = dse = z = float("nan")
    return dict(seed=seed, degraded=bool(degraded), delta=delta, dse=dse, z=z,
                rhat_noL=out["noL"]["rhat"], rhat_L=out["L"]["rhat"],
                div_noL=out["noL"]["div"], div_L=out["L"]["div"],
                elpd_noL=out["noL"]["elpd"], elpd_L=out["L"]["elpd"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M4 L 臂：2EMC 加串联 L 值不值")
    ap.add_argument("--budget", choices=sorted(S.BUDGETS), default="quick",
                    help="§9.4.14.3 冻结为 quick；改它等于改预注册")
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="用前 n 个 seed（§9.4.14.3 冻结为 5）")
    ap.add_argument("--json", default=None)
    ap.add_argument("--root", default="archive", help="谱文件所在目录")
    a = ap.parse_args(argv)

    if a.budget != "quick" or a.seeds != len(SEEDS):
        print("⚠ 预算/seeds 与 §9.4.14.3 的冻结值不一致：结果不能与预注册对照，"
              "只能当探索性诊断。", file=sys.stderr)

    kw = dict(S.BUDGETS[a.budget])
    seeds = SEEDS[:a.seeds]
    print(f"L 臂：voigt K={K}，预算 {a.budget}，seeds={list(seeds)}，"
          f"阈值 Δ≥{DELTA} 且 z≥{ZMIN}\n" + "=" * 86)

    results = []
    t_all = time.time()
    for name in SPECTRA:
        path = os.path.join(a.root, name)
        if not os.path.exists(path):
            print(f"{name:<26} 缺文件，跳过")
            continue
        f, Z = R.load_spectrum(path)
        rows = []
        for sd in seeds:
            t0 = time.time()
            r = run_one(f, Z, sd, kw)
            rows.append(r)
            if r["degraded"]:
                print(f"  {name:<24} seed={sd}  降级（R̂={r['rhat_noL']:.3f}/"
                      f"{r['rhat_L']:.3f} div={r['div_noL']}/{r['div_L']}）"
                      f"  {time.time() - t0:.0f}s")
            else:
                print(f"  {name:<24} seed={sd}  elpd {r['elpd_noL']:8.2f} → "
                      f"{r['elpd_L']:8.2f}  Δ={r['delta']:+7.2f} "
                      f"dse={r['dse']:5.2f} z={r['z']:+6.2f}  "
                      f"R̂={r['rhat_noL']:.3f}/{r['rhat_L']:.3f}  "
                      f"{time.time() - t0:.0f}s")

        verdict, useful, flat, harm = classify(rows)
        ok = [r for r in rows if not r["degraded"]]
        signs = {np.sign(r["delta"]) for r in ok if r["delta"] != 0}
        med = float(np.median([r["delta"] for r in ok])) if ok else float("nan")
        same_sign = len(signs) <= 1
        results.append(dict(name=name, n=len(rows), verdict=verdict,
                            useful=useful, flat=flat, harm=harm,
                            median_delta=med, same_sign=bool(same_sign),
                            rows=rows))
        print(f"  → {verdict}（有用 {useful}/5、无变化 {flat}/5、有害 {harm}/5；"
              f"中位 Δ={med:+.2f}；同号={'是' if same_sign else '否'}）\n")

    print("=" * 86)
    print(f"{'谱':<26}{'判定':<10}{'中位 Δ':>9}{'同号':>5}  逐 seed Δ")
    for r in results:
        ds = " ".join(f"{x['delta']:+.1f}" if not x["degraded"] else "—"
                      for x in r["rows"])
        print(f"{r['name']:<26}{r['verdict']:<10}{r['median_delta']:>9.2f}"
              f"{'是' if r['same_sign'] else '否':>5}  {ds}")

    tally = {}
    for r in results:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    print("\n汇总：" + "；".join(f"{k} {v}/{len(results)}" for k, v in tally.items()))
    print("预测核对（§9.4.14.5）：")
    P1 = sum(1 for r in results if r["verdict"] == "L 有用")
    P2 = all(r["same_sign"] for r in results)
    print(f"  P1 至少 5/7 判「L 有用」→ {'通过' if P1 >= 5 else '未通过'}（{P1}/{len(results)}）")
    print(f"  P2 全部 5/5 seed 同号   → {'通过' if P2 else '未通过'}")
    rmax = max((max(x["rhat_noL"] or 0, x["rhat_L"] or 0)
                for r in results for x in r["rows"]), default=0)
    # P4 只对**未降级**的格清点（降级格已按 K3 剔除，不该再拿来判 P4）；
    # 但降级本身要单独报，否则等于静默丢数据。
    ndiv = sum(x["div_noL"] + x["div_L"]
               for r in results for x in r["rows"] if not x["degraded"])
    n_drop = sum(1 for r in results for x in r["rows"] if x["degraded"])
    dropped = [f"{r['name']}#{x['seed']}" for r in results for x in r["rows"]
               if x["degraded"]]
    print(f"  P4 R̂ 最大 {rmax:.3f}（<1.05 要求）、未降级格的 div 合计 {ndiv}（=0）"
          f" → {'通过' if rmax < RHAT_MAX and ndiv == 0 else '未通过'}")
    if n_drop:
        print(f"     按 K3 剔除 {n_drop} 格：{dropped}")
    print("\n边界（§9.4.14.7 K4）：Δ 显著 ≠ 样品有电感。这些点在 8e3–1e5 Hz 是"
          "仪器/引线伪影区；voigt 的 include_L 是最外层串联，代表仪器不是界面过程。"
          "本臂也不构成对 nested 的支持。")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fp:
            json.dump(dict(k=K, budget=a.budget, seeds=list(seeds),
                           delta_thr=DELTA, z_thr=ZMIN, rhat_max=RHAT_MAX,
                           spectra=results,
                           elapsed_s=round(time.time() - t_all, 1)),
                      fp, ensure_ascii=False, indent=1)
        print(f"\n结果已写入 {a.json}")
    print(f"总耗时 {time.time() - t_all:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
