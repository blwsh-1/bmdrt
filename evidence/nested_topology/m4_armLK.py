#!/usr/bin/env python
"""M4 K 臂：K=2 下还需要 L 吗；K 选择会不会被高频残差污染。

预注册见 `NESTED_TOPOLOGY_PROJECT.md` §9.4.16（v1.15）。下列常数与该节逐字对应，
**改这里等于改预注册**。

    7 条谱 × 5 seeds × K∈{1,2} × 两臂（include_L False/True），
    target_accept **0.98**（修 §9.4.15.3：0.90 会让 K=1 都发散），
    quick 预算、sigma=None + sigma_floor、zeta=1.0、阈值 4/2、单谱内逐点配对。

与 §9.4.15 **不比较绝对值**（target_accept 不同）。K 的效应只看
K=1@0.98 与 K=2@0.98 之差；L 的效应在每个 K 内部各自判。

跑法：
    python evidence/nested_topology/m4_armLK.py --json out/m4_armLK.json
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
_EV = os.path.join(_ROOT, "evidence", "nested_topology")
if _EV not in sys.path:
    sys.path.insert(0, _EV)

import bm_real as R                                  # noqa: E402
import bm_select as S                                # noqa: E402
from m4_armL_2emc import (SPECTRA, RHAT_MAX, classify,  # noqa: E402
                         paired)

KS = (1, 2)               # §9.4.16.3
SEEDS = (0, 1, 2, 3, 4)   # §9.4.16.3
TARGET_ACCEPT = 0.98      # §9.4.16.3
DELTA = S.SWITCH_DELTA
ZMIN = S.SWITCH_Z


def fit_pair(f, Z, k: int, seed: int, kw: dict) -> dict:
    """一个 (谱, seed, K) 上的两臂，返回配对结果。"""
    got = {}
    for tag, il in (("noL", False), ("L", True)):
        c = S.fit_voigt(f, Z, k, seed=seed, include_L=il, kw=dict(kw))
        got[tag] = c
    a, b = got["noL"], got["L"]
    degraded = (a["pointwise"] is None or b["pointwise"] is None
                or a["div"] > 0 or b["div"] > 0
                or (a["rhat"] or 0) > RHAT_MAX or (b["rhat"] or 0) > RHAT_MAX)
    if degraded:
        delta = dse = z = float("nan")
    else:
        delta, dse, z = paired(a["pointwise"], b["pointwise"])
    return dict(seed=seed, k=k, degraded=bool(degraded), delta=delta, dse=dse,
                z=z, rhat_noL=a["rhat"], rhat_L=b["rhat"],
                div_noL=a["div"], div_L=b["div"],
                elpd_noL=a["elpd"], elpd_L=b["elpd"])


def ratio_of(rows1, rows2):
    """Δ_K2 / Δ_K1 的中位比。分母取中位，避免某格异常主导。"""
    d1 = [r["delta"] for r in rows1 if not r["degraded"]]
    d2 = [r["delta"] for r in rows2 if not r["degraded"]]
    if not d1 or not d2:
        return float("nan")
    m1, m2 = float(np.median(d1)), float(np.median(d2))
    return m2 / m1 if m1 != 0 else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M4 K 臂：K=2 下还需要 L 吗")
    ap.add_argument("--budget", choices=sorted(S.BUDGETS), default="quick")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--json", default=None)
    ap.add_argument("--root", default="archive")
    a = ap.parse_args(argv)

    if a.budget != "quick" or a.seeds != len(SEEDS):
        print("⚠ 预算/seeds 偏离 §9.4.16.3 的冻结值：结果不能与预注册对照。",
              file=sys.stderr)

    kw = dict(S.BUDGETS[a.budget])
    kw["target_accept"] = TARGET_ACCEPT
    seeds = SEEDS[:a.seeds]
    print(f"K 臂：K={list(KS)}，target_accept={TARGET_ACCEPT}，"
          f"预算 {a.budget}，seeds={list(seeds)}，阈值 Δ≥{DELTA} 且 z≥{ZMIN}\n"
          + "=" * 92)

    spectra = []
    t_all = time.time()
    for name in SPECTRA:
        path = os.path.join(a.root, name)
        if not os.path.exists(path):
            print(f"{name:<26} 缺文件，跳过")
            continue
        f, Z = R.load_spectrum(path)
        by_k = {k: [] for k in KS}
        for sd in seeds:
            for k in KS:
                t0 = time.time()
                r = fit_pair(f, Z, k, sd, kw)
                by_k[k].append(r)
                if r["degraded"]:
                    print(f"  {name:<24} seed={sd} K={k}  降级"
                          f"（R̂={r['rhat_noL']:.3f}/{r['rhat_L']:.3f} "
                          f"div={r['div_noL']}/{r['div_L']}）"
                          f"  {time.time() - t0:.0f}s")
                else:
                    print(f"  {name:<24} seed={sd} K={k}  Δ={r['delta']:+8.2f} "
                          f"dse={r['dse']:6.2f} z={r['z']:+6.2f}  "
                          f"R̂={r['rhat_noL']:.3f}/{r['rhat_L']:.3f}  "
                          f"div={r['div_noL']}/{r['div_L']}  "
                          f"{time.time() - t0:.0f}s")

        rec = dict(name=name, rows=by_k)
        for k in KS:
            v, u, fl, hm = classify(by_k[k])
            rec[f"verdict_k{k}"] = v
            rec[f"useful_k{k}"] = u
            rec[f"flat_k{k}"] = fl
            rec[f"harm_k{k}"] = hm
            ok = [r for r in by_k[k] if not r["degraded"]]
            rec[f"median_k{k}"] = (float(np.median([r["delta"] for r in ok]))
                                   if ok else float("nan"))
            rec[f"n_ok_k{k}"] = len(ok)
            rec[f"n_drop_k{k}"] = len(by_k[k]) - len(ok)
        rec["ratio"] = ratio_of(by_k[1], by_k[2])
        spectra.append(rec)
        print(f"  → K=1: {rec['verdict_k1']}（中位 Δ={rec['median_k1']:+.1f}）；"
              f"K=2: {rec['verdict_k2']}（中位 Δ={rec['median_k2']:+.1f}）；"
              f"比值={rec['ratio']:.2f}\n")

    print("=" * 92)
    print(f"{'谱':<26}{'K=1 判定':<11}{'Δ₁':>9}{'K=2 判定':<11}{'Δ₂':>9}{'Δ₂/Δ₁':>8}"
          f"{'降级 1/2':>9}")
    for r in spectra:
        print(f"{r['name']:<26}{r['verdict_k1']:<11}{r['median_k1']:>9.1f}"
              f"{r['verdict_k2']:<11}{r['median_k2']:>9.1f}{r['ratio']:>8.2f}"
              f"{r['n_drop_k1']:>5}/{r['n_drop_k2']:<3}")

    # ---- 预注册核对 ----
    print("\n预注册核对（§9.4.16.5）：")
    u2 = sum(1 for r in spectra if r["verdict_k2"] == "L 有用")
    print(f"  P1 K=2 下 ≥5/7 判「L 有用」→ {'通过' if u2 >= 5 else '未通过'}（{u2}/{len(spectra)}）")

    rats = [r["ratio"] for r in spectra
            if np.isfinite(r["ratio"])]
    med_ratio = float(np.median(rats)) if rats else float("nan")
    p2 = bool(rats) and 0.7 <= med_ratio <= 1.3
    print(f"  P2 比值中位落在 [0.7, 1.3] → {'通过' if p2 else '未通过'}（{med_ratio:.2f}）")

    d1 = sum(r["n_drop_k1"] for r in spectra)
    d2 = sum(r["n_drop_k2"] for r in spectra)
    print(f"  P3 K=2 降级格 ≥ K=1 → {'通过' if d2 >= d1 else '未通过'}（{d1} vs {d2}）")

    dropped = [r for r in spectra if r["name"] == "2EMC1-1_0005mA.csv"]
    if dropped:
        nd = dropped[0]["n_drop_k1"]
        print(f"  P4 K=1@0.98 纳入 2EMC1-1_0005mA → "
              f"{'通过' if nd == 0 else '未通过'}（降级 {nd} 格）")

    weakest = [r for r in spectra if r["name"] == "2EP-2_exp2_0500mA.csv"]
    if weakest and rats:
        rmin = min((r for r in spectra if np.isfinite(r["ratio"])),
                   key=lambda r: r["ratio"])
        ok = rmin["name"] == "2EP-2_exp2_0500mA.csv"
        print(f"  P5 2EP-2 比值最小 → {'通过' if ok else '未通过'}"
              f"（最小是 {rmin['name']}，比值 {rmin['ratio']:.2f}）")

    # ---- 逐条报「掉档」的谱（K2 否决条件）----
    worse = [r for r in spectra
             if r["verdict_k1"] == "L 有用" and r["verdict_k2"] != "L 有用"]
    if worse:
        print("\n  逐条报：K=1 有用但 K=2 掉档的谱")
        for r in worse:
            print(f"    {r['name']:<26} K=2 → {r['verdict_k2']}，"
                  f"中位 Δ={r['median_k2']:+.1f}，比值={r['ratio']:.2f}")

    print("\n边界（§9.4.16.6 K4/K5）：Δ 显著 ≠ 检出电感；L 在 voigt 里是最外层串联，"
          "代表引线与仪器。本臂只比 K=1/K=2 两端，不推断最优 K。")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fp:
            json.dump(dict(ks=list(KS), seeds=list(seeds), budget=a.budget,
                           target_accept=TARGET_ACCEPT,
                           delta_thr=DELTA, z_thr=ZMIN, rhat_max=RHAT_MAX,
                           spectra=spectra, median_ratio=med_ratio,
                           elapsed_s=round(time.time() - t_all, 1)),
                      fp, ensure_ascii=False, indent=1)
        print(f"\n结果已写入 {a.json}")
    print(f"总耗时 {time.time() - t_all:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
