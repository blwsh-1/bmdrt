"""E1 判据的**判别力基线**：不同状态之间，残差形状本来就相关到什么程度？

**只读，不采样、不拟合、不改判定。** 输入 `out/m4_holdout2_ess.json` 的后验中位参数，
正演得残差，再算 §14 这 14 条谱**两两之间**的残差形状相关（91 对）。

为什么必须跑：E1 的 J1 要求「同状态 3 次重复的两两 |r| ≥0.70」。若**不同状态**之间
的残差形状就已经普遍 0.7–0.9，那 0.70 这条线分不开「同状态可再现」与「模型固有
残差形状」——`J1` 会被高估。本脚本只量化这个基线，**不改 E1 的任何阈值**。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/e1_baseline.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evidence", "nested_topology"))

import bm_real as R          # noqa: E402
import bm_top as T           # noqa: E402

SPEC_DIR = os.path.join("out", "_m6_samples", "z_csv")


def residuals(src: dict) -> dict:
    """按 M-b 口径重建每条谱的残差（中位参数正演）。"""
    out = {}
    for row in src["rows"]:
        f, Z = R.load_spectrum(os.path.join(SPEC_DIR, row["spec"] + ".csv"))
        keep = np.ones(f.size, bool)
        for hz in row.get("dropped_hz") or []:
            keep &= ~np.isclose(f, float(hz), rtol=1e-6)
        f, Z = f[keep], Z[keep]
        r = row["rows"][0]
        Zm = T.z_nested(f, r["params"], branch=row["branch"], layers=row["layers"])
        out[row["spec"]] = dict(f=f, Z=Z, res=Zm - Z, state=row["state"])
    return out


def _r(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.std() <= 0 or b.std() <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def stat(pairs: list[dict], key: str, sub=None) -> dict:
    v = np.array([abs(p[key]) for p in pairs
                  if p.get(key) is not None and (sub is None or sub(p))], float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), median=float(np.median(v)),
                p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                min=float(v.min()), max=float(v.max()),
                frac_ge_070=float(np.mean(v >= 0.70)),
                frac_le_030=float(np.mean(v <= 0.30)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join("out", "m4_holdout2_ess.json"))
    ap.add_argument("--out", default=os.path.join("out", "e1_baseline.json"))
    a = ap.parse_args(argv)

    src = json.load(open(a.json, encoding="utf-8"))
    D = residuals(src)

    # 只保留栅格完全一致的谱（12Li_150 因 M1 剔了 1 点 → n=49，排除在矩阵外）
    ref, same, odd = None, [], []
    for k, v in D.items():
        if ref is None:
            ref = v["f"]
        if v["f"].size == ref.size and np.allclose(v["f"], ref, rtol=1e-9):
            same.append(k)
        else:
            odd.append(k)
    if odd:
        print(f"栅格不一致、排除在矩阵外：{odd}")

    pairs = []
    for i, j in itertools.combinations(same, 2):
        A, B = D[i], D[j]
        ra, rb = A["res"], B["res"]
        za, zb = np.abs(A["Z"]), np.abs(B["Z"])
        fam = lambda x: x.split("-")[1].split("_")[0]   # AZ31-8Li_50 -> 8Li
        pairs.append(dict(a=i, b=j, same_series=fam(i) == fam(j),
                          r_re=_r(ra.real, rb.real), r_im=_r(ra.imag, rb.imag),
                          r_re_norm=_r(ra.real / za, rb.real / zb),
                          r_im_norm=_r(ra.imag / za, rb.imag / zb)))

    summ = {k: stat(pairs, k) for k in
            ("r_re", "r_im", "r_re_norm", "r_im_norm")}
    same_series = {k: stat(pairs, k, lambda p: p["same_series"])
                   for k in ("r_re", "r_im")}

    print(f"参与两两比较：{len(same)} 条谱，{len(pairs)} 对"
          f"（同系列 {sum(p['same_series'] for p in pairs)} 对）\n")
    print(f"{'量':<12}{'n':>4}{'中位':>8}{'p10':>8}{'p90':>8}{'|r|>=0.7':>10}{'|r|<=0.3':>10}")
    for k, v in summ.items():
        if v.get("n"):
            print(f"{k:<12}{v['n']:>4}{v['median']:>8.3f}{v['p10']:>8.3f}"
                  f"{v['p90']:>8.3f}{v['frac_ge_070']:>10.0%}{v['frac_le_030']:>10.0%}")
    print("\n只取同系列（同 Li 含量、不同状态；同状态重复的最接近代理）：")
    for k, v in same_series.items():
        if v.get("n"):
            print(f"  {k:<8} n={v['n']:<3} 中位 {v['median']:.3f}  "
                  f"范围 {v['min']:.3f}–{v['max']:.3f}  ≥0.70 占 {v['frac_ge_070']:.0%}")

    # ---- 判据统计量的零分布：从「不同状态」的 78 对里随机抽 3 对取最小 |r| ----
    # 这是 E1 真实使用的统计量（3 条重复 -> 3 对，取最小 |r|）在「无可重复性」下的分布。
    rng = np.random.default_rng(0)
    null = {}
    for k in ("r_re", "r_im", "r_re_norm", "r_im_norm"):
        v = np.array([abs(p_[k]) for p_ in pairs], float)
        v = v[np.isfinite(v)]
        draw = rng.choice(v, size=(5000, 3), replace=True).min(axis=1)
        null[k] = dict(median=float(np.median(draw)),
                       p90=float(np.percentile(draw, 90)),
                       p95=float(np.percentile(draw, 95)),
                       p99=float(np.percentile(draw, 99)),
                       prob_ge_070=float(np.mean(draw >= 0.70)),
                       prob_le_030=float(np.mean(draw <= 0.30)))
    print("\n「无可重复性」零假设下 min|r|（3 对取最小）的分布：")
    print(f"{'量':<12}{'中位':>8}{'p90':>8}{'p95':>8}{'p99':>8}"
          f"{'P(>=0.70)':>11}{'P(<=0.30)':>11}")
    for k, v in null.items():
        print(f"{k:<12}{v['median']:>8.3f}{v['p90']:>8.3f}{v['p95']:>8.3f}"
              f"{v['p99']:>8.3f}{v['prob_ge_070']:>11.4f}{v['prob_le_030']:>11.3f}")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(note="跨状态残差形状相关 = E1 判据 r>=0.70 的判别力基线；只读，不改阈值",
                       spectra=same, excluded=odd, summary=summ,
                       same_series=same_series, null_min_of_3=null, pairs=pairs),
                  fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
