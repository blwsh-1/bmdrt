# -*- coding: utf-8 -*-
"""P3：把 `to_inductive` 接到吸附支路做**前置预筛**的可行性（影子研究，不改门控）。

要解决的问题
------------
`bm_real.py:2585` 在 `--auto-adsorption` 下**无条件**调用
`bm_ad.inv_from_lowfreq(f, Z)` 去粗估 `(R_ad, τ_ad)` 作 NUTS 初值。
`inv_from_lowfreq` 的可靠性标记 `init_reliable` 只做三条**结构性**检查
（内部极大 / 峰在几何中心以下 / 突出度 ≥0.20），**不看低频尾是感抗还是容性**。

于是一条低频是容性尾的谱（Ws / Wo / 未闭合弧），只要在低频半区有一个
`Im` 的局部极大，就会拿到 `init_reliable=True`，NUTS 从一个物理上不属于
吸附环的位置起步。反过来，`bm_wscrit.terminal` 的 `to_inductive`
判的正是"低频端有没有真的感抗环"，两者本该是同一个问题的两面。

本脚本只量化这两者的一致性与分歧，**不改任何门控、不改先验、不动初值**。

读法
----
    · `to_inductive` 高 + `init_reliable` 高 → 预筛是冗余的（没必要加）
    · `to_inductive` 低 + `init_reliable` 高 → **误放行**，预筛有价值
    · `to_inductive` 高 + `init_reliable` 低 → 预筛会漏掉真吸附环（危险）

    $PY evidence/ws_crit/p3_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import bm_ad as AD                              # noqa: E402
import bm_wscrit as W                           # noqa: E402
import bench as B                               # noqa: E402

OUT = HERE / "p3_gate.json"


def one(f, Z):
    """一条谱的全部诊断量。"""
    f, Z = np.asarray(f, float), np.asarray(Z, complex)
    o = np.argsort(f)
    f, Z = f[o], Z[o]
    tm = W.terminal(f, Z)
    try:
        inv = AD.inv_from_lowfreq(f, Z)
    except Exception as e:                       # noqa: BLE001
        inv = dict(error=f"{type(e).__name__}: {e}")
    d = dict(term=tm["label"], im_ratio=tm.get("im_ratio"),
             se=tm.get("se"), inductive=bool(tm["label"] == "to_inductive"))
    for k in ("Rad", "tau_ad", "f_peak", "im_peak", "prominence",
              "interior", "low_half", "re_rel_span", "init_reliable",
              "n_used", "error"):
        if k in inv:
            d[k] = inv[k]
    # 初值是否落在窗口内（`f_corner = 1/(2πτ)` 应在 [f_min, f_max] 内）
    if d.get("tau_ad") and np.isfinite(d["tau_ad"]) and d["tau_ad"] > 0:
        fc = 1.0 / (2 * np.pi * d["tau_ad"])
        d["f_corner"] = float(fc)
        d["corner_in_window"] = bool(f.min() <= fc <= f.max())
        # 拐点离窗口下沿多近：<2×f_min 说明吸附环只露了不到半支，
        # `tau_ad` 是被窗口截断后的下界，不是环的真实时间常数。
        d["corner_over_fmin"] = float(fc / f.min())
        d["edge_limited"] = bool(fc < 2.0 * f.min())
    return d


def run_real(zdir=None):
    zdir = Path(zdir) if zdir else ROOT / "out/_m6_samples/z_csv"
    rows = []
    for p in sorted(Path(zdir).glob("*.csv")):
        d = np.genfromtxt(p, delimiter=",", names=True)
        nm = d.dtype.names
        f = np.asarray(d[nm[0]], float)
        Z = np.asarray(d[nm[1]], float) + 1j * np.asarray(d[nm[2]], float)
        r = one(f, Z)
        r["stem"] = p.stem
        rows.append(r)
    return rows


def run_bench(n_per=20, npts=31, fmin=0.1, fmax=1e3, noise=0.01, seed=0):
    f = np.logspace(np.log10(fmin), np.log10(fmax), npts)
    rng = np.random.default_rng(seed)
    tab = {}
    for cls in B.CLASSES:
        ind = rel = mis = miss = 0
        for _ in range(n_per):
            Z = B.make(f, cls, rng)
            sd = noise * np.abs(Z)
            Zn = Z + sd * (rng.standard_normal(f.size)
                           + 1j * rng.standard_normal(f.size))
            r = one(f, Zn)
            i, q = bool(r["inductive"]), bool(r.get("init_reliable"))
            ind += i
            rel += q
            mis += int(q and not i)          # 误放行：不是感抗尾却 claim 可靠
            miss += int(i and not q)         # 预筛会漏：是感抗尾却没拿到可靠初值
        tab[cls] = dict(n=n_per, p_inductive=ind / n_per,
                        p_reliable=rel / n_per,
                        n_false_pass=mis, n_would_miss=miss)
    return tab, dict(fmin=fmin, fmax=fmax, n_per=n_per, npts=npts)


def main():
    real = run_real()
    tab, meta = run_bench()

    print("===== A. 18 条真实 AZ31 谱 =====")
    print(f"{'stem':22s} {'term':>14s} {'imratio':>7s} {'rel':>4s} "
          f"{'tau_ad':>9s} {'f_corner':>9s} {'fc/fmin':>7s} {'edge':>5s} {'prom':>5s}")
    for r in real:
        tau = r.get("tau_ad", float("nan"))
        fc = r.get("f_corner", float("nan"))
        rr = r.get("corner_over_fmin", float("nan"))
        print(f"{r['stem']:22s} {r['term']:>14s} "
              f"{(r['im_ratio'] if r['im_ratio'] is not None else float('nan')):+7.2f} "
              f"{str(bool(r.get('init_reliable'))):>4s} {tau:9.3f} {fc:9.4f} "
              f"{rr:7.2f} {('*' if r.get('edge_limited') else ''):>5s} "
              f"{r.get('prominence', float('nan')):5.2f}")
    n_ind = sum(r["inductive"] for r in real)
    n_rel = sum(bool(r.get("init_reliable")) for r in real)
    n_both = sum(bool(r.get("init_reliable")) and r["inductive"] for r in real)
    print(f"\n  to_inductive {n_ind}/{len(real)}；init_reliable {n_rel}/{len(real)}；"
          f"两者同时成立 {n_both}/{len(real)}")
    n_edge = sum(bool(r.get("edge_limited")) for r in real)
    print(f"  拐点贴窗口下沿（f_corner < 2·f_min）{n_edge}/{len(real)} 条："
          f"这些谱的吸附环只露了不到半支，`tau_ad` 是窗口截断后的下界，"
          f"**不可当环的真实时间常数引用**")
    n_fp = sum(bool(r.get("init_reliable")) and not r["inductive"] for r in real)
    print(f"  **若加 to_inductive 预筛，会挡掉 {n_fp}/{len(real)} 条本来被判可靠的"
          f"（这些谱的低频尾是容性/其它，不是吸附环）**")

    print("\n===== B. 合成 9 类 × 20 条：两个判据的一致性 =====")
    print(f"{'class':11s} {'P(ind)':>7s} {'P(rel)':>7s} {'误放行':>7s} {'预筛会漏':>8s}")
    for cls in B.CLASSES:
        r = tab[cls]
        print(f"{cls:11s} {r['p_inductive']:7.2f} {r['p_reliable']:7.2f} "
              f"{r['n_false_pass']:7d} {r['n_would_miss']:8d}")

    tot_fp = sum(tab[c]["n_false_pass"] for c in B.CLASSES)
    tot_miss = sum(tab[c]["n_would_miss"] for c in B.CLASSES)
    tot = sum(tab[c]["n"] for c in B.CLASSES)
    print(f"\n  合计：误放行 {tot_fp}/{tot}；预筛会漏 {tot_miss}/{tot}")
    print("  说明：'误放行' = init_reliable=True 但 to_inductive=False；"
          "它只表示**播种位置**可能不对，\n"
          "        不表示后验错——初值只影响采样效率，不进先验、不进似然。")

    print(f"\n写入 {OUT}")
    OUT.write_text(json.dumps(dict(real=real, bench=tab, meta=meta,
                                   real_counts=dict(n=len(real), n_inductive=n_ind,
                                                    n_reliable=n_rel, n_both=n_both,
                                                    n_edge_limited=n_edge,
                                                    n_would_block=n_fp)),
                              ensure_ascii=False, indent=1, default=float),
                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
