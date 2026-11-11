# -*- coding: utf-8 -*-
"""Lin-KK 的阶数曲线：8.49% 到底是「地板」还是「某一阶的拟合水平」。

**起因**：`AZ31-4Li_100_RESID.md` 原文写「Lin-KK 8.49% ⇒ 这份数据违反 KK 8.5%，
任何因果 LTI 模型都不可能更好 ⇒ acf≤0.50 结构性不可达」。外部审查指出这步越界，
理由是**有限阶 Lin-KK 的残差不是所有因果 LTI 模型的数学下界**——加基函数，
残差可能继续降。这个质疑在代码层面有直接依据：

    `bm_kk.lin_kk()` **不是取最小残差的阶**，而是按肘部准则
    `pick = argmin r_all·(1 + 0.02·M)` 选阶（见 `bm_kk.py:189`）。
    所以 8.49% 对应的是「肘部选中的 M=12」，不是「扫到的最好值」。

本脚本把整条阶数曲线打出来，用**实测**代替原来的逻辑断言：

· 若 r_all 在某个 M 之后就平了（再加密基函数也降不下去）→「参考拟合水平」这个
  说法站得住，但**仍然不是数学下界**，措辞必须以此为准；
· 若 r_all 随 M 持续下降 → 原来的「地板」结论作废，`acf≤0.50 不可达` 也要撤回。

不碰生产代码、不改判据、不重跑 MCMC。产物 `out/linkk_order_probe.{json,md}`。

跑法（**必须在仓库根**，数据与产物都是 cwd 相对路径）：

    python docs/round_2026-11-11/linkk_order_probe.py
    python docs/round_2026-11-11/linkk_order_probe.py --specs AZ31-4Li_100 AZ31-8Li_50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import bm_experiments as E             # noqa: E402
import bm_kk as KK                     # noqa: E402
import bm_real as R                    # noqa: E402

SPEC_DIR = os.path.join("out", "_m6_samples", "z_csv")
OUT_JSON = os.path.join("out", "linkk_order_probe.json")
OUT_MD = os.path.join("out", "linkk_order_probe.md")
BANDS = ((0.01, 1.0, "low"), (1.0, 100.0, "mid"), (100.0, 1e5, "high"))
ELBOW = 0.02


def _fit_M(f, Z, M, add_cap=False):
    """直接调 impedance 的两个纯 numpy 函数，**绕过 `lin_kk` 的选阶与停滞护栏**，
    这样才看得到整条曲线（`lin_kk` 在 M≥8 时连续 5 阶改善 <2% 就 break）。

    阻抗重建复用 `bm_kk._eval_linkk`（**单一来源**）：自己再写一份，`L`/`add_cap`
    的位次一错就会静默算错，而这里要比的正是小数位。
    """
    from impedance.validation import fit_linKK, get_tc_distribution

    ts = get_tc_distribution(f, M)
    elements, mu = fit_linKK(f, ts, M, Z, "complex", add_cap)
    z = KK._eval_linkk(np.asarray(elements, float), ts, f)
    return z, float(mu), float(elements[-1])        # elements[-1] 恒为串联 L


def _metrics(f, Z, Zm):
    r = Zm - Z
    q = np.abs(r) / np.abs(Z)
    out = dict(
        rwp=float(np.sqrt(np.mean(q ** 2)) * 100),
        acf=float(max(E.acf_max(r.real), E.acf_max(r.imag))),
        bands={},
    )
    for a, b, tag in BANDS:
        m = (f >= a) & (f < b)
        out["bands"][tag] = (float(np.sqrt(np.mean(q[m] ** 2)) * 100)
                             if m.any() else None)
    return out


def scan(f, Z, *, max_M=50, add_cap=False, allow_overfit=False):
    n = f.size
    guard = max(4, (n - 2) // 2)          # 参数个数 M+2 不超过残差条数的一半
    m_hi = int(max_M if allow_overfit else min(max_M, guard))
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for M in range(1, m_hi + 1):
            try:
                Zm, mu, L = _fit_M(f, Z, M, add_cap=add_cap)
            except Exception as exc:               # 单阶病态不影响其它阶
                rows.append(dict(M=M, ok=False, err=f"{type(exc).__name__}: {exc}"))
                continue
            met = _metrics(f, Z, Zm)
            rows.append(dict(M=M, ok=True, mu=mu if np.isfinite(mu) else None,
                             L=L, beyond_guard=M > guard,
                             mu_overfit=bool(np.isfinite(mu) and mu < 0.1), **met))
    ok = [r for r in rows if r["ok"]]
    best = min(ok, key=lambda r: r["rwp"]) if ok else None
    elbow = min(ok, key=lambda r: r["rwp"] * (1.0 + ELBOW * r["M"])) if ok else None
    # 平了没有：最后 5 阶的相对下降
    tail = [r for r in ok if r["M"] > (ok[-1]["M"] - 5)] if len(ok) >= 5 else []
    pd = ((tail[0]["rwp"] - tail[-1]["rwp"]) / tail[0]["rwp"]) if len(tail) >= 2 else None
    return dict(n=n, m_hi=m_hi, guard=guard, add_cap=add_cap,
                allow_overfit=allow_overfit, rows=rows,
                best_M=best["M"] if best else None,
                best_rwp=best["rwp"] if best else None,
                best_acf=best["acf"] if best else None,
                elbow_M=elbow["M"] if elbow else None,
                elbow_rwp=elbow["rwp"] if elbow else None,
                min_acf=min(r["acf"] for r in ok) if ok else None,
                min_acf_M=min(ok, key=lambda r: r["acf"])["M"] if ok else None,
                tail_rel_drop=pd)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lin-KK 阶数曲线（回答「地板」质疑）")
    ap.add_argument("--specs", nargs="+", default=["AZ31-4Li_100"])
    ap.add_argument("--max-M", type=int, default=50)
    ap.add_argument("--allow-overfit", action="store_true",
                    help="越过「参数不超过残差条数一半」的护栏继续加阶"
                         "（高阶层会进 μ→0 的负元件区，只为看曲线走向）")
    ap.add_argument("--json", default=OUT_JSON)
    ap.add_argument("--md", default=OUT_MD)
    a = ap.parse_args(argv)

    out = {}
    for spec in a.specs:
        f, Z = R.load_spectrum(os.path.join(SPEC_DIR, spec + ".csv"))
        rec = scan(f, Z, max_M=a.max_M, allow_overfit=a.allow_overfit)
        out[spec] = rec
        print(f"\n=== {spec}  n={f.size}  M_hi={rec['m_hi']}"
              f"（护栏 {rec['guard']}，allow_overfit={rec['allow_overfit']}）===")
        print(f"{'M':>3} {'rwp%':>8} {'acf':>6} {'mu':>8} {'low':>7} {'mid':>6} {'high':>6}")
        for r in rec["rows"]:
            if not r["ok"]:
                print(f"{r['M']:>3}  —— {r['err']}")
                continue
            mu = "   n/a  " if r["mu"] is None else f"{r['mu']:8.3f}"
            b = r["bands"]
            print(f"{r['M']:>3} {r['rwp']:8.3f} {r['acf']:6.3f} {mu} "
                  f"{b['low']:7.2f} {b['mid']:6.2f} {b['high']:6.2f}"
                  + ("   ← 最小 rwp" if r["M"] == rec["best_M"] else "")
                  + ("   ← 肘部选中" if r["M"] == rec["elbow_M"] else "")
                  + ("   ← 越过护栏" if r.get("beyond_guard") else "")
                  + ("   （μ<0.1 负元件区）" if r.get("mu_overfit") else ""))
        print(f"最小 rwp = {rec['best_rwp']:.3f}%（M={rec['best_M']}, acf={rec['best_acf']:.3f}）"
              f"；肘部 = {rec['elbow_rwp']:.3f}%（M={rec['elbow_M']}）"
              f"；acf 最小 = {rec['min_acf']:.3f}（M={rec['min_acf_M']}）"
              f"；末 5 阶相对下降 = "
              f"{rec['tail_rel_drop'] if rec['tail_rel_drop'] is None else round(rec['tail_rel_drop'] * 100, 2)}%")

    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, allow_nan=False, indent=1)

    L = ["# Lin-KK 阶数曲线（实测，非下界证明）", "",
         "> 目的：把「8.49% 是地板」这个**逻辑断言**换成一串**实测残差**。",
         "> 有限阶 Lin-KK 的残差**不是**所有因果 LTI 模型的数学下界；",
         "> 本表只说明「在这一族基函数、这一组阶数下能到多少」。", ""]
    for spec, rec in out.items():
        L += [f"## {spec}（n={rec['n']}，M≤{rec['m_hi']}，护栏 {rec['guard']}，"
              f"越过护栏={rec['allow_overfit']}，add_cap={rec['add_cap']}）", "",
              "| M | Rwp% | acf | μ | low% | mid% | high% | 备注 |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rec["rows"]:
            if not r["ok"]:
                L.append(f"| {r['M']} | — | — | — | — | — | — | {r['err']} |")
                continue
            tag = []
            if r["M"] == rec["best_M"]:
                tag.append("最小 rwp")
            if r["M"] == rec["elbow_M"]:
                tag.append("肘部选中")
            if r.get("beyond_guard"):
                tag.append("越过护栏")
            if r.get("mu_overfit"):
                tag.append("μ<0.1 负元件区")
            b = r["bands"]
            mu = "n/a" if r["mu"] is None else f"{r['mu']:.3f}"
            L.append(f"| {r['M']} | {r['rwp']:.3f} | {r['acf']:.3f} | {mu} "
                     f"| {b['low']:.2f} | {b['mid']:.2f} | {b['high']:.2f} "
                     f"| {'、'.join(tag)} |")
        L += ["",
              f"最小 Rwp **{rec['best_rwp']:.3f}%**（M={rec['best_M']}，acf {rec['best_acf']:.3f}）；"
              f"肘部选中 **{rec['elbow_rwp']:.3f}%**（M={rec['elbow_M']}）；"
              f"acf 最小 **{rec['min_acf']:.3f}**（M={rec['min_acf_M']}）。", ""]
    with open(a.md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"\n[done] {a.json}\n[done] {a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
