# -*- coding: utf-8 -*-
"""修复前后 B 类 guard 状态翻转表（`param_degenerate` / `param_at_bound`）。

2026-11-11 修的两处「族适配」缺口：
  · `bm_select._corr_pairs()` 只吃**标量列**（`a.size == n0`），voigt 的
    `tau/phi/w` 是 `(n_draw, K)` → 整列被丢，只剩 `(R_inf, W)`：
    K=1 恒 `triggered`（`W≡w₁`，与 `R_inf` 分摊同一份直流电阻）、K≥2 恒 `ok`。
  · `bm_select._uniform_bound_frac()` 只认 `n*` 键名，voigt 的指数叫 `phi`
    → 恒 `not_evaluable`，φ 贴 0.5（CPE 退化成 Warburg）与 ζ 贴边从不报。

本脚本**不重跑 MCMC**：拿 `out/**/real_*.json` 旁边已落盘的
`*_samples.npz` 后验，对**同一份样本**分别跑修复前（冻结实现）与修复后
（生产代码）两套判据，逐条比状态。

边界（写进 `docs/round_2026-11-11/guard_semantics.md`）：
  · 只动记账层。A 类门禁、拟合、选 K、M4 结论都不在这里；
  · 翻转 = "此前没适配上该模型族的 guard 被修好了"，**不是**模型变好、
    也不是参数可辨识性改善——同一份后验，只是判据换了。

用法（仓库根）：
    python docs/round_2026-11-11/guard_fix_flip.py            # 全量
    python docs/round_2026-11-11/guard_fix_flip.py --limit 20 # 冒烟
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import bm_core as B          # noqa: E402
import bm_corr as CORR       # noqa: E402
import bm_select as S        # noqa: E402

OUT_JSON = os.path.join("out", "guard_fix_flip.json")
#: `param_degenerate` 的阈值：修复前（@6ee7122）是硬编码 0.95，修复后读
#: `param_corr.strong_tol`（`bm_corr.STRONG`，默认 0.90）。0.95 在此**硬编码**
#: 冻结，不再从生产代码取——2026-11-12 起 `S.CORR_STRONG` 已统一到 0.90，
#: 直接引用会污染「修复前」的复现。
OLD_STRONG = 0.95


# --------------------------------------------------------------------------
# 修复前的实现：**原样冻结**（只用于对照，不许被"顺手修好"）
# --------------------------------------------------------------------------

def old_corr_pairs(s: dict) -> list[tuple]:
    """`bm_select._corr_pairs` @ 6ee7122 的逐行副本。"""
    arr, n0 = {}, None
    for k, v in (s or {}).items():
        if k.startswith("s_") or k == "L_ad":
            continue
        a = np.asarray(v, float).ravel()
        if a.size < 2:
            continue
        if n0 is None:
            n0 = a.size
        if a.size == n0 and np.isfinite(a).all() and np.std(a) > 0:
            arr[k] = a
    if len(arr) < 2:
        return []
    keys = list(arr)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(np.stack([arr[k] for k in keys]))
    return [(keys[i], keys[j], float(C[i, j]))
            for i in range(len(keys)) for j in range(i + 1, len(keys))
            if np.isfinite(C[i, j]) and abs(C[i, j]) > OLD_STRONG]


def old_bound_frac(s: dict) -> float | None:
    """`bm_select._uniform_bound_frac` @ 6ee7122 的逐行副本（只认 `n*`）。"""
    near = total = 0
    for k, v in (s or {}).items():
        if not k.startswith("n"):
            continue
        a = np.asarray(v, float).ravel()
        if a.size < 2 or a.min() < 0.5 - 1e-6 or a.max() > 1.0 + 1e-6:
            continue
        total += a.size
        near += int(np.sum((a < 0.5 + S.BOUND_EPS * 0.5) |
                           (a > 1.0 - S.BOUND_EPS * 0.5)))
    return near / total if total else None


# --------------------------------------------------------------------------
# 修复后的判据：直接调生产代码
# --------------------------------------------------------------------------

def _states(s: dict, family: str, k: int | None, pc: dict | None) -> dict:
    ob = old_bound_frac(s) if family == "nested" else old_bound_frac(s)
    old_b = ("not_evaluable" if ob is None else
             "triggered" if ob > 0.05 else "ok")
    nb, ndet = S.at_bound_frac(s, family)
    new_b = ("not_evaluable" if nb is None else
             "triggered" if nb > S.AT_BOUND_FRAC else "ok")
    op = old_corr_pairs(s)
    old_d = "triggered" if op else "ok"
    dp95: list[dict] = []
    if family == "voigt":
        dp, dw = S._guard_deg_pairs(pc, k)
        new_d = "not_evaluable" if dw else ("triggered" if dp else "ok")
        # 隔离「阈值效应」：`param_corr.strong` 用 `bm_corr.STRONG=0.90`，
        # 而 nested 的标量路径仍是 0.95。按 0.95 再滤一遍，才能回答
        # 「这条翻转是修好了族适配，还是只是阈值放宽了」。
        dp95 = [p for p in dp if abs(p["rho"]) >= OLD_STRONG]
        new_d95 = "not_evaluable" if dw else ("triggered" if dp95 else "ok")
        why = (f"{len(op)} 对|ρ|>{OLD_STRONG}（标量路径）" if op else "") + \
              (f" | param_corr: {dw}" if dw else
               f" | 分量对 {len(dp)}（≥{OLD_STRONG}: {len(dp95)}）")
    else:
        new_d = old_d                               # nested 未改判据
        dp = [dict(a=a, b=b, rho=r) for a, b, r in op]
        dp95 = dp
        new_d95 = new_d
        why = "nested 沿用标量路径（本轮不改）"
    return dict(old_deg=old_d, new_deg=new_d, new_deg95=new_d95,
                old_bound=old_b, new_bound=new_b,
                old_deg_pairs=[[a, b, round(r, 3)] for a, b, r in op],
                new_deg_pairs=dp[:6], new_deg95_pairs=dp95[:6],
                new_bound_frac=nb,
                new_bound_worst=(ndet or {}).get("worst"),
                new_bound_detail=ndet, why=why, pc_note=(pc or {}).get("note", ""))


def _load_samples(npz_path: str) -> dict:
    z = np.load(npz_path)
    return {k: z[k] for k in z.files
            if k not in ("f", "Zre", "Zim")}


def scan(limit: int | None = None) -> list[dict]:
    rows = []
    files = sorted(glob.glob(os.path.join("out", "**", "real_*.json"),
                             recursive=True))
    for p in files:
        npz = p[:-5] + "_samples.npz"
        if not os.path.exists(npz):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:                                  # noqa: BLE001
            continue
        if not d.get("fit_performed", True):
            continue
        s = _load_samples(npz)
        family = "nested" if "Rs" in s else "voigt"
        k = d.get("K")
        pc = d.get("param_corr") or None
        if family == "voigt" and (not pc or not pc.get("names")):
            try:                        # 早期产物没存 param_corr：现算（只读后验）
                mm = B.match_components(s)
                pc = CORR.param_corr(s, include_L=bool(d.get("include_L")),
                                     mm=mm)
            except Exception:                              # noqa: BLE001
                pc = None
        st = _states(s, family, k, pc)
        rows.append(dict(file=p, family=family, K=k,
                         zeta=d.get("zeta"), **st))
        if limit and len(rows) >= limit:
            break
    return rows


def summarize(rows: list[dict]) -> dict:
    def _flip(key_old, key_new):
        c = Counter((r[key_old], r[key_new]) for r in rows)
        return {f"{a}->{b}": n for (a, b), n in sorted(c.items())}

    d_flip = _flip("old_deg", "new_deg")
    d95_flip = _flip("old_deg", "new_deg95")
    b_flip = _flip("old_bound", "new_bound")

    def _nflip(d):
        return sum(n for k, n in d.items() if k.split("->")[0] != k.split("->")[1])

    # 旧路径到底在抓什么：按对名统计（这决定了它是"简并"还是"同量异名"）
    inv: Counter = Counter()
    for r in rows:
        for a, b, _ in r["old_deg_pairs"]:
            if r["K"] == 1 or True:
                inv[(r["K"], "+".join(sorted((str(a), str(b)))))] += 1
    return dict(n=len(rows),
                n_deg_flip=_nflip(d_flip), n_deg_flip_at95=_nflip(d95_flip),
                n_bound_flip=_nflip(b_flip),
                deg=d_flip, deg_at95=d95_flip, bound=b_flip,
                old_pair_inventory={f"K={k} {nm}": n
                                    for (k, nm), n in inv.most_common(10)},
                by_k={str(kk): nn for kk, nn in
                      Counter(r["K"] for r in rows).items()},
                deg_flip_by_k={f"K={kk} {a}->{b}": nn
                               for (kk, a, b), nn in
                               Counter((r["K"], r["old_deg"], r["new_deg"])
                                       for r in rows).items()},
                bound_triggered_by_k={f"K={kk}": nn for kk, nn in
                                      Counter(r["K"] for r in rows
                                              if r["new_bound"] == "triggered")
                                      .items()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rows = scan(a.limit)
    if not rows:
        print("没找到 real_*.json + *_samples.npz 配对")
        return 1
    sm = summarize(rows)

    print(f"\n样本 {sm['n']} 条（K 分布 {sm['by_k']}）")
    print(f"\n== param_degenerate 翻转 {sm['n_deg_flip']} 条"
          f"（按旧阈值 0.95 只滤一遍：{sm['n_deg_flip_at95']} 条）==")
    for k, v in sm["deg"].items():
        print(f"   {k:<34}{v:>5}")
    print("   其中只由阈值 0.90/0.95 差异造成的：")
    for k, v in sm["deg_at95"].items():
        print(f"      {k:<31}{v:>5}")
    print(f"\n== param_at_bound 翻转 {sm['n_bound_flip']} 条 ==")
    for k, v in sm["bound"].items():
        print(f"   {k:<34}{v:>5}")
    print("\n== 旧路径实际命中的参数对（前 10） ==")
    for k, v in sm["old_pair_inventory"].items():
        print(f"   {k:<34}{v:>5}")

    print("\n== 翻转逐条（最多 12 条） ==")
    shown = 0
    for r in rows:
        if r["old_deg"] == r["new_deg"] and r["old_bound"] == r["new_bound"]:
            continue
        print(f"  {os.path.basename(r['file']):<44} K={r['K']} "
              f"deg {r['old_deg']}->{r['new_deg']}  "
              f"bound {r['old_bound']}->{r['new_bound']}")
        if r["old_deg"] != r["new_deg"]:
            print(f"      旧对={r['old_deg_pairs'][:4]}  新对={r['new_deg_pairs']}")
        if r["old_bound"] != r["new_bound"]:
            print(f"      worst={r['new_bound_worst']} "
                  f"frac={r['new_bound_frac']}  "
                  f"detail={r['new_bound_detail']}")
        shown += 1
        if shown >= 12:
            print("  ...")
            break

    payload = dict(script="docs/round_2026-11-11/guard_fix_flip.py",
                   old_strong=OLD_STRONG,
                   at_bound_frac=S.AT_BOUND_FRAC,
                   bound_eps=S.BOUND_EPS,
                   prior_bounds={k: {kk: list(vv) for kk, vv in v.items()}
                                 for k, v in S.PRIOR_BOUNDS.items()},
                   summary=sm, rows=rows)
    Path(os.path.dirname(OUT_JSON) or ".").mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\n落盘 {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
