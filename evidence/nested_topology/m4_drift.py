"""§16 执行器：非稳态 / 端点漂移的**纯数据**判据（跑前预注册）。

硬边界（`M6_VALIDATION_PLAN.md` §16.0）：
  * 只读实测谱 + `m4_spectrum_health` 的两个既有函数；**不调用任何 fit / 采样**，
    也不读 nested 或 voigt 的拟合残差。
  * 唯一用到的外部标签是 `out/m4_holdout2.json` 的 `state`，**仅用于分组**。
  * 不改模型、不改判定：§13 的 0/3、§14 的 4/14 原样保留。

三条判据（§16.1，阈值全部来自既有来源）：
  D1 `D:shape`        stability_shape().verdict == 'unverified'（书 p.57）
  D2 `D:rough`        drift_roughness().jmax > 0.10（bm_kk 构造表「低频漂移 0.10R」）
  D3 `D:seq_reverse`  A_ind = max(Im) 在 0→5→10→50→100→150 序列里出现回落（纯符号）

用法
----
    python evidence/nested_topology/m4_drift.py
    python evidence/nested_topology/m4_drift.py --json out/m4_drift.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "evidence", "nested_topology"))

import bm_real as R                                    # noqa: E402
from m4_spectrum_health import (IND_TOL, END_TOL, NEG_TOL,  # noqa: E402
                                drift_roughness, stability_shape)
from m4_holdout2 import SPEC_DIR                       # noqa: E402

TIMES = (0, 5, 10, 50, 100, 150)
SERIES = ("4", "8", "12")
JMAX_CUT = 0.10          # bm_kk.py 构造表「低频漂移 0.10R → 1.5% 残差，判 ok」

# `AD_M6_SAMPLING.md` §1 表格原文（Ω）。只作对照展示，**不是判据、不参与计算**。
DOC_TABLE = {
    "4":  (369, 663, 626, 1004, 1175, 1658),
    "8":  (160,  54, 487,  287,  599,  777),
    "12": ( 68,  54, 104,  461,  507,  506),
}


def _key(path: str) -> tuple[str, int]:
    base = os.path.basename(path)[:-4]           # AZ31-8Li_10
    li = base.split("-")[1].split("Li")[0]      # 8
    t = int(base.rsplit("_", 1)[1])             # 10
    return li, t


def _load_all() -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(SPEC_DIR, "AZ31-*.csv"))):
        f, Z = R.load_spectrum(p)
        li, t = _key(p)
        out.setdefault(li, {})[t] = dict(
            spec=os.path.basename(p)[:-4], f=f, Z=Z,
            A_ind=float(np.max(Z.imag)))        # 全谱正虚部峰值（感抗环强度）
    return out


def _seq_flags(allsp: dict) -> dict:
    """D3：按 Li 分组、按时间升序，标出相对前一点的回落。"""
    flags = {}
    for li in SERIES:
        prev = None
        for t in TIMES:
            if t not in allsp[li]:
                continue
            a = allsp[li][t]["A_ind"]
            flags[allsp[li][t]["spec"]] = bool(prev is not None and a < prev)
            prev = a
    return flags


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out/m4_drift.json")
    a = ap.parse_args(argv)

    hold = json.load(open("out/m4_holdout2.json", encoding="utf-8"))
    states = {r["spec"]: r["state"] for r in hold["rows"]}
    assert len(states) == 14, f"§14 应有 14 条，实得 {len(states)}"

    allsp = _load_all()
    seq = _seq_flags(allsp)

    print("§16 非稳态 / 端点漂移（纯数据，不拟合、不采样、不改判定）")
    print(f"  D2 线 jmax > {JMAX_CUT:.2f}（bm_kk 构造表 0.10R）  "
          f"D1 用既有 IND/END/NEG={IND_TOL}/{END_TOL}/{NEG_TOL}")

    print("\nA. 感抗环强度 A_ind = max(Im) 序列（Ω），与 AD_M6_SAMPLING.md §1 表格并列")
    for li in SERIES:
        vals = [allsp[li][t]["A_ind"] for t in TIMES]
        doc = DOC_TABLE[li]
        print(f"  {li}Li 重算: " + " ".join(f"{v:7.0f}" for v in vals))
        print(f"  {li}Li 文档: " + " ".join(f"{v:7.0f}" for v in doc))
        rev = ", ".join(str(t) for k, t in enumerate(TIMES)
                        if k and vals[k] < vals[k - 1]) or "无"
        print(f"       差: " + " ".join(f"{v-d:+7.0f}" for v, d in zip(vals, doc)) +
              f"   回落点: {rev}")

    print(f"\nB. 逐谱（§14 的 14 条）")
    print(f"  {'谱':<15}{'组':<5}{'D1':<20}{'jmax':>7}{'ratio':>7}"
          f"{'A_ind':>8}{'D3':>5}  标记")
    rows = []
    for spec, state in states.items():
        li, t = (spec.split("-")[1].split("Li")[0], int(spec.rsplit("_", 1)[1]))
        d = allsp[li][t]
        f, Z = d["f"], d["Z"]
        sh = stability_shape(f, Z)
        dr = drift_roughness(f, Z)
        flags = []
        if sh["verdict"] == "unverified":
            flags.append("D:shape")
        if np.isfinite(dr["jmax"]) and dr["jmax"] > JMAX_CUT:
            flags.append("D:rough")
        if seq.get(spec):
            flags.append("D:seq_reverse")
        print(f"  {spec:<15}{('未过' if state != '过' else '对照'):<5}"
              f"{sh['verdict']:<20}{dr['jmax']:>7.3f}{dr['ratio']:>7.2f}"
              f"{d['A_ind']:>8.0f}{('Y' if seq.get(spec) else '-'):>5}  "
              + ",".join(flags))
        rows.append(dict(spec=spec, li=li, t=t, state=state,
                         shape=sh["verdict"], shape_why=sh["why"],
                         jmax=dr["jmax"], ratio=dr["ratio"],
                         rough_low=dr["low"], rough_mid=dr["mid"],
                         A_ind=d["A_ind"], seq_reverse=bool(seq.get(spec)),
                         flags=flags))

    fail = [r for r in rows if r["state"] != "过"]
    ctrl = [r for r in rows if r["state"] == "过"]
    print(f"\nC. 区分力（未过 {len(fail)} / 对照 {len(ctrl)}）")
    power = {}
    for tag in ("D:shape", "D:rough", "D:seq_reverse"):
        nf = sum(tag in r["flags"] for r in fail)
        nc = sum(tag in r["flags"] for r in ctrl)
        rf, rc = nf / len(fail), nc / len(ctrl)
        verdict = "无区分力" if rc >= rf else ("有区分力" if nf else "无命中")
        power[tag] = dict(n_fail=nf, n_ctrl=nc, rate_fail=rf, rate_ctrl=rc,
                          verdict=verdict)
        print(f"  {tag:<16}{nf:>3}/{len(fail)}  vs {nc:>2}/{len(ctrl)}   → {verdict}")
    na = sum(r["shape"] == "not_applicable" for r in rows)
    print(f"  D1 前提不适用（存在感抗 → 书 p.57 规则不成立）：{na}/14"
          "  ← 既不算命中也不算通过")

    strong = [t for t, v in power.items() if v["verdict"] == "有区分力"]
    covered = [r["spec"] for r in fail if any(t in r["flags"] for t in strong)]
    ctrl_hit = [r["spec"] for r in ctrl if r["flags"]]
    if strong and len(covered) == len(fail) and not ctrl_hit:
        concl = "第一句：未过谱被 D 类解释，且对照组无命中 → 可作下一步假设的输入"
    elif strong and covered:
        concl = ("第二句：部分命中（" + str(len(covered)) + "/" + str(len(fail)) +
                 " 条未过谱被有区分力的 D 判据覆盖），其余仍归 §15 原类")
    else:
        concl = "第三句：判据无区分力 / 前提不适用 → 归因失败，如实写"
    print(f"\nD. 结论：{concl}")
    print("   §13 的 0/3、§14 的 4/14 不变；不接生产。")

    out = dict(prereg=dict(section="M6_VALIDATION_PLAN.md §16",
                           jmax_cut=JMAX_CUT, times=list(TIMES),
                           d1_tol=dict(ind=IND_TOL, end=END_TOL, neg=NEG_TOL),
                           doc_table={k: list(v) for k, v in DOC_TABLE.items()},
                           note="纯数据、探索性；不改判定"),
               A_ind={li: {str(t): allsp[li][t]["A_ind"] for t in TIMES}
                      for li in SERIES},
               rows=rows, power=power, conclusion=concl,
               covered_by_strong=covered, ctrl_hit=ctrl_hit,
               d1_not_applicable=na)
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    with open(a.json, "w", encoding="utf-8", newline="") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\nJSON → {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
