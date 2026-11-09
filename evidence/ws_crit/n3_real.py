# -*- coding: utf-8 -*-
"""N3：在 18 条真实谱上**只读登记** `verdict` / `family`（不改判据、不改数据）。

为什么要这一步
--------------
`M6_VALIDATION_PLAN.md` 把这些实测谱的 Ws 状态记成 `non_evaluable`——理由
是"没有 `truth`、`independence_status=inferred`"。本脚本**不改变**这个登记，
只把"数据侧看得见什么"逐条写下来，供人工对照。三件事必须同时出现，
缺一条就会被误读：

  1. `verdict`（`candidate` / `reject` / `undecidable`）——**不是**"有没有扩散"；
  2. `truth = unknown`、`independence_status = inferred`——不可算 TP/FP、
     不进 G8b（`WS_DECISION_PROJECT` 的边界，一字不改）；
  3. `f_min`——`Ws` 类的形状量依赖窗口下限（`WS_CRITERION_PLAN` §4）。

同时并列现行 L1（`bm_kk.diffusion_tail_candidate`）的 `amp_ok`/`is_candidate`，
这样"换判据会不会改变结论"是可见的，而不是靠叙述。

产物：`real_register.json`（18 条全字段）+ 标准输出的分组表。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:/eis训练数据/bmdrt")
sys.path.insert(0, r"D:/eis训练数据/bmdrt/evidence/ws_crit")

import bm_kk
import bm_wscrit as W

ROOT = Path(r"D:/eis训练数据/bmdrt")
SRC = ROOT / "out" / "_m6_samples" / "z_csv"
OUT = ROOT / "evidence" / "ws_crit" / "real_register.json"


def load(p: Path):
    d = np.genfromtxt(p, delimiter=",", names=True)
    names = d.dtype.names
    return np.asarray(d[names[0]], float), np.asarray(d[names[1]], float) \
        + 1j * np.asarray(d[names[2]], float)


def main():
    files = sorted(SRC.glob("*.csv"))
    rows = []
    for p in files:
        f, Z = load(p)
        old = bm_kk.diffusion_tail_candidate(f, Z)
        try:
            d = W.ws_diagnostics(f, Z)
        except Exception as e:                      # 数据本身无法判，如实记
            rows.append(dict(name=p.stem, n=int(f.size), f_min=float(f.min()),
                             f_max=float(f.max()), error=f"{type(e).__name__}: {e}"))
            continue
        rows.append(dict(
            name=p.stem, n=int(f.size),
            f_min=float(f.min()), f_max=float(f.max()),
            # 现行 L1（只读调用，用于并列，不作判据）
            l1_amp_ok=bool(old.get("amp_ok")) if old.get("amp_ok") is not None else None,
            l1_candidate=bool(old.get("is_candidate")),
            l1_slope_z=float(old["slope"]) if old.get("slope") is not None else None,
            # 新判据
            slope_im=d["slope_im"], r2_im=d["r2_im"], im_usable=d["im_usable"],
            amp45=d["amp45"], amp90=d["amp90"],
            band_dec=d["band_dec"], band_dec_ny=d["band_dec_ny"],
            band_dec_im=d["band_dec_im"], reach_edge=d["reach_edge"],
            term=d["term"], term_se=d["term_se"],
            term_im_ratio=d["term_im_ratio"],
            family=d["family"], verdict=d["verdict"],
            # 边界标签：与 M6 的登记保持一致，不因本脚本而变
            truth="unknown", independence_status="inferred",
        ))

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    hdr = (f"{'name':16s} {'n':>3s} {'f_min':>7s} | {'L1:amp':>6s} {'L1:cand':>7s} "
           f"| {'s_im':>6s} {'imratio':>7s} {'band':>5s} {'term':>13s} "
           f"{'family':>14s} {'verdict':>11s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['name']:16s} {r['n']:3d} {r['f_min']:7.2g} | ERROR {r['error']}")
            continue
        print(f"{r['name']:16s} {r['n']:3d} {r['f_min']:7.2g} | "
              f"{str(r['l1_amp_ok']):>6s} {str(r['l1_candidate']):>7s} | "
              f"{r['slope_im']:6.2f} {r['term_im_ratio']:+7.2f} {r['band_dec']:5.2f} "
              f"{r['term']:>13s} "
              f"{r['family']:>14s} {r['verdict']:>11s}")

    # ---- 汇总：只数类别，不解释成物理结论 ----
    ok = [r for r in rows if "verdict" in r]
    print()
    print("汇总（18 条实测，truth 全部 unknown、independence_status 全部 inferred）：")
    for v in ("candidate", "reject", "undecidable"):
        n = sum(r["verdict"] == v for r in ok)
        print(f"  {v:12s} {n:2d}/{len(ok)}")
    print("  家族分布：", {k: sum(r["family"] == k for r in ok)
                      for k in ("Ws_or_wide_CPE", "Wo_like", "Wshort_like",
                                "inductive_tail", "unknown")})
    chg = [r["name"] for r in ok if r["l1_candidate"] != (r["verdict"] == "candidate")]
    print(f"  与现行 L1 的 candidate 判定不一致：{len(chg)}/{len(ok)} → {chg}")
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
