"""把 `l3_calib.json` 汇总成「口径 × 窗口」的判据表。

要回答三个问题：
  1. 每个口径跨窗口稳不稳（同类不同窗口的离散度）；
  2. 同一窗口下「真 Ws」与「不该切」的三类分不分得开；
  3. 哪个口径可以用**固定阈值**，哪个必须按窗口查表。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROBES = [("cv_aw", "cv(A_W)", 0.10, True),        # (字段, 名, 建议阈值, 越大越该拒)
          ("share_med", "share", 0.05, True),
          ("snr_med", "snr", 15.0, False)]


def load(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    return [r for r in d["rows"] if r.get("ok")]


def main(argv=None):
    argv = argv or sys.argv[1:]
    src = Path(argv[0]) if argv else HERE / "l3_calib.json"
    rows = load(src)
    if not rows:
        print("没有可用行（都失败或还没跑完）")
        return 1

    kinds = ["ws", "wo", "wo_below", "hn", "none"]
    wins = sorted({r["spec_min"] for r in rows})

    print(f"来源 {src.name}：{len(rows)} 条 ok\n")

    for field, name, thr, big_is_bad in PROBES:
        print(f"===== {name}（阈值 {thr}，{'越大越该拒' if big_is_bad else '越小越该拒'}）=====")
        hdr = f"{'kind':9s} {'a_w':>4s} " + " ".join(f"{w:>8g}" for w in wins) + \
              "  跨窗跨度"
        print(hdr)
        for kind in kinds:
            for a_w in sorted({r.get("a_w", 0.0) for r in rows if r["kind"] == kind}):
                vals = []
                for w in wins:
                    hit = [r for r in rows if r["kind"] == kind
                           and r["spec_min"] == w
                           and r.get("a_w", 0.0) == a_w]
                    vals.append(hit[0].get(field) if hit else None)
                vs = [v for v in vals if v is not None]
                if not vs:
                    continue
                span = (max(vs) / min(vs)) if min(vs) > 0 else float("inf")
                cells = " ".join(f"{v:>8.3f}" if v is not None else f"{'—':>8s}"
                                 for v in vals)
                print(f"{kind:9s} {a_w:>4g} {cells}  {span:8.2f}x")
        print()

    # 闸门不是用来分「真 Ws vs hn」的——后者与 Ws 窗内同形，任何单点判据都分不开
    # （`Ws ≡ 窗外 R||CPE(0.5)`，`WS_CRITERION_PLAN` §5.3）。它要分的是
    # 「信息充足」与「信息不足」：`none`（无扩散硬加）与极弱真 Ws。
    print("===== 同窗口取值区间（按 kind）=====")
    for field, name, thr, big_is_bad in PROBES:
        print(f"\n-- {name}（闸门线 {thr}）--")
        for w in wins:
            cells = []
            for kind in ("ws", "hn", "wo", "wo_below", "none"):
                v = sorted(r[field] for r in rows
                           if r["kind"] == kind and r["spec_min"] == w
                           and r.get(field) is not None)
                if v:
                    cells.append(f"{kind} {min(v):.3f}..{max(v):.3f}")
            print(f" f_min={w:<5g} " + " | ".join(cells))
    print("\n注：`hn` / `wo_below` 与真 Ws 同区间是**已知不可分**（窗内同形），"
          "不是闸门失效；闸门只需把 `none` 挡在阈值之外。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
