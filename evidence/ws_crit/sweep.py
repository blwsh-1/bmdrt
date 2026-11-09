# -*- coding: utf-8 -*-
"""从 bench_rows.json 扫 45° 段长阈值，看灵敏度/特异度怎么换。

不做手工调参：把 `min_dec` 当唯一阈值扫一遍，正例（真含扩散元件）与
阴例（不含）的反应各自画出来，再决定工作点。
"""
import json
import numpy as np

POS = ["ws_vis", "ws_buried", "wo_in", "wo_below", "wshort_in"]
NEG = ["none_2zarc", "rq05_in", "sorp_rl"]
AMB = ["rq05_low"]

rows = json.load(open(r"D:/eis训练数据/bmdrt/evidence/ws_crit/bench_rows.json", encoding="utf-8"))

for win in sorted({r["window"] for r in rows}):
    sub = [r for r in rows if r["window"] == win]
    print(f"\n===== {win} =====")
    print(f"{'thr':>5s} {'sens_vis':>8s} {'sens_buried':>11s} {'sens_wo_in':>10s} "
          f"{'sens_wo_bel':>11s} {'sens_wshort':>11s} {'spec_neg':>8s} {'amb':>5s}")
    for thr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        def frac(cls):
            s = [r for r in sub if r["cls"] == cls]
            ok = sum((r["band"] >= thr or r["amp45"]) and r["term"] != "noisy" for r in s)
            return ok / len(s)
        negok = 1.0 - np.mean([frac(c) for c in NEG])
        print(f"{thr:5.1f} {frac('ws_vis'):8.2f} {frac('ws_buried'):11.2f} "
              f"{frac('wo_in'):10.2f} {frac('wo_below'):11.2f} {frac('wshort_in'):11.2f} "
              f"{negok:8.2f} {frac('rq05_low'):5.2f}")
