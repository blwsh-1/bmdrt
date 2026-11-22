"""M-b 卡点与动态范围的关联（只读）。

假说
----
§13/§14 两次验收都卡在 `acf<=0.50`（M-b）。若某条谱的 D 很大，`free` 的
sigma=0.01*zmax 会让高频段的信噪比掉到 1 以下，高频残差因此不被似然惩罚，
NUTS 不会去拟合它 ⇒ 高频残差保留结构 ⇒ `acf` 高、`drop`（相对 voigt 的
改善）小。这能解释"精度够（rms 低）但结构不够（acf 高）"。

本脚本只读 `out/m4_holdout2.json` 与原始谱，不重跑任何拟合。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/mb_d_link.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import bm_real as R  # noqa: E402
import circuit_runner as RU  # noqa: E402

SRC = os.path.join("out", "m4_holdout2.json")
SPEC_DIR = os.path.join("out", "_m6_samples", "z_csv")
OUT = os.path.join("out", "noise_scale_probe")


def main() -> int:
    d = json.load(open(SRC, encoding="utf-8"))
    rows = []
    for r in d["rows"]:
        p = os.path.join(SPEC_DIR, r["spec"] + ".csv")
        if not os.path.exists(p):
            print("missing", p)
            continue
        diag = RU.dynamic_range_diag(R.load_spectrum(p)[1])
        a_nest = float(r["rows"][0]["acf"])
        a_voigt = float(r["voigt_base"]["acf"])
        rows.append(dict(spec=r["spec"], state=r["state"], D=diag["impedance_dynamic_range"],
                         snr=diag["free_hf_snr_ref"], acf_nest=a_nest,
                         acf_voigt=a_voigt, drop=a_voigt - a_nest,
                         rms=float(r["rows"][0]["rms_rel"]),
                         warn=diag["free_hf_warning"], severe=diag["free_hf_severe"]))

    print("%-14s %8s %8s %7s %7s %7s %6s  %s" % (
        "spec", "D", "snr_ref", "acf_n", "acf_v", "drop", "rms%", "state"))
    print("-" * 78)
    for r in rows:
        print("%-14s %8.1f %8.3g %7.3f %7.3f %+7.3f %6.2f  %s" % (
            r["spec"], r["D"], r["snr"], r["acf_nest"], r["acf_voigt"],
            r["drop"], r["rms"], r["state"]))

    D = np.array([r["D"] for r in rows])
    acf = np.array([r["acf_nest"] for r in rows])
    drop = np.array([r["drop"] for r in rows])
    lD = np.log10(D)
    pr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    from scipy.stats import spearmanr
    print("-" * 78)
    print("n=%d  D 中位 %.1f（%.1f–%.1f）" % (len(rows), np.median(D), D.min(), D.max()))
    print("log D vs acf_nested : pearson %+.2f  spearman %+.2f" % (
        pr(lD, acf), spearmanr(lD, acf).statistic))
    print("log D vs drop       : pearson %+.2f  spearman %+.2f" % (
        pr(lD, drop), spearmanr(lD, drop).statistic))
    ok = [r for r in rows if r["state"] != "未过"]
    bad = [r for r in rows if r["state"] == "未过"]
    if ok:
        print("过 的 %d 条：D 中位 %.1f，acf 中位 %.3f" % (
            len(ok), np.median([r["D"] for r in ok]),
            np.median([r["acf_nest"] for r in ok])))
    print("未过 %d 条：D 中位 %.1f，acf 中位 %.3f" % (
        len(bad), np.median([r["D"] for r in bad]),
        np.median([r["acf_nest"] for r in bad])))
    print("M-b 单独命中：%d/%d；D>100 的谱里 M-b 命中 %d/%d" % (
        sum(1 for r in rows if r["acf_nest"] <= 0.50), len(rows),
        sum(1 for r in rows if r["D"] > 100 and r["acf_nest"] <= 0.50),
        sum(1 for r in rows if r["D"] > 100)))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "mb_d_link.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1, allow_nan=False)
    print("->", os.path.join(OUT, "mb_d_link.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
