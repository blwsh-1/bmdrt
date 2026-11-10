"""把 `out/m4_holdout2.json` 汇总成 markdown 表（§9.4.25 用）。

只读 JSON，不重算、不判定；判定字段直接取执行器写入的 `state`，避免转录时
把口径敏感误写成通过。
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out/m4_holdout2.json")
    a = ap.parse_args()
    d = json.load(open(a.json, encoding="utf-8"))
    rows = [r for r in d["rows"] if "error" not in r]
    errs = [r for r in d["rows"] if "error" in r]

    print("| 谱 | 剔点 | rms_rel % | 低/中/高 % | acf | voigt acf | R̂ | div | M-a | M-b | M-c | 状态 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        row = r["rows"][0]
        v = r["verdict"][0]
        b = r["voigt_base"]
        bd = "/".join(f"{x:.1f}" for x in row["bands"])
        print(f"| `{r['spec']}` | {len(r['dropped_hz'])} | {row['rms_rel']:.2f} | {bd} | "
              f"{row['acf']:.3f} | {b['acf']:.3f} | {row['rhat_max']:.3f} | {row['div']} | "
              f"{'✓' if v['Ma'] else '✗'} | {'✓' if v['Mb'] else '✗'} | "
              f"{'✓' if v['Mc'] else '✗'} | **{r['state']}** |")

    print(f"\n过线 {sum(r['passed_all'] for r in rows)}/{len(rows)}"
          f"（§14.4 要求 {len(rows)}/{len(rows)}）")
    for tag, key in (("未过", "未过"), ("口径敏感", "口径敏感")):
        sel = [r["spec"] for r in rows if r["state"].startswith(key)]
        print(f"  {tag}：{sel}")
    drops = [(r["spec"], r["dropped_hz"]) for r in rows if r["dropped_hz"]]
    print(f"  M1 实际剔点：{drops if drops else '无'}")
    if errs:
        print(f"  失败谱：{[(r['spec'], r['error']) for r in errs]}")

    # 分布（只是描述，不设通过线）
    for nm, fn in (("rms_rel %", lambda r: r["rows"][0]["rms_rel"]),
                   ("acf", lambda r: r["rows"][0]["acf"]),
                   ("R̂", lambda r: r["rows"][0]["rhat_max"])):
        x = np.array([fn(r) for r in rows])
        print(f"  {nm:<10} 中位 {np.median(x):.3f}  范围 {x.min():.3f}–{x.max():.3f}")


if __name__ == "__main__":
    main()
