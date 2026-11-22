"""层 0 门缺失的影响：按 results.json 里已有的 rhat/div 离线重选 best。

**不重跑任何 MCMC**。`out/batch21/results.json` 的每个候选都自带
`rms / rhat / div / mode`，所以"如果当时加了采样门，best 会换成谁"
可以纯离线回答。

判据（与 `circuit_runner.SAMPLING_RHAT_MAX` / 层 0 定义一致）：
  可用 = `ok and rhat <= 1.05 and div == 0`
  best_L0 = 可用候选里 rms 最小者；无可用 ⇒ `non_evaluable`（不给 rms）
对照：
  best_old = 现状脚本口径 `min(rms)`，不看采样（`batch_fit_21.py:159`）
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
SRC = os.path.join(REPO, "out", "batch21", "results.json")
OUT = os.path.join(REPO, "out", "layer0_gate")
RHAT_MAX = 1.05


def usable(c):
    return (c.get("ok") and c.get("rms") is not None
            and c.get("rhat") is not None and c["rhat"] <= RHAT_MAX
            and int(c.get("div") or 0) == 0)


def main() -> int:
    recs = json.load(open(SRC, encoding="utf-8"))
    rows = []
    for r in recs:
        fixed = [c for c in r["cands"] if c.get("mode") == "fixed"
                 and c.get("ok") and c.get("rms") is not None]
        if not fixed:
            continue
        old = min(fixed, key=lambda c: c["rms"])
        ok = [c for c in fixed if usable(c)]
        new = min(ok, key=lambda c: c["rms"]) if ok else None
        rows.append(dict(
            cls=r["class"], spec=os.path.basename(r["path"]),
            old_label=old["label"], old_rms=float(old["rms"]),
            old_rhat=old.get("rhat"), old_div=old.get("div"),
            old_usable=usable(old),
            new_label=(new["label"] if new else None),
            new_rms=(float(new["rms"]) if new else None),
            new_rhat=(new.get("rhat") if new else None),
            new_div=(new.get("div") if new else None),
            n_usable=len(ok), n_fixed=len(fixed),
            changed=(new is None or new["label"] != old["label"]),
        ))

    print("%-6s %-22s %-14s %8s %7s %-14s %8s" % (
        "类", "谱", "旧 best", "rms%", "R̂", "新 best(层0)", "rms%"))
    print("-" * 96)
    for x in rows:
        print("%-6s %-22s %-14s %8.2f %7s %-14s %8s%s" % (
            x["cls"], x["spec"][:22], x["old_label"][:14], x["old_rms"],
            ("%.2f" % x["old_rhat"]) if x["old_rhat"] is not None else "-",
            (x["new_label"] or "**none**")[:14],
            ("%.2f" % x["new_rms"]) if x["new_rms"] is not None else "-",
            "   <== 换" if x["changed"] else ""))

    n = len(rows)
    bad_old = [x for x in rows if not x["old_usable"]]
    none_new = [x for x in rows if x["new_label"] is None]
    changed = [x for x in rows if x["changed"]]

    print("\n" + "=" * 96)
    print("谱数 %d" % n)
    print("旧口径 best 采样不过：%d 条（%.0f%%）" % (len(bad_old), 100.0 * len(bad_old) / n))
    print("加层 0 门后换 best：%d 条" % len(changed))
    print("加门后无可用候选（non_evaluable）：%d 条" % len(none_new))
    if changed:
        d = [x["new_rms"] - x["old_rms"] for x in changed if x["new_rms"] is not None]
        if d:
            print("换后 rms 变化：中位 %+.2f pp，范围 %+.2f ~ %+.2f"
                  % (sorted(d)[len(d) // 2], min(d), max(d)))
    # 按类汇总
    print("\n按类（旧 best 是否采样不过 / 是否换）")
    for cls in dict.fromkeys(x["cls"] for x in rows):
        sub = [x for x in rows if x["cls"] == cls]
        print("  %-6s n=%d  旧不过 %d  换 %d  无可用 %d" % (
            cls, len(sub), sum(1 for x in sub if not x["old_usable"]),
            sum(1 for x in sub if x["changed"]),
            sum(1 for x in sub if x["new_label"] is None)))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gate.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(dict(rule="usable = ok and rhat<=%.2f and div==0" % RHAT_MAX,
                       n=n, rows=rows), fh, ensure_ascii=False,
                  indent=1, allow_nan=False)
    print("\n->", os.path.join(OUT, "gate.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
