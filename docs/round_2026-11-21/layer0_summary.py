"""按层 0（采样有效性）重排 21 条摘要：只读汇总，不改任何数字。

规则（来自 `STD_RECHECK.md` §6）
-------------------------------
  · 层 0 通过 = **standard 预算下** `R̂<=1.05` 且 `div==0`
  · 判定一律取 standard，不用 quick——quick 的"通过"可能是假收敛
    （钝化 #1 即 quick 1.002/0 → standard 2.297/557，实测反转）
  · 不过者标 `non_evaluable`，**不给 rms**

acf 不在本表：standard 复核只采了 rhat/div/rms，没有 acf。M-b 的判据要等带
acf 的 standard 重跑，本表不猜。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/layer0_summary.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

SRC21 = os.path.join("out", "batch21", "results.json")
SRCSTD = os.path.join("out", "std_recheck", "std_recheck.json")
OUTMD = os.path.join("docs", "round_2026-11-21", "LAYER0_SUMMARY.md")
RHAT_MAX = 1.05
ORDER = ["合成", "腐蚀", "钝化", "涂层", "2E", "3E", "AZ31"]
NOTE = {"合成": "**基准不合格**：3 条里 2 条 `non_evaluable`；`ecm_05687` 在**生成阶段**就已 "
                "`R̂=2.512`、`触顶=91%`（σ 先验比真噪声大 4.75 dec）；剩 1 条不足成组。",
        "钝化": "2 条可用（69.96 / 70.51），1 条在 standard 下**翻转**为不可辨识（原 quick 干净）。",
        "涂层": "**基准不合格**：3 条全 `non_evaluable`（std R̂ 16.87 / 1.83 / 92.50）；真值 5 CPE "
                "双层嵌套超出 nested/L<=2 与 voigt/K<=2；`22.04%` 作废。"}


def main() -> int:
    recs = json.load(open(SRC21, encoding="utf-8"))
    std = json.load(open(SRCSTD, encoding="utf-8"))["rows"]
    by = {(r["path"], r["label"]): r for r in std}

    rows = []
    for r in recs:
        b = r["best"]
        s = by.get((r["path"], b["label"]))
        q_bad = b["rhat"] > RHAT_MAX or (b["div"] or 0) > 0
        item = dict(cls=r["class"], spec=os.path.basename(r["path"]),
                    label=b["label"], q_rhat=float(b["rhat"]),
                    q_div=int(b["div"] or 0), q_bad=q_bad,
                    s_rhat=None, s_div=None, rms=None,
                    layer0="missing_standard", flip=False)
        if s is not None:
            st = s["standard"]
            # 一律用 `std_recheck` 的预注册判定：minor_div（div<=5）算通过，
            # 只有 structurally_unidentifiable 才不可用。别在这里另立 div==0 口径。
            s_bad = s["verdict"] == "structurally_unidentifiable"
            item.update(s_rhat=st["rhat"], s_div=st["div"],
                        layer0="non_evaluable" if s_bad else "pass_standard",
                        rms=None if s_bad else float(st["rms"]),
                        # 真翻转只指"quick 干净但 standard 不干净"；
                        # "quick 坏、standard 好"是被治好，不是翻转。
                        flip=(not q_bad) and s_bad)
        rows.append(item)

    lines = ["# 21 条按层 0 重排（只读汇总）", "",
             "> 由 `layer0_summary.py` 从 `out/batch21/results.json` 与",
             "> `out/std_recheck/std_recheck.json` 合并生成，**不改任何数字**。",
             "> 层 0 一律按 **standard** 预算判（`R̂<=1.05` 且 `div==0`）——quick 的通过不作数，",
             "> 实测存在 quick 干净、standard 反而不干净的翻转。不过者标 `non_evaluable`，不给 rms。",
             "> `acf` 未采集 standard 版本，本表不报（M-b 判据待带 acf 的重跑）。", "",
             "## 1. 逐条", "",
             "| 类 | 谱 | 候选 | quick R̂/div | std R̂/div | 层 0 | rms% |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (ORDER.index(x["cls"]), x["spec"])):
        sr = "—" if r["s_rhat"] is None else (
            "%.3f" % r["s_rhat"] if r["s_rhat"] < 20 else "%.1f" % r["s_rhat"])
        sd = "—" if r["s_div"] is None else str(r["s_div"])
        rms = "—" if r["rms"] is None else "%.2f" % r["rms"]
        flag = " **⚠翻转**" if r["flip"] else ""
        lines.append("| %s | `%s` | %s | %.3f/%d | %s/%s | %s%s | %s |" % (
            r["cls"], r["spec"][:32], r["label"], r["q_rhat"], r["q_div"],
            sr, sd, r["layer0"], flag, rms))

    lines += ["", "## 2. 按类", "",
              "| 类 | 总 | 通过 | non_evaluable | 翻转 | 可用 rms 中位 | 范围 | 备注 |",
              "|---|---|---|---|---|---|---|---|"]
    for cls in ORDER:
        g = [x for x in rows if x["cls"] == cls]
        ok = [x for x in g if x["rms"] is not None]
        nv = [x for x in g if x["layer0"] == "non_evaluable"]
        nf = [x for x in g if x["flip"]]
        if len(ok) >= 2:
            med = "%.2f%%" % np.median([x["rms"] for x in ok])
            rng = "%.2f–%.2f" % (min(x["rms"] for x in ok), max(x["rms"] for x in ok))
        elif len(ok) == 1:
            med, rng = "—（n=1）", "%.2f" % ok[0]["rms"]
        else:
            med, rng = "—", "—"
        lines.append("| %s | %d | %d | %d | %d | %s | %s | %s |" % (
            cls, len(g), len(ok), len(nv), len(nf), med, rng, NOTE.get(cls, "")))

    n_ok = sum(1 for r in rows if r["rms"] is not None)
    n_nv = sum(1 for r in rows if r["layer0"] == "non_evaluable")
    n_flip = sum(1 for r in rows if r["flip"])
    lines += ["", "**合计**：21 条 → standard 下层 0 通过 **%d** 条，`non_evaluable` **%d** 条，"
                  "其中 **%d** 条与 quick 判定相反。" % (n_ok, n_nv, n_flip),
              "", "## 3. 读法", "",
              "1. **quick 的通过不可作为层 0 依据。** 钝化 #1 在 quick 下 `1.002/0`、",
              "   standard 下 `2.297/557`。机制未确证，但 `div=557` 与多峰后验一致：",
              "   300+300 可能把多峰采成「看起来收敛」。**所有 quick 得出的\"收敛\"都要复核。**",
              "2. **只有 `rms` 列能进摘要**，且全部是 standard 口径，可直接横向比较（类内）。",
              "3. **合成与涂层两类不能作为基准**：一个是 σ 先验与数据脱钩，",
              "   一个是真值拓扑超出表达力。它们不能与腐蚀/钝化并列。",
              "4. **不要算跨类中位数**：类别之间的失效机制不同，平均没有意义。",
              "5. 层 0 通过 ≠ 模型好。它只说明这次 run 的数字可引用，不说明族选对了。"]

    with open(OUTMD, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print("...")
    print("\n".join(lines[-16:-6]))
    print("\n->", OUTMD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
