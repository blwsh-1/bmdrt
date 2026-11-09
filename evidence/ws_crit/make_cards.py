# -*- coding: utf-8 -*-
"""生成 `EVIDENCE_CARDS.md` 的样例（只读，不跑拟合）。

两个用途：
  1. 给人看「证据卡长什么样」——Blank 走 `--auto-mechanism-full` 的新结果
     （`grade` 是字段），AZ31 走旧 JSON（`grade` 由 `bm_grade` 同表补算），
     两条路径都覆盖；
  2. 顺手把 WP2 的 blank 四臂拼成 L2 多模型并列表——`multi_model` 在
     `config.json` 里默认关，这里显式打开，只为样例。

用法：
    D:/autoeis_env/Scripts/python.exe evidence/ws_crit/make_cards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bm_evidence as EV                                  # noqa: E402

RUNS = [
    ("Blank_0.1KBr（--auto-mechanism-full，新跑）",
     "out/_card_probe/Blank_0.1KBr_pH10_EIS_PtCu_1.2V/"
     "real_Blank_0.1KBr_pH10_EIS_PtCu_1.2V.json"),
    ("AZ31-4Li_100（--auto-mechanism，旧 JSON：grade 由同表补算）",
     "out/_l123/az31/AZ31-4Li_100/real_AZ31-4Li_100.json"),
]

ARMS = [
    ("k1_none", "out/_wp2_blank/k1_none/summary.json"),
    ("k1_Ws", "out/_wp2_blank/k1_Ws/summary.json"),
    ("k2_none", "out/_wp2_blank/k2_none/summary.json"),
    ("k2_Ws", "out/_wp2_blank/k2_Ws/summary.json"),
]

OUT = Path(__file__).with_name("EVIDENCE_CARDS.md")


def main():
    runs, labels = [], []
    for lbl, rel in RUNS:
        p = ROOT / rel
        if not p.exists():
            print(f"[跳过] 不存在 {rel}")
            continue
        r = EV.load_run(p)
        r["_source_path"] = rel
        runs.append(r)
        labels.append(lbl)
    if not runs:
        print("没有可用的真实结果，先跑一条 --auto-mechanism-full")
        return 2

    doc = EV.build_document(runs, labels)
    items = []
    for lbl, rel in ARMS:
        p = ROOT / rel
        if p.exists():
            items.append((lbl, EV.load_run(p)))
    if items:
        doc["multi_model"] = EV.multi_model_table(items)

    md = EV.render_markdown(doc)
    head = ("# 证据卡样例（只读生成，来自真实结果）\n\n"
            "> 生成：`python bm_evidence.py <json> --md`；"
            "多模型表：`--multi-model`（config 默认关闭，本样例用 API 打开）。\n"
            "> 本文件由 `evidence/ws_crit/make_cards.py` 生成，可随时重跑覆盖。\n\n")
    body = md.split("\n", 2)[2] if md.startswith("# ") else md
    OUT.write_text(head + body, encoding="utf-8", newline="\n")
    print(f"已写 {OUT}（{len(runs)} 个 run，{len(items)} 个 arm）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
