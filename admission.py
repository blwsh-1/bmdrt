# -*- coding: utf-8 -*-
"""准入状态：**provisional（临时准入）**，含义是"能跑"，不是"过线"。

用户裁决（2026-11-16）：「合起来能跑就行，本来就没有合格，不能声明是过线了的。」
所以本模块只做一件事：把"没过线"这件事**结构化地钉在每份产物上**，让下游
无论怎么读都读不到"通过/合格"。

三层事实必须分开写（合并任一条都会被读成"基本可用"）
----------------------------------------------------
1. `state = "provisional"` —— 这是**接入状态**，回答"能不能调"，不回答"准不准"。
   `passed` 恒为 `False`，且不是可配置开关：要改它得走 §9.4.21 的正式准入。
2. `gap` —— 逐条列出**没达到的具体阈值与实际值**。不许只写一句"未过线"：
   没有数字的"未过"无法复核，也无法判断离通过还有多远。
3. `usable_scope` —— 哪一批数据上有证据、哪一批没有。现在只有 `_10`/`_50`
   中段状态的 4 条谱过 §14，`_0`/`_5`/`_150` 一条都没过。

`judge_caveat` 是 M-B-SIM-01 的结论：M-b 这个判据对"仪器寄生 vs 样品侧结构"
零区分力（拟合后 acf AUC=0.52，95%CI [0.36,0.68]）。所以：
**未过 ≠ 模型差；过 ≠ 模型对**。这句话必须跟着每个 provisional 产物走，
否则"未过"会被当成对模型不利的证据，而它其实只是判据没分辨率。

为什么不做成 config 开关
------------------------
开关会被默认值悄悄打开。这里只用模块级常量：改 `STATE` 必须改代码并过 review，
不存在"某个 config 文件里某个字段写成 true 就变成合格"的路径。
"""
from __future__ import annotations

#: 接入状态。**唯一合法值**：下次改它必须同时改 §9.4.21 的判定，不是配置项。
STATE = "provisional"
#: 是否通过验收线。恒 `False`；不是开关，不接受外部赋值。
PASSED = False

#: 逐条差距。`required` / `actual` / `note` 三件齐全，缺一项就不写进产物。
GAP = {
    "S13_holdout": dict(
        doc="M6_VALIDATION_PLAN.md §13.4",
        required="3/3", actual="0/3",
        note="3 条留出谱全部未过（4Li_150 / 8Li_100 / 12Li_100）"),
    "S14_full": dict(
        doc="NESTED_TOPOLOGY_PROJECT.md §9.4.25",
        required="14/14", actual="4/14",
        note="M-a(rms) 8/14、M-b(acf) 4/14、M-c(R̂/div) 12/14；"
             "达到 §14 的 4 条全是 _10/_50 中段状态"),
    "M-b": dict(
        doc="M6_VALIDATION_PLAN.md §14.3",
        required="acf≤0.50 且比同口径 voigt 低≥0.20", actual="4/14",
        note="未过的 10 条 acf 0.505–0.896；8Li_5 只差 0.005（0.505），不放宽"),
    "S17_four_circuits": dict(
        doc="M6_VALIDATION_PLAN.md §17",
        required="（归因用，不作门槛）", actual="0/10",
        note="四个文献电路在 10 条未过谱上 M-b 全不触发 ⇒ 不是拓扑选错"),
}

#: 有证据的范围 / 无证据的范围。写死，不许用"部分适用"之类含糊话。
USABLE_SCOPE = (
    "仅 _10/_50 中段状态有证据（8Li_10 4.52%、8Li_50 2.77%、"
    "12Li_10 2.68%、12Li_50 3.59%，acf 0.341–0.466）；"
    "_0/_5/_150 全部无证据（rms 9.76–30.80%、acf 0.505–0.896）"
)

#: M-B-SIM-01（类型 C）的结论：判据本身没有归因分辨率。
JUDGE_CAVEAT = (
    "M-b 判据经 M-B-SIM-01 盲测证伪：仪器寄生与样品侧结构两组拟合后 acf "
    "AUC=0.52（95%CI [0.36,0.68]），两类机制都被 10 参数模型吸收；"
    "真实谱高频 acf 0.669–0.916 vs 模拟组 0.39 ⇒ "
    "未过 ≠ 模型差，过 ≠ 模型对（风险 R27）"
)

#: 解锁正式准入需要什么（写死，防止"再跑一组谱试试"）。
UNLOCKED_BY = (
    "M-b 判据被替换或重新标定（需独立信息：开-短路 5 min 校准 / "
    "E1 同状态重复 / 多频窗），并按新判据重算 §13 3/3 与 §14 14/14",
)


def block() -> dict:
    """产物里的 `admission` 块。纯 JSON 可序列化，不含任何"通过"字样。"""
    return dict(
        state=STATE,
        passed=PASSED,
        meaning="能跑，不代表证据等级达到验收线；参数不得当物理量或真值",
        gap={k: dict(v) for k, v in GAP.items()},
        usable_scope=USABLE_SCOPE,
        parameters_are_physical=False,
        judge_caveat=JUDGE_CAVEAT,
        unlocked_by=list(UNLOCKED_BY),
    )


def banner() -> str:
    """一行控制台提示。**不许**出现"通过/合格/过线"。"""
    return (f"[准入状态] {STATE}：§13 {GAP['S13_holdout']['actual']}、"
            f"§14 {GAP['S14_full']['actual']}、M-b {GAP['M-b']['actual']}、"
            f"§17 {GAP['S17_four_circuits']['actual']} —— "
            f"未过验收线，产物只作探索性使用，参数不得当物理量")


def selftest() -> int:
    ok, total, bad = 0, 0, []

    def c(name, cond):
        nonlocal ok, total
        total += 1
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")
        if cond:
            ok += 1
        else:
            bad.append(name)

    b = block()
    c("state=provisional", b["state"] == "provisional")
    c("passed 恒 False", b["passed"] is False)
    c("四条差距齐全", set(b["gap"]) == {"S13_holdout", "S14_full", "M-b",
                                        "S17_four_circuits"})
    c("每条差距都有 required/actual/note",
      all({"required", "actual", "note"} <= set(v) for v in b["gap"].values()))
    c("参数不当物理量", b["parameters_are_physical"] is False)
    c("带 M-b 判据证伪的告诫", "AUC=0.52" in b["judge_caveat"])
    c("写了解锁条件", len(b["unlocked_by"]) >= 1 and "M-b" in b["unlocked_by"][0])
    # 红线：不许出现"通过/合格"这类声明（否定式也不算——grep 会命中，
    # 单看那一行的人会误读）。所以整块文案里这两个字一次都不出现。
    txt = banner() + str(b)
    for w in ("通过", "合格"):
        c(f"文案不含「{w}」", w not in txt)
    # JSON 可序列化（allow_nan=False 的产物直接能写）
    import json
    try:
        json.dumps(b, ensure_ascii=False, allow_nan=False)
        json_ok = True
    except (TypeError, ValueError):
        json_ok = False
    c("block() 可直接 json.dump(allow_nan=False)", json_ok)

    print(f"\n自检 {ok}/{total} 通过" if not bad else f"\nFAIL {len(bad)}/{total}: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
