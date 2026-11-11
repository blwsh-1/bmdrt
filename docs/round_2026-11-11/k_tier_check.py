"""R22 检查：`select_k()` 的采样分级分档是否真的生效（旧行为 vs 新行为）。

背景
----
`bm_real.select_k()` 的 docstring 一直写「**优先在严格通过的里选**，没有才降级」，
但 2026-11-11 的审查发现实现与文档不一致：分级 1（可用）和分级 2（严格）被塞进
**同一个** `ok` 池按 `elpd` 混排，于是降级候选可以赢过严格通过的候选；而
`status` 只看「池非空」就报 `converged`。两者叠加产生自相矛盾的输出：

    选中 K 的采样分级 = 1（可用），k_selection_status = "converged"

历史产物里这个组合出现 **3/381** 次（全在 `AZ31-12Li_0` 两个文件）。
本脚本不跑 MCMC，只用构造候选逐条验证分档规则。

用法
----
    python docs/round_2026-11-11/k_tier_check.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import bm_real as R  # noqa: E402


def cand(K: int, grade: int, elpd: float, *, dse: float = 3.0,
         elpd_diff=None, compared: bool = True) -> dict:
    """只造 `select_k` 真正读到的字段；其余一律给安全默认。"""
    return dict(
        K=K, grade=grade, elpd=elpd, dse=dse, elpd_diff=elpd_diff,
        comparison_performed=compared,
        grade_bad=[] if grade == 2 else [f"分级{grade}：R̂=1.03"],
        grade_loose=[f"分级{grade}：不可用"] if grade == 0 else [],
        p_peak_out_max=0.0, frac_neg=0.0, bad_k=0.0,
    )


def main() -> int:
    ok = True

    def chk(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  {'✓' if good else '✗'} {name}: {got}（应为 {want}）")

    print("=" * 68)
    print("一、核心缺陷：降级候选 elpd 更高时能否越过严格候选（R22 主修）")
    # 严格档 K=2 elpd=-100；可用档 K=3 elpd=-96（更贴 4 nats）
    res = [cand(2, 2, -100.0), cand(3, 1, -96.0)]
    best, note = R.select_k(res)
    chk("选中 K（必须留在严格档）", best["K"], 2)
    chk("状态（被选中者是分级 2）", best["selection_status"], "converged")
    chk("选中的是哪一档", best["selection_tier"], 2)
    said = any("不参与最优档" in x for x in note)
    ok &= said
    print(f"  {'✓' if said else '✗'} note 显式说明 K=3 被挡在最优档外: {said}")
    print(f"     note 原文：{[x for x in note if '不参与最优档' in x]}")

    print("\n二、严格池为空时必须降到可用档（不能一律 rejected）")
    best, note = R.select_k([cand(2, 1, -100.0), cand(3, 1, -96.0)])
    chk("选中 K", best["K"], 3)
    chk("状态（选中者为分级 1）", best["selection_status"], "degraded")
    chk("档位", best["selection_tier"], 1)

    print("\n三、全部不可用 → rejected，且不包装成结论")
    best, note = R.select_k([cand(2, 0, -100.0), cand(3, 0, -96.0)])
    chk("状态", best["selection_status"], "rejected")
    chk("档位", best["selection_tier"], 0)
    said = any("不输出最优 K 的结论" in x for x in note)
    ok &= said
    print(f"  {'✓' if said else '✗'} 首条写明不输出结论: {said}")

    print("\n四、简约优先只在同档内生效（不得被降级的小 K 触发）")
    # 严格档只有 K=3；可用档有 K=1 且 elpd 差 1（< max(2,dse)）
    # 旧实现：池=两档混排，K=1 会被「简约优先」拉成最终选择 → 用分级 1 的 K=1 出结论
    res = [cand(3, 2, -100.0, dse=3.0), cand(1, 1, -101.0, dse=3.0, elpd_diff=1.0)]
    best, note = R.select_k(res)
    chk("选中 K（不得被可用档的 K=1 拉走）", best["K"], 3)
    chk("档位", best["selection_tier"], 2)

    print("\n五、同档内简约优先仍然有效（回归：别把规则 4 修没了）")
    res = [cand(3, 2, -100.0, dse=3.0), cand(2, 2, -101.0, dse=3.0, elpd_diff=1.0)]
    best, note = R.select_k(res)
    chk("选中 K（应取更小的 K=2）", best["K"], 2)
    said = any("简约优先" in x for x in note)
    ok &= said
    print(f"  {'✓' if said else '✗'} note 记录简约优先: {said}")

    print("\n六、硬性剔除优先于分档（分级 1 但 LOO 不可用 → 不能进可用池）")
    bad = cand(2, 1, float("nan"))
    res = [cand(3, 2, -100.0), bad]
    best, note = R.select_k(res)
    chk("选中 K", best["K"], 3)
    said = any("K=2 **剔除**" in x for x in note)
    ok &= said
    print(f"  {'✓' if said else '✗'} K=2 被硬剔除: {said}")

    print("\n" + "=" * 68)
    print("全部通过" if ok else "有失败项")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
