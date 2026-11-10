#!/usr/bin/env python
"""分量「未定位」护栏（`bm_select.unresolved_components` / guard `comp_unresolved`）自检。

分两段，**不混报**：

  C. 门禁逻辑（秒级）：`--gate-comp` 改变什么、什么时候整族降级（§9.4.18.2）。
  A. 逻辑段（不采样，秒级）：用合成后验打真假阳性 / 四态语义。
     A1 标签互换但两分量都定得住 → 必须 `ok`（直接取列分位会误判，这是关键假阳性）
     A2 一个分量 τ 跨 5 decade      → `triggered`，且只有该槽被标
     A3 幅值 3%（<5%）但 τ 定得死  → `triggered`（幅值尺单独立生效）
     A4 τ 跨 1.7 decade、幅值 6%    → `ok`（阈值内侧不误伤）
     A5 K=1                         → `not_applicable`（不是 ok）
     A6 后验缺 tau/w                → `not_evaluable`（fail-closed）
     A7 nested 族                    → `not_applicable`

  B. 实测段（`--real`，委托 `m4_armLK_diag.py`，约 4 分钟 4 次拟合）
     **不设门槛**：这一段是证据不是断言。护栏与 R̂ 的分工是
       · 「缺 L 时次分量跑到窗口外（τ→3.6 s）」这格 R̂=1.037 **合格** → 只有护栏会响；
       · 「两链各占一个紧模态」那格 R̂=1.84 响、护栏不响（排序后不摊开）。
     任缺其一都会漏，所以两条必须同时报；若哪天两者完全一致了，
     说明阈值或数据变了，要重写这段而不是调门槛。

跑法：
    python evidence/nested_topology/m4_compempty_check.py [--real]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "evidence", "nested_topology")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bm_select as S                                 # noqa: E402

NAME = "2EMC1-1_0050mA.csv"
N = 800


def _s(tau, w):
    return dict(tau=np.asarray(tau, float), w=np.asarray(w, float))


def _c(s=None, family="voigt"):
    return dict(family=family, raw=({} if s is None else dict(s=s)))


def logic_checks() -> int:
    rng = np.random.default_rng(0)
    bad = 0

    def chk(tag, got, want, extra=""):
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {tag:<34} {str(got):<14} 期望 {want} {extra}")

    # A1 标签互换：一半样本 tau=(1e-2,1e-4)，另一半对调；两个都紧（1% 抖动）
    half = N // 2
    t = np.empty((N, 2))
    jit = rng.normal(0, 0.01, size=(N, 2))
    t[:half] = np.array([1e-2, 1e-4]) * np.exp(jit[:half])
    t[half:] = np.array([1e-4, 1e-2]) * np.exp(jit[half:])
    w = np.tile([0.6, 0.4], (N, 1))
    chk("A1 标签互换但都定得住", S.guard_state(_c(_s(t, w)), N)["comp_unresolved"],
        "ok", f"span={np.round(S.unresolved_components(_s(t, w))['span_dec'], 2).tolist()}")

    # A2 一个分量 τ 跨 5 decade（对数均匀），幅值 15%
    t2 = np.empty((N, 2))
    t2[:, 0] = 1e-2 * np.exp(rng.normal(0, 0.01, N))
    t2[:, 1] = 10 ** rng.uniform(-9, -4, N)
    w2 = np.column_stack([0.85 * np.ones(N), 0.15 * np.ones(N)])
    e2 = S.unresolved_components(_s(t2, w2))
    chk("A2 一个分量 τ 跨 5 decade", S.guard_state(_c(_s(t2, w2)), N)["comp_unresolved"],
        "triggered", f"span={np.round(e2['span_dec'], 2).tolist()} bad={e2['unresolved']}")
    chk("A2 只有该槽被标（槽=按 τ 升序的位置，不是原列号）", e2["unresolved"], [0])

    # A3 幅值 3% 但 τ 紧：幅值尺单独立生效
    t3 = np.column_stack([1e-2 * np.exp(rng.normal(0, 0.01, N)),
                          1e-4 * np.exp(rng.normal(0, 0.01, N))])
    w3 = np.column_stack([0.97 * np.ones(N), 0.03 * np.ones(N)])
    e3 = S.unresolved_components(_s(t3, w3))
    chk("A3 幅值 3%（τ 定得死）", S.guard_state(_c(_s(t3, w3)), N)["comp_unresolved"],
        "triggered", f"amp={np.round(e3['amp_frac'], 3).tolist()} bad={e3['unresolved']}")

    # A4 阈值内侧：τ 跨 1.7 decade、幅值 6%
    t4 = np.empty((N, 2))
    t4[:, 0] = 1e-2 * np.exp(rng.normal(0, 0.01, N))
    t4[:, 1] = 10 ** rng.uniform(-3.85, -2.15, N)     # ≈1.7 decade
    w4 = np.column_stack([0.94 * np.ones(N), 0.06 * np.ones(N)])
    e4 = S.unresolved_components(_s(t4, w4))
    chk("A4 阈值内侧", S.guard_state(_c(_s(t4, w4)), N)["comp_unresolved"], "ok",
        f"span={np.round(e4['span_dec'], 2).tolist()} amp={np.round(e4['amp_frac'], 3).tolist()}")

    # A5 K=1
    chk("A5 K=1", S.guard_state(_c(_s(np.ones((N, 1)) * 1e-2, np.ones((N, 1)))), N)
        ["comp_unresolved"], "not_applicable")

    # A6 缺字段
    chk("A6 后验缺 tau/w", S.guard_state(_c({}), N)["comp_unresolved"], "not_evaluable")

    # A7 nested
    chk("A7 nested 族", S.guard_state(_c(_s(t, w), family="nested"), N)
        ["comp_unresolved"], "not_applicable")

    chk("阈值常数", (S.COMP_UNRES_AMP, S.COMP_UNRES_DECADE), (0.05, 2.0))
    chk("guard 清单含新项", "comp_unresolved" in S.GUARD_B, True)
    return bad


def gate_checks() -> int:
    """C. 门禁逻辑（合成候选，不采样）：`--gate-comp` 到底改变什么。

    C1 gate 关：K=2（elpd 更高、comp 未定位）照旧参赛 → best=K=2，不降级
    C2 gate 开且 K=2 领先 >DEGRADE_MARGIN：K=2 被剔 → **整族 degraded**（并列），
       这正是 §9.4.18.2 预先写死的代价
    C3 gate 开但 K=2 只领先 2 nats（< DEGRADE_MARGIN）：干净回落到 K=1，不降级
    C4 `_k_note` 只在 k_eff<K 时出话，且不塞进 reason
    """
    bad = 0

    def chk(tag, got, want):
        nonlocal bad
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {tag:<40} {str(got):<22} 期望 {want}")

    def cand(label, dof, elpd, comp, k_eff):
        return dict(family="voigt", label=label, dof=dof, elpd=float(elpd),
                    pointwise=np.linspace(1.0, 2.0, 20), rhat=1.01, div=0,
                    ok=True, raw={}, comp_state=comp, k_eff=k_eff)

    c1 = cand("K=1", 1, 100.0, "not_applicable", 1)
    c2 = cand("K=2", 2, 110.0, "triggered", 1)
    c3 = cand("K=2", 2, 102.0, "triggered", 1)

    b, d, _ = S._family_state([c1, c2], 800)                 # 注意：显式传的是位置参数
    chk("C1 gate 关：K=2 照旧参赛", (b["label"], d), ("K=2", False))
    b, d, why = S._family_state([c1, c2], 800, gate_b=True)
    chk("C2 gate 开 + 领先 10 nats：回落 K=1 且整族降级",
        (b["label"], d, "降配参赛" in why), ("K=1", True, True))
    b, d, _ = S._family_state([c1, c3], 800, gate_b=True)
    chk("C3 gate 开 + 领先 2 nats：干净回落 K=1", (b["label"], d), ("K=1", False))
    chk("C4 k_eff<K 才出话", (bool(S._k_note(c2)), bool(S._k_note(c1))), (True, False))
    return bad


def real_checks() -> int:
    """实测段**委托给 `m4_armLK_diag.py`**，不另写一份拟合逻辑。

    第一版本文件自己调 `bm_core.fit` 跑真实谱，与 K 臂（`fit_voigt`，ζ=1.0）
    不是同一个模型，于是「护栏抓到谁」的结论跟着模型漂。证据脚本只能有一处
    拟合入口，否则两份数字永远对不上。
    """
    import m4_armLK_diag as D
    D.main()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="分量未定位护栏自检")
    ap.add_argument("--real", action="store_true", help="追加真实谱 4 次拟合")
    a = ap.parse_args(argv)
    print("A. 逻辑段（合成后验）")
    bad = logic_checks()
    print("\nC. 门禁逻辑（合成候选，不采样）")
    bad += gate_checks()
    if a.real:
        bad += real_checks()
    print(f"\n{'全部通过' if bad == 0 else f'{bad} 项失败'}（B 类 guard 共 "
          f"{len(S.GUARD_B)} 项，新增 comp_unresolved）")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
