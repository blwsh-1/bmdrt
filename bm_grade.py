# -*- coding: utf-8 -*-
"""证据资格分档（A / B / C）：只贴标签，不参与任何决策。

为什么单独一层
--------------
`THREE_LAYER_SCOPE.md` §2 把「给不出答案」拆成三种，而现有代码把它们压成
同一个 `uncertain` / `guard_degraded`：

  **A 算法无法判断** —— 采样崩了、LOO 不可用、压根没拟合。
      → 该修代码或重跑。重跑一次可能变的是 A，重跑一万次也不会变的是 B。
  **B 数据无法判断** —— 采样健康，但两个模型在本窗+本噪声下预测等价。
      → 该并列输出、保留多候选；不得单选，也不得说「不存在」。
  **C 物理证据不足** —— 模型比较有结论，但缺「机制唯一性」证据
      （真值 / 独立实验 / 正交表征）。
      → 只报相容性，不得升级为机制结论。**这是真实谱最常见的终态**，
        不是失败。

本模块**只做映射**：把已有的 `status` / `guards` / `level` 贴一个档位，
外加一句「这条推不出什么」。它不读数据、不改判据、不改任何 `status`、
`reason`、`choice`、`delta`。任何一处映射都能指回代码里的具体字段。

硬约束（改这个文件前先看）
--------------------------
1. `grade` 只允许 `"A"` / `"B"` / `"C"` / `None`，**不允许分数**。
   一旦出现「0.72 分」，它就会被当概率读，等于把 B 档伪造成统计结论。
2. 新增字段**不得**改变既有字段的取值。调用方原样给出 `status=None` /
   `reason` / `guards`，本模块只往 dict 里加键。
3. 档位必须能指回代码事实：`GUARD_CLASS` / `STATUS_CLASS` / `LEVEL_CLASS`
   三张表各写一行，新增 guard 时同步补——补不上就是「未分类」，会退化成
   `?`，此时按 A 处理（宁可说「算法不确定」，不要说「数据不确定」）。

用法
----
    import bm_grade as G

    out = G.annotate(out, guards)          # 加 guard_classes / grade / grade_reason
    out = G.annotate_level(out)            # diffusion_resolution 用（按 level 表）
    out["component_grade"] = {k: G.component_grade(v, p) for ...}

自检：`python bm_grade.py`
"""

from __future__ import annotations

GRADE_A, GRADE_B, GRADE_C = "A", "B", "C"

# 排序：A 最该先处理（它能被工程手段消掉）
_RANK = {GRADE_A: 3, GRADE_B: 2, GRADE_C: 1, None: 0}

LABEL = {
    GRADE_A: "算法无法判断（修代码 / 重跑 / 提高预算；不得当数据结论）",
    GRADE_B: "数据无法判断（并列输出、保留多候选；不得单选，也不得说「不存在」）",
    GRADE_C: "物理证据不足（只报与物理相容；不得升级为机制结论）",
}

# ---------------------------------------------------------------- guard 分类
# 键是 `select_mechanism` / `select_adsorption` 里 guards dict 的键名。
GUARD_CLASS = {
    # --- select_mechanism：Ws 侧采样质量（A：工程问题）
    "ws_grade0": GRADE_A,            # 采样分级 0（收敛/ESS 未过）
    "ws_bad_k": GRADE_A,             # Pareto-k>0.7 的点 >10%
    "ws_loo_unavailable": GRADE_A,   # 拿不到 elpd
    # --- select_mechanism：物理/结构护栏（B：数据能力边界）
    "diff_amp_unidentifiable": GRADE_B,   # p_diff_small>0.5，幅值与噪声同量级
    "hn_degenerate": GRADE_B,             # none 的 HN 已呈 Warburg 退化指纹
    "ws_amp_unidentified": GRADE_B,       # cv(A_W)>CV_MAX，幅值没被定住
    # --- select_adsorption：ad 侧采样质量（A）
    "ad_grade0": GRADE_A,
    "ad_bad_k": GRADE_A,
    "ad_loo_unavailable": GRADE_A,
    # --- select_adsorption：物理/结构护栏（B）
    "amp_unidentifiable": GRADE_B,        # P(R_ad<3σ̂)>0.5
    "corner_out_of_window": GRADE_B,      # P(f_c>f_max)>0.5，退化为串联 L
}

# --------------------------------------------------------------- status 分类
# 键是选择器返回的 `status` 字面量。
STATUS_CLASS = {
    # A：算法侧（没拟合 / 没有可配对 LOO / 门控本身缺失）
    "not_tested": GRADE_A,
    "fit_missing": GRADE_A,
    "comparison_incomplete": GRADE_A,
    "ws_taken": GRADE_A,          # 一期屏蔽 Ws+ad 组合（代码范围限制，非数据结论）
    # B：数据侧（形状不符 / 预测等价）
    "uncertain": GRADE_B,
    "degenerate": GRADE_B,        # HN 与 Ws 同形
    "gate_not_strong": GRADE_B,   # 三闸未全过（gate 缺失时在 annotate 里升 A）
    "skipped_inductive": GRADE_B,     # 感抗尾：符号不符即否证窗内容性扩散尾
    "skipped_by_residual": GRADE_B,   # none 已拟合到噪声水平
    "skipped_by_gate": GRADE_B,
    # C：模型比较有结论、但没有机制唯一性证据
    "switched": GRADE_C,
    "ws_worse": GRADE_C,
    "ad_worse": GRADE_C,
}

# ------------------------------------------------------- diffusion_resolution
# `bm_diff.diffusion_resolution` 的 level → 档位。
LEVEL_CLASS = {
    "no_diffusion": None,          # 未启用，不构成资格限制
    "unknown": GRADE_A,            # 拿不到 tauD 后验（模型/配置缺项）
    "indistinguishable": GRADE_B,  # 两核在带内形状差 < tol
    "resolvable": None,            # 本频段可分，本身不是限制（机制结论仍归 C）
}

# 每档一句「这条推不出什么」
_READS = {
    GRADE_A: "推不出任何数据侧结论：采样/LOO/拟合未通过，先修或重跑",
    GRADE_B: "推不出「某机制不存在」：本窗+本噪声下只能给「不可区分」"
             "或「幅值定不住」，不能反推不存在",
    GRADE_C: "推不出「机制已被证实」：模型比较有结论，但缺真值/独立实验/正交表征",
}


def worst(grades):
    """取最该先处理的一档（A > B > C > None）。"""
    best = None
    for g in grades:
        if _RANK.get(g, 3) > _RANK.get(best, 0):
            best = g
    return best


def explain(grade) -> str:
    return LABEL.get(grade, "无资格限制") if grade else "无资格限制"


def reads_as(grade) -> str:
    return _READS.get(grade, "该结论不落在 A/B/C 三档内")


def _class_of_guard(name):
    """未登记的 guard 按 A 处理：宁可说「算法不确定」，不要说「数据不确定」。"""
    return GUARD_CLASS.get(name) or GRADE_A


def annotate(out, guards=None):
    """给选择器输出加 `guard_classes` / `grade` / `grade_reason` / `not`。

    只加键，不改任何已有键。`guards` 缺省时读 `out["guards"]`。
    """
    if not isinstance(out, dict):
        return out
    gs = dict(guards if guards is not None else (out.get("guards") or {}))
    fired = [k for k, v in gs.items() if v]
    classes = {k: _class_of_guard(k) for k in fired}

    status = out.get("status")
    if fired:
        grade = worst(classes.values())
        src = "guards: " + "、".join(fired)
        # 未登记的 guard 名单独点出来，避免静默归到 A
        unknown = [k for k in fired if k not in GUARD_CLASS]
        if unknown:
            src += f"（其中 {'、'.join(unknown)} 未在 bm_grade.GUARD_CLASS 登记，暂按 A）"
    else:
        grade = STATUS_CLASS.get(status)
        src = f"status={status}"
        # gate 本身缺失（不是「三闸没过」而是「没门控」）→ A 不是 B
        if status == "gate_not_strong":
            g = out.get("gate") or {}
            if not g or g.get("status") in (None, "missing"):
                grade = GRADE_A
                src = "status=gate_not_strong 且 gate 缺失"

    out["guard_classes"] = classes
    out["grade"] = grade
    out["grade_reason"] = (
        f"由 {src} 判 {grade}：{explain(grade)}" if grade
        else f"由 {src}：该结论不涉及 A/B/C 限制")
    out["not"] = reads_as(grade)
    return out


def annotate_level(out):
    """给 `diffusion_resolution` 之类带 `level` 的输出加档位（只加键）。"""
    if not isinstance(out, dict):
        return out
    lvl = out.get("level")
    grade = LEVEL_CLASS.get(lvl, GRADE_A if lvl else None)
    out["grade"] = grade
    out["grade_reason"] = (
        f"由 level={lvl} 判 {grade}：{explain(grade)}" if grade
        else f"由 level={lvl}：该结论不涉及 A/B/C 限制")
    out["not"] = reads_as(grade)
    return out


def component_grade(state, p_peak_out=None):
    """组分可辨识性状态 → 档位。

    `in_window` → None（无限制）
    `out_of_window`（P(峰位出窗)>0.5）→ **C**：参数定得住、预测算得出，
        但本窗没覆盖该特征频率，**无权说该过程存在与否**。
    `uncertain` → 分两种，这是原来混在一起的地方：
        · 后验缺失（nan）→ **A**（拿不到 p，属计算问题）
        · 0.05 < p ≤ 0.5 → **B**（数据只给了「说不清」）
    """
    if state == "in_window":
        return None
    if state == "out_of_window":
        return GRADE_C
    if p_peak_out is None:
        return GRADE_A
    try:
        v = float(p_peak_out)
    except (TypeError, ValueError):
        return GRADE_A
    if v != v:                       # nan
        return GRADE_A
    return GRADE_B


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------
def _selfcheck():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        assert cond, msg
        ok += 1

    chk(worst(["B", "A", "C"]) == "A", "worst 应取 A")
    chk(worst([None, "C"]) == "C", "worst 应忽略 None")
    chk(worst([]) is None, "空列表应为 None")

    # A 类 guard
    o = annotate(dict(status="guard_degraded", guards={"ws_grade0": True}))
    chk(o["grade"] == "A", "ws_grade0 应为 A")
    chk(o["guard_classes"] == {"ws_grade0": "A"}, "guard_classes 只列触发项")
    chk(o["status"] == "guard_degraded", "不得改 status")
    # B 类 guard
    o = annotate(dict(status="guard_degraded",
                      guards={"diff_amp_unidentifiable": True}))
    chk(o["grade"] == "B", "diff_amp_unidentifiable 应为 B")
    # 混合取严
    o = annotate(dict(status="guard_degraded",
                      guards={"diff_amp_unidentifiable": True,
                              "ws_bad_k": True}))
    chk(o["grade"] == "A", "混合 guard 应取 A")
    # 未登记 guard → A，且要写进 reason
    o = annotate(dict(status="guard_degraded", guards={"brand_new": True}))
    chk(o["grade"] == "A" and "未在 bm_grade.GUARD_CLASS 登记" in o["grade_reason"],
        "未登记 guard 应归 A 并注明")
    # status 映射
    chk(annotate(dict(status="uncertain"))["grade"] == "B", "uncertain 应为 B")
    chk(annotate(dict(status="switched"))["grade"] == "C", "switched 应为 C")
    chk(annotate(dict(status="fit_missing"))["grade"] == "A", "fit_missing 应为 A")
    chk(annotate(dict(status="degenerate"))["grade"] == "B", "degenerate 应为 B")
    # gate 缺失 → A
    chk(annotate(dict(status="gate_not_strong", gate=None))["grade"] == "A",
        "gate 缺失应为 A")
    chk(annotate(dict(status="gate_not_strong",
                      gate={"status": "monotonic_rise"}))["grade"] == "B",
        "三闸未过应为 B")
    # level 映射
    chk(annotate_level(dict(level="unknown"))["grade"] == "A", "unknown 应为 A")
    chk(annotate_level(dict(level="indistinguishable"))["grade"] == "B",
        "indistinguishable 应为 B")
    chk(annotate_level(dict(level="resolvable"))["grade"] is None,
        "resolvable 无限制")
    chk(annotate_level(dict(level="no_diffusion"))["grade"] is None,
        "no_diffusion 无限制")
    # 组分
    chk(component_grade("in_window") is None, "in_window 无限制")
    chk(component_grade("out_of_window") == "C", "出窗应为 C")
    chk(component_grade("uncertain", float("nan")) == "A", "nan → A")
    chk(component_grade("uncertain", 0.3) == "B", "0.3 → B")
    chk(component_grade("uncertain", None) == "A", "None → A")
    # 不可序列化风险：只允许 str/None
    for g in (GRADE_A, GRADE_B, GRADE_C, None):
        chk(isinstance(g, (str, type(None))), "grade 只能是 str/None")

    # 表完整性：每个 guard / status 都必须有档位
    for k, v in GUARD_CLASS.items():
        chk(v in (GRADE_A, GRADE_B, GRADE_C), f"{k} 档位非法")
    for k, v in STATUS_CLASS.items():
        chk(v in (GRADE_A, GRADE_B, GRADE_C), f"{k} 档位非法")
    for k, v in LEVEL_CLASS.items():
        chk(v in (GRADE_A, GRADE_B, GRADE_C, None), f"{k} 档位非法")

    print(f"bm_grade 自检 {ok}/{ok} 通过")
    return ok


if __name__ == "__main__":
    _selfcheck()
