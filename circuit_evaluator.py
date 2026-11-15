# -*- coding: utf-8 -*-
"""候选观测 → evidence 状态的**推导层**（registry 的 `evidence_block` 只做校验，不做推导）。

三层架构里这是第三层：

    circuit_registry   能不能被表达 / 签发身份 / 被复核      （已有）
    circuit_runner     候选集 → 预检 → 逐候选拟合 → 观测      （已有）
    circuit_evaluator  **观测数组 → 一个状态 + 一张决策表**   （本文件）

设计原则
--------
1. **纯函数**：只吃字典、只吐字典，不 import `bm_*`、不拟合、不读磁盘以外的状态。
   输入是 `circuit_runner` 落下的 `plan.json` + `observations.jsonl`。
2. **只判"能不能谈"，不判"谁是对的"**：`identified` / `model_inadequate` 在当前
   口径下**不可达**（见 `UNREACHABLE`），本模块不会产出它们，也不允许调用方绕过。
3. **有争议的规则显式标 `disputed`**：`tie_pp`（多近算"分不开"）是需要人裁决的数，
   默认 1.0 个百分点，输出里带 `disputed: true` 与 `pending_decision`。改它等于改
   判定口径，所以不藏在常数堆里。
4. **结论必须能被追溯**：每个状态都带 `rules`（哪几条触发）与 `caveats`（哪些边界
   没覆盖），而不是只给一个词。

状态与规则
----------
| 规则 | 内容 | 性质 |
|---|---|---|
| E1 | `sampling_status=failed` 的候选不参与比较（`bm_select` 同款门槛，R̂>1.05 或 div>0） | firm |
| E2 | 预检 `excluded` 的候选不进比较；`unknown` 只是"没有判据"，**不当通过也不当排除** | firm |
| E3 | viable < 2 ⇒ `insufficient_evidence`（一个候选谈不上"区分"，R12） | firm |
| E4 | 前两名 `rms_rel` 差 ≤ `tie_pp` **且** `Δelpd` 不显著 ⇒ `non_identifiable` | **disputed** |
| E5 | 否则 ⇒ `weak_preference`（"有倾向、但不足以 `identified`"） | firm |
| E6 | `identified` / `model_inadequate` 一律不产出，理由写进 `unreachable_states` | firm |

用法
----
    D:/autoeis_env/Scripts/python.exe circuit_evaluator.py --eval out/circuit_runner/<stem>
    D:/autoeis_env/Scripts/python.exe circuit_evaluator.py --eval "out/circuit_runner/*"
    D:/autoeis_env/Scripts/python.exe circuit_evaluator.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import circuit_registry as CR                                    # noqa: E402

#: 与 `bm_select` 同值（只读引用，不重定义判据语义）。
SIG_DELTA = 4.0
SIG_Z = 2.0
RHAT_MAX = 1.05

#: **有争议**：前两名 `rms_rel` 相差多少个百分点以内算"分不开"。
#: 这个数是判定口径的一部分 —— 改它等于改验收标准，必须登记。
#:
#: 实证锚点（2026-11-16 补，不再是"随手取的"）：同一模型、同一条件重复测，
#: `rms_rel` 本身就会波动 1pp 量级 ——
#:   `3E-3 full_cell 0100mA` 4 次独立实验：2.99/3.46/4.55/5.32% → sd=1.05pp、极差 2.33pp
#:   `3E-2 full_cell 0100mA` 3 次独立实验：2.26/2.31/3.25%    → sd=0.56pp、极差 0.99pp
#: 两候选差小于这个量级时，"谁更低"不可分辨。故 1.0pp ≈ 1 个重复性 sd，
#: 是"刚好能分辨"的**下限**而非上界；保守可放 1.5–2.0pp。
#: **不能放到 10pp**：那会把 9pp 的真实差异也判「分不开」，等于取消 E4。
#:
#: 2026-11-16 用户裁决：**定 1.0pp**（采纳上面的实证依据）。
DEFAULT_TIE_PP = 1.0

#: **有争议**：最优候选的 `rms_rel` 高于此值时，判 `insufficient_evidence`
#: （理由 `no_candidate_within_acceptance`）而不是 `non_identifiable`。
#: 取值直接沿用 §14 的 `RMS_MAX=10.0`，**没有新造阈值**；但它在这里的用法是新的
#: ——`model_inadequate` 被 `truth_known/noise_calibrated` 挡住，本规则是它的**代理**。
#: 传一个很大的值（如 `--rms-accept 1e9`）即可关掉。
#:
#: 2026-11-16 用户裁决：**定 10%**（"10% 就行"）。
DEFAULT_RMS_ACCEPT = 10.0

#: 当前口径下不可达的两个状态，及其原因（照抄 `circuit_registry` §evidence 与 R21）。
UNREACHABLE = {
    "identified": (
        "需要 `out_of_sample_valid=True`：§14 口径下逐点 σ（`sigma_from_data=True`）使 "
        "`elpd_loo` 不是 `p(Z_j|Z_-j)`；跨族口径（`sigma=None`）下先验尺度 "
        "`zmax=max|Z|` 又依赖本批观测（R21）。两条路各自被堵。"
        "要解锁必须先裁 σ/先验尺度口径，**不是放宽这里**。"),
    "model_inadequate": (
        "需要 `truth_known` 或 `noise_calibrated`：真实谱两者都没有 ⇒ 只能落 "
        "`insufficient_evidence`（§18：合成 12/12 过、真实 0.505–0.896，"
        "高 acf 不能读作「候选集不足」）。"),
}


# --------------------------------------------------------------------------
# 输入：读 circuit_runner 的产物
# --------------------------------------------------------------------------
def load_run(run_dir: str) -> dict:
    """读一个 `circuit_runner` 输出目录，返回 `{plan, records}`。

    只读 `plan.json` 与 `observations.jsonl`；缺哪个就报错，不猜。
    """
    pj = os.path.join(run_dir, "plan.json")
    oj = os.path.join(run_dir, "observations.jsonl")
    if not os.path.exists(pj):
        raise FileNotFoundError(f"{run_dir} 缺 plan.json（不是 circuit_runner 的输出目录？）")
    if not os.path.exists(oj):
        raise FileNotFoundError(f"{run_dir} 缺 observations.jsonl")
    with open(pj, "r", encoding="utf-8") as fh:
        plan = json.load(fh)
    records = []
    with open(oj, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{oj} 第 {i} 行不是合法 JSON：{exc}") from exc
    return dict(plan=plan, records=records, run_dir=run_dir)


def _num(v):
    return None if v is None else float(v)


def tabulate(records) -> list:
    """把原始记录摊平成一行一个候选，字段名固定（下游/报告都依赖它）。"""
    rows = []
    for rec in records:
        ob = rec.get("observation") or {}
        fo = ob.get("fit_observation") or {}
        sd = fo.get("sampling_diagnostics") or {}
        pr = ob.get("predictive") or {}
        pre = rec.get("precheck") or {}
        rows.append(dict(
            candidate_id=ob.get("candidate_id") or pre.get("candidate_id"),
            topology_id=rec.get("topology_id"),
            precheck=pre.get("precheck_state"),
            precheck_reason=pre.get("precheck_reason") or "",
            run_status=ob.get("run_status"),
            sampling_status=ob.get("sampling_status"),
            interpretation=fo.get("interpretation"),
            rms_rel=_num(fo.get("rms_rel")),
            acf=_num(fo.get("acf")),
            rhat_max=_num(sd.get("rhat_max")),
            ess_min=_num(sd.get("ess_min")),
            divergences=(None if sd.get("divergences") is None
                         else int(sd["divergences"])),
            elpd=_num(pr.get("elpd")),
            elpd_se=_num(pr.get("se")),
            elpd_valid=pr.get("valid_for_comparison"),
            error=rec.get("error"),
        ))
    return rows


# --------------------------------------------------------------------------
# 判定
# --------------------------------------------------------------------------
def _significant(a: dict, b: dict) -> bool:
    """两候选的 elpd 差是否显著（`bm_select` 同款两条同时满足才算）。"""
    if a.get("elpd") is None or b.get("elpd") is None:
        return False
    d = abs(a["elpd"] - b["elpd"])
    se = max(a.get("elpd_se") or 0.0, b.get("elpd_se") or 0.0)
    if se <= 0:
        return d >= SIG_DELTA
    return (d >= SIG_DELTA) and (d / se >= SIG_Z)


def evaluate(rows, *, tie_pp: float = DEFAULT_TIE_PP,
             rms_accept: float = DEFAULT_RMS_ACCEPT,
             candidate_set_complete: bool = True) -> dict:
    """核心纯函数：候选行 → 评价块。状态一定落在 `EVIDENCE_STATES` 内。"""
    rules, caveats = [], []

    def fire(rid, text, firm=True, **extra):
        rules.append(dict(id=rid, text=text, firm=bool(firm), **extra))

    # E2：预检排除的不进比较；unknown 只记状态
    excluded = [r for r in rows if r["precheck"] == "excluded"]
    unknown = [r for r in rows if r["precheck"] == "unknown"]
    entered = [r for r in rows if r["precheck"] != "excluded"]
    fire("E2", f"预检 excluded 的 {len(excluded)} 个候选不进比较；"
               f"unknown {len(unknown)} 个（没有判据 ≠ 通过，也不当排除）")

    # E1：采样不达标的不进比较
    failed_run = [r for r in entered if r["run_status"] != "completed"]
    sampling_failed = [r for r in entered
                       if r["run_status"] == "completed"
                       and r["sampling_status"] == "failed"]
    viable = [r for r in entered
              if r["run_status"] == "completed"
              and r["sampling_status"] == "passed"
              and r["rms_rel"] is not None]
    fire("E1", f"采样不达标/未跑完的 {len(sampling_failed) + len(failed_run)} 个候选"
               f"不参与比较（R̂>{RHAT_MAX} 或 div>0 即 `failed`）")

    viable = sorted(viable, key=lambda r: r["rms_rel"])
    caveats.append("`sampling_status=passed` 只表示 R̂≤1.05 且 div=0，"
                   "不表示模型对这条谱充分")
    if any(r.get("elpd_valid") is False for r in viable):
        caveats.append("本批 `predictive.valid_for_comparison=False`：Δelpd 只能作"
                       "**相对参考**（R21），不构成样本外证据；`identified` 因此不可达")

    if not candidate_set_complete:
        caveats.append("候选集不完整（registry 里仍有 `wired=False` 的元件/拓扑），"
                       "`non_identifiable` 的作用域仅限**已登记的候选集**（R12）")

    # E3：viable < 2 ⇒ 谈不上区分
    if len(viable) < 2:
        fire("E3", f"viable 候选只有 {len(viable)} 个（<2）⇒ 不能谈「可区分」")
        status = "insufficient_evidence"
        reason = ("no_viable_candidate" if not viable else "single_viable_candidate")
        # 早退分支必须把 excluded / sampling_failed 一起带上，否则报告里的候选表会
        # 只剩 viable —— 那正好把"被排除的候选"从证据里抹掉，是最不该丢的信息。
        return _finish(status, viable, rules, caveats, reason=reason,
                       best=(viable[0] if viable else None), tie=None,
                       excluded=excluded,
                       sampling_failed=sampling_failed + failed_run,
                       tie_pp=tie_pp, rms_accept=rms_accept,
                       candidate_set_complete=candidate_set_complete)

    # E7（有争议）：所有 viable 都达不到 §14 的残差接受线 ⇒ 谈"分不分得开"没意义
    if viable[0]["rms_rel"] > rms_accept:
        fire("E7", f"最优候选 rms={viable[0]['rms_rel']:.2f}% 已高于接受线 "
                   f"{rms_accept:g}%（沿用 §14 `RMS_MAX`）⇒ 没有一个候选够格，"
                   f"此处的「分不开」不能读作「候选集完备」",
             firm=False, disputed=True, rms_accept=float(rms_accept),
             best_rms=float(viable[0]["rms_rel"]),
             pending_decision=("`rms_accept` 已于 2026-11-16 裁决为 10%（沿用 §14 `RMS_MAX`）；"
                               "但它只是 `model_inadequate` 的**代理**——"
                               "后者要 `truth_known/noise_calibrated`，不由这个阈值解锁"))
        return _finish("insufficient_evidence", viable, rules, caveats,
                       reason="no_candidate_within_acceptance",
                       best=viable[0], tie=None, excluded=excluded,
                       sampling_failed=sampling_failed + failed_run,
                       tie_pp=tie_pp, rms_accept=rms_accept,
                       candidate_set_complete=candidate_set_complete)

    best, second = viable[0], viable[1]
    gap_pp = float(second["rms_rel"] - best["rms_rel"])
    sig = _significant(best, second)
    d_elpd = (None if best["elpd"] is None or second["elpd"] is None
              else float(best["elpd"] - second["elpd"]))
    se = max(best.get("elpd_se") or 0.0, second.get("elpd_se") or 0.0)
    z = (None if d_elpd is None else (d_elpd / se if se > 0 else None))

    # E4（有争议）：残差地板相同且 Δelpd 不显著 ⇒ 分不开
    tied = (gap_pp <= tie_pp) and (not sig)
    fire("E4", f"前两名 rms 差 {gap_pp:.2f}pp（并列阈值 {tie_pp:g}pp）、"
               f"Δelpd {'显著' if sig else '不显著'} "
               f"⇒ {'判为并列（non_identifiable）' if tied else '不构成并列'}",
         firm=False, disputed=True,
         tie_pp=float(tie_pp), gap_pp=gap_pp, delta_elpd=d_elpd, delta_z=z,
         pending_decision=("`tie_pp` 已于 2026-11-16 裁决为 1.0pp，依据是同条件重复的 "
                           "`rms` sd≈1.05pp（3E-3 四次）／0.56pp（3E-2 三次）；"
                           "它不是随手取的数，但仍是「重复性尺度」而非「统计显著性」"))

    # 两个判据可能互相矛盾（残差说"分不开"、elpd 说"分得开"，或反之）。
    # 这不是 bug，是判据体系的现状；必须显式记账，否则会被读成"结论唯一"。
    conflict = (gap_pp <= tie_pp) != (not sig)
    if conflict:
        caveats.append(
            f"**判据冲突**：残差口径说{'并列' if gap_pp <= tie_pp else '可分'}"
            f"（差 {gap_pp:.2f}pp vs 阈值 {tie_pp:g}pp），"
            f"elpd 口径说{'可分' if sig else '不可分'}"
            f"（Δ={d_elpd if d_elpd is not None else float('nan'):.2f}、"
            f"Δ/se={z if z is not None else float('nan'):.2f}）"
            "；本状态按 elpd 口径给出，冲突本身作为证据保留（无仲裁规则）")

    if tied:
        status = "non_identifiable"
        reason = "top2_within_tie_and_delta_insignificant"
    else:
        fire("E5", f"{best['candidate_id']} 在 rms 上领先 {gap_pp:.2f}pp、"
                   f"Δelpd {'显著' if sig else '不显著'} ⇒ weak_preference")
        status = "weak_preference"
        reason = "best_leading_but_not_identified"

    # E6：两个状态不产出
    fire("E6", "`identified` / `model_inadequate` 当前口径下不可达，不产出",
         unreachable=list(UNREACHABLE))
    return _finish(status, viable, rules, caveats, reason=reason, best=best,
                   tie=dict(gap_pp=gap_pp, tie_pp=float(tie_pp),
                            significant=bool(sig), delta_elpd=d_elpd, delta_z=z,
                            criteria_conflict=bool(conflict)),
                   second=second, excluded=excluded,
                   sampling_failed=sampling_failed + failed_run,
                   tie_pp=tie_pp, rms_accept=rms_accept,
                   candidate_set_complete=candidate_set_complete)


def build_confirmation(*, status, best, tie, viable, excluded, sampling_failed,
                       rms_accept, tie_pp, candidate_set_complete) -> dict:
    """把"已确认到哪一步"与"哪些不充分/为什么"分开写。

    **纪律**：`confirmed` 里只放**描述性事实**（谁跑了、谁被排除、残差多少、
    两条口径怎么说），**不得**出现"模型正确""最优模型是 X""identified"这类断言。
    状态标签由 `evaluate()` 给，这里只是把标签背后的证据摊开，防止读者把
    `weak_preference` 读成"已经选出来了"，也防止把空白的 `identified` 读成"没做"。
    """
    n_all = len(viable) + len(excluded) + len(sampling_failed)
    confirmed, not_confirmed = [], []

    def cf(item, text):
        confirmed.append(dict(item=item, text=text))

    def nc(item, why, needs):
        not_confirmed.append(dict(item=item, why=why, needs=needs))

    cf("candidate_set", f"候选集 {n_all} 个（全部来自 registry 登记，"
                        f"不是临时拼的）")
    if excluded:
        cf("precheck_excluded",
           f"{len(excluded)} 个被预检判 `excluded`：{'、'.join(excluded)}"
           "（有明确判据说它表达不了，不参与比较）")
    cf("sampling",
       f"{len(viable)}/{n_all} 个候选通过采样门槛（R̂≤{RHAT_MAX} 且 0 发散）")
    if best is not None:
        line = (f"最优候选 `{best['candidate_id']}` 的 rms_rel = "
                f"{best['rms_rel']:.2f}%")
        line += ("，在 §14 接受线内" if best["rms_rel"] <= rms_accept
                 else f"，**已超接受线 {rms_accept:g}%**")
        cf("residual_floor", line)
    if tie is not None:
        tie_txt = ("残差口径：前两名差 "
                   f"{tie['gap_pp']:.2f}pp vs 重复性尺度 {tie_pp:g}pp ⇒ "
                   + ("不可分" if tie["gap_pp"] <= tie_pp else "可分"))
        el_txt = ("elpd 口径：" + ("显著" if tie["significant"] else "不显著")
                  + (f"（Δ={tie['delta_elpd']:.2f}、Δ/se={tie['delta_z']:.2f}）"
                     if tie.get("delta_elpd") is not None
                     and tie.get("delta_z") is not None else ""))
        cf("two_criteria", tie_txt + "；" + el_txt
           + ("；**两者冲突，无仲裁规则**" if tie.get("criteria_conflict") else ""))

    if sampling_failed:
        nc("sampling_completeness",
           f"{len(sampling_failed)} 个候选采样不达标（{'、'.join(sampling_failed)}），"
           "它们的 rms 没有进入比较",
           "提高采样预算或换初始化后重跑；在达标前不得引用其 rms")
    nc("identified", UNREACHABLE["identified"],
       "裁决 σ/先验尺度口径（会让 §13/§14/§18/§19 的 elpd 数字全部重算）")
    nc("model_inadequate", UNREACHABLE["model_inadequate"],
       "`noise_calibrated` 可由 CAL-01 开-短路+已知 RC 校准提供；"
       "`truth_known` 只能来自合成数据")
    nc("out_of_sample_evidence",
       "本批 `predictive.valid_for_comparison=False`：Δelpd 只是候选间相对参考（R21）",
       "同上；在解开之前，「选得更好」不等于「样本外预测更好」")
    if not candidate_set_complete:
        nc("candidate_set_completeness",
           "registry 里仍有 `wired=False` 的元件（`w_inf`/`w_open`/`w_short`/`adsorption`）"
           "且没有拓扑提供槽位 ⇒ 本次结论的作用域仅限已登记候选集（R12）",
           "给拓扑加扩散/吸附槽位，把 `wired` 翻成 True")

    return dict(
        confirmed=confirmed,
        not_confirmed=not_confirmed,
        scope=(f"状态 `{status}` 的作用域 = 已登记候选集 {n_all} 个；"
               "它不陈述「真实电路是什么」，只陈述「在这批候选与这个口径下能说到哪一步」"),
    )


def _finish(status, viable, rules, caveats, *, reason, best, tie,
            second=None, excluded=(), sampling_failed=(),
            tie_pp=DEFAULT_TIE_PP, rms_accept=DEFAULT_RMS_ACCEPT,
            candidate_set_complete=True) -> dict:
    """组装并通过 `evidence_block` 的 schema 校验（校验失败当场 raise，不降级）。"""
    block = CR.evidence_block(
        status=status,
        n_viable=len(viable),
        truth_known=False,          # 真实谱没有真值，不给这一列留口子
        noise_calibrated=False,     # 没有独立噪声测量（CAL-01 尚未执行）
        out_of_sample_valid=False,  # R21：先验尺度依赖观测
        note=reason,
    )
    # 传进来的是 row dict，`build_confirmation` 只需要 id 与计数
    ex_ids = [r["candidate_id"] for r in excluded]
    sf_ids = [r["candidate_id"] for r in sampling_failed]
    confirm = build_confirmation(
        status=status, best=best, tie=tie, viable=viable, excluded=ex_ids,
        sampling_failed=sf_ids, rms_accept=rms_accept, tie_pp=tie_pp,
        candidate_set_complete=candidate_set_complete)
    return dict(
        verdict=block,
        status=status,
        reason=reason,
        confirmation=confirm,
        best=(None if best is None else best["candidate_id"]),
        second=(None if second is None else second["candidate_id"]),
        tie=tie,
        n_viable=len(viable),
        viable=[r["candidate_id"] for r in viable],
        excluded=ex_ids,
        sampling_failed=sf_ids,
        rules=rules,
        caveats=caveats,
        unreachable_states=UNREACHABLE,
        candidates=viable + [r for r in excluded if r not in viable]
        + [r for r in sampling_failed if r not in viable and r not in excluded],
    )


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def render(ev: dict, *, title: str = "") -> str:
    """人读版：状态 + 决策表 + 候选表 + 未覆盖边界。不新增任何数字。"""
    L = []
    if title:
        L += [f"## {title}", ""]
    L += [f"**状态**：`{ev['status']}`（{ev['reason']}）", ""]
    L += [f"- viable 候选：{ev['n_viable']} 个"
          + (f"，最优 `{ev['best']}`" if ev["best"] else ""), ""]
    conf = ev.get("confirmation") or {}
    if conf:
        L += ["**已确认（描述性事实，不含「模型正确」）**", ""]
        L += [f"- {x['text']}" for x in conf.get("confirmed", [])]
        L += ["", "**不充分 / 不可达（以及缺什么）**", ""]
        L += [f"- `{x['item']}`：{x['why']} ｜ 需要：{x['needs']}"
              for x in conf.get("not_confirmed", [])]
        if conf.get("scope"):
            L += ["", f"**作用域**：{conf['scope']}"]
        L += [""]
    if ev["tie"]:
        t = ev["tie"]
        de = ("—" if t["delta_elpd"] is None else f"{t['delta_elpd']:.2f}")
        L += [f"- 前两名：rms 差 **{t['gap_pp']:.2f}pp**（并列阈值 {t['tie_pp']:g}pp）、"
              f"`Δelpd={de}`"
              + (f"、`Δ/se={t['delta_z']:.2f}`" if t.get("delta_z") is not None else "")
              + f"，显著={t['significant']}"
              + ("，**两口径冲突**" if t.get("criteria_conflict") else "")]
    L += ["", "| 候选 | 预检 | 运行 | 采样 | rms% | acf | R̂ | div | elpd |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in ev["candidates"]:
        L.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["candidate_id"], r["precheck"], r["run_status"], r["sampling_status"],
            "—" if r["rms_rel"] is None else f"{r['rms_rel']:.2f}",
            "—" if r["acf"] is None else f"{r['acf']:.3f}",
            "—" if r["rhat_max"] is None else f"{r['rhat_max']:.3f}",
            "—" if r["divergences"] is None else r["divergences"],
            "—" if r["elpd"] is None else f"{r['elpd']:.2f}"))
    L += ["", "**触发的规则**", ""]
    for r in ev["rules"]:
        tag = "firm" if r["firm"] else "**disputed**"
        L.append(f"- `{r['id']}`（{tag}）{r['text']}")
    if ev["caveats"]:
        L += ["", "**边界**", ""] + [f"- {c}" for c in ev["caveats"]]
    L += ["", "**不可达的状态**", ""] + [
        f"- `{k}`：{v}" for k, v in ev["unreachable_states"].items()]
    return "\n".join(L)


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------
def _obs(cid, *, pre="not_excluded", run="completed", sampling="passed",
         rms=5.0, acf=0.5, rhat=1.01, div=0, elpd=-100.0, se=5.0):
    return CR.observation_block(
        candidate_id=cid, run_status=run, sampling_status=sampling,
        rms_rel=(None if run != "completed" else rms),
        acf=(None if run != "completed" else acf),
        rhat_max=(None if run != "completed" else rhat),
        divergences=(None if run != "completed" else div),
        predictive={"elpd": elpd, "se": se, "valid_for_comparison": False,
                    "reason": "R21"},
        interpretation=(None if sampling == "passed" else "descriptive_only"))


def _row(cid, **kw):
    pre = kw.pop("pre", "not_excluded")
    ob = _obs(cid, pre=pre, **kw)
    return tabulate([dict(observation=ob, precheck=dict(precheck_state=pre))])[0]


def _selftest() -> int:
    ok, total, bad = 0, 0, []

    def c(name, cond):
        nonlocal ok, total
        total += 1
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")
        if cond:
            ok += 1
        else:
            bad.append(name)

    print("E4：残差地板相同 ⇒ non_identifiable（disputed 规则）")
    # rms 必须落在 E7 的接受线内，否则先被 E7 拦下（12.55% 那组专门留给 E7 用例）
    rows = [_row("C-a", rms=6.12, elpd=-388.0), _row("C-b", rms=6.45, elpd=-392.0)]
    ev = evaluate(rows)
    c("状态 non_identifiable", ev["status"] == "non_identifiable")
    c("E4 标 disputed", any(r["id"] == "E4" and r["disputed"] for r in ev["rules"]))
    c("E4 带 pending_decision", any("tie_pp" in (r.get("pending_decision") or "")
                                    for r in ev["rules"]))
    c("tie 记录 gap 与 Δelpd（best 的 elpd 更高 ⇒ Δ≥0）",
      ev["tie"]["gap_pp"] > 0 and ev["tie"]["delta_elpd"] == 4.0)
    c("判据冲突被显式标记（rms 并列、elpd 不显著 ⇒ 不冲突）",
      ev["tie"]["criteria_conflict"] is False)

    print("\nE5：残差差得远且 Δelpd 显著 ⇒ weak_preference")
    rows = [_row("C-a", rms=8.0, elpd=-300.0, se=5.0),
            _row("C-b", rms=16.0, elpd=-400.0, se=5.0)]
    ev2 = evaluate(rows)
    c("状态 weak_preference", ev2["status"] == "weak_preference")
    c("best 是 rms 最小的那个", ev2["best"] == "C-a")
    c("显著标记为 True", ev2["tie"]["significant"] is True)

    print("\nE1：采样不达标不进比较")
    rows = [_row("C-a", rms=8.0, elpd=-300.0),
            _row("C-b", rms=9.0, elpd=-320.0, sampling="failed", rhat=1.5)]
    ev3 = evaluate(rows)
    c("viable 只剩 1 个", ev3["n_viable"] == 1)
    c("降 insufficient_evidence", ev3["status"] == "insufficient_evidence")
    c("reason=single_viable_candidate", ev3["reason"] == "single_viable_candidate")
    c("failed 的候选仍列在表里", "C-b" in [r["candidate_id"] for r in ev3["candidates"]])

    print("\nE2：预检 excluded 不进比较")
    rows = [_row("C-a", rms=8.0), _row("C-x", pre="excluded", rms=5.0)]
    ev4 = evaluate(rows)
    c("excluded 没进 viable", ev4["n_viable"] == 1)
    c("excluded 单独列出", ev4["excluded"] == ["C-x"])
    c("状态 insufficient_evidence", ev4["status"] == "insufficient_evidence")

    print("\nE3：zero viable")
    ev5 = evaluate([_row("C-a", run="failed"), _row("C-b", pre="excluded")])
    c("理由 no_viable_candidate", ev5["reason"] == "no_viable_candidate")
    c("best 为 None", ev5["best"] is None)

    print("\nE6：两个状态永不产出")
    rows = [_row("C-a", rms=1.0, elpd=-10.0), _row("C-b", rms=90.0, elpd=-900.0)]
    ev6 = evaluate(rows)
    c("即使拟合极好也不会 identified", ev6["status"] != "identified")
    c("也不会 model_inadequate", ev6["status"] != "model_inadequate")
    c("两个不可达状态都给了理由",
      set(ev6["unreachable_states"]) == {"identified", "model_inadequate"}
      and all(ev6["unreachable_states"].values()))

    print("\n状态一定落在 registry 允许的集合里")
    for evx in (ev, ev2, ev3, ev4, ev5, ev6):
        c(f"状态 {evx['status']} 合法", evx["status"] in CR.EVIDENCE_STATES)
        c(f"evidence_block 已通过校验（{evx['status']}）",
          evx["verdict"]["status"] == evx["status"]
          and evx["verdict"]["scope"] == CR.EVIDENCE_SCOPE)

    print("\ntie_pp 是显式参数，不是硬编码")
    rows = [_row("C-a", rms=10.0), _row("C-b", rms=11.5)]
    c("tie_pp=1.0 → 差 1.5pp 判 weak_preference",
      evaluate(rows, tie_pp=1.0)["status"] == "weak_preference")
    c("tie_pp=2.0 → 同一个输入判 non_identifiable",
      evaluate(rows, tie_pp=2.0)["status"] == "non_identifiable")

    print("\nE7（disputed）：所有候选都超接受线 ⇒ insufficient_evidence")
    rows = [_row("C-a", rms=19.3, elpd=-1251.0), _row("C-b", rms=20.2, elpd=-1252.0)]
    ev7 = evaluate(rows)
    c("状态 insufficient_evidence", ev7["status"] == "insufficient_evidence")
    c("理由 no_candidate_within_acceptance",
      ev7["reason"] == "no_candidate_within_acceptance")
    c("E7 标 disputed", any(r["id"] == "E7" and r["disputed"] for r in ev7["rules"]))
    c("关掉后落回 non_identifiable（同一输入）",
      evaluate(rows, rms_accept=1e9)["status"] == "non_identifiable")
    c("§14 接受线内的谱不受影响（默认 1pp 并列）",
      evaluate([_row("C-a", rms=6.12), _row("C-b", rms=6.45)])["status"]
      == "non_identifiable")

    print("\n确认 / 不充分两段（confirmed vs not_confirmed）")
    cf = ev2["confirmation"]
    c("confirmation 恰好三键", set(cf) == {"confirmed", "not_confirmed", "scope"})
    c("confirmed 非空且每项都有 item/text",
      bool(cf["confirmed"])
      and all(x.get("item") and x.get("text") for x in cf["confirmed"]))
    c("not_confirmed 必含 identified / model_inadequate / out_of_sample_evidence",
      {"identified", "model_inadequate", "out_of_sample_evidence"}
      <= {x["item"] for x in cf["not_confirmed"]})
    c("每条 not_confirmed 都写了「需要什么」（不许只说不可达）",
      all((x.get("needs") or "").strip() for x in cf["not_confirmed"]))
    c("纪律：confirmed 文本里不出现 identified 字样",
      not any("identified" in x["text"] for x in cf["confirmed"]))
    c("E3 早退（viable<2）也带 confirmation",
      bool(ev3.get("confirmation", {}).get("not_confirmed")))
    c("采样不达标时列出 sampling_completeness",
      "sampling_completeness"
      in {x["item"] for x in ev3["confirmation"]["not_confirmed"]})
    c("E7 早退的 confirmed 里写明「已超接受线」",
      any("已超接受线" in x["text"] for x in ev7["confirmation"]["confirmed"]))
    c("scope 里带状态名", ev2["status"] in cf["scope"])

    print("\n报告可渲染且不改数")
    md = render(ev, title="自检")
    c("含状态行", "non_identifiable" in md)
    c("含候选表", "| 候选 |" in md and "C-a" in md)
    c("含 disputed 标记", "disputed" in md)
    c("含不可达状态段", "不可达的状态" in md)

    print(f"\n自检 {ok}/{total} 通过" if not bad else f"\nFAIL {len(bad)}/{total}: {bad}")
    return 0 if not bad else 1


# --------------------------------------------------------------------------
def expand_runs(pattern: str) -> list:
    """把 `--eval` 入参展开成 runner 目录列表。

    **不能只靠 glob**：`测试数据/` 的谱文件名里带 `[P1,P2-R3]`，glob 会把它当字符类，
    于是整目录静默漏掉。所以 `.../` 或 `.../*` 这类"目录下的全部子目录"一律走
    `os.listdir`，只有真带通配符且不是 `*` 结尾的才交给 glob。
    """
    import glob as _glob

    def is_run(d):
        return os.path.isdir(d) and os.path.exists(
            os.path.join(d, "observations.jsonl"))

    if is_run(pattern):
        return [pattern]
    base = pattern.rstrip("\\/")
    if base.endswith("/*") or base.endswith("\\*"):
        parent = base[:-2] or "."
        if os.path.isdir(parent):
            return sorted(os.path.join(parent, n) for n in os.listdir(parent)
                          if is_run(os.path.join(parent, n)))
    return sorted(d for d in _glob.glob(pattern) if is_run(d))


def evaluate_run(run_dir: str, *, tie_pp: float = DEFAULT_TIE_PP,
                 rms_accept: float = DEFAULT_RMS_ACCEPT) -> dict:
    """读目录 → 评价 → 落 `evaluation.json` / `evaluation.md`。"""
    d = load_run(run_dir)
    rows = tabulate(d["records"])
    complete = not [e for e in CR.list_elements()
                    if not CR.get_element(e).wired]
    ev = evaluate(rows, tie_pp=tie_pp, rms_accept=rms_accept,
                  candidate_set_complete=complete)
    ev["inputs"] = dict(run_dir=run_dir,
                        data_path=(d["plan"] or {}).get("data_path"),
                        data_sha256=(d["plan"] or {}).get("data_sha256"),
                        budget_profile=(d["plan"] or {}).get("budget_profile"),
                        n_records=len(d["records"]),
                        registry_version=CR.REGISTRY_VERSION,
                        registry_digest=CR.registry_digest(),
                        tie_pp=float(tie_pp), rms_accept=float(rms_accept))
    stem = os.path.basename(os.path.normpath(run_dir))
    with open(os.path.join(run_dir, "evaluation.json"), "w", encoding="utf-8") as fh:
        json.dump(ev, fh, ensure_ascii=False, indent=1, allow_nan=False, default=str)
    with open(os.path.join(run_dir, "evaluation.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(render(ev, title=stem) + "\n")
    return ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="观测 → evidence 状态（推导层）")
    ap.add_argument("--eval", default=None, help="circuit_runner 输出目录或 glob")
    ap.add_argument("--tie-pp", type=float, default=DEFAULT_TIE_PP,
                    help=f"并列阈值（百分点，默认 {DEFAULT_TIE_PP:g}；有争议，需裁决）")
    ap.add_argument("--rms-accept", type=float, default=DEFAULT_RMS_ACCEPT,
                    help=f"残差接受线（默认 {DEFAULT_RMS_ACCEPT:g}%%=§14 RMS_MAX；"
                         "有争议，传 1e9 可关闭）")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return _selftest()
    if not a.eval:
        ap.print_help()
        return 2

    runs = expand_runs(a.eval)
    if not runs:
        print(f"没有找到可评价的 runner 目录：{a.eval}", file=sys.stderr)
        return 2
    for r in runs:
        ev = evaluate_run(r, tie_pp=a.tie_pp, rms_accept=a.rms_accept)
        if not a.quiet:
            print(render(ev, title=os.path.basename(os.path.normpath(r))))
            print()
    print(f"已评价 {len(runs)} 个目录；tie_pp={a.tie_pp:g}pp、"
          f"rms_accept={a.rms_accept:g}%（两者已于 2026-11-16 裁决）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
