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
| E7 | rms 最低候选 > `rms_accept`（10%）⇒ `insufficient_evidence`（绝对验收线） | **disputed** |
| E8 | 只在 rms≤`rms_accept` 的 qualified 集内比较；未过线者单列、不删、不参与排序 | firm |
| E4 | qualified 前两名 `rms_rel` 绝对差 ≤ `tie_pp` **且** 相对差 ≤ `tie_rel` ⇒ `non_identifiable`；配对 Δelpd 只作证据强弱、不否决 rms 并列 | **disputed** |
| E5 | 否则 ⇒ `weak_preference`（"有倾向、但不足以 `identified`"） | firm |
| E9 | `non_identifiable` 且三条件（rms 并列、配对 elpd 不显著、分段连续比较无实质更差段）均满足时，显示顺序按 `n_elements` 升序；不改状态/不改 best 语义 | firm |
| E10 | 任一频段比 best 实质更差（差 > `tie_pp`，连续口径）或缺 `bands` 的候选不进元件数重排组；仍在候选表、不删 | firm |
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

#: **有争议**：前两名 `rms_rel` 的**相对差**多少以内算"分不开"。
#: 2026-11-20 加：绝对 pp 无法区分"低 rms 谱差 1pp"与"高 rms 谱差 1pp"。
#: 实证判例（用户给定）：
#:   · 1.2/1.3/9.9 vs 1.8/1.0/9.5 ⇒ rms 5.97 vs 5.81，abs 0.16pp、rel 2.8% ⇒ 不可分
#:   · 1/1/9 vs 3/3/9             ⇒ rms 1.0 vs 3.0，abs 2.0pp、rel 200% ⇒ 可分
#: 10%/15%/20%/25% 对上述判据分类一致；取 10%（最保守、更容易判可分的一端）。
#: 判并列须 **rel≤10% 且 abs≤tie_pp**：低 rms 时 rel 会虚高（1.0→1.1 即 10%，
#: 但 abs 0.1pp 无意义），由绝对门槛兜住。
DEFAULT_TIE_REL = 10.0

#: **有争议**：rms 最低候选的 `rms_rel` 高于此值时，判 `insufficient_evidence`
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
        cand = rec.get("candidate") or {}
        # 元件个数：顶层 `candidate.elements` 已是登记事实，不是从参数个数猜。
        # 用它做「不可区分时元件少者优先」的显示顺序，不参与任何状态判定。
        els = cand.get("elements")
        n_el = len(els) if isinstance(els, (list, tuple)) else None
        rows.append(dict(
            candidate_id=ob.get("candidate_id") or pre.get("candidate_id"),
            topology_id=rec.get("topology_id"),
            n_elements=n_el,
            # 分段 rms%（低/中/高三段，来自 spectrum_metrics）。用于 E10
            # 「分段过线模式一致性」门槛；没有就标不可用，绝不伪造。
            bands=([_num(b) for b in fo["bands"]]
                   if isinstance(fo.get("bands"), (list, tuple)) else None),
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
            # 逐点 LOO：配对 dse 的输入，在记录**顶层**（registry 的 predictive
            # 是四键白名单，放里面会被 _check_predictive 丢掉）。fit json 是
            # 另一条同源路径，也没有就 None ⇒ 边缘 se 降级 + caveat。
            pointwise_elpd=([_num(v) for v in rec["pointwise_elpd"]]
                            if isinstance(rec.get("pointwise_elpd"), (list, tuple))
                            else None),
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


def _paired_delta(a: dict, b: dict) -> dict:
    """两条候选的配对 elpd 差与 dse（`bm_select.paired_diff` 同口径）。

    Δ = Σ_i (a_i − b_i)，dse = sqrt(n·Var(a_i − b_i))，只在两边都有
    `pointwise_elpd` 且长度/频率对齐（由调用方保证同源同网格）时算。

    返回 `dict(delta, dse, z, method)`；配对不可用则 `delta/dse=None`、
    `method="marginal_se"`（调用方须记 caveat，不得把边缘 se 冒充配对 dse）。
    """
    out = dict(delta=None, dse=None, z=None, method="unavailable")
    if a.get("elpd") is None or b.get("elpd") is None:
        return out
    pa, pb = a.get("pointwise_elpd"), b.get("pointwise_elpd")
    if pa and pb and len(pa) == len(pb) and len(pa) > 1:
        d = [float(x) - float(y) for x, y in zip(pa, pb)]
        n = len(d)
        delta = float(sum(d))
        mean = delta / n
        var = sum((x - mean) ** 2 for x in d) / (n - 1)
        dse = float((n * var) ** 0.5)
        out.update(delta=delta, dse=dse,
                   z=(delta / dse if dse > 0 else None), method="paired")
        return out
    # 降级：只有边缘 se，按 max(se) 作参考（**不是**配对 dse）。
    se = max(a.get("elpd_se") or 0.0, b.get("elpd_se") or 0.0)
    delta = float(a["elpd"] - b["elpd"])
    out.update(delta=delta, dse=(se if se > 0 else None),
               z=(delta / se if se > 0 else None), method="marginal_se")
    return out


def evaluate(rows, *, tie_pp: float = DEFAULT_TIE_PP,
             tie_rel: float = DEFAULT_TIE_REL,
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

    # E7/E8：绝对验收线先于任何比较。rms<=rms_accept 才 qualified。
    # 用户裁决的三段式：① 先过线 ② 过线内才比谁更好 ③ 不可区分时元件少者列前。
    # 未过线者**不删**：单列进 not_qualified，随候选表一起输出（给事实，不丢证据）。
    qualified = [r for r in viable if r["rms_rel"] <= rms_accept]
    not_qualified = [r for r in viable if r["rms_rel"] > rms_accept]
    if not_qualified:
        ids = "、".join(r["candidate_id"] for r in not_qualified)
        fire("E8", f"{len(not_qualified)} 个候选 rms 超绝对验收线 {rms_accept:g}%"
                   f"（{ids}）：单列不参与比较，不删（仅超线，不等于机制错）")
    if not qualified:
        fire("E7", f"rms 最低的候选 rms={viable[0]['rms_rel']:.2f}% 已高于接受线 "
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
                       not_qualified=not_qualified,
                       tie_pp=tie_pp, rms_accept=rms_accept,
                       candidate_set_complete=candidate_set_complete)

    # 比较只在 qualified 集内进行；viable 仍保留全量给候选表/计数。
    pool = qualified
    best, second = pool[0], (pool[1] if len(pool) > 1 else None)
    if second is None:
        fire("E3", f"qualified 候选只有 1 个（viable {len(viable)} 个）⇒ 不能谈「可区分」")
        return _finish("insufficient_evidence", viable, rules, caveats,
                       reason="single_qualified_candidate",
                       best=best, tie=None, excluded=excluded,
                       sampling_failed=sampling_failed + failed_run,
                       not_qualified=not_qualified,
                       tie_pp=tie_pp, rms_accept=rms_accept,
                       candidate_set_complete=candidate_set_complete)
    gap_pp = float(second["rms_rel"] - best["rms_rel"])
    # E4（2026-11-20 改）：rms 相对+绝对双门槛决定"并列"，配对 elpd 降级为证据强弱。
    # 旧逻辑 tied=(rms并列 and not elpd显著)，elpd 一显著就否决 rms 并列 ——
    # 3e 里 5.81/5.97（三段同档、abs 0.16pp、rel 2.8%）被配对 z=6 判成可区分，
    # 与用户判例冲突。现在并列由 rms 口径单独决定；elpd 只写进规则/caveat。
    _lo = min(float(best["rms_rel"]), float(second["rms_rel"]))
    gap_rel = (gap_pp / _lo * 100.0) if _lo > 0 else float("inf")
    rms_tied = (gap_pp <= tie_pp) and (gap_rel <= tie_rel)
    # E4 的 elpd 轴：优先配对 dse（同一批数据、同一 σ 口径下逐点差）。
    # 边缘 se 是单模型逐点 elpd 的离散度，不是两条**差**的离散度 —— 用它
    # 会把"边缘 se 大、配对差小"的两条误判为不可分（漏判）。配对不可用
    # 才降级为 max(边缘 se)，并记 caveat，绝不混称。
    pair = _paired_delta(best, second)
    d_elpd = pair["delta"]
    dse = pair["dse"]
    z = pair["z"]
    if pair["method"] == "marginal_se":
        caveats.append("逐点 elpd 缺失（或长度不齐）⇒ Δelpd 用 max(边缘 se) 作"
                       "**降级参考**，不是配对 dse；此时 E4 的显著性判定偏保守")
        sig = _significant(best, second)
    else:
        # dse==0 且 Δ≠0（逐点差恒定）是**最强**的显著，不能因为 z 无定义
        # 就漏判；dse==0 且 Δ==0 才是不显著。
        sig = (d_elpd is not None and dse is not None
               and abs(d_elpd) >= SIG_DELTA
               and ((dse == 0) or (dse > 0 and z is not None and abs(z) >= SIG_Z)))
    any_invalid = any(r.get("elpd_valid") is False for r in (best, second))
    if any_invalid:
        caveats.append("这两条的 `valid_for_comparison=False`（R21/σ 泄漏）⇒ "
                       "配对 Δelpd 只能作**候选间相对参考**，不是样本外证据")

    # 并列由 rms 双门槛决定；elpd 只作证据强弱，不再否决 rms 并列。
    tied = rms_tied
    fire("E4", f"前两名 rms 差 {gap_pp:.2f}pp（绝对阈值 {tie_pp:g}pp）、"
               f"相对差 {gap_rel:.1f}%（阈值 {tie_rel:g}%）"
               f" ⇒ {'rms 口径判并列' if rms_tied else 'rms 口径判可分'}；"
               f"配对 Δelpd {'显著' if sig else '不显著'}"
               f"（{pair['method']}：Δ={d_elpd if d_elpd is not None else float('nan'):.2f}、"
               f"dse={dse if dse is not None else float('nan'):.2f}）"
               f" ⇒ 最终{'并列（non_identifiable）' if tied else '不构成并列'}"
               "（elpd 只作证据强弱，不单独改状态）",
         firm=False, disputed=True,
         tie_pp=float(tie_pp), tie_rel=float(tie_rel),
         gap_pp=gap_pp, gap_rel=gap_rel, rms_tied=rms_tied,
         delta_elpd=d_elpd, delta_dse=dse, delta_z=z, elpd_method=pair["method"],
         pending_decision=("`tie_pp=1.0pp` 依重复性 sd；`tie_rel=10%` 依用户判例"
                           "（1.2/1.3/9.9 vs 1.8/1.0/9.5 不可分、1/1/9 vs 3/3/9 可分）。"
                           "配对 elpd 显著性不再否决 rms 并列——它只写进证据强弱。"))

    # 两个判据可能互相矛盾（rms 说"分不开"、elpd 说"分得开"，或反之）。
    # 这不是 bug，是判据体系的现状；必须显式记账，否则会被读成"结论唯一"。
    conflict = rms_tied != (not sig)
    if conflict:
        caveats.append(
            f"**判据冲突**：rms 口径说{'并列' if rms_tied else '可分'}"
            f"（abs {gap_pp:.2f}pp / rel {gap_rel:.1f}%），"
            f"elpd 口径说{'可分' if sig else '不可分'}"
            f"（Δ={d_elpd if d_elpd is not None else float('nan'):.2f}、"
            f"Δ/dse={z if z is not None else float('nan'):.2f}）"
            "；本状态按 rms 口径给出，elpd 分歧作为证据保留（不仲裁、不宣称结论）")

    if tied:
        status = "non_identifiable"
        reason = "top2_within_tie_and_delta_insignificant"
        # E9：不可区分时按元件数升序作**显示顺序**。不改 best/second 的语义
        # （它们仍指向 rms 最低/次低这个事实），只让元件少的候选列前。
        #
        # 三条件门槛（2026-11-20，防"把真谱排掉、留元件少但不那么好的"）：
        #   ① rms 差 ≤ tie_pp         —— 已在 tied 里
        #   ② elpd 不显著             —— 已在 tied 里（sig=False）
        #   ③ 分段连续比较：任一频段比 best 差超过容差 ⇒ 不进重排组（E10）
        # 不用二值"过线/超线"：9.9(过) 与 10.1(超) 只差 0.2pp 却被分两类，
        # 而 1.2(过) 与 3.4(过) 差 2.2pp 却同类 —— 二值化正好丢了连续信息。
        # 容差用 tie_pp（1pp，同条件重复的 rms sd）：段间差 ≤1pp 视为同档，
        # >1pp 视为实质更差。不引入新阈值。
        tied_ids = [r["candidate_id"] for r in pool
                    if (r["rms_rel"] - best["rms_rel"]) <= tie_pp]
        reorderable = all(r.get("n_elements") is not None for r in pool)
        if not reorderable:
            caveats.append("候选行缺 `n_elements`（来自 candidate.elements）⇒ "
                           "不可区分时无法按元件数排显示顺序")
        best_bands = best.get("bands")
        if not best_bands or len(best_bands) != 3:
            caveats.append("rms 最低候选缺三段 `bands` ⇒ E10 分段门槛不可用，"
                           "本次**不做**元件数重排（缺数据不等于分段一致）")
            reorderable = False
        else:
            keep, dropped = [], []
            for cid in tied_ids:
                r = next(x for x in pool if x["candidate_id"] == cid)
                rb = r.get("bands")
                if not rb or len(rb) != 3:
                    dropped.append((cid, "分段数据缺失"))
                    continue
                deltas = [float(rb[i]) - float(best_bands[i]) for i in range(3)]
                worse = [d for d in deltas if d > tie_pp]
                if worse:
                    loc = ["低", "中", "高"][max(
                        range(3), key=lambda i: deltas[i])]
                    dropped.append((cid, f"分段连续比较：{loc}段比 best 高 "
                                         f"{max(worse):.2f}pp（容差 {tie_pp:g}pp）"))
                else:
                    keep.append(cid)
            tied_ids = keep
            if dropped:
                fire("E10", "任一频段比 best 实质更差（或缺分段数据）的候选"
                            "不进元件数重排组："
                            + "；".join(f"{c}（{why}）" for c, why in dropped)
                            + "；它们仍在候选表、仍参与状态判定，只是不因元件少被提前",
                     firm=True)

        if reorderable and len(tied_ids) > 1:
            order = sorted(tied_ids, key=lambda cid: (
                next(r["n_elements"] for r in pool if r["candidate_id"] == cid), cid))
            if order != tied_ids:
                fire("E9", f"这些候选三条件均满足（rms 并列、配对 elpd 不显著、"
                           f"分段连续比较无实质更差段）且元件数不同 ⇒ "
                           f"显示顺序按元件数升序：{'、'.join(order)}"
                           f"（只调整显示顺序，不宣布最优、不删候选）")
    else:
        fire("E5", f"{best['candidate_id']} 的 rms 比 rms 次低的候选低 "
                   f"{gap_pp:.2f}pp、"
                   f"Δelpd {'显著' if sig else '不显著'} ⇒ weak_preference")
        status = "weak_preference"
        reason = "best_leading_but_not_identified"

    # E6：两个状态不产出
    fire("E6", "`identified` / `model_inadequate` 当前口径下不可达，不产出",
         unreachable=list(UNREACHABLE))
    return _finish(status, viable, rules, caveats, reason=reason, best=best,
                   tie=dict(gap_pp=gap_pp, gap_rel=gap_rel,
                            tie_pp=float(tie_pp), tie_rel=float(tie_rel),
                            rms_tied=bool(rms_tied),
                            significant=bool(sig),
                            delta_elpd=d_elpd, delta_z=z,
                            delta_dse=dse, elpd_method=pair["method"],
                            criteria_conflict=bool(conflict)),
                   second=second, excluded=excluded,
                   sampling_failed=sampling_failed + failed_run,
                   not_qualified=not_qualified,
                   tie_pp=tie_pp, tie_rel=tie_rel, rms_accept=rms_accept,
                   candidate_set_complete=candidate_set_complete)


def build_confirmation(*, status, best, tie, viable, excluded, sampling_failed,
                       rms_accept, tie_pp, candidate_set_complete,
                       tie_rel=DEFAULT_TIE_REL) -> dict:
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
        line = (f"rms 最低候选 `{best['candidate_id']}` 的 rms_rel = "
                f"{best['rms_rel']:.2f}%")
        line += ("，在 §14 接受线内" if best["rms_rel"] <= rms_accept
                 else f"，**已超接受线 {rms_accept:g}%**")
        cf("residual_floor", line)
    if tie is not None:
        _rel = tie.get("gap_rel")
        tie_txt = ("残差口径：前两名差 "
                   f"{tie['gap_pp']:.2f}pp（阈值 {tie_pp:g}pp）"
                   + (f"、相对 {_rel:.1f}%（阈值 {tie_rel:g}%）" if _rel is not None else "")
                   + " ⇒ " + ("不可分" if tie.get("rms_tied",
                                                tie["gap_pp"] <= tie_pp) else "可分"))
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
        # 文案按 registry 实际状态生成（2026-11-20）：三个 Warburg 元件已 wired=True
        # 且有槽位，写死名单会一直说"w_inf/w_open/w_short 没接"——与产物不符。
        _unwired = [e for e in CR.list_elements() if not CR.get_element(e).wired]
        nc("candidate_set_completeness",
           f"registry 里仍有 `wired=False` 的元件 `{'/'.join(_unwired)}`"
           "（没有拓扑槽位或接点语义未定）⇒ 本次结论的作用域仅限已登记候选集（R12）",
           "给对应拓扑加槽位、把该元件的 `wired` 翻成 True（并同步预注册新臂）")

    return dict(
        confirmed=confirmed,
        not_confirmed=not_confirmed,
        scope=(f"状态 `{status}` 的作用域 = 已登记候选集 {n_all} 个；"
               "它不陈述「真实电路是什么」，只陈述「在这批候选与这个口径下能说到哪一步」"),
    )


def _finish(status, viable, rules, caveats, *, reason, best, tie,
            second=None, excluded=(), sampling_failed=(), not_qualified=(),
            tie_pp=DEFAULT_TIE_PP, tie_rel=DEFAULT_TIE_REL,
            rms_accept=DEFAULT_RMS_ACCEPT,
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
        tie_rel=tie_rel, candidate_set_complete=candidate_set_complete)
    return dict(
        verdict=block,
        status=status,
        reason=reason,
        confirmation=confirm,
        best=(None if best is None else best["candidate_id"]),
        second=(None if second is None else second["candidate_id"]),
        tie=tie,
        n_viable=len(viable),
        not_qualified=[r["candidate_id"] for r in not_qualified],
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
          + (f"，rms 最低 `{ev['best']}`" if ev["best"] else ""), ""]
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
        L += [f"- 前两名：rms 绝对差 **{t['gap_pp']:.2f}pp**（阈值 {t['tie_pp']:g}pp）"
              + (f"、相对差 **{t['gap_rel']:.1f}%**"
                 f"（阈值 {t.get('tie_rel', DEFAULT_TIE_REL):g}%）"
                 if t.get("gap_rel") is not None else "")
              + f" ⇒ rms 口径{'并列' if t.get('rms_tied', t['gap_pp'] <= t['tie_pp']) else '可分'}"
              + f"；`Δelpd={de}`"
              + (f"、`Δ/dse={t['delta_z']:.2f}`" if t.get("delta_z") is not None else "")
              + f"（{'paired' if t.get('elpd_method') == 'paired' else '边缘 se'}、"
              f"显著={t['significant']}；只作证据强弱，不改状态）"
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
         rms=5.0, acf=0.5, rhat=1.01, div=0, elpd=-100.0, se=5.0,
         bands=None):
    ob = CR.observation_block(
        candidate_id=cid, run_status=run, sampling_status=sampling,
        rms_rel=(None if run != "completed" else rms),
        acf=(None if run != "completed" else acf),
        rhat_max=(None if run != "completed" else rhat),
        divergences=(None if run != "completed" else div),
        predictive={"elpd": elpd, "se": se, "valid_for_comparison": False,
                    "reason": "R21"},
        interpretation=(None if sampling == "passed" else "descriptive_only"))
    if bands is not None:
        fo = ob.setdefault("fit_observation", {})
        fo["bands"] = list(bands)
    return ob


def _row(cid, **kw):
    pre = kw.pop("pre", "not_excluded")
    n_el = kw.pop("n_elements", None)
    pw = kw.pop("pw", None)
    ob = _obs(cid, pre=pre, **kw)
    cand = {"elements": [("e", "r")] * n_el} if n_el is not None else {}
    rec = dict(observation=ob, precheck=dict(precheck_state=pre), candidate=cand)
    if pw is not None:
        rec["pointwise_elpd"] = list(pw)
    return tabulate([rec])[0]


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
    # 两条都放在 10% 验收线内（16% 那档现在归 E8，不参与 E5 比较）
    rows = [_row("C-a", rms=8.0, elpd=-300.0, se=5.0),
            _row("C-b", rms=9.0, elpd=-400.0, se=5.0)]
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
    rows = [_row("C-a", rms=1.0, elpd=-10.0), _row("C-b", rms=9.5, elpd=-900.0)]
    ev6 = evaluate(rows)
    c("即使拟合极好也不会 identified", ev6["status"] != "identified")
    c("也不会 model_inadequate", ev6["status"] != "model_inadequate")
    c("两个不可达状态都给了理由",
      set(ev6["unreachable_states"]) == {"identified", "model_inadequate"}
      and all(ev6["unreachable_states"].values()))

    print("\nE8/E9/E10：先过验收线再比较，三条件都满足才按元件数列前")
    # 三条都过 10% 线：rms 6.0(3 元件) / 6.2(6 元件) / 6.1(9 元件)。
    # rms 差都在 tie_pp=1pp 内、Δelpd 不显著、分段模式全过 ⇒ non_identifiable 且重排。
    _allpass = (2.0, 3.0, 4.0)
    rows = [_row("C-many", rms=6.0, elpd=-100.0, se=5.0, n_elements=9,
                 bands=_allpass),
            _row("C-few", rms=6.2, elpd=-102.0, se=5.0, n_elements=3,
                 bands=_allpass),
            _row("C-mid", rms=6.1, elpd=-101.0, se=5.0, n_elements=6,
                 bands=_allpass)]
    ev8 = evaluate(rows)
    c("三条过线 ⇒ non_identifiable", ev8["status"] == "non_identifiable")
    c("E9 触发（三条件满足，按元件数排显示）",
      any(r["id"] == "E9" for r in ev8["rules"]))
    c("best 仍是 rms 最低的事实指针", ev8["best"] == "C-many")
    c("E10 不触发（分段模式全一致）",
      not any(r["id"] == "E10" for r in ev8["rules"]))

    # E10 核心：rms 差在并列带内、elpd 不显著，但一条分段模式不同 ⇒ 不进重排组。
    # 这正是 3e 里 5.81(nested,L=2,全过) vs 6.28(voigt K=1,高频超) 的情形。
    rows_e10 = [_row("C-lowrms", rms=5.81, elpd=-100.0, se=5.0,
                     n_elements=3, bands=(1.8, 1.0, 9.5)),
                _row("C-hfover", rms=6.28, elpd=-102.0, se=5.0,
                     n_elements=2, bands=(3.4, 3.5, 10.1))]
    ev10 = evaluate(rows_e10)
    c("分段不一致时状态仍 non_identifiable（E10 只管显示顺序）",
      ev10["status"] == "non_identifiable")
    c("E10 触发（分段模式不同者被踢出重排组）",
      any(r["id"] == "E10" for r in ev10["rules"]))
    c("E9 不触发（重排组只剩 1 个）",
      not any(r["id"] == "E9" for r in ev10["rules"]))
    c("best 仍是 rms 最低（真谱没被元件少的挤掉）",
      ev10["best"] == "C-lowrms")

    # E10 降级：bands 缺失 ⇒ 不重排，记 caveat，绝不拿缺失当"分段一致"。
    rows_nb = [_row("C-a", rms=5.8, elpd=-100.0, se=5.0, n_elements=3),
               _row("C-b", rms=6.1, elpd=-102.0, se=5.0, n_elements=2)]
    evnb = evaluate(rows_nb)
    c("缺 bands 时 E9 不触发（保守不重排）",
      not any(r["id"] == "E9" for r in evnb["rules"]))
    c("缺 bands 时记 caveat（缺数据≠分段一致）",
      any("bands" in x for x in evnb["caveats"]))

    print("\n配对 dse：边缘 se 大但配对差小 ⇒ 判显著（不再漏判）")
    # 两条逐点 elpd 各自波动大（边缘 se≈5），但逐点**差**恒定 ⇒ dse≈0 ⇒ 显著。
    _pw_a = [-100.0, -105.0, -110.0, -115.0, -120.0]
    _pw_b = [x - 9.0 for x in _pw_a]          # 逐点差恒为 +9
    rows_pw = [_row("C-pwa", rms=5.0, elpd=sum(_pw_a), se=8.0, pw=_pw_a),
               _row("C-pwb", rms=5.9, elpd=sum(_pw_b), se=8.0, pw=_pw_b)]
    evpw = evaluate(rows_pw)
    c("配对 dse 口径被采用", evpw["tie"]["elpd_method"] == "paired")
    c("配对 dse≈0（逐点差恒定）", abs(evpw["tie"]["delta_dse"]) < 1e-6)
    c("边缘 se 会漏判、配对判显著 ⇒ weak_preference",
      evpw["status"] == "weak_preference")
    c("E4 文本标出 paired",
      any("paired" in (r.get("text") or "") for r in evpw["rules"] if r["id"] == "E4"))
    # 用户判例：rms 差 0.16pp、rel 2.8%、三段同档 ⇒ 即使配对 elpd z=6 也必须判不可分。
    _pw_c = [-100.0 + i * 0.1 for i in range(5)]
    rows_case = [
        _row("C-low", rms=5.97, elpd=sum(_pw_c), se=8.0, pw=_pw_c,
             bands=(1.2, 1.3, 9.9), n_elements=2),
        _row("C-high", rms=5.81, elpd=sum(x - 7 for x in _pw_c), se=8.0,
             pw=[x - 7 for x in _pw_c], bands=(1.8, 1.0, 9.5), n_elements=3),
    ]
    evcase = evaluate(rows_case)
    c("1.2/1.3/9.9 vs 1.8/1.0/9.5 ⇒ 不可区分（elpd 显著也不否决）",
      evcase["status"] == "non_identifiable")
    c("该判例 rel 2.8% 被记录",
      abs(evcase["tie"]["gap_rel"] - 2.8) < 0.5)
    c("该判例 elpd 仍显著、冲突被记账",
      evcase["tie"]["significant"] and evcase["tie"]["criteria_conflict"])
    # 用户判例：1/1/9 vs 3/3/9 ⇒ abs 2pp、rel 200% ⇒ 可分
    rows_far = [_row("C-a", rms=1.0, bands=(1.0, 1.0, 9.0)),
                _row("C-b", rms=3.0, bands=(3.0, 3.0, 9.0))]
    c("1/1/9 vs 3/3/9 ⇒ 可分",
      evaluate(rows_far)["status"] == "weak_preference")
    # 无逐点时降级为边缘 se，记 caveat，不冒充配对
    evmarg = evaluate([_row("C-a", rms=5.0, elpd=-100.0, se=6.0),
                       _row("C-b", rms=5.9, elpd=-101.0, se=6.0)])
    c("无逐点时降级为 marginal_se", evmarg["tie"]["elpd_method"] == "marginal_se")
    c("降级时记 caveat", any("边缘 se" in x for x in evmarg["caveats"]))

    print("\n分段连续比较：9.9 vs 10.1 不因二值被拆，1.2 vs 3.4 被识别为实质差异")
    # 两条三段都过线，但一条三段 1.x、另一条低中频 3.x（差 >1pp）⇒ 踢出重排组
    rows_cont = [_row("C-1x", rms=5.0, elpd=-100.0, se=5.0,
                      n_elements=3, bands=(1.2, 1.3, 1.5)),
                 _row("C-3x", rms=5.4, elpd=-101.0, se=5.0,
                      n_elements=2, bands=(3.4, 3.5, 2.0))]
    evc = evaluate(rows_cont)
    c("1.x vs 3.x：E10 识别为实质更差段",
      any(r["id"] == "E10" for r in evc["rules"]))
    c("1.x vs 3.x：元件少者不排前（E9 不触发）",
      not any(r["id"] == "E9" for r in evc["rules"]))
    # 9.9 vs 10.1 只差 0.2pp（<1pp 容差）⇒ 不被拆，仍可重排
    rows_edge = [_row("C-99", rms=5.0, elpd=-100.0, se=5.0,
                      n_elements=3, bands=(1.0, 1.0, 9.9)),
                 _row("C-101", rms=5.3, elpd=-101.0, se=5.0,
                      n_elements=2, bands=(1.1, 1.1, 10.1))]
    eve = evaluate(rows_edge)
    c("9.9 vs 10.1：差 0.2pp < 容差 ⇒ E10 不触发",
      not any(r["id"] == "E10" for r in eve["rules"]))
    c("9.9 vs 10.1：元件少者列前（E9 触发）",
      any(r["id"] == "E9" for r in eve["rules"]))
    # 部分过线：一条 15% 超线，两条过线 ⇒ 超线的单列、不删、不进比较
    rows2 = [_row("C-ok1", rms=6.0, elpd=-100.0, se=5.0, n_elements=3),
             _row("C-ok2", rms=6.2, elpd=-102.0, se=5.0, n_elements=6),
             _row("C-over", rms=15.0, elpd=-90.0, se=5.0, n_elements=1)]
    ev8b = evaluate(rows2)
    c("E8 触发（超线者单列）", any(r["id"] == "E8" for r in ev8b["rules"]))
    c("超线者在 not_qualified 里", ev8b["not_qualified"] == ["C-over"])
    c("超线者仍在候选表里不丢", "C-over" in [r["candidate_id"] for r in ev8b["candidates"]])
    c("比较只在两条过线者间做", set(ev8b["viable"]) == {"C-ok1", "C-ok2", "C-over"}
      and ev8b["best"] in ("C-ok1", "C-ok2"))

    print("\n状态一定落在 registry 允许的集合里")
    for evx in (ev, ev2, ev3, ev4, ev5, ev6, ev8, ev8b):
        c(f"状态 {evx['status']} 合法", evx["status"] in CR.EVIDENCE_STATES)
        c(f"evidence_block 已通过校验（{evx['status']}）",
          evx["verdict"]["status"] == evx["status"]
          and evx["verdict"]["scope"] == CR.EVIDENCE_SCOPE)

    print("\ntie_pp / tie_rel 是显式参数，不是硬编码")
    # 两条都在验收线内，rel 也调宽以单独测 tie_pp 本身，不是验收线/rel 门槛
    rows = [_row("C-a", rms=8.0), _row("C-b", rms=9.5)]
    c("tie_pp=1.0 → 差 1.5pp 判 weak_preference",
      evaluate(rows, tie_pp=1.0, tie_rel=100.0)["status"] == "weak_preference")
    c("tie_pp=2.0 → 同一个输入判 non_identifiable",
      evaluate(rows, tie_pp=2.0, tie_rel=100.0)["status"] == "non_identifiable")
    # rel 门槛单独生效：abs 够小但 rel 超线 ⇒ 仍判可分（低 rms 谱的 1pp 不是小差异）
    rows_rel = [_row("C-a", rms=1.0), _row("C-b", rms=1.8)]
    c("abs 0.8pp<1.0 但 rel 80%>10% ⇒ 判可分",
      evaluate(rows_rel, tie_pp=1.0, tie_rel=10.0)["status"] == "weak_preference")
    c("tie_rel=100% → 同一对判不可分（证明 rel 生效）",
      evaluate(rows_rel, tie_pp=1.0, tie_rel=100.0)["status"] == "non_identifiable")

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

    print("\n给事实不排序：人读文案不含排序/推荐结论词")
    # 内部字段名 `best` / `second` 只指向「rms 最低/次低」这个事实（显示顺序可用），
    # 禁的是把它们说成「最优模型 / 推荐 / 选中」这类结论。检查范围只覆盖人读文案。
    prose = "\n".join(
        [md, render(ev2), render(ev7)]
        + [r["text"] for r in ev["rules"] + ev2["rules"] + ev7["rules"]]
        + [x["text"] for e in (ev, ev2, ev7) for x in e["confirmation"]["confirmed"]])
    for word in ("最优", "最佳", "推荐", "选中", "胜出", "领先"):
        c(f"人读文案无「{word}」", word not in prose)

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
                 tie_rel: float = DEFAULT_TIE_REL,
                 rms_accept: float = DEFAULT_RMS_ACCEPT) -> dict:
    """读目录 → 评价 → 落 `evaluation.json` / `evaluation.md`。"""
    d = load_run(run_dir)
    rows = tabulate(d["records"])
    complete = not [e for e in CR.list_elements()
                    if not CR.get_element(e).wired]
    ev = evaluate(rows, tie_pp=tie_pp, tie_rel=tie_rel, rms_accept=rms_accept,
                  candidate_set_complete=complete)
    ev["inputs"] = dict(run_dir=run_dir,
                        data_path=(d["plan"] or {}).get("data_path"),
                        data_sha256=(d["plan"] or {}).get("data_sha256"),
                        budget_profile=(d["plan"] or {}).get("budget_profile"),
                        n_records=len(d["records"]),
                        registry_version=CR.REGISTRY_VERSION,
                        registry_digest=CR.registry_digest(),
                        tie_pp=float(tie_pp), tie_rel=float(tie_rel),
                        rms_accept=float(rms_accept))
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
                    help=f"并列绝对阈值（百分点，默认 {DEFAULT_TIE_PP:g}；有争议，需裁决）")
    ap.add_argument("--tie-rel", type=float, default=DEFAULT_TIE_REL,
                    help=f"并列相对阈值（%%，默认 {DEFAULT_TIE_REL:g}；"
                         "须与 --tie-pp 同时满足才算并列）")
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
        ev = evaluate_run(r, tie_pp=a.tie_pp, tie_rel=a.tie_rel,
                          rms_accept=a.rms_accept)
        if not a.quiet:
            print(render(ev, title=os.path.basename(os.path.normpath(r))))
            print()
    print(f"已评价 {len(runs)} 个目录；tie_pp={a.tie_pp:g}pp、"
          f"rms_accept={a.rms_accept:g}%（两者已于 2026-11-16 裁决）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
