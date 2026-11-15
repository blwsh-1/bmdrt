# -*- coding: utf-8 -*-
"""test_identity_p0.py —— 身份链 P0 修复的四项反例 + 版本升级（v1.5）。

用户给的通过标准：**四项 P0 都有反例测试，错误记录不能生成有效身份**。
每一条反例都断言 `evaluate()` ⇒ `unconfirmed` 且 `run_id` / `workflow_id`
**同时为 None**——只降一档发号也算失败。

分组
----
· **G1**（6 + 正例）`source=recorded` 的结构化证据门槛：裸块、缺证据块、
  上游未捕获、指纹篡改、`origin=unknown`、`where` 缺失；
· **G2**（7）生产捕获接口 + 事件→六键适配：真实事件链路、词表对齐、
  算法逐位对齐、禁止升级、拒绝外部自称闭合；
· **G3**（3）`retry_kw` 去二次承载：含五键不发号、策略键不换号、载荷无整块；
· **G4**（4）重跑身份独立：同预算两次不同号、不同预算不同号、
  子记录不带父预算/实例、子记录可发号；
· **G5**（6）版本升级 1.4→1.5：记录字段/载荷/校验三项齐全、跨世代不可比、
  升键表版本必换号。
· **F 组**（8，Gate 2A 最小修复集 F-03～F-09）值域 / hex 指纹 / 捕获词表闭集 /
  workflow 禁止二次承载预算 / children 只放引用 / lineage 不撞号 / 重跑原因区分，
  外加 F-00 主路径未误伤的正对照。

    python test_identity_p0.py
"""
from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import budget_capture_adapter as BA                 # noqa: E402
import budget_params as B                           # noqa: E402
import exec_identity as E                           # noqa: E402
import sampler_provenance as SP                     # noqa: E402
import test_exec_identity as TE                     # noqa: E402


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def _ev(rec, **kw) -> dict:
    return E.evaluate(rec, **kw)


def _no_identity(rec, **kw) -> list[str]:
    """错误记录不许生成任何有效身份（不是降 derived，也不是只空一个号）。"""
    out = _ev(rec, **kw)
    assert out["status"] == "unconfirmed", f"错误记录却不是 unconfirmed：{out}"
    assert out["run_id"] is None, f"错误记录发了 run_id：{out}"
    assert out["workflow_id"] is None, f"错误记录发了 workflow_id：{out}"
    return out["reasons"]


def _mutate(**over) -> dict:
    r = TE._rec()
    r.update(over)
    return r


def _evidence(block: dict, **over) -> dict:
    e = TE._budget_evidence(block)
    e.update(over)
    return e


#: F-08：子运行必须带合法父工作流引用（否则不同 lineage 会撞号）。
_PARENT_REF = "W2-" + "a" * 32


def _child(attempt: int, values=(2000, 1000, 2, 0.98, 10), *,
           reason: str | None = "sampler_error",
           parent_ref: str | None = _PARENT_REF) -> dict:
    bud = TE._budget_cells(values)
    evi = TE._budget_evidence(bud)
    parent = TE._rec(nonce=f"parent-{attempt}")
    inst = {"data_sha256": parent["data"]["sha256"],
            "candidate_id": parent["candidate_id"],
            "site": E.FROZEN_PRIMARY_SITE,
            "nonce": f"rerun-attempt-{attempt}"}
    att = E.issue_capture_attestation(event_sha256=evi["event_sha256"],
                                      budget_sha256=evi["evidence_sha256"],
                                      instance=inst)
    return E.make_rerun_record(
        parent, attempt=attempt, budget_params=bud, budget_evidence=evi,
        reason=reason, parent_workflow_ref=parent_ref,
        execution_instance=inst, capture_attestation=att)


def _real_event() -> tuple[dict | None, str]:
    """取一条**真实** `capture_at_construction()` 事件；numpyro 不在场就明账降级。"""
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import sampler_capture as SC
        m = SC._mcmc(kernel="NUTS", num_chains=1, num_warmup=8, num_samples=8,
                     target_accept=0.9, max_tree_depth=10, progress=False)
        passed = dict(n_warmup=8, n_samples=8, num_chains=1, target_accept=0.9,
                      max_tree_depth=10, progress_bar=False,
                      extra_fields=("diverging",))
        ev = SC.capture_at_construction(
            m, site="bm_core.fit", passed=passed,
            run_call=dict(extra_fields=("diverging",)))
        return ev, "真实 MCMC 构造边界捕获事件"
    except ImportError as exc:
        return None, f"numpyro 不在场（{exc.__class__.__name__}）：本格改用等价构造事件"


def _synthetic_event() -> dict:
    """与真实事件同形的构造事件（供无 numpyro 环境跑形状链路）。"""
    return {
        "capture_version": "1.0", "site": "bm_core.fit",
        "scope": "construction_boundary", "upstream_origin_captured": False,
        "judged": True,
        "fields": {k: {"resolved": {"value": v, "origin": "derived",
                                    "where": "bm_core.fit:构造实参"},
                       "observed": {"value": v, "how": "attr", "where": "mcmc"},
                       "observed_status": "captured", "status": "complete",
                       "judged": True}
                   for k, v in zip(B.BUDGET_KEYS, (8, 8, 1, 0.9, 10))},
    }


# ---------------------------------------------------------------------------
# G1：结构化证据门槛
# ---------------------------------------------------------------------------

def case_G1_positive():
    r = TE._rec()
    out = _ev(r)
    assert out["status"] == "recorded", out["reasons"]
    assert out["run_id"].startswith("R2-") and out["workflow_id"].startswith("W2-")
    assert E._chk_budget(r["budget_params"], r[E.BUDGET_EVIDENCE_FIELD]) == []
    return f"正例：六键 + 合规证据块 ⇒ recorded（{out['run_id'][:12]}…）"


def case_G1_bare_recorded_cells_rejected():
    """第 4 步复裁的原漏洞：裸 `{value, source:'recorded'}` 曾照发 R1- 号。"""
    r = TE._rec()
    r["budget_params"] = {k: {"value": v, "source": "recorded"} for k, v in
                          zip(E.BUDGET_KEYS, (2000, 1000, 2, 0.98, 10))}
    why = _no_identity(r)
    assert any(x.startswith("budget_cell_shape:") for x in why), why
    return "G1① 裸 recorded 复合块 ⇒ 形状门槛拒，零号"


def case_G1_missing_evidence_block():
    r = TE._rec()
    del r[E.BUDGET_EVIDENCE_FIELD]
    why = _no_identity(r)
    assert "budget_evidence_missing" in why, why
    return "G1② 六键齐全但缺显式证据块 ⇒ 拒"


def case_G1_upstream_not_captured():
    r = TE._rec()
    r[E.BUDGET_EVIDENCE_FIELD] = _evidence(
        r["budget_params"], upstream_origin_captured=False)
    why = _no_identity(r)
    assert "budget_evidence_upstream_not_captured" in why, why
    assert "budget_recorded_without_upstream" in why, why
    return "G1③ upstream_origin_captured=False 却五项 recorded ⇒ 拒"


def case_G1_evidence_sha_tampered():
    r = TE._rec()
    r[E.BUDGET_EVIDENCE_FIELD] = _evidence(
        r["budget_params"], evidence_sha256="0" * 64)
    why = _no_identity(r)
    assert "budget_evidence_sha_mismatch" in why, why
    return "G1④ 指纹篡改（验证完再改块）⇒ 重算拒"


def case_G1_origin_unknown():
    r = TE._rec()
    r["budget_params"]["n_warmup"]["origin"] = "unknown"
    r[E.BUDGET_EVIDENCE_FIELD] = _evidence(r["budget_params"])
    why = _no_identity(r)
    assert "budget_recorded_origin_unknown:n_warmup" in why, why
    return "G1⑤ recorded 但 origin=unknown ⇒ 拒（不把推断值升级）"


def case_G1_where_missing():
    r = TE._rec()
    r["budget_params"]["n_warmup"]["where"] = None
    r[E.BUDGET_EVIDENCE_FIELD] = _evidence(r["budget_params"])
    why = _no_identity(r)
    assert "budget_recorded_where_missing:n_warmup" in why, why
    return "G1⑥ recorded 但 where 缺失 ⇒ 拒"


# ---------------------------------------------------------------------------
# G2：生产捕获接口 + 事件→六键适配层
# ---------------------------------------------------------------------------

def case_G2_real_capture_carries_six_key_block():
    ev, how = _real_event()
    if ev is None:
        synthetic = _synthetic_event()
        out = BA.adapt_capture_event(synthetic)
        clean = B.validate_budget_params(
            out["budget_params"], upstream_origin_captured=False)
        assert list(clean) == list(B.BUDGET_KEYS)
        return f"G2① {how}；六键合规（降级路径）"
    assert ev["budget_adaptation_error"] is None, ev["budget_adaptation_error"]
    assert list(ev["budget_params"]) == list(B.BUDGET_KEYS)
    assert all(list(c) == list(B.CELL_KEYS) for c in ev["budget_params"].values())
    clean = B.validate_budget_params(
        ev["budget_params"], upstream_origin_captured=False)
    assert list(clean) == list(B.BUDGET_KEYS)
    assert ev["budget_evidence"]["evidence_sha256"] == \
        BA.budget_evidence_sha256(clean)
    return f"G2① {how}：事件自带六键块 + 证据块，产出层校验通过"


def case_G2_adapted_output_cannot_get_identity():
    """红线：值已捕获但来源未闭合 ⇒ unconfirmed 无号，不是 recorded。"""
    ev, how = _real_event()
    out = BA.adapt_capture_event(ev if ev is not None else _synthetic_event())
    r = TE._rec()
    r["budget_params"] = out["budget_params"]
    r[E.BUDGET_EVIDENCE_FIELD] = out["budget_evidence"]
    why = _no_identity(r)
    assert any(x.startswith("budget_not_recorded:") for x in why), why
    assert "budget_evidence_upstream_not_captured" in why, why
    assert all(c["source"] == "unavailable"
               for c in out["budget_params"].values())
    return f"G2② 适配输出喂 evaluate() ⇒ unconfirmed 零号（{how}）"


def case_G2_promote_unavailable_to_recorded_rejected():
    """把适配输出的 unavailable 硬改成 recorded，仍不许发号。"""
    ev, _ = _real_event()
    out = BA.adapt_capture_event(ev if ev is not None else _synthetic_event())
    bud = copy.deepcopy(out["budget_params"])
    for k, v in zip(E.BUDGET_KEYS, (8, 8, 1, 0.9, 10)):
        bud[k].update(value=v, source="recorded", origin="cli_explicit",
                      where=f"bm_real.KW.{k}")
    r = TE._rec()
    r["budget_params"] = bud
    e = dict(out["budget_evidence"], evidence_sha256=E.budget_evidence_sha256(bud))
    r[E.BUDGET_EVIDENCE_FIELD] = e
    why = _no_identity(r)
    assert "budget_evidence_upstream_not_captured" in why, why
    assert "budget_recorded_without_upstream" in why, why
    return "G2③ unavailable→recorded 硬改 ⇒ 上游证据门槛仍拒"


def case_G2_external_upstream_claim_rejected():
    ev, _ = _real_event()
    src = ev if ev is not None else _synthetic_event()
    try:
        BA.adapt_capture_event(src, upstream_claim=True)
    except ValueError as exc:
        assert "upstream_claim" in str(exc), str(exc)
    else:
        raise AssertionError("外部自称上游已闭合竟被接受")
    try:
        BA.adapt_capture_event(dict(src, upstream_origin_captured=True))
    except ValueError as exc:
        assert "upstream_origin_captured" in str(exc), str(exc)
    else:
        raise AssertionError("事件自称上游已闭合竟被接受")
    return "G2④ 外部 upstream_claim / 事件 True 两种上调入口都 raise"


def case_G2_origin_vocabulary_refuses_mapping():
    assert not (set(SP.ORIGINS) & set(B.BUDGET_ORIGINS)), \
        "SP.ORIGINS 与 BUDGET_ORIGINS 交集非空，需重审拒绝映射"
    assert BA.ORIGIN_MAPPING == {}, "不许静默挑选相近 origin"
    assert BA.SP_ORIGINS == SP.ORIGINS
    assert BA.BUDGET_ORIGINS == B.BUDGET_ORIGINS
    return "G2⑤ 词表交集 ∅、映射表为空、两份复刻词表逐项相等"


def case_G2_vocabularies_align_with_identity():
    assert BA.BUDGET_KEYS == B.BUDGET_KEYS == E.BUDGET_KEYS
    assert BA.CELL_KEYS == B.CELL_KEYS == E.BUDGET_CELL_KEYS
    assert BA.BUDGET_SOURCE_STATES == B.BUDGET_SOURCE_STATES == E.BUDGET_SOURCE_STATES
    assert BA.BUDGET_ORIGINS == B.BUDGET_ORIGINS == E.BUDGET_ORIGINS
    assert BA.EVIDENCE_KEYS == E.BUDGET_EVIDENCE_KEYS
    blocks = [TE._budget_cells(), B.validate_budget_params(
        {k: B.make_cell(value=v, source="unavailable", origin="unknown")
         for k, v in zip(B.BUDGET_KEYS, (None,) * 5)},
        upstream_origin_captured=False)]
    for blk in blocks:
        assert BA.budget_evidence_sha256(blk) == E.budget_evidence_sha256(blk)
    samples = [{"a": 1, "b": [1, 2.5, None, True]}, [1, "x", {"z": (1, 2)}],
               {"value": 0.1, "where": "bm_real.KW"}]
    for obj in samples:
        assert BA.canonical_json(obj) == E.canonical_json(obj), obj
    return "G2⑥ 三层词表逐项相等；两个指纹算法与 canonical_json 逐位对齐"


def case_G2_observations_are_not_identity_sources():
    ev, _ = _real_event()
    out = BA.adapt_capture_event(ev if ev is not None else _synthetic_event())
    obs = out["observations"]
    assert set(obs) == set(E.BUDGET_KEYS)
    assert obs["max_tree_depth"]["resolved_value"] == 10
    assert obs["max_tree_depth"]["resolved_origin_literal"] == "derived"
    assert out["budget_params"]["max_tree_depth"]["value"] is None
    r = TE._rec()
    payload = E.run_id_payload(r)
    assert "observations" not in E.canonical_json(payload)
    assert "budget_observations" not in E.canonical_json(payload)
    return "G2⑦ 观测值只在 observations 明账，不进任何身份载荷"


# ---------------------------------------------------------------------------
# G3：retry_kw 去二次承载
# ---------------------------------------------------------------------------

def case_G3_budget_keys_in_retry_kw_no_identity():
    vals = dict(zip(E.BUDGET_KEYS, (800, 400, 2, 0.95, 13)))
    r = _mutate(workflow=dict(TE._rec()["workflow"], retry_kw=vals))
    why = _no_identity(r)
    assert "retry_kw_budget_keys:" + ",".join(E.BUDGET_KEYS) in why, why
    return "G3① retry_kw 含五项预算 ⇒ fatal、两号皆 None"


def case_G3_strategy_only_retry_kw_no_id_change():
    base = _ev(TE._rec())
    r = _mutate(workflow=dict(TE._rec()["workflow"],
                              retry_kw={"trigger": "acf>0.50", "max_retries": 2}))
    out = _ev(r)
    assert out["status"] == "recorded", out["reasons"]
    assert out["workflow_id"] == base["workflow_id"], "策略键竟换了 workflow_id"
    assert out["run_id"] == base["run_id"]
    return "G3② 只含重跑策略 ⇒ 号不变（预算不进父号）"


def case_G3_retry_kw_removed_from_payload():
    r = TE._rec()
    assert "retry_kw" not in E.workflow_payload(r)
    assert "retry_kw" not in E.run_id_payload(r)
    return "G3③ workflow_payload 顶层已无 retry_kw 键"


# ---------------------------------------------------------------------------
# G4：重跑身份独立
# ---------------------------------------------------------------------------

def case_G4_same_budget_two_attempts_different_ids():
    a = _ev(_child(1), role="child")
    b = _ev(_child(2), role="child")
    assert a["status"] == b["status"] == "recorded", (a["reasons"], b["reasons"])
    assert a["run_id"] != b["run_id"], "同预算重跑两次竟同号"
    return "G4① 同预算 attempt 1/2 ⇒ 两个独立 run_id"


def case_G4_different_budget_different_id():
    a = _ev(_child(1), role="child")
    b = _ev(_child(1, (800, 400, 2, 0.95, 13)), role="child")
    assert a["status"] == b["status"] == "recorded", (a["reasons"], b["reasons"])
    assert a["run_id"] != b["run_id"], "换预算竟没换号"
    return "G4② 换预算 ⇒ 换子 run_id"


def case_G4_child_has_no_parent_budget_or_instance():
    parent = TE._rec()
    parent["workflow"]["retry_kw"] = {"trigger": "acf>0.50"}
    parent["instance"] = {"pid": 4242}
    bud = TE._budget_cells()
    child = E.make_rerun_record(
        parent, attempt=1, budget_params=bud,
        budget_evidence=TE._budget_evidence(bud),
        reason="sampler_error", parent_workflow_ref=_PARENT_REF)
    assert "instance" not in child
    assert "provenance" not in child
    assert child["workflow"]["retry_kw"] == {}
    assert child["workflow"]["children"] == []
    assert child[E.RERUN_FIELD]["attempt"] == 1
    assert child["budget_params"] is not bud
    return "G4③ 子记录不含父预算 / 实例 / 结果；children 只留空引用"


def case_G4_children_only_hold_references():
    parent = TE._rec()
    sub = _ev(_child(1), role="child")
    p2 = _mutate(workflow=dict(parent["workflow"],
                               children=[sub["run_id"]], retry=1))
    out = _ev(p2)
    assert out["status"] == "recorded", out["reasons"]
    assert len(p2["workflow"]["children"]) == 1
    assert not any(k in E.canonical_json(p2["workflow"]["children"])
                   for k in E.BUDGET_KEYS)
    return "G4④ 父 children 只放子号引用，不含任何预算数值"


# ---------------------------------------------------------------------------
# G5：版本升级 1.4 → 1.5
# ---------------------------------------------------------------------------

def case_G5_missing_keys_version_no_identity():
    r = TE._rec()
    del r[E.RUN_ID_CONFIG_KEYS_VERSION_FIELD]
    why = _no_identity(r)
    assert any(x.startswith("unhashable:") for x in why), why
    return "G5① 记录缺 run_id_config_keys_version ⇒ 无号"


def case_G5_stale_and_malformed_keys_version():
    for bad in (1, "2", True, None, 0):
        r = TE._rec()
        r[E.RUN_ID_CONFIG_KEYS_VERSION_FIELD] = bad
        _no_identity(r)
    return "G5② 旧世代 1 / 字符串 / bool / None / 0 全部无号"


def case_G5_version_in_both_payloads():
    r = TE._rec()
    for payload in (E.run_id_payload(r), E.workflow_payload(r)):
        assert payload["config_keys_version"] == E.RUN_ID_CONFIG_KEYS_VERSION
        assert payload["v"] == E.IDENTITY_CONTRACT_VERSION
    assert E.IDENTITY_CONTRACT_VERSION == "1.5"
    assert E.RUN_ID_CONFIG_KEYS_VERSION == 2
    return "G5③ 1.4→1.5 + 键表版本 1→2，两个载荷都含版本字段"


def case_G5_new_generation_not_comparable():
    assert E.id_generation("R2-" + "a" * 32) == "exec_identity_1.5"
    assert E.id_generation("D2-" + "a" * 32) == "exec_identity_1.5"
    assert E.id_generation("R1-" + "a" * 32) == "exec_identity_1.4"
    cross = E.can_compare_ids("R1-" + "a" * 32, "R2-" + "b" * 32)
    assert cross["comparable"] is False and cross["declares_equivalence"] is False
    assert E.can_compare_ids("R2-" + "a" * 32, "R2-" + "b" * 32)["comparable"] is True
    return "G5④ R2-/D2- 是新世代；与 R1-/D1- 跨世代不可比、恒不声明等价"


def case_G5_bumping_keys_version_changes_id():
    r = TE._rec()
    before = _ev(r)["run_id"]
    old = E.RUN_ID_CONFIG_KEYS_VERSION
    try:
        E.RUN_ID_CONFIG_KEYS_VERSION = 3
        after = _ev(TE._rec())["run_id"]
        stale = _ev(r)  # 记录仍写 2，常量已是 3 ⇒ 不许发号
    finally:
        E.RUN_ID_CONFIG_KEYS_VERSION = old
    assert before and after and before != after, "升键表版本却没换号"
    assert stale["run_id"] is None and stale["workflow_id"] is None
    return "G5⑤ 键表版本 2→3 且记录同步 ⇒ 全换号；记录不同步 ⇒ 无号"


def case_G5_identity_module_still_zero_production_import():
    tree = ast.parse((_ROOT / "exec_identity.py").read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    forbidden = {"budget_params", "budget_capture_adapter", "sampler_capture",
                 "sampler_provenance", "patch_bundle", "capture_contract"}
    assert not (mods & forbidden), mods & forbidden
    return f"G5⑥ 身份层零生产 import（实际 {sorted(mods)}）"


# ---------------------------------------------------------------------------
# Gate 2A 最小修复集反例：F-03～F-09
# 每条错误记录都断言 unconfirmed + 两号同时 None（与 G 组同一通过标准）
# ---------------------------------------------------------------------------

def _with_budget_value(key: str, value) -> dict:
    """改一个预算值并**同步重算证据指纹**：确保测到的是值域门禁本身。"""
    bud = TE._budget_cells()
    bud[key]["value"] = value
    return _mutate(budget_params=bud,
                   budget_evidence=TE._budget_evidence(bud))


def case_F03_value_domain():
    for key, val in [("n_warmup", -1), ("n_warmup", 0), ("n_warmup", 2.5),
                     ("num_chains", 0), ("max_tree_depth", -3),
                     ("target_accept", 0.0), ("target_accept", 1.0),
                     ("target_accept", 1.2), ("n_samples", None),
                     ("n_samples", True), ("n_samples", "1000")]:
        why = _no_identity(_with_budget_value(key, val))
        assert any("budget_value_invalid" in x for x in why), (key, val, why)
    out = _ev(_with_budget_value("n_samples", 1000))
    assert out["status"] == "recorded" and out["run_id"].startswith("R2-"), out
    return "F-03 负/零/非整数/越界/类型错误全无号；合法 1000 仍发 R2-"


def case_F04_event_sha_is_hex():
    bud = TE._budget_cells()
    for tag, es in [("非 hex", "z" * 64), ("中文", "中" * 64),
                    ("空格", " " * 64), ("63 位", "a" * 63),
                    ("大写 hex", "A" * 64), ("None", None), ("int", 123)]:
        evi = TE._budget_evidence(bud)
        evi["event_sha256"] = es
        why = _no_identity(_mutate(budget_evidence=evi))
        assert any("event_sha_invalid" in x for x in why), (tag, why)
    return "F-04 event_sha256 必须是 64 位小写 hex；只校长度已修"


def case_F05_capture_closed_sets():
    bud = TE._budget_cells()
    for key, val, code in [("capture_version", "FORGED", "capture_version_invalid"),
                           ("capture_version", "2.0", "capture_version_invalid"),
                           ("capture_version", None, "capture_version_invalid"),
                           ("capture_site", "not.a.real.site", "capture_site_invalid"),
                           ("capture_site", "", "capture_site_invalid")]:
        evi = TE._budget_evidence(bud)
        evi[key] = val
        why = _no_identity(_mutate(budget_evidence=evi))
        assert any(code in x for x in why), (key, val, why)
    import sampler_capture as SC
    assert set(E.BUDGET_CAPTURE_SITES) == set(SC.CALL_SITES), "站点闭集与捕获层不一致"
    assert E.BUDGET_CAPTURE_VERSIONS == (SC.CAPTURE_VERSION,), "版本闭集与捕获层不一致"
    return "F-05 版本/站点只认闭集，且与 sampler_capture 逐项交叉核验"


def case_F06_workflow_no_budget_smuggling():
    base_wid = _ev(TE._rec())["workflow_id"]
    for f in ("search", "extras", "guards", "gate", "retry"):
        r = TE._rec()
        wf = dict(r["workflow"])
        cur = wf.get(f)
        val = dict(cur) if isinstance(cur, dict) else {}
        val["n_samples"] = 1000
        wf[f] = val
        why = _no_identity(_mutate(workflow=wf))
        assert any(f"workflow_budget_keys:{f}" in x for x in why), (f, why)
    wf = dict(TE._rec()["workflow"])
    wf["pair_mode"] = "n_samples=1000"
    why = _no_identity(_mutate(workflow=wf))
    assert any("workflow_budget_keys:pair_mode" in x for x in why), why
    wf = dict(TE._rec()["workflow"])
    wf["retry_kw"] = {"outer": {"n_warmup": 800}}
    why = _no_identity(_mutate(workflow=wf))
    assert any("retry_kw_budget_keys" in x for x in why), why
    wf = dict(TE._rec()["workflow"])
    wf["retry_kw"] = {"trigger": "acf>0.50", "max_retries": 2}
    assert _ev(_mutate(workflow=wf))["workflow_id"] == base_wid, "纯策略键不该换号"
    return "F-06 五个 _plain 整块 + pair_mode + 嵌套 retry_kw 全被拦；策略键不换号"


def case_F07_children_reference_only():
    for ch, code in [(["n_samples=1000"], "workflow_children_not_reference"),
                     ([{"n_samples": 1000}], "workflow_children_not_reference"),
                     ([None], "workflow_children_not_reference"),
                     ("R2-" + "a" * 32, "workflow_children_not_list")]:
        wf = dict(TE._rec()["workflow"])
        wf["children"] = ch
        why = _no_identity(_mutate(workflow=wf))
        assert any(code in x for x in why), (ch, why)
    wf = dict(TE._rec()["workflow"])
    wf["children"] = ["R2-" + "f" * 32]
    out = _ev(_mutate(workflow=wf))
    assert out["status"] == "recorded", out["reasons"]
    assert out["workflow_id"] != _ev(TE._rec())["workflow_id"], "合法引用应改 workflow_id"
    return "F-07 children 只许身份号引用；合法引用仍正常进 workflow_id"


def case_F08_lineage_no_collision():
    pa, pb = TE._rec(), TE._rec()
    pa["workflow"]["retry"] = 1
    pb["workflow"]["retry"] = 3
    wa, wb = _ev(pa)["workflow_id"], _ev(pb)["workflow_id"]
    assert wa != wb, "两个父工作流的 workflow_id 本该不同"
    bud = TE._budget_cells()
    evi = TE._budget_evidence(bud)

    def _signed_child(parent, ref, tag):
        inst = {"data_sha256": parent["data"]["sha256"],
                "candidate_id": parent["candidate_id"],
                "site": E.FROZEN_PRIMARY_SITE, "nonce": tag}
        att = E.issue_capture_attestation(event_sha256=evi["event_sha256"],
                                          budget_sha256=evi["evidence_sha256"],
                                          instance=inst)
        return E.make_rerun_record(parent, attempt=1, budget_params=bud,
                                   budget_evidence=evi, reason="sampler_error",
                                   parent_workflow_ref=ref,
                                   execution_instance=inst,
                                   capture_attestation=att)

    ca = _signed_child(pa, wa, "lineage-a")
    cb = _signed_child(pb, wb, "lineage-b")
    a, b = _ev(ca, role="child"), _ev(cb, role="child")
    assert a["status"] == b["status"] == "recorded", (a["reasons"], b["reasons"])
    assert a["run_id"] != b["run_id"], "F-08 阻断项：不同 lineage 同 attempt 又撞号"
    # 基础条件仍相同：两载荷除 lineage 外逐位相等
    la, lb = E.run_id_payload(ca, role="child"), E.run_id_payload(cb, role="child")
    assert {k: v for k, v in la.items() if k != "rerun_lineage"} == \
           {k: v for k, v in lb.items() if k != "rerun_lineage"}, \
        "重跑不该被误认成模型定义不同"
    _no_identity(_child(1, parent_ref=None))
    _no_identity(_child(1, parent_ref="n_samples=1000"))
    return "F-08 不同父工作流同 attempt 子号已分离；缺/假父引用无号；基础条件载荷不变"


def case_F09_reason_distinguishes_retries():
    fail = _ev(_child(1, reason="sampler_error"), role="child")
    ok = _ev(_child(1, reason="acf_gate_retry"), role="child")
    assert fail["status"] == ok["status"] == "recorded", (fail, ok)
    assert fail["run_id"] != ok["run_id"], "F-09：失败重试与门禁重试竟同号"
    for reason in (None, "", "   "):
        why = _no_identity(_child(1, reason=reason))
        assert any("rerun_reason_missing" in x for x in why), (reason, why)
    return "F-09 重跑原因进号：失败/成功重试不同号，空原因无号"


def case_F00_positive_path_intact():
    r = TE._rec()
    out = _ev(r)
    assert out["status"] == "recorded", out["reasons"]
    assert out["run_id"].startswith("R2-") and out["workflow_id"].startswith("W2-"), out
    assert E._chk_budget(r["budget_params"], r[E.BUDGET_EVIDENCE_FIELD]) == []
    assert E._chk_workflow(r["workflow"]) == []
    assert E._chk_rerun(r) == []
    return "F-00 修复集未误伤主路径：合规记录仍发 R2-/W2- 号"


CASES = [
    ("G1 正例：合规记录", case_G1_positive),
    ("G1① 裸 recorded 块", case_G1_bare_recorded_cells_rejected),
    ("G1② 缺证据块", case_G1_missing_evidence_block),
    ("G1③ 上游未捕获", case_G1_upstream_not_captured),
    ("G1④ 指纹篡改", case_G1_evidence_sha_tampered),
    ("G1⑤ origin=unknown", case_G1_origin_unknown),
    ("G1⑥ where 缺失", case_G1_where_missing),
    ("G2① 真实捕获六键", case_G2_real_capture_carries_six_key_block),
    ("G2② 适配输出不得发号", case_G2_adapted_output_cannot_get_identity),
    ("G2③ unavailable→recorded", case_G2_promote_unavailable_to_recorded_rejected),
    ("G2④ 外部自称闭合被拒", case_G2_external_upstream_claim_rejected),
    ("G2⑤ 词表拒绝映射", case_G2_origin_vocabulary_refuses_mapping),
    ("G2⑥ 三层算法对齐", case_G2_vocabularies_align_with_identity),
    ("G2⑦ 观测值不是来源", case_G2_observations_are_not_identity_sources),
    ("G3① retry_kw 含五键", case_G3_budget_keys_in_retry_kw_no_identity),
    ("G3② 策略键不换号", case_G3_strategy_only_retry_kw_no_id_change),
    ("G3③ 载荷移除整块", case_G3_retry_kw_removed_from_payload),
    ("G4① 同预算两次重跑", case_G4_same_budget_two_attempts_different_ids),
    ("G4② 换预算换号", case_G4_different_budget_different_id),
    ("G4③ 子记录不含父预算", case_G4_child_has_no_parent_budget_or_instance),
    ("G4④ children 只放引用", case_G4_children_only_hold_references),
    ("G5① 缺版本字段", case_G5_missing_keys_version_no_identity),
    ("G5② 版本值非法", case_G5_stale_and_malformed_keys_version),
    ("G5③ 两个载荷含版本", case_G5_version_in_both_payloads),
    ("G5④ 新世代不可比", case_G5_new_generation_not_comparable),
    ("G5⑤ 升版本必换号", case_G5_bumping_keys_version_changes_id),
    ("G5⑥ 身份层零生产 import", case_G5_identity_module_still_zero_production_import),
    ("F-00 主路径未误伤", case_F00_positive_path_intact),
    ("F-03 value 值域", case_F03_value_domain),
    ("F-04 event_sha hex", case_F04_event_sha_is_hex),
    ("F-05 捕获词表闭集", case_F05_capture_closed_sets),
    ("F-06 workflow 不承载预算", case_F06_workflow_no_budget_smuggling),
    ("F-07 children 只放引用", case_F07_children_reference_only),
    ("F-08 lineage 不撞号", case_F08_lineage_no_collision),
    ("F-09 重跑原因区分", case_F09_reason_distinguishes_retries),
]


def main() -> int:
    ok, bad = 0, []
    for name, fn in CASES:
        try:
            msg = fn()
            ok += 1
            print(f"[ok] {name}：{msg}")
        except AssertionError as exc:
            bad.append((name, str(exc)))
            print(f"[FAIL] {name}：{exc}")
        except BaseException as exc:                 # noqa: BLE001
            bad.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[ERR ] {name}：{type(exc).__name__}: {exc}")
    print(f"\n{ok}/{len(CASES)} 通过"
          + ("；未通过：" + ", ".join(n for n, _ in bad) if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
