# -*- coding: utf-8 -*-
"""test_budget_params.py —— ③ 第 2 步 `budget_params` v1.4 产出层正式测试。

测的是**裁决有没有被真正执行**，不是"能不能跑通"。九组 45 格：

· **C**（5）契约正例 + 与 v1.4 `exec_identity` 的真实交叉核验
· **S**（4）`source` 证据等级 / `origin` 来源类别的分离与哈希行为
· **T**（5）`requested` / `transformed` / `where` 追溯字段，列表与顺序
· **U**（5）上游闭合硬边界：`upstream_origin_captured` 必填、False 禁 recorded
· **H**（5）registry 0.7.0 历史兼容：块级 source 不复制、档名不进块、跨世代不比号
· **R**（4）Q5 `retry_kw` 分工
· **E**（5）Q6 三扩展字段分流
· **G**（4）整体证据状态 + `E.evaluate()` 的 `unconfirmed` 实际后果
· **X**（8）AST / JSON / 入参 / 常量对齐护栏

    python test_budget_params.py
"""
from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import budget_params as B                      # noqa: E402
import exec_identity as E                      # noqa: E402  （v1.4 校验器，只用于交叉核验）

VALUES = dict(zip(B.BUDGET_KEYS, (3000, 1000, 3, 0.9, 13)))


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def cells(*, source: str = "derived", origin: str = "module_constant",
          where: bool = True, **over) -> dict:
    out = {k: B.make_cell(value=v, source=source, origin=origin,
                          where=(f"bm_real.KW.{k}" if where else None))
           for k, v in VALUES.items()}
    out.update(over)
    return out


def block(*, upstream: bool = False, **over) -> dict:
    return B.validate_budget_params(cells(**over), upstream_origin_captured=upstream)


def recorded_block() -> dict:
    return B.validate_budget_params(
        cells(source="recorded"), upstream_origin_captured=True)


def raises(fn, *keys: str) -> str:
    try:
        fn()
    except ValueError as exc:
        msg = str(exc)
        miss = [k for k in keys if k not in msg]
        assert not miss, f"异常理由缺关键词 {miss}：{msg}"
        return msg
    raise AssertionError("本该 raise 却通过了")


def _src_const(name: str, path: str = "exec_identity.py"):
    tree = ast.parse((_ROOT / path).read_text(encoding="utf-8"))
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == name and value is not None:
            return ast.literal_eval(value)
    raise AssertionError(f"{path} 里找不到常量 {name}")


# ---------------------------------------------------------------------------
# C 契约正例 + v1.4 交叉核验
# ---------------------------------------------------------------------------

def case_C1_shape():
    b = block()
    assert list(b) == list(B.BUDGET_KEYS), list(b)
    assert all(list(c) == list(B.CELL_KEYS) for c in b.values())
    assert b["max_tree_depth"]["value"] == 13, "A 路径 --quick 后 depth 仍是 13，不许按档名写成 10"
    return f"五项齐全、每项六键、value 是最终构造实参（含 depth={b['max_tree_depth']['value']}）"


def case_C2_hash_payload_matches_v14():
    b = recorded_block()
    got = B.budget_hash_payload(b, upstream_origin_captured=True)
    want = E._budget_payload(b)
    assert got == want, f"与 v1.4 分叉：{got} vs {want}"
    assert all(set(v) == {"value", "source"} for v in got.values())
    return "逐位等于 exec_identity._budget_payload()，且每项只有 value/source"


def case_C3_no_second_budget_entry_in_config():
    dup = [k for k in B.BUDGET_KEYS if k in E.RUN_ID_CONFIG_KEYS]
    assert not dup, f"预算键在 config 里有第二入口：{dup}"
    return "五个预算键都不在 RUN_ID_CONFIG_KEYS，预算数值只有 budget_params 一个载体"


def case_C4_v14_chk_budget_all_recorded():
    b = recorded_block()
    evi = {"upstream_origin_captured": True,
           "capture_version": "1.0", "capture_site": "bm_core.fit",
           "evidence_sha256": E.budget_evidence_sha256(b),
           "event_sha256": "a" * 64}
    assert E._chk_budget(b, evi) == [], E._chk_budget(b, evi)
    # C1：v1.4 只要五项写着 recorded 就放行；v1.5 必须有显式证据块。
    assert E._chk_budget(b) != [], "缺显式证据块却放行 recorded（G1 回归）"
    return "全 recorded + 合规证据块 ⇒ v1.5 _chk_budget() 空；缺证据块 ⇒ 拒"


def case_C5_build_equals_validate():
    c = cells()
    a = B.build_budget_params(c, upstream_origin_captured=False)
    b = B.validate_budget_params(c, upstream_origin_captured=False)
    assert a == b
    return "build_budget_params 与 validate_budget_params 产出逐位相同（无第二条判定路径）"


# ---------------------------------------------------------------------------
# S source / origin 分离与哈希
# ---------------------------------------------------------------------------

def case_S1_origin_not_in_hash():
    b = recorded_block()
    h0 = B.budget_hash_payload(b, upstream_origin_captured=True)
    b2 = dict(b)
    b2["n_warmup"] = dict(b["n_warmup"], origin="library_default")
    h1 = B.budget_hash_payload(b2, upstream_origin_captured=True)
    assert h0 == h1
    return "同 value 改 origin ⇒ 预算哈希逐位不变（origin 是来源解释，不是执行条件）"


def case_S2_trace_not_in_hash():
    b = recorded_block()
    h0 = B.budget_hash_payload(b, upstream_origin_captured=True)
    b2 = dict(b)
    steps = [{"stage": "module_constant", "value": 3000, "where": "bm_real.KW"},
             {"stage": "quick_override", "value": 800, "where": "bm_real.run"},
             {"stage": "cli_ta_override", "value": 0.98, "where": "bmdrt._apply_cli"}]
    b2["n_warmup"] = dict(b["n_warmup"], requested=800, transformed=steps,
                          where="bm_core.fit:n_warmup")
    assert B.budget_hash_payload(b2, upstream_origin_captured=True) == h0
    return "改 requested/transformed/where 不换号：同一最终执行条件不因入口写法不同而换身份"


def case_S3_source_is_in_hash():
    b = recorded_block()
    h0 = B.budget_hash_payload(b, upstream_origin_captured=True)
    b2 = dict(b)
    b2["n_samples"] = dict(b["n_samples"], source="derived")
    h1 = B.budget_hash_payload(b2, upstream_origin_captured=True)
    assert h0 != h1
    return "同数值但证据等级不同 ⇒ 换号（source 表示同样数值有无同样证据）"


def case_S4_closed_sets():
    c = cells()
    raises(lambda: B.validate_budget_params(
        dict(c, n_warmup=dict(c["n_warmup"], source="derived_from_tier")),
        upstream_origin_captured=False), "source", "证据等级")
    raises(lambda: B.validate_budget_params(
        dict(c, n_warmup=dict(c["n_warmup"], origin="bm_real.KW")),
        upstream_origin_captured=False), "origin", "来源类别")
    return "source 拒 registry 的 derived_from_tier；origin 拒把具体位置塞进枚举"


# ---------------------------------------------------------------------------
# T 追溯字段
# ---------------------------------------------------------------------------

def case_T1_transformed_must_be_list():
    c = cells()
    for bad in (800, "quick", {"stage": "quick_override"}):
        raises(lambda bad=bad: B.validate_budget_params(
            dict(c, n_warmup=dict(c["n_warmup"], transformed=bad)),
            upstream_origin_captured=False), "列表")
    return "单值 / 字符串 / 单个映射三种写法都被拒：transformed 必须是列表"


def case_T2_order_preserved():
    c = cells()
    steps = [{"stage": "module_constant", "value": 3000, "where": "bm_real.KW"},
             {"stage": "quick_override", "value": 800, "where": "bm_real.run"},
             {"stage": "cli_ta_override", "value": 0.98, "where": "bmdrt._apply_cli"},
             {"stage": "function_default", "value": 600, "where": "bm_core.fit"}]
    b = B.validate_budget_params(
        dict(c, n_warmup=dict(c["n_warmup"], transformed=steps, value=600)),
        upstream_origin_captured=False)
    assert [s["stage"] for s in b["n_warmup"]["transformed"]] == [s["stage"] for s in steps]
    return "四次覆盖按原顺序保留，不被压缩成一个无法审计的中间值"


def case_T3_step_shape():
    c = cells()
    for bad in ({"value": 800}, {"stage": "", "value": 800},
                {"stage": "quick", "value": 800, "tier": "quick"}):
        raises(lambda bad=bad: B.validate_budget_params(
            dict(c, n_warmup=dict(c["n_warmup"], transformed=[bad])),
            upstream_origin_captured=False), "transformed")
    return "步骤缺 stage / 空 stage / 夹带 tier 都被拒"


def case_T4_requested_none_with_library_default():
    c = cells()
    b = B.validate_budget_params(
        dict(c, n_warmup=dict(c["n_warmup"], origin="library_default",
                              requested=None, where=None)),
        upstream_origin_captured=False)
    assert b["n_warmup"]["value"] == 3000 and b["n_warmup"]["requested"] is None
    assert b["n_warmup"]["origin"] == "library_default"
    return "requested=None 不代表最终值未知；结合 origin 才知道是库默认"


def case_T5_requested_no_mapping():
    c = cells()
    raises(lambda: B.validate_budget_params(
        dict(c, n_warmup=dict(c["n_warmup"], requested={"quick": True})),
        upstream_origin_captured=False), "requested")
    raises(lambda: B.validate_budget_params(
        dict(c, n_warmup=dict(c["n_warmup"], requested=float("nan"))),
        upstream_origin_captured=False), "requested")
    return "requested 不许塞整块配置、不许带 NaN"


# ---------------------------------------------------------------------------
# U 上游闭合硬边界
# ---------------------------------------------------------------------------

def case_U1_false_blocks_recorded():
    raises(lambda: B.validate_budget_params(
        cells(source="recorded"), upstream_origin_captured=False),
        "upstream_origin_captured", "recorded")
    return "上游链未闭合 ⇒ 五项最终值都采到也不许标 recorded"


def case_U2_true_positive_control():
    b = B.validate_budget_params(cells(source="recorded"),
                                 upstream_origin_captured=True)
    assert b["n_warmup"]["source"] == "recorded"
    return "同一批记录 + upstream=True 通过（正对照，证明 U1 不是一律拒绝 recorded）"


def case_U3_missing_flag_raises():
    raises(lambda: B.validate_budget_params(cells()), "显式传入")
    raises(lambda: B.budget_hash_payload(recorded_block()), "显式传入")
    raises(lambda: B.budget_evidence_state(recorded_block()), "显式传入")
    return "省略 upstream 一律 raise：不许用默认 False 把「忘了传」伪装成「已确认未闭合」"


def case_U4_flag_must_be_bool():
    for bad in (None, 1, 0, "False", [], object()):
        raises(lambda bad=bad: B.validate_budget_params(
            cells(), upstream_origin_captured=bad), "布尔")
    return "None / truthy / 容器都不是显式布尔，全部拒绝"


def case_U5_recorded_needs_per_item_evidence():
    c = cells(source="recorded")
    raises(lambda: B.validate_budget_params(
        dict(c, n_warmup=dict(c["n_warmup"], origin="unknown")),
        upstream_origin_captured=True), "origin")
    raises(lambda: B.validate_budget_params(
        dict(c, n_warmup=dict(c["n_warmup"], where=None)),
        upstream_origin_captured=True), "where")
    raises(lambda: B.validate_budget_params(
        dict(c, n_warmup=dict(c["n_warmup"], value=True)),
        upstream_origin_captured=True), "value")
    return "recorded 还要求逐项 origin≠unknown / where 非空 / value 非 bool"


# ---------------------------------------------------------------------------
# H registry 0.7.0 历史兼容
# ---------------------------------------------------------------------------

def _legacy():
    return {"source": "recorded", **VALUES}


def case_H1_detect_legacy():
    assert B.is_legacy_registry_block(_legacy())
    assert not B.is_legacy_registry_block(recorded_block())
    return "扁平 {source, 五裸数值} 被识别，v1.4 复合块不被误判"


def case_H2_legacy_rejected():
    raises(lambda: B.validate_budget_params(
        _legacy(), upstream_origin_captured=False), "块级", "registry")
    return "registry 合法记录直接喂 v1.4 ⇒ 明确拒绝，不做静默形状转换"


def case_H3_no_mechanical_copy():
    copied = {k: {"value": v, "source": "recorded"} for k, v in VALUES.items()}
    raises(lambda: B.build_budget_params(copied, upstream_origin_captured=True),
           "证据判断")
    return "块级 source 复制到五项（缺每项 origin/where）必须被拒：每项要有自己的证据判断"


def case_H4_tier_not_evidence():
    c = cells()
    for key in ("tier", "derived_from_tier", "budget_profile", "profile"):
        raises(lambda key=key: B.validate_budget_params(
            dict(c, **{key: "quick"}), upstream_origin_captured=False), "块级")
    return "档名 / tier / profile 一律不进预算身份，derived_from_tier 不作来源证据"


def case_H5_no_cross_generation_compare():
    a, b = "R-0123456789", "R1-" + "a" * 64
    cross = B.can_compare_ids(a, b)
    assert cross["comparable"] is False and "不同契约世代" in cross["reason"]
    assert cross["declares_equivalence"] is False
    assert B.can_compare_ids("R-0123456789", "D-9876543210")["comparable"] is True
    assert B.id_generation(a) == "registry_0.7.0"
    assert B.id_generation(b) == "exec_identity_1.4"
    return "R-/D-(10hex) 与 R1-/D1-(64hex) 不可直接比号；同世代可比（正对照）"


# ---------------------------------------------------------------------------
# R Q5 retry_kw
# ---------------------------------------------------------------------------

def case_R1_single_budget_key():
    raises(lambda: B.check_retry_kw({"n_samples": 2000, "retry": 2}), "n_samples", "Q5")
    return "retry_kw 含 n_samples ⇒ 拒绝：重跑改预算应产生新的子运行身份"


def case_R2_all_five_listed():
    msg = raises(lambda: B.check_retry_kw(dict(VALUES)), "五项预算键")
    assert all(k in msg for k in B.BUDGET_KEYS), msg
    return "五个键全在 ⇒ 理由里逐一点名，不静默吞掉"


def case_R3_policy_only_positive_control():
    r = B.check_retry_kw({"retry": 2, "trigger": "acf", "policy": "backoff"})
    assert r["budget_keys_found"] == [] and r["n_keys"] == 3
    assert B.check_retry_kw(None)["n_keys"] == 0
    return "只含重跑策略 / None 都合法（正对照）"


def case_R4_similar_names_not_false_positive():
    r = B.check_retry_kw({"num_warmup": 600, "n_samples_log": [1000, 2000]})
    assert r["n_keys"] == 2
    return "num_warmup / n_samples_log 不是 BUDGET_KEYS，不误判"


# ---------------------------------------------------------------------------
# E Q6 扩展字段
# ---------------------------------------------------------------------------

def case_E1_extension_not_in_budget():
    c = cells()
    for key in B.EXTENSION_FIELDS:
        raises(lambda key=key: B.validate_budget_params(
            dict(c, **{key: B.make_cell(value=1, source="derived",
                                        origin="library_default")}),
            upstream_origin_captured=False), "扩展字段")
    return "三扩展字段塞进 budget_params 一律拒绝（Q1：五项就是全部）"


def case_E2_three_channels():
    sp = B.split_extensions({"chain_method": "sequential", "progress_bar": False,
                             "extra_fields": ("diverging",)})
    assert sp["buckets"]["config.effective"] == {"chain_method": "sequential"}
    assert sp["buckets"]["sampler_capture_event"] == {"progress_bar": False}
    assert sp["buckets"]["mcmc_run_event"] == {"extra_fields": ("diverging",)}
    return "chain_method→config.effective / progress_bar→capture 事件 / extra_fields→run 事件"


def case_E3_chain_method_pending_not_claimed():
    sp = B.split_extensions({"chain_method": "sequential"})
    assert sp["run_id_fields"] == ["chain_method"]
    assert sp["pending_v14_registration"] == ["chain_method"]
    assert sp["warning"] is not None and "RUN_ID_CONFIG_KEYS" in sp["warning"]
    assert B.EXECUTION_CHANNELS["chain_method"]["registered_in_v14"] is False
    return "裁决是「暂时纳入执行条件」，当前 v1.4 未登记 ⇒ 单列待办，不谎称已生效"


def case_E4_two_non_identity_fields():
    sp = B.split_extensions({"progress_bar": False, "extra_fields": ("num_steps",)})
    assert set(sp["non_identity_fields"]) == {"progress_bar", "extra_fields"}
    assert sp["run_id_fields"] == [] and sp["warning"] is None
    return "progress_bar (display_only) / extra_fields (诊断采集) 都不进身份"


def case_E5_split_is_classification_only():
    sp = B.split_extensions({})
    assert sp["constructs_identity"] is False
    raises(lambda: B.split_extensions({"n_samples": 1000}), "不是三个扩展字段")
    return "只分流不构造号；收到预算键直接拒绝"


# ---------------------------------------------------------------------------
# G 整体证据状态 + v1.4 实际后果
# ---------------------------------------------------------------------------

def case_G1_all_recorded():
    st = B.budget_evidence_state(recorded_block(), upstream_origin_captured=True)
    assert st["state"] == "recorded" and st["supports_v14_recorded"] is True
    assert st["v14_consequence"] == "recorded"
    return "五项全 recorded 才支持整体执行身份进 recorded"


def case_G2_one_derived():
    c = cells(source="recorded")
    c["n_samples"] = dict(c["n_samples"], source="derived")
    b = B.validate_budget_params(c, upstream_origin_captured=True)
    st = B.budget_evidence_state(b, upstream_origin_captured=True)
    assert st["state"] == "derived" and st["weakest_keys"] == ["n_samples"]
    evi = {"upstream_origin_captured": True,
           "capture_version": "1.0", "capture_site": "bm_core.fit",
           "evidence_sha256": E.budget_evidence_sha256(b),
           "event_sha256": "a" * 64}
    assert E._chk_budget(b, evi) == ["budget_not_recorded:n_samples"], \
        E._chk_budget(b, evi)
    return "最弱项 derived；v1.5 _chk_budget()（带合规证据块）精确点名 n_samples"


def case_G3_one_unavailable():
    c = cells()
    c["max_tree_depth"] = dict(c["max_tree_depth"], value=None,
                               source="unavailable")
    b = B.validate_budget_params(c, upstream_origin_captured=False)
    st = B.budget_evidence_state(b, upstream_origin_captured=False)
    assert st["state"] == "unavailable" and st["v14_consequence"] == "unconfirmed:no_run_id"
    return "unavailable ⇒ 整体最弱 unavailable，value 必须为 None"


def case_G4_v14_evaluate_unconfirmed_not_derived():
    rec = {"identity_contract_version": E.IDENTITY_CONTRACT_VERSION,
           "candidate_id": "C-test", "budget_params": block()}
    out = E.evaluate(rec)
    assert out["status"] == "unconfirmed", out
    assert out["run_id"] is None and out["workflow_id"] is None
    assert any(r.startswith("budget_not_recorded:") for r in out["reasons"]), out["reasons"]
    return "真实调用 E.evaluate()：非全 recorded ⇒ unconfirmed **无号**，不是降 derived 发号"


# ---------------------------------------------------------------------------
# X 护栏
# ---------------------------------------------------------------------------

def case_X1_production_zero_reference():
    targets = [p for p in sorted(_ROOT.glob("*.py"))
               if p.name not in ("budget_params.py", "test_budget_params.py")
               and not p.name.startswith("test_")]
    hits = []
    for p in targets:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                hits += [(p.name, a.name) for a in node.names
                         if a.name == "budget_params"]
            elif isinstance(node, ast.ImportFrom) and node.module == "budget_params":
                hits.append((p.name, node.module))
    assert not hits, f"生产模块引用了产出层：{hits}"
    return f"{len(targets)} 个顶层 .py 的 AST 零 import（本层不接生产）"


def case_X2_standard_library_only():
    tree = ast.parse((_ROOT / "budget_params.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    allowed = {"__future__", "math", "typing", "collections", "sys"}
    assert mods <= allowed, mods - allowed
    return f"只 import {sorted(mods)}（sys 仅 __main__ 自检用），无 numpy / jax / numpyro / exec_identity"


def case_X3_inputs_not_mutated():
    c = cells()
    before = copy.deepcopy(c)
    B.validate_budget_params(c, upstream_origin_captured=False)
    B.build_budget_params(c, upstream_origin_captured=False)
    B.budget_hash_payload(c, upstream_origin_captured=False)
    assert c == before
    return "三个入口都不改入参"


def case_X4_json_safe():
    b = B.validate_budget_params(
        dict(cells(), n_warmup=dict(cells()["n_warmup"], requested=None,
                                    transformed=[{"stage": "module_constant",
                                                  "value": 3000}])),
        upstream_origin_captured=False)
    s = json.dumps(b, ensure_ascii=False, allow_nan=False, sort_keys=True)
    assert json.loads(s) == b
    assert json.dumps(b, ensure_ascii=False, allow_nan=False) == \
           json.dumps(b, ensure_ascii=False, allow_nan=False)
    return "allow_nan=False 直出可 JSON 往返，字段顺序稳定"


def case_X5_keys_match_exec_identity():
    assert B.BUDGET_KEYS == _src_const("BUDGET_KEYS") == E.BUDGET_KEYS
    assert B.BUDGET_KEYS == _src_const("BUDGET_PARAM_KEYS", "circuit_registry.py")
    return "五项与 exec_identity.BUDGET_KEYS、registry.BUDGET_PARAM_KEYS 逐位相同（AST 读源码）"


def case_X6_chain_method_registration_is_fact():
    keys = set(_src_const("RUN_ID_CONFIG_KEYS"))
    assert "chain_method" not in keys, "事实变了：须同步 registered_in_v14 并升版本"
    assert "progress" in keys and E.RUN_ID_CONFIG_KEYS["progress"][1] == "display_only"
    assert B.EXECUTION_CHANNELS["chain_method"]["registered_in_v14"] is False
    return "AST 实测 chain_method 未登记（progress=display_only）；标注与事实一致"


def case_X7_no_identity_construction():
    src = (_ROOT / "budget_params.py").read_text(encoding="utf-8")
    for banned in ("hashlib", "run_id_digest", "workflow_id_digest", "sha256("):
        assert banned not in src, f"产出层不该出现 {banned}"
    return "不 import hashlib、不调用任何号构造函数：本层不产 run_id / workflow_id"


def case_X8_closures_carry_no_locations():
    assert set(B.BUDGET_SOURCE_STATES) == {"recorded", "derived", "unavailable"}
    assert set(B.BUDGET_ORIGINS) == {
        "cli_explicit", "module_constant", "programmatic_override",
        "function_default", "library_default", "runtime_derived", "unknown"}
    assert all(("." not in s and "/" not in s and "tier" not in s)
               for s in B.BUDGET_SOURCE_STATES + B.BUDGET_ORIGINS)
    assert "where" in B.TRACE_KEYS and "where" not in B.BUDGET_ORIGINS
    return "两个闭集都不含路径/档名；具体位置只在 where，且 where 不进哈希"


CASES = [
    # C
    ("C1 形状与最终值", case_C1_shape),
    ("C2 与 v1.4 _budget_payload 逐位对齐", case_C2_hash_payload_matches_v14),
    ("C3 预算键无第二入口", case_C3_no_second_budget_entry_in_config),
    ("C4 v1.4 _chk_budget 全 recorded", case_C4_v14_chk_budget_all_recorded),
    ("C5 两入口无分叉", case_C5_build_equals_validate),
    # S
    ("S1 origin 不进哈希", case_S1_origin_not_in_hash),
    ("S2 追溯字段不进哈希", case_S2_trace_not_in_hash),
    ("S3 source 进哈希", case_S3_source_is_in_hash),
    ("S4 两个闭集", case_S4_closed_sets),
    # T
    ("T1 transformed 必须是列表", case_T1_transformed_must_be_list),
    ("T2 多次覆盖顺序保留", case_T2_order_preserved),
    ("T3 步骤形状", case_T3_step_shape),
    ("T4 requested=None 的含义", case_T4_requested_none_with_library_default),
    ("T5 requested 不许塞整块", case_T5_requested_no_mapping),
    # U
    ("U1 upstream=False 禁 recorded", case_U1_false_blocks_recorded),
    ("U2 upstream=True 正对照", case_U2_true_positive_control),
    ("U3 省略参数必 raise", case_U3_missing_flag_raises),
    ("U4 非布尔必 raise", case_U4_flag_must_be_bool),
    ("U5 recorded 的逐项证据", case_U5_recorded_needs_per_item_evidence),
    # H
    ("H1 识别 registry 扁平块", case_H1_detect_legacy),
    ("H2 扁平块不做静默转换", case_H2_legacy_rejected),
    ("H3 块级 source 不机械复制", case_H3_no_mechanical_copy),
    ("H4 档名不作来源证据", case_H4_tier_not_evidence),
    ("H5 跨世代不比号", case_H5_no_cross_generation_compare),
    # R
    ("R1 retry_kw 单个预算键", case_R1_single_budget_key),
    ("R2 五键全在逐一点名", case_R2_all_five_listed),
    ("R3 纯策略正对照", case_R3_policy_only_positive_control),
    ("R4 相似键不误判", case_R4_similar_names_not_false_positive),
    # E
    ("E1 扩展字段不进预算块", case_E1_extension_not_in_budget),
    ("E2 三通道分流", case_E2_three_channels),
    ("E3 chain_method 待登记", case_E3_chain_method_pending_not_claimed),
    ("E4 两个非身份字段", case_E4_two_non_identity_fields),
    ("E5 只分流不构造号", case_E5_split_is_classification_only),
    # G
    ("G1 全 recorded", case_G1_all_recorded),
    ("G2 一项 derived", case_G2_one_derived),
    ("G3 一项 unavailable", case_G3_one_unavailable),
    ("G4 v1.4 evaluate ⇒ unconfirmed 无号", case_G4_v14_evaluate_unconfirmed_not_derived),
    # X
    ("X1 生产零引用", case_X1_production_zero_reference),
    ("X2 只用标准库", case_X2_standard_library_only),
    ("X3 不改入参", case_X3_inputs_not_mutated),
    ("X4 JSON 直出", case_X4_json_safe),
    ("X5 常量与身份层对齐", case_X5_keys_match_exec_identity),
    ("X6 chain_method 登记事实", case_X6_chain_method_registration_is_fact),
    ("X7 不构造号", case_X7_no_identity_construction),
    ("X8 闭集不含位置", case_X8_closures_carry_no_locations),
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
    total = len(CASES)
    print(f"\n{ok}/{total} 通过"
          + ("；未通过：" + ", ".join(n for n, _ in bad) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
