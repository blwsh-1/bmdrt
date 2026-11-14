# -*- coding: utf-8 -*-
"""test_circuit_registry.py —— 阶段 0C：schema 契约测试（registry 0.2.0）。

目的：`circuit_registry.py` 的 schema 必须能**表达**下面这些事实，且不能把
不该合并的维度合并掉。这里测的是"能不能装下 + 乱填会不会报错"，不是模型好坏。

    A  定义完整性      —— 拓扑、元件接线、registry 指纹覆盖结构校验规则
    B  候选构造        —— 槽位/元件/初始化档/**数学结构一致性**真的在拦
    C  screening       —— only_survivor 只由 survivors==1 算；汇总必须能被逐条复算
    D  execution       —— 跑没跑成与失败原因分开记账
    E  observation     —— 执行 / 采样 / 观测三块分开；跑完但采样失败仍保留残差
    F  evidence 五态   —— identified / model_inadequate 的硬条件真的在拦
    G  L1 三态         —— 快速拟合失败**永远**不返回 excluded
    H  manifest        —— 拟合前冻结、不可追加/删除/改写、指纹可校验
    I  L0 预检         —— 绑定候选配置，真实调用 `voigt_can_represent`

用法：python test_circuit_registry.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import circuit_registry as R  # noqa: E402


def _expect_raise(fn, *a, **kw) -> str:
    try:
        fn(*a, **kw)
    except (ValueError, KeyError) as exc:
        return str(exc)
    raise AssertionError("期望报错但成功了")


def _voigt(K: int = 2, include_L: bool = False, **kw):
    els = [("relaxation_unit", "hn")] * K
    cfg = {"K": K, "include_L": include_L}
    if include_L:
        els.append(("series_inductor", "rl_series"))
    return R.make_candidate(
        "voigt", els, cfg, init_profile=kw.pop("init_profile", "uniform"), **kw)


def _nested(branch="capacitive", layers=2, init="uniform", **kw):
    els = [("outer_cpe", "cpe"), ("branch",
                                  {"parallel": "rl_parallel",
                                   "series": "rl_series",
                                   "capacitive": "cpe"}[branch])]
    if layers == 2:
        els.insert(1, ("inner_cpe", "cpe"))
    return R.make_candidate("nested", els, {"branch": branch, "layers": layers},
                            init_profile=init, **kw)


def _manifest(cs, **kw):
    return R.candidate_set_manifest(
        candidate_set_id=kw.pop("candidate_set_id", "CS-1"), candidates=cs,
        likelihood_profile=kw.pop("likelihood_profile", "perpoint-2pct"),
        budget_profile=kw.pop("budget_profile", "standard"),
        seed_policy=kw.pop("seed_policy", "seed=0..2"),
        created_before_fit=kw.pop("created_before_fit", True), **kw)


# --------------------------------------------------------------------------
# A 定义完整性
# --------------------------------------------------------------------------
def case_definitions():
    assert set(R.list_topologies()) == {"voigt", "nested"}, R.list_topologies()
    for tid in R.list_topologies():
        t = R.get_topology(tid)
        assert "Z" in t.skeleton and len(t.skeleton) > 40, tid
        assert t.param_space and t.priors, tid
        assert t.maturity in R.MATURITY
        # 每个拓扑都必须有显式的结构校验版本，否则 digest 发现不了规则变化
        assert t.consistency_id, tid
        assert t.consistency is not None, tid
    wired = set(R.list_elements(wired_only=True))
    assert {"hn", "cpe", "rl_parallel", "rl_series"} <= wired, wired
    for eid in set(R.list_elements()) - wired:
        assert R.get_element(eid).blocked_reason, eid
    assert R.registry_digest() == R.registry_digest(), "指纹不稳定"
    return f"{len(R.TOPOLOGIES)} 拓扑 / {len(R.ELEMENTS)} 元件，接线 {len(wired)}"


def case_unwired_needs_reason():
    _expect_raise(R.ElementSpec, id="x", name="x", forward_ref="bm_diff:z_winf",
                  params=("A",), maturity="frozen", wired=False,
                  blocked_reason="")
    return "wired=False 缺 blocked_reason 被拒"


def case_ref_whitelist():
    _expect_raise(R.TopologySpec, id="bad", name="bad",
                  skeleton="Z = 1 这是够长的骨架描述啊啊啊",
                  forward_ref="bogus_mod:f", param_space=("a",), derived=(),
                  priors={"a": "x"}, init_profiles={"uniform": "x"},
                  slots=(R.SlotSpec("s", "s", ("hn",)),), maturity="production")
    return "非法模块名被拒"


def case_digest_covers_consistency():
    old = R.TOPOLOGIES["nested"]
    before = R.registry_digest()
    R.TOPOLOGIES["nested"] = dataclasses.replace(
        old, consistency_id="nested_consistency_v99")
    try:
        after = R.registry_digest()
    finally:
        R.TOPOLOGIES["nested"] = old
    assert after != before, "改了 consistency_id 指纹却没变"
    assert R.registry_digest() == before, "恢复后指纹应回到原值"
    return "consistency_id 进指纹：改校验规则版本 → digest 变"


# --------------------------------------------------------------------------
# B 候选构造
# --------------------------------------------------------------------------
def case_candidate_ok():
    c1, c2 = _voigt(2), _nested("capacitive", 2)
    assert c1.topology_id == "voigt" and c2.topology_id == "nested"
    assert len([e for e in c1.elements if e[0] == "relaxation_unit"]) == 2
    assert dict(c2.elements)["branch"] == "cpe"
    return "voigt K=2 与 nested capacitive L=2 均可构造"


def case_candidate_id_covers_init():
    a = _nested(init="uniform")
    b = _nested(init="data_driven")
    assert a.candidate_id != b.candidate_id, "换初始化档必须是不同候选"
    c = _nested(init="uniform", budget_profile="quick")
    assert c.candidate_id != a.candidate_id, "换预算档必须是不同候选"
    return "candidate_id 覆盖 init/budget 档"


def case_branch_mismatch():
    _expect_raise(R.make_candidate, "nested",
                  [("outer_cpe", "cpe"), ("inner_cpe", "cpe"),
                   ("branch", "rl_parallel")],
                  {"branch": "capacitive", "layers": 2}, init_profile="uniform")
    return "branch 元件与 config 不一致被拒"


def case_layers1_with_inner():
    _expect_raise(R.make_candidate, "nested",
                  [("outer_cpe", "cpe"), ("inner_cpe", "cpe"),
                   ("branch", "cpe")],
                  {"branch": "capacitive", "layers": 1}, init_profile="uniform")
    return "layers=1 带 inner_cpe 被拒"


def case_layers3_rejected():
    """旧实现只分支 1 和 2，layers=3 会静默通过（z_nested 的 1..3 是笔误）。"""
    msg = _expect_raise(R.make_candidate, "nested",
                        [("outer_cpe", "cpe"), ("branch", "cpe")],
                        {"branch": "capacitive", "layers": 3},
                        init_profile="uniform")
    assert "1、2" in msg or "不支持" in msg, msg
    _expect_raise(R.make_candidate, "nested",
                  [("outer_cpe", "cpe"), ("branch", "cpe")],
                  {"branch": "capacitive"}, init_profile="uniform")
    return "layers=3 与缺 layers 都被拒"


def case_voigt_K_mismatch():
    """登记的是 K 个 HN，就必须真挂 K 个；否则 manifest 里的数学结构是假的。"""
    msg = _expect_raise(R.make_candidate, "voigt",
                        [("relaxation_unit", "hn")], {"K": 3},
                        init_profile="uniform")
    assert "不一致" in msg, msg
    _expect_raise(R.make_candidate, "voigt",
                  [("relaxation_unit", "hn")], {}, init_profile="uniform")
    return "voigt K≠HN 数、缺 K 都被拒"


def case_voigt_include_L_binding():
    _expect_raise(R.make_candidate, "voigt",
                  [("relaxation_unit", "hn")], {"K": 1, "include_L": True},
                  init_profile="uniform")
    _expect_raise(R.make_candidate, "voigt",
                  [("relaxation_unit", "hn"), ("series_inductor", "rl_series")],
                  {"K": 1, "include_L": False}, init_profile="uniform")
    c = _voigt(1, include_L=True)
    assert any(s == "series_inductor" for s, _ in c.elements)
    return "include_L 与 series_inductor 必须同侧"


def case_describe_keeps_duplicate_slots():
    d = _voigt(3).describe()
    assert isinstance(d["elements"], list), d["elements"]
    assert len([e for e in d["elements"] if e[0] == "relaxation_unit"]) == 3
    return "describe 用 list 保留 K 个重复槽位（dict 会压成一条）"


def case_unknown_init_profile():
    _expect_raise(R.make_candidate, "nested",
                  [("outer_cpe", "cpe"), ("branch", "cpe")],
                  {"branch": "capacitive", "layers": 1},
                  init_profile="不存在的档")
    return "未登记 init_profile 被拒"


def case_bad_slot():
    _expect_raise(R.make_candidate, "voigt",
                  [("relaxation_unit", "hn"), ("不存在的槽位", "hn")],
                  {"K": 1}, init_profile="uniform")
    return "未知槽位被拒"


# --------------------------------------------------------------------------
# C screening
# --------------------------------------------------------------------------
def case_screening_closes():
    s = R.screening_block(registered=6, excluded=2, not_excluded=3, unknown=1)
    assert s["entered_fit"] == 4 and s["survivors"] is None
    assert s["only_survivor"] is False
    _expect_raise(R.screening_block, registered=6, excluded=2,
                  not_excluded=3, unknown=0)
    return "6=2+3+1 闭合；不闭合被拒"


def case_only_survivor_is_count():
    s = R.screening_block(registered=6, excluded=2, not_excluded=3,
                          unknown=1, survivors=1)
    assert s["only_survivor"] is True
    s0 = R.screening_block(registered=6, excluded=2, not_excluded=3,
                           unknown=1, survivors=0)
    assert s0["only_survivor"] is False
    _expect_raise(R.screening_block, registered=6, excluded=2, not_excluded=3,
                  unknown=1, survivors=5)
    return "only_survivor 只由 survivors==1 得出；超界被拒"


def case_screening_records_must_match_tally():
    import numpy as np
    import bm_top as T
    f = np.logspace(-2, 5, 50)
    Z = 100.0 + T.z_rl_parallel(f, 1000.0, 10.0)
    cs = [_voigt(2), _nested("capacitive", 2)]
    p = R.precheck_candidates(cs, f, Z)
    s = R.screening_block(registered=p["registered"], excluded=p["excluded"],
                          not_excluded=p["not_excluded"], unknown=p["unknown"],
                          precheck_records=p["records"])
    assert len(s["precheck_records"]) == 2
    assert {r["candidate_id"] for r in s["precheck_records"]} == {c.candidate_id for c in cs}
    # 自述的汇总不能与逐条记录矛盾
    _expect_raise(R.screening_block, registered=2, excluded=0,
                  not_excluded=1, unknown=1, precheck_records=p["records"])
    # 记录条数必须等于 registered
    _expect_raise(R.screening_block, registered=3, excluded=1,
                  not_excluded=2, unknown=0, precheck_records=p["records"])
    # excluded 必须给理由
    _expect_raise(R.screening_block, registered=1, excluded=1,
                  not_excluded=0, unknown=0,
                  precheck_records=[{**p["records"][0], "precheck_reason": ""}])
    return "汇总必须由逐条预检记录复算得出"


# --------------------------------------------------------------------------
# D execution
# --------------------------------------------------------------------------
def case_execution_closes():
    e = R.execution_block(total=4, completed=1, failed=2, not_run=1,
                          failure_kinds={"init_collapse": 1, "sampling": 1})
    assert e["not_run"] == 1 and sum(e["failure_kinds"].values()) == 2
    _expect_raise(R.execution_block, total=4, completed=1, failed=1, not_run=1)
    _expect_raise(R.execution_block, total=2, completed=0, failed=1, not_run=1,
                  failure_kinds={"a": 2})
    return "执行层闭合；失败原因可细分"


# --------------------------------------------------------------------------
# E observation：执行 / 采样 / 观测三块
# --------------------------------------------------------------------------
def case_observation_sampling_failed_keeps_residuals():
    """P0-1 的核心用例：跑完了、残差算了、R̂ 没过——三者必须能同时表达。

    数值取真实历史产物（涂层 nested parallel：10.81% / acf 0.944 / R̂ 14.21）。
    """
    o = R.observation_block(
        candidate_id="C-coating-parallel", run_status="completed",
        sampling_status="failed", rms_rel=10.8101, acf=0.9444,
        rhat_max=14.2089, ess_min=1.0042, divergences=0,
        interpretation="descriptive_only")
    fo = o["fit_observation"]
    assert o["run_status"] == "completed" and o["sampling_status"] == "failed"
    assert abs(fo["rms_rel"] - 10.8101) < 1e-9 and abs(fo["acf"] - 0.9444) < 1e-9
    assert fo["sampling_diagnostics"]["rhat_max"] == 14.2089
    assert fo["interpretation"] == "descriptive_only"
    return "跑完但采样失败：残差保留且强制 descriptive_only"


def case_observation_sampling_failed_needs_explicit_interp():
    msg = _expect_raise(R.observation_block, candidate_id="C-x",
                        run_status="completed", sampling_status="failed",
                        rms_rel=5.0, acf=0.4)
    assert "descriptive_only" in msg, msg
    msg2 = _expect_raise(R.observation_block, candidate_id="C-x",
                         run_status="completed", sampling_status="not_evaluated",
                         rms_rel=5.0, acf=0.4)
    assert "descriptive_only" in msg2, msg2
    return "采样未通过/未诊断时不给解读 → 报错，不给默认值"


def case_observation_passed_defaults_reliable():
    o = R.observation_block(candidate_id="C-ok", run_status="completed",
                            sampling_status="passed", rms_rel=7.0765,
                            acf=0.2794, rhat_max=1.0047)
    assert o["fit_observation"]["interpretation"] == "reliable"
    return "采样通过时默认 reliable"


def case_observation_no_numbers_when_not_completed():
    """E3 的语义修正版：只有"根本没跑完"才不许报残差。"""
    msg = _expect_raise(R.observation_block, candidate_id="C-x",
                        run_status="failed", rms_rel=5.0, acf=0.4)
    assert "没跑完" in msg, msg
    o = R.observation_block(candidate_id="C-y", run_status="not_run")
    assert o["fit_observation"]["rms_rel"] is None
    assert o["fit_observation"]["interpretation"] is None
    return "run_status≠completed 不报数（不再用 converged 布尔）"


def case_observation_null_reason():
    o = R.observation_block(
        candidate_id="C", run_status="completed", sampling_status="passed",
        rms_rel=7.0, acf=None, unavailable={"acf": "not_computed"})
    assert o["fit_observation"]["unavailable"]["acf"] == "not_computed"
    _expect_raise(R.observation_block, candidate_id="C", run_status="completed",
                  sampling_status="passed", rms_rel=7.0, acf=None,
                  unavailable={"acf": "没算"})
    _expect_raise(R.observation_block, candidate_id="C", run_status="completed",
                  sampling_status="passed", rms_rel=7.0, acf=-0.2,
                  unavailable={"acf": "failed"})
    _expect_raise(R.observation_block, candidate_id="C", run_status="completed",
                  sampling_status="passed", rms_rel=7.0, acf=None,
                  unavailable={"不存在的量": "not_applicable"})
    return "null 三语义（不适用/没算/失败）可表达；与有值互斥"


def case_observation_silent_absence_rejected():
    _expect_raise(R.observation_block, candidate_id="C",
                  run_status="completed", sampling_status="passed", rms_rel=7.0)
    return "已跑完但 acf 既无值也无原因 → 报错"


def case_observation_acf_can_be_negative():
    o = R.observation_block(candidate_id="C-neg", run_status="completed",
                            sampling_status="passed", rms_rel=6.65, acf=-0.244)
    assert o["fit_observation"]["acf"] == -0.244
    return "acf 允许负值"


def case_observation_floor_but_unstable():
    o = R.observation_block(
        candidate_id="C-demo", run_status="completed", sampling_status="passed",
        rms_rel=7.0765, acf=0.2794, rhat_max=1.0047,
        param_stability={
            "R2": {"median": 1.31e5, "ci90": [0.9e5, 1.9e5], "flag": "wide"},
            "n2": {"median": 0.899, "flag": "at_bound"},
        })
    assert o["fit_observation"]["rms_rel"] == 7.0765
    assert o["param_stability"]["n2"]["flag"] == "at_bound"
    return "贴底（rms_rel 7.08%）与参数不稳（n2 at_bound）可同时保留"


def case_observation_no_scalar_score():
    _expect_raise(R.observation_block, candidate_id="C-z", run_status="completed",
                  sampling_status="passed", rms_rel=5.0, acf=0.1,
                  param_stability={"n2": 0.42})
    _expect_raise(R.observation_block, candidate_id="C-z", run_status="completed",
                  sampling_status="passed", rms_rel=5.0, acf=0.1,
                  param_stability={"n2": {"median": 0.5}})
    return "参数可信度不许压成标量"


def case_sampling_status_from():
    assert R.sampling_status_from(run_status="not_run", rhat_max=None) == "not_evaluated"
    assert R.sampling_status_from(run_status="completed", rhat_max=14.21) == "failed"
    assert R.sampling_status_from(run_status="completed", rhat_max=1.0047) == "passed"
    assert R.sampling_status_from(run_status="completed", rhat_max=1.01,
                                  ess_min=0.9) == "failed"
    return "R̂/ESS 折算采样态（只折算，不出结构结论）"


# --------------------------------------------------------------------------
# F evidence 五态
# --------------------------------------------------------------------------
def case_evidence_insufficient_default():
    e = R.evidence_block(status="insufficient_evidence", n_viable=1)
    assert e["scope"] == R.EVIDENCE_SCOPE and e["status"] == "insufficient_evidence"
    return "真实谱单候选 → insufficient_evidence（不是 non_identifiable）"


def case_evidence_non_identifiable_needs_two():
    _expect_raise(R.evidence_block, status="non_identifiable", n_viable=1)
    _expect_raise(R.evidence_block, status="weak_preference", n_viable=1)
    e = R.evidence_block(status="non_identifiable", n_viable=2)
    assert e["n_viable"] == 2
    return "不可区分/弱倾向都要求 >=2 个 viable"


def case_evidence_identified_blocked():
    msg = _expect_raise(R.evidence_block, status="identified", n_viable=2)
    assert "样本外" in msg
    _expect_raise(R.evidence_block, status="identified", n_viable=1,
                  out_of_sample_valid=True)
    e = R.evidence_block(status="identified", n_viable=2,
                         out_of_sample_valid=True)
    assert e["out_of_sample_valid"] is True
    return "identified 无有效样本外证据被拒；两个条件齐了才放行"


def case_evidence_inadequate_needs_truth():
    msg = _expect_raise(R.evidence_block, status="model_inadequate", n_viable=0)
    assert "真实谱" in msg
    e = R.evidence_block(status="model_inadequate", n_viable=0, truth_known=True)
    assert e["truth_known"] is True
    e2 = R.evidence_block(status="model_inadequate", n_viable=0,
                          noise_calibrated=True)
    assert e2["noise_calibrated"] is True
    return "model_inadequate 需真值或标定噪声"


def case_only_survivor_not_evidence_state():
    msg = _expect_raise(R.evidence_block, status="only_survivor", n_viable=1)
    assert "only_survivor" in msg
    return "only_survivor 不能作为证据状态"


# --------------------------------------------------------------------------
# G L1 三态
# --------------------------------------------------------------------------
def case_l1_never_excludes():
    for kw in ({"converged": False, "init_explored": True},
               {"converged": False, "init_explored": False},
               {"converged": True, "init_explored": True},
               {"converged": True, "init_explored": True,
                "budget_sufficient": False}):
        out = R.l1_outcome(**kw)
        assert out in R.L1_OUTCOMES, out
        assert out != "excluded"
    return "L1 四种组合都不返回 excluded"


def case_l1_init_not_explored_defers():
    assert R.l1_outcome(converged=False, init_explored=False) == "defer"
    assert R.l1_outcome(converged=False, init_explored=True) == "defer"
    assert R.l1_outcome(converged=True, init_explored=True) == "promote"
    assert R.l1_outcome(converged=True, init_explored=True,
                        budget_sufficient=False) == "not_evaluated_budget"
    return "初始未探索 → defer（不是排除）"


# --------------------------------------------------------------------------
# H manifest
# --------------------------------------------------------------------------
def case_manifest_frozen():
    cs = [_voigt(2), _nested("capacitive", 2), _nested("parallel", 1)]
    m = _manifest(cs)
    for k in ("candidate_set_id", "candidate_ids", "candidate_definitions",
              "registry_version", "precheck_version", "likelihood_profile",
              "budget_profile", "seed_policy", "created_before_fit",
              "manifest_digest"):
        assert k in m, k
    assert len(m["candidate_ids"]) == 3 and m["immutable"] is True
    json.dumps(m, ensure_ascii=False)
    _expect_raise(_manifest, cs, created_before_fit=False)
    return "10 字段齐（含指纹）+ 可序列化；created_before_fit=False 被拒"


def case_manifest_no_append():
    cs = [_voigt(2)]
    m = _manifest(cs)
    R.assert_no_append(m, cs)
    msg = _expect_raise(R.assert_no_append, m, [_voigt(2), _nested("capacitive", 2)])
    assert "新的 candidate_set_id" in msg
    return "冻结后追加被拒，提示新建候选集"


def case_manifest_duplicate():
    cs = [_voigt(2), _voigt(2)]
    _expect_raise(_manifest, cs)
    return "重复 candidate_id 被拒"


def case_verify_manifest_ok_and_tamper():
    cs = [_voigt(2), _nested("capacitive", 2)]
    m = _manifest(cs)
    R.verify_manifest(m, cs)

    bad = dict(m)
    bad["budget_profile"] = "quick"
    msg = _expect_raise(R.verify_manifest, bad, cs)
    assert "指纹" in msg, msg

    no_digest = {k: v for k, v in m.items() if k != "manifest_digest"}
    _expect_raise(R.verify_manifest, no_digest, cs)
    return "verify_manifest 通过；改内容/缺指纹都被抓"


def case_verify_manifest_rejects_removal_and_swap():
    cs = [_voigt(2), _nested("capacitive", 2)]
    m = _manifest(cs)
    msg = _expect_raise(R.verify_manifest, m, [_voigt(2)])
    assert "缺少" in msg, msg

    # 同名换定义：候选集里用不同 init 的候选，但把 manifest 的定义改成另一个
    a = _nested("capacitive", 2, init="uniform")
    b = _nested("capacitive", 2, init="data_driven")
    m2 = _manifest([a])
    forged = dict(m2)
    forged["candidate_definitions"] = [
        {**m2["candidate_definitions"][0], "init_profile": "data_driven"}]
    forged["manifest_digest"] = R.manifest_digest(forged)
    msg2 = _expect_raise(R.verify_manifest, forged, [a])
    assert "同名换定义" in msg2, msg2
    assert b.candidate_id != a.candidate_id
    return "删除候选、同名换定义都被抓"


def case_verify_manifest_registry_drift():
    cs = [_voigt(2)]
    m = _manifest(cs)
    old = R.TOPOLOGIES["voigt"]
    R.TOPOLOGIES["voigt"] = dataclasses.replace(old, maturity="exploratory")
    try:
        msg = _expect_raise(R.verify_manifest, m, cs)
    finally:
        R.TOPOLOGIES["voigt"] = old
    assert "registry_digest" in msg, msg
    return "注册表结构变化 → 历史候选集不可直接续用"


# --------------------------------------------------------------------------
# I L0 预检（真实调用，绑定候选配置）
# --------------------------------------------------------------------------
def case_l0_az31_style_excluded():
    import numpy as np
    import bm_top as T
    f = np.logspace(-2, 5, 50)
    Z = 100.0 + T.z_rl_parallel(f, 1000.0, 10.0)
    rec = R.l0_precheck(_voigt(2), f, Z)
    assert rec["precheck_state"] == "excluded", rec
    assert "低频感抗" in rec["precheck_reason"]
    assert rec["precheck_config"]["include_L"] is False
    assert rec["precheck_version"] == R.PRECHECK_VERSION
    return "低频感抗谱 → voigt excluded（记录含配置与理由）"


def case_l0_capacitive_not_excluded():
    import numpy as np
    import bm_top as T
    f = np.logspace(0, 4.4, 60)
    p = dict(Rs=141.2, R1=9.28e6, Q1=6.62e4, n1=0.782,
             Q_b=3.30e5, n_b=0.5002)
    Z = T.z_nested(f, p, branch="capacitive", layers=1)
    rec = R.l0_precheck(_voigt(2), f, Z)
    assert rec["precheck_state"] == "not_excluded", rec
    assert rec["precheck_reason"] == ""
    return "容性嵌套谱 → voigt not_excluded（预检覆盖不到，已知的洞）"


def case_l0_nested_unknown():
    import numpy as np
    f = np.logspace(0, 4, 30)
    rec = R.l0_precheck(_nested("capacitive", 1), f, 1.0 + 0j)
    assert rec["precheck_state"] == "unknown" and rec["precheck_reason"]
    return "nested 未登记预检 → unknown"


def case_l0_binds_candidate_config():
    """同一 voigt，include_L 不同必须能产生不同记录；旧签名靠关键字传，容易漏。"""
    import numpy as np
    import bm_top as T
    f = np.logspace(-2, 5, 50)
    Z = 100.0 + T.z_rl_parallel(f, 1000.0, 10.0)
    a = R.l0_precheck(_voigt(2, include_L=False), f, Z)
    b = R.l0_precheck(_voigt(2, include_L=True), f, Z)
    assert a["candidate_id"] != b["candidate_id"]
    assert a["precheck_config"]["include_L"] is False
    assert b["precheck_config"]["include_L"] is True
    msg = _expect_raise(R.l0_precheck, "voigt", f, Z)
    assert "Candidate" in msg, msg
    return "预检绑定候选；旧签名（拓扑名+关键字）被拒"


# ---------------------------------------------------------------- J 身份等级
def case_init_unknown_is_not_a_real_profile():
    c = _voigt(2, init_profile=R.INIT_UNKNOWN)
    assert c.candidate_id.startswith("U-"), c.candidate_id
    assert c.identity_status == "unconfirmed"
    assert c.comparable is False
    assert c.describe()["comparable"] is False
    msg = _expect_raise(R.candidate_set_manifest, candidate_set_id="x",
                        candidates=[c], likelihood_profile="p",
                        budget_profile="b", seed_policy="s", created_before_fit=True)
    assert "身份不可复现" in msg, msg
    return "init 未知 → U- 前缀 + unconfirmed + 不可进 manifest"


def case_unknown_init_cannot_claim_recorded():
    msg = _expect_raise(_voigt, 2, init_profile=R.INIT_UNKNOWN,
                        identity_status="recorded")
    assert "矛盾" in msg, msg
    return "init=unknown 却声称 recorded 被拒"


def case_unregistered_init_rejected():
    msg = _expect_raise(_voigt, 2, init_profile="nested_fit_default")
    assert "未登记 init_profile" in msg, msg
    return "没登记的档名（含近似名）被拒，只能传 INIT_UNKNOWN"


def case_identity_levels():
    c = _voigt(2, identity_status="derived", init_source="producer_code:x")
    d = c.describe()
    assert d["identity_status"] == "derived" and d["comparable"] is True
    assert d["init_source"] == "producer_code:x"
    msg = _expect_raise(_voigt, 2, identity_status="guessed")
    assert "identity_status" in msg, msg
    return "describe 带 identity_status/init_source/comparable；非法等级被拒"


def case_voigt_has_data_driven_profile():
    topo = R.get_topology("voigt")
    assert "data_driven" in topo.init_profiles, list(topo.init_profiles)
    c = _voigt(2, init_profile="data_driven")
    assert c.candidate_id != _voigt(2, init_profile="uniform").candidate_id
    return "voigt 登记 data_driven（pilot 的 voigt_init），与 uniform 是不同 id"


# ------------------------------------------------------- K 六个描述块
def case_budget_profile_is_not_params():
    msg = _expect_raise(R.observation_block, candidate_id="C-1",
                        run_status="completed", sampling_status="passed",
                        rms_rel=1.0, acf=0.1, budget_params="standard")
    assert "档名" in msg, msg
    return "budget_params 收到档名字符串被拒（档名≠数值）"


def case_budget_params_recorded_needs_all():
    msg = _expect_raise(R.observation_block, candidate_id="C-1",
                        run_status="completed", sampling_status="passed",
                        rms_rel=1.0, acf=0.1,
                        budget_params={"source": "recorded", "n_warmup": 2000})
    assert "齐全" in msg, msg
    ok = R.observation_block(candidate_id="C-1", run_status="completed",
                             sampling_status="passed", rms_rel=1.0, acf=0.1,
                             budget_params={"source": "derived_from_tier",
                                            "tier": "quick", "n_warmup": 800,
                                            "target_accept": 0.9})
    assert ok["budget_params"]["target_accept"] == 0.9
    return "recorded 缺项被拒；derived_from_tier 可部分，且保留实际 0.9"


def case_ratio_needs_denominator():
    msg = _expect_raise(R.observation_block, candidate_id="C-1",
                        run_status="completed", sampling_status="passed",
                        rms_rel=1.0, acf=0.1,
                        ratio={"value": 3.36, "n_total": 60,
                               "noise_floor_source": "truth_forward"})
    assert "n_used" in msg, msg
    b = R.observation_block(candidate_id="C-1", run_status="completed",
                            sampling_status="passed", rms_rel=1.0, acf=0.1,
                            ratio={"value": 3.36, "n_used": 60, "n_total": 60,
                                   "n_usable": 14,
                                   "noise_floor_source": "truth_forward"})
    r = b["ratio"]
    assert r["denominator_scope"] == "converged_subset" and r["n_usable"] == 14
    return "ratio 缺 n_used 被拒；n_usable<n_total → converged_subset"


def case_ratio_scope_all_runs():
    b = R.observation_block(candidate_id="C-1", run_status="completed",
                            sampling_status="passed", rms_rel=1.0, acf=0.1,
                            ratio={"value": 0.94, "n_used": 60, "n_total": 60,
                                   "noise_floor_source": "truth_forward"})
    assert b["ratio"]["denominator_scope"] == "all_runs"
    assert b["ratio"]["n_usable"] == 60
    return "n_usable 缺省=n_used；60/60 → all_runs"


def case_noise_floor_cannot_fake_source():
    msg = _expect_raise(R.observation_block, candidate_id="C-1",
                        run_status="completed", sampling_status="passed",
                        rms_rel=1.0, acf=0.1,
                        noise_floor={"value": 6.7, "source": "guess"})
    assert "NOISE_FLOOR_SOURCES" in msg or "source" in msg, msg
    msg = _expect_raise(R.observation_block, candidate_id="C-1",
                        run_status="completed", sampling_status="passed",
                        rms_rel=1.0, acf=0.1,
                        noise_floor={"value": 6.7, "source": "not_available"})
    assert "value" in msg, msg
    b = R.observation_block(candidate_id="C-1", run_status="completed",
                            sampling_status="passed", rms_rel=1.0, acf=0.1,
                            noise_floor={"value": None, "source": "not_available"})
    assert b["noise_floor"]["value"] is None
    return "底来源非法/自相矛盾被拒；真实谱 not_available 合法"


def case_param_estimate_truth_gate():
    msg = _expect_raise(R.observation_block, candidate_id="C-1",
                        run_status="completed", sampling_status="passed",
                        rms_rel=1.0, acf=0.1, truth_known=False,
                        param_estimate={"Rs": {"estimate": 267.0, "truth": 150.0,
                                               "rel_err": 0.78}})
    assert "真实谱没有真值" in msg, msg
    b = R.observation_block(candidate_id="C-1", run_status="completed",
                            sampling_status="passed", rms_rel=1.0, acf=0.1,
                            truth_known=True,
                            param_estimate={"Rs": {"rel_err": 2.65}})
    assert b["param_estimate"]["Rs"]["rel_err"] == 2.65
    assert "truth" not in b["param_estimate"]["Rs"]
    return "真实谱带真值被拒；真有真值时只记 rel_err 也合法"


def case_predictive_validity_must_be_explicit():
    msg = _expect_raise(R.observation_block, candidate_id="C-1",
                        run_status="completed", sampling_status="passed",
                        rms_rel=1.0, acf=0.1,
                        predictive={"elpd": -1898.5, "se": 3.0})
    assert "布尔" in msg, msg
    msg = _expect_raise(R.observation_block, candidate_id="C-1",
                        run_status="completed", sampling_status="passed",
                        rms_rel=1.0, acf=0.1,
                        predictive={"elpd": -1898.5, "valid_for_comparison": False})
    assert "reason" in msg, msg
    b = R.observation_block(candidate_id="C-1", run_status="completed",
                            sampling_status="passed", rms_rel=1.0, acf=0.1,
                            predictive={"elpd": -1898.5, "valid_for_comparison": False,
                                        "reason": "逐点 σ 不是留一预测"})
    assert b["predictive"]["valid_for_comparison"] is False
    assert b["predictive"]["reason"]
    return "可用性必须显式；False 必须带原因"


_BP = dict(source="recorded", n_warmup=2000, n_samples=1000, num_chains=2,
           target_accept=0.98, max_tree_depth=10)


_SHA = "a" * 64
_SHA2 = "b" * 64


def _exec(**kw):
    """执行身份 fixture。默认：同一条谱（带内容哈希）、seed 由产物记录、预算 recorded、
    代码版本未知 ⇒ `derived`。"""
    kw.setdefault("candidate_id", _nested().candidate_id)
    kw.setdefault("data", {"ref": "coating/R1-[P2,R3-[P4,R5-P6]]", "sha256": _SHA,
                          "f_min": 0.1, "f_max": 1e5})
    kw.setdefault("seed", 0)
    kw.setdefault("seed_source", "artifact:config.seed")
    kw.setdefault("budget_params", dict(_BP))
    kw.setdefault("software", {"producer": "nested_fit.py"})
    return R.execution_identity(**kw)


def case_run_id_separates_budget_values():
    """P1：同档名、不同预算数值 ⇒ candidate_id 相同、run_id 不同。"""
    c = _voigt(budget_profile="standard")
    a = _exec(candidate_id=c.candidate_id, budget_params=dict(_BP))
    bp2 = dict(_BP, target_accept=0.90)          # BUDGETS["standard"] 的真实值
    b = _exec(candidate_id=c.candidate_id, budget_params=bp2)
    assert a["run_id"] != b["run_id"], (a["run_id"], b["run_id"])
    assert c.candidate_id == _voigt(budget_profile="standard").candidate_id
    o1 = R.observation_block(candidate_id=c.candidate_id, run_status="completed",
                             sampling_status="passed", rms_rel=7.0, acf=0.2,
                             budget_params=dict(_BP), execution=a)
    o2 = R.observation_block(candidate_id=c.candidate_id, run_status="completed",
                             sampling_status="passed", rms_rel=7.0, acf=0.2,
                             budget_params=bp2, execution=b)
    assert o1["candidate_id"] == o2["candidate_id"]
    assert o1["execution"]["run_id"] != o2["execution"]["run_id"]
    return "同 candidate_id / 不同 run_id：预算数值进执行身份，不进模型身份"


def case_run_id_refuses_default_seed():
    e = _exec(seed=None, seed_source="absent")
    assert e["run_id"] is None and e["run_id_status"] == "unconfirmed"
    assert e["unavailable_reason"] == "seed_not_recorded"
    b = R.observation_block(candidate_id=_nested().candidate_id, run_status="completed",
                            sampling_status="passed", rms_rel=7.0, acf=0.2,
                            budget_params=dict(_BP), execution=e)
    assert b["execution"]["run_id"] is None
    msg = _expect_raise(R.execution_identity, candidate_id=_nested().candidate_id,
                        seed=None, seed_source="producer_code:x", budget_params=dict(_BP))
    assert "absent" in msg, msg
    return "无 seed 不发 run_id（带原因）；seed=None 却标来源矛盾被拒"


def case_run_id_status_ladder():
    e = _exec()
    assert e["run_id_status"] == "derived" and e["run_id"].startswith("D-"), e
    e2 = _exec(software={"producer": "nested_fit.py", "code_version": "a1b2c3d"})
    assert e2["run_id_status"] == "recorded" and e2["run_id"].startswith("R-"), e2
    e3 = _exec(seed_source="producer_code:bm_real._resolve_seed")
    assert e3["run_id_status"] == "derived" and e3["run_id"] != e["run_id"], e3
    e4 = _exec(budget_params=dict(source="derived_from_tier", tier="quick",
                                  n_warmup=800, n_samples=400, num_chains=2,
                                  target_accept=0.9, max_tree_depth=10))
    assert e4["run_id_status"] == "derived", e4
    assert e4["run_id"] != e["run_id"]
    e5 = _exec(budget_params={"source": "unavailable", "reason": "not_computed"})
    assert e5["run_id"] is None and e5["unavailable_reason"] == "budget_not_recorded"
    return "recorded 需 seed+预算+代码版本三者齐；seed 来源不同 ⇒ id 不同；预算缺 ⇒ 不发"


def case_run_id_rejects_unknown_candidate():
    msg = _expect_raise(R.execution_identity, candidate_id="U-abc123",
                        seed=0, seed_source="artifact:x", budget_params=dict(_BP))
    assert "U-" in msg, msg
    return "候选身份未确认（U-）时不许发执行身份"


def case_observation_recomputes_run_id():
    c = _nested()
    e = _exec(candidate_id=c.candidate_id)
    msg = _expect_raise(R.observation_block, candidate_id=c.candidate_id,
                        run_status="completed", sampling_status="passed",
                        rms_rel=7.0, acf=0.2,
                        budget_params=dict(_BP, n_warmup=800), execution=e)
    assert "不符" in msg, msg
    msg = _expect_raise(R.observation_block, candidate_id=c.candidate_id,
                        run_status="completed", sampling_status="passed",
                        rms_rel=7.0, acf=0.2, execution=e)
    assert "budget_params" in msg, msg
    bad = dict(e, run_id="R-0000000000")
    msg = _expect_raise(R.observation_block, candidate_id=c.candidate_id,
                        run_status="completed", sampling_status="passed",
                        rms_rel=7.0, acf=0.2, budget_params=dict(_BP), execution=bad)
    assert "状态" in msg or "不符" in msg, msg
    return "run_id 与观测预算重算不符被拒；有 id 无预算被拒；前缀与状态不符被拒"


def case_duplicate_needs_provenance():
    """P0：摘要相同**不**等于重复；证据不足只标 possible，不从分母里扣。"""
    c = _nested()
    a = {"candidate_id": c.candidate_id, "source": {"path": "x/a.json", "sha256": "s1"},
         "execution": _exec(candidate_id=c.candidate_id)}
    b = {"candidate_id": c.candidate_id, "source": {"path": "x/b.json", "sha256": "s2"},
         "execution": _exec(candidate_id=c.candidate_id,
                            seed_source="producer_code:bm_real._resolve_seed")}
    v = R.duplicate_verdict(a, b, summary_equal=True)
    assert v["duplicate_status"] == "possible_duplicate", v
    assert v["independent"] is True, v
    assert v["evidence"]["summary_equal"] is True
    assert v["evidence"]["conditions_observed"] is False
    return "seed 靠代码推出 ⇒ 只标 possible_duplicate，仍计入独立样本"


def case_duplicate_confirmed_with_recorded_seed():
    """确认重复需要**完整**的 `recorded` 执行身份（含代码版本与数据内容哈希）。"""
    c = _nested()
    ver = {"producer": "nested_fit.py", "code_version": "a1b2c3d"}
    a = {"candidate_id": c.candidate_id, "source": {"path": "x/a.json", "sha256": "s1"},
         "execution": _exec(candidate_id=c.candidate_id, software=ver)}
    b = {"candidate_id": c.candidate_id, "source": {"path": "x/b.json", "sha256": "s2"},
         "execution": _exec(candidate_id=c.candidate_id, software=ver)}
    v = R.duplicate_verdict(a, b, summary_equal=True)
    assert v["duplicate_status"] == "confirmed_duplicate", v
    assert v["duplicate_kind"] == "rerun_same_conditions", v
    assert v["independent"] is False, v
    # 同产物被登记两次
    v2 = R.duplicate_verdict(a, dict(b, source=dict(a["source"])), summary_equal=True)
    assert v2["duplicate_kind"] == "same_artifact", v2
    # 同 seed 同条件但结果不同 ⇒ 不是重复，是可复现性问题
    v3 = R.duplicate_verdict(a, b, summary_equal=False)
    assert v3["duplicate_status"] == "distinct", v3
    assert v3["evidence"].get("contradiction") is True, v3
    # 不同候选即使摘要相同也不构成重复
    other = {"candidate_id": _nested(branch="parallel").candidate_id,
             "source": {"path": "x/c.json", "sha256": "s3"},
             "execution": _exec(candidate_id=_nested(branch="parallel").candidate_id,
                                software=ver)}
    v4 = R.duplicate_verdict(a, other, summary_equal=True)
    assert v4["duplicate_status"] == "distinct", v4
    return "完整 recorded 才确认重复；同产物另算；同条件异结果标 contradiction；跨候选不合并"


def case_duplicate_status_is_enum():
    msg = _expect_raise(R._verdict, "duplicate", None, {}, "x")
    assert "DUPLICATE_STATES" in msg, msg
    msg = _expect_raise(R._verdict, "confirmed_duplicate", None, {}, "x")
    assert "kind" in msg, msg
    return "非法重复态/无 kind 的确认重复被拒"


def case_run_id_includes_data():
    """同一候选同 seed 同预算下跑 60 条谱 ⇒ 60 个 run_id，不能共享一个。"""
    c = _nested()
    a = _exec(candidate_id=c.candidate_id, data="s0.csv")
    b = _exec(candidate_id=c.candidate_id, data="s1.csv")
    assert a["run_id"] != b["run_id"], (a["run_id"], b["run_id"])
    rec_a = {"candidate_id": c.candidate_id, "source": {"path": "x/a", "sha256": "1"},
             "execution": a}
    rec_b = {"candidate_id": c.candidate_id, "source": {"path": "x/b", "sha256": "2"},
             "execution": b}
    v = R.duplicate_verdict(rec_a, rec_b, summary_equal=False)
    assert v["duplicate_status"] == "distinct", v
    assert "contradiction" not in v["evidence"], v
    # 频率窗口变了也是另一次执行
    c1 = _exec(candidate_id=c.candidate_id, data={"ref": "s0.csv", "f_min": 0.1, "f_max": 1e5})
    c2 = _exec(candidate_id=c.candidate_id, data={"ref": "s0.csv", "f_min": 1.0, "f_max": 1e5})
    assert c1["run_id"] != c2["run_id"]
    return "数据/频率窗口进执行身份；不同谱不会被误判成'同条件不同结果'"


def case_d1_duplicate_needs_code_version():
    """D1：同 seed、同预算、同摘要，但代码版本未知 ⇒ 只能 possible，不能扣。"""
    c = _nested()
    a = {"candidate_id": c.candidate_id, "source": {"path": "x/a.json", "sha256": "s1"},
         "execution": _exec(candidate_id=c.candidate_id)}
    b = {"candidate_id": c.candidate_id, "source": {"path": "x/b.json", "sha256": "s2"},
         "execution": _exec(candidate_id=c.candidate_id)}
    assert a["execution"]["run_id_status"] == "derived", a["execution"]
    v = R.duplicate_verdict(a, b, summary_equal=True)
    assert v["duplicate_status"] == "possible_duplicate", v
    assert v["independent"] is True, v
    assert v["evidence"]["run_id_equal"] is True, v
    assert v["evidence"]["code_version"] == [None, None], v
    assert v["evidence"]["conditions_observed"] is False, v
    return "run_id 相同 + 摘要相同，但代码版本未记录 ⇒ possible（不从独立数里扣）"


def case_d2_duplicate_confirmed_with_full_recorded():
    """D2：同 seed、同预算、同摘要，且代码版本与数据内容都记了 ⇒ confirmed。"""
    c = _nested()
    ver = {"producer": "nested_fit.py", "code_version": "a1b2c3d"}
    a = {"candidate_id": c.candidate_id, "source": {"path": "x/a.json", "sha256": "s1"},
         "execution": _exec(candidate_id=c.candidate_id, software=ver)}
    b = {"candidate_id": c.candidate_id, "source": {"path": "x/b.json", "sha256": "s2"},
         "execution": _exec(candidate_id=c.candidate_id, software=ver)}
    for r in (a, b):
        assert r["execution"]["run_id_status"] == "recorded", r["execution"]
    v = R.duplicate_verdict(a, b, summary_equal=True)
    assert v["duplicate_status"] == "confirmed_duplicate", v
    assert v["evidence"]["conditions_observed"] is True, v
    # 代码版本不同 ⇒ run_id 不同 ⇒ 不是同一次执行
    other = _exec(candidate_id=c.candidate_id,
                  software={"producer": "nested_fit.py", "code_version": "fffffff"})
    v2 = R.duplicate_verdict(a, {"candidate_id": c.candidate_id,
                                 "source": {"path": "x/c.json", "sha256": "s3"},
                                 "execution": other}, summary_equal=True)
    assert v2["duplicate_status"] == "possible_duplicate", v2
    assert v2["evidence"]["run_id_equal"] is False, v2
    return "四要素齐才确认重复；换代码版本 ⇒ run_id 变、退成 possible"


def case_d3_data_identity_by_content():
    """D3：同文件名、同频率窗，数据内容不同 ⇒ 不同 run_id。"""
    c = _nested()
    a = _exec(candidate_id=c.candidate_id,
              data={"ref": "spectrum.csv", "sha256": _SHA, "f_min": 0.1, "f_max": 1e5})
    b = _exec(candidate_id=c.candidate_id,
              data={"ref": "spectrum.csv", "sha256": _SHA2, "f_min": 0.1, "f_max": 1e5})
    assert a["run_id"] != b["run_id"], (a["run_id"], b["run_id"])
    assert a["data_id_kind"] == b["data_id_kind"] == "content_sha256"
    # 内容哈希相同、ref 不同（换了机器/目录）⇒ 仍是同一次执行
    cc = _exec(candidate_id=c.candidate_id,
               data={"ref": "D:/other/dir/spectrum.csv", "sha256": _SHA,
                     "f_min": 0.1, "f_max": 1e5})
    assert cc["run_id"] == a["run_id"], (cc["run_id"], a["run_id"])
    # 伪哈希被拒
    for bad in ("spectrum.csv", "z" * 64, "a" * 63):
        msg = _expect_raise(R.execution_identity, candidate_id=c.candidate_id,
                            data={"ref": "spectrum.csv", "sha256": bad}, seed=0,
                            seed_source="artifact:x", budget_params=dict(_BP))
        assert "64 位十六进制" in msg, msg
    return "数据身份按内容哈希：同名不同内容 ⇒ 两个 run_id；同内容不同路径 ⇒ 同一个"


def case_d6_execution_status_must_match_elements():
    """D6：有完整执行条件却声明"缺 seed"/标 unconfirmed ⇒ 校验失败。"""
    c = _nested()
    e = _exec(candidate_id=c.candidate_id, software={"producer": "x",
                                                     "code_version": "a1b2c3d"})
    assert e["run_id_status"] == "recorded", e
    bad = dict(e, run_id=None, run_id_status="unconfirmed",
               unavailable_reason="seed_not_recorded")
    msg = _expect_raise(R.observation_block, candidate_id=c.candidate_id,
                        run_status="completed", sampling_status="passed",
                        rms_rel=7.0, acf=0.2, budget_params=dict(_BP), execution=bad)
    assert "run_id_status" in msg or "要素" in msg, msg
    # 状态对、原因乱填也不行
    bad2 = dict(e, unavailable_reason="seed_not_recorded")
    msg = _expect_raise(R.observation_block, candidate_id=c.candidate_id,
                        run_status="completed", sampling_status="passed",
                        rms_rel=7.0, acf=0.2, budget_params=dict(_BP), execution=bad2)
    assert "unavailable_reason" in msg, msg
    # 真缺 seed 时必须报 seed 而不是别的
    miss = _exec(candidate_id=c.candidate_id, seed=None, seed_source="absent")
    assert miss["unavailable_reason"] == "seed_not_recorded", miss
    wrong = dict(miss, unavailable_reason="budget_not_recorded")
    msg = _expect_raise(R.observation_block, candidate_id=c.candidate_id,
                        run_status="completed", sampling_status="passed",
                        rms_rel=7.0, acf=0.2, budget_params=dict(_BP), execution=wrong)
    assert "unavailable_reason" in msg, msg
    # 完全不给执行块时的默认哨兵，本身也必须满足"未发号 ⇒ 原因在枚举里"
    blank = R.observation_block(candidate_id=c.candidate_id, run_status="failed")
    ex = blank["execution"]
    assert ex["run_id"] is None and ex["run_id_status"] == "unconfirmed", ex
    assert ex["unavailable_reason"] in R.RUN_ID_MISSING, ex
    assert ex["unavailable_reason"] == "not_provided", ex
    return "状态/原因按要素整块复算：齐全却称缺失、缺 A 却报 B，都被拒；默认哨兵合法"


def case_d7_ref_only_cannot_be_recorded():
    """D7：只有文件名、没有内容哈希 ⇒ 最高只能到 `derived`。"""
    c = _nested()
    ver = {"producer": "nested_fit.py", "code_version": "a1b2c3d"}
    e = _exec(candidate_id=c.candidate_id, data={"ref": "spectrum.csv"}, software=ver)
    assert e["data_id_kind"] == "ref_only", e
    assert e["run_id_status"] == "derived", e
    assert e["run_id"].startswith("D-"), e
    # data_id_kind 自报与实际不符 ⇒ 被拒
    e2 = _exec(candidate_id=c.candidate_id, data={"ref": "spectrum.csv"}, software=ver)
    bad = dict(e2, data_id_kind="content_sha256")
    msg = _expect_raise(R.observation_block, candidate_id=c.candidate_id,
                        run_status="completed", sampling_status="passed",
                        rms_rel=7.0, acf=0.2, budget_params=dict(_BP), execution=bad)
    assert "data_id_kind" in msg, msg
    return "文件名不是身份：ref_only 不能升 recorded；自报等级与实际不符被拒"


CASES = [
    ("A1 定义完整性", case_definitions),    ("A2 未接线须给原因", case_unwired_needs_reason),
    ("A3 正演引用白名单", case_ref_whitelist),
    ("A4 指纹覆盖结构校验版本", case_digest_covers_consistency),
    ("B1 候选可构造", case_candidate_ok),
    ("B2 candidate_id 覆盖 init/budget", case_candidate_id_covers_init),
    ("B3 branch 不一致", case_branch_mismatch),
    ("B4 layers=1 带 inner_cpe", case_layers1_with_inner),
    ("B5 未知 init_profile", case_unknown_init_profile),
    ("B6 未知槽位", case_bad_slot),
    ("B7 layers=3 被拒", case_layers3_rejected),
    ("B8 voigt K≠HN 数", case_voigt_K_mismatch),
    ("B9 include_L 与感抗件同侧", case_voigt_include_L_binding),
    ("B10 describe 保留重复槽位", case_describe_keeps_duplicate_slots),
    ("C1 screening 闭合", case_screening_closes),
    ("C2 only_survivor 是计数", case_only_survivor_is_count),
    ("C3 汇总须由逐条复算", case_screening_records_must_match_tally),
    ("D1 execution 闭合", case_execution_closes),
    ("E1 采样失败仍保留残差", case_observation_sampling_failed_keeps_residuals),
    ("E2 sampling 未过须显式解读", case_observation_sampling_failed_needs_explicit_interp),
    ("E3 采样通过默认 reliable", case_observation_passed_defaults_reliable),
    ("E4 未跑完不报数", case_observation_no_numbers_when_not_completed),
    ("E5 null 三语义", case_observation_null_reason),
    ("E6 静默缺席被拒", case_observation_silent_absence_rejected),
    ("E7 acf 允许负值", case_observation_acf_can_be_negative),
    ("E8 贴底但参数不稳", case_observation_floor_but_unstable),
    ("E9 参数不许压成标量", case_observation_no_scalar_score),
    ("E10 采样态折算", case_sampling_status_from),
    ("F1 真实谱默认证据不足", case_evidence_insufficient_default),
    ("F2 不可区分需>=2", case_evidence_non_identifiable_needs_two),
    ("F3 identified 被拦", case_evidence_identified_blocked),
    ("F4 model_inadequate 需真值", case_evidence_inadequate_needs_truth),
    ("F5 only_survivor 非证据态", case_only_survivor_not_evidence_state),
    ("G1 L1 永不出 excluded", case_l1_never_excludes),
    ("G2 L1 初始未探 → defer", case_l1_init_not_explored_defers),
    ("H1 manifest 冻结", case_manifest_frozen),
    ("H2 不可追加", case_manifest_no_append),
    ("H3 重复候选", case_manifest_duplicate),
    ("H4 指纹自洽与篡改", case_verify_manifest_ok_and_tamper),
    ("H5 删除/同名换定义", case_verify_manifest_rejects_removal_and_swap),
    ("H6 注册表漂移", case_verify_manifest_registry_drift),
    ("I1 AZ31 风格 excluded", case_l0_az31_style_excluded),
    ("I2 容性谱 not_excluded", case_l0_capacitive_not_excluded),
    ("I3 nested unknown", case_l0_nested_unknown),
    ("I4 预检绑定候选配置", case_l0_binds_candidate_config),
    ("J1 init 未知不是档名", case_init_unknown_is_not_a_real_profile),
    ("J2 unknown 不能自称 recorded", case_unknown_init_cannot_claim_recorded),
    ("J3 近似档名被拒", case_unregistered_init_rejected),
    ("J4 身份等级", case_identity_levels),
    ("J5 voigt 的 data_driven", case_voigt_has_data_driven_profile),
    ("K1 档名≠预算数值", case_budget_profile_is_not_params),
    ("K2 预算参数齐全性", case_budget_params_recorded_needs_all),
    ("K3 ratio 分母必填", case_ratio_needs_denominator),
    ("K4 ratio 全样本口径", case_ratio_scope_all_runs),
    ("K5 噪声底来源不可伪造", case_noise_floor_cannot_fake_source),
    ("K6 真值列只许合成谱", case_param_estimate_truth_gate),
    ("K7 ELPD 可用性显式", case_predictive_validity_must_be_explicit),
    ("L1 run_id 分开预算数值", case_run_id_separates_budget_values),
    ("L2 无 seed 不发 run_id", case_run_id_refuses_default_seed),
    ("L3 执行身份三档", case_run_id_status_ladder),
    ("L4 U- 候选不发 id", case_run_id_rejects_unknown_candidate),
    ("L5 观测重算 run_id", case_observation_recomputes_run_id),
    ("L6 重复须有来源证据", case_duplicate_needs_provenance),
    ("L7 确认重复的两种情形", case_duplicate_confirmed_with_recorded_seed),
    ("L8 重复态是枚举", case_duplicate_status_is_enum),
    ("L9 执行身份含数据", case_run_id_includes_data),
    ("D1 确认重复需代码版本", case_d1_duplicate_needs_code_version),
    ("D2 四要素齐才确认", case_d2_duplicate_confirmed_with_full_recorded),
    ("D3 数据按内容识别", case_d3_data_identity_by_content),
    ("D6 执行态整块复算", case_d6_execution_status_must_match_elements),
    ("D7 文件名不是身份", case_d7_ref_only_cannot_be_recorded),
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
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
