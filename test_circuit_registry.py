# -*- coding: utf-8 -*-
"""test_circuit_registry.py —— 阶段 0C：schema 契约测试。

目的：`circuit_registry.py` 的 schema 必须能**表达**下面六类事实，且不能把
不该合并的维度合并掉。这里测的是"能不能装下 + 乱填会不会报错"，不是模型好坏。

    A  定义完整性      —— 两个拓扑、元件接线状态、registry 指纹
    B  候选构造        —— 槽位/元件/初始化档的约束真的在拦
    C  screening       —— only_survivor 只能由 survivors==1 算出来
    D  execution       —— 跑没跑成与失败原因分开记账
    E  observation     —— 贴底但参数不稳要能同时保留；未收敛不许报数
    F  evidence 五态   —— identified / model_inadequate 的硬条件真的在拦
    G  L1 三态         —— 快速拟合失败**永远**不返回 excluded
    H  manifest        —— 拟合前冻结、不可追加
    I  L0 预检         —— 真实调用 `voigt_can_represent` 的三条路径

用法：python test_circuit_registry.py
"""
from __future__ import annotations

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


def _voigt(K: int = 2, **kw):
    return R.make_candidate(
        "voigt", [("relaxation_unit", "hn")] * K,
        {"K": K}, init_profile=kw.pop("init_profile", "uniform"), **kw)


def _nested(branch="capacitive", layers=2, init="uniform", **kw):
    els = [("outer_cpe", "cpe"), ("branch",
                                  {"parallel": "rl_parallel",
                                   "series": "rl_series",
                                   "capacitive": "cpe"}[branch])]
    if layers == 2:
        els.insert(1, ("inner_cpe", "cpe"))
    return R.make_candidate("nested", els, {"branch": branch, "layers": layers},
                            init_profile=init, **kw)


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
# E observation
# --------------------------------------------------------------------------
def case_observation_floor_but_unstable():
    o = R.observation_block(
        candidate_id="C-demo", converged=True, rms_rel=2.40, acf=0.279,
        rhat_max=1.01, param_stability={
            "R2": {"median": 1.31e5, "ci90": [0.9e5, 1.9e5], "flag": "wide"},
            "n2": {"median": 0.899, "flag": "at_bound"},
        })
    assert o["rms_rel"] == 2.40 and o["acf"] == 0.279
    assert o["param_stability"]["n2"]["flag"] == "at_bound"
    return "贴底（2.40%）与参数不稳（n2 at_bound）可同时保留"


def case_observation_acf_can_be_negative():
    o = R.observation_block(candidate_id="C-neg", converged=True,
                            rms_rel=6.65, acf=-0.244)
    assert o["acf"] == -0.244
    return "acf 允许负值"


def case_observation_no_numbers_when_unconverged():
    msg = _expect_raise(R.observation_block, candidate_id="C-x",
                        converged=False, rms_rel=5.0, acf=0.4)
    assert "未收敛" in msg
    o = R.observation_block(candidate_id="C-y", converged=False)
    assert o["rms_rel"] is None and o["acf"] is None
    return "未收敛不得报 rms/acf（E3）"


def case_observation_no_scalar_score():
    _expect_raise(R.observation_block, candidate_id="C-z", converged=True,
                  param_stability={"n2": 0.42})
    _expect_raise(R.observation_block, candidate_id="C-z", converged=True,
                  param_stability={"n2": {"median": 0.5}})
    return "参数可信度不许压成标量"


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
    m = R.candidate_set_manifest(candidate_set_id="CS-1", candidates=cs,
                                 likelihood_profile="perpoint-2pct",
                                 budget_profile="standard",
                                 seed_policy="seed=0..2",
                                 created_before_fit=True)
    for k in ("candidate_set_id", "candidate_ids", "candidate_definitions",
              "registry_version", "precheck_version", "likelihood_profile",
              "budget_profile", "seed_policy", "created_before_fit"):
        assert k in m, k
    assert len(m["candidate_ids"]) == 3 and m["immutable"] is True
    json.dumps(m, ensure_ascii=False)
    _expect_raise(R.candidate_set_manifest, candidate_set_id="CS-2",
                  candidates=cs, likelihood_profile="x", budget_profile="x",
                  seed_policy="x", created_before_fit=False)
    return "9 字段齐 + 可序列化；created_before_fit=False 被拒"


def case_manifest_no_append():
    cs = [_voigt(2)]
    m = R.candidate_set_manifest(candidate_set_id="CS-1", candidates=cs,
                                 likelihood_profile="x", budget_profile="x",
                                 seed_policy="x", created_before_fit=True)
    R.assert_no_append(m, cs)
    msg = _expect_raise(R.assert_no_append, m, [_voigt(2), _nested("capacitive", 2)])
    assert "新的 candidate_set_id" in msg
    return "冻结后追加被拒，提示新建候选集"


def case_manifest_duplicate():
    cs = [_voigt(2), _voigt(2)]
    _expect_raise(R.candidate_set_manifest, candidate_set_id="CS-1",
                  candidates=cs, likelihood_profile="x", budget_profile="x",
                  seed_policy="x", created_before_fit=True)
    return "重复 candidate_id 被拒"


# --------------------------------------------------------------------------
# I L0 预检（真实调用）
# --------------------------------------------------------------------------
def case_l0_az31_style_excluded():
    import numpy as np
    import bm_top as T
    f = np.logspace(-2, 5, 50)
    Z = 100.0 + T.z_rl_parallel(f, 1000.0, 10.0)
    state, why = R.l0_precheck("voigt", f, Z)
    assert state == "excluded", (state, why)
    assert "低频感抗" in why
    return "低频感抗谱 → voigt excluded"


def case_l0_capacitive_not_excluded():
    import numpy as np
    import bm_top as T
    f = np.logspace(0, 4.4, 60)
    p = dict(Rs=141.2, R1=9.28e6, Q1=6.62e4, n1=0.782,
             Q_b=3.30e5, n_b=0.5002)
    Z = T.z_nested(f, p, branch="capacitive", layers=1)
    state, why = R.l0_precheck("voigt", f, Z)
    assert state == "not_excluded", (state, why)
    return "容性嵌套谱 → voigt not_excluded（预检覆盖不到，这是已知的洞）"


def case_l0_nested_unknown():
    import numpy as np
    f = np.logspace(0, 4, 30)
    state, why = R.l0_precheck("nested", f, 1.0 + 0j)
    assert state == "unknown" and why
    return "nested 未登记预检 → unknown"


CASES = [
    ("A1 定义完整性", case_definitions),
    ("A2 未接线须给原因", case_unwired_needs_reason),
    ("A3 正演引用白名单", case_ref_whitelist),
    ("B1 候选可构造", case_candidate_ok),
    ("B2 candidate_id 覆盖 init/budget", case_candidate_id_covers_init),
    ("B3 branch 不一致", case_branch_mismatch),
    ("B4 layers=1 带 inner_cpe", case_layers1_with_inner),
    ("B5 未知 init_profile", case_unknown_init_profile),
    ("B6 未知槽位", case_bad_slot),
    ("C1 screening 闭合", case_screening_closes),
    ("C2 only_survivor 是计数", case_only_survivor_is_count),
    ("D1 execution 闭合", case_execution_closes),
    ("E1 贴底但参数不稳", case_observation_floor_but_unstable),
    ("E2 acf 允许负值", case_observation_acf_can_be_negative),
    ("E3 未收敛不报数", case_observation_no_numbers_when_unconverged),
    ("E4 参数不许压成标量", case_observation_no_scalar_score),
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
    ("I1 AZ31 风格 excluded", case_l0_az31_style_excluded),
    ("I2 容性谱 not_excluded", case_l0_capacitive_not_excluded),
    ("I3 nested unknown", case_l0_nested_unknown),
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
