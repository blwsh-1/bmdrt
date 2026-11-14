# -*- coding: utf-8 -*-
"""test_exec_identity.py —— 执行身份契约 v1.2 的反例矩阵（T1–T24 + W + S + M）。

测的是**语义**不是数字：每一格问的都是"这个改动该不该换号"。所以记录是人工搭的
最小 v1 记录，不跑拟合——`run_id` 的输入是记录，不是 numpyro。

三组补充（这一轮新增，接审查的"不能只测 `run_id`"）：

- **W1–W6** `workflow_id`：改 `bm_grade`/重跑逻辑/门禁/`children` 必须变，
  改画图模块、改 `kk_status` 这类**结果**必须不变；
- **S1–S7** `sigma` 数组摘要：`<f8` 小端 C 顺序、形状进载荷、拒 NaN/Inf/空数组、
  空数组 ≠ `None` ≠ `kind=none`、可复算；
- **M1–M7** `source_manifest.json`：三份 Profile 与源码一致、**没有任何乐观降级**、
  降级缺证据会被拒。

说明：T10/W1 用**夹具**里降级的模块（`plots.py`/`grade.py`）验证机制；
真实清单里 `bm_circuit`/`bm_grade` 仍是 `numeric`，降级要过一次真实运行（M4、M6）。
不为了让测试通过而降低真实源码的角色——这是本轮三条硬验收条件的第一条。

    python test_exec_identity.py
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np                        # noqa: E402

import exec_identity as E                 # noqa: E402
import circuit_registry as R              # noqa: E402


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------
def _sha(tag: str) -> str:
    import hashlib
    return hashlib.sha256(tag.encode()).hexdigest()


_DATA = _sha("data/a.csv")
_UNSET = E.UNSET


def _items(*, plot_demoted: bool = False, decision_demoted: bool = False,
           plot_in_workflow: bool = True) -> list[dict]:
    """最小声明集：入口 + 数值模块（+ 可选的两个降级模块）。"""
    items = [
        {"path": "main.py", "role": "entry", "sha256": _sha("main")},
        {"path": "core.py", "role": "module", "sha256": _sha("core")},
    ]
    if plot_demoted:
        items.append({"path": "plots.py", "role": "module", "sha256": _sha("plots"),
                      "compute_role": "plot_only",
                      "why_not_numeric": "只画图",
                      "evidence": "夹具 T10/W3：plot=False 的运行不加载它",
                      "workflow_scope": plot_in_workflow})
    if decision_demoted:
        items.append({"path": "grade.py", "role": "module", "sha256": _sha("grade"),
                      "compute_role": "decision",
                      "why_not_numeric": "事后评分，不改后验",
                      "evidence": "夹具 W1：改无关常量后 params 逐位相同"})
    return items


def _code(items: list[dict], **over) -> dict:
    code = {
        "git_commit": "a" * 40,
        "declared_source_set": items,
        "declared_source_tree_sha256": E.declared_source_tree_sha256(items),
        "numeric_source_tree_sha256": E.numeric_source_tree_sha256(items),
        "workflow_source_tree_sha256": E.workflow_source_tree_sha256(items),
        "declared_not_loaded": [],
        "runtime_loaded_modules": [{"path": it["path"], "role": it["role"]}
                                   for it in items],
        "runtime_snapshot_taken": True,
        "runtime_snapshot_points": ["before_fit", "after_fit"],
        "source_coverage": "complete",
        "undeclared_modules": [],
        "external_modules": [],
        "git_dirty": False,
        "dirty_paths": [],
        "dirty_scope": "clean",
        "patch_bundle_ref": None,
        "patch_bundle_sha256": None,
        "bundle_excluded": [],
        "git_describe": "v0.7.0-3-gabc",
    }
    code.update(over)
    return code


def _env(**over) -> dict:
    env = {"python": "3.11.9", "python_impl": "CPython",
           "platform": "Windows-10-10.0.22631-SP0",
           "packages": {"numpy": "1.26.4", "numpyro": "0.15.3"}}
    env["env_sha256"] = E._env_sha256(env["python"], env["python_impl"],
                                      env["platform"], env["packages"])
    env.update({"backend": "cpu", "x64_enabled": True,
                "cpu_threads": 8, "device_kind": "cpu", "hardware_tag": "x86_64",
                "lock_digest": None})
    env.update(over)
    return env


def _rec(*, items: list[dict] | None = None) -> dict:
    items = list(items if items is not None else _items())
    sig = E.sigma_block(kind="none", mode="free")
    return {
        "identity_contract_version": E.IDENTITY_CONTRACT_VERSION,
        "candidate_id": "voigt:hn2+rl_parallel",
        "data": {"ref": "a.csv", "sha256": _DATA, "n_points": 50,
                 "subset_sha256": None, "f_min": 0.1, "f_max": 1e5,
                 "transform": {"fscale": 1.0, "zscale": 1.0, "mask": None},
                 "id_kind": "content_sha256"},
        "rng": {"plan": {"root_seed": 0, "prng": "jax.threefry2x32",
                         "derivation": "numpyro.handlers.seed(node,key); chain i <- split(key,n)[i]",
                         "n_chains": 2, "init_strategy": "init_to_uniform"},
                "seed_evidence": {"source": "artifact:config.seed"},
                "actual": {"chain_keys": ["k0", "k1"], "init_values_sha256": None}},
        "budget_params": {k: {"value": v, "source": "recorded"} for k, v in
                          zip(E.BUDGET_KEYS, (2000, 1000, 2, 0.98, 10))},
        "config": {
            "effective": {
                "branch": {"value": "parallel", "origin": "cli",
                           "effect": "model_definition"},
                "layers": {"value": 2, "origin": "cli", "effect": "model_definition"},
                "include_L": {"value": False, "origin": "derived",
                              "effect": "model_definition"},
                "adsorption": {"value": None, "origin": "derived",
                               "effect": "model_definition"},
                "sigma": {"value": sig, "origin": "cli",
                          "effect": "execution_condition"},
                "sigma_floor": {"value": 0.0, "origin": "default",
                                "effect": "execution_condition"},
                "kk_mode": {"value": "warn", "origin": "cli",
                            "effect": "admission_condition"},
                "plot": {"value": False, "origin": "default", "effect": "display_only"},
                "progress": {"value": False, "origin": "default",
                             "effect": "display_only"},
            },
            "capture_interface": {"callable": "nested_fit.run", "module": "nested_fit",
                                  "line": 319, "kwargs_rule": "explicit+kwargs"},
            "unknown_keys": [],
            "unclassified_keys": [],
            "normalize": "eol-lf-text",
        },
        "code": _code(items),
        "env": _env(),
        "workflow": {
            "role": "parent",
            "search": {"krange": None, "kmax": 5, "wide": False, "fixed_K": None,
                       "layers": [2], "branches": ["parallel", "series"],
                       "n_seeds": 1},
            "extras": {"auto_mechanism": False, "auto_mechanism_full": False,
                       "auto_adsorption": False, "with_voigt": False, "ref_rwp": False},
            "gate": {"kk_mode": "warn", "kk_opts": {}, "strict": False,
                     "kk_status": "pass", "kk_performed": True},
            "guards": {"diag": {}, "drt_opts": {}, "strict": False, "gate_b": False},
            "pair_mode": "pooled",
            "retry": 0,
            "retry_kw": {},
            "children": [],
        },
        "instance": {"artifact_path": "out/r.json", "artifact_sha256": None,
                     "started_at": "2026-11-14T20:00:00", "finished_at": "2026-11-14T20:02:00",
                     "elapsed_s": 120.0, "pid": 4242, "host": "ws1",
                     "init_values_sha256": None},
        "provenance": {"registry_version": R.REGISTRY_VERSION},
    }


def _set(rec: dict, dotted: str, value) -> dict:
    cur = rec
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value
    return rec


def _val(rec: dict, dotted: str):
    cur = rec
    for p in dotted.split("."):
        cur = cur[p]
    return cur


def _ids(rec: dict, **kw) -> tuple:
    ev = E.evaluate(rec, **kw)
    return ev["run_id"], ev["workflow_id"], ev["status"], ev["reasons"]


def _must_change(rec: dict, dotted: str, value, *, field: int = 0, label: str = "") -> str:
    """改一处，断言 `run_id`（field=0）或 `workflow_id`（field=1）变。"""
    before = _ids(rec)
    after = _ids(_set(copy.deepcopy(rec), dotted, value))
    assert before[field] != after[field], f"{label or dotted}：该变号却没变"
    return f"{label or dotted}：{before[field][:12]}… → {after[field][:12]}…"


def _must_same(rec: dict, dotted: str, value, *, extra: str = "") -> str:
    """改一处，断言两个号都不变（且状态仍是 recorded）。"""
    before = _ids(rec)
    after = _ids(_set(copy.deepcopy(rec), dotted, value))
    assert before[0] == after[0], f"{dotted}：run_id 不该变却变了 {extra}"
    assert before[1] == after[1], f"{dotted}：workflow_id 不该变却变了 {extra}"
    assert after[2] == "recorded", f"{dotted}：状态被拖降级了 {after[3]}"
    return f"{dotted}：同号（{before[0][:12]}…）"


def _expect_raise(fn, *a, **kw) -> str:
    try:
        fn(*a, **kw)
    except (ValueError, KeyError) as exc:
        return str(exc)[:70]
    raise AssertionError("期望报错但成功了")


# --------------------------------------------------------------------------
# T1–T4 模型定义必须换号（candidate_id 固定，单独隔离 config 通道）
# --------------------------------------------------------------------------
def case_T1_branch():
    return _must_change(_rec(), "config.effective.branch.value", "series", label="T1 branch")


def case_T2_include_L():
    return _must_change(_rec(), "config.effective.include_L.value", True, label="T2 include_L")


def case_T3_layers():
    return _must_change(_rec(), "config.effective.layers.value", 1, label="T3 layers")


def case_T4_adsorption():
    return _must_change(_rec(), "config.effective.adsorption.value", "RL", label="T4 adsorption")


# --------------------------------------------------------------------------
# T5–T8 展示/实例字段不许换号
# --------------------------------------------------------------------------
def case_T5_progress():
    return _must_same(_rec(), "config.effective.progress.value", True)


def case_T6_plot():
    return _must_same(_rec(), "config.effective.plot.value", True)


def case_T7_instance():
    r1 = _rec()
    r2 = _set(copy.deepcopy(r1), "instance.elapsed_s", 999.0)
    _set(r2, "instance.started_at", "2027-01-01T00:00:00")
    _set(r2, "instance.pid", 1)
    a, b = _ids(r1), _ids(r2)
    assert a[0] == b[0] and a[1] == b[1], "实例字段进了条件哈希（F6）"
    return "T7 实例字段：同号，仅 instance.* 变"


def case_T8_init_values():
    r1 = _rec()
    r2 = _set(copy.deepcopy(r1), "rng.actual.init_values_sha256", _sha("init-B"))
    a, b = _ids(r1), _ids(r2)
    assert a[0] == b[0], "实际初值进了 run_id：那同条件两次跑会换号"
    assert _val(r1, "rng.actual.init_values_sha256") != _val(r2, "rng.actual.init_values_sha256")
    return "T8 同 seed 不同实际初值：同号，init_values_sha256 留痕"


# --------------------------------------------------------------------------
# T9–T11 代码身份的三层
# --------------------------------------------------------------------------
def case_T9_numeric_change():
    r1 = _rec()
    items = copy.deepcopy(_val(r1, "code.declared_source_set"))
    items[1]["sha256"] = _sha("core-v2")            # 改一个数值模块
    R2 = _set(copy.deepcopy(r1), "code", _code(items))
    a, b = _ids(r1), _ids(R2)
    assert a[0] != b[0], "改了参与计算的常量，run_id 没变（F2）"
    assert b[1] != a[1], "workflow 层默认包含数值模块，此时也该变"
    return f"T9 数值模块变更：{a[0][:10]}… → {b[0][:10]}…"


def case_T10_plot_only():
    base = _rec(items=_items(plot_demoted=True))
    items = copy.deepcopy(_val(base, "code.declared_source_set"))
    items[2]["sha256"] = _sha("plots-v2")            # 只改画图模块
    r2 = _set(copy.deepcopy(base), "code", _code(items))
    a, b = _ids(base), _ids(r2)
    assert a[0] == b[0], "plot_only 改动换了 run_id（F11：装饰性改动切碎样本）"
    assert _val(base, "code.declared_source_tree_sha256") != _val(r2, "code.declared_source_tree_sha256")
    assert _val(base, "code.numeric_source_tree_sha256") == _val(r2, "code.numeric_source_tree_sha256")
    return "T10 plot_only：run_id 同、declared 摘要变、numeric 摘要不变"


def case_T11_registry_version():
    r1 = _rec()
    r2 = _set(copy.deepcopy(r1), "provenance.registry_version", "0.8.0")
    return _must_same(r2 if False else r1, "provenance.registry_version", "0.8.0")


# --------------------------------------------------------------------------
# T12–T14 数据身份
# --------------------------------------------------------------------------
def case_T12_other_file():
    return _must_change(_rec(), "data.sha256", _sha("data/b.csv"), label="T12 换文件")


def case_T13_subset():
    r = _rec()
    out = [_must_change(r, "data.n_points", 40, label="T13a 点数")]
    out.append(_must_change(r, "data.subset_sha256", _sha("mask-1"), label="T13b 掩码"))
    return "；".join(out)


def case_T14_zscale():
    return _must_change(_rec(), "data.transform.zscale", 2.0, label="T14 zscale")


# --------------------------------------------------------------------------
# T15–T18 档位/工作流/门禁/阈值
# --------------------------------------------------------------------------
def case_T15_budget_tier():
    r = _rec()
    r2 = copy.deepcopy(r)
    for k, v in zip(E.BUDGET_KEYS, (800, 400, 2, 0.9, 10)):   # quick 档的**数值**
        r2["budget_params"][k] = {"value": v, "source": "recorded"}
    a, b = _ids(r), _ids(r2)
    assert a[0] != b[0], "预算数值变了 run_id 没变（档名冒充数值，F4）"
    assert "standard" not in json.dumps(E.run_id_payload(r2)), "档名混进了载荷"
    return f"T15 standard→quick：{a[0][:10]}… → {b[0][:10]}…（载荷里只有数值）"


def case_T16_with_voigt():
    parent = _rec()
    child = copy.deepcopy(parent)
    _set(child, "workflow.role", "child")
    child_items = copy.deepcopy(_val(parent, "code.declared_source_set"))
    child_run = "R1-" + E.run_id_digest(child)[:32]
    p2 = _set(copy.deepcopy(parent), "workflow.extras.with_voigt", True)
    _set(p2, "workflow.children", [child_run])
    a, b = _ids(parent), _ids(p2)
    assert a[0] == b[0], "多一个对照子运行改了父的 run_id（附加对照污染主身份）"
    assert a[1] != b[1], "children 变了 workflow_id 却没变"
    assert child_items, "夹具声明集不该为空"
    return f"T16 with_voigt：父 run_id 同、workflow_id 变（{a[1][:10]}…→{b[1][:10]}…）"


def case_T17_kk_mode():
    r = _rec()
    a = _ids(r)
    r2 = _set(_set(copy.deepcopy(r), "config.effective.kk_mode.value", "gate"),
              "workflow.gate.kk_mode", "gate")
    b = _ids(r2)
    assert a[0] == b[0], "门禁档位进了 run_id：两次后验一样却换号"
    assert a[1] != b[1], "门禁档位变了 workflow_id 却没变（工作流身份漏门禁）"
    return "T17 kk_mode warn→gate：run_id 同、workflow_id 变"


def case_T18_diag():
    r = _rec()
    a = _ids(r)
    r2 = _set(copy.deepcopy(r), "workflow.guards.diag", {"r_min": 0.9})
    b = _ids(r2)
    assert a[0] == b[0], "事后阈值进了 run_id"
    assert a[1] != b[1], "事后阈值变了 workflow_id 却没变"
    return "T18 diag 阈值：run_id 同、workflow_id 变"


# --------------------------------------------------------------------------
# T19–T21 复合键与哨兵
# --------------------------------------------------------------------------
def case_T19_sigma_array_not_mode():
    z = np.array([[0.02, 0.03, 0.04], [0.02, 0.03, 0.04]])
    r1 = _rec()
    s1 = E.sigma_block(kind="perpoint", mode="perpoint", sigma=z, pct=2.0, from_data=True)
    s2 = E.sigma_block(kind="perpoint", mode="scalar", sigma=z, abs_=0.03, from_data=True)
    _set(r1, "config.effective.sigma.value", s1)
    r2 = _set(copy.deepcopy(r1), "config.effective.sigma.value", s2)
    a, b = _ids(r1), _ids(r2)
    assert a[0] == b[0], "实际 σ 数组相同、只是来源不同，却换了号（T19 的核心主张）"
    assert s1["mode"] != s2["mode"], "夹具没构造出 mode 差异"
    # 反向：数组不同就必须换号——这才是 D-8 真正的落点
    s3 = E.sigma_block(kind="perpoint", mode="perpoint", sigma=z * 1.0001, pct=2.0)
    r3 = _set(copy.deepcopy(r1), "config.effective.sigma.value", s3)
    assert _ids(r3)[0] != a[0], "实际 σ 数组变了 run_id 没变（D-8 没落地）"
    return "T19 同数组不同 mode=同号；数组差 0.01%=换号"


def case_T19b_sigma_none_vs_block():
    r1 = _rec()
    r2 = _set(copy.deepcopy(r1), "config.effective.sigma.value",
              E.sigma_block(kind="none", mode="free"))
    assert _ids(r1)[0] == _ids(r2)[0]
    r3 = _set(copy.deepcopy(r1), "config.effective.sigma.value",
              E.sigma_block(kind="scalar_pair", mode="scalar", sigma=(0.03, 0.03)))
    assert _ids(r3)[0] != _ids(r1)[0], "自由 σ → 固定 σ 必须换号"
    return "T19b kind=none 与固定 σ：换号；两个 free 写法同号"


def case_T20_unset_resolved():
    r = _rec()
    _set(r, "config.effective.adsorption.value", None)
    _set(r, "config.effective.adsorption.origin", "cli")
    assert _ids(r)[0] == _ids(_rec())[0], "显式 None 与解析后的 None 应当同号"
    msg = _expect_raise(E.norm_effective, "adsorption", _UNSET)
    assert "UNSET" in msg or "_UNSET" in msg, msg
    return f"T20 _UNSET 拒收（{msg}）、显式 None 与解析值同号"


def case_T21_zeta_norm():
    r = _rec()
    _set(r, "config.effective.zeta", {"value": "1.0", "origin": "config_file",
                                      "effect": "model_definition"})
    a = _ids(r)[0]
    _set(r, "config.effective.zeta", {"value": 1.0, "origin": "cli",
                                      "effect": "model_definition"})
    assert _ids(r)[0] == a, "`zeta=\"1.0\"` 与 `1.0` 没归一（F12）"
    _set(r, "config.effective.zeta", {"value": "free", "origin": "cli",
                                      "effect": "model_definition"})
    assert _ids(r)[0] != a, "`free` 与固定值必须换号"
    return "T21 zeta：'1.0' ≡ 1.0，'free' 不同号"


# --------------------------------------------------------------------------
# T22–T24 覆盖性与补丁包
# --------------------------------------------------------------------------
def case_T22_declared_not_loaded():
    r = _rec()
    r2 = _set(copy.deepcopy(r), "code.declared_not_loaded", ["core.py"])
    _set(r2, "code.runtime_loaded_modules",
         [{"path": "main.py", "role": "entry"}])
    a, b = _ids(r), _ids(r2)
    assert a[0] == b[0] and b[2] == "recorded", f"声明了没用到不该降级：{b[3]}"
    assert b[1] == a[1]
    return "T22 declared_not_loaded 非空：仍 recorded、同号"


def case_T23_undeclared_loaded():
    r = _rec()
    r2 = _set(copy.deepcopy(r), "code.undeclared_modules", ["sneaky.py"])
    _set(r2, "code.source_coverage", "undeclared_loaded")
    ev = E.evaluate(r2)
    assert ev["status"] == "derived", f"加载了未声明模块却没降级：{ev['status']}"
    assert ev["run_id"].startswith("D1-"), ev["run_id"]
    return f"T23 未声明模块：降 derived（{ev['reasons']}）"


def case_T24_bundle_gap():
    r = _rec()
    r2 = _set(copy.deepcopy(r), "code.git_dirty", True)
    _set(r2, "code.dirty_scope", "code_only")
    _set(r2, "code.dirty_paths", ["core.py"])
    ev_nobundle = E.evaluate(r2)
    assert ev_nobundle["status"] == "derived", "dirty 但没补丁包却 recorded"
    assert "bundle_missing_when_dirty" in ev_nobundle["reasons"]
    r3 = _set(copy.deepcopy(r2), "code.patch_bundle_sha256", _sha("bundle"))
    _set(r3, "code.patch_bundle_ref", "out/patch.tgz")
    ev_ok = E.evaluate(r3)
    assert ev_ok["status"] == "recorded", f"补齐补丁包应当 recorded：{ev_ok['reasons']}"
    r4 = _set(copy.deepcopy(r3), "code.bundle_excluded",
              [{"path": "core.py", "reason": "超出 BUNDLE_MAX_BYTES"}])
    ev_bad = E.evaluate(r4)
    assert ev_bad["status"] == "derived", "补丁包排除了声明集文件却仍 recorded（F8）"
    return f"T24 dirty：无包/排除声明文件 ⇒ derived（{ev_bad['reasons'][-1]}）"


# --------------------------------------------------------------------------
# W1–W6 workflow_id（这一轮新增：不能只测 run_id）
# --------------------------------------------------------------------------
def case_W1_decision_module():
    base = _rec(items=_items(decision_demoted=True))
    items = copy.deepcopy(_val(base, "code.declared_source_set"))
    idx = [i for i, it in enumerate(items) if it["path"] == "grade.py"][0]
    items[idx]["sha256"] = _sha("grade-v2")
    r2 = _set(copy.deepcopy(base), "code", _code(items))
    a, b = _ids(base), _ids(r2)
    assert a[0] == b[0], "decision 模块改了却换了 run_id（误降的反方向）"
    assert a[1] != b[1], "评分逻辑改了 workflow_id 没变——这正是 D-5 要堵的洞"
    return f"W1 bm_grade 类改动：run_id 同、workflow_id {a[1][:10]}…→{b[1][:10]}…"


def case_W2_entry_retry():
    base = _rec(items=_items())
    items = copy.deepcopy(_val(base, "code.declared_source_set"))
    items[0]["sha256"] = _sha("main-v2")             # 入口含重跑逻辑
    r2 = _set(copy.deepcopy(base), "code", _code(items))
    a, b = _ids(base), _ids(r2)
    assert a[0] != b[0] and a[1] != b[1], "改入口（重跑/选择逻辑）两个号都该变"
    return "W2 入口改动：run_id 与 workflow_id 都变"


def case_W3_plot_out_of_workflow():
    base = _rec(items=_items(plot_demoted=True, plot_in_workflow=False))
    items = copy.deepcopy(_val(base, "code.declared_source_set"))
    idx = [i for i, it in enumerate(items) if it["path"] == "plots.py"][0]
    items[idx]["sha256"] = _sha("plots-v2")
    r2 = _set(copy.deepcopy(base), "code", _code(items))
    a, b = _ids(base), _ids(r2)
    assert a[0] == b[0] and a[1] == b[1], "退出 workflow 层的画图模块仍影响 workflow_id"
    assert _val(base, "code.declared_source_tree_sha256") != _val(r2, "code.declared_source_tree_sha256")
    return "W3 workflow_scope=False 的模块：两个号都不变，declared 仍记录"


def case_W4_gate_result_not_identity():
    r = _rec()
    a = _ids(r)
    r2 = _set(copy.deepcopy(r), "workflow.gate.kk_status", "fail")
    _set(r2, "workflow.gate.kk_performed", False)
    b = _ids(r2)
    assert a[1] == b[1], "门禁**结果**进了 workflow_id：工作流身份变成了结果身份"
    assert a[0] == b[0]
    return "W4 kk_status pass→fail：两个号都不变（结果不是条件）"


def case_W5_children():
    r = _rec()
    a = _ids(r)
    r2 = _set(copy.deepcopy(r), "workflow.children", ["R1-" + "f" * 32])
    b = _ids(r2)
    assert a[1] != b[1], "children 变了 workflow_id 没变"
    assert a[0] == b[0], "children 不该影响父 run_id"
    return "W5 children 变化：workflow_id 变、run_id 同"


def case_W6_search_space():
    r = _rec()
    a = _ids(r)
    r2 = _set(copy.deepcopy(r), "workflow.search.kmax", 6)
    b = _ids(r2)
    assert a[1] != b[1], "搜索空间变了 workflow_id 没变"
    assert a[0] == b[0], "搜索空间不该进单次 run_id"
    return "W6 kmax 5→6：workflow_id 变、run_id 同"


# --------------------------------------------------------------------------
# S1–S7 sigma 数组摘要的规范化（D-8 的补齐要求）
# --------------------------------------------------------------------------
def case_S1_f8_c_order():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    got = E.sigma_array_sha256(a)
    want = __import__("hashlib").sha256(b"v1\n2x2\n" + a.astype("<f8").tobytes(order="C")).hexdigest()
    assert got == want, "摘要口径与 v1\\n<shape>\\n<f8 C 顺序> 不一致"
    return f"S1 复算一致：{got[:12]}…"


def case_S2_shape_in_payload():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = a.reshape(3, 2)
    assert E.sigma_array_sha256(a) != E.sigma_array_sha256(b), "形状没进载荷：2x3 与 3x2 撞号"
    return "S2 (2,3) ≠ (3,2)：形状在载荷里"


def case_S3_reject_nonfinite():
    for bad in (np.array([1.0, np.nan]), np.array([1.0, np.inf])):
        msg = _expect_raise(E.sigma_array_sha256, bad)
        assert "NaN" in msg or "Inf" in msg, msg
    return "S3 NaN/Inf 被拒"


def case_S4_empty_vs_none_vs_kind():
    msg = _expect_raise(E.sigma_array_sha256, np.array([]))
    assert "空" in msg, msg
    assert E.sigma_block(kind="none", mode="free")["array_sha256"] is None
    msg2 = _expect_raise(E.sigma_block, kind="perpoint", mode="perpoint", sigma=None)
    assert "None" in msg2 or "必须" in msg2, msg2
    msg3 = _expect_raise(E.sigma_block, kind="none", mode="free",
                         sigma=np.array([[1.0], [1.0]]))
    assert "None" in msg3, msg3
    return "S4 空数组报错；kind=none ⇒ None；kind≠none 缺数组报错"


def case_S5_dtype_endian():
    """字节序不影响摘要；`float32` 与 `float64` **应当**不同。

    这是我原来写错的一格：`np.float32(0.1)` 转成 f8 是 0.10000000149011612，不是 0.1。
    摘要必须认"实际数组的值"，所以 float32 与 float64 不同号才是对的；
    同 dtype、不同字节序（`>f8`）才是"同一个值两种写法"，必须同摘要。
    """
    a32 = np.array([0.1, 0.2], dtype=np.float32)
    assert E.sigma_array_sha256(a32) == E.sigma_array_sha256(a32.astype("<f8")), \
        "同值不同存储 dtype 应同摘要"
    b = np.array([0.1, 0.2], dtype="<f8")
    c = np.array([0.1, 0.2], dtype=">f8")
    assert E.sigma_array_sha256(b) == E.sigma_array_sha256(c), "字节序不该影响摘要"
    assert E.sigma_array_sha256(a32) != E.sigma_array_sha256(b), \
        "float32(0.1) 与 float64(0.1) 是不同的数，不应同摘要"
    return "S5 字节序无关；float32 ≠ float64（值确实不同）"


def case_S6_c_order_canonical():
    """C 顺序是**归一化**，不是敏感性：同一逻辑数组换内存布局必须同摘要。

    原来这一格写成了"C/F 布局不同 ⇒ 摘要不同"，方向反了——若真那样，
    同一次运行换一下 `np.asfortranarray` 就会换号。真正要测的是：
    值相同 ⇒ 摘要相同；**元素排布不同**（转置）⇒ 摘要不同。
    """
    a = np.array([[1.0, 2.0], [3.0, 4.0]], order="C")
    f = np.asfortranarray(a)
    assert E.sigma_array_sha256(a) == E.sigma_array_sha256(f), \
        "同一逻辑数组的 C/F 内存布局不该影响摘要"
    t = a.T
    assert E.sigma_array_sha256(a) != E.sigma_array_sha256(t), \
        "转置后元素排布变了，摘要必须变（形状也进了载荷）"
    return "S6 内存布局无关；转置（值排布变了）换号"


def case_S7_sign_positive():
    msg = _expect_raise(E.sigma_block, kind="perpoint", mode="perpoint",
                        sigma=np.array([[0.0, 1.0], [1.0, 1.0]]))
    assert "正" in msg, msg
    return "S7 σ≤0 被拒（Normal.scale 不接受 0）"


# --------------------------------------------------------------------------
# M1–M7 source_manifest.json
# --------------------------------------------------------------------------
_MANIFEST = _ROOT / "source_manifest.json"


def _manifest() -> dict:
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def case_M1_three_profiles():
    d = _manifest()
    assert set(d["profiles"]) == {"A", "B", "C"}, d["profiles"].keys()
    assert d["profiles"]["A"]["entries"] == ["bm_real"]
    assert d["profiles"]["B"]["entries"] == ["nested_fit"]
    assert d["profiles"]["C"]["entries"] == ["bm_select"]
    return "M1 三份 Profile，入口与 §3.3.4 一致"


def case_M2_no_optimistic_demotion():
    d = _manifest()
    bad = []
    for k, prof in d["profiles"].items():
        for it in prof["declared_source_set"]:
            if it.get("compute_role", "numeric") != "numeric":
                bad.append(f"{k}:{it['path']}")
            if it.get("workflow_scope", True) is False:
                bad.append(f"{k}:{it['path']}(wf)")
    assert not bad, f"清单里出现未经真实运行验证的降级：{bad}"
    assert d["defaults"]["compute_role"] == "numeric"
    return "M2 全部 numeric / workflow_scope=True（降级零条）"


def case_M3_paths_exist():
    d = _manifest()
    missing = sorted({it["path"] for prof in d["profiles"].values()
                      for it in prof["declared_source_set"]
                      if not (_ROOT / it["path"]).is_file()})
    assert not missing, f"声明集里有不存在的文件：{missing}"
    return "M3 声明集每个路径都在盘上"


def case_M4_pending_not_applied():
    d = _manifest()
    assert d["pending_demotions"], "降级候选表不该为空"
    for p in d["pending_demotions"]:
        assert p["applied"] is False, f"{p['paths']} 被提前应用了"
        assert p["required_test"].strip(), "降级候选必须写清要什么实测"
    return f"M4 {len(d['pending_demotions'])} 条降级候选，全部 applied=false"


def case_M5_declared_set_valid():
    d = _manifest()
    for k, prof in d["profiles"].items():
        items = [dict(it, sha256=E.source_file_sha256(_ROOT / it["path"]))
                 for it in prof["declared_source_set"]]
        E.check_declared_set(items)               # 入口在位、降级有理由
        sig = (E.declared_source_tree_sha256(items), E.numeric_source_tree_sha256(items),
               E.workflow_source_tree_sha256(items))
        assert len(set(sig)) == 3, "三层摘要撞值：层名域分隔丢了"
    return "M5 三份 Profile 的声明集与三层摘要都可复算"


def case_M6_manifest_matches_source():
    out = subprocess.run([sys.executable,
                          str(_ROOT / "docs/round_2026-11-13/build_source_manifest.py"),
                          "--check"], capture_output=True, text=True)
    assert out.returncode == 0, (out.stdout + out.stderr)[:400]
    return "M6 `--check` rc=0（清单与源码同步）"


def case_M7_demotion_needs_evidence():
    msg = _expect_raise(E.check_declared_set,
                        [{"path": "x.py", "role": "module", "compute_role": "plot_only"}])
    assert "why_not_numeric" in msg, msg
    msg2 = _expect_raise(E.check_declared_set,
                         [{"path": "x.py", "role": "module", "compute_role": "plot_only",
                           "why_not_numeric": "只画图"}])
    assert "evidence" in msg2, msg2
    msg3 = _expect_raise(E.check_declared_set,
                         [{"path": "x.py", "role": "entry", "workflow_scope": False,
                           "why_not_workflow": "x"}])
    assert "workflow" in msg3, msg3
    return "M7 降级缺 evidence / 入口退出 workflow 层 都被拒"


def case_M8_preview_recomputes():
    """`digest_preview` 不是身份，但必须能从文件复算——否则它只是一句假话。"""
    d = _manifest()
    for k, prof in d["profiles"].items():
        items = [dict(it, sha256=E.source_file_sha256(_ROOT / it["path"]))
                 for it in prof["declared_source_set"]]
        pv = prof["digest_preview"]
        assert pv["declared"] == E.declared_source_tree_sha256(items), f"{k}: declared 预览过期"
        assert pv["numeric"] == E.numeric_source_tree_sha256(items), f"{k}: numeric 预览过期"
        assert pv["workflow"] == E.workflow_source_tree_sha256(items), f"{k}: workflow 预览过期"
    return "M8 三份 Profile 的三层预览都可从文件复算"


# --------------------------------------------------------------------------
# Y1–Y4 状态阶梯：六个块各自缺一项时降哪一档
# --------------------------------------------------------------------------
def case_Y1_env_digest():
    r = _rec()
    r2 = _set(copy.deepcopy(r), "env.env_sha256", _sha("other-env"))
    ev = E.evaluate(r2)
    assert ev["status"] == "derived", f"env 摘要对不上却 recorded：{ev}"
    assert "env_digest_mismatch" in ev["reasons"], ev["reasons"]
    assert ev["run_id"].startswith("D1-"), ev["run_id"]
    return f"Y1 env 摘要不符 ⇒ derived（{ev['reasons']}）"


def case_Y2_config_unclassified():
    r = _rec()
    _set(r, "config.effective.mystery_switch",
         {"value": True, "origin": "cli", "effect": "display_only"})
    ev = E.evaluate(r)
    assert ev["status"] == "derived", f"未登记键却 recorded：{ev}"
    assert any(x.startswith("config_unclassified") for x in ev["reasons"]), ev["reasons"]
    return f"Y2 未登记配置键 ⇒ derived（{ev['reasons'][-1]}）"


def case_Y3_budget_not_recorded():
    r = _rec()
    r2 = _set(copy.deepcopy(r), "budget_params.target_accept",
              {"value": 0.98, "source": "derived_from_tier"})
    ev = E.evaluate(r2)
    assert ev["status"] == "unconfirmed", f"预算不是 recorded 却发了号：{ev}"
    assert ev["run_id"] is None, "unconfirmed 不许发 run_id（发了就是编造）"
    return f"Y3 预算 derived_from_tier ⇒ unconfirmed、无号（{ev['reasons']}）"


def case_Y4_rng_plan_missing():
    r = _rec()
    r2 = copy.deepcopy(r)
    del r2["rng"]["plan"]["init_strategy"]
    ev = E.evaluate(r2)
    assert ev["status"] == "unconfirmed" and ev["run_id"] is None, ev
    assert any("rng_plan_missing" in x for x in ev["reasons"]), ev["reasons"]
    r3 = _set(copy.deepcopy(r), "rng.plan.n_chains", 4)      # 与 budget 不一致
    ev3 = E.evaluate(r3)
    assert ev3["status"] == "unconfirmed", ev3
    assert "rng_budget_mismatch" in ev3["reasons"], ev3["reasons"]
    return "Y4 rng.plan 缺项 / n_chains 与预算不符 ⇒ unconfirmed"


# --------------------------------------------------------------------------
# X 跨模块一致性
# --------------------------------------------------------------------------
def case_X1_sha_semantics_match_registry():
    good = "A" * 64
    assert E.valid_sha256(good) and R._valid_sha256(good)
    assert E.canon_sha256(good) == good.lower()
    for bad in ("same", "", None, "a" * 63, "a" * 65, "g" * 64, 123):
        assert E.valid_sha256(bad) == R._valid_sha256(bad), f"与 registry 口径分叉：{bad!r}"
    return "X1 `valid_sha256` 与 registry `_valid_sha256` 逐例一致"


def case_X2_v0_record_rejected():
    r = _rec()
    del r["identity_contract_version"]
    msg = _expect_raise(E.run_id_payload, r)
    assert "identity_contract_version" in msg, msg
    r2 = _set(_rec(), "identity_contract_version", "1.0")
    msg2 = _expect_raise(E.run_id_payload, r2)
    assert "1.0" in msg2 or "identity_contract_version" in msg2, msg2
    return "X2 缺版本 / v1.0 记录不许套 v1.2 载荷"


def case_X4_voigt_K_dual_effect():
    """`voigt_K` 是唯一"同一个键两种 `effect`"的键：父=workflow_scope，子=model_definition。"""
    assert E.config_effect("voigt_K", role="parent") == "workflow_scope"
    assert E.config_effect("voigt_K", role="child") == "model_definition"
    r = _rec()
    _set(r, "config.effective.voigt_K", {"value": 3, "origin": "cli",
                                         "effect": "model_definition"})
    parent_before = E.run_id_digest(r, role="parent")
    parent_after = E.run_id_digest(_set(copy.deepcopy(r), "config.effective.voigt_K.value", 4),
                                   role="parent")
    child_after = E.run_id_digest(_set(copy.deepcopy(r), "config.effective.voigt_K.value", 4),
                                  role="child")
    assert parent_before == parent_after, "父运行里 voigt_K 不该进 run_id"
    assert child_after != E.run_id_digest(_set(copy.deepcopy(r), "config.effective.voigt_K",
                                              {"value": 3, "origin": "cli",
                                               "effect": "model_definition"}), role="child"), \
        "子运行里 voigt_K 就是模型定义，必须换号"
    return "X4 voigt_K：父不进 run_id、子进（同一键两种 effect）"


def case_X3_determinism():
    r = _rec()
    assert E.run_id_digest(r) == E.run_id_digest(copy.deepcopy(r)), "同一记录两次算号不同"
    payload = E.canonical_json(E.run_id_payload(r))
    assert "NaN" not in payload and "Infinity" not in payload
    return "X3 同一记录：摘要稳定、载荷无 NaN/Inf"


# --------------------------------------------------------------------------
CASES = [
    ("T1 branch", case_T1_branch),
    ("T2 include_L", case_T2_include_L),
    ("T3 layers", case_T3_layers),
    ("T4 adsorption", case_T4_adsorption),
    ("T5 progress", case_T5_progress),
    ("T6 plot", case_T6_plot),
    ("T7 instance", case_T7_instance),
    ("T8 init_values", case_T8_init_values),
    ("T9 numeric 常量", case_T9_numeric_change),
    ("T10 plot_only", case_T10_plot_only),
    ("T11 registry_version", case_T11_registry_version),
    ("T12 换数据", case_T12_other_file),
    ("T13 掩码/点数", case_T13_subset),
    ("T14 zscale", case_T14_zscale),
    ("T15 预算档", case_T15_budget_tier),
    ("T16 with_voigt", case_T16_with_voigt),
    ("T17 kk_mode", case_T17_kk_mode),
    ("T18 diag", case_T18_diag),
    ("T19 sigma 数组", case_T19_sigma_array_not_mode),
    ("T19b sigma kind", case_T19b_sigma_none_vs_block),
    ("T20 _UNSET", case_T20_unset_resolved),
    ("T21 zeta", case_T21_zeta_norm),
    ("T22 声明未加载", case_T22_declared_not_loaded),
    ("T23 未声明加载", case_T23_undeclared_loaded),
    ("T24 补丁包缺口", case_T24_bundle_gap),
    ("W1 decision 模块", case_W1_decision_module),
    ("W2 入口重跑逻辑", case_W2_entry_retry),
    ("W3 退出 workflow 层", case_W3_plot_out_of_workflow),
    ("W4 门禁结果", case_W4_gate_result_not_identity),
    ("W5 children", case_W5_children),
    ("W6 搜索空间", case_W6_search_space),
    ("S1 f8/C/形状", case_S1_f8_c_order),
    ("S2 形状进载荷", case_S2_shape_in_payload),
    ("S3 NaN/Inf", case_S3_reject_nonfinite),
    ("S4 空/None/kind", case_S4_empty_vs_none_vs_kind),
    ("S5 dtype/字节序", case_S5_dtype_endian),
    ("S6 C 顺序归一", case_S6_c_order_canonical),
    ("S7 σ>0", case_S7_sign_positive),
    ("M1 三份 Profile", case_M1_three_profiles),
    ("M2 无乐观降级", case_M2_no_optimistic_demotion),
    ("M3 路径存在", case_M3_paths_exist),
    ("M4 降级未应用", case_M4_pending_not_applied),
    ("M5 声明集合法", case_M5_declared_set_valid),
    ("M6 清单同步", case_M6_manifest_matches_source),
    ("M7 降级要证据", case_M7_demotion_needs_evidence),
    ("Y1 env 摘要", case_Y1_env_digest),
    ("Y2 配置未登记", case_Y2_config_unclassified),
    ("Y3 预算未记录", case_Y3_budget_not_recorded),
    ("Y4 rng.plan 缺项", case_Y4_rng_plan_missing),
    ("M8 预览可复算", case_M8_preview_recomputes),
    ("X1 sha 口径", case_X1_sha_semantics_match_registry),
    ("X4 voigt_K 双效", case_X4_voigt_K_dual_effect),
    ("X2 版本隔离", case_X2_v0_record_rejected),
    ("X3 确定性", case_X3_determinism),
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
