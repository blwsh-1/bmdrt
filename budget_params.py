# -*- coding: utf-8 -*-
"""`budget_params` v1.4 产出层：五项实际生效预算的纯函数构造与校验。

本模块**只做构造与判定，不做采集**：不 import numpyro / exec_identity、
不读环境变量、不碰磁盘、不看时间、不构造 `run_id` / `workflow_id`。
输入是"有人已经在构造实参处采到了什么"，输出是"这份记录能不能被当成执行条件"。
真正的观测点在调用边界上，属于第 3 步的独立探针，本轮不实现。

三个必须可区分的概念（Q1–Q6 裁决）
------------------------------------
    · 预算数值      `value`      最终**实际传入采样器**的构造实参，不是档名 / CLI 原值 /
                                 `BUDGETS` 查表值 / `KW` 模块初值 / 函数签名默认值
    · 预算来源证据  `source`     逐项**证据等级**，闭集三值；与 `value` 一起进预算哈希
    · 来源追溯过程  `origin` / `requested` / `transformed` / `where`
                                 只作溯源记录，**一律不进哈希**

`source` 与 `origin` 不能互顶：`source` 回答"这条值有没有可接受的记录证据"，
`origin` 回答"这个值从哪一类位置产生"。具体变量名、函数名、代码位置放 `where`，
不塞进 `origin` 枚举——否则"来源写法"会冒充"执行条件"（D-8 同构）。

两套身份契约用在不同层级（不得互相转换）
------------------------------------------
registry 0.7.0 的 `source` 是**块级**来源类别（整组预算记录从哪来），
v1.4 的 `source` 是**逐项**证据等级。把一个块级 `source="recorded"` 机械复制到
五个参数上，等于把"这组记录有来源"伪装成"五项各自都核验过"。所以本层见到
registry 扁平形状一律拒绝；旧记录迁移只能做显式、可追溯的映射，且不得声称
`R-/D-`(10hex) 与 `R1-/D1-`(64hex) 等价。

v1.4 的两处实现事实（本层如实反映，不擅自改 `exec_identity.py`）
------------------------------------------------------------------
1. `_chk_budget()` 逐项要求 `source == "recorded"`，任一不满足 ⇒ `fatal=True`
   ⇒ 整体 `status="unconfirmed"` **不发 `run_id`**，不是降 `derived`。
2. `RUN_ID_CONFIG_KEYS` 当前**未登记** `chain_method`。Q6 裁决把它归入
   `config.effective` 作为执行条件，但 `config_keys_in_run_id()` 只认登记键，
   所以本轮放进 `config.effective` **暂时不产生 `run_id` 影响**。让裁决真正生效
   需要第 4 步升 `RUN_ID_CONFIG_KEYS_VERSION` 并登记；本轮不改身份模块。

硬边界（用户实施前追加）
------------------------
即使五项最终值都捕获成功，只要 `KW` 运行时修改、默认值来源等上游链未完整追溯
（`upstream_origin_captured` 不为显式 `True`），任何一项都不许标 `source="recorded"`。
本层把这个约束做成必填参数：省略 / `None` / truthy 非布尔一律 raise。

    python budget_params.py      # 跑内置自检
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

BUDGET_PARAMS_VERSION = "1.0"

#: Q1：五项核心是 `budget_params` 的**全部**。与 `exec_identity.BUDGET_KEYS` 逐位一致（X5 锁）。
BUDGET_KEYS = ("n_warmup", "n_samples", "num_chains", "target_accept",
               "max_tree_depth")

#: Q3a：逐项**证据等级**。`recorded` = 最终值及其来源链都被实际捕获、核验；
#: `derived` = 最终值已观测但上游来源链有缺口；`unavailable` = 最终值本身无法可靠取得。
BUDGET_SOURCE_STATES = ("recorded", "derived", "unavailable")

#: Q3b：逐项**来源类别**，不进哈希。具体位置放 `where`。
BUDGET_ORIGINS = (
    "cli_explicit",             # 命令行显式给出的值
    "module_constant",          # `bm_real.KW` / `BUDGETS` 等模块常量初值
    "programmatic_override",    # 外部脚本运行内改了可变 dict
    "function_default",         # 缺键落到 `fit_top` / `bm_core.fit` 签名默认
    "library_default",          # 落到 numpyro 库签名默认
    "runtime_derived",          # 设备数 / 环境变量等在运行时派生
    "unknown",                  # 来源链有缺口，不足以支撑 `recorded`
)

#: Q1/Q6：三个扩展字段不是五项预算数值，留在捕获事件 / `config.effective`。
EXTENSION_FIELDS = ("chain_method", "progress_bar", "extra_fields")

#: Q4：进 `run_id` 预算哈希的键。v1.4 `_budget_payload()` 实际读取的也只有这两个。
HASH_KEYS = ("value", "source")
#: Q4：只作溯源记录，绝不进哈希。
TRACE_KEYS = ("origin", "requested", "transformed", "where")
#: 每项恰好这六个键，顺序固定；多 / 少都拒（防止从块级 `source` 继承）。
CELL_KEYS = HASH_KEYS + TRACE_KEYS

#: Q6 分流。`in_run_id` 是**契约意图**，`registered_in_v14` 是**当前实现事实**，
#: 两者分开写，不许把"打算纳入"说成"已经纳入"。
EXECUTION_CHANNELS = {
    "chain_method": {"record": "config.effective", "in_run_id": True,
                     "registered_in_v14": False,
                     "note": "parallel 被设备数改成 sequential 会改变实际执行方式；"
                             "当前无逐位等价性证据，保守纳入执行条件"},
    "progress_bar": {"record": "sampler_capture_event", "in_run_id": False,
                     "registered_in_v14": False,
                     "note": "identity 表中 `progress` = display_only；CI 覆盖造成的 "
                             "mismatch 是证据链记录，不是执行条件"},
    "extra_fields": {"record": "mcmc_run_event", "in_run_id": False,
                     "registered_in_v14": False,
                     "note": "只影响诊断数据采集，在 `MCMC.run` 调用边界观测"},
}

#: registry 0.7.0 的块级字段；出现在 v1.4 形状里一律拒。
LEGACY_BLOCK_MARKERS = ("source", "tier", "derived_from_tier", "reason",
                        "budget_profile", "profile")
LEGACY_SOURCES = ("recorded", "derived_from_tier", "unavailable")

#: 两套号属于不同契约世代，禁止直接拿前缀或摘要比对来判断历史记录漂移。
IDENTITY_GENERATIONS = {
    "R-": "registry_0.7.0", "D-": "registry_0.7.0",
    "R1-": "exec_identity_1.4", "D1-": "exec_identity_1.4",
}

_STATE_RANK = {"recorded": 0, "derived": 1, "unavailable": 2}
_MISSING = object()


# ---------------------------------------------------------------------------
# 内部校验
# ---------------------------------------------------------------------------

def _check_upstream(flag: Any) -> None:
    """`upstream_origin_captured` 必须显式传布尔；省略 / None / truthy 都拒。

    用哨兵默认值而不是 `False`：默认 `False` 会让"忘了传"看起来像"已确认未闭合"，
    正是阶段 B `adapt_step_size` 缺参漏洞的同构。
    """
    if flag is _MISSING:
        raise ValueError("upstream_origin_captured 必须显式传入（不许省略）")
    if not isinstance(flag, bool):
        raise ValueError(
            f"upstream_origin_captured 必须是显式布尔，收到 {flag!r}")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v: Any) -> bool:
    if not _is_number(v):
        return False
    try:
        return math.isfinite(float(v))
    except (OverflowError, ValueError):
        return False


def _jsonish(v: Any, *, depth: int = 0) -> bool:
    """`requested` / 转换步骤值允许的形状：标量或一层标量序列。

    拒 Mapping（不许把一整块配置塞进溯源字段）、拒 NaN/Inf。
    """
    if v is None or isinstance(v, (bool, str)):
        return True
    if _is_number(v):
        return _finite(v)
    if isinstance(v, (list, tuple)) and depth == 0:
        return all(_jsonish(x, depth=depth + 1) for x in v)
    return False


def _check_value(value: Any, *, key: str, source: str) -> None:
    if source == "unavailable":
        if value is not None:
            raise ValueError(
                f"budget_params[{key!r}]：source='unavailable' 时 value 必须是 None"
                f"（最终值无法可靠取得），收到 {value!r}")
        return
    if not _finite(value):
        raise ValueError(
            f"budget_params[{key!r}].value 必须是有限数值（非 bool / None / NaN / Inf），"
            f"收到 {value!r}；它必须是最终实际构造实参")


def _check_requested(requested: Any, *, key: str) -> Any:
    if not _jsonish(requested):
        raise ValueError(
            f"budget_params[{key!r}].requested 只能是标量 / 标量序列 / None，"
            f"不许塞整块配置，收到 {requested!r}")
    return list(requested) if isinstance(requested, tuple) else requested


def _check_transformed(transformed: Any, *, key: str) -> list[dict]:
    """Q4：`transformed` 必须是**列表**，保留多次覆盖的顺序，不压缩成单个中间值。"""
    if isinstance(transformed, (str, bytes, Mapping)) or not isinstance(transformed, (list, tuple)):
        raise ValueError(
            f"budget_params[{key!r}].transformed 必须是列表（保留多次覆盖顺序），"
            f"收到 {type(transformed).__name__}")
    out: list[dict] = []
    allowed = {"stage", "value", "where"}
    for i, step in enumerate(transformed):
        if not isinstance(step, Mapping):
            raise ValueError(
                f"budget_params[{key!r}].transformed[{i}] 必须是步骤记录 {{stage, value, where?}}")
        extra = [k for k in step if k not in allowed]
        if extra:
            raise ValueError(
                f"budget_params[{key!r}].transformed[{i}] 出现未登记键 {extra}")
        stage = step.get("stage")
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError(
                f"budget_params[{key!r}].transformed[{i}].stage 必须是非空字符串")
        if "value" not in step or not _jsonish(step["value"]):
            raise ValueError(
                f"budget_params[{key!r}].transformed[{i}].value 缺失或不是可记录标量")
        where = step.get("where")
        if where is not None and not isinstance(where, str):
            raise ValueError(
                f"budget_params[{key!r}].transformed[{i}].where 必须是字符串或 None")
        out.append({"stage": stage, "value": step["value"], "where": where})
    return out


def _check_where(where: Any, *, key: str) -> Any:
    if where is None:
        return None
    if not isinstance(where, str) or not where.strip():
        raise ValueError(
            f"budget_params[{key!r}].where 必须是非空字符串或 None，收到 {where!r}")
    return where


def _validate_cell(cell: Any, key: str, *, upstream_origin_captured: bool) -> dict:
    if not isinstance(cell, Mapping):
        raise ValueError(
            f"budget_params[{key!r}] 必须是 {{value, source, origin, requested, "
            f"transformed, where}} 复合块，收到裸值 {cell!r}；"
            "registry 0.7.0 的扁平数值块不能直接喂给 v1.4")
    missing = [k for k in CELL_KEYS if k not in cell]
    extra = [k for k in cell if k not in CELL_KEYS]
    if missing or extra:
        raise ValueError(
            f"budget_params[{key!r}] 键必须恰好是 {list(CELL_KEYS)}；缺 {missing}、"
            f"多 {extra}。每项都要有自己的证据判断，不许从块级 source 继承")
    source = cell["source"]
    if source not in BUDGET_SOURCE_STATES:
        raise ValueError(
            f"budget_params[{key!r}].source={source!r} 不在证据等级闭集 "
            f"{BUDGET_SOURCE_STATES}；不得用 registry 的 'derived_from_tier'")
    origin = cell["origin"]
    if origin not in BUDGET_ORIGINS:
        raise ValueError(
            f"budget_params[{key!r}].origin={origin!r} 不在来源类别闭集 {BUDGET_ORIGINS}")
    _check_value(cell["value"], key=key, source=source)
    requested = _check_requested(cell["requested"], key=key)
    transformed = _check_transformed(cell["transformed"], key=key)
    where = _check_where(cell["where"], key=key)
    if source == "recorded":
        if upstream_origin_captured is not True:
            raise ValueError(
                f"budget_params[{key!r}]：upstream_origin_captured 必须为 True 才允许 "
                f"source='recorded'；KW 运行时修改、默认值来源等上游链未完整追溯时"
                f"不得标 recorded（本次 {upstream_origin_captured!r}）")
        if origin == "unknown":
            raise ValueError(
                f"budget_params[{key!r}]：origin='unknown' 时来源链未闭合，"
                f"不许标 source='recorded'")
        if not where:
            raise ValueError(
                f"budget_params[{key!r}]：source='recorded' 要求 where 给出具体位置")
    # 键顺序固定，便于逐位比对与 JSON 直出
    return {
        "value": cell["value"],
        "source": source,
        "origin": origin,
        "requested": requested,
        "transformed": transformed,
        "where": where,
    }


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def make_cell(*, value: Any, source: str, origin: str,
              requested: Any = None, transformed: Any = None,
              where: Any = None) -> dict:
    """构造单项的规范形状（缺省 `requested=None`、`transformed=[]`）。

    只保证形状，不做**全局**证据门槛；`recorded` 是否成立由
    `validate_budget_params(..., upstream_origin_captured=...)` 统一判定。
    """
    return {
        "value": value,
        "source": source,
        "origin": origin,
        "requested": requested,
        "transformed": [] if transformed is None else transformed,
        "where": where,
    }


def validate_budget_params(block: Any, *,
                           upstream_origin_captured: Any = _MISSING) -> dict:
    """校验并返回规范化的五项预算块（深拷贝，不改入参）。

    顶层恰好五项；每项恰好六键；拒绝 registry 块级字段、档名、
    三个扩展字段；逐项校验 `source` / `origin` / 数值 / 追溯字段；
    并在 `upstream_origin_captured is not True` 时禁止任何 `recorded`。
    """
    _check_upstream(upstream_origin_captured)
    if not isinstance(block, Mapping):
        raise ValueError("budget_params 必须是五项复合块的映射")
    legacy = [k for k in LEGACY_BLOCK_MARKERS if k in block]
    if legacy:
        raise ValueError(
            f"budget_params 出现 registry 0.7.0 块级字段 {legacy}：块级 source "
            "不能机械复制到五项；旧记录迁移必须显式、可追溯，且不声称 "
            "R-/D- 与 R1-/D1- 等价")
    ext = [k for k in EXTENSION_FIELDS if k in block]
    if ext:
        raise ValueError(
            f"budget_params 不得含扩展字段 {ext}（Q1/Q6）："
            "chain_method / progress_bar / extra_fields 不是五项预算数值")
    unknown = [k for k in block if k not in BUDGET_KEYS]
    if unknown:
        raise ValueError(
            f"budget_params 出现未登记键 {unknown}；档名 / tier / profile "
            "只能放 budget_profile，不能进预算身份")
    absent = [k for k in BUDGET_KEYS if k not in block]
    if absent:
        raise ValueError(f"budget_params 缺五项核心键 {absent}（Q1：五项不可缺）")
    return {k: _validate_cell(block[k], k,
                              upstream_origin_captured=upstream_origin_captured)
            for k in BUDGET_KEYS}


def build_budget_params(cells: Any, *,
                        upstream_origin_captured: Any = _MISSING) -> dict:
    """从**逐项观测记录**构造规范块。

    每项必须自己给出 `value` / `source` / `origin`（`where` 对 `recorded` 必填）；
    `requested`、`transformed` 可省。这样"块级 source 机械复制到五项"天然通不过：
    复制只带 value + source，给不出每项独立的来源证据。
    """
    _check_upstream(upstream_origin_captured)
    if not isinstance(cells, Mapping):
        raise ValueError("观测记录必须是 {预算键: 逐项记录} 的映射")
    bad_top = [k for k in LEGACY_BLOCK_MARKERS + EXTENSION_FIELDS if k in cells]
    if bad_top:
        raise ValueError(f"观测记录出现非预算键 {bad_top}")
    out: dict = {}
    for k, spec in cells.items():
        if k not in BUDGET_KEYS:
            raise ValueError(f"观测记录出现未登记预算键 {k!r}（Q1 只有五项）")
        if not isinstance(spec, Mapping):
            raise ValueError(f"观测记录[{k!r}] 必须是逐项复合记录，收到裸值 {spec!r}")
        absent = [x for x in ("value", "source", "origin") if x not in spec]
        if absent:
            raise ValueError(
                f"观测记录[{k!r}] 缺 {absent}：每项都要有自己的证据判断，"
                "不许从块级 source 继承")
        unknown = [x for x in spec if x not in CELL_KEYS]
        if unknown:
            raise ValueError(f"观测记录[{k!r}] 出现未登记字段 {unknown}")
        out[k] = make_cell(
            value=spec["value"], source=spec["source"], origin=spec["origin"],
            requested=spec.get("requested"), transformed=spec.get("transformed"),
            where=spec.get("where"))
    return validate_budget_params(out, upstream_origin_captured=upstream_origin_captured)


def budget_hash_payload(block: Any, *,
                        upstream_origin_captured: Any = _MISSING) -> dict:
    """投影成 v1.4 `_budget_payload()` 期望的 `{key: {value, source}}`。

    **只有 value 与 source 进预算哈希**：`origin` / `requested` /
    `transformed` / `where` 一律不出现。同一最终执行条件不应因入口写法不同而换号。
    注意这只管 `run_id` 载荷里的 `budget` 段，不代表整个 `run_id` 只有这两个键。
    """
    clean = validate_budget_params(
        block, upstream_origin_captured=upstream_origin_captured)
    return {k: {"value": clean[k]["value"], "source": clean[k]["source"]}
            for k in BUDGET_KEYS}


def budget_evidence_state(block: Any, *,
                          upstream_origin_captured: Any = _MISSING) -> dict:
    """五项证据等级的最弱项，以及它在 v1.4 里的实际后果。

    如实写明：v1.4 `_chk_budget()` 逐项要求 `recorded`，任一不满足 ⇒
    `status='unconfirmed'` **且不发 run_id**，不是"降 derived 发个号"。
    """
    clean = validate_budget_params(
        block, upstream_origin_captured=upstream_origin_captured)
    per = {k: clean[k]["source"] for k in BUDGET_KEYS}
    state = max(per.values(), key=lambda s: _STATE_RANK[s])
    all_recorded = state == "recorded"
    return {
        "state": state,
        "per_key": per,
        "weakest_keys": [k for k in BUDGET_KEYS if per[k] == state],
        "supports_v14_recorded": all_recorded,
        "v14_consequence": "recorded" if all_recorded else "unconfirmed:no_run_id",
        "v14_note": (
            "v1.4 _chk_budget() 逐项要求 source=='recorded'；任一不满足 ⇒ "
            "status='unconfirmed' 且不发 run_id（不是降 derived）"),
    }


def check_retry_kw(retry_kw: Any) -> dict:
    """Q5：`workflow.retry_kw` 禁止重复承载五项预算键。

    预算数值只由**该次子运行**的 `budget_params` 进 `run_id`；重跑策略
    （retry / 触发条件 / 次数）归 `workflow_id`；`children` 只放身份引用。
    """
    if retry_kw is None:
        retry_kw = {}
    if not isinstance(retry_kw, Mapping):
        raise ValueError("workflow.retry_kw 必须是映射或 None")
    found = [k for k in BUDGET_KEYS if k in retry_kw]
    if found:
        raise ValueError(
            f"workflow.retry_kw 不得重复承载五项预算键 {found}（Q5）："
            "预算数值只由子运行的 budget_params 进 run_id；重跑改 n_samples "
            "应产生新的子运行身份，children 只放身份引用")
    return {
        "budget_keys_found": [],
        "n_keys": len(retry_kw),
        "allowed_keys": sorted(str(k) for k in retry_kw),
        "note": "重跑策略归 workflow_id；每次实际执行的五项预算归该次子运行 run_id",
    }


def split_extensions(observations: Any) -> dict:
    """Q6 三个扩展字段的分流。**只分类，不构造任何身份号**。

    `chain_method` 按裁决归 `config.effective`（`in_run_id=True`），但 v1.4
    `RUN_ID_CONFIG_KEYS` 未登记它 ⇒ 当前不产生 `run_id` 影响，单列
    `pending_v14_registration` + `warning`，不许把"打算纳入"说成"已经纳入"。
    """
    if observations is None:
        observations = {}
    if not isinstance(observations, Mapping):
        raise ValueError("扩展字段观测必须是映射或 None")
    unknown = [k for k in observations if k not in EXTENSION_FIELDS]
    if unknown:
        raise ValueError(f"不是三个扩展字段：{unknown}（Q1：它们不进 budget_params）")
    buckets: dict[str, dict] = {
        "config.effective": {},
        "sampler_capture_event": {},
        "mcmc_run_event": {},
    }
    for k, v in observations.items():
        buckets[EXECUTION_CHANNELS[k]["record"]][k] = v
    run_id_fields = [k for k in EXTENSION_FIELDS
                     if k in observations and EXECUTION_CHANNELS[k]["in_run_id"]]
    pending = [k for k in run_id_fields
               if not EXECUTION_CHANNELS[k]["registered_in_v14"]]
    return {
        "buckets": buckets,
        "run_id_fields": run_id_fields,
        "non_identity_fields": [k for k in EXTENSION_FIELDS
                                if k in observations
                                and not EXECUTION_CHANNELS[k]["in_run_id"]],
        "pending_v14_registration": pending,
        "warning": (
            "chain_method 按 Q6 裁决归执行条件，但 exec_identity v1.4 的 "
            "RUN_ID_CONFIG_KEYS 未登记它 ⇒ 当前放进 config.effective 不会被 "
            "config_keys_in_run_id() 收进哈希。要让裁决生效须第 4 步升 "
            "RUN_ID_CONFIG_KEYS_VERSION 并登记；本轮不改身份模块。"
        ) if pending else None,
        "constructs_identity": False,
    }


def is_legacy_registry_block(obj: Any) -> bool:
    """判定 registry 0.7.0 扁平块（顶层块级 `source` + 五项裸数值）。"""
    if not isinstance(obj, Mapping):
        return False
    if obj.get("source") not in LEGACY_SOURCES:
        return False
    return any(k in obj and not isinstance(obj[k], Mapping) for k in BUDGET_KEYS)


def id_generation(run_id: Any) -> str | None:
    """号的契约世代；未识别返回 None（长前缀先匹配，避免 `R1-` 被 `R-` 吃掉）。"""
    if not isinstance(run_id, str):
        return None
    for prefix in ("R1-", "D1-", "R-", "D-"):
        if run_id.startswith(prefix):
            return IDENTITY_GENERATIONS[prefix]
    return None


def can_compare_ids(a: Any, b: Any) -> dict:
    """跨世代不比号。同世代也**不声明等价**，只说可按各世代自己的规则比对。"""
    ga, gb = id_generation(a), id_generation(b)
    comparable = ga is not None and gb is not None and ga == gb
    return {
        "comparable": comparable,
        "generation_a": ga,
        "generation_b": gb,
        "reason": (
            "同世代，可按各世代自己的规则比对" if comparable else
            "不同契约世代（registry 0.7.0 的 R-/D- 10hex 与 v1.4 的 "
            "R1-/D1- 64hex）：禁止直接拿前缀或摘要比对来判断历史记录漂移"),
        "declares_equivalence": False,
    }


# ---------------------------------------------------------------------------
# 内置自检：异常必须核理由，且每条反例都配正对照
# ---------------------------------------------------------------------------

def _selftest() -> bool:
    ok = True

    def chk(name: str, cond: bool, got: Any = None) -> None:
        nonlocal ok
        if not cond:
            ok = False
        print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f" -> {got!r}" if got is not None else ""))

    def why_raised(fn: Any) -> str | None:
        try:
            fn()
        except ValueError as exc:
            return str(exc)
        return None

    def derived_cells() -> dict:
        return {k: make_cell(value=v, source="derived", origin="module_constant",
                             where=f"bm_real.KW.{k}")
                for k, v in zip(BUDGET_KEYS, (3000, 1000, 3, 0.9, 13))}

    def recorded_cells() -> dict:
        return {k: make_cell(value=v, source="recorded", origin="module_constant",
                             where=f"bm_real.KW.{k}")
                for k, v in zip(BUDGET_KEYS, (3000, 1000, 3, 0.9, 13))}

    # 1–3 常量与闭集
    chk("Q1 五项闭集", BUDGET_KEYS == ("n_warmup", "n_samples", "num_chains",
                                      "target_accept", "max_tree_depth"))
    chk("Q3 证据等级三值、来源类别七值",
        BUDGET_SOURCE_STATES == ("recorded", "derived", "unavailable")
        and len(BUDGET_ORIGINS) == 7)
    chk("Q4 哈希键只有 value/source，追溯键四",
        HASH_KEYS == ("value", "source")
        and len(TRACE_KEYS) == 4 and len(CELL_KEYS) == 6)

    # 4–5 正例（upstream=False 只能 derived）
    d = derived_cells()
    block = validate_budget_params(d, upstream_origin_captured=False)
    chk("五项 derived + upstream=False ⇒ 通过且键序固定",
        list(block) == list(BUDGET_KEYS)
        and all(list(c) == list(CELL_KEYS) for c in block.values()))
    chk("五项 derived ⇒ 整体 derived，不支持 v1.4 recorded",
        budget_evidence_state(block, upstream_origin_captured=False)["state"] == "derived"
        and budget_evidence_state(block, upstream_origin_captured=False)["v14_consequence"]
        == "unconfirmed:no_run_id")

    # 6–7 哈希投影：正对照 + 反对照
    r = recorded_cells()
    rb = validate_budget_params(r, upstream_origin_captured=True)
    hp = budget_hash_payload(rb, upstream_origin_captured=True)
    chk("哈希投影每项恰好 {value, source}",
        all(list(v) == ["value", "source"] for v in hp.values()) and len(hp) == 5)
    changed = dict(rb)
    changed["n_warmup"] = dict(rb["n_warmup"], origin="library_default",
                               requested=999, where="numpyro.MCMC.__init__",
                               transformed=[{"stage": "library_default", "value": 1000}])
    chk("改 origin/requested/transformed/where ⇒ 哈希逐位不变（正对照）",
        budget_hash_payload(changed, upstream_origin_captured=True) == hp)
    downgraded = dict(rb)
    downgraded["n_warmup"] = dict(rb["n_warmup"], source="derived")
    chk("同数值但 source 降级 ⇒ 哈希变化（证据等级进哈希）",
        budget_hash_payload(downgraded, upstream_origin_captured=True) != hp)

    # 8 transformed 必须是列表、顺序保留
    steps = [{"stage": "module_constant", "value": 3000, "where": "bm_real.KW"},
             {"stage": "quick_override", "value": 800, "where": "bm_real.run"},
             {"stage": "cli_ta_override", "value": 0.98, "where": "bmdrt._apply_cli"}]
    tb = validate_budget_params(
        dict(d, n_warmup=make_cell(value=800, source="derived",
                                   origin="programmatic_override",
                                   transformed=steps, where="bm_core.fit")),
        upstream_origin_captured=False)
    chk("transformed 列表保留多次覆盖顺序",
        [s["stage"] for s in tb["n_warmup"]["transformed"]]
        == ["module_constant", "quick_override", "cli_ta_override"])
    msg = why_raised(lambda: validate_budget_params(
        dict(d, n_warmup=dict(d["n_warmup"], transformed=800)),
        upstream_origin_captured=False))
    chk("transformed 单个值必须 raise 且理由含「列表」",
        msg is not None and "列表" in msg, msg or "没有 raise")

    # 9–11 上游闭合硬边界（反例 + 正对照）
    msg = why_raised(lambda: validate_budget_params(
        r, upstream_origin_captured=False))
    chk("upstream=False + recorded ⇒ raise 且理由含 upstream_origin_captured",
        msg is not None and "upstream_origin_captured" in msg, msg or "没有 raise")
    chk("同五项 + upstream=True 不 raise（正对照）",
        why_raised(lambda: validate_budget_params(r, upstream_origin_captured=True)) is None)
    msg = why_raised(lambda: validate_budget_params(r))
    chk("省略 upstream 必须 raise（不许默认 False）",
        msg is not None and "显式传入" in msg, msg or "没有 raise")
    msg = why_raised(lambda: validate_budget_params(r, upstream_origin_captured=1))
    chk("truthy 非布尔必须 raise",
        msg is not None and "布尔" in msg, msg or "没有 raise")

    # 12 recorded 的逐项证据条件
    msg = why_raised(lambda: validate_budget_params(
        dict(r, n_warmup=dict(r["n_warmup"], origin="unknown")),
        upstream_origin_captured=True))
    chk("origin=unknown 不许 recorded，理由含 origin",
        msg is not None and "origin" in msg, msg or "没有 raise")
    msg = why_raised(lambda: validate_budget_params(
        dict(r, n_warmup=dict(r["n_warmup"], where=None)),
        upstream_origin_captured=True))
    chk("recorded 缺 where 必须 raise", msg is not None and "where" in msg,
        msg or "没有 raise")

    # 13 unavailable
    un = dict(d, n_samples=dict(d["n_samples"], value=None, source="unavailable"))
    chk("unavailable + value=None 合法，整体最弱 unavailable",
        budget_evidence_state(
            validate_budget_params(un, upstream_origin_captured=False),
            upstream_origin_captured=False)["state"] == "unavailable")
    msg = why_raised(lambda: validate_budget_params(
        dict(d, n_samples=dict(d["n_samples"], source="unavailable")),
        upstream_origin_captured=False))
    chk("unavailable 带数值必须 raise，理由含 unavailable",
        msg is not None and "unavailable" in msg, msg or "没有 raise")

    # 14–16 registry 历史兼容
    legacy = {"source": "recorded", "n_warmup": 3000, "n_samples": 1000,
              "num_chains": 3, "target_accept": 0.9, "max_tree_depth": 13}
    chk("registry 扁平块被识别", is_legacy_registry_block(legacy))
    msg = why_raised(lambda: validate_budget_params(
        legacy, upstream_origin_captured=False))
    chk("registry 扁平块喂 v1.4 ⇒ raise 且理由含「块级」",
        msg is not None and "块级" in msg, msg or "没有 raise")
    msg = why_raised(lambda: validate_budget_params(
        dict(d, tier="quick"), upstream_origin_captured=False))
    chk("档名 tier 不许进预算块（derived_from_tier 不作证据）",
        msg is not None and "块级" in msg, msg or "没有 raise")
    cmp_cross = can_compare_ids("R-0123456789", "R1-" + "a" * 64)
    chk("R- 与 R1- 跨世代不可比、恒不声明等价",
        cmp_cross["comparable"] is False
        and "不同契约世代" in cmp_cross["reason"]
        and cmp_cross["declares_equivalence"] is False)
    chk("同世代可比（正对照）",
        can_compare_ids("R1-" + "a" * 64, "R1-" + "b" * 64)["comparable"] is True)

    # 17 块级 source 机械复制：每键只给 value+source 必须被拒
    copied = {k: {"value": v, "source": "recorded"}
              for k, v in zip(BUDGET_KEYS, (3000, 1000, 3, 0.9, 13))}
    msg = why_raised(lambda: build_budget_params(copied, upstream_origin_captured=True))
    chk("块级 source 复制到五项（缺 origin/where）必须 raise",
        msg is not None and ("证据判断" in msg or "origin" in msg), msg or "没有 raise")

    # 18 Q5 retry_kw
    msg = why_raised(lambda: check_retry_kw({"n_samples": 2000, "retry": 2}))
    chk("retry_kw 含 n_samples ⇒ raise 且理由含键名",
        msg is not None and "n_samples" in msg, msg or "没有 raise")
    chk("只含重跑策略的 retry_kw 合法（正对照）",
        why_raised(lambda: check_retry_kw({"retry": 2, "trigger": "acf"})) is None)

    # 19 Q6 分流
    sp = split_extensions({"chain_method": "sequential", "progress_bar": False,
                           "extra_fields": ("diverging",)})
    chk("Q6 三通道分流正确且 chain_method 单列待登记",
        sp["buckets"]["config.effective"] == {"chain_method": "sequential"}
        and sp["buckets"]["sampler_capture_event"] == {"progress_bar": False}
        and sp["buckets"]["mcmc_run_event"] == {"extra_fields": ("diverging",)}
        and sp["pending_v14_registration"] == ["chain_method"]
        and sp["constructs_identity"] is False)

    # 20 扩展字段不许塞进预算块
    msg = why_raised(lambda: validate_budget_params(
        dict(d, chain_method=make_cell(value="parallel", source="derived",
                                       origin="library_default")),
        upstream_origin_captured=False))
    chk("扩展字段进 budget_params 必须 raise",
        msg is not None and "扩展字段" in msg, msg or "没有 raise")

    # 21 不改入参
    before = {k: dict(v) for k, v in d.items()}
    validate_budget_params(d, upstream_origin_captured=False)
    chk("校验不改入参", d == before)

    print(f"\n自检{'通过' if ok else '失败'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
