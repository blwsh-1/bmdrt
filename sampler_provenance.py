# -*- coding: utf-8 -*-
"""采样参数来源契约（sampler provenance）：八项配置的请求 / 解析 / 观测三态。

这个模块只做判定，**不做采集**：不 import numpyro、不读环境变量、不碰磁盘、不看时间。
它的输入是"有人已经采到了什么"，输出是"这条值能不能当证据"。真正的观测点在
调用边界上，由之后的捕获层提供（本轮不做）。

为什么必须有这一层
------------------
`bm_real.py:1172` 的 `sampling` 块注释写着「实际生效的采样配置（不是命令行请求值）」，
但它读的是 `kw`——解析后的**声明值**，不是从 `mc` 读的。今天两者恰好相等，
所以是"对的结论 + 错的证据链"。这一层要回答的正是**证据链**：
请求值、解析值、生效值三者各自在哪、谁被改过、改了算什么。

四个概念与来源（不新造第五套词）
-------------------------------
· `requested`：用户/档位请求的值 + 出处（`origin`）。可能没有（全默认）。
· `resolved`：解析后的**权威**值（数值，不是档名）。`origin ∈ ORIGINS`。
· `observed`：在**契约点**读到的值，带 `how`（通道）与 `where`（位置）。
· `status`：以上三者的**证据关系**，五态。

八项配置与观测通道（实测，见 `docs/round_2026-11-14/SAMPLER_PROVENANCE_AUDIT_B.md`）
--------------------------------------------------------------------------------
· `n_warmup` / `n_samples` / `num_chains` → `attr`，`mc.<name>`；属性即请求值。
· `target_accept` / `max_tree_depth` → `attr`，但在**内核**上
  （`mc.sampler._target_accept_prob` / `_max_tree_depth`），不在 `mc` 上。
· `chain_method` / `progress_bar` → `attr`，但属性只保存**被覆盖后的生效值**；
  请求值不在对象里（`_chain_method` 之类的键一个都没有），所以请求值必须由
  构造边界另外提供。
· `extra_fields` → **没有属性**，只活在 `MCMC.run` 的栈帧里 ⇒ 只能 `call_arg`。
  `get_extra_fields().keys()` 反推属于 `derived`，**不许标成 `captured`**。

两个字段，两种职责
------------------
    observed_status：观测过程成不成功、属性适不适用
                     captured / attribute_missing / hook_not_reached /
                     capture_error / not_applicable
    status         ：请求、解析、观测三者的证据关系
                     complete / defaulted / derived / mismatch / incomplete

混成一个字段的后果是「采集失败」与「参数被悄悄改写」同码——一个该重试，
一个该查代码，处理方式完全不同。

适用性先于值比较（硬约束）
--------------------------
`HMC.__init__` 把 `_max_tree_depth` 硬编码成 **10**，它**不是**任何人的设置。
若不先判适用性就去比数值，换内核会被报成"传递链被篡改"（假 `mismatch`）。
所以顺序固定：**适用性 → 观测通道 → 值比较 → origin**。
不适用记 `not_applicable`，跳过比较，`status` 落 `derived`，并**仍然参与最弱项聚合**
（不适用不能让整体看起来更完整）。

    python sampler_provenance.py      # 跑内置自检
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

PROVENANCE_CONTRACT_VERSION = "1.0"

#: 八项配置。前五项是"预算"，后三项是"执行条件"——它们同样影响记录的含义，
#: 只是不进数值模型。
CORE_FIELDS = ("n_warmup", "n_samples", "num_chains", "target_accept",
               "max_tree_depth")
EXEC_FIELDS = ("chain_method", "progress_bar", "extra_fields")
FIELDS = CORE_FIELDS + EXEC_FIELDS

#: 值的出处。`tier` 是档名（`quick`/`standard`）——档名只是标签，不是值。
ORIGINS = ("cli", "config_file", "default", "tier", "derived")

#: 观测过程状态（语义独立于 `status`）。
OBSERVED_STATUSES = ("captured", "attribute_missing", "hook_not_reached",
                     "capture_error", "not_applicable")
#: 证据关系状态。
STATUSES = ("complete", "defaulted", "derived", "mismatch", "incomplete")
#: 聚合用：越大越弱。`not_applicable` 不在表里——它先映射成 `derived`。
STATUS_RANK = {"complete": 0, "defaulted": 1, "derived": 2, "mismatch": 3,
               "incomplete": 4}
#: 观测**失败**（不是"不适用"）：这几档一律 `incomplete`，且不许用 `resolved` 顶替。
_FAILED_OBS = ("attribute_missing", "hook_not_reached", "capture_error")

#: 内核白名单。判别必须由捕获层用 `isinstance(sampler, NUTS)` 做——
#: **不许用 `hasattr(sampler, "_max_tree_depth")`**：HMC 也有这个属性（值恒 10）。
KERNELS = ("NUTS", "HMC")

#: 契约观测点。捕获层应当直接引用这里的常量，别自己拼字符串。
OBSERVE_AT = {
    "n_warmup": "mcmc.num_warmup",
    "n_samples": "mcmc.num_samples",
    "num_chains": "mcmc.num_chains",
    "target_accept": "mcmc.sampler._target_accept_prob",
    "max_tree_depth": "mcmc.sampler._max_tree_depth",
    "chain_method": "mcmc.chain_method",
    "progress_bar": "mcmc.progress_bar",
    "extra_fields": "MCMC.run.extra_fields",
}

#: 每个字段允许的观测通道。只有 `extra_fields` 允许 `derived`（且只作为降级档）。
HOW_BY_FIELD = {
    "n_warmup": ("attr",),
    "n_samples": ("attr",),
    "num_chains": ("attr",),
    "target_accept": ("attr",),
    "max_tree_depth": ("attr",),
    "chain_method": ("attr",),
    "progress_bar": ("attr",),
    "extra_fields": ("call_arg", "derived"),
}

#: 原因码闭集。它不是状态：状态说"弱不弱"，原因码说"为什么弱"。
#: `sequentially_resolved` / `forced_by_env` 属于 schema 内的偏1移——
#: 值可能完全正确，但执行条件与声明不等价，必须能被筛选出来。
REASON_CODES = (
    "sequentially_resolved",     # chain_method 被 local_device_count 改成 sequential
    "forced_by_env",             # progress_bar 被 CI / PYTEST_XDIST_WORKER 关掉
    "single_chain",              # num_chains == 1 ⇒ chain_method 不被 run 读取
    "kernel_inapplicable",       # 非 NUTS 内核 ⇒ max_tree_depth 不是设置值
    "step_size_not_adapting",    # adapt_step_size=False ⇒ target_accept 不被消费
    "declared_not_delivered",    # 声明里有、实际没传到（B 路径的 extra_fields）
    "inferred_only",             # 只有反推，没有真观测
    "origin_default_unsourced",  # origin=default 但没有出处 ⇒ 不能当"源码默认"
    "observation_missing",       # 该采没采到
    "observed_nan",              # 采到了个 NaN（哨兵值不是值）
    "resolved_missing",          # 只知道档名，没有数值
    "value_differs",             # 请求/解析与观测不等
)

#: 属于"执行条件偏移"的原因码：状态是 `mismatch`，但性质是"条件不等价"，
#: 不是"证据链断裂"。聚合里单独计数，避免两类混在一起没法处置。
OVERRIDE_CODES = ("sequentially_resolved", "forced_by_env")


# ---------------------------------------------------------------------------
# 适用性（R1–R6）：必须在值比较之前判定
# ---------------------------------------------------------------------------

def applicability(name: str, *, kernel: str | None = None,
                  num_chains: int | None = None,
                  adapt_step_size: bool | None = True) -> dict:
    """该字段在当前配置下**是否适用**。返回 `{applicable, reason}`。

    · `max_tree_depth`（R2/R3）：只在 NUTS 上适用。HMC 没有这个形参，
      `_max_tree_depth` 是构造函数里的字面量 10 ⇒ `kernel_inapplicable`。
      `kernel` 必须显式给（`None` 直接 raise）：默认值在这里正好是错的那一边。
    · `chain_method`（R5）：`num_chains == 1` 时 `MCMC.run` 走单链分支，
      根本不读它 ⇒ `single_chain`。属性读得出 `parallel`，但没有语义。
    · `target_accept`（R1）：只在步长自适应开启时被消费；关掉后属性还在 ⇒
      `step_size_not_adapting`。
    · 其余四项恒适用。
    """
    if name not in FIELDS:
        raise ValueError(f"applicability: 未知字段 {name!r}，可选 {FIELDS}")
    if name == "max_tree_depth":
        if kernel is None:
            raise ValueError(
                "applicability: max_tree_depth 必须显式给 kernel——"
                "HMC 的 _max_tree_depth 恒为 10，不是用户设置；"
                "用 hasattr 判别会把它当成'采到了'")
        if kernel not in KERNELS:
            raise ValueError(f"applicability: kernel 只能是 {KERNELS}，收到 {kernel!r}")
        ok = kernel == "NUTS"
        return {"applicable": ok, "reason": None if ok else "kernel_inapplicable"}
    if name == "chain_method":
        if num_chains is None:
            raise ValueError("applicability: chain_method 必须显式给 num_chains")
        if not isinstance(num_chains, int) or num_chains < 1:
            raise ValueError(f"applicability: num_chains 必须是正整数，收到 {num_chains!r}")
        ok = num_chains > 1
        return {"applicable": ok, "reason": None if ok else "single_chain"}
    if name == "target_accept":
        ok = bool(adapt_step_size)
        return {"applicable": ok,
                "reason": None if ok else "step_size_not_adapting"}
    return {"applicable": True, "reason": None}


def tree_budget(max_tree_depth: int) -> int:
    """NUTS 在一次迭代里最多能走的 leapfrog 步数上界（`2**d - 1`）。

    depth 13 与 10 不是"记个数字不同"：8191 vs 1023，轨迹预算差 8 倍。
    """
    if not isinstance(max_tree_depth, int) or max_tree_depth < 1:
        raise ValueError(f"tree_budget: 需要正整数，收到 {max_tree_depth!r}")
    return 2 ** max_tree_depth - 1


# ---------------------------------------------------------------------------
# 观测
# ---------------------------------------------------------------------------

def observe(name: str, value: Any, *, how: str = "attr",
            where: str | None = None) -> dict:
    """构造一条 `observed`。**通道决定位置，位置不许自己发明。**

    · `how="attr"`：`where` 必须是 `OBSERVE_AT[name]`（`mcmc.…`）；
    · `how="call_arg"`：`where` 必须是 `MCMC.run.extra_fields`——调用边界；
    · `how="derived"`：`where` 必须**不是**契约点（例如
      `"keys(get_extra_fields)-default_fields"`），因为反推不发生在契约点上。

    三种通道不许互相顶替：把调用实参伪装成属性、把反推伪装成观测，都会让
    记录看起来完整而实际没有证据。`how="derived"` 还受一条额外约束：
    `evaluate()` 里它不能被标成 `captured`（见该函数的 raise）。
    """
    if name not in FIELDS:
        raise ValueError(f"observe: 未知字段 {name!r}，可选 {FIELDS}")
    if how not in HOW_BY_FIELD[name]:
        raise ValueError(
            f"observe: {name} 的观测通道只能是 {HOW_BY_FIELD[name]}，收到 {how!r}"
            + ("（它没有属性可读）" if name == "extra_fields" and how == "attr" else ""))
    if where is None:
        where = OBSERVE_AT[name]
    if not isinstance(where, str) or not where.strip():
        raise ValueError("observe: where 必须是非空字符串（证据要能指到位置）")
    if how == "derived":
        if where == OBSERVE_AT[name]:
            raise ValueError(
                f"observe: derived 的观测点不能写成契约点 {where!r}——"
                "反推不是在那一点上发生的")
    elif where != OBSERVE_AT[name]:
        raise ValueError(
            f"observe: {name} 的契约观测点是 {OBSERVE_AT[name]!r}，收到 {where!r}；"
            "换了位置就是换了语义（例如从 kw 读冒充从 mc 读）")
    return {"value": value, "how": how, "where": where}


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------

def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _same_value(a: Any, b: Any) -> bool:
    """值比较。元组/列表按**无序**比：`collect_fields` 是 dict 建的，
    `("a","b")` 与 `("b","a")` 采到的是同一组字段。"""
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return sorted(map(str, a)) == sorted(map(str, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        if _is_nan(a) or _is_nan(b):
            return False
        return float(a) == float(b)
    return a == b


def _norm_slot(slot: Any, *, name: str, what: str) -> dict | None:
    """`requested` / `resolved` 的规范化与校验。"""
    if slot is None:
        return None
    if not isinstance(slot, Mapping):
        raise ValueError(f"{name}.{what}: 必须是映射或 None，收到 {type(slot).__name__}")
    if "value" not in slot:
        raise ValueError(f"{name}.{what}: 缺 value")
    val = slot["value"]
    if val is None:
        raise ValueError(
            f"{name}.{what}: value 不能是 None——'不知道'要写 None 整个槽位，"
            "不能用哨兵值冒充观测值（0.2.0 的 hint or 'uniform' 就是这么错的）")
    if _is_nan(val):
        raise ValueError(f"{name}.{what}: value 不能是 NaN——哨兵值不是值")
    origin = slot.get("origin")
    if origin not in ORIGINS:
        raise ValueError(f"{name}.{what}: origin 必须是 {ORIGINS}，收到 {origin!r}")
    where = slot.get("where")
    if where is not None and (not isinstance(where, str) or not where.strip()):
        raise ValueError(f"{name}.{what}: where 必须是非空字符串或 None")
    return {"value": val, "origin": origin, "where": where}


def evaluate(name: str, *, resolved: Mapping | None = None,
             requested: Mapping | None = None,
             observation: Mapping | None = None,
             observed_status: str = "captured",
             kernel: str | None = None, num_chains: int | None = None,
             adapt_step_size: bool | None = True,
             reason_code: str | None = None,
             why: str | None = None) -> dict:
    """判定单项，返回一条记录。**顺序固定，不许调换**。

        ① 适用性（不适用 ⇒ `not_applicable` / `derived`，**跳过值比较**）
        ② 观测通道（失败 ⇒ `incomplete`，**不许用 `resolved` 顶替**）
        ③ `resolved` 缺失（⇒ `derived`）
        ④ 值不等（⇒ `mismatch`）
        ⑤ `origin="default"`（⇒ `defaulted`，**不是** `derived`）
        ⑥ 其余（⇒ `complete`）

    为什么 ① 在 ④ 前面：HMC 的 `_max_tree_depth` 恒为 10，先比数值就会把
    "换了内核"报成"传递链被篡改"。为什么 ② 在 ③④ 前面：采集中断时手上有
    `resolved`，拿它顶替会让"没采到"变成"采到了且一致"。
    """
    if name not in FIELDS:
        raise ValueError(f"evaluate: 未知字段 {name!r}，可选 {FIELDS}")
    if observed_status not in OBSERVED_STATUSES:
        raise ValueError(
            f"evaluate: observed_status 必须是 {OBSERVED_STATUSES}，"
            f"收到 {observed_status!r}")
    if reason_code is not None and reason_code not in REASON_CODES:
        raise ValueError(f"evaluate: 未知原因码 {reason_code!r}，闭集见 REASON_CODES")
    if why is not None and (not isinstance(why, str) or not why.strip()):
        raise ValueError("evaluate: why 必须是非空字符串或 None")

    resolved = _norm_slot(resolved, name=name, what="resolved")
    requested = _norm_slot(requested, name=name, what="requested")

    if observation is not None:
        if not isinstance(observation, Mapping):
            raise ValueError(f"evaluate: {name}.observed 必须是映射或 None")
        for k in ("value", "how", "where"):
            if k not in observation:
                raise ValueError(f"evaluate: {name}.observed 缺 {k}")
        if observation["how"] not in HOW_BY_FIELD[name]:
            raise ValueError(
                f"evaluate: {name}.observed.how 只能是 {HOW_BY_FIELD[name]}，"
                f"收到 {observation['how']!r}")
        if observation["how"] == "derived" and observed_status == "captured":
            raise ValueError(
                f"evaluate: {name} 是反推（how=derived），不能标 captured——"
                '反推要显式降级成 observed_status="hook_not_reached" + '
                'why="inferred_only"，让它落 incomplete，而不是冒充观测')

    supplied = observed_status
    rec = {
        "name": name,
        "requested": requested,
        "resolved": resolved,
        "observed": dict(observation) if observation is not None else None,
        "observed_status": supplied,
        "observed_status_supplied": supplied,
        "applicable": True,
        "reason_code": reason_code,
        "status": None,
        "why": why,
    }

    # ① 适用性：先于一切比较
    app = applicability(name, kernel=kernel, num_chains=num_chains,
                        adapt_step_size=adapt_step_size)
    if not app["applicable"]:
        rec["applicable"] = False
        rec["observed_status"] = "not_applicable"
        rec["status"] = "derived"
        rec["reason_code"] = app["reason"]
        note = app["reason"]
        if supplied != "not_applicable":
            note = f"{note};supplied={supplied}"
        rec["why"] = rec["why"] or note
        return rec
    if supplied == "not_applicable":
        raise ValueError(
            f"evaluate: {name} 在当前配置下**是适用的**，不能标 not_applicable——"
            "不适用必须由 applicability() 判出来，不能由调用方自报")

    # ② 观测通道
    if supplied in _FAILED_OBS:
        rec["status"] = "incomplete"
        rec["why"] = rec["why"] or supplied
        return rec
    if observation is None:
        rec["status"] = "incomplete"
        rec["reason_code"] = rec["reason_code"] or "observation_missing"
        rec["why"] = rec["why"] or "observation_missing"
        return rec
    if _is_nan(observation["value"]):
        rec["status"] = "incomplete"
        rec["reason_code"] = rec["reason_code"] or "observed_nan"
        rec["why"] = rec["why"] or "observed_nan"
        return rec

    # ③ 解析值缺失：只知道档名
    if resolved is None:
        rec["status"] = "derived"
        rec["reason_code"] = rec["reason_code"] or "resolved_missing"
        rec["why"] = rec["why"] or "resolved_missing"
        return rec

    # ④ 值比较
    if not _same_value(resolved["value"], observation["value"]):
        rec["status"] = "mismatch"
        rec["reason_code"] = rec["reason_code"] or "value_differs"
        rec["why"] = rec["why"] or (
            f"resolved={resolved['value']!r} != observed={observation['value']!r}")
        return rec

    # ⑤ 默认值：是生效值，但与"用户指定"不同档
    if resolved["origin"] == "default":
        rec["status"] = "defaulted"
        if resolved["where"] is None:
            rec["reason_code"] = rec["reason_code"] or "origin_default_unsourced"
            rec["why"] = rec["why"] or "origin_default_unsourced"
        rec["why"] = rec["why"] or "origin_default"
        return rec

    # ⑥
    rec["status"] = "complete"
    return rec


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------

def aggregate(records: Iterable[Mapping], *, event_id: str | None = None,
              require_all: bool = True) -> dict:
    """把八项记录合成一条采样事件的结论。

    **取最弱项**（`incomplete > mismatch > derived > defaulted > complete`），
    但八项明细一条不许省——聚合值只用于筛选，不用于下结论。
    `not_applicable` 在单项里已映射成 `derived`，所以它天然把整体封顶在
    `derived`：不适用**不能**让整体看起来更完整。

    `require_all=True` 时缺项/重复直接 raise：少一项就"聚合通过"是这轮要防的
    那种不报错的错。
    """
    recs = [dict(r) for r in records or ()]
    for r in recs:
        if not isinstance(r.get("name"), str) or r["name"] not in FIELDS:
            raise ValueError(f"aggregate: 记录缺合法 name：{r.get('name')!r}")
        if r.get("status") not in STATUSES:
            raise ValueError(
                f"aggregate: {r['name']} 的 status 非法 {r.get('status')!r}，"
                f"只能是 {STATUSES}")
    names = [r["name"] for r in recs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"aggregate: 同一字段出现多次（会掩盖另一条）：{dup}")
    if require_all:
        missing = [f for f in FIELDS if f not in names]
        if missing:
            raise ValueError(f"aggregate: 缺项 {missing}——八项齐全才算一条完整记录")

    by_status = {s: 0 for s in STATUSES}
    for r in recs:
        by_status[r["status"]] += 1
    weakest = max(recs, key=lambda r: STATUS_RANK[r["status"]])["status"]
    overrides = [r["name"] for r in recs
                 if r["status"] == "mismatch" and r.get("reason_code") in OVERRIDE_CODES]
    return {
        "event_id": event_id,
        "n": len(recs),
        "weakest": weakest,
        "n_by_status": by_status,
        "n_not_applicable": sum(1 for r in recs
                                if r["observed_status"] == "not_applicable"),
        "n_override": len(overrides),
        "override_fields": sorted(overrides),
        "fields": {r["name"]: r["status"] for r in recs},
        "observed_status": {r["name"]: r["observed_status"] for r in recs},
        "reason_codes": {r["name"]: r["reason_code"] for r in recs
                         if r.get("reason_code")},
        "why": {r["name"]: r["why"] for r in recs if r.get("why")},
        "complete_evidence": weakest == "complete",
    }


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def _slot(value, origin="cli", where=None):
    return {"value": value, "origin": origin, "where": where}


def _selftest() -> bool:
    ok = True

    def chk(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {label}{('：' + str(detail)) if detail else ''}")

    # 适用性
    chk("HMC 上 max_tree_depth 不适用",
        applicability("max_tree_depth", kernel="HMC") ==
        {"applicable": False, "reason": "kernel_inapplicable"})
    chk("NUTS 上 max_tree_depth 适用",
        applicability("max_tree_depth", kernel="NUTS")["applicable"])
    chk("单链 chain_method 不适用",
        applicability("chain_method", num_chains=1)["reason"] == "single_chain")
    chk("depth 13 vs 10 的预算", tree_budget(13) == 8191 and tree_budget(10) == 1023)

    # 假 mismatch 必须被拦住：HMC 的 10 不参与比较
    r = evaluate("max_tree_depth", resolved=_slot(7),
                 observation=observe("max_tree_depth", 10), kernel="HMC")
    chk("HMC 不产生假 mismatch",
        r["status"] == "derived" and r["observed_status"] == "not_applicable"
        and r["reason_code"] == "kernel_inapplicable", r["status"])

    # 反推不能冒充观测
    try:
        evaluate("extra_fields", resolved=_slot(("num_steps",)),
                 observation=observe("extra_fields", ("num_steps",),
                                     how="derived",
                                     where="keys(get_extra_fields)-default_fields"))
        chk("derived 标 captured 必须 raise", False)
    except ValueError as exc:
        chk("derived 标 captured 必须 raise", "captured" in str(exc))

    r = evaluate("extra_fields", resolved=_slot(("num_steps",)),
                 observation=observe("extra_fields", ("num_steps",), how="derived",
                                     where="keys(get_extra_fields)-default_fields"),
                 observed_status="hook_not_reached", why="inferred_only")
    chk("derived 降级后落 incomplete", r["status"] == "incomplete", r["status"])

    # 八项齐全的一条（全显式、全一致）
    recs = [
        evaluate("n_warmup", resolved=_slot(2000), observation=observe("n_warmup", 2000)),
        evaluate("n_samples", resolved=_slot(1000),
                 observation=observe("n_samples", 1000)),
        evaluate("num_chains", resolved=_slot(2),
                 observation=observe("num_chains", 2), num_chains=2),
        evaluate("target_accept", resolved=_slot(0.98),
                 observation=observe("target_accept", 0.98)),
        evaluate("max_tree_depth", resolved=_slot(10),
                 observation=observe("max_tree_depth", 10), kernel="NUTS"),
        evaluate("chain_method", resolved=_slot("parallel"),
                 observation=observe("chain_method", "parallel"), num_chains=2),
        evaluate("progress_bar", resolved=_slot(False),
                 observation=observe("progress_bar", False)),
        evaluate("extra_fields", resolved=_slot(("num_steps",)),
                 observation=observe("extra_fields", ("num_steps",), how="call_arg")),
    ]
    agg = aggregate(recs)
    chk("八项全显式且一致 ⇒ complete", agg["weakest"] == "complete",
        agg["n_by_status"])

    # 项目从不传 chain_method ⇒ 它天然是 defaulted，整体封顶在 defaulted。
    # 这不是缺陷，是"链方式由库默认决定"这句事实的机器可读形式。
    lib = list(recs)
    lib[5] = evaluate("chain_method",
                      resolved=_slot("parallel", origin="default",
                                     where="numpyro.MCMC.__init__:chain_method"),
                      observation=observe("chain_method", "parallel"), num_chains=2)
    aggl = aggregate(lib)
    chk("chain_method 走库默认 ⇒ 整体 defaulted（不是 complete）",
        aggl["weakest"] == "defaulted"
        and aggl["n_by_status"]["defaulted"] == 1, aggl["weakest"])

    # 一项被改写 ⇒ 只有它是 mismatch
    bad = list(recs)
    bad[0] = evaluate("n_warmup", resolved=_slot(2000),
                      observation=observe("n_warmup", 800))
    agg2 = aggregate(bad)
    chk("单项被改 ⇒ mismatch 只此一项",
        agg2["weakest"] == "mismatch"
        and agg2["n_by_status"]["mismatch"] == 1, agg2["fields"])

    # 缺项必须 raise
    try:
        aggregate(recs[:7])
        chk("缺一项必须 raise", False)
    except ValueError as exc:
        chk("缺一项必须 raise", "缺项" in str(exc))

    print(f"\n自检{'通过' if ok else '失败'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
