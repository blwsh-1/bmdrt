# -*- coding: utf-8 -*-
"""采样参数捕获层（阶段 C 第一轮）：在两处生产构造点读八项配置。

与纯函数层的分工
----------------
`sampler_provenance.py`：**只判定**，不 import numpyro / os，不看环境。
`sampler_capture.py`：**只采集**，一个状态都不自己定。所有 `status` /
`observed_status` / `reason_code` 都由 `evaluate()` 给；这里只负责三件事——

1. 在**构造边界**拿到请求值（`resolved`）及其出处；
2. 在**对象上**读生效值（`observed{value,how,where}`）；
3. 在 **`MCMC.run` 调用边界**拿 `extra_fields`。

它不碰 `run_id`、不碰 `exec_identity.py`、不写盘、不改变拟合行为。

第一轮的目标只有一句话
----------------------
**捕获到的值，是否等于真实构造实参与真实对象属性。**
所以本轮 `origin` 一律是 `derived`：值确定、上游出处未闭合（`cli`/`tier`/签名默认
要等第二轮接 `BUDGETS`/`KW` 才写得出来）。事件里 `upstream_origin_captured=False`
就是这个意思——**`status=complete` 读作"构造实参 → 对象属性"这一段闭合，
不是"整条来源链闭合"**。

两个实测陷阱（写进代码，别再踩）
--------------------------------
· **`adapt_step_size` 的真实属性是 `_adapt_step_size`**，两内核都有；`getattr(sampler,
  "adapt_step_size")` 直接 `AttributeError`。
· **`isinstance(nuts_instance, HMC)` 为 True**（`NUTS` 继承 `HMC`）⇒ 内核判别必须
  **先判 NUTS**，否则每条 NUTS 记录都错标成 HMC，`max_tree_depth` 的适用性跟着错。

`kernel` 与 `adapt_step_size` 是两个**独立**捕获量：前者只用于 R2/R3 的
`max_tree_depth`，后者只用于 R1 的 `target_accept`。合成一个表达式
（`isinstance(sampler, NUTS) and sampler.adapt_step_size`）会在 HMC 上造假
`step_size_not_adapting`——这一条已经写进契约 §5。

    python sampler_capture.py     # 内置自检（构造两个内核，各跑一次捕获）
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Mapping

import budget_capture_adapter as BA
import sampler_provenance as SP
import sampling_hook as HOOK

CAPTURE_VERSION = "1.0"

#: 记录挂在 MCMC 对象上的属性名。**不叫** `sampler_provenance`：生产模块里出现
#: 那个字符串会让 `test_sampler_provenance.X1`（生产零引用）误判。
PROVENANCE_ATTR = "provenance"

#: 构造边界能拿到的七项。`extra_fields` 不在其中——它只活在 `MCMC.run` 调用边界。
CONSTRUCTION_FIELDS = ("n_warmup", "n_samples", "num_chains", "target_accept",
                       "max_tree_depth", "chain_method", "progress_bar")
RUN_FIELDS = ("extra_fields",)

#: 已接入的生产构造点。key 是站点名（**冻结闭集**，见 `exec_identity.BUDGET_CAPTURE_SITES`），
#: value 是可读的锚点描述；行号不再写死，由 `site_anchor()` 用 AST 现场解析。
#: 锚点 = (相对路径, 所属函数名, 该函数内第几个 MCMC 调用)。函数+序号对注释/空行
#: 免疫；新增或删除构造点会让账目对不上，当场 FAIL 而不是静默漏掉。
CALL_SITES = {
    "bm_core.fit": "bm_core.py:fit()#1 MCMC(kern, ...)",
    "bm_top.fit_top": "bm_top.py:fit_top()#1 MCMC(NUTS(model, ...), ...)",
}

#: 站点 → AST 锚点（三者唯一确定一处构造）。与 CALL_SITES 的 key 必须逐项对齐，
#: 由 `test_sampler_capture.G4` 交叉核验。
SITE_ANCHORS = {
    "bm_core.fit": ("bm_core.py", "fit", 1),
    "bm_top.fit_top": ("bm_top.py", "fit_top", 1),
}

#: **明确不接**的构造点。写在这里是为了"不许静默漏掉"：G5 核对生产构造点数
#: == 已接 + 明确不接，数量对不上就报错。
#:
#: 注意两类的区别：
#: · `m35_cond_n2.py` 在仓库研究目录，不在生产模块清单里；
#: · `fit_adr` **在生产模块 bm_top.py 里**，是 ADR 研究拓扑，冻结裁决下不写
#:   provenance、也不许进 CALL_SITES（见 bm_top.py:1221-1223）。它必须同时出现在
#:   "生产构造点账目"里，否则 G5 会把第三处当成漏接而 FAIL。
NON_PRODUCTION_SITES = {
    "evidence/nested_topology/m35_cond_n2.py:116":
        "独立实验路径：自己构造 NUTS+MCMC，绕过 fit/fit_top 与 BUDGETS；"
        "未纳入生产捕获范围",
    "bm_top.py:fit_adr()#1":
        "ADR 研究拓扑（9b8e792 新增）：在生产模块内自带 MCMC 构造，"
        "按冻结裁决不写 provenance、不进 CALL_SITES；此处仅作明账",
}
NON_PRODUCTION_ANCHORS = {
    "evidence/nested_topology/m35_cond_n2.py:116":
        ("evidence/nested_topology/m35_cond_n2.py", "fit", 1),
    "bm_top.py:fit_adr()#1": ("bm_top.py", "fit_adr", 1),
}

#: G5 枚举构造点时扫描的生产模块（与 test_sampler_capture.G5 历史口径一致）。
PRODUCTION_MODULES = ("bm_core.py", "bm_top.py", "bm_real.py", "bm_select.py",
                      "nested_fit.py", "bmdrt.py")

_ROOT = Path(__file__).resolve().parent


def _mcmc_calls(source: str, *, filename: str = "<source>") -> list[dict]:
    """用 AST 找源码里所有 `MCMC(...)` 调用，返回其行号与所属函数名。

    **不用行号做身份**：只返回观测到的位置，供 G4/G5 与 `SITE_ANCHORS` 比对。
    AST 解析忽略注释、空行与字符串里的 `MCMC(` 字样 —— 这正是"不因无关文本误报"
    的关键。语法错误当场抛 `SyntaxError`，不静默返回空表。
    """
    tree = ast.parse(source, filename=filename)
    out = []

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name) \
                    and child.func.id == "MCMC":
                out.append(dict(line=child.lineno, scope=scope))
            else:
                visit(child, scope)

    visit(tree, "<module>")
    return out


def find_mcmc_calls(rel_path: str, *, root=None) -> list[dict]:
    """读一个仓库内相对路径并列出 MCMC 调用；缺文件返回空表。"""
    base = _ROOT if root is None else Path(root)
    path = base / rel_path
    if not path.exists():
        return []
    return _mcmc_calls(path.read_text(encoding="utf-8"), filename=rel_path)


def site_anchor(site: str, *, root=None) -> dict | None:
    """把站点名解析成当前源码里的实际锚点；核对所属函数与函数内序号。

    返回值带 `line`（给人看）与 `scope`/`ordinal`（给账目核对）。任一
    锚点找不到、或被同名函数里新增的 MCMC 挤了序号，返回 None —— 调用方据此
    FAIL，不猜。
    """
    anchor = SITE_ANCHORS.get(site)
    if anchor is None:
        return None
    rel, scope, ordinal = anchor
    calls = [c for c in find_mcmc_calls(rel, root=root) if c["scope"] == scope]
    if len(calls) < ordinal:
        return None
    hit = calls[ordinal - 1]
    return dict(file=rel, scope=scope, ordinal=ordinal, line=hit["line"])

#: 字段 → `MCMC.__init__` 的形参名（用于在**没传**该字段时读库默认值）。
#: 读的是 `inspect.signature`，不是记忆里的字面量——numpyro 改默认值这里会跟着变。
_MCMC_PARAM = {
    "n_warmup": "num_warmup",
    "n_samples": "num_samples",
    "num_chains": "num_chains",
    "chain_method": "chain_method",
    "progress_bar": "progress_bar",
}
#: 这两项在内核构造上，不在 `MCMC.__init__` 上。
_KERNEL_PARAM = {"target_accept": "target_accept_prob",
                 "max_tree_depth": "max_tree_depth"}

_ABSENT = object()


# ---------------------------------------------------------------------------
# 上下文：内核 / 自适应开关 / 设备数 / 环境
# ---------------------------------------------------------------------------

def kernel_of(mcmc) -> str:
    """内核类型。**先判 NUTS**——`NUTS` 是 `HMC` 的子类，反过来判会把 NUTS 全标错。"""
    from numpyro.infer import HMC, NUTS

    s = getattr(mcmc, "sampler", None)
    if isinstance(s, NUTS):
        return "NUTS"
    if isinstance(s, HMC):
        return "HMC"
    raise TypeError(
        f"kernel_of: 认不出内核类型 {type(s).__name__}——只支持 NUTS/HMC。"
        "不许用 hasattr(sampler, '_max_tree_depth') 代替：HMC 也有该属性（值恒 10）")


def adapt_step_size_of(mcmc, *, fallback=None):
    """真实的自适应开关，**与内核类型无关**。返回 `(value, source)`。

    `source`：`"attr"`（从 `sampler._adapt_step_size` 读到的）或
    `"call_arg"`（属性不在、由调用点给的构造实参）。两者都不许默认成 `True`。
    """
    val = getattr(getattr(mcmc, "sampler", None), "_adapt_step_size", _ABSENT)
    if val is _ABSENT:
        if fallback is None:
            raise ValueError(
                "adapt_step_size_of: 内核对象上没有 `_adapt_step_size`，也没有 fallback——"
                "不能默认 True（假适用），也不能默认 False（假不适用）")
        val, source = fallback, "call_arg"
    else:
        source = "attr"
    if not isinstance(val, bool):
        raise ValueError(f"adapt_step_size_of: 必须是 bool，收到 {val!r}")
    return val, source


def progress_bar_forced_by_env() -> bool:
    """numpyro 在 `CI` / `PYTEST_XDIST_WORKER` 存在时**无条件**把 progress_bar 置 False。

    这里只是把那个条件的判读搬出来（numpyro 是静默改的，没有警告可抓）。
    读到了也只是**候选原因**：只有"请求 True、生效 False"真发生了才算 `forced_by_env`。
    """
    return "CI" in os.environ or "PYTEST_XDIST_WORKER" in os.environ


def _context(mcmc) -> dict:
    import jax

    kernel = kernel_of(mcmc)
    adapt, adapt_src = adapt_step_size_of(mcmc)
    num_chains = getattr(mcmc, "num_chains", _ABSENT)
    if isinstance(num_chains, bool) or not isinstance(num_chains, int):
        raise ValueError(f"捕获层: mcmc.num_chains 读不出正整数，收到 {num_chains!r}")
    return {
        "kernel": kernel,
        "adapt_step_size": adapt,
        "adapt_step_size_source": adapt_src,
        "num_chains": num_chains,
        "device_count": int(jax.local_device_count()),
        "progress_bar_env_forced": progress_bar_forced_by_env(),
    }


def _lib_default(field: str, kernel: str) -> tuple[Any, str]:
    """库签名里的默认值 + 出处。读不到就 raise——不猜。"""
    import inspect

    from numpyro.infer import HMC, MCMC, NUTS

    if field in _MCMC_PARAM:
        param = _MCMC_PARAM[field]
        sig = inspect.signature(MCMC.__init__)
        owner = "numpyro.MCMC.__init__"
    else:
        param = _KERNEL_PARAM[field]
        cls = NUTS if kernel == "NUTS" else HMC
        sig = inspect.signature(cls.__init__)
        owner = f"numpyro.{kernel}.__init__"
    p = sig.parameters.get(param, None)
    if p is None:
        raise ValueError(f"_lib_default: {owner} 没有形参 {param!r}，无法给出默认值")
    if p.default is inspect.Parameter.empty:
        raise ValueError(f"_lib_default: {owner}:{param} 没有默认值（必填项），不能当成默认")
    return p.default, f"{owner}:{param}={p.default!r}"


# ---------------------------------------------------------------------------
# 观测
# ---------------------------------------------------------------------------

_GETTERS = {
    "n_warmup": lambda m: m.num_warmup,
    "n_samples": lambda m: m.num_samples,
    "num_chains": lambda m: m.num_chains,
    "target_accept": lambda m: m.sampler._target_accept_prob,
    "max_tree_depth": lambda m: m.sampler._max_tree_depth,
    "chain_method": lambda m: m.chain_method,
    "progress_bar": lambda m: m.progress_bar,
}


def _read_attr(name: str, mcmc):
    """读属性。**读失败不抛**：采不到记 `attribute_missing` / `capture_error`，
    不能让一个属性改名把整次拟合打断。"""
    try:
        return _GETTERS[name](mcmc), "captured", None
    except AttributeError as exc:
        return None, "attribute_missing", f"{type(exc).__name__}: {exc}"
    except Exception as exc:                     # noqa: BLE001 —— 采集失败必须留痕
        return None, "capture_error", f"{type(exc).__name__}: {exc}"


def _slot(value: Any, origin: str, where: str) -> dict:
    return {"value": value, "origin": origin, "where": where}


def _reason_code(name: str, resolved: Mapping | None, observed: Mapping | None,
                 ctx: Mapping) -> str | None:
    """偏移原因只从**两条可核对的事实**里认，不从"值不相等"倒推。

    · `sequentially_resolved`：请求 `parallel`、生效 `sequential`，且
      `local_device_count() < num_chains`（numpyro 改写的唯一条件）；
    · `forced_by_env`：请求 True、生效 False，且 `CI`/`PYTEST_XDIST_WORKER` 在场
      （numpyro 静默改写的唯一条件）。

    两条都不成立就是普通改值，落 `value_differs`——不给证据链断裂发免责声明。
    """
    if name == "chain_method":
        if (resolved and observed and resolved["value"] == "parallel"
                and observed["value"] == "sequential"
                and ctx["num_chains"] > 1
                and ctx["device_count"] < ctx["num_chains"]):
            return "sequentially_resolved"
        return None
    if name == "progress_bar":
        if (resolved and observed and bool(resolved["value"])
                and observed["value"] is False and ctx["progress_bar_env_forced"]):
            return "forced_by_env"
        return None
    return None


# ---------------------------------------------------------------------------
# 单项与事件
# ---------------------------------------------------------------------------

def _failed_record(name: str, why: str) -> dict:
    """采集中断时的占位记录。**没有经过 `evaluate()`**，所以不带任何判定结论：
    `status` 只能是 `incomplete`，且事件里 `judged=False`。"""
    return {"name": name, "requested": None, "resolved": None, "observed": None,
            "observed_status": "capture_error", "observed_status_supplied": "capture_error",
            "applicable": None, "reason_code": None, "status": "incomplete",
            "why": why, "judged": False}


def _one_field(name: str, mcmc, *, site: str, passed: Mapping, run_call: Mapping,
               ctx: Mapping) -> dict:
    """一项：先在构造边界拿 `resolved`，再到契约点拿 `observed`，交给 `evaluate()`。

    `judged=True` 的含义是"这条结论出自 `evaluate()`"，不是"观测成功"——观测失败
    也有结论（`incomplete`），那同样是判定。只有 `_failed_record` 才是没判过的。
    """
    def judged(rec: dict) -> dict:
        rec["judged"] = True
        return rec

    kw = {"kernel": ctx["kernel"], "num_chains": ctx["num_chains"],
          "adapt_step_size": ctx["adapt_step_size"]}

    if name == "extra_fields":
        # 声明值来自调用点（`fit` 的形参），观测值来自 `MCMC.run` 实参——两者不同源，
        # 这一项的 mismatch 才有意义。`fit_top` 没有这个形参 ⇒ 声明值缺失 ⇒ `derived`。
        declared = passed.get(name, _ABSENT)
        resolved = None if declared is _ABSENT else _slot(
            declared, "derived", f"{site}:调度给 MCMC.run 的声明值")
        got = run_call.get(name, _ABSENT)
        if got is _ABSENT or got is None:      # None 不是合法的 extra_fields

            return judged(SP.evaluate(
                name, resolved=resolved, observation=None,
                observed_status="hook_not_reached",
                why="run_call 未提供：调用点没接 MCMC.run 边界", **kw))
        obs = SP.observe(name, got, how="call_arg")
    else:
        if name in passed:
            resolved = _slot(passed[name], "derived", f"{site}:构造实参")
        else:
            default, where = _lib_default(name, ctx["kernel"])
            resolved = _slot(default, "default", where)
        value, observed_status, why = _read_attr(name, mcmc)
        if observed_status != "captured":
            return judged(SP.evaluate(name, resolved=resolved, observation=None,
                                      observed_status=observed_status, why=why, **kw))
        obs = SP.observe(name, value)

    code = _reason_code(name, resolved, obs, ctx)
    return judged(SP.evaluate(name, resolved=resolved, observation=obs,
                              reason_code=code, **kw))


def install_hook() -> Any:
    """把本层显式装进内核钩子，恢复"构造点自动捕获"。返回旧的钩子实现。

    内核默认 no-op（不 import 本层）。需要审计的调用方（复现 harness、身份
    测试、带审计的运行）显式调用本函数，用完用 `sampling_hook.uninstall()`
    或把返回值交回 `sampling_hook.install()` 成对恢复。
    """
    return HOOK.install(capture_at_construction)


def capture_at_construction(mcmc, *, site: str, passed: Mapping | None = None,
                            run_call: Mapping | None = None) -> dict:
    """在构造点采集八项，返回事件并挂到 `mcmc.provenance` 上。

    `passed`：该站点**明确传给构造**的字段值（不在其中 ⇒ 视为走库默认）。
              允许出现 `extra_fields`，含义是"打算传给 `MCMC.run` 的声明值"。
    `run_call`：`MCMC.run` 实参（目前只有 `extra_fields`）。

    除编程错误（未知站点、字段名写错）外**不抛异常**：采不到就记 `capture_error`。
    拟合不能因为采集失败而死，但失败也**不许**被悄悄吞掉——事件里数得出来。
    """
    if site not in CALL_SITES:
        raise ValueError(f"capture_at_construction: 未知站点 {site!r}，已接入的是 "
                         f"{sorted(CALL_SITES)}；新站点要同时登记 CALL_SITES 与构造位置")
    passed = dict(passed or {})
    run_call = dict(run_call or {})
    allowed = set(CONSTRUCTION_FIELDS) | set(RUN_FIELDS)
    bad = sorted(set(passed) - allowed)
    if bad:
        raise ValueError(f"capture_at_construction: passed 里有非构造字段 {bad}，"
                         f"可选 {sorted(allowed)}（run 边界的东西放 run_call）")
    bad_run = sorted(set(run_call) - set(RUN_FIELDS))
    if bad_run:
        raise ValueError(f"capture_at_construction: run_call 里有 {bad_run}，"
                         f"run 边界只接 {list(RUN_FIELDS)}")

    try:
        ctx = _context(mcmc)
    except Exception as exc:                     # noqa: BLE001
        recs = [_failed_record(n, f"上下文不可用: {type(exc).__name__}: {exc}")
                for n in SP.FIELDS]
        return _event(mcmc, site, ctx=None, recs=recs, judged=False,
                      why=f"上下文不可用: {type(exc).__name__}: {exc}")

    recs = []
    for name in SP.FIELDS:
        try:
            recs.append(_one_field(name, mcmc, site=site, passed=passed,
                                   run_call=run_call, ctx=ctx))
        except Exception as exc:                 # noqa: BLE001
            recs.append(_failed_record(
                name, f"采集异常: {type(exc).__name__}: {exc}"))
    judged = all(r.get("judged", False) for r in recs)
    return _event(mcmc, site, ctx=ctx, recs=recs, judged=judged, why=None)


def _event(mcmc, site: str, *, ctx: Mapping | None, recs: list, judged: bool,
           why: str | None) -> dict:
    """事件 = 八项记录 + 聚合 + 上下文。同时挂到对象上，免得调用点忘了带走。"""
    ev = {
        "capture_version": CAPTURE_VERSION,
        "site": site,
        "site_where": CALL_SITES[site],
        "scope": "construction_boundary",
        "upstream_origin_captured": False,
        "judged": judged,
        "kernel": None if ctx is None else ctx["kernel"],
        "adapt_step_size": None if ctx is None else ctx["adapt_step_size"],
        "adapt_step_size_source": None if ctx is None else ctx["adapt_step_size_source"],
        "num_chains": None if ctx is None else ctx["num_chains"],
        "device_count": None if ctx is None else ctx["device_count"],
        "progress_bar_env_forced": None if ctx is None else ctx["progress_bar_env_forced"],
        "n_capture_errors": sum(1 for r in recs if not r.get("judged", False)),
        "why": why,
        "fields": {r["name"]: r for r in recs},
        "aggregate": SP.aggregate(recs),
        "attached_to": PROVENANCE_ATTR,
    }
    # G2：捕获事件自带可落盘的 `budget_params` 六键块与显式证据块。
    # **拒绝式适配**：值已捕获、上游来源链未闭合 ⇒ 五项 source='unavailable'。
    # 绝不把事件的 `upstream_origin_captured=False` 上调；观测值只进
    # `budget_observations` 明账，不进身份哈希。适配失败不吞：单独落错误字段，
    # 让 runner / 准入复查看得见，而不是拿一个假块当身份来源。
    try:
        adapted = BA.adapt_capture_event(ev)
    except (ValueError, TypeError, KeyError) as exc:
        ev["budget_params"] = None
        ev["budget_evidence"] = None
        ev["budget_observations"] = None
        ev["budget_adaptation_error"] = f"{type(exc).__name__}: {exc}"
    else:
        ev["budget_params"] = adapted["budget_params"]
        ev["budget_evidence"] = adapted["budget_evidence"]
        ev["budget_observations"] = adapted["observations"]
        ev["budget_adaptation_error"] = None
    try:
        setattr(mcmc, PROVENANCE_ATTR, ev)
    except Exception:                            # noqa: BLE001 —— 挂不上不影响返回值
        pass
    return ev


# ---------------------------------------------------------------------------
# 自检：不跑采样，只构造对象
# ---------------------------------------------------------------------------

def _mcmc(*, kernel="NUTS", num_chains=1, num_warmup=8, num_samples=8,
          target_accept=0.9, max_tree_depth=10, progress=True,
          adapt_step_size=None):
    """造一个 MCMC 对象（**不 run**）。够读属性就行，采样不在本轮范围内。"""
    import warnings

    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro import handlers
    from numpyro.infer import HMC, MCMC, NUTS

    def model():
        x = numpyro.sample("x", dist.Normal(0.0, 1.0))
        numpyro.sample("o", dist.Normal(x, 1.0), obs=jnp.array(0.0))

    m = handlers.seed(model, 0)
    if kernel == "NUTS":
        kern = NUTS(m, target_accept_prob=target_accept,
                    max_tree_depth=max_tree_depth)
    else:
        kern = HMC(m, target_accept_prob=target_accept)
    if adapt_step_size is not None:
        kern = (NUTS if kernel == "NUTS" else HMC)(
            m, target_accept_prob=target_accept, adapt_step_size=adapt_step_size,
            **({"max_tree_depth": max_tree_depth} if kernel == "NUTS" else {}))
    with warnings.catch_warnings():              # 3 链 1 设备会 warn，正常现象
        warnings.simplefilter("ignore")
        return MCMC(kern, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=num_chains, progress_bar=progress)


def _selftest() -> bool:
    ok = True

    def chk(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {label}"
              f"{('：' + str(detail)) if detail else ''}")

    from numpyro.infer import HMC

    m1 = _mcmc(progress=False)
    chk("NUTS 识别", kernel_of(m1) == "NUTS", kernel_of(m1))
    chk("NUTS 也是 HMC 子类（所以必须先判 NUTS）",
        isinstance(m1.sampler, HMC), type(m1.sampler).__name__)
    mh = _mcmc(kernel="HMC")
    chk("HMC 识别", kernel_of(mh) == "HMC", kernel_of(mh))

    chk("NUTS 的自适应开关", adapt_step_size_of(m1)[0] is True,
        adapt_step_size_of(m1))
    chk("HMC 关掉自适应仍读得到（不被内核类型吞掉）",
        adapt_step_size_of(_mcmc(kernel="HMC", adapt_step_size=False)) ==
        (False, "attr"), adapt_step_size_of(_mcmc(kernel="HMC", adapt_step_size=False)))

    class _NoFlag:
        sampler = type("S", (), {})()

    try:
        adapt_step_size_of(_NoFlag())
        chk("缺 `_adapt_step_size` 必须 raise", False)
    except ValueError as exc:
        chk("缺 `_adapt_step_size` 必须 raise", "adapt_step_size" in str(exc), exc)

    # 单链：chain_method 不适用
    ev = capture_at_construction(
        m1, site="bm_core.fit",
        passed={"n_warmup": 8, "n_samples": 8, "num_chains": 1,
                "target_accept": 0.9, "max_tree_depth": 10, "progress_bar": False,
                "extra_fields": ("num_steps",)},
        run_call={"extra_fields": ("num_steps",)})
    chk("单链 chain_method 落 not_applicable/single_chain",
        ev["fields"]["chain_method"]["observed_status"] == "not_applicable"
        and ev["fields"]["chain_method"]["reason_code"] == "single_chain",
        ev["fields"]["chain_method"]["status"])
    chk("八项都没漏判", ev["judged"] and ev["n_capture_errors"] == 0, ev["n_capture_errors"])
    chk("记录挂到对象上", getattr(m1, PROVENANCE_ATTR, None) is ev)
    chk("单链事件聚合不是 complete", ev["aggregate"]["weakest"] != "complete",
        ev["aggregate"]["weakest"])

    # 三链 1 设备：chain_method 被改写
    m3 = _mcmc(num_chains=3)
    ev3 = capture_at_construction(
        m3, site="bm_core.fit",
        passed={"n_warmup": 8, "n_samples": 8, "num_chains": 3,
                "target_accept": 0.9, "max_tree_depth": 10},
        run_call={"extra_fields": ()})
    chk("三链 1 设备 ⇒ chain_method 被改写成 sequential",
        ev3["fields"]["chain_method"]["status"] == "mismatch"
        and ev3["fields"]["chain_method"]["reason_code"] == "sequentially_resolved",
        ev3["fields"]["chain_method"]["reason_code"])
    chk("偏移单列，不混进普通改值",
        ev3["aggregate"]["override_fields"] == ["chain_method"],
        ev3["aggregate"]["override_fields"])
    chk("聚合被 mismatch 封顶", ev3["aggregate"]["weakest"] == "mismatch",
        ev3["aggregate"]["weakest"])

    # extra_fields 三种情形
    chk("声明与实参一致 ⇒ complete",
        ev["fields"]["extra_fields"]["status"] == "complete",
        ev["fields"]["extra_fields"]["status"])
    evb = capture_at_construction(
        _mcmc(), site="bm_top.fit_top",
        passed={"n_warmup": 8, "n_samples": 8, "num_chains": 1,
                "target_accept": 0.9, "max_tree_depth": 10},
        run_call={"extra_fields": ("diverging",)})
    chk("fit_top 没有 extra_fields 形参 ⇒ derived/resolved_missing",
        evb["fields"]["extra_fields"]["status"] == "derived"
        and evb["fields"]["extra_fields"]["reason_code"] == "resolved_missing",
        evb["fields"]["extra_fields"]["status"])
    evn = capture_at_construction(
        _mcmc(), site="bm_core.fit",
        passed={"n_warmup": 8, "n_samples": 8, "num_chains": 1,
                "target_accept": 0.9, "max_tree_depth": 10},
        run_call={})
    chk("run 边界没接 ⇒ incomplete/hook_not_reached",
        evn["fields"]["extra_fields"]["status"] == "incomplete"
        and evn["fields"]["extra_fields"]["observed_status"] == "hook_not_reached",
        evn["fields"]["extra_fields"]["status"])

    # 上下文缺失：不许打断调用方，也不许伪造成通过
    class _Broken:
        sampler = type("S", (), {})()
        num_warmup = 8
        num_samples = 8
        num_chains = 1

    evx = capture_at_construction(_Broken(), site="bm_core.fit",
                                  passed={"n_warmup": 8}, run_call={})
    chk("上下文不可用 ⇒ 八项全 incomplete，judged=False",
        evx["aggregate"]["weakest"] == "incomplete" and evx["judged"] is False
        and len(evx["fields"]) == 8, evx["aggregate"]["n_by_status"])

    chk("已接入站点 2 个、未接入站点 2 个（m35 + fit_adr 明账）",
        len(CALL_SITES) == 2 and len(NON_PRODUCTION_SITES) == 2
        and len(NON_PRODUCTION_ANCHORS) == len(NON_PRODUCTION_SITES),
        sorted(CALL_SITES))
    chk("站点名与 AST 锚点逐项对齐",
        set(SITE_ANCHORS) == set(CALL_SITES)
        and len(SITE_ANCHORS) == 2, sorted(SITE_ANCHORS))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
