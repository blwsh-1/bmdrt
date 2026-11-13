# -*- coding: utf-8 -*-
"""circuit_registry.py —— 候选电路注册表（阶段 0B 最小原型）

定位
----
**只登记结构，不调度拟合，不产出识别结论。**

本模块只做四件事：
1. 描述每个候选电路的**完整数学结构**（连接骨架 + 元件 + 参数 + 先验 + 初始化档）；
2. 把"候选"定义成可枚举、可校验、可写进 manifest 的对象；
3. 提供 screening / execution / evidence 三个**互不推导**的输出块及其校验；
4. 冻结候选集（拟合前写入，之后不许原地追加）。

**不做**：拟合调度（阶段 1 的 `circuit_runner`）、证据判定（阶段 2/3 的 evaluator）、
自动择优、任何门禁。本模块不 import `numpyro`/`jax`，导入成本接近 0。

三层结构（不要压成一层）
------------------------
- `TopologySpec` 拓扑模板：连接骨架 + 正演函数引用 + 参数空间 + 先验 + 初始化档；
- `ElementSpec` 可嵌入元件：单一阻抗元件 + 自己的参数 + 允许挂载的槽位；
- `Candidate` 候选 = 拓扑 + (槽位→元件) + 配置 + 初始化档 + 预算档 + 似然档。

**为什么不建"平级模型族"**：`bm_diff` 的 Warburg、`bm_ad` 的吸附阻抗是**元件**，
可作为某个拓扑的一个槽位，本身不是完整电路。把它们登记成与 voigt/nested 平级的
顶层条目，每加一个元件就要新增一个顶层族，最后无法公平比较。
（结论来自 2026-11-13 二轮评审 §4。）

三个输出块互不推导
------------------
| 块 | 回答的问题 | 不允许回答的问题 |
|---|---|---|
| `screening` | 候选集里还剩几个 | 谁被识别了 |
| `execution` | 跑没跑成 | 模型对不对 |
| `evidence` | 证据支持什么结论 | 剩一个就等于识别 |

`only_survivor` 是**数量关系**，只能由 `screening_block` 内的 `survivors == 1`
计算，**不接受调用方传入**，也**不是** `evidence.status` 的取值。
AZ31 上 voigt 被结构排除、只剩 nested，只说明候选集剩一个，不说明 nested 被识别。

证据五态与可达条件
------------------
| status | 含义 | 硬条件 |
|---|---|---|
| `identified` | 有有效样本外证据支持区分 | `out_of_sample_valid=True` 且 `n_viable>=2` |
| `weak_preference` | 有倾向但不足 | `n_viable>=2` |
| `non_identifiable` | 至少两个候选无法区分 | `n_viable>=2` |
| `model_inadequate` | 当前候选集合不足 | `truth_known or noise_calibrated` |
| `insufficient_evidence` | 证据不足以支持以上判断 | 无（默认档） |

`identified` 当前口径下不可达：逐点 σ（`sigma_from_data=True`）的 `elpd_loo`
不是 `p(Z_j|Z_{-j})`（`bm_top.fit_top` 已标注
`elpd_valid_for_predictive_comparison=False`），自由 σ 又是绝对加权、高频被系统性
轻视（R16/R17）。要开 `identified` 必须先解决 σ 口径——不是放宽这里。

`model_inadequate` 对**真实谱**默认不开放：真实数据没有真值，acf 高来自测量与
噪声相关结构，不能读作"候选集不足"（§18：合成 12/12 过，真实 0.505–0.896）。
需要独立标定的噪声模型后才允许（本轮不实现，只把条件写成参数）。

预检分级（L0/L1/L2）
--------------------
- L0 只出 `excluded` / `not_excluded` / `unknown`，必须是**确定性判据**
  （如 `bm_select.voigt_can_represent`：只看感抗落在哪个频段/哪个接点，不采样）；
- L1 只出 `promote` / `defer` / `not_evaluated_budget`，**永远不出 `excluded`**。
  单次快速拟合不收敛 ≠ 模型不适用（钝化谱：默认初值塌陷，换数据驱动初值 13.1s
  收敛、R̂ 1.004）；
- L2 出最终证据，但"采样成功"也不等于"模型成立"。

边界
----
- 拓扑只登记 `voigt` / `nested` 两个；Warburg 与吸附登记为**未接线元件**
  （`wired=False`），因为现有拓扑没有对应的槽位。
- 不接 `bmdrt.py`，不改 `bm_core`/`bm_top`/`bm_select`/`bm_identifiability` 任何一行。
- 本模块的校验是**格式与约束层**的强制，不证明语义正确；后者靠
  `test_circuit_registry.py` 的 fixture 契约测试。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

REGISTRY_VERSION = "0.1.0"
PRECHECK_VERSION = "0.1.0"

PRECHECK_STATES = ("excluded", "not_excluded", "unknown")
L1_OUTCOMES = ("promote", "defer", "not_evaluated_budget")
EXEC_STATES = ("completed", "failed", "not_run")
EVIDENCE_STATES = ("identified", "weak_preference", "non_identifiable",
                   "model_inadequate", "insufficient_evidence")
PRECHECK_LEVELS = ("L0", "L1", "L2")
MATURITY = ("production", "exploratory", "frozen")

#: 允许出现在 `forward_ref` 里的模块。防止拼错模块名后到阶段 1 才发现。
ALLOWED_MODULES = ("bm_core", "bm_top", "bm_diff", "bm_ad", "bm_select")

EVIDENCE_SCOPE = "registered_candidate_set"


# --------------------------------------------------------------------------
# 规格：拓扑与元件
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SlotSpec:
    """拓扑上的一个可挂载位置。"""
    id: str
    name: str
    accepts: tuple
    min_count: int = 1
    max_count: int | None = 1
    note: str = ""

    def __post_init__(self):
        if self.min_count < 0:
            raise ValueError(f"槽位 {self.id}：min_count 不能为负")
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError(f"槽位 {self.id}：max_count < min_count")


@dataclass(frozen=True)
class TopologySpec:
    """拓扑模板：连接骨架 + 正演入口 + 参数空间 + 先验 + 初始化档。

    `skeleton` 必须是**可对照的数学结构**，不能只写名字——这是本模块的验收要求。
    """
    id: str
    name: str
    skeleton: str
    forward_ref: str
    param_space: tuple
    derived: tuple
    priors: Mapping[str, str]
    init_profiles: Mapping[str, str]
    slots: tuple
    maturity: str
    min_level: str = "L1"
    precheck_ref: str | None = None
    consistency: Any = None          # (elements_by_slot, config) -> None，违规抛 ValueError
    notes: tuple = ()

    def __post_init__(self):
        if len(self.skeleton.strip()) < 20:
            raise ValueError(f"拓扑 {self.id}：skeleton 太短，必须写清连接结构")
        if not self.param_space:
            raise ValueError(f"拓扑 {self.id}：param_space 不能为空")
        _check_ref(self.forward_ref, f"拓扑 {self.id}.forward_ref")
        if self.precheck_ref is not None:
            _check_ref(self.precheck_ref, f"拓扑 {self.id}.precheck_ref")
        if self.maturity not in MATURITY:
            raise ValueError(f"拓扑 {self.id}：maturity={self.maturity!r} 不在 {MATURITY}")
        if self.min_level not in PRECHECK_LEVELS:
            raise ValueError(f"拓扑 {self.id}：min_level={self.min_level!r} 不在 {PRECHECK_LEVELS}")
        if not self.init_profiles:
            raise ValueError(f"拓扑 {self.id}：至少要登记一个 init_profile")


@dataclass(frozen=True)
class ElementSpec:
    """可嵌入元件。

    `wired=False` 表示**登记在册但当前没有任何拓扑提供槽位**——不是"可用但没测"。
    这种元件不许被任何 Candidate 引用（`validate_candidate` 会拒）。
    """
    id: str
    name: str
    forward_ref: str
    params: tuple
    maturity: str
    wired: bool = True
    blocked_reason: str = ""
    formula: str = ""
    notes: tuple = ()

    def __post_init__(self):
        if not self.params:
            raise ValueError(f"元件 {self.id}：params 不能为空")
        _check_ref(self.forward_ref, f"元件 {self.id}.forward_ref")
        if self.maturity not in MATURITY:
            raise ValueError(f"元件 {self.id}：maturity={self.maturity!r} 不在 {MATURITY}")
        if not self.wired and not self.blocked_reason:
            raise ValueError(f"元件 {self.id}：wired=False 时必须写 blocked_reason")


def _check_ref(ref: str, where: str) -> None:
    if not isinstance(ref, str) or ref.count(":") != 1:
        raise ValueError(f"{where}：必须写成 'module:attr'，收到 {ref!r}")
    mod = ref.split(":")[0]
    if mod not in ALLOWED_MODULES:
        raise ValueError(f"{where}：模块 {mod!r} 不在允许列表 {ALLOWED_MODULES}")


# --------------------------------------------------------------------------
# 候选
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """一个可比较的最小单元：拓扑 + 元件 + 配置 + 三个 profile。

    `candidate_id` 覆盖全部这些字段，所以"同一拓扑换初始化"是**不同候选**，
    不会在 manifest 里被合并掉——钝化谱的默认初值与数据驱动初值，结论完全不同。
    """
    candidate_id: str
    topology_id: str
    elements: tuple              # tuple[(slot_id, element_id)]
    config: tuple                # tuple[(key, value)]，已排序
    init_profile: str
    budget_profile: str
    likelihood_profile: str

    @property
    def config_dict(self) -> dict:
        return dict(self.config)

    def describe(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "topology": self.topology_id,
            "elements": dict(self.elements),
            "config": self.config_dict,
            "init_profile": self.init_profile,
            "budget_profile": self.budget_profile,
            "likelihood_profile": self.likelihood_profile,
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def compute_candidate_id(topology_id: str, elements, config: Mapping,
                         init_profile: str, budget_profile: str,
                         likelihood_profile: str) -> str:
    payload = {
        "topology": topology_id,
        "elements": sorted([list(e) for e in elements]),
        "config": sorted([[str(k), _jsonable(v)] for k, v in config.items()]),
        "init": init_profile,
        "budget": budget_profile,
        "likelihood": likelihood_profile,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return "C-" + hashlib.sha256(blob).hexdigest()[:10]


# --------------------------------------------------------------------------
# 注册表
# --------------------------------------------------------------------------
TOPOLOGIES: dict = {}
ELEMENTS: dict = {}


def register_topology(spec: TopologySpec) -> TopologySpec:
    if spec.id in TOPOLOGIES:
        raise ValueError(f"拓扑 {spec.id!r} 重复登记")
    TOPOLOGIES[spec.id] = spec
    return spec


def register_element(spec: ElementSpec) -> ElementSpec:
    if spec.id in ELEMENTS:
        raise ValueError(f"元件 {spec.id!r} 重复登记")
    ELEMENTS[spec.id] = spec
    return spec


def get_topology(tid: str) -> TopologySpec:
    if tid not in TOPOLOGIES:
        raise KeyError(f"未登记的拓扑 {tid!r}；已有 {sorted(TOPOLOGIES)}")
    return TOPOLOGIES[tid]


def get_element(eid: str) -> ElementSpec:
    if eid not in ELEMENTS:
        raise KeyError(f"未登记的元件 {eid!r}；已有 {sorted(ELEMENTS)}")
    return ELEMENTS[eid]


def list_topologies() -> tuple:
    return tuple(sorted(TOPOLOGIES))


def list_elements(wired_only: bool = False) -> tuple:
    ids = sorted(ELEMENTS)
    if wired_only:
        ids = [i for i in ids if ELEMENTS[i].wired]
    return tuple(ids)


def resolve_forward(ref: str):
    """惰性解析 `module:attr`。本模块自身不 import 这些重模块。"""
    _check_ref(ref, "forward_ref")
    import importlib
    mod, attr = ref.split(":")
    return getattr(importlib.import_module(mod), attr)


# --------------------------------------------------------------------------
# 候选构造与校验
# --------------------------------------------------------------------------
def make_candidate(topology_id: str, elements, config: Mapping | None = None,
                   *, init_profile: str, budget_profile: str = "standard",
                   likelihood_profile: str = "perpoint-2pct") -> Candidate:
    """构造并校验一个候选。

    `elements` 形如 `[("relaxation_unit", "hn"), ("relaxation_unit", "hn")]`
    或 `[("branch", "cpe")]`，同一槽位可以出现多次（voigt 的 K 个单元）。
    """
    topo = get_topology(topology_id)
    el = tuple((str(a), str(b)) for a, b in elements)
    cfg = dict(config or {})

    if init_profile not in topo.init_profiles:
        raise ValueError(
            f"拓扑 {topology_id} 未登记 init_profile {init_profile!r}；"
            f"已有 {sorted(topo.init_profiles)}")

    cand = Candidate(
        candidate_id=compute_candidate_id(topology_id, el, cfg, init_profile,
                                         budget_profile, likelihood_profile),
        topology_id=topology_id,
        elements=el,
        config=tuple(sorted((str(k), _jsonable(v)) for k, v in cfg.items())),
        init_profile=init_profile,
        budget_profile=budget_profile,
        likelihood_profile=likelihood_profile,
    )
    validate_candidate(cand)
    return cand


def validate_candidate(cand: Candidate) -> None:
    topo = get_topology(cand.topology_id)
    by_slot: dict = {}
    for slot_id, elem_id in cand.elements:
        by_slot.setdefault(slot_id, []).append(elem_id)

    slots = {s.id: s for s in topo.slots}
    for slot_id in by_slot:
        if slot_id not in slots:
            raise ValueError(
                f"候选 {cand.candidate_id}：拓扑 {topo.id} 没有槽位 {slot_id!r}；"
                f"可用 {sorted(slots)}")

    for slot_id, elem_ids in by_slot.items():
        slot = slots[slot_id]
        for elem_id in elem_ids:
            spec = get_element(elem_id)
            if not spec.wired:
                raise ValueError(
                    f"候选 {cand.candidate_id}：元件 {elem_id!r} 未接线"
                    f"（{spec.blocked_reason}），不能出现在候选中")
            if elem_id not in slot.accepts:
                raise ValueError(
                    f"候选 {cand.candidate_id}：槽位 {slot_id} 只接受 {slot.accepts}，"
                    f"收到 {elem_id!r}")
        if len(elem_ids) < slot.min_count:
            raise ValueError(
                f"候选 {cand.candidate_id}：槽位 {slot_id} 至少需要 {slot.min_count} 个，"
                f"实得 {len(elem_ids)}")
        if slot.max_count is not None and len(elem_ids) > slot.max_count:
            raise ValueError(
                f"候选 {cand.candidate_id}：槽位 {slot_id} 最多 {slot.max_count} 个，"
                f"实得 {len(elem_ids)}")

    for slot in topo.slots:
        if slot.id not in by_slot and slot.min_count > 0:
            raise ValueError(
                f"候选 {cand.candidate_id}：槽位 {slot.id} 未能省略（min_count={slot.min_count}）")

    if topo.consistency is not None:
        topo.consistency(by_slot, cand.config_dict)


# --------------------------------------------------------------------------
# 三个输出块
# --------------------------------------------------------------------------
def screening_block(*, registered: int, excluded: int, not_excluded: int,
                    unknown: int, survivors: int | None = None,
                    notes: tuple = ()) -> dict:
    """筛选层。只记数量关系，不出任何证据结论。

    `survivors` 是**拟合后仍 viable 的候选数**（阶段 0 不跑拟合，传 None）。
    `only_survivor` 由这里算，不接受调用方传入——这是硬约束。
    """
    for k, v in dict(registered=registered, excluded=excluded,
                     not_excluded=not_excluded, unknown=unknown).items():
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"screening.{k} 必须是非负整数，收到 {v!r}")
    if registered != excluded + not_excluded + unknown:
        raise ValueError(
            f"screening 不闭合：registered={registered} ≠ "
            f"excluded+not_excluded+unknown={excluded + not_excluded + unknown}")
    if survivors is not None:
        if not isinstance(survivors, int) or survivors < 0:
            raise ValueError(f"screening.survivors 必须是非负整数或 None，收到 {survivors!r}")
        if survivors > not_excluded + unknown:
            raise ValueError(
                f"screening.survivors={survivors} 超过进入拟合的候选数 "
                f"{not_excluded + unknown}")

    return {
        "registered": registered,
        "excluded": excluded,
        "not_excluded": not_excluded,
        "unknown": unknown,
        "entered_fit": not_excluded + unknown,
        "survivors": survivors,
        "only_survivor": survivors == 1,
        "notes": list(notes),
    }


def execution_block(*, total: int, completed: int = 0, failed: int = 0,
                    not_run: int = 0, failure_kinds: Mapping | None = None,
                    notes: tuple = ()) -> dict:
    """执行层。只记跑没跑成，不评价模型。

    `failure_kinds` 用来区分 0A 里那三种"没成"：初始化塌陷、采样不收敛、数值报错。
    它们对 schema 的意义不同（前者可能靠换 init 救，后者不行）。
    """
    if total != completed + failed + not_run:
        raise ValueError(
            f"execution 不闭合：total={total} ≠ completed+failed+not_run="
            f"{completed + failed + not_run}")
    kinds = dict(failure_kinds or {})
    if sum(kinds.values()) > failed:
        raise ValueError("failure_kinds 之和不能超过 failed")

    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "not_run": not_run,
        "failure_kinds": kinds,
        "notes": list(notes),
    }


def observation_block(*, candidate_id: str, converged: bool,
                      rms_rel: float | None = None, acf: float | None = None,
                      rhat_max: float | None = None, ess_min: float | None = None,
                      divergences: int | None = None,
                      param_stability: Mapping | None = None,
                      notes: tuple = ()) -> dict:
    """单个候选单次运行的**描述性**记录。不出结论、不排序、不合成总分。

    硬约束（沿用 E3 纪律）：`converged=False` 时**不许**报 `rms_rel` / `acf`。
    没收敛却报"残差 5%"，会被读成"拟合得还行"，而它只是某条链的最后一步。

    `acf` 允许负值（钝化那条 `truth_init` 实测 −0.244），不要假设它在 [0,1]。

    `param_stability` 必须是"参数名 → 描述块"，每块含 `ci90` 或 `flag`，
    **不允许**传单个标量分数：把参数可信度压成一个数，等于又做了一次隐式判决。
    """
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("observation.candidate_id 必须是非空字符串")
    if not converged and (rms_rel is not None or acf is not None):
        raise ValueError(
            f"候选 {candidate_id} 未收敛，不得报 rms_rel/acf"
            "（E3：not_converged 不报数）")

    stab = None
    if param_stability is not None:
        if not isinstance(param_stability, Mapping):
            raise ValueError("param_stability 必须是 参数名→描述块 的映射")
        stab = {}
        for k, v in param_stability.items():
            if not isinstance(v, Mapping):
                raise ValueError(
                    f"param_stability[{k!r}] 必须是描述块（含 ci90/flag），不能是标量")
            if "ci90" not in v and "flag" not in v:
                raise ValueError(f"param_stability[{k!r}] 至少要给 ci90 或 flag")
            stab[str(k)] = {str(a): _jsonable(b) for a, b in v.items()}

    return {
        "candidate_id": candidate_id,
        "converged": bool(converged),
        "rms_rel": None if rms_rel is None else float(rms_rel),
        "acf": None if acf is None else float(acf),
        "rhat_max": None if rhat_max is None else float(rhat_max),
        "ess_min": None if ess_min is None else float(ess_min),
        "divergences": None if divergences is None else int(divergences),
        "param_stability": stab,
        "notes": list(notes),
    }


def evidence_block(*, status: str, n_viable: int,
                   truth_known: bool = False, noise_calibrated: bool = False,
                   out_of_sample_valid: bool = False,
                   note: str = "") -> dict:
    """证据层。五态，带硬条件校验。

    这些校验不是文档，是**可执行的护栏**：写错状态会当场报错，而不是默默产出一个
    看起来合理的结论。
    """
    if status not in EVIDENCE_STATES:
        raise ValueError(f"evidence.status={status!r} 不在 {EVIDENCE_STATES}")
    if not isinstance(n_viable, int) or n_viable < 0:
        raise ValueError(f"evidence.n_viable 必须是非负整数，收到 {n_viable!r}")

    if status == "identified":
        if not out_of_sample_valid:
            raise ValueError(
                "identified 需要有效的样本外证据（out_of_sample_valid=True）。"
                "逐点 σ 下 elpd_loo 没有留一预测解释（bm_top.fit_top 已标注）；"
                "要开这一态先修 σ 口径，不是改这里。")
        if n_viable < 2:
            raise ValueError("identified 需要至少两个 viable 候选才能谈'区分'")

    if status in ("weak_preference", "non_identifiable") and n_viable < 2:
        raise ValueError(f"{status} 至少需要两个 viable 候选，实得 {n_viable}")

    if status == "model_inadequate" and not (truth_known or noise_calibrated):
        raise ValueError(
            "model_inadequate 需要 truth_known 或 noise_calibrated。"
            "真实谱的 acf 高来自测量与噪声相关结构，不能读作'候选集不足'"
            "（§18：合成 12/12 过，真实 0.505–0.896）。真实谱默认 insufficient_evidence。")

    return {
        "status": status,
        "scope": EVIDENCE_SCOPE,
        "n_viable": n_viable,
        "out_of_sample_valid": bool(out_of_sample_valid),
        "truth_known": bool(truth_known),
        "noise_calibrated": bool(noise_calibrated),
        "note": note,
    }


# --------------------------------------------------------------------------
# 预检分级
# --------------------------------------------------------------------------
def l1_outcome(*, converged: bool, init_explored: bool,
               budget_sufficient: bool = True) -> str:
    """L1 快速拟合的三种结局。

    **永远不返回 `excluded`。** 单次快速拟合失败可能是初始化问题（钝化谱：
    默认初值塌陷、数据驱动初值 13.1s 收敛），把它当结构排除会把整库的判定带偏。
    只有 L0 的确定性判据能出 `excluded`。
    """
    if not budget_sufficient:
        return "not_evaluated_budget"
    if not init_explored:
        return "defer"
    return "promote" if converged else "defer"


def l0_precheck(topology_id: str, f, Z, **kw) -> tuple:
    """调用拓扑登记的确定性预检，返回 `(state, reason)`。

    `state` 严格落在 `PRECHECK_STATES`。未登记预检的拓扑一律返回 `unknown`——
    "没有判据"不等于"通过"。
    """
    topo = get_topology(topology_id)
    if topo.precheck_ref is None:
        return "unknown", "该拓扑未登记确定性预检"
    fn = resolve_forward(topo.precheck_ref)
    ok, reason = fn(f, Z, **kw)
    if ok is True:
        return "not_excluded", ""
    if ok is False:
        return "excluded", str(reason)
    raise ValueError(f"预检 {topo.precheck_ref} 返回值必须是 bool，收到 {ok!r}")


# --------------------------------------------------------------------------
# 候选集冻结（manifest 契约）
# --------------------------------------------------------------------------
def candidate_set_manifest(*, candidate_set_id: str, candidates,
                           likelihood_profile: str, budget_profile: str,
                           seed_policy: str, created_before_fit: bool,
                           extra: Mapping | None = None) -> dict:
    """拟合前写入的候选集清单。之后不可原地追加。

    `created_before_fit` 必须是 True。发现新电路时要建**新的**
    `candidate_set_id`，而不是往旧集合里加——否则多重比较没有记录。
    """
    if created_before_fit is not True:
        raise ValueError(
            "候选集必须在首次拟合前写入（created_before_fit=True）。"
            "事后追加会让候选集变化与多重比较失去记录。")
    cands = list(candidates)
    if not cands:
        raise ValueError("候选集不能为空")
    ids = [c.candidate_id for c in cands]
    if len(set(ids)) != len(ids):
        dup = [i for i in set(ids) if ids.count(i) > 1]
        raise ValueError(f"候选集里有重复 candidate_id：{dup}")

    return {
        "candidate_set_id": candidate_set_id,
        "candidate_ids": ids,
        "candidate_definitions": [c.describe() for c in cands],
        "registry_version": REGISTRY_VERSION,
        "precheck_version": PRECHECK_VERSION,
        "likelihood_profile": likelihood_profile,
        "budget_profile": budget_profile,
        "seed_policy": seed_policy,
        "created_before_fit": True,
        "registry_digest": registry_digest(),
        "immutable": True,
        "extra": dict(extra or {}),
    }


def assert_no_append(manifest: Mapping, candidates) -> None:
    """拒绝往已冻结的候选集里追加新候选。"""
    frozen = set(manifest.get("candidate_ids", ()))
    incoming = {c.candidate_id for c in candidates}
    extra = incoming - frozen
    if extra:
        raise ValueError(
            f"候选集 {manifest.get('candidate_set_id')} 已冻结；新增 {sorted(extra)} "
            "需要建立新的 candidate_set_id（这是一次新分析，不是原集合的延伸）")


def registry_digest() -> str:
    """注册表内容指纹。manifest 记下它，注册表被改动后可被发现。"""
    payload = {
        "topologies": {
            k: {
                "skeleton": v.skeleton,
                "forward_ref": v.forward_ref,
                "param_space": list(v.param_space),
                "derived": list(v.derived),
                "priors": dict(v.priors),
                "init_profiles": dict(v.init_profiles),
                "slots": [[s.id, list(s.accepts), s.min_count, s.max_count]
                          for s in v.slots],
                "maturity": v.maturity,
                "min_level": v.min_level,
                "precheck_ref": v.precheck_ref,
            }
            for k, v in TOPOLOGIES.items()
        },
        "elements": {
            k: {
                "forward_ref": v.forward_ref,
                "params": list(v.params),
                "maturity": v.maturity,
                "wired": v.wired,
                "blocked_reason": v.blocked_reason,
                "formula": v.formula,
            }
            for k, v in ELEMENTS.items()
        },
        "registry_version": REGISTRY_VERSION,
        "precheck_version": PRECHECK_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------
# 已登记内容
# --------------------------------------------------------------------------
def _nested_consistency(by_slot: Mapping, cfg: Mapping) -> None:
    """nested 的两条结构约束（写在这里而不是注释里，错误当场抛）。"""
    layers = int(cfg.get("layers", 2))
    branch = str(cfg.get("branch", "parallel"))
    expect = {"parallel": "rl_parallel", "series": "rl_series", "capacitive": "cpe"}
    if branch not in expect:
        raise ValueError(f"nested.config.branch={branch!r} 只能是 {sorted(expect)}")
    got = by_slot.get("branch", [])
    if got != [expect[branch]]:
        raise ValueError(
            f"nested：branch={branch!r} 要求槽位 branch 挂 {expect[branch]!r}，实得 {got}")
    inner = by_slot.get("inner_cpe", [])
    if layers == 2 and len(inner) != 1:
        raise ValueError(f"nested：layers=2 需要恰好一个 inner_cpe，实得 {inner}")
    if layers == 1 and inner:
        raise ValueError(f"nested：layers=1 不含第二并联节点，inner_cpe 必须为空，实得 {inner}")


register_topology(TopologySpec(
    id="voigt",
    name="串联 HN 弛豫叠加",
    skeleton="Z(f) = R_inf + Σ_{k=1..K} Z_HN(τ_k, W_k, φ_k, ζ_k)  [+ jωL，仅 include_L=True]",
    forward_ref="bm_core:model_z",
    param_space=("R_inf", "W[k]", "pi[k]", "ln_tau[k]", "zeta[k]", "L"),
    derived=("K",),
    priors={
        "R_inf": "LogNormal 相对 zmax 的固定档（见 bm_core.make_model）",
        "W[k]": "LogNormal，K 个分量共享尺度",
        "pi[k]": "φ ∈ (0,1)，Uniform/截断（bm_select.VOIGT_CAPS['phi']）",
        "zeta[k]": "ζ ∈ (0.3,2.0)，zeta_fixed=1.0 时退化为 Cole-Cole",
        "L": "仅在 include_L=True 时采样",
    },
    init_profiles={
        "uniform": "init_to_uniform（生产默认）",
        "value": "init_to_value，需要调用方给约束空间初值",
    },
    slots=(
        SlotSpec("relaxation_unit", "HN 弛豫单元", ("hn",), min_count=1, max_count=None,
                 note="K 由 config['K'] 给出，每个单元一组 (W,τ,φ,ζ)"),
        SlotSpec("series_inductor", "顶层串联感抗", ("rl_series",), min_count=0, max_count=1,
                 note="仅 include_L=True；默认关闭"),
    ),
    maturity="production",
    min_level="L1",
    precheck_ref="bm_select:voigt_can_represent",
    notes=(
        "预检只判感抗接点，不判容性最内层，因此对涂层/钝化返回 not_excluded 而非 excluded",
        "§13 留出 0/3、§14 全量 4/14；默认族的准入结论被这两项卡住",
    ),
))

register_topology(TopologySpec(
    id="nested",
    name="层状嵌套并联",
    skeleton=(
        "layers=2: Z = Rs + Q1 ∥ (R1 + [Q2 ∥ (R2 + Z_b)])\n"
        "layers=1: Z = Rs + Q1 ∥ (R1 + Z_b)\n"
        "其中 Q 为窗口幅值参数化的 CPE：Z_Q = Q·(ω_ref/ω)^n·exp(-j nπ/2)"
    ),
    forward_ref="bm_top:z_nested",
    param_space=("Rs", "R1", "Q1", "n1", "R2[l2]", "Q2[l2]", "n2[l2]",
                 "R_ad|R_L|Q_b", "tau_ad|tau_L|n_b"),
    derived=("L_ad",),
    priors={
        "Rs": "LogNormal(log(0.2·zmax), 1.0)",
        "R1": "LogNormal(log(0.1·zmax), 2.0)",
        "Q1": "LogNormal(log(zmax), 2.0)，Q=|Z(ω_ref)|",
        "n1": "Uniform(0,1) 截断",
        "R2/Q2/n2": "仅 layers=2 采样，同上层尺度",
        "R_ad/tau_ad": "branch=parallel，R∥L",
        "R_L/tau_L": "branch=series，R+L；L_ad=R_L·tau_L 为派生",
        "Q_b/n_b": "branch=capacitive，纯 CPE",
    },
    init_profiles={
        "uniform": "init_to_uniform（生产默认）",
        "data_driven": "R_inf=min|Z|、W=(zmax-zmin)/2、τ 在窗内均铺、φ=0.8",
        "value": "init_to_value，需要调用方给物理值",
    },
    slots=(
        SlotSpec("outer_cpe", "外层 CPE", ("cpe",), min_count=1, max_count=1),
        SlotSpec("inner_cpe", "内层 CPE（仅 layers=2）", ("cpe",), min_count=0, max_count=1),
        SlotSpec("branch", "最内层支路", ("rl_parallel", "rl_series", "cpe"),
                 min_count=1, max_count=1),
    ),
    maturity="exploratory",
    min_level="L1",
    precheck_ref=None,
    consistency=_nested_consistency,
    notes=(
        "§13 留出 0/3、§14 全量 4/14，卡 M-b acf≤0.50；不接 bmdrt.py 主流程",
        "骨架与 测试数据/涂层、钝化 对到 1e-15；这两族只是同构，不构成通用证据",
        "未登记确定性预检：unknown 不等于 not_excluded",
    ),
))

register_element(ElementSpec(
    id="hn",
    name="Havriliak-Negami 弛豫单元",
    forward_ref="bm_core:_hn_z",
    params=("W", "tau", "phi", "zeta"),
    maturity="production",
    formula="Z_HN(ω) = W / [1 + (jωτ)^φ]^ζ",
    notes=("φ=1,ζ=1 退化为 Debye；ζ=1 退化为 Cole-Cole",),
))

register_element(ElementSpec(
    id="cpe",
    name="常相位元件（窗口幅值参数化）",
    forward_ref="bm_top:z_cpe",
    params=("Q", "n"),
    maturity="exploratory",
    formula="Z = Q·(ω_ref/ω)^n·exp(-j nπ/2)，Q=|Z(ω_ref)|，ω_ref=2π√(f_min·f_max)",
    notes=("Q 与数据侧 Q_data 的换算：Q_ours = Q_data / ω_ref^n",),
))

register_element(ElementSpec(
    id="rl_parallel",
    name="感抗支路 R∥L",
    forward_ref="bm_top:z_rl_parallel",
    params=("R_ad", "tau_ad"),
    maturity="exploratory",
    formula="Z = R·jωτ/(1+jωτ)；ω→0 时 →0，ω→∞ 时 →R",
    notes=("低频感抗峰在 1/(2πτ)；这是 nested 的默认支路，也是 AZ31 需要的那个接点",),
))

register_element(ElementSpec(
    id="rl_series",
    name="感抗支路 R+L",
    forward_ref="bm_top:z_rl_series",
    params=("R_L", "tau_L"),
    maturity="exploratory",
    formula="Z = R_L + jωL，L = R_L·τ_L",
    notes=("与 rl_parallel 不等价：只产生高频感抗",),
))

register_element(ElementSpec(
    id="w_inf",
    name="半无限 Warburg",
    forward_ref="bm_diff:z_winf",
    params=("A",),
    maturity="frozen",
    wired=False,
    blocked_reason="现有两个拓扑都没有扩散槽位；接入需要先定义槽位与接点",
    formula="Z = A/√(jω)（渐近形式，见 bm_diff.z_winf）",
))

register_element(ElementSpec(
    id="w_open",
    name="有限长度 Warburg（开路）",
    forward_ref="bm_diff:z_wopen",
    params=("Rw", "tauD"),
    maturity="frozen",
    wired=False,
    blocked_reason="同上；且 §11 未把扩散接入列为当前范围",
    formula="Z = Rw·coth(√(jωτD))/√(jωτD)",
))

register_element(ElementSpec(
    id="w_short",
    name="有限长度 Warburg（短路）",
    forward_ref="bm_diff:z_wshort",
    params=("Rw", "tauD"),
    maturity="frozen",
    wired=False,
    blocked_reason="同上；另有实现未接 CLI 的记录",
    formula="Z = Rw·tanh(√(jωτD))/√(jωτD)",
))

register_element(ElementSpec(
    id="adsorption",
    name="吸附阻抗",
    forward_ref="bm_ad:z_ad",
    params=("Rad", "tau_ad"),
    maturity="exploratory",
    wired=False,
    blocked_reason="只允许挂在第二并联节点内；现有拓扑的 branch 槽位语义不同，尚未定义接点",
    formula="见 bm_ad.z_ad；低频极限为 R∥L 型回路",
    notes=("§9.4.12 实测：把 R∥L 挂到 voigt 最外层串联，noise=0 残差只从 44.0% 降到 40.7%",),
))


__all__ = [
    "REGISTRY_VERSION", "PRECHECK_VERSION",
    "PRECHECK_STATES", "L1_OUTCOMES", "EXEC_STATES", "EVIDENCE_STATES",
    "PRECHECK_LEVELS", "MATURITY", "EVIDENCE_SCOPE",
    "SlotSpec", "TopologySpec", "ElementSpec", "Candidate",
    "TOPOLOGIES", "ELEMENTS",
    "register_topology", "register_element", "get_topology", "get_element",
    "list_topologies", "list_elements", "resolve_forward",
    "make_candidate", "validate_candidate", "compute_candidate_id",
    "screening_block", "execution_block", "observation_block", "evidence_block",
    "l0_precheck", "l1_outcome",
    "candidate_set_manifest", "assert_no_append", "registry_digest",
]
