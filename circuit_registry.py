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
  单次快速拟合不收敛 ≠ 模型不适用：钝化 P1 在默认初值下是 98.40% / R̂ 197.79，
  而 `diag_passivation_init.py` 用**文件名里的真值参数**做初值（`truth_init`，
  即 oracle，生产环境拿不到）能到 6.65% / R̂ 1.004 / acf −0.244。
  这条只证明"拓扑能表达这条谱"；**P1 目前没有生产可达的初始化解法**。
  真正验证过的数据驱动初值出现在另外三条钝化谱（6.03–6.09%）与涂层 46 条；
- L2 出最终证据，但"采样成功"也不等于"模型成立"。

观测记录为什么拆成三块（0.2.0）
--------------------------------
`run_status`（拟合跑完没有）/ `sampling_status`（后验诊断过没过）/
`fit_observation`（算出了哪些残差、能不能当证据）是**三个维度**，不能用一个
`converged` 布尔代替。历史产物里到处是"跑完了、残差算了、R̂ 没过"的记录，
用一个布尔要么丢掉真实残差，要么把它伪装成收敛。0.1.0 的 E3 约束
（未收敛不得报 rms/acf）就是被这一条推翻的——它只适用于"根本没跑完"。

候选身份为什么单列一档（0.3.0）
--------------------------------
`candidate_id` 是六个字段的哈希，哈希本身不携带"凭什么认为它指向那一次运行"。
0.2.0 的映射脚本在目录名没命中时把 `init_profile` 回落成 `"uniform"`，
于是 13 条记录拿到了**看起来合法、实际无依据**的身份键；更糟的是 AZ31 的
2 条 voigt 行被算成 nested 的 id，与真实 nested 行撞车——voigt 的残差记到了
nested 名下。所以 0.3.0 规定：

- 查不到初始化就传 `INIT_UNKNOWN`，id 前缀变 `U-`，`identity_status=unconfirmed`；
- `unconfirmed` 的候选**不能进** `candidate_set_manifest`；
- 默认值只能是"不可用"，不能是"常见值"。

六个描述块（0.3.0）各回答一个不能互相替代的问题，见 `observation_block`。
其中 `budget_params`（实际数值）与 `budget_profile`（档名）必须分开：
同一档名下 `target_accept` 可以是 0.98 也可以是 0.9。

两层身份：模型 vs 执行（0.4.0）
--------------------------------
`candidate_id` 哈希的 `budget` 是**档名**，所以"两次都叫 `standard`、`target_accept`
一个 0.98 一个 0.90"能拿到同一个 `candidate_id`。把它们塞进 `candidate_id` 又会犯
反向的错：同一个模型换算力再跑一次，会被读成两个不同电路。因此：

| 层 | 号 | 含义 |
|---|---|---|
| 模型 | `candidate_id`（`C-`/`U-`） | 结构 + 元件 + 配置 + 初始化 + 似然 + 预算**档名** |
| 执行 | `run_id`（`R-`/`D-`/无） | 候选 + **数据** + seed + **实际预算数值** + 软件 |

`run_id` 是**条件指纹**，不是墙钟实例号：同条件重复执行共享同一个号
（这正是它能当去重键的原因），实例由产物路径区分。缺数据/seed/预算就不发号。

独立观测为什么也单列（0.4.0）
------------------------------
摘要相同**不能**证明两条记录来自同一次运行。0.3.0 的映射脚本用
`candidate_id + 残差 + 参数` 的哈希相同就直接标 `independent=false`，
而函数里既没读 seed 也没读产物路径。`duplicate_verdict()` 改成按来源判三态：
`distinct` / `possible_duplicate`（标记但**不扣**）/ `confirmed_duplicate`；
"同条件但结果不同"记为 `distinct` + `contradiction`，是**可复现性问题**，不靠合并记录消掉。

数据身份与产物定位（0.5.0 / 0.6.0 / 0.7.0）
-----------------------------------------
"确认重复"必须整体按 `recorded` 判（seed 来自产物 + 预算 `recorded` + 代码版本已知
+ 数据有内容哈希），只挑两项自查就是绕过自己定的门槛。数据身份认**内容**不认文件名：
`execution.data.sha256` 进 `run_id`，`ref` 只作标签；只有文件名时 `data_id_kind="ref_only"`，
最高到 `derived`、不能参与确认重复。

另一支 `same_artifact` 要求 `path` 与 `sha256` **均非空**，且文件级/行级定位
（`("file", ...)` vs `("row", ..., row, section)`）**互不相认**——"两个 `None` 相等"
不是来源证据。缺字段时原因写进 `duplicate_evidence["artifact_gap"]`。

0.7.0 再进一步：**非空不等于有效**。`sha256` 必须是 64 位十六进制，否则
`_artifact_key()` 直接抛 `ValueError`——`{"path": "x", "sha256": "same"}` 这种
"字段相等"曾经会凑出一个 `same_artifact`，与 0.5.0 的"缺失值相等"是同一个毛病，
只是换了件外衣。格式错误不是"来源缺失"，不能静默降级成 `possible_duplicate`：
那会把调用方的 bug 藏起来。

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

REGISTRY_VERSION = "0.7.0"        # 0.7：产物 sha256 必须是 64 位十六进制（非空 ≠ 有效）
PRECHECK_VERSION = "0.1.0"

PRECHECK_STATES = ("excluded", "not_excluded", "unknown")
L1_OUTCOMES = ("promote", "defer", "not_evaluated_budget")
EXEC_STATES = ("completed", "failed", "not_run")
#: 采样/后验诊断是**独立维度**：拟合跑完了不等于后验可用，反之亦然。
SAMPLING_STATES = ("passed", "failed", "not_evaluated")
#: 残差指标算出来了、但后验不可用时，只允许这一种解读，避免被读成"拟合得还行"。
INTERPRETATIONS = ("reliable", "descriptive_only")
#: 一个指标为 `None` 时的原因。**`None` 不许裸奔**——三种语义必须可区分：
#: 不适用（该配置没有这个量）/ 没算（能算但没跑）/ 算了但失败。
NULL_REASONS = ("not_applicable", "not_computed", "failed")
#: 单个候选的观测记录里允许出现、也允许缺席的指标名。
OBSERVATION_METRICS = ("rms_rel", "acf", "rhat_max", "ess_min", "divergences",
                       "budget_params")
#: 候选身份的**证据等级**。不是"完整度打分"，而是"凭什么认为这个 id 指向那一次运行"：
#: - `recorded`  产物自身写明了全部六项（拓扑/元件/配置/初始化/预算/似然）；
#: - `derived`   产物缺字段，但生成脚本唯一且代码路径可读（如 `nested_fit.py`
#:               从不传 `init_values` ⇒ numpyro `init_to_uniform`）；
#: - `unconfirmed` 生成者不可确定，或同一产物可能由多个脚本产出。
#: 只有 `recorded`/`derived` 能进 manifest；`unconfirmed` 不许当身份键用。
IDENTITY_STATES = ("recorded", "derived", "unconfirmed")
#: 初始化档不可确认时的占位值。**不能**用 `uniform` 之类真实档名冒充——
#: 默认回落成真实档名，正是 0.2.0 映射里 13 条记录身份错认的成因。
INIT_UNKNOWN = "unknown"
#: 噪声底的来源。三者不可互换：
#: - `truth_forward`：数据与**真值拓扑正演**的差（只有合成谱能给）；
#: - `same_data_estimate`：从残差自身估出来的底（与拟合共享数据，会低估）；
#: - `not_available`：没有噪声底（真实谱无真值时的常态）。
NOISE_FLOOR_SOURCES = ("truth_forward", "same_data_estimate", "not_available")
#: 一次拟合**实际使用**的计算预算参数名。`budget_profile` 只是档名（`"standard"`），
#: 两者不能互相顶替——`target_accept` 在档名相同的情况下可以差 0.98 vs 0.9。
BUDGET_PARAM_KEYS = ("n_warmup", "n_samples", "num_chains",
                     "target_accept", "max_tree_depth")
BUDGET_PARAM_SOURCES = ("recorded", "derived_from_tier", "unavailable")
#: 执行身份（0.4.0）。**`candidate_id` 与 `run_id` 是两层，不是一回事**：
#:
#: - `candidate_id` 指"哪个电路模型 + 哪套初始化/似然/预算**档名**"——
#:   同档名两次跑（`standard` 但 `target_accept` 0.98 vs 0.90）**必须**共享它，
#:   否则"同一个候选换算力再跑一次"会被读成两个不同模型；
#: - `run_id` 指"在什么**执行条件**下跑的"——候选 + seed + 实际预算数值 + 软件。
#:   两者混用会犯两个方向相反的错（见 `docs/round_2026-11-13/STAGE0_RECORDS_MAP.md` §5）。
#:
#: `run_id` 是**条件指纹**，不是墙钟实例号：同条件重复执行共享同一个 `run_id`，
#: 实例由 `source.path/sha256` 区分。这正是它能当去重键的原因。
RUN_ID_SCOPE = "execution_conditions"
#: 进 `run_id` 哈希的字段。改这里必须改 `REGISTRY_VERSION`。
#: - `data` 不能省：同一候选、同 seed、同预算下跑 60 条不同谱，若数据不进身份，
#:   这 60 条会共享一个 `run_id`，去重时被读成"同条件不同结果"（假的可复现性告警）。
#: - `seed_evidence` 单独一项而不是并进 `seed`：数值相同、一个是产物记录一个是
#:   代码默认值，可信度不同，不能拿到同一个号。
RUN_ID_BASIS = ("candidate_id", "data", "seed", "seed_evidence",
                "budget_params", "software")
#: seed 的证据等级（按前缀判定，`:` 后可写具体位置）：产物记录 / 生成脚本代码路径 /
#: 档名默认值 / 没有。`artifact` 以外的都推不出"同一次执行的同一随机流"。
SEED_SOURCE_KINDS = ("artifact", "producer_code", "tier", "absent")
#: 数据身份的等级。**文件名不是身份**：两个实验都可能叫 `spectrum.csv`，
#: 频率窗也一样。`content_sha256` 是唯一能说"跑的是同一份数据"的凭据；
#: `ref_only` 只能当阅读标签，不能据此宣称唯一，也不够格升到 `recorded`。
DATA_ID_KINDS = ("content_sha256", "ref_only")
#: `run_id` 不可发时的原因。**不许发一个看起来合法的 id**。
#: `not_provided` 是"调用方根本没给执行块"的哨兵——它必须在枚举里，否则
#: `unconfirmed ⇒ 原因 ∈ RUN_ID_MISSING` 这条不变量在默认块上就已经不成立。
RUN_ID_MISSING = ("not_provided", "data_not_recorded", "seed_not_recorded",
                  "budget_not_recorded")
#: 重复判定的三态。**`independent` 只是它的派生值**，不能反过来用摘要相同去推。
#: - `distinct`：没有证据表明是重复（含"摘要撞车但来源不同"）；
#: - `possible_duplicate`：摘要相同、但来源证据不足 ⇒ **不从独立样本数里扣除**；
#: - `confirmed_duplicate`：来源证据足够——同产物，或同 `run_id` 且**两侧执行身份
#:   都是 `recorded`**（seed 由产物记录 + 预算 `recorded` + 代码版本已知 + 数据有内容
#:   哈希）⇒ 计数时必须合并。`derived` 的执行身份**不足以**确认重复：同 seed 同预算
#:   同摘要仍可能出自不同版本的拟合代码。
DUPLICATE_STATES = ("distinct", "possible_duplicate", "confirmed_duplicate")
DUPLICATE_KINDS = ("same_artifact", "rerun_same_conditions")
#: 产物定位的层级。**文件级与行级不能互相顶替**：`row` 缺失只说明这份产物一条记录
#: 对应整份文件，不说明"两个缺失值相等 ⇒ 同一行"。
ARTIFACT_SCOPES = ("file", "row")
#: 产物定位必须非空的字段。少一个就只能到 `possible_duplicate`，不能确认同一产物。
ARTIFACT_KEY_FIELDS = ("path", "sha256")
#: 内容哈希的合法形态：`SHA256_HEX_LEN` 位十六进制（大小写不敏感）。**非空不等于有效**：
#: `"same"`、文件名、63 位截断哈希都是非空字符串，却不能证明"跑的是同一份东西"。
SHA256_HEX_LEN = 64
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
    consistency_id: str | None = None   # 结构校验规则的**显式版本**，进 registry_digest
    notes: tuple = ()

    def __post_init__(self):
        if len(self.skeleton.strip()) < 20:
            raise ValueError(f"拓扑 {self.id}：skeleton 太短，必须写清连接结构")
        if not self.param_space:
            raise ValueError(f"拓扑 {self.id}：param_space 不能为空")
        _check_ref(self.forward_ref, f"拓扑 {self.id}.forward_ref")
        if self.precheck_ref is not None:
            _check_ref(self.precheck_ref, f"拓扑 {self.id}.precheck_ref")
        if (self.consistency is None) != (self.consistency_id is None):
            raise ValueError(
                f"拓扑 {self.id}：consistency 与 consistency_id 必须同时给或同时不给。"
                "校验规则改了而 id 不变，registry_digest 就发现不了结构约束变化。")
        if self.consistency_id is not None and len(str(self.consistency_id).strip()) < 4:
            raise ValueError(f"拓扑 {self.id}：consistency_id 太短，需要可读的版本标识")
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

    `identity_status`
    ----------------
    id 是哈希，哈希本身不告诉你"凭什么认为它指向那一次运行"。所以另带一列：

    - `recorded`：产物写明了全部六项；
    - `derived`：产物缺项，但生成脚本唯一且代码路径可读；
    - `unconfirmed`：初始化不可确认 → **id 前缀改成 `U-`**，且不许进 manifest。

    为什么要这条：0.2.0 的映射脚本在目录名没命中时把 `init_profile` 回落成
    `"uniform"`，于是 13 条记录拿到了一个**看起来合法、实际无依据**的身份键，
    其中 2 条 AZ31 的 voigt 行甚至被算成了 nested 的 id，与真实 nested 行撞车。
    默认值必须是"不可用"，不能是"常见值"。
    """
    candidate_id: str
    topology_id: str
    elements: tuple              # tuple[(slot_id, element_id)]
    config: tuple                # tuple[(key, value)]，已排序
    init_profile: str
    budget_profile: str
    likelihood_profile: str
    identity_status: str = "recorded"
    init_source: str = "recorded"

    @property
    def config_dict(self) -> dict:
        return dict(self.config)

    @property
    def comparable(self) -> bool:
        """身份可复现才允许进候选集比较。`unconfirmed` 只能作文档性记录。"""
        return self.identity_status in ("recorded", "derived")

    def describe(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "topology": self.topology_id,
            # 用 list 不用 dict：voigt 的 K 个 HN 单元挂的是同一槽位，
            # dict 会把它们压成一条，定义里就看不出实际有几个单元。
            "elements": [list(e) for e in self.elements],
            "config": self.config_dict,
            "init_profile": self.init_profile,
            "budget_profile": self.budget_profile,
            "likelihood_profile": self.likelihood_profile,
            "identity_status": self.identity_status,
            "init_source": self.init_source,
            "comparable": self.comparable,
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
    prefix = "U-" if init_profile == INIT_UNKNOWN else "C-"
    return prefix + hashlib.sha256(blob).hexdigest()[:10]


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
                   likelihood_profile: str = "perpoint-2pct",
                   identity_status: str | None = None,
                   init_source: str = "recorded") -> Candidate:
    """构造并校验一个候选。

    `elements` 形如 `[("relaxation_unit", "hn"), ("relaxation_unit", "hn")]`
    或 `[("branch", "cpe")]`，同一槽位可以出现多次（voigt 的 K 个单元）。

    `init_profile=INIT_UNKNOWN`（`"unknown"`）是合法输入，表示"这次运行的初始化
    查不到"。它会把 `identity_status` 强制为 `unconfirmed`、id 前缀改成 `U-`，
    并且**拒绝**被写进 manifest。不要为了让记录看起来完整而填一个真实档名。
    """
    topo = get_topology(topology_id)
    el = tuple((str(a), str(b)) for a, b in elements)
    cfg = dict(config or {})

    if init_profile == INIT_UNKNOWN:
        if identity_status not in (None, "unconfirmed"):
            raise ValueError(
                f"init_profile={INIT_UNKNOWN!r} 与 identity_status={identity_status!r} "
                "矛盾：初始化没查到时身份只能是 unconfirmed")
        identity_status = "unconfirmed"
    else:
        if init_profile not in topo.init_profiles:
            raise ValueError(
                f"拓扑 {topology_id} 未登记 init_profile {init_profile!r}；"
                f"已有 {sorted(topo.init_profiles)}。"
                "查不到就传 INIT_UNKNOWN（会强制 U- 前缀），不要用近似档名顶替")
        if identity_status is None:
            identity_status = "recorded"

    if identity_status not in IDENTITY_STATES:
        raise ValueError(
            f"identity_status={identity_status!r} 不在 {IDENTITY_STATES}")

    cand = Candidate(
        candidate_id=compute_candidate_id(topology_id, el, cfg, init_profile,
                                         budget_profile, likelihood_profile),
        topology_id=topology_id,
        elements=el,
        config=tuple(sorted((str(k), _jsonable(v)) for k, v in cfg.items())),
        init_profile=init_profile,
        budget_profile=budget_profile,
        likelihood_profile=likelihood_profile,
        identity_status=identity_status,
        init_source=init_source,
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
                    precheck_records: "list | tuple | None" = None,
                    notes: tuple = ()) -> dict:
    """筛选层。只记数量关系，不出任何证据结论。

    `survivors` 是**拟合后仍 viable 的候选数**（阶段 0 不跑拟合，传 None）。
    `only_survivor` 由这里算，不接受调用方传入——这是硬约束。

    `precheck_records` 是 `l0_precheck` 对每个候选的逐条结果。**数量汇总必须
    能被逐条记录复算出来**，否则"2 个被排除"就只能靠调用方自述，无法核对。
    给了记录就必须覆盖全部候选，且各状态计数与四个入参一致。
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

    records = None
    if precheck_records is not None:
        records = [_check_precheck_record(r) for r in precheck_records]
        if len(records) != registered:
            raise ValueError(
                f"screening：逐条预检记录 {len(records)} 条，与 registered={registered} 不符")
        ids = [r["candidate_id"] for r in records]
        if len(set(ids)) != len(ids):
            raise ValueError("screening：逐条预检记录里有重复 candidate_id")
        tally = {s: 0 for s in PRECHECK_STATES}
        for r in records:
            tally[r["precheck_state"]] += 1
        got = dict(excluded=tally["excluded"],
                   not_excluded=tally["not_excluded"], unknown=tally["unknown"])
        want = dict(excluded=excluded, not_excluded=not_excluded, unknown=unknown)
        if got != want:
            raise ValueError(
                f"screening：数量汇总 {want} 与逐条记录复算 {got} 不一致；"
                "汇总不能脱离逐条记录自述")

    return {
        "registered": registered,
        "excluded": excluded,
        "not_excluded": not_excluded,
        "unknown": unknown,
        "entered_fit": not_excluded + unknown,
        "survivors": survivors,
        "only_survivor": survivors == 1,
        "precheck_records": records,
        "notes": list(notes),
    }


def _check_precheck_record(rec: Mapping) -> dict:
    """校验单条 L0 预检记录，返回规范化副本。"""
    if not isinstance(rec, Mapping):
        raise ValueError("precheck 记录必须是映射")
    need = ("candidate_id", "precheck_state", "precheck_config",
            "precheck_reason", "precheck_version")
    missing = [k for k in need if k not in rec]
    if missing:
        raise ValueError(
            f"precheck 记录缺 {missing}；排除理由与所用配置必须随记录保存，"
            "否则事后无法核对它凭什么被排除")
    if rec["precheck_state"] not in PRECHECK_STATES:
        raise ValueError(
            f"precheck_state={rec['precheck_state']!r} 不在 {PRECHECK_STATES}")
    if rec["precheck_state"] == "excluded" and not str(rec["precheck_reason"]).strip():
        raise ValueError("precheck_state=excluded 必须给 precheck_reason")
    if not isinstance(rec["precheck_config"], Mapping):
        raise ValueError("precheck_config 必须是映射（记录实际参与的配置项）")
    return {
        "candidate_id": str(rec["candidate_id"]),
        "precheck_state": str(rec["precheck_state"]),
        "precheck_config": {str(k): _jsonable(v) for k, v in rec["precheck_config"].items()},
        "precheck_reason": str(rec["precheck_reason"]),
        "precheck_version": str(rec["precheck_version"]),
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


def observation_block(*, candidate_id: str, run_status: str,
                      sampling_status: str = "not_evaluated",
                      rms_rel: float | None = None, acf: float | None = None,
                      rhat_max: float | None = None, ess_min: float | None = None,
                      divergences: int | None = None,
                      param_stability: Mapping | None = None,
                      param_estimate: Mapping | None = None,
                      ratio: Mapping | None = None,
                      noise_floor: Mapping | None = None,
                      predictive: Mapping | None = None,
                      budget_params: Mapping | None = None,
                      execution: Mapping | None = None,
                      truth_known: bool = False,
                      interpretation: str | None = None,
                      unavailable: Mapping | None = None,
                      notes: tuple = ()) -> dict:
    """单个候选单次运行的**描述性**记录。不出结论、不排序、不合成总分。

    三件事分开记，任何一个都不能代表另外两个：

    | 字段 | 回答的问题 | 取值 |
    |---|---|---|
    | `run_status` | 拟合过程跑完了吗 | `completed` / `failed` / `not_run` |
    | `sampling_status` | 后验诊断过了吗 | `passed` / `failed` / `not_evaluated` |
    | `fit_observation` | 已经算出了哪些残差、能否当证据 | 指标 + `interpretation` |

    **为什么不能用一个 `converged` 布尔**（0.1.0 的错，已改）：历史产物里到处是
    "跑完了、残差也算出来了、但 R̂ 没过"的记录——涂层 `parallel` 是
    10.81% / acf 0.944 / R̂ 14.21，钝化默认初值是 98.40% / 0.750 / R̂ 197.79。
    用单布尔表达，要么把真实残差丢掉（schema 装不下自己的压测数据），
    要么把它伪装成 `converged=True`（污染证据）。这两种都违背阶段 0 的初衷。

    硬约束（都可执行，不是注释）：
    - `run_status != "completed"` 时**不得**报任何残差指标：没跑完就没有残差。
      原来的 E3「未收敛不报数」只适用于这一种情形，不适用于"跑完但采样没过"。
    - `sampling_status == "failed"` 而报了指标，`interpretation` 必须**显式**为
      `descriptive_only`。不设默认值——默认值会让调用方忘掉这句限定。
    - `sampling_status == "not_evaluated"` 同样只能是 `descriptive_only`：
      没做过诊断就不能声称可靠。
    - `run_status == "completed"` 时，`rms_rel` / `acf` 要么给值，要么在
      `unavailable` 里给原因；不许静默缺席。
    - `unavailable` 的键必须是指标名，值必须是 `NULL_REASONS` 之一；
      给了值的指标不许同时出现在 `unavailable` 里。

    `acf` 允许负值（钝化那条 `truth_init` 实测 −0.244），不要假设它在 [0,1]。

    `param_stability` 必须是"参数名 → 描述块"，每块含 `ci90` 或 `flag`，
    **不允许**传单个标量分数：把参数可信度压成一个数，等于又做了一次隐式判决。

    六个描述块各回答一个**互不替代**的问题
    ----------------------------------------
    | 字段 | 回答的问题 | 不许做的事 |
    |---|---|---|
    | `param_estimate` | 点估计算到多少、离真值多远 | 真实谱伪造真值列 |
    | `param_stability` | 后验区间稳不稳、参数可辨识吗 | 压成单一分数 |
    | `ratio` | 残差相对噪声底几倍 | 不报 `n_used`/`n_total` |
    | `noise_floor` | 噪声底从哪来 | 拿同数据估计冒充真值正演 |
    | `predictive` | ELPD 能不能当样本外证据 | 不可用却不写原因 |
    | `budget_params` | 这次实际用多少算力 | 拿档名（`"standard"`）顶替数值 |

    `param_estimate`：`{参数名: {"estimate":..., "truth":..., "rel_err":...}}`。
    给了 `truth` 才算 `rel_err`；`truth_known=False` 时出现 `truth` 直接报错——
    真实谱没有真值，伪造这一列会让"参数恢复"变成自证。

    `ratio`：`{"value":..., "n_used":..., "n_total":..., "noise_floor_source":...}`。
    `n_used`/`n_total` 是**必填**：中位数依赖分母，历史上出现过收敛子集口径
    3.36 与全样本口径 6.47 被混报。`n_used < n_total` 时自动标 `denominator_scope`。

    `predictive`：`{"elpd":..., "se":..., "valid_for_comparison":bool, "reason":...}`。
    `valid_for_comparison=False` 时 `reason` 非空——`sigma_from_data=True` 下
    `elpd_loo` 不是 `p(Z_j|Z_{-j})`，这个事实必须跟着数字走。

    `budget_params`：实际数值 + `source`。`"recorded"` 要求五个核心参数齐全；
    `"derived_from_tier"` 表示产物只记档名、数值由代码推出（可复现性弱一档），
    这种情形必须带 `tier`（档名），否则数值无从复核；`"unavailable"` 要写 `reason`。

    `execution`：`execution_identity()` 的返回值——**执行条件**身份（候选 + seed +
    实际预算 + 软件），与 `candidate_id`（模型定义）是两层。给了 `run_id` 就必须
    同时给 `budget_params`，函数会**重算**并比对；不给则记
    `{"run_id": None, "run_id_status": "unconfirmed", "unavailable_reason": "not_computed"}`，
    不伪造身份。
    """
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("observation.candidate_id 必须是非空字符串")
    if run_status not in EXEC_STATES:
        raise ValueError(
            f"observation.run_status={run_status!r} 不在 {EXEC_STATES}；"
            "它只描述拟合过程，不描述后验质量（后者用 sampling_status）")
    if sampling_status not in SAMPLING_STATES:
        raise ValueError(
            f"observation.sampling_status={sampling_status!r} 不在 {SAMPLING_STATES}")

    raw = dict(rms_rel=rms_rel, acf=acf, rhat_max=rhat_max,
               ess_min=ess_min, divergences=divergences)
    provided = {k: v for k, v in raw.items() if v is not None}

    if run_status != "completed" and provided:
        raise ValueError(
            f"候选 {candidate_id} run_status={run_status!r}，不得报 {sorted(provided)}："
            "没跑完就没有残差。「跑完了但采样没过」请用 "
            "run_status='completed' + sampling_status='failed' + "
            "interpretation='descriptive_only'，不要走这一支。")

    unavail = {}
    if unavailable is not None:
        if not isinstance(unavailable, Mapping):
            raise ValueError("unavailable 必须是 指标名→原因 的映射")
        for k, why in unavailable.items():
            if k not in OBSERVATION_METRICS:
                raise ValueError(f"unavailable[{k!r}] 不是已知指标；可用 {OBSERVATION_METRICS}")
            if why not in NULL_REASONS:
                raise ValueError(
                    f"unavailable[{k!r}]={why!r} 不在 {NULL_REASONS}；"
                    "三种缺失语义必须分开：不适用 / 没算 / 算了但失败")
            if k in provided:
                raise ValueError(
                    f"候选 {candidate_id}：{k} 既有值 {provided[k]!r} 又被标成 "
                    f"{why!r}，二者互斥")
            unavail[str(k)] = str(why)

    if run_status == "completed":
        silent = [k for k in ("rms_rel", "acf") if raw[k] is None and k not in unavail]
        if silent:
            raise ValueError(
                f"候选 {candidate_id} 已跑完，但 {silent} 既无值也无原因；"
                f"请在 unavailable 里标明 {NULL_REASONS} 之一")

    if provided:
        if sampling_status == "passed":
            interp = interpretation or "reliable"
        else:
            interp = interpretation
            if interp != "descriptive_only":
                raise ValueError(
                    f"候选 {candidate_id} sampling_status={sampling_status!r} "
                    f"却报了解读 {interp!r}。"
                    "后验没通过（或没诊断）时，残差只能作描述性记录，"
                    "必须显式写 interpretation='descriptive_only'")
        if interp not in INTERPRETATIONS:
            raise ValueError(f"interpretation={interp!r} 不在 {INTERPRETATIONS}")
    else:
        if interpretation is not None:
            raise ValueError("没有报任何指标时不应给 interpretation")
        interp = None

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

    est = _check_param_estimate(param_estimate, truth_known=truth_known,
                                candidate_id=candidate_id)
    floor = _check_noise_floor(noise_floor, candidate_id=candidate_id)
    rat = _check_ratio(ratio, floor, candidate_id=candidate_id)
    pred = _check_predictive(predictive, candidate_id=candidate_id)
    bud = _check_budget_params(budget_params, candidate_id=candidate_id)
    exe = _check_execution(execution, candidate_id=candidate_id, budget_params=budget_params)

    return {
        "candidate_id": candidate_id,
        "run_status": run_status,
        "sampling_status": sampling_status,
        "fit_observation": {
            "rms_rel": None if rms_rel is None else float(rms_rel),
            "acf": None if acf is None else float(acf),
            "sampling_diagnostics": {
                "rhat_max": None if rhat_max is None else float(rhat_max),
                "ess_min": None if ess_min is None else float(ess_min),
                "divergences": None if divergences is None else int(divergences),
            },
            "interpretation": interp,
            "unavailable": unavail,
        },
        "param_estimate": est,
        "param_stability": stab,
        "ratio": rat,
        "noise_floor": floor,
        "predictive": pred,
        "budget_params": bud,
        "execution": exe,
        "notes": list(notes),
    }


def _check_param_estimate(v, *, truth_known: bool, candidate_id: str):
    """点估计 vs 真值。**真实谱不许有真值列**——没有真值就没有"偏差"。"""
    if v is None:
        return None
    if not isinstance(v, Mapping):
        raise ValueError("param_estimate 必须是 参数名→{estimate,truth?,rel_err?} 的映射")
    out = {}
    for k, blk in v.items():
        if not isinstance(blk, Mapping):
            raise ValueError(f"param_estimate[{k!r}] 必须是描述块，不能是标量")
        item = {"estimate": _jsonable(blk.get("estimate"))}
        has_truth = blk.get("truth") is not None
        has_err = blk.get("rel_err") is not None
        if (has_truth or has_err) and not truth_known:
            raise ValueError(
                f"候选 {candidate_id}：param_estimate[{k!r}] 带 "
                f"{'truth' if has_truth else 'rel_err'}，但 truth_known=False。"
                "真实谱没有真值，不能伪造这一列——参数恢复只在有真值的合成谱上成立")
        if has_truth:
            item["truth"] = _jsonable(blk["truth"])
            if not has_err:
                raise ValueError(
                    f"候选 {candidate_id}：param_estimate[{k!r}] 给了 truth 却没给 rel_err")
        if has_err:
            # 只记了误差、没记真值也合法：产物常常只存 |fit/truth − 1|。
            item["rel_err"] = float(blk["rel_err"])
        if blk.get("flag") is not None:
            item["flag"] = str(blk["flag"])
        out[str(k)] = item
    return out


def _check_noise_floor(v, *, candidate_id: str):
    if v is None:
        return None
    if not isinstance(v, Mapping):
        raise ValueError("noise_floor 必须是 {value, source, note?} 的映射")
    src = v.get("source")
    if src not in NOISE_FLOOR_SOURCES:
        raise ValueError(
            f"候选 {candidate_id}：noise_floor.source={src!r} 不在 {NOISE_FLOOR_SOURCES}")
    val = v.get("value")
    if src == "not_available" and val is not None:
        raise ValueError(f"候选 {candidate_id}：source=not_available 却给了 value")
    if src != "not_available" and val is None:
        raise ValueError(
            f"候选 {candidate_id}：noise_floor.source={src!r} 却没有 value")
    return {"value": None if val is None else float(val),
            "source": str(src),
            "note": str(v.get("note") or "")}


def _check_ratio(v, floor, *, candidate_id: str):
    """残差 / 噪声底。分母口径必须随数走，否则中位数无法复核。"""
    if v is None:
        return None
    if not isinstance(v, Mapping):
        raise ValueError("ratio 必须是 {value,n_used,n_total,noise_floor_source} 的映射")
    val = v.get("value")
    if val is None:
        why = v.get("unavailable_reason")
        if why not in NULL_REASONS:
            raise ValueError(
                f"候选 {candidate_id}：ratio 没有值时必须给 "
                f"unavailable_reason ∈ {NULL_REASONS}")
        return {"value": None, "unavailable_reason": str(why),
                "n_used": None, "n_total": None, "denominator_scope": None}
    n_used, n_total = v.get("n_used"), v.get("n_total")
    if not isinstance(n_used, int) or not isinstance(n_total, int):
        raise ValueError(
            f"候选 {candidate_id}：ratio 报了值就必须给整数 n_used/n_total。"
            "历史上出现过收敛子集 3.36 与全样本 6.47 被混报")
    if not 0 <= n_used <= n_total:
        raise ValueError(f"候选 {candidate_id}：ratio 的 n_used={n_used} 越界（n_total={n_total}）")
    # `n_usable`：其中**后验诊断通过**的条数。文献/报告里的中位数常只在这部分上算
    # （pilot 的 voigt 自由 σ：60 条都有比值，但只有 14 条收敛）。
    # 不写它就等于默认"比值有定义 = 可以拿来做统计"，那正是被混报的地方。
    n_usable = v.get("n_usable", n_used)
    if not isinstance(n_usable, int) or not 0 <= n_usable <= n_used:
        raise ValueError(
            f"候选 {candidate_id}：ratio.n_usable={n_usable!r} 必须在 [0, {n_used}]")
    src = v.get("noise_floor_source")
    if src not in NOISE_FLOOR_SOURCES or src == "not_available":
        raise ValueError(
            f"候选 {candidate_id}：ratio 的 noise_floor_source={src!r} 无效——"
            "没有噪声底就算不出比值")
    if floor is not None and floor.get("source") != src:
        raise ValueError(
            f"候选 {candidate_id}：ratio 声明的底来源 {src!r} 与 "
            f"noise_floor.source={floor.get('source')!r} 不一致")
    return {"value": float(val), "n_used": n_used, "n_total": n_total,
            "n_usable": n_usable,
            "usable_rule": str(v.get("usable_rule") or "sampling_status==passed"),
            "denominator_scope": ("all_runs" if n_usable == n_total
                                  else "converged_subset"),
            "noise_floor_source": str(src)}


def _check_predictive(v, *, candidate_id: str):
    """ELPD 的可用性必须和数值绑在一起走。"""
    if v is None:
        return None
    if not isinstance(v, Mapping):
        raise ValueError("predictive 必须是 {elpd,se,valid_for_comparison,reason} 的映射")
    valid = v.get("valid_for_comparison")
    if not isinstance(valid, bool):
        raise ValueError(
            f"候选 {candidate_id}：predictive.valid_for_comparison 必须显式布尔，"
            "不能留空——'算了但不可用'与'可用'必须能区分")
    reason = (v.get("reason") or "").strip()
    if not valid and not reason:
        raise ValueError(
            f"候选 {candidate_id}：predictive.valid_for_comparison=False 必须给 reason"
            "（逐点 σ 下 elpd_loo 不是 p(Z_j|Z_{-j})，这条要跟着数字走）")
    return {"elpd": None if v.get("elpd") is None else float(v["elpd"]),
            "se": None if v.get("se") is None else float(v["se"]),
            "valid_for_comparison": valid,
            "reason": reason}


def _check_budget_params(v, *, candidate_id: str):
    """实际预算数值。**档名不是数值**——`target_accept` 同档可差 0.98 vs 0.9。"""
    if v is None:
        return None
    if isinstance(v, str):
        raise ValueError(
            f"候选 {candidate_id}：budget_params 收到字符串 {v!r}。"
            "那是 budget_profile（档名），不是实际数值；两者不能互相顶替")
    if not isinstance(v, Mapping):
        raise ValueError("budget_params 必须是 {source, n_warmup?, ...} 的映射")
    src = v.get("source")
    if src not in BUDGET_PARAM_SOURCES:
        raise ValueError(
            f"候选 {candidate_id}：budget_params.source={src!r} 不在 {BUDGET_PARAM_SOURCES}")
    if src == "unavailable":
        why = v.get("reason")
        if why not in NULL_REASONS:
            raise ValueError(
                f"候选 {candidate_id}：budget_params 不可用时 reason 必须在 {NULL_REASONS}")
        return {"source": src, "reason": str(why)}
    out = {"source": str(src)}
    for k in BUDGET_PARAM_KEYS:
        if v.get(k) is None:
            continue
        val = v[k]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(
                f"候选 {candidate_id}：budget_params[{k!r}]={val!r} 不是数值")
        out[k] = float(val) if isinstance(val, float) else int(val)
    if src == "recorded":
        missing = [k for k in BUDGET_PARAM_KEYS if k not in out]
        if missing:
            raise ValueError(
                f"候选 {candidate_id}：source=recorded 时预算参数必须齐全，缺 {missing}。"
                "只记档名请用 derived_from_tier")
    if src == "derived_from_tier":
        # 档名是数值的**推导依据**，不写就等于数值无从复核。只保留字符串，不做换算。
        tier = v.get("tier")
        if tier is not None:
            if not isinstance(tier, str) or not tier.strip():
                raise ValueError(
                    f"候选 {candidate_id}：budget_params.tier 必须是非空字符串（档名）")
            out["tier"] = tier.strip()
    return out


def execution_identity(*, candidate_id: str, data=None, seed: int | None = None,
                       seed_source: str = "absent",
                       budget_params: Mapping | None = None,
                       software: Mapping | None = None,
                       notes: tuple = ()) -> dict:
    """**执行身份**：候选 + 数据 + seed + 实际预算数值 + 软件。

    为什么不能只扩 `candidate_id`（0.3.0 的缺口）
    --------------------------------------------
    `candidate_id` 哈希的是 `budget_profile`（档名 `"standard"`），不是
    `target_accept=0.98` 这类实际数值。同一次历史对照里，
    `VALIDATION_STANDARD` 的 `standard` 是 0.98，`BUDGETS["standard"]` 是 0.90——
    两个 `standard` 能拿到同一个 `candidate_id`。把它们塞进 `candidate_id` 又会
    犯反向的错：同一个模型换算力再跑一次，会被读成两个不同电路。

    所以：**`candidate_id` 不动**（模型定义层），执行条件单独一层。
    反过来，执行层也不能只看模型：同一候选在同一预算下跑不同谱，是 N 次不同执行，
    所以 `data`（谱标识 + 拟合频率窗口）必须进身份。

    三档状态与 `IDENTITY_STATES` 同名同义，门槛写在代码里：
    - `recorded`：seed 由产物记录 + 预算 `recorded` + 代码版本已知 + 数据有内容哈希，
      四者齐；
    - `derived`：至少一项靠代码路径/档名推出，或数据只有文件名（本仓库当前绝大多数
      是这一档）；
    - `unconfirmed`：数据/seed/预算缺 ⇒ **不发 `run_id`**（`None` + 原因），
      而不是拿默认值拼一个看起来合法的 id。
    """
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("execution.candidate_id 必须是非空字符串")
    if candidate_id.startswith("U-"):
        raise ValueError(
            f"候选 {candidate_id} 的身份未确认（U- 前缀），它的执行身份不能高于候选身份。"
            "先把 init_profile 查清（recorded/derived），再来发 run_id。")

    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"execution.seed 必须是整数，收到 {seed!r}")
    kind = _seed_kind(seed_source)
    if seed is None and kind != "absent":
        raise ValueError(
            f"seed=None 时 seed_source 必须是 'absent'，收到 {seed_source!r}："
            "没有 seed 就没有随机流信息，不能声称它来自某处")
    if seed is not None and kind == "absent":
        raise ValueError(f"给了 seed={seed} 却把来源标成 absent，二者矛盾")

    dat = _check_data(data, candidate_id=candidate_id)
    bud = _check_budget_params(budget_params, candidate_id=candidate_id)
    soft = _check_software(software, candidate_id=candidate_id)

    status, reason = _expected_status(dat, seed, kind, bud, soft)

    run_id = None
    if status != "unconfirmed":
        run_id = _run_id_from(candidate_id, dat, int(seed), kind, bud, soft, status)

    return {
        "run_id": run_id,
        "run_id_status": status,
        "run_id_scope": RUN_ID_SCOPE,
        "run_id_basis": list(RUN_ID_BASIS),
        "data": dat,
        # 数据身份是"内容"还是"文件名"必须单独留痕：`run_id` 哈希了它，但哈希之后
        # 看不出来，而"能不能确认重复"（`duplicate_verdict`）恰好取决于这一点。
        "data_id_kind": None if dat is None else dat["id_kind"],
        "seed": None if seed is None else int(seed),
        "seed_source": str(seed_source),
        # 预算的证据等级单独留一份：`run_id` 里哈希了它，但哈希之后看不出来，
        # 而"能不能扣掉一条重复"恰好取决于这一点（`duplicate_verdict` 要用）。
        "budget_source": None if bud is None else bud["source"],
        "software": soft,
        "unavailable_reason": reason,
        "notes": list(notes),
    }


def _check_data(v, *, candidate_id: str) -> dict | None:
    """这次执行**跑的是哪条谱、在哪个频率窗口**。缺了就不发执行身份。

    只认标识，不认数组：把 Z 本身塞进哈希会让 id 随浮点末位漂移。

    `sha256` 是数据**内容**的哈希，`ref` 只是给人看的标签。两者的区别不是洁癖：

    | 情形 | 身份 | 后果 |
    |---|---|---|
    | 有 `sha256` | `content_sha256` | 两条记录能证明"跑的是同一份数据" |
    | 只有 `ref` | `ref_only` | 只能说"文件名一样"；**不够格升 `recorded`** |

    历史产物引用的是 `R1-[P2,R3-[P4,R5-P6]]-[150.0, ...].csv` 这种文件名即真值的
    谱，文件名本身信息量很高但仍然不是内容——同一个名字换一批生成参数就会变内容。
    所以 `sha256` 一栏在 schema 里是**显式可选但会降级**，而不是可以静默省掉。
    """
    if v is None:
        return None
    if isinstance(v, str):
        if not v.strip():
            raise ValueError(f"候选 {candidate_id}：execution.data 不能是空串")
        return {"ref": v.strip(), "sha256": None, "f_min": None, "f_max": None,
                "id_kind": "ref_only"}
    if not isinstance(v, Mapping):
        raise ValueError(
            "execution.data 必须是谱标识字符串或 {ref, sha256?, f_min?, f_max?}")
    ref = v.get("ref")
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError(f"候选 {candidate_id}：execution.data.ref 必须是非空字符串")
    sha = v.get("sha256")
    if sha is not None:
        if not isinstance(sha, str):
            raise ValueError(
                f"候选 {candidate_id}：execution.data.sha256 必须是 64 位十六进制字符串")
        sha = sha.strip().lower()
        if not _valid_sha256(sha):
            raise ValueError(
                f"候选 {candidate_id}：execution.data.sha256={sha!r} 不是 64 位十六进制。"
                "别把文件名当哈希填进来——那会让两条不同数据拿到同一个身份")
    out = {"ref": ref.strip(), "sha256": sha, "f_min": None, "f_max": None,
           "id_kind": "content_sha256" if sha else "ref_only"}
    for k in ("f_min", "f_max"):
        if v.get(k) is None:
            continue
        val = v[k]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"候选 {candidate_id}：execution.data[{k!r}] 不是数值")
        out[k] = float(val)
    return out


def _expected_status(dat, seed, kind: str, bud, soft) -> tuple:
    """给定证据要素，**应当**得到什么状态与原因。执行身份与校验共用这一处规则。

    三档门槛（顺序即优先级，缺哪一项就报哪一项）：
    `data` → `seed` → `budget` → 才是 `recorded` 的四个条件。
    """
    if dat is None:
        return "unconfirmed", "data_not_recorded"
    if seed is None:
        return "unconfirmed", "seed_not_recorded"
    if bud is None or bud.get("source") == "unavailable":
        return "unconfirmed", "budget_not_recorded"
    if (kind == "artifact" and bud.get("source") == "recorded"
            and soft.get("code_version") and dat.get("id_kind") == "content_sha256"):
        return "recorded", None
    return "derived", None


def _run_id_from(candidate_id: str, dat: Mapping, seed: int, seed_kind: str,
                 bud: Mapping, soft: Mapping, status: str) -> str:
    # 数据只哈希**内容**（有 sha256 时），不哈希 `ref`：`ref` 可能是完整路径，
    # 同一个实验换台机器就会换号。没有内容哈希时才退到 `ref`，并由 `id_kind`
    # 记录这次降级——哈希之后看不出来的东西必须在 `data_id_kind` 里留痕。
    did = dat.get("sha256") or dat.get("ref")
    payload = {
        "candidate_id": candidate_id,
        "data": {"id_kind": dat.get("id_kind"), "id": did,
                 "f_min": _jsonable(dat.get("f_min")),
                 "f_max": _jsonable(dat.get("f_max"))},
        "seed": int(seed),
        "seed_evidence": seed_kind,
        # source 一起哈希：数值相同但证据不同（产物记录 vs 档名推导）**不能**
        # 拿到同一个号，否则推出来的条件和观测到的条件会被混成同一个身份。
        "budget_params": {k: _jsonable(v) for k, v in bud.items()},
        "software": {k: _jsonable(v) for k, v in soft.items()},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return ("R-" if status == "recorded" else "D-") + hashlib.sha256(blob).hexdigest()[:10]


def _seed_kind(seed_source: str) -> str:
    if not isinstance(seed_source, str) or not seed_source.strip():
        raise ValueError("seed_source 必须是非空字符串（如 'artifact:config.seed'）")
    kind = seed_source.split(":", 1)[0].strip()
    if kind not in SEED_SOURCE_KINDS:
        raise ValueError(
            f"seed_source={seed_source!r} 的类别 {kind!r} 不在 {SEED_SOURCE_KINDS}；"
            "必须能看出 seed 是产物写的、代码推的、还是档名默认的")
    return kind


def _check_software(v, *, candidate_id: str) -> dict:
    """代码/软件版本。**未知就写 None，不要拿注册表版本冒充拟合器版本**——
    后者是映射工具的版本，和被拟合那次运行的代码不是一回事。"""
    if v is None:
        return {"producer": None, "code_version": None,
                "registry_version": REGISTRY_VERSION}
    if not isinstance(v, Mapping):
        raise ValueError("execution.software 必须是 {producer, code_version, ...} 的映射")
    code = v.get("code_version")
    if code is not None and (not isinstance(code, str) or not code.strip()):
        raise ValueError(f"候选 {candidate_id}：software.code_version 必须是非空字符串或 None")
    return {"producer": None if v.get("producer") is None else str(v["producer"]),
            "code_version": None if code is None else code.strip(),
            "registry_version": REGISTRY_VERSION}


def _check_execution(v, *, candidate_id: str, budget_params: Mapping | None):
    """校验执行身份块，并**整块重算**（状态、原因、`run_id`）与观测对拍。

    四条硬约束，都是历史错误的直接对策：
    1. `run_id` 前缀必须与 `run_id_status` 一致（`R-`/`D-`/`None`），
       否则"推出来的执行条件"会被当成产物记录读；
    2. **状态也要重算**，不能只信 `run_id_status` 里的字符串。0.4.0 只重算
       `run_id`、对 `unconfirmed` 分支只查了"原因在枚举里"，于是
       "seed、数据、预算都齐却自称缺 seed" 能通过——这正是审计 D6 那条；
    3. 重算：用观测里的 `budget_params` 重新哈希，必须等于给定的 `run_id`。
       这样预算数值事后被改动、或 run_id 从别处抄来，都会被当场抓住；
    4. 未发 `run_id` 时必须写原因，且原因要**指向真正缺的那一项**
       （`data` → `seed` → `budget` 的优先序）。
    """
    if v is None:
        return {"run_id": None, "run_id_status": "unconfirmed",
                "run_id_scope": RUN_ID_SCOPE, "run_id_basis": list(RUN_ID_BASIS),
                "data": None, "data_id_kind": None,
                "seed": None, "seed_source": "absent",
                "budget_source": None,
                "software": {"producer": None, "code_version": None,
                             "registry_version": REGISTRY_VERSION},
                "unavailable_reason": "not_provided", "notes": []}
    if not isinstance(v, Mapping):
        raise ValueError("observation.execution 必须是 execution_identity() 的返回值")
    for k in ("run_id_status", "seed_source", "run_id_basis", "seed",
              "software", "data", "data_id_kind"):
        if k not in v:
            raise ValueError(f"execution 缺字段 {k!r}；请用 execution_identity() 构造")
    status = v["run_id_status"]
    if status not in IDENTITY_STATES:
        raise ValueError(f"execution.run_id_status={status!r} 不在 {IDENTITY_STATES}")
    if list(v.get("run_id_basis") or []) != list(RUN_ID_BASIS):
        raise ValueError(
            f"候选 {candidate_id}：run_id_basis={v.get('run_id_basis')!r} 与 "
            f"{list(RUN_ID_BASIS)} 不一致——换了哈希口径却没换 id")

    rid = v.get("run_id")
    reason = v.get("unavailable_reason")
    seed = v.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError(f"候选 {candidate_id}：execution.seed 必须是整数，收到 {seed!r}")
    dat = _check_data(v.get("data"), candidate_id=candidate_id)
    if v.get("data_id_kind") != (None if dat is None else dat["id_kind"]):
        raise ValueError(
            f"候选 {candidate_id}：data_id_kind={v.get('data_id_kind')!r} 与 "
            f"实际数据 {None if dat is None else dat['id_kind']!r} 不符")
    kind = _seed_kind(v.get("seed_source"))
    soft = _check_software(v.get("software"), candidate_id=candidate_id)
    bud = None if budget_params is None else _check_budget_params(
        budget_params, candidate_id=candidate_id)
    if bud is None and status != "unconfirmed":
        raise ValueError(
            f"候选 {candidate_id}：给了 run_id 却没给 budget_params，"
            "无法复核这个号是怎么算出来的")

    # --- 状态与原因整块复算 ---
    exp_status, exp_reason = _expected_status(dat, seed, kind, bud, soft)
    if exp_status != status:
        raise ValueError(
            f"候选 {candidate_id}：run_id_status={status!r} 与要素不符（应为 "
            f"{exp_status!r}）。有数据/seed/预算却标 unconfirmed，或反之，"
            "都会让'能不能扣掉一条重复'判错")
    if exp_reason != reason:
        raise ValueError(
            f"候选 {candidate_id}：unavailable_reason={reason!r} 与真正缺的要素不符"
            f"（应为 {exp_reason!r}）")

    want = {"recorded": "R-", "derived": "D-"}.get(status)
    if want is None:
        if rid is not None:
            raise ValueError(
                f"候选 {candidate_id}：run_id_status=unconfirmed 时不许有 run_id（收到 {rid!r}）")
    else:
        if not isinstance(rid, str) or not rid.startswith(want) or len(rid) != len(want) + 10:
            raise ValueError(
                f"候选 {candidate_id}：run_id={rid!r} 与状态 {status!r} 不符（应为 {want}+10 位）")
        if v.get("budget_source") not in (None, bud["source"]):
            raise ValueError(
                f"候选 {candidate_id}：execution.budget_source={v.get('budget_source')!r} "
                f"与观测预算来源 {bud['source']!r} 不一致")
        expect = _run_id_from(candidate_id, dat, seed, kind, bud, soft, status)
        if rid != expect:
            raise ValueError(
                f"候选 {candidate_id}：run_id={rid!r} 与观测预算重算结果 {expect!r} 不符。"
                "执行身份与观测不是同一组条件——这正是「同档名不同数值」要挡住的情形")
    return dict(v)


def _conditions_recorded(e: Mapping) -> bool:
    """这条执行身份是否**整体**达到 `recorded`。判重时只认这一种。

    逐项而不是只看 `run_id_status`：状态是别人写进来的字符串，逐项判定才能
    在字段被改、状态没跟着改时发现。四项与 `_expected_status` 的 `recorded`
    门槛一一对应，改一处必须改另一处（`registry_digest` 覆盖这条口径）。
    """
    soft = e.get("software") or {}
    return (e.get("run_id_status") == "recorded"
            and str(e.get("seed_source", "")).split(":", 1)[0] == "artifact"
            and e.get("budget_source") == "recorded"
            and bool(soft.get("code_version"))
            and e.get("data_id_kind") == "content_sha256")


def _nonempty_str(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


_HEX_DIGITS = frozenset("0123456789abcdef")


def _valid_sha256(v) -> bool:
    """是否是**格式合法**的内容哈希：64 位十六进制，大小写不敏感。

    存在的理由不是洁癖：`_nonempty_str` 只能回答"填了没有"，回答不了"填的是不是哈希"。
    `"same"` 显然是两个字的字符串，两条 `{"path": "x", "sha256": "same"}` 会因此
    被判成同一产物——0.5.0 把"缺失值相等"当证据，0.6.0 又留下"无效值相等"当证据，
    同一个毛病换了一件外衣。字段存在 ≠ 字段内容有效。
    """
    v = v.strip().lower() if isinstance(v, str) else v
    return (isinstance(v, str) and len(v) == SHA256_HEX_LEN
            and all(c in _HEX_DIGITS for c in v))


def _artifact_key(src: Mapping) -> tuple | None:
    """把 `source` 归一成「同一个产物」的可比键；定位字段不足时返回 `None`。

    四道门槛，都是"缺失值不能当证据"的直接对策：

    1. **`path` 与 `sha256` 都必须非空**。0.5.0 只查 `path is not None`，于是
       `{"path": "result.json", "sha256": None}` 两条会被判成同一产物——
       `same_artifact` 因此能把两条"什么都不知道"的记录合并掉，
       与"确认重复必须靠来源证据"的原则直接冲突。
    2. **`sha256` 必须是 64 位十六进制**（0.7.0）。非空但格式无效不许退化成 `None`
       ——那会把调用方的错误静默降级成"证据不足"，看似安全实则把 bug 藏起来；
       也不许直接用来比相等——`"same"` 相等不证明任何事。所以**报错**，
       这是"调用方提供了违反字段定义的数据"，不是"来源信息缺失"。
    3. **`row` 决定层级**：有 `row` ⇒ 行级键 `("row", path, sha256, row, section)`；
       无 `row` ⇒ 文件级键 `("file", path, sha256)`。键前缀不同，
       所以"一条文件级、一条行级"永不相等——它们本来就不是同一个定位。
    4. **`section` 不能脱离 `row`**：section 是文件内的行内分段定位，缺行号就无从
       定位；静默忽略它等于把同一 `path` 下两个不同分段的记录当成同一行。
    """
    if not isinstance(src, Mapping):
        raise ValueError(f"source 必须是 Mapping，收到 {type(src).__name__}")
    path, sha = src.get("path"), src.get("sha256")
    blank_sha = sha is None or (isinstance(sha, str) and not sha.strip())
    if not _nonempty_str(path) or blank_sha:
        return None
    if not _valid_sha256(sha):
        raise ValueError(
            f"source.sha256={sha!r} 不是 {SHA256_HEX_LEN} 位十六进制："
            "字段非空不等于内容有效。拿随手写的字符串（\"same\"、文件名、截断哈希）"
            "当内容哈希，两条不同产物就会因为「这个字符串相等」被判成同一份；"
            "确认重复必须靠真哈希，缺就留空（走 possible_duplicate），别编")
    sha = sha.strip().lower()
    row, sec = src.get("row"), src.get("section")
    if row is None:
        if sec is not None:
            raise ValueError(
                f"source 给了 section={sec!r} 却没有 row：section 是文件内的行内定位，"
                "缺行号时它不能充当定位证据（要么补 row，要么文件级别就不记 section）")
        return ("file", path.strip(), sha)
    if isinstance(row, bool) or not isinstance(row, int):
        raise ValueError(f"source.row 必须是整数，收到 {row!r}")
    return ("row", path.strip(), sha, row, sec)


def _artifact_gap(sa: Mapping, sb: Mapping, ka, kb) -> str | None:
    """两条记录**没**被判成同一产物的具体原因（判成了就返回 `None`）。

    写事实，不写结论：调用方拿它填 `duplicate_evidence`，避免"artifact_equal=False"
    这种只说结果不说理由的证据——上一轮的错误正是靠这种含糊才活下来。
    """
    if ka is not None and ka == kb:
        return None
    if ka is None or kb is None:
        miss = [f"{t}.{f}" for t, s in (("a", sa), ("b", sb))
                for f in ("path", "sha256") if not _nonempty_str(s.get(f))]
        return "定位字段不足（缺 " + "、".join(miss) + "）⇒ 不能据此确认同一产物"
    if ka[0] != kb[0]:
        return (f"定位层级不同（{ka[0]} vs {kb[0]}）：文件级定位与行级定位"
                "不是同一个产物")
    if ka[1] != kb[1]:
        return "path 不同"
    if ka[2] != kb[2]:
        return "path 相同但 sha256 不同：同名不同内容，不是同一份产物"
    return "同一文件的不同行/分段（row 或 section 不同）⇒ 是两条记录，不是重复导出"


def duplicate_verdict(a: Mapping, b: Mapping, *, summary_equal: bool) -> dict:
    """两条记录是否算**两次独立观测**。只看来源，不看指标摘要。

    0.3.0 的错：用"摘要哈希相同"直接推出 `independent=false`。摘要相同只能
    证明这几个字段相同，不能证明它们来自同一次（甚至同一个）运行——同种子重跑
    会逐位相同，不同种子也可能在四舍五入后相同。

    因此本函数的输入是**来源证据**，`summary_equal` 只是其中一个输入事实：

    | 情形 | 判定 | 独立计数 |
    |---|---|---|
    | 候选定义不同 | `distinct` | 算 |
    | 同一产物（`path` 与 `sha256` **均非空**、`sha256` 是合法 64 位十六进制且相同，`row`/`section` 同层相同） | `confirmed_duplicate` / `same_artifact` | 不算 |
    | 同 `run_id` + 摘要相同 + 两侧执行身份均 `recorded` | `confirmed_duplicate` / `rerun_same_conditions` | 不算 |
    | 同 `run_id` + 摘要相同 + 执行身份未达 `recorded` | `possible_duplicate` | **算**（只标记） |
    | 同 `run_id` + 摘要不同 | `distinct` + `contradiction` | 算（这是可复现性问题） |
    | `run_id` 不同/缺失 + 摘要相同 | `possible_duplicate` | **算**（只标记） |

    产物定位的"缺字段"一律不算证据（0.5.0 的错）：`path` 相同但 `sha256` 两边都
    没有、或一边没有、或 `sha256` 不同，都不能确认同一产物；文件级定位与行级定位
    互不相认。理由逐条写进 `evidence["artifact_gap"]`。

    "字段非空但内容无效"更不算证据（0.7.0 的错）：`sha256="same"` 两条不判重复，
    而是**直接报错**——那是调用方给了违反字段定义的数据，不是来源缺失。

    最后一行是关键：来源不同的两份摘要相同的记录，可能是两个不同实验的巧合，
    也可能是同一次运行的两次导出而产物没记来源。扣掉是**猜**，不扣也是猜——
    所以不扣，但标记出来，并在 coverage 里给"严格口径"的对照数。

    入参是**身份视图**，不是整条记录：`{candidate_id, execution, source}`。
    """
    for tag, rec in (("a", a), ("b", b)):
        for k in ("candidate_id", "execution", "source"):
            if k not in rec:
                raise ValueError(f"duplicate_verdict 的 {tag} 缺字段 {k!r}")
    ea, eb = a["execution"], b["execution"]
    sa, sb = a["source"], b["source"]

    ra, rb = ea.get("run_id"), eb.get("run_id")
    same_cand = a["candidate_id"] == b["candidate_id"]
    same_run = ra is not None and ra == rb
    # 产物定位要比 row/section：pilot 的 `nested` 与 `sensitivity` 两段来自**同一行**，
    # 只比 (path, sha256, row) 会把两份不同模型的记录当成"同一产物"。
    # 但"两侧字段都缺失"**不是**匹配证据：0.5.0 用 `path is not None` 当门槛，
    # 于是 `{path: "result.json", sha256: None}` 两条会被判成同一产物（见 §10）。
    ka, kb = _artifact_key(sa), _artifact_key(sb)
    same_art = ka is not None and ka == kb
    # "同条件被产物记录"必须按 `recorded` 的定义整体判，而不是挑两项自查：
    # seed 来自产物 + 预算 `recorded` + **代码版本已知** + **数据有内容哈希**。
    # 0.4.0 只查了前两项，于是本批 239 条全是 `D-` 的情况下仍把两对判成
    # confirmed——同 seed 同预算同摘要仍可能出自不同版本的拟合代码（审计 D1）。
    observed = all(_conditions_recorded(e) for e in (ea, eb))

    ev = {
        "candidate_id_equal": same_cand,
        "run_id": [ra, rb],
        "run_id_equal": same_run,
        "run_id_status": [ea.get("run_id_status"), eb.get("run_id_status")],
        "seed": [ea.get("seed"), eb.get("seed")],
        "seed_source": [ea.get("seed_source"), eb.get("seed_source")],
        "budget_source": [ea.get("budget_source"), eb.get("budget_source")],
        "code_version": [(ea.get("software") or {}).get("code_version"),
                         (eb.get("software") or {}).get("code_version")],
        "data_id_kind": [ea.get("data_id_kind"), eb.get("data_id_kind")],
        "conditions_observed": bool(observed),
        "artifact_equal": bool(same_art),
        "artifact_scope": [None if ka is None else ka[0],
                           None if kb is None else kb[0]],
        "artifact_key": [None if ka is None else list(ka),
                         None if kb is None else list(kb)],
        "artifact_gap": _artifact_gap(sa, sb, ka, kb),
        "summary_equal": bool(summary_equal),
        "paths": [sa.get("path"), sb.get("path")],
    }

    if not same_cand:
        return _verdict("distinct", None, ev,
                        "候选定义不同：摘要即使相同也不是同一实验")
    if same_art:
        where = "整份文件" if ka[0] == "file" else f"第 {ka[3]} 行"
        return _verdict("confirmed_duplicate", "same_artifact", ev,
                        f"同一产物被登记了两次（path+sha256 均非空且相同，{where}）")
    if same_run:
        if not summary_equal:
            ev["contradiction"] = True
            return _verdict("distinct", None, ev,
                            "执行条件相同但结果不同：这不是重复，是可复现性问题，"
                            "不能靠合并记录把它消掉")
        if observed:
            return _verdict("confirmed_duplicate", "rerun_same_conditions", ev,
                            "同候选、同 seed、同预算数值、同代码版本、同数据内容，"
                            "四项都由产物记录；指标逐位相同 ⇒ 同条件的重复执行，无新信息")
        return _verdict("possible_duplicate", None, ev,
                        "执行条件指纹相同且指标逐位相同，但执行身份未达 recorded"
                        "（seed/预算/代码版本/数据内容至少一项不是产物记录）⇒ 不足以扣除")
    if summary_equal:
        return _verdict("possible_duplicate", None, ev,
                        "指标摘要相同，但执行身份不同或不可判；来源不同即可能巧合")
    return _verdict("distinct", None, ev, "来源与摘要均不同")


def _verdict(status: str, kind: str | None, ev: dict, reason: str) -> dict:
    if status not in DUPLICATE_STATES:
        raise ValueError(
            f"duplicate_status={status!r} 不在 DUPLICATE_STATES={DUPLICATE_STATES}；"
            "三态必须区分「没证据是重复」「可能是重复」「确认重复」")
    if kind is not None and kind not in DUPLICATE_KINDS:
        raise ValueError(f"duplicate_kind={kind!r} 不在 {DUPLICATE_KINDS}")
    if status == "confirmed_duplicate" and kind is None:
        raise ValueError("confirmed_duplicate 必须给 duplicate_kind（凭什么确认的）")
    return {"duplicate_status": status,
            "duplicate_kind": kind,
            "independent": status != "confirmed_duplicate",
            "reason": reason,
            "evidence": ev}


def sampling_status_from(*, run_status: str, rhat_max: float | None,
                         ess_min: float | None = None,
                         rhat_tol: float = 1.05, ess_min_tol: float = 100.0) -> str:
    """把 R̂/ESS 折算成 `sampling_status`。**只做折算，不做判定**。

    阈值来自生产口径（`bm_real` 的 R̂ ≤1.05 当门）。这里不判断模型对不对，
    只回答"后验诊断过没过"——所以本函数永远不出 `excluded` 之类的结构结论。
    """
    if run_status != "completed" or rhat_max is None:
        return "not_evaluated"
    bad = rhat_max > rhat_tol or (ess_min is not None and ess_min < ess_min_tol)
    return "failed" if bad else "passed"


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

    **永远不返回 `excluded`。** 单次快速拟合失败可能是初始化问题（钝化 P1：
    默认初值塌陷成 `n1=n_b=0.661` 的两个相同 CPE、R̂ 197.8；换成**真值初值**
    能到 6.65%，但那是 oracle，不是生产可达的解法），把它当结构排除会把整库的判定带偏。
    只有 L0 的确定性判据能出 `excluded`。
    """
    if not budget_sufficient:
        return "not_evaluated_budget"
    if not init_explored:
        return "defer"
    return "promote" if converged else "defer"


#: 预检会读取的候选配置键。预检结论依赖它们，所以**必须来自 Candidate 本身**，
#: 不能由调用方另传——否则同一个 voigt 在 include_L=True/False 下会被同一个名字
#: 给出看起来很确定的结论。
PRECHECK_CONFIG_KEYS = ("include_L", "adsorption")


def l0_precheck(cand, f, Z) -> dict:
    """对一个**具体候选**跑确定性 L0 预检，返回可入库的记录。

    接口是 `l0_precheck(candidate, f, Z)`，不再是
    `l0_precheck(topology_id, f, Z, **kw)`（0.1.0 的写法已废）：同一拓扑的不同
    配置可能得出不同结论，把配置做成可选关键字，调用方一旦忘记传，就会拿到一个
    "看起来确定"的错误答案。现在配置只能来自 `cand.config_dict`。

    `bm_select.voigt_can_represent` 的 `include_L=None` / `adsorption` 未传时
    会回落到全局 `VOIGT_CAPS`，那反映的是**生产默认**而不是这个候选。所以这里
    总是显式传值，并把实际传入的配置会话写进 `precheck_config`。

    返回：
        {candidate_id, precheck_state, precheck_config, precheck_reason,
         precheck_version}
    `precheck_state` 严格落在 `PRECHECK_STATES`；未登记预检的拓扑一律 `unknown`，
    "没有判据"不等于"通过"。
    """
    if not isinstance(cand, Candidate):
        raise ValueError(
            "l0_precheck 只接受 Candidate；按拓扑名+关键字调用会让预检脱离候选配置"
            "（旧签名 l0_precheck(topology_id, f, Z, **kw) 已废弃）")
    topo = get_topology(cand.topology_id)
    cfg = cand.config_dict
    used = {k: cfg.get(k) for k in PRECHECK_CONFIG_KEYS if k in cfg}

    if topo.precheck_ref is None:
        return {
            "candidate_id": cand.candidate_id,
            "precheck_state": "unknown",
            "precheck_config": used,
            "precheck_reason": "该拓扑未登记确定性预检",
            "precheck_version": PRECHECK_VERSION,
        }

    fn = resolve_forward(topo.precheck_ref)
    kwargs = {}
    if topo.id == "voigt":
        # 显式传，避免回落到全局 VOIGT_CAPS
        kwargs["include_L"] = bool(cfg.get("include_L", False))
        kwargs["adsorption"] = cfg.get("adsorption")
        used = {"include_L": kwargs["include_L"],
                "adsorption": kwargs["adsorption"]}
    ok, reason = fn(f, Z, **kwargs)
    if ok is True:
        state = "not_excluded"
    elif ok is False:
        state = "excluded"
    else:
        raise ValueError(f"预检 {topo.precheck_ref} 返回值必须是 bool，收到 {ok!r}")

    return {
        "candidate_id": cand.candidate_id,
        "precheck_state": state,
        "precheck_config": used,
        "precheck_reason": "" if ok is True else str(reason),
        "precheck_version": PRECHECK_VERSION,
    }


def precheck_candidates(candidates, f, Z) -> dict:
    """对一组候选逐条预检，直接产出可喂给 `screening_block` 的汇总 + 记录。"""
    records = [l0_precheck(c, f, Z) for c in candidates]
    tally = {s: 0 for s in PRECHECK_STATES}
    for r in records:
        tally[r["precheck_state"]] += 1
    return {"records": records, "registered": len(records), **tally}


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
    unconfirmed = [c.candidate_id for c in cands if not c.comparable]
    if unconfirmed:
        raise ValueError(
            f"候选集里有身份不可复现的候选：{unconfirmed}。"
            "`identity_status=unconfirmed`（或 init_profile=unknown）只能作文档性记录，"
            "不能进冻结候选集——身份错了，后面的'谁更好'全部无意义")
    ids = [c.candidate_id for c in cands]
    if len(set(ids)) != len(ids):
        dup = [i for i in set(ids) if ids.count(i) > 1]
        raise ValueError(f"候选集里有重复 candidate_id：{dup}")

    out = {
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
    out["manifest_digest"] = manifest_digest(out)
    return out


MANIFEST_DIGEST_FIELD = "manifest_digest"


def manifest_digest(manifest: Mapping) -> str:
    """对 manifest 除 `manifest_digest` 外的全部内容做指纹。

    `immutable=True` 只是个字段，它自己不阻止任何修改。真正能发现"冻结后被改"
    的是这个指纹：手改任一候选定义、预算档或 registry 指纹，校验都会失败。
    """
    payload = {k: _jsonable(v) for k, v in manifest.items()
               if k != MANIFEST_DIGEST_FIELD}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def verify_manifest(manifest: Mapping, candidates) -> None:
    """校验一份已冻结的 manifest 与当前候选集/注册表仍然一致。

    `assert_no_append` 只看"有没有新增"，会漏掉三类更隐蔽的改动：删候选、
    改写已有候选定义、直接篡改 manifest 内容。这里一并挡住：

    1. `manifest_digest` 与内容自洽（防手改）；
    2. `registry_digest` 与当前注册表一致（防结构约束在事后变化）；
    3. 候选 ID 集合**相等**（多出=追加，缺少=删除）；
    4. 每个 `candidate_definitions` 与当前 `describe()` 逐字段相同（防同名换定义）。
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest 必须是映射")
    if manifest.get("created_before_fit") is not True:
        raise ValueError("manifest.created_before_fit 不是 True，这份候选集不是拟合前写入的")

    stored = manifest.get(MANIFEST_DIGEST_FIELD)
    if not stored:
        raise ValueError(
            f"manifest 缺 {MANIFEST_DIGEST_FIELD}，无法判断冻结后是否被改动；"
            "请用 candidate_set_manifest() 重新生成")
    if stored != manifest_digest(manifest):
        raise ValueError(
            f"manifest 内容与其指纹不符（记录 {stored}，实算 {manifest_digest(manifest)}）："
            "冻结后的候选集被修改过")

    reg_now = registry_digest()
    if manifest.get("registry_digest") != reg_now:
        raise ValueError(
            f"registry_digest 变化：manifest 记 {manifest.get('registry_digest')}，"
            f"当前为 {reg_now}。拓扑、元件、结构校验规则或观测 schema 的任何改动都会"
            "影响历史候选集的可比性；请建立新的 candidate_set_id")

    frozen_ids = list(manifest.get("candidate_ids", ()))
    incoming = [c.candidate_id for c in candidates]
    extra = sorted(set(incoming) - set(frozen_ids))
    if extra:
        raise ValueError(
            f"候选集 {manifest.get('candidate_set_id')} 已冻结；新增 {extra} "
            "需要建立新的 candidate_set_id（这是一次新分析，不是原集合的延伸）")
    gone = sorted(set(frozen_ids) - set(incoming))
    if gone:
        raise ValueError(
            f"候选集 {manifest.get('candidate_set_id')} 已冻结；缺少 {gone}。"
            "删候选同样是改变候选集，需要新的 candidate_set_id")

    frozen_defs = {d["candidate_id"]: d for d in manifest.get("candidate_definitions", ())}
    for c in candidates:
        old = frozen_defs.get(c.candidate_id)
        if old is None:
            raise ValueError(f"manifest 缺少 {c.candidate_id} 的定义")
        now = _jsonable(c.describe())
        if _jsonable(old) != now:
            raise ValueError(
                f"候选 {c.candidate_id} 的定义与冻结时不同（同名换定义）；"
                "candidate_id 必须覆盖全部字段，定义变了 id 就该变")


def assert_no_append(manifest: Mapping, candidates) -> None:
    """拒绝往已冻结的候选集里追加新候选（只查追加；完整校验用 `verify_manifest`）。"""
    frozen = set(manifest.get("candidate_ids", ()))
    incoming = {c.candidate_id for c in candidates}
    extra = incoming - frozen
    if extra:
        raise ValueError(
            f"候选集 {manifest.get('candidate_set_id')} 已冻结；新增 {sorted(extra)} "
            "需要建立新的 candidate_set_id（这是一次新分析，不是原集合的延伸）")


def registry_digest() -> str:
    """注册表内容指纹。manifest 记下它，注册表被改动后可被发现。

    **结构校验规则也进指纹**（`consistency_id`）：只哈希字段的话，改
    `_nested_consistency` 的判断逻辑而字段不动时，指纹不会变，manifest 就无法
    发现"数学结构约束变了"。这里不对函数对象做 repr 哈希（不稳定），而是要求
    拓扑给一个显式、可读的版本标识。
    """
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
                "consistency_id": v.consistency_id,
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
        "precheck_config_keys": list(PRECHECK_CONFIG_KEYS),
        "observation_schema": {
            "exec_states": list(EXEC_STATES),
            "sampling_states": list(SAMPLING_STATES),
            "interpretations": list(INTERPRETATIONS),
            "null_reasons": list(NULL_REASONS),
            "metrics": list(OBSERVATION_METRICS),
            # 0.3.0：身份与四项描述块的枚举也进指纹。改了"什么算可复现身份"、
            # 或加了新的缺失语义，manifest 必须失效——否则旧候选集会悄悄继承新语义。
            "identity_states": list(IDENTITY_STATES),
            "init_unknown": INIT_UNKNOWN,
            "noise_floor_sources": list(NOISE_FLOOR_SOURCES),
            "budget_param_keys": list(BUDGET_PARAM_KEYS),
            "budget_param_sources": list(BUDGET_PARAM_SOURCES),
            # 0.4.0：执行身份与重复判定的口径也进指纹。"什么算同一次执行"、
            # "什么算两次独立观测"改了而指纹不变，旧记录会被新规则静默重判。
            "run_id_scope": RUN_ID_SCOPE,
            "run_id_basis": list(RUN_ID_BASIS),
            "run_id_missing": list(RUN_ID_MISSING),
            "seed_source_kinds": list(SEED_SOURCE_KINDS),
            # 0.5.0：数据身份等级 + 判重门槛（recorded 才确认）也进指纹。
            "data_id_kinds": list(DATA_ID_KINDS),
            "confirm_duplicate_requires_recorded": True,
            "duplicate_states": list(DUPLICATE_STATES),
            "duplicate_kinds": list(DUPLICATE_KINDS),
            # 0.6.0：产物定位的门槛（缺字段不算证据、文件级/行级不互认）也进指纹。
            "artifact_scopes": list(ARTIFACT_SCOPES),
            "artifact_key_fields": list(ARTIFACT_KEY_FIELDS),
            # 0.7.0：**字段内容有效性**也进指纹。"非空即有效"与"必须 64 位十六进制"
            # 会让同一对记录得到不同判定，改了口径而不换指纹，旧记录会被静默重判。
            "confirm_same_artifact_requires_valid_sha256": True,
            "artifact_sha256_hex_len": SHA256_HEX_LEN,
        },
        "registry_version": REGISTRY_VERSION,
        "precheck_version": PRECHECK_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------
# 已登记内容
# --------------------------------------------------------------------------
def _voigt_consistency(by_slot: Mapping, cfg: Mapping) -> None:
    """voigt 的结构约束：K 必须等于实际挂上的 HN 单元数。

    没有这条，`config={"K": 3}` 配一个 HN 单元也能通过槽位校验——登记表里写的
    数学结构和实际要拟合的结构就对不上了，而这正是"名称统一了、模型却没法公平比较"
    的来源（2026-11-13 二轮评审 §4）。
    """
    units = by_slot.get("relaxation_unit", [])
    if not units:
        raise ValueError("voigt：至少要有一个 relaxation_unit（HN 单元）")
    if any(u != "hn" for u in units):
        raise ValueError(f"voigt：relaxation_unit 只接受 hn，实得 {units}")

    if "K" not in cfg:
        raise ValueError(
            f"voigt：config 必须显式给 K（实得 {len(units)} 个 HN 单元），"
            "不允许由槽位数量隐含推出——manifest 里要能一眼核对")
    try:
        K = int(cfg["K"])
    except (TypeError, ValueError):
        raise ValueError(f"voigt：config['K']={cfg['K']!r} 不是整数")
    if K != len(units):
        raise ValueError(
            f"voigt：config['K']={K} 与实际挂载的 HN 单元数 {len(units)} 不一致；"
            "数学结构与登记定义对不上")

    inc = bool(cfg.get("include_L", False))
    inductors = by_slot.get("series_inductor", [])
    if inc and len(inductors) != 1:
        raise ValueError(
            f"voigt：include_L=True 需要恰好一个 series_inductor，实得 {inductors}")
    if not inc and inductors:
        raise ValueError(
            f"voigt：include_L=False 但挂了 series_inductor={inductors}；"
            "顶层感抗只在 include_L=True 时采样")


def _nested_consistency(by_slot: Mapping, cfg: Mapping) -> None:
    """nested 的结构约束（写在这里而不是注释里，错误当场抛）。

    `layers` 必须**严格**属于 {1,2}：`z_nested` 的签名虽然写着 1..3，
    但 3 层从未实现（§6.3 已订正）。旧实现只分支 1 和 2，传 3 会静默通过，
    于是"登记了 3 层"这种不存在的结构能一路走进 manifest。
    """
    if "layers" not in cfg:
        raise ValueError("nested：config 必须显式给 layers（1 或 2）")
    raw_layers = cfg["layers"]
    if isinstance(raw_layers, bool) or not isinstance(raw_layers, int):
        # 允许 "2" 这种字符串，但必须在整数集里；bool 单独挡掉（True == 1）
        try:
            layers = int(raw_layers)
        except (TypeError, ValueError):
            raise ValueError(f"nested.config.layers={raw_layers!r} 不是整数")
        if str(layers) != str(raw_layers).strip():
            raise ValueError(f"nested.config.layers={raw_layers!r} 不是整数")
    else:
        layers = raw_layers
    if layers not in (1, 2):
        raise ValueError(
            f"nested.config.layers={layers} 不支持；只有 1、2 两层实现过"
            "（z_nested 文档的 1..3 是笔误，§6.3 已订正，3 层从未实现）")

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
        "uniform": "init_to_uniform（生产默认；`bm_core.fit` 未给 init_values 时）",
        "data_driven": "由 f/|Z| 统计量算出（pilot 的 `voigt_init`：min|Z|、"
                       "τ 窗内均铺、φ=0.8），不读真值",
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
    consistency=_voigt_consistency,
    consistency_id="voigt_consistency_v1",
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
    consistency_id="nested_consistency_v2",
    notes=(
        "§13 留出 0/3、§14 全量 4/14，卡 M-b acf≤0.50；不接 bmdrt.py 主流程",
        "骨架与 测试数据/涂层、钝化 对到 1e-15；这两族只是同构，不构成通用证据",
        "未登记确定性预检：unknown 不等于 not_excluded",
        # 登记合法 ≠ 准入通过：`circuit_registry` 只回答"这个候选能不能被表达与比较"，
        # 不回答"该不该用它"。容性支路准入状态仍是 CIRCUIT_TOPOLOGY_ADMISSION.md
        # §4.1.3/§4.1.4「参数层未验收」，生产中默认关。
        "branch='capacitive' 是研究接口（admission §4.1.3/§4.1.4 参数层未验收；"
        "nested_fit.py 默认 branch='parallel'；不在 bm_select 候选池）——"
        "config 层可表达，不等于准入通过",
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
