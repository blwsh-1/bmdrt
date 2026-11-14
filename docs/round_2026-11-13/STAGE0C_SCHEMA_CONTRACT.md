# 阶段 0C：schema 契约验收

状态：**schema 层通过，判定逻辑未实现**。本轮不跑拟合、不接生产、不改任何门禁。

> 0.4.0 修订（2026-11-14 第三轮审计后）：新增**执行身份**（`run_id`）与**重复判定三态**
> （`duplicate_verdict`）。`candidate_id` 定义**未动**。用例 57 → **66**。见 §8。
> 同时推翻 0.3.0 的一处判定：pilot 的 `voigt_prod`/`voigt_fix2` 是**默认初始化**，
> 0.3.0 标成 `data_driven` 是读函数 docstring 读错了（§8.3）。
>
> 0.3.0 修订（2026-11-14 第二轮审计后）：候选**身份等级**、六项描述块
> （`param_estimate` / `ratio` / `noise_floor` / `predictive` / `budget_params`、
> 沿用 `param_stability`）。用例 45 → **57**。修订清单见 §7。
>
> 0.2.0 修订（第一轮审计后）：执行状态与采样状态拆分、补数学一致性校验、
> L0 预检绑定候选、manifest 可校验（见 §6）。原始 29 条用例中有两处语义被推翻
> （E3 与 I 段签名），**不是新增测试掩盖旧失败**。

产物：
- `circuit_registry.py`（注册表 + 三个输出块 + 执行身份 + 重复判定 + 冻结契约，导入 0.05s，不加载 jax/numpyro）
- `test_circuit_registry.py`（66 条用例，实跑 **66/66 通过**）
- `docs/round_2026-11-13/STAGE0A_SCHEMA_PRESSURE.md`（三谱压测，只读既有产物）
- `docs/round_2026-11-13/STAGE0_RECORDS_MAP.md`（239 条历史记录映射，v3）

## 1. 六类验收情形 → 用例对照

| 验收情形（本轮要求） | 用例 | 结论 |
|---|---|---|
| 结构排除与预检未知 | `I1` / `I3` / `I2` / `I4` | 低频感抗谱 → voigt `excluded`；nested 未登记预检 → `unknown`；容性谱 → `not_excluded`（预检覆盖不到，已知的洞）；预检记录含 `candidate_id` 与实际配置，旧签名被拒 |
| 拟合未运行 / 失败 / 采样失败 | `D1` / `E4` / `E1` | 执行层三者闭合；`run_status≠completed` 不报残差；**跑完但采样失败时残差保留且强制 `descriptive_only`** |
| 拟合贴底但参数不稳 | `E8` | `rms_rel 7.076%`（`rel_median 2.403%`）与 `n2 flag=at_bound` 同时保留，不合成单一分数 |
| 多候选可解释但不可区分 | `F2` / `F3` | `non_identifiable` 要求 `n_viable>=2`；`identified` 无样本外证据被拒 |
| 合成谱真值下全候选不足 | `F4` | `model_inadequate` 需 `truth_known` 或 `noise_calibrated` |
| 真实谱证据不足、不得误报 | `F1` / `F4` | 真实谱单候选 → `insufficient_evidence`；无真值时 `model_inadequate` 直接报错 |
| **候选身份不可复现** | `J1`–`J5` | `init_profile=unknown` → `U-` 前缀 + `unconfirmed` + **禁止进 manifest** |
| **六项描述块语义独立** | `K1`–`K7` | 档名≠数值、分母必填、真值列只许合成谱、ELPD 可用性必须显式 |
| **执行身份与模型身份分层** | `L1`–`L5` / `L9` | 同档名不同预算数值 ⇒ 同 `candidate_id`、不同 `run_id`；缺 seed/预算/数据不发号；`run_id` 由观测预算重算复核 |
| **重复判定的证据门槛** | `L6`–`L8` | 摘要相同 ≠ 重复；证据不足只标 `possible_duplicate` 且**不扣**独立数 |

## 2. schema 已强制的硬约束（可执行，不是注释）

1. `screening` 闭合：`registered = excluded + not_excluded + unknown`。
2. `only_survivor` **只由** `survivors == 1` 计算，不接受调用方传入；`status="only_survivor"` 直接报错。
3. `evidence` 五态互斥，`identified` 需 `out_of_sample_valid=True` 且 `n_viable>=2`。
4. `model_inadequate` 需 `truth_known` 或 `noise_calibrated`。
5. **执行 / 采样 / 观测分开**：`run_status ∈ {completed,failed,not_run}`、
   `sampling_status ∈ {passed,failed,not_evaluated}`、`fit_observation` 记已算出的残差。
   `run_status≠completed` 不得报任何残差；`sampling_status≠passed` 而报了残差，
   `interpretation` 必须显式为 `descriptive_only`（不设默认值）。
6. 指标为 `None` 必须在 `unavailable` 里给 `NULL_REASONS`（`not_applicable` /
   `not_computed` / `failed`）之一；给了值的指标不许同时出现在 `unavailable` 里。
   已 `completed` 的候选，`rms_rel`/`acf` 既无值也无原因会直接报错。
7. `param_stability` 必须是"参数名 → 描述块"，不许传标量分数。
8. `L1` 返回集合 `{promote, defer, not_evaluated_budget}`，**永不含 `excluded`**。
9. `wired=False` 的元件必须写 `blocked_reason`，且不能被候选引用。
10. `candidate_id` 覆盖 `topology + elements + config + init + budget + likelihood` 六项。
11. **数学结构一致性**：`nested.layers` 严格 ∈ {1,2}（3 层从未实现）；`voigt.K`
    必须等于实际 HN 单元数；`include_L` 与 `series_inductor` 必须同侧。
12. **L0 预检绑定候选**：`l0_precheck(candidate, f, Z)`，配置只能来自候选本身；
    返回值含 `candidate_id` / `precheck_state` / `precheck_config` / `precheck_reason` /
    `precheck_version`。汇总数必须能由逐条记录复算。
13. 候选集 manifest 的 `created_before_fit` 必须为 `True`；冻结后追加、删除候选、
    改写已有定义、篡改内容（`manifest_digest` 不符）、注册表漂移（`registry_digest` 不符）
    都会报错。`immutable=True` 只是字段，真正的拦截靠 `verify_manifest()`。
14. **身份等级**：`identity_status ∈ {recorded, derived, unconfirmed}`；
    `init_profile=unknown` 时 id 前缀强制 `U-`、等级强制 `unconfirmed`、`comparable=False`；
    `unconfirmed` 的候选**不能进** `candidate_set_manifest`。
    没登记的档名（含"近似档名"）一律报错，不允许拿相近档名顶替。
15. **`budget_params` 不是 `budget_profile`**：传档名字符串直接报错；
    `source=recorded` 要求五个核心参数齐全；`derived_from_tier` 允许部分并保留档名。
16. **`ratio` 必须带分母**：`n_used`/`n_total` 为整数且 `n_used ≤ n_total`；
    `n_usable`（后验诊断通过的条数）缺省等于 `n_used`，小于 `n_total` 时
    `denominator_scope` 自动为 `converged_subset`；`noise_floor_source` 必须与
    `noise_floor.source` 一致，且不能是 `not_available`。
17. **`param_estimate` 的真值列只许合成谱**：`truth_known=False` 时出现
    `truth` 或 `rel_err` 直接报错；`noise_floor.source` 非法、或 `not_available`
    却带 `value`，都报错；`predictive.valid_for_comparison` 必须是显式布尔，
    为 `False` 时必须给 `reason`。
18. **执行身份与模型身份分层**：`candidate_id` 只含模型定义与预算**档名**；
    `run_id` = 候选 + 数据 + seed + **实际预算数值** + 软件。`R-` 要求 seed 来自产物、
    预算 `recorded`、代码版本已知；`D-` 为推导；缺任一项**不发号**（`None` +
    `RUN_ID_MISSING` 原因）。候选为 `U-` 时不许发 `run_id`。
19. **`run_id` 可复核**：`observation_block` 用观测里的 `budget_params` 重算
    `run_id`，不符即报错；给了 `run_id` 却没给 `budget_params` 也报错。
20. **重复判定看来源，不看摘要**：`duplicate_verdict()` 的三态
    `distinct` / `possible_duplicate` / `confirmed_duplicate`；
    `independent` 是派生值，只有 `confirmed_duplicate` 才为 `False`；
    `confirmed_duplicate` 必须给 `kind`（`same_artifact` / `rerun_same_conditions`）；
    "同条件但结果不同"记为 `distinct` + `contradiction`（可复现性问题），不合并。

## 3. 注册内容

| 层 | 条目 |
|---|---|
| 拓扑（2） | `voigt`（串联 HN 弛豫，production）、`nested`（层状嵌套，exploratory） |
| 初始化档 | voigt：`uniform` / `data_driven`（pilot 的 `voigt_init`）/ `value`；nested：`uniform` / `data_driven` / `value`；另有 `unknown`（不可确认，强制 `U-`） |
| 结构校验版本 | `voigt_consistency_v1`、`nested_consistency_v2`（进 `registry_digest`） |
| 元件已接线（4） | `hn`、`cpe`、`rl_parallel`、`rl_series` |
| 元件未接线（4） | `w_inf`、`w_open`、`w_short`、`adsorption`——登记公式与参数，但现有拓扑**没有对应槽位**，`wired=False` |

未接线不是"没测"，是"没有接点"。AZ31 那次失败正是接点问题（`R∥L` 必须在第二并联节点内，
挂到 voigt 最外层串联只能把 noise=0 残差从 44.0% 降到 40.7%），所以这一档必须显式区分。

`registry_digest = 0f59a5686c2d0c5b`（本版，`0.4.0`）。manifest 会记录它，注册表被改可被发现；
`consistency_id`、观测 schema 常量（含身份枚举、缺失语义、预算参数名）、
**执行身份的哈希口径（`RUN_ID_BASIS`/`RUN_ID_SCOPE`/`RUN_ID_MISSING`）与重复判定三态**
都在指纹内，所以改校验规则、改"什么算可复现身份"、改"什么算两次独立观测"
都不会"指纹不变"。

## 4. 本轮未覆盖（留给阶段 1/2/3）

- **状态转移逻辑本身**：契约测试只证明"每种状态能被表达且乱填会报错"，不证明
  "从观测到 state 的推导正确"。后者属于 evaluator，且依赖 σ 口径。
- **runner 的调度一致性**：同一候选在不同 OS/进程下是否给出同一 `candidate_id`、
  同一 manifest digest，未测。
- **阈值**：`acf` 上限、参数区间宽度、`n_viable` 的具体判定，本轮一个都没定。
  在拿到一批描述性观测之前定阈值就是拍脑袋。
- **真实谱的 `model_inadequate`**：条件参数已就位（`noise_calibrated`），但没有标定噪声
  的实现，所以这一态在真实谱上仍然打不出来——这是有意为之。

## 5. 边界

- 不接 `bmdrt.py`；`voigt` 仍是生产默认族；`nested` 仍 `exploratory`。
- §13 `0/3`、§14 `4/14`、§17 `0/10` 原样，未因本轮改动重估。
- 三谱压测是 2 条合成谱 + 1 条实测谱，样本量不支持任何总体陈述。

## 6. 0.2.0 修订（针对 2026-11-14 只读审计的四项 P0）

| 编号 | 缺陷 | 修法 | 用例 |
|---|---|---|---|
| P0-1 | `observation_block(converged=False)` 禁止报 `rms_rel`/`acf`，导致 schema **装不下自己的压测数据**（涂层 parallel 10.81%/acf 0.944/R̂ 14.21 就是"跑完但采样没过"） | 拆 `run_status` / `sampling_status` / `fit_observation`；后验未过时残差必须保留且强制 `descriptive_only` | `E1`–`E10` |
| P0-2 | `layers=3` 静默通过；voigt 的 `config["K"]` 与实挂 HN 数不校验 | `layers` 严格 ∈{1,2}；`K == len(relaxation_unit)`；`include_L` 与 `series_inductor` 同侧 | `B7`–`B9` |
| P0-3 | `l0_precheck(topology_id, f, Z, **kw)` 可由调用方另传配置；screening 只存数量 | 只接受 `Candidate`，配置取自候选本身；记录含 `candidate_id`/`precheck_config`/`reason`/`version`；汇总必须能被逐条复算 | `C3` / `I4` |
| P0-4 | `registry_digest` 不含结构校验规则；`assert_no_append` 只查追加 | `consistency_id` 进指纹；新增 `manifest_digest` 与 `verify_manifest()`（查追加/删除/同名换定义/内容篡改/注册表漂移） | `A4` / `H4`–`H6` |

被推翻的旧语义，明确记录，不要"改测试让它过"：
- 旧 `E3`「`converged=False` 不得报 rms/acf」→ 只在 `run_status≠completed` 时成立。
- 旧 `I1/I2/I3` 用 `l0_precheck("voigt", f, Z)` → 新签名为 `l0_precheck(candidate, f, Z)`，
  旧写法直接报错（`test_circuit_registry.py` 的 `I4`）。

本轮仍未解决（留给下一轮，不进阶段 1）：
- 历史产物到 `observation_block` 的机器可读映射 → **已在 0.3.0 完成**（239 条，见 §7）；
- `budget_profile` 仍未成为结构化字段，改由 `budget_params` 承载实际数值；
- 232/239 条的预算参数是"由档名推出来的"，不是产物记录的（详见 `STAGE0_RECORDS_MAP.md` §5）。

## 7. 0.3.0 修订（针对第二轮审计：候选身份与字段承载）

### 7.1 三个身份缺陷（都是真错，不是措辞）

| 缺陷 | 事实 | 后果 | 修法 | 用例 |
|---|---|---|---|---|
| `_INIT_HINT` 按目录名推断**从未命中** | 归档文件平铺在 `stage0_three_spectra/`，`p.parent.name` 恒为该名 | `init_profile` 回落成真实档名 `"uniform"`，13 条记录拿到**看似合法、实际无依据**的 `candidate_id` | 查不到就 `INIT_UNKNOWN` → `U-` 前缀 + `unconfirmed` + 禁进 manifest | `J1`–`J4` |
| AZ31 的 2 条 **voigt 行被算成 nested** | voigt 行 `branch=None`，v1 写 `or "parallel"` | 与真实 nested parallel 行 `candidate_id` 完全撞车，6 行只剩 4 id；**voigt 残差记到了 nested 名下** | 按 `label` 分流 + 断言"同一 id 必须同一份六字段定义" | `J4`（映射脚本内置断言） |
| pilot 的 voigt 初始化标成 `uniform` | `voigt_init` 文档写"数据驱动初值（只看 f、\|Z\|）" | schema 连"用的是哪种初值"都写不下 | voigt 补 `data_driven` 档 | `J5` |

> **0.4.0 推翻第三行**：补 `data_driven` 档本身是对的（`J5` 保留），但**不应把它套到
> `voigt_prod`/`voigt_fix2` 上**——那两个调用点传的是 `init=None`，走 `init_to_uniform`。
> 详见 §8.3。

补充事实（曾两边说法不一致，已核清）：`fit_family_compare.py` **只写**
`family_fit_results.json`，**不写** `.nested.json`；`.nested.json` 的 `files.json`
记录了它自己写到哪，`input` 记录了 `f_min/f_max` 与含真值的文件名。
所以 `*_1113.nested.json` 是**同配置同 seed 的重跑**（elapsed 不同、指标逐位相同），
不是拷贝，也**不是** `data_driven`。

### 7.2 六项描述块入库

| 字段 | 新增的硬约束 | 用例 |
|---|---|---|
| `param_estimate` | 真值列只许合成谱；只记 `rel_err` 合法（产物常只存 \|fit/truth−1\|） | `K6` |
| `ratio` | `n_used`/`n_total` 必填；`n_usable` 与 `denominator_scope` | `K3` / `K4` |
| `noise_floor` | `source` 三态；`not_available` 不许带值 | `K5` |
| `predictive` | 可用性必须显式布尔；`False` 必须给原因 | `K7` |
| `budget_params` | 档名字符串直接报错；`recorded` 要求五项齐全 | `K1` / `K2` |
| `param_stability` | （0.2.0 已有）不接受标量分数 | `E7` |

### 7.3 与 0.2.0 的关系

0.2.0 的 13 条硬约束**一条未放宽**，只新增。三处旧语义的修正在 0.3.0 里继续成立
（`run_status`/`sampling_status` 拆分、`layers` 严格 ∈{1,2}、`l0_precheck` 只收 `Candidate`）。

### 7.4 仍未做的

- 四态判定的阈值：一个都没定。
- runner（阶段 1）：未写。
- `consistency_id` 仍是**人工维护**的版本号。改校验函数必须同步改 ID，代码无法自动识别
  逻辑变化——这是 `A4` 用例的设计前提，不是遗漏。

## 8. 0.4.0 修订（针对第三轮审计：独立观测计数与两层身份）

### 8.1 两个 P0/P1 都是真缺口

| 编号 | 审计意见 | 核实结论 | 修法 | 用例 |
|---|---|---|---|---|
| P0 | 去重指纹只含摘要字段，`independent=false` 缺来源证据 | **成立**：v2 的 `_fingerprint` 是 `candidate_id + fit_observation + param_estimate`，函数内既不读 seed 也不读产物路径 | `duplicate_verdict()` 按来源判三态；`summary_equal` 降级为一个输入事实 | `L6`–`L8` |
| P1 | `candidate_id` 不含实际预算数值 | **成立**：它哈希的是 `budget_profile` 档名 | **不改 `candidate_id`**，新增 `execution_identity()` → `run_id` | `L1`–`L5` / `L9` |

### 8.2 `run_id` 的定义（写清边界，避免它变成第二个 `candidate_id`）

| 问题 | 答案 |
|---|---|
| 它是什么 | **执行条件**指纹，不是墙钟实例号 |
| 哈希什么 | `candidate_id` + 数据（文件名 + `f_min/f_max`）+ seed + seed 的证据类别 + 实际预算（含 `source`）+ 软件 |
| 为什么含数据 | 同候选同 seed 同预算跑 60 条谱是 60 次执行；不写数据会共享一个号 |
| 为什么含 `source` | 数值相同、证据不同（产物记录 vs 档名推导）不能拿同一个号，否则推导出的条件被当成观测到的条件 |
| 同条件重跑 | **共享同一个 `run_id`**，实例由 `source.path` 区分。这正是它能当去重键的原因 |
| 什么时候不发 | 缺数据 / seed / 预算 ⇒ `None` + `unavailable_reason` |

`recorded`（`R-`）要求 seed 来自产物、预算 `recorded`、代码版本已知三者齐。
**本批 0 条**——没有一份历史产物记代码版本。这是待补字段，不是降门槛的理由。

### 8.3 推翻 0.3.0 的一处判定 + 查出的一处同类错误

| 项 | 0.3.0 的说法 | 事实 | 后果 |
|---|---|---|---|
| pilot `voigt_prod`/`voigt_fix2` 初值 | `data_driven` | `voigt_fit(f,Z,...)` 不传 `init` ⇒ `C.fit(init_values=None)` ⇒ `init_to_uniform` | 120 行标反。0.3.0 是**照函数 docstring 判身份**，正是它自己反对的做法 |
| pilot `sensitivity` 拓扑 | `nested / capacitive / L2` | `voigt_fit(f,Z,None,seed=0,init=voigt_init(...))` = **voigt K=2 / 自由 σ / 数据驱动** | 46 条的残差被记到另一个模型名下，且与 `nested` 段共享 `candidate_id` |

两处都由**调用点**判定，不是由函数名或文档判定。修正后
`uniform 132 / data_driven 106 / value 1`（原 `uniform 12 / data_driven 226 / value 1`）。

### 8.4 与 0.3.0 的关系

0.2.0/0.3.0 的 17 条硬约束**一条未放宽**，只新增 18–20 三条；`candidate_id` 的
六个字段与两处结构一致性校验**原样**。被推翻的两处（§8.3）是映射层的判定，
不是 schema 约束。

### 8.5 仍未做的

- runner（阶段 1）：未写。本轮只补身份与计数口径，**不是** runner 的前置条件已满足。
- 代码版本字段（`software.code_version`）：schema 有位置，历史产物没有值。
- 阈值：`acf`、区间宽度、`n_viable`，一个都没定。

