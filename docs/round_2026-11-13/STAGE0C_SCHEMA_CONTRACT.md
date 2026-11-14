# 阶段 0C：schema 契约验收

状态：**schema 层通过，判定逻辑未实现**。本轮不跑拟合、不接生产、不改任何门禁。

> 0.7.0 修订（2026-11-14 第六轮审计后）：**`sha256` 非空不等于有效**——`same_artifact`
> 要求 64 位十六进制，`"same"`/截断哈希/非十六进制字符直接报 `ValueError`，不再退化成
> `possible_duplicate`。空串按字段分层：`source.sha256` 算缺失、`execution.data.sha256`
> 算错误（§11.5）；格式校验只证外形、**不证哈希真实性**（§11.6）。用例 75 → **76**（`D12`），
> 快照用例仍 9 条。见 §11。
>
> 0.6.0 修订（2026-11-14 第五轮审计后）：**`same_artifact` 不再把"缺失字段相等"
> 当来源证据**——`path` 与 `sha256` 必须均非空，文件级与行级定位显式分层、互不相认。
> 用例 71 → **75**，快照用例 8 → **9**。见 §10。
>
> 0.5.0 修订（2026-11-14 第四轮审计后）：**确认重复的门槛提到 `recorded`**（不再只用
> seed+预算两项自查）、**数据身份改用内容哈希**（`DATA_ID_KINDS`）、**执行状态整块复算**
> （不再只信 `run_id_status` 字符串）、`--check` 校验整个证据快照。用例 66 → **71**，
> 另有 `test_stage0_freeze.py` **8/8** 测快照漂移。见 §9。
>
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
- `test_circuit_registry.py`（76 条用例，实跑 **76/76 通过**）
- `test_stage0_freeze.py`（9 条快照完整性用例：篡改副本/清单/原始证据/删字段，实跑 **9/9**）
- `docs/round_2026-11-13/STAGE0A_SCHEMA_PRESSURE.md`（三谱压测，只读既有产物）
- `docs/round_2026-11-13/STAGE0_RECORDS_MAP.md`（239 条历史记录映射，v4，0.7.0 下重跑 `--check` 通过）

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
| **重复判定的证据门槛** | `L6`–`L8` / `D1`–`D2` | 摘要相同 ≠ 重复；**确认重复要求两侧执行身份都是 `recorded`**；证据不足只标 `possible_duplicate` 且**不扣**独立数 |
| **数据身份不是文件名** | `D3` / `D7` | 同名不同内容 ⇒ 两个 `run_id`；`ref_only` 不能升 `recorded`；自报等级与实际不符被拒 |
| **执行状态不可自述** | `D6` | 有数据/seed/预算却标 `unconfirmed`、或原因指向错的要素，都被拒 |
| **同一产物必须来源可定位** | `D8`–`D11` | 缺 `sha256`、同名不同内容、文件级/行级混比、行号不同，都**不能**确认同一产物；`section` 脱离 `row` 报错 |

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
    预算 `recorded`、**代码版本已知**、**数据有内容哈希**；`D-` 为推导；缺任一项
    **不发号**（`None` + `RUN_ID_MISSING` 原因）。候选为 `U-` 时不许发 `run_id`。
19. **`run_id` 可复核**：`observation_block` 用观测里的 `budget_params` 重算
    `run_id`，不符即报错；给了 `run_id` 却没给 `budget_params` 也报错。
20. **重复判定看来源，不看摘要**：`duplicate_verdict()` 的三态
    `distinct` / `possible_duplicate` / `confirmed_duplicate`；
    `independent` 是派生值，只有 `confirmed_duplicate` 才为 `False`；
    `confirmed_duplicate` 必须给 `kind`（`same_artifact` / `rerun_same_conditions`）；
    "同条件但结果不同"记为 `distinct` + `contradiction`（可复现性问题），不合并。
21. **确认重复要求两侧执行身份都是 `recorded`**（0.5.0）：`_conditions_recorded()`
    逐项检查 `run_id_status=="recorded"` ∧ seed `artifact` ∧ 预算 `recorded` ∧
    `software.code_version` 非空 ∧ `data_id_kind=="content_sha256"`。只查其中两项
    ——0.4.0 的做法——会把"同 seed 同预算同摘要但代码版本未知"判成确认重复。
22. **数据身份只认内容哈希**（0.5.0）：`execution.data` 的 `sha256` 必须是 64 位
    十六进制（拿文件名冒充直接报错）；没有 `sha256` 时 `data_id_kind="ref_only"`，
    **最高只到 `derived`**，不能参与确认重复。`ref` 不进 `run_id` 哈希
    （路径是机器相关的），有 `sha256` 时只哈希内容、无则退到 `ref` 并留痕。
23. **执行状态整块复算**（0.5.0）：`_check_execution` 不只重算 `run_id`，还用
    `_expected_status()` 复算 `run_id_status` 与 `unavailable_reason`（优先序
    `data` → `seed` → `budget`）。"要素齐全却自称缺失""缺 A 却报 B"都报错，
    `data_id_kind` 自报与实际不符也报错。
24. **同一产物必须来源可定位**（0.6.0）：`_artifact_key()` 要求 `path` 与 `sha256`
    **均非空**；有 `row` 走行级键 `("row", path, sha256, row, section)`，无 `row`
    走文件级键 `("file", path, sha256)`，两类键**不互比**；`section` 脱离 `row`
    直接报错。缺字段不构成证据——"两个 `None` 相等"不能推出同一产物。
25. **`sha256` 非空 ≠ 有效**（0.7.0）：`_artifact_key()` 在造键前要求 64 位十六进制
    （大小写不敏感，首尾空白归一），否则**抛 `ValueError`**。两条
    `{"path": "x", "sha256": "same"}` 不许因为"这个字符串相等"凑出 `same_artifact`；
    格式错误也不许静默降级成 `None`（那是调用方的 bug，不是来源缺失）。
    **空串语义按字段分层，两处故意不同**（§11.5）：`source.sha256` 的空串/全空白算缺失
    （返回 `None`）；`execution.data.sha256` 的空串算错误（`ValueError`，"没有哈希"要写
    不给键或 `None`）。同一条格式判定在 `_check_data` 里复用。
    注意这只校验**外形**：64 位十六进制不等于"这个哈希是对文件真算出来的"，
    后者靠 `MANIFEST.data_sha256_files` / `source_sha256` / `--check` 的证据链（§11.6）。

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

`registry_digest = b2675c8255bea716`（本版，`0.7.0`）。manifest 会记录它，注册表被改可被发现；
`consistency_id`、观测 schema 常量（含身份枚举、缺失语义、预算参数名）、
**执行身份的哈希口径（`RUN_ID_BASIS`/`RUN_ID_SCOPE`/`RUN_ID_MISSING`）与重复判定三态**、
**数据身份等级（`DATA_ID_KINDS`）与"确认重复必须 `recorded`"这条口径**、
**产物定位的层级与"`sha256` 必须 64 位十六进制"（`ARTIFACT_SCOPES`/`ARTIFACT_KEY_FIELDS`/
`artifact_sha256_hex_len`）**
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
| 哈希什么 | `candidate_id` + 数据（**内容 sha256** + `f_min/f_max`；无 sha256 时退到 `ref` 并由 `id_kind` 留痕）+ seed + seed 的证据类别 + 实际预算（含 `source`）+ 软件 |
| 为什么含数据 | 同候选同 seed 同预算跑 60 条谱是 60 次执行；不写数据会共享一个号 |
| 为什么不哈希 `ref` | 完整路径是机器相关的，会让同一个实验换台机器就换号 |
| 为什么含 `source` | 数值相同、证据不同（产物记录 vs 档名推导）不能拿同一个号，否则推导出的条件被当成观测到的条件 |
| 同条件重跑 | **共享同一个 `run_id`**，实例由 `source.path` 区分。这正是它能当去重键的原因 |
| 什么时候不发 | 缺数据 / seed / 预算 ⇒ `None` + `unavailable_reason` |

`recorded`（`R-`）要求四项齐：seed 来自产物、预算 `recorded`、代码版本已知、
数据有内容哈希。**本批 0 条**——六条 `nested.json` 已满足 seed、预算、数据三项，
只差代码版本：没有一份历史产物记 git sha / 包版本。这是待补字段，不是降门槛的理由。

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

## 9. 0.5.0 修订（针对第四轮审计：判重门槛、数据身份、快照校验）

### 9.1 三项都成立，其中第一项是自相矛盾

| 编号 | 审计意见 | 核实结论 | 修法 | 用例 |
|---|---|---|---|---|
| P0 | `duplicate_verdict` 的确认条件只查 seed+预算，不查 `run_id_status` | **成立且自相矛盾**：同一份报告说"0 条 `recorded`（没有产物记代码版本）"，却把两对判成 `confirmed_duplicate`——用的正是自己定义的门槛之外的两项 | `_conditions_recorded()` 按 `recorded` 的**完整定义**整体判；本批两对降为 `possible_duplicate`，`confirmed` 归零 | `D1` / `D2` |
| P1 | 数据身份用文件名，可能碰撞 | **成立**：`data.ref` 在 v3 就是 `Path(...).name`；两条同名同窗不同内容的谱会撞 `run_id` | `DATA_ID_KINDS`：`sha256` 进哈希、`ref` 只作标签；`ref_only` 不能升 `recorded` | `D3` / `D7` |
| P1 | `--check` 没校验 MANIFEST 的注册表指纹与 `source_sha256` | **成立**：v3 只比重算的两份数据文件；注册表改了但序列化没变时会"通过" | `verify_frozen()` 四层校验 + `--frozen <dir>`；`test_stage0_freeze.py` 用篡改副本测 | `D4a`/`D4b`/`D5a`–`D5e` |
| — | `_check_execution` 的 `unconfirmed` 分支过宽 | **成立**：0.4.0 只查"原因在枚举里"，"要素齐全却自称缺 seed"能过 | `_expected_status()` 复算状态与原因，优先序 `data`→`seed`→`budget` | `D6` |

顺带补一处同类隐患：`execution=None` 的默认块原本写 `unavailable_reason="not_computed"`，
而 `unconfirmed` 只允许 `RUN_ID_MISSING` 里的原因——**默认块自己就违反了这条不变量**。
现在 `RUN_ID_MISSING` 增哨兵 `not_provided`（"调用方根本没给执行块"），默认块用它，
`D6` 里加断言：未给执行块时 `run_id is None` ∧ 原因 ∈ `RUN_ID_MISSING`。

### 9.2 计数口径因此改变（必须两数同报）

| 口径 | 值 | 含义 |
|---|---|---|
| `n_independent` | **239** | 只扣**已确认**重复；0 条确认 ⇒ 不扣 |
| `n_independent_strict` | **237** | 把 2 条 `possible_duplicate` 也扣掉的对照 |
| 唯一 `run_id` | **237** | 那两对的条件指纹相同（同候选/同 seed/同预算/同数据） |

v3 报的"237 独立"其实是严格口径，却写成了主口径。**现在两个数一起给**，
单独引用"239 独立"会把两对疑似重跑算成新信息，单独引用"237"则是把猜当结论。

### 9.3 与 0.4.0 的关系

20 条硬约束**一条未放宽**，只新增 21–23；`candidate_id` 六个字段、两处结构一致性、
三态枚举全部原样。被收紧的只有"什么算确认重复"——这是**收紧**，不是让步。
`registry_digest` 因此变化（`0f59a5686c2d0c5b` → `9e7051c9b4cd720b`），冻结清单同步更新。

### 9.4 仍未做的

- `R-` 前缀仍 0 条：卡在代码版本字段，不是规则问题。
- runner（阶段 1）：未写，**Stage 0 不冻结**。
- 阈值（`acf` 上限、区间宽度、`n_viable`）：仍一个没定。

## 10. 0.6.0 修订（针对第五轮审计：`same_artifact` 的缺字段碰撞）

### 10.1 确实是仍在的边界，而且与 0.5.0 同源

0.5.0 把"确认重复"整体收紧到 `recorded`，但 `same_artifact` 这一支**绕过了那道门**：

```python
loc = ("path", "sha256", "row", "section")
same_art = (sa.get("path") is not None and all(sa.get(k) == sb.get(k) for k in loc))
```

两条 `{"path": "result.json", "sha256": None}` 会因为"两个缺失值相等"被判成同一产物，
直接进 `confirmed_duplicate`、从独立数里扣掉。这和 0.4.0 的错**是同一个毛病**：
拿"字段相等"当"证据存在"。0.5.0 只在 `rerun_same_conditions` 那一支修了，这一支漏了。

### 10.2 修法：定位键显式分层，缺字段不构成证据

`_artifact_key(source)` 归一成两类**不可互比**的键：

| 输入 | 键 | 何时可得 |
|---|---|---|
| `path` 与 `sha256` 均非空、无 `row` | `("file", path, sha256)` | 整份产物一条记录 |
| `path` 与 `sha256` 均非空、有 `row` | `("row", path, sha256, row, section)` | 产物内多条记录 |
| `path` 或 `sha256` 缺失 | `None` | **不能参与确认** |

- 前缀不同 ⇒ 文件级与行级永不相等。"一条文件级、一条行级"不是同一个定位。
- `section` 脱离 `row` 直接报错：分段定位没有行号就无从定位，静默忽略它等于
  把同一 `path` 下两个不同分段的记录当成同一行。
- 不确认时把原因逐条写进 `duplicate_evidence["artifact_gap"]`
  （缺哪个字段 / sha256 不同 / 层级不同 / 行不同），不留 `artifact_equal=False` 这种只说结果的话。

### 10.3 七种情形现在是分开的

| `path` | `sha256` | `row`/`section` | 判定 |
|---|---|---|---|
| 同 | 两边都缺 | — | `possible_duplicate`（缺字段不算证据） |
| 同 | 一边缺 | — | `possible_duplicate` |
| 同 | 不同 | — | `possible_duplicate`（同名不同内容） |
| 不同 | 同 | — | `possible_duplicate`（同内容不同路径是另一条来源） |
| 同 | 同 | 行不同 / 分段不同 | `possible_duplicate`（是两条记录） |
| 同 | 同 | 一条文件级、一条行级 | `possible_duplicate`（层级不同） |
| 同 | 同 | 同层且相同 | `confirmed_duplicate` / `same_artifact` |

### 10.4 用例与计数

- `test_circuit_registry.py`：71 → **75**（`D8` 缺 sha256、`D9` 同名不同内容、
  `D10` 文件级/行级分层、`D11` section 必须有 row）
- `test_stage0_freeze.py`：8 → **9**（新增 `D5f`：清单删掉一个 `files` 条目必须失败
  ——`verify_frozen` 第 2 层遍历的是清单自己的 `files`，少一条就少查一条，
  真正抓住它的是第 1 层"重新生成的 MANIFEST 文本 vs 冻结字节"）
- 239 条记录的**判重结论**没有一条受影响：本批没有缺 `sha256` 的 `source`，两对仍是
  `possible_duplicate`，`n_independent` 239 / 严格 237 不变
- 但 239 行的 `run_id` **全部变号**（`D-xxxxxxxxxx`）：`software.registry_version`
  在 `RUN_ID_BASIS` 里，0.5.0 → 0.6.0 会改变执行指纹。这是设计后果，不是漂移——
  唯一 `run_id` 仍是 237 个，去重关系不变。代价是"跨 registry 版本比较 run_id"
  这件事本身不成立：版本不同即不同指纹。
- `registry_digest`：`9e7051c9b4cd720b` → `43f2d2ac08f84c38`（`ARTIFACT_SCOPES` /
  `ARTIFACT_KEY_FIELDS` / `confirm_same_artifact_requires_nonempty_sha256` 进指纹）

### 10.5 仍未做的

- `R-` 仍 0 条（卡代码版本）、runner 未写、阈值未定——与前两轮一致，**Stage 0 不冻结**。

## 11. 0.7.0 修订（针对第六轮审计：`sha256` 只查非空）

### 11.1 缺陷：从"缺失值相等"换成"无效值相等"

0.6.0 把门槛改成 `_nonempty_str(path) and _nonempty_str(sha)`。它挡住了 `None`，
但只回答"填了没有"，回答不了"填的是不是哈希"：

```python
a = {"path": "result.json", "sha256": "same"}
b = {"path": "result.json", "sha256": "same"}
```

两条都会构造出 `("file", "result.json", "same")`，键相等 ⇒ `confirmed_duplicate` /
`same_artifact` ⇒ 从独立数里扣掉。这与 0.5.0 拿"两个 `None` 相等"当证据**是同一个
毛病**，只是把缺失值换成了随手写的字符串。字段存在 ≠ 字段内容有效。

### 11.2 修法：格式校验独立成函数，两处复用

```python
def _valid_sha256(v) -> bool:      # 64 位十六进制，大小写不敏感，首尾空白归一
    v = v.strip().lower() if isinstance(v, str) else v
    return isinstance(v, str) and len(v) == SHA256_HEX_LEN and all(c in _HEX_DIGITS for c in v)
```

`_artifact_key()` 在造键前调用它。**无效就抛 `ValueError`，不返回 `None`**：

| 输入 | 0.6.0 | 0.7.0 |
|---|---|---|
| `source.sha256=None` / 空串 / 全空白 | `None` ⇒ 不能确认 | 同（这是"缺失"，见 §11.5） |
| `execution.data.sha256=""` / 全空白 | `ValueError` | 同（**不是**缺失，见 §11.5） |
| `sha256="same"` | 造键 ⇒ **可确认** | `ValueError` |
| `sha256` 63 位 / 65 位 | 造键 ⇒ **可确认** | `ValueError` |
| `sha256` 含非十六进制字符 | 造键 ⇒ **可确认** | `ValueError` |
| `sha256` 是 `int`/`list` | 造键/`None` 不定 | `ValueError` |
| 合法 64 位、大小写不同 | 两个不同键 ⇒ 永不相等 | 归一后**同一个键** |

选报错而不是返回 `None`：这不是"来源信息缺失"，是"调用方提供了违反字段定义的数据"。
静默降级会把 bug 藏成"证据不足"，看起来安全，实际让人永远查不到是谁写错了字段。
同一函数也替换了 `_check_data()` 里原来内联的 `len==64 and all(...)`，两处口径只剩一份。

### 11.3 用例

- `test_circuit_registry.py`：75 → **76**。新增 `D12`：合法值可造键（文件级与行级）、
  大小写/首尾空白归一、伪哈希 / 63 位 / 65 位 / 非十六进制 / 非字符串一律报错、
  `duplicate_verdict` 入口同样拦住两条 `"same"`、空串在 `source` 算缺失而
  在 `execution.data` 算错误（§11.5 那两处一并钉进用例）。
- `D8`–`D11` 的夹具同步换掉：原来用 `"s1"`/`"s2"`/`"s3"` 当哈希（**它们本身就是
  11.1 那个 bug 的样本**），现在统一用 `_SHA`/`_SHA2`/`_SHA3`（`a`×64 / `b`×64 / `c`×64）。
  这些用例的断言内容未变——它们证的是缺字段与层级，不是格式。
- `test_stage0_freeze.py`：仍 **9/9**。注册表版本变了，快照按 0.7.0 重冻结。

### 11.4 计数与指纹

- 239 条记录**零影响**：本批 `source.sha256` 全部是 `hashlib` 算出的 64 位十六进制
  （实测 239/239 长度 64）。`coverage.json` 重算字节与 0.6.0 冻结副本**逐位相同**
  （`d58a14a6d0398a00`）：239 条 / 独立 239 / 严格 237 / 唯一 `run_id` 237 /
  2 对 `possible_duplicate` / 0 条 `confirmed_duplicate` / `R-` 0 条，一个都没动。
- `records.jsonl` 字节变了（`ebd8f8a08f1ed071` → `bf22f64b75ea8ea9`）：239 行
  `run_id` 全部换号，因为 `software.registry_version` 在 `RUN_ID_BASIS` 里。
  这是设计后果，不是漂移——**跨 registry 版本比 `run_id` 不成立**这个结论继续有效。
- `registry_digest`：`43f2d2ac08f84c38` → `b2675c8255bea716`（`confirm_same_artifact_
  requires_valid_sha256` / `artifact_sha256_hex_len` 进指纹，`…requires_nonempty_sha256`
  退出——被更严的那条取代）。
- 第六轮审查后的补充（§11.5/§11.6）**只动注释、文档与 `D12` 的断言**，不改任何判定：
  `_check_data()` 对空串报错的行为本来就是这样，这轮只是把它写清楚并钉进用例。
  因此 `registry_digest` 保持 `b2675c8255bea716`、快照无需重冻结——把它也塞进指纹
  会让"零行为变化"看起来像新规则，反而误导以后读 `MANIFEST` 的人。

### 11.5 空串语义：两处**故意**不同（第六轮审计指出的口径差）

同一件事在两处有不同答案，这本身不是缺陷，写不清楚才是：

| 字段 | 空串 / 全空白 | 为什么 |
|---|---|---|
| `source.sha256` | **缺失** ⇒ `_artifact_key()` 返回 `None` ⇒ 最多 `possible_duplicate` | 产物定位是映射脚本**事后补**的描述字段，老产物本来就可能没算哈希。缺了就是缺了，不该报错，也不该被当成证据 |
| `execution.data.sha256` | **错误** ⇒ `ValueError` | 这层是**身份输入**，直接决定 `recorded` 能不能发。"没哈希"已经有专门写法（不给这个键，或给 `None`），会得到 `id_kind="ref_only"`、如实降级到 `derived`；再写一个空串，等于把"忘了算"伪装成"算出来是空"，而这层身份的全部意义就是不让降级**静默**发生 |

`_check_data()` 里那段注释把这个理由写在代码旁边，`D12` 把它钉进用例
（`source` 空串 → `None`；`execution.data` 空串 → 报错；不给键 / 给 `None` → `ref_only` + `derived`），
免得将来有人为了"统一口径"把两边改成一样。

### 11.6 认识边界：格式校验 **不等于** 哈希真实性

`_valid_sha256()` 只证明字符串**长得像** SHA256（64 位十六进制），不证明它是对某份
文件内容真算出来的。真实性靠证据链，不靠格式：

- 映射脚本对原始谱重算内容哈希，结果落在 `MANIFEST.data_sha256_files`（63 份），
  `--check` 逐份重算比对；
- `source_sha256`（9 份证据文件）同样逐份重算；
- `--check` 四层里的"重算文本 vs 冻结字节"挡住记录被改而清单没改。

所以本轮的修复只回答"这个值有没有资格**参与比较**"，不回答"这个值是不是真的"。
把 64 位十六进制读成"哈希可信"就是把外形当成了事实。

### 11.7 仍未做的

- **历史执行身份缺代码版本**（233 条 `producer_code` seed、0 条 `code_version`）：
  结构问题，不是本轮能修的，`R-` 继续 0 条。已做只读审计：6 条只差 `code_version`
  （且这 6 条含本批仅有的 2 对 `possible_duplicate`——若能补，独立数会降到 237），
  但产物里零身份字段、时间只给上界、库模块无副本 ⇒ **不能事后补**。见
  `HISTORICAL_EXEC_IDENTITY_GAP.md`。
- **233 条预算数值由档名推导**：同上。
- runner 未写、阈值未定、`n_viable` 判据未定——**Stage 0 不冻结**，不进阶段 1。

原则记在这里：Stage 0 的目标不是让所有历史记录都能被确认，而是让每条记录**只能
获得它实际拥有的证据等级**。`R-` 为零不是系统失败，是"历史产物没记代码版本"这个
事实的正确反映；同样，两条随手写 `"same"` 的记录不该被"合并"，而该让写错的地方报错。
