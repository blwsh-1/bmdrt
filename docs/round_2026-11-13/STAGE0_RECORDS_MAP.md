# 阶段 0：历史产物的结构化映射（机器可读，v3）

状态：**映射已按第二轮审计返工**。本轮只做两件事——去重判定改按来源、候选/执行两层身份分开。
不跑拟合、不改生产、不进入阶段 1。

产物：
- `map_stage0_records.py`（v3，只读 `evidence/`，含身份完整性断言）
- `out/stage0_records/{records.jsonl, coverage.json}` → 字节级副本在 `evidence/stage0_records/`
- `evidence/stage0_records/MANIFEST.json`（**上一轮并不存在**，见 §8）
- 记录数 **239**，独立 **237**（2 条确认重复），`possible_duplicate` **0**

---

## 0. v3 改的两件事 + 顺带查出的两处身份错误

### 0.1 去重不能拿"摘要哈希相同"当证据（审计 P0，成立）

v2 的 `_fingerprint()` 只哈希 `candidate_id + fit_observation + param_estimate`，
`mark_duplicates()` 见指纹相同就直接写 `independent=false`。问题不是"这两条不该合并"，
而是**这个函数证明不了它声称的事**：摘要相同只能说明几个字段相同，
既可能是同一次运行的两次导出，也可能同种子重跑，还可能只是四舍五入后撞车。

v3 把判定移进 `circuit_registry.duplicate_verdict()`，输入是**来源**：
产物 `path/sha256/row/section`、`run_id`、seed 及其证据等级（`artifact` /
`producer_code` / `tier` / `absent`）、预算数值及其来源（`recorded` /
`derived_from_tier`）。`summary_equal` 只是其中一个输入事实，不是结论。

三态（`DUPLICATE_STATES`）：

| 状态 | 条件 | 计数 |
|---|---|---|
| `distinct` | 候选不同；或来源与摘要均不同；或**同条件但结果不同**（`contradiction`） | 算 |
| `possible_duplicate` | 摘要相同但 seed/预算证据不足，或执行身份不同/不可判 | **算**（只标记） |
| `confirmed_duplicate` | 同一产物；或同 `run_id` + 摘要相同 + seed 与预算都由产物记录 | 不算 |

`independent` 是 `duplicate_status != "confirmed_duplicate"` 的**派生值**。
证据不足时只标 `possible_duplicate`，`coverage.json` 另给 `n_independent_strict`
（把 possible 也扣掉的对照口径）——本批两者都是 237，因为 0 条 possible。

### 0.2 `candidate_id` 与执行身份是两层（审计 P1，成立）

`compute_candidate_id()` 哈希的是 `budget_profile`（档名 `"standard"`），
**不含** `target_accept=0.98` 这类实际数值。同一次历史对照里
`VALIDATION_STANDARD` 的 `standard` 是 0.98、`bm_select.BUDGETS["standard"]` 是 0.90——
两个 `standard` 可以拿到同一个 `candidate_id`。

处置：**`candidate_id` 不动**，新增 `execution_identity()` → `run_id`：

| 层 | 表示什么 | 进不进哈希 |
|---|---|---|
| `candidate_id`（`C-`） | 电路结构 + 配置 + 初始化 + 似然 + 预算**档名** | 模型定义 |
| `run_id`（`R-`/`D-`） | 候选 + **数据** + seed + **实际预算数值** + 软件 | 执行条件 |

`R-` = 三者都来自产物；`D-` = 至少一项靠代码/档名推出；`None` = 缺要素，
**不发号**（写 `unavailable_reason`，属 `RUN_ID_MISSING`）。

`data` 必须进哈希：同一候选、同 seed、同预算下跑 60 条不同谱是 60 次执行，
不写数据就会共享一个 `run_id`，去重时全被读成"同条件不同结果"（假告警，用例 L9）。

`observation_block` 会**重算** `run_id` 并与观测里的 `budget_params` 对拍，
预算事后被改动、或 id 从别处抄来，当场报错（用例 L5）。

### 0.3 pilot 的 `sensitivity` 段不是 nested（v2 遗留，同类身份错误）

代码是 `voigt_fit(f, Z, None, seed=0, init=voigt_init(f, Z))`
（`coating_voigt_pilot.py:268`）——**voigt K=2、自由 σ、数据驱动初值**，
只对 `verdict=unusable_sampling` 的 46 条重跑。v2 把它映射成 `nested/capacitive/L2`，
于是这 46 条与真正的 `nested` 段**共享 `candidate_id`**，残差被记到了另一个模型名下。
v3 按调用点分流，`sensitivity` 现在是 `voigt / data_driven / free-sigma`。

### 0.4 voigt 初值：v2 把 v1 对的改错了

`voigt_init` 的 docstring 写着"数据驱动初值"。但 `voigt_prod` / `voigt_fix2` 调的是
`voigt_fit(f, Z, ...)`，**不传 `init`** ⇒ `voigt_fit:160` 把 `None` 传给
`C.fit(init_values=None)` ⇒ `bm_core.fit:603` 只在 `if init_values:` 时设
`init_to_value`，None 走 numpyro `init_to_uniform`。

v1 标 `uniform` 是**对的**，v2 因为读 docstring 改成了 `data_driven`，把 120 行标反。
v3 改回，并把规则键拆成 `voigt_default` / `voigt_data_init` 两条：
**函数文档描述的是它自己，不是谁在调用它。**

### 0.5 v2 修对的部分（保留）

- `_INIT_HINT` 用 `p.parent.name` 从未命中（归档是平铺的），回落成真实档名 `"uniform"`
  → 改为按代码路径判定，查不到就 `INIT_UNKNOWN`、`U-` 前缀、禁进 manifest。
- AZ31 的 2 条 voigt 行被 `branch = r.get("branch") or "parallel"` 算成 nested，
  与真实 nested 行 `candidate_id` 撞车 → 按 `label` 分流 + 断言 id 唯一。

---

## 1. 初始化判定表（v3 更正后）

| 产物 | 条数 | 生成者 / 调用点 | 判定 |
|---|---|---|---|
| `*.nested.json` | 6 | `nested_fit.py:339` 不传 `init_values` | `uniform` |
| AZ31 6 行 | 6 | `fit_nested:161` / `fit_voigt:105` 都不传 | `uniform` |
| pilot `voigt_prod` / `voigt_fix2` | 120 | `voigt_fit(init=None)` | `uniform`（v3 改回） |
| pilot `nested` | 60 | `voigt_fit`→`data_init`（:219） | `data_driven` |
| pilot `sensitivity` | 46 | `voigt_fit(init=voigt_init)`（:268） | `data_driven`（v3 换拓扑） |
| oracle | 1 | 读文件名里的真值参数 | `value` |

核对：`uniform 132 + data_driven 106 + value 1 = 239`。全部 `identity_status=derived`。

---

## 2. 记录构成（239 条 / 15 个唯一候选）

| 来源 | 条数 | 说明 |
|---|---|---|
| `*.nested.json` | 6 | 涂层 4（parallel/capacitive/quick/1113）、钝化 2（default/1113） |
| oracle 诊断 | 1 | `truth_init`，初值用真值构造，`production_reachable=false` |
| pilot `voigt_prod` | 60 | voigt / 自由 σ / 默认初值 |
| pilot `voigt_fix2` | 60 | voigt / 固定 2% / 默认初值 |
| pilot `nested` | 60 | nested capacitive L2 / 逐点 2% / 数据驱动初值 |
| pilot `sensitivity` | 46 | **voigt / 自由 σ / 数据驱动初值**，仅 unusable 行 |
| AZ31 | 6 | 4 条 nested + 2 条 voigt，真实谱无真值 |

---

## 3. 重复判定：本批的两个确认案例

| 记录 | `duplicate_of` | `kind` | 证据 |
|---|---|---|---|
| coating `standard_1113` | coating `standard` | `rerun_same_conditions` | 同候选、`config.seed=0`**产物记录**、`budget_kw` 五项逐项相等、`metrics/diagnostics/params.median` 逐位相同、输出目录不同（`out/_user_fit_nested` vs `out/nested_fit_cap_standard`）、`elapsed` 57.9 vs 110.1、`with_voigt` 不同（只多跑一次参照，不改本候选后验） |
| passivation `default_init` | passivation `1113` | `rerun_same_conditions` | 同上；20.1 vs 31.4 s |

`with_voigt` 不同不影响本候选身份：它只是额外跑一次 voigt 参照，
`candidate_id` 与 `run_id` 都不含它。两条因此 `run_id` 相同、`summary_equal=true`。

**为什么这次可以扣，v2 不行**：v2 只有"摘要相同"；v3 有 `config.seed`（产物自己写的）
和 `budget_kw` 五项（产物自己写的）——`conditions_observed=true`。
若 seed 只是从代码推出的（`producer_code`），同条件同摘要也只标 `possible_duplicate`，
不从分母里扣。规则在 `duplicate_verdict()` 里，不在文档里。

---

## 4. 三类事实必须分开写（审计要求）

同一句话可能来自三种强度完全不同的依据。本文件与 README 都按此标注：

| 类别 | 含义 | 本批例子 |
|---|---|---|
| **A 产物核对** | 直接读产物字段，可逐位复核 | `config.seed=0`、`budget_kw` 数值、`metrics`、`elapsed_s`、`files.json`、`input.path` |
| **B 代码路径推导** | 产物没写，但生成脚本唯一且调用点可读 | 初始化档（`init_values` 未传）、pilot/AZ31 的 seed（调用点写死 0）、预算数值（`BUDGETS` 档名 + 脚本覆盖） |
| **C 脚本自动判定** | 映射脚本按 A/B 组合算出来的结论 | `candidate_id`、`run_id`、`duplicate_status`、`ratio` 的 `denominator_scope` |

**C 类不能倒推 A 类。** 例如 `duplicate_status=confirmed_duplicate` 是 C 类结论，
它的依据（A 类的 seed 与预算、`summary_equal`）逐项写在 `provenance.duplicate_evidence` 里，
可以复核；而 `run_id` 是 C 类指纹，`run_id_status=derived` 明确说明它不是产物记录的。

v2 文档里"6 个 nested.json 只有 4 次唯一运行，两对是同配置同 seed 重跑"这句，
当时只有 C 类依据（摘要指纹）却写成了 A 类事实。**现在它才是 A 类**：seed 就在
`config.seed` 里。v3 的措辞按此重写。

---

## 5. 六项字段的落地情况（v3 计数）

| 字段 | 回答的问题 | 本批 |
|---|---|---|
| `param_estimate` | 估到多少、离真值多远 | **67 条**（pilot nested 60 + nested.json 6 + oracle 1）；172 条 `null`（pilot voigt/敏感性未存逐参数误差 166 + AZ31 6 条真实谱无真值） |
| `param_stability` | 后验区间稳不稳 | **6 条**（只有 `nested.json` 存了 `ci90`） |
| `ratio` | 残差 / 噪声底 | **226 条**，全部带 `n_used`/`n_total`/`n_usable`（v2 是 180：`sensitivity` 段原本漏了，v3 补上） |
| `noise_floor` | 底从哪来 | 226 `truth_forward`、13 `not_available` |
| `predictive` | ELPD 能否当样本外证据 | 12 条有 `elpd`（6 条 `nested.json` 标 `False` + 原因；6 条 AZ31 标 `True`，自由 σ） |
| `budget_params` | 实际用了多少算力 | 6 `recorded`、**233** `derived_from_tier`（v2 是 232 + 1 `unavailable`；oracle 的预算其实写在 `diag_passivation_init.py:57-60`，v3 按代码推出，并带 `tier=standard` 与 `target_accept=0.98` 覆盖） |

`sensitivity` 的 `ratio` 分母：`n_total=60`（全表行数）、`n_used=46`（实际重跑的行）、
`n_usable=46`，`usable_rule` 里写明"只在 unusable 行上重跑"，不要读成收敛子集。

---

## 6. 执行身份落地情况

| 项 | 本批 |
|---|---|
| 唯一 `run_id` | **237**（239 条记录，2 条确认重复共享） |
| `run_id_status` | `derived` **239**、`recorded` **0** |
| seed 来源 | `artifact` **6**（`nested.json` 的 `config.seed`）、`producer_code` **233** |
| 预算来源（身份内） | `recorded` **6**、`derived_from_tier` **233** |
| 代码版本已知 | **0** |

**`recorded` 目前不可达**，原因具体：`nested.json` 记了 seed 与预算，但**没有一份产物
记代码版本**（git sha / 包版本）。这是待补字段，不是把门槛降低的理由。
`R-` 前缀因此一条都没发出去——这本身就是结论：历史记录的"执行身份"最高只能到"推出来的"。

---

## 7. 承载力判定

| 能力 | 结论 | 依据 |
|---|---|---|
| 残差 / 采样状态 / 解释限定 | ✅ | 70 条"跑完但采样未过"入库且标 `descriptive_only` |
| 候选身份（模型层） | ✅ 有条件 | 239 条全 `derived`；0 条 `recorded` |
| 执行身份（条件层） | ⚠️ 部分 | 239 条 `derived`；seed 只有 6 条是产物记录；代码版本 0 条 |
| 参数估计 vs 真值 | ⚠️ 部分 | 67 条带 `rel_err` |
| 比值与噪声底 | ✅ | 226 条带完整分母与底来源 |
| ELPD 可用性 | ✅ | 12 条带 `valid_for_comparison` + 原因 |
| **独立观测计数** | ✅ | 三态判定；本批 2 条 `confirmed_duplicate`（证据逐项可复核）、0 条 `possible` |

**阶段 1 仍不进。** 剩下的硬缺口没变：233/239 条的预算数值是"由档名/脚本推出来的"，
且**没有一条记录代码版本**。做历史对照时"同档名"不等于"同预算"，更不等于"同代码"。

---

## 8. `MANIFEST.json`：上一轮确实不存在（更正）

上一轮报告里提到 `evidence/stage0_records/MANIFEST.json`，实际**没有写入**——该目录
当时只有 `records.jsonl`、`coverage.json`、`README.md`。审计读不到文件是对的。

v3 补上，并且由脚本生成、不手写：

```bash
python docs/round_2026-11-13/map_stage0_records.py --freeze   # 写副本 + MANIFEST
python docs/round_2026-11-13/map_stage0_records.py --check    # 与冻结副本逐字节比对
```

`MANIFEST.json` 含：两个文件的 `sha256`/字节数、`registry_version` 与 `registry_digest`、
记录数/独立数/重复态、`run_id` 统计、以及 9 份原始证据的 `path → sha256`
（可用 `git cat-file blob` 直接复核入库副本）。

---

## 9. 复算

```bash
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-13/map_stage0_records.py
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-13/map_stage0_records.py --check
D:/autoeis_env/Scripts/python.exe test_circuit_registry.py
```

脚本内置断言（不通过直接退出，不产出半成品）：
`record_id` 唯一；同一 `candidate_id` 必须对应同一份六字段定义；
`identity_status ∈ IDENTITY_STATES`；`init_profile=unknown` ⇒ `U-` 且不可比较；
`duplicate_status ∈ DUPLICATE_STATES` 且 `independent` 与它一致；
`run_id` 与状态前缀一致，未发号必须有 `RUN_ID_MISSING` 里的原因。
