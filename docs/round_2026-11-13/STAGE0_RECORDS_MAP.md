# 阶段 0：历史产物的结构化映射（机器可读，v2）

状态：**映射已返工，身份层重做**。本轮不跑拟合、不改生产、不进入阶段 1。

产物：
- `map_stage0_records.py`（v2，只读 `evidence/`，含身份完整性断言）
- `out/stage0_records/{records.jsonl, coverage.json}` → 副本在 `evidence/stage0_records/`
- 记录数 **239**（独立 **237**，重复 **2**）

---

## 0. v2 修的是三个真实缺陷，不是措辞

上一版（v1）声称"13 条身份靠目录名推断"。核查后，那套推断**一次都没有命中**，
而且掩盖了两个更严重的问题。三条都影响身份，不是文档笔误。

### 0.1 `_INIT_HINT` 从未命中，13 条记录的回落值是个真实档名

v1 写 `hint = _INIT_HINT.get(p.parent.name)`，但归档文件是平铺的
`evidence/stage0_three_spectra/*.nested.json`，父目录恒为该目录名，
`_INIT_HINT` 的键（`nested_fit_cap_standard` 等）一个都不匹配。

于是 `init_profile` 走 `hint or "uniform"` → **回落成 `"uniform"`**，
标记 `default_assumed`。问题不在"没推断出来"，而在**回落值是一个合法档名**：
`candidate_id` 照常算出一个 `C-…`，看起来是可复现身份，实际毫无依据。

v2 的处理：查不到就 `INIT_UNKNOWN`，id 前缀变 `U-`，`identity_status=unconfirmed`，
并且**禁止进 manifest**（用例 J1）。默认值只能是"不可用"，不能是"常见值"。

### 0.2 AZ31 的 2 条 voigt 行被算成了 nested，id 与真实 nested 行撞车

v1 写 `branch = r.get("branch") or "parallel"`。AZ31 的 voigt 行 `branch` 是 `None`
（`voigt_row()` 返回 `branch=None, layers=K`），于是被当成 `nested/parallel/L=K`：

| AZ31 行 | v1 记录成 | 后果 |
|---|---|---|
| `parallel/L=1` | nested parallel L1 | — |
| `voigt K=1` | **nested parallel L1** | `candidate_id` 与上一行**完全相同** |
| `parallel/L=2` | nested parallel L2 | — |
| `voigt K=2` | **nested parallel L2** | 同上 |

6 行只剩 4 个 id，`record_id` 也重复两条 ⇒ **voigt 的残差被记在 nested 名下**。
这正是"名称统一了、模型却没法公平比较"的最坏形态。

v2：按 `label` 分流，voigt 行走 `voigt` 拓扑；并加断言
——`record_id` 唯一、同一 `candidate_id` 必须对应同一份六字段定义。现为 6 行 6 id。

### 0.3 pilot 的 voigt 初始化标错

`coating_voigt_pilot.py:118` 的 `voigt_init` 文档明写"**数据驱动**初值（只看 f、|Z|）"，
v1 标成 `uniform`。v2 标 `data_driven`，并把该档补进注册表的 voigt（用例 J5）——
否则 schema 连"这次用的是哪种初值"都写不下来。

### 0.4 顺带查清的来源矛盾（曾经两边说法不一致）

- `fit_family_compare.py` **只写** `family_fit_results.json`，**不写** `.nested.json`。
  所以 `out/_user_fit_nested/*.nested.json` **不是**它的产物，不能因为目录名叫
  `_user_fit_nested` 就认定用了 `data_init`。
- 每个 `nested.json` 的 `files.json` 记录它**自己写到哪**，`input` 记录
  `f_min/f_max` 与含真值参数的文件名 ⇒ 原始目录与真值都能从**产物自身**复原。
  实测：`*_1113.nested.json` 的 `files.json` 指向 `out/_user_fit_nested/`，
  且 `elapsed_s` 不同（110.1s vs 57.9s）、指标逐位相同 ⇒ 是**同配置同 seed 的重跑**，
  不是拷贝。这是可复现性证据，但**不是独立观测**（见 §2.2）。
- 因此 `STAGE0_EVIDENCE_MAP.md` 里把 1113 归入"默认初值"列是**对的**，
  而 v1 的 `_INIT_HINT["_user_fit_nested"]="data_driven"` 是**错的**。

---

## 1. 初始化是怎么判定的（不再是猜档名）

判定依据是**生成脚本的代码路径**，逐条可引：

| 产物 | 生成者 | 判定 | 依据 |
|---|---|---|---|
| `*.nested.json`（6 条） | `nested_fit.py` | `uniform` | `nested_fit.py:339` 调 `T.fit_top(...)` **不传 `init_values`**；`bm_top.fit_top:1006` 当 `init_params=None` ⇒ numpyro 默认 `init_to_uniform` |
| AZ31 6 行 | `m4_real_nested_fit.py` | `uniform` | `S.fit_nested` / `S.fit_voigt` 均不传 `init_values` |
| pilot voigt（120 条） | `coating_voigt_pilot.py:voigt_init` | `data_driven` | `R_inf=min\|Z\|`、`W=(zmax−zmin)/2`、τ 窗内均铺、`φ=0.8` |
| pilot nested（106 条） | `coating_voigt_pilot.py:data_init` | `data_driven` | 由 `min\|Z\|`/`zmax`/中心频点算出 |
| oracle（1 条） | `diag_passivation_init.py` | `value` | 直接读文件名里的真值参数 |

全部 239 条 `identity_status = derived`（**没有一条是 `recorded`**：产物里确实没有 init 字段），
`init_source` 逐条写明到函数级。`production_reachable=false` 的只有 1 条（oracle）。

---

## 2. 记录构成与两个易错点

### 2.1 构成（239 条）

| 来源 | 条数 | 说明 |
|---|---|---|
| `*.nested.json` | 6 | 涂层 4（parallel/capacitive/quick/1113）、钝化 2（default/1113） |
| oracle 诊断 | 1 | `truth_init`，初值用真值构造 |
| pilot `results.jsonl` | 180 | 60 条 ×（voigt 自由 σ / voigt 固定 2% / nested） |
| pilot 敏感性 | 46 | 46 条 unusable 换数据驱动初值后的复跑 |
| AZ31 | 6 | 4 条 nested + 2 条 voigt |

### 2.2 6 个 `nested.json` 只有 4 次唯一运行

按（指标 + 参数 + 采样诊断）指纹比对：

| 指纹 | 文件 | 结论 |
|---|---|---|
| 唯一 | coating quick | 独立 |
| **成对** | coating standard = coating standard_1113 | 同配置同 seed 重跑 |
| 唯一 | coating parallel | 独立 |
| **成对** | passivation default = passivation 1113 | 同配置同 seed 重跑 |

v2 把后两份标成 `independent=false` + `duplicate_of`。**计数时不能重复算**：
若当成 6 次独立观测，等于把同一次运行的两份导出算两遍。

### 2.3 `status` 撞名（v1 踩过）

`nested.json` 的 `status` 恒为 `"exploratory"`，是**成熟度标签**，不是执行状态。
v1 首跑直接把它映射成 `run_status`，6 条全判 `failed`，被 `observation_block` 拦住。
v2 的执行状态只由"有没有 `metrics.rms_rel`"判断。

---

## 3. 六项字段现在能装下了

0.3.0 把审计指出的六项纳入 schema，**语义各自独立**，不能互相顶替：

| 字段 | 回答的问题 | 本批落地情况 |
|---|---|---|
| `param_estimate` | 估到多少、离真值多远 | **67 条**有值（估计 + 真值 + `rel_err`）；其余 172 条为 `null`（pilot 的 voigt/敏感性未存逐参数误差 166 条 + AZ31 6 条，真实谱无真值） |
| `param_stability` | 后验区间稳不稳 | 仅 **6 条** `nested.json` 有 `ci90`；pilot 未存区间 |
| `ratio` | 残差 / 噪声底 | **180 条**有值，全部带 `n_used`/`n_total`/`n_usable` |
| `noise_floor` | 底从哪来 | 226 条 `truth_forward`、13 条 `not_available` |
| `predictive` | ELPD 能否当样本外证据 | **12 条**有 `elpd`（6 条 `nested.json` 标 `False` 并给原因；6 条 AZ31 标 `True`，自由 σ）；其余 227 条为"未计算"并写明理由 |
| `budget_params` | 实际用了多少算力 | 6 条 `recorded`（含 `target_accept=0.98`）、232 条 `derived_from_tier`、1 条 `unavailable` |

### 3.1 `budget_profile` 与 `budget_params` 必须分开

同一个档名 `standard`，`target_accept` 可以是 0.98（`VALIDATION_STANDARD` 硬编码）
也可以是 0.9（`bm_select.BUDGETS`）。只记档名 ⇒ 事后无法复现。
0.3.0 里传档名字符串给 `budget_params` 直接报错（用例 K1）。

### 3.2 `ratio` 的分母：三种口径必须写明

| 口径 | 定义 | 本批数 |
|---|---|---|
| `n_total` | 该批总条数 | 60 |
| `n_used` | 比值有定义的条数 | 60（产物对全部 60 条都算了比值） |
| `n_usable` | 其中**后验诊断通过**的条数 | voigt 自由 σ **15**、voigt 固定 σ **40**、nested **59** |

`n_usable < n_total` 时自动标 `denominator_scope=converged_subset`。
这一步是必要的：文档里 `ratio` 中位 3.36 是在**收敛子集**上算的，
不写分母就会与全样本的 6.468 混报。

---

## 4. 本轮发现的一处成文错误（已改）

`COATING_VOIGT_PILOT_FINDINGS.md` §2 表里，nested 行写"收敛 **60/60**"，
但同表 voigt 行的 14/60、40/60 用的是 `plot_pilot.py:18` 的口径
（`R̂≤1.05` **且** `div==0`）。按同一口径，nested 是 **57/60**，不是 60/60。

两种分母下的数（都已重算）：

| 口径 | nested `ratio` 中位 | `acf` 中位 | `acf>0.5` |
|---|---|---|---|
| 全 60 条 | 0.936 | 0.319 | 2/60 |
| 收敛 57 条 | 0.935 | 0.322 | 2/57 |

结论不受影响（差 0.001），但"60/60"这个数错了，已在 FINDINGS 更正。
另注：registry 的 `sampling_status` 用 `R̂≤1.05 且 ESS≥100`（nested 59/59），
与出图脚本的 `R̂+div` 是**两条规则**，两个数都要写明出处，不能混用。

---

## 5. 承载力判定

| 能力 | 结论 | 依据 |
|---|---|---|
| 残差 / 采样状态 / 解释限定 | ✅ | 70 条"跑完但采样未过"全部入库且标 `descriptive_only` |
| 候选身份 | ✅ 有条件 | 239 条全 `derived`（脚本级依据）；**没有一条 `recorded`** |
| 参数估计 vs 真值 | ⚠️ 部分 | 67 条带 `rel_err`；172 条为 `null`（其中 166 条是产物没存逐参数误差，6 条是真实谱无真值） |
| 比值与噪声底 | ✅ | 180 条带完整分母与底来源 |
| ELPD 可用性 | ✅ | 12 条带 `valid_for_comparison` + 原因 |
| 预算可复现 | ⚠️ 部分 | 仅 6 条 `recorded`；232 条靠档名推数值，**重跑未必一致** |
| 独立观测计数 | ✅ | 2 条重复已标 `independent=false` |

**阶段 1 仍不进。** 剩下的硬缺口只有一个但很实在：
**232/239 条的预算参数是"由档名推出来的"，不是产物记录的**。
在 runner 里这不算问题（新产物会把实际参数写进 JSON），
但用它做历史对照时，"同档名"不等于"同预算"——`target_accept` 就是现成的反例。

---

## 6. 复算

```bash
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-13/map_stage0_records.py
```

脚本内置断言（不通过直接退出，不产出半成品）：
`record_id` 唯一；同一 `candidate_id` 必须对应同一份六字段定义（这条正是 AZ31 撞车的防线）；
`identity_status ∈ IDENTITY_STATES`；`init_profile=unknown` ⇒ id 前缀 `U-` 且不可比较。
