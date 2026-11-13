# 阶段 0：历史产物的结构化映射（机器可读）

状态：**映射已完成，承载力部分通过**。本步不跑拟合、不改原始产物、不定阈值。

> 这一步是审计要求的第 5 项："18 个证据文件入库 ≠ 18 条结构化记录已完成"。
> 本文件与 `records.jsonl` 才是"registry 能不能承载历史证据"的实际检验结果。
> 结论先说：**能承载残差与采样状态，承载不了候选身份与点估计误差。**

## 1. 产物

| 文件 | 内容 |
|---|---|
| `docs/round_2026-11-13/map_stage0_records.py` | 映射脚本，只读 `evidence/`，不跑拟合 |
| `evidence/stage0_records/records.jsonl` | **239 条**记录，一行一条，`allow_nan=False` |
| `evidence/stage0_records/coverage.json` | 字段覆盖率与"旧 schema 装不下"的计数 |

复现：`python docs/round_2026-11-13/map_stage0_records.py`

## 2. 记录构成

| 来源 | 条数 | 说明 |
|---|---|---|
| `*.nested.json` | 6 | nested_fit 产物，换元件 / 换预算 / 换初始化 |
| `passivation__diag_truth_init.json` | 1 | **oracle 初值**诊断，不是生产可达解 |
| `coating_voigt_pilot/results.jsonl` | 226 | 60 谱 × 3 套（voigt 自由 σ / voigt 固定 2% / nested）+ 46 条敏感性 |
| `az31__m4_real_nested_az31.json` | 6 | 真实谱，6 个配置并列，无真值 |
| 合计 | **239** | |

## 3. 三条硬结论

### 3.1 旧 schema 会丢掉 70 条真实记录

| 指标 | 值 |
|---|---|
| `run_status=completed` | 239 / 239 |
| `sampling_status=passed` | 168 |
| `sampling_status=failed` | **71** |
| **"跑完、残差算了、后验没过"** | **70** |

这 70 条正是 0.1.0 的 `converged=False → 不许报 rms_rel/acf` 会直接拒绝入库的记录。
它们占全部记录的 29%，不是边缘情况。最典型的：涂层 `nested_parallel` 是
`rms_rel 10.81% / acf 0.944 / R̂ 14.21`；钝化 P1 默认初值是
`98.40% / 0.750 / R̂ 197.79`；pilot 里 `voigt_prod` 60/60 都记了 rms，
即使其中 46 条 R̂ > 1.05。

⇒ **P0-1 不是理论缺陷，是 70 条真实记录装不进去。** 现在的 `descriptive_only`
解读标记就是为它们存在的，且**必须显式给**，不给就报错。

### 3.2 候选身份不可复现：13 / 239

| 项 | 值 |
|---|---|
| `identity_status=complete`（字段齐、可复现 `candidate_id`） | 226 |
| `identity_status=inferred`（缺字段，靠目录名/默认值推断） | **13** |

根因只有一个：**`nested.json` 的 `config` 里没有 `init_profile`**。
`config` 只有 `branch / layers / budget / budget_kw / sigma_* / seed`。
而 `candidate_id` 覆盖 `init_profile`（这是对的——钝化谱的默认初值与数据驱动初值
结论完全不同），所以历史产物的身份只能靠**目录名**（`nested_fit_cap_pass` → 默认初值，
`_user_fit_nested` → `data_init`）推断。

映射里如实标成 `init_profile_source: "inferred_from_dirname" | "default_assumed"`，
并在记录里写明 `candidate_id` 因此**不是可复现身份**。没有替它补一个"看起来合理"的值。

⇒ 阶段 1 的 runner 必须把 `init_profile` 写进产物。这不是 schema 能补的，
是**产物侧缺字段**。

### 3.3 `status` 撞名：产物用 `status` 表示成熟度

`nested.json` 的 `status` 字段值**恒为 `"exploratory"`**（6/6），意思是"研究性接口"，
不是"拟合成功/失败"。而 registry 用 `run_status` 表达执行状态。

直接映射会把 6 条全部判成 `failed`（第一次跑就撞上了，报错见下）。正确做法是
**由 `metrics` + `diagnostics` 是否存在判断**，`status` 原文进 `unmapped` 并注明撞名。

⇒ 这类同词异义是"名称统一了、模型却没法比较"的另一种形态。阶段 1 的产物字段名
要么统一，要么显式加前缀（`maturity` vs `run_status`）。

## 4. schema 装不下、已进 `unmapped` 的字段

| 字段 | 条数 | 为什么装不下 |
|---|---|---|
| `param_err` / `param_max_err` | 226 | 是**点估计相对误差**，不是后验区间；`param_stability` 只收 `ci90`/`flag` 描述块。强行塞会把它伪装成"参数可辨识性证据" |
| `ratio` | 226 | schema 无该字段，更缺 `n_used`/`n_total`（收敛子集 3.36 vs 全样本 6.47 就是分母不同） |
| `floor` | 226 | 噪声底来源未进 schema；而"贴底"是本项目最核心的判据之一 |
| `verdict` | 226 | 预注册判定属证据层，观测记录里没有位置（合理，但需要一条从观测到判定的链路） |
| `elpd` | 12 | schema 无证据用量字段（`nested.json` 里有 `elpd_valid_for_predictive_comparison` 这个**好设计**，但没进 schema） |
| `rel_median` / `rel_max` / `rel_med` | 12 | 与 `rms_rel` 不同口径，不能互相替代（0C 曾把 2.40% 当成 rms_rel，已更正） |
| `budget_kw` / `target_accept_override` | 6+6 | `budget_profile` 只是标签，未结构化保存预算参数；`None` 不等于"走生产 0.9"（实际 0.98） |
| `sigma_mode` | 6 | 已折进 `likelihood_profile` 标签，原始字段未单列 |
| `status` / `warning` | 6+6 | 撞名 + 无警示文本字段 |
| `derived.L_ad_H` | 6 | capacitive 下 `null` = **不适用**，与"没算"不同语义 |
| `elapsed_s` / `seconds` | 7 | 无成本字段 |
| `median_vs_truth` | 1 | 参数恢复对照，schema 无该结构 |

## 5. 承载力判定

| 能力 | 判定 | 依据 |
|---|---|---|
| 记"跑完但采样没过"并保留残差 | ✅ 通过 | 70 条全部入库，均带 `descriptive_only` |
| 区分 `not_run` / `failed` / `completed` | ✅ 通过 | 239 条 `run_status=completed`，AZ31 的"未跑"靠 `execution.not_run` 表达 |
| 记录后验区间（`params.ci90`） | ✅ 通过 | 6 条 `nested.json` 的 `params` 全带 `median/ci90/rhat/ess` |
| **复现候选身份** | ❌ **不通过** | 13 条缺 `init_profile`，产物侧未记录 |
| **承载点估计参数误差** | ❌ **不通过** | 226 条 `param_err` 无处安放 |
| **承载 `ratio` 及其分母** | ❌ **不通过** | 226 条无 `n_used`/`n_total` |
| 承载噪声底来源 | ❌ 不通过 | 226 条 `floor` 只有值，没有来源标记 |

⇒ **阶段 1 暂不进入**：上表四项不通过里，后三项都要动 schema，第一项要动产物侧。
按审计的顺序，先补字段、再谈 runner。

## 6. 下一步要补的 schema 字段（下一轮，需批）

1. `param_estimate`（点估计 + 相对误差 + 参照来源），与 `param_stability`（后验区间）
   **分开两个字段**——它们回答的问题不同："估到了多少"vs"这个估计稳不稳"。
2. `ratio` 结构：`{value, n_used, n_total, denominator: "converged_subset"|"all_rows"}`。
3. `noise_floor`：`{value, source: "truth_forward"|"unknown", topology_ref}`。
4. `predictive`：`{elpd, se, valid_for_comparison, reason}`（照抄 `nested.json` 现成的好设计）。
5. `budget_params`：`{n_warmup, n_samples, num_chains, target_accept, max_tree_depth, effective_target_accept}`。
6. 产物侧：`nested_fit.py` 的 `config` 增加 `init_profile`（缺这个，任何 `candidate_id` 都不可复现）。

以上**本轮一律未实现**——它们会改变 `registry_digest`，需要与审计结论一起决定，
不适合在映射脚本里顺手加。
