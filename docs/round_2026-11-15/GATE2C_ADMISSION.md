# Gate 2C 正式准入裁决

> **基线**：`0008635`（Gate 2B：F-01/F-02 实现与执行链测试）
> **范围**：按用户既定的 11 项准入表 + 另 7 项身份问题逐项取证，给出明确结论。
> **性质**：准入裁决，**不是**新一轮功能修改、**不是**代码审查报告。
> **本轮未做**：未启动 runner、未跑完整 EIS 拟合、未产生新的实验拟合结果。

---

## 0. 结论

> ### **CONDITIONAL —— 仅对「验证留证链」这个目标有条件准入。**
>
> **若目标是「产出可信、有效的 `run_id`」，当前应判 `FAIL`。**
> 二者不可混为一谈：预算五项来源证据仍未闭合，真实执行仍是
> `unconfirmed`、两号 `None`。CONDITIONAL 只授权第一版 runner 去验证
> 「执行 → 捕获 → 组装 → 落盘 → 重读」这条链是否完整，**不授权把零号记录
> 当成有效身份样本**。

| 汇总 | 数量 | 项 |
|---|---|---|
| `PASS`（有实测） | 8 / 11 | 1、2、3、4、9、10、11 +（4 带注记） |
| `DESIGN`（设计具备、待真实 runner 实测） | 2 / 11 | 5（部分）、6 |
| `PARTIAL`（限冻结路径与受控工况） | 1 / 11 | 8 |
| **`NOT MET`**（准入硬伤） | **1 / 11** | **7 预算身份** |

另 7 项：`PASS` 5（a/b/c/f/g）、`DESIGN` 2（d/e）。

---

## 1. 版本裁决（上一轮遗留待裁项）

**裁决：保持 `IDENTITY_CONTRACT_VERSION = "1.5"`、`RUN_ID_CONFIG_KEYS_VERSION = 2`，不升 1.6。**

取证（载入 `e19fe4b` 旧实现独立比对）：

| 事实 | 结果 |
|---|---|
| `run_id_payload` 键集 / 内容 | **逐位相同** |
| `workflow_payload` 键集 / 内容 | **逐位相同** |
| `run_id_digest` | `3bff356af1383db1ff14745009d1a914…`，**相同** |
| `workflow_id_digest` | 相同 |
| 对照 1.4→1.5 | 载荷真变了：加 `config_keys_version`/`rerun_attempt`/`rerun_lineage`、移除 `retry_kw` ⇒ 按 D-4 全换号合理 |

依据：

1. D-4 原文是「进 `run_id` 哈希的契约世代标签」+「**载荷语义变了** ⇒ 必须换号」。
   本轮只改了 `evaluate()` 的**准入门槛**（哪些记录能发号），没改号怎么算，
   载荷语义未变。升 1.6 只会制造一次纯前缀替换的表面换号，**零区分力**。
2. 重蹈 F-11 的教训：`keys_version=2` 但 38 键未变，已经是一次无意义升版本；
   不为「看起来变了」再 +1。
3. 门槛世代**已有更强载体**：`attestation.version = "1"` 在凭证内部、
   被 HMAC 签名覆盖、由 `_chk_attestation()` 校验，改动即零号。
   签名覆盖的可靠性高于一个不进载荷的顶层版本字段。
4. 全换号的代价（239 条历史对照、世代表、前缀断言、清单、审查脚本全部重做）
   换不来任何语义收益。

**落地**：`exec_identity.py` 常量区写明两层版本分工；新增
`test_execution_chain.case_version_layer_separation()` 回归锁（15/15）：
门槛版本被签名覆盖、改版本零号、两个载荷不含 attestation/instance、
载荷键集固定、号仍为 `R2-`。**将来载荷形状一变，该测试会失败，提示按 D-4 升世代。**

---

## 2. 准入表 11 项

| # | 准入项目 | 判定 | 取证依据 |
|---|---|---|---|
| 1 | 身份契约版本 | **PASS** | `_require_version()` 同时校 `identity_contract_version` 与 `run_id_config_keys_version`；两个载荷含 `config_keys_version`；G5①② 35/35；§1 裁决不升世代 |
| 2 | `run_id` 区分执行条件 | **PASS** | 实测改 `data.sha256` / `budget.n_warmup.value` / `config.layers` 三组均换号；68/68 的 T 组 |
| 3 | `workflow_id` 区分工作流 | **PASS** | 实测改 `workflow.search` / `children` / `retry` 三组均换号；F-06 已拦住整块字段二次承载预算 |
| 4 | 数据身份 | **PASS**（带注记） | `_data_payload` 六键（`sha256`/`n_points`/`subset_sha256`/`f_min`/`f_max`/`transform`）、`id_kind=content_sha256`、`_chk_data` 合规。**注记**：处理后数据无独立内容哈希，靠 `subset_sha256` + `transform` 参数表达，不冒充内容身份 |
| 5 | 源码身份 | **DESIGN / 部分** | `source_manifest.json` A/B/C 三 profile、`--check rc=0`、`runtime_loaded_modules` 31/31。**缺口 G9**：清单是声明集/角色，**不含文件内容哈希** ⇒ 改算号规则不换号 |
| 6 | dirty 状态 | **DESIGN** | `patch_bundle.verify()` / `patch_bundle_io.rebuild()` 分两文件、25/25；干净树 19/19 base 重建逐位相同、脏克隆 19+3+2。**未实测**：真实 runner 下的 dirty patch；且「完整性≠真实性」、Windows mode `skipped` |
| 7 | **预算身份（五项真实来源证据）** | **⛔ NOT MET** | 真实捕获事件 `upstream_origin_captured=False`、五项 `source=unavailable`、`value=None`；S1/S2/S6 来源链未闭合。**这是准入硬伤，不因 CONDITIONAL 而豁免** |
| 8 | 采样捕获最终执行参数 | **PARTIAL** | 两站点真实构造点已接 `capture_at_construction()`；但三处旧挂账不动（`bm_real.py:1172` 读 kw、`tree_depth_stats` 用声明值、`fit_top` 缺 `extra_fields`），且 C8 探针证实「转发层 2000→16 ⇒ 声明记录≠执行条件」。结论**仅限冻结路径与受控工况**，不外推 |
| 9 | 重跑身份 | **PASS** | `rerun_attempt` + `rerun_lineage` 进载荷；`make_rerun_record()` 不继承父凭证；F-08/F-09 反例 35/35 + 14/14 |
| 10 | 历史兼容 | **PASS** | Stage 0 239 条 / 独立 239 / 严格 237，`--check rc=0`；v1.4 记录喂 v1.5 `evaluate()` ⇒ 零号、按旧契约解释；registry 世代 `R-` 独立不跨代比号 |
| 11 | 负面门禁 | **PASS** | 35/35 + 15/15；去掉受控签发凭证即 `attestation_missing` 零号；伪造证据、跨运行复用、缺证据、落盘失败全部两号 `None` |

---

## 3. 另 7 项身份问题（用户点名不可跳过）

| 项 | 判定 | 依据 |
|---|---|---|
| a. `compute_role` 与 `workflow_scope` 分离 | **PASS** | `compute_role` 定义在 `capture_contract.harness_item`、写进 manifest profiles；两者都不进 `run_id_payload` |
| b. `numeric_source_tree_sha256` 覆盖数值计算代码 | **PASS** | 载荷 `code` = `{git_commit, numeric_source_tree_sha256}`，`_chk_code()` 强制要求 |
| c. `workflow_source_tree_sha256` 覆盖决策与工作流 | **PASS** | `workflow_payload` 含该字段 |
| d. `runtime_loaded_modules` 满足声明集约束 | **DESIGN** | `classify_runtime()` / `harness_item()` 齐全、C-1/C-2/C-3 契约测试 31/31；真实 runner 下未实测 |
| e. dirty patch bundle 完整 | **DESIGN** | `source ∈ {base,patch,blob}`、`verify`/`rebuild` 两段不合并、25/25；真实性未证（同 #6） |
| f. 环境摘要与源码身份分离 | **PASS** | `env` 与 `code` 是载荷里两个独立顶层键；`env = {env_sha256, backend, x64_enabled}`，`_env_sha256()` 独立计算 |
| g. sigma 按固定 dtype/形状/顺序规范化 | **PASS** | `sigma_block` 复合块（`array_sha256` + `floor` 为唯一载体）、`SIGMA_MEMBER_KEYS` 退出哈希、`sigma_run_id_value()` 可调用 |

---

## 4. CONDITIONAL 的边界（必须逐条遵守）

**允许**

- 仅在**唯一冻结路径** `bm_core.fit` 上；`bm_top.fit_top` 只作第二站点兼容验证。
- 单次、少量、预先冻结的候选集 / 数据集 / 预算 / 软件身份 / 执行计划。
- 第一版 runner 只做**执行与留证**：读冻结清单 → 校验身份 → 创建记录 → 调用现有拟合入口 → 捕获 → 保存结果/证据 → 更新 workflow 状态。
- **接受 `unconfirmed` 零号**：runner 必须能如实落盘「本次无有效身份」，不得为让号非空而放宽门禁。

**不允许**

- 批量拟合；自动选最佳拓扑 / 改 K / 调采样预算；自动重跑失败模型；接管模型评价与物理解释。
- 把零号记录写成有效身份、或当成新版身份闭环样本；外推到 nested / K3 / `bm_select` retry / 其余入口。
- 修改 `bm_core.py` / `bm_top.py` 的拟合算法；三处旧挂账不动；239 条历史记录保持原证据等级。
- 把 `unavailable` 升级成 `recorded`，或用推断来源补历史缺口。

**待 T1/T2/T3 实测才能关闭**（不得写成 PASS）

dirty patch 真实性、三时点快照（`before_fit` / `sampler_entry` / `after_fit`）、
harness 归属（`runtime_scope`）、多次采样逐次事件、失败/中断/重启/重复提交的恢复行为。

---

## 5. 仍未闭合（如实明账）

- **预算来源 S1/S2/S6 未闭合** ⇒ 真实执行零号；F-01 仍只是**条件性关闭**
  （内部可信边界 + 落盘可审计，**不是抗抵赖密码学证明**）。
- G9（清单不含内容哈希）、F-10–F-13（两套世代表 / `keys_version=2` 但 38 键未变 /
  前缀与世代耦合 / 残留 1.4 标签）、P1×5 / P2×1、契约正文 C1–C8 订正，均未处理。
- 算法主线验收（`§13` 留出 0/3、`§14` 全量 4/14，两次卡 M-b `acf≤0.50`）**仍未过线**；
  本阶段所有工作都在执行身份链上，未回到算法验收。

---

## 6. 下一步（各自仍需单独授权）

1. **Gate 3 最小 runner** —— 按 §4 边界实现，只执行与留证。
2. **T1 / T2 / T3 试运行**：单次执行 / 重复执行 / 失败与恢复；关闭 §4 的待实测项。
3. **真实 EIS 单次拟合** —— 待身份记录能可靠落盘、重读、验证后单独授权。

> ⛔ 本轮**未宣布无条件准入、未启动 runner、未跑完整 EIS 拟合、未修改任何拟合算法**。
> 本文件只改文档与一条回归锁测试；`8015721` 三个被锁 blob 逐位 SAME。
