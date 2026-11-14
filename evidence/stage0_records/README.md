# stage0_records —— 历史产物的机器可读映射（v4）

生成：`python docs/round_2026-11-13/map_stage0_records.py --freeze`
自查：`python docs/round_2026-11-13/map_stage0_records.py --check`
漂移测试：`python docs/round_2026-11-13/test_stage0_freeze.py`

产物：`records.jsonl`（**239 行**）、`coverage.json`、`MANIFEST.json`

| 文件 | 说明 |
|---|---|
| `records.jsonl` | 一条一行，`json.dumps(allow_nan=False)`；每条含 `source`（路径 + sha256 + 生产者 + 原始目录）、`candidate`、`observation`、`provenance`、`unmapped` |
| `coverage.json` | 记录数、独立/严格/重复三态、执行身份统计、六项字段的覆盖与来源、未映射字段计数 |
| `MANIFEST.json` | 三份文件的 sha256/字节数、`registry_version`/`registry_digest`、9 份原始证据与 **63 份原始谱**的 `path → sha256` |

> 这个文件由脚本生成，不手写。`--check` 会**重算整个快照**并逐项比对（四层：
> 重算文本 vs 冻结字节 / 清单 `files` vs 磁盘 / 注册表指纹 / 原始证据与原始谱的 sha256），
> 任一层不一致都返回非零。

## 读法

- `observation` 是 `circuit_registry.observation_block()` 的**原样输出**，不是另写一套结构。
  三块分开：`run_status`（跑完没有）/ `sampling_status`（后验过没过）/
  `fit_observation`（残差 + `interpretation` + `unavailable`）。
- 六项描述块各自回答一个问题，**不能互相替代**：`param_estimate`（离真值多远）/
  `param_stability`（后验稳不稳）/ `ratio`（残差是噪声底的几倍）/ `noise_floor`（底从哪来）/
  `predictive`（ELPD 能否当样本外证据）/ `budget_params`（实际算力）。
- `observation.execution` 是**执行身份**：`run_id` + `run_id_status` + `data` + `data_id_kind`
  + `seed` + `seed_source` + `budget_source` + `software`。`R-` = 四项都由产物记录，
  `D-` = 至少一项靠代码/档名推出，`None` = 缺要素不发号（带 `unavailable_reason`）。
  **本批 239 条全是 `D-`**：产物记了 seed（6 条）与预算（6 条）、数据也都有内容哈希，
  但没有一份记代码版本。
- `provenance` 记**独立性与去重依据**：`duplicate_status`（`distinct` /
  `possible_duplicate` / `confirmed_duplicate`）、`duplicate_kind`、`independent`
  （派生值）、`duplicate_of`、`duplicate_evidence`（逐项列出比对了什么）。
- `unmapped` 是 schema **装不下**的原始字段，每条都写了原因。

## 三条必须知道的口径

1. **摘要相同 ≠ 重复；确认重复要求执行身份是 `recorded`**。`duplicate_status` 只看来源
   （产物 path/sha、seed 及证据等级、预算数值及来源、代码版本、数据内容哈希、`run_id`）。
   本批 **0 条 `confirmed_duplicate`、2 条 `possible_duplicate`**：那两对的 seed 与预算
   都由产物记录、指标逐位相同，但**代码版本未记录**，证明不了"同一软件条件下的同一次执行"。
   因此 `n_independent=239`（不扣），另给 `n_independent_strict=237`（扣掉）作对照。
   **两个数要一起报**：只给 239 会把疑似重跑算成新信息，只给 237 是把猜当结论。
2. `nested.json` 的 `status` 字段恒为 `"exploratory"`，是**成熟度标签**，不是执行状态。
   执行状态由 `metrics` + `diagnostics` 是否存在判断。原文保留在 `unmapped.status`。
3. `passivation__diag_truth_init.json` 是 **oracle 初值**（脚本直接读文件名里的真值参数），
   生产环境拿不到，`provenance.production_reachable=false`。它能证明"拓扑表达得了"，
   不能证明"这个候选能拟合"。它的预算数值由 `diag_passivation_init.py:57-60` 推出
   （`BUDGETS["standard"]` + `target_accept=0.98`），不是产物记录。

## 身份

`candidate.identity_status` 三档，本批**全部是 `derived`**：

| 等级 | 含义 | 本批 |
|---|---|---|
| `recorded` | 产物自身写明六项 | **0** |
| `derived` | 缺字段，但生成脚本唯一且代码路径可读 | **239** |
| `unconfirmed` | 查不到 → `candidate_id` 前缀 `U-`，禁进 manifest | 0 |

判定依据写在 `candidate.init_note` 里，例如
"`nested_fit.py:339` 未传 `init_values` ⇒ numpyro `init_to_uniform`"。

**v2 的错**（v3 改回）：把 pilot 的 `voigt_prod`/`voigt_fix2` 标成 `data_driven`。
这两个调用点传 `init=None`，走 `init_to_uniform`；`voigt_init` 只被 `sensitivity` 段用。
v2 是读了函数 docstring 判身份——函数文档描述的是它自己，不是谁在调用它。

**v4 的数据身份**：`execution.data.sha256` 是原始谱文件的**内容**哈希，`ref` 只是文件名。
本批 239/239 都是 `content_sha256`（63 份唯一原始谱）。文件名不是身份——两条不同实验的谱
都可能叫 `spectrum.csv`；那种情况只能标 `ref_only`，最高到 `derived`，且不能参与确认重复。

## 为什么这个目录入库

`out/` 被忽略，映射结果若不入库，仓库外无法检验"schema 到底能不能承载历史证据"。
本目录是冻结副本；重跑映射脚本会覆盖 `out/`，只有 `--freeze` 才会改动这里。

源头 `evidence/stage0_three_spectra/` 与 `evidence/coating_voigt_pilot/` 的字节级副本
已在库内，每条记录的 `source.sha256` 可直接用 `git cat-file blob` 复核。

`MANIFEST.data_sha256_files` 里的原始谱（`D:/eis训练数据/测试数据/...`、`out/_m6_samples/...`）
**不在库内**，只记路径与哈希：它们随机器而变，属预期；`--check` 在这些文件缺失时会报出来，
不会静默跳过。
