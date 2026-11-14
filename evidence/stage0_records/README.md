# stage0_records —— 历史产物的机器可读映射（v3）

生成：`python docs/round_2026-11-13/map_stage0_records.py --freeze`
自查：`python docs/round_2026-11-13/map_stage0_records.py --check`

产物：`records.jsonl`（**239 行**）、`coverage.json`、`MANIFEST.json`

| 文件 | 说明 |
|---|---|
| `records.jsonl` | 一条一行，`json.dumps(allow_nan=False)`；每条含 `source`（路径 + sha256 + 生产者 + 原始目录）、`candidate`、`observation`、`provenance`、`unmapped` |
| `coverage.json` | 记录数、独立/重复三态、执行身份统计、六项字段的覆盖与来源、未映射字段计数 |
| `MANIFEST.json` | 两个文件的 sha256/字节数、`registry_version`/`registry_digest`、9 份原始证据的 `path → sha256` |

> **上一轮这个目录没有 `MANIFEST.json`**（只有下面三个文件里的前两个 + README）。
> 报告里提到过它，但没有写入。现在由脚本生成，不手写；`--check` 会重算并逐字节比对。

## 读法

- `observation` 是 `circuit_registry.observation_block()` 的**原样输出**，不是另写一套结构。
  三块分开：`run_status`（跑完没有）/ `sampling_status`（后验过没过）/
  `fit_observation`（残差 + `interpretation` + `unavailable`）。
- 六项描述块各自回答一个问题，**不能互相替代**：`param_estimate`（离真值多远）/
  `param_stability`（后验稳不稳）/ `ratio`（残差是噪声底的几倍）/ `noise_floor`（底从哪来）/
  `predictive`（ELPD 能否当样本外证据）/ `budget_params`（实际算力）。
- `observation.execution` 是**执行身份**：`run_id` + `run_id_status` + `data` + `seed`
  + `seed_source` + `budget_source` + `software`。`R-` = 三者都由产物记录，
  `D-` = 至少一项靠代码/档名推出，`None` = 缺要素不发号（带 `unavailable_reason`）。
  **本批 239 条全是 `D-`**：产物记了 seed（6 条）或预算（6 条），但没有一份记代码版本。
- `provenance` 记**独立性与去重依据**：`duplicate_status`（`distinct` /
  `possible_duplicate` / `confirmed_duplicate`）、`duplicate_kind`、`independent`
  （派生值）、`duplicate_of`、`duplicate_evidence`（逐项列出比对了什么）。
- `unmapped` 是 schema **装不下**的原始字段，每条都写了原因。

## 三条必须知道的口径

1. **摘要相同 ≠ 重复**。`duplicate_status` 只看来源（产物 path/sha、seed 及证据等级、
   预算数值及来源、`run_id`）。本批 2 条 `confirmed_duplicate` 的依据是
   `config.seed` 与 `budget_kw` **都由产物自己记录**、且指标逐位相同。
   证据不足时只标 `possible_duplicate`，`coverage.json` 的 `n_independent` **不扣**，
   另给 `n_independent_strict` 作对照（本批两者都是 237）。
2. `nested.json` 的 `status` 字段恒为 `"exploratory"`，是**成熟度标签**，不是执行状态。
   执行状态由 `metrics` + `diagnostics` 是否存在判断。原文保留在 `unmapped.status`。
3. `passivation__diag_truth_init.json` 是 **oracle 初值**（脚本直接读文件名里的真值参数），
   生产环境拿不到，`provenance.production_reachable=false`。它能证明"拓扑表达得了"，
   不能证明"这个候选能拟合"。它的预算数值由 `diag_passivation_init.py:57-60` 推出
   （`BUDGETS["standard"]` + `target_accept=0.98`），不是产物记录。

## 身份（v3 与 v2 的差别）

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

## 为什么这个目录入库

`out/` 被忽略，映射结果若不入库，仓库外无法检验"schema 到底能不能承载历史证据"。
本目录是冻结副本；重跑映射脚本会覆盖 `out/`，只有 `--freeze` 才会改动这里。

源头 `evidence/stage0_three_spectra/` 与 `evidence/coating_voigt_pilot/` 的字节级副本
已在库内，每条记录的 `source.sha256` 可直接用 `git cat-file blob` 复核。
