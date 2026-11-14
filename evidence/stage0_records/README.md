# stage0_records —— 历史产物的机器可读映射（v2）

生成：`python docs/round_2026-11-13/map_stage0_records.py`
产物：`records.jsonl`（**239 行**，其中独立 **237**）、`coverage.json`

| 文件 | 说明 |
|---|---|
| `records.jsonl` | 一条一行，`json.dumps(allow_nan=False)`；每条含 `source`（路径 + sha256 + 生产者 + 原始目录）、`candidate`、`observation`、`provenance`、`unmapped` |
| `coverage.json` | 记录数、独立/重复、执行/采样状态、身份等级、六项字段的覆盖与来源、未映射字段计数 |

## 读法

- `observation` 是 `circuit_registry.observation_block()` 的**原样输出**，不是另写一套结构。
  三块分开：`run_status`（跑完没有）/ `sampling_status`（后验过没过）/
  `fit_observation`（残差 + `interpretation` + `unavailable`）。
- 六项描述块各自回答一个问题，**不能互相替代**：`param_estimate`（离真值多远）/
  `param_stability`（后验稳不稳）/ `ratio`（残差是噪声底的几倍）/ `noise_floor`（底从哪来）/
  `predictive`（ELPD 能否当样本外证据）/ `budget_params`（实际算力）。
- `unmapped` 是 schema **装不下**的原始字段，每条都写了原因。
- `provenance.independent`：`false` 表示另一条记录的**同配置同 seed 重跑**
  （指标逐位相同），计数时不能重复算。

## 身份（v2 与 v1 的关键差别）

`candidate.identity_status` 三档，本批**全部是 `derived`**：

| 等级 | 含义 | 本批 |
|---|---|---|
| `recorded` | 产物自身写明六项 | **0**（没有一份 `nested.json` 记录 init） |
| `derived` | 缺字段，但生成脚本唯一且代码路径可读 | **239** |
| `unconfirmed` | 查不到 → `candidate_id` 前缀 `U-`，禁进 manifest | 0 |

判定依据写在 `candidate.init_note` 里，例如
"`nested_fit.py:339` 未传 `init_values` ⇒ numpyro `init_to_uniform`"。

**v1 的错误**：用 `p.parent.name` 查目录名表，但归档文件是平铺的，一次都没命中，
于是回落成真实档名 `"uniform"`，13 条记录拿到看起来合法、实际无依据的 id；
其中 2 条 AZ31 的 voigt 行还被算成了 nested，`candidate_id` 与真实 nested 行撞车。
详见 `docs/round_2026-11-13/STAGE0_RECORDS_MAP.md` §0。

## 两条必须知道的口径

1. `nested.json` 的 `status` 字段恒为 `"exploratory"`，是**成熟度标签**，不是执行状态。
   执行状态由 `metrics` + `diagnostics` 是否存在判断。原文保留在 `unmapped.status`。
2. `passivation__diag_truth_init.json` 是 **oracle 初值**（脚本直接读文件名里的真值参数），
   生产环境拿不到，`provenance.production_reachable=false`。它能证明"拓扑表达得了"，
   不能证明"这个候选能拟合"。

## 为什么这个目录入库

`out/` 被忽略，映射结果若不入库，仓库外无法检验"schema 到底能不能承载历史证据"。
本目录是冻结副本；重跑映射脚本会覆盖 `out/`，不会改动这里。

源头 `evidence/stage0_three_spectra/` 与 `evidence/coating_voigt_pilot/` 的字节级副本
已在库内，每条记录的 `source.sha256` 可直接用 `git cat-file blob` 复核。
