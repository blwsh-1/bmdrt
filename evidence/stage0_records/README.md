# stage0_records —— 历史产物的机器可读映射

生成：`python docs/round_2026-11-13/map_stage0_records.py`
产物：`records.jsonl`（239 行）、`coverage.json`

| 文件 | 说明 |
|---|---|
| `records.jsonl` | 一条一行，`json.dumps(allow_nan=False)`；每条含 `source`（路径 + sha256 + 生产者）、`candidate`、`observation`、`unmapped` |
| `coverage.json` | 记录数、执行/采样状态分布、候选身份完整率、旧 schema 装不下的条数、未映射字段计数 |

## 读法

- `observation` 是 `circuit_registry.observation_block()` 的**原样输出**，不是另写一套结构。
  三块分开：`run_status`（跑完没有）/ `sampling_status`（后验过没过）/
  `fit_observation`（残差 + `interpretation` + `unavailable`）。
- `unmapped` 是 schema **装不下**的原始字段，每条都写了原因。不要把它当"次要字段"：
  `param_err`、`ratio`、`floor` 这三样是本项目最常用的量，它们装不进去就是 schema 的缺口。
- `candidate.identity_status`：
  - `complete` = 字段齐、`candidate_id` 可复现（226 条，全部来自 pilot）；
  - `inferred` = 缺 `init_profile`，靠目录名或默认值推断（13 条，来自 `nested.json` 与 AZ31）。
    **这些 `candidate_id` 不能当作可复现身份。**

## 两条必须知道的口径

1. `nested.json` 的 `status` 字段恒为 `"exploratory"`，是**成熟度标签**，不是执行状态。
   执行状态由 `metrics` + `diagnostics` 是否存在判断。原文保留在 `unmapped.status`。
2. `passivation__diag_truth_init.json` 是 **oracle 初值**（脚本直接读文件名里的真值参数），
   生产环境拿不到。它能证明"拓扑表达得了"，不能证明"这个候选能拟合"。

## 为什么这个目录入库

`out/` 被忽略，映射结果若不入库，仓库外无法检验"schema 到底能不能承载历史证据"。
本目录是冻结副本；重跑映射脚本会覆盖 `out/`，不会改动这里。

源头 `evidence/stage0_three_spectra/` 与 `evidence/coating_voigt_pilot/` 的字节级副本
已在库内，每条记录的 `source.sha256` 可直接用 `git cat-file blob` 复核。
