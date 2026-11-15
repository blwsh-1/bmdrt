# 真实 EIS 单次拟合：运行记录

> 用户「可以」＝ 单独授权上一轮末尾提出的真实 EIS 单次拟合。基线 `10c7ad6`。
> 本文是**运行记录（非审查报告）**；范围严格限于 `GATE2C_ADMISSION.md` §4 CONDITIONAL。

## 0. 结论速览

**真的跑了一次完整采样**：`bm_core.fit` + `MCMC.run`，冻结数据
`archive/2EMC1-1_0005mA.csv`（56 点）、冻结试运行预算 20/20、单链、单次。

```
status = succeeded        fit_ok = True        elapsed_s = 8.94
n_draws = 20              posterior_sha256 = 2efb4c8ca16f22c8f917…
three_points = True       numeric_digest_same = True
sampling_events = 1 (complete=1)
identity_status = unconfirmed   run_id = None   workflow_id = None
```

**最重要的一句：真实拟合成功 ≠ 身份闭环。** 五项预算仍 `unavailable`、
`upstream_origin_captured=False`，`evaluate()` 如实给 `unconfirmed` 零号。
本轮证明的是「真实执行也能被完整留证」，不是「现在能拿到有效 `run_id`」。

## 1. 冻结范围

| 项 | 值 |
| --- | --- |
| 主路径 | `bm_core.fit`（`bm_core.py:610`），唯一；站点替换直接拒 |
| 候选 | `voigt:hn2+rl_parallel`（冻结清单唯一项） |
| 数据 | `archive/2EMC1-1_0005mA.csv`，sha `560ba5f2…f17030`，56 点 |
| 拟合预算 | `K=10, n_warmup=20, n_samples=20, num_chains=1, ta=0.9, depth=10, seed=0`；**试运行小预算**，不是正式实验预算 |
| 额外字段 | `("diverging", "num_steps")` 透传，只记账不评价 |
| 禁止 | 批量、自动选拓扑/改 K/调预算、自动重跑、模型评价、改 `bm_*.py` 拟合算法、运行时改 `fit_cfg` |

新增 `real_fit_plan` 与 `fit_cfg` 进 `FROZEN_MANIFEST`，`--check` 校验两者与
dry-run 计划互斥、`n_fits=1`、预算有界；`run_job(fit_cfg=...)` / 换站点 /
未登记模式三道禁令反例全拒。

## 2. 补上的四项机制（不改生产代码）

1. **真实拟合**：`real_fit_runner.py` 调 `bm_core.fit(f, Z, **fit_cfg)`，真的跑
   `MCMC.run`；捕获事件取生产代码已挂的 `mc.provenance`（`site=bm_core.fit`、
   `judged=True`、适配无错），不另造一份。后验只落 `n_draws` / 参数均值 /
   固定 `<f8` 顺序的 `posterior_sha256` / `divergences` / `num_steps`，
   指纹可复算。
2. **三时点**：包装 `numpyro.infer.MCMC.run` 类属性取 `sampler_entry`，
   `before_fit` / `after_fit` 在调用前后读 `sys.modules`；路径一律过
   `capture_contract.normalize_path`。三个点的项目侧文件表 + 状态计数 +
   sha 落盘，可逐位复算。
3. **harness 归属**：6 个 runner / harness / 采集 / 身份 / 条件夹具文件用
   `harness_item` 显式声明为 `cli_only` / `reference_only`；`classify_runtime`
   处理全量运行时文件，不许按文件名过滤；核验
   `numeric_source_tree_sha256` 在「含 / 不含 harness」两个声明集下**逐位相同**。
4. **多次采样逐次事件**：`SamplingEventLog` 每次 `MCMC.run` 一条事件、
   显式 `session(owner)` 归属、复用前必须 `reset()`；`running` / `error`
   不算完整事件。单次拟合 ⇒ 1 条 `ok`；另用两次 tiny 拟合验证不覆盖，
   以及同一 owner 不 `reset()` 必报错。

拟合结果、三时点、harness 实测、逐次事件统一写 runner 侧
`execution_evidence`，**不进身份哈希**；条件模板 `code` 仍是已审夹具。

## 3. 本轮实测的三处新事实（如实明账）

1. **三时点不再逐位相同**：`before_fit` 指纹 `29ee1f7d01…`，
   `sampler_entry` / `after_fit` 为 `c1db13224a…`。真实拟合过程中项目侧
   模块集合确有变化，比阶段 C dry-run 的「三点逐位相同 ⇒ 三点齐 ≠ 采集深入」
   更有信息量；但「采到了差异」仍不等于「源码身份已闭环」。
2. **`source_coverage = undeclared_loaded`**：未声明模块
   `circuit_registry.py`（profile B 声明集不覆盖真实 `bm_core.fit` 路径）。
   这是 Gate 2C「`runtime_loaded_modules` 声明集约束」仍为 DESIGN 的
   **实测证据**，不伪装成 `complete`；按真实入口重建声明集仍需单独授权。
3. **记录体积**：首版把全量 `sys.modules`（千余条第三方路径）落进三时点，
   单条 721 KB；收紧为「项目侧文件表 + 状态计数」后 **24.3 KB**，
   三时点指纹仍可复算，第三方按契约只计数不落路径。

另：本次 20 draws 中 `divergences=10`、`num_steps_max=148`。试运行小预算
采样质量明显不足，runner **只记账、不下收敛或物理解释**（不接管模型评价）。

## 4. 验收测试

`test_real_fit_runner.py` **10/10**：R1 单次真实拟合与留证链、R2 后验与结果
指纹可复算、R3 三时点、R4 harness 归属、R5 逐次采样事件、R6 采样异常注入
（无后验结果、事件标 error、`failed` 不误报成功）、R7 冻结禁令、
R8 多次采样逐次事件不覆盖 + `reset()` 规则、R9 跨运行复用被拒、
R10 证据缺失仍零号。

另：dry-run `19/19`、执行链 `15/15`、`real_fit_runner` 自检 `5/5`。
全套回归全绿：68/68、45/45、35/35、31/31、25/25、45/45、28/28、27/27、
76/76；清单 `--check rc=0`、冻结 9/9、阶段 0 `--check`（239/239/严格 237）、
三探针 `rc=0`、审查脚本 51 条 FAIL 0、`8015721` 三 blob SAME、
`git diff --check rc=0`、新增/修改文件 `CRLF=0`。

改动面：修改 `minimal_runner.py`，新增 `real_fit_runner.py`、
`test_real_fit_runner.py`、`REAL_FIT_RUN.md`，同步 `docs/INDEX.md`。
**`bm_*.py` / 身份层 / 既有测试 / 源码清单 / 被锁 blob 零改动**。

## 5. 仍未闭合

- **身份仍零号**：S1/S2/S6 上游预算来源链未闭合；`rng_budget_mismatch`
  是条件模板 `n_chains=2` 与事件 `num_chains=None` 的组装副作用，非生产缺陷。
- **F-01 条件性关闭**（内部可信边界，非抗抵赖）；**dirty patch 真实性**、
  **`runtime_loaded_modules` 声明集**（本轮实测 `undeclared_loaded`）仍 DESIGN。
- G9（清单不含内容哈希）、F-10–F-13、P1×5/P2×1、契约正文 C1–C8 未处理；
  239 条历史记录保持原证据等级。
- 算法主线 `§13` 留出 0/3、`§14` 全量 4/14（两次卡 M-b `acf≤0.50`）
  **仍未过线**，本轮没有回到算法验收。

## 6. 下一步（各自仍需单独授权）

1. 若要让真实拟合**产出有效 `run_id`**：先闭合五项预算的真实来源
   （S1/S2/S6，不许把 `unavailable` 升 `recorded`），并按真实入口重建声明集
   消除 `undeclared_loaded`。
2. **回到算法主线验收**：`§13` / `§14` 的 M-b `acf≤0.50` 未过线是另一条主线，
   身份链不能替代它。
3. ⛔ **Gate 2C 维持 CONDITIONAL，不转 PASS**；未宣布无条件准入、未批量拟合、
   未改任何拟合算法。红线不变：不升级来源、不填历史缺口、三挂账不动、
   零号不得写成有效身份。
