# Gate 3 第一版最小 runner：运行记录

> 基线 `c3a9144`（Gate 2C 准入裁决 CONDITIONAL）。本文是**运行记录**，不是审查报告。
> 日期 2026-11-15。

## 0. 结论速览

按 `GATE2C_ADMISSION.md` §4 的 CONDITIONAL 边界，实现并**真实运行**了第一版最小 runner：
只执行与留证，不跑完整 EIS 拟合、不批量、不自动决策。

- 实际执行：`minimal_runner.py --run --out out/gate3_runner/final --job-key gate3-final`
  ⇒ `status=succeeded`、`reload_consistent=True`、记录原子落盘、workflow 状态行 append-only。
- 身份结果：**`unconfirmed`，`run_id=None`、`workflow_id=None`**。五项预算 `unavailable`、
  `upstream_origin_captured=False`。**这是来源未闭合时的正确结果**，runner 如实落盘
  「本次无有效身份」，没有为让号非空放宽任何门禁。
- 试运行验收：`test_minimal_runner.py` **19/19**（T1×4 / T2×3 / T3×7 / 冻结禁令×5）。
- 全套回归全绿；`bm_*.py` 零改动，身份层 / 既有测试 / 被锁 blob 均未改。
- ⛔ **Gate 2C 的 CONDITIONAL 判定不因此转 PASS**；未宣布无条件准入，未跑完整 EIS 拟合。

## 1. 冻结清单与唯一执行路径

`FROZEN_MANIFEST`（`minimal_runner.py` 内常量，`--check` 逐项校验、失败即 raise）：

| 项 | 冻结值 |
| --- | --- |
| 契约 | `identity_contract_version=1.5` / `run_id_config_keys_version=2` |
| 候选集 | 仅 `voigt:hn2+rl_parallel`（运行时增加候选直接拒） |
| 数据集 | `archive/2EMC1-1_0005mA.csv`，sha256 `560ba5f2…f17030`，56 点，实测比对 |
| dry-run 构造参数 | 1 链 / 8 warmup / 8 samples / ta 0.9 / depth 10（**不是身份层预算**） |
| 主路径 | `bm_core.fit`；`bm_top.fit_top` 仅兼容验证 |
| 排除 | nested 独立接口、K3 `m35_cond_n2.py:116`、`bm_select` retry、其余 CLI/程序化入口 |
| 执行计划 | `capture_only_dry_run`，`calls_mcmc_run=False`、`produces_fit_result=False`，禁自动重试/改 K/改预算/模型评价 |

**执行模式边界**：只调用生产捕获接口与真实 numpyro MCMC 构造点，**不调用 `MCMC.run()`、
不产生拟合结果**。Gate 3 工作流第 4 步「调用现有拟合入口」要等真实 EIS 单次拟合单独
授权后接入，本版不伪装成已调用。

**软件身份实测**（`software_identity` 块，runner 侧明账、**不进身份哈希**）：git commit、
`git status --porcelain` 实测 dirty 路径、Python / 平台、`source_manifest.json` 的 sha 与版本。
实测本次 `git_dirty=True`（新增文件未跟踪）；条件模板 `code` 里的 `git_dirty=False` 是
**夹具值**，两者不混同——源码身份仍属 Gate 2C 的 DESIGN 项，未闭环。

## 2. 七步工作流

1. 冻结清单校验（契约版本、候选、数据 sha/点数、清单版本、站点闭集、dry-run 声明）。
2. 数据与源码身份实测（数据 sha 重算、清单、dirty、环境）。
3. 创建记录：已审条件夹具 + 真实数据身份 + runner 侧明账。
4. 执行：真实 `capture_at_construction()`（不 run、不拟合）。
5. 捕获：事件三块 `budget_params` / `budget_evidence` / `budget_observations` 只搬运、不推断。
6. 保存：`check_reuse` → `finalize` → tmp + `os.replace` 原子落盘 → append-only 审计 → `load_and_verify` 重读复核。
7. 更新 `workflow_state.jsonl`（append-only）：`succeeded` / `failed` / `rejected` + attempt + 实例 token + 事件指纹 + 父/子引用。

**执行关联绑定**：捕获层事件本身不含实例标识，两次同参数捕获的原始事件指纹相同。
runner 在绑定阶段派生 `event_sha256 := SHA256(原始事件指纹 | instance_token)`，原始值另存
`runner.capture_raw_event_sha256`，落盘后可逐位复算。这让两次独立执行不互撞，同时保留
「A 次完整记录交 B 次执行」的复用拒绝。**这是捕获层已知缺口的补偿，不是缺口已关闭**。

## 3. 真实运行结果

```
job_key = gate3-final  attempt=1  fault=none
status  = succeeded     execution_status=ok
identity_status = unconfirmed   identity_valid=False
run_id = None           workflow_id = None
attestation_status = verified    reload_consistent = True
manifest_sha256 = b55a0b1e04543c0e6be08cac9976cedd8f3ea421f6f1a32422279fadd498fdd7
```

落盘记录 `out/gate3_runner/final/gate3-final__a1.json`（10612 字节，gitignored）。
重读复核：真实数据 sha / 56 点、五项 `unavailable`、`upstream=False`、
`execution_mode=capture_only_dry_run`、`identity` 与 `E.evaluate()` 重算一致。

## 4. T1 / T2 / T3 试运行（19/19）

| 组 | 用例 | 结论 |
| --- | --- | --- |
| T1 | 单次执行链全闭合 / 记录字段与事件绑定可复算 / 不产生拟合结果 / 零号不写成有效身份 | 4/4 |
| T2 | 同条件两次独立执行 / 绑定事件指纹区分 / 恢复保留旧记录 | 3/3 |
| T3 | 采样异常 / 拟合中断 / 证据缺失 / 结果写入失败 / 重复提交 / **跨进程重启恢复** / 跨运行复用被拒 | 7/7 |
| 禁令 | 临时增加候选 / 运行时调预算 / 外推站点 / 未冻结自动行为 / 篡改冻结清单 | 5/5 |

用户特别关注的三个必留测试都在：`T1` 零号如实落盘、`T3` 跨运行复用被拒、
`T1/T3` 落盘后恢复（含**起新 Python 进程**重读重算）。T2 实测两次的
`run_id_payload` 规范化摘要**逐位相同**（基础条件相同）、`instance_token` 不同、
两份记录并存不合并；两次都零号，没有因条件相同合并、也没有因实例不同造出有效号。

失败处理：采样异常 / 中断 / 落盘失败均写 `failed`，审计留痕，旧记录不覆盖；
证据缺失照常落盘 `unconfirmed`；重复提交 `rejected` 且旧文件逐位不变。

## 5. 全套回归

`68/68`、`45/45`、`35/35`、`31/31`、`25/25`、`45/45`、`28/28`、`27/27`、`76/76`
+ 执行链 `15/15` + runner `19/19`；harness / 适配层 / 产出层 / 纯函数 / 捕获层自检通过；
`minimal_runner.py --check rc=0`；清单 `--check rc=0`、冻结 `9/9`、阶段 0 `--check`
（239 / 239 / 严格 237）、三探针 `rc=0`、冻结审查脚本 **51 条 FAIL 0**、
`8015721` 三文件 blob 逐位 SAME、`git diff --check rc=0`、新增文件 `CRLF=0`。

改动面：`docs/round_2026-11-15/minimal_runner.py`（新增）、
`test_minimal_runner.py`（新增）、`docs/INDEX.md`、本文。
**身份层 / 生产拟合代码 / 既有测试 / 源码清单 / 被锁 blob 全部零改动**。

## 6. 仍未闭合（如实明账）

1. **身份仍零号**：S1/S2/S6 上游来源链未闭合，五项 `unavailable`。runner 验证的是
   「留证链」，不是「能产出有效身份」。
2. **捕获层无实例标识**：同参数两次捕获原始事件指纹相同，本版靠 runner 绑定层区分；
   Gate 2C 列的「多次采样逐次事件」仍需在**真实拟合**下实测，dry-run 覆盖不到。
3. **dirty patch 真实性 / 三时点 / harness 归属**：仍为 DESIGN。`software_identity` 是
   runner 侧实测明账，不进身份哈希，条件模板 `code` 不是真实源码摘要。
4. **软件身份半真半假已明账**：实测 `git_dirty=True` vs 模板 `code.git_dirty=False`，
   本版不让它进载荷，也不写成通过。
5. F-01 仍为**条件性关闭**（内部可信边界，非抗抵赖）；G9、F-10–F-13、P1×5/P2×1、
   契约正文 C1–C8 未处理；239 条历史记录保持原证据等级。
6. 算法主线 `§13` 留出 0/3、`§14` 全量 4/14（两次卡 M-b `acf≤0.50`）**仍未过线**。

## 7. 下一步（各自仍需单独授权）

- **真实 EIS 单次拟合**：把 runner 第 4 步从 dry-run 换成真实拟合入口，并同时补
  三时点快照、harness 归属、多次采样逐次事件；这一步才可能产生新的实验拟合结果。
- 在此之前 Gate 2C 维持 **CONDITIONAL**，不宣布无条件准入。
- 红线不变：`unavailable` 不升 `recorded`，推断来源不填历史缺口，三挂账不动，
  不改 `bm_core.py` / `bm_top.py` 拟合算法，零号不得写成有效身份。
