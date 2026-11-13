# 阶段 0C：schema 契约验收

状态：**schema 层通过，判定逻辑未实现**。本轮不跑拟合、不接生产、不改任何门禁。

产物：
- `circuit_registry.py`（注册表 + 三个输出块 + 冻结契约，导入 0.05s，不加载 jax/numpyro）
- `test_circuit_registry.py`（29 条用例，实跑 **29/29 通过**）
- `docs/round_2026-11-13/STAGE0A_SCHEMA_PRESSURE.md`（三谱压测，只读既有产物）

## 1. 六类验收情形 → 用例对照

| 验收情形（本轮要求） | 用例 | 结论 |
|---|---|---|
| 结构排除与预检未知 | `I1` / `I3` / `I2` | 低频感抗谱 → voigt `excluded`；nested 未登记预检 → `unknown`；容性谱 → `not_excluded`（预检覆盖不到，已知的洞） |
| 拟合未运行 / 失败 / 采样失败 | `D1` | `total = completed + failed + not_run` 闭合，`failure_kinds` 可细分 |
| 拟合贴底但参数不稳 | `E1` | `rms_rel 2.40%` 与 `n2 flag=at_bound` 同时保留，不合成单一分数 |
| 多候选可解释但不可区分 | `F2` / `F3` | `non_identifiable` 要求 `n_viable>=2`；`identified` 无样本外证据被拒 |
| 合成谱真值下全候选不足 | `F4` | `model_inadequate` 需 `truth_known` 或 `noise_calibrated` |
| 真实谱证据不足、不得误报 | `F1` / `F4` | 真实谱单候选 → `insufficient_evidence`；无真值时 `model_inadequate` 直接报错 |

## 2. schema 已强制的硬约束（可执行，不是注释）

1. `screening` 闭合：`registered = excluded + not_excluded + unknown`。
2. `only_survivor` **只由** `survivors == 1` 计算，不接受调用方传入；`status="only_survivor"` 直接报错。
3. `evidence` 五态互斥，`identified` 需 `out_of_sample_valid=True` 且 `n_viable>=2`。
4. `model_inadequate` 需 `truth_known` 或 `noise_calibrated`。
5. `converged=False` 时 `rms_rel`/`acf` 必须为 `None`（沿用 E3：not_converged 不报数）。
6. `param_stability` 必须是"参数名 → 描述块"，不许传标量分数。
7. `L1` 返回集合 `{promote, defer, not_evaluated_budget}`，**永不含 `excluded`**。
8. `wired=False` 的元件必须写 `blocked_reason`，且不能被候选引用。
9. `candidate_id` 覆盖 `topology + elements + config + init + budget + likelihood` 六项。
10. 候选集 manifest 的 `created_before_fit` 必须为 `True`；冻结后追加直接报错并提示新建 `candidate_set_id`。

## 3. 注册内容

| 层 | 条目 |
|---|---|
| 拓扑（2） | `voigt`（串联 HN 弛豫，production）、`nested`（层状嵌套，exploratory） |
| 元件已接线（4） | `hn`、`cpe`、`rl_parallel`、`rl_series` |
| 元件未接线（4） | `w_inf`、`w_open`、`w_short`、`adsorption`——登记公式与参数，但现有拓扑**没有对应槽位**，`wired=False` |

未接线不是"没测"，是"没有接点"。AZ31 那次失败正是接点问题（`R∥L` 必须在第二并联节点内，
挂到 voigt 最外层串联只能把 noise=0 残差从 44.0% 降到 40.7%），所以这一档必须显式区分。

`registry_digest = 460373a48cd2e9b1`（本版）。manifest 会记录它，注册表被改可被发现。

## 4. 本轮未覆盖（留给阶段 1/2/3）

- **状态转移逻辑本身**：契约测试只证明"每种状态能被表达且乱填会报错"，不证明
  "从观测到 state 的推导正确"。后者属于 evaluator，且依赖 σ 口径。
- **runner 的调度一致性**：同一候选在不同 OS/进程下是否给出同一 `candidate_id`、
  同一 manifest digest，未测。
- **阈值**：`acf` 上限、参数区间宽度、`n_viable` 的具体判定，本轮一个都没定。
  在拿到一批描述性观测之前定阈值就是拍脑袋。
- **真实谱的 `model_inadequate`**：条件参数已就位（`noise_calibrated`），但没有标定噪声
  的实现，所以这一态在真实谱上仍然打不出来——这是有意为之。

## 5. 边界

- 不接 `bmdrt.py`；`voigt` 仍是生产默认族；`nested` 仍 `exploratory`。
- §13 `0/3`、§14 `4/14`、§17 `0/10` 原样，未因本轮改动重估。
- 三谱压测是 2 条合成谱 + 1 条实测谱，样本量不支持任何总体陈述。
