# 采样参数来源契约（sampler provenance）v1.0

> 实现：`sampler_provenance.py`（纯函数层，**不被生产模块 import**）
> 反例矩阵：`test_sampler_provenance.py`（**36/36**，其中 15 格是裁决过的矩阵）
> 上游：`docs/round_2026-11-14/SAMPLER_PROVENANCE_AUDIT.md`（阶段 A）、
> `SAMPLER_PROVENANCE_AUDIT_B.md`（补充审计：三执行字段 + 内核适用性）
>
> 本契约只回答一件事：**手上这条采样记录，八项配置各自的证据链闭没闭、哪一环断的。**
> 它不采集、不落盘、不进 `run_id`、不碰 `exec_identity.py`。

---

## 1. 边界

做：

- 八项配置的单项判定（含适用性、通道合法性、值比较）；
- 一条采样事件的聚合（最弱项 + 八项明细）；
- 原因码闭集，让"为什么弱"可筛选。

不做（本阶段明确不做）：

- 不挂 `MCMC.__init__` / `MCMC.run` 的钩子——那是捕获层；
- 不产出 `budget_params`、不改 `exec_identity.py`、不升 `identity_contract_version`；
- 不启动 runner，不跑批量拟合；
- **不修** `bm_real.py:1172` 的 `sampling` 块（读 `kw` 不读 `mc`）、不修
  `tree_depth_stats(mc, kw["max_tree_depth"])`（用声明值当上限）。两处都是生产采集链
  的问题，等本层语义稳定后单独立项。

---

## 2. 八项配置与观测通道

| 字段 | 类别 | 契约观测点 `OBSERVE_AT` | 允许的 `how` | 适用条件 |
|---|---|---|---|---|
| `n_warmup` | 预算 | `mcmc.num_warmup` | `attr` | 恒 |
| `n_samples` | 预算 | `mcmc.num_samples` | `attr` | 恒 |
| `num_chains` | 预算 | `mcmc.num_chains` | `attr` | 恒 |
| `target_accept` | 预算 | `mcmc.sampler._target_accept_prob` | `attr` | `adapt_step_size=True` |
| `max_tree_depth` | 预算 | `mcmc.sampler._max_tree_depth` | `attr` | **仅 NUTS** |
| `chain_method` | 执行 | `mcmc.chain_method` | `attr` | `num_chains > 1` |
| `progress_bar` | 执行 | `mcmc.progress_bar` | `attr` | 恒 |
| `extra_fields` | 执行 | `MCMC.run.extra_fields` | **`call_arg`** / `derived` | 恒 |

三条硬事实（实测，见补充审计）：

1. `target_accept` / `max_tree_depth` 在**内核**上，不在 `mc` 上；
2. `chain_method` / `progress_bar` 的属性只保存**被覆盖后的生效值**，请求值不在对象里
   ⇒ 请求值必须由构造边界另外提供（`resolved`）；
3. `extra_fields` **没有属性**（`hasattr=False`），只活在 `MCMC.run` 的栈帧里。

`observe()` 会拒绝把通道和位置混搭：`attr` 的 `where` 必须等于 `OBSERVE_AT`，
`call_arg` 的必须是 `MCMC.run.extra_fields`，`derived` 的必须**不是**契约点。
"从 `kw` 读"写成 `attr` 会被直接拒绝——那正是 F-4 的形状。

---

## 3. 两个状态字段

| 字段 | 表达 | 取值 |
|---|---|---|
| `observed_status` | 观测过程成不成功、属性适不适用 | `captured` / `attribute_missing` / `hook_not_reached` / `capture_error` / `not_applicable` |
| `status` | 请求、解析、观测三者的证据关系 | `complete` / `defaulted` / `derived` / `mismatch` / `incomplete` |

职责不同的两个例子：

- `observed_status="captured"` + 值不等 → `status="mismatch"`（**采到了，被改了**）；
- `observed_status="hook_not_reached"` → `status="incomplete"`（**没采到**）；
- `observed_status="not_applicable"` → `status="derived"` + `reason_code=kernel_inapplicable|single_chain`。

合成一个字段的后果是"该重试的"和"该查代码的"同码。
`observed_status_supplied` 另存调用方原始声明：适配性判成不适用而被改判时，这一栏能看出来。

---

## 4. 判定顺序（不许调换）

```
① 适用性     不适用 ⇒ observed_status=not_applicable / status=derived，跳过值比较
② 观测通道   attribute_missing|hook_not_reached|capture_error ⇒ incomplete
②' 观测缺失  observation=None ⇒ incomplete（resolved 不许顶替）
②'' 观测 NaN  observed.value 是 NaN ⇒ incomplete / observed_nan
③ 解析值缺失 resolved=None（只知道档名）⇒ derived / resolved_missing
④ 值不等     ⇒ mismatch
⑤ origin=default ⇒ defaulted（**不是** derived）
⑥ 其余       ⇒ complete
```

- **① 在 ④ 之前**：`HMC.__init__` 把 `_max_tree_depth` 硬编码成 `10`，它**不是**任何人的
  设置。先比数值会把"换了内核"报成"传递链被篡改"（假 `mismatch`）。这条是本轮特意
  锁死的实现细节。
- **② 在 ③④ 之前**：采集中断时手上有 `resolved`，拿它顶替会把"没采到"变成"采到了且一致"。
- **⑤ 用 `defaulted` 不用 `derived`**：默认值是**生效值**，与"只知道档名"不是一回事。
  区别在捕获位置——在调用点把默认值物化成实参并记下来算"记录实际值"；
  事后从源码读默认值算"猜"，不行。

`resolved.origin="default"` 且 `resolved.where=None` 时另加 `reason_code=origin_default_unsourced`：
`bm_real.KW` 是**模块级可变 dict**（`evidence/ws_crit/wp2_run.py:34-36` 等直接改它），
所以"`default`"不能自动读成"源码常量"。本层只保留来源信息，真实来源由捕获层提供。

---

## 5. 适用性规则 R1–R6

| 规则 | 内容 | 实现 |
|---|---|---|
| R1 | `target_accept` 两内核都有，但只在 `adapt_step_size=True` 时被消费 | `applicability(..., adapt_step_size=)` |
| R2 | `max_tree_depth` **仅 NUTS**；HMC 上是字面量 10 | `applicability(..., kernel=)` |
| R3 | 判别**不许用 `hasattr`**——HMC 也有 `_max_tree_depth` | `kernel` 必须显式传，`None` 直接 raise |
| R4 | 命中不适用：`not_applicable`→`derived`，**跳过值比较**，不记 `mismatch` | 判定顺序 ① |
| R5 | `num_chains == 1` 时 `chain_method` 不适用（`run` 不读它） | `applicability(..., num_chains=)` |
| R6 | depth 是轨迹预算：上界 `2**d - 1` ⇒ 13 vs 10 是 8191 vs 1023 | `tree_budget()` |

`applicability()` 缺上下文时**直接 raise**，不给默认值：默认值在这里正好是错的那一边。

---

## 6. 原因码（闭集，12 个）

| 码 | 含义 | 影响状态 |
|---|---|---|
| `sequentially_resolved` | `chain_method` 被 `local_device_count` 改成 `sequential` | 计入 `override_fields` |
| `forced_by_env` | `progress_bar` 被 `CI` / `PYTEST_XDIST_WORKER` 关掉 | 计入 `override_fields` |
| `single_chain` | `num_chains == 1` ⇒ `chain_method` 不适用 | — |
| `kernel_inapplicable` | 非 NUTS ⇒ `max_tree_depth` 不是设置值 | — |
| `step_size_not_adapting` | `adapt_step_size=False` ⇒ `target_accept` 不被消费 | — |
| `declared_not_delivered` | 声明里有、实际没送到（B 路径的 `extra_fields`） | — |
| `inferred_only` | 只有反推，没有真观测 | — |
| `origin_default_unsourced` | `default` 无出处 | — |
| `observation_missing` | 该采没采到 | — |
| `observed_nan` | 采到 NaN | — |
| `resolved_missing` | 只知道档名 | — |
| `value_differs` | 请求/解析与观测不等 | 不计入 override |

`OVERRIDE_CODES`（前两个）是**执行条件偏移**：状态是 `mismatch`，性质是"条件不等价"，
不是"证据链断裂"。聚合里单列 `n_override` / `override_fields`，避免两类混在一起没法处置。

> 现实后果，先说清楚：A 路径 `num_chains=3` + 1 台设备**每次**触发
> `sequentially_resolved` ⇒ 这条运行的聚合会停在 `mismatch`。这是**如实记录**，
> 不是回归：值没错，执行条件与声明不等价，而这件事以前没有任何字段记录。

---

## 7. 聚合

- 取**最弱项**：`incomplete > mismatch > derived > defaulted > complete`
  （`STATUS_RANK` 0–4），与记录顺序无关（G1）。
- `not_applicable` 在单项里已映射成 `derived` ⇒ 它把整体封顶在 `derived`：
  **不适用不能让整体看起来更完整**。
- 八项明细一条不许省：`fields` / `observed_status` / `reason_codes` / `why` 全落盘，
  聚合值只用于筛选。
- `require_all=True`（默认）时缺项、重复直接 raise。少一项就"聚合通过"是这轮要防的
  那种不报错的错。
- `complete_evidence = (weakest == "complete")`：一个字段，不是形容词。

---

## 8. 反例矩阵（15 格）与用例

| # | 场景 | 期望 | 用例 |
|---|---|---|---|
| 1 | 八项全显式且一致 | 8/8 `complete` | M1 |
| 2 | 八项全走默认 | 8/8 `defaulted`，**不等于** `derived` | M2 |
| 3 | 部分显式 | 逐项不同，聚合取最弱 | M3 |
| 4 | `--quick` 改三个数 | 记**改后**值，`origin≠default` | M4 |
| 5a | 五参数之一被改 | 该项 `mismatch`，其余不动，`override_fields=[]` | M5a |
| 5b | `chain_method` 显式传 `vectorized` / 被设备数改 | `complete` / `mismatch`+`sequentially_resolved` | M5b |
| 5c | `progress_bar` 请求 True + `CI=1` | `mismatch`+`forced_by_env`；False→False 不造假错 | M5c |
| 5d | `extra_fields` 声明 `num_steps`、收到 `diverging` | `mismatch`；反推通道报同一差异但落 `incomplete` | M5d |
| 6 | `observed` 缺失 | `incomplete`，`resolved` 不顶替 | M6 |
| 7 | 属性改名（库升级） | `attribute_missing` → `incomplete` | M7 |
| 8 | 同工作流两次不同预算 | 两条独立记录，互不覆盖 | M8 |
| 9 | `num_chains=3` + 1 设备 | 五项仍 `complete`，`chain_method` 记偏移 | M9 |
| 10 | 档里缺键 ⇒ 落函数默认 | `defaulted` 且 `why` 指到**哪个函数**；无出处另加原因码 | M10 |
| 11 | 换 HMC | `max_tree_depth` = `not_applicable`→`derived`，**不是** `complete=10`、**不是** `mismatch` | M11 |
| 12 | `num_chains=1` | `chain_method` = `not_applicable` | M12 |
| 13 | B 路径声明 `num_steps` | 实为 `("diverging",)` ⇒ `mismatch`/`declared_not_delivered` | M13 |

另有 P 组 10 格（通道白名单、契约观测点、origin 必填、NaN、原因码闭集、不许自报不适用、
元组无序比较、树深预算、适用性缺上下文、`where` 必填）、G 组 5 格（聚合）、
X 组 5 格（生产零引用、无重依赖、不改入参、JSON 直出、字段顺序稳定）。
合计 **36/36**。

---

## 9. 复现

```bash
D:/autoeis_env/Scripts/python.exe sampler_provenance.py        # 内置自检
D:/autoeis_env/Scripts/python.exe test_sampler_provenance.py   # 36/36
```

回归不受影响（本层零耦合）：identity 68/68、capture 31/31、bundle 25/25、
stage0 冻结 9/9、`map_stage0_records --check` rc=0（239/239，严格 237）。
