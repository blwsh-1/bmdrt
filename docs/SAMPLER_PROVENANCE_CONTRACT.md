# 采样参数来源契约（sampler provenance）v1.0

> 实现：`sampler_provenance.py`（纯函数层，**不被生产模块 import**）
> 反例矩阵：`test_sampler_provenance.py`（**45/45**，其中 15 格是裁决过的矩阵，另 2 格为矩阵外追加、5 格为审后修复的护栏、2 格为「缺参不许默认」）
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

`observe()` 与 `evaluate()` **两处都**守通道与位置的对应：`attr` / `call_arg` 的 `where`
必须等于 `OBSERVE_AT`，`derived` 的必须**不是**契约点。"从 `kw` 读"写成 `attr` 会被直接拒绝
——那正是 F-4 的形状。

为什么 `evaluate()` 要再查一遍：它收的是**任意 Mapping**，绕过 `observe()` 手写
`{"value": 800, "how": "attr", "where": "kw.n_warmup"}` 就能拿到 `complete`。
判定入口是最后一道，不能假设上游一定用了构造函数（P11）。

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
             （不适用 + 声明偏移 ⇒ 当场 raise，不许改写成 single_chain）
② 观测通道   attribute_missing|hook_not_reached|capture_error ⇒ incomplete
②' 观测缺失  observation=None ⇒ incomplete（resolved 不许顶替）
②'' 观测 NaN  observed.value 是 NaN ⇒ incomplete / observed_nan
②‴ 偏移自洽   声明了 sequentially_resolved|forced_by_env 就必须有一对不同的值
③ 解析值缺失 resolved=None（只知道档名）⇒ derived / resolved_missing
④ 值不等     ⇒ mismatch
⑤ origin=default ⇒ defaulted（**不是** derived）
⑥ 其余       ⇒ complete
```

- **① 在 ④ 之前**：`HMC.__init__` 把 `_max_tree_depth` 硬编码成 `10`，它**不是**任何人的
  设置。先比数值会把"换了内核"报成"传递链被篡改"（假 `mismatch`）。这条是本轮特意
  锁死的实现细节。
- **② 在 ③④ 之前**：采集中断时手上有 `resolved`，拿它顶替会把"没采到"变成"采到了且一致"。
- **① 里的偏移禁用**：不适用说的是"这次执行根本不读它"，与"读到了、被改成别的值"
  是两件事。单链下声明 `sequentially_resolved` 描述的是一个**不可能发生的改写**
  ⇒ raise，不静默降级成 `single_chain`（P12）。
- **②‴ 偏移自洽**：偏移是两值之比，见 §6 的五条护栏。
- **⑤ 用 `defaulted` 不用 `derived`**：默认值是**生效值**，与"只知道档名"不是一回事。
  区别在捕获位置——在调用点把默认值物化成实参并记下来算"记录实际值"；
  事后从源码读默认值算"猜"，不行。
- **⑤ 只能在 ④ 通过后到达**：`defaulted` 说的是**请求值的来源**（来自库默认，不是显式指定），
  不是"这次执行没问题"。`origin=default` 但两值不等时在 ④ 就返回 `mismatch`，
  而 `resolved.origin="default"` 仍留在记录里——**来源事实与证据结论分两处存**，
  谁也不许把对方吸收掉。所以"聚合 = `defaulted`"只在**所有默认来源项都值相等**时成立；
  A 路径（3 链 / 1 台设备，`chain_method` 被改成 `sequential`）的聚合是 **`mismatch`**，
  不是 `defaulted`（M14）。

`resolved.origin="default"` 且 `resolved.where=None` 时另加 `reason_code=origin_default_unsourced`：
`bm_real.KW` 是**模块级可变 dict**（`evidence/ws_crit/wp2_run.py:34-36` 等直接改它），
所以"`default`"不能自动读成"源码常量"。本层只保留来源信息，真实来源由捕获层提供。

---

## 5. 适用性规则 R1–R6

| 规则 | 内容 | 实现 |
|---|---|---|
| R1 | `target_accept` 两内核都有，但只在 `adapt_step_size=True` 时被消费 | `applicability(..., adapt_step_size=)`，**无默认值**，缺参 raise |
| R2 | `max_tree_depth` **仅 NUTS**；HMC 上是字面量 10 | `applicability(..., kernel=)` |
| R3 | 判别**不许用 `hasattr`**——HMC 也有 `_max_tree_depth` | `kernel` 必须显式传，`None` 直接 raise |
| R4 | 命中不适用：`not_applicable`→`derived`，**跳过值比较**，不记 `mismatch` | 判定顺序 ① |
| R5 | `num_chains == 1` 时 `chain_method` 不适用（`run` 不读它） | `applicability(..., num_chains=)` |
| R6 | depth 是轨迹预算：上界 `2**d - 1` ⇒ 13 vs 10 是 8191 vs 1023 | `tree_budget()` |

`applicability()` 与 `evaluate()` 缺上下文时**直接 raise**，签名里**一个默认值都不给**
（`kernel` / `num_chains` / `adapt_step_size` 全是 `None`）：默认值在这里正好是错的那一边。
类型边界也封闭：`num_chains` 显式拒 `bool`（`True` 会被当成 1）、`adapt_step_size` 只收
`True/False`（`None` 不能被读成"关闭自适应"）、`tree_budget` 拒 `bool`（`True` 会算出 1）。

`adapt_step_size` 这条是审后补的第二道：**显式传 `None`** 与**干脆不传**必须**同罚**。
早先签名写成 `adapt_step_size=True`，于是 `applicability("target_accept")` 会静默返回
"适用"——等于替调用方假定自适应已开，R1 明说不能这么假定。反过来默认 `False` 也不行：
那会把每一次都判成 `step_size_not_adapting`，是另一个方向的假结论。**不知道就是不知道。**

> 对阶段 C 的硬要求：捕获层必须给出**真实**的 `adapt_step_size`（如
> `isinstance(mcmc.sampler, NUTS) and mcmc.sampler.adapt_step_size`，或构造实参里的同名值），
> 不能依赖本层默认值，也不能在调用点写死 `True` —— 那等于把这道护栏又关掉。

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

偏移是**声明**，不是从值里推的；正因为是声明，才不能被随口声明。四处当场 raise（M15）：

| # | 假的偏移声明 | 为什么不行 |
|---|---|---|
| 1 | 跨字段套用（`n_warmup` + `sequentially_resolved`） | 只有 `chain_method` 会被设备数改、只有 `progress_bar` 会被 CI 改；跨字段套用等于给"改值"发免责声明 |
| 2 | 值没变却声称偏移 | 偏移是两值之比，两值相等就没有"偏移"这回事 |
| 3 | 没采到值却声称偏移（`attribute_missing` / `None` / NaN） | `incomplete` 想借"环境干的"洗白，正好是这轮要防的静默 |
| 4 | 只知道档名（`resolved=None`）却声称偏移 | 没有请求值就没有比较对象 |
| 5 | 字段本身**不适用**却声明偏移（单链声 `sequentially_resolved`） | 不适用是"不会被读"，不是"被改成别的值"；放行等于把"没看"洗成"看过且被改" |

反过来，**漏声明**不会让记录变好看：同一处差异不带码就落 `value_differs`，计入
`mismatch` 但不计入 `n_override`。"漏声明"只会让整体看起来更糟，不会更干净——
失败方向是安全的（M15 末段）。

> 现实后果，先说清楚：A 路径 `num_chains=3` + 1 台设备**每次**触发
> `sequentially_resolved` ⇒ 这条运行的聚合会停在 `mismatch`（`n_override=1`）。
> 这是**如实记录**，不是回归：值没错，执行条件与声明不等价，而这件事以前没有任何字段
> 记录。反过来，**不许**把它读成 `defaulted`："来源是库默认"与"执行被改写过"是两件事。

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
| 2 | 八项全走默认**且值都相等** | 8/8 `defaulted`，**不等于** `derived` | M2 |
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

另有 P 组 **17 格**（通道白名单、契约观测点、origin 必填、NaN、原因码闭集、不许自报不适用、
元组无序比较、树深预算、适用性缺上下文、`where` 必填，加审后 5 格：`evaluate` 复核 `where`、
不适用不许声明偏移、`bool` 不是正整数、`tree_budget` 拒 `bool`、`adapt_step_size` 严格，
再加 2 格「缺参不许默认」：P16 `applicability("target_accept")` 直接 raise、P17 `evaluate()` 同罪，
两格都带正对照证明 `True`⇒`complete`、`False`⇒`derived`/`step_size_not_adapting` 照旧）、
G 组 5 格（聚合）、X 组 5 格（生产零引用、无重依赖、不改入参、JSON 直出、字段顺序稳定）。

**矩阵外追加 2 格**（不改动上面 15 格的编号）：

| # | 场景 | 期望 | 用例 |
|---|---|---|---|
| 14 | `origin=default` 且值被改；A 路径（3 链 / 1 设备）整体 | `status=mismatch` 但 `resolved.origin` 仍为 `default`；聚合 `mismatch`，**不是** `defaulted` | M14 |
| 15 | 四类假的偏移声明 + 同一差异带码/不带码 | 四处 raise；带码进 `override_fields`，不带码落 `value_differs` | M15 |

合计 **45/45**。

---

## 9. 复现

```bash
D:/autoeis_env/Scripts/python.exe sampler_provenance.py        # 内置自检 17 项
D:/autoeis_env/Scripts/python.exe test_sampler_provenance.py   # 45/45
```

两层职责分清楚，**不互相顶替**：

| | 内置自检（17 项） | `test_sampler_provenance.py`（45 格） |
|---|---|---|
| 定位 | 不装 pytest 也能跑的最小体检 | 完整反例矩阵 + 接口护栏 |
| 失败时 | 只报"哪条最核心的规则破了" | 精确到哪一格、期望与实际 |
| 覆盖 | 适用性三条规则、缺上下文 raise（**核理由**）、`True`/`False` 两条正路、假 `mismatch` 两处（HMC depth、`adapt_step_size=False`）、八项聚合三档 | P 17 / M 18 / G 5 / X 5，117 条断言 |

自检里**禁止**写裸 `except ValueError: pass`：只要抛了任何 `ValueError` 就算通过，
等于让"别的参数报错"替这条护栏背书。统一走 `why_raised()`——返回异常消息再核关键词，
没 raise 返回 `None`。缺参检查还必须配正对照，否则"整个分支被删掉"也会全绿。

回归不受影响（本层零耦合）：identity 68/68、capture 31/31、bundle 25/25、
stage0 冻结 9/9、`map_stage0_records --check` rc=0（239/239，严格 237）。

---

## 10. 夹具与覆盖审计（阶段 B 收口前）

审的是"测试自己会不会骗人"，不是判定逻辑。三条结论都有计数支撑。

**① 夹具复用：22/45 格拿整份 `_recs()`**

`_recs(**over)` 造八项齐全的一致记录，45 格里 22 格直接复用。`_slot = S._slot`
是**同一个对象不是副本**——模块自检与 pytest 共用一个构造器，不存在两份悄悄走样。

代价与对策：复用让"八项齐全"有单点保证，但夹具**自身**的一致性没有第二个人签字。
它靠 M1（8/8 `complete`）反证——夹具里任一处 `resolved` 与 `observed` 写得不等，M1 立刻红。
这不是循环论证：M1 断言的是**聚合结果**，夹具写错只能让它失败，不能让它通过。

**② 两处刻意手写（不是漏改）**

| 用例 | 为什么不能复用 `_recs()` |
|---|---|
| G1（顺序无关） | 必须手排 8 条并让状态混合（`n_samples` 走 `default`、`num_chains` 走 `resolved=None`）；复用一份全 `complete` 的夹具备，等于测"8 个 complete 的顺序无关"，最弱项逻辑根本没被激活 |
| M2（八项全默认） | 数值与 `_recs()` 完全重合、只差 `origin`。**这是真重复**，但它同时是 M1 的对照（8 `complete` vs 8 `defaulted`）；合并会把两格的差异藏进参数，读代码的人看不出"这一格换了什么" |

**③ `_must_raise` 的单一依赖：37 次调用全部带 `needle`**（AST 核过，裸调用 0 次）

单一 helper 的风险是"它自己写错，37 格同时失守"。现在 helper 内部是
`assert needle in msg`——helper 失效表现为 AssertionError，不会退化成"返回 False 被忽略"。

**④ 自检 vs pytest 的差异（修前 → 修后）**

| | 修前 | 修后 |
|---|---|---|
| 缺参检查 | 裸 `except ValueError` ⇒ 只判"有没有抛"，不核理由 | `why_raised()` 取消息再核 `adapt_step_size` |
| 正对照 | 无 ⇒ 删掉整个 `target_accept` 分支也全绿 | `True`⇒适用、`False`⇒`step_size_not_adapting` 各一条 |
| `evaluate` 两条路 | 无 | `True`⇒`complete`、`False`⇒`derived`，且**观测值故意取 0.98≠0.9** 确认不适用先于值比较 |
| 变异验证 | — | 四种改法（两处默认值回退、报错换词、`adapt_step_size` 被无视）自检全部 rc=1 |

仍只在 pytest 里的（**有意不搬**）：`where` 通道（P1/P2/P11）、原因码闭集（P5）、
自报不适用（P6）、`bool`/`None` 边界（P13/P14）、以及 M/G/X 全部。
自检的定位是"没有 pytest 的环境也能跑的最小体检"，不是矩阵的子集副本。

**不做**：不合并 G1/M2 与 `_recs()`，不把 45 格压成参数化表——这两处的"手写"本身就是被测性质。
