# ③ 补充审计：三个执行行为字段 + 内核适用性

> 阶段 A 的**补丁**，不是重做。阶段 A（`SAMPLER_PROVENANCE_AUDIT.md`）已实测
> 「五参数值一致」，但没有回答「八项配置各自的传递链」。
> 本文只补三件事：`chain_method` / `progress_bar` / `extra_fields` 的独立传递链，
> `target_accept` / `max_tree_depth` 在 NUTS 与 HMC 上的适用规则，
> 以及由此得出的**观测位置分层**（哪些字段根本没有 `attr_path`）。
>
> 不写 `sampler_provenance.py`，不改 `exec_identity.py`，不改生产代码，不跑拟合。

---

## 1. 方法

| 手段 | 范围 | 结论强度 |
|---|---|---|
| 静态阅读 | 全仓 `chain_method` / `progress_bar` / `extra_fields` 三词逐一搜（`out/pb/rebuilt/` 的补丁包副本一并搜过，里面是同样的代码），三条生产路径 + `evidence/` | 可复核 |
| 库源码 | `numpyro 0.19.0`：`MCMC.__init__` / `MCMC.run` / `MCMC.get_extra_fields` / `NUTS.__init__` / `HMC.__init__` / `HMC.default_fields` | **实测**（`inspect.getsource`） |
| 微预算探针 | `exec_fields_probe.py`：4 步 warmup / 4 步采样，只读属性与实参，不看数值 | **实测** |
| 未做 | 没跑 `bm_real.fit_one` 全链路；没有比较 `sequential` 与 `parallel` 是否逐位同数 | — |

探针输出（本机：`jax.local_device_count()=1`，`CI`/`PYTEST_XDIST_WORKER` 均未设）：

| 构造 | `num_chains` | `progress_bar` 实参 | `mc.chain_method` | `mc.progress_bar` | `mc.sampler._max_tree_depth` | `get_extra_fields()` 键 |
|---|---|---|---|---|---|---|
| 1 链 | 1 | `True` | `parallel` | `True` | 7 | `diverging` |
| 3 链 | 3 | `True` | **`sequential`** | `True` | 7 | `diverging` |
| 3 链 | 3 | `False` | **`sequential`** | `False` | 7 | `diverging`,`num_steps` |
| 1 链 + `CI=1` | 1 | `True` | `parallel` | **`False`** | 7 | `diverging` |

三条可直接读出的事实：

1. **`mc` 上没有任何字段保存"用户请求的"`chain_method` 与 `progress_bar`。**
   探针查了 `_chain_method` / `requested_chain_method` / `_progress_bar` /
   `requested_progress_bar`，四个都不存在。属性里只有**被覆盖后的生效值**。
2. **`mc` 上没有 `extra_fields` 属性**（`hasattr` 为 `False`），整个对象里也没有
   任何字段记住它——`MCMC.run` 把 `extra_fields` 与 `_sample_field`、`_default_fields`
   合并成局部变量 `collect_fields`，用完即弃。
3. `chain_method` 的 `sequential` 警告在**构造时**抛（探针里那行 traceback 指向
   `MCMC(...)` 而不是 `mc.run(...)`），所以只挂 `MCMC.run` 的钩子永远看不到它。

---

## 2. 三字段传递链

### 2.1 `chain_method`

| 环节 | 事实 |
|---|---|
| 请求来源 | **项目代码一次都没传**。`grep -rn chain_method --include=*.py .` 全仓**零命中** ⇒ 永远取 `MCMC.__init__` 形参默认 `"parallel"` |
| 中间转换 | `if chain_method == "parallel" and local_device_count() < num_chains: chain_method = "sequential"` + `warnings.warn` |
| 实际生效位置 | `mc.chain_method`（构造后即定稿），`MCMC.run` 依它三选一：`_laxmap` / `pmap` / 直接调用 |
| 适用条件 | `num_chains > 1`。`num_chains == 1` 时 `run` 走 `if self.num_chains == 1` 分支，**`chain_method` 完全不被读取**（属性可读，读数无意义） |
| 观测位置 | 生效值：`mc.chain_method`（构造边界之后任意时刻）<br>请求值：**只能**在 `MCMC.__init__` 实参边界取 |

今天会不会触发：**A 路径每次触发**（`KW["num_chains"]=3` > 1 台设备），B/C 路径同样
（`num_chains=2`）。A 路径的警告被 `bm_real.py:90` 的 `filterwarnings("ignore")` 吞掉。
生产链上只有 1 台设备这一点写在 `bm_core.fit` docstring 里（"CPU 上多链是顺序执行的"），
但**没有任何运行产物记录本机实际走的是哪支**。

> 数值等价性**未验证**：`_laxmap` 与 `pmap` 都按 `random.split(key, num_chains)` 分键，
> 理论上同流；但两者的浮点归约顺序不同，不排除末位差异。**没测过就不写"等价"**。

### 2.2 `progress_bar`

| 环节 | 事实 |
|---|---|
| 请求来源 | A：`KW["progress"]=False`（`bm_real.py:98`）→ `fit_one(**kw)` → `bm_core.fit(progress=)` → `progress_bar=progress`（`:607`）<br>B：`nested_fit.run(progress=False)` 默认 → `fit_top(progress=)`（`:1004`）；`--with-voigt` 分支 `voigt_reference` **硬编码 `progress=False`**（`nested_fit.py:217`）<br>C：`QUICK["progress"]=False` → `fit_voigt`→`fit_one`→`B.fit(**kw)` **读到**；`fit_nested` **硬编码 `progress=False`**（`bm_select.py:394`），完全不读 `kw["progress"]` |
| 中间转换 | `if callable(chain_method) and num_chains>1: progress_bar=False`（不触发，项目从不传 callable）<br>**`if "CI" in os.environ or "PYTEST_XDIST_WORKER" in os.environ: self.progress_bar = False`**（无条件覆盖，探针第 4 行实测） |
| 实际生效位置 | `mc.progress_bar`（构造时定稿，`run` 不复查） |
| 适用条件 | 全部。纯 UI，不参与数值 |
| 观测位置 | 生效值：`mc.progress_bar`<br>请求值：**只能**在 `MCMC.__init__` 实参边界取。**环境变量一开，请求值被永久丢弃** |

关键不对称：`chain_method` 的覆盖有警告（哪怕被吞），`progress_bar` 的覆盖**静默、无警告**。
所以"记录里写着 `progress_bar=False`"这件事，既可能是用户要的，也可能是 CI 环境改的，
**事后无法区分**。这正好落在你要的 `observed_status` 里：`captured` 但 `why=forced_by_env`
必须能标出来，否则等于没记。

### 2.3 `extra_fields`

这条是三项里唯一**没有属性**的。

| 环节 | 事实 |
|---|---|
| 请求来源 | A：`KW["extra_fields"]=("num_steps",)`（`:99`）→ `bm_core.fit(extra_fields=)` → `mc.run(extra_fields=)`（`:608`）<br>B：**不存在请求入口**。`fit_top` 签名里没有这个参数，`:1006` 硬编码 `extra_fields=("diverging",)`<br>B 的 voigt 分支：`voigt_reference` 用 `**kw`，而 `kw` 来自 `budget_kw`，`FIT_KEYS`（`nested_fit.py:82`）只含五参数 ⇒ 落到 `bm_core.fit` 的默认 `()` ⇒ **不带 `num_steps`**<br>C：`QUICK["extra_fields"]=("num_steps",)` → 同样只有 voigt 那侧吃到；`fit_nested` 不传 ⇒ `("diverging",)` |
| 中间转换 | `collect_fields = (sample_field,) + default_fields + extra_fields`，`default_fields=("z","diverging")`（`HMC.default_fields`）；`~` 前缀表示"删位点"，本仓未用 |
| 实际生效位置 | **`MCMC.run` 的形参 `extra_fields`**，且只活在 `run` 的栈帧里 |
| 适用条件 | 全部；但它决定"事后能算什么"——没有 `num_steps` 就算不出树深触顶率 |
| 观测位置 | **只有** `MCMC.run` 调用边界。属性不存在，`attr_path` 无从谈起 |

由此产生一条**结构性后果**（此前只被记成"不对称"）：

`num_steps` 不采 ⇒ `bm_experiments.tree_depth_stats` 返回 `None` ⇒ `bm_select` 的
`tree_cap` guard 只能记 `not_applicable`。所以 **nested 族的树深触顶率是不可能测到的**，
不是因为忘了算，而是因为 `fit_top` 从没请求过那个字段。这条要写进 guard 的
`why`，否则读起来像"算失败"。

---

## 3. 路径差异矩阵

`✔` = 该路径下这组值**真的传到了 `MCMC`/`MCMC.run`**；`✘` = 声明里有但被丢弃；
`—` = 该路径没有这个概念。

| 路径 | 入口 | `chain_method` | `progress_bar` | `extra_fields` |
|---|---|---|---|---|
| **A** | `bm_real.run` → `fit_one` → `bm_core.fit` | `—`（库默认，被改 `sequential`）| ✔ `KW` 的 `False` | ✔ `("num_steps",)` |
| **B** | `nested_fit.run` → `fit_top` | `—`（同上，`num_chains=2`）| ✔ `run(progress=)` 显式传 | ✘ **硬编码 `("diverging",)`**；预算里的 `("num_steps",)` 被 `FIT_KEYS` 丢 |
| **B+voigt** | 同上 + `voigt_reference` → `bm_core.fit` | `—` | ✔ 硬编码 `False` | ✘ `**kw` 不含 ⇒ 默认 `()` |
| **C** | `bm_select.fit_voigt` / `fit_nested` | `—` | voigt ✔ 读 `kw`；nested ✘ 硬编码 `False` | voigt ✔ `("num_steps",)`；nested ✘ `("diverging",)` |
| **C 重跑** | `_refit(kw=BUDGETS["standard"])` | `—` | 同 C | 同 C |

读法：**同一个 `BUDGETS` 字典，C 路径的 nested 侧只吃到五参数**；`progress` 靠两边
恰好都是 `False` 而看不出差别，`extra_fields` 直接不同。B 路径同理，且 B+voigt 与
B 的 nested 在同一次 run 里 `extra_fields` 也不一样（`()` vs `("diverging",)`）。

这条差异今天的后果是**可观测的**：`bm_real` 的 `tds`（树深统计）只存在于 A 路径，
B/C 的 nested 一律 `not_applicable`；voigt 侧在 C 里反而有 `num_steps`。
所以「两族同一组对照」里，一族有树深指标一族没有，**是配置差异不是数据差异**。

---

## 4. 观测位置分层（`attr_path` 不是通用字段）

你那句"不能为了统一数据结构就假设所有字段都有 `attr_path`"实测成立。八项分三类：

| 类别 | `how` | 字段 | 观测点 | 请求值能否事后恢复 |
|---|---|---|---|---|
| 属性可读，且属性=请求值 | `attr` | `num_warmup` `num_samples` `num_chains` | `mc.<name>` | 能（属性没被改写） |
| 属性可读，但属性只是**生效值** | `attr`（+ 必须并列构造实参） | `chain_method` `progress_bar` | `mc.<name>` / `mc.sampler._target_accept_prob` / `_max_tree_depth` | **不能**。覆盖一旦发生，请求值不在对象里 |
| 属性不存在 | `call_arg` | `extra_fields` | `MCMC.run` 形参 | 只有调用边界能取 |
| 只能反推 | `derived` | `extra_fields`（备选） | `get_extra_fields().keys() − default_fields` | 是**推断**，不是观测；`num_steps` 出现与否能推出来，但"请求了但库忽略"推不出来 |

所以 `observed` 的载荷建议改成三件套，`attr_path` 只属于第一类：

```
observed: { value, how: "attr"|"call_arg"|"derived", where: "mc.num_warmup"
                                                     | "MCMC.run.extra_fields"
                                                     | "keys(get_extra_fields)-default_fields" }
```

`call_arg` 的取法只有一个：包 `MCMC.run`（`functools.wraps` + 读 `kwargs`/
`signature.bind`）。`attr` 的取法就是读属性。两者都是"观测"，不许互相顶替——
用 `get_extra_fields()` 反推 `extra_fields` 必须记 `how="derived"`，
哪怕推出来的值是对的。

---

## 5. NUTS / HMC 适用性规则

`inspect.signature` 实测：

| 形参 | `NUTS.__init__` | `HMC.__init__` | 属性 | 默认 |
|---|---|---|---|---|
| `target_accept_prob` | ✔ | ✔ | `sampler._target_accept_prob` | 两者都是 **0.8** |
| `max_tree_depth` | ✔ | **✘**（传了就 `TypeError`） | `sampler._max_tree_depth` | `NUTS` 从形参取；**`HMC` 硬编码 `10`** |
| `chain_method` | ✘ | ✘ | —— | 在 `MCMC` 上 |

规则（写进契约前请确认，我按这个实现）：

- **R1** `target_accept`：两内核都适用，语义相同（dual averaging 目标接受率）。
  但只有 `adapt_step_size=True`（默认）时才真被消费；`adapt_step_size=False` 时
  属性仍在、值不生效 ⇒ 记 `observed_status="not_applicable"`，不记 `complete`。
  今天项目从不设 `adapt_step_size`，**不触发**，但规则要有。
- **R2** `max_tree_depth`：**仅当内核是 NUTS 才适用**。HMC 上的 `_max_tree_depth`
  是构造函数里的字面量 `10`，不是任何人的设置。
- **R3** 判别**不许用 `hasattr`**：`HMC` 也有 `_max_tree_depth`（值 10），
  `hasattr` 会给出"采到了"的假象。必须 `isinstance(mc.sampler, NUTS)`，或用
  `type(mc.sampler).__name__` 显式查表。
- **R4** 命中 R2 不适用时：该项 `status = "derived"` + `why = "kernel_inapplicable"`，
  **跳过值比较**（记 `mismatch` 会把"换了内核"误报成"传递链被改"），
  **且仍然参与最弱项聚合**（聚合到 `derived` 是诚实的：这条运行的 provenance 确实没闭合）。
- **R5** `chain_method` 在 `num_chains == 1` 时记 `not_applicable`：属性读得出
  `parallel`，但 `run` 根本不看它。今天的 A/C 单链工况会命中。
- **R6** `max_tree_depth` 的语义量级：NUTS 树深是"最多翻倍 d 次"，即上界约 `2^d` 步。
  A 的 `13` 与 B/C 的 `10` 差 `8192` vs `1024` 步上限——F-5 里"同名档不同 depth"
  不只是记个数字不同，是**每步轨迹预算差 8 倍**。

---

## 6. 八项配置的证据边界（对应你那张表）

| 配置 | 当前证据 | 补充结论 | 状态 |
|---|---|---|---|
| `num_warmup` | 15/15 一致 | 属性=请求值，无中间改写 | 可进契约 |
| `num_samples` | 15/15 一致 | 同上 | 可进契约 |
| `num_chains` | 15/15 一致 | 同上；但它是 `chain_method`/`target_accept` 的适用条件输入 | 可进契约 |
| `target_accept` | 15/15 一致 | 在 `sampler` 上，不在 `mc` 上；两内核通用（R1） | 可进契约，带 R1 |
| `max_tree_depth` | 15/15 一致 | 在 `sampler` 上；**仅 NUTS**（R2/R3/R4） | 可进契约，带 R2–R4 |
| `chain_method` | 发现设备数改写 | 请求值**不在对象里**；构造时警告、被 `bm_real.py:90` 吞；`num_chains=1` 不适用（R5） | 可进契约，`how=attr`+构造实参，强制 `sequentially_resolved` 原因码 |
| `progress_bar` | 发现 CI 环境变量影响 | 覆盖**静默无警告**；请求值被永久丢弃；生效值构造时定稿 | 可进契约，`how=attr`+构造实参，强制 `forced_by_env` 原因码 |
| `extra_fields` | 发现 A/B 路径不同 | **无属性**；B 路径无入口且硬编码；同一次 C 运行内两族不同 | 可进契约，`how=call_arg`，**不许**用 `derived` 冒充 |

八项里**只有 5 项满足"属性即请求"**；`chain_method` / `progress_bar` 需要
"生效值 + 构造实参"两条一起记；`extra_fields` 需要 `run` 边界单点观测。
"统一成一张表"可以，"统一成一个 `attr_path`"不行。

---

## 7. 对阶段 B 的接口修订

阶段 A §8 的接口保留，改三处：

**(1) 八项逐条记录**，字段：

```
{ name,
  requested: {value, origin, where}      # origin ∈ {cli, config_file, default, tier, derived}
  resolved:  {value, origin, where}      # 归一化数值；档名只是 tier 标签
  observed:  {value, how, where} | null  # how ∈ {attr, call_arg, derived}
  observed_status: "captured"|"attribute_missing"|"hook_not_reached"
                 |"capture_error"|"not_applicable"
  status, why }
```

**(2) `status` 判定（顺序即优先级，改自阶段 A §8）**

| 序 | 条件 | 状态 |
|---|---|---|
| 1 | `observed_status ∈ {attribute_missing, hook_not_reached, capture_error}` | `incomplete`（不许用 `resolved` 反推） |
| 2 | `observed_status == "not_applicable"` | `derived`，`why="<reason>"`，**跳过值比较** |
| 3 | `observed.value ≠ resolved.value` | `mismatch` |
| 4 | `resolved` 缺失（只知道档名） | `derived` |
| 5 | `resolved.origin == "default"` | `defaulted`（**不是** `derived`） |
| 6 | 其余 | `complete` |

`observed_status` 与 `status` **两个字段都要落盘**：前者说"采到没有"，后者说
"这条值可不可信"。混成一个字段就会出现"采集失败"和"值被改"同码。

**(3) 聚合**：取八项最弱（`incomplete > mismatch > derived > defaulted > complete`），
明细一条不许省；另记 `n_not_applicable`、`reasons: {字段: 原因码}`。
`chain_method`/`progress_bar` 的原因码（`sequentially_resolved` / `forced_by_env`）
**不改任何状态**，只增记——值可能完全正确，但执行条件不等价。

**反例矩阵扩到 8 项（15 格，第 5 格仍是核心）**

| # | 场景 | 期望 |
|---|---|---|
| 1 | 八项全显式 | 五项 `complete`；`max_tree_depth` `complete`；`chain_method` `complete`+`sequentially_resolved`；`progress_bar` `complete`；`extra_fields` `complete` |
| 2 | 八项全走默认 | 逐项 `defaulted`，**不等于** `derived` |
| 3 | 部分显式 | 逐项不同，聚合取最弱 |
| 4 | `--quick` 改三个数 | 记**改后**值，`origin≠default` |
| 5a | 人为改五参数中任一传递值 | 该项 `mismatch`，其余 `complete` |
| 5b | 人为改 `chain_method` 请求（传 `vectorized`） | 属性读回 `vectorized` ⇒ `complete`；若被设备数改 ⇒ `mismatch`+原因码 |
| 5c | 人为改 `progress_bar` 请求 + 置 `CI=1` | 属性 `False` ⇒ 与请求不符时 `mismatch`；不得因"环境覆盖"就静默放过 |
| 5d | 人为改 `extra_fields` 请求（去掉 `num_steps`） | `call_arg` 观测到 `()` ⇒ `mismatch`；`derived` 路径也须报同一个差异 |
| 6 | `observed` 缺失 | `incomplete`，且**不出现** `resolved` 冒充 |
| 7 | 属性改名（模拟库升级） | `attribute_missing` → `incomplete` |
| 8 | 同工作流两次不同预算 | 两条独立记录，按 C-3 `event_id` 关联 |
| 9 | `num_chains=3` + 1 设备 | 五项 `complete`，附 `sequentially_resolved` |
| 10 | 档里缺键 ⇒ 落到函数默认 | `defaulted` 且 `origin` 能指到**哪个函数**的默认 |
| 11 | 内核换 HMC | `max_tree_depth` = `not_applicable`→`derived`，**不是** `complete=10`，**不是** `mismatch` |
| 12 | `num_chains=1` | `chain_method` = `not_applicable` |
| 13 | B 路径声明 `num_steps` | `extra_fields` 实为 `("diverging",)` ⇒ `mismatch`（声明与生效不符） |

第 5a–5d 四种改法各改一处：只测"改五参数之一"会漏掉三种执行条件**根本不在同一条路上**。

---

## 8. 边界（与阶段 A 一致，未扩）

- 不写 `sampler_provenance.py`，不接 `exec_identity.py`，不接 runner；
- 不比较 `sequential` 与 `parallel` 的数值等价性（未测，不写等价）；
- 不修 `bm_real.py:1172` 的 `sampling` 块（仍读 `kw`）、不修 `tree_depth_stats(mc, kw[...])`
  （同样读声明值不读生效值）——**列出来是给你决定要不要在 ③ 之后单独修**；
- 不动 K3（`evidence/m35_cond_n2.py`），不裁 `reference_only`，不做签名。

另有一条**范围外但会影响 `defaulted` 判定**的事实，先记账不处理：
`bm_real.KW` 是模块级**可变 dict**，`evidence/ws_crit/wp2_run.py:34-36`、
`evidence/stage0_three_spectra/scripts/fit_family_compare.py:69`、
`docs/round_2026-11-13/coating_voigt_pilot.py:113` 都在进程内直接改它
（`n_warmup` / `max_tree_depth` / `target_accept`）。所以 A 路径的
"`origin=default`"**不能指向源码常量**——`kw=dict(KW)` 拷到的是被改过的进程状态。
`resolved` 的值仍然对（拷的就是实际要用的），但 `why` 不能写"源码默认 3000/13"。

---

## 9. 复现

```bash
# 观测位置与内核适用性（4 步采样，无产物）
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-14/exec_fields_probe.py
```

读码入口：`bm_core.py:605/606/608` → `bm_top.py:1001/1005-1006` →
`nested_fit.py:82/104-112/217` → `bm_real.py:97-99/1172` →
`bm_select.py:127/203-207/345/391-394/896-910` →
`numpyro/infer/mcmc.py: MCMC.__init__ / MCMC.run` → `HMC.__init__` → `NUTS.__init__`。
