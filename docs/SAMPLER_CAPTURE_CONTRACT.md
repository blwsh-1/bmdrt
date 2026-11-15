# 采样参数捕获契约（阶段 C 第一轮）v1.0

> 实现：`sampler_capture.py`（采集层）+ `test_sampler_capture.py`（**28/28**）
> 判定：`sampler_provenance.py`（阶段 B，**45/45**；本层一个状态都不自己定）
> 探针：`docs/round_2026-11-15/capture_probe.py`（四条微预算路径，独立取证）
> 上游：`round_2026-11-14/SAMPLER_PROVENANCE_AUDIT.md`（阶段 A）、`_B.md`（补充审计）

---

## 1. 边界

做：

- 在**两处生产构造点**采集八项配置：请求值来自构造实参、生效值来自契约点；
- 把事件挂到 `mcmc.provenance`（`bm_core.fit` 随返回值带出；`bm_top.fit_top` 另放一个
  `"provenance"` 键）；
- 采集失败**留痕**：`capture_error` + `n_capture_errors` + `judged=False`。

不做：

- 不判定（`status` / `observed_status` / `reason_code` 全部来自 `evaluate()`，
  G7 用 AST/正则守着"状态字面量只许有采集中断那一个 `incomplete`"）；
- 不构造 `run_id`、不改 `exec_identity.py`、不产 `budget_params`（§9 澄清「不进号」的边界）；
- 不改拟合行为：不启动批量拟合，不写产物；
- **不修** `bm_real.py` 的 `sampling` 块（读 `kw`）与 `tree_depth_stats(mc, kw[...])`
  （用声明值当上限）。两处继续挂账。

---

## 2. 三个构造点：两处接、一处明账

| 构造点 | 路径 | 处理 |
|---|---|---|
| `bm_core.py:610` | Voigt 族（A 路径） | **已接**，`site="bm_core.fit"` |
| `bm_top.py:1004` | nested 族（B 路径） | **已接**，`site="bm_top.fit_top"` |
| `evidence/nested_topology/m35_cond_n2.py:116` | 独立实验 | **不接**，登记在 `NON_PRODUCTION_SITES` |

"不接"不等于"忘了"：G4 校站点字符串里的行号**真的**指到 `MCMC(`；G5 数生产模块里的
构造点，必须**等于** `CALL_SITES` 的条数，否则报"漏了一个"。

> 旧审计里写的 `bm_core.py:606` / `bm_top.py:1001` 是**接线前**的行号；接上捕获后
> 那两行各下移 4 行和 3 行，现行值就是上表的 `610` / `1004`。

---

## 3. 每项怎么采

| 字段 | `resolved`（请求） | `observed`（生效） |
|---|---|---|
| `n_warmup` / `n_samples` / `num_chains` | 构造实参；没传则读 `inspect.signature(MCMC.__init__)` 的默认值 | `mc.num_warmup` / `mc.num_samples` / `mc.num_chains` |
| `target_accept` / `max_tree_depth` | 构造实参（在**内核**构造上）；没传则读内核签名默认 | `mc.sampler._target_accept_prob` / `_max_tree_depth` |
| `chain_method` / `progress_bar` | 构造实参；**项目一处都没传** ⇒ 库默认 | `mc.chain_method` / `mc.progress_bar`（只存生效值） |
| `extra_fields` | 调用点的**声明值**（`fit` 的形参）；`fit_top` 没有这个形参 ⇒ 无声明 | `MCMC.run` 的实参（`how="call_arg"`） |

`extra_fields` 是唯一"请求与观测不同源"的字段：声明值来自 `fit` 的形参，观测值来自
`run` 实参。两者不同源，这一项的 `mismatch` 才有意义。`fit_top` 硬编码 `("diverging",)`
⇒ 只有观测、没有声明 ⇒ `derived / resolved_missing`。这是 **F-6 的如实记账**：
要闭就得给 `fit_top` 加形参，不在本轮范围。

默认值一律从**库签名**读，不从记忆里写字面量：numpyro 改默认值这里会跟着变。

---

## 4. 两个捕获量必须分开取（实测教训）

| 捕获量 | 取法 | 不能写成 |
|---|---|---|
| `kernel` | `isinstance(sampler, NUTS)` → `"NUTS"`；否则 `isinstance(sampler, HMC)` → `"HMC"` | `hasattr(sampler, "_max_tree_depth")` |
| `adapt_step_size` | `sampler._adapt_step_size`（两内核都有） | `isinstance(sampler, NUTS) and sampler.adapt_step_size` |

三条实测事实：

1. **属性名是 `_adapt_step_size`**。两个内核上 `getattr(k, "adapt_step_size")` 都
   `AttributeError`——契约 §5 旧示例写的就是这个名字，属于"照着写一定报错"。
2. **`isinstance(nuts_instance, HMC)` 是 `True`**（`NUTS` 继承 `HMC`）。所以必须先判
   NUTS；反过来每一条 NUTS 记录都会错标内核，`max_tree_depth` 的适用性跟着错。
3. `adapt_step_size` 与内核类型**无关**：HMC 上 `adapt_step_size=False` 必须读出
   `False`，NUTS 上 `True` 必须读出 `True`（A1–A3 各一格）。

`adapt_step_size_of()` 拿不到值时**raise**，不给默认；只有调用点显式给 `fallback`
才用，且来源标 `call_arg`（A4/A5）。调用点今天都没传 `adapt_step_size`，所以真实值
来自属性（`adapt_step_size_source="attr"`）。

---

## 5. 第一轮的证据边界：`origin=derived` 怎么读

本轮所有"构造实参"来源都记 `origin="derived"`，事件里同时写
`upstream_origin_captured=False`。读法**必须**是这样：

> `status=complete` = **"构造实参 → 对象属性"这一段闭合**，
> **不是**"参数来源链整条闭合"。

上游（`cli` / `tier` / 签名默认）要等第二轮接 `BUDGETS` / `bm_real.KW` 才写得出来。
`bm_real.KW` 是**模块级可变 dict**（`evidence/ws_crit/wp2_run.py:34-36` 直接改它），
所以第一轮**不**把 `default` 读成"源码常量"；唯一用 `origin="default"` 的是
`chain_method` / `progress_bar` 这种"项目从没传过、只可能是库默认"的字段，且
`where` 指向 `numpyro.MCMC.__init__:<param>=<默认值>`。

---

## 6. 失败策略：不打断拟合，也不假装通过

| 情形 | 记录 |
|---|---|
| 某个属性读不到（改名 / 删除） | 该项 `attribute_missing` → `incomplete`；**其余七项照旧有结论**（B9） |
| 上下文不可用（认不出内核、缺 `_adapt_step_size`） | 八项全 `capture_error` / `incomplete`，`judged=False`，`n_capture_errors=8`（B10） |
| 库签名被裸包装、读不出默认值 | 只坏那一项：`capture_error`，**不编默认值**（B11） |
| `MCMC.run` 边界没接 | `extra_fields` → `hook_not_reached` / `incomplete`（B8） |

`judged` 的含义是"这条结论出自 `evaluate()`"。采集中断的记录 `judged=False`，
事件级 `judged` 取八项之合。**没有任何一条路径会把采集中断写成 `complete`。**

> 已知脆弱点（B11 由探针第一版踩出）：`_lib_default()` 依赖
> `inspect.signature(MCMC.__init__)`。任何**没有 `functools.wraps`** 的包装都会让签名变成
> `(obj, sampler, **kw)`，于是"没传的字段"读不到默认值。捕获层的选择是记
> `capture_error`，**不**猜一个 `"parallel"` 糊过去。探针因此必须用 `functools.wraps`。

---

## 7. 探针结论（四条微预算路径，8 warmup + 8 samples）

独立取证方式：探针自己在 `MCMC.__init__` / `MCMC.run` / `NUTS(HMC).__init__` 上挂
类级包装，记录**真正绑定**的实参，再与记录逐字段对照；对象属性由探针自己复读，不复用
捕获层的读取器。

| 路径 | 情况 | 结果 |
|---|---|---|
| A | `bm_core.fit` 单链，`extra_fields=("num_steps",)` | 7 项 `complete`；`chain_method` `derived`/`single_chain`；聚合 **derived** |
| B | `bm_core.fit` 3 链 / 1 设备 | `chain_method` `mismatch` + `sequentially_resolved`；聚合 **mismatch**，`n_override=1` |
| C | `bm_top.fit_top` 单链 | `extra_fields` `derived`/`resolved_missing`；聚合 **derived** |
| D | `bm_core.fit`，`CI=1`，请求 `progress=True` | `progress_bar` `mismatch` + `forced_by_env`；聚合 **mismatch**，`n_override=1` |

四条路径上 **`resolved` 全部等于真实构造实参、`observed` 全部等于真实属性、
`extra_fields` 全部等于 `run` 实参**（探针输出 `全部字段对得上`，rc=0）。

两个值得记住的读数：

- **B 的 `origin` 仍是 `default`**：`chain_method` 被改写成 `sequential` 是 `mismatch`，
  但"请求值来自库默认"这个来源事实不被 mismatch 吸收（M14 的口径）。
- **单链事件聚合到不了 `complete`**：`chain_method` 不适用 → `derived` → 整体封顶在
  `derived`。不适用不能让记录看起来更完整，这正是阶段 B 的裁决。

---

## 8. 已知边界 / 下一步

- `origin` 的上游来源未接（本轮 `upstream_origin_captured=False`）；
- `extra_fields` 在 B 路径没有声明入口（要闭得给 `fit_top` 加形参）；
- 只覆盖两条生产路径；`bm_real.run` 的 A 路径预算装配（`BUDGETS`/`KW`）尚未接；
- 事件还没进结果文件 / `manifest`，也还没进 `run_id`——按约定这一步等捕获稳定后再谈。

    D:/autoeis_env/Scripts/python.exe sampler_capture.py                 # 自检 17 项
    D:/autoeis_env/Scripts/python.exe test_sampler_capture.py            # 28/28
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/capture_probe.py

---

## 9. 接进生产后的连带变化（收口实测，2026-11-15）

阶段 C 之前 `source_manifest.json` 三 Profile = **A 15 / B 19 / C 18** 文件。
`bm_core.py` / `bm_top.py` 模块级 `import sampler_capture`，而 `sampler_capture.py`
模块级 `import sampler_provenance` ⇒ AST 静态闭包多出 2 个项目模块，三份声明集变成
**A 17 / B 21 / C 20**（仍全 `numeric` / `workflow_scope=true`、零降级）。

这不是可选项：`build_source_manifest.py --check` 立刻从 rc=0 变 rc=1，
`test_exec_identity.py` 的 M6「清单同步」与 M8「预览可复算」两格同时失败。
收口做法是**重新生成清单**，不是放宽那两格——放宽等于允许改了生产代码却拿旧
声明集算号。历史文档里 15/19/18 是 v1.2 当时的快照，不回改。

同时把 §1 的「不动 `run_id`」读准：

> 本层**不构造** `run_id`、**不改** `exec_identity.py`、**事件内容不进**身份载荷；
> 但捕获层现在是被生产 import 的代码，其字节**必须**被 `numeric` 层摘要包含——
> 否则改了判定逻辑仍拿到同一个号，正是「静默合并两次不同代码的运行」。
> 本轮 runner 未启动（§14 ⛔），只同步清单，不产任何号。

另一处护栏：`test_sampler_provenance.py` 的 X1「生产零引用」是**文本扫描**
`_PROD` 里的文件名，`bm_core.py` 注释里一句 `sampler_provenance.evaluate()`
就让它从 45/45 掉到 44/45。收口把注释改成「捕获层转交的纯函数判定器」，
**一个护栏都没放宽**；生产只直接连 `sampler_capture`（G6 锁着），
纯函数层仍只经由捕获层被间接引用。

收口后逐条重跑的回归：identity **68/68**、capture **31/31**、bundle **25/25**、
provenance **45/45**、sampler_capture **28/28**、`sampler_capture.py` 与
`sampler_provenance.py` 自检通过、`manifest --check rc=0`、
冻结 **9/9**、`map_stage0_records --check rc=0`（239 / 独立 239 / 严格 237 / 唯一 237）、
探针「全部字段对得上」rc=0。
