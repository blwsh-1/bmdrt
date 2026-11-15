# `budget_params` 上游来源链审计与字段契约审查（只读）

> **基线**：`5609ef2`（阶段 C 用户报告基线）。按约定不回头改阶段 C 已收口的代码。
> **定位**：用户 5 步计划里的**第 1 步**——只审计、只审查契约。产出层（第 2 步）、
> 独立验证探针（第 3 步）、身份边界复裁（第 4 步）、最小 runner（第 5 步）**都不在本轮**。
> **取证**：`budget_source_probe.py`（只读；不构造 MCMC、不 run、不拟合、不写产物，
> **不产出 `budget_params`**），`rc=0`。
> **三处旧挂账继续挂账**，本轮一项都不顺手修：`bm_real.py:1172` 的 `kw` 读取、
> `tree_depth_stats(mc, kw[...])` 用声明值当上限、`fit_top` 缺 `extra_fields` 入口。

---

## 0. 结论

阶段 C 闭合的是**捕获链**：真实构造实参 → 对象属性 / `MCMC.run` 实参，且可被探针独立核验。
本轮确认另一层——**执行条件来源闭合**——至少经过 **6 个可能改值的转换层**，而现有
产物记录（`bm_real.py:1172` 的 `sampling`、`nested_fit.py:383` 的 `config.budget_kw`）
记的是**配置声明/查表结果**，不是实际执行条件。

所以 `upstream_origin_captured=False` 仍是正确且保守的结论：**不能因为第一层完整，
就把整个采样配置标成完整**。runner 写之前必须先产出 `budget_params`，否则记录的
只是"打算怎么跑"，不是"实际按什么条件跑"。

审查中还发现一个**结构性冲突**：registry 0.7.0 与 `exec_identity.py` v1.4 的
`budget_params` **同名但形状不同**，同一份数值喂给 v1.4 校验器直接 `ValueError`
（§5.1）。这是第 4 步身份边界复裁必须先解决的输入。

---

## 1. 六层来源链（每层都可能改值）

①–⑥ 是待闭合的上游链；⑦ 是阶段 C 已闭合的观测层，列在这里只为标出边界。

| 层 | 代码位置 | 改值方式 | 现有记录覆盖吗 |
|---|---|---|---|
| ① 声明 | `bmdrt.py` `_apply_cli` / `bm_real.py` `main` / `bm_select.py` `main` / `nested_fit.py` `main` | `--quick`、`--ta`、`--budget`、`--target-accept`、`--progress`、`config.json` 的 `fit` 段 | 只记档名/开关，**不记数值** |
| ② 查表 | `bm_real.KW`、`bm_select.QUICK/BUDGETS`、`nested_fit.VALIDATION_STANDARD` / `budget_kw()` | 档名 → 五键数值 | `config.budget_kw` 记的是**查表结果** |
| ③ 覆盖 | `bm_real.run:2770`（quick）、`:2773`（ta）、`nested_fit.py:335` | `kw.update(...)` / `kw["target_accept"]=...` | 覆盖后没有逐项留痕 |
| ④ 转发 | `fit_voigt`（整包）、`fit_nested`（挑五键）、`fit_one`（`**kw`）、`nested_fit`（`**kw`） | 丢字段 / 硬编码 | 不记 |
| ⑤ 签名默认 | `bm_core.fit` / `bm_top.fit_top` / numpyro 库签名 | 缺键落到函数默认或库默认 | 不记 |
| ⑥ 运行时覆盖 | numpyro `MCMC.__init__` / `MCMC.run` | 设备数改 `chain_method`、`CI` 改 `progress_bar` | 阶段 C 事件记 `observed`，**不进身份** |
| ⑦ 观测 | 阶段 C `sampler_capture.py` | 属性 / `run` 实参 | **已闭合**（`5609ef2`） |

三层默认值实测（探针 §1）：

| 签名 | 默认值 |
|---|---|
| `MCMC.__init__` | `num_warmup`/`num_samples` **必填**；`num_chains=1`、`chain_method="parallel"`、`progress_bar=True` |
| `NUTS.__init__` | `target_accept_prob=0.8`、`max_tree_depth=10`、`adapt_step_size=True` |
| `bm_core.fit` | `600 / 1200 / 1 / 0.9 / 10`，`progress=True`，`extra_fields=()` |
| `bm_top.fit_top` | `600 / 1200 / 4 / 0.9 / 10`，`progress=True` |

**项目从不传 `chain_method`** ⇒ 它是"库默认 + 运行时派生"，不是一条预算请求。

---

## 2. `config.json` 里没有预算数值（探针 §2）

`config.json` 的 `fit` 段 17 个键中，五个核心预算键**一个都没有**；`bmdrt.py` 的
`DEFAULTS["fit"]` 同样没有。它只提供 `quick` / `ta` / `seed` 三个**开关**。
预算数值全部来自 Python 模块常量（`bm_real.KW`、`bm_select.BUDGETS`、
`nested_fit.VALIDATION_STANDARD`）。

⇒ 用户列的六类来源里，"配置文件"对**预算数值**不产生作用。
`source` 闭集不能写成"来自 config.json"，也不能把"配置声明"当成数值证据。

---

## 3. 五个核心参数：三路径逐项（探针 §3/§4）

`quick` 档最终构造实参：

| 参数 | A `bm_real --quick` | B `bm_select --budget quick`（voigt） | B nested 侧 | C `nested_fit --budget quick` |
|---|---|---|---|---|
| `n_warmup` | 800 | 800 | 800 | 800 |
| `n_samples` | 400 | 400 | 400 | 400 |
| `num_chains` | 2 | 2 | 2 | 2 |
| `target_accept` | 0.9 | 0.9 | 0.9 | 0.9 |
| `max_tree_depth` | **13** | 10 | 10 | 10 |

**同名档不同值**（档名不能当预算身份）：

- `--quick` 只覆盖三键（`bm_real.py:2770`），A 的 `max_tree_depth` 仍是 `KW` 的 **13**，
  B/C 是 **10**；深度 13 vs 10 = 8191 vs 1023 步。
- 两个 `standard`：`BUDGETS["standard"].target_accept = 0.9`，
  `VALIDATION_STANDARD.target_accept = 0.98`（`nested_fit.budget_kw("standard")`
  = `2000/1000/2/0.98/10`）。
- A 默认 `KW` 是 `3000/1000/3/0.9/13`，与 B/C 的 quick 完全不同源。

**缺键反应不对称**（阶段 A 结论，本轮复核）：

| 位置 | 缺键时 |
|---|---|
| `bm_select.fit_nested` | 硬索引 `kw["..."]` ⇒ `KeyError` |
| `nested_fit.budget_kw` | `{k: b[k] for k in FIT_KEYS if k in b}` ⇒ **静默丢键**，落到 `fit_top` 签名默认 |
| `bm_select.fit_voigt` | 整包转发 ⇒ 落到 `bm_core.fit` 签名默认 |

三种反应不同 ⇒ `source` 必须能区分"查表命中"与"静默落默认"，否则 `derived`
会把两类混成一类。

**可变 `KW`**：`bm_real.KW` 是模块级 dict，`evidence/ws_crit/wp2_run.py:34-36` 直接改它；
`run()` / `fit_one()` 每次 `dict(KW)` 快照，运行内不变，但**没有任何记录说明这次
运行时 `KW` 已被外部改过**。阶段 A 已定"`origin=default` 不能指向源码常量"，
本轮补上第二个缺口：**快照时点没有留痕**。

---

## 4. 三个扩展字段

| 字段 | A | B | C | 最终 | 该不该进 `budget_params` |
|---|---|---|---|---|---|
| `chain_method` | 不传 | 不传 | 不传 | 库默认 `parallel`；1 设备多链 → `sequential` | **否**：运行时派生，不是配置 |
| `progress_bar` | `KW.progress=False` | voigt 整包传 `False`；nested 硬编码 `False` | `--progress` | `CI`/`PYTEST_XDIST_WORKER` 再静默压 `False` | **否**：`RUN_ID_CONFIG_KEYS["progress"] = ("bool","display_only")`，不进 `run_id` |
| `extra_fields` | `KW ("num_steps",)` | voigt `("num_steps",)`；nested `("diverging",)` | `("diverging",)` | `MCMC.run` 实参 | **否**：观测配置，且 C/B-nested 硬编码（`fit_top` 挂账） |

三个扩展字段都不在 `BUDGET_KEYS` 里。它们属于**捕获事件 / 配置记录**，
塞进 `budget_params` 就是把观测结果误当执行条件。

特别地：阶段 C 探针的 D 路径（CI 在场）实测到 `progress_bar` 的
`mismatch / forced_by_env`，那是**证据链记录**；
`progress` 在身份表里明确是 `display_only`，所以这条 mismatch 不该让 `run_id` 换号。

---

## 5. 现有 `budget_params` 定义审查（本轮重点）

### 5.1 同名不同形状 —— 实测冲突（探针 §5）

| | registry 0.7.0 | `exec_identity.py` v1.4 |
|---|---|---|
| 形状 | `{"source": "recorded", n_warmup: 3000, ...}` **扁平数值 + 整块 source** | `{n_warmup: {"value":…, "source":…}, ...}` **每键复合块** |
| 校验 | `_check_budget_params()`：`source ∈ {recorded, derived_from_tier, unavailable}` | `_budget_payload()` 要求每键 Mapping；`_chk_budget()` 要求每项 `source == "recorded"` |
| 号 | `R-`/`D-` + 10 hex | `R1-`/`D1-` + 64 hex |

把 registry 合法记录直接喂给 v1.4：

```
ValueError: budget_params['n_warmup'] 必须是 {value, source}
```

⇒ 阶段 0 冻结的 **237 个唯一 run_id**（registry 口径）与未来 runner 的号
**不可直接比对**，否则会误报"重复/漂移"。第 4 步复裁必须先把这个输入问题定下来。

### 5.2 档名进哈希的现成反例

`_check_budget_params()` 在 `derived_from_tier` 时把 `tier` 写进 `bud`，
`_run_id_from()` 又把**整个 `bud`** 哈希 ⇒ 档名进了 `run_id`；而 `candidate_id`
已经哈希 `budget_profile`（档名）。同一个档名被哈希两次，与契约自己写的
"档名不是预算身份"直接冲突。v1.4 的 `_budget_payload()` 只取 `{value, source}`、
不含 `tier`，相对更干净。

### 5.3 `source` 同名不同粒度

registry 的 `source` 是**整块**来源类别；v1.4 的 `source` 是**每项**证据等级
（`_chk_budget()` 逐项查 `recorded`）。产出层不能把两层的 `source` 当成同一个
字段复制；也不能让 `source` 携带档名或完整路径（否则"来源写法"冒充执行条件，
与 `sigma` 复合块的 D-8 教训同构）。

### 5.4 `run_id` / `workflow_id` 的职责交叉（探针 §6）

- `run_id_payload()`：`budget` 五键 `{value, source}` 是预算数值**唯一**载体；
  `RUN_ID_CONFIG_KEYS` **不含**五个预算键（实测确认），没有重复入口。
- `workflow_payload()`：含 `retry_kw`（**数值 dict**）与 `search`/`extras`/`guards`
  ⇒ 重跑预算数值可能**第二次**被哈希（进 `workflow_id`）。
- 阶段 C 事件的 `observed` / `status` / `reason_code` 一律不进身份载荷。

⇒ 第 4 步要裁的是"**哪一层承载预算数值**"，不是"再加一个字段"。

---

## 6. 字段契约草案：**只列待裁决项，本轮不实现**

以下是给第 2 步的输入，不是产出层代码。未经裁决不改 `exec_identity.py` /
`circuit_registry.py`，也不动 `run_id`。

**建议形状（草案）**：五项核心，每项
`{"final": <构造实参>, "requested": <入口声明或 null>,`
`"transformed": <查表/覆盖后的中间值或 null>, "source": <闭集>, "where": <位置>}`；
`profile` 只作展示、`upstream_origin_captured` 单列、`differences` 列出三者不等的项。

| # | 待裁决 | 冲突点 |
|---|---|---|
| Q1 | 五项核心是否就是 `budget_params` 的全部？三个扩展字段放哪 | 我倾向"三个扩展字段留捕获事件"，但要你裁 |
| Q2 | `final` 与 v1.4 `_budget_payload()` 期望的 `value` 键名怎么对齐 | 不对齐就会 §5.1 的 `ValueError` |
| Q3 | `source` 闭集怎么定、进不进哈希 | registry 整块 vs v1.4 每项；`source` 进哈希是两层的既有口径，但不得含档名/路径 |
| Q4 | `requested`/`transformed`/`final` 三者差异**不进哈希**，只 `final`(+`source`) 进 | 否则"来源写法"冒充执行条件 |
| Q5 | `workflow.retry_kw` 的数值与 `budget` 块怎么分工 | 避免预算数值在 `run_id`/`workflow_id` 各哈希一次 |
| Q6 | 运行时覆盖（`chain_method`/`progress_bar`）算不算执行条件 | 现有契约判定不算（`display_only`、不在 `BUDGET_KEYS`）；若要改须先给"数值逐位是否变化"的证据 |

**状态约定**：五项里任何一项缺 `final`，不得标 `recorded`；
`upstream_origin_captured` 在 Q1–Q5 全部裁决并实现前**保持 `False`**。

---

## 7. 明确不动的东西

- 三处旧挂账：`bm_real.py:1172` 读 `kw`、`tree_depth_stats(mc, kw[...])` 用声明值当上限、
  `fit_top` 缺 `extra_fields` 入口。本轮只把它们作为"记录的是配置声明"的证据引用，不改代码。
- `5609ef2` 的 `bm_core.py` 捕获点注释有一处重复措辞（旧句残留 + 新句），
  纯文字、行数不变、不影响行为；按"不回头重复已收口工作"的约定本轮不动，留待统一整理。
- 本轮**零生产代码改动**：只新增 `budget_source_probe.py`（只读）与本报告。

---

## 8. 下一步（按你的顺序，不越界）

1. **第 2 步**：先裁 Q1–Q6，再实现 `budget_params` 产出（只记实际生效预算 + 来源，
   保留请求/转换/最终差异，档名不进身份）。
2. **第 3 步**：独立探针，覆盖默认值、显式值、环境覆盖、单链不适用、缺参、
   人为篡改传递值，与真实构造实参和对象属性交叉核验。
3. **第 4 步**：复裁 `budget_params` ↔ `run_id` / `workflow_id` 的职责（§5.1–§5.4）。
4. **第 5 步**：最小 runner，单次、单候选，身份记录与产物闭环；§14 ⛔ 不启批量拟合。

    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/budget_source_probe.py   # 只读，rc=0
