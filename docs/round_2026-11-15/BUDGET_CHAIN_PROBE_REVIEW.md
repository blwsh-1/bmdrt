# ③ 第 3 步：独立探针的正式代码审查报告

> **审查基线**：远端 `origin/main` = `8015721ddfb8157edc4712723294d0d7d085ac13`（第 3 步探针）。
> 本轮按用户要求**重新读取远端 `main`**、以该提交为起点核对实际文件与提交关系，
> **不直接启动第 4 步**。
> **审查性质**：只读。不修改 `8015721` 的任何既有文件（探针 / 报告 / 生产 / 身份层一律不动），
> 不写产物、不构造 `run_id` / `workflow_id`、不启动 runner、三处旧挂账不动。
> **新增文件**：本报告 + `budget_chain_probe_review.py`（静态核对脚本，纯只读）+ `INDEX` 同步。
> **本轮 errata（用户裁决 2026-11-15）**：F1–F3 **全部订正**，权威文本见 §3.1；F4 / F5 按裁决本轮不处理。
> 订正只写在本审查报告 + `docs/INDEX.md` + 项目日志；`8015721` 的被审 blob（含
> `BUDGET_CHAIN_PROBE_REPORT.md`）**字节保持不变**，其 §2 / §3 对应原句按 §3.1 视为作废。
> 订正不改探针 / 生产代码 / 身份层 / 测试，不改变 C1–C8 实测判定与 `upstream_origin_captured=False`。
> **结论一句话**：四项重点检查的**实测部分成立**，但结论必须限定在 C1–C8 覆盖的生产路径
> 与测试工况内；预算的数值传递链已有独立实测证据，**预算来源溯源仍未闭合，
> 完整执行身份仍不可生成**。

---

## 0. 审查结论速览

| 用户指定的重点检查 | 审查结论 | 主要证据 |
|---|---|---|
| 探针是否**真正独立** | **成立**（另有一条护栏强度边界，已用额外静态证据闭合） | B 组 8 条断言；复跑 `sys.modules` 新增 880 模块、产出层加载数 0 |
| 埋点是否覆盖**声明到构造的真实链路** | **必须分两层表述**：④–⑦ 是真实生产 hook；①–③ 是探针侧标注 / 等价复现，**不是 CLI 入口埋点** | C 组 7 条断言 + F2 |
| 注入是否**只改变允许的两项** | **成立** | D 组 6 条断言；`mcmc_init` 唯一改动 `eff.update(MICRO)` @168 |
| 报告八条 case 是否与**测试实现逐一对应** | **成立**（含两处「观察陈述 ≠ 断言」边界） | E 组 15 条断言；逐条核对见 §2.4 |

审查脚本 `budget_chain_probe_review.py` 共 **51 条机械断言、FAIL 0、rc=0**；
探针复跑 **rc=0、8/8 case + 辅助核验全 PASS**。同时固化 **5 条文档/实现对应性发现**
（F1–F5，见 §3），其中 F1–F3 是**表述 / 范围问题，不是实测结论错误**，
**已按用户裁决在 §3.1 给出权威订正文（errata）**；F4 / F5 本轮不扩大处理。

**两个关键事实复核成立**（§5）：C8 的声明值 / 执行值分叉是真因果，不是探针写死的两个数字；
A / B / C 的同名档数值确实不同、`quick` 确实没有统一覆盖五项。

---

## 1. 审查基线与提交关系（A 组）

`git ls-remote origin main` 与 `git fetch origin main` 实测：

```
8015721ddfb8157edc4712723294d0d7d085ac13  refs/heads/main
```

提交链（无后续提交，`HEAD` = 本地 `main` = `origin/main`）：

```
8015721  第 3 步 五项预算传递链独立探针        ← 本次审查对象
a4a4c59  第 2 步 budget_params v1.4 产出层    ← 8015721 的父提交（实测一致）
2416062  第 1 步 上游来源链只读审计
5609ef2  ③ 阶段 C 捕获层接入生产
74324ce  阶段 B 自检补严
```

- `8015721` 相对父提交**只改 3 个文件**：`docs/INDEX.md`（+1/-1）、
  `BUDGET_CHAIN_PROBE_REPORT.md`（+199）、`budget_chain_probe.py`（+533）；
  合计 **+733/-1，零生产代码改动**。

> **基线自指说明**：审查执行时 `origin/main` 恰为 `8015721`；本审查报告提交后
> `origin/main` 会前进，但审查对象始终是 **`8015721` 这个提交对象及其 blob**。
> 审查脚本因此用 `merge-base --is-ancestor` 校验祖先关系、用 `git ls-tree 8015721`
> 取 blob，而**不是**断言「当前 main == 8015721」——否则脚本会被自己的提交判 FAIL。
> （该修正在 `7c2c309` 之后的修订提交中落地。）

- 审查执行时 `8015721` 的 blob 与本地工作树字节**逐位一致**（`sha256` 前 16 位）：

| 文件 | blob oid | blob sha256(前16) |
|---|---|---|
| `budget_chain_probe.py` | `64dafc0f` | `7fad65bcf6c75430` |
| `BUDGET_CHAIN_PROBE_REPORT.md` | `ef26307b` | `205a9dec329f4dae` |
| `docs/INDEX.md` | `c82f84ad` | `e286eb76f9a98806` |
| `budget_params.py`（第 2 步，未改） | `dc6ae77f` | `e45c6975fa9e4429` |
| `BUDGET_PARAMS_CONTRACT.md`（未改） | `7e9b43a6` | `b74230abf58c687a` |

> `docs/INDEX.md` 一行在**本审查报告提交后**会追加审查条目，故脚本不拿它与本地字节比，
> 改核「`8015721` 相对 `a4a4c59` 的 INDEX 单行 diff 是否新增第 3 步探针条目」。
> 表中 `c82f84ad` / `e286eb76ebf…` 是**审查时刻** `8015721` 的 blob 值，保持不变。


用户此前说明「尚未独立读取 `a4a4c59` 远端代码，正式审查以最新远端 `main` 为准」——
本轮按此执行：从远端重新核了整条提交链与上述 blob，**没有把用户报告当作已审过**。

---

## 2. 四项重点检查

### 2.1 独立性：成立

| 断言 | 结果 |
|---|---|
| 探针 AST 全节点（Import / ImportFrom / Name / Attribute）零 `budget_params` 标识符 | PASS（`[]`） |
| 无 `__import__` / `import_module` 动态导入（字符串形式绕过 AST 标识符扫描的常见手法） | PASS（`[]`） |
| 产出层 20 个公开函数与探针调用名交集为空（排除探针自定义函数后） | PASS（`[]`） |
| 顶层生产 / 工具 `.py` 零 `budget_params` import（仅 `test_budget_params.py` 引用） | PASS |
| 既有身份层 `exec_identity.py` 不 import 产出层 | PASS |
| 探针 `KEYS` 与产出层 `BUDGET_KEYS` 静态对齐 | PASS（五键逐位相同） |
| `sys.modules` 差集护栏存在、`projection_check` import 的是 `exec_identity` | PASS |

**一条护栏强度边界，审查时补证**：`before = set(sys.modules)` 位于 `main()` 内，
因此探针**模块级** `import bm_core / bm_real / bm_select / bm_top / nested_fit`
这一条 import 链不在差集覆盖范围内。若产出层被生产模块间接 import，差集护栏会漏。
本轮用两条独立静态证据闭合：① 顶层生产 / 工具 `.py` 全部零产出层 import；
② `exec_identity.py` 也不 import 产出层。加上复跑实测新增 880 模块、产出层加载数 0，
可判定「探针模块级 import 链没有污染」。`_assert_independent()` 本身只拦标识符节点，
不拦字符串动态导入；但审查确认探针未使用动态导入，且差集兜底能拦住 `main()` 之后的动态加载。

`_assert_keys_aligned()` 只 `ast.parse` 读 `BUDGET_KEYS` 的**形状**，不执行产出层、不取运行值；
`KEYS` / `CTOR` 由探针自己写死。辅助核验 `projection_check()` 用的是
**探针采集的 C2 真实构造实参**（`p.last_real`），喂给既有身份层 `E._budget_payload()` /
`E._chk_budget()`；探针全文出现 `budget_hash_payload` **0 次**（见 F4）。
⇒ **采集值可作为外部基准，不构成自证。**

### 2.2 埋点覆盖：真实 hook 覆盖 ④–⑦；①–③ 是标注 / 等价复现

**真实 hook 覆盖（实测成立）**：

- `WRAPS` 恰 5 个生产函数站点：`fit_voigt` / `fit_nested` / `fit_one` / `bm_core.fit` / `bm_top.fit_top`。
- 生产源码确以**模块属性**调用：`bm_select.py:346 R.fit_one(`、`bm_real.py:964 B.fit(`、
  `bm_select.py:389` 与 `nested_fit.py:339 T.fit_top(` ⇒ 替换模块属性的 wrapper **真拦得到**。
- 两处 `MCMC(...)` 构造点（`bm_core.py:610` / `bm_top.py:1004`）**全用关键字实参**
  （`num_warmup` / `num_samples` / `num_chains` / `progress_bar`，NUTS 侧
  `target_accept_prob` / `max_tree_depth`）⇒ `mcmc_init(obj, sampler, **kw)` 收得到全部值。
- 类级 hook 覆盖 `MCMC.__init__` / `MCMC.run` / `NUTS.__init__` / `HMC.__init__`；
  全部用 `functools.wraps` 保住签名；`close()` 还原 4 个类属性与模块函数。
- ⑤ 层用 `bind_partial().apply_defaults()` 得到**真实绑定**并列出「签名补齐」键；
  ⑥ 层是注入前真实构造实参；⑦ 层读 `mc.*` / `mc.sampler._*` 属性。
- C4 / C7(b) 的 ④ 层为 `budget_kw()` + `fit_top` 的**等价展开**，报告 §2 已明账
  「薄封装会写产物，未调用」——**用户边界 2 成立，这个限制继续保留**。

**订正后的表述（F2，权威文本见 §3.1）**：①–③ **不是生产入口埋点**。

- ① 全部是探针注入的档名 / 开关标注，实测 **11 处 `p.mark(...)` 传空 dict**；
  探针未调用 `bmdrt.py` 的 CLI 合并，也未调用 `bm_real.run()` 批处理入口。
- ② 是真实生产常量与 `nested_fit.budget_kw()` 的真实输出（这一层是实况）。
- ③ 的 quick 覆盖是探针在 `case_C3` 里**照抄** `bm_real.py:2771` 的 `kw.update(...)` 与
  `:2774` 的 `kw["target_accept"]`（代码注释标「原文」），是**等价复现**，不是 hook 在
  `bm_real.run()` 内取到的值。报告 §2 把该层写成「`bm_real.py:2771 kw.update(...)` 原文三行」
  容易被读成生产内埋点。

**未覆盖的入口（明账）**：`bmdrt.py` CLI / `DEFAULTS` 合并、`bm_real.run()` 批处理入口、
`nested_fit.fit_one`（会写产物）、`bm_top_gen`、任意脚本直构 `MCMC` 的方式，以及
**K3 = `evidence/nested_topology/m35_cond_n2.py:116`**（阶段 A 编号；
`sampler_capture.NON_PRODUCTION_SITES` 明账「独立实验路径，绕过 fit/fit_top 与 BUDGETS，
未纳入生产捕获范围」）。

⇒ 准确说法是：**「生产常量 / 查表输出 → ④⑤ 转发与签名绑定 → ⑥ 真实构造实参 →
⑦ 对象属性」有独立实测证据**；「CLI 声明 → 查表 → 覆盖」是受控等价复现，
**不能写成所有入口都已端到端取证**。

### 2.3 注入边界：成立

- `MICRO = {"num_warmup": 8, "num_samples": 8}`，恰两项、值均为 8。
- `mcmc_init` 函数体内对 `kw` 的**唯一**改动是 `eff.update(MICRO)`（AST 实测
  `mutating=['update@168']`，无 `pop` / `setdefault` / 下标赋值）；
  `kern_hook` 直接 `return orig(obj, model, **kw)`，不改内核 kwargs。
- `outer.ctor_passed = ...` 在第 165 行、`eff.update(MICRO)` 在第 168 行 ⇒ **取证用注入前值**。
- `finish()` 的 `diff` 只在 MCMC 侧三键上计算并断言 `⊆ {n_warmup, n_samples}`；
  NUTS 侧两键用 `obj == real` 交叉；另有 `obj.n_warmup == 8` / `obj.n_samples == 8` 兜底。
  复跑日志每条 case 的注入差异键都是 `['n_samples','n_warmup']`（C6 显式传 8/8 ⇒
  注入不改值，故判定用**子集**而非「恰好两项」）。
- **两种注入层次不同**：类边界微预算只改实际采样步数、不改取证值；
  C8 的 `p.tamper` 改在**转发层**（仅当 `mod is BC and name == "fit"` 生效），会真改传递链。
  两者在日志里分列 ④′篡改 与 ⑦ 注释，没有互相顶替。

⇒ **注入只改变允许的两项，且取证值未被注入污染。**

### 2.4 八条 case 与实现逐一对应：成立

| # | 报告 §3 描述 | 实现（远端 blob） | 对应性 |
|---|---|---|---|
| C1 | 默认值 / 单链，`bm_core.fit` 不传五键，600/1200/1/0.9/10 | `BC.fit(..., K=2, seed=0, progress=False, extra_fields=...)`；`exp` 手写常量 | ✓ 但预期靠 **C7(c)** 的 `inspect.signature` 默认值逐位相等交叉锁定（报告未说明该依赖） |
| C2 | 显式值全链 2000/1000/2/0.9/10 | `std = dict(BS.BUDGETS["standard"])`；真调 `BS.fit_voigt`；`exp` 动态取 | ✓ `sampling` 声明只**打印在 note、不断言**；「本例与构造实参一致」是观察陈述 |
| C3 | quick 覆盖 800/400/2/**0.95**/**13** | 真调 `R.fit_one`；③ 等价复现；末尾断言 `R.KW["max_tree_depth"] == 13` | ✓ |
| C4 | C 路径 standard，**0.98** | `NF.budget_kw("standard")` + `BT.fit_top`（④ 等价展开明账）；ta 从生产函数取、未硬断言 | ✓ 限制保留 |
| C5 | 挑键转发 | 真调 `BS.fit_nested(f, Z, 1, kw=std)` | ✓ |
| C6 | 环境覆盖，五项不变、`progress_bar` 实参/属性分叉 | `os.environ["CI"]="1"` + `BC.fit(..., progress=True)`；附加断言 `ctor_passed.progress_bar != mcmc_obj.progress_bar` | ✓ |
| C7 | 缺键三态 | (a) 真调 `BS.fit_nested` 抓 `KeyError('max_tree_depth')`；(b) 临时档 `NF.BUDGETS["_probe_partial"]` + `finally` 还原；(c) 静态签名检查并明账引用 C1、不重跑 | ✓ 三态齐全 |
| C8 | 转发层 2000→16，`sampling` 记 2000、构造实参 16 | `p.tamper = lambda kw: {**kw, "n_warmup": 16}`；`diverged` 三项硬断言 | ✓ |

汇总：`main()` 恰好 9 条 `results`（8 case + 1 辅助核验），`ok = all(v for _, v in results)`
决定退出码，**无硬编码 `return True`**；报告 §3 表格恰 8 行、全部关键实测数值都能在实现里找到。
辅助核验在报告与实现里都标明「不是传递链证据 / 不构造任何号」。

---

## 3. 审查发现与 F1–F3 订正裁决（errata）

| 编号 | 级别 | 发现 | 处置建议 |
|---|---|---|---|
| **F1** | P1 表述错误 | 报告 §2 写「每条 case 都 `reset()`、`close()` 还原全部埋点」。实现**没有 `reset()` 方法**（方法只有 `begin` / `finish` / `close`），`p.close()` 全文**调用 1 次**（`main()` 的 `finally`）。实际是 hooks 在 `Probe.__init__` 安装一次、每条 case 用 `begin()` 重置账本。行为无缺陷，但描述与实现不符。另：`begin()` 不重置 `tamper` / `micro` / `last_real`（`tamper` 由 C8 `try/finally` 清、`micro` 恒 True、`last_real` 每条 `finish()` 更新）。 | **已订正 → §3.1 F1**：按实际实现给出权威文本，删除不存在的 `reset()` 描述 |
| **F2** | P1 范围表述 | ①–③ 层是探针侧标注 / 等价复现，非生产入口埋点（§2.2）。报告 §2 表格 ③ 的位置写法易被读成生产内 hook。 | **已订正 → §3.1 F2**：分层明账（①–③ 标注 / 等价复现，④–⑦ 真实 hook）+ 逐项列出未覆盖入口 |
| **F3** | P1 扩大化（正对应用户边界 1） | 报告 §3 写「五项**总是**被生产 fit 显式传给 `MCMC` / `NUTS`」。这是 C1–C8 覆盖的**两处生产构造点 + 五条转发路径**上的实测，不宜扩大成所有入口、所有调用方式；K3 未纳入。 | **已订正 → §3.1 F3**：限定 C1–C8 路径与工况，明账 K3 独立构造路径未纳入、禁止外推 |
| **F4** | P2 引用澄清 | 报告 §0 的「`budget_hash_payload()` 与 v1.4 投影一致」是**第 2 步结论的引用**；本轮探针全文 `budget_hash_payload` 出现 **0 次**，只调了既有 `E._budget_payload()`。 | **按用户裁决本轮不处理**：报告未声称探针直接验证该函数，非需补代码的缺陷（原引用澄清保留） |
| **F5** | P2 状态确认 | `chain_method` 未登记 `RUN_ID_CONFIG_KEYS` ⇒ Q6 裁决仍未产生 `run_id` 影响。 | **按用户裁决本轮不处理**：属第 4 步复裁输入，不提前改身份层 |

另有两处**不影响结论**的健壮性 / 整洁边界（P3）：C7(b) 的 `b_lost` 在 `try` 内赋值，
若 `budget_kw()` 抛异常会在 `return` 处 `UnboundLocalError`（本次 rc=0 未触发）；
`bm_core.py:613` 有一行阶段 C 改注释时留下的重复注释残留，不改行为、非本轮范围。

**订正方式与效力**：用户已裁决 F1–F3 全部订正。订正文只写在本审查报告 §3.1 + `docs/INDEX.md`
+ 项目日志；`8015721` 的任何既有文件（含被审 `BUDGET_CHAIN_PROBE_REPORT.md`）**字节保持不变**——
这既是为守住「审查对象 = 远端提交对象及其 blob」的 A 组断言（51/51 仍成立），也是为避免把
文档订正混入代码改动。因此 §3.1 采用 **errata** 形式：被审报告 §2 / §3 对应原句按 §3.1
视为删除 / 作废，一切后续引用以 §3.1 为准。订正不改变 C1–C8 的实测判定、
不改变 `upstream_origin_captured=False` 的结论、不构造任何号。F4 / F5 按裁决本轮不扩大处理。

### 3.1 F1–F3 权威订正文（用户裁决落地）

> 适用范围声明：以下三条**不是新的实测**，只是把既有实测的证据边界写准。凡与
> `BUDGET_CHAIN_PROBE_REPORT.md`（`8015721` blob）§2 / §3 表述冲突处，**一律以本节为准**。

#### F1 — 探针生命周期：按实际实现描述

- **hooks 安装一次**：`Probe.__init__` 调用 `_install()` **一次**，装 4 个类级 hook
  （`MCMC.__init__` / `MCMC.run` / `NUTS.__init__` / `HMC.__init__`）与 5 个模块函数
  wrapper；所有 case 共用这一组 hooks，不逐 case 重装。
- **每条 case 用 `begin()` 重置账本**：`begin(label, expected)` 重置 `rows`、
  `ctor_passed`、`ctor_effective`、`kern_passed`、`run_passed`、`mcmc_obj` 并置
  `active=True`；`finish()` 置 `active=False` 并更新 `last_real`。
  **`begin()` 不重置** `tamper` / `micro` / `last_real`：`tamper` 由 C8 的
  `try/finally` 清除，`micro` 恒 `True`，`last_real` 由每条 `finish()` 更新。
- **`close()` 的真实调用范围**：全文仅 `main()` 的 `finally` 中 `p.close()` **调用 1 次**，
  统一还原 4 个类属性与 5 个模块函数；**不是每条 case 调一次**。
- **实现中不存在 `reset()` 方法**（方法集只有 `_install` / `close` / `begin` / `mark` / `finish`）。
  被审报告 §2 末句「每条 case 都 `reset()`、`close()` 还原全部埋点」中的
  **`reset()` 描述删除、该句作废**；权威替换文本为上面三条。

行为无缺陷，仅表述与实现不符；订正不影响 C1–C8 判定。

#### F2 — ①–③ 与 ④–⑦ 分层，逐项列出未覆盖入口

| 层 | 取证性质 | 依据 |
|---|---|---|
| ① 入口声明 | **探针侧标注**（档名 / 开关，不当数值证据） | 11 处 `p.mark(...)` 传空 dict；未调 `bmdrt.py` CLI 合并、未调 `bm_real.run()` |
| ② 查表 | **实况**（真实生产常量与函数输出） | `bm_real.KW` / `bm_select.BUDGETS` / `nested_fit.budget_kw()` |
| ③ 覆盖 | **等价复现**（不是生产内埋点） | `case_C3` 照抄 `bm_real.py:2771` 的 `kw.update(...)` 与 `:2774` 的 `kw["target_accept"]`，非 `bm_real.run()` 内 hook 取值 |
| ④ 转发 | **真实生产 hook** | 5 个模块函数 wrapper 的 `kw` 入参快照；C4 / C7(b) 的 ④ 是 `budget_kw()` + `fit_top` 的**等价展开**（薄封装会写产物，未调用） |
| ⑤ 签名默认 | **真实生产 hook** | `bind_partial().apply_defaults()` 的真实绑定 + 「签名补齐」键 |
| ⑥ 运行时 | **真实生产 hook** | 类级 hook 取真实构造实参 / run 实参 |
| ⑦ 观测 | **真实生产 hook** 后读属性 | `mc.*` / `mc.sampler._*` |

⇒ **④–⑦ 才是真实生产 hook；①–③ 是探针侧标注或等价复现**，
「CLI 声明 → 查表 → 覆盖」是**受控等价复现**，不是 CLI 入口埋点。

**未覆盖入口（逐项明账，不得外推）**：

1. `bmdrt.py` CLI 与 `DEFAULTS` 合并；
2. `bm_real.run()` 批处理入口；
3. `nested_fit.fit_one`（唯一会写产物的薄封装，C4 / C7(b) 只调其公开的 `budget_kw()` 查表 + 生产 `bm_top.fit_top`）；
4. `bm_top_gen`；
5. 任意脚本直接构造 `MCMC` 的方式；
6. **K3 = `evidence/nested_topology/m35_cond_n2.py:116`**（阶段 A 编号；
   `sampler_capture.NON_PRODUCTION_SITES` 明账「独立实验路径，绕过 fit/fit_top 与 BUDGETS，
   未纳入生产捕获范围」）。

⇒ 准确说法：**「生产常量 / 查表输出 → ④⑤ 转发与签名绑定 → ⑥ 真实构造实参 → ⑦ 对象属性」
有独立实测证据**；**不能写成所有入口都已端到端取证，更不能写成整个生产系统已完成预算捕获**。

#### F3 — 五项显式传递：限定路径与工况

被审报告 §3「五项**总是**被生产 fit 显式传给 `MCMC` / `NUTS`，不落 numpyro 库默认」
中的 **「总是」删除、该断言作废**；权威替换文本：

> **在本探针 C1–C8 覆盖的路径与测试工况内**——即两处生产构造点
> （`bm_core.py:610` / `bm_top.py:1004`）与所列五条转发路径
> （`fit_voigt` / `fit_nested` / `fit_one` / `bm_core.fit` / `bm_top.fit_top`），
> 且仅在八条 case 设定的工况（默认值 / 覆盖 / 挑键 / 环境覆盖 / 缺键 / 人为篡改）下——
> 五项预算被显式以关键字实参传给 `MCMC` / `NUTS`，不落 numpyro 库默认；
> 只有 `chain_method` 从不传 ⇒ 读库默认 `parallel`（多链时由设备数派生，
> 单链按阶段 B R4 标 `not_applicable`）。

- **K3 独立构造路径尚未纳入**：`m35_cond_n2.py:116` 绕过 fit / fit_top 与 `BUDGETS`，
  本探针**没有**为它取证；它是否满足「五项显式传递」**未知**，不得据 C1–C8 推断。
- 其余未覆盖入口见 F2 的六项清单；**不得外推到全部生产入口、全部调用方式或任意运行时工况**。
- 该项只描述**数值传递链**，不构成 `origin` 来源证据，也不使 `source=recorded` 成立。

#### 订正不改变的部分

`upstream_origin_captured=False`、无一项标 `recorded`、不构造 `run_id` / `workflow_id`、
执行身份链不整体冻结、三处旧挂账不动、C1–C8 八条 PASS 与辅助核验结论——**全部不变**。
F4（`budget_hash_payload` 引用澄清）与 F5（`chain_method` 未登记）按用户裁决本轮不扩大处理，
F5 作为第 4 步复裁输入。

> **与审查脚本打印的关系（明账）**：`budget_chain_probe_review.py` 末尾仍会打印
> 「F1 报告 §2 写 reset()/close()」「F3 含扩大化措辞 = True」。这些打印描述的是
> **被审对象 `8015721` 的历史字节**，按本轮裁决**不改动**；它们与被审 blob 的
> A 组字节对齐断言同属「审查时刻快照」，不是现行表述。因此打印保持原样是预期行为，
> 与 §3.1 的 errata **不矛盾**：§3.1 管后续引用口径，脚本管历史 blob 取证。
> 本轮未改该脚本（用户限定「测试一律不动」），复跑 **51 条机械断言 FAIL 0、rc=0**。

---

## 4. 用户三点边界的落地核查

| 用户边界 | 审查核实 | 结论 |
|---|---|---|
| **五项显式传递只是受控路径实测结论，不宜扩大；K3 未纳入** | 探针只挂 5 个模块函数 + 2 处生产构造点；`NON_PRODUCTION_SITES` 明账 K3 未纳入；`bmdrt.py` CLI、`bm_real.run()`、`nested_fit.fit_one` 均未调用 | **边界成立**，报告 §3「总是」已按 §3.1 F3 订正作废 |
| **C4 / C7(b) 等价展开不替代 `fit_one` 端到端证据** | 报告 §2 明账「薄封装会写产物，未调用」；实现确实只调 `budget_kw()` + `fit_top`，④ 层标「等价展开」 | **边界成立**，限制继续保留 |
| **辅助投影核验只证明接口兼容，不能改 `True`、不能生成号** | `projection_check()` 用探针采集值喂 `E._budget_payload()` / `E._chk_budget()`，五项 `source="derived"`；探针不构造号、不改身份层、不改清单；报告 §0 / §3 / §4 三处保留 `False` | **边界成立** |

补充精确后果（第 2 步已锁、本轮复核）：v1.4 中只要五项不全为 `recorded`，
`E.evaluate()` 判 `fatal` ⇒ 整体 `unconfirmed`，`run_id` / `workflow_id` **都是 `None`**，
不是「降 `derived` 发号」。

---

## 5. 两个关键事实的复核

**① 声明值与实际执行值确实可能分叉（C8）**——这是真因果，不是探针写死两个数字：

- `p.tamper` 只在 `BC.fit` wrapper 生效，用 `{**kw, "n_warmup": 16}` 生成**新 dict**，
  因此 `R.fit_one` 收到的 `kw` 原字典**没有被改**。
- `bm_real.py:1172` 的 `sampling=dict(n_warmup=int(kw.get("n_warmup", 0)), ...)` 读的正是这个
  `kw`；其上方注释还写着「实际生效的采样配置（不是命令行请求值）」。
- 日志实证：`④'篡改` 层 16、`⑥构造实参` 16，而 `fit_one.sampling` 仍
  `{'n_warmup': 2000, ...}`。⇒ **现有 `sampling` 块不能当作预算身份来源**，
  这一条同时把三处旧挂账之一 `bm_real.py:1172` 的问题从"静态怀疑"升级为"实测证伪"。
  另两处挂账（`tree_depth_stats(mc, kw[...])`、`fit_top` 缺 `extra_fields` 入口）本轮未动。

**② 同名预算档不是稳定的执行条件**：

| 路径 / 档 | n_warmup / n_samples / num_chains / target_accept / max_tree_depth |
|---|---|
| A 默认 `bm_real.KW` | 3000 / 1000 / 3 / 0.9 / **13** |
| A `--quick` + `--ta 0.95` 覆盖后（C3 实测） | 800 / 400 / 2 / **0.95** / **13** |
| B `bm_select.BUDGETS["standard"]`（C2 实测） | 2000 / 1000 / 2 / 0.9 / 10 |
| C `nested_fit.budget_kw("standard")`（C4 实测） | 2000 / 1000 / 2 / **0.98** / 10 |

`--quick` 只 `kw.update(n_warmup, n_samples, num_chains)` 三键，`max_tree_depth` 保留 `KW` 的
13；同名 `standard` 的 `target_accept` 是 0.9 vs 0.98（`nested_fit.py` 源码注释明确两套口径）。
⇒ **身份必须依据最终实际传递值，不能用档位名称**（Q1 / Q2 裁决的实证支撑）。

---

## 6. 复现与回归证据

```
python docs/round_2026-11-15/budget_chain_probe_review.py      # 审查静态核对，rc=0，51/51
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/budget_chain_probe.py   # rc=0
```

> 两条命令的解释器要求不同：审查脚本是纯 `ast` + `git` 静态核对，无需 jax，
> 但第 294 行 f-string 内含反斜杠 ⇒ 需 **Python ≥ 3.12**（本轮实测系统 Python 3.13.14 通过）；
> `D:/autoeis_env/Scripts/python.exe` 是 3.10.7，解析该脚本会 `SyntaxError`。
> 探针脚本需 jax / numpyro，只能用 autoeis_env。这是既有环境约束，本轮未改任何脚本。


- 审查脚本：**51 条机械断言 FAIL 0**；A 组基线 8 条、B 组独立性 8 条、C 组埋点 7 条、
  D 组注入 6 条、E 组 case 对应 22 条；另打印 F1–F5 的可复现静态证据。
- 探针复跑：**rc=0、8/8 case + 辅助核验全 PASS**，`sys.modules` 新增 **880** 模块、
  产出层加载数 **0**，结论行 `全部字段对得上`。
- 全套回归：`budget_params.py` 自检通过、`test_budget_params.py` **45/45**、
  `test_exec_identity.py` **68/68**、`test_capture_contract.py` **31/31**、
  `test_patch_bundle.py` **25/25**、`test_sampler_provenance.py` **45/45**、
  `test_sampler_capture.py` **28/28**（拟合环境）、`sampler_capture.py` /
  `sampler_provenance.py` 自检通过；
  `build_source_manifest.py --check rc=0`（零生产改动 ⇒ AST 闭包不变，清单**不用**重生成）、
  阶段 0 冻结 **9/9**、`map_stage0_records --check rc=0`
  （239 / 独立 239 / 严格 237 / 唯一 237）、第 1 步只读探针 `rc=0`。
- 冻结 / 阶段 0 / 两个探针须用 `D:/autoeis_env/Scripts/python.exe`（系统 Python 无 jax 会假报失败）。
  `nested_fit.py:42` 既有注释里的 `\S` 会报一条 `SyntaxWarning`，与本轮无关；审查脚本已忽略。

---

## 7. 阶段状态与下一步

| 阶段 | 判断 |
|---|---|
| 第 1 步：上游来源链只读审计 | 已完成（`2416062`） |
| 第 2 步：`budget_params` 纯函数产出层 | 已完成（`a4a4c59`） |
| 第 3 步：独立探针 | **按报告完成；远端代码审查已完成；F1–F3 已按用户裁决订正（§3.1 errata）** |
| 第 4 步：身份闭集复裁 | **用户已批准开始**（另立复裁报告收口；本报告只收口 F1–F3 文档订正） |
| 第 5 步：最小 runner | **尚未授权、未启动** |
| 三处旧挂账 | 保持不动 |

**本轮没有把任何判断上调**：`upstream_origin_captured` 保持 `False`、无一项标 `recorded`、
不构造任何号、执行身份链不整体冻结。探针是**外部挂点**，`KW` 可变 dict 的运行内修改仍无留痕、
`origin` 七类仍是按路径推断而非生产记录、`sampling` 已被 C8 证伪、`chain_method` 未登记 ⇒
报告 §4 的四处缺口**逐一复核仍成立**。

用户裁决（2026-11-15）：① F1–F3 **全部订正**，已在本报告 §3.1 落地（本轮只做文档修订，
与第 4 步分开收口）；② **批准开始第 4 步身份闭集复裁**，但其范围被限定为
「先冻结事实 → 逐项复裁承载关系 → 专门处理来源闭合 → 交付准入结论」，
**本步不改 `exec_identity.py`、不登记 `chain_method`、不启动 runner**。
`chain_method` 登记 / 升 `RUN_ID_CONFIG_KEYS_VERSION`、`retry_kw` 分工、外部挂点常驻化、
完整号可生成性、跨世代兼容均是复裁的**输入与裁决项**，不是本轮已授权的实现动作。
**最小 runner 在任何情况下仍需单独授权。**

---

> **最终结论（与用户判断一致，不因探针通过而改变）**：
> **预算的数值传递链已有独立实测证据（限 C1–C8 覆盖的生产路径与测试工况），
> 但预算来源溯源仍未闭合，完整执行身份仍不可生成。**
>
> F1–F3 的订正只收紧表述与证据边界，**不改变**上述结论中的任何一项。
