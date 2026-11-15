# ③ 第 3 步：五项预算传递链独立探针报告

> **基线**：`a4a4c59`（第 2 步产出层）。本轮**零生产代码改动**，只新增本目录两个文件 + `INDEX` 同步。
> **取证**：`budget_chain_probe.py`，`D:/autoeis_env/Scripts/python.exe` 下 **rc=0，八条 case + 辅助核验全 PASS**，
> 结论行 `全部字段对得上`。不写产物、不拟合批量、不构造 `run_id` / `workflow_id`。
> **三处旧挂账不动**：`bm_real.py:1172` 的 `kw` 读取、`tree_depth_stats(mc, kw[...])`、
> `fit_top` 缺 `extra_fields` 入口。最小 runner **未授权启动**。

---

## 0. 结论与状态

| 阶段 | 状态 |
|---|---|
| ① 上游来源审计 | 已完成（`2416062`） |
| ② `budget_params` 纯函数产出层 | 已完成（`a4a4c59`） |
| **③ 独立探针** | **已完成（本轮）** |
| ④ 身份闭集复裁 | 待做 |
| ⑤ 最小 runner | 未授权启动 |
| 三处旧挂账 | 保持不动 |

**在受控调用路径下**，五项预算从入口声明、查表、覆盖、转发、签名绑定到 `MCMC` / `NUTS`
真实构造实参，逐层可取证且与构造后对象属性交叉一致；默认值、覆盖、缺键、环境覆盖、
单链不适用、人为篡改传递值六类行为都被实测复现。

**但必须按用户的边界读这份结果**：`budget_hash_payload()` 与 v1.4 投影一致，
**只证明预算身份投影兼容**，不证明上游来源已经闭合，也不证明完整
`run_id` / `workflow_id` 已经可以可靠生成。因此：

- `upstream_origin_captured` **保持 `False`**；五项 `source` 最多只能是 `derived`，
  本轮**没有任何一条记录被标成 `recorded`**；
- 探针不构造任何号，不改 `exec_identity.py`，不改 `source_manifest.json`；
- **执行身份链不整体冻结**，第 4 步仍须复裁。

理由见 §5：探针是**外部挂点**，能证明「这条链上的值没被改错」，
但不能证明「生产任意一次运行都自动留下了同等强度的来源证据」。

---

## 1. 独立性：不自证

用户要求「探针必须从真实生产调用路径取证，不能直接调用产出层的构造函数，
再用构造函数自己的输出证明自己正确」。实现上三重护栏：

| 护栏 | 做法 | 实测 |
|---|---|---|
| 静态对齐 | `_assert_keys_aligned()` 用 `ast.parse` **只读**产出层文件里的 `BUDGET_KEYS` 常量，逐位对齐探针写死的五键 | 通过 |
| 运行时零引用 | `_assert_independent()` AST 扫本文件所有 `Import` / `ImportFrom` / `Name` / `Attribute` 节点，出现产出层标识符即拒跑；并查探针命名空间 | 通过 |
| 加载差集 | 取证前后比对 `sys.modules`，新增 880 个模块中产出层加载数 **0** | 通过 |

探针里 `KEYS` / `CTOR`（预算键→构造形参名）**自己写死**，不与产出层共用常量；
静态对齐只查形状，不取运行值。所有数值只来自五个地方：

1. 生产模块常量：`bm_real.KW`、`bm_select.BUDGETS`、`nested_fit.VALIDATION_STANDARD`；
2. 生产函数真实输出：`nested_fit.budget_kw()`；
3. 挂在**真实生产调用链**上的 wrapper 入参快照；
4. `MCMC` / `NUTS` / `HMC` 构造与 `MCMC.run` 的真实实参（类级 hook）；
5. 构造后 `mcmc` 对象属性 + `inspect.signature` 对生产签名的默认绑定。

本探针**没有调用产出层的任何构造函数、校验器或投影函数**，所以它的采集值可以作为
后续验证产出层的**外部基准**。

### 微预算注入（明账，不是「人为篡改传递值」）

生产档位是 2000/1000 步，真跑几十分钟。探针在 **`MCMC.__init__` 类边界**把
`num_warmup` / `num_samples` 降为 8，只影响实际采样步数：

- `ctor_passed` = 注入**前**的真实构造实参 ⇒ **这才是传递链终点、取证用这个**；
- `ctor_effective` = 注入后真正交给 numpyro 的值 ⇒ 与对象属性交叉核验。

不改任何生产文件、不改五项预算的取证值；探针断言注入差异
**⊆ `{n_warmup, n_samples}`**，且对象属性与注入后构造实参逐项相等。
所以表格里 ⑥ 是 2000/1000、⑦ 是 8/8，这个差值是**探针自己制造并明账**的，
不是生产传递链出错。C8 的「人为篡改」是**另一种**注入（改在转发层），见 §3。

---

## 2. 埋点地图

| 层 | 埋点位置 | 取到什么 |
|---|---|---|
| ① 入口声明 | 探针注入 `--quick` / `--ta` / `budget=` / `CI` | 只记档名、开关，**不当数值证据** |
| ② 查表 | 生产常量 + `nested_fit.budget_kw()` | 档名 → 五键数值 |
| ③ 覆盖 | `bm_real.py:2771 kw.update(...)` / `:2774 kw["target_accept"]` 原文三行 | 覆盖后逐项值 |
| ④ 转发 | wrapper：`bm_select.fit_voigt` / `fit_nested` / `bm_real.fit_one` | `kw` 入参快照（整包 / 挑五键） |
| ⑤ 签名默认 | wrapper：`bm_core.fit` / `bm_top.fit_top` + `bind_partial().apply_defaults()` | 实际绑定值，并列出哪些键是**签名补齐** |
| ⑥ 运行时 | 类级 hook：`MCMC.__init__` / `NUTS.__init__` / `HMC.__init__` / `MCMC.run` | 真实构造实参、真实 run 实参 |
| ⑦ 观测 | 构造后读 `mc.num_*` / `mc.sampler._target_accept_prob` / `_max_tree_depth` | 对象属性 |

两个已验证的工程细节：类级 hook 必须 `functools.wraps`（否则 `inspect.signature`
读不到库默认值）；`NUTS.__init__` 的 `super()` 会再进 `HMC` hook，内核钩子要直接调
原函数并合并 kw，否则递归。每条 case 都 `reset()`、`close()` 还原全部埋点。

**明账边界**：`nested_fit.fit_one` 是唯一会写产物的薄封装，C4 / C7(b) 只调用它公开的
`budget_kw()` 查表 + 生产 `bm_top.fit_top`，把 `**kw` 转发等价展开，并在表格 ④ 层标注
「薄封装会写产物，未调用」。因此这两条 case 的 ④ 层是**等价展开**而非调用实况。

---

## 3. 八条 case 结果

| # | 场景 / 真实调用 | ①→⑥ 逐层最终值 | 判定 |
|---|---|---|---|
| C1 | **默认值 + 单链**：`bm_core.fit` 一个预算键都不传 | 600 / 1200 / 1 / 0.9 / 10；⑤ 层列出「签名补齐」五项 | PASS |
| C2 | **显式值全链**：`BUDGETS["standard"]` → `fit_voigt` → `fit_one` → `bm_core.fit` | 2000 / 1000 / 2 / 0.9 / 10，五层逐位一致 | PASS |
| C3 | **quick 覆盖**：`bm_real.KW` + `run:2771` + `run:2774 --ta 0.95` | 800 / 400 / 2 / **0.95** / **13** | PASS |
| C4 | **C 路径 standard**：`nested_fit.budget_kw("standard")` → `fit_top` | 2000 / 1000 / 2 / **0.98** / 10 | PASS |
| C5 | **挑键转发**：`bm_select.fit_nested` 逐项 `int()/float()` → `fit_top` | 2000 / 1000 / 2 / 0.9 / 10 | PASS |
| C6 | **环境覆盖**：`CI=1` + 请求 `progress=True` | 五项不变；`progress_bar` 构造实参 `True` / 属性 `False` | PASS |
| C7 | **缺键三态** | (a) `KeyError('max_tree_depth')`；(b) 静默丢键→落默认 10；(c) 落签名默认 600/1200/1/0.9/10 | PASS |
| C8 | **人为篡改传递值**：转发层把 `n_warmup` 2000→16 | `sampling` 记 **2000**，真实构造实参 **16** | PASS |

**关键实测**：

- **C3 印证 Q1 举例**：`--quick` 只 `kw.update(n_warmup, n_samples, num_chains)` 三键，
  `bm_real.KW` 的 `max_tree_depth=13` **原样保留**，不因档名写成 `BUDGETS` 的 10。
- **C4 印证同名档不同值**：同为 `standard`，`nested_fit.VALIDATION_STANDARD` 的
  `target_accept=0.98`，`bm_select.BUDGETS["standard"]` 是 `0.9`；两者都在 ② 层并列打印。
- **C7 印证「缺键」不是一种行为**：`fit_nested` 硬索引 ⇒ 拟合根本没启动；
  `budget_kw()` 的 `if k in b` ⇒ 静默丢、`fit_top` 用签名默认补齐；`bm_core.fit` ⇒
  落函数默认。三种反应共存，产出层不能假定缺键统一处理。
- **C1/C6 印证「库默认」边界**：五项**总是**被生产 fit 显式传给 `MCMC` / `NUTS`，
  不落 numpyro 库默认；只有 `chain_method` 项目从不传 ⇒ 读库默认 `parallel`，
  多链时由设备数派生，单链时按阶段 B 的 R4 标 `not_applicable`。
- **C8 是本轮最有价值的一条**：只改 `bm_core.fit` 的转发层 kwargs，`fit_one` 自己的
  `sampling` 仍是 `{n_warmup: 2000, ...}`，而 hooks 看到 `MCMC(num_warmup=16)`。
  探针独立抓到分叉 ⇒ **现有产物 `sampling` 记的是配置声明，不是实际执行条件**，
  正是第 1 步审计的结论；也说明为什么第 2 步的 `value` 必须以最终构造实参为准。

**三扩展字段**（Q6）在每条 case 都单独打印：`chain_method` 构造实参未传 ⇒ 库默认
`parallel`；`progress_bar` 只在 C6 被环境压掉；`extra_fields` 只在 `MCMC.run` 边界取到
（`fit_top` 硬编码 `('diverging',)`）。三者都**没有被塞进五项预算**。

### 辅助核验（不是传递链证据）

用**探针自己采集的 C2 真实构造实参**（不是查表常量）构造一个最小
`{value, source:"derived"}` 块，喂给既有身份层 `exec_identity._budget_payload()`：

- `E._budget_payload(...)` 逐位相等 ✓
- `E._chk_budget(...)` 五项全部 `budget_not_recorded` ✓

这里 import 的是**既有身份层**，不是第 2 步产出层。它只说明投影形状兼容，
并再次印证：只要 `source` 不是全 `recorded`，v1.4 就不支持 `recorded` 身份。
**不构造任何号**。

---

## 4. 为什么 `upstream_origin_captured` 仍是 `False`

探针能逐层取证，靠的是**本进程外部挂点**；生产运行时并没有常驻的采集链。
至少四处缺口未闭合：

1. `bm_real.KW` 是模块级**可变 dict**，外部脚本运行内改它（对应
   `origin=programmatic_override`）没有留痕；`dict(KW)` 快照不带时点也不带来源标记。
2. `origin` 的七类来源在本轮是**探针按路径推断**的，不是生产侧记录的；
   `cli_explicit` 与 `module_constant` 在构造实参上无法反查区分。
3. 三处旧挂账之一 `bm_real.py:1172` 的 `sampling` 已被 C8 实证不是执行条件；
   `tree_depth_stats(mc, kw[...])` 用声明值当上限、`fit_top` 缺 `extra_fields` 入口
   仍未处理。
4. `chain_method` 按 Q6 裁决应进执行条件，但 `RUN_ID_CONFIG_KEYS` **尚未登记**
   （第 2 步已如实写明、本轮未改身份层），裁决目前不产生 `run_id` 影响。

因此**不能**因为「传递链在受控路径下完整」就把整个采样配置标成完整。
本轮没有产出任何 `budget_params` 记录，也没有把任何项标 `recorded`。

---

## 5. 第 4 步（身份闭集复裁）的输入清单，本轮不实现

1. `chain_method` 登记进 `RUN_ID_CONFIG_KEYS` 并升 `RUN_ID_CONFIG_KEYS_VERSION`；
   确认 `config.effective` 落点，使 Q6 裁决真正生效。
2. 复裁 `retry_kw` 与预算块分工（Q5），确认 `workflow_id` 不二次承载预算数值。
3. 决定是否需要把探针这类外部挂点**常驻化**为生产侧运行时采集点，
   以便 `KW` 修改、默认值来源等能支撑 `source=recorded`；这是新决策，不擅自做。
4. 复裁**完整** `run_id` / `workflow_id` 的可生成性——本轮只证明了预算投影兼容。
5. 跨世代兼容复查：registry 0.7.0 扁平块与 v1.4 逐项块不互转、
   `R-/D-` 与 `R1-/D1-` 不比号、`derived_from_tier` 不作证据。

---

## 6. 复现与回归

```
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/budget_chain_probe.py
```

本轮回归（探针完成后跑）：

- 新增探针 **rc=0**，8/8 case + v1.4 投影辅助核验 PASS；
- `budget_params.py` 自检 **29/29**、`test_budget_params.py` **45/45**、
  `test_exec_identity.py` **68/68**、`test_capture_contract.py` **31/31**、
  `test_patch_bundle.py` **25/25**、`test_sampler_provenance.py` **45/45**、
  `test_sampler_capture.py` **28/28**（拟合环境）；
- `build_source_manifest.py --check rc=0`（零生产改动 ⇒ AST 闭包不变，清单**不用**重生成）、
  阶段 0 冻结 **9/9**、`map_stage0_records --check rc=0`
  （239 / 独立 239 / 严格 237 / 唯一 237）、第 1 步只读探针 `rc=0`。

冻结与阶段 0 校验必须用 `D:/autoeis_env/Scripts/python.exe`（系统 Python 无 jax 会假报失败）。
`nested_fit.py:42` 既有注释里的 `\S` 会报一条 `SyntaxWarning`，与本轮无关。
