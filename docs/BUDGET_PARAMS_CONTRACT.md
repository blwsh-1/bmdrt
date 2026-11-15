# `budget_params` v1.4 产出契约（③ 第 2 步）

> **基线**：阶段 C 用户报告基线 `5609ef2`；第 1 步只读审计 `2416062`
> （见 `BUDGET_PARAMS_SOURCE_AUDIT.md`）。不回头改已收口的工作。
> **定位**：用户 5 步计划里的**第 2 步**——只实现 `budget_params` 产出层（纯函数）
> 及其纯函数测试。第 3 步独立探针、第 4 步身份边界复裁、第 5 步最小 runner
> **都不在本轮**。
> **零生产代码改动**：不改 `exec_identity.py` / `circuit_registry.py`、
> 不接生产、不构造任何 `run_id` / `workflow_id`、不写真实运行记录。
> 三处旧挂账继续挂账（§11）。

---

## 0. 结论：这不是字段改名问题

用户审查结论原样固定在这里：

> **两套身份契约把 `budget_params` 用在了不同层级。**
> registry 0.7.0 的**块级** `source` 描述整组预算记录从哪里来；
> `exec_identity.py` v1.4 的**逐项** `source` 描述每个预算值是否有可接受的记录证据。

由此推出三条不可让步的边界：

1. 两种 `source` **不能直接互相转换**；更不能为了让校验器通过，把块级来源机械
   复制到五个参数上——那是把"这组记录有来源"伪装成"五项各自都核验过"。
2. `R-/D-`(10 hex) 与 `R1-/D1-`(64 hex) 属于**不同契约世代**，不拿前缀或摘要
   直接比对来判断历史记录漂移。
3. **预算数值 / 预算来源证据 / 来源追溯过程必须是三个可区分的概念**：
   `value` / `source` / (`origin`, `requested`, `transformed`, `where`)。

阶段 C（`5609ef2`）闭合的是**捕获链**（构造实参 → 对象属性 / `MCMC.run` 实参），
不是**执行条件来源闭合**（入口声明 → 最终构造实参）。所以
`upstream_origin_captured=False` 仍是正确且保守的结论，本轮不上调（§5）。

---

## 1. 七条裁决与落地位置

实现文件 `budget_params.py`（纯函数）、`test_budget_params.py`（正式测试）。

| # | 裁决 | 落地 | 测试格 |
|---|---|---|---|
| 1 | 五项核心是**唯一**预算块；三扩展字段留捕获 / provenance | `BUDGET_KEYS`；`validate_budget_params()` 要求顶层恰好五项，扩展字段与档名一律拒 | C1 / E1 |
| 2 | 统一用 `value`，即最终**实际构造实参** | `HASH_KEYS=("value","source")`；`_check_value()` 拒 `bool`/`None`/`NaN`/`Inf`；A 路径 `--quick` 后 `max_tree_depth=13` 不按档名写成 10 | C1 |
| 3 | `source`=证据等级、`origin`=来源类别，双闭集，只有 `source` 进哈希 | `BUDGET_SOURCE_STATES` 三值 / `BUDGET_ORIGINS` 七值；具体位置放独立 `where` | S1 / S4 / X8 |
| 4 | 只有 `value`+`source` 进 `run_id`；`requested`/`transformed` 只作溯源 | `TRACE_KEYS` 四键不进 `budget_hash_payload()`；`transformed` 必须是**列表**保留多次覆盖顺序 | S2 / T1 / T2 |
| 5 | 预算归 `budget_params`；重跑策略归 `workflow`；`retry_kw` 禁五键 | `check_retry_kw()` | R1–R4 |
| 6 | `chain_method` 暂按执行条件；`progress_bar`/`extra_fields` 不进身份 | `EXECUTION_CHANNELS` + `split_extensions()`（**§7.2 有重要事实修正**） | E2–E5 / X6 |
| 7 | registry 0.7.0 身份保持不变，不跨版本直接比号 | `is_legacy_registry_block()` / `can_compare_ids()` / 拒绝块级字段 | H1–H5 |

---

## 2. 数据形状

```python
budget_params = {
    "n_warmup": {
        "value": 3000,                     # 最终实际传入采样器的构造实参
        "source": "derived",               # 证据等级（本轮上游链未闭合 ⇒ 不能是 recorded）
        "origin": "module_constant",       # 来源类别，不进哈希
        "requested": None,                 # 入口声明；None 不代表值未知
        "transformed": [                   # 列表，保留多次覆盖的完整顺序
            {"stage": "module_constant", "value": 3000, "where": "bm_real.KW"},
            {"stage": "quick_override",  "value": 800,  "where": "bm_real.run"},
        ],
        "where": "bm_core.fit:n_warmup",  # 具体变量名 / 函数名 / 代码位置
    },
    # ... 其余四项同形
}
```

**这是结构示意，不是真实运行记录**——本轮没有采集、没有接生产，不产任何一次
真实执行的 `budget_params`。用户给的示例中 `source="recorded"` +
`origin="module_constant"` + `transformed=[]` 对**可变 `KW`** 恰好不成立：
`bm_real.KW` 会被外部脚本运行内改写、每次 `dict(KW)` 快照又没有留痕
（`BUDGET_PARAMS_SOURCE_AUDIT.md` §3），所以本层把它判为 `derived`。

| 字段 | 语义 | 进哈希 |
|---|---|---|
| `value` | 最终实际构造实参；不是 CLI 原值 / `BUDGETS` 查表值 / `KW` 模块初值 / 签名默认 | **是** |
| `source` | 逐项证据等级：`recorded` / `derived` / `unavailable` | **是** |
| `origin` | 来源类别（七值闭集）；来源解释，不是执行条件 | 否 |
| `requested` | 入口是否显式请求；`None` 须结合 `origin` 才知是函数默认还是库默认 | 否 |
| `transformed` | 转换步骤**列表**，元素 `{stage, value, where?}` | 否 |
| `where` | 具体变量名 / 函数名 / 代码位置，不塞进 `origin` 枚举 | 否 |

`transformed` 用列表而非单值，因为一次预算可能经过：模块常量 → quick 覆盖 →
`target_accept` 命令行覆盖 → 函数默认补齐 → 最终构造实参。压缩成一个中间值
就无法审计"哪一步改了数"。

`v1.4` 的 `_budget_payload()` 只读每项的 `value` 与 `source`，所以四个追溯字段
**不会自动进入哈希**；本层再用 `budget_hash_payload()` 显式投影一次，把这件事
变成可测的（§4）。

---

## 3. 两个闭集，两种职责

```
BUDGET_SOURCE_STATES = ("recorded", "derived", "unavailable")
BUDGET_ORIGINS = ("cli_explicit", "module_constant", "programmatic_override",
                  "function_default", "library_default", "runtime_derived", "unknown")
```

- `recorded`：最终值**及其来源链**都被实际捕获、核验；
- `derived`：最终值已观测，但上游来源链仍有缺口；
- `unavailable`：最终值本身无法可靠取得 ⇒ `value` 必须为 `None`。

`origin` 举例：`bm_real.KW` 初始预算 = `module_constant`；外部脚本改 `KW` =
`programmatic_override`；`fit_top` 缺键落签名默认 = `function_default`；
numpyro 自身默认 = `library_default`。具体变量名、函数名、代码位置放 `where`。

两个闭集都不含路径 / 档名（X8 锁住），因此"来源写法"没有机会冒充"执行条件"。
`source` 进哈希，表示**同样数值是否具有同样的证据等级**；`origin` 不进哈希。

⚠️ 逐项 `source` 的 `unavailable` **不是** v1.4 的整体身份状态。v1.4 整体三态是
`IDENTITY_STATES = ("recorded", "derived", "unconfirmed")`。两个层级不能混读，
这正是用户"两套契约用在不同层级"结论的第二个实例。

---

## 4. 哈希边界（与 v1.4 真实交叉核验）

`budget_hash_payload(block, upstream_origin_captured=...)` 只输出
`{key: {"value":…, "source":…}}`。实测：

- 与 `exec_identity._budget_payload()` **逐位相同**（C2）——这是 Q2 用 `value`
  而不是 `final` 的直接收益，registry 扁平块喂进来会 `ValueError` 的问题在新层
  不可能重现。
- 改 `origin` / `requested` / `transformed` / `where` ⇒ 哈希**逐位不变**（S1/S2）：
  同一最终执行条件不因入口写法不同而换号。
- 同数值但 `source` 从 `recorded` 降为 `derived` ⇒ **换号**（S3）。
- 五个预算键都不在 `RUN_ID_CONFIG_KEYS` ⇒ 预算数值没有第二入口（C3）。

注意这只管 `run_id` 载荷里的 **`budget` 段**，不代表整个 `run_id` 只有这两个键
（还有 `data` / `rng_plan` / `config` / `code` / `env`）。

---

## 5. 硬边界：`upstream_origin_captured`

用户实施前追加的约束原样固定：

> 即使五项最终值都捕获成功，只要 `KW` 的运行时修改、默认值来源等上游链没有被
> 完整追溯，`source` 就不能标成 `recorded`，整体 `upstream_origin_captured`
> 仍然必须是 `False`。

实现要点：

- 四个公开入口的 `upstream_origin_captured` 用**哨兵 `_MISSING`，没有默认值**：
  省略 / `None` / truthy 非布尔一律 `raise`。若默认写 `False`，"忘了传"会被
  读成"已确认未闭合"——与阶段 B `adapt_step_size` 缺参漏洞同构（U3/U4）。
- `source="recorded"` 需**同时**满足：`upstream_origin_captured is True`、
  `origin != "unknown"`、`where` 非空、`value` 有效（U1/U5）。
- U2 是正对照：同一批记录 + `upstream=True` 正常通过，证明 U1 不是一律拒绝
  `recorded`。
- 本轮**没有任何真实记录被标 `recorded`**：产出层不采集，这个布尔由调用方
  显式声明；第 3 步探针之前它保持 `False`，`upstream_origin_captured` 不上调。

---

## 6. 历史兼容（不得强行改写）

| 项 | 裁决 | 落地 |
|---|---|---|
| registry 0.7.0 | **保持原样**，继续负责历史 Stage 0 记录 | 本轮一行 registry 代码都没改 |
| v1.4 | 负责**未来**执行身份；新产出层只生成 v1.4 规范形状 | `validate_budget_params()` |
| 旧记录迁移 | 只能做**显式、可追溯**的映射 | `is_legacy_registry_block()` 识别后由调用方决定；本层**不做**静默转换 |
| 两套号 | 不得声称 `R-/D-` 与 `R1-/D1-` 等价 | `can_compare_ids()` 跨世代 `comparable=False`、`declares_equivalence` 恒 `False` |
| 档名 | 只保留在 `budget_profile`，不替代数值、不进新预算身份 | `tier` / `derived_from_tier` / `profile` / `budget_profile` 一律拒（H4） |
| 块级 `source` | 不许直接复制到五项 | 复制只带 `value`+`source`，给不出每项 `origin`/`where` ⇒ 被拒（H3） |

第 1 步审计 §5.2 指出 registry 的 `derived_from_tier.tier` 会经 `_run_id_from()`
进旧 `run_id`、与 `candidate_id` 里的档名重复哈希。本轮**不改 registry**，只在
v1.4 侧杜绝：`derived_from_tier` 不在 `BUDGET_SOURCE_STATES` 里，`tier` 不进块。

阶段 0 冻结的 **237 个唯一 `run_id`**（registry 口径）与未来 runner 的号不可直接
比对，否则会误报"重复 / 漂移"。

---

## 7. 两处必须如实写明的事实（本轮不改 `exec_identity.py`）

### 7.1 `非 recorded` 的后果是 `unconfirmed` 无号，不是降 `derived`

用户裁决写的是"只有五项都达到 `recorded`，才允许预算部分支持整体执行身份进入
`recorded` 状态"，方向正确，但 v1.4 的**实际后果更严格**：

```
_chk_budget()  逐项要求 source == "recorded"
evaluate()     data/rng/budget 任一不过 ⇒ fatal=True
               ⇒ status="unconfirmed"、run_id=None、workflow_id=None
```

G4 用真实 `E.evaluate()` 锁住这条：`budget_not_recorded:*` 出现时状态是
`unconfirmed` 且两个号都是 `None`，**不是"降 derived 发个号"**。
`budget_evidence_state()` 的 `v14_consequence` 因此写
`unconfirmed:no_run_id`，并附 `v14_note` 说明来源。

### 7.2 `chain_method` 当前未登记进哈希闭集，裁决尚未生效

Q6 裁决把 `chain_method` 归 `config.effective` 作为执行条件（`in_run_id=True`）。
但 `config_keys_in_run_id()` 只收 `RUN_ID_CONFIG_KEYS` 里登记且
`effect ∈ RUN_ID_EFFECTS` 的键，而 **v1.4 的 `RUN_ID_CONFIG_KEYS` 没有
`chain_method`**（X6 用 AST 实测锁住）⇒ 本轮放进 `config.effective`
**暂时不产生 `run_id` 影响**。

所以 `split_extensions()` 把"契约意图"与"实现事实"分开写：
`run_id_fields=["chain_method"]`、`pending_v14_registration=["chain_method"]`、
`warning` 明确说明待办、`constructs_identity=False`。**不许把"打算纳入"
说成"已经纳入"。**

让裁决真正生效属于**第 4 步**：升 `RUN_ID_CONFIG_KEYS_VERSION`（当前 1）、
登记 `chain_method`，并复核它与 `progress`(display_only) 的边界。本轮不改身份模块。

保留裁决理由：`parallel` 被设备数条件改成 `sequential` 意味着实际执行方式变化，
当前审计**没有证明两种方式数值结果逐位等价**，因此不提前降为纯展示信息；
将来有严格等价性证据可另行改身份契约并升版本。

另两个扩展字段：`progress_bar` 在身份表明确 `display_only`，CI 覆盖造成的
`mismatch / forced_by_env` 是证据链记录、不该让 `run_id` 换号
（`BUDGET_PARAMS_SOURCE_AUDIT.md` §4）；`extra_fields` 只在 `MCMC.run`
调用边界观测，影响诊断数据采集，不属于五项采样预算。

---

## 8. Q5：`run_id` / `workflow_id` 的分工

1. 父工作流的 `retry`、触发条件、重试策略、尝试次数 → **`workflow_id`**；
2. **每次实际执行**的五项预算 → **该次子运行的 `run_id`**；
3. `retry_kw` **不含** `BUDGET_KEYS` 中任何键（`check_retry_kw()`）。

这不是纸上谈兵：`workflow_payload()` 里 `retry_kw: _plain(...)` 是**整块**
进 `workflow_id` 的（第 1 步审计 §5.4）。重跑时把 `n_samples` 从 1000 改到
2000，应产生**新的子运行身份**，而不是仅改变父工作流里的一个数值字典；
`children` 继续记录子运行身份引用，不等于再次把预算原始数值塞进 `retry_kw`。

---

## 9. 与阶段 C 的分层（为什么本轮不接生产）

| | 本轮 `budget_params.py`（第 2 步） | 阶段 C `sampler_capture.py`（`5609ef2`） |
|---|---|---|
| 层 | 构造 / 校验**纯函数** | 真实构造 / 调用边界**采集** |
| 输入 | 逐项观测记录（由调用方给） | `mcmc` 对象、构造实参、`MCMC.run` 实参 |
| 输出 | 规范 `budget_params` 块 + 证据状态 | 事件挂 `mcmc.provenance` |
| 哈希 | 只投影 `{value, source}`，**不构造号** | 事件内容不进身份载荷 |
| 生产 import | **零**（X1 AST 扫 41 个顶层 `.py`） | 两处生产构造点接入 |

本轮零生产代码改动 ⇒ AST 静态闭包不变，`source_manifest.json` **不用重生成**，
`--check rc=0`。这与阶段 C 收口时"接进生产 ⇒ 闭包 +2 模块、清单
15/19/18 → 17/21/20"形成对照：改了生产就必须重生成清单，没改就不该动。

---

## 10. 验证结果

- `budget_params.py` 内置自检 **29 项**：异常必须核理由，且每条反例都配正对照。
- `test_budget_params.py` **45/45**，九组 C5 / S4 / T5 / U5 / H5 / R4 / E5 / G4 / X8。
- 与 v1.4 的真实交叉核验：`_budget_payload()` 逐位对齐（C2）、`_chk_budget()`
  （C4 / G2）、`evaluate()` 的 `unconfirmed` 无号（G4）、AST 常量对齐
  （X5：`BUDGET_KEYS` 与 `exec_identity` / `circuit_registry` 逐位相同；
  X6：`chain_method` 未登记）。
- 全套回归：identity **68/68**、capture **31/31**、bundle **25/25**、
  provenance **45/45**、sampler_capture **28/28**、两模块自检通过、
  `manifest --check rc=0`、冻结 **9/9**、
  `map_stage0_records --check rc=0`（239 / 独立 239 / 严格 237 / 唯一 237）、
  第 1 步只读探针 `rc=0`。
- 冻结与阶段 0 校验必须用 `D:/autoeis_env/Scripts/python.exe`（系统 Python
  无 jax 会假报失败）。
- 唯一噪声：AST 解析 `nested_fit.py` 时其第 42 行既有注释
  `D:\autoeis_env\Scripts\python.exe` 会报一条
  `SyntaxWarning: invalid escape sequence '\S'`，parse 成功、与本轮无关，
  按"不回头重复已收口工作"本轮不动。

---

## 11. 明确不动的东西 / 下一步

**继续挂账、不顺手修**：

- `bm_real.py:1172` 的 `kw` 读取问题；
- `tree_depth_stats(mc, kw[...])` 使用声明值而非实际值；
- `fit_top` 缺少 `extra_fields` 入口。

**本轮未动**：阶段 C 捕获层、`exec_identity.py`、`circuit_registry.py`、
`run_id` / `workflow_id`、三处捕获站点、源码清单。

**下一步（按用户顺序，不越界）**：

1. **第 3 步**：独立探针，用真实构造实参和对象属性交叉核验，覆盖默认值、
   显式值、环境覆盖、单链不适用、缺参、人为篡改传递值。
2. **第 4 步**：复裁 `budget_params` ↔ `run_id` / `workflow_id` 职责，
   含 §7.2 的 `chain_method` 登记与两套号的边界。
3. **第 5 步**：最小 runner，单次、单候选，身份记录与产物闭环；
   §14 ⛔ 不启批量拟合。

**执行身份链不整体冻结，最小 runner 不启动。**
