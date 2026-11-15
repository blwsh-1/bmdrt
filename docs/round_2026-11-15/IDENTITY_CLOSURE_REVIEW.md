# ③ 第 4 步：身份闭集复裁报告

> **审查基线**：`origin/main` = `HEAD` = `2786a2fe051c8c98710e71ae66f2849be15ea509`
> （F1–F3 文档订正单独收口后的最新远端 `main`；已 `git fetch`，工作树干净）。
> 本轮按用户要求**重新读取实际文件**，所有结论以本轮读到的代码、blob 与只读机械核对为准，
> **不把此前任何报告的结论当作本轮代码事实**。
> **性质**：只读复裁。不改 `exec_identity.py`、不登记 `chain_method`、不升级任何版本常量、
> 不改 `budget_params.py` / `sampler_capture.py` / `source_manifest.json` / 测试 / 生产代码、
> 不构造可落盘的 `run_id` / `workflow_id`、不启动 runner、三处旧挂账不动。
> **交付物**：本报告 + `docs/INDEX.md` 同步 + 项目日志。机械核对以内联只读方式执行，不落盘。
> **核心问题**：现有身份契约能否准确表达真实生产路径中的**执行条件、来源证据和工作流条件**？
> **结论一句话**：**不能**。预算段投影形状兼容，但 `source="recorded"` 无证据门槛、
> 五项预算在生产侧**没有落盘路径**、`retry_kw` 二次承载预算、每次重跑身份独立无实现保证，
> 且版本/键表机制本身失效 ⇒ **不具备进入最小 runner 阶段的准入**。

---

## 0. 结论速览

| 项 | 结论 |
|---|---|
| 准入结论 | ⛔ **不具备**。否决理由不是"拟合没跑通"，而是**Runner 发出的号无法证明对应哪套真实执行条件** |
| 身份投影闭合 | **仅形状兼容**（`budget_hash_payload()` ↔ `_budget_payload()` 逐位对齐，第 2 步已证）；**语义未闭合** |
| 来源证据闭合 | ❌ 未闭合；`upstream_origin_captured=False` 在捕获事件里**硬编码**，且无生产捕获接口 |
| 运行时采集闭合 | 🔶 两处生产构造点已接捕获；K3 / CLI / `run()` / `fit_one` / `bm_top_gen` 等入口未覆盖 |
| 补丁重建闭合 | 🔶 `pb1` 25/25 干净树与脏克隆可重建；**完整性 ≠ 真实性**、身份层/产出层不在声明集 |
| 缺口统计 | **P0×4**（G1–G4）、**P1×5**（G5–G9）、**P2×1**（G10） |
| 本轮是否上调任何判断 | **否**：`upstream_origin_captured` 保持 `False`，无一项可标 `recorded`，不构造可落盘的号 |

**三项比既有报告更精确的实测事实**（本轮重新取证，见 §1.9）：

1. `RUN_ID_CONFIG_KEYS_VERSION = 1` **既不进 `run_id` / `workflow_id` 载荷，也不被任何校验读取**；
   契约 §6 要求"闭集 + 显式版本化（`run_id_config_keys_version`）"，但记录 schema 里没有这个字段。
   ⇒ **单升该版本不会让任何旧号失效**，Q6 的"升版本并登记"按字面执行是空操作。
2. `chain_method` 未登记时放进 `config.effective`，实际后果是
   **`config_unclassified:chain_method` ⇒ `recorded` 降 `derived`**，而 `run_id_digest()` **逐位不变**
   （仅前缀 `R1-`→`D1-`）。既有措辞"暂时不产生 `run_id` 影响"只在**摘要**层面成立。
3. 把裸 `{value, source:"recorded"}` 写进记录（无 `origin`/`where`/`requested`/`transformed`、
   无 `upstream_origin_captured`）⇒ `evaluate()` 判 **`recorded`** 并照发 `R1-` 号；
   同一块喂 `budget_params.validate_budget_params()` 被 **拒**（缺六键）。
   ⇒ 身份层的 `recorded` 与产出层的 `recorded` 是**两个不同门槛**，前者是纯字符串自检。

---

## 1. 冻结事实（重新读取，基线 `2786a2f`）

### 1.1 基线 blob（本轮 `git ls-tree 2786a2f`，前 8 位）

| 文件 | oid | 文件 | oid |
|---|---|---|---|
| `exec_identity.py` | `93912de4` | `budget_params.py` | `dc6ae77f` |
| `sampler_capture.py` | `bc523b84` | `sampler_provenance.py` | `182b75b8` |
| `source_manifest.json` | `3aaa2371` | `bm_select.py` | `70c8a655` |
| `bm_core.py` | `84f26e51` | `bm_top.py` | `4f5f45af` |
| `bm_real.py` | `d21f68bb` | `nested_fit.py` | `ec90e730` |
| `EXECUTION_IDENTITY_CONTRACT.md` | `b6ef3234` | `BUDGET_PARAMS_CONTRACT.md` | `7e9b43a6` |
| `SAMPLER_CAPTURE_CONTRACT.md` | `70a7b662` | | |

### 1.2 版本与闭集（`exec_identity.py`）

- `IDENTITY_CONTRACT_VERSION = "1.4"` → 进 `run_id_payload["v"]` / `workflow_payload["v"]`，
  且 `_require_version()` **硬校**（≠ `"1.4"` 直接 raise，v0 记录不许套 v1 载荷）。
- `RUN_ID_CONFIG_KEYS_VERSION = 1` → **仅定义处的常量**，无第二处代码读取；
  `run_id_payload` / `workflow_payload` 顶层键里没有它，`_require_version()` 不校它。
- `RUN_ID_CONFIG_KEYS` **38 键** × `(类型标签, effect)`；`EFFECTS` 六值；
  `RUN_ID_EFFECTS = ("model_definition", "execution_condition")` 才进 `run_id`。
- `SIGMA_MEMBER_KEYS` 四键（`sigma_mode`/`sigma_pct`/`sigma_abs`/`sigma_floor`）
  **留在表内但退出哈希**（`config_keys_in_run_id()` 跳过），体现"归类"与"承载"两问分离。
- `BUDGET_KEYS` 五项与 `RUN_ID_CONFIG_KEYS` **交集为空** ⇒ 预算数值只有 `budget_params` 一个载体。
- `RNG_PLAN_KEYS` 五项；`SNAPSHOT_POINTS` 三时点；`IDENTITY_STATES` 三态。

### 1.3 `run_id` 的实际哈希投影

```
run_id_payload(rec, role) = {
  v, candidate_id,
  data:      {sha256, n_points, subset_sha256, f_min, f_max, transform},
  rng_plan:  {root_seed, prng, derivation, n_chains, init_strategy},
  budget:    _budget_payload(budget_params)   # 每项只 {value, source}
  config:    {k: norm_effective(k, raw) for k in config_keys_in_run_id(effective, role)},
  code:      {git_commit, numeric_source_tree_sha256},
  env:       {env_sha256, backend, x64_enabled},
}
```

`_budget_payload()`（`exec_identity.py:633–641`）**只取 `value` 与 `source`**；
`origin` / `requested` / `transformed` / `where` **一律不进哈希**。
`config` 只收「已登记 & `effect ∈ RUN_ID_EFFECTS` & 不在 `SIGMA_MEMBER_KEYS`」的键，
`_EFFECT_BY_ROLE` 有唯一特例 `("voigt_K","child") → model_definition`。
刻意不进：`git_describe` / `git_dirty` / `patch_bundle_*` / 覆盖关系 / `declared_source_tree_sha256` /
`workflow.*` / `rng.actual.*` / `instance.*` / `provenance.registry_version`（§6 表与实现一致）。

### 1.4 `workflow_id` 的实际载荷组成

```
workflow_payload(rec) = {
  v, workflow_source_tree_sha256,
  search:   _plain(wf.search),        extras: _plain(wf.extras),
  gate:     {kk_mode, kk_opts, strict},          # 不含 kk_status / kk_performed
  guards:   _plain(wf.guards),
  pair_mode, retry,
  retry_kw: _plain(wf.retry_kw),      # ★ 整块原样进哈希
  children: sorted(wf.children),      # 只放身份引用（字符串）
}
```

**符合契约的部分**：结果（`kk_status`/`kk_performed`/`Rwp`/`elpd`）不在载荷里，
门禁起作用体现为 `children` 少一项；`children` 排序且父 `workflow_id` 不回流进子 `run_id`。
**不符合 Q5 的部分**：`retry_kw` 整块 `_plain` 进哈希（见 §1.7）。

### 1.5 校验闭集与状态阶梯（`evaluate()`）

```
fatal = _chk_data | _chk_rng | _chk_budget   ⇒ unconfirmed（run_id = workflow_id = None）
其余  _chk_config | _chk_code | _chk_env    ⇒ derived
全过                                        ⇒ recorded
载荷规范化失败                              ⇒ unhashable:… ⇒ unconfirmed 无号
```

`_chk_budget()`（`exec_identity.py:776–784`）**只校 `cell.get("source") != "recorded"`**：
不校 `value` 是否真是最终构造实参，不校 `origin`/`where`，不校六键形状，
不校 `upstream_origin_captured`。`_chk_rng()` 会校 `rng.plan.n_chains == budget_params.num_chains.value`。
**实测**：把五项换成裸 `{value, source:"recorded"}`（数值与夹具一致）⇒ `status="recorded"`、
`run_id="R1-5f9727fb58cf390c71b7f82b009f77a2"`，与基线**逐位相同**；
同块喂 `budget_params.validate_budget_params(..., upstream_origin_captured=False)` ⇒ **raise**
「键必须恰好是 `['value','source','origin','requested','transformed','where']`」。
⇒ **门槛在产出层，算号在身份层，两者零 import**（`exec_identity.py` 的 import 集只有
`__future__/hashlib/json/numpy/pathlib/re/typing`）。

### 1.6 `budget_params.py` 五项记录与证据等级（本轮重读）

- `BUDGET_PARAMS_VERSION = "1.0"`（同样不进哈希、不被身份层读）。
- `CELL_KEYS = ("value","source","origin","requested","transformed","where")` 六键，多/少都拒；
  `HASH_KEYS = ("value","source")`；`TRACE_KEYS` 四键只作溯源。
- `BUDGET_SOURCE_STATES` 三值 `recorded/derived/unavailable`；`BUDGET_ORIGINS` **七值**
  `cli_explicit / module_constant / programmatic_override / function_default /
   library_default / runtime_derived / unknown`。
- `recorded` 四道护栏（`_validate_cell`）：`upstream_origin_captured is True`（哨兵无默认值，
  省略/`None`/truthy 一律 raise）、`origin != unknown`、`where` 非空、`value` 有效。
- `EXECUTION_CHANNELS`（Q6 分流）：`chain_method → config.effective / in_run_id=True /
  registered_in_v14=False`；`progress_bar → sampler_capture_event / False`；
  `extra_fields → mcmc_run_event / False`。`split_extensions()` 单列 `pending_v14_registration`。
- `check_retry_kw()` 禁五键；`budget_hash_payload()` 投影 `{value,source}`；
  `budget_evidence_state()` 写 `unconfirmed:no_run_id`。
- `can_compare_ids()`：`R-/D-`(10hex, registry 0.7.0) 与 `R1-/D1-`(64hex, v1.4) 跨世代不可比。
- **零生产 import**：顶层生产/工具 `.py` 里没有任何模块 import `budget_params`（本轮 AST 实测）。

### 1.7 `retry_kw` 与预算字段的重复承载（本轮新取证）

生产 `bm_select.py` 实际代码：

```
:1433  retry_kw=dict(BUDGETS[retry_budget])          # 整包五项预算数值
:1084  rkw = dict(BUDGETS["standard"] if retry_kw is None else retry_kw)
:1126  V = [_refit(c, f, Z, seed=s2, kw=rkw, sigma_floor=...) for c in V]
:1132  warmup=int(rkw["n_warmup"]), samples=int(rkw["n_samples"])
```

⇒ 生产把 **五项预算数值整包**放在名为 `retry_kw` 的字典里，重跑时用它换档。
而 `exec_identity.workflow_payload()` 把 `wf["retry_kw"]` **整块 `_plain` 进 `workflow_id`**。
**实测**：把 `BUDGETS` 灌进 `workflow.retry_kw` ⇒ `workflow_id` **变化**；
`B.check_retry_kw(该 dict)` ⇒ **raise**；但 `workflow_payload.__code__.co_names` **不含**
`check_retry_kw`，`exec_identity` 也不 import `budget_params` ⇒ **护栏不在算号路径上**。

**契约内部三处口径互相矛盾**：

| 位置 | `retry_kw` 口径 |
|---|---|
| `EXECUTION_IDENTITY_CONTRACT.md:543`（§3.3.4 键表） | `effect = execution_condition`（**子运行**）、归属 `budget_params`（重跑档） |
| `EXECUTION_IDENTITY_CONTRACT.md:856`（§6.3）+ `exec_identity.py:726` | 整块进 **`workflow_id`** |
| `BUDGET_PARAMS_CONTRACT.md:227–236`（§8 Q5） | **禁五键**、预算只由该次子运行 `run_id` 承载 |

### 1.8 `sampler_capture.py` 八项字段与来源证据（本轮重读）

- `SP.FIELDS = CORE_FIELDS(五项) + EXEC_FIELDS(chain_method, progress_bar, extra_fields)`；
  `SC.CONSTRUCTION_FIELDS = 7` + `RUN_FIELDS = (extra_fields,)` ⇒ **八项逐位一致**。
- `OBSERVE_AT` 8 项；`HOW_BY_FIELD`（`extra_fields` 才允许 `call_arg`/`derived`）；
  `REASON_CODES` 12 值；`OBSERVED_STATUSES` 5 值 + `STATUSES` 5 态（双状态都落盘）。
- 事件每项 **11 键**：`name, requested, resolved, observed, observed_status,
  observed_status_supplied, applicable, reason_code, status, why, judged`；
  `resolved` = `_slot(value, origin, where)` 三键。
- **`resolved.origin` 是字面量推断**：在 `passed` 里 ⇒ `"derived"`，不在 ⇒ `"default"`
  （`sampler_capture.py:273–277`）；`where` 是构造点字符串或 `numpyro.*.__init__:<param>=<默认>`。
  ⇒ **按代码路径推断，不是生产端真实记录**。
- 事件级 `upstream_origin_captured: False` **硬编码**（`_event()` `:343`）；
  文档头与契约 §5 明账「`status=complete` 只表示构造实参→对象属性闭合」。
- `CALL_SITES` 两处（`bm_core.py:610`、`bm_top.py:1004`）；
  `NON_PRODUCTION_SITES` 明账 **K3 = `evidence/nested_topology/m35_cond_n2.py:116`** 未纳入。
- `CAPTURE_VERSION = "1.0"`、`PROVENANCE_CONTRACT_VERSION = "1.0"` 都不进哈希。
- **词表/形状不相容（实测）**：
  `SP.ORIGINS = (cli, config_file, default, tier, derived)` 与 `B.BUDGET_ORIGINS`（七值）
  **交集为空**；事件字段 11 键 **不覆盖** `CELL_KEYS` 六键；
  `sampler_capture.py` **不 import** `budget_params` ⇒ **捕获事件 → `budget_params` 没有适配层**。

### 1.9 只读机械核对结果（内联执行，未落盘、未改代码）

| 组 | 核对 | 结果 |
|---|---|---|
| A | `IDENTITY_CONTRACT_VERSION="1.4"`；`RUN_ID_CONFIG_KEYS_VERSION=1`；两个 payload 顶层键均无版本字段 | ✅ 与代码一致 |
| B | `config_effect("chain_method")=None`；五预算键 ∩ `RUN_ID_CONFIG_KEYS` = `[]`；塞入后 `status=derived`、新增 `config_unclassified:chain_method`、`run_id_digest` **不变**、config 段相同 | ✅ 实测 |
| C | 身份表有 `progress`(display_only)、**无** `progress_bar`；塞 `progress_bar` ⇒ `config_unclassified:progress_bar` ⇒ `derived` | ✅ 实测 |
| D | `workflow_payload.co_names` 不含 `check_retry_kw`；灌 `BUDGETS` ⇒ `workflow_id` 变；`check_retry_kw` 对同块 raise | ✅ 实测 |
| E | 八项字段一致；`BUDGET_ORIGINS` ∩ `SP.ORIGINS` = `∅`；事件形状不含 `CELL_KEYS` | ✅ 实测 |
| F | 真实 `capture_at_construction()` 事件：`resolved.origin="derived"`、`observed.where="mcmc.num_warmup"`、`upstream_origin_captured=False`、五项 `status=complete`、聚合 `weakest=mismatch`（因 `chain_method`） | ✅ 实测 |
| G | 五项 `source=derived` ⇒ `budget_evidence_state.v14_consequence="unconfirmed:no_run_id"`；`E.evaluate()` ⇒ `status=unconfirmed`、两号 `None`、5 条 `budget_not_recorded` | ✅ 实测 |
| H | 清单 A/B/C 均含 `sampler_capture.py`/`sampler_provenance.py`；**不含** `exec_identity.py`/`budget_params.py`/`capture_contract.py`/`patch_bundle*.py` | ✅ 实测 |
| I | 裸 `{value,source:"recorded"}` ⇒ `recorded` + 发 `R1-` 号（与基线同号）；同块被产出层拒 | ✅ 实测 |
| J | `_type_ok(enum三值, callable)=False`；`norm_effective`/`_plain` 对 callable **raise** | ✅ 实测 |

复现环境：`D:/autoeis_env/Scripts/python.exe`（须有 `numpy` / `numpyro` / `jax`）。
`test_exec_identity._rec()` 作为夹具；所有调用均为纯函数或对象构造，**不跑拟合、不写产物**。

### 1.10 生产 import 闭包与 `source_manifest.json` 覆盖

AST 实测（顶层 `.py`，排除 `test_*`）：

```
bm_core.py        → sampler_capture
bm_top.py         → sampler_capture
sampler_capture.py→ sampler_provenance
patch_bundle_io.py→ capture_contract, exec_identity, patch_bundle
```

`exec_identity.py`、`budget_params.py` **零生产 import**。清单三档
**A 17 / B 21 / C 20** 文件，均含两个捕获模块（阶段 C 接入后 +2），
但**不含身份层、产出层、采集契约层、补丁层**（`excluded` 只列 `requirements.txt` /
`out|evidence|archive|__pycache__|docs` / 数据文件 / 第三方包，并未显式讨论这四个纯函数模块）。
⇒ **`exec_identity.py`（算号规则）与 `budget_params.py`（证据门槛）改了也不会进
`numeric_source_tree_sha256` ⇒ 不换号**；只有手动升 `IDENTITY_CONTRACT_VERSION` 才换。
另 `pending_demotions` 第 5 条写「数值效应已由 `budget_params` 捕获」，
但产出层零生产 import、无产出路径 ⇒ 该假设前提**未闭合**。

---

## 2. 逐项裁决：身份承载关系

> 表按用户指定八行逐项给出「应进入哪里 / 核心问题 / 本轮裁决 / 依据」。
> **裁决一律不越界**：不登记、不升级、不接生产、不上调 `recorded`。

| 信息 | 应进入哪里 | 核心问题 | 本轮裁决 | 依据 |
|---|---|---|---|---|
| 五项预算 `value` | `budget_params` | 是否确实对应最终构造实参？ | **C1–C8 覆盖路径上「是」（探针实测），但生产侧没有产出 `budget_params` 记录的接口** ⇒ 目前只是采集值，不是落盘身份。需**生产捕获接口 + 事件→六键适配层**；在此之前 `value` 不得自称已闭合 | §1.6/§1.8；零生产 import、11 键 vs 6 键、词表交集 ∅ |
| 五项预算 `source` | `budget_params` | 是否有足够证据支持 `recorded`？ | **否。一律 `derived` / `unavailable`，禁止上调**。`upstream_origin_captured=False` 硬编码；产出层非 `True` 禁 `recorded`；身份层只校字符串，不构成证据 | §1.5/§1.6/§1.8；`_validate_cell` 四护栏 |
| `origin`、`where` | 来源追溯（**不进哈希**） | 能否证明真实来源，而不是按代码路径推断？ | **不能**。捕获层 `origin` 是字面量 `derived`/`default`（按「是否在 `passed` 里」推断），`where` 是构造点字符串。保持追溯用途，**不得当证据、不得升级 `source`**；且 `SP.ORIGINS` 与 `BUDGET_ORIGINS` 交集为空，映射须单独定义 | §1.8；`sampler_capture.py:273–277` |
| `chain_method` | 待复裁的执行条件 | 是否登记进 `RUN_ID_CONFIG_KEYS`、如何升级版本？ | **本步不登记**。当前未登记；塞入 `config.effective` 会 `config_unclassified` **降 `derived`**（摘要不变）。登记前须先裁：类型标签、**callable 值策略**（合法值含 callable，登记 enum 会 `unhashable:` 无号）、与 `progress` 的命名边界、以及**升 `IDENTITY_CONTRACT_VERSION` 而非只升 `RUN_ID_CONFIG_KEYS_VERSION`** | §1.2/§1.9 B,J；numpyro `MCMC.__init__` |
| `progress_bar` | display-only（采样调用事件） | 是否确实不影响数值执行条件？ | **当前证据支持 display-only，但结论限定在当前 numpyro 版本与已测路径**（`MCMC.run` 源码不用它；`__init__` 仅显示逻辑与 callable `chain_method` 联动关闭；C6 实测声明 True/属性 False）。**身份表键名是 `progress` 而非 `progress_bar`** ⇒ 不做映射不得塞进 `config.effective`（会降 `derived`） | §1.9 C；`EXECUTION_IDENTITY_CONTRACT.md` §7.2 表述需订正 |
| `extra_fields` | 采样调用事件（`mcmc_run_event`） | 是否只记录观测配置、不误入预算身份？ | **是**：不进 `budget_params`、不在 `RUN_ID_CONFIG_KEYS`、不进任何号。但 `fit_top` 无入口（`bm_top.py:1009` 硬编码 `("diverging",)`）⇒ `derived`/`resolved_missing`，**不能当 `recorded` 证据**。两处挂账不动 | §1.6/§1.8；`EXECUTION_CHANNELS` |
| retry 策略 | `workflow_id` | 是否避免在 `retry_kw` 中重复承载预算？ | **否，当前重复承载**。`workflow_payload` 整块 `_plain`；生产 `bm_select.py:1433` 传 `BUDGETS` 整包；`check_retry_kw()` **不在算号路径**。须裁决：从载荷移除，或在算号前强制校验；并统一三处契约口径 | §1.4/§1.7；实测 `workflow_id` 变 |
| 每次重跑的实际预算 | 子运行 `run_id` | 是否保证每次重跑身份独立？ | **无保证**。`_refit(..., kw=rkw)` 换预算但没有生成子运行记录；`children` 只放身份引用；runner 未启动。若 `retry_kw` 再带预算，父 `workflow_id` **二次承载**，父子身份耦合 | §1.7；`bm_select.py:1084/1126`；契约 §6.3 |

---

## 3. 字段 → `run_id` / `workflow_id` / provenance 映射表

图例：`✓` 进哈希 · `✗` 不进哈希 · `记录` 只落盘作证据/追溯 · `事件` 只在 `mcmc.provenance` 事件里。

| 字段 / 块 | `run_id` | `workflow_id` | provenance / 事件 | 备注（本轮实测） |
|---|---|---|---|---|
| `budget_params.{5}.value` | ✓ | ✗ | 记录 | 经 `_budget_payload()`；**生产侧无产出路径** |
| `budget_params.{5}.source` | ✓ | ✗ | 记录 | 身份层只校字符串 ⇒ G1 |
| `origin` / `requested` / `transformed` / `where` | ✗ | ✗ | 记录 | Q4：不许冒充执行条件；`origin` 目前是推断值 |
| `upstream_origin_captured` | ✗ | ✗ | 事件 | 捕获事件恒 `False`；产出层是**必填参数**（不落盘） |
| `rng.plan.*`（五项） | ✓ | ✗ | 记录 | `_chk_rng` 校 `n_chains` 与 `num_chains.value` 一致 |
| `data.*` / `candidate_id` / `v` | ✓ | (`v` ✓) | — | `v` = `IDENTITY_CONTRACT_VERSION`，**硬校** |
| `config.effective` 已登记且 `effect ∈ RUN_ID_EFFECTS` | ✓ | ✗ | 记录 | 38 键闭集；`SIGMA_MEMBER_KEYS` 四键退出哈希 |
| `sigma` 复合块 | ✓ | ✗ | 记录 | 只 `array_sha256` + `floor` 进哈希 |
| `code.git_commit` / `numeric_source_tree_sha256` | ✓ | ✗ | — | `declared_*` / `workflow_*` 摘要不进 `run_id` |
| `env_sha256` / `backend` / `x64_enabled` | ✓ | ✗ | — | `_chk_env` 复算 |
| `workflow_source_tree_sha256` | ✗ | ✓ | — | 工作流源码身份 |
| `workflow.search` / `extras` / `guards` | ✗ | ✓ | 记录 | 结果（`kk_status` 等）**不进** |
| `gate{kk_mode, kk_opts, strict}` | ✗ | ✓ | 记录 | `kk_performed` / `kk_status` 明确排除 |
| `pair_mode` / `retry` | ✗ | ✓ | 记录 | `retry` 在键表里是 `workflow_scope` |
| **`workflow.retry_kw`** | ✗ | **✓ 整块** | 记录 | ★ G3：Q5 禁五键，但护栏不在算号路径 |
| `children` | ✗ | ✓（sorted） | 引用 | 只放子 `run_id`；父号不回流 |
| `chain_method` | ✗（未登记） | ✗ | 事件 | 塞 `config.effective` ⇒ 降 `derived` 且不换摘要 |
| `progress` / `progress_bar` | ✗（display_only） | ✗ | 事件 | 键名不一致 ⇒ G8 |
| `extra_fields` | ✗ | ✗ | 事件 | 只在 `MCMC.run` 边界；`fit_top` 缺入口 |
| `adapt_step_size` / `kernel` / `device_count` / `num_chains` / `n_capture_errors` / `aggregate` / `judged` | ✗ | ✗ | 事件 | 八项判定的上下文与聚合，**不进任何号** |
| `resolved` / `observed` / `observed_status` / `status` / `reason_code` / `why` | ✗ | ✗ | 事件 | 双状态落盘；`complete` ≠ 来源链闭合 |
| `CAPTURE_VERSION` / `PROVENANCE_CONTRACT_VERSION` / `BUDGET_PARAMS_VERSION` | ✗ | ✗ | 标签 | 三个 `1.0` 均**不进哈希、不被校验** |
| `RUN_ID_CONFIG_KEYS_VERSION` | ✗ | ✗ | 标签 | ★ G5：契约要求"显式版本化"但实现是空常量 |
| `IDENTITY_CONTRACT_VERSION` | ✓ | ✓ | 标签 | 唯一真正生效的世代标签 |
| `sampling`（`bm_real.py:1172` 读 `kw`） | ✗ | ✗ | 产物声明 | C8 已证伪「声明 ≠ 执行条件」，**不得当身份来源** |
| `git_describe` / `git_dirty` / `patch_bundle_*` / 覆盖关系 / `rng.actual.*` / `instance.*` | ✗ | ✗ | 记录 | 重建 / 覆盖性 / 实例，按 §6 明确排除 |
| `software.registry_version` | ✗ | ✗ | provenance | D-1：映射工具版本，不进号 |

---

## 4. 来源闭合专项裁决（本步最重要）

> **不得为了让校验通过，把推断值升级成 `recorded`。** 以下逐项决定归属：
> 「记录层可补」/「需生产捕获接口」/「保持 `unavailable` 并阻止身份生成」。

| 缺口（本轮重新核对） | 现状 | 裁决 | 落地条件（**本步不实现**） |
|---|---|---|---|
| **S1 `bm_real.KW` 可变 dict 运行期修改无留痕** | `bm_real.py:97 KW = dict(...)`；`SAMPLER_CAPTURE_CONTRACT.md §5` 已举外部脚本直接改它的例子；`dict(KW)` 快照不带时点与来源标记 | **需生产捕获接口**。记录层补不齐：事后快照无法区分「初值 / CLI 覆盖 / 运行内改」 | 在赋值点与快照点挂带时点的事件；`origin=programmatic_override` 才可能成立 |
| **S2 `origin` 按路径推断，非生产记录** | 捕获层 `resolved.origin` 字面量 `derived`/`default`；`SP.ORIGINS`(5) 与 `BUDGET_ORIGINS`(7) **交集为空**；事件 11 键 ≠ `CELL_KEYS` 六键；无适配层 | **需生产捕获接口 + 显式词表映射**。映射未定义前保持 `unavailable`，**禁止把推断值写成 `recorded`** | 先定义「探测到的来源类别」→ 七值闭集的**有损/拒绝规则**（不可静默挑一个）；再补适配层 |
| **S3 `sampling` 声明值与构造实参分叉** | `bm_real.py:1172` 读 `kw`；C8 实测声明 2000 / 构造实参 16；`tree_depth_stats(mc, kw["max_tree_depth"])`（`:973`）用声明值当上限 | **保持不可用**：`sampling` 块**不得当身份来源**，只能作产物声明。三处旧挂账不动 | 需生产侧改读真实构造实参（单独授权）；本轮只读复裁不碰 |
| **S4 `chain_method` 未登记** | 38 键无它；塞入 `config.effective` ⇒ `config_unclassified` 降 `derived`，摘要不变 | **本步不登记**，保持挂账；作为「执行条件待复裁」明账 | 登记须与 §6 版本升级方案**同一批**授权；先裁 callable 策略 |
| **S5 K3 独立构造路径未纳入** | `sampler_capture.NON_PRODUCTION_SITES` 明账 `m35_cond_n2.py:116` 绕过 fit/fit_top 与 BUDGETS | **保持 `unavailable`**：K3 是否满足「五项显式传递」**未知**，不得据 C1–C8 外推 | 是否纳入生产捕获是**新决策**，需单独授权并同步 `CALL_SITES` / `source_manifest.json` |
| **S6 事件级 `upstream_origin_captured=False` 硬编码** | `_event()` `:343` 常量；`status=complete` 只读作「构造实参→对象属性」闭合 | **保持 `False`，不上调**。任何 `source="recorded"` 都不成立 ⇒ v1.4 `unconfirmed` 无号 | 只有 S1+S2 都闭合且捕获证据落盘后才可讨论；**这是准入闸门，不是可调参数** |

**结论**：S1/S2 需生产捕获接口，S3/S5/S6 必须保持 `unavailable` 并阻止身份生成，S4 保持挂账。
**目前没有任何一条生产路径能产出一份合规的 `budget_params` 记录**——
捕获事件与六键形状/七值词表不相容、零适配层、零生产 import。
`budget_hash_payload()` 与 `_budget_payload()` 逐位相等只证明**投影形状兼容**，
不证明来源闭合，也不证明完整 `run_id` / `workflow_id` 可生成。

---

## 5. 缺口及影响等级

> **P0**：会让发出的号对应不上真实执行条件，或把"不知道"包装成"知道" ⇒ 阻断 runner。
> **P1**：准入前必须裁决/改契约；不修则裁决无法生效或跨世代不可比。
> **P2**：已明账、不改行为，但必须限定结论边界。

| 编号 | 级别 | 缺口 | 影响 | 依据 |
|---|---|---|---|---|
| **G1** | **P0** | `source="recorded"` **无证据门槛**：`_chk_budget()` 只校字符串，`exec_identity` 零 import 产出层 | 可凭空写五项 `recorded` ⇒ `evaluate()` 判 `recorded` 并照发 `R1-` 号；与「禁把推断值升级成 recorded」直接冲突 | §1.5 / §1.9 I |
| **G2** | **P0** | 五项预算**在生产侧无落盘路径**：事件 11 键 ≠ 六键、词表交集 ∅、无适配层、零生产 import、`upstream_origin_captured=False` 硬编码 | 「`value` 对应最终构造实参」只在 C1–C8 受控探针里成立；真实运行没有 `budget_params` 记录可进哈希 | §1.6 / §1.8 / §4 |
| **G3** | **P0** | `retry_kw` **二次承载**预算：生产传 `BUDGETS` 整包、`workflow_payload` 整块进号、`check_retry_kw()` 不在算号路径 | 同一组预算数值同时进 `workflow_id` 与（应有但未实现的）子 `run_id`；改预算会换父号，违反 Q5 | §1.7 / §1.9 D |
| **G4** | **P0** | 每次重跑身份独立**无实现保证**：`_refit` 换 `rkw` 但不生成子运行记录，`children` 只有引用，runner 未启动 | 重跑结果无法各自对应独立身份；与 G3 叠加造成父子耦合 | §1.7 / §2 |
| **G5** | **P1** | `RUN_ID_CONFIG_KEYS_VERSION` **不进哈希、不被校验**，契约却要求"显式版本化" | 登记 `chain_method` 后旧号不失效；`R1-` 跨键表世代不可区分；Q6「升版本」按字面执行是空操作 | §1.2 / §1.9 A |
| **G6** | **P1** | `chain_method` 未登记的后果被写成「不产生 `run_id` 影响」；且合法值含 **callable** | 实测会 `config_unclassified` **降 `derived`**；若登记为 enum，callable 会 `_type_ok=False` 降 `derived` 或 `_plain` raise ⇒ `unhashable` **无号** | §1.9 B,J |
| **G7** | **P1** | 契约三处 `retry_kw` 口径矛盾（§3.3.4 / §6.3 + 代码 / `BUDGET_PARAMS_CONTRACT §8`） | 实现按哪条走都不自洽；裁决无法落地为可审查条款 | §1.7 表 |
| **G8** | **P1** | `progress_bar` vs `progress` **命名映射缺口**；`BUDGET_PARAMS_CONTRACT §7.2` 写「`progress_bar` 在身份表明确 display_only」与代码不符 | 捕获字段直接灌 `config.effective` ⇒ `config_unclassified` 降 `derived`；display-only 结论缺可核对落点 | §1.9 C |
| **G9** | **P1** | `source_manifest.json` **不含** `exec_identity.py` / `budget_params.py`；`pending_demotions` 一条假设「数值效应已由 `budget_params` 捕获」 | 算号规则 / 证据门槛改了不进 `numeric_source_tree_sha256` ⇒ 不换号；降级理由前提未闭合 | §1.10 |
| **G10** | **P2** | K3 未纳入、三处旧挂账、CLI/`DEFAULTS` 合并 / `bm_real.run()` / `fit_one` / `bm_top_gen` 未覆盖、`extra_fields` 在 `fit_top` 缺入口 | 结论必须限定在 C1–C8 覆盖路径与测试工况，**不得外推到全部生产入口** | §1.8 / F1–F3 订正文 |

---

## 6. 必须修改的契约条款与版本升级建议（**只给建议，本步不实现**）

### 6.1 条款修改清单

| # | 位置 | 必须改什么 | 关联缺口 |
|---|---|---|---|
| C1 | `EXECUTION_IDENTITY_CONTRACT.md §5`、`§9`、`exec_identity._chk_budget()` | 把「`source == recorded`」从**字符串自检**改为**结构化证据门槛**：要求六键形状 / `where` 非空 / `origin != unknown` / `upstream_origin_captured=True` 的**显式落盘字段**，并与 `budget_params.validate_budget_params()` 逐项对齐。**不许靠 runner 自觉** | G1 |
| C2 | `§3.3 / §6 / §13.4`、`run_id_payload` / `workflow_payload` / `_require_version()` | `run_id_config_keys_version` 从**注释性常量**升为**记录字段 + 载荷键 + 校验项**；否则登记键也不换号 | G5 |
| C3 | `§6.3 workflow_id` + `§3.3.4 retry_kw` + `BUDGET_PARAMS_CONTRACT §8` | 统一 `retry_kw` 口径：要么从 `workflow_payload` **移除**（只留 `retry`/策略），要么在**算号前**强制 `check_retry_kw()`；明确「每次重跑预算 → 该次子 `run_id`」，`children` 只放引用 | G3 / G4 / G7 |
| C4 | `§3.3.1 键表 / §6` | `chain_method` 登记方案：类型标签、**callable 表示或显式拒绝策略**、与 `progress` 的命名边界、单链 `not_applicable` 处置、`parallel→sequential` 偏移是否作为条件；并订正 `BUDGET_PARAMS_CONTRACT §7.2`「不产生 run_id 影响」为「**不进摘要，但会 `config_unclassified` 降 `derived`**」 | G6 |
| C5 | `§9 硬规则` / `SAMPLER_CAPTURE_CONTRACT §5,§8` | 补一条：「捕获事件的 `origin` 字面量（`derived`/`default`）**不得经任何转换升级为 `source=recorded`**」；定义 `SP.ORIGINS`(5) → `BUDGET_ORIGINS`(7) 的**有损/拒绝映射**，不许静默挑值 | G2 / S2 / S6 |
| C6 | `BUDGET_PARAMS_CONTRACT §7.2` | 订正「`progress_bar` 在身份表明确 `display_only`」⇒ 身份表登记的是 **`progress`**；补命名映射缺口说明 | G8 |
| C7 | `§3.1.2 声明集` / `§14 实现进度` / `§18 冻结台账` | 明确 `exec_identity.py` / `budget_params.py` **不在声明集**的后果（改规则不换号）与缓解方式；把「③ 参数捕获作为身份来源」状态更新为「**来源闭合未达成、无 `budget_params` 产出路径**」；Runner 前置阻塞项补 G1–G4；`pending_demotions` 里以 `budget_params` 为前提的假设标 `draft` | G9 / G2 |
| C8 | `§14` 准入判据 | 把「来源证据、身份投影、运行时采集、补丁重建**四项都闭合**」写成可勾选清单，任一项未闭合 ⇒ ⛔ 不授权 runner | 准入 |

### 6.2 版本升级建议

- **不能只升 `RUN_ID_CONFIG_KEYS_VERSION 1→2`**：该常量既不进哈希也不被校，升了等于没升。
- 若登记 `chain_method` 且 `config_keys_in_run_id()` 的实际键集变化 ⇒ 按 D-4「语义变了必须换号」，
  应**同时**：① `IDENTITY_CONTRACT_VERSION 1.4 → 1.5`；② `RUN_ID_CONFIG_KEYS_VERSION 1 → 2`；
  ③ 把该版本**写进记录与 `_require_version()` 校验**；④ `IDENTITY_GENERATIONS` /
  `can_compare_ids()` 增加新世代映射（registry `R-/D-` 10hex 与 v1 `R1-/D1-` 64hex 仍不可比）。
- 升 `IDENTITY_CONTRACT_VERSION` 会让**所有** v1.4 号整体失效（全换号）——这是**预期代价**，
  须在授权时明确接受，不得用「兼容旧号」绕过。
- `BUDGET_PARAMS_VERSION` / `CAPTURE_VERSION` / `PROVENANCE_CONTRACT_VERSION` 三个 `1.0`
  目前都不进哈希、不被校：须明确它们是**文档/契约标签**还是**身份输入**，不留模糊地带。
- 实现授权后的**连带收口**（预先明账，本步不做）：`source_manifest.json` 是否因生产 import 变化
  需重生成；`test_exec_identity.py` 68 格 / `test_budget_params.py` 45 格 /
  `test_sampler_capture.py` 28 格 / 清单 `--check` / 冻结 9/9 / 阶段 0 `--check` 需复跑并**补新反例**
  （尤其 G1「凭空 recorded 必须被拒」、G3「`retry_kw` 含五键必须被拒或不进号」、
  G5「版本字段变化必须换号」）。
- **实现顺序建议**：先裁 C1–C8（条款）→ 单独授权实现 → 补反例并复跑全套 →
  再做一次准入复核 → **最后**才单独授权最小 runner。

---

## 7. 是否具备进入最小 runner 阶段的准入结论

按用户给定的四条闭合线逐项复裁：

| 闭合条件 | 判定 | 本轮证据 |
|---|---|---|
| **来源证据** | ❌ **未闭合** | S1/S2 需生产捕获接口；S3/S5/S6 保持 `unavailable`；`upstream_origin_captured=False` 硬编码；**无任何生产路径产出合规 `budget_params` 记录** |
| **身份投影** | 🔶 **形状兼容、语义未闭合** | `budget_hash_payload()` ↔ `_budget_payload()` 逐位相等（第 2 步）；但 G1 证据门槛、G5 版本机制、G6 键表未登记、G7 口径矛盾均未解决 |
| **运行时采集** | 🔶 **部分闭合** | 两处生产构造点已接捕获（阶段 C）；K3 / CLI 合并 / `bm_real.run()` / `fit_one` / `bm_top_gen` 未覆盖；三时点快照、harness 归属（C-1）、多次采样逐次事件（C-3）尚需在**真实 runner** 里验证，不能由夹具推定 |
| **补丁重建** | 🔶 **可重建但未闭合** | `pb1` 25/25 干净树/脏克隆在独立目录重建且三层摘要逐位相同；但契约明账**完整性 ≠ 真实性**、Windows mode `skipped`、`reference_only` 待裁；`exec_identity.py` / `budget_params.py` 不在声明集（G9） |

> ### ⛔ 准入结论：**不具备进入最小 runner 阶段的条件。**
>
> 理由不是"实验没跑完"，而是：**当前契约无法准确表达真实生产路径中的执行条件、
> 来源证据和工作流条件**。P0×4 中任意一条都足以让 Runner 发出的号不可信——
> 尤其 G1（凭空 `recorded` 可发 `R1-` 号）直接违背「不能把推断值升级成 `recorded`」，
> G3/G4 会让重跑身份无法回溯，G2 让预算段根本没有真实落盘来源。
> 在这种情况下即使大量拟合顺利完成，也**无法证明每条结果对应哪套真实执行条件**——
> 与用户核心判断一致。
>
> **本次复裁不放宽、不接生产、不构造可落盘的号、不整体冻结执行身份链。**
> `upstream_origin_captured` 保持 `False`，五项 `source` 最高 `derived`，
> v1.4 `evaluate()` 的实际后果仍是 `unconfirmed` 且 `run_id` / `workflow_id` 均为 `None`。

**进入 runner 前必须完成的最小集合**（只列，不实施）：
① 裁 C1–C8 条款并单独授权实现；② 补 `budget_params` 的生产捕获接口与事件→六键适配层
（含词表映射与 `upstream_origin_captured` 的真实落盘）；③ 把 `source=recorded`
做成身份层可校验的结构化门槛；④ 消掉 `retry_kw` 二次承载并给出子运行身份生成规则；
⑤ 版本/键表升级并重跑全套反例与清单；⑥ 真实运行里验证三时点、harness 归属与补丁重建；
⑦ 再做一次准入复核。**⑦ 通过后，最小 runner 仍需单独授权。**

---

## 8. 本轮边界与下一步

- **两件事分开收口已完成**：① F1–F3 文档订正（`2786a2f`，只改审查报告 + INDEX + 日志）；
  ② 本报告（第 4 步复裁）。本报告**不预支实现授权**。
- **不改**：`exec_identity.py`、不登记 `chain_method`、不升任何版本常量、
  `budget_params.py` / `sampler_capture.py` / `source_manifest.json` / 生产代码 / 测试一律不动；
  不构造可落盘的 `run_id` / `workflow_id`；不启动 runner；三处旧挂账
  （`bm_real.py:1172` 读 `kw`、`tree_depth_stats(mc, kw[...])`、`fit_top` 缺 `extra_fields` 入口）不动。
- §1.9 的机械核对仅用既有夹具做**只读函数/对象调用**，未跑拟合、未写产物、未落盘脚本。
- **结论边界**：所有实测结论限定在 C1–C8 覆盖的两处生产构造点、五条转发路径与所列测试工况；
  K3 独立构造路径及 §5 G10 所列入口尚未取证，**不得外推**。
- **下一步由用户裁决**：是否按 §6 批准条款修改与版本升级的实现授权（单独一轮），
  以及实现完成后是否重新做准入复核。**最小 runner 在任何情况下仍需单独授权。**
