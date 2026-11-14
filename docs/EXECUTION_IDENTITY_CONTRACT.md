# 执行身份契约 v1（草案，未实现，待审查）

状态：**草案 `v1.1-draft`，2026-11-14。只定义，不实现。** 本轮仍零代码改动：
registry `0.7.0`（`b2675c8255bea716`）、`records.jsonl` 未动、快照未重冻结、
`R-` 仍 0 条、独立数 239 / 严格 237。

`v1.1` 只做一件事：把 `v1.0` 的三处 P0 缺陷（源码集不完整、`git_diff` 不能重建、
按类型选配置键）与三处内部不一致改掉。**D-1～D-4 已裁决**（§13），但**实现仍未授权**：
字段表、哈希纳入表、`recorded` 判据要先冻结成文本，再写 runner。

为什么先写契约，而不是先去改 runner：`run_id` 是 `RUN_ID_BASIS` 六个字段的哈希，
**加一个字段就等于换一次号**（0.5.0 → 0.7.0 已经把 239 行全换过一遍）。字段定义错一次，
判重结果和独立数就跟着错一次，而且是**静默**错——没有任何报错。这一条是
`HISTORICAL_EXEC_IDENTITY_GAP.md` §6 的直接结论：缺口只能向前关，而向前关的门必须先画好。

---

## 0. 三个不能互相替代的问题

用户在审查里点出的三分法，是本契约的第一条，也是后面所有字段的判据来源：

| 问题 | 问的是什么 | 靠什么回答 | 常见误写 |
|---|---|---|---|
| **同条件**（condition） | 两次运行是否同一组执行条件 | 进 `run_id` 的那些字段 + `recorded` 门槛 | 写成"可逐位复现" |
| **同实例**（instance） | 是否来自同一次实际运行 | `instance.*` + 产物 `path`/`sha256` | 拿实例摘要当条件哈希 |
| **可重建**（reconstruct） | 有没有材料重新造出当时的代码 | `patch_bundle` + `declared_source_set` | 有 SHA256 就说"能恢复原文件" |

第四层只是标签，不是身份：**包络**（envelope）指这次运行落在哪个契约世代
（`identity_contract_version` + `run_id` 前缀），进哈希但只用于防止跨世代撞号。

`init_values_sha256`、`chain_keys`、`started_at`、`elapsed_s` 全部是**实例**；
把它们塞进 `run_id` 会让"同条件跑两次"换号，去重直接失效——**这是最该防的反向错误**。

**同条件 ≠ 逐位相同**（D-3 的认识边界）：相同 `run_id` 只表示条件字段一致，不表示两次
浮点结果逐位相等。线程数、归约顺序、硬件实现都可能改变末位；这些当前不量化、也不进身份。

---

## 1. 失败模式清单

每一条都是已经发生过、或差一步就会发生的，字段设计逐条对着它们：

| # | 失败模式 | 具体形态 | 对策 |
|---|---|---|---|
| F1 | **用上界当身份** | mtime 给区间，取区间前的最近提交当 `code_version` | `git_commit` 只能由运行时 `git rev-parse` 写入（§3.1） |
| F2 | **用入口脚本当代码** | 入口 `.py` 逐字节相同，`bm_top`/`nested_fit` 无副本 | 声明集 + 运行时加载集双向核对（§3.1） |
| F3 | **`--dirty` 当完整指纹** | 两次都 `v0.7.0-3-gabcdef-dirty`，改动内容完全不同 | `git_describe` 只作展示；dirty 降为范围判据（§4） |
| F4 | **档名/默认值冒充实际值** | `budget_profile="standard"` 被当成 `target_accept` 数值 | `config.effective` 在采样调用点捕获，带 `origin`（§3.3） |
| F5 | **一个 seed 号含糊带过** | 初始化固定、采样随机流随机，共用一个 `seed` 字段 | 拆 `rng.plan`（条件）/ `rng.actual`（实例）（§3.5） |
| F6 | **实例字段误入条件哈希** | 把 `init_values_sha256` 放进 `run_id` | §6 纳入表整行排除实例字段 |
| **F7** | **`sys.modules` 被当成完整代码集**（v1.1 新增） | 延迟导入 / 动态 import / 声明之外的路径加载的项目模块 | 声明集与运行时集**分离**，`runtime ⊆ declared` 强制核对（§3.1） |
| **F8** | **`git diff` 被当成可重建**（v1.1 新增） | 未跟踪的 `new_model.py`：文件名进了清单，内容没进 diff | 补丁包必须含未跟踪文件**内容**，否则不许说"可重建"（§7.2） |
| **F9** | **按类型选配置键**（v1.1 新增） | 只哈希数值 ⇒ `include_L`/`branch`/`adsorption` 这类开关漏掉，两个不同条件拿到同一个号 | 判据改为"是否改变算什么/怎么算"，键集合带类型与效果分类（§3.3） |
| **F10** | **摘要声称覆盖未参与计算的字段**（v1.1 新增） | 说 `python` 已被 `env_sha256` 覆盖，而摘要只由包名版本拼成 | 解释器版本进摘要，或单列进哈希（§3.2） |

F1/F2/F3 是这次审计的对象；F4/F5/F6 是这批 239 条记录里**同样存在、但被"代码版本"挡住了**
的问题——`R-` 0 条让人以为只缺一个字段。实际上就算补上代码版本，`budget_source` 仍有 233 条是
`derived_from_tier`，seed 仍有 233 条无产物级证据。F7～F10 是 v1.0 草案自己的缺陷。

---

## 2. 五个新块，叠在已有两层上

```
execution
├── candidate_id          模型定义（已有，不动）：拓扑 + 元件 + 配置 + init + 预算档名 + 似然
├── data                  数据身份（已有，扩两点）：内容哈希 + 频率窗 + 点集
├── code                  代码身份（新）：commit + 声明源码集摘要 + 运行时加载集 + dirty 范围 + 补丁包
├── env                   环境身份（新）：解释器 + 包集合 → env_sha256，加 backend/x64
├── config                运行配置（新）：实际生效值 + origin + 调用接口 + 键集合
├── rng                   随机流（新）：plan（条件）+ actual（实例）
├── instance              运行实例（新）：产物定位 + 起止时刻 + 初始化值摘要
└── software              兼容层（保留）：producer、code_version（= describe，人读）、registry_version
```

**兼容原则**：`software.code_version` 语义不变（人读标签），但**不再承担身份**；
`code.*` 才是身份。239 条老记录、现有读取方、`_check_software()` 全都不动，
新字段只在 v1 记录里出现。

---

## 3. 字段表

列含义：**必填**= v1 `recorded` 是否要求它存在且合法；**进 `run_id`**= 是否进哈希；
**缺失后果**= 不发 / `derived` / 不影响。

### 3.1 `code` —— 代码身份（v1.1 重写）

**核心改动**：`v1.0` 让 `source_tree_sha256` 同时承担"声明"和"实际"两件事，做不到。
现在拆成两套事实 + 一条强制核对，另加一个补丁包满足"可重建"。

| 字段 | 类型 | 来源 | 必填 | 校验 | 缺失后果 | 进 `run_id` |
|---|---|---|---|---|---|---|
| `git_commit` | str | 运行时 `git rev-parse HEAD` | ✓ | 40 位十六进制；**不许短哈希** | `derived` / `code_commit_missing` | ✓ |
| `declared_source_set` | list[{path, sha256, role}] | 声明清单（§3.1.1） | ✓ | role ∈ {entry, module, config, asset}；path 为 POSIX 相对路径；按路径码位排序；无重复 | `derived` / `declared_set_missing` | ✗（摘要由它算） |
| `declared_source_tree_sha256` | str | 对上面这份集合求摘要 | ✓ | 64 位十六进制，可按 `declared_source_set` 复算 | `derived` / `declared_set_missing` | ✓ |
| `declared_not_loaded` | list[str] | 声明集 − 运行时集 | ✓（可为空表） | 子集关系必须成立 | —（只记录） | ✗ |
| `runtime_loaded_modules` | list[{path, role}] | 三个时点的 `sys.modules` 快照并集（§3.1.2） | ✓ | 只含项目代码；realpath 记录 | `derived` / `runtime_snapshot_missing` | ✗ |
| `runtime_snapshot_taken` | bool | runner | ✓ 且必须 `true` | 静态扫描得出的集合**不算** | `derived` / `runtime_snapshot_missing` | ✗ |
| `runtime_snapshot_points` | list[str] | runner | ✓ | 至少含 `"before_fit"` 与 `"after_fit"` | 同上 | ✗ |
| `source_coverage` | enum | 由声明集、运行时集、模块根求交 | ✓ | `complete`/`undeclared_loaded`/`external_path`/`snapshot_missing` | 非 `complete` ⇒ `derived`（原因同值） | ✗ |
| `undeclared_modules` | list[str] | 运行时集 − 声明集 | ✓（可为空表） | 必须为空才 `complete` | 同上 | ✗ |
| `git_dirty` | bool | `git status --porcelain -uall` | ✓ | 必须是 `true`/`false`，**不许 `null`** | `derived` / `dirty_unknown` | ✗（见 §4） |
| `dirty_paths` | list[str] | 同上 | ✓（可为空表） | POSIX 相对路径 | `derived` / `dirty_unknown` | ✗ |
| `dirty_scope` | enum | 由 `dirty_paths`、声明集、排除集三方求交 | ✓ | `clean`/`code_only`/`noncode_ignored`/`stray`/`unknown` | `stray`/`unknown` ⇒ `derived` | ✗ |
| `patch_bundle_ref` | str? | runner 落盘路径 | 条件性（`git_dirty=true`） | 相对路径，文件存在 | `derived` / `bundle_missing_when_dirty` | ✗ |
| `patch_bundle_sha256` | str? | 对补丁包字节求摘要 | 条件性（`git_dirty=true`） | 64 位十六进制 | 同上 | ✗ |
| `bundle_excluded` | list[{path, reason}] | 超限/排除项 | 条件性 | 若含**声明集**里的文件 ⇒ 视为 `patch_bundle_sha256` 无效 | 同上 | ✗ |
| `git_describe` | str | `git describe --tags --dirty --always` | ✗ | 非空 | 只影响可读性 | ✗（**明确不作身份**） |

#### 3.1.1 声明集怎么来（`declared_source_set`）

声明集是**代码身份的分母**，必须可复算、可审查，不能"跑到哪算哪"：

1. **静态导入闭包**：对入口脚本做 AST 分析，递归展开 `import`/`from ... import`，
   包括**函数体内的延迟导入**（AST 能看见它们，这是它比 `sys.modules` 强的地方）；
2. **显式补充**：`source_manifest.json`（入库文件）里手工列出的项——动态
   `importlib.import_module(<计算出的名字>)`、模板/配置/先验表等无法静态求出的依赖，
   每项必须写 `reason`；
3. **角色标注**：`entry`（入口脚本）、`module`（项目模块）、`config`（影响数值的配置/先验表）、
   `asset`（其它必须随代码一起的静态资源）；
4. **摘要**：`sha256("v1\n" + 逐行 "<posix_path>\0<content_sha256>\n" 按路径码位排序)`；
   行尾按 `\n` 归一（`*.py`/`*.toml`/`*.json`/`*.yaml`/`*.cfg`），其余按字节；
   归一规则写进 `config.normalize`。

**排除**：`.git/`、`out/`、`evidence/`、`__pycache__/`、`*.pyc`、`docs/`、`archive/`、
数据文件（数据属 `data` 块，重复计算会让"改数据"和"改代码"混成同一种变化）。

#### 3.1.2 覆盖规则（这是 F7 的唯一对策）

| 关系 | 判定 | 处理 |
|---|---|---|
| 运行时 ⊂ 声明 | `complete` | 正常；`declared_not_loaded` 如实记录（声明了但这次没用到，无害） |
| 运行时 ⊄ 声明 | `undeclared_loaded` | **降 `derived`**，`undeclared_modules` 列出全部；**不许当场补进声明集**（补了就是改身份） |
| 模块 realpath 在声明根之外（editable install 指向别处 / site-packages 里有一份项目副本） | `external_path` | **降 `derived`**，列出 `external_modules` |
| 三个时点快照没取全 | `snapshot_missing` | **降 `derived`** |

三个时点：`before_fit`（进入拟合前）、`sampler_entry`（进入采样器前，若能挂）、`after_fit`（返回后）。
取并集，避免"延迟导入发生在拟合中途"被漏掉。

**因此 `declared_source_tree_sha256` 的含义是"已声明源码集合摘要"**，它的完整性来自
**覆盖规则被强制核对**，而不是来自"我们从 `sys.modules` 里看到了多少"。这是 `v1.0`
最实质的一处修正：`v1.0` 说"盖不住未加载但会被 import 的下一次"，现在这句话不再成立——
没被声明的东西一旦真的加载，运行就降级。

### 3.2 `env` —— 环境身份（v1.1 修 F10）

| 字段 | 类型 | 来源 | 必填 | 校验 | 缺失后果 | 进 `run_id` |
|---|---|---|---|---|---|---|
| `python` | str | `platform.python_version()` | ✓ | `x.y.z` | `derived` / `env_missing` | 经 `env_sha256` |
| `python_impl` | str | `platform.python_implementation()` | ✓ | 非空 | 同上 | 经 `env_sha256` |
| `platform` | str | `platform.platform()` | ✓ | 非空 | 同上 | 经 `env_sha256` |
| `packages` | {name: version} | `importlib.metadata.version` | ✓ 非空 | 键按字典序；版本非空 | 同上 | 经 `env_sha256` |
| `env_sha256` | str | 见下 | ✓ | 64 位；可由上述四项复算 | `derived` / `env_missing` | ✓ |
| `backend` | str | `jax.default_backend()` | ✓ | `cpu`/`gpu`/`tpu` | `derived` / `env_missing` | ✓ |
| `x64_enabled` | bool | `jax.config.jax_enable_x64` | ✓ | bool | `derived` / `env_missing` | ✓ |
| `device_kind` | str | `jax.devices()[0].device_kind` | ✗ | 非空 | — | ✗ |
| `cpu_threads` | int | `OMP_NUM_THREADS` 或实际线程数 | ✗ | >0 | — | ✗（D-3） |
| `hardware_tag` | str? | 设备标识（型号/驱动） | ✗ | 非空 | — | ✗ |
| `lock_digest` | str? | 锁文件内容 | ✗ | 64 位 | — | ✗（锁文件是**意图**，`env_sha256` 才是**事实**） |

**`env_sha256` 的定义（修正 v1.0 的错误声称）**：

```
sha256(
  "python==" + python + "\n" +
  "impl=="   + python_impl + "\n" +
  "platform==" + platform + "\n" +
  "pkg==" + name + "==" + version + "\n"  × 全部包（按名字典序）
)
```

v1.0 把 `python`/`python_impl` 写成"被 `env_sha256` 覆盖"，但当时的摘要只由包名版本拼成——
**两个 Python 版本不同、包完全相同的环境会拿到同一个 `env_sha256`**。这是"把没参与计算的
字段说成已被覆盖"的典型错误，现按 F10 修掉：解释器版本与平台进摘要，不再另列。

`cpu_threads`/`device_kind`/`hardware_tag` 只记录、不进哈希（D-3），并在 §0 写明
"同 `run_id` ≠ 逐位相同"。若将来随机流敏感性实验证明线程配置显著改变结果，
**升契约版本**处理，不在 v1 里偷偷改语义。

### 3.3 `config` —— 实际生效的运行配置（v1.1 重写）

| 字段 | 类型 | 来源 | 必填 | 校验 | 缺失后果 | 进 `run_id` |
|---|---|---|---|---|---|---|
| `effective` | {name: `{value, origin, effect}`} | **采样调用点**捕获 | ✓ | `origin ∈ {cli, config_file, default, tier, derived}`；`effect ∈ {model_definition, execution_condition, display_only}` | `derived` / `config_incomplete` | 仅 `effect ∈ {model_definition, execution_condition}` 的键 |
| `capture_interface` | {callable, module, line, kwargs_rule} | runner | ✓ | `callable` 必须是被实际调用的那一层 | `derived` / `config_incomplete` | ✗ |
| `unknown_keys` | list[str] | 见下 | ✓ 必须 `[]` | 空表 | `derived` / `config_incomplete` | ✗ |
| `unclassified_keys` | list[str] | 有 `effective` 但不在闭集、也没给 `effect` 的键 | ✓ 必须 `[]` | 空表 | `derived` / `config_unclassified` | ✗ |
| `normalize` | str | 由本契约固定 | ✓ | `eol-lf-text` | — | ✗ |

#### 3.3.1 判据：改变"算什么/怎么算"，不是"是不是数字"

这是 P0-3 的修法。`v1.0` 的 `NUMERIC_CONFIG_KEYS` 会漏掉布尔与枚举开关，而后者的影响往往
更大——本仓库里现成的例子（`bm_select.fit_voigt` / `fit_nested` 的真实签名）：

| 键 | 类型 | `effect` | 为什么 |
|---|---|---|---|
| `include_L` | bool | `model_definition` | 打开串联 `jωL`，族空间变了 |
| `adsorption` | enum(str\|None) | `model_definition` | 打开 `R·jωτ/(1+jωτ)` 低频支路 |
| `branch` | enum | `model_definition` | `parallel` / `series` 是两种拓扑 |
| `layers` | int | `model_definition` | 嵌套层数 |
| `sigma_floor` | float | `execution_condition` | 改变似然尺度 |
| `sigma_mode` | enum | `execution_condition` | `make_sigma` 的口径 |
| `progress` | bool | `display_only` | 只影响输出，不进哈希 |

所以闭集改名为 **`RUN_ID_CONFIG_KEYS`**，并带 `run_id_config_keys_version: 1`。
闭集的目的**不是**限制类型，而是**防止 `run_id` 随无害开关抖动**（每个多余键都把样本切碎）；
类型只是每行的一个属性。

不进哈希、也不重复记录的键：`seed`（在 `rng.plan`）、五项预算（在 `budget_params`）、
`init_strategy`（在 `rng.plan`）。§13 D-2 规定了这层权威关系。

#### 3.3.2 `unknown_keys` 怎么求（不能只靠 `inspect.signature`）

必须先钉住"捕获的是哪个接口"，否则会把内部参数当采样配置、或漏掉经 `**kwargs` 传进来的东西：

1. `capture_interface.callable` 记**实际被调用的那一层**（例如 `bm_select.fit_voigt`，
   而不是它内部再调用的 `bm_real.fit_one`）；`module`/`line` 供核对；
2. 该 callable 签名里每个**显式命名参数**都必须出现在 `effective`；
3. 若签名有 `**kwargs` 捕获器：凡实际经它传入的键，必须**按名字**逐项出现在 `effective`，
   不许整体塞成一个字典——否则"传了什么"就查不到；
4. 没有出现在 `effective` 的签名参数 ⇒ `unknown_keys`；出现了但不在 `RUN_ID_CONFIG_KEYS`
   且没标 `effect` ⇒ `unclassified_keys`；
5. 两者非空都降级。`unclassified` 单列一条原因，因为它的含义是"有个开关我们不知道它改不改条件"，
   不能和"漏记录"混成一种。

**默认值的界线**（与 §9 一致）：`origin=default` 的值是**生效值**，可以 `recorded`；
区别在捕获位置——runner 在调用点把默认值物化成显式实参并记下来，是"记录实际值"；
事后从代码里读一遍默认值是"猜当时用了什么"，不行。

### 3.4 `data` —— 数据身份（已有，扩两点）

| 字段 | 类型 | 必填 | 校验 | 进 `run_id` |
|---|---|---|---|---|
| `ref` | str | ✓ | 非空（**标签**，不作身份） | 只在无内容哈希时退用 |
| `sha256` | str | ✓ | 64 位十六进制 | ✓ |
| `f_min` / `f_max` | float? | ✗ | 数值 | ✓（频率窗改变 = 不同条件） |
| `n_points` | int | ✓（v1 新增） | >0，且 ≤ 源文件行数 | ✓ |
| `subset_sha256` | str? | 条件性（v1 新增） | 64 位；施加了掩码/抽样才要 | ✓（有掩码时） |
| `id_kind` | enum | ✓ | `content_sha256`/`ref_only` | ✗（哈希后看不出来，必须留痕） |

为什么要 `n_points`/`subset_sha256`：同一份 CSV 用 50 点还是 60 点、按频率窗截还是按行号截，
是**两种不同的输入**。只哈希文件内容会把它们合成同一个条件。

### 3.5 `rng` —— 随机流（条件 / 实例分开）

**A. 生成规则（条件，进 `run_id`）—— 五项**

| 字段 | 类型 | 必填 | 校验 |
|---|---|---|---|
| `plan.root_seed` | int | ✓ | 整数；**必须由产物写入**（沿用 `seed_source` 的 `artifact` 判定） |
| `plan.prng` | enum | ✓ | `jax.threefry2x32` / `numpy.PCG64` / `numpy.SeedSequence` |
| `plan.derivation` | str | ✓ | 规则串，如 `numpyro.handlers.seed(node, key); chain i ← split(key, n)[i]` |
| `plan.n_chains` | int | ✓ | 必须等于 `budget_params.num_chains` |
| `plan.init_strategy` | str | ✓ | 档名（`init_to_uniform` / `init_to_median` / `custom:<name>`） |

`v1.0` 在 §5 写"四项齐"、§3.5 却列了五项，现统一为**五项**（§5 同步改）。

**B. 实际结果（实例，不进 `run_id`）**

| 字段 | 类型 | 必填 | 校验 |
|---|---|---|---|
| `actual.chain_keys` | list[str] | ✓ | 长度 = `n_chains`；可从 `root_seed`+`derivation` 复算 |
| `actual.init_values_sha256` | str? | 固定初值时 ✓ | 64 位；对**实际传入**的初值求摘要 |
| `actual.init_values_path` | str? | ✗ | 产物内位置（如 `artifact:config.init`） |

`init_values_sha256` 的规范化口径：按键排序的 `canonical_json`，浮点用 `repr`（17 位有效数字），
**不四舍五入**——四舍五入会把"改了第 12 位"这一整类改动抹平。

这一块直接对着 `IDENTIFIABILITY_REPORT_PREREG.md` 与 B3b 的 `sampler_sensitive`：
"同一个 seed 配置"和"实际采样入口相同"是两件事，只有当 `init_values_sha256` 也相同，
才谈得上"同一次执行的同一随机流"。

### 3.6 `instance` —— 运行实例

`artifact_path` / `artifact_sha256`（与 `same_artifact` 共用定位）、`started_at` / `finished_at`、
`elapsed_s`、`pid` / `host`、`init_values_sha256`。**全部不进 `run_id`**，
只用于"这份产物是哪一次跑出来的"和取证。

---

## 4. dirty tree：`--dirty` 不是代码指纹

两次都 `v0.7.0-3-gabcdef-dirty`，未提交改动可以完全不同。所以 dirty 从"身份"降为"范围判据"：

| `dirty_scope` | 判据 | 对 `recorded` 的影响 |
|---|---|---|
| `clean` | `dirty_paths == []` | 无影响 |
| `code_only` | 非空，全部落在 `declared_source_set` 内 | 无影响（tree digest 已反映），但**补丁包必落** |
| `noncode_ignored` | 非空，全部落在排除集（`docs/`、`out/`、`evidence/`…） | 无影响（不参与计算，已在 `dirty_paths` 留痕） |
| `stray` | 有一条路径既不在声明集、也不在排除集 | **降 `derived`**（`dirty_outside_source`） |
| `unknown` | 没有 git、或 `git status` 失败 | **降 `derived`**（`dirty_unknown`） |

真正危险的是 `stray`：改动落在没人声明的地方（比如一个偷偷改了却没进声明集的
`params.json`），这时"改了什么"就不知道了。`dirty_paths` 而不是 `--dirty` 那个布尔值，
才让这条判据成立。

`git_dirty` 为什么**不进哈希**：同一个 commit、同一个源码树，脏与不脏对数值没有影响；
把布尔值塞进 `run_id` 会让"改了 README 再跑一次"换号，判重白丢一条真重复。
**换号改不了正确性，只会让样本变少。**

---

## 5. 六项证据凑齐才算 `recorded`（不新增状态）

v1 **不引入新的状态**，仍是 `recorded` / `derived` / `unconfirmed` 三档——状态集合在
`IDENTITY_STATES` 里，改它要动 schema、动 239 条记录的判定口径、动 `duplicate_verdict`。
v1 只做一件事：把 `recorded` 的**必填集**写清楚。

| 块 | `recorded` 要求 | 缺了会怎样 |
|---|---|---|
| `data` | `id_kind == content_sha256` + `n_points` | `unconfirmed`（不发 `run_id`） |
| `rng` | `seed_source` 前缀 `artifact` + `plan` **五项**齐 + `init_values_sha256`（固定初值时） | `unconfirmed` |
| `budget_params` | 五项全 `source == recorded`，且与 `config.effective` 重叠项逐项一致（§13 D-2；不一致 ⇒ `budget_config_mismatch`） | `unconfirmed` |
| `config` | `unknown_keys == []` 且 `unclassified_keys == []` 且 `capture_interface` 完整 | `derived` |
| `code` | `git_commit` + `declared_source_tree_sha256` + `runtime_snapshot_taken` + `source_coverage == complete` + `dirty_scope ∉ {stray, unknown}` + **`git_dirty` 时 `patch_bundle_sha256` 有效** | `derived` |
| `env` | `env_sha256` + `backend` + `x64_enabled` | `derived` |

`v1.0` 的表漏了"dirty 时补丁包必落"这一条（§3.1 写了必填、§5 没写），现补上：
**dirty 而没有可用补丁包 ⇒ 不能 `recorded`**，因为这时"可重建"不成立，而 `recorded`
是对外宣称"这次运行可复核"的那一档。

**为什么两组缺失后果不同**：`data`/`rng`/`budget` 缺了，我们连"跑的是什么"都不知道，
这时发任何 id 都是编造（现有规则，`RUN_ID_MISSING` 三个真实原因 + `not_provided` 哨兵）；
`code`/`env`/`config` 缺了，条件仍大致可描，发 `D-` 但必须标出缺哪一项。
**`D-` 号永远不能当判重依据**——两条 `D-` 相同只说明"在已知的那部分条件上相同"。

---

## 6. `run_id` 纳入口径

`run_id` 是**条件**指纹，判据只有一条：**这个字段变化时，两次运行是否已经不是同一组执行条件？**

| 字段 | 进 `run_id` | 理由 |
|---|---|---|
| `candidate_id` | ✓ | 模型定义层（已有，不动） |
| `data.sha256` / `n_points` / `subset_sha256` / `f_min` / `f_max` | ✓ | 换了数据就是换了实验 |
| `rng.plan.*`（五项） | ✓ | 随机流来源不同 = 条件不同 |
| `budget_params`（五项 + source） | ✓ | 已有；数值与来源都进，档名不顶替 |
| `config.effective` 中 `effect ∈ {model_definition, execution_condition}` 的键 | ✓ | 判据见 §3.3.1；含布尔/枚举（`include_L`/`branch`/`adsorption`/`layers`…） |
| `code.git_commit` + `code.declared_source_tree_sha256` | ✓ | 基线 + 实际声明代码（缺一不可：commit 覆盖不到未跟踪文件，摘要覆盖不到未声明模块——后者由 `source_coverage` 核对兜住） |
| `env.env_sha256` + `backend` + `x64_enabled` | ✓ | 环境会改数值；`x64` 改 dtype |
| `identity_contract_version` | ✓ | 世代标签，防"语义变了但载荷没变"撞号（D-4） |
| `code.git_describe` / `git_dirty` / `dirty_paths` / `dirty_scope` / `patch_bundle_*` | ✗ | 展示、范围判据或重建；由 commit+摘要或其自身派生 |
| `code.runtime_loaded_modules` / `source_coverage` / `undeclared_modules` | ✗ | 覆盖性核对，不区分条件（用作降级判据） |
| `config.unknown_keys` / `unclassified_keys` / `capture_interface` / `normalize` | ✗ | 校验与溯源 |
| `rng.actual.*` / `instance.*` | ✗ | **实例**——进来会让同条件两次跑换号，判重失效（F6） |
| `software.registry_version` | ✗（D-1 已定） | 它是**映射工具**版本，不是被拟合的代码；移入 `provenance` |

`RUN_ID_CONFIG_KEYS` 必须是**闭集 + 显式版本化**（`run_id_config_keys_version`），
否则 `run_id` 会随任何无害开关抖动，样本量被无意义地切碎。

---

## 7. 校验、重建与不可自证

### 7.1 三层校验，各管一段

| 层 | 谁做 | 查什么 |
|---|---|---|
| 写入时 | runner 自检 | 字段格式（40/64 位十六进制）；`unknown_keys`/`unclassified_keys` 为空；`dirty_scope` 与 `dirty_paths` 自洽；`runtime ⊆ declared`；`declared_source_tree_sha256` 与声明集可复算；`env_sha256` 可复算；dirty 时补丁包存在且可复算 |
| 读入时 | registry（`_check_execution` 现有机制） | 整块复算 `status`/`reason`/`run_id` 与观测对拍；带 `identity_contract_version ≥ 1.0` 的按 v1 必填集检查 |
| 冻结时 | `map_stage0_records.py --check` | 文本 vs 冻结字节、清单 vs 磁盘、指纹、`source_sha256` / `data_sha256_files` 逐份重算 |

### 7.2 「身份」与「可重建」是两件事（v1.1 重写）

v1.0 声称"用 `dirty_paths` + `git_diff` 全文重建工作树"，这是**错的**：
`git diff HEAD` 不含未跟踪文件的**内容**。未跟踪的 `new_model.py` 只有名字进清单，
内容既不在 diff 里、也无法从 `source_set` 的 SHA256 反推。**有哈希 ≠ 能恢复。**

现在按"可重建优先"定义补丁包（`code.patch_bundle_*`），内容必须包含：

1. `git diff HEAD --binary`（已跟踪文件的全部改动，含 staged + unstaged）；
2. `dirty_paths ∪ declared_source_set` 中**未跟踪或被 ignore** 的文件**完整内容**；
3. `bundle_manifest.json`：逐项 `{path, sha256, mode, source: diff|content}` + `git_commit`；
4. 排除项写进 `bundle_excluded`（单个超过 `BUNDLE_MAX_BYTES`、或落在 `out/`/`evidence/`/数据目录）。

**判据**：`bundle_excluded` 里只要出现**声明集**中的文件，`patch_bundle_sha256` 即视为无效、
降 `derived`——声明集里的东西本来就该能重建，不能一边说它是代码一边不存它。

**重建校验（实现阶段做，现在只定口径）**：
`git checkout <git_commit>` → 解开补丁包 → 复算 `declared_source_tree_sha256`，
必须与产物里记录的一致。做不到这一步，"可重建"就只是措辞。

### 7.3 不能自证的部分（必须承认）

- `declared_source_tree_sha256`、`env_sha256`、`patch_bundle_sha256` 都是 **runner 自称**的值。
  第三方能验证的是"与它落下的 `declared_source_set` / `packages` / 补丁包一致"，
  不能验证"这个进程真的加载了这些文件"。
- 覆盖规则（§3.1.2）把"漏声明"变成**可发现的降级**，但它只能发现**这次真的加载过**的模块；
  一条从不执行的动态导入路径，声明集里没有也不会被发现——这时 `source_coverage=complete`
  仍成立，含义是"本次执行覆盖完整"，不是"这个程序所有可能路径都已声明"。
- 所以 §7.2 的补丁包不是装饰：它是从"信 runner"走向"能复核 runner"的唯一材料。

---

## 8. 版本化：v0 与 v1 两套逻辑，不许互相套用

```
identity_contract_version: "1.0"     # 新记录必须带；缺省 = v0（现行 0.7.0 规则）
run_id_prefix: "R1-"/"D1-"           # D-4 已定；前缀只是辅助标识
```

- **两套判定**：`_expected_status()` 现在只看四项（数据内容哈希、seed 来源、预算
  `recorded`、`code_version` 非空），**这是 v0 的定义，不改**；v1 的必填集见 §5。
  读入时按记录自带的 `identity_contract_version` 选规则，缺省走 v0。
- **不回改、不倒灌**：239 条没有这个字段 ⇒ 按 v0 读，判定不变。**不许**用 v1 规则重算旧记录
  再声称"旧记录身份升级"（D-1 的同一条约束）。旧记录的 `run_id` 也不会因为 v1 落地而变。
- **必然换号**：v1 的哈希载荷加了字段，新记录与旧记录不可能同号。这是"跨版本比号无意义"
  的又一处实例（`STAGE0_RECORDS_MAP.md` §3.1 已写明）。
- **D-1 的落地方式**：`registry_version` 移入 `provenance`，`software` 不再整块进哈希；
  `identity_contract_version` 进哈希。注意 `_run_id_from()` 目前哈希的是**整个 `software` 块**，
  所以这是一次真正的算法改动，必须换 `REGISTRY_VERSION`。
- **并存策略**：stage 0 快照冻结后**不再写入**；v1 记录落新目录（如 `evidence/stage1_runs/`），
  两套各自 `MANIFEST`。历史基线的字节不会被未来的运行改动。

---

## 9. 硬规则：默认值不许自动获得 `recorded`

| 情形 | 禁止 | 允许 |
|---|---|---|
| 拿不到 git 信息 | 填 `unknown` / 当前日期 / 注册表版本 / 上次运行的值 | `code = null` ⇒ `derived`，原因 `code_commit_missing` |
| seed 未被捕获 | 用配置文件里的预设 seed 顶替 | 没有 seed ⇒ `unconfirmed`（现状） |
| budget 未捕获 | 由档名推导后标 `recorded` | `derived_from_tier`（现状，且不能升 `recorded`） |
| dirty 范围不明 | 宣称代码身份完整 | `dirty_scope=unknown` ⇒ `derived` |
| dirty 但补丁包不全 | 说"可重建" | 降 `derived`，`bundle_excluded` 列出缺口 |
| 模块未声明却加载了 | 当场补进声明集 | `source_coverage=undeclared_loaded` ⇒ `derived` |
| 配置键不认识 | 默认它"不影响结果" | `unclassified_keys` 非空 ⇒ `derived` |
| 数据只有文件名 | 用文件名的哈希当内容哈希 | `ref_only` ⇒ 最高 `derived` |

**唯一的例外**（与 §3.3.2 同一条）：`origin=default` 的配置值是**生效值**，可以 `recorded`。
界线在捕获位置，不在值本身。

---

## 10. 6 条重跑：是重建基准，不是补历史

**目标定为「按现行 runner 规范重新执行」，不是「复现原运行」。** 理由很硬：原运行的 seed
来源、初始化值、依赖版本三项都不全，严格复现不可达；说"复现"是给不可达的目标贴标签。

| 规则 | 内容 |
|---|---|
| 旧记录 | 6 条保持 `derived`，`record_id` 不变，永远不拿新 run 的身份去填 |
| 新记录 | 新 `record_id`、新 `run_id`、`purpose: "baseline_rebuild"`、`related_history: [旧 id…]` |
| 指针方向 | **只允许新 → 旧**；旧记录不得出现指向新 run 的字段（防身份倒灌） |
| 数值比较 | 允许并鼓励（Rwp/acf/参数），但必须标注"跨身份世代比较"，差值不可归因于单一因素 |
| 计数 | 旧 6 条（4 次执行）与新 6 条（4 次执行）各自成批；**不得**因此把 239 改成 237，也不得把 237 改成 231 |

一对 `possible_duplicate` 会在新记录里自然变成 `confirmed_duplicate`（因为 v1 给足了证据）——
**那是新记录之间的判定，不是给旧记录翻案。**

## 11. 233 条：暂缓，归 Stage 1

不重跑。重跑前必须先回答五个问题（否则只是把历史上的不规范流程再执行一遍）：

1. 哪些记录有明确的**科学比较价值**（对应哪条验收线、哪张图、哪个结论）？
2. 哪些只是探索性试跑（`exploratory`）？
3. 哪些记录之间**本来就不是可比条件**（模型族/窗口/预处理不同）？
4. 原有初始化策略要不要保留（`init_to_uniform` 是"没传"的结果，不是选择）？
5. 重跑后**实验单位**是什么（谱？配置？条件？）——这直接决定独立数怎么数。

这五个问题属于 `STAGE2_SPEC.md` 的实验设计，不属于 stage 0 的历史修复。

## 12. Stage 0 的完成判据（不是 `R-` 数量）

| # | 判据 | 现状 |
|---|---|---|
| 1 | 历史记录全部完成审计（239/239 有覆盖，缺口分类齐） | ✅ `coverage.json` + `HISTORICAL_EXEC_IDENTITY_GAP.md` |
| 2 | 可恢复的证据有明确来源（文件 + 字段 + 复算方式） | ✅ 数据内容哈希 239/239；6 条 seed 有产物来源 |
| 3 | 不可恢复的明确标为不可恢复（原因 + 为什么不可恢复） | ✅ 9 份产物零身份字段、mtime 只给上界、库模块无副本 |
| 4 | 判重与独立性可复算（双口径同报、三套测试、`--check`） | ✅ 契约 76/76、快照 9/9、`--check` rc=0 |
| 5 | 后续 runner 有一套**经过实跑验证**的身份契约 | ❌ 本文件是草案，未实现、未实跑 |

**冻结判据**：1–4 成立 + 5 的契约被审查通过并至少经一次真实运行验证（P1）。
不由 `R-` 数量决定——233 条永远 `derived` 也可以是可信的历史基线；把"重跑补齐"当冻结条件，
等于用未来的运行给过去盖章。

---

## 13. 裁决记录（v1.1）

### 13.1 D-1～D-4：已定

| # | 裁决 | 约束（写进实现要求） |
|---|---|---|
| D-1 | **同意移出哈希**，`registry_version` 只写 `provenance`；由 `identity_contract_version` 进哈希 | v0 记录仍用原 `RUN_ID_BASIS`；v1 用新规则；**不能拿新规则重算旧记录再宣称身份升级**。实现上必须换 `REGISTRY_VERSION`（`_run_id_from()` 现在哈希整个 `software` 块） |
| D-2 | **同意保留 `budget_params`**，`config.effective` 补全 | `budget_params` = 五项预算的**规范化身份字段**；`config.effective` = 完整运行配置记录；两者重叠的五项**逐项一致**（不一致 ⇒ `budget_config_mismatch` ⇒ `derived`）；`config` 的哈希部分**不重复包含**这五项（已由 `budget_params` 进哈希）；`NUMERIC_CONFIG_KEYS` → `RUN_ID_CONFIG_KEYS`（类型无关） |
| D-3 | **同意先只记录、不进哈希** | 同时记录 `cpu_threads` / `backend` / `device_kind` / `hardware_tag`；**明确写入"同 `run_id` ≠ 逐位相同"**（§0）；将来若证据显示显著影响，**升版本**处理，不偷改 v1 语义 |
| D-4 | **同意改 `R1-`/`D1-`**，但前缀只作辅助标识 | `identity_contract_version` 才是权威；校验时**同时检查**状态、契约版本、前缀三者一致；实现前要审计所有解析 `run_id` 的地方 |

D-4 的实现影响已核对：前缀的**写入**只有 `_run_id_from()`（`circuit_registry.py:1222`）、
**校验**只有 `_check_execution()`（`:1320-1328`），另有 48 处测试断言含 `run_id`。
`grep` 时的坑：`docs/round_2026-11-12/noise_acf_truth.py:109` 的 `"R1-[P2"` 是**谱文件名**、
`truth_recovery.py:203` 的 `k.startswith("R")` 是同一批谱名，都不属于 `run_id` 前缀，别误改。

### 13.2 P0-1～P0-3：已按审查意见修订

| 项 | 修订 | 落在哪 |
|---|---|---|
| P0-1 `sys.modules` 不能单独承担 | 拆 `declared_source_set`（静态导入闭包 + 显式清单）与 `runtime_loaded_modules`（三时点快照）；强制 `runtime ⊆ declared`，违反 ⇒ `undeclared_loaded` 降级；`source_tree_sha256` 改名 `declared_source_tree_sha256`，含义限定为"**已声明**集合摘要" | §3.1.1、§3.1.2 |
| P0-2 `git_diff` 不能重建 | 选"**可重建优先**"：补丁包含未跟踪文件**内容** + `bundle_manifest.json`；声明集文件被排除即视为无效；重建校验口径写清（checkout + 解包 + 复算摘要） | §3.1、§7.2 |
| P0-3 按类型选键太窄 | `NUMERIC_CONFIG_KEYS` → `RUN_ID_CONFIG_KEYS`（闭集 + 版本号 + 每键类型 + `effect` 三分）；判据改为"是否改变算什么/怎么算"；`**kwargs` 必须按名字逐项记录；新增 `unclassified_keys` | §3.3.1、§3.3.2 |

### 13.3 三处内部不一致：已修

| 位置 | 问题 | 处理 |
|---|---|---|
| §3.1 / §5 | `source_set_captured_at_runtime` 与 `captured_at_runtime` 名称不同 | 统一为 `code.runtime_snapshot_taken`（另加 `runtime_snapshot_points`） |
| §3.1 / §5 | dirty 时补丁包必填、`recorded` 门槛里没写 | §5 的 `code` 行补"`git_dirty` 时 `patch_bundle_sha256` 有效" |
| §3.2 | 说 `python` 被 `env_sha256` 覆盖，而摘要只由包名版本拼成 | `env_sha256` 定义改为含 `python`/`impl`/`platform`/`packages`（§3.2 公式） |
| §5 / §3.5 | "`rng.plan` 四项齐" vs 实际五项 | 统一为**五项** |

---

## 14. 实现仍未授权（下一步顺序）

契约文本冻结前不写 runner。P0/P1/P2 的顺序按审查意见：

| 优先级 | 工作 | 是否动生产代码 |
|---|---|---|
| P0 | 冻结本契约（字段表 / 哈希纳入表 / `recorded` 判据 / 四处裁决） | 否 |
| P0 | 定 `RUN_ID_CONFIG_KEYS` v1 的具体键表与 `source_manifest.json` 初始内容 | 否 |
| P1 | 写 runner 落字段 + **1 次真实运行**验证字段确实来自实际执行（含补丁包重建校验） | 是，限 runner |
| P1 | 重跑对应 4 次历史条件，建 6 条新记录（`purpose=baseline_rebuild`） | 是，跑实验 |
| P2 | 233 条的实验单位与优先级 | 否 |

## 15. 本契约**不**做的事

- 不改任何生产拟合模块（`bm_core`/`bm_top`/`bm_select`/`bmdrt`/`bm_result`）。
- 不写 runner、不加字段到 registry、不升 `REGISTRY_VERSION`、不重冻结快照。
- 不新增状态枚举、不改 `duplicate_verdict` 三态、不改 `DATA_ID_KINDS`。
- 不触碰准入判定：§13 `0/3`、§14 `4/14`、§17 `0/10` 与本文件无关。
  **证据齐备 ≠ 结论成立**——本契约只管"这次运行是哪次运行"，不管"拟合结果可不可信"。

## 16. 与既有文件的关系

| 文件 | 关系 |
|---|---|
| `round_2026-11-13/HISTORICAL_EXEC_IDENTITY_GAP.md` | 缺口的**事实**审计；本文件是它 §6「向前关」的**规格** |
| `round_2026-11-13/STAGE0C_SCHEMA_CONTRACT.md` | 现行 0.7.0 的 76 条硬约束（v0 语义）；v1 是其**后继**，不推翻 |
| `round_2026-11-13/STAGE0_RECORDS_MAP.md` | 239 条记录如何由 v0 规则产出；§3.1 的"跨版本比号无意义"在 v1 继续成立 |
| `CIRCUIT_TOPOLOGY_ADMISSION.md` §8.1 | 登记合法 ≠ 准入通过；同理 **身份齐备 ≠ 结果可信** |
| `round_2026-11-13/STAGE2_SPEC.md` | §11 的五个问题在这里回答；233 条重跑属该文件 |

## 17. 修订记录

| 版本 | 日期 | 改了什么 |
|---|---|---|
| `v1.0-draft` | 2026-11-14 | 首版：五个块、`run_id` 纳入口径、降级规则、6 条/233 条处置、Stage 0 判据、D-1～D-4 待裁 |
| `v1.1-draft` | 2026-11-14 | 修 P0-1（声明集/运行时集分离 + 覆盖规则）、P0-2（补丁包含未跟踪内容）、P0-3（`RUN_ID_CONFIG_KEYS` 类型无关 + 捕获接口 + `unclassified_keys`）；修四处内部不一致；D-1～D-4 记为已定并附约束 |
