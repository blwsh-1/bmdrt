# 执行身份契约 v1（草案，未实现，待审查）

状态：**草案 `v1.0-draft`，2026-11-14。只定义，不实现。** 本轮零代码改动：
registry 仍 `0.7.0`（`b2675c8255bea716`）、`records.jsonl` 未动、快照未重冻结、
`R-` 仍 0 条、独立数仍 239 / 严格 237。**审查通过后才授权实现**，待裁项见 §13、边界见 §14。

为什么先写契约，而不是先去改 runner：`run_id` 是 `RUN_ID_BASIS` 六个字段的哈希，
**加一个字段就等于换一次号**（0.5.0 → 0.7.0 已经把 239 行全换过一遍）。字段定义错一次，
判重结果和独立数就跟着错一次，而且是**静默**错——没有任何报错。所以"先把字段、来源、
校验、降级、进不进哈希定清楚"比"先有产物"更重要。这一条是 `HISTORICAL_EXEC_IDENTITY_GAP.md`
§6 的直接结论：缺口只能向前关，而向前关的门必须先画好。

---

## 0. 一句话

**执行身份 = 「哪份代码、在哪套环境、用什么配置、跑哪份数据的哪一段、随机流怎么来的」
这五件事的可复核记录。** 记录不全时，身份降级，不许用默认值、档名、当前时间、
文件名或"看起来像"的字符串补位。

三个必须分清的东西（混一个就出一次错）：

| 概念 | 回答的问题 | 进不进 `run_id` | 判重时能不能当证据 |
|---|---|---|---|
| **条件（condition）** | 两次运行是不是"同一组执行条件" | 进 | 能（配合 `recorded` 门槛） |
| **实例（instance）** | 这一次具体跑出了什么 | 不进 | 只能证明"这份产物"（`same_artifact`） |
| **包络（envelope）** | 这次运行落在哪个版本世代 | 进（仅版本标签） | 不能 |

`init_values_sha256`、`chain_keys`、`started_at`、`elapsed_s` 全部是**实例**；
把它们塞进 `run_id` 会让"同条件跑两次"换号，去重直接失效——**这是本轮最需要提前防住的反向错误**。

---

## 1. 从这次审计反推的六类失败模式

每一条都是已经发生过、或差一步就会发生的，字段设计逐条对着它们：

| # | 失败模式 | 具体形态 | 本契约的对策 |
|---|---|---|---|
| F1 | **用上界当身份** | mtime 给区间，取区间前的最近提交当 `code_version` | `code.git_commit` 只能由运行时 `git rev-parse` 写入（§3.1） |
| F2 | **用入口脚本当代码** | 入口 `.py` 逐字节相同，`bm_top`/`nested_fit` 无副本 | `code.source_tree_sha256` 从运行时 `sys.modules` 取（§3.1） |
| F3 | **`--dirty` 当完整指纹** | 两次都 `v0.7.0-3-gabcdef-dirty`，改动内容完全不同 | `git_describe` 只作展示；身份靠 commit + tree digest；dirty 只决定"够不够"（§4） |
| F4 | **档名/默认值冒充实际值** | `budget_profile="standard"` 被当成 `target_accept` 数值 | `config.effective` 在**采样调用点**捕获，带 `origin`（§3.3） |
| F5 | **一个 seed 号含糊带过** | 初始化固定、采样随机流随机，共用一个 `seed` 字段 | 拆 `rng.plan`（条件）与 `rng.actual`（实例）（§5） |
| F6 | **实例字段误入条件哈希** | 把 `init_values_sha256` 放进 `run_id` | §6 的纳入口径表把实例字段全部排除 |

F1/F2/F3 是这次的审计对象；F4/F5/F6 是这批 239 条记录里**同样存在、但被"代码版本"这一项挡住了**
的问题——`R-` 0 条让人以为只缺一个字段。实际上就算补上代码版本，`budget_source` 仍有 233 条是
`derived_from_tier`，seed 仍有 233 条无产物级证据。**契约要一次把这几层都定对。**

---

## 2. 五个新块，叠在已有两层上

```
execution
├── candidate_id          模型定义（已有，不动）：拓扑 + 元件 + 配置 + init + 预算档名 + 似然
├── data                  数据身份（已有，扩展一点）：内容哈希 + 频率窗 + 点集
├── code                  代码身份（新）：commit + 源码树摘要 + dirty 范围
├── env                   环境身份（新）：包版本 → env_sha256 + 后端 + x64
├── config                运行配置（新）：实际生效值 + 来源 + 覆盖率
├── rng                  随机流（新）：plan（条件）+ actual（实例）
├── instance              运行实例（新）：产物定位 + 起止时刻 + 初始化值摘要
└── software              兼容层（保留）：producer、code_version（= describe，人读）、registry_version
```

**兼容原则**：`software.code_version` 语义不变（人读标签），但**不再承担身份**；
`code.*` 才是身份。这样 239 条老记录、现有读取方、`_check_software()` 全都不动，
新字段只在 v1 记录里出现。

---

## 3. 字段表

列含义：**必填**= v1 `recorded` 是否要求它存在且合法；**进 `run_id`**= 是否进哈希；
**缺失后果**= 不发 / `derived` / 不影响。

### 3.1 `code` —— 代码身份

| 字段 | 类型 | 来源（谁写） | 必填 | 校验 | 缺失后果 | 进 `run_id` |
|---|---|---|---|---|---|---|
| `git_commit` | str | 运行时 `git rev-parse HEAD` | ✓ | 40 位十六进制；**不许短哈希** | `derived` / `code_commit_missing` | ✓ |
| `source_tree_sha256` | str | 运行时对源码集求摘要（见下） | ✓ | 64 位十六进制；源码集非空且含入口 | `derived` / `source_tree_missing` | ✓ |
| `source_set` | list[{path,sha256}] | 同上，逐文件 | ✓ | path 为 POSIX 相对路径、按码位排序、无重复 | `derived` / `source_set_missing` | ✗（可由它复算上一行，不必再哈希一次） |
| `source_set_captured_at_runtime` | bool | runner | ✓ 且必须 `true` | 静态扫描得出的集合**不算** | `derived` / `source_set_static_guess` | ✗ |
| `git_dirty` | bool | `git status --porcelain -uall` | ✓ | 必须是 `true`/`false`，**不许 `null`** | `derived` / `dirty_unknown` | ✗（见 §4） |
| `dirty_paths` | list[str] | 同上 | ✓（可为空表） | POSIX 相对路径 | `derived` / `dirty_unknown` | ✗ |
| `dirty_scope` | enum | 由 `dirty_paths` 与 `source_set`、排除集三方求交 | ✓ | `clean`/`code_only`/`noncode_ignored`/`stray`/`unknown` | `stray` 或 `unknown` ⇒ `derived`（`dirty_outside_source` / `dirty_unknown`） | ✗ |
| `git_describe` | str | `git describe --tags --dirty --always` | ✗ | 非空 | 只影响可读性 | ✗（**明确不作身份**） |
| `git_diff_sha256` | str | `git diff HEAD --binary` + 未跟踪文件清单摘要 | 条件性（`git_dirty=true` 时 ✓） | 64 位十六进制 | `derived` / `diff_missing_when_dirty` | ✗（重建用，见 §7.2） |

**源码集的定义**（`source_set`，这是本契约最关键的一处，也是 F2 的唯一对策）：

1. 入口：实际执行的脚本文件（`sys.argv[0]` 解析后的真实文件），以及 `-m` 形式的模块文件；
2. **运行时已加载的项目模块**：在采样调用前对 `sys.modules` 取值，只保留 `__file__`
   落在**仓库根目录内**的模块，排除 `site-packages`、标准库、`__pycache__`；
3. **影响数值的非代码文件**：显式声明并纳入（配置、先验表、常数表）。数据文件**不进**——
   它属于 `data` 块，重复计算会让"改数据"与"改代码"混成同一种变化；
4. 摘要：`sha256` over 按路径码位排序的 `"<posix_path>\0<content_sha256>\n"` 拼接串，
   前缀一行 `v1\n`。行尾按 `\n` 归一（`*.py`/`*.toml`/`*.json`/`*.yaml`/`*.cfg`），
   其余文件按字节原样——**归一规则本身写进 `normalize` 字段**，见 §8。

排除集（不进 `source_set`，但可以出现在 `dirty_paths`）：`.git/`、`out/`、`evidence/`、
`__pycache__/`、`*.pyc`、`docs/`、`archive/`。

### 3.2 `env` —— 环境身份

| 字段 | 类型 | 来源 | 必填 | 校验 | 缺失后果 | 进 `run_id` |
|---|---|---|---|---|---|---|
| `packages` | {name: version} | `importlib.metadata.version` | ✓ 非空 | 键按字典序；版本非空 | `derived` / `env_missing` | ✗ |
| `env_sha256` | str | `sha256("name==version\n"…)` | ✓ | 64 位；可由 `packages` 复算 | `derived` / `env_missing` | ✓ |
| `backend` | str | `jax.default_backend()` | ✓ | `cpu`/`gpu`/`tpu` | `derived` / `env_missing` | ✓ |
| `x64_enabled` | bool | `jax.config.jax_enable_x64` | ✓ | bool | `derived` / `env_missing` | ✓ |
| `python` / `python_impl` | str | `platform.python_version()` | ✓ | `x.y.z` | `derived` | ✗（被 `env_sha256` 覆盖） |
| `platform` | str | `platform.platform()` | ✓ | 非空 | `derived` | ✗ |
| `device_kind` | str | `jax.devices()[0].device_kind` | ✗ | | — | ✗ |
| `cpu_threads` | int | `OMP_NUM_THREADS` 或实际线程数 | ✗ | >0 | — | ✗（D-3：线程数会改浮点求和顺序，但影响未标定） |
| `lock_digest` | str? | 锁文件内容 | ✗ | 64 位 | — | ✗（锁文件是**意图**，`env_sha256` 才是**事实**） |

`env_sha256` 是"环境事实"的一次哈希：**同一台机器上装了别的包也会改号**，这是有意的——
它记录的是环境，不是声明。

### 3.3 `config` —— 实际生效的运行配置

| 字段 | 类型 | 来源 | 必填 | 校验 | 缺失后果 | 进 `run_id` |
|---|---|---|---|---|---|---|
| `effective` | {name: `{value, origin}`} | **采样调用点**捕获 | ✓ | `origin ∈ {cli, config_file, default, tier, derived}` | `derived` / `config_incomplete` | 仅 `NUMERIC_CONFIG_KEYS`（§6） |
| `unknown_keys` | list[str] | `inspect.signature(sampler)` − `effective` 键 | ✓ 必须 `[]` | 空表 | `derived` / `config_incomplete` | ✗（校验用） |
| `capture_site` | str | `模块.函数:行` | ✓ | 非空 | `derived` | ✗ |
| `normalize` | str | 由本契约固定 | ✓ | `eol-lf-text` | — | ✗ |

`unknown_keys` 是这条契约里最像"多此一举"、实际最防漏的一项：采样器接受的每一个参数
都必须在 `effective` 里有一份**实际值**，否则跑的时候用了什么就无从知道。宁可报
`config_incomplete`，不要"反正默认值大概是这样"。

### 3.4 `data` —— 数据身份（已有，扩两点）

| 字段 | 类型 | 必填 | 校验 | 进 `run_id` |
|---|---|---|---|---|
| `ref` | str | ✓ | 非空（**标签**，不作身份） | 只在无内容哈希时退用 |
| `sha256` | str | ✓ | 64 位十六进制（已有规则） | ✓ |
| `f_min` / `f_max` | float? | ✗ | 数值 | ✓（频率窗改变 = 不同条件） |
| `n_points` | int | ✓（v1 新增） | >0，且 ≤ 源文件行数 | ✓ |
| `subset_sha256` | str? | 条件性（v1 新增） | 64 位；施加了掩码/抽样才要 | ✓（有掩码时） |
| `id_kind` | enum | ✓ | `content_sha256`/`ref_only` | ✗（哈希后看不出来，必须留痕） |

为什么要 `n_points`/`subset_sha256`：同一份 CSV 用 50 点还是 60 点、按频率窗截还是按行号截，
是**两种不同的输入**。只哈希文件内容会把它们合成同一个条件。

### 3.5 `rng` —— 随机流（条件 / 实例分开）

**A. 生成规则（条件，进 `run_id`）**

| 字段 | 类型 | 必填 | 校验 |
|---|---|---|---|
| `plan.root_seed` | int | ✓ | 整数；**必须由产物写入**（沿用 `seed_source` 的 `artifact` 判定） |
| `plan.prng` | enum | ✓ | `jax.threefry2x32` / `numpy.PCG64` / `numpy.SeedSequence` |
| `plan.derivation` | str | ✓ | 规则串，例如 `numpyro.handlers.seed(node, key); chain i ← split(key, n)[i]` |
| `plan.n_chains` | int | ✓ | 必须等于 `budget_params.num_chains` |
| `plan.init_strategy` | str | ✓ | 档名（`init_to_uniform` / `init_to_median` / `custom:<name>`） |

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
`elapsed_s`、`pid` / `host`。**全部不进 `run_id`**，只用于"这份产物是哪一次跑出来的"和取证。

---

## 4. dirty tree：`--dirty` 不是代码指纹

你的例子正是关键：两次都 `v0.7.0-3-gabcdef-dirty`，未提交改动可以完全不同。
所以 v1 的处理是**把 dirty 从"身份"降为"范围判据"**：

| `dirty_scope` | 判据 | 对 `recorded` 的影响 |
|---|---|---|
| `clean` | `dirty_paths == []` | 无影响 |
| `code_only` | 非空，且全部落在 `source_set` 覆盖的路径内 | 无影响（tree digest 已反映） |
| `noncode_ignored` | 非空，且全部落在排除集（`docs/`、`out/`、`evidence/`…） | 无影响（不参与计算，且已在 `dirty_paths` 留痕） |
| `stray` | 有一条路径**既不在** `source_set`、**也不在**排除集 | **降 `derived`**（`dirty_outside_source`） |
| `unknown` | 没有 git、或 `git status` 失败 | **降 `derived`**（`dirty_unknown`） |

看起来"`noncode_ignored` 也放行"很松，实际是它**可证无关**：排除集里的文件不会被
`sys.modules` 加载、也不是数值输入。真正危险的是 `stray`——一条改动落在没人声明的地方
（比如一个偷偷改了却没进源码集的 `params.json`），这时"改了什么"就不知道了，必须降级。
这正是 F3 要挡的那种情形；`dirty_paths` 而不是 `--dirty` 那个布尔值，才让这条判据成立。

`git_dirty` 为什么**不进哈希**：同一个 commit、同一个源码树，脏与不脏对数值没有影响；
而"脏"这件事在 `dirty_paths` / `git_diff_sha256` 里已经可供取证。把布尔值塞进 `run_id`
会让"改了 README 再跑一次"换号，判重白丢一条真重复。**换号改不了正确性，只会让样本变少。**

---

## 5. 六项证据凑齐才算 `recorded`（不新增状态）

v1 **不引入新的状态**，仍是 `recorded` / `derived` / `unconfirmed` 三档——因为状态集合在
`IDENTITY_STATES` 里，改它要动 schema、动 239 条记录的判定口径、动 `duplicate_verdict`。
v1 只做一件事：把 `recorded` 的**必填集**写清楚。

| 块 | `recorded` 要求 | 缺了会怎样 |
|---|---|---|
| `data` | `id_kind == content_sha256` + `n_points` | `unconfirmed`（不发 `run_id`） |
| `rng` | `seed_source` 前缀 `artifact` + `plan` 四项齐 + `init_values_sha256`（固定初值时） | `unconfirmed` |
| `budget_params` | 五项全 `source == recorded` | `unconfirmed` |
| `config` | `unknown_keys == []` 且 `NUMERIC_CONFIG_KEYS` 全有 `effective` | `derived` |
| `code` | `git_commit` + `source_tree_sha256` + `captured_at_runtime` + `dirty_scope ∉ {stray, unknown}` | `derived` |
| `env` | `env_sha256` + `backend` + `x64_enabled` | `derived` |

**为什么两组缺失后果不同**：`data`/`rng`/`budget` 缺了，我们连"跑的是什么"都不知道，
这时发任何 id 都是编造（现有规则，`RUN_ID_MISSING` 三个真实原因 + `not_provided` 哨兵）；`code`/`env`/`config` 缺了，
条件仍然大致可描，发 `D-` 但必须标出缺哪一项。**`D-` 号永远不能当判重依据**——
两条 `D-` 相同只说明"在已知的那部分条件上相同"，不同代码版本会给同一个号。

---

## 6. `run_id` 纳入口径

`run_id` 是**条件**指纹，判据只有一条：**这个字段变化时，两次运行是否已经不是同一组执行条件？**

| 字段 | 进 `run_id` | 理由 |
|---|---|---|
| `candidate_id` | ✓ | 模型定义层（已有） |
| `data.sha256` / `n_points` / `subset_sha256` / `f_min` / `f_max` | ✓ | 换了数据就是换了实验 |
| `rng.plan.*` | ✓ | 随机流来源不同 = 条件不同 |
| `budget_params`（五项 + source） | ✓ | 已有；数值与来源都进，档名不顶替 |
| `config.effective` 中的 `NUMERIC_CONFIG_KEYS` | ✓ | 影响数值的开关必须进，否则"改了个开关"和"没改"同号 |
| `code.git_commit` + `code.source_tree_sha256` | ✓ | 基线 + 实际代码（缺一不可：commit 覆盖不到未跟踪文件，tree 覆盖不到未加载文件） |
| `env.env_sha256` + `backend` + `x64_enabled` | ✓ | 环境会改数值；`x64` 改 dtype |
| `identity_contract_version` | ✓ | 世代标签，防"语义变了但载荷没变"撞号（D-4） |
| `code.git_describe` / `git_dirty` / `git_diff_sha256` / `dirty_paths` | ✗ | 展示或重建；由 commit+tree 或其自身派生 |
| `config.unknown_keys` / `capture_site` | ✗ | 校验与溯源，不区分条件 |
| `rng.actual.*` / `instance.*` | ✗ | **实例**——进来会让同条件两次跑换号，判重失效（F6） |
| `software.registry_version` | ✗（**D-1，需你裁**） | 它是**映射工具**版本，不是被拟合的代码；它进哈希的代价是每次改校验规则都把 239 行重排一次 |

`NUMERIC_CONFIG_KEYS` 必须是**闭集**（写在契约里，随版本增删），否则 `run_id` 会随任何
无害开关抖动，样本量被无意义地切碎。

---

## 7. 校验与不可自证的部分

### 7.1 三层校验，各管一段

| 层 | 谁做 | 查什么 |
|---|---|---|
| 写入时 | runner 自检 | 字段格式（40/64 位十六进制）、`unknown_keys == []`、`dirty_scope` 与 `dirty_paths` 自洽、`source_set` 非空且含入口、`env_sha256` 与 `packages` 可复算 |
| 读入时 | registry（`_check_execution` 现有机制） | 整块复算 `status` / `reason` / `run_id`，与观测对拍；带 `identity_contract_version ≥ 1.0` 的按 v1 必填集检查 |
| 冻结时 | `map_stage0_records.py --check` | 文本 vs 冻结字节、清单 vs 磁盘、指纹、`source_sha256` / `data_sha256_files` 逐份重算 |

### 7.2 「身份」和「可重建」是两件事

- **身份字段**回答"两次运行是不是同一套代码"（`git_commit` + `source_tree_sha256`）；
- **重建字段**回答"那套代码到底是什么"（`git_diff_sha256` + `dirty_paths` + `source_set` 逐文件哈希）。

只有身份字段时，能判"不同"，不能判"哪里不同"；只有重建字段时，能看内容，但没法一眼比。
两者都要，且**不能互相替代**：`git_diff_sha256` 盖不住未跟踪文件，`source_tree_sha256`
盖不住"未加载但会被 import 的下一次"。

### 7.3 不能自证的部分（必须承认）

- `source_tree_sha256` 与 `env_sha256` 都是**runner 自称**的值。第三方能验证的是"与它自己
  报的 `source_set` / `packages` 一致"，不能验证"这个进程真的加载了这些文件"。
- 要往"可验证"推一步，只能靠**外部手段**：以 runner 落下的 `dirty_paths` + `git_diff` 全文
  重建工作树，再复算摘要。这是实现阶段的事，契约这里只要求**重建所需的原始材料必须同时落盘**。
- 所以 §3.1 的 `git_diff_sha256` 虽然不进哈希，**在 `git_dirty=true` 时是必落项**
  （缺了降 `derived`）：没有它，`dirty` 运行永远只能"信"，不能"查"。

---

## 8. 版本化：v1 只对新记录生效

```
identity_contract_version: "1.0"     # 新记录必须带；缺省 = v0（现行 0.7.0 规则）
run_id_prefix: "R1-"/"D1-"           # D-4，若你同意，世代直接从号上看得出
```

- **不回改**：239 条记录没有这个字段 ⇒ 按 v0 读，判定不变，`R-` 仍 0 条、239/237 不变。
- **必然换号**：v1 的哈希载荷加了字段，新记录与旧记录不可能同号。这不需要道歉，
  它是"跨版本比号无意义"的又一处实例（`STAGE0_RECORDS_MAP.md` §3.1 已写明）。
- **并存策略**：stage 0 快照冻结后**不再写入**；v1 记录落新目录（如 `evidence/stage1_runs/`），
  两套各自 `MANIFEST`。这样历史基线的字节不会被未来的运行改动。

---

## 9. 硬规则：默认值不许自动获得 `recorded`

| 情形 | 禁止 | 允许 |
|---|---|---|
| 拿不到 git 信息 | 填 `unknown` / 当前日期 / 注册表版本 / 上次运行的值 | `code = null` ⇒ `derived`，原因 `code_commit_missing` |
| seed 未被捕获 | 用配置文件里的预设 seed 顶替 | 没有 seed ⇒ `unconfirmed`（现状） |
| budget 未捕获 | 由档名推导后标 `recorded` | `derived_from_tier`（现状，且不能升 `recorded`） |
| dirty 范围不明 | 宣称代码身份完整 | `dirty_scope=unknown` ⇒ `derived` |
| 数据只有文件名 | 用文件名的哈希当内容哈希 | `ref_only` ⇒ 最高 `derived` |

**一处必须写清的例外**：`origin = "default"` 的配置值是**生效值**，不是"推测值"。
区别在**捕获位置**：runner 在采样调用点把默认值物化成显式实参并记下来，这是"记录实际值"；
事后从代码里读一遍默认值是"猜当时用了什么"。前者可以 `recorded`（并保留 `origin=default`
让读者知道它没被用户显式指定），后者不行。这条如果写反，`config` 会退回 F4。

---

## 10. 6 条重跑：是重建基准，不是补历史

**目标定为「按现行 runner 规范重新执行」，不是「复现原运行」。** 理由很硬：原运行的 seed
来源、初始化值、依赖版本三项都不全，严格复现不可达；说"复现"是给不可达的目标贴标签。

| 规则 | 内容 |
|---|---|
| 旧记录 | 6 条保持 `derived`，`record_id` 不变，永远不拿新 run 的身份去填 |
| 新记录 | 新 `record_id`、新 `run_id`、`purpose: "baseline_rebuild"`、`related_history: [旧 id…]` |
| 指针方向 | **只允许新 → 旧**；旧记录不得出现指向新 run 的字段（防止身份倒灌） |
| 数值比较 | 允许并鼓励（Rwp/acf/参数），但必须标注"跨身份世代比较"，差值不可归因于单一因素 |
| 计数 | 旧 6 条（4 次执行）与新 6 条（4 次执行）各自成批；**不得**因此把 239 改成 237，也不得把 237 改成 231——旧批次计数只由旧批次的证据决定 |

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

## 13. 待你裁的四处（其余按推荐执行）

| # | 问题 | 推荐 | 影响 |
|---|---|---|---|
| D-1 | `software.registry_version` 是否继续进 `run_id` 哈希 | **移出**，只写 `provenance`；由 `identity_contract_version` 进哈希替代 | 移出会再换一次号（v1 本来就要换）；不移则每次改映射工具都重排全部记录 |
| D-2 | `config.effective` 与现有 `budget_params` 的关系 | 保留 `budget_params` 不动（已测），`effective` 是超集，`NUMERIC_CONFIG_KEYS` 只覆盖预算以外的数值开关 | 改 `budget_params` 会牵动 76 条契约测试与 239 条记录 |
| D-3 | `cpu_threads` 是否进哈希 | 先只记录、不进哈希；若后续做 seed 敏感性专项时发现它改数值，再升 v1.1 | 现在进会把"换机器"和"改代码"混成同类变化 |
| D-4 | `run_id` 前缀是否改 `R1-`/`D1-` | **建议改**：世代从号上直接可见，避免有人跨世代比号 | 改前缀要动 `_check_execution` 的前缀校验与测试断言 |

---

## 14. 本契约**不**做的事

- 不改任何生产拟合模块（`bm_core`/`bm_top`/`bm_select`/`bmdrt`/`bm_result`）。
- 不写 runner、不加字段到 registry、不升 `REGISTRY_VERSION`、不重冻结快照。
- 不新增状态枚举、不改 `duplicate_verdict` 三态、不改 `DATA_ID_KINDS`。
- 不触碰准入判定：§13 `0/3`、§14 `4/14`、§17 `0/10` 与本文件无关。
  **证据齐备 ≠ 结论成立**——本契约只管"这次运行是哪次运行"，不管"拟合结果可不可信"。

---

## 15. 与既有文件的关系

| 文件 | 关系 |
|---|---|
| `round_2026-11-13/HISTORICAL_EXEC_IDENTITY_GAP.md` | 缺口的**事实**审计；本文件是它 §6「向前关」的**规格** |
| `round_2026-11-13/STAGE0C_SCHEMA_CONTRACT.md` | 现行 0.7.0 的 76 条硬约束（v0 语义）；v1 是其**后继**，不推翻 |
| `round_2026-11-13/STAGE0_RECORDS_MAP.md` | 239 条记录如何由 v0 规则产出；§3.1 的"跨版本比号无意义"在 v1 继续成立 |
| `CIRCUIT_TOPOLOGY_ADMISSION.md` §8.1 | 登记合法 ≠ 准入通过；同理 **身份齐备 ≠ 结果可信** |
| `round_2026-11-13/STAGE2_SPEC.md` | §11 的五个问题在这里回答；233 条重跑属该文件 |
