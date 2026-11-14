# 执行身份契约 v1.2（部分冻结；`exec_identity.py` 已实现，runner 未授权）

状态：**`v1.2`，2026-11-14。声明集已落盘、反例矩阵已跑通；runner 仍未实现。**
registry `0.7.0`（`b2675c8255bea716`）、`records.jsonl`、快照、`R-` 计数（0 条）、
独立数（239 / 严格 237）**全部未动**——本轮只新增四个文件（`exec_identity.py`、
`source_manifest.json`、`round_2026-11-13/build_source_manifest.py`、
`test_exec_identity.py`），没有一行生产拟合代码被改。

本轮按审查裁决（§13.4）**停止审规格、开始动工**，顺序正是审查给的五步：
① 写 `source_manifest.json`（三份 Profile，全部 `numeric` 默认值，零降级）；
② 同批补 `workflow_source_tree_sha256`（第三层摘要，D-5 的附加条件）；
③ 边写边跑反例矩阵（T1–T24 + 新增 W/S/Y/M/X 共 **54/54**）；
④ 只冻结**跑过反例**的字段与规则（§18 冻结台账）；
⑤ runner 仍待单独授权。

**本轮没有做的事**：没有为了让测试变绿而降低任何真实模块的 `compute_role`（M2 钉住
"降级零条"）；没有把"同号"说成"结果相同"（T7/T8/W4 三格专测这条边界）。

为什么先写契约再改 runner：`run_id` 是 `RUN_ID_BASIS` 六个字段的哈希，
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
| **F11** | **导入闭包被当成"参与计算的代码"**（v1.2 新增） | Profile B 的静态闭包 17 个模块，其中 `drt`/`bm_kk`/`bm_ws`/`bm_grade`/`bm_circuit` 只是被 `bm_real` 顶层的 `import` 拖进来，`fit_top` 一次都不调用；改一行画图脚本就换号 | 声明集（完整性）与 **numeric 集（进哈希）** 分开，`role` 默认 `numeric`，降级要有理由且要过验证（§3.1.3） |
| **F12** | **"没写"与"写了但等于默认"混成一种状态**（v1.2 新增） | `adsorption=_UNSET` vs `None`（`bm_select.py:269/419`）、`zeta="1.0"` 是字符串、`include_L=None` 走 `detect_L`（`bm_real.py:2696`）、`sigma_abs=None` 在 `scalar` 模式下变成 `0.01·max|Z|`（`nested_fit.py:129`）——记原值等于什么都没记 | `effective` 记**归一化后的值** + `origin`，哨兵必须先解析成具体值（§3.3.6） |

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
| `declared_source_set` | list[{path, sha256, role, compute_role, workflow_scope, why_not_numeric?, evidence?, why_not_workflow?}] | 声明清单（§3.1.1） | ✓ | `role ∈ {entry, module, config, asset}`；`compute_role ∈ {numeric, decision, plot_only, cli_only, reference_only, budget_source}`，**省略即 `numeric`**；非 numeric 必须给 `why_not_numeric` **且** `evidence`；`workflow_scope` **省略即 `true`**，写 `false` 必须给 `why_not_workflow`；`role=entry` 不许 `workflow_scope=false`；path 为 POSIX 相对路径；按路径码位排序；无重复 | `derived` / `declared_set_missing` | ✗（三层摘要由它算） |
| `declared_source_tree_sha256` | str | 对**整份**声明集求摘要 | ✓ | 64 位十六进制，可按 `declared_source_set` 复算 | `derived` / `declared_set_missing` | ✗（完整性证据） |
| `numeric_source_tree_sha256` | str | 对 `compute_role == numeric` 的子集求摘要 | ✓ | 64 位；子集为空表 ⇒ 报错（不许"什么都不算"） | `derived` / `declared_set_missing` | ✓ |
| `workflow_source_tree_sha256` | str | 对 `workflow_scope != false` 的子集求摘要 | ✓ | 64 位；同上；（D-5 追加层，进 `workflow_id` 不进 `run_id`） | `derived` / `declared_set_missing` | ✗（进 `workflow_id`，§6.3） |
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
3. **角色标注**（三个维度，别混）：
   - `role`（文件是什么）：`entry` / `module` / `config` / `asset`；
   - `compute_role`（改不改**单次拟合**的数字）：`numeric`/`decision`/`plot_only`/`cli_only`/
     `reference_only`/`budget_source`，**省略即 `numeric`**，见 §3.1.3；
   - `workflow_scope`（改不改**整个流程**的门禁/筛选/选择/重跑）：bool，**省略即 `true`**。
     它与 `compute_role` **独立**：入口脚本既 `numeric` 又 `workflow_scope=true`
     ——不然"改了 `bm_select` 的重跑逻辑，`run_id` 变了而工作流身份没变"（D-5 的缺口）；
4. **摘要（三层，同一份声明集算三次）**：`sha256("v1\n" + <layer> + "\n" + 逐行
   "<posix_path>\0<content_sha256>\n" 按路径码位排序)`；`<layer> ∈ {declared, numeric, workflow}`
   是**域分隔**——三层永不撞值，即使成员完全相同（当前就是这种情况，见 M5）。
   行尾按 `\n` 归一（`*.py`/`*.toml`/`*.json`/`*.yaml`/`*.cfg`/`*.ini`），其余按字节；
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

#### 3.1.3 三份 Profile 的声明集（数字从源码算出，不是列目录）

`docs/round_2026-11-13/audit_source_profiles.py`（只读，AST，不执行被审代码）从三个入口
分别求导入闭包。**它立刻暴露了 F11**：

| Profile | 入口 | 闭包模块数 | 行数 | 函数体内导入的项目模块（拟合前不可见） |
|---|---|---|---|---|
| A | `bm_real`（Voigt 单族） | **13** | 9392 | `bm_real → bm_circuit, bm_ref_rwp`；`bm_ad → bm_kk`；`bm_corr → bm_core`；`bm_diff → bm_core`；`bm_kk → fit_store`；`bm_ref_rwp → bm_core` |
| B | `nested_fit`（nested） | **17** | 13376 | 同 A，另加 `nested_fit → bm_core`、`bm_top_gen → bm_top` |
| C | `bm_select`（跨族） | **16** | 12860 | 同 B（`bm_select` 自身不延迟导入项目模块） |

三条结论，都直接影响键表和 Profile：

1. **`nested_fit` 在函数体内 `import bm_core`**（`nested_fit.py:211`，只在
   `with_voigt=True` 的对照路径上）。这就是 F7 的活样本：**只在 `before_fit` 取一次
   `sys.modules` 会漏掉它**，而 AST 闭包看得见。
2. **Profile B 的闭包被 `bm_real` 拖宽**：`nested_fit` 只用到 `bm_real.load_spectrum`，
   但 `bm_real` 顶层 `import bm_core/bm_experiments/bm_grade/drt/bm_kk`，于是
   `drt`、`bm_kk`、`bm_ws`、`bm_grade`、`fit_store`、`bm_circuit`、`bm_ref_rwp`
   全进了 B 的闭包——**`fit_top` 一次都不调用它们**。若把整份闭包哈希进 `run_id`，
   改一行 `bm_circuit` 的画图代码就会让所有 nested 运行换号。这正是 F11。
3. **`nested_fit → bm_select` 只为 `BUDGETS` 常量**（`nested_fit.py:73`）。它的数值效应
   已经由 `budget_params`（`standard` 档 2000/1000/2/0.98/10）捕获，不需要再进 numeric 集。

**所以 `role` 之外，一个文件要回答两个独立问题**，各有自己的摘要与落点（D-5 裁决：
"三层"，不是"两份"）：

| 摘要 | 包含内容 | 进哪里 | 判据 |
|---|---|---|---|
| `declared_source_tree_sha256` | 完整声明集 | **只记录**，做完整性核对 | 本次执行覆盖了哪些文件 |
| `numeric_source_tree_sha256` | `compute_role == numeric` 的子集 | **单次 `run_id`** | 改完，这一次拟合的数会变 |
| `workflow_source_tree_sha256` | `workflow_scope != false` 的子集 | **`workflow_id`**（§6.3） | 改完，门禁/候选筛选/选择/重跑会变 |

**三层不是一个划分，允许重叠——入口脚本必须重叠。** 这是审查 D-5 指出的缺口：
`bm_grade` 的判定逻辑、`bm_select` 的选择与重跑逻辑、`bm_real` 里的 KK 门禁，都不会改
单次拟合的 `run_id`，但会改"哪个候选被保留、最后推荐谁、某次拟合被拦还是放行、要不要多
跑一轮"。若它们被排除在 numeric 之外、而 `workflow_id` 又只由搜索空间和子运行列表生成，
就会出现**工作流逻辑变了、工作流身份没变**。W1/W2 两格专测这条。

`budget_source`（`bm_select` 对 Profile B 只贡献 `BUDGETS`）**不进第三层**：它的数值效应
已经由 `budget_params` 捕获，进 `workflow_id` 只会让"改一个预算常量"变成换工作流号。
`plot_only` / `reference_only` 也不进前三层里的后两层——它们只影响附加产物，
**明确不影响拟合条件**，但仍在 `declared` 层里被记录（改了要看得见）。

| `compute_role` | 含义 | 进 `numeric` 摘要 | 允许的模块（举例，最终以 `source_manifest.json` 为准） |
|---|---|---|---|
| `numeric` | 参与数据解析 / 正演 / 似然 / 先验 / 初值 / 采样调用 | ✓ | `bm_real` `bm_core` `bm_top` `bm_diff` `bm_ad` `bm_corr` `drt`（初值来源） |
| `budget_source` | 只被读取常量（预算档） | ✗ | `bm_select`（Profile B 里只为 `BUDGETS`） |
| `decision` | 事后诊断 / 选择 / 评分 | ✗ | `bm_experiments` `bm_grade` `bm_select`（Profile C 的判定逻辑） |
| `reference_only` | 附加对照产物 | ✗ | `bm_ref_rwp` |
| `plot_only` | 只画图 | ✗ | `bm_circuit` |
| `cli_only` | 只解析命令行 / 打印 | ✗ | 入口脚本本身（`role=entry` 时可同时 `cli_only`？**不行**，见下） |

**入口脚本必须是 `numeric` 且 `workflow_scope=true`**：它决定调用哪个函数、传什么值，
也含门禁与重跑逻辑。把入口降成 `cli_only` 等于让"改了一行默认参数"不换号——正好是 F4；
把入口移出 workflow 层，等于让"改了重跑逻辑"不换工作流号。`check_declared_set()`
对这两条**直接报错**（M7）。

#### 3.1.4 降级要什么证据（本轮实测结论：零降级）

审查给了定性判断："改无关常量后数值逐位相同"**只能证明某个改动没影响，不能证明整个模块
都不影响数值**，它应当作**回归验证**，不是 `compute_role` 分类正确的充分证据。据此，
`v1.2` 对降级采用**双条件**（缺一条就退回 `numeric`）：

- `plot_only` / `reference_only`：**关掉该功能的那次运行里，模块不出现在
  `runtime_loaded_modules`**（这是唯一能证明"这次没用它"的证据）；
- `decision` / `budget_source`：**改动该模块里一个不相关的常量或注释**后重跑，
  产物数值字段（`params`/`metrics`/`elpd`）逐位相同 **且** 声明集与 `runtime` 集合未变；
- 两条都要在一次**真实运行**里做——静态审查做不到，所以本轮一条也没降。

当前 `source_manifest.json` 里 **`compute_role` 全部 `numeric`、`workflow_scope` 全部
`true`**（M2 断言"降级零条"）。候选降级写在 `pending_demotions` 里、`applied: false`，
每条都写了要什么实测（M4）：

| 候选 | 拟降为 | 需要的实测 |
|---|---|---|
| `bm_circuit.py`（A/B） | `plot_only` | `plot=False` 的运行不加载它，且产物数值逐位相同 |
| `bm_ref_rwp.py`（A） | `reference_only` | `ref_rwp=False` 的运行不加载它，指标逐位相同 |
| `bm_grade.py`/`bm_experiments.py` | `decision` | 改无关常量后 `params`/`metrics`/`elpd` 逐位相同 |
| `bm_select.py`（B） | `budget_source` | 改选择器一行后 nested 数值逐位相同、`budget_params` 不变 |
| `bm_kk.py`（A/B） | `decision` | `kk_mode=off/warn` 两次数值相同且不加载 |
| `config.json` | 按段 | **文件粒度拆不开**（fit/kk/diag 段影响数值、paths 段不影响）⇒ 要么不降，要么先把段拆成独立文件 |

**默认值的保守方向只有一个**：不写 `compute_role` 就是 `numeric`，不写 `workflow_scope`
就是 `true`。宁可多换号，不能让一次真实的代码改动悄悄沿用旧号——后者会把两次不同代码的
执行判成重复（F6 那类反向错误）。**多换号是可见的噪音，少换号是不可见的错判。**

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

### 3.3 `config` —— 实际生效的运行配置（v1.1 重写，v1.2 补键表与委托）

| 字段 | 类型 | 来源 | 必填 | 校验 | 缺失后果 | 进 `run_id` |
|---|---|---|---|---|---|---|
| `effective` | {name: `{value, origin, effect}`} | **采样调用点**捕获 | ✓ | `origin ∈ {cli, config_file, default, tier, derived}`；`effect ∈ {model_definition, execution_condition, display_only}` | `derived` / `config_incomplete` | 仅 `effect ∈ {model_definition, execution_condition}` 的键 |
| `capture_interface` | {callable, module, line, kwargs_rule} | runner | ✓ | `callable` 必须是被实际调用的那一层 | `derived` / `config_incomplete` | ✗ |
| `unknown_keys` | list[str] | 见下 | ✓ 必须 `[]` | 空表 | `derived` / `config_incomplete` | ✗ |
| `unclassified_keys` | list[str] | 有 `effective` 但不在闭集、也没给 `effect` 的键 | ✓ 必须 `[]` | 空表 | `derived` / `config_unclassified` | ✗ |
| `normalize` | str | 由本契约固定 | ✓ | `eol-lf-text` | — | ✗ |

#### 3.3.1 判据：改变"算什么/怎么算"，不是"是不是数字"

这是 P0-3 的修法。`v1.0` 的 `NUMERIC_CONFIG_KEYS` 会漏掉布尔与枚举开关，而后者的影响往往
更大。`v1.2` 按审查给的分类把 `effect` 从三值扩到**六值闭集**（D-6 已定）：

| `effect` | 进 `run_id` | 含义 | 判据一句话 |
|---|---|---|---|
| `model_definition` | ✓ | 改"算什么"：拓扑、元件、层数、族能力 | 改完，模型空间变了 |
| `execution_condition` | ✓ | 改"怎么算"：噪声口径、预算、种子、先验尺度 | 改完，同一次运行不再可复现 |
| `admission_condition` | ✗ | 门禁：决定这次运行**有没有发生**，不改已发生运行的后验 | 门禁只拦，不改数 |
| `decision_condition` | ✗ | 事后诊断/选择阈值：决定**选哪个** run | 改完，后验不变、结论可能变 |
| `workflow_scope` | ✗ | 搜索空间、附加子运行、对照产物 | 它描述"跑几次"，不描述"哪一次" |
| `display_only` | ✗ | 打印/画图/进度 | 改完，产物数值逐位不变 |

为什么 `admission_condition` 和 `decision_condition` 不进哈希：`run_id` 回答的是
**"这一次运行的条件"**，而门禁与选择回答的是"哪几次运行被允许发生、最后选了谁"。
把它们塞进哈希，会让"把 KK 门禁从 gate 换成 warn 再跑一次"变成换号——两次的后验其实
一模一样。它们的值记在 `workflow`（§3.3.7），**门禁的实际结果**（`kk.status`/`kk_performed`）
记在产物里。

**`RUN_ID_CONFIG_KEYS` 全局注册表（v1，`run_id_config_keys_version: 1`）**

闭集的目的**不是**限制类型，而是**防止 `run_id` 随无害开关抖动**（每个多余键都把样本切碎）；
类型只是每行的一个属性。键名一律用小写原样（不改写、不加前缀），`effect` 只允许上表六值。

| 键 | 类型 | `effect` | 出现在 | 出处 |
|---|---|---|---|---|
| `branch` | enum(`parallel`/`series`) | `model_definition` | B、C | `bm_top.py:913`、`bm_top.py:162` |
| `layers` | int | `model_definition` | B、C | 同上 |
| `K` | int | `model_definition` | A、C | `bm_real.py:944`、`bm_select.py:331` |
| `include_L` | bool（`None` 先解析） | `model_definition` | A、C | `bm_real.py:2696`、`bm_select.py:419` |
| `adsorption` | enum(`none`/`RL`)（`_UNSET` 先解析） | `model_definition` | A、C | `bm_select.py:269/419` |
| `diffusion` | enum(`none`/`Ws`/`Wo`…) | `model_definition` | A | `bm_real.py:2657` |
| `zeta` | enum(`free`) 或 float | `model_definition` | A | `bm_real.py:2659` → `zeta_fixed` |
| `sigma` | 复合（§3.3.5） | `execution_condition` | A、B | `bm_real.py:964`、`nested_fit.py:115` |
| `sigma_floor` | float | `execution_condition` | A、B、C | `bm_real.py:944`、`bm_top.py:645` |
| `sigma_mode` | enum(`free`/`scalar`/`perpoint`) | `execution_condition` | B | `nested_fit.py:121-133` |
| `sigma_pct` | float | `execution_condition` | B | 同上 |
| `sigma_abs` | float? | `execution_condition` | B | 同上（`None` ⇒ `0.01·max\|Z\|`） |
| `zmax_override` | float? | `execution_condition` | A | `bm_real.py:944`、`bm_top.py:913` |
| `krange` / `wide` / `kmax` / `K_force` | 见各 Profile | `workflow_scope` | A、C | `bm_real.py:2762-2778` |
| `branches` / `pair_mode` / `n_seeds` / `retry` | 见各 Profile | `workflow_scope` | C | `bm_select.py:1057/1169` |
| `auto_mechanism` / `auto_mechanism_full` / `auto_adsorption` | bool | `workflow_scope` | A | `bm_real.py:2657` |
| `with_voigt` / `voigt_K` | bool / int | `workflow_scope`（子运行里 `voigt_K` 是 `model_definition`） | B | `nested_fit.py:319/393` |
| `ref_rwp` | bool | `workflow_scope` | A | `bm_real.py:2657` |
| `kk_mode` / `kk_opts` | enum / dict | `admission_condition` | A | `bm_real.py:2715` |
| `diag` / `drt_opts` | dict | `decision_condition` | A | `bm_real.py:2657` |
| `gate_b` / `strict` | bool | `decision_condition` | C | `bm_select.py:1072/1074` |
| `progress` / `verbose` / `plot` / `dry` / `trace` | bool | `display_only` | A、B、C | 各处 |

**不进这张表、也不重复记录的**（权威在别处，§3.3.3）：`seed`（`rng.plan`）、
五项预算 `n_warmup`/`n_samples`/`num_chains`/`target_accept`/`max_tree_depth`（`budget_params`）、
`init_values`/`init_strategy`（`rng.plan`/`rng.actual`）、`input_path`/`path`（`data`）、
`outdir`（`instance`）。§13 D-2 规定这层权威关系，§3.3.3 给判定公式。

#### 3.3.2 `unknown_keys` 怎么求（不能只靠 `inspect.signature`）

必须先钉住"捕获的是哪个接口"，否则会把内部参数当采样配置、或漏掉经 `**kwargs` 传进来的东西：

1. `capture_interface.callable` 记**实际被调用的那一层**（例如 `bm_select.fit_voigt`，
   而不是它内部再调用的 `bm_real.fit_one`）；`module`/`line` 供核对；
2. 该 callable 签名里每个**显式命名参数**必须**有归属**：要么在 `effective`，
   要么被委托给 `data`/`rng`/`budget_params`/`instance`/`workflow`（§3.3.3）——
   `v1.1` 写的是"必须出现在 `effective`"，与 §3.3.1 的委托条款直接冲突
   （`nested_fit.run` 的 `seed`/`budget`/`input_path` 都不该重复进 `effective`）；
3. 若签名有 `**kwargs` 捕获器：凡实际经它传入的键，必须**按名字**逐项出现在 `effective`，
   不许整体塞成一个字典——否则"传了什么"就查不到；
4. 归属算完仍有剩余 ⇒ `unknown_keys`；落在 `effective` 但不在 `RUN_ID_CONFIG_KEYS`
   且没标 `effect` ⇒ `unclassified_keys`；
5. 两者非空都降级。`unclassified` 单列一条原因，因为它的含义是"有个开关我们不知道它改不改条件"，
   不能和"漏记录"混成一种。

**默认值的界线**（与 §9 一致）：`origin=default` 的值是**生效值**，可以 `recorded`；
区别在捕获位置——runner 在调用点把默认值物化成显式实参并记下来，是"记录实际值"；
事后从代码里读一遍默认值是"猜当时用了什么"，不行。

#### 3.3.3 字段委托表（"每个参数只有一个权威字段"）

`v1.1` 只说了"`seed` 在 `rng.plan`、预算在 `budget_params`"，没说清**怎么算剩余**，
于是为了让 `unknown_keys=[]`，实现者完全可以把 `seed` 再塞一遍进 `effective`。
现在把它写成表 + 公式：

| 参数类别 | 权威字段 | 允许在 `effective` 重复 | 例子 |
|---|---|---|---|
| 数据输入与解析 | `data` | **✗** | `path` / `input_path` / `fscale` / `zscale` / 掩码 |
| 随机流与初值策略 | `rng.plan` | **✗** | `seed` / `init_strategy` |
| 采样预算 | `budget_params` | **✗** | `n_warmup` / `n_samples` / `num_chains` / `target_accept` / `max_tree_depth` |
| 运行实例 | `instance` | **✗** | `outdir` / 实际初值 / `elapsed_s` |
| 搜索空间与附加运行 | `workflow` | **✗** | `krange` / `n_seeds` / `with_voigt` / `ref_rwp` |
| 其它执行配置 | `config.effective` | **必须** | `branch` / `layers` / `include_L` / `sigma.*` |

```
actual_kwargs = 被调用接口本次实际收到的全部实参（含调用点物化的默认值）
delegated     = { seed, init_* } ∪ { n_warmup, n_samples, num_chains,
                                    target_accept, max_tree_depth }
              ∪ { path, input_path, outdir, *workflow 键 }
unknown_keys  = actual_kwargs − effective − delegated        （非空 ⇒ 降级）
unclassified  = { k ∈ effective : effect(k) 未登记 }          （非空 ⇒ 降级）
```

**唯一的"两处出现"是 `num_chains`**：它同时是 `rng.plan.n_chains` 与
`budget_params.num_chains`，且必须相等。这不算重复定义，算一致性校验
（不一致 ⇒ `rng_budget_mismatch` ⇒ 降级），因为两处回答的是不同问题：
"几条链"（随机流结构）与"多少采样预算"。

同理，候选级参数（`K`/`layers`/`branch`/族能力）在 `candidate_id` 与
`config.effective` 里都会出现——前者是**模型定义层**（已有，不动），后者是
**执行记录层**；两者必须一致（不一致 ⇒ `candidate_config_mismatch` ⇒ 降级），
否则"候选说 K=2、实际按 K=3 跑"这种错查不出来。

#### 3.3.4 三个入口的键表（从实际签名反推，不是举例）

审查指出的问题成立：`v1.1` 的七行示例只覆盖 `bm_select` 那一层，直接调
`bm_real.run()` / `nested_fit.run()` 时需要的键根本没进表。下面三张表按**实际签名**
逐参数列，出处写到行号。

**Profile A —— `bm_real.run()`（`bm_real.py:2657`，Voigt 单族）**

| 参数 | 类型 | `effect` | 归属 | 归一化 / 备注 |
|---|---|---|---|---|
| `path` | str | — | `data.ref` + `sha256` | 解析固定 `load_spectrum(zscale=1)` |
| `K_force` | int? | `workflow_scope` | `workflow.search.fixed_K` | 固定 K ⇒ 只跑一个候选；模型身份在 `candidate_id` |
| `krange` | "a,b"? | `workflow_scope` | `workflow.search.krange` | 默认 `[K_drt−1, K_drt+1] ∩ [1, kmax]`，**`K_drt` 由 DRT 预判数据决定** ⇒ "实际跑了几个 K"是数据派生的 |
| `wide` | bool | `workflow_scope` | `workflow.search.wide` | |
| `kmax` | int | `workflow_scope` | `workflow.search.kmax` | 默认 `K_MAX=5` |
| `include_L` | bool? | `model_definition` | `effective.include_L` | `None` ⇒ `detect_L(f,Z)`（`:2696`）⇒ 记**解析后的 bool**，`origin=derived` |
| `zeta` | str | `model_definition` | `effective.zeta_fixed` | `"free"`⇒null；`"1.0"`⇒1.0（F12） |
| `dry` | bool | `display_only` | — | 不跑拟合 |
| `quick` | bool | `execution_condition` | `budget_params` | 800/400/2（`:2770`）⇒ **档名不进哈希，数值进** |
| `ta` | float? | `execution_condition` | `budget_params.target_accept` | `None` ⇒ 沿用档位 0.9 |
| `diffusion` | str? | `model_definition` | `effective.diffusion` | `bm_core` 元件 |
| `kk_mode` | enum | `admission_condition` | `workflow.kk.mode` | 只拦不改数；**实测结果** `kk.status`/`kk_performed` 记产物 |
| `kk_opts` | dict | `admission_condition` | `workflow.kk.opts`（逐键） | `auto_L` 只影响 KK 内部去电感估计（`bm_kk.py:684`），**不回写** `include_L` |
| `diag` | dict | `decision_condition` | `workflow.diag` | 事后阈值（相关矩阵 / 扩散可分辨性） |
| `drt_opts` | dict | `decision_condition` | `workflow.drt` | 影响 DRT 去电感 → 初值 ⇒ 实际初值由 `rng.actual.init_values_sha256` 兜住 |
| `sigma_floor` | float | `execution_condition` | `effective.sigma.floor` | 与 `sigma` 互斥（`bm_top.py:645`） |
| `auto_mechanism` / `auto_mechanism_full` | bool | `workflow_scope` | `workflow.extras` | 触发 Ws / ad 补跑 |
| `adsorption` | "RL"? | `model_definition` | `effective.adsorption` | 与 `diffusion` 互斥（一期屏蔽 both） |
| `auto_adsorption` | bool | `workflow_scope` | `workflow.extras` | 触发 `(K, RL)` 补跑 |
| `seed` | int? | `execution_condition` | `rng.plan.root_seed` | `None` ⇒ `KW["seed"]=0`（`_resolve_seed`，`:924`），`origin=default` |
| `ref_rwp` | bool | `workflow_scope` | `workflow.extras` | 只加参考指标，不改参数推断 |

**Profile B —— `nested_fit.run()`（`nested_fit.py:319`，nested 单族）**

| 参数 | 类型 | `effect` | 归属 | 归一化 / 备注 |
|---|---|---|---|---|
| `input_path` | str | — | `data.ref` + `sha256` | `bm_real.load_spectrum` |
| `outdir` | str | — | `instance.outdir` | 会影响产物落在哪，与条件无关 |
| `branch` | enum(`parallel`/`series`) | `model_definition` | `effective.branch` | |
| `layers` | int | `model_definition` | `effective.layers` | |
| `budget` | enum | `execution_condition` | `budget_params` | `budget_kw`（`:101`）：`standard` = 2000/1000/2/**0.98**/10（`VALIDATION_STANDARD`），其它档取 `bm_select.BUDGETS` 的五项 ⇒ **档名不进哈希** |
| `sigma_mode` | enum | `execution_condition` | `effective.sigma.mode` | |
| `sigma_pct` | float | `execution_condition` | `effective.sigma.pct` | `perpoint` 时 σ= pct%·\|Z\|（`:122`） |
| `sigma_abs` | float? | `execution_condition` | `effective.sigma.abs` | `scalar` 且 `None` ⇒ **`0.01·max\|Z\|`**（`:129`）⇒ 必须记实际 σ 摘要 |
| `seed` | int | `execution_condition` | `rng.plan.root_seed` | |
| `target_accept` | float? | `execution_condition` | `budget_params.target_accept` | `None` ⇒ 用档位值（standard 0.98 / quick 0.9） |
| `plot` | bool | `display_only` | `effective.plot` | 值照记，不进哈希 |
| `with_voigt` | bool | `workflow_scope` | `workflow.parent.children` | 只加对照子运行，**nested 后验不变** ⇒ 不进 nested 的 `run_id` |
| `voigt_K` | int | 父：`workflow_scope` / 子：`model_definition` | `child.voigt_K` | 父运行的 `config.voigt_K` 在未开启时是 `None`（`:393`） |
| `progress` / `verbose` | bool | `display_only` | `effective` | |

**Profile C —— `bm_select.select_across_families()` / `_select_one()`（`:1169` / `:1057`）**

| 参数 | 类型 | `effect` | 归属 | 归一化 / 备注 |
|---|---|---|---|---|
| `krange` | tuple | `workflow_scope` | `workflow.search.krange` | 候选 K 网格 |
| `layers` | tuple | `workflow_scope`（搜索）/ `model_definition`（每候选） | `workflow.search.layers` | 每个候选的 `layers` 进它自己的 `run_id` |
| `branches` | tuple | `workflow_scope` | `workflow.search.branches` | 默认 `("parallel","series")`（`:235`） |
| `seed` | int | `execution_condition` | `rng.plan.root_seed` | `select_across_families` 用 `seed+i` ⇒ 每个子运行一个根种子 |
| `sigma_floor` | float | `execution_condition` | `effective.sigma.floor` | 默认 `SIGMA_FLOOR=0.0` |
| `kw` | dict | `execution_condition` | `budget_params` | 默认 `QUICK`= 800/400/2/0.9/10（`:127`） |
| `trace` | bool | `display_only` | `effective` | |
| `pair_mode` | enum(`pooled`/`per-branch`) | `workflow_scope` | `workflow.pair_mode` | 比较口径：改它不改任何单次运行 |
| `retry` | int | `workflow_scope` | `workflow.retry` | 触发整批换档重跑 |
| `strict` | bool | `decision_condition` | `workflow.guards.strict` | A 类 guard 是否进可用性门禁 |
| `retry_kw` | dict | `execution_condition`（**子运行**） | `budget_params`（重跑档） | 默认 `standard` |
| `include_L` | bool? | `model_definition` | `effective.include_L` | `None` ⇒ `VOIGT_CAPS["include_L"]`（`:419`） |
| `adsorption` | `_UNSET` / `None` / `"RL"` | `model_definition` | `effective.adsorption` | `_UNSET` ⇒ `VOIGT_CAPS`；`None` = **显式关**——两者语义不同（F12） |
| `gate_b` | bool | `decision_condition` | `workflow.guards.gate_b` | 把 `comp_unresolved` 接进族内择优 |
| `n_seeds` | int | `workflow_scope` | `workflow.n_seeds` | 每个 seed 是一次独立运行，各自一个 `run_id` |
| `K`（`fit_voigt`）/ `layers`（`fit_nested`） | int | `model_definition` | 候选 `run_id` | 候选级参数，见 §3.3.3 的一致性规则 |

#### 3.3.5 `sigma` 是复合键，不是一个枚举

审查点出的问题：两次都写 `sigma_from_data=True`，实际噪声数组可以不同。所以 `sigma`
既不是 `sigma_mode` 一个枚举，也不是 `sigma_pct` 一个数：

| 子字段 | 类型 | 必填 | 进 `run_id` | 说明 |
|---|---|---|---|---|
| `kind` | enum(`none`/`scalar_pair`/`perpoint`) | ✓ | ✗（由 `array_sha256` 承担） | `fit_top` 的三种形态（`_check_sigma`，`bm_top.py:645`） |
| `mode` | enum(`free`/`scalar`/`perpoint`) | ✓ | ✗ | `make_sigma` 的**来源**口径（`nested_fit.py:121`） |
| `pct` / `abs` | float? | 对应 mode | ✗ | 参数值；`abs=None` 会被物化成 `0.01·max\|Z\|` |
| `array_sha256` | str? | `kind≠none` 时 ✓ | **✓** | 对**实际**传入的数组求摘要 |
| `floor` | float | ✓ | **✓** | `sigma_floor`，与 `sigma` 互斥 |
| `from_data` | bool | ✓ | ✗ | 决定 `elpd` 有没有留一解释（`fit_top` 的 `elpd_valid_*`） |
| `elpd_valid` | bool | ✓ | ✗ | 派生自 `from_data`，进产物供读者判断 |

**进哈希的是投影，不是整个块**：`sigma_run_id_value()` 只取
`{array_sha256, floor}` 两个子字段，其余六个**留记录、不进哈希**。
这一条是本轮实现时补的——`v1.2-draft` 写"`kind`/`mode` 不进哈希"，但代码只要直接
`canonical_json(整个块)`，`pct`/`abs` 就会跟着进去，于是 T19 立刻失败。**"某字段不进哈希"
必须落成一次显式投影**，不能靠"我没把它写进去"。

`array_sha256` 的口径（D-8 裁决后的完整规则）：

```
sha256( b"v1\n" + f"{a.shape[0]}x{a.shape[1]}\n" + np.ascontiguousarray(a, "<f8").tobytes(order="C") )
```

| 规则 | 处理 | 为什么 |
|---|---|---|
| dtype | 一律转 `<f8` 小端 | 只看数；`>f8` 与 `<f8` 是同一个数组（S5） |
| 顺序 | `np.ascontiguousarray` + `order="C"` | 内存布局不是数值，F 数组不该换号（S6） |
| 形状 | **进头部载荷** | `(2,3)` 与 `(3,2)` 不该靠字节长度区分（S2） |
| NaN/Inf | **报错**（`ValueError`） | 非法噪声不该有身份；静默哈希会把 bug 变成号 |
| 空数组 | **报错** | 空数组 ≠ `kind=none`：前者是"忘了填"，后者是"噪声被采样"（S4） |
| `None` | `kind=none` 时合法且 `array_sha256=null`；`kind≠none` 时**报错** | 同上 |
| 复算 | S1 用 `b"v1\n2x2\n" + bytes` 独立重算并对拍 | 摘要必须能被第三方从数组复现 |
| **dtype 差异** | `float32(0.1)` 与 `float64(0.1)` **不同号** | 它们真的是两个数（0.10000000149… vs 0.1）；"dtype 无关"是错的 |

**`kind`/`mode` 不进哈希**是故意的：两次运行的来源口径不同、但实际 σ 数组逐位相同
⇒ 似然完全一样 ⇒ 是同一组条件。而 `from_data` 只改 `elpd` 的**解释**，不改后验
⇒ 归 `workflow`（`decision` 层），不进 `run_id`。

**T19 的原措辞已订正**：原写"`mode=perpoint,pct=2` → `scalar,abs=` 使实际 σ 数组逐位相同"，
这是**做不到的**——`_check_sigma` 里 `scalar` 是 `(2,)`、`perpoint` 是 `(2,n)`，形状不同
就不可能是"同一个数组"。改为两格：
**T19a 同一数组、不同 `mode` ⇒ 同号**（测"来源不进哈希"）；
**T19b 同一 `mode`、数组差 0.01% ⇒ 换号**（测"实际数组进哈希"，这才是 D-8 的落点）。

#### 3.3.6 归一化规则（F12 的对策）

1. **布尔**写成 `true`/`false`；**整数原样**，且 `bool` 不许落进整数位（0.7.0 踩过：
   `True` 被当成 `1`）；
2. **浮点**用 `repr`（17 位有效数字），**不四舍五入**——四舍五入会把"改了第 12 位"抹平；
3. **枚举**用小写原样字符串，且必须落在闭集里（写错就报错，不静默接受）；
4. **哨兵必须先解析**，不许把哨兵本身记成值：

   | 原值 | 解析后 | `origin` |
   |---|---|---|
   | `adsorption=_UNSET` | `VOIGT_CAPS["adsorption"]`（当前 `None`） | `derived` |
   | `adsorption=None` | `None`（显式关） | `cli`/`config_file` |
   | `include_L=None` | `detect_L(f, Z)` 的布尔结果 | `derived` |
   | `target_accept=None` | 档位值（standard 0.98 / quick 0.9） | `tier` |
   | `sigma_abs=None` + `mode="scalar"` | `0.01·max\|Z\|` | `derived` |
   | `seed=None` | `KW["seed"]`=0 | `default` |

5. **复合**（`kk_opts`/`diag`/`drt_opts`/`workflow.*`）按键名码位排序写 canonical JSON，
   键名闭集校验；
6. **`null` 与空容器不同**：`diag={}` ≠ `diag=null`；
7. **禁止"就近相等"**：`0.1+0.2` 与 `0.3` 不归一，比较一律逐位。

#### 3.3.7 `workflow` 块（不进 `run_id`，但必须有）

`with_voigt=True` 不是"nested 模型变了"，而是**一次工作流多跑了一个对照**。审查给的
处置是对的：父工作流 + 独立子运行，避免"多加一个对照"改掉 nested 自己的身份。

| 字段 | 类型 | 说明 |
|---|---|---|
| `workflow_id` | str | `W1-` + 32 位；由**工作流源码摘要 + 搜索空间 + 门禁设定 + 子运行列表**算出（§6.3）；**不反馈**进任何子 `run_id` |
| `role` | enum(`parent`/`child`) | 子运行里 `parent_workflow_id` 指回父 |
| `search` | {krange, kmax, wide, fixed_K, layers, branches, n_seeds} | 搜索空间 |
| `extras` | {auto_mechanism, auto_mechanism_full, auto_adsorption, with_voigt, ref_rwp} | 触发了哪些附加运行 |
| `gate` | {kk_mode, kk_opts, strict, **kk_status, kk_performed**} | 门禁设定**与实测结果**——设定进 `workflow_id`，结果**不进** |
| `guards` | {strict, gate_b, diag, drt_opts} | 事后判定阈值 |
| `pair_mode` / `retry` / `retry_kw` | 见 Profile C | 比较与重跑口径 |
| `children` | list[run_id] | 子运行清单（每个都是独立条件身份） |

**`gate` 里同时有设定和结果，必须分开处理**：`kk_mode`/`kk_opts`/`strict` 是条件，
`kk_status`/`kk_performed` 是观测。把观测也塞进 `workflow_id`，工作流身份就变成了
**结果身份**——同一套设定、因为数据噪声不同得到 `pass`/`fail`，会被判成两个工作流。
门禁真的起作用时，它改变的是 `children`（那个子运行没跑），结果通过"跑没跑"
进来，而不是通过一个状态字符串（W4 专测这条）。

### 3.4 `data` —— 数据身份（已有，扩两点）

| 字段 | 类型 | 必填 | 校验 | 进 `run_id` |
|---|---|---|---|---|
| `ref` | str | ✓ | 非空（**标签**，不作身份） | 只在无内容哈希时退用 |
| `sha256` | str | ✓ | 64 位十六进制 | ✓ |
| `f_min` / `f_max` | float? | ✗ | 数值 | ✓（频率窗改变 = 不同条件） |
| `n_points` | int | ✓（v1 新增） | >0，且 ≤ 源文件行数 | ✓ |
| `subset_sha256` | str? | 条件性（v1 新增） | 64 位；施加了掩码/抽样才要 | ✓（有掩码时） |
| `transform` | {fscale, zscale, mask} | ✓（v1.2 新增，可为恒等） | 数值/枚举；`zscale≠1` 会缩放 Z，`fscale≠1` 会缩放频率 | ✓（D-7） |
| `id_kind` | enum | ✓ | `content_sha256`/`ref_only` | ✗（哈希后看不出来，必须留痕） |

为什么要 `n_points`/`subset_sha256`：同一份 CSV 用 50 点还是 60 点、按频率窗截还是按行号截，
是**两种不同的输入**。只哈希文件内容会把它们合成同一个条件。

为什么要 `transform`：`bm_real.load_spectrum(path, fscale=None, zscale=1.0)`（`bm_real.py:132`）
允许在**读取时**缩放频率与阻抗。`zscale=2` 与 `zscale=1` 读同一个文件，进似然的 Z 差一倍，
而文件哈希完全一样。当前三个 Profile 的入口都不传这两个参数（都是默认值），但字段必须在，
否则将来一旦有人用上，它就会静默地和原值共用同一个 `run_id`（D-7）。

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
| `config` | `unknown_keys == []` 且 `unclassified_keys == []` 且 `capture_interface` 完整；候选级参数与 `candidate_id` 一致（`candidate_config_mismatch` 为空） | `derived` |
| `code` | `git_commit` + `numeric_source_tree_sha256`（**进哈希的那个**）+ `declared_source_tree_sha256` + `runtime_snapshot_taken` + `source_coverage == complete` + `dirty_scope ∉ {stray, unknown}` + **`git_dirty` 时 `patch_bundle_sha256` 有效** | `derived` |
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
| `config.effective` 中 `effect ∈ {model_definition, execution_condition}` 的键 | ✓ | 判据见 §3.3.1；含布尔/枚举（`include_L`/`branch`/`adsorption`/`layers`…）与 `sigma.array_sha256`/`sigma.floor` |
| `code.git_commit` + `code.numeric_source_tree_sha256` | ✓ | 基线 + **参与计算**的代码（D-5）。`declared_source_tree_sha256` 只作完整性证据，**不进哈希**——否则改一行画图脚本就换号（F11） |
| `env.env_sha256` + `backend` + `x64_enabled` | ✓ | 环境会改数值；`x64` 改 dtype |
| `identity_contract_version` | ✓ | 世代标签，防"语义变了但载荷没变"撞号（D-4） |
| `code.git_describe` / `git_dirty` / `dirty_paths` / `dirty_scope` / `patch_bundle_*` | ✗ | 展示、范围判据或重建；由 commit+摘要或其自身派生 |
| `code.runtime_loaded_modules` / `source_coverage` / `undeclared_modules` | ✗ | 覆盖性核对，不区分条件（用作降级判据） |
| `code.declared_source_tree_sha256` / `compute_role` / `why_not_numeric` / `workflow_scope` | ✗ | 完整性证据与角色声明；进 `run_id` 的是它的 numeric 子集（D-5） |
| `code.workflow_source_tree_sha256` | ✗ | 进 **`workflow_id`**（§6.3）：它管门禁/筛选/选择/重跑，不管这一次拟合 |
| `workflow.*`（含 `kk_mode`/`kk_opts`/`diag`/`drt_opts`/`with_voigt`/`ref_rwp`/`krange`/`n_seeds`） | ✗ | 描述"跑几次、选哪个"，不描述"哪一次"；`admission_condition`/`decision_condition`/`workflow_scope` 三类都在这里。**但它们全部进 `workflow_id`** |
| `config.unknown_keys` / `unclassified_keys` / `capture_interface` / `normalize` | ✗ | 校验与溯源 |
| `rng.actual.*` / `instance.*` | ✗ | **实例**——进来会让同条件两次跑换号，判重失效（F6） |
| `software.registry_version` | ✗（D-1 已定） | 它是**映射工具**版本，不是被拟合的代码；移入 `provenance` |

`RUN_ID_CONFIG_KEYS` 必须是**闭集 + 显式版本化**（`run_id_config_keys_version`），
否则 `run_id` 会随任何无害开关抖动，样本量被无意义地切碎。

### 6.3 `workflow_id`：第二个号，回答"整个流程是哪一次"

审查 D-5 的附加条件：**单次拟合的身份和工作流的身份不能混为一谈**。所以有两个号：

| 号 | 前缀 | 回答 | 载荷 |
|---|---|---|---|
| `run_id` | `R1-`/`D1-` | 这一次拟合的条件 | §6 表（含 `numeric_source_tree_sha256`） |
| `workflow_id` | `W1-` | 这一套"搜索 + 门禁 + 选择 + 重跑"是哪一次 | 下式 |

```
workflow_id = "W1-" + sha256(
    "v1\nworkflow_id\n" + canonical_json({
      v:                    identity_contract_version,
      workflow_source_tree_sha256:  ...,
      search:               {krange, kmax, wide, fixed_K, layers, branches, n_seeds},
      extras:               {auto_mechanism*, auto_adsorption, with_voigt, ref_rwp},
      gate:                 {kk_mode, kk_opts, strict},        # 设定，不含 kk_status
      guards:               {strict, gate_b, diag, drt_opts},
      pair_mode, retry, retry_kw,
      children:             sorted([子运行的 run_id …])
    }))[:32]
```

四条口径，每条都有反例护着：

1. **工作流源码在里面**（`workflow_source_tree_sha256`）。没有它，改 `bm_grade` 的判定、
   改 `bm_select` 的选择/重跑、改 KK 门禁，`workflow_id` 一律不动——这正是审查指出的洞（W1/W2）。
2. **结果不在里面**。`kk_status`/`kk_performed`/`Rwp`/`elpd` 是观测，不是条件（W4）。
   门禁真的拦住一次运行，体现为 `children` 少一项。
3. **子运行的 `run_id` 在，父的 `workflow_id` 不回流**。子运行不带
   `parent_workflow_id` 进自己的 `run_id`，否则父子互为输入、无法收敛（T16）。
4. **`with_voigt` 只改 `extras`+`children`**，不改 nested 子运行自己的 `run_id`（T16）。

**同号的含义（三条硬边界，与 §0 一致）**：
`run_id` 相同 ⇒ 拟合条件相同；**不**表示初值/采样实例相同（`rng.actual.*` 不进哈希，T8）；
**不**表示产物相同（`instance.*` 不进哈希，T7）；**不**表示工作流相同（那是 `workflow_id`）。

### 6.2 反例矩阵（T1–T24 + W/S/Y/M/X 五组）

这组用例比再加十几个字段有用：它直接检验 `run_id` 的语义是否成立。左列是改动，
右两列是**必须**出现的差异。任一格不符，键表或委托规则就是错的。

**本轮已实现并跑通 54/54**（`test_exec_identity.py`）。表里"实测"列写的是这条
**现在由哪个用例守着**；`v1.2-draft` 时这一列全是"待跑"。

| # | 改动 | `run_id` | 该变什么 | 防的失败模式 |
|---|---|---|---|---|
| T1 | `branch=parallel` → `series` | **必须不同** | `candidate_id` 也变 | 拓扑不同却同号（F9） |
| T2 | `include_L=False` → `True` | **必须不同** | — | 族能力漏哈希（F9） |
| T3 | `layers=2` → `1` | **必须不同** | — | 同上 |
| T4 | `adsorption=None` → `"RL"` | **必须不同** | — | 同上 |
| T5 | `progress=False` → `True` | 同 | `effective.progress` 值变 | `display_only` 混进哈希（样本被切碎） |
| T6 | `plot=True` → `False` | 同 | `effective.plot` 值变 | 同上 |
| T7 | `elapsed_s` / `started_at` 不同 | 同 | 仅 `instance.*` 变 | 实例字段误入条件哈希（F6） |
| T8 | 同 seed、不同实际初值 | 同 | `rng.actual.init_values_sha256` 变 | 实例 vs 条件的边界（B3b 那条线） |
| T9 | 源码里改一个影响数值的常量 | **必须不同** | `numeric_source_tree_sha256` 变 | 代码改动被旧号掩盖（F2） |
| T10 | 改 `bm_circuit`（`plot_only`）一行 | 同 | `declared_source_tree_sha256` 变、numeric 不变 | 装饰性改动换号（F11） |
| T11 | 只升 `registry_version`（映射工具） | 同 | `provenance` 变 | D-1 的回退检验 |
| T12 | 换一份同点数、同频率窗的数据文件 | **必须不同** | `data.sha256` 变 | 数据身份 |
| T13 | 同一文件加掩码 / 改 `n_points` | **必须不同** | `subset_sha256`/`n_points` 变 | 输入截取被当成同一次 |
| T14 | `zscale=1` → `2`（同文件） | **必须不同** | `data.transform` 变 | D-7 |
| T15 | `budget="standard"` → `"quick"` | **必须不同** | `budget_params` 数值变、档名不变 | 档名冒充数值（F4） |
| T16 | `with_voigt=True` → `False` | nested 子运行**同**；父 `workflow_id` 不同 | `workflow.children` 变 | 附加对照污染主身份 |
| T17 | `kk_mode="gate"` → `"warn"` | 同 | `workflow.gate` 变 | 门禁进哈希 |
| T18 | `diag` 阈值改 | 同 | `workflow.guards` 变 | 事后阈值进哈希 |
| T19a | 实际 σ 数组逐位相同、`sigma.mode` 不同（`perpoint`→`scalar` 标签） | **同** | `sigma.mode` 变 | 复合键要认"实际数组"，不是"来源枚举" |
| T19b | 同一 `mode`、实际 σ 数组差 0.01% | **必须不同** | `sigma.array_sha256` 变 | D-8 的落点：数组才是噪声条件 |
| T20 | `adsorption=_UNSET` → 显式 `None`（等价） | **同** | `origin` 变 | 哨兵未解析（F12） |
| T21 | `zeta="1.0"` → `1.0` | **同** | 归一化后同值 | 字符串/数字未归一（F12） |
| T22 | 声明了 `numeric` 模块但这次没加载 | 同 | `declared_not_loaded` 非空 | 覆盖规则误判（应无害） |
| T23 | 加载了未声明的项目模块 | 降 `derived` | `source_coverage=undeclared_loaded` | F7 |
| T24 | dirty 且补丁包缺未跟踪文件内容 | 降 `derived` | `bundle_excluded` 含声明集文件 | F8 |

T10/T19/T20/T21 是 `v1.2` 的核心主张，也是最容易被"改回去"的四格：
它们都要求**同号**，而直觉上"改了配置"应该换号。理由在 §3.3.5–§3.3.6——
`run_id` 认的是"进似然的东西"，不是"写下来的字符串"。

**逐条实测状态（`test_exec_identity.py`，54/54）**：T1–T24 全部有对应用例，名字就是
`case_T1_branch` … `case_T24_bundle_gap`。三处**在写测试时才发现的措辞错误**已订正，
这正是"边写边测"的价值：

| 原措辞 | 问题 | 订正 |
|---|---|---|
| T19「`perpoint,pct=2` → `scalar,abs=` 使 σ 数组逐位相同」 | **做不到**：`_check_sigma` 里 `scalar` 是 `(2,)`、`perpoint` 是 `(2,n)`，形状不同 | 拆成 T19a（同数组不同 `mode`）+ T19b（同 `mode` 数组差 0.01%） |
| T1「`branch` 变，`candidate_id` 也变」 | 真跑时 `candidate_id` 当然也变，于是这条用例即使配置通道漏了也会"通过" | 测试里**固定 `candidate_id`** 只改 `branch`：漏哈希立刻暴露 |
| T10「改 `bm_circuit` 一行，`run_id` 同」 | 真实清单里 `bm_circuit` 仍是 `numeric`（未过运行验证），拿真实清单测必然失败 | 用**夹具**里已写 `why_not_numeric`+`evidence` 的 `plots.py` 测机制；真实模块等实测（M2 钉住零降级） |

### 6.2.1 新增四组：`workflow_id`、`sigma` 数组、状态阶梯、清单自校验

审查要求"不能只测 `run_id`"。W 组测第二个号，S 组测 D-8 的规范化，
Y 组测六块各自缺项时降哪一档，M 组测清单本身（含"降级零条"这条硬约束）。

| # | 改动 | 期望 | 防的失败模式 |
|---|---|---|---|
| W1 | `decision` 模块（夹具 `grade.py`）内容变 | `run_id` 同、**`workflow_id` 变** | 工作流逻辑变了身份不变（D-5 的洞） |
| W2 | 入口脚本内容变（含重跑/选择逻辑） | 两个号**都变** | 入口退出 workflow 层 |
| W3 | `workflow_scope=false` 的模块内容变 | 两个号都同、`declared` 仍变 | 把画图模块算进工作流身份 |
| W4 | `kk_status` `pass`→`fail`、`kk_performed` 翻转 | 两个号**都同** | 结果混进身份（结果不是条件） |
| W5 | `children` 多一个子运行 | `workflow_id` 变、`run_id` 同 | 附加对照污染父身份 |
| W6 | `search.kmax` 5→6 | `workflow_id` 变、`run_id` 同 | 搜索空间混进单次条件 |
| S1 | 用 `b"v1\n2x2\n" + bytes` 独立重算 | 与实现一致 | 摘要不可被第三方复现 |
| S2 | `(2,3)` vs `(3,2)` | 不同 | 形状没进载荷 |
| S3 | 含 NaN / Inf | 报错 | 非法噪声静默获得身份 |
| S4 | 空数组 / `sigma=None`（kind≠none）/ `kind=none` 带数组 | 全部报错；`kind=none` ⇒ `array_sha256=null` | 空数组当成"没有噪声" |
| S5 | `<f8` vs `>f8`；`float32(0.1)` vs `float64(0.1)` | 前两者同摘要；后者**必须不同** | 把"字节序"和"数值"混为一谈（原措辞写反） |
| S6 | C/F 内存布局；转置 | 布局无关；转置**不同** | 把"归一化"写成"敏感性"（原措辞写反） |
| S7 | σ ≤ 0 | 报错 | `Normal.scale` 不接受 0，写进身份才发现就晚了 |
| Y1–Y4 | `env` 摘要不符 / 未登记配置键 / 预算非 `recorded` / `rng.plan` 缺项或 `n_chains` 不符 | `derived`、`derived`、`unconfirmed`（无号）、`unconfirmed`（无号） | 三档台阶被抹平；"没有号"被伪造成号 |
| M1–M7 | 三份 Profile 齐全 / **降级零条** / 路径存在 / 降级候选未应用 / 声明集可复算 / 清单与源码同步 / 降级缺证据被拒 | 见测试输出 | 为了让测试变绿而降低源码角色 |

---

## 7. 校验、重建与不可自证

### 7.1 三层校验，各管一段

| 层 | 谁做 | 查什么 |
|---|---|---|
| 写入时 | runner 自检 | 字段格式（40/64 位十六进制）；`unknown_keys`/`unclassified_keys` 为空；`dirty_scope` 与 `dirty_paths` 自洽；`runtime ⊆ declared`；`declared_source_tree_sha256` 与 `numeric_source_tree_sha256` 均可按声明集复算（且 numeric 子集非空）；`env_sha256` 可复算；`sigma.array_sha256` 可复算；dirty 时补丁包存在且可复算 |
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
| 5 | 后续 runner 有一套**经过实跑验证**的身份契约 | ⚠️ 契约与声明集已实现并跑通 52/52，**但没有一次真实运行**（runner 未授权）⇒ 本条仍不成立 |

**冻结判据**：1–4 成立 + 5 的契约被审查通过并至少经一次真实运行验证（P1）。
不由 `R-` 数量决定——233 条永远 `derived` 也可以是可信的历史基线；把"重跑补齐"当冻结条件，
等于用未来的运行给过去盖章。

---

## 13. 裁决记录（D-1～D-8 全部已定）

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

### 13.4 D-5～D-8：已定（2026-11-14 裁决）

| # | 裁决 | 落地方式（本轮已实现） |
|---|---|---|
| **D-5** | **有条件采纳**：拆摘要，但**必须补第三层** `workflow_source_tree_sha256` | 三层各自落点：`declared`=只记录、`numeric`=`run_id`、`workflow`=`workflow_id`（§3.1.3、§6.3）。`compute_role` 与 `workflow_scope` 是**两个独立维度**，不是划分；入口两处都必须在。审查的原话"改无关常量后数值逐位相同只能作回归验证"写进 §3.1.4，与"必须不出现在运行时集"合为**双条件** |
| **D-6** | **采纳**六值 `effect` | `EFFECTS` 闭集 + `RUN_ID_EFFECTS` 子集；每键的类型与 `effect` 在 `RUN_ID_CONFIG_KEYS` 里一处登记（`run_id_config_keys_version: 1`）。三条原则写进 §3.3.1：不进 `run_id` ≠ 不记录；不改后验 ≠ 不改结论；workflow 身份必须覆盖筛选与选择逻辑 |
| **D-7** | **采纳** `data.transform` 进哈希 | `{fscale, zscale, mask}` 进 `_data_payload`；**职责分离**：`transform` 记**处理规则**，`subset_sha256` 记**处理后的实际子集**（两者都进哈希，改哪个都换号；规则相同而子集不同时只有 `subset_sha256` 变）。T13/T14 各测一半 |
| **D-8** | **采纳**实际数组摘要 | `sigma_run_id_value()` 只投影 `{array_sha256, floor}`；规范化补齐五项：`<f8` 小端、C 顺序、形状进载荷、拒 NaN/Inf、空数组 ≠ `None` ≠ `kind=none`；S1 用独立字节串复算对拍。`mode`/`kind`/`from_data` 留记录、不作替代身份 |

**D-5 的附加条件为什么必要**（审查给的四个例子，逐个对上用例）：`bm_grade` 判定变 →
W1；`bm_select` 选择/重跑变 → W2（入口）；`kk_mode` 门禁变 → T17/W4（设定进 `workflow_id`、
结果不进）；`bm_select` 重跑导致额外子运行 → W5（`children` 变）。
**若只有两层**，这四种改动里 W1 会"身份不动"——这就是原方案的洞，现已堵上。

---

## 14. 实现进度：契约层已完成，runner 待授权

审查明确要求**改串行为并行迭代**（写→测→修，不攒到最后）。按此执行的进度：

| 优先级 | 工作 | 状态 | 动生产代码 |
|---|---|---|---|
| P0 | 裁 D-5～D-8（§13.4） | ✅ 已定（D-5 有条件：补第三层） | 否 |
| P0 | 认下 §3.3.4 三张键表 | ✅ 已落成 `RUN_ID_CONFIG_KEYS`（38 键 × 类型 × `effect`） | 否 |
| P1 | 落 `source_manifest.json`（三份 Profile） | ✅ 15/19/18 个文件，**降级零条** | 否 |
| P1 | 实现身份计算（纯函数） | ✅ `exec_identity.py`：三层摘要、`run_id`、`workflow_id`、status | 否 |
| P1 | 跑 §6.2 反例矩阵 | ✅ **54/54**（T×25、W×6、S×7、Y×4、M×8、X×4） | 否 |
| P0 | 冻结契约（**只冻结验过的**） | ✅ §18 冻结台账；未验证项标 `draft` | 否 |
| **P1** | **写 runner 落字段 + 1 次真实运行**（含补丁包重建校验） | ⛔ **仍待单独授权** | 是，限 runner |
| P1 | 重跑 4 次历史条件建 6 条新记录（`purpose=baseline_rebuild`） | ⛔ 依赖上一行 | 是，跑实验 |
| P2 | 233 条的实验单位与优先级 | ⛔ 未开始（`STAGE2_SPEC.md`） | 否 |

**授权 runner 前的唯一阻塞项**不是设计，而是这轮测试**证明不了**的两件事：
① 声明集是否覆盖了**真实加载**的模块（要一次运行的 `runtime_loaded_modules`，T23 只用夹具）；
② 补丁包能否**重建**工作树（要 dirty 状态下的真实产物，T24 只用夹具）。
其余字段与规则都已有反例护着（§18）。

## 15. 本契约**不**做的事

- 不改任何生产拟合模块（`bm_core`/`bm_top`/`bm_select`/`bmdrt`/`bm_result`）。
  本轮新增的 `exec_identity.py` 是**只读纯函数层**：没有 I/O、没有时间戳、没有环境读取，
  它不参与拟合，也不被任何生产模块 import。
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
| `exec_identity.py`（根） | 本契约**已实现的部分**：三层摘要、`run_id`、`workflow_id`、状态阶梯 |
| `source_manifest.json`（根） | §3.1.3 三份 Profile 的声明集；由 `round_2026-11-13/build_source_manifest.py` 生成 |
| `test_exec_identity.py`（根） | §6.2 / §6.2.1 的反例矩阵（52 条），§18 的证据来源 |

## 17. 修订记录

| 版本 | 日期 | 改了什么 |
|---|---|---|
| `v1.0-draft` | 2026-11-14 | 首版：五个块、`run_id` 纳入口径、降级规则、6 条/233 条处置、Stage 0 判据、D-1～D-4 待裁 |
| `v1.1-draft` | 2026-11-14 | 修 P0-1（声明集/运行时集分离 + 覆盖规则）、P0-2（补丁包含未跟踪内容）、P0-3（`RUN_ID_CONFIG_KEYS` 类型无关 + 捕获接口 + `unclassified_keys`）；修四处内部不一致；D-1～D-4 记为已定并附约束 |
| `v1.2-draft` | 2026-11-14 | **从实际签名反推参数表**：三入口键表（§3.3.4）、字段委托表（§3.3.3）、`sigma` 复合键（§3.3.5）、归一化规则（§3.3.6）、`workflow` 块（§3.3.7）；`effect` 三值 → 六值；新增 F11/F12、§3.1.3（三份 Profile，数字由 `audit_source_profiles.py` 算出）、§6.2 反例矩阵；新提 D-5～D-8 待裁；`data.transform`（D-7） |
| **`v1.2`** | 2026-11-14 | **动工**：D-5 有条件采纳（补第三层 `workflow_source_tree_sha256`）、D-6/D-7/D-8 采纳；新增 `exec_identity.py`（三层摘要 / `run_id` / `workflow_id` / 状态阶梯，纯函数）、`source_manifest.json`（三 Profile，15/19/18 文件，降级零条）、`build_source_manifest.py`、`test_exec_identity.py`（**54/54**）；新增 §3.1.4（降级证据）、§6.3（`workflow_id`）、§6.2.1（W/S 组）、§18（冻结台账）；**订正三处原措辞**（T19、T1、T10，见 §6.2）；`sigma` 改为显式投影（只 `array_sha256`+`floor` 进哈希） |

---

## 18. 冻结台账（只冻结跑过的）

审查的第四条要求：**只冻结已经经过反例验证的字段和规则**，不提前冻结没验过的角色分类。
所以"冻结"在这里有明确含义：

> 本文件这几行不再改措辞、不再改口径；要改必须**升 `identity_contract_version`**
> 并给出对应的反例（`test_exec_identity.py` 里加一格，或说明为什么原来那格是错的）。

### 18.1 已冻结（有可执行反例）

| 项 | 证据 | 口径 |
|---|---|---|
| 三层摘要 + 两个独立维度（`compute_role` / `workflow_scope`） | T9 T10 W1 W2 W3 M2 M5 | `declared` 只记录、`numeric` 进 `run_id`、`workflow` 进 `workflow_id`；层名域分隔 |
| `run_id` 载荷构成（§6 表） | T1–T4 T11–T15 T19–T21 X2 X3 | 六块：`candidate_id` / `data` / `rng.plan` / `budget` / `config(effect∈两值)` / `code+env` + 版本 |
| 展示与实例字段不进哈希 | T5 T6 T7 T8 W4 | `progress`/`plot`/`elapsed_s`/`init_values_sha256`/`kk_status` |
| `workflow_id` 载荷（§6.3） | T16 T17 T18 W1 W2 W3 W5 W6 | 源码 + 搜索 + 门禁设定 + guards + children；**结果不进** |
| `sigma` 显式投影与数组摘要 | T19a T19b S1–S7 | 只 `{array_sha256, floor}`；`<f8`/C/形状/拒 NaN/空数组拒收 |
| `data.transform` 与 `subset_sha256` 分工 | T13 T14 | 规则 vs 实际子集，两者都进哈希 |
| `effect` 六值闭集 + 键表（38 键，含 `voigt_K` 父子双效） | Y2 M2 X4 | 未登记键 ⇒ `config_unclassified` ⇒ `derived` |
| 状态阶梯（六块 → 三档） | Y1 Y2 Y3 Y4 T22 T23 | `data`/`rng`/`budget` 缺 ⇒ `unconfirmed` 且**无号**；`code`/`env`/`config` 缺 ⇒ `derived` |
| `sha256` 语义与 registry 一致 | X1（+ registry D12） | 64 位十六进制；无效**抛错**不退化 |
| 声明集合法性与清单同步（含预览可复算） | M1–M8 | 降级须 `why_not_numeric`+`evidence`；入口两处都在；`--check` rc=0 |

### 18.2 未冻结（**没有真实运行**，所以不许当结论用）

| 项 | 为什么还不行 | 需要什么 |
|---|---|---|
| `runtime_loaded_modules` / `source_coverage` 的判定 | T23 用夹具，未读过真 `sys.modules`；闭包与运行时的差集还没实测 | 一次真实运行的三个时点快照 |
| 补丁包**重建**工作树 | T24 只测"缺包 ⇒ 降级"，没测过"解包后复算摘要一致" | dirty 状态下的一次真实运行 |
| 降级双条件（`plot_only`/`decision` 等） | 本轮**零降级**（M2）；`pending_demotions` 全部 `applied:false` | 关功能运行 / 改常量重跑 |
| `capture_interface` / `unknown_keys` 的真实值 | 夹具里是手写的；真实调用点的 `**kwargs` 还没被枚举 | runner 捕获 |
| `env` 五项的真实采集 | `platform`/`packages` 在夹具里是常量 | runner 采集 |
| D-3「线程数不进哈希」的量化前提 | 没有线程敏感性实验 | `sampler_sensitive` 那条线的实测 |
| §12 判据 5（经过实跑验证的契约） | 现在就停在这一条 | 上面两行 |

**这张表不是待办清单，是免责边界**：18.1 的每一项都有反例护着，18.2 的每一项都只是
设计合理。把 18.2 说成"已验证"，等于重犯 `HISTORICAL_EXEC_IDENTITY_GAP.md` 里那类错——
**没有证据的地方，措辞必须软。**
