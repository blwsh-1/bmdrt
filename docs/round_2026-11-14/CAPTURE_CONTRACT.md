# 采集契约：harness 归属 / 路径规范化 / 采样事件归属

**日期**：2026-11-14　**基线**：`878daa7`（身份契约 v1.4）
**实现**：`capture_contract.py`（纯函数 + 一个无 I/O 的事件日志）
**反例**：`test_capture_contract.py` —— **31/31**（P×10 / H×7 / E×8 / I×6）
**真实运行证据**：`snapshot_probe.py` → `out/snapshot_probe/*.json`（见 §6）

本文件只定**采集**口径。`exec_identity.py` 的哈希载荷、状态阶梯、覆盖判定一行没改：
`identity_contract_version` 仍是 `1.4`，三层摘要与两个号逐位不变（I1/I5 守着）。

---

## 0. 三条裁决

| 编号 | 裁决 | 一句话理由 | 状态 |
|---|---|---|---|
| **C-1** | runner/harness 作为**显式声明的 harness 项**纳入源码身份，`compute_role` 必须非 `numeric`；采集层**不许按文件名过滤** | 不声明 ⇒ 每次运行永久 `derived`；声明成 `numeric` ⇒ 改编排代码就静默换 `run_id` | 已实现，31 格里 H1–H7 + I1–I6 |
| **C-2** | 两侧都走 `realpath → 是否在仓库根内 → POSIX 相对路径`；仓库外的**同名**文件记 `external_modules`，其余忽略并计数 | `D:/…/bm_top.py` 与 `bm_top.py` 用"非绝对、无反斜杠"都合法却永不相等 ⇒ **全量**未声明 | 已实现，P1–P10 |
| **C-3** | 每次 `MCMC.run` 一条**事件**（序号/入口快照/生效参数/所属会话/开始结束状态），会话显式开关，进程内复用必须 `reset()` | 单槽位会被后一次采样覆盖；状态不清零会让第二次读到第一次 | 已实现，E1–E8 |

三条都只改**采集**，不改身份。这是"先处理采集口径、再做补丁包重建"的全部内容。

---

## 1. 为什么先做这个

`SNAPSHOT_PROBE.md` §4 记录的四处在实测里都**不报错**：

1. 探针自己出现在 `sys.modules` 里 ⇒ `undeclared_loaded`；
2. 路径写成 `D:/…` ⇒ 与声明集永不相等 ⇒ 全量未声明；
3. 一次工作流两次采样 ⇒ 单槽位留最后一次 ⇒ 记录张冠李戴；
4. 钩子状态跨运行泄漏 ⇒ 后两次报第一次的快照（本轮实测踩到）。

这四条的共同点：**记录看起来完整，判定也给了，只是错的**。补丁包重建得再完美，
也修不了"这次采样的快照属于另一次拟合"。所以顺序是先 1–3，再 ②。

---

## 2. C-1：harness 归属

### 2.1 规则

harness（runner、采集脚本、编排脚本）在声明集里长这样：

```json
{"path": "run_identity.py", "role": "entry", "sha256": "…",
 "compute_role": "cli_only", "runtime_scope": "harness",
 "why_not_numeric": "只做采集与编排，不进阻抗模型计算",
 "evidence": "I6：把它当 numeric 会改变 run_id", "workflow_scope": true}
```

`capture_contract.harness_item()` 强制四条，缺一即报错：

| 约束 | 缺了会怎样 |
|---|---|
| `compute_role != "numeric"`，且在 `HARNESS_COMPUTE_ROLES` 内 | 编排代码一改就换 `run_id`，且**不报错**（I6 实测：`R1-5f9727f…` → `R1-cc20325…`） |
| `why_not_numeric` 非空 | 与 `check_declared_set` 的既有要求一致，提前报错比在 `evaluate()` 里变成 `declared_set_invalid` 好 |
| `evidence` 非空 | 同上 |
| `path` 是 POSIX 相对路径 | 绝对路径与声明集永不相等（C-2） |

默认值在这里正好是错的那一边：`exec_identity` 对省略的 `compute_role` 按 `numeric` 算
（§3.1.3 的保守默认，防止误降合并两次不同运行）。所以 harness **不许省略**，必须显式写。

### 2.2 三层摘要各进哪一个（I1，真实 19 文件声明集上复算）

| 层 | 加 harness 前 | 加 harness 后 | 结论 |
|---|---|---|---|
| `declared_source_tree_sha256` | `96094464…` | `2d9d79ad…` | **变**：改 runner 要能被看见 |
| `numeric_source_tree_sha256` | `d99bcdd0…` | `d99bcdd0…` | **不变**：编排代码不换 `run_id` |
| `workflow_source_tree_sha256` | `180f29e2…` | `7a00583f…` | **变**：`workflow_scope=True` |

`declared_source_tree_sha256` 只哈希 `path`+`sha256`（`source_tree_sha256` 的实现），
`role`/`compute_role`/`runtime_scope` 只**选择**层、不进哈希。因此新增 `runtime_scope`
**不改变任何号码**——这是它能被现在引入而不升契约版本的原因（I5 守住这一点：
`runtime_class`/`role`/`modules` 这些新标签加进去，两个号一位不变）。

### 2.3 为什么不是"采集时过滤文件名"

`classify_runtime()` **不做任何排除**：没声明的 harness 照样是
`runtime_class="undeclared"`，照样留在 `runtime_loaded_modules` 里，照样让
`exec_identity` 的集合复算得到 `undeclared_loaded`（H6、I3）。

排除的唯一合法途径是**声明**。这样"记录里为什么没有那个文件"永远有一个可查的答案：
它在声明集里，带 `runtime_scope="harness"` 和一条 `why_not_numeric`。

---

## 3. C-2：路径规范化

### 3.1 规则（`normalize_path`）

1. 非字符串/空白 ⇒ `invalid`；
2. 相对路径按**仓库根**解释（不是 cwd），`abspath` 消掉 `..`；
3. `realpath` 解析符号链接；
4. 不在仓库根内 ⇒ 再分：字面在根内但 realpath 在外 ⇒ `symlink_escape`，字面也在外 ⇒ `external`；
5. 在根内 ⇒ `relpath` 后换成 `/`，得到 POSIX 相对路径。

四个状态：`project` / `external` / `symlink_escape` / `invalid`。
`symlink_escape` 单独一档，因为它"看起来在仓库里"，最容易被误认成项目模块。

### 3.2 冲突三类（`normalize_set`）

| 类 | 判据 | 声明集（`strictly=True`） | 运行时（宽松） |
|---|---|---|---|
| `duplicate` | 规范化后 path 相同 | **报错**（分母不可复算） | 只记一条 |
| `case_collision` | 只差大小写 | **报错** | Windows 上合并，POSIX 上是两个文件 |
| `alias` | 不同 raw 指向同一 realpath | 记证据 | 正常（同一文件多个模块名） |

平台语义实测（P6）：

- **已存在**的文件，`realpath` 会归一到磁盘上的真实大小写 ⇒ `BM_TOP.py` 与 `bm_top.py` 是同一个身份；
- **都不存在**的一对名字，只能靠 `normcase` 判 ⇒ Windows 记 `case_collision`，POSIX 是两个独立文件。

### 3.3 什么才算 `external_path`（这一条是本轮最容易被做错的地方）

第一版实现把"不在仓库根内"一律当 external —— 实际跑出 **2016 个外部模块**
（numpy、arviz、标准库 `.pyd`……），`external_path` 立刻失去意义：每条记录都降级。

契约 §3 表对 `external_path` 的定义是"editable install 指向别处 / site-packages 里有一份
**项目副本**"——那是一份**跑错位置的项目代码**，不是依赖。区分办法是**按名字**：

> 只有当仓库外那个文件的 basename（或模块名）与声明集里某一项对得上，才算 `external_path`。

对不上的（`numpy`、`sitecustomize`、标准库 `.pyd`）进 `n_ignored` 计数，不成条目、不降级。
P9/P10 分别锁住两侧：仓库外的同名 `core.py` ⇒ `external`；无关的 `vendored.py` ⇒ 只计数。

限制（写在这里而不是藏着）：仓库外一个**从未声明过**的项目模块不会被记成
`external_path`，只会进 `n_ignored`。它本来也不在身份里，但"没被追踪"和"被追踪后判定通过"
是两件事，读者要能从 `n_ignored` 看出来。

---

## 4. C-3：采样事件归属

### 4.1 一条事件

```json
{"event_id": "run-a#0", "owner": "run-a", "idx": 0, "seq": 0,
 "sampler": "numpyro.infer.MCMC.run", "started": true, "status": "ok",
 "error": null, "snapshot": {...}, "effective": {...}}
```

| 字段 | 作用 |
|---|---|
| `event_id` / `idx` | 会话内序号，**不覆盖**：`#0` 永远是第一次 |
| `seq` | 全局单调序号（跨会话），用于排序与对账 |
| `owner` | 归属会话：运行实例或子任务。父子会话可嵌套（`with_voigt` 的子采样） |
| `status` | `running`（没关）/ `ok` / `error`。`complete()` 只返回 `ok` |
| `snapshot` / `effective` | 进入采样器时的模块快照与生效参数 |

### 4.2 三条不许

1. **不许覆盖**：第二次采样不能冲掉第一次（E1）。
2. **不许猜归属**：没有打开的会话时，采样事件直接 `SamplerOwnerRequired`，不落到"最近一个"（E6）。
3. **不许复用**：同一 owner 再次 `session()` 必须 `reset()`（E7）——这正是本轮实测踩到的
   "后两次报第一次快照"的结构性版本，把它从"不报错的错"变成"会报的错"。

### 4.3 归属的真实形态

`two_fits` 实测（`out/snapshot_probe/two_fits.json`）：

```
probe#0      ok   主拟合（parallel）
probe/ref#0  ok   子会话里的对照拟合（series）
```

主会话的两次采样不会被对照采样顶掉；`with_voigt=True` 与 Profile C 的两族对拍是同一个形状。

---

## 5. 反例矩阵（31 格）

| 组 | 格子 | 测什么 |
|---|---|---|
| **P**（10） | P1 绝对/相对同一身份；P2 四种写法归一；P3 越界判外部；P4 链接指向仓库内；P5 链接逃出仓库；P6 只差大小写；P7 重复路径；P8 非法写法；P9 仓库外同名文件；P10 无关外部只计数 | `normalize_path` / `normalize_set` 的归一与冲突 |
| **H**（7） | H1 拒绝 `numeric`；H2 必须给理由与证据；H3 必须相对路径；H4 字段形状；H5 声明后不算未声明；H6 未声明照样失败；H7 解析不出的不丢 | harness 的四条约束 + "过滤不参与判定" |
| **E**（8） | E1 两次采样不覆盖；E2 会话隔离；E3 嵌套会话；E4 未关闭不算完整；E5 异常标 `error`；E6 无会话拒收；E7 复用必须 `reset`；E8 `seq` 单调 | 事件归属的所有错法 |
| **I**（6） | I1 harness 不进 numeric（真实声明集）；I2 声明后 `recorded`；I3 未声明则 `derived`；I4 外部则 `derived`；I5 新标签不进哈希；I6 numeric harness 换号 | **桥接到身份层**：证明"只需声明，不用改算法" |

I 组是这一轮的验收条件：I2 用真实记录夹具 + 真实 `classify_runtime` 输出喂
`exec_identity.evaluate()`，得到 `recorded`；I3 用同一份代码去掉声明，得到
`derived + undeclared_loaded:run_identity.py`。**同一条代码路径，一个号一个降级，
差别只在一行声明。**

---

## 6. 真实运行的证据（不是夹具）

`snapshot_probe.py` 用 `capture_contract` 做采集，跑真实 `bm_top.fit_top`（谱
`archive/2EMC1-1_0005mA.csv`，56 点）：

| 口径 | 判定 | 未声明集 | 计数（project/harness/ext/undecl） |
|---|---|---|---|
| 声明集 = manifest 原样 | `undeclared_loaded` | `capture_contract.py`、`snapshot_probe.py`、`exec_identity.py` | 14 / 0 / 0 / 3 |
| 声明集 = manifest + 3 个 harness | **`complete`** | 空 | 14 / 3 / 0 / 0 |

- 独立进程 ×3、进程内 ×3、`quick` 800/400/2 档 ×1：**全部 `complete`**，`numeric_unchanged = True`；
- 三时点项目文件数 17/17/17，`point_delta` 全空（14 个 project + 3 个 harness）；
- `declared_not_loaded` = `bm_circuit`/`bm_ref_rwp`/`bmdrt`/`fit_store` + **`config.json`**——
  最后一项恒在，因为它不是模块（见 `SNAPSHOT_PROBE.md` §3.4）。

---

## 7. 三条边界（不许越读）

1. **"B 路径三时点同集"不等于"三时点机制没用"**。它只说明 B 的项目模块在导入期就加载完了。
   不许为了证明三时点有增量而刻意制造延迟导入——那会把一个采集问题变成一个造假问题。
2. **"五项采样参数可读"不等于"采样参数身份已闭合"**。事件里存的是进入采样器那一刻的
   属性值；它**没有**区分显式传入/默认值/委托入口，也没有核对三入口的委托表。那是 ③ 的活。
3. **"能 `complete`"不等于"覆盖判定已经可信"**。`complete` 的输入是
   `runtime_loaded_modules`，而它由 harness 自己采。harness 少记一个文件，判定只会更宽松
   （少一个 undeclared）。这条没法靠采集层自证，只能靠 ④ 的 runner 落盘后做一次
   "同一谱两次运行、快照逐位相同"的复算。

---

## 8. 未做 / 待裁

| 项 | 状态 |
|---|---|
| 把 `runtime_scope` / `runtime_class` / `sampler_events` 写进**真实记录** | 未做（runner 未授权）。写进去也不改哈希（I5），但那时该给 `code` 块补一节文档 |
| `exec_identity.py` 该归 `reference_only` 还是另有分类 | 本轮在探针里按 `reference_only` 声明，**属身份算法自身的分类问题，需单独裁** |
| `sitecustomize` 这类解释器注入的 `.py` | 归 `n_ignored`（不同名）。若哪天它的名字撞上声明项，会被判 `external_path`——宁可误报不可漏报 |
| `.pyd`/`.so` 编译扩展 | 快照只收 `.py`。若将来有项目自带扩展，需要扩 `.py` 这一条 |
| Profile A / C 的真实采集 | 未测（A 的进程入口不同，C 一次比较两次采样已由 `two_fits` 间接覆盖形状） |
