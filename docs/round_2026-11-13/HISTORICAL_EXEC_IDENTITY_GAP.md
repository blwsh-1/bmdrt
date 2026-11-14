# 历史执行身份缺口：只读审计（2026-11-14）

状态：**审计完成，结论是"不补"**。这份文件回答"缺什么、能不能从现有证据补上"，
不回答"怎么补"——因为唯一可行的答案在**往前**的 runner 里，不在历史产物里。

- 不改 `records.jsonl`（仍是 239 条 / 独立 239 / 严格 237 / `R-` 0 条）；
- 不改 `circuit_registry.py`（仍 `0.7.0` / `b2675c8255bea716`）；
- 不写 runner、不进阶段 1；**Stage 0 仍不冻结**。

复跑：

```
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-13/audit_exec_identity_gap.py
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-13/audit_exec_identity_gap.py --json
```

审计脚本只读：不写 `evidence/`、不改 `out/`，所有数字从 `records.jsonl` 与磁盘现状重算
（不引用任何成文结论，避免"文档说 233，所以 233"）。

## 1. 缺口大小：6 条一键之遥，233 条差两项

`_conditions_recorded()` 的四个条件逐条重算（不看 `run_id_status` 字符串）：

| 齐备度 | 记录数 | 缺什么 |
|---|---|---|
| 1/4 | **233** | `seed_artifact` + `budget_recorded` + `code_version` |
| 3/4 | **6** | 只缺 `code_version` |

四要素里 `data_content_sha256` 是**唯一满的**（239/239）。也就是说缺口不是"到处都漏"，
而是集中在两处：**seed 是不是产物写的**、**代码版本有没有**。

那 6 条更值得看清：它们只占 **4 个唯一 `run_id`**，而**本批仅有的 2 对
`possible_duplicate` 全在这 6 条里**（`1113` 与 `default_init` 两对，seed 与预算都由产物记录、
指标逐位相同）。所以，**假如**能补上 `code_version`，这两对会升成
`confirmed_duplicate`/`rerun_same_conditions`，独立数会从 239 掉到 237（与严格口径重合）。
这条不是细节：它说明"补一个版本号"不是给记录贴个标签，**它会改计数**——正因如此，更不能用
推测值去补。

## 2. 产物里没有任何软件身份痕迹

9 份证据产物全扫（`version|git|commit|env|python|numpyro|jax|script|code|platform|timestamp|dirty` 等键名）：
**0 命中**。`*.nested.json` 的顶层键只有
`status / warning / notes / input / config / params / derived / metrics / diagnostics / voigt_reference / files / elapsed_s`；
pilot 的 `results.jsonl` 行键里也没有。**没有时间戳、没有版本、没有环境快照。**

结论：`code_version` 不是"记了但没解析出来"，是**从来没写进产物**。

## 3. 产出时刻只能给上界，不是身份

| 产物产出时刻（文件 mtime） | 目录 | 之前最近一次提交 |
|---|---|---|
| 2026-11-12 21:45 / 21:47 / 21:48 / 21:49 | `out/nested_fit_cap_{smoke,standard,control,pass}` | `a309633`（21:24，早 21 分钟） |
| 2026-11-13 15:39–15:48 | `out/_user_fit_nested` | `fa5c5f9`（14:11，早约 1.5 小时） |

两个理由让它不能当身份：

1. **脏工作树无法排除。** 那几个小时里正在改脚本（`out/_user_fit_nested/` 里就躺着 15:44 的
   `fit_family_compare.py` 副本），提交时间早于产出时间不等于产出时就等于那个提交。
2. 时间只给**上界**：`a309633` 是"至早于此"，不是"即此"。把上界写成 `code_version`，
   等于把"没有反证"当成"有证据"——与 0.3.0 拿默认值冒充真实档名是同一类错。

## 4. 磁盘上的代码副本不够格

盘上共 5 个 `.py`：2 个入口脚本各两份（`out/_user_fit_nested/` 与
`evidence/stage0_three_spectra/scripts/`，**逐字节相同**），另有 1 个 `redraw.py` 无仓库对应件。

但它们 `import bm_top / bm_real / bm_select / nested_fit`——**这些库模块一份副本都没有**。
所以即便入口脚本与仓库同字节，也不能证明"当时被 import 的库"是同字节。`recorded` 要的是
**软件**版本，不是入口脚本版本。而且 6 条里只有 2 条经过 `fit_family_compare.py`，
其余 4 条连入口脚本的副本都没有。

## 5. 三条捷径，逐条否掉

| 捷径 | 为什么不行 |
|---|---|
| 拿 `a309633` / `fa5c5f9` 当 `code_version` | 时间上界 ≠ 身份；脏树不可排除；等于把推测写进产物 |
| 拿入口脚本 sha256 当 `code_version` | 覆盖不到被 import 的库；6 条里 4 条无脚本副本；`software` 的语义是软件不是脚本 |
| "重跑一次指标相同，所以旧记录也是这份代码" | 重跑是**新的执行**，拿新 `run_id`；旧记录仍是 `derived`。重跑能加新记录，不能给旧记录补字段——补了就是伪造来源 |

## 6. 缺口只能向前关

唯一可行路径，都属于阶段 1 的 runner（本轮**不写**）：

1. 产物落 `software.code_version`：`git describe --tags --dirty --always`。**`--dirty` 必须如实记**，
   否则"版本"会被读成"这份代码"——脏树上跑出的 `R-` 要么不发，要么带 `dirty` 标记后仍降级。
2. 落依赖版本（python / numpyro / jax / numpy），因为 HN 拟合的行为会随采样器版本变。
3. 数据内容哈希继续在写（已具备）；`seed` 与 `budget_params` 必须由产物写，不能由档名推。

代价要提前说清：**6 条 = 4 次运行**（`elapsed_s` 约 58 s/次，合计约 4 分钟）重跑即可拿到 6 条
`R-`；而 **233 条要重跑 233 次**，且这些记录多数连 seed 都没由产物写——重跑前得先定 seed 策略，
那是阶段 1 的事，不是本轮的事。

**这份清单已按 §6 展开成规格**：`EXECUTION_IDENTITY_CONTRACT.md`（草案 `v1.1-draft`，未实现）。
其中与本文件直接相关的三处修正：`git describe --tags --dirty --always` 只作**展示**，
不承担身份——两次 `-dirty` 的未提交内容可以完全不同；代码身份由 `git_commit` +
**已声明源码集摘要**承担，且声明集与**运行时加载集**必须互相核对（`runtime ⊆ declared`，
违反即降级）——入口脚本相同 ≠ 库相同，正是本文件 §4 查出来的问题；dirty 从「身份」
降为「范围判据」，且 dirty 时必须落含未跟踪文件内容的补丁包，否则不许说「可重建」。
见契约 §3.1、§4、§7.2。

## 7. 与既有结论的关系

- 与 `STAGE0C_SCHEMA_CONTRACT.md` §9.4/§10.5/§11.7 的"`R-` 0 条卡在代码版本"一致，且更具体：
  **卡的是一个从未被写下的字段**，不是解析规则。
- 与 `STAGE0_RECORDS_MAP.md` §4 的三类事实划分一致：本审计的所有数字都是 A 类（产物核对）或
  B 类（代码路径可读），没有一条把 C 类（脚本判定）倒推成 A 类。
- 与准入文档 §8.1 一致：登记合法 ≠ 准入通过；同理，**证据齐备 ≠ 结论成立**——本文件只处理
  前者，不触碰 §13 `0/3`、§14 `4/14`、§17 `0/10`。
