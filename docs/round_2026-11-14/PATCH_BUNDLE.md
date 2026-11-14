# 补丁包重建（②）：一个 dirty 工作树能不能被完整打包并原样恢复

**结论：能，但只在"每一份字节都有出处"这个前提下。** `pb1` 契约 + 参考实现 +
25 格反例 + 四个真实场景往返都过了；`complete` 的定义是**静态完整 ∧ 重建逐项相同**，
不是"脚本没报错"。

四个验收问题的直接回答：

| 问 | 答 | 依据 |
|---|---|---|
| dirty 工作树能否被完整打包 | **能**。19 base + 3 patch + 2 blob 的脏克隆往返一次过；每一份字节都有 `source` 解释从哪来 | §5 C |
| bundle 能否在独立目录重建 | **能**。干净树 398 个文件、脏树 398 个文件，都在临时目录重出，不覆盖工作树 | §5 A/C |
| 重建后三层摘要是否逐一相同 | **是**。`declared` / `numeric` / `workflow` 三个都逐位相同（干净树与上一轮采集实测同值） | §5 A |
| 缺一个声明文件能否可靠拒绝 | **能**。`incomplete` + **拒绝重建**，不是"打了大部分算成功" | §5 D |

改了什么：新增 `patch_bundle.py`（纯函数）、`patch_bundle_io.py`（I/O）、
`test_patch_bundle.py`（25 格）。`exec_identity.py` 一行没改，
`identity_contract_version` 仍 `1.4`，生产拟合代码零改动，runner 未启动。

---

## 1. 为什么不能只有 `git diff`

这是本轮唯一的"设计"判断，其余都是落实。三条，每条都在真实仓库会踩：

1. **diff 不含未跟踪文件。** 声明集里的新模块只要还没 `git add`，diff 里一个字都没有，
   而它恰恰可能是这次运行的数值代码。
2. **diff 不含 ignored 文件。** `out/` 被 ignore；`config.json` 今天是跟踪的，
   但"将来会不会落进 ignore 规则"不是它能保证的——打包器不该依赖这个偶然。
3. **diff 可能"沉默"而字节不同。** `core.autocrlf` 归一、clean/smudge filter、
   `core.fileMode`，三者都会让 `git diff` 什么都不报、工作树字节却和 commit 里的
   blob 不相等。只看 diff 的打包器会漏掉它，然后重建出一个**摘要不同的工作树**——
   不报错，只是身份错了。

所以 bundle 对每个声明文件记 `source`，三选一，必须能解释每一份字节：

| `source` | 含义 | 存什么 | 谁验 |
|---|---|---|---|
| `base` | 与 base commit 的 blob **逐字节相同**（比对过 sha256） | 只存 sha256 + git mode | 重建后逐文件摘要 |
| `patch` | 由 staged / unstaged diff 恢复 | 只存 sha256 + 引用哪段 patch | 重建后逐文件摘要 |
| `blob` | 其余一切：未跟踪、ignored、**diff 沉默但字节不同** | 完整内容（base64）+ mode | 静态即可验（`decode_blob`） |

`blob` 是**兜底**，不是"未跟踪专用"。宁可多存几 KB，也不留一份没有出处的字节。
脏克隆那个场景里 bundle 从 6.4 KB 涨到 109 KB，涨幅全是 diff 与两段 blob——
这个代价是明码的。

## 2. 格式 `pb1`

单个 JSON 文件，确定性序列化（键排序、ASCII 转义、无空格、拒 NaN/Infinity）。
字段：

| 字段 | 内容 |
|---|---|
| `format` / `version` | `pb1` / `1.0`。载荷布局一变就换号（同 `IDENTITY_CONTRACT_VERSION` 的纪律） |
| `base.commit` | 基准 commit（7–64 位小写十六进制） |
| `declaration[]` | 声明集：`path` + 身份字段（`role`/`compute_role`/`workflow_scope`/…）。**自带**，不重建时反查工作树 |
| `files[]` | 每个声明文件一条：`path`/`mode`/`source`/`sha256`/`bundle_digest`（+ `content_b64` 或 `patch_ref`） |
| `patches[]` | `name`/`sha256`/`bytes`/`paths`/`text`（`git diff --binary` 原样输出） |
| `missing[]` | 有声明、无记录的文件（**这是硬失败的输入**，不是警告） |
| `counts` | 各档计数，便于一眼看出"19 个文件里 12 个是 blob"这种异常 |
| `capture` | 采集侧元数据：工具版本、`core.autocrlf=false`、`notes`（diff 沉默清单） |
| `tree_digest` | 整体摘要，**覆盖除自身外全部字段** |

两个摘要不是冗余：

- `sha256` = **裸内容**的摘要，与 `exec_identity` 消费的源码哈希同口径；
- `bundle_digest` = **路径 + mode + 内容**的摘要（分域 + 长度前缀）。内容一样但
  exec 位不同、或同一份内容出现在两条路径上，都必须是两个身份。
- `tree_digest` 只管"整包没被改过"。**没有 `bundle_digest` 时它一个人挡不住**：
  改完内容再把总摘要重算一遍，总摘要看起来完全自洽（P16 就是这个反例）。

## 3. 判定：两段，不合并成一个布尔

```
verify(bundle)     →  corrupt > incomplete > complete      （只看 bundle 自身）
rebuild(bundle)    →  matched / mismatched / not_rebuilt    （独立目录重建后逐项比）
roundtrip.complete =  verify == complete  AND  rebuild == matched
```

`verify.complete` **只**表示"结构完整、每份内嵌内容自洽"。它**不**表示"这份 bundle
能重建出原来的工作树"——那是 `rebuild` 的事。把两者混成一个 `success` 字段，
就是这一轮要防的那类错误：措辞比证据走得远。

`rebuild` 比三样，缺一不可：

1. 逐文件 `sha256`（19/22/24 项全等）——"三层摘要相同"不能替代它，因为少一个
   **不参与某层**的文件时，那一层看不出来；
2. `mode`（Windows 上标 `skipped_windows`，见 §6.1）；
3. 身份三层 `declared` / `numeric` / `workflow`（`exec_identity.source_tree_sha256`，
   声明字段取自 bundle，字节取自重建目录）。

## 4. 反例矩阵（`test_patch_bundle.py` 25/25）

16 格纯函数 + 8 格真实 Git 往返 + 1 格引用护栏。挑几格说为什么：

| 编号 | 情形 | 要求的判定 |
|---|---|---|
| P4 | 声明 3 个文件，只打包 2 个 | `incomplete` + `missing=["c.py"]` + 身份项少一项。**P0** |
| P5 | 改 blob 内容 | `corrupt`，且 `file_digest_mismatch` 与 `tree_digest_mismatch` **两道都报** |
| P6 | 改任意顶层字段（`capture`/`counts`/`base`/`declaration`/`missing`） | 全部撞 `tree_digest` |
| P16 | 改内容**并重算总摘要** | 只剩 `file_digest_mismatch` 能抓——这格证明"一个总摘要不够" |
| P8 | 文件不在声明集 | `assemble` 直接拒绝；`verify` 报 `file_not_declared` |
| P9 | `../x`、`/abs`、`A:/`、反斜杠、`CON.py`、尾空格、只差大小写 | 全部拒绝（跨平台一致，不按平台行为分叉） |
| P10 | 采集契约归一出的 `D:/…` 路径 | 与 bundle 的写入侧检查**同解**（不是第二套规则） |
| P15 | `tier_items` 喂给 `exec_identity` | 三层互不相等；改 `workflow_scope=False` 的文件只动 declared/numeric |
| G2 | staged + unstaged + 未跟踪 + ignored 同时存在 | 五种归属正确；`sub/c.py` 被两段 patch 同时认领（`staged+unstaged`） |
| G3 | 删一个声明文件 | `incomplete`，且 `rebuild` 返回 `not_rebuilt`（**不重建**） |
| G4 | 落盘后被改一个 `sha256` | `corrupt`（完整性不只靠内存对象） |
| G8 | diff 沉默但字节不同 | 兜底 `blob`（`why=diff_silent_but_bytes_differ`），重建仍 `matched` |
| X1 | 生产模块引用这三层 | 报错（`patch_bundle_io` 被 `exec_identity` 反向引用会把身份和工具链缠住） |

**"退回即失败"实测**（改源码 → 跑对应反例 → 恢复，恢复后 sha256 与原文件一致）：

| 退回的防线 | 结果 |
|---|---|
| 非声明文件检查 | P8 失败（`应当报错，实际没有`） |
| `missing` 计算 | P4 变 `complete` |
| 拒绝 NaN/Infinity | P14 通过（不该通过） |
| `decode_blob` 三道校验 | P5 只剩 `tree_digest_mismatch`（少了内容级的证据） |
| `base` 逐字节比对 | G2 里 `a.py` 变 `source=base`，重建 `mismatched` |

最后一条最能说明问题：少了逐字节比对不会崩，只会把"改过的文件"记成"没改过"，
然后重建出来的摘要**不同**——错在身份，不在异常。

## 5. 真实工作树四场景（`patch_bundle_probe.py`，commit `a526fe5`，24.7 s）

B 声明集（`source_manifest.json` 的 19 个文件）+ 3 个 harness 项。脏状态在
`git clone --no-local` 出来的**克隆**里造，不碰用户的工作树。

| 场景 | 声明 | 归属 | bundle | 重建 | 三层摘要 |
|---|---|---|---|---|---|
| **A** 干净工作树 | 19 | 19 `base` | 6 461 B | 398 文件，19 项全等 | `96094464…` / `d99bcdd0…` / `180f29e2…` 全等 |
| **B** 干净 + 3 harness | 22 | 22 `base` | 8 220 B | 22 项全等 | `declared` 变（`c6f0d7b8…`）、**`numeric` 与 A 逐位相同**、`workflow` 变 |
| **C** 脏克隆 | 24 | 19 `base` + 3 `patch` + 2 `blob` | 109 468 B | 24 项全等 | `b82100fe…` / `b207b637…` / `e8c0350c…`，与 A 全不同 |
| **D** 脏克隆删声明文件 | 24 | 23 条记录，缺 `bm_ws.py` | — | **`not_rebuilt`** | — |

C 的归属明细（这一段是"每份字节都有出处"的证据）：

```
bm_ad.py              patch  ← unstaged
bm_core.py            patch  ← staged
bm_corr.py            patch  ← staged+unstaged
probe_new_module.py   blob   ← untracked
out/probe_ignored.json blob  ← ignored
```

两段 patch：`staged` 57 234 B（`bm_core.py`、`bm_corr.py`）、`unstaged` 14 901 B
（`bm_ad.py`、`bm_corr.py`）。体积大是因为探针**整文件重写**而不是追加——
这反而是更严的测试（每一行都要靠 patch 恢复）。

A 的三层摘要与上一轮采集实测**逐位相同**（`96094464…` / `d99bcdd0…` / `180f29e2…`），
这不是巧合：两条路径都在算"磁盘字节的 sha256 按层分域聚合"。任何一条算错，
这两个数就对不上——所以它同时是跨轮的回归点。

B 顺带给出 C-1 的独立复算：加 3 个 harness 后 `declared` 变、`numeric` **一位没变**、
`workflow` 变。harness 的改动看得见，但不会让每次拟合换号。

## 6. 五个必须写明的边界

### 6.1 Windows 上 mode 比较是**跳过**，不是通过

`core.filemode=false` + Windows 的 `st_mode` 没有 exec 位 ⇒ 重建后读到的 mode
恒为 `100644`。报告里写 `mode_check="skipped_windows"` 并附原因，**不拿"两边都读到
644"冒充验过**。三层摘要不含 mode，所以这条不影响身份结论；但"bundle 能恢复可执行位"
这句话在本机**没有证据**。

### 6.2 patch 只覆盖声明集

`git diff -- <声明路径>`。理由：bundle 声称的范围就是声明集；把无关路径卷进来，
`git apply` 会因为 bundle 之外的改动而失败，也会让"恢复工作树"悄悄带上没声明的代码。
**代价**：这份 bundle 不是"整棵树"的快照，它只承诺声明集。要整棵树得另立契约。

### 6.3 完整性 ≠ 真实性

`tree_digest` 挡**意外损坏**（改一个字符、编码转换、手工编辑、另一个版本的打包器），
挡不住**蓄意伪造**：能改内容的人也能重算摘要。真实的"这份 bundle 是谁打的"要靠
外部锚点（签名、远端记录），本轮不做，也不假装做了。

### 6.4 `exec_identity.py` 的 `reference_only` 分类仍待裁

探针把它按 `reference_only` 声明（沿用上一轮），**这仍是待裁项**，本轮没有顺手决定。
它在 `source_manifest.json` 里根本不在声明集内（那份清单描述的是拟合代码），
所以这里的用法只影响探针自己的声明集。

### 6.5 采样参数来源（③）一点没动

`sampler_entry` 那里"五项属性可读"**不等于**"执行条件已准确记录"：显式传入、
默认值、委托入口、实际生效值四者还没分开。② 完成不代表 ③ 有任何进展。

## 7. 复现

```bash
# 反例矩阵（25 格，约 12 s）
D:/autoeis_env/Scripts/python.exe test_patch_bundle.py

# 纯函数自检（22 项）
D:/autoeis_env/Scripts/python.exe patch_bundle.py

# 四个真实场景（约 25 s，写 out/pb/report.json）
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-14/patch_bundle_probe.py

# 单次采集 / 校验 / 重建
P=patch_bundle_io.py
D:/autoeis_env/Scripts/python.exe $P capture --root . --profile B --out out/pb/b.json
D:/autoeis_env/Scripts/python.exe $P verify  --bundle out/pb/b.json
D:/autoeis_env/Scripts/python.exe $P rebuild --bundle out/pb/b.json --dest out/pb/rebuilt
```

`out/` 不入库；`out/pb/report.json` 是本轮的一手产物，重跑脚本可复现（脏克隆是
临时目录，每次运行时现建现删）。

## 8. 一手数据清单

| 文件 | 内容 |
|---|---|
| `out/pb/report.json` | A/B/C/D 四场景的完整报告（含每场景 `tree_digest`、计数、三层摘要、patch 清单） |
| `out/pb/b.json` | 干净工作树 B 声明集的 bundle（6 461 B，19 全 `base`） |
| `out/pb/rebuilt/` | A 场景的重建目录（398 个文件，与工作树逐个 sha256 相同） |

## 9. 与契约的接口

- 身份层不动：`identity_contract_version` 仍 `1.4`，`exec_identity.py` 一行没改；
- `patch_bundle_io` **import** `exec_identity`（算三层摘要），反向不成立，
  X1 格守着；
- 采集侧（`capture_contract`）与打包侧（`patch_bundle`）共用同一条路径规则，
  P10 格守着"不是第二套规则"；
- bundle 里的 `declaration` 就是 `source_manifest.json` 的 profile + harness 项，
  `counts.declared` 与 `n_files` 必须相等（`declaration_from_manifest` 会报错）。
