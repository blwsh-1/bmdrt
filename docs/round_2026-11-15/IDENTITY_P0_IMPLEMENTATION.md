# 身份链 P0 修复实现报告（`IDENTITY_CONTRACT_VERSION` 1.4 → 1.5）

> **审查基线**：`17cdfc51d6a0278818a8b5e179981cd4580235ff`（第 4 步复裁报告 `IDENTITY_CLOSURE_REVIEW.md` 已推送）。
> **授权范围**：用户「最后实质性准备阶段」的**关口 1 = 身份链 P0 修复**，只处理 P0×4（G1–G4）
> 与用户点名的版本升级（G5）。**不含**关口 2 独立准入复核、关口 3 最小 runner。
> **本轮不宣布准入通过。**

---

## 0. 结论速览

| 项 | 结果 |
|---|---|
| **四项 P0** | 全部落地到代码/测试，不是文档结论 |
| **反例通过标准** | `test_identity_p0.py` **27/27**；每条错误记录都断言 `unconfirmed` 且 `run_id`、`workflow_id` **同时为 `None`** |
| **版本升级** | `IDENTITY_CONTRACT_VERSION 1.4→1.5`；`RUN_ID_CONFIG_KEYS_VERSION 1→2`；前缀 `R1-/D1-/W1-` → `R2-/D2-/W2-`；新世代映射 |
| **红线** | `upstream_origin_captured=False` 保持不变；**没有任何生产路径能把 `unavailable` 升级成 `recorded`**；不用推断来源补历史缺口 |
| **真实链路** | 真实 `capture_at_construction()` 事件 → 六键适配 → `budget_params.validate_budget_params()` → `exec_identity.evaluate()` **`unconfirmed` 零号**（来源未闭合时的正确结果） |
| **runner** | **未启动**；未跑批量拟合；未构造任何可落盘的号 |
| **准入** | ⛔ **仍停在「待独立准入复核」**，四条闭合线由关口 2 判定，本轮不自行裁决 |
| **历史冻结** | `8015721` 三个被审文件 blob 逐位 `SAME`；`budget_chain_probe_review.py` **51/51 FAIL 0**；三处旧挂账不动 |

---

## 1. 授权边界与实现纪律

用户授权原文要点：① 只处理 P0×4，不顺手扩展其他功能；② 不能只改文档，必须实现
`IDENTITY_CONTRACT_VERSION 1.4→1.5` 的完整升级（**不能只升 `RUN_ID_CONFIG_KEYS_VERSION`**）；
③ 完成后立即做一次独立准入复核，重点检查真实生产记录能否从捕获走到身份校验；
④ 准入通过后、再单独授权最小 runner；⑤ 红线：不能把 `unavailable` 改成 `recorded`，
不能用推断来源填历史缺口。

据此，本轮的**实现约束**：

* **不动 P1**：不登记 `chain_method`（G6/F5 保持 `True`）、不改 `progress` 命名（G8）、
  不统一契约三处 `retry_kw` 口径（G7）、不动 G9 声明集缺口的缓解方案、
  不裁 C4/C6/C7/C8 条款正文。
* **不动三处旧挂账**：`bm_real.py:1172` 读 `kw`；`tree_depth_stats(mc, kw[...])` 用声明值；
  `fit_top` 缺 `extra_fields` 入口。
* **不改两处生产构造点的数值逻辑**：`bm_core.py` / `bm_top.py` 本轮**零改动**；
  适配接入放在捕获层 `_event()`，只往事件上追加三个键，不读环境、不改实参、不碰拟合。
* **`budget_params.py` 字节不变**：它被 `budget_chain_probe_review.py` A 组按 `8015721`
  blob 锁定；且多个既有 AST 守护禁止顶层 `.py` / `exec_identity.py` import 产出层。
  因此 v1.5 的闭集词表与指纹算法在身份层/适配层**本地复刻**，由测试逐项交叉核验。

---

## 2. 四项 P0 的实现与验收

### 2.1 G1 —— `recorded` 成为身份层可校验的结构化证据门槛（C1）

| 位置 | 改动 |
|---|---|
| `exec_identity.py` | 新增 `BUDGET_CELL_KEYS` / `BUDGET_SOURCE_STATES` / `BUDGET_ORIGINS` / `BUDGET_EVIDENCE_FIELD` / `BUDGET_EVIDENCE_KEYS`；新增 `budget_evidence_sha256()`、`_chk_budget_evidence()`；`_chk_budget()` 从字符串自检重写为结构化门槛 |
| `test_identity_p0.py` | G1 正例 + 6 条反例 |

`_chk_budget(bud, evidence)` 现在校验：五项恰好 / 每项恰好六键 / `source` 三值闭集 /
`origin` 七值闭集 / `value` 数值且 finite（拒 `bool`）/ `where` 与 `transformed` 形状；
`source="recorded"` 还要求 `origin != unknown` 与 `where` 非空。
外加 `_chk_budget_evidence()`：证据块**恰好五键**、`upstream_origin_captured is True`
（布尔，不许 truthy）、捕获版本/站点非空、`evidence_sha256` 用预算块**重算比对**
（防「验证完再篡改」）、`event_sha256` 合法；并对 `recorded` 追加
`budget_recorded_without_upstream` 复核。

**状态阶梯的自然短路**：`source != "recorded"` 的项只精确点名
`budget_not_recorded:{k}` 后跳过形状/证据细节——它已经 `fatal`、
必然 `unconfirmed` 无号，形状与证据门槛是为「可能发号」的 `recorded` 项设的。
证据块检查在「存在 `recorded` 项」或「显式带了证据块」时执行。
**这不是放宽护栏**：任何非 `recorded` 块仍必然两号 `None`；它同时让第 3 步
历史探针（字节冻结）的 v1.4 投影辅助核验继续如实成立。

第 4 步复裁的原漏洞（裸 `{value, source:"recorded"}` ⇒ `evaluate()` 照发 `R1-` 号）
已由 G1① 反例锁死：现在 ⇒ `unconfirmed`、`run_id=None`、`workflow_id=None`。

### 2.2 G2 —— 生产捕获接口 + 事件→六键适配层（C5 / S2 / S6）

| 位置 | 改动 |
|---|---|
| 新增 `budget_capture_adapter.py` | 纯函数适配层，零生产 import，内置自检 15 项 |
| `sampler_capture.py` | `_event()` 追加 `budget_params` / `budget_evidence` / `budget_observations` / `budget_adaptation_error` 四个键 |
| `test_identity_p0.py` | G2 7 格，其中 ①② 走**真实 MCMC 构造边界捕获事件** |

事件 11 键 ≠ 六键、`SP.ORIGINS`(5) ∩ `BUDGET_ORIGINS`(7) = ∅、事件 `resolved.origin`
是按路径写死的字面量 `derived`/`default`、事件级 `upstream_origin_captured=False`。
适配层因此是**拒绝式**的：

* `ORIGIN_MAPPING = {}`（C5：不许静默把 `default` 挑成 `function_default`）；
* 五项一律 `source="unavailable"` / `value=None` / `origin="unknown"`；
* 观测数值只进 `observations` 明账块，**不进任何身份载荷**；
* `adapt_capture_event(..., upstream_claim=...)` 与外部自称闭合的两种上调入口都 `raise`；
* 适配失败不吞：单独写 `budget_adaptation_error`，让准入复查看得见。

**真实链路实测**（`test_identity_p0.py` G2①，NUTS 单链、8/8 warmup/samples）：

```
capture_at_construction(site="bm_core.fit")
  → ev["budget_adaptation_error"] is None
  → ev["budget_params"] 五项六键，B.validate_budget_params(..., upstream_origin_captured=False) 通过
  → ev["budget_evidence"]["evidence_sha256"] == BA.budget_evidence_sha256(clean)
  → 喂 E.evaluate() ⇒ unconfirmed，run_id=None，workflow_id=None
```

最后一步是**正确结果，不是失败**：`S1`（`bm_real.KW` 可变无留痕）/`S2`（来源词表不相交）
/`S6`（上游未闭合）仍未解决，五项 `source` 最高只能 `unavailable`。
**本轮没有构造任何可落盘的合规 `recorded` 记录**，也没有为通过校验而上调标志。

### 2.3 G3 —— 消除 `retry_kw` 二次承载（C3）

| 位置 | 改动 |
|---|---|
| `exec_identity.py` | `workflow_payload()` 移除 `retry_kw` 整块；新增 `_chk_retry_kw()` 并在 `evaluate()` 中归入 `fatal`；新增 `config_keys_version` |
| `test_identity_p0.py` | G3 3 格 |

生产曾把 `BUDGETS` 五数值整包传进 `workflow.retry_kw`（`bm_select.py`），v1.4 又把它
整块塞进 `workflow_id` ⇒ 改预算换父号。v1.5 **两处都做**：载荷里移除整块；
算号前 `retry_kw` 含五键 ⇒ `retry_kw_budget_keys:...` ⇒ `unconfirmed` 两号 `None`。
重跑策略由 `retry` / `children` 表达，每次实际预算由该次子 `run_id` 承担。
`budget_params.check_retry_kw()` 之前**不在算号路径**上，本轮把护栏放进身份层本身。

### 2.4 G4 —— 重跑身份独立（C3）

| 位置 | 改动 |
|---|---|
| `exec_identity.py` | `RERUN_FIELD` / `RERUN_KEYS` / `_rerun_attempt()`；`run_id_payload()` 新增 `rerun_attempt`；新增 `make_rerun_record()` |
| `test_identity_p0.py` | G4 4 格 |

`rerun.attempt` 是**条件**不是结果，进 `run_id` 载荷 ⇒ 同一组预算重跑两次也必然不同号；
`rerun.reason` / `parent_workflow_ref` 只作记录不进哈希。
`make_rerun_record()` 只搬身份条件（候选/数据/rng/config/code/env/两个版本字段），
换掉本次 `budget_params` + `budget_evidence`，丢掉 `instance` / `provenance` /
`kk_status` / 父 `retry_kw` 预算 / `children`；父 `workflow.children` 只放子号引用。
换预算时 `rng.plan.n_chains` 由调用方用 `rng=` 显式给，函数不猜、不自动折算。

---

## 3. 版本升级 1.4 → 1.5 的完整清单（G5 / C2）

第 4 步复裁取证：`RUN_ID_CONFIG_KEYS_VERSION = 1` **既不进两个载荷也不被任何校验读取**，
单升它等于没升。本轮按报告 §6.2 做完整升级：

| # | 动作 | 实现位置 |
|---|---|---|
| ① | `IDENTITY_CONTRACT_VERSION "1.4" → "1.5"` | `exec_identity.py` 常量 |
| ② | `RUN_ID_CONFIG_KEYS_VERSION 1 → 2` | 同上 |
| ③ | 版本写进**记录**：新增 `run_id_config_keys_version` 字段 | `_require_version()` 校验整数、非 bool、等于 2；缺/旧值 ⇒ `unhashable:` ⇒ 无号 |
| ④ | 版本写进**两个载荷**：`config_keys_version` | `run_id_payload()` / `workflow_payload()` |
| ⑤ | 新世代：`R2-` / `D2-` / `W2-` | `RUN_ID_PREFIX` / `WORKFLOW_PREFIX` |
| ⑥ | 新世代映射与跨世代比较 | `IDENTITY_GENERATIONS`（6 前缀，`registry_0.7.0` / `exec_identity_1.4` / `exec_identity_1.5`）、`id_generation()` / `can_compare_ids()` |
| ⑦ | 载荷键集变化 ⇒ 全换号 | 新增 `config_keys_version`、`rerun_attempt`；移除 `retry_kw` |

> `budget_params.IDENTITY_GENERATIONS` 是第 2 步的 **v1.4 投影**，其文件字节被
> `8015721` 审查脚本按 blob 锁定，本轮**不改**；v1.5 的世代判定以
> `exec_identity.id_generation()` 为权威。两套表各自服务自己的世代，已明账。
>
> **全换号是预期代价**：所有 v1.4 的 `R1-/D1-/W1-` 号整体失效，
> 不用「兼容旧号」绕过。`R2-` 与 `R1-` 跨世代 `comparable=False`、
> `declares_equivalence` 恒 `False`。

---

## 4. 改动面

| 文件 | 性质 | 说明 |
|---|---|---|
| `exec_identity.py` | 修改 | v1.5 常量/世代/词表、`_require_version`、`_rerun_attempt`、两个载荷、`_chk_budget*` / `_chk_retry_kw`、`evaluate`、`make_rerun_record`、三个公开纯函数 |
| `sampler_capture.py` | 修改 | 新增 `import budget_capture_adapter`；`_event()` 追加四键（+18 行） |
| `budget_capture_adapter.py` | **新增** | 拒绝式适配层 + 15 项自检 |
| `test_identity_p0.py` | **新增** | 27 格 P0 反例（G1 正例+6 / G2 7 / G3 3 / G4 4 / G5 6） |
| `test_exec_identity.py` | 修改 | `_rec()` 升级为 v1.5 合规记录（六键 + `budget_evidence` + 版本字段）；前缀同步 `R2-/D2-`；T15 换块 |
| `test_budget_params.py` | 修改 | C4 / G2 交叉核验适配新门槛，并补「缺证据块必须拒」反例 |
| `source_manifest.json` | 重生成 | `identity_contract_version "1.5"`；AST 闭包 +1 模块 ⇒ A 17→**18** / B 21→**22** / C 20→**21**；`--check rc=0` |

**零改动**：`budget_params.py`、`budget_chain_probe.py`、`BUDGET_CHAIN_PROBE_REPORT.md`
（三个 `8015721` blob 逐位 `SAME`）、`bm_core.py`、`bm_top.py`、`bm_real.py`、
`bm_select.py`、`nested_fit.py`、`config.json`、三处旧挂账。

---

## 5. 全套回归

| 项目 | 结果 |
|---|---|
| `test_exec_identity.py` | **68/68** |
| `test_budget_params.py` | **45/45** |
| `test_capture_contract.py` | **31/31** |
| `test_patch_bundle.py` | **25/25** |
| `test_sampler_provenance.py` | **45/45** |
| `test_sampler_capture.py` | **28/28** |
| `test_identity_p0.py`（新增） | **27/27** |
| `budget_capture_adapter.py` 自检 | 15/15 |
| `budget_params.py` / `sampler_provenance.py` / `sampler_capture.py` 自检 | 通过 |
| `source_manifest.json --check` | `rc=0`，与源码一致 |
| `test_stage0_freeze.py` | **9/9** |
| `map_stage0_records.py --check` | `rc=0`（239 条 / 独立 239 / 严格 237 / 唯一 237） |
| `budget_source_probe.py` | `rc=0` |
| `budget_chain_probe.py` | **`rc=0`，全部字段对得上**（8/8 case + 辅助核验） |
| `capture_probe.py` | `rc=0`，全部字段对得上 |
| `budget_chain_probe_review.py` | **51 条机械断言，FAIL 0**（需系统 Python ≥3.12） |
| `git diff --check` | `rc=0`；7 个改动文件全部 `CRLF=0` |

> 拟合/冻结/探针用 `D:/autoeis_env/Scripts/python.exe`；审查脚本因 f-string 反斜杠
> 需系统 Python ≥3.12。**未跑拟合、未跑批量、未写产物、未启动 runner。**

---

## 6. 未闭合 / 明账（关口 2 的输入，本轮不裁决）

1. **来源证据线仍未闭合**：S1（`bm_real.KW` 可变无留痕）、S2（词表交集 ∅、
   事件 11 键 ≠ 六键）、S3（`sampling` 声明值与构造实参分叉，C8 已证伪）、
   S5（K3 `m35_cond_n2.py:116` 未纳入）、S6（`upstream_origin_captured=False`）。
   本轮的适配层**如实产出 `unavailable`**，没有把任何推断值升级成 `recorded`。
   **真实运行经 `evaluate()` 仍 `unconfirmed` 且两号 `None`。**
2. **P1×5 / P2×1 未处理**：G6（`chain_method` callable 策略，F5 仍为 `True`）、
   G7（三处 `retry_kw` 契约口径矛盾）、G8（`progress_bar` vs `progress`）、
   G9（身份层/产出层不在声明集）、G10（K3 与其余未覆盖入口）。
3. **契约条款正文 C1–C8 尚未同步订正**：本轮是**实现 + 反例**，
   C1/C2/C3/C5 已由代码落地，但 `EXECUTION_IDENTITY_CONTRACT.md` /
   `BUDGET_PARAMS_CONTRACT.md` / `SAMPLER_CAPTURE_CONTRACT.md` 的旧条款正文仍写
   v1.4 口径；C4/C6/C7/C8 按裁决未动。按用户「不顺手扩展其他功能」的边界，
   正文订正留到关口 2 一并收口，**不隐瞒这处不一致**。
4. **补丁重建「完整性 ≠ 真实性」**、Windows mode `skipped`、`reference_only` 待裁；
   三时点快照、harness 归属（C-1）、多次采样逐次事件（C-3）仍需在**真实 runner** 里验证，
   不能由夹具推定。
5. **结论边界**：所有实测限定在 C1–C8 覆盖的两处生产构造点、五条转发路径与所列测试工况；
   K3 独立构造路径及其余入口未取证，**不得外推**。

---

## 7. 关口 2「独立准入复核」应检查什么

按第 4 步报告 §7 的四条闭合线，逐项**重新取证、不沿用本报告结论**：

| 闭合线 | 关口 2 必须核的事 |
|---|---|
| 来源证据 | 真实生产记录能否从 `capture_at_construction()` 一路走到 `budget_evidence` 落盘再进 `evaluate()`；`upstream_origin_captured` 在什么条件下才允许从 `False` 变 `True`；有无任何路径能产出合规 `recorded` |
| 身份投影 | 新 v1.5 载荷（含 `config_keys_version` / `rerun_attempt`、移除 `retry_kw`）与产出层投影是否仍逐位对齐；结构化门槛是否与 `budget_params.validate_budget_params()` 完全同门槛 |
| 运行时采集 | 三时点、harness 归属、多次采样逐次事件、失败与重跑的独立身份，在真实执行里是否成立 |
| 补丁重建 | 新增 `budget_capture_adapter.py` 进声明集后，`verify`/`rebuild` 是否仍逐位相同；「完整性≠真实性」边界是否仍成立 |

另外须核：契约正文 C1–C8 订正、P1 缺口的裁决状态、
`IDENTITY_GENERATIONS` 两套表（v1.4 投影 vs v1.5 权威）的分工是否会造成误比号。

---

## 8. 本轮结论

> ### 关口 1（身份链 P0 修复 + 1.4→1.5 完整升级）：**实现完成，反例通过。**
>
> 四项 P0 都有反例测试，错误记录一律 `unconfirmed` 且 `run_id` / `workflow_id`
> 同时为 `None`。`IDENTITY_CONTRACT_VERSION` 已从 1.4 升到 1.5，版本字段写进
> 记录、两个载荷与校验，新世代映射已建立，所有 v1.4 号按预期整体失效。
>
> ### ⛔ **本轮不宣布准入通过，也不启动最小 runner。**
>
> 理由与第 4 步复裁一致：来源证据线仍未闭合，真实运行的正确结果仍是
> `unconfirmed` 无号。**Runner 的准入复核（关口 2）与最小 runner（关口 3）
> 各自仍需单独授权。** `upstream_origin_captured` 保持 `False`，
> 五项 `source` 最高 `unavailable`；三处旧挂账不动；执行身份链不整体冻结。
