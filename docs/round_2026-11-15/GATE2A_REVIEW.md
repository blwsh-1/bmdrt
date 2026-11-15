# Gate 2A 独立审查报告：Gate 1（P0 身份契约修复）实现复裁

> **审查基线**：`origin/main` = `HEAD` = `7e915f1b176fe5f6d26a5f3e348d32a242f10eb9`。
> `git ls-remote` 复核 `7e915f1` 之后**无新提交**，工作树干净。
> **授权**：用户授权启动 Gate 2A 只读独立审查；Gate 2B 范围裁定为「收窄到一条路径」。
> **本轮边界**：只读。**不改任何实现代码 / 测试**、不跑生产拟合、不写 Runner、
> 不宣布准入、不启动 Gate 2B 修复。

---

## 0. 结论速览

> ### ⛔ Gate 2A 判定：**FAIL**（不满足用户给定的通过标准）。
>
> Gate 1 的**主路径确实成立**：五项 P0 在纯函数层与捕获层都能复现，形状/类型/
> 来源/证据类不合规数据确实发不出号。但通过标准的后三条**不成立**：
> ① **引入了新的身份重复**（F-08）；② 存在**可绕过的二次承载**（F-06/F-07）
> 与**捕获→落盘断链**（F-02）；③ 存在**兼容性分歧**（F-10/F-12/F-13）。
>
> 更根本的是：方案 2.1 问「一个没有真实执行来源证据的预算对象，能否伪装成
> `recorded` 并获得 `R2-` 身份？必须证明不能」——**独立实测的答案是可以**。
> 这不属于 Gate 1 的实现缺陷，而是**门槛类型的边界**：纯函数层只能校结构与
> 自洽，无法证伪「是否真的捕获过」，须移交 Gate 2B 由可信边界闭合。
>
> **本报告不修改代码。** 所有「需修改」项列入 §5，等用户单独授权后才动。

| 分类 | 数量 | 处置 |
|---|---|---|
| 已闭环 | 5 组主路径 | 记录为 Gate 1 成立的部分 |
| 未闭环（结构性） | 2（F-01/F-02） | 移交 **Gate 2B** |
| 需修改（具体缺陷） | 7（F-03–F-09） | 方案第五步「修复后重新审查」 |
| 需裁决 | 4（F-10–F-13） | 用户裁口径，不改代码 |
| 无法证明（本轮范围外） | 3 类 | 见 §7 |

---

## 1. 审查基线与方法

### 1.1 被审对象（`git ls-tree 7e915f1`）

| 文件 | blob 前 12 |
|---|---|
| `exec_identity.py` | `0b66c917481e` |
| `budget_capture_adapter.py` | `d6a216322920` |
| `sampler_capture.py` | `6ce90eedb547` |
| `budget_params.py` | `dc6ae77f3330` |
| `sampler_provenance.py` | `182b75b8e283` |
| `test_identity_p0.py` | `85f6a28ad41c` |
| `source_manifest.json` | `6b8e4bd3c109` |

### 1.2 独立性口径

用户指出「完成状态依据实现报告，不代表已独立读取或运行最新代码」。本轮据此：

* **重新读取实际代码并独立设计反例**，**不引用 `test_identity_p0.py` 的任何 CASES**；
  合法基线夹具只借 `test_exec_identity._rec()`（68 格既有夹具）。
* G1 的 `evidence_sha256` 由**审查侧独立实现**（`hashlib` + `json.dumps`，不调用
  `E.budget_evidence_sha256`）重算，用于模拟「攻击者自己也能算出合规指纹」。
* G2 换一个 Gate 1 反例未覆盖的**真实工况**：NUTS / 3 链 1 设备 / warmup 12 /
  samples 24 / ta 0.85 / depth 7（Gate 1 用的是单链 8/8）。
* G5 用 `git show 17cdfc5:exec_identity.py` **载入 v1.4 实现为独立模块**，
  同一份条件分别算两个世代的摘要，验证「全换号」不是自述。
* 核验全部为**内联只读执行、未落盘脚本**；`8015721` 三个被锁 blob 逐位 `SAME`。

> **审查脚本自身的两处笔误已修正重跑**，并在下文明确区分：
> ① G1 首轮的独立指纹实现对 `nan`/`inf`/numpy/Decimal 抛异常，属审查工具限制，
> 已改用身份层公开函数 + 哨兵重跑（§2.1 B 组结论以补跑为准）；
> ② G3/G4 首轮 `ids()` 解包笔误、G5 首轮字符串引号笔误与
> 「构造 keys_version 反例时误删 identity_contract_version」，均已校正后重跑。
> **这些不是生产代码缺陷，不计入差距表。**

### 1.3 与方案的三处术语澄清（上一轮已提出，此处固化）

| 方案表述 | 实测 | 本轮按什么审 |
|---|---|---|
| 2.1「六键结构：六项预算参数」 | `BUDGET_KEYS` = **5** 项；`CELL_KEYS` = **6** 键 | **五项参数 × 每项六键** |
| 2.1「`budget_evidence` 覆盖五个实际预算值」 | `EVIDENCE_KEYS` 是五键元数据，**不存五个值** | 核 `evidence_sha256` 是否对五项×六键**全部单元格**重算 |
| 2.4 第 3 点「重试次数影响 `workflow_id`」 | `workflow_payload` 有 `retry`；`run_id_payload` 有 `rerun_attempt` | **次数/策略 → workflow_id；第几次执行 → run_id** |

---

## 2. 逐项裁决（G1–G5）

### 2.1 G1 预算证据门禁 —— 主路径 ✅，但「必须证明不能」❌

| 检查点 | 独立实测 | 判定 |
|---|---|---|
| 六键结构 | 五项×六键；cell 少键/多键/裸块/顶层多未知键 ⇒ 全 `unconfirmed`、两号 `None` | ✅ |
| 参数类型 | `bool` / `str` / `nan` / `inf` / `np.float32` / `np.int64` / `Decimal` / `None` / `list` 全拒，原因码 `budget_value_invalid` | ✅（值域见 F-03） |
| 来源字段 | `"Recorded"` / `"recorded "` / `1` / `True` / `derived` / `unavailable` / `derived_from_tier` 全拒 | ✅ |
| 证据完整性 | 指纹重算覆盖五项×六键；改 `requested` 不更新 sha ⇒ `budget_evidence_sha_mismatch` 拒；同步更新 sha ⇒ **同号**（追溯字段不进 `run_id`）；改 `value` ⇒ 换号 | ✅ |
| 防篡改 | `E.budget_evidence_sha256` 与审查侧独立实现逐位相等；验证完再改块被抓 | ✅ |
| 上游证明 | `upstream_origin_captured` 为 `1` / `"True"` / `None` / `[]` 全拒；证据块缺键/多键、`capture_version=None`、`capture_site=""`、`event_sha` 长度错误全拒 | ✅（校验深度见 F-04/F-05） |
| 状态阶梯短路 | 全 derived / 全 unavailable / 混来源 / 非法值 `object()` 均未引入可发号路径，`evaluate()` 不抛未捕获异常 | ✅ |
| 第 4 步原漏洞 | 裸 `{value, source:"recorded"}` ⇒ `unconfirmed`、两号 `None`（`budget_cell_shape:*`） | ✅ |

**方案 2.1 最关键问题的实测答案**：

```
A0 合法基线                              → recorded, R2-673424b2…
A1 审查侧自算 evidence_sha256 + 自填 upstream=True → recorded, 拿到 R2- 号
A2 同上，capture_site 改为 'forged.site'  → recorded, 拿到 R2- 号
A3 同上，event_sha256 任意伪造            → recorded, 拿到 R2- 号
```

⇒ **不能证明「无真实来源证据的对象无法伪装成 recorded」**。身份层校验的是
**结构 + 自洽 + 显式布尔**，这三者都可被计算出来；`upstream_origin_captured=True`
是记录里的一个自述字段，不是可验证的上游证明。Gate 1 报告 §2.1 已明账
「这不是『真的捕获过』的密码学证明」，本轮独立复现证实该边界真实存在。
**这不是靠收紧字段能解决的**（见 F-01），须由 Gate 2B 的可信边界闭合。

### 2.2 G2 生产捕获链路 —— 拒绝式适配 ✅，落盘组装 ❌

真实工况逐项（3 链 1 设备，`aggregate.weakest = mismatch`）：

| 参数 | resolved | observed | `origin` 字面量 | status | 适配输出 |
|---|---|---|---|---|---|
| `n_warmup` | 12 | 12 | `derived` | complete | `unavailable` / `value=None` |
| `n_samples` | 24 | 24 | `derived` | complete | 同上 |
| `num_chains` | 3 | 3 | `derived` | complete | 同上 |
| `target_accept` | 0.85 | 0.85 | `derived` | complete | 同上 |
| `max_tree_depth` | 7 | 7 | `derived` | complete | 同上 |

* **「捕获到值 ≠ 捕获到来源」守住了**：五项 `resolved.origin` 全是按代码路径写死的
  `derived`，`status=complete` 只表示「构造实参→对象属性」闭合；适配层逐项输出
  `unavailable`/`None`/`unknown`，观测值只进 `budget_observations` 明账。
* **伪造攻击无效**：把事件里 `origin` 全改成 `cli_explicit`、`status` 全改 `complete`，
  适配输出仍是 `{'unavailable'}`、`upstream=False`。事件自称 `upstream=True`、
  外部 `upstream_claim=True`、`judged=False` 三种入口全部 `raise`。
* **产出层逐项校验**：`B.validate_budget_params(bp, upstream_origin_captured=False)`
  通过，`per_key` 五项全 `unavailable`，`state=unavailable`，指纹与适配层一致。
* **身份层**：审查侧独立组装记录喂 `E.evaluate()` ⇒ `unconfirmed`，两号 `None`。
  适配失败（`budget_params=None`）时同样 `unconfirmed` 两号 `None`。
* **注意**：`aggregate.weakest=mismatch` 未被适配层用作证据输入（安全方向，
  不会升级），但这条应明账为「聚合状态不是证据来源」。

**断链（F-02）**：`Grep` 精确核对生产模块只有三处 `provenance` 引用——
`bm_core.py:616` / `bm_top.py:1013` 把事件挂到对象，`bm_top.py:1081` 把事件原样
塞进返回字典。**没有任何模块**把 `ev["budget_params"]` / `ev["budget_evidence"]`
搬进 `exec_identity` 的身份记录。所以「真实生产记录从捕获走到身份校验」目前
**只由审查侧手工组装验证**，生产链路尚不存在（Runner 未授权）。

### 2.3 G3 重试参数 —— 主路径 ✅，旁路 ❌

| 反例 | 结果 | 判定 |
|---|---|---|
| `retry_kw` 含单个预算键 `n_warmup` | `retry_kw_budget_keys:n_warmup` ⇒ 无号 | ✅ |
| `retry_kw` 含五键 | fatal ⇒ 两号 `None` | ✅ |
| 相近名 `n_warmup_extra` / 纯策略键 / 空 dict | 均发号且 `workflow_id` 与基线**相同** | ✅ |
| `workflow_payload` 顶层 | 已无 `retry_kw` | ✅ |

**旁路（F-06）**：`workflow.retry` / `guards` / `extras` / `search` / `pair_mode`
走 `_plain` 整块进 `workflow_id`，塞 `n_samples=1000` **会改号且载荷含 1000**。
`gate` 是三键白名单投影，不受影响。⇒ 「预算参数只在规定字段出现」**未彻底成立**，
G3 只堵了 `retry_kw` 一条路。

**children（F-07）**：纯引用字符串、字符串 `'n_samples=1000'`、单元素 dict 含预算
都能进 `workflow_id`；混合 dict+str 时 `sorted()` 抛 `TypeError` ⇒ `unhashable` 无号。
⇒ 「children 只存引用」是**约定，不是校验**。

> 澄清一处误报：`exec_identity.py` 含字符串 `check_retry_kw`，命中于
> `_chk_retry_kw()` 的 docstring（说明「`budget_params.check_retry_kw()` 之前不在
> 算号路径」），**不是调用**；`B.check_retry_kw()` 仍存在但不在身份层算号路径。

### 2.4 G4 重跑身份 —— 主路径 ✅，两处不符合通过标准 ❌

✅ 同基础条件 attempt 1/2 ⇒ 不同 `run_id`，两个载荷**只差 `rerun_attempt`**；
换预算 ⇒ 换号；子记录不含 `instance` / 父 `retry_kw` / `children`；
`retry=0/1/2` ⇒ 三个不同 `workflow_id`。

❌ **F-08 身份重复**：两个父工作流（`retry=1` vs `retry=3`，`workflow_id` 不同）、
同基础条件、同 `attempt=1` ⇒ 两条子记录的 `run_id` **完全相同**
（`R2-6f3b96a24de04344e…`）。`rerun.parent_workflow_ref` 不进载荷，
`make_rerun_record()` 又把子 `workflow` 重置为固定值 ⇒ **不同 lineage 无法区分**。
方案 2A 通过标准明确「没有引入新的身份重复」，此项**不成立**。

❌ **F-09 失败/成功重试不可区分**：`rerun.reason` 不进哈希，
`reason='sampler_error'` 与 `'acf_gate_retry'` 在同 attempt、同预算下 **同号**。
方案 2.4 第 5 点未满足。

### 2.5 G5 版本升级 —— 机制 ✅，兼容性 ❌

✅ 两个载荷都含 `v='1.5'` 与 `config_keys_version=2`，记录也显式携带两者。
`_require_version()` **真校验**：`identity_contract_version` 缺/1.4/1.6 全拒；
`run_id_config_keys_version` 缺/1/`"2"`/`True`/`2.0`/`None`/`3` 全拒，
只有整数 `2` 放行（此项已隔离重跑，见 §1.2）。
前缀只是辅助标识：改 `RUN_ID_PREFIX` 后 `run_id_digest` 逐位不变，
只有发出的号前缀变。

✅ **全换号独立验证**：把 `17cdfc5:exec_identity.py` 载入为独立模块，同一份条件
分别算摘要——

```
v1.4 run_id     digest = 5f9727fb58cf390c71b7f82b…
v1.5 run_id     digest = 673424b209443a59bb4ad8b4…   （不同）
workflow_id digest 亦不同
```

`v1.4` 记录喂 v1.5 `evaluate()` ⇒ `unconfirmed` 两号 `None`
（`unhashable:identity_contract_version …`），旧记录按旧契约解释，未用新规则重算。
`keys_version 2→3` 且记录同步 ⇒ 换号；记录不同步 ⇒ 两号 `None`。

❌ **F-10 两套世代表分歧**：

```
E.IDENTITY_GENERATIONS = 六前缀（含 R2-/D2-）
B.IDENTITY_GENERATIONS = 四前缀（R-/D-/R1-/D1-，被 8015721 blob dc6ae77f3330 锁定）
E.id_generation("R2-…")            = exec_identity_1.5
B.id_generation("R2-…")            = None
E.can_compare_ids(两新号)           = comparable=True
B.can_compare_ids(两新号)           = comparable=False
```

结果上「新号不与旧号混比」不算错，但语义是「未识别世代」，且**两个 v1.5 号
在 legacy 层也判不可比**。方案 2.5「避免部分代码仍按 1.4 规则运行」命中此项。

❌ **F-11 键表版本语义**：`RUN_ID_CONFIG_KEYS` 仍是 **38 键**、
`chain_method` 未登记，与 `17cdfc5` 键集**逐位相同**。
`RUN_ID_CONFIG_KEYS_VERSION 1→2` **不对应任何键集变化**——需裁它表示
「载荷语义代际」还是「键表代际」，并与契约「加键/改 effect 必须 +1」对齐。

❌ **F-12 前缀与世代表耦合**：`id_generation()` 只靠前缀判世代，
落盘号字符串本身**不携带契约版本**；契约 D-4「前缀只作辅助标识」在落盘号上
并不严格成立。

❌ **F-13 残留 1.4 标签**：`budget_params.py` 39 处、`test_budget_params.py`
29 处、`BUDGET_PARAMS_CONTRACT.md` 17 处含 `1.4`/`v14`/`R1-`
（含 `budget_evidence_state.v14_consequence` / `v14_note` 字段名）。
均为文档或标签、非算号路径，但需裁是否随 1.5 统一。

---

## 3. 差距表（四级）

| 编号 | 级别 | 发现 | 依据 | 处置 |
|---|---|---|---|---|
| **F-01** | 未闭环·结构性 | **完全自洽伪造证据块可拿 `R2-` 号**；方案 2.1「必须证明不能」的答案是「能」 | §2.1 A1–A3 | **移交 Gate 2B**（可信边界），不是改字段能解决 |
| **F-02** | 未闭环·结构性 | **捕获→落盘记录组装断链**；生产侧无人把事件块搬进身份记录 | §2.2 + Grep | **移交 Gate 2B / Gate 3**（Runner 记录组装） |
| **F-03** | 需修改 | `budget_params.*.value` 无值域校验：`n_warmup=-1`、`0` 都能拿号 | §2.1 B 组补跑 | 修复轮 |
| **F-04** | 需修改 | `event_sha256` 只校长度 64，不校 hex：`'z'*64` / 中文 / 空格 / 全 0 都放行 | §2.1 E 组 | 修复轮 |
| **F-05** | 需修改 | `capture_version` / `capture_site` 只校非空、不校闭集：伪造字符串放行 | §2.1 E2 组 | 修复轮 |
| **F-06** | 需修改 | `workflow.retry/guards/extras/search/pair_mode` 用 `_plain` 整块进 `workflow_id`，可二次承载预算且改号 | §2.3 | 修复轮 |
| **F-07** | 需修改 | `children` 无内容约束，字符串/单元素 dict 含预算可进 `workflow_id` | §2.3 | 修复轮 |
| **F-08** | 需修改·**阻断** | **不同 lineage、同基础条件、同 attempt ⇒ 子 `run_id` 撞号** | §2.4 | 修复轮（优先） |
| **F-09** | 需修改 | 失败重试 / 成功重试 `reason` 不进哈希 ⇒ 同号 | §2.4 | 修复轮 |
| **F-10** | 需裁决 | 两套 `IDENTITY_GENERATIONS` 分歧，legacy 层不认识 `R2-` | §2.5 | 用户裁口径 |
| **F-11** | 需裁决 | `keys_version=2` 但键集 38 键逐位未变，版本语义不明 | §2.5 | 用户裁口径 |
| **F-12** | 需裁决 | 前缀与世代表耦合，落盘号不携带契约版本 | §2.5 | 用户裁口径 |
| **F-13** | 需裁决 | 三份文件残留 1.4 标签（非算号路径） | §2.5 | 用户裁口径 |
| **F-14** | 已澄清 | `check_retry_kw` 字符串命中 docstring，非调用 | §2.3 | 无需动作 |
| **F-15** | 已澄清 | 方案三处术语：五项×六键 / 证据块是整体指纹 / retry 与 rerun 分工 | §1.3 | 无需动作 |
| — | 无法证明 | CLI→profile→函数→MCMC 的逐层来源链 | Gate 2B 范围 | §6 |
| — | 无法证明 | 三时点 / harness 归属 / 多次采样逐次事件 / dirty patch 真实性 | 需真实 runner | §7 |
| — | 无法证明 | 239 条历史 Stage 0 记录按新契约的解释 | 保持原证据等级 | §7 |

---

## 4. Gate 2A 通过标准逐条判定

用户给定：**「五项 P0 修改在实际代码路径上成立，负面测试能够证明不合规数据
无法生成有效身份，且没有引入新的身份重复、遗漏或兼容性问题。」**

| 通过条件 | 判定 | 依据 |
|---|---|---|
| 五项 P0 在实际代码路径上成立 | 🔶 **部分** | 纯函数层 + 捕获层成立（§2 五组主路径）；**生产「捕获→落盘」路径不存在**（F-02） |
| 负面测试证明不合规数据无法生成有效身份 | 🔶 **部分** | 形状/类型/来源/证据类不合规确实不发号；**无真实来源但自洽伪造可发号**（F-01） |
| 没有引入新的身份重复 | ❌ | F-08 撞号 |
| 没有遗漏 | ❌ | F-02 断链、F-06 二次承载旁路、F-07 children 无约束、F-03/04/05 校验深度 |
| 没有兼容性问题 | ❌ | F-10 两套世代表、F-12 前缀耦合、F-13 残留标签、F-11 版本语义 |

> ### ⛔ **Gate 2A = FAIL。**
>
> 按用户「不允许把未知状态伪装成通过」的要求，不给 PASS 也不给 CONDITIONAL：
> 2A 是**实现正确性审查**，其通过标准有明确的不满足项。F-01/F-02 属设计边界、
> 应移交 2B，但**不能因此把 2A 判成通过**；F-03–F-09 是 2A 自身发现的实现缺陷，
> 需按方案第五步「修复后重新审查」重新判一次。
>
> 这不否定 Gate 1 的价值：主路径与版本机制确实成立，且全换号已被独立复现。
> 但**在 F-08（撞号）修复并重新审查之前，不得进入 Gate 2C**。

---

## 5. 建议的修复范围（等用户单独授权，本轮不动）

按方案第五步「只针对审查发现的问题修改，不扩展 P1 功能」：

* **最小修复集 = F-03, F-04, F-05, F-06, F-07, F-08, F-09**（7 项），
  优先 F-08（身份重复）。
* F-08 的方向建议（**仅建议，不实施**）：把 `parent_workflow_ref` 或一个
  lineage 标识纳入子 `run_id` 条件；F-09 需先裁「`reason` 是结果还是条件」，
  若属结果则不应进条件身份，而应由工作流状态表达。
* F-06/F-07 的方向建议：给 `workflow_payload` 的整块字段建立**预算键黑名单**
  或改为白名单投影，`children` 校验为「仅身份引用字符串」。
* **明确不含**：F-01/F-02（移交 2B）、F-10–F-13（需裁决）、
  P1×5/P2×1（`chain_method` 登记、`progress` 命名、`retry_kw` 三处契约口径、
  G9 声明集、K3 与其余入口）、契约正文 C1–C8 订正、三处旧挂账。
* 修复轮完成后**必须重新做一次 Gate 2A 性质的复裁**（含本报告全部独立反例），
  再判 2A 是否转为通过。

---

## 6. 移交 Gate 2B 的事项与范围前置条件

用户已裁 **「2B 收窄到一条路径」**。但按当前事实，这「一条路径」**尚无法指定**：
最小 Runner 的冻结候选清单 / 数据集 / 执行入口都还没定（Gate 3 未授权）。
因此 2B 启动前需要一个**单独的前置确认**：

1. 冻结第一版 Runner 的**唯一执行路径**（例如 `bm_real.run → bm_core.fit → MCMC`
   或指定 `bm_top.fit_top`），并明确它的 CLI / 程序化入口。
2. 明确把以下内容列为**不纳入、不外推**：`nested_fit.py` 独立研究接口
   （`bmdrt.py` 未接、默认 voigt）、K3 `m35_cond_n2.py:116`
   （`NON_PRODUCTION_SITES` 明账）、`bm_select.py` 的 retry 路径、
   其余历史实验入口、三处旧挂账。
3. 按方案 3.1 建立五项预算的**逐项来源审计表**，坚持「不能因为最终数值与
   某默认值相同就反推来源」，以**最终构造点**为准
   （C8 已证伪 `fit_one.sampling` 声明 2000 / 构造实参 16）。
4. 3.3 历史记录纪律：239 条 Stage 0 记录保持原证据等级，不重跑补身份、
   不按提交时间或入口 SHA 推断代码版本；只作回归样本，不伪装成新契约闭环样本。

> **红线不变**：不能把 `unavailable` 改成 `recorded`，不能用推断来源填历史缺口。
> F-01 的闭合只能靠「生产捕获接口的唯一产出路径 + 可信边界」，
> 不能靠放宽或收紧身份层字段。

---

## 7. Gate 2C 预判与「无法证明」清单

Gate 2C 的准入表里，**以下项在 Gate 2 阶段（不跑 runner）无法实测**，
只能写「设计具备、待 T1/T2/T3 验证」：

* dirty 状态 / patch bundle 的**真实性**（契约已明账「完整性 ≠ 真实性」）；
* 三时点快照、harness 归属（C-1）、多次采样逐次事件（C-3）；
* 源码声明集与运行时集的**真实**覆盖关系、`numeric_source_tree_sha256` /
  `workflow_source_tree_sha256` 的实际覆盖、`sigma` 规范化的真实运行路径。

⇒ 即便 2A 修复完成、2B 闭合，Gate 2C 最诚实的结果仍应是
**CONDITIONAL（限定路径 + 明列待实测项）**，而不是无条件 PASS。
用户方案允许有明确边界的 CONDITIONAL，本轮审查支持该判断。

**本轮无法证明**（不在只读范围）：CLI→profile→函数→MCMC 逐层来源链；
上述真实运行取证；239 条历史记录按新契约的解释（保持原等级，不补身份）。

---

## 8. 本轮边界与下一步

* **只交付本审查报告 + `docs/INDEX.md` 同步 + 项目日志**；
  **实现代码、测试、清单、契约正文、生产模块一律未改**，`git status` 除文档外为空。
* 未跑生产拟合、未写产物、未构造可落盘号、未启动 Runner、
  未宣布准入通过、未启动 2B 修复。
* 内联核验未落盘脚本；`8015721` 三文件 blob 逐位 `SAME`；
  全套既有回归在基线 `7e915f1` 上仍全绿（本轮不重跑拟合，仅复核其未被改动）。
* **下一步由用户裁决**：① 是否按 §5 单独授权「最小修复集」并重新审查；
  ② F-10–F-13 的口径裁决；③ 2B 前置的唯一执行路径冻结。
  **Gate 2B / 2C / Gate 3 与试运行验收各自仍需单独授权。**
