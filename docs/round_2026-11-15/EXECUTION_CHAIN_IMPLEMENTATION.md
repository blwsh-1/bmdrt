# Gate 2B：F-01 / F-02 实现、单次执行链测试与身份闭环复查

> **基线**：`e19fe4b`（最小修复集 F-03–F-09）
> **授权范围**：修复 F-01、F-02，做真实生产链路测试，**暂不运行完整 EIS 拟合**。
> **本轮性质**：实现 + 测试 + 闭环复查。**⛔ 不宣布准入通过、未启动 runner。**

---

## 0. 结论速览

| 项 | 结论 |
|---|---|
| 唯一执行路径 | 已冻结：`bm_core.fit` 主路径，`bm_top.fit_top` 仅兼容验证 |
| **F-02** | **已闭环**：真实捕获 → 绑定实例 → 组装正式记录 → 原子落盘 → 重读复核，四种情况均有测试 |
| **F-01** | **条件性关闭**：三个必留测试全过、自洽字典不再能拿号；但属「内部可信边界」，**不是抗抵赖密码学证明** |
| 单次执行链测试 | **14/14**，未跑完整 EIS 拟合、未产生新实验拟合结果 |
| 真实执行身份 | 仍 `unconfirmed`、两号 `None` —— **来源未闭合时的正确结果** |
| Gate 2C | **仍不具备无条件准入**，维持 CONDITIONAL 预判；本轮不作准入结论 |
| 生产拟合代码 | `bm_*.py` **零改动**；三处旧挂账不动 |

---

## 1. 冻结的唯一执行路径

`execution_harness.FROZEN_PATH`：

| 项 | 值 |
|---|---|
| 主路径 | `bm_core.fit`（构造点 `bm_core.py:610 MCMC(...)`） |
| 捕获接口 | `sampler_capture.capture_at_construction` |
| 记录格式 | `exec_identity_1.5 + capture_attestation_v1` |
| 兼容验证 | `bm_top.fit_top`（第二站点，仅验证） |
| **不纳入、不外推** | `nested_fit.py` 独立研究接口；K3 `m35_cond_n2.py:116`；`bm_select.py` retry/refit；其余 CLI 与程序化入口 |

非冻结站点传给 `build_execution_instance()` 直接 `raise`。

---

## 2. F-01：三分分别验证

`exec_identity.py` 新增 `capture_attestation` 与 `execution_instance`，`_chk_attestation()` 进入
`evaluate()` 的 **fatal 组**（缺一即 `unconfirmed`、两号 `None`）。

| 维度 | 实现 | 反例 |
|---|---|---|
| 内容完整性 | `event_sha256` / `budget_sha256` 与 `budget_evidence` 逐位对齐；证据指纹重算 | 改预算值不重算指纹 ⇒ 零号 |
| 来源真实性 | HMAC-SHA256 签名由**受控签发接口** `issue_capture_attestation()` 生成，验证时重算比对；`issued_by` 只认 `capture_boundary/v1` | 自洽字典 + 自造签名 / 不带凭证 ⇒ 零号 |
| 执行关联性 | `instance_token` 绑定 `data_sha256` + `candidate_id` + `site` + 本次唯一 `nonce`，且与本条记录逐字段比对 | 换 nonce / 换站点 / 贴别次凭证 ⇒ 零号 |

`make_rerun_record()` **不继承**父记录的 instance/attestation —— 子运行是新的执行实例，
继承就等于跨运行复用凭证，必须由调用方重新签发。

> **不得夸大的边界**：`evaluate()` 接收普通 dict，纯函数层无法密码学地证明
> 「证据来自真实 MCMC 构造过程」。这里是**受控接口签发 + 执行实例强绑定 +
> 落盘可审计**：项目密钥写在源码里（因此验证是纯函数、可跨进程复现，落盘后重读
> 仍可验证），但它不是私钥。能读源码并主动复刻签发逻辑的内部调用者仍可构造。
> 这一层由审计日志与落盘约束覆盖，**不宣称抗抵赖**。

**正对照**：受控接口签发的合规记录仍正常发 `R2-3bff356a…` / `W2-7ab370d1…`，
说明门槛只拒「未经签发」，不是一律拒发。

---

## 3. F-02：落盘组装 harness

`docs/round_2026-11-15/execution_harness.py`（harness 侧，不在顶层生产模块）：

- `capture_real_event()` —— 构造真实 numpyro MCMC 对象并调用**生产捕获接口**，
  **不调用 `MCMC.run()`**、不跑拟合。
- `assemble_record()` —— 只搬运事件里的 `budget_params` / `budget_evidence` /
  `budget_observations`，绑定执行实例；**不猜、不补、不按结果改身份**。
- `finalize()` —— 身份字段就是 `evaluate()` 的结果，不另行裁决；
  `attestation_status` 三态 `verified` / `rejected` / `absent`。
- `persist()` —— tmp + `os.replace` 原子落盘；审计日志 append-only JSONL。
- `load_and_verify()` —— 重读后复核关键字段、两个指纹、身份计算结果。
- `check_reuse()` —— 两道检测：实例 token 重复提交、A 次捕获事件被 B 次执行使用。

**明确不做**：决定拓扑 / 选择 K / 调整采样预算 / 自动补全缺失来源 /
根据拟合结果改变身份字段。**缺证据照常落盘**，只是 `unconfirmed`、两号 `None`。

`bm_core.py` / `bm_top.py` / `bm_real.py` / `bm_select.py` **零改动**；
`budget_params.py`（`8015721` blob 锁定）、`sampler_capture.py`、
`budget_capture_adapter.py` 均未动。

---

## 4. 单次执行链测试 `test_execution_chain.py` — 14/14

**四种情况**

| 场景 | 结果 |
|---|---|
| T1 正常结束 | 真实捕获 → 签发 → 落盘 → 重读一致；`unconfirmed` 零号（正确保守行为） |
| T2 采样异常 | 不签发、照常落盘完整记录、审计留痕、零号 |
| T3 证据缺失 | 不自动补、观测值仍保留、落盘、零号 |
| T4 落盘失败 | 不留下半条成功记录、审计记 `persist_failed`、不误报成功 |

**三个必留测试**

| 测试 | 结果 |
|---|---|
| 伪造证据 | 内部完全自洽的预算证据 + 自造签名 / 无凭证，全部零号 |
| **跨运行复用** | ① 凭证错关联（`attestation_data_binding_mismatch`）；② A 的事件被 B 复用（审计检测）；③ 同一实例重复提交 —— 三重拒绝 |
| 落盘后恢复 | 关键字段、两个指纹、身份计算逐位一致；并**起新 Python 进程**重读重算，结果与落盘 `identity` 一致 |

另含 F-01 三分单独断言、受控签发正对照、`bm_top.fit_top` 3 链 12/24 兼容验证、
冻结路径与排除清单、失败不误报成功。

---

## 5. 全套回归（提交前后各一遍）

68/68、45/45、35/35、31/31、25/25、45/45、28/28、27/27、76/76、
**执行链 14/14**、harness 自检 4/4、四模块自检通过；
`source_manifest.json --check rc=0`、冻结 9/9、阶段 0 `--check`（239/239/严格 237）、
`budget_source_probe.py` / `budget_chain_probe.py` / `capture_probe.py` `rc=0`、
冻结审查脚本 **51/51 FAIL 0**、`8015721` 三个被锁 blob 逐位 `SAME`、
5 个改动文件 `CRLF=0`、`git diff --check rc=0`。

---

## 6. 身份闭环复查（本轮最后一步）

| 问题 | 判定 | 依据 |
|---|---|---|
| **F-02** | **已闭环** | 真实捕获事件确实进入了正式身份记录并可靠落盘、可重读复核；四种情况都有测试；生产拟合代码零改动 |
| **F-01** | **条件性关闭** | 三个必留测试全过、自洽字典不再能拿号；但「来源真实性」是内部可信边界，不是抗抵赖证明；且 `upstream_origin_captured` 仍 `False`，上游来源链 S1/S2/S6 未闭合 |
| **Gate 2C** | **⛔ 仍不具备无条件准入** | 真实执行仍发不出有效 `run_id`；dirty patch 真实性 / 三时点 / harness 归属 / 多次采样逐次事件须真实 runner 实测；F-10–F-13、契约正文 C1–C8、P1×5/P2×1 未处理 |

**关键区分**：F-01/F-02 的工程闭环 ≠ 身份已可信到可以批量跑。
本轮证明的是「记录能被可信地产生、绑定、落盘、复核，且缺证据时如实零号」，
**不是**「现在能拿到有效身份号」。后者仍卡在上游来源证据。

---

## 7. 新发现、需单独裁决（本轮不顺手改）

1. **契约语义变了但版本未升**：本轮新增了「必须经受控签发才能发号」的门槛，
   但 `IDENTITY_CONTRACT_VERSION` 仍 `1.5`、`RUN_ID_CONFIG_KEYS_VERSION` 仍 `2`。
   实测与 `e19fe4b` 旧实现对比：**两个载荷键集相同、逐位相同，digest 相同**
   （`3bff356af1383db1ff147450…`）。即「号怎么算」没变，但「哪些记录能发号」变了。
   按项目 D-4「语义变了必须换号」，是否应 `1.5→1.6`（全换号是预期代价）
   **需单独裁决**，本轮不自行升级。
2. `source_manifest.json` 与源码一致但内容未变 —— 清单是声明集/角色，不含内容哈希，
   即第 4 步已记的 **G9** 缺口（改算号规则不换号）仍在。
3. F-10–F-13、P1×5/P2×1、契约正文 C1–C8 订正仍未处理。
4. 239 条 Stage 0 历史记录保持原证据等级，不用新规则重算、不补身份。

---

## 8. 边界与下一步

本轮**未跑完整 EIS 拟合**、未产生新的实验拟合结果、未启动 runner、
未宣布准入通过，也未把 `unavailable` 升级成 `recorded`、未用推断来源补历史缺口。

下一步建议（各自仍需单独授权）：

1. 裁决 §7.1 的版本升级口径；
2. **Gate 2C 正式准入**：按用户的准入表逐项给出 `PASS` / `CONDITIONAL` / `FAIL`，
   并明确哪些「设计具备、待 T1/T2/T3 实测」；
3. Gate 3 最小 runner；
4. **真实 EIS 单次拟合** —— 待身份记录能可靠落盘、重读、验证后单独授权。
