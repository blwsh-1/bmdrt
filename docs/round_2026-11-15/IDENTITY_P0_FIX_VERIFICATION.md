# Gate 2A 最小修复集：实现与测试收口

> **基线**：`c8a7bfe`（Gate 2A 独立审查报告，判定 **FAIL**）
> **用户裁决**：同意上一轮建议的「最小修复集 F-03–F-09，优先 F-08」，并要求
> **尽快开始测试，不要再一轮一轮出审查报告**。
> **本轮性质**：实现 + 测试。**不是**准入复核、**不是** runner、**没有**跑生产拟合。
> 本文件是简短收口记录，不是新的独立审查报告；Gate 2A 结论仍以
> `GATE2A_REVIEW.md` 为准。

---

## 0. 结论速览

| 项 | 结果 |
|---|---|
| F-03～F-09 七项缺陷 | 全部落地到 `exec_identity.py`，并各配反例 |
| P0 反例套件 | **35/35**（原 27 格 + F 组 8 格） |
| 全套回归 | 全绿（详见 §4） |
| 真实链路端到端测试 | **两个生产站点各一次，均 PASS**（详见 §3） |
| Gate 2A 是否因而转 PASS | **否**。F-01 / F-02 未动，F-10–F-13 待裁；本轮不作准入结论 |
| 最小 runner / 批量拟合 | **未启动**，仍需单独授权 |

---

## 1. 改了什么（只改身份层与测试，生产拟合代码零改动）

| 缺口 | 实现 |
|---|---|
| **F-03** `value` 无值域 | 新增 `_budget_value_ok()`：整数预算必须是 ≥1 的整数值，`target_accept ∈ (0,1)`；类型门禁仍在本层。负 / 零 / 2.5 / 越界 / bool / str 全部拒 |
| **F-04** `event_sha256` 只校长度 | 改为 `re.fullmatch(r"[0-9a-f]{64}", es)`，非 hex / 中文 / 空格 / 大写 / 长度错全部拒 |
| **F-05** 捕获词表不校闭集 | 新增 `BUDGET_CAPTURE_VERSIONS=("1.0",)`、`BUDGET_CAPTURE_SITES=("bm_core.fit","bm_top.fit_top")`；由测试与 `sampler_capture.CAPTURE_VERSION` / `CALL_SITES` 逐项交叉核验 |
| **F-06** workflow 整块二次承载预算 | 新增 `_budget_keys_in()` 递归 + `_chk_workflow()`：`search`/`extras`/`guards`/`gate`/`retry` 与 `pair_mode` 出现五键即 fatal；`_chk_retry_kw()` 也升级为递归查嵌套 |
| **F-07** `children` 无内容约束 | 新增 `_is_identity_ref()`：只许「已登记世代前缀 + hex 摘要」字符串；dict / `None` / 夹预算数值的字符串 / 非 list 全部拒 |
| **F-08**（阻断）不同 lineage 撞号 | `run_id_payload()` 新增 `rerun_lineage`；`_chk_rerun()` 强制子运行带合法 `parent_workflow_ref`。两个不同父工作流、同 attempt 的子 `run_id` 已分离，且两载荷除 lineage 外**逐位相等**（重跑没被误认成模型定义不同） |
| **F-09** 失败/成功重试同号 | `reason` 进 `rerun_lineage` 并强制非空；`sampler_error` vs `acf_gate_retry` 已不同号，空原因无号 |

**连带**：`make_rerun_record()` docstring 明示 `parent_workflow_ref` / `reason` 必填；
`test_identity_p0.py` 的 `_child()` 夹具补齐这两项。
`source_manifest.json` 重生成并 `--check rc=0`（版本仍 1.5，A18 / B22 / C21）；
重生成结果与 `HEAD` **逐位相同**，故不在本轮改动文件里 —— 清单记录的是
「声明集/角色」，不含文件内容哈希。这本身也是第 4 步 G9 已记的 P1 缺口
（改算号规则不换号），本轮不顺手扩展。

**未动**：`budget_params.py`（`8015721` blob 锁定）、`sampler_capture.py`、
`budget_capture_adapter.py`、`bm_core.py` / `bm_top.py` / `bm_real.py` / `bm_select.py`；
三处旧挂账不动；P1×5 / P2×1 不顺手扩展。

---

## 2. 反例：错误记录不能生成有效身份

`test_identity_p0.py` 每条错误记录都断言 `status == "unconfirmed"` 且
`run_id` / `workflow_id` **同时为 None**。新增 8 格：

- F-00 正对照：合规记录仍发 `R2-` / `W2-` 号，三个新检查函数均返回空
- F-03 十一组非法值（含同步重算证据指纹，确保测到的是值域门禁本身）+ 合法边界正例
- F-04 七种非法 `event_sha256`；F-05 五种非法版本/站点 + 与捕获层词表交叉核验
- F-06 五个 `_plain` 整块 + `pair_mode` + 嵌套 `retry_kw` 全被拦；纯策略键不换号
- F-07 四种非法 `children` + 合法引用仍正常改 `workflow_id`
- F-08 不同 lineage 子号分离、缺/假父引用无号、基础条件载荷不变
- F-09 失败/成功重试不同号、三种空原因无号

---

## 3. 真实链路端到端测试（不是纯函数自测）

两个生产构造站点各跑一次真实 `capture_at_construction()`（不 `run`、不拟合）：

| 站点 | 工况 | 结果 |
|---|---|---|
| `bm_core.fit` | NUTS / 1 链 / 8+8 / ta 0.90 / depth 10 | 捕获 `judged=True`、0 错误；五项观测值 8,8,1,0.9,10 |
| `bm_top.fit_top` | NUTS / 3 链 / 12+24 / ta 0.85 / depth 7 | 捕获 `judged=True`、0 错误；五项观测值 12,24,3,0.85,7 |

链路：真实 MCMC 构造对象 → `capture_at_construction()` →
`adapt_capture_event()` → `B.validate_budget_params(..., False)` →
组装 v1.5 记录 → `E.evaluate()`。

两次都得到 **`unconfirmed`、`run_id=None`、`workflow_id=None`**，原因码精确含
五项 `budget_not_recorded:*`；适配层/身份层两个指纹算法与事件里的
`evidence_sha256` 逐位一致；`upstream_origin_captured=False` 保持。
**这是来源未闭合时的正确结果，不是测试失败。**

> 明账：测试里另有 `rng_budget_mismatch`，因为其余条件字段借用已审夹具
> （`rng.plan.n_chains=2`），与 1 链 / 3 链工况不匹配；这是组装夹具的副作用，
> 不是生产缺陷。事件里 `aggregate.weakest` 分别为 `derived` / `mismatch`，
> 但适配层仍逐项输出 `unavailable` —— **捕获到值 ≠ 捕获到来源**。

---

## 4. 全套回归（提交前后各跑一遍）

| 类别 | 结果 |
|---|---|
| 身份 / 预算 / 采集 / 补丁 / provenance / 捕获 | 68/68、45/45、31/31、25/25、45/45、28/28 |
| P0 + F 组反例 | **35/35** |
| 契约 / registry | 27/27、76/76 |
| 四个模块自检 | 全部通过 |
| `source_manifest.json --check` | `rc=0` |
| 冻结 9/9；阶段 0 `--check` | 239 条 / 239 独立 / 严格 237 |
| `budget_source_probe.py` / `budget_chain_probe.py` / `capture_probe.py` | `rc=0`（全部字段对得上） |
| 冻结审查脚本（系统 Python ≥3.12） | **51/51，FAIL 0** |
| `8015721` 三个被锁 blob | 逐位 `SAME` |
| `git diff --check` | `rc=0`；改动文件 `CRLF=0` |

---

## 5. 仍未闭合（如实明账，不把未知写成通过）

- **F-01 完全自洽伪造证据块仍可拿 `R2-` 号**：本轮实测再次确认 `True`。
  纯函数层只能校结构 + 自洽 + 显式布尔，无法证伪「是否真的捕获过」。
  按上一轮裁决**移交 Gate 2B 可信边界**，不是收紧字段能解决的。
- **F-02 捕获 → 落盘记录组装断链**：静态扫描 `bm_*.py` 仍为「无」。
  §3 的组装是测试侧手工做的，生产链路不存在。
- **F-10–F-13** 待裁：两套 `IDENTITY_GENERATIONS`、`keys_version=2` 但 38 键未变、
  前缀与世代耦合、残留 1.4 标签。
- 来源证据 S1/S2/S6 未闭合；真实运行正确结果仍是 `unconfirmed` 无号；
  契约正文 C1–C8 未同步订正；P1×5 / P2×1 未处理。
- **Gate 2C 预判仍为 CONDITIONAL**：dirty patch 真实性 / 三时点 / harness 归属 /
  多次采样逐次事件，不跑 runner 无法实测。

---

## 6. 下一步（不再出审查报告，直接动手）

1. **冻结第一版 Runner 的唯一执行路径**（2B 前置，尚未冻结）。
   建议 `nested_fit.py` 独立接口、K3 `m35_cond_n2.py:116`、`bm_select.py` retry
   与其余入口明确列「不纳入、不外推」；三挂账不动。
2. **补 F-02 的落盘组装接口**，让真实捕获事件的 `budget_params` /
   `budget_evidence` 能被生产侧搬进身份记录；随后跑一次真实单次执行链测试。
3. **F-01 由可信边界闭合**：`upstream_origin_captured` 不能继续是记录里的自述字段。
   红线不变 —— 不把 `unavailable` 升级成 `recorded`、不用推断来源补历史缺口、
   239 条历史记录保持原证据等级。
4. 之后再谈 Gate 2C 准入结论；**Gate 3 最小 runner 与试运行验收各自仍需单独授权**。

> ⛔ 本轮**不宣布准入通过、不启动 runner、未跑生产拟合**。
