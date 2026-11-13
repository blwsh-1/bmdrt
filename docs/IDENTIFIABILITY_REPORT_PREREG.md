# 统一参数可辨识性报告：预注册（P0 ①，2026-11-12）

> **预注册先行**：本文件先冻结 schema 与边界，再写实现。
> 对应 `CIRCUIT_TOPOLOGY_ADMISSION.md` §4.2 / §7 的 P0 ① 与
> `MODEL_COVERAGE_PLAN.md` §5（四类结论）、§6（四条准入规则）。
>
> 用户评审定的优先级：「这是当前模型评价体系的基础能力，不是可有可无的电路扩展」。
> 本文**不改任何拟合代码、判据、阈值、先验、验收线**。

---

## 0. 范围与不做的事

| 做 | 不做 |
|---|---|
| 定义 voigt / nested **同 schema** 的可辨识性报告 | 不注入 `AnalysisResult.payload()` / `bm_dash.py` 面板 |
| 新增独立模块 `bm_identifiability.py`（纯函数） | 不改 `bm_result.py` / `bm_dash.py` / `bmdrt.py` 一行 |
| 批处理器：每个已有候选模型**各出一份**报告 | 不改契约 F1–F10 的断言（只新增 F11） |
| 报告**暴露**现有产物缺失的诊断项 | 不伪造 `unavailable` 项、不把缺数据写成「通过」 |
| 只记账、只列已有结论 | **不裁决**：不判通过/拒绝、不给分数、不改 `status` / `sample_grade` / 门禁 |

**为什么本阶段不注入生产路径**：注入 `AnalysisResult` 会改变 `payload()` 与封包内容，
直接触碰契约 F4（面板载荷逐值相等）与 F5（封包 finalized）。按项目既有纪律
（默认关、`exploratory`、生产侧改动需单独预注册），本次以**独立报告产物**落地
「每族必出」，生产注入留作 P0 ① 的第二阶段，届时另开预注册并同步改 F4/F5。

---

## 1. 六类诊断（对齐用户评审表格）

| # | 类别 | 要回答 | voigt 来源 | nested 来源 |
|---|---|---|---|---|
| 1 | `posterior` 参数后验 | 参数是否被数据约束 | `real_*.json:param_corr.summary`（**原尺度**） | `*.nested.json:params[].median/ci90` |
| 2 | `correlation` 参数相关 | 是否存在补偿关系 | `param_corr.strong/medium/corr` | **无**（无样本 → 相关矩阵不存在） |
| 3 | `boundary` 边界诊断 | 是否由先验截断决定 | `param_corr.fixed` + `component_review.parameter_identifiability` + `degeneracy[].at_bound` | **无**（同样需后验样本） |
| 4 | `combinations` 参数组合 | 只能定组合还是能定单参数 | `param_corr.degenerate`（`Rinf-Wtot` / `R-tau ridge`） | **无** |
| 5 | `sampling` 采样诊断 | 后验是否可信 | `diag[].rhat/bulk/tail` + `divergences` | `diagnostics.rhat_by_param/ess_min/divergences` |
| 6 | `prediction` 预测诊断 | 模型是否充分描述数据 | `acf` / `rms_re,rms_im` / `resid_bins` / `elpd*` | `metrics.acf/rms_rel/bands_pct` / `diagnostics.elpd*` |

**PPC**：两族产物均未落盘后验预测检查，统一写 `ppc.status = "unavailable"` +
说明，**不得**用残差 acf 冒充 PPC。
**窗口截断 / 剖面平坦度**：`evidence/nested_topology/identifiability_check.py` 是
独立脚本、非逐谱产物，本阶段只作为 `notes` 里的已知结论引用，不并入逐谱 schema。

---

## 2. 冻结的 schema

```
{
  schema_version: "1.0",
  family:            "voigt" | "nested",
  model_id:          str,          # 谱名 + 复杂度（voigt K / nested L+branch）
  source_files:      [str],
  status:            str,          # 原产物的 status（原样抄，不改）
  disclaimer:        str,          # 「只记账不裁决」固定声明
  diagnostics: {
    <六类 key>: {
      status:        "available" | "partial" | "unavailable",
      source:        str,          # 具体字段路径，缺则写缺什么
      comparable:    bool,         # 跨族是否可比（口径不同必须 false）
      ...各类专属字段
      note:          str
    }
  },
  coverage: { available: int, partial: int, unavailable: int },
  notes: [str]
}
```

### 2.1 三态规则（硬性）

| 状态 | 含义 | 允许写什么 |
|---|---|---|
| `available` | 所需字段齐全且口径明确 | 数值 + 来源 |
| `partial` | 有部分字段，或口径与另一族不同 | 已有数值 + `note` 写明缺哪部分 |
| `unavailable` | 所需数据根本没落盘 | **数值一律 `null`**，只写 `source`/`note` |

**禁止**：把 `unavailable` 写成 `pass`/`ok`/「通过」；用 0 或空列表代替 `null`；
跨族把 `rms_rel` 与 `Rwp` 摆在同一字段下比较（`nested_fit.rms_rel = Rwp/√2`，
见 `CAPACITIVE_BRANCH.md` 的口径订正）—— 两族的拟合优度字段各留原名并标
`comparable: false`。

### 2.2 专属字段

| 类 | 字段 |
|---|---|
| `posterior` | `n_params`, `params: [{name, q05, q50, q95, rel_span}]`；`rel_span=(q95−q05)/|q50|`（`q50≈0` 时 `null`，不写 0） |
| `correlation` | `n_draw`, `strong_tol`, `medium_tol`, `strong_pairs`, `medium_pairs`, `matrix_names` |
| `boundary` | `fixed_params`, `at_bound_params`, `at_bound_fraction`（`null` 表示算不出）, `n_params` |
| `combinations` | `known_combinations: [{kind, comp, rho, text}]`（**原样转述 `param_corr.degenerate`，不新增判断**）, `single_param_limitation: [str]` |
| `sampling` | `rhat_max`, `ess_bulk_min`, `ess_tail_min`, `divergences`, `per_param` |
| `prediction` | `acf`, `fit_metrics`（各族**原名**键→值）, `residual_bands: {kind, values}`, `elpd`, `elpd_valid`, `ppc: {status, note}` |

### 2.3 数值纪律

- `param_corr.summary` 的分位**已是原尺度**（对数参数已 `exp` 回），
  实现里**禁止再取 `exp()`**（否则显示 1.0000，见 `bm_dash.py` 注释）。
- 所有输出必须过 `json.dumps(..., allow_nan=False)`；非有限值写 `null`。
- 报告里**不得出现** `grade` / `score` / `verdict` / `pass` / `accept` 键，
  也不调用 `bm_grade.annotate()`。可辨识性只记账，裁决权仍在生产门禁。
- 不新增「可信度总分」：面板二轮评审已明确不做（`BMDRT_RESULT_AND_FIGURE_DESIGN.md`）。

---

## 3. 跨族同口径的准确含义

「同口径」= **同一 schema、同一字段名、同一三态规则**，不是「数值可直接比」：

- 族特有量各留原名（`voigt.rms_re/rms_im` vs `nested.rms_rel`），`comparable: false`；
- `acf` 两边定义相同（`acf_max` 滞后 1–5、尺度不变），可跨族并列展示但仍不据此裁决；
- `family` 只是标签，不改变字段集合；
- 任何一族缺项时，另一族**不跟着降级**，各自独立标状态。

---

## 4. 预期会暴露的缺口（报告的价值之一）

跑通后预期结论（**待实测确认，不预写进代码**）：

1. **nested 侧 `correlation` / `boundary` / `combinations` 会是 `unavailable`**：
   `nested_fit.py` 不落盘后验样本，`*.nested.json` 只存逐参数分位与 R̂/ESS。
   ⇒ 这正是「增加任何电路都难判断它是否有效」的具体表现。补样本落盘属
   `nested_fit.py` 的改动，另立预注册，本阶段**不补**。
2. **两族 PPC 都是 `unavailable`**：产物里没有后验预测检查。
3. **voigt 侧 `boundary` 大概率 `partial`**：有 `fixed` / `at_bound` 线索，
   但没有逐参数先验边界表 + 后验样本的贴边比例，不能声称算出了「比例」。
4. 已知结论应能被自动复现：`R_inf` 与 `W` 强负相关（`Rinf-Wtot`）、
   `Rs` 剖面平坦 → 报告 `combinations` 里出现对应文本，而不是一句「参数不可辨识」。

**这些缺口是报告要如实写出的结果，不是实现失败，也不构成任何模型的通过/拒绝。**

---

## 5. 验收用例（契约 F11，新增，不改 F1–F10）

| 编号 | 断言 |
|---|---|
| F11-1 | 两族报告的 `diagnostics` 键集合完全相同（同 schema） |
| F11-2 | `param_corr.summary` 的原尺度分位被原样搬进 `posterior`，数值逐值相等（防 `exp()` 回归） |
| F11-3 | 缺数据项 `status="unavailable"` 且所有数值字段为 `null`；不许出现 `pass`/`ok` |
| F11-4 | 报告不含 `grade`/`score`/`verdict`/`pass`/`accept` 键；`status` 与原产物一致 |
| F11-5 | `allow_nan=False` 严格 JSON 可序列化；`coverage` 三态计数与 `diagnostics` 一致 |
| F11-6 | 两族拟合优度字段不同名且 `comparable=false`（防 `rms_rel` 当 `Rwp` 比） |

回归：`test_contract.py`（全绿）、`bmdrt.py check` 18/18、`bm_top.py` 自检、
`py_compile`；**契约 F1–F10 一行不改**。

---

## 6. 复现

```bash
cd D:/eis训练数据/bmdrt
python test_contract.py --skip-fit
python docs/round_2026-11-12/build_identifiability_report.py \
    --out out/identifiability_report
```

产物：`out/identifiability_report/<stem>.<family>.json` + `summary.md` + `manifest.json`。
批处理器只读 `out/` 下已有产物，不重跑拟合、不改源文件。

---

## 7. 与既有文档的关系

| 文档 | 关系 |
|---|---|
| `MODEL_COVERAGE_PLAN.md` §5 | 四类结论规范已存在；本预注册把它落成**可落盘的逐谱产物** |
| `CIRCUIT_TOPOLOGY_ADMISSION.md` §4.2 / §7 | P0 ① 的准入状态由「冻结、未实现」改为「已交付（独立报告产物）」 |
| `THREE_LAYER_SCOPE.md` / `bm_grade.py` | 只记账不裁决；`grade` 只许 A/B/C/null、不给分数 |
| `BMDRT_RESULT_AND_FIGURE_DESIGN.md` | 不做可信度总分、面板不改 |
