# `K_SELECTION_AMBIGUOUS` 判定规格（预先登记，未实施）

> **边界**：本文件是**规格与离线回放**，不是代码改动。
> `D:/eis训练数据/bmdrt` 未改一行，HEAD `f745205`，工作区干净，未提交。
> 所有阈值在**看到回放结果之前**写定（见 §3），回放只用于检查是否误伤，不用来反推阈值。

---

## 1. 要解决的问题

现有 `select_k()` 的顺序是：

```
采样分级 → 硬性剔除（P(峰出窗)>0.5 / 分级0 / φζ>1 / Pareto-k）→ elpd 最大 → 差<max(2,dse) 取小 K
```

它把「峰位出窗」当成「模型无效」，于是出现本案例的边界情形：K=2 的 LOO 好 187±14.9（12.6σ），却因慢分量峰位出窗概率 0.892 被整体剔除，退回 K=1，留下系统性残差。

需要区分两个判断：

- **出窗 ≠ 无效**：慢分量在 0.1 Hz 提供模型虚部幅值的 95%，其窗口内阻抗贡献是可辨的；
- **LOO 更优 ≠ 物理机制正确**：K=2/Ws 只是把峰挪进窗口（`P_out` 0.892→0.049），拟合指标没有实质改善；`none/Ws/Wo` 的总阻抗预测最大差仅 0.51%。

因此新增一个**模型选择状态**（不是新物理模型），在「LOO 明显更优」与「峰位约束不满足」同时成立时，拒绝输出唯一最佳 K。

---

## 2. 新判定顺序

```
① KK 门禁（拟合前，已有）
      fail → rejected_by_kk（不变）
② 采样诊断（每个候选 K）
      不合格 → 该候选进 bad 集，不参与后续（不变）
      全部不合格 → rejected（不变）
③ 只有一个可接受候选 → not_compared（不变）
④ LOO 比较（可接受候选之间）
      Δ = elpd_best − elpd_smaller，阈值 thr = max(2, k·dse)
⑤ 检查峰位约束（只对 best 候选）
      若 Δ > thr 且 best.P_out_max > 0.5 且 best.K > smaller.K
          → **ambiguous**
      否则按原规则：Δ ≤ thr 取 smaller；Δ > thr 取 best → converged
⑥ 简并/降级判定（已有 degraded 逻辑）保持不变
```

关键点：**第 ⑤ 步不剔除 best，也不放宽 0.5 阈值**。它只是把「LOO 说该选大的、峰位说大的不可信」这个冲突显式化。

---

## 3. 预先登记的阈值（待确认项用 ⚠️ 标出）

| 项 | 取值 | 理由 | 状态 |
|---|---|---|---|
| `p_peak_out` | **0.50** | 沿用现有硬筛选阈值，不放宽（审查意见明确要求） | 固定 |
| `elpd_abs` | **2.0** | 与现有「差 < max(2, dse) 取更小 K」同一个常数，保持对称 | 固定 |
| `elpd_k` | **1.0 × dse** | 「显著更优」= Δ > max(2, 1·dse)，与「不显著」判据严格互补 | ⚠️ 待确认：是否要更严（2·dse） |
| `rhat` | 1.01 | 现有严格线 | 固定 |
| `ess` | bulk 与 tail 均 ≥ 400 | 现有严格线；quick 档达不到 → 只有 full 档能触发 | 固定 |
| `div` | 0 | 现有严格线 | 固定 |
| `tree_cap` | < 1% | 现有严格线 | 固定 |
| `grade_min` | **1** | 分级 0 = 采样不可用，不允许进入判断 | ⚠️ 待确认：是否要 ≥2 |
| `kk_status` | 必须 `ok` | `warn` 时数据自洽性存疑，不应进入 K 判定 | ⚠️ 待确认：`warn` 是否也放行 |

---

## 4. 状态与输出字段

### 4.1 状态优先级

```
rejected_by_kk  >  rejected  >  ambiguous  >  degraded  >  converged  >  not_compared
```

`ambiguous` 排在 `rejected` 之后：采样不合格时仍然直接拒绝，不允许用「不确定」掩盖「不可用」。

### 4.2 新增/变更字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `k_selection_status` | str | 新增取值 `ambiguous` |
| `k_candidates` | list[int] | ambiguous 时保留参与比较的 K（如 `[1,2]`），不丢信息 |
| `k_reference` | int | **仅用于绘图/预测的参考模型**（建议取 LOO 最高的 K），不得当作结论 |
| `k_ambiguous_reason` | str | 例：`K=2 LOO 优于 K=1（Δ=187.0 > 14.9），但分量 P(峰出窗)=0.89 > 0.5` |
| `required_controls` | list[str] | `["Ws","Wo"]`（已跑 none）；对照未完成前不输出唯一 K |
| `k_selection` | list[str] | 保留两个候选的逐条判定文本（现有字段，不删） |

### 4.3 报告与批量汇总

- 单条日志：打印「**K 选择不确定（ambiguous）**：保留 K=1/K=2，请先做 none/Ws/Wo 对照」，不再打印「→ K=1」。
- `batch_2e.py` / `report_2e.py`：`ambiguous` **单列计数**，不计入「拟合成功」也不计入「KK 拒绝」；`index.html` 与 `summary_all.csv` 增加一列。
- 退出码：`ambiguous` 不算失败（与 `rejected_by_kk` 同样的处理原则）。
- 绘图：仍用 `k_reference` 画，但图标题加 `K ref only (ambiguous)`。

---

## 5. 离线回放（现有 10 组，不改代码）

脚本 `replay_ambiguous.py`；结果 `replay_ambiguous.csv`。只解析已落盘的 `run.log` / JSON，不重跑 NUTS。

| 案例 | 旧状态 | 旧 K | 新状态 | 参考 K | 说明 |
|---|---|---|---|---|---|
| `K1_none`（原运行） | `converged` | 1 | **`ambiguous`** | 2 | K=2 elpd 67.5 vs K=1 −119.5，Δ=187.0 > 14.9；K=2 的 `P_out=0.89` |
| `K1_cut1Hz` | `converged` | 1 | **`ambiguous`** | 2 | Δ=75.4 > 14.1；`P_out=0.94` |
| `K2_Ws` | `converged` | 2 | `converged` | 2 | Δ=76.2 > 14.1，但 `P_out=0.049 < 0.5` → 不触发 |
| `K2_none` | `not_compared` | 2 | `not_compared` | 2 | 单候选（`--krange 2,2`） |
| `G_K1_none` … `G_K2_Wo`（6 格） | `not_compared` | 1/2 | `not_compared` | 1/2 | 全部强制单 K，不涉及 K 比较 |

**回放结论**：新状态只命中「LOO 显著更优 + 峰位出窗」的两条真实谱，不改变正常选定的 `K2_Ws`，也不改变强制单 K 的 6 格。没有出现「把已经能定的 K 变成不确定」的误伤。

---

## 6. 改代码后的回归测试清单

1. `bmdrt.py check` 18/18。
2. `bmdrt.py diff --fast` 全通过（合成基准 B1/B2/C1/Wo/可辨识性/DRT）。
3. 本案例 4 组基线 + 6 格网格重跑，`k_selection_status` 与 `replay_ambiguous.csv` 逐条一致（`K1_none`/`K1_cut1Hz` → `ambiguous`，其余不变）。
4. 2E 117 条 `--quick` 与 `--full` 抽样回归：确认 `ambiguous` 只在「K 扫描 + 峰位出窗」出现，不把 5 mA 组 13 条（`P_out=1.0`）误判成 `rejected`。
5. `batch_2e.py` 汇总与退出码：`ambiguous` 不计失败、不染红整批。
6. ECM 模拟数据 797/1139：K 由 LOO 选、峰位在窗内 → 应保持 `converged`，不触发。

---

## 7. 明确未做的事

- 未改 `bm_real.select_k()`、未新增状态枚举、未改 `p_peak_out` 阈值、未改 `batch_2e.py` / `report_2e.py`。
- 未把 `required_controls` 做成强制流程（目前只是建议字段）。
- 未确定 `ambiguous` 在 HTML 报告里的具体样式。
- 未提交仓库。

**需要你拍板的四个 ⚠️**：`elpd_k` 用 1×dse 还是 2×dse；`grade_min` 取 1 还是 2；`kk=warn` 是否放行；`k_reference` 取 LOO 最高还是最小 K。
