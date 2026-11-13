# 参数恢复 / LS 反事实实验预注册（准入流程第 ④ 步）

日期：2026-11-12　状态：**冻结口径后再跑**（先写本文件，再写 `truth_recovery.py`）
上游：`CIRCUIT_TOPOLOGY_ADMISSION.md` §4.1.3 / §6、`IDENTIFIABILITY_REPORT_PREREG.md`、
`out/truth_dataset/truth_manifest.json`（168 条，`status=finalized`）。

## 0. 定位与红线

- 本实验是准入文件 §6 统一流程的 **④ 查残差结构 / 参数恢复 / 可辨识性**，
  只到 **LS 反事实** 为止。**不是** ⑤ Bayesian 采样，更不是模型选择门禁验证。
- **不裁决生产**：不接 `bm_select.BRANCHES`、不接 `bmdrt.py`、不改任何判据、阈值、
  先验或验收线。§13 `0/3`、§14 `4/14`、§17 `0/10` 全部原样。
- **合成谱不是真仪器数据**：`truth_manifest.json` 已注明噪声为人为生成过程
  （white / correlated / heavy）。本实验的结论**不得**升格为真实样品的准入证据。
- 输出只记账、不给 `grade`/`score`/`pass`/`verdict` 键（沿用
  `IDENTIFIABILITY_REPORT_PREREG.md` 的 L3 纪律）。

## 1. anti-inverse-crime：参数恢复层的三条硬保证

§6 已指出「正演层交叉验证 ≠ 参数恢复层独立」。本轮必须同时满足：

| # | 保证 | 实现方式 |
|---|---|---|
| A1 | **生成侧与拟合侧不共用正演代码** | 拟合侧 `truth_recovery.py` 另写一套 `z_cpe_admittance` / `nested_by_admittance` / `voigt_chain`，用**导纳累加**递推；生成器用 `par(a,b)=ab/(a+b)` 显式循环。两者数值表达式与代码路径都不同 |
| A2 | **初值与真值无关** | 所有起点由 `z_max`、`ω_ref`、稳定哈希 seed 生成，**绝不读 `truth_params` 起步**；真值只在**评估阶段**用于算恢复误差 |
| A3 | **第三方对拍** | `--selftest` 用项目 `bm_top.z_nested` 与生成器 `truth_dataset.forward` 三方对拍 4 个 `(L, branch)` 组合；对拍结果**不参与拟合**。并要求拟合侧 7 类拓扑正演与生成侧逐条一致（相对误差 <1e-10） |

若 A1–A3 任一不成立，实验结论作废。A3 只证明**正演等价**，不证明拟合能恢复参数 ——
参数恢复必须实跑。

## 2. 数据

- 只读 `out/truth_dataset/*.csv`（`freq,Zreal,Zimag`，60 点升序 0.1–1e5 Hz）
  与 `truth_manifest.json`（提供 `family` / `difficulty` / `noise` / `truth_params`）。
- **实际噪声分层以 manifest 的 `noise.kind` 为准**（跑前核实）：
  `none` 56 条 (0%)、`correlated` 56 条 (2%，带频率相关系统项)、
  `heavy` 56 条 (5%，Student-t(3) 重尾但 **i.i.d.、无频率相关结构**)。
  `structures` 声明的 `white` **从未生成**（0% 档在 `add_noise` 里被短路成 `none`）；
  `README.md` 的噪声描述与生成代码不一致 → 记为**数据集声明缺口**，本轮不改数据集
  （改会动 168 条 sha256），只在报告里如实记录，留待单独订正。
- **`correlated` 档自带频率相关结构**（`NOISE_ACF_TRUTH.md`：acf 尺度不变、只响应
  频率相关结构）。因此 **条件 ①/②/④ 的主判据只在 `kind ∈ {none, heavy}` 上算**；
  `correlated` 单列为敏感性分析，不得用来宣称「模型缺结构」或「结构已改善」。
  `heavy` 虽重尾但 i.i.d.，按 §18 的 i.i.d. 限定可纳入主集。

## 3. 候选池（9 个，全部用拟合侧独立正演）

| 候选 | 参数数 k | 角色 |
|---|---|---|
| `voigt_K1`（≡ ZARC） | 4 | 负样本真族 / 最简基线 |
| `voigt_K2` | 7 | 自由度对照 |
| `voigt_K3` | 10 | 自由度对照（§9.4.10.1 纯 RC 下 C3 可退化成 voigt K=3） |
| `nested_L1_inductive` | 6 | **无容性支路**对照 |
| `nested_L2_inductive` | 9 | **无容性支路**对照 |
| `nested_L1_capacitive` | 6 | 新增支路 |
| `nested_L2_capacitive` | 9 | 新增支路 |
| `C2_L3` | 12 | 三层嵌套（自由度红利对照） |
| `C3_mixed` | 12 | 混合拓扑（含容性最内层 + 串联 ZARC） |

- 参数口径统一用 **`Q_data`（`Z_P = Q_data/(iω)^n`）**，与 manifest 的 `truth_params`
  同口径，才能直接算恢复误差；项目 `bm_top` 的 `Q_ours = Q_data/ω_ref^n` 只在 A3 对拍时换算。
- 感性支路统一 `R_ad∥L`（`tau_ad`），与生成侧一致；`series` 支路不在本数据集真值里，不进池。

## 4. LS 拟合口径（预注册，跑完不许改）

- 残差向量 = 实部/虚部拼接的**相对残差**
  `r = [(Z_fit−Z).real/|Z|, (Z_fit−Z).imag/|Z|]`，长度 `n = 120`。
  与 `Rwp = sqrt(mean(|Z_fit−Z|²/|Z|²))·100` 同口径（**不除 √2**）。
- 参数在 **log 空间**拟合正数（R/Q/tau），`n` 与 `n_b`/`n_i` 用 `logit` 限在 `[0.30, 0.999]`；
  边界 `R,Q,tau ∈ [1e-4·z_max … 1e4·z_max]`（tau 用 `[1e-8, 1e8]` s）。
- 每候选 **3 个起点**：由 `z_max`、`ω_ref` 的尺度中心 + 稳定哈希扰动生成
  （`md5(candidate|spectrum_id|start_index)`），**与真值无关**。
- `scipy.optimize.least_squares`，`max_nfev=1500`，`ftol/xtol/gtol=1e-10`。
  收敛判据只认 `res.success`；全部起点失败 → 记 `not_converged`，**不报数**。
- 取 chi2 最小的收敛解。多起点解的**参数离散度**作为 LS 层局部可辨识性线索记账。

## 5. 指标（每条谱 × 每候选）

| 字段 | 定义 |
|---|---|
| `rwp_pct` | `sqrt(mean(|Z_fit−Z|²/|Z|²))·100` |
| `acf` | `max(acf_max(res.real), acf_max(res.imag))`，滞后 1–5，与 `bm_experiments.acf_max` 同口径 |
| `chi2` | `sum(r²)`（相对残差） |
| `bic` | `n·ln(chi2/n) + k·ln(n)`，`n=120`；BIC 只作**谱内候选比较** |
| `k` | 参数数 |
| `converged` | 是否有起点 `success=True` |
| `start_spread` | 收敛起点间 `max|ln(p_i/p_j)|` 的最大值（LS 层线索） |
| `jac_cond` | Jacobian 条件数（LS 层线索，非后验） |
| `param_error` | 仅对**真族候选**算：正参数 `|ln(fit/truth)|`，指数 `|Δn|`，逐参数 + 中位 |
| `f0_error` | 组合量：最内层 CPE 特征频率 `f0=(R/Q)^(1/n)/2π` 的 `|ln(fit/truth)|`（单参数不可辨识时看组合） |

**BIC 不是门禁**：它只用来定义「按拟合优度+复杂度自动选谁」，不得写成项目验收结论。

## 6. 四条最低验证条件的可判定形式（阈值先冻结）

主样本集 `S = {真值 family ∈ {nested_L1_capacitive, nested_L2_capacitive}，
noise.kind ∈ {none, heavy}}` = 2×4 难度×2 噪声×2 seed = **32 条**；
负样本集 `N = {真值 family ∈ {ZARC, nested_L1_inductive, nested_L2_inductive, C2_L3}，
noise.kind ∈ {none, heavy}}` = **64 条**。

| # | 条件 | 预注册判据 | 本轮能否判定 |
|---|---|---|---|
| ① | 无支路时**稳定**结构性残差 | `S` 上 `nested_L*_inductive` 的 acf **>0.50 的谱占比 ≥75%** 且中位 acf ≥0.50（按真值 L 配对） | 能 |
| ② | 加入后**结构改善**而非只降 Rwp | `S` 上配对 `Δacf = acf_ind − acf_cap` ≥0.20 的谱占比 ≥75%；同时报 `Δrwp` 与 `C2_L3` 的 `Δacf` 作自由度红利对照 | 能 |
| ③ | 新增参数有**可接受的后验约束** | **不能判定**。LS 无后验；本轮只报 `start_spread` / `jac_cond` / 贴边界计数 / 参数恢复误差作**线索**，状态仍保持 ❌，留待 Bayesian（M3） | **否** |
| ④ | 错误拓扑上**不频繁误选** | `N` 上按 BIC 最小选中含容性候选（`*_capacitive` 或 `C3_mixed`）的比例 **≤10%**；另报真族识别率作有效性 sanity check（真族识别率 <50% 时误选率不作结论） | 能 |

补充报告（不设通过线，只记账）：
- `C2_L3` / `voigt_K3` 在非其真值谱上的 BIC 胜率（自由度红利）；
- 按 `difficulty` 分层的真族识别率与参数恢复误差（简并/窗外预期更差，属预期不是失败）；
- `correlated` 档的 acf 抬升幅度（验证 §18 规则 2 的 i.i.d. 限定）；
- `heavy` vs `none` 的 acf 差异（验证 acf 对幅度/重尾不敏感）。

## 7. 诚实边界（写进报告，不许省）

1. **LS 证据 ≠ 后验证据**：条件 ③ 本轮不通过也不失败，状态维持 ❌。
2. **合成谱 ≠ 真实样品**：即使 ①②④ 全过，准入状态仍为 `exploratory`，
   不得写成「容性支路已通过验收」或据此接 `bm_select` / `bmdrt.py`。
3. **参数恢复误差大 ≠ 模型错**：简并/窗外难度下可能存在「单参数不可辨识但组合可辨识」，
   必须同时看 `f0_error` 与 `start_spread`，不得只看单参数。
4. **acf 不是 PPC**，`correlated` 噪声自带 acf，主判据已排除该档；
   `white` 声明未生成，本轮不据它作任何结论。
5. BIC / 误选率是本实验自定义的**诊断量**，不是项目门禁或 §13/§14 的判据。

## 8. 产物

- `docs/round_2026-11-12/truth_recovery.py`（独立拟合侧正演 + LS + 报告，只读数据集）
- `out/truth_recovery/results.json`（逐谱逐候选）、`summary.md`、`manifest.json`
- 本文件 `RECOVERY_PREREG.md`（口径冻结）
- 回写 `CIRCUIT_TOPOLOGY_ADMISSION.md` §4.1.3 / §6 的**证据状态**（准入判定本身不变）

不改生产代码、不注入 `bm_result.payload()`、不动契约 F1–F11。
