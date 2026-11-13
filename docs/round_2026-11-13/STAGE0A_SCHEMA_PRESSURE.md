# 阶段 0A：三谱 schema 压力测试

状态：**只整理既有产物**。本轮不运行任何新拟合、不修改任何原始产物、不动生产代码。
目的：用三条已跑过的谱，检验候选电路注册表的 schema 能否表达它们的真实差异。
结论只针对 schema 的表达力，不针对任何模型是否"正确"。

## 0. 方法

三条谱各覆盖一种"必须在 schema 里被区分开"的情形：

| 谱 | 为什么选它 | 既有产物（只读） |
|---|---|---|
| 涂层 `R1-[P2,R3-[P4,R5-P6]]` | 同一拓扑下换元件类型，结果从"崩"变"贴底" | `out/nested_fit_cap_control/`、`out/nested_fit_cap_standard/` |
| 钝化 `[P1,P2-R3]-R4` | 同候选、同数据，只换初始化，从"塌陷"变"收敛" | `out/_user_fit_nested/diag_passivation_init.json`、`out/nested_fit_cap_pass/` |
| AZ31 `AZ31-4Li_100.csv` | 一个族被结构判据排除，另一个可跑但残差仍有结构 | `out/m4_real_az31.json`、`out/m4_real_nested_az31.json` |

三者的取值全部抄自上述文件，未重算。

## 1. 涂层：同一拓扑、换元件类型

同一条谱、同 `standard` 档、同 `σ_j = 2%·|Z_j|`：

| 候选 | `branch` | `rms_rel` | `acf` | `rel_median` | `R̂_max` | 关键参数 |
|---|---|---|---|---|---|---|
| nested L=2 | `parallel`（默认） | 10.81% | **0.944** | 12.77% | **14.21** | `n2` 0.95 → **0.503** |
| nested L=2 | `capacitive` | **7.08%** | **0.279** | **2.40%** | **1.005** | 参数偏 10–40%，无数量级错 |

真值（由文件名解析）：`Rs=150`、`R1=309325`、`Q1=4.43e6`、`n1=0.9`、`R2=1e5`、`Q2=2.92e5`、`n2=0.95`、`Q_b=1761`、`n_b=0.5`。

**这条谱逼出的 schema 需求**

1. `branch` 是**元件选择**，不是第二个拓扑。schema 必须能表达"Nested 拓扑 + 最内层元件 = 纯 CPE"，而不是新增一个叫 `nested_capacitive` 的平级模型。
2. `parallel` 的 `n2` 从 0.95 掉到 0.503 —— 这是**参数退化到边界**，不是"没收敛"。`rms 10.81%` 也不算灾难性（噪声底约 2%），所以"拟合差"这个单一维度描述不了它。需要单独的 `param_stability` 字段。
3. `R̂=14.21` 与 `acf=0.944` 同时出现：前者是**采样层**（后验诊断没过），后者是**结构层**。两者不能合并成一个"不合格"；而残差已经算出来了，也不能因为采样没过就丢掉——这正是 0.2.0 把 `run_status` / `sampling_status` / `fit_observation` 拆开的直接原因。
4. `capacitive` 的 `rel_median` 2.40% ≈ σ 本身（`rms_rel` 仍是 7.08%——实虚联合 RMS 由低频大幅值点主导），`acf` 0.279 低，但参数仍偏 10–40% —— **贴底 ≠ 参数可信**，schema 必须同时保留两列；两个残差口径也不能互换（7.08% 与 2.40% 是不同量）。

## 2. 钝化：同候选、只换初始化

| 运行 | 初始化 | 收敛 | `rms_rel` | `acf` | `R̂_max` | `R1` 估计（真值 9.28e6） |
|---|---|---|---|---|---|---|
| `truth_init` | 真值附近 | 13.1s | **6.65%** | **−0.244** | 1.004 | 9.42e6 |
| 默认 init | `R_inf=min\|Z\|`、τ 均铺 | 未收敛/塌陷 | 98.40% | 0.750 | **197.787**（`ess_min` 1.0005） | **108.4** |

真值：`Rs=141.24`、`R1=9277753`、`Q1=66170`、`n1=0.78245`、`Q_b=329605`、`n_b=0.5002`。
`R1` 与真值差 5 个数量级以上，形态退化为"两个 CPE 并联"。

**这条谱逼出的 schema 需求**

1. **初始化是候选配置的一部分**，不是拟合的实现细节。同一 `candidate_id` 配不同 `init_profile` 会产生完全不同的 `execution` 结果，所以 `init_profile` 必须进候选定义（否则 manifest 无法复现）。
2. 需要区分 `not_run` / `failed` / `completed`。默认 init 那次不是"结构不行"，是"没进入有效后验"。
3. **这是 L1 不能单次失败即排除的直接证据**：如果 L1 用默认初值快速跑一次就 `excluded`，这条谱会把初始化问题永久记成结构问题。
4. `acf = −0.244` 说明 `acf` 可以为负 —— schema 不能假设它落在 [0,1]，阈值判断要按绝对值或按既有口径处理。

## 3. AZ31：一个被排除，一个可跑但不过

| 候选 | 预检 | 结果 |
|---|---|---|
| voigt（`include_L=False`、无 adsorption） | **18/18 `ok: false`** | 原因：低频感抗 0.01–0.373 Hz（12 点）需把 `R∥L` 放进第二并联节点 |
| nested `parallel`/L=1 | 通过 | `rel_med` 14.3%、`acf` 从基线 **0.909 → 0.587**、`budget=quick` |

**这条谱逼出的 schema 需求**

1. `excluded` 与前两种情形必须分开：voigt 是**没跑**（结构不可表示），nested 是**跑了但没过**。混成一个"不可用"会丢掉最有价值的信息。
2. `voigt_can_represent` 判的是感抗接点，它**不判容性最内层**。所以在涂层/钝化上 voigt 返回 `not_excluded` —— 这正是用户指出的洞，schema 必须有 `unknown` 一档来容纳"判据覆盖不到"。
3. 真实谱 `acf=0.587` 不能判 `model_inadequate`：残差里的结构来自测量与噪声，不是"候选集不足"。真实谱默认应为 `insufficient_evidence`。
4. `budget=quick` 这一档要能记录：同一候选换 `standard` 档可能改变结论（`bm_select` 已实测换 seed 会翻转判定）。

## 4. 覆盖矩阵：三条谱 + fixture

| 验收情形（用户列） | 由谁触发 | 仅靠三谱是否足够 |
|---|---|---|
| 结构排除与预检未知 | AZ31（排除）+ 涂层（unknown） | 排除够；`unknown` 需 fixture |
| 拟合未运行 / 失败 / 采样失败 | 钝化（未运行）、涂层（R̂ 14.21） | 需 fixture 补 `failed` |
| 贴底但参数不稳 | 涂层 `capacitive` | ✅ |
| 多候选可解释但不可区分 | — | **需 fixture** |
| 合成谱真值下全候选不足 | — | **需 fixture** |
| 真实谱证据不足、不得误报 | AZ31 | ✅ |

三谱只能证明 schema 能**装下**这些事实，不能证明状态转移逻辑正确。后者要靠人工构造的 fixture + 契约测试（0C）。

## 5. 对 registry 的直接约束（交给 0B）

由上面三节抽出，逐条可对照：

1. 输出必须分三块且互不推导：`screening` / `execution` / `evidence`。
2. `only_survivor` 只能由 `survivors == 1` 计算，**不得**成为 `evidence.status` 的取值。
3. `evidence.status` 五态；`identified` 当前不可达（无有效样本外证据），`model_inadequate` 需 `truth_known` 或 `noise_calibrated`。
4. 候选 = 拓扑 + 元件 + 挂载位置 + `init_profile` + `budget_profile` + `likelihood_profile`；`candidate_id` 必须覆盖全部这些字段。
5. L0 只能出 `excluded`；L1 只能出 `promote` / `defer` / `not_evaluated_budget`。
6. `param_stability` 与 `rms_rel` 并列保存，不合成单一分数。
7. `acf` 允许负值，不设 [0,1] 假设。
8. 候选集 manifest 在首次拟合前写入，之后不可追加。

## 6. 边界

- 本报告不评价任何候选模型的科学正确性，只记录 schema 需求。
- 三条谱是**合成谱 + 一条实测谱**的混合，样本量不足以支撑任何总体陈述。
- 涂层/钝化的真值来自文件名解析，属于生成器真值；AZ31 无真值。
- 未运行任何拟合，所有数字与所列产物一致，未重新计算。
