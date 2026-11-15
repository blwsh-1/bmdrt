# circuit_evaluator.py：观测 → evidence 状态的推导层（2026-11-16）

> 三层架构的最后一块。`circuit_registry` 回答"能不能被表达/签发/复核"，
> `circuit_runner` 负责"跑出观测"，**本文件负责"把观测推成一个状态"**。
> registry 的 `evidence_block` 是**校验器**（状态要人传，它只负责"你声称 `identified`
> 但 `out_of_sample_valid=False` 就报错"），推导一直是缺的。

## 1. 设计约束

1. **纯函数**：只 import `circuit_registry`，不 import `bm_*`、不拟合、不碰 numpy。
   输入是 runner 落下的 `plan.json` + `observations.jsonl`。
2. **只判"能不能谈"，不判"谁是对的"**：`identified` / `model_inadequate` 当前口径下
   不可达，本模块**不会产出**它们，并把理由写进 `unreachable_states`。
3. **争议规则显式标 `disputed`**：每条规则带 `firm: true|false`；争议项另带
   `pending_decision`，说明它为什么需要人裁决。改它等于改验收口径，不藏在常数堆里。
4. **冲突必须留证**：两个判据给相反答案时，`tie.criteria_conflict=true` 并进 `caveats`，
   不静默选一个。

## 2. 规则表

| 规则 | 内容 | 性质 |
|---|---|---|
| **E1** | `sampling_status=failed`（R̂>1.05 或 div>0）不参与比较 | firm |
| **E2** | 预检 `excluded` 不进比较；`unknown` 只记状态（"没有判据"≠ 通过，也不当排除） | firm |
| **E3** | viable < 2 ⇒ `insufficient_evidence`（`single_viable_candidate` / `no_viable_candidate`），依据 R12 | firm |
| **E7** | 最优候选 `rms_rel` > `rms_accept`（默认 **10.0%**，沿用 §14 `RMS_MAX`）⇒ `insufficient_evidence`（`no_candidate_within_acceptance`） | **disputed** |
| **E4** | 前两名 `rms_rel` 差 ≤ `tie_pp`（默认 **1.0pp**）且 `Δelpd` 不显著 ⇒ `non_identifiable` | **disputed** |
| **E5** | 否则 ⇒ `weak_preference`（"有倾向、但不足以 `identified`"） | firm |
| **E6** | `identified` / `model_inadequate` 一律不产出，附不可达理由 | firm |

`Δelpd` 显著的判据与 `bm_select` 同值：`|Δ| ≥ 4` **且** `|Δ|/se ≥ 2`。

E4/E7 的 `disputed` 标记**保留**：两个参数已于 2026-11-16 裁决（1.0pp / 10%），
但它们仍是**口径选择**而不是客观事实，改它们依然等于改验收标准。

### 2.1 输出要写「确认到哪一步」，不能只给一个标签

`evaluate()` 除 `status` 外还产一个 `confirmation` 块，三段：

| 段 | 内容 | 纪律 |
|---|---|---|
| `confirmed` | 描述性事实：候选集几个、几个被预检排除、几个过采样门槛、最优 rms 多少、两条口径各说什么 | **不得**出现「模型正确」「最优模型是 X」「identified」——自检有断言 |
| `not_confirmed` | 每条写 `why`（为什么不足）+ `needs`（缺什么才能补），`needs` **不许为空** | 防止「不可达」被读成「没做」 |
| `scope` | 本次状态的作用域：只对已登记候选集、只对该口径 | 防止 `non_identifiable` 被读成「所有电路都不行」 |

**为什么必须这样**：`identified` 现在不可达。若输出只印一个 `weak_preference`，
读者分不清「已确认残差地板、并列关系、采样质量」与「什么都没确认」。
把已确认的摊开写、把不充分的连「缺什么」一起写，「停在这里」才是一个**有内容的结论**。

`not_confirmed` 的固定四项（口径不变就一直报）：

| item | 为什么不足 | 缺什么 |
|---|---|---|
| `identified` | `out_of_sample_valid` 不能为 True（逐点 σ / `zmax` 依赖观测，R21） | 裁 σ/先验尺度口径（会重算 §13/§14/§18/§19 的 elpd） |
| `model_inadequate` | 真实谱无 `truth_known` / `noise_calibrated` | CAL-01（开-短路+已知 RC）／合成数据 |
| `out_of_sample_evidence` | Δelpd 只是候选间相对参考 | 同上 |
| `candidate_set_completeness` | 扩散/吸附 `wired=False` 且无槽位（R12） | 给拓扑加槽位 |

另有两条**条件触发**：`sampling_completeness`（有候选采样不达标时）、
`confirmed.two_criteria`（当前两名都存在时，报告两条口径各说什么、是否冲突）。

示例（`3E-1_HER_exp1_0005mA`，真实输出）：

```
**状态**：`weak_preference`（best_leading_but_not_identified）
**已确认（描述性事实，不含「模型正确」）**
- 候选集 3 个（全部来自 registry 登记，不是临时拼的）
- 3/3 个候选通过采样门槛（R̂≤1.05 且 0 发散）
- 最优候选 `C-96c2bc452a` 的 rms_rel = 6.12%，在 §14 接受线内
- 残差口径：前两名差 0.32pp vs 重复性尺度 1pp ⇒ 不可分；elpd 口径：显著
  （Δ=61.48、Δ/se=4.88）；**两者冲突，无仲裁规则**
**不充分 / 不可达（以及缺什么）**
- `identified`：… ｜ 需要：裁决 σ/先验尺度口径（会让 §13/§14/§18/§19 的 elpd 全部重算）
- `model_inadequate`：… ｜ 需要：CAL-01 提供 `noise_calibrated`；`truth_known` 只能来自合成数据
- `out_of_sample_evidence`：… ｜ 需要：同上
- `candidate_set_completeness`：… ｜ 需要：给拓扑加扩散/吸附槽位
**作用域**：只对已登记候选集 3 个；不陈述「真实电路是什么」
```

### 两个 disputed 参数为什么必须标出来

- **`tie_pp`**：等价于"多近算分不开"。默认 **1.0pp**，现在有实证锚点：

  | 同条件重复（含重新装夹、跨时间） | rms% | sd | 极差 |
  |---|---|---|---|
  | `3E-3 full_cell 0100mA` × 4 | 2.99 / 3.46 / 4.55 / 5.32 | **1.05pp** | 2.33pp |
  | `3E-2 full_cell 0100mA` × 3 | 2.26 / 2.31 / 3.25 | 0.56pp | 0.99pp |

  **同一模型重复测，`rms` 本来就会差 1pp 量级**；两候选的差小于这个量级时"谁更低"不可分辨。
  所以 1.0pp ≈ 1 个重复性 sd，是"刚好能分辨"的**下限**而非上界，保守可放 1.5–2.0pp。
  **不能放到 10pp**：那会把 9pp 的真实差异也判"分不开"，等于取消 E4。
  `--tie-pp 2.0` 会把同一份输入从 `weak_preference` 改成 `non_identifiable`
  （自检里有这条对照）。
- **`rms_accept`**：数值沿用 §14 的 `RMS_MAX=10.0`（**没有新造阈值**），但**用法是新的**
  ——`model_inadequate` 被 `truth_known/noise_calibrated` 挡住，本规则是它的**代理**。
  **2026-11-16 用户裁决：定 10%**。传 `--rms-accept 1e9` 仍可关闭。

## 3. 实测（现有 4 个 runner 目录）

| 目录 | 状态 | 理由 |
|---|---|---|
| `3E-1_HER_exp1_0005mA` | `weak_preference` | 3 viable；nested L=1（6.12%）领先 L=2（6.45%）0.32pp，但 `Δelpd=61.5`、`Δ/se=4.88` 显著 ⇒ **两口径冲突**已记录 |
| `3E-1_HER_exp1_0050mA` | `insufficient_evidence` | 该谱 voigt 被预检 `excluded`，nested 只剩 1 个 viable |
| `AZ31-8Li_50` | `insufficient_evidence` | nested L=1 过（5.99%），L=2 `R̂=1.161` 被判 `failed`（E1）⇒ viable 只剩 1 |
| `测试数据/腐蚀` 那条 | `insufficient_evidence` | 两个 viable 都 19–20%，超 10% 接受线（E7）⇒ "分不开"不能读作"候选集完备" |

第三条是 E1/E3 联动的实例：**看起来两个候选都跑完了，但其中一个采样不达标，于是这条谱其实没有可比较的第二候选**。

## 4. 用法

```
D:/autoeis_env/Scripts/python.exe circuit_evaluator.py --eval out/circuit_runner/<stem>
D:/autoeis_env/Scripts/python.exe circuit_evaluator.py --eval "out/circuit_runner/*"
D:/autoeis_env/Scripts/python.exe circuit_evaluator.py --selftest        # 52/52
```

产物：每个 runner 目录下多两件 —— `evaluation.json`（含 rules / caveats / tie / 输入指纹）
与 `evaluation.md`（人读版）。

**目录展开不能只靠 glob**：`测试数据/` 的谱文件名带 `[P1,P2-R3]`，glob 会把它当字符类，
整个目录会被静默漏掉。`--eval` 对 `.../*` 一律走 `os.listdir` 再过滤。

## 5. 这层**不**解决什么

- 不解 `identified`：需要 `out_of_sample_valid=True`，被 §14 的逐点 σ 与 R21 的先验尺度
  两头堵住；解锁要先裁 σ/先验尺度口径，**不是放宽这里**。
- 不解 `model_inadequate`：需要 `truth_known` 或 `noise_calibrated`，真实谱两者都没有；
  `E7` 只是代理，已标 `disputed`。
- 不替人做仲裁：两条口径冲突时只**留证**（`tie.criteria_conflict`），不选一个当结论。
- 不给"物理来源"：它只说"在已登记候选集内能不能区分"。仪器/样品归因仍需要独立信息
  （CAL-01 开-短路校准、E1 同状态重复、多频窗）。

## 6. 边界

未动 `bm_*.py`、任何阈值、§13 `0/3`、§14 `4/14`、§17 `0/10`、`bmdrt.py` 接入。
`rms_accept = 10%`、`tie_pp = 1.0pp` 均已于 2026-11-16 裁决（依据见 §2）；
改它们仍然是改验收口径，必须重新登记。

附带订正一处措辞：`circuit_runner.R21_REASON` 原先写成"`sigma_from_data=True` 导致
elpd 不可用"，实际 `bm_select` 的跨族口径传 `sigma=None`（σ 被采样），那一支
`elpd_valid_for_predictive_comparison` 反而是 `True`；真正的原因是先验尺度
`zmax=max|Z|` 依赖本批观测。已改字符串，**早于本次修的冒烟记录里仍是旧措辞**，
按"原文保留"处理，不追改。
