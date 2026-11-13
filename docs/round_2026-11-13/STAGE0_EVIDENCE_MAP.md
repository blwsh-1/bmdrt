# 阶段 0 证据映射：三谱 × 候选配置的实存矩阵与字段审计

状态：**只读归档**。本轮不重算、不重拟合、不改阈值、不动生产代码。
配套：`evidence/stage0_three_spectra/MANIFEST.json`（逐文件 sha256 + 源路径）。
生成脚本：`snapshot_stage0_evidence.py`（复制时逐条断言副本与源 sha256 相等）。

本文的目的只有一个：**让 registry 的 schema 接受外部检验**——历史证据能不能被装进去，
缺的字段是"没有"还是"有但为空"。不评价任何模型是否科学正确。

---

## 0. 如何独立校验

```bash
# 清单本身
python -c "import json;m=json.load(open('evidence/stage0_three_spectra/MANIFEST.json'));print(m['n_files'],m['total_bytes'])"
# 逐文件比对（源仍在 out/ 时才有意义；否则只能校验副本自身）
sha256sum evidence/stage0_three_spectra/*.json evidence/coating_voigt_pilot/*.json*
```

清单里的 `source_basename` 是**原始文件名**。涂层/钝化的原始文件名里直接写着真值参数
（`R1-[P2,R3-[P4,R5-P6]]-[Rs, Q_P2, n_P2, R3, Q_P4, n_P4, R5, Q_P6, n_P6]`），
重命名只是为了在目录里区分角色，真值以 `source_basename` 为准。

---

## 1. 覆盖矩阵：实存的是**参差网格**，不是完整 3×6

先给结论，因为它直接影响审计方法：**三条谱能凑齐的配置数不同**。
所以 schema 必须能表达 `absent`（该组合从未运行），不能假设"每条谱 × 每个候选"都有值。
这一点在 `circuit_registry.py` 里对应 `execution.not_run`，而不是用 `null` 冒充。

图例：`●` 有产物 / `—` 从未运行。括号内是该产物的文件名前缀。

| 谱 | voigt K=2 自由 σ | voigt K=2 固定 2% | nested cap. std 默认初值 | nested cap. std 数据驱动初值 | nested cap. std 真值初值 | nested par. std 默认初值 | nested cap. quick 默认初值 |
|---|---|---|---|---|---|---|---|
| 涂层 S1（`100926299.76`） | — | — | `●` (1113) | — | — | `●` (cap_control) | `●` (cap_smoke) |
| 钝化 P1（`1002185.22`） | — | — | `●` (cap_pass，rms 98.40%) | — | `●` (diag) | — | — |
| AZ31（`AZ31-4Li_100`） | `●` | — | — | — | — | `●` | — |

三条谱之外，`six_runs__family_fit_results.json` 覆盖**另外六条**谱（涂层 3 + 钝化 3），
每条都有 voigt K=2 固定 2% 与 nested cap. 数据驱动初值两列——这是唯一一处两族同口径配对。
60 条涂层 pilot（`coating_voigt_pilot/`）不含 S1，是分层抽样的另一批。

**这行"实存"本身就是一条 schema 事实**：同一条谱上，"换元件""换初始化""换预算"
三种变更分别发生在不同谱上，从未在同一条谱上做全交叉。所以 schema 不能靠"补跑"来补，
只能把 `absent` 记成事实。

---

## 2. 逐配置数值（全部抄自产物，未重算）

### 2.1 涂层 S1：只换最内层元件 / 只换预算

| 文件名前缀 | branch | budget | `rms_rel` | `rel_median` | `acf` | `R̂_max` | `ess_min` |
|---|---|---|---|---|---|---|---|
| cap_control | `parallel` | standard | 10.81% | 12.77% | **0.944** | 14.209 | 1.004 |
| cap_standard | `capacitive` | standard | 7.08% | 2.40% | 0.279 | 1.0047 | 411.0 |
| 1113（次日重跑） | `capacitive` | standard | 7.08% | 2.40% | 0.279 | 1.0047 | 411.0 |
| cap_smoke | `capacitive` | **quick** | 67.50% | 4.00% | 0.794 | **40.236** | 1.002 |

`cap_standard` 与 `1113` 不只是 `metrics` 相同：`diagnostics` 的 `rhat_max`
（1.0046667157241422）、`ess_min`（411.0416230655529）、`elpd`（−1898.5776397996142）
全部**逐位相同** ⇒ 同 seed 同预算的重跑可复现到这个程度，这是本批最硬的可复现证据。

`cap_smoke` 是关键：**同一候选、同一元件、只降预算**，`rms_rel` 7.08%→67.50% 且
`R̂` 1.0047→**40.24**、`ess_min` 掉到 1.0。
⇒ `budget_profile` 不进 `candidate_id`，manifest 就无法复现；而且"降预算"不是"更快的同一结论"，
它会同时毁掉幅度与采样。

`parallel` 那行的参数是 `Rs/R1/Q1/n1/R2/Q2/n2/R_ad/tau_ad`（9 个），
`capacitive` 是 `…/Q_b/n_b`（9 个）——**参数字面量不同名**，这是"元件换了"而不是"同参数重估"。

### 2.2 钝化 P1：只换初始化

| 文件名前缀 | 初始化 | `rms_rel` | `acf` | `R̂_max` | `ess_min` | `R1`（真值 9.28e6） |
|---|---|---|---|---|---|---|
| cap_pass（= 1113，两文件 `diagnostics` 逐位相同） | 默认（`R_inf=min\|Z\|`） | **98.40%** | 0.750 | **197.787** | **1.0005** | **108.4** |
| diag `truth_init` | **真值**（oracle） | 6.65% | −0.244 | 1.004 | 449.4 | 9.42e6 |

塌陷的形态可以直接读出来：`Q1=235309`、`n1=0.661`、`Q_b=235866`、`n_b=0.661`
——**两组参数几乎相等**，即两个相同 CPE 并联，`R1→108` 被挤到可忽略。
这不是"残差大"，是后验落到了另一个结构上，`R̂=197.8` 只是它的副产物。

两个文件 `rms_rel`/`acf`/`rhat`/`ess`/`elpd` 全部逐位相同 ⇒ 11-13 那次重跑
**复现了同一个塌陷**，不是新结果。

**`truth_init` 用的是文件名里的真值参数**（`diag_passivation_init.py:17-19` 硬编码，
`T.physical_to_unconstrained(truth, …)`），所以它是**上界诊断**，不是可选初始化策略。
生产上拿不到，不能记成"这个候选能拟合"。

钝化上"数据驱动初值有效"的证据**不在 P1 上**，而在 `six_runs` 的另外三条钝化谱
（6.034 / 6.093 / 6.031%，`acf` −0.441 / 0.025 / **0.501**）。其中一条 `acf=0.501`
正好压在 0.50 线上——审计时不要把它当"过"。

### 2.3 AZ31：6 个配置并列（这是唯一一处真正的 6 列同谱）

`az31__m4_real_nested_az31.json` 的 `rows` 就是 6 个配置，同谱、同 `budget=quick`、
`acf_base=0.909`、`rel_bound=0.067`：

| label | `rel_med` | `acf` | `acf_drop` | `R̂` | `div` | `sigma_re/im` |
|---|---|---|---|---|---|---|
| parallel/L=1 | **14.29%** | 0.587 | 0.322 | 1.0096 | 0 | 128.5 / 126.7 |
| parallel/L=2 | 14.41% | 0.598 | 0.311 | 1.0279 | 0 | 130.6 / 127.1 |
| series/L=1 | 75.95% | 0.767 | 0.142 | 1.0047 | 2 | 412.1 / 341.6 |
| series/L=2 | 62.25% | 0.868 | 0.041 | **1.3177** | 1 | 415.2 / 335.4 |
| voigt K=1 | 60.51% | 0.908 | 0.001 | 1.0022 | 0 | **`null`** |
| voigt K=2 | 59.56% | 0.907 | 0.002 | 1.0186 | 0 | **`null`** |

三处必须记下：

1. **`sigma_re/sigma_im` 在 voigt 两行是 `null`**，其余四行是数值。这就是"未知值保留为 null"
   的现成实例——但产物里**没有字段说明为什么**为 `null`。schema 必须给 `unavailable` 附 `reason`，
   否则审计者只能猜。
2. `passes_rel` 六行全 `False`，`rel_bound=0.067`。但 `series/L=2` 的 `R̂=1.3177` 与
   `div=1` 说明**它连执行层都没过**，却和 `parallel/L=1`（`R̂` 1.0096、`div` 0）共用同一个
   `passes_rel=False`。两层判定必须在 schema 里保持分开。
3. `m4_real_az31.json` 里 voigt 是 **18/18 `ok:false`**，`why` 字段写明"低频感抗 0.01–0.373 Hz
   （12 点）需把 `R∥L` 放进第二并联节点"。这是**没跑**的结构排除，与上面 voigt K=1/K=2
   的**跑了但 rel 60%** 完全不是一件事——同一个族在两个文件里分别占 `excluded` 和 `completed`，
   说明"排除"必须挂候选配置粒度（含 `include_L`/节点位置），不是族粒度。

---

## 3. 字段审计：你点名的四项

### 3.1 同一候选不同初始化，是否独立保存

**是，但有两种存法，且都不完整。**

| 情形 | 存法 | 问题 |
|---|---|---|
| 不同初始化/预算 → 不同文件 | `cap_pass` vs `diag`；`cap_control` vs `cap_standard` vs `cap_smoke` | 目录名承载语义（`_pass`/`_control`/`_smoke`），**文件内 `config` 块不含 `init_profile`**，只有 `seed`/`budget` |
| 同一条 pilot 记录内的敏感性 | `sensitivity`（`method:"data_init"`）与 `voigt_prod`/`voigt_fix2` 并列成子块 | 子块带 `method`/`note`，是全部产物里**唯一显式写了初始化方法**的地方 |
| 历史遗留 | `sensitivity_old_warmup`（4/60） | 存的是 M2 修订前的加力重采样，`note` 写明"非参数初值替换" |

⇒ `nested.json` 的 `config` 块里**没有 `init_profile` 字段**。目前靠文件名和目录名区分，
一旦目录被改名或产物被移动，初始化就不可追溯。这直接坐实 0A 的结论：
`init_profile` 必须进 `candidate_id`（`circuit_registry.py` 已按此实现，见 `make_candidate(init_profile=)`）。

### 3.2 指标口径是否可比

不可直接比，存在**四类口径差**：

| 口径 | 取值 | 在哪 |
|---|---|---|
| σ | 自由（标量绝对残差）vs 逐点 2%·\|Z\| | `voigt_prod` vs `voigt_fix2`；`sigma_from_data` 字段 |
| 预算 | `quick` vs `standard` | AZ31 全部 `quick`；S1 三档 `standard` + 一档 `quick` |
| `target_accept` | `nested.json` 实为 **0.98**（`nested_fit.VALIDATION_STANDARD` 硬编码）vs 生产 `bm_real.KW` 的 0.9 | `nested.json` 的 `config.target_accept_override=None` **只表示没有额外覆盖**，不等于"走生产 0.9"；有效值看 `config.budget_kw.target_accept=0.98`。pilot 里 `target_accept_override` 则记成 0.98（那也是硬编码的同一档，不是覆盖） |
| `ratio` 的分母 | 收敛子集（14/60、40/60）vs 全 60 条 | FINDINGS 表用收敛子集；全 60 条算是 6.47 / 1.13 |

最后一行值得单独说：`ratio` 中位在"收敛子集"口径下是 3.36（自由 σ）/ 1.03（固定 2%），
在"全 60 条"口径下是 **6.47 / 1.13**。两个都对，但**分母不同**。
所以任何 `ratio` 入库都必须同时带 `n_used` 和 `n_total`，否则跨谱汇总会混口径。

`elpd` 在两类产物里**都有**（`nested.json` 的 `diagnostics.elpd` 带 `elpd_se`，
AZ31 的 `rows[].elpd` 带 `se`），但 `nested.json` 同时写了
`elpd_valid_for_predictive_comparison: false` 和 `elpd_note`。这是本批**最值得照抄的字段设计**：
数值保留、可用性单独标记、原因随值走。schema 应把这个模式推广到所有"算了但不可用"的量。

不可比的点因此不在"有没有 elpd"，而在**它被明确标为不可用于预测比较**——
跨候选比 `elpd` 在现有 σ 口径下一律无效，与 §0 的结论一致。

### 3.3 未知值是否保留为 null / unavailable

**部分保留，缺失本身没有记账。** 逐条：

| 字段 | 现状 | 判断 |
|---|---|---|
| `nested.json` 的 `R̂` / `ess` / `div` | 在 `diagnostics.rhat_max` / `ess_min` / `divergences`，**不在 `metrics`** | 有值，但同名概念分散两处：`metrics` 是拟合质量、`diagnostics` 是采样质量。按 `metrics` 找会误判为"缺失" |
| `nested.json` 的逐参数区间 | `params.<name>.ci90` / `rhat` / `ess` **齐备** | 后验区间在 nested 链上是全的，`observation_block` 要求的 `ci90` 原生就对得上 |
| pilot `nested` 的逐参数区间 | 只有 `median`，**无 `ci90`** | 60 条这一批确实没有区间 ⇒ `param_identifiable` 只能记 `unavailable`，且必须带原因（"pilot 只落中位数"），不能记成"不可辨识" |
| `derived.L_ad_H` / `f_ad_Hz`（capacitive 候选） | `null` | 这是**"不适用"**（纯 CPE 支路没有吸附电感）；而 pilot 缺失字段是**"没算"**。同一 `null` 承载两种语义，且产物里都没标注 |
| AZ31 voigt 行的 `sigma_re/sigma_im` | `null` | 保留了，**无 `reason`** |
| AZ31 真实谱的 `truth` | 无此字段 | 正确（无真值不应伪造空对象） |
| pilot `sensitivity` | 46/60 存在（= 46 条 unusable 那批） | 缺失的 14 条是没触发该分支，语义由 `verdict` 可推，可接受 |

⇒ 结论要改写成两条：

1. **最有价值的反例不是"缺失"，是"同一个 `null` 有两种含义"**（不适用 / 没算）。
   schema 必须给每个缺值一个 `null_reason ∈ {not_applicable, not_computed, failed}`，
   否则审计者只能靠上下文猜——而上下文在跨产物汇总时就丢了。
2. `circuit_registry.py` 的 `observation_block` 要求 `ci90` 或 `flag` 二选一、不接受标量，
   这一点与 `nested.json` 的 `params` 结构天然吻合；但它**没有强制"缺值必须带原因"**，
   这是本轮发现的真实缺口。

### 3.4 原始结果能否追溯

能，链条完整：

```
MANIFEST.json
  └─ evidence_path（本仓库路径）
     ├─ source_path（out/ 原始路径）
     ├─ source_basename（含真值参数的原名）
     └─ sha256（副本 = 源，复制时断言相等）
```

断链的只有一处：`out/` 不入版本库，所以**克隆仓库的人拿不到 `source_path` 去比对**，
只能靠 `sha256` 与副本自身一致。这是 `out/` 策略的必然代价，不是本轮的疏漏；
若要根治需把谱原始 CSV 一并入库（`测试数据/` 在仓外，本轮未做）。

---

## 4. 我上一轮报告里需要更正的两条

以下两条是我在对话里说错的，产物不支持，按原样更正。

### 更正 1：「60 条 pilot 没做逐条参数恢复」——错

`results.jsonl` 里 `nested.param_err` 与 `nested.RsR1` **60/60 全部存在**，
`COATING_VOIGT_PILOT_FINDINGS.md` 也早已写了"参数除 `Rs` 外恢复 2–20%"。

实际数（我重算，60 条全涂层）：

| 量 | 值 |
|---|---|
| `param_max_err` 中位 | **2.934** |
| `param_max_err` > 1.0 | **48/60** |
| `param_max_err` < 0.2 | **1/60** |
| `Rs` | 中位 2.65、p90 8.05、max 13.12 |
| `R1` / `n1` / `n_b` | 0.022 / 0.023 / 0.052 |
| `Q1` / `Q2` / `R2` / `n2` | 0.100 / 0.176 / 0.099 / 0.037 |
| `Q2` p90 / `R2` p90 | **4.10** / **1.34** |
| `Rs+R1` | 中位 2.01%、p90 21.75% |

所以"2–20%"是**中位数叙事**，尾部远不止：`Q2` 的 p90 是 4 倍、`R2` 是 1.34 倍。
准确说法是：**该族 60 条里，参数整体可信的只有 1 条（`param_max_err<0.2`），
其余被 `Rs` 与中频 `R2∥Q2` 的脊拖累**。真正缺的不是参数恢复，是**后验区间判据**
（`param_identifiable` 0/60，字段未实现）。

### 更正 2：「换数据驱动初值 13s 收敛、R̂ 1.003、6.65%」——把两件事并成了一件

13.1s / `R̂` 1.003 / 6.65% 属于 **`truth_init`**，用的是文件名里的真值参数（oracle）。
数据驱动初值的证据是另外三条钝化谱（6.03–6.09%，`six_runs`），**不是 6.65% 那条**。
P1 这条谱上，默认初值 98.40% 塌陷，唯一收敛的是 oracle 初值 ⇒
**P1 目前没有任何生产可达的初始化解法**。这条不能再写成"换个初值就好了"。

---

## 5. 对 registry 的直接影响

按上面四项审计，`circuit_registry.py` 现状与应改之处：

| 发现 | registry 现状 | 处置 |
|---|---|---|
| `absent`（从未运行）与 `null`（跑了但无值）是两回事 | 有 `execution.not_run`，无字段级缺值语义 | 本轮未改；已登记为待补 |
| 同一个 `null` 承载"不适用 / 没算 / 失败"三种语义 | `observation_block` 要求 `ci90` 或 `flag`，无 `null_reason` | **应补 `null_reason` 枚举**（待批） |
| `elpd_valid_for_predictive_comparison` + `elpd_note` 是好模式 | 无对应字段 | **应照抄**：所有"算了但不可用"的量都带可用性标记与原因（待批） |
| `init_profile` 必须进 `candidate_id` | 已实现 | 一致 |
| `budget_profile` 必须进 `candidate_id` | 已实现 | 一致 |
| `target_accept` 属于 `budget_profile`，不是全局常量 | 未见登记 | **应进 `budget_profile`**（待批） |
| `ratio` 必须带 `n_used`/`n_total` | 无此字段 | **应补**（待批） |
| 排除挂配置粒度、不挂族粒度 | `l0_precheck(topology, f, Z)` 按族调用 | **粒度不对**，应改为按候选配置（含 `include_L`/节点位置） |
| 两层判定（执行层 / 结构层）分开 | `execution` 与 `observation` 已分块 | 一致 |
| 后验区间在 nested 链上原生存在 | `observation_block` 已支持 `ci90` | 一致 |

七条属 schema 改动，**本轮一律未动**：`circuit_registry.py` 与 29 条契约用例保持原状，
等你审计完给结论再改。这样本轮的证据与代码可以分开检验。

---

## 6. 边界

- 本文所有数字抄自所列 JSON，除 §4 的四项统计（明示重算口径）外未重算。
- 三条谱中两条是合成谱（真值来自文件名），一条是实测谱（无真值）。
- 样本量 3 条，不支持任何总体陈述；覆盖矩阵的参差本身是事实，不是缺陷。
- 未运行任何拟合，未改任何阈值，`STAGE0C_SCHEMA_CONTRACT.md` 的 29 条用例与
  `circuit_registry.py` 未改动。
