# 两入口口径合并（2026-11-16）

> 范围：只做**外围整合**。不改拟合路径、不改 `bm_core` / `bm_top` / `bm_real`、
> 不改 M-b 阈值与 §13 `0/3` / §14 `4/14` / §17 `0/10`、不改 evaluator 的 E4/E5。
> 目标不是把两个入口合成一个，而是让它们在同一科学口径下调用同一套基础设施。

## 1. 问题定位（先证伪"采样内核重复"）

`docs/round_2026-11-16/entry_equiv_probe.py` 把 `bm_top.fit_top` 换成只记录实参的桩，
让两条路径各抓一次真实调用：

```
standard 档（AZ31-8Li_50，50 点）
A = nested_fit.run             13 个实参
B = circuit_runner/fit_nested  13 个实参
  同名同值 13 项：branch / layers / seed / n_warmup 2000 / n_samples 1000 /
                  num_chains 2 / target_accept 0.98 / max_tree_depth 10 /
                  sigma（(2,50) 逐元素相同）/ sigma_floor 0.0 / progress False
  差异 1 项：sigma_from_data   A=True  B=<缺省 False>
```

所以**采样内核不是重复实现的根源**——两边调的就是同一个 `fit_top`。真正重复在外围：

| 重复点 | `nested_fit.py` | `circuit_runner.py` | 实测后果 |
|---|---|---|---|
| `metrics()` | 自写 13 字段 | 自写 5 字段（同 §14 式） | 字段集合不同，跨入口读产物会 KeyError |
| 逐点 σ | `pct/100·\|Z\|` | 常数 `0.02·\|Z\|` | 数值相同，但两处各写一份 |
| 预算表 | `VALIDATION_STANDARD` | 自己一份 `BUDGETS` | **`quick` 同名不同值**：800/400/10 vs 300/300/8 |
| `sigma_from_data` | 传 `True`（elpd 标不可比） | 不传 ⇒ 底层标 `False` | 同一口径，两个入口对 elpd 的说法相反 |

`quick` 那一行是**双份预算表的直接代价**：`standard` 数值相同纯属巧合（两边都写
2000/1000/2/0.98/10），`quick` 一跑就暴露。

## 2. `sigma_from_data` 的语义审查（本轮最重要的一条）

### 2.1 三个概念必须拆开

| 概念 | 问的是什么 | 判定字段 |
|---|---|---|
| σ 泄漏 | `σ_j` 是否含**被留出点自身**的观测值 | `sigma_from_data` / `sigma_leak` |
| 先验尺度依赖 | `zmax=max\|Z\|` 由本批观测算出、留出 j 时未重算（R21） | `prior_scale_data_dependent` |
| 候选间同噪声 | 各候选是否共享**同一组**固定 σ | `noise_shared_across_candidates` |

`σ_j = 2%·|Z_j^obs|` **同时**满足「由观测算出」与「外部规定的相对误差模型」：
2% 是外部给定，但 `σ_j` 确实含被预测点自身的幅值。对 ELPD 起作用的是后者——
留出第 j 点时 `σ_j` 无处可算，似然尺度用了要预测的那个数，算出来不是
`p(Z_j|Z_{-j})`。这与「σ 由残差反向估计」不是一回事，结论却相同：
**单个 `elpd_loo` 不能当样本外预测证据**。

反过来，共享同一组 σ 让 `Δelpd` 的差异来自模型而非噪声口径，这一点**成立**；
但 R21 仍在，所以 `Δelpd` 只能作**候选间相对参考**。两条结论独立，
不许用其中一条替代另一条。

### 2.2 结论：`valid_for_predictive_comparison` 只答 L1，不答"能否当证据"

`bm_top.fit_top` 的这个字段历史上就是 `not sigma_from_data`（只答 σ 泄漏）。
它**不能**被读成"可以当验收证据"，因为 R21 与它无关、恒成立。

因此统一定义（`spectrum_metrics.elpd_validity`）：

| 字段 | perpoint | free |
|---|---|---|
| `sigma_leak` | True | False |
| `valid_for_predictive_comparison`（与 `fit_top` 同义） | False | True |
| `prior_scale_data_dependent`（R21） | True | True |
| `comparison_scope` | `relative_only` | `relative_only` |
| `reason` | L1 + L2 | 只有 L2 |

`circuit_runner` 写进 `observation_block.predictive.valid_for_comparison` 的值是
`not sigma_leak and not prior_scale_data_dependent` ⇒ **两口径都 False**。
这与它改造前的硬编码 `False` 行为一致，但理由从"只有 R21"变成"两层都写全"。

### 2.3 传播链修好了

改造前 `bm_select.fit_nested` 不把 `sigma_from_data` 透传给 `fit_top`，
所以 perpoint 口径下底层返回 `elpd_valid_for_predictive_comparison=True`（错标），
只是被 runner 上层的硬编码 `False` 掩盖了。现在：

- `bm_select.fit_nested(..., sigma_from_data=False)` 新增形参，**默认 False = 生产
  路径逐位不变**（`_select_one` 等调用点一律不传，`test_entry_parity.py` T6 守着）；
- `circuit_runner` 按 `likelihood == "perpoint"` 传 `True`；
- `fit_candidate` 用底层自报值参与核对：若口径与标注不一致，
  `elpd_validity.derived_mismatch=True` 落盘，不静默。

`fit_voigt` 走 `bm_real.fit_one`，底层**没有**这个字段，属"缺失"而非"错标"，
由公共判定按口径补齐。

## 3. 四项改动

### 3.1 `spectrum_metrics.py`（新增，纯函数，39/39 自检）

不 import `bm_top` / `bm_core` / `bm_real` / `bm_select`，不跑拟合、不读写文件。
唯一数值依赖是 `bm_experiments.acf_max`（§14 同式的残差自相关）。

| 函数 | 作用 |
|---|---|
| `metrics(f, Z, Zm)` | **超集**：nested 13 字段 + runner 5 字段（`bands`/`bands_pct` 同值别名） |
| `sigma_perpoint(Z, pct=2.0)` | 单一实现，返回 `(2, n)`；`pct` 是百分数；零/非有限显式拒绝 |
| `elpd_validity(likelihood, sigma_from_data=None)` | 三层语义分开报，见 §2.2 |
| `budget_actual(budget, tier)` | 五键数值 + 档名 + `tier_shared_across_entries` |
| `STANDARD` | §13/§14 冻结档 **唯一来源** |

字段公式、分母、频段边界**逐字未改**：

- `rms_rel = 100·√mean(concat(d.real, d.imag)²)`，`d=(Z_fit−Z_obs)/|Z_obs|`（= `Rwp/√2`）
- `rms_re`/`rms_im` 是绝对残差（Ω）
- 频段 `[0.01,1)/[1,100)/[100,1e5)`，无点的段给 `None`（不是 0）
- `acf = max(acf_max(Re), acf_max(Im))`，对残差整体缩放不变

### 3.2 预算

- `standard` 两入口都引用 `spectrum_metrics.STANDARD`（`circuit_runner.BUDGETS["standard"]`
  与 `nested_fit.VALIDATION_STANDARD` 都是它）。
- `quick` **保留各自历史数值、不追溯修改**：runner 300/300/2/0.9/8，
  nested_fit 走 `bm_select.QUICK` 800/400/2/0.9/10。
- 每份产物落 `budget_actual`：五项数值 + `tier` + `tier_shared_across_entries`。
  `standard` 为 `True`，`quick` 为 `False`——**不许靠档名推断"两次同预算"**。
- `bm_select.BUDGETS["standard"]` 是 ta=0.9 的另一套（§9.4.4 A2 档），**保持不动**，
  T6 有断言防止它被 standard 口径污染。

### 3.3 落盘位置

| 产物 | 落什么 |
|---|---|
| `out/compare/<stem>/fits/<cid>.json` | `budget_actual`（完整）、`elpd_validity`（三层） |
| `out/compare/<stem>/observations.jsonl` | `observation_block`：`predictive`（elpd/se/valid/reason）+ `budget_params`（五键数值）；档名与 scope 进 `notes`（registry 的这两块是白名单 schema，放不下） |
| `out/nested_fit/<stem>.nested.json` | `config.budget_actual`、`diagnostics.elpd_validity`；`diagnostics.elpd_valid_for_predictive_comparison` 保持与 `fit_top` 同义 |

## 4. 六项验收

`python test_entry_parity.py` ⇒ **105/105**（不跑 MCMC，秒级）。

| 编号 | 条件 | 结果 |
|---|---|---|
| T1 | 两入口调同一份 `metrics()` | 三处返回逐字段相同；主式只出现在 `spectrum_metrics.py`；两入口均为 `return SM.metrics(...)` 委托 |
| T2 | nested 13 字段全保留、数值逐项一致 | 三种残差形态（纯比例/带结构/端点离群）对**独立硬算式**逐字段 ≤1e-12 |
| T3 | 同一份 σ、逐元素一致 | `SM.sigma_perpoint` = `NF.make_sigma` = runner 旧式 `0.02·\|Z\|`，`np.array_equal` |
| T4 | `sigma_from_data` 语义/传播/判定一致 | 两口径的 `sigma_leak`/`valid`/`R21`/`scope`/`reason` 逐项断言；runner 的 `predictive` 在 free 口径下仍 False 且理由不含 L1；`derived_mismatch` 有独立断言 |
| T5 | `standard` 唯一来源 + `quick` 实际预算落盘 | 三处常量相等；`quick` 两入口数值确实不同；`budget_actual` 五键 + tier 标记；缺键报错 |
| T6 | 采样器实参不变、生产默认不变 | `fit_nested.sigma_from_data` 默认 `False`；`bm_select` 内只有一处定义一处透传；`QUICK` 与 `BUDGETS["standard"]` 未被改；`bm_core`/`bm_top`/`bm_real` 无 diff |

外加**差分测试**（同一条真实谱 AZ31、standard 档、拦截 `fit_top` 不采样）：
两入口传给 `fit_top` 的实参逐键一致（`sigma_floor` 缺省 0.0 视为等价）、
`sigma` 数组逐元素一致、`sigma_from_data` 两边都是 `True`、预算五项一致、
ELPD 判定相同。

## 5. 端到端抽查（quick 档，探索性，不作证据）

```
bmdrt.py compare out/fig3/inputs/az31.csv --ks 1 --layers 1 --budget quick
  → out/compare_consol/az31/fits/*.json
     budget_actual = {300,300,2,0.9,8, tier=quick, shared=False}
     elpd_validity: sigma_leak=True, valid=False, scope=relative_only, mismatch=False
     metrics 14 键（13 nested + bands）

nested_fit.py out/fig3/inputs/az31.csv --budget quick
  → out/nested_consol/az31.nested.json
     config.budget_actual = {800,400,2,0.9,10, tier=quick, shared=False}
     diagnostics.elpd_validity: sigma_leak=True, scope=relative_only
     rms_rel 2.77%（与上一轮 standard 档同谱同值）
```

`mismatch=False` 说明 `fit_nested` 的透传生效：底层自报与口径判定已经一致。

## 6. 边界与未闭合

- **`quick` 档仍不统一**：按裁决保留历史数值。若将来要统一，另开版本号，
  不顺手调整（否则历史 quick 产物不可比）。
- **voigt 侧没有 `sigma_from_data` 原生标注**：`bm_real.fit_one` 走 `bm_core.fit`，
  无此概念；公共判定按口径补齐，不会有"错标"，但是"补标"。
- **`identified` 仍不可达**：`out_of_sample_valid=True` 被 L1/L2 同时封死，
  与本次整合无关。解锁需要裁 σ/先验尺度口径或 CAL-01 提供 `noise_calibrated`。
- **`R21_REASON` 常量保留**：`observation_block.predictive.reason` 现在由
  `elpd_validity` 生成（含 R21 原文），常量作为历史引用保留，不再被写入。
- 未改 evaluator 的 E4/E5；未改 M-b 阈值；未改 §13/§14/§17 判定。

## 7. 复现

```bash
D:/autoeis_env/Scripts/python.exe spectrum_metrics.py       # 39/39
D:/autoeis_env/Scripts/python.exe test_entry_parity.py      # 105/105
D:/autoeis_env/Scripts/python.exe circuit_runner.py --selftest   # 51/51
D:/autoeis_env/Scripts/python.exe cmd_compare.py --selftest      # 9/9
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-16/entry_equiv_probe.py
D:/autoeis_env/Scripts/python.exe tools/repo_map.py --check      # OK
```
