# `bmdrt compare`：多族候选自动并列 + 统一图

> 建立于 2026-11-16。回答三个质疑：**拓扑要不要手动选 / 默认是不是快速档 / 图为什么不统一**。
> 不改 `bm_*.py` 的默认路径（已用逐位对拍验证），不改 §13/§14/§17 判定。

## 1. 三个问题的实际状态

| 问题 | 上一轮的实际做法 | 这一轮的修正 |
|---|---|---|
| 拓扑要手动选吗 | `auto` **只有 voigt 一族**；nested 得手动跑 `nested_fit.py`，两套图、两套指标 | 新增 `bmdrt compare`：voigt K∈{1,2,3} 与 nested L∈{1,2} **一起跑、一起判、画一张图** |
| 默认是不是快速档 | `config.json` 的 `quick=false`、`KW=3000/1000/3链`——**默认本来就是正常档**；上一轮是我在命令行加了 `--quick` | `compare` 默认 `--budget standard`（2000/1000/2 链/ta=0.98）；`quick` 只在显式指定时用，且输出标「探索性」 |
| 图统一吗 | **不统一**：`bm_real.plot_real` 6 子图（绑死 voigt 的 DRT/PPC 中间量），`nested_fit.write_plot` 3 子图 | 新增 `bm_figure_unified.plot_fit`：只依赖 `(f, Z, curves)`，一条谱一张 2×3 图，所有候选叠在同一组坐标里 |

**没有改动的**：`bmdrt auto` 仍走 `bm_real`（voigt、K 由 DRT 预判、L 由机制判断），
`nested_fit.py` 仍用自己那套三联图。把三者合并会改两条已有 CLI 的输出，属于另一个决定。

## 2. 用法

```
python bmdrt.py compare 谱.csv                          # 默认 standard + 逐点 σ
python bmdrt.py compare 谱.csv --ks 1,2 --layers 1,2
python bmdrt.py compare 谱.csv --budget quick           # 只作预览
python bmdrt.py compare 谱.csv --likelihood free        # 与 auto 同口径（σ 被采样）
python bmdrt.py compare 目录/ --limit-files 3
```

产物：`out/compare/<stem>/`
- `compare.png` —— 统一 6 子图（Nyquist / Bode 幅值 / Bode 相位 / 相对残差 / 残差 Re / 残差 Im）
- `compare.json` —— 状态 + 候选指标 + 输入指纹
- `evaluation.md` —— `circuit_evaluator` 的完整判定（含「已确认 / 不充分」两段）
- `observations.jsonl` / `fits/*.json` / `plan.json` / `run_meta.json` —— runner 的原始记录

## 3. 两个口径，不能混读

| 口径 | 含义 | 谁在用 |
|---|---|---|
| `perpoint`（compare 默认） | 固定 `σ_j = 2%·|Z_j|`，σ 不进采样 | §14 / `nested_fit --sigma perpoint` / §13 验收 |
| `free`（auto 默认） | σ 被采样（自由估计） | `bmdrt auto` / `bm_select` 跨族 |

**为什么 compare 默认 perpoint**：`|Z|` 跨几个量级的谱（腐蚀 2e3–1e6 Ω）在 `free` 下
σ 会被拉到 1.3e4，高频点几乎不进似然——同一模型同一谱，`free` 给 18.65%、
`perpoint` 给 4.76%。这不是模型差异，是噪声口径差异。

`--likelihood` 会进缓存键：同一候选在两种口径下是两条独立记录，不会互相顶替。

## 4. 为了打通 perpoint，改了什么（默认路径逐位不变）

voigt 族原来**没有**固定 σ 的路径（`fit_one` 里写死 `sigma=None`），一路撞到三个硬编码：

| 位置 | 症状 | 改法 |
|---|---|---|
| `bm_real.fit_one` | `B.fit(sigma=None)` 写死 | 加 `sigma=None` 形参并透传 |
| `bm_real.fit_one` → `posterior_predictive_check` | 固定 σ 时样本里没有 `s_re/s_im`，PPC 直接 raise | 传 `sigma=sigma`，`s_re/s_im` 用给定值取中位（**不回退到残差 std**） |
| `bm_real.loglik_matrix` | 同上，`float(sigma[0])` 遇数组即 TypeError | 支持逐点数组 σ |
| `bm_core.posterior_predictive_check` | 同上 | 同上 |
| `bm_select.fit_voigt` / `fit_nested` | 都把 `sigma=None` 写死 | 加形参并透传 |
| `circuit_runner.build_candidates` | `include_L=True` 时没挂 `series_inductor` 槽位 | 补 `("series_inductor","rl_series")` |

**默认路径逐位不变的证据**：同一谱（3E）、同一命令（`auto --krange 1,3 --quick`）在改动前后：

```
K=2  rms_re 0.017707194024327282 → 0.017707194024327282  SAME
     rms_im 0.018434524224301776 → 0.018434524224301776  SAME
     acf    0.9005485904967949    → 0.9005485904967949    SAME
```

## 5. 实测三条（standard / perpoint / seed 0）

| 谱 | 状态 | 候选（rms%） |
|---|---|---|
| 3E `3E-1_HER_exp1_0005mA`（67 点） | `non_identifiable` | nested L=2 **5.81** ／ L=1 5.97 ／ voigt K=1+L 6.28 ／ K=2+L 7.34（R̂ 1.83）／ K=3+L 9.24（R̂ 1.84） |
| 腐蚀 `R1-[R2,P3]`（60 点，真值 K=1） | `non_identifiable` | voigt K=3 4.57 ／ nested L=1 4.76 ／ **voigt K=1 4.76** ／ K=2 4.78（R̂ 1.14）／ nested L=2 4.78（R̂ 1.07） |
| AZ31 `8Li_50`（50 点） | `weak_preference` | nested L=2 **2.77** ／ L=1 5.59；**voigt 三个全部被预检 excluded** |

三条读数：

1. **腐蚀谱上 K=3 略好于真值 K=1（4.57 vs 4.76）**，差 0.19pp，判 `non_identifiable`。
   这正是 evaluator 该拦的：多两个分量换 0.19 个百分点，不是证据。
2. **AZ31 的 voigt 被预检正确排除**：理由写的是「低频感抗 0.01–0.139 Hz（8 点）需把
   R∥L 放进第二并联节点（= 改族定义）」——这条谱就是 `nested` 存在的理由，不是 bug。
3. **3E 上 nested 赢 voigt（5.81 vs 6.28）但在 `perpoint` 下 voigt K=2 不收敛**（R̂ 1.83），
   而 `free` 下它收敛且 rms 更小。逐点 σ 让 0.5 Ω 的高频点权重变大、似然变尖，
   这是口径带来的采样难度，不是模型好坏。

## 6. 仍然缺的

- **族已到三个（2026-11-20）**：voigt / nested / adr 都在候选集里，每族可选挂
  `diffusion`（Ws/Wo/Wshort）与顶层 `L`（`include_L`，由 `need_L` 自动判定）。
  > **2026-11-22 补注（原文保留）**：`adr` **默认不再进候选集**，要显式 `--adr`
  > 才追加（`--no-adr` 留作兼容，现已无效果）。理由见 `ADR_FAMILY_VERDICT.md`：
  > 同参数数跨 5 类 13 条，串联式带 `L` 时 **12 胜 / 1 平 / 0 负**（均 rms 9.05%→6.93%），
  > `adr` 多出的自由度正好是不可辨那一维（`1/Rct+1/R_ad` 只能以和出现）。
  > 上面「都在候选集里」描述的是 2026-11-20 至 11-22 之间的状态。
  `adsorption` 元件仍 `wired=False`：内核挂顶层串联，而 §9.4.12 裁决只允许第二并联节点，
  接点冲突未裁 ⇒ 有意锁。
- **曲线口径 = 逐样本 Z 的逐点中位**（F1）：`fits/*.json` 落 `z_fit`，图与表用同一条数组。
  旧产物没有该字段的，按参数中位重建并在图注里标出（`Z(中位θ) ≠ median Z(θ)`，
  收敛差时两者能差几个百分点）。
- **`identified` 仍不可达**（σ/先验尺度口径未裁），状态最多到 `weak_preference`。
- **三条支路只能三选一**：`branch` 是单槽位，`merged`/`merged3` 在 `branch_merge_probe.py`
  里验证过，13 参数只换 0.14pp、cond 涨 8 个量级（§9.4.42.1）。
- `compare` 是**串行**的：5 候选 × standard ≈ 3–5 分钟/条，整目录要小时级。
- **多候选只有叠图，没有"分开的图"**：`compare.png` 把所有候选叠在同一组坐标，
  候选 ≥2 时 Nyquist / Bode 幅值 / Bode 相位（①②③）里的曲线几乎重合，
  只能靠残差子图（④⑤⑥）反推谁更好。需求已登记到
  `BMDRT_RESULT_AND_FIGURE_DESIGN.md` §11（`layout="overlay"|"facet"`，
  默认仍是 overlay），**按 2026-11-16 裁决留到输出层页面一起做**，本轮不实现。
