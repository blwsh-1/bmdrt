# B 类 guard 的「族适配」修复（2026-11-11）

**范围一句话**：`param_degenerate` 与 `param_at_bound` 在 voigt 侧此前**根本没有
真正在判**——前者恒被「同量异名」的参数化冗余触发、后者恒 `not_evaluable`。
本修把两条都接到 voigt 的真实参数上。**只动记账层**：A 类门禁、拟合、选 K、
M4 结论、任何阈值数值都不变；不重跑 MCMC。

## 1. 修之前的问题（385 条已落盘后验实测）

`bm_select.guard_state()` 里两条判据都按**嵌套族**的参数形态写的，voigt 是
数组型参数、指数叫 `phi`，两边对不上：

| guard | 旧实现 | voigt 实测后果 |
|---|---|---|
| `param_degenerate` | `_corr_pairs()` 只收**标量列**（`a.size==n_draw`） | `tau/phi/w` 是 `(n_draw,K)` 被整列丢掉；K=1 时 `(n,1)` 列却 ravel 成 `n_draw`，把 `W`↔`w`（`w≡W·π`）、`ln_tau`↔`tau` 当两个参数 |
| `param_at_bound` | `_uniform_bound_frac()` 只认 `n*` 键名 | voigt 指数叫 `phi` → 恒 `not_evaluable`（381/381） |

**旧 `param_degenerate` 在抓什么**（`out/guard_fix_flip.json` 的
`old_pair_inventory`，381 条）：`K=1 W+w` 279、`K=1 ln_tau+tau` 235、
`K=1 R_inf+W` 42、`K=1 Aw+Rw` 42、`K=2 W+v` 11 —— 几乎全是**定义式/参数化
冗余**，不是「两个分量互相补偿」那种真实简并。所以 §9.4.15 / §9.4.17 里
「`param_degenerate` 在多数候选 triggered（M3 §19 的 `R1/R2` 共线是真实结构
限制）」的读法是**反的**：触发源不是 `R1/R2`，而且 K=2（`R1/R2` 真正会在的
地方）恰恰几乎从不触发。

## 2. 改成什么

- **`param_degenerate`（voigt）**：不再自己展平后验，改读 `bm_corr.param_corr`
  的 `strong`（已经 `match_components` 对齐、按分量拆列）。只在**分量/形状
  参数**（`ln τ_j`、`ln w_j`、`φ_j`、`ζ_j`、`R_inf`、`R_ad`/`tau_ad`）之间取；
  噪音位点 `s_*`、仪器项 `L`、扩散支路 `Rw/Rwo/tauD` 排除；K=1 的
  `R_inf↔w₁` 排除（`w₁≡W`，是总电阻分配，已由 `bm_corr.degenerate` 的
  `Rinf-Wtot` 单列）。nested 侧沿用原标量路径（本轮**不改**，见 §4 待决策）。
- **`param_at_bound`**：改成**表驱动**（`bm_select.PRIOR_BOUNDS`），voigt 看
  `phi∈[0.5,1]`、`zeta∈[0.3,2]`，nested 看 `n1/n2∈[0.5,1]`。LogNormal 的
  `R/Q/R_inf/W` 没有有限边界，**不进**这一项。带宽按各自区间宽度的 1%
  （`eps_j=0.01·(U_j−L_j)`）；**按参数分别统计再取最大**，不再混池稀释；
  固定参数（`zeta_fixed`）记 `fixed` 不参与。
- 两处阈值**数值都没动**：贴边 5%、带宽 1%。`param_degenerate` 的阈值来源
  变成 `param_corr.strong_tol`（`bm_corr.STRONG`，默认 0.90）。

## 3. 修复前后翻转（381 条生产拟合，同一份后验，不重跑）

`out/guard_fix_flip.json`，脚本 `docs/round_2026-11-11/guard_fix_flip.py`
（冻结了修复前的两个函数作对照）。

| 判据 | 翻转 | 说明 |
|---|---|---|
| `param_degenerate` | **334/381** 翻转 | `triggered→ok` 261（K=1 的假阳性）、`ok→triggered` 73（K=2 新发现的简并） |
| └ 按旧阈值 0.95 复筛 | **311/381** | 其中 `ok→triggered` 剩 40 条 → **33 条只由 0.90/0.95 阈值差造成**，40 条是真新发现 |
| `param_at_bound` | **381/381** 翻转 | `not_evaluable→ok` 303、`→triggered` 78 |

`param_at_bound` 新揪出的 78 条（此前完全看不见）分方向：
**38 条 `phi` 贴下界 0.5（CPE 退化成半无限 Warburg）**、**40 条 `phi` 贴上界
1.0（退化成纯电容）**，`zeta` 在两条自由采样里都没有触发。K 分布：K=1 38 条、
K=2 38 条、K=3/4 各 1 条。

> 翻转 ≠ 模型变好。同一份后验，只是「此前没适配上该模型族的 guard 被修好了」。
> 任何把上面数字读成「参数可辨识性改善 / 模型性能改善」的写法都是错的。

## 4. 阈值统一（2026-11-12 决定并落地）

- **决策**：voigt 与 nested 两族 `param_degenerate` 阈值统一到 **0.90**
  （`bm_corr.STRONG`），判定符号对齐为 `|ρ|≥0.90`（`_corr_pairs` 原为 `>0.95`）。
- **依据**（对 189 对对齐后强相关的 ρ 分布实测）：0.90–0.95 区间约 112 对，
  绝大多数是分量内 `ln τ–ln w`（50 对）、`ln w–φ`（分量内幅值–指数）——真实
  简并，不是采样噪声（ρ=0.92 在 ~400 样本下采样误差约 ±0.01）。B 类只记账
  不进门禁，误报代价低、漏报代价高 → 偏严（0.90）优于偏松（0.95）；且 0.90
  与 `MEDIUM=0.70` 构成弱/中/强三档语义，0.95 是旧路径遗留的无配套孤值。
- **影响面**：nested 侧真实后验样本从不落盘（`nested_fit` 只存 summary；M4 臂
  候选 `raw` 为空、`guard_detail=None`、产物为 2026-11-10 旧实现），故**现有
  产物无一条可复算翻转**，0.90 从下次重跑 M4 臂起生效。voigt 侧上一轮已用
  0.90，本轮不变。翻转脚本 `guard_fix_flip.py` 的 `OLD_STRONG` 已改硬编码 0.95
  冻结，防止随生产常量漂移。

## 5. 契约与自检

- `test_contract.py` F8（`case_f8_guard_semantics`，F8-1…F8-6）+ F9
  （`case_f9_nested_and_outwindow`，F9-1…F9-4）。全量 **25/25** 通过。
- `bm_check.py` C 组（C1–C6）：`PRIOR_BOUNDS` 对拍 + nested τ 派生公式物理自检。
  **24/24 通过**。
- `bmdrt.py check` **18/18** 通过（生产回归无扰动）。

## 6. 文件

| 文件 | 改动 |
|---|---|
| `bm_select.py` | `_corr_pairs` 留 nested 用；新增 `_guard_deg_pairs`/`at_bound_frac`/`nested_component_spans`/`_comp_out_state`/`guard_detail`；`guard_state` 两族分判；候选带 `guard_detail`/`comp_out_*` |
| `bm_check.py` | 新增 C 组先验边界对拍（C1–C5）+ nested τ 拐点自检（C6） |
| `test_contract.py` | 新增 F8、F9 |
| `docs/round_2026-11-11/guard_fix_flip.py` | 翻转表脚本（含修复前冻结实现） |
| `out/guard_fix_flip.json` | 381 条翻转明细 + 汇总 |
| `NESTED_TOPOLOGY_PROJECT.md` | 订正 §9.4.15/§9.4.17 对 `param_degenerate` 的错误归因 |
| `docs/WS_CRITERION_PLAN.md` | §5.1 补 npj 第三套独立实现外部证据 |
| `docs/REFS_DIGEST.md` | §2.B3 收窄为「限定元件形态」、§2.C 记 A4 只到第三层 |

## 7. nested `comp_unresolved`（预注册口径，2026-11-12）

补的是主线族的对称缺口：nested 侧此前 `comp_unresolved` 恒 `not_applicable`。

**口径**（写死，实现 `bm_select.nested_component_spans`）：

- 「分量」= 每层 R∥CPE 的拐点时间常数 + 支路：
  - `τ1 = (R1/Q1)^(1/n1) / (2π·f_ref)`，`τ2 = (R2/Q2)^(1/n2) / (2π·f_ref)`（`layers=2` 才有）；
  - `τ_branch = tau_ad`（parallel）/ `tau_L`（series）；
  - `f_ref = sqrt(f_min·f_max)` 取 `fit_top` 返回的 `f_ref`；`Q` 是窗口幅值
    （`Z_CPE = Q·(ω_ref/ω)^n·e^{-jnπ/2}`），拐点取 `|Z_CPE| = R`。
- **labeled 标量参数，不做排序/对齐**（无编号跳变），直接按名取；
- 判据只认 τ 的 5–95% 跨 `> 2 decade`（与 voigt 同一理由：窗口约 5 decade）；
- **无幅值判据**（nested 无 stick-breaking 权重）；「支路幅值近零」由 `loop_amp`
  （`R_ad`/`R_L` < 3σ）单独管，与本项互补、不合并。

**语义差异必须写清**：voigt 的 `comp_unresolved` 管「多余分量没被定住（K 冗余）」，
nested 这里管「结构上的每一层/支路有没有被定住」——同名 guard、两个对象。

## 8. voigt `comp_out_window`（2026-11-12）

把 `bm_real` 早已算好的 `p_tau_out`（`P(f_τ ∉ [f_lo,f_hi])`，逐分量）接成 B 类
guard：`max_k p_tau_out > 0.5 → triggered`（0.5 与 `bm_real` 既有报警口径一致）。
与 `comp_unresolved` **分开报、不压成一个布尔**：
- `comp_unresolved` = 后验 τ 跨 >2 dec 或幅值 <5%（"宽/小"，窗内没定位）；
- `comp_out_window` = τ 分布落在窗外（"远"，跑到看不见的地方）。
合成探针已证「远」这一支两个现有 guard 都漏（τ 紧贴推到 ~1e4 s，span 仅 1.3 dec）。

nested 侧 `comp_out_window` 暂 `not_applicable`（`fit_top` 未算 `p_tau_out`；
其 τ 已在窗外与否可用 §7 的 `tau_med` 对照窗口，留待后续）。
