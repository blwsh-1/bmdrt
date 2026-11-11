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

## 4. 待决策（本轮**不改**，防阈值漂移）

- voigt 侧 `param_degenerate` 现在用 `param_corr.strong_tol=0.90`，nested 侧
  标量路径仍是 `CORR_STRONG=0.95`。两个数值统一（任选一边）会改变其中一族的
  既有状态，属**阈值调整**，需要单独预注册；本轮只在翻转表里用 0.95 复筛
  隔离了这个效应，未改任何常量。
- `p_tau_out`（分量 τ 跑出窗口）与 nested 侧「分量是否定位」仍不在 guard 里，
  属下一轮（与 `comp_unresolved` 分开报告，不压成一个布尔）。

## 5. 契约与自检

- `test_contract.py` F8（`case_f8_guard_semantics`，F8-1…F8-6）：K=1 总量补偿
  不误报、K=2 的 `w₁–τ₁` 简并必报、`φ/ζ/n1` 贴边必报、nested 对照、固定参数
  记 `fixed`。**24/24 通过**（全量含 2 次 quick 拟合）。
- `bm_check.py` C 组（C1–C5）：`PRIOR_BOUNDS` 与 `bm_core.make_model` 签名
  默认值 / 源码字面量、`bm_top._prior_table` 真实分布逐条对拍。**23/23 通过**。
- `bmdrt.py check` **18/18** 通过（生产回归无扰动）。

## 6. 文件

| 文件 | 改动 |
|---|---|
| `bm_select.py` | `_corr_pairs` 留 nested 用；新增 `_guard_deg_pairs`/`at_bound_frac`/`guard_detail`；`guard_state` 两族分判；候选带 `guard_detail` |
| `bm_check.py` | 新增 C 组先验边界对拍 |
| `test_contract.py` | 新增 F8 |
| `docs/round_2026-11-11/guard_fix_flip.py` | 翻转表脚本（含修复前冻结实现） |
| `out/guard_fix_flip.json` | 381 条翻转明细 + 汇总 |
| `NESTED_TOPOLOGY_PROJECT.md` | 订正 §9.4.15/§9.4.17 对 `param_degenerate` 的错误归因 |
