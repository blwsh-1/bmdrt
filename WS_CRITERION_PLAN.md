# Ws 判据的算法探索（v0.2，探索性）

> 立项：2026-11-09。定位：**给「要不要挂一个 `A_W(jω)^(-1/2)` 项」这件事
> 换一套可测量的判据**，并把每条判据的能力边界用合成真值钉死。
>
> v0.2（2026-11-09）：N1–N4 已执行（见 §9）。判据模块新增 `to_inductive`
> 档（18 条真实谱暴露的缺档）；Boukamp 梳齿指纹**验证为不可用**（§9.3）。
>
> 本文**不改生产代码**：`bm_wscrit.py` 等放在 `evidence/ws_crit/` 下，与
> `bm_kk` / `bm_diff` / `bm_real` 无 import 关系（只读调用）。N1 的改动以
> **补丁 + 影子实现**形式给出，`git apply --check` 通过但**未应用**。
> `DIFFUSION_PLAN.md` / `WS_DECISION_PROJECT.md` / `M6_VALIDATION_PLAN.md`
> 的既有结论与预注册判据**一个字没改**。

---

## 0. 一句话结论

| 项 | 结论 | 依据 |
|---|---|---|
| 现行 L1 幅度判据（`log\|Z\| ~ log f`） | **会被 `R_inf` 压平，在两条真含 Ws 的样例上都漏报** | 实测：`caseE` 斜率 −0.03、`caseE_warburg` −0.02，而 `-Im` 斜率 −0.49/−0.45 |
| 换 `log(-Im Z) ~ log f` | **`R_inf` 完全免疫**（理想 −0.50）；把 `caseE`/`caseE_warburg` 的 `is_candidate` 从 False 修成 True | 自检 [1][2]、§9.1 |
| 端点走向 | 能分 `Ws` / `Wo` / `Wshort` 三支，是 `DIFFUSION_PLAN` §3.2 三个方向的**数据侧**对应物；但**必须再加第四档 `to_inductive`** | 自检 [4][7]、§9.2 |
| `to_inductive`（低频 `-Im<0`） | **18/18 真实 AZ31 谱都落这一档**——低频支路是吸附/感抗环，不是任何扩散尾。原实现把它并进 `to_plateau` 会给出 `Wshort_like` 的错误断言 | §9.2 |
| 段长门槛 | **不能**用单一阈值，必须**按家族分别设** | 基准扫描，§2.2 |
| `Ws` vs「窗外 `R||CPE(0.5)`」 | **数学同形，任何只依赖窗内形状的判据都分不开**；自由拟合 chi2 比值 **0.995 / 1.004** | 自检 [3]、§3.1、§5.1 |
| Boukamp 梳齿指纹（N2） | 理论成立（`R₁=0.811Z₀`、`τ₁=0.4053τ₀`），但**DRT 分辨不出高次齿**，且 `R_top/R_total` 与单一 ZARC 重叠 → **做不成判据** | §9.3 |
| 可检测下限（N4） | `P(candidate)=50%` 需 `SNR_im ≈ 15`；天花板 0.88（达不到 0.9） | §9.4 |
| 三个阴例的假阳性 | **0/20 × 3 类 × 2 窗口** | §3 |

---

## 1. 文献 → 判据的映射

六篇里真正能落到算法上的只有四个点，其余是背景。

| 文献 | 可落地的部分 | 对应到哪条判据 |
|---|---|---|
| Huang 2018, *Electrochim. Acta* 281:170 | 半无限成立条件是 `ω ≪ D/L_d²`（即 `f ≫ f_D`）；有界扩散 `λ_L = coth` 或 `tanh`；**Warburg 的第三条假设是"平面"，第四条是"纯扩散"**，电解质里的带电物种扩散与迁移强耦合，原始 Warburg 物理上不成立 | 支持「端点走向」判据；支持 §5.4 |
| Bumberger & Fleig 2024, *PCCP* 26:15068 | Warburg 是传输线的特例：**一个载流子电阻可忽略 + 端阻抗为短路/开路**；两种边界给出 `Ws`（tanh）与 `Wo`（coth）；文献明确指出"**低频 Warburg 特征没被数据完整包含时，拟合误差超过 100%**" | 支持「端点必须落在窗内」；`Wshort` 单列的理由 |
| Boukamp 2017, *Electrochim. Acta* 252:154 | 理想 FLW 的 DFRT 是**无限个 δ 函数**：`tanh(z)/z = Σ 2/(z²+π²(k−½)²)`、`coth(z)/z = 1/z² + Σ 2/(z²+n²π²)` | N2 已实测：**理论对，但做不成判据**（§9.3） |
| Boškoski 2024, *J. Power Sources* 622:235359 | "极重要性 = DRT 峰面积 = 相对极化贡献"；RQ 极的积分有闭式 | 模型侧 `rpc_share` |
| Zhang 2023, *J. Electrochem. Soc.* 170:086502（AutoEIS） | 贝叶斯模型选择 + 结构模板库 | 项目已用，本文只作对照口径 |
| Lazanas & Prodromidis 2023, *ACS Meas. Sci. Au* 3:162 | EIS 教程（Randles、Warburg 三种边界、45° 的常见误读） | 背景，无新判据 |

> 注：Boukamp 那篇的 `tanh` 是**透射**（固定活度背边界），对应本项目的 `Wshort`，
> 不是 `Ws`。拿它当 `Ws` 的 DRT 依据会直接把方向搞反（`DIFFUSION_PLAN` §1.2 的同一个坑）。

---

## 2. 判据定义（`evidence/ws_crit/bm_wscrit.py`）

### 2.1 数据侧（不需要拟合，不需要真值）

| 量 | 定义 | 为什么这么定 |
|---|---|---|
| `slope_im` | 最低 25% 段 `log(-Im Z) ~ log f` 的斜率，要求 `r2 ≥ 0.90`、`-Im` 随 f 下降单调升 | `R_inf` 只在实部，`-Im` 里没有它。理想 `Ws` 恒为 −0.5 |
| `amp45` / `amp90` | `\|slope_im+0.5\| ≤ 0.15` / `\|slope_im+1.0\| ≤ 0.15` | 45° 与 90° 是两种不同的低频机制，必须分开 |
| `band_dec` | 45° 段**最长连续长度**（decade），取两口径较大者：① Nyquist `d(-Im)/d(Re)` ∈ 1±0.15；② `-Im` 自身对数斜率 ∈ [−0.68,−0.32] | ① 在 `Re` 几乎不动的段上分母趋零：只用 ① 时 `wo_in` 段长中位数 **0.00** decade（0/20 到得了 0.5），用 ② 后中位数 0.67、13/20 到得了 |
| `reach_edge` | 该段是否触及最低频点 | "到边缘仍 45°"是半无限的必要条件 |
| `term` | 端点走向，用**最低 5 点的 `-Im` 对数斜率**判：≤−0.80 → `to_cap`；−0.70..−0.30 → `still_45`；≥−0.15 → `to_plateau`；**最低 5 点 `-Im` 中位数 < −`_IND_TOL`·max\|Im\| → `to_inductive`**；其余 `noisy` | 绝对相位会被 `R_inf` 拉向 0（`R_inf=500` 的纯 `Ws` 在 0.1 Hz 只有 −15°）；`d(-Im)/d(Re)` 在电容尾上分母趋零，实测把 13/20 条 `Wo` 误判成平台 |
| `_IND_TOL = 0.10` | 判 `to_inductive` 的相对深度门槛 | **只按符号判会误伤**：`wshort_in` 的 `-Im` 真值只有 0.18–1.7 Ω 而噪声 sd≈0.5–1 Ω，符号随机翻负，纯符号判把 3/20 真 Wshort 打成"感抗"。实测 `im_ratio` 分布：`sorp_rl` −0.18…−0.99、`wshort_in`/`none_2zarc` 最负只到 −0.05、18 条真实谱 −0.42…−0.58。0.05 与 0.18 之间有空档，0.10 落在中间 |

→ `family ∈ {Ws_or_wide_CPE, Wo_like, Wshort_like, inductive_tail, unknown}`，
`verdict ∈ {candidate, reject, undecidable}`。

### 2.2 判定规则（**按家族分别设门槛**）

```
term == to_inductive              → reject（感抗：不是任何扩散尾，走吸附支路）
term == noisy                     → undecidable
term == to_cap     且 band ≥ 0.10 或 amp45 → candidate (Wo_like)
term == to_cap     且 都不成立              → undecidable   ← 见 §5.3
term == still_45   且 (band ≥ 0.50 或 reach_edge 或 amp45) → candidate (Ws_or_wide_CPE)
term == to_plateau 且 band ≥ 0.50           → undecidable   ← Wshort 家族，未接入
其余                                        → reject
```

**为什么不能用一个门槛通吃**（扫 `min_dec` 的实测，`0.1–1e3 Hz` 窗口；
简化规则 `band ≥ thr 且 term ≠ noisy`，只为看单调性）：

| 门槛 | `ws_vis` | `ws_buried` | `wo_in` | `wo_below` | `wshort_in` | 阴例特异度 |
|---|---|---|---|---|---|---|
| 0.0 | 1.00 | 0.90 | 1.00 | 1.00 | 0.95 | **0.02** |
| 0.2 | 1.00 | 0.80 | 1.00 | 1.00 | 0.55 | 0.45 |
| 0.5 | 1.00 | 0.70 | 0.65 | 0.95 | 0.25 | 0.67 |
| 1.0 | 0.90 | 0.45 | 0.20 | 0.95 | 0.20 | 0.88 |

`Wo` 的 45° 段**天生就短**——它到拐点就结束了，拐点之后是电容。门槛抬到
1.0 decade，`wo_in` 从 1.00 掉到 0.20，而阴例特异度只从 0.02 涨到 0.88：
**用段长换特异度，代价不成比例**。按家族分门槛（§2.2 规则）在同一基准上
拿到 `ws_vis`/`wo_in`/`wo_below` 全部 20/20、阴例特异度 1.00。

### 2.3 模型侧（读已有后验样本，不重新采样）

| 量 | 定义 | 适用 |
|---|---|---|
| `rpc_share` | `R_w /(R_w + Σ w_k)` 的后验中位数 + 90% 区间 + `P(<5%)` | **只对 `Wo`** |
| `ws_share_low` | `Re Z_Ws(f_min) /(Re Z_Ws(f_min) + Σ w_k)`，`Re Z_Ws(f) = A_W/√(4πf)` | **只对 `Ws`**，必须连 `f_min` 一起报 |

---

## 3. 基准结果（9 类 × 20 条 × 2 窗口，1% 相对噪声）

窗口 `0.1–1e3 Hz` / `0.01–1e5 Hz`，均 31 点，真值由构造给出。

| 类别 | 构造 | `candidate` | 家族命中 | 判读 |
|---|---|---|---|---|
| `ws_vis` | 真含 Ws，`\|Z_W(f_min)\| ≈ 1–2·R_p` | **20/20 · 20/20** | 20/20 · 20/20 | ✓ |
| `ws_buried` | 真含 Ws，幅值只有 `0.3·R_p` | 10/20 · 11/20 | 11/20 · 14/20 | 噪声限，见 §9.4 |
| `wo_in` | 真含 Wo，`f_D` 在窗内 | **20/20 · 19/20** | 20/20 · 20/20 | ✓ 家族全对 |
| `wo_below` | 真含 Wo，`f_D` 在窗下 | **20/20 · 19/20** | 20/20 · 20/20 | ✓ 与 Ws 同形，标对 |
| `wshort_in` | 真含 Wshort，`f_D` 在窗内 | 0 · 0（拒/未决 = 14/6、17/3） | 19/20 · 19/20 | 家族对，但**不给 Ws 家族**（`bm_core` 未接 Wshort） |
| `none_2zarc` | 两个真弛豫，无扩散 | **0/20 · 0/20** | — | ✓ 零假阳 |
| `rq05_low` | `R\|\|CPE(0.5)`，峰远在窗下 | 20/20 · 20/20 | 20/20 · 20/20 | **已知不可分**，§5.1 |
| `rq05_in` | `R\|\|CPE(0.5)`，峰在窗内 | **0/20 · 0/20** | — | ✓ 零假阳 |
| `sorp_rl` | `R\|\|L` 并联吸附环（低频感抗） | **0/20 · 0/20** | 20/20 → `inductive_tail` | ✓ 零假阳，且家族标对 |

三个阴例合计 **120 次判定 0 例假阳**。`rq05_low` 的 20/20 不是假阳，是
**不可分**（§5.1），报告里必须写成 `Ws_or_wide_CPE` 而不是"检出扩散"。

### 3.1 真实文件对照（`examples/`，只读）

| 文件 | 真值 | 现行 L1 `is_candidate` | 新判据 | 变化 |
|---|---|---|---|---|
| `example_caseE.csv` | **未登记**；拟合 `A_W=2.03`（§3.2） | False（漏报） | `slope_im=−0.49`、band 1.63、`still_45` → **candidate** | 修掉一个已知漏报 |
| `example_caseE_warburg.csv` | 含 `Ws(A_W=2)`（`DIFFUSION_PLAN` §9） | False（漏报） | `slope_im=−0.45`、band 0.53、`still_45` → **candidate** | 同上 |
| `example_caseF_wo.csv` | 含 `Wo(τ_D=0.3)` | False | band 0.27、`to_cap`、`Wo_like` → **candidate** | 给出家族 |
| `example_caseG_wo_only.csv` | 纯 `Wo`，σ=0.803 Ω | False | band 0.00、`to_cap` → **undecidable** | 见 §5.3 |
| `example_caseF.csv` | 未登记；拟合无 ω^-1/2 项 | False | `to_plateau` → reject | 一致 |

### 3.2 `example_caseE.csv` 的真值：拟合给出的答案，以及一个必须记下的事实

`bm_kk.py:389` 把这条谱注释成「无扩散元件」，并把它当作**幅度判据的立论依据**
（"它之所以被挡掉，靠的是幅度判据（`|Z|` 斜率 −0.03）"）。实测不成立：

- 低频 `A_W = -Im·2√(πf)` 在 1.8 个 decade 内稳定在 **1.98–2.02**；
- `R∞(10) + ZARC(30Ω, 306Hz, 0.846) + Ws` 拟合得到 **`A_W = 2.030`**，rms 0.0082；
- 与 `caseE_warburg`（同拓扑、窗口窄 3 个 decade）的参数一致
  （`R∞=10.08, R=29.92, f0=295.4, n=0.850, A_W=2.046`）。

**但同一组数上，"窗外 `R||CPE(0.5)`"拟合得一模一样**：

| 模型 | `caseE` chi2（rms） | `caseE_warburg` chi2（rms） |
|---|---|---|
| `R∞ + ZARC + Ws(A_W)` | 0.0042（0.0082） | 0.0016（0.0051） |
| `R∞ + ZARC + ZARC(R₂, f₂, n₂)` | 0.0042（0.0082） | 0.0016（0.0051） |
| 比值 `chi2(A)/chi2(B)` | **0.995** | **1.004** |
| B 的参数 | `R₂=93.28, f₂=1.0e-4 Hz, n₂=0.518` | `R₂=78.68, f₂=1.0e-4 Hz, n₂=0.491` |

拟合自己把第二个 ZARC 推到了 `f₂≈1e-4 Hz`（窗外）、`n₂≈0.50`——正是 §5.1 说的
同形解。所以：

- 「`caseE` 含 `A_W=2` 的 `ω^-1/2` 项」这句**有数值支撑**；
- 「所以它是扩散」这句**没有支撑**，两条模型不可区分；
- 「`caseE` 无扩散元件」这句**与数据矛盾**（`-Im` 的幂律与 Zre 的同步下降
  都对得上 `A_W=2`）。

结论只写到这一层。`bm_kk.py:389` 的注释方向反了这件事，`WS_DECISION_PROJECT`
WP4 已登记；本文补上定量证据，但**不改那句话**（冻结文档纪律）。

---

## 4. 模型侧：两个份额量，只有一个能用于 Ws

读冻结样本（`evidence/b2x/`、`out/_caseE_diff/`），**不重新采样**：

| 组 | 元件 | `rpc_med`（RPC） | `ws_share_low`（`f_min`） | `A_W` |
|---|---|---|---|---|
| `b2x k1 +Wo`（对元件，Wo 数据） | Wo | **0.481** [0.438, 0.516] | — | — |
| `b2x k1 +Ws`（错元件，同一数据） | Ws | 0.002 [0.001, 0.013] | **0.011**（`f_min`=0.1） | 13.65 |
| `caseE k1 +Ws`（真含 Ws） | Ws | 0.005 | **0.161**（`f_min`=0.01） | 2.035 |
| `caseE_warburg k1 +Ws`（同上，窗口窄 1 dec） | Ws | 0.009 | **0.057**（`f_min`=0.1） | 2.043 |
| `caseE k2 +Ws` | Ws | 0.005 | 0.158（`f_min`=0.01） | 2.013 |

三条读法，**必须分开说**：

1. **`Wo` 的 RPC 可以直接读**：正确元件承担 48% 的极化电阻，是一等过程，不是
   修修补补。错元件（`Ws` 配 `Wo` 数据）只有 0.2%。
2. **`Ws` 的 RPC 不能读**。`Ws` 没有有限极化电阻，样本里的 `Rw` 是
   `|Z_Ws(f_ref)|`（`f_ref=√(f_min·f_max)`）。`caseE` 真值 `A_W=2`、`R_p=30`，
   在 `f_ref=31.6 Hz` 只有 0.14 Ω，RPC 给 0.005——看起来像"扩散项可忽略"，
   其实是归一化点的选择。
3. **`Ws` 要看 `ws_share_low`，且必须连 `f_min` 一起报**：同一条 `A_W≈2` 的谱，
   窗口 0.01–1e5 Hz 给 16.1%，窗口 0.1–1e3 Hz 只给 5.7%。而"错元件"那组
   给 1.1%——**判据在这里有区分度（16% vs 1%），但数值不能脱离窗口引用**。

---

## 5. 边界（写死，不许外推）

### 5.1 `Ws` ≡「峰位远在窗下的 `R||CPE(0.5)`」，不可分

`R||CPE(0.5)` 在 `ω ≫ 1/τ` 时 `Z → 1/(Q(jω)^{1/2})`，与 `Ws` 是**同一个
渐近式**；差别只在 `ω→0` 的极限（一个趋于 `R`，一个发散），而那个极限在窗口
外面。实测（自检 [3]）：两者给出**完全相同**的 `family=Ws_or_wide_CPE`，基准
里 `rq05_low` 20/20 判成 `candidate`；§3.2 的自由拟合把第二支推到
`f₂=1e-4 Hz, n₂≈0.50`，chi2 与 Ws 模型**比值 0.995–1.004**。

所以：

- 这条判据**只能否证**（不是 45° 尾 ⇒ 不用挂扩散项），不能确认；
- 报告里**禁止**写"检出扩散""确认为 Ws"，只能写"需要额外低频自由度，
  且这些解释在窗内不可唯一区分"（与 `WS_DECISION_PROJECT` §3 Q2 一致）。

### 5.2 端点分类依赖"45° 段可见"

45° 段落在噪声地板以下时，端点分类仍然对（`to_cap`），但"有没有 45°"判不出来。
`ws_buried` 只有 10–11/20 通过就是这个原因，不是判据写错。定量边界见 §9.4。

### 5.3 `to_cap` + 无 45° 必须给 `undecidable`，不能给 `reject`

`example_caseG_wo_only.csv`（纯 Wo，σ=0.803 Ω）就是这样：低频确实转电容，
但它的 45° 段在高频、而那里 `-Im ≈ 0.5 Ω` 已经在噪声底下面。此时
"理想阻塞电极（纯电容）"与"噪声底下的 Wo"在窗内完全同形。给 `reject` 就等于
宣称"没有扩散"，那是错的。这是**"没看见 ≠ 没有"**，必须留出未决档。

### 5.4 判据不涉及物理机制

Huang 2018 的第 4 条假设（纯扩散、中性物种）在电解质里本来就不成立
（带电物种的扩散与迁移强耦合）。所以即使判出 `Ws` 家族，也只能说
"这条谱在窗口内需要一个 `ω^{-1/2}` 项"，**不能说存在半无限扩散**。

### 5.5 低频 `-Im < 0` 是独立的第四档，不能并进 "回到实轴"

2026-11-09 在 18 条真实 AZ31 谱上暴露：它们的低频端 `Zim > 0`（`-Im < 0`），
是**吸附/感抗环**（`ADSORPTION_PROJECT` 的 `R∥L` 类），不是"回到实轴"。
原实现把两种情形都归进 `to_plateau`，于是给出 `family=Wshort_like`
（有限长**透射扩散**）——一个完全错误的断言：那条谱低频根本没有 `-Im>0`
的扩散尾。Warburg 尾的必要条件是 `-Im > 0 且 ∝ f^(-1/2)`，**符号不符即否证**，
不需要再看段长。

`to_inductive` 的限定必须一起写：
- 它只否证**窗内可见**的电容性扩散尾；若吸附环与扩散尾共存，感抗环会把它
  掩盖，此档**不等于**断言「不存在扩散」；
- 判它要用**相对深度**（`_IND_TOL`），不能只看符号，否则噪声底附近的
  符号翻转会把真 `Wshort`（`-Im` 真值 0.18–1.7 Ω）误杀（实测 3/20）。

### 5.6 18 条真实 AZ31 谱的登记结果（**不改变 M6 的任何结论**）

`n3_real.py` 逐条只读登记，`truth=unknown`、`independence_status=inferred`
（与 `M6_VALIDATION_PLAN` 一致，**不算 TP/FP、不进 G8b**）：

| 结果 | 条数 |
|---|---|
| `candidate` | **0/18** |
| `reject` | **18/18** |
| `undecidable` | 0/18 |
| `family = inductive_tail` | **18/18** |
| `f_min` | 全部 0.01 Hz，50 点，窗 0.01–1e5 Hz |
| 与现行 L1 的 candidate 判定不一致 | **0/18** |

`im_ratio` 范围 **−0.42…−0.58**（中位数 −0.50），远深于噪声翻号的 −0.05，
所以这 18 条的 `to_inductive` 不是噪声假象。

**读法**（三句，不能合并）：
1. 这 18 条的低频支路是**感抗**，不是任何扩散尾——所以"Ws 判据给 `reject`"
   在数据侧是有理由的，不只是"没有真值所以 `non_evaluable`"；
2. 但 `truth` 仍然是 `unknown`：**这不构成 TN**，不能进 G8b 的混淆矩阵；
3. `reject` 的限定是"窗内看不见电容性扩散尾"，**不是**"不存在扩散"（§5.5）。

---

## 6. 与现有 L0–L4 的关系（不改生产代码）

`WS_DECISION_PROJECT.md` §2 的现状：

| 层 | 现状 | 本文给什么 |
|---|---|---|
| L1 事前形状 `bm_kk.diffusion_tail_candidate` | **`log(-Im)`** 幅度判据 + Nyquist 相位判据（2026-11-09 起；旧为 `log\|Z\|`） | 补丁 `patches/n1_l1_amplitude.patch` **已应用**（§9.1、§10.4）；**注意它有下游行为影响**：`kk_check` 用 `tail["f_cut"]` 决定要不要跑 `add_cap` 复验 —— `caseE`/`caseE_warburg` 的 `f_cut` 由 None 变为 0.4299/0.8577，`caseF_wo`/`caseG_wo_only` 反过来变为 None |
| L1' 结构指纹 `bm_diff.looks_like_warburg` | 后验分量级（`φζ`、峰位、简并） | **已接入**（2026-11-09）：`bm_ws.ws_diagnostics` 入生产，`diffusion_gate` 输出 `ws_diag`；`to_inductive` 登记为感抗尾（只否证，不改门控结论） |
| L2 触发 `bm_real._none_needs_ws` | `acf>0.8` 或 `rms/σ>1.5` | **已接入**（2026-11-09）：`gate.status=="candidate"` 与残差条件取**或**；`inductive_tail` 时短路（`skipped_inductive`，`--auto-mechanism-full` 除外） |
| L3 决策 `select_mechanism` | 配对 LOO | **只挂诊断**（2026-11-09）：`share` 写进 `note`，**不参与 choice**。`ws_share_low` 随 `f_min` 漂移（同一真值 0.161 @0.01 Hz vs 0.057 @0.1 Hz），5% 闸门未标定，见 `L123_REPORT.md` §4 |
| L4 前提 `diffusion_resolution` | `f_D` 是否入窗 | 与本文 `term` 同一件事，但 L4 需要先拟合；本文可先给 |

**接入顺序**：L1 幅度判据（P1，已应用）→ L1'/L2（2026-11-09 接入，见 §10.6）→
L3 闸门（**未接**：阈值未标定，只挂诊断）。

---

## 7. 复现

```bash
cd D:/eis训练数据/bmdrt
PY=D:/autoeis_env/Scripts/python.exe

$PY evidence/ws_crit/bm_wscrit.py      # 7 项自检（<1 s）
$PY evidence/ws_crit/bench.py          # 9 类 × 20 条 × 2 窗口（≈10 s）
$PY evidence/ws_crit/sweep.py          # 段长阈值扫描（读 bench_rows.json）
$PY evidence/ws_crit/run_examples.py   # examples/ 五个文件逐条对照
$PY evidence/ws_crit/model_side.py     # 读冻结样本算 RPC / low-share
$PY evidence/ws_crit/n3_real.py        # N3：18 条真实谱只读登记
$PY evidence/ws_crit/n4_limit.py       # N4：SNR × 幅值比 检测下限（≈1 min）
$PY evidence/ws_crit/n2_comb.py        # N2：梳齿指纹（≈2 min，含 DRT）
$PY evidence/ws_crit/n1_shadow.py      # N1：影子对照 + 生成补丁（不应用）
$PY evidence/ws_crit/n1_fitcasee.py    # caseE 的双模型对拍（§3.2，只读）
```

产物：`bench_rows.json`（360 条）、`model_side.json`、`real_register.json`（18 条）、
`limit_surface.json`（1280 条）、`comb_n2.json`、`n1_shadow.json`、
`patches/n1_l1_amplitude.patch` + `patches/bm_kk_n1_tmp.py`（副本）。

---

## 8. 下一步（P1–P3 已于 2026-11-09 执行，见 §10）

| # | 内容 | 状态 |
|---|---|---|
| P1 | 打 `n1_l1_amplitude.patch`，跑 `bm_check` + 全量回归 | **已执行（2026-11-09 14:31–14:32）**：`nofit` 18/18、`fast` 18 通过/0 失败/7 跳过、影子对拍 0/24 不一致、溯源 20 处/10 条谱；补丁保留在工作树（**未提交**），详见 §10.4 |
| P2 | L1 的幅度带宽按端点走向分设 | **已否**：三变体逐行相同，`wo_in` 被相位判据挡住（Nyquist 中位数 −17.0，0/20 在 45°）；L1 天生只能做 `Ws`（§10.1） |
| P3 | 把 `to_inductive` 接到吸附支路做预筛 | **已否**：18 条真实谱上零收益（会挡掉 0/18），合成侧只影响不被采用的分支（§10.2） |
| P4 | `ws_buried` 的可检测下限写进 `WS_DECISION_PROJECT` 的「够不够判」一节 | **已执行（2026-11-09）**：写入 `WS_DECISION_PROJECT.md` **§3.1**（`SNR_im=14.9` 为 50% 检测点、天花板 0.876、真实谱只能用代理指标做事后诊断）；纯文档，见 §10.5 |
| P5 | 给 18 条真实谱一个有**正面**模型解释（`R∥L` 吸附支路） | 未启动；`ADSORPTION_PROJECT` 侧的前置是「用 `--auto-adsorption` 跑一条正式档」，E3/E4 缩减口径下未批准 |
| P6 | L1'/L2/L3 接入生产：`bm_ws` 端点走向、`is_candidate` 触发、感抗尾短路、份额只报不判 | **已执行（2026-11-09）**：迁移对拍 0 差异；L2 在 31 条谱上不一致 0；`caseE` Δelpd 逐位不变（131.93）、`AZ31-4Li_100` 拟合次数 2→1；回归全绿；详见 `evidence/ws_crit/L123_REPORT.md` 与 §10.6 |

**不做**（已证不可行）：Boukamp 梳齿作为判据（§9.3）；用 `R_top/R_total`
区分 `Wshort` 与单一 ZARC（§9.3）；用 DRT 处理 `Wo`（串联电容不在表示能力内）。

---

## 9. N1–N4 执行结果（2026-11-09）

### 9.1 N1：L1 幅度判据换 `log(-Im)`

**做法**：不直接改生产文件。`n1_shadow.py` 在内存里复制一份
`diffusion_tail_candidate`，只换幅度判据的量；另生成
`patches/n1_l1_amplitude.patch`（对副本 diff，`git apply --check` 已通过）。
校验：副本与影子函数在 32 条谱 × 6 字段上**逐字段一致（0 处不一致）**。

| 谱 | 旧 `amp` | 旧 `cand` | 新 `amp` | 新 `cand` | `slope_z` | `slope_im` |
|---|---|---|---|---|---|---|
| `example_caseE.csv` | False | False | **True** | **True** | −0.03 | −0.49 |
| `example_caseE_warburg.csv` | False | False | **True** | **True** | −0.02 | −0.45 |
| `example_caseF_wo.csv` | True | False | **False** | False | −0.54 | −0.97 |
| `example_caseF.csv` | False | False | False | False | +0.00 | — |
| `example_caseG_wo_only.csv` | True | False | **False** | False | −0.81 | −0.96 |

合成谱（20 条/类，1% 噪声，`is_candidate` 计数）：

| class | 旧 | 新 | 说明 |
|---|---|---|---|
| `ws_vis` | 12 | **18** | 正例，改善 |
| `ws_buried` | 0 | **10** | 正例，改善 |
| `wo_in` / `wshort_in` | 0 | 0 | 相位判据挡住，L1 本来就不管它们 |
| `wo_below` | 12 | **19** | 正例，改善 |
| `none_2zarc` / `rq05_in` / `sorp_rl` | 0 | 0 | 阴例，零假阳 |
| `rq05_low` | 7 | **18** | 阴例但**不可分**（§5.1），触发上升是预期的 |

18 条真实谱：旧 0/18、新 0/18，`f_cut` 变化 **0/18**——因为全部 `-Im<0`，
新判据早退，不硬判。

**三条必须一起说**：
1. 修掉 `caseE` / `caseE_warburg` 的漏报（这是 L1 的设计目标）；
2. `rq05_low` 触发数 7→18 是同形的代价，不是 bug；
3. **真 `Wo` 的 `amp_ok` 反而翻成 False**（Wo 低频 `-Im ∝ f⁻¹`，斜率 −0.97
   落在为 `Ws` 设的 `[-0.85,-0.20]` 之外）。它们的 `is_candidate` 本来就是
   False（相位判据挡住），所以**最终标志没变**，但 `why` 的解释文字会变。
   若要让 L1 管 `Wo`，带宽必须按端点走向分设（§8 P2）。

**下游影响（不能忽略）**：`kk_check` 用 `tail["f_cut"]` 决定要不要跑
`add_cap` 复验。`amp_ok` False→True 会让 `f_cut` 由 `None` 变成有值，从而打开
那条路径。所以这不是"只改注释"。补丁保留 `slope_z` 的 `|Z|` 口径
（`kk_check` 用它填 `tail_slope`），不随判据一起换成 `-Im`。

### 9.2 N3：18 条真实谱只读登记

见 §5.6。核心是**发现并补上 `to_inductive` 缺档**（`bm_wscrit.py` 的
`terminal` / `_FAMILY` / `_IND_TOL` / 自检 [7]）。补档后基准无回归：
`wshort_in` 家族命中回到 19/20 · 19/20，`sorp_rl` 20/20 判成
`inductive_tail`，阴例仍 **0 假阳**。

### 9.3 N2：Boukamp 梳齿指纹——理论对，判据不成

**理论（数值验证）**：`tanh(z)/z = Σ 2/(z²+π²(k−½)²)`、
`coth(z)/z = 1/z² + Σ 2/(z²+n²π²)`，误差按 `O(1/K)` 收敛
（K=200/2000/20000 → 3.73e-3 / 3.73e-4 / 3.73e-5，恒等式无误）。
齿参数：

| | `τ_k/τ_0` | `R_k/Z_0` |
|---|---|---|
| Wshort | 0.4053, 0.045, 0.0162, 0.0083, 0.0050 | **0.8106**, 0.0901, 0.0324, 0.0165, 0.0100 |
| Wo | 0.1013, 0.0253, 0.0113, 0.0063, 0.0041 | **0.2026**, 0.0507, 0.0225, 0.0127, 0.0081 |

**DRT 实测（`m=60`，61 点，7 decade）**：

| τ₀ | σ | chi2 | `R_total` | 峰数 | `τ_top/τ₀`（应 0.405） | `R_top/R_total`（应 0.811） | 次峰 τ 比（应 9） |
|---|---|---|---|---|---|---|---|
| 1e-3 | 1e-3 | 697 | 10.00 | 4 | 0.361 | **0.820** | 11.7 |
| 1e-2 | 1e-3 | 398 | 10.00 | 3 | 0.422 | **0.820** | 11.7 |
| 1e-1 | 1e-2 | 130 | 10.05 | 4 | 0.347 | **0.818** | 11.7 |

三条否证理由（都有数）：
1. **高次齿幅值掉太快**：第 2 齿 11%（Wshort）、第 3 齿 4%；1% 噪声下第 2 齿
   就没了。次峰 τ 比在 11.7–194 之间乱跳（预期 9），是网格/边沿伪峰。
2. **唯一稳的量 `R_top/R_total=0.82` 与单一 ZARC 重叠**：`zarc n=0.85` 给
   0.876（σ=1e-3）/0.924（σ=1e-2），`n=0.95` 给 0.946/0.965。差 0.05 而散度
   就有 0.03，**判不开**。
3. **`Wo` 根本不在 DRT 的表示能力里**：`1/z²` 是串联纯电容（DRT 是 Debye 之和，
   没有纯电容项），实测 chi2 = 4.1e4…2.9e5（`2n=122`），`R_total` 跑到 1e4–1e5。

**结论**：`Ws ≡ 窗外 R||CPE(0.5)` 的死结**没有**被解开。能留下的只有两个常数，
用途是**拟合后的一致性检查**，不是事前判据：
`τ_peak(DRT) = 0.4053 τ_D`（DRT 峰位比 `f_D` 低 **0.39 decade**）、
`R_基齿 = 0.811 Z_0`。

### 9.4 N4：可检测下限曲面（按 SNR，不按幅值比）

扫 `幅值比 0.1–1.0·R_p` × `σ = 0.25/0.5/1/2%` × 2 窗口 × 20 条 = **1280 条**。
`SNR_im = |Z_W(f_min)| / (σ·|Z(f_min)|)`（低频扩散项的信号 vs 噪声地板）。

`P(candidate)` 对 `SNR_im` 分箱（两窗口合并）：

| `SNR_im` | n | P(cand) | P(undec) | P(rej) |
|---|---|---|---|---|
| [3,5) | 43 | 0.09 | 0.33 | 0.58 |
| [5,8) | 93 | 0.08 | 0.40 | 0.53 |
| [8,12) | 100 | 0.19 | 0.30 | 0.51 |
| [12,20) | 204 | **0.53** | 0.21 | 0.26 |
| [20,40) | 332 | 0.76 | 0.14 | 0.10 |
| [40,∞) | 500 | **0.88** | 0.08 | 0.04 |

- `P(candidate)=50%` → `SNR_im ≈ 15`；**90% 达不到**（天花板 0.88，
  因为族内还有 `to_cap`/`noisy` 误档）；
- candidate 的 `SNR_im` 中位数 42.4，非 candidate 的 12.6；
- 用法：`SNR_im < 15` 的谱应当预期 `undecidable`，而不是判"没有扩散"。
  这正是 §5.3（"没看见 ≠ 没有"）的定量版本。

## 10. P1–P6 执行（2026-11-09，E2 跑完后自动接手）

### 10.1 P2：幅度带宽**不按**端点走向分设 —— 结论是"不改"

动机：N1 换成 `log(-Im)` 后，真 `Wo` 的低频段是容性尾（`-Im ∝ f^-1`，斜率
−0.97），落在为 `Ws` 设的 `[-0.85, -0.20]` 之外，`amp_ok` 由 True 翻成 False。
看上去"该按家族分设带宽"。实测否掉了这个想法。

三个变体（共用同一套相位判据 `bm_kk.nyquist_slope_low`，45°±0.35）：

| class | V0 现状 | V1 下界放宽到 −1.15 | V2 按 `terminal` 分档 |
|---|---|---|---|
| `ws_vis` | 0.90 | 0.90 | 0.90 |
| `ws_buried` | 0.50 | 0.50 | 0.50 |
| **`wo_in`** | **0.00** | **0.00** | **0.00** |
| `wo_below` | 0.95 | 0.95 | 0.95 |
| `wshort_in` | 0.00 | 0.00 | 0.00 |
| `none_2zarc` | 0.00 | 0.00 | 0.00 |
| `rq05_low` | 0.90 | 0.90 | 0.90 |
| `rq05_in` | 0.00 | 0.00 | 0.00 |
| `sorp_rl` | 0.00 | 0.00 | 0.00 |
| 三阴例均值 | 0.00 | 0.00 | 0.00 |

**三列逐行相同。** 原因不是带宽不够，而是被**相位判据**挡住了：`wo_in` 最低
25% 段的 Nyquist 斜率中位数 **−17.0**（范围 −42…+39），20 条里 **0 条**落在
45°±0.35 —— 到那一段时 45° 已经过去了，之后是电容。所以放宽幅度带宽换不到
任何真阳性，只会把容性弧的风险加进来。

**结构性结论**：L1（"最低 `frac` 段"口径）**天生只能做 `Ws`**；`Wo` 的 45°
段天生短（到拐点就结束），必须靠 `bm_wscrit` 的频段判据定位拐点前的段，
用 `family=Wo_like` 表达。生产代码不改。

> 注意口径：本表的 P(candidate) 是**L1 语义**（`-Im` 幅度 + Nyquist 45°），
> 和 §3 表里 `bm_wscrit` 的 verdict 不是同一个判据。`ws_vis` 在这里 0.90、
> §3 是 20/20，差的就是"段长/家族"那一层，不能混读。

产物：`evidence/ws_crit/p2_band.py`、`p2_band.json`。

### 10.2 P3：用 `to_inductive` 给吸附支路做预筛 —— 结论也是"不改"

动机：`bm_real.py:2585` 在 `--auto-adsorption` 下**无条件**调
`bm_ad.inv_from_lowfreq(f, Z)` 播种，而它的 `init_reliable` 只做三条结构性
检查（内部极大 / 峰在几何中心以下 / 突出度 ≥0.20），**不看低频尾是感抗还是
容性**。看上去"该加一道 `to_inductive` 预筛"。

18 条真实谱：`to_inductive` **18/18**、`init_reliable` **12/18**、同时成立
**12/18** → 预筛会挡掉 **0/18**，**零收益**（因为 18 条全是感抗尾）。

顺带量出两条可引用的事实：

- `tau_ad` 跨度 **0.43–11.45 s**，对应 `f_corner` **0.014–0.37 Hz**；
- **4/18 条**（`AZ31-4Li_0`、`AZ31-4Li_50`、`AZ31-8Li_100`、`AZ31-8Li_150`）
  的 `f_corner < 2·f_min`，即吸附环**只露了不到半支**，`tau_ad` 是窗口截断后的
  **下界**，不可当环的真实时间常数引用。这 4 条的 `init_reliable` 恰好都是
  `False`（突出度 0.09/0.04/0.00/−0.15）→ 现有结构性检查**已经**把它们挡了。

合成侧（9 类 × 20 条）：唯一分歧是 `wshort_in` **10/20** "误放行"
（`init_reliable=True` 但 `to_inductive=False`），`none_2zarc` 1/20；
合计误放行 11/180、预筛会漏 **0/180**。但 `init_reliable` 只决定**播种位置**，
不进先验、不进似然，且 `wshort_in` 在 `select_adsorption` 里因低频虚部残差
为负直接走 `capacitive_tail` 分支、根本不选吸附支路 → 那个 seed 用不上。

结论：预筛在真实谱上零收益、在合成侧只影响一个不会被采用的分支。
**不建议改门控**（要改也该等有非感抗尾却真含吸附支路的谱出现）。

产物：`evidence/ws_crit/p3_gate.py`、`p3_gate.json`。

### 10.3 P1：补丁改为 **E2 跑完后自动应用**（守护脚本）

E2 是逐 job 起 `subprocess.run`，所以 P1 只能在 E2 完成后落地。为了不再占用
人工等待，`evidence/ws_crit/p_after_e2.py` 做这件事：

1. 轮询 `out/m6/e2/e2_summary.json`（`run_e2.summarize()` 的最后一步）；
2. 确认 `bm_kk.py` 工作树干净（有别人改动则拒绝打补丁）；
3. 备份 `bm_kk.py` → `patches/bm_kk.py.pre_p1`，记录 sha256；
4. 在 **E2 用到的谱 + `examples/`** 上快照 `diffusion_tail_candidate` 六字段
   → `p1_before.json`（这是 E2 的**代码版本溯源**：30 个 job 全跑在补丁前）；
5. 取**回归基线**（`bm_check.py nofit` / `fast`）——必须先取，否则"补丁后
   rc≠0"分不清是补丁引入还是本来就没过；
6. `git apply` 补丁 → 快照 `p1_after.json` → 写 `p1_provenance.json`；
7. 校验真实函数与影子实现逐字段一致（`n1_shadow.candidate_new`）；
8. 再跑同一组回归，**只要比基线差或影子不一致就 `git checkout -- bm_kk.py`
   回滚**，并重跑 `nofit` 确认回到基线。

日志 `evidence/ws_crit/_p_after_e2.log`，报告 `evidence/ws_crit/P_AFTER_E2.md`。

### 10.4 P1 实跑记录（2026-11-09 14:31–14:32，`p_after_e2.py --now`）

| 步骤 | 结果 |
|---|---|
| 工作树 | `bm_kk.py` 干净（HEAD `88f6c8d`）；`git apply --check` 通过 |
| 备份 | `patches/bm_kk.py.pre_p1`，sha `b47f69d85595` |
| 回归基线（应用前） | `nofit` rc=0（5.2 s）、`fast` rc=0（20.8 s） |
| 应用 | sha `b47f69d85595` → `230114002295` |
| 溯源快照 | 31 条谱，**20 处变化 / 10 条谱** |
| 影子对拍 | 24 条谱 × 6 字段，**不一致 0 处** |
| 回归（应用后） | `nofit` rc=0（4.9 s）、`fast` rc=0（20.3 s） |
| 结论 | **P1 通过，补丁保留在工作树**（未 `git commit`，可随时 `git checkout --` 撤销） |

变化明细（`evidence/ws_crit/p1_provenance.json`）：

| 谱 | 字段变化 |
|---|---|
| 6 条 `Blank_0.1KBr_pH10_...`（同一份数据的副本） | 仅新增字段 `slope_im`：None → −0.110，判定结果不变 |
| `example_caseE` | `amp_ok`/`is_candidate` False→True，`f_cut` None→0.4299，`slope_im` −0.492 |
| `example_caseE_warburg` | 同上，`f_cut` None→0.8577，`slope_im` −0.448 |
| `example_caseF_wo` | `amp_ok` True→False，`f_cut` 0.8577→None，`slope_im` −0.975 |
| `example_caseG_wo_only` | 同上，`slope_im` −0.965 |
| **18 条真实 AZ31** | **六字段零变化**（低频 `-Im ≤ 0` 走早退分支，只报 `slope_z`） |

> 18 条 AZ31 零变化是**设计使然**：补丁只换幅度口径，不碰感抗尾。它们的
> `slope_im` 恒为 None、`amp_ok`/`is_candidate` 恒 False，`why` 记「幅度判据不可用」。
> 因此**补丁不改变 M6 / E2 的任何结论**；E2 的 30 个 job 也全部跑在补丁之前。

#### WP4 同步：docstring 按真值重写（14:36，只改注释）

补丁落地后 `bm_kk.diffusion_tail_candidate` 的 docstring 与行为脱节——原文按旧
`log|Z|` 口径写，还把 `example_caseE` 当成「无扩散元件、被幅度判据挡掉的假阳性」，
方向与真值相反。按 `WS_DECISION_PROJECT.md` WP4 只改注释、不改逻辑：

- 判据 A 改写为 `log(-Im)`，并写明「不用 `|Z|` 是因为 `R_inf` 只加实部」；
- 删掉「caseE 无扩散元件 / 被挡掉」的叙述，补 2026-11-09 注：真值是
  `R∞=10+ZARC(30Ω,300Hz,.85)+Ws(A_W=2)`，旧口径是**漏报**，新口径 −0.49 → 候选；
- 补 `-Im ≤ 0` 早退分支的说明（只填 `slope_z` 与相位字段，`why` 记「幅度判据不可用」）；
- 返回字段补 `slope_im`；`slope_z` 标注「只报数不判定（`kk_check` 用它填 `tail_slope`）」；
- 末尾两条盲区改写：`caseF_wo`/`caseG_wo_only` 在新旧口径下**都**给非候选（理由不同）；
  「相位过、幅度不可用」＝感抗尾，是**判不了**，不是「没有扩散」。

验证：sha `230114002295` → `d10b4c07d73a`；31 条谱六字段与 `p1_after.json`
**0 处差异**（行为未变）；`nofit` 18/18、`fast` 18 通过 / 0 失败 / 7 跳过。

### 10.5 P4：检测下限登记（纯文档，2026-11-09）

`WS_DECISION_PROJECT.md` 新增 **§3.1**，把 §9.4 的曲面写成可用条款：

- `SNR_im = |Z_W(f_min)| / (σ_rel·|Z(f_min)|)`；**50% 检测点 `SNR_im = 14.9`**
  （`limit_surface.json` 的 `limits.p50`），`candidate` 中位 42.4、非 `candidate` 12.6；
- **天花板 0.876**（`limits.p90 = null`）：族内 `to_cap`/`noisy` 误档，提高信噪比消不掉；
- `SNR_im ≲ 15` 应预期 `undecidable`，**不得**判「没有扩散」；
- 真实谱硬边界：`Z_W` 不可知 → 只有 `|Im|/(σ_rel|Z|)` 代理，无扩散时高估 →
  **只能事后诊断，不能事前筛选**；不参与真假阳性统计、不改 `G8b`。

不改代码、不改判据、不重跑任何拟合。

### 10.6 P6：L1'/L2/L3 接入（2026-11-09）

**结论先行**：L1'、L2 **判据与接口都完善，已接入**；L3 **只挂诊断、不接闸门**。

改动：

- 新建 `bm_ws.py`（由 `evidence/ws_crit/bm_wscrit.py` 派生，**只改文件头定位**，
  判据一行未动；自检 7/7）；
- `bm_real.diffusion_gate(..., f, Z)`：新增 `ws_diag`（端点走向 / 家族 / verdict /
  段长 / 斜率）与 `status="inductive_tail"`；`to_inductive` 优先于「非候选」判定；
- 触发条件：`is_candidate`（`gate.status=="candidate"`）与残差条件取**或**；
- 感抗尾短路：`status="skipped_inductive"`，不补跑 Ws（`--auto-mechanism-full` 除外）；
- `select_mechanism(..., share=...)`：`ws_share_low` 只写 `note`，**不改 choice**。

验证：

| 项 | 结果 |
|---|---|
| 迁移对拍 | 31 条谱 × 9 字段 + 8 组 npz × 2 函数，**0 差异**（`relocate_check.json`） |
| L1' 分档 | AZ31→`inductive_tail`；caseE 两条→`candidate`；`caseF_wo`→`no_candidate`（数据侧 `Wo_like` 不带动门控）；简并优先 `degenerate` |
| L2 一致性 | 31 条谱 `verdict==candidate` 与 `enter` 不一致 **0** |
| 实跑 `caseE` | `switched`，Δelpd=131.93 ≥ 25.83，与接入前逐位相同 |
| 实跑 `AZ31-4Li_100` | 拟合 2→1 次，`skipped_inductive` 保持 none |
| 回归 | `nofit` 18/18、`fast` 18/0/7、`bmdrt.py check` 18/18、单测 4/4+10/10+6/6 |

L3 不接闸门的理由（实测份额）：

| 组 | `f_min` | `share` 中位 [90% CI] |
|---|---|---|
| `caseE` 真 Ws | 0.01 Hz | 0.161 [0.157, 0.165] |
| `caseE_warburg` 同一真值 | 0.1 Hz | **0.057** [0.053, 0.061] |
| `caseF_wo` 错元件（Ws 拟合） | 0.1 Hz | 0.011 [0.004, 0.085] |

真 Ws 在窄窗距 5% 阈值只剩 0.7 个百分点 → 阈值不是窗口不变量。要接闸门，先做
「份额 × 窗口」标定（或改用 `Re Z_Ws(f_min)/σ_im` 的信噪比口径，与 §9.4 同源）。

**默认路径逐位不变**：`--auto-mechanism` 默认关，`diffusion_gate` 只在该分支被调用；
`select_k` / `select_mechanism` 的决策规则未改。
