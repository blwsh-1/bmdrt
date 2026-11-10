# 证据卡样例（只读生成，来自真实结果）

> 生成：`python bm_evidence.py <json> --md`；多模型表：`--multi-model`（config 默认关闭，本样例用 API 打开）。
> 本文件由 `evidence/ws_crit/make_cards.py` 生成，可随时重跑覆盖。

> 证据卡不排序、不打分、不单选；每条证据自带「这条推不出什么」
> 资格档：**A** 算法无法判断（重跑/修代码）｜**B** 数据无法判断（并列，不得单选）｜**C** 物理证据不足（只报相容）。**不是分数**。

## Blank_0.1KBr（--auto-mechanism-full，新跑）

来源：`out/_card_probe/Blank_0.1KBr_pH10_EIS_PtCu_1.2V/real_Blank_0.1KBr_pH10_EIS_PtCu_1.2V.json`

| # | 卡 | 资格档 |
|---|---|---|
| 1 | diffusion | B |
| 2 | adsorption | A |
| 3 | component_2 | C |
| 4 | k_selection | C |

### 卡 1｜本谱低频尾需要 ω^(-1/2) 扩散项（Ws）

- **资格档：B** — 数据无法判断（并列输出、保留多候选；不得单选，也不得说「不存在」）
- 判据：由 guards: diff_amp_unidentifiable、ws_amp_unidentified 判 B：数据无法判断（并列输出、保留多候选；不得单选，也不得说「不存在」）
- 判据来源：字段（选择器已写）
- **整卡推不出**：推不出「某机制不存在」：本窗+本噪声下只能给「不可区分」或「幅值定不住」，不能反推不存在

已采到的证据：

- **`gate_shape`** status=no_candidate，enter=否
  - 可以读成：门控看到的低频尾形状（45° 候选 / 感抗尾 / 无候选 / HN 已退化）
  - **推不出**：门控跳过或判感抗尾，只否证「本窗内可见的电容性扩散尾」；不等于全谱没有扩散，也不等于扩散不存在
  - 来源：`auto_mechanism.gate.status`
  - 备注：KK 低频尾不是 45° 候选（必要不充分），不触发 Ws
- **`aw_amp`** A_W=0.7693，sd=0.3479，cv=0.4522，identifiable=否
  - 可以读成：A_W 后验中位数、标准误与相对不确定度 cv
  - **推不出**：「A_W > 0」不等于扩散存在；「A_W 小 / cv 大」也不等于不存在——只说明幅值没被本窗+本噪声定住
  - 来源：`auto_mechanism.share（bm_ws.ws_share_low）`
- **`paired_loo`** delta=0.01372，dse=1.32，threshold=2，choice=none，status=guard_degraded，n=50
  - 可以读成：两模型逐点配对 LOO 的 Δelpd ± dse 与判决门槛
  - **推不出**：Δ 过线只说明「本窗预测更好」，不能推出「该机制为真」
  - 来源：`auto_mechanism.delta / dse / threshold`
- **`param_corr`** n_strong=1，top=ln τ_2↔ln w_2 ρ=0.9413，level=degenerate
  - 可以读成：参数相关矩阵里的强相关对（|ρ| ≥ 0.9）
  - **推不出**：两个参数强相关不等于两个机制等价；只说明边际分布不能单独引用
  - 来源：`param_corr.strong / medium`
  - 备注：1 对参数 |ρ|≥0.90：这些参数只能定出一个组合，**不要单独引用其中任何一个**。
- **`degeneracy`** n_warburg_like=0，at_bound_max=0
  - 可以读成：HN 分量是否已呈 Warburg 退化指纹（ζ→1、峰位、implied A）
  - **推不出**：HN 与 Ws 同形不等于「扩散就是原因」；同形只说明这段自由度已被占
  - 来源：`degeneracy / n_warburg_like`
- **`residual`** acf=0.3811，trend_re_sigma=0.7538，trend_im_sigma=1.087，systematic=是
  - 可以读成：残差 RMS、ACF、趋势是否已到噪声水平
  - **推不出**：拟合到噪声水平不等于机制正确：多个模型可以同时拟合到噪声
  - 来源：`component_review.residual_review`
- **`kk`** status=ok，mu=1，M=11
  - 可以读成：Lin-KK 是否通过（数据自洽性）
  - **推不出**：KK 通过只说数据满足因果/线性，不说明任何模型或机制正确
  - 来源：`kk`
  - 备注：Lin-KK μ=1.00、M=11，中段相对残差 0.3% ≤ 5.0% → 数据自洽

未采到 / 不可用：

- `lowf_slope`（不可用） slope_im=-0.1103，r2=0.8884，band_dec=0.6122，term=to_plateau；备注：斜率不可用（-Im 在低频端 ≤0 或线性度不足） — **推不出**：斜率 ≠ −0.5 不能推出「没有 Ws」：弧尾共存时真 Ws 的斜率会被拖到 −0.41…−0.49（WP2 §3.5.1）
- `freq_window`（缺）；备注：本 run 落盘的是所选模型（none），Ws 臂的 τ_D 后验没保存；auto_mechanism.ws_resolvable=False。要看 f_D 需要把 Ws 臂结果单独落盘 — **推不出**：f_D 在窗外不能推出「没有该过程」；只能说「本窗无权评价」（C 档）
- `cross_model`（缺）；备注：单臂 JSON 只有自己；要跨模型差跑 multi_model — **推不出**：预测重叠不能推出「机制等价」；只说明本窗+本噪声不可分（B 档）

还缺什么（next_evidence_needed）：

- 把 f_min 下移 1 decade，让 45° 段变长
- cv(A_W)=0.45 > 0.15：幅值没被定住，重跑不会变，需要更宽窗或更低噪声
- 补测使 f_D 进窗：转折在窗下就下移 f_min，在窗上就上移 f_max
- 同一谱上的 Ws/Wo/K+1 配对 LOO 与逐点预测差（multi_model 表）
- 并列保留候选模型，不要单选；要分开需更宽频窗或独立重复实验

### 卡 2｜低频感性特征需要并联吸附支路（R ∥ L）

- **资格档：A** — 算法无法判断（修代码 / 重跑 / 提高预算；不得当数据结论）
- 判据：由 status=not_tested 判 A：算法无法判断（修代码 / 重跑 / 提高预算；不得当数据结论）
- 判据来源：按状态表派生（该结论没有选择器级 grade）
- **整卡推不出**：推不出任何数据侧结论：采样/LOO/拟合未通过，先修或重跑
- 备注：该谱未运行 --auto-adsorption，吸附支路未检验

未采到 / 不可用：

- `ad_gate_shape`（缺） — **推不出**：门控没进不等于「没有吸附」；只说明本窗内没看到完整的感性环形状
- `ad_amp`（缺） — — **推不出**：R_ad 小不等于没有吸附：幅值不可辨识只说明定不住，不说明不存在
- `ad_corner`（缺） — — **推不出**：f_c 在窗外不等于没有吸附；串联 L 只是本窗内的等效描述
- `ad_paired_loo`（缺） — **推不出**：Δ 过线只说明本窗预测更好，不能推出吸附机制为真

还缺什么（next_evidence_needed）：

- 更宽频窗或更低 f_min，覆盖吸附环的完整半圆
- 更低噪声 / 更长采样，把 R_ad 从噪声里拉出来
- 把 f_min 下移，让 f_c 进窗
- 独立实验（浓度 / 电位系列）验证吸附
- 先修/重跑：A 档是工程问题，重跑一次可能就变（不是数据结论）

### 卡 3｜component_2 对应一个独立物理弛豫过程

- **资格档：C** — 物理证据不足（只报与物理相容；不得升级为机制结论）
- 判据：峰位出窗（P=0.99）判 C：参数定得住、预测算得出，但本窗没覆盖该特征频率，无权说该过程存在与否
- 判据来源：字段（选择器已写）
- **整卡推不出**：推不出「机制已被证实」：模型比较有结论，但缺真值/独立实验/正交表征
- 备注：组分审查不改变 K；这条卡只说明该分量不被本窗约束，不是「分量不存在」

已采到的证据：

- **`component_peak`** state=out_of_window，p_peak_out=0.9877，f0_hz=0.07134，w_ohm=10.02
  - 可以读成：该分量峰位后验出窗概率 p_peak_out 与 f0
  - **推不出**：峰位在窗外不能推出「没有这个分量」；只能说本窗没覆盖它的特征频率（C 档）
  - 来源：`component_review.window_contribution`
- **`component_contrib`** frac_abs_at_fmin=0.08641，frac_im_at_fmin=-0.9488，frac_abs_at_geom=0.008672
  - 可以读成：该分量在 f_min / 几何中心处对总阻抗的贡献（幅值与虚部占比）
  - **推不出**：窗内有贡献不等于该分量是独立的物理过程；贡献大也可能只是拟合自由度
  - 来源：`component_review.window_contribution`
- **`component_param`** ln τ_2↔ln w_2 ρ=0.9413，n=1
  - 可以读成：该分量的 ln τ / ln w 与其他参数的强相关对
  - **推不出**：强相关不等于该分量不存在；只说明这些参数只能定出一个组合
  - 来源：`component_review.parameter_identifiability`
- **`component_degeneracy`** level=no，at_bound=0，phi=0.5103，a_impl=6.683
  - 可以读成：该分量的退化指纹（at_bound、φ 后验、implied A）
  - **推不出**：退化指纹不等于该分量就是 Warburg；只说明它和扩散同形
  - 来源：`degeneracy`

还缺什么（next_evidence_needed）：

- 峰位在窗外：本窗没覆盖它的特征频率（C 档）
- τ–w 强相关：两个参数只能定出一个组合
- 机制唯一性证据：独立重复实验 / 正交表征 / 真值

### 卡 4｜K=2 个 RC 分量是本频窗内的模型选择结果

- **资格档：C** — 物理证据不足（只报与物理相容；不得升级为机制结论）
- 判据：由 k_selection_status=converged 派生 C：K=2 是 L2 模型比较结论，缺「过程数=K」的唯一性证据
- 判据来源：按状态表派生（该结论没有选择器级 grade）
- **整卡推不出**：推不出「机制已被证实」：模型比较有结论，但缺真值/独立实验/正交表征
- 备注：K 是模型选择结论，不是物理过程数；引用 K 时要同时引用组分审查

已采到的证据：

- **`k_loo`** K=2，elpd_diff=0，dse=0
  - 可以读成：K 的配对 LOO 差与 dse
  - **推不出**：K 选出来只说明本窗预测最好，不说明有 K 个物理过程
  - 来源：`elpd_diff / dse（arviz：正数=更差）`
  - 备注：elpd_diff 为 arviz 口径：正数表示相对最好者更差
- **`k_hard`** sample_grade=2，frac_neg=0，pareto_k_max=0.7491
  - 可以读成：采样分级、硬筛理由（Pareto-k、φζ>1 占比、LOO 可用性）
  - **推不出**：采样通过不等于模型正确；它只说明这次拟合可信
  - 来源：`sample_grade / grade_fail / frac_neg / pareto_k`
- **`k_residual`** cover_re=0.94，cover_im=0.94，acf=0.3811，systematic=是
  - 可以读成：PPC 覆盖率、ACF 与残差趋势
  - **推不出**：残差白不等于机制正确；多个 K 可以同样白
  - 来源：`component_review.residual_review / cover_re / cover_im`
- **`k_components`** component_1:in_window、component_2:out_of_window，n_out=1
  - 可以读成：各分量峰位是否在窗内（in / uncertain / out）
  - **推不出**：分量状态是「本窗可辨识性」，不是「物理过程存在性」
  - 来源：`component_review.component_identifiability`

还缺什么（next_evidence_needed）：

- 机制唯一性证据：独立重复实验 / 正交表征 / 真值

## AZ31-4Li_100（--auto-mechanism，旧 JSON：grade 由同表补算）

来源：`out/_l123/az31/AZ31-4Li_100/real_AZ31-4Li_100.json`

| # | 卡 | 资格档 |
|---|---|---|
| 1 | diffusion | B |
| 2 | adsorption | A |
| 3 | k_selection | A |

### 卡 1｜本谱低频尾需要 ω^(-1/2) 扩散项（Ws）

- **资格档：B** — 数据无法判断（并列输出、保留多候选；不得单选，也不得说「不存在」）
- 判据：由 guards: diff_amp_unidentifiable 判 B：数据无法判断（并列输出、保留多候选；不得单选，也不得说「不存在」）
- 判据来源：按 bm_grade 同一张表补算（原 JSON 无 grade 字段）
- **整卡推不出**：推不出「某机制不存在」：本窗+本噪声下只能给「不可区分」或「幅值定不住」，不能反推不存在

已采到的证据：

- **`gate_shape`** status=inductive_tail，enter=否
  - 可以读成：门控看到的低频尾形状（45° 候选 / 感抗尾 / 无候选 / HN 已退化）
  - **推不出**：门控跳过或判感抗尾，只否证「本窗内可见的电容性扩散尾」；不等于全谱没有扩散，也不等于扩散不存在
  - 来源：`auto_mechanism.gate.status`
  - 备注：低频 `-Im<0`（感抗尾，数据侧判据）：三类扩散尾都要求`-Im>0` 且按 f^(-1/2)~f^(-1) 增长，符号不符即否证，不用再看 45° 段长；该特征走吸附支路（R∥L）
- **`paired_loo`** delta=-2.187，dse=0.4573，threshold=2，choice=none，status=guard_degraded，n=50
  - 可以读成：两模型逐点配对 LOO 的 Δelpd ± dse 与判决门槛
  - **推不出**：Δ 过线只说明「本窗预测更好」，不能推出「该机制为真」
  - 来源：`auto_mechanism.delta / dse / threshold`
- **`param_corr`** n_strong=0，level=identified
  - 可以读成：参数相关矩阵里的强相关对（|ρ| ≥ 0.9）
  - **推不出**：两个参数强相关不等于两个机制等价；只说明边际分布不能单独引用
  - 来源：`param_corr.strong / medium`
  - 备注：没有 |ρ|≥0.90 的参数对，边际区间可以单独引用。
- **`degeneracy`** n_warburg_like=0，at_bound_max=0
  - 可以读成：HN 分量是否已呈 Warburg 退化指纹（ζ→1、峰位、implied A）
  - **推不出**：HN 与 Ws 同形不等于「扩散就是原因」；同形只说明这段自由度已被占
  - 来源：`degeneracy / n_warburg_like`
- **`residual`** acf=0.9088，trend_re_sigma=1.388，trend_im_sigma=1.678，systematic=是
  - 可以读成：残差 RMS、ACF、趋势是否已到噪声水平
  - **推不出**：拟合到噪声水平不等于机制正确：多个模型可以同时拟合到噪声
  - 来源：`component_review.residual_review`
- **`kk`** status=warn，mu=0.151，M=12
  - 可以读成：Lin-KK 是否通过（数据自洽性）
  - **推不出**：KK 通过只说数据满足因果/线性，不说明任何模型或机制正确
  - 来源：`kk`
  - 备注：中段残差 2.4% ≤ 5.0% 正常，但端部需归因（见 attribution）→ 可拟合，引用参数时须说明端部处理

未采到 / 不可用：

- `lowf_slope`（不可用） band_dec=0.1429，term=to_inductive；备注：斜率不可用（-Im 在低频端 ≤0 或线性度不足） — **推不出**：斜率 ≠ −0.5 不能推出「没有 Ws」：弧尾共存时真 Ws 的斜率会被拖到 −0.41…−0.49（WP2 §3.5.1）
- `aw_amp`（不可用） A_W=13.5；备注：只有 A_W 点估计，没有 sd/cv（该 run 的 share 未算 cv_aw） — **推不出**：「A_W > 0」不等于扩散存在；「A_W 小 / cv 大」也不等于不存在——只说明幅值没被本窗+本噪声定住
- `freq_window`（缺）；备注：本 run 落盘的是所选模型（none），Ws 臂的 τ_D 后验没保存；auto_mechanism.ws_resolvable=False。要看 f_D 需要把 Ws 臂结果单独落盘 — **推不出**：f_D 在窗外不能推出「没有该过程」；只能说「本窗无权评价」（C 档）
- `cross_model`（缺）；备注：单臂 JSON 只有自己；要跨模型差跑 multi_model — **推不出**：预测重叠不能推出「机制等价」；只说明本窗+本噪声不可分（B 档）

还缺什么（next_evidence_needed）：

- 把 f_min 下移 1 decade，让 45° 段变长
- 更长采样或更低噪声，把 cv(A_W) 压到判据线以下
- 补测使 f_D 进窗：转折在窗下就下移 f_min，在窗上就上移 f_max
- 同一谱上的 Ws/Wo/K+1 配对 LOO 与逐点预测差（multi_model 表）
- 并列保留候选模型，不要单选；要分开需更宽频窗或独立重复实验

### 卡 2｜低频感性特征需要并联吸附支路（R ∥ L）

- **资格档：A** — 算法无法判断（修代码 / 重跑 / 提高预算；不得当数据结论）
- 判据：由 status=not_tested 判 A：算法无法判断（修代码 / 重跑 / 提高预算；不得当数据结论）
- 判据来源：按状态表派生（该结论没有选择器级 grade）
- **整卡推不出**：推不出任何数据侧结论：采样/LOO/拟合未通过，先修或重跑
- 备注：该谱未运行 --auto-adsorption，吸附支路未检验

未采到 / 不可用：

- `ad_gate_shape`（缺） — **推不出**：门控没进不等于「没有吸附」；只说明本窗内没看到完整的感性环形状
- `ad_amp`（缺） — — **推不出**：R_ad 小不等于没有吸附：幅值不可辨识只说明定不住，不说明不存在
- `ad_corner`（缺） — — **推不出**：f_c 在窗外不等于没有吸附；串联 L 只是本窗内的等效描述
- `ad_paired_loo`（缺） — **推不出**：Δ 过线只说明本窗预测更好，不能推出吸附机制为真

还缺什么（next_evidence_needed）：

- 更宽频窗或更低 f_min，覆盖吸附环的完整半圆
- 更低噪声 / 更长采样，把 R_ad 从噪声里拉出来
- 把 f_min 下移，让 f_c 进窗
- 独立实验（浓度 / 电位系列）验证吸附
- 先修/重跑：A 档是工程问题，重跑一次可能就变（不是数据结论）

### 卡 3｜K=1 个 RC 分量是本频窗内的模型选择结果

- **资格档：A** — 算法无法判断（修代码 / 重跑 / 提高预算；不得当数据结论）
- 判据：由 k_selection_status=not_compared 派生 A：该状态不是可引用的 K 结论
- 判据来源：按状态表派生（该结论没有选择器级 grade）
- **整卡推不出**：推不出任何数据侧结论：采样/LOO/拟合未通过，先修或重跑
- 备注：K 是模型选择结论，不是物理过程数；引用 K 时要同时引用组分审查

已采到的证据：

- **`k_loo`** K=1，elpd_diff=0
  - 可以读成：K 的配对 LOO 差与 dse
  - **推不出**：K 选出来只说明本窗预测最好，不说明有 K 个物理过程
  - 来源：`elpd_diff / dse（arviz：正数=更差）`
  - 备注：elpd_diff 为 arviz 口径：正数表示相对最好者更差
- **`k_hard`** sample_grade=2，frac_neg=0，pareto_k_max=0.4363
  - 可以读成：采样分级、硬筛理由（Pareto-k、φζ>1 占比、LOO 可用性）
  - **推不出**：采样通过不等于模型正确；它只说明这次拟合可信
  - 来源：`sample_grade / grade_fail / frac_neg / pareto_k`
- **`k_residual`** cover_re=0.86，cover_im=0.86，acf=0.9088，systematic=是
  - 可以读成：PPC 覆盖率、ACF 与残差趋势
  - **推不出**：残差白不等于机制正确；多个 K 可以同样白
  - 来源：`component_review.residual_review / cover_re / cover_im`
- **`k_components`** component_1:in_window，n_out=0
  - 可以读成：各分量峰位是否在窗内（in / uncertain / out）
  - **推不出**：分量状态是「本窗可辨识性」，不是「物理过程存在性」
  - 来源：`component_review.component_identifiability`

还缺什么（next_evidence_needed）：

- 先修/重跑：A 档是工程问题，重跑一次可能就变（不是数据结论）

## L2 多模型并列（开关已打开）

> 按输入顺序列出，不排序、不打分、不单选；每行的 Δ 只相对它自己的 none 基线，跨臂不做配对检验（那需要 WP2 式的逐点样本）

| arm | K | 扩散 | elpd ± se | Δ vs none | dse | 档 | A_W | 塌陷/出窗标记 |
|---|---|---|---|---|---|---|---|---|
| k1_none | 1 | none | -116.6 ± 10.82 | — | — | — | — | — |
| k1_Ws | 1 | Ws | -6.737 ± 13.04 | — | — | — | 4.184 | — |
| k2_none | 2 | none | 91.22 ± 8.945 | — | — | — | — | 峰位出窗：component_2 |
| k2_Ws | 2 | Ws | 90.81 ± 9.147 | — | — | — | 0.7779 | 幅值落到噪声（P=0.53） |

跨臂 A_W 范围：0.7779 – 4.184（5.4×）。跨臂 A_W 范围只作描述：哪个 arm 塌到小值由人看，程序不判「哪个对」
