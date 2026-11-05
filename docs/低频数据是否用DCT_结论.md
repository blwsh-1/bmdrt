# 低频数据用 DCT 吗

> 结论：**不用，也不该用**。我们缺的不是"低频反卷积工具"，而是"扩散对照的决策分支"。
> 代码证据来自 `D:\eis训练数据\bmdrt\`；本次只读，未改代码。

---

## 0. 先澄清术语：DCT 不是"低频专用工具"

按 *DeCarbon* 2026 综述《Review of timescale distributions for electrochemical impedance spectroscopy analysis: Advantages, synergies, and future directions》的统一框架，EIS 有**四种**特征时间尺度分布：

| 分布 | 全称 | 反卷积对象 | 峰面积含义 | 失效边界 |
|---|---|---|---|---|
| **DRT** | Distribution of Relaxation Times | **阻抗** `Z(ω)` 对 τ 积分，核 `1/(1+jωτ)` | 电阻 | **低频阻抗发散**时在高 τ 端发散 |
| **DCT** | Distribution of **Capacitive** Times | **导纳** `Y(ω)` 对 τ 积分（Maxwell **并联**组合），形式与 DRT 类似但符号相反 | **电导** | **高频阻抗→0** 时在低 τ 端发散 |
| DDC | Distribution of Differential Capacity | 微分电容 `C(ω)`；`γ_DDC = e^{logτ}·γ_DCT` | 电容 | 低频发散时在高 τ 端发散 |
| **DDT** | Distribution of **Diffusion** Times | 导纳对扩散时间积分，核基于**有限长度扩散/传输线模型** | — | 无闭式反演，计算最贵；**专门适用于低频阻抗无界的谱** |

**所以：**
- UP主把 DCT 挂在低频，本质是"**换到导纳域**做反卷积，从而避开 DRT 在低频发散时失效"——是一个工程上有效的 workaround。
- 学术上更准确对应"低频扩散"的是 **DDT**，不是 DCT。DCT 只是"低频有界性更好"的替代之一。
- 综述给的协同策略是：**联合恢复多种分布**，或**把数据截断到各分布各自有效的频段**。
- 综述强调的工作流首步仍是 **K-K 检验**（避免实验伪影被带进分布），以及后处理的全频重建对比与渐近行为检查。

> 注：DCT 用于锂氧电池低频解析的原始工作见 Chen J. et al., *Chem* **2023**, 9(8), 2267–2281（doi:10.1016/j.chempr.2023.04.022），即上一轮文档 [11]。

---

## 1. 我们的算法里，低频是怎么处理的

**全在阻抗域，没有导纳域反卷积，因此没有 DCT，也没有 DDT。**

- 似然只用原始复阻抗：`Normal(Zt.real, s_re)` + `Normal(Zt.imag, s_im)`（`bm_core.py:291-292`）
- 低频靠两类东西表达：
  1. **HN 基底** `Z=R/[1+(jωτ)^φ]^ζ`（默认 ζ=1，即 ZARC）——可以把峰推向极低 τ 去"冒充"扩散尾
  2. **可选的显式 Warburg 族元件**（`bm_diff.py`，解析式齐备）：
     - `W_inf` 半无限：`A_W/√(jω)`
     - `W_open` 有限长阻挡：`R_W·coth(z)/z`
     - `W_short` 有限长透射：`R_W·tanh(z)/z`，`z=√(jωτ_D)`，`τ_D=L²/D`
     - 开关：`--diffusion none|Ws|Wo|both`，**默认 none**

### 关键事实：117 条谱的批量运行，扩散项一项都没开

- `batch_2e.py:101-106` 构造的命令只有 `bmdrt.py auto <csv> --out ... --no-move [--quick]`，**没有 `--diffusion`**
- `report_2e.py:172-173` 白纸黑字写着：
  > 「基础模型默认 `ζ=1`（ZARC 基底）、`diffusion=none`；低频 −45° 尾若要显式建模，需要单独跑 `--diffusion Ws/Wo` 对照，**本批没做**。」

### 后果：正是 `bm_diff.py:10-18` 自己警告的退化

> 「单个 HN 在有限频率窗口内能逼近半无限 Warburg 到 **1.2e-6** 相对误差，代价是 `φζ=0.5`、`R=A_W·√τ`、**f_peak 在窗口下方 8 个量级**……R 与 τ 沿 `lnR − ½lnτ = lnA_W` 这条脊线完全简并……把 4 个无意义参数换 1 个有意义的 `A_W`，才是加显式扩散项的理由——这是**重参数化 + 可辨识性修复**，不是表达力扩展。」

这正好解释了 full 档的现象：**37 条端部堆积、18 条"原始 0 峰"、18 条 `P(峰位出窗)>0.5`、5 mA 组 13/13 全出窗但采样完美（grade=2 / div=0 / R̂<1.005）**。不是采样问题，是 HN 分量被逼到窗口外去吸收低频扩散尾。

而且代码**已经知道该怎么做**：`bm_real.py:880-885` 在检测到"分量在代替扩散尾"指纹时打印

> `** 先加 --diffusion Ws 做对照，再判断 K **`

——但那只是一句打印，`judge()` 的结果不进 `select_k`，扩散项也不在模型比较的维度里。

---

## 2. 该不该引入 DCT？判断：不该

**理由是架构不匹配，不是"实现难度"。**

| 维度 | DCT（无模型反卷积） | 我们的 BM-DRT（参数化贝叶斯） |
|---|---|---|
| 输出 | 一个分布函数 γ_DCT(logτ) | 一组参数后验 + K |
| 求解 | 导纳域岭回归/贝叶斯反卷积 | 阻抗域 NUTS + HN 混合 |
| 不确定性 | 需另做（GP/贝叶斯反卷积） | 后验自带 |
| 进似然 | 不能（不是参数） | 本来就是似然主体 |
| 用 PSIS-LOO 与基线比较 | 不行 | 可以 |

挂一套 DCT 进来，等于在贝叶斯框架外再摆一套口径不同的结果：**不能进同一个似然、不能用 elpd 与 `diffusion=none` 基线比较、无法回答"该不该加扩散"这个真正的问题**。反而会破坏现有"一切结论由逐点似然 + LOO 比较选出"的可复核性。

更重要的是：**我们不缺低频表达能力**。`bm_diff` 的 Warburg 族解析式齐备、参数化后自带后验区间，比一个反卷积分布更适配贝叶斯框架。缺的是**开关 + 决策**，不是新工具。

> 唯一值得从 DCT 借鉴的是它的**视角**：把 `Z → Y = 1/Z` 看低频是否发散（−45° 尾、`|Y|` 是否随 `√ω` 走）。这可以做成廉价的**前置体检指标**，用来判断"该不该开扩散项"，而不是做成反卷积模块。

---

## 3. 该怎么做（对应上一轮评估文档的 P0）

把 `bm_real.py:884` 那句提示升级成**决策**，而不是补 DCT：

```
① 前置体检（廉价）
   K-K / Lin-KK（AutoECM/validate_fit.py:58-77 已有 lin_kk 可复用）
 + 低频体检：Y=1/Z 的渐近、端部堆积、HN↔Warburg 指纹
   → 标记"低频可能发散 / 有未分辨扩散"
        │
② 若被标记 → 把 diffusion 纳入模型比较维度
   同一套逐频率点联合似然 + PSIS-LOO，比较
       (K, diffusion=none) vs (K, diffusion=Ws) vs (K, diffusion=Wo)
   而不是只扫 K
        │
③ 判定
   ├─ 加扩散后 K 稳定、峰位回窗、elpd 明显更好 → 采用带扩散模型
   ├─ 加扩散后仍有分量出窗/简并 → 数据里可能不止一种扩散，扩大对照
   └─ K 随对照漂移 → **拒绝定 K**，标注"低频发散、窗口未覆盖直流极限"
        │
④ 把 diffusion、低频体检指标、对照 elpd 一起写进 real_*.json
```

**改动面**：`bm_diff` 元件已就绪、`--diffusion` 参数已存在，只需要在 `run()` 里按体检结果增加一次对照拟合、把 `diffusion` 纳入 `select_k` 的比较维度、`save_out` 加字段。不需要任何反卷积代码。

**顺带修正一处报告口径**：`report_2e.py:172-173` 已经诚实写了"本批没做扩散对照"，但结论里若直接给出 K 和峰位，读者容易忽略这一限制。建议把"低频发散谱的 K 不可引用"从"使用限制"提升到摘要层。

---

## 4. 与上一轮 DRT 文献 SOP 的对照（更新）

| UP主 / 文献做法 | 我们的对应 | 评价 |
|---|---|---|
| 低频 45° 尾扣掉或交 **DCT** | 不截断、不反卷积；用 HN 吸收 + 可选 Warburg 元件 | **思路更适配贝叶斯框架**，但本批没开扩散项，等于既没截断也没建模 |
| 分频域 RRDRT / DRT / DCT | 单一阻抗域 HN 混合 + 可选 L + 可选 Ws/Wo | 参数化统一、可用 LOO 比较；但 diffusion 未进模型比较 |
| 全频重建对比、渐近行为检查（综述 §3.3.4） | PPC 覆盖、分频段残差、趋势零分布 | 我们有等价甚至更强的诊断，只是不触发决策 |

---

## 5. 参考资料

- *Review of timescale distributions for electrochemical impedance spectroscopy analysis: Advantages, synergies, and future directions*, **DeCarbon**, 2026（DRT/DCT/DDC/DDT 统一框架、适用边界、协同工作流）
- Chen J., Quattrocchi E., Ciucci F., et al. *Charging processes in lithium-oxygen batteries unraveled through the lens of the distribution of relaxation times*. **Chem** 2023, 9(8), 2267–2281（DCT 解析低频）
- `bm_diff.py:1-65`（Warburg 族元件、HN↔Warburg 退化的完整论证）
- `report_2e.py:165-175`（本批"没做扩散对照"的正式声明）

---

*本次为只读代码审查，未修改 `D:\eis训练数据\bmdrt\` 下任何文件。*
