# 外部基线盘点：ADAPT-ECM

> 2026-11-12 · 用于回答"我们用贝叶斯 DRT，是不是没有优势"这类问题
> 依据：Zenodo 记录、GitHub README、配套论文 **＋ 本地 4.0.1 源码核对与实测**。
> 本文件是**盘点与定位**；源码核行、逐谱数字与对照结论见
> `round_2026-11-12/ADAPT_BASELINE.md`（同轮证据脚本 `adapt_baseline.py`）。

## 1. 它是什么

**Bosah, O., & Kalu, E. E. (2026).** *DRT-guided constraints and CI-admissibility
for series-based equivalent-circuit modeling of EIS.* Electrochimica Acta 571, 149057.
配套软件 `ADAPT-ECM` 4.0.1（Zenodo 2026-10-04，GitHub `Bosahobi/ADAPT-ECM`，MIT）。

一个 Tkinter 桌面工作台，把 **DRT → 选拓扑 → 种子初值/边界 → ECM 拟合**
串成一条流程。它同时提供三档 DRT：

| 档位 | 基底 / 方法 | 关键引用 |
|---|---|---|
| Simple DRT | 径向基函数 + 重建平台法选 λ | Saccoccio 2014；Wan 2015 |
| Bayesian DRT | MAP/Laplace + **HMC** 后验，分层贝叶斯正则 | Ciucci & Chen 2015；Effat & Ciucci 2017 |
| **BM-DRT** | **贝叶斯混合，HN 基函数 + 可选 Warburg** | **Liu & Ciucci 2025, JPCC 129: 20268** |

外加：自适应 Lin-KK（Liu/Wan/Ciucci 2020）、MC 峰值不确定性、跨谱峰
persistence map、λ 稳定性汇总、`impedance.py` 的 NLSQ + 参数 CI、
可视化电路构建器、单谱/批量两种模式。

**先说最容易踩的一点**：Liu & Ciucci 2025 那篇论文标题就叫 *Adaptive
Distribution of Relaxation Time Analysis Using Bayesian Mixtures*，缩写同样是
ADAPT。看到"ADAPT 用贝叶斯 DRT"时要先分清指的是**论文**还是**这个软件**——
软件里的 BM-DRT 档就是那篇论文的 HN 混合实现。

## 2. 重叠：我们的 BM-DRT 与它的第三档同源

`bm_core.make_model()` 是串联 K 个 HN、stick-breaking 权重、`R∞`、ζ 自由
（0.3–2）、φ∈(0.5,1)、NUTS 采样；它第三档是 HN 混合 + 可选 Warburg +
NUTS（numpyro，默认单链、无收敛诊断）。
**骨架是同一个东西，都源自 Liu & Ciucci 2025。** 所以：

> "我们用了贝叶斯 DRT" 不构成差异化。这不是谦虚，是事实。

差异不只是措辞层面的实现细节，逐项源码对照见
`round_2026-11-12/ADAPT_BASELINE.md` §2。最要紧的三条：
我们 ζ 自由且带下界（并记录过 ζ→0 会冒充 `R∞`、`R∞` 从 10 掉到 4.0 的
实测）、扩散/吸附是**串联**项而它的 Warburg 是带 `p_W_fraction` 的**并联
混合**、以及**它的 BM-DRT 模型族里没有电感项**（HN 与 `1/(iωτ)^α` 都是负
虚部，`R∞` 是实数，无法产生 `Im Z > 0`）。第三条直接落到我们 18/18 低频
感抗的 AZ31 谱上。

## 3. 我们有、它公开信息里没有的

按"能不能被检验"排序，不按好听程度：

1. **拒答**（已按源码核实，见 `ADAPT_BASELINE.md` §3）。`bm_core.fit` 不收敛
   只报 `not_converged`、**不报数**；`bm_real.py:2730` 的 gate 模式在
   `kk.status=fail` 时**无条件拒绝定 K、本次不做拟合**；`R̂>1.05 / div>0`
   触发分级降级。它那边：`run_bm_drt_simple.py:231-241` 跑完 NUTS 直接
   `get_samples()`，无 R̂/ESS/divergence 检查，返回的 result 里没有
   status/warning/quality 字段；GUI `:9392` 写死 `"ok": True`。
   README 的"残差显式绘制、λ 选择可见、给不确定性带"是**透明化**，不是拒答。
   但也要说清楚：**没有证据说它藏失败**，它把该画的都画了；GUI 里的
   "fit converged" 指 ECM 最小二乘，不是 MCMC 收敛，别混为一谈。
   论文标题的 `CI-admissibility` 在这个发行包里只出现在引文中，代码无对应判据。
2. **预注册验收线 + 公开失败**。§13 留出 0/3、§14 全量 4/14，两次都没过，
   写进文档不删（见 `M6_VALIDATION_PLAN.md`）。这条的价值不在"我们更严谨"，
   而在于**结论被自己的线卡住过**，所以现在报的每个数都知道边界在哪。
3. **跨族拓扑选择**。`nested_fit.py` 做嵌套模型比较（M4），不是"看 DRT 峰
   选电路再 NLSQ"。但必须写明：**M4 未过线**（§9.4.21 A1–A4），不进生产。
   它的 DRT 引导选拓扑是**人看峰 + 自动种子**，属于另一条路，不构成我们领先。
4. **可辨识性结论**。`Rs` 不可单报（`Rs–R1` ρ(log)≈−0.94）、`C1≡C4` 严格恒等
   （R19，61 点 2.7e-16）、`comp_unresolved` guard（τ 跨 >2 dec 或 w/W<5%）。
   这些是"我们知道自己测不出什么"，它没有对应输出——但也没有证据说它需要。
5. **数据基数审计**。156 唯一 sha256（171 谱）、Blank 6 份同 sha256、
   假阳性率估不出 → 不训分类器。

## 4. 它有、我们没有的

- GUI、可视化电路构建器、会话存取 —— 易用性上不是一个量级，**不要比**。
- 三档 DRT 并列，能当场比较"正则 vs 贝叶斯 vs 混合"的差异。
- 跨谱 persistence map（老化/工况序列里哪个峰稳定）。
- λ 扫描可视化（不是静默选一个 λ）。
- MIT 开源，可改可分发。

## 5. 判断

分三种目标，结论不同：

| 目标 | 结论 |
|---|---|
| 做一个更好用的 EIS 分析工具 | **不做。** 用 ADAPT-ECM。它的功能面是超集，重做没有意义。 |
| 发一篇"新 DRT 算法"的方法学论文 | **不做。** 路已经被 Ciucci 组占满（2014 λ、2015 RBF/HMC、2017 分层、2020 Lin-KK、2025 混合）。 |
| 在腐蚀/涂层体系上给**可信判据** | **还有位置**，但产出必须是"什么时候 DRT/ECM 的结论不可信"，不是又一个反演。 |

第三条的位置具体在哪：我们的谱型（AZ31 低频感抗尾、2E 高频感抗、3E 端部专用
元件）不在它的示例集里（PEMFC / 电池 / 超容 / SOFC）。而这些谱型恰好让
DRT 的容性半圆假设、Z-HIT 的一阶近似、`Ws` 的 45° 判据**同时失效**——
见 `WS_CRITERION_PLAN.md`（18/18 是感抗）、`round_2026-11-12/ZHIT_PROTOTYPE.md`
（22/22 fail 但原因是谱型）。**"方法在这些谱型上哪里会骗人"是我们真正独占的
东西**，不是算法。

## 6. 已执行（2026-11-12 晚）

**前置障碍不存在**：`run_bm_drt_simple.run_bm_drt_programmatic(freq, Z)`
是纯函数，可脱离 GUI 批调。已用 GUI 默认参数在 7 条我们的谱上跑完，
逐谱数字、源码核行与频率重排陷阱见
`round_2026-11-12/ADAPT_BASELINE.md`（脚本 `adapt_baseline.py`）。

原列三问的实测答案：

1. **K / 峰位**：不可直接比（它固定 `k_hn=5`，无 K 选择；我们 `refused`
   时不报 K）。但两条 2E 谱的 DRT 主峰顶到 τ 网格上界 15.89 s，
   与我们的 `comp_unresolved`「窗口外不可辨识」定性一致。
2. **区间宽度**：两边都是逐样本算 Z 再逐点取分位，但 σ 口径不同
   （它的 `HalfNormal(0.08)` 相对标量 vs 我们的绝对/逐点 σ）。
   **R17 仍成立：区间宽度不能当跨族证据。**
3. **"它给区间 / 我们给降级"是不是同一批**：**不是一一对应**。
   7/7 它都返回 `ok=True` 与完整区间、`result` 无任何质量字段；
   我们 `refused` 的 2 条它照样给数。但 `AZ3` 我们因 R̂/ESS 降级时它
   acf 0.427，`3E` 我们 `reported` 时它 acf 0.786。**共同信号是残差 acf
   排序，不是判定标签**——两边门禁测的本就不是同一件事。

**意外收获（比上面三问更有价值）**：四条有我方产物的谱，两边 acf
**Pearson r = 0.9971**、ADAPT 一律高 0.027–0.082、`0.50` 分界无交叉。
换独立实现 / 不同先验 / 不同 σ 口径 / 多一个 Warburg 混合，残差结构排序
不变 → 比 §17 四电路反事实更外部地**支持**（不是证明）§18/§19 的归因：
**卡 acf 的是数据，不是我们挑的 ECM**。限制是 ADAPT 在坏谱上 Rwp 29–82%，
"数据整谱偏差"与"外部拟合失败"本轮分不开。

**净结论**：第 5 节三种目标的判断一字不改，但"第三条路还有位置"现在
不再只是定位判断，而有了一条可引用的外部基线证据；同时它也说清了
我们的门禁究竟抓的是什么——**不是更准的 DRT，而是别人根本不报的那部分
信息（模型缺项、采样退化、峰跑出窗口）**。
