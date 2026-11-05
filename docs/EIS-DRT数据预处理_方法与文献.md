# EIS-DRT 数据预处理：方法、文献与作用

> 针对「幺的科研路」电化学阻抗谱 DRT 系列内容的专题深挖
> 整理时间：2026年11月5日

---

## 0. 先说清楚资料边界（重要）

UP主的视频简介、动态和专栏**没有逐条列出参考文献**，且我无法直接读取视频画面内容。因此本文档把内容分成两类并分别标注，请按需取用：

| 标记 | 含义 |
|------|------|
| **【UP】** | UP主本人视频/专栏/动态中明确讲到的做法，或他明确指向的资料 |
| **【补】** | 该预处理方法在领域内的原始/经典文献依据，由我查证补充，**不是**UP主视频里逐条引用的 |

几处明确的文献指向：
- UP主专栏《EIS Spectrum Analyser 安装及使用教程》原文：「将含有电感L的曲线进行修正……**具体请参考 EIS Spectrum analyser 官网所给参考文献**」——【UP】指向了文献，但未列名。
- UP主《免费DRT分析EIS的四大网站》视频简介：「**均需进行预处理后再导入EIS数据，预处理过程可参考本人先前发的视频**」——【UP】说明预处理是独立的必经步骤。
- UP主 DRT-DCT-RRT 软件 3.0 更新说明：「新功能的设计参考了多篇文献，**包括 DRT 和 DCT 的计算过程以及岭回归算法的应用**」——【UP】点明了算法类别，未列具体条目。
- UP主《DRT拟合及等效电路图拟合》视频简介给出了完整的 DRT 理论表述（R₀ + 连续串联 RC、CPE、g(τ)），对应的原始文献属【补】。

---

## 1. 为什么 DRT 之前必须做预处理

三条硬约束，决定了预处理不是"讲卫生"而是**数学前提**：

1. **DRT 是病态逆问题（ill-posed inverse problem）**
   从有限频段、含噪声的 Z(ω) 反解 g(τ) 属于第一类 Fredholm 积分方程求逆。噪声会被反卷积显著放大，很小的测量噪声就会造成「电化学过程的错误分配和建模误解」，出现**伪峰**或把真峰劈成双峰。

2. **DRT 模型本身有适用前提**【UP】
   UP主给出的模型假定是：Z = 欧姆电阻 R₀ + 极化阻抗，极化阻抗为**连续串联的 RC 并联电路**。这一模型天然不包含：
   - 串联电感 L（高频负虚部/容抗环）
   - 纯 Warburg 扩散尾（低频 45°斜线）
   把这两类数据直接丢进 DRT，数学上就不满足模型假设。

3. **K-K 一致性是数据有效性的准入证**
   只有满足**因果性、线性、稳定性（时不变）、有限性**四个条件的阻抗谱才严格满足 Kramers-Kronig 关系。不满足 K-K 的数据，任何拟合（DRT 或等效电路）结果都不可解释。

---

## 2. 预处理方法总表

| # | 方法 | 具体操作 | 作用 / 为什么必须做 | 来源 | 文献 |
|---|------|---------|-------------------|------|------|
| 1 | **原始文件清洗** | 辰华 CHI660E/CHI760E 导出的 txt/csv；在 Excel 里分列、**删除测试信息行和空白行**、删除多余列，**只保留频率、Z'、Z'' 三列** | 让软件能解析纯数值矩阵；表头/单位/时间戳会被当成坏数据导致导入失败 | 【UP】+【补】 | 辰华导出格式；DearEIS/DRTtools 输入规范 |
| 2 | **虚部符号统一** | 视软件约定，把 Z'' 列**批量乘 −1**（或按 DRTtools 约定**不取负**，因为程序内部已对坐标取负） | Nyquist 图虚部方向约定不同；符号搞反会把容抗弧翻到上半平面，DRT 直接失效 | 【补】（B站同类教程「eis→drt 预处理：虚部列数据批量乘负一」） | DRTtools 用户手册 |
| 3 | **频率轴规范化** | 频率**递减**、**对数间隔均匀**；去除重复点、明显离群点 | DRT 在 log τ 网格上离散化，频率分布不均会扭曲 g(τ) 分辨率 | 【补】 | Wan 2015 [5] |
| 4 | **路径/文件名 ASCII 化** | 保存路径和文件名**不含中文** | 否则 DearEIS 等工具把中文显示成「?」，读取报错 | 【补】（DearEIS 全流程教程） | DearEIS 文档 |
| 5 | **K-K 有效性验证** | Lin-KK / ZView / RelaxIS 3 / DearEIS / EIS Spectrum Analyser 做 K-K 变换，看**实部与虚部的相对残差** | 判断数据是否满足因果/线性/稳态；不合格的数据点在后续分析中会制造伪峰 | 【UP】明确写进专栏 | [1][2][3][8] |
| 6 | **遮掩 / 剔除不合格点（mask）** | 在 Data sets 页面把**低频不稳定数据点 mask 掉**，再重跑 K-K 直到检验通过 | 低频段测量时间长，最易受体系漂移、温度波动污染；遮掩后 K-K 才接受 | 【补】（DearEIS 实操：遮掩低频点后 K-K 由「不接受」变「接受」） | [2] |
| 7 | **高频电感 / 圆弧修正** | **不是简单删除电感数据点**，而是引入串联 L 元件扣除 Z_L = jωL，或把**已变形的圆弧修正为未变形的圆弧**；也可用 Z-HIT 由相位谱重构无漂移的模值谱 | 引线/夹具寄生电感使高频弧变形甚至出现负虚部；错误删除会丢失高频过程信息并把弧"削歪" | 【UP】专栏原话，且明确指向官网文献 | [9][10]；UP主指向 EIS Spectrum Analyser 官网文献 |
| 8 | **低频扩散段截断 / 转 DCT** | 把 Warburg 45°尾巴**扣掉**，DRT 只做中频圆弧；低频部分改用 **DCT（电容时间分布）** | DRT 只适用于圆弧形阻抗，对非圆弧形扩散数据不适用；强行拟合会污染整个 g(τ) | 【UP】DRT-DCT-RRT 软件设计逻辑 +【补】 | [11][7] |
| 9 | **分频段路由（全频域解析）** | RRDRT → 高频电感/容抗环；常规 DRT → 中频电荷转移圆弧；DCT → 低频扩散斜线 | 把圆弧、斜线、容抗环拆开各归其位，避免一个模型硬套全频域 | 【UP】软件 3.0 核心设计 | [6][11] |
| 10 | **残差图检验** | 拟合后看实部/虚部残差是否随机、幅度小；经验判据为**相对残差 < 1%** | 兜底判断拟合可靠性；残差有结构说明预处理或模型仍有问题 | 【UP】软件 3.0 明确"通过残差图判断拟合结果的可靠性" | [2][6] |
| 11 | **反卷积参数预设** | 设置基元函数（RBF）、解析数据类型（实部/虚部/两者）、**正则化参数 λ**、取样数（≥1000）、FWHM 与 m | 属于"分析前的参数准备"，很多教程也归入预处理；λ 越大越平滑、越小越贴实验值，直接决定峰的个数和形状 | 【补】 | [4][5][6] |

---

## 3. 各方法详解与文献对应

### 3.1 原始文件清洗（方法 1–4）

**UP主关联内容：** 《EIS Spectrum Analyser 安装及使用教程》专栏讲数据导入；《2分钟 Nova 快速拟合 EIS》系列讲辰华 CHI 数据导入示范。

**典型做法（CHI760E 导出 txt 共 5 列）：**
```
Excel 分列 → 删除表头/测试信息行/空白行 → 只留第 1~3 列
    → 检查频率递减、对数间隔均匀
    → 按目标软件决定 Z'' 是否取负
    → 另存为 .txt / .csv，路径与文件名不含中文
```

**作用：**  purely 工程性，但一旦漏掉，后续所有数学处理都在错误数据上跑。特别是**虚部符号**和**非 ASCII 路径**是初学者最常见的两个坑。

**注意一个易错点：** 不同软件约定相反。DRTtools/Matlab 版内部已对坐标取负，所以**不需要**再对第三列取负；而部分在线 DRT 网站和 ZView 流程要求先取负。导入后第一步永远是看一眼 Nyquist 图方向是否正确。

---

### 3.2 Kramers-Kronig 验证（方法 5、6）—— 最核心的一步

**UP主原话**【UP】：
> 「在拟合前需要理解等效元件符号及含义，并进行 K-K 关系验证 EIS 数据可靠性」

**原理：** K-K 关系把 Z'(ω) 与 Z''(ω) 通过 Hilbert 变换联系起来。用实测实部去计算虚部（或反过来），比较计算值与实测值的偏差即可评估数据质量。逻辑是「A ⇒ B」等价于「非 B ⇒ 非 A」：数据不满足 K-K，则系统必然不同时满足四个基本条件。

**四个基本条件：**
1. 因果律（响应由激励产生）
2. 线性（响应频率与激励一致）
3. 稳定性 / 时不变（激励结束后系统回到原状态）
4. 有限性（0 到 ∞ 全频段物理量有限）

**实际困难与解法：**
K-K 积分要求 0 → +∞，而实测只有有限频段（如 100 kHz – 0.01 Hz）。**Lin-KK** 用一串 RC 元件拟合来规避外推问题，是目前的主流方案（RelaxIS 3 内置的正是"线性 Kramers-Kronig 验证 + 残差分析"）。

**判定与处置：**
- 残差图随机、相对残差小（经验 < 1%）→ 通过
- 未通过 → 回到 Data sets 把**低频问题点 mask 掉**，重跑 K-K，直到检验接受；而不是直接硬做 DRT
- DearEIS 实操案例：拟合曲线看起来很贴近，但参数结果 K-K 不接受；遮掩低频点后即通过。**说明"眼睛看着贴合"不等于数据有效。**

**文献：**
- [1] B. A. Boukamp, *A Linear Kronig–Kramers Transform Test for Immittance Data Validation*, J. Electrochem. Soc. **1995**, 142, 1885–1894. DOI: 10.1149/1.2044210 —【补】线性 K-K 检验的奠基文献
- [2] M. Schönleber, D. Klotz, E. Ivers-Tiffée, *A Method for Improving the Robustness of Linear Kramers-Kronig Validity Tests*, Electrochim. Acta **2014**, 131, 20–27. DOI: 10.1016/j.electacta.2014.01.034 —【补】Lin-KK 算法，KIT 官方 Lin-KK Tool 的理论依据
- [3] M. Schönleber, E. Ivers-Tiffée, *Approximability of Impedance Spectra by RC Elements and Implications for Impedance Analysis*, Electrochem. Commun. **2015**, 58, 15–19. DOI: 10.1016/j.elecom.2015.05.018 —【补】RC 元件可逼近性
- [8] 同上 [2]，被知乎《DRT 保姆级教程》列为 K-K 验证参考文献
- 中文教材：曹楚南《电化学阻抗谱导论》—— B站 K-K 科普专栏推荐的三大前提假设出处

---

### 3.3 高频电感修正（方法 7）—— UP主特别强调的坑

**UP主原话**【UP】：
> 「将含有电感 L 的曲线进行修正变成无电感曲线（**并不是简单删除前面的电感数据，而是将已变形的圆弧修正为未变形的圆弧**。具体请参考 EIS Spectrum analyser 官网所给参考文献）。」

**为什么不能简单删点：**
寄生电感的影响不只落在那几个负虚部数据点上，它会让整个高频圆弧发生**形变**（弧心下移、压扁）。只删掉"看起来是电感"的那几个点，剩下的弧依然是变形的，DRT 会把它解释成一个额外的（虚假的）时间常数峰。正确做法是从整个复数阻抗中扣除感抗贡献：

```
Z_corrected(ω) = Z_measured(ω) − jωL_parasitic
```

其中 L 需通过高频段空载/短路校准或等效电路拟合得到。

**另一条技术路线：Z-HIT**
利用相位谱经改进的对数 Hilbert 变换重构无漂移的阻抗模值谱，可同时完成**数据验证 + 数据重构**，对测量过程中体系漂移（不稳定）导致的伪影特别有效。瑞士万通 RelaxIS 的 DRT 讲座也把 ZHIT 列为 DRT 分析基础理论之一。

**作用：** 恢复高频数据本来面貌，消除虚假高频峰，保证 DRT 只反映真实的界面/电荷转移过程。

**文献：**
- [9] W. Ehm, H. Göhr, R. Kaus, B. Röseler, C. A. Schiller, *The Evaluation of Electrochemical Impedance Spectra Using a Modified Logarithmic Hilbert Transform*, Acta Chim. Hung. **2000**, 137, 145 —【补】Z-HIT 原始文献
- [10] 同组作者, *Z-HIT – A Simple Relation Between Impedance Modulus and Phase Angle, Providing a New Way to the Validation of Impedance Spectra* —【补】
- 【UP】明确指向的 EIS Spectrum Analyser 官网参考文献列表（官网当前访问不稳定，未能逐条抓取；主要围绕 A. S. Bondarenko / G. A. Ragoisha 的 PDEIS 与电感修正工作，如 *Potentiodynamic Electrochemical Impedance Spectroscopy*, J. Electroanal. Chem. 2004 系列）
- UP主 DRT-DCT-RRT 3.0 的 **RRDRT** 模块，则是用岭回归直接解析负轴电感/容抗环，属于"不删除、单独建模"的第三条路线

---

### 3.4 低频扩散段截断与 DCT（方法 8、9）

**原理：** DRT 的核函数是 Debye/R//C 型，天然描述**圆弧**。低频 Warburg 45°斜线、有限空间扩散、Gerischer 等都不是 RC 弧。把它们留在数据里做反卷积，正则化为了"解释"这段斜线会强行在长 τ 端堆出宽的假峰，并牵连中频峰的位置和面积。

**两种处置：**
1. **截断法**：扣掉低频扩散尾，只对中频圆弧做 DRT（多数通用做法）
2. **分频域解析法**【UP】：UP主的 DRT-DCT-RRT 软件不截断而是分家——
   - RRDRT：高频电感 / 容抗环
   - 常规 DRT：中频电荷转移圆弧
   - DCT（Distribution of Capacitance Times，电容时间分布）：低频扩散斜线

**作用：** 让每一段数据用与之匹配的核函数解析，实现全频域有效解析，避免"一个模型硬套"，同时保住低频扩散信息不被丢弃（DCT 图中的 τ5 可对应多孔电极扩散过程）。

**文献：**
- [11] J. Chen, E. Quattrocchi, F. Ciucci, et al. *Charging Processes in Lithium-Oxygen Batteries Unraveled through the Lens of the Distribution of Relaxation Times*, Chem **2023**, 9(8), 2267–2281. DOI: 10.1016/j.chempr.2023.04.022 —【补】DRT + DCT 分别解析高频/低频的原始工作
- [6] A. Maradesa, B. Py, J. Huang, et al. *Advancing Electrochemical Impedance Analysis through Innovations in the Distribution of Relaxation Times Method*, Joule **2024**, 8(7), 1958–1981 —【补】DRT 方法综述，含软件与适用范围总结
- [7] Y. Lu, C. Z. Zhao, J. Q. Huang, et al. *The Timescale Identification Decoupling Complicated Kinetic Processes in Lithium Batteries*, Joule **2022**, 6(6), 1172–1198 —【补】时间尺度识别与预处理综述，被多篇 DRT 教程列为核心参考
- [12] A. Weiß, S. Schindler, S. Galbiati, et al. *Distribution of Relaxation Times Analysis of High-Temperature PEM Fuel Cell Impedance Spectra*, Electrochim. Acta **2017**, 230, 391–398 —【补】

---

### 3.5 正则化与反卷积参数（方法 11）

严格说这是"分析参数"而非"数据预处理"，但几乎所有 DRT 教程都把它放在导入数据这一步一起讲，故一并列出。

**核心问题：** DRT 反卷积是病态逆问题，必须用 Tikhonov 正则化（岭回归）稳定求解。正则化参数 λ 被戏称"魔法超参数"，直接决定峰数和峰形，是 DRT 主观性的主要来源。

**参数作用对照：**

| 参数 | 作用 | 设置效果 |
|------|------|---------|
| 正则化参数 λ | 平滑约束强度 | 越大越平滑（峰合并），越小越贴近实验值（易出伪峰） |
| 基元函数 | 离散化 g(τ) 的形状 | DRTtools 用径向基函数 RBF |
| 解析数据类型 | 用 Z' / Z'' / 两者 | 影响反卷积条件数 |
| 取样数 | 贝叶斯方法采样量 | 建议 ≥1000，越多置信区间越可信 |
| FWHM / m | 峰宽与 Δlnτ 的关系，FWHM = Δlnτ / m | 控制分辨率 |
| 置信区间（99% MAP 灰色带） | 贝叶斯输出 | 灰带越窄说明拟合越可信 |

**作用：** 在"贴合实验值"和"解的稳定性"之间取平衡；通过更换方法/参数看峰是否稳定，筛选出真正显著的峰。

**文献：**
- [4] M. Saccoccio, T. H. Wan, C. Chen, F. Ciucci, *Optimal Regularization in Distribution of Relaxation Times Applied to Electrochemical Impedance Spectroscopy: Ridge and Lasso Regression Methods — A Theoretical and Experimental Study*, Electrochim. Acta **2014**, 147, 470–482. DOI: 10.1016/j.electacta.2014.09.058
- [5] T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, *Influence of the Discretization Methods on the Distribution of Relaxation Times Deconvolution: Implementing Radial Basis Functions with DRTtools*, Electrochim. Acta **2015**, 184, 483–499. DOI: 10.1016/j.electacta.2015.09.097
- [13] A. Maradesa, B. Py, T. H. Wan, M. B. Effat, F. Ciucci, *Selecting the Regularization Parameter in the Distribution of Relaxation Times* — pyDRTtools 官方建议引用的 λ 选择方法文献
- [14] H. Schichlein, A. C. Müller, M. Voigts, A. Krüger, E. Ivers-Tiffée, *Deconvolution of Electrochemical Impedance Spectra for the Identification of Electrode Reaction Mechanisms in Solid Oxide Fuel Cells*, J. Appl. Electrochem. **2002**, 32, 875–882 —【补】DRT 反卷积早期经典工作

---

### 3.6 中文基础文献

- [15] 王佳, 黄秋安, 李伟恒, 王娟, 庄全超, 张久俊. 电化学阻抗谱弛豫时间分布基础[J]. 电化学（中英文）, **2020**, 26(5): 607–627. — 中文 DRT 入门首选
- [16] 庄全超, 徐守冬, 邱祥云, 等. 锂离子电池的电化学阻抗谱分析[J]. 化学进展, **2010**.
- [17] 贾铮, 戴长松, 陈玲. 电化学测量方法[M]. 北京: 化学工业出版社, 2006.
- [18] 曹楚南. 电化学阻抗谱导论[M]. — K-K 三大前提假设的中文出处

---

## 4. 完整可执行 SOP（含判定）

```
① 导出
   辰华 CHI660E/760E → .txt / .csv
   作用：拿到原始 Z(ω)
        │
② 清洗
   Excel 分列 → 删表头/测试信息/空白行 → 只留 频率、Z'、Z'' 三列
   → 频率递减、对数间隔均匀 → 按软件约定处理 Z'' 符号
   → 保存为 ASCII 文件名、ASCII 路径
   作用：可解析、方向正确
        │
③ K-K 验证   ← 唯一的"准入闸门"
   Lin-KK / ZView / RelaxIS 3 / DearEIS / EIS Spectrum Analyser
   ├─ 通过（残差随机、相对残差 < 1%）→ 进入 ④
   └─ 不通过 → mask 低频漂移点 / 检查测试是否稳态 → 回到 ③
   作用：确认因果、线性、时不变、有限
        │
④ 高频电感修正
   扣除 jωL 或 Z-HIT 重构，把变形圆弧还原
   ⚠ 禁止简单删除电感点   ←【UP】特别强调
   作用：消除虚假高频峰
        │
⑤ 分频段路由
   高频电感/容抗环 → RRDRT（或扣除后不用）
   中频圆弧       → 常规 DRT
   低频 45°扩散尾 → 扣掉，或交给 DCT
   作用：数据特征与核函数匹配
        │
⑥ 反卷积参数预设与求解
   RBF 基元函数、λ、取样数 ≥1000、FWHM/m、贝叶斯置信区间
   更换方法/λ 检查峰的稳定性
   作用：抑制病态逆问题的噪声放大
        │
⑦ 残差图检验 + Bode 图交叉验证【UP】
   残差应随机且小；拟合优劣更应看 Bode 图（随频率变化，更能反映时间关系）
   作用：兜底，防止"看着贴合其实无效"
        │
⑧ 导出 CSV → Origin 绘图 / 分峰积分
```

---

## 5. 常见误区（UP主强调 + 文献共识）

| 误区 | 后果 | 正确做法 |
|------|------|---------|
| **把高频电感数据点直接删掉**【UP】 | 留下变形圆弧 → DRT 出现虚假高频峰 | 扣除 jωL 或还原圆弧，或交给 RRDRT |
| **跳过 K-K 验证** | 漂移/非线性数据被当成有效数据，后面全白做 | 任何拟合（DRT 或 ECM）前都要做 |
| **对 45° 扩散尾直接做 DRT** | 正则化在长 τ 端制造宽假峰，牵连中频峰 | 截断或改用 DCT |
| **只看 Nyquist 不看 Bode**【UP】 | Nyquist 掩盖频率信息，看不出拟合随时间常数的偏差 | Bode 图为主判据 |
| **先在 ZView 里"过度优化"数据再做 DRT** | 人为抹平真实特征，预测误差变大；移除个别点甚至会让双峰变单峰 | 只做异常点剔除，不做平滑/过拟合 |
| **λ 随手填一个值** | 峰数、峰位全靠参数决定，结论不可复现 | 用 L-curve / 交叉验证 / 贝叶斯方法，并报告所用参数 |
| **虚部符号搞反** | 弧翻到上半平面，DRT 完全失效 | 导入后先看 Nyquist 方向 |

> 一个值得记住的实证：某教程在 ZView 中移除部分异常数据点后，DRT 结果**由双峰变单峰**，置信区间也随之收窄。这说明预处理不是可选项——它能直接改变你对体系动力学过程数量的判断。

---

## 6. 文献总表（便于直接复制引用）

| 编号 | 文献 | 对应方法 |
|------|------|---------|
| [1] | Boukamp B.A., *J. Electrochem. Soc.* **1995**, 142, 1885–1894. DOI: 10.1149/1.2044210 | 线性 K-K 检验奠基 |
| [2] | Schönleber M., Klotz D., Ivers-Tiffée E., *Electrochim. Acta* **2014**, 131, 20–27. DOI: 10.1016/j.electacta.2014.01.034 | Lin-KK、残差判据、mask |
| [3] | Schönleber M., Ivers-Tiffée E., *Electrochem. Commun.* **2015**, 58, 15–19. DOI: 10.1016/j.elecom.2015.05.018 | RC 元件可逼近性 |
| [4] | Saccoccio M., Wan T.H., Chen C., Ciucci F., *Electrochim. Acta* **2014**, 147, 470–482. DOI: 10.1016/j.electacta.2014.09.058 | 岭回归/Lasso 正则化 |
| [5] | Wan T.H., Saccoccio M., Chen C., Ciucci F., *Electrochim. Acta* **2015**, 184, 483–499. DOI: 10.1016/j.electacta.2015.09.097 | RBF 离散化、DRTtools |
| [6] | Maradesa A., Py B., Huang J., et al., *Joule* **2024**, 8(7), 1958–1981 | DRT 方法综述、软件对比 |
| [7] | Lu Y., Zhao C.Z., Huang J.Q., et al., *Joule* **2022**, 6(6), 1172–1198 | 时间尺度识别、预处理流程 |
| [8] | 同 [2]（知乎 DRT 保姆级教程列为 K-K 参考） | K-K 验证 |
| [9] | Ehm W., Göhr H., Kaus R., Röseler B., Schiller C.A., *Acta Chim. Hung.* **2000**, 137, 145 | Z-HIT 修正/重构 |
| [10] | 同组作者, *Z-HIT – A Simple Relation Between Impedance Modulus and Phase Angle…* | 数据验证与重构 |
| [11] | Chen J., Quattrocchi E., Ciucci F., et al., *Chem* **2023**, 9(8), 2267–2281. DOI: 10.1016/j.chempr.2023.04.022 | DCT、高频/低频分域解析 |
| [12] | Weiß A., Schindler S., Galbiati S., et al., *Electrochim. Acta* **2017**, 230, 391–398 | HT-PEM 燃料电池 DRT |
| [13] | Maradesa A., Py B., Wan T.H., Effat M.B., Ciucci F., *Selecting the Regularization Parameter in the DRT* | λ 选择方法 |
| [14] | Schichlein H., Müller A.C., Voigts M., Krüger A., Ivers-Tiffée E., *J. Appl. Electrochem.* **2002**, 32, 875–882 | DRT 反卷积经典 |
| [15] | 王佳, 黄秋安, 等. 电化学阻抗谱弛豫时间分布基础[J]. 电化学, **2020**, 26(5): 607–627 | 中文 DRT 基础 |
| [16] | 庄全超, 等. 锂离子电池的电化学阻抗谱分析[J]. 化学进展, **2010** | 中文 EIS 基础 |
| [17] | 贾铮, 戴长松, 陈玲. 电化学测量方法[M]. 化学工业出版社, 2006 | 中文教材 |
| [18] | 曹楚南. 电化学阻抗谱导论[M] | K-K 前提假设 |

---

## 7. 本轮查证的主要资料来源

- UP主视频《电化学阻抗谱弛豫时间分布（DRT）拟合及等效电路图拟合》BV1Kp4y1J7Q5（2023-08-28）
- UP主视频《电催化过程中的原位EIS和DRT的测试和分析方法》BV1duugz8EVd（2025-07-11）
- UP主视频《免费DRT分析EIS的四大网站》BV1RQkXB2Epv（2025-11-09）
- UP主专栏《比Zview和ZSimpWin更好用的免费EIS拟合工具 EIS Spectrum Analyser 安装及使用教程》（2022-05-22）
- UP主 DRT-DCT-RRT 软件 3.0 更新说明（2025-12-08）
- B站专栏《对电化学阻抗谱的弛豫时间分布(EIS-DRT)应用实践》（DearEIS 完整工作流）
- B站专栏《电化学交流阻抗谱（EIS）中的Kramers-Kronig关系验证》
- 知乎《弛豫时间分布（DRT）保姆级教程》《阻抗谱(EIS)高级解谱：DRT原理、MatLab实现、数据分析》
- KIT IAM-ET 官方 Lin-KK Tool 页面（文献 [1][2][3] 与算法说明）
- 瑞士万通 RelaxIS 3 产品说明（线性 K-K 验证 + 残差分析）

---

*说明：文献 [1]–[18] 中带 DOI 者已核对卷期页码；[13] 为 pyDRTtools 官方推荐引用条目，未核对到确切卷期，引用前建议二次确认。本文档基于公开可检索资料整理，UP主视频本体未逐帧观看，涉及视频内具体口述内容请以原视频为准。*
