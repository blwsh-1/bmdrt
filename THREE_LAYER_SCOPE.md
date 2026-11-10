# 三层职责与证据资格（第三层定位调整，v1）

> 日期：2026-11-09（**2026-11-10 增补 §2.1**：B 档分「等价 / 缺项」两个子类，
> 来自 M4 跨族选择的同构判定，见 `NESTED_TOPOLOGY_PROJECT.md` §9.4.10）
> 触发：外部只读审查（WP2 v2 复核）+ 项目方判断「现有数据不足以支撑系统性物理机制判定」。
> 关系：不改 `WS_DECISION_PROJECT.md` §3 的能力边界表（那张表是**结论**），本文定**职责与输出口径**。
> 硬约束：本文**不新增算法能力**，只重新分配「谁负责说什么」。

---

## 0. 为什么改定位

两条约束，都不是「做得不够好」，而是「目标定错了」。

### 0.1 数据量不支持统计判据，更不支持分类器

`evidence/m6/data_inventory.csv`（E1 全量清单，只读复算）：

| 事实 | 值 |
|---|---|
| 谱文件总数 | **171** |
| 唯一 `sha256` | **156**（15 条是内容重复） |
| 独立 `group_id` | **11** |
| 　其中：2E 系列 | **7 组 / 127 条**（已冻结，不进本轮结论） |
| 　其中：AZ31 三组 + Blank | **4 组 / 39 条唯一谱** ← 真正参与结论的部分 |
| `independence_status` | `inferred` **171/171** |
| `group_id_confidence` | `filename` **171/171** |

`evidence/m6/inventory.py` 文件头写死：**「E1 不产出 `confirmed`」**——那是人工核对实验记录后才能填的值，脚本无权升级。也就是说，11 个「独立实验单元」目前**全部是文件名推断**，真实独立数只可能更少。

再往下切一层，`Blank` 那 6 条**不是重复实验，是同一条谱复制 6 份**（`sha256` 6/6 相同，E2 已踩过：`Blank_*` 6 文件 sha256 相同）。AZ31 三组内则各自 12/14/12 条全不同。所以非 2E 部分的真实结构是：

    AZ31-4Li 14 条 + AZ31-8Li 12 条 + AZ31-12Li 12 条 + Blank 1 条 = 39 条唯一谱 / 4 个样品名

在这个基数上：任何「物理判据」的假阳性率都估不出来；任何机制分类器都只能过拟合到 **4 个样品**。**这不是样本量偏小，是量级不够。**

### 0.2 判据与机制之间有一条填不上的缝

三个已实测的例子（都在本项目里发生过）：

| 判据 | 只能说明 | 不能说明 | 出处 |
|---|---|---|---|
| `−Im` 斜率 = −0.5 | 该段与半无限扩散**相容** | Ws 存在 | WP2 §3.5.1：弧尾共存时真 Ws 的斜率被拖到 −0.41…−0.49 |
| `A_W > 0`（后验非零） | 该参数被数据定住了 | 扩散过程真实存在 | WP2：K=2 时 `A_W` 4.18→0.78 且 `p_diff_small`=0.53 |
| `f_D` 在窗外 | 本窗看不到转折 | 该过程不存在 | `diffusion_resolution` 的 `indistinguishable` 分支 |

第三条尤其危险：**窗外的过程不是「不存在」，是「本窗无权评价」**。现有代码在这一点上是保守的（判 `indistinguishable`），但输出措辞容易被读成「没有」。

**结论：缝不是靠更强的判据补上的，是靠「不跨越」补上的。**

---

## 1. 三层职责（写死）

| 层 | 算法负责什么 | 输出 | **不得输出** |
|---|---|---|---|
| **L1 拟合与质量控制** | 收敛、残差、KK、参数边界、DRT 一致性 | 这次拟合**是否可信** | 任何模型选择、任何物理含义 |
| **L2 模型比较与可辨识性** | LOO、后验预测、参数相关性、模型间预测差异 | **哪些模型得到支持**、**哪些无法区分** | 「最优模型 = 真实机制」 |
| **L3 物理解释辅助** | 检查物理约束、展示参数特征、**标记证据缺口** | 物理**相容性**与**解释限制** | 任何**裁决**：不加分、不排序、不单选、不宣称机制 |

L3 的关键改动：**从「裁决器」变成「记账器」**。它输出的是「我有哪些证据、每条证据能撑到哪一步、哪一步没撑住」，而不是「答案是 Ws」。

这与 `WS_DECISION_PROJECT.md` §1 已钉死的口径一致：`Ws 判断` **仅指**「是否加 ω^(-1/2) 项」的模型选择，不指识别物理机制。本文把这句话从「Ws 专项」推广到所有机制判断。

---

## 2. 证据资格三分法（A / B / C）

现有代码把所有「给不出答案」压成一个 `uncertain`。这是错的：三种情形要**三种处理**，混在一起会让使用者误以为「算法不行」或「数据不行」是同一件事。

| 档 | 名称 | 含义 | 归属 | 正确处理 |
|---|---|---|---|---|
| **A** | **算法无法判断** | 采样不稳定、模型实现或数值方法有问题 | 工程缺陷 | **修代码 / 重跑 / 提高预算**；不得当作数据结论 |
| **B** | **数据无法判断** | 算法正常，但不同机制在本窗+本噪声下近似等价预测 | 数据能力边界 | **并列输出、保留多候选**；不得单选，也不得说「不存在」 |
| **C** | **物理证据不足** | 数据能区分**数学模型**，但不足以确定对应的**真实物理过程** | 方法学边界 | **只报相容性**；不得升级为机制结论 |

判据归属规则（按顺序判，先命中先归）：

1. 采样门禁未过 / LOO 不可用 / 拟合未执行 → **A**
2. 采样通过、配对差在预注册线内、或形状差 < 容差 → **B**（内部再分「等价 / 缺项」两个子类，见 §2.1；档位不变，只改输出形态与处理动作）
3. 模型比较有结论（Δ 过线），但缺少「机制唯一性」证据（真值、独立实验、正交表征） → **C**

> 注意 C 档的性质：**它不是失败，是正常且最常见的终态**。真实谱上几乎所有机制结论都会停在 C。项目此前的问题是把 C 写成 B（「不可区分」）或直接升到结论（「检测到扩散」）。

### 2.1 B 档的两个子类：**等价** vs **缺项**（2026-11-10 增补）

B 档原本只有一个含义：「候选集完整，是数据分不出」。M4 跨族选择暴露出第二种情形，
**必须分开**——因为两者的处理动作相反：

| 子类 | 含义 | 判据 | 处理 |
|---|---|---|---|
| **B-等价** | 候选集**完整**，是数据分不出 | 配对差 < 阈值，且各族结构上都能表示该谱 | 加采样 / 换频窗**可能**改善 |
| **B-缺项** | 候选集**缺项**，某族结构上表示不了这类谱 | 谱里存在该族**结构边界**之外的特征（如正虚部感抗落在 voigt 的实极点族之外） | **加采样无效**；只能**换族定义或换候选集** |

区别不在措辞。B-等价 记「并列」是**信息充分**下的合规输出；B-缺项 记「并列」会让人
以为两族都试过了，实际上有一族**从未进入比较**。

**已实现**（`bm_select.py`，2026-11-10）：M4 跨族选择器把 B-缺项输出为
`voigt_inadmissible` 而不是 `indistinguishable`。判据**分频段**——串联 `jωL` 只覆盖
高频感抗，低频感抗要 `R∥L`，两者不可互换（依据：`docs/REFS_DIGEST.md` §2.I ⑥，
AZ31 实测低频感抗比串联 `ωL` 的 2σ 上界大 6.9–8.5 个数量级）。

> **`note` 的措辞已改（2026-11-10 18:09）**：原来写「需 `adsorption=RL`」，**这是错的**。
> B 臂实测（`NESTED_TOPOLOGY_PROJECT.md` §9.4.12）证明：把 `R∥L` 加在 voigt 的
> **最外层串联**（`bm_core.model_z` 的实现），`noise=0` 对 nested 谱的残差只从
> 44.0% 降到 40.7%，Δ 完全不动，降级反而 8/20→10/20。真正缺的是**接点**——
> nested 的 `R∥L` 在第二个并联节点内（`C2 ∥ (R2+Z_ad)`），voigt 没有那个节点。
> 现在的措辞是「需把 `R∥L` 放进第二并联节点（= 改族定义；顶层 `adsorption` 不足以表示）」。

**典型例子**：30 条真实谱（AZ31 18 + `archive/` 12，后者 2 对重复、独立 10）在**基线配置**下
**全部**落在 B-缺项（正虚部感抗 + voigt 实极点族）。

**B-缺项内部还要再分两档**（2026-11-10 18:24 增补，来自 `bm_select.py --real` 的实测）——
因为处理成本差一个量级，混在一起说会让人去开一个对某批谱根本没用的开关：

| 细档 | 例子 | 处理 | 成本 |
|---|---|---|---|
| **缺项-开关可补** | `archive/` 的 2EMC 6 条（感抗在 8e3–1e5 Hz） | 打开已有的 `include_L` | 一个开关，**不动族定义**；已实测值得（Δ +37…+139，`NESTED_TOPOLOGY_PROJECT.md` §9.4.15） |
| **缺项-需改族** | AZ31 18 条（感抗在 0.01–1.4 Hz） | 换族定义或排除该族 | 重跑 + 重新预注册 |

分界就是感抗落在哪一段：串联 `jωL` 只覆盖高频，低频要 `R∥L`。实测加 `include_L`
后 30 条里 **8 条**（独立 7 条）进候选空间，**全是 2EMC 的高频段谱，AZ31 一条都没有**。
详见 `NESTED_TOPOLOGY_PROJECT.md` §9.4.13。

> 注意「可表示」只表示**进了候选空间**，不表示拟合得好。2EMC 那 8 条要断言
> 「开 `include_L` 更好」，仍须按 §9.4.11 的格式预注册一次能力臂。
这不是「都试过了、只是分不开」，而是**有一族从未进入比较**。

**一条硬边界**：

> **排除一族 ≠ 证明另一族。** B-缺项只讲候选集缺了谁，不讲剩下那族是不是正确机制。
> 「AZ31 的低频感抗来自 MgOH_ads」仍是 **C 档**，与「nested 能否表示」是两个独立问题。

**它不改三分法的档位**：仍归 **B**（候选集的能力边界）——不是 A（代码没坏、重跑无用），
也不是 C（Δ 未过线，谈不上「模型能区分但机制定不了」）。三档的划分与判据归属规则的
**顺序不变**，只是在 B 内部多了一个可指认的子标签。

---

## 3. 现状映射（代码事实，2026-11-09 核对；§3.4 为拆分后的口径）

> 本节 §3.1–§3.3 记录**拆分前**的样子（保留是为了能复核「原来哪里混了」）。
> 拆分后的实际字段见 **§3.4**，那是现在 JSON 里的东西。

### 3.1 已有状态字面量 → 档位

| 位置 | 状态 | 档 |
|---|---|---|
| `select_k` | `converged` | —（L1 通过） |
| | `degraded` / `rejected` / `exploratory` | **A** |
| `diffusion_gate` | `no_candidate` / `inductive_tail` | **B**（形状不符，数据无权支持） |
| | `degenerate` | **B**（HN 已退化，与 Ws 同形） |
| `select_mechanism` | `switched` / `ws_worse` | L2 结论 |
| | `uncertain` | **B** |
| | `degenerate` | **B**（HN 已退化，与 Ws 同形） |
| | `fit_missing` / `comparison_incomplete` | **A**（未拟合 / 无配对 LOO） |
| | `guard_degraded` | **A 与 B 混用（见 3.2）** |
| `diffusion_resolution` | `resolvable` | L2 可判 |
| | `indistinguishable` | **B** |
| | `unknown` | **A**（拿不到 `tauD` 后验） |
| | `no_diffusion` | —（未启用） |
| `bm_kk` | `ok` / `warn` / `fail` / `unavailable` | L1 |
| `component_identifiability` | `in_window` | L2 |
| | `uncertain` | **A 或 B 混用** |
| | — | **C 档：完全缺失** |

### 3.2 三处实际混用（可指认）

**(1) `select_mechanism` 的 8 个 guard 全部汇入一个 `guard_degraded`**

`select_mechanism()`（`_select_mechanism_impl()` 里的 `guards` dict）分属两档：

- **A 类**（算法）：`ws_grade0`、`ws_bad_k`、`ws_loo_unavailable`
- **B 类**（数据）：`diff_amp_unidentifiable`、`hn_degenerate`、`ws_amp_unidentified`

但该函数只输出 `status="guard_degraded"` 加一条拼起来的 `reason`。消费端无法区分「采样崩了」与「数据定不住幅值」——前者该重跑，后者重跑一万次也一样。

**(2) `diffusion_resolution` 的 `unknown` 与 `indistinguishable` 都读作「不可分」**

`diffusion_resolution()` 的 `unknown` 是「模型里没有 `tauD` 后验」（A：配置缺失），`indistinguishable`（两个分支：形状差 ≤ tol / 拐点在窗下）是「数据边界」（B）。两者 `note` 都写「不可分」，只有细读才分得开。

**(3) `p_peak_out_max` 高 → 「分量峰在窗外」被读成「没有这个分量」**

峰位出窗（`P(out)≈1`）是 **C**：数据能定参数、能算预测，但**本窗没有覆盖该特征频率**，所以无权说该过程存在与否。现有措辞已经收窄（「不要把该分量解释成已确定的独立物理弛豫过程」），但没有字段表达「这是 C 而不是 B」。

### 3.3 C 档缺失的后果

现有 JSON 里最接近 C 的表达是 `diffusion_resolution.note` 的自然语言段落。也就是说：**C 档目前只存在于文字里，不存在于结构里**。任何下游程序（批量表、报告生成、软件界面）都无法可靠地把「物理相容」与「物理已确认」分开——因为它俩现在都是同一个空字段。

### 3.4 拆分后（v2，已落地 2026-11-09）

新增 `bm_grade.py`（表驱动，`python bm_grade.py` 自检 56/56），只做映射：
**不读数据、不改任何 `status` / `reason` / `guards` / `choice` / `delta`**。

| 输出位置 | 新增字段 | 取值 |
|---|---|---|
| `auto_mechanism` / `ad_selection` | `grade` | `"A"` / `"B"` / `"C"` / `null` |
| | `grade_reason` | 一句话：由哪个 `status` 或哪些 guard 判的 + 该档该怎么处理 |
| | `guard_classes` | **只列触发的** guard → 档（`{}` = 没触发） |
| | `not` | 写死「这条推不出什么」 |
| `diffusion_resolution` | `grade` / `grade_reason` / `not` | `unknown`→A、`indistinguishable`→B、`resolvable`/`no_diffusion`→`null` |
| `component_review` | `component_grade` | 与 `component_identifiability` 一一对应：`in_window`→`null`、`out_of_window`→**C**、`uncertain` 分两种（后验缺失→A / 0.05<p≤0.5→B） |

映射表在 `bm_grade.py` 顶部三张表：`GUARD_CLASS`（11 条）、`STATUS_CLASS`（13 条）、`LEVEL_CLASS`（4 条）。未登记的 guard/status **一律按 A**（宁可说「算法不确定」，不要说「数据不确定」），并在 `grade_reason` 里点名，避免静默归类。

`grade = null` 有明确含义：**该结论不落在三档内**，不构成资格限制（例如 `resolvable` = 「本频段能分 Ws/Wo」，那是能力陈述，不是机制结论；机制结论仍归 C）。

**回归（`evidence/ws_crit/grade_check.py`）**

1. 合成 30 例覆盖全部分支（6 种 status × 各 guard 单独/混合 × 4 个 `level` × 4 个组分状态）。**改前快照入库**为 `evidence/ws_crit/GRADE_SNAPSHOT.json`，任何人在改后跑
   `python evidence/ws_crit/grade_check.py --verify evidence/ws_crit/GRADE_SNAPSHOT.json`
   即可复核：**旧字段 0 处变化，只多出 114 个新键**（`grade`×30、`grade_reason`×30、`not`×30、`guard_classes`×24）。逐例档位表见 `GRADE_TABLE.txt`，对拍记录见 `GRADE_DIFF.txt`。
2. 端到端三对（`git worktree` 取改动前代码，同一命令、同一 seed、同一 `--ta` 各跑一次）：

| 谱 / 配置 | 旧代码 | 新代码 | JSON 差异 |
|---|---|---|---|
| AZ31-4Li_100 `--auto-mechanism` | `skipped_inductive` | 同，档 **B** | 仅 `t_fit_s`（计时） |
| Blank `--auto-mechanism-full` | `switched`（Δ=109.89） | 同，档 **C** | 仅 `t_fit_s` |
| AZ31-4Li_100 `--auto-mechanism-full` | `guard_degraded`（`diff_amp_unidentifiable`+`ws_amp_unidentified`） | 同，档 **B** | 仅 `t_fit_s` |

即：**三个真实路径的判决、`elpd`、分量、残差全部逐位不变**，只有计时字段和 8 个新键。

**`not` 的两级分工（2026-11-09 定）**：生产 JSON 里的 `not` 仍是**每条结论一个**（选择器级，跟着 `grade` 走）；**每条证据一个 `not`** 由 §4.2 的证据卡生成器（`bm_evidence.py`）在输出时合成，**不写回**生产 JSON——否则每份结果要再翻一倍字段，且会诱导下游把逐条 `not` 当独立判据读。

---

## 4. 证据卡规格（L3 的唯一输出形态）

**证据卡不排序、不打分、不建议**。它是一张「每条证据 + 它的资格档」的清单。

### 4.1 字段表（以 Ws 为例）

| 检查项 | 字段 | 现有来源 | 可以输出 | **不可以输出** |
|---|---|---|---|---|
| 幅值后验 | `A_W`, `A_W_sd`, `cv_aw` | `bm_ws.ws_share_low` | 中位数、CI、相对不确定度 | 「扩散已被证实」 |
| 低频斜率 | `slope_im`, `r2`, OLS/块 bootstrap CI | `bm_ws.slope_im`；CI 需另算 | 频段斜率 + 区间 + `usable` | 「斜率≠−0.5 ⇒ 无 Ws」 |
| 频率窗口 | `f_min`, `f_max`, `f_D`, `below_fD` | `bm_diff.diffusion_resolution` | 特征频率是否在窗内、差几个 decade | 「窗外 ⇒ 不存在」 |
| 参数相关性 | `param_corr` | `bm_real` 拟合输出 | 相关矩阵、|ρ|>0.9 的成对标记 | 「两机制已被区分」 |
| 传播/退化指纹 | `degeneracy`, `n_warburg_like` | `bm_real` | 与 HN 的同形程度 | 「因为同形所以是 Ws」 |
| 跨模型预测重叠 | 需多模型联合 | **缺** | 两两 `max|ΔZ|/σ`、包络内/外 | 「重叠 ⇒ 机制等价」 |
| **资格档** | `grade ∈ {A,B,C}` | **缺** | 见 §2 | 综合评分、置信度总分 |

### 4.2 接口形状（v1，已实现 2026-11-09）

生成器：`bm_evidence.py`（只读；不重算、不排序、不打分、不单选）。

```json
{
  "claim_id": "diffusion",
  "claim": "本谱低频尾需要 ω^(-1/2) 扩散项（Ws）",
  "grade": "B",
  "grade_reason": "由 guards: diff_amp_unidentifiable 判 B：数据无法判断（…）",
  "grade_source": "field",
  "not": "推不出「某机制不存在」：本窗+本噪声下只能给「不可区分」…",
  "evidence": [
    {"item": "aw_amp", "state": "read",
     "value": {"A_W": 0.78, "sd": 0.19, "cv": 0.24, "identifiable": false},
     "source": "auto_mechanism.share（bm_ws.ws_share_low）",
     "reads_as": "A_W 后验中位数、标准误与相对不确定度 cv",
     "not": "「A_W > 0」不等于扩散存在；「A_W 小 / cv 大」也不等于不存在…",
     "need": ["cv(A_W)=0.24 > 0.15：幅值没被定住，重跑不会变，需要更宽窗或更低噪声"]},
    {"item": "lowf_slope", "state": "unusable",
     "value": {"slope_im": -0.11, "r2": 0.9, "term": "to_cap"},
     "source": "auto_mechanism.gate.ws_diag（bm_ws.slope_im）",
     "reads_as": "最低频段 log(−Im)–log f 的斜率、线性度与 45° 段长",
     "not": "斜率 ≠ −0.5 不能推出「没有 Ws」：弧尾共存时真 Ws 的斜率会被拖到 −0.41…−0.49",
     "note": "斜率不可用（-Im 在低频端 ≤0 或线性度不足）",
     "need": ["把 f_min 下移 1 decade，让 45° 段变长"]},
    {"item": "freq_window", "state": "read",
     "value": {"level": "indistinguishable", "f_D": 0.071, "f_min": 0.1, "below_fD": 0.1466},
     "source": "diffusion_resolution（bm_diff）",
     "reads_as": "扩散特征频率 f_D 是否落在本频窗内（差几个 decade）",
     "not": "f_D 在窗外不能推出「没有该过程」；只能说「本窗无权评价」（C 档）",
     "need": ["f_D 在窗下：当前窗口看不到转折，需要把 f_min 下移"]}
  ],
  "next_evidence_needed": ["cv(A_W)=0.24 > 0.15：…", "并列保留候选模型，不要单选…"],
  "notes": []
}
```

字段约定（生成器实际输出）：

| 字段 | 取值 | 说明 |
|---|---|---|
| `claim_id` | `diffusion` / `adsorption` / `component_<k>` / `k_selection` | 一张卡一个可检验命题 |
| `state` | `read` / `unusable` / `missing` | 拿到可读值 / 有值但读不了（如 `im_usable=false`）/ 压根没采 |
| `grade` | `A` / `B` / `C` / `null` | 唯一判定，**不是分数** |
| `grade_source` | `field` / `recomputed` / `derived` | 选择器已写 / 按 `bm_grade` 同一张表补 / 按状态表派生 |
| `not` | 字符串 | 卡级与**每条证据**各一句，逐条不同 |
| `need` | 字符串列表 | 该条证据缺失/不可用时还差什么（卡级 `next_evidence_needed` 是并集） |

四条设计约束：

1. **每条证据自带 `not` 字段**——写死「这条不能推出什么」。这是防越界最有效的一招，比在报告里写免责声明可靠。`evidence_card_check.py` 会查「同一张卡里 `not` 的去重数 = 证据条数」，防止退回结论级。
2. **`grade` 是整张卡唯一的判定**，且只允许 A/B/C，**不允许分数**。不给分数就没有「0.72 分所以更可信」这种伪定量。
3. **`next_evidence_needed` 必填且非空**——逐条 `need` ∪ 档位最低要求（A 重跑 / B 并列 / C 独立实验），保证说不出「还缺什么」的卡不会出现。
4. **只读**：入口 `copy.deepcopy`，`fingerprint()` 对拍；生成器不改 `status` / `choice` / `delta` / `grade`。

### 4.3 多模型并列（开关，默认关）

`config.json` 的 `evidence.multi_model` 默认为 `false`——测试阶段不输出成堆结果；做成软件或需要人工从多个模型里挑时改 `true`，命令行 `--multi-model` 可临时打开。

打开后多一张 `multi_model` 表：每个 arm 一行（`Δ±dse`、资格档、`A_W`、塌陷/出窗标记），按输入顺序，**不排序、不打分、不单选、不做跨臂配对检验**（那需要 WP2 的逐点样本）。单臂没有 `auto_mechanism` 时 Δ 记 `null`，不补算。

标记只描述单臂状态，不是判据：`幅值落到噪声（p_diff_small>0.5）`、`幅值未定住（cv>0.15）`、`f_D 在窗下`、`HN 已呈 Warburg 指纹`、`峰位出窗`。跨臂差异另给一行 `amp_spread`（范围 + 倍数），并写明「哪个 arm 塌到小值由人看，程序不判哪个对」。

### 4.4 回归

`evidence/ws_crit/evidence_card_check.py`：**2889/2889 通过**（30 个选择器级合成例 + 3 个真实结果）。

覆盖：只读指纹逐位不变；每条证据 `not` 非空且互不重复；`next_evidence_needed` 非空；`grade ∈ {A,B,C,null}`；全文档无 `score` / `confidence` / `probability` / `weight` / `rank` / `best` / `selected` / `recommend`；`multi_model` 默认关闭、打开后保序；严格 JSON（`allow_nan=False`）。

真实数据抽查：WP2 的 blank 四臂（k1_none / k1_Ws / k2_none / k2_Ws）跑 `--multi-model`，`A_W` 范围 **0.778 – 4.184（5.4×）**，`k2_Ws` 标 `幅值落到噪声（P=0.53）`、`k2_none` 标 `峰位出窗：component_2`——与 WP2 §3 的「谁塌陷」逐项一致，且表里没有任何「推荐哪个 arm」的字段。


---

## 5. 明确不做（防 scope creep）

| 不做 | 理由 |
|---|---|
| 物理机制自动评分器 | 11 个独立单元估不出权重；分数会被当概率读 |
| 机制分类器（训练） | 同上，且合成谱标签是「推定」不是真值（AutoECM 9327 条只推定「无 ad 支路」） |
| 把判据阈值调得更激进以「多判出几个」 | 提高的是假阳性，不是能力 |
| 用文献值当真实谱的先验真值 | 文献只给量级（τ≈2.4 s、f_c≈0.065 Hz），不是本谱真值 |
| L3 输出单一「推荐模型」 | L2 已能给 Δ±dse；L3 再选一次等于把 B 档伪装成结论 |
| 把 **B-缺项** 当 A 档修（加采样、加迭代、提高预算） | 缺的是候选集不是采样。实测 AZ31 低频感抗比串联 `ωL` 的 2σ 上界大 6.9–8.5 个数量级，采样预算改不了这个量级；正确动作是补候选或换候选集 |

---

## 6. 近期顺序（与项目方建议一致）

1. **M6 真实数据基线**——确认现有算法在真实数据上的行为（不改判据）。
2. **Ws/Wo/K+1 比较与不可辨识性**——已完成（WP2），待 v2 复核后冻结。
3. **统一证据卡输出接口**——**已完成（2026-11-09，`bm_evidence.py`）**：按 §4 只读消费 `grade` / `guard_classes` / `component_grade` / `auto_mechanism`，逐条证据带 `not`，卡级 `next_evidence_needed` 必填；**不建评分**。多模型并列放在 `evidence.multi_model` 开关后面（默认关）。
4. **积累独立实验数据后**，再讨论是否自动化物理判定。

第 3 步的前置条件——拆开 §3.2 的三处混用——**已完成**（§3.4：`grade` / `guard_classes` / `grade_reason` / `not` / `component_grade`，生产代码只加字段、判决逐位不变）。证据卡生成器也已落地（§4.2，`bm_evidence.py` + `evidence_card_check.py` 2889/2889）。多模型并列按项目方要求**先做开关、默认关**（§4.3），等做成软件时再打开——测试阶段输出成堆 arm 反而难读。

---

## 7. 一句话

**目标不是「能自动解释 EIS 的算法」，是「知道自己何时有资格作物理解释的算法」。** 后者把科学边界本身当作产出，前者只是把边界藏进置信度里。
