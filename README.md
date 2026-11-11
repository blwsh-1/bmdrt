# bmdrt — 贝叶斯混合 DRT 的 EIS 拟合工具

把电化学阻抗谱 `Z(ω)` 分解成若干 Havriliak–Negami（HN）弛豫分量，用
**逐频率点 PSIS-LOO** 比较不同分量数 `K` 与扩散配置，自动选出最简约且可辨识的模型。
传统 Tikhonov DRT 只用来给 `K` 的搜索范围做预判断，**不参与似然、不决定最终 K**。

> 设计原则：**DRT 只做预判断，K 由证据选，不由 DRT 定。**

---

## 1. 三层算法架构

| 层 | 文件 | 作用 |
|----|------|------|
| ① DRT 预判断 | `drt.py` | Tikhonov 正则化反卷积 + L-curve 选 λ + 峰识别，只给出 `K` 的候选范围与 NUTS 初值 |
| ② 贝叶斯拟合 | `bm_core.py` | HN 基底 + stick-breaking 权重 + NUTS；似然只用原始 `Z'` / `Z''` |
| ③ 模型比较 | `bm_real.py` | 逐点联合似然 → PSIS-LOO → `select_k()` → 残差 / PPC / 峰位 / 扩散退化诊断 |

核心模型：

```
Z(ω) = R_inf + jωL + Σ_k w_k · R_k / [1 + (jωτ_k)^φ_k]^ζ_k  (+ 可选扩散项)
```

- `ζ=1, φ<1` → ZARC（R‖CPE）；`φ=ζ=1` → Debye；`φ=1, ζ=1/2` → Gerischer 型
- 权重用 stick-breaking：`v_k ~ Beta(1, α)`，`α` 默认固定 0.5
- 可选扩散：`Ws`（半无限 Warburg）/ `Wo`（有限长阻挡 Warburg）/ `both`（仅对照），默认 `none`
- 噪声为绝对口径：`Normal(Z', s_re)` 与 `Normal(Z'', s_im)` 分别作用

**逐频率点联合似然**是关键：同一频率的 `Z'` 与 `Z''` 必须一起留出/计入，

```
ll[a, j] = log p(Z'_j | θ_a) + log p(Z''_j | θ_a)
```

不能摊成 `2n` 个独立观测，否则 LOO 泄漏、`elpd` 系统性高估。

---

## 2. 快速开始

```bash
# 建议使用独立虚拟环境
python -m venv .venv
pip install -r requirements.txt

# 数学自检（不依赖真实数据，先跑这个确认环境）
python bmdrt.py check

# 单个样例
python bmdrt.py auto examples/example_caseE.csv --out out --no-move

# 无参数 = 处理 inbox/ 里的全部文件（等价于 auto）
python bmdrt.py

# Windows 一键：双击 bmdrt.bat
```

> **解释器**：本项目实际跑通的环境是 `D:\autoeis_env\Scripts\python.exe`（Python 3.10.7），
> `bmdrt.bat` 已把它写死。若你的 `python` 指向别的环境（例如系统里另装的 3.13），
> 模块自检会直接 `ModuleNotFoundError: numpy`——用绝对路径调解释器即可，不要重新装包。

输出按「一个输入 → 一个独立目录」组织，互不覆盖：

```
out/<名>/real_<名>.png          2×3 诊断图（数据 + 拟合 + 残差 + DRT）
out/<名>/real_<名>_circuit.png  等效电路图（参数 = 后验中位数）
out/<名>/real_<名>.json         全部指标
out/<名>/real_<名>_samples.npz  后验样本
out/<名>/run.log                完整控制台输出
```

---

## 3. 输入数据格式

`bm_real.load_spectrum()` 自动识别 `,`/`\t`/`;`/空白分隔，可带表头或纯数据。
支持两种列组合：

```csv
f_Hz,Zre,Zim
100000,12.3,-0.8
...
```

或 `|Z|,phase`（相位默认角度，`max|phase|>7` 时自动按度转弧度）。
无表头文件要求 3 列，顺序固定 `f, Zre, Zim`。

读数据时会自动做：非有限值过滤、`f>0`、按频率升序、重复频率点取平均。
**容性虚部为负**；Gamry 导出若符号相反，用 `convert_2e.py` 的相位自检自动翻转。

`examples/` 里有 8 个样例，覆盖两种列格式、无表头、含 Warburg 尾、仅 `Wo` 等情形。

---

## 4. 命令行

```
python bmdrt.py                      # = auto：处理 inbox/
python bmdrt.py auto <文件或目录...>  # 指定输入
python bmdrt.py watch                # 常驻监听 inbox/
python bmdrt.py check [--fit]        # 数学自检（--fit 连拟合一起跑）
python bmdrt.py diff [--fast]        # 扩散项回归（B1/B2/C1/Wo/B2-ident/DRT）
python bmdrt.py bench                # 合成 benchmark
python bmdrt.py list / config / help
```

常用覆盖项（`auto` / `watch` 有效）：

```
--krange 1,4   --wide   --kmax 6   --quick   --ta 0.95
--L / --no-L   --zeta 1.0|free   --diffusion none|Ws|Wo|both
--kk gate|warn|off   --sigma-floor 1e-4   --no-deinduce
--no-move      --out DIR   --inbox DIR   --interval 30
```

`--kk` 控制**拟合前**的 Lin-KK 门禁（默认 `gate`）：

| 模式 | 行为 |
|------|------|
| `gate` | 判 `fail` 就跳过整条拟合链，写 `fit_performed=false` 的结果 json，不输出 K |
| `warn` | 只打印门禁报告，仍继续拟合 |
| `off`  | 完全跳过门禁 |

`fit_performed=false` 是批量流程的契约字段：`KK 拒绝` **不是运行失败**，
也不是一次成功拟合——`bmdrt.py` 的汇总会把它单独计数，`batch_2e.py`
不会因此把整批退出码变红。数值阈值（`c` / `max_M` / `res_tol` / `auto_L` /
`auto_tail` / `auto_cap`）在 `config.json` 的 `kk` 段。

`--sigma-floor` 是**噪声地板**（相对 `zmax`，默认 `0` = 关闭，行为与改动前
逐位一致）。`s_re/s_im` 原本是无下界的 LogNormal，遇到无噪/近无噪数据会
一路塌到相对 1e-8，形成 σ→0 的漏斗，NUTS 树深饱和（实测触顶 53%）、
采样分级掉到 0、K 选择被全判"不可用"。地板把漏斗截断。

> **不要为了让 R̂ 好看而调地板。** 它是数值正则化项，不是测量噪声估计。
> 真实 2E 谱的 `σ_im/zmax` 在 1e-3–1e-2，1e-4 的地板对它们基本不起作用。
> 要定值请用 `sweep_sigma_floor.py` 按五档扫描，看参数/K 稳定性、
> 树深饱和、参数相关性、预测精度五项，而不是只看 R̂。

**实测（`out/ecm_sweep/SIGMA_SWEEP.md`，797 + 1139 × 四档，quick 档）：**

| | 1e-6 | 1e-5 | 1e-4 | 1e-3 |
|---|---|---|---|---|
| 触顶 797 | **54%** | 2% | 0% | 0% |
| 触顶 1139 | 19% | **51%** | 0% | 0% |
| `R_inf` 797（真值 0.0508） | 0.05085 | 0.05085 | 0.05097 | **0.06334** |
| `R_inf` 1139（真值 728） | 727.8 | 727.9 | 729.1 | **804.5** |

- **`1e-4` 是两条谱上都能把触顶压到 0% 的最小档**；`1e-3` 压不出更多，只把
  `R_inf` 推偏 10–25%。
- **地板不单调**：1139 上 `1e-5` 比 `1e-6` 更糟（触顶 51% vs 19%）。别以为
  取中间值更安全，必须逐谱确认。
- **`R̂` 不能用来选地板**：797 在触顶 54% 时 `R̂=1.008`，比触顶 0% 的档还漂亮。
- 代价：1139 在 `1e-4` 下 `rms/σ=0.01`，即 σ 比实际残差大 100 倍，噪声模型
  过度离散。参数仍准，但 **σ 不能再当噪声估计报出去**。

**full 档确认（3000/1000/3，`out/ecm_full/SIGMA_SWEEP.md`）**：`1e-4` 与 `1e-3`
都到分级 2（触顶 0%、tail ESS 1179/1286、`R̂≈1.002`、div 0），说明 quick 档的
分级 1 只是预算不足。但 `R_inf` 相对真值：`1e-4` 是 +0.4% / +0.03%，
`1e-3` 是 **+24.4% / +10.2%**。所以：

> **无噪/近无噪数据用 `1e-4`，并在报告里声明 σ 不代表测量噪声。**
> 不设为全局默认——真实 2E 谱的 `σ_im/zmax` 在 1e-3–1e-2，地板对它们不起作用。

> 另外注意：漏斗有两个来源。无噪数据本身会让 σ→0（1139 即如此）；**模型缺项**
> 也会——797 真值含 `L=1.18e-7 H`，但 `detect_L()` 因高频 `Im Z<0` 投了 `False`，
> 缺 `jωL` 后实部精确、虚部留结构残差，`σ_re/σ_im` 差 6000 倍。
> 但**不能据此说「根因在 include_L、补上就好了」**：`test_include_L.py` 的对照
> 显示补上 `L` 后 `σ_im/zmax` 确实从 8.12e-3 掉到 4.22e-8，触顶却只从 53.4%
> 降到 45.6%，而 `R̂` 反而 1.009 → 1.841、bulk ESS 掉到 3 —— `L` 与
> `f0=232 kHz` 的快分量近简并。这是拿残差问题换收敛问题，必须单独一轮做。

`--no-deinduce` 关闭 **DRT 去电感通道**（见 §6）。

### v6 冻结决策与后验几何解锁条件

三条路径必须**独立控制**，不要当成同一个「电感问题」：

| 路径 | 作用 | 决定 |
|---|---|---|
| `detect_L / include_L` | BM-DRT 似然里是否含 `jωL` | **冻结**，不改判定逻辑与模型结构 |
| `deinduce_for_drt()` | 是否从传统 DRT 输入扣 `jωL` | 保留现有实现，不扩展 |
| `sigma_floor` | 限制有效噪声标准差 | 优先验证，第一轮不换先验分布 |

第一轮实验协议（`sweep_sigma_floor.py --floors 0,1e-5,3e-5,1e-4`）：
固定数据、固定 K=真值 K、固定 `include_L`、固定 NUTS 预算与 `seed=0`，
**只让 `sigma_floor` 一个变量变**；不做 K 扫描。完整只读审计与结果见工作区
《v6_噪声模型第一轮审计与F0-F3对照》（已归档，不在本仓库 `docs/`）。

进入第二轮 `include_L` 消融的**六项解锁条件**（门槛写死在扫描脚本）：

| 检查项 | 门槛 |
|---|---|
| R̂ | ≤ 1.01 |
| Bulk / Tail ESS | 均 ≥ 400（必须用正式档判，quick 档样本不够） |
| Divergences | 0 |
| Tree-depth saturation | < 1% |
| 噪声后验 | 不堆积在人为下限；若贴地板，必须**明确声明为数值正则化** |
| 参数相关性 | 无严重且**无法解释**的简并 |

**触顶降到 0% 不等于后验健康。** 地板把漏斗截断了而已；
`s_re_base`/`s_im_base` 落在后验样本 `real_*_samples.npz` 里，
报告会自动算「σ ≤ 1.1×floor 的样本占比」来判这一项。

只有当六项都通过，才按「真值无 L / 明显 L / 弱 L」三类做
`include_L=False/True` 配对消融；**消融完成后才考虑改 `detect_L` 阈值**。

**第一轮 F0–F3 + 正式档回归实测结论**：
quick 档 8 组六项判定**全部 `NOT YET`**（`out/ecm_f0f3/SIGMA_SWEEP.md`）；
补跑正式档 F1/F2 并与已有 F3 合并成 6 组（`out/ecm_f0f3full/SIGMA_SWEEP.md`）后，
R̂ / ESS / div / 触顶 **6/6 全部达标**，但总判定仍一律 `CONDITIONAL`
—— 卡在 ⑤ σ 后验被地板主导、⑥ 797 的三对结构性简并，与采样预算无关。
所以**第 5 步消融不启动**。

`1e-4` 的「定值」地位已**撤回**：正式档下 `1e-5` 就已触顶 0%、ESS 充足，
且 `floor 占 σ_eff` 更低（1139 86% vs 90%）、`rms/σ` 失真更轻（0.08 vs 0.01）。
它现在只是「无噪/近无噪谱的**数值正则化**候选开关之一」，不是默认值、
不是正式推荐值、也不是测量噪声估计。

另：**quick 档 1139 的「非单调触顶 + 6 对 `|ρ|≈1` 简并」已被正式档推翻**
（F1/F2 均 0% 触顶、0 对强相关），属 800 样本 / 2 链的预算假象 ——
quick 档的触顶率、ESS 和相关矩阵只能用于筛方向，不能用于定结论。

**0.5% 噪声正式档回归**（`out/ecm_noise50bp_full/SIGMA_SWEEP.md`）：
1139 在 `floor=0` 与 `1e-4` 下**六项全 `PASS`**（项目首个完全解锁单元），
说明 σ 塌陷只是无噪合成谱的数值现象、真实噪声下地板不必要；
797 在同样条件下 R̂≈1.67、bulk ESS≈5、`R_inf` 偏 +210%，正式档不改善，
属 `include_L` 缺失导致的模型结构问题 —— 按审查意见冻结到第二轮，本轮不动。

**正式档 K 选择**（`out/ecm_kselect_full/SIGMA_SWEEP.md`，`--auto-k --kmax 3 --full`）：
797 / 1139 分别选到真值 `K=2` / `K=1`，`k_selection_status=converged`，
R̂ / ESS / div / 触顶全达标，`K=3`（R̂ 1.053 / div 169）与 `K=2`（div 69）
被硬筛选剔除；但六项总判定仍 `CONDITIONAL`（797 三对结构性简并、
1139 σ 后验 34% 贴地板 / `floor 占 σ_eff` 90%）。
**放开 K 扫描不改变瓶颈位置**，第 5 步 `include_L` 消融仍不启动。
完整只读审计、逐项判定与第 4 步清单见工作区《v6_噪声模型第一轮审计与F0-F3对照》（已归档）。

### 采样状态的四种取值

`k_selection_status` 现在区分四档，**不要把它们混为一谈**：

| 状态 | 含义 |
|---|---|
| `converged` | **被选中的 K** 采样达到严格线（分级 2，原 `ok`） |
| `degraded` | **被选中的 K** 只到「可用」档（分级 1），没过严格线 |
| `rejected` | 正式档下所有 K 都被剔除，结果只是相对排序 |
| `exploratory` | **quick 档专属**：采样预算撑不起严格收敛线，K 与 elpd 都不是选择结论 |

quick 档（800 warmup / 400 采样 / 2 链）与严格线（`tail ESS>400`、`触顶<1%`）
不在同一个采样预算尺度上，所以它永远不会输出 `rejected`——那会把「预算不足」
误读成「模型被统计证据否定」。

**2026-11-11 修正（R22）**：上面两条 `converged`/`degraded` 的定义此前与实现不一致。
旧实现把分级 1 和分级 2 的候选混进同一个池按 `elpd` 混排，且 `status` 只看
「池非空」就报 `converged`，于是会出现「选中 K 的分级只有 1、状态却写
`converged`」的自相矛盾（历史产物里 **3/381** 条，全在 `AZ31-12Li_0`）。
现在改为**分档选择**：严格池非空时「可用」档不参与最优档比较，`status` 跟被选中的
那一档走，并落盘 `selection_tier`（2/1/0）。降级候选若 `elpd` 反而更高，
会在 `k_selection` 的 note 里**显式写出**，不静默丢弃。
检查脚本：`docs/round_2026-11-11/k_tier_check.py`。

任何 `cmd_<名字>.py` 里写一个 `run(argv)` 就会自动多一个子命令，无需修改入口。

默认参数集中在 `config.json`，改配置比记命令行更可靠：

```json
{"fit": {"kmax": 5, "zeta": "1.0", "diffusion": "none", ...},
 "batch": {"min_points": 8, "min_decades": 1.0, ...}}
```

---

## 5. 目录结构

保持**扁平布局**：`bmdrt.py` 用 `Path(__file__).parent` 定位 `config.json`，
并靠文件名约定自动发现 `cmd_*.py`，移动源码会破坏这一机制。

```
bmdrt.py          唯一入口
config.json       默认参数
bm_core.py        模型 / NUTS / 诊断原语
bm_real.py        主流程与 K 选择
bm_kk.py          拟合前 Lin-KK 分级门禁（ok/warn/fail/unavailable）
bm_corr.py        全参数后验相关矩阵 + 90% 区间（可单独对 npz 重算）
bm_diff.py        扩散元件 + HN↔Warburg 退化诊断 + Ws/Wo 可分辨性
bm_ad.py          低频吸附电感元件 R_ad∥L_ad（M1–M5，默认关）
bm_top.py         嵌套并联拓扑（M1 正演 + M3 原型 fit_top 独立 NUTS，未经注册表）
bm_top_gen.py     嵌套并联拓扑的**独立生成器**（M3 档 1：四种拓扑手写导纳递推）
drt.py            传统 Tikhonov DRT 预判断
bm_experiments.py R̂ / ESS / 树深 / acf
bm_check.py       18 项数学自检
bm_bench.py       合成 benchmark
bm_ident.py       可辨识性分析
bm_circuit.py     等效电路图绘制
cmd_diff.py       扩散项回归
cmd_status.py     汇总 out/ 结果表
batch_2e.py       117 条谱批量运行
convert_2e.py     Gamry xlsx → 三列 CSV
report_2e.py      批次报告生成
fit_store.py      drt.py 命令行模式的轻量读取器（兼容 shim）
examples/         8 个最小样例数据
docs/cases/       真实谱模型选择诊断案例（规格、回放、逐点 LOO 数据）
evidence/b2x/     配对 LOO 事后重算脚本
```

---

## 6. 数据预处理现状

**已做**（在读取与建模环节内）：
文件清洗与列识别、虚部符号自检、频率轴升序/去重/非有限值过滤、
高频感抗自动检测 `detect_L()` 并在模型内以 `jωL` 显式建模、
逐点联合似然与 LOO、残差 / PPC / 峰位诊断。

**已接入**（本轮新增）：

- **Lin-KK 门禁** `bm_kk.py`：分频段相对残差（按 `log f` 三分）+ 残差结构判据
  （实/虚部 lag-1 自相关同时 > 0.5 才算系统性），输出 `ok / warn / fail /
  unavailable` 四级，`gate` 模式下 `fail` 直接拒绝定 K。RC 阶数 M 由残差
  肘部 `rms_all·(1+0.02M)` 选择（带病态/停滞护栏）；μ 只作过拟合参考，
  不再当收敛门禁——官方 `while mu > c` 早停会让含大电感的低 |Z| 谱欠拟合、
  误判为不自洽。
- **全参数相关矩阵** `bm_corr.py`：先 `match_components` 对齐编号再算 Pearson
  （不对齐会得到假相关），尺度参数取对数；输出 `|ρ|≥0.9` / `0.7–0.9` 两级成对
  清单与已知简并方向（R–τ 脊线、`R_inf`–`W` 补偿、`R_W`–`τ_D` 简并）。
- **Ws/Wo 可分辨性** `bm_diff.diffusion_resolution()`：两核在 `ωτ_D≫1` 时同形，
  先判 `f_min < f_D` 且带内最大相对差 > `tol`，否则报「不可分」，
  不许用 elpd 差宣称有限长扩散。
- **DRT 去电感数据通道** `bm_real.deinduce_for_drt()`：见下。
- **多分量扩散弱识别** `bm_diff.warburg_band_slope()` + `warburg_multicomponent()`：
  `looks_like_warburg()` 只看单个分量的 φζ / 峰位 / R–τ 简并；K≥2 时模型可以
  用「多个 φ≈0.5 的分量拼 45° 段」逼近扩散尾，这条路径它看不到。补一条查
  **合成阻抗 Nyquist 斜率**（`d(−Im Z)/d(Re Z) ≈ 1`）的弱识别。
  用 Nyquist 而不是 `d log|Z|/d log f`：后者被串联电阻压平（实测 4382 的
  `|Z|` 斜率只有 −0.03，Nyquist 是 +0.40）。**只给 `weak`、不改 K 选择**，
  因为单个宽化的 `R‖CPE(n≈0.5)` 也可能满足。

### DRT 去电感通道

DRT 是 Tikhonov 非负反演，基底 `Σ γ(τ)/(1+jωτ)` **表示不了 `jωL`**。
有感性尾时它会在高频网格边缘长出一个伪峰去扛，把 K 候选撑大、把峰位挤偏。

所以分两条数据通道：

```
Z_DRT = Z_raw − jωL_est   →  只喂 drt.prescreen()（K 候选 + 初值）
Z_raw                     →  喂 BM-DRT 的似然（不能改，否则后验不再是
                             「给定观测」的后验）
```

**不能无条件扣**，四道检查全过才扣（阈值在 `config.json` 的 `drt` 段）：

| 检查 | 条件 | 不过时 |
|---|---|---|
| 有感抗点 | 最高 5 点里 ≥3 个 `Im Z > 0` | 用原始 Z |
| 稳定性 | `ImZ/ω` 的 `(p90−p10)/median ≤ 35%` | 保留原始 Z，标记电感干扰 |
| 占比 | `ω_max·L / |Z(f_max)| ≥ 0.5%` | 扣了白扣，跳过 |
| 不过扣 | 扣完**最高 3 点** `Im Z ≥ −2%·|Z|` | L 估大，保留原始 Z |
| 有收益 | DRT 高频段相对残差至少降 20% | 保留原始 Z |

过扣检查只查最高 3 点：`ImZ/ω` 在**高频段下沿**本来就会偏低（容性贡献还没
衰减完），按 top-5 取最小必然误报（实测 2EMC1-1_0005mA：top-5 最小 −4.78%，
top-3 是 0.00/+1.13/+1.49%）。

实测效果（真实 2E 谱，f=1–1e5 Hz）：

| 谱 | L_est | `ImZ/ω` 离散度 | f_max 处占比 | DRT 高频残差 |
|---|---|---|---|---|
| 2EMC1-1_0005mA | 6.06e-8 H | 7% | 87.3% | 48.54% → **14.65%** |
| 2EMC1-2_0005mA | 4.32e-8 H | 1% | 96.9% | 63.90% → **12.54%** |

决策（`used` / `L` / 四道检查 / 扣前后残差）全部写进 `real_*.json` 的
`drt_deinduce` 段，事后可复核。


**仍缺**：
遮掩不合格点后重跑 KK、跨噪声实现的重复性统计、`(K, diffusion)` 三倍扫描
正式纳入 `select_k`、`phi_trend.py` 的跨温度/循环趋势检查。

> **KK 是事前门禁，不是事后诊断**：
> 数据不满足因果/线性/稳定时，增加 K 只会更会贴噪声，PSIS-LOO 只能在
> 「似然假设成立」的前提下比较模型。

实现上有两个坑必须记住（都是实测逼出来的）：

1. **不要在 Lin-KK 之前扣 `jωL`。** `impedance.validation.fit_linKK` 的
   设计矩阵已含自由串联电感列（`a_im[:, -1] = w/|Z|`），再扣一次是重复计入。
   实测真实 2E 谱：原谱高/中/低 = 9.60/1.92/11.44 %，扣 `jωL` 后变成
   10.78/2.20/11.44 %——**高频残差反而变差**。`estimate_L()` 只作诊断量。
2. **低频扩散尾不要截断丢点。** 用 `add_cap=True`（同一频率网格加串联电容）
   复验：原谱低段 11.44 % → 4.72 %，且中/高频同时改善；直接截尾会把
   中段从 1.92 % 推到 4.33 %。截断点 `f_cut` 只作诊断，不真正丢数据。

> 关于低频：本仓库**不使用 DCT**。DCT 是导纳域的电容时间分布，
> 不是低频专用工具；在阻抗域正确做法是把
> `(K, diffusion=none) / (K, diffusion=Ws) / (K, diffusion=Wo)`
> 纳入同一套 LOO 模型比较（详见工作区《低频数据是否用DCT_结论》，已归档）。
> **前置条件**：本批实测谱 `f_min ≈ 1 Hz`，而 `W_open/W_short` 只在
> `f ≲ f_D = 1/(2πτ_D)` 才偏离半无限 Warburg；频段不够时三者不可分，
> 应报告「不可分」而不是选出「最优扩散模型」。这条判据现在由
> `bm_diff.diffusion_resolution()` 自动给出，结果写进
> `real_*.json` 的 `diffusion_resolution`。

**已修复的风险**：`select_k()` 原来的 `pool = ok or usable or list(res_list)`
在全部模型被硬剔除时仍会输出「相对最好」，等于把相对排序包装成最优 K。
现在返回 `selection_status ∈ {converged, degraded, rejected, exploratory,
not_compared, rejected_by_kk}`（旧结果文件里的 `ok` 是 `converged` 的别名），
`rejected` 时报告与 JSON 都写明「不输出最优 K 的结论」。

**仍存在的限制**：`report_2e.py` 记录该批 117 条谱未做扩散对照——这是
`batch_2e.py` 没传 `--diffusion` 造成的，不是判据问题。

---

## 7. 文档

2026-11-11 起，设计与验证文档统一放在 **`docs/`**，总索引见 `docs/INDEX.md`
（本节只列主链路相关的几份；`THREE_LAYER_SCOPE.md`、`WS_*`、两轮代码审查报告等
都在 `docs/` 下）。代码注释里的交叉引用仍用裸文件名 + 章节号，不受迁移影响。

| 文档 | 内容 |
|------|------|
| `docs/DIFFUSION_PLAN.md` | 扩散扩展方案：数学、先验、测试矩阵、模型选择 |
| `docs/ADSORPTION_PROJECT.md` | **低频吸附电感元件立项书**：`R_ad∥L_ad` 定义、负 RC 恒等式、DRT 盲区、三区间可辨识性、M0–M6 里程碑与验收标准 |
| `docs/MODEL_COVERAGE_PLAN.md` | **模型体系规划与覆盖地图**：三层架构（基础 / 特殊响应 / 族选择与可辨识性）、11 项现象覆盖状态与证据等级、注册表接口约定、可辨识性报告规范、按证据排序的立项队列与准入规则 |
| `docs/BM_TOP_M3_PLAN.md` | **`fit_top()` 接口设计与 M3 验收清单（v3，已按审查冻结）**：现有采样接口摊开、候选 A/B/C 对比、参数先验冻结表、生成器独立性、三档验收与失败记录规范、`fit_top()` 实现记录、档 1 记录（§12）、档 2 第一组 5 seed（§15）、事前可辨识性筛查（§16）、M3 收尾（§19）、M3.5 逐点相对噪声的接口/回归/预注册与 LOO 有效性、双 seed 分离、判定链路分离（退出码 / 收敛数 / 主终点，§20.13–20.14）、两层判定与覆盖口径（§20.4.1 / §20.15）、正式采样配置对齐与首次正式跑（§20.16）、warmup 5000 重采与 M3.5 结案（§20.17） |
| `docs/NESTED_TOPOLOGY_PROJECT.md` | **嵌套并联电路拓扑立项书**：为何串联 Voigt 表达不了低频感抗环（三个硬约束）、理论上限对照（30.5% vs 6.7%）、渐近极限、`Rs≈0` 不可辨识警报、CPE 窗口幅值参数化、M0–M6（M0/M1 完成；M2 以独立 `fit_top()` 替代，未动 `bm_core`；M3 进行中） |
| `evidence/nested_topology/theoretical_bound.py` | 表达力证据：scipy 全局最小二乘对照，内置合成谱自包含，也可用 `--data` 跑真实谱 |
| `evidence/nested_topology/identifiability_check.py` | 可辨识性证据：渐近极限 / `Rs` 剖面 / 高频窗口截断，同样支持内置合成与 `--data` |
| `evidence/nested_topology/m3_stage2.py` | 参数恢复驱动：`--stage a`（固定 σ）/ `b`（自由 σ）/ `r`（**逐点相对 `rel-perpoint-2pct`**，M3.5）分记、`--truth-set a/b` 切真值组、**`--data-seeds` 与 `--seeds` 分离数据/采样器随机源**、多 seed 稳健性汇总（均值/SD/max\|err\|/逐 seed 达标）、**预注册主终点 `sd_r`（逐 draw 算 `r=R1/(R1+R2)`，带收敛门禁；不可判时 `value=null`、原始读数在 `observed_value`）**、**两层判定（`full_success` / `primary_only` / `partial` / `negative` / `indeterminate`，覆盖口径 `all_seeds` 冻结）**、`--selftest-verdict`（9 例纯函数自检）、**`exit_gate()` 分离退出码判据（`r` 只看运行有效性、`a`/`b` 沿用 `acceptance`）**、`n_mcmc_converged` 与 `n_pass_acceptance` 分列、逐参数 R̂/ESS、组合参数与相关矩阵、`--save-samples` 留后验 |
| `evidence/nested_topology/m3_design_check.py` | **事前可辨识性筛查**：真值处一阶 Fisher → 各参数"数据能定到多准"，与先验 sd 比判"是否先验主导"；跑 MCMC 之前就能筛掉本来就定不住的参数。含 A/B1/B2/B3 候选真值扫描与"换杠杆"对照 |
| `evidence/nested_topology/m3_rank_check.py` | 秩与拆分方向：噪声无关 Jacobian 的秩/条件数（判定"结构性"还是"实践性"不可辨识）+ `R1/(R1+R2)` 的 delta 法区间；含**列归一化**版本（无量纲条件数、相对阈值秩）与 `J_log=J_θ·θ` 尺度恒等式核对；归一化前后最弱方向在**同一坐标系**比夹角（`cos≈1.000`，更正上一轮的 0.60）；秩明确标为**局部**满列秩；`split_block` 前向统一走 `z_nested`（与 `jacobian` 同源） |
| `evidence/nested_topology/m35_r_w5000.{json,log}` | **M3.5 重采**（warmup 5000，其余同）：门禁 3/5，仍 `indeterminate`。两套 warmup 的 `sd_r` 逐 seed 差 ≤1.6e-3 → 后验可复现，但按预注册停止规则不再重采 |
| `evidence/nested_topology/m35_r.{json,log}` | **M3.5 首次正式跑**：5 个独立噪声实现 × 固定 sampler seed（4 链 × 1200 / warmup 2000 / ta 0.95 / depth 12）。判定 `indeterminate`（收敛门禁 2/5），`sd_r` 中位 0.0300 记在 `observed_value`、`value=null` |
| `evidence/nested_topology/m3_regress_scalar.py` | **标量噪声路径回归**：同环境同 seed 重跑冻结配置，与 `g1_seeds_a.json` 对比，分"逐位一致 / 容差一致 / 失败"三档（`bm_top` 引入逐点 σ 后用它证明旧路径没动） |
| `evidence/nested_topology/g1_seeds_a.{json,log}` | 档 2 第一组（parallel L2，固定 2% 噪声）5 个 seed 的逐次记录与汇总 |
| `evidence/nested_topology/g2_seeds_b.{json,log}` | 档 2 B 组（`R_ad=300`）2 个 seed：形状参数未收缩的负结果，`R1/(R1+R2)` 90% 区间覆盖 `[0.01,0.99]` |
| `docs/AD_M6_SAMPLING.md` | **吸附电感 M6 抽样验证记录**：18 条真实含环谱盘点、4+10 分层抽样设计、首个真实正例（正式档复核）、门控两处缺陷、`inv_from_lowfreq` 偏差实证、抽样置信上界与分阶段全量方案 |
| `docs/AD_GATE_SPEC.md` | **吸附门控边界状态规格**：`enter`/`strong` 严格分离、五状态定义、新增 `edge_peak`、`d_low` 与采样密度判据（`2/dens`）、"不增加假阳性"的不变性论证、M6 开发顺序与验收口径 |
| `docs/M6_VALIDATION_PLAN.md` | **M6 实测数据验证预注册方案（v0.1，待审查，未执行）**：合并吸附线 / nested 线的两个 M6 并分开验收；24 条真实谱盘点（AZ31 18 + Blank 6）、117 条 2E 的旧基线为何不能直接比（11-05 早于三处规则变更）；M6.1 基线 / M6.2 边界 / M6.3 假阳性三层判据（G1–G7）、**K 范围必须 `--wide` 而非 DRT 预判**、复用与必须重做的证据清单（E1–E7）、报告字段与五条禁止事项、判据变更规则 |
| `evidence/ad_m6/sampling_check.py` | M6 抽样脚本：`inventory` 盘点 / `prepare` 转 csv+分层 / `summarize` 汇总结果 / `edges` 边界峰与 `d_low` 分析 |
| `evidence/kk_review/verify_review.py` | **外部审查意见的逐条核实**（残差口径是绝对还是相对 / Lin-KK 是否自带串联电感、M 有没有被强行加大 / 三条护栏常数），只读复现，不改代码与结果 |
| `docs/cases/blank_0.1KBr_pH10/` | **真实 EIS 谱模型选择诊断案例**（最新版本）：四层可辨识性规格、逐频点 LOO 与配对 dse、精确 HN 参考频率贡献、2×3 统一比较、合成谱用例与历史回放。文档 / 脚本 / CSV / JSON 齐备；NPZ 与 run.log 按约定不入库 |

`evidence/b2x/paired_loo.py` 用于跨配置配对 LOO：
`bmdrt.py auto --krange 1,1` 只跑一个模型，占位 `elpd_diff/dse` 需事后从 `samples.npz` 重算。

---

## 8. 数据与许可

原始 Gamry `.xlsx` 测量数据、`out/`、`figures/`、`*.npz` 等运行产物由
`.gitignore` 排除，不包含在本仓库中。`convert_2e.py` 需要本地
`EISmeasurements/` 目录才能运行。

许可证见 `LICENSE`；若需改为专有/不公开，直接删除该文件即可。

---

## 9. 运行环境

在 Python 3.10.7 / Windows 上验证（`D:\autoeis_env`），实际跑通版本：
`numpy 2.2.6, scipy 1.15.3, jax/jaxlib 0.6.2, numpyro 0.19.0,
arviz 0.23.4, matplotlib 3.10.9, pandas 2.3.3, openpyxl 3.1.5,
impedance 1.7.1`。

`bmdrt.bat` 里写死了 `D:\autoeis_env\Scripts\python.exe`，
找不到时回退到 `PATH` 上的 `python`；换机器请改这一行。
