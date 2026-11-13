# 新数据到底要补什么：容性最内层支路（2026-11-12，exploratory）

> 用户问「新数据应该会用到新拓扑结构，还是得加上，元件哪些有必要加哪些没必要」。
> 结论先说：**要加，但只加一个元件类型，不加新层、不加第三族。**
> 骨架已经对了（真值对拍 1e-15/1e-16），缺的是最内层 `R + CPE` 的**容性**。

## 1. 为什么不是「新拓扑 / 第 3 层」

`测试数据/` 三类合成谱的文件名即真值（`NOISE_ACF_TRUTH.md` §1）：

| 类 | 拓扑串 | 归属 | 需要什么 |
|---|---|---|---|
| 涂层 | `R1-[P2,R3-[P4,R5-P6]]` | nested **L=2** + 容性最内层 `R5+P6` | `branch=capacitive`, `L=2` |
| 钝化 | `[P1,P2-R3]-R4` | nested **L=1** + 容性最内层 `P2+R3` | `branch=capacitive`, `L=1` |
| 腐蚀 | `R1-[R2,P3]` | ZARC = voigt **K=1** | 已覆盖，什么都不用加 |

把 `Z_branch` 从 `R∥L` / `R+L` 换成**纯 CPE**（`Q_b`, `n_b` 两个参数）后，
现有两层骨架就能精确表示涂层、单层骨架就能精确表示钝化：串联电阻本来就在
网络里的 `R2`（L=2）/ `R1`（L=1）上，所以**不需要第三个并联节点**。

⇒ `layers=3`（文献 C2）与 C3「串联+嵌套混合」在这批数据里**都没有对应真值谱**；
§17 里 C2/C3 在 10 条未过真实谱上 M-b `0/10`，多 3 个参数只买 rms 不买 acf。
先补这两项属于「一条谱一个模型」，正是 `MODEL_COVERAGE_PLAN` §6 准入规则要拦的。

## 2. 元件取舍表

| 元件 / 结构 | 判断 | 依据 |
|---|---|---|
| **容性最内层 `Z_b = CPE(Q_b,n_b)`** | **必加，本轮已加** | 涂层/钝化 2000 条真值谱族外；骨架对、元件类型错 |
| 高频串联电感 `include_L` | **不用加，已有** | NaOH 14 条的 20–37 个 `Im>0` 点是已有元件，按 §9.4.11 单独预注册「能力臂」再开即可 |
| `layers=3` | 先别加 | 本批无第三个时间常数；§17 C2 `0/10`、对照 `12Li_50` 0.463→0.537 由过变不过；§9.4.37 上 C2 的 `n2` 顶 1.0 / `Q2=5.3e6` 退化 |
| C3 第三族 / 组合族 | 先别加 | 本批无串联+嵌套接点谱；§17 C3 `0/10`；§9.4.10.2 已证两族带 L 后不相交，加第三族要重做可辨识性与跨族选择 |
| 端部 `PhaHF` / `CPEtail` / 贴 `f_max` 的 R\|CPE | 先别加 | 那是 3E 外部第一名电路专用自由度（0.65% vs 1.47%），贴窗口端部，加了会把 acf 打散但不代表模型对 |
| `Wshort`(tanh) | 先别加 | `bm_diff.z_wshort` 有实现未接 CLI（P3）；`WS_CRITERION_PLAN` 里 Ws 判据死结、18 条真实谱低频是感抗；本批无有限长透射真值 |
| Gerischer / G | 不加 | 完全没有；HN 的 `φ=1,ζ=1/2` 只是极限说明，`Uniform(0.5,1)` 取不到 1；无真值谱、无判据 |
| 负电容 `C_neg` | 不加 | 覆盖地图 E0 另立项；`REFS_DIGEST` 引书 p.175 明确「不宜随意引入负电容」。别和 signed-DRT 的负 γ 混 |
| 多步吸附 ≥2 环 | 不加 | `ADSORPTION_PROJECT` §11 明确非目标；单 RL 都还有 `sampler_sensitive` 问题 |
| 理想电容 `n=1` | 不加 | 先验上界 1.0 取不到；§9.4.21.2 禁止把 `n` 固定成 1.0（拓扑主张会消失） |
| 并联混合 Warburg（ADAPT 式） | 不加 | 那是外部工具的模型族差异，不是我们的缺口；且 ADAPT 在坏谱 Rwp 29–82% 自己也没拟合好 |
| 低频感抗环 + 扩散共存 | 不加 | 覆盖地图 #6 一期明确屏蔽，需先实测 `ρ(R_ad,R_W)` |
| 非稳态 / 漂移建模 | 不加 | KK 只拦截不建模；L3 只记账不裁决 |
| 通用等效电路搜索 / 拓扑自动生成 / 全局有符号 DRT | 不加 | `NESTED_TOPOLOGY_PROJECT.md` §11 写死非目标 |

## 3. 本轮实现范围（守住的边界）

改动文件：

- `bm_top.py`：新增 `VALID_BRANCHES=("parallel","series","capacitive")`；`z_nested` /
  `j_z_nested` / `sampled_keys` / `_prior_table` / `param_spec` / `ALL_PARAM_KEYS`
  支持 `Q_b`,`n_b`；自检新增 `[14]`（逐节点、jax、渐近、采样键）。
- `nested_fit.py`：`--branch capacitive` 可选，`derived_quantities` 不再把容性档
  硬套 `L_ad` 公式。
- 新增只读验证 `docs/round_2026-11-12/capacitive_branch_check.py`。

**没动**：`bmdrt.py`（仍无 `--topology`，默认族仍 voigt）、`bm_core.py` /
`make_model` / `select_k`、`bm_select.py` 的 `BRANCHES`（跨族候选池仍是两支路）、
`bm_top_gen.py`、任何判据 / 阈值 / 先验（旧两支路）、§13 `0/3` / §14 `4/14` /
§17 `0/10` / §18 / §19。新档全标 `exploratory`，默认关。

## 4. 验证结果

`python bm_top.py` 自检 **全部通过**；旧两支路与 `git show HEAD:bm_top.py` 逐位对拍
`max|ΔZ| = 0.000e+00`（4 个组合，numpy + jax）；契约 23/23、`bmdrt check` 18/18、
`bm_top_gen` 档 1 全过、`py_compile` OK。

真值对拍（脚本独立写 `Z_P=Q/(i2πf)^n`，不调用项目正演）：

| 类 | n | `layers` | 与真值最大相对误差 |
|---|---|---|---|
| 涂层 | 5 | 2 | **2.57e-16** |
| 钝化 | 5 | 1 | **5.85e-16** |
| 腐蚀 | 5 | — | 由 voigt K=1 覆盖，不需要新支路 |

模型对时 CSV 残差 = 生成噪声：涂层 Rwp 11–12%、钝化 8.5–9.5%，与
`NOISE_ACF_TRUTH.md` 的噪声底同量级；acf 中位 0.354 / 0.376（噪声自带频率相关结构）。

**同谱、同 `standard` 档，`nested_fit.py` 配对对照**（涂层第一条）：

| 支路 | rms_rel | acf | R̂ | div |
|---|---|---|---|---|
| `parallel/L=2`（旧感性支路） | 10.81% | **0.944** | 14.21 | 0 |
| `capacitive/L=2`（新） | **7.08%** | **0.279** | **1.005** | 0 |

`acf` 0.944 → 0.279，且降到 0.50 线以下；`rms_rel 7.08%` 对应 Rwp≈10.0%，
贴到该谱噪声底 11–12%。高频端残差 16.86% 与 `NOISE_ACF_TRUTH.md` §4 记的
「17 点窗口端部、与模型无关」一致 → 换元件买到的是结构，不是硬贴高频。

## 5. 诚实留下的两个缺口

1. **`quick` 档不收敛**：同一涂层谱 `quick` 给 R̂ 40.2 / rms_rel 67.5%；`standard`
   才收敛。新支路必须用验收口径 `standard`（2000/1000、ta=0.98），别用 quick 预览。
2. **钝化 L=1 仍未收敛**：`standard` 给 R̂ 197.8 / acf 0.750，尽管正演精确。
   诊断是先验尺度不是拓扑：`zmax` 被低频阻断尾撑到 `1.98e7`，而真值
   `Rs=141 Ω` / `Q1=6.6e4` / `Q_b=3.3e5`，相对先验中位分别差 **10.24σ / 5.70σ / 4.10σ**
   （涂层也差 6.51σ / 5.65σ，但 NUTS 走出来了）。即 `0.2·zmax`、`1·zmax`
   这套为 AZ31 谱标定的先验中心，在低频被电容堵住的涂层/钝化谱上系统性偏移。
   **按项目纪律不擅自改先验**（§9.4.21.2「先诊断，不先改模型」、
   §11「不改 Voigt 族先验」），留作新支路的 M3 标定任务：先给 `Rs` / `Q_b`
   做数据无关的重参数化与剖面平坦度，再谈采样结论。

⇒ **「正演精确」≠「参数可辨识」**。本轮交付的是元件与接口，不是验收结论；
`nested` 仍未接 `bmdrt.py`，§13/§14 数字一个没改。

## 6. 复跑

```bash
cd D:/eis训练数据/bmdrt
python bm_top.py                                      # 自检含 [14]
python docs/round_2026-11-12/capacitive_branch_check.py --n-per-class 5
python test_contract.py --skip-fit                     # 23/23
python bmdrt.py check                                  # 18/18

# 研究接口（venv：D:/autoeis_env/Scripts/python.exe）
python nested_fit.py <涂层谱.csv> --branch capacitive --layers 2 --budget standard --no-plot
python nested_fit.py <钝化谱.csv> --branch capacitive --layers 1 --budget standard --no-plot
```
