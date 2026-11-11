# 2026-11-11 复核轮：证据脚本与残差报告

这一轮（二轮外部审查之后）新增的诊断脚本和报告。原来分散在
`evidence/k_select/`、`evidence/nested_topology/`、`evidence/data_mismatch/` 下；
`evidence/*` 在 `.gitignore` 里是白名单制，散放容易漏入库，2026-11-11 起统一收到这里
（`docs/` 不在忽略范围内）。

五个脚本都**只读诊断**：不改门禁、不改选 K 规则、不接生产管线。

## 文件

| 文件 | 干什么 | 跑法（**必须在仓库根目录**） | 产物 |
|---|---|---|---|
| `k_tier_check.py` | R22：`select_k()` 的分级分档是否真的生效（构造候选，6 条用例，**不跑 MCMC**） | `python docs/round_2026-11-11/k_tier_check.py` | 无（stdout 判据） |
| `m4_zmax_scale_probe.py` | R21：先验尺度 `zmax = max\|Z\|` 的数据依赖对跨族 `Δelpd` 的影响 | `python docs/round_2026-11-11/m4_zmax_scale_probe.py --specs AZ31-8Li_50 --scales 0.5 1 2 --budget light` | `out/m4_zmax_scale_probe.json` |
| `ext_circuit_repro.py` | 复现外部软件（EIS Battery Intelligence）两个吸附电路，用来先证明口径对得上 | `python docs/round_2026-11-11/ext_circuit_repro.py --specs AZ31-4Li_100` | `out/ext_circuit_repro.json` |
| `resid_structure_cmp.py` | 八模型 + Lin-KK（M=12 肘部阶）的残差结构对比（幅值 / 形状 / 符号 / 峰位），含文献 Shahri 2024 Table 1 的 C1–C3 | `python docs/round_2026-11-11/resid_structure_cmp.py --specs AZ31-4Li_100` | `out/resid_structure_cmp.{json,png}` |
| `linkk_order_probe.py` | Lin-KK 阶数曲线（M=1…45）：8.49% 是「某一阶的拟合水平」而非下界 | `python docs/round_2026-11-11/linkk_order_probe.py --allow-overfit --max-M 45` | `out/linkk_order_probe.{json,md}` |
| `dash_smoke.js` | 面板脚本的 DOM 桩冒烟（node）：boot 不抛异常、四个渲染器都产出非空 SVG | `node docs/round_2026-11-11/dash_smoke.js <提取出的 dashboard.js>` | — |
| `AZ31-4Li_100_RESID.md` | 上面那条谱的完整报告：接法判定、六个模型、Lin-KK 阶数曲线、七条边界 | — | — |

`resid_structure_cmp.py` 会 `from ext_circuit_repro import predict`，两个文件必须放在同一目录。

## 三条结论

1. **R22（K 选择）**：`select_k()` 原先把分级 1（降级可用）和分级 2（严格通过）塞进同一个
   池按 `elpd` 混排，且 `status` 只看池非空 → 出现「选中 K 分级=1、状态却 `converged`」。
   修复后分档取（strict → usable）、`status` 跟选中档走。审计 381 条既有产物：
   该组合只出现 **3 条**（全在 `AZ31-12Li_0`），K 会变仅 **1 条**
   （`AZ31-12Li_0__ge10Hz`，2→1）。
2. **R21（先验尺度）**：`zmax` 随留出点变化，严格说 `elpd` 不是标准 `p(Z_j|Z_{-j})`。
   把尺度人为移动 ±2×：`8Li_50` 的 `Δelpd` = +146.34 / +146.61 / +146.57（跨度 0.27 nats），
   `4Li_0` = +25.67 / +25.11 / +25.60（跨度 1.21），**0/2 决策翻转**。
   即：口径上有瑕疵，结论上不翻转；标注 + 敏感性即可，不必为此重跑 LOO。
3. **数据失配（`AZ31-4Li_100`）**：残差不是「一样大」，是**同一条曲线**——六个电路
   （我们阶梯、外部两个、文献 C1/C2/C3）允许缩放后 R² ≥ 0.99、峰位逐点相同
   （Re @0.1 Hz、Im @0.2683 Hz）、低频段同号率 100%。Lin-KK 在肘部阶（M=12）
   是 8.49% / acf 0.519，**但这不是地板**：阶数曲线（M=1…45）显示 Rwp 一路降到
   6.84%，acf 却始终 ≥ 0.519（见 `out/linkk_order_probe.md`）。
   文献那条电路的关系：**C1 ≡ C4 ≡ 我们的 `parallel/L=2`**（R-L 星三角，本脚本用两套
   独立方程复核到 3.5e-5），C2 多的一层在本谱退化（参数顶边界），C3 与 `EXT1` 残差
   完全重合。详见 `AZ31-4Li_100_RESID.md` §2.1、§4。

## 边界

- 第 3 条只覆盖 **1 条谱**，跨谱可复现性未验；三个电路同属「集总 RQ + `R∥L`」一大类，
  所以结论是「在这一类电路内换拓扑不改变残差结构」，不是「任何物理模型都不行」。
- **Lin-KK 的 8.49% 不是地板**（订正 2026-11-11）：`lin_kk()` 用肘部准则选阶，
  加阶残差还能降（M=45 → 6.84%，代价 μ→0、元件为负）。所以既不能把它当
  「数据违反 KK 的程度」，也不能由它推出「`acf ≤ 0.50` 结构性不可达」。
  能保留的只是：M=1…45 每一阶 acf 都 ≥ 0.519（最小恰在 M=12）。
- 残差结构的三种可能来源（非稳态 / 仪器夹具偏差 / 频窗截断）在这里**分不开**，
  仍需同状态独立重复谱（R18 未解）。
- 详见 `AZ31-4Li_100_RESID.md` §6。

## 运行前提

- Python 环境同仓库其他脚本（`numpy / scipy`；`resid_structure_cmp.py` 还要能 import
  `bm_core / bm_experiments / bm_kk / bm_real / bm_top`）。
- 数据与产物路径都是 **cwd 相对的 `out/...`**，所以只能在仓库根目录跑；
  脚本本身用 `dirname³(__file__)` 回推仓库根，放在 `docs/round_2026-11-11/` 深度与原
  `evidence/<子目录>/` 相同，`sys.path` 不需要改。
- `AZ31-4Li_100` 的输入谱来自 `out/_m6_samples/z_csv/`，需要先跑过 M6 抽样准备。
