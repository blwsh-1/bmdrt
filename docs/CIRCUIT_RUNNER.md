# circuit_runner.py：registry 驱动的候选调度层（2026-11-15）

> **性质**：研究/比较接口，**不是准入通道，也不是生产默认路径**。
> 不改 `bm_*.py`、不改任何阈值、不改 `bm_real.fit_one` 的默认族、不接 `bmdrt.py auto`。
> 边界照 `CIRCUIT_TOPOLOGY_ADMISSION.md` §8.1：“登记合法 ≠ 准入通过”。

## 1. 它补的是哪一层

三层架构里缺的两块，本轮只做了第一块：

| 模块 | 状态 | 职责 |
|---|---|---|
| `circuit_registry.py` | 已有（0.7.0，`b2675c8255bea716`） | 候选能否被**表达、签发身份、被复核** |
| **`circuit_runner.py`** | **本轮新增** | 候选集 → 预检 → 逐候选拟合 → 观测落盘（缓存/续跑） |
| `circuit_evaluator.py` | **已有**（2026-11-16，见 `CIRCUIT_EVALUATOR.md`） | 从观测数组**推导**状态（E1–E7；两个 disputed 参数待裁） |

registry 里 `evidence_block` 是**校验器**不是推导器：状态要由调用方传入，它只负责
“你声称 `identified` 但 `out_of_sample_valid=False` 就当场报错”。推导层现由
`circuit_evaluator.py` 承担（规则与争议项见 `CIRCUIT_EVALUATOR.md`）。

## 2. 三条设计约束（写进代码注释，不是文档口号）

1. **统一误差定义**：`rms_rel` / `acf` 由 `metrics()` 从 `Z_model − Z_data` 现算，
   与 §14 `_metrics` 逐字同式；**不读各族自报的数**。族自报口径不同
   （voigt 走 `bm_real` 的 band 中位、nested 根本没算），直接比会得出假结论。
2. **`elpd` 一律 `valid_for_comparison=False`**：`sigma=None + sigma_floor` 下
   `sigma_from_data=True`，`elpd_loo` 不是 `p(Z_j|Z_{-j})`（R21），这个事实跟数字一起走。
3. **不合成总分、不排序**：输出是逐候选的 `observation_block`，排序与状态判定留给
   evaluator。

## 3. 整合了什么（含用户指出的三组腐蚀/涂层数据）

`测试数据/` 三组合成谱的真值拓扑（`NOISE_ACF_TRUTH.md` §1 的数值断言）：

| key | 组 | 真值拓扑 | 候选 id | 谱数 |
|---|---|---|---|---|
| `coating` | 涂层 | `R1-[P2,R3-[P4,R5-P6]]` = nested **L=2 / capacitive** | `C-231fe53771` | 1000 |
| `passivation` | 钝化 | `[P1,P2-R3]-R4` = nested **L=1 / capacitive** | `C-93399cbbe9` | 1000 |
| `corrosion` | 腐蚀 | `R1-[R2,P3]` = ZARC = **voigt K=1** | `C-3d92776406` | 1000 |

**关键点：涂层与钝化的最内层是容性 CPE，不是 `R∥L`。** 所以它们必须用
`branch="capacitive"` 才落在真值拓扑上；默认 `branches=("parallel",)` 里根本没有这两个候选
（自检里有一条断言就是“涂层/钝化的真值候选与默认候选集不相交”）。

新增能力：
- `build_candidates(branches=...)` / `--branches parallel,capacitive`（`branch`→槽位元件映射
  `parallel→rl_parallel`、`series→rl_series`、`capacitive→cpe`，全部走 registry 一致性校验）；
- `DATASETS` / `--datasets` / `truth_candidate_for()`：三组数据的真值拓扑与文件定位；
- 缓存键 `(candidate_id, data_sha256, seed, budget_profile)`，`observations.jsonl` append-only。

**2026-11-20 更新（扩散与 L 已接入，吸附仍锁）**：
- `w_inf` / `w_open` / `w_short` 已 `wired=True` 并挂在 voigt/nested/adr 三族的
  `diffusion` 槽位（`min_count=0`，默认关）；`build_candidates(include_diffusion=True)`
  才生成，`compare --diffusion` 是唯一显式入口。`w_short` 另补了 `bm_core` 的采样分支。
- `include_L`（顶层串联 `jωL`）三族同口径：registry 的 `series_inductor` 槽位 +
  `_check_inductor_slot` 统一校验；**同批候选的 `include_L` 必须全同**
  （`_assert_single_L`），否则两个模型空间混比。
- `adsorption` 仍 `wired=False`：内核实现挂**顶层串联**，而 §9.4.12 的裁决是
  **只允许第二并联节点**，接点冲突未裁 ⇒ 有意锁，不是漏接。

**仍未接入**：只有 `adsorption`（理由见上）。扩散/L 的接入 ≠ 准入：§4 三条准入
一条未满足，全部候选一律 research、默认关。

## 4. 实测（`quick` 档、seed 0、单条谱，**探索性，不是证据**）

| 组 | 候选 | rms_rel | acf | 备注 |
|---|---|---|---|---|
| 腐蚀 | voigt K=1（真值） | **19.3%** | 0.224 | 见 §5 的口径问题 |
| 腐蚀 | nested L=1 / capacitive | 20.2% | 0.177 | 非真值拓扑 |
| 涂层 | nested L=1 / capacitive | 816% | 0.927 | 层数错 + 初值塌陷 |
| AZ31-8Li_50 | voigt K=1/K=2 | — | — | 预检 `excluded`（高频感抗） |
| AZ31-8Li_50 | nested L=1/L=2 | 5.99 / 5.94% | 0.326 / 0.318 | `unknown`（nested 无确定性预检） |

## 5. 两条必须记账的发现

**(a) 统一噪声口径与这批数据不匹配。** `bm_select` 的跨族口径是
`sigma=None + sigma_floor`（R3），在这批 `|Z|` 跨 3 个量级（2.9e3–1.0e6）的合成谱上，
自由 σ 被 `LogNormal(log 0.01·zmax, 1)` 拉到 ~1e4，**高频点几乎不参与似然**：
腐蚀真值 voigt K=1 的 `R_inf` 拟合到 4594 Ω（真值 1000 Ω），相对 rms **19–20%**。
对照 `NOISE_ACF_TRUTH.md` §1 同一谱在**逐点 2%** 口径下是 **2.85%**。
⇒ 差异来自噪声口径，不是适配器错误。要用这三组做回归，必须给 runner 加
“逐点 σ”口径分支（`bm_top.fit_top` 支持 `sigma=(2,n)`；voigt 侧 `bm_select.fit_voigt`
目前不暴露该通道），**且这条口径必须与 `sigma=None` 的结果分开报**，否则正是
你指出的“噪声模型不一致却直接比 ELPD”。

**(b) 两处“离病因很远”的隐性契约（都已修，并各加了回归断言）**
1. `bm_real.grade()` 读 `r["tds"]["frac_cap"]`，而 `tds` 来自
   `E.tree_depth_stats(mc, max_tree_depth)`，**必须有 `extra_fields=("num_steps",)` 才算得出来**。
   预算档漏这一项不会报“缺参数”，而是在 `grade` 里抛 `TypeError: 'NoneType' object is not
   subscriptable`。`bm_select.QUICK` 本来就带，runner 必须对齐。
2. `observation_block` 的 `fit_observation` 只有 `rms_rel / acf / interpretation /
   sampling_diagnostics / unavailable`；`rhat_max` 在 `sampling_diagnostics` 下。
   按顶层取会 KeyError，且发生在**记录已落盘之后**，表现为“落盘成功但行汇总标 failed”。

## 6. 用法

```
python circuit_runner.py --list                       # registry 内容 + 是否可调度
python circuit_runner.py --datasets                   # 三组真值拓扑 + 文件定位
python circuit_runner.py --plan  <csv>                # 候选集 + 三态预检，不拟合
python circuit_runner.py --run   <csv> --budget quick --branches parallel,capacitive
python circuit_runner.py --selftest                   # 38/38
```

产物：`out/circuit_runner/<stem>/` 下 `plan.json` / `screening.json` /
`observations.jsonl` / `run_meta.json`。`observations.jsonl` 按 cache_key 续跑。

## 7. 下一步（按依赖顺序，均需单独授权）

1. ~~`circuit_evaluator.py`~~：已完成（`CIRCUIT_EVALUATOR.md`，自检 43/43）；
   剩 `tie_pp=1pp` / `rms_accept=10%` 两个**待裁决**的争议参数。
2. **逐点 σ 口径分支**（§5a）：让三组数据在真值拓扑上回到 ~3% 量级，且与自由 σ 结果分母报。
3. **扩散/吸附槽位**：`bm_core.model_z` 已支持 `diffusion="Ws"/"Wo"/"both"`、`bm_diff` 正演齐全，
   缺的只是拓扑槽位定义。
4. **`bm_select` 与 registry 合流**：现在两套“候选”概念并存，registry 这套至今只在
   `circuit_runner` 里被调用过，`bm_select` 仍是硬编码两族比较器。

**未动**：`bm_*.py`、M-b 阈值、§13 `0/3`、§14 `4/14`、§17 `0/10`、`bmdrt.py` 接入、身份链。
