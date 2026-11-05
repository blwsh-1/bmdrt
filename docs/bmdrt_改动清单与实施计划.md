# bmdrt 改动清单与实施计划

> 把前面五份文档的待改项**去重合并**、按依赖与风险重排。
> 上一轮止于"只读审查"，本轮起进入实施。
> 代码根目录：`D:\eis训练数据\bmdrt`（本地 git 仓库，`main`）。

---

## 0. 来源归并

| 来源 | 提出的待改项 | 归并到 |
|------|-------------|-------|
| `算法链路与数据预处理必要性评估.md` | 接 lin_kk 前置体检；低频截断敏感性 + 强制 diffusion 对照 + 漂移则拒绝定 K；DRT 预判扣电感；澄清两套噪声口径 | #1 #2 #3 #7 #P2-3 #P2-4 |
| `低频数据是否用DCT_结论.md` | 不引入 DCT；把 `bm_real.py:884` 的文字提示升级成决策，`diffusion` 纳入模型比较 | #7 #P2-2 |
| `手动拟合实践者建议_与BM-DRT对照评估.md` | KK 作为事前门禁；处理 L/扩散尾误报；KK 不过拒绝定 K、堵 `select_k` 降级路径；全参数相关矩阵；跨噪声重复性；φ 趋势；Ws/Wo 不可分降级 | #1 #2 #3 #4 #6 #7 #P2-1 |
| `EIS-DRT数据预处理_方法与文献.md` | K-K 位置提到"第一步"；电感修正不是删点；低频尾截断；分频段看残差 | #1（设计依据） |
| `幺的科研路_DRT阻抗数据处理方法整理.md` | 高频扣 `jωL` 还原变形圆弧；低频 45° 尾截断；分频段路由 | #1 #7（设计依据） |

**两条明确不做**（历次结论一致）：不引入 DCT；不做手工离群点剔除
（109/117 的"异常"其实是感抗上升沿，不是离群点）。

**核心原则**：KK 是**事前门禁**，其余全是事后诊断；
优先级顺序 = **KK → 电感/低频路由 → 基底与扩散 → K → 参数可辨识性**。

---

## 1. 本轮实施范围（P0 + P1）

### #1 新增 `bm_kk.py`：Lin-KK 分级门禁　【P0】

- **现状**：`bmdrt/*.py` 零实现；可复用 `D:\eis训练数据\AutoECM\validate_fit.py:38-77`，
  但该模块依赖 `ext_fit` / `fit_store`，且不在本仓库内 → **不能跨目录 import，必须自包含移植**。
- **改法**：
  - 移植 `lin_kk()` + `_eval_linkk()`（保留绕开 numpy 2.x 两个 bug 的处理）。
  - 新增 `kk_check(f, Z, include_L, ...)`，输出 `status ∈ {ok, warn, fail, unavailable}`：
    - 先按原谱做 Lin-KK（`μ<0.85` 收敛判据）；
    - **分频段归因**：把相对残差按 `log f` 分高/中/低三段，避免把 L 与扩散尾
      误判成"数据不自洽"；
    - **高频感抗**：`include_L` 为真时用最高若干点的 `Im Z / ω` 稳健估计 `L`，
      扣掉 `jωL` 后复验；
    - **低频扩散尾**：按相位接近 −45° 自动定截断点 `f_cut`，截断后复验；
    - 只有"扣 L / 截尾之后**中段或全谱**残差仍显著超阈"才判 `fail`；
      端部残差可归因于 L/扩散时降为 `warn` 并写明归因。
  - `impedance` 未安装 / Lin-KK 抛异常 / 频点 <12 → `unavailable`，**只 warn 不中断**。
- **验收**：`examples/example_caseE.csv`（无 L）→ `ok`；
  `example_caseF.csv`（含 L）→ 不因感抗尾被判 `fail`；
  真实 `csv_2e/2EMC1-1_0005mA.csv` → 给出明确归因；
  构造一条漂移/非因果的合成谱 → `fail`。

> **【实施中修正了两处前提错误，已落地并验证】**
>
> 1. **不能扣 `jωL` 再跑 Lin-KK。** 读 `impedance.validation.fit_linKK` 源码确认：
>    其设计矩阵显式含 `a_im[:, -1] = w/|Z|`，注释为 "series inductance to
>    capture inevitable contributions from the measurement system"，
>    返回 `elements=[R0,R1..RM,L]`。即 **Lin-KK 本来就带自由串联电感**，
>    预扣 L 是重复计入。实测真实 2E 谱高频段残差 9.60% → 10.78%（更差），
>    截断+扣 L 后 9.60% → 11.68%。`estimate_L()` 保留为诊断量，
>    留给 P2-4 的「DRT 预判扣电感」（Tikhonov DRT 才是真表示不了 `jωL`）。
> 2. **低频扩散尾用 `add_cap=True` 复验，不截断丢点。** `get_tc_distribution()`
>    的 τ 只覆盖测量窗口，表示不了窗口外继续发散的 Warburg 尾；官方
>    `add_cap` 正是给"无低频截距"数据的解法。实测 2E 谱低频残差
>    11.44% → 4.72% 且**一个点都不丢**；而截掉最低 25% 频点反使中段
>    1.92% → 4.33%、高频 9.60% → 14.83%。`f_cut` 只作诊断记录。
>    且**只允许 add_cap 放宽低频段**，中段判据一律用 raw，
>    避免新增自由度把真实的中段不自洽吸收掉。
>
> **【新增第 3 条判据：残差结构】** 加噪谱（如 `example_caseG_wo_only.csv`）
> 中段残差可达 8.53%，单靠 5% 阈值会被误杀。沿用 `AutoECM/validate_fit.py`
> 的滞后-1 自相关思路，只有**实部与虚部中段残差同时系统性**（ac1>0.5）
> 或中段 >2×tol 才判 `fail`；否则降为 `warn` 并注明"噪声 / Lin-KK 表达力不足，
> 不能判定谱不自洽"。标定：加噪 Wo-only ac1=+0.71/+0.09 → `warn`；
> 实部鼓包 0.20R ac1=+0.72/+0.96 → `fail`；漂移 0.10R ac1=+0.83/+0.97 → `fail`。
> 另注意 μ 在严重违规谱上仍 ≈0.80「收敛」，只能当次要信号。

### #2 `bm_real.run()` 接入门禁 + 拒绝定 K　【P0】

- **现状**：`run()` 在 `load_spectrum()` 后直接进 `prescreen_stage()`（`bm_real.py:1226-1237`）。
- **改法**：
  - `run()` 增参 `kk_mode="gate"`（三档 `gate | warn | off`）。
  - 门禁在**读数据之后、DRT 预判之前**执行，打印一段 KK 报告。
  - `gate` + `fail`：**跳过整条拟合链**，只写一份
    `real_<stem>.json`（`fit_performed=false`、`kk` 全量字段、`reason`），
    打印「数据不自洽，拒绝定 K」，返回 `None`。
  - `warn` / `ok` / `unavailable`：照常拟合，但把 KK 结果挂到 `best` 上，
    由 #3 #8 写进报告与 JSON。
- **风险**：批量流程 `bmdrt.py::_run_one` 用 `real_<stem>.json` 是否存在判成功
  （`bmdrt.py:232-236`），不写 JSON 会被记成"跑完但没结果"→ 由 #4 配套处理。

### #3 `select_k()` 堵住降级路径　【P0】

- **现状**：`bm_real.py:752` 的
  `pool = ok or usable or list(res_list)` 在**全部被硬剔除**时仍返回"相对最好"，
  恰好把"数据不自洽 / 模型全不可信"包装成「最优 K」。
- **改法**：
  - 保留 `best` 的返回（下游 `report/plot_real/save_out` 需要一个 dict，不能抛异常），
    但新增 `best["selection_status"]`：
    `ok`（正常）/ `degraded`（仅降到"可用"档）/ `rejected`（无任何可用模型）。
  - `rejected` 时 `note` 首条写明"**不输出最优 K 的结论**"，并把 `elpd` 排名
    明确标成"仅相对排序、无物理意义"。
  - `run()` 打印时按 `selection_status` 改措辞；`save_out()` 落
    `k_selection_status` 字段。
- **验收**：构造全部 grade=0 的 `res_list`，返回 `selection_status="rejected"`
  且 note 含拒绝措辞；正常样例仍为 `ok`，结果不变。

### #4 `bmdrt.py` 批量流程兼容 KK 拒绝　【P0】

- **改法**：`_run_one` 在 `js.exists()` 后读 `fit_performed`：
  `False` → `rec["ok"]=True`（流程正常完成，不是崩溃）、`rec["kk_rejected"]=True`，
  并写进 `_write_meta`；汇总表新增该列。
- **验收**：批量处理一条被拒谱不产生 `[FAIL]`，也不被当成成功拟合统计。

### #5 配置、CLI、依赖　【P0】

- `config.json` 增 `kk` 段：`{"mode": "gate", "c": 0.85, "max_M": 50,
  "res_tol": 0.05, "auto_L": true, "auto_tail": true}`。
- `bmdrt.py` 的 `_apply_cli()` / `opt` 解析增 `--kk gate|warn|off`；
  `bm_real.py:main()` 的 argparse 同步。
- `requirements.txt`：`impedance>=1.7` 由"可选"改为**建议必装**
  （缺失时 `unavailable` 降级，不硬失败）。
- `README.md` 增「KK 门禁」一节与开关说明。

### #6 新增 `bm_corr.py`：全参数相关矩阵　【P1】

- **现状**：只算了 `lnR↔lnτ`（`bm_real.py:856`）与 `lnR_W↔lnτ_D`（`bm_real.py:290`），
  没有全参数矩阵，也没有 `|corr|>0.95` 警告。
- **改法**：从 `res["s"]`（后验样本，键见 `bm_core.py:227-286`；
  `pi`/`tau` 由 `samples_to_np` 派生）构造逐样本参数向量：
  `lnR_inf, lnW, {lnτ_k, φ_k, ζ_k, v_k}, lnL, lns_re, lns_im, {lnRw, lnRwo, lnτ_D}`；
  **按 `f0` 降序对齐**解决标签交换；对非正/常数列自动跳过。
  返回 `names, corr, high_pairs(|corr|>0.95), ci90`（各参数 5/95 分位）。
  纯后处理，**不动似然、不重跑 NUTS**。
- **验收**：对一条 `*_samples.npz` 能直接算；`bm_bench.py` C 组（间距 1.5×）
  应出现高相关对。

### #7 Ws/Wo「不可分」降级　【P1】

- **依据**：`bm_diff.py:39-42,503-511` 已证 `W_open/W_short` 在 `ωτ_D≫1`
  都退化为 `A_W(jω)^(-1/2)`；本批实测 `f_min=1.002 Hz`。
- **改法**：新增 `diffusion_resolution(res, f)`：
  - 由 `tauD` 后验中位数算 `f_D = 1/(2πτ_D)`；若 `f_min > f_D`（转折在窗口之下）
    或 `corr(lnR_W,lnτ_D)>0.95`、`ridge_ratio<0.5`、`p_diff_small` 高 → `resolvable=False`；
  - 报告写「频率窗口不足以区分有限长扩散，仅可报告组合量 `A_W` / `C_D`」，
    并**不再把 `diffusion` 当作已被选出的物理结论**；
  - 结果落 `diffusion_resolution` 字段到 JSON。
- **注意**：这是把上一轮"把 diffusion 纳入 select_k 比较维度"的**前置条件**补上；
  真正跑 `(K, none/Ws/Wo)` 的三倍扫描见 #P2-2，本轮不做。

### #8 报告与 JSON 汇出　【P1】

- `report()` 开头打印 KK 段；`judge()` 首条插入 KK 门禁结论；
  `save_out()` 增 `kk`、`k_selection_status`、`param_corr`、`diffusion_resolution` 字段。
- 三者都写成"有则打印 / 缺失则跳过"，保证旧结果文件仍能读。

---

## 2. 本轮**不做**（P2，列清楚以免遗漏）

| # | 事项 | 暂缓原因 |
|---|------|---------|
| P2-1 | 跨噪声实现的参数重复性脚本（`tools/param_repeat.py`）：对 A/B/C/E/F/G 组各跑 `seed=0..19`，报 `mean±sd`、区间覆盖率、真值偏差 | 需跑大量 NUTS，耗时；纯新增脚本，不阻塞 P0/P1 |
| P2-2 | 把 `diffusion` 正式纳入 `select_k` 比较维度（`(K,none)/(K,Ws)/(K,Wo)` 同一套逐点 LOO） | 要把 `run()` 的扫描从「扫 K」改成「扫 (K, diffusion) 组合」，拟合格数 ×3（117 条谱约 3 倍机时），需单独一轮 + 重跑验证；先由 #7 把"不可分"标出来更稳妥 |
| P2-3 | `tools/phi_trend.py`：`φ`（≡CPE-n）按电流密度/循环数连成趋势 | 后处理脚本，等 P0/P1 稳定后再做 |
| P2-4 | `prescreen_stage()` 给 DRT 预判扣电感（贝叶斯似然不动） | 与 #1 的"扣 L 复验"共用估计器，待 #1 落地后单独验证 DRT 峰数变化 |
| P2-5 | 两套噪声口径（DRT 用相对 0.5%·|Z|、似然用绝对 σ）文档澄清 | 文档任务 |

---

## 3. 兼容性与风险约束（改动时逐条守住）

1. **不破坏既有一条谱的流程**：新增逻辑一律"有则做、缺则跳过"；
   `impedance` 缺失、Lin-KK 抛异常、样本键缺失都不得让 `run()` 崩。
2. **扁平布局不变**：`bmdrt.py` 靠 `cmd_*.py` 文件名自动发现子命令、
   靠 `Path(__file__).parent` 定位 `config.json`，不能改目录结构。
3. **似然与采样口径不动**：#6 #7 是纯后处理，#1 #2 只加门禁，
   不碰 `bm_core.py` 的 `model_z` / `make_model` / `fit`。
4. **批量流程的错误判定**：KK 拒绝 ≠ 运行失败（#4），否则 117 条谱汇总会误报。
5. **默认档位**：`kk.mode` 默认 `gate`，但判据必须保守——
   含感抗尾、低频扩散尾的正常谱只应 `ok/warn`，只有扣 L / 截尾后
   中段仍不自洽才 `fail`；提供 `--kk warn|off` 逃生开关。
6. 每改完一项跑一次 `bmdrt.py check` 与 `drt.py` 冒烟，最后整体验证 + git 提交。

---

## 4. 验收总表

| 检查 | 期望 |
|------|------|
| `python bmdrt.py check` | 18 项自检全通过（与改动前一致） |
| `python drt.py examples/example_caseE.csv` | exit 0 |
| `python bmdrt.py config` | 输出含 `kk` 段且值正确 |
| `python bm_kk.py examples/example_caseE.csv` | `status=ok`，`μ<0.85` |
| `python bm_kk.py examples/example_caseF.csv` | `status=ok`，归因写明 L 且**不扣 `jωL`** |
| `python bm_kk.py examples/example_caseG_wo_only.csv` | `status=warn`（加噪谱，不被误杀） |
| `python bm_kk.py EISmeasurements/csv_2e/2EMC1-1_0005mA.csv` | `status=warn`，低频归因写明 `add_cap` 11.4%→4.7% |
| 合成实部鼓包 0.20R / 0.40R、低频漂移 0.10R | `status=fail` |
| `python bm_corr.py out/real_<名>_samples.npz` | 输出相关矩阵与高相关对 |
| 合成一条非因果谱 | `kk.status=fail` → gate 模式写 `fit_performed=false` JSON，批量不报 FAIL |
| `git diff --stat` | 只涉及计划内文件；工作区提交后干净 |

---

## 5. 实施结果（#1–#8 已完成）

### 5.1 逐项状态

| # | 事项 | 状态 | 落地位置 |
|---|------|------|---------|
| 1 | `bm_kk.py` Lin-KK 分级门禁 | 完成 | 新增 `bm_kk.py`（496 行） |
| 2 | `bm_real.run()` 接入门禁 + 拒绝定 K | 完成 | `bm_real.py` ⓪ 段、`_write_kk_reject()` |
| 3 | `select_k()` 堵住降级路径 | 完成 | `selection_status ∈ {ok,degraded,rejected,not_compared,rejected_by_kk}` |
| 4 | 批量流程兼容 KK 拒绝 | 完成 | `bmdrt.py` `_enrich/_verdict/write_summary/cmd_auto/cmd_watch`；`batch_2e.py` `_is_kk_reject()`；`cmd_status.py`；`report_2e.py` |
| 5 | 配置、CLI、依赖、README | 完成 | `config.json` `kk`+`diag` 段、`--kk`、`requirements.txt`、`README.md` |
| 6 | `bm_corr.py` 全参数相关矩阵 | 完成 | 新增 `bm_corr.py`（含 CLI，可对 npz 重算） |
| 7 | Ws/Wo 不可分降级 | 完成 | `bm_diff.diffusion_resolution()` |
| 8 | 报告与 JSON 汇出 | 完成 | `report()`/`judge()`/`save_out()` |

### 5.2 实施中发现并修正的**前提错误**（计划里写错了）

1. **不能"扣 `jωL` 再跑 Lin-KK"。** 计划原写"电感修正后再检验"。
   实际 `impedance.validation.fit_linKK` 的设计矩阵已含自由串联电感列
   （`a_im[:, -1] = w/|Z|`，注释写明 `series inductance to capture inevitable
   contributions from the measurement system`）。再扣一次是重复计入。
   实测真实 2E 谱：原谱高/中/低 = 9.60/1.92/11.44 %，扣 `jωL` 后
   10.78/2.20/11.44 %——**高频残差反而变差**。现改为：`estimate_L()` 只作诊断量。
2. **不能"截断低频扩散尾再复验"。** 实测：截尾把中段从 1.92% 推到 4.33%、
   高频从 9.60% 推到 14.83%（丢点后 Lin-KK 少了约束）。改用
   `fit_linKK(..., add_cap=True)` 在同一频率网格复验，不丢点：
   低频 11.44% → 4.72%，中/高频同时改善。`f_cut` 只作诊断，不真正截断。
3. **中段残差超阈不能直接判 fail。** `example_caseG_wo_only.csv` 是加噪但合法的
   扩散谱，中段 8.53% > 5%，单靠 RMS 会误杀。新增残差结构判据
   （实/虚部 lag-1 自相关同时 > 0.5 才算系统性）。标定：
   caseG `ac1=+0.71/+0.09` → warn；鼓包 `+0.72/+0.96`、漂移 `+0.83/+0.97` → fail。

### 5.3 实施中新发现的问题（计划里没写）

1. **`W` 不能和分量 `w_k` 同时进相关矩阵。** `W = Σ_k w_k`，K=1 时
   `w_1 ≡ W` 逐样本完全相等，矩阵里会出现恒等于 `+1.00` 的**伪相关**。
   这是参数化的必然结果，不是物理简并。现改为：`W` 不进矩阵，
   `R_inf ↔ W` 的补偿关系用派生量 `_degenerate_wtot()` 单独算。
2. **`|Wo−Ws|/|Ws|` 不是好的"可分辨性"指标。** 低频端 `Wo → 1/(jωC)` 发散、
   `Ws → R_W` 有限，实测 `τ_D=1e-3 s` 时该比值到 **15890%**，没有信息量。
   改为 `|Wo−Ws| / max(|Wo|,|Ws|)`，压到 `[0,1]`：0 = 完全同形，1 = 完全不像。
3. **`f_D` 落在窗口下方但边缘仍有 >tol 差异**（实测 `τ_D=1 s`、`f_min=1 Hz`，
   边缘差 10.9%）。拐点没进窗口时不能宣称可分辨——`τ_D` 基本只由高频尾约束，
   差异只体现在最低几个点，抗噪极差。单列 `indistinguishable` 并写明
   `f_D` 在窗口下方多少 decade。
4. **`batch_2e.py` 会把 KK 拒绝记成 `fail(rc=0)`。** 原判据
   `status = "ok" if (js.exists() and png.exists())`——拒绝记录没有 png，
   于是整批退出码变红，而实际什么都没坏。新增 `_is_kk_reject()` 单独识别，
   并让"已处理完"的跳过条件也认它（否则每次重跑都白跑一遍）。

### 5.4 验收执行结果

| 检查 | 结果 |
|------|------|
| `python bmdrt.py check` | **18/18 PASS** |
| `python bmdrt.py diff --fast` | 全部通过（B1/B2/C1/Wo/B2-ident/DRT） |
| `python bm_kk.py` 8 个样例 + 真实 2E 谱 | caseE/F ok；caseG/2E warn；鼓包/漂移 fail；缺包/6 点 unavailable |
| `bm_corr.py` 真实 npz（K=1 / K=2） | 正确给出 R–τ 脊线 `+0.99`、`τ↔φ -0.94`；`ζ` 钉死时列入 fixed |
| `diffusion_resolution` 5 组 τ_D | resolvable / indistinguishable 分级符合预期；JSON 严格序列化通过 |
| 端到端 `bmdrt.py auto example_caseE --quick --krange 1,1` | 19 s，JSON 含 `kk/param_corr/diffusion_resolution/k_selection_status`，5.5 KB |
| 端到端 KK 拒绝（合成漂移谱） | 退出码 **0**，汇总写「拟合成功 0，KK 拒绝 1」，`status` 显示「KK 拒绝」 |
| `report_2e.py` 新格式汇总表 | 正确输出「拟合成功 115/117（另有 2 条被 K-K 门禁拒绝）」 |
