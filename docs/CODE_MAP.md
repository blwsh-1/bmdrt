# 代码地图：分层、产品候选与软件化阻塞项

> 生成依据：`tools/repo_map.py`（AST 静态扫描，不 import 项目模块、不跑拟合）。
> 机器可读版本 `docs/code_map.json`；新增模块漏分类时 `python tools/repo_map.py --check` 会报 FAIL。
> 本文件是**分类与判断**，不是重构记录：本轮**没有移动、没有改名、没有改任何模块内容**。

## 1. 总量

| 项 | 值 |
|---|---|
| 根目录 Python 模块 | **62** |
| 其中测试 | 19 |
| 根目录总行数 | **41,119** |
| 带 `__main__` 的入口 | 57 |
| 纯库（无 `__main__`） | 5：`bm_circuit` / `circuit_registry` / `cmd_diff` / `cmd_status` / `exec_identity` |
| 被依赖最多的 | `bm_core`(17) → `bm_real`(13) → `bm_experiments`(8) → `bm_ad`(6)=`bm_top`(6)=`bmdrt`(6)=`exec_identity`(6) |
| 目录 | `evidence/` 96 个脚本、`docs/` 63 个脚本 —— 全部是**研究/证据**，不进产品 |

## 2. 七层分类

层次是**逻辑分层**，不代表当前代码已经遵守（见 §4 的 11 处反向依赖）。

### L0 数值内核（8）——纯数学，无文件 IO

| 模块 | 职责 | 产品 |
|---|---|---|
| `bm_core.py` | 贝叶斯混合（stick-breaking + HN）voigt 族模型与 `fit` | **core** |
| `bm_top.py` | 嵌套并联拓扑 `z_nested` 正演 + `fit_top` 独立采样路径 | **core** |
| `bm_diff.py` | Warburg 族扩散元件与 HN↔Warburg 退化诊断 | **core** |
| `bm_ws.py` | Ws 判据候选量（被 `bm_real.diffusion_gate` 消费） | **core** |
| `bm_ad.py` | 低频吸附电感元件 `Z_ad` | opt |
| `drt.py` | DRT 粗解（只报过程数 / 时间常数 / 电阻） | opt |
| `circuit_registry.py` | 候选电路注册表（schema 承载能力） | opt |
| `circuit_runner.py` | registry 驱动的候选调度：预检→逐候选拟合→观测落盘（研究接口，不改默认族、不接 `auto`） | opt |
| `circuit_evaluator.py` | 候选观测 → evidence 状态的推导层（E1–E7；争议规则显式标 `disputed`） | opt |
| `bm_figure_unified.py` | 统一拟合图：一条谱 + 任意多条候选曲线 → 同一张 6 子图（只依赖 f/Z/curves） | opt |
| `cmd_compare.py` | `bmdrt compare`：多族候选自动并列 → 统一状态 + 统一图（默认 standard / 逐点 σ） | opt |
| `bm_circuit.py` | 等效电路图绘制 | no |

### L1 数据与 IO（7）

| 模块 | 职责 | 产品 |
|---|---|---|
| `bm_real.py` | 真实数据主程序：`load_spectrum` / Lin-KK 门禁 / 选 K / 报告 —— **3123 行、职责过载，产品化第一刀要切它** | **core** |
| `bm_ref_rwp.py` | 参考性 Rwp（独立 LS 重拟合，只作跨工具对照） | opt |
| `convert_2e.py` / `convert_3e.py` | Gamry xlsx → 三列 CSV | opt |
| `fit_store.py` | `drt` 命令行兼容读取器 | no |
| `batch_2e.py` / `report_2e.py` | 2E 批处理调度与交付说明 | no |

### L2 分析与诊断（11）

| 模块 | 职责 | 产品 |
|---|---|---|
| `bm_kk.py` | 拟合前 Lin-KK 门禁（gate / warn / off） | **core** |
| `bm_experiments.py` | 实验脚本分发；`acf_max` 等验收指标也在这里 | opt |
| `bm_identifiability.py` | 统一可辨识性报告（六类诊断 / 三态） | opt |
| `bm_corr.py` | 全参数后验相关矩阵 + 置信区间 | opt |
| `bm_bench.py` / `bm_check.py` | 合成 benchmark / 端到端 PASS-FAIL | opt |
| `bm_evidence.py` / `bm_grade.py` | 证据卡生成 / 资格分档 A-B-C（只贴标签） | opt |
| `bm_ident.py` / `bm_top_gen.py` / `bm_zhit.py` | 峰位归因 / 独立生成器 / Z-HIT 原型（已否决） | no |

### L3 结果与呈现（3）

| 模块 | 职责 | 产品 |
|---|---|---|
| `bm_result.py` | `AnalysisResult` 唯一结果来源与导出包 | **core** |
| `bm_dash.py` | 自包含 HTML 交互面板（零运行时依赖） | **core** |
| `bm_figure.py` | 出版级静态图（与面板共用同一份数据） | **core** |

### L4 编排与入口（6）

| 模块 | 职责 | 产品 |
|---|---|---|
| `bmdrt.py` | 唯一 CLI 入口；`cmd_*.py` 里写 `run(argv)` 自动注册子命令 | **core** |
| `bm_select.py` | M4 跨族选择器骨架（voigt vs nested） | opt |
| `nested_fit.py` | nested 独立研究接口（**未接主流程**） | opt |
| `cmd_status.py` | 子命令：查看 `out/` 里跑过的结果 | opt |
| `cmd_diff.py` | 子命令：扩散项数学回归 | no |
| `sweep_sigma_floor.py` | σ 噪声地板敏感性扫描 | no |

### L5 执行身份与审计（8）

`exec_identity.py` / `capture_contract.py` / `sampler_provenance.py` / `sampler_capture.py` /
`budget_params.py` / `budget_capture_adapter.py` / `patch_bundle.py` / `patch_bundle_io.py`

全部 `product=no`。它们的价值是**可复现性与审计**（`run_id` / `workflow_id` / 预算来源五键 /
补丁包重建），不是拟合能力。产品若要「可复现认证」再单独接，且**必须保持不变式**：
`upstream_origin_captured=False` 时不得产出有效号（当前真实执行仍是 `unconfirmed` 零号）。

### L6 测试（19）

`test_exec_identity.py`(1293 行) / `test_circuit_registry.py`(1262) / `test_contract.py`(1222) /
`test_sampler_provenance.py`(927) / `test_budget_params.py` / `test_capture_contract.py` /
`test_sampler_capture.py` / `test_patch_bundle.py` / `test_identity_p0.py` / `test_ecm.py` /
`test_ad_gate.py` / `test_ad_gate_synth.py` / `test_bm_ad_e2e.py` / `test_bm_ad_sampler.py` /
`test_bm_ad_select.py` / `test_drt_signed.py` / `test_audit_findings.py` / `test_include_L.py` /
`test_seed.py`

## 3. 产品候选与依赖闭包

**产品核心 10 个**：`bm_core` `bm_top` `bm_diff` `bm_ws` `bm_real` `bm_kk` `bm_result`
`bm_dash` `bm_figure` `bmdrt`。加上依赖，**闭包 = 26 个模块**。

闭包里**不属于核心**的 16 个：`bm_ad` `bm_bench` `bm_check` `bm_circuit` `bm_corr`
`bm_experiments` `bm_grade` `bm_ident` `bm_ref_rwp` `bm_select` `bm_top_gen` `drt` `fit_store`
`sampler_capture` `sampler_provenance` `budget_capture_adapter`。

其中**最后三个是「不该进产品却被拉进来」的**：`bm_core` 与 `bm_top` 在生产构造点直接
`import sampler_capture`（审计层）。这是软件化必须解掉的耦合，见 §4-⑤。

## 4. 五个软件化阻塞项（按严重度）

**① 路径全部相对 cwd。** 14 个模块硬编码 `"out"/"results"/"figures"`，`bm_real` 默认
`inbox/`。一旦打包成可执行文件或从别的目录启动，全部失效。→ 需要一层 `Paths` 配置
（数据目录、输出目录、缓存目录），默认值保持兼容。

**② 无包结构。** 62 个模块平铺在仓库根，`bm_` 前缀命名，没有 `pyproject.toml`、
没有 `__init__.py`，**不能 `pip install`、不能被外部 import**。22 个模块顶部还有
`sys.path.insert(...)`，把「在仓库根运行」写死成前提。

**③ `bm_real.py` 职责过载（3123 行）。** 它同时是：数据读取、Lin-KK 门禁调用、
K 选择、机制选择、结果组装、报告输出。产品要复用它，只能整块拿走或先拆。
建议拆成 `io`（读谱/写产物）/ `gate`（KK + 结构检查）/ `select`（K 与机制）/ `report`。

**④ 全局可变状态。** `bm_real.KW` 是模块级 dict，运行时被 CLI 覆盖；`config.json` 是第二处
默认来源。同进程内跑两个不同配置的任务会互相污染（`run_e2.py` 那类批处理只能靠起子进程绕开）。
→ 需要显式 `Config` 对象贯穿调用，`KW` 只作向后兼容的默认值。

**⑤ L0 反向依赖 L5（最硬的一条）。** `bm_core.py` / `bm_top.py` 在构造点 `import sampler_capture`，
使**数值内核依赖审计层**。产品若不带审计层，这两个模块 import 就失败；若带，则把
1000+ 行的身份契约绑进拟合引擎。→ 需要依赖倒置：内核只暴露 `on_sample_construct` 钩子，
捕获实现由调用方注入；默认不注册（等价于今天的「不接 runner」）。

另外 11 处反向依赖是同一病根的不同表现，已由 `repo_map.py` 机械列出（`layer_violations`）：
`bm_ad→bm_kk`、`bm_diff→bm_grade`、`bm_real→{bm_kk,bm_corr,bm_grade,bm_experiments}`、
`bm_check→bm_select`、`drt→fit_store`、`sweep_sigma_floor→test_ecm`。

## 5. 目标结构（提案，未执行）

```
bmdrt_app/                  # 新产品包；不动根目录研究树
├── pyproject.toml          # 可 pip install，console_scripts: bmdrt = bmdrt_app.cli:main
├── bmdrt_app/
│   ├── kernel/             # 由 L0 抽出的纯数值层（bm_core / bm_top / bm_diff / bm_ws）
│   ├── io/                 # load_spectrum、转换器、产物写入（含 Paths）
│   ├── pipeline/           # gate(bm_kk) → fit → select → result
│   ├── render/             # bm_result / bm_figure / bm_dash
│   ├── audit/              # 可选：L5，靠钩子注入，默认关闭
│   └── cli/                # bmdrt.py 的子命令注册（cmd_* 约定保留）
└── tests/                  # 19 个测试搬过来 + 新增契约测试
```

**迁移顺序**（每步都能单独回滚）：`kernel` → `io`（含 Paths）→ `pipeline` → `render` →
`cli`；`audit` 最后，用钩子接。每一步都要求「新包跑出的数值与根目录逐位相同」，
用现有 `test_ecm` / `test_contract` 当对照。

## 6. 为什么这一轮不物理移动文件

仓库里有一整套**冻结证据链**，它对文件路径与字节是敏感的：

- `source_manifest.json` 记录三套源码树的**内容哈希**，`--check` 会对齐；
- `budget_chain_probe_review.py` 把三个文件**字节锁**到 `8015721` 的 blob；
- 多个测试有**扫描型 import 守卫**（禁止生产模块 import 身份/采集/打包层），路径一变就失效；
- `.gitignore` 是**逐文件白名单**（`!evidence/...` 数十行），移动即失配；
- `docs/` 与 `docs/INDEX.md` 大量按路径引用脚本。

结论：**物理搬迁 = 一次性作废上面五样**，且要重跑全套回归并重写白名单。所以本轮只做
「分类 + 阻塞项清单 + 目标结构」，把搬迁留到新包里做增量，而不是在原树上大动。

## 7. 维护

```bash
python tools/repo_map.py           # 重新生成 docs/code_map.json + 打印摘要
python tools/repo_map.py --check    # 校验：磁盘上的每个 .py 都在分类表里
```

新增模块时：先在 `ROLE` 里给它 (层, 职责, 产品候选, 稳定性)，再跑 `--check`。
`--check` 同时会报出「分类表里有、磁盘上没有了」的残留项。

## 8. 变更日志

| 日期 | 改动 |
|---|---|
| 2026-11-15 | 建立；62 模块分类完成，产品核心 10 / 闭包 26；列 5 个阻塞项与 11 处层间反向依赖 |
