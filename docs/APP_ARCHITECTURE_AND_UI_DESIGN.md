# BM-DRT 应用架构与页面体系设计（v0.1 草案 · 待评审）

**日期**：2026-11-12　**状态**：草案，未冻结；本轮**未修改任何代码**，新增文档 1 份、索引 1 行。

**依据**：只读调研了 `THREE_LAYER_SCOPE.md`、`WS_DECISION_PROJECT.md`、`NESTED_TOPOLOGY_PROJECT.md`、`ADSORPTION_PROJECT.md`、`MODEL_COVERAGE_PLAN.md`、`M6_VALIDATION_PLAN.md`、`BMDRT_RESULT_AND_FIGURE_DESIGN.md`、`DASHBOARD_UI_SPEC.md`、`README.md`，以及 `bmdrt/` 全部主程序的入口、分层、产物与契约。下文的文件名/行号/章节号均为核对结果。

---

## 0. 设计立场：任务栏照拟合软件，内核不能照

用户诉求是"像一般拟合软件里一样有明确的任务栏"。但项目书对这个软件的定位恰好**反对单一结论**：

| 出处 | 原文要点 |
|---|---|
| `WS_DECISION_PROJECT.md` §5 WP2 产品化备注（2026-11-09 用户提出） | 不可区分性不该由程序单选；软件版应把六臂（或前 3）拟合图与参数表**并列输出、由人挑**；程序只给 `Δelpd`、`dse` 和「不可分」标记，**不给单一结论** |
| `THREE_LAYER_SCOPE.md` §4.3 / §6 | `evidence.multi_model` 默认 `false`，"**等做成软件时再打开**"；打开后按输入顺序、**不排序、不打分、不单选、不做跨臂配对检验**；"哪个 arm 塌到小值由人看，程序不判哪个对" |
| `THREE_LAYER_SCOPE.md` §4.2 | `grade` 只允许 A/B/C，**不允许分数**；每条证据必带 `not`（写死"这条不能推出什么"）；`next_evidence_needed` 必填；检查器禁词 `score / confidence / probability / weight / rank / best / selected / recommend`（`evidence_card_check.py` 2889/2889 通过） |
| `config.json` `evidence.multi_model` 注释 | "做成软件、需要人工从多个模型里挑时改成 `true`" |

ZView / EC-Lab 那类软件的信息架构是"选模型 → 拟合 → 给一组参数 → 结论"。照抄它，等于在 UI 层偷偷引入项目书明令禁止的"单选/最优/推荐"。

**因此本设计的定位是「证据工作台」**：外壳（菜单栏、工具栏、任务状态条、工程树、状态栏）照专业软件做，让人一看就会用；信息架构按项目书的三层职责来分——L1 拟合质控 / L2 模型比较与可辨识性 / L3 物理解释辅助。任务栏是**只读状态呈现**，不是动作向导。

---

## 1. 现状：代码事实与阻碍

### 1.1 已有分层（依赖方向自上而下）

| 层 | 模块 | 职责 |
|---|---|---|
| 入口 | `bmdrt.py` | `ROOT/CONFIG` 靠 `__file__`（:113-114）、`load_cfg`(:154)、`resolve`(:166)；`BUILTIN`(:722)=auto/watch/list/config/check/bench；`discover()`(:726) 扫 `cmd_*.py` 取 `run(argv)` → 现只有 `cmd_diff.py:394` / `cmd_status.py:89`；`main`(:850) 空参即 auto |
| 算法 | `bm_core.py`(jax/numpyro)、`bm_real.py`、`drt.py`、`bm_kk.py`、`bm_diff.py`、`bm_ws.py`、`bm_grade.py`、`bm_evidence.py`、`bm_circuit.py` | 拟合、门禁、判据、证据卡 |
| 结果契约 | `bm_result.py` | `AnalysisResult.load/payload/bundle/finalize_bundle`，`SPEC_VERSION="1.0"`，唯一数据源 |
| 展示 | `bm_dash.py`(build:116, `_TEMPLATE`:183, `DEFAULT_LAYOUT`:100)、`bm_figure.py`(`DEFAULT_SPEC`:96, `validate_spec`:140, `draw`:445) | 单文件 HTML / 出版图 |
| 研究脚本 | `nested_fit.py`(未接主流程)、`bm_zhit.py`(评估弃用)、`bm_top*.py`、`bm_select.py`、`bm_experiments.py`、`convert_2e/3e.py`、`batch_2e.py`、`evidence/`、`docs/cases/` | 一次性分析，不属于应用 |

**数据流**：Gamry xlsx → `convert_2e/3e.py` → `EISmeasurements/csv_{2e,3e}/` 三列 CSV → `inbox/` → `scan_inputs`(:483)/`process_one`(:231) → `out/<stem>/real_*.json` + `*_samples.npz` + `meta.json` + `run.log` → `AnalysisResult` → `results/<stem>/{dashboard.html, figures/, data/, metadata/, manifest.json}`。批次汇总 `out/summary.{json,md}`(:403)。

**契约**：`test_contract.py` F1–F7（数据一致 / 判定诚实 / spec 往返 / 载荷逐值相等 / 封包 / 残差半失效 / 配置严格）；`docs/round_2026-11-11/dash_smoke.js` 12 项 node 断言。

### 1.2 阻碍（为什么不能直接给 GUI 调这些函数）

| 阻碍 | 位置 |
|---|---|
| 全局态 `bm_real.OUT / NULL_CACHE / KW`，靠入口注入覆写 | `bm_real.py:97-106`、`bmdrt.py:262-263` |
| `load_spectrum` 用 `sys.exit` 报错 | `bm_real.py:217` |
| 全链路 `print` + `Tee` / `redirect_stdout`，无结构化进度 | `bmdrt.py:188,249` |
| 异常被吞、`_write_meta` 静默、`process_one` 宽捕获 | `bmdrt.py:188,317-329,342` |
| 退出码恒 0；`discover` 把 `None` 抹成 0 | `bmdrt.py:694,703,863`、`bm_dash.py:179`、`bm_figure.py:590`、`cmd_status.py:140` |
| CWD 相对路径、批处理假设（扫 inbox、`_move_to`、`skip_done`、`watch` 死循环）、无任务 id / 并发 / 进度回调 | `bmdrt.py:508,516,640-660`、`bm_dash.py:153` |

这与第二轮代码审查（`代码审查报告_2026-11-11.md`）指出的系统性问题同源：**异常被转成看似合理的默认值**。应用层必须显式三态，不能继承这种静默回退。

**环境**：`D:\autoeis_env\Scripts\python.exe`（Python 3.10.7，jax/numpyro/arviz 齐全）；`impedance` 缺失时 KK 为 `unavailable`；内网/离线，不能依赖 CDN 或临时 pip 安装。

---

## 2. 目标架构：六层 + 双模

| 层 | 内容 | 改动 |
|---|---|---|
| L0 资产与配置 | `inbox/ out/ results/ config.json`；**产物即接口**，以 `manifest.json` 为索引单元 | 既有 |
| L1 算法内核 | `bm_core / bm_real / bm_kk / bm_ws / bm_grade / bm_evidence` | **冻结，零改动**（M6/WP 判据已预注册） |
| L2 结果契约 | `AnalysisResult`、`SPEC_VERSION=1.0`、F1–F7 | **冻结，零改动** |
| L3 **应用服务层** | `Task` 状态机、`Runner` 子进程隔离、`Index` 扫 manifest、`Evidence API`、本地 JSON API | **新增（唯一新增后端）** |
| L4 展示层 | `bm_dash.py` 单文件 HTML、`bm_figure.py` 出版图 | **零改动** |
| L5 宿主外壳 | 工程树、菜单栏、任务状态条、四工作区、证据卡并列页、运行日志 | **新增** |

**双模是硬约束**：

- **静态模式**：`python bm_dash.py <sample>` 继续产出完全自包含、离线双击可看的 HTML，可发给同事存档。不因做应用而丢失。
- **应用模式**：宿主用同一套 HTML 模板（iframe 或可选注入 bridge）呈现 L1 内容，其余工作区由 L5 提供。面板本体仍只读、不发起计算（对齐 `BMDRT_RESULT_AND_FIGURE_DESIGN.md` §9 与 `DASHBOARD_UI_SPEC.md` §0）。

### 2.1 L3 详细设计（新增目录 `app/`，不碰现有模块）

| 文件 | 职责 | 关键约束 |
|---|---|---|
| `app/contracts.py` | `Task`（id、输入路径、参数快照、状态、产物路径、起止时间）、`JobResult` | 状态三态 `ok / fail / not_evaluable`；**禁止 fail 静默转 ok**；数值回退必须落 `*_fallback` 标记 |
| `app/runner.py` | 以**子进程**执行 `bmdrt.py auto <file>`，捕获 stdout → 结构化事件（进度/阶段/错误），支持超时与取消 | 隔离 `bm_real` 全局态与 `sys.exit`；崩溃不拖垮宿主；可并发 |
| `app/index.py` | 只读扫 `results/*/manifest.json` + `out/*/meta.json` 构建 `ProjectIndex`；工程定义写 `app/workspace/project.json` | **不引入数据库、不改写 `results/` 结构**；`manifest.reproduce` 直接作复现入口 |
| `app/evidence.py` | 调 `bm_evidence.load_run`(:779) / `cards_for_run`(:676) / `multi_model_table`(:736) / `build_document(runs, labels, multi_model)`(:757) | 应用模式下以 `multi_model=True` 生成并列证据；**不改 `config.json` 全局默认值**，只显式传 `--multi-model`，避免影响 CLI 与测试基线 |
| `app/api.py` | 本地 HTTP/JSON：任务提交、状态查询、事件流、结果索引、证据文档、导出命令 | 只读消费 `payload()`，**不重算任何物理量** |

**任务状态机**：`queued → running → (ok | fail | not_evaluable) → archived`。

**完成判定三判**（绕开恒 0 退出码）：① 子进程 `returncode`；② `out/<stem>/real_<stem>.json` 与 `*_samples.npz` 存在且能被 `AnalysisResult.load` 读通；③ `results/<stem>/manifest.json` 的 `status=finalized` 且 `complete` 与 `missing` 自洽。三者取交集，任一不满足即 `not_evaluable` 并保留原始日志——**不猜、不补默认值**。

**为什么用子进程而不是把算法改成库**：算法层冻结、全局态、`sys.exit`、print 日志、退出码失真，全部在"进程边界"上一次性隔离；同时天然获得可取消、可并发、崩溃隔离，且完全不动 M6/WP 已冻结的判据。

### 2.2 稳定可复用 API（不需改动即可调用）

`bmdrt.load_cfg / resolve / process_one / scan_inputs / write_summary`、`bm_real.load_spectrum / run / save_out / select_k`、`bm_result.AnalysisResult.load / payload / bundle / finalize_bundle`、`bm_dash.build / DEFAULT_LAYOUT`、`bm_figure.validate_spec / draw / DEFAULT_SPEC`、`bm_evidence.build_document / multi_model_table / render_markdown`。

---

## 3. 页面体系：四工作区 + 常驻状态条

外壳五层（菜单栏 / 标题工具栏 / 任务状态条 / 页签 / 状态栏）与 6 步只读状态点见 `DASHBOARD_UI_SPEC.md`；该文是本文在**单谱面板**上的下位细化，两者不冲突。下面定义的是**应用级页面**。

| 工作区 | 对应职责 | 内容 | 数据来源 |
|---|---|---|---|
| **W1 工程台** | 数据与任务 | 谱列表/批次、每条谱的任务状态、导入与重跑入口、manifest 复现命令、跨谱检索 | `app/index.py` 扫 manifest |
| **W2 L1 拟合质控** | "拟合是否可信" | 现有 dashboard 全部内容：Nyquist / DRT / 残差、KPI、判定、抽屉、主题与导出 | `AnalysisResult.payload()` |
| **W3 L2 并列比较** | "哪些受支持 / 无法区分" | **多臂并列**：每个 arm 一张等宽卡（拟合图 + 参数表 + `Δelpd±dse` + 资格档 + 塌陷/出窗标记）；"不可分"标记；参数相关性 | `bm_evidence.multi_model_table` / `build_document` |
| **W4 L3 证据与报告** | "哪些结论可以写、哪些必须保留" | 证据卡（A/B/C）、每条证据的 `not` 与 `next_evidence_needed`、三出口导出、manifest 与 `figure_spec.json` 复现 | `bm_evidence.cards_for_run` / `render_markdown` |

### 3.1 W3 是本项目与普通拟合软件的分水岭

按 `WS_DECISION_PROJECT.md` §5 与 `THREE_LAYER_SCOPE.md` §4.3：

- **等宽、等高、等色，按输入顺序排列**；不设选中态、不高亮、不提供排序控件、不显示"推荐/最优/最佳"。
- 卡片只陈述事实：`Δelpd ± dse`、资格档 `A/B/C`、`A_W`、`cv`、塌陷/出窗标记、`amp_spread`；**程序不判哪个 arm 对**。
- `grade=0` 记 `non_evaluable`，不进选模型，不用它的 elpd 做结论（`WS_DECISION_PROJECT.md` §5 判读边界）。
- 真实谱无真值，禁止"证明存在扩散""确认为 Ws"；允许措辞是"低频段需要额外自由度，且这些解释不可唯一区分"。

### 3.2 W4 的证据卡约束（必须做成一等 UI 元素）

- 每条证据的 **`not`（这条不能推出什么）** 与 **`next_evidence_needed`（还缺什么）** 常驻可见，不做折叠、不做小字免责声明——`THREE_LAYER_SCOPE.md` §4 明确"这比在报告里写免责声明可靠"。
- `grade` 只允许 A/B/C 三档标签，**不显示分数、进度条或百分比**。
- **C（物理证据不足）是正常且最常见的终态**，用中性色，不染红、不写成失败。
- A（算法无法判断）必须引导"修码/重跑/提高预算"，不得呈现为数据结论；B（数据无法判断）必须并列保留多候选，不得单选或写"不存在"。

### 3.3 全局 UI 红线（沿用上位文档）

判定分两层陈述（采样档 + 结构判据 n/3）；Lin-KK 四态 `ok/warn/fail/unavailable` → 通过/警告/未通过/不可评价，`warn` 不染红；缺后验/缺 σ 写"不可用"、CSV 留空不写 0、不套 2%；只跑一个 K 写"未做跨 K 比较"、不显示 `ΔELPD 0.0`；Rwp 只报数值不上色；不做可信度总分、不把 `acf/Rwp/ELPD` 换排名；风格预设不宣称符合期刊规范；状态条不设当前步/完成度/总分；`band` 与 `pred` 不同图混用。

UI 与数据模型中**禁止出现** `score / confidence / probability / weight / rank / best / selected / recommend` 语义字段（`THREE_LAYER_SCOPE.md` §4.2 检查器口径）。

---

## 4. 技术选型与部署形态

| 项 | 建议 | 理由 |
|---|---|---|
| 宿主形态 | Python 本地服务 + 系统浏览器 `--app` 模式（零额外依赖）；如需窗口外壳再评估 pywebview | 内网离线，优先标准库 `http.server`；需要并发时再换 FastAPI（须先确认离线可装） |
| 前端 | 不引框架、不引 CDN；沿用 `bm_dash.py` 的原生 SVG 与主题变量 | 1649 行手写绘图已被 12 项契约守住；换框架等于重写并重新对齐所有图 |
| 数据 | 无数据库；`manifest.json` 即索引，`project.json` 只存工程组织 | 产物即接口，复现命令已在 manifest 里 |
| 打包 | 优先"绿色目录 + 启动脚本（起服务 + 开浏览器）"；安装包（PyInstaller）作为 S3 备选 | 离线内网零安装最稳；打包不依赖前端构建链 |
| **不选** | Electron、Plotly CDN、前端 SPA 化改造面板 | 双运行时/体积/离线；且会丢掉单文件 HTML 可交付性 |

**并发与隔离**：Runner 用子进程，一条谱一个进程；`config.json` 的 `batch.move_done/skip_done` 在应用模式下应显式关闭（由 Task 自行管理归档），否则 GUI 与 CLI 会抢同一批 inbox 文件。

---

## 5. 分期路线

| 期 | 交付 | 验收 |
|---|---|---|
| **S0**（本轮） | 本设计文档 | 代码零改动 |
| **S1** | 新增 `app/`：Task + Runner + Index + 只读 API；CLI 完全不变 | 从 API 提交一条谱，产物与 CLI 逐值一致；`test_contract.py` F1–F7 全过；失败用例返回 `not_evaluable` 而非静默 ok |
| **S2** | L5 宿主外壳 + W1–W4 四工作区；应用模式显式打开 `evidence.multi_model` | 并列页三臂等权、无禁词、无选中态；静态 `dashboard.html` 仍可离线双击打开；`dash_smoke.js` 12 项全过 |
| **S3** | P4 跨样品拼版（R18 补强）+ 打包分发 | 同状态重复谱对照；不影响单谱面板 |

**S2 就是项目书说的"做成软件时"**——`THREE_LAYER_SCOPE.md` §6 把 `evidence.multi_model` 的解锁条件明确挂在这个时点上，不是随手打开一个开关。

**明确不做**：机制评分器/分类器、L3 单一推荐、自动物理解释、面板内拟合/选 K/改门禁、自由坐标拖放、可信度总分、Electron、数据库、改 `SPEC_VERSION`、改 `PAGES`/`DEFAULT_PANELS` 的 id 序列（会让 localStorage 版式回默认）。

---

## 6. 风险与回归

| 风险 | 处置 |
|---|---|
| 打开 `multi_model` 影响 CLI/测试基线 | 不修改 `config.json` 默认值，只在应用模式显式传 `--multi-model`；并列产物写 `app/workspace/`，不改写 `results/` |
| 并列结果被 UI 层"优化"成单选 | 把 §3.1/§3.2 的禁词与"无选中态"写成契约断言，纳入 S2 验收 |
| 静态 HTML 可交付性丢失 | 双模为硬约束；L4 零改动；S2 验收含离线双击 |
| 多谱工程树与 `BMDRT_RESULT_AND_FIGURE_DESIGN.md` §10「单谱」限制冲突 | 工程树只是**索引层**，不改单谱面板；跨谱对照作为 P4 另立 |
| 界面改造挤掉 P4（§8 下一轮优先、R18 可视化补强） | S3 显式列出，不与 S2 互换顺序 |
| 全局态/退出码失真导致假成功 | 完成三判 + 三态，禁止静默回退 |

---

## 7. 与既有文档的关系

- **上位决策**：`THREE_LAYER_SCOPE.md`（三层职责、A/B/C、禁词）、`WS_DECISION_PROJECT.md`（并列输出产品化备注）、`BMDRT_RESULT_AND_FIGURE_DESIGN.md`（三出口同源、只读展示层、单谱、P4）。冲突时以上位为准。
- **下位细化**：`DASHBOARD_UI_SPEC.md`（单谱面板外壳五层、6 步状态条、垂直预算、实施改动点）。
- **本文**：整体架构 + 应用级页面体系，规定新增层的位置与边界。

---

## 8. 未决问题（需拍板）

1. 宿主形态选"系统浏览器 `--app`"还是 pywebview 窗口？前者零依赖，后者更像原生软件。
2. 本地服务是否允许引入 FastAPI（离线安装可行性）？建议先按标准库实现，不够再换。
3. 工程树（W1）是否与 P4 跨样品拼版合并立项？还是 W1 只做索引、P4 单独做对照图？
4. `multi_model` 并列产物落在 `app/workspace/` 还是写进 `results/<stem>/metadata/`？前者不动现有导出契约，后者可随包分发但会改 manifest。
5. 应用模式下 `batch.move_done / skip_done` 由 Task 接管，是否需要同步改 `config.json` 的默认值（本文建议不改全局默认，只按模式覆盖）。
