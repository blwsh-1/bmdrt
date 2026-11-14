# 文档索引

**2026-11-11 起**：仓库根目录下的设计与验证文档全部迁到这里（`README.md` 仍在根目录，
它是仓库首页）。代码与文档正文里的交叉引用一直用**裸文件名 + 章节号**
（例如「见 `M6_VALIDATION_PLAN.md` §14」），不带路径，所以迁移后这些引用依然能找到；
需要按路径定位时到本目录。

## 顶层文档

| 文档 | 内容 |
|---|---|
| `NESTED_TOPOLOGY_PROJECT.md` | **嵌套并联电路拓扑立项书**：为何串联 Voigt 表达不了低频感抗环（三个硬约束）、理论上限对照（30.5% vs 6.7%）、渐近极限、`Rs≈0` 不可辨识警报、CPE 窗口幅值参数化、M0–M6 里程碑；§9.4.x 是跨族选择的全过程记录（§13/§14/§17/§18/§19 验收、§9.4.31–36 接口与判据修正），末尾是风险登记 **R1–R22** |
| `M6_VALIDATION_PLAN.md` | **M6 实测数据验证预注册**：吸附线 / nested 线两个 M6 分开验收；24 条真实谱盘点；M6.1 基线 / M6.2 边界 / M6.3 假阳性三层判据（G1–G7）；§13 留出验收、§14 全量验收与变更日志、§15–§19 归因与合成谱对照 |
| `BM_TOP_M3_PLAN.md` | **`fit_top()` 接口设计与 M3 验收清单（v3，已冻结）**：候选 A/B/C 对比、参数先验冻结表、生成器独立性、三档验收与失败记录规范、M3.5 逐点相对噪声的接口/回归/预注册与 LOO 有效性、两层判定与覆盖口径 |
| `ADSORPTION_PROJECT.md` | **低频吸附电感元件立项书**：`R_ad∥L_ad` 定义、负 RC 恒等式、DRT 盲区、三区间可辨识性、M0–M6 里程碑与验收标准 |
| `AD_GATE_SPEC.md` | **吸附门控边界状态规格**：`enter`/`strong` 严格分离、五状态定义、`edge_peak`、`d_low` 与采样密度判据、「不增加假阳性」的不变性论证 |
| `AD_M6_SAMPLING.md` | **吸附电感 M6 抽样验证记录**：18 条真实含环谱盘点、4+10 分层抽样设计、首个真实正例、门控两处缺陷、抽样置信上界与分阶段全量方案 |
| `DIFFUSION_PLAN.md` | 扩散扩展方案：数学、先验、测试矩阵、模型选择 |
| `MODEL_COVERAGE_PLAN.md` | **模型体系规划与覆盖地图**：三层架构、11 项现象覆盖状态与证据等级、注册表接口约定、按证据排序的立项队列与准入规则 |
| `WS_DECISION_PROJECT.md` | **Ws 判断专项立项**：把「要不要挂扩散项」从一次模型选择整理成能力边界清楚、可复核的专项 |
| `WS_CRITERION_PLAN.md` | **Ws 判据的算法探索（v0.2，探索性）**：N1–N4 判据设计与合成真值标定、Boukamp 梳齿指纹验证为不可用 |
| `THREE_LAYER_SCOPE.md` | **三层职责与证据资格**：A 算法 / B 数据 / C 物理证据不足的划分与输出口径，本文只重新分配「谁负责说什么」 |
| `CIRCUIT_TOPOLOGY_ADMISSION.md` | **电路/元件扩展准入评估（冻结，2026-11-12）**：逐缺口的物理动机、现有证据、最低验证条件与准入状态；§8.1 明确「登记合法 ≠ 准入通过」、registry 只作记录与比较接口，§11 列非目标，§13 `0/3` / §14 `4/14` / §17 `0/10` 原样不动 |
| `EXECUTION_IDENTITY_CONTRACT.md` | **执行身份契约 v1.3（部分冻结；契约层已实现、runner 未授权）**：同条件 / 同实例 / 可重建三分（有 SHA256 ≠ 能恢复原文件）；`git describe --dirty` 只作展示，身份 = `git_commit` + **numeric 源码集摘要**（声明集与运行时集分离、`runtime ⊆ declared` 强制核对，`compute_role` 默认 numeric）；dirty 降为范围判据且 dirty 时补丁包必落（含未跟踪文件内容）；`effect` 六值闭集（model/execution 进哈希，admission/decision/workflow 不进）、**三入口参数表按实际签名逐条反推**（§3.3.4）、字段委托表、`sigma` 复合键（**唯一载体**：只 `array_sha256`+`floor` 进哈希，`sigma_mode/pct/abs` 退出哈希）、反例矩阵 **T/W/S/Y/M/X/C 共 65/65 已跑通**；D-1～D-8 全部已定（D-5 有条件：补第三层 `workflow_source_tree_sha256`）；**三层摘要**（declared 只记录 / numeric→`run_id` / workflow→`workflow_id`，`compute_role` 与 `workflow_scope` 两个独立维度）；§6.3 `workflow_id` 载荷（源码+搜索+门禁设定+children，**结果不进**）；覆盖关系**由集合重算**（`source_coverage` 不是可信输入）、配置类型/枚举进判定、三时点契约化（`sampler_entry` 缺失照样降级）、结构性非法 ⇒ `unconfirmed` 无号；§18 冻结台账（18.1 有反例 / 18.2 无真实运行）；实例字段一律不进 `run_id` |
| `BMDRT_RESULT_AND_FIGURE_DESIGN.md` | **结果与图形系统设计（v1.0）**：为什么把「分析总览」与「论文作图」分开、`AnalysisResult` 单一数据源、面板布局与理由（Nyquist 跨两行 / Bode 进详情 / 表格不塞进子图）、视觉编码固定表、口径三条（两套区间、σ 口径、判定分两层）、导出目录契约与验收条件 |
| `REFS_DIGEST.md` | **EIS 文献与课件整理**：只做阅读与对照，按引用纪律一律**非真值**、不用于改判据门槛 |
| `EXTERNAL_BASELINE_ADAPT_ECM.md` | **外部基线盘点（ADAPT-ECM）**：三档 DRT 与我们的 BM-DRT 同源、源码核过的模型差异、我们独占的判据能力、"别去比工具"的定位判断；逐谱实测见 `round_2026-11-12/ADAPT_BASELINE.md` |
| `代码审查报告.md` | 第一轮代码审查（2026-11-06）：3 个用测试复现的高危缺陷 + 10 余个次级问题 |
| `代码审查报告_2026-11-11.md` | 第二轮代码审查（2026-11-11）：3 P1 + 14 P2 + P3，修复记录见 `NESTED_TOPOLOGY_PROJECT.md` 与提交 `6072932` |
| `DASHBOARD_UI_SPEC.md` | **面板界面外壳规范（v0.1 草案，未冻结，2026-11-12）**：为什么「任务栏」是只读结果状态条而非动作入口、五层信息架构、6 步与 `payload()` 字段映射、与 v1.2 布局及 12 项 node 契约的兼容矩阵、垂直预算、实施改动点与验收；本轮代码零改动 |
| `APP_ARCHITECTURE_AND_UI_DESIGN.md` | **应用架构与页面体系设计（v0.1 草案，未冻结，2026-11-12）**：依据项目书与代码事实，论证「任务栏照拟合软件、内核不能照」——六层架构中只新增 L3 应用服务层与 L5 宿主外壳、L1/L2 冻结 L4 零改动、静态/应用双模；四工作区按 L1 质控 / L2 并列比较 / L3 证据卡三层职责划分；S2 即项目书所说「做成软件时」解锁 `evidence.multi_model` 的时点；含子进程隔离、完成三判、禁词红线与分期验收 |

## 子目录

| 目录 | 内容 |
|---|---|
| `cases/blank_0.1KBr_pH10/` | **真实 EIS 谱模型选择诊断案例**：四层可辨识性规格、逐频点 LOO 与配对 dse、精确 HN 参考频率贡献、2×3 统一比较、合成谱用例与历史回放 |
| `round_2026-11-11/` | **2026-11-11 复核轮**：五个只读诊断脚本（K 选择分档、先验尺度敏感性、外部电路复现、残差结构对比、Lin-KK 阶数曲线）+ `AZ31-4Li_100` 残差报告 |
| `round_2026-11-12/` | **2026-11-12 四轮评估**：① Z-HIT 一阶形式合成标定（完美谱自身 0.95–3.25%）、22 条真实谱全 fail 且由谱型驱动 →「不接」；② `ADAPT-ECM 4.0.1` BM-DRT 外部基线（`adapt_baseline.py` / `ADAPT_BASELINE.md`）：可脱离 GUI 批调、模型族与拒答机制逐行核对、7 条谱对照（含 `ours` 漏检勘误与 5 条 acf 配对）；③ `ADAPT_TUNE_AZ31.md` + `adapt_tune_az31.py` / `our_side_az31.py`：AZ31-4Li_100 预注册 13 组调参扫描，澄清「我们的」= voigt 默认族（`real_*.json`）vs 嵌套研究接口（`*.nested.json`），同口径三方对照；④ `NOISE_ACF_TRUTH.md` + `noise_acf_truth.py`：利用 `测试数据/` 的拓扑+参数真值标定 acf，数值断言三类拓扑归属（涂层=nested L=2、钝化=L=1、腐蚀=ZARC），复现 §13.3 零分布，钉死 acf 尺度不变、只响应相关结构，并量化「跑错模型」把 acf 从 0.29 推到 0.89；附频率重排与 `AZ31-4Li.txt` 六列 index 格式陷阱 |
| `round_2026-11-13/` | **2026-11-13 五轮工作**：① `POWER_AUDIT_*` 检出力审计（配置数↔留出精度、`√n_h` 修正、噪声底 `√2` 修正）；② `STAGE2_SPEC.md` 第二段规格第四稿（24 配置全析因、三组分层、P1–P3 三门，**未冻结未执行**）+ `stage2_design.py` 设计矩阵诊断（条件数 1.000000、strength 残差 5.2e-16）；③ `COATING_VOIGT_PILOT_PREREG.md` / `FINDINGS.md` / `coating_voigt_pilot.py`：涂层 60 条分层 pilot（voigt `ok=0`、nested 贴底（收敛 57/60，见 FINDINGS 更正）、`R̂` 爆表源于标签置换对称）；④ **候选电路库阶段 0**：`STAGE0A_SCHEMA_PRESSURE.md`（三谱压测）、`STAGE0C_SCHEMA_CONTRACT.md`（**76 条契约，registry 0.7.0**：身份等级 + 六项描述块 + 执行身份 + 重复判定三态 + 数据内容哈希 + 产物定位分层（`same_artifact` 要求 `sha256` 非空且是 64 位十六进制、文件级/行级不互认）；另 `test_stage0_freeze.py` 9 条快照漂移用例）、`STAGE0_EVIDENCE_MAP.md`（实存矩阵 + 字段缺口 + 两处成文更正）、`snapshot_stage0_evidence.py`（证据副本与 sha256 清单）；⑤ **阶段 0 审计后修订（11-14）**：`STAGE0_RECORDS_MAP.md` + `map_stage0_records.py`（**v4**：239 条 / 独立 239（严格 237）、去重改按来源证据（三态），确认重复要求执行身份 `recorded`（本批 0 条，两对降为 possible）、数据身份改用内容 sha256、新增执行身份 `run_id` 与 `candidate_id` 分层、`--check` 校验整个快照；0.6.0 复核：`same_artifact` 缺字段碰撞已修、0.7.0 复核：`sha256` 格式校验（非空 ≠ 有效）已补，两轮对本批均零影响；修掉五处身份缺陷——目录名推断从未命中、AZ31 的 voigt 行被算成 nested 撞车、pilot `sensitivity` 段被当成 nested、pilot voigt 初值在 v2 被改反（实为默认初始化）、oracle 预算误标 unavailable；70 条"跑完但采样未过"证明旧 schema 装不下自己的数据）、`evidence/stage0_records/`（含 `MANIFEST.json`：三份文件 + 注册表指纹 + 9 份证据 + 63 份原始谱的 sha256）；⑥ 图与诊断脚本 `plot_pilot.py` / `mode_diag.py` / `k3_check.py` / `config_check.py`；⑦ **历史执行身份缺口只读审计（11-14）**：`HISTORICAL_EXEC_IDENTITY_GAP.md` + `audit_exec_identity_gap.py`——6 条只差 `code_version`（含本批仅 2 对 `possible`，若能补则独立数 239→237）、233 条还缺 seed/预算、9 份产物**零**身份字段、mtime 只给上界、库模块无副本 ⇒ 缺口只能向前关（runner 写 `git describe --dirty`），**6 条与 233 条一律不补**，`R-` 仍 0 条；⑧ **执行身份契约 v1 草案（11-14）**：`EXECUTION_IDENTITY_CONTRACT.md`（`v1.1-draft`）——同条件/同实例/可重建三分、`--dirty` 不作身份、声明集与运行时集分离 + `runtime ⊆ declared` 核对、dirty 补丁包含未跟踪内容、配置键类型无关闭集、实例字段不进 `run_id`、6 条重跑是「重建基准」而非补历史、233 条归 Stage 1；D-1～D-4 已裁决、**只定义不实现**，两轮均零代码改动；⑨ **契约 v1.2（11-14）**：三入口参数表（§3.3.4）+ 字段委托表 + `audit_source_profiles.py`（AST 闭包：A 13 / B 17 / C 16 模块，暴露 `nested_fit→bm_core` 函数体内导入与「闭包 ⊋ 参与计算的代码」）+ 反例矩阵 T1–T24，新提 D-5～D-8 待裁；⑩ **契约 v1.2 动工（11-14）**：D-5 有条件采纳（补第三层）+ D-6/D-7/D-8 采纳；新增 `exec_identity.py`（三层摘要 / `run_id` / `workflow_id` / 状态阶梯，纯函数、无 I/O）、`source_manifest.json`（三 Profile：A 15 / B 19 / C 18 文件，`compute_role` 全 numeric、`workflow_scope` 全 true，6 条待验证降级候选 `applied:false`）+ `round_2026-11-13/build_source_manifest.py`、`test_exec_identity.py`（**54/54**）；订正三处原措辞（T19 形状不可能相同、T1 固定 `candidate_id` 才有分辨力、T10 用夹具而非真实清单）；§3.1.4 降级双条件（不出现在运行时集 **且** 改常量后数值逐位相同，后者只算回归验证）；⑪ **契约 v1.3 实现修复（11-14）**：审查指出两处 P0——`sigma_mode/pct/abs` 与复合键**同时主张同一条件**（同数组不同来源会换号，D-8 失效）、覆盖关系只听自称字段（`source_coverage=complete` + `undeclared_modules=[]` 就能骗过）；两处 P1——配置类型标签没被执行、`sigma.floor` 用 `or 0.0` 把 `None` 当 0。已修：`SIGMA_MEMBER_KEYS` 退出哈希 + 复合块成唯一载体（缺块降级）、覆盖关系从 `runtime_loaded_modules` 与声明集重算（三码）、`_type_ok()` 执行类型/枚举标签（`bool≠int`、`kk_mode∈{gate,warn,off}`）、`floor` 三分、`kind↔mode` 强制自洽、三时点 + `runtime_snapshot_limitation`（声明限制**不免降级**）、结构性非法 ⇒ `unconfirmed` 无号；`identity_contract_version` `1.2`→`1.3`；反例 54→**65/65**；T19a 按真实捕获结构重写（把旧键集放回去立刻失败）、T23c 新增（修复前会通过） |

## 注意

- 本目录文档里的跑批命令都假定**当前工作目录是仓库根目录**。
- `docs/round_2026-11-11/` 里的脚本用 `dirname³(__file__)` 回推仓库根，目录深度与
  原 `evidence/<子目录>/` 相同，`sys.path` 不需要改；产物仍写 `out/`。
