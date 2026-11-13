# 阶段 0 证据快照

本目录是**只读副本**，不是运行产物。复制自 `out/`，逐文件 sha256 记在 `MANIFEST.json`，
复制时断言过副本与源字节相等（脚本 `docs/round_2026-11-13/snapshot_stage0_evidence.py`）。

为什么放在版本库里：`out/` 被 `.gitignore` 排除，而候选电路库的 schema 能不能承载
历史证据，必须能被仓库外的人独立检验。所以这里是"证据副本"，`out/` 仍是唯一生产目录。

## 内容

| 文件前缀 | 是什么 |
|---|---|
| `coating__*` | 涂层谱 S1，同拓扑换最内层元件（`parallel`/`capacitive`）、换预算（`standard`/`quick`） |
| `passivation__*` | 钝化谱 P1，同候选换初始化（默认 / 真值 oracle） |
| `az31__m4_real_az31.json` | AZ31 的 voigt 结构预检：18/18 `ok:false`，**没跑采样** |
| `az31__m4_real_nested_az31.json` | 同谱 6 个配置并列（parallel/series × L1/L2 + voigt K1/K2） |
| `six_runs__family_fit_results.json` | 另外六条谱的两族同口径配对（nested 走数据驱动初值，voigt 走逐点 2%） |
| `scripts/` | 产生上述两个复合产物的脚本，原样归档 |
| `MANIFEST.json` | 每个文件的 `source_path` / `source_basename` / `sha256` / 字节数 |

原始文件名含真值参数，重命名只为区分角色；真值以 `MANIFEST.json` 的 `source_basename` 为准。

## 读之前必须知道的三件事

1. **这不是完整网格。** 三条谱能凑齐的配置数不同，很多格子是"从未运行"，不是"跑失败"。
2. **`passivation__diag_truth_init.json` 是 oracle 初值**，用文件名里的真值参数，
   生产拿不到。它证明"拓扑能表达"，**不证明"这个候选能拟合"**。
3. `out/` 不入库 ⇒ 克隆仓库的人无法用 `source_path` 回比，只能靠 `sha256` 校验副本自身。

完整审计（覆盖矩阵、字段缺口、两处更正）见
`docs/round_2026-11-13/STAGE0_EVIDENCE_MAP.md`。
