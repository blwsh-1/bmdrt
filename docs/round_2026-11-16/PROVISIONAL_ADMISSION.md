# 临时准入（provisional）：能跑 ≠ 达标

> 用户 2026-11-16 裁决：「合起来能跑就行，本来就没有合格，不能声明是过线了的。」
> 本文记录**准入到底差什么**、临时准入落地成什么、以及哪些断言守着"不许声称过线"。

## 1. 裁决与边界

| 项 | 内容 |
|---|---|
| 接入状态 | `provisional`（临时） |
| 允许 | 一条命令跑起来、产物可读、可作探索性分析 |
| 禁止 | 声称过线、写进 `summary.json` 的达标统计、参与 KK 门禁、参数当物理量/真值 |
| 不改 | `bmdrt auto` 的默认族（仍是 voigt）、M-b 阈值、§13/§14/§17 判定、evaluator E4/E5 |

## 2. 准入差什么（两个层次，别混）

### 2.1 第一层：单谱门槛 A1–A4（§9.4.21.1）

| # | 条件 | 阈值 | 现状 | 判定 |
|---|---|---|---|---|
| A1 | rms（`AZ31-4Li_100`） | ≤10% | 绝对 σ 18.03% ✗ / 逐点 2% **6.75%** ✓ | 满足（依赖 A4 口径） |
| A2 | acf 较 voigt 明显下降 | — | **0.598 vs 0.907** | 满足 |
| A3 | 全参数 R̂、无发散 | ≤1.05 | `quick` 1.29–1.96 ✗ → `standard` **≤1.003** ✓ | 满足（须走 standard） |
| A4 | 噪声口径预注册 | — | 已写进 §13 | 满足 |

**结论：A1–A4 不是当前的墙。**§9.4.21 那句「A1–A4 一条未满足」是 2026-11-11 的状态注，
A3 在 §9.4.22 用 `standard` 档解掉后已失效；本次不改那句原文（避免改写历史判定），
只在这里记录当前实测。

### 2.2 第二层：留出集验收（真正没过）

| 验收 | 要求 | 现状 | 差 |
|---|---|---|---|
| §13 留出 3 条 | 3/3 | **0/3** | 3 条全未过（`4Li_150` / `8Li_100` / `12Li_100`） |
| §14 全量 14 条 | 14/14 | **4/14** | 10 条未过 |
| ├ M-a（rms ≤10%） | | 8/14 | 6 条（11.05–30.80%） |
| ├ M-b（acf ≤0.50 且比 voigt 低 ≥0.20） | | **4/14** | 10 条，acf 0.505–0.896 |
| └ M-c（R̂/div） | | 12/14 | `12Li_0` div=7、`12Li_5` R̂=22.07 |
| §17 四个文献电路 | 归因用 | **0/10** | 10 条未过谱上 M-b 全不触发 ⇒ 不是拓扑选错 |

有结构，不是随机：达到 §14 的 4 条全在 `_10` / `_50`（中段状态），
`_0` / `_5` / `_150` 全灭，残差集中在两端（低频 23–32%、高频 42–58%）。

### 2.3 判据本身已被证伪（这是"临时"不是"打折"的理由）

`M-B-SIM-01`（80 条盲测，判类型 C）：仪器寄生（B）与样品侧结构（C）两组，
拟合后 acf **AUC = 0.52**，95% CI `[0.36, 0.68]` —— 等于随机。
原因不是偏差小（最大 56.5%、拟合前 acf 0.88），而是 10 参数 `parallel × L=2`
把两类机制**都吸收**了；真实谱高频 acf 0.669–0.916，模拟组只有 0.39。

所以：**未过 ≠ 模型差；过 ≠ 模型对**。风险登记 R27。

## 3. 落地

| 文件 | 作用 |
|---|---|
| `admission.py` | `STATE="provisional"`、`PASSED=False`（**不是 config 开关**）、逐条 `GAP`、`USABLE_SCOPE`、`JUDGE_CAVEAT`、`UNLOCKED_BY`；`block()` 进产物、`banner()` 进控制台 |
| `cmd_nested.py` | `bmdrt nested` 子命令，薄转发 `nested_fit.main`（零算法、不另写参数表） |
| `nested_fit.py` | 产物加 `admission=AD.block()`；`WARNING` 订正为「可作为 `bmdrt nested` 调用，但不改 `auto` 默认族」 |
| `cmd_compare.py` | `compare.json` 加 `admission=AD.block()`；启动打印 banner |

`bmdrt nested` 与 `python nested_fit.py` 逐位相同：同一个 `main()`、同一份参数表、
同一套产物路径（默认 `out/nested_fit/`）。

**为什么不给 provisional 做 config 开关**：开关会被默认值悄悄打开。
这里只有模块级常量，改它必须改代码并过 review。

## 4. 红线由断言守着（`test_entry_parity.py` T8）

- `state="provisional"`、`passed is False`
- 四条差距都有 `required`/`actual`/`note`（只写「未过线」不可复核）
- `parameters_are_physical is False`
- `usable_scope` 必须同时写明「仅 `_10`/`_50`」与「`_0`/`_5`/`_150`」
- **文案里「通过」「合格」一次都不出现**（否定式也会被单独摘录误读）
- 两处产物引用**同一个** `AD.block()`
- `nested` 出现在 `bmdrt.discover()` 但**不在 `BUILTIN`**
- `bmdrt.py` 源码里没有 `nested` / `family`（默认族仍是 voigt），`process_one` 仍直连 `bm_real.run`

## 5. 解锁正式准入的条件（写死）

1. **M-b 判据被替换或重新标定** —— 需要独立信息：开-短路 5 min 校准（CAL-01）、
   E1 同状态重复、多频窗；M-B-SIM-01 已证现有 M-b 不能归因。
2. 用新判据重算 §13 **3/3** 与 §14 **14/14**，且不改噪声口径与采样档。
3. 在此之前，**不许**为了数字好看换谱、放宽阈值、挑 seed、剔点。

## 6. 复现

```bash
D:/autoeis_env/Scripts/python.exe admission.py                  # 10/10
D:/autoeis_env/Scripts/python.exe cmd_nested.py --selftest       # 5/5
D:/autoeis_env/Scripts/python.exe test_entry_parity.py           # 118/118
D:/autoeis_env/Scripts/python.exe bmdrt.py nested out/fig3/inputs/az31.csv --budget quick
```
