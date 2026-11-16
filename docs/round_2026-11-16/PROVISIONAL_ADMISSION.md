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
| `bmdrt.bat` | **Windows 唯一入口**（2026-11-16 起）：`%*` 透传子命令，`nested_fit.bat` 已删除 |
| `bmdrt.py` | **族分派**（本轮新增）：`_voigt_family_verdict()` 调 registry 预检，`excluded` ⇒ `_run_nested_branch()`；`family_auto` 开关在 `config.json` |
| `config.json` | `fit.family_auto = true` + 说明（`false` 即回到旧行为） |

`bmdrt nested` 与 `python nested_fit.py` 逐位相同：同一个 `main()`、同一份参数表、
同一套产物路径（默认 `out/nested_fit/`）。

**为什么不给 provisional 做 config 开关**：开关会被默认值悄悄打开。
这里只有模块级常量，改它必须改代码并过 review。

## 3.5 族分派：`auto` 自己判到并联（2026-11-16 第二轮）

### 3.5.1 上一轮实现得太窄

用户指出：「不是暂时解除限制了吗，只是输出说明一下」。
上一轮只加了一条**旁路**子命令 `bmdrt nested`，`auto` 里完全不知道 nested 存在 ——
那不是解除限制，是旁边开了个口子。

### 3.5.2 判据本来就有，只是没接

`circuit_registry.l0_precheck` 早就把 voigt 判为 **`excluded`** 了，理由写在产物里：

```
低频感抗 0.01–0.139 Hz（9 点）需把 R∥L 放进第二并联节点（= 改族定义）
```

`AZ31-4Li_150` 和 `AZ31-8Li_50` 都是这个状态。但 `process_one` 直接 `R.run`
→ `bm_core.fit`（voigt），从头到尾没问过预检。

### 3.5.3 现在的规则（`bmdrt.py::_voigt_family_verdict`）

确定性预检，**不跑拟合**，秒级：

| 预检结果 | 动作 |
|---|---|
| `not_excluded` | 按生产路径继续跑 voigt（**行为逐位不变**） |
| `excluded` | 改跑 nested，产物标 `provisional`，summary 判定写「未过验收线」 |
| `unknown` | **不切族**（"没登记预检" ≠ "不能用"），按 voigt 继续 |

开关：`config.json` → `fit.family_auto`（默认 `true`，用户要求"不用选"）。
设为 `false` 即完全回到旧行为。

### 3.5.4 口径差异必须跟着数字走

| 项 | voigt 主路径 | nested 分支 |
|---|---|---|
| 噪声 σ | 自由采样 + `sigma_floor` | **逐点 2%·\|Z\|**（§13/§14 验收口径） |
| 预算档 | `KW`（3000/1000/3 链） | `standard`（2000/1000/2）或 `quick`（800/400/2） |
| 产物 | `real_<stem>.json/.png` | `<stem>.nested.json/.nested_curve.csv/.nested.png` |

**为什么 nested 不用自由 σ**：实测同一条 `AZ31-4Li_150`，自由 σ 下
rms **52.60%**（高频端点 149%），逐点 2% 下 **6.39%** —— `|Z|` 跨 3 个量级时
自由 σ 会把高频点"合法地不拟合"（R16/R17）。用自由 σ 得到的数字没有意义。

代价：**两族的 rms 不能直接比**，两个数各自带口径标注，控制台也打印。

### 3.5.5 实测

```
bmdrt.bat auto archive/AZ31-4Li_150.csv --quick
  === 族预检：voigt → excluded ===
  === 族分派：voigt 预检 excluded ⇒ 改用 nested（provisional）===
    理由：低频感抗 0.01–0.139 Hz（9 点）…
    [准入状态] provisional：§13 0/3、§14 4/14、M-b 4/14、§17 0/10
    口径：σ = 逐点 2%·|Z|；与 voigt 的自由 σ 不可直接比
    预算：quick 档（800/400/2/0.9/10）
  → rms_rel 6.39%、acf 0.397、R̂ 1.0017、div 0（21.1 s）

bmdrt.bat auto out/fig3/inputs/3e.csv --quick
  === 族预检：voigt → not_excluded ===
    voigt 可表达这条谱，按生产路径继续     ← 行为不变
```

> `4Li_150` 的 quick 档 acf 0.397 **不能读成"过了 M-b"**：§14 同谱 standard 档
> 是 **0.726**。quick 只 400 样本，控制台会打这条警告（`quick 档只作预览…`）。

### 3.5.6 没做的

- **两族都跑再比**（`compare` 的逻辑）还没接进 `auto`。现在的规则只解决
  「voigt 表达不了」，不解决「两族都能表达、哪个更好」（如 3E：voigt 6.28%
  vs nested 5.81%）。要自动选优劣得跑两族，单谱时间 ×2~3。
- nested 的电路图仍画不出来（`bm_circuit.draw` 只认 voigt 的串联骨架）。


## 4. 红线由断言守着（`test_entry_parity.py` T8 + T9）

T8（准入标注）
- `state="provisional"`、`passed is False`
- 四条差距都有 `required`/`actual`/`note`（只写「未过线」不可复核）
- `parameters_are_physical is False`
- `usable_scope` 必须同时写明「仅 `_10`/`_50`」与「`_0`/`_5`/`_150`」
- **文案里「通过」「合格」一次都不出现**（否定式也会被单独摘录误读）
- 两处产物引用**同一个** `AD.block()`
- `nested` 出现在 `bmdrt.discover()` 但**不在 `BUILTIN`**
- 仓库顶层只有一个 `.bat`，用 `%*` 透传、不硬编码子命令

T9（族分派）
- `AZ31-8Li_50` 的 voigt 预检 = `excluded`，理由含「低频感抗」
- `3e.csv` 的 voigt 预检 = `not_excluded`，且**不切族**
- 族分派受 `family_auto` 控制；只有 `excluded` 才调 `_run_nested_branch`
- `R.run(` 的位置在分派之后 ⇒ voigt 主路径不会被短路
- `nested` 的判定含 `provisional`、不等于 `OK`、不含「通过/合格」
- nested 产物与 voigt **分文件名**、用 `perpoint` 2%、summary 有「族」列

## 5. 解锁正式准入的条件（写死）

1. **M-b 判据被替换或重新标定** —— 需要独立信息：开-短路 5 min 校准（CAL-01）、
   E1 同状态重复、多频窗；M-B-SIM-01 已证现有 M-b 不能归因。
2. 用新判据重算 §13 **3/3** 与 §14 **14/14**，且不改噪声口径与采样档。
3. 在此之前，**不许**为了数字好看换谱、放宽阈值、挑 seed、剔点。

## 6. 复现

```bash
D:/autoeis_env/Scripts/python.exe admission.py                  # 10/10
D:/autoeis_env/Scripts/python.exe cmd_nested.py --selftest       # 5/5
D:/autoeis_env/Scripts/python.exe test_entry_parity.py           # 143/143（T1–T9）
bmdrt.bat nested out/fig3/inputs/az31.csv --budget quick         # 显式走 nested
bmdrt.bat auto   archive/AZ31-4Li_150.csv --quick                # 自动族分派（→ nested）
```
