# 层 0 两阶段固化入口（一条命令重排 21 条）

## 0. 解决什么

`LAYER0_GATE_AUDIT.md` §3 定位的缺口**不在判据在自动化**：`batch_fit_21.py` 跑完不会提示
"你的 best 有 67% 采样不过"，必须靠人记得先跑 `std_recheck.py` 再跑 `layer0_summary.py`。
本轮把这两步串成一条命令（`layer0_pipeline.py`，编排器），**原两脚本零改动**——它们是
被 `INDEX.md` 与 `STD_RECHECK.md` 引用的证据脚本。

两阶段（顺序不可颠倒）：

```
阶段 1  std_recheck.py      quick 采样不过者 → standard 预算复核（带 MCMC，progress.jsonl 断点续跑）
        └→ out/std_recheck/std_recheck.json  （21 条）
阶段 2  layer0_summary.py   合并 results.json + std_recheck.json
        └→ docs/round_2026-11-21/LAYER0_SUMMARY.md  （15 通过 / 6 non_evaluable / 1 翻转）
```

## 1. 用法

```bash
cd D:/eis训练数据/bmdrt
PY=D:/autoeis_env/Scripts/python.exe

$PY docs/round_2026-11-21/layer0_pipeline.py              # 全量两阶段（幂等，通常 no-op）
$PY docs/round_2026-11-21/layer0_pipeline.py --check      # 只读核对现有产物，不跑 MCMC、不写文件
$PY docs/round_2026-11-21/layer0_pipeline.py --dry        # 只打印计划
$PY docs/round_2026-11-21/layer0_pipeline.py --limit 2    # 阶段 1 部分复核（阶段 2 自动跳过）
$PY docs/round_2026-11-21/layer0_pipeline.py --budget deep --limit 2
```

**幂等性**：`std_recheck` 靠 `progress.jsonl` 断点续跑，21 条都已完成 ⇒ 默认全量运行不会
新跑任何 MCMC，两个产物逐字节不变（实测 sha 前后相同）。想强制重跑某条须先删其 progress 行
（见 `STD_RECHECK.md`），本入口不代做这件事。

## 2. 三道守卫（这是固化的真正价值）

| 守卫 | 查什么 | 为什么必须有 |
|---|---|---|
| **G1 前置** | `results.json` / `std_recheck.json` / `progress.jsonl` / 汇总表都存在、可解析、条数自洽（21=21=21） | 缺一份就跑，会在阶段 2 生成一份"大部分 `missing_standard`"的假汇总 |
| **G2 部分跑保护** | `--limit`/`--new-only` 触发时：①拒绝阶段 2；②用 `progress.jsonl` **重建** `std_recheck.json` | 这是本轮**实测抓到的坑**：`std_recheck.py` 在 `--limit 1` 下把 `std_recheck.json` 的 rows 从 21 **重写成 1**（不是只跳过汇总表）。若不重建，下一轮全量的 G1 就会因条数不自洽而红 |
| **G3 后置** | 汇总表生成后重读：「逐条」段的条数必须 = results 条数；`non_evaluable` 数必须 = `std_recheck` 的 `structurally_unidentifiable` 数；verdict 取值合法 | 防止阶段 2 静默产出不一致的表（如把按类统计表的行也算进去——开发中实测误计 24≠21） |

**负对照（已跑）**：篡改逐条表 1 处层 0 判定（不改 `std_recheck.json`）→ `--check` 报
`non_evaluable 7 != 6`，rc=1。把汇总表恢复后 rc=0。G2 实测：`--limit 1` 跑后汇总表
sha 不变、`std_recheck.json` rows 从 1 重建回 21、CRLF 原生格式保持。

## 3. 格式一致性（Windows 专属坑）

`std_recheck.py` 用默认 `open(..., "w")` 写 JSON —— Windows 下 `\n` 自动转 `\r\n`（CRLF），
且**不补末尾换行**。编排器的部分模式重建若用 `newline="\n"` 或补 `\n`，会造成纯格式造成的
字节差异（实测 sha `6f32bd55` → `ce6927ce`）。已改为与原生一致：不指定 `newline`、不补末尾换行。
全量运行复验：`产物变化: 无（幂等）`。

## 4. 边界（没做什么）

- **不改判据**：standard R̂≤1.05、minor_div（div≤5）算通过、`structurally_unidentifiable`
  才不可用——判定全在两个原脚本里，本入口只编排 + 校验；
- **不改数字**：21 条的所有 rms/acf/R̂/div 原样；汇总表 15/6/1 不变；
- **不改产物的语义**：`--check` 完全只读；部分模式只做"探查 + 不留痕"，不产生可引用结果；
- **不接生产**：这是 harness 侧工具，与 `bm_*.py` / `circuit_*.py` 无关；
- **不代删 progress**：强制重跑要人工删行，避免一键抹掉历史判定。

## 5. 验收

```
--check                    rc 0（G1/G3 全绿）
--dry / --limit 1 --dry    rc 0（计划正确、部分模式提示跳过阶段 2）
--limit 1（真实）          std_recheck.json rows 21→1→重建 21；汇总表 sha 不变
全量默认                   产物变化: 无（幂等）；G3 21 条 / 6 non_evaluable
负对照（篡改汇总表 1 处）   rc 1，报 non_evaluable 7 != 6
py_compile                 OK
```

回归：`repo_integrity 3/3`、`bmdrt check 18/18`、两个清单守卫 OK。生产代码零改动。
