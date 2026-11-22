# 层 0 门到底在哪一层缺失（2026-11-22）

> 起因：21 条里 adr 有 10 次采样不过却仍参与比较。查"排名时有没有过滤
> `R̂>1.05` / `div>0` 的候选"。**答案：库层有、落盘标注有、分析脚本没有。**

## 1. 三处定位

| 位置 | 有没有门 | 证据 |
|---|---|---|
| `bm_select._family_state`（族内选优） | **有，而且比"过滤"更细** | `_quality` 走 `GUARD_A`（rhat/div_frac/ess/loo…）；剔除后若被剔除者 elpd 高于参赛最优 + `DEGRADE_MARGIN`，整族记 `degraded`、输出**并列** |
| `circuit_runner.to_observation` | **有标注，不裁决** | `passed = rhat<=1.05 and div==0` → `sampling_status=passed/failed`；`interpretation=None if passed else "descriptive_only"` |
| `batch_fit_21.py:159` | **没有** | `good = [c for c in cands if ok and rms is not None and mode=="fixed"]` → `min(good, key=rms)`，**完全不看 `rhat`/`div`** |

`bm_select` 的那段注释值得原文引一次，它说明"过滤"这个做法本身有坑：

> 旧实现只做「剔除不合格 → 取剩余里 elpd 最大」，这在候选池大小不对称时会制造
> 跨族偏差：例如 voigt 的 K=2 因 R̂=1.84 被剔除后，族内只剩 K=1 参赛，nested 却
> 仍满配参赛，于是 Δelpd 被凭空抬高 65 nats，把一条纯 voigt 谱判成 nested。

所以**正确的失败处置是"并列"，不是"换一个更差的"**。这一点 `bm_select` 已经做对了；
`batch_fit_21.py` 的 `min(rms)` 既没有门、也没有"并列"这个出口。

## 2. 21 条的实测影响（纯离线，不重跑 MCMC）

`layer0_gate_probe.py` 用 `results.json` 里每个候选自带的 `rms/rhat/div` 重选 best：

| 门宽 | 换 best | 无可用 | 换后 rms 变化（中位） |
|---|---|---|---|
| `div==0`（严格） | **14 / 21** | 1 | +1.47 pp |
| `div<=5`（宽容） | 13 / 21 | 1 | +0.97 pp |
| 只看 `R̂` | 13 / 21 | 1 | +0.97 pp |

门宽只影响 1 条（`R̂` 通过但 `div>5`），所以结论对阈值不敏感。

**旧口径 best 采样不过：14 / 21 = 67%**，按类：

| 类 | 旧 best 采样不过 | 加门后换 best | 无可用 |
|---|---|---|---|
| 合成 | **3 / 3** | 3 | 1（`ecm_05687` R̂=986） |
| 腐蚀 | 2 / 3 | 2 | 0 |
| 钝化 | **0 / 3** | 0 | 0 |
| 涂层 | **3 / 3** | 3 | 0 |
| 2E | **3 / 3** | 3 | 0 |
| 3E | 1 / 3 | 1 | 0 |
| AZ31 | 2 / 3 | 2 | 0 |

典型个案：

| 谱 | 旧 best（按 rms） | `R̂` / `div` | 加门后 best | rms |
|---|---|---|---|---|
| `ecm_05687` | nested/L=2 0.00% | **986.0** / — | **无可用** | — |
| 涂层 #1 | nested/L=2 22.04% | **53.1** / — | nested/L=1 | 32.49% |
| 涂层 #3 | nested/L=2 10.25% | **26.0** / — | voigt/K=1 | 22.69% |
| 3E-2_full | nested/L=2 0.33% | **4.43** / — | nested/L=1 | 0.81% |
| 腐蚀 #1 | nested/L=2 6.31% | **4.29** / — | voigt/K=1 | 7.78% |
| 腐蚀 #2 | nested/L=2 4.56% | 1.01 / **div=1** | nested/L=1 | 6.98% |

## 3. 但不能直接把这 14 条按门重选——会误杀 7 条

`STD_RECHECK.md` 已证：这 14 条里有 **7 条是 `budget_limited`**（standard 2000+1000
下 `R̂≤1.05`、`div≤2`），加预算就好了。所以：

- **硬门 `div==0` 直接把 quick 结果判死 ⇒ 误杀 7 条**
- 硬门 `div<=5` 少杀 1 条，仍然误杀 6 条

正确的处置顺序是**两阶段**，而这正是已经做过、但没写进脚本的流程：

```
quick 跑 → 采样不过 ⇒ 标 pending_recheck（不是弃用、不是降级，是"待复核"）
        → standard 复核 ⇒ 通过则允许参赛
                        → 仍不过 ⇒ non_evaluable（rms/acf 一律不引用）
```

`std_recheck.py` + `layer0_summary.py` 就是这两阶段的人工执行版。

**缺口不在判据，在自动化**：`batch_fit_21.py` 跑完不会提示"你的 best 有 67% 采样不过"，
所以必须靠人记得去复核。这次是我事后才发现。

## 4. 一个必须区分的两件事

`std_recheck` 复核的是 **`results.json` 里已选中的那一个 best**（rms 口径），
不是"加门重选后谁最优"。所以它证明的是：

- ✅ "这 14 条 best 候选里有 7 条加预算能通过"（候选级）
- ❌ 不是 "加门重选后 best 会变成谁"（选择级）——那需要**所有候选**的 standard 版，
  目前只跑了 best 一个

`layer0_gate_probe.py` 补的是后者，但用的是 quick 的 `rhat/div`，所以它回答的是
"**如果当时就按 quick 的采样结果硬选**会怎样"，那是下界估计，不是最终结论。

## 5. 结论与建议

| 项 | 判断 |
|---|---|
| `bm_select` 要不要改 | **不改**。它已有门，且用 `degraded` 并列避免了"剔除后换更差的"这个陷阱 |
| `circuit_runner` 要不要改 | **不改**。`sampling_status` / `interpretation` 已正确落盘 |
| `batch_fit_21.py` 要不要改 | **建议加，但只是标注**：`min()` 前把不过的候选标 `pending_recheck`，发现"全部不过"时**不要**降级挑一个，而是记 `non_evaluable` |
| 21 条既有结果 | **不回改**。`LAYER0_SUMMARY.md` 已经是正确口径（standard 复核 + `div≤5`），本文件只是补充"脚本为什么没自动做到" |

**一句话**：层 0 门从来没有缺失，缺的是"脚本自己会承认它没过门"。
`min(rms)` 这种写法会让一个没收敛的候选安静地当上 best，而这件事在 21 条里
发生了 14 次。

## 6. 复现

```
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/layer0_gate_probe.py
# -> out/layer0_gate/gate.json（21 行，逐候选 old/new best 与判据版本）
```
