# 21 条按层 0 重排（只读汇总）

> 由 `layer0_summary.py` 从 `out/batch21/results.json` 与
> `out/std_recheck/std_recheck.json` 合并生成，**不改任何数字**。
> 层 0 一律按 **standard** 预算判（`R̂<=1.05` 且 `div==0`）——quick 的通过不作数，
> 实测存在 quick 干净、standard 反而不干净的翻转。不过者标 `non_evaluable`，不给 rms。
> `acf` 未采集 standard 版本，本表不报（M-b 判据待带 acf 的重跑）。

## 1. 逐条

| 类 | 谱 | 候选 | quick R̂/div | std R̂/div | 层 0 | rms% |
|---|---|---|---|---|---|---|
| 合成 | `ecm_04375_RCPE_RCPE_RCPE.csv` | voigt/K=2 | 2.200/41 | 1.853/0 | non_evaluable | — |
| 合成 | `ecm_04382_Rs_Ws.csv` | nested/L=2 | 1.102/75 | 1.002/0 | pass_standard | 0.08 |
| 合成 | `ecm_05687_L_R_RCPE.csv` | nested/L=2 | 986.040/9 | 176.7/0 | non_evaluable | — |
| 腐蚀 | `R1-[R2,P3]-[1000.0, 2512134.2009` | nested/L=2 | 4.295/121 | 1.004/2 | pass_standard | 8.46 |
| 腐蚀 | `R1-[R2,P3]-[1000.0, 5512287.3909` | nested/L=2 | 1.008/1 | 1.004/1 | pass_standard | 4.38 |
| 腐蚀 | `R1-[R2,P3]-[1000.0, 8503199.2236` | nested/L=1 | 1.006/0 | 1.001/0 | pass_standard | 6.42 |
| 钝化 | `[P1,P2-R3]-R4-[2506431.227387929` | nested/L=1 | 1.002/0 | 2.297/557 | non_evaluable **⚠翻转** | — |
| 钝化 | `[P1,P2-R3]-R4-[5502377.557033613` | voigt/K=1 | 1.006/0 | 1.002/0 | pass_standard | 69.96 |
| 钝化 | `[P1,P2-R3]-R4-[8500629.506318517` | voigt/K=2 | 1.007/0 | 1.003/0 | pass_standard | 70.51 |
| 涂层 | `R1-[P2,R3-[P4,R5-P6]]-[150.0, 25` | nested/L=2 | 53.141/1 | 16.874/0 | non_evaluable | — |
| 涂层 | `R1-[P2,R3-[P4,R5-P6]]-[150.0, 55` | voigt/K=2 | 1.843/0 | 1.831/0 | non_evaluable | — |
| 涂层 | `R1-[P2,R3-[P4,R5-P6]]-[150.0, 84` | nested/L=2 | 25.976/0 | 92.5/0 | non_evaluable | — |
| 2E | `2EMC1-3_exp1_0100mA.csv` | nested/L=2 | 1.289/0 | 1.020/0 | pass_standard | 4.03 |
| 2E | `2EMC2-2_0350mA.csv` | nested/L=2 | 1.269/0 | 1.018/0 | pass_standard | 3.83 |
| 2E | `2EP-2_exp1_0750mA.csv` | voigt/K=2 | 1.844/0 | 1.003/0 | pass_standard | 8.95 |
| 3E | `3E-2_HER_exp1_0300mA.csv` | nested/L=2 | 1.007/0 | 1.006/0 | pass_standard | 0.45 |
| 3E | `3E-2_full_electrolysis_cell_exp2` | nested/L=2 | 4.433/0 | 1.010/0 | pass_standard | 0.33 |
| 3E | `3E-3_full_electrolysis_cell_exp1` | nested/L=2 | 1.021/0 | 1.004/0 | pass_standard | 0.26 |
| AZ31 | `AZ31-12Li_150.csv` | nested/L=2 | 1.020/0 | 1.012/0 | pass_standard | 9.82 |
| AZ31 | `AZ31-4Li_150.csv` | nested/L=2 | 1.100/0 | 1.026/0 | pass_standard | 11.05 |
| AZ31 | `AZ31-8Li_50.csv` | nested/L=2 | 1.133/12 | 1.037/0 | pass_standard | 5.74 |

## 2. 按类

| 类 | 总 | 通过 | non_evaluable | 翻转 | 可用 rms 中位 | 范围 | 备注 |
|---|---|---|---|---|---|---|---|
| 合成 | 3 | 1 | 2 | 0 | —（n=1） | 0.08 | **基准不合格**：3 条里 2 条 `non_evaluable`；`ecm_05687` 在**生成阶段**就已 `R̂=2.512`、`触顶=91%`（σ 先验比真噪声大 4.75 dec）；剩 1 条不足成组。 |
| 腐蚀 | 3 | 3 | 0 | 0 | 6.42% | 4.38–8.46 |  |
| 钝化 | 3 | 2 | 1 | 1 | 70.24% | 69.96–70.51 | 2 条可用（69.96 / 70.51），1 条在 standard 下**翻转**为不可辨识（原 quick 干净）。 |
| 涂层 | 3 | 0 | 3 | 0 | — | — | **基准不合格**：3 条全 `non_evaluable`（std R̂ 16.87 / 1.83 / 92.50）；真值 5 CPE 双层嵌套超出 nested/L<=2 与 voigt/K<=2；`22.04%` 作废。 |
| 2E | 3 | 3 | 0 | 0 | 4.03% | 3.83–8.95 |  |
| 3E | 3 | 3 | 0 | 0 | 0.33% | 0.26–0.45 |  |
| AZ31 | 3 | 3 | 0 | 0 | 9.82% | 5.74–11.05 |  |

**合计**：21 条 → standard 下层 0 通过 **15** 条，`non_evaluable` **6** 条，其中 **1** 条与 quick 判定相反。

## 3. 读法

1. **quick 的通过不可作为层 0 依据。** 钝化 #1 在 quick 下 `1.002/0`、
   standard 下 `2.297/557`。机制未确证，但 `div=557` 与多峰后验一致：
   300+300 可能把多峰采成「看起来收敛」。**所有 quick 得出的"收敛"都要复核。**
2. **只有 `rms` 列能进摘要**，且全部是 standard 口径，可直接横向比较（类内）。
3. **合成与涂层两类不能作为基准**：一个是 σ 先验与数据脱钩，
   一个是真值拓扑超出表达力。它们不能与腐蚀/钝化并列。
4. **不要算跨类中位数**：类别之间的失效机制不同，平均没有意义。
5. 层 0 通过 ≠ 模型好。它只说明这次 run 的数字可引用，不说明族选对了。
