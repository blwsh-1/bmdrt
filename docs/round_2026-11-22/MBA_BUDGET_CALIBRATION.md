# M-b 外部验证的预算标定：quick 不够，但 standard 也不是单调改善

> 2026-11-22。用户问「quick 档预算会不会不够，要不要直接 s 档」。
> 在 batch21 的 6 条 2E/3E 谱上把 quick 与 standard 并排跑，默认先验口径
> （与 §14 一致，不传 `zmax_override`）。脚本 `mba_budget_calibration.py`。

## 0. 结论

| 问题 | 答案 |
|---|---|
| quick 够不够？ | **不够**。6 条里 2 条 nested R̂ > 1.05（2.807 / 2.508），其中一条还有 83 次 div |
| standard 是不是直接解？ | **基本是，但不单调**：修好那 2 条，同时**新引入 1 条**（`2EP-2` nested R̂ 1.004 → **7.310**） |
| 净效果 | quick 采样不过 2/6；standard 达标 5/6；被修好 2/2 |
| 正式方案 | **standard 主档 + 失败自动 deep 复核 + 失败谱保留不剔除** |

## 1. 已有证据：quick 的 acf 不可信

`batch21` 的 6 条 2E/3E、nested `layers=2` 候选：

| 谱 | R̂ | div |
|---|---|---|
| 2EMC1-3_exp1_0100mA | 1.289 | 0 |
| 2EMC2-2_0350mA | 1.269 | 0 |
| 3E-2_full_electrolysis_cell | **4.433** | 0 |

3/6 采样不过。更早的全 15 条（layers=2）里 **9/15** 有 `div>0` 或 `R̂>1.05`，
而 `ok` 字段竟然 **15/15 全为 True**——`batch_fit_21.py` 的 `ok` 只表示
`fit_candidate` 没抛异常，**不检查采样质量**。用这批 acf 做判据检验会掺入
未收敛谱的虚假结构（§14 的 `12Li_5`：未收敛谱 legacy acf 0.896 vs F1 0.713）。

## 2. 标定实测：6 条 × 2 档 × 2 族

| 谱 | quick nested acf / R̂ / div | standard nested acf / R̂ / div | 变化 |
|---|---|---|---|
| 2EMC1-3_exp1_0100mA | 0.186 / **2.807** / 83 | 0.211 / 1.037 / 0 | **修好** |
| 2EMC2-2_0350mA | 0.400 / 1.004 / 0 | 0.405 / 1.003 / 0 | 均无问题 |
| 2EP-2_exp1_0750mA | 0.347 / 1.004 / 0 | 0.421 / **7.310** / 0 | **新引入失败** |
| 3E-2_HER_exp1_0300mA | 0.448 / 1.019 / 0 | 0.448 / 1.002 / 0 | 均无问题 |
| 3E-2_full_electrolysis_cell | 0.685 / **2.508** / 0 | 0.673 / 1.008 / 0 | **修好** |
| 3E-3_full_electrolysis_cell | 0.812 / 1.031 / 0 | 0.806 / 1.005 / 0 | 均无问题 |

voigt K=1 在两档下全部达标（R̂ ≤ 1.021），问题只在 nested。

`2EP-2` 这条尤其值得记：**acf 从 0.347 变成 0.421**，跨过了 0.50 吗？没有，仍达标；
但它说明**同一个谱换预算可以把后验从收敛搞到 R̂=7.3**。这与
`B2_BIAS_ATTRIBUTION.md` 的「R̂ 在该配方上不是先验的稳定函数」、
`NESTED_RHAT_ATTRIBUTION.md` 的「二维先验平片让链随机落点」是同一现象的不同切面。

⇒ **不能用“换一档预算”当作一劳永逸的修复。**

## 3. 口径边界（这条最容易出错）

`batch21` 的候选 acf 全部来自 `zmax_override = 5·|Z(f_max)|` 的 **fixed** 口径；
§14 用**默认**先验（不传 override）。本标定**两档都用默认口径**。

所以表中 acf **不能**与 `MBA_EXTERNAL_VALIDATION.md` §2 的 batch21 数字直接比
（那里是 fixed 口径）。先验定标已证 override 会同时缩放 `W`/`Q`/`R1`。

## 4. 正式方案

1. **主预算 standard**（`2000/1000/2/ta0.98/depth10`）：6 条里 5 条达标，且是 §14 的正式档；
2. **失败自动 deep**（`4000/2000/2/ta0.97/depth11`）：只对 standard 下仍失败的族加跑；
3. **失败谱保留不剔除**，标 `sampling_ok=False` 并单独统计。

**执行中又发现一条（2026-11-22）**：deep 档在 **voigt K1** 上会触发生产代码的
崩溃——`bm_real.py:1232 grade()` 里 `r["tds"]` 为 `None`，`frac_cap` 取值时
`TypeError`。这是 deep 预算下某个诊断没落盘导致的，**本轮不改生产代码**。
正式脚本因此把 deep 限制在**失败那一族**，并 `try/except` 捕获：救不回来就
保留 standard 结果并标 `sampling_ok=False`，绝不丢谱。

第 3 条是硬约束。若把采样失败的谱悄悄丢掉，留下的就是“容易拟合”的谱——
那正是 M-b 可能通过的那一类，**结论会被选择偏差完全决定**。

成本实测：6 条 × 2 档 × 2 族 = 24 次拟合耗时 **7m42s**（约 19s/次混合）。
40 条 × 2 族 standard ≈ 40 分钟。

## 5. 复现

```bash
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-22/mba_budget_calibration.py \
    --json out/mba_budget_calib.json
```
