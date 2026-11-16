# 高频电感分派修复 + adr 并入 nested 分支（2026-11-16）

两件独立的事，同一天落地，都不改采样内核与任何验收口径。

## 1. 3E 谱高频电感没加：族分派方向错了（用户指出）

### 现象

`out/3E-3_OER_exp3_0100mA`（`3E-3_OER_exp3_0100mA.csv`）拟合后：
- ③ Bode 相位高频端（1e4–1e5 Hz）实测翘到 **+40°**（感抗），拟合曲线卡在 0°
- ⑥ 残差虚部高频端掉到 −0.1 Ω 的系统性大偏差
- 产物 rms_rel 10.38%、acf 0.850，`family=nested`

### 根因

族预检理由写得清清楚楚：`高频感抗 2.31e+04–1e+05 Hz（8 点）需 include_L`。
但 `auto` 的族分派把**任何** `excluded` 都一律切到 nested。

问题在于：
- `l0_precheck` 是在 `include_L=False` 的 voigt 候选上跑的（`_voigt_family_verdict`
  的 `build_candidates(include_L=False)`），所以高频感抗谱必被判 excluded
- `include_L` 是 **voigt 的顶层槽位**，`bm_real.run` 在 `include_L=None`（默认）时
  本来就会调 `detect_L(f, Z)` 自动挂上 `jωL`（`bm_real.py:2714`）
- nested 的支路是**低频**感抗元件（R∥L 嵌在第二并联节点），高频极限是容性，
  根本没有高频电感接点

结论：这条谱本可留在 voigt 自动加电感，切族反而把电感需求丢掉了。

### 验证（只读探针 `hf_inductor_probe.py`，quick 档逐点 2% σ）

| 配置 | rms | acf | 高频残差 | 末点相位 fit/obs |
|---|---|---|---|---|
| **voigt K=1 + include_L** | **0.78%** | **0.432** | 1.8% | **40.5° / 40°** |
| nested（修复前产物） | 10.38% | 0.850 | 28.3% | −0.1° / 40° |

（K=1+L 的 R̂=1.83 是 quick 档采样问题，不影响结论；standard 档收敛。）

### 修复

`bmdrt.py` 新增 `_voigt_excluded_only_by_L(path)`：voigt 被判 excluded 时，
用两条确定性条件判断是否「只缺 L」——
1. `voigt_can_represent(f, Z, include_L=True)` 判 True；
2. `bm_real.detect_L(f, Z)` 判 True。

两条都成立就**不切族**，交给生产路径 `detect_L()` 自动挂电感；否则照常切 nested。
低频感抗谱（AZ31-4Li_150 等）在第 1 条就失败（打开 include_L 仍表达不了 R∥L），
不受影响。

重跑（quick 档）确认：`excluded 只因高频感抗需 include_L ⇒ 不切族`，
走 voigt K=2 +L，rms/σ 0.95/0.97、acf 0.479（M-b 参考线内）；
相位图高频端实测与拟合一起翘到 +40°，残差虚部回到零附近。

## 2. adr 并入 nested 的 adsorption 分支（用户裁决）

### 裁决

「adr 作为 nested 的一个 branch」——取消独立入口，统一走 `nested_fit`。

### 落地（方案：薄接线，不动采样内核）

- `nested_fit.run(branch="adsorption")` 可用；底层复用 `bm_top.fit_adr/z_adr`
  （与 `fit_top` 同构的采样器/先验/σ 口径），`layers` 强制 1（单节点三支路，
  无第二层）
- `param_table` 用 `T.ADR_KEYS`；`derived_quantities` 带 `branch_kind="adsorption"`
- 电路图按 branch 分派：adsorption → `bm_circuit_adr.draw_adr`，其余 →
  `bm_circuit_nested.draw_nested`
- 产物后缀 `.adr.*`，与 nested 的 `.nested.*` 互不覆盖（auto fallback 两份都保留）
- 删除 `adr_fit.py` / `cmd_adr.py`（`bmdrt adr` 子命令随之消失）
- `bmdrt.py::_run_adr_branch` 改调 `nested_fit.run(branch="adsorption")`

**没有做**的事：`bm_top.VALID_BRANCHES` 不加 adsorption，`z_nested` / `_make_model`
/ `_loglik_chained` 一行不改。ADR 不是 `z_nested` 的两层嵌套骨架，硬塞进去会让
`layers` 语义混乱；把它放在 `nested_fit` 这一层分派是最小改动。

`bmdrt compare` 里的 adr 候选仍走 `bm_select.fit_adr`（与删除的文件无关），
保留 voigt/nested/adr 三族并列。

### 验证

- `bmdrt nested archive/AZ31-12Li_5.csv --branch adsorption --budget quick`
  → rms 7.10%，产物 `.adr.json/.adr.png/.adr_circuit.png/.adr_curve.csv` 齐全，
  参数 Rs=104.9/Q=73.1/Rct=114.7（与整合前的独立 adr 一致）
- `bmdrt auto archive/AZ31-12Li_5.csv --quick` → nested 5.80% + adr 7.10% 两份产物
  都保留，`meta.family=adr`，nested 结果在 `meta.nested` 子键
- `test_entry_parity.py` 177/177（新增 T11 十条断言守整合边界）

## 3. 未闭合

- auto 的 adr fallback 条件是 `R̂>1.05 或 div>0`。12Li_5 这次 nested R̂=1.004（收敛）
  仅因 1 个发散就被 adr 覆盖，而 adr rms 7.10% 反而比 nested 5.80% 差。
  这是旧逻辑，本轮未动；后续可考虑「只在 R̂ 不达标时才 fallback」。
- M-b（acf≤0.50）判据本身被 M-B-SIM-01 证伪（R27），本轮两件事都不碰验收线。
