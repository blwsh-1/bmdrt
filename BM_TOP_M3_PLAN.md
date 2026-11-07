# `fit_top()` 接口设计与 M3 预实验清单（v2，已按审查冻结）

> 状态：**研究性原型**。不解除 `NESTED_TOPOLOGY_PROJECT.md` §7.0 的架构冻结；
> 不建设 `bm_families`；不调用、不修改 `bm_core.make_model`。
>
> **本文件是"参数恢复与可辨识性预实验"清单，不是完整 M3 验收。**
> 完整 M3 还包含"纯 Voigt 生成（检验过度拟合）"与"混合边界生成（检验错误切换）"
> 两类合成谱，见 §7。第一阶段通过**不得**宣布 M3 完成。

---

## 1. 只读审查材料（v2 修订）

### 1.1 `bm_core.fit` 的噪声模型

```python
s_re ~ LogNormal(log(0.01·zmax), 1.0)
s_im ~ LogNormal(log(0.01·zmax), 1.0)
numpyro.sample("obs_re", Normal(Zt.real, s_re), obs=Zr)
numpyro.sample("obs_im", Normal(Zt.imag, s_im), obs=Zi)
```

- 实部、虚部**各自一个 σ**，不是复高斯；
- `sigma_floor>0` 时 `s = base + floor`；
- 显式传 `sigma=(s_re, s_im)` 时固定不采。

### 1.2 LOO 口径（v2 修正）

`bm_real._paired_loo_diff(a, b)` **只读取** `a["pointwise_elpd"]` 与
`b["pointwise_elpd"]`，检查长度/形状/有限性后算逐点配对差。因此：

- `pointwise_elpd` 必须有，且必须是**每个频率点一个联合观测**的 LOO 值：

  ```
  ℓ_j = log p(Z'_j | θ) + log p(Z''_j | θ)
  ```

  **不能**把实部/虚部拆成 2N 个独立观测（`bm_real.loglik_matrix` 已明确规定
  同频实虚必须联合留出）。
- `elpd` 不是 `_paired_loo_diff` 的直接依赖，但报告与有效性判断要用，保留。
- `elpd_se`、`pareto_k`、`idata` 保留，供跨族比较与 LOO 质量审查。

### 1.3 `bm_top.py` 现有接口（M1）

```python
omega_ref(f); z_cpe(f, Q, n, f_ref); z_rl_parallel(f, R, tau);
z_rl_series(f, R, L); z_nested(f, p, f_ref, branch, layers);
j_z_nested(omega, p, w_ref, branch, layers); param_spec()
```

自检 8/8（`python bm_top.py`）。

---

## 2. `fit_top()` 接口（候选 A，已批准）

### 2.1 签名

```python
fit_top(f, Z, *, branch="parallel", layers=2,
        n_warmup=600, n_samples=1200, seed=0, num_chains=4,
        target_accept=0.9, max_tree_depth=10,
        sigma=None, sigma_floor=0.0, init_values=None, progress=True) -> dict
```

`num_chains` 默认 **4**（审查要求：单链不能判 R̂）。

### 2.2 返回字段（用 `s`，与 `fit_one` 对齐）

```python
{
  "s": ...,                # 后验参数字典，与 fit_one 同名
  "elpd": ..., "elpd_se": ...,
  "pointwise_elpd": ...,   # 每频点一个，实虚联合
  "pareto_k": ..., "idata": ...,
  "sigma_re": ..., "sigma_im": ...,
  "divergences": ..., "rhat_max": ..., "ess_min": ...,
  "branch": ..., "layers": ..., "f_ref": ...,
}
```

诊断直接调 `bm_experiments` / `bm_real._paired_loo_diff`，不重写。

### 2.3 不做的

不改 `make_model` / `fit_one` / `select_k`；不建注册表；不做跨族选择。

---

## 3. 参数与先验（冻结）

### 3.1 表

| 参数 | 先验 | 角色 |
|---|---|---|
| `Rs` | `LogNormal(log(0.2·zmax), 1.0)` | 形状参数 |
| `R1` | `LogNormal(log(0.1·zmax), 2.0)` | 形状参数 |
| `R2` | `LogNormal(log(0.1·zmax), 2.0)` | 形状参数 |
| `Q1` | `LogNormal(log(zmax), 2.0)` | 形状参数（窗口幅值，非 `Y`） |
| `n1` | `Uniform(0.5, 1.0)` | 形状参数（有硬边界） |
| `Q2` | `LogNormal(log(zmax), 2.0)` | 形状参数 |
| `n2` | `Uniform(0.5, 1.0)` | 形状参数（有硬边界） |
| `R_ad` | `LogNormal(log(0.1·zmax), 2.0)` | **优先恢复**（branch=parallel） |
| `tau_ad` | `LogNormal(log(1/ω_ref), 2.0)` | **优先恢复** |
| `R_L` | `LogNormal(log(0.1·zmax), 2.0)` | **优先恢复**（branch=series） |
| `tau_L` | `LogNormal(log(1/ω_ref), 2.0)` | **优先恢复**；`L_ad = R_L·tau_L` 为**派生量** |

### 3.2 与 `bm_top.param_spec()` 的统一

`bm_top.param_spec()` 原先把 `L_ad` 写成 `LogNormal(log(R_L·τ_ref), 2)`，
与本表不一致。**已统一为**：`tau_L` 是采样参数，`L_ad` 是确定性派生量；
`z_nested` 仍接受 `p["L_ad"]`（由 `fit_top` 派生后传入），正演接口不变。

### 3.3 冻结规则

M3 第一轮跑完后，**只允许因量纲写错改一次**，并记录改动；不允许为了
"反演更准"逐步收窄先验。

---

## 4. 噪声口径（v2 修正）

### 4.1 合成谱噪声：固定标准差、实虚独立

第一轮统一用：

```
ε_re, ε_im ~ N(0, σ²),   σ = 0.02·zmax（或 0.05·zmax）
```

即 **2% / 5% 是相对整条谱最大阻抗的固定标准差**，不是逐点乘 `|Z(f)|`。
这与 `fit` 的同方差 Normal 似然一致。

### 4.2 为什么不用 `0.02·|Z|`

`evidence/nested_topology/theoretical_bound.py` 的合成谱按逐点 `0.02·abs(Z)`
加噪，属于**异方差**噪声，与采样器的同方差似然不一致。它继续用于
**表达能力验证**没问题，但**不能**作为本轮采样器的噪声标定生成器。

### 4.3 无噪声：不跑自由 σ 的 NUTS

精确无噪声数据 + 自由 `s_re/s_im` 会让 σ 向 0 塌缩（`bm_core` 已记录：
漏斗 + 树深饱和）。因此：

- 无噪声只做**确定性正演交叉验证**（档 1）；
- 若要在无噪声上验证参数恢复，必须用**固定小 σ**（显式传 `sigma=`），
  并标注"非统计推断，仅数值恢复"。

---

## 5. 独立生成器（v2 修正）

### 5.1 覆盖矩阵

`theoretical_bound.py:model_b` 只实现
`Rs + CPE1∥(R1 + CPE2∥(R2 + (R_ad∥L)))`，**不支持** `R_L + jωL` 与 `layers=1`。
因此生成器必须补齐为独立实现：

| 生成器 | parallel | series | layers=1 | layers=2 |
|---|---|---|---|---|
| `gen_parallel_L2` | ✅ | — | — | ✅ |
| `gen_series_L2` | — | ✅ | — | ✅ |
| `gen_*_L1` | ✅ | ✅ | ✅ | — |

实现方式：逐节点导纳递推，**手写**，不 import `bm_top.z_nested`。
用 `Y = 1/(Q·ω_ref^n)` 从 `Q` 换算。

### 5.2 与拟合器的交叉验证

档 1 用"生成器 → `bm_top.z_nested`"的相对误差 <1e-10。两边同时写错的概率低，
且档 2/3 会暴露（真值反演不回来）。

---

## 6. M3 预实验验收（三档，v2 修订）

### 6.1 档 1：结构正确（确定性）

| 用例 | 判据 |
|---|---|
| 1-1 生成器 vs `z_nested`（parallel L2 / series L2 / L1） | rel err <1e-10 |
| 1-2 极限 `ω→0/∞` | parallel `Rs+R1+R2`/`Rs`；series `+R_L`/`Rs` |
| 1-3 无源性 | `min Re ≥ 0` |

### 6.2 档 2：参数恢复（分三类，**不要求全部精确还原**）

| 类别 | 参数 | 判据 |
|---|---|---|
| **优先恢复** | `R_ad/tau_ad`（parallel）或 `R_L/tau_L`（series） | 无噪声（固定小 σ）中位数误差 <1%；2% 噪声 <4%；5% 噪声 <8% |
| **形状参数** | `Rs/R1/R2` | **不要求**精确还原；检查：① `Rs` 剖面平坦度 ② 窗口截断漂移（砍高频点后是否漂）③ 与 `Q1/n1` 的相关。与 `NESTED_TOPOLOGY_PROJECT.md` §4.6 的结论保持一致 |
| **组合参数** | 无法分别辨识者 | 报告可辨识的组合（如 `R2·Q2^n`、`R_L·tau_L`）及其后验区间 |

> 立项书 §9 的要求保留：`Rs` 必须单独做剖面与平移窗口检查，**不能**因为
> 后验 R̂ 好就说它可辨识。

### 6.3 档 3：统计可辨识

| 指标 | 判据 | 备注 |
|---|---|---|
| R̂ | <1.01 | **至少 4 链** |
| ESS | >200 | — |
| divergences | <0.1% 采样步 | #2 口径 |
| 边界贴边率 | **分开定义**：`n1/n2`（Uniform）检查是否贴 0.5/1.0，阈值 <5%；`LogNormal` 参数无硬边界，改报"先验主导程度"= 后验 90% 宽度 / 先验 90% 宽度，接近 1 说明数据没约束 | v2 修正 |
| 成对相关 `|ρ|>0.95` | 逐对列出 | — |
| 多峰 | 直方图/KDE 峰数 >1 即记录 | — |
| 拓扑退化 | `layers=1` vs `2` 的 LOO 差 | — |

### 6.4 覆盖率口径（v2 修正）

3 组参数 × 单 seed 只能说"3/3 覆盖真值"，**不能**声称 90% 区间覆盖率达标。
若要报告覆盖率：

- 每种参数类型跑**多个独立噪声 seed**（第一轮先 5 个，作探索）；
- 报告 `覆盖次数 / 总数`，并注明样本量；
- 不把 3/3 写成统计结论。

---

## 7. 与完整 M3 的边界

`NESTED_TOPOLOGY_PROJECT.md` §8 的完整 M3 含三类合成谱：

| 类型 | 检验 | 本文件 |
|---|---|---|
| 纯 nested 生成 | 自身参数恢复 | ✅ 本阶段（§6） |
| **纯 Voigt 生成** | nested 是否过度拟合普通容性谱 | ❌ 下一阶段 |
| **混合边界生成** | 退化/截断/噪声下是否错误切换 | ❌ 下一阶段（需 M4 选择器） |

所以本文件通过 = **M3 的参数恢复与可辨识性预实验完成**，不等于 M3 完成。

---

## 8. 失败记录

每次运行输出一行（CSV 或表），成功与失败都留：

```
case, branch, layers, noise, seed, n_chains, rhat_max, ess_min, div_frac,
max|rho|, rho_pair, width90_rel, prior_dom_ratio, n_modes,
at_bound_frac_uniform, median_err, covered
```

- 失败不删、不换 seed 掩盖；换 seed 必须保留原结果；
- 报告先列失败，再列通过。

---

## 9. 不在本阶段范围

- 不建 `bm_families`、不改 `make_model` / `fit_one` / `select_k`；
- 不做 `select_across_families`；
- 不做真实 AZ31 的 M4 回放；
- 不做纯 Voigt / 混合边界两类合成谱（属完整 M3 后续）；
- 不调先验直到结果好看。

---

## 10. 已冻结的决定（审查结论）

| 决策项 | 决定 |
|---|---|
| 采样架构 | 候选 A：独立 NumPyro + NUTS |
| 生产代码 | 不动 |
| `bm_families` | 暂不建设 |
| LOO | 复用 `_paired_loo_diff`；`pointwise_elpd` 同频联合 |
| 返回参数字典 | `s`，与 `fit_one` 对齐 |
| 似然 | 实虚分离 Normal |
| σ 先验 | 完全沿用 `LogNormal(log 0.01·zmax, 1)` |
| series 支路 | 采 `R_L` / `tau_L`，`L_ad = R_L·tau_L` 派生 |
| 2% / 5% 噪声 | 固定标准差（相对 `zmax`）、实虚独立高斯 |
| 4% / 8% | 保留为**工程验收目标**，非普适标准 |
| 无噪声 | 不跑自由 σ 的 NUTS；只做确定性正演，或固定小 σ |
| 正式 R̂ 验收 | ≥4 链 |
| 参数恢复 | 分优先/形状/组合三类，不要求全部单独恢复 |
| M3 范围 | 本阶段只做参数恢复预实验，不冒称完整 M3 |
