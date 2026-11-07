# `fit_top()` 接口设计与 M3 验收清单（草案，待审查）

> 状态：**研究性原型，不是生产模型接入**。不解除 `NESTED_TOPOLOGY_PROJECT.md`
> §7.0 的架构冻结；不建设 `bm_families`；不调用、不修改 `bm_core.make_model`。
>
> 本文只做两件事：① 把现有采样接口摊开供审查；② 给出 `fit_top()` 的接口
> 设计与 M3 三档验收清单。**尚未写 `fit_top()` 代码**。

---

## 1. 只读审查材料：现有采样接口

### 1.1 `bm_core.make_model`

```python
make_model(K=10, alpha=1.0, sigma=None, include_L=False, pad_dec=1.5,
           zeta_fixed=None, phi_lo=0.5, diffusion=None, sigma_floor=0.0,
           adsorption=None)
```

内部把 `Zt` 逐项相加（`R_inf + ΣHN + jωL + Z_W + Z_ad`），先验写在函数体里。
**nested 无法通过它表达**——这正是不能复用的原因。

### 1.2 `bm_core.fit`

```python
fit(f, Z, K=10, alpha=1.0, sigma=None, include_L=False,
    n_warmup=600, n_samples=1200, seed=0, progress=True, num_chains=1,
    target_accept=0.9, zeta_fixed=None, init_values=None,
    max_tree_depth=10, extra_fields=(), diffusion=None,
    sigma_floor=0.0, adsorption=None)
```

**噪声模型（跨族比较必须一致，这是关键）**：

```python
s_re ~ LogNormal(log(0.01·zmax), 1.0)
s_im ~ LogNormal(log(0.01·zmax), 1.0)
numpyro.sample("obs_re", Normal(Zt.real, s_re), obs=Zr)
numpyro.sample("obs_im", Normal(Zt.imag, s_im), obs=Zi)
```

- 实部、虚部**各自一个 σ**，不是复高斯；
- `sigma_floor>0` 时 `s = base + floor`，`base ~ LogNormal(log 0.01·zmax, 1)`；
- 显式传 `sigma=(s_re, s_im)` 时固定不采。

### 1.3 后处理与诊断（可复用的基础设施）

| 函数 | 作用 | `fit_top` 能否复用 |
|---|---|---|
| `bm_core.samples_to_np` | 后验 dict 转 numpy、还原 `pi` | 部分（nested 无 `v`/`pi`） |
| `bm_real._paired_loo_diff` | 逐点 LOO 配对差 `Δ`/`dse` | **能**（只要 res 里有 `pointwise_elpd`） |
| `bm_real.component_review` | 参数相关/贴边/残差 | 部分（字段名绑定 Voigt） |
| `bm_experiments.acf_max` / `resid_bins` | 残差统计 | **能**（纯数组函数） |
| `bm_bench` 的 R̂/ESS/divergence 汇总 | 采样诊断 | **能**（读 samples dict） |

### 1.4 `bm_top.py` 现有接口（M1 已交付）

```python
omega_ref(f)                       # 2π√(f_min·f_max)
z_cpe(f, Q, n, f_ref=None)         # Q = |Z(ω_ref)|
z_rl_parallel(f, R, tau)           # R∥L
z_rl_series(f, R, L)               # R+L
z_nested(f, p, f_ref=None, branch="parallel", layers=2)
j_z_nested(omega, p, w_ref, branch="parallel", layers=2)
param_spec()                       # 参数 → 含义/单位/先验建议
```

自检 8/8（`python bm_top.py`）：CPE 幅值参数化、逐节点一致、两种支路极限、
无源性、`n=1` 退化、`layers=1` 退化、极端参数×18 dec、jax 一致性。

---

## 2. `fit_top()` 接口设计（三个候选，推荐 A）

### 候选 A（推荐）：独立 NumPyro 模型 + 复用诊断/LOO 基础设施

```python
fit_top(f, Z, *, branch="parallel", layers=2,
        n_warmup=600, n_samples=1200, seed=0, num_chains=1,
        target_accept=0.9, max_tree_depth=10,
        sigma=None, sigma_floor=0.0, init_values=None, progress=True)
    -> dict
```

返回字段（对齐 `fit_one` 的 res，便于 `_paired_loo_diff` 直接用）：

| 字段 | 内容 |
|---|---|
| `samples` | dict，参数名 → 后验数组（`Rs/R1/R2/Q1/n1/Q2/n2` + 支路参数） |
| `elpd` / `pointwise_elpd` | PSIS-LOO（用 arviz，同 `fit_one` 口径） |
| `sigma_re` / `sigma_im` | 后验中位数 |
| `res_re` / `res_im` | 残差 |
| `divergences` / `rhat_max` / `ess_min` | 采样健康 |
| `branch` / `layers` / `f_ref` | 拓扑元信息 |

- **先验与噪声模型完全照抄 §1.2**（同一 `LogNormal(log 0.01·zmax, 1)`、
  实虚分离 `Normal`），否则跨族 `elpd` 不可比；
- 不碰 `bm_core.make_model`，独立写 `numpyro.model`；
- 诊断直接调 `bm_experiments` / `bm_real._paired_loo_diff`，不重写。

### 候选 B：完全独立（自己的 res 结构与诊断）

代价：将来做 `select_across_families` 时要再写一遍配对 LOO 与诊断，容易出现
"两族 σ 口径不一致"这类隐性错误。**不推荐**。

### 候选 C：把 `make_model` 改成支持注册表再复用

违反 §7.0 冻结与"不改现有模型"的约束。**不做**。

---

## 3. 参数与先验冻结表（约束 1：先冻结，不许为凑结果调）

| 参数 | 先验（`zmax` = max|Z|，`ω_ref = 2π√(f_min·f_max)`） | 说明 |
|---|---|---|
| `Rs` | `LogNormal(log(0.2·zmax), 1.0)` | 复用 `R_inf` 口径 |
| `R1` | `LogNormal(log(0.1·zmax), 2.0)` | 与 `Rw/Rwo` 同尺度 |
| `R2` | `LogNormal(log(0.1·zmax), 2.0)` | 同上 |
| `Q1` | `LogNormal(log(zmax), 2.0)` | 窗口幅值，非 `Y` |
| `n1` | `Uniform(0.5, 1.0)` | 与 `phi_lo` 一致；文献 0.64–0.97 |
| `Q2` | `LogNormal(log(zmax), 2.0)` | 同上 |
| `n2` | `Uniform(0.5, 1.0)` | 同上 |
| `R_ad` | `LogNormal(log(0.1·zmax), 2.0)` | `branch="parallel"` |
| `tau_ad` | `LogNormal(log(1/ω_ref), 2.0)` | 拐点中心在窗口几何中心 |
| `R_L` | `LogNormal(log(0.1·zmax), 2.0)` | `branch="series"` |
| `L_ad` | `LogNormal(log(zmax/ω_ref), 2.0)` | 量纲 Ω·s，尺度 `zmax/ω_ref` |

**冻结规则**：M3 第一轮跑完后，只允许因为"先验写错量纲"改一次，并记录改动；
不允许为了"反演更准"逐步收窄先验。

---

## 4. 生成器独立性（约束 2）

M3 的合成真值**不复用 `bm_top.z_nested`**，改用已有独立实现：

- `evidence/nested_topology/theoretical_bound.py:model_b` —— 逐层手写，
  参数化用 `Y`（CPE 导纳）而非 `Q`，转换 `Y = 1/(Q·ω_ref^n)`；
- 或 `test_ad_gate_synth.py` 风格的独立解析式（`z_cpe` 手写 + 导纳递推）。

两条独立路径在 M1 自检里已交叉验证（逐节点 0 误差），跨文件复用进一步降低
"实现自洽但错误"的风险。

**验证方式**：档 1 用"生成器 → `bm_top.z_nested` 正演"的相对误差 <1e-10；
若两边同时写错，这个检查会通过但档 2/3 会暴露（真值反演不回来）。

---

## 5. M3 三档验收清单

### 档 1：结构正确（无噪声、真值代入）

| 用例 | 判据 |
|---|---|
| 1-1 生成器 vs `z_nested`（parallel，2% 噪声前） | 相对误差 <1e-10 |
| 1-2 生成器 vs `z_nested`（series） | 同上 |
| 1-3 极限：`ω→0/∞` | parallel `Rs+R1+R2` / `Rs`；series `+R_L` / `Rs` |
| 1-4 无源性 `min Re ≥ 0` | 全域扫 |

### 档 2：参数可恢复

| 用例 | 数据 | 判据 |
|---|---|---|
| 2-1 无噪声 | 真值正演（3 组参数：典型/弱感抗/强感抗） | 后验中位数相对误差 **<1%** |
| 2-2 加噪 2% | 同上 | 真值落入 **90% 区间**；中位数误差 **<4%**（借 #2 口径） |
| 2-3 加噪 5% | 同上 | 中位数误差 **<8%**，90% 区间覆盖率 ≥2/3 |
| 2-4 两种 branch | parallel / series 各跑 | 分别验收，不混用真值 |

### 档 3：统计可辨识

| 指标 | 判据 | 来源 |
|---|---|---|
| R̂ | <1.01 | 现有诊断 |
| ESS | >200 | 现有诊断 |
| divergences | <0.1% 采样步 | #2 |
| 贴边率（任一参数 `at_bound`） | <5% | `component_review` 口径 |
| 后验相对宽度（90% 区间 / 中位数） | 关键参数 <100%，宽于则标"弱可辨识" | 新增 |
| 成对相关 `|ρ|>0.95` | 逐对列出，不隐藏 | `param_corr` |
| 多峰 | 直方图/KDE 峰数 >1 即记录 | 新增 |
| 拓扑退化 | `layers=1` vs `2` 的 LOO 差 | 新增 |

**结论口径**：三档全过才讨论 `bm_families` 适配层。若某些参数不可分别辨识，
**不判拓扑失败**，但必须明确"哪组参数组合才是可辨识的"（例如
`R2·Q2^n` 的乘积、`R_L·τ` 的组合）。

---

## 6. 失败记录（约束 4）

每次运行输出一行结构化记录（CSV 或 markdown 表），**成功与失败都留**：

```
case, branch, layers, noise, seed, rhat_max, ess_min, div_frac,
max|rho|, rho_pair, width_90_rel, n_modes, at_bound_frac, median_err, covered
```

- 失败用例不删、不重跑换 seed 掩盖；
- 若换 seed，必须同时保留原 seed 的结果；
- 报告里先列失败，再列通过。

---

## 7. 不在本阶段范围

- 不建 `bm_families/` 注册表；
- 不改 `bm_core.make_model` / `bm_real.fit_one` / `select_k`；
- 不做 `select_across_families`（等 M3 结论）；
- 不做真实 AZ31 的 M4 回放（先证明合成可辨识性）；
- 不调先验直到结果好看。

---

## 8. 待审查确认的三点

1. **候选 A 的返回字段**是否够 `_paired_loo_diff` 直接用（需要
   `pointwise_elpd` 与 `elpd`）？
2. **`L_ad` 的先验尺度** `zmax/ω_ref` 是否合理（`R_L` 与 `L` 的量纲耦合，
   是否应该改成先采 `τ_L = L/R_L` 再派生 `L`）？
3. **档 2 的 4% / 8% 阈值**是否沿用 #2 的模拟口径，还是要按 EIS 常见噪声水平
   重新定？
