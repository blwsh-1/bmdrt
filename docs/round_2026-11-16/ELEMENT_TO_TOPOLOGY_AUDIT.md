# 元件 → 拓扑 接入审计（2026-11-16）

> 目的：回答「现有的元件能不能加入拓扑、怎么加入」。**只整理，不改代码**。
> 触发：用户对 `AZ31-12Li_5` 拟合差（rms 30.8%、R̂=22）提出对标别人的吸附模型
> `Rs + [CPEdl∥Rct] ∥ (Rads+Lads)`（Rwp 8.5%），问代码基础。

## 0. 一句话结论

**缺的是拓扑槽位，不是元件数学。** 登记在册的 8 个元件里，7 个正演函数都写好了
（`bm_core`/`bm_top`/`bm_diff`/`bm_ad`），只有 4 个接了槽位（`wired=True`）。
未接的 4 个里：

- `adsorption` 的数学**已经被** `rl_parallel` 覆盖（同一条公式），单独塞槽位无意义；
- `w_inf`/`w_open`/`w_short`（Warburg）数学现成，且 voigt 侧 `bm_core.model_z`
  **已经**支持 `diffusion="Ws"/"Wo"/"both"`，只是 registry 没登记扩散槽位；
- **完全缺**的只有 Gerischer（`bm_diff` 里没有）。

## 1. 元件清单（`circuit_registry.ELEMENTS`）

| id | 数学位置 | wired | maturity | 参数 | 已挂槽位 |
|---|---|---|---|---|---|
| `hn` | `bm_core:_hn_z` | ✅ | production | `W,tau,phi,zeta` | `voigt.relaxation_unit` |
| `cpe` | `bm_top:z_cpe` | ✅ | exploratory | `Q,n` | `nested.outer/inner/branch` |
| `rl_parallel` | `bm_top:z_rl_parallel` | ✅ | exploratory | `R_ad,tau_ad` | `nested.branch` |
| `rl_series` | `bm_top:z_rl_series` | ✅ | exploratory | `R_L,tau_L` | `voigt.series_inductor`、`nested.branch` |
| `adsorption` | `bm_ad:z_ad` | ❌ | exploratory | `Rad,tau_ad` | 无 |
| `w_inf` | `bm_diff:z_winf` | ❌ | frozen | `A` | 无 |
| `w_open` | `bm_diff:z_wopen` | ❌ | frozen | `Rw,tauD` | 无 |
| `w_short` | `bm_diff:z_wshort` | ❌ | frozen | `Rw,tauD` | 无 |

## 2. 拓扑清单（`circuit_registry.TOPOLOGIES`）

| 拓扑 | 骨架 | 槽位 |
|---|---|---|
| `voigt` | `R_inf + Σ HN_k [+jωL]` | `relaxation_unit`(hn,≥1)、`series_inductor`(rl_series,0–1) |
| `nested` | `Rs + [Q1∥(R1 + [Q2∥(R2 + Z_b)])]` | `outer_cpe`(cpe)、`inner_cpe`(cpe,仅 L=2)、`branch`(rl_parallel\|rl_series\|cpe) |

## 3. 逐个未接元件的判断

### 3.1 `adsorption` —— 数学已被覆盖，缺的是新拓扑

`bm_ad.z_ad = R_ad·(jωτ)/(1+jωτ)`，与 `bm_top.z_rl_parallel` **逐字相同**。
所以 `nested.branch="parallel"` 拟合的就是吸附电感（R∥L 回路），只是参数名不同
（`Rad` vs `R_ad`，τ 同名）。

**结论**：把 `adsorption` 塞进 `nested.branch` 的 accepts 列表**没有意义**——数学
等价，换名字/换 forward 引用只改变 `candidate_id`，不会改善拟合。

真正要补的是别人的**三支路并联拓扑**（单节点 CPE∥纯电阻∥串联感抗），见 §5 路径 A。

### 3.2 `w_inf` / `w_open` / `w_short` —— 数学现成，只差扩散槽位

三个正演都在 `bm_diff`。且 **voigt 侧已经接了一半**：`bm_core.model_z` 有
`diffusion="Ws"/"Wo"/"both"` 参数（默认 `None` 逐位不变）。缺口是：

1. registry 的 `voigt` 拓扑 `slots` 里**没登记** diffusion 槽位 ⇒ 三个 Warburg 被标
   `wired=False`；
2. `nested` 拓扑**完全没有**扩散接点（Warburg 通常串在 `Rct` 后，或 `Rct∥(CPE+Ws)`，
   nested 的 branch 槽位语义是"感抗/容性支路"，不是扩散）；
3. `w_short` 连 voigt 的 `diffusion` 参数都没接（`bm_core` 只认 `Ws`/`Wo`/`both`），
   "有实现未接 CLI"。

**结论**：能加，路径见 §5 路径 B。

### 3.3 Gerischer —— 完全缺

`bm_diff`、`bm_core`、`bm_top` 都没有。要加得先写正演（`Z_G = R/√(1+jωτ)` 及其
jax 版 + 可辨识性区间），不在本次"整理现有元件"范围内。

## 4. 加入拓扑的标准路径（registry 已内置框架）

给一个元件"打开 wired"需要五步，缺一不可（否则 `--selftest` 会红）：

1. **登记**：`register_element(ElementSpec(...))`，给 `forward_ref`、`params`、
   `maturity`，去掉 `wired=False` 与 `blocked_reason`。
2. **开槽位**：在目标 `TopologySpec.slots=` 里加
   `SlotSpec(slot_id, 描述, (element_id,), min_count, max_count, note)`。
3. **consistency**：写/复用 `xxx_consistency(by_slot, cfg)`，校验 config（如
   `layers`/`diffusion` 开关）与挂载数的对应，违规抛 `ValueError`。
4. **正演**：目标正演函数（`bm_core.model_z` 或 `bm_top.z_nested`）加分支 + jax 版
   `j_*`，且默认路径逐位不变。
5. **采样与调度**：`sampled_keys`/参数表/先验补齐；`circuit_runner.build_candidates`
   能生成该候选、`circuit_evaluator` 的候选池纳入。

## 5. 两条具体落地路径

### 路径 A：吸附三支路拓扑（对标别人，12Li_5 的缺口）

> **状态（2026-11-16 同日已落地）**：`adr` 拓扑已登记进 `circuit_registry`
> （含 `resistor` 元件 + `_adr_consistency`），`bm_select.fit_adr` / `circuit_runner`
> / `cmd_compare` 已接入，`bmdrt compare` 默认并列 voigt/nested/adr。
> 吸附支路最终用 `rl_parallel`（R∥L）而非本文案里写的 `bm_ad.z_ad`——两者数学同式，
> 复用 `rl_parallel` 避免 registry 里出现两个同式元件。实测见提交信息。

新拓扑 `adr`：

```
Z = Rs + [ CPE ∥ Rct ∥ (Rads + Z_ad) ]      # 单节点三支路并联
    Z_ad = R_ad·jωτ/(1+jωτ)                # 吸附电感（数学复用 bm_ad.z_ad）
```

- 参数 6 个：`Rs, Q, n, Rct, Rad, tau_ad`（正是别人那个规模，少而专一）。
- 元件全部复用现有数学（cpe + 纯电阻 + bm_ad），**只需写拓扑正演 + jax 版**。
- 注意别人的 `Rads+Lads` 是**串联**、我们 `Z_ad` 是 **R∥L 并联**；两者低频极限
  方向相反。§9.4.12 已实测过「R∥L 挂 voigt 最外层串联」只降 44.0%→40.7%，所以
  `adr` 是否比 nested 好，要跑了才知道，不预设。

### 路径 B：扩散槽位（把三个 Warburg 打开）

1. `voigt` 拓扑 `slots` 加 `diffusion` 槽位（`w_inf|w_open|w_short`，0–1），
   `wired` 打开；`bm_core` 已支持 `Ws/Wo`，`w_short` 需补 `diffusion="Wshort"` 分支。
2. `nested` 加扩散接点（这是更大的改动：nested 现在没有"扩散该串在哪"的定义，
   需要先定接点语义，再动 `z_nested`）。

## 6. 不建议现在做的

- **单独把 `adsorption` 挂进 `nested.branch`**：数学等价，零收益。
- **加 Gerischer**：无数学基础，且不在 12Li_5/4Li_150 的需求里。
