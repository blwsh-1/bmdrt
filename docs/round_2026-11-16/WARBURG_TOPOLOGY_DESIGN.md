# Warburg 拓扑接入设计（P1-2，2026-11-16）

> 只设计，不实装。用户裁决：Warburg 是**元件**不是拓扑族，不能因为正演写好
> 就算有完整电路族。本轮不改任何代码、不跑拟合。

## 1. 现状（读代码核对）

| 元件 | 正演 | jax 版 | registry | `bm_core.model_z` 接入 |
|---|---|---|---|---|
| `w_inf`（半无限 Warburg） | `bm_diff.z_winf` | `j_z_winf` | `wired=False` | ✅ 靠 key `Aw` |
| `w_open`（有限长阻挡） | `bm_diff.z_wopen` | `j_z_wopen` | `wired=False` | ✅ 靠 key `Rwo`+`tauD` |
| `w_short`（有限长短路） | `bm_diff.z_wshort` | `j_z_wshort` | `wired=False` | ❌ **未接 model_z，也未接 CLI** |

三个元件的 `blocked_reason` 都是「现有两个拓扑都没有扩散槽位；接入需要先定义
槽位与接点」。注意这里写「两个拓扑」是历史措辞——现在已有三个（voigt/nested/adr），
但**三个都没有扩散接点**，结论不变。

## 2. 为什么 Warburg 不能直接成一个族

用户指出的两条接法不等价，是设计的核心约束：

```
接法 A（扩散在并联节点内）：  Z = Rs + (Rct ∥ (CPE + Z_W))
接法 B（扩散串联在外）：      Z = Rs + (Rct ∥ CPE) + Z_W
```

- A 里 Warburg 与 CPE 串联后再与 Rct 并联 —— 扩散影响**电荷转移支路内部**；
- B 里 Warburg 串在整个并联块外 —— 扩散是**独立的串联传输阻抗**。

两者低频/高频极限、参数可辨识性都不同。正演 `z_winf` 写好只说明「元件能算」，
不说明「接在哪里」。所以接入对象是**拓扑槽位**，不是新建 `warburg` 族。

## 3. 建议接入路径（按最小改动排序）

### 3.1 第一步：voigt 顶层串联槽位（B 型接法）

`bm_core.model_z` 已经支持 `Aw` / `Rwo+tauD` key，且走的是**顶层串联**
（`z = z + ...`）——这正对应接法 B。缺的只是：

1. registry 的 voigt 拓扑登记 diffusion 槽位（把 `w_inf`/`w_open` 的 `wired`
   打开，定义槽位约束：顶层串联，最多一个扩散元件）；
2. `circuit_runner.build_candidates` 加 `diffusion` 开关生成候选；
3. `bm_real.fit_one` / `bm_core.fit` 的 `diffusion` 参数已存在（`fit_one` 签名
   有 `diffusion`），确认 CLI（`bmdrt.py` 的 `fit.diffusion` 配置）已透传——
   记录显示 CLI 层已接，主要是 registry/runner 没登记。

**风险低**：不动采样器，正演已有，只是把「已有能力」登记成候选。

### 3.2 第二步：`w_short` 补 model_z（仍属 B 型）

`bm_core.model_z` 的 diffusion 分支只认 `Aw` 和 `Rwo+tauD`。要支持短路
Warburg，需加 `p["Rws"]+p["tauDs"]` → `D.z_wshort`（纯新增 key，默认 None
时逐位不变），并把 registry 的 `w_short.wired` 打开。

**风险中**：碰 `bm_core.model_z`（生产内核），虽是纯新增分支，但要走完整回归
（默认路径逐位不变断言）。

### 3.3 第三步：nested 的并联节点内槽位（A 型接法）——**暂不做**

这是真正需要设计的部分：nested 有两个并联节点，扩散可以挂在
`R2 + Z_b` 里（第二节点内）。但这会引入新参数与简并：
- `CPE2 ∥ (R2 + Z_W)` 与 `CPE2 ∥ (R2 + R∥L)` 的支路语义不同；
- `Z_W` 的低频 `ω^{-1/2}` 与 CPE 的 `ω^{-n}` 可能互相吸收（R21 类似问题）。

工作量大且有可辨识性风险。**建议等 voigt 顶层槽位跑出证据，确认这批谱真有
扩散需求后再做**（当前 14 条失败谱没有一条指向扩散，见
`TOPOLOGY_COVERAGE_AUDIT.md` §2-B）。

## 4. 接入的准入条件（不是现在做）

满足以下任一条才启动 3.1：
1. 实际谱在低频段出现 `ω^{-1/2}` 斜率且现有三拓扑都拟合不掉；
2. 有独立的扩散真值/机制依据（不是从残差反推）；
3. CAL-01 校准排除了仪器低频伪影。

在此之前，`w_inf/w_open/w_short` 保持 `wired=False`，只作为「数学已备、未接
拓扑」登记在案。

## 5. 与 Gerischer 的区别

Gerischer 是**真缺元件**（`bm_diff` 无实现、registry 无登记），Warburg 是
**元件已有、槽位缺失**。两者在 P 清单里分属 P2（暂缓）和 P1-2（设计完成、
待实际需求）。不要因为都叫「扩散相关」就合并处理。
