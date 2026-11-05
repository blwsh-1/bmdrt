# -*- coding: utf-8 -*-
"""扩散阻抗元件（Warburg 族）与 HN↔Warburg 退化诊断。

**这个模块是加法，不改 `bm_core.py` 的既有模型。** `bm_core.make_model` 只有
显式传 `diffusion="Ws"/"Wo"/"both"` 时才会调用这里的元件；不传则行为与改动前
逐位一致。

为什么需要单独一层
------------------
实测（`python bm_diff.py` 会重跑）：单个 HN 在有限频率窗口内能逼近半无限
Warburg 到 **1.2e-6 相对误差**，代价是

    φζ = 0.500000,   R = A_W·√τ,   f_peak 在窗口下方 8 个量级

所以"模型表达不出 Warburg"是错的。真正的问题是**这组参数不可辨识、且没有物理
含义**：R 与 τ 沿 `lnR − ½lnτ = lnA_W` 这条脊线完全简并，脊线之外的东西一个都
定不住。把 4 个无意义参数换 1 个有意义的 A_W，才是加显式扩散项的理由——这是
**重参数化 + 可辨识性修复**，不是表达力扩展。

命名约定（必须先钉死，否则一定出错）
------------------------------------
同一个字母在不同软件里指相反的东西：

| 本模块 | 物理边界 | 数学式 | impedance.py 的名字 | Bisquert 的名字 |
|---|---|---|---|---|
| `W_inf` | 半无限 | A_W (jω)^(-1/2) | `W` | semi-infinite Warburg |
| `W_open` | 阻挡/开路（L 处无通量） | R_W coth(z)/z | **`Wo`** | `Wo`（open） |
| `W_short` | 透射/短路（L 处浓度固定） | R_W tanh(z)/z | **`Ws`** | `Ws`（short） |

`z = sqrt(jωτ_D)`，`τ_D = L²/D`。

**坑在这里**：`impedance.py` 的 `Ws` 是 tanh（有限长、透射），**不是**半无限。
bm_bench 里那个 `_zwarburg` 叫 Warburg，指半无限。中文文献里 "Ws" 也常指半无限。
所以本模块**不用 Ws/Wo 当函数名**，一律写 `W_inf` / `W_open` / `W_short`，
只在上表的对照里出现 Ws/Wo。

低频极限（决定可辨识性，别只看公式）
----------------------------------
    W_open  (coth): ωτ_D ≫ 1 → A_W (jω)^(-1/2)     （Warburg 尾）
                    ωτ_D ≪ 1 → R_W/3 + 1/(jωC_D),  C_D = τ_D/R_W
    W_short (tanh): ωτ_D ≫ 1 → A_W (jω)^(-1/2)
                    ωτ_D ≪ 1 → R_W

`W_open` 的低频极限里那个 `R_W/3` 是个**常数实部**，会被 R∞ 完全吸收。所以当
拐点落在窗口下方时，只剩 `C_D = τ_D/R_W` 可辨识，R_W 与 τ_D 单独都定不住。
`W_short` 的低频极限是常数 R_W，同样与 R∞ 冲突。

Warburg 的 DRT
--------------
    γ_W(lnτ) = A_W·√τ / π

数值核对（本文件 `_selftest`）：∫γ_W/(1+jωτ)dlnτ 与 A_W/√(jω) 的相对误差
< 1e-9。注意 **∫γ_W dlnτ 发散**，而任何有限个 HN 的面积都是有限的。

**这只说明什么、不说明什么**（容易推过头）：

- 说明：**全频域上**二者不可能严格等价。ω→0 时 HN 给有限电阻、半无限
  Warburg 发散，这个差异是结构性的，与 K 无关。
- **不说明**：有限频率窗口内误差有正下界。反例就在本文件里——单个 HN 在
  0.1–1e3 Hz 内能把半无限 Warburg 逼近到 1.2e-6，靠的是把峰推到窗口外
  8 个量级、把差异藏到观测范围之外。所以"发散 ⇒ 窗口内逼近不了"这一步
  是不成立的。
- 有限窗口内到底能逼到多近，只能靠**数值测试**回答（`cmd_diff.case_wo`：
  当前 K/参数范围/优化条件下仍有 28% 残差），那是经验结论，不是定理。
"""

from __future__ import annotations

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# --------------------------------------------------------------------------
# 物理有效性的容忍阈值（**唯一来源，不要在别处再写一个数**）
# --------------------------------------------------------------------------

#: `φζ > 1` 的（分量×样本）占比超过这个值，该分量就**不能**再当作非负弛豫
#: 分布使用（解析 DRT 出负值，峰位/面积都不可引用）。
#:
#: 为什么不是 0：`φζ = φ·ζ` 是两个连续参数相乘，NUTS 在边界附近总会漏出
#: 极少数样本越过 1（实测健康模型也有 0.1%~0.5%），写"必须严格为 0"
#: 会让所有模型都被判死。1% 是工程阈值：既挡得住"优化器把 φζ 推到 1 以上
#: 去凑电容尾"这种系统性越界（实测非物理分量通常在 30%~100%），
#: 又不会因为浮点噪声误杀。
#:
#: 这个阈值同时被 `hn_warburg_signature()`（判分量）和
#: `bm_real.select_k()`（硬筛选）使用，改一处即全局一致。
NONPHYS_TOL = 0.01


# --------------------------------------------------------------------------
# 元件（numpy，供后处理 / 绘图 / 测试）
# --------------------------------------------------------------------------

def z_winf(f, A):
    """半无限 Warburg：Z = A / sqrt(jω)。A 单位 Ω·s^(-1/2)。"""
    return A / np.sqrt(1j * 2 * np.pi * np.asarray(f, float))


def _coth_over_z(z):
    """coth(z)/z，数值稳定版。

    直接写 cosh(z)/(z·sinh(z)) 在 |z|>~350 时溢出（实测 τ_D=1e4 s 就炸）。
    分三段：小 |z| 用 Laurent 级数，大 |z| 用 1/z，中间用 exp(2z) 形式。
    """
    az = np.abs(z)
    out = np.empty_like(z)
    sm = az < 1e-3
    lg = az > 30.0
    mid = ~(sm | lg)
    zs, zm_, zl = z[sm], z[mid], z[lg]
    out[sm] = 1.0 / zs ** 2 + 1.0 / 3.0 - zs ** 2 / 45.0
    e2 = np.exp(2.0 * zm_)
    out[mid] = (e2 + 1.0) / (e2 - 1.0) / zm_
    out[lg] = 1.0 / zl
    return out


def _tanh_over_z(z):
    """tanh(z)/z，数值稳定版。"""
    az = np.abs(z)
    out = np.empty_like(z)
    sm = az < 1e-3
    lg = az > 30.0
    mid = ~(sm | lg)
    zs, zm_, zl = z[sm], z[mid], z[lg]
    out[sm] = 1.0 - zs ** 2 / 3.0 + 2.0 * zs ** 4 / 15.0
    e2 = np.exp(-2.0 * zm_)
    out[mid] = (1.0 - e2) / (1.0 + e2) / zm_
    out[lg] = 1.0 / zl
    return out


def z_wopen(f, Rw, tauD):
    """有限长、阻挡边界（开路）：Z = R_W·coth(z)/z，z = sqrt(jωτ_D)。

    低频 → R_W/3 + 1/(jωC_D)；高频 → R_W/√(jωτ_D)（Warburg 尾）。
    """
    z = np.sqrt(1j * 2 * np.pi * np.asarray(f, float) * tauD)
    return Rw * _coth_over_z(z)


def z_wshort(f, Rw, tauD):
    """有限长、透射边界（短路）：Z = R_W·tanh(z)/z。

    低频 → R_W；高频 → Warburg 尾。
    """
    z = np.sqrt(1j * 2 * np.pi * np.asarray(f, float) * tauD)
    return Rw * _tanh_over_z(z)


def gamma_winf(ln_tau, A):
    """半无限 Warburg 的 DRT：γ(lnτ) = A√τ/π，∫γ dlnτ 发散。"""
    return A * np.exp(0.5 * np.asarray(ln_tau, float)) / np.pi


# --------------------------------------------------------------------------
# 元件（jax，供 make_model 前向计算）
# --------------------------------------------------------------------------

def _j_coth_over_z(z):
    """coth(z)/z 的 jax 版。

    `jnp.clip` 不接受复数，而 `jnp.where` 会**先把三个分支都算出来**，
    所以大 |z| 分支里的 exp(2z) 会溢出成 inf，即使最终不选它也会污染梯度。
    这里把实部夹到 [-30, 30]：中间分支本来就只在 |z|≤30 时生效，夹了不影响
    取值；两端分支的结果根本不用这个数，夹了只是让它保持有限。
    """
    az = jnp.abs(z)
    zc = jnp.clip(z.real, -30.0, 30.0) + 1j * z.imag
    e2 = jnp.exp(2.0 * zc)
    mid = (e2 + 1.0) / (e2 - 1.0) / z
    return jnp.where(az < 1e-3, 1.0 / z ** 2 + 1.0 / 3.0 - z ** 2 / 45.0,
                     jnp.where(az > 30.0, 1.0 / z, mid))


def _j_tanh_over_z(z):
    az = jnp.abs(z)
    zc = jnp.clip(z.real, -30.0, 30.0) + 1j * z.imag
    e2 = jnp.exp(-2.0 * zc)
    mid = (1.0 - e2) / (1.0 + e2) / z
    return jnp.where(az < 1e-3, 1.0 - z ** 2 / 3.0 + 2.0 * z ** 4 / 15.0,
                     jnp.where(az > 30.0, 1.0 / z, mid))


def j_z_winf(omega, A):
    return A / jnp.sqrt(1j * omega)


def j_z_wopen(omega, Rw, tauD):
    return Rw * _j_coth_over_z(jnp.sqrt(1j * omega * tauD))


def j_z_wshort(omega, Rw, tauD):
    return Rw * _j_tanh_over_z(jnp.sqrt(1j * omega * tauD))


# --------------------------------------------------------------------------
# 参数化：把 A_W 换成"窗口内的阻抗幅值"
# --------------------------------------------------------------------------

def a_from_rw(Rw, f_ref):
    """A_W = R_W·√ω_ref。R_W 的物理含义：|Z_W| 在 f_ref 处的值（Ω）。"""
    return Rw * np.sqrt(2 * np.pi * f_ref)


def rw_from_a(A, f_ref):
    return A / np.sqrt(2 * np.pi * f_ref)


# --------------------------------------------------------------------------
# HN ↔ Warburg 退化诊断
# --------------------------------------------------------------------------

def hn_peak_ln_tau(tau, phi, zeta, span: float = 20.0, n_coarse: int = 401,
                   n_fine: int = 201, half: float = 0.20):
    """一般 HN 的 DRT **实际峰位** ln τ_peak（逐样本，矢量化的）。

    为什么不能直接用 τ：HN 的 DRT 峰不在 τ 上。把 `bm_core._hn_drt_r` 里的
    自变量取成 r = ln(τ/τ0)，γ 的形状只由 (φ, ζ) 决定；峰位 r* 求出来，
    真实峰位就是 ln τ0 + r*。
    ζ=1（Cole-Cole，关于 lnτ 对称）时 r* = 0，与 1/(2πτ) 一致；
    ζ≠1 时会偏。与独立暴力解（lnτ 网格 −1..1 共 4e5 点 + 抛物线插值）逐点对照：

        φ=0.87 ζ=1.31 → r*=+0.0879  f_peak/f_τ = 0.9158  (−8.4%)
        φ=0.87 ζ=0.94 → r*=−0.0200  f_peak/f_τ = 1.0202  (+2.0%)
        φ=0.70 ζ=1.43 → r*=+0.2878  f_peak/f_τ = 0.7498  (−25%)
        ζ=1.0         → r*= 0        f_peak/f_τ = 1.0000

    本函数与暴力解的差 ≤ 4e-4（ln τ 单位，即 0.04% 频率），够诊断用。

    **为什么要两级网格**：一步到位要么分辨率不够、要么内存爆。粗网格
    （步长 0.1）先把峰框住，再在峰附近 ±0.2 内细化到 0.002。

    **踩过的坑（两次，都记在这里）**：
    ① 单级 601 点（步长 0.05）直接 argmax，ζ=0.94 那 −0.02 的真偏移会被
       量化成 −0.02 或 0 两者之一，取决于网格相位。峰形极平
       （r 从 −0.02 到 0，γ 只从 0.709 变到 0.713），必须细化 + 插值。
    ② 更隐蔽的一次：`_hn_drt_r` 里把指数写成了 `exp(−πφ·r)`（应为
       `exp(−φ·r)`），γ 在 lnτ 轴上被压缩 π 倍。峰值位置看着"对"，但
       ζ≠1 的偏移全部缩水 π 倍（−8.4% 报成 −2.8%），`∫γ dlnτ = 1/π`。
       是 `bm_check` 的 A2/A3 抓到的，不是这里的自检——所以**峰位断言要
       与独立暴力解对比，不能与另一个共用同一实现的东西对比**。

    返回 (n,) 的 ln τ_peak。
    """
    import bm_core as B                     # 延迟导入：bm_core 顶层已 import bm_diff

    lnt = np.log(np.asarray(tau, float))
    phi = np.asarray(phi, float)[:, None]
    zeta = np.asarray(zeta, float)[:, None]

    g = np.linspace(-span, span, n_coarse)                 # (G,)
    c = g[np.argmax(B._hn_drt_r(g[None, :], phi, zeta), axis=1)]   # (n,)
    h = np.linspace(-half, half, n_fine)                   # (F,)
    grid = c[:, None] + h[None, :]                         # (n,F)
    j = np.argmax(B._hn_drt_r(grid, phi, zeta), axis=1)    # (n,)
    return lnt + grid[np.arange(len(c)), j]


def hn_warburg_signature(w, tau, phi, zeta, f_lo, f_hi, pad_dec=1.5):
    """单个 HN 分量的"像不像 Warburg"指纹。

    参数
    ----
    w, tau, phi, zeta : (n,) 逐样本的该分量参数
    f_lo, f_hi        : 实测频率窗口
    pad_dec           : 与 `bm_core.make_model` 的 τ 先验边界一致的 pad

    返回 dict
    ---------
    pz            : φζ 的中位数。Warburg 尾巴要求 = 0.5
    pz_lo/pz_hi   : φζ 的 5/95 分位
    a_impl        : 由 R/√τ 反推的等效 A_W（只在 φζ≈0.5 时有意义）
    ridge_ratio   : std(lnR − ½lnτ) / std(lnR)。≪1 表示 R 与 τ 沿脊线简并
    corr_lnR_lntau: lnR 与 lnτ 的后验相关系数。→ +1 表示完全简并
    f_tau         : 特征频率 1/(2πτ) 的中位数（**不是**峰位）
    f_peak        : DRT 实际峰位的中位数（由 `hn_peak_ln_tau` 数值求）
    tau_outside_dec : 特征频率在窗口外多少 dec（0 = 在内），逐样本取中位数
    outside_dec   : 峰位在窗口外多少 dec（0 = 在内），逐样本取中位数
    side          : 峰位的 "below" / "above" / "in"
    peak_shift_dec: log10(f_peak/f_tau)。ζ=1 时为 0；|它|>0.1 时两个口径
                    会给出不同结论，报告里要提醒
    at_bound      : 落在 τ 先验边界 0.5 decade 内的样本占比
    p_tau_out     : **后验概率** P(f_τ ∉ [f_lo, f_hi])，无条件
    p_peak_out    : **条件后验概率**
                    P(f_peak ∉ [f_lo, f_hi] | φζ ≤ 1)
                    ——注意是**条件**概率，不是无条件概率。非物理样本
                    （φζ>1）没有可引用的物理峰位，只能排除；所以分母是
                    "物理样本数"而不是"总样本数"。读这个数时必须同时看
                    `frac_nonphysical`：若 40% 样本非物理，剩下 60% 的
                    峰位统计代表不了整个模型。
    frac_nonphysical : φζ>1 的样本占比（此时解析 γ 有负值，峰位不是物理量）
    nonphysical   : frac_nonphysical > `NONPHYS_TOL`（默认 1%）
                    → 该分量**物理上不可用**，`bm_real.select_k` 据此硬淘汰
    peak_usable   : 物理样本是否够做峰位统计（≥ max(20, 5%)）
                    → False 时 `f_peak`/`p_peak_out` 给 NaN。这与
                    `nonphysical` 是**两件事**：2% 越界时分量已判非物理，
                    但峰位统计仍然算得出来。
    _d_pk/_d_tau  : 逐样本的窗口外距离数组（私有，供 `looks_like_warburg` 用）

    **中位数 ≠ 概率**。`outside_dec` 回答"中位数在窗口外多远"，
    `p_peak_out` 回答"多少比例的样本在窗口外"。判级必须用后者：
    中位数在窗口外 1.2 dec 的样本也可能只有 55% 真在窗口外。

    为什么两套口径都要给：τ 是**模型参数**，峰是**物理位置**。判"分量是不是
    被挤到窗口外去替扩散尾打工"，看峰；判"参数是不是顶在先验边界上"，看 τ
    （即 at_bound）。混用会让 ζ≠1 时的结论偏 8% 以上。

    **φζ>1 时峰位不是物理峰**：解析 γ 会出现负值（见 `bm_core._hn_drt_unit`），
    那时 `hn_peak_ln_tau` 求出的只是"解析函数的最大值位置"。这些样本一律
    排除在峰位诊断之外，只统计占比；物理样本不足 5% 时峰位相关字段给 NaN。
    """
    w = np.asarray(w, float)
    tau = np.asarray(tau, float)
    phi = np.asarray(phi, float)
    zeta = np.asarray(zeta, float)
    lnR, lnt = np.log(np.maximum(w, 1e-300)), np.log(tau)
    pz = phi * zeta

    s_lnR = float(np.std(lnR)) or 1e-300
    ridge = float(np.std(lnR - 0.5 * lnt)) / s_lnR
    if s_lnR > 1e-12 and np.std(lnt) > 1e-12:
        corr = float(np.corrcoef(lnR, lnt)[0, 1])
    else:
        corr = float("nan")

    f_tau = 1.0 / (2 * np.pi * tau)
    f_pk = np.exp(-hn_peak_ln_tau(tau, phi, zeta)) / (2 * np.pi)

    # ---- φζ>1：解析 γ 有负值，那个"最大值位置"只是解析函数的极值点 ----
    # 不能当成物理弛豫峰。这类样本**不参与峰位诊断**，只统计占比。
    # 判据用 φζ≤1（严格非负的条件），不用"γ_min≥0"——后者要对每个样本算
    # 一整条曲线，代价高且与 φζ 等价。
    phys = pz <= 1.0 + 1e-12
    frac_np = float(np.mean(~phys))
    use = phys & np.isfinite(f_pk)
    enough = use.sum() >= max(20, int(0.05 * use.size))
    if not enough:
        use = np.zeros_like(phys)

    def _dist(fx):
        """逐样本的"窗口外距离"（dec）；在窗口内为 0。

        这是**逐样本**的，不是"先取中位数再比"——后者只能回答
        "中位数在不在窗口内"，回答不了"多少比例的样本在窗口外"。
        """
        d = np.zeros_like(fx)
        hi, lo = fx > f_hi, fx < f_lo
        d[hi] = np.log10(fx[hi] / f_hi)
        d[lo] = np.log10(f_lo / fx[lo])
        return d

    d_tau, d_pk = _dist(f_tau), _dist(f_pk)
    d_pk_u = d_pk[use] if use.any() else np.array([np.nan])

    def _med(a):
        a = a[np.isfinite(a)]
        return float(np.median(a)) if a.size else float("nan")

    out_tau = _med(d_tau)
    out_pk = _med(d_pk_u)
    side_tau = "in" if out_tau <= 0 else ("above" if np.median(f_tau) > f_hi else "below")
    if not np.isfinite(out_pk) or out_pk <= 0:
        side = "in" if np.isfinite(out_pk) else "unknown"
    else:
        side = "above" if np.nanmedian(f_pk[use]) > f_hi else "below"

    t_lo = 1.0 / (2 * np.pi * f_hi) / 10 ** pad_dec
    t_hi = 1.0 / (2 * np.pi * f_lo) * 10 ** pad_dec
    near = (np.abs(np.log(tau / t_lo)) < 0.5 * np.log(10)) | \
           (np.abs(np.log(tau / t_hi)) < 0.5 * np.log(10))

    return {
        "phi": float(np.median(phi)),
        "zeta": float(np.median(zeta)),
        "pz": float(np.median(pz)),
        "pz_lo": float(np.percentile(pz, 5)),
        "pz_hi": float(np.percentile(pz, 95)),
        "a_impl": float(np.median(w / np.sqrt(tau))),
        "ridge_ratio": ridge,
        "corr_lnR_lntau": corr,
        "f_tau": float(np.median(f_tau)),
        "f_peak": (float(np.nanmedian(f_pk[use])) if use.any() else float("nan")),
        "peak_shift_dec": (float(np.log10(np.nanmedian(f_pk[use])
                                          / np.median(f_tau))) if use.any() else float("nan")),
        "tau_outside_dec": out_tau,
        "tau_side": side_tau,
        "outside_dec": out_pk,
        "side": side,
        "at_bound": float(np.mean(near)),
        # ---- 两个口径的**后验概率**（不是中位数）----
        "p_tau_out": float(np.mean(d_tau > 0)),
        "p_peak_out": (float(np.mean(d_pk_u > 0)) if use.any() else float("nan")),
        "frac_nonphysical": frac_np,
        # 两个**不同**的问题，别合并：
        #   nonphysical → 该分量能不能用（阈值 NONPHYS_TOL，进硬筛选）
        #   peak_usable → 峰位统计算不算得出来（物理样本够不够）
        "nonphysical": bool(frac_np > NONPHYS_TOL),
        "peak_usable": bool(enough),
        # 私有：逐样本距离，供 looks_like_warburg 用（不进 JSON）
        "_d_pk": d_pk_u,
        "_d_tau": d_tau,
    }


def looks_like_warburg(sig, tol_pz=0.04, min_bound=0.30, min_out=1.0):
    """把指纹翻成 strong / weak / no 三级，并给出理由。

    **主判据是简并度，不是 φζ。** 这条是被实测逼出来的：τ 被先验边界
    （pad_dec=1.5）截住时，分量根本跑不到 Warburg 渐近区，φζ 停在 0.59 左右
    而不是 0.5；但它依然坐在边界上、R 与 τ 依然沿脊线完全相关。只认 φζ=0.5
    会漏掉这种——恰恰是最常见的那种。

      strong : φζ 落在 0.5±tol_pz，且（峰位在窗口外 ≥min_out dec 或
               贴 τ 边界 ≥min_bound），且 R-τ 简并
      weak   : 后两条成立但 φζ 没落到 0.5 —— 典型是 τ 被边界截住
      no     : 其余（含 φζ>1 占多数、峰位诊断不适用的情况）

    **"窗口外"用后验概率判，不用中位数判**：判据是
    `P(峰位在窗口外 ≥min_out dec) ≥ 0.5`。中位数在窗口外 1.2 dec 的样本
    也可能只有 55% 在窗口外，"多少比例"和"中位数在哪"是两个问题。

    即使 strong 也只说明"这个分量在代替扩散尾"，**不能**说"它是 Warburg"。
    """
    # φζ>1 超容忍阈值 → 解析 DRT 有负值，峰位不是物理量，整条判据不适用。
    # 注意这里判的是"分量能不能用"，不是"峰位算不算得出来"（后者看
    # peak_usable）；两者阈值不同，别混。
    if sig.get("nonphysical"):
        return "no", [f"{sig['frac_nonphysical'] * 100:.1f}% 的样本 φζ>1"
                      f"（容忍上限 {NONPHYS_TOL * 100:.0f}%），"
                      f"解析 DRT 出现负值，该分量不是非负弛豫分布，峰位诊断不适用"]

    d = sig.get("_d_pk")
    if d is not None and np.size(d):
        d = np.asarray(d, float)
        d = d[np.isfinite(d)]
    if d is not None and np.size(d):
        p_far = float(np.mean(d >= min_out))
    else:                                   # 退化口径（没有逐样本距离时）
        p_far = 1.0 if sig["outside_dec"] >= min_out else 0.0
    outside = p_far >= 0.5
    atb = sig["at_bound"] >= min_bound
    degen = (sig["corr_lnR_lntau"] > 0.8) or (sig["ridge_ratio"] < 0.5)
    pinned = abs(sig["pz"] - 0.5) <= tol_pz

    why = []
    if outside:
        why.append(f"峰位在窗口{('下' if sig['side'] == 'below' else '上')}方 "
                   f"{sig['outside_dec']:.1f} dec（P={p_far:.2f}）")
    elif atb:
        why.append(f"{sig['at_bound'] * 100:.0f}% 样本贴 τ 先验边界")
    if degen:
        why.append(f"corr(lnR,lnτ)={sig['corr_lnR_lntau']:+.2f}、"
                   f"脊线比={sig['ridge_ratio']:.2f}")
    if pinned:
        why.append(f"φζ={sig['pz']:.3f}≈0.5")
    elif why:
        why.append(f"φζ={sig['pz']:.3f} 未到 0.5（τ 被边界截住时的典型值）")

    if pinned and (outside or atb) and degen:
        return "strong", why
    if (outside or atb) and degen:
        return "weak", why
    return "no", why


# --------------------------------------------------------------------------
# 模型比较的前提：这个频段能不能区分 Ws 与 Wo
# --------------------------------------------------------------------------

def diffusion_resolution(f, s=None, diffusion=None, tol: float = 0.02) -> dict:
    """判断**本频段**能不能区分 W_short（透射）与 W_open（阻挡）。

    为什么必须单独判
    ----------------
    `W_open = R_W·coth(z)/z`、`W_short = R_W·tanh(z)/z`，`z = sqrt(jωτ_D)`。
    当 `ωτ_D ≫ 1` 时 `coth(z) ≈ tanh(z) ≈ 1`，两者都退化成
    `R_W/z = A_W (jω)^(-1/2)` —— **在数据上完全同形**。弯曲只出现在
    `f ≲ f_D = 1/(2πτ_D)`。所以：

      · 若整条谱都在 `f ≫ f_D`，`none / Ws / Wo` 三个模型给出几乎相同的
        阻抗，elpd 差多少都只是噪声，**不能用它宣称"证明了有限长扩散"**；
      · 若 `f_min < f_D` 且两核在带内相对差 > tol，才真正可分辨。

    这是"模型比较的前提检验"，和 K-K 门禁同一性质：**先确认比较有意义，
    再比较**。不做这一步，`--diffusion both` 的 elpd 排名会被当成物理结论。

    参数
    ----
    f        : 频率数组（Hz）
    s        : 后验样本 dict；有 `tauD` 时用它算 f_D
    diffusion: 本次实际启用的扩散项（None/none/Ws/Wo/both）
    tol      : 两核在带内的最大相对差阈值，低于此值视为不可分

    返回 dict（全部可 JSON 序列化）
    ------------------------------
    diffusion / f_min / f_max / tauD / f_D / below_fD / max_reldiff /
    tauD_max_resolvable / tol / resolvable / level / note
    """
    f = np.asarray(f, float)
    f = f[np.isfinite(f) & (f > 0)]
    out = dict(diffusion=diffusion, f_min=(float(f.min()) if f.size else None),
               f_max=(float(f.max()) if f.size else None),
               tauD=None, f_D=None, below_fD=None, max_reldiff=None,
               # 只有 τ_D 小到这个值以下，f_D 才会落进测量窗口
               tauD_max_resolvable=(float(1.0 / (2 * np.pi * f.min()))
                                    if f.size else None),
               tol=float(tol), resolvable=False, level="unknown", note="")

    if f.size == 0:
        out["note"] = "没有有效频点，无法判断扩散项可分辨性"
        return out

    tauD = None
    if s is not None and "tauD" in s:
        v = np.asarray(s["tauD"], float)
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            tauD = float(np.median(v))
    out["tauD"] = tauD

    if tauD is not None:
        out["f_D"] = float(1.0 / (2 * np.pi * tauD))
        out["below_fD"] = bool(f.min() < out["f_D"])

        # 两核在**本频段**的最大形状差异（R_W=1）。
        #
        # 用 `|Wo−Ws| / max(|Wo|,|Ws|)` 而不是 `/|Ws|`：
        # 低频端 Wo → 1/(jωC) 会发散、Ws → R_W 有限，用 |Ws| 作分母会
        # 得到 15000% 这种没有信息量的数（实测 τ_D=1e-3 s 时就是这样）。
        # 除以较大者把指标压到 [0,1]：0 = 完全同形，1 = 完全不像。
        z = np.sqrt(1j * 2 * np.pi * f * tauD)
        wo = _coth_over_z(z)
        ws = _tanh_over_z(z)
        den = np.maximum(np.maximum(np.abs(wo), np.abs(ws)), 1e-300)
        out["max_reldiff"] = float(np.max(np.abs(wo - ws) / den))

    if diffusion in (None, "", "none"):
        out["level"] = "no_diffusion"
        out["note"] = (
            "未启用扩散项。要判断需不需要，先跑 --diffusion Ws 拿到 τ_D 的后验；"
            f"本频段只有在 τ_D ≲ {out['tauD_max_resolvable']:.3g} s"
            f"（即 f_D ≳ {f.min():.3g} Hz）时才可能看到有限长扩散的弯曲，"
            f"τ_D 更大时 Ws 与 Wo 在带内完全重合。")
        return out

    if tauD is None:
        out["level"] = "unknown"
        out["note"] = ("拿不到 τ_D 后验（模型里没有 tauD），无法判断 Ws/Wo 是否可分；"
                       "不要用 elpd 差宣称有限长扩散")
        return out

    if out["max_reldiff"] <= tol:
        dec = float(np.log10(f.min() / out["f_D"])) if out["f_D"] else float("nan")
        where = (f"f_D = {out['f_D']:.3g} Hz 在窗口下界 {f.min():.3g} Hz 之下 "
                 f"{dec:.1f} decade" if out["f_D"] < f.min()
                 else f"f_D = {out['f_D']:.3g} Hz 高于窗口上界 {f.max():.3g} Hz")
        out["level"] = "indistinguishable"
        out["note"] = (
            f"Ws 与 Wo 在 {f.min():.3g}–{f.max():.3g} Hz 内最大形状差异只有 "
            f"{out['max_reldiff']:.2%}（阈值 {tol:.0%}）：{where}，"
            f"整段都在 ωτ_D ≫ 1 的渐近区，**两个模型在本频段不可分**。"
            f"此时 none/Ws/Wo 的 elpd 差不能当证据，只能报「不可分」。")
        return out

    if not out["below_fD"]:
        # 走到这里说明带内还有 >tol 的差异，但都集中在窗口边缘：
        # f_D 在窗口下方，弯曲的"拐点"本身没进来，只剩过渡段的尾巴。
        # 这种情况不能宣称可分辨（拐点没测到，τ_D 的后验基本由高频尾定，
        # Ws/Wo 的差别只体现在最低几个点上，抗噪能力极差）。
        dec = float(np.log10(f.min() / out["f_D"])) if out["f_D"] else float("nan")
        out["level"] = "indistinguishable"
        out["note"] = (
            f"带内最大差异 {out['max_reldiff']:.1%} 集中在窗口边缘，但 "
            f"f_D = {out['f_D']:.3g} Hz 在窗口下界 {f.min():.3g} Hz 之下 "
            f"{dec:.1f} decade——弯曲的拐点没进入窗口，τ_D 基本只由高频尾约束，"
            f"按**不可分**处理（保守）。")
        return out

    out["resolvable"] = True
    out["level"] = "resolvable"
    out["note"] = (
        f"f_D = {out['f_D']:.3g} Hz 落在窗口内（f_min = {f.min():.3g} Hz），"
        f"两核在带内最大形状差异 {out['max_reldiff']:.1%} > {tol:.0%}："
        f"本频段**可以**区分 Ws 与 Wo，elpd 比较有意义。"
        f"仍要注意：可分辨 ≠ 数据里真有有限长扩散。")
    return out


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------

def warburg_band_slope(f, Zs, slope_tol: float = 0.15,
                       min_dec: float = 1.0) -> dict:
    """检查模型阻抗是否在**连续频段**上呈 Nyquist 45° 直线。

    为什么需要这一步：`looks_like_warburg()` 只看**单个分量**的 φζ、峰位和
    R–τ 简并。K≥2 时模型可以改用「多个 φ≈0.5 的分量拼出 45° 段」来逼近
    扩散尾，这时**没有任何分量是简并的**，逐分量判据必然漏判（实测 4382：
    真值 `Rs_Ws`，两分量 φζ=0.502/0.501，判 `no`）。

    但「两个分量 φζ≈0.5」本身不足以说明是扩散——一个 `R‖CPE(n=0.5)` 也是
    φ=0.5。区别在**合成阻抗的几何形状**上，所以直接查形状。

    **用 `d(−Im Z)/d(Re Z)` 而不是 `d log|Z| / d log f`**：后者被串联电阻
    压平。实测 4382（真值就是半无限 Warburg，R_inf=106 Ω 与扩散幅值同量级）
    的 `|Z|` 斜率只有 −0.03，看起来完全不像扩散；而 Warburg 的
    `Z_W = A(jω)^(−1/2)` 满足 `Re = −Im`，即 Nyquist 斜率恒为 1，与
    `R_inf` 无关（`R_inf` 只是沿实轴平移）。

    参数
    ----
    f   : 频率（Hz），要求单调（内部会排序）
    Zs  : 模型阻抗，形状 (n_samp, n_f) 或 (n_f,)。逐样本算斜率再取中位数，
          **不要先平均参数再算曲线**（非线性）。
    slope_tol : 允许偏离 1 的幅度。0.15 → 斜率落在 [0.85, 1.15]。
    min_dec   : 连续满足条件的最短频段长度（decade）。

    返回 dict：band_dec（最长连续段长度）、f_lo/f_hi、slope_med、frac_band
    """
    f = np.asarray(f, float)
    o = np.argsort(f)
    f, lf = f[o], np.log10(f[o])
    Z = np.asarray(Zs, complex)
    if Z.ndim == 1:
        Z = Z[None, :]
    Z = Z[:, o]
    if f.size < 5:
        return dict(band_dec=0.0, f_lo=float("nan"), f_hi=float("nan"),
                    slope_med=float("nan"), frac_band=0.0, usable=False)

    # 以 log f 为参数求两个方向的变化率，再取比值 = d(−Im)/d(Re)。
    # 不直接用 np.gradient(-Im, Re)：Re 未必随 f 单调（有电感/多弧时）。
    dre = np.gradient(Z.real, lf, axis=1)
    dim = np.gradient(-Z.imag, lf, axis=1)
    # 分母过小（Re 在该点几乎不随 f 变）→ 比值无意义，标记为 nan
    with np.errstate(divide="ignore", invalid="ignore"):
        sl = np.where(np.abs(dre) > 1e-12 * np.maximum(np.abs(Z.real).max(axis=1, keepdims=True), 1e-300),
                      dim / dre, np.nan)
    slope = np.nanmedian(sl, axis=0)
    ok = np.isfinite(slope) & (np.abs(slope - 1.0) <= slope_tol)

    # 最长连续 True 段
    best_len, best_i, i = 0, -1, 0
    while i < ok.size:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < ok.size and ok[j + 1]:
            j += 1
        if j - i + 1 > best_len:
            best_len, best_i = j - i + 1, i
        i = j + 1
    if best_len < 2:
        return dict(band_dec=0.0, f_lo=float("nan"), f_hi=float("nan"),
                    slope_med=float(np.nanmedian(slope)), frac_band=0.0,
                    usable=False)
    lo, hi = best_i, best_i + best_len - 1
    dec = float(lf[hi] - lf[lo])
    return dict(band_dec=dec, f_lo=float(f[lo]), f_hi=float(f[hi]),
                slope_med=float(np.nanmedian(slope[lo:hi + 1])),
                frac_band=float(dec / max(lf[-1] - lf[0], 1e-12)),
                usable=True)


def warburg_multicomponent(sig, band: dict | None, tol_pz: float = 0.04,
                           min_n: int = 2, min_dec: float = 0.5) -> tuple:
    """多分量拼 45° 段的**弱识别**。

    返回 `("weak", why)` 或 `("no", why)`。**永远不给 strong**：
    两个 φζ≈0.5 的分量加上一段 45° 斜率，仍然可能是一个宽化的
    `R‖CPE(n≈0.5)`，不足以断定是半无限扩散。它的用途是提示人去看，
    不是改变 K 选择。

    `min_dec` 取 0.5 而不是 1.0：实测用 K 个 `R‖CPE(φ=0.5)` 离散化
    半无限 Warburg，K=4 只有 0.51 decade、K=6 只有 0.68 decade 落在
    45°±0.15 内——有限个分量本来就拼不出长段的干净 45°。要求 1 decade
    会让这条判据永远不触发。段长会写进 why，由人判断。

    `band` 来自 `warburg_band_slope()`；None 表示没算（就只按参数判，
    此时一律不给 weak，避免只凭参数下结论）。
    """
    pinned, inside = [], []
    for k, s in enumerate(sig or []):
        if s.get("nonphysical"):
            continue
        if abs(float(s.get("pz", 1.0)) - 0.5) <= tol_pz:
            pinned.append(k)
            if float(s.get("p_peak_out", 1.0)) < 0.5:
                inside.append(k)
    if len(inside) < min_n:
        return "no", []
    why = [f"{len(inside)} 个分量 φζ 同时落在 0.5±{tol_pz}（#{'、#'.join(str(k + 1) for k in inside)}），"
           f"且峰位都在窗口内——逐分量判据看不到这种拼法"]
    if band is None or not band.get("usable"):
        return "no", why + ["但没算合成阻抗的斜率，**不据此下结论**"]
    if band["band_dec"] < min_dec:
        return "no", why + [f"合成阻抗的 45° 连续段只有 {band['band_dec']:.2f} decade"
                            f"（要求 ≥{min_dec:.1f}）→ 更像宽化 R‖CPE，不是扩散"]
    why.append(f"合成阻抗在 {band['f_lo']:.3g}–{band['f_hi']:.3g} Hz 上连续 "
               f"{band['band_dec']:.2f} decade 呈 {band['slope_med']:+.2f} 斜率"
               f"（占全带 {band['frac_band']:.0%}）")
    why.append("**弱识别**：可能是扩散尾被拆成几段近似；不改变 K 选择，需人工确认")
    return "weak", why


def _selftest():
    print("=" * 68)
    print("bm_diff 自检")
    print("=" * 68)
    ok = True

    # 1) coth / tanh 稳定版与定义一致
    z = np.array([1e-6, 1e-4, 1e-2, 0.5, 5.0, 40.0])
    ref_c = np.where(np.abs(z) < 1e-3, 1 / z ** 2 + 1 / 3 - z ** 2 / 45,
                     (1 / np.tanh(z)) / z)
    ref_t = np.where(np.abs(z) < 1e-3, 1 - z ** 2 / 3 + 2 * z ** 4 / 15,
                     np.tanh(z) / z)
    e_c = np.max(np.abs(_coth_over_z(z) - ref_c) / np.abs(ref_c))
    e_t = np.max(np.abs(_tanh_over_z(z) - ref_t) / np.abs(ref_t))
    print(f"[1] coth 稳定版   rel err = {e_c:.2e}")
    print(f"    tanh 稳定版   rel err = {e_t:.2e}")
    ok &= e_c < 1e-10 and e_t < 1e-10

    # 2) 极端 τ_D 不再溢出
    try:
        big = z_wopen(np.array([1.0, 1e3]), 10.0, 1e8)
        print(f"[2] τ_D=1e8 不溢出：|Z| = {np.abs(big)}")
        ok &= np.all(np.isfinite(big))
    except Exception as e:
        print(f"[2] 失败：{e}")
        ok = False

    # 3) 两个边界条件的高频尾都等于 Warburg
    #    必须在 ωτ_D ≫ 1 的渐近区测（f = f_D 时 |z|=1，coth(1)/1=1.31，差 4%）
    tauD = 1 / (2 * np.pi * 1e3)
    Rw = 1.0
    A = Rw / np.sqrt(tauD)          # A_W = R_W/√τ_D
    f = np.array([1e7])             # |z| = sqrt(2π·1e7·tauD) = 100
    for nm, fn in (("W_open", z_wopen), ("W_short", z_wshort)):
        r = fn(f, Rw, tauD) / z_winf(f, A)
        print(f"[3] {nm} 高频尾 / Warburg = {r[0]:.6f}  (|z|={abs(np.sqrt(1j*2*np.pi*f[0]*tauD)):.0f})")
        ok &= abs(r[0] - 1) < 1e-6

    # 4) 低频极限：open → R_W/3 + 1/(jωC_D)；short → R_W
    f = np.array([1e-6])
    Rw, tauD = 10.0, 1.0
    CD = tauD / Rw
    pred = Rw / 3 + 1 / (1j * 2 * np.pi * f[0] * CD)
    e = abs(z_wopen(f, Rw, tauD)[0] - pred) / abs(pred)
    print(f"[4] W_open 低频 → R_W/3+1/(jωC_D)，rel err = {e:.2e}")
    ok &= e < 1e-4
    e = abs(z_wshort(f, Rw, tauD)[0] - Rw) / Rw
    print(f"    W_short 低频 → R_W，rel err = {e:.2e}")
    ok &= e < 1e-4

    # 5) Warburg 的 DRT：∫γ/(1+jωτ)dlnτ = A/√(jω)
    #    γ∝√τ 两端都慢收敛：τ 下界误差 ~√(ωτ_min)、上界误差 ~1/√(ωτ_max)，
    #    所以区间要拉到 [1e-16, 1e12]，否则看到的全是截断误差而不是公式误差
    A = 2.0
    u = np.linspace(np.log(1e-16), np.log(1e12), 800001)
    tau = np.exp(u)
    gam = gamma_winf(u, A)
    ff = np.array([1e-1, 1e0, 1e1])
    w = 2 * np.pi * ff
    Z = (gam[None, :] / (1 + 1j * w[:, None] * tau[None, :])).sum(axis=1) * (u[1] - u[0])
    e = np.max(np.abs(Z - z_winf(ff, A)) / np.abs(z_winf(ff, A)))
    print(f"[5] γ_W=A√τ/π 重建 Z，max rel err = {e:.2e}（截断误差，恒等式本身精确）")
    ok &= e < 1e-5

    # 6) A_W ↔ R_W 换算
    f_ref = 100.0
    print(f"[6] R_W=5Ω @ {f_ref}Hz → A_W={a_from_rw(5, f_ref):.6g} Ω·s^-0.5，"
          f"回算 R_W={rw_from_a(a_from_rw(5, f_ref), f_ref):.6f}")
    ok &= abs(rw_from_a(a_from_rw(5, f_ref), f_ref) - 5) < 1e-12

    # 7) 退化指纹：伪造一组"贴着脊线"的样本
    rng = np.random.default_rng(0)
    n = 4000
    lnt = rng.normal(np.log(1e5), 1.0, n)
    R = 2.0 * np.exp(0.5 * lnt) * np.exp(rng.normal(0, 0.01, n))
    sig = hn_warburg_signature(R, np.exp(lnt), np.full(n, 0.9),
                               np.full(n, 0.5 / 0.9), 1e-1, 1e3)
    lvl, why = looks_like_warburg(sig)
    print(f"[7] 伪造的 Warburg 脊线：φζ={sig['pz']:.3f}  "
          f"corr={sig['corr_lnR_lntau']:+.3f}  ridge={sig['ridge_ratio']:.3f}  "
          f"窗口外 {sig['outside_dec']:.1f} dec → 判级 = {lvl}")
    print(f"    理由：{'；'.join(why)}")
    ok &= lvl == "strong"

    # 8) 反向：一个正常的窗口内弛豫不该被判成扩散尾
    lnt2 = rng.normal(np.log(1 / (2 * np.pi * 100)), 0.15, n)
    R2 = 20.0 * np.exp(rng.normal(0, 0.05, n))
    sig2 = hn_warburg_signature(R2, np.exp(lnt2), rng.normal(0.85, 0.02, n),
                                np.ones(n), 1e-1, 1e3)
    lvl2, _ = looks_like_warburg(sig2)
    print(f"[8] 窗口内正常弛豫（100 Hz, φ=0.85, ζ=1）：判级 = {lvl2}（应为 no）")
    ok &= lvl2 == "no"

    # 9) DRT 实际峰位：ζ=1 时等于 1/(2πτ)，ζ≠1 时偏，且与暴力解一致
    #    断言"必须与暴力解一致"，而不是"偏移必须够大"——偏移量本身是被测对象，
    #    拿它当阈值等于把答案写进题目里。
    import bm_core as B

    def _brute(ph, ze, half=1.0, n=400001):
        g = np.linspace(-half, half, n)
        y = B._hn_drt_r(g[None, :], np.array([ph])[:, None],
                        np.array([ze])[:, None])[0]
        i = int(np.argmax(y))
        if 0 < i < n - 1:                       # 抛物线插值，消掉量化误差
            y0, y1, y2 = y[i - 1], y[i], y[i + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if abs(den) > 0 else 0.0
        else:
            d = 0.0
        return g[i] + d * (g[1] - g[0])

    # 只挑 φζ≤1 的组合——φζ>1 时峰位按设计给 NaN（见第 [10] 项）
    for ph, ze, tag in ((0.85, 1.0, "ζ=1"), (0.87, 1.10, "ζ=1.10"),
                        (0.87, 0.94, "ζ=0.94"), (0.70, 1.40, "ζ=1.40")):
        tau0 = 1.0 / (2 * np.pi * 100.0)
        r_f = float(hn_peak_ln_tau(np.array([tau0]), np.array([ph]),
                                   np.array([ze]))[0]) - np.log(tau0)
        r_b = _brute(ph, ze)
        ratio = np.exp(-r_f)
        print(f"[9] φ={ph} {tag}: f_peak/f_τ = {ratio:.4f}"
              f"（{np.log10(ratio):+.3f} dec）  暴力解 {np.exp(-r_b):.4f}"
              f"  差={abs(r_f - r_b):.1e}")
        ok &= abs(r_f - r_b) < 1e-3
        if abs(ze - 1.0) < 1e-9:
            ok &= abs(ratio - 1.0) < 1e-6
        else:
            ok &= abs(ratio - 1.0) > 1e-4       # 真的偏了，只是可能很小
    # 1/(2πτ) 与峰位的差距要在指纹里被明确报出来
    sigp = hn_warburg_signature(np.full(200, 10.0), np.full(200, tau0),
                                np.full(200, 0.87), np.full(200, 1.10), 1e-1, 1e3)
    print(f"    指纹里的 peak_shift_dec = {sigp['peak_shift_dec']:+.4f}"
          f"（f_tau={sigp['f_tau']:.3f} Hz, f_peak={sigp['f_peak']:.3f} Hz）")
    ok &= abs(sigp["peak_shift_dec"]) > 1e-2

    # 10) φζ>1：峰位不是物理量，必须给 NaN 并标 nonphysical
    #     0.87×1.31 = 1.14 > 1，解析 γ 会出现负值
    sig_np = hn_warburg_signature(np.full(200, 10.0), np.full(200, tau0),
                                  np.full(200, 0.87), np.full(200, 1.31), 1e-1, 1e3)
    lvl_np, why_np = looks_like_warburg(sig_np)
    print(f"[10] φζ={sig_np['pz']:.2f}>1：f_peak={sig_np['f_peak']}，"
          f"frac_nonphysical={sig_np['frac_nonphysical']:.2f}，判级={lvl_np}")
    print(f"     {why_np[0]}")
    ok &= np.isnan(sig_np["f_peak"]) and np.isnan(sig_np["p_peak_out"])
    ok &= bool(sig_np["nonphysical"]) and lvl_np == "no"

    # 11) 中位数在窗口外 ≠ 多数样本在窗口外
    #     一半样本峰在 0.3 Hz（窗口内），一半在 3e-3 Hz（窗口下方 1.5 dec）
    lnt = np.log(1 / (2 * np.pi * np.r_[np.full(100, 0.3), np.full(100, 3e-3)]))
    sig_mix = hn_warburg_signature(np.full(200, 10.0), np.exp(lnt),
                                   np.full(200, 0.85), np.full(200, 0.9), 1e-1, 1e3)
    print(f"[11] 一半在窗口内一半在窗口外：中位数距窗 {sig_mix['outside_dec']:.2f} dec，"
          f"P(出窗)={sig_mix['p_peak_out']:.2f}")
    ok &= abs(sig_mix["p_peak_out"] - 0.5) < 0.02

    # 12) 容忍阈值：越界比例落在 (NONPHYS_TOL, 0.5) 之间时，
    #     **分量已判非物理**，但**峰位统计仍然可用**——这是两个问题，别合并。
    #     也顺带验证阈值确实来自 NONPHYS_TOL 而不是硬编码的 0.5。
    n = 200
    bad = max(4, int(np.ceil(0.02 * n)))            # 2% > 1% 容忍，但远小于 50%
    pz_mix = np.ones(n) * 0.85
    pz_mix[:bad] = 0.87 * 1.31                      # 这几个 φζ=1.14 > 1
    ph_mix = np.where(np.arange(n) < bad, 0.87, 0.85)
    ze_mix = np.where(np.arange(n) < bad, 1.31, 1.0)
    sig_tol = hn_warburg_signature(np.full(n, 10.0), np.full(n, tau0),
                                   ph_mix, ze_mix, 1e-1, 1e3)
    lvl_tol, _ = looks_like_warburg(sig_tol)
    print(f"[12] {sig_tol['frac_nonphysical'] * 100:.0f}% 越界（容忍 "
          f"{NONPHYS_TOL * 100:.0f}%）：nonphysical={sig_tol['nonphysical']}，"
          f"peak_usable={sig_tol['peak_usable']}，判级={lvl_tol}")
    ok &= bool(sig_tol["nonphysical"])              # 超容忍 → 分量不可用
    ok &= bool(sig_tol["peak_usable"])              # 但峰位算得出来
    ok &= np.isfinite(sig_tol["p_peak_out"])        # 条件概率有值
    ok &= lvl_tol == "no"
    ok &= NONPHYS_TOL < 0.5                         # 阈值确实不是老的 0.5

    print("=" * 68)
    print("全部通过" if ok else "有失败项")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
