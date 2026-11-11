# -*- coding: utf-8 -*-
"""BM-DRT：用贝叶斯混合自动定弛豫峰数（对齐 Liu & Ciucci 2025 原方案）

论文
----
J. Liu, F. Ciucci, "Adaptive Distribution of Relaxation Time Analysis Using
Bayesian Mixtures", J. Phys. Chem. C 2025, 129, 20268-20281.
doi:10.1021/acs.jpcc.5c04766

模型
----
    γ(ln τ) = Σ_{k=1..K} π_k · φ_k(ln τ; θ_k)                 论文式 (3)

  · 基底 φ = Havriliak-Negami（HN），θ = (R, τ0, φ, ζ)        论文式 (15)(17)

        Z_HN(ω) = R / [1 + (jωτ0)^φ]^ζ

    **参数与电路元件的对应（此前写错过，已更正）**：

        ζ=1 且 φ<1    → ZARC，即 R‖CPE（Cole-Cole 型）
        φ=1 且 ζ=1    → 理想 Debye，即 R‖C
        φ=1 且 ζ=1/2  → Gerischer 型

    注意本文件的 φ 支撑是 (0.5, 1) 开区间，取不到 1，所以**严格意义上的
    Debye 与 Gerischer 不在参数空间内**，只能靠 φ→1 逼近。

  一个基底覆盖这三种形态，这是论文选 HN 而不是 Debye/CPE 的理由：
  基底形状与真实 DRT 形状不匹配时，k 必错（论文 §4.1.2 实测）。

  · 权重 π 用 stick-breaking，v_k ~ Beta(1, α)                 论文式 (9)

        π_1 = v_1,  π_k = v_k·Π_{l<k}(1-v_l),  π_K = Π_{l<K}(1-v_l)

    **α < 1 时先验把 v_k 推向 1** → 前面的分量吃掉全部权重，后面自动归零。
    这就是"自动定 k"的机制，也是 K 只当截断上限、不当模型维度的原因。

  · K = 10 是截断上限（论文 §3.2），不是模型复杂度。判据是"若干 π_k 收敛到
    近 0"，而不是"K 选得对不对"。

与论文的差异（不写清楚会误读结果）
--------------------------------
  1) 论文 α ~ Exp(1) 也进采样。这里**默认固定 α**、另做敏感性扫描。
     原因：α 只通过 Beta(1,α) 的 log 先验影响后验，信号很弱，NUTS 在它上面
     混合很差，且不同 α 给出的 k 会不同——固定住反而让"α 是分辨率旋钮"
     这件事变得可见。想复现论文就传 alpha=None 让它采样。
  2) 论文的 σ_re / σ_im 用弱信息先验。这里默认固定（已知噪声时），
     未知时传 sigma=None 让它进采样。
  3) 参数化用 w_k = W·π_k（W 为总极化电阻），使 Σ w_k = W。
     论文把 R 放在基底里、π 只当相对权重，两者等价，但这里少一个
     不可辨识的冗余方向。

     **w_k 的准确含义**：它是该分量的**低频电阻贡献**
     （ω→0 时 Z_k → w_k），同时等于 ∫γ_k dlnτ。
     它**只有在 DRT 非负（即 φ_k·ζ_k ≤ 1）时才等于一个物理的"峰面积"**。
     φ_k·ζ_k > 1 时 γ_k 会出现负值（见 `_hn_drt_unit` 的说明），
     此时 w_k 仍是那个积分值，但**不能再解释成"一个弛豫峰的峰面积"**。
     代码里 `comp_table` 返回的键名仍是 "area"，读的时候请带上这个前提。

已知的坑
--------
  · 基底不匹配时 k 会错，而且**阻抗拟合仍然很好**（论文 §4.1.2 的原话：
    "impedance fitting accuracy alone cannot guarantee correct DRT recovery"）。
    所以看 k 之前先看基底选得对不对，不能只看 χ²。
  · K 增大时总量守恒但各分量幅度会重新分配（论文 §4.1.3）：K=10 和 K=20
    给出同样的两个主弧，K=30 时单个弧的幅度会挪动、区间变宽。这是截断的
    正常表现，不是不稳定。
"""

from __future__ import annotations

import warnings

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# numpyro 只在**采样**时需要；DRT 解析式、后处理、bm_diff 的自检都是纯 numpy。
# 所以放在 try 里：没装 numpyro 的环境仍能 import bm_core、跑 bm_check /
# bm_diff / bm_circuit，只有真正调用 fit() 时才会报错。
try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    numpyro.set_host_device_count(1)    # CPU 上多链顺序执行，墙钟≈单链×链数
    HAS_NUMPYRO = True
except ImportError:                     # pragma: no cover
    numpyro = None
    dist = MCMC = NUTS = None
    HAS_NUMPYRO = False

try:                                    # 扩散元件是可选层，缺了不影响基础模型
    import bm_diff as D
except Exception:                       # pragma: no cover
    D = None

try:                                    # 吸附 RL 元件（M2）同样是可选层
    import bm_ad as A
except Exception:                       # pragma: no cover
    A = None


# --------------------------------------------------------------------------
# 阻抗与 DRT 的解析式（纯 numpy，供前向计算和后处理用）
# --------------------------------------------------------------------------

def _hn_z(f, w, tau, phi, zeta):
    """w / [1 + (jωτ)^φ]^ζ，w 为该分量的直流电阻贡献（= 峰面积）。"""
    x = np.maximum(2 * np.pi * np.asarray(f, float)[:, None] * np.asarray(tau, float)[None, :], 1e-300)
    mag = np.exp(np.asarray(phi, float)[None, :] * np.log(x))
    ang = np.asarray(phi, float)[None, :] * (np.pi / 2)
    return np.asarray(w, float)[None, :] / (1.0 + mag * (np.cos(ang) + 1j * np.sin(ang))) ** np.asarray(zeta, float)[None, :]


def _hn_drt_r(r, phi, zeta):
    """γ 作为 r = ln(τ/τ0) 的函数。广播形状由调用方定。

    单独抽出来，是为了让"求峰位"能在**每个样本自己的网格**上算：
    `_hn_drt_unit` 的签名要求一个公共的 lnτ 网格，做逐样本细化时对不上。

    **两个 φ 不能混**：`a = πφ` 只出现在 `sin/cos/atan2` 里；指数上的
    `q = (τ0/τ)^φ = exp(−φ·r)` 用的是 **φ 本身**，不是 πφ。写成
    `exp(−a*r)` 会让 γ 在 lnτ 轴上被压缩 π 倍：峰值位置对，但宽度错、
    `∫γ dlnτ = 1/π`、ζ≠1 的峰位偏移也会缩水 π 倍（−8.8% 报成 −2.8%）。
    这个错是抽函数时引入的，被 `bm_check` 的 A2/A3 抓到。
    """
    phi = np.asarray(phi, float)
    ze = np.asarray(zeta, float)
    r = np.asarray(r, float)
    a = np.pi * phi
    q = np.exp(-phi * r)
    th = np.arctan2(q * np.sin(a), 1.0 + q * np.cos(a))
    den = (1.0 + 2 * q * np.cos(a) + q ** 2) ** (ze / 2)
    return np.sin(ze * th) / (np.pi * den)


def _hn_drt_unit(ln_tau, ln_tau0, phi, zeta):
    """一般 HN 的归一化 DRT，∫γ dlnτ = 1。

        q  = (τ0/τ)^φ
        θ  = atan2( q·sin(πφ), 1 + q·cos(πφ) )
        γ  = sin(ζθ) / [ π (1 + 2q·cos(πφ) + q²)^{ζ/2} ]

    **ζ=1 时严格退化为 Cole-Cole**（数值核对 max|差| = 1e-15，见 bm_selftest）。

    此前版本把这个公式写错了（把 q 写成 (τ/τ0)^φ、atan2 的分母写成 1/q+cos），
    导致 r→∞ 时 γ→常数、∫γ dlnτ 对数发散，于是误判成"HN 的 DRT 在 ζ=1 处
    不连续"。**那是公式错，不是 HN 的性质**，此处已更正。

    **重要限制：ζφ > 1 时 γ 会出现负值。** 因为 θ∈(0, πφ)，ζθ 可以越过 π。
    此时该分量不对应非负的经典 DRT，物理上不能解释为一个弛豫峰。
    当前模型的参数范围（φ∈(0.5,1)，ζ∈(0.3,2)）允许 ζφ 最大到 2，所以
    **正权重混合并不能保证每个分量的 DRT 都非负**。
    """
    r = np.asarray(ln_tau, float)[:, None] - np.asarray(ln_tau0, float)[None, :]
    return _hn_drt_r(r, np.asarray(phi, float)[None, :], np.asarray(zeta, float)[None, :])


def _cole_drt_unit(ln_tau, ln_tau0, phi):
    """Cole-Cole（ζ=1）的 DRT 的 cosh 闭式，仅用于交叉验证 `_hn_drt_unit`。

        γ(lnτ) = sin((1-φ)π) / { 2π [ cosh(φ·ln(τ/τ0)) − cos((1-φ)π) ] }
    """
    r = np.asarray(ln_tau, float)[:, None] - np.asarray(ln_tau0, float)[None, :]
    ph = np.asarray(phi, float)[None, :]
    a = (1.0 - ph) * np.pi
    return np.sin(a) / (2 * np.pi * (np.cosh(ph * r) - np.cos(a)))


def model_z(f, p, include_L=True):
    """按一组后验样本点算 Z(f)。p 是 dict，含 R_inf / W / pi / tau / phi / zeta / (L)。

    扩散项按 p 里**有没有对应键**决定加不加，所以同一个函数能同时服务
    "基础模型"和"基础模型 + 扩散项"，不需要分支参数。
      · p["Aw"]              → 半无限 Warburg，A_W 单位 Ω·s^(-1/2)
      · p["Rwo"] + p["tauD"] → 有限长阻挡 Warburg（coth）
      · p["Rad"] + p["tau_ad"] → 吸附 RL（M2，Z = R·jωτ/(1+jωτ)）
    """
    z = np.full(len(f), float(p["R_inf"]), complex)
    if include_L and "L" in p:
        z = z + 1j * 2 * np.pi * np.asarray(f, float) * float(p["L"])
    w = float(p["W"]) * np.asarray(p["pi"], float)
    z = z + _hn_z(f, w, p["tau"], p["phi"], p["zeta"]).sum(axis=1)
    if "Aw" in p or ("Rwo" in p and "tauD" in p):
        if D is None:                   # pragma: no cover
            raise ImportError("需要扩散项但 bm_diff 不可用")
        if "Aw" in p:
            z = z + D.z_winf(f, float(p["Aw"]))
        if "Rwo" in p and "tauD" in p:
            z = z + D.z_wopen(f, float(p["Rwo"]), float(p["tauD"]))
    if "Rad" in p and "tau_ad" in p:
        if A is None:                   # pragma: no cover
            raise ImportError("需要吸附项但 bm_ad 不可用")
        z = z + A.z_ad(f, float(p["Rad"]), float(p["tau_ad"]))
    return z


# --------------------------------------------------------------------------
# numpyro 模型
# --------------------------------------------------------------------------

def make_model(K: int = 10, alpha: float | None = 1.0, sigma: tuple | None = None,
               include_L: bool = False, pad_dec: float = 1.5,
               zeta_fixed: float | None = None, phi_lo: float = 0.5,
               diffusion: str | None = None, sigma_floor: float = 0.0,
               adsorption: str | None = None):
    """alpha=None → 让 α~Exp(1) 进采样（复现论文）；否则固定。

    zeta_fixed：把 ζ 钉死（如 1.0 = 纯 ZARC 基底），用于隔离
    "ζ 的自由度是不是采样困难的根源"。None 表示自由。

    sigma_floor：**噪声地板**，相对 `zmax` 的比例（`0.0` = 关闭，逐位等于
    改动前的行为）。开启后 `s = floor + LogNormal(...)`，即 σ 不可能低于
    `sigma_floor·zmax`。

    为什么需要它：`s_re/s_im` 原本是**无下界**的 LogNormal。遇到无噪声
    或近无噪声数据（ECM 正演谱、仿真、极高信噪比测量），只要模型族能精确
    表示实部，似然就把 σ 一路推向 0 —— 实测落在先验中心下方 12σ，形成
    σ→0 的漏斗，NUTS 树深饱和（触顶 53%），采样分级掉到 0，K 选择被全判
    "不可用"。地板把漏斗截断。任何真实测量都有噪声底，所以地板有物理含义，
    但**具体数值必须按仪器精度标定，不能为了让 R̂ 好看而调**。

    diffusion：扩散项，**默认 None = 与改动前逐位一致**。
      · None   基础模型
      · "Ws"   加半无限 Warburg      Z = A_W (jω)^(-1/2)
      · "Wo"   加有限长阻挡 Warburg  Z = R_W coth(√(jωτ_D))/√(jωτ_D)
      · "both" 两项都加（注意：两者高频尾完全相同，同时放会强简并，见 DIFFUSION_PLAN.md）

    参数化说明（为什么要换算）
    --------------------------
    A_W 的量纲是 Ω·s^(-1/2)，直接给它设先验没有尺度感。所以采样的是
    `Rw = A_W/√ω_ref`，即"该扩散项在窗口几何中心处的阻抗幅值"（Ω），
    与 zmax 同量纲。A_W = Rw·√ω_ref 在模型内部算出来。
    Wo 同理用 `Rwo`（Ω），另加 `tauD`（s），拐点频率 1/(2πτ_D)。
    """

    def model(f, Zr, Zi):
        zmax = jnp.max(jnp.sqrt(Zr ** 2 + Zi ** 2))
        fmin, fmax = jnp.min(f), jnp.max(f)
        omega = 2 * jnp.pi * f
        omega_ref = 2 * jnp.pi * jnp.sqrt(fmin * fmax)
        tau_lo = 1.0 / (2 * jnp.pi * fmax) / 10 ** pad_dec
        tau_hi = 1.0 / (2 * jnp.pi * fmin) * 10 ** pad_dec
        l0, l1 = jnp.log(tau_lo), jnp.log(tau_hi)

        R_inf = numpyro.sample("R_inf", dist.LogNormal(jnp.log(0.2 * zmax), 1.0))
        W = numpyro.sample("W", dist.LogNormal(jnp.log(zmax), 1.0))

        if alpha is None:
            a = numpyro.sample("alpha", dist.Exponential(1.0))
        else:
            a = jnp.asarray(float(alpha))

        # ---- stick-breaking ----
        # K=1 时没有 v（plate 尺寸会是 0，numpyro 会断言失败），直接给 π=[1]。
        if K > 1:
            with numpyro.plate("stick", K - 1):
                v = numpyro.sample("v", dist.Beta(1.0, a))
            rem = jnp.concatenate([jnp.ones(1), jnp.cumprod(1.0 - v)[:-1]])
            pi = jnp.concatenate([v * rem, jnp.cumprod(1.0 - v)[-1:]])
        else:
            pi = jnp.ones(1)

        with numpyro.plate("comp", K):
            ln_tau = numpyro.sample("ln_tau", dist.Uniform(l0, l1))
            phi = numpyro.sample("phi", dist.Uniform(phi_lo, 1.0))
            # ζ 必须有下界。ζ→0 时 Z_HN = w/[1+(jωτ)^φ]^ζ → w，即**纯电阻**，
            # 分量可以跑到测量窗口外、冒充 R∞，把 R∞ 抢走（实测 R∞ 从 10 掉到
            # 4.0，残差 2.8 Ω）。文献里 HN 的 ζ 典型落在 0.3~2，取 0.3 作下界。
            if zeta_fixed is None:
                zeta = numpyro.sample("zeta", dist.Uniform(0.3, 2.0))
            else:
                zeta = jnp.full((K,), float(zeta_fixed))

        w = W * pi
        tau = jnp.exp(ln_tau)
        u = omega[:, None] * tau[None, :]
        base = jnp.exp(phi[None, :] * (jnp.log(u) + 0.5j * jnp.pi))
        Zc = (w[None, :] / jnp.exp(zeta[None, :] * jnp.log(1.0 + base))).sum(axis=1)

        Zt = R_inf + Zc

        # ---- 扩散项（默认不加，行为与旧版一致）----
        # 先验刻意放宽到 σ=2.0：A_W 在窗口中心处的幅值可以从 zmax 的千分之一
        # 到几倍，事先不该替数据做主。窄先验会把"没有扩散"硬推成"有一点扩散"。
        if diffusion in ("Ws", "both"):
            if D is None:               # pragma: no cover
                raise ImportError("diffusion='Ws' 但 bm_diff 不可用")
            Rw = numpyro.sample("Rw", dist.LogNormal(jnp.log(0.1 * zmax), 2.0))
            Zt = Zt + D.j_z_winf(omega, Rw * jnp.sqrt(omega_ref))
        if diffusion in ("Wo", "both"):
            if D is None:               # pragma: no cover
                raise ImportError("diffusion='Wo' 但 bm_diff 不可用")
            Rwo = numpyro.sample("Rwo", dist.LogNormal(jnp.log(0.1 * zmax), 2.0))
            # 拐点 1/(2πτ_D)：中心放在窗口几何中心，σ=2 只覆盖 ±1.7 decade，
            # 故意不放到"窗口外很远"——那里本来就不该由数据决定。
            tauD = numpyro.sample("tauD", dist.LogNormal(jnp.log(1.0 / omega_ref), 2.0))
            Zt = Zt + D.j_z_wopen(omega, Rwo, tauD)
        if include_L:
            L = numpyro.sample("L", dist.LogNormal(jnp.log(1e-6), 2.0))
            Zt = Zt + 1j * omega * L
        # ---- 吸附 RL 项（M2，默认不加，行为与旧版一致）----
        # 恒等式 Z_ad = R − R/(1+jωτ)：常数部分与 R_inf 竞争，采样参数是
        # 环直径 Rad 与弛豫时间 tau_ad（L_ad=Rad·tau_ad 只报告不采样，
        # 见 bm_ad 命名表）。先验与 Rw/Rwo、tauD 同尺度同理由：宽 σ=2，
        # 不替数据做主；tau_ad 中心放窗口几何中心，±1.7 dec。
        if adsorption == "RL":
            if A is None:               # pragma: no cover
                raise ImportError("adsorption='RL' 但 bm_ad 不可用")
            Rad = numpyro.sample("Rad", dist.LogNormal(jnp.log(0.1 * zmax), 2.0))
            tau_ad = numpyro.sample("tau_ad",
                                    dist.LogNormal(jnp.log(1.0 / omega_ref), 2.0))
            Zt = Zt + A.j_z_ad(omega, Rad, tau_ad)

        if sigma is None:
            if sigma_floor > 0:
                # 有地板：先采基量，再加地板得到**有效** σ。
                # `s_re` 用 deterministic 落盘，保证报告里的 sigma 就是
                # 似然里真正用的那个，不是基量。
                floor = sigma_floor * zmax
                b_re = numpyro.sample("s_re_base",
                                      dist.LogNormal(jnp.log(0.01 * zmax), 1.0))
                b_im = numpyro.sample("s_im_base",
                                      dist.LogNormal(jnp.log(0.01 * zmax), 1.0))
                s_re = numpyro.deterministic("s_re", b_re + floor)
                s_im = numpyro.deterministic("s_im", b_im + floor)
            else:
                # 默认路径：与加 sigma_floor 之前**逐位一致**，不要改。
                s_re = numpyro.sample("s_re", dist.LogNormal(jnp.log(0.01 * zmax), 1.0))
                s_im = numpyro.sample("s_im", dist.LogNormal(jnp.log(0.01 * zmax), 1.0))
        else:
            # 逐点 σ（与 `bm_top._make_model` 同一语义）：数组按频点广播。
            # **标量输入仍走 `jnp.asarray(float(...))` 的旧路径**，0 维、数值相同，
            # 所以默认行为逐位不变（`bmdrt.py check` 守这条）。
            # 长度不匹配必须显式报错：numpyro 会让 `(n,)` 与 `(n,)` 正常广播，
            # 但错位的数组会静默产出错误的 likelihood，事后从 R̂ 看不出来。
            s_re = jnp.asarray(np.asarray(sigma[0], dtype=float))
            s_im = jnp.asarray(np.asarray(sigma[1], dtype=float))
            for _nm, _s in (("s_re", s_re), ("s_im", s_im)):
                if _s.ndim > 1:
                    raise ValueError(
                        f"sigma[{_nm}] 只能是标量或一维（逐点），收到 ndim={_s.ndim}")
                if _s.ndim == 1 and _s.shape[0] != Zr.shape[0]:
                    raise ValueError(
                        f"sigma[{_nm}] 长度 {_s.shape[0]} 与频点数 {Zr.shape[0]} 不一致")

        numpyro.sample("obs_re", dist.Normal(Zt.real, s_re), obs=Zr)
        numpyro.sample("obs_im", dist.Normal(Zt.imag, s_im), obs=Zi)

    return model


# --------------------------------------------------------------------------
# 后处理
# --------------------------------------------------------------------------

def stick_pi(v: np.ndarray) -> np.ndarray:
    rem = np.concatenate([[1.0], np.cumprod(1.0 - v)[:-1]])
    return np.concatenate([v * rem, [np.cumprod(1.0 - v)[-1]]])


def samples_to_np(s: dict) -> dict:
    out = {k: np.asarray(v, float) for k, v in s.items()}
    if "v" in out:
        out["pi"] = np.array([stick_pi(vv) for vv in out["v"]])
    else:                                  # K=1：只有一个分量，π 恒为 1
        out["pi"] = np.ones((len(out["W"]), 1))
    out["tau"] = np.exp(out["ln_tau"])
    out["w"] = out["W"][:, None] * out["pi"]
    return out


def n_eff(s: dict) -> np.ndarray:
    """权重分布的**熵有效数** exp(H(π))，不是"真实弛豫峰数"。

    性质（注意别误读）：
      · 下界恒为 1（一个分量吃满），**上界**才随 K 增大（K 个等权 → K）。
        所以跨 K 比较时的问题是"上界/截断不同"，不是"地板被抬高"。
      · 一个峰被两个重叠分量拆分 → N_eff 变大；
        两个真实过程面积悬殊 → N_eff 可能远小于 2。
    它只描述权重分散程度，不能当峰数估计量。
    """
    pi = np.clip(s["pi"], 1e-300, None)
    return np.exp(-(pi * np.log(pi)).sum(axis=1))


def comp_table(s: dict, tol: float = 0.01):
    """按权重从大到小排（**不是**按编号——编号在样本间会乱跳）。

    报 tol 以上的分量；每个分量的 τ / 面积 / φ / ζ 都给 5-95% 区间。
    注意：这是"边际"汇总，不是"同一个分量在每张样本里都是第 k 大"的严格匹配。
    权重差异明显时两者一致；差异接近 tol 时区间会偏宽（保守，可接受）。

    （P3 二轮审查：删掉从未被使用的 `sort_by` 死参数——排序固定按平均
    权重，参数挂在签名上会误导调用方以为可以换口径。）

    **"area" 字段的前提**：它是 w_k = ∫γ_k dlnτ（也是 ω→0 时的电阻贡献），
    只有当 φ_k·ζ_k ≤ 1 时 DRT 非负、它才是一个物理意义上的"峰面积"。
    为此每行额外返回 `frac_neg`：该分量的（样本占比）满足 φζ>1 的比例。
    frac_neg > 0 时请把 area 读成"低频电阻贡献"，不要读成"峰面积"。
    """
    pi = s["pi"]
    order = np.argsort(-pi.mean(axis=0))     # 用**平均权重**定一个固定顺序
    rows = []
    for j in order:
        wj = s["w"][:, j]
        if wj.mean() < tol * s["W"].mean():
            continue
        rows.append({
            "f_Hz": float(np.median(1.0 / (2 * np.pi * s["tau"][:, j]))),
            "tau_s": float(np.median(s["tau"][:, j])),
            "area": float(np.median(wj)),
            "area_lo": float(np.percentile(wj, 5)),
            "area_hi": float(np.percentile(wj, 95)),
            "pi": float(np.median(pi[:, j])),
            "phi": float(np.median(s["phi"][:, j])),
            "zeta": float(np.median(s["zeta"][:, j])),
            "zeta_lo": float(np.percentile(s["zeta"][:, j], 5)),
            "zeta_hi": float(np.percentile(s["zeta"][:, j], 95)),
            "frac_neg": float(np.mean(s["phi"][:, j] * s["zeta"][:, j] > 1.0)),
        })
    return rows


def match_components(s: dict, n_iter: int = 3, w_pen: float = 0.25):
    """把每个样本的分量指派到一套**共同编号**上（Hungarian，以 ln τ 为主键）。

    为什么需要
    ----------
    混合模型的编号在样本间自由交换。按 π 降序排只能对齐"主次"，对不齐"身份"：
    两个峰面积接近时，主次本身就在跳（实测 π 不保证递减，v=[0.3,0.9] → π=[0.3,0.63]）。
    直接按编号汇总会把两个峰混成一个，把区间撑得很宽。

    做法
    ----
    代价矩阵 C[i,j] = (lnτ_i − 锚点_j)² + w_pen·(lnw_i − 锚点_j)²，
    逐样本解一次最优指派；锚点初值取"最接近 lnτ 逐列中位数"的那个样本，
    之后用已指派结果的中位数迭代更新 n_iter 轮。

    **重要限制**
    -----------
    这是**事后对齐**，会人为把跳变抹平。所以：
      · 可以用它画 trace、做汇总、算匹配后的区间；
      · **不能**用它证明收敛。收敛必须看原始编号的 R̂/ESS（见 4.7 节）。
    另外它假设"分量数恒定 = K"，多余分量（面积≈0）没有稳定身份，
    会被指派到某个锚点上并显示为大方差——这本身是信息，不是 bug。

    返回 dict：'order' (n,K) 每个样本的列置换，
    以及重排后的 'tau'/'w'/'phi'/'zeta'/'pi'/'f0'（f0 单位 Hz）。
    """
    from scipy.optimize import linear_sum_assignment

    lt = np.log(np.asarray(s["tau"], float))
    lw = np.log(np.maximum(np.asarray(s["w"], float), 1e-12))
    n, K = lt.shape
    med = np.median(lt, axis=0)
    i0 = int(np.argmin(np.abs(lt - med).sum(axis=1)))   # 最"典型"的样本做初值
    anchor_t, anchor_w = lt[i0].copy(), lw[i0].copy()
    order = np.tile(np.arange(K), (n, 1))
    for _ in range(n_iter):
        for i in range(n):
            C = ((lt[i][:, None] - anchor_t[None, :]) ** 2
                 + w_pen * (lw[i][:, None] - anchor_w[None, :]) ** 2)
            r, c = linear_sum_assignment(C)
            # C 的行是"原编号"、列是"锚点"；我们要的是 新编号=锚点 → 原编号
            # 所以是 order[锚点] = 原编号，不是反过来。
            order[i, c] = r
        anchor_t = np.median(np.take_along_axis(lt, order, axis=1), axis=0)
        anchor_w = np.median(np.take_along_axis(lw, order, axis=1), axis=0)
    out = {"order": order}
    for nm in ("tau", "w", "phi", "zeta", "pi"):
        if nm in s:
            out[nm] = np.take_along_axis(np.asarray(s[nm], float), order, axis=1)
    if "tau" in out:
        out["f0"] = 1.0 / (2 * np.pi * out["tau"])
    return out


def k_dist(s: dict, w_frac: float = 0.01) -> dict:
    """k 的分布：权重占比 > w_frac 的分量个数（论文用"π_k 趋近 0"的同一判据）。"""
    k = (s["pi"] > w_frac).sum(axis=1)
    u, c = np.unique(k, return_counts=True)
    return {int(a): float(b) / len(k) for a, b in zip(u, c)}


def drt_band(s: dict, n_grid: int = 240, pad: float = 4.0):
    """后验 DRT 曲线：中位数 + 5/95% 带，用一般 HN 的闭式（`_hn_drt_unit`）。

    **仍然只作可视化**，原因不是公式近似（公式在 ζ=1 处是严格连续的），
    而是模型本身允许 ζφ > 1，此时 γ 会出现负值，曲线不再是一个物理的
    "峰"。函数会在检测到 φζ>1 的样本时发警告并报出占比。

    要定量重建 γ，正确做法是对每个后验样本的 Z 用 drt.py 的 Tikhonov 反演，
    而不是把各分量的解析 DRT 相加。

    逐样本算曲线再取分位，**不能**先平均参数再算曲线（非线性）。
    """
    lt = np.log(s["tau"])
    grid = np.linspace(np.median(lt.min(axis=1)) - pad,
                       np.median(lt.max(axis=1)) + pad, n_grid)
    n = min(len(s["pi"]), 300)
    idx = np.linspace(0, len(s["pi"]) - 1, n).astype(int)
    # ζφ > 1 的分量 DRT 会出负值，不是物理的峰，先报占比
    bad = (s["phi"][idx] * s["zeta"][idx] > 1.0)
    frac = float(bad.sum()) / float(bad.size)
    if frac > 0:
        warnings.warn(
            f"{frac*100:.1f}% 的（分量×样本）满足 φζ>1，其 DRT 出现负值，"
            f"这些样本不是非负弛豫分布，曲线仅作可视化")
    g = np.empty((n, n_grid))
    for a, i in enumerate(idx):
        u = _hn_drt_unit(grid, lt[i], s["phi"][i], s["zeta"][i])
        w = s["W"][i] * s["pi"][i]        # 不依赖派生的 s["w"]，直接算
        g[a] = (w[None, :] * u).sum(axis=1)
    return grid, np.median(g, axis=0), np.percentile(g, 5, axis=0), np.percentile(g, 95, axis=0)


# --------------------------------------------------------------------------
# 拟合入口
# --------------------------------------------------------------------------

def sample_params(s: dict, i: int, include_L: bool = True) -> dict:
    """把第 i 个后验样本拼成 `model_z` 要的 dict。

    集中在这里，是因为"扩散项加不加"要同时影响残差统计、后验预测检验、
    绘图三条路径；散在三处各拼一次，早晚有一处漏掉 Aw，于是"拟合残差"和
    "图上曲线"对不上，而且不会有任何报错。
    """
    p = {"R_inf": s["R_inf"][i], "W": s["W"][i], "pi": s["pi"][i],
         "tau": s["tau"][i], "phi": s["phi"][i], "zeta": s["zeta"][i]}
    if include_L and "L" in s:
        p["L"] = s["L"][i]
    if "Aw" in s:
        p["Aw"] = s["Aw"][i]
    if "Rwo" in s and "tauD" in s:
        p["Rwo"] = s["Rwo"][i]
        p["tauD"] = s["tauD"][i]
    if "Rad" in s and "tau_ad" in s:
        p["Rad"] = s["Rad"][i]
        p["tau_ad"] = s["tau_ad"][i]
    return p


def fit(f, Z, K: int = 10, alpha: float | None = 1.0, sigma: tuple | None = None,
        include_L: bool = False, n_warmup: int = 600, n_samples: int = 1200,
        seed: int = 0, progress: bool = True, num_chains: int = 1,
        target_accept: float = 0.9, zeta_fixed: float | None = None,
        init_values: dict | None = None, max_tree_depth: int = 10,
        extra_fields: tuple = (), diffusion: str | None = None,
        sigma_floor: float = 0.0, adsorption: str | None = None):
    """跑 NUTS。

    num_chains > 1 时才有 R-hat；单链的 R-hat 恒等于 1，不能当收敛证据。
    CPU 上多链是**顺序**执行的（numpyro.set_host_device_count(1)），
    所以 3 链的墙钟时间约为单链的 3 倍。

    init_values：约束空间的值（通常传 MAP），用于 init_to_value 初始化。

    extra_fields：透传给 MCMC.run。默认 numpyro 只存 ("diverging",)，
    要诊断**树深度触顶率**必须显式加 ("num_steps",)，否则事后无法恢复
    （num_steps 不进 get_samples，也不在默认 extra_fields 里）。

    diffusion：透传给 `make_model`。None = 基础模型（与改动前逐位一致）。

    sigma_floor：透传给 `make_model`。0.0 = 关闭（默认，行为不变）。

    adsorption：透传给 `make_model`。None = 基础模型（与改动前逐位一致）；
    "RL" = 采样 Rad / tau_ad 并加 Z_ad（M2，见 bm_ad / ADSORPTION_PROJECT）。
    """
    if not HAS_NUMPYRO:                 # pragma: no cover
        raise ImportError("需要 numpyro 才能采样。纯后处理/自检（bm_check、"
                          "bm_diff、bm_circuit）不需要它。")
    from numpyro.infer import init_to_value

    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    m = make_model(K, alpha=alpha, sigma=sigma, include_L=include_L,
                   zeta_fixed=zeta_fixed, diffusion=diffusion,
                   sigma_floor=sigma_floor, adsorption=adsorption)
    # init_strategy 不能传 None（numpyro 0.19 会走到 substitute(data=None) 分支报
    # "Only one of data or substitute_fn"），只在真要初始化时才加这个 kwarg。
    kw = dict(target_accept_prob=target_accept, max_tree_depth=max_tree_depth)
    if init_values:
        kw["init_strategy"] = init_to_value(values=init_values)
    kern = NUTS(m, **kw)
    mc = MCMC(kern, num_warmup=n_warmup, num_samples=n_samples,
              num_chains=num_chains, progress_bar=progress)
    mc.run(jax.random.PRNGKey(seed), f, Z.real, Z.imag, extra_fields=extra_fields)
    s = samples_to_np(mc.get_samples())
    # ζ 被钉死时模型不采样 zeta，样本里没有这个键；但下游（drt_band、
    # z_chain_diag、match_components）都假设它存在。在这里补成常数数组，
    # 让"固定 ζ"和"自由 ζ"对下游完全同构。
    if zeta_fixed is not None and "zeta" not in s:
        s["zeta"] = np.full((len(s["R_inf"]), K), float(zeta_fixed))
    # 采样的是窗口归一化的 Rw，下游要的是物理的 A_W。换算放在这里做一次，
    # 免得每个后处理函数各自记着乘 sqrt(omega_ref)。
    if "Rw" in s:
        omega_ref = 2 * np.pi * np.sqrt(f.min() * f.max())
        s["Aw"] = s["Rw"] * np.sqrt(omega_ref)
    return s, mc


def fit_residual_stats(s: dict, f, Z, include_L=True, n_max: int = 300):
    """后验**拟合**残差分布：r⁽ˢ⁾ = Z_obs − Z_model(θ⁽ˢ⁾)。

    注意这不是完整的后验预测检验——它没有对观测噪声再抽一次样，
    所以衡量的是"模型能否贴住数据"，不是"模型能否生成这样的数据"。
    真正的检验见 `posterior_predictive_check`。

    **不能**用中位参数算 Z 再比：参数是非线性的，逐个取中位数拼起来的
    那组参数不对应任何一条真实的曲线（K=6 时它给出 rms 3.2 Ω，而实际
    后验里每条样本都能到 0.2 Ω）。必须逐样本算 Z，再对残差取分位。
    """
    m = len(s["pi"])
    idx = np.linspace(0, m - 1, min(n_max, m)).astype(int)
    rms = np.empty(len(idx))
    for a, i in enumerate(idx):
        p = sample_params(s, i, include_L)
        rms[a] = np.sqrt(np.mean(np.abs(Z - model_z(f, p, include_L=include_L)) ** 2))
    return {
        "rms_med": float(np.median(rms)),
        "rms_lo": float(np.percentile(rms, 5)),
        "rms_hi": float(np.percentile(rms, 95)),
        "rel_med": float(np.median(rms) / np.mean(np.abs(Z))),
    }


# 兼容旧名
residual_stats = fit_residual_stats


def posterior_predictive_check(s: dict, f, Z, include_L=True, n_rep: int = 400,
                               level: float = 0.90, seed: int = 0, sigma=None):
    """真正的后验预测检验：Z_rep⁽ˢ⁾ = Z_model(θ⁽ˢ⁾) + ε⁽ˢ⁾。

    逐样本：用该样本的参数算阻抗，再从观测模型抽一次噪声，得到"复制数据"。
    然后检查实测点是否落在复制数据的区间内。

    sigma：固定噪声路径（σ 不进采样、样本里没有 s_re/s_im）下**必须**显式
    传入的噪声标准差（标量或 (s_re, s_im)）。缺省时优先用样本里的
    `s_re`/`s_im`；两者都没有时**直接 raise**——用残差 std 冒充噪声会把
    cover_re/cover_im 变成"把模型自己的拟合误差当成测量误差"的循环论证
    （审查 P2-3）。

    返回
    ----
    cover_re / cover_im : 实测实部/虚部落在 `level` 区间内的比例。
                          **校准良好时应接近 level**。显著低于 → 模型
                          低估了不确定性或存在系统偏差；显著高于 → σ 过大。
    rms_med             : 复制数据与实测的 rms 中位数
    band_lo / band_hi   : 逐频率的预测区间（实部/虚部各一对）
    """
    rng = np.random.default_rng(seed)
    m = len(s["pi"])
    idx = np.linspace(0, m - 1, min(n_rep, m)).astype(int)
    nf = len(f)
    Zr = np.empty((len(idx), nf))
    Zi = np.empty((len(idx), nf))
    # σ 口径：显式入参 > 样本里的 s_re/s_im > 报错（审查 P1-1/P2-3：
    # 不允许用残差 std 等任何"看似合理"的默认值顶替）。
    if sigma is not None:
        if np.ndim(sigma) == 0:
            _sr_fix = _si_fix = float(sigma)
        else:
            _sr_fix, _si_fix = float(sigma[0]), float(sigma[1])
    else:
        _sr_fix = None
    has_sig = "s_re" in s and "s_im" in s
    if _sr_fix is None and not has_sig:
        raise ValueError(
            "posterior_predictive_check：样本里没有 s_re/s_im（固定 σ 路径），"
            "且调用方未传 sigma。请显式传入本次拟合实际使用的 σ。")
    for a, i in enumerate(idx):
        p = sample_params(s, i, include_L)
        mu = model_z(f, p, include_L=include_L)
        if _sr_fix is not None:
            sr, si = _sr_fix, _si_fix
        elif has_sig:
            sr, si = float(s["s_re"][i]), float(s["s_im"][i])
        else:  # 不可达（上方已拦截），防御性保底
            raise ValueError("posterior_predictive_check：σ 口径缺失")
        Zr[a] = mu.real + sr * rng.standard_normal(nf)
        Zi[a] = mu.imag + si * rng.standard_normal(nf)

    q = [50 - level * 50, 50 + level * 50]
    lo_r, hi_r = np.percentile(Zr, q, axis=0)
    lo_i, hi_i = np.percentile(Zi, q, axis=0)
    return {
        "cover_re": float(np.mean((Z.real >= lo_r) & (Z.real <= hi_r))),
        "cover_im": float(np.mean((Z.imag >= lo_i) & (Z.imag <= hi_i))),
        "level": level,
        "rms_med": float(np.median(np.sqrt(np.mean(
            (Z.real - Zr) ** 2 + (Z.imag - Zi) ** 2, axis=1)))),
        "band_lo_re": lo_r, "band_hi_re": hi_r,
        "band_lo_im": lo_i, "band_hi_im": hi_i,
    }


# ==========================================================================
# 合成数据生成（原 synth.py）
# ==========================================================================


# ---------------------------------------------------------------- 真值参数
R_INF = 10.0                 # 串联欧姆电阻，Ω

# 两个 ZARC：中心频率 f0 = 1/(2πτ)，电阻 R，指数 φ
ZARC = [
    dict(f0=357.0, R=33.0, phi=0.85),    # 高频弧
    dict(f0=20.0,  R=15.8, phi=0.90),    # 低频弧
]

SIGMA = 0.2                  # 噪声标准差，Ω（实部、虚部**各自**独立）
SEED = 0
F_LO, F_HI, N_PT = 1e-2, 1e5, 31


def zarc(f, R, f0, phi):
    """Z = R / (1 + (j·2πf·τ)^φ)，τ = 1/(2πf0)。

    注意 ω = 2πf（与 AutoEIS / impedance.py 一致）。数据集里的 ω=f 口径
    会差一个 2π，这里不涉及。
    """
    tau = 1.0 / (2 * np.pi * f0)
    return R / (1.0 + (1j * 2 * np.pi * f * tau) ** phi)


def true_spectrum(f=None):
    """无噪声的 Z(f)。"""
    if f is None:
        f = np.logspace(np.log10(F_LO), np.log10(F_HI), N_PT)
    Z = np.full(np.shape(f), R_INF, complex)
    for e in ZARC:
        Z = Z + zarc(f, e["R"], e["f0"], e["phi"])
    return np.asarray(f, float), Z


def noisy(f=None, sigma=SIGMA, seed=SEED):
    """加噪：Z_obs = Z_true + σ·(ε_re + j·ε_im)，ε_re, ε_im ~ N(0,1) 独立。

    关键点：**绝对**噪声（Ω），不是相对噪声。所以高频端 |Z|≈20 时
    信噪比 ~100，低频端 |Z|≈59 时 ~295。若换成相对噪声，结论会变。
    """
    f, Z = true_spectrum(f)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)
    return f, Z + sigma * eps


def truth_dict():
    """给 bm_drt.model_z 用的真值参数字典（核对用）。"""
    R = np.array([e["R"] for e in ZARC], float)
    f0 = np.array([e["f0"] for e in ZARC], float)
    return {
        "R_inf": R_INF,
        "W": float(R.sum()),
        "pi": R / R.sum(),
        "tau": 1.0 / (2 * np.pi * f0),
        "phi": np.array([e["phi"] for e in ZARC], float),
        "zeta": np.ones(len(ZARC)),          # ZARC ⇒ ζ = 1
    }


if __name__ == "__main__":
    f, Zc = true_spectrum()
    f, Z = noisy()
    print(f"频率 {f.size} 点：{f.min():.2f} ~ {f.max():.0f} Hz（对数等距）")
    print(f"|Z| {np.abs(Z).min():.2f} ~ {np.abs(Z).max():.2f} Ω")
    print(f"真值 R∞={R_INF} Ω，ZARC={ZARC}")
    print(f"总极化电阻 W={sum(e['R'] for e in ZARC)} Ω")
    print(f"噪声 σ={SIGMA} Ω（实部/虚部各自独立），seed={SEED}")
    print(f"实现噪声 rms={np.sqrt(np.mean(np.abs(Z-Zc)**2)):.4f} Ω"
          f"（期望 {SIGMA*np.sqrt(2):.4f}）")
