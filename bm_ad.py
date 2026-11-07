# -*- coding: utf-8 -*-
"""低频吸附电感元件（`Z_ad`）。

**这个模块是加法，不改 `bm_core.py` 的既有模型。** M1 阶段只提供元件数学、
初值反演与自检；接入采样器是 M2，自动判定是 M4，见 `ADSORPTION_PROJECT.md`。

元件定义
--------
    Z_ad(ω) = R_ad · (jωτ_ad) / (1 + jωτ_ad),      τ_ad = L_ad / R_ad  [s]

物理实现是 **R_ad 与 L_ad 并联**（`Y = 1/R + 1/(jωL)`）。这是电化学里
"吸附中间体"的等效元件：低频下电感主导（虚部为正，Nyquist 第四象限），
高频下电阻主导（回到实轴）。

**恒等式（本项目的核心洞察）**

    Z_ad = R_ad − R_ad/(1 + jωτ_ad)

也就是说：吸附电感 = 一个常数电阻 + 一个**负 RC 弛豫**。常数部分会被
`R_inf` 吸收，剩下的是一个幅值为负的弛豫分量。两条直接推论：

1. **非负 DRT 看不见它**——标准 DRT 强制 `γ ≥ 0`，而 `Z_ad` 对应
   `γ_ad = −R_ad·δ(lnτ − lnτ_ad)`，是负 δ 峰（见 `gamma_ad`）。
2. 初值可以**直接反演**：低频段取 `Im` 的峰位定 `τ_ad`，峰高定 `R_ad`
   （`-Im` 峰值 = `R/2`，见 `inv_from_lowfreq`）。

可辨识性三区间（`f_c = 1/(2πτ_ad)`，窗口 `[f_min, f_max]`）
-----------------------------------------------------------
| 拐点位置 | 窗内行为 | 可辨识 | 处理 |
|---|---|---|---|
| `f_c ≳ f_max` | `Z ≈ jωR_adτ_ad` | 只有 `L_eff = R_ad·τ_ad` | 交给串联 L，不加 ad |
| `f_min ≲ f_c ≲ f_max` | 完整 RL 半圆 | `R_ad`、`τ_ad` | 可加 |
| `f_c ≲ f_min` | `Z ≈ R_ad`（实部） | 无（被 `R_inf` 吸收） | 不可见，不加 |

命名（避免和既有符号打架）
--------------------------
| 本模块 | 含义 | 不要写成 |
|---|---|---|
| `Z_ad` | 吸附 RL 元件 | `L`（那是高频串联电感） |
| `R_ad` | 环的直径（Ω） | `R`（HN 分量用的是 `w`） |
| `tau_ad` | 弛豫时间（s） | `tauD`（那是扩散的） |
| `L_ad` | 等效电感 `= R_ad·τ_ad`，**只报告不采样** | — |

`L_ad` 是派生量：它与 `R_ad`、`τ_ad` 完全简并，绝不能当第三个自由参数。

数值注意
--------
`(1j*x)/(1+1j*x)` 用**直接式**而不是 `1 − 1/(1+1j*x)`：

- 小 `x` 时直接式的实部是 `x²` 量级（`x=1e-8` → `1e-16`），乘法得到，
  没有相消；写成 `1 − 1/(1+1jx)` 会把实部相消成 0，相对误差 100%。
- 大 `x` 时直接式的分母是 `1+1jx`，**恒不为零**（`x` 为实数），
  所以 `ω→0` 不会除零（`x=0` 时结果是 0，正确）。

自检：`python bm_ad.py`
"""

from __future__ import annotations

import numpy as np

try:                                   # jax 只在前向计算时需要，缺了不影响自检
    import jax
    import jax.numpy as jnp

    # 与 bm_core / bm_diff 保持一致：默认 32 位会让 1e-8 级相对误差冒出来
    jax.config.update("jax_enable_x64", True)
except Exception:                      # pragma: no cover
    jax, jnp = None, None


# --------------------------------------------------------------------------
# 元件
# --------------------------------------------------------------------------

def z_ad(f, Rad, tau_ad):
    """吸附 RL 元件（numpy）：`Z = R·jωτ/(1+jωτ)`。

    参数
    ----
    f : array_like   频率（Hz）
    Rad : float      环的直径（Ω），`-Im` 峰值 = `Rad/2`
    tau_ad : float   弛豫时间（s），拐点 `f_c = 1/(2πτ)`
    """
    x = 2.0 * np.pi * np.asarray(f, float) * tau_ad
    return Rad * (1j * x) / (1.0 + 1j * x)


def j_z_ad(omega, Rad, tau_ad):
    """吸附 RL 元件（jax，供 `make_model` 前向计算）。

    分母 `1+1j*x` 对实数 `x` 恒不为零，`omega=0` 时结果为 0，无需特判。
    """
    if jnp is None:                    # pragma: no cover
        raise ImportError("jax 不可用，无法调用 j_z_ad")
    x = omega * tau_ad
    return Rad * (1j * x) / (1.0 + 1j * x)


# --------------------------------------------------------------------------
# 派生量
# --------------------------------------------------------------------------

def l_eff(Rad, tau_ad):
    """等效串联电感 `L_eff = R_ad·τ_ad`（H）。

    只在拐点**高于窗口**时才有意义：此时 `Z_ad ≈ jωL_eff`，与串联 `L`
    完全简并。**只用于报告，不作为采样参数。**
    """
    return Rad * tau_ad


def f_corner(tau_ad):
    """拐点频率 `f_c = 1/(2πτ_ad)`（Hz）。"""
    return 1.0 / (2.0 * np.pi * tau_ad)


def tau_from_peak(f_peak):
    """由 `-Im` 峰位反算 `τ_ad`。"""
    return 1.0 / (2.0 * np.pi * f_peak)


def gamma_ad(ln_tau, Rad, tau_ad, width: float = 0.05):
    """`Z_ad` 的 DRT：一个**负**的窄峰，面积 `−R_ad`。

    `width → 0` 时趋于 `−R_ad·δ(lnτ − lnτ_ad)`。这里用高斯展宽只为可视化与
    数值积分；**非负 DRT 的反演结果里不可能出现它**——这是 M5 要处理的问题。
    """
    u = np.asarray(ln_tau, float) - np.log(tau_ad)
    g = np.exp(-0.5 * (u / width) ** 2) / (width * np.sqrt(2.0 * np.pi))
    return -Rad * g


# --------------------------------------------------------------------------
# 初值反演（§7.3）
# --------------------------------------------------------------------------

def inv_from_lowfreq(f, Z, frac: float = 0.60, min_pts: int = 6,
                     prominence_min: float = 0.20) -> dict:
    """从低频段反演 `(R_ad, τ_ad)` 作 NUTS 初值。

    步骤（§7.3）：
    1. 在窗口**偏低频的一半**里找 `Im Z` 的内部极大（即 `-Im` 的内部极小）；
    2. `τ_ad ≈ 1/(2π f_peak)`，`R_ad ≈ 2·Im_peak`（峰值 `Im_max = R/2`）。

    为什么不只看"最低 30% 点"：拐点若落在 30%~50% 之间，硬截 30% 会漏掉峰，
    退化成端点极大。这里放宽到最低 `frac`（默认 60%），再用**峰必须在几何中心
    以下**（`f_peak ≤ √(f_min·f_max)`）保证它是低频环，不是中频 HN 的尾巴。

    **可靠性标记 `init_reliable`**（§7.3 的失效条件）——只做**结构性**检查：
    - 峰必须是**内部**极大（端点极大 → 峰在窗口外，不是环）；
    - 峰必须在几何中心以下（`f_peak ≤ √(f_min·f_max)`），排除中频 HN 尾巴；
    - 峰的突出度 `(Im_peak − max(Im_两端))/Im_peak ≥ prominence_min`。

    三条全过才 `init_reliable=True`；否则调用方改用先验中位数起步。

    `re_rel_span`（低频段 `Re` 变化 / 全谱 `Re` 极差）**只作诊断记录，不作硬闸**：
    它是"HN 是否已平台化"的粗糙指标，但量纲依赖于谱里还有多少别的分量——
    纯 RL 合成谱上它恒等于 1（环自己就是全部变化），拿它当门会误杀正确情形。
    真正的门槛留给 M3 合成标定去定，不在 M1 拍脑袋。
    返回值**只作采样起点**，不进先验、不进似然，不影响后验。
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    order = np.argsort(f)
    f, Z = f[order], Z[order]
    n = f.size
    n_sel = min(n, max(int(min_pts), int(round(n * frac))))
    fs, Zs = f[:n_sel], Z[:n_sel]

    im = Zs.imag
    # 内部极大：排除两端点
    k = int(np.argmax(im[1:-1])) + 1 if n_sel >= 3 else 0
    interior = bool(0 < k < n_sel - 1)
    im_peak = float(im[k]) if interior else float(np.max(im))

    f_geom = float(np.sqrt(f[0] * f[-1]))
    f_peak = float(fs[k])
    low_half = bool(f_peak <= f_geom)

    edge = float(max(im[0], im[-1]))
    prominence = float((im_peak - edge) / im_peak) if im_peak > 0 else -np.inf

    re_all = Z.real
    span_all = float(np.max(re_all) - np.min(re_all))
    re_band = Zs.real
    span_band = float(np.max(re_band) - np.min(re_band))
    re_rel_span = float(span_band / span_all) if span_all > 0 else 0.0

    reliable = bool(interior and low_half and im_peak > 0
                    and prominence >= prominence_min)

    tau = tau_from_peak(f_peak) if f_peak > 0 else float("nan")
    Rad = 2.0 * im_peak if im_peak > 0 else float("nan")
    return {"Rad": Rad, "tau_ad": tau, "f_peak": f_peak,
            "im_peak": im_peak, "interior": interior, "low_half": low_half,
            "prominence": prominence, "re_rel_span": re_rel_span,
            "init_reliable": reliable, "n_used": int(n_sel)}


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------

def _selftest():
    print("=" * 68)
    print("bm_ad 自检（M1）")
    print("=" * 68)
    ok = True

    # 1) 恒等式 Z_ad = R − R/(1+jωτ)
    Rad, tau = 10.0, 1.0 / (2 * np.pi * 10.0)
    f = np.logspace(-2, 4, 61)
    x = 2 * np.pi * f * tau
    Z = z_ad(f, Rad, tau)
    ident = Rad - Rad / (1 + 1j * x)
    e = float(np.max(np.abs(Z - ident)) / np.max(np.abs(Z)))
    print(f"[1] 恒等式 Z=R−R/(1+jωτ)   max rel err = {e:.2e}")
    ok &= e < 1e-14

    # 2) 峰位 ωτ=1，峰值 Im=R/2
    xs = np.logspace(-4, 4, 400001)
    Zs = Rad * (1j * xs) / (1 + 1j * xs)
    ipk = int(np.argmax(Zs.imag))
    print(f"[2] 峰位 ωτ = {xs[ipk]:.6f}（理论 1）  Im_max/R = {Zs.imag[ipk]/Rad:.6f}（理论 0.5）")
    ok &= abs(xs[ipk] - 1.0) < 1e-3 and abs(Zs.imag[ipk] / Rad - 0.5) < 1e-6

    # 3) 低频极限 Z ≈ jωRτ（取 ωτ=1e-4，二阶项 ~1e-8）
    xlo = 1e-4
    Zlo = Rad * (1j * xlo) / (1 + 1j * xlo)
    r = abs(Zlo / (1j * xlo * Rad))
    print(f"[3] 低频 Z/(jωRτ) = {r:.9f}（理论 →1）")
    ok &= abs(r - 1) < 1e-8

    # 4) 高频极限 Z → R
    xhi = 1e4
    Zhi = Rad * (1j * xhi) / (1 + 1j * xhi)
    r = abs(Zhi / Rad)
    print(f"[4] 高频 Z/R = {r:.9f}（理论 →1）")
    ok &= abs(r - 1) < 1e-8

    # 5) 无源性 min Re ≥ 0（全域扫）
    rmin = float(np.min(Zs.real) / Rad)
    print(f"[5] 无源性 min Re/R = {rmin:.2e}（须 ≥ 0）")
    ok &= rmin >= 0.0

    # 6) K-K 自洽：R_inf + Z_ad 过 Lin-KK，残差应很小
    #    线性无源系统的充要条件，能提前验证数学实现没写错
    try:
        from bm_kk import lin_kk
        ff = np.logspace(-1, 4, 61)
        Zt = 5.0 + z_ad(ff, 2.0, tau)
        kk = lin_kk(ff, Zt)
        rms = float(np.sqrt(np.mean(kk["res_re"] ** 2 + kk["res_im"] ** 2)))
        mu = kk["mu"]
        # μ 极负 = 所有 R_k 同号为负，正是"负弛豫"的指纹（非负 DRT 的盲区）
        mu_txt = "全负(≈−∞)" if (mu is not None and abs(mu) > 1e6) else f"{mu:.3f}"
        print(f"[6] K-K 自洽 M={kk['M']} mu={mu_txt} "
              f"L_fit={kk['L_fit']*1e6:.3f} μH  rms={rms:.2%}")
        # 吸附环在低频，Lin-KK 的串联 L 抓不到它（L_fit 应接近 0）
        ok &= rms < 0.02 and abs(kk["L_fit"]) < 1e-7
        ok &= (mu is None) or (mu < -1.0)
    except Exception as exc:                       # noqa: BLE001
        print(f"[6] K-K 自洽 跳过（{type(exc).__name__}: {exc}）")

    # 7) jax 版与 numpy 版一致；ω=0 不炸
    if jnp is not None:
        try:
            om = jnp.asarray(2 * np.pi * f)
            Zj = np.asarray(j_z_ad(om, Rad, tau))
            e = float(np.max(np.abs(Zj - Z)) / np.max(np.abs(Z)))
            Z0 = complex(np.asarray(j_z_ad(jnp.asarray(0.0), Rad, tau)))
            print(f"[7] jax vs numpy rel err = {e:.2e}；ω=0 → {Z0}")
            ok &= e < 1e-12 and np.isfinite(Z0) and abs(Z0) < 1e-12
        except Exception as exc:                   # noqa: BLE001
            print(f"[7] jax 检查失败：{type(exc).__name__}: {exc}")
            ok = False
    else:
        print("[7] jax 不可用，跳过")

    # 8) 初值反演：合成谱应还原真值
    ff = np.logspace(-2, 4, 61)
    Rt, taut = 3.0, 1.0 / (2 * np.pi * 0.5)        # 拐点 0.5 Hz，在窗口内
    Zt = 20.0 + z_ad(ff, Rt, taut)                 # 加个大 R_inf 模拟真实体系
    inv = inv_from_lowfreq(ff, Zt - 20.0)
    er = abs(inv["Rad"] - Rt) / Rt
    et = abs(inv["tau_ad"] - taut) / taut
    print(f"[8] 初值反演 R_ad {inv['Rad']:.4f}（真 {Rt}） τ {inv['tau_ad']:.4g}"
          f"（真 {taut:.4g}） prom={inv['prominence']:.2f} "
          f"re={inv['re_rel_span']:.2f} reliable={inv['init_reliable']}")
    ok &= er < 0.05 and et < 0.25 and inv["init_reliable"]

    # 8b) 无环的单调谱：不得误报（峰在端点 → unreliable）
    Zmono = 20.0 + 1.0 / (1j * 2 * np.pi * ff * 1e-3)   # 纯电容尾，Im 单调
    inv2 = inv_from_lowfreq(ff, Zmono - 20.0)
    print(f"[8b] 单调容性尾：interior={inv2['interior']} "
          f"low_half={inv2['low_half']} reliable={inv2['init_reliable']}")
    ok &= not inv2["init_reliable"]

    # 9) 拐点高于窗口 → 退化为串联 L（只有 L_eff 可辨识）
    fh = np.logspace(0, 4, 41)
    tau_small = 1.0 / (2 * np.pi * 1e7)            # 拐点 10 MHz，远高于窗口
    Zh = z_ad(fh, 2.0, tau_small)
    Leff = l_eff(2.0, tau_small)
    r = np.max(np.abs(Zh - 1j * 2 * np.pi * fh * Leff)) / np.max(np.abs(Zh))
    print(f"[9] f_c≫f_max：Z 与 jωL_eff 偏差 = {r:.2%}（L_eff={Leff*1e6:.4f} μH）")
    ok &= r < 0.005

    # 10) 拐点低于窗口 → 被 R_inf 吸收（Z ≈ R_ad）
    fl = np.logspace(2, 4, 21)
    tau_big = 1.0 / (2 * np.pi * 1e-3)             # 拐点 1 mHz，远低于窗口
    Zl = z_ad(fl, 2.0, tau_big)
    r = np.max(np.abs(Zl - 2.0)) / 2.0
    print(f"[10] f_c≪f_min：|Z−R_ad|/R_ad = {r:.2e}（不可辨识）")
    ok &= r < 1e-4

    # 11) DRT：负峰，面积 = −R_ad（非负 DRT 的可达空间之外）
    u = np.linspace(np.log(tau) - 2, np.log(tau) + 2, 400001)
    g = gamma_ad(u, Rad, tau)
    area = float(np.sum(g) * (u[1] - u[0]))
    print(f"[11] γ_ad 面积 = {area:.6f}（理论 −R_ad = −{Rad}），"
          f"min γ = {g.min():.1f}，全负 = {bool(np.all(g <= 0))}")
    ok &= abs(area + Rad) < 1e-3 and np.all(g <= 0) and g.min() < -0.5 * Rad

    print("=" * 68)
    print("全部通过" if ok else "有失败项")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
