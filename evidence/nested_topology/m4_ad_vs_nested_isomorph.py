"""voigt+adsorption 与 nested 是否**结构等价**——纯数值 + 极点结构探针（不采样）。

## 为什么需要这个

§9.4.10 证明了：默认配置（voigt 无 L、无吸附）下两族在纯 RC 时**严格恒等**，
一带感抗支路就**拓扑级不相交**。

现在要给 voigt 打开 `adsorption="RL"`（§9.4.11 B 臂）。必须先回答：

> 补上吸附支路之后，voigt 与 nested 会不会又变回"恒等"？
> 如果会，B 臂的 Δ 会全部塌向 0——那不是"两族对等"，而是**又造出一对不可分的模型**，
> 实验就白跑了。

实验本身答不了这个问题：Δ 小既可能是"都对"，也可能是"都错"。

## 三层结论（从硬到软）

**第一层（精确，与参数无关）——极点位置**

- voigt(K=2)+ad 的阻抗是部分分式 `Rs + R1/(1+sτ1) + R2/(1+sτ2) + Ra·sτa/(1+sτa)`，
  分母恒为 `(1+sτ1)(1+sτ2)(1+sτa)` → **极点只能落在 `−1/τ1, −1/τ2, −1/τa`，
  永远实负**（可被零点抵消，但不会凭空出现复极点）。
- nested(L=2, parallel) 的分母是三次多项式（系数见 `nested_den_coeffs`），
  实测可以出现**复共轭对**（AZ31 量级的一例：`−0.479 ± 0.438i`）。
- 两个有理函数相同 ⇒ 极点集合相同。**故两族不可能全域恒等。**

**第二层（统计）——复极点有多常见**

在 AZ31 量级参数盒里随机采样，约 **1/5** 的参数组会让 nested 出现复极点。
落在这一区域的谱，voigt+ad **严格不能**表示；其余区域尚未证明。

**第三层（数值，弱）——全实极点时能否精确复现**

取极点全实的 nested 谱，用 voigt+ad 多起点最小二乘：最小相对残差 **1.4%–7.2%**，
未找到精确解。但这是**优化器得不到**，不是"不存在"的证明——参数计数上
（3 个极点定 τ、3 个零点定 R）是可能恰好够的。**结论只能写成"未找到"。**

## 用过就删的教训（两处都真踩了）

1. **并联必须用导纳相加。** 第一版把 `z_nested` 写成
   `1/(1/(s*C1) + ...)`——`1/(s*C1)` 是**阻抗**，当导纳用等于又取了一次倒数，
   整条链退化成 `Rs + jωC1`（HF 处 `Im=+12.57` 的指纹）。
   这和 `m4_cdc_isomorph.py` 是**同一个坑**，隔了几天又踩一次。
   现已在 `z_nested` 里对拍过 `bm_top.z_nested`（相对偏差 1.3e-16）。
2. **`least_squares` 只吃实数残差。** 直接 `.ravel()` 复数数组会让每个起点都抛
   `Cannot cast complex128 to float64`，被 `except` 吞掉后全报 `inf`，
   看起来像"结构性不能表示"——其实是探针自己坏了。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# 正演
# ---------------------------------------------------------------------------

def grid(n: int = 60, f_lo: float = 0.01, f_hi: float = 1e5) -> np.ndarray:
    return np.logspace(np.log10(f_lo), np.log10(f_hi), n)


def z_ad(w, R: float, tau: float) -> np.ndarray:
    """吸附支路 `R·jωτ/(1+jωτ)`：DC→0，HF→R，Im>0（感抗）。与 `bm_top.z_rl_parallel` 同式。"""
    s = 1j * w
    return R * s * tau / (1.0 + s * tau)


def z_voigt(w, p) -> np.ndarray:
    """`Rs + Σ R_i/(1+jωτ_i) + [可选 Z_ad]`。p = [Rs,R1,τ1,R2,τ2,(Rad,τad)]。"""
    Rs, R1, t1, R2, t2 = p[:5]
    s = 1j * w
    z = Rs + R1 / (1 + s * t1) + R2 / (1 + s * t2)
    if len(p) > 5:
        z = z + z_ad(w, p[5], p[6])
    return z


def z_nested(w, p, branch: str = "parallel") -> np.ndarray:
    """`Rs + C1 ∥ (R1 + (C2 ∥ (R2 + Z_branch)))`，与 `bm_top.z_nested(n=1)` 同式。

    p = [Rs, R1, C1, R2, C2, R_ad, tau_ad]（C 是真电容，不是项目的窗口幅值 Q；
    换算 `C = 1/(Q·ω_ref)`）。
    """
    Rs, R1, C1, R2, C2, Rad, tad = p
    s = 1j * w
    if branch == "parallel":
        y = z_ad(w, Rad, tad)
    elif branch == "series":
        y = Rad + s * (Rad * tad)          # R_L + jωL，L = R_L·τ_L
    else:
        raise ValueError(branch)
    inner = R2 + y
    # 并联用**导纳**相加（见模块文档「教训 1」）：
    return Rs + 1.0 / (s * C1 + 1.0 / (R1 + 1.0 / (s * C2 + 1.0 / inner)))


# ---------------------------------------------------------------------------
# 极点结构（精确层）
# ---------------------------------------------------------------------------

def nested_den_coeffs(R1, C1, R2, C2, Ra, ta):
    """`Z_nested − Rs` 的分母三次多项式系数 `[a3, a2, a1, 1]`（sympy 求得）。

    `D(s) = a3 s³ + a2 s² + a1 s + 1`
    """
    a3 = C1 * C2 * R1 * ta * (R2 + Ra)
    a2 = (C1 * C2 * R1 * R2 + C1 * R1 * ta + C1 * R2 * ta
          + C1 * Ra * ta + C2 * R2 * ta + C2 * Ra * ta)
    a1 = C1 * R1 + C1 * R2 + C2 * R2 + ta
    return np.array([a3, a2, a1, np.ones_like(np.asarray(a1, float))])


def nested_poles(R1, C1, R2, C2, Ra, ta):
    return np.roots(nested_den_coeffs(R1, C1, R2, C2, Ra, ta))


def voigt_ad_poles(t1, t2, ta):
    """voigt+ad 的极点**恒为** `−1/τ`，全实负。这是结构结论，不是拟合结果。"""
    return -1.0 / np.array([t1, t2, ta], float)


# ---------------------------------------------------------------------------
# 数值可辨识性（弱层）
# ---------------------------------------------------------------------------

def _resid(target_fn, w, z_true, x, kind):
    """`least_squares` 只吃实数残差（见模块文档「教训 2」）。"""
    dz = target_fn(w, _pack(kind, x)) - z_true
    return np.concatenate([dz.real, dz.imag])


def _pack(kind: str, x: np.ndarray):
    v = np.exp(x)
    if kind == "voigt_ad":
        return v
    if kind == "voigt":
        return v[:5]
    return v


def fit_to(target_fn, kind: str, n_par: int, w, z_true, n_start: int = 40):
    """多起点对数参数最小二乘，返回 (最小相对残差, 参数)。"""
    zmax = float(np.max(np.abs(z_true)))
    best = (np.inf, None)
    rng = np.random.default_rng(0)
    base = np.log([10.0, 800.0, 0.1, 2000.0, 1.0, 2000.0, 1.5][:n_par])
    for k in range(n_start):
        x0 = base.copy() if k == 0 else base + rng.normal(0.0, 1.5, n_par)
        try:
            r = least_squares(lambda x: _resid(target_fn, w, z_true, x, kind),
                              x0, method="lm", max_nfev=50000)
        except Exception:
            continue
        rel = float(np.max(np.abs(target_fn(w, _pack(kind, r.x)) - z_true)) / zmax)
        if rel < best[0]:
            best = (rel, _pack(kind, r.x))
    return best


TRUE_NESTED = dict(Rs=10.0, R1=800.0, C1=2.0e-5, R2=2500.0, C2=2.5e-4,
                   Rad=3000.0, tau_ad=1.6)
TRUE_VOIGT = dict(Rs=10.0, R1=900.0, tau1=0.05, R2=2400.0, tau2=2.0,
                  Rad=3000.0, tau_ad=1.6)


def _nz(p):
    return [p[k] for k in ("Rs", "R1", "C1", "R2", "C2", "Rad", "tau_ad")]


def main():
    w = 2 * np.pi * grid()
    print("voigt+ad vs nested：三层判据（极点结构 / 复极点占比 / 数值可辨识性）")
    print(f"网格 {grid().size} 点，{grid()[0]:g}–{grid()[-1]:g} Hz，无噪声\n")

    # ---- 第一层 ----
    print("一、极点结构（精确，与参数无关）")
    pr = nested_poles(TRUE_NESTED["R1"], TRUE_NESTED["C1"], TRUE_NESTED["R2"],
                      TRUE_NESTED["C2"], TRUE_NESTED["Rad"], TRUE_NESTED["tau_ad"])
    print(f"  nested(parallel) 极点："
          + ", ".join(f"{np.real(x):+.4f}{np.imag(x):+.4f}i" for x in pr))
    pv = voigt_ad_poles(TRUE_VOIGT["tau1"], TRUE_VOIGT["tau2"], TRUE_VOIGT["tau_ad"])
    print(f"  voigt(K=2)+ad  极点："
          + ", ".join(f"{x:+.4f}" for x in pv) + "   （恒为实负）")
    has_cplx = bool(np.max(np.abs(pr.imag)) > 1e-9)
    print(f"  → nested {'存在' if has_cplx else '不存在'}复共轭极点"
          f"{'：极点集合不同 ⇒ 两族不可能全域恒等' if has_cplx else ''}\n")

    # ---- 第二层 ----
    print("二、复极点有多常见（AZ31 量级参数盒，n=20000）")
    rng = np.random.default_rng(7)
    n = 20000
    lu = lambda lo, hi: 10 ** rng.uniform(np.log10(lo), np.log10(hi), n)
    a3, a2, a1, _ = nested_den_coeffs(lu(10, 1e4), lu(1e-7, 1e-2),
                                      lu(10, 1e4), lu(1e-7, 1e-2),
                                      lu(1, 1e4), lu(1e-3, 1e2))
    imax = np.empty(n)
    for i in range(n):
        imax[i] = np.max(np.abs(np.roots([a3[i], a2[i], a1[i], 1.0]).imag))
    frac = float((imax > 1e-6).mean())
    print(f"  出现复共轭极点的比例 = {frac:.1%}"
          f"（max|Im| 中位 {np.median(imax):.3g}）\n")

    # ---- 第三层 ----
    print("三、数值可辨识性（多起点最小二乘，无噪声）")
    zN = z_nested(w, _nz(TRUE_NESTED))
    zV = z_voigt(w, [TRUE_VOIGT[k] for k in
                     ("Rs", "R1", "tau1", "R2", "tau2", "Rad", "tau_ad")])
    cases = [
        ("nested(parallel)", zN, "voigt_ad", 7),
        ("nested(parallel)", zN, "voigt", 5),
        ("voigt(无支路)", zV, "nested", 7),
    ]
    print(f"  {'目标':<20}{'用哪族拟合':<14}{'最小相对残差':>14}")
    for tag, zt, kind, npar in cases:
        fn = z_voigt if kind.startswith("voigt") else \
            (lambda ww, p: z_nested(ww, p, "parallel"))
        rel, _ = fit_to(fn, kind, npar, w, zt)
        print(f"  {tag:<20}{kind:<14}{rel:14.2e}")

    # 全实极点区：额外验证"未找到精确解"
    print("\n  全实极点的 nested（未找到精确解，≠不存在解）：")
    for R1, C1, R2, C2, Ra, ta in ((46.9, 3e-6, 1270, 1.5e-4, 52.1, 0.73),
                                   (287, 2.1e-6, 931, 1.7e-5, 251, 1.1)):
        pk = nested_poles(R1, C1, R2, C2, Ra, ta)
        if np.max(np.abs(pk.imag)) > 1e-9:
            continue
        zt = z_nested(w, [10.0, R1, C1, R2, C2, Ra, ta])
        rel, _ = fit_to(z_voigt, "voigt_ad", 7, w, zt, n_start=60)
        print(f"    R1={R1:g} R2={R2:g} Ra={Ra:g} → 残差 {rel:.2e}")

    print("\n" + "=" * 74)
    print("结论（按三层写，不合并成一句）：")
    print("  ① 精确层：nested 的极点可为复共轭，voigt+ad 的极点恒为实负")
    print("     → **两族不可能全域恒等**。开 adsorption 不是把两族变成同一个。")
    print(f"  ② 统计层：约 {frac:.0%} 的参数盒落在复极点区，那里 voigt+ad 严格不能表示。")
    print("  ③ 数值层：全实极点区未找到精确解（1e-2 量级残差），但**未证明不存在**。")
    print("  ⇒ B 臂的 Δ 若变小，可读作「少了系统性吃亏」，但不能读作「两族等价」。")


if __name__ == "__main__":
    main()
