#!/usr/bin/env python
"""CDC 同构判定：voigt 族与 nested 族是否可能表示**同一个阻抗函数**。

对应 `docs/REFS_DIGEST.md` §2.I。书 p.128–133 的 CDC（电路描述码）规则：
**奇数级括号 = 并联，偶数级 = 串联**，元件依次串在所在层里。用它把两族写成
规范串，先做**结构**比对，再做**函数**比对——两件事不能混：

    · CDC 串不同 ⇔ 结构不同（机械结论，无需拟合）；
    · 但结构不同**不等于**函数不同。本脚本第二部分证明：在纯 RC（φ=ζ=n=1）、
      无感抗支路的条件下，两族**恒等**，且参数映射是闭式的、唯一的。

## 一、结构（CDC）

    voigt  K=2          R(CR)(CR)                → 两个并联块**并排串联**
    nested L=2 无支路    R(C(R(CR)))             → **嵌套**三层
    nested L=2 series   R(C(R(C(R(RL)))))        → 再嵌一层，且 `R∥L` 在最内层

前两者**结构不同**，第三者在结构上根本不含 `L` 的对应物。

## 二、函数（纯 RC，无支路）：两族**恒等**

设 voigt 为 `Zv = Rs + R1/(1+sτ1) + R2/(1+sτ2)`，
nested 为 `Zn = Rs + C1 ∥ (R1n + (C2n ∥ R2n))`，`s = jω`。

两者相减通分，令分子各次系数为 0，得 4 个方程、4 个 nested 未知量。闭式解：

    S    = R1 + R2
    A    = R1·τ2 + R2·τ1
    den  = A·(τ1+τ2) − τ1·τ2·S
    R1n  = A² / den
    τ2n  = den / A
    R2n  = S − R1n
    C1n  = (τ1+τ2−τ2n) / S

**全部为正的充要条件**（本脚本用展开验证，不是数值试出来的）：

    den  = R1τ2² + R2τ1² > 0          → 恒成立
    C1n>0 ⇔ τ1τ2S > 0                 → 恒成立
    R2n>0 ⇔ R2/R1 · (1−τ1/τ2)² > 0    → 恒成立，**除非 τ1 = τ2**

所以：**除「两个时间常数相等」这一退化情形外，每一个 voigt K=2 谱都能被某个
nested L=2 以全正参数精确表示。** 反向同理。这不是"近似不可分"，是恒等。

## 三、一旦带感抗支路，恒等**不可能**

`series` 支路 `R_L + jωL` 与 `parallel` 支路 `R∥L` 的 `Im Z` 在低频都为正，
而纯 RC 的 voigt 有 `Im Z ≤ 0` 恒成立。符号不相容 → 无解。这部分用解析论证
加数值扫描双重确认。

## 四、CPE 指数偏离 1 时，"恒等"退化为"近似"

n≠1 时 `(jω)^n` 不是有理函数，上面的 4 方程 4 未知不再适用。这里做数值扫描：
按 §二 的映射取参数，再让 CPE 指数偏离 1，看相对残差随偏离量增长多快。

用法：`python evidence/nested_topology/m4_cdc_isomorph.py`
"""
from __future__ import annotations

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 一、CDC 规范编码（书 p.128：奇数级括号=并联，偶数级=串联）
# ---------------------------------------------------------------------------


def cdc_voigt(K: int = 2) -> str:
    """`Rs + Σ(C_i ∥ R_i)`，最外层串联、每个并联块套 1 级括号。"""
    return "R" + "(CR)" * int(K)


def cdc_nested(layers: int = 2, branch: str | None = None) -> str:
    """`Rs + CPE1 ∥ (R1 + (CPE2 ∥ (R2 + Z_branch)))`。

    书 p.128 的括号是**层级标记**（奇数级并联、偶数级串联），不是"包住并联项"。
    所以每层 `R + (C ∥ inner)` 要写成 `(C(R inner))`：外层括号翻转层级，
    `R` 与 `inner` 在新层级里串联。拼错一次会多出/少掉一层括号（第一版多了一个 `R`）。

    `branch=None` 表示支路退化掉（`Z_branch=0`），用于与无 L 的 voigt 对拍。
    """
    if branch == "parallel":
        inner = "(RL)"          # R_ad ∥ L_ad → 再深一级并联
    elif branch == "series":
        inner = "RL"            # R_L + jωL → 同级串联
    else:
        inner = ""
    for _ in range(int(layers)):
        inner = "(C(R" + inner + "))"
    return "R" + inner


# ---------------------------------------------------------------------------
# 二、纯 RC 的闭式映射与代数验证
# ---------------------------------------------------------------------------


def map_voigt_to_nested(r1, r2, t1, t2):
    """§二的闭式映射。返回 dict；`R2n<=0`（即 τ1==τ2 退化）时 `R2n` 记为 0。"""
    S = r1 + r2
    A = r1 * t2 + r2 * t1
    den = A * (t1 + t2) - t1 * t2 * S
    r1n = A * A / den
    t2n = den / A
    return dict(R1n=r1n, R2n=S - r1n, C1n=(t1 + t2 - t2n) / S, tau2n=t2n)


def z_voigt_rc(f, r_inf, r1, r2, t1, t2):
    s = 1j * 2 * np.pi * np.asarray(f, float)
    return r_inf + r1 / (1 + s * t1) + r2 / (1 + s * t2)


def z_nested_rc(f, r_inf, R1n, R2n, C1n, tau2n):
    s = 1j * 2 * np.pi * np.asarray(f, float)
    X = R1n + R2n / (1 + s * tau2n)          # R1 + (C2 ∥ R2)，C2 用 τ2/R2 表示
    return r_inf + X / (1 + s * C1n * X)     # C1 ∥ X


def verify_symbolic():
    """用 sympy 独立验证闭式映射（不依赖 §二 的手推）。"""
    import sympy as sp

    s = sp.symbols("s")
    R1, R2, t1, t2 = sp.symbols("R1 R2 tau1 tau2", positive=True)
    R1n, R2n, C1n, t2n = sp.symbols("R1n R2n C1n tau2n", positive=True)

    Zv = R1 / (1 + s * t1) + R2 / (1 + s * t2)
    X = R1n + R2n / (1 + s * t2n)
    Zn = X / (1 + s * C1n * X)

    num = sp.together(Zv - Zn).as_numer_denom()[0]
    poly = sp.Poly(sp.expand(num), s)
    eqs = poly.all_coeffs()
    print(f"  相减通分后分子是 {len(eqs)} 次多项式 → {len(eqs)} 个系数方程，"
          f"{len([R1n, R2n, C1n, t2n])} 个 nested 未知量")

    sol = sp.solve(eqs, [R1n, R2n, C1n, t2n], dict=True)
    print(f"  sympy.solve 给出 {len(sol)} 组解")
    if not sol:
        return False
    got = sol[0]

    S, A = R1 + R2, R1 * t2 + R2 * t1
    den = A * (t1 + t2) - t1 * t2 * S
    expect = {R1n: A**2 / den, R2n: S - A**2 / den,
              C1n: (t1 + t2 - den / A) / S, t2n: den / A}
    ok = True
    for k, v in expect.items():
        d = sp.simplify(got.get(k, sp.nan) - v)
        same = d == 0
        ok &= same
        print(f"    {str(k):5s} 闭式 vs sympy：{'一致' if same else '不一致 → ' + str(d)}")
    return ok


def check_positivity():
    """把 `R2n>0` 化成 `R2/R1·(1−τ1/τ2)²` 并验证符号。"""
    import sympy as sp

    R1, R2, t1, t2 = sp.symbols("R1 R2 tau1 tau2", positive=True)
    S, A = R1 + R2, R1 * t2 + R2 * t1
    den = sp.expand(A * (t1 + t2) - t1 * t2 * S)
    lhs = sp.expand(den * S - A**2)          # R2n>0 ⇔ den·S > A² ⇔ 此式 > 0
    want = sp.expand(R1 * R2 * (t1 - t2) ** 2)
    print(f"  den·S − A² 化简后 = {sp.factor(lhs)}")
    print(f"  期望 R1·R2·(τ1−τ2)² = {sp.factor(want)}")
    same = sp.simplify(lhs - want) == 0
    print(f"  {'一致' if same else '不一致'}"
          f"  → R2n>0 ⇔ R2/R1·(1−τ1/τ2)² > 0，只需 τ1≠τ2")
    return same


def sweep_numeric(n=4000, seed=0, f=None):
    """随机参数：验证映射的数值精度与正性覆盖率。"""
    if f is None:
        f = np.logspace(-2, 5, 50)
    rng = np.random.default_rng(seed)
    worst, n_pos, n_tot = 0.0, 0, n
    bad = []
    for _ in range(n):
        r1 = float(np.exp(rng.uniform(np.log(1), np.log(1e5))))
        r2 = float(np.exp(rng.uniform(np.log(1), np.log(1e5))))
        t1 = float(np.exp(rng.uniform(np.log(1e-5), np.log(1e3))))
        t2 = float(np.exp(rng.uniform(np.log(1e-5), np.log(1e3))))
        if abs(np.log(t2 / t1)) < 1e-6:
            continue
        m = map_voigt_to_nested(r1, r2, t1, t2)
        if min(m.values()) <= 0:
            bad.append((r1, r2, t1, t2, m))
            continue
        n_pos += 1
        za = z_voigt_rc(f, 10.0, r1, r2, t1, t2)
        zb = z_nested_rc(f, 10.0, m["R1n"], m["R2n"], m["C1n"], m["tau2n"])
        rel = np.max(np.abs(za - zb)) / np.max(np.abs(za))
        worst = max(worst, float(rel))
    return dict(tried=n_tot, positive=n_pos, worst_rel=worst, bad=bad[:3])


# ---------------------------------------------------------------------------
# 三、带感抗支路：Im 符号不相容
# ---------------------------------------------------------------------------


def check_inductive():
    """纯 RC voigt 的 Im Z 恒非正；两种感抗支路在低频都给出正 Im。"""
    f = np.logspace(-2, 5, 200)
    w = 2 * np.pi * f
    z = z_voigt_rc(f, 10.0, 500.0, 2000.0, 1e-3, 1e-1)
    im_max = float(np.max(z.imag))

    R_ad, tau = 3000.0, 1.6
    z_par = R_ad * (1j * w * tau) / (1 + 1j * w * tau)      # R ∥ L
    R_L, L = 800.0, 1300.0
    z_ser = R_L + 1j * w * L                                # R + L 串联
    print(f"  voigt(纯 RC) 的 max Im Z = {im_max:+.3e}  （应为 ≤ 0）")
    print(f"  R∥L  支路的 max Im = {z_par.imag.max():+.3e}  "
          f"（f={f[np.argmax(z_par.imag)]:.3g} Hz）")
    print(f"  R+L  支路的 max Im = {z_ser.imag.max():+.3e}  "
          f"（f={f[np.argmax(z_ser.imag)]:.3g} Hz）")
    ok = im_max <= 0 and z_par.imag.max() > 0 and z_ser.imag.max() > 0
    print(f"  → {'符号不相容成立：纯 RC 无法表示含 L 的谱' if ok else '未成立，需复查'}")
    return ok


# ---------------------------------------------------------------------------
# 三·补、高频极限：`ω→∞` 的行为是**结构级**的判据，不需要拟合
# ---------------------------------------------------------------------------


def high_freq_limit():
    """`s→∞` 时各配置的阻抗走向。

    · voigt 无 L → `Rs`（电容全部短路）；
    · voigt 有 L → `sL`，**发散**（串联电感在高频主导）；
    · nested（无论支路）→ `Rs`（`C1 ∥ ...` 里 C1 短路，支路被 C2 挡住）。

    所以 `nested` 永远追不上 `voigt+L` 的高频发散，`voigt` 也永远补不上 `nested`
    的感抗尾——这是**拓扑级**的不相交，与参数无关。
    """
    f = np.array([1e3, 1e4, 1e5, 1e6], float)
    w = 2 * np.pi * f
    rows = []

    zv0 = z_voigt_rc(f, 10.0, 500.0, 2000.0, 1e-3, 1e-1)
    zvL = zv0 + 1j * w * 1e-3                     # voigt + 串联 L
    zn = z_nested_rc(f, 10.0, 3500.0, 4000.0, 2e-4, 2e-1)
    # nested 带 series 支路：Z = C1 ∥ (R1 + (C2 ∥ (R2 + R_L + jωL)))
    s = 1j * w
    C2, R2 = 2e-6, 4000.0
    # 并联必须用**导纳**相加：`A ∥ B = 1/(1/A + 1/B)`，而 CPE/电容的导纳是 `sC`。
    # 第一版写成 `1.0/(C2*s + ...)` 又把 `1/(sC)` 当成导纳，于是高频不但没收敛
    # 反而算出 1.2e3 的假发散——把「nested 高频 → Rs」这条结论反向证伪了。
    X = 3500.0 + 1.0 / (C2 * s + 1.0 / (R2 + 800.0 + 1j * w * 1300.0))
    zns = 10.0 + 1.0 / (2e-4 * s + 1.0 / X)

    for i, fi in enumerate(f):
        rows.append((fi, zv0[i], zvL[i], zn[i], zns[i]))
    print(f"  {'f/Hz':>9s} {'voigt(无L)':>14s} {'voigt+L':>14s} "
          f"{'nested(无支路)':>16s} {'nested(series)':>16s}")
    for fi, a, b, c, d in rows:
        print(f"  {fi:9.0e} {abs(a):14.3g} {abs(b):14.3g} {abs(c):16.3g} {abs(d):16.3g}")
    diverge = abs(rows[-1][2]) > 10 * abs(rows[-1][1])
    print(f"  → voigt+L 在 1 MHz 处幅值是 voigt(无L) 的 "
          f"{abs(rows[-1][2]) / abs(rows[-1][1]):.0f} 倍：高频**发散**，"
          f"nested 两条支路都停在 Rs 附近 {'✓' if diverge else '✗'}")
    return diverge


# ---------------------------------------------------------------------------
# 四、CPE 指数偏离 1：恒等退化为近似
# ---------------------------------------------------------------------------


def cpe_scan(ns=(1.0, 0.98, 0.95, 0.9, 0.8, 0.7)):
    """固定 RC 映射，把 CPE 指数从 1 拉开，看残差增长。"""
    f = np.logspace(-2, 5, 50)
    s = 1j * 2 * np.pi * f
    r1, r2, t1, t2 = 3000.0, 8000.0, 1e-3, 3e-1
    m = map_voigt_to_nested(r1, r2, t1, t2)
    za = z_voigt_rc(f, 10.0, r1, r2, t1, t2)
    zmax = np.max(np.abs(za))

    # n=1 时 CPE 退化为电容：Z_CPE(ω) = 1/(C (jω)^n)。两个 CPE 的"等效电容"
    # 分别取 C1n 与 C2 = tau2n/R2n —— 第二版曾把 C2 写成 C1n，残差恒为 72%
    # 且与 n 完全无关，正是这个笔误的指纹。
    C2 = m["tau2n"] / m["R2n"]
    print(f"  {'n1':>5s} {'n2':>5s}  {'相对残差':>10s}   （n1=n2=1 即纯 RC，应回到 ~1e-9）")
    out = []
    for n1 in ns:
        for n2 in ns:
            Zc1 = 1.0 / (m["C1n"] * s ** n1)
            Zc2 = 1.0 / (C2 * s ** n2)
            X = m["R1n"] + 1.0 / (1.0 / Zc2 + 1.0 / m["R2n"])
            zb = 10.0 + 1.0 / (1.0 / Zc1 + 1.0 / X)
            rel = float(np.max(np.abs(za - zb)) / zmax)
            out.append((n1, n2, rel))
            print(f"  {n1:5.2f} {n2:5.2f}  {rel:10.3%}")
    return out


def main():
    print("=" * 74)
    print("一、CDC 结构比对（书 p.128 规则：奇数级括号=并联、偶数级=串联）")
    print("=" * 74)
    print(f"  voigt K=2          {cdc_voigt(2)}")
    print(f"  voigt K=1          {cdc_voigt(1)}")
    print(f"  nested L=2 无支路   {cdc_nested(2)}")
    print(f"  nested L=2 series  {cdc_nested(2, 'series')}")
    print(f"  nested L=2 parallel{cdc_nested(2, 'parallel')}")
    print("  → 结构**不同**：voigt 是「并联块并排串联」，nested 是「逐层嵌套」。")
    print("    但结构不同不蕴含函数不同，见第二部分。")

    print("\n" + "=" * 74)
    print("二、纯 RC（φ=ζ=n=1）、无支路：两族是否恒等")
    print("=" * 74)
    print("  [sympy 独立求解]")
    ok_sym = verify_symbolic()
    print("\n  [正性条件的展开验证]")
    ok_pos = check_positivity()
    print("\n  [随机参数数值验证]")
    r = sweep_numeric()
    print(f"    试算 {r['tried']} 组，全正解 {r['positive']} 组，"
          f"最大相对残差 {r['worst_rel']:.3e}")
    if r["bad"]:
        print(f"    非正解样例：{r['bad'][0]}")
    # 阈值 1e-6：参数跨 1e-5…1e5 共 10 个数量级，双精度下的相对残差
    # 地板就在 1e-9 附近（实测 2.9e-9），不是模型误差。
    print(f"  → {'恒等成立' if (ok_sym and ok_pos and r['worst_rel'] < 1e-6) else '需要复查'}")

    print("\n" + "=" * 74)
    print("三、带感抗支路：Im 符号不相容")
    print("=" * 74)
    check_inductive()

    print("\n" + "=" * 74)
    print("三·补、高频极限（结构级，不需拟合）")
    print("=" * 74)
    high_freq_limit()

    print("\n" + "=" * 74)
    print("四、CPE 指数偏离 1：恒等 → 近似")
    print("=" * 74)
    cpe_scan()


if __name__ == "__main__":
    main()
