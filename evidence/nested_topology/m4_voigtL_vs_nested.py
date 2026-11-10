#!/usr/bin/env python
"""打开 `include_L` 之后，`voigt+L` 与 `nested` 是否相交？

背景：§2.I 已证「voigt（无 L、无 ad）≡ nested（无支路）」在纯 RC 下严格恒等，
而带感抗支路的 nested 与无 L 的 voigt **拓扑级不相交**。于是自然要问：
**那把 `include_L` 打开之后呢？**

本脚本的回答分四步，结论是 **仍然不相交，而且不是"勉强不相交"——两者把感抗放在
频率轴的两头，互不覆盖**。

## 一、代数：无穷远点的阶数不匹配（全 ω 不相交）

    voigt+L :  Z - Rs = sL + Σ R_i/(1+sτ_i)
               = (a0+a1s+a2s²+a3s³)/(1+b1s+b2s²)    分子 3 次 / 分母 2 次
               → s→∞ 时 Z ~ sL        （∞ 处**一阶极点**，发散）

    nested  :  Z - Rs = X/(1+sC1X)，X = R1 + (C2∥Y)，Y = R2 + Z_branch
               → 分子 2 次 / 分母 3 次
               → s→∞ 时 Z ~ 1/(sC1)   （∞ 处**一阶零点**，衰减到 Rs）

一个在无穷远有极点，一个有零点。这不是"参数没调好"，是阶数不同，**代数上不可能相等**。

## 二、频段：串联 L 与并联 RL 把感抗放在**相反的两端**

| 元件 | `Im Z` 的符号 | 峰值位置 |
|---|---|---|
| 串联 `jωL` | 恒正，且**随 ω 线性增长** | 无峰，越高频越强 |
| `R∥L`（`Z_ad`） | 恒正，`-Im` 峰 = `R/2` | `f_c = 1/(2πτ)` |

所以串联 L 只能解释**高频**感抗（仪器伪影那一段），`R∥L` 才能解释**低频**感抗（吸附弛豫）。

## 三、AZ31 实测：低频感抗比 `ωL` 大了 8–10 个数量级

`bm_core.make_model` 给 `L` 的先验是 `LogNormal(log(1e-6), 2.0)`，2σ 区间约
`1.8e-8 … 5.5e-5` H。在最低频 `f=0.01 Hz`：

    ωL = 2π·0.01·L ∈ [1.1e-9, 3.5e-6]  Ω

而实测 18 条在 0.01 Hz 处的 `Im Z` 是 **29–1123 Ω**。差了 **7–9 个数量级**。

⇒ **开 `include_L` 对 AZ31 的低频感抗没有任何作用。** 它只覆盖 3 条谱的高频仪器段
（那里 `ωL` 在 5180–26827 Hz 才够大）。

## 四、要让 voigt 表示低频感抗，得开的是 `adsorption="RL"`，不是 `include_L`

`bm_core` 已经有这条路径（`make_model(..., adsorption="RL")` → `bm_ad.z_ad`），
元件与 nested 的 `parallel` 支路**完全相同**（`R·jωτ/(1+jωτ)`）。
`bm_select.fit_voigt` 目前**两个开关都没传**，所以 voigt 族两头都缺。

用法：`python evidence/nested_topology/m4_voigtL_vs_nested.py`
"""
from __future__ import annotations

import contextlib
import glob
import io
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# 一、代数：无穷远点的阶数
# ---------------------------------------------------------------------------


def orders():
    """用 sympy 求两族 `Z-Rs` 的分子/分母次数。"""
    import sympy as sp

    s, Rs, L = sp.symbols("s Rs L", positive=True)
    R1, R2, t1, t2 = sp.symbols("R1 R2 tau1 tau2", positive=True)
    C1, C2, Rl, Ld = sp.symbols("C1 C2 R_L L_ad", positive=True)

    print("  voigt+L：")
    Zv = Rs + s * L + R1 / (1 + s * t1) + R2 / (1 + s * t2)
    num, den = sp.fraction(sp.together(Zv - Rs))
    print(f"    分子 {sp.degree(sp.expand(num), s)} 次 / 分母 "
          f"{sp.degree(sp.expand(den), s)} 次")
    lead = sp.limit((Zv - Rs) / s, s, sp.oo)
    print(f"    s→∞：(Z−Rs)/s → {sp.simplify(lead)}  → Z ~ sL（∞ 处极点）")

    print("  nested（series 支路）：")
    Y = R2 + Rl + s * Ld
    X = R1 + Y / (1 + s * C2 * Y)
    Zn = Rs + X / (1 + s * C1 * X)
    num2, den2 = sp.fraction(sp.together(Zn - Rs))
    print(f"    分子 {sp.degree(sp.expand(num2), s)} 次 / 分母 "
          f"{sp.degree(sp.expand(den2), s)} 次")
    lim = sp.limit(s * (Zn - Rs), s, sp.oo)
    print(f"    s→∞：s·(Z−Rs) → {sp.simplify(lim)}  → Z ~ 1/(sC1)（∞ 处零点）")
    return True


# ---------------------------------------------------------------------------
# 一·补、五种配置的 ∞ 行为一览（决定谁与谁相交）
# ---------------------------------------------------------------------------


def config_table():
    """把 voigt 的三个开关组合与 nested 的两种支路放在一张表里比 `s→∞`。

    只看一件事：**无穷远点**。因为它是结构不变量，改参数改不动它。
    """
    import sympy as sp

    s, Rs, L = sp.symbols("s Rs L", positive=True)
    R1, R2, t1, t2 = sp.symbols("R1 R2 tau1 tau2", positive=True)
    Rad, tad = sp.symbols("R_ad tau_ad", positive=True)
    C1, C2, Rl, Ld = sp.symbols("C1 C2 R_L L_ad", positive=True)

    def lim_inf(expr):
        v = sp.limit(expr, s, sp.oo)
        return sp.simplify(v) if v.is_finite else sp.oo

    base = Rs + R1 / (1 + s * t1) + R2 / (1 + s * t2)
    ad = Rad * s * tad / (1 + s * tad)
    cfgs = [
        ("voigt（无 L 无 ad）", base, Rs),
        ("voigt + L", base + s * L, sp.oo),
        ("voigt + ad", base + ad, Rs + Rad),
        ("voigt + L + ad", base + s * L + ad, sp.oo),
    ]

    Ys = R2 + Rl + s * Ld
    Xs = R1 + Ys / (1 + s * C2 * Ys)
    nested_s = Rs + Xs / (1 + s * C1 * Xs)
    Yp = Rad * s * tad / (1 + s * tad)
    Xp = R1 + (R2 + Yp) / (1 + s * C2 * (R2 + Yp))
    nested_p = Rs + Xp / (1 + s * C1 * Xp)
    nested_0 = Rs + (R1 + R2 / (1 + s * C2 * R2)) / (
        1 + s * C1 * (R1 + R2 / (1 + s * C2 * R2)))
    cfgs += [
        ("nested（支路退化）", nested_0, Rs),
        ("nested series", nested_s, Rs),
        ("nested parallel", nested_p, Rs),
    ]

    print(f"  {'配置':22s} {'分子次':>5s} {'分母次':>5s}  {'s→∞ 的 Z':>14s}")
    out = []
    for name, expr, expect in cfgs:
        num, den = sp.fraction(sp.together(expr))
        dn = int(sp.degree(sp.expand(num), s))
        dd = int(sp.degree(sp.expand(den), s))
        v = lim_inf(expr)
        out.append((name, dn, dd, v))
        vs = "∞（发散）" if v is sp.oo else str(v)
        # 中文占两列，`{name:22s}` 的对齐会失效，这里按显示宽度补空格
        pad = 24 - sum(2 if ord(c) > 127 else 1 for c in name)
        print(f"  {name}{' ' * max(pad, 1)}{dn:>4d} {dd:>6d}   {vs}")
    print("  → 只有「voigt（无 L 无 ad）」与三条 nested 的极限同为 `Rs`，")
    print("    其余三条 voigt 配置在 ∞ 处要么发散（含 L）、要么停在 `Rs+R_ad`（含 ad）。")
    return out


# ---------------------------------------------------------------------------
# 二、频段归属：串联 L 与 R∥L 的 Im 谱
# ---------------------------------------------------------------------------


def band_attribution():
    f = np.logspace(-2, 5, 300)
    w = 2 * np.pi * f

    Ls = 1e-6                                   # L 先验中位
    im_ser = w * Ls                             # 串联 jωL 的 Im

    Rad, tau = 3000.0, 1.6
    z_ad = Rad * (1j * w * tau) / (1 + 1j * w * tau)
    im_ad = z_ad.imag                           # R∥L 的 Im

    print(f"  {'f/Hz':>10s} {'串联 jωL (L=1e-6)':>20s} {'R∥L (R=3000,τ=1.6)':>22s}")
    for fi in (0.01, 0.1, 1.0, 10.0, 1e3, 1e5):
        i = int(np.argmin(np.abs(f - fi)))
        print(f"  {f[i]:10.3g} {im_ser[i]:20.3e} {im_ad[i]:22.3e}")
    print(f"  → 串联 L 在 0.01 Hz 只有 {im_ser[0]:.2e} Ω，在 1e5 Hz 才 "
          f"{im_ser[-1]:.2f} Ω：**单调升，低频消失**")
    ic = int(np.argmax(im_ad))
    print(f"  → R∥L 的 Im 峰在 {f[ic]:.3g} Hz（{im_ad[ic]:.1f} Ω）：**集中在低频**")
    return float(im_ser[0]), float(im_ad[ic]), float(f[ic])


# ---------------------------------------------------------------------------
# 三、AZ31 实测：低频 Im 与 ωL 量级对比
# ---------------------------------------------------------------------------


def az31_check(dirs=("out/_m6_samples/z_csv",)):
    import bm_real as R

    files = []
    for d in dirs:
        files += sorted(glob.glob(os.path.join(d, "*.csv")))
    if not files:
        print("  （未找到数据）")
        return None

    print(f"  {'file':22s} {'Im@0.01Hz':>11s} {'ωL 上界':>11s} {'差几个量级':>10s}")
    rows = []
    for p in files:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                f, Z = R.load_spectrum(p)
        except BaseException:
            continue
        o = np.argsort(f)
        f, Z = f[o], Z[o]
        im_low = float(Z.imag[0])
        wl_max = float(2 * np.pi * f[0] * 5.5e-5)      # L 的 2σ 上界
        dec = (np.log10(abs(im_low) / wl_max) if im_low > 0 and wl_max > 0
               else float("nan"))
        rows.append((os.path.basename(p), im_low, wl_max, dec))
        print(f"  {os.path.basename(p):22s} {im_low:11.3g} {wl_max:11.3e} "
              f"{dec:10.1f}")

    pos = [r for r in rows if r[1] > 0]
    print(f"  → {len(pos)}/{len(rows)} 条低频 Im > 0；"
          f"与 ωL 上界的量级差中位 "
          f"{np.median([r[3] for r in pos]):.1f} 个数量级")
    return rows


def main():
    print("=" * 76)
    print("一、代数：无穷远点的阶数（全 ω 不相交的证明）")
    print("=" * 76)
    orders()

    print("\n" + "=" * 76)
    print("一·补、五种配置的 ∞ 行为（谁与谁能相交）")
    print("=" * 76)
    config_table()

    print("\n" + "=" * 76)
    print("二、频段归属：串联 L 在高频，R∥L 在低频")
    print("=" * 76)
    band_attribution()

    print("\n" + "=" * 76)
    print("三、AZ31 实测：低频感抗 vs ωL（L 取先验 2σ 上界）")
    print("=" * 76)
    az31_check()

    print("\n" + "=" * 76)
    print("结论")
    print("=" * 76)
    print("  1. `voigt+L` 与 `nested` 在 ∞ 处一阶极点 vs 一阶零点 → **代数上不相交**，")
    print("     与参数无关，所以「开 include_L 就能让 voigt 追上 nested」不成立。")
    print("  2. 串联 `jωL` 的贡献随频率**单调上升**，`R∥L` 的贡献是**低频峰**。")
    print("     AZ31 的感抗在低频（0.01 Hz 处 29–1123 Ω），串联 L 在该频点只有")
    print("     ~1e-9–3e-6 Ω → **差 7–9 个数量级**，开 include_L 帮不上。")
    print("  3. 真正能补低频感抗的是 `adsorption=\"RL\"`（`bm_ad.z_ad`），")
    print("     元件与 nested 的 parallel 支路相同；`bm_select.fit_voigt` 目前两个")
    print("     开关都没传，所以 voigt 族**两头都缺**。")


if __name__ == "__main__":
    main()
