"""cmd_diff —— 扩散项的数学与可辨识性回归测试（不需要 MCMC）。

为什么要有它
------------
`bm_diff.py` 的自检只验元件公式。这一层验的是**决策**：
给定一份已知成分的合成谱，加不加扩散项会不会给出不同结论。
其中 C1 是安全阀——如果没有扩散却被"检出"扩散，整套方案就不能用。

用法与耗时（多起点优化，**对硬件敏感，不要当固定值**）
-----------------------------------------------------
    python bmdrt.py diff          # 全部 6 组，本机 ≈39 s
    python bmdrt.py diff --fast   # 只跑 B1 / B2 / C1，本机 ≈18 s

    --fast 不是"几十秒级的快速检查"：它含 B2 的 7 组多起点优化
    （每组 60 次 restart），单 B2 就占 ~9.5 s，是 --fast 的大头。
    本机逐项：B1 6.4 / B2 9.5 / C1 1.7 / Wo 11.4 / B2-ident 6.7 / DRT 0.1 s。
    另一台机器实测 --fast ≈33.5 s（约 1.9 倍）。按 2~4 倍估预算。
    只想确认代码还能跑 → `python bm_diff.py`（元件自检 12 项，<1 s）。

**常数 σ 最小二乘 ≠ NUTS 似然下的模型选择行为。**
这里所有拟合都是"常数 σ 加权的最小二乘"，它与 `bm_core` 里
`Normal(μ, σ)` 的**点估计**同构（σ 固定时最大化似然 = 最小化平方和），
但两者只在**固定 σ** 下等价。真正跑贝叶斯时：

  · σ 是采样的（`s_re`/`s_im`），似然对残差大的点惩罚更软，K 大时
    后验不会被少数离群点主导，容易比这里"更愿意"接受高 K；
  · 先验（τ 的 Uniform 区间、w 的 LogNormal）会主动收缩，这里没有；
  · LOO 是**留一频率点**的预测，不是残差范数，模型比较口径完全不同。

所以本文件回答的是"**这个参数空间能不能表达它**"（表达力上限），
不是"贝叶斯会选哪个 K"。后者看 `bm_real.py` 的 elpd 与 `run_loo`。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import bm_diff as D


# --------------------------------------------------------------------------
# 参数空间：**必须与 bm_core.make_model 一致**
# --------------------------------------------------------------------------

#: `bm_core.make_model(phi_lo=...)` 的默认值。此前这里写 0.30，与采样器
#: 的 0.50 不一致，导致"K 个 HN 能到 0.9%"这类数字是在一个**采样器永远
#: 到不了**的参数空间里算出来的。默认一律用 0.50。
PHI_LO = 0.50

#: 只在"表达力上限"测试里显式启用：放宽到 0.30，看**即使给足自由度**，
#: HN 能不能补上 Wo 的缺口。结论必须两种下都成立才算稳。
PHI_LO_WIDE = 0.30


# --------------------------------------------------------------------------
# 拟合器：R∞ 自由 + K 个 HN（ζ 固定=1，即 bmdrt 默认）+ 可选扩散项
# --------------------------------------------------------------------------

def _z_hn(f, w, tau, phi):
    x = 2 * np.pi * np.asarray(f, float)[:, None] * np.asarray(tau, float)[None, :]
    return (np.asarray(w, float)[None, :] /
            (1 + np.exp(np.asarray(phi, float)[None, :]
                        * (np.log(x) + 0.5j * np.pi)))).sum(axis=1)


def fit(f, zt, K, diffusion="none", pad=1.5, rel=0.01, seed=0, n_try=60,
        phi_lo=PHI_LO, fix_taud=None):
    """常数 σ 加权最小二乘。返回 dict：相对误差、σ 归一 rms、R∞、分量、扩散参数。

    diffusion : "none" / "Ws"（半无限，1 个参数 A_W）/ "Wo"（coth，2 个参数 R_W, τ_D）
    phi_lo    : 默认与 `bm_core` 一致；只有表达力上限测试才传 `PHI_LO_WIDE`
    fix_taud  : 把 τ_D 钉在这个值上（用等号型 bounds 实现），其余全自由。
                用来做 profile——"如果 τ_D 是 X，最好的残差是多少"。
                这是检验**参数组合**可辨识性的直接补充诊断，**不能单独**
                证明两个参数分别可辨识：网格步长通常比脊线宽，只能看到
                "偏离一个量级代价很大"，看不到脊线内部的平坦方向。
                要判"可分"还得看后验相关性、脊线宽度与先验敏感性
                （见 case_b2_ident 与 DIFFUSION_PLAN.md §3.2）。
    """
    rng = np.random.default_rng(seed)
    sig = rel * np.abs(zt).max()
    fw = np.sqrt(f[0] * f[-1])
    tlo = 1 / (2 * np.pi * f.max()) / 10 ** pad
    thi = 1 / (2 * np.pi * f.min()) * 10 ** pad

    def res(p):
        w = p[1:1 + K]
        tau = np.exp(p[1 + K:1 + 2 * K])
        ph = p[1 + 2 * K:1 + 3 * K]
        zm = p[0] + _z_hn(f, w, tau, ph)
        if diffusion == "Ws":
            zm = zm + D.z_winf(f, p[1 + 3 * K])
        elif diffusion == "Wo":
            zm = zm + D.z_wopen(f, p[1 + 3 * K], p[2 + 3 * K])
        return np.concatenate([(zm.real - zt.real) / sig,
                               (zm.imag - zt.imag) / sig])

    lo = np.concatenate([[0.0], np.full(K, 1e-4), np.full(K, np.log(tlo)),
                         np.full(K, phi_lo)])
    hi = np.concatenate([[1e6], np.full(K, 1e9), np.full(K, np.log(thi)),
                         np.full(K, 1.0)])
    if diffusion == "Ws":
        lo = np.append(lo, 1e-8)
        hi = np.append(hi, 1e3)
    elif diffusion == "Wo":
        lo = np.append(lo, [1e-4, 1e-7])
        hi = np.append(hi, [1e9, 1e7])
    if fix_taud is not None and diffusion == "Wo":
        i = 2 + 3 * K                       # τ_D 的下标
        lo[i] = fix_taud * (1 - 1e-6)       # 不能用相等的 bounds：
        hi[i] = fix_taud * (1 + 1e-6)       # trf 要求 lo<hi 严格成立

    best = None
    for _ in range(n_try):
        p0 = np.concatenate([[abs(zt).max() * rng.uniform(0.05, 0.6)],
                             np.full(K, abs(zt).max() * rng.uniform(0.05, 0.4)),
                             np.log((1 / (2 * np.pi * fw))
                                    * np.exp(rng.normal(0, 2.0, K))),
                             rng.uniform(0.5, 1.0, K)])
        if diffusion == "Ws":
            p0 = np.append(p0, rng.uniform(0, 2))
        elif diffusion == "Wo":
            # R_W 用"窗口中心处幅值"起头；τ_D 在对数尺度上散开
            p0 = np.append(p0, [abs(zt).max() * rng.uniform(0.1, 1.0),
                                (1 / (2 * np.pi * fw)) * np.exp(rng.normal(0, 2.0))])
        p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
        if fix_taud is not None and diffusion == "Wo":
            p0[2 + 3 * K] = fix_taud
        r = least_squares(res, p0, bounds=(lo, hi), max_nfev=6000)
        if best is None or r.cost < best.cost:
            best = r

    p = best.x
    w, tau = p[1:1 + K], np.exp(p[1 + K:1 + 2 * K])
    ph = p[1 + 2 * K:1 + 3 * K]
    zm = p[0] + _z_hn(f, w, tau, ph)
    out = dict(cost=float(best.cost), Rinf=float(p[0]), w=w, tau=tau, phi=ph,
               aw=float("nan"), rw=float("nan"), taud=float("nan"))

    if diffusion == "Ws":
        out["aw"] = float(p[1 + 3 * K])
        zm = zm + D.z_winf(f, out["aw"])
    elif diffusion == "Wo":
        out["rw"] = float(p[1 + 3 * K])
        out["taud"] = float(p[2 + 3 * K])
        zm = zm + D.z_wopen(f, out["rw"], out["taud"])

    o = np.argsort(-(1 / (2 * np.pi * tau)))
    # rms_sig = 残差范数 / σ，即"平均几个 σ"。比 max 相对误差更稳：后者总被
    # |Z| 最小的高频端点主导，会夸大/掩盖真实差距。
    out.update(fpk=(1 / (2 * np.pi * tau))[o], w=w[o], phi=ph[o],
               rel=float(np.max(np.abs(zm - zt) / np.abs(zt))),
               rms_sig=float(np.sqrt(2 * best.cost / (2 * f.size))))
    return out


# --------------------------------------------------------------------------
# 用例
# --------------------------------------------------------------------------

def _f():
    return np.logspace(-1, 3, 31)


def case_b1():
    """B1：1 个 ZARC + 半无限 Warburg。核心命题：加 Ws 后 K 从 2~3 回到 1。"""
    f = _f()
    zt = 10.0 + _z_hn(f, [30.0], [1 / (2 * np.pi * 300)], [0.85]) + D.z_winf(f, 2.0)
    rows = []
    for K, dif in ((1, "none"), (2, "none"), (3, "none"), (1, "Ws")):
        r = fit(f, zt, K, dif, seed=K * 5 + len(dif))
        rows.append((f"K={K}{' +Ws' if dif == 'Ws' else '    '}", r))
    ok = (rows[0][1]["rel"] > 0.02            # K=1 明显差
          and rows[-1][1]["rel"] < 1e-4       # 1+Ws 几乎精确
          and abs(rows[-1][1]["aw"] - 2.0) < 0.05)   # A_W 还原
    print("  B1  R∞=10 + ZARC(30Ω,300Hz,.85) + Ws(A=2)")
    for tag, r in rows:
        extra = f"  A_W={r['aw']:.3f}（真值 2.0）" if np.isfinite(r["aw"]) else ""
        print(f"     {tag}: rel err={r['rel']*100:8.4f}%  R∞={r['Rinf']:5.2f}"
              f"  f_pk={np.round(r['fpk'], 4)}{extra}")
    print(f"     → {'通过' if ok else '** 不通过 **'}")
    return ok


def case_b2():
    """B2（关键对照）：1 个 ZARC + 有限长阻挡 Warburg（Wo）。

    与 B1 的区别在**低频极限**：Ws 的 Warburg 尾是发散的，可以被一个 HN
    推到窗口外顶替；Wo 在低频是有限电容区（R_W/3 + 1/(jωC_D)），必须在
    窗口内真的出现一个拐点。

    这里同时看两个指标：
      rel     = 最大相对误差（直观，但被 |Z| 最小的高频端点主导）
      rms_sig = 残差范数 / σ（"平均几个 σ"，与似然的量纲一致）

    **结论的适用边界（不要外推）**
    ------------------------------
    1. 本测试里 **ζ 固定 = 1**（`_z_hn` 只收 φ），即"φ 自由、ζ=1"的 ZARC
       基底，不是完整的自由 HN（φ 与 ζ 各自自由）。所以能说的只是：
       **在 φ∈[0.5,1]、ζ=1、τ 先验 ±1.5 dec 及本文件的优化条件下，
       HN-only 无法达到显式 Wo 的拟合精度。**
       要推广到 `--zeta free`，必须另跑一组 ζ 自由的对照（尚未跑）。
    2. **K≤6 是有限范围**。"K 从 1 涨到 6 不动"只能说明到 6 为止没看到
       改善，**不能推出"任何有限个 HN 都无法逼近 Wo"**。真正支持"补不平"
       的是 §3 的物理/渐近论证（单分量只有一个渐近指数，要同时给出 −45°
       与 −90° 至少需要两个不同指数，而两个又都要撞 τ 边界）加上
       `case_wo`（Wo 占满全谱，∫γdlnτ 发散）这一独立证据。
    3. 这里是最小二乘，不是 NUTS。贝叶斯行为见 DIFFUSION_PLAN.md §3.3。
    """
    f = _f()
    RW, TD = 30.0, 0.3                     # f_D = 0.53 Hz，拐点整个在窗口内
    zt = 10.0 + _z_hn(f, [30.0], [1 / (2 * np.pi * 300)], [0.85]) + D.z_wopen(f, RW, TD)
    rows = [(f"K={K}     ", fit(f, zt, K, "none", seed=K * 7)) for K in (1, 2, 3, 4, 6)]
    rows.append(("K=1 +Ws ", fit(f, zt, 1, "Ws", seed=31)))
    rows.append(("K=1 +Wo ", fit(f, zt, 1, "Wo", seed=37)))

    floor = min(r["rel"] for _, r in rows[:5])
    r_ws, r_wo = rows[5][1], rows[6][1]
    ok = (floor > 1e-3                       # HN 补不平（最好也只到 2.4%）
          and r_ws["rel"] > 0.05             # 元件选错也补不平
          and r_wo["rel"] < 1e-6             # 只有 Wo 能，且是精确解
          and abs(r_wo["rw"] - RW) / RW < 0.01
          and abs(np.log(r_wo["taud"] / TD)) < 0.02)
    print(f"\n  B2  R∞=10 + ZARC(30Ω,300Hz,.85) + Wo(R_W={RW:.0f}Ω, τ_D={TD} s, "
          f"f_D={1 / (2 * np.pi * TD):.2f}Hz)")
    for tag, r in rows:
        print(f"     {tag}: rel err={r['rel'] * 100:9.4f}%   rms/σ={r['rms_sig']:9.3f}"
              f"   R∞={r['Rinf']:5.2f}   f_pk={np.round(r['fpk'], 3)}")
    print(f"     K=1 +Wo 还原：R_W={r_wo['rw']:.3f} Ω（真值 {RW:.0f}）、"
          f"τ_D={r_wo['taud']:.4g} s（真值 {TD}）")
    print(f"     ** K 从 1 涨到 6 也只是把 rel 从 138% 压到 2.4%，再涨不动；"
          f"而 K=1+Wo 用 6 个参数（K=6 要 19 个）就精确解出 **")
    print(f"     ** 范围限定：ζ 固定=1（ZARC 基底）、K≤6。这只说明"
          f"「在此参数空间与优化条件下 HN-only 达不到 Wo 的精度」，"
          f"不能推出「任意有限个 HN 都无法逼近 Wo」 **")
    print(f"     → {'通过' if ok else '** 不通过 **'}")
    return ok


def case_wo():
    """Wo 占满整个谱时，HN 的表达力缺口有多大（两套 φ 下界都跑）。

    这里故意**不放 ZARC**：Wo 的幅值从 10 Ω 一直涨到 1.6e3 Ω，横跨两个
    量级，HN 的缺口比 B2 大得多。加上 ZARC 之后缺口会缩到 2.4%（见 B2）
    ——因为拐点特征相对整条谱变小了。**两个数字都是真的**，差别只在
    "扩散项占多大比重"，报告时必须说清是哪一种情形。

    **三层论证要分开，不能串成一条链**（这是本节最容易写错的地方）：

    1. `∫γ dlnτ` 发散 vs 有限 HN 面积有限
       → 只能支持"**全频域**上二者不严格等价"。ω→0 极限处 HN 有限、
         Warburg 发散，这是结构性的，与 K 无关。
       → **不能**支持"有限窗口内误差有下界"。反例就在 §2.1：单个 HN 在
         0.1–1e3 Hz 内把半无限 Warburg 逼到 1.2e-6，靠的是把峰推到窗口外
         8 个量级。有限窗口不含完整的低频/高频极限信息，全频域性质推不过来。
    2. 本用例的数值拟合
       → 支持"**在当前 K、参数范围、优化条件下**，有限窗口内仍有明显残差"。
         这是经验结论（28.3%），随窗口、K、φ 下界而变，不是定理。
    3. B2 的贝叶斯对照（§3.3 / §3.4）
       → 支持"在给定先验、噪声、采样条件下，显式 Wo 模型的后验表现更好"。

    所以这里**不写"任何有限个 HN 都补不平"**。正确说法是：全频域严格表示上
    存在 DRT 面积性质上的差异；本测试进一步显示，在所测的有限频率窗口、
    K 范围与参数约束下，HN-only 仍有明显误差。两者结论方向一致，
    但**证据等级不同**，混着写会被"全频域性质推不出窗口误差下界"击穿。
    """
    f = _f()
    zt = 10.0 + D.z_wopen(f, 10.0, 1e-2)
    rows = [(K, fit(f, zt, K, "none", seed=K * 11)) for K in (2, 3, 4, 6)]
    rows_w = [(K, fit(f, zt, K, "none", seed=K * 11, phi_lo=PHI_LO_WIDE))
              for K in (2, 3, 4, 6)]
    floor = min(r["rel"] for _, r in rows)
    floor_w = min(r["rel"] for _, r in rows_w)
    ok = floor > 0.20 and floor_w > 0.20
    print("\n  Wo  R∞=10 + Wo(R_W=10, τ_D=1e-2)，扩散项占满整个谱")
    for K, r in rows:
        print(f"     K={K}  φ≥{PHI_LO:.2f}: rel err={r['rel'] * 100:8.4f}%"
              f"  rms/σ={r['rms_sig']:9.2f}  f_pk={np.round(r['fpk'], 4)}")
    for K, r in rows_w:
        print(f"     K={K}  φ≥{PHI_LO_WIDE:.2f}: rel err={r['rel'] * 100:8.4f}%"
              f"  rms/σ={r['rms_sig']:9.2f}  f_pk={np.round(r['fpk'], 4)}")
    print(f"     → 最好 {floor * 100:.1f}%（φ≥{PHI_LO:.2f}）/ {floor_w * 100:.1f}%"
          f"（φ≥{PHI_LO_WIDE:.2f}）；放宽 φ 也补不平，说明不是先验太窄"
          f" → {'通过' if ok else '** 不通过 **'}")
    return ok


def case_b2_ident():
    """Wo 的可辨识性：**按拐点位置分三种情形**做 profile。

    这是对 `bm_real._diff_line` 那三个分支的数值验证，也是"R_W 与 τ_D
    能不能分开"这个问题的**直接补充诊断**——它固定一个参数、让其余全自由
    重拟合，直接看最优残差怎么变。比 Jacobian 相关系数可靠（后者在无噪声、
    病态的算例上给出 corr≈0.998 的假象，实测过），但**也不能单独定论**：
    网格比脊线宽时，它只能排除"完全不可辨识"，不能证明"分别可辨识"。

      f_D 在窗口内   → 曲线有明确极小 → τ_D 有**有限的后验区间**；
                       但"有限"不等于"R_W 与 τ_D 各自可辨识"（见下）
      f_D ≫ f_max    → 曲线全平，只有 C_D = τ_D/R_W 被定住（R_W ∝ τ_D）
      f_D ≪ f_min    → 曲线单边平，只有 A_W = R_W/√τ_D 被定住（R_W ∝ √τ_D）

    **为什么第一行不能写成"可分"**：profile 的网格步长（×0.1 / ×10）比
    脊线宽度（实测 ±9%）粗，看到的是"离 10 倍代价很大"，看不到脊线内部
    的平坦部分。而且即便后验区间有限，那个有限性也可能主要来自先验支撑
    （τ_D 的 LogNormal σ=2.0）与其他参数的联合约束，不是数据单独定出来的。
    要判"可分"必须看后验的 corr 与脊线宽度（`bm_real` 的 degeneracy 块），
    profile 只能排除"完全不可辨识"。
    """
    f = _f()
    base = 10.0 + _z_hn(f, [30.0], [1 / (2 * np.pi * 300)], [0.85])
    cases = (
        ("f_D 在窗口内", 0.3, (0.1, 1.0, 10.0)),
        ("f_D ≫ f_max ", 1e-5, (0.1, 1.0, 10.0, 100.0)),
        ("f_D ≪ f_min ", 1e2, (1.0, 10.0, 100.0, 1000.0)),
    )
    print("\n  Wo 可辨识性 profile（钉住 τ_D，其余全自由）")
    ok = True
    for tag, TD, mult in cases:
        zt = base + D.z_wopen(f, 30.0, TD)
        grid = TD * np.array(mult)
        cs, rws = [], []
        for t in grid:
            r = fit(f, zt, 1, "Wo", seed=5, fix_taud=float(t))
            cs.append(r["cost"])
            rws.append(r["rw"])
        cs, rws = np.array(cs), np.array(rws)
        d = cs - cs.min()
        print(f"     {tag}（τ_D={TD:g}）")
        print("        τ_D/τ_D0 " + "".join(f"{m:>10g}" for m in mult))
        print("        Δcost    " + "".join(f"{x:>10.3g}" for x in d))
        print("        R_W / Ω  " + "".join(f"{x:>10.3g}" for x in rws))
        if TD == 0.3:
            good = d[0] > 4 and d[-1] > 4
            verdict = ("τ_D 有有限后验区间（**不等于 R_W 与 τ_D 可分**："
                       "网格比脊线粗，看不到脊线内部）" if good
                       else "** 连极小都没有 **")
            print(f"        两翼 Δcost = {d[0]:.1f} / {d[-1]:.1f} > 4 → {verdict}")
            ok &= good
        elif TD < 1:
            cv = np.ptp(grid / rws) / np.median(grid / rws)
            print(f"        C_D=τ_D/R_W 的相对变动 = {cv:.2e} → "
                  f"{'只有 C_D 被定住' if cv < 1e-3 else '** 组合不恒定 **'}")
            ok &= cv < 1e-3
        else:
            a = rws / np.sqrt(grid)
            cv = np.ptp(a) / np.median(a)
            print(f"        A_W=R_W/√τ_D 的相对变动 = {cv:.2e} → "
                  f"{'只有 A_W 被定住' if cv < 1e-2 else '** 组合不恒定 **'}")
            ok &= cv < 1e-2
    print(f"     → {'通过' if ok else '** 不通过 **'}")
    return ok


def case_warburg_drt():
    """Warburg 的 DRT 是 γ=A√τ/π，且 ∫γ dlnτ 发散。"""
    A = 2.0
    u = np.linspace(np.log(1e-16), np.log(1e12), 800001)
    tau, gam = np.exp(u), D.gamma_winf(u, A)
    ff = np.array([1e-1, 1e0, 1e1])
    w = 2 * np.pi * ff
    Z = (gam[None, :] / (1 + 1j * w[:, None] * tau[None, :])).sum(axis=1) * (u[1] - u[0])
    e = np.max(np.abs(Z - D.z_winf(ff, A)) / np.abs(D.z_winf(ff, A)))
    area = np.trapezoid(gam, u)
    print("\n  DRT  γ_W=A√τ/π")
    print(f"     重建 Z 的相对误差 = {e:.2e}（截断误差）")
    print(f"     ∫γ dlnτ 在 τ≤1e12 内已累积到 {area:.3g} Ω 且仍在涨（无界）")
    print(f"     → 全频域上有限个 HN（面积有限）不可能严格等价于它；"
          f"**但这不给出有限窗口内的误差下界**，窗口内能逼多近看 case_wo")
    ok = e < 1e-5
    print(f"     → {'通过' if ok else '** 不通过 **'}")
    return ok


def case_c1():
    """C1（安全阀）：两个真弛豫都在窗口内，Ws 不该被用上。"""
    f = _f()
    zt = 10.0 + _z_hn(f, [30.0, 15.0],
                      [1 / (2 * np.pi * 300), 1 / (2 * np.pi * 3)], [0.85, 0.90])
    r2 = fit(f, zt, 2, "none", seed=14)
    r2w = fit(f, zt, 2, "Ws", seed=15)
    r1w = fit(f, zt, 1, "Ws", seed=13)
    ok = (r2["rel"] < 1e-4                    # 无扩散时 2 个 HN 就够
          and r1w["rel"] > 0.05               # Ws 替代不了真弛豫
          and r2w["aw"] < 1e-3)               # A_W 退化到 0
    print("\n  C1  R∞=10 + HN(30Ω,300Hz,.85) + HN(15Ω,3Hz,.90)，无扩散")
    print(f"     K=2     : rel err={r2['rel']*100:8.4f}%  f_pk={np.round(r2['fpk'], 3)}")
    print(f"     K=1 +Ws : rel err={r1w['rel']*100:8.4f}%  A_W={r1w['aw']:.4g}（救不回来）")
    print(f"     K=2 +Ws : rel err={r2w['rel']*100:8.4f}%  A_W={r2w['aw']:.3g}（应 ≈0）")
    print(f"     → {'通过' if ok else '** 不通过：出现假阳性 **'}")
    return ok


def run(argv) -> int:
    fast = "--fast" in argv
    print("=" * 70)
    print("扩散项回归测试（常数 σ 最小二乘 = 固定 σ 下的点估计，非 NUTS 似然）")
    print(f"参数空间：φ ∈ [{PHI_LO}, 1]、ζ 固定 = 1（ZARC 基底），与 bm_core 默认一致")
    print("结论范围：K≤6 的有限搜索 + 上述参数空间。不可外推到"
          "「任意有限个 HN 都补不平」或 --zeta free。")
    print("证据分三层：∫γ 发散只证「全频域不等价」；case_wo 的 28.3% 是"
          "「当前窗口/K/参数下」的经验值；贝叶斯结论只在给定先验噪声下成立。")
    print("=" * 70)
    results = [("B1", case_b1()), ("B2", case_b2())]
    results.append(("C1", case_c1()))
    if not fast:
        results.append(("Wo", case_wo()))
        results.append(("B2-ident", case_b2_ident()))
        results.append(("DRT", case_warburg_drt()))
    print("\n" + "=" * 70)
    bad = [k for k, v in results if not v]
    print(("全部通过" if not bad else f"失败：{', '.join(bad)}"))
    return 0 if not bad else 1
