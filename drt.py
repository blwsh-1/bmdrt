"""DRT（弛豫时间分布）粗解 —— 只回答"有几个过程、各自时间常数和电阻"。

为什么要它
----------
candidate_gen 的 `num_peaks` 特征在全库扫描里判别力为零（互信息 6.2%），
因为它是在 log10(-Im) 上做峰值拾取，遇到压扁的弧（n 小）、重叠的弧、
或者 CPE 尾巴就失效。DRT 把"到底几个 RC"变成一个**良态的反卷积问题**：

    Z(ω) = R_inf + ∫ γ(lnτ) / (1 + jωτ) dlnτ

γ 的每个峰 = 一个弛豫过程，峰位给 τ（→ f_peak），峰面积给该过程的 R。
比拾峰有原则得多。

它**不给**什么
--------------
不给拓扑。γ 里有 2 个峰，只说明"有 2 个并联 RC 型过程"，至于是
`[P1,R2]-[P3,R4]` 还是 `R1-[P2,R3]-[P4,R5]`、元件是 C 还是 CPE 还是 W，
DRT 完全看不出来。所以它只能约束"k 和 p0"，不能替代结构搜索。

口径
----
物理上 ω = 2πf，τ = RC，f_peak = 1/(2πτ)。数据集内部用 ω = f 是它自己的
记账方式，不影响这里的换算；两边算出来的 f_peak 应该对得上（实测验证过）。

两个必须知道的坑
----------------
1. 电感在高频给出**负的 γ**，而非负约束会把它压成 0，其贡献被 R_inf 吸收
   → Rs 会被高估。所以 DRT 出来的 R_inf 不能当 Rs 用。
2. λ（正则化强度）直接决定峰数。必须按"残差降到噪声水平"来选，
   不能手调，否则就是在调结论。
"""

from __future__ import annotations

import numpy as np


def _hat_matrix(ln_tau_fine: np.ndarray, ln_tau: np.ndarray) -> np.ndarray:
    """M 个三角基函数（lnτ 上等距、支撑 2Δ）在细网格上的取值，(Nfine, M)"""
    d = ln_tau[1] - ln_tau[0]
    # 第 j 个基：在 [t_{j-1}, t_j] 上升、[t_j, t_{j+1}] 下降
    H = np.zeros((len(ln_tau_fine), len(ln_tau)))
    for j in range(len(ln_tau)):
        lo, mid, hi = ln_tau[j] - d, ln_tau[j], ln_tau[j] + d
        left = (ln_tau_fine - lo) / d
        right = (hi - ln_tau_fine) / d
        H[:, j] = np.clip(np.minimum(left, right), 0.0, 1.0)
    return H


def _kernel(f, tau):
    """A_re[k,j] = ∫ hat_j/(1+x²) dlnτ, A_im[k,j] = ∫ -hat_j·x/(1+x²) dlnτ"""
    w = 2 * np.pi * np.asarray(f, float)
    x = np.outer(w, tau)                       # (nf, ntau)
    Kre = 1.0 / (1.0 + x ** 2)
    Kim = -x / (1.0 + x ** 2)
    return Kre, Kim


def drt(f, Z, sigma: float | None = None, m: int = 60,
        lam: float | None = None, lam_grid=None, tau_pad: float = 1.0,
        verbose: bool = False) -> dict:
    """解 γ(lnτ)。

    sigma : 相对噪声（|Z| 口径）。给 None 时用 0.5% 兜底。
    lam   : 给定则不做扫描；否则按 discrepancy principle 自动选
            （取"残差刚好降到噪声水平"的最强正则化）。
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    n = len(f)
    if sigma is None:
        sigma = 5e-3

    # τ 网格：覆盖测量窗口并两侧各留 1 个数量级（用于吸收窗口外的尾巴）
    fmin, fmax = f.min(), f.max()
    t_lo = 1.0 / (2 * np.pi * fmax) * 10 ** (-tau_pad)
    t_hi = 1.0 / (2 * np.pi * fmin) * 10 ** (tau_pad)
    ln_tau = np.linspace(np.log(t_lo), np.log(t_hi), m)
    tau = np.exp(ln_tau)
    d = ln_tau[1] - ln_tau[0]

    # 细网格积分（三角基是分段线性，细网格足够）
    ln_fine = np.linspace(ln_tau[0] - d, ln_tau[-1] + d, 8 * m + 1)
    dx = ln_fine[1] - ln_fine[0]              # 积分权重 = 细网格步长
    # 注意：这里必须是 dx，不是 d。基宽 d 是"三角底边的一半"，
    # 用 d 当积分权重会让 A 的列和偏大 d/dx 倍，γ 的量级和峰面积全错。
    H = _hat_matrix(ln_fine, ln_tau) * dx
    Kre, Kim = _kernel(f, np.exp(ln_fine))
    A_re = Kre @ H
    A_im = Kim @ H

    # 比例加权：按 |Z| 归一，再把噪声折成 1σ
    s = np.abs(Z) * sigma
    Aw = np.vstack([np.c_[A_re, np.ones(n)] / s[:, None],
                    np.c_[A_im, np.zeros(n)] / s[:, None]])
    bw = np.r_[Z.real / s, Z.imag / s]

    # 二阶差分平滑算子
    L = np.zeros((m - 2, m))          # 只作用在 γ 上，R_inf 列补 0
    for i in range(m - 2):
        L[i, i:i + 3] = [1.0, -2.0, 1.0]

    from scipy.optimize import lsq_linear

    # discrepancy principle 的目标：残差降到噪声水平，即 χ² ≈ 方程个数 2n。
    # 不能用 2n-(m+1)：γ 的 60 个节点不是 60 个独立参数（有平滑约束和
    # 非负约束，有效自由度远小于 m），拿它当 dof 会让目标过严、λ 偏小。
    dof = 2 * n

    def solve(lam_):
        A = np.vstack([Aw, np.c_[lam_ * L, np.zeros(m - 2)]])
        b = np.r_[bw, np.zeros(m - 2)]
        r = lsq_linear(A, b, bounds=(0.0, np.inf), method="trf",
                       max_iter=400)
        x = r.x
        chi2 = float(np.sum((Aw @ x - bw) ** 2))
        return x, chi2

    if lam is None:
        grid = lam_grid if lam_grid is not None else np.logspace(-2, 5, 40)
        rows = []
        for lm in grid:
            x, chi2 = solve(lm)
            reg = float(np.linalg.norm(L @ x[:m]))
            rows.append((lm, x, chi2, reg))
            if verbose:
                print(f"    λ={lm:9.3g}  χ²={chi2:9.2f}  ‖Lγ‖={reg:9.3g}  "
                      f"峰数≈{_count_peaks(x[:m], ln_tau)}")
        # L 曲线拐点：不依赖 σ 的估计，比 discrepancy 稳。
        # 数据里只要有电感（γ 非负表示不了），χ² 就永远降不到 2n，
        # 用 discrepancy 会一路退到最弱正则化，γ 全是毛刺。
        lam, x, chi2 = _lcurve_pick(rows)
    else:
        x, chi2 = solve(lam)

    gamma = x[:m]
    r_inf = float(x[m])
    Z_fit = r_inf + A_re @ gamma + 1j * (A_im @ gamma)
    resid = (Z_fit - Z) / np.abs(Z)

    gm = max(float(gamma.max()), 1e-30)
    return {"f": f, "Z": Z, "tau": tau, "ln_tau": ln_tau, "gamma": gamma,
            "R_inf": r_inf, "lam": float(lam), "chi2": chi2, "dof": dof,
            "Z_fit": Z_fit, "resid": resid, "sigma": sigma,
            "R_total": float(np.sum(gamma) * d),
            # 端部堆积：γ 在网格边界仍很高，说明窗口外还有未分辨的弛豫，
            # 此时不能把它当成一个"峰"，也不能信它的 R
            "edge_low_frac": float(gamma[-1] / gm),
            "edge_high_frac": float(gamma[0] / gm)}


def _count_peaks(gamma, ln_tau, rel: float = 0.02):
    from scipy.signal import find_peaks
    g = np.asarray(gamma, float)
    if g.max() <= 0:
        return 0
    pk, _ = find_peaks(g, prominence=rel * g.max())
    return len(pk)


def _lcurve_pick(rows, tol: float = 0.05):
    """残差比准则（quasi-optimal 的简化版）：取"残差不超过最优值 (1+tol)
    倍的最强正则化"。

    为什么不用 L 曲线几何拐点：χ² 在弱正则化区几乎是平的，随后突然陡升，
    弦距法会落到曲率最大的中段（实测 λ=702，残差 12.6%，把两个峰糊成一个）。
    也不用 discrepancy（χ²≈2n）：数据里有电感时 γ 非负表示不了，
    χ² 永远降不到 2n，会一路退到最弱正则化，γ 全是毛刺。

    残差比准则不依赖 σ 的绝对标定，也不要求模型能拟合到噪声水平，
    对"模型本身有系统偏差"这种情况最稳。
    """
    chi = np.array([r[2] for r in rows], float)
    cmin = float(chi.min())
    thr = cmin * (1.0 + tol)
    ok = np.where(chi <= thr)[0]
    i = int(ok.max()) if len(ok) else 0
    lm, xx, c2, _ = rows[i]
    return lm, xx, c2


def peaks(res: dict, rel: float = 0.03) -> list[dict]:
    """找 γ 的峰：位置 → f_peak，面积 → 该过程的 R。

    面积用"峰两侧的谷"作为积分边界，避免相邻峰重叠时重复计数。
    """
    from scipy.signal import find_peaks

    g = res["gamma"]
    ln_tau = res["ln_tau"]
    d = ln_tau[1] - ln_tau[0]
    tau = res["tau"]
    if g.max() <= 0:
        return []
    idx, _ = find_peaks(g, prominence=rel * g.max())

    out = []
    for i in idx:
        # 每个峰各自向两侧走到最近的局部极小，作为积分边界。
        # 不能用"全表极小点集合"——相邻峰的重叠区会被算两次或漏掉。
        lo = i
        while lo > 0 and g[lo - 1] < g[lo]:
            lo -= 1
        hi = i
        while hi < len(g) - 1 and g[hi + 1] < g[hi]:
            hi += 1
        area = float(np.sum(0.5 * (g[lo:hi] + g[lo + 1:hi + 1])) * d) \
            if hi > lo else 0.0
        out.append({
            "f_peak_Hz": float(1.0 / (2 * np.pi * tau[i])),
            "tau_s": float(tau[i]),
            "gamma_max": float(g[i]),
            "R_ohm": area,
            "idx": int(i),
            "width_dec": float((ln_tau[hi] - ln_tau[lo]) / np.log(10)),
        })
    out.sort(key=lambda r: -r["f_peak_Hz"])
    return out



def prescreen(f, Z, sigma: float | None = None, rel: float = 0.03,
              r_min: float = 0.02):
    """预判断：只用手里的数据做 DRT 反演 + 找峰，**不涉及任何 HN/ZARC 参数**。

    返回 (res, pk, pk_all)：
      res    —— drt() 的原始输出（含 γ 曲线、R_inf、λ、端部堆积比例）
      pk_all —— peaks() 找到的全部峰
      pk     —— 过滤后的峰（喂给拟合做初值、给 K 一个候选范围）

    sigma : 相对噪声（|Z| 口径）。给 None 时用 0.5% 兜底——真实数据本来
            也不知道 σ，所以这是默认路径。
    rel   : 峰显著度阈值（相对最大 γ），0.03 = 至少 3% 峰高才算一个峰。

    **必须再过一道过滤。** 正则化反演会在 τ 网格两端（窗口外）长出伪峰：
    非负约束把拟合不掉的尾巴推到了边界基函数上。实测案例 A 原始找到 4 个峰，
    其中 172 kHz（R=0.67 Ω）与 0.8 Hz（R=0.61 Ω）都是这样来的，
    而两个真峰（310 / 18.7 Hz，29.8 / 16.6 Ω）反而很准。不过滤会把 K 撑大一倍。

    过滤规则：峰位落在**测量频段内**，且面积 ≥ 总面积的 r_min。
    """
    res = drt(f, Z, sigma=sigma)
    pk_all = peaks(res, rel=rel)
    tot = sum(p["R_ohm"] for p in pk_all) or 1.0
    pk = [p for p in pk_all
          if (np.min(f) <= p["f_peak_Hz"] <= np.max(f))
          and p["R_ohm"] >= r_min * tot]
    return res, pk, pk_all

# ------------------------------------------------------------------ 自检
def _selftest():
    """造一个已知答案的谱：Rs + 两个 RC，看能不能原样解出来。"""
    rng = np.random.default_rng(0)
    f = np.logspace(-1, 4, 50)
    w = 2 * np.pi * f
    Rs, units = 18.0, [(33.0, 1 / (2 * np.pi * 357)), (15.8, 1 / (2 * np.pi * 0.0215))]
    Z = Rs + sum(R / (1 + 1j * w * tau) for R, tau in units)
    Z = Z * (1 + 2e-3 * rng.standard_normal(len(Z)))     # 0.2% 噪声

    res = drt(f, Z, sigma=2e-3, verbose=False)
    pk = peaks(res)
    print(f"自检：真值 Rs={Rs}, R={[u[0] for u in units]}, "
          f"f_peak={[round(1/(2*np.pi*u[1]), 4) for u in units]}")
    print(f"      λ={res['lam']:.3g}  R_inf={res['R_inf']:.2f}  "
          f"残差RMS={np.sqrt(np.mean(np.abs(res['resid'])**2))*100:.3f}%")
    for p in pk:
        print(f"      峰 f={p['f_peak_Hz']:10.4g} Hz  R={p['R_ohm']:8.2f}  "
              f"宽 {p['width_dec']:.2f} decade")
    print(f"      端部堆积: 低频端 {res['edge_low_frac']:.2f}，"
          f"高频端 {res['edge_high_frac']:.2f}（>0.2 说明窗口外还有弛豫）")
    # 判据：窗内那个峰（357 Hz）必须对上，R 误差 < 5%
    ok = (len(pk) >= 1 and abs(pk[0]["f_peak_Hz"] / 357 - 1) < 0.25
          and abs(pk[0]["R_ohm"] / 33.0 - 1) < 0.05)
    print(f"      -> 窗内峰 {'通过' if ok else '不通过'}")
    return res, pk


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        from fit_store import load_spectrum_csv

        f, Z = load_spectrum_csv(sys.argv[1])
        res = drt(f, Z, verbose=True)
        print(f"\nλ={res['lam']:.3g}  R_inf={res['R_inf']:.2f} Ω  "
              f"残差RMS={np.sqrt(np.mean(np.abs(res['resid'])**2))*100:.3f}%")
        for p in peaks(res):
            print(f"  峰 f={p['f_peak_Hz']:10.4g} Hz  R={p['R_ohm']:8.2f} Ω  "
                  f"宽 {p['width_dec']:.2f} decade  γmax={p['gamma_max']:.1f}")
    else:
        _selftest()
