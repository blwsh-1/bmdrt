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
        verbose: bool = False, signed: bool = False) -> dict:
    """解 γ(lnτ)。

    sigma : 相对噪声（|Z| 口径）。给 None 时用 0.5% 兜底。
    lam   : 给定则不做扫描；否则按 discrepancy principle 自动选
            （取"残差刚好降到噪声水平"的最强正则化）。
    signed: **M5 方案 A（Plank 2025 Eq. 24/28）**。True 时放开 γ≥0，
            允许负弛豫（感性/吸附环 = 负 RC 权重，`bm_ad` 恒等式
            `R·jωτ/(1+jωτ) = R − R/(1+jωτ)`）。代价是欧姆偏移被叠加：
            `R₀,DRT = R₀,true + Σ_{γ<0}|γ|`，因此返回修正量
            `R_inf_true = R_inf − neg_pol`（neg_pol 按 dlnτ 加权）。
            False（默认）时行为与引入 signed 之前**逐位一致**——这是
            `test_drt_signed.case4` 锁死的回归红线。
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
        # signed=False：与原实现完全一致（bounds=(0,inf)）。
        # signed=True：放开非负约束（Plank 2025 的 ϒ ∈ ℝ），R_inf 列本来就
        # 无约束；负 γ 叶 = 感性/吸附弛豫（bm_ad 恒等式的基系数形式）。
        bnds = (-np.inf, np.inf) if signed else (0.0, np.inf)
        r = lsq_linear(A, b, bounds=bnds, method="trf",
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
        if signed:
            # 有符号模型的核族覆盖了感性/吸附弛豫（Plank 2025 的核心论点），
            # χ² 可以真正降到噪声水平 —— 这里 discrepancy 原则成立：
            # 取"χ² 仍 ≤ 2n 的最强正则化"。非负路径不敢用它（模型缺项时
            # χ² 永远降不到 2n，会退到最弱正则化），所以两分支不同。
            lam, x, chi2 = _discrepancy_pick(rows, dof)
        else:
            # L 曲线拐点式残差比准则：不依赖 σ 的估计，比 discrepancy 稳。
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
    # Plank 2025 Eq. 28：负 ϒ 节点用恒等式 −a/(1+jωτ) = −a + a·jωτ/(1+jωτ)
    # 展开后，常数项 −a 全部叠进了 R_inf，所以拟合出的 R_inf 是
    # **真欧姆电阻 + Σ|负极化|**。修正只在有负极化时才非平凡；
    # 非负路径 gamma≥0 → neg_pol=0 → R_inf_true ≡ R_inf（数值恒等）。
    neg_pol = float(np.sum(np.abs(gamma[gamma < 0.0])) * d)
    return {"f": f, "Z": Z, "tau": tau, "ln_tau": ln_tau, "gamma": gamma,
            "R_inf": r_inf, "lam": float(lam), "chi2": chi2, "dof": dof,
            "Z_fit": Z_fit, "resid": resid, "sigma": sigma,
            "signed": bool(signed),
            "neg_pol": neg_pol, "R_inf_true": r_inf - neg_pol,
            "R_total": float(np.sum(gamma) * d),
            # 端部堆积：γ 在网格边界仍很高，说明窗口外还有未分辨的弛豫，
            # 此时不能把它当成一个"峰"，也不能信它的 R
            "edge_low_frac": float(gamma[-1] / gm),
            "edge_high_frac": float(gamma[0] / gm)}


def _discrepancy_pick(rows, dof: float):
    """差异原理（discrepancy principle）：取 χ² 仍 ≤ dof 的**最强**正则化。

    只适用于 signed 反演：有符号模型的核族覆盖了感性/吸附弛豫
    （Plank 2025 的核心论点），χ² 能真正降到噪声水平（≈2n）。非负路径
    有模型缺项时 χ² 永远到不了 2n，用它会退到最弱正则化，所以那边的
    注释警告仍然成立，两分支不共用一个选法。
    """
    chi = np.array([r[2] for r in rows], float)
    ok = np.where(chi <= dof)[0]
    i = int(ok.max()) if len(ok) else int(np.argmin(chi))
    lm, xx, c2, _ = rows[i]
    return lm, xx, c2


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

    signed 反演时**同时**扫描 −γ 的峰（= 感性/吸附的负叶），返回项带
    `sign` 字段（+1 容性 / −1 感性），负叶的 `R_ohm`/`gamma_max` 为负。
    非负反演里 −γ 没有正峰可找，行为与引入 sign 之前等价。
    """
    from scipy.signal import find_peaks

    ln_tau = res["ln_tau"]
    d = ln_tau[1] - ln_tau[0]
    tau = res["tau"]

    out = []
    # (sign, 待扫描曲线)：负叶 = 对 −γ 找正峰。非 signed 时只扫 γ。
    scans = [(+1, np.asarray(res["gamma"], float))]
    if res.get("signed"):
        scans.append((-1, -np.asarray(res["gamma"], float)))
    for sign, g in scans:
        gm = g.max()
        if gm <= 0:
            continue
        idx, _ = find_peaks(g, prominence=rel * gm)
        for i in idx:
            # 每个峰各自向两侧走到最近的局部极小**或零穿越**，作为积分边界。
            # 不能用"全表极小点集合"——相邻峰的重叠区会被算两次或漏掉。
            # 零穿越截断只对 signed 有意义（非负路径 g 就是 γ≥0，永不触发）：
            # 有符号反演的振铃让负叶嵌在正叶之间，不截到 0 会把反号区算进
            # 面积，实测出现 sign=+1 但 R<0 的鬼峰。
            lo = i
            while lo > 0 and g[lo - 1] < g[lo] and g[lo] > 0:
                lo -= 1
            hi = i
            while hi < len(g) - 1 and g[hi + 1] < g[hi] and g[hi] > 0:
                hi += 1
            area = float(np.sum(0.5 * (g[lo:hi] + g[lo + 1:hi + 1])) * d) \
                if hi > lo else 0.0
            out.append({
                "f_peak_Hz": float(1.0 / (2 * np.pi * tau[i])),
                "tau_s": float(tau[i]),
                "gamma_max": float(sign * g[i]),
                "R_ohm": float(sign * area),
                "idx": int(i),
                "sign": int(sign),
                "width_dec": float((ln_tau[hi] - ln_tau[lo]) / np.log(10)),
            })
    # 同一网格点若正负两侧都冒出峰（毛刺），只保留幅值大的方向
    best: dict[int, dict] = {}
    for p in out:
        k = p["idx"]
        if k not in best or abs(p["R_ohm"]) > abs(best[k]["R_ohm"]):
            best[k] = p
    out = list(best.values())
    out.sort(key=lambda r: -r["f_peak_Hz"])
    return out



def _raw_support(f, Z, f_peak, sign, frac: float = 1.0,
                 min_prom: float = 0.3, min_pts: int = 5) -> bool:
    """峰方向必须在**原始数据** Im Z 里有对应的内部极值结构。

    - sign=+1（容性峰）：`-Im Z` 在 `f_peak×10^±frac` 窗口内有**内部**局部极大；
    - sign=-1（感性/吸附负叶）：`Im Z` 在同样窗口内有内部极值。

    这道闸门挡两类假阳性（立项 §6.1.1 的"完全单调上翘不算"同构）：
      · 反演**振铃**（Gibbs 叶）：二阶平滑对窄真峰两侧必然出反号侧瓣
        （实测无环纯容性谱出现 4 个负叶，位置全在真峰的 ±1dec 边带、
        原始 Im 数据在那里单调 → 无内部极值 → 拒）；
      · 网格**端部堆积**伪峰：argmax 落在窗口端点（单调），非内部 → 拒。
    真吸附环的峰位有 `Im` 内部极大（RL 恒等式的直接表现），必过。

    **必须是内部极大。** 端点极大 = 数据在窗口内单调，那是邻近更宽弧的
    **肩部**（或振铃边带），不是极值。实测教训（两版都错过）：
      · 窗口 ±0.5 dec：10 kHz 理想弧在窗口两侧的 -Im 仍有 0.27·R，
        prom 0.46 边缘值把**真峰**误杀（漏判）；
      · 端点"另一侧更低就放行"：把 3.8 Hz 肩部（-Im 单调升向 100 Hz 弧）
        放成了第三个 HN 候选（假阳性）——端点极大永远不是支撑，
        因为"峰外侧更低"恰是"朝峰爬升"的镜像。
    ±1.0 dec 给宽弧留边距，prom 0.3 挡浅肩。精细判定（acf、半圆完整性、
    可复现）是 M4 `detect_ad` 的职责，这里只做形状一票否决。
    """
    f = np.asarray(f, float)
    lo, hi = f_peak / 10 ** frac, f_peak * 10 ** frac
    m = (f >= lo) & (f <= hi)
    if int(m.sum()) < min_pts:
        return True          # 窗口太窄判不了 → 不拦（宁松勿误杀）
    y = np.asarray(Z.imag, float)[m] * (-float(sign))   # +1 看 -Im，-1 看 Im
    k = int(np.argmax(y))
    if not (0 < k < y.size - 1):                        # 端点极大 = 肩部/振铃
        return False
    peak = float(y[k])
    if peak <= 0:
        return False
    edge = float(max(y[0], y[-1]))
    return bool((peak - edge) / peak >= min_prom)


def prescreen(f, Z, sigma: float | None = None, rel: float = 0.03,
              r_min: float = 0.02, signed: bool = False):
    """预判断：只用手里的数据做 DRT 反演 + 找峰，**不涉及任何 HN/ZARC 参数**。

    返回 (res, pk, pk_all)：
      res    —— drt() 的原始输出（含 γ 曲线、R_inf、λ、端部堆积比例；
                signed=True 时额外有 R_inf_true / neg_pol / ad_peaks）
      pk_all —— peaks() 找到的全部峰（含负叶，带 sign 字段）
      pk     —— 过滤后的峰（喂给拟合做初值、给 K 一个候选范围）

    sigma : 相对噪声（|Z| 口径）。给 None 时用 0.5% 兜底——真实数据本来
            也不知道 σ，所以这是默认路径。
    rel   : 峰显著度阈值（相对最大 γ），0.03 = 至少 3% 峰高才算一个峰。
    signed: M5 方案 A（Plank 2025）。True 时 γ 允许为负，负叶（感性/吸附
            弛豫）**不进 pk**——它们是 `Z_ad` 候选，不是 HN 分量，
            混进来会把 K 撑大（立项书 §3 的"伪峰撑大 K"在反方向再现）。
            负叶改存 `res["ad_peaks"]`，供 M4 的 detect_ad / 吸附选择器消费。
            过滤规则与默认路径同构：峰位在测量频段内、面积占比 ≥ r_min。

    **必须再过一道过滤。** 正则化反演会在 τ 网格两端（窗口外）长出伪峰：
    非负约束把拟合不掉的尾巴推到了边界基函数上。实测案例 A 原始找到 4 个峰，
    其中 172 kHz（R=0.67 Ω）与 0.8 Hz（R=0.61 Ω）都是这样来的，
    而两个真峰（310 / 18.7 Hz，29.8 / 16.6 Ω）反而很准。不过滤会把 K 撑大一倍。

    过滤规则：峰位落在**测量频段内**，且面积 ≥ 总面积的 r_min。
    """
    res = drt(f, Z, sigma=sigma, signed=signed)
    pk_all = peaks(res, rel=rel)
    if not signed:
        tot = sum(p["R_ohm"] for p in pk_all) or 1.0
        pk = [p for p in pk_all
              if (np.min(f) <= p["f_peak_Hz"] <= np.max(f))
              and p["R_ohm"] >= r_min * tot]
        return res, pk, pk_all

    pos_all = [p for p in pk_all if p["sign"] > 0]
    neg_all = [p for p in pk_all if p["sign"] < 0]
    tot = sum(p["R_ohm"] for p in pos_all) or 1.0
    in_band = lambda p: (np.min(f) <= p["f_peak_Hz"] <= np.max(f))
    pk = [p for p in pos_all
          if in_band(p) and p["R_ohm"] >= r_min * tot
          and _raw_support(f, Z, p["f_peak_Hz"], +1)]
    ad = [p for p in neg_all
          if in_band(p) and abs(p["R_ohm"]) >= r_min * tot
          and _raw_support(f, Z, p["f_peak_Hz"], -1)]
    res["ad_peaks"] = ad
    ad_pol = float(sum(abs(p["R_ohm"]) for p in ad))
    res["ad_pol"] = ad_pol
    # drt() 的 R_inf_true 是 Plank Eq. 28 的**逐节点**口径；但二阶平滑对
    # 窄特征的振铃（Gibbs 叶）也计进 Σ|ϒ<0|，实测过冲把修正推成负电阻。
    # 有原始数据支持（§6.1.1 同款闸门）的负叶才是真吸附极化，
    # 下游用这个修正值。
    res["R_inf_true"] = float(res["R_inf"] - ad_pol)
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
