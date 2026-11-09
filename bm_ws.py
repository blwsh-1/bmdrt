# -*- coding: utf-8 -*-
"""Ws 判据的候选诊断量（**生产模块**：`bm_real.diffusion_gate` / `select_mechanism` 消费）。

定位
----
`bm_kk.diffusion_tail_candidate`（L1，事前形状）只用两个斜率，且幅度判据
`log|Z| ~ log f` 会被串联电阻 `R_inf` 压平——`example_caseE.csv` 与
`example_caseE_warburg.csv` 上都实测到 `slope_z ≈ 0`，明明真值含 `Ws(A_W=2)`
却判"无候选"。本模块做两件事：

  1. 换一个**对 `R_inf` 免疫**的幅度判据：`log(-Im Z) ~ log f`。
     `Ws` 的 `-Im Z = A/√(2ω) ∝ f^(-1/2)`，`R_inf` 只加在实部上，动不了它。
  2. 补一个**端点走向**判据。45° 段落在窗口哪里、之后往哪儿弯，决定的是
     「半无限 / 有限长阻挡 / 有限长透射」，而这一步现在只有拟合完 `tauD`
     之后才拿得到（`bm_diff.diffusion_resolution`）。

来源与验证
----------
本模块 2026-11-09 从 `evidence/ws_crit/bm_wscrit.py`（探索期原版）迁入生产，
**只改文件头定位，不改任何判据**。验证记录：
  · 7 项自检：`python bm_ws.py`；
  · 9 类 × 20 条 × 2 窗基准 360 条（`evidence/ws_crit/bench_rows.json`）：
    `ws_vis` 20/20、`wo_in` 20/20、`wo_below` 20/20、`wshort_in` 19/20、
    三类阴例共 120 次 0 假阳；
  · 18 条真实 AZ31 谱（`real_register.json`）：`inductive_tail` 18/18、
    `candidate` 0/18；
  · `bm_kk` 的 L1 幅度判据（P1）同源于本模块的 `slope_im`。

**本模块不做的事**：不判「物理上存在扩散」。`Ws` 与「峰位远在窗口下方的
`R||CPE(0.5)`」在有限频窗内是同一个渐近式，任何只依赖窗口内形状的判据都
分不开它们（`bm_wscrit` 的 `_selftest` 第 [3] 项把它钉成数值事实）。
所以输出里必须保留 `undecidable`：判据只能否证，不能确认。
"""
from __future__ import annotations

import numpy as np

#: 45° 判据的容差（Nyquist 斜率偏离 1 的幅度）
TOL_NY = 0.15
#: 幅度判据 `log(-Im) ~ log f` 的容差（理想 Ws = -0.5）
TOL_IM = 0.15
#: 认为"45° 段够长"的最小 decade。0.5 与 `bm_diff.warburg_multicomponent`
#: 的 `min_dec` 对齐：有限个 R||CPE(0.5) 拼出来的 45° 本来就短。
MIN_DEC = 0.5
#: 端点评判用的点数（最低频端）。5 点是因为 1% 噪声下 3 点的对数斜率抖动
#: 约 ±0.07，落在 `to_cap`（≤-0.80）与 `still_45`（-0.70..-0.30）的判定间隔里。
K_EDGE = 5
#: 判 `to_inductive` 的相对深度门槛：低频 `-Im` 的中位数必须小于
#: `-_IND_TOL × max|Im|`（整条谱）。**只按符号判会在噪声底附近误伤**：
#: `wshort_in` 的 `-Im` 真值只有 0.18–1.7 Ω 而噪声 sd≈0.5–1 Ω，符号随机翻负，
#: 纯符号判把 3/20 真 Wshort 打成"感抗"。实测两个窗口的 `im_ratio` 分布：
#:   `sorp_rl`（真 `R∥L`）  −0.18…−0.99（20/20 都能被抓到）
#:   `wshort_in`/`none_2zarc`（噪声翻号）**最负只到 −0.05**
#:   18 条真实 AZ31 谱            −0.42…−0.58
#: 0.05 与 0.18 之间有一条空档，0.10 落在中间——不是拍脑袋选的中点。
_IND_TOL = 0.10
#: `A_W` 后验相对不确定度（`sd/median`）的可辨识上限，L3 闸门用。
#: 标定见 `evidence/ws_crit/L3_CALIB.md`（32 条合成谱 × 4 个窗口）：
#:   `none`（无扩散硬加 Ws）  0.540–0.729   ← 必须拒
#:   真 Ws（snr≥5）           0.011–0.083
#:   真 Ws（snr≈2.4）         0.156–0.166   ← 幅值已偏 7–16%，不切
#:   `hn`（K+1 宽弧）         0.021–0.147   ← 与真 Ws 同分布（已知不可分）
#: 0.15 对应 `SNR_ws ≈ 2.7`（实测 `cv ≈ 0.40/snr`，32/32 命中），
#: 即"`A_W` 的相对不确定度不超过 15%"——能说清幅值才允许切。
CV_MAX = 0.15


# --------------------------------------------------------------------------
# 基础量
# --------------------------------------------------------------------------

def _prep(f, Z):
    f = np.asarray(f, float).ravel()
    Z = np.asarray(Z, complex).ravel()
    if f.size != Z.size:
        raise ValueError(f"f 与 Z 长度不一致：{f.size} vs {Z.size}")
    o = np.argsort(f)
    f, Z = f[o], Z[o]
    if np.any(f <= 0):
        raise ValueError("频率必须为正")
    return f, Z


def local_nyquist_slope(f, Z, win: int = 5):
    """逐点 `d(-Im)/d(Re)`，以 `log f` 为参数（不直接对 Re 求导）。

    直接对 Re 求导在 Re 不随 f 单调时（有电感、多弧）会炸；以 `log f` 为
    参数算两个方向的变化率再取比值，等价但稳定（同
    `bm_diff.warburg_band_slope`）。

    **导数是按 `win` 点滑窗做局部线性回归**，不是 `np.gradient` 的三点差分：
    实测 1% 噪声下三点差分的斜率逐点抖动到 ±0.5，`|s-1|≤0.15` 只剩
    10%~23% 的点命中，真实的 45° 段会被打碎（`ws_vis` 类只判出 5/20）。
    滑窗回归把抖动压到 ±0.1 以内，代价是频段两端各损失 `win//2` 个点。
    """
    f, Z = _prep(f, Z)
    n = f.size
    win = int(min(max(3, win), n if n % 2 else n - 1))
    half = win // 2
    lf = np.log10(f)
    s = np.full(n, np.nan)
    scale = max(np.abs(Z.real).max(), 1e-300)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        if b - a < 3:
            continue
        x = lf[a:b]
        re = np.polyfit(x, Z.real[a:b], 1)[0]
        im = np.polyfit(x, -Z.imag[a:b], 1)[0]
        if abs(re) > 1e-9 * scale:
            s[i] = im / re
    return s


def slope_im(f, Z, frac: float = 0.25, min_r2: float = 0.90):
    """`log(-Im Z) ~ log f` 在最低 `frac` 段的斜率（**对 `R_inf` 免疫**）。

    为什么必须换掉 `log|Z|`
    ----------------------
    `|Z| = |R_inf + Z_W|`，`R_inf` 一大就把低频段压平。`example_caseE.csv`
    （真值含 `Ws(A_W=2)`，`R_inf=10`）实测 `log|Z|` 斜率只有 **-0.03**，
    判成"不是扩散尾"；而 `-Im Z` 里根本没有 `R_inf`，同一条谱的斜率是
    **-0.49**。

    返回值
    ------
    slope / r2 / n / usable
    `usable=False` 的三种情形（都会给 `nan`）：
      · 该段的 `-Im` 不随 f 下降单调上升（有电感环、`-Im` 过零）；
      · `-Im` 在最低频点已接近 0（相对 `|Z|max` < 1e-4），取对数无意义；
      · 拟合 `r2 < min_r2` —— 说明不是幂律段，斜率不该被引用。
    """
    f, Z = _prep(f, Z)
    n = f.size
    k = max(4, int(round(frac * n)))
    im = -Z.imag[:k]
    fk = f[:k]
    zmax = max(np.abs(Z).max(), 1e-300)
    out = dict(slope=float("nan"), r2=float("nan"), n=int(k), usable=False, why="")

    if im.min() <= 1e-4 * zmax:
        out["why"] = ("-Im 在低频端 ≤0（最低点 "
                      f"{-Z.imag.min():+.2f} Ω，|Z|max={zmax:.1f} Ω）："
                      "低频进感抗区，不是扩散尾") if im.min() < 0 else \
                     "-Im 在低频端接近 0，取对数无意义"
        return out
    if im[0] < im[-1]:
        out["why"] = "-Im 不随 f 下降而上升（低频有电感/过零）"
        return out

    y = np.log10(np.maximum(im, 1e-300))
    x = np.log10(fk)
    s, b = np.polyfit(x, y, 1)
    pred = s * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    out.update(slope=float(s), r2=float(r2), usable=bool(r2 >= min_r2))
    if not out["usable"]:
        out["why"] = f"幂律拟合 r2={r2:.3f} < {min_r2}，该段不是单一幂律"
    return out


def band45(f, Z, tol: float = TOL_NY):
    """Nyquist 斜率落在 `1±tol` 的最长**连续**段。"""
    f, Z = _prep(f, Z)
    s = local_nyquist_slope(f, Z)
    ok = np.isfinite(s) & (np.abs(s - 1.0) <= tol)
    return _longest_run(f, ok, s)


def band_im(f, Z, lo: float = -0.68, hi: float = -0.32, win: int = 5):
    """**只用 `-Im`** 的 45° 段：`log(-Im) ~ log f` 的局部斜率落在 [-0.68,-0.32]。

    为什么还要一条 `-Im` 版的段长
    ----------------------------
    Nyquist 斜率 `d(-Im)/d(Re)` 在 `Re` 几乎不动的地方分母趋零，比值被放大：
    实测 `Wo`（`f_D` 在窗内）的 `Re` 在 45° 段附近基本是常数，`band45` 的段长
    **中位数是 0.00 decade（20 条里 0 条到得了 0.5）**。而 `-Im` 自己的斜率
    不受影响：同 20 条中位数 0.67 decade、13 条能到 0.5。两条取较大者，
    才不会把"Re 平"当成"没有 45°"。
    """
    f, Z = _prep(f, Z)
    n = f.size
    win = int(min(max(3, win), n if n % 2 else n - 1))
    half = win // 2
    lf = np.log10(f)
    im = -Z.imag
    zmax = max(np.abs(Z).max(), 1e-300)
    se = np.full(n, np.nan)
    if im.min() > 1e-6 * zmax:
        y = np.log10(np.maximum(im, 1e-300))
        for i in range(n):
            a, b = max(0, i - half), min(n, i + half + 1)
            if b - a >= 3:
                se[i] = np.polyfit(lf[a:b], y[a:b], 1)[0]
    ok = np.isfinite(se) & (se >= lo) & (se <= hi)
    return _longest_run(f, ok, se)


def _longest_run(f, ok, vals):
    lf = np.log10(f)
    best, lo_i, hi_i = 0.0, None, None
    i, n = 0, f.size
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        if lf[j] - lf[i] > best:
            best, lo_i, hi_i = float(lf[j] - lf[i]), i, j
        i = j + 1
    if lo_i is None:
        return dict(band_dec=0.0, f_lo=float("nan"), f_hi=float("nan"),
                    reach_edge=False, frac=float(np.mean(ok)) if n else 0.0,
                    med=float("nan"))
    return dict(band_dec=best, f_lo=float(f[lo_i]), f_hi=float(f[hi_i]),
                reach_edge=bool(lo_i == 0), frac=float(np.mean(ok)),
                med=float(np.nanmedian(vals[lo_i:hi_i + 1])))


def terminal(f, Z, k: int = K_EDGE):
    """最低频端往哪儿走：`still_45` / `to_cap` / `to_plateau` / `to_inductive` / `noisy`。

    这是 Ws / Wo / Wshort 的分水岭，也是 `DIFFUSION_PLAN.md` §3.2 那三个
    方向在**数据侧**的对应物（那边是拟合完 `tauD` 才能算的）。

    **判的是 `-Im` 自己的对数斜率，不是 `d(-Im)/d(Re)`，也不是绝对相位。**
    两个坑都踩过：

      · 绝对相位会被 `R_inf` 拉向 0：实测 `R_inf=500` 的纯 `Ws` 在 0.1 Hz
        只有 -15°，被误判成"平台"。
      · `d(-Im)/d(Re)` 在电容尾上分母趋于 0（`Re` 几乎不动），比值的噪声
        远大于信号：实测 `Wo` 有 13/20 被误判成平台。

    `-Im` 里没有 `R_inf`，它的对数斜率就是**纯指数**

      se ≈ -0.5 → `still_45`  （半无限 Ws **或** 峰位在窗外的 R||CPE(0.5)，同形）
      se ≤ -0.8 → `to_cap`    （低频转电容 ≈ f^-1：有限长**阻挡** Wo）
      se ≥ -0.15→ `to_plateau`（-Im 在低频反而变小、但仍是正的：回到实轴，
                               Wshort 或弧的下降段）
      最低 5 点 `-Im` 的中位数为负 → `to_inductive`（**低频进感抗区**）
      其余      → `noisy`

    `to_inductive` 是 2026-11-09 在 18 条真实 AZ31 谱上暴露出来的缺档
    -----------------------------------------------------------------
    那 18 条的低频端 `Zim > 0`（`-Im < 0`），是**吸附/感抗环**（`ADSORPTION_PROJECT`
    的 `R∥L` 类），不是"回到实轴"。原实现把 `-Im<0` 和 `-Im→0+` 都归到
    `to_plateau`，于是给出 `family=Wshort_like`（有限长**透射扩散**）——
    一个完全错误的断言：那条谱低频根本没有 `-Im>0` 的扩散尾。
    Warburg 尾的必要条件是 `-Im > 0 且 ∝ f^(-1/2)`，负 `-Im` 直接否证，
    不需要再看段长。所以必须单列，且 `family` 不能是任何扩散家族。

    `still_45` **不是**"确认 Ws"，见模块 docstring。
    """
    f, Z = _prep(f, Z)
    k = max(3, min(int(k), f.size))
    im = -Z.imag[:k]
    x = np.log10(f[:k])
    zmax = max(np.abs(Z).max(), 1e-300)
    out = dict(label="noisy", se=float("nan"), k=int(k),
               im_rises=bool(im[0] > im[-1]),
               ph_med=float(np.median(np.degrees(np.angle(Z[:k])))))
    # 低频负 `-Im` 的**相对深度**：与整条谱里最大的容性 `-Im` 比。
    # 只按符号判会把"信号在噪声底附近随机翻号"也算成感抗（实测 `wshort_in`
    # 有 3/20 这样被误杀），必须要求负得够深。见 `_IND_TOL`。
    im_scale = float(np.abs(Z.imag).max())
    out["im_ratio"] = float(np.median(im) / im_scale) if im_scale > 0 else 0.0

    # 感抗区必须先判：负的 `-Im` 不能进对数，也不能被当成"贴地"
    if out["im_ratio"] < -_IND_TOL:
        out["label"] = "to_inductive"
        return out
    if im.min() <= 1e-4 * zmax:
        out["label"] = "to_plateau"      # -Im 已经贴地，就是回到实轴
        return out
    y = np.log10(np.maximum(im, 1e-300))
    se = float(np.polyfit(x, y, 1)[0])
    out["se"] = se
    if se <= -0.80:
        out["label"] = "to_cap"
    elif -0.70 <= se <= -0.30:
        out["label"] = "still_45"
    elif se >= -0.15:
        out["label"] = "to_plateau"
    else:
        out["label"] = "noisy"
    return out


# --------------------------------------------------------------------------
# 合成判据
# --------------------------------------------------------------------------

#: `terminal` 标签 → 允许的扩散元件家族。`still_45` 落进 undecidable 分支。
_FAMILY = {
    "to_cap": "Wo_like",          # 有限长阻挡
    "to_plateau": "Wshort_like",  # 有限长透射
    "still_45": "Ws_or_wide_CPE",  # 半无限 or 窗外 R||CPE(0.5)：同形，不可分
    "to_inductive": "inductive_tail",  # 低频感抗（吸附环）：**不是** 任何扩散家族
    "noisy": "unknown",
}


def ws_diagnostics(f, Z, frac: float = 0.25, tol_ny: float = TOL_NY,
                   tol_im: float = TOL_IM, min_dec: float = MIN_DEC) -> dict:
    """一次性给出全部候选诊断量与**四分类**结论。

    输出
    ----
    slope_im / r2_im / im_usable     幅度判据（R_inf 免疫）
    amp45                             |slope_im + 0.5| ≤ tol_im
    amp90                             |slope_im + 1.0| ≤ tol_im（容性尾）
    band_dec / reach_edge / frac_ny   几何判据
    term / term_ph                    端点走向
    family                            五分类：Ws_or_wide_CPE / Wo_like /
                                      Wshort_like / inductive_tail / unknown
    verdict                           三态：`candidate` / `reject` / `undecidable`
    why                               人类可读理由（逐条）

    _verdict 的规则按家族分别设门槛**（见下），不是单一"段长 ≥ min_dec"：
      · `to_cap`（有限长阻挡）→ 45° 段 ≥0.10 dec 或幅度判据成立即 candidate
        ——`Wo` 的 45° 段天生短，它到拐点就结束；用 0.5 会把正例全灭
      · `still_45`（半无限 / 窗外宽 CPE）→ 段 ≥min_dec **或**触及低频边缘
      · `to_plateau` → 段 ≥min_dec 时给 `undecidable`（属 Wshort 家族，
        `bm_core` 未接入），否则 `reject`
    """
    f, Z = _prep(f, Z)
    si = slope_im(f, Z, frac=frac)
    bd = band45(f, Z, tol=tol_ny)
    bi = band_im(f, Z)
    tm = terminal(f, Z)

    # 两条 45° 段长取较大者：`Re` 平的段上 Nyquist 斜率会失真（见 band_im）
    if bi["band_dec"] > bd["band_dec"]:
        bd = dict(bi, band_dec_ny=bd["band_dec"], band_dec_im=bi["band_dec"],
                  med_ny=bd.get("med"))
    else:
        bd = dict(bd, band_dec_ny=bd["band_dec"], band_dec_im=bi["band_dec"],
                  med_ny=bd.get("med"), med_im=bi.get("med"))

    amp45 = bool(si["usable"] and abs(si["slope"] + 0.5) <= tol_im)
    amp90 = bool(si["usable"] and abs(si["slope"] + 1.0) <= tol_im)
    family = _FAMILY.get(tm["label"], "unknown")
    why = []

    if si["usable"]:
        why.append(f"log(-Im) 斜率 {si['slope']:+.2f}（r2={si['r2']:.3f}）"
                   f"→ {'45° 幅值指数成立' if amp45 else ('容性尾' if amp90 else '既非 45° 也非容性')}")
    else:
        why.append(f"幅度判据不可用：{si['why']}")

    why.append(f"45° 段最长 {bd['band_dec']:.2f} decade"
               f"（Nyquist 口径 {bd['band_dec_ny']:.2f} / -Im 口径 "
               f"{bd['band_dec_im']:.2f}，取大者）"
               f"{'（触及最低频点）' if bd['reach_edge'] else ''}，"
               f"Nyquist 命中占比 {bd['frac']:.0%}")
    se_txt = f"{tm['se']:+.2f}" if np.isfinite(tm["se"]) else "不可用"
    why.append(f"低频端 log(-Im)~log f 斜率 {se_txt}"
               f"（-Im {'随 f 下降而上升' if tm['im_rises'] else '随 f 下降而回落'}）"
               f"→ {tm['label']}")

    # family 已由 _FAMILY 映射好，这里不再重复赋值（重复会让新增标签漏掉）

    # ---- 判定：**按家族分别设门槛**，不能用一个"段长 ≥ min_dec"通吃 ----
    # 实测（`bench.py`，9 类 × 20 条 × 2 窗口）：如果把"45° 段 ≥0.5 decade"
    # 当唯一门槛，有限长阻挡（`f_D` 在窗内）的灵敏度只剩 0.00–0.25——因为
    # Wo 的 45° 段**天生就短**，它到拐点就结束了。反过来放宽门槛，纯弛豫谱
    # 又被放进来了。所以门槛必须跟着端点走向走。
    band = bd["band_dec"]
    if tm["label"] == "to_inductive":
        verdict = "reject"
        why.append("→ reject：低频 `-Im < 0`（感抗）。三类扩散尾（Ws/Wo/Wshort）"
                   "都要求 `-Im > 0` 且低频按 `f^(-1/2)`~`f^(-1)` 增长，"
                   "符号不符即否证，不需要再看段长。"
                   "该低频特征应走**吸附支路**（`ADSORPTION_PROJECT` 的 `R∥L`）。"
                   "限定：本判据只否证**窗内可见**的电容性扩散尾；"
                   "若吸附环与扩散尾共存，感抗环会把它掩盖，"
                   "此档**不**等于断言「不存在扩散」")
    elif tm["label"] == "noisy":
        verdict = "undecidable"
        why.append("→ undecidable：低频端 log(-Im) 斜率不落在任何一支，"
                   "端点走向判不出来，此时任何段长门槛都没有意义")
    elif tm["label"] == "to_cap":
        if band >= 0.10 or amp45:
            verdict = "candidate"
            why.append("→ candidate(Wo_like)：45° 之后转电容，是有限长阻挡的指纹；"
                       "只有 f_D 落在窗口内才还原得出 R_W 与 τ_D（§3.2）")
        else:
            # 实测 `example_caseG_wo_only.csv`（纯 Wo，σ=0.803 Ω）就落在这里：
            # 它的 45° 段在**高频**，而那里 `-Im ≈ 0.5 Ω` 已经在噪声地板下面，
            # 数据上根本看不见。此时"没看见"不等于"没有"——纯阻塞电极（理想
            # 电容）与"噪声底下的 Wo"在这个窗口内完全同形。所以不给 reject。
            verdict = "undecidable"
            why.append("→ undecidable：低频确实转电容，但窗口里没有可见的 45° 段。"
                       "两种可能分不开：(a) 它就是一条未闭合的容性弧；"
                       "(b) 45° 段落在噪声地板以下（纯 Wo + 大 σ 时实测如此）。"
                       "要定需要更低频或更小噪声的数据")
    elif tm["label"] == "still_45":
        if band >= min_dec or bd["reach_edge"] or amp45:
            verdict = "candidate"
            why.append("→ candidate，但 family=Ws_or_wide_CPE：**不能**与"
                       "「峰位远在窗口下方的 R||CPE(n=0.5)」区分（同形，"
                       "见自检 [3]），也不能与 R||CPE(n≠0.5) 的下降段区分")
        else:
            verdict = "reject"
            why.append(f"→ reject：低频端斜率像 45°，但最长连续段只有 "
                       f"{band:.2f} decade、未触及边缘、幅度判据也不成立")
    else:  # to_plateau
        if band >= min_dec:
            verdict = "undecidable"
            why.append("→ undecidable：45° 之后转平台，属**有限长透射**（Wshort）"
                       "家族；`bm_core` 未接入 Wshort，且「45°+平台」与宽化 "
                       "R||CPE 在窗内同形，本判据不给 Ws 家族")
        else:
            verdict = "reject"
            why.append("→ reject：低频回到实轴且没有成段的 45°，"
                       "是普通弛豫弧（或其下降段），不是扩散尾")

    return dict(slope_im=si["slope"], r2_im=si["r2"], im_usable=si["usable"],
                amp45=amp45, amp90=amp90,
                band_dec=bd["band_dec"], band_dec_ny=bd["band_dec_ny"],
                band_dec_im=bd["band_dec_im"],
                reach_edge=bd["reach_edge"], frac_ny=bd["frac"],
                term=tm["label"], term_se=tm["se"],
                term_im_ratio=tm.get("im_ratio", float("nan")),
                family=family, verdict=verdict, why=why)


# --------------------------------------------------------------------------
# 模型侧：扩散项占多少极化电阻（Boškoski 的 relative polarisation contribution）
# --------------------------------------------------------------------------

def rpc_share(samples: dict, key: str = "auto") -> dict:
    """扩散项在**极化电阻**里占的份额，及其后验分布。

    为什么另立一个量，而不是用 `bm_real` 的 `p_diff_small`
    --------------------------------------------------------
    `p_diff_small = P(Rw < 1e-3·|Z|max)`（`DIFFUSION_PLAN.md` §6 问题 C1）的
    参照物是**整条谱的最大模**，回答的是"扩散项相对整条谱是否可忽略"。
    但模型选择真正要问的是"**要不要为这点极化电阻多挂一个元件**"，参照物
    应该是极化电阻 `R_p`，不是 `|Z|max`——一条 `R_inf` 很大的谱，扩散项占
    `R_p` 的一半也可能远小于 `1e-3·|Z|max`。

    Boškoski 2024（*J. Power Sources* 622, 235359）§2.4 把"极重要性"定义成
    DRT 峰面积（相对极化贡献），并给出 RQ 极的闭式积分。对 `Z = R∞ + Σ R_k
    /(1+jωτ_k)`，各峰面积就是各自的 `R_k`，归一化即份额。这里直接用采样参数
    算：

        share_diff = R_w / (R_w + Σ w_k)

    参数
    ----
    samples : 后验样本 dict（`npz` 加载或 `bm_core` 的 `s`）
    key     : `"Rw"`（Ws）/ `"Rwo"`（Wo）/ `"auto"`（按存在性自动挑）

    返回
    ----
    key / share_med / share_lo / share_hi（90% 区间）/ p_below_5pct /
    p_below_10pct / n / per_comp（各 HN 分量的份额中位数）
    """
    if key == "auto":
        key = "Rw" if "Rw" in samples else ("Rwo" if "Rwo" in samples else None)
    out = dict(key=key, share_med=float("nan"), share_lo=float("nan"),
               share_hi=float("nan"), p_below_5pct=float("nan"),
               p_below_10pct=float("nan"), n=0, per_comp=[],
               note="")
    if key is None or key not in samples:
        out["note"] = "样本里没有扩散项幅值（Rw/Rwo），本条不适用"
        return out

    rd = np.asarray(samples[key], float).ravel()
    w = np.asarray(samples["w"], float)
    if w.ndim == 1:
        w = w[:, None]
    n = min(rd.size, w.shape[0])
    rd, w = rd[:n], w[:n]
    good = np.isfinite(rd) & np.isfinite(w).all(axis=1) & (rd >= 0) & (w >= 0).all(axis=1)
    rd, w = rd[good], w[good]
    if rd.size == 0:
        out["note"] = "有效样本为 0"
        return out

    tot = rd + w.sum(axis=1)
    share = np.where(tot > 0, rd / np.maximum(tot, 1e-300), np.nan)
    share = share[np.isfinite(share)]
    out.update(n=int(share.size),
               share_med=float(np.median(share)),
               share_lo=float(np.percentile(share, 5)),
               share_hi=float(np.percentile(share, 95)),
               p_below_5pct=float(np.mean(share < 0.05)),
               p_below_10pct=float(np.mean(share < 0.10)),
               per_comp=[float(x) for x in np.median(w / np.maximum(tot, 1e-300)[:, None], axis=0)])
    return out




def ws_share_low(samples: dict, f_min: float | None = None) -> dict:
    """`Ws` 在**最低频点**吃掉的实部占极化电阻多少（`Ws` 版的 RPC）。

    为什么 `Ws` 不能直接用 `rpc_share`
    ----------------------------------
    `rpc_share` 的分母是极化电阻，而 `Ws` **没有有限极化电阻**（`Z_W ∝
    ω^(-1/2)`，`ω→0` 时实部发散）。样本里的 `Rw` 只是窗口归一化幅值
    `|Z_Ws(f_ref)|`，拿它跟 `R_p` 比没有物理意义：实测 `example_caseE`
    （真值 `A_W=2`、`R_p=30`）会算出 `share=0.005`，看起来像"扩散项可忽略"，
    其实是归一化点的选择问题——`f_ref=√(f_min·f_max)=31.6 Hz` 距离
    `f_min=0.01 Hz` 有 3.5 个量级。

    所以 `Ws` 用**窗口内可测的那部分**：`Re Z_Ws(f_min) = A_W/√(4π f_min)`。
    代价是这个量**依赖窗口下限**：同一条 `A_W=2` 的谱，窗口 0.01–1e5 Hz
    给 16.1%，窗口 0.1–1e3 Hz 只给 5.7%。所以它必须连 `f_min` 一起报，
    不能脱离窗口引用。

    `share` **不做闸门**，闸门用 `cv_aw`
    ------------------------------------
    2026-11-09 标定（`evidence/ws_crit/L3_CALIB.md`，32 条合成谱 × 4 个窗口）
    推翻了"份额 ≥5% 才值得加 Ws"这个想法：

      · 真 `Ws`、`A_W=0.5`、窗 0.01 Hz 的份额只有 **0.048** —— 5% 阈值直接误杀；
      · 份额随 `f_min` 系统漂移（`A_W=2`：0.160 → 0.057），不是窗口不变量；
      · 它衡量的是"这点扩散相对背景有多大"，**不是**"这点扩散能不能被定住"。

    能跨窗口比较的是**后验相对不确定度** `cv_aw = sd(A_W)/median(A_W)`
    （`CV_MAX`）。实测它与 `SNR_ws = Re Z_Ws(f_min)/σ_im` 严格反比
    （`cv ≈ 0.40/snr`，32/32），但 `cv` 不需要假设噪声模型、也不依赖
    `σ_rel·|Z|` 口径，所以闸门取 `cv`。

    注意 `cv` **不能判"元件选错"**：真 `Wo`（转折在窗内）用 `Ws` 拟合，
    `cv` 低到 0.010–0.038（参数被"很确定地"定在错误的值上）。错元件由
    LOO 的 `Δ` 与 `diffusion_resolution.resolvable` 管，不归这里。

    参数
    ----
    f_min : 缺省取样本里的 `f.min()`

    返回
    ----
    A_W / A_W_sd / cv_aw / identifiable
    f_min / re_ws / rp_hn / share_med / share_lo / share_hi
    snr_med / snr_lo / sig_im / n / note
    """
    out = dict(A_W=float("nan"), A_W_sd=float("nan"), cv_aw=float("nan"),
               identifiable=None,
               f_min=float("nan"), re_ws=float("nan"),
               rp_hn=float("nan"), share_med=float("nan"),
               share_lo=float("nan"), share_hi=float("nan"),
               snr_med=float("nan"), snr_lo=float("nan"),
               sig_im=float("nan"), n=0, note="")
    if "Aw" not in samples:
        out["note"] = "样本里没有 Aw（未拟合 Ws）"
        return out
    A = np.asarray(samples["Aw"], float).ravel()
    w = np.asarray(samples["w"], float)
    if w.ndim == 1:
        w = w[:, None]
    if f_min is None:
        f = np.asarray(samples.get("f", []), float)
        if f.size == 0:
            out["note"] = "样本里没有 f，需要显式给 f_min"
            return out
        f_min = float(f.min())
    n = min(A.size, w.shape[0])
    A, w = A[:n], w[:n]
    good = np.isfinite(A) & (A > 0) & np.isfinite(w).all(axis=1)
    A, w = A[good], w[good]
    if A.size == 0:
        out["note"] = "有效样本为 0"
        return out
    re_ws = A / np.sqrt(4 * np.pi * f_min)
    tot = re_ws + w.sum(axis=1)
    share = np.where(tot > 0, re_ws / np.maximum(tot, 1e-300), np.nan)

    med = float(np.median(A))
    sd = float(np.std(A))                     # ddof=0：后验分布本身，不是抽样估计
    cv = sd / med if med > 0 else float("nan")

    # SNR 只在噪声尺度可得时算（生产样本一定有；手工构造的可能没有）
    sig = np.asarray(samples.get("s_im", []), float).ravel()
    sig_m = float(np.median(sig)) if sig.size else float("nan")
    if np.isfinite(sig_m) and sig_m > 0:
        snr = re_ws / sig_m
        out["snr_med"] = float(np.nanmedian(snr))
        out["snr_lo"] = float(np.nanpercentile(snr, 5))
    out["sig_im"] = sig_m

    out.update(A_W=med, A_W_sd=sd, cv_aw=cv,
               identifiable=(bool(cv <= CV_MAX) if np.isfinite(cv) else None),
               f_min=float(f_min),
               re_ws=float(np.median(re_ws)), rp_hn=float(np.median(w.sum(axis=1))),
               n=int(share.size),
               share_med=float(np.nanmedian(share)),
               share_lo=float(np.nanpercentile(share, 5)),
               share_hi=float(np.nanpercentile(share, 95)))
    return out




def _zarc(f, R, f0, phi):
    x = (1j * 2 * np.pi * f / (2 * np.pi * f0)) ** phi
    return R / (1.0 + x)


def _rq(f, R, tau, n):
    Q = tau ** n / R
    return R / (1.0 + R * Q * (1j * 2 * np.pi * f) ** n)


def _zw(f, A):
    return A / np.sqrt(1j * 2 * np.pi * f)


def _zcoth(f, Rw, tD):
    z = np.sqrt(1j * 2 * np.pi * f * tD)
    return Rw * (1.0 / np.tanh(z)) / z


def _ztanh(f, Rw, tD):
    z = np.sqrt(1j * 2 * np.pi * f * tD)
    return Rw * np.tanh(z) / z


def _selftest():
    f = np.logspace(-1, 3, 31)
    checks = []

    # [1] Ws 的 -Im 斜率必须 ≈ -0.5，且 R_inf 完全不进 -Im
    a = ws_diagnostics(f, 5 + _zw(f, 20.0))
    b = ws_diagnostics(f, 500 + _zw(f, 200.0))
    checks.append(("[1] Ws 的 R_inf 免疫：slope_im 不随 R_inf 变",
                   abs(a["slope_im"] - b["slope_im"]) < 1e-6
                   and abs(a["slope_im"] + 0.5) < 0.05,
                   f"{a['slope_im']:+.3f} vs {b['slope_im']:+.3f}"))

    # [2] 老幅度判据（log|Z|）在同一条谱上会被压平 —— 对照，证明换指标有必要
    lf = np.log10(f[:8])
    sz = np.polyfit(lf, np.log10(np.abs((500 + _zw(f, 200.0))[:8])), 1)[0]
    checks.append(("[2] 同一条谱 log|Z| 斜率被 R_inf 压平",
                   abs(sz) < 0.15 and abs(b["slope_im"] + 0.5) < 0.05,
                   f"log|Z| {sz:+.3f}  vs  log(-Im) {b['slope_im']:+.3f}"))

    # [3] **不可分**：Ws 与峰位远在窗外的 R||CPE(0.5) 必须给出同一个 family
    c = ws_diagnostics(f, 10 + _rq(f, 2000.0, 1.0 / (2 * np.pi * 1e-6), 0.5))
    checks.append(("[3] Ws vs 窗外 R||CPE(.5)：本模块不给区分（同形）",
                   c["family"] == "Ws_or_wide_CPE" and b["family"] == "Ws_or_wide_CPE",
                   f"{b['family']} / {c['family']}"))

    # [4] 端点走向三分类：Ws / Wo / Wshort
    d = ws_diagnostics(f, 10 + _zcoth(f, 40.0, 1e-2))       # f_D=16 Hz 在窗内
    e = ws_diagnostics(f, 10 + _ztanh(f, 40.0, 1e-2))
    checks.append(("[4] 端点走向：Ws→still_45、Wo→to_cap、Wshort→to_plateau",
                   b["term"] == "still_45" and d["term"] == "to_cap"
                   and e["term"] == "to_plateau",
                   f"{b['term']} / {d['term']} / {e['term']}"))

    # [5] 纯弛豫谱必须 reject
    g = ws_diagnostics(f, 10 + _zarc(f, 30, 300, 0.85) + _zarc(f, 15, 3, 0.9))
    checks.append(("[5] 两个真弛豫（无扩散）→ reject",
                   g["verdict"] == "reject", g["verdict"]))

    # [6] 容性尾必须单列，不能算 45°
    h = ws_diagnostics(f, 10 + _zcoth(f, 40.0, 1e-6))       # f_D 远在窗口上方
    checks.append(("[6] 低频转 90° 的容性尾不给 amp45",
                   (not h["amp45"]) and h["amp90"], f"{h['slope_im']:+.2f}"))

    # [7] 低频感抗（R||L 吸附环）必须与"回到实轴"分开：前者不是任何扩散家族
    def _rl(f, R, L):
        w = 2 * np.pi * f
        zl = 1j * w * L
        return R * zl / (R + zl)
    i = ws_diagnostics(f, 10 + _zarc(f, 30, 300, 0.85) + _rl(f, 60, 60 / (2 * np.pi * 0.4)))
    checks.append(("[7] R||L 低频感抗 → to_inductive / 非扩散家族",
                   i["term"] == "to_inductive" and i["family"] == "inductive_tail"
                   and i["verdict"] == "reject",
                   f"{i['term']} / {i['family']} / {i['verdict']}"))

    ok = sum(1 for c in checks if c[1])
    print(f"bm_ws 自检：{ok}/{len(checks)} 通过")
    for name, passed, note in checks:
        print(f"  [{'OK' if passed else 'FAIL'}] {name}   ({note})")
    return ok == len(checks)


if __name__ == "__main__":
    import sys
    good = _selftest()
    sys.exit(0 if good else 1)
