# -*- coding: utf-8 -*-
"""拟合前的 Lin-KK（线性时不变）门禁。

为什么需要这道门禁
------------------
PSIS-LOO 只能在**似然假设成立**的前提下比较模型，它不能检验似然假设本身。
谱若因漂移 / 非因果 / 非线性而不自洽，增加 K 就是增加贴噪声的自由度，
`elpd` 会系统性偏向更大的 K —— 拟合得越"漂亮"，参数离物理越远。
所以 K-K 是**事前门禁**，不是又一个事后诊断指标。

为什么不是直接 import AutoECM
----------------------------
`D:\\eis训练数据\\AutoECM\\validate_fit.py:38-77` 已有可用的 `lin_kk()`，
但它依赖同目录的 `ext_fit` / `fit_store`，且不在本仓库内。
这里把两个必需函数自包含移植，只依赖 `numpy` + `impedance`；
`impedance` 缺失时返回 `status="unavailable"`，**只警告、不中断**。

一个必须澄清的事实：Lin-KK 本来就带自由串联电感
------------------------------------------------
`impedance.validation.fit_linKK` 的设计矩阵里显式有一列
`a_im[:, -1] = w / |Z|`，注释写的是 "series inductance to capture
inevitable contributions from the measurement system"。返回值
`elements = [R0, R1..RM, L]`（add_cap 时再插一个 C）。
**因此扣掉 `jωL` 再跑 Lin-KK 是重复计入**，实测会让高频残差变差：
真实 2E 谱 9.60% → 10.78%，截断+扣 L 后 9.60% → 11.68%。
这里不再扣 L；`estimate_L()` 只作为诊断量保留，供后续
「DRT 预判扣电感」（计划 P2-4）使用——Tikhonov DRT 才是真的表示不了 `jωL`。

低频扩散尾同理：RC 链时间常数只覆盖测量窗口
`get_tc_distribution()` 给的是 `[1/(2πf_max), 1/(2πf_min)]`，
表示不了窗口外继续发散的 Warburg 尾。`fit_linKK(add_cap=True)`
加一列串联电容 `-1/(ωC)`，正是官方给"无低频截距"数据的解法。
实测真实 2E 谱：低频段残差 11.44% → 4.72%，且**一个数据点都不丢**；
而直接截掉最低 25% 频点反而让中段 1.92% → 4.33%、高频 1.92%→14.83%。
所以本门禁只用 `add_cap` 复验，`f_cut` 仅作诊断记录、不真正截断。

M 阶数怎么选：μ 早停会造成假阳性
--------------------------------
`impedance.validation` 官方的 `linKK` 用 `while mu > c: M += 1` 选 RC 阶数。
μ 不是残差，而是 `calc_mu(R_k) = 1 − Σ|R_k<0| / Σ|R_k≥0|`，衡量 RC 系数里
负值的比例（过拟合信号）。这个停止条件在**含大电感、|Z| 很小**的 2E 谱上
会过早触发：实测 `2EMC2-1_0200mA.csv`（56 点、1–1e5 Hz、|Z|≈0.01 Ω）
在 M=4 就因出现负 R_k 得到 μ=0.565 而停，RC 链严重欠拟合 → 中段相对残差
14.72%、ac1 0.97/0.93，被下面的「系统性」判据误判成谱不自洽、拒绝定 K。
同一份数据继续加阶：M=6 → 2.51%、M=8 → 1.02%、M=9 → 0.83%。
所以 M 改为**残差驱动**：

    score(M) = rms_all(M) · (1 + 0.02·M)      # 每多一阶要求残差至少改善约 2%

取 score 最小的 M，并加三道数值护栏：
  · M ≤ min(max_M, (n−2)//2)（参数个数不超过点数的一半）；
  · 残差突然放大 3 倍（或非有限）→ 停止，不采用该 M；
  · 连续 5 阶相对改善 < 2% → 停止扫描。
μ 仍返回，但**只作过拟合参考**（μ<0.1 时在归因里提示），不再决定 M，
也不再参与 `kk_check` 的分级。实测 118 条 2E 谱：旧 μ 早停 → fail 68 条；
新规则 → fail 4 条（其余 warn 为端部归因，不拦拟合）。

分级判定的关键：中段才是主证据
------------------------------
    1. 原谱做 Lin-KK，按 **log f 三分**统计复相对残差 RMS；
    2. 低频若有扩散尾候选（`|Z|` 斜率 **且** Nyquist 45° 双判据成立）/ 低频残差超阈
       → 用 `add_cap=True` 在同一频率网格复验；
    3. **只允许 add_cap 放宽低频段**：`rms_clean["low"]` 取 raw/cap 更优者；
       `mid` / `high` 一律用 raw。这样 add_cap 绝不可能把一段真实的
       中段不自洽"吸收"掉，门禁不会被自己引入的自由度骗过；
    4. **中段残差超阈还要看结构才判 fail**（沿用 AutoECM/validate_fit.py 的
       "滞后-1 自相关 + 符号变化率"思路）。加噪谱的相对残差可以很大但杂乱；
       只有实部**和**虚部中段残差都呈系统性（ac1>0.5）才是真的不自洽：
           `fail` = (mid > tol 且 系统性) 或 (mid > 2*tol)
       其余中段超阈降为 `warn`，注明"噪声 / Lin-KK 表达力不足，不能判定不自洽"；
       端部超阈但能归因 / μ 未收敛 → warn。
       有电感、有扩散尾这类纯信息性备注**不**改变 status。

    实测标定（本仓库复现：`examples/example_caseE.csv` + 实部鼓包
    exp(−½((log f−log f0)/0.25)²)，幅度按 R=Re 跨度计）：
    谱                        中段残差   判定
    干净 caseE                0.7%      ok
    实部鼓包 0.10R            2.6%      ok
    实部鼓包 0.20R            4.9%      ok（低于阈值）
    实部鼓包 0.40R            8.6%      fail（系统性 + 残差过大）
    低频漂移 0.10R            1.5%      ok（本构造较弱）
    加噪 1%|Z|                 1.0%      ok
    注意：鼓包幅度/宽度与旧版标定不同，数值不能逐行对比；
    关键次序不变——「0.40R 仍 fail，干净/小扰动/噪声不 fail」。

用法
----
    python bm_kk.py examples/example_caseE.csv
    python bm_kk.py data.csv --tol 0.05 --c 0.85

    from bm_kk import kk_check
    gate = kk_check(f, Z, include_L=None)
    if gate["status"] == "fail": ...
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

EPS = 1e-30


# --------------------------------------------------------------------------
# AutoECM/validate_fit.py 的自包含移植
# --------------------------------------------------------------------------

def _eval_linkk(elements, ts, f):
    """重建 lin-KK 的 RC 链电路并求值。

    不能直接用 `impedance.validation.eval_linKK`：它把参数用 repr 拼进字符串，
    而 numpy 2.x 的 `repr(np.float64(1.0)) == 'np.float64(1.0)'`，
    拼出来的代码里出现 np 但 eval 的命名空间没有 np → NameError。
    这里全部转成 float 再拼，其余逻辑与官方一致。

    elements = [R0, R1..RM, L]（长度 M+2）；
    add_cap 时 = [R0, R1..RM, 1/C, L]（长度 M+3）。
    """
    from impedance.models.circuits.elements import circuit_elements

    fl = np.asarray(f, dtype=float).tolist()
    parts = [f"R([{float(elements[0])}], {fl})"]
    for rk, tk in zip(elements[1:], ts):
        parts.append(f"K([{float(rk)}, {float(tk)}], {fl})")
    parts.append(f"L([{float(elements[-1])}], {fl})")
    if elements.size == ts.size + 3:          # add_cap：末尾多一个串联电容
        parts.append(f"C([{float(1.0 / elements[-2])}], {fl})")
    return eval("s([" + ",".join(parts) + "])", circuit_elements)


def lin_kk(f, Z, c: float = 0.85, max_M: int = 50,
           fit_type: str = "complex", add_cap: bool = False,
           elbow: float = 0.02, patience: int = 5,
           blowup: float = 3.0) -> dict:
    """线性 Kramers-Kronig 检验，返回 M / mu / Z_fit / 实虚部相对残差。

    绕开 impedance.py 1.7.1 在 numpy 2.x 下的两个兼容问题：
    ① `M % 10 == 0` 分支里留了一行调试 print，eval_linKK 里 np 未定义会崩；
    ② eval_linKK 本身用 repr 拼字符串，同样崩。
    `fit_linKK` / `get_tc_distribution` 是纯 numpy 运算，可以直接用。

    **M 不再用官方的 `while mu > c` 早停**（见模块 docstring）：改为扫描
    M=1..M_hi，取 `rms_all·(1+elbow·M)` 最小者，带数值病态与停滞护栏。
    `c` 只用来标注 `mu_converged`，不参与选阶，也不参与门禁分级。

    额外回传 `L_fit`（Lin-KK 自己拟出的串联电感）、`C_fit`（add_cap 时）、
    `mu_overfit`（μ<0.1，RC 系数大量负值）与 `M_scanned`（实际扫了几阶）。
    """
    from impedance.validation import fit_linKK, get_tc_distribution

    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    n = f.size
    # 参数个数 M+2（add_cap 再 +1）不超过点数的一半，避免设计矩阵病态。
    m_hi = int(min(max_M, max(4, (n - 2) // 2)))
    zabs = np.maximum(np.abs(Z), EPS)
    hist = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for M in range(1, m_hi + 1):
            try:
                ts = get_tc_distribution(f, M)
                elements, mu = fit_linKK(f, ts, M, Z, fit_type, add_cap)
                Z_fit = _eval_linkk(elements, ts, f)
            except Exception:                  # 单个 M 病态不影响其它阶
                continue
            err = (Z - Z_fit) / zabs
            r_all = float(np.sqrt(np.mean(np.abs(err) ** 2)))
            if not np.isfinite(r_all):
                continue
            # 数值病态护栏：残差相对已见过的最好值突然放大 → 停。
            if hist and r_all > blowup * min(h["r_all"] for h in hist):
                break
            hist.append({"M": M, "mu": mu, "elements": elements, "ts": ts,
                         "err": err, "r_all": r_all})
            # 停滞护栏：连续 patience 阶相对改善 < 2% → 停。
            if M >= 8 and len(hist) >= patience + 1:
                r0 = hist[-(patience + 1)]["r_all"]
                if r0 > 0 and (r0 - r_all) / r0 < 0.02:
                    break
    if not hist:
        raise RuntimeError("Lin-KK 没有产生拟合结果")

    # 肘部：全谱残差 × 复杂度惩罚。比"取最小残差"更简约，也避免为了
    # 端部最后一点残差把 M 推到数值病态区。
    pick = min(hist, key=lambda h: h["r_all"] * (1.0 + elbow * h["M"]))
    elements, ts, err = pick["elements"], pick["ts"], pick["err"]
    mu = pick["mu"]
    L_fit = float(elements[-1])
    C_fit = None
    if add_cap and elements.size == ts.size + 3 and elements[-2] != 0:
        C_fit = float(1.0 / elements[-2])
    # 纯电阻这类退化谱上 calc_mu 可能给出 -inf / nan；对外用 None，
    # 避免 JSON 里出现 Infinity / NaN，也让报告能写 "n/a"。
    mu_val = float(mu) if np.isfinite(mu) else None
    return {"M": pick["M"], "mu": mu_val, "Z_fit": Z - err * zabs,
            "res_re": err.real, "res_im": err.imag,
            # 非有限 μ 只出现在退化谱上，不构成"不自洽"的证据。
            "mu_converged": bool(mu <= c) if np.isfinite(mu) else True,
            "mu_overfit": bool(np.isfinite(mu) and mu < 0.1),
            "M_scanned": len(hist), "M_hi": m_hi,
            "L_fit": L_fit, "C_fit": C_fit, "add_cap": bool(add_cap)}


def _lin_kk_safe(f, Z, c, max_M, add_cap=False):
    """包一层：Lin-KK 在点数太少 / 频段太窄 / 数值病态时可能抛异常。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return lin_kk(f, Z, c=c, max_M=max_M, add_cap=add_cap)
    except Exception as e:                     # noqa: BLE001 - 门禁不能让主流程崩
        return {"error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
# 残差按 log f 分频段统计
# --------------------------------------------------------------------------

def _band_rms(r: dict, f=None) -> dict:
    """把复相对残差按 **log f 三分位点**分高/中/低，各段报 RMS。

    为什么要分频段：端部的偏差最可能来自 L / 窗口外扩散尾这类
    **Lin-KK 表达不了但物理正常**的东西；中段才是判断
    "数据本身自不自洽"的主证据。用 log f 分位而非索引三分，
    是为了在频点分布不均匀时仍按十倍频程切段。
    """
    if "res_re" not in r:
        return {"high": float("nan"), "mid": float("nan"),
                "low": float("nan"), "all": float("nan")}
    e = r["res_re"] + 1j * r["res_im"]
    mag = np.abs(e)
    n = mag.size
    v = float(np.sqrt(np.mean(mag ** 2)))
    if n < 9 or f is None or np.asarray(f).size != n:
        return {"high": v, "mid": v, "low": v, "all": v}

    lf = np.log10(np.asarray(f, float))
    q1, q2 = np.quantile(lf, [1.0 / 3.0, 2.0 / 3.0])
    low = lf <= q1
    high = lf > q2
    mid = ~(low | high)
    if not (low.any() and mid.any() and high.any()):
        return {"high": v, "mid": v, "low": v, "all": v}

    rms = lambda m: float(np.sqrt(np.mean(mag[m] ** 2)))  # noqa: E731
    return {"low": rms(low), "mid": rms(mid),
            "high": rms(high), "all": v}


def _residual_structure(f, r: dict) -> dict:
    """中段残差的"系统性 vs 噪声"判据。

    沿用 `AutoECM/validate_fit.py` 残差结构检查的思路（滞后-1 自相关 +
    符号变化率）。随机噪声的 ac1 接近 0 或为负、符号变化率高；
    漂移 / 非因果这类系统性偏差在两个分量上都平滑，ac1 明显为正。

    必须**实部与虚部同时**系统性才算：只在一个分量上平滑，往往只是
    加噪谱在低 |Z| 端被相对残差放大，或 Lin-KK 基底表达力不足。

    **lag-1 只在频率点近似对数等间距时才成立。** 原始扫描通常是对数等间距，
    但清洗后的数据常有合并/剔除，相邻两点的频率跨度可能差好几倍；这时
    "跨 3 decade 的一对"和"跨 0.05 decade 的一对"被同等对待，自相关就失真。
    处理：只保留间距落在中位间距 [0.5, 2] 倍内的相邻对；不足 3 对时退回全部
    （并保持原判据不变，避免在稀疏谱上悄悄换判据）。
    """
    nan = float("nan")
    out = {"ac1_re": nan, "ac1_im": nan, "scr_re": nan, "scr_im": nan,
           "rms_re": nan, "rms_im": nan, "systematic": False,
           "n_pairs": 0, "spacing_uniform": True}
    if "res_re" not in r or f is None:
        return out
    lf = np.log10(np.asarray(f, float))
    q1, q2 = np.quantile(lf, [1.0 / 3.0, 2.0 / 3.0])
    m = (lf > q1) & (lf <= q2)
    if int(m.sum()) < 5:
        m = np.ones(lf.size, bool)

    # 间距筛选：只留对数间距与中位数可比的相邻对
    dlf = np.diff(lf[m])
    pair = np.ones(dlf.size, bool)
    if dlf.size >= 4:
        med = float(np.median(dlf))
        if med > 0:
            pair = (dlf >= 0.5 * med) & (dlf <= 2.0 * med)
            out["spacing_uniform"] = bool(pair.mean() >= 0.8)
    if int(pair.sum()) < 3:
        pair = np.ones(dlf.size, bool)      # 太稀疏就别筛，保持旧行为
        out["spacing_uniform"] = False
    out["n_pairs"] = int(pair.sum())

    for name, key in (("re", "res_re"), ("im", "res_im")):
        x = np.asarray(r[key], float)[m]
        rms = float(np.sqrt(np.mean(x ** 2)))
        out[f"rms_{name}"] = rms
        if x.size < 4:
            continue
        a, b = x[:-1][pair], x[1:][pair]
        sd = float(np.std(np.concatenate([a, b])))
        if sd <= 1e-30:
            ac1 = 1.0 if rms > 1e-30 else 0.0
        else:
            ac1 = float(np.corrcoef(a, b)[0, 1])
        out[f"ac1_{name}"] = ac1
        out[f"scr_{name}"] = float(np.mean(np.sign(a) != np.sign(b)))
    out["systematic"] = bool(out["ac1_re"] > 0.5 and out["ac1_im"] > 0.5)
    return out


# --------------------------------------------------------------------------
# 两种"可解释的端部偏差"诊断量（都不改写数据）
# --------------------------------------------------------------------------

def estimate_L(f, Z, n_hi: int = 5):
    """由最高若干点的 Im Z / ω 稳健估计串联电感 L（单位 H）。

    高频时 `Im Z ≈ ωL`（容性部分已衰减），取 Im>0 的点求中位数；
    若没有感抗点返回 None。

    **注意**：这个估计值**不要**在 Lin-KK 之前扣掉（Lin-KK 已含自由 L）。
    它的用途是给 DRT 预判（Tikhonov 反演表示不了 jωL）和报告归因，
    见计划 P2-4。
    """
    f, Z = np.asarray(f, float), np.asarray(Z, complex)
    hi = np.argsort(f)[-min(n_hi, f.size):]
    w = 2 * np.pi * f[hi]
    pos = Z.imag[hi] > 0
    if not np.any(pos):
        return None
    L = float(np.median(Z.imag[hi][pos] / w[pos]))
    return L if np.isfinite(L) and L > 0 else None


def nyquist_slope_low(f, Z, frac: float = 0.25, min_re_span: float = 0.01):
    """低频段 Nyquist 斜率，用 `−Im Z` 对 `Re Z` 做**最小二乘**直线拟合。

    半无限 Warburg `Z_W = A(jω)^(−1/2)` 满足 `Re = −Im`，斜率 = **+1**
    （Nyquist 图上的 45° 直线）。

    两个实现选择都是踩过坑才定的：

    * **不除以 `d Re/d log f`**：`Re` 饱和（阻塞电极、`Wo` 低频平台）时
      分母趋 0，比值法直接炸成 ±∞（实测 `caseF_wo` +21、`caseG` −30，
      符号还随 `frac` 翻转）。最小二乘对分母不敏感。
    * **不用 `d log|Z| / d log f ≈ −0.5`**：那个量会被串联 `R_inf` 压平。
      实测 `caseE_warburg`（真值含 `Ws`）`|Z|` 斜率只有 −0.02，
      因为 40 Ω 的实部把 2.5 Ω 的 Warburg 淹了。

    返回 `(slope, re_span_rel, stability)`：
        slope        最小二乘斜率；`Re` 跨度不足时为 None（斜率无定义）
        re_span_rel  该段 `Re` 跨度 / `|Z|max`，用来看这个斜率可不可信
        stability    逐点斜率落在 1±0.35 的比例（噪声大时会很低）
    """
    f, Z = np.asarray(f, float), np.asarray(Z, complex)
    n_low = max(4, int(round(f.size * frac)))
    idx = np.argsort(f)[:n_low]
    re, im = Z.real[idx], -Z.imag[idx]
    zmax = max(float(np.max(np.abs(Z[idx]))), EPS)
    span = float(re.max() - re.min())
    rel = span / zmax
    if span <= min_re_span * zmax:
        # Re 基本不动：Nyquist 斜率没有定义（不是"斜率=0"）
        return None, rel, None
    try:
        b = float(np.polyfit(re, im, 1)[0])
    except Exception:                          # noqa: BLE001
        return None, rel, None
    lf = np.log10(np.maximum(f[idx], EPS))
    dre = np.gradient(re, lf)
    dim = np.gradient(im, lf)
    ok = np.abs(dre) > 1e-9 * zmax
    stab = float(np.mean(np.abs(dim[ok] / dre[ok] - 1.0) <= 0.35)) if ok.any() else None
    return (b if np.isfinite(b) else None), rel, stab


def diffusion_tail_candidate(f, Z, frac: float = 0.25, min_keep: int = 16,
                             slope_lo: float = -0.85, slope_hi: float = -0.20,
                             ny_tol: float = 0.35):
    """低频扩散尾**候选**（诊断量，不截断任何数据点）。

    两条判据必须同时成立才给出"候选"：

      A. 幅度判据：最低 `frac` 段的 `log(−Im) ~ log f` 斜率 ∈ [slope_lo, slope_hi]；
         **不用 `log|Z|`**——`R_inf` 只加在实部上，会把 `|Z|` 的斜率压平
         （实测 `example_caseE`：`|Z|` −0.03、`−Im` −0.49）。
      B. 相位判据：同段 Nyquist 斜率 `d(−Im)/d(Re)` ∈ 1 ± `ny_tol`。

    **两条都只是必要条件，不是充分条件。** 这一点必须写清楚，因为：
      · **任何一条圆弧的下降段都会经过 45°**。`example_caseE` 最低 10% 段
        Nyquist 斜率 +1.02、逐点稳定度 1.00，跨度占 `|Z|` 的 10% ——
        只看相位判据，任何未闭合弧都会被判成扩散尾。
      · 审查意见提到的「单个 `R‖CPE(n=0.5)` 也有 φ=45°」同理：45° 不是
        扩散阻抗独有的签名。本次实测窄窗口（`f` 跨 4 个数量级、
        `τ=0.16 s`、`φ=0.5`）里最低 25% 落在弧的已弛豫段（斜率 −0.78），
        没有复现假阳性；但这只是"没测到"，不是"不会发生"。
        `evidence/ws_crit/bench.py` 的三类阴性对照（2×ZARC、窗内
        `R‖CPE(0.5)`、`R‖L` 吸附环）共 120 次，假阳 0 次。

    > **2026-11-09 注（`WS_DECISION_PROJECT.md` WP4）**：本函数此前的注释把
    > `example_caseE.csv` 写成「无扩散元件」并当成「被幅度判据成功挡掉的
    > 假阳性」举例，**方向与真值相反**——该谱真值是
    > `R∞=10 + ZARC(30Ω,300Hz,.85) + Ws(A_W=2)`（复算见
    > `evidence/ws_crit/n1_fitcasee.py`）。旧 `|Z|` 口径把它判成"无候选"
    > 是**漏报**，不是挡对了；换成 `−Im` 口径后 `−Im` 斜率 −0.49 →
    > `is_candidate=True`。原文的"挡掉"叙述已删，教训（45° 不唯一）保留。

    所以函数名叫 `candidate`：它能**否证**（斜率对不上就不是扩散尾），
    不能**确认**。真正确认要靠"在连续频段上停在 45°"以及窗口外扩散的
    物理约束（见 `bm_diff.warburg_band_slope` 的多点判据）。

    旧 `log|Z|` 口径会漏掉被 `R_inf` 淹没的短尾：`caseE_warburg`（真值
    `R∞=10+ZARC+Ws`，`A_W=2`）在 0.1 Hz 的 Warburg 只有 2.5 Ω、实部 40 Ω，
    `|Z|` 斜率 −0.02 → 判不成（**保守漏报**）。换成 `log(−Im)` 后同一段是
    −0.45 → 判成候选。`−Im` 口径的残余盲区是**噪声**而不是 `R_inf`：
    `SNR_im ≲ 15` 时检出率约 50%（`WS_CRITERION_PLAN.md` §9.4）。

    低频 `−Im ≤ 0`（第四象限感抗、或尾已贴实轴）时取对数无意义，函数**提前
    返回**：只填 `slope_z` 与相位字段，`slope_im=None`、`amp_ok` /
    `is_candidate` 恒 False，`why` 记「幅度判据不可用」。18 条真实 AZ31
    全部走这条（低频是感抗，见 `WS_CRITERION_PLAN.md` §5.5）——
    这是**"判不了"**，不是"判成没有扩散"。

    返回 dict：
        f_cut         候选截断点（Hz）；A 不成立时为 None。**仅诊断，不截断**
        slope_z       最低段的 `log|Z| ~ log f` 斜率。**只报数不判定**：
                      `kk_check` 用它填 `tail_slope`，口径与旧实现一致
        slope_im      最低段的 `log(−Im) ~ log f` 斜率，**这才是幅度判据**；
                      低频 `−Im ≤ 0` 时为 None
        slope_ny      相位判据的斜率（None = `Re` 跨度不足，斜率无定义）
        ny_span       该段 `Re` 跨度 / `|Z|`，判断 slope_ny 可信度
        ny_stability  逐点斜率落在 45°±0.35 的比例
        amp_ok        幅度判据（用 `slope_im`）。**对 `R_inf` 免疫**，
                      但对低频噪声 / 短尾敏感
        phase_ok      相位判据（对 `R_inf` 免疫：平移不改斜率）
        is_candidate  两条同时成立（**必要不充分**）
        why           人类可读说明

    两条判据各有一个方向的盲区，**不要**拿单条去下结论：
      · 幅度成立、相位不成立 → 不是扩散尾。`caseF_wo` / `caseG_wo_only`
        在旧 `|Z|` 口径下幅度成立（−0.54 / −0.81）而 Ny 远超 45°；新
        `−Im` 口径下两个都落在区间外（−0.97 / −0.96）。**两种口径都给
        "非候选"**，只是理由不同——`Wo` 本来就不该进 Ws 家族
        （`WS_CRITERION_PLAN.md` §10.1）。
      · 相位成立、幅度不可用（`−Im ≤ 0`）→ **本地判不了**，不是"没有扩散"。
        真实 AZ31 谱全是这种尾（低频感抗），只能记 `inductive_tail`。
        "被 `R_inf` 压平"这一类已由 `−Im` 口径解决；剩下的盲区是**噪声**
        （`SNR_im` 门槛），不是形态。
        持续性属于频段判据，交给 `bm_diff.warburg_band_slope`。
    """
    f, Z = np.asarray(f, float), np.asarray(Z, complex)
    out = dict(f_cut=None, slope_z=None, slope_im=None,
               slope_ny=None, ny_span=None,
               ny_stability=None, amp_ok=False, phase_ok=False,
               is_candidate=False, why=[])
    n = f.size
    n_low = max(4, int(round(n * frac)))
    if n - n_low < min_keep:
        out["why"].append(f"点数不足（{n} 点，最低段 {n_low} 点），不做扩散尾判断")
        return out

    idx = np.argsort(f)[:n_low]
    lf = np.log10(np.maximum(f[idx], EPS))
    try:
        # slope_z 保持 |Z| 口径：`kk_check` 用它填 `tail_slope`，即使幅度判据
        # 不成立也要照着报（旧实现就是这么做的，别让它变成 None）。
        slope_z = float(np.polyfit(lf, np.log10(np.maximum(np.abs(Z[idx]), EPS)), 1)[0])
    except Exception:                          # noqa: BLE001
        return out
    # --- N1：幅度判据改用 -Im。R_inf 只加在实部上，log|Z| 会被它压平。
    # 实测 caseE / caseE_warburg：|Z| 斜率 -0.03/-0.02（判不成），
    # -Im 斜率 -0.49/-0.45。
    im_low = -Z[idx].imag
    if im_low.min() <= 0:
        ny, span, stab = nyquist_slope_low(f, Z, frac)
        out.update(slope_z=slope_z, slope_ny=ny, ny_span=span,
                   ny_stability=stab,
                   phase_ok=bool(ny is not None and abs(ny - 1.0) <= ny_tol))
        out["why"].append("幅度判据不可用：低频段 -Im ≤ 0（感抗或贴实轴），"
                          "取对数无意义")
        return out
    try:
        slope = float(np.polyfit(lf, np.log10(im_low), 1)[0])
    except Exception:                          # noqa: BLE001
        return out
    ny, span, stab = nyquist_slope_low(f, Z, frac)
    out.update(slope_z=slope_z, slope_im=slope, slope_ny=ny,
               ny_span=span, ny_stability=stab)

    amp_ok = bool(slope_lo <= slope <= slope_hi)
    pha_ok = bool(ny is not None and abs(ny - 1.0) <= ny_tol)
    out.update(amp_ok=amp_ok, phase_ok=pha_ok)

    if amp_ok:
        out["f_cut"] = float(f[idx][-1])
    ny_txt = f"{ny:+.2f}" if ny is not None else "无定义"
    if amp_ok and pha_ok:
        out.update(is_candidate=True,
                   why=[f"幅度斜率 {slope:+.2f} 且 Nyquist 斜率 {ny_txt}"
                        f"（45°±{ny_tol:.2f}，Re 跨度 {span:.1%}·|Z|）"
                        f"→ **扩散尾候选**（必要不充分）"])
    elif amp_ok:
        out["why"].append(
            f"幅度斜率 {slope:+.2f} 落在扩散区间，但 Nyquist 斜率 {ny_txt}"
            f"（Re 跨度 {span:.1%}·|Z|）偏离 45° → **不判扩散尾**；"
            f"更可能是串联电阻压平 / 窗口边缘容性衰减 / 单条弧的下降段")
    elif pha_ok:
        out["why"].append(
            f"Nyquist 斜率 {ny_txt} 在 45°±{ny_tol:.2f} 内，但幅度斜率 "
            f"{slope:+.2f} 不在 [{slope_lo:.2f}, {slope_hi:.2f}]"
            f"（Re 跨度 {span:.1%}·|Z|）→ **本地无法判定**："
            f"可能是被 R_inf 压平的短尾，也可能是未闭合弧的下降段"
            f"（两者在此窗口形态相同）。持续性请查频段判据")
    else:
        out["why"].append(
            f"幅度斜率 {slope:+.2f}、Nyquist 斜率 {ny_txt}，两判据都不成立")
    return out


def diffusion_tail_cut(f, Z, frac: float = 0.25, min_keep: int = 16):
    """**已弃用**，保留仅为兼容旧调用；请用 `diffusion_tail_candidate()`。

    旧版只凭 `|Z|` 斜率断言「−45° 扩散尾」，是过度断言（见新函数 docstring）。
    这里只回传 `(f_cut, slope_z)`，语义与原版一致，但**不再**代表"确认是扩散尾"。
    """
    c = diffusion_tail_candidate(f, Z, frac=frac, min_keep=min_keep)
    return c["f_cut"], c["slope_z"]


# --------------------------------------------------------------------------
# 门禁主入口
# --------------------------------------------------------------------------

def kk_check(f, Z, include_L=None, c: float = 0.85, max_M: int = 50,
             res_tol: float = 0.05, auto_L: bool = True,
             auto_tail: bool = True, auto_cap: bool = True,
             diagnose_L=None) -> dict:
    """对一条谱做 Lin-KK 分级体检。

    diagnose_L：感抗**诊断**开关（估计 L_est / 高频归因）。None（默认）=
    跟随 include_L（旧行为）；False = 明确关闭诊断——与「不建 L 参数」
    解耦（审查 P2-7，二轮）：主模型不加 L 参数时，端部感抗诊断反而更
    需要，不该被同一开关连坐关掉。

    返回 dict（全部可 JSON 序列化）：
        status        ok | warn | fail | unavailable
        reason        人类可读结论
        mu / M / mu_converged（μ 只作过拟合参考）/ mu_overfit / M_scanned
        rms_raw       原谱 高/中/低/全频段 相对残差 RMS
        rms_cap       add_cap 复验的残差（未触发则 None）
        rms_clean     仅低频段取 raw / cap 更优者，中/高频仍用 raw
        residual_structure  中段残差 ac1/符号变化率/systematic
        L_fit         Lin-KK 自身拟出的串联电感（H）
        L_est         高频 Im Z/ω 稳健估计（H），None 表示没有感抗点
        f_cut         低频扩散尾**候选**截断点（Hz），**仅诊断不截断**
        tail_slope    低频 `log|Z|~log f` 斜率（幅度判据）
        tail          `diffusion_tail_candidate()` 的完整返回（含 Nyquist 斜率、
                      `Re` 跨度与 `is_candidate`；两条判据只是必要不充分）
        attribution   归因字符串列表
        n / f_min / f_max

    判定口径（保守，避免把正常谱全拒）：
      · 中段 > res_tol 且 实/虚部残差同时系统性 → fail
      · 中段 > 2*res_tol                       → fail（无论结构）
      · 中段 > res_tol 但非系统性              → warn（噪声 / 表达力不足）
      · 端部超阈                               → warn（写明归因）
      · 其余                                   → ok
      · impedance 缺失 / Lin-KK 抛异常         → unavailable（只警告不中断）
    μ 不参与分级：它只反映 RC 系数正负号，既可能因欠拟合偏高，也可能因
    过拟合偏低；μ<0.1 时在 attribution 里提示"残差判据可能偏松"。
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    order = np.argsort(f)
    f, Z = f[order], Z[order]

    base = dict(status="ok", reason="", mu=None, M=None, mu_converged=None,
                mu_overfit=None, M_scanned=None,
                        rms_raw=None, rms_cap=None, rms_clean=None,
                residual_structure=None, res_tol=float(res_tol),
                L_fit=None, L_est=None, f_cut=None, tail_slope=None, tail=None,
                attribution=[], n=int(f.size),
                f_min=float(f.min()), f_max=float(f.max()))

    if f.size < 12:
        base.update(status="unavailable",
                    reason=f"仅 {f.size} 个频点，Lin-KK 检验不可靠，跳过门禁"
                           f"（需 ≥12 点）")
        return base

    try:
        import impedance                        # noqa: F401
    except Exception as e:                     # noqa: BLE001
        base.update(status="unavailable",
                    reason=f"未安装 impedance，跳过 K-K 门禁（{type(e).__name__}）；"
                           f"请 pip install impedance>=1.7")
        return base

    raw = _lin_kk_safe(f, Z, c, max_M)
    if "error" in raw:
        base.update(status="unavailable",
                    reason=f"Lin-KK 计算失败，跳过门禁：{raw['error']}")
        return base
    rms_raw = _band_rms(raw, f)
    base.update(mu=raw["mu"], M=raw["M"], mu_converged=raw["mu_converged"],
                mu_overfit=raw.get("mu_overfit"),
                M_scanned=raw.get("M_scanned"),
                L_fit=raw["L_fit"], rms_raw=rms_raw)

    # ---- 端部诊断量（不改写数据）----
    # P2-7（二轮审查）：诊断开关 diagnose_L 与建模开关 include_L 解耦——
    # None（默认）跟随 include_L（逐位兼容旧行为），显式 False 才真正关闭。
    _diag_L = (include_L if diagnose_L is None else diagnose_L)
    L_est = estimate_L(f, Z) if (auto_L and _diag_L is not False) else None
    tail = diffusion_tail_candidate(f, Z) if auto_tail else None
    f_cut = (tail or {}).get("f_cut")
    slope = (tail or {}).get("slope_z")
    base.update(L_est=L_est, f_cut=f_cut, tail_slope=slope, tail=tail)

    # ---- 低频扩散尾：add_cap 在同一网格复验，不丢点 ----
    rms_cap = None
    cap = None
    if auto_cap and (f_cut is not None or rms_raw["low"] > res_tol):
        cap = _lin_kk_safe(f, Z, c, max_M, add_cap=True)
        if "error" not in cap:
            rms_cap = _band_rms(cap, f)
            base["rms_cap"] = rms_cap

    # 只放宽低频段：add_cap 是给"窗口外扩散尾"的解释自由度，
    # 不能让它顺带把中段的不自洽吸收掉（否则门禁会被自己骗过）。
    rms_clean = dict(rms_raw)
    if rms_cap is not None:
        rms_clean["low"] = min(rms_raw["low"], rms_cap["low"])
    base["rms_clean"] = rms_clean
    struct = _residual_structure(f, raw)
    base["residual_structure"] = struct
    mu_txt = (f"μ={base['mu']:.2f}" if base["mu"] is not None else "μ=n/a")

    # ---- 归因 ----
    attr = []
    if rms_cap is not None and rms_cap["low"] < rms_raw["low"]:
        # 不能因为「加串联电容后低频变好了」就说是扩散尾——add_cap 只是在
        # 低频端给了一个额外的自由发散项，能吸收很多种端部效应。
        # 只有 Nyquist 45° 成立时才用"扩散尾"这个词。
        ny = (tail or {}).get("slope_ny")
        ny_txt = f"{ny:+.2f}" if ny is not None else "无定义"
        if (tail or {}).get("is_candidate"):
            attr.append(
                f"低频扩散尾候选（Nyquist 斜率 {ny_txt}≈45°，|Z| 斜率 "
                f"{slope if slope is not None else float('nan'):+.2f}）："
                f"Lin-KK 加串联电容后低频残差 {rms_raw['low']:.1%} → "
                f"{rms_cap['low']:.1%}，未丢弃数据点")
        else:
            attr.append(
                f"低频端部被串联电容解释（残差 {rms_raw['low']:.1%} → "
                f"{rms_cap['low']:.1%}），但 Nyquist 斜率 {ny_txt}"
                f"（|Z| 斜率 {slope if slope is not None else float('nan'):+.2f}）"
                f"→ **未达 45°，不判为扩散尾**；归因为窗口外端部效应即可")
        if cap.get("C_fit") is not None:
            attr.append(f"  等效串联电容 C={cap['C_fit']:.3g} F（仅为 KK 校验用，非物理元件）")
    if L_est is not None:
        attr.append(
            f"高频感抗尾：稳健估计 L={L_est:.3g} H；Lin-KK 已含自由串联电感 "
            f"L_fit={raw['L_fit']:.3g} H，故不再扣 jωL（重复计入会放大残差）")
    elif rms_clean["high"] > res_tol:
        # include_L=False 时 L_est 恒为 None，不能因此断言"无感抗点"——
        # 那是"跳过诊断"，不是"数据没有感抗"（审查 P2-10）。
        if _diag_L is False:
            attr.append("高频段残差超阈：感抗诊断已被显式关闭，"
                        "未做归因，需人工复核")
        else:
            attr.append("高频段残差超阈，但高频无感抗点可归因，需人工复核")
    if rms_clean["low"] > res_tol:
        attr.append(f"低频段残差 {rms_clean['low']:.1%} 仍超阈："
                    f"窗口外扩散 / 阻截电极端点效应，定 K 时勿当作额外弧")
    if rms_clean["high"] > res_tol and L_est is not None:
        attr.append(f"高频段残差 {rms_clean['high']:.1%} 仍超阈："
                    f"局部感抗未被单一 L 完全吸收，定 K 时勿当作额外弧")
    if raw.get("mu_overfit"):
        attr.append(f"μ={base['mu']:.2f} <0.1：RC 系数出现大量负值，接近过拟合，"
                    f"残差判据可能偏松，引用结论前需人工复核")
    base["attribution"] = attr

    # ---- 分级 ----
    mid = rms_clean["mid"]
    mid_bad = mid > res_tol
    gross = mid > 2 * res_tol
    end_bad = (rms_clean["high"] > res_tol) or (rms_clean["low"] > res_tol)
    sys_txt = (f"中段残差 ac1(实/虚)={struct['ac1_re']:.2f}/"
               f"{struct['ac1_im']:.2f}")

    if (mid_bad and struct["systematic"]) or gross:
        why = ("系统性（实/虚部同时自相关）" if struct["systematic"]
               else "残差超过 2 倍阈值")
        base.update(status="fail",
                    reason=f"中段相对残差 {mid:.1%} > {res_tol:.1%} 且{why}；"
                           f"{sys_txt}，与端部电感/扩散无关 → 谱不自洽，拒绝定 K")
        return base
    if mid_bad:
        base.update(status="warn",
                    reason=f"中段残差 {mid:.1%} > {res_tol:.1%}，但 {sys_txt} "
                           f"未同时呈系统性 → 更可能是噪声或 Lin-KK 对扩散谱的"
                           f"表达力不足，**不能判定谱不自洽**；继续拟合但需看残差图",
                    attribution=attr + [f"中段残差超阈但非系统性：{sys_txt}"])
        return base
    if end_bad:
        base.update(status="warn",
                    reason=f"中段残差 {mid:.1%} ≤ {res_tol:.1%} 正常，"
                           f"但端部需归因（见 attribution）→ 可拟合，"
                           f"引用参数时须说明端部处理")
        return base
    base.update(status="ok",
                reason=f"Lin-KK {mu_txt}、M={base['M']}，"
                       f"中段相对残差 {mid:.1%} ≤ {res_tol:.1%} → 数据自洽")
    return base


def format_report(g: dict) -> str:
    """把 kk_check 结果打成几行可读文本（供 bm_real.report / CLI 用）。"""
    tag = {"ok": "通过", "warn": "通过（有归因）", "fail": "不通过",
           "unavailable": "未执行"}.get(g["status"], g["status"])
    lines = [f"  Lin-KK 门禁：{tag} — {g['reason']}"]
    if g.get("mu") is not None:
        r = g["rms_raw"] or {}
        c = g["rms_clean"] or {}
        lines.append(f"    μ={g['mu']:.3f}  M={g['M']}"
                     f"（M 由残差肘部选择；μ 仅作过拟合参考）  "
                     f"相对残差 RMS 高/中/低 = "
                     f"{r.get('high', float('nan')):.2%}/"
                     f"{r.get('mid', float('nan')):.2%}/"
                     f"{r.get('low', float('nan')):.2%}"
                     + (f"  →复验后 {c.get('high', float('nan')):.2%}/"
                        f"{c.get('mid', float('nan')):.2%}/"
                        f"{c.get('low', float('nan')):.2%}"
                        if c != r else ""))
        s = g.get("residual_structure") or {}
        if np.isfinite(s.get("ac1_re", float("nan"))) and \
           np.isfinite(s.get("ac1_im", float("nan"))):
            cm = (c.get("mid", float("nan")))
            if np.isfinite(cm) and cm <= float(g.get("res_tol", 0.05)):
                tail = "中段未超阈，该判据不触发"
            else:
                tail = ("系统性 → 判不自洽" if s.get("systematic")
                        else "非系统性 → 更可能是噪声/表达力不足")
            lines.append(f"    中段残差结构 ac1(实/虚)={s['ac1_re']:.2f}/"
                         f"{s['ac1_im']:.2f}；{tail}")
        if g.get("L_fit") is not None:
            lines.append(f"    Lin-KK 自由串联电感 L_fit={g['L_fit']:.3g} H"
                         + (f"（高频稳健估计 L={g['L_est']:.3g} H）"
                            if g.get("L_est") is not None else ""))
        tl = g.get("tail") or {}
        if tl.get("slope_z") is not None:
            ny = tl.get("slope_ny")
            lines.append(
                f"    低频端部：|Z| 斜率 {tl['slope_z']:+.2f}，Nyquist 斜率 "
                + (f"{ny:+.2f}" if ny is not None else "无定义")
                + (f"（Re 跨度 {tl['ny_span']:.1%}·|Z|）"
                   if tl.get("ny_span") is not None else "")
                + ("  → 45° 扩散尾候选（必要不充分）" if tl.get("is_candidate")
                   else "  → 不判扩散尾"))
    for a in g.get("attribution", []):
        lines.append(f"    · {a}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="拟合前的 Lin-KK 门禁体检")
    ap.add_argument("csv")
    ap.add_argument("--c", type=float, default=0.85, help="μ 收敛阈值")
    ap.add_argument("--max-M", type=int, default=50, help="RC 链最大阶数")
    ap.add_argument("--tol", type=float, default=0.05, help="相对残差 RMS 阈值")
    ap.add_argument("--no-L", action="store_true", help="不做电感诊断")
    ap.add_argument("--no-tail", action="store_true", help="不做低频尾识别")
    ap.add_argument("--no-cap", action="store_true", help="不做 add_cap 复验")
    a = ap.parse_args()
    from fit_store import load_spectrum_csv
    f, Z = load_spectrum_csv(a.csv)
    g = kk_check(f, Z, c=a.c, max_M=a.max_M, res_tol=a.tol,
                 auto_L=not a.no_L, auto_tail=not a.no_tail,
                 auto_cap=not a.no_cap)
    print(f"=== {Path(a.csv).name} ===")
    print(format_report(g))
    print("\n" + json.dumps(g, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
