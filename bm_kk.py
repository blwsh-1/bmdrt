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

分级判定的关键：中段才是主证据
------------------------------
    1. 原谱做 Lin-KK，按 **log f 三分**统计复相对残差 RMS；
    2. 低频若有 −45° 尾 / 低频残差超阈 → 用 `add_cap=True` 在同一频率网格复验；
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

    实测标定（`examples/example_caseE.csv` 派生合成谱）：
    谱                        中段残差   中段 ac1(re/im)     判定
    干净 caseE                0.65%     -0.57/-0.13        ok
    实部鼓包 0.10R            2.89%     +0.52/+0.93        ok（低于阈值）
    实部鼓包 0.20R            5.50%     +0.72/+0.96        fail（系统性）
    实部鼓包 0.40R           15.13%     +0.90/+0.83        fail（残差过大）
    低频漂移 0.10R           17.01%     +0.83/+0.97        fail（残差过大）
    加噪 Wo-only (caseG)      8.53%     +0.71/+0.09        warn（噪声，非系统性）
    真实 2E                   1.84%     +0.84/+0.83        warn（端部归因）
    注意 μ 在这些非因果谱上仍然 ≈0.80「收敛」，所以 μ 只能当次要信号。

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
           fit_type: str = "complex", add_cap: bool = False) -> dict:
    """线性 Kramers-Kronig 检验，返回 M / mu / Z_fit / 实虚部相对残差。

    绕开 impedance.py 1.7.1 在 numpy 2.x 下的两个兼容问题：
    ① `M % 10 == 0` 分支里留了一行调试 print，eval_linKK 里 np 未定义会崩；
    ② eval_linKK 本身用 repr 拼字符串，同样崩。
    `fit_linKK` / `get_tc_distribution` 是纯 numpy 运算，可以直接用。

    额外回传 `L_fit`（Lin-KK 自己拟出的串联电感）与 `C_fit`（add_cap 时）。
    """
    from impedance.validation import fit_linKK, get_tc_distribution

    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    M, mu, elements, ts = 0, 1.0, None, None
    while mu > c and M < max_M:
        M += 1
        ts = get_tc_distribution(f, M)
        elements, mu = fit_linKK(f, ts, M, Z, fit_type, add_cap)
    if elements is None:                       # 理论上只在 max_M=0 时发生
        raise RuntimeError("Lin-KK 没有产生拟合结果")
    Z_fit = _eval_linkk(elements, ts, f)
    err = (Z - Z_fit) / np.maximum(np.abs(Z), EPS)
    L_fit = float(elements[-1])
    C_fit = None
    if add_cap and elements.size == ts.size + 3 and elements[-2] != 0:
        C_fit = float(1.0 / elements[-2])
    # 纯电阻这类退化谱上 calc_mu 可能给出 -inf / nan；对外用 None，
    # 避免 JSON 里出现 Infinity / NaN，也让报告能写 "n/a"。
    mu_val = float(mu) if np.isfinite(mu) else None
    return {"M": M, "mu": mu_val, "Z_fit": Z_fit,
            "res_re": err.real, "res_im": err.imag,
            # 非有限 μ 只出现在 M 太小的退化谱上，不构成"不自洽"的证据。
            "mu_converged": bool(mu <= c) if np.isfinite(mu) else True,
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
    """
    nan = float("nan")
    out = {"ac1_re": nan, "ac1_im": nan, "scr_re": nan, "scr_im": nan,
           "rms_re": nan, "rms_im": nan, "systematic": False}
    if "res_re" not in r or f is None:
        return out
    lf = np.log10(np.asarray(f, float))
    q1, q2 = np.quantile(lf, [1.0 / 3.0, 2.0 / 3.0])
    m = (lf > q1) & (lf <= q2)
    if int(m.sum()) < 5:
        m = np.ones(lf.size, bool)
    for name, key in (("re", "res_re"), ("im", "res_im")):
        x = np.asarray(r[key], float)[m]
        rms = float(np.sqrt(np.mean(x ** 2)))
        out[f"rms_{name}"] = rms
        if x.size < 4:
            continue
        sd = float(np.std(x))
        if sd <= 1e-30:
            ac1 = 1.0 if rms > 1e-30 else 0.0
        else:
            ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        out[f"ac1_{name}"] = ac1
        out[f"scr_{name}"] = float(np.mean(np.sign(x[:-1]) != np.sign(x[1:])))
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


def diffusion_tail_cut(f, Z, frac: float = 0.25, min_keep: int = 16):
    """识别低频 −45° 扩散尾，返回 (建议截断点 f_cut, 斜率)。

    判据：最低 `frac` 比例的点上 `log|Z| ~ log f` 斜率接近 −0.5
    （半无限 Warburg 的 `|Z| ∝ f^(-1/2)`）。含 R_inf 时斜率会偏缓，
    放宽到 [-0.85, -0.20]。

    **只作诊断**：本门禁改用 `add_cap=True` 复验而不丢点，
    见模块 docstring。返回值仍保留给后续「低频截断敏感性分析」用。
    """
    f, Z = np.asarray(f, float), np.asarray(Z, complex)
    n = f.size
    n_low = max(4, int(round(n * frac)))
    if n - n_low < min_keep:
        return None, None
    idx = np.argsort(f)[:n_low]
    lf, lz = np.log10(f[idx]), np.log10(np.maximum(np.abs(Z[idx]), EPS))
    try:
        slope = float(np.polyfit(lf, lz, 1)[0])
    except Exception:                          # noqa: BLE001
        return None, None
    if not (-0.85 <= slope <= -0.20):
        return None, slope
    return float(f[idx][-1]), slope


# --------------------------------------------------------------------------
# 门禁主入口
# --------------------------------------------------------------------------

def kk_check(f, Z, include_L=None, c: float = 0.85, max_M: int = 50,
             res_tol: float = 0.05, auto_L: bool = True,
             auto_tail: bool = True, auto_cap: bool = True) -> dict:
    """对一条谱做 Lin-KK 分级体检。

    返回 dict（全部可 JSON 序列化）：
        status        ok | warn | fail | unavailable
        reason        人类可读结论
        mu / M / mu_converged
        rms_raw       原谱 高/中/低/全频段 相对残差 RMS
        rms_cap       add_cap 复验的残差（未触发则 None）
        rms_clean     仅低频段取 raw / cap 更优者，中/高频仍用 raw
        residual_structure  中段残差 ac1/符号变化率/systematic
        L_fit         Lin-KK 自身拟出的串联电感（H）
        L_est         高频 Im Z/ω 稳健估计（H），None 表示没有感抗点
        f_cut         低频 −45° 尾建议截断点（Hz），**仅诊断不截断**
        tail_slope    低频 log|Z|~log f 斜率
        attribution   归因字符串列表
        n / f_min / f_max

    判定口径（保守，避免把正常谱全拒）：
      · 中段 > res_tol 且 实/虚部残差同时系统性 → fail
      · 中段 > 2*res_tol                       → fail（无论结构）
      · μ 未收敛 且 全谱残差 > res_tol         → fail
      · 中段 > res_tol 但非系统性              → warn（噪声 / 表达力不足）
      · 端部超阈 / μ 未收敛                    → warn（写明归因）
      · 其余                                   → ok
      · impedance 缺失 / Lin-KK 抛异常         → unavailable（只警告不中断）
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    order = np.argsort(f)
    f, Z = f[order], Z[order]

    base = dict(status="ok", reason="", mu=None, M=None, mu_converged=None,
                        rms_raw=None, rms_cap=None, rms_clean=None,
                residual_structure=None, res_tol=float(res_tol),
                L_fit=None, L_est=None, f_cut=None, tail_slope=None,
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
                L_fit=raw["L_fit"], rms_raw=rms_raw)

    # ---- 端部诊断量（不改写数据）----
    L_est = estimate_L(f, Z) if (auto_L and include_L is not False) else None
    f_cut, slope = diffusion_tail_cut(f, Z) if auto_tail else (None, None)
    base.update(L_est=L_est, f_cut=f_cut, tail_slope=slope)

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
    mu_ok = bool(raw["mu_converged"])
    struct = _residual_structure(f, raw)
    base["residual_structure"] = struct
    mu_txt = (f"μ={base['mu']:.2f}" if base["mu"] is not None else "μ=n/a")

    # ---- 归因 ----
    attr = []
    if rms_cap is not None and rms_cap["low"] < rms_raw["low"]:
        attr.append(
            f"低频 −45° 扩散尾（斜率 "
            f"{slope if slope is not None else float('nan'):.2f}）："
            f"Lin-KK 加串联电容后低频残差 {rms_raw['low']:.1%} → "
            f"{rms_cap['low']:.1%}，未丢弃数据点")
        if cap.get("C_fit") is not None:
            attr.append(f"  等效串联电容 C={cap['C_fit']:.3g} F（仅为 KK 校验用，非物理元件）")
    if L_est is not None:
        attr.append(
            f"高频感抗尾：稳健估计 L={L_est:.3g} H；Lin-KK 已含自由串联电感 "
            f"L_fit={raw['L_fit']:.3g} H，故不再扣 jωL（重复计入会放大残差）")
    elif rms_clean["high"] > res_tol:
        attr.append("高频段残差超阈，但高频无感抗点可归因，需人工复核")
    if rms_clean["low"] > res_tol:
        attr.append(f"低频段残差 {rms_clean['low']:.1%} 仍超阈："
                    f"窗口外扩散 / 阻截电极端点效应，定 K 时勿当作额外弧")
    if rms_clean["high"] > res_tol and L_est is not None:
        attr.append(f"高频段残差 {rms_clean['high']:.1%} 仍超阈："
                    f"局部感抗未被单一 L 完全吸收，定 K 时勿当作额外弧")
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
    if not mu_ok and rms_clean["all"] > res_tol:
        base.update(status="fail",
                    reason=f"Lin-KK {mu_txt} 未收敛到 {c} 且全谱残差 "
                           f"{rms_clean['all']:.1%} > {res_tol:.1%} "
                           f"→ 谱不自洽，拒绝定 K")
        return base
    if mid_bad:
        base.update(status="warn",
                    reason=f"中段残差 {mid:.1%} > {res_tol:.1%}，但 {sys_txt} "
                           f"未同时呈系统性 → 更可能是噪声或 Lin-KK 对扩散谱的"
                           f"表达力不足，**不能判定谱不自洽**；继续拟合但需看残差图",
                    attribution=attr + [f"中段残差超阈但非系统性：{sys_txt}"])
        return base
    if end_bad or not mu_ok:
        base.update(status="warn",
                    reason=f"中段残差 {mid:.1%} ≤ {res_tol:.1%} 正常，"
                           f"但端部/收敛性需归因（见 attribution）→ 可拟合，"
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
        lines.append(f"    μ={g['mu']:.3f}  M={g['M']}  "
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
