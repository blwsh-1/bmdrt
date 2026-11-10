# -*- coding: utf-8 -*-
"""test_ad_gate_synth.py —— M6 合成谱门控矩阵（独立生成器，不调用被测试模型）。

为什么单独一套
--------------
`test_ad_gate.py` 用 `bm_ad.z_ad` 造吸附谱，属于"模型自己生成数据、自己识别"
的循环验证。本文件按审查意见重写：**所有真值用独立解析式手写**，不 import
`bm_ad` / `bm_core` 的任何元件；只复用生产的统计函数（`resid_bins` / `acf_max`）
与被测门控 `detect_ad`。

三层报告（分开统计，不混算）
--------------------------
  A. 状态识别准确性 —— S1–S9 + C1–C3 逐条比对预期状态；
  B. 门控不变性 —— 同一批谱，新规则与旧规则（`k_left>=2`）的 strong 集合是否一致；
  C. 鲁棒性 —— 幅值 / 峰位 / HN 重叠 / 噪声 / 漂移 / 采样密度 / 非均匀采样扫描。

**模拟假阳性率只代表给定模拟分布下的值，不能替代真实对照谱的统计上界。**

用法：python test_ad_gate_synth.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import bm_real as R                      # noqa: E402
from bm_experiments import acf_max       # noqa: E402

R_INF = 10.0
ARC_TRUE = [(30.0, 300.0, 0.85), (8.0, 3.0e4, 0.90)]   # none 家族真值（ZARC）


# --------------------------------------------------------------------------
# 独立解析元件（手写，**不** import bm_ad / bm_core）
# --------------------------------------------------------------------------

def z_rl(f, R_ad, f_c):
    """R∥L：Z = R·jωτ/(1+jωτ)，τ = 1/(2π f_c)。"""
    x = np.asarray(f, float) / f_c
    return R_ad * (1j * x) / (1.0 + 1j * x)


def z_zarc(f, R, f_c, phi):
    """ZARC：Z = R/[1+(jωτ)^φ]。"""
    x = np.asarray(f, float) / f_c
    return R / (1.0 + (1j * x) ** phi)


def z_cpe(f, Y, n):
    """纯 CPE：Z = 1/(Y·(jω)^n)。"""
    return 1.0 / (Y * (2j * np.pi * np.asarray(f, float)) ** n)


def z_series_l(f, L):
    return 1j * 2 * np.pi * np.asarray(f, float) * L


def z_ws(f, A_W, f_D=None):
    """扩散尾（独立解析式，**不** import bm_ws / bm_diff）。

    半无限 Warburg：Z = A_W / √(jω)，幅值 A_W/√ω、相位 −45°；
    给 f_D 时乘 tanh(√(jωτ_D)) 得有限长（FLW），τ_D = 1/(2π f_D)。
    A_W 与 `bm_ws` 的 `A_W` 同量纲（Ω·s^-1/2），只用于构造重叠形态。
    """
    w = 2 * np.pi * np.asarray(f, float)
    s = np.sqrt(1j * w)
    z = A_W / s
    if f_D is not None:
        z = z * np.tanh(s / np.sqrt(2 * np.pi * f_D))
    return z


def z_none(f, arcs=ARC_TRUE, r_inf=R_INF):
    z = np.full(np.shape(f), r_inf, complex)
    for rad, fc, phi in arcs:
        z = z + z_zarc(f, rad, fc, phi)
    return z


def drift_ramp(f, amp):
    """低频单调漂移：最低 40% 频段内 Im 从 0 线性爬到 amp（对数频率）。"""
    lf = np.log10(f)
    m = lf <= np.quantile(lf, 0.40)
    r = np.zeros(f.size)
    if m.sum() > 1 and np.ptp(lf[m]) > 0:
        r[m] = amp * (lf[m] - lf[m].min()) / np.ptp(lf[m])
    return 1j * r


def make_spectrum(f, rl=None, arcs=ARC_TRUE, drift=0.0, pseudo_l=0.0,
                  noise=0.0, seed=0, ws=None):
    """合成观测：none 真值 + 可选 RL / 扩散尾 / 漂移 / 伪感抗 + 噪声。

    rl = (R_ad, f_c) 用独立解析式生成，**不经** bm_ad。
    ws = (A_W, f_D) 半无限（f_D=None）或有限长扩散尾，**不经** bm_ws / bm_diff。
    pseudo_l：宽频常数虚部（仪器伪感抗的简化），单位 Ω。
    """
    z = z_none(f, arcs)
    if rl is not None:
        z = z + z_rl(f, rl[0], rl[1])
    if ws is not None:
        z = z + z_ws(f, ws[0], ws[1] if len(ws) > 1 else None)
    if drift:
        z = z + drift_ramp(f, drift)
    if pseudo_l:
        z = z + 1j * pseudo_l
    if noise:
        rng = np.random.default_rng(seed)
        z = z + noise * np.abs(z).max() * (rng.standard_normal(f.size)
                                           + 1j * rng.standard_normal(f.size))
    return z


def fit_none(f, Zobs, n_start=6, seed=11):
    """用 scipy LS 拟合 none 家族（R_inf + 2 ZARC），返回 Z_none。

    比 NUTS 便宜得多，且这就是"none 模型能给出的最好拟合"，用来生成
    与生产链路同形的残差统计（resid_bins / acf）。
    """
    w = 2 * np.pi * f
    lo = np.array([np.log(1.0), np.log(1.0), np.log(1e-5), 0.5,
                   np.log(1.0), np.log(1e-5), 0.5])
    hi = np.array([np.log(1e3), np.log(1e3), np.log(1e2), 1.05,
                   np.log(1e3), np.log(1e2), 1.05])

    def model(p):
        z = np.exp(p[0])
        for k in range(2):
            z = z + np.exp(p[1 + 3 * k]) / (1.0 + (1j * w * np.exp(p[2 + 3 * k]))
                                            ** p[3 + 3 * k])
        return z

    wabs = np.abs(Zobs)

    def resid(p):
        d = (model(p) - Zobs) / wabs
        return np.concatenate([d.real, d.imag])

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_start):
        p0 = lo + (hi - lo) * rng.random(7)
        try:
            r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=1200)
            c = float(np.sqrt(np.mean(r.fun ** 2)))
            if best is None or c < best[0]:
                best = (c, model(r.x))
        except Exception:                      # noqa: BLE001
            continue
    if best is None:
        return z_none(f), 1.0
    return best[1], best[0]


def res_of(Zobs, Znone, f):
    resid = Zobs - Znone
    return dict(bins=R.resid_bins(resid, f),
                acf=max(acf_max(resid.real), acf_max(resid.imag)),
                s_re=float(np.std(resid.real)), s_im=float(np.std(resid.imag)),
                f_lo=float(f.min()), f_hi=float(f.max()))


def gate(f, Zobs, **kw):
    Znone, cost = fit_none(f, Zobs, **kw)
    return R.detect_ad(f, Zobs, res_of(Zobs, Znone, f))


def old_rule(g):
    """旧规则：只看 interior（k_left>=2 且 k_right>=2），其余一律 enter=False。"""
    c = g.get("checks") or {}
    k_l, k_r = c.get("k_left"), c.get("k_right")
    if k_l is None or k_r is None:
        return False
    return bool(k_l >= 2 and k_r >= 2 and c.get("half_circle"))


# --------------------------------------------------------------------------
# 频率网格
# --------------------------------------------------------------------------

def grid(n=60, fmin=1e-2, fmax=1e5, jitter=0.0, seed=5):
    f = np.logspace(np.log10(fmin), np.log10(fmax), n)
    if jitter:
        rng = np.random.default_rng(seed)
        f = f * (1.0 + jitter * rng.uniform(-1, 1, n))
        f = np.sort(f)
    return f


# 采样密度口径与生产一致：dens = n / log10(f_max/f_min)（见 edges()）。
# 真实 AZ31：50 点 / 7 dec = 7.14；真实 2E：56 点 / 5 dec = 11.2。
F60 = grid()                     # 8.57 pts/dec（默认）
F_AZ = grid(n=50)                # 7.14 pts/dec，匹配 AZ31 的平均密度
F_2E = grid(n=78)                # 11.14 pts/dec，匹配 2E 的平均密度
FJ = grid(n=60, jitter=0.15)     # 8.57 pts/dec + 15% 非均匀


def band_pts(f, frac=0.40):
    lf = np.log10(f)
    m = lf <= np.quantile(lf, frac)
    return f[m]


def peak_f(f, Z):
    lf = np.log10(f)
    m = lf <= np.quantile(lf, 0.40)
    return float(f[m][int(np.argmax(Z[m].imag))])


# --------------------------------------------------------------------------
# A. 确定性用例（预期写死，不让输出反过来定义答案）
# --------------------------------------------------------------------------

def _chk(name, f, Zobs, exp_status, exp_enter, exp_strong):
    g = gate(f, Zobs)
    ok = (g["status"] == exp_status and bool(g["enter"]) == exp_enter
          and bool(g["strong"]) == exp_strong)
    c = g["checks"]
    detail = (f"{g['status']} enter={g['enter']} strong={g['strong']} "
              f"k_left={c.get('k_left')} k_right={c.get('k_right')}")
    return ok, f"{name}: 期望 {exp_status}/{exp_enter}/{exp_strong} → {detail}"


def kleft_of(f, Z):
    """独立计算峰左侧点数（与 detect_ad 同口径），用于验证构造确实贴边。"""
    lf = np.log10(f)
    m = lf <= np.quantile(lf, 0.40)
    fb, zb = f[m], Z[m]
    k = int(np.argmax(zb.imag))
    return int(np.count_nonzero(fb < fb[k])), float(fb[k])


def cases():
    """返回 [(ok, msg)]；S1–S9 + C1–C3。"""
    out = []
    # S1 完整 RL 弧（窗内、幅值足够撑起半圆）
    out.append(_chk("S1 完整RL弧", F60, make_spectrum(F60, rl=(100.0, 1.0)),
                    "strong_candidate", True, True))
    # S2 有感抗特征但半圆不完整（只翘不复位）。
    # 注意：真 RL 元件的 ΔRe/(2|Im|) 恒 ≈1，靠改幅值造不出 partial_arc；
    # 这里用纯虚部鼓包模拟"被遮蔽/伪感抗"导致的 Re 不复位。
    bump_mid = 1j * 100.0 * np.exp(-0.5 * (np.log10(F60 / 1.0) / 0.12) ** 2)
    out.append(_chk("S2 半圆不完整", F60, make_spectrum(F60) + bump_mid,
                    "partial_arc", True, False))
    # S3 峰贴低频边界（k_left=1）
    out.append(_chk("S3 贴边峰", F60, make_spectrum(F60, rl=(100.0, float(F60[1]))),
                    "edge_peak", True, False))
    # S4 峰在最低频点（k_left=0）
    out.append(_chk("S4 峰在最低点", F60, make_spectrum(F60, rl=(100.0, float(F60[0]))),
                    "monotonic_rise", False, False))
    # S5 峰贴 band 上沿（k_right<2）
    fb = band_pts(F60)
    out.append(_chk("S5 峰贴上沿", F60, make_spectrum(F60, rl=(100.0, float(fb[-1]))),
                    "monotonic_rise", False, False))
    # S6 纯容性
    out.append(_chk("S6 纯容性", F60, make_spectrum(F60), "capacitive_tail", False, False))
    # S7 正向偏移但噪声主导
    out.append(_chk("S7 噪声主导", F60,
                    make_spectrum(F60, pseudo_l=50.0, noise=0.5, seed=3),
                    "noise_only", False, False))
    # S8 只有 HN/ZARC，无 RL（小噪声下符号可能随机，capacitive_tail 或
    # noise_only 都算合格——验收点只有"不自动进入吸附比较"）
    g8 = gate(F60, make_spectrum(F60, noise=0.002, seed=4))
    out.append(((not g8["enter"]) and (not g8["strong"]),
                f"S8 仅HN无RL: {g8['status']} enter={g8['enter']} strong={g8['strong']}"))
    # S9 非吸附过程 + 低频漂移 + 仪器伪感抗（最危险的假阳性形态）
    Z9 = make_spectrum(F60, drift=120.0, pseudo_l=40.0, noise=0.005, seed=6)
    g9 = gate(F60, Z9)
    ok9 = (not g9["strong"]) and (g9["status"] in
                                  ("monotonic_rise", "edge_peak", "noise_only",
                                   "partial_arc", "capacitive_tail"))
    out.append((ok9, f"S9 漂移+伪感抗: {g9['status']} enter={g9['enter']} "
                     f"strong={g9['strong']}（只要求不自动选）"))

    # ---- C1–C3：贴边 + 两道前置闸门的组合。断言必须落到 checks 的底层字段，
    #      不能只看最终 status —— 否则"符号闸门失败"也会被 capacitive_tail 放行。
    f1 = float(F60[1])

    # 贴边峰：窄高斯（真 RL 峰较宽，argmax 会偏到 band 内部，造不出 k_left=1）
    bump1 = 1j * 100.0 * np.exp(-0.5 * (np.log10(F60 / f1) / 0.12) ** 2)

    # C1：k_left=1，但符号闸门不通过。
    # 构造：整段低频偏负（-60 Ω），只在 f1 处抬到 +30 —— 峰位在 f1（k_left=1），
    # 但最低频段残差均值仍为负 → 必须在符号关返回。
    Zc1 = make_spectrum(F60) - 1j * 60.0 + 0.9 * bump1
    g1 = gate(F60, Zc1)
    c1 = g1["checks"]
    kl1, _ = kleft_of(F60, Zc1)
    ok1 = (kl1 == 1
           and c1.get("sign_inductive") is False
           and g1["status"] == "capacitive_tail"
           and g1["enter"] is False and g1["strong"] is False
           and c1.get("k_left") is None)          # 未进入形状闸
    out.append((ok1, f"C1 贴边+符号不过: 构造k_left={kl1} "
                     f"sign={c1.get('sign_inductive')} → {g1['status']} "
                     f"enter={g1['enter']} k_left(checks)={c1.get('k_left')}"))

    # C2：k_left=1，符号通过、系统性不通过 → 必须在系统性关返回（关键用例）
    # 构造：整段低频正偏（+20 Ω）+ f1 处尖峰（+120），再加 30% 噪声压垮系统性。
    Zc2 = make_spectrum(F60) + 1j * 20.0 + bump1
    rng2 = np.random.default_rng(7)
    Zc2 = Zc2 + 0.30 * np.abs(Zc2).max() * (rng2.standard_normal(F60.size)
                                            + 1j * rng2.standard_normal(F60.size))
    g2 = gate(F60, Zc2)
    c2 = g2["checks"]
    kl2, _ = kleft_of(F60, Zc2)
    ok2 = (kl2 == 1
           and c2.get("sign_inductive") is True
           and c2.get("systematic") is False
           and g2["status"] == "noise_only"
           and g2["enter"] is False and g2["strong"] is False
           and c2.get("k_left") is None)          # 未进入形状闸
    out.append((ok2, f"C2 贴边+系统性不过: 构造k_left={kl2} "
                     f"sign={c2.get('sign_inductive')} "
                     f"systematic={c2.get('systematic')} → {g2['status']} "
                     f"enter={g2['enter']} k_left(checks)={c2.get('k_left')}"))

    # C3：k_left=1，两关都过，但背景不是 RL（窄虚部鼓包）
    #     → 进 edge_peak（enter=True），但不得自动选择（strong=False）
    Zc3 = make_spectrum(F60) + bump1
    g3 = gate(F60, Zc3)
    c3 = g3["checks"]
    kl3, _ = kleft_of(F60, Zc3)
    ok3 = (kl3 == 1
           and c3.get("sign_inductive") is True
           and c3.get("systematic") is True
           and c3.get("k_left") == 1
           and g3["status"] == "edge_peak"
           and g3["enter"] is True and g3["strong"] is False)
    out.append((ok3, f"C3 非RL贴边+两关过: 构造k_left={kl3} "
                     f"sign={c3.get('sign_inductive')} "
                     f"systematic={c3.get('systematic')} → {g3['status']} "
                     f"enter={g3['enter']} strong={g3['strong']}"))
    return out


# --------------------------------------------------------------------------
# B. 不变性：新规则 vs 旧规则（k_left>=2）
# --------------------------------------------------------------------------

def invariance():
    specs = []
    for rad in (2.0, 10.0, 100.0):
        for fc in (0.1, 1.0, 10.0):
            specs.append((F60, make_spectrum(F60, rl=(rad, fc))))
    for fc in (float(F60[0]), float(F60[1]), float(F60[2])):
        specs.append((F60, make_spectrum(F60, rl=(100.0, fc))))
    specs.append((F60, make_spectrum(F60)))
    specs.append((F60, make_spectrum(F60, noise=0.02, seed=9)))

    diffs = []
    for f, Z in specs:
        g = gate(f, Z)
        if bool(g["strong"]) != old_rule(g):
            diffs.append(f"f_peak={peak_f(f, Z):.3g}: 新 {g['strong']} vs 旧 {old_rule(g)}")
    return diffs, len(specs)


# --------------------------------------------------------------------------
# C. 鲁棒性扫描
# --------------------------------------------------------------------------

def sweep():
    """每条扫描带**显式预期**；只有 strict=True 的条目计入验收。

    峰落在 band 外（f_c < f_min·1.2 或 > band 上沿·0.8）时预期
    `monotonic_rise`（不可辨识区间，必须不建环）；峰在 band 内且幅值足够
    时预期 `strong_candidate`；弱幅值（R<10）只作敏感性观察，不做严格断言。
    """
    rows = []

    def rec(group, param, g, exp_enter=None, exp_strong=None,
            exp_status=None, strict=True):
        c = g["checks"]
        ok = True
        if strict:
            ok = (bool(g["enter"]) == exp_enter
                  and bool(g["strong"]) == exp_strong
                  and (exp_status is None or g["status"] in exp_status))
        rows.append(dict(group=group, param=param, status=g["status"],
                         enter=bool(g["enter"]), strong=bool(g["strong"]),
                         k_left=c.get("k_left"), k_right=c.get("k_right"),
                         ok=ok, strict=strict,
                         exp=(exp_enter, exp_strong, exp_status)))

    fb = band_pts(F60)
    for rad in (0.5, 2.0, 10.0, 50.0, 200.0):
        for fc in (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0):
            g = gate(F60, make_spectrum(F60, rl=(rad, fc)))
            tag = f"R={rad:g},fc={fc:g}"
            if fc < fb[0] * 1.2 or fc > fb[-1] * 0.8:
                rec("幅值×峰位(峰出窗)", tag, g, False, False,
                    {"monotonic_rise", "capacitive_tail", "noise_only"})
            elif rad >= 10.0:
                rec("幅值×峰位(峰在窗内)", tag, g, True, True,
                    {"strong_candidate"})
            else:
                rec("幅值×峰位(弱幅值观察)", tag, g, strict=False)

    i1 = int(np.argmin(np.abs(F60 - 1.0)))
    for hnr in (0.0, 0.5, 1.0, 2.0, 4.0):
        arcs = [(30.0 * (1 + hnr), 1.0, 0.85), (8.0, 3e4, 0.90)]
        Z = make_spectrum(F60, rl=(100.0, 1.0), arcs=arcs)
        net = float(z_rl(F60[i1:i1 + 1], 100.0, 1.0).imag[0]
                    + z_zarc(F60[i1:i1 + 1], 30.0 * (1 + hnr), 1.0, 0.85).imag[0])
        g = gate(F60, Z)
        tag = f"arc×{1 + hnr:.1f}(净Im={net:+.1f})"
        if net > 5.0:
            rec("HN重叠(可辨识)", tag, g, True, True, {"strong_candidate"})
        else:
            # 净虚部 <5 Ω：RL 被抵消到不可观测，正确行为是**不建环**，
            # 这是物理不可判定，不是门控缺陷。
            rec("HN重叠(抵消至不可见)", tag, g, False, False,
                {"monotonic_rise", "capacitive_tail", "noise_only"})

    for nz in (0.001, 0.01, 0.05, 0.2):
        g1 = gate(F60, make_spectrum(F60, rl=(100.0, 1.0), noise=nz, seed=12))
        if nz <= 0.01:
            rec("噪声(有RL)", f"{nz:.1%}", g1, True, True, {"strong_candidate"})
        elif nz < 0.2:
            # 5%：系统性判据在 low_rms/σ≈1.5 附近，允许两种结果
            rec("噪声(有RL,边界)", f"{nz:.1%}", g1, strict=False)
        else:
            rec("噪声(有RL)", f"{nz:.1%}", g1, False, False,
                {"noise_only", "capacitive_tail"})
        g0 = gate(F60, make_spectrum(F60, noise=nz, seed=12))
        rec("噪声(无RL)", f"{nz:.1%}", g0, False, False,
            {"capacitive_tail", "noise_only"})

    for dr in (0.0, 20.0, 120.0, 400.0):
        g = gate(F60, make_spectrum(F60, drift=dr))
        rec("低频漂移", f"{dr:g}", g, False, False,
            {"capacitive_tail", "noise_only", "monotonic_rise"})

    for tag, fg in (("7.1 pts/dec", F_AZ), ("11.2 pts/dec", F_2E),
                    ("15% 非均匀", FJ)):
        rec("采样", tag + "(有RL)",
            gate(fg, make_spectrum(fg, rl=(100.0, 1.0))), True, True,
            {"strong_candidate"})
        rec("采样", tag + "(无RL)", gate(fg, make_spectrum(fg)), False, False,
            {"capacitive_tail", "noise_only"})
    return rows


# --------------------------------------------------------------------------
# D. RL + 扩散尾重叠（M6 §4.2 M6.2b / E5）：**只统计，不设通过线**
# --------------------------------------------------------------------------

E5_CSV = ROOT / "out/m6/e5/diffusion_overlap.csv"


def diffusion_overlap():
    """RL 峰与 Warburg / FLW 尾共存时的门控行为，只记计数。

    重叠比 r = Im(RL @ f_c) / |Im(Ws @ f_c)|（用解析式算，不经拟合）：

      · 峰不在 band 内（f_c < 1.2·band下沿 或 > 0.8·band上沿）→ `峰出窗`，
        门控本来就不该建环，**不计假阴性**；
      · r ≥ 3   峰主导   → `enter=False` 记**假阴性**（§4.2 判据：必须 0 条）；
      · r ≤ 1/3 扩散主导 → `strong=True` 记**误建环**；
      · 其余    混合区   → 只计数，不判定。

    `A_W=0` 行是**对照**（RL 无扩散）：峰在窗内时应 `strong=True`，
    不成立记为 `对照失效`——这是生成器/门控的回归信号，不是扩散重叠的结论。
    第一次跑没有基线，故本组不进退出码，只打印计数。
    """
    rows = []
    fb = band_pts(F_AZ)
    specs = [("半无限Ws", None), ("FLW f_D=1Hz", 1.0), ("FLW f_D=1kHz", 1.0e3)]
    for typ, fd in specs:
        for rad in (30.0, 100.0):
            for fc in (0.03, 0.1, 0.5, 1.0, 3.0, 10.0):
                for aw in (0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
                    Z = make_spectrum(F_AZ, rl=(rad, fc), ws=(aw, fd))
                    g = gate(F_AZ, Z)
                    c = g["checks"]
                    in_band = fb[0] * 1.2 <= fc <= fb[-1] * 0.8
                    i_pk = int(np.argmin(np.abs(F_AZ - fc)))
                    zrl = z_rl(F_AZ[i_pk:i_pk + 1], rad, fc)[0]
                    if aw <= 0:
                        ratio = float("nan")
                        regime = "对照(无扩散)"
                    elif not in_band:
                        ratio = float("nan")
                        regime = "峰出窗"
                    else:
                        zws = z_ws(F_AZ[i_pk:i_pk + 1], aw, fd)[0]
                        ratio = float(zrl.imag / max(abs(zws.imag), 1e-300))
                        regime = ("峰主导" if ratio >= 3.0 else
                                  "扩散主导" if ratio <= 1.0 / 3.0 else "混合")
                    fn = regime == "峰主导" and not g["enter"]
                    fp = regime == "扩散主导" and bool(g["strong"])
                    ctl_fail = (regime == "对照(无扩散)" and in_band
                                and not g["strong"])
                    rows.append(dict(
                        type=typ, R_ad=rad, f_c=fc, A_W=aw,
                        peak_in_band=str(in_band).lower(),
                        ratio=("" if not np.isfinite(ratio) else round(ratio, 3)),
                        regime=regime, status=g["status"],
                        enter=bool(g["enter"]), strong=bool(g["strong"]),
                        k_left=c.get("k_left"), k_right=c.get("k_right"),
                        fn=str(fn).lower(), fp=str(fp).lower(),
                        ctl_fail=str(ctl_fail).lower()))
    return rows


def _dump_csv(path: Path, rows: list[dict]) -> None:
    import csv as _csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("A. 状态识别准确性（独立解析生成器，不调用 bm_ad）")
    print("=" * 100)
    res = cases()
    bad = 0
    for ok, msg in res:
        print(("[PASS] " if ok else "[FAIL] ") + msg)
        bad += 0 if ok else 1

    print("\n" + "=" * 100)
    print("B. 门控不变性：新规则 vs 旧规则（strong 集合）")
    print("=" * 100)
    diffs, n = invariance()
    print(f"  {n} 条合成谱，strong 判定差异 {len(diffs)} 条")
    for d in diffs:
        print("   ", d)

    print("\n" + "=" * 100)
    print("C. 鲁棒性扫描（逐条预期断言；仅给定模拟分布，不可替代真实统计上界）")
    print("=" * 100)
    rows = sweep()
    from collections import Counter
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    for gname, grp in groups.items():
        c = Counter(r["status"] for r in grp)
        ent = sum(1 for r in grp if r["enter"])
        sg = sum(1 for r in grp if r["strong"])
        st = sum(1 for r in grp if r["strict"])
        print(f"  {gname:<22} n={len(grp):3d} strict={st:3d} enter={ent:2d} "
              f"strong={sg:2d} 状态={dict(c)}")

    strict = [r for r in rows if r["strict"]]
    bad_c = [r for r in strict if not r["ok"]]
    obs = [r for r in rows if not r["strict"]]
    print(f"\n  严格断言 {len(strict)} 条，失败 {len(bad_c)} 条；"
          f"观察项 {len(obs)} 条（弱幅值/噪声边界，不做断言）")
    for r in bad_c:
        print(f"    [FAIL] [{r['group']}] {r['param']}: {r['status']} "
              f"enter={r['enter']} strong={r['strong']} "
              f"k_left={r['k_left']} k_right={r['k_right']} 预期={r['exp']}")
    for r in obs:
        print(f"    [观察] [{r['group']}] {r['param']}: {r['status']} "
              f"enter={r['enter']} strong={r['strong']}")

    d = diffusion_overlap()
    _dump_csv(E5_CSV, d)
    from collections import Counter as _C
    print(f"  n={len(d)} 条（3 类扩散尾 × 2 R_ad × 6 f_c × 8 A_W），"
          f"明细 {E5_CSV.relative_to(ROOT)}")
    for reg in ("对照(无扩散)", "峰出窗", "扩散主导", "混合", "峰主导"):
        sub = [r for r in d if r["regime"] == reg]
        if not sub:
            continue
        ent = sum(1 for r in sub if r["enter"])
        sg = sum(1 for r in sub if r["strong"])
        print(f"  {reg:<12} n={len(sub):3d} enter={ent:2d} strong={sg:2d} "
              f"状态={dict(_C(r['status'] for r in sub))}")
    fn = [r for r in d if r["fn"] == "true"]
    fp = [r for r in d if r["fp"] == "true"]
    ctl = [r for r in d if r["ctl_fail"] == "true"]
    print(f"\n  假阴性（峰主导却 enter=False）{len(fn)} 条"
          f"{' → 全部列在下面' if fn else ''}")
    for r in fn:
        print(f"    [FN] {r['type']} R={r['R_ad']:g} f_c={r['f_c']:g} "
              f"A_W={r['A_W']:g} r={r['ratio']} → {r['status']} "
              f"k_left={r['k_left']} k_right={r['k_right']}")
    print(f"  误建环（扩散主导却 strong=True）{len(fp)} 条")
    for r in fp:
        print(f"    [FP] {r['type']} R={r['R_ad']:g} f_c={r['f_c']:g} "
              f"A_W={r['A_W']:g} r={r['ratio']} → {r['status']}")
    print(f"  对照失效（A_W=0、峰在窗内却未 strong）{len(ctl)} 条")
    for r in ctl:
        print(f"    [CTL] {r['type']} R={r['R_ad']:g} f_c={r['f_c']:g} "
              f"→ {r['status']}")

    print("\n" + "=" * 100)
    print(f"A 结果：{len(res) - bad}/{len(res)} 通过；B 差异 {len(diffs)}；"
          f"C 严格断言 {len(strict) - len(bad_c)}/{len(strict)} 通过")
    print(f"D（E5 扩散重叠）：{len(d)} 条，假阴性 {len(fn)} 条，误建环 {len(fp)} 条；"
          f"第一次跑无基线，不设通过线，不进退出码。")
    print("注意：C/D 的假阳性率只代表给定模拟分布，不能替代真实对照谱统计上界。")
    raise SystemExit(1 if (bad or diffs or bad_c) else 0)


if __name__ == "__main__":
    main()
