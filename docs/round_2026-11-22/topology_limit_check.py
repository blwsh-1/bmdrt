"""拓扑极限检查：每个族/接点的 `ω→0` 与 `ω→∞` 极限，与解析推导比对。

**为什么需要它**（P-8 的可执行部分）

`adr` 的复盘发现：立项时没人写"低频极限是什么"，于是
`Z_ad = R_ad·jωτ/(1+jωτ)` 在 ω→0 时 → 0（不是 → `R_ad`）这件事，
让 `adr` 的直流阻抗塌到 `Rs`（吸附支路把 CPE 和 Rct 一起短路），
两周后才被算出来。若立项时被迫填"低频极限"，当场就会暴露。

**检查什么**

阻抗在两端都是实数（DC 无电容电流；HF 电容短路，若带 L 则虚部发散）：

    Re Z(ω→0)   低频平台  = 直流总电阻
    Re Z(ω→∞)   高频截距  = 溶液/引线电阻
    Im Z(ω→∞)/ω 带 L 时    = 顶层电感

只查实部是刻意的：虚部在两端都趋于 0 或发散，比对没有信息量。

**怎么算"实测"**

在 `f = 1e-8 … 1e10 Hz` 的宽窗上正演，取两端。`z_cpe` 的
`w_ref = sqrt(f.min()·f.max())` 是窗内几何均值，所以解析期望**必须用同一个
`w_ref`** 算，否则两边不同尺（这是上一轮 `dc_limit_check.py` 踩过的口径）。

**怎么判**

`|Re Z_measured - Re Z_expected| / max(1, |Z_expected|) <= TOL`。
TOL 用 1e-6：两端都是渐近值，1e-8 / 1e10 Hz 已足够远离任何拐点。

用法：
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-22/topology_limit_check.py
"""

import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import bm_core as C            # noqa: E402
import bm_top as T             # noqa: E402

TOL = 1e-6
F_LO, F_HI = 1e-8, 1e10
N_PT = 241

#: 统一参数点。Q 是**窗口幅值**（|Z(ω_ref)|，单位 Ω），不是 F。
BASE = dict(Rs=50.0, R1=100.0, R2=80.0,
            Q1=100.0, n1=0.85, Q2=100.0, n2=0.90,
            R_ad=200.0, tau_ad=1e-2,
            R_L=100.0, L_ad=1e-2,
            Q_b=100.0, n_b=0.90)
VOIGT_P = dict(R_inf=50.0, W=100.0, tau=1e-3, phi=0.9, zeta=1.0)
L_TOP = 1e-5


def _freq():
    f = np.logspace(np.log10(F_LO), np.log10(F_HI), N_PT)
    return f, 2.0 * np.pi * float(np.sqrt(f.min() * f.max()))


def _voigt(f, flat, include_L):
    p = dict(flat)
    p.update(pi=np.array([1.0]), tau=np.array([flat["tau"]]),
             phi=np.array([flat["phi"]]), zeta=np.array([flat["zeta"]]))
    if include_L:
        p["L"] = L_TOP
    return C.model_z(f, p, include_L=include_L)


def _nested(f, p, branch, layers, include_L):
    q = dict(p)
    if include_L:
        q["L"] = L_TOP
    return T.z_nested(f, q, branch=branch, layers=layers, include_L=include_L)


def _adr(f, p, include_L):
    q = dict(p, Q=p["Q1"], n=p["n1"], Rct=p["R1"])
    if include_L:
        q["L"] = L_TOP
    return T.z_adr(f, q, include_L=include_L)


# ---------------------------------------------------------------------------
# 解析期望。每个条目写清**推导**，因为这就是 P-8 要求填的那一栏。
#   Z_ad = R_ad·jωτ/(1+jωτ)      ω→0 → 0        ω→∞ → R_ad
#   Z_CPE = Q·(ω_ref/ω)^n·e^{-jnπ/2}   ω→0 → ∞    ω→∞ → 0
#   Z_series = R_L + jωL         ω→0 → R_L      ω→∞ → 发散(Im)
#   Z_HN = W/(1+(jωτ)^φ)^ζ       ω→0 → W        ω→∞ → 0
# ---------------------------------------------------------------------------

def cases():
    """(key, 正演, DC 期望实部, HF 期望实部, 说明, 是否带 L)"""
    out = []

    out.append((
        "voigt_K1", lambda f: _voigt(f, VOIGT_P, False),
        VOIGT_P["R_inf"] + VOIGT_P["W"], VOIGT_P["R_inf"],
        "R_inf + W / R_inf（HN 两端收敛）", False))

    out.append((
        "voigt_K1_L", lambda f: _voigt(f, VOIGT_P, True),
        VOIGT_P["R_inf"] + VOIGT_P["W"], VOIGT_P["R_inf"],
        "同上 + jωL", True))

    out.append((
        "nested_L1_parallel",
        lambda f: _nested(f, BASE, "parallel", 1, False),
        BASE["Rs"] + BASE["R1"], BASE["Rs"],
        "Z_ad→0 ⇒ Rs+R1；CPE→0 ⇒ Rs", False))

    out.append((
        "nested_L1_series",
        lambda f: _nested(f, BASE, "series", 1, False),
        BASE["Rs"] + BASE["R1"] + BASE["R_L"], BASE["Rs"],
        "串联支路不通直流 ⇒ Rs+R1+R_L", False))

    # **DC 期望是无穷大**：内层 CPE 阻断直流 ⇒ 外层 CPE 与它并联后
    # Y → c1·ω^n1 + c2·ω^n_b → 0 ⇒ Z → ∞。首版误写 Rs+R1，被 1e-8 Hz 的
    # 7e8 打回（那是代数收敛没到极限，不是实现错）。
    out.append((
        "nested_L1_capacitive",
        lambda f: _nested(f, BASE, "capacitive", 1, False),
        math.inf, BASE["Rs"],
        "内层 CPE 阻断直流 ⇒ Z→∞（代数发散）；CPE→0 ⇒ Rs", False))

    out.append((
        "nested_L2_parallel",
        lambda f: _nested(f, BASE, "parallel", 2, False),
        BASE["Rs"] + BASE["R1"] + BASE["R2"], BASE["Rs"],
        "Z_ad→0 ⇒ Rs+R1+R2", False))

    out.append((
        "nested_L1_parallel_L",
        lambda f: _nested(f, BASE, "parallel", 1, True),
        BASE["Rs"] + BASE["R1"], BASE["Rs"],
        "同上 + 顶层 jωL", True))

    # ---- 关键对照：adr 的低频不是 Rs+Rct，而是 Rs（Z_ad 短路整个节点）
    out.append((
        "adr", lambda f: _adr(f, BASE, False),
        BASE["Rs"], BASE["Rs"],
        "**Z_ad→0 短路 CPE 与 Rct ⇒ Re Z→Rs**", False))

    out.append((
        "adr_L", lambda f: _adr(f, BASE, True),
        BASE["Rs"], BASE["Rs"],
        "同上 + 顶层 jωL", True))

    return out


def main() -> int:
    f, w_ref = _freq()
    print("频率窗 %.0e … %.0e Hz（%d 点），w_ref = %.4g rad/s"
          % (f.min(), f.max(), f.size, w_ref))
    print("TOL = %.0e（相对 max(1, |expect|)）\n" % TOL)

    hdr = ("%-22s %12s %12s %10s   %s"
           % ("case", "Re@DC 实测/期望", "Re@HF 实测/期望", "状态", "推导"))
    print(hdr)
    print("-" * 118)

    rows, bad = [], []
    for key, fwd, dc_exp, hf_exp, why, has_L in cases():
        Z = np.asarray(fwd(f), complex)
        dc, hf = float(Z[0].real), float(Z[-1].real)

        def ok(got, exp):
            if math.isinf(exp):
                return None                      # 由下面的发散检查单独判
            return abs(got - exp) / max(1.0, abs(exp)) <= TOL

        ok_dc, ok_hf = ok(dc, dc_exp), ok(hf, hf_exp)
        dc_note = ""
        if ok_dc is None:
            # 期望 ∞：再取一个更低的 f_lo，要求 Re Z 至少涨 100 倍（6 dec 窗，
            # 幂律 ω^-n 下应涨 ~1e5）——判"方向"而不判绝对值。
            f2 = np.logspace(np.log10(F_LO / 1e6), np.log10(F_HI), N_PT)
            z2 = float(np.asarray(fwd(f2), complex)[0].real)
            ok_dc = (z2 > 100.0 * dc) and (z2 > 1e6)
            dc_note = "   [∞: f_lo/1e6 处 %.3g，比值 %.0fx]" % (z2, z2 / max(dc, 1e-30))
        # 带 L 的再查 Im/ω 是否等于 L（顶层电感的高频指纹）
        slope = None
        if has_L:
            w_hi = 2.0 * np.pi * float(f[-1])
            slope = float(Z[-1].imag) / w_hi
        slope_ok = (slope is None
                    or abs(slope - L_TOP) / L_TOP <= 1e-6)
        good = ok_dc and ok_hf and slope_ok
        if not good:
            bad.append(key)

        dc_txt = "inf" if math.isinf(dc_exp) else "%.4f" % dc_exp
        print("%-22s %12.4g %12.4f %10s   %s"
              % (key, dc, hf, "ok" if good else "**MISMATCH**", why))
        print("%-22s %12s %12.4f%s%s"
              % ("", dc_txt, hf_exp,
                 ("" if slope is None
                  else "   Im/ω=%.4g（期望 %.4g）" % (slope, L_TOP)),
                 dc_note))
        rows.append(dict(key=key, dc=dc,
                         dc_expected=(None if math.isinf(dc_exp) else dc_exp),
                         dc_is_inf=math.isinf(dc_exp),
                         hf=hf, hf_expected=hf_exp,
                         slope=slope, slope_expected=(L_TOP if has_L else None),
                         ok=good, why=why))

    print("-" * 118)
    print("\n**低频平台一览**（这是 P-8 要填的那一栏）")
    print("%-22s %12s %s" % ("case", "Re Z(ω→0)", "读法"))
    for r in rows:
        note = ""
        if r["key"].startswith("adr"):
            note = "← 塌到 Rs：吸附支路短路了 Rct"
        dv = "inf（CPE 阻断直流）" if r["dc_is_inf"] else "%.4f" % r["dc_expected"]
        print("%-22s %12s %s" % (r["key"], dv, note))

    print("\n%d/%d 与解析一致" % (len(rows) - len(bad), len(rows)))
    if bad:
        print("不一致：%s" % ", ".join(bad))

    out = os.path.join(REPO, "out", "topology_limit_check")
    os.makedirs(out, exist_ok=True)
    import json
    with open(os.path.join(out, "limits.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(dict(tol=TOL, freq=[F_LO, F_HI], n=N_PT,
                       w_ref=w_ref, base=BASE, voigt=VOIGT_P, L_top=L_TOP,
                       rows=rows), fh, ensure_ascii=False, indent=1,
                  allow_nan=False)
    print("->", os.path.join(out, "limits.json"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
