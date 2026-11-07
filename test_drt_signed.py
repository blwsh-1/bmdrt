# -*- coding: utf-8 -*-
"""test_drt_signed.py —— M5 方案 A（有符号 DRT）的可复现测试。

方法来源（两篇论文）
--------------------
Plank et al. 2025, J. Electrochem. Soc. 172 060514（Part I. Deconvolution
Methods）：
  · 放弃非负约束，用单个无约束分布 ϒ(τ) + 标准 RC 核拟合整谱（Eq. 24）；
    感性特征 = 负 ϒ 分量，等价于恒等式 R·jωτ/(1+jωτ) = R − R/(1+jωτ)
    （即 bm_ad 恒等式，与立项书 §2.1 相同）。
  · 代价（Eq. 28）：无约束反演给出的欧姆偏移叠加了负极化，必须修正
        R₀,true = R₀,DRT − Σ_{k, ϒₖ<0} |ϒₖ|
    （离散节点 ϒₖ = γₖ·Δlnτ，实现见 drt.py 的 neg_pol）。
  · 他们的合成验证**特意把 RK 与 RQ 的时间常数选得远隔**以减少重叠
    （原文：更显著的重叠留给后续论文）——本测试同一设计，否则
    Tikhonov 二阶平滑在 1 个倍频程内的相邻特征间必然振铃（实测）。
  · 实测精度：修正后欧姆偏移误差 <0.9%（gDRT），VanCittert/SSSD 同。

Yang & Zhuang 2026, JACS Au 6 4819（CA-DRT Perspective, §6/§7）：
  · 负峰是**基系数**，不是"负电阻"；报告应锚定通道参数
    （inductive channel：R_L 与 τ=L/R_L）。据此，负叶不进 K 候选，
    单独分流到 ad_peaks 供 M4 吸附选择器消费。

测试策略
--------
先跑 RED（drt.py 无 signed 参数时 4 例失败），实现后全绿。
case4 锁死默认非负路径**逐位不变**（立项 §5 回归红线）。

    python test_drt_signed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import drt as DR        # noqa: E402

# ---------------------------------------------------------------- 合成谱
# 特征远隔（Plank 同款设计）：吸附环 0.5 Hz / RC 100 Hz / RC 10 kHz，
# 相邻间距 ≥2 decade；τ 网格 m=60 的分辨率 ~0.14 decade，足以分离。
R0 = 5.0                    # 真欧姆偏移
RCs = [(20.0, 100.0), (8.0, 1e4)]     # (R, f_peak)
AD = dict(Rad=3.0, f_c=0.5)           # 吸附环：拐点 0.5 Hz，窗口内
NOISE = 3e-3                # 相对噪声 0.3%
F = np.logspace(-1, 5, 61)  # 0.1 Hz – 100 kHz


def synth(f, with_ad=True):
    w = 2 * np.pi * f
    Z = np.full(f.shape, R0, complex)
    for R, fp in RCs:
        tau = 1.0 / (2 * np.pi * fp)
        Z = Z + R / (1.0 + 1j * w * tau)
    if with_ad:
        taud = 1.0 / (2 * np.pi * AD["f_c"])
        Z = Z + AD["Rad"] * (1j * w * taud) / (1.0 + 1j * w * taud)
    rng = np.random.default_rng(7)
    return Z * (1.0 + NOISE * rng.standard_normal(f.size))


# ---------------------------------------------------------------- 用例
def case1_signed_finds_negative_lobe():
    """含吸附环的谱：signed=True 必须看到负峰并估出 (R_ad, τ_ad)。"""
    Z = synth(F)
    res = DR.drt(F, Z, sigma=NOISE, signed=True)
    peaks = DR.peaks(res)
    neg = [p for p in peaks if p["sign"] < 0]
    assert neg, "signed 反演没找到任何负峰 —— 负弛豫仍然不可见"
    p = max(neg, key=lambda q: abs(q["R_ohm"]))
    err_f = abs(p["f_peak_Hz"] / AD["f_c"] - 1.0)
    err_a = abs(abs(p["R_ohm"]) / AD["Rad"] - 1.0)
    assert err_f < 0.6, f"主负峰峰位 {p['f_peak_Hz']:.3g} Hz，真值 {AD['f_c']} Hz"
    assert err_a < 0.35, f"主负峰面积 {p['R_ohm']:.3g}，真值 −{AD['Rad']}"
    return f"负峰 f={p['f_peak_Hz']:.3g} Hz, 面积={p['R_ohm']:.2f}（Δf {err_f:.0%}, ΔA {err_a:.0%}）"


def case2_ohmic_offset_correction():
    """Plank Eq. 28：R₀,true = R₀,DRT − Σ|负极化|，修正后必须接近真值。

    口径：用 prescreen(signed=True) 的闸门后负极化总量 `ad_pol`
    （只算有原始数据支持的真负叶）。drt() 里的逐节点 `neg_pol` 会把
    二阶平滑的振铃小叶也计进去，实测过冲——见 drt.py `_raw_support` 注释。
    """
    Z = synth(F)
    res, pk, pk_all = DR.prescreen(F, Z, sigma=NOISE, signed=True)
    assert "R_inf_true" in res and "ad_pol" in res, "修正字段未实现"
    raw_err = abs(res["R_inf"] / R0 - 1.0)
    fix_err = abs(res["R_inf_true"] / R0 - 1.0)
    assert fix_err < 0.08, (
        f"修正后 R₀,true={res['R_inf_true']:.3f}，真值 {R0}（误差 {fix_err:.1%}）；"
        f"未修正 {res['R_inf']:.3f}（误差 {raw_err:.1%}）")
    assert raw_err > fix_err, "修正没有改善欧姆偏移，公式方向可能反了"
    return (f"R₀,DRT={res['R_inf']:.3f}（{raw_err:.0%}）→ 修正 "
            f"R₀,true={res['R_inf_true']:.3f}（{fix_err:.1%}），"
            f"闸门后负极化 ad_pol={res['ad_pol']:.3f}")


def case3_control_nonneg_blind():
    """对照组：默认非负路径看不见环（χ² 到不了噪声底、无负峰）。"""
    Z = synth(F)
    res = DR.drt(F, Z, sigma=NOISE)          # signed 默认 False
    assert bool(np.all(res["gamma"] >= -1e-12)), "非负路径γ出现负值"
    rms = float(np.sqrt(np.mean(np.abs(res["resid"]) ** 2)))
    assert rms > 2 * NOISE, (
        f"非负反演残差 rms={rms:.2%} 已贴噪声底 {NOISE:.2%} —— "
        f"对照组前提不成立（环被什么吸收掉了？）")
    return f"非负路径：γ≥0 恒成立，残差 rms={rms:.2%} ≫ 噪声 {NOISE:.2%}（盲区复现）"


def case4_default_path_unchanged():
    """回归红线：signed=False 时行为必须与改动前逐位一致。

    基准文件 out/_signed_ref/nonneg_ref.npz 由**改动前**的 drt.py 生成。
    """
    ref_path = ROOT / "out" / "_signed_ref" / "nonneg_ref.npz"
    assert ref_path.exists(), f"缺基准文件 {ref_path}——需先改动前生成"
    import bm_core as B
    ff, Z0 = B.noisy()                        # 合成两 ZARC 谱，固定 seed=0
    res = DR.drt(ff, Z0, sigma=2e-3)
    ref = np.load(ref_path)
    assert np.array_equal(res["gamma"], ref["gamma"]), "默认路径 γ 变了（逐位红线破坏）"
    assert res["R_inf"] == float(ref["R_inf"]), "默认路径 R_inf 变了"
    assert res["lam"] == float(ref["lam"]), "默认路径 λ 变了"
    assert res["chi2"] == float(ref["chi2"]), "默认路径 χ² 变了"
    # peaks 的非负路径也要与旧输出一致：不得多出 sign 依赖导致的差异
    pk = DR.peaks(res)
    assert all(p["sign"] > 0 for p in pk), "非负路径出现负 sign"
    return "默认（非负）路径输出与基准逐位一致"


def case5_pure_capacitive_no_false_negative():
    """无环纯容性谱：signed 反演不得造出过门槛的吸附候选（假阳性=0）。"""
    Z = synth(F, with_ad=False)
    res, pk, pk_all = DR.prescreen(F, Z, sigma=NOISE, signed=True)
    assert res["ad_peaks"] == [], (
        f"纯容性谱出现吸附候选 {[(round(p['f_peak_Hz'],2), round(p['R_ohm'],3)) for p in res['ad_peaks']]}")
    assert len(pk) == 2, f"HN 候选 {len(pk)} 个，应为 2（振铃不得混入正峰）"
    assert abs(res["ad_pol"]) < 1e-9, \
        f"ad_pol={res['ad_pol']:.3f}：无环时闸门后负极化必须为 0"
    return f"ad_peaks 为空，ad_pol={res['ad_pol']:.1f}（振铃负叶全被原始数据闸门拒掉）"


def case6_prescreen_ad_candidates():
    """prescreen 集成：K 候选只数正峰，负峰单独进 ad_peaks（不撑大 K）。"""
    Z = synth(F)
    res, pk, pk_all = DR.prescreen(F, Z, sigma=NOISE, signed=True)
    assert len(pk) == 2, f"HN 候选峰数 {len(pk)}，应为 2（负峰不得混入）"
    ad = res["ad_peaks"]
    assert len(ad) == 1, f"ad_peaks={len(ad)} 个，应给出 1 个吸附候选"
    a = ad[0]
    assert a["R_ohm"] < 0, "ad_peaks 面积应为负"
    err_f = abs(a["f_peak_Hz"] / AD["f_c"] - 1.0)
    assert err_f < 0.6, f"吸附候选峰位 {a['f_peak_Hz']:.3g} Hz，真值 {AD['f_c']}"
    return f"pk=2（正峰）, ad_peaks=1 @ {a['f_peak_Hz']:.3g} Hz, R={a['R_ohm']:.2f}"


CASES = [case1_signed_finds_negative_lobe,
         case2_ohmic_offset_correction,
         case3_control_nonneg_blind,
         case4_default_path_unchanged,
         case5_pure_capacitive_no_false_negative,
         case6_prescreen_ad_candidates]

if __name__ == "__main__":
    bad = []
    for fn in CASES:
        try:
            print(f"[PASS] {fn.__name__}  {fn()}")
        except AssertionError as e:
            bad.append(fn.__name__)
            print(f"[FAIL]  {fn.__name__}\n        {e}")
        except Exception as e:                      # noqa: BLE001
            bad.append(fn.__name__)
            print(f"[ERROR] {fn.__name__}  {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - len(bad)}/{len(CASES)} 通过；未通过：{bad}")
    raise SystemExit(1 if bad else 0)
