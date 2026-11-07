# -*- coding: utf-8 -*-
"""test_bm_ad_select.py —— M4：detect_ad（门控三闸）+ select_adsorption（外层选择器）。

流程铁律：先写测试跑出 RED，再实现。用例断言**期望的正确行为**。

判据来源（ADSORPTION_PROJECT.md v1.1 §6.1/§6.2）
------------------------------------------------
detect_ad —— 四条件（低频虚部残差符号 / acf 或 rms/σ / 拐点可估 / 半圆）
  + 三闸假阳性防守：
    ① 内部局部极小（完全单调上翘 = Drift，不算吸附，R7）
    ② 半圆完整性（两臂 Re 同向、ΔRe ≈ 2|Im|_peak 的 0.3–3 倍）
    ③ 可复现性（无重复数据不判否，只降置信度）
  只看"有斜线/有环"不做决策。

select_adsorption —— 对齐 select_mechanism 的口径：
  · 切换门槛 Δ ≥ max(2, 1.5·dse)（_paired_loo_diff 同一条路径）；
  · **先拦后比**（R9）：ad 采样不健康（分级0 / Pareto-k / LOO 不可用）
    直接 guard_degraded，失败结果绝不送去比 LOO；
  · 幅值不可辨识按 σ 判定（v1.1）：P(R_ad < 3·σ̂) > 0.5 → 不加；
  · 拐点 P(f_c > f_max) > 0.5 → 退化串联 L，不加；
  · corr(Rad, R_inf) 强负相关 → M3 标定发现的简并区，只报"存在感性弛豫"，
    不报 R_ad 单独值 → 降级；
  · 0 ≤ Δ < 门槛 → uncertain（合法输出，保持 none）。

用法：python test_bm_ad_select.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import bm_ad as AD            # noqa: E402
import bm_real as R           # noqa: E402
from bm_experiments import acf_max      # noqa: E402

# ---------------------------------------------------------------- 测试基元
F = np.logspace(0, 5, 41)
ARCS = [(30.0, 300.0, 0.85), (8.0, 3e4, 0.90)]
R_INF = 10.0
NOISE = 0.10


def _fit_none_z(f, seed_noise=None):
    """「none 家族能给出的最好拟合」= 不含环的真值正演 + 噪声观测。

    用它造 res_none 是诚实的：这正是缺项拟合的残差形态（低频系统性偏
    感性 / 偏容性），不需要跑 NUTS。bins/acf 走生产函数（resid_bins /
    acf_max），字段与 fit_one 落盘一致。
    """
    w = 2 * np.pi * f
    Zn = np.full(f.shape, R_INF, complex)
    for Rad_, fp, phi in ARCS:
        tau = 1.0 / (2 * np.pi * fp)
        Zn = Zn + Rad_ / (1.0 + (1j * w * tau) ** phi)
    return Zn


def _mk_obs(with_ad, seed=3, ad=(5.0, 10.0)):
    Zn = _fit_none_z(F)
    Zobs = Zn.copy()
    if with_ad:
        taud = 1.0 / (2 * np.pi * ad[1])
        Zobs = Zobs + AD.z_ad(F, ad[0], taud)
    rng = np.random.default_rng(seed)
    Zobs = Zobs + NOISE * (rng.standard_normal(F.size)
                           + 1j * rng.standard_normal(F.size))
    return Zobs, Zn


def _mk_res_none(Zobs, Zfit):
    """把「观测 − none 拟合」打包成 detect_ad 需要的 res 替身。"""
    resid = Zobs - Zfit
    bins = R.resid_bins(resid, F)
    return dict(bins=bins, acf=max(acf_max(resid.real), acf_max(resid.imag)),
                s_re=float(np.std(resid.real)), s_im=float(np.std(resid.imag)),
                f_lo=float(F.min()), f_hi=float(F.max()))


# ---------------------------------------------------------------- detect_ad
def case1_gate_opens_for_real_loop():
    """真吸附环（拐点窗内 + 内部极小 + 两臂回基线）→ enter/strong 全过。"""
    Zobs, Zn = _mk_obs(True)
    gate = R.detect_ad(F, Zobs, _mk_res_none(Zobs, Zn))
    c = gate["checks"]
    assert c["sign_inductive"], f"低频残差符号判据未过：{gate['reason']}"
    assert c["systematic"], f"系统性残差判据未过：{gate['reason']}"
    assert c["interior_min"], "内部极小闸未过"
    assert c["half_circle"], "半圆完整性闸未过"
    assert gate["enter"] and gate["strong"], (
        f"真环没能开门：status={gate['status']} reason={gate['reason']}")
    return (f"enter/strong=True, f_c≈{gate['corner']['f_peak']:.3g} Hz, "
            f"R_ad≈{gate['corner']['Rad']:.2f}")


def case2_pure_capacitive_stays_closed():
    """纯容性（无环）→ 符号判据先拦，status 明确 written 为容性/无候选。"""
    Zobs, Zn = _mk_obs(False)
    gate = R.detect_ad(F, Zobs, _mk_res_none(Zobs, Zn))
    assert not gate["enter"], (
        f"无环谱误触发：{gate['status']} {gate['reason']}")
    assert not gate["strong"]
    return f"enter=False（{gate['status']}）"


def case3_drift_monotonic_blocked():
    """R7 假阳性：全低频段**单调**上翘（Drift 型，无内部极小）→ 第①闸必须拦。"""
    Zn = _fit_none_z(F)
    # 造一个纯单调上升的感性漂移：最低 40% 频点 Im 从 0 单调爬到正、无回落
    m = F <= np.quantile(F, 0.4)
    ramp = np.zeros(F.size)
    lf = np.log10(F)
    ramp[m] = 0.8 * (lf[m] - lf[m].min()) / (lf[m].max() - lf[m].min())
    Zobs = Zn + 1j * ramp * np.abs(Zn)[m].mean()
    gate = R.detect_ad(F, Zobs, _mk_res_none(Zobs, Zn))
    assert not gate["checks"].get("interior_min"), (
        f"单调上翘被当成内部极小放行了：{gate}")
    assert not gate["strong"], "Drift 型上翘进了 strong"
    return f"第①闸拦住（status={gate['status']}, reason 含内部极小判据）"


def case4_false_positive_rate():
    """验收标准 3（合成层）：10 个无环噪声实现，误触发 ≤5% 即 0 个。"""
    hits = []
    for seed in range(10):
        Zobs, Zn = _mk_obs(False, seed=seed)
        gate = R.detect_ad(F, Zobs, _mk_res_none(Zobs, Zn))
        if gate["enter"]:
            hits.append(seed)
    assert len(hits) <= 0, f"无环谱误触发 {hits}（10 例，要求 0 例）"
    return "10/10 无环合成谱全部未触发"


# ------------------------------------------------------- select_adsorption
def _res_family(K=2, grade=2, elpd=-100.0, bad_k=0.0, pt=None, p_ad_small=None,
                corr=None, f_c_out=None, s=None):
    pt = pt if pt is not None else np.random.default_rng(0).standard_normal(41)
    r = dict(K=K, grade=grade, elpd=elpd, bad_k=bad_k, pointwise_elpd=pt,
             p_ad_small=p_ad_small, f_c_out_prob=f_c_out, s=s or {})
    if corr is not None:
        r["param_corr"] = corr
    return r


def case5_switches_when_strong_evidence():
    """ad 健康 + Δ ≥ max(2,1.5dse) → choice=ad / status=switched。"""
    none = _res_family(elpd=-100.0)
    ad = _res_family(elpd=-100.0 + 40.0)   # ad 明显更好
    d = np.zeros(41)
    ad["pointwise_elpd"] = none["pointwise_elpd"] + 1.0   # Δ=41, dse=0
    sel = R.select_adsorption(none, ad, {"strong": True, "enter": True})
    assert sel["choice"] == "ad" and sel["status"] == "switched", \
        f"证据充分却没切换：{sel}"
    return f"Δ={sel['delta']:.1f} ≥ {sel['threshold']:.1f} → ad"


def case6_blocks_unhealthy_sampling_before_compare():
    """R9 先拦后比：ad 分级 0 → guard_degraded，且不出现 delta（没送去比）。"""
    none = _res_family(elpd=-100.0)
    ad = _res_family(grade=0, elpd=-60.0)
    sel = R.select_adsorption(none, ad, {"strong": True, "enter": True})
    assert sel["status"] == "guard_degraded" and sel["choice"] == "none", sel
    assert sel.get("delta") is None, "失败采样仍被送去配对了 LOO"
    assert "grade" in str(sel.get("guards", {})) or "采样" in sel["reason"], \
        f"理由未写明采样问题：{sel['reason']}"
    return "ad 分级 0 → 未比 LOO 直接拦回 none"


def case7_small_effect_stays_uncertain():
    """0 ≤ Δ < 门槛 → uncertain 保持 none（合法输出，不硬选）。"""
    none = _res_family(elpd=-100.0)
    ad = _res_family(elpd=-100.0)
    ad["pointwise_elpd"] = none["pointwise_elpd"] + 0.01
    sel = R.select_adsorption(none, ad, {"strong": True, "enter": True})
    assert sel["choice"] == "none" and sel["status"] == "uncertain", sel
    return f"Δ={sel['delta']:.2f} < {sel['threshold']:.1f} → uncertain"


def case8_identifiability_guards():
    """两条**阻断型**护栏：幅值按 3σ̂、拐点出窗（立项 §6.2 原列的两条）。"""
    none = _res_family(elpd=-100.0)
    good_ad = _res_family(elpd=-60.0)
    good_ad["pointwise_elpd"] = none["pointwise_elpd"] + 1.0

    # 8a 幅值不可辨识（v1.1：按 σ 不是 |Z|max）
    sel = R.select_adsorption(none, dict(good_ad, p_ad_small=0.9),
                              {"strong": True, "enter": True})
    assert sel["status"] == "guard_degraded", f"幅值护栏未拦：{sel}"
    assert sel["choice"] == "none"
    # 8b 拐点大概率出窗（退化为串联 L，已被 include_L 建模）
    sel = R.select_adsorption(none, dict(good_ad, f_c_out_prob=0.8),
                              {"strong": True, "enter": True})
    assert sel["status"] == "guard_degraded", f"拐点出窗护栏未拦：{sel}"
    assert sel["choice"] == "none"
    return "幅值 / 拐点出窗两条阻断护栏命中"


def case8b_degeneracy_downgrades_report_not_switch():
    """M3 标定教训的正确落法：Rad–R_inf 简并**不阻断切换**，只降级报告。

    实测背景（test_bm_ad_e2e case3）：某谱 Δelpd=+121、acf 0.89→0.49，
    环明显为真，但 corr(R_ad,R_inf) 强负。若按 guard_degraded 拦回 none，
    等于为了"R_ad 绝对值不可单独引用"而丢掉一个明显更好的模型——
    立项 §6.2 的护栏清单里本来也没有这一条；M3 的原话是"只报存在低频
    感性弛豫，不报 R_ad 单独值"，即**改报告口径**。
    """
    none = _res_family(elpd=-100.0)
    ad = _res_family(elpd=-60.0)
    ad["pointwise_elpd"] = none["pointwise_elpd"] + 1.0     # Δ=41，显著
    corr = dict(strong=[dict(a="R_ad", b="R_inf", rho=-0.95, why="")])

    sel = R.select_adsorption(none, dict(ad, param_corr=corr),
                              {"strong": True, "enter": True})
    assert sel["choice"] == "ad" and sel["status"] == "switched", (
        f"简并不该阻断切换（Δ 显著时应切 ad）：{sel['status']}/{sel['choice']}")
    assert sel.get("rad_rinf_degenerate") is True, "未标记简并，报告会给出假精度"
    assert any("R_ad" in t for t in sel.get("note", [])), \
        f"note 里没写降级说明：{sel.get('note')}"
    return "简并 → 仍切 ad，但标记 rad_rinf_degenerate 并在 note 降级 R_ad 口径"


def case9_weak_gate_blocks_compare():
    """三闸未全过（strong=False）→ 不进比较，直接 uncertain/保持 none。"""
    none = _res_family(elpd=-100.0)
    ad = _res_family(elpd=-60.0)
    ad["pointwise_elpd"] = none["pointwise_elpd"] + 1.0
    sel = R.select_adsorption(none, ad, {"strong": False, "enter": True})
    assert sel["choice"] == "none" and sel.get("delta") is None, \
        f"gate 未过还去比了 LOO：{sel}"
    return "gate.strong=False → 不比较"


CASES = [case1_gate_opens_for_real_loop, case2_pure_capacitive_stays_closed,
         case3_drift_monotonic_blocked, case4_false_positive_rate,
         case5_switches_when_strong_evidence,
         case6_blocks_unhealthy_sampling_before_compare,
         case7_small_effect_stays_uncertain, case8_identifiability_guards,
         case8b_degeneracy_downgrades_report_not_switch,
         case9_weak_gate_blocks_compare]

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
