# -*- coding: utf-8 -*-
"""test_ad_gate.py —— M6 第一阶段：`detect_ad` 边界状态 + 不变性回归。

规格来源：`AD_GATE_SPEC.md` §3（五状态）、§4（离散点数主判据）、§5（不变性）、
§7.0（规格冻结四件事）。本文件对应第 1/2/4 项的验收。

覆盖两类：
  A. 合成谱单元测试 —— 五种状态各一条，外加「前置条件」与「状态不再被覆盖」；
  B. 真实数据回放 —— 若 `out/_m6_samples/` 有结果 json，用 json 里的
     `resid_bins/acf/sigma_im` 当 `res_none` 重算 gate（不重跑 NUTS），
     断言 strong 集合与最终选择不变、贴边谱只增加候选诊断。

用法：python test_ad_gate.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import bm_real as R                      # noqa: E402
from bm_experiments import acf_max       # noqa: E402

F = np.logspace(-2, 5, 60)
R_INF = 10.0
ARCS = [(30.0, 300.0, 0.85), (8.0, 3e4, 0.90)]
NOISE = 0.10


def _zn(f=F):
    w = 2 * np.pi * f
    z = np.full(f.shape, R_INF, complex)
    for rad, fp, phi in ARCS:
        tau = 1.0 / (2 * np.pi * fp)
        z = z + rad / (1.0 + (1j * w * tau) ** phi)
    return z


def _mk(extra=None, seed=3, noise=NOISE):
    """extra: 叠加到 Zn 上的复数项（ad 元件 / 虚部鼓包 / 负鼓包）。"""
    Zn = _zn()
    Z = Zn.copy() if extra is None else Zn + extra
    rng = np.random.default_rng(seed)
    return Z + noise * (rng.standard_normal(F.size) + 1j * rng.standard_normal(F.size)), Zn


def _res(Zobs, Zn):
    resid = Zobs - Zn
    return dict(bins=R.resid_bins(resid, F),
                acf=max(acf_max(resid.real), acf_max(resid.imag)),
                s_re=float(np.std(resid.real)), s_im=float(np.std(resid.imag)),
                f_lo=float(F.min()), f_hi=float(F.max()))


def _bump(center_hz, amp, width=0.15):
    return 1j * amp * np.exp(-0.5 * (np.log10(F / center_hz) / width) ** 2)


def _ad(rad=100.0, fp=1.0):
    import bm_ad as AD
    return AD.z_ad(F, rad, 1.0 / (2 * np.pi * fp))


def _band(f=F, frac=0.40):
    lf = np.log10(f)
    return lf <= float(np.quantile(lf, frac))


# ---------------------------------------------------------------- 合成谱
def case1_strong_candidate():
    """完整环（内部极小 + 半圆完整）→ enter/strong 全过。"""
    Zobs, Zn = _mk(_ad())
    g = R.detect_ad(F, Zobs, _res(Zobs, Zn))
    c = g["checks"]
    assert c["interior_min"] and c["half_circle"], f"形状闸未过：{g['reason']}"
    assert g["status"] == "strong_candidate" and g["enter"] and g["strong"]
    assert c["k_left"] >= 2 and c["k_right"] >= 2
    return f"strong_candidate, k_left={c['k_left']}, d_low={c['d_low']:.2f}"


def case2_partial_arc_status_not_overwritten():
    """内部极小但半圆不完整 → partial_arc（**不再被 weak_candidate 覆盖**）。"""
    Zobs, Zn = _mk(_bump(1.0, 120.0))       # 纯虚部鼓包：ΔRe 相对 2|Im| 偏小
    g = R.detect_ad(F, Zobs, _res(Zobs, Zn))
    assert g["checks"]["interior_min"], "应当有内部极小"
    assert not g["checks"]["half_circle"], "构造上应当半圆不完整"
    assert g["status"] == "partial_arc", f"状态被覆盖成了 {g['status']}"
    assert g["enter"] is True and g["strong"] is False
    assert "不进模型比较" in g["reason"] and "候选诊断" in g["reason"]
    return f"partial_arc, enter=True/strong=False, ratio={g['corner']['re_over_im_ratio']:.2f}"


def case3_edge_peak_kleft1():
    """峰在窗内但贴边（k_left=1）→ edge_peak：开诊断、不开自动选择。"""
    f1 = float(F[1])
    Zobs, Zn = _mk(_bump(f1, 100.0, width=0.12))
    g = R.detect_ad(F, Zobs, _res(Zobs, Zn))
    assert g["checks"]["k_left"] == 1, f"k_left={g['checks']['k_left']}，构造失败"
    assert g["status"] == "edge_peak", f"status={g['status']} {g['reason']}"
    assert g["enter"] is True and g["strong"] is False
    assert g["corner"] and g["corner"]["k_left"] == 1
    return f"edge_peak, f_peak={g['corner']['f_peak']:.4g} Hz, enter/strong=True/False"


def case4_kleft0_stays_monotonic():
    """峰在最低频点（k_left=0）→ 保持 monotonic_rise，enter=False。"""
    Zobs, Zn = _mk(_bump(float(F[0]), 100.0, width=0.12))
    g = R.detect_ad(F, Zobs, _res(Zobs, Zn))
    assert g["checks"]["k_left"] == 0, f"k_left={g['checks']['k_left']}"
    assert g["status"] == "monotonic_rise" and g["enter"] is False
    return f"monotonic_rise, k_left=0, enter=False"


def case5_right_edge_stays_monotonic():
    """峰贴 band 上沿（k_right<2）→ 不属于 edge_peak，仍 enter=False。

    叠加一个常数正虚部，保证低频残差符号判据能过（否则会在符号关提前返回，
    根本走不到形状闸，测不到 k_right）。
    """
    m = _band()
    f_last = float(F[m][-1])
    Zobs, Zn = _mk(_bump(f_last, 200.0, width=0.12) + 1j * 60.0)
    g = R.detect_ad(F, Zobs, _res(Zobs, Zn))
    assert g["checks"].get("k_right") is not None, \
        f"未进入形状闸（status={g['status']}）：{g['reason']}"
    assert g["checks"]["k_right"] < 2, f"k_right={g['checks']['k_right']}"
    assert g["status"] == "monotonic_rise" and g["enter"] is False
    return f"monotonic_rise, k_right={g['checks']['k_right']}, enter=False"


def case6_pure_capacitive_blocked_by_sign():
    """无环（纯容性）→ 符号关先拦，不进形状闸。"""
    Zobs, Zn = _mk(None)
    g = R.detect_ad(F, Zobs, _res(Zobs, Zn))
    assert g["status"] == "capacitive_tail" and g["enter"] is False
    return "capacitive_tail, enter=False"


def case7_noise_only_blocked_by_systematic():
    """噪声主导 → 系统性关拦住，不进形状闸。

    用常数正虚部（不是环）保证符号判据确定通过，再叠加大噪声让 acf 低、
    低频 rms/σ 接近 1 —— 这正是「符号为正但不可信」的形态。
    """
    # c 要够大让符号判据稳过（20 点均值噪声 ~45），又要小到 low_rms/σ < 1.5：
    # c=150, σ=200 → low_snr≈1.25，符号均值 150±45，两边都稳。
    Zobs, Zn = _mk(1j * 150.0, noise=200.0)
    g = R.detect_ad(F, Zobs, _res(Zobs, Zn))
    assert g["status"] == "noise_only" and g["enter"] is False, \
        f"status={g['status']} checks={g['checks']}"
    return "noise_only, enter=False"


def case8_edge_peak_needs_sign_and_systematic():
    """**前置条件**：k_left=1 但虚部为负 → 必须在符号关返回，绝不进 edge_peak。

    这是 2E 对照谱不因 k_left=1 进入候选诊断的机制保证（AD_GATE_SPEC §4）。
    """
    f1 = float(F[1])
    Zobs, Zn = _mk(-_bump(f1, 100.0, width=0.12))   # 负鼓包 = 容性
    g = R.detect_ad(F, Zobs, _res(Zobs, Zn))
    assert g["status"] != "edge_peak", "负虚部却进了 edge_peak"
    assert g["enter"] is False
    return f"符号关先拦（{g['status']}），不进 edge_peak"


# ---------------------------------------------------------------- 真实回放
def _load_real(root):
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*", "real_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        stem = os.path.basename(p)[5:-5]
        cand = [os.path.join("out/_m6_samples/z_csv", stem + ".csv"),
                os.path.join("EISmeasurements/csv_2e", stem + ".csv")]
        path = next((c for c in cand if os.path.exists(c)), None)
        if path is None:
            continue
        a = np.loadtxt(path, delimiter=",", skiprows=1)
        out.append((stem, a[:, 0], a[:, 1] + 1j * a[:, 2], d))
    return out


def case9_real_strong_set_unchanged():
    """真实 4 条 + 对照 10 条：新 gate 的 strong 集合与旧 json 完全一致。"""
    rows = (_load_real("out/_m6_samples/fit_real")
            + _load_real("out/_m6_samples/fit_control"))
    if not rows:
        return "SKIP：out/_m6_samples 无结果 json"
    bad = []
    for stem, f, Z, d in rows:
        old = d.get("ad_gate") or {}
        if old.get("status") in (None, "-"):
            continue
        g = R.detect_ad(f, Z, d)          # json 自带 resid_bins/acf/sigma_im
        if bool(g["strong"]) != bool(old.get("strong")):
            bad.append(f"{stem}: strong {old.get('strong')}→{g['strong']}"
                       f"（{old.get('status')}→{g['status']}）")
    assert not bad, "strong 集合变了：" + "；".join(bad)
    return f"{len(rows)} 条谱 strong 集合不变"


def case10_real_edge_peak_and_no_escalation():
    """AZ31 贴边谱 → edge_peak；k_left=0 不升级；对照不因 k_left=1 进诊断。"""
    real = dict((s, (f, Z, d)) for s, f, Z, d in _load_real("out/_m6_samples/fit_real"))
    ctrl = dict((s, (f, Z, d)) for s, f, Z, d in _load_real("out/_m6_samples/fit_control"))
    if not real:
        return "SKIP：无真实谱结果"

    edge = []
    for s in ("AZ31-4Li_50", "AZ31-8Li_150"):
        if s in real:
            f, Z, d = real[s]
            g = R.detect_ad(f, Z, d)
            assert g["status"] == "edge_peak" and g["enter"] and not g["strong"], \
                f"{s} 未按 edge_peak 处理：{g['status']} enter={g['enter']}"
            edge.append(s)

    for s in ("AZ31-4Li_0", "AZ31-4Li_5", "AZ31-8Li_100"):
        if s in real:
            f, Z, d = real[s]
            g = R.detect_ad(f, Z, d)
            assert g["status"] == "monotonic_rise" and not g["enter"], \
                f"{s} 被误升级：{g['status']} enter={g['enter']}"

    # 对照谱：旧 enter=False 的，新 enter 也必须 False（不因 k_left=1 开诊断）
    bad = []
    for s, (f, Z, d) in ctrl.items():
        old = d.get("ad_gate") or {}
        if old.get("enter") is False:
            g = R.detect_ad(f, Z, d)
            if g["enter"]:
                bad.append(f"{s}: enter False→True（{g['status']}）")
    assert not bad, "对照谱新增候选诊断：" + "；".join(bad)
    return f"edge_peak={edge}；k_left=0 未升级；对照无新增 enter"


def case11_final_choice_unchanged():
    """最终选择不变：旧 choice=ad 的谱，新 gate 仍 strong；skipped 的仍非 strong。"""
    rows = (_load_real("out/_m6_samples/fit_real")
            + _load_real("out/_m6_samples/fit_control"))
    if not rows:
        return "SKIP：无结果 json"
    bad = []
    for stem, f, Z, d in rows:
        sel = d.get("ad_selection") or {}
        old = d.get("ad_gate") or {}
        if old.get("status") in (None, "-"):
            continue
        g = R.detect_ad(f, Z, d)
        if sel.get("choice") == "ad" and not g["strong"]:
            bad.append(f"{stem}: 旧选择 ad，新 gate 不再 strong")
        if sel.get("status") == "skipped_by_gate" and g["strong"]:
            bad.append(f"{stem}: 旧为 skipped，新 gate 变 strong")
    assert not bad, "最终选择路径被改变：" + "；".join(bad)
    return "旧 choice 与新 gate 推演一致（strong 集合不变）"


CASES = [case1_strong_candidate,
         case2_partial_arc_status_not_overwritten,
         case3_edge_peak_kleft1,
         case4_kleft0_stays_monotonic,
         case5_right_edge_stays_monotonic,
         case6_pure_capacitive_blocked_by_sign,
         case7_noise_only_blocked_by_systematic,
         case8_edge_peak_needs_sign_and_systematic,
         case9_real_strong_set_unchanged,
         case10_real_edge_peak_and_no_escalation,
         case11_final_choice_unchanged]


def main():
    bad, skip = [], []
    for fn in CASES:
        try:
            msg = fn()
            if isinstance(msg, str) and msg.startswith("SKIP"):
                skip.append(fn.__name__)
                print(f"[SKIP] {fn.__name__}  {msg}")
            else:
                print(f"[PASS] {fn.__name__}  {msg}")
        except AssertionError as e:
            bad.append(fn.__name__)
            print(f"[FAIL] {fn.__name__}  {e}")
        except Exception as e:                      # noqa: BLE001
            bad.append(fn.__name__)
            print(f"[ERROR] {fn.__name__}  {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - len(bad) - len(skip)}/{len(CASES)} 通过"
          f"；跳过 {len(skip)}；未通过：{bad}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
