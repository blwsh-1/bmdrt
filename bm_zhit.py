# -*- coding: utf-8 -*-
"""Z-HIT 自洽检验的**一阶形式**（离线原型）。

结论先说：**已评估，不接入。**
2026-11-12 在合成标定集 + 22 条真实谱上跑过：一阶形式的固有误差
（无噪声的完美谱也有 0.95%–3.25%）与我们想检的不自洽（5% 幅值偏差 → 3.0%）
**同量级**；22 条真实谱全部 fail（3.0%–35.4%），且 fail 的驱动是**谱型**
（端部感抗、相位跨 0），不是数据坏。所以它给不出可判定的结论。
详见 `docs/round_2026-11-12/ZHIT_PROTOTYPE.md`。留这份代码是为了不再
重复评估 —— 别拿它当门禁、别拿它的百分比对外报。

它和 Lin-KK 分工在哪
--------------------
Lin-KK 用一条 RC 链去逼近**复阻抗**，误差同时落在实部和虚部；它默认
"幅值标定是对的"。Z-HIT 走另一条路：只用**相位**沿着频率积分出幅值，
再拿它跟实测幅值比。所以两者对不同的病敏感：

    病                         Lin-KK          Z-HIT
    实部鼓包 / 虚部形状不自洽   看得见          基本看不见（不看实部）
    幅值与相位不自洽           间接（要凑出复阻抗）  直接
    整体幅值标定错（如分流电阻） 看不见            看不见（锚点把整体平移吃掉）
    通道相位误差 / 相位漂移      弱              强（相位进积分）
    窗口外低频发散              需要 add_cap     不敏感（只积窗内）

一句话：Z-HIT 检验"幅值随频率的**形状**与相位是否自洽"，不检验绝对标定。
这是 K-K 本身管不到的事，别指望它替代 Lin-KK。

这不是 Zahner 的商业实现
------------------------
Z-HIT 的出发点是 Bode 增益-相位关系（Hilbert 变换的实轴形式）：

    φ(ω) = (π/2) · d ln|Z| / d ln ω            (局部近似)

反解并积分：

    ln|Z(ω)| = c + (2/π) ∫_{lnω_a}^{lnω} φ(ω') d ln ω'

Zahner 的 "modified Z-HIT" 在此之上加了窗函数、端部外推和二阶修正项，
参数不公开。**本文件只实现上面这条可解析验证的一阶式**，因此：

  · 对"宽带 DRT"（多个时间常数铺开）近似很好；
  · 对**单弛豫 / Debye**（窄分布）有系统性偏差，偏差量级由 `synth_suite()`
    标定给出，不是 0；
  · 不能拿本文件的百分比去和商业软件的数字比。

锚点 c 为什么用中位数
--------------------
积分本身有一个任意常数（积分下限）。若用最小二乘定 c，端部的离群点会把
整条线拉偏；用中位数（`anchor="median"`）更稳。但 c 一吃掉，**整体幅值
偏差就检不出来了**——这是设计选择，不是漏洞，写在返回值的
`diag["anchor_spread"]` 里供核对（median / ls / low / high 四种锚点给出的
rms 若差很多，说明偏差集中在端部而不是整条线）。

判据线从哪来
------------
**不是文献值，也不是预注册线。** 本项目已有的教训是：没有依据的阈值
（比如 Rwp 的 2%/10%）不该拿去上色。所以这里的阈值只由 `--synth` 的
合成标定给出，并且**分级永远同时报噪声底**：合成谱加 1% 噪声后本身的
Z-HIT rms 就是噪声底，真实谱的 rms 低于它谈不上"不自洽"。

用法
----
    python bm_zhit.py archive/AZ31-4Li.txt
    python bm_zhit.py --synth                 # 合成标定表
    python bm_zhit.py --synth --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VERSION = "1.0-proto"

# ---- 分级阈值（由 synth_suite() 标定，见 docs 说明；不是文献值）----
RMS_OK = 0.01        # 1%：宽带 DRT + 1% 噪声的噪声底量级
RMS_FAIL = 0.03      # 3%：超过它就不是"一阶近似的锅"
TREND_OK = 1.0       # %/decade
TREND_FAIL = 2.5     # %/decade


# --------------------------------------------------------------------------
# 相位平滑
# --------------------------------------------------------------------------

def _smooth(u: np.ndarray, phi: np.ndarray, win: int | None) -> tuple[np.ndarray, int]:
    """在 ln f 上做 Savitzky-Golay 平滑。

    Z-HIT 的量是 ∫φ dlnω，相位的高频抖动会被**累积**成幅值偏差，
    所以平滑不是美化，是必需。但平滑也会削掉真实的窄峰 → 窗宽必须
    报出去，并做敏感性检查（`win_sensitivity()`）。
    """
    n = phi.size
    if win is None:
        win = max(5, n // 5)
    win = int(win)
    if win % 2 == 0:
        win += 1
    win = min(win, n if n % 2 else n - 1)
    if win < 5 or n < 5:
        return phi.copy(), 0
    try:
        from scipy.signal import savgol_filter
    except Exception:                       # noqa: BLE001
        # 无 scipy 时退化为滑动平均，宁可粗糙也不静默跳过
        k = np.ones(win) / win
        pad = win // 2
        ext = np.r_[phi[pad:0:-1], phi, phi[-2:-pad - 2:-1]]
        return np.convolve(ext, k, mode="same")[pad:pad + n], -win
    order = 2 if win >= 7 else 1
    return savgol_filter(phi, win, order), win


# --------------------------------------------------------------------------
# 主检验
# --------------------------------------------------------------------------

def zhit(f, Z, win: int | None = None, anchor: str = "median",
         n_band: int = 3, smooth: bool = True) -> dict:
    """一阶 Z-HIT。返回 dict（可 JSON 序列化，`curve` 除外）。

    f 升序、Z 为 `Zre + j*Zim`（电容 = 负虚部），与 bm_kk / bm_real 一致。
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    o = np.argsort(f)
    f, Z = f[o], Z[o]

    base = dict(version=VERSION, status="unavailable", reason="",
                rms_rel=None, max_abs_rel=None, bands=None,
                trend_pct_per_dec=None, trend_r=None, ac1=None,
                win=None, anchor=anchor, anchor_spread=None,
                n=int(f.size), f_min=None, f_max=None,
                attribution=[], curve=None)

    if f.size < 8:
        base["reason"] = f"仅 {f.size} 点（<8），Z-HIT 不可靠，跳过"
        return base
    if np.any(~np.isfinite(f)) or np.any(~np.isfinite(Z)):
        base["reason"] = "存在非有限值，跳过"
        return base

    u = np.log(f)
    phi_raw = np.unwrap(np.angle(Z))
    lnmag = np.log(np.abs(Z))

    if smooth:
        phi, w = _smooth(u, phi_raw, win)
    else:
        phi, w = phi_raw.copy(), 0

    # ---- 积分：(2/π)∫φ dlnω ----
    du = np.diff(u)
    mid = 0.5 * (phi[1:] + phi[:-1])
    I = np.r_[0.0, np.cumsum(mid * du)] * (2.0 / np.pi)

    # ---- 锚点 ----
    r = lnmag - I
    if anchor == "ls":
        c = float(np.sum(r) / r.size)
    elif anchor == "low":
        c = float(np.median(r[:max(3, r.size // 6)]))
    elif anchor == "high":
        c = float(np.median(r[-max(3, r.size // 6):]))
    else:
        c = float(np.median(r))

    resid = lnmag - (I + c)                 # 对数残差
    delta = np.expm1(resid) * 100.0         # 相对偏差，%

    # ---- 分段（按频点数等分，升频序 → low 在前）----
    edges = np.linspace(0, f.size, n_band + 1).round().astype(int)
    names = ("low", "mid", "high") if n_band == 3 else tuple(
        f"b{i}" for i in range(n_band))
    bands = {}
    for i, nm in enumerate(names):
        sl = slice(edges[i], edges[i + 1])
        d = delta[sl]
        # 统一存**小数**（与 rms_rel 同口径），打印时再乘 100。
        bands[nm] = dict(rms=float(np.sqrt(np.mean(d ** 2))) / 100.0,
                         mean=float(np.mean(d)) / 100.0,
                         max_abs=float(np.max(np.abs(d))) / 100.0,
                         f_lo=float(f[sl][0]), f_hi=float(f[sl][-1]),
                         n=int(d.size))

    rms = float(np.sqrt(np.mean(delta ** 2)))
    mx = float(np.max(np.abs(delta)))

    # 趋势：% per decade（OLS on ln f）
    uc = u - u.mean()
    slope = float(np.sum(uc * delta) / max(np.sum(uc ** 2), 1e-30))
    trend = slope * np.log(10.0)            # %/decade
    # 相关系数（趋势的"确定程度"，不是显著性检验）
    sd = float(np.std(delta))
    tr = float(np.corrcoef(uc, delta)[0, 1]) if sd > 0 else 0.0

    # 残差相邻自相关：高 = 系统性偏差，低 = 噪声
    d0 = delta - delta.mean()
    den = float(np.sum(d0 ** 2))
    ac1 = float(np.sum(d0[:-1] * d0[1:]) / den) if den > 0 else 0.0

    # ---- 锚点敏感性 ----
    spread = {}
    for a in ("median", "ls", "low", "high"):
        rr = lnmag - I
        cc = (float(np.sum(rr) / rr.size) if a == "ls"
              else float(np.median(rr[:max(3, rr.size // 6)])) if a == "low"
              else float(np.median(rr[-max(3, rr.size // 6):])) if a == "high"
              else float(np.median(rr)))
        dd = np.expm1(rr - cc) * 100.0
        spread[a] = float(np.sqrt(np.mean(dd ** 2)))

    base.update(rms_rel=rms / 100.0, max_abs_rel=mx / 100.0, bands=bands,
                trend_pct_per_dec=trend, trend_r=tr, ac1=ac1,
                win=w, anchor_spread=spread,
                n=int(f.size), f_min=float(f.min()), f_max=float(f.max()),
                curve=dict(f=f.tolist(), delta=delta.tolist(),
                           phi_deg=np.degrees(phi).tolist(),
                           phi_raw_deg=np.degrees(phi_raw).tolist()))

    # ---- 分级 ----
    attr = []
    if smooth and w and w > 0 and f.size >= 10:
        # 平滑削掉了多少相位起伏（粗略：SG 残差 rms，度）
        cut = float(np.degrees(np.sqrt(np.mean((phi - phi_raw) ** 2))))
        attr.append(f"相位平滑窗 {w} 点（{w / f.size:.0%} 频点），"
                    f"被削掉的相位起伏 rms={cut:.2f}°")
    if not smooth:
        attr.append("未平滑：噪声会被积分累积，rms 偏大属预期")

    mid_bad = bands["mid"]["rms"] * 100 > RMS_OK if n_band == 3 else False
    gross = rms > RMS_FAIL * 100
    trend_bad = abs(trend) > TREND_FAIL
    trend_soft = abs(trend) > TREND_OK
    systematic = ac1 > 0.5

    for nm in ("low", "high"):
        b = bands[nm]
        if b["rms"] * 100 > RMS_FAIL * 100:
            attr.append(f"{nm} 段偏差 rms={b['rms'] * 100:.1f}%"
                        f"（{b['f_lo']:.3g}–{b['f_hi']:.3g} Hz），"
                        f"端部不自洽；一阶近似在端部最不可信，须结合 Lin-KK 归因")
    if abs(trend) > TREND_OK:
        attr.append(f"偏差随频率单调（{trend:+.1f} %/decade，r={tr:+.2f}）"
                    f"→ 形状性偏差，非随机噪声")
    if ac1 > 0.5:
        attr.append(f"偏差相邻点自相关 ac1={ac1:.2f} → 系统性")

    if gross or (trend_bad and systematic):
        why = ("总 rms 超阈" if gross else "趋势大且系统性")
        base.update(status="fail",
                    reason=f"Z-HIT 偏差 rms={rms:.1f}%（{why}）："
                           f"幅值形状与相位不自洽")
    elif rms > RMS_OK * 100 or (trend_soft and systematic):
        base.update(status="warn",
                    reason=f"Z-HIT 偏差 rms={rms:.1f}%、趋势 {trend:+.1f} %/decade："
                           f"高于噪声底但不能判定不自洽，需与 Lin-KK 并看")
    else:
        base.update(status="ok",
                    reason=f"Z-HIT 偏差 rms={rms:.1f}%、趋势 {trend:+.1f} %/decade"
                           f" → 幅值形状与相位自洽")
    base["attribution"] = attr
    return base


def win_sensitivity(f, Z, wins=(None, 7, 11, 15, 21)) -> list:
    """换平滑窗宽重跑，看判定是否翻转。

    不稳定的结论不该写进正文——这是本项目对"参数敏感性"的一贯口径。
    """
    out = []
    for w in wins:
        r = zhit(f, Z, win=w)
        out.append(dict(win=w, used=r.get("win"), status=r["status"],
                        rms=r["rms_rel"], trend=r["trend_pct_per_dec"]))
    return out


# --------------------------------------------------------------------------
# 合成标定
# --------------------------------------------------------------------------

def _z_debye(f, R_inf=1.0, R=1.0, f0=1.0):
    w = 2 * np.pi * f
    return R_inf + R / (1 + 1j * w / (2 * np.pi * f0))


def _z_zarc(f, R_inf=1.0, R=1.0, f0=1.0, phi=0.8):
    w = 2 * np.pi * f
    return R_inf + R / (1 + (1j * w / (2 * np.pi * f0)) ** phi)


def _z_hn(f, R_inf=1.0, w0=1.0, tau=1e-3, phi=0.8, zeta=1.0):
    u = 2 * np.pi * f * tau
    return R_inf + w0 / (1 + u ** phi * np.exp(0.5j * np.pi * phi)) ** zeta


def _z_rl(f, R_inf=1.0, R=0.5, tau=1e-3):
    """串联电感 + 电阻：纯感性高频端，用来测 Z-HIT 在 φ→+π/2 的表现。"""
    return R_inf + R + 1j * 2 * np.pi * f * (tau * R)


def _z_inductive_tail(f):
    """AZ31 那类"低频是感抗"的谱（用来测 Z-HIT 的非容性尾巴）。"""
    w = 2 * np.pi * f
    z = 5.0 + 40.0 / (1 + (1j * w / 0.05) ** 0.85) + 1j * w * 3e-6
    return z + 0.5 * 1j * w / (1 + 1j * w / 0.01)


def _noise(Z, rel, seed=0):
    rng = np.random.default_rng(seed)
    mag = np.abs(Z)
    n = rel * mag * (rng.normal(size=Z.size) + 1j * rng.normal(size=Z.size)) / np.sqrt(2)
    return Z + n


def _smooth_bias(Z, f, amp=0.05, width=1.0):
    """§19 用过的构造：沿 log f 的宽高斯**幅值**偏差（相对 |Z|.max）。"""
    u = np.log(f)
    u0 = np.median(u)
    g = np.exp(-0.5 * ((u - u0) / width) ** 2)
    return Z * (1 + amp * g)


def synth_suite(n=50, seed=0) -> list:
    """合成标定集：每种构造都拿 Z-HIT 跑一遍，给"噪声底"和"一阶近似误差"。"""
    f = np.logspace(5, -2, n)              # 100 kHz → 10 mHz，与实测同量级
    cases = []

    def add(name, Z, note=""):
        r = zhit(f, Z)
        cases.append(dict(case=name, note=note, status=r["status"],
                          rms=r["rms_rel"], trend=r["trend_pct_per_dec"],
                          ac1=r["ac1"],
                          bands={k: v["rms"] for k, v in (r["bands"] or {}).items()}))

    add("R_only", np.full(f.size, 10.0) + 0j, "纯电阻：Z-HIT 应精确")
    add("debye", _z_debye(f, 1.0, 1.0, 100.0), "单弛豫，最窄分布 → 一阶近似误差最大")
    add("zarc_0.9", _z_zarc(f, 1.0, 1.0, 100.0, 0.9), "近 Debye")
    add("zarc_0.7", _z_zarc(f, 1.0, 1.0, 100.0, 0.7), "中等展宽")
    add("zarc_0.5", _z_zarc(f, 1.0, 1.0, 100.0, 0.5), "宽带")
    add("hn_2comp", _z_hn(f, 1.0, 0.5, 1e-2, 0.8)
        + _z_hn(f, 0.0, 0.5, 1e-4, 0.8, 0.8), "两个 HN 串联（接近我们的模型族）")
    add("RL_inductive", _z_rl(f), "串联电感主导高频")
    add("ind_tail", _z_inductive_tail(f), "低频感抗尾（AZ31 型）")

    zz = _z_zarc(f, 1.0, 1.0, 100.0, 0.7)
    for rel in (0.002, 0.005, 0.01, 0.02):
        add(f"zarc0.7_noise{rel:.1%}", _noise(zz, rel, seed), "宽带 + 白噪声 → 噪声底")
    for amp in (0.02, 0.05, 0.10):
        add(f"zarc0.7_magbias{amp:.0%}", _smooth_bias(zz, f, amp, 1.0),
            "§19 型宽幅值偏差（5% 曾复现 acf 超标）")
    add("zarc0.7_phase_1deg",
        zz * np.exp(1j * np.radians(1.0)),
        "整体 1° 相位旋转（通道相位误差）")
    add("zarc0.7_phase_2deg",
        zz * np.exp(1j * np.radians(2.0)), "整体 2° 相位旋转")
    add("debye_noise1%", _noise(_z_debye(f, 1.0, 1.0, 100.0), 0.01, 7),
        "窄分布 + 噪声：一阶误差与噪声叠加")
    return cases


def format_synth(cases: list) -> str:
    lines = [f"{'case':24s} {'rms%':>7s} {'trend%/dec':>10s} {'ac1':>6s}  status"]
    lines.append("-" * 62)
    for c in cases:
        lines.append(f"{c['case']:24s} {c['rms'] * 100:7.2f} "
                     f"{c['trend']:10.2f} {c['ac1']:6.2f}  {c['status']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Z-HIT 一阶形式自洽检验（原型）")
    ap.add_argument("path", nargs="?", help="谱文件（csv/txt）")
    ap.add_argument("--synth", action="store_true", help="跑合成标定集")
    ap.add_argument("--win", type=int, default=None, help="相位平滑窗（奇数点）")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--sens", action="store_true", help="窗宽敏感性")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    if a.synth:
        cases = synth_suite()
        print(format_synth(cases))
        if a.json:
            Path(a.json).write_text(json.dumps(cases, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        return 0

    if not a.path:
        ap.error("需要文件路径，或用 --synth")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bm_real import load_spectrum
    f, Z = load_spectrum(a.path)
    r = zhit(f, Z, win=a.win, smooth=not a.no_smooth)

    print(f"\nZ-HIT（一阶，v{VERSION}）  {Path(a.path).name}")
    print(f"  {r['status'].upper()} — {r['reason']}")
    if r["bands"]:
        for k, b in r["bands"].items():
            print(f"  {k:4s} {b['f_lo']:9.3g}–{b['f_hi']:9.3g} Hz  "
                  f"rms={b['rms'] * 100:6.2f}%  mean={b['mean'] * 100:+6.2f}%")
    for t in r["attribution"]:
        print(f"  · {t}")
    if a.sens:
        print("\n  窗宽敏感性：")
        for s in win_sensitivity(f, Z, wins=(7, 11, 15, 21)):
            print(f"    win={s['win']} → {s['status']:4s} "
                  f"rms={s['rms'] * 100:5.2f}% trend={s['trend']:+6.2f}")
    if a.json:
        out = {k: v for k, v in r.items() if k != "curve"}
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                          default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
