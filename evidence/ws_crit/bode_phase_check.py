# -*- coding: utf-8 -*-
"""Bode 相位能不能当 Ws / Wo / Wshort 的判据？

背景（用户 2026-11-09 15:09 提出）：低频端相位趋近的极限不同——
趋 0° 是纯电阻、趋 -90° 是电容，比 Nyquist 几何形状更容易量化。

先说清元件归属（`bm_diff.py`）：
  · `Ws`   半无限      A/sqrt(jω)                    相位恒 -45°，**没有低频极限**
  · `Wo`   有限长阻挡  R·coth(z)/z, z=sqrt(jωτ)      ωτ≪1 → 电容，相位 → -90°
  · `Wshort` 有限长透射 R·tanh(z)/z, z=sqrt(jωτ)      ωτ≪1 → 纯电阻，相位 → 0°
所以「趋 0°」是 Wshort，「趋 -90°」是 Wo；Ws 不在这个谱系的两端。

本脚本量化三件事：
  A 三类元件在三种转折位置（窗内 / 窗口下沿 / 窗外下部）的相位与 log(-Im) 斜率；
  B **同一条 Ws** 叠加不同 `R_inf` 后相位怎么变（slope_im 应完全不动）；
  C 1% 相对噪声下两者的离散度；
  D 18 条真实 AZ31 谱的低频相位（对照：它们本来就是感抗尾）。
"""
import io
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import bm_ws
from bm_diff import _coth_over_z

RNG = np.random.default_rng(20261109)
F = np.logspace(np.log10(0.01), 5, 41)      # 0.01 – 1e5 Hz，41 点


def zws(f, A):
    return A / np.sqrt(1j * 2 * np.pi * np.asarray(f, float))


def zwo(f, R, tau):
    z = np.sqrt(1j * 2 * np.pi * np.asarray(f, float) * tau)
    return R * _coth_over_z(z)


def zshort(f, R, tau):
    z = np.sqrt(1j * 2 * np.pi * np.asarray(f, float) * tau)
    return R * np.tanh(z) / z


def phase_deg(Z):
    return np.degrees(np.angle(Z))


def slope_im(f, Z):
    o = np.argsort(f)
    f, Z = np.asarray(f)[o], np.asarray(Z)[o]
    si = bm_ws.slope_im(f, Z)
    return si["slope"] if si.get("usable") else float("nan")


def noise(Z, rel=0.01, rng=None):
    rng = rng or RNG
    s = rel * np.abs(Z)
    return Z + rng.normal(0, s) + 1j * rng.normal(0, s)


def main():
    out = {}

    # ---------------- A 三类 × 三种转折位置 ----------------
    print("=" * 78)
    print("A  三类元件：低频相位 vs log(-Im) 斜率（R_inf = 0，无噪声）")
    print("    f_D = 1/(2πτ) 是转折频率；窗口 0.01–1e5 Hz")

    A_W, R_W = 2.0, 100.0
    cases = []
    for fD, name in ((1e-4, "转折在窗外下部"), (1e-2, "转折在窗口下沿"),
                     (1e2, "转折在窗内上部")):
        tau = 1.0 / (2 * np.pi * fD)
        cases.append((f"Ws   {name}", zws(F, A_W)))
        cases.append((f"Wo   {name}", zwo(F, R_W, tau)))
        cases.append((f"Wshort {name}", zshort(F, R_W, tau)))

    print(f"\n{'case':26s} {'ph(f_min)':>10s} {'slope_im':>9s} "
          f"{'bm_ws 判的档':>14s}")
    rowsA = []
    for tag, Z in cases:
        ph = float(phase_deg(Z)[0])
        tm = bm_ws.terminal(F, Z)
        r = dict(tag=tag, ph_min=ph, slope_im=slope_im(F, Z),
                 term=tm["label"], se=tm["se"])
        rowsA.append(r)
        print(f"{tag:26s} {ph:10.1f} {r['slope_im']:9.3f} {r['term']:>14s}")
    out["A"] = rowsA

    # ---------------- B R_inf 稀释 ----------------
    print("\n" + "=" * 78)
    print("B  同一条真 Ws，叠加不同 R_inf（相位被拉向 0°，还是指数变了？）")
    Zw = zws(F, A_W)
    zref = abs(Zw[0])
    print(f"\n{'R_inf/|Z_W(f_min)|':>20s} {'R_inf (Ω)':>10s} {'ph(f_min)°':>11s} "
          f"{'slope_im':>9s}  phase 会掉进 Wshort 的判定区？")
    rowsB = []
    for ratio in (0.0, 0.3, 1.0, 3.0, 10.0, 50.0):
        Rinf = ratio * zref
        Z = Rinf + Zw
        ph = phase_deg(Z)[0]
        s = slope_im(F, Z)
        rowsB.append(dict(ratio=ratio, R_inf=float(Rinf), ph_min=float(ph),
                          slope_im=s))
        flag = "是（|ph|<15°）" if abs(ph) < 15 else ""
        print(f"{ratio:20.1f} {Rinf:10.1f} {ph:11.1f} {s:9.3f}  {flag}")
    out["B"] = rowsB

    # ---------------- C 1% 噪声离散度 ----------------
    print("\n" + "=" * 78)
    print("C  1% 相对噪声下，ph(f_min) 与 slope_im 的重复性（n=200）")
    print(f"\n{'case':22s} {'ph 均值°':>9s} {'ph σ°':>7s} {'slope 均值':>10s} "
          f"{'slope σ':>8s}")
    rowsC = []
    probes = [("Ws 纯（R_inf=0）", zws(F, A_W)),
              ("Ws + R_inf×3", 3 * zref + zws(F, A_W)),
              ("Wo 转折在窗内", zwo(F, R_W, 1 / (2 * np.pi * 1e2))),
              ("Wshort 转折在窗内", zshort(F, R_W, 1 / (2 * np.pi * 1e2)))]
    for tag, Z in probes:
        phs, sls = [], []
        for _ in range(200):
            Zn = noise(Z)
            phs.append(phase_deg(Zn)[0])
            sls.append(slope_im(F, Zn))
        r = dict(tag=tag, ph_mean=float(np.mean(phs)), ph_sd=float(np.std(phs)),
                 sl_mean=float(np.nanmean(sls)), sl_sd=float(np.nanstd(sls)))
        rowsC.append(r)
        print(f"{tag:22s} {r['ph_mean']:9.1f} {r['ph_sd']:7.2f} "
              f"{r['sl_mean']:10.3f} {r['sl_sd']:8.3f}")
    out["C"] = rowsC

    # ---------------- D 真实谱 ----------------
    print("\n" + "=" * 78)
    print("D  18 条真实 AZ31 的低频相位（它们本来就是感抗尾，不是扩散尾）")
    rowsD = []
    bi = sorted(bm_real_csv())
    for tag, path in bi:
        f, Z = _load(path)
        if f is None:
            continue
        o = np.argsort(f)
        ph = phase_deg(Z[o])[0]
        w = bm_ws.ws_diagnostics(f, Z)
        tm = bm_ws.terminal(f, Z)
        rowsD.append(dict(tag=tag, ph_min=float(ph), term=w["term"],
                          im_ratio=float(tm["im_ratio"])))
    phs = np.array([r["ph_min"] for r in rowsD])
    print(f"   n={len(rowsD)}  phase(f_min) 中位 {np.median(phs):+.1f}°，"
          f"范围 [{phs.min():+.1f}, {phs.max():+.1f}]°")
    print(f"   落在 |ph|<15°（会被 phase 规则当成 Wshort/纯电阻）的有 "
          f"{int((np.abs(phs) < 15).sum())}/{len(phs)} 条")
    print(f"   term 分布：" + ", ".join(
        f"{k}×{v}" for k, v in
        {t: sum(1 for r in rowsD if r["term"] == t) for t in
         {r["term"] for r in rowsD}}.items()))
    out["D"] = rowsD

    with io.open("evidence/ws_crit/bode_phase_check.json", "w",
                 encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n写出 evidence/ws_crit/bode_phase_check.json")


def bm_real_csv():
    import glob
    import os
    for p in sorted(glob.glob("out/_m6_samples/z_csv/*.csv")):
        yield os.path.basename(p)[:-4], p


def _load(path):
    import bm_real
    try:
        f, Z = bm_real.load_spectrum(path)
        return np.asarray(f, float), np.asarray(Z, complex)
    except Exception:                                       # noqa: BLE001
        return None, None


if __name__ == "__main__":
    main()
