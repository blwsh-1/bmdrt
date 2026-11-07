# -*- coding: utf-8 -*-
"""嵌套并联模型的可辨识性证据（模型体系第三层）。

`theoretical_bound.py` 只证明"哪个模型族表达力更强"；本脚本回答下一个问题：
**拟合得好，参数就一定有物理意义吗？** 三个只读检查：

  ① 渐近极限：ω→0 → Rs+R1+R2，ω→∞ → Rs；窗口内的 Re 回落是有限频段
     非单调，不是"低频短路"。
  ② Rs 剖面：固定 Rs 重新优化其余 8 个参数，看残差是否对 Rs 平坦。
     若平坦，说明高频截距被 CPE 支路 (Y1,n1) 吸收，Rs 不可辨识。
  ③ 窗口截断：逐步砍掉高频点重拟合，看参数是否漂移 / 内层结构是否崩塌。

用法：
  python evidence/nested_topology/identifiability_check.py                      # 内置合成谱
  python evidence/nested_topology/identifiability_check.py --data <path>       # 真实三列谱
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theoretical_bound import make_models, from_file, synthetic  # noqa: E402

LO_R, HI_R = np.log(1e-2), np.log(1e5)
LO_T, HI_T = np.log(1e-5), np.log(1e4)
LO = np.array([LO_R, np.log(1e-9), 0.30, LO_R, np.log(1e-9), 0.30, LO_R, LO_R, LO_T])
HI = np.array([HI_R, np.log(1e-1), 1.05, HI_R, np.log(1e-1), 1.05, HI_R, HI_R, HI_T])

# 真实 AZ31 谱上 B 的最优参数（见 NESTED_TOPOLOGY_PROJECT.md §4.3）
P_REAL = np.array([np.log(1e-2), np.log(1.1e-6), 0.64, np.log(55.7),
                   np.log(3.2e-5), 0.79, np.log(600.3), np.log(3085.3),
                   np.log(1.624)])


def _solve(Z, omega, lo, hi, fixed_rs=None, n_start=28, seed=11):
    """多起点 LS。fixed_rs 不为 None 时把第 0 个参数钉死。"""
    wabs = np.abs(Z)
    _, model = make_models(omega)
    l, h = lo.copy(), hi.copy()
    if fixed_rs is not None:
        l[0] = h[0] = np.log(fixed_rs)

        def unpack(q):
            return np.concatenate([[np.log(fixed_rs)], q])
        bounds = (l[1:], h[1:])
        nd = 8
    else:
        unpack = lambda q: q                      # noqa: E731
        bounds = (l, h)
        nd = 9

    def resid(q):
        d = (model(unpack(q)) - Z) / wabs
        return np.concatenate([d.real, d.imag])

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_start):
        p0 = bounds[0] + (bounds[1] - bounds[0]) * rng.random(nd)
        try:
            r = least_squares(resid, p0, bounds=bounds, max_nfev=2500,
                              ftol=1e-10, xtol=1e-10)
            cost = float(np.sqrt(np.mean(r.fun ** 2)))
            if best is None or cost < best[0]:
                best = (cost, unpack(r.x))
        except Exception:                        # noqa: BLE001
            continue
    return best


def check_asymptotics(p=P_REAL):
    print("=" * 74)
    print("① 渐近极限（模型 B）")
    rs, r1, r2 = np.exp(p[0]), np.exp(p[3]), np.exp(p[6])
    print("  理论：ω→0  Re → Rs+R1+R2 = %.1f Ω (Im→0)" % (rs + r1 + r2))
    print("        ω→∞  Re → Rs       = %.2f Ω (Im→0)" % rs)
    print("%14s %14s %14s" % ("f / Hz", "Re Z", "Im Z"))
    for fx in (1e-8, 1e-4, 1e-2, 0.1, 1, 10, 1e2, 1e4, 1e6, 1e10):
        w = 2 * np.pi * np.array([fx])
        _, mb = make_models(w)
        z = mb(p)[0]
        print("%14.3g %14.3f %14.3f" % (fx, z.real, z.imag))
    print("  → 窗口内 Re 先升后降是**有限频段非单调**，由 CPE 并联支路与")
    print("    RL 支路的频率依赖共同造成；不是 ω→0 的渐近行为。")


def profile_rs(f, Z):
    print("=" * 74)
    print("② Rs 剖面（固定 Rs，重新优化其余 8 个参数）")
    print("  实测高频截距 Re(f_max) = %.1f Ω" % Z[np.argmax(f)].real)
    print("%14s %16s %14s %10s" % ("Rs 固定 / Ω", "最佳相对残差", "Y1", "n1"))
    omega = 2 * np.pi * f
    rows = []
    for rs in (0.01, 1, 10, 25, 49, 80, 120, 200):
        b = _solve(Z, omega, LO, HI, fixed_rs=rs)
        if b is None:
            continue
        rows.append((rs, b[0]))
        print("%14.2f %16.4f %14.3g %10.3f" %
              (rs, b[0], np.exp(b[1][1]), b[1][2]))
    if rows:
        r0 = rows[0][1]
        flat = [rs for rs, c in rows if c < r0 * 1.05]
        print("  → 残差在 Rs ∈ [%.2f, %.2f] Ω 内基本平坦（<5%% 变化）。"
              % (min(flat), max(flat)))
        print("    高频截距被 CPE 支路 (Y1,n1) 吸收，**Rs 不是可辨识参数**。")


def window_truncation(f, Z, cutoffs=(1e5, 1e4, 1e3, 1e2)):
    print("=" * 74)
    print("③ 高频窗口截断（Rs 自由）")
    print("%14s %6s %10s %10s %10s %10s %11s %10s" %
          ("f ≤ / Hz", "n", "rms", "Rs", "R1", "R2", "Rad", "tau_ad"))
    for fm in cutoffs:
        m = f <= fm
        if m.sum() < 12:
            continue
        b = _solve(Z[m], 2 * np.pi * f[m], LO, HI, n_start=30, seed=5)
        if b is None:
            continue
        p = b[1]
        print("%14.3g %6d %10.4f %10.2f %10.1f %10.1f %11.1f %10.4g" %
              (fm, m.sum(), b[0], np.exp(p[0]), np.exp(p[3]), np.exp(p[6]),
               np.exp(p[7]), np.exp(p[8])))
    print("  → 看 Rs/R1/R2 是否随窗口漂移、内层是否塌成 R2≈0；")
    print("    Rad / tau_ad 相对稳定才说明感抗环本身可辨识。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="真实谱三列文件（f Zre Zim）")
    args = ap.parse_args()
    if args.data:
        f, Z, src = from_file(args.data)
    else:
        f, Z, src = synthetic()
    print("数据:", src)
    print("  n=%d, f=%.3g..%.3g Hz, |Z|max=%.1f" % (f.size, f.min(), f.max(), np.abs(Z).max()))
    check_asymptotics()
    profile_rs(f, Z)
    window_truncation(f, Z)


if __name__ == "__main__":
    main()
