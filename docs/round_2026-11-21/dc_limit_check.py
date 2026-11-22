"""核实 adr / nested(parallel) / nested(series) 的直流（omega->0）极限。

动机：TOPOLOGY_AUDIT.md 3.5 里写了
    adr（并联）  Z_DC = Rs + (Rct || R_ad)
但 Z_ad = R_ad*j*w*tau/(1+j*w*tau) 在 w->0 时 -> 0（短路），
所以 adr 的 Z_DC 应为 Rs，而不是 Rs + Rct||R_ad。此处用数值验算。
"""
import sys

import numpy as np

sys.path.insert(0, ".")
import bm_top as T  # noqa: E402

f = np.array([1e-6, 1e-5, 1e-4, 1e-2, 1.0, 1e3, 1e6])

# 参数：让三条支路都有可比的量级
base = dict(Rs=50.0, R1=100.0, Q1=100.0, n1=0.85,
            R_ad=100.0, tau_ad=1e-3, L_ad=0.1, R_L=100.0)

zp = T.z_adr(f, dict(base, Q=base["Q1"], n=base["n1"], Rct=base["R1"]),
             include_L=False)
zn = T.z_nested(f, base, branch="parallel", layers=1, include_L=False)
zs = T.z_nested(f, base, branch="series", layers=1, include_L=False)

print("%-10s %14s %14s %14s" % ("f / Hz", "adr", "nested P", "nested S"))
for w, a, b, c in zip(f, zp, zn, zs):
    print("%-10.0e %14.4f %14.4f %14.4f" % (w, a.real, b.real, c.real))

print("\n各结构的直流极限（外推）")
print("  adr        Re -> %.4f  (= Rs，因为 Z_ad -> 0 短路整个并联节点)" % zp[0].real)
print("  nested P   Re -> %.4f  (= Rs + R1，因为 Z_ad -> 0 后剩 R1)" % zn[0].real)
print("  nested S   Re -> %.4f  (= Rs + R1 + R_L，内部串联不通直流)" % zs[0].real)
print("  |Ys|/w 在 f=1e-6 的虚部：adr %.4f / P %.4f / S %.4f"
      % (zp[0].imag, zn[0].imag, zs[0].imag))
