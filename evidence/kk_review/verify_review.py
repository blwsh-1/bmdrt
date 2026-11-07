# -*- coding: utf-8 -*-
"""逐条核实外部审查意见（只读，不改任何代码/结果）。

意见 1：残差是"绝对残差"，低阻体系会被高频噪声主导 → 建议改相对残差。
意见 2：用一堆 RC 拟感抗造成参数污染，M 被强行加到 9 才通过；
        建议开 impedancefitter 的 inductance=True（串联 L），M=5/6 就能到 1%。
意见 3：护栏常数（(n-2)//2、0.02·M、残差放大 3 倍停）评价。

用法（在仓库任意位置均可）：

    D:\\autoeis_env\\Scripts\\python.exe evidence/kk_review/verify_review.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # evidence/kk_review/ → 仓库根
sys.path.insert(0, str(ROOT))
import numpy as np
import bm_real as R
import bm_kk as K

f, Z = R.load_spectrum(str(ROOT / "archive" / "2EMC1-2_0750mA.csv"))
n = f.size
zabs = np.maximum(np.abs(Z), K.EPS)

print("\n" + "=" * 74)
print("【意见 1】残差口径：绝对还是相对？")
print("=" * 74)
kk = K.lin_kk(f, Z)
err = kk["Z_fit"] - Z                      # 绝对残差（Ω）
rel = err / zabs                           # 相对残差
print(f"  bm_kk.lin_kk 返回的 res_re/res_im 量级：{np.abs(kk['res_re']).max():.4f}")
print(f"  绝对残差 |ΔZ| 量级：            {np.abs(err).max():.6g} Ω")
print(f"  相对残差 |ΔZ|/|Z| 量级：        {np.abs(rel).max():.4f}")
print(f"  → 两者相差 {np.abs(err).max()/max(np.abs(rel).max(),1e-300):.4g} 倍"
      f" = |Z| 的典型量级 {np.median(np.abs(Z)):.4g} Ω")
print(f"  结论：lin_kk 内部 err=(Z-Z_fit)/zabs（bm_kk.py:170），"
      f"返回的 res_* 是**相对**残差")
r_abs = float(np.sqrt(np.mean(np.abs(err) ** 2)))
r_rel = float(np.sqrt(np.mean(np.abs(rel) ** 2)))
print(f"  rms 绝对 = {r_abs:.4g} Ω；rms 相对 = {r_rel:.2%}"
      f"  → 选阶用的 r_all 是相对口径（{kk['M']} 阶时）")

print("\n" + "=" * 74)
print("【意见 2】Lin-KK 里到底有没有串联电感？M 是不是被强行加到 9？")
print("=" * 74)
print(f"  本次 M = {kk['M']}，M_scanned = {kk['M_scanned']}，M_hi = {kk['M_hi']}")
print(f"  L_fit = {kk['L_fit']:.4g} H  ← fit_linKK 的设计矩阵自带串联电感列")
from impedance.validation import fit_linKK
import inspect
src = inspect.getsource(fit_linKK)
for line in src.splitlines():
    if "series inductance" in line or "plus 1 series" in line:
        print(f"  库源码原文：{line.strip()}")
print(f"  KK 门禁结论：status = "
      f"{K.kk_check(f, Z)['status']}（**没有通过**，不是'被 M 压到通过'）")

print("\n  逐阶扫描（看 M 加大能否把中段残差压到 1% 以下）：")
print(f"  {'M':>3} {'rms_all(相对)':>13} {'中段':>9} {'高频':>9} {'低频':>9} "
      f"{'L_fit/H':>11} {'mu':>8}")
import warnings
from impedance.validation import get_tc_distribution
m_hi = int(min(50, max(4, (n - 2) // 2)))
for M in range(1, m_hi + 1):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = get_tc_distribution(f, M)
            el, mu = fit_linKK(f, ts, M, Z, "complex", False)
        zf = K._eval_linkk(el, ts, f)
        e = (zf - Z) / zabs
        lf = np.log10(f)
        q1, q2 = np.quantile(lf, [1 / 3, 2 / 3])
        lo, mid, hi = lf <= q1, (lf > q1) & (lf <= q2), lf > q2
        rms = lambda m: float(np.sqrt(np.mean(np.abs(e[m]) ** 2)))
        print(f"  {M:>3} {rms(np.ones(n,bool)):>13.2%} {rms(mid):>9.2%} "
              f"{rms(hi):>9.2%} {rms(lo):>9.2%} {float(el[-1]):>11.3g} "
              f"{mu:>8.3f}")
    except Exception as ex:
        print(f"  {M:>3} 病态跳过（{type(ex).__name__}）")

print("\n" + "=" * 74)
print("【意见 3】护栏常数与选阶规则")
print("=" * 74)
print(f"  m_hi = min(max_M=50, max(4,(n-2)//2)) = {m_hi}（n={n}）")
print(f"  选阶 = argmin r_all·(1+elbow·M)，elbow=0.02（bm_kk.py:189）")
print(f"  病态护栏 = r_all > 3×已见最优 即停（blowup=3.0，bm_kk.py:175）")
print(f"  停滞护栏 = M≥8 且连续 5 阶改善<2% 即停（bm_kk.py:180-183）")
print(f"  → 意见里说的 0.02·M / 放大 3 倍 / (n-2)//2 三条，与代码一致")
