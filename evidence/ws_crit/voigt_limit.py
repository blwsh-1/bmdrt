"""K 个 R∥C（Voigt/CPE-ζ=1）能多接近有限长扩散？——Montella 2020 的定量落地。

Montella 2020（J. Electroanal. Chem. 879:114785）：
  · tanh(z)/z = 8 Σ_{k≥1} 1/[(2k−1)²π² + 4z²]  （式 7，部分分式）
  · 「无限级 Voigt 是有限长扩散（透射/transmissive）的精确表示」
  · τ1 要 n≈6 才收敛；τ2 在 n=6 仍有 ~5% 偏差（其 Fig.6，无噪声数据）

本项目：K 个 R∥CPE 串联（ζ=1 即 R∥C），默认扩散用 **coth**（阻挡型 Wo）。
  · tanh（= 项目 Wshort）无 1/z² 项 → 纯 Voigt → K 级可逼近
  · coth（= 项目 Wo）含 1/z² 串联电容项 → 纯 R∥C 结构上补不上

本脚本：在真实谱的 50 点频窗、τ_D=0.495 s 下，用最优重拟合算 K=1..6 的
最大相对误差，与噪声水平（σ_im/|Z|）对照。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from scipy.optimize import least_squares

TAU_D = 0.4952
F = np.load("out/_wp2_blank/k2_none/Blank_0.1KBr_pH10_EIS_PtCu_1.2V/"
            "real_Blank_0.1KBr_pH10_EIS_PtCu_1.2V_samples.npz", allow_pickle=True)["f"]
F = np.sort(F)


def z_flw(kind, f, td=TAU_D):
    z = np.sqrt(1j * 2 * np.pi * f * td)
    if kind == "tanh":                     # 透射 / Wshort
        return np.tanh(z) / z
    return 1.0 / np.tanh(z) / z            # 阻挡 / Wo


def voigt(f, p, K):
    tau = np.exp(p[:K])
    R = np.exp(p[K:])
    w = 2 * np.pi * f
    return (R[None, :] / (1 + 1j * w[:, None] * tau[None, :])).sum(1)


def fit(kind, K, tries=30, seed=0):
    Zt = z_flw(kind, F)
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        lt = np.log(np.exp(rng.uniform(np.log(1 / (2 * np.pi * F.max() * 20)),
                                      np.log(1 / (2 * np.pi * F.min() / 20)), K)))
        p0 = np.concatenate([lt, np.full(K, np.log(max(abs(Zt[0]), 1e-3)))])
        try:
            r = least_squares(
                lambda p: np.concatenate([
                    ((voigt(F, p, K) - Zt) / np.abs(Zt)).real,
                    ((voigt(F, p, K) - Zt) / np.abs(Zt)).imag]),
                p0, method="lm", max_nfev=4000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    rel = np.abs(voigt(F, best.x, K) - Zt) / np.abs(Zt)
    return rel.max(), rel.mean(), np.exp(best.x[:K]), np.exp(best.x[K:])


def main():
    print(f"τ_D = {TAU_D} s  (f_D = {1/(2*np.pi*TAU_D):.3f} Hz)，窗 {F.min():.3g}–{F.max():.3g} Hz，"
          f"{len(F)} 点")
    print("目标：Z_W = tanh(z)/z（Wshort，透射）或 coth(z)/z（Wo，阻挡）；归一化 R_W=1\n")
    print(f"{'kind':6s} {'K':>2s} {'max rel':>9s} {'mean rel':>9s}   时间常数（s）")
    for kind in ("tanh", "coth"):
        for K in (1, 2, 3, 4, 6):
            mx, mn, tau, R = fit(kind, K)
            ts = " ".join(f"{t:.3g}" for t in np.sort(tau))
            print(f"{kind:6s} {K:2d} {mx*100:8.2f}% {mn*100:8.2f}%   {ts}")

    print("\n噪声水平参照：本谱 σ_im=0.098 Ω，0.1 Hz 处 |Z|=57.45 Ω → σ/|Z| ≈ 0.17%")
    print("             低频 1 Hz 处 |Z|≈50 Ω → ≈0.2%")


if __name__ == "__main__":
    main()
