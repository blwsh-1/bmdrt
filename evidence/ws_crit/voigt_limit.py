"""K 个 R∥C（Voigt / CPE-ζ=1）能多接近有限长扩散？——Montella 2020 的定量落地。

Montella 2020（J. Electroanal. Chem. 879:114785）：
  · tanh(z)/z = 8 Σ_{k≥1} 1/[(2k−1)²π² + 4z²]  （式 7，部分分式）
  · 「无限级 Voigt 是有限长扩散（透射/transmissive）的精确表示」
  · τ1 要 n≈6 才收敛；τ2 在 n=6 仍有 ~5% 偏差（其 Fig.6，无噪声数据）

本项目：K 个 R∥CPE 串联（ζ=1 即 R∥C），扩散用 coth（阻挡型 Wo）或 tanh（Wshort）。
  · tanh 无 1/z² 项 → 纯 Voigt → K 级可逼近
  · coth 含 1/z² 串联电容项 → 纯 R∥C 结构上补不上

两套误差口径（**不能混用**，见 WP2_REPORT.md §3.6 的更正说明）：
  ① `rel`：|Z_Voigt − Z_W| / |Z_W|（相对**元件本身**）——只描述形状，不能与整谱噪声比；
  ② `abs`：|Z_Voigt − Z_W| / R_W（无量纲，× 实际 R_W 得 Ω）——这才是能与 σ_im 比的口径。
本脚本主输出为 ②，① 仅作形状参照。
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from scipy.optimize import least_squares, minimize

STEM_B = "Blank_0.1KBr_pH10_EIS_PtCu_1.2V"
NPZ = f"out/_wp2_blank/k2_none/{STEM_B}/real_{STEM_B}_samples.npz"
F = np.sort(np.load(NPZ, allow_pickle=True)["f"])
F_E = np.sort(np.load("out/_wp2_blank/k2_none/" + STEM_B +
                      f"/real_{STEM_B}_samples.npz", allow_pickle=True)["f"])

# 各臂实测参数（json 里 diffusion_params / sigma_im）
ARMS = [
    # tag, 谱, R_W, τ_D, σ_im
    ("Blank k2_Wo", "blank", 0.0611, 0.4952, 0.0992),
    ("Blank k1_Wo", "blank", 12.362, 9.1729, 0.2960),
    ("caseE k1_Wo", "caseE", 23.965, 137.39, 0.1899),
    ("caseE k2_Wo", "caseE", 0.0793, 0.7098, 0.1894),
]


def freq(which):
    if which != "caseE":
        return F
    p = "out/_wp2_caseE/k2_Wo/example_caseE/real_example_caseE_samples.npz"
    return np.sort(np.load(p, allow_pickle=True)["f"])


def z_flw(kind, f, td):
    z = np.sqrt(1j * 2 * np.pi * f * td)
    th = np.ones_like(z)                   # |z|>30 时 tanh→1，避开 cosh 溢出
    m = np.abs(z) <= 30
    th[m] = np.tanh(z[m])
    if kind == "tanh":                     # 透射 / Wshort
        return th / z
    return 1.0 / th / z                    # 阻挡 / Wo


def voigt(f, p, K):
    tau = np.exp(np.clip(p[:K], -50, 50))
    R = np.exp(np.clip(p[K:], -50, 50))
    w = 2 * np.pi * f
    return (R[None, :] / (1 + 1j * np.outer(w, tau))).sum(1)


def _l2(kind, K, f, td, tries, seed):
    """以 |ΔZ|/R_W（R_W=1）为残差做 L2 拟合。"""
    Zt = z_flw(kind, f, td)
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        lt = rng.uniform(np.log(1 / (2 * np.pi * f.max() * 20)),
                         np.log(1 / (2 * np.pi * f.min() / 20)), K)
        p0 = np.concatenate([lt, np.full(K, np.log(max(abs(Zt[0]), 1e-3)))])
        try:
            r = least_squares(
                lambda p: np.concatenate([(voigt(f, p, K) - Zt).real,
                                          (voigt(f, p, K) - Zt).imag]),
                p0, method="lm", max_nfev=3000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x if best is not None else None


def fit_abs(kind, K, f, td, tries=20, seed=0):
    """L2 起点 + Nelder-Mead 打磨 → 近似 minimax 的 max|ΔZ|/R_W（无量纲）。

    返回 (max_abs_per_Rw, max_rel, tau, R)。
    """
    Zt = z_flw(kind, f, td)
    x = _l2(kind, K, f, td, tries, seed)
    if x is None:
        return np.nan, np.nan, None, None
    obj = lambda p: np.abs(voigt(f, p, K) - Zt).max()
    r = minimize(obj, x, method="Nelder-Mead",
                 options=dict(maxiter=20000, xatol=1e-10, fatol=1e-12))
    p = r.x if r.fun <= obj(x) else x
    d = np.abs(voigt(f, p, K) - Zt)
    return d.max(), (d / np.abs(Zt)).max(), np.exp(p[:K]), np.exp(p[K:])


def main():
    print("=" * 78)
    print("① 形状口径（相对元件本身，**不可**与整谱噪声比）")
    print("=" * 78)
    td0 = 0.4952
    print(f"τ_D = {td0} s（f_D = {1/(2*np.pi*td0):.3f} Hz），窗 {F.min():.3g}–{F.max():.3g} Hz，"
          f"{len(F)} 点，R_W=1\n")
    print(f"{'kind':6s} {'K':>2s} {'max |ΔZ|/R_W':>13s} {'max rel':>9s}")
    abs_tab = {}
    for kind in ("tanh", "coth"):
        for K in (1, 2, 3, 4, 6, 8, 10, 12):
            a, rel, _, _ = fit_abs(kind, K, F, td0)
            abs_tab[(kind, K)] = a
            print(f"{kind:6s} {K:2d} {a:13.4f} {rel*100:8.2f}%")

    print()
    print("=" * 78)
    print("② 绝对口径：ε_abs = R_W × (max|ΔZ|/R_W)，与各臂 σ_im 比（**这才是可比的**）")
    print("=" * 78)
    print(f"{'臂':13s} {'R_W/Ω':>8s} {'τ_D/s':>8s} {'K':>2s} {'f':>6s} "
          f"{'ε_abs/Ω':>9s} {'ε/σ':>7s}  判定")
    for tag, which, Rw, td, sig in ARMS:
        f = freq(which)
        for K in (2,):
            a, _, _, _ = fit_abs("coth", K, f, td)
            eps = Rw * a
            verdict = "落进噪声（不可分辨）" if eps < sig else f"可分辨（{eps/sig:.1f}σ）"
            print(f"{tag:13s} {Rw:8.3f} {td:8.3f} {K:2d} {len(f):6d} "
                  f"{eps:9.4f} {eps/sig:7.2f}  {verdict}")

    print()
    print("=" * 78)
    print("③ 要在 σ 以内表示一条满幅 FLW，需要几级？（min K s.t. ε_abs ≤ σ）")
    print("=" * 78)
    for tag, which, Rw, td, sig in ARMS:
        f = freq(which)
        need = sig / Rw                      # 容许的 max|ΔZ|/R_W
        hit = None
        for K in range(1, 13):
            a, _, _, _ = fit_abs("coth", K, f, td)
            if a <= need:
                hit = (K, a)
                break
        if hit:
            print(f"{tag:13s} R_W={Rw:8.3f}Ω σ={sig:.4f}Ω → 需 max|ΔZ|/R_W ≤ {need*100:6.3f}%；"
                  f"K={hit[0]} 达标（{hit[1]*100:.3f}%）")
        else:
            print(f"{tag:13s} R_W={Rw:8.3f}Ω σ={sig:.4f}Ω → 需 ≤ {need*100:6.3f}%；"
                  f"K≤12 仍未达标（K=12: {abs_tab[('coth', 12)]*100:.3f}%）")

    print()
    print("=" * 78)
    print("④ 结论（写进报告的口径）")
    print("=" * 78)
    a2 = abs_tab[("coth", 2)]
    print(f"K=2 的 max|ΔZ|/R_W（coth，τ_D={td0}）= {a2:.3f}")
    for tag, which, Rw, td, sig in ARMS[:1]:
        print(f"  → 对 Blank k2_Wo 的 R_W={Rw:.3f} Ω：ε_abs={Rw*a2:.4f} Ω，"
              f"σ_im={sig:.4f} Ω → ε/σ={Rw*a2/sig:.2f}")
    print("注意：σ_im 由各臂自估，K=1 臂的 σ 被差模型撑大（0.30/0.38），"
          "比较时应以最好模型的 σ（≈0.10）为准，见报告 §3.6 脚注。")


if __name__ == "__main__":
    main()
