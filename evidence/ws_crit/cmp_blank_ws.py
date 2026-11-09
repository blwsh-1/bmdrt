"""把 Blank_0.1KBr_pH10_EIS_PtCu_1.2V 的两份后验放在一起：K=2(无扩散) vs K=1+Ws。

命令行的 `--krange 1,1` 只跑一个模型，`select_k()` 里没有第二个可比，所以
json 里的 elpd_diff/dse 是占位。这里从两份 samples.npz 重算逐点 elpd 再配对相减，
并回答「φ=0.5 的 CPE 是不是就是一个 Ws」这个具体问题。

    python evidence/ws_crit/cmp_blank_ws.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bm_real as R          # noqa: E402

OUT = ROOT / "out"
STEM = "Blank_0.1KBr_pH10_EIS_PtCu_1.2V"
A_DIR = OUT / STEM                                        # K=2, diffusion=none
B_DIR = OUT / "_blank_wsR1Q1" / STEM                      # K=1, +Ws
C_DIR = OUT / "_blank_wsR1Q1_wo" / STEM                   # K=1, +Wo（对照）


def load(d: Path):
    z = np.load(d / f"real_{STEM}_samples.npz", allow_pickle=True)
    s = {k: z[k] for k in z.keys()}
    f, Z = s["f"], s["Zre"] + 1j * s["Zim"]
    ll = R.loglik_matrix(s, f, Z, True)
    _, elpd, se, bad_k, pt, _ = R.loo_pointwise(ll)
    return dict(s=s, f=f, Z=Z, elpd=elpd, se=se, bad_k=bad_k, pt=pt)


def med(s, k, j=None):
    return float(np.median(s[k])) if j is None else float(np.median(s[k][:, j]))


def main():
    A, B = load(A_DIR), load(B_DIR)
    C = load(C_DIR)
    f = A["f"]

    print("=== 1. 单模型 elpd_loo（同一条谱、同一批 50 点）===")
    for name, x in (("K=2  L-R∞-2×(R∥CPE)     ", A),
                    ("K=1  L-R∞-(R∥CPE)-Ws   ", B),
                    ("K=1  L-R∞-(R∥CPE)-Wo   ", C)):
        print(f"  {name}  elpd={x['elpd']:8.2f} ± {x['se']:5.2f}"
              f"   Pareto-k>0.7: {x['bad_k'] * 100:.0f}%")

    print("\n=== 2. 配对差（逐点相减；正值 = 前者更好）===")
    tags = [("K=2", A), ("Ws", B), ("Wo", C)]
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            (na, xa), (nb, xb) = tags[i], tags[j]
            dd = xa["pt"] - xb["pt"]
            diff = float(dd.sum())
            dse = float(np.sqrt(dd.size) * np.std(dd, ddof=1))
            thr = max(2.0, 1.5 * dse)
            verdict = f"{na} 更好" if diff >= thr else (
                f"{nb} 更好" if -diff >= thr else "不可区分")
            print(f"  {na:4s} − {nb:4s}: Δ={diff:+8.2f}  dse={dse:5.2f}"
                  f"  z={abs(diff) / dse:4.1f}  → {verdict}")

    # ---------------- 3. φ=0.5 的 CPE 到底等不等价于 Ws ----------------
    sA = A["s"]
    print("\n=== 3. 「φ=0.5 的 CPE ≡ 半无限 Warburg」在本谱上的核算 ===")
    print("  恒等式本身对：Z_CPE(φ=0.5) = 1/[Q(jω)^0.5] = (1/Q)(jω)^-0.5 = Ws(A_W=1/Q)。")
    print("  但 K=2 的 #2 是 **R2 ∥ CPE2**，并联的 R2 把这个恒等式截断了。\n")
    pars = []
    for j in range(2):
        Rk, ph, ta = med(sA, "w", j), med(sA, "phi", j), med(sA, "tau", j)
        Q = ta ** ph / Rk
        pars.append((Rk, ph, ta, Q))
        print(f"  #{j + 1}: R={Rk:7.3f} Ω  φ={ph:.3f}  Q={Q:.4g} S·s^φ  "
              f"τ={ta:.3g} s  f_τ={1 / (2 * np.pi * ta):.4f} Hz")
    R2, ph2, ta2, Q2 = pars[1]
    aw_eq = 1.0 / Q2
    # 转折点：|Z_CPE| = R2  →  (1/(Q R2))^(1/φ) / 2π
    f_turn = (1.0 / (Q2 * R2)) ** (1.0 / ph2) / (2 * np.pi)
    print(f"  → 若把 #2 当成纯 Ws：A_W = 1/Q2 = {aw_eq:.3f} Ω·s^-0.5")
    print(f"  → 并联 R2 开始起作用（|Z_CPE|=R2）在 {f_turn:.4f} Hz，"
          f"窗口最低 {f.min():.3g} Hz（{np.log10(f_turn / f.min()):+.2f} dec，"
          f"{'窗内' if f_turn > f.min() else '窗外'}）")

    w = 2 * np.pi * f
    z_cpe = 1.0 / (Q2 * (1j * w) ** ph2)
    z_par = 1.0 / (1.0 / R2 + 1.0 / z_cpe)
    z_ws = aw_eq / np.sqrt(1j * w)
    d_abs = np.abs(z_par - z_ws)
    sig = float(np.hypot(med(sA, "s_re"), med(sA, "s_im")))
    print(f"\n  R2∥CPE2 与 Ws(A_W={aw_eq:.2f}) 在窗口内的差（σ_pair={sig:.2f} Ω）：")
    for fi in (0.1, 0.15, 0.2, 0.3, 0.5, 1.0):
        k = int(np.argmin(np.abs(f - fi)))
        print(f"    f={f[k]:8.3f} Hz  |Z_par|={abs(z_par[k]):6.2f}  "
              f"|Z_Ws|={abs(z_ws[k]):6.2f}  Δ={d_abs[k]:5.2f} Ω"
              f" ({d_abs[k] / sig:5.1f}×σ_pair)"
              f"  相位 {np.degrees(np.angle(z_par[k])):6.1f}° vs "
              f"{np.degrees(np.angle(z_ws[k])):6.1f}°")

    # ---------------- 4. 残差分段：谁在哪一段欠拟合 ----------------
    print("\n=== 4. 拟合优度与残差分段 ===")
    for tag, d in (("K=2   ", A_DIR), ("K=1+Ws", B_DIR), ("K=1+Wo", C_DIR)):
        js = json.loads((d / f"real_{STEM}.json").read_text(encoding="utf-8"))
        print(f"  {tag}: rms_re={js['rms_re']:.3f}  rms_im={js['rms_im']:.3f}"
              f"  acf={js['acf']:.2f}  trend={js['trend_re']:.2f}/{js['trend_im']:.2f}σ"
              f"  elpd={js['elpd_loo']:.1f}  n_pts={js['n']}")
        for row in js["resid_bins"]:
            print(f"      低→高 {row['f_lo']:.3g}–{row['f_hi']:.3g} Hz  n={row['n']:2d}"
                  f"  mean Re/Im = {row['mean_re']:+.3f}/{row['mean_im']:+.3f}"
                  f"  rms Re/Im = {row['rms_re']:.3f}/{row['rms_im']:.3f}")
        if js.get("diffusion_params"):
            print(f"      diffusion_params = {js['diffusion_params']}")


if __name__ == "__main__":
    main()
