"""门禁泛化测试：在**已知真值**（真含半无限 Ws）的合成谱上，量两条门禁的错判率。

背景（WP2_REPORT.md §3.5）：现用门禁 `r² ≥ 0.90` 在 6 条真实语料里误拒了一条真 Ws
（caseE_warburg 0.870）与 Blank（0.888）；候选门禁「95%CI 排除 −0.5」在同样 6 条上语义正确。
6 条不足以证明泛化，本脚本补上合成扫描。

判据语义（只作否决）：
  · A（现用）：`usable = r² ≥ 0.90`。真 Ws 上 `usable=False` 即**误拒**。
  · B（候选）：95%CI 是否排除 −0.5。真 Ws 上「排除」即**误否**（把真扩散判成非 Ws）。
理想：真 Ws 上 A 应 usable、B 应不排除。两者的错误率都随 SNR 变化。

合成谱（解析，无拟合）：Z = R∞ + R/(1+(jωτ)^φ) + A_W (jω)^{-1/2} + 噪声
  · 噪声按 `σ_rel × |Z|` 加在实部与虚部（与生产 `sigma_re/sigma_im` 口径不同，见文末注）
  · 频率窗与点数复制两条实测谱：Blank 0.1–1e4/50 点、caseE 0.01–1e5/31 点
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import bm_ws

WINDOWS = {
    "blank(0.1–1e4, 50pt)": (0.1, 1e4, 50),
    "caseE(0.01–1e5, 31pt)": (0.01, 1e5, 31),
}
R_INF, R_ARC, F_ARC, PHI = 10.0, 30.0, 300.0, 0.85
A_WS = (0.3, 0.6, 1.0, 2.0, 4.0)
SIG = (0.002, 0.01)          # 相对噪声（|Z| 的比例）
REPS = 200


def synth(f, a_w, sig, rng):
    w = 2 * np.pi * f
    tau = 1.0 / (2 * np.pi * F_ARC)
    Z = R_INF + R_ARC / (1 + (1j * w * tau) ** PHI)
    if a_w > 0:
        Z = Z + a_w * (1j * w) ** -0.5
    s = sig * np.abs(Z)
    return Z + s * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))


def gate_b(f, Z):
    """返回 (usable_A, r2, slope, ci_lo, ci_hi, why)。"""
    si = bm_ws.slope_im(f, Z)
    r2 = si["r2"]
    a = bool(si["usable"])
    why = si.get("why", "")
    # slope_im 内部已做 r² 门槛；为独立评估 B，这里重算不加门槛的 CI
    n = max(4, int(round(0.25 * f.size)))
    x = np.log10(f[:n])
    y = np.log10(-Z.imag[:n])
    A = np.vstack([x, np.ones_like(x)]).T
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ b
    dof = max(n - 2, 1)
    s2 = (resid ** 2).sum() / dof
    se = float(np.sqrt(s2 * np.linalg.inv(A.T @ A)[0, 0]))
    return a, float(r2), float(b[0]), float(b[0] - 1.96 * se), float(b[0] + 1.96 * se), why


def main():
    rng = np.random.default_rng(0)
    print("真含 Ws 的合成谱：A=现用(r²≥0.9)，B=候选(95%CI 排除 −0.5，只作否决)")
    print(f"R∞={R_INF}, R∥CPE={R_ARC}@{F_ARC}Hz φ={PHI}；噪声=|Z| 的 0.2% / 1.0%；每格 {REPS} 次\n")
    for tag, (f0, f1, npt) in WINDOWS.items():
        f = np.logspace(np.log10(f0), np.log10(f1), npt)
        print(f"### {tag}")
        print(f"{'A_W':>5s} {'σ_rel':>6s} {'r²中位':>7s} "
              f"{'A误拒':>6s} {'·早退':>6s} {'·r²':>5s} "
              f"{'B误否':>6s} {'slope中位':>9s} {'CI宽':>6s}")
        for a_w in A_WS:
            for sg in SIG:
                rej_a = early = rej_b = 0
                r2s, sls, ws = [], [], []
                for _ in range(REPS):
                    Z = synth(f, a_w, sg, rng)
                    a, r2, s, lo, hi, why = gate_b(f, Z)
                    r2s.append(r2)
                    sls.append(s)
                    ws.append(hi - lo)
                    if not a:
                        rej_a += 1
                        if "r2" in why or "r²" in why:
                            pass
                        else:
                            early += 1
                    veto = not (lo <= -0.5 <= hi)     # B 声称能排除 −0.5
                    rej_b += veto                      # 对真 Ws 就是误否
                print(f"{a_w:5.2f} {sg*100:5.1f}% {np.median(r2s):7.3f} "
                      f"{rej_a/REPS*100:5.0f}% {early/REPS*100:5.0f}% "
                      f"{(rej_a-early)/REPS*100:4.0f}% "
                      f"{rej_b/REPS*100:5.0f}% {np.median(sls):9.3f} {np.median(ws):6.3f}")
        print()

    print("读法：")
    print("  · A误拒 = 真 Ws 被判 usable=False；其中「早退」= 几何/单调性检查挡掉，"
          "「r²」= 线性度不足")
    print("  · B误否 = 真 Ws 的 95%CI 排除 −0.5（候选门禁把真扩散判成非 Ws）")
    print("  · 两者都应随 A_W↑、σ↓ 下降。**若 B 在强幅值处仍高**，说明偏差（残差弧尾）"
          "被精度放大成「显著偏离 −0.5」，CI 门禁不可用。")
    print()
    print("注：噪声按 |Z| 比例加，是简化；生产用同方差 s_re/s_im 由 MCMC 估计。")
    print("    因此本表给的是**趋势**，不能直接换算成真实谱的绝对误判率。")


if __name__ == "__main__":
    main()
