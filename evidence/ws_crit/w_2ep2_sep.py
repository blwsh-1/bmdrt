"""2EP-2_exp2_0500mA：加 W 之后，图上能不能看出差别？

复用 WP2 `viz_separability.py` 的做法：各臂后验中位数重建 Z(f)，算两两
差相对 σ 的大小、画低频放大叠图。**不是判据**，只是把"不可区分"画出来。

    python evidence/ws_crit/w_2ep2_sep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bm_core as B  # noqa: E402

STEM = "2EP-2_exp2_0500mA"
BASE = ROOT / "out" / "_w_2ep2"
ARMS = ["k1_none", "k1_Ws", "k1_Wo", "k2_none", "k2_Ws"]
COL = {"k1_none": "#555555", "k1_Ws": "#d62728", "k1_Wo": "#ff7f0e",
       "k2_none": "#1f77b4", "k2_Ws": "#2ca02c"}
LBL = {"k1_none": "K=1 none (baseline)", "k1_Ws": "K=1 + Ws",
       "k1_Wo": "K=1 + Wo", "k2_none": "K=2 none (grade 0)",
       "k2_Ws": "K=2 + Ws (grade 0)"}


def load(tag):
    b = BASE / tag / STEM / f"real_{STEM}"
    js = json.loads(Path(str(b) + ".json").read_text(encoding="utf-8"))
    z = np.load(str(b) + "_samples.npz", allow_pickle=True)
    p = dict(R_inf=float(np.median(z["R_inf"])),
             W=float(np.median(z["W"])),
             pi=np.median(z["pi"], axis=0),
             tau=np.median(z["tau"], axis=0),
             phi=np.median(z["phi"], axis=0),
             zeta=np.median(z["zeta"], axis=0))
    if js["include_L"]:
        p["L"] = float(np.median(z["L"]))
    dp = js.get("diffusion_params") or {}
    for k in ("Aw", "Rwo", "tauD"):
        if k in dp:
            p[k] = float(dp[k])
    f = z["f"]
    Zo = z["Zre"] + 1j * z["Zim"]
    Zm = B.model_z(f, p, include_L=bool(js["include_L"]))
    return dict(tag=tag, f=f, Zo=Zo, Zm=Zm, js=js,
                s_re=js["sigma_re"], s_im=js["sigma_im"])


def main() -> int:
    A = {t: load(t) for t in ARMS}
    f = A["k1_none"]["f"]
    Zo = A["k1_none"]["Zo"]
    s = np.hypot(A["k1_none"]["s_re"], A["k1_none"]["s_im"])
    lf = f <= 10.0

    print("=== ① 曲线两两最大差（以 σ 为单位） ===")
    print(f"{'A - B':<21} {'max|dZ|/σ 全窗':>16} {'低频 f<=10Hz':>15} "
          f"{'低频落在1σ内':>14}")
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1:]:
            dz = np.abs(A[a]["Zm"] - A[b]["Zm"])
            inband = float(np.mean(dz[lf] <= s))
            print(f"{a + ' - ' + b:<21} {np.max(dz) / s:16.3f} "
                  f"{np.max(dz[lf]) / s:15.3f} {inband:13.0%}")

    print("\n=== ② 低频实测 -Im 与三条预测（Ω） ===")
    print(f"{'f/Hz':>9} {'-Im obs':>10} {'none':>10} {'Ws':>10} {'Wo':>10} "
          f"{'σ_im':>9}")
    idx = np.where(lf)[0]
    for i in idx[np.argsort(f[idx])]:
        print(f"{f[i]:9.3f} {-Zo.imag[i]:10.5f} "
              f"{-A['k1_none']['Zm'].imag[i]:10.5f} "
              f"{-A['k1_Ws']['Zm'].imag[i]:10.5f} "
              f"{-A['k1_Wo']['Zm'].imag[i]:10.5f} "
              f"{A['k1_none']['s_im']:9.5f}")

    # ---------------- 图 ----------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    a0 = ax[0][0]
    a0.scatter(Zo.real, -Zo.imag, s=18, c="k", label="data", zorder=5)
    for t in ARMS:
        a0.plot(A[t]["Zm"].real, -A[t]["Zm"].imag, lw=1.4,
                color=COL[t], label=LBL[t], alpha=0.9)
    a0.set_xlabel("Re Z (ohm)"); a0.set_ylabel("-Im Z (ohm)")
    a0.set_title("Nyquist (full window)"); a0.legend(fontsize=7)
    a0.grid(alpha=0.3)

    a1 = ax[0][1]
    a1.errorbar(Zo.real[lf], -Zo.imag[lf], yerr=A["k1_none"]["s_im"],
                fmt="o", ms=4, color="k", capsize=2, label="data +/- 1 sigma")
    for t in ("k1_none", "k1_Ws", "k1_Wo"):
        a1.plot(A[t]["Zm"].real[lf], -A[t]["Zm"].imag[lf], "o-",
                ms=3, lw=1.2, color=COL[t], label=LBL[t])
    a1.set_xlabel("Re Z (ohm)"); a1.set_ylabel("-Im Z (ohm)")
    a1.set_title("Low-f zoom (f <= 10 Hz): Ws/Wo vs noise band")
    a1.legend(fontsize=7); a1.grid(alpha=0.3)

    a2 = ax[1][0]
    a2.errorbar(f[lf], -Zo.imag[lf], yerr=A["k1_none"]["s_im"],
                fmt="o", ms=4, color="k", capsize=2, label="data")
    for t in ("k1_none", "k1_Ws", "k1_Wo"):
        a2.plot(f[lf], -A[t]["Zm"].imag[lf], "s-", ms=3, lw=1.2,
                color=COL[t], label=LBL[t])
    a2.axhline(0, color="gray", lw=0.8, ls=":")
    a2.set_xscale("log"); a2.set_xlabel("f (Hz)"); a2.set_ylabel("-Im Z (ohm)")
    a2.set_title("Low-f imaginary part (45-deg tail would rise as f^-1/2)")
    a2.legend(fontsize=7); a2.grid(alpha=0.3, which="both")

    a3 = ax[1][1]
    for t in ("k1_none", "k1_Ws", "k1_Wo"):
        a3.plot(f[lf], (Zo.imag[lf] - A[t]["Zm"].imag[lf]) /
                A["k1_none"]["s_im"], "o-", ms=3, lw=1.2,
                color=COL[t], label=LBL[t])
    a3.axhline(0, color="gray", lw=0.8)
    a3.axhspan(-1, 1, color="gray", alpha=0.18, label="+/- 1 sigma")
    a3.set_xscale("log"); a3.set_xlabel("f (Hz)")
    a3.set_ylabel("imag residual / sigma")
    a3.set_title("Low-f residuals: adding W does not remove structure")
    a3.legend(fontsize=7); a3.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = BASE / "separability.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
