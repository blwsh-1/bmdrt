"""k1_Ws 的低频失败机理：残差符号、外推行为。

问：k1+Ws 在低频整段与数据分离，是参数没调好，还是结构错了？
做法：① 最低 6 点逐点 Im 残差（看是否单调恶化）
      ② 各臂外推到窗下 1 decade，看 −Im 是发散还是趋向平台
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import bm_core as B

ROOT, STEM = "out/_wp2_blank", "Blank_0.1KBr_pH10_EIS_PtCu_1.2V"
ARMS = ["k1_none", "k1_Ws", "k1_Wo", "k2_none", "k2_Ws", "k2_Wo"]
LBL = {"k1_none": "K=1 none", "k1_Ws": "K=1 +Ws", "k1_Wo": "K=1 +Wo",
       "k2_none": "K=2 none", "k2_Ws": "K=2 +Ws", "k2_Wo": "K=2 +Wo"}


def load(tag):
    b = f"{ROOT}/{tag}/{STEM}/real_{STEM}"
    js = json.load(open(b + ".json", encoding="utf-8"))
    z = np.load(b + "_samples.npz", allow_pickle=True)
    p = {"R_inf": float(np.median(z["R_inf"])), "W": float(np.median(z["W"])),
         "pi": np.median(z["pi"], 0), "tau": np.median(z["tau"], 0),
         "phi": np.median(z["phi"], 0), "zeta": np.median(z["zeta"], 0)}
    if js["include_L"]:
        p["L"] = float(np.median(z["L"]))
    dp = js["diffusion_params"] or {}
    if "Aw" in dp:
        p["Aw"] = dp["Aw"]
    if "Rwo" in dp:
        p["Rwo"], p["tauD"] = dp["Rwo"], dp["tauD"]
    f = z["f"]
    Zo = z["Zre"] + 1j * z["Zim"]
    return dict(f=f, Zo=Zo, Zm=B.model_z(f, p, include_L=js["include_L"]),
                p=p, L=js["include_L"], sig=js["sigma_im"], js=js)


def main():
    A = {t: load(t) for t in ARMS}
    f = A["k2_none"]["f"]
    Zo = A["k2_none"]["Zo"]
    o = np.argsort(f)

    print("实测最低 8 点（真实数据本身）：")
    for i in o[:8]:
        print(f"  f={f[i]:8.4f} Hz  Re={Zo[i].real:7.3f}  Im={Zo[i].imag:7.3f}"
              f"  ph={np.degrees(np.angle(Zo[i])):6.2f}°  |Z|={abs(Zo[i]):6.2f}")

    print("\n" + "=" * 76)
    print("① 最低 6 点的 Im 残差（Zmod−Zobs；负 = 模型高估容性，即尾巴太长）")
    hdr = "".join(f"{f[i]:>9.4f}" for i in o[:6])
    print(f"{'arm':11s}{hdr}   单调?")
    for t in ARMS:
        r = (A[t]["Zm"] - A[t]["Zo"]).imag
        vals = [r[i] for i in o[:6]]
        mono = all(vals[k] < vals[k + 1] for k in range(5))
        mono_dn = all(vals[k] > vals[k + 1] for k in range(5))
        tagm = "递减(越到低频越负)" if mono else ("递增" if mono_dn else "非单调")
        print(f"{LBL[t]:11s}" + "".join(f"{v:>9.3f}" for v in vals) + f"   {tagm}")

    print("\n" + "=" * 76)
    print("② 外推到窗下 1 decade：−Im（Ω）在 0.01/0.03/0.1/0.3/1 Hz")
    fg = np.array([0.01, 0.03, 0.1, 0.3, 1.0])
    print(f"{'arm':11s}" + "".join(f"{x:>9.2f}" for x in fg) + "   相位@0.01Hz")
    for t in ARMS:
        Zg = B.model_z(fg, A[t]["p"], include_L=A[t]["L"])
        print(f"{LBL[t]:11s}" + "".join(f"{-v:>9.2f}" for v in Zg.imag)
              + f"   {np.degrees(np.angle(Zg[0])):7.2f}°")

    print("\n" + "=" * 76)
    print("③ 低频段（≤2 Hz）残差统计：均值 / rms / 与 σ 的倍数")
    lo = f <= 2.0
    for t in ARMS:
        r = (A[t]["Zm"] - A[t]["Zo"]).imag[lo]
        print(f"  {LBL[t]:11s} mean={r.mean():+7.3f} Ω  rms={np.sqrt((r**2).mean()):6.3f} Ω"
              f"  σ={A[t]['sig']:.3f}  mean/σ={r.mean()/A[t]['sig']:+6.1f}"
              f"  rms/σ={np.sqrt((r**2).mean())/A[t]['sig']:5.1f}")

    print("\n" + "=" * 76)
    print("④ A_W 扫描：固定 K=1 骨架最优参数（k1_none），只加不同幅值的 Ws")
    print("   看低频偏差能否靠「调幅值」消掉，以及代价落在哪个频段")
    p0 = dict(A["k1_none"]["p"])
    band_lo = f <= 1.0
    band_mid = (f > 2.0) & (f <= 500.0)
    print(f"{'A_W':>7s} {'低频mean':>10s} {'低频rms':>9s} {'中频rms':>9s} {'全段rms':>9s}")
    for aw in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.18):
        pp = dict(p0)
        if aw > 0:
            pp["Aw"] = aw
        r = (B.model_z(f, pp, include_L=True) - Zo).imag
        print(f"{aw:7.2f} {r[band_lo].mean():+10.3f} "
              f"{np.sqrt((r[band_lo]**2).mean()):9.3f} "
              f"{np.sqrt((r[band_mid]**2).mean()):9.3f} "
              f"{np.sqrt((r**2).mean()):9.3f}")
    print("   （k1_none 自报 σ_im=%.3f；K=2 全段 rms=%.3f）"
          % (A["k1_none"]["sig"],
             np.sqrt((((A["k2_none"]["Zm"] - A["k2_none"]["Zo"]).imag) ** 2).mean())))


if __name__ == "__main__":
    main()
