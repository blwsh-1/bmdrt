"""参考频率阻抗贡献复核：B2 单位依赖、A* 渐近近似适用性、精确 HN 幅值。

审查意见（第 2 轮）：
- B2 = w·tau^(-phi) 单位 Ω·s^(-phi)，换时间单位会非统一缩放 -> 不是单位不变判据。
- A* = w·(omega*tau)^(-phi) 单位 Ω，但低频端 omega*tau 未必 >> 1，渐近近似不成立。
- 因此增加精确 HN 幅值
      A_exact(omega*) = | w / [1 + (j omega* tau)^phi]^zeta |
  并把指标改名为 reference_frequency_contribution。

只读 NPZ，不改算法代码。运行：
  D:/autoeis_env/Scripts/python.exe asymptotic_recheck.py
输出：reference_frequency_contribution.csv（第四层状态不含 asymptotic_only）
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = next((q for q in [ROOT, *ROOT.parents] if (q / "bm_core.py").exists()),
            Path(r"D:/eis训练数据/bmdrt"))
sys.path.insert(0, str(REPO))
import bm_core as B  # noqa: E402

FMIN, FMAX = 0.1, 1e4
W1 = 2 * np.pi * FMIN
W2 = 2 * np.pi * np.sqrt(FMIN * FMAX)
RHO_MIN = 0.90
FAMILIES = [("none", "K2_none"), ("Ws", "K2_Ws"), ("Wo", "G_K2_Wo")]


def cv(a):
    a = np.asarray(a, float)
    return a.std(ddof=1) / a.mean()


def rho(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def main():
    rows = []
    for fam, key in FAMILIES:
        p = glob.glob(str(ROOT / "results" / key / "*_samples.npz"))[0]
        z = np.load(p, allow_pickle=True)
        s = {k: np.asarray(z[k], float) for k in z.files
             if k not in ("f", "Zre", "Zim")}
        mm = B.match_components(s)
        o = np.argsort(-np.median(mm["f0"], axis=0))
        j = o[-1]
        w, tau, phi, zeta = (mm["w"][:, j], mm["tau"][:, j],
                             mm["phi"][:, j], mm["zeta"][:, j])

        B_s = w * tau ** (-phi)
        B_ms = w * (tau * 1e3) ** (-phi)
        r = rho(np.log(w), np.log(tau))
        c_w, c_t = cv(w), cv(tau)
        rec = dict(family=fam, run_id=key, omega_fmin=W1, omega_geom=W2,
                   CV_B2_seconds=cv(B_s), CV_B2_milliseconds=cv(B_ms),
                   CV_w=c_w, CV_tau=c_t, rho_lnw_lntau=r)
        for tag, om in [("fmin", W1), ("geom", W2)]:
            x = om * tau
            mag = x ** phi
            ang = phi * np.pi / 2
            den = 1 + mag * (np.cos(ang) + 1j * np.sin(ang))
            A_exact = w / np.abs(den) ** zeta
            A_asym = w * x ** (-phi)
            rec[f"omega_tau_{tag}"] = float(np.median(x))
            rec[f"CV_Aexact_{tag}"] = cv(A_exact)
            rec[f"CV_Aasym_{tag}"] = cv(A_asym)
            rec[f"A_exact_q50_{tag}"] = float(np.median(A_exact))
            rec[f"three_exact_{tag}"] = bool(
                cv(A_exact) < c_w and cv(A_exact) < c_t and abs(r) > RHO_MIN)
            rec[f"asymptotic_valid_{tag}"] = bool(np.median(x) > 10)
            rec[f"P_omega_tau_gt10_{tag}"] = float(np.mean(x > 10))
        # 第四层状态只由精确 HN 幅值的稳定性决定；渐近有效性另列记录，不混用。
        ok1, ok2 = rec["three_exact_fmin"], rec["three_exact_geom"]
        rec["verdict"] = ("reference_frequency_stable" if ok1 and ok2
                          else "reference_frequency_dependent" if (ok1 or ok2)
                          else "unidentifiable")
        rows.append(rec)
    d = pd.DataFrame(rows)
    out = ROOT / "reference_frequency_contribution.csv"
    d.to_csv(out, index=False, float_format="%.6g")
    print(f"omega_fmin={W1:.4f}, omega_geom={W2:.4f} rad/s")
    print(d[["family", "omega_tau_fmin", "omega_tau_geom",
             "asymptotic_valid_fmin", "CV_B2_seconds", "CV_B2_milliseconds",
             "CV_Aasym_fmin", "CV_Aexact_fmin", "CV_Aexact_geom",
             "three_exact_fmin", "three_exact_geom", "verdict"]]
          .to_string(index=False))
    print("\n整体结论:", "reference_frequency_stable"
          if (d.verdict == "reference_frequency_stable").all()
          else "reference_frequency_dependent / 混合（依赖频率尺度）")
    print("DONE ->", out)


if __name__ == "__main__":
    main()
