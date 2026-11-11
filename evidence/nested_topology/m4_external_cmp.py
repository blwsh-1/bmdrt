"""§9.4.29 外部工具同谱对照（exploratory，不改任何判定）。

用户提供的两张截图来自某 EIS 拟合界面，谱 = `AZ31-4Li_100`（50 点 0.01–1e5 Hz）。
电路与参数按截图抄录：

    Rs + (Rct1 || CPE1) + [CPEd || (Rct || [Rads + Lads])]
    Rs=1e-8 Ω, Rct1=58.44 Ω, Q1=1.638e-5 S·s^n, n1=0.4445
    Qdl=3.124e-5 S·s^n, ndl=0.7954, Rct=3645 Ω
    Rads=712.5 Ω, Lads=7191 H          ← L 与 Rads **串联**

本脚本做两件事：
  1. 按上式独立算出预测阻抗，复现截图报的四个指标（自检，防止参数抄错）；
  2. 与项目 `nested parallel/L=2`（逐点 2% σ、standard）用同一套指标对比，
     重点看两条**不同拓扑**的预测曲线差多少、以及 acf 是否过 §13 的 N-b。

用法：python evidence/nested_topology/m4_external_cmp.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import bm_experiments as E                     # noqa: E402
import bm_real as R                            # noqa: E402
import bm_top as T                             # noqa: E402

SPEC = os.path.join("out", "_m6_samples", "z_csv", "AZ31-4Li_100.csv")
CONV = os.path.join("out", "m4_conv_2000.json")

# 截图中外部界面报的数（用于自检，不参与任何计算）
EXT_REPORT = dict(dz_rms=206.17, dz_mean=118.04, dz_max=650.64,
                  rmse=145.79, rwp=9.3789)

# 截图中读到的元件参数（唯一来源；不得为了让结果好看而调整）
P = dict(Rs=1e-8, Rct1=58.44, Q1=1.638e-5, n1=0.4445,
         Qdl=3.124e-5, ndl=0.7954, Rct=3645.0, Rads=712.5, Lads=7191.0)


def z_external(f, p=P):
    """两个并联节点**串联**；第二节点含 (Rads + jωLads) 支路。"""
    om = 2 * np.pi * np.asarray(f, float)
    z_a = 1.0 / (1.0 / p["Rct1"] + p["Q1"] * (1j * om) ** p["n1"])
    z_b = 1.0 / (p["Qdl"] * (1j * om) ** p["ndl"] + 1.0 / p["Rct"]
                 + 1.0 / (p["Rads"] + 1j * om * p["Lads"]))
    return p["Rs"] + z_a + z_b


def metrics(f, Z, Zm):
    d = Zm - Z
    a = np.abs(d)
    w = np.abs(Z)
    return dict(
        dz_rms=float(np.sqrt(np.mean(a ** 2))),
        dz_mean=float(np.mean(a)),
        dz_max=float(np.max(a)),
        rmse=float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2))),
        rwp=float(np.sqrt(np.mean((a / w) ** 2)) * 100),
        rms_rel_proj=float(np.sqrt(np.mean(
            np.concatenate([(d.real / w) ** 2, (d.imag / w) ** 2]))) * 100),
        acf_proj=float(max(E.acf_max(d.real), E.acf_max(d.imag))),
        band_rms=[float(np.sqrt(np.mean(a[(f >= lo) & (f < hi)] ** 2)))
                  for lo, hi in ((0.01, 1), (1, 100), (100, 1e5))])


def main() -> int:
    f, Z = R.load_spectrum(SPEC)
    Ze = z_external(f)
    me = metrics(f, Z, Ze)

    print("A. 复现自检（按图参数独立算 vs 截图报数）")
    for k, tag in (("dz_rms", "|ΔZ|RMS Ω"), ("dz_mean", "平均 Ω"),
                   ("dz_max", "最大 Ω"), ("rmse", "RMSE Ω"), ("rwp", "Rwp %")):
        rep = EXT_REPORT["rwp" if k == "rwp" else k]
        print(f"   {tag:<14}{me[k]:>10.4f}   截图 {rep:>10.4f}   差 {me[k]-rep:+.4f}")
        assert abs(me[k] - rep) < 0.1, f"复现失败：{tag}"
    print("   → 参数与电路结构抄录正确；Rwp = sqrt(mean((|ΔZ|/|Z|)²))，未除 √2")

    d = json.load(open(CONV, encoding="utf-8"))
    r = d["rows"][0]
    Zo = T.z_nested(f, r["params"], branch=d["branch"], layers=int(d["layers"]))
    mo = metrics(f, Z, Zo)

    print(f"\nB. 同谱对照（§13 判据：N-a rms_rel≤10%，N-b acf≤0.50）")
    print(f"   {'':<22}{'外部电路':>12}{'我们 nested':>14}")
    for k, tag, fmt in (("dz_rms", "|ΔZ|RMS (Ω)", "{:.1f}"),
                        ("dz_mean", "平均 |ΔZ| (Ω)", "{:.1f}"),
                        ("dz_max", "最大 |ΔZ| (Ω)", "{:.1f}"),
                        ("rmse", "RMSE (Ω)", "{:.1f}"),
                        ("rwp", "Rwp / 相对RMS (%)", "{:.3f}"),
                        ("rms_rel_proj", "项目口径 rms_rel (%)", "{:.3f}"),
                        ("acf_proj", "acf（项目口径）", "{:.3f}")):
        print(f"   {tag:<22}{fmt.format(me[k]):>12}{fmt.format(mo[k]):>14}")
    print(f"   {'低/中/高频 |ΔZ|RMS':<22}"
          f"{'/'.join(f'{v:.0f}' for v in me['band_rms']):>12}"
          f"{'/'.join(f'{v:.0f}' for v in mo['band_rms']):>14}")

    diff = np.abs(Ze - Zo)
    print(f"\nC. 两条预测曲线之差：RMS {np.sqrt(np.mean(diff**2)):.1f} Ω、"
          f"最大 {diff.max():.1f} Ω、相对 |Z| RMS "
          f"{np.sqrt(np.mean((diff/np.abs(Z))**2))*100:.2f}%")
    print("   拓扑差异：外部 = 两并联节点串联 + (Rads+Lads) 串联支路；"
          "我们 = 嵌套 + (R_ad∥L_ad) 并联支路。")
    print("   → 拟合优度不能反推接法；两者 acf 都 >0.50，均未过 §13 N-b。")
    print("   边界：单谱、截图参数、外部工具无 R̂/区间/留出谱 → exploratory，不改判定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
