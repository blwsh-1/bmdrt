"""要不要给「低频偏差」立判据？——先证明现有判据的门禁用错了统计量。

核心：判据必须是模型无关的数据量（只用 f, Z）。「低频偏差」= Zmod−Zobs 依赖模型，
同一谱用 K=1 / K=2 拟合会给出相反答案，不能当判据。

本脚本对比两种门禁在已知语料上的行为：
  A. 现用：r2 ≥ 0.90（要求「是单一幂律」）
  B. 建议：斜率 95%CI 是否排除 −0.5（要求「斜率定得够准」）
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import bm_real
import bm_ws

FILES = [
    ("examples/example_caseE.csv", "Ws 真值 A_W=2.03"),
    ("examples/example_caseE_warburg.csv", "Ws 真值（纯扩散）"),
    ("examples/example_caseF.csv", "无扩散（ZARC 系）"),
    ("examples/example_caseF_wo.csv", "Wo 真值"),
    ("examples/example_caseG_wo_only.csv", "纯 Wo"),
    ("archive/Blank_0.1KBr_pH10_EIS_PtCu_1.2V.txt", "实测 Blank（本次主谱）"),
]


def ols_slope(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = A @ b
    n = len(x)
    dof = n - 2
    s2 = ((y - pred) ** 2).sum() / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = float(np.sqrt(cov[0, 0]))
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(b[0]), se, float(r2)


def main():
    print(f"{'谱':34s} {'n':>3s} {'slope':>7s} {'±se':>6s} {'95%CI':>18s} "
          f"{'r2':>6s} {'A:r2':>6s} {'B:CI排除-0.5':>12s}")
    print("-" * 100)
    for path, note in FILES:
        if not os.path.exists(path):
            print(f"{note:34s}  文件缺失")
            continue
        f, Z = bm_real.load_spectrum(path)
        o = np.argsort(f)
        f, Z = f[o], Z[o]
        si = bm_ws.slope_im(f, Z)
        n = si["n"]
        x = np.log10(f[:n])
        y = np.log10(-Z.imag[:n])
        s, se, r2 = ols_slope(x, y)
        lo, hi = s - 1.96 * se, s + 1.96 * se
        a_ok = r2 >= 0.90
        # B: 95%CI 是否排除 −0.5（= 可明确否决「该段是半无限 Ws」）
        if not np.isfinite(se):
            b_txt = "n/a"
        else:
            b_txt = "yes" if not (lo <= -0.5 <= hi) else "no"
        tag = f"[{note}]"
        print(f"{tag:34s} {n:3d} {s:7.4f} {se:6.4f} [{lo:7.3f},{hi:7.3f}] "
              f"{r2:6.3f} {'pass' if a_ok else 'FAIL':>6s} {b_txt:>12s}")


if __name__ == "__main__":
    main()
