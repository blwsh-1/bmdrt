"""要不要给「低频偏差」立判据？——先证明现有判据的门禁用错了统计量。

核心：判据必须是模型无关的数据量（只用 f, Z）。「低频偏差」= Zmod−Zobs 依赖模型，
同一谱用 K=1 / K=2 拟合会给出相反答案，不能当判据。

本脚本对比两种门禁在已知语料上的行为：
  A. 现用：r2 ≥ 0.90（要求「是单一幂律」）
  B. 建议：斜率 95%CI 是否排除 −0.5（要求「斜率定得够准」）

**B 目前只是候选方案，未验证**（WP2 审查 2026-11-09 第 7 条）。本脚本补齐其可复现性：
  · OLS 标准误（假定残差同方差独立）
  · iid bootstrap（4000 次）
  · moving-block bootstrap（块长 3，处理残差自相关）
三者并列输出；若块 bootstrap 区间明显更宽，说明 OLS 的 se 低估。
样本仅 6 条（5 条可算），**不足以证明新门禁的泛化性能**。
"""
import contextlib
import io
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

B_BOOT = 4000
BLOCK = 3


def ols(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = A @ b
    n = len(x)
    dof = n - 2
    s2 = ((y - pred) ** 2).sum() / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = float(np.sqrt(cov[0, 0]))
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    resid = y - pred
    ac1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if n > 3 else np.nan
    return float(b[0]), se, float(r2), ac1


def _block_idx(n, L, rng):
    """循环移动块 bootstrap 的下标。"""
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, nb)
    idx = np.concatenate([(s + np.arange(L)) % n for s in starts])[:n]
    return idx


def boot(x, y, B=B_BOOT, L=None, seed=0):
    n = len(x)
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n) if L is None else _block_idx(n, L, rng)
        A = np.vstack([x[idx], np.ones(n)]).T
        out[b] = np.linalg.lstsq(A, y[idx], rcond=None)[0][0]
    return out


def main():
    print(f"bootstrap B={B_BOOT}，block L={BLOCK}（循环移动块，处理残差自相关）")
    print(f"{'谱':30s} {'n':>3s} {'slope':>7s} {'se':>6s} {'ac1':>6s} "
          f"{'iid 95%CI':>17s} {'block 95%CI':>17s} {'A:r2':>6s} {'B:CI':>7s} {'B(blk)':>7s}")
    print("-" * 122)
    n_ok = 0
    n_b = 0
    n_b_fail = 0
    for path, note in FILES:
        if not os.path.exists(path):
            print(f"{note:30s}  文件缺失")
            continue
        with contextlib.redirect_stdout(io.StringIO()):   # 压掉 load_spectrum 的「读入…」
            f, Z = bm_real.load_spectrum(path)
        o = np.argsort(f)
        f, Z = f[o], Z[o]
        si = bm_ws.slope_im(f, Z)
        n = si["n"]
        if n < 4 or not np.all(np.isfinite(np.log10(-Z.imag[:n]))):
            print(f"{note:30s} {n:3d}  —— 无法计算（段内 −Im ≤ 0 或点数不足）")
            continue
        x = np.log10(f[:n])
        y = np.log10(-Z.imag[:n])
        s, se, r2, ac1 = ols(x, y)
        lo, hi = s - 1.96 * se, s + 1.96 * se
        bi = boot(x, y)
        bb = boot(x, y, L=BLOCK)
        li, hi_i = np.percentile(bi, [2.5, 97.5])
        lb, hb = np.percentile(bb, [2.5, 97.5])

        a_ok = r2 >= 0.90
        b_txt = "yes" if not (lo <= -0.5 <= hi) else "no"
        bb_txt = "yes" if not (lb <= -0.5 <= hb) else "no"
        n_ok += 1
        n_b += b_txt == "yes"
        n_b_fail += bb_txt != b_txt
        print(f"{note:30s} {n:3d} {s:7.4f} {se:6.4f} {ac1:6.2f} "
              f"[{li:7.3f},{hi_i:7.3f}] [{lb:7.3f},{hb:7.3f}] "
              f"{'pass' if a_ok else 'FAIL':>6s} {b_txt:>7s} {bb_txt:>7s}")
    print()
    print(f"可计算 {n_ok} 条；B 判 yes {n_b}/{n_ok}；块 bootstrap 改变结论 {n_b_fail} 条")
    print("注意：这是 6 条语料上的**工程观察**，不构成新门禁的统计验证；"
          "泛化性能需要更多合成谱（不同 φ、窗宽、噪声）。")


if __name__ == "__main__":
    main()
