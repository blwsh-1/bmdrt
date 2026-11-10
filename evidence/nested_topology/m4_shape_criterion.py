"""结构判据：voigt 族的 `Re Z` 单调性可否定真实谱。

依据 `NESTED_TOPOLOGY_PROJECT.md` §2.3 的两族渐近对比：
    · A（voigt）：`ω→0` 时 `Re Z = R_inf + Σw_k` 为**最大值**，窗口内
      `Re` 随频率降低**单调不减**（每个 HN 实部从 0 单调升到 `w_k`）。
    · B（nested）：`ω→0` 时 `Re = Rs+R1+R2`，窗口内**可非单调**（中间频段
      出现峰与回落），因为 RL 支路在低频把实部拉回 0。

所以按频率**升序**排：
    · voigt 的 `Re` 应**单调不增**；任何"随频率升高而上升"的段都是反例。
    · 这不是 elpd 比较，是零拟合成本的形态检查 —— 它只能**排除**
      voigt，不能证明 nested（`Re` 单调时两族都可能）。

输出：每条的上升段数、最大上升幅度（占 max Re）、低频虚部符号、峰位。
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bm_real as R          # noqa: E402


#: 显著上升的下限。标定依据（`m4_Lprobe` / 四例反向对照）：
#: `voigt_true`（真值无 RL）最大上升 2.2%，来自 1% 加噪的两点差分；
#: `nested_true` 19.6%、`arcs_merged` 21.5%、真实 AZ31 32–93%。
#: 取 8% 让 voigt 真值留 3.6× 余量，同时保住两个强 nested 例。
UP_MIN_REL = 0.08
#: 低频虚部为感抗的下限（占 |Z|）。voigt 族（HN / Ws / Wo / ad）在 ω→0 时
#: Im ≤ 0：HN 与 Ws 恒负，Wo→实数，ad→0，jωL→0。所以显著正 Im 是排除性证据。
IM_LOW_MIN = 0.05


def analyze(f, Z, up_min_rel: float = UP_MIN_REL):
    o = np.argsort(f)
    f, Z = f[o], Z[o]
    re, im = Z.real, Z.imag
    scale = float(np.max(np.abs(Z)))
    dre = np.diff(re)                       # f 升序：正 = 上升 = 违反 voigt
    ups = dre[dre > 0]
    thresh = up_min_rel * scale
    sig = ups[ups > thresh]
    low = im[0] / scale
    return dict(
        n=len(f), scale=scale,
        n_up=int(ups.size), n_up_sig=int(sig.size),
        max_up=float(ups.max()) if ups.size else 0.0,
        max_up_rel=float(ups.max() / scale) if ups.size else 0.0,
        # 低频端是否比中低频低（nested 的 RL 指纹）
        low_dip=float((re[len(re) // 2] - re[0]) / scale),
        im_low=float(im[0]), im_low_rel=low,
        # 两条排除性判据
        excl_mono=bool(sig.size > 0),
        excl_im=bool(low > IM_LOW_MIN),
        f_at_max=float(f[int(np.argmax(re))]),
        f_min=float(f[0]),
    )


def main(argv):
    pat = argv[1] if len(argv) > 1 else "out/_m6_samples/z_csv/*.csv"
    files = sorted(glob.glob(pat))
    if not files:
        print(f"没有匹配文件：{pat}")
        return 2
    rows = []
    print(f"{'sample':<20}{'显著上升':>9}{'最大上升':>10}{'峰位/Hz':>10}"
          f"{'低频Im/|Z|':>11}  排除 voigt?")
    for p in files:
        f, Z = R.load_spectrum(p)
        a = analyze(f, Z)
        rows.append((Path(p).stem, a))
        why = "+".join(x for x, on in (("Re非单调", a["excl_mono"]),
                                      ("低频感抗", a["excl_im"])) if on) or "否"
        print(f"{Path(p).stem:<20}{a['n_up_sig']:>7}/{a['n'] - 1:<2}"
              f"{a['max_up_rel']:>9.1%}{a['f_at_max']:>10.3g}"
              f"{a['im_low_rel']:>+11.3f}  {why}")

    bad = [s for s, a in rows if a["excl_mono"] or a["excl_im"]]
    mono = sum(1 for _, a in rows if a["excl_mono"])
    ims = sum(1 for _, a in rows if a["excl_im"])
    print(f"\n合计 {len(rows)} 条：可排除 voigt {len(bad)} 条 "
          f"({len(bad) / len(rows):.0%})；其中 Re 非单调 {mono} 条、"
          f"低频感抗 {ims} 条")
    print(f"阈值：上升 > {UP_MIN_REL:.0%}·max|Z| 记非单调；低频 Im > "
          f"{IM_LOW_MIN:.0%}·|Z| 记感抗（两者任一成立即排除 voigt）。")
    print("读法：这是**排除**性判据 —— 成立则 voigt 族在数学上做不出该形状；")
    print("      不成立**不能**反推 voigt 成立，也不替代 elpd 比较。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
