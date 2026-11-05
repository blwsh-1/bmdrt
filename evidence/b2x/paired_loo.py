"""配对 LOO 差：用各组的 samples.npz 重算逐点 elpd，再做配对相减。

为什么要单独一个脚本
--------------------
`bmdrt.py auto --krange 1,1` 只跑一个模型，`select_k()` 里没有第二个模型可比，
所以 `elpd_diff` / `dse` 字段是占位（0 / nan）。而 §3.4 要的正是
**跨配置**的配对差，只能事后从后验样本重算：

    ll[sample, f]  ← loglik_matrix(s, f, Z)     与 bm_real 内部同一条路径
    elpd_pt[f]     ← PSIS-LOO 的逐点 elpd
    diff           = Σ (elpd_pt_A − elpd_pt_B)   同一批频率点相减，方差才算得对
    dse            = √n · std(elpd_pt_A − elpd_pt_B)

用法：
    python evidence/b2x/paired_loo.py k1 k3 k1ws k1wo
    （参数是 evidence/b2x/ 下的子目录名，顺序即表格顺序）
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bm_real as R          # noqa: E402


def load(tag: str):
    p = glob.glob(str(ROOT / "evidence" / "b2x" / tag / "*" / "*_samples.npz"))
    if not p:
        raise SystemExit(f"{tag}: 找不到 samples.npz")
    z = np.load(p[0], allow_pickle=True)
    s = {k: z[k] for k in z.keys()}
    f, Z = z["f"], z["Zre"] + 1j * z["Zim"]
    ll = R.loglik_matrix(s, f, Z, False)
    _, elpd, se, bad_k, pt = R.loo_pointwise(ll)
    return dict(tag=tag, elpd=elpd, se=se, bad_k=bad_k, pt=pt, n=int(f.size))


def main(tags):
    d = {t: load(t) for t in tags}
    print("=== 单模型 elpd_loo ===")
    for t in tags:
        x = d[t]
        print(f"  {t:6s} elpd={x['elpd']:9.2f} ± {x['se']:5.2f}"
              f"   pareto-k>0.7: {x['bad_k'] * 100:.1f}%")

    print("\n=== 配对差（同一批频率点；正值 = 前者更好）===")
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            dd = d[a]["pt"] - d[b]["pt"]
            diff = float(np.sum(dd))
            # ddof=1：样本标准差。与 arviz.compare() 的标准误约定一致；
            # n=31 时与 ddof=0 差约 1.6%，不改定性结论但该对齐。
            dse = float(np.sqrt(dd.size) * np.std(dd, ddof=1))
            z = abs(diff) / dse if dse > 0 else float("inf")
            print(f"  {a:6s} - {b:6s}: {diff:+8.2f} ± {dse:6.2f}   ({z:.1f}σ)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["k1", "k3", "k1ws", "k1wo"])
