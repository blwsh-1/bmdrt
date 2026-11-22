"""只读盘点：csv_3e 里有哪些 (电池, 类型, 电流) 组有 >=3 次跨实验重复。

用于 noise_scale_probe.py 的样本选择；不产出结论。
"""
from __future__ import annotations

import collections
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import bm_real as R  # noqa: E402

PAT = re.compile(r"(3E-\d)_(HER|OER|full_electrolysis_cell)_exp(\d)_(\d+)mA\.csv$")


def main() -> int:
    g = collections.defaultdict(list)
    for p in sorted(glob.glob("EISmeasurements/csv_3e/*.csv")):
        m = PAT.search(os.path.basename(p))
        if m:
            g[(m.group(1), m.group(2), m.group(4))].append((int(m.group(3)), p))
    rows = []
    for k, v in sorted(g.items()):
        if len(v) < 3:
            continue
        Zs, fs = [], []
        for _, p in v:
            f, Z = R.load_spectrum(p)
            fs.append(np.asarray(f, float))
            Zs.append(np.asarray(Z, complex))
        a = np.abs(np.stack([Z[:min(map(len, Zs))] for Z in Zs]))
        rs = a[:, -5:].real.mean(axis=1)          # 高频端 Re ≈ Rs
        lo = a[:, 0].mean()
        rows.append((k, len(v), lo, a.max(), float(np.std(rs, ddof=1) / np.mean(rs))))
    hdr = "%-40s %2s %9s %9s %8s" % ("group", "n", "|Z|lo", "|Z|hi", "Rs_CV")
    print(hdr)
    print("-" * len(hdr))
    for k, n, lo, hi, cv in rows:
        print("%-40s %2d %9.3g %9.3g %7.1f%%" % (str(k), n, lo, hi, 100 * cv))
    print("\ngroups n>=3: %d   n==4: %d" % (len(rows), sum(1 for r in rows if r[1] == 4)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
