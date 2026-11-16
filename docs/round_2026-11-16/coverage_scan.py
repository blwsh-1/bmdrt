#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三拓扑覆盖性审查（只读）：汇总 §14 逐谱结果与感抗频段分布。

不改生产代码、不跑拟合。只读 out/m4_holdout2.json 与 archive/ 的原始谱，
输出每条谱的 rms/acf/Rhat/div/分频段残差 + 低频/高频感抗点数，供归类。
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import bm_real as R

HOLDOUT = _ROOT / "out" / "m4_holdout2.json"
ARCHIVE = _ROOT / "archive"
ZCSV = _ROOT / "out" / "_m6_samples" / "z_csv"   # §14 的原始谱目录


def band_ind(f, Z):
    """低频/高频感抗点数（split=几何中频，与 voigt_can_represent 同口径）。"""
    split = float(np.sqrt(f.min() * f.max()))
    pos = Z.imag > 0
    lo = int(np.sum(pos & (f <= split)))
    hi = int(np.sum(pos & (f > split)))
    return lo, hi, round(split, 2)


def main():
    import json
    d = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    print(f"{'谱':<18} {'通过':<5} {'rms%':>6} {'acf':>5} {'Rhat':>6} {'div':>3} "
          f"{'低/中/高残差%':<20} {'感抗 lo/hi':<10} {'detect_L'}")
    print("-" * 96)
    for r in d["rows"]:
        spec = r["spec"]
        sub = r["rows"][0] if r["rows"] else {}
        bands = sub.get("bands") or []
        bs = "/".join(f"{float(x):.1f}" for x in bands) if bands else "-"
        # 找原始谱
        p = ZCSV / f"{spec}.csv"
        if not p.exists():
            p = ARCHIVE / f"{spec}.csv"
        lo = hi = "-"
        det = "-"
        if p.exists():
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                f, Z = R.load_spectrum(str(p))
            o = np.argsort(f)
            f, Z = f[o], Z[o]
            l, h, _ = band_ind(f, Z)
            lo, hi = l, h
            det = "Y" if R.detect_L(f, Z) else "n"
        print(f"{spec:<18} {'Y' if r['passed_all'] else 'N':<5} "
              f"{sub.get('rms_rel', 0):>6.2f} {sub.get('acf', 0):>5.3f} "
              f"{sub.get('rhat_max', 0):>6.3f} {sub.get('div', 0):>3} "
              f"{bs:<20} {f'{lo}/{hi}':<10} {det}")


if __name__ == "__main__":
    main()
