"""只读探针：3E 高频感抗谱，voigt+include_L 能不能修好（不落产物）。

用户指出 `out/3E-3_OER_exp3_0100mA` 高频电感没加、拟合明显错。
本脚本只做验证，不改任何生产代码、不写 out/。
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]          # bmdrt 仓库根
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import bm_real as R
import bm_select as SEL
import bm_core as B
import spectrum_metrics as SM

SRC = "archive/3E-3_OER_exp3_0100mA.csv"
KW = dict(n_warmup=300, n_samples=300, num_chains=2,
          target_accept=0.9, max_tree_depth=8, extra_fields=("num_steps",))


def main():
    f, Z = R.load_spectrum(SRC)
    o = np.argsort(f)
    f, Z = f[o], Z[o]
    sig = np.stack([0.02 * np.abs(Z), 0.02 * np.abs(Z)])
    hi = f >= 1e4
    print(f"读入 {f.size} 点；高频(>=1e4Hz) {int(hi.sum())} 点；"
          f"末点实测相位 {np.degrees(np.angle(Z[-1])):.1f}°")
    for K, il in ((1, True), (2, True), (1, False), (2, False)):
        c = SEL.fit_voigt(f, Z, K, include_L=il, seed=0, kw=KW, sigma=sig)
        s = c["raw"]["s"]
        # 逐后验样本算 Z，再逐点取中位数（与 §14 图形口径一致）
        n = len(s[list(s)[0]].reshape(-1))
        Zs = np.stack([B.model_z(f, B.sample_params(s, i, il), include_L=il)
                       for i in np.linspace(0, n - 1, 60).astype(int)])
        Zm = np.median(Zs, axis=0)
        m = SM.metrics(f, Z, Zm)
        hr = np.abs(Zm[hi] - Z[hi]) / np.abs(Z[hi])
        print(f"K={K} include_L={il}: rms={m['rms_rel']:.2f}% acf={m['acf']:.3f} "
              f"Rhat={c['rhat']:.3f} div={c['div']} 高频残差={np.mean(hr)*100:.1f}% "
              f"末点相位 fit={np.degrees(np.angle(Zm[-1])):.1f}°")


if __name__ == "__main__":
    main()
