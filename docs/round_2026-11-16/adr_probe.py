#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""adr（吸附三支路并联）验证脚本：跑一条谱，出统一 6 子图。

研究路径（`admission=provisional` 同类），不改生产默认族。
用法：python adr_probe.py <谱.csv> [out.png]
"""
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]          # bmdrt 仓库根
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import bm_real as R          # noqa: E402
import bm_top as T           # noqa: E402
import spectrum_metrics as SM  # noqa: E402
from bm_figure_unified import plot_fit  # noqa: E402


def run(spec_path: str, out_png: str) -> dict:
    f, Z = R.load_spectrum(spec_path)
    o = np.argsort(f)
    f, Z = f[o], Z[o]
    sig = np.stack([0.02 * np.abs(Z), 0.02 * np.abs(Z)])   # §13/§14 口径：逐点 2%
    o2 = T.fit_adr(f, Z, n_warmup=2000, n_samples=1000, num_chains=2,
                   target_accept=0.98, max_tree_depth=10, seed=0,
                   sigma=sig, sigma_from_data=True, progress=False)
    p = {k: float(np.median(np.asarray(o2["s"][k]).reshape(-1))) for k in T.ADR_KEYS}
    Zm = T.z_adr(f, p)
    m = SM.metrics(f, Z, Zm)
    print(f"\n=== {Path(spec_path).stem} ===")
    print(f"  rms_rel {m['rms_rel']:.2f}%   acf {m['acf']:.3f} "
          f"(re/im {m['acf_re']:.3f}/{m['acf_im']:.3f})   "
          f"R̂ {o2['rhat_max']}   div {o2['divergences']}")
    print(f"  分频段残差 {[round(b, 1) for b in m['bands_pct']]}  "
          f"最高频端点 {m['top_rel_pct']:.1f}%")
    for k in T.ADR_KEYS:
        print(f"    {k:<8} {p[k]:.5g}")
    plot_fit(
        out_png, f, Z,
        [dict(label=f"adr (rms {m['rms_rel']:.1f}%)", Zc=Zm, color="#c0392b")],
        title=f"{Path(spec_path).stem}  adr 吸附三支路并联（研究性，provisional）",
        meta=(f"rms_rel={m['rms_rel']:.2f}%   acf={m['acf']:.3f}   "
              f"R̂={o2['rhat_max']}   div={o2['divergences']}   "
              f"Rs={p['Rs']:.1f}Ω Rct={p['Rct']:.1f}Ω"),
        sigma_re=sig[0], sigma_im=sig[0])
    return dict(params=p, metrics=m, diag=o2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python adr_probe.py <谱.csv> [out.png]")
        raise SystemExit(2)
    spec = sys.argv[1]
    png = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join("out", "adr_" + Path(spec).stem + ".png")
    os.makedirs(os.path.dirname(png) or ".", exist_ok=True)
    run(spec, png)
    print("图：", png)
