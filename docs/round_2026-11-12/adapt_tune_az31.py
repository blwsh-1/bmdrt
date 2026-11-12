"""ADAPT-ECM 4.0.1 在 AZ31-4Li_100 上的**预注册参数网格**（只读对照）。

必须先固定网格再跑，跑完按「同数据集、同口径的全谱 Rwp」排序取最小者；
**全部组都落盘**，不因为某组好看就只报它。D1 是数据侧诊断，参与展示
但不参与排名（它换了数据集，n 从 50 变 39）。

用法（仓库根 bmdrt/ 下）：
    python docs/round_2026-11-12/adapt_tune_az31.py --list
    python docs/round_2026-11-12/adapt_tune_az31.py --all

外部包只读：只调 run_bm_drt_simple.run_bm_drt_programmatic，不改它一行，
不 import GUI，不写任何生产结果。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_ADAPT_DEFAULT = ROOT.parent / "_external" / "ADAPT-ECM-4.0.1" / "src" / "adapt_ecm"
ADAPT_SRC = Path(os.environ.get("ADAPT_ECM_SRC", str(_ADAPT_DEFAULT)))
if not ADAPT_SRC.is_dir():
    raise SystemExit(f"找不到 ADAPT-ECM 源码：{ADAPT_SRC}")
sys.path.insert(0, str(ADAPT_SRC))

from bm_real import load_spectrum          # noqa: E402
from bm_experiments import acf_max          # noqa: E402
from bm_ref_rwp import rwp                 # noqa: E402
import run_bm_drt_simple as A              # noqa: E402

SPECTRUM = "archive/AZ31-4Li_100.txt"
OUT = ROOT / "out" / "adapt_tune_az31"

# ---- 预注册网格：先定后跑，不在看完结果后增删 -------------------------------
# 基础 = GUI 默认（EIS_DRT_WORKBENCH_BM_DRT.py:3712-3718, 9288-9300）
BASE = dict(include_warburg=True, k_hn=5, k_warburg=2,
            warmup=500, samples=800, chains=1, seed=0,
            max_posterior_samples=500, tau_points=300,
            tau_extend_decades=2.0)


def _g(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


GRID = [
    # id,       说明,                                参数,                 是否进排名
    ("S0", "GUI 默认（复现基线）",                    _g(),                     True),
    ("S1", "采样加量 1500/2500 + 全样本区间",         _g(warmup=1500, samples=2500,
                                                       max_posterior_samples=2500), True),
    ("S2", "多链 4 × (800/1200)",                    _g(chains=4, warmup=800,
                                                       samples=1200,
                                                       max_posterior_samples=1200), True),
    ("S3", "关 Warburg 混合",                        _g(include_warburg=False),  True),
    ("S4", "更多弧 k_hn=8 / k_w=3",                   _g(k_hn=8, k_warburg=3),  True),
    ("S5", "最少弧 k_hn=2",                          _g(k_hn=2),               True),
    ("S6", "τ 网格扩域 3 dec / 600 点",               _g(tau_extend_decades=3.0,
                                                       tau_points=600),        True),
    ("S7", "给足预算：W off + k8 + 加量 + τ 扩域",    _g(include_warburg=False, k_hn=8,
                                                       k_warburg=3, warmup=1500,
                                                       samples=2500,
                                                       max_posterior_samples=2500,
                                                       tau_extend_decades=3.0,
                                                       tau_points=600),         True),
    ("S8", "S0 设置换 seed=1（默认随机性）",           _g(seed=1),               True),
    ("S9", "S0 设置换 seed=2（默认随机性）",           _g(seed=2),               True),
    ("S10", "S7 设置换 seed=1（最优稳定性）",          _g(include_warburg=False, k_hn=8,
                                                       k_warburg=3, warmup=1500,
                                                       samples=2500,
                                                       max_posterior_samples=2500,
                                                       tau_extend_decades=3.0,
                                                       tau_points=600, seed=1), True),
    ("S11", "S7 设置换 seed=2（最优稳定性）",          _g(include_warburg=False, k_hn=8,
                                                       k_warburg=3, warmup=1500,
                                                       samples=2500,
                                                       max_posterior_samples=2500,
                                                       tau_extend_decades=3.0,
                                                       tau_points=600, seed=2), True),
    # 数据侧诊断：GUI 的 BM-DRT 没有这个开关（_snapshot_base_spectrum 里
    # discard_inductive=False 写死），这是我们手动筛点，只作归因，不进排名。
    ("D1", "丢 11 个低频感抗点（仅 39 点，非同数据集）", _g(),                  False),
]


def _bands(d: np.ndarray, z: np.ndarray) -> list[float]:
    n = d.size
    e = [0, n // 3, 2 * n // 3, n]
    return [float(np.sqrt(np.mean(np.abs(d[e[i]:e[i + 1]]) ** 2 /
                                  np.abs(z[e[i]:e[i + 1]]) ** 2)) * 100)
            for i in range(3)]          # runner 频率降序：高 / 中 / 低


def evaluate(tag: str, f_in: np.ndarray, z_in: np.ndarray, kw: dict) -> dict:
    t0 = time.time()
    r = A.run_bm_drt_programmatic(f_in, z_in, **kw)
    R = r["result"]
    # runner 会按频率降序重排；一律用它自己返回的数组对齐，不拿输入相减
    f, z = np.asarray(r["freq"], float), np.asarray(r["Z"], complex)
    assert np.array_equal(np.asarray(R["Z_exp"]), z), "Z_exp 与返回的 Z 不一致"
    Zf = np.asarray(R["Z_fit"], complex)
    if not np.all(np.isfinite(Zf)):
        raise ValueError("Z_fit 含非有限值")
    d = Zf - z
    az = np.abs(z)

    qre_lo = np.asarray(R["Zre_fit_q05"], float)
    qre_hi = np.asarray(R["Zre_fit_q95"], float)
    qim_lo = np.asarray(R["Zim_fit_q05"], float)
    qim_hi = np.asarray(R["Zim_fit_q95"], float)
    cov_re = float(np.mean((z.real >= qre_lo) & (z.real <= qre_hi)))
    cov_im = float(np.mean((z.imag >= qim_lo) & (z.imag <= qim_hi)))
    cov_both = float(np.mean(((z.real >= qre_lo) & (z.real <= qre_hi)) &
                             ((z.imag >= qim_lo) & (z.imag <= qim_hi))))

    ps = {x["parameter"]: x["q50"] for x in R["parameter_summary"]}
    tau = np.asarray(R["tau_fine"], float)
    gam = np.asarray(R["gamma_total_q50"], float)
    peak = float(tau[int(np.argmax(gam))]) if gam.size == tau.size else None

    # 反事实：把 11 个观测 Im>0 的点从残差里拿掉，看剩下 39 点的 Rwp。
    # 这不重拟合，只量「全谱 Rwp 有多少来自不可表示的感抗端」。
    cap = z.imag <= 0
    rwp_cap = float(rwp(f[cap], z[cap], Zf[cap])) if cap.sum() >= 8 else None

    return dict(
        tag=tag, settings={k: kw[k] for k in sorted(kw)},
        n=int(f.size), n_obs_im_pos=int(np.sum(z.imag > 0)),
        n_fit_im_pos=int(np.sum(Zf.imag > 0)),
        rwp_pct=float(rwp(f, z, Zf)),
        rwp_without_inductive_pct=rwp_cap,
        acf=float(max(abs(acf_max(d.real)), abs(acf_max(d.imag)))),
        bands_pct=_bands(d, z),
        ci_re_rel=float(np.median((qre_hi - qre_lo) / az)),
        ci_im_rel=float(np.median((qim_hi - qim_lo) / az)),
        ci_cov_re=cov_re, ci_cov_im=cov_im, ci_cov_both=cov_both,
        gamma_peak_tau_s=peak, tau_max_s=float(tau[-1]),
        sigma_re_ohm=ps.get("sigma_re_ohm"),
        sigma_im_ohm=ps.get("sigma_im_ohm"),
        p_W_fraction=ps.get("p_W_fraction"),
        R_inf_ohm=ps.get("R_inf_ohm"),
        elapsed_s=float(time.time() - t0),
        has_quality_flag=any(k in R for k in
                             ("rhat", "divergences", "converged", "status",
                              "warning", "quality")),
        # 逐点留档，便于事后做任何分段/去点归因，不必重跑 NUTS
        curves=dict(f=f.tolist(), Zre=z.real.tolist(), Zim=z.imag.tolist(),
                     Zfit_re=Zf.real.tolist(), Zfit_im=Zf.imag.tolist(),
                     q05_re=qre_lo.tolist(), q95_re=qre_hi.tolist(),
                     q05_im=qim_lo.tolist(), q95_im=qim_hi.tolist()),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.list:
        for tag, desc, kw, rank in GRID:
            diff = {k: v for k, v in kw.items() if v != BASE[k]}
            print(f"{tag:4s} rank={str(rank):5s} {desc:38s} {diff}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    f, z = load_spectrum(SPECTRUM)
    print(f"谱 {SPECTRUM}: n={f.size}  f={f.min()}..{f.max()} Hz  "
          f"Im>0 点={int((z.imag > 0).sum())}", flush=True)
    todo = GRID if (a.all or not a.only) else [g for g in GRID if g[0] in a.only]
    rows = []
    for tag, desc, kw, rank in todo:
        f_in, z_in = f, z
        if tag == "D1":
            keep = z.imag <= 0
            f_in, z_in = f[keep], z[keep]
        print(f"[START] {tag:4s} {desc}  n={f_in.size}", flush=True)
        try:
            row = evaluate(tag, f_in, z_in, kw)
            row.update(description=desc, ranked=rank,
                       dataset="full-50" if tag != "D1" else "capacitive-39")
            print(f"  Rwp={row['rwp_pct']:7.2f}%  acf={row['acf']:.3f}  "
                  f"Rwp(去感抗点)={None if row['rwp_without_inductive_pct'] is None else round(row['rwp_without_inductive_pct'], 2)}%  "
                  f"CI覆盖={row['ci_cov_both']:.2f}  "
                  f"σ_re={row['sigma_re_ohm']}  {row['elapsed_s']:.0f}s", flush=True)
        except Exception as exc:
            row = dict(tag=tag, description=desc, ranked=rank,
                       error=f"{type(exc).__name__}: {exc}",
                       settings={k: kw[k] for k in sorted(kw)})
            print(f"  [ERROR] {row['error']}", flush=True)
        (OUT / f"{tag}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8")
        rows.append({k: v for k, v in row.items() if k != "curves"})
    (OUT / "summary.json").write_text(
        json.dumps(dict(spectrum=SPECTRUM, grid=[g[0] for g in todo],
                        rows=rows), ensure_ascii=False, indent=2,
                   allow_nan=False), encoding="utf-8")
    print(f"\n[done] {len(rows)} 组 -> {OUT}")


if __name__ == "__main__":
    main()
