# -*- coding: utf-8 -*-
"""只读探针：`branch="parallel"`（R∥L）与 `branch="series"`（R+L）在现有谱上分不分得开。

背景（`NESTED_TOPOLOGY_PROJECT.md` §9.4.10.2 已证「带感抗支路后恒等不可能」）：
两条支路的**极限**不同 ——

    Z∥  = R_ad·jωτ/(1+jωτ)    ω→0: → jωL（感性、|Z|→0）；ω→∞: → R_ad（实）
    Z_s = R_L + jωL           ω→0: → R_L（实）；      ω→∞: → jωL（感性、|Z|→∞）

所以**数学上可区分**。真正的问题是：在**当前频率窗 + 当前噪声**下，数据能不能把
这个差异顶出来。两条支路落在观测上等价的唯一通道是
`ωτ ≪ 1 全窗成立`（此时 Z∥ ≈ jωL，退化成一个串联 L）——也就是支路拐点
`f_b = 1/(2πτ)` **落在窗口之上**。

本探针不预设结论，只报三件事：
1. 两支路各自的 `rms_rel` / `acf`（走 `circuit_runner.metrics`，与 §14 同口径）；
2. `Δelpd = elpd_parallel − elpd_series` 与 `Δelpd/se`（同噪声口径，可作相对参考，R21）；
3. 支路拐点 `f_b` 与窗口的比值，以及两条拟合曲线的最大相对差 ——
   若 `f_b ≫ f_max` 且曲线差 < 噪声，则"分不开"的原因是**窗口没覆盖拐点**，
   不是数据量或采样预算。

边界：只读。不改 `bm_*.py`、不改阈值、不接 `bmdrt.py`、不写任何准入结论。
结果一律 `exploratory`。

用法：
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/sp_indistinguish_probe.py
    ... --specs AZ31-12Li_0,AZ31-8Li_0 --budget quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import bm_real as R                       # noqa: E402
import bm_select as SEL                   # noqa: E402
import bm_top as T                        # noqa: E402
import circuit_runner as RU               # noqa: E402

SPEC_DIR = os.path.join("out", "_m6_samples", "z_csv")
OUT_JSON = os.path.join("out", "sp_probe.json")
BRANCHES = ("parallel", "series", "capacitive")


def _branch_summary(branch: str, raw: dict) -> dict:
    """从后验中位参数里取支路量。parallel 给 R_ad/tau_ad，series 给 R_L/tau_L。"""
    s = raw["s"]
    if branch == "parallel":
        R_ad = float(np.median(s["R_ad"]))
        tau = float(np.median(s["tau_ad"]))
        L = R_ad * tau                      # L_ad = R_ad·tau_ad（派生量）
        return dict(R_branch=R_ad, tau_branch=tau, L_ad=L,
                    f_branch=1.0 / (2 * np.pi * tau) if tau > 0 else float("inf"))
    if branch == "capacitive":
        # CPE 支路没有单一拐点：Z_b = Q_b·(ω_ref/ω)^{n_b}·e^{-j n_b π/2}，幂律无特征频率。
        return dict(R_branch=None, tau_branch=None, L_ad=None,
                    f_branch=None,
                    Q_b=float(np.median(s["Q_b"])),
                    n_b=float(np.median(s["n_b"])))
    R_L = float(np.median(s["R_L"]))
    tau = float(np.median(s["tau_L"]))
    return dict(R_branch=R_L, tau_branch=tau, L_ad=R_L * tau,
                f_branch=1.0 / (2 * np.pi * tau) if tau > 0 else float("inf"))


def one(spec: str, layers: int, budget: str, seed: int,
        branches=BRANCHES) -> dict:
    f, Z = R.load_spectrum(os.path.join(SPEC_DIR, spec + ".csv"))
    kmax = int(np.argmax(f))
    out = dict(spec=spec, n=int(f.size), layers=int(layers),
               f_range=[float(f.min()), float(f.max())],
               im_at_fmax=float(Z.imag[kmax]),
               n_im_pos=int((Z.imag > 0).sum()), branches={})

    curves = {}
    for b in branches:
        c = SEL.fit_nested(f, Z, layers, branch=b, seed=seed,
                           kw=dict(RU.BUDGETS[budget]))
        raw = c["raw"]
        pm = RU._median_params_nested(raw, b, layers)
        if b == "series":
            # series 支路只采 R_L / tau_L，`L_ad` 是派生量（sampled_keys 不采它），
            # 但 `z_nested` 要读 `p["L_ad"]`。漏这一步会在正演处 KeyError。
            pm = dict(pm, L_ad=float(pm["R_L"]) * float(pm["tau_L"]))
        Zm = T.z_nested(f, pm, branch=b, layers=layers)
        met = RU.metrics(f, Z, Zm)
        bs = _branch_summary(b, raw)
        curves[b] = Zm
        out["branches"][b] = dict(
            rms_rel=met["rms_rel"], acf=met["acf"],
            rms_bands=met["bands"],
            elpd=float(c["elpd"]), se=float(c["se"]),
            rhat_max=float(raw["rhat_max"]), div=int(raw["divergences"] or 0),
            ess_min=None if raw.get("ess_min") is None else float(raw["ess_min"]),
            **bs,
            f_branch_over_fmax=(None if bs["f_branch"] is None
                                else bs["f_branch"] / float(f.max())),
            inside_window=(None if bs["f_branch"] is None
                           else bool(f.min() <= bs["f_branch"] <= f.max())),
            params={k: (v.tolist() if np.ndim(v) else float(v)) for k, v in pm.items()},
        )

    # 两条拟合曲线在**同一频率网格**上的最大相对差（观测层面等价的直接度量）
    if "parallel" in curves and "series" in curves:
        d = np.abs(curves["parallel"] - curves["series"]) / np.abs(Z)
        out["curve_max_rel_diff"] = float(np.max(d))
        out["curve_max_rel_diff_hf"] = (float(np.max(d[f >= 1e4]))
                                        if (f >= 1e4).any() else None)
        e_p = out["branches"]["parallel"]["elpd"]
        e_s = out["branches"]["series"]["elpd"]
        se = max(out["branches"]["parallel"]["se"], out["branches"]["series"]["se"])
        out["delta_elpd"] = float(e_p - e_s)
        out["delta_z"] = float((e_p - e_s) / se) if se > 0 else float("nan")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="AZ31-12Li_0,AZ31-8Li_0")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--budget", default="quick", choices=sorted(RU.BUDGETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--branches", default=",".join(BRANCHES))
    a = ap.parse_args(argv)

    rows = []
    for spec in [s.strip() for s in a.specs.split(",") if s.strip()]:
        bs = tuple(x.strip() for x in a.branches.split(",") if x.strip())
        r = one(spec, a.layers, a.budget, a.seed, branches=bs)
        rows.append(r)
        print(f"\n=== {r['spec']}  n={r['n']}  f {r['f_range'][0]:.3g}–{r['f_range'][1]:.3g} Hz"
              f"  Im@fmax={r['im_at_fmax']:+.2f}  Im>0 点数 {r['n_im_pos']}")
        print(f"{'branch':<10}{'rms%':>8}{'acf':>7}{'elpd':>10}{'R̂':>7}{'div':>5}"
              f"{'L_ad(H)':>11}{'f_b(Hz)':>11}{'f_b/fmax':>10}  窗口内")
        for b, d in r["branches"].items():
            lad = "—" if d["L_ad"] is None else f"{d['L_ad']:.3g}"
            fb = "—" if d["f_branch"] is None else f"{d['f_branch']:.3g}"
            ratio = "—" if d["f_branch_over_fmax"] is None else f"{d['f_branch_over_fmax']:.3g}"
            win = "—" if d["inside_window"] is None else ("是" if d["inside_window"] else "否")
            print(f"{b:<10}{d['rms_rel']:>8.2f}{d['acf']:>7.3f}{d['elpd']:>10.2f}"
                  f"{d['rhat_max']:>7.3f}{d['div']:>5}{lad:>11}{fb:>11}{ratio:>10}  {win}")
        if "delta_elpd" in r:
            print(f"  Δelpd(par−ser) = {r['delta_elpd']:+.2f}   Δ/se = {r['delta_z']:+.2f}")
        if "curve_max_rel_diff" in r:
            hf = r["curve_max_rel_diff_hf"]
            print(f"  两条曲线最大相对差 = {r['curve_max_rel_diff'] * 100:.3f}%"
                  + (f"（HF 段 {hf * 100:.3f}%）" if hf is not None else ""))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(note="只读探针；exploratory；不构成准入结论",
                       budget=a.budget, seed=a.seed, rows=rows),
                  fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"\n产物：{a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
