"""E1-3E 执行器：**跨实验**残差结构重复性（判据冻结于 `E1_3E_PREREG.md`）。

与 `e1_repeatability.py` 的区别只有三处，其余全部复用它的判据引擎：
1. 数据来自 `EISmeasurements/csv_3e`（4 条同条件、不同 `exp`），不是 `inbox/e1`；
2. **先过收敛门槛**（任一条 `R̂>1.05` 或 `div>0` ⇒ `blocked_convergence`，不出 J 判定）；
3. 单组，无"两组合并"这一步。

结论标签只能写「跨实验重复」，不得与 E1（同状态）的判定合并或互相背书。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/e1_3e.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evidence", "nested_topology"))
sys.path.insert(0, HERE)

import bm_real as R                      # noqa: E402
import bm_top as T                       # noqa: E402
import m4_holdout2 as H                  # noqa: E402
import e1_repeatability as E1            # noqa: E402  判据引擎唯一来源

SPEC_DIR = os.path.join("EISmeasurements", "csv_3e")
GROUP = tuple(f"3E-3_full_electrolysis_cell_exp{i}_0100mA.csv" for i in (1, 2, 3, 4))
OUT = os.path.join("out", "e1_3e")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args(argv)

    loaded = [R.load_spectrum(os.path.join(SPEC_DIR, fn)) for fn in GROUP]
    fs = [np.asarray(f, float) for f, _ in loaded]
    Zs = [np.asarray(Z, complex) for _, Z in loaded]
    g = E1.check_grid(fs)
    if g["mode"] == "blocked_grid":
        print(f"blocked_grid: {g}")
        return 2
    fs = [f[i] for f, i in zip(fs, g["idx"])]
    Zs = [Z[i] for Z, i in zip(Zs, g["idx"])]

    fit_dir = os.path.join(a.out_dir, "posterior")
    res_dir = os.path.join(a.out_dir, "residuals")
    os.makedirs(fit_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    reps, res_re, res_im, res_re_n, res_im_n = [], [], [], [], []
    for fn, f, Z in zip(GROUP, fs, Zs):
        tag = os.path.splitext(fn)[0]
        rows = H._fit_set(f, Z, [0], save_dir=fit_dir, tag=tag)["rows"][0]
        Zm = T.z_nested(f, rows["params"], branch=H.BRANCH, layers=H.LAYERS)
        res = Zm - Z
        acf_here = max(E1.E.acf_max(res.real), E1.E.acf_max(res.imag))
        rec = dict(file=fn, n=int(f.size), rms_rel=float(rows["rms_rel"]),
                   **E1.acf_pair(res), band_acf=E1.band_acf(f, res),
                   rhat_max=float(rows["rhat_max"]), div=int(rows["div"]),
                   ess_min=float(rows["ess_min"]), params=rows["params"],
                   ci90=rows["ci90"],
                   acf_matches_reported=bool(abs(acf_here - rows["acf"]) <= 5e-3),
                   npz=os.path.join(fit_dir, f"{tag}_seed0.npz"))
        reps.append(rec)
        res_re.append(res.real)
        res_im.append(res.imag)
        res_re_n.append(res.real / np.abs(Z))
        res_im_n.append(res.imag / np.abs(Z))
        np.savetxt(os.path.join(res_dir, f"{tag}_res.csv"),
                   np.column_stack([f, Z.real, Z.imag, Zm.real, Zm.imag,
                                    res.real, res.imag,
                                    res.real / np.abs(Z), res.imag / np.abs(Z)]),
                   delimiter=",", fmt="%.10g",
                   header="f,zre_obs,zim_obs,zre_fit,zim_fit,res_re,res_im,"
                          "res_re_norm,res_im_norm", comments="")
        print(f"{fn:<50} rms {rec['rms_rel']:5.2f}%  acf {rec['acf']:.3f}  "
              f"R̂ {rec['rhat_max']:.3f}  div {rec['div']}  ESS {rec['ess_min']:.0f}")

    # ---- 收敛门槛（§2）----
    bad = [r["file"] for r in reps if r["rhat_max"] > H.RHAT_MAX or r["div"] != 0]
    if bad:
        out = dict(status="blocked_convergence", failed=bad, replicas=reps,
                   note="任一条 R̂>1.05 或 div>0 ⇒ 不出 J 判定（E1_3E_PREREG §2）")
    else:
        pairs = []
        for form, (rr, ii) in dict(raw=(res_re, res_im),
                                   norm=(res_re_n, res_im_n)).items():
            for comp, series in (("re", rr), ("im", ii)):
                for p in E1.pairwise_r(series):
                    pairs.append(dict(form=form, comp=comp, **p))
        rmin = lambda form: E1.aggregate_r([p for p in pairs if p["form"] == form])
        acf_all = [r["acf"] for r in reps]
        acf_range = float(max(acf_all) - min(acf_all))
        null_p95 = E1.surrogate_null_range(
            np.asarray(res_re[0]) + 1j * np.asarray(res_im[0]))
        v_raw, why = E1.classify(rmin("raw"), acf_range, null_p95)
        v_norm, _ = E1.classify(rmin("norm"), acf_range, null_p95)
        verdict, reason = ((v_raw, why) if v_raw == v_norm
                           else ("J3", f"primary({v_raw})!=secondary({v_norm})"))
        if not all(r["acf_matches_reported"] for r in reps):
            verdict, reason = "J3", "residual_not_reproducing_reported_acf"
        o_raw, _ = E1.classify_orig(rmin("raw"), acf_range)
        o_norm, _ = E1.classify_orig(rmin("norm"), acf_range)
        out = dict(status="ok", replicas=reps, pairs=pairs,
                   r_min_primary=rmin("raw"), r_min_secondary=rmin("norm"),
                   r_min_im_only=E1.aggregate_r(
                       [p for p in pairs if p["form"] == "raw" and p["comp"] == "im"]),
                   r_min_re_only=E1.aggregate_r(
                       [p for p in pairs if p["form"] == "raw" and p["comp"] == "re"]),
                   acf_range=acf_range, null_p95_range=null_p95,
                   class_primary=v_raw, class_secondary=v_norm,
                   verdict=verdict, reason=reason,
                   verdict_orig_sec3=(o_raw if o_raw == o_norm else "J3"),
                   label="跨实验重复（跨装夹/跨批次），不是同状态重复")
        print(f"\nmin|r| 主 {out['r_min_primary']:.3f} / 副 {out['r_min_secondary']:.3f} "
              f"| 只 Im {out['r_min_im_only']:.3f} / 只 Re {out['r_min_re_only']:.3f}")
        print(f"acf 极差 {acf_range:.3f}（护栏 null_p95 {null_p95:.3f}）")
        print(f"判定（§13 修正口径）：{verdict}（{reason}）  "
              f"｜§3 原文口径：{out['verdict_orig_sec3']}")

    meta = dict(prereg="docs/round_2026-11-15/E1_3E_PREREG.md",
                group=list(GROUP), grid_mode=g["mode"], n_kept=g["n_kept"],
                fit=dict(entry="m4_holdout2._fit_set", budget=H.BUDGET,
                         sigma_pct=H.SIGMA_PCT, branch=H.BRANCH, layers=H.LAYERS,
                         seed=0),
                thresholds=dict(r_hi=E1.R_HI, r_lo=E1.R_LO,
                                acf_range_gate="removed->guard(null_p95, 95%)"),
                git_commit=E1._git("rev-parse", "HEAD"),
                code_sha256={p: E1.sha256(os.path.join(REPO, p)) for p in
                             ("evidence/nested_topology/m4_holdout2.py", "bm_top.py")},
                generated=E1.time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(os.path.join(a.out_dir, "verdict.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(meta=meta, **out), fh, ensure_ascii=False, indent=1,
                  allow_nan=False)
    print(f"→ {os.path.join(a.out_dir, 'verdict.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
