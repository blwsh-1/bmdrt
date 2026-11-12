"""把「我们侧」AZ31-4Li_100 的**所有已落盘产物**按同一口径重算。

存在的坑（2026-11-12 实测踩过）：`bmdrt.py` 默认族是 voigt（串联 HN，
无电感项），而 M4 主线嵌套拓扑在 `nested_fit.py` 这个**独立研究接口**里，
产物是 `*.nested.json` + `*.nested_curve.csv`，不在 `real_*.json` 里。
只搜 `out/*/real_*.json` 会拿到 voigt 的 122% 残差并误当成「我们的结果」。

本脚本只读、不重拟合、不改任何产物；输出 out/adapt_tune_az31/ours.json。
口径：Rwp[%] = sqrt(mean(|dZ|²/|Z_obs|²))·100（bm_ref_rwp.rwp），
      acf    = bm_experiments.acf_max 对实/虚各取一次再取 max。

注意 nested_fit.py 的 `rms_rel` = Rwp/√2（它把实、虚残差拼起来算 RMS），
两者不能直接比大小。
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bm_result import AnalysisResult         # noqa: E402
from bm_ref_rwp import rwp                    # noqa: E402
from bm_experiments import acf_max             # noqa: E402


def _metric(f, z, zf):
    d = zf - z
    return dict(
        rwp_pct=float(rwp(f, z, zf)),
        acf=float(max(abs(acf_max(d.real)), abs(acf_max(d.imag)))),
        n_obs_im_pos=int(np.sum(z.imag > 0)),
        n_fit_im_pos=int(np.sum(zf.imag > 0)),
        n=int(f.size),
    )


def scan_voigt():
    rows = []
    for jp in sorted(glob.glob(str(ROOT / "out" / "**" / "real_AZ31-4Li_100.json"),
                               recursive=True)):
        d = Path(jp).parent
        try:
            R = AnalysisResult.load(d)
            m = _metric(R.f, R.z_obs, R.z_fit)
            raw = json.loads(Path(jp).read_text(encoding="utf-8"))
            meta_p = d / "meta.json"
            meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
            m.update(
                label=str(d.relative_to(ROOT)).replace("\\", "/"),
                family="voigt (bmdrt.py 默认族)",
                K=raw.get("K"),
                include_L=raw.get("include_L"),
                kk_status=(raw.get("kk") or {}).get("status"),
                sample_grade=raw.get("sample_grade"),
                grade_fail=raw.get("grade_fail"),
                k_selection_status=raw.get("k_selection_status"),
                sigma_mode="free (项目默认)",
                warnings=meta.get("warnings"),
            )
            rows.append(m)
        except Exception as exc:
            rows.append(dict(label=str(d.relative_to(ROOT)).replace("\\", "/"),
                             family="voigt (bmdrt.py 默认族)",
                             error=f"{type(exc).__name__}: {exc}"))
    return rows


def scan_nested():
    rows = []
    for cp in sorted(glob.glob(str(ROOT / "out" / "**" / "AZ31-4Li_100.nested_curve.csv"),
                               recursive=True)):
        p = Path(cp)
        jp = Path(str(cp).replace("_curve.csv", ".json"))
        try:
            t = pd.read_csv(p)
            f = t.f_Hz.to_numpy(float)
            z = t.Zre_obs.to_numpy(float) + 1j * t.Zim_obs.to_numpy(float)
            zf = t.Zre_fit.to_numpy(float) + 1j * t.Zim_fit.to_numpy(float)
            m = _metric(f, z, zf)
            J = json.loads(jp.read_text(encoding="utf-8"))
            cfg, diag = J.get("config", {}), J.get("diagnostics", {})
            m.update(
                label=str(p.parent.relative_to(ROOT)).replace("\\", "/"),
                family="nested (nested_fit.py 研究接口)",
                status=J.get("status"),
                branch=cfg.get("branch"), layers=cfg.get("layers"),
                budget=cfg.get("budget"), sigma_mode=cfg.get("sigma_desc"),
                rms_rel_json=J.get("metrics", {}).get("rms_rel"),
                voigt_ref_rms_rel=(J.get("voigt_reference") or {}).get("rms_rel"),
                rhat_max=diag.get("rhat_max"), ess_min=diag.get("ess_min"),
                divergences=diag.get("divergences"),
                warning=J.get("warning"),
            )
            rows.append(m)
        except Exception as exc:
            rows.append(dict(label=str(p.parent.relative_to(ROOT)).replace("\\", "/"),
                             family="nested", error=f"{type(exc).__name__}: {exc}"))
    return rows


def main():
    rows = scan_voigt() + scan_nested()
    out = ROOT / "out" / "adapt_tune_az31"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ours.json").write_text(
        json.dumps(dict(spectrum="AZ31-4Li_100", n_rows=len(rows), rows=rows),
                   ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    print(f"{'label':52s} {'family':12s} {'Rwp%':>9s} {'acf':>6s} {'Im>0':>7s} 备注")
    for r in rows:
        if "error" in r:
            print(f"{r['label']:52s} {r['family'][:12]:12s} ERROR {r['error'][:50]}")
            continue
        fam = "voigt" if r["family"].startswith("voigt") else "nested"
        note = (f"K={r.get('K')} kk={r.get('kk_status')}" if fam == "voigt"
                else f"{r.get('branch')}/L={r.get('layers')} {r.get('budget')} "
                     f"R̂={r.get('rhat_max')}")
        print(f"{r['label']:52s} {fam:12s} {r['rwp_pct']:9.2f} {r['acf']:6.3f} "
              f"{r['n_fit_im_pos']:3d}/{r['n_obs_im_pos']:<3d} {note}")
    print(f"\n[done] {len(rows)} 行 -> {out / 'ours.json'}")


if __name__ == "__main__":
    main()
