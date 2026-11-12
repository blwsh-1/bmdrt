# -*- coding: utf-8 -*-
"""§9.4.39 执行器：用 KK 自洽性给 §14 的 10 条未过谱分层（6 数据 / 4 模型）。

**只读**：现算 `bm_kk.kk_check()`，不重跑 MCMC，不动任何门禁/阈值/族定义。
判据只用已有字段（`rms_raw.all`、`phase_resid.rms_deg`）；分层线 4% / 2° 是
**描述性切分**，不是判据，不得升格为门禁（见 §9.4.39.3）。

缘起：用户问 `34a8ea5` 的 `kk.phase_resid`（只记账）对残差判断有没有用。
实测：与 KK 幅值残差 r=0.984（近乎共线），对 M-b 的预测力 0.477 vs 0.468
→ 作为判据无新增量；但按 KK 质量分层能区分「数据坏」与「模型不够」。

用法（项目 venv）：
    python evidence/nested_topology/m4_kk_layer.py
    python evidence/nested_topology/m4_kk_layer.py --json out/phase_resid_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import bm_kk as KK          # noqa: E402
import bm_real as R         # noqa: E402

SPECS = ["AZ31-4Li_0", "AZ31-4Li_5", "AZ31-4Li_10", "AZ31-4Li_50",
         "AZ31-8Li_0", "AZ31-8Li_5", "AZ31-8Li_10", "AZ31-8Li_50",
         "AZ31-8Li_150", "AZ31-12Li_0", "AZ31-12Li_5", "AZ31-12Li_10",
         "AZ31-12Li_50", "AZ31-12Li_150"]
SPEC_DIR = os.path.join("out", "_m6_samples", "z_csv")
HOLDOUT_JSON = os.path.join("out", "m4_holdout2.json")

#: 分层线（**描述性**，不是门禁）：KK 复阻抗相对残差 ≤4% 且相位 rms ≤2°。
CLEAN_RMS = 0.04
CLEAN_PHASE_DEG = 2.0


def run(out_json: str) -> dict:
    mc = json.load(open(HOLDOUT_JSON, encoding="utf-8"))
    by_spec = {r["spec"]: r for r in mc["rows"]}

    rows = []
    for sp in SPECS:
        f, Z = R.load_spectrum(os.path.join(SPEC_DIR, sp + ".csv"))
        kk = KK.kk_check(f, Z)
        pr = kk.get("phase_resid") or {}
        rs = kk.get("residual_structure") or {}
        r0 = by_spec[sp]["rows"][0]
        rms = (kk.get("rms_raw") or {}).get("all")
        ph = pr.get("rms_deg")
        clean = (rms is not None and rms <= CLEAN_RMS
                 and ph is not None and ph <= CLEAN_PHASE_DEG)
        rows.append(dict(
            spec=sp,
            passed=bool(by_spec[sp]["verdict"][0].get("passed")),
            acf_mc=float(r0["acf"]), rms_mc=float(r0["rms_rel"]),
            kk_status=kk.get("status"), kk_M=kk.get("M"),
            kk_rms=float(rms) if rms is not None else None,
            kk_rms_low=(kk.get("rms_raw") or {}).get("low"),
            kk_rms_high=(kk.get("rms_raw") or {}).get("high"),
            kk_syst=bool(rs.get("systematic")),
            ph_rms=float(ph) if ph is not None else None,
            ph_max=pr.get("max_abs_deg"), ph_bands=pr.get("bands"),
            ph_trend=pr.get("trend_deg_per_decade"), ph_ac1=pr.get("ac1"),
            ph_syst=bool(pr.get("systematic")),
            clean=bool(clean),
        ))

    # ---- 分组与相关性 ----
    pas = [r for r in rows if r["passed"]]
    fail = [r for r in rows if not r["passed"]]
    fail_clean = [r for r in fail if r["clean"]]
    fail_dirty = [r for r in fail if not r["clean"]]

    def _r(key_a, key_b):
        a = np.array([r[key_a] for r in rows], float)
        b = np.array([r[key_b] for r in rows], float)
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else None

    summary = dict(
        n=len(rows), n_pass=len(pas), n_fail=len(fail),
        n_fail_clean=len(fail_clean), n_fail_dirty=len(fail_dirty),
        phase_pass_mean=float(np.mean([r["ph_rms"] for r in pas])),
        phase_pass_range=[float(min(r["ph_rms"] for r in pas)),
                          float(max(r["ph_rms"] for r in pas))],
        phase_fail_mean=float(np.mean([r["ph_rms"] for r in fail])),
        phase_fail_range=[float(min(r["ph_rms"] for r in fail)),
                          float(max(r["ph_rms"] for r in fail))],
        r_phase_kk=_r("ph_rms", "kk_rms"),
        r_phase_acf=_r("ph_rms", "acf_mc"),
        r_kk_acf=_r("kk_rms", "acf_mc"),
        clean_line=dict(kk_rms=CLEAN_RMS, phase_deg=CLEAN_PHASE_DEG,
                        note="描述性切分，不是判据/门禁"),
        fail_clean_specs=[r["spec"] for r in fail_clean],
        fail_dirty_specs=[r["spec"] for r in fail_dirty],
    )

    out = dict(status="exploratory", prereg="NESTED_TOPOLOGY_PROJECT.md §9.4.39",
               summary=summary, rows=rows)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    json.dump(out, open(out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join("out", "phase_resid_probe.json"))
    a = ap.parse_args(argv)

    out = run(a.json)
    s = out["summary"]
    print(f"{'谱':<14}{'M-b':>5}{'acf':>7}{'KK':>6}{'M':>4}"
          f"{'KKrms%':>8}{'phase°':>8}  分层")
    print("-" * 66)
    for r in out["rows"]:
        print(f"{r['spec']:<14}{'过' if r['passed'] else '—':>5}"
              f"{r['acf_mc']:>7.3f}{str(r['kk_status']):>6}{str(r['kk_M']):>4}"
              f"{100 * r['kk_rms']:>8.2f}{r['ph_rms']:>8.3f}  "
              f"{'数据干净' if r['clean'] else '数据偏差大'}")

    print(f"\n过组 n={s['n_pass']}  phase {s['phase_pass_mean']:.3f}° "
          f"{s['phase_pass_range']}")
    print(f"不过组 n={s['n_fail']}  phase {s['phase_fail_mean']:.3f}° "
          f"{s['phase_fail_range']}")
    print(f"\n不过组里：数据干净 {s['n_fail_clean']} 条 "
          f"{s['fail_clean_specs']}")
    print(f"          数据偏差大 {s['n_fail_dirty']} 条")
    print(f"\nr(phase, KK)={s['r_phase_kk']:+.3f}  "
          f"r(phase, acf)={s['r_phase_acf']:+.3f}  "
          f"r(KK, acf)={s['r_kk_acf']:+.3f}")
    print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
