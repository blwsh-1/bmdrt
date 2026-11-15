"""M-B-SIM-01 判别端（盲）：只读 `out/mb_sim01/generated/`，不看 `truth/`。

对每条模拟谱跑 §14 冻结档（`m4_holdout2.BUDGET/SIGMA_PCT/BRANCH/LAYERS`）的
`T.fit_top`，再按 `MB_SIM01_PROTOCOL.md` §6 算指标。输出**不含任何组标签**。

用法
----
    python docs/round_2026-11-15/mb_sim_analyze.py --check      # 盲性自检，不拟合
    python docs/round_2026-11-15/mb_sim_analyze.py --limit 2    # 冒烟
    python docs/round_2026-11-15/mb_sim_analyze.py              # 全量（断点续跑）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evidence", "nested_topology"))

import bm_experiments as E   # noqa: E402
import bm_real as R          # noqa: E402
import bm_top as T           # noqa: E402
import m4_holdout2 as H      # noqa: E402  §14 冻结档（BUDGET/SIGMA_PCT/BRANCH/LAYERS）

SIM_ROOT = os.path.join("out", "mb_sim01")
GEN = os.path.join(SIM_ROOT, "generated")
RES = os.path.join(SIM_ROOT, "blind_results")
BANDS = ((0.01, 1.0), (1.0, 100.0), (100.0, 1e5))


def load_gen(sid: str):
    p = os.path.join(GEN, sid + ".csv")
    if "truth" in os.path.abspath(p):
        raise SystemExit("判别端禁止读 truth/：路径含 truth")
    return R.load_spectrum(p)


def _acf_lag(x) -> int:
    x = np.asarray(x, float)
    x = x - x.mean()
    dd = float(x @ x)
    if dd <= 0:
        return 0
    vals = [abs(float(x[:-L] @ x[L:]) / dd) for L in (1, 2, 3, 4, 5)]
    return int(np.argmax(vals) + 1)


def metrics(f, Z, Zm) -> dict:
    w = np.abs(Z)
    res = Zm - Z
    d = res / w
    out = dict(
        rms_rel=float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2))) * 100,
        acf=float(max(E.acf_max(res.real), E.acf_max(res.imag))),
        acf_re=float(E.acf_max(res.real)), acf_im=float(E.acf_max(res.imag)),
        acf_lag_re=_acf_lag(res.real), acf_lag_im=_acf_lag(res.imag),
        band_acf={}, band_rms={})
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        k = f"{lo:g}-{hi:g}"
        if m.sum() >= 8:
            out["band_acf"][k] = dict(
                re=float(E.acf_max(res.real[m])), im=float(E.acf_max(res.imag[m])),
                n=int(m.sum()))
        else:
            out["band_acf"][k] = dict(re=None, im=None, n=int(m.sum()))
        out["band_rms"][k] = (float(np.sqrt(np.mean(np.abs(d[m]) ** 2))) * 100
                              if m.any() else None)
    # 高频段残差方向（mb_hf_l_equiv 口径）
    hf = f >= 100.0
    i_top = int(np.argmax(f))
    out["L_eq_uH"] = float(-res.imag[i_top] / (2 * np.pi * f[i_top]) * 1e6)
    out["im_res_hf_med"] = float(np.median(res.imag[hf])) if hf.any() else None
    out["re_res_hf_med"] = float(np.median(res.real[hf])) if hf.any() else None
    s = np.sign(np.median(res.imag[hf])) if hf.any() else 0.0
    out["sign_hf_im"] = float(s)
    out["im_sign_changes"] = int(np.sum(np.diff(np.sign(res.imag)) != 0))
    return out


def band_max(acf_band: dict, key: str):
    v = [acf_band[key]["re"], acf_band[key]["im"]]
    v = [x for x in v if x is not None]
    return max(v) if v else None


def analyse(sid: str, voigt: bool = False) -> dict:
    f, Z = load_gen(sid)
    sg = (H.SIGMA_PCT / 100.0 * np.abs(Z), H.SIGMA_PCT / 100.0 * np.abs(Z))
    t0 = time.time()
    o = T.fit_top(f, Z, branch=H.BRANCH, layers=H.LAYERS, seed=0, sigma=sg,
                  sigma_from_data=True, progress=False, **H.BUDGET)
    names = list(T.sampled_keys(H.BRANCH, H.LAYERS))
    pm = {k: float(np.median(o["s"][k])) for k in names}
    met = metrics(f, Z, T.z_nested(f, pm, branch=H.BRANCH, layers=H.LAYERS))
    rec = dict(sim_id=sid, status="ok", n=int(f.size), elapsed_s=round(time.time() - t0, 1),
               n_warmup=H.BUDGET["n_warmup"], n_samples=H.BUDGET["n_samples"],
               n_chains=H.BUDGET["num_chains"], target_accept=H.BUDGET["target_accept"],
               sigma_pct=H.SIGMA_PCT, params=pm,
               rhat_max=float(o["rhat_max"]), div=int(o["divergences"] or 0),
               ess_min=float(o["ess_min"]), **met)
    if voigt:
        s, mc = H.B.fit(f, Z, K=1, alpha=0.5, sigma=sg, zeta_fixed=1.0, seed=0,
                        progress=False, **H.BUDGET)
        p = dict(R_inf=float(np.median(s["R_inf"])), W=float(np.median(s["W"])),
                 pi=np.median(s["pi"], axis=0), tau=np.median(s["tau"], axis=0),
                 phi=np.median(s["phi"], axis=0), zeta=np.median(s["zeta"], axis=0))
        rec["voigt_acf"] = float(max(E.acf_max((H.B.model_z(f, p, include_L=False) - Z).real),
                                     E.acf_max((H.B.model_z(f, p, include_L=False) - Z).imag)))
    return rec


def check_blind() -> int:
    bad = []
    if not os.path.isdir(GEN):
        bad.append(f"缺目录 {GEN}")
    ids = sorted(x[:-4] for x in os.listdir(GEN) if x.endswith(".csv")) if os.path.isdir(GEN) else []
    if len(ids) != 80:
        bad.append(f"generated/ 谱数 {len(ids)} != 80")
    for sid in ids[:5]:
        f, Z = load_gen(sid)
        if f.size != 50:
            bad.append(f"{sid} 点数 {f.size}")
        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(Z))):
            bad.append(f"{sid} 含非有限值")
    # 文件名不得携带组别
    for sid in ids:
        if any(t in sid for t in ("A_", "B_", "C1", "C2", "D_", "group")):
            bad.append(f"{sid} 名字带标签")
    if os.path.isdir(os.path.join(GEN, "truth")):
        bad.append("generated/ 下出现 truth/")
    print(f"盲性自检：{len(ids)} 条，{'通过' if not bad else 'FAIL ' + str(bad)}")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--voigt", action="store_true", help="另跑 voigt 基线（验证相对项等价性）")
    a = ap.parse_args(argv)
    if a.check:
        return check_blind()

    os.makedirs(RES, exist_ok=True)
    ids = (a.ids.split(",") if a.ids else
           sorted(x[:-4] for x in os.listdir(GEN) if x.endswith(".csv")))
    if a.limit:
        ids = ids[:a.limit]
    todo = [s for s in ids if not os.path.exists(os.path.join(RES, s + ".json"))]
    print(f"待跑 {len(todo)}/{len(ids)}（已有结果跳过）", flush=True)
    for k, sid in enumerate(todo, 1):
        t0 = time.time()
        try:
            rec = analyse(sid, voigt=a.voigt)
        except Exception as exc:            # noqa: BLE001
            rec = dict(sim_id=sid, status="error", error=f"{type(exc).__name__}: {exc}")
        json.dump(rec, open(os.path.join(RES, sid + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        msg = (f"[{k}/{len(todo)}] {sid} {rec.get('status')} "
               + (f"acf={rec.get('acf'):.3f} rms={rec.get('rms_rel'):.2f}% "
                  f"rhat={rec.get('rhat_max'):.3f} div={rec.get('div')} "
                  f"{time.time() - t0:.0f}s" if rec.get("status") == "ok" else rec.get("error", "")))
        print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
