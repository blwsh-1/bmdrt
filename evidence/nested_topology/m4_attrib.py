"""§15 归因执行器：只用实测谱 / 既有门禁，不用「拟合残差大」当理由。

判据全部来自 `M6_VALIDATION_PLAN.md` §15.1，脚本里不新造阈值。
`kk_check` 一律用默认参数（`res_tol=0.05`、`c=0.85`），`n_cut=2.0` 与
`R̂ 1.05` 分别沿用 §14.1 / §14.3。

对照组 = §14 里**已通过**的 4 条谱，用来检验判据有没有区分力（§15.2）。

用法
----
    python evidence/nested_topology/m4_attrib.py                 # 全 14 条
    python evidence/nested_topology/m4_attrib.py --json out/m4_attrib.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bm_kk as K          # noqa: E402
import bm_real as R        # noqa: E402
import bm_top as T         # noqa: E402
from m4_holdout2 import SET14, SPEC_DIR, N_CUT, RHAT_MAX, cut_top  # noqa: E402

HOLDOUT2 = os.path.join("out", "m4_holdout2.json")


def _n_top(f, Z):
    """§14.1 末段局部幂指数（同一底数）；None = 不可判（含感抗段）。"""
    idx = np.argsort(f)[-2:]
    im = -Z.imag[idx]
    if np.any(im <= 0):
        return None
    return float(np.log(im[1] / im[0]) / np.log(f[idx[1]] / f[idx[0]]))


def _band_resid(f, Z, params):
    """§15.3.3：只归 M 的谱才调用，描述残差落在哪（不参与分类）。"""
    Zm = T.z_nested(f, params, branch="parallel", layers=2)
    d = (Zm - Z) / np.abs(Z)
    out = []
    for lo, hi, nm in ((0.01, 1, "低"), (1, 100, "中"), (100, 1e5, "高")):
        m = (f >= lo) & (f < hi)
        if not m.any():
            continue
        same_re = float(np.mean(np.sign(d.real[m]) == np.sign(np.median(d.real[m]))))
        same_im = float(np.mean(np.sign(d.imag[m]) == np.sign(np.median(d.imag[m]))))
        out.append(dict(band=nm, rms=float(np.sqrt(np.mean(np.abs(d[m]) ** 2))),
                        same_re=same_re, same_im=same_im))
    return out


def one(spec: str, rec: dict) -> dict:
    f, Z = R.load_spectrum(os.path.join(SPEC_DIR, spec + ".csv"))
    try:
        kk = K.kk_check(f, Z)                     # 默认参数，一个都不改
        status = str(kk.get("status"))
        attribution = list(kk.get("attribution") or [])
        tail = kk.get("tail") or {}
        tail_cand = bool(tail.get("is_candidate"))
        rms_raw = dict(kk.get("rms_raw") or {})
    except Exception as e:                       # 门禁不可用要显式记，不能静默跳过
        kk = {}
        status, attribution, tail_cand, rms_raw = "unavailable", [], False, {}
        kk["kk_error"] = f"{type(e).__name__}: {e}"

    n = _n_top(f, Z)
    im_at_fmax = float(-Z.imag[int(np.argmax(f))])
    row = rec["rows"][0] if rec.get("rows") else {}
    rhat = float(row.get("rhat_max", float("nan")))
    div = int(row.get("div", 0))

    flags = []
    if status != "ok":
        flags.append("Q:kk")
    if n is not None and n > N_CUT:
        flags.append("Q:jump")
    if (rhat > RHAT_MAX) or div > 0:
        flags.append("S:sampling")
    if im_at_fmax < 0:
        flags.append("K:hf_L")
    if tail_cand:
        flags.append("K:lf_tail")
    cls = ("Q" if any(x.startswith("Q") for x in flags) else
           "S" if any(x.startswith("S") for x in flags) else
           "K" if any(x.startswith("K") for x in flags) else "M")

    out = dict(spec=spec, passed=bool(rec.get("passed_all")), cls=cls, flags=flags,
               status=status, attribution=attribution, n=n,
               tail_is_candidate=tail_cand, rms_raw=rms_raw,
               im_at_fmax=im_at_fmax, rhat=rhat, div=div,
               kk_reason=str(kk.get("reason", "")))
    if cls == "M" and row.get("params"):
        out["band_resid"] = _band_resid(f, Z, row["params"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join("out", "m4_attrib.json"))
    a = ap.parse_args(argv)
    d = json.load(open(HOLDOUT2, encoding="utf-8"))
    rec = {r["spec"]: r for r in d["rows"] if "error" not in r}
    assert len(rec) == len(SET14), f"§14 结果不全：{len(rec)}/{len(SET14)}"

    rows = [one(s, rec[s]) for s in SET14]
    print("§15 归因（判据全部为纯数据 / 既有门禁默认阈值）")
    print(f"{'谱':<14}{'§14':<6}{'类':<3}{'命中':<34}{'kk':>8}{'n':>7}{'tail':>7}{'-Im@1e5':>10}{'R̂':>9}")
    for r in rows:
        print(f"{r['spec']:<14}{'过' if r['passed'] else '未过':<6}{r['cls']:<3}"
              f"{','.join(r['flags']) or '—':<34}{r['status']:>8}"
              f"{('不可判' if r['n'] is None else '%.2f' % r['n']):>7}"
              f"{str(r['tail_is_candidate']):>7}{r['im_at_fmax']:>10.1f}{r['rhat']:>9.3f}")

    fail = [r for r in rows if not r["passed"]]
    ctrl = [r for r in rows if r["passed"]]
    keys = ("Q:kk", "Q:jump", "S:sampling", "K:hf_L", "K:lf_tail")
    print(f"\n区分力（未过 {len(fail)} 条 / 对照 {len(ctrl)} 条）：")
    print(f"{'判据':<14}{'未过命中':>10}{'对照命中':>10}  判读")
    for k in keys:
        a1 = sum(k in r["flags"] for r in fail)
        a2 = sum(k in r["flags"] for r in ctrl)
        useful = a1 > a2
        print(f"{k:<14}{a1:>5}/{len(fail):<4}{a2:>5}/{len(ctrl):<4}  "
              f"{'有区分力' if useful else '无区分力（不得用于解释失败）'}")
    for nm, grp in (("未过组", fail), ("对照组", ctrl)):
        c = {x: sum(r["cls"] == x for r in grp) for x in "QSKM"}
        print(f"  {nm}主类：Q={c['Q']} S={c['S']} K={c['K']} M={c['M']}")
    m = [r["spec"] for r in fail if r["cls"] == "M"]
    print(f"  未解释（M）：{m if m else '无'}")
    for r in fail:
        if r["cls"] == "M":
            print(f"    {r['spec']} 分频段：" + "  ".join(
                f"{b['band']} {b['rms']:.1%}(re同号{b['same_re']:.0%}/im{b['same_im']:.0%})"
                for b in r.get("band_resid", [])))

    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(dict(prereg="M6_VALIDATION_PLAN.md §15", rows=rows,
                       n_fail=len(fail), n_ctrl=len(ctrl)), fh, ensure_ascii=False, indent=1)
    print(f"\n已写 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
