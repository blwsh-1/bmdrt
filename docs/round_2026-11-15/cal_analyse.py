"""CAL-01 执行器：开-短路 + 已知 RC 的最小化实物校准（按 `CAL_PREREG.md` 冻结口径）。

四步：读入 → 四道门 → 联合反解 φ̂ → 与 14 条真实谱的残差对照 → K1/K2/K3。

用法
----
    python docs/round_2026-11-15/cal_analyse.py --selftest   # 合成标准件自检，不写 verdict
    python docs/round_2026-11-15/cal_analyse.py --dry        # 只跑到门禁，缺数据即阻断
    python docs/round_2026-11-15/cal_analyse.py              # 正式
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evidence", "nested_topology"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bm_real as R                     # noqa: E402
import bm_top as T                      # noqa: E402
import m4_holdout2 as H                 # noqa: E402

IN = os.path.join("inbox", "cal")
OUT = os.path.join("out", "cal01")
OBJS = ("short", "open", "rc100")
RC_NOM = 100.0
BAND = (100.0, 1e5)
LO = np.array([0.0, -9.0, -13.0, 2.0])      # Rs, log10 Ls, log10 Cp, log10 Rp
HI = np.array([10.0, -3.0, -6.0, 12.0])
G_SHORT_MAX, G_OPEN_MIN, G_RC_TOL, G_RC_PH, G_REP_TOL = 2.0, 20.0, 0.05, 3.0, 0.02
K_CORR, K_RATIO = 0.70, (0.5, 2.0)
K_MIN_SPECTRA, K_LEQ_COVER = 8, 0.50


def z_fix(f, Ztrue, phi):
    """`Z_meas = Rs + jωLs + 1/(1/Z_true + 1/Rp + jωCp)`；Ztrue=None ⇒ open，=0 ⇒ short。"""
    w = 2.0 * np.pi * np.asarray(f, float)
    ser = float(phi["Rs"]) + 1j * w * float(phi["Ls"])
    if Ztrue is None:
        return ser + 1.0 / (1.0 / float(phi["Rp"]) + 1j * w * float(phi["Cp"]))
    Ztrue = np.asarray(Ztrue, complex)
    y = np.where(np.abs(Ztrue) > 0, 1.0 / np.where(np.abs(Ztrue) > 0, Ztrue, 1.0), 0.0)
    if np.all(np.abs(Ztrue) == 0):
        return ser
    return ser + 1.0 / (y + 1.0 / float(phi["Rp"]) + 1j * w * float(phi["Cp"]))


def unpack(u):
    return dict(Rs=float(u[0]), Ls=10.0 ** float(u[1]),
                Cp=10.0 ** float(u[2]), Rp=10.0 ** float(u[3]))


def fit_fixture(data: dict) -> tuple[dict, float]:
    """data = {'short': (f, Z), 'open': (f, Z), 'rc100': (f, Z)}；联合反解一套 φ̂。"""
    def resid(u):
        phi = unpack(np.clip(u, LO, HI))
        out = []
        for k, (f, Z) in data.items():
            Zt = (0.0 if k == "short" else None if k == "open" else RC_NOM)
            m = z_fix(f, Zt, phi)
            scale = {"short": 1.0, "open": float(np.median(np.abs(Z))), "rc100": RC_NOM}[k]
            r = (m - Z) / scale
            out.append(np.concatenate([r.real, r.imag]))
        return np.concatenate(out)

    best = None
    for s in (np.array([0.1, -6.0, -10.0, 6.0]), np.array([1.0, -5.0, -11.0, 5.0]),
              np.array([0.0, -7.0, -9.0, 7.0])):
        r = least_squares(resid, np.clip(s, LO + 1e-9, HI - 1e-9), bounds=(LO, HI),
                          method="trf", max_nfev=6000)
        if best is None or r.cost < best.cost:
            best = r
    return unpack(np.clip(best.x, LO, HI)), float(np.sqrt(np.mean(best.fun ** 2)))


def load_dir(obj: str) -> list:
    d = os.path.join(IN, obj)
    if not os.path.isdir(d):
        return []
    return [R.load_spectrum(os.path.join(d, x)) for x in sorted(os.listdir(d))
            if x.lower().endswith((".csv", ".txt"))]


def gates(rep: dict) -> dict:
    g, detail = {}, {}
    if any(len(rep[k]) == 0 for k in OBJS):
        return dict(blocked=True, reason="missing_object",
                    missing=[k for k in OBJS if len(rep[k]) == 0])
    sh, op, rc = rep["short"][0], rep["open"][0], rep["rc100"][0]
    ms = (sh[0] >= 1e4) & (sh[0] <= 1e5)
    g["G-1"] = bool(ms.any() and np.max(np.abs(sh[1][ms])) < G_SHORT_MAX)
    detail["G-1"] = float(np.max(np.abs(sh[1][ms]))) if ms.any() else float("nan")
    m1 = op[0] >= 1.0
    zref = float(np.median(np.abs(sh[1][m1]))) if m1.any() else 0.0
    g["G-2"] = bool(m1.any() and np.median(np.abs(op[1][m1])) > G_OPEN_MIN * max(zref, 1e-9))
    detail["G-2"] = float(np.median(np.abs(op[1][m1]))) if m1.any() else float("nan")
    mrc = (rc[0] >= 1.0) & (rc[0] <= 1e4)
    dev = np.abs(np.abs(rc[1][mrc]) - RC_NOM) / RC_NOM
    ph = np.abs(np.angle(rc[1][mrc], deg=True))
    g["G-3"] = bool(mrc.any() and dev.max() < G_RC_TOL and ph.max() < G_RC_PH)
    detail["G-3"] = dict(max_dev=float(dev.max()) if mrc.any() else float("nan"),
                         max_phase_deg=float(ph.max()) if mrc.any() else float("nan"))
    ok4 = True
    for k in OBJS:
        rs = rep[k]
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if rs[i][0].size != rs[j][0].size or not np.allclose(rs[i][0], rs[j][0], rtol=1e-9):
                    ok4 = False
                    continue
                d = np.abs(rs[j][1] - rs[i][1]) / np.abs(rs[i][1])
                ok4 &= bool(np.median(d) < G_REP_TOL)
    g["G-4"] = ok4
    return dict(blocked=not all(g.values()), gates=g, detail=detail,
                all_pass=bool(all(g.values())))


def compare(phi: dict) -> dict:
    """14 条真实谱：res = Z_fit − Z_obs，Δ = fixture(Z_fit) − Z_fit；夹具为真则 res ≈ −Δ。"""
    d = json.load(open(os.path.join("out", "m4_holdout2_ess.json"), encoding="utf-8"))
    rows, lq_obs, lq_pred = [], [], []
    for row in d["rows"]:
        f, Z = R.load_spectrum(os.path.join(H.SPEC_DIR, row["spec"] + ".csv"))
        keep = np.ones(f.size, bool)
        for hz in row.get("dropped_hz") or []:
            keep &= ~np.isclose(f, float(hz), rtol=1e-6)
        f, Z = f[keep], Z[keep]
        Zf = T.z_nested(f, row["rows"][0]["params"], branch=H.BRANCH, layers=H.LAYERS)
        res, Dl = Zf - Z, z_fix(f, Zf, phi) - Zf
        hf = (f >= BAND[0]) & (f <= BAND[1])
        a = np.concatenate([res.real[hf], res.imag[hf]])
        b = np.concatenate([-Dl.real[hf], -Dl.imag[hf]])
        c = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        ratio = (float(np.median(np.abs(res[hf])) / np.median(np.abs(Dl[hf])))
                 if np.median(np.abs(Dl[hf])) > 0 else float("inf"))
        i = int(np.argmax(f))
        lq_obs.append(float(-res.imag[i] / (2 * np.pi * f[i]) * 1e6))
        lq_pred.append(float(-Dl.imag[i] / (2 * np.pi * f[i]) * 1e6))
        rows.append(dict(spec=row["spec"], corr=c, ratio=ratio,
                         ok=bool(np.isfinite(c) and c >= K_CORR and K_RATIO[0] <= ratio <= K_RATIO[1])))
    n_ok = int(sum(r["ok"] for r in rows))
    span_obs = float(np.ptp(lq_obs))
    span_pred = float(np.ptp(lq_pred))
    if n_ok >= K_MIN_SPECTRA:
        verdict, why = "K1_fixture_explains", f"{n_ok}/14 通过 corr&ratio"
    elif (14 - n_ok) >= K_MIN_SPECTRA:
        verdict, why = "K2_fixture_not_enough", f"仅 {n_ok}/14 通过"
    else:
        verdict, why = "K3_undetermined", f"{n_ok}/14 通过，落在中间"
    return dict(rows=rows, n_ok=n_ok, verdict=verdict, why=why,
                L_eq_obs_span=span_obs, L_eq_pred_span=span_pred,
                L_eq_cover=float(span_pred / span_obs) if span_obs > 0 else float("nan"),
                L_eq_obs_range=[float(min(lq_obs)), float(max(lq_obs))],
                L_eq_pred_range=[float(min(lq_pred)), float(max(lq_pred))])


def selftest() -> int:
    f = np.asarray(R.load_spectrum(os.path.join(H.SPEC_DIR, "AZ31-8Li_50.csv"))[0], float)
    truth = dict(Rs=0.42, Ls=2.6e-6, Cp=1.4e-10, Rp=4.0e5)
    data = dict(short=(f, z_fix(f, 0.0, truth)),
                open=(f, z_fix(f, None, truth)),
                rc100=(f, z_fix(f, RC_NOM, truth)))
    phi, fit = fit_fixture(data)
    ok, total, bad = 0, 0, []

    def c(n, cond):
        nonlocal ok, total
        total += 1
        print(f"  [{'ok' if cond else 'FAIL'}] {n}")
        ok += int(bool(cond))
        if not cond:
            bad.append(n)

    c("无噪联合反解 Rs 误差 <1%", abs(phi["Rs"] - truth["Rs"]) / truth["Rs"] < 0.01)
    c("无噪反解 Ls 误差 <5%", abs(np.log10(phi["Ls"] / truth["Ls"])) < 0.02)
    c("无噪反解 Cp 误差 <5%", abs(np.log10(phi["Cp"] / truth["Cp"])) < 0.02)
    c("联合残差 ≈0", fit < 1e-6)
    # 噪声版：2% 复噪声（与真实口径同量级）
    rng = np.random.default_rng(0)
    dn = {}
    for k, (ff, ZZ) in data.items():
        sg = 0.02 * np.maximum(np.abs(ZZ), 1e-3)
        dn[k] = (ff, ZZ + sg * (rng.normal(size=ZZ.size) + 1j * rng.normal(size=ZZ.size)))
    phin, fitn = fit_fixture(dn)
    c("2% 噪声下 Rs 误差 <20%", abs(phin["Rs"] - truth["Rs"]) / truth["Rs"] < 0.20)
    c("2% 噪声下 Ls 误差 <0.3 dex", abs(np.log10(phin["Ls"] / truth["Ls"])) < 0.3)
    c("2% 噪声下 Cp 误差 <0.3 dex", abs(np.log10(phin["Cp"] / truth["Cp"])) < 0.3)
    # 门禁在真值数据上应全过
    rep = dict(short=[data["short"]] * 3, open=[data["open"]] * 3, rc100=[data["rc100"]] * 3)
    g = gates(rep)
    c("合成标准件过四道门", g.get("all_pass") is True)
    # compare 的判据方向：夹具为真 ⇒ corr(res,−Δ) 应高
    d = json.load(open(os.path.join("out", "m4_holdout2_ess.json"), encoding="utf-8"))
    row = [r for r in d["rows"] if r["spec"] == "AZ31-8Li_50"][0]
    f2, Z2 = R.load_spectrum(os.path.join(H.SPEC_DIR, "AZ31-8Li_50.csv"))
    Zf = T.z_nested(f2, row["rows"][0]["params"], branch=H.BRANCH, layers=H.LAYERS)
    obs_fix = z_fix(f2, Zf, truth)                 # 观测 = 夹具作用后的真值
    Dl = z_fix(f2, Zf, truth) - Zf
    hf = (f2 >= BAND[0]) & (f2 <= BAND[1])
    a = np.concatenate([(Zf - obs_fix).real[hf], (Zf - obs_fix).imag[hf]])
    b = np.concatenate([-Dl.real[hf], -Dl.imag[hf]])
    c("夹具为真时 corr(res,−Δ) ≈ 1", float(np.corrcoef(a, b)[0, 1]) > 0.99)
    obs_none = Zf
    a0 = np.concatenate([(Zf - obs_none).real[hf], (Zf - obs_none).imag[hf]])
    c("无夹具时 corr 无定义/为 0", not np.isfinite(np.corrcoef(a0, b)[0, 1]) or
      abs(float(np.corrcoef(a0, b)[0, 1])) < 0.7)
    print(f"\n自检 {ok}/{total} 通过" if not bad else f"\nFAIL {len(bad)}/{total}: {bad}")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    rep = {k: load_dir(k) for k in OBJS}
    miss = [k for k in OBJS if not rep[k]]
    g = gates(rep)
    os.makedirs(OUT, exist_ok=True)
    json.dump(g, open(os.path.join(OUT, "gates.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("门禁：" + json.dumps(g, ensure_ascii=False))
    if g.get("blocked") or g.get("missing"):
        print(f"阻断：缺 {miss}；期望布局 {IN}/<obj>/rep*.csv（obj ∈ {OBJS}）")
        return 2
    if a.dry:
        print("dry：门禁已跑，未做反解")
        return 0
    if not g["all_pass"]:
        print("门禁未全过 ⇒ 按预注册 §3 判 K3，不做反解")
        json.dump(dict(verdict="K3_undetermined", why="gate_failed", gates=g),
                  open(os.path.join(OUT, "verdict.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        return 3

    data = {k: rep[k][0] for k in OBJS}
    phi, fit = fit_fixture(data)
    json.dump(dict(phi=phi, joint_residual=fit),
              open(os.path.join(OUT, "fixture_fit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    cmp = compare(phi)
    json.dump(cmp, open(os.path.join(OUT, "compare.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    out = dict(verdict=cmp["verdict"], why=cmp["why"], n_ok=cmp["n_ok"],
               L_eq_cover=cmp["L_eq_cover"], gates=g)
    json.dump(out, open(os.path.join(OUT, "verdict.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"φ̂ = {json.dumps(phi, ensure_ascii=False)}")
    print(f"{cmp['n_ok']}/14 通过 corr&ratio；L_eq 跨度 实测 {cmp['L_eq_obs_span']:.0f} µH / "
          f"预测 {cmp['L_eq_pred_span']:.0f} µH（覆盖 {cmp['L_eq_cover']:.2f}）")
    print("判定：" + cmp["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
