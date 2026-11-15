"""M-B-SIM-01 附加分析 S6：跨体系「单一共享夹具」能否同时解释两个体系（无 MCMC）。

只用生成端同一套变换（`mb_sim_generate.z_parasitic`）做判据校准，不改主实验的 80 条谱。

判据形式（**联合反解**，这是「一个夹具解释两个体系」的正确提法）：

- **H1 情形**：两体系用**同一组** φ 测量（各自样品不同）。用两体系数据**联合**反解一个 φ，
  残差应只剩噪声水平 ⇒ 检验有功效。
- **H2 情形**：两体系各自带**不同的**样品侧额外结构，无夹具。联合反解一个 φ 后，
  残差应显著大于噪声 ⇒ 检验有特异性（夹具不能冒充样品结构）。

阈值（写死）：联合残差 rms ≤ 5% 判「可解释」。功效 = P(H1 残差 ≤5%)，特异性 = P(H2 残差 >5%)。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evidence", "nested_topology"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bm_top as T                        # noqa: E402
import mb_sim_generate as G               # noqa: E402

N_FIX = 12
LO = np.array([0.0, -9.0, -13.0, 3.0])       # Rs, log10 Ls, log10 Cp, log10 Rp
HI = np.array([10.0, -3.0, -7.0, 12.0])
TOL = 0.05


def unpack(u):
    return dict(Rs=float(u[0]), Ls=10.0 ** float(u[1]),
                Cp=10.0 ** float(u[2]), Rp=10.0 ** float(u[3]))


def pack(phi):
    return np.array([phi["Rs"], np.log10(phi["Ls"]), np.log10(phi["Cp"]),
                     np.log10(max(phi["Rp"], 1e3))])


def deviation(f, theta, phi):
    Zs = T.z_nested(f, theta, branch="parallel", layers=2)
    if phi is None:
        return Zs, np.zeros_like(Zs)
    return Zs, G.z_parasitic(f, Zs, phi) - Zs


def add_noise(rng, f, th, delta, pct):
    Zs = T.z_nested(f, th, branch="parallel", layers=2)
    sg = pct / 100.0 * np.abs(Zs)
    return delta + sg * (rng.normal(size=delta.size) + 1j * rng.normal(size=delta.size))


def fit_joint(f, items):
    """items = [(theta, delta_obs, |Z|), ...]；反解**一个**共享 φ。"""
    def resid(u):
        out = []
        for th, d, za in items:
            _, dp = deviation(f, th, unpack(np.clip(u, LO, HI)))
            r = (dp - d) / za
            out.append(np.concatenate([r.real, r.imag]))
        return np.concatenate(out)

    best = None
    for s in (pack(dict(Rs=1.0, Ls=1e-5, Cp=1e-10, Rp=1e6)),
              pack(dict(Rs=0.2, Ls=1e-6, Cp=1e-9, Rp=1e5)),
              pack(dict(Rs=3.0, Ls=1e-4, Cp=1e-11, Rp=1e7))):
        r = least_squares(resid, np.clip(s, LO + 1e-9, HI - 1e-9),
                          bounds=(LO, HI), method="trf", max_nfev=4000)
        if best is None or r.cost < best.cost:
            best = r
    return unpack(np.clip(best.x, LO, HI)), float(np.sqrt(np.mean(best.fun ** 2)))


def main() -> int:
    f = G.grid()
    pool = G.truth_pool()
    rng = np.random.default_rng(np.random.SeedSequence([20261115, 4242]))
    rows = []
    keys = ("Rs", "Ls", "Cp", "Rp")
    for k in range(N_FIX):
        phi = G.draw_phi(rng)
        th1, th2 = G.draw_theta(rng, pool[:7]), G.draw_theta(rng, pool[7:])
        pct = (2.0, 4.0)[k % 2]
        Za1 = np.abs(T.z_nested(f, th1, branch="parallel", layers=2))
        Za2 = np.abs(T.z_nested(f, th2, branch="parallel", layers=2))
        # H1：共享 φ
        d1 = add_noise(rng, f, th1, deviation(f, th1, phi)[1], pct)
        d2 = add_noise(rng, f, th2, deviation(f, th2, phi)[1], pct)
        phi_h1, e_h1 = fit_joint(f, [(th1, d1, Za1), (th2, d2, Za2)])
        # H2：无夹具、各带不同样品侧结构
        h1 = add_noise(rng, f, th1, G.z_extra(f, G.draw_psi(rng, "C1")), pct)
        h2 = add_noise(rng, f, th2, G.z_extra(f, G.draw_psi(rng, "C2")), pct)
        phi_h2, e_h2 = fit_joint(f, [(th1, h1, Za1), (th2, h2, Za2)])
        amp_h1 = float(np.median(np.concatenate([np.abs(d1) / Za1, np.abs(d2) / Za2])))
        amp_h2 = float(np.median(np.concatenate([np.abs(h1) / Za1, np.abs(h2) / Za2])))
        rec = float(np.max(np.abs(
            np.log10(np.maximum(np.array([phi_h1[x] for x in keys]), 1e-30)) -
            np.log10(np.maximum(np.array([phi[x] for x in keys]), 1e-30)))))
        phi_h2_ok = {x: phi_h2[x] for x in keys}
        rows.append(dict(k=k, pct=pct, phi=phi, phi_h1=phi_h1, phi_h2=phi_h2_ok,
                         err_h1=e_h1, err_h2=e_h2, amp_h1=amp_h1, amp_h2=amp_h2,
                         recover_max_log_err=rec))
        print(f"[{k + 1}/{N_FIX}] pct={pct:.0f}%  H1 联合残差 {e_h1 * 100:5.2f}%"
              f"   H2 联合残差 {e_h2 * 100:6.2f}%   φ 反解 max|Δlog10| {rec:.4f}")

    e1 = np.array([r["err_h1"] for r in rows])
    e2 = np.array([r["err_h2"] for r in rows])
    out = dict(n_fix=N_FIX, tol=TOL, rows=rows, summary=dict(
        h1_med=float(np.median(e1)), h1_p90=float(np.percentile(e1, 90)),
        h2_med=float(np.median(e2)), h2_p10=float(np.percentile(e2, 10)),
        power=float(np.mean(e1 <= TOL)), specificity=float(np.mean(e2 > TOL)),
        recover_med=float(np.median([r["recover_max_log_err"] for r in rows]))))
    s = out["summary"]
    print(f"\nH1 联合残差 中位 {s['h1_med'] * 100:.2f}% (p90 {s['h1_p90'] * 100:.2f}%)"
          f"  ⇒ 功效 {s['power']:.2f}")
    print(f"H2 联合残差 中位 {s['h2_med'] * 100:.2f}% (p10 {s['h2_p10'] * 100:.2f}%)"
          f"  ⇒ 特异性 {s['specificity']:.2f}")
    # 条件化的特异性：只在「样品侧结构明显高于噪声（>2×pct）」的例子上看
    sel = [r for r in rows if r["amp_h2"] > 2.0 * r["pct"] / 100.0]
    spec_cond = (float(np.mean([r["err_h2"] > TOL for r in sel])) if sel else float("nan"))
    s["n_strong"] = len(sel)
    s["specificity_strong"] = spec_cond
    print(f"  其中「结构 >2×噪声」的 {len(sel)}/{len(rows)} 例：特异性 {spec_cond:.2f}"
          f"（H2 结构幅度中位 {np.median([r['amp_h2'] for r in rows]) * 100:.2f}%）")
    out["verdict"] = ("shared_fixture_test_usable"
                      if s["power"] >= 0.8 and s["specificity"] >= 0.8
                      else "shared_fixture_test_not_usable")
    print("判定：" + out["verdict"])
    p = os.path.join("out", "mb_sim01", "summary", "shared_fixture.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("已写 " + p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
