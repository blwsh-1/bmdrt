"""吸附元件的可恢复性验证：真值含吸附环的合成谱。

要回答的问题（2026-11-21）
-------------------------
"吸附元件要不要补"这个问题的前提是"它缺席"。实际核对结果：
`bm_ad.z_ad` 与 `bm_top.z_rl_parallel` 是**同一个函数**（max|Δ|=0.0），
有效接点（第二并联节点）已由 nested 的 `branch="parallel"` 与 adr 的
`adsorption_branch` 承载。所以缺的不是元件，是**证据**：
从没验证过"真值里真有吸附环时，这两族能不能把 `R_ad`/`tau_ad` 恢复出来"。

本脚本同时复核 §9.4.12 的结论在**参数层面**是否还成立：
voigt 顶层串联 `adsorption="RL"` 到底能不能补上能力。

设计
----
两条真值谱（1%·|Z| 噪声，固定 seed，不重抽）：

- 谱 A：`nested(layers=2, branch=parallel)` 真值，含 `R_ad=40 Ω, tau_ad=3 ms`；
- 谱 B：`adr` 真值，同样含 `R_ad=40 Ω, tau_ad=3 ms`。

候选：
- 谱 A：nested(parallel) / nested(capacitive，**无感性支路**) /
  voigt K=2 / voigt K=2 + `adsorption="RL"`（顶层串联）；
- 谱 B：adr / voigt K=2 / voigt K=2 + `adsorption="RL"`。

判据（预注册，跑之前写死，未过不重抽、不改阈值）
  A1  谱A：nested(parallel) 的 rms ≤ 2%（噪声水位的 2 倍）
  A2  谱A：nested 恢复 R_ad 在真值 ±40% 内
  A3  谱A：去掉感性支路（capacitive）后 rms 至少差 2pp
  A4  谱A：voigt+ad 相对 voigt 无 ad 的改善 < 20%（复现"接点无效"）
  A5  谱B：adr 的 rms ≤ 2%
  A6  谱B：adr 恢复 R_ad 在真值 ±40% 内

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/ad_recovery.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import bm_core as B                                          # noqa: E402
import bm_select as SEL                                      # noqa: E402
import bm_top as T                                           # noqa: E402
import circuit_runner as RU                                  # noqa: E402

#: 吸附环真值：R_ad = 环直径，tau_ad 决定峰位 f_c = 1/(2πτ) ≈ 53 Hz
AD = dict(R_ad=40.0, tau_ad=3e-3)
SEED = 0
NOISE = 0.01
F = np.logspace(-2, 5, 50)

NESTED_TRUE = dict(Rs=15.0, R1=120.0, Q1=90.0, n1=0.88,
                   R2=60.0, Q2=25.0, n2=0.80, **AD)
ADR_TRUE = dict(Rs=15.0, Q=90.0, n=0.88, Rct=60.0, **AD)


def _noisy(z):
    """1%·|Z| 的相对高斯噪声（实虚部各 1/√2 权重 ⇒ 合成功率恰为 1%）。"""
    rng = np.random.default_rng(SEED)
    sig = NOISE * np.abs(z)
    n = rng.normal(0.0, 1.0, z.size) + 1j * rng.normal(0.0, 1.0, z.size)
    return z + sig * n / np.sqrt(2.0)


def synth_nested(f):
    return _noisy(T.z_nested(f, NESTED_TRUE, branch="parallel", layers=2))


def synth_adr(f):
    return _noisy(T.z_adr(f, ADR_TRUE))


def run_cand(cand, f, Z):
    o = RU.fit_candidate(cand, f, Z, seed=SEED, budget_profile="quick",
                         likelihood="free")
    p = o["params"]
    return dict(topo=cand.topology_id,
                label=f"{cand.topology_id}/"
                      f"{dict(cand.config).get('branch') or dict(cand.config).get('K')}",
                rms=float(o["metrics"]["rms_rel"]),
                acf=float(o["metrics"]["acf"]),
                bands=[None if b is None else float(b)
                       for b in (o["metrics"].get("bands") or [])],
                R_ad=(None if p.get("R_ad") is None else float(p["R_ad"])),
                tau_ad=(None if p.get("tau_ad") is None
                        else float(p["tau_ad"])),
                rhat=float(o["rhat_max"]), div=int(o["divergences"] or 0))


def run_voigt(f, Z, K, adsorption=None):
    """voigt 直接调 `fit_voigt`（不走候选集）：`adsorption` 不是 registry 槽位。"""
    c = SEL.fit_voigt(f, Z, K, seed=SEED, adsorption=adsorption,
                      kw=None, sigma=None)          # kw=None ⇒ 用 QUICK
    raw = c["raw"]
    p = RU._median_params_voigt(raw, diffusion=None)
    if adsorption and p.get("Rad") is None:
        # 开了 adsorption 却没采到 Rad ⇒ 参数根本没进模型，结论会失真，必须炸
        raise RuntimeError("声明 adsorption=RL 但后验里没有 Rad/tau_ad")
    Zm = B.model_z(f, p, include_L=False)
    m = RU.metrics(f, Z, Zm)
    tag = f"voigt K={K}" + ("+ad(顶层串联)" if adsorption else "")
    return dict(topo="voigt", label=tag,
                rms=float(m["rms_rel"]), acf=float(m["acf"]),
                bands=[None if b is None else float(b)
                       for b in (m.get("bands") or [])],
                R_ad=(None if p.get("Rad") is None else float(p["Rad"])),
                tau_ad=(None if p.get("tau_ad") is None
                        else float(p["tau_ad"])),
                rhat=float(raw["rhat_max"]),
                div=int(raw.get("ndiv") or 0))


def main() -> int:
    print(f"吸附真值：R_ad={AD['R_ad']} Ω，tau_ad={AD['tau_ad'] * 1e3:.1f} ms"
          f"（峰位 ≈ {1 / (2 * np.pi * AD['tau_ad']):.0f} Hz）；噪声 {NOISE:.0%}")

    rows_a, rows_b = [], []

    # `--from-json`：只按已落盘的拟合结果复算判据，**不重跑采样**
    # （预注册纪律：判据未过不许重抽 seed / 改阈值；复算不是重抽）
    if "--from-json" in sys.argv:
        with open(os.path.join("out", "ad_recovery.json"),
                  encoding="utf-8") as fh:
            d = json.load(fh)
        rows_a, rows_b = d["rows_nested"], d["rows_adr"]
        print(f"（复算模式：读 out/ad_recovery.json，"
              f"{len(rows_a)} + {len(rows_b)} 行，未重新采样）")
        for r in rows_a + rows_b:
            rad = "—" if r["R_ad"] is None else f"{r['R_ad']:.1f} Ω"
            print(f"  {r['label']:22s} rms={r['rms']:7.3f}%  acf={r['acf']:.3f}  "
                  f"R_ad={rad:>9s}  R̂={r['rhat']:.3f} div={r['div']}")
        return _judge(rows_a, rows_b)

    # ---------- 谱 A：nested 真值 ----------
    ZA = synth_nested(F)
    print("\n=== 谱 A（nested 真值，含吸附环）===")
    cA = RU.build_candidates(ks=(1,), layers=(2,), include_adr=False,
                             branches=("parallel", "capacitive"),
                             budget_profile="quick")
    for cand in cA:
        r = run_cand(cand, F, ZA)
        rows_a.append(r)
        rad = "—" if r["R_ad"] is None else f"{r['R_ad']:.1f} Ω"
        print(f"  {r['label']:22s} rms={r['rms']:7.3f}%  acf={r['acf']:.3f}  "
              f"R_ad={rad:>9s}  R̂={r['rhat']:.3f} div={r['div']}")
    for ads in (None, "RL"):
        r = run_voigt(F, ZA, 2, adsorption=ads)
        rows_a.append(r)
        rad = "—" if r["R_ad"] is None else f"{r['R_ad']:.1f} Ω"
        print(f"  {r['label']:22s} rms={r['rms']:7.3f}%  acf={r['acf']:.3f}  "
              f"R_ad={rad:>9s}  R̂={r['rhat']:.3f} div={r['div']}")

    # ---------- 谱 B：adr 真值 ----------
    ZB = synth_adr(F)
    print("\n=== 谱 B（adr 真值，含吸附环）===")
    cB = [c for c in RU.build_candidates(ks=(1,), layers=(1,), include_adr=True,
                                        branches=("parallel",),
                                        budget_profile="quick")
          if c.topology_id == "adr"]
    for cand in cB:
        r = run_cand(cand, F, ZB)
        rows_b.append(r)
        rad = "—" if r["R_ad"] is None else f"{r['R_ad']:.1f} Ω"
        print(f"  {r['label']:22s} rms={r['rms']:7.3f}%  acf={r['acf']:.3f}  "
              f"R_ad={rad:>9s}  R̂={r['rhat']:.3f} div={r['div']}")
    for ads in (None, "RL"):
        r = run_voigt(F, ZB, 2, adsorption=ads)
        rows_b.append(r)
        rad = "—" if r["R_ad"] is None else f"{r['R_ad']:.1f} Ω"
        print(f"  {r['label']:22s} rms={r['rms']:7.3f}%  acf={r['acf']:.3f}  "
              f"R_ad={rad:>9s}  R̂={r['rhat']:.3f} div={r['div']}")

    out = dict(adsorption_truth=AD, seed=SEED, noise=NOISE,
               n_points=int(F.size),
               nested_truth={k: float(v) for k, v in NESTED_TRUE.items()},
               adr_truth={k: float(v) for k, v in ADR_TRUE.items()},
               rows_nested=rows_a, rows_adr=rows_b)
    os.makedirs("out", exist_ok=True)
    with open(os.path.join("out", "ad_recovery.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, allow_nan=False)

    # ---------- 预注册判据 ----------
    return _judge(rows_a, rows_b)


def _judge(rows_a, rows_b) -> int:
    byA = {r["label"]: r for r in rows_a}
    byB = {r["label"]: r for r in rows_b}
    tol = 0.40 * AD["R_ad"]

    def _ok_rad(r):
        return r["R_ad"] is not None and abs(r["R_ad"] - AD["R_ad"]) <= tol

    n_par = byA["nested/parallel"]
    n_cap = byA["nested/capacitive"]
    v_off, v_on = byA["voigt K=2"], byA["voigt K=2+ad(顶层串联)"]
    # adr 的 config 是空 dict，label 由 run_cand 拼成 "adr/None"；
    # 按 topo 取，避免被 label 拼法绑住。
    adr = [r for r in rows_b if r["topo"] == "adr"][0]

    gain_v = (v_off["rms"] - v_on["rms"]) / v_off["rms"] * 100
    rad_a = "—" if n_par["R_ad"] is None else f"{n_par['R_ad']:.1f} Ω"
    rad_b = "—" if adr["R_ad"] is None else f"{adr['R_ad']:.1f} Ω"

    checks = [
        (f"A1 谱A nested(parallel) rms ≤2%（实得 {n_par['rms']:.3f}%）",
         n_par["rms"] <= 2.0),
        (f"A2 谱A nested 恢复 R_ad ±40%（实得 {rad_a}，"
         f"真值 {AD['R_ad']:.1f}）", _ok_rad(n_par)),
        (f"A3 谱A 去掉感性支路后 rms 差 ≥2pp"
         f"（实得 {n_cap['rms'] - n_par['rms']:.2f}pp）",
         (n_cap["rms"] - n_par["rms"]) >= 2.0),
        (f"A4 谱A voigt+ad 改善 <20%（实得 {gain_v:.1f}%，"
         f"{v_off['rms']:.2f}%→{v_on['rms']:.2f}%）", gain_v < 20.0),
        (f"A5 谱B adr rms ≤2%（实得 {adr['rms']:.3f}%）", adr["rms"] <= 2.0),
        (f"A6 谱B adr 恢复 R_ad ±40%（实得 {rad_b}）", _ok_rad(adr)),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    bad = [n for n, ok in checks if not ok]
    print(f"\n判据 {len(checks) - len(bad)}/{len(checks)} 通过"
          + (f"；未过：{bad}" if bad else ""))
    print("产物：out/ad_recovery.json")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
