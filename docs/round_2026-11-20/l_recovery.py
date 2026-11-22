"""L（顶层串联感抗）的可恢复性验证：真值已知的合成谱。

要回答的问题（2026-11-20）
-------------------------
前面只验证了"三族都能带 L 跑通"，没验证"该加 L 时能不能恢复 L"。
没有这一步，`include_L` 就只是多了一个参数，不是被检验过的能力。

设计
----
- 真值：`Z = R_inf + jωL + Σ HN`（voigt K=1），L = 3 μH；
- 噪声：1%·|Z| 的相对高斯噪声（固定 seed，不重抽）；
- 对照：同一谱分别用 `include_L=True/False` 拟合三族（voigt/nested/adr）；
- 判据（预注册，跑之前写死）：
  L1  `need_L(f, Z)` 必须判 True（否则 auto 根本不会开 L）；
  L2  带 L 的 rms 必须显著低于不带 L（相对降幅 ≥ 30%）；
  L3  L 的后验中位落在真值 ±50% 内；
  L4  不带 L 时高频段 rms 必须明显更差（否则这条谱没有检验力）。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-20/l_recovery.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import bm_core as B                                          # noqa: E402
import circuit_runner as RU                                  # noqa: E402
import cmd_compare as C                                      # noqa: E402

TRUTH = dict(R_inf=20.0, W=120.0, tau=2e-3, phi=0.85, zeta=1.0, L=3e-6)
SEED = 0
NOISE = 0.01


def synth(f):
    p = dict(R_inf=TRUTH["R_inf"], W=np.array([TRUTH["W"]]),
             pi=np.array([1.0]), tau=np.array([TRUTH["tau"]]),
             phi=np.array([TRUTH["phi"]]), zeta=np.array([TRUTH["zeta"]]),
             L=TRUTH["L"])
    z = B.model_z(f, p, include_L=True)
    rng = np.random.default_rng(SEED)
    sig = NOISE * np.abs(z)
    noise = rng.normal(0.0, 1.0, z.size) + 1j * rng.normal(0.0, 1.0, z.size)
    return z + sig * noise / np.sqrt(2.0)


def run_one(cand, f, Z):
    o = RU.fit_candidate(cand, f, Z, seed=SEED, budget_profile="quick",
                         likelihood="free")
    cfg = dict(cand.config)
    return dict(topo=cand.topology_id, include_L=bool(cfg.get("include_L")),
                rms=float(o["metrics"]["rms_rel"]),
                acf=float(o["metrics"]["acf"]),
                bands=[None if b is None else float(b)
                       for b in (o["metrics"].get("bands") or [])],
                L=(None if o["params"].get("L") is None
                   else float(o["params"]["L"])),
                ci90=o["ci90"].get("L"),
                rhat=float(o["rhat_max"]), div=int(o["divergences"] or 0))


def main() -> int:
    f = np.logspace(-2, 5, 50)
    Z = synth(f)
    need = C.need_L(f, Z)
    print(f"need_L = {need}（真值含 L={TRUTH['L'] * 1e6:.1f} μH）")

    rows = []
    for inc in (True, False):
        cands = RU.build_candidates(ks=(1,), layers=(1,), include_adr=True,
                                    include_L=inc, budget_profile="quick")
        for cand in cands:
            r = run_one(cand, f, Z)
            rows.append(r)
            l_txt = "—" if r["L"] is None else f"{r['L'] * 1e6:.2f} μH"
            print(f"  {r['topo']:7s} L={str(r['include_L']):5s} "
                  f"rms={r['rms']:7.3f}%  acf={r['acf']:.3f}  "
                  f"L={l_txt:>10s}  R̂={r['rhat']:.3f} div={r['div']}")

    out = dict(truth=TRUTH, seed=SEED, noise=NOISE, n_points=int(f.size),
               need_L=bool(need), rows=rows)
    os.makedirs("out", exist_ok=True)
    with open(os.path.join("out", "l_recovery.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, allow_nan=False)

    # ---- 预注册判据 ----
    by = {(r["topo"], r["include_L"]): r for r in rows}
    checks = [("L1 need_L 判 True", bool(need) is True)]
    for t in ("voigt", "nested", "adr"):
        on, off = by[(t, True)], by[(t, False)]
        drop = (off["rms"] - on["rms"]) / off["rms"] * 100 if off["rms"] else 0.0
        checks.append((f"L2 {t} 带 L 的 rms 降幅 ≥30%（实得 {drop:.1f}%）",
                       drop >= 30.0))
        lor = on["L"]
        l_txt = "—" if lor is None else f"{lor * 1e6:.2f} μH"
        checks.append((f"L3 {t} 恢复 L 真值 ±50%（实得 {l_txt}）",
                       lor is not None
                       and abs(lor - TRUTH["L"]) <= 0.5 * TRUTH["L"]))
        hi = 2                                   # 高频段 = bands[2]
        gain = (None if (on["bands"][hi] is None or off["bands"][hi] is None)
                else off["bands"][hi] - on["bands"][hi])
        g_txt = "—" if gain is None else f"{gain:.2f}pp"
        checks.append((f"L4 {t} 不带 L 时高频段更差 ≥1pp（实得 {g_txt}）",
                       gain is not None and gain >= 1.0))
    print()
    for name, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    bad = [n for n, ok in checks if not ok]
    print(f"\n判据 {len(checks) - len(bad)}/{len(checks)} 通过"
          + (f"；未过：{bad}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
