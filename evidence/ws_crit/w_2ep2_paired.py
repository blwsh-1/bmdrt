"""2EP-2_exp2_0500mA：K×扩散 五臂配对对照（探索性，非预注册）。

复用 WP2 的逐点配对算法（`wp2_paired.py` 的 load/verdict），但判据只作
量级描述：WP2 预注册只覆盖 caseE 与 Blank 两条谱，本谱不在其中。

    python evidence/ws_crit/w_2ep2_paired.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bm_real as R  # noqa: E402

STEM = "2EP-2_exp2_0500mA"
BASE = ROOT / "out" / "_w_2ep2"
TAGS = ["k1_none", "k1_Ws", "k1_Wo", "k2_none", "k2_Ws"]


def load(tag: str):
    d = BASE / tag
    npz = list(d.glob(f"*/real_{STEM}_samples.npz"))
    if not npz:
        return None
    z = np.load(npz[0], allow_pickle=True)
    s = {k: z[k] for k in z.keys()}
    js = json.loads((npz[0].parent / f"real_{STEM}.json").read_text(encoding="utf-8"))
    f, Z = s["f"], s["Zre"] + 1j * s["Zim"]
    ll = R.loglik_matrix(s, f, Z, bool(js["include_L"]))
    _, elpd, se, bad_k, pt, pk = R.loo_pointwise(ll)
    return dict(tag=tag, elpd=float(elpd), se=float(se),
                bad_k=float(bad_k), pk_max=float(np.max(pk)), pt=pt,
                n=int(len(pt)), grade=int(js["sample_grade"]),
                div=int(js["divergences"]), cap=float(js["tree_cap"]),
                js=js, s=s, f=f, Z=Z)


def verdict(delta: float, dse: float) -> tuple[float, str]:
    thr = max(2.0, 1.5 * dse)
    if abs(delta) < thr:
        return thr, "不可区分"
    return thr, ("前者优" if delta > 0 else "后者优")


def main() -> int:
    rows = {t: load(t) for t in TAGS}
    print(f"谱：{STEM}  n={rows['k1_none']['n']}  "
          f"f={rows['k1_none']['f'].min():.3g}–{rows['k1_none']['f'].max():.3g} Hz")
    print("\n=== ① 逐臂（elpd 由逐点 loglik 重算，与落盘值核对） ===")
    print(f"{'arm':<9} {'elpd':>8} {'se':>7} {'k>.7%':>7} {'pk_max':>7} "
          f"{'grade':>5} {'div':>4} {'cap%':>5} {'评估':<14} diffusion_params")
    for t, r in rows.items():
        if not r:
            print(f"{t:<9} MISSING")
            continue
        dp = r["js"].get("diffusion_params") or {}
        ev = "non_evaluable" if r["grade"] == 0 else "evaluable"
        print(f"{t:<9} {r['elpd']:8.2f} {r['se']:7.2f} "
              f"{r['bad_k'] * 100:6.1f}% {r['pk_max']:7.2f} "
              f"{r['grade']:5d} {r['div']:4d} {r['cap'] * 100:5.1f} "
              f"{ev:<14} {json.dumps({k: round(v, 6) for k, v in dp.items()}, ensure_ascii=False)}")

    print("\n=== ② 配对差（Δ=Σ逐点差；dse=√n·std(逐点差)；线=max(2,1.5·dse)） ===")
    print(f"{'A - B':<21} {'Δ':>8} {'dse':>7} {'线':>7}  判读")
    for i, a in enumerate(TAGS):
        for b in TAGS[i + 1:]:
            ra, rb = rows[a], rows[b]
            if not ra or not rb:
                continue
            dvec = ra["pt"] - rb["pt"]
            delta = float(np.sum(dvec))
            dse = float(np.sqrt(len(dvec)) * np.std(dvec, ddof=1))
            thr, v = verdict(delta, dse)
            flag = ""
            if ra["grade"] == 0 or rb["grade"] == 0:
                flag = "  [含 grade=0 臂，仅参考]"
            print(f"{a + ' - ' + b:<21} {delta:8.2f} {dse:7.2f} {thr:7.2f}  {v}{flag}")

    print("\n=== ③ 扩散幅值 vs 噪声（后验中位数 / sd / cv） ===")
    for t in ("k1_Ws", "k1_Wo", "k2_Ws"):
        r = rows[t]
        if not r:
            continue
        s, js = r["s"], r["js"]
        key = "Rw" if t.endswith("Ws") else "Rwo"
        arr = np.asarray(s.get(key))
        if arr is None or arr.size == 0:
            print(f"{t}: 没有 {key} 后验")
            continue
        x = arr.reshape(-1)
        med, sd = float(np.median(x)), float(np.std(x))
        cv = sd / abs(med) if med else float("inf")
        sig = float(np.median(np.sqrt(js["sigma_re"] ** 2 + js["sigma_im"] ** 2)))
        fmin = float(r["f"].min())
        omega_ref = 2 * np.pi * np.sqrt(fmin * float(r["f"].max()))
        # Rw 定义在几何中心频率；换算回 f_min 处的 |Z_Ws| = A_W/sqrt(omega_min)
        aw = float(js["diffusion_params"].get("Aw", 0))
        at_fmin = aw / np.sqrt(2 * np.pi * fmin) if aw else float("nan")
        print(f"{t}: {key} med={med:.6g} sd={sd:.6g} cv={cv:.2f} "
              f"q05={np.quantile(x, 0.05):.6g} q95={np.quantile(x, 0.95):.6g}")
        print(f"    σ≈{sig:.6g}  A_W={aw:.6g}  |Z_Ws(f_min)|={at_fmin:.6g} Ω "
              f"= {at_fmin / sig:.3f}σ  |Rw|/zmax={med / np.max(np.abs(r['Z'])):.4%}")

    print("\n=== ④ 加 W 后 RC 分量漂移（简并诊断） ===")
    for t in TAGS:
        r = rows[t]
        if not r:
            continue
        c = r["js"]["components"]
        print(f"{t:<9} n={len(c)} " + " | ".join(
            f"f0={x['f0']:.1f}Hz w={x['w']:.5f}Ω φ={x['phi']:.3f}"
            for x in c))
        dres = r["js"].get("diffusion_resolution") or {}
        print(f"    dres.level={dres.get('level')} f_D={dres.get('f_D')} "
              f"resolvable={dres.get('resolvable')} max_reldiff={dres.get('max_reldiff')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
