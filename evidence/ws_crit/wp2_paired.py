"""WP2：六臂配对 LOO（预注册判读表在 `WS_DECISION_PROJECT.md` §WP2）。

两条谱各六臂，`K∈{1,2} × diffusion∈{none,Ws,Wo}`。逐臂独立落盘，不合并后验；
这里只做两件事：

1. 逐臂 elpd / grade / Pareto-k（grade=0 的臂标 `non_evaluable`，不进比较）；
2. 6 选 2 的配对差（`Δ=Σ逐点差`、`dse=√n·std(逐点差)`、可分线 `max(2, 1.5·dse)`）。

    python evidence/ws_crit/wp2_paired.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bm_real as R          # noqa: E402

TAGS = ["k1_none", "k1_Ws", "k1_Wo", "k2_none", "k2_Ws", "k2_Wo"]

SPECTRA = {
    "caseE": dict(
        stem="example_caseE",
        dirs={t: ROOT / "out" / "_caseE_diff" / t for t in TAGS}
        | {"k2_Ws": ROOT / "out" / "_wp2_caseE" / "k2_Ws",
           "k2_Wo": ROOT / "out" / "_wp2_caseE" / "k2_Wo"},
        note="合成真值 R∞=10 + ZARC(30,300Hz,.85) + Ws(A_W=2.03)，31 点 0.01–1e5 Hz",
    ),
    "blank": dict(
        stem="Blank_0.1KBr_pH10_EIS_PtCu_1.2V",
        dirs={t: ROOT / "out" / "_wp2_blank" / t for t in TAGS},
        note="真实谱，50 点 0.1–1e4 Hz（KK ok，0.1 Hz 相位 −2.3°）",
    ),
}


def load(d: Path, stem: str):
    npz = list(d.glob(f"*/real_{stem}_samples.npz"))
    if not npz:
        return None
    z = np.load(npz[0], allow_pickle=True)
    s = {k: z[k] for k in z.keys()}
    js = json.loads((npz[0].parent / f"real_{stem}.json").read_text(encoding="utf-8"))
    f, Z = s["f"], s["Zre"] + 1j * s["Zim"]
    ll = R.loglik_matrix(s, f, Z, bool(js["include_L"]))
    _, elpd, se, bad_k, pt, pk = R.loo_pointwise(ll)
    return dict(elpd=float(elpd), se=float(se), bad_k=float(bad_k),
                pk_max=float(np.max(pk)), pt=pt, n=int(len(pt)),
                grade=int(js["sample_grade"]), div=int(js["divergences"]),
                cap=float(js["tree_cap"]), t=float(js["t_fit_s"]),
                sampling=js["sampling"], path=str(npz[0].parent))


def verdict(delta: float, dse: float) -> str:
    thr = max(2.0, 1.5 * dse)
    if abs(delta) < thr:
        return "不可区分"
    return "A 优" if delta > 0 else "B 优"


def main():
    for name, spec in SPECTRA.items():
        rows = {}
        for t in TAGS:
            r = load(spec["dirs"][t], spec["stem"])
            if r is None:
                print(f"[{name}] {t}: 缺结果（未跑或未完成）")
                continue
            rows[t] = r
        print(f"\n{'=' * 78}\n=== {name}：{spec['note']}\n{'=' * 78}")
        print(f"{'臂':<9}{'elpd':>9}{'se':>7}{'grade':>6}{'div':>5}{'cap':>7}"
              f"{'k_max':>7}{'t/s':>7}  可用性")
        for t in TAGS:
            r = rows.get(t)
            if r is None:
                print(f"{t:<9}{'—':>9}{'':>7}{'':>6}{'':>5}{'':>7}{'':>7}{'':>7}  缺")
                continue
            ok = "可用" if r["grade"] > 0 else "non_evaluable"
            print(f"{t:<9}{r['elpd']:9.2f}{r['se']:7.2f}{r['grade']:6d}"
                  f"{r['div']:5d}{r['cap']:7.3f}{r['pk_max']:7.2f}{r['t']:7.0f}  {ok}")

        ok_tags = [t for t in TAGS if t in rows and rows[t]["grade"] > 0]
        print(f"\n  配对差（仅 {len(ok_tags)} 个可用臂；正值 = 前者更好）")
        print(f"  {'A − B':<20}{'Δelpd':>9}{'dse':>7}{'|Δ|/dse':>9}  判读")
        for i, a in enumerate(ok_tags):
            for b in ok_tags[i + 1:]:
                dd = rows[a]["pt"] - rows[b]["pt"]
                delta = float(dd.sum())
                dse = float(np.sqrt(dd.size) * np.std(dd, ddof=1))
                print(f"  {a + ' − ' + b:<20}{delta:9.2f}{dse:7.2f}"
                      f"{abs(delta) / dse:9.2f}  {verdict(delta, dse)}")


if __name__ == "__main__":
    main()
