# -*- coding: utf-8 -*-
"""L1'/L2/L3 接入的只读验证（不拟合、不改数据）。

查三件事
--------
1. **L1'（`diffusion_gate` 的数据侧端点走向）**：`to_inductive` 是否优先登记为
   感抗尾；`still_45`/`to_cap`/`to_plateau` 是否只记录家族、不改门控结论。
2. **L2（触发）**：`is_candidate` 为真时 `gate.status == "candidate"`，
   调用方据此在残差无系统性时也补跑 Ws。
3. **L3（份额）**：`ws_share_low` 只写进 `select_mechanism(...)["note"]`，
   **不改变 `choice`**；同时把窗口依赖的实测值摊开（阈值未标定的证据）。

产物：`l123_check.json`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/eis训练数据/bmdrt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evidence" / "ws_crit"))

import bm_real as R                            # noqa: E402
import bm_ws as WS                             # noqa: E402
import p_after_e2 as P                         # noqa: E402


def _corpus():
    d = {}
    for tag, f, Z in P.corpus():
        d[tag] = (f, Z)
    return d


def gate_case(f, Z, is_cand, nw=0, note=""):
    kk = {"tail": dict(is_candidate=bool(is_cand), slope_ny=1.02, slope_z=-0.03)}
    g = R.diffusion_gate(kk, {"n_warburg_like": nw}, pres=None, pk=None, f=f, Z=Z)
    wd = g.get("ws_diag") or {}
    return dict(tag=note, is_candidate=is_cand, n_warburg_like=nw,
                status=g["status"], enter=g["enter"],
                term=wd.get("term"), family=wd.get("family"),
                verdict=wd.get("verdict"),
                band_dec=wd.get("band_dec"), slope_im=wd.get("slope_im"))


def main():
    C = _corpus()
    rows = []

    # ---- 1. L1' 四档 + 优先级 ----
    pick = [
        ("e2:AZ31-4Li_100", False, 0, "真实 AZ31：感抗尾"),
        ("ex:example_caseE", True, 0, "真含 Ws（宽窗）"),
        ("ex:example_caseE_warburg", True, 0, "真含 Ws（窄窗）"),
        ("ex:example_caseF_wo", False, 0, "真 Wo：L1 不给候选"),
        ("ex:example_caseE", True, 1, "HN 简并优先于候选"),
    ]
    for tag, cand, nw, note in pick:
        if tag not in C:
            rows.append(dict(tag=tag, error="语料里没有这条"))
            continue
        f, Z = C[tag]
        rows.append(gate_case(f, Z, cand, nw, f"{tag}｜{note}"))

    # ---- 2. L2：is_candidate=True 时 enter 必须为 True（非简并、非感抗） ----
    l2 = []
    for tag, (f, Z) in C.items():
        d = WS.ws_diagnostics(f, Z)
        g = R.diffusion_gate({"tail": dict(is_candidate=bool(d["verdict"] == "candidate"))},
                             {"n_warburg_like": 0}, f=f, Z=Z)
        l2.append(dict(tag=tag, verdict=d["verdict"], gate=g["status"], enter=g["enter"]))
    n_cand = sum(1 for x in l2 if x["enter"])
    n_bad = sum(1 for x in l2 if (x["verdict"] == "candidate") != x["enter"]
                and x["gate"] != "inductive_tail")

    # ---- 3. L3：share 只进 note，不改 choice ----
    l3 = []
    for rel, label in [
        ("out/_caseE_diff/k1_Ws/example_caseE/real_example_caseE_samples.npz",
         "caseE 真 Ws，0.01 Hz 窗"),
        ("out/example_caseE_warburg/real_example_caseE_warburg_samples.npz",
         "caseE 真 Ws，0.1 Hz 窗"),
        ("evidence/b2x/k1ws/example_caseF_wo/real_example_caseF_wo_samples.npz",
         "错元件：Wo 数据用 Ws 拟合"),
    ]:
        p = ROOT / rel
        if not p.exists():
            l3.append(dict(label=label, error="npz 不存在"))
            continue
        s = dict(np.load(p, allow_pickle=True))
        share = WS.ws_share_low(s)
        # ws_res 只有可分辨性、没有可配对 LOO → 必然 comparison_incomplete，
        # 用来验证 share 是否被写进 note、以及 choice 是否仍为 none。
        sel = R.select_mechanism({}, {"diffusion_resolution": {}, "elpd": float("nan")},
                                 {"status": "candidate"}, share=share)
        l3.append(dict(label=label, f_min=share["f_min"], A_W=share["A_W"],
                       share_med=share["share_med"],
                       share_lo=share["share_lo"], share_hi=share["share_hi"],
                       choice=sel["choice"], status=sel["status"],
                       note_has_share=any("简约诊断" in t for t in sel["note"])))

    out = dict(gate_rows=rows, l2=dict(n=len(l2), n_enter=n_cand, n_mismatch=n_bad,
                                      rows=l2),
               l3=l3)
    dst = ROOT / "evidence" / "ws_crit" / "l123_check.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                   encoding="utf-8")

    print(f"{'case':46s} {'is_cand':7s} {'status':15s} {'enter':5s} "
          f"{'term':12s} {'family':18s} {'verdict':11s}")
    for r in rows:
        if "error" in r:
            print(f"{r['tag']:46s} {r['error']}")
            continue
        print(f"{r['tag']:46s} {str(r['is_candidate']):7s} {r['status']:15s} "
              f"{str(r['enter']):5s} {str(r['term']):12s} {str(r['family']):18s} "
              f"{str(r['verdict']):11s}")
    print(f"\nL2：{len(l2)} 条谱，enter={n_cand}，verdict 与 enter 不一致 {n_bad} 条")
    print("\nL3：share 只进 note（choice 一律 none）")
    for r in l3:
        if "error" in r:
            print(f"  {r['label']}: {r['error']}")
            continue
        print(f"  {r['label']:32s} f_min={r['f_min']:<6.3g} A_W={r['A_W']:.2f} "
              f"share={r['share_med']:.3f} [{r['share_lo']:.3f},{r['share_hi']:.3f}] "
              f"choice={r['choice']} note={r['note_has_share']}")
    print(f"\n写入 {dst}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
