"""mba_external_validate.py —— 在干净的 2E/3E 谱上正式检验 M-b 判据。

授权：用户 2026-11-22「跑，quick档预算会不会不够，要不要直接s档」。

预算决策（见 6 条标定 `mba_budget_calibration.py`）
--------------------------------------------------
- quick：6 条里 2 条 nested R̂ > 1.05（2.807 / 2.508），acf 不可信；
- standard：修好那 2 条，但**新引入** 1 条（2EP-2 nested R̂ 1.004 → 7.310）。
  ⇒ R̂ 随预算**非单调**，不能靠一次换预算解决。
- 方案：主预算 standard；采样失败（R̂>1.05 或 div>0）自动加 deep
  （4000+2000/2/ta0.97/depth11）复核；**仍失败则标 `non_evaluable` 但保留记录**，
  绝不剔除——失败谱恰恰是 M-b 要研究的对象，剔除会制造选择偏差。

口径
----
默认先验（`zmax_override=None`），与 §14 一致。**不复用 batch21 的 fixed 口径**
（那里的 `zmax_override=5·|Z(fmax)|` 会同时缩放 W/Q/R1）。

判据（不改）
------------
A = `nested acf ≤ 0.50`；B = `voigt acf − nested acf ≥ 0.20`；M-b = A ∧ B。
voigt 取 K=1，nested 取 layers=2，include_L 由 `need_L` 决定（与 batch21 同）。

输出
----
`out/mba_external_validate.json`（先写 out/，复核后再决定是否入库）。

用法
----
    python docs/round_2026-11-22/mba_external_validate.py --n 20
    python docs/round_2026-11-22/mba_external_validate.py --n 20 --json out/mba_external_validate.json
"""

import argparse
import contextlib
import glob
import io
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "docs", "round_2026-11-21"))

import numpy as np                    # noqa: E402
import bm_real as BR                  # noqa: E402
import circuit_runner as RU           # noqa: E402
import cmd_compare as C               # noqa: E402
from batch_fit_21 import SEED, _pick_even        # noqa: E402

ACF_MAX = 0.50
IMP_MIN = 0.20
RHAT_MAX = 1.05

# deep 档：仅在本脚本进程内注册，不写生产模块
RU.BUDGETS["deep"] = dict(n_warmup=4000, n_samples=2000, num_chains=2,
                          target_accept=0.97, max_tree_depth=11)


def sample_paths(n):
    """与 batch21 同一批目录、同一等距取样，只是每条多取几根。"""
    out = []
    for cls, sub in (("2E", "EISmeasurements/csv_2e"),
                     ("3E", "EISmeasurements/csv_3e")):
        ps = _pick_even(glob.glob(os.path.join(REPO, sub, "*.csv")), n)
        out += [(cls, p) for p in ps]
    return out


def fit_family(f, Z, budget, inc_l, family):
    """单族拟合，默认先验口径。`family` ∈ {voigt, nested}。"""
    cands = RU.build_candidates(ks=(1,), layers=(2,), include_adr=False,
                                include_L=inc_l, budget_profile=budget)
    cand = [c for c in cands if c.topology_id == family][0]
    o = RU.fit_candidate(cand, f, Z, seed=SEED,
                         budget_profile=budget, likelihood="free")
    m = o["metrics"]
    return dict(acf=float(m["acf"]), rms=float(m["rms_rel"]),
                rhat=float(o["rhat_max"]),
                div=int(o["divergences"] or 0),
                ess=float(o.get("ess_min") or 0))


def fit_pair(f, Z, budget, inc_l):
    """同一预算下 voigt K1 + nested L2。"""
    return {k: fit_family(f, Z, budget, inc_l, k)
            for k in ("voigt", "nested")}


def bad(pair):
    return any(pair[k]["rhat"] > RHAT_MAX or pair[k]["div"] > 0
               for k in ("voigt", "nested"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="每类取多少条（等距）")
    ap.add_argument("--json", default="out/mba_external_validate.json")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="预算上限（秒），到点后收尾并写已完成的谱；0=不限")
    a = ap.parse_args(argv)

    cases = sample_paths(a.n)
    print(f"M-b 外部验证：{len(cases)} 条干净谱（2E/3E 各 {a.n}），"
          f"standard 主档 + 失败自动 deep，默认先验口径", flush=True)
    t0 = time.time()
    rows = []
    for i, (cls, path) in enumerate(cases, 1):
        if a.seconds and time.time() - t0 > a.seconds:
            print(f"  预算上限 {a.seconds:.0f}s 已到，收尾", flush=True)
            break
        name = os.path.basename(path)
        with contextlib.redirect_stdout(io.StringIO()):
            f, Z = BR.load_spectrum(path)
            inc_l = bool(C.need_L(f, Z))
        pair = fit_pair(f, Z, "standard", inc_l)
        used = "standard"
        if bad(pair):
            # deep 只救**失败那一族**：voigt 在 deep 档会触发
            # bm_real.grade() 的 `r["tds"] is None` 崩溃（生产代码，本轮不改）。
            # 救不回来就保留 standard 结果并标记，绝不丢谱。
            rescued = {}
            for fam in ("voigt", "nested"):
                v = pair[fam]
                if v["rhat"] <= RHAT_MAX and v["div"] == 0:
                    continue
                try:
                    d = fit_family(f, Z, "deep", inc_l, fam)
                    if d["rhat"] <= RHAT_MAX and d["div"] == 0:
                        rescued[fam] = d
                except Exception as ex:              # noqa: BLE001
                    print(f"    ! {name[:22]} deep/{fam} 失败: "
                          f"{type(ex).__name__}", flush=True)
            if rescued:
                pair = {**pair, **rescued}
                used = "standard+deep(" + ",".join(rescued) + ")"
        v, n = pair["voigt"], pair["nested"]
        imp = v["acf"] - n["acf"]
        rec = dict(cls=cls, name=name, path=path, include_L=inc_l,
                   n_pts=int(f.size), budget=used, voigt=v, nested=n,
                   improved=float(imp),
                   A=bool(n["acf"] <= ACF_MAX),
                   B=bool(imp >= IMP_MIN),
                   M_b=bool(n["acf"] <= ACF_MAX and imp >= IMP_MIN),
                   sampling_ok=not bad(pair))
        rows.append(rec)
        print(f"  [{i:>2}/{len(cases)}] {cls} {name[:26]:<27} "
              f"acf {n['acf']:.3f} imp {imp:+.3f} "
              f"R̂ {n['rhat']:.3f}/{v['rhat']:.3f} div {n['div']}/{v['div']} "
              f"{used}", flush=True)

    # ---------- 汇总 ----------
    print()
    print("=" * 96)
    nA = sum(r["A"] for r in rows)
    nB = sum(r["B"] for r in rows)
    nAB = sum(r["M_b"] for r in rows)
    AbB = [r["name"] for r in rows if r["A"] and not r["B"]]
    BnA = [r["name"] for r in rows if r["B"] and not r["A"]]
    print(f"样本 {len(rows)} 条（sampling_ok {sum(r['sampling_ok'] for r in rows)}）")
    print(f"A (nested acf≤0.50) : {nA}/{len(rows)}")
    print(f"B (改善≥0.20)       : {nB}/{len(rows)}")
    print(f"M-b = A∧B           : {nAB}/{len(rows)}")
    print(f"A ⊆ B ? {set(r['name'] for r in rows if r['A']) <= set(r['name'] for r in rows if r['B'])}")
    print(f"  A\\B（acf 达标但改善不够）: {len(AbB)} 条")
    print(f"  B\\A（改善够但 acf 不达标）: {len(BnA)} 条")
    print()
    acfs = np.array([r["nested"]["acf"] for r in rows])
    print("nested acf 升序:", " ".join(f"{x:.3f}" for x in sorted(acfs)))
    print(f"中位 {np.median(acfs):.3f}，≤0.50 占 {100*np.mean(acfs<=0.5):.0f}%")
    # 双峰粗判： acf 在 [0.50,0.65] 的个数（AZ31 双峰此处应≈0）
    gap = int(np.sum((acfs > 0.50) & (acfs < 0.65)))
    print(f"落在 0.50–0.65 过渡带: {gap} 条"
          f"（AZ31 双峰为 1/14；>3 说明外部谱更连续）")
    print("=" * 96)

    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(dict(
            purpose="M-b 判据在干净 2E/3E 谱上的外部验证",
            n_per_class=a.n, budgets=dict(primary="standard", rescue="deep",
                                           deep_cfg=RU.BUDGETS["deep"]),
            prior="default (no zmax_override), same as §14",
            criteria=dict(A="nested acf <= 0.50", B="voigt acf - nested acf >= 0.20"),
            summary=dict(n=len(rows), A=nA, B=nB, M_b=nAB,
                         A_subset_B=not bool(AbB),
                         n_AB=len(AbB), n_BA=len(BnA),
                         acf_median=float(np.median(acfs)),
                         transition_050_065=gap),
            rows=rows), fh, ensure_ascii=False, indent=1)
    print(f"已写 {a.json}（耗时 {time.time()-t0:.0f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
