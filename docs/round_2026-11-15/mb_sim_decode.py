"""M-B-SIM-01 解盲端：合并 `truth/` 与 `blind_results/`，按 PROTOCOL §7 出台判定。

只在判别结果全部落盘后运行。做四件事：
1. 分组指标汇总（acf / 分频段 / rms / L_eq / 健康）；
2. `AUC(B vs C)` 与配对 bootstrap 95% CI，并分层到两个噪声档；
3. B 组内「acf ~ 寄生参数」敏感性 + D 组最近质心归因；
4. 按 §7 的类型 A/B/C/D 判定，写 `summary/`。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evidence", "nested_topology"))

import bm_real as R      # noqa: E402
import bm_top as T       # noqa: E402

SIM = os.path.join("out", "mb_sim01")
AUC_MIN, AUC_MIN_STRAT = 0.90, 0.80
FP_A_MAX, TRIG_MIN, AUC_B_MAX = 0.10, 0.80, 0.75
ACF_MB, ACF_HF_MIN = 0.50, 0.60
RHO_MAX, CROSS_TOL = 0.90, 0.05
BANDS = ((0.01, 1.0), (1.0, 100.0), (100.0, 1e5))


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(allv.size, float)
    ranks[order] = np.arange(1, allv.size + 1)
    # 处理并列：取平均秩
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    rp = ranks[:pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return float("nan")
    rx = np.argsort(np.argsort(x[m]))
    ry = np.argsort(np.argsort(y[m]))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def band_max(r, k):
    b = r["band_acf"][k]
    v = [x for x in (b["re"], b["im"]) if x is not None]
    return max(v) if v else float("nan")


def load_all() -> tuple[dict, dict]:
    truth = {}
    for fn in os.listdir(os.path.join(SIM, "truth")):
        if fn.startswith("truth_") and fn.endswith(".json"):
            t = json.load(open(os.path.join(SIM, "truth", fn), encoding="utf-8"))
            truth[t["sim_id"]] = t
    blind = {}
    rd = os.path.join(SIM, "blind_results")
    for fn in os.listdir(rd):
        if fn.endswith(".json"):
            r = json.load(open(os.path.join(rd, fn), encoding="utf-8"))
            blind[r["sim_id"]] = r
    return truth, blind


def structural(truth: dict) -> dict:
    """结构偏差 Δ(f) = Z_meas − Z_ideal(θ)，用于 type-D 的「阻抗层面近似相同」判据。"""
    out = {}
    for sid, t in truth.items():
        f, Z = R.load_spectrum(os.path.join(SIM, "generated", sid + ".csv"))
        Zi = T.z_nested(f, t["theta"], branch="parallel", layers=2)
        out[sid] = dict(f=f, d=Z - Zi, Zabs=np.abs(Zi))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)

    truth, blind = load_all()
    ids = sorted(set(truth) & set(blind))
    bad = [s for s in ids if blind[s].get("status") != "ok"]
    ids = [s for s in ids if s not in bad]
    print(f"可用 {len(ids)} 条（truth {len(truth)} / blind {len(blind)}，坏 {len(bad)}）")

    grp = {s: truth[s]["group"] for s in ids}
    by = lambda g: [s for s in ids if grp[s] == g]     # noqa: E731
    A, B, C1, C2, D = by("A"), by("B"), by("C1"), by("C2"), by("D")
    C = C1 + C2
    inst = {s: truth[s]["inst"] for s in ids}

    # ---- 1. 分组汇总 --------------------------------------------------------
    rows = []
    print("\n%-4s %3s %7s %7s %7s %7s %7s %7s" %
          ("grp", "n", "acf", "acfHF", "rms%", "rmsHF%", "L_eq", "bad"))
    for g, ss in (("A", A), ("B", B), ("C1", C1), ("C2", C2), ("D", D)):
        if not ss:
            continue
        acf = np.array([blind[s]["acf"] for s in ss])
        ahf = np.array([band_max(blind[s], "100-100000") for s in ss])
        rms = np.array([blind[s]["rms_rel"] for s in ss])
        rhf = np.array([blind[s]["band_rms"]["100-100000"] for s in ss])
        lq = np.array([blind[s]["L_eq_uH"] for s in ss])
        nb = sum(1 for s in ss if blind[s]["rhat_max"] > 1.05 or blind[s]["div"] > 0)
        print("%-4s %3d %7.3f %7.3f %7.2f %7.2f %7.1f %7d" %
              (g, len(ss), np.median(acf), np.nanmedian(ahf), np.median(rms),
               np.nanmedian(rhf), np.median(lq), nb))
        rows.append(dict(group=g, n=len(ss), acf_med=float(np.median(acf)),
                         acf_iqr=float(np.subtract(*np.percentile(acf, [75, 25]))),
                         acf_hf_med=float(np.nanmedian(ahf)), rms_med=float(np.median(rms)),
                         rms_hf_med=float(np.nanmedian(rhf)), L_eq_med=float(np.median(lq)),
                         trig_rate=float(np.mean(acf > ACF_MB)), n_health_bad=nb))

    # ---- 2. AUC(B vs C) + 配对 bootstrap ------------------------------------
    fa = lambda ss: np.array([blind[s]["acf"] for s in ss])   # noqa: E731
    auc0 = auc(fa(B), fa(C))
    rng = np.random.default_rng(20261115)
    boots = []
    for _ in range(a.boot):
        pick = rng.integers(0, 20, 20)
        bb = [s for s in B if inst[s] in set(pick.tolist())]
        cc = [s for s in C if inst[s] in set(pick.tolist())]
        if len(bb) > 1 and len(cc) > 1:
            boots.append(auc(fa(bb), fa(cc)))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) >= 20
               else (float('nan'), float('nan')))
    auc_pct = {}
    for p in (2.0, 4.0):
        bb = [s for s in B if truth[s]["pct"] == p]
        cc = [s for s in C if truth[s]["pct"] == p]
        auc_pct[p] = auc(fa(bb), fa(cc))
    print(f"\nAUC(B vs C, acf) = {auc0:.3f}  95%CI [{lo:.3f}, {hi:.3f}]"
          f"   分层：2% {auc_pct[2.0]:.3f} / 4% {auc_pct[4.0]:.3f}")

    # ---- 3. B 组内参数敏感性 + D 归因 ---------------------------------------
    sens = []
    for p in ("Rs", "Ls", "Cp", "Rp"):
        v = np.log10([truth[s]["phi"][p] for s in B])
        sens.append(dict(param=p, rho=spearman(v, fa(B))))
    print("B 组 acf~log(φ) Spearman：" +
          "  ".join(f"{d['param']} {d['rho']:+.2f}" for d in sens))

    feats = lambda ss: np.array([[blind[s]["acf"], band_max(blind[s], "100-100000"),
                                  blind[s]["band_rms"]["100-100000"],
                                  blind[s]["band_rms"]["1-100"]]
                                 for s in ss]).reshape(-1, 4)  # noqa: E731
    UA = np.vstack([feats(A), feats(B), feats(C)])
    mu, sd = UA.mean(0), UA.std(0) + 1e-12
    z = lambda ss: (feats(ss) - mu) / sd    # noqa: E731
    cenB, cenC = z(B).mean(0), z(C).mean(0)
    dD = z(D)
    dB = np.linalg.norm(dD - cenB, axis=1)
    dC = np.linalg.norm(dD - cenC, axis=1)
    frac_to_B = float(np.mean(dB < dC))
    print(f"D 组最近质心归因：归为 B 的比例 {frac_to_B:.2f}（n={len(D)}）")

    # ---- 4. type-D 的阻抗层面判据（配对）------------------------------------
    st = structural(truth)
    gap = []
    for i in range(20):
        sb = [s for s in B if inst[s] == i]
        sc = [s for s in C if inst[s] == i]
        if not sb or not sc:
            continue
        d1, d2 = st[sb[0]]["d"], st[sc[0]]["d"]
        gap.append(float(np.median(np.abs(d1 - d2) / st[sb[0]]["Zabs"])))
    gap_med = float(np.median(gap))
    print(f"配对结构偏差差 |Δ_B − Δ_C|/|Z| 中位 {gap_med * 100:.1f}%")

    # ---- 5. 判定 ------------------------------------------------------------
    fpA = float(np.mean(fa(A) > ACF_MB))
    trigB, trigC = float(np.mean(fa(B) > ACF_MB)), float(np.mean(fa(C) > ACF_MB))
    hfB = float(np.nanmedian([band_max(blind[s], "100-100000") for s in B]))
    hfC = float(np.nanmedian([band_max(blind[s], "100-100000") for s in C]))
    rho_max = float(np.nanmax([abs(d["rho"]) for d in sens]))
    nan = float("nan")
    cond = dict(
        fp_A_ok=bool(fpA <= FP_A_MAX), auc_ok=bool(auc0 >= AUC_MIN),
        ci_ok=bool(np.isfinite(lo) and lo >= 0.75),
        strat_ok=bool(all(np.isfinite(v) and v >= AUC_MIN_STRAT
                          for v in auc_pct.values())),
        single_param_ok=bool(rho_max < RHO_MAX), d_attr_ok=bool(0.20 <= frac_to_B <= 0.80))
    del nan
    is_A = all(cond.values())
    is_C = (hfB < ACF_HF_MIN) or (hfC < ACF_HF_MIN)
    is_D = (gap_med < CROSS_TOL) and (auc0 < AUC_B_MAX)
    is_B = (trigB >= TRIG_MIN) and (trigC >= TRIG_MIN) and (auc0 < AUC_B_MAX)
    verdict = ("C_model_insufficient" if is_C else
               "D_non_identifiable" if is_D else
               "B_detect_only" if is_B else
               "A_discriminative" if is_A else "UNDETERMINED")
    print(f"\n误报率 A {fpA:.2f} | 触发率 B {trigB:.2f} C {trigC:.2f} | "
          f"acf_HF B {hfB:.3f} C {hfC:.3f} | max|ρ| {rho_max:.2f}")
    print("条件：" + json.dumps(cond, ensure_ascii=False))
    print("判定：" + verdict)

    # ---- 6. 落盘 ------------------------------------------------------------
    sd_dir = os.path.join(SIM, "summary")
    os.makedirs(sd_dir, exist_ok=True)
    os.makedirs(os.path.join(SIM, "decoded"), exist_ok=True)
    json.dump(dict(verdict=verdict, conditions=cond, auc=auc0, auc_ci=[float(lo), float(hi)],
                   auc_by_pct={str(k): v for k, v in auc_pct.items()}, fp_A=fpA,
                   trig_B=trigB, trig_C=trigC, acf_hf_B=hfB, acf_hf_C=hfC,
                   rho=rho_max, frac_D_to_B=frac_to_B, gap_paired=gap_med,
                   n_used=len(ids), n_bad=len(bad)),
              open(os.path.join(SIM, "decoded", "decoded.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    with open(os.path.join(sd_dir, "group_metrics.csv"), "w", encoding="utf-8", newline="\n") as fh:
        if not rows:
            raise SystemExit("没有任何可用结果，不写汇总")
        k = list(rows[0])
        fh.write(",".join(k) + "\n")
        for r in rows:
            fh.write(",".join(str(r[x]) for x in k) + "\n")

    with open(os.path.join(sd_dir, "parameter_sensitivity.csv"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("param,rho_acf,n\n")
        for d in sens:
            fh.write(f"{d['param']},{d['rho']:.4f},{len(B)}\n")

    pred = ["B" if x else "C" for x in (dB < dC)]
    with open(os.path.join(sd_dir, "confusion_matrix.csv"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("true,attr_B,attr_C\nD,%d,%d\n" % (pred.count("B"), pred.count("C")))

    per = [dict(sim_id=s, group=grp[s], inst=inst[s], pct=truth[s]["pct"],
                acf=blind[s]["acf"], acf_hf=band_max(blind[s], "100-100000"),
                rms=blind[s]["rms_rel"], L_eq=blind[s]["L_eq_uH"],
                rhat=blind[s]["rhat_max"], div=blind[s]["div"]) for s in ids]
    json.dump(per, open(os.path.join(SIM, "decoded", "per_spectrum.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n已写 {sd_dir}/group_metrics.csv, parameter_sensitivity.csv, confusion_matrix.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
