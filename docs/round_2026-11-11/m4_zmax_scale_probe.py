"""R21 探针：先验尺度 `zmax` 的**数据依赖**对跨族 `Δelpd` 有多大影响。

要回答的问题
------------
`bm_top.fit_top` 与 `bm_core.make_model` 都取 `zmax = max|Z|` 作为先验尺度
（`Rs`、`R1`、`Q1`… 的中位都是它的倍数）。逐点 PSIS-LOO 留出第 j 点时，
严格说 `zmax` 应该重算，现有实现没有。于是有两个问题要分开：

1. **对不对**：不对，这不是标准 `p(Z_j | Z_{-j})`，`elpd` 不能写成留一预测性能
   —— 这一条已由 R21 定性认定，本脚本不重复论证；
2. **会不会改变结论**：跨族 `Δelpd` 在 `zmax` 被移动后会不会翻转？
   —— 这才是本脚本要量的东西，也是「要不要为它重跑 LOO」的依据。

做法
----
同一谱、同一 seed、同一预算，把 `zmax` 换成 `s·max|Z|`（`s ∈ {0.5, 1, 2}`）
两族各拟一次，报 `Δelpd(s)` 与决策。同时**不跑拟合**地算出「留出最坏一点后
`zmax` 实际会变成多少」——这条给出 `s` 的现实范围，用来判断极端的 ±2× 是否多余。
若该现实偏移 >2%，脚本会**自动**把这一档也加上（标 `realistic=True`），
因为「LOO 到底把尺度挪了多少」比人为 ±2× 更有说服力。

口径与硬边界
------------
· 两族都是 `sigma=None` + `sigma_floor=0`（跨族同类口径，R3），预算取
  `bm_select.BUDGETS[--budget]`（**默认 `light`**：s≠1 时先验被整体平移，
  `standard` 档会频繁触顶、墙钟时间不可控）；配对用
  `bm_select.paired_diff`（与选择器同源）。同一谱的所有尺度**必须**用同一档。
· 候选**写死**：nested = `parallel/L=2`、voigt = `K=1`（§14 同款基线对）。
  不跑族内选优 —— 本探针只量「同一对候选在尺度移动下的漂移」，
  不重做 §14/A2 的结论。
· 绝对 `elpd` **不可**与 §13/§14/§18/§19 的数字对比；只读 `Δelpd` 的**差**。
· 全部 `exploratory`；不改判据、不改阈值、不接 `bmdrt.py`。

用法
----
    python docs/round_2026-11-11/m4_zmax_scale_probe.py --dry
    python docs/round_2026-11-11/m4_zmax_scale_probe.py
    python docs/round_2026-11-11/m4_zmax_scale_probe.py \
        --specs AZ31-8Li_50 AZ31-4Li_0 --scales 0.5 1.0 2.0
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

import bm_real as R          # noqa: E402
import bm_select as S        # noqa: E402
import bm_top as T           # noqa: E402

SPEC_DIR = os.path.join("out", "_m6_samples", "z_csv")
#: 一条已过（acf 0.341）、一条未过（acf 0.819）——跨族判定的两种典型结局各取一条
SET_DEFAULT = ("AZ31-8Li_50", "AZ31-4Li_0")
LAYERS, K_VOIGT = 2, 1
SCALES_DEFAULT = (0.5, 1.0, 2.0)
SEED = 0


def zmax_loo_stats(f, Z) -> dict:
    """留出单点后 `zmax` 会怎么变（纯算术，不拟合）。

    `elpd_loo` 的第 j 项对应「留出 j」。若 j 恰好是 `|Z|` 最大的点，
    `zmax` 的变化最大 —— 这条给出**现实的**尺度扰动上界。
    """
    w = np.abs(Z)
    zf = float(w.max())
    order = np.argsort(-w)
    out = {"zmax_full": zf, "worst_point_hz": float(f[order[0]])}
    ratios = []
    for j in order[:5]:                      # 只看最大的 5 个点就够（上界在此）
        rest = np.delete(w, j)
        ratios.append(float(rest.max() / zf))
    out["ratio_drop_top1"] = ratios[0]
    out["ratio_drop_min"] = float(min(ratios))
    out["ratio_max_abs_shift_pct"] = 100.0 * float(max(abs(1.0 - r) for r in ratios))
    return out


def _fit_nested(f, Z, *, zmax, bud, seed=SEED) -> dict:
    o = T.fit_top(f, Z, branch="parallel", layers=LAYERS, seed=seed,
                  n_warmup=int(bud["n_warmup"]), n_samples=int(bud["n_samples"]),
                  num_chains=int(bud["num_chains"]),
                  target_accept=float(bud["target_accept"]),
                  max_tree_depth=int(bud["max_tree_depth"]),
                  sigma=None, sigma_floor=float(S.SIGMA_FLOOR),
                  zmax_override=zmax, progress=False)
    return dict(elpd=float(o["elpd"]), se=float(o["elpd_se"]),
                pointwise=np.asarray(o["pointwise_elpd"], float),
                rhat=o.get("rhat_max"), div=int(o.get("divergences") or 0),
                ess=o.get("ess_min"), zmax_tag=o.get("zmax"))


def _fit_voigt(f, Z, *, zmax, bud, seed=SEED) -> dict:
    r = R.fit_one(f, Z, K_VOIGT, False, "1.0", seed=seed, kw=dict(bud),
                  sigma_floor=float(S.SIGMA_FLOOR), adsorption=None,
                  zmax_override=zmax)
    return dict(elpd=float(r["elpd"]), se=float(r["elpd_se"]),
                pointwise=np.asarray(r["pointwise_elpd"], float),
                rhat=r.get("rhat_max"), div=int(r.get("ndiv") or 0),
                ess=r.get("ess_min"))


def _decide(d, dse) -> dict:
    if not np.isfinite(dse) or float(dse) <= 0:
        return dict(decision="indistinguishable", z=None,
                    why="dse 不可用")
    z = d / dse
    sw = d >= S.SWITCH_DELTA and z >= S.SWITCH_Z
    bk = -d >= S.SWITCH_DELTA and -z >= S.SWITCH_Z
    return dict(decision=("nested" if sw else "voigt" if bk else "indistinguishable"),
                z=float(z), why="")


def one(spec: str, scales, bud: dict) -> dict:
    f, Z = R.load_spectrum(os.path.join(SPEC_DIR, spec + ".csv"))
    zs = zmax_loo_stats(f, Z)
    # 现实尺度：留出 `|Z|` 最大的点后 zmax 的真实变化（纯算术，不是假设）。
    # 偏离 1.0 超过 2% 才单独加一个点——那条才是「LOO 到底把尺度挪了多少」。
    use = list(scales)
    if zs["ratio_max_abs_shift_pct"] > 2.0:
        use = sorted(set([float(x) for x in use] + [zs["ratio_drop_top1"]]))
    rows, deltas = [], []
    print(f"  尺度 {use}；zmax={zs['zmax_full']:.4g}", flush=True)
    for s in use:
        zm = s * zs["zmax_full"]
        n = _fit_nested(f, Z, zmax=zm, bud=bud)
        print(f"    s={s:>6.4f} nested 完成：elpd {n['elpd']:9.2f} "
              f"R̂ {n['rhat']:.3f} div {n['div']}", flush=True)
        v = _fit_voigt(f, Z, zmax=zm, bud=bud)
        print(f"    s={s:>6.4f} voigt  完成：elpd {v['elpd']:9.2f} "
              f"R̂ {v['rhat']:.3f} div {v['div']}", flush=True)
        d, dse = S.paired_diff(n["pointwise"], v["pointwise"], f=f)
        dec = _decide(d, dse)
        rows.append(dict(scale=float(s), zmax=float(zm),
                         realistic=bool(abs(float(s) - zs["ratio_drop_top1"]) < 1e-9),
                         nested={k: n[k] for k in ("elpd", "se", "rhat", "div", "ess")},
                         voigt={k: v[k] for k in ("elpd", "se", "rhat", "div", "ess")}))
        deltas.append(dict(scale=float(s), d_elpd=float(d), dse=float(dse),
                           realistic=bool(abs(float(s) - zs["ratio_drop_top1"]) < 1e-9),
                           **dec))
    ds = [x["d_elpd"] for x in deltas]
    decs = {x["decision"] for x in deltas}
    d_real = [x["d_elpd"] for x in deltas if x["realistic"]]
    return dict(spec=spec, n_pts=int(f.size), zmax_stats=zs, fits=rows,
                delta=deltas,
                summary=dict(d_min=float(min(ds)), d_max=float(max(ds)),
                             d_span=float(max(ds) - min(ds)),
                             # 只看「真实 LOO 会挪到的那一档」：这一档才是现实扰动
                             d_at_realistic=(None if not d_real else
                                             float(np.mean(d_real))),
                             decisions=sorted(decs),
                             decision_stable=len(decs) == 1,
                             # 相对判据 4 nats：尺度移动带来的 Δ 漂移占多少
                             span_over_switch=float((max(ds) - min(ds)) / S.SWITCH_DELTA)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=list(SET_DEFAULT))
    ap.add_argument("--scales", type=float, nargs="+", default=list(SCALES_DEFAULT))
    ap.add_argument("--json", default=os.path.join("out", "m4_zmax_scale_probe.json"))
    ap.add_argument("--budget", choices=sorted(S.BUDGETS), default="light",
                    help="探针预算档（默认 light=1200/600）。s≠1 时先验被整体平移，"
                         "standard 档会频繁触顶、墙钟时间不可控，故默认降一档；"
                         "**同一谱的所有尺度必须用同一档**，否则差异里混进预算效应")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    bud = dict(S.BUDGETS[a.budget])

    print(f"R21 探针：zmax × {a.scales}；候选写死 nested=parallel/L={LAYERS} vs "
          f"voigt K={K_VOIGT}；σ 自由(floor={S.SIGMA_FLOOR})；预算 {a.budget} {bud}")
    print(f"  判据（只读，不改）：Δelpd≥{S.SWITCH_DELTA:g} 且 z≥{S.SWITCH_Z:g} 才切")
    if a.dry:
        for sp in a.specs:
            f, Z = R.load_spectrum(os.path.join(SPEC_DIR, sp + ".csv"))
            z = zmax_loo_stats(f, Z)
            print(f"  {sp:<14} {f.size} 点  zmax={z['zmax_full']:.4g}  "
                  f"留出最大点后 ratio={z['ratio_drop_top1']:.4f}"
                  f"（最大偏移 {z['ratio_max_abs_shift_pct']:.2f}%）")
        return 0

    res = []
    for sp in a.specs:
        print(f"\n=== {sp} ===", flush=True)
        try:
            r = one(sp, a.scales, bud)
        except Exception as e:
            print(f"  失败：{type(e).__name__}: {e}", flush=True)
            res.append(dict(spec=sp, error=f"{type(e).__name__}: {e}"))
            continue
        z = r["zmax_stats"]
        print(f"  zmax={z['zmax_full']:.4g}（留出最坏点 ratio {z['ratio_drop_top1']:.4f}，"
              f"最大偏移 {z['ratio_max_abs_shift_pct']:.2f}%）")
        for fit, d in zip(r["fits"], r["delta"]):
            tag = "←现实LOO" if fit["realistic"] else ""
            print(f"  s={fit['scale']:>6.4f}: nested elpd {fit['nested']['elpd']:9.2f} "
                  f"(R̂ {fit['nested']['rhat']:.3f} div {fit['nested']['div']})  "
                  f"voigt {fit['voigt']['elpd']:9.2f} "
                  f"(R̂ {fit['voigt']['rhat']:.3f} div {fit['voigt']['div']})  "
                  f"→ Δ {d['d_elpd']:+8.2f}  z {('%.2f' % d['z']) if d['z'] is not None else 'NA'}  "
                  f"{d['decision']} {tag}", flush=True)
        s = r["summary"]
        print(f"  Δ 跨度 {s['d_span']:.2f} nats（= {s['span_over_switch']:.2f}× 切换阈值）；"
              f"决策 {'稳定' if s['decision_stable'] else '**翻转**'}：{s['decisions']}；"
              f"现实档 Δ={'NA' if s['d_at_realistic'] is None else '%.2f' % s['d_at_realistic']}",
              flush=True)
        res.append(r)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(prereg="R21 / M6_VALIDATION_PLAN.md §14.6 变更日志 3",
                           meta=dict(scales=list(a.scales), seed=SEED, budget=bud,
                                     budget_name=a.budget,
                                     nested=f"parallel/L={LAYERS}",
                                     voigt=f"K={K_VOIGT}",
                                     sigma="free(floor=0)",
                                     switch_delta=S.SWITCH_DELTA,
                                     switch_z=S.SWITCH_Z,
                                     # bm_real.fit_one 目前不返回 ess_min → voigt 侧记 None
                                     ess_note="voigt 侧 ESS 未由 bm_real.fit_one 返回（None）",
                                     exploratory=True),
                           rows=res), fh, ensure_ascii=False, indent=1)
        stab = [r for r in res if r.get("summary", {}).get("decision_stable")]
        print(f"\n决策稳定 {len(stab)}/{len(res)}；已写 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
