# -*- coding: utf-8 -*-
"""M3.5 的 `r` 偏低诊断（**探索性**，`BM_TOP_M3_PLAN.md` §20.20）。

这不是新的验收判据。它的用途是把 §20.17 记下的"`r` 中位系统性偏低"拆成可分别
检验的几条机制，并明确哪些**不能**当证据（链未收敛的那几条运行）。

五个问题，逐条对应代码里的一个量：

    Q1 `R1` / `R2` 的边际后验是否**不对称**收缩？
       → 各自的 `rel_err`、以及**真值在该边际中的分位**（0.5 = 正中）
    Q2 联合后验是否沿**脊线**偏移？
       → `(log R1, log R2)` 的协方差主轴 v，偏移向量与 v 的夹角、沿/跨脊线分量
         （单位为该方向的后验 sd）
    Q3 `n1`/`Q1`/`n2`/`Q2` 与 `r` 是否强相关？
       → 逐 draw 的 `r` 与各参数的 Pearson ρ
    Q4 先验是否把后验中位推向更小的 `r`？
       → 直接从**当前先验**抽 `R1`,`R2` 算 `r`：先验中位、`P_prior(r ≤ 真值)`、
         以及沿脊线方向先验对数密度对 `r` 的斜率（正 = 先验偏好更大 `r`）
    Q5 真值附近与后验高密度区之间是否有系统偏移？
       → 偏移的模长（以脊线 sd 计）、真值在 `r` 后验中的分位、`r` 的偏度

读数规则（§20.20 写定）：后验中位是右偏分布的中位，**不能**直接与真值比"偏差
百分比"；判定"系统性偏低"至少要看 ① 偏离量相对该谱自己的后验 sd，② 符号检验。
链未收敛的运行（`run_valid.pass=False`）只参与描述，**不能**当偏差证据。

样本文件按 `m3_stage2.py --save-samples` 的命名：`<json>.r.d<data>.s<sampler>.samples.npz`。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import m3_stage2 as M                       # noqa: E402  真值、先验常数与门禁的唯一来源

PARAM_KEYS = ("n1", "Q1", "n2", "Q2", "Rs", "R_ad", "tau_ad")


def prior_r_stats(zmax: float, truth_r: float, n: int = 200_000,
                  seed: int = 0) -> dict:
    """先验隐含的 `r` 分布（`R1`,`R2` 同为 `LogNormal(log 0.1·zmax, 2)`）。"""
    rng = np.random.default_rng(seed)
    mu, sg = np.log(0.1 * zmax), 2.0
    r1 = np.exp(rng.normal(mu, sg, n))
    r2 = np.exp(rng.normal(mu, sg, n))
    r = r1 / (r1 + r2)
    lo, med, hi = np.percentile(r, [0.5, 50, 99.5])
    # 沿脊线（log R1 增、log R2 减）方向，先验对数密度如何随 r 变化：
    # 正数 = 先验偏好更大的 r。取真值两侧各 0.5σ_log 的位移求差。
    u0 = np.array([math.log(M.TRUTH["R1"]), math.log(M.TRUTH["R2"])])
    v = np.array([1.0, -1.0]) / math.sqrt(2.0)
    out = []
    for s in (-0.5, 0.5):
        u = u0 + s * v
        lp = -float(((u - mu) ** 2).sum()) / (2 * sg ** 2)
        rr = math.exp(u[0]) / (math.exp(u[0]) + math.exp(u[1]))
        out.append((rr, lp))
    slope = out[1][1] - out[0][1]
    return {"median": float(np.median(r)), "mean": float(r.mean()),
            "p_le_truth": float((r <= truth_r).mean()),
            "p005": float(lo), "p995": float(hi),
            "along_ridge_dlogprior": slope,
            "r_minus": out[0][0], "r_plus": out[1][0],
            "note": "先验量，与数据无关；用于判断'先验是否把 r 推小'"}


def ridge_geometry(R1: np.ndarray, R2: np.ndarray, truth: dict) -> dict:
    """`(log R1, log R2)` 的主轴、真值偏移在该坐标系里的分解。"""
    u = np.column_stack([np.log(R1), np.log(R2)])
    C = np.cov(u, rowvar=False)
    w, V = np.linalg.eigh(C)                       # 升序
    v = V[:, -1]
    if v[0] < 0:
        v = -v                                     # 约定：R1 增大方向为正
    off = u.mean(0) - np.array([math.log(truth["R1"]), math.log(truth["R2"])])
    n = np.linalg.norm(off)
    along = float(off @ v)
    across = float(off @ np.array([-v[1], v[0]]))
    sd_along, sd_across = math.sqrt(w[-1]), math.sqrt(w[0])
    ang = (math.degrees(math.acos(min(1.0, abs(along) / n))) if n > 0 else None)
    return {"eigvals_log": [float(x) for x in w],
            "ridge_dir": [float(x) for x in v],
            "sd_along": float(sd_along), "sd_across": float(sd_across),
            "offset_norm": float(n),
            "offset_angle_deg": ang,
            "along_in_sd": (along / sd_along if sd_along > 0 else None),
            "across_in_sd": (across / sd_across if sd_across > 0 else None),
            "elongation": (sd_along / sd_across if sd_across > 0 else None)}


def sign_test(k_below: int, n: int) -> dict:
    """双侧精确符号检验：`k_below` 个中位低于真值算不算系统性。"""
    if n == 0:
        return {"p_two_sided": None, "p_one_sided": None}
    p1 = sum(math.comb(n, i) for i in range(k_below, n + 1)) / 2 ** n
    p2 = min(1.0, 2 * min(p1, 1 - p1 + (1 / 2 ** n if p1 > 0.5 else 0)))
    # 直接按"离 0.5 等距的两侧"算，避免上面那种补丁式写法：
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    tail = min(sum(pmf[:k_below + 1]), sum(pmf[k_below:]))
    p2 = min(1.0, 2 * tail)
    return {"p_two_sided": float(p2), "p_one_sided": float(p1),
            "k_below": k_below, "n": n}


def diagnose(s: dict, truth: dict, r_truth: float, gate_pass: bool) -> dict:
    """一条样本文件的全部诊断量。`s` 是 npz 里 `k → array` 的字典。"""
    g = lambda k: np.asarray(s[k], float).ravel()
    R1, R2 = g("R1"), g("R2")
    r = R1 / (R1 + R2)
    lo, med, hi = np.percentile(r, [5, 50, 95])
    sd = float(np.std(r, ddof=1))
    skew = float(((r - r.mean()) ** 3).mean() / max(r.std() ** 3, 1e-30))
    marg = {}
    for k in ("R1", "R2"):
        v = g(k)
        lo_k, med_k, hi_k = np.percentile(v, [5, 50, 95])
        marg[k] = {"truth": truth[k], "median": float(med_k),
                   "rel_err": float((med_k - truth[k]) / truth[k]),
                   "truth_pct": float((v < truth[k]).mean()),
                   "covered": bool(lo_k <= truth[k] <= hi_k)}
    summ = R1 + R2
    corr = {k: float(np.corrcoef(r, g(k))[0, 1]) for k in PARAM_KEYS if k in s}
    return {
        "gate_pass": bool(gate_pass),
        "r": {"median": float(med), "mean": float(r.mean()), "sd": sd,
              "p05": float(lo), "p95": float(hi), "truth": r_truth,
              "truth_pct": float((r < r_truth).mean()),
              "bias_median": float(med - r_truth),
              "bias_mean": float(r.mean() - r_truth),
              "bias_in_sd": float((med - r_truth) / sd) if sd > 0 else None,
              "skew": skew, "covered": bool(lo <= r_truth <= hi)},
        "marginal": marg,
        "sum_R1R2": {"median": float(np.median(summ)),
                     "truth": truth["R1"] + truth["R2"],
                     "rel_err": float(np.median(summ) / (truth["R1"] + truth["R2"]) - 1)},
        "ridge": ridge_geometry(R1, R2, truth),
        "corr_with_r": corr,
    }


def summarize(rows: list[dict], prior: dict) -> dict:
    """跨谱汇总：符号检验 + 偏离量 vs 后验 sd。只用门禁通过的运行下结论。"""
    good = [r for r in rows if r["gate_pass"]]
    use = good if len(good) >= 3 else rows
    basis = ("只用门禁通过的运行" if len(good) >= 3
             else "门禁通过的运行不足 3 个 → 全部运行仅作描述")
    meds = [r["r"]["median"] for r in use]
    rt = rows[0]["r"]["truth"] if rows else None
    k_below = sum(1 for m in meds if m < rt) if rt is not None else 0
    biases_sd = [r["r"]["bias_in_sd"] for r in use if r["r"]["bias_in_sd"] is not None]
    return {"basis": basis, "n_used": len(use), "n_gate_pass": len(good),
            "median_r_used": meds, "truth": rt,
            "sign": sign_test(k_below, len(meds)),
            "mean_bias_median": (float(np.mean([m - rt for m in meds])) if meds else None),
            "mean_abs_bias_in_sd": (float(np.mean(np.abs(biases_sd))) if biases_sd else None),
            "max_abs_bias_in_sd": (float(np.max(np.abs(biases_sd))) if biases_sd else None),
            "prior": prior}


def _fmt(x, w=8, s=".4f"):
    # 注意格式串的字段顺序是 [align][sign][width]，所以不能拼成 ">8+.2f"；
    # 统一先格式化再右对齐，避免 sign 与 width 的位置打架。
    return "—".rjust(w) if x is None else format(x, s).rjust(w)


def report(rows: list[dict], summary: dict, meta: dict) -> str:
    L = []
    L.append(f"样本：{meta['n']} 个谱（data_seed/采样器 seed：{meta['keys']}）")
    L.append("**探索性诊断**：不产生 M3.5 判定；门禁未过的运行只作描述（§20.20）")
    L.append("")
    L.append(f"{'谱':>4}{'门禁':>5}{'r 中位':>9}{'r 均值':>9}{'sd_r':>8}{'真值分位':>9}"
             f"{'偏度':>7}{'R1 中位':>9}{'真值分位':>9}{'R2 中位':>9}{'真值分位':>9}"
             f"{'(R1+R2)':>9}{'误差':>8}")
    for r in rows:
        g = r["marginal"]
        L.append(f"{r['label']:>4}{'✓' if r['gate_pass'] else '✗':>5}"
                 f"{r['r']['median']:>9.4f}{r['r']['mean']:>9.4f}{r['r']['sd']:>8.4f}"
                 f"{r['r']['truth_pct']:>9.2f}{r['r']['skew']:>+7.2f}"
                 f"{g['R1']['median']:>9.1f}{g['R1']['truth_pct']:>9.2f}"
                 f"{g['R2']['median']:>9.1f}{g['R2']['truth_pct']:>9.2f}"
                 f"{r['sum_R1R2']['median']:>9.1f}{r['sum_R1R2']['rel_err']:>+8.1%}")
    L.append("")
    L.append(f"{'谱':>4}{'夹角°':>7}{'沿脊线σ':>9}{'跨脊线σ':>9}{'脊线σ(log)':>11}"
             f"{'拉长比':>8}   corr(r, ·)")
    for r in rows:
        rg, c = r["ridge"], r["corr_with_r"]
        cs = "  ".join(f"{k}={c[k]:+.2f}" for k in ("n1", "Q1", "n2", "Q2") if k in c)
        L.append(f"{r['label']:>4}{_fmt(rg['offset_angle_deg'], 7, '.0f')}"
                 f"{_fmt(rg['along_in_sd'], 9, '+.2f')}{_fmt(rg['across_in_sd'], 9, '+.2f')}"
                 f"{rg['sd_along']:>11.2f}{rg['elongation']:>8.1f}   {cs}")
    p = summary["prior"]
    L.append("")
    L.append(f"先验隐含的 r（与数据无关）：中位 {p['median']:.3f}  "
             f"0.5/99.5% = [{p['p005']:.3f}, {p['p995']:.3f}]  "
             f"P_prior(r ≤ 真值) = {p['p_le_truth']:.2f}")
    L.append(f"      沿脊线先验对数密度：r={p['r_minus']:.3f} → {p['r_plus']:.3f} 变化 "
             f"{p['along_ridge_dlogprior']:+.2f} nats（正 = 先验偏好更大的 r）")
    L.append("")
    s = summary
    L.append(f"汇总（{s['basis']}，n={s['n_used']}，门禁通过 {s['n_gate_pass']}）：")
    L.append(f"      r 中位 {['%.4f' % m for m in s['median_r_used']]}  真值 {s['truth']:.4f}")
    L.append(f"      符号检验：{s['sign']['k_below']}/{s['sign']['n']} 低于真值，"
             f"双侧 p = {s['sign']['p_two_sided']:.3f}（单侧 {s['sign']['p_one_sided']:.3f}）")
    L.append(f"      偏离量（中位）：均值 {s['mean_bias_median']:+.4f}；"
             f"|偏离|/后验 sd 均值 {s['mean_abs_bias_in_sd']:.2f}、最大 {s['max_abs_bias_in_sd']:.2f}")
    return "\n".join(L)


# ---------------------------------------------------------------- 自检

def selftest() -> bool:
    import tempfile
    ok, cases = True, []

    def chk(name, cond, extra=""):
        nonlocal ok
        cases.append((name, bool(cond), extra))
        ok = ok and bool(cond)

    truth = dict(M.TRUTH)
    rt = truth["R1"] / (truth["R1"] + truth["R2"])

    # 1) 构造"只沿脊线偏移"的样本 → 夹角≈0、沿脊线分量≈设定值、跨≈0
    #    噪声也必须**沿/跨脊线**生成；若按坐标轴生成各向异性噪声，主轴会落到
    #    坐标轴上，测出来的夹角就不是这条构造想验的东西了。
    rng = np.random.default_rng(1)
    n = 20000
    u0 = np.array([math.log(truth["R1"]), math.log(truth["R2"])])
    v = np.array([1.0, -1.0]) / math.sqrt(2.0)
    vp = np.array([1.0, 1.0]) / math.sqrt(2.0)
    for shift, tol_ang in ((0.0, 10.0), (0.4, 3.0)):
        a = rng.normal(0, 0.5, n)                  # 沿脊线
        b = rng.normal(0, 0.02, n)                 # 跨脊线
        u = u0 + shift * v + np.outer(a, v) + np.outer(b, vp)
        rg = ridge_geometry(np.exp(u[:, 0]), np.exp(u[:, 1]), truth)
        # 主轴正向约定为 R1 增大（→ r 增大），所以 +shift 应给出正的沿脊线分量
        want = shift / rg["sd_along"]
        chk(f"1 沿脊线偏移 {shift:+.1f} → 夹角 {rg['offset_angle_deg']:.1f}°、"
            f"沿 {rg['along_in_sd']:+.2f}σ",
            (rg["offset_angle_deg"] is None or rg["offset_angle_deg"] < tol_ang)
            and abs(rg["across_in_sd"]) < 0.5
            and abs(rg["along_in_sd"] - want) < 0.5)
    # 2) 构造"只跨脊线偏移" → 夹角≈90°
    u = u0 + np.array([0.03, 0.03]) + rng.normal(0, 0.005, size=(n, 2))
    rg = ridge_geometry(np.exp(u[:, 0]), np.exp(u[:, 1]), truth)
    chk("2 跨脊线偏移 → 夹角≈90°", rg["offset_angle_deg"] > 80)
    # 3) 先验量：中位 r≈0.5，P(r≤真值)≈0.20
    p = prior_r_stats(3049.109465143577, rt)
    chk("3 先验 median r≈0.5", abs(p["median"] - 0.5) < 0.02, f"{p['median']:.3f}")
    chk("3b 先验 P(r≤真值)≈0.20", abs(p["p_le_truth"] - 0.20) < 0.02,
        f"{p['p_le_truth']:.3f}")
    chk("3c 先验沿脊线偏好更大 r", p["along_ridge_dlogprior"] > 0,
        f"{p['along_ridge_dlogprior']:+.2f}")
    # 4) 符号检验：4/5 双侧 p≈0.375、单侧 0.1875
    st = sign_test(4, 5)
    chk("4 符号检验 4/5", abs(st["p_two_sided"] - 0.375) < 1e-9
        and abs(st["p_one_sided"] - 0.1875) < 1e-9, str(st))
    chk("4b 符号检验 5/5", abs(sign_test(5, 5)["p_two_sided"] - 0.0625) < 1e-9)
    # 5) 真值正好居中（对数对称抽样）→ 分位≈0.5、bias≈0
    a = rng.normal(0, 0.3, n)
    b = rng.normal(0, 0.3, n)
    s = {"R1": np.exp(u0[0] + a), "R2": np.exp(u0[1] + b),
         **{k: np.zeros(n) for k in PARAM_KEYS}}
    d = diagnose(s, truth, rt, True)
    chk("5 真值居中 → 分位≈0.5 / bias≈0",
        abs(d["r"]["truth_pct"] - 0.5) < 0.02
        and abs(d["r"]["bias_median"]) < 0.02 * d["r"]["sd"],
        f"pct={d['r']['truth_pct']:.3f} bias={d['r']['bias_median']:+.4f}")
    # 6) 汇总只用门禁通过的运行
    mk = lambda med, gp: {"label": "x", "gate_pass": gp,
                          "r": {"median": med, "truth": rt, "bias_in_sd": (med - rt) / 0.03}}
    rows = [mk(0.03, True), mk(0.04, True), mk(0.05, True), mk(0.02, False)]
    sm = summarize(rows, p)
    chk("6 汇总只用 pass 的运行", sm["n_used"] == 3 and sm["n_gate_pass"] == 3)
    # 7) 样本数不足时降级为"全部仅作描述"
    rows = [mk(0.03, True), mk(0.04, False), mk(0.05, False)]
    chk("7 pass 不足 3 → 降级", "不足" in summarize(rows, p)["basis"])
    # 8) npz 往返：真实调用路径（读文件 → diagnose）
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "x.samples.npz"
        rng2 = np.random.default_rng(3)
        ss = {"R1": np.exp(rng2.normal(math.log(55), 0.3, 500)),
              "R2": np.exp(rng2.normal(math.log(600), 0.3, 500)),
              **{k: rng2.normal(1, 1, 500) for k in PARAM_KEYS}}
        np.savez_compressed(f, **ss)
        back = {k: np.load(f)[k] for k in np.load(f).files}
        d = diagnose(back, truth, rt, True)
        chk("8 npz 往返 → 诊断量有限", all(math.isfinite(x) for x in
                                          (d["r"]["median"], d["ridge"]["along_in_sd"])))
    for name, good, extra in cases:
        print(f"  [{'✓' if good else '✗'}] {name}{'' if good else '  ' + extra}")
    print(f"{'通过' if ok else '失败'}：{sum(1 for _, g, _ in cases if g)}/{len(cases)}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M3.5 r 偏低诊断（探索性，§20.20）")
    ap.add_argument("--json", help="m3_stage2 记录文件（含 truth 与门禁判定）")
    ap.add_argument("--samples-from", default=None,
                    help="样本文件命名模板，默认按 <json>.{stage}.d{data}.s{sampler}.samples.npz")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        print("§20.20 诊断链自检（合成输入）")
        return 0 if selftest() else 1
    if not args.json:
        ap.error("需要 --json（或 --selftest）")

    d = json.loads(Path(args.json).read_text(encoding="utf-8"))
    recs = d.get("records") or []
    truth = dict(d["config"]["truth"])
    rt = truth["R1"] / (truth["R1"] + truth["R2"])
    zmax = recs[0]["zmax"]
    prior = prior_r_stats(zmax, rt)

    # 与 m3_stage2 的 `--save-samples` 命名一致：它用 `with_suffix(...)` 替换
    # `.json`，所以这里也要先把后缀去掉，否则会拼出 `x.json.r.d0...`
    base = str(Path(args.json).with_suffix(""))
    tmpl = args.samples_from or (base + ".{stage}.d{data}.s{sampler}.samples.npz")
    rows, keys = [], []
    for rec in recs:
        p = Path(tmpl.format(stage=rec["stage"], data=rec["data_seed"],
                             sampler=rec["sampler_seed"]))
        if not p.exists():
            print(f"!! 缺样本文件 {p}，跳过 data_seed={rec['data_seed']}")
            continue
        z = np.load(p)
        s = {k: z[k] for k in z.files}
        row = diagnose(s, truth, rt, bool(rec["run_valid"]["pass"]))
        row["label"] = f"d{rec['data_seed']}"
        row["samples"] = str(p)
        rows.append(row)
        keys.append((rec["data_seed"], rec["sampler_seed"]))
    if not rows:
        print("没有可用样本，检查 --samples-from")
        return 1

    summary = summarize(rows, prior)
    print(report(rows, summary, {"n": len(rows), "keys": keys}))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"time": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
             "source": args.json, "exploratory_only": True,
             "note": "不产生 M3.5 判定；门禁未过的运行只作描述",
             "truth_r": rt, "rows": rows, "summary": summary},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"诊断写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
