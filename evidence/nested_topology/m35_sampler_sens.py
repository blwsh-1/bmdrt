# -*- coding: utf-8 -*-
"""M3.5 采样器随机性敏感性基线的判定链（`BM_TOP_M3_PLAN.md` §20.18）。

**它回答的问题不是"`r` 为什么偏低"**，而是：在同一条合成谱上只换采样器 seed，
收敛门禁会不会跟着翻面。所以输入必须是一份 `data_seed` 全相同、`sampler_seed`
各不相同的 `m3_stage2.py` 记录文件：

    python evidence/nested_topology/m3_stage2.py --stage r \\
        --seeds 0,1,2,3,4 --data-seeds 4,4,4,4,4 \\
        --chains 4 --samples 1200 --warmup 5000 \\
        --target-accept 0.95 --max-tree-depth 12 \\
        --save-samples --json evidence/nested_topology/m35_sens_d4.json

    python evidence/nested_topology/m35_sampler_sens.py \\
        --json evidence/nested_topology/m35_sens_d4.json \\
        --ref evidence/nested_topology/m35_r_w5000.json

判定规则在跑之前就写定（§20.18），这里只做实现，不做事后挑选：

    k = 通过 `run_validity` 的次数（判据本身从 `m3_stage2` 导入，不另立一套）
    k = n           → all_pass            本谱上重跑都过；不外推、不改判 M3.5
    1 ≤ k ≤ n-1     → sampler_sensitive   门禁对采样随机性敏感
    k = 0 且交集非空 → systematic          同一组参数每次都超阈值
    k = 0 且交集为空 → systematic_varies   每次都失败但失败位置不同

**这不是 §20.4 的主终点判定**。那里的 5 个 seed 必须是**独立噪声实现**；这里 5 次
共用同一份谱，pooled 后验不对应任何单一后验，因此不重算 `sd_r` 主终点、不产生
`full_success`/`partial` 之类的层级。本脚本只输出诊断与"读数是否稳定"。

`--ref` 的可比性自检（强制项，§20.18）：本实验的 `data_seed=4 / sampler_seed=0`
就是 §20.17 的第 5 条拟合，同 seed 同配置应逐位复现。不复现说明环境变了，
与 §20.17 的逐项比较要降级——这一条不能静默通过。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import m3_stage2 as M                      # noqa: E402  判据与阈值的唯一来源

# 这两个阈值只用于**定位**哪些参数把门禁顶掉了（逐参数明细），门禁本身仍读
# `rec["run_valid"]["pass"]`。抄一份是为了逐参数归因，`_consistency_check`
# 会验证它们与 `m3_stage2.acceptance` 的实际判定一致；不一致直接报错，
# 不静默漂移。
RHAT_MAX = 1.01
ESS_MIN = 200
MEDIAN_SE_FACTOR = 1.2533                  # 正态下 median 的标准误系数

# 参与"失败位置交集"的检查项。R̂ / ESS 两项**不进这里**：它们的失败位置要归到
# 具体参数（`bad_rhat` / `bad_ess`），否则每次都是同一个检查名，交集恒非空，
# "5 次是否在同一组参数上失败"这个问题就问不出来了。
_UNPARAMETRISED_CHECKS = ("div_lt_0.1pct", "pointwise_len_ok")


def per_run(rec: dict) -> dict:
    """把一条记录压成一行诊断。不改动原记录，不重算任何判据。"""
    d = rec.get("diagnostics") or {}
    sp = rec.get("split_r") or {}
    rh = d.get("rhat_by_param") or {}
    es = d.get("ess_by_param") or {}
    bad_r = sorted((k for k, v in rh.items() if v is not None and v >= RHAT_MAX),
                   key=lambda k: -rh[k])
    bad_e = sorted((k for k, v in es.items() if v is not None and v <= ESS_MIN),
                   key=lambda k: es[k])
    sd_r, ess_min = sp.get("sd"), d.get("ess_min")
    se_med = (MEDIAN_SE_FACTOR * sd_r / math.sqrt(ess_min)
              if sd_r and ess_min and ess_min > 0 else None)
    rv = rec.get("run_valid") or {}
    row = {
        "data_seed": rec.get("data_seed"),
        "sampler_seed": rec.get("sampler_seed"),
        "pass": bool(rv.get("pass")),
        "failing_checks": [k for k, v in (rv.get("checks") or {}).items() if not v],
        "rhat_max": d.get("rhat_max"),
        "rhat_worst_param": (bad_r[0] if bad_r else
                             (max(rh, key=lambda k: rh[k]) if rh else None)),
        "ess_min": ess_min,
        "ess_worst_param": bad_e[0] if bad_e else None,
        "divergences": d.get("divergences"),
        "r_median": sp.get("median"),
        "sd_r": sd_r,
        "width90": sp.get("width90"),
        "covered": sp.get("covered"),
        "bias": sp.get("bias"),
        "se_median_r": se_med,
        "bad_rhat": bad_r,
        "bad_ess": bad_e,
    }
    row["failure_signature"] = sorted(_signature(row))
    return row


def _signature(row: dict) -> set:
    """一次运行"失败落在哪里"的签名。

    R̂ / ESS 归到**参数名**（这才是"同一组参数每次都失守"能回答的粒度）；
    divergence / 长度这类没有参数归属的检查项按检查名记。
    """
    return (set(row["bad_rhat"]) | set(row["bad_ess"])
            | (set(row["failing_checks"]) & set(_UNPARAMETRISED_CHECKS)))


def verdict(rows: list[dict]) -> dict:
    """§20.18 预注册的四分支判定。纯函数，只看传入的行。"""
    n = len(rows)
    if n == 0:
        return {"tier": "no_data", "k": 0, "n": 0,
                "verdict": "无记录，不判定"}
    k = sum(1 for r in rows if r["pass"])
    sigs = [_signature(r) for r in rows if not r["pass"]]
    # 交集只在**失败的那些次**之间取。通过的运行签名是空集，把它并进来会让交集
    # 恒为空，于是"几次失败是否落在同一组参数上"这个预注册问题就永远问不出来。
    common = set.intersection(*sigs) if sigs else set()
    if k == n:
        tier = "all_pass"
    elif k == 0:
        tier = "systematic" if common else "systematic_varies"
    else:
        tier = "sampler_sensitive"
    meaning = {
        "all_pass": "本谱上重跑全部过门禁 → §20.17 的失败*可能*是随机；不外推到其他谱，"
                    "也不改判 M3.5（门禁是对 5 个独立噪声实现的要求）",
        "sampler_sensitive": "门禁随采样器 seed 翻面 → 该谱的门禁失败不能当作后验几何证据",
        "systematic": "每次都失败且失败位置相同 → 指向后验几何/参数化/先验，而非 RNG 流",
        "systematic_varies": "每次都失败但失败位置不同 → 高失败率，不能归为固定几何，"
                             "也不等于随机（n 太小）",
    }[tier]
    return {"tier": tier, "k": k, "n": n, "verdict": meaning,
            "common_among_failing": sorted(common),
            "n_failing": len(sigs),
            "signatures_of_failing": [sorted(s) for s in sigs]}


def readings(rows: list[dict]) -> dict:
    """读数稳定性（描述性）：逐 seed 后验中位的散布 vs MC 尺度。"""
    meds = [r["r_median"] for r in rows if r["r_median"] is not None]
    ses = [r["se_median_r"] for r in rows if r["se_median_r"] is not None]
    sds = [r["sd_r"] for r in rows if r["sd_r"] is not None]
    if len(meds) < 2 or not ses:
        return {"flag": "insufficient", "n": len(meds)}
    span = max(meds) - min(meds)
    lim = 2.0 * max(ses)
    return {"flag": ("readings_stable" if span <= lim else "readings_unstable"),
            "n": len(meds), "span_median_r": span, "limit": lim,
            "ratio": span / lim if lim > 0 else None,
            "median_r_min": min(meds), "median_r_max": max(meds),
            "sd_r_min": (min(sds) if sds else None),
            "sd_r_max": (max(sds) if sds else None),
            "note": "ESSmin 是**边际**最小 ESS，不构成派生量 r（R1/R2 的非线性组合）"
                    "的 ESS 下界；这里的 2·max(se) 只是描述性参考尺度，"
                    "不是统计稳定性检验"}


def determinism(recs_new: list[dict], recs_ref: list[dict] | None,
                tol: float = 1e-9) -> dict | None:
    """同 (data_seed, sampler_seed) 的记录必须逐位复现（§20.18 强制项）。"""
    if not recs_ref:
        return None
    idx = {(r.get("data_seed"), r.get("sampler_seed")): r for r in recs_ref}
    pairs, worst, bad = [], 0.0, []
    for r in recs_new:
        key = (r.get("data_seed"), r.get("sampler_seed"))
        ref = idx.get(key)
        if ref is None:
            continue
        a, b = per_run(r), per_run(ref)
        fields = ("rhat_max", "ess_min", "r_median", "sd_r", "width90")
        diffs = {}
        for f in fields:
            x, y = a[f], b[f]
            if x is None or y is None:
                diffs[f] = None if (x is None and y is None) else float("inf")
                continue
            rel = abs(x - y) / max(abs(y), 1e-30) if y != 0 else abs(x - y)
            diffs[f] = rel
            worst = max(worst, rel)
        ok = all(d is not None and d <= tol for d in diffs.values())
        if not ok:
            bad.append({"data_seed": key[0], "sampler_seed": key[1], "diffs": diffs})
        pairs.append({"data_seed": key[0], "sampler_seed": key[1],
                      "ok": ok, "diffs": diffs})
    return {"n_matched": len(pairs), "tol": tol, "max_rel_diff": worst,
            "ok": bool(pairs) and not bad, "mismatches": bad, "pairs": pairs}


def _consistency_check(rows: list[dict], recs: list[dict]) -> list[str]:
    """逐参数阈值的抄件必须与 `m3_stage2.acceptance` 的实际判定同向。"""
    probs = []
    for row, rec in zip(rows, recs):
        chk = (rec.get("run_valid") or {}).get("checks") or {}
        if "rhat_lt_1.01" in chk:
            mine = (row["rhat_max"] is not None and row["rhat_max"] < RHAT_MAX)
            if mine != bool(chk["rhat_lt_1.01"]):
                probs.append(f"data_seed={row['data_seed']} sampler_seed="
                             f"{row['sampler_seed']}: R̂ 阈值与 acceptance 不一致")
        if "ess_gt_200" in chk:
            mine = (row["ess_min"] is not None and row["ess_min"] > ESS_MIN)
            if mine != bool(chk["ess_gt_200"]):
                probs.append(f"data_seed={row['data_seed']} sampler_seed="
                             f"{row['sampler_seed']}: ESS 阈值与 acceptance 不一致")
    return probs


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(x, w=8, s=".4f"):
    if x is None:
        return "—".rjust(w)
    return format(x, f">{w}{s}")


def report(rows: list[dict], vd: dict, rd: dict, det: dict | None,
           cfg: dict | None = None) -> str:
    lines = []
    if cfg:
        lines.append(f"谱：data_seed={cfg.get('data_seeds')}  "
                     f"采样器 seed={cfg.get('sampler_seeds')}")
        lines.append(f"配置：{cfg.get('n_chains')} 链 × {cfg.get('n_samples')} 样本 / "
                     f"warmup {cfg.get('n_warmup')} / ta {cfg.get('target_accept')} / "
                     f"depth {cfg.get('max_tree_depth')}  初值={cfg.get('init')}")
    lines.append("")
    lines.append(f"{'seed':>5}{'门禁':>6}{'R̂max':>10}{'最差':>7}"
                 f"{'ESSmin':>9}{'div':>5}{'r 中位':>10}{'sd_r':>9}{'宽90':>9}{'覆盖':>5}")
    for r in rows:
        lines.append(
            f"{r['sampler_seed']:>5}{'✓' if r['pass'] else '✗':>6}"
            f"{_fmt(r['rhat_max']):>10}{(r['rhat_worst_param'] or '—'):>7}"
            f"{_fmt(r['ess_min'], 9, '.0f')}{_fmt(r['divergences'], 5, 'd')}"
            f"{_fmt(r['r_median']):>10}{_fmt(r['sd_r'], 9)}{_fmt(r['width90'], 9)}"
            f"{('✓' if r['covered'] else '✗' if r['covered'] is False else '—'):>5}")
    for r in rows:
        if not r["pass"]:
            lines.append(f"      s{r['sampler_seed']} 未过："
                         f"{', '.join(r['failing_checks']) or '—'}；"
                         f"R̂≥{RHAT_MAX}: {r['bad_rhat'] or '无'}；"
                         f"ESS≤{ESS_MIN}: {r['bad_ess'] or '无'}")
    lines.append("")
    lines.append(f"门禁通过 {vd['k']}/{vd['n']}  → 判定 `{vd['tier']}`")
    lines.append(f"      {vd['verdict']}")
    lines.append(f"失败各次共同失守的位置（只看失败的 {vd['n_failing']} 次）："
                 f"{vd['common_among_failing'] or '空'}")
    if rd.get("flag") != "insufficient":
        lines.append(f"读数跨度（**描述性参考，不是统计检验**）：range(median r)="
                     f"{rd['span_median_r']:.4f} "
                     f"{'≤' if rd['flag'] == 'readings_stable' else '>'} "
                     f"2·max(se)={rd['limit']:.4f}（比值 {rd['ratio']:.2f}）"
                     f" → `{rd['flag']}`")
        lines.append(f"      median r ∈ [{rd['median_r_min']:.4f}, {rd['median_r_max']:.4f}]  "
                     f"sd_r ∈ [{rd['sd_r_min']:.4f}, {rd['sd_r_max']:.4f}]")

    else:
        lines.append(f"读数稳定性：{rd.get('flag')}（可比较的中位不足）")
    if det is not None:
        if det["n_matched"] == 0:
            lines.append("可比性自检：未找到 (data_seed, sampler_seed) 均相同的参照记录 → 跳过")
        else:
            lines.append(f"可比性自检：{det['n_matched']} 对记录与参照文件比对，"
                         f"最大相对差 {det['max_rel_diff']:.1e}（容差 {det['tol']:.0e}）"
                         f" → {'✓ 逐位复现' if det['ok'] else '✗ 未复现，与旧文件的逐项比较需降级'}")
            for m in det["mismatches"]:
                lines.append(f"      d{m['data_seed']}/s{m['sampler_seed']}: {m['diffs']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 自检

def _mk(seed, pass_, rhat=1.004, ess=500, rmed=0.05, sd=0.03, width=0.10,
        bad_rhat=(), bad_ess=(), failed=()):
    """构造一条形状与 m3_stage2 记录一致的最小 rec（只含判定用到的字段）。

    基线参数全部合规，`bad_rhat` / `bad_ess` 指定谁失守；`rhat_max` / `ess_min`
    由逐参数表导出，与真实记录同构（不是额外塞进去的独立数字）。
    """
    rh = {"n1": 1.0012, "Q1": 1.0009, "R1": 1.0004, "R2": 1.0003}
    es = {"n1": ess, "Q1": ess + 60, "R1": ess + 90, "R2": ess + 30}
    for k in bad_rhat:
        rh[k] = max(float(rhat), RHAT_MAX + 1e-3)
    for k in bad_ess:
        es[k] = 50
    rmax, emin = max(rh.values()), min(es.values())
    checks = {"rhat_lt_1.01": rmax < RHAT_MAX, "ess_gt_200": emin > ESS_MIN,
              "div_lt_0.1pct": True, "pointwise_len_ok": True}
    for c in failed:
        checks[c] = False
    return {"data_seed": 4, "sampler_seed": seed,
            "run_valid": {"pass": pass_, "checks": checks},
            "diagnostics": {"rhat_max": rmax, "ess_min": emin, "divergences": 0,
                            "rhat_by_param": rh, "ess_by_param": es},
            "split_r": {"median": rmed, "sd": sd, "width90": width,
                        "covered": True, "bias": rmed - 0.084}}


def selftest() -> bool:
    ok = True
    cases = []

    def chk(name, cond, extra=""):
        nonlocal ok
        cases.append((name, bool(cond), extra))
        ok = ok and bool(cond)

    # 1) 全部通过
    rows = [per_run(_mk(i, True)) for i in range(5)]
    chk("1 全过 → all_pass", verdict(rows)["tier"] == "all_pass")
    # 2) 3/5 → sampler_sensitive（对应 §20.17 的实际形态）
    rows = [per_run(_mk(i, i < 3, rhat=(1.02 if i >= 3 else 1.005),
                        bad_rhat=("n1",) if i >= 3 else ())) for i in range(5)]
    chk("2 3/5 → sampler_sensitive", verdict(rows)["tier"] == "sampler_sensitive")
    # 3) 0/5 且失败参数相同 → systematic
    rows = [per_run(_mk(i, False, rhat=1.02, bad_rhat=("n1", "Q1"))) for i in range(5)]
    v = verdict(rows)
    chk("3 0/5 同参数 → systematic",
        v["tier"] == "systematic" and set(v["common_among_failing"]) == {"n1", "Q1"})
    # 4) 0/5 且失败位置不同 → systematic_varies
    names = ("n1", "Q1", "R1", "n2", "Q2")
    rows = [per_run(_mk(i, False, rhat=1.02, bad_rhat=(names[i],))) for i in range(5)]
    chk("4 0/5 异参数 → systematic_varies",
        verdict(rows)["tier"] == "systematic_varies")
    # 5) 边界 k=1 / 6) 边界 k=n-1
    rows = [per_run(_mk(i, i == 0, rhat=(1.02 if i else 1.005),
                        bad_rhat=() if i == 0 else ("n1",))) for i in range(5)]
    chk("5 k=1 → sampler_sensitive", verdict(rows)["tier"] == "sampler_sensitive")
    rows = [per_run(_mk(i, i < 4, rhat=(1.02 if i == 4 else 1.005),
                        bad_rhat=("n1",) if i == 4 else ())) for i in range(5)]
    chk("6 k=n-1 → sampler_sensitive", verdict(rows)["tier"] == "sampler_sensitive")
    # 6b) 混合通过时，交集只在失败的那些次之间取（否则恒为空，问题问不出来）
    rows = [per_run(_mk(i, i < 3, rhat=(1.02 if i >= 3 else 1.005),
                        bad_rhat=("n1",) if i >= 3 else ())) for i in range(5)]
    v = verdict(rows)
    chk("6b 混合通过 + 失败位置相同 → 交集非空",
        v["tier"] == "sampler_sensitive" and v["common_among_failing"] == ["n1"]
        and v["n_failing"] == 2)
    # 7) 读数明显变化 → readings_unstable
    rows = [per_run(_mk(i, True, rmed=m, sd=0.03, ess=400)) for i, m in
            enumerate((0.03, 0.04, 0.06, 0.08, 0.12))]
    chk("7 中位跨度大 → readings_unstable",
        readings(rows)["flag"] == "readings_unstable")
    # 8) 读数几乎不动 → readings_stable
    rows = [per_run(_mk(i, True, rmed=0.08 + 1e-4 * i, sd=0.03, ess=400))
            for i in range(5)]
    chk("8 中位几乎不动 → readings_stable",
        readings(rows)["flag"] == "readings_stable")
    # 9) 确定性自检：同记录 → ok；扰动 → 不复现
    a = [_mk(0, True), _mk(1, True)]
    chk("9a 同记录 → 复现", determinism(a, a)["ok"] is True)
    b = [_mk(0, True, rmed=0.05001), _mk(1, True)]
    det = determinism(a, b)
    chk("9b 扰动后 → 不复现", det["ok"] is False and det["n_matched"] == 2)
    # 10) 抄来的逐参数阈值必须与 acceptance 同向（防止阈值漂移后静默失效）
    recs = [_mk(0, True, rhat=1.009, ess=210), _mk(1, False, rhat=1.02, ess=100)]
    probs = _consistency_check([per_run(r) for r in recs], recs)
    chk("10 逐参数阈值与 acceptance 同向", not probs, str(probs))
    # 11) 空输入不崩
    chk("11 空输入 → no_data", verdict([])["tier"] == "no_data")

    for name, good, extra in cases:
        print(f"  [{'✓' if good else '✗'}] {name}{'' if good else '  ' + extra}")
    print(f"{'通过' if ok else '失败'}：{sum(1 for _, g, _ in cases if g)}/{len(cases)}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M3.5 采样器随机性敏感性判定（§20.18）")
    ap.add_argument("--json", required=False, help="m3_stage2 记录文件（同一 data_seed、多个 sampler seed）")
    ap.add_argument("--ref", default=None, help="参照文件（§20.17），用于逐位复现自检")
    ap.add_argument("--json-out", default=None, help="把判定结果写成 JSON")
    ap.add_argument("--selftest", action="store_true", help="只跑纯函数自检，不读文件")
    args = ap.parse_args(argv)

    if args.selftest:
        print("§20.18 判定链自检（合成输入）")
        return 0 if selftest() else 1
    if not args.json:
        ap.error("需要 --json（或 --selftest）")

    d = load(args.json)
    recs = d.get("records") or []
    if not recs:
        ap.error(f"{args.json} 里没有 records")
    dseeds = sorted({r.get("data_seed") for r in recs})
    if len(dseeds) > 1:
        ap.error(f"这份记录混了 {len(dseeds)} 个 data_seed {dseeds}；"
                 f"§20.18 的设计要求全部共用同一份谱（--data-seeds 4,4,4,4,4）。"
                 f"多谱的 pool 不能回答采样器随机性")
    c = d.get("config") or {}
    if c.get("stages") and c["stages"] != ["r"]:
        ap.error(f"只适用于 --stage r（收到 {c['stages']}）")

    rows = [per_run(r) for r in recs]
    probs = _consistency_check(rows, recs)
    if probs:
        print("!! 阈值一致性检查失败，先修代码再看结果：")
        for p in probs:
            print("   " + p)
        return 2

    vd = verdict(rows)
    rd = readings(rows)
    det = determinism(recs, (load(args.ref).get("records") if args.ref else None))
    print("采样器随机性敏感性（§20.18；不是 §20.4 主终点判定）")
    print(report(rows, vd, rd, det, c))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"time": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
             "source": str(args.json), "ref": args.ref,
             "design": {"data_seeds": c.get("data_seeds"),
                        "sampler_seeds": c.get("sampler_seeds"),
                        "n_chains": c.get("n_chains"), "n_samples": c.get("n_samples"),
                        "n_warmup": c.get("n_warmup"),
                        "target_accept": c.get("target_accept"),
                        "max_tree_depth": c.get("max_tree_depth")},
             "runs": rows, "verdict": vd, "readings": rd, "determinism": det,
             "not_primary_endpoint": True,
             "note": "5 次共用同一份谱 → 不是 §20.4 的独立噪声实现，不产生 M3.5 层级"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"判定写入 {args.json_out}")
    return 0 if vd["tier"] != "no_data" else 1


if __name__ == "__main__":
    sys.exit(main())
