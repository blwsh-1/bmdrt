"""用 standard 预算复核采样诊断异常的谱（只读分析，不改生产）。

动机
----
21 条 quick/free 里，`rhat>1.05` 或 `div>0` 的有 14 条。按 §13/§14 的规矩，
这些谱的 rms/acf **本来就不可引用**。但没验证过：这是"quick 预算不够"，
还是"谱/模型本身不可辨识"。两者的处置完全不同：

  · quick 不够 → 21 条的数字要在 standard 下重估，结论可能变；
  · 结构性不可辨识 → 加预算无用，那几条应直接标 `non_evaluable`。

判据（预注册在本文件顶部，跑前不改）
------------------------------------
  A. standard 下 `rhat<=1.05` 且 `div==0` ⇒ `budget_limited`（纯预算受限）
  B. `rhat<=1.05` 但 `0<div<=5` ⇒ `budget_limited_minor_div`（预算受限，
     残余极轻发散；2000 draws 里 2 次 = 0.1%，不足以判结构性）
  C. `rhat>1.05` 或 `div>5` ⇒ `structurally_unidentifiable`
  D. 拟合抛错 ⇒ `failed`

只重跑每条谱的 **best 候选**（results.json 记录的那一个），不重选模型。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/std_recheck.py --limit 2
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/std_recheck.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import bm_real as BR  # noqa: E402
import circuit_runner as RU  # noqa: E402
import cmd_compare as C  # noqa: E402

SRC = os.path.join("out", "batch21", "results.json")
OUT = os.path.join("out", "std_recheck")
PROG = os.path.join(OUT, "progress.jsonl")
RHAT_MAX = 1.05


def load_prog() -> dict:
    d = {}
    if os.path.exists(PROG):
        with open(PROG, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    d[r["path"] + "|" + r["label"]] = r
    return d


def save_prog(r: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(PROG, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def pick(cands, row):
    """按 topology + config 精确匹配候选对象（不靠顺序）。"""
    cfg = dict(row["config"])
    for c in cands:
        cc = dict(c.config)
        if c.topology_id != row["topology"]:
            continue
        if all(cc.get(k) == v for k, v in cfg.items()):
            return c
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--new-only", type=int, default=0,
                    help="本次最多新跑几条（已完成的自动跳过）")
    ap.add_argument("--all", action="store_true",
                    help="21 条全跑（不只异常），用于把 clean 的也拉到同一预算")
    ap.add_argument("--budget", default="standard")
    a = ap.parse_args(argv)

    recs = json.load(open(SRC, encoding="utf-8"))
    if a.all:
        bad = [r for r in recs if (r.get("best") or {}).get("rhat") is not None]
    else:
        bad = []
        for r in recs:
            b = r.get("best") or {}
            rh, dv = b.get("rhat"), b.get("div")
            if rh is None:
                continue
            if rh > RHAT_MAX or (dv or 0) > 0:
                bad.append(r)
    # 严重度：div 权重高（发散比 R̂ 更硬），再按 rhat
    bad.sort(key=lambda r: (-(r["best"].get("div") or 0), -(r["best"].get("rhat") or 0)))
    if a.limit:
        bad = bad[:a.limit]

    os.makedirs(OUT, exist_ok=True)
    print("待复核 %d 条（判据：standard 下 R̂<=%.2f 且 div=0 ⇒ budget_limited）"
          % (len(bad), RHAT_MAX))
    print("%-30s %-6s %-14s %10s %6s %8s %7s %6s %7s %6s %7s" % (
        "spec", "class", "label", "rms%", "acf",
        "q_rhat", "q_div", "s_rhat", "s_div", "s_rms", "secs"))
    print("-" * 116)

    out = []
    prog = load_prog()
    n_new = 0
    for r in bad:
        path = r["path"]
        b = r["best"]
        key = path + "|" + b["label"]
        if key in prog:
            out.append(prog[key])
            continue
        if a.new_only and n_new >= a.new_only:
            continue
        n_new += 1
        row = next((x for x in r["cands"]
                    if x["label"] == b["label"] and x.get("rms") is not None
                    and abs(x["rms"] - b["rms"]) < 1e-9), None)
        if row is None:
            print("%-30s 匹配失败（找不到 best 对应的候选行）" % path[-30:])
            continue
        f, Z = BR.load_spectrum(path)
        z_hf = float(np.abs(Z[np.argmax(f)]))
        inc_l = bool(C.need_L(f, Z))
        # 与 batch_fit_21 同步：adr 已移出候选集（2026-11-22）。这里只重跑
        # results.json 记下的那一个 best，而 21 条里 adr 从未当选 best ⇒ 匹配不受影响。
        cands = RU.build_candidates(ks=(1, 2), layers=(1, 2), include_adr=False,
                                    include_L=inc_l, budget_profile=a.budget)
        cand = pick(cands, row)
        if cand is None:
            print("%-30s 候选重建失败" % path[-30:])
            continue
        zov = 5.0 * z_hf if row["mode"] == "fixed" else None
        t0 = time.time()
        try:
            o = RU.fit_candidate(cand, f, Z, seed=0,
                                 budget_profile=a.budget, likelihood="free",
                                 zmax_override=zov)
            s_rhat, s_div = float(o["rhat_max"]), int(o["divergences"] or 0)
            s_rms = float(o["metrics"]["rms_rel"])
            state = ("budget_limited" if (s_rhat <= RHAT_MAX and s_div == 0)
                     else "budget_limited_minor_div"
                     if (s_rhat <= RHAT_MAX and s_div <= 5)
                     else "structurally_unidentifiable")
        except Exception as e:                        # noqa: BLE001
            s_rhat = s_div = s_rms = None
            state = "failed:" + type(e).__name__
        dt = time.time() - t0
        print("%-30s %-6s %-14s %10.2f %6.3f %8.3f %6d %7s %6s %7s %6.1f  %s" % (
            path.split(os.sep)[-1][:29], r["class"][:6], b["label"], b["rms"],
            b["acf"], b["rhat"], b["div"],
            "None" if s_rhat is None else "%.3f" % s_rhat,
            "-" if s_div is None else str(s_div),
            "None" if s_rms is None else "%.2f" % s_rms, dt, state))
        item = dict(path=path, cls=r["class"], label=b["label"],
                    quick=b, standard=dict(rhat=s_rhat, div=s_div, rms=s_rms),
                    seconds=dt, verdict=state,
                    zmax_override=zov, include_L=inc_l)
        save_prog(item)
        out.append(item)

    with open(os.path.join(OUT, "std_recheck.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(budget=a.budget, rhat_max=RHAT_MAX, rows=out), fh,
                  ensure_ascii=False, indent=1, allow_nan=False)
    n_ok = sum(1 for x in out if x["verdict"] == "budget_limited")
    n_minor = sum(1 for x in out if x["verdict"] == "budget_limited_minor_div")
    n_str = sum(1 for x in out if x["verdict"] == "structurally_unidentifiable")
    print("-" * 116)
    print("budget_limited %d / 预算受限但残余轻发散 %d / 结构性不可辨识 %d / failed %d"
          % (n_ok, n_minor, n_str,
             sum(1 for x in out if x["verdict"].startswith("failed"))))
    print("->", os.path.join(OUT, "std_recheck.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
