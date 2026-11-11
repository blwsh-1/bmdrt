# -*- coding: utf-8 -*-
"""§19 执行器：全频段系统性偏差扫描（复现真实谱 acf 需要多宽/多大的偏差）。

口径全部来自 `M6_VALIDATION_PLAN.md` §19 与 §18；**复用
`m4_synth_probe.fit_one`（项目纪律：证据脚本只能有一处拟合入口）**，
只通过新增的 `Z_override` 传入构造好的观测谱。

**边界**：只读探针，`exploratory`，不改生产代码/族定义/阈值/判定，
不接 `bmdrt.py`；注入是人为构造，不许反推真实谱成因。

用法（项目 venv）：
    python evidence/nested_topology/m4_drift_scan.py --verify   # 零回归验证
    python evidence/nested_topology/m4_drift_scan.py --dry
    python evidence/nested_topology/m4_drift_scan.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bm_top_gen as G              # noqa: E402
import m4_synth_probe as P          # noqa: E402

REF_JSON = os.path.join("out", "m4_synth_probe.json")
OUT_JSON = os.path.join("out", "m4_drift_scan.json")
AMPS = (0.05, 0.15, 0.30)
SHAPES = ("S1", "S2", "S3")
DRIFTS = (0.10, 0.30)
TRUTH_TAG = "T2c"                  # AZ31-4Li_0，真实谱 acf 0.819
SEED = 0
NOISE = 0.02
ACF_BAR = 0.50                     # §19.3 主终点；即 §14 M-b 的上限


def verify() -> dict:
    """§19 硬边界 2：`Z_override` 改动后默认路径必须逐位不变。"""
    st = P.selftest()
    cases = P.build_cases()
    c = next(x for x in cases if x["group"] == "G" and x["truth"] == "T1"
             and x["seed"] == 0)
    t0 = time.time()
    r = P.fit_one(c)                      # 默认路径（Z_override=None）
    ref_rows = json.load(open(REF_JSON, encoding="utf-8"))["rows"]
    ref = next(x for x in ref_rows if x["group"] == "G" and x["truth"] == "T1"
               and x["seed"] == 0 and "acf" in x)
    keys = ("acf", "rms_rel", "rhat_max", "voigt_acf", "drop", "hf_band")
    diff = {k: abs(float(r[k]) - float(ref[k])) for k in keys}
    ok = bool(st["ok"] and max(diff.values()) < 1e-9)
    print(f"[verify] 默认路径重跑 G/T1/s0：{time.time() - t0:.1f}s")
    for k, v in diff.items():
        print(f"  {k:<10} 本次 {r[k]:.9f}  §18 {ref[k]:.9f}  |Δ| {v:.3e}")
    print("[verify] " + ("通过（逐位一致）" if ok else "失败：默认路径已变，整节作废"))
    return dict(ok=ok, selftest=st, diff=diff,
                this={k: r[k] for k in keys}, ref={k: ref[k] for k in keys})


def _shape_g(shape: str, t: np.ndarray) -> np.ndarray:
    if shape == "S1":
        return t
    if shape == "S2":
        return (2 * t - 1) ** 2
    if shape == "S3":
        return np.sin(2 * np.pi * t)
    raise ValueError(shape)


def build_cases() -> list[dict]:
    """11 条注入谱；全部复用同一真值、同一噪声实现（seed 0）。"""
    p = P.truth_sets()[TRUTH_TAG]
    f = G.real_grid("AZ31")
    z_true = G.gen(f, p, branch=P.BRANCH, layers=P.LAYERS)
    t = np.log10(f / f.min()) / np.log10(f.max() / f.min())
    z_noisy = G.add_noise(z_true, NOISE, mode="relative", seed=SEED)
    noise_inc = z_noisy - z_true            # 固定噪声增量，保证组间可比

    cases = []
    for shape in SHAPES:
        for a in AMPS:
            Z = z_noisy + 1j * a * np.abs(z_true) * _shape_g(shape, t)
            cases.append(dict(group="D", truth=f"{shape}/a{a:g}", seed=SEED,
                              frac=NOISE, inject=True, f=f, z_true=z_true, p=p,
                              Z=Z, shape=shape, amp=a))
    for d in DRIFTS:
        pp = dict(p)
        pp["R_ad"] = p["R_ad"] * (1 + d)
        # 整条 f 网格分别生成，保证 CPE 的 w_ref 基准一致（§19.1）
        z_b = G.gen(f, pp, branch=P.BRANCH, layers=P.LAYERS)
        z_clean = z_true.copy()
        z_clean[t < 0.5] = z_b[t < 0.5]    # 低频半＝后测，用漂移后的参数
        Z = z_clean + noise_inc
        cases.append(dict(group="D", truth=f"S4/d{d:g}", seed=SEED,
                          frac=NOISE, inject=True, f=f, z_true=z_true, p=p,
                          Z=Z, shape="S4", amp=d))
    return cases


def summarize(rows: list[dict]) -> dict:
    matrix = {}
    for r in rows:
        matrix.setdefault(r["shape"], {})[f"{r['amp']:g}"] = dict(
            acf=r["acf"], rms=r["rms_rel"], Ma=r["Ma"], Mb=r["Mb"],
            dropped=len(r["dropped_hz"]))
    over = [r for r in rows if r["acf"] >= ACF_BAR]
    ma_not_mb = [r for r in rows if r["Ma"] and not r["Mb"]]
    min_amp = {}
    for shape in (*SHAPES, "S4"):
        hit = [r for r in over if r["shape"] == shape]
        min_amp[shape] = min((r["amp"] for r in hit), default=None)
    if over:
        verdict = ("有形态能复现 acf≥0.50；最小幅度见 min_amp"
                   if any(v is not None for v in min_amp.values())
                   else "无过线形态")
    else:
        verdict = ("11 条全 <0.50：这些简单确定性偏差都不够，真实谱残差更复杂，"
                   "结论收敛到「必须补同状态重复谱」")
    return dict(matrix=matrix, n_over=int(len(over)),
                over=[dict(shape=r["shape"], amp=r["amp"], acf=r["acf"],
                           rms=r["rms_rel"]) for r in over],
                min_amp=min_amp,
                ma_pass_mb_fail=int(len(ma_not_mb)),
                ma_pass_mb_fail_cases=[r["truth"] for r in ma_not_mb],
                acf_median=float(np.median([r["acf"] for r in rows])),
                rms_median=float(np.median([r["rms_rel"] for r in rows])),
                verdict=verdict)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="§19 全频段偏差扫描（exploratory）")
    ap.add_argument("--json", default=OUT_JSON)
    ap.add_argument("--verify", action="store_true", help="只跑零回归验证")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--shapes", nargs="+", default=[*SHAPES, "S4"])
    a = ap.parse_args(argv)

    v = verify()
    if a.verify:
        return 0 if v["ok"] else 1
    if not v["ok"]:
        print("§19 硬边界 2：默认路径已变，整节作废", file=sys.stderr)
        return 1

    cases = [c for c in build_cases() if c["shape"] in a.shapes]
    print(f"\n§19：真值 {TRUTH_TAG}、seed {SEED}、{NOISE:.0%} 逐点噪声、"
          f"§18 同口径；基线 G/T2c/s0 acf=0.363（§18，不重跑）")
    print(f"共 {len(cases)} 条注入谱", flush=True)
    if a.dry:
        for c in cases:
            print(f"  {c['truth']:<12} shape={c['shape']} amp={c['amp']:g}")
        return 0

    rows, t_all = [], time.time()
    for i, c in enumerate(cases, 1):
        try:
            r = P.fit_one(c, Z_override=c["Z"])
            r.update(shape=c["shape"], amp=c["amp"], tag=c["truth"])
            rows.append(r)
            print(f"[{i:2d}/{len(cases)}] {c['truth']:<12} rms {r['rms_rel']:6.2f}%  "
                  f"acf {r['acf']:.3f}  voigt {r['voigt_acf']:.3f}  "
                  f"drop {r['drop']:+.3f}  R̂ {r['rhat_max']:.3f} div {r['divergences']}  "
                  f"M-a{'✓' if r['Ma'] else '✗'} M-b{'✓' if r['Mb'] else '✗'}  "
                  f"剔点{len(r['dropped_hz'])}  [{r['seconds']}s]", flush=True)
        except Exception as exc:
            print(f"[{i:2d}/{len(cases)}] {c['truth']} 失败："
                  f"{type(exc).__name__}: {exc}", flush=True)
            rows.append(dict(tag=c["truth"], shape=c["shape"], amp=c["amp"],
                             error=f"{type(exc).__name__}: {exc}"))

    good = [r for r in rows if "acf" in r]
    s = summarize(good)
    print("\n=== §19 汇总 ===")
    for shape, d in s["matrix"].items():
        print(f"  {shape}: " + "  ".join(
            f"a={k} acf {v['acf']:.3f}/rms {v['rms']:.1f}%" for k, v in d.items()))
    print(f"  acf≥0.50：{s['n_over']}/{len(good)}  最小过线幅度 {s['min_amp']}")
    print(f"  「M-a 过但 M-b 不过」{s['ma_pass_mb_fail']} 条"
          f"（§14 真实谱指纹）")
    print(f"  判读：{s['verdict']}")
    print(f"总耗时 {time.time() - t_all:.1f}s")

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(prereg="M6_VALIDATION_PLAN.md §19",
                           status="exploratory", verify=v,
                           baseline=dict(tag="G/T2c/s0", acf=0.363,
                                         source="out/m4_synth_probe.json"),
                           constants=dict(truth=TRUTH_TAG, seed=SEED, noise=NOISE,
                                          amps=list(AMPS), drifts=list(DRIFTS),
                                          acf_bar=ACF_BAR),
                           rows=rows, summary=s), fh, ensure_ascii=False, indent=1)
        print(f"已写 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
