"""bmdrt compare —— 一条谱、多个族的候选、一张图、一个状态。

与 `bmdrt auto` 的分工（这是本次改动要回答的三个问题）：

1. **拓扑要不要手动选**：`auto` 只覆盖 voigt 一族（K 由 DRT 预判自动取，L 由机制
   判断自动开），nested 必须显式走 `nested_fit.py`。`compare` 把 voigt K=1..ks 与
   nested L∈layers 一起跑，再由 `circuit_evaluator` 出状态——族是自动并列的，
   **但结论仍不写进生产默认**（admission §9.4.21 A1–A4 未过）。
2. **默认档**：`standard`（2000/1000/2 链/ta=0.98）。`--budget quick` 只作预览，
   输出里会标"探索性"。
3. **出图统一**：走 `bm_figure_unified.plot_fit`，一条谱一张 2×3 图，所有候选叠在
   同一组坐标里。`bm_real.plot_real`（6 子图但只认 voigt 中间量）与
   `nested_fit.write_plot`（3 子图）保持原样不动——合并它们会改两条已有 CLI 的输出。

用法：
    python bmdrt.py compare 谱.csv
    python bmdrt.py compare 谱.csv --ks 1,2 --layers 1,2 --budget quick
    python bmdrt.py compare 目录/ --limit-files 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bm_core as B                      # noqa: E402
import bm_top as T                       # noqa: E402
import bm_real as R                      # noqa: E402
import admission as AD                   # noqa: E402  临时准入状态（恒 provisional）
import circuit_runner as RU              # noqa: E402
import circuit_evaluator as EV           # noqa: E402
import bm_figure_unified as FIG          # noqa: E402

OUT_ROOT = os.path.join("out", "compare")


def need_L(f, Z) -> bool:
    """高频端是不是感抗（Im>0）⇒ 不串联 L 就表达不了。

    **判据必须看高频端，不能看"有没有 Im>0 的点"**：容性谱在最低频 Im 也会
    变正（半圆收尾），那样几乎每条谱都会被判成需要 L。这里取最高频 10% 的点，
    过半为正才算。

    这是 `auto_mechanism` 的**简化替身**，不是同一套判据：`compare` 只用它决定
    要不要生成 include_L=True 的 voigt 候选，别处不引用。
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    hi = Z.imag[f >= f.max() / 10.0]
    return bool(hi.size and np.mean(hi > 0) >= 0.5)


def params_from_json(topo: str, p: dict) -> dict:
    """`fits/*.json` 里的参数读回来：voigt 的 pi/tau/phi/zeta 是 list，要转数组。"""
    if topo == "voigt":
        return {k: (np.asarray(v, float) if isinstance(v, (list, tuple))
                    else float(v)) for k, v in p.items()}
    return {k: float(v) for k, v in p.items()}


def forward(topo: str, cfg: dict, params: dict, f) -> np.ndarray:
    if topo == "voigt":
        return B.model_z(f, params, include_L=bool(cfg.get("include_L", False)))
    if topo == "nested":
        p = dict(params)
        if str(cfg.get("branch")) == "series" and "L_ad" not in p:
            # series 支路只采 R_L/tau_L，`L_ad` 是派生量；漏了正演处会 KeyError
            p["L_ad"] = float(p["R_L"]) * float(p["tau_L"])
        return T.z_nested(f, p, branch=str(cfg["branch"]),
                          layers=int(cfg["layers"]))
    raise ValueError(f"拓扑 {topo!r} 没有正演适配器")


def label_of(topo: str, cfg: dict) -> str:
    if topo == "voigt":
        return f"voigt K={cfg.get('K')}" + ("+L" if cfg.get("include_L") else "")
    return f"nested L={cfg.get('layers')}/{cfg.get('branch')}"


def one(fp: str, a, ks, layers, branches) -> dict:
    f, Z = R.load_spectrum(fp)
    stem = os.path.splitext(os.path.basename(fp))[0]
    out = os.path.join(a.out, stem)
    include_L = need_L(f, Z) if a.include_L is None else bool(a.include_L)

    meta = RU.run(fp, seed=a.seed, budget_profile=a.budget, out_dir=out,
                  ks=ks, layers=layers, include_L=include_L,
                  branches=branches, likelihood=a.likelihood)
    ev = EV.evaluate_run(out)

    curves = []
    for row in meta["rows"]:
        if row.get("run_status") not in ("completed", "cached"):
            continue
        cid = row["candidate_id"]
        jp = os.path.join(out, "fits", cid + ".json")
        if not os.path.exists(jp):
            continue
        with open(jp, encoding="utf-8") as fh:
            d = json.load(fh)
        cfg = d.get("config") or {}
        topo = d["topology_id"]
        try:
            Zm = forward(topo, cfg, params_from_json(topo, d["params"]), f)
        except Exception:                       # noqa: BLE001
            continue
        m = RU.metrics(f, Z, Zm)
        curves.append(dict(label=label_of(topo, cfg), Zc=Zm,
                           rms=m["rms_rel"], acf=m["acf"], cid=cid,
                           rhat=d.get("rhat_max"), div=d.get("divergences")))

    curves.sort(key=lambda c: c["rms"])
    for i, c in enumerate(curves):
        c["color"] = FIG.COLORS[i % len(FIG.COLORS)]

    head = (f"{stem}  n={f.size}  {f.min():.3g}–{f.max():.3g} Hz  "
            f"{a.budget}  σ={a.likelihood}  seed={a.seed}"
            + ("  +L(auto)" if include_L else ""))
    lines = [f"{c['label']:<22} rms={c['rms']:5.2f}%  acf={c['acf']:.3f}"
             + (f"  R̂={c['rhat']:.3f}  div={c['div']}" if c["rhat"] else "")
             for c in curves]
    # 每行 2 个候选：一行放 5 个会被 `bbox_inches="tight"` 裁掉
    chunks = [lines[i:i + 2] for i in range(0, len(lines), 2)]
    meta_txt = head + "\n" + "\n".join("   ".join(c) for c in chunks) \
        if lines else head + "\n（无可行候选）"

    png = os.path.join(out, "compare.png")
    FIG.plot_fit(png, f, Z, curves,
                 title=f"{stem} —— {len(curves)} 个候选并列",
                 meta=meta_txt)

    rec = dict(file=fp, data_sha256=meta["data_sha256"], n_points=int(f.size),
               budget_profile=a.budget, likelihood=a.likelihood,
               seed=a.seed, include_L=include_L,
               status=ev["status"], reason=ev["reason"],
               # 临时准入：`compare` 用的族（含 nested）没过 §13/§14，所以每份
               # 产物都要能自证"这是能跑、不是达标"。见 `admission.py`。
               admission=AD.block(),
               confirmation=ev.get("confirmation"),
               candidates=[dict(cid=c["cid"], label=c["label"],
                                rms_rel=c["rms"], acf=c["acf"],
                                rhat_max=c["rhat"], divergences=c["div"])
                           for c in curves],
               png=png, run_dir=out)
    with open(os.path.join(out, "compare.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, allow_nan=False)

    print(f"\n[{stem}] 状态 {ev['status']}（{ev['reason']}）")
    for ln in lines:
        print("  " + ln)
    print(f"  图：{png}")
    print(f"  评价：{os.path.join(out, 'evaluation.md')}")
    return rec


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="bmdrt compare",
        description="多族候选自动并列：voigt K 扫描 + nested L 扫描 → 统一状态 + 统一图")
    ap.add_argument("inputs", nargs="+", help="谱文件/目录/glob")
    ap.add_argument("--ks", default="1,2,3", help="voigt 的 K 列表，默认 1,2,3")
    ap.add_argument("--layers", default="1,2", help="nested 层数，默认 1,2")
    ap.add_argument("--branches", default="parallel",
                    help="nested 支路，默认 parallel；research 用 series/capacitive")
    ap.add_argument("--budget", default="standard",
                    choices=sorted(RU.BUDGETS), help="采样档，默认 standard")
    ap.add_argument("--likelihood", default="perpoint",
                    choices=("perpoint", "free"),
                    help="噪声口径：perpoint=固定 2%%·|Z|（默认）；"
                         "free=σ 被采样（与 bmdrt auto 同口径）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", dest="include_L", action="store_true", default=None,
                    help="强制开串联 L")
    ap.add_argument("--no-L", dest="include_L", action="store_false",
                    help="强制关串联 L")
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--limit-files", type=int, default=None)
    a = ap.parse_args(argv)

    ks = tuple(int(x) for x in a.ks.split(",") if x.strip())
    layers = tuple(int(x) for x in a.layers.split(",") if x.strip())
    branches = tuple(x.strip() for x in a.branches.split(",") if x.strip())

    files = []
    for pat in a.inputs:
        files.extend(RU.expand_inputs(pat, a.limit_files))
    if not files:
        print("没有可拟合的文件", file=sys.stderr)
        return 2
    if a.limit_files:
        files = files[:a.limit_files]

    print(f"共 {len(files)} 条谱；预算 {a.budget}；σ={a.likelihood}；"
          f"voigt K∈{ks}、nested L∈{layers}/{','.join(branches)}")
    print(AD.banner())
    n_bad = 0
    for i, fp in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {os.path.basename(fp)}")
        try:
            one(fp, a, ks, layers, branches)
        except Exception as exc:                 # noqa: BLE001
            n_bad += 1
            print(f"  失败：{type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"\n完成 {len(files)} 条；失败 {n_bad} 条。产物根：{a.out}")
    return 0 if n_bad == 0 else 1


def selftest() -> int:
    ok, total, bad = 0, 0, []

    def c(name, cond):
        nonlocal ok, total
        total += 1
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")
        if cond:
            ok += 1
        else:
            bad.append(name)

    f = np.logspace(-2, 5, 20)
    Zc = 50 + 200 / (1 + 1j * 2 * np.pi * f * 1e-2)
    Zin = np.concatenate([Zc[:10], Zc[10:] + 10j])   # 高频段带感抗
    c("need_L：纯容性 → False", need_L(f, Zc) is False)
    c("need_L：含 Im>0 → True", need_L(f, Zin) is True)

    p = params_from_json("voigt", dict(R_inf=1.0, W=2.0, pi=[0.5],
                                       tau=[0.1], phi=[0.5], zeta=[1.0]))
    c("voigt 参数：list 转 ndarray", isinstance(p["pi"], np.ndarray)
      and isinstance(p["R_inf"], float))
    p2 = params_from_json("nested", dict(Rs=1.0, R1=2.0))
    c("nested 参数：全转 float", all(isinstance(v, float) for v in p2.values()))

    Zm = forward("voigt", dict(K=1, include_L=False), p, f)
    c("voigt 正演形状与频率一致", Zm.shape == f.shape and np.isfinite(Zm).all())
    # 容性谱：最低频 Im 变正、最高频为负 —— 旧判据会误判成"需要 L"
    Ztail = Zc.copy()
    Ztail[0] = Ztail[0].real + 5j
    c("低频 Im>0 但高频为负 ⇒ 不需要 L（旧写法会误判）", need_L(f, Ztail) is False)
    c("未注册拓扑 → 抛错", _raises(lambda: forward("drt", {}, {}, f)))

    c("label：voigt 带 K 与 +L",
      label_of("voigt", dict(K=2, include_L=True)) == "voigt K=2+L")
    c("label：nested 带层数与支路",
      label_of("nested", dict(layers=2, branch="parallel")) == "nested L=2/parallel")

    print(f"\n自检 {ok}/{total} 通过" if not bad else f"\nFAIL {len(bad)}/{total}: {bad}")
    return 0 if not bad else 1


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:                            # noqa: BLE001
        return True
    return False


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(run())
