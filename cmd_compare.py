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


#: 扩散声明的值 → 正演必须收到的参数键。`model_z` 靠"键在不在"识别扩散项，
#: 没有 diffusion 形参；`z_nested`/`z_adr` 靠显式 `diffusion=` 形参。
#: 两种机制不同，但**口径必须统一**：声明了扩散就一定要画出带扩散的曲线。
_DIFF_PARAM_KEYS = {"Ws": ("Aw",), "Wo": ("Rwo", "tauD"),
                    "Wshort": ("Rws", "tauD")}


def forward(topo: str, cfg: dict, params: dict, f) -> np.ndarray:
    """按候选 config 重算整条谱（作图与图内 rms/acf 的唯一取数口）。

    **三族必须走同一口径**（2026-11-20 修）：此前只有 voigt 正确——因为
    `bm_core.model_z` 靠参数键自动识别扩散项；`z_nested`/`z_adr` 是显式
    `diffusion=` 形参，而这里没传，于是 nested/adr 的扩散候选被画成
    **无扩散曲线**（实测 nested 偏差 1215 Ω），图内 rms/acf 也跟着错。
    现在三族都把 `cfg['diffusion']` 传下去，并核对声明与参数一致。
    """
    diff = cfg.get("diffusion") or None
    if topo == "voigt":
        missing = [k for k in _DIFF_PARAM_KEYS.get(diff, ()) if k not in params]
        if missing:
            raise ValueError(
                f"voigt：config['diffusion']={diff!r} 但参数缺 {missing}；"
                "图与拟合结构必须一致（不能悄悄画成无扩散）")
        return B.model_z(f, params, include_L=bool(cfg.get("include_L", False)))
    if topo == "nested":
        p = dict(params)
        if str(cfg.get("branch")) == "series" and "L_ad" not in p:
            # series 支路只采 R_L/tau_L，`L_ad` 是派生量；漏了正演处会 KeyError
            p["L_ad"] = float(p["R_L"]) * float(p["tau_L"])
        return T.z_nested(f, p, branch=str(cfg["branch"]),
                          layers=int(cfg["layers"]), diffusion=diff,
                          include_L=bool(cfg.get("include_L", False)))
    if topo == "adr":
        return T.z_adr(f, dict(params), diffusion=diff,
                       include_L=bool(cfg.get("include_L", False)))
    raise ValueError(f"拓扑 {topo!r} 没有正演适配器")


#: 族 → 线型。12 条候选（三族 × 三种扩散 + 基线）时单靠颜色分不清，
#: 线型让"这条属于哪个族"一眼可读；单族出图时仍是全实线（外观不变）。
_FAMILY_STYLE = {"voigt": "-", "nested": "--", "adr": "-."}


def curve_from_fit(d: dict, f):
    """取 `fits/*.json` 里**已落盘**的曲线（逐样本中位，F1 口径）。

    优先用 `z_fit`：它保证"图上的线"和"表里的 rms"是同一条数组。
    旧记录没有该字段时返回 None，调用方按参数重建——但那种重建是
    `Z(中位参数)`，与 F1 的 `median Z(θ)` 不完全等价，必须计数并标注。
    """
    zf = d.get("z_fit")
    if not isinstance(zf, dict):
        return None
    re, im = zf.get("re"), zf.get("im")
    if re is None or im is None:
        return None
    z = np.asarray(re, float) + 1j * np.asarray(im, float)
    return z if z.size == np.asarray(f).size else None


def label_of(topo: str, cfg: dict) -> str:
    if topo == "voigt":
        return (f"voigt K={cfg.get('K')}"
                + ("+L" if cfg.get("include_L") else "")
                + (f"+{cfg['diffusion']}(research)" if cfg.get("diffusion") else ""))
    if topo == "adr":
        return ("adr" + ("+L" if cfg.get("include_L") else "")
                + (f"+{cfg['diffusion']}(research)" if cfg.get("diffusion") else ""))
    return (f"nested L={cfg.get('layers')}/{cfg.get('branch')}"
            + ("+L" if cfg.get("include_L") else "")
            + (f"+{cfg['diffusion']}(research)" if cfg.get("diffusion") else ""))


def one(fp: str, a, ks, layers, branches) -> dict:
    f, Z = R.load_spectrum(fp)
    stem = os.path.splitext(os.path.basename(fp))[0]
    out = os.path.join(a.out, stem)
    include_L = need_L(f, Z) if a.include_L is None else bool(a.include_L)
    # L 的**来源**要落盘（2026-11-20）：auto 判定 vs 人工强制，解释"元件数"
    # 和"为什么这批都有 L"时不能靠回忆。`need_L` 是简化的 auto_mechanism
    # 替身（只看高频端 Im>0 过半），不是同一套判据，来源必须写清。
    include_L_source = "auto_need_L" if a.include_L is None else "forced"

    meta = RU.run(fp, seed=a.seed, budget_profile=a.budget, out_dir=out,
                  ks=ks, layers=layers, include_L=include_L,
                  branches=branches, include_adr=a.include_adr,
                  include_diffusion=a.include_diffusion,
                  likelihood=a.likelihood)
    ev = EV.evaluate_run(out)

    # 元件数：plan.json 的 candidate.elements 才是登记事实。
    # 不能用 fit json 的 params 键数 —— HN 的多个分量被压成向量（K=3 只有 6 键）。
    n_el_by_cid = {}
    pj = os.path.join(out, "plan.json")
    if os.path.exists(pj):
        with open(pj, encoding="utf-8") as fh:
            _plan = json.load(fh)
        for _c in (_plan.get("candidates") or []):
            _els = _c.get("elements")
            if isinstance(_els, (list, tuple)):
                n_el_by_cid[_c.get("candidate_id")] = len(_els)

    curves = []
    _rebuilt = 0
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
            Zm = curve_from_fit(d, f)
            if Zm is None:                  # 旧记录没有 z_fit → 按参数重建
                Zm = forward(topo, cfg, params_from_json(topo, d["params"]), f)
                _rebuilt += 1
        except Exception:                       # noqa: BLE001
            continue
        m = RU.metrics(f, Z, Zm)
        # n_elements 取自 plan.json 的 candidate.elements（登记事实），
        # bands 取自 metrics（低/中/高三段 rms%）。都只用于显示顺序与分段陈述。
        curves.append(dict(label=label_of(topo, cfg), Zc=Zm, topo=topo,
                           style=_FAMILY_STYLE.get(topo, "-"),
                           rms=m["rms_rel"], acf=m["acf"], cid=cid,
                           rhat=d.get("rhat_max"), div=d.get("divergences"),
                           n_elements=n_el_by_cid.get(cid),
                           bands=list(m.get("bands") or [])))

    # 显示顺序（三段式，不宣布最优、不删候选）：
    #   ① 先按绝对验收线 rms<=10% 划 qualified；超线者排后面并标"（超验收线）"
    #   ② qualified 内按 rms 升序
    #   ③ rms 绝对差<=tie_pp 且相对差<=tie_rel、分段无实质更差段者，才按元件数升序。
    #      三者缺一不重排 —— 防"把真谱排掉、留元件少者"。
    _accept = 10.0
    _tie_pp = getattr(EV, "DEFAULT_TIE_PP", 1.0)
    _tie_rel = getattr(EV, "DEFAULT_TIE_REL", 10.0)
    _qualified = [c for c in curves if c["rms"] <= _accept]
    _over = [c for c in curves if c["rms"] > _accept]
    _qualified.sort(key=lambda c: c["rms"])
    # ③ 三条件门槛：先在 rms 并列带里找候选，再按分段模式筛，最后按元件数排。
    #    分段模式不同者**不重排**（仍留在原 rms 位置），防"把真谱排掉、留元件少者"。
    _sort_note = ""
    if _qualified:
        _floor = _qualified[0]["rms"]
        _tie_group = [c for c in _qualified
                      if (c["rms"] - _floor <= _tie_pp
                          and (_floor > 0 and (c["rms"] - _floor) / _floor * 100 <= _tie_rel))]

        def _bands_ok(c):
            """候选相对 best 的连续分段比较：任一频段差 > tie_pp 即不算同档。

            不用二值过/超：9.9(过) 与 10.1(超) 只差 0.2pp 会被二值拆开，
            而 1.2(过) 与 3.4(过) 差 2.2pp 却同类 —— 二值化丢了连续信息。
            """
            b = c.get("bands")
            if not b or len(b) != 3 or any(x is None for x in b):
                return False
            bb = _qualified[0].get("bands")
            if not bb or len(bb) != 3:
                return False
            return all(float(b[i]) - float(bb[i]) <= _tie_pp for i in range(3))

        if not _qualified[0].get("bands"):
            _tie_group = []            # 基准分段不可用 ⇒ 保守不重排
            _sort_note = "基准候选缺分段数据 ⇒ 未做元件数重排"
        else:
            _keep = [c for c in _tie_group if _bands_ok(c)]
            _dropped = [c["label"] for c in _tie_group if not _bands_ok(c)]
            if _dropped:
                _sort_note = "分段连续比较有实质更差段者不参与元件数重排：" + "、".join(_dropped)
            _tie_group = _keep
        if len(_tie_group) > 1 and all(c.get("n_elements") is not None for c in _tie_group):
            _tie_group.sort(key=lambda c: (c["n_elements"], c["rms"]))
            # 只替换并列带里那几个的位置；分段不一致者留在原处。
            _slot = [_qualified.index(c) for c in _tie_group]
            for idx, c in zip(sorted(_slot), _tie_group):
                _qualified[idx] = c
    curves = _qualified + _over
    for i, c in enumerate(curves):
        c["color"] = FIG.COLORS[i % len(FIG.COLORS)]
        c["over_accept"] = c["rms"] > _accept

    head = (f"{stem}  n={f.size}  {f.min():.3g}–{f.max():.3g} Hz  "
            f"{a.budget}  σ={a.likelihood}  seed={a.seed}"
            + ("  +L(auto)" if include_L else ""))
    # 顺序说明：并列带内按元件数排会把 rms 更低但元件多的候选放后面，
    # 不写清楚会被误读成"排序错了"。这里是事实陈述，不是宣布最优。
    _order_note = ("显示顺序：过验收线组在前；rms 绝对差 ≤ tie_pp 且相对差 ≤ tie_rel、"
                   "配对 elpd 不显著、且分段连续比较无实质更差段者才按元件数升序"
                   "（元件少者列前，故 rms 非严格升序）")
    if _sort_note:
        _order_note += "；" + _sort_note
    head += "\n  " + _order_note
    if len({c.get("topo") for c in curves}) > 1:
        head += "\n  线型：voigt 实线 / nested 虚线 / adr 点划线（族可辨，不是拟合质量编码）"
    if _rebuilt:
        head += (f"\n  {_rebuilt} 条旧记录缺 z_fit，曲线按参数中位重建"
                 "（≠逐样本中位，仅参考）——缓存按 candidate_id 复用，"
                 "要修正需换输出目录重跑")
    def _band_status(bands):
        """三段 rms% 对照全谱验收线 10%：过/未过。数据没有就标不可用，不猜。"""
        if not bands or len(bands) != 3:
            return "分段不可用"
        marks = ["".join("过" if (b is not None and b <= _accept) else "超"
                         for b in bands)]
        return "低/中/高=" + "/".join(
            ("—" if b is None else f"{b:.1f}") for b in bands) + marks[0]

    # 标签列宽按实际最长标签算（2026-11-20）：三族扩散候选的标签可达
    # `nested L=2/parallel+Wshort(research)`（35 字符），固定 22 会把后续列
    # 顶歪，读起来像数据错位。
    _lw = max([len(c["label"]) for c in curves] + [22])
    lines = [f"{c['label']:<{_lw}} rms={c['rms']:5.2f}%  acf={c['acf']:.3f}"
             + (f"  元件={c['n_elements']}" if c.get("n_elements") else "")
             + ("（超验收线）" if c.get("over_accept") else "")
             + (f"  R̂={c['rhat']:.3f}  div={c['div']}" if c["rhat"] else "")
             for c in curves]
    band_lines = [f"    {c['label']:<{_lw}} {_band_status(c.get('bands'))}"
                  for c in curves]
    head = head + "\n  分段 rms%（低 0.01–1 / 中 1–100 / 高 100–1e5；线=10%）"
    meta_txt = head + "\n" + "\n".join(lines + band_lines) \
        if lines else head + "\n（无可行候选）"

    png = os.path.join(out, "compare.png")
    FIG.plot_fit(png, f, Z, curves,
                 title=f"{stem} —— {len(curves)} 个候选并列",
                 meta=meta_txt)

    rec = dict(file=fp, data_sha256=meta["data_sha256"], n_points=int(f.size),
               budget_profile=a.budget, likelihood=a.likelihood,
               seed=a.seed, include_L=include_L,
               include_L_source=include_L_source,
               status=ev["status"], reason=ev["reason"],
               # 配对 elpd 差（2026-11-20）：method=paired 时 dse 才是配对 dse；
               # marginal_se 是降级参考，不得混称。与 evaluation.json 同源。
               elpd_comparison=ev.get("tie"),
               # 临时准入：`compare` 用的族（含 nested）没过 §13/§14，所以每份
               # 产物都要能自证"这是能跑、不是达标"。见 `admission.py`。
               admission=AD.block(),
               confirmation=ev.get("confirmation"),
               candidates=[dict(cid=c["cid"], label=c["label"],
                                rms_rel=c["rms"], acf=c["acf"],
                                rhat_max=c["rhat"], divergences=c["div"],
                                n_elements=c.get("n_elements"),
                                bands=c.get("bands"),
                                over_accept=bool(c.get("over_accept")))
                           for c in curves],
               acceptance=dict(rms_max=_accept, tie_pp=_tie_pp,
                               tie_rel=_tie_rel,
                               reorder_conditions=("rms abs≤tie_pp 且 rel≤tie_rel "
                                                   "且 elpd 不显著 "
                                                   "且分段过线模式一致"),
                               sort_note=_sort_note,
                               note="显示顺序：先按验收线分组，过线组内 rms 升序，"
                                    "三条件都满足（rms 并列、配对 elpd 不显著、"
                                    "分段无实质更差段）的并列带内元件少者列前；"
                                    "分段实质更差者不重排；不删候选、不宣布最优"),
               png=png, run_dir=out)
    with open(os.path.join(out, "compare.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1, allow_nan=False)

    print(f"\n[{stem}] 状态 {ev['status']}（{ev['reason']}）")
    _tie = ev.get("tie") or {}
    if _tie.get("delta_elpd") is not None:
        _dse = _tie.get("delta_dse")
        _z = _tie.get("delta_z")
        _ds = (f"{'配对 dse' if _tie.get('elpd_method') == 'paired' else '边缘 se(降级)'}"
               f"={_dse:.2f}" if _dse is not None else "dse 不可用")
        _zs = f"  z={_z:.2f}" if _z is not None else ""
        print(f"  elpd：Δ={_tie['delta_elpd']:.2f}  {_ds}{_zs}"
              + ("（显著）" if _tie.get("significant") else "（不显著）"))
    print(f"  {_order_note}")
    for ln in lines:
        print("  " + ln)
    for bl in band_lines:
        print(bl)
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
    # 2026-11-22：默认从"追加"改为"不追加"。理由见 ADR_FAMILY_VERDICT.md —
    # 同参数数跨 5 类 13 条，串联式带 L 时 12 胜 / 1 平 / 0 负，adr 多出的自由度
    # 正好是不可辨那一维（1/Rct+1/R_ad 只能以和出现）。`--no-adr` 保留为兼容别名：
    # 它现在与默认行为相同，传了也无效果，但不让旧命令行报错。
    ap.add_argument("--adr", dest="include_adr", action="store_true",
                    default=False,
                    help=("追加 adr（吸附弛豫三支路并联）候选；默认**不**追加。"
                          "研究用途：对标外部 adsorption_RQ / 作欠定负对照"))
    ap.add_argument("--no-adr", dest="include_adr", action="store_false",
                    default=argparse.SUPPRESS,
                    help="兼容旧命令；2026-11-22 起已默认关，本参数无效果")
    ap.add_argument("--diffusion", dest="include_diffusion", action="store_true",
                    help=("追加 voigt 扩散 research 候选（w_inf/Ws、w_open/Wo、"
                          "w_short/Wshort）；默认关；接入≠准入，只给统计事实"))
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
          f"voigt K∈{ks}、nested L∈{layers}/{','.join(branches)}"
          + ("、+adr" if a.include_adr else "")
          + ("、+diffusion(research)" if a.include_diffusion else ""))
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

    def c(name, cond, why=""):
        nonlocal ok, total
        total += 1
        print(f"  [{'ok' if cond else 'FAIL'}] {name}"
              + ("" if cond or not why else f"   ← {why}"))
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
    c("label：扩散候选带 research 标记（三种接法可区分）",
      all(label_of("voigt", dict(K=2, include_L=True, diffusion=d))
          == f"voigt K=2+L+{d}(research)" for d in ("Ws", "Wo", "Wshort")))
    c("label：nested/adr 的扩散候选也带 research 标记（三族不混淆）",
      label_of("nested", dict(layers=2, branch="parallel", diffusion="Wo"))
      == "nested L=2/parallel+Wo(research)"
      and label_of("adr", dict(diffusion="Wshort")) == "adr+Wshort(research)")
    c("label：nested 带层数与支路",
      label_of("nested", dict(layers=2, branch="parallel")) == "nested L=2/parallel")
    c("label：adr 固定名", label_of("adr", {}) == "adr")

    padr = params_from_json("adr", dict(Rs=1.0, Q=2.0, n=0.8, Rct=3.0,
                                        R_ad=4.0, tau_ad=5.0))
    c("adr 参数全转 float",
      all(isinstance(v, float) for v in padr.values()) and len(padr) == 6)
    Zmadr = forward("adr", {}, padr, f)
    c("adr 正演形状与频率一致", Zmadr.shape == f.shape and np.isfinite(Zmadr).all())

    # 三族画图口径一致性（2026-11-20）：声明了扩散，forward 必须真的加进去，
    # 且加在**该族的正确节点**上（voigt 顶层串联 / nested 最内层 / adr 与 Rct 串联）。
    # 判据不用"和拟合路径比"（那是同义反复），而是按各族网络**手算合成值**：
    # 这样既能抓"根本没传"，也能抓"传了但加错位置"。
    import bm_diff as DF
    _wr = 2.0 * np.pi * float(np.sqrt(f.min() * f.max()))

    def _nested_l1(p, zw):
        zb = T.z_rl_parallel(f, p["R_ad"], p["tau_ad"])
        zc = T.z_cpe(f, p["Q1"], p["n1"], f_ref=_wr / (2 * np.pi))
        return p["Rs"] + 1.0 / (1.0 / zc + 1.0 / (p["R1"] + zb + zw))

    def _adr(p, zw):
        zc = T.z_cpe(f, p["Q"], p["n"], f_ref=_wr / (2 * np.pi))
        za = T.z_rl_parallel(f, p["R_ad"], p["tau_ad"])
        return p["Rs"] + 1.0 / (1.0 / zc + 1.0 / (p["Rct"] + zw) + 1.0 / za)

    probes = (
        ("voigt", dict(K=1, include_L=False),
         dict(R_inf=10., W=20., pi=np.array([1.]), tau=np.array([.1]),
              phi=np.array([.8]), zeta=np.array([1.])),
         "Wshort", dict(Rws=5., tauD=0.2),
         lambda p, zw: B.model_z(f, {k: v for k, v in p.items()
                                     if k not in ("Rws", "tauD")}) + zw),
        ("nested", dict(layers=1, branch="parallel"),
         dict(Rs=5., R1=20., Q1=30., n1=.85, R_ad=10., tau_ad=.05),
         "Wo", dict(Rwo=7., tauD=0.2),
         _nested_l1),
        ("adr", {},
         dict(Rs=5., Q=30., n=.85, Rct=20., R_ad=10., tau_ad=.05),
         "Ws", dict(Aw=3.),
         _adr),
    )
    for topo, base_cfg, base_p, dv, extra, expect_fn in probes:
        p_on = dict(base_p, **extra)
        zw = {"Ws": lambda: DF.z_winf(f, 3.),
              "Wo": lambda: DF.z_wopen(f, 7., 0.2),
              "Wshort": lambda: DF.z_wshort(f, 5., 0.2)}[dv]()
        try:
            got = forward(topo, dict(base_cfg, diffusion=dv), p_on, f)
            want = expect_fn(p_on, zw)
            d_off = forward(topo, dict(base_cfg), base_p, f)
            cond = (float(np.max(np.abs(got - want))) < 1e-9
                    and float(np.max(np.abs(got - d_off))) > 1e-6)
            why = (f"{topo}+{dv} 对拍 max|Δ|="
                   f"{float(np.max(np.abs(got - want))):.2e}，"
                   f"与无扩散差={float(np.max(np.abs(got - d_off))):.2e}")
        except Exception as e:                   # noqa: BLE001
            cond, why = False, f"{topo}+{dv} 抛错 {e!r}"
        c(f"画图口径：{topo} 的 forward 真的加了 {dv}", cond, why)
    c("画图口径：voigt 声明扩散但参数缺键 → 抛错（不静默画成无扩散）",
      _raises(lambda: forward("voigt", dict(K=1, diffusion="Wo"),
                              dict(R_inf=10., W=20., pi=np.array([1.]),
                                   tau=np.array([.1]), phi=np.array([.8]),
                                   zeta=np.array([1.])), f)))

    # 顶层电感三族同口径（2026-11-20）：need_L 判定要 L 时，三族都必须能加，
    # 否则"高频感抗"只有 voigt 能表达，跨族比较不是同一模型空间。
    import circuit_registry as CR
    _cl = RU.build_candidates(ks=(1,), layers=(1,), include_L=True,
                              include_adr=True, budget_profile="quick")
    c("L：三族都生成 include_L 候选（各挂一个顶层感抗槽位）",
      {x.topology_id for x in _cl} == {"voigt", "nested", "adr"}
      and all(dict(x.config).get("include_L") is True for x in _cl)
      and all([e[0] for e in x.elements].count("series_inductor") == 1
              for x in _cl))
    for _t, _cfg, _p in (
            ("voigt", dict(K=1),
             dict(R_inf=10., W=20., pi=np.array([1.]), tau=np.array([.1]),
                  phi=np.array([.8]), zeta=np.array([1.]))),
            ("nested", dict(layers=1, branch="parallel"),
             dict(Rs=5., R1=20., Q1=30., n1=.85, R_ad=10., tau_ad=.05)),
            ("adr", {},
             dict(Rs=5., Q=30., n=.85, Rct=20., R_ad=10., tau_ad=.05))):
        try:
            got = (forward(_t, dict(_cfg, include_L=True), dict(_p, L=2e-6), f)
                   - forward(_t, dict(_cfg), _p, f))
            want = 1j * 2 * np.pi * f * 2e-6
            _cond = float(np.max(np.abs(got - want))) < 1e-9
            _why = f"{_t} max|Δ|={float(np.max(np.abs(got - want))):.2e}"
        except Exception as e:                   # noqa: BLE001
            _cond, _why = False, f"{_t} 抛错 {e!r}"
        c(f"画图口径：{_t} 的 forward 真的加了 jωL", _cond, _why)
    c("L：include_L=True 但没挂顶层感抗 → registry 拒绝（不静默丢 L）",
      _raises(lambda: CR.make_candidate(
          "nested", [("outer_cpe", "cpe"), ("branch", "rl_parallel")],
          {"layers": 1, "branch": "parallel", "include_L": True})))
    c("label：nested/adr 的 L 也带 +L",
      label_of("nested", dict(layers=2, branch="parallel", include_L=True))
      == "nested L=2/parallel+L"
      and label_of("adr", dict(include_L=True)) == "adr+L")

    # 曲线取数口径（2026-11-20）：优先用落盘的 z_fit（逐样本中位，F1），
    # 缺字段必须返回 None 让调用方计数——不能悄悄换成 Z(中位参数)。
    _zf = dict(z_fit=dict(re=list(np.zeros(5)), im=list(np.ones(5))))
    c("曲线取数：有 z_fit 就用落盘数组",
      curve_from_fit(_zf, np.arange(5.)) is not None
      and np.allclose(curve_from_fit(_zf, np.arange(5.)), 1j))
    c("曲线取数：缺 z_fit / 长度不符 → None（由调用方重建并标注）",
      curve_from_fit({}, np.arange(5.)) is None
      and curve_from_fit(dict(z_fit=dict(re=[1.], im=[0.])),
                         np.arange(5.)) is None)

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
