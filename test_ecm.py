# -*- coding: utf-8 -*-
"""test_ecm.py —— 用 AutoECM 的 ECM 模拟数据给 bmdrt 做端到端回归。

数据来源
--------
`AutoECM/data/train_data.csv`：7462 条由**等效电路模型**（ECM）正演出来的
阻抗谱，每行带 `Circuit`（电路结构）与 `Parameters`（真值参数）。
这是目前唯一一批「真值已知、且不是我们自己造的」外部数据，适合当回归集。

为什么不能直接拿来比 K
----------------------
数据集的参数是**在宽区间里随机撒**的，与测量窗口无关。统计下来：
只有 **16.3%** 的谱「所有分量的峰位都落在测量窗口内」，75.7% 至少有一个
分量在窗口外。所以随机抽到的谱大多是难例——这本身是数据性质，不是代码问题。
需要「干净恢复」的对照时用 `--stratified`（只从 16.3% 里抽）。

两个必须记住的口径
------------------
1. **数据集用 ω=f 生成**（不是 ω=2πf）。`synth.py` 里也写了这一点。
   但对 R‖CPE 元件，`f0 = (1/(RQ))^(1/n)` 在两种口径下**是同一个数**
   （τ 相差 2π，而 f0=1/(2πτ) 的换算正好抵消），所以报告里的 f0 可以直接比。
   真正受影响的只有 τ 本身，以及由 f 窗口反推的 τ 先验——后者也是按 f0 对齐的，
   所以拟合窗口覆盖范围没问题。
2. **R_inf=0 的电路是 bmdrt 的结构性短板**：模型里
   `R_inf ~ LogNormal(log(0.2·|Z|max), 1)`，3σ 下界约 `0.01·|Z|max`，
   表达不了「没有串联电阻」的 ECM。这类谱要预期 R_inf 被顶高。

用法
----
    python test_ecm.py                       # 随机 3 组（seed=20261105），全量采样
    python test_ecm.py --quick               # 粗筛，只验证流程跑得通
    python test_ecm.py --stratified          # 只从「分量全在窗口内」的子集抽
    python test_ecm.py --only 5940,5687      # 指定行号（复现某次抽样）
    python test_ecm.py --n 5 --seed 7

产物
----
    out/ecm_test/csv/<stem>.csv      导出给 bmdrt 的三列 CSV
    out/ecm_test/res/<stem>/         bmdrt 的完整输出（图 / json / npz / log）
    out/ecm_test/ECM_TEST.md         对真值的逐项核对表
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
#: 默认数据源。换机器/换位置用 --src 覆盖。
DEFAULT_SRC = Path(r"D:\eis训练数据\AutoECM\data\train_data.csv")
OUTBASE = ROOT / "out" / "ecm_test"


# --------------------------------------------------------------------------
# 读数据集 / 解析真值
# --------------------------------------------------------------------------

def parse_params(s: str) -> dict:
    """把 `R1: 6.30e+01, CPE1_t: 7.46e-01, ...` 解析成 dict。"""
    return {k: float(v) for k, v in
            re.findall(r"([A-Za-z0-9_]+)\s*:\s*([-+0-9.eE]+)", s)}


def truth_of(row) -> dict:
    """从 `Circuit` + `Parameters` 还原真值。

    只覆盖能用 ZARC（R‖CPE / R‖C）表达的部分；Gerischer(G) 与
    广义 Warburg(Ws) 单独记，因为它们不是 ζ=1 的 HN 基底。
    """
    p = parse_params(row["Parameters"])
    circ = row["Circuit"]
    f = np.array(ast.literal_eval(row["freq"]), float)

    comps = []
    # **电阻编号不总等于 CPE 编号**：`L-R-RCPE-RCPE` 里 R1 是串联电阻，
    # CPE1 配的是 R2、CPE2 配的是 R3。漏了这个偏移会把 R_inf 当成
    # 第一个分量的电阻，f0 直接算错 8 个量级。
    off = 1 if circ.startswith("L-R-") else 0
    # R‖CPE：参数名里出现 CPE{k}_C 就对应一个分量
    for k in re.findall(r"CPE(\d+)_C", row["Parameters"]):
        R, Q, n = p[f"R{int(k) + off}"], p[f"CPE{k}_C"], p[f"CPE{k}_t"]
        # τ=(RQ)^(1/n)，f0=1/(2πτ)。数据集用 ω=f，但 f0 与口径无关（见模块说明）。
        f0 = (1.0 / (R * Q)) ** (1.0 / n)
        comps.append(dict(kind="RCPE", tag=f"CPE{k}", R=R, phi=n, f0=f0))
    # R‖C：没有同名 CPE 的 C{k}
    for k in re.findall(r"(?<![A-Za-z])C(\d+)\s*:", row["Parameters"]):
        if f"CPE{k}_C" in p or f"R{k}" not in p:
            continue
        R, C = p[f"R{k}"], p[f"C{k}"]
        comps.append(dict(kind="RC", tag=f"C{k}", R=R, phi=1.0,
                          f0=1.0 / (2 * np.pi * R * C)))
    # Gerischer：Z = R/sqrt(1+jωt)，是 ζ=0.5 的 HN，ZARC 基底表达不了
    ger = [dict(R=p[k], t=p[k.replace("R_g", "t_g")])
           for k in re.findall(r"R_g\d+", row["Parameters"])]
    has_ws = "Ws" in circ

    # 串联 R_inf：L-R-* / R1 在 RC-RC-* 里属于第一个 RC，不是 R_inf
    r_inf = 0.0
    if circ.startswith("L-R-") or circ in ("Rs_Ws",):
        r_inf = p.get("R1", 0.0)
    L = p.get("L1", 0.0)

    inside = [f.min() <= c["f0"] <= f.max() for c in comps]
    return dict(circ=circ, params=p, f=f, n=f.size, f_min=float(f.min()),
                f_max=float(f.max()), comps=comps, K=len(comps),
                gerischer=ger, has_ws=has_ws, R_inf=r_inf, L=L,
                n_inside=sum(inside),
                all_inside=bool(inside) and all(inside))


def _can_build(row) -> bool:
    """能否用真值参数重建 Z（用于自检导出是否正确）。"""
    return not truth_of(row)["has_ws"]      # Ws 的正演式未识别，跳过自检


def rebuild(row, t=None) -> np.ndarray:
    """按真值参数重建 Z（ω=f 口径），只用于校验导出。"""
    t = t or truth_of(row)
    p, f = t["params"], t["f"]
    Z = np.full(f.size, t["R_inf"], complex)
    for c in t["comps"]:
        if c["kind"] == "RCPE":
            k = c["tag"][3:]
            Q, n = p[f"CPE{k}_C"], p[f"CPE{k}_t"]
            Z += c["R"] / (1 + c["R"] * Q * (1j * f) ** n)
        else:
            k = c["tag"][1:]
            Z += c["R"] / (1 + 1j * f * c["R"] * p[f"C{k}"])
    for g in t["gerischer"]:
        Z += g["R"] / np.sqrt(1 + 1j * f * g["t"])
    if t["L"]:
        Z += 1j * f * t["L"]
    return Z


# --------------------------------------------------------------------------
# 抽样 / 导出
# --------------------------------------------------------------------------

def load_table(src: Path):
    import pandas as pd
    if not src.exists():
        sys.exit(f"找不到数据源：{src}\n用 --src 指定 train_data.csv 的路径")
    return pd.read_csv(src)


def sample(df, n: int, seed: int, stratified: bool,
           composable: bool = False, by_k: bool = False):
    """抽样。

    stratified : 只留「所有分量峰位都在窗口内」的行——数据良态，用来检验
                 K 与参数能不能被恢复。
    composable : 只留纯 ZARC/RC 电路（排除 Gerischer 与 Ws）。这两个元素
                 不是 ζ=1 的 HN 基底，K 与它们没有可比性。
    by_k       : 按真值 K 分组，每组抽 1 条（避免全抽到同一个结构）。
    """
    # 不加任何筛选时**保留全表**：否则同一 seed 会因为池子变了而抽出
    # 不同的行，`--reuse` 就对不上之前的结果文件。
    need_truth = stratified or composable or by_k
    pool = []
    for i in range(len(df)):
        if not need_truth:
            pool.append((i, -1))
            continue
        t = truth_of(df.iloc[i])
        if not t["K"]:
            continue
        if composable and (t["has_ws"] or t["gerischer"]):
            continue
        if stratified and not t["all_inside"]:
            continue
        pool.append((i, t["K"]))
    if not pool:
        sys.exit("筛选后样本池为空，放宽条件再试")
    print(f"[抽样] 池 {len(pool)} 条"
          + ("（分量全在窗口内）" if stratified else "")
          + ("（纯 ZARC 可组合）" if composable else ""))

    rng = np.random.default_rng(seed)
    if by_k:
        groups: dict[int, list[int]] = {}
        for i, k in pool:
            groups.setdefault(k, []).append(i)
        ks = sorted(groups)[:n]
        picked = [int(rng.choice(groups[k])) for k in ks]
        print("[抽样] 按 K 分组：" + "、".join(f"K={k}({len(groups[k])} 条)"
                                              for k in ks))
        return sorted(picked)

    idx = [i for i, _ in pool]
    if len(idx) < n:
        sys.exit(f"样本池只有 {len(idx)} 条，不够抽 {n} 条")
    return sorted(int(x) for x in rng.choice(idx, size=n, replace=False))


def export(t: dict, path: Path, noise: float = 0.0, seed: int = 0):
    """把真值谱导出成 CSV，返回 (路径, 实际写出的 Z, 无噪 Z)。

    noise : 相对噪声水平（占 |Z| 的比例），0 = 数据集原样的**无噪**谱。

    **为什么要能加噪声**：数据集里的 Z 是 ECM 正演出来的，没有测量噪声。
    bmdrt 的似然里 σ_re/σ_im 是自由参数且先验 `LogNormal(log(0.01·zmax), 1)`
    **没有下界**——遇到无噪数据它会一路塌到相对 1e-8 量级，NUTS 在
    σ→0 的漏斗里树深饱和（实测触顶 28%–53%），采样分级直接掉到 0，
    于是所有 K 都被判「不可用」、`k_selection_status=rejected`。
    这跟模型对不对无关，是似然对无噪数据不适配。要检验 K 选择必须注入噪声。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Z0 = np.array(ast.literal_eval(t["row_Z"]), complex)
    Z = Z0
    if noise > 0:
        rng = np.random.default_rng(seed)
        mag = np.abs(Z0)
        # 实/虚各给 σ=noise·|Z|/√2 的独立高斯，合起来 |ΔZ| 的 RMS ≈ noise·|Z|
        Z = Z0 + noise * mag * (rng.standard_normal(Z0.size)
                                + 1j * rng.standard_normal(Z0.size)) / np.sqrt(2.0)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("f_Hz,Zre,Zim\n")
        for a, b in zip(t["f"], Z):
            fh.write(f"{a:.10g},{b.real:.10g},{b.imag:.10g}\n")
    return path, Z, Z0


# --------------------------------------------------------------------------
# 跑 bmdrt
# --------------------------------------------------------------------------

def run_one(csv: Path, outdir: Path, quick: bool, kmax: int,
            sigma_floor: float = 0.0, k_force: int | None = None,
            quiet: bool = False):
    """跑一条谱。

    k_force 给定时**跳过 K 扫描**，只拟合这一个 K。做 σ 地板这类
    "固定模型、只改一个旋钮"的对照时必须这样——否则 K 也在变，
    参数漂移分不清是地板造成的还是 K 造成的；而且每多一个 K 就多一次
    JAX 重编译（实测一条谱 3 个 K ≈ 15 分钟）。

    quiet=True 时把 run() 的打印吞掉（扫描时几十次输出没人看）。
    """
    import bmdrt as M

    cfg = M._deep_merge(M.DEFAULTS,
                        json.loads((ROOT / "config.json").read_text(encoding="utf-8")))
    cfg = M.resolve(cfg)
    cfg["paths"]["out"] = outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cfg["batch"]["move_done"] = False
    cfg["batch"]["skip_done"] = False
    cfg["fit"]["quick"] = bool(quick)
    cfg["fit"]["kmax"] = int(kmax)
    cfg["fit"]["sigma_floor"] = float(sigma_floor or 0.0)

    if k_force is not None:
        return _run_forced(csv, outdir, cfg, int(k_force), quiet=quiet)

    t0 = time.time()
    rec = M.process_one(csv, cfg)
    rec["wall"] = time.time() - t0
    js = outdir / csv.stem / f"real_{csv.stem}.json"
    rec["result"] = (json.loads(js.read_text(encoding="utf-8"))
                     if js.exists() else None)
    return rec


def _run_forced(csv: Path, outdir: Path, cfg: dict, K: int, quiet: bool = False):
    """固定 K 直接调 bm_real.run，绕开 bmdrt 的 K 扫描与批处理簿记。"""
    import contextlib
    import io
    import bm_real as R

    fit = cfg["fit"]
    kkcfg = cfg.get("kk") or {}
    kk_opts = {k: kkcfg[k] for k in ("c", "max_M", "res_tol",
                                     "auto_L", "auto_tail", "auto_cap")
               if k in kkcfg}
    sub = outdir / csv.stem
    sub.mkdir(parents=True, exist_ok=True)
    R.OUT = sub
    R.NULL_CACHE = {}
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf if quiet else sys.stdout):
        R.run(str(csv), K_force=K, kmax=int(fit["kmax"]),
              include_L=fit["include_L"], zeta=str(fit["zeta"]),
              quick=bool(fit["quick"]), ta=fit["ta"],
              kk_mode=str(kkcfg.get("mode", "gate")), kk_opts=kk_opts,
              diag=cfg.get("diag") or {}, drt_opts=cfg.get("drt") or {},
              sigma_floor=float(fit.get("sigma_floor") or 0.0))
    js = sub / f"real_{csv.stem}.json"
    return dict(wall=time.time() - t0, ok=js.exists(),
                result=(json.loads(js.read_text(encoding="utf-8"))
                        if js.exists() else None))


# --------------------------------------------------------------------------
# 对真值
# --------------------------------------------------------------------------

def compare(t: dict, res: dict | None) -> dict:
    """把结果和真值逐项对齐。分量按 f0 最近匹配（log 距离）。"""
    if not res or res.get("fit_performed") is False:
        return dict(fit=False, kk=(res or {}).get("kk", {}).get("status"),
                    k_sel=None, verdict="未拟合（KK 拒绝）", rows=[])

    got = res.get("components") or []
    # 结果里只给 f0；真值分量按 f0 排序，两边都按 f0 降序配对
    pairs, used = [], set()
    for g in sorted(got, key=lambda c: -c["f0"]):
        best, bi = None, None
        for i, c in enumerate(t["comps"]):
            if i in used:
                continue
            d = abs(np.log10(g["f0"] / c["f0"]))
            if best is None or d < best:
                best, bi = d, i
        if bi is not None and best < 1.0:      # 1 decade 内才算匹配上
            used.add(bi)
            c = t["comps"][bi]
            pairs.append(dict(
                f0_true=c["f0"], f0_fit=g["f0"],
                dec=float(np.log10(g["f0"] / c["f0"])),
                R_true=c["R"], R_fit=g["w"],
                R_err=float(g["w"] / c["R"] - 1) if c["R"] else float("nan"),
                phi_true=c["phi"], phi_fit=g["phi"],
                p_out=g.get("p_peak_outside")))
        else:
            pairs.append(dict(f0_true=None, f0_fit=g["f0"], dec=None,
                              R_true=None, R_fit=g["w"], R_err=None,
                              phi_true=None, phi_fit=g["phi"],
                              p_out=g.get("p_peak_outside")))
    matched = len(used)
    kk = (res.get("kk") or {}).get("status")
    st = res.get("k_selection_status")
    return dict(fit=True, kk=kk, k_sel=res.get("K"), status=st,
                grade=res.get("sample_grade"),
                k_match=(matched == t["K"] and len(got) == t["K"]),
                n_matched=matched, rows=pairs,
                R_inf_fit=res.get("R_inf"), R_inf_true=t["R_inf"],
                L_fit=res.get("L_uH"), L_true=(t["L"] * 1e6 if t["L"] else 0.0),
                rms_re=res.get("rms_re"), sig_re=res.get("sigma_re"),
                rms_im=res.get("rms_im"), sig_im=res.get("sigma_im"),
                cover=min(res.get("cover_re") or 1.0, res.get("cover_im") or 1.0),
                acf=res.get("acf"),
                warnings=(res.get("warnings") or [])[:3],
                verdict=_verdict(t, res, matched))


def _verdict(t: dict, res: dict, matched: int) -> str:
    st = res.get("k_selection_status")
    k = res.get("K")
    # 真值没有 ZARC 分量（纯 R / Ws / 纯 Gerischer）时「K 对不对」没有意义，
    # 只能看模型用几个分量去近似它——别把它算成「多给了 K」。
    if t["K"] == 0:
        return (f"真值无 ZARC 分量（{t['circ']}），模型用 K={k} 个分量近似"
                + (f"；选择状态 {st}" if st else ""))
    if st == "rejected":
        return "模型选择被拒绝（无 K 过硬筛选）"
    if st == "exploratory":
        # quick 档：采样预算撑不起严格线，"rejected" 不代表模型被否定。
        # 这里仍然报 K 与分量数，但必须标明不是选择结论。
        base = (f"K={k}（真值 {t['K']}），对上 {matched} 个分量"
                if k != t["K"] else "K 与分量数都对上")
        return f"{base}；**exploratory（quick 档，非选择结论）**"
    if st == "degraded":
        return "降级档（未过严格线）"
    if k == t["K"] and matched == t["K"]:
        return "K 与分量数都对上"
    if matched == t["K"]:
        return f"分量都对上，但 K={k}（真值 {t['K']}）多了碎片"
    return f"K={k}（真值 {t['K']}），只对上 {matched} 个分量"


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------

def _fmt_f(x):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.3g}"


def write_report(items, args, path: Path):
    L = ["# bmdrt × AutoECM 模拟数据回归", "",
         f"- 数据源：`{args.src}`",
         f"- 抽样：{'分层（分量全在窗口内）' if args.stratified else '全表均匀随机'}"
         f"，seed={args.seed}，行号 {[it['idx'] for it in items]}",
         f"- 采样：{'quick（粗筛，指标不可当结论）' if args.quick else 'full（3000/1000/3 链）'}",
         f"- 噪声："
         + (f"注入 {args.noise:.2%}·|Z|（数据集原谱无噪）" if getattr(args, "noise", 0) > 0
            else "**无**（数据集原样；σ 会塌、采样分级不可信）"),
         f"- 噪声地板："
         + (f"{args.sigma_floor:.1e}·zmax" if getattr(args, "sigma_floor", 0) > 0
            else "关闭（σ 无下界）"),
         f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", ""]

    L += ["## 总览", "",
          "| 行号 | 电路 | 窗口 / Hz | 真值 K | 选中 K | 对上分量 | KK | 选择状态 | 判定 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for it in items:
        t, c = it["truth"], it["cmp"]
        L.append(f"| {it['idx']} | {t['circ']} | {t['f_min']:.3g}–{t['f_max']:.3g} "
                 f"| {t['K']} | {c.get('k_sel') or '—'} | "
                 f"{c.get('n_matched', 0)}/{t['K']} | {c.get('kk') or '—'} | "
                 f"{c.get('status') or '—'} | {c.get('verdict')} |")
    L.append("")

    for it in items:
        t, c = it["truth"], it["cmp"]
        L += [f"## [{it['idx']}] {t['circ']}", "",
              f"- 数据：{t['n']} 点，{t['f_min']:.3g}–{t['f_max']:.3g} Hz",
              f"- 真值：K={t['K']}，R_inf={t['R_inf']:.4g} Ω，"
              f"L={t['L'] * 1e6:.3g} µH"
              + (f"，Gerischer×{len(t['gerischer'])}" if t["gerischer"] else "")
              + ("，含 Ws" if t["has_ws"] else ""),
              f"- 窗口内分量：{t['n_inside']}/{t['K']}"
              + ("" if t["all_inside"] else "　**有分量峰位在窗口外**"),
              "",
              "| 真值 f0 / Hz | 拟合 f0 / Hz | Δlog10 | 真值 R / Ω | 拟合 w / Ω | R 相对误差 "
              "| 真值 φ | 拟合 φ | P(峰出窗) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in c.get("rows", []):
            dec = "—" if r["dec"] is None else f"{r['dec']:+.2f}"
            rerr = "—" if r["R_err"] is None else f"{r['R_err'] * 100:+.0f}%"
            pout = "—" if r["p_out"] is None else f"{r['p_out']:.2f}"
            L.append(f"| {_fmt_f(r['f0_true'])} | {_fmt_f(r['f0_fit'])} | {dec} "
                     f"| {_fmt_f(r['R_true'])} | {_fmt_f(r['R_fit'])} | {rerr} "
                     f"| {_fmt_f(r['phi_true'])} | {_fmt_f(r['phi_fit'])} | {pout} |")
        if any(r["f0_true"] is None for r in c.get("rows", [])):
            L.append("> 「真值 f0」为 — 表示 1 decade 内没有对应真值分量：通常是"
                     "真值峰位落在窗口外，模型只能用一个贴边界的分量去近似。")
        L.append("")
        if c.get("fit"):
            L += [f"- 拟合：K={c['k_sel']}，分级 {c.get('grade')}，"
                  f"rms/σ={c['rms_re'] / c['sig_re']:.2f}(实)/{c['rms_im'] / c['sig_im']:.2f}(虚)，"
                  f"PPC={c['cover']:.3f}，acf={c['acf']:.2f}",
                  f"- R_inf：拟合 {c['R_inf_fit']:.4g} Ω / 真值 {c['R_inf_true']:.4g} Ω",
                  f"- 判定：**{c['verdict']}**"]
        else:
            L += [f"- **未拟合**：KK 门禁 {c.get('kk')}"]
        for w in c.get("warnings", []):
            L.append(f"  - {w}")
        L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="用 AutoECM 的 ECM 模拟数据回归 bmdrt")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--only", default=None, help="指定行号，逗号分隔")
    ap.add_argument("--stratified", action="store_true",
                    help="只从「分量全在窗口内」的子集抽（检验干净恢复）")
    ap.add_argument("--composable", action="store_true",
                    help="只留纯 ZARC/RC 电路（排除 Gerischer 与 Ws）")
    ap.add_argument("--by-k", action="store_true",
                    help="按真值 K 分组，每组抽 1 条")
    ap.add_argument("--quick", action="store_true", help="粗筛采样")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="注入相对噪声（占 |Z| 的比例），如 0.005 = 0.5%%。"
                         "数据集本身无噪，检验 K 选择时应加噪声")
    ap.add_argument("--sigma-floor", type=float, default=0.0,
                    help="噪声地板（相对 zmax），如 1e-4。默认 0 = 关闭。"
                         "用来切断无噪数据上 σ→0 的漏斗")
    ap.add_argument("--reuse", action="store_true",
                    help="复用已有结果 JSON 重新出报告（不重拟合）")
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--tag", default="",
                    help="输出子目录后缀：out/ecm_test[_tag]，避免两次抽样互相覆盖")
    a = ap.parse_args(argv)
    a.src = Path(a.src)
    global OUTBASE
    # 加了噪声/地板就别往无噪结果目录里写，否则两份结论混在一起没法比
    tag = a.tag
    if a.noise > 0:
        tag = f"{tag}_noise{int(round(a.noise * 1e4))}bp" if tag \
            else f"noise{int(round(a.noise * 1e4))}bp"
    if a.sigma_floor > 0:
        tag = f"{tag}_floor{int(round(a.sigma_floor * 1e6))}e6" if tag \
            else f"floor{int(round(a.sigma_floor * 1e6))}e6"
    if tag:
        OUTBASE = ROOT / "out" / f"ecm_test_{tag}"

    df = load_table(a.src)
    idx = ([int(x) for x in a.only.split(",")] if a.only
           else sample(df, a.n, a.seed, a.stratified,
                       composable=a.composable, by_k=a.by_k))
    print(f"[test_ecm] 行号 {idx}，共 {len(idx)} 条")

    items = []
    for i in idx:
        row = df.iloc[i]
        t = truth_of(row)
        t["row_Z"] = row["Z"]
        stem = f"ecm_{i:05d}_{t['circ'].replace('-', '_')}"
        csv = OUTBASE / "csv" / f"{stem}.csv"
        csv, Z, Z0 = export(t, csv, noise=a.noise, seed=i)

        # 导出自检：能用真值参数重建 Z 的话，确认 CSV 没写错（对无噪 Z 比）
        note = ""
        if _can_build(row):
            e = float(np.median(np.abs(rebuild(row, t) - Z0) / np.abs(Z0)))
            note = f"正演自检 {e:.1e}"
        else:
            note = "Ws 正演式未识别，跳过自检"

        print(f"\n{'=' * 72}\n[{i}] {t['circ']}  {t['n']} 点 "
              f"{t['f_min']:.3g}–{t['f_max']:.3g} Hz  K={t['K']}  ({note})")
        for c in t["comps"]:
            side = ("窗口内" if t["f_min"] <= c["f0"] <= t["f_max"]
                    else ("窗口下方" if c["f0"] < t["f_min"] else "窗口上方"))
            print(f"    真值分量 {c['tag']:6s} f0={c['f0']:10.3g} Hz  "
                  f"R={c['R']:9.4g} Ω  φ={c['phi']:.3f}  ({side})")

        if a.reuse:
            js = OUTBASE / "res" / stem / f"real_{stem}.json"
            res = json.loads(js.read_text(encoding="utf-8")) if js.exists() else None
            rec = dict(wall=0.0, result=res, reused=True)
            print(f"    复用已有结果：{'是' if res else '否（缺文件）'}")
        else:
            rec = run_one(csv, OUTBASE / "res", a.quick, a.kmax,
                          sigma_floor=a.sigma_floor)
        cmp = compare(t, rec.get("result"))
        print(f"    → {cmp['verdict']}；KK={cmp.get('kk')}；"
              f"状态={cmp.get('status')}；{rec['wall']:.0f}s")
        items.append(dict(idx=int(i), truth=t, cmp=cmp, rec=rec, stem=stem))

    md = write_report(items, a, OUTBASE / "ECM_TEST.md")
    print(f"\n{'=' * 72}\n报告：{md}")
    ok = sum(1 for it in items if it["cmp"].get("k_match"))
    print(f"K 完全对上：{ok}/{len(items)}；"
          f"未拟合：{sum(1 for it in items if not it['cmp'].get('fit'))}/{len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
