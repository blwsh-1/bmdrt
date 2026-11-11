#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""nested_fit.py —— nested 拓扑的**独立研究性接口**（不接 `bmdrt.py` 主流程）。

用途
----
给用户拿自己的谱直接跑 `bm_top.fit_top`：一条命令出参数（含 90% 区间）、
拟合曲线 CSV、Nyquist/Bode/残差三联图、诊断 JSON。也可以当 Python API 用：

    import nested_fit
    r = nested_fit.run("inbox/my_spectrum.csv", budget="standard")

**这个接口不改 `bmdrt.py` / `bm_core.py` / `bm_top.py` / `bm_select.py` 的任何
一行**，不写门禁、不做跨族判决、不改默认族。`bmdrt.bat` 的行为逐位不变。**

状态（必须随结果一起看）
------------------------
`exploratory`。nested 在 `M6_VALIDATION_PLAN.md` §13 留出验收 **0/3**、
§14 全量留出 **4/14**，两次都卡在 M-b 残差自相关；§17 又证四个文献电路
（含 C2/C3 更多自由度）在 10 条未过谱上 M-b 全 0/10。
所以本接口的输出**不是验收结论**，参数不得当物理量/真值直接引用，
`acf` 提示只是把 §14 的口径摆出来供参考，不构成「通过/不通过」。

口径
----
· 默认 `sigma=perpoint`（逐点 `σ_j = 2%·|Z_j|`）与 `budget=standard`
  —— 与 §13/§14 完全同一套口径；`standard` 下 R̂ 稳定（§9.4.22）。
  逐点 σ 由同一批观测算出 → `sigma_from_data=True`，`elpd_loo` **没有**
  标准留一预测解释，不得拿它做模型比较（见 `bm_top.fit_top` 文档）。
· `--sigma free` 走项目默认（σ 由后验采样）；`--sigma scalar` 固定绝对 σ。
  注意 **R16**：绝对 σ 会让高频段「合法地不拟合」；**R17**：噪声口径会
  改变哪族看起来更好，故 `--with-voigt` 的对照只是同口径参考，不是跨族证据。
· `branch=parallel`（默认）= 文献 Table 1 的 C4（`R3 ∥ L`），
  `branch=series` 是另一种感抗支路接法，两者**不等价**。

用法
----
    # 下面 `python` 指项目 venv D:\autoeis_env\Scripts\python.exe
    # （系统 Python 没装 numpyro/jax 会 ImportError；双击 .bat 即可）
    python nested_fit.py                          # 扫 inbox/ 里的谱
    python nested_fit.py data.csv                 # 单条
    python nested_fit.py a.csv b.csv --budget quick
    python nested_fit.py data.csv --sigma free --branch series --layers 1
    python nested_fit.py data.csv --with-voigt --no-plot
    nested_fit.bat data.csv                       # Windows 一键（用项目 venv）

输出默认写到 `out/nested_fit/`，每个输入谱一组：
`<stem>.nested.json` / `<stem>.nested_curve.csv` / `<stem>.nested.png`。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import bm_experiments as E          # noqa: E402
import bm_real as R                # noqa: E402
import bm_top as T                 # noqa: E402
from bm_select import BUDGETS      # noqa: E402  只读常量，不改口径

WARNING = (
    "研究性接口（exploratory）：nested 在 §13 留出 0/3、§14 全量 4/14，"
    "未过验收线；§17 四个文献电路在 10 条未过谱上 M-b 全 0/10。"
    "本接口不做跨族判决、不写门禁、不接 bmdrt.py 主流程；"
    "参数不得当物理量/真值，输出不是验收结论。"
)

FIT_KEYS = ("n_warmup", "n_samples", "num_chains",
            "target_accept", "max_tree_depth")
SUFFIXES = (".csv", ".txt", ".tsv", ".z", ".dat")
DEFAULT_OUT = os.path.join("out", "nested_fit")

# §13/§14 的 `standard` 是 2000/1000/2 + **target_accept=0.98**
# （`evidence/nested_topology/m4_holdout2.py::BUDGET`，预注册口径）；
# `bm_select.BUDGETS["standard"]` 是 §9.4.4 的 2000/1000/2 + 0.9（A2 档）。
# 两者同名但不同，这里显式分开，避免接口悄悄用了另一套口径。
VALIDATION_STANDARD = dict(n_warmup=2000, n_samples=1000, num_chains=2,
                           target_accept=0.98, max_tree_depth=10)


# ---------------------------------------------------------------------------
# 口径工具
# ---------------------------------------------------------------------------

def budget_kw(name: str) -> dict:
    """`standard` = §13/§14 验收口径；`quick`/`light` = `bm_select.BUDGETS` 档。

    `BUDGETS` 里还有 `progress` / `extra_fields`，`fit_top` 不接受，
    这里显式挑字段，避免把跨族选择器的配置悄悄带进单族拟合。
    """
    if name == "standard":
        return dict(VALIDATION_STANDARD)
    if name not in BUDGETS:
        raise ValueError(f"未知 budget {name!r}，可选 {sorted(BUDGETS) + ['standard']}")
    b = BUDGETS[name]
    return {k: b[k] for k in FIT_KEYS if k in b}


def make_sigma(Z: np.ndarray, mode: str, pct: float, abs_val: float | None):
    """返回 `(sigma_for_fit_top, sigma_from_data, 说明)`。

    `perpoint` 必须是 `(2, n_freq)` 数组；`(n_freq,)` 会被 `_check_sigma`
    明确拒绝（防与标量对混淆）。
    """
    if mode == "perpoint":
        s = pct / 100.0 * np.abs(Z)
        if not np.all(s > 0):
            raise ValueError("逐点 σ 含非正值，请检查 |Z|")
        return np.stack([s, s]), True, f"逐点 σ_j={pct:g}%·|Z_j|"
    if mode == "free":
        return None, False, "σ 由后验采样（项目默认）"
    if mode == "scalar":
        v = float(abs_val) if abs_val is not None else 0.01 * float(np.abs(Z).max())
        if v <= 0:
            raise ValueError("绝对 σ 必须为正")
        return (v, v), False, f"固定绝对 σ={v:g} Ω（R16：高频可能被判成噪声内）"
    raise ValueError(f"未知 sigma 模式 {mode!r}")


def metrics(f: np.ndarray, Z: np.ndarray, Zm: np.ndarray) -> dict:
    """§14 `_metrics` 同口径；额外给端点与高频段（只报，不判决）。"""
    w = np.abs(Z)
    res = Zm - Z
    d = res / w
    rel = np.abs(res) / w
    bands = []
    for lo, hi in ((0.01, 1.0), (1.0, 100.0), (100.0, 1e5)):
        m = (f >= lo) & (f < hi)
        bands.append(float(np.sqrt(np.mean(np.abs(d[m]) ** 2))) * 100
                     if m.any() else None)
    i_top = int(np.argmax(f))
    return dict(
        rms_rel=float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2))) * 100,
        rms_re=float(np.sqrt(np.mean(res.real ** 2))),
        rms_im=float(np.sqrt(np.mean(res.imag ** 2))),
        rel_median=float(np.median(rel)) * 100,
        rel_max=float(np.max(rel)) * 100,
        rel_max_at_Hz=float(f[int(np.argmax(rel))]),
        bands_pct=bands,
        acf=float(max(E.acf_max(res.real), E.acf_max(res.imag))),
        acf_re=float(E.acf_max(res.real)), acf_im=float(E.acf_max(res.imag)),
        acf_mag=float(E.acf_max(rel)),
        top_hz=float(f[i_top]), top_rel_pct=float(rel[i_top]) * 100)


def param_table(o: dict, branch: str, layers: int) -> tuple[dict, dict]:
    """后验 → `{name: {median, ci90, rhat, ess}}` + `z_nested` 要的物理参数。"""
    keys = list(T.sampled_keys(branch, layers))
    s = o["s"]
    table, p = {}, {}
    for k in keys:
        a = np.asarray(s[k], float).reshape(-1)
        med = float(np.median(a))
        p[k] = med
        table[k] = dict(median=med,
                        ci90=[float(np.percentile(a, 5)), float(np.percentile(a, 95))],
                        rhat=o.get("rhat_by_param", {}).get(k),
                        ess=o.get("ess_by_param", {}).get(k))
    if branch == "series":
        # L_ad 是派生量（R_L·tau_L），不在 sampled_keys 里，但 z_nested 要用。
        if "L_ad" in s:
            a = np.asarray(s["L_ad"], float).reshape(-1)
            p["L_ad"] = float(np.median(a))
            table["L_ad"] = dict(median=p["L_ad"],
                                 ci90=[float(np.percentile(a, 5)),
                                       float(np.percentile(a, 95))],
                                 rhat=o.get("rhat_by_param", {}).get("L_ad"),
                                 ess=None, derived=True)
        else:
            p["L_ad"] = p["R_L"] * p["tau_L"]
    elif layers == 1:
        # 单层没有第二个并联节点，z_nested 会自己跳过 R2/Q2/n2。
        pass
    return table, p


def derived_quantities(branch: str, p: dict) -> dict:
    """`R∥L` 支路的电感与转折频率（与 §9.4.19/§9.4.27.4 同定义）。"""
    if branch == "parallel":
        R_ad, tau = p["R_ad"], p["tau_ad"]
        return dict(L_ad_H=float(R_ad * tau),
                    f_ad_Hz=float(1.0 / (2 * np.pi * tau)))
    L = p.get("L_ad")
    return dict(L_ad_H=float(L) if L is not None else None,
                f_ad_Hz=(float(p["R_L"] / (2 * np.pi * L)) if L else None))


def voigt_reference(f, Z, sigma_top, from_data, kw: dict, seed: int,
                    sigma_desc: str, K: int = 1) -> dict:
    """同口径 voigt 基线（只作参照，不是跨族判决；见 R17）。"""
    import bm_core as C
    if isinstance(sigma_top, np.ndarray):
        sg = (sigma_top[0], sigma_top[1])
    else:
        sg = sigma_top
    sv, _mc = C.fit(f, Z, K=K, alpha=0.5, sigma=sg, zeta_fixed=1.0,
                    seed=seed, progress=False, **kw)
    pv = dict(R_inf=float(np.median(np.asarray(sv["R_inf"], float))),
              W=float(np.median(np.asarray(sv["W"], float))),
              pi=np.median(np.asarray(sv["pi"], float), axis=0),
              tau=np.median(np.asarray(sv["tau"], float), axis=0),
              phi=np.median(np.asarray(sv["phi"], float), axis=0),
              zeta=np.median(np.asarray(sv["zeta"], float), axis=0))
    Zm = C.model_z(f, pv, include_L=False)
    m = metrics(f, Z, Zm)
    m.update(K=K, sigma=sigma_desc, sigma_from_data=bool(from_data),
             note="同口径参考曲线，不是跨族判决（R17）；两族 elpd 不可直接比")
    # 曲线只用于 CSV/图，调用方 `pop` 取走；不进 JSON（复数数组不可序列化）。
    m["curve"] = Zm
    return m


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def write_curve(path: str, f, Z, Zm, Zv=None) -> None:
    cols = ["f_Hz", "Zre_obs", "Zim_obs", "Zre_fit", "Zim_fit",
            "absZ_obs", "absZ_fit", "rel_err"]
    data = [f, Z.real, Z.imag, Zm.real, Zm.imag, np.abs(Z), np.abs(Zm),
            np.abs(Zm - Z) / np.abs(Z)]
    if Zv is not None:
        cols += ["Zre_voigt", "Zim_voigt"]
        data += [Zv.real, Zv.imag]
    head = ",".join(cols)
    arr = np.column_stack(data)
    np.savetxt(path, arr, delimiter=",", header=head, comments="",
               fmt="%.10g")


def write_plot(path: str, f, Z, Zm, m: dict, Zv=None, tag: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cn = any(x.name in ("Microsoft YaHei", "SimHei")
             for x in font_manager.fontManager.ttflist)
    if cn:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    L = (lambda a, b: a) if cn else (lambda a, b: b)
    c_data, c_n, c_v = "#1f2d3d", "#c0392b", "#2e86c1"

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
    a = ax[0]
    a.plot(Z.real, -Z.imag, "o", ms=5, mfc="none", mec=c_data, mew=1.2,
           label=L("实测", "data"))
    a.plot(Zm.real, -Zm.imag, "-", lw=1.9, color=c_n,
           label=f"nested ({m['rms_rel']:.1f}%)")
    if Zv is not None:
        a.plot(Zv.real, -Zv.imag, "--", lw=1.6, color=c_v, label="voigt")
    a.axhline(0, color="0.6", lw=0.8, zorder=0)
    a.set_xlabel(L("Z′ / Ω", "Z' / Ohm"))
    a.set_ylabel(L("−Z″ / Ω", "-Z'' / Ohm"))
    a.set_title(L("① Nyquist", "(1) Nyquist"))
    a.legend(fontsize=9)
    a.grid(alpha=0.25)

    a = ax[1]
    a.loglog(f, np.abs(Z), "o", ms=5, mfc="none", mec=c_data, mew=1.2,
             label=L("实测", "data"))
    a.loglog(f, np.abs(Zm), "-", lw=1.9, color=c_n, label="nested")
    if Zv is not None:
        a.loglog(f, np.abs(Zv), "--", lw=1.6, color=c_v, label="voigt")
    a.set_xlabel(L("频率 / Hz", "f / Hz"))
    a.set_ylabel(L("|Z| / Ω", "|Z| / Ohm"))
    a.set_title(L("② Bode 幅值", "(2) Bode magnitude"))
    a.legend(fontsize=9)
    a.grid(alpha=0.25, which="both")

    a = ax[2]
    a.semilogx(f, np.abs(Zm - Z) / np.abs(Z), "-o", ms=3.5, lw=1.5,
               color=c_n, label=f"nested rms {m['rms_rel']:.1f}%")
    a.axhline(0.10, color="#7f8c8d", ls=":", lw=1.2)
    a.set_xlabel(L("频率 / Hz", "f / Hz"))
    a.set_ylabel(L("|ΔZ| / |Z|", "|dZ| / |Z|"))
    a.set_yscale("log")
    a.set_title(L("③ 相对残差", "(3) relative residual"))
    a.legend(fontsize=9)
    a.grid(alpha=0.25, which="both")

    fig.suptitle(L(f"{tag}  nested 拟合（研究性，未过 §13/§14 验收线）",
                   f"{tag}  nested fit (exploratory; validation NOT passed)"),
                 fontsize=12)
    fig.text(0.5, 0.015,
             f"rms_rel={m['rms_rel']:.2f}%   acf={m['acf']:.3f}   "
             f"acf_mag={m['acf_mag']:.3f}   1e5Hz endpoint={m['top_rel_pct']:.1f}%",
             ha="center", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------

def run(input_path: str | os.PathLike, *, outdir: str = DEFAULT_OUT,
        branch: str = "parallel", layers: int = 2, budget: str = "standard",
        sigma_mode: str = "perpoint", sigma_pct: float = 2.0,
        sigma_abs: float | None = None, seed: int = 0,
        target_accept: float | None = None,
        plot: bool = True, with_voigt: bool = False, voigt_K: int = 1,
        progress: bool = False, verbose: bool = True) -> dict:
    """拟合一条谱，写产物，返回结果 dict（Python API 入口）。"""
    src = Path(input_path)
    f, Z = R.load_spectrum(str(src))
    if f.size < 8:
        raise ValueError(f"频率点太少（{f.size}），`fit_top` 要求 ≥8 点")
    order = np.argsort(f)
    f, Z = f[order], Z[order]

    kw = budget_kw(budget)
    if target_accept is not None:
        kw["target_accept"] = float(target_accept)
    sigma_top, from_data, sigma_desc = make_sigma(Z, sigma_mode, sigma_pct, sigma_abs)
    t0 = time.time()
    o = T.fit_top(f, Z, branch=branch, layers=layers, seed=seed,
                  sigma=sigma_top, sigma_from_data=from_data,
                  progress=progress, **kw)
    table, p = param_table(o, branch, layers)
    Zm = T.z_nested(f, p, branch=branch, layers=layers)
    m = metrics(f, Z, Zm)

    notes = [WARNING]
    if from_data:
        notes.append(o.get("elpd_note") or
                     "逐点 σ 来自同一批观测：elpd_loo 不是标准留一预测，不可用于模型比较")
    if o.get("rhat_max") is not None and o["rhat_max"] > 1.05:
        notes.append(f"R̂_max={o['rhat_max']:.3f} >1.05：本次采样未收敛，"
                     f"建议换 standard 档或换 seed 重跑")
    if (o.get("divergences") or 0) > 0:
        notes.append(f"divergences={o['divergences']}：存在发散转移，参数区间需谨慎")
    if m["acf"] > 0.50:
        notes.append(f"acf={m['acf']:.3f} >0.50（§14 M-b 的参考上限）："
                     f"曲线贴合不代表残差无结构")

    Zv, vref = None, None
    if with_voigt:
        try:
            vref = voigt_reference(f, Z, sigma_top, from_data, kw, seed,
                                   sigma_desc, K=voigt_K)
            Zv = vref.pop("curve", None)              # 复数曲线不进 JSON
        except Exception as exc:                       # 对照失败不影响主结果
            notes.append(f"voigt 对照失败（不影响 nested 结果）：{type(exc).__name__}: {exc}")

    os.makedirs(outdir, exist_ok=True)
    stem = src.stem
    json_path = os.path.join(outdir, stem + ".nested.json")
    csv_path = os.path.join(outdir, stem + ".nested_curve.csv")
    png_path = os.path.join(outdir, stem + ".nested.png")

    write_curve(csv_path, f, Z, Zm, Zv)
    if plot:
        write_plot(png_path, f, Z, Zm, m, Zv, tag=stem)

    result = dict(
        status="exploratory", warning=WARNING, notes=notes,
        input=dict(path=str(src.resolve()), n_points=int(f.size),
                   f_min=float(f.min()), f_max=float(f.max()),
                   absZ_max=float(np.abs(Z).max())),
        config=dict(branch=branch, layers=layers, budget=budget, budget_kw=kw,
                    sigma_mode=sigma_mode, sigma_desc=sigma_desc,
                    sigma_from_data=bool(from_data), sigma_pct=sigma_pct,
                    sigma_abs=sigma_abs, seed=seed,
                    target_accept_override=target_accept,
                    with_voigt=with_voigt,
                    voigt_K=voigt_K if with_voigt else None),
        params=table, derived=derived_quantities(branch, p),
        metrics=m,
        diagnostics=dict(rhat_max=o.get("rhat_max"),
                         rhat_by_param=o.get("rhat_by_param"),
                         ess_min=o.get("ess_min"),
                         divergences=o.get("divergences"),
                         sigma_mode=o.get("sigma_mode"),
                         elpd=float(o["elpd"]), elpd_se=float(o["elpd_se"]),
                         elpd_valid_for_predictive_comparison=bool(
                             o["elpd_valid_for_predictive_comparison"]),
                         elpd_note=o.get("elpd_note")),
        voigt_reference=vref,
        files=dict(json=json_path, curve_csv=csv_path,
                   plot=png_path if plot else None),
        elapsed_s=round(time.time() - t0, 1))
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)

    if verbose:
        print(f"\n=== {stem} ===")
        print(f"  {f.size} 点  {f.min():.3g}–{f.max():.3g} Hz  |Z|max={np.abs(Z).max():.3g} Ω")
        print(f"  {branch}/L={layers}  {budget}  {sigma_desc}  seed={seed}  "
              f"{result['elapsed_s']}s")
        print(f"  rms_rel {m['rms_rel']:.2f}%  acf {m['acf']:.3f} "
              f"(re/im {m['acf_re']:.3f}/{m['acf_im']:.3f})  "
              f"R̂ {o.get('rhat_max')}  div {o.get('divergences')}")
        print(f"  低/中/高频残差 {m['bands_pct']}  最高频端点 {m['top_rel_pct']:.1f}%")
        for k, v in table.items():
            if v.get("derived"):
                continue
            print(f"    {k:<8} {v['median']:>12.5g}   90% [{v['ci90'][0]:.5g}, "
                  f"{v['ci90'][1]:.5g}]   R̂ {v['rhat']}")
        if vref:
            print(f"  voigt K={voigt_K} 参考：rms {vref['rms_rel']:.2f}%  "
                  f"acf {vref['acf']:.3f}（仅同口径参考，非跨族判决）")
        for n in notes:
            print(f"  [!] {n}")
        print(f"  产物：{json_path}\n        {csv_path}"
              + (f"\n        {png_path}" if plot else ""))
    return result


def collect_inputs(args: list[str]) -> list[str]:
    if args:
        out = []
        for a in args:
            p = Path(a)
            if p.is_dir():
                out += [str(x) for x in sorted(p.iterdir())
                        if x.suffix.lower() in SUFFIXES]
            else:
                out.append(a)
        return out
    inbox = _ROOT / "inbox"
    if inbox.is_dir():
        return [str(x) for x in sorted(inbox.iterdir())
                if x.suffix.lower() in SUFFIXES and not x.name.startswith(".")]
    return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="nested 拓扑独立拟合接口（exploratory，不改主流程）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：python nested_fit.py data.csv --budget quick --with-voigt")
    ap.add_argument("inputs", nargs="*", help="谱文件/目录；不给则扫 inbox/")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出目录")
    ap.add_argument("--branch", choices=("parallel", "series"), default="parallel")
    ap.add_argument("--layers", type=int, choices=(1, 2), default=2)
    ap.add_argument("--budget", choices=sorted(BUDGETS), default="standard",
                    help="standard 与 §13/§14 同口径；quick 快但 R̂ 常 >1.05（只作预览）")
    ap.add_argument("--sigma", choices=("perpoint", "free", "scalar"),
                    default="perpoint", help="噪声口径（默认逐点 2%%·|Z|）")
    ap.add_argument("--sigma-pct", type=float, default=2.0,
                    help="perpoint 模式的百分比（默认 2.0）")
    ap.add_argument("--sigma-abs", type=float, default=None,
                    help="scalar 模式的绝对 σ（Ω）；不给则取 1%%·|Z|max")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-accept", type=float, default=None,
                    help="覆盖预算档的 target_accept（默认 standard=0.98）")
    ap.add_argument("--with-voigt", action="store_true",
                    help="同口径跑 voigt 基线作参照（不是跨族判决，R17）")
    ap.add_argument("--voigt-K", type=int, default=1)
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    ap.add_argument("--progress", action="store_true", help="显示 NUTS 进度条")
    a = ap.parse_args(argv)

    print(WARNING)
    files = collect_inputs(a.inputs)
    if not files:
        print("没有输入谱：请给文件路径，或把文件放进 inbox/。", file=sys.stderr)
        return 2
    print(f"共 {len(files)} 条谱；输出目录 {a.out}\n", flush=True)

    ok, bad = 0, []
    for path in files:
        try:
            run(path, outdir=a.out, branch=a.branch, layers=a.layers,
                budget=a.budget, sigma_mode=a.sigma, sigma_pct=a.sigma_pct,
                sigma_abs=a.sigma_abs, seed=a.seed,
                target_accept=a.target_accept, plot=a.plot,
                with_voigt=a.with_voigt, voigt_K=a.voigt_K,
                progress=a.progress)
            ok += 1
        except Exception as exc:
            bad.append((path, f"{type(exc).__name__}: {exc}"))
            print(f"  [失败] {path}: {type(exc).__name__}: {exc}", flush=True)
    print(f"\n完成 {ok}/{len(files)}")
    for p, e in bad:
        print(f"  失败：{p}  {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
