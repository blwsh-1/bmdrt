#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""nested_fit.py —— nested 拓扑的研究接口（**`bmdrt nested` 的实现体**）。

用途
----
给用户拿自己的谱直接跑 `bm_top.fit_top`：一条命令出参数（含 90% 区间）、
拟合曲线 CSV、Nyquist/Bode/残差三联图、诊断 JSON。也可以当 Python API 用：

    import nested_fit
    r = nested_fit.run("inbox/my_spectrum.csv", budget="standard")

两种调用方式**逐位等价**（同一个 `main()`、同一份参数表）：

    bmdrt.bat nested data.csv          # 2026-11-16 起推荐，唯一的 Windows 入口
    python nested_fit.py data.csv      # 直接跑脚本，等价

**这个接口不改 `bm_core.py` / `bm_top.py` / `bm_select.py` 的任何一行**，
不写门禁、不做跨族判决、不改 `bmdrt auto` 的默认族（仍是 voigt）。
它是子命令而不是生产族：产物恒带 `admission.state="provisional"`。

状态（必须随结果一起看）
------------------------
`exploratory` + **临时准入（provisional）**。nested 在 `M6_VALIDATION_PLAN.md`
§13 留出验收 **0/3**、§14 全量留出 **4/14**，两次都卡在 M-b 残差自相关；
§17 又证四个文献电路（含 C2/C3 更多自由度）在 10 条未过谱上 M-b 全 0/10。
而且 M-b 这个判据本身被 M-B-SIM-01 证伪（拟合后 acf AUC=0.52）——
**未过 ≠ 模型差，过 ≠ 模型对**（R27）。
所以本接口的输出**不是验收结论**，参数不得当物理量/真值直接引用，
`acf` 提示只是把 §14 的口径摆出来供参考，不构成「通过/不通过」。
准入差距与解锁条件写死在 `admission.py` 与 `PROVISIONAL_ADMISSION.md`。

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
· `branch=capacitive`（2026-11-12 新增，`exploratory`）= 最内层换成纯 CPE，
  只多 `Q_b`/`n_b` 两个参数。用于 `测试数据/` 的涂层（配 `--layers 2`）与
  钝化（配 `--layers 1`）：骨架已对到 1e-15，缺的是容性元件类型。
  **不在 `bm_select.py` 的跨族候选池里**，不得据此改写 §13/§14 判定。

用法
----
    # 下面 `python` 指项目 venv D:\autoeis_env\Scripts\python.exe
    # （系统 Python 没装 numpyro/jax 会 ImportError；双击 .bat 即可）
    bmdrt.bat nested                              # 扫 inbox/ 里的谱
    bmdrt.bat nested data.csv                     # 单条
    bmdrt.bat nested a.csv b.csv --budget quick
    bmdrt.bat nested data.csv --sigma free --branch series --layers 1
    bmdrt.bat nested 涂层.csv --branch capacitive --layers 2
    bmdrt.bat nested data.csv --with-voigt --no-plot

    python nested_fit.py data.csv                 # 直接跑脚本，参数完全相同

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
import admission as AD             # noqa: E402  临时准入状态（恒 provisional，非开关）
import spectrum_metrics as SM      # noqa: E402  两入口共用的指标/σ/预算
from bm_select import BUDGETS      # noqa: E402  只读常量，不改口径

WARNING = (
    "研究性接口（exploratory）：nested 在 §13 留出 0/3、§14 全量 4/14，"
    "未过验收线；§17 四个文献电路在 10 条未过谱上 M-b 全 0/10。"
    "自 2026-11-16 起可作为 `bmdrt nested` 子命令调用（临时准入，"
    "`admission.state=provisional`、`passed=False`），但**不改 `bmdrt auto` 的"
    "默认族（仍是 voigt）**、不做跨族判决、不写门禁；"
    "参数不得当物理量/真值，输出不是验收结论。"
)

FIT_KEYS = ("n_warmup", "n_samples", "num_chains",
            "target_accept", "max_tree_depth")
SUFFIXES = (".csv", ".txt", ".tsv", ".z", ".dat")
# 输出目录默认锚在**脚本所在项目**下，不随调用者的 CWD 漂移
# （审查 P3 二轮：从项目外调用时产物曾散落到调用者目录）。
DEFAULT_OUT = str(_ROOT / "out" / "nested_fit")

# §13/§14 的 `standard` 是 2000/1000/2 + **target_accept=0.98**
# （`evidence/nested_topology/m4_holdout2.py::BUDGET`，预注册口径）；
# `bm_select.BUDGETS["standard"]` 是 §9.4.4 的 2000/1000/2 + 0.9（A2 档）。
# 两者同名但不同，这里显式分开，避免接口悄悄用了另一套口径。
#
# 2026-11-16 起数值**只从 `spectrum_metrics.STANDARD` 取**（唯一来源），
# `circuit_runner.BUDGETS["standard"]` 也引用同一个常量——两个入口的
# "standard 档"从此是同一件事，不是两份碰巧相同的字面量。
VALIDATION_STANDARD = dict(SM.STANDARD)


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

    `perpoint` 的数值由 `spectrum_metrics.sigma_perpoint` 生成（**唯一实现**，
    与 `circuit_runner` 同一份）。返回形状是 `(2, n_freq)` 的二维数组而不是
    `(arr, arr)` tuple：`_check_sigma` 两种写法等价，但数组能避免调用方各自
    决定容器类型，差分测试里不会因为容器不同被误报。

    `free` / `scalar` 两支不走那个函数（口径不同，不是"逐点相对噪声"），
    但 `sigma_from_data` 的语义统一由 `spectrum_metrics.elpd_validity` 判定。
    """
    if mode == "perpoint":
        return SM.sigma_perpoint(Z, pct), True, f"逐点 σ_j={pct:g}%·|Z_j|"
    if mode == "free":
        return None, False, "σ 由后验采样（项目默认）"
    if mode == "scalar":
        v = float(abs_val) if abs_val is not None else 0.01 * float(np.abs(Z).max())
        if v <= 0:
            raise ValueError("绝对 σ 必须为正")
        return (v, v), False, f"固定绝对 σ={v:g} Ω（R16：高频可能被判成噪声内）"
    raise ValueError(f"未知 sigma 模式 {mode!r}")


def metrics(f: np.ndarray, Z: np.ndarray, Zm: np.ndarray) -> dict:
    """委托 `spectrum_metrics.metrics`（**唯一实现**，两入口共用）。

    本函数原来自己算 13 个字段，`circuit_runner.metrics` 另算 5 个，公式相同
    但字段集合不同。2026-11-16 合并成超集，**本接口原有的 13 个字段一个不少、
    公式与频段边界逐字未变**（逐字段回归见 `test_entry_parity.py`）。
    """
    return SM.metrics(f, Z, Zm)


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
    if branch == "capacitive":
        # 容性支路没有电感，别硬套感抗公式（会静默给出无意义的 L_ad_H=None）。
        return dict(branch_kind="capacitive", L_ad_H=None, f_ad_Hz=None,
                    Q_b=float(p["Q_b"]), n_b=float(p["n_b"]))
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


def write_plot(path: str, f, Z, Zm, m: dict, Zv=None, m_v: dict | None = None,
               tag: str = "", sigma_re=None, sigma_im=None) -> None:
    """委托 `bm_figure_unified.plot_fit`（**唯一 2×3 实现**）。

    2026-11-16 之前这里是自己画 1×3（Nyquist / Bode 幅值 / 相对残差），
    `bmdrt compare` 用的却是 2×3，两套图不一致。现在统一：nested 也走
    `plot_fit`，六个子图（+Bode 相位 +Re/Im 残差）与 `compare` 完全同构。
    `sigma_re`/`sigma_im` 让残差子图带上 ±σ 点线带（voigt 风格）。
    """
    from bm_figure_unified import plot_fit

    curves = [dict(label="nested", Zc=Zm, color="#c0392b",
                   rms=m["rms_rel"])]
    if Zv is not None:
        curves.append(dict(label="voigt", Zc=Zv, color="#2e86c1", style="--",
                           rms=(m_v["rms_rel"] if m_v else None)))
    head = f"{tag}  nested 拟合（研究性，未过 §13/§14 验收线）" if tag \
        else "nested 拟合（研究性，未过 §13/§14 验收线）"
    meta = (f"rms_rel={m['rms_rel']:.2f}%   acf={m['acf']:.3f} "
            f"(re/im {m['acf_re']:.3f}/{m['acf_im']:.3f})   "
            f"acf_mag={m['acf_mag']:.3f}   1e5Hz 端点={m['top_rel_pct']:.1f}%")
    plot_fit(path, f, Z, curves, title=head, meta=meta,
             sigma_re=sigma_re, sigma_im=sigma_im)


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

    # ELPD 可比性：**与 `circuit_runner` 调同一个判定函数**。两层理由分开：
    # L1 perpoint 的 σ_j 含被留出点自身；L2 R21 先验尺度数据依赖（两口径都成立）。
    # 以前这里只写了 L1（`o["elpd_note"]`），于是同一条谱在 runner 的产物里写的是
    # R21、在本接口里写的是 σ 泄漏——两句话都对，但都不是完整理由。
    ev = SM.elpd_validity(likelihood=("perpoint" if sigma_mode == "perpoint"
                                      else "free"),
                          sigma_from_data=from_data)

    notes = [WARNING]
    if from_data:
        notes.append(o.get("elpd_note") or
                     "逐点 σ 来自同一批观测：elpd_loo 不是标准留一预测，不可用于模型比较")
    notes.append(f"ELPD 完整性（两层，与 circuit_runner 同一判定）：{ev['reason']}；"
                 f"comparison_scope={ev['comparison_scope']}")
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
    circ_path = os.path.join(outdir, stem + ".nested_circuit.png")

    write_curve(csv_path, f, Z, Zm, Zv)
    if plot:
        # 残差子图的 ±σ 点线带：perpoint=逐点 2%|Z|；scalar=固定标量；
        # free=后验 s_re/s_im（若采样了）。None 就不画带。
        sr = si = None
        if sigma_mode == "perpoint":
            sr = si = np.asarray(sigma_top, float)[0]
        elif sigma_mode == "scalar":
            sr = si = float(np.asarray(sigma_top, float)[0])
        elif "s_re" in (o.get("s") or {}):
            s_ = o["s"]
            sr = float(np.median(np.asarray(s_["s_re"], float)))
            si = float(np.median(np.asarray(s_["s_im"], float)))
        write_plot(png_path, f, Z, Zm, m, Zv, m_v=vref, tag=stem,
                   sigma_re=sr, sigma_im=si)
        # 并联电路图：只画 `z_nested` 这一族（`bm_circuit.draw` 只认 voigt
        # 串联骨架，画不了梯形网络）。失败只记 warning，不拦拟合结果。
        try:
            import bm_circuit_nested as CN
            CN.draw_nested(p, branch=branch, layers=layers,
                           f_ref=float(np.sqrt(f.min() * f.max())),
                           path=circ_path, name=stem,
                           admission_state=AD.STATE)
        except Exception as exc:                       # noqa: BLE001
            notes.append(f"电路图失败（不影响拟合结果）："
                         f"{type(exc).__name__}: {exc}")
            circ_path = None

    result = dict(
        status="exploratory", warning=WARNING, notes=notes,
        # 临时准入块：`state="provisional"` 且 `passed=False`，逐条列没达到的
        # 阈值与有/无证据的谱型范围。用户裁决「能跑就行、不许声称过线」，
        # 所以这块跟着**每一份**产物走，不是只印在控制台（见 `admission.py`）。
        admission=AD.block(),
        input=dict(path=str(src.resolve()), n_points=int(f.size),
                   f_min=float(f.min()), f_max=float(f.max()),
                   absZ_max=float(np.abs(Z).max())),
        config=dict(branch=branch, layers=layers, budget=budget, budget_kw=kw,
                    # 五项数值 + 档名 + 是否与 circuit_runner 共享。`quick` 两个
                    # 入口数值不同（本接口走 bm_select.QUICK 800/400/10，
                    # runner 是 300/300/8），所以不许靠档名推断"同预算"。
                    budget_actual=SM.budget_actual(kw, tier=budget),
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
                         # 这个字段与 `bm_top.fit_top` 同义：**只答 σ 泄漏**，
                         # 不含 R21。完整的可比性结论在同级的 `elpd_validity`。
                         elpd_valid_for_predictive_comparison=bool(
                             o["elpd_valid_for_predictive_comparison"]),
                         elpd_validity=ev,
                         elpd_note=o.get("elpd_note")),
        voigt_reference=vref,
        files=dict(json=json_path, curve_csv=csv_path,
                   plot=png_path if plot else None,
                   circuit=circ_path if plot else None),
        elapsed_s=round(time.time() - t0, 1))
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)

    if verbose:
        print(f"\n=== {stem} ===")
        print(f"  {AD.banner()}")
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
              + (f"\n        {png_path}\n        {circ_path}" if plot else ""))
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
    ap.add_argument("--branch", choices=("parallel", "series", "capacitive"),
                    default="parallel")
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
    print(AD.banner())
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
        except KeyboardInterrupt:
            # Ctrl-C 就是想停：如实中断，不伪装成"这条谱失败"继续跑下一条。
            raise
        except (Exception, SystemExit) as exc:
            # `bm_real.load_spectrum` 的错误分支用 `raise SystemExit` 报错
            # （如"找不到文件"/"有效数据行太少"）。SystemExit 继承
            # BaseException，`except Exception` 接不住——审查 P1-3（二轮）
            # 实测一条坏谱中断整批。与主链路 `bmdrt.py` 的约定一致：
            # 一个文件的问题，不能让整批死在这儿。
            bad.append((path, f"{type(exc).__name__}: {exc}"))
            print(f"  [失败] {path}: {type(exc).__name__}: {exc}", flush=True)
    print(f"\n完成 {ok}/{len(files)}")
    for p, e in bad:
        print(f"  失败：{p}  {e}")
    # 审查 P1-3（二轮）：部分失败必须红灯。旧口径 `0 if ok else 1` 会让
    # "9 成功 + 1 失败"退出码为 0，批处理上层无从感知。
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
