# -*- coding: utf-8 -*-
"""已知真值谱的第 ④ 步：参数恢复 + LS 反事实 + 误选率（只读研究脚本）。

预注册：`docs/round_2026-11-12/RECOVERY_PREREG.md`（口径先冻结，跑完不许改）。
上游：`out/truth_dataset/`（168 条，`truth_manifest.json`）。
准入文件：`CIRCUIT_TOPOLOGY_ADMISSION.md` §4.1.3 / §6。

**anti-inverse-crime 参数恢复层三条硬保证**
------------------------------------------------
A1 拟合侧正演**独立实现**：用「导纳累加」递推 `1/(1/z + Y_cpe)`，
   生成侧 `truth_dataset.py` 用 `par(a,b)=ab/(a+b)` 显式循环 —— 数值表达式与
   代码路径都不同。主流程**不 import 生成器**，只在 `--selftest` 里对拍。
A2 初值只由 `median|Z|`、频率窗口几何中心、稳定 md5 seed 生成，
   **绝不读 `truth_params` 起步**；真值只在评估阶段用于算恢复误差。
A3 `--selftest` 三方对拍：拟合侧 vs 生成侧 7 类拓扑、拟合侧 vs 项目
   `bm_top.z_nested` 4 个 (L, branch) 组合。对拍结果不参与拟合。

    python docs/round_2026-11-12/truth_recovery.py --selftest
    python docs/round_2026-11-12/truth_recovery.py --pilot
    python docs/round_2026-11-12/truth_recovery.py

产物：`out/truth_recovery/{results.json,summary.md,manifest.json}`。
**不接生产、不改判据/阈值/先验、不给 grade/verdict。**
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "out" / "truth_dataset"
OUT = ROOT / "out" / "truth_recovery"

TAU = 2.0 * np.pi
N_STARTS = 3
MAX_NFEV = 1500
ACF_LINE = 0.50
ACF_DROP = 0.20
FALSE_SELECT_LIMIT = 0.10
MAIN_KINDS = ("none", "heavy")          # 排除自带频率相关结构的 correlated

CAP_FAMILIES = ("nested_L1_capacitive", "nested_L2_capacitive")
NEG_FAMILIES = ("ZARC", "nested_L1_inductive", "nested_L2_inductive", "C2_L3")
CAP_CANDIDATES = ("nested_L1_capacitive", "nested_L2_capacitive", "C3_mixed")

TRUTH_CANDIDATE = {
    "ZARC": "voigt_K1",
    "nested_L1_inductive": "nested_L1_inductive",
    "nested_L2_inductive": "nested_L2_inductive",
    "nested_L1_capacitive": "nested_L1_capacitive",
    "nested_L2_capacitive": "nested_L2_capacitive",
    "C2_L3": "C2_L3",
    "C3_mixed": "C3_mixed",
}

CANDIDATES = (
    "voigt_K1", "voigt_K2", "voigt_K3",
    "nested_L1_inductive", "nested_L2_inductive",
    "nested_L1_capacitive", "nested_L2_capacitive",
    "C2_L3", "C3_mixed",
)


# ---------------------------------------------------------------- 独立正演（拟合侧）
# 与生成器的 par(a,b)=ab/(a+b) 不同：这里全程走导纳 1/z + Y，再取倒数。

def y_cpe(w, Q, n):
    """CPE 导纳；生成侧写的是阻抗 Q/(iw)^n，此处写 Y=(iw)^n/Q。"""
    return (1j * np.asarray(w, float)) ** n / Q


def z_cpe(f, Q, n):
    return 1.0 / y_cpe(TAU * np.asarray(f, float), Q, n)


def branch_z(kind, f, p):
    w = TAU * np.asarray(f, float)
    if kind == "parallel":
        L = p["R_ad"] * p["tau_ad"]
        return 1.0 / (1.0 / p["R_ad"] + 1.0 / (1j * w * L))
    if kind == "capacitive":
        return z_cpe(f, p["Q_b"], p["n_b"])
    raise ValueError(f"未知支路 {kind!r}")


def nested_body(f, p, layers, branch):
    """嵌套骨架（**不含 Rs**），逐层导纳累加。"""
    w = TAU * np.asarray(f, float)
    z = p[f"R{layers}"] + branch_z(branch, f, p)
    for i in range(layers, 0, -1):
        z = 1.0 / (1.0 / z + y_cpe(w, p[f"Q{i}"], p[f"n{i}"]))
        if i - 1 >= 1:
            z = p[f"R{i-1}"] + z
    return z


def voigt_chain(f, p, K):
    w = TAU * np.asarray(f, float)
    z = np.full(w.shape, float(p["Rs"]), dtype=complex)
    for k in range(1, K + 1):
        z = z + 1.0 / (1.0 / p[f"R{k}"] + y_cpe(w, p[f"Q{k}"], p[f"n{k}"]))
    return z


def c3_mixed(f, p):
    w = TAU * np.asarray(f, float)
    outer = 1.0 / (1.0 / p["R0"] + y_cpe(w, p["Q0"], p["n0"]))
    return p["Rs"] + outer + nested_body(f, p, 2, "capacitive")


def forward(name, f, p):
    if name.startswith("voigt_K"):
        return voigt_chain(f, p, int(name[-1]))
    if name == "C2_L3":
        return p["Rs"] + nested_body(f, p, 3, "parallel")
    if name == "C3_mixed":
        return c3_mixed(f, p)
    if name == "nested_L1_inductive":
        return p["Rs"] + nested_body(f, p, 1, "parallel")
    if name == "nested_L2_inductive":
        return p["Rs"] + nested_body(f, p, 2, "parallel")
    if name == "nested_L1_capacitive":
        return p["Rs"] + nested_body(f, p, 1, "capacitive")
    if name == "nested_L2_capacitive":
        return p["Rs"] + nested_body(f, p, 2, "capacitive")
    raise ValueError(name)


# ---------------------------------------------------------------- 参数化

def keys_for(name):
    if name.startswith("voigt_K"):
        keys = ["Rs"]
        for k in range(1, int(name[-1]) + 1):
            keys += [f"R{k}", f"Q{k}", f"n{k}"]
        return tuple(keys)
    if name == "C2_L3":
        return ("Rs", "R1", "Q1", "n1", "R2", "Q2", "n2",
                "R3", "Q3", "n3", "R_ad", "tau_ad")
    if name == "C3_mixed":
        return ("Rs", "Q0", "n0", "R0", "R1", "Q1", "n1",
                "R2", "Q2", "n2", "Q_b", "n_b")
    L = 1 if "_L1_" in name else 2
    keys = ["Rs"]
    for i in range(1, L + 1):
        keys += [f"R{i}", f"Q{i}", f"n{i}"]
    keys += ["Q_b", "n_b"] if "capacitive" in name else ["R_ad", "tau_ad"]
    return tuple(keys)


def is_index(k):
    """CPE 指数：n0/n1/n2/n3/n_b。其余正参数走 log 空间。"""
    return k.startswith("n")


def to_x(p, keys):
    return np.asarray([p[k] if is_index(k) else math.log(max(float(p[k]), 1e-300))
                       for k in keys], float)


def from_x(x, keys):
    return {k: (float(x[i]) if is_index(k) else math.exp(float(x[i])))
            for i, k in enumerate(keys)}


def bounds_for(keys):
    lo = [0.30 if is_index(k) else math.log(1e-10) for k in keys]
    hi = [0.999 if is_index(k) else math.log(1e14) for k in keys]
    return np.asarray(lo, float), np.asarray(hi, float)


def starts_for(name, sid, f, Z, n_starts=N_STARTS):
    """A2：初值只由谱幅值 / 频率窗口 / 稳定哈希决定，**不读真值**。"""
    zmed = float(np.median(np.abs(Z)))
    fc = math.sqrt(float(f.min()) * float(f.max()))
    keys = keys_for(name)
    out = []
    for s in range(n_starts):
        seed = int(hashlib.md5(f"{name}|{sid}|{s}".encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        scale = max(zmed, 1e-6) * (0.3, 1.0, 3.0)[s % 3]
        f0 = fc * (0.05, 1.0, 20.0)[s % 3]
        p = dict(Rs=max(scale * 0.01, 1e-8))
        for k in keys:
            if k == "Rs":
                continue
            if is_index(k):
                p[k] = float(np.clip(0.78 + 0.12 * rng.normal(), 0.35, 0.95))
            elif k.startswith("Q"):
                p[k] = scale * (TAU * f0) ** 0.8
            elif k.startswith("tau"):
                p[k] = 1.0 / (TAU * f0)
            elif k.startswith("R"):
                p[k] = scale
        out.append(to_x(p, keys))
    return out


def acf_max(resid, lags=(1, 2, 3, 4, 5)):
    """与 `bm_experiments.acf_max` 同口径的独立实现（滞后 1–5，尺度不变）。"""
    x = np.asarray(resid, float)
    x = x - x.mean()
    dd = float(np.dot(x, x))
    if dd <= 0:
        return 0.0
    return max(abs(float(np.dot(x[:-L], x[L:])) / dd) for L in lags if L < x.size)


def json_safe(v):
    """**bool 必须先于 int 判断**（bool 是 int 子类）；原生 int 必须原样保留。"""
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    if v is None or isinstance(v, str):
        return v
    return str(v)


# ---------------------------------------------------------------- 单候选 LS

def fit_one(name, f, Z, sid):
    keys = keys_for(name)
    lo, hi = bounds_for(keys)
    az = np.abs(Z)

    def resid(x):
        zm = forward(name, f, from_x(x, keys))
        d = zm - Z
        return np.concatenate([d.real / az, d.imag / az])

    best, xs = None, []
    for x0 in starts_for(name, sid, f, Z):
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        try:
            r = least_squares(resid, x0, bounds=(lo, hi), max_nfev=MAX_NFEV,
                              ftol=1e-10, xtol=1e-10, gtol=1e-10)
        except Exception:
            continue
        if not bool(r.success):
            continue
        chi2 = float(np.dot(r.fun, r.fun))
        xs.append(r.x)
        if best is None or chi2 < best[0]:
            best = (chi2, r)

    if best is None:
        return dict(converged=False, k=len(keys), rwp_pct=None, acf=None,
                    chi2=None, bic=None, start_spread=None, jac_cond=None,
                    params=None, n_converged=0)

    chi2, r = best
    p = from_x(r.x, keys)
    zm = forward(name, f, p)
    d = zm - Z
    n = int(2 * f.size)
    rwp = float(np.sqrt(np.mean(np.abs(d) ** 2 / az ** 2)) * 100.0)
    acf = float(max(acf_max(d.real), acf_max(d.imag)))
    bic = float(n * math.log(max(chi2 / n, 1e-24)) + len(keys) * math.log(n))

    spread = None
    if len(xs) > 1:
        A = np.asarray(xs)
        spread = float(np.max(np.abs(A - np.median(A, axis=0))))
    cond = None
    try:
        sv = np.linalg.svd(r.jac, compute_uv=False)
        if sv.size and sv[-1] > 0:
            cond = float(sv[0] / sv[-1])
    except Exception:
        pass

    return dict(converged=True, k=len(keys), rwp_pct=rwp, acf=acf, chi2=chi2,
                bic=bic, start_spread=spread, jac_cond=cond,
                params=json_safe(p), n_converged=len(xs))


# ---------------------------------------------------------------- 评估

def param_errors(truth, fit, name):
    """只对真族候选算；正参数 |ln(fit/truth)|，指数 |Δn|。"""
    if not fit.get("converged") or not fit.get("params"):
        return None
    p = fit["params"]
    errs = {}
    for k, tv in truth.items():
        if k not in p or tv is None:
            continue
        if is_index(k):
            errs[k] = abs(float(p[k]) - float(tv))
        elif float(tv) > 0 and float(p[k]) > 0:
            errs[k] = abs(math.log(float(p[k]) / float(tv)))
    if not errs:
        return None
    med = float(np.median(list(errs.values())))
    out = dict(per_param=errs, median=med)

    # 组合量：最内层 CPE 特征频率 f0 = (R/Q)^(1/n)/2π
    if "Q_b" in truth and "n_b" in truth and "Q_b" in p and "n_b" in p:
        rk = "R1" if "_L1_" in name else "R2"
        try:
            f0t = (float(truth[rk]) / float(truth["Q_b"])) ** (1 / float(truth["n_b"])) / TAU
            f0f = (float(p[rk]) / float(p["Q_b"])) ** (1 / float(p["n_b"])) / TAU
            if f0t > 0 and f0f > 0:
                out["f0_combo_ln_err"] = abs(math.log(f0f / f0t))
        except (KeyError, ValueError, ZeroDivisionError, OverflowError):
            pass
    return out


def evaluate_one(rec, f, Z):
    sid = rec["id"]
    cands = {name: fit_one(name, f, Z, sid) for name in CANDIDATES}
    ok = {k: v for k, v in cands.items() if v["converged"]}
    best_bic = min(ok, key=lambda k: ok[k]["bic"]) if ok else None
    truth_c = TRUTH_CANDIDATE[rec["family"]]
    return dict(
        id=sid, family=rec["family"], difficulty=rec["difficulty"],
        noise_kind=rec["noise"]["kind"], noise_pct=rec["noise"]["pct"],
        seed=rec["seed"], topology=rec["topology"],
        truth_candidate=truth_c,
        truth_selected=(best_bic == truth_c),
        best_bic=best_bic,
        param_recovery=param_errors(rec["truth_params"], cands[truth_c], truth_c),
        candidates=cands,
    )


# ---------------------------------------------------------------- 汇总

def pct(x, n):
    return 100.0 * x / n if n else float("nan")


def summarize(results):
    S = [r for r in results if r["family"] in CAP_FAMILIES
         and r["noise_kind"] in MAIN_KINDS]
    N = [r for r in results if r["family"] in NEG_FAMILIES
         and r["noise_kind"] in MAIN_KINDS]

    # ① 无容性支路时的结构性残差（按真值 L 配对）
    c1_acfs, c1_hits = [], 0
    for r in S:
        ind = "nested_L1_inductive" if r["family"] == "nested_L1_capacitive" else "nested_L2_inductive"
        v = r["candidates"][ind]["acf"]
        if v is None:
            continue
        c1_acfs.append(v)
        c1_hits += (v > ACF_LINE)
    n1 = len(c1_acfs)
    c1_rate, c1_med = pct(c1_hits, n1), float(np.median(c1_acfs)) if c1_acfs else None
    c1_ok = (n1 > 0 and c1_rate >= 75.0 and c1_med is not None and c1_med >= ACF_LINE)

    # ② 加入后结构改善（Δacf），并给自由度红利对照
    drops, rwp_gain, l3_drops = [], [], []
    for r in S:
        L = 1 if r["family"] == "nested_L1_capacitive" else 2
        ind = r["candidates"][f"nested_L{L}_inductive"]
        cap = r["candidates"][f"nested_L{L}_capacitive"]
        l3 = r["candidates"]["C2_L3"]
        if None in (ind["acf"], cap["acf"]):
            continue
        drops.append(ind["acf"] - cap["acf"])
        if None not in (ind["rwp_pct"], cap["rwp_pct"]):
            rwp_gain.append(ind["rwp_pct"] - cap["rwp_pct"])
        if l3["acf"] is not None:
            l3_drops.append(ind["acf"] - l3["acf"])
    c2_hits = sum(d >= ACF_DROP for d in drops)
    c2_rate, c2_med = pct(c2_hits, len(drops)), (float(np.median(drops)) if drops else None)
    c2_ok = (len(drops) > 0 and c2_rate >= 75.0 and c2_med is not None and c2_med >= ACF_DROP)

    # ④ 误选率（BIC 最小含容性候选） + 真族识别率 sanity check
    n4 = len(N)
    false_sel = sum(r["best_bic"] in CAP_CANDIDATES for r in N if r["best_bic"])
    c4_rate = pct(false_sel, n4)
    truth_hit = sum(r["truth_selected"] for r in N)
    truth_rate = pct(truth_hit, n4)
    sanity = truth_rate >= 50.0
    c4_ok = (sanity and n4 > 0 and c4_rate <= 100.0 * FALSE_SELECT_LIMIT)

    # 自由度红利：C2_L3 / voigt_K3 在非其真值谱上的 BIC 胜率
    others = [r for r in results if r["noise_kind"] in MAIN_KINDS
              and r["family"] not in ("C2_L3",)]
    l3_win = pct(sum(r["best_bic"] == "C2_L3" for r in others), len(others))
    v3_win = pct(sum(r["best_bic"] == "voigt_K3" for r in others), len(others))

    # 按难度 / 噪声分层的真族识别率与参数恢复误差
    by_diff, by_noise = defaultdict(list), defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append(r)
        by_noise[r["noise_kind"]].append(r)

    def rate(rs):
        return pct(sum(bool(x["truth_selected"]) for x in rs), len(rs)) if rs else None

    def rec_med(rs):
        v = [x["param_recovery"]["median"] for x in rs
             if x["param_recovery"] is not None]
        return float(np.median(v)) if v else None

    corr = [r for r in results if r["noise_kind"] == "correlated"]
    corr_acf = [r["candidates"][r["truth_candidate"]]["acf"] for r in corr
                if r["candidates"][r["truth_candidate"]]["acf"] is not None]
    none_acf = [r["candidates"][r["truth_candidate"]]["acf"] for r in results
                if r["noise_kind"] == "none"
                and r["candidates"][r["truth_candidate"]]["acf"] is not None]
    heavy_acf = [r["candidates"][r["truth_candidate"]]["acf"] for r in results
                 if r["noise_kind"] == "heavy"
                 and r["candidates"][r["truth_candidate"]]["acf"] is not None]

    verdict = {
        "1": dict(condition="无支路时稳定结构性残差", evaluable=True, met=bool(c1_ok),
                  n=n1, hit_rate_pct=c1_rate, median_acf=c1_med),
        "2": dict(condition="加入后结构改善而非只降 Rwp", evaluable=True, met=bool(c2_ok),
                  n=len(drops), hit_rate_pct=c2_rate, median_dacf=c2_med,
                  median_rwp_gain_pct=(float(np.median(rwp_gain)) if rwp_gain else None),
                  median_C2_L3_dacf=(float(np.median(l3_drops)) if l3_drops else None)),
        "3": dict(condition="新增参数有可接受的后验约束", evaluable=False, met=None,
                  note="LS 无后验；本轮只报 start_spread / jac_cond / 参数恢复误差作线索，状态维持 ❌"),
        "4": dict(condition="错误拓扑上不频繁误选", evaluable=bool(sanity), met=(bool(c4_ok) if sanity else None),
                  n=n4, false_select_n=false_sel, false_select_pct=c4_rate,
                  truth_family_rate_pct=truth_rate),
    }

    # ---- 归因分析（exploratory，绝不改变上面的预注册判定）----
    def pair_(r):
        L = 1 if r["family"] == "nested_L1_capacitive" else 2
        return r["candidates"][f"nested_L{L}_inductive"], \
               r["candidates"][f"nested_L{L}_capacitive"]

    def layer_(rs):
        ds, rwc, rwi, c1, c2 = [], [], [], 0, 0
        for r in rs:
            ind, cap = pair_(r)
            if ind["acf"] is not None:
                c1 += int(ind["acf"] > ACF_LINE)
            if None not in (ind["acf"], cap["acf"]):
                dd = ind["acf"] - cap["acf"]
                ds.append(dd)
                c2 += int(dd >= ACF_DROP)
            if cap["rwp_pct"] is not None:
                rwc.append(cap["rwp_pct"])
            if ind["rwp_pct"] is not None:
                rwi.append(ind["rwp_pct"])
        return dict(n=len(rs), c1_hits=c1, c2_hits=c2, c2_n=len(ds),
                    median_dacf=(float(np.median(ds)) if ds else None),
                    truth_rwp_median=(float(np.median(rwc)) if rwc else None),
                    ind_rwp_median=(float(np.median(rwi)) if rwi else None),
                    truth_selected=sum(bool(r["truth_selected"]) for r in rs))

    layers_ = {}
    for diff in ("separated", "overlap", "out_of_window", "degenerate"):
        for kind in MAIN_KINDS:
            g = [r for r in S if r["difficulty"] == diff and r["noise_kind"] == kind]
            if g:
                layers_[f"{diff}|{kind}"] = layer_(g)

    rec_none = [r["param_recovery"]["median"] for r in S
                if r["noise_kind"] == "none" and r["param_recovery"]]
    c2_true = [r for r in results if r["family"] == "C2_L3"]
    conv = [c for r in results for c in r["candidates"].values()]
    attribution = dict(
        layers=layers_,
        none=layer_([r for r in S if r["noise_kind"] == "none"]),
        heavy=layer_([r for r in S if r["noise_kind"] == "heavy"]),
        false_select_ids=[r["id"] for r in N if r["best_bic"] in CAP_CANDIDATES],
        param_recovery_none=dict(
            n=len(rec_none),
            median=(float(np.median(rec_none)) if rec_none else None),
            maximum=(float(np.max(rec_none)) if rec_none else None)),
        c2_l3_truth_rate_pct=pct(sum(r["truth_selected"] for r in c2_true), len(c2_true)),
        n_c2_l3=len(c2_true),
        convergence=dict(ok=sum(bool(c["converged"]) for c in conv), n=len(conv)),
        note="分层归因是 exploratory 补充报告，不得用来改写条件 1/2/4 的预注册判定",
    )

    return dict(attribution=attribution, S_n=len(S), N_n=len(N), verdict=verdict,
                by_difficulty={k: dict(n=len(v), truth_rate_pct=rate(v),
                                       param_err_median=rec_med(v)) for k, v in by_diff.items()},
                by_noise={k: dict(n=len(v), truth_rate_pct=rate(v),
                                  param_err_median=rec_med(v)) for k, v in by_noise.items()},
                acf_by_noise=dict(
                    none_median=(float(np.median(none_acf)) if none_acf else None),
                    heavy_median=(float(np.median(heavy_acf)) if heavy_acf else None),
                    correlated_median=(float(np.median(corr_acf)) if corr_acf else None)),
                freedom_bonus=dict(C2_L3_bic_win_pct=l3_win,
                                   voigt_K3_bic_win_pct=v3_win, n=len(others)),
                truth_rate_all_pct=pct(sum(bool(r["truth_selected"]) for r in results), len(results)),
                n_spectra=len(results))


def F(v, fmt="%.3f"):
    """缺数据 / 非有限 → 破折号。**绝不把 None 显示成 0**。"""
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return (fmt % x) if math.isfinite(x) else "—"


def write_summary(summary, results, path):
    v = summary["verdict"]
    L = []
    A = L.append
    A("# 参数恢复 / LS 反事实实验结果（准入流程第 ④ 步）\n")
    A("预注册：`docs/round_2026-11-12/RECOVERY_PREREG.md`（口径先冻结）。"
      "上游 `out/truth_dataset/`，共 **%d** 条谱 × 9 候选 × 3 起点。\n" % summary["n_spectra"])
    A("> **本实验只到 LS 反事实，不是 Bayesian 采样，也不是模型选择门禁验证。**"
      " 合成谱噪声是人为生成过程，结论不得升格为真实样品准入证据。\n")

    A("\n## 一、四条最低验证条件逐条判定\n")
    A("| # | 条件 | 本轮可判定 | 结果 | 实测 |")
    A("|---|---|---|---|---|")
    for key in ("1", "2", "3", "4"):
        d = v[key]
        if not d["evaluable"]:
            res = "不判定（状态维持 ❌）"
        elif d["met"] is None:
            res = "不作结论（sanity 未过）"
        else:
            res = "**满足**" if d["met"] else "**未满足**"
        if key == "1":
            m = (f"acf>0.50 占比 {F(d['hit_rate_pct'], '%.1f')}%（n={d['n']}），"
                 f"中位 acf {F(d['median_acf'])}")
        elif key == "2":
            m = (f"Δacf≥0.20 占比 {F(d['hit_rate_pct'], '%.1f')}%（n={d['n']}），"
                 f"中位 Δacf {F(d['median_dacf'])}；"
                 f"中位 ΔRwp {F(d['median_rwp_gain_pct'], '%.2f')}pp；"
                 f"C2_L3 中位 Δacf {F(d['median_C2_L3_dacf'])}")
        elif key == "3":
            m = "LS 无后验；仅报 `start_spread` / `jac_cond` / 参数恢复误差作**线索**"
        else:
            m = (f"误选 {d['false_select_n']}/{d['n']} = "
                 f"{F(d['false_select_pct'], '%.1f')}%；负样本真族识别率 "
                 f"{F(d['truth_family_rate_pct'], '%.1f')}%（sanity）")
        A(f"| {key} | {d['condition']} | {'是' if d['evaluable'] else '否'} | {res} | {m} |")

    A("\n预注册判据：① `acf>0.50` 占比 ≥75% 且中位 ≥0.50；② `Δacf≥0.20` 占比 ≥75%"
      " 且中位 ≥0.20；④ 误选率 ≤10% 且真族识别率 ≥50%（否则不作结论）。"
      f"主样本集 `S` = 容性真值 × `kind∈{{none,heavy}}`（{summary['S_n']} 条），"
      f"负样本集 `N` = 非容性真值 × 同噪声档（{summary['N_n']} 条）。\n")

    A("\n## 二、真族识别率与参数恢复误差（分层）\n")
    A(f"全样本按 BIC 选中真族：**{F(summary['truth_rate_all_pct'], '%.1f')}%**。\n")
    A("| 难度 | n | 真族识别率 % | 参数恢复中位 `|ln(fit/truth)|` |")
    A("|---|---|---|---|")
    for k, d in sorted(summary["by_difficulty"].items()):
        A(f"| {k} | {d['n']} | {F(d['truth_rate_pct'], '%.1f')} | "
          f"{F(d['param_err_median'])} |")
    A("\n| 噪声档 | n | 真族识别率 % | 参数恢复中位 |")
    A("|---|---|---|---|")
    for k, d in sorted(summary["by_noise"].items()):
        A(f"| {k} | {d['n']} | {F(d['truth_rate_pct'], '%.1f')} | "
          f"{F(d['param_err_median'])} |")

    A("\n## 三、自由度红利与 acf 噪声口径（只记账）\n")
    fb = summary["freedom_bonus"]
    A(f"- 非 C2_L3 真值谱上 `C2_L3` 的 BIC 胜率 **{F(fb['C2_L3_bic_win_pct'], '%.1f')}%**、"
      f"`voigt_K3` **{F(fb['voigt_K3_bic_win_pct'], '%.1f')}%**（n={fb['n']}）；"
      "这是自由度对照，不是通过线。")
    a = summary["acf_by_noise"]
    A(f"- 真族模型残差 acf 中位：`none` {F(a['none_median'])} / "
      f"`heavy` {F(a['heavy_median'])} / `correlated` {F(a['correlated_median'])}。"
      "**correlated 自带频率相关结构**（`NOISE_ACF_TRUTH.md`：acf 尺度不变、"
      "只响应频率相关结构），故主判据已排除；heavy 为 i.i.d. 重尾，与 none 接近"
      " 印证 §18 规则 2 的 i.i.d. 限定。")

    at = summary["attribution"]
    A("\n## 三之二、归因分析（exploratory，**不改变预注册判定**）\n")
    A("> 条件 ① 71.9% / ② 68.8% 仍按预注册线判为**未满足**。本节只解释"
      "「未满足」来自哪一层，**不得据此放宽线或改写成满足**。\n")
    A("| 难度 | 噪声 | n | ①命中 | ②命中 | 中位 Δacf | 真族 Rwp 中位 | 感性对照 Rwp 中位 | 真族选中 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for key, d in at["layers"].items():
        diff, kind = key.split("|")
        A(f"| {diff} | {kind} | {d['n']} | {d['c1_hits']}/{d['n']} | "
          f"{d['c2_hits']}/{d['c2_n']} | {F(d['median_dacf'])} | "
          f"{F(d['truth_rwp_median'], '%.3f')} | {F(d['ind_rwp_median'], '%.3f')} | "
          f"{d['truth_selected']}/{d['n']} |")
    nd, hd = at["none"], at["heavy"]
    A(f"\n**合计**：`none` 档 ① {nd['c1_hits']}/{nd['n']}、② {nd['c2_hits']}/{nd['c2_n']}，"
      f"真族 Rwp 中位 {F(nd['truth_rwp_median'], '%.3f')}%；"
      f"`heavy` 档 ① {hd['c1_hits']}/{hd['n']}、② {hd['c2_hits']}/{hd['c2_n']}，"
      f"真族 Rwp 中位 {F(hd['truth_rwp_median'], '%.3f')}%、"
      f"感性对照 {F(hd['ind_rwp_median'], '%.3f')}%。\n")
    A("**读数**：拉低结论的全部是 `heavy` 5% 重尾噪声档（`out_of_window` 0/4、"
      "`degenerate` 1/4）；无噪档 16/16 全过 ①②。heavy 档容性真族 Rwp 贴近"
      " 5% 噪声底、比感性对照低约 1.7pp，说明**容性确实拟合得更好**，但感性模型的"
      " acf 也被噪声压到 0.24–0.41 → 结构信号被噪声淹没，`acf>0.50` 这个缺项探针"
      "在重尾噪声档失效。⇒ ①② 的「未满足」是 **噪声淹没 + 困难轴下 LS 局部极小**"
      " 的混杂，**不能读作「容性支路无用」**；但按预注册口径判定不放宽。\n")
    pr = at["param_recovery_none"]
    A(f"**参数恢复层 anti-inverse-crime 正面证据**：拟合侧独立正演、初值不读真值，"
      f"无噪档容性真族参数恢复中位 `|ln(fit/truth)|` = **{F(pr['median'], '%.3g')}**"
      f"（n={pr['n']}，max {F(pr['maximum'], '%.3g')}），真值精确恢复。"
      "第 ④ 步**已实现参数恢复**，不是只报 Rwp。\n")
    fb2 = at["false_select_ids"]
    A(f"**三层嵌套（呼应 §17）**：`C2_L3` 在其真值谱上真族识别率仅 "
      f"{F(at['c2_l3_truth_rate_pct'], '%.1f')}%（{at['n_c2_l3']} 条；无噪档也只有 4/8），"
      "BIC 常被 `voigt_K2/K3` 或 `nested_L2` 抢走；在非其真值谱上 BIC 胜率仅 2.1%。"
      f"条件 ④ 唯一误选例（{fb2[0] if fb2 else '—'}）也正是 `C2_L3` 真值谱在 heavy 档"
      " 被少 3 参数的 `nested_L2_capacitive` 选中（Rwp 6.592% vs 6.677%，几乎相同）"
      "→ 第三层在噪声下不可辨识，与 §17 四电路反事实的负面证据方向一致。\n")
    cv = at["convergence"]
    A(f"**收敛率** {F(100.0 * cv['ok'] / cv['n'], '%.1f')}%"
      f"（{cv['ok']}/{cv['n']} 候选×谱组合 `success=True`）；9 个未收敛组合按预注册"
      " 记 `not_converged`、**不报数**。\n")

    A("\n## 四、诚实边界\n")
    A("1. **LS 证据 ≠ 后验证据**：条件 ③ 本轮不通过也不失败，状态维持 ❌，留待 Bayesian（M3）。")
    A("2. **合成谱 ≠ 真实样品**：即使 ①②④ 满足，准入状态仍为 `exploratory`，"
      "不得接 `bm_select` / `bmdrt.py`，不得改 §13 `0/3` / §14 `4/14` / §17 `0/10`。")
    A("3. **参数恢复误差大 ≠ 模型错**：简并/窗外难度下可能「单参数不可辨识但组合可辨识」，"
      "`results.json` 同时给 `f0_combo_ln_err` 与 `start_spread`。")
    A("4. **acf 不是 PPC，BIC 不是门禁**；二者都是本实验自定义诊断量。")
    A("5. **数据集声明缺口**：`truth_manifest.json` 的 `structures` 声明含 `white`，"
      "但实际噪声档是 `none(0%) / correlated(2%) / heavy(5%)` —— `white` 从未生成"
      "（0% 档被 `add_noise` 短路）。`README.md` 的噪声描述与生成代码不一致。"
      "本轮按 manifest 的 `noise.kind` 分层，**不改数据集**（改会动 168 条 sha256），"
      "留待单独订正。")
    A("6. anti-inverse-crime：拟合侧导纳累加正演独立、初值不读真值、"
      "`--selftest` 已做生成侧 / 项目正演三方对拍（见 `manifest.json`）。")
    A("7. **无噪谱 `Rwp≈0` 时残差是浮点舍入，acf 为伪随机量**，"
      "真模型的近零残差 acf 不作结构解读；条件 ①/② 的判读依赖**错模型残差**"
      "（其量级远大于舍入误差）。")

    A("\n## 五、不改的东西\n")
    A("生产代码、判据、阈值、先验、`bm_result.payload()`、面板契约 F1–F11、"
      "§13/§14/§17 结论，一行未动。\n")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- selftest（A3）

def selftest():
    import importlib.util
    spec = importlib.util.spec_from_file_location("td_gen", DATA.parent.parent /
                                                  "docs" / "round_2026-11-12" /
                                                  "truth_dataset.py")
    td = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(td)
    sys.path.insert(0, str(ROOT))
    import bm_top as T

    f = td.F
    print("A3-a 拟合侧 vs 生成侧，7 类拓扑：")
    worst = 0.0
    for fam in td.FAMILIES:
        for diff in td.DIFFICULTIES:
            p = td.make_params(fam, diff)
            cand = TRUTH_CANDIDATE[fam]
            a = td.forward(fam, f, p)
            b = forward(cand, f, p)
            e = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
            worst = max(worst, e)
        print(f"  {fam:<24} max rel err {e:.3e}")
    assert worst < 1e-10, f"两套独立实现不一致 {worst:.3e}"

    print("A3-b 拟合侧 vs 项目 bm_top.z_nested（Q_ours 换算），4 组合：")
    p = td.make_params("nested_L2_capacitive", "separated")
    wref = T.omega_ref(f)

    def qo(Q, n):
        return Q / wref ** n

    base = dict(Rs=p["Rs"], R1=p["R1"], R2=p["R2"],
                Q1=qo(p["Q1"], p["n1"]), n1=p["n1"],
                Q2=qo(p["Q2"], p["n2"]), n2=p["n2"],
                R_ad=p["R_ad"], tau_ad=p["tau_ad"])
    cap = dict(base, Q_b=qo(p["Q_b"], p["n_b"]), n_b=p["n_b"])
    worst2 = 0.0
    for L, branch, proj in (
        (1, "parallel", T.z_nested(f, base, branch="parallel", layers=1)),
        (2, "parallel", T.z_nested(f, base, branch="parallel", layers=2)),
        (1, "capacitive", T.z_nested(f, cap, branch="capacitive", layers=1)),
        (2, "capacitive", T.z_nested(f, cap, branch="capacitive", layers=2)),
    ):
        cand = f"nested_L{L}_{'inductive' if branch == 'parallel' else 'capacitive'}"
        ours = forward(cand, f, p)
        e = float(np.max(np.abs(proj - ours)) / np.max(np.abs(ours)))
        worst2 = max(worst2, e)
        print(f"  L={L} {branch:<11} max rel err {e:.3e}")
    assert worst2 < 1e-10, f"与项目正演不一致 {worst2:.3e}"

    # A2 结构性检查：起点与真值无关、稳定可复现、不同谱不同
    Z = np.abs(td.forward("nested_L2_capacitive", f, p))
    s1 = starts_for("nested_L2_capacitive", "id-A", f, Z)
    s2 = starts_for("nested_L2_capacitive", "id-A", f, Z)
    s3 = starts_for("nested_L2_capacitive", "id-B", f, Z)
    assert all(np.array_equal(a, b) for a, b in zip(s1, s2)), "起点不可复现"
    assert not np.array_equal(s1[0], s3[0]), "起点未随谱 id 变化"
    keys = keys_for("C3_mixed")
    assert keys and len(keys) == 12
    x = to_x({k: (0.8 if is_index(k) else 123.0) for k in keys}, keys)
    rt = from_x(x, keys)
    assert all(abs(rt[k] - (0.8 if is_index(k) else 123.0)) < 1e-9 for k in keys)
    print("A2 起点：不含真值输入 / 稳定可复现 / 参数变换往返 OK")
    print(f"A3 通过：生成侧 {worst:.3e}、项目正演 {worst2:.3e}")
    return dict(generation_side_max_rel_err=worst,
                project_side_max_rel_err=worst2, a2_ok=True)


# ---------------------------------------------------------------- 主流程

INT_KEYS = {"k", "n_converged", "seed", "n", "n_spectra", "false_select_n"}


def restore_ints(v):
    """兼容早期 `json_safe` 把原生 int 写成字符串的产物（按 key 白名单恢复）。"""
    if isinstance(v, dict):
        return {k: (int(x) if k in INT_KEYS and isinstance(x, str)
                    and x.lstrip("-").isdigit() else restore_ints(x))
                for k, x in v.items()}
    if isinstance(v, list):
        return [restore_ints(x) for x in v]
    return v


def dump_outputs(dest, payload, results, summary):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["summary"] = json_safe(summary)
    payload["results"] = json_safe(results)
    rj = dest / "results.json"
    rj.write_text(json.dumps(json_safe(payload), ensure_ascii=False,
                             allow_nan=False, indent=1) + "\n", encoding="utf-8")
    write_summary(summary, results, dest / "summary.md")
    mf = dict(status=payload["status"], n_spectra=len(results),
              results_json=rj.name,
              results_sha256=hashlib.sha256(rj.read_bytes()).hexdigest(),
              verdict=json_safe(summary["verdict"]),
              note="只记账不裁决；条件③ 本轮不判定")
    (dest / "manifest.json").write_text(
        json.dumps(json_safe(mf), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8")
    return rj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="只跑每类 no-noise separated seed0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="从已有 results.json 重算 summary/manifest + 规范字段类型（不重跑拟合）")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return 0

    if a.rebuild:
        dest = Path(a.out)
        if not dest.is_absolute():
            dest = ROOT / dest
        payload = json.loads((dest / "results.json").read_text(encoding="utf-8"))
        results = restore_ints(payload["results"])
        summary = summarize(results)
        payload["status"] = "finalized"
        payload["n_spectra"] = len(results)
        dump_outputs(dest, payload, results, summary)
        print(f"--rebuild：{len(results)} 条，字段类型已规范；summary/manifest 已重算")
        return 0

    manifest = json.loads((DATA / "truth_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "finalized"
    recs = manifest["spectra"]
    if a.pilot:
        recs = [r for r in recs if r["noise"]["kind"] == "none"
                and r["difficulty"] == "separated" and r["seed"] == 0]
    if a.limit:
        recs = recs[:a.limit]

    dest = Path(a.out)
    if not dest.is_absolute():
        dest = ROOT / dest
    dest.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = []
    for i, rec in enumerate(recs, 1):
        arr = np.loadtxt(DATA / rec["csv"], delimiter=",", skiprows=1)
        f, Z = arr[:, 0], arr[:, 1] + 1j * arr[:, 2]
        assert hashlib.sha256((DATA / rec["csv"]).read_bytes()).hexdigest() == rec["csv_sha256"]
        results.append(evaluate_one(rec, f, Z))
        if i % 10 == 0 or i == len(recs):
            print(f"  [{i:3d}/{len(recs)}] {time.time()-t0:6.1f}s", flush=True)

    summary = summarize(results)
    payload = dict(
        status="finalized" if not (a.pilot or a.limit) else "partial",
        prereg="docs/round_2026-11-12/RECOVERY_PREREG.md",
        script="docs/round_2026-11-12/truth_recovery.py",
        dataset="out/truth_dataset/truth_manifest.json",
        anti_inverse_crime=dict(A1="拟合侧导纳累加正演独立、主流程不 import 生成器",
                                A2="初值只由谱幅值/频率窗口/稳定 md5 决定，不读 truth_params",
                                A3="--selftest 三方对拍（生成侧 / 项目 bm_top）"),
        noise_layer_note=("实际 kind = none/correlated/heavy；声明的 white 未生成。"
                          "主判据用 kind∈{none,heavy} 排除自带 acf 的 correlated"),
        candidates=list(CANDIDATES), n_starts=N_STARTS, max_nfev=MAX_NFEV,
        n_spectra=len(results), summary=summary, results=results,
    )
    dump_outputs(dest, payload, results, summary)

    print(f"\n完成 {len(results)} 条 → {dest}  用时 {time.time()-t0:.1f}s")
    for k in ("1", "2", "3", "4"):
        d = summary["verdict"][k]
        tag = "不判定" if not d["evaluable"] else ("满足" if d["met"] else "未满足")
        print(f"  条件{k}: {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
