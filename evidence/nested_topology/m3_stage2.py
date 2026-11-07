# -*- coding: utf-8 -*-
"""M3 档 2：单组参数恢复（`BM_TOP_M3_PLAN.md` §6.2 / §12）。

一次只跑一个条件，分三个子步骤：

    A  固定噪声 `sigma = (0.02·zmax, 0.02·zmax)` —— 隔离后验几何，只看
       `R_ad` / `tau_ad` 能不能从数据里回来；
    B  自由噪声（`sigma=None`，走 `LogNormal(log 0.01·zmax, 1)`）—— 看噪声
       参数与电路参数共同采样是否引入新的简并；
    R  逐点相对噪声，条件名 **`rel-perpoint-2pct`**（M3.5，§20.1）——
       `σ_gen,j = 2%·|Z_j^true|`（生成）、`σ_fit,j = 2%·|Z_j^obs|`（拟合，已知、
       不采）。**与 A/B 不是同一种统计模型**：ELPD 不可跨口径比较。

**各步必须分开记录**。A 通过不等于 B 通过：A 里 σ 是已知的，B 多两个自由度，
后验几何完全不同（`bm_core` 记录过无噪声 + 自由 σ 的漏斗）。R 与 A 的差别在
**噪声口径**而不是自由度，两者的后验也不能互相印证。

合成谱由 `bm_top_gen` 独立生成（不调 `bm_top.z_nested`），噪声按 §4.1 的
固定标准差（A/B）或逐点相对（R）口径，实虚独立。

两个 seed 必须分开（§20.11）
---------------------------
    `--seeds`        **采样器** seed（NUTS 的 `PRNGKey`）
    `--data-seeds`   **数据** seed（合成谱的噪声实现），长度与 `--seeds` 一致

    不给 `--data-seeds` 时沿用旧行为（data_seed = 采样器 seed），并在表头
    打印"两种随机源同时变化"。要**固定谱只测采样器敏感性**，写
    `--data-seeds 0,0,0,0,0`；要**独立噪声实现**，写 `--data-seeds 0,1,2,3,4`
    （采样器 seed 可固定也可变，但报告里现在能分开读）。

R 条件的两条硬约束（§20.10）
---------------------------
    · `--noise` 必须是 2%（`rel-perpoint-2pct` 的名字由它生成，别名不等于新条件）；
    · `elpd`/`pareto_k` **不作验收判据**：`σ_j = 2%·|Z_j^obs|` 让留出点的尺度
      依赖该点自身，`elpd_i` 不是 `p(Z_j|Z_-j)` 的估计。`acceptance()` 会把
      `pareto_k_ok` 摘进 `skipped_checks`，输出里另有明示。

退出码与三套判据（§20.13）
--------------------------
    | 判据 | 内容 | 谁决定退出码 |
    |---|---|---|
    | `run_validity` | R̂ / ESS / divergence / 逐点数组长度 | **R 条件** |
    | `acceptance` | 上面 + 覆盖 + 优先恢复参数 4% | **A / B 条件** |
    | `split_verdict` | 多 seed `sd_r`（M3.5 主终点） | 都不决定；单列打印 |

    旧 `acceptance` 的覆盖与 4% 是 M3 的**精度**门槛，不是 M3.5 的成败判据；
    `r` 条件仍照算并落盘（追溯用），但只由 `run_validity` 决定退出码。

用法
----
    # 单组正式跑（4 链 × 1200 样本）
    python evidence/nested_topology/m3_stage2.py --stage both

    # M3.5 的独立条件（逐点相对 2%，5 个独立噪声实现 + 固定采样器 seed）
    python evidence/nested_topology/m3_stage2.py --stage r \
        --seeds 0,0,0,0,0 --data-seeds 0,1,2,3,4 \
        --json evidence/nested_topology/m35_r.json

    # 只跑 A、换采样器 seed、写 JSON
    python evidence/nested_topology/m3_stage2.py --stage a --seed 1 --json out.json

    # 链路自检（1 链 150 样本，只证明能跑通，结果不可引用）
    python evidence/nested_topology/m3_stage2.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import bm_top as T                      # noqa: E402  （拟合器）
import bm_top_gen as G                  # noqa: E402  （独立生成器）

# 审查批准的基准真值（沿用 §11.2 冒烟的那组，便于对照）
TRUTH = dict(Rs=10.0, R1=55.0, R2=600.0, Q1=200.0, n1=0.75,
             Q2=900.0, n2=0.85, R_ad=3000.0, tau_ad=1.6)

# A 组：感抗环主导，回答"感抗支路能否恢复"；
# B 组：只把 R_ad 降到 300（zmax 3049→891、σ 61→17.8），回答"形状参数能否改善"。
# 两组共用同一噪声口径（固定 2%·zmax），差异只有一个因子，便于归因。
TRUTH_SETS = {"a": dict(TRUTH), "b": dict(TRUTH, R_ad=300.0)}

F_GRID = (1e-2, 1e5, 61)


SERIES_TRUTH = dict(R_L=800.0, tau_L=1.625)     # L_ad = 1300，与 parallel 支路可比


def truth_params(branch: str, layers: int, override: dict | None = None) -> dict:
    """当前拓扑的真值（只含该拓扑真正采样的参数）。`override` 覆盖个别值。"""
    p = dict(TRUTH)
    if override:
        p.update(override)
    if branch == "series":
        p.pop("R_ad", None)
        p.pop("tau_ad", None)
        p.update(SERIES_TRUTH)
    return {k: p[k] for k in T.sampled_keys(branch, layers) if k in p}


def make_spectrum(noise: float = 0.02, seed: int = 0, branch: str = "parallel",
                  layers: int = 2, truth: dict | None = None,
                  noise_mode: str = "fixed"):
    """独立生成 + 加噪。返回 (f, Z, zmax)。

    `noise_mode="fixed"` → `σ = noise·zmax`（A/B 组，同方差）；
    `noise_mode="relative"` → `σ_j = noise·|Z_j^true|`（R 组，逐点异方差）。
    """
    f = np.logspace(np.log10(F_GRID[0]), np.log10(F_GRID[1]), F_GRID[2])
    p = truth if truth is not None else truth_params(branch, layers)
    Z0 = G.gen(f, p, branch=branch, layers=layers)
    zmax = float(np.max(np.abs(Z0)))
    Z = G.add_noise(Z0, noise, mode=noise_mode, seed=seed, zmax=zmax)
    return f, Z, zmax


# 条件名 → 生成噪声口径。名字由 `condition_name()` 按**实际** noise 生成，
# 不再写死在表里——否则 `--stage r --noise 0.05` 会自称 `rel-perpoint-2pct`，
# 报告与实验不一致。2% 时结果逐字等于 §20.1 冻结的名字。
NOISE_MODE = {"a": "fixed", "b": "fixed", "r": "relative"}

# §20.1 冻结的 R 条件噪声水平（含生成与拟合）。改这个值等于新条件，不要复用名字。
FROZEN_REL_NOISE = 0.02


def _pct_tag(noise: float) -> str:
    """0.02 → '2pct'（`round` 吸收 0.02*100=2.0000000000000004 这类浮点尾巴）。"""
    return f"{round(float(noise) * 100.0, 6):g}pct"


def condition_name(stage: str, noise: float) -> str:
    if stage == "b":
        # 口径是"自由估计 σ"，与噪声水平无关；数据侧 noise 另由 rec["noise"] 记录，
        # 表头也一直打印，所以 (condition, noise) 合起来无歧义
        return "free-lognormal"
    return {"a": f"abs-fixed-{_pct_tag(noise)}",
            "r": f"rel-perpoint-{_pct_tag(noise)}"}[stage]


def sigma_for_stage(stage: str, noise: float, Z: np.ndarray, zmax: float):
    """把条件映射成 `fit_top` 的 `sigma` 参数。

    R 组用**加噪后**的 `|Z_obs|`（不是真值谱）——用真值算 σ 等于给拟合器额外
    信息，§20.1 已冻结"只用实测谱"这一条。
    """
    if stage == "a":
        return (noise * zmax, noise * zmax)
    if stage == "r":
        s = noise * np.abs(np.asarray(Z, complex))
        return np.vstack([s, s])             # (2, n_freq)，实虚同 σ
    if stage == "b":
        return None
    raise ValueError(f"未知 stage {stage!r}")


def _truth_of_comb(name: str, truth: dict):
    """组合参数对应的真值（真值来自同一组 `truth_params`）。"""
    if name == "R1+R2":
        return truth["R1"] + truth["R2"]
    if name.startswith("L_ad"):
        if "R_ad" in truth:
            return truth["R_ad"] * truth["tau_ad"]
        return truth["R_L"] * truth["tau_L"]
    if name.startswith("R2·Q2"):
        return truth["R2"] * truth["Q2"] ** truth["n2"]
    return None


def summarize(res: dict, truth: dict, branch: str, layers: int,
              sigma_note: str) -> dict:
    """把一个 `fit_top` 结果压成可记录的指标。"""
    s = res["s"]
    keys = list(T.sampled_keys(branch, layers))
    if branch == "series":
        keys.append("L_ad")                     # 派生量也看一下
    out = {"sigma_note": sigma_note, "params": {}, "diagnostics": {}}

    for k in keys:
        if k not in s:
            continue
        v = np.asarray(s[k], float).ravel()
        lo, med, hi = np.percentile(v, [5, 50, 95])
        rec = {"median": float(med), "p05": float(lo), "p95": float(hi),
               "width90": float(hi - lo), "n": int(v.size)}
        if k in truth:
            tv = float(truth[k])
            rec.update(truth=tv,
                       rel_err=float((med - tv) / tv),
                       covered=bool(lo <= tv <= hi))
        out["params"][k] = rec

    # 成对相关：只取采样参数，避免派生量重复计入
    mat_keys = [k for k in T.sampled_keys(branch, layers) if k in s and
                np.asarray(s[k]).size > 1]
    M = np.column_stack([np.asarray(s[k], float).ravel() for k in mat_keys])
    C = np.corrcoef(M, rowvar=False)
    idx = {k: i for i, k in enumerate(mat_keys)}
    pairs = []
    for i in range(len(mat_keys)):
        for j in range(i + 1, len(mat_keys)):
            pairs.append((mat_keys[i], mat_keys[j], float(C[i, j])))
    pairs.sort(key=lambda t: -abs(t[2]))

    # 审查点名要看：优先恢复参数 vs 形状参数（§6.2 的"结构简并"来源）
    pri_keys = [k for k in ("R_ad", "tau_ad", "R_L", "tau_L") if k in idx]
    shape_keys = [k for k in ("Rs", "R1", "R2", "Q1", "n1", "Q2", "n2") if k in idx]
    pri_shape = [{"a": a, "b": b, "rho": float(C[idx[a], idx[b]])}
                 for a in pri_keys for b in shape_keys]

    # 多 seed 要盯的几对固定组合（"权衡是否稳定"看的就是这几个 ρ）
    def _rho(a, b):
        return float(C[idx[a], idx[b]]) if (a in idx and b in idx) else None

    named = {"R_ad,tau_ad": _rho("R_ad", "tau_ad"),
             "R_L,tau_L": _rho("R_L", "tau_L"),
             "R1,R2": _rho("R1", "R2"),
             "Rs,n1": _rho("Rs", "n1")}

    # 组合参数（§6.2："无法分别辨识者，报告可辨识的组合"）
    comb = {}
    if {"R1", "R2"} <= set(s):
        comb["R1+R2"] = np.asarray(s["R1"], float).ravel() + np.asarray(s["R2"], float).ravel()
    if branch == "parallel":
        comb["L_ad(=R_ad·tau_ad)"] = (np.asarray(s["R_ad"], float).ravel()
                                      * np.asarray(s["tau_ad"], float).ravel())
    else:
        comb["L_ad(=R_L·tau_L)"] = (np.asarray(s["R_L"], float).ravel()
                                    * np.asarray(s["tau_L"], float).ravel())
    if branch == "parallel" and "R2" in s:      # CPE 的"等效容抗"组合，Q 与 n 强耦合时看
        comb["R2·Q2^n2"] = (np.asarray(s["R2"], float).ravel()
                            * np.asarray(s["Q2"], float).ravel()
                            ** np.median(np.asarray(s["n2"], float).ravel()))
    for k, v in comb.items():
        lo, med, hi = np.percentile(v, [5, 50, 95])
        rec = {"median": float(med), "p05": float(lo), "p95": float(hi),
               "width90": float(hi - lo), "n": int(v.size)}
        tv = _truth_of_comb(k, truth)
        if tv is not None:
            rec.update(truth=float(tv), rel_err=float((med - tv) / tv),
                       covered=bool(lo <= tv <= hi))
        out.setdefault("combinations", {})[k] = rec

    # 预注册主终点（§20.4）：r = R1/(R1+R2) 的**后验**，不是线性化预测。
    # 只有该拓扑真有两个并联节点时才有定义（layers=1 没有 R2）。
    # 这里必须逐 draw 算 r 再统计：先取 R1/R2 的中位再相除会得到不同的量
    # （中位不保序于非线性变换，且会抹掉 R1 与 R2 的负相关）。
    if {"R1", "R2"} <= set(s):
        r1 = np.asarray(s["R1"], float).ravel()
        r2 = np.asarray(s["R2"], float).ravel()
        rs = r1 / (r1 + r2)
        lo, med, hi = np.percentile(rs, [5, 50, 95])
        rec = {"median": float(med), "sd": float(np.std(rs, ddof=1)),
               "p05": float(lo), "p95": float(hi), "width90": float(hi - lo),
               "n": int(rs.size)}
        if {"R1", "R2"} <= set(truth):
            rt = float(truth["R1"]) / (float(truth["R1"]) + float(truth["R2"]))
            rec.update(truth=rt, bias=float(med - rt),
                       covered=bool(lo <= rt <= hi))
        out["split_r"] = rec

    pw = res.get("pointwise_elpd")
    pk = np.asarray(res.get("pareto_k", []), float)
    div = res.get("divergences")
    n_draws = int(np.asarray(s[mat_keys[0]]).size)
    out["diagnostics"] = {
        "rhat_max": res.get("rhat_max"),
        "ess_min": res.get("ess_min"),
        "rhat_by_param": res.get("rhat_by_param") or {},
        "ess_by_param": res.get("ess_by_param") or {},
        "divergences": div,
        "div_frac": (None if div is None else float(div) / max(n_draws, 1)),
        "n_draws": n_draws,
        "elpd": res.get("elpd"), "elpd_se": res.get("elpd_se"),
        # σ_j 由本批观测算出时 elpd/pareto_k 没有标准 LOO 解释（§20.10）：
        # 标志由 `fit_top(sigma_from_data=...)` 给，不在这里猜
        "predictive_diag_valid": bool(
            res.get("elpd_valid_for_predictive_comparison", True)),
        "elpd_note": res.get("elpd_note"),
        "pointwise_len": (None if pw is None else int(np.asarray(pw).size)),
        "pareto_k_max": (float(np.max(pk)) if pk.size else None),
        "pareto_k_gt07_frac": (float(np.mean(pk > 0.7)) if pk.size else None),
        "corr_max_abs": (abs(pairs[0][2]) if pairs else None),
        "corr_top": [{"a": a, "b": b, "rho": r} for a, b, r in pairs[:5]],
        "corr_pri_vs_shape": pri_shape,
        "corr_pri_vs_shape_max": (max(abs(c["rho"]) for c in pri_shape)
                                  if pri_shape else None),
        "corr_named": named,
    }
    return out


def acceptance(rec: dict, n_freq: int,
               err_targets=(0.01, 0.04, 0.08),
               predictive_diag: bool = True) -> dict:
    """§6.3 的机器判定。只判**这一次**记录，不当覆盖率结论。

    `predictive_diag=False`（σ_j 由本批观测算出，§20.10）时 `pareto_k_ok`
    **从判定里摘掉**并记进 `skipped_checks`——既不判通过也不判失败。把它留在
    判定里等于用一个没有标准 LOO 解释的量当门禁；直接删掉又会让人以为漏检。
    `pointwise_len_ok` 保留：它只检查"逐点数组的长度对不对"，与解释无关。
    """
    d = rec["diagnostics"]
    p = rec["params"]
    pri = ["R_ad", "tau_ad"] if "R_ad" in p else ["R_L", "tau_L"]
    worst_err = max(abs(p[k].get("rel_err", np.inf)) for k in pri if k in p)
    checks = {
        "rhat_lt_1.01": (d["rhat_max"] is not None and d["rhat_max"] < 1.01),
        "ess_gt_200": (d["ess_min"] is not None and d["ess_min"] > 200),
        "div_lt_0.1pct": (d["div_frac"] is not None and d["div_frac"] < 1e-3),
        "pointwise_len_ok": (d["pointwise_len"] == n_freq),
        "pareto_k_ok": (d["pareto_k_max"] is not None and d["pareto_k_max"] <= 0.7),
        "covered": all(bool(p[k].get("covered")) for k in pri if k in p),
        "pri_err_le_4pct": bool(worst_err <= err_targets[1]),
    }
    skipped = []
    if not predictive_diag:
        checks.pop("pareto_k_ok")
        skipped = ["pareto_k_ok"]
    return {"checks": checks, "skipped_checks": skipped,
            "pri_err_worst": float(worst_err),
            "pass": all(checks.values())}


# 采样器健康度的三个判据。收敛统计与"运行有效性"共用这一份清单，
# 避免出现"两套收敛定义"（§20.13）。
CONVERGENCE_CHECKS = ("rhat_lt_1.01", "ess_gt_200", "div_lt_0.1pct")


def run_validity(rec: dict) -> dict:
    """**运行有效性**：链健康 + 逐点数组长度。**不含**精度、不含覆盖。

    为什么必须跟 `acceptance()` 分开：M3.5 的结论由 `sd_r` 给出，而 `sd_r` 能不能
    读只取决于链有没有收敛。旧 `acceptance` 里的 `covered` / `pri_err_le_4pct` 是
    M3 的**精度**门槛——拿它决定 M3.5 的退出码，会出现两种错配：

    · 主终点 `sd_r` 达标，但 `R_ad` 单次误差 >4% → 退出码 1，看起来像 M3.5 失败；
    · 旧判据全过（参数恢复得准），但链没收敛、后验 SD 被压小 → 退出码 0，
      看起来像 M3.5 成功。

    两套口径都落盘，但只有这里（`r` 条件）决定退出码。
    """
    c = rec["acceptance"]["checks"]
    gates = {k: bool(c[k]) for k in CONVERGENCE_CHECKS if k in c}
    if "pointwise_len_ok" in c:              # 只查长度，与解释无关，保留
        gates["pointwise_len_ok"] = bool(c["pointwise_len_ok"])
    return {"checks": gates, "pass": all(gates.values())}


def exit_gate(rec: dict) -> dict:
    """退出码依据哪套判据（§20.13）。

    · `a` / `b`（M3）：沿用 `acceptance().pass` —— **不改旧语义**，与前几轮记录可比；
    · `r`（M3.5）：只看 `run_validity().pass`。旧 acceptance 仍照算并落盘，只作
      追溯，不冒充 M3.5 结论。
    """
    if rec["stage"] == "r":
        return {"used": "run_validity", "pass": bool(rec["run_valid"]["pass"])}
    return {"used": "acceptance", "pass": bool(rec["acceptance"]["pass"])}


def run_stage(stage: str, *, data_seed: int, sampler_seed: int, noise: float,
              branch: str, layers: int,
              n_warmup: int, n_samples: int, n_chains: int, progress: bool,
              init_mode: str = "prior", target_accept: float = 0.9,
              max_tree_depth: int = 10, samples_path: str | None = None,
              truth_override: dict | None = None) -> dict:
    """跑一个条件的一次拟合。

    **两个 seed 必须分开**（§20.11）：`data_seed` 只控制合成谱的噪声实现，
    `sampler_seed` 只控制 NUTS 的随机数。旧版用一个 `seed` 同时喂两处，于是
    "5 个 seed" 既是 5 份不同数据、又是 5 次不同初始化，误差 SD 里两种随机源
    混在一起，无法归因。
    """
    truth = truth_params(branch, layers, override=truth_override)
    noise_mode = NOISE_MODE[stage]
    condition = condition_name(stage, noise)
    f, Z, zmax = make_spectrum(noise=noise, seed=data_seed, branch=branch,
                               layers=layers, truth=truth, noise_mode=noise_mode)
    sigma = sigma_for_stage(stage, noise, Z, zmax)
    if init_mode == "oracle":
        # 真值经物理 → 无约束转换。只用于"先验/初值是否卡住采样器"的诊断，
        # **不能**当正式结果：链从真值出发时，"收敛得好"可能只是初值的回声。
        # 必须把 sigma 一起传进去：固定噪声时模型没有 s_re/s_im 位点，
        # 多给会被 NumPyro 当成多余的树节点。
        init = T.physical_to_unconstrained(
            truth, zmax=zmax, w_ref=float(2 * np.pi * np.sqrt(f.min() * f.max())),
            branch=branch, layers=layers, sigma=sigma, num_chains=n_chains)
    else:
        init = None                            # 走 numpyro 默认（先验支撑内的随机初值）

    t0 = time.time()
    res = T.fit_top(f, Z, branch=branch, layers=layers,
                    n_warmup=n_warmup, n_samples=n_samples, num_chains=n_chains,
                    seed=sampler_seed, sigma=sigma, init_values=init,
                    progress=progress,
                    # R 条件的 σ_j 由 |Z_j^obs| 算出 → elpd 无标准 LOO 解释
                    sigma_from_data=(stage == "r"),
                    target_accept=target_accept, max_tree_depth=max_tree_depth)
    dt = time.time() - t0
    if samples_path:
        np.savez_compressed(samples_path, **{k: np.asarray(v) for k, v in res["s"].items()})

    note = {"a": f"fixed {noise:.0%}·zmax",
            "b": "free LogNormal(0.01·zmax,1)",
            "r": f"perpoint {noise:.0%}·|Z_obs|"}[stage]
    rec = summarize(res, truth, branch, layers, note)
    rec.update(stage=stage, condition=condition, data_seed=data_seed,
               sampler_seed=sampler_seed, noise=noise,
               noise_mode=noise_mode, branch=branch, layers=layers,
               n_chains=n_chains, n_samples=n_samples, n_warmup=n_warmup,
               n_freq=int(f.size), zmax=zmax, seconds=round(dt, 1),
               init_mode=init_mode, target_accept=target_accept,
               max_tree_depth=max_tree_depth, truth=truth,
               sigma_mode=res.get("sigma_mode"),
               # §20.1：含 Σ log σ_j 的归一化常数依赖噪声口径 → ELPD 不能跨口径比
               elpd_comparable_across_conditions=(stage != "r"))
    rec["acceptance"] = acceptance(
        rec, int(f.size),
        predictive_diag=bool(rec["diagnostics"]["predictive_diag_valid"]))
    rec["run_valid"] = run_validity(rec)       # 链健康 + 长度，不含精度/覆盖
    rec["exit_gate"] = exit_gate(rec)          # 本次记录对退出码负责的判据
    return rec


def print_record(rec: dict) -> None:
    d, p = rec["diagnostics"], rec["params"]
    print(f"\n{'='*74}")
    print(f"档 2  [{rec['stage']}] {rec.get('condition', '—')}  {rec['branch']}_L{rec['layers']}  "
          f"data_seed={rec['data_seed']}  sampler_seed={rec['sampler_seed']}  "
          f"噪声={rec['noise']:.0%}  噪声口径={rec['sigma_note']}")
    print(f"       {rec['n_chains']} 链 × {rec['n_samples']} 样本 / warmup {rec['n_warmup']}"
          f"  初值={rec['init_mode']}  → {rec['seconds']}s")
    print(f"{'='*74}")
    print(f"{'参数':<8}{'真值':>10}{'中位':>12}{'90% 区间':>26}{'相对误差':>10}{'覆盖':>6}")
    for k, v in p.items():
        ci = "[{:.4g}, {:.4g}]".format(v["p05"], v["p95"])
        if "truth" in v:
            cover = "✓" if v["covered"] else "✗"
            print(f"{k:<8}{v['truth']:>10.4g}{v['median']:>12.4g}{ci:>26}"
                  f"{v['rel_err']:>+9.2%}{cover:>6}")
        else:
            print(f"{k:<8}{'—':>10}{v['median']:>12.4g}{ci:>26}")
    comb = rec.get("combinations") or {}
    if comb:
        print(f"\n{'组合参数':<20}{'真值':>12}{'中位':>12}{'90% 区间':>26}{'相对误差':>10}{'覆盖':>6}")
        for k, v in comb.items():
            ci = "[{:.4g}, {:.4g}]".format(v["p05"], v["p95"])
            if "truth" in v:
                cover = "✓" if v["covered"] else "✗"
                print(f"{k:<20}{v['truth']:>12.4g}{v['median']:>12.4g}{ci:>26}"
                      f"{v['rel_err']:>+9.2%}{cover:>6}")
            else:
                print(f"{k:<20}{'—':>12}{v['median']:>12.4g}{ci:>26}")
    print(f"\n诊断  R̂max={d['rhat_max']:.4f}  ESSmin={d['ess_min']:.0f}  "
          f"div={d['divergences']} ({d['div_frac']:.4%})")
    print(f"      elpd={d['elpd']:.2f}±{d['elpd_se']:.2f}  "
          f"pointwise_len={d['pointwise_len']} (N={rec['n_freq']})  "
          f"pareto_k_max={d['pareto_k_max']:.3f}  "
          f"k>0.7 占比={d['pareto_k_gt07_frac']:.1%}")
    if not d.get("predictive_diag_valid", True):
        print("      ⚠ σ_j 由 |Z_j^obs| 算出：上面两个数字**不是**样本外预测评分"
              "（留出 j 后 σ_j 会变）；\n"
              "        pareto_k 同源，已从本组验收判据中摘除（§20.10）")
    sp = rec.get("split_r")
    if sp:
        ci = "[{:.4f}, {:.4f}]".format(sp["p05"], sp["p95"])
        extra = (f"  真值 {sp['truth']:.4f} 偏差 {sp['bias']:+.4f}"
                 f" 覆盖 {'✓' if sp['covered'] else '✗'}" if "truth" in sp else "")
        print(f"      拆分 r=R1/(R1+R2) = {sp['median']:.4f}  sd={sp['sd']:.4f}  "
              f"90% {ci} 宽 {sp['width90']:.4f}{extra}")
    print("      最强相关：")
    for c in d["corr_top"][:3]:
        print(f"        corr({c['a']}, {c['b']}) = {c['rho']:+.3f}")
    rb = d.get("rhat_by_param") or {}
    if rb:
        bad = sorted(rb.items(), key=lambda kv: -kv[1])[:3]
        print("      R̂ 最差参数：" + "，".join(f"{k}={v:.4f}" for k, v in bad))
    if d.get("corr_pri_vs_shape_max") is not None:
        worst = max(d["corr_pri_vs_shape"], key=lambda c: abs(c["rho"]))
        print(f"      优先恢复 × 形状参数 最大 |ρ| = {abs(worst['rho']):.3f}"
              f"（{worst['a']} vs {worst['b']}）")
    a = rec["acceptance"]
    rv = rec.get("run_valid") or {"pass": a["pass"], "checks": a["checks"]}
    if rec["stage"] == "r":
        print(f"\n运行有效性：{'通过' if rv['pass'] else '未通过'}"
              f"（R̂ / ESS / divergence / 数据长度）→ **M3.5 退出码依据**")
        print(f"旧 M3 acceptance（含覆盖与 4% 精度）："
              f"{'全过' if a['pass'] else '未全过'}"
              f"  优先恢复参数最大误差 {a['pri_err_worst']:.2%}"
              f"——这是 M3 的**精度**判据，不作 M3.5 结论")
    else:
        print(f"\n判定（仅本组）：{'通过' if a['pass'] else '未通过'}"
              f"  优先恢复参数最大误差 {a['pri_err_worst']:.2%}")
    for k, v in a["checks"].items():
        print(f"      {'✓' if v else '✗'} {k}")
    if a.get("skipped_checks"):
        print(f"      – 不判定：{', '.join(a['skipped_checks'])}"
              "（σ 由本批数据算出，无标准 LOO 解释，见 §20.10）")
    if not rec.get("elpd_comparable_across_conditions", True):
        print("      注：本条件的 elpd 含 Σ log σ_j（逐点口径），"
              "**不可**与 A/B 组的 elpd 比较。")


def _fmt_pct(x, w: int = 11) -> str:
    return f"{x:>+{w}.1%}" if x is not None else f"{'—':>{w}}"


def _fmt_num(x, w: int = 8, s: str = "g") -> str:
    s = s.lstrip(".")                     # 允许传 ".3%" 或 "3%"，统一成 "3%"
    return f"{x:>{w}.{s}}" if x is not None else f"{'—':>{w}}"


def _seed_variation(recs: list[dict]) -> str:
    """多 seed 究竟变了什么。**不能靠想当然**：旧版一个 seed 同时喂数据与采样器。"""
    ds = {r.get("data_seed") for r in recs}
    ss = {r.get("sampler_seed") for r in recs}
    if None in ds or None in ss:               # 老记录没有这两个字段
        return "未知（老记录只存了单个 seed，两种随机源未分离）"
    if len(ds) == 1 and len(ss) > 1:
        return (f"固定谱（data_seed={sorted(ds)[0]}）、只变采样器 → 敏感性=采样随机性"
                f"（含初值与 NUTS 随机流，不只是初始点）")
    if len(ds) > 1 and len(ss) == 1:
        return (f"独立噪声实现（sampler_seed={sorted(ss)[0]} 固定）→ 变化的来源是"
                f"噪声实现，不是采样器 seed")
    if len(ds) > 1 and len(ss) > 1:
        return ("数据与采样器**同时**变化 → SD 混合两种随机源，不能单独归因到"
                "参数恢复或采样器（§20.11）")
    return "单一实现（不是重复实验）"


def _mcmc_converged(rec: dict) -> bool:
    """只看采样器健康度（R̂/ESS/divergence），**不含**精度与覆盖。

    为什么要单独拆出来：`acceptance()` 的 `pass` 混了"恢复得准不准"，而 sd_r
    能不能读只取决于链有没有收敛。不收敛的链会把后验 SD 压小（链没走开），
    此时"sd_r≤0.05"是假达标。判据清单与 `run_validity()` 共用
    `CONVERGENCE_CHECKS`，避免出现两套收敛定义。
    """
    c = rec["acceptance"]["checks"]
    return all(bool(c.get(k)) for k in CONVERGENCE_CHECKS)


def split_stats(recs: list[dict]) -> dict | None:
    """预注册主终点（§20.4）：`r=R1/(R1+R2)` 的**后验** SD，多 seed 汇总。

    读的是 `summarize()` 里**逐 draw** 算出的 `split_r`（不是拿 R1/R2 的中位相除：
    中位不保序于非线性变换，而且会抹掉 R1 与 R2 的强负相关）。
    判据只用 `sd_median`；`covered`/`n_sd_le_0.05` 是探索性描述，seed 数太少，
    不能当覆盖率检验。
    """
    import statistics as stats

    rows = [r["split_r"] for r in recs if r.get("split_r")]
    if not rows:
        return None
    sds = [float(r["sd"]) for r in rows]
    n_conv = sum(1 for r in recs if _mcmc_converged(r))
    return {
        "param": "r=R1/(R1+R2)",
        "n": len(rows),
        "n_mcmc_converged": n_conv,
        "sd_median": float(stats.median(sds)),
        "sd_mean": float(stats.mean(sds)),
        "sd_min": float(min(sds)), "sd_max": float(max(sds)),
        "width90_median": float(stats.median([r["width90"] for r in rows])),
        "r_median_median": float(stats.median([r["median"] for r in rows])),
        "truth": (float(rows[0]["truth"]) if "truth" in rows[0] else None),
        "covered": int(sum(1 for r in rows if r.get("covered"))),
        "n_sd_le_0.05": int(sum(1 for x in sds if x <= 0.05)),
        "n_sd_ge_0.30": int(sum(1 for x in sds if x >= 0.30)),
    }


def split_verdict(sp: dict) -> str:
    """§20.4 的三档判据，只按中位 sd_r；**链没收敛就不判**。"""
    if sp.get("n_mcmc_converged", sp["n"]) < sp["n"]:
        return (f"不可判（{sp['n'] - sp['n_mcmc_converged']}/{sp['n']} 个 seed 的链"
                f"未收敛，sd_r 被压小，不能当终点读数）")
    if sp["sd_median"] <= 0.05:
        return "达标（sd_r≤0.05）"
    if sp["sd_median"] >= 0.30:
        return "判负（sd_r≥0.30，r 的 90% 区间宽>0.5）"
    return "部分约束（0.05<sd_r<0.30）"


def print_seed_table(recs: list[dict]) -> dict:
    """逐 seed 一行，再给汇总。**不汇总成单一数字**——审查要求逐 seed 保留。"""
    import statistics as stats

    r0 = recs[0]
    branch = r0["branch"]
    p1, p2 = ("R_ad", "tau_ad") if branch == "parallel" else ("R_L", "tau_L")
    lk = (f"L_ad(={p1}·{p2})")
    has_split = any(r.get("split_r") for r in recs)
    print(f"\n{'='*104}")
    print(f"多 seed 汇总：{branch}_L{r0['layers']}  {r0['noise']:.0%} 噪声"
          f"（{r0.get('condition', '—')} / {r0['sigma_note']}）  "
          f"{r0['n_chains']} 链 × {r0['n_samples']} / warmup {r0['n_warmup']}  "
          f"ta={r0['target_accept']}  depth={r0['max_tree_depth']}  初值={r0['init_mode']}")
    print(f"随机源：{_seed_variation(recs)}")
    print(f"{'='*104}")
    hdr = (f"{'data/samp':>10}{'R_ad 误差':>12}{'tau_ad 误差':>13}{'L_ad 误差':>12}"
           f"{'覆盖':>6}{'ρ(ad,τ)':>10}{'ρ(R1,R2)':>11}")
    if has_split:
        hdr += f"{'r 中位':>9}{'sd_r':>8}"
    hdr += f"{'R̂max':>8}{'ESSmin':>8}{'div':>9}{'gate':>7}"
    print(hdr)
    for r in recs:
        p, d, c = r["params"], r["diagnostics"], r.get("combinations", {})
        e1 = p.get(p1, {}).get("rel_err")
        e2 = p.get(p2, {}).get("rel_err")
        e3 = c.get(lk, {}).get("rel_err")
        cov = sum(1 for k in (p1, p2) if p.get(k, {}).get("covered"))
        named = d.get("corr_named") or {}
        r1, r2 = named.get(f"{p1},{p2}"), named.get("R1,R2")
        tag = f"{r['data_seed']}/{r['sampler_seed']}"
        line = (f"{tag:>10}"
                f"{_fmt_pct(e1, 12)}{_fmt_pct(e2, 13)}{_fmt_pct(e3, 12)}"
                f"{f'{cov}/2':>6}"
                f"{(f'{r1:+.3f}' if r1 is not None else '—'):>10}"
                f"{(f'{r2:+.3f}' if r2 is not None else '—'):>11}")
        if has_split:
            sp = r.get("split_r") or {}
            line += (f"{_fmt_num(sp.get('median'), 9, '4f')}"
                     f"{_fmt_num(sp.get('sd'), 8, '4f')}")
        gate = r.get("exit_gate") or {"pass": r["acceptance"]["pass"]}
        line += (f"{_fmt_num(d['rhat_max'], 8, '4f')}{_fmt_num(d['ess_min'], 8, '0f')}"
                 f"{_fmt_num(d['div_frac'], 8, '.3%')}"
                 f"{'✓' if gate['pass'] else '✗':>7}")
        print(line)

    e1s = [r["params"].get(p1, {}).get("rel_err") for r in recs]
    e2s = [r["params"].get(p2, {}).get("rel_err") for r in recs]
    e1s = [x for x in e1s if x is not None]
    e2s = [x for x in e2s if x is not None]
    cov1 = sum(1 for r in recs if r["params"].get(p1, {}).get("covered"))
    cov2 = sum(1 for r in recs if r["params"].get(p2, {}).get("covered"))
    npass = sum(1 for r in recs if r["acceptance"]["pass"])
    nconv = sum(1 for r in recs if _mcmc_converged(r))
    n = len(recs)
    print("-" * 96)
    print(f"汇总  {p1} 误差 中位 {stats.median(e1s):+.2%}  范围 "
          f"[{min(e1s):+.2%}, {max(e1s):+.2%}]")
    print(f"      {p2} 误差 中位 {stats.median(e2s):+.2%}  "
          f"均值 {stats.mean(e2s):+.2%} ± {stats.stdev(e2s):.2%}  "
          f"范围 [{min(e2s):+.2%}, {max(e2s):+.2%}]  同号 "
          f"{sum(1 for x in e2s if x > 0)}/{len(e2s)} 正")
    print(f"      90% 区间覆盖：{p1} {cov1}/{n}、{p2} {cov2}/{n}；"
          f"|{p2}|≤4% 的 seed "
          f"{sum(1 for x in e2s if abs(x) <= 0.04)}/{len(e2s)}")
    print(f"      MCMC 收敛（R̂/ESS/div）：{nconv}/{n}；"
          f"acceptance 全过（含覆盖与 4% 精度）：{npass}/{n}"
          f" ← 后者**不是**收敛数")
    print("      gate = 退出码依据（a/b：acceptance 全过；r：运行有效性），"
          "不是 M3.5 主终点结论（§20.13）")
    print("      注：seed 数是探索性重复，不是 90% 覆盖率的统计检验。")
    sp = split_stats(recs)
    if sp:
        print(f"      主终点 r=R1/(R1+R2)：sd_r 中位 {sp['sd_median']:.4f}"
              f"（逐 seed {sp['sd_min']:.4f}…{sp['sd_max']:.4f}），"
              f"90% 区间宽中位 {sp['width90_median']:.4f}，"
              f"真值覆盖 {sp['covered']}/{sp['n']}")
        print(f"      → {split_verdict(sp)}；单 seed 达 ≤0.05 的个数 "
              f"{sp['n_sd_le_0.05']}/{sp['n']}（探索性，不作判据）；"
              f"链收敛 {sp['n_mcmc_converged']}/{sp['n']}")
    return {"n_seeds": n, "n_pass_acceptance": npass,
            "n_mcmc_converged": nconv,
            "seed_variation": _seed_variation(recs),
            "covered": {p1: cov1, p2: cov2},
            "split": sp,
            "err": {p1: {"median": stats.median(e1s), "min": min(e1s),
                         "max": max(e1s)},
                    p2: {"median": stats.median(e2s), "mean": stats.mean(e2s),
                         "sd": stats.stdev(e2s) if len(e2s) > 1 else 0.0,
                         "min": min(e2s), "max": max(e2s),
                         "n_positive": sum(1 for x in e2s if x > 0)}}}


def aggregate_seeds(recs: list[dict], params: list[str] | None = None) -> dict:
    """多 seed 的**稳健性**汇总。只读，不改 `acceptance()` 的语义。

    三个层次分开记（审查要求）：
      1. 单次精度：每次拟合的 |中位误差| 是否 ≤4%（`acceptance()` 里的判据，逐 seed）；
      2. 多 seed 稳健性：误差均值 / SD / 最大绝对误差 —— **变化来源由
         `_seed_variation()` 从记录里读**，这里不预设"就是采样随机性"；
      3. 后验不确定性：90% 区间宽度与覆盖次数。

    这里**不给**"5 个 seed 的均值 ≤4% 就算达标"的结论：这些重复共用同一组真值、
    同一频栅，不是 5 次独立实验。均值 0% 也推不出"精度 4%"，正负抵消的单次误差
    同样没达精度。

    另外三个计数**不能混**：`n_mcmc_converged` 只看 R̂/ESS/divergence；
    `n_pass_acceptance` 是旧判据（含覆盖与 4%）；两者都不是"实验成功"。
    判定 M3.5 成败的是 `primary_endpoint`。

    **变化的到底是什么，由 `_seed_variation()` 从记录里读**，不写死。旧版
    `run_stage` 用一个 `seed` 同时喂数据生成与 NUTS，所以"5 seed"其实是 5 份
    不同噪声 + 5 次不同初始化，SD 不能归因；现在 `data_seed` / `sampler_seed`
    分开记录，固定谱与独立噪声实现可以分别设计（§20.11）。

    另附**预注册主终点** `split`（`r=R1/(R1+R2)` 的多 seed `sd_r`），见 `split_stats`。
    """
    import statistics as stats

    r0 = recs[0]
    branch = r0["branch"]
    if params is None:
        pri = ["R_ad", "tau_ad"] if branch == "parallel" else ["R_L", "tau_L"]
        params = [k for k in pri if k in r0["params"]] + \
                 [k for k in ("Rs", "R1", "R2", "Q1", "n1", "Q2", "n2")
                  if k in r0["params"]]

    def pack(errs, widths, cov, n_le4):
        return {"n": len(errs),
                "mean": float(stats.mean(errs)) if errs else None,
                "sd": float(stats.stdev(errs)) if len(errs) > 1 else None,
                "median": float(stats.median(errs)) if errs else None,
                "max_abs": float(max(abs(x) for x in errs)) if errs else None,
                "n_le_4pct": int(n_le4), "covered": int(cov),
                "width90_median": float(stats.median(widths)) if widths else None}

    out = {"n_seeds": len(recs),
           "seed_variation": _seed_variation(recs),
           "n_pass_acceptance": sum(1 for r in recs if r["acceptance"]["pass"]),
           "n_mcmc_converged": sum(1 for r in recs if _mcmc_converged(r)),
           "params": {}, "combinations": {}}

    for k in params:
        errs, widths, cov, n4 = [], [], 0, 0
        for r in recs:
            v = r["params"].get(k)
            if not v or "rel_err" not in v:
                continue
            errs.append(float(v["rel_err"]))
            widths.append(float(v["width90"] / v["truth"]))
            cov += bool(v["covered"])
            n4 += abs(v["rel_err"]) <= 0.04
        if errs:
            out["params"][k] = pack(errs, widths, cov, n4)

    for k in ("L_ad(=R_ad·tau_ad)", "L_ad(=R_L·tau_L)", "R1+R2"):
        errs, widths, cov, n4 = [], [], 0, 0
        for r in recs:
            v = (r.get("combinations") or {}).get(k)
            if not v or "rel_err" not in v:
                continue
            errs.append(float(v["rel_err"]))
            widths.append(float(v["width90"] / v["truth"]))
            cov += bool(v["covered"])
            n4 += abs(v["rel_err"]) <= 0.04
        if errs:
            out["combinations"][k] = pack(errs, widths, cov, n4)

    sp = split_stats(recs)
    if sp:
        out["split"] = sp
        judgeable = sp["n_mcmc_converged"] == sp["n"]
        out["primary_endpoint"] = {
            "name": "r=R1/(R1+R2) 的多 seed 后验 SD",
            "criterion": "sd_r ≤ 0.05（预注册 §20.4）；≥0.30 判负",
            # `value` 只在**可判**时给数：链没收敛时 sd_r 被压小，那个数字看着
            # 完全正常（甚至"达标"），下游脚本只取 value 就会当成有效结果。
            # 原始读数不丢，放在 observed_value。
            "valid": bool(judgeable),
            "value": (sp["sd_median"] if judgeable else None),
            "observed_value": sp["sd_median"],
            "reason_if_invalid": (None if judgeable else
                                  f"{sp['n'] - sp['n_mcmc_converged']}/{sp['n']} "
                                  f"个 seed 的链未过 R̂/ESS/div 门禁，"
                                  f"后验 SD 不可读；observed_value 仅供追溯"),
            "verdict": split_verdict(sp),
            "source": "summarize() 的 split_r（逐 draw 计算，非线性化预测）",
        }
    return out


def print_robustness(agg: dict) -> None:
    """三层指标并排打印。列名写死，避免把'均值'读成'精度'。"""
    n = agg["n_seeds"]
    print(f"\n{'='*104}")
    print("多 seed 稳健性（**不改变单次 4% 判据**；本表只描述波动，"
          "波动来源见下方『随机源』，不预设是采样随机性）")
    print(f"{'='*104}")
    print(f"{'参数':<20}{'均值':>10}{'SD':>9}{'中位':>9}{'max|err|':>10}"
          f"{'|err|≤4%':>10}{'覆盖':>8}{'区间宽中位':>12}")
    for grp, title in (("params", "单参数"), ("combinations", "组合参数")):
        if not agg[grp]:
            continue
        print(f"-- {title} " + "-" * 88)
        for k, v in agg[grp].items():
            frac4 = "{}/{}".format(v["n_le_4pct"], n)
            cov = "{}/{}".format(v["covered"], n)
            print(f"{k:<20}{_fmt_pct(v['mean'], 10)}{_fmt_pct(v['sd'], 9)}"
                  f"{_fmt_pct(v['median'], 9)}{_fmt_pct(v['max_abs'], 10)}"
                  f"{frac4:>10}{cov:>8}{v['width90_median']:>11.1%}")
    print("-" * 104)
    print(f"MCMC 收敛（R̂/ESS/div）：{agg.get('n_mcmc_converged', 0)}/{n}；"
          f"acceptance 全过（含覆盖与 4% 精度）：{agg['n_pass_acceptance']}/{n}"
          f" ← **不是**收敛数，也不是 M3.5 结论")
    print(f"随机源：{agg.get('seed_variation', '—')}")
    sp = agg.get("split")
    if sp:
        pe = agg.get("primary_endpoint") or {}
        mark = "" if pe.get("valid", True) else "（**无效读数**，仅追溯）"
        print(f"主终点 r=R1/(R1+R2)：sd_r 中位 {sp['sd_median']:.4f}{mark}"
              f"（均值 {sp['sd_mean']:.4f}，范围 {sp['sd_min']:.4f}…{sp['sd_max']:.4f}）；"
              f"90% 区间宽中位 {sp['width90_median']:.4f}；真值覆盖 {sp['covered']}/{sp['n']}")
        print(f"        → {split_verdict(sp)}"
              + (f"；真值 r={sp['truth']:.4f}" if sp["truth"] is not None else ""))
        print("        注：sd_r 是后验 SD，不是线性化预测；判据只看中位，"
              "不看单个 seed 是否 ≤0.05。")
        if not pe.get("valid", True):
            print(f"        JSON 里 primary_endpoint.value = null，"
                  f"原始读数在 observed_value（{sp['sd_median']:.4f}）")
    print("注意 ①：均值接近 0 **不等于**达到 4% 精度——正负误差可以互相抵消。")
    print("注意 ②：区间宽远超 100% 的参数，其'覆盖真值'没有信息量（宽区间包得住任何值）。")
    print("注意 ③：SD 与 4% 目标同量级时，单次拟合判达标/不达标接近抛硬币；")
    print("        结论只能按分布说，不能拿单个 seed 说'达标'或'不达标'。")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M3 档 2 单组参数恢复 / M3.5 逐点相对条件")
    ap.add_argument("--stage", choices=["a", "b", "r", "both", "all"], default="both",
                    help="a=固定绝对 σ；b=自由 σ；r=逐点相对 2%%（rel-perpoint-2pct）；"
                         "both=a+b；all=a+b+r")
    ap.add_argument("--seed", type=int, default=0,
                    help="单个采样器 seed（等价于 --seeds 只给一项）")
    ap.add_argument("--seeds", default=None,
                    help="逗号分隔的**采样器** seed 列表（如 0,1,2,3,4）")
    ap.add_argument("--data-seeds", default=None,
                    help="逗号分隔的**数据** seed 列表（合成谱的噪声实现），"
                         "长度须与 --seeds 一致；不给则沿用旧行为（data_seed=采样器 seed）")
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--truth-set", choices=sorted(TRUTH_SETS), default="a",
                    help="真值组：a=感抗环主导(R_ad=3000)，b=感抗环减弱(R_ad=300)")
    ap.add_argument("--truth", default=None,
                    help="个别真值覆盖，如 R_ad=300,tau_ad=1.6（优先于 --truth-set）")
    ap.add_argument("--branch", choices=["parallel", "series"], default="parallel")
    ap.add_argument("--layers", type=int, choices=[1, 2], default=2)
    ap.add_argument("--smoke", action="store_true", help="1 链 150 样本，只验链路")
    ap.add_argument("--init", choices=["prior", "oracle"], default="prior",
                    help="prior=默认随机初值（正式用）；oracle=真值初值（只做诊断）")
    ap.add_argument("--warmup", type=int, default=None, help="覆盖 warmup 步数")
    ap.add_argument("--samples", type=int, default=None, help="覆盖每链样本数")
    ap.add_argument("--chains", type=int, default=None, help="覆盖链数")
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--save-samples", action="store_true", help="把后验样本存成 npz 备查")
    ap.add_argument("--progress", action="store_true", help="打印 MCMC 进度条")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    kw = dict(n_warmup=150, n_samples=150, n_chains=1) if args.smoke else \
        dict(n_warmup=600, n_samples=1200, n_chains=4)
    if args.warmup:
        kw["n_warmup"] = args.warmup
    if args.samples:
        kw["n_samples"] = args.samples
    if args.chains:
        kw["n_chains"] = args.chains

    stages = {"a": ["a"], "b": ["b"], "r": ["r"],
              "both": ["a", "b"], "all": ["a", "b", "r"]}[args.stage]
    seeds = ([args.seed] if not args.seeds
             else [int(s) for s in args.seeds.split(",") if s.strip()])
    data_seeds = None
    if args.data_seeds:
        data_seeds = [int(s) for s in args.data_seeds.split(",") if s.strip()]
        if len(data_seeds) != len(seeds):
            ap.error(f"--data-seeds 有 {len(data_seeds)} 项，--seeds 有 {len(seeds)} 项，"
                     f"两者必须一一对应（固定谱就重复写同一个数）")

    # §20.1 冻结：R 条件的噪声水平是 2%。名字由 noise 生成，一旦放开就会
    # 出现"报告写 rel-perpoint-2pct、实际跑 5%"这种不可追溯的错配。
    if "r" in stages and abs(args.noise - FROZEN_REL_NOISE) > 1e-12:
        ap.error(f"--stage r 的噪声口径已冻结为 {FROZEN_REL_NOISE:.0%}"
                 f"（rel-perpoint-{_pct_tag(FROZEN_REL_NOISE)}）；"
                 f"收到 --noise {args.noise:g}。换噪声水平等于新条件，"
                 f"请先改 §20.1 的定义再动代码")
    truth_override = dict(TRUTH_SETS[args.truth_set])
    if args.truth:
        for kv in args.truth.split(","):
            k, _, v = kv.partition("=")
            truth_override[k.strip()] = float(v)
    print(f"真值组：{ {k: v for k, v in truth_override.items()} }")
    print(f"随机源：data_seeds={data_seeds or seeds}（{'未分离，沿用旧行为'if data_seeds is None else '显式给出'}）"
          f"  sampler_seeds={seeds}")
    t_all = time.time()
    recs = []
    for st in stages:
        for i, sd in enumerate(seeds):
            ds = sd if data_seeds is None else data_seeds[i]
            sp = None
            if args.save_samples and args.json:
                sp = str(Path(args.json).with_suffix(f".{st}.d{ds}.s{sd}.samples.npz"))
            rec = run_stage(st, data_seed=ds, sampler_seed=sd, noise=args.noise,
                            branch=args.branch, layers=args.layers,
                            progress=args.progress,
                            init_mode=args.init, target_accept=args.target_accept,
                            max_tree_depth=args.max_tree_depth, samples_path=sp,
                            truth_override=truth_override, **kw)
            print_record(rec)
            recs.append(rec)

    summary = robustness = None
    if len(seeds) > 1 and len(stages) == 1:
        summary = print_seed_table(recs)
        robustness = aggregate_seeds(recs)
        print_robustness(robustness)
    elif len(seeds) > 1:
        for st in stages:
            sub = [r for r in recs if r["stage"] == st]
            summary = print_seed_table(sub)
            robustness = aggregate_seeds(sub)
            print_robustness(robustness)

    # 退出码分两套口径（§20.13）：M3 条件沿用 acceptance（不改旧语义），
    # R 条件只看运行有效性——旧 acceptance 里的覆盖/4% 是 M3 的精度门槛，
    # 让它决定 M3.5 的成败等于用错判据。
    ok = True if args.smoke else all(r["exit_gate"]["pass"] for r in recs)
    print(f"\n{'='*74}")
    if args.smoke:
        print("冒烟结束：只证明链路通，结果不可引用（1 链 150 样本不收敛）")
    else:
        for st in stages:
            sub = [r for r in recs if r["stage"] == st]
            nv = sum(1 for r in sub if r["run_valid"]["pass"])
            nc = sum(1 for r in sub if _mcmc_converged(r))
            na = sum(1 for r in sub if r["acceptance"]["pass"])
            used = sub[0]["exit_gate"]["used"]
            print(f"[{st}] 运行有效性 {nv}/{len(sub)}（R̂/ESS/div/长度）  "
                  f"MCMC 收敛 {nc}/{len(sub)}  acceptance 全过 {na}/{len(sub)}"
                  f"  → 退出码依据："
                  f"{'运行有效性' if used == 'run_validity' else 'acceptance'}")
        for st in stages:
            if st != "r":
                continue
            sp = split_stats([r for r in recs if r["stage"] == st])
            if sp:
                print(f"M3.5 研究结论（R 条件，只看 r 的 sd_r）：{split_verdict(sp)}")
        print("通过" if ok else "有未通过项——按 §8 记录，失败不删不改 seed")
    if len(recs) > 1:
        print(f"{len(recs)} 次拟合总耗时 {time.time()-t_all:.0f}s")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "smoke": bool(args.smoke),
                        "config": {"branch": args.branch, "layers": args.layers,
                                   "noise": args.noise,
                                   "sampler_seeds": seeds,
                                   "data_seeds": (data_seeds if data_seeds is not None
                                                  else seeds),
                                   "seed_pairing": ("explicit --data-seeds"
                                                    if data_seeds is not None
                                                    else "legacy: data_seed=sampler_seed"),
                                   "stages": stages,
                                   "conditions": [condition_name(s, args.noise)
                                                  for s in stages],
                                   "truth_set": args.truth_set,
                                   "truth": truth_override,
                                   "n_chains": kw["n_chains"],
                                   "n_samples": kw["n_samples"],
                                   "n_warmup": kw["n_warmup"],
                                   "target_accept": args.target_accept,
                                   "max_tree_depth": args.max_tree_depth,
                                   "init": args.init},
                        "summary": summary, "robustness": robustness,
                        "records": recs},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"记录写入 {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
