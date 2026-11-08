# -*- coding: utf-8 -*-
"""M3.5 条件化诊断：把 `n2` 钉在真值后，`d0` 的 `r` 会怎样（`BM_TOP_M3_PLAN.md` §20.21）。

**只回答一个问题**：在 `d0` 的观测数据下，已知真实 `n2` 能否显著改变 `r` 的后验
位置和不确定性？若不能，剩余不确定性是否仍主要沿 `Q2` 与阻抗拆分方向分布。

三条边界（写进代码里，免得结果被当成别的东西）：

1. **条件后验 ≠ 恢复性能**。固定 `n2` 得到 `p(r, Q2, … | Z, n2 = n2_true)`，原来的是
   边缘后验 `p(r, Q2, … | Z)`；两者之差是"已知 `n2`"这条信息的影响，不是"`n2` 单向
   导致 `r` 偏低"的证明，更不能当真实（`n2` 未知）场景下的恢复性能。
2. **不纳入 M3.5 验收**，不产生 `primary_endpoint`，不改 §20.4 的任何判定。
3. **不重新生成噪声**：谱来自与 §20.17 完全相同的 `m3_stage2.make_spectrum(seed=0)`；
   `control` 这一跑必须与归档的 `d0` 复现——**逐位不一致或对照未过门禁就直接中止**
   （`check_control()`，返回码 3，不产出判定；§20.23 审查要求）。

存档口径（§20.23）：`*.samples.npz` 里存的是**展平**样本，顺序为**链优先**
（chain 0 的 `n_samples` 个 draw 连续存放，再 chain 1…）。因此
`s_ch[k].reshape(-1, *s_ch[k].shape[2:])` 与展平数组**逐位相等**（`fit()` 里恒等断言），
链结构可无损恢复；链级统计（每条链的 `r` 中位、链间离散、每链 divergence 数）
直接写进 JSON 的 `chain_diag`，不必额外存一份 chain 形状的文件。

为什么用 `condition` 而不是 `numpyro.do`
----------------------------------------
`do` 是**因果干预**语义：它把位点换成常数，但**仍保留一个同名采样位点**
（"introduce fresh sample sites with the same names whose values do not propagate"），
所以 `n2` 照样被采样，拿到手还是边缘后验。`condition` 才把位点标成观测。
实测（`--selftest` 守着）`condition` 用的是**约束空间**值：`log_density` 与
"把 0.85 直接写进观测项"相等，而不是 `sigmoid(0.85)` 的解释。
代价只是去掉 `n2` 的先验项——那是常数倍数，不改变其余参数的条件后验。

用法
----
    python evidence/nested_topology/m35_cond_n2.py --selftest
    python evidence/nested_topology/m35_cond_n2.py --data-seed 0 --fix n2=0.85 \
        --ref evidence/nested_topology/m35_r_w5000.json \
        --json evidence/nested_topology/m35_cond_n2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import bm_top as T                          # noqa: E402  模型工厂（生产代码，不改）
import m3_stage2 as M                       # noqa: E402  谱、σ 口径、真值、阈值
from m35_sampler_sens import RHAT_MAX, ESS_MIN      # noqa: E402  门禁阈值同一来源

BRANCH, LAYERS = "parallel", 2
# §20.21：配置同 §20.17，warmup 5000（FROZEN_SAMPLER 里是预注册的 2000，这里显式覆盖）
CFG = dict(M.FROZEN_SAMPLER, n_warmup=5000)
REPRO_TOL = 1e-9            # 对照重跑的逐位复现门槛（超过即中止）
NEAR_TRUTH_TOL = 0.02       # 与判据 A 同一容差；§20.23 起写进 JSON，不再只靠现算


# ------------------------------------------------------------------ 数据

def build_data(data_seed: int, noise: float = M.FROZEN_REL_NOISE):
    """重建 `d<seed>` 的谱与 σ（与 m3_stage2 同一函数、同一口径）。"""
    truth = M.truth_params(BRANCH, LAYERS)
    f, Z, zmax = M.make_spectrum(noise=noise, seed=data_seed, branch=BRANCH,
                                 layers=LAYERS, truth=truth, noise_mode="relative")
    sigma = M.sigma_for_stage("r", noise, Z, zmax)
    spec_hash = hashlib.sha256(
        np.concatenate([np.asarray(f, float), Z.real, Z.imag]).tobytes()
    ).hexdigest()[:16]
    return truth, f, Z, zmax, sigma, spec_hash


# ------------------------------------------------------------------ 采样

def div_by_chain(div, n_chains: int):
    """发散计数按链拆开。`MCMC.get_extra_fields()` 给的是**展平**数组（链优先），
    必须先 reshape；否则 `sum(axis=-1)` 会把 4 条链压成 1 个数（本轮踩过）。"""
    if div is None:
        return None
    a = np.asarray(div)
    if a.ndim == 1 and n_chains and a.size % n_chains == 0:
        a = a.reshape(n_chains, -1)
    if a.ndim == 1:
        return [int(x) for x in a]
    return [int(x) for x in np.sum(a, axis=-1)]


def fit(f, Z, *, sigma, condition: dict | None, cfg: dict, seed: int,
        progress: bool = False) -> dict:
    """跑一次 NUTS；`condition` 非空时求条件后验。

    模型本身**完全来自** `bm_top._make_model`：不复制模型定义，避免"诊断脚本里的
    模型"和"生产模型"悄悄分叉。
    """
    from numpyro import handlers
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    sig_mode, sig_re, sig_im = T._check_sigma(f, sigma, 0.0)
    model = T._make_model(BRANCH, LAYERS, sig_mode, sig_re, sig_im, 0.0)
    if condition:
        model = handlers.condition(model, data={k: float(v)
                                                for k, v in condition.items()})

    mcmc = MCMC(NUTS(model, target_accept_prob=cfg["target_accept"],
                     max_tree_depth=cfg["max_tree_depth"]),
                num_warmup=cfg["n_warmup"], num_samples=cfg["n_samples"],
                num_chains=cfg["n_chains"], progress_bar=progress)
    t0 = time.time()
    mcmc.run(__import__("jax").random.PRNGKey(seed), f, Z.real, Z.imag,
             extra_fields=("diverging",), init_params=None)
    dt = time.time() - t0

    s_ch = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    s = {k: v.reshape((-1,) + v.shape[2:]) for k, v in s_ch.items()}
    # 存档是展平的，链结构只能靠"链优先的纯 reshape"恢复——这里守住这个恒等式。
    flat_exact = all(
        np.array_equal(np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:]), s[k])
        for k, v in s_ch.items())
    if not flat_exact:
        raise RuntimeError("展平样本不是 chain 数组的纯 reshape，链结构不可恢复")
    st = summary(s_ch, prob=0.9, group_by_chain=True)
    rhat_map = {k: float(np.nanmax(np.asarray(v["r_hat"]))) for k, v in st.items()}
    ess_map = {k: float(np.nanmin(np.asarray(v["n_eff"]))) for k, v in st.items()}
    div = mcmc.get_extra_fields().get("diverging")
    n_draws = int(np.asarray(s["R1"]).size)
    divs = int(np.sum(div)) if div is not None else None
    n_chains = int(np.shape(s_ch["R1"])[0])
    per_chain = div_by_chain(div, n_chains)
    return {"s": s, "s_ch": s_ch, "flat_exact": flat_exact,
            "n_chains": n_chains,
            "n_draws_per_chain": (int(n_draws // n_chains) if n_chains else None),
            "div_by_chain": per_chain,
            "seconds": round(dt, 1),
            "rhat_by_param": rhat_map, "ess_by_param": ess_map,
            "rhat_max": max(rhat_map.values()) if rhat_map else None,
            "ess_min": min(ess_map.values()) if ess_map else None,
            "divergences": divs,
            "div_frac": (None if divs is None else divs / max(n_draws, 1)),
            "sigma_mode": sig_mode}


def gate(rec: dict) -> dict:
    """与 `m3_stage2.CONVERGENCE_CHECKS` 同义的三个检查（诊断用）。"""
    checks = {"rhat_lt_1.01": (rec["rhat_max"] is not None and rec["rhat_max"] < RHAT_MAX),
              "ess_gt_200": (rec["ess_min"] is not None and rec["ess_min"] > ESS_MIN),
              "div_lt_0.1pct": (rec["div_frac"] is not None and rec["div_frac"] < 1e-3)}
    return {"checks": checks, "pass": all(checks.values())}


class ReproError(RuntimeError):
    """对照不可比。继续跑下去，所有"与边缘后验之差"都不可读，所以直接中止。"""


def check_control(repro: float, gate_ctrl: dict, ref_gate: bool | None = None) -> None:
    """硬保护：对照不成立就不产出条件化判定（§20.23 第一项审查意见）。

    三条都中止而非降级成 `indeterminate`：`indeterminate` 是"跑了但判不了"，
    对照失效是"根本没跑成可比的实验"，两者的归档含义必须分开。

    1. `repro` 有限且 ≤ `REPRO_TOL`——否则条件与对照用的就不是同一份边缘后验语境；
    2. 对照重跑**过门禁**——逐位复现一个未过门禁的归档结果，不足以给宽度比较背书；
    3. 归档记录本身过门禁（给了 `ref_gate` 时）——否则"边缘参照"这一侧也不健康。
    """
    if not math.isfinite(repro) or repro > REPRO_TOL:
        raise ReproError(f"对照重跑与归档样本不一致：max_rel_diff={repro:.3e} "
                         f"> {REPRO_TOL:.0e}（后面的差值不能读）")
    if not (gate_ctrl or {}).get("pass"):
        bad = [k for k, v in ((gate_ctrl or {}).get("checks") or {}).items() if not v]
        raise ReproError(f"对照重跑未过门禁（{'、'.join(bad) or '原因未知'}）："
                         f"逐位复现也不足以为条件化判定背书")
    if ref_gate is False:
        raise ReproError("归档记录本身未过门禁，不能充当边缘参照")


# ------------------------------------------------------------------ 统计

def stats(s: dict, truth: dict) -> dict:
    """后验读数。`r` 逐 draw 算；同时带上审查点名要看的 Q2/n1/Q1 与 R1+R2。"""
    g = lambda k: np.asarray(s[k], float).ravel()
    R1, R2 = g("R1"), g("R2")
    r = R1 / (R1 + R2)
    r_truth = truth["R1"] / (truth["R1"] + truth["R2"])
    lo, med, hi = np.percentile(r, [5, 50, 95])
    # 派生统计量：与判据 A 同一容差。§20.23 审查要求归档，避免以后只能从样本重算。
    near = {"tol": NEAR_TRUTH_TOL,
            "prob": float((np.abs(r - r_truth) <= NEAR_TRUTH_TOL).mean()),
            "n_draws": int(r.size)}
    out = {"r": {"median": float(med), "sd": float(np.std(r, ddof=1)),
                 "p05": float(lo), "p95": float(hi), "width90": float(hi - lo),
                 "mean": float(r.mean()), "truth": r_truth,
                 "gap": float(med - r_truth),
                 "truth_pct": float((r < r_truth).mean()),
                 "covered": bool(lo <= r_truth <= hi),
                 "near_truth": near},
           "params": {}, "corr": {}}
    for k in ("Q2", "n1", "Q1", "R1", "R2", "n2", "R_ad", "tau_ad", "Rs"):
        if k not in s:
            continue
        v = g(k)
        a, m, b = np.percentile(v, [5, 50, 95])
        rec = {"median": float(m), "p05": float(a), "p95": float(b),
               "width90": float(b - a)}
        if k in truth:
            rec.update(truth=float(truth[k]),
                       rel_err=float((m - truth[k]) / truth[k]),
                       truth_pct=float((v < truth[k]).mean()),
                       covered=bool(a <= truth[k] <= b))
        out["params"][k] = rec
    for k in ("n1", "Q1", "n2", "Q2", "Rs", "R_ad"):
        if k in s:
            out["corr"][f"r~{k}"] = float(np.corrcoef(r, g(k))[0, 1])
    if "n1" in s and "Q1" in s:
        out["corr"]["Q2~n1"] = float(np.corrcoef(g("Q2"), g("n1"))[0, 1])
        out["corr"]["Q2~Q1"] = float(np.corrcoef(g("Q2"), g("Q1"))[0, 1])
        out["corr"]["n1~Q1"] = float(np.corrcoef(g("n1"), g("Q1"))[0, 1])
    summ = R1 + R2
    out["sum_R1R2"] = {"median": float(np.median(summ)),
                       "truth": float(truth["R1"] + truth["R2"]),
                       "rel_err": float(np.median(summ) / (truth["R1"] + truth["R2"]) - 1),
                       "width90": float(np.percentile(summ, 95) - np.percentile(summ, 5))}
    from m35_r_bias_diag import ridge_geometry
    out["ridge"] = ridge_geometry(R1, R2, truth)
    return out


def chain_diag(s_ch: dict, truth: dict, *, div_per_chain=None,
               flat_exact: bool | None = None) -> dict:
    """链级诊断：从 chain 维度算每条链的 `r` 与链间离散（§20.23 第四项意见）。

    展平存档是**链优先**的纯 reshape，所以链结构无损；这里把"需要看链混合时
    真正要读的那几个数"直接归档，省去以后重新载入 chain 形状的文件。
    """
    if "R1" not in s_ch or "R2" not in s_ch:
        return {}
    R1 = np.asarray(s_ch["R1"], float)
    R2 = np.asarray(s_ch["R2"], float)
    if R1.ndim < 2:
        return {}
    r = R1 / (R1 + R2)
    r_truth = truth["R1"] / (truth["R1"] + truth["R2"])
    med = np.median(r, axis=1)
    out = {"n_chains": int(r.shape[0]),
           "n_draws_per_chain": int(r.shape[1]),
           "chain_median_r": [float(x) for x in med],
           "chain_mean_r": [float(x) for x in r.mean(axis=1)],
           "chain_truth_pct": [float((row < r_truth).mean()) for row in r],
           "between_chain_sd_of_median": (float(np.std(med, ddof=1))
                                          if med.size > 1 else None),
           "within_chain_sd_median": float(np.median(np.std(r, axis=1, ddof=1))),
           "gap_spread_across_chains": float(med.max() - med.min()),
           "flat_order": "chain-major：chain i 的 n_draws_per_chain 个 draw 连续存放",
           "flatten_roundtrip_exact": flat_exact}
    if div_per_chain is not None:
        out["div_by_chain"] = [int(x) for x in np.asarray(div_per_chain).ravel()]
    return out


def interpret(marg: dict, cond: dict, gate_ok: bool) -> dict:
    """§20.21 预注册判定：A 位置 / B 减半 / C 变窄 / D 中心不动。"""
    if not gate_ok:
        return {"tier": "indeterminate",
                "verdict": "本次条件化实验不可判",
                "not_say": "后验几何必然存在问题"}
    A = abs(cond["r"]["gap"]) <= 0.02
    B = abs(cond["r"]["gap"]) <= 0.5 * abs(marg["r"]["gap"])
    C = cond["r"]["width90"] <= 0.7 * marg["r"]["width90"]
    D = abs(cond["r"]["gap"] - marg["r"]["gap"]) < 0.01
    checks = {"A_gap_le_0.02": A, "B_gap_halved": B,
              "C_width_le_0.7x": C, "D_center_unchanged": D}
    if A:
        tier, can, cannot = ("recovered",
                             "n2 的边际不确定性对 r 的位置有重要贡献",
                             "n2 是唯一原因")
    elif B:
        tier, can, cannot = ("partial",
                             "固定 n2 把与真值的差距缩小了一半以上，但没回到真值",
                             "偏移已解决")
    elif C and D:
        tier, can, cannot = ("narrowed_only",
                             "n2 主要贡献不确定性，而不是后验中心偏移",
                             "r 已经可以可靠恢复")
    else:
        tier, can, cannot = ("persists",
                             "仅固定 n2 不足以解除偏移，其他参数或联合脊线仍重要",
                             "n2 与 r 无关")
    return {"tier": tier, "verdict": can, "not_say": cannot, "checks": checks}


def max_rel_diff(a: dict, b: dict) -> float:
    """两个样本字典的逐元素最大相对差（"必须逐位复现"的硬保护判据）。

    任何"不能证明相同"的情形都返回 `inf`——这个函数是 `check_control()` 的第一道闸，
    宁可误报不一致，也不能把异常数据当成 0（§20.24 审查意见）：

    · **键集合**必须相等：只查 `a` 的键会漏掉"`b` 多一个参数"（多出来的参数没人比对，
      于是照样返回 0）；
    · **形状**必须相等；
    · **两侧都必须全有限**：`nan` 会让 `np.max(d)` 变成 `nan`，而 `max(0.0, nan)`
      在 Python 里返回 `0.0`（`nan` 比较恒假），等于把"有 NaN"读成"完全一致"；
    · **比值也必须有限**：`inf/inf` 之类会得到 `nan`。
    """
    if set(a) != set(b):
        return float("inf")
    worst = 0.0
    for k in a:
        x = np.asarray(a[k], dtype=float)
        y = np.asarray(b[k], dtype=float)
        if x.shape != y.shape:
            return float("inf")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return float("inf")
        d = np.abs(x - y) / np.maximum(np.abs(y), 1e-30)
        if not np.all(np.isfinite(d)):
            return float("inf")
        worst = max(worst, float(np.max(d)))
    return worst


def _rel(p) -> str:
    """仓库相对路径（POSIX 分隔符）——归档里不再写本机绝对路径（§20.23 第四项）。"""
    try:
        return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(p).resolve().as_posix()


def _sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ 自检

def selftest() -> bool:
    ok, cases = True, []

    def chk(name, cond, extra=""):
        nonlocal ok
        cases.append((name, bool(cond), extra))
        ok = ok and bool(cond)

    # 1) condition 用约束空间值：与"把值写死"的参照相等，且不等于 logit 解释
    import jax
    import numpyro
    import numpyro.distributions as dist
    from numpyro import handlers
    from numpyro.infer.util import log_density

    DATA = jax.numpy.array(0.5)

    def mini():
        x = numpyro.sample("x", dist.Uniform(0.0, 1.0))
        numpyro.sample("obs", dist.Normal(x, 0.1), obs=DATA)

    def ref(val):
        def m():
            numpyro.sample("obs", dist.Normal(val, 0.1), obs=DATA)
        return m

    lp_c = float(log_density(handlers.condition(mini, data={"x": 0.85}), (), {}, {})[0])
    lp_r = float(log_density(ref(0.85), (), {}, {})[0])
    lp_logit = float(dist.Normal(jax.nn.sigmoid(0.85), 0.1).log_prob(0.5))
    chk("1 condition 取约束空间值", abs(lp_c - lp_r) < 1e-5 and abs(lp_c - lp_logit) > 1.0,
        f"{lp_c:.6f} vs 参照 {lp_r:.6f} / logit {lp_logit:.6f}")
    # 2) do 不是条件化：它保留同名采样位点（无 key 时仍尝试采样并报错）
    try:
        log_density(handlers.do(mini, data={"x": 0.85}), (), {}, {})
        do_failed = False
    except TypeError:
        do_failed = True
    chk("2 do 保留采样（不是条件后验）", do_failed)
    # 3) stats：合成后验的中位/区间可用，truth_pct 自洽
    rng = np.random.default_rng(0)
    n = 20000
    truth = dict(M.TRUTH)
    s = {"R1": np.exp(rng.normal(math.log(55), 0.3, n)),
         "R2": np.exp(rng.normal(math.log(600), 0.3, n)),
         "Q2": rng.normal(900, 90, n), "n1": rng.normal(0.75, 0.03, n),
         "Q1": rng.normal(200, 20, n), "n2": rng.normal(0.85, 0.02, n),
         "Rs": rng.normal(10, 1, n), "R_ad": rng.normal(3000, 100, n),
         "tau_ad": rng.normal(1.6, 0.05, n)}
    st = stats(s, truth)
    chk("3 stats 自洽", st["r"]["p05"] < st["r"]["median"] < st["r"]["p95"]
        and abs(st["sum_R1R2"]["rel_err"]) < 0.02)
    # 4) 判定五个分支
    mk = lambda med, w: {"r": {"median": med, "width90": w, "gap": med - 0.084}}
    marg = mk(0.0328, 0.0856)                   # §20.17 d0
    chk("4a 回到真值 → recovered", interpret(marg, mk(0.086, 0.08), True)["tier"] == "recovered")
    chk("4b 差距减半 → partial", interpret(marg, mk(0.062, 0.08), True)["tier"] == "partial")
    chk("4c 变窄但中心不动 → narrowed_only",
        interpret(marg, mk(0.036, 0.05), True)["tier"] == "narrowed_only")
    chk("4d 依旧低且不窄 → persists", interpret(marg, mk(0.030, 0.08), True)["tier"] == "persists")
    chk("4e 门禁未过 → indeterminate", interpret(marg, mk(0.086, 0.05), False)["tier"] == "indeterminate")
    # 5) 逐位复现检查：同数组 → 0；扰动 → >0
    a = {"R1": np.array([1.0, 2.0]), "R2": np.array([3.0, 4.0])}
    chk("5a 同数组 → 0", max_rel_diff(a, a) == 0.0)
    chk("5b 扰动 → >0", max_rel_diff(a, {"R1": np.array([1.0, 2.001]),
                                         "R2": np.array([3.0, 4.0])}) > 1e-5)
    # 5c-5e §20.24：键集合 / NaN / Inf 都必须判成不一致（返回 inf）
    chk("5c b 多一个键 → inf",
        max_rel_diff(a, dict(a, R3=np.zeros(2))) == float("inf"))
    chk("5d a 含 NaN → inf",
        max_rel_diff({"R1": np.array([1.0, np.nan])},
                     {"R1": np.array([1.0, np.nan])}) == float("inf"))
    chk("5e b 含 Inf → inf",
        max_rel_diff({"R1": np.array([1.0, 2.0])},
                     {"R1": np.array([1.0, np.inf])}) == float("inf"))
    # 6) 谱是可重建的（同 seed 两次一致），这是"不重新生成噪声"的前提
    t1 = build_data(0)[1:3]
    t2 = build_data(0)[1:3]
    chk("6 同 seed 谱可重建", np.array_equal(t1[0], t2[0]) and np.array_equal(t1[1], t2[1]))
    # 7) near_truth：边界内 3 个计入、边界外 2 个不计入（用 ±1e-9 内缩避开浮点刀口，
    #    因为 r 由 R1/(R1+R2) 现算，不可能与构造值逐位相等）
    rt = truth["R1"] / (truth["R1"] + truth["R2"])
    xs = np.array([rt - NEAR_TRUTH_TOL + 1e-9, rt - 0.0199,
                   rt + NEAR_TRUTH_TOL - 1e-9, rt + 0.0201, 0.5])
    xs = np.tile(xs, 40)                     # 复制到 200 个 draw，避免脊线几何退化
    s7 = {"R1": xs / (1.0 - xs), "R2": np.ones_like(xs)}
    n7 = stats(s7, truth)["r"]["near_truth"]
    chk("7 near_truth 边界（含等号）", abs(n7["prob"] - 0.6) < 1e-12
        and n7["tol"] == NEAR_TRUTH_TOL and n7["n_draws"] == 200,
        f"prob={n7['prob']}")
    # 8) 展平是 chain 数组的纯 reshape：round-trip 相等，chain_diag 给出逐链中位
    rng8 = np.random.default_rng(7)
    s_ch8 = {"R1": np.exp(rng8.normal(math.log(55), 0.3, (4, 500))),
             "R2": np.exp(rng8.normal(math.log(600), 0.1, (4, 500)))}
    flat8 = {k: v.reshape((-1,) + v.shape[2:]) for k, v in s_ch8.items()}
    rt8 = all(np.array_equal(np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:]),
                             flat8[k]) for k, v in s_ch8.items())
    cd8 = chain_diag(s_ch8, truth, div_per_chain=[0, 1, 0, 2], flat_exact=rt8)
    chk("8 链结构可无损恢复 + chain_diag", rt8 and cd8["n_chains"] == 4
        and cd8["n_draws_per_chain"] == 500 and len(cd8["chain_median_r"]) == 4
        and cd8["div_by_chain"] == [0, 1, 0, 2]
        and cd8["between_chain_sd_of_median"] > 0)
    # 9) 对照保护：可复现且过门禁才放行；不一致 / 未过门禁 / 归档未过门禁都中止
    ok9 = gate({"rhat_max": 1.005, "ess_min": 500, "div_frac": 0.0})
    bad9 = gate({"rhat_max": 1.02, "ess_min": 500, "div_frac": 0.0})
    def _raises(*a, **kw):
        try:
            check_control(*a, **kw)
            return False
        except ReproError:
            return True
    chk("9a 一致且过门禁 → 放行", not _raises(0.0, ok9, True))
    chk("9b 不一致 → 中止", _raises(1e-3, ok9, True))
    chk("9c 对照未过门禁 → 中止", _raises(0.0, bad9, True))
    chk("9d 归档未过门禁 → 中止", _raises(0.0, ok9, False))
    chk("9e NaN → 中止", _raises(float("nan"), ok9, True))
    # 10) 发散计数按链拆分：展平（链优先）与已是二维两种输入给同一答案
    flat = np.zeros(4 * 1200, dtype=bool)
    flat[[1200 + 3, 3600 + 7, 3600 + 99]] = True      # 链1 一次、链3 两次
    chk("10 div_by_chain 按链拆分", div_by_chain(flat, 4) == [0, 1, 0, 2]
        and div_by_chain(flat.reshape(4, 1200), 4) == [0, 1, 0, 2]
        and div_by_chain(None, 4) is None)

    for name, good, extra in cases:
        print(f"  [{'✓' if good else '✗'}] {name}{'' if good else '  ' + extra}")
    print(f"{'通过' if ok else '失败'}：{sum(1 for _, g, _ in cases if g)}/{len(cases)}")
    return ok


# ------------------------------------------------------------------ 报告

def _row(name: str, st: dict, gate_pass: bool | None) -> str:
    g = "✓" if gate_pass else ("✗" if gate_pass is False else "—")
    p = st["params"]
    q2 = p.get("Q2", {})
    return (f"{name:>8}{g:>4}{st['r']['median']:>9.4f}{st['r']['sd']:>8.4f}"
            f"{st['r']['p05']:>8.4f}{st['r']['p95']:>8.4f}{st['r']['width90']:>8.4f}"
            f"{st['r']['gap']:>+9.4f}{(q2.get('median', float('nan'))):>9.1f}"
            f"{st['sum_R1R2']['median']:>9.1f}{st['sum_R1R2']['rel_err']:>+8.1%}")


def report(ref: dict, ctrl: dict, cond: dict, vd_cond: dict, vd: dict,
           meta: dict) -> str:
    L = []
    L.append(f"条件化诊断（§20.21）：data_seed={meta['data_seed']}  "
             f"fix={meta['condition']}  采样器 seed={meta['seed']}")
    L.append(f"配置：{meta['cfg']['n_chains']} 链 × {meta['cfg']['n_samples']} 样本 / "
             f"warmup {meta['cfg']['n_warmup']} / ta {meta['cfg']['target_accept']} / "
             f"depth {meta['cfg']['max_tree_depth']}  谱 sha256[:16]={meta['spec_hash']}")
    L.append("**诊断性**：条件后验 ≠ 未知 n2 下的恢复性能；不纳入 M3.5 验收")
    L.append("")
    L.append(f"{'':>8}{'门禁':>4}{'r 中位':>9}{'sd_r':>8}{'p05':>8}{'p95':>8}"
             f"{'宽90':>8}{'gap':>9}{'Q2 中位':>9}{'R1+R2':>9}{'误差':>8}")
    L.append(_row("归档 d0", ref, meta["ref_gate"]))
    L.append(_row("对照重跑", ctrl, meta["ctrl_gate"]["pass"]))
    L.append(_row("条件 n2", cond, vd_cond["pass"]))
    L.append("")
    L.append(f"逐位复现：对照重跑 vs 归档 d0 最大相对差 = {meta['repro_max_rel']:.1e} "
             f"（门槛 {meta.get('repro_tol', REPRO_TOL):.0e}）"
             f"→ {'✓' if meta['repro_max_rel'] <= meta.get('repro_tol', REPRO_TOL) else '✗'}"
             f"；对照门禁 {'✓' if meta['ctrl_gate']['pass'] else '✗'}")
    L.append("")
    L.append("位置与不确定性")
    L.append(f"  gap：边缘 {ref['r']['gap']:+.4f} → 条件 {cond['r']['gap']:+.4f}"
             f"（变化 {cond['r']['gap'] - ref['r']['gap']:+.4f}）")
    L.append(f"  width90：边缘 {ref['r']['width90']:.4f} → 条件 {cond['r']['width90']:.4f}"
             f"（比值 {cond['r']['width90'] / ref['r']['width90']:.2f}）")
    L.append(f"  sd_r：{ref['r']['sd']:.4f} → {cond['r']['sd']:.4f}"
             f"（比值 {cond['r']['sd'] / ref['r']['sd']:.2f}）")
    L.append(f"  r 的 90% 区间：边缘 [{ref['r']['p05']:.4f}, {ref['r']['p95']:.4f}] → "
             f"条件 [{cond['r']['p05']:.4f}, {cond['r']['p95']:.4f}]；真值分位 "
             f"{ref['r']['truth_pct']:.2f} → {cond['r']['truth_pct']:.2f}")
    L.append(f"  P(|r − r*| ≤ {NEAR_TRUTH_TOL})：边缘 {ref['r']['near_truth']['prob']:.1%} → "
             f"条件 {cond['r']['near_truth']['prob']:.1%}（派生统计量，已随 JSON 归档）")
    L.append("")
    L.append("相关结构（审查要求：相关性是否从 n2 转到 Q2）")
    for k in ("r~Q2", "r~n1", "r~Q1", "r~Rs", "r~R_ad", "Q2~n1", "Q2~Q1", "n1~Q1"):
        a = ref["corr"].get(k)
        b = cond["corr"].get(k)
        if a is None or b is None:
            continue
        L.append(f"  {k:<7} 边缘 {a:+.3f} → 条件 {b:+.3f}（Δ {b - a:+.3f}）")
    L.append("")
    L.append("脊线几何（(log R1, log R2)）")
    for tag, st in (("边缘", ref), ("条件", cond)):
        rg = st["ridge"]
        L.append(f"  {tag}：拉长比 {rg['elongation']:.1f}×  沿脊线 {rg['along_in_sd']:+.2f}σ  "
                 f"跨脊线 {rg['across_in_sd']:+.2f}σ  夹角 {rg['offset_angle_deg']:.0f}°")
    L.append("")
    L.append("参数（条件后验里 n2 不参与采样）")
    keys = [k for k in ("Q2", "n1", "Q1", "R1", "R2", "R_ad", "tau_ad", "Rs")
            if k in cond["params"]]
    L.append(f"  {'参数':<7}{'边缘中位':>11}{'条件中位':>11}{'变化':>9}"
             f"{'边缘宽90':>10}{'条件宽90':>10}{'比值':>7}")
    for k in keys:
        a, b = ref["params"].get(k), cond["params"].get(k)
        if not a or not b:
            continue
        L.append(f"  {k:<7}{a['median']:>11.4g}{b['median']:>11.4g}"
                 f"{(b['median'] / a['median'] - 1):>+8.1%}"
                 f"{a['width90']:>10.4g}{b['width90']:>10.4g}"
                 f"{(b['width90'] / a['width90'] if a['width90'] else float('nan')):>7.2f}")
    L.append("")
    L.append(f"判定：`{vd['tier']}` —— 可说：{vd['verdict']}；不可说：{vd['not_say']}")
    if "checks" in vd:
        L.append(f"      判据 {vd['checks']}")
    L.append(f"条件后验门禁：R̂max {cond['_gate']['rhat_max']:.4f}、"
             f"ESSmin {cond['_gate']['ess_min']:.0f}、div {cond['_gate']['divergences']}"
             f" → {'通过' if vd_cond['pass'] else '未过'}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M3.5 n2 条件化诊断（§20.21，诊断性）")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--fix", default="n2=0.85", help="形如 n2=0.85（约束空间值）")
    ap.add_argument("--seed", type=int, default=0, help="采样器 seed")
    ap.add_argument("--ref", default=str(HERE / "m35_r_w5000.json"),
                    help="归档记录（读 d0 的门禁与对照读数）")
    ap.add_argument("--json", default=str(HERE / "m35_cond_n2.json"))
    ap.add_argument("--warmup", type=int, default=CFG["n_warmup"])
    ap.add_argument("--save-samples", action="store_true",
                    help="把两跑的后验样本存成 npz（<json>.ctrl / .cond）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        print("§20.21 诊断链自检（含 condition 语义）")
        return 0 if selftest() else 1

    k, _, v = args.fix.partition("=")
    condition = {k.strip(): float(v)}
    cfg = dict(CFG, n_warmup=int(args.warmup))
    truth, f, Z, zmax, sigma, spec_hash = build_data(args.data_seed)
    print(f"谱：data_seed={args.data_seed}  N={f.size}  zmax={zmax:.1f}  "
          f"sha256[:16]={spec_hash}")
    print(f"条件：{condition}（约束空间）；其余参数照采样")

    # 归档读数：从 §20.17 的 npz 取（同一份谱的边际后验）
    ref_json = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    rec0 = next((r for r in ref_json["records"] if r["data_seed"] == args.data_seed), None)
    if rec0 is None:
        print(f"!! {args.ref} 里没有 data_seed={args.data_seed} 的记录")
        return 2
    npz = Path(args.ref).with_suffix("")
    npz = npz.parent / f"{npz.name}.{rec0['stage']}.d{args.data_seed}.s{rec0['sampler_seed']}.samples.npz"
    if not npz.exists():
        print(f"!! 缺归档样本 {_rel(npz)}（需要 §20.17 的 --save-samples）")
        return 2
    zref = np.load(npz)
    s_ref = {kk: zref[kk] for kk in zref.files}
    ref = stats(s_ref, truth)

    print("\n[1/2] 对照重跑（不加条件，必须与归档逐位复现且过门禁）…")
    ctrl = fit(f, Z, sigma=sigma, condition=None, cfg=cfg, seed=args.seed)
    repro = max_rel_diff(ctrl["s"], s_ref)
    g_ctrl = gate(ctrl)
    print(f"      最大相对差 {repro:.1e}；对照门禁 R̂max {ctrl['rhat_max']:.4f}、"
          f"ESSmin {ctrl['ess_min']:.0f} → {'通过' if g_ctrl['pass'] else '未过'}")
    try:
        check_control(repro, g_ctrl, bool(rec0["run_valid"]["pass"]))
    except ReproError as exc:
        # 不产出判定：对照不成立时，条件与边缘之差没有可读含义（§20.23 审查要求）
        print(f"!! 中止：{exc}")
        if args.json:
            Path(args.json).write_text(json.dumps({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "aborted": True, "reason": str(exc),
                "diagnostic_only": True, "not_recovery_performance": True,
                "meta": {"data_seed": args.data_seed, "condition": condition,
                         "seed": args.seed, "cfg": cfg, "spec_hash": spec_hash,
                         "repro_max_rel": repro},
                "gate_control": g_ctrl,
                "marginal": ref, "control": stats(ctrl["s"], truth),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"中止记录写入 {args.json}（无 verdict 字段）")
        return 3

    print(f"[2/2] 条件采样 {condition} …")
    cond = fit(f, Z, sigma=sigma, condition=condition, cfg=cfg, seed=args.seed)

    st_ref, st_ctrl, st_cond = ref, stats(ctrl["s"], truth), stats(cond["s"], truth)
    g_cond = gate(cond)
    vd = interpret(st_ref, st_cond, g_cond["pass"])
    st_cond["_gate"] = {"rhat_max": cond["rhat_max"], "ess_min": cond["ess_min"],
                        "divergences": cond["divergences"]}

    # 后验样本落盘：没有样本就只能读汇总数，没法复核
    saved, sums = {}, {}
    if args.save_samples and args.json:
        base = Path(args.json).with_suffix("")
        for tag, rec in (("ctrl", ctrl), ("cond", cond)):
            p = Path(f"{base}.{tag}.samples.npz")
            np.savez_compressed(p, **{k: np.asarray(v) for k, v in rec["s"].items()})
            saved[tag] = _rel(p)
            sums[tag] = {"sha256": _sha256(p), "bytes": p.stat().st_size,
                         "keys": sorted(rec["s"]),
                         "n_draws": int(np.asarray(rec["s"]["R1"]).size),
                         "n_chains": rec["n_chains"],
                         "n_draws_per_chain": rec["n_draws_per_chain"]}
    meta = {"data_seed": args.data_seed, "condition": condition, "seed": args.seed,
            "cfg": cfg, "spec_hash": spec_hash,
            "ref_gate": bool(rec0["run_valid"]["pass"]),
            "ctrl_gate": g_ctrl, "repro_max_rel": repro,
            "repro_tol": REPRO_TOL}
    cd_ctrl = chain_diag(ctrl["s_ch"], truth, div_per_chain=ctrl["div_by_chain"],
                         flat_exact=ctrl["flat_exact"])
    cd_cond = chain_diag(cond["s_ch"], truth, div_per_chain=cond["div_by_chain"],
                         flat_exact=cond["flat_exact"])
    print()
    print(report(st_ref, st_ctrl, st_cond, g_cond, vd, meta))
    for tag, cd in (("对照", cd_ctrl), ("条件", cd_cond)):
        print(f"链级（{tag}）：逐链 r 中位 {[round(x, 4) for x in cd['chain_median_r']]}"
              f"  链间 sd {cd['between_chain_sd_of_median']:.4f}"
              f"  极差 {cd['gap_spread_across_chains']:.4f}"
              f"  div/链 {cd.get('div_by_chain')}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "diagnostic_only": True,
            "not_recovery_performance": True,
            "note": "条件后验 p(r,Q2,…|Z,n2=n2_true) 与边缘后验的差反映'已知 n2'的影响，"
                    "不是 n2 单向致因的证明，也不能当未知 n2 下的恢复性能",
            "meta": meta, "archive_ref": _rel(npz),
            "samples": saved, "samples_sha256": sums,
            "marginal": st_ref, "control": st_ctrl, "conditional": st_cond,
            "chain_diag_control": cd_ctrl, "chain_diag_conditional": cd_cond,
            "gate_control": g_ctrl, "gate_conditional": g_cond,
            "verdict": vd,
            "config": {"branch": BRANCH, "layers": LAYERS, "sigma_mode": cond["sigma_mode"],
                       "seconds_control": ctrl["seconds"], "seconds_conditional": cond["seconds"]},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"记录写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
