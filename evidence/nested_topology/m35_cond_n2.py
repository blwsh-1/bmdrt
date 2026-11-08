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
   `control` 这一跑必须与归档的 `d0` 复现，逐位不一致就直接失败退出。

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
    st = summary(s_ch, prob=0.9, group_by_chain=True)
    rhat_map = {k: float(np.nanmax(np.asarray(v["r_hat"]))) for k, v in st.items()}
    ess_map = {k: float(np.nanmin(np.asarray(v["n_eff"]))) for k, v in st.items()}
    div = mcmc.get_extra_fields().get("diverging")
    n_draws = int(np.asarray(s["R1"]).size)
    divs = int(np.sum(div)) if div is not None else None
    return {"s": s, "s_ch": s_ch, "seconds": round(dt, 1),
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


# ------------------------------------------------------------------ 统计

def stats(s: dict, truth: dict) -> dict:
    """后验读数。`r` 逐 draw 算；同时带上审查点名要看的 Q2/n1/Q1 与 R1+R2。"""
    g = lambda k: np.asarray(s[k], float).ravel()
    R1, R2 = g("R1"), g("R2")
    r = R1 / (R1 + R2)
    r_truth = truth["R1"] / (truth["R1"] + truth["R2"])
    lo, med, hi = np.percentile(r, [5, 50, 95])
    out = {"r": {"median": float(med), "sd": float(np.std(r, ddof=1)),
                 "p05": float(lo), "p95": float(hi), "width90": float(hi - lo),
                 "mean": float(r.mean()), "truth": r_truth,
                 "gap": float(med - r_truth),
                 "truth_pct": float((r < r_truth).mean()),
                 "covered": bool(lo <= r_truth <= hi)},
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
    """两个样本字典的逐元素最大相对差（用于"必须逐位复现"的自检）。"""
    worst = 0.0
    for k in a:
        if k not in b:
            return float("inf")
        x, y = np.asarray(a[k], float), np.asarray(b[k], float)
        if x.shape != y.shape:
            return float("inf")
        d = np.abs(x - y) / np.maximum(np.abs(y), 1e-30)
        worst = max(worst, float(np.max(d)))
    return worst


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
    # 6) 谱是可重建的（同 seed 两次一致），这是"不重新生成噪声"的前提
    t1 = build_data(0)[1:3]
    t2 = build_data(0)[1:3]
    chk("6 同 seed 谱可重建", np.array_equal(t1[0], t2[0]) and np.array_equal(t1[1], t2[1]))

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
             f"→ {'✓' if meta['repro_max_rel'] <= 1e-9 else '✗ 不可比，后面的差值不能读'}")
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
        print(f"!! 缺归档样本 {npz}（需要 §20.17 的 --save-samples）")
        return 2
    zref = np.load(npz)
    s_ref = {kk: zref[kk] for kk in zref.files}
    ref = stats(s_ref, truth)

    print("\n[1/2] 对照重跑（不加条件，必须与归档逐位复现）…")
    ctrl = fit(f, Z, sigma=sigma, condition=None, cfg=cfg, seed=args.seed)
    repro = max_rel_diff(ctrl["s"], s_ref)
    print(f"      最大相对差 {repro:.1e}  → {'✓' if repro <= 1e-9 else '✗'}")

    print(f"[2/2] 条件采样 {condition} …")
    cond = fit(f, Z, sigma=sigma, condition=condition, cfg=cfg, seed=args.seed)

    st_ref, st_ctrl, st_cond = ref, stats(ctrl["s"], truth), stats(cond["s"], truth)
    g_ctrl, g_cond = gate(ctrl), gate(cond)
    vd = interpret(st_ref, st_cond, g_cond["pass"])
    st_cond["_gate"] = {"rhat_max": cond["rhat_max"], "ess_min": cond["ess_min"],
                        "divergences": cond["divergences"]}

    # 后验样本落盘：没有样本就只能读汇总数，没法复核
    saved = {}
    if args.save_samples and args.json:
        base = Path(args.json).with_suffix("")
        for tag, rec in (("ctrl", ctrl), ("cond", cond)):
            p = Path(f"{base}.{tag}.samples.npz")
            np.savez_compressed(p, **{k: np.asarray(v) for k, v in rec["s"].items()})
            saved[tag] = str(p)
    meta = {"data_seed": args.data_seed, "condition": condition, "seed": args.seed,
            "cfg": cfg, "spec_hash": spec_hash,
            "ref_gate": bool(rec0["run_valid"]["pass"]),
            "ctrl_gate": g_ctrl, "repro_max_rel": repro}
    print()
    print(report(st_ref, st_ctrl, st_cond, g_cond, vd, meta))

    if args.json:
        Path(args.json).write_text(json.dumps({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "diagnostic_only": True,
            "not_recovery_performance": True,
            "note": "条件后验 p(r,Q2,…|Z,n2=n2_true) 与边缘后验的差反映'已知 n2'的影响，"
                    "不是 n2 单向致因的证明，也不能当未知 n2 下的恢复性能",
            "meta": meta, "archive_ref": str(npz), "samples": saved,
            "marginal": st_ref, "control": st_ctrl, "conditional": st_cond,
            "gate_control": g_ctrl, "gate_conditional": g_cond,
            "verdict": vd,
            "config": {"branch": BRANCH, "layers": LAYERS, "sigma_mode": cond["sigma_mode"],
                       "seconds_control": ctrl["seconds"], "seconds_conditional": cond["seconds"]},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"记录写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
