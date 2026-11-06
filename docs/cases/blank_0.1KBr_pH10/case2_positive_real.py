"""用例 2 正向：真实 NUTS 谱上让 K=1/K=2 都合格且 Δ 不显著。

审查员建议「固定噪声模型或调整参数化」：本脚本只在**测试设置**里
通过 `bm_core.fit(..., sigma=(s,s))` 固定噪声标准差（不采 σ，从而消除
近无噪谱的 σ→0 漏斗），**不修改任何算法代码 / 先验 / 阈值**。
该设置改变的是似然的噪声处理方式，须单独报告它是否影响模型选择。

谱：两个相近 HN 峰（1000 / 700 Hz）+ 2% 相对噪声；KK 门禁检查；
K=1、K=2 各跑 3 链 NUTS；从后验样本重算逐频点 LOO 得到配对 Δ / dse。

判定：两候选 grade 都 >=2（R̂<=1.01、bulk ESS>=400、div=0）、KK=ok，
且 Δ <= max(2, 2*dse) → K_SELECTION_AMBIGUOUS 正向触发成立。

运行：
  D:/autoeis_env/Scripts/python.exe case2_positive_real.py
输出：case2_positive.csv
"""
import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:/eis训练数据/bmdrt")
sys.path.insert(0, str(REPO))
import bm_core as B  # noqa: E402
import bm_kk as KK  # noqa: E402
import bm_real as R  # noqa: E402
import v2_state_machine as SM  # noqa: E402

F = np.logspace(np.log10(0.1), np.log10(1e5), 50)
NOISE = 0.02
SEED = 71
# (第二峰 f0 / Hz, 第二分量幅度 w2 / Ω)；W=60 固定，w1=60-w2
GRID = [(100.0, 5.0), (300.0, 8.0), (500.0, 10.0), (700.0, 4.0)]


def run_one(f0_2, w2, out_rows):
    p = dict(R_inf=20.0, W=60.0, pi=np.array([(60.0 - w2) / 60.0, w2 / 60.0]),
             tau=np.array([1 / (2 * np.pi * 1000.0), 1 / (2 * np.pi * f0_2)]),
             phi=np.array([0.85, 0.80]), zeta=np.ones(2))
    Z0 = B.model_z(F, p, include_L=False)
    mag = np.abs(Z0)
    rng = np.random.default_rng(SEED)
    Z = Z0 + NOISE * mag * (rng.standard_normal(F.size)
                            + 1j * rng.standard_normal(F.size)) / np.sqrt(2)
    zmax = float(np.max(np.abs(Z)))
    sigma = (NOISE * zmax, NOISE * zmax)

    kk = KK.kk_check(F, Z)
    kk_status = kk.get("status")
    diag_rows, elpd = [], {}
    for K in (1, 2):
        s, mc = B.fit(F, Z, K=K, alpha=0.5, sigma=sigma, include_L=False,
                      zeta_fixed=1.0, n_warmup=1500, n_samples=1000,
                      num_chains=3, seed=0, progress=False)
        n = len(s["R_inf"])
        s["s_re"] = np.full(n, sigma[0])
        s["s_im"] = np.full(n, sigma[1])
        ch = mc.get_samples(group_by_chain=True)
        post = {k: np.asarray(v) for k, v in ch.items()}
        idata = az.from_dict(posterior=post)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rhat = float(np.nanmax(az.rhat(idata).to_array().values))
            ess = float(np.nanmin(az.ess(idata, method="bulk").to_array().values))
        div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
        ll = R.loglik_matrix(s, F, Z, include_L=False, n_max=800)
        li = az.from_dict(posterior={"dummy": np.zeros((1, ll.shape[0], 1))},
                          log_likelihood={"obs": ll[None, :, :]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            elpd[K] = np.asarray(az.loo(li, pointwise=True).loo_i.values).ravel()
        grade = 2 if (rhat <= 1.01 and ess >= 400 and div == 0) else 0
        diag_rows.append(dict(K=K, rhat_max=rhat, ess_bulk_min=ess,
                              divergences=div, grade=grade,
                              fixed_sigma=sigma[0], n_observations=F.size))
    d = elpd[2] - elpd[1]
    N = len(d)
    delta, dse = float(d.sum()), float(np.sqrt(N) * d.std(ddof=1))
    thr = max(2.0, 2.0 * dse)
    diag = pd.DataFrame(diag_rows)
    all_good = bool((diag.grade >= 2).all())
    kk_ok = (kk_status == "ok")
    positive = bool(all_good and kk_ok and delta <= thr
                    and np.isfinite(delta) and np.isfinite(dse))

    prec = dict(delta=delta, dse=dse, thr=thr, n_observations=N,
                comparison_type="K_selection", K_a=2, K_b=1,
                model_a="K2", model_b="K1", direction="A>B")
    cand = dict(elpd=0.0, grade=2, div=0, cap=0.0, rhat=1.001,
                bulk=1000, tail=1200)
    sm = SM.decide("ok" if kk_ok else kk_status,
                   {1: dict(cand), 2: dict(cand, elpd=delta)}, {}, 2,
                   [], ["none"], [], [], pairwise_comparisons={(2, 1): prec})
    out_rows.append(dict(
        case="case2_positive_real",
        f0s=f"1000/{f0_2:g} Hz", w2=w2, noise=NOISE, fixed_sigma=sigma[0],
        kk_status=kk_status, kk_ok=kk_ok,
        grade_K1=int(diag.loc[diag.K == 1, "grade"].iloc[0]),
        grade_K2=int(diag.loc[diag.K == 2, "grade"].iloc[0]),
        rhat_max=float(diag.rhat_max.max()),
        ess_bulk_min=float(diag.ess_bulk_min.min()),
        divergences=int(diag.divergences.sum()),
        all_candidates_qualified=all_good,
        n_observations=N, delta=delta, dse=dse, thr=thr,
        delta_not_significant=bool(delta <= thr),
        sm_k_selection_status=sm["k_selection_status"],
        positive_case_established=positive,
        note="固定噪声仅为测试设置，未改算法代码/先验/阈值",
    ))
    print(diag.to_string(index=False))
    print("  KK:", kk_status, "| Δ=%.3f dse=%.3f thr=%.3f | 状态机: %s | 正向: %s"
          % (delta, dse, thr, sm["k_selection_status"], positive))
    return positive


def main():
    out_rows = []
    for f0_2, w2 in GRID:
        print("=== f0_2=%g Hz, w2=%g Ω ===" % (f0_2, w2))
        if run_one(f0_2, w2, out_rows):
            print("  -> 找到正向用例，停止搜索")
            break
    out = pd.DataFrame(out_rows)
    out.to_csv(ROOT / "case2_positive.csv", index=False, float_format="%.6g")
    print("\n正向成立:", bool(out.positive_case_established.any()))
    print("DONE ->", ROOT / "case2_positive.csv")


if __name__ == "__main__":
    main()
