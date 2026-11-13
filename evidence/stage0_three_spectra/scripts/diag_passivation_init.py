"""钝化谱的初值诊断：同一拓扑、同一噪声口径，只换初值。

目的：区分「拓扑表达不了」与「采样找不到」。
只读诊断，不改生产代码、不改先验、不改判据。
"""
import sys, json, time
import numpy as np

sys.path.insert(0, "D:/eis训练数据/bmdrt")
import bm_top as T
import bm_real as R
from bm_select import BUDGETS

PATH = (r"D:/eis训练数据/测试数据/钝化/Passivating_Generated EIS data and Nyquist plots/"
        r"Generated EIS data and Nyquist plots/[P1,P2-R3]-R4-[1002185.2194945764, 0.782449807, "
        r"1873214.9711620852, 0.5002444313837133, 9277752.99835045, 141.24023283906809].csv")
TRUTH_DATA = dict(Rs=141.24023283906809, R1=9277752.99835045,
                  Q1=1002185.2194945764, n1=0.782449807,
                  Q_b=1873214.9711620852, n_b=0.5002444313837133)


def rmsrel(f, Z, Zm):
    d = (Zm - Z) / np.abs(Z)
    return float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2))) * 100


def acf1(x):
    x = np.asarray(x, float) - np.mean(x)
    d = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / d) if d > 0 else 0.0


f, Z = R.load_spectrum(PATH)
f = np.asarray(f, float); Z = np.asarray(Z, complex)
zmax = float(np.abs(Z).max())
w_ref = 2.0 * np.pi * np.sqrt(f.min() * f.max())
s = 0.02 * np.abs(Z)
sigma = np.stack([s, s])

# 真值 → 项目 Q 约定（Q = |Z(w_ref)|）
truth = dict(TRUTH_DATA)
truth["Q1"] /= w_ref ** truth["n1"]
truth["Q_b"] /= w_ref ** truth["n_b"]
Zt = T.z_nested(f, truth, branch="capacitive", layers=1)
print(f"噪声底（数据 vs 真值拓扑正演）: rms_rel = {rmsrel(f, Z, Zt):.2f}%")

kw = {k: v for k, v in BUDGETS["standard"].items()
      if k in ("n_warmup", "n_samples", "num_chains", "max_tree_depth")}
kw["target_accept"] = 0.98
print("采样:", kw)

res = {}
for tag, init in [("truth_init", T.physical_to_unconstrained(
        truth, zmax=zmax, w_ref=w_ref, branch="capacitive", layers=1,
        sigma=(1.0, 1.0), num_chains=kw["num_chains"]))]:
    t0 = time.time()
    o = T.fit_top(f, Z, branch="capacitive", layers=1, sigma=sigma,
                  sigma_from_data=True, init_values=init, progress=False,
                  seed=0, **kw)
    dt = time.time() - t0
    smp = o["s"]
    med = {k: float(np.median(smp[k])) for k in T.sampled_keys("capacitive", 1)}
    Zm = T.z_nested(f, med, branch="capacitive", layers=1)
    res[tag] = dict(
        seconds=round(dt, 1), rhat_max=float(o["rhat_max"]),
        ess_min=float(o["ess_min"]), divergences=int(o["divergences"]),
        rms_rel=round(rmsrel(f, Z, Zm), 3),
        acf=round(max(acf1((Zm - Z).real), acf1((Zm - Z).imag)), 3),
        median=med,
        truth={k: float(truth[k]) for k in med if k in truth},
        curve=dict(f=f.tolist(), Zre=Z.real.tolist(), Zim=Z.imag.tolist(),
                   Mre=Zm.real.tolist(), Mim=Zm.imag.tolist(),
                   Tre=Zt.real.tolist(), Tim=Zt.imag.tolist()))
    r = res[tag]
    print(f"\n[{tag}] {dt:.1f}s  Rhat_max={r['rhat_max']:.3f}  ESS_min={r['ess_min']:.0f}  "
          f"div={r['divergences']}  rms_rel={r['rms_rel']:.2f}%  acf={r['acf']:.3f}")
    print(f"  {'param':6} {'拟合中位':>12} {'真值':>12} {'比值':>9}")
    for k, v in med.items():
        tv = truth.get(k)
        ratio = (v / tv) if tv else float("nan")
        print(f"  {k:6} {v:12.4g} {(tv if tv else float('nan')):12.4g} {ratio:9.3f}")

out = "D:/eis训练数据/bmdrt/out/_user_fit_nested/diag_passivation_init.json"
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n写入", out)
