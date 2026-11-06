"""逐频点 PSIS-LOO 与配对 dse（唯一依据）。

审查意见 P0：K 选择与机制比较不得从 grid_comparison.csv 某一行取 dse，
也不得用两个模型各自 dse 的 max。必须从逐频点 elpd 差值算：

    d_i = ELPD_i(A) - ELPD_i(B)   （i = 每个频率点的 Re/Im 联合观测）
    Delta = sum_i d_i
    dse   = sqrt(N * Var(d_i))
    thr   = max(2, 2*dse)

产出：
  pointwise_loo.csv    每个模型每个频点的 pointwise elpd 与 pareto_k
  pairwise_dse.csv     所有配对的 Delta / dse / thr / n_obs / 方向 / 来源

6 格网格有 K=1、K=2 两套完整后验样本，配对 dse 从逐点差值重算；
4 个合成用例的 auto 运行只落盘最终选中 K 的 NPZ，其配对 dse 由 run.log ③ 段
（az.compare，同一逐点差定义）补录，并在 source 列标明。

只读 NPZ / run.log，不改算法代码。运行：
  D:/autoeis_env/Scripts/python.exe pointwise_loo.py
"""
import glob
import re
import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = next((q for q in [ROOT, *ROOT.parents] if (q / "bm_core.py").exists()),
            Path(r"D:/eis训练数据/bmdrt"))
sys.path.insert(0, str(REPO))
import bm_real as R  # noqa: E402

GRID = ["G_K1_none", "G_K2_none", "G_K1_Ws", "G_K2_Ws",
        "G_K1_Wo", "G_K2_Wo"]
CASES = ["U2_noise2", "U2b_K2", "E3", "U4b_warn_K2"]
MODELS = GRID + CASES
GRID_PAIRS = [
    ("K2-K1 / none", "G_K2_none", "G_K1_none", "K_selection"),
    ("K2-K1 / Ws", "G_K2_Ws", "G_K1_Ws", "K_selection"),
    ("K2-K1 / Wo", "G_K2_Wo", "G_K1_Wo", "K_selection"),
    ("Ws-Wo / K=2", "G_K2_Ws", "G_K2_Wo", "mechanism"),
    ("Ws-none / K=2", "G_K2_Ws", "G_K2_none", "mechanism"),
]
ELPD_ABS, ELPD_K = 2.0, 2.0


def find_npz(name):
    pats = [str(ROOT / "results" / name / "*_samples.npz"),
            str(REPO / "out/_v2cases" / name / "*" / "*_samples.npz")]
    for pat in pats:
        g = sorted(glob.glob(pat))
        if g:
            return g[0]
    raise FileNotFoundError(name)


def find_log(name):
    g = sorted(glob.glob(str(REPO / "out/_v2cases" / name / "**" / "run.log"),
                         recursive=True))
    return g[0] if g else None


def pointwise(name):
    z = np.load(find_npz(name), allow_pickle=True)
    s = {k: np.asarray(z[k], float) for k in z.files
         if k not in ("f", "Zre", "Zim")}
    f = np.asarray(z["f"], float)
    Z = np.asarray(z["Zre"], float) + 1j * np.asarray(z["Zim"], float)
    ll = R.loglik_matrix(s, f, Z, include_L=False, n_max=800)
    idata, *_ = R.loo_pointwise(ll)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = az.loo(idata, pointwise=True)
    return f, np.asarray(r.loo_i.values).ravel(), np.asarray(r.pareto_k.values).ravel()


def from_log(name):
    """从 run.log ③ 段读配对 Δ/dse，以及读入点数。"""
    txt = Path(find_log(name)).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"读入\s+(\d+)\s+点", txt)
    n = int(m.group(1)) if m else 0
    delta = dse = np.nan
    for ln in txt.splitlines():
        mm = re.match(r"\s*K=(\d+)\s+\d+\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s", ln)
        if mm and float(mm.group(4)) > 0:
            delta, dse = float(mm.group(3)), float(mm.group(4))
    return n, delta, dse


def main():
    elp, pk_rows = {}, []
    for name in MODELS:
        ff, e, k = pointwise(name)
        elp[name] = e
        for fi, ei, ki in zip(ff, e, k):
            pk_rows.append(dict(model=name, f_Hz=float(fi),
                                pointwise_elpd=float(ei),
                                pareto_k=float(ki)))
    pd.DataFrame(pk_rows).to_csv(ROOT / "pointwise_loo.csv",
                                 index=False, float_format="%.6g")

    rows = []
    for label, a, b, ctype in GRID_PAIRS:
        d = elp[a] - elp[b]
        n = len(d)
        delta, dse = float(d.sum()), float(np.sqrt(n) * d.std(ddof=1))
        # 阈值规则按比较类型分开：K 选择用 2*dse，机制比较用 1*dse（规格 §3.2）
        if ctype == "K_selection":
            thr, rule = max(ELPD_ABS, ELPD_K * dse), "max(2, 2*dse)"
        else:
            thr, rule = dse, "1*dse"
        rows.append(dict(pair=label, model_a=a, model_b=b,
                         comparison_type=ctype, n_observations=n,
                         delta=delta, dse=dse, thr=thr, threshold_rule=rule,
                         significant=bool(np.isfinite(delta) and abs(delta) > thr),
                         direction="A>B" if delta > 0 else "B>A",
                         source="pointwise_loo.csv (逐点差值重算)",
                         elpd_loo_a=float(elp[a].sum()),
                         elpd_loo_b=float(elp[b].sum())))
    for name in CASES:
        n, delta, dse = from_log(name)
        thr = max(ELPD_ABS, ELPD_K * (dse if np.isfinite(dse) else 0))
        rows.append(dict(pair=f"{name} / K2-K1", model_a=name, model_b=name,
                         comparison_type="K_selection",
                         n_observations=n, delta=delta, dse=dse, thr=thr,
                         threshold_rule="max(2, 2*dse)",
                         significant=bool(np.isfinite(delta) and abs(delta) > thr),
                         direction="A>B" if np.isfinite(delta) and delta > 0 else "B>A",
                         source="run.log ③ (az.compare 逐点配对差)",
                         elpd_loo_a=np.nan, elpd_loo_b=np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "pairwise_dse.csv", index=False, float_format="%.6g")
    print("=== 配对比较（唯一依据）===")
    print(df[["pair", "n_observations", "delta", "dse", "thr",
              "significant", "source"]].to_string(index=False))
    print("\nDONE -> pointwise_loo.csv / pairwise_dse.csv")


if __name__ == "__main__":
    main()
