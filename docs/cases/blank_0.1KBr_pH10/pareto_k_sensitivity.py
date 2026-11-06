"""高 Pareto-k 频点的敏感性复核（只读）。

审查意见：不能只看「超阈点占比 < 10%」。本脚本对每个配对：
  1) 全频点计算 Delta / dse / 判定；
  2) 标出任一模型 pareto_k > 0.7 的频点并给出其逐点贡献；
  3) 去掉这些点后重算 Delta' / dse' / 判定，看结论是否变化。

注意：这是「删除高 k 点」的敏感性复核，不是 moment matching / 精确 LOO。
机制 ambiguous 仍应标注为探索性证据，待更严格的复核后才能当严格统计结论。

运行：
  D:/autoeis_env/Scripts/python.exe pareto_k_sensitivity.py
输出：pareto_k_sensitivity.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PK_LIMIT = 0.7
ELPD_ABS, ELPD_K = 2.0, 2.0


def calc(d, ctype):
    n = len(d)
    delta = float(d.sum())
    dse = float(np.sqrt(n) * d.std(ddof=1)) if n > 1 else float("nan")
    thr = max(ELPD_ABS, ELPD_K * dse) if ctype == "K_selection" else dse
    if ctype == "K_selection":
        verdict = "K2_better" if delta > thr else "not_significant"
    else:
        verdict = "ambiguous" if abs(delta) < dse else "preference"
    return n, delta, dse, thr, verdict


def main():
    df = pd.read_csv(ROOT / "pointwise_loo.csv")
    elp = df.pivot(index="f_Hz", columns="model", values="pointwise_elpd")
    pk = df.pivot(index="f_Hz", columns="model", values="pareto_k")
    pairs = pd.read_csv(ROOT / "pairwise_dse.csv")
    rows = []
    for _, r in pairs.iterrows():
        a, b = r.model_a, r.model_b
        if a not in elp.columns or b not in elp.columns or a == b:
            continue
        # pivot 后 index 是 10 个模型频率的并集（合成用例点数不同），
        # 必须只保留两模型都有的频点，否则 len(d) 会把 NaN 行算进 N，dse 被放大。
        d = (elp[a] - elp[b]).dropna()
        bad = ((pk[a] > PK_LIMIT) | (pk[b] > PK_LIMIT)).reindex(d.index).fillna(False)
        n0, delta0, dse0, thr0, v0 = calc(d, r.comparison_type)
        kept = d[~bad]
        n1, delta1, dse1, thr1, v1 = calc(kept, r.comparison_type)
        bad_fs = [float(f) for f in d.index[bad]]
        rows.append(dict(
            pair=r.pair, comparison_type=r.comparison_type,
            n_all=n0, delta_all=delta0, dse_all=dse0, thr_all=thr0,
            verdict_all=v0,
            n_high_pk=int(bad.sum()), high_pk_freqs=",".join(f"{x:.4g}" for x in bad_fs),
            high_pk_contribution=float(d[bad].sum()) if bad.any() else 0.0,
            n_kept=n1, delta_kept=delta1, dse_kept=dse1, thr_kept=thr1,
            verdict_kept=v1,
            conclusion_unchanged=(v0 == v1),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "pareto_k_sensitivity.csv", index=False,
               float_format="%.6g")
    print(out[["pair", "n_high_pk", "high_pk_contribution", "delta_all",
               "delta_kept", "dse_all", "dse_kept", "verdict_all",
               "verdict_kept", "conclusion_unchanged"]].to_string(index=False))
    print("\nDONE ->", ROOT / "pareto_k_sensitivity.csv")


if __name__ == "__main__":
    main()
