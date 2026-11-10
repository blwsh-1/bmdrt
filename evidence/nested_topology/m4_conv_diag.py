"""A3 诊断：逐点相对 σ 口径下，nested 的 R̂ 卡在哪个参数上（§9.4.21.3）。

结论先写在这里，跑完对不上就改这段而不是改数字。

用法
----
    python evidence/nested_topology/m4_conv_diag.py                    # 2% / 1% 两档
    python evidence/nested_topology/m4_conv_diag.py --pct 2 --seed 0

做三件事，**不采信任何单点结论**：

1. 逐参数 R̂（numpyro split-rank，与 `bm_real` 同一口径），找出超 1.05 的
   参数名，而不是只看一个 `rhat_max`；
2. 后验相关矩阵（对数参数 Pearson，按 draw 展平）——若 R̂ 高的参数与另一个
   参数 |ρ|>0.9，就是**同一方向被数了两遍**，属参数化问题；若 |ρ| 普遍小，
   则更像多模态/链混合不足，属采样问题。两者的修法完全不同，不能混。
3. 每条链的中位数（物理量）：**链间不一致**才是 R̂ 的直接来源；若链间一致
   而 R̂ 仍高，那是链内自相关/有效样本不足。

只诊断，不改模型、不改默认路径。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bm_real as R          # noqa: E402
import bm_top as T           # noqa: E402

SPEC = os.path.join("out", "_m6_samples", "z_csv", "AZ31-4Li_100.csv")
BRANCH, LAYERS = "parallel", 2


def _post(o: dict) -> dict:
    """(chain, draw, ...) 形状的后验；`fit_top` 只回展平的 `s`，链要从 idata 取。"""
    po = o["idata"].posterior
    return {k: np.asarray(po[k].values) for k in po.data_vars}


def _corr(chains: dict, keys: list[str]) -> np.ndarray:
    """对数参数的 Pearson 相关（draw 展平；正参数取 log 更线性）。"""
    cols = []
    for k in keys:
        v = np.asarray(chains[k], float)
        v = v.reshape(-1, *v.shape[2:])            # 合并链 × draw
        cols.append(np.log(v.reshape(v.shape[0], -1)))
    M = np.column_stack(cols)
    return np.corrcoef(M, rowvar=False)


def run(pct: float, seed: int, warmup: int, samples: int, chains_n: int) -> dict:
    f, Z = R.load_spectrum(SPEC)
    sg = np.vstack([pct / 100.0 * np.abs(Z), pct / 100.0 * np.abs(Z)])
    o = T.fit_top(f, Z, branch=BRANCH, layers=LAYERS, n_warmup=warmup,
                  n_samples=samples, num_chains=chains_n, seed=seed,
                  target_accept=0.98, sigma=sg, progress=False)
    ch = _post(o)
    rh = dict(o["rhat_by_param"])          # 与 bm_real 同一口径（numpyro split-rank）
    keys = [k for k in T.sampled_keys(BRANCH, LAYERS) if k in ch]
    rh_p = {k: rh[k] for k in keys if k in rh}
    C = _corr(ch, [k for k in keys if k in ch])
    names = [k for k in keys if k in ch]

    # 链中位数（物理量）：链间不一致 = R̂ 的直接来源
    med = {}
    for k in names:
        v = np.asarray(ch[k], float)
        med[k] = [float(np.median(v[c].reshape(v.shape[1], -1)[:, 0]))
                  for c in range(v.shape[0])]

    print(f"\n=== 逐点 {pct:g}% 相对 σ · seed {seed} · {warmup}/{samples}×{chains_n} ===")
    print(f"{'参数':<10}{'R̂':>8}{'链中位':>34}   链间相对差")
    for k in names:
        m = med[k]
        spread = (max(m) - min(m)) / max(abs(np.median(m)), 1e-300)
        flag = "  ←超限" if rh_p[k] > 1.05 else ""
        print(f"{k:<10}{rh_p[k]:>8.3f}  " + " ".join(f"{x:>9.4g}" for x in m) +
              f"   {spread:>8.1%}{flag}")

    bad = sorted([k for k in names if rh_p[k] > 1.05], key=lambda k: -rh_p[k])
    print(f"\n超 1.05 的参数：{bad if bad else '无'}")
    if len(names) > 1:
        idx = {n: i for i, n in enumerate(names)}
        pairs = sorted(((abs(C[i][j]), names[i], names[j])
                        for i in range(len(names)) for j in range(i + 1, len(names))),
                       reverse=True)[:5]
        print("最强相关对（|ρ|）：" + ", ".join(f"{a}-{b} {r:.2f}" for r, a, b in pairs))
        # 与超限参数相关的项单独列出
        if bad:
            k = bad[0]
            i = idx[k]
            row = sorted(((abs(C[i][j]), names[j]) for j in range(len(names)) if j != i),
                         reverse=True)[:3]
            print(f"{k} 的 top3 相关：" + ", ".join(f"{b} {r:.2f}" for r, b in row))

    # 点估计的残差（中位参数）与 acf：R̂ 好不等于拟合好，两者必须同表报
    pm = {k: float(np.median(ch[k])) for k in names}
    Zm = T.z_nested(f, pm, branch=BRANCH, layers=LAYERS)
    rel = np.abs(Zm - Z) / np.abs(Z)
    d = (Zm - Z) / np.abs(Z)
    cost = float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2)))
    bands = []
    for lo, hi in ((0.01, 1), (1, 100), (100, 1e5)):
        m = (f >= lo) & (f < hi)
        bands.append(float(np.sqrt(np.mean(np.abs(d)[m] ** 2))) if m.any() else float("nan"))
    # 残差自相关：一阶（与 bm_real 的 acf_max 同一含义的简化版，只看 lag=1..3）
    acf = []
    for lag in (1, 2, 3):
        a = rel[:-lag] - rel[:-lag].mean()
        b = rel[lag:] - rel[lag:].mean()
        acf.append(float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b) + 1e-300)))
    print(f"  残差：rms_rel {cost*100:.2f}%（低/中/高 {bands[0]*100:.1f}/{bands[1]*100:.1f}/{bands[2]*100:.1f}%）"
          f"  acf(lag1..3) {acf[0]:.3f}/{acf[1]:.3f}/{acf[2]:.3f}"
          f"  Rs={pm['Rs']:.1f} R1={pm['R1']:.1f} Q1={pm['Q1']:.3g}")

    return dict(pct=pct, seed=seed, rhat=rh_p, bad=bad, med=med,
                rms_rel=cost, bands=bands, acf=acf, params=pm,
                corr_names=names, corr=C.tolist(),
                rhat_max=float(max(rh.values())), div=int(np.sum(o["divergences"])),
                elpd=float(o["elpd"]) if o.get("elpd") is not None else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, nargs="+", default=[2.0, 1.0])
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    seeds = a.seed if isinstance(a.seed, list) else [a.seed]
    out = [run(p, sd, a.warmup, a.samples, a.chains)
           for sd in seeds for p in a.pct]
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(spec=SPEC, branch=BRANCH, layers=LAYERS, rows=out),
                      fh, ensure_ascii=False, indent=1)
        print(f"\n已写 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
