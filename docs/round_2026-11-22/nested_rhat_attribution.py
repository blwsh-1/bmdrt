"""nested_rhat_attribution.py —— `nested_L2_L` 的 R̂=1.100 到底卡在哪。

背景
----
`B2_REPORT.md` §2/§4：`nested_L2_L` 在 `standard` 预算下 R̂=**1.100**（div=0），
按 `b2_baseline` 的层 0 口径记 `non_evaluable`；`deep`（4000+2000/2 链/ta 0.97）
掉到 **1.007** ⇒ 报告判为 `budget_limited`。

但这只回答了"加预算能不能过"，没回答**为什么**标准预算过不了。两条可能：
  (a) 后验有一条**很扁但良性**的方向（强相关脊），NUTS 走得慢 ⇒ 加预算就好；
  (b) 有**漏斗/多峰**这类型病态 ⇒ 加预算是掩盖，不是修好。
两者对真实数据的含义完全不同：(a) 可以只加预算，(b) 得改参数化。

判据（本脚本）
--------------
不做"再多跑一组看看"，而是先用**解析几何**定位最扁的方向，再看采样的实际表现：

1. **解析**：`H = JᵀJ + Λ` 的最小特征值方向 `v_min`（`J` = 白化雅可比，
   与 `b2_bias_attribution.py` 同一口径）。看它是谁组成的。
2. **实测**：一次带链拟合（2 链），逐参数 split-R̂ / ESS；
   并把样本**投影到 `v_min`** 上算这条方向的 R̂ —— 若整模型的 R̂_max 就是
   这条方向的 R̂，则 (a) 成立；若 `v_min` 方向的 R̂ 正常而某个参数单独差，
   则问题不在几何而在别处（多峰、边界、初始化）。
3. **形状**：沿 `v_min` 的后验分布是单峰钟形（(a)）还是双峰（(b)）；
   用投影样本的直方图/峰数 + 与对数密度的相关性（funnel 的特征是
   `corr(param, logp)` 显著）判。

用法
----
    python docs/round_2026-11-22/nested_rhat_attribution.py --smoke   # 20/20，只为验通路
    python docs/round_2026-11-22/nested_rhat_attribution.py           # standard 全量，约 70 秒
    python docs/round_2026-11-22/nested_rhat_attribution.py --deep \
        --save out/nested_rhat_chains_deep.npz

实测结论（2026-11-22，见 `NESTED_RHAT_ATTRIBUTION.md`）
------------------------------------------------------
既不是 (a) 也不是 (b)，而是 **(a) + 一个被拖走的补偿参数**：

· `H` 的两条平方向（λ=0.2651/0.3042 vs 先验 0.25）几乎全在 `(R_ad, tau_ad)` 上，
  数据只贡献先验的 **6.0% / 21.7%** ⇒ 那里后验 ≈ 先验；
· `R_ad` 先验中心 `0.1·zmax=35.3 Ω`，真值 0.5 Ω ⇒ 先验把后验往 `Z_ad≈R_ad`
  的纯电阻区推；
· standard 的链 0 在 `tau_ad` 上漂到 0.605 s（链 1 是 5.3e-4），
  于是 `ρ(R2, log tau_ad) = −0.61`，`R2` 被压到 **182±52** —— 报告的 R̂=1.1005 就是 `R2`；
· deep 下真收敛（`R2` 198.61/198.79，链内 sd 1.8 = 0.90%，已达到 Laplace Fisher 界 0.76%；
  `R2` 偏差 −4.8% → **−0.7%**）。

⇒ `budget_limited` 成立，但 **standard 下 `nested_L2_L` 的 8/8 是"数字对、后验没抽对"**，
这条配方的 B2 数字只能引 deep 档。
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "docs", "round_2026-11-21"))
sys.path.insert(0, os.path.join(REPO, "docs", "round_2026-11-22"))

import b2_baseline as B2                    # noqa: E402
import b2_bias_attribution as A             # noqa: E402
import bm_select as SEL                     # noqa: E402

NAME = "nested_L2_L"


def recipe():
    for r in B2.RECIPES:
        if r["name"] == NAME:
            return r
    raise KeyError(NAME)


def geometry(rec):
    """JᵀJ+Λ 的特征分解：最小方向是谁。"""
    f, zt, zo, sg, _ = B2.generate(rec)
    flat = {k: float(v) for k, v in rec["truth"].items()}
    coords = A.sampled_coords(rec, flat)
    J, names = A.jacobian(rec["family"], f, flat, sg, coords,
                          int(rec.get("layers", 2)))
    lam = np.array([A.PRIOR_PREC_LOG[rec["family"]].get(k, 0.0)
                    if coords[k] == "log" else 0.0 for k in names])
    H = J.T @ J + np.diag(lam)
    w, V = np.linalg.eigh(H)
    order = np.argsort(w)
    return dict(f=f, zt=zt, zo=zo, sg=sg, flat=flat, coords=coords,
                J=J, names=names, H=H, w=w[order], V=V[:, order])


def fit(rec, kw):
    f, zt, zo, sg, _ = B2.generate(rec)
    c = SEL.fit_nested(f, zo, int(rec.get("layers", 2)), branch="parallel",
                       seed=int(rec["seed"]), sigma_floor=0.0, kw=kw,
                       sigma=sg, sigma_from_data=True, include_L=True)
    return c, f, zt, zo, sg


def per_param(raw, names):
    """逐参数 split-R̂ / ESS（链维必须保留）。"""
    from numpyro.diagnostics import summary
    po = raw["idata"].posterior
    ch = {k: np.asarray(po[k].values) for k in po}      # (chain, draw[, comp])
    flat = {}
    for k, v in ch.items():
        flat[k] = v[..., 0] if v.ndim == 3 else v       # 分量维只取 0（K=1/单分量）
    sm = summary(flat, prob=0.9, group_by_chain=True)
    rows = {}
    for k, v in sm.items():
        mu = np.asarray(v["mean"], float)
        rows[k] = dict(rhat=float(np.max(np.asarray(v["r_hat"]))),
                       ess=float(np.min(np.asarray(v["n_eff"]))),
                       mean=float(mu) if mu.size == 1 else float(mu[0]))
    return flat, rows


def _1d_v(v):
    """(chain, draw[, comp]) → (chain*draw,)。"""
    return _2d(v).ravel()


def _2d(v):
    """(chain, draw[, comp]) → (chain, draw)。"""
    v = np.asarray(v, float)
    return v[..., 0] if v.ndim == 3 else v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="20/20 仅验通路")
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--deep", action="store_true",
                    help="用 deep 档（4000+2000/2 链/ta 0.97）复核 'budget_limited' 的含义")
    ap.add_argument("--save", default="out/nested_rhat_chains.npz",
                    help="保存逐链样本（供离线复查；默认写 out/，不入库）")
    a = ap.parse_args(argv)

    rec = recipe()
    print("=" * 78)
    print("1. 解析几何：H = JᵀJ+Λ 的最扁方向")
    print("=" * 78)
    g = geometry(rec)
    names, w, V = g["names"], g["w"], g["V"]
    print(f"  特征值（升序，前 5）：{np.array2string(w[:5], precision=3)}")
    print(f"  条件数 λmax/λmin = {w[-1]/max(w[0],1e-30):.3g}")
    for j in range(3):
        v = V[:, j]
        comp = sorted(zip(names, v), key=lambda t: -abs(t[1]))[:4]
        print(f"  v[{j}] (λ={w[j]:.4g})："
              + "、".join(f"{k} {c:+.2f}" for k, c in comp))
    vmin = V[:, 0]
    print("  ⇒ 最扁方向主要由 "
          + "、".join(f"`{k}`" for k, c in sorted(zip(names, vmin),
                                                  key=lambda t: -abs(t[1]))[:3])
          + " 组成")
    # 所有"扁"方向 + 与先验精度的对比：λ ≈ 先验精度 ⇒ 数据在该方向几乎无信息
    lam = np.array([A.PRIOR_PREC_LOG[rec["family"]].get(k, 0.0)
                    if g["coords"][k] == "log" else 0.0 for k in names])
    print()
    print("  扁方向（λ < 10）与先验精度对比：")
    for j in range(min(5, len(w))):
        if w[j] >= 10:
            break
        v = V[:, j]
        # 该方向上的先验精度 vᵀΛv
        pv = float(np.sum(lam * v ** 2))
        tag = "≈ 纯先验（数据无信息）" if abs(w[j] - pv) < 0.05 * max(pv, 1e-9) else ""
        comp = sorted(zip(names, v), key=lambda t: -abs(t[1]))[:3]
        print(f"    λ={w[j]:.4g}  先验 vᵀΛv={pv:.4g}  数据贡献={w[j]-pv:+.3g}  "
              + "、".join(f"{k} {c:+.2f}" for k, c in comp) + ("  " + tag if tag else ""))

    kw = (dict(n_warmup=20, n_samples=20, num_chains=a.chains,
               target_accept=0.9, max_tree_depth=10)
          if a.smoke else dict(
              B2.BUDGETS_LOCAL["deep" if a.deep else "standard"],
              num_chains=a.chains))
    print()
    print("=" * 78)
    print(f"2. 实测：{NAME}，预算 {kw}")
    print("=" * 78)
    c, f, zt, zo, sg = fit(rec, kw)
    raw = c["raw"]
    if "idata" not in raw:
        print("  raw 里没有 idata，无法取逐链样本 ⇒ 本脚本前提不成立，停止。")
        print("  raw keys:", sorted(raw))
        return 1
    flat, rows = per_param(raw, names)
    print(f"  raw.rhat_max = {raw.get('rhat_max')}   div = {raw.get('divergences')}"
          f"   ess_min = {raw.get('ess_min')}")
    print(f"  {'参数':<9}{'split-R̂':>10}{'ESS':>9}{'后验均值':>12}")
    worst = max(rows.items(), key=lambda t: t[1]["rhat"])
    for k in names:
        if k in rows:
            r = rows[k]
            mark = "  ← 最差" if k == worst[0] else ""
            print(f"  {k:<9}{r['rhat']:>10.4f}{r['ess']:>9.0f}"
                  f"{r['mean']:>12.5g}{mark}")

    # 后验相关：谁和谁绑在一起。**与几何同坐标**（log 参数取 log），
    # 否则相关矩阵和 `v_min` 的分量没法互相参照（原始尺度下 tau_ad 的 ρ 会偏）。
    print()
    print("  后验相关（|ρ| > 0.80 的对，按强弱排序；log 参数在 log 坐标）：")
    M = np.stack([(np.log(_1d_v(flat[k])) if g["coords"][k] == "log"
                   else _1d_v(flat[k])) for k in names], axis=1)      # (N, p)
    C = np.corrcoef(M, rowvar=False)
    pairs = [(abs(C[i, j]), names[i], names[j], C[i, j])
             for i in range(len(names)) for j in range(i + 1, len(names))
             if abs(C[i, j]) > 0.80]
    for _, a_, b_, r_ in sorted(pairs, reverse=True)[:10]:
        print(f"    {a_:<8}–{b_:<8} ρ = {r_:+.3f}")
    if not pairs:
        print("    （无 |ρ|>0.80 的对）")
    print(f"  `{worst[0]}` 的全行相关（|ρ| 降序前 4）：")
    iw = names.index(worst[0]) if worst[0] in names else 0
    row = sorted(((abs(C[iw, j]), names[j], C[iw, j])
                  for j in range(len(names)) if j != iw), reverse=True)[:4]
    print("    " + "；".join(f"{b_} {r_:+.2f}" for _, b_, r_ in row))

    # --- 逐链均值：链间差 vs 链内散度 ---
    print("  逐链均值（最差参数 + 两条平方向上的参数）：")
    for k in [worst[0], "R_ad", "tau_ad"]:
        if k not in flat:
            continue
        m = _2d(flat[k]).mean(axis=1)
        print(f"    {k:<8} 链均值 {np.array2string(m, precision=4)}"
              f"   链间 sd {m.std(ddof=1):.4g}")

    # --- 投影到最扁方向 ---
    print()
    print("=" * 78)
    print("3. 沿最扁方向的采样表现")
    print("=" * 78)
    nsamp = min(len(flat[k]) for k in names if k in flat)
    vmin = V[:, 0]

    # (chain, draw, param)，坐标统一到"相对真值的 log 偏移 / 绝对偏移"
    X = np.stack([
        (np.log(_2d(flat[k])) if g["coords"][k] == "log" else _2d(flat[k]))
        - (np.log(g["flat"][k]) if g["coords"][k] == "log" else g["flat"][k])
        for k in names], axis=-1)
    proj = X @ vmin                                   # (chain, draw)
    # 投影本身的 split-R̂（按链切）
    nch = proj.shape[0]
    nd = proj.shape[1]
    half = nd // 2
    chains = np.concatenate([proj[:, :half], proj[:, half:2 * half]], axis=0)
    W = chains.var(axis=1, ddof=1).mean()
    Bv = chains.mean(axis=1).var(ddof=1) * half
    var_hat = (half - 1) / half * W + Bv / half
    rhat_proj = float(np.sqrt(var_hat / W))
    print(f"  投影 v_min·(θ−θ*)：sd {proj.std():.4g}，"
          f"split-R̂ = {rhat_proj:.4f}")
    print(f"  整模型最差参数的 split-R̂ = {worst[1]['rhat']:.4f}"
          f"（{worst[0]}）")

    # 与对数密度的相关（funnel 特征）
    lp = None
    try:
        st = getattr(raw["idata"], "sample_stats", None)
        if st is not None:
            lp = np.asarray(st["lp"].values, float)
        else:
            lp = np.asarray(raw["idata"]["sample_stats"]["lp"], float)
    except Exception:                        # noqa: BLE001
        lp = None
    if lp is None:
        print("  取 logp 失败（该 idata 未带 sample_stats），跳过 funnel 相关")
    else:
        corr = float(np.corrcoef(lp.ravel(), proj.ravel())[0, 1])
        print(f"  corr(logp, v_min 投影) = {corr:+.3f}"
              f"{'  ← 漏斗特征' if abs(corr) > 0.5 else ''}")

    # 形状：投影的分位数 + 众数个数（粗糙）
    q = np.percentile(proj, [5, 25, 50, 75, 95])
    print("  投影分位 5/25/50/75/95："
          + " ".join(f"{x:+.3g}" for x in q))
    hist, edges = np.histogram(proj, bins=30)
    peaks = sum(1 for i in range(1, len(hist) - 1)
                if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]
                and hist[i] > 0.3 * hist.max())
    print(f"  直方图峰数（30 bin，>30% 峰高）= {peaks}"
          f"{'  ← 多峰嫌疑' if peaks >= 2 else '  ← 单峰'}")

    print()
    print("=" * 78)
    print("4. 判定")
    print("=" * 78)
    same = abs(rhat_proj - worst[1]["rhat"]) < 0.02
    # 最差参数是否与"平方向上的参数"强相关（= 被拖走，而不是它自己有病）
    flat_names = [t[0] for t in sorted(zip(names, vmin),
                                       key=lambda t: -abs(t[1]))[:2]]
    dragged = [(n_, C[iw, names.index(n_)]) for n_ in flat_names
               if n_ != worst[0] and abs(C[iw, names.index(n_)]) > 0.5]
    if rhat_proj > 1.05 and same:
        print("  ⇒ (a) 扁而良性的脊：整模型的 R̂ 就是最扁方向的 R̂。")
        print("     加预算或沿该方向重参数化都能解，不是病态。")
    elif dragged:
        print(f"  ⇒ 最差参数 `{worst[0]}` 是被平方向拖走的，不是自身有病：")
        for n_, r_ in dragged:
            print(f"     ρ(`{worst[0]}`, `{n_}`) = {r_:+.2f}"
                  f"（`{n_}` 在最扁方向上，数据曲率仅占先验的"
                  f"{100*(w[0]-float(np.sum(lam*vmin**2)))/float(np.sum(lam*vmin**2)):.1f}%）")
        print("     加预算能过（deep 实测 R̂ 1.0075、R2 偏差 −4.8%→−0.7%），但 standard 下")
        print("     该参数的 `within` 是侥幸 ⇒ 这条配方的数字只能引 deep 档。")
    elif worst[1]["rhat"] > 1.05:
        print(f"  ⇒ 最差参数是 `{worst[0]}`，不是几何最扁方向（投影 R̂ "
              f"{rhat_proj:.3f}），也不与被拖走的参数强相关 ⇒ 查该参数自身"
              "（多峰/边界/初始化）。")
    else:
        print("  ⇒ 本次采样已达标（R̂ ≤ 1.05），复现了 deep 档的结论。")
    print()
    print("=" * 78)
    print("5. 已保存")
    print("=" * 78)
    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        np.savez_compressed(
            a.save, names=np.array(names),
            **{f"post_{k}": _2d(flat[k]) for k in names if k in flat},
            proj=proj, vmin=vmin, eig=w,
            prior_prec=lam, rhat={k: rows[k]["rhat"] for k in rows},
            budget=np.array([kw["n_warmup"], kw["n_samples"], kw["num_chains"]]))
        print(f"  链样本 → {a.save}（{os.path.getsize(a.save)/1e6:.1f} MB，out/ 不入库）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
