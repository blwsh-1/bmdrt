"""诊断（事后、探索性）：把 `unusable_sampling` 拆成「标签互换」与「真多峰」。

不改变任何已冻结判定，不写 results.jsonl。输出 `out/coating_voigt_pilot/mode_diag.json`。

口径：
- 与生产完全一致（`bm_real.KW`：3 链 / warmup 3000 / tree_depth 13 / ta 0.9，自由 σ）。
- `rhat_naive`：对原始 `ln_tau[j]` 求 R̂（= pilot 报的那个）。
- `rhat_sorted`：每个样本内把 `ln_tau` 升序重排后再求 R̂（**对标签置换不变**）。
- 模式数：把各链中位 `ln_tau`（排序后）按 0.3 dec 聚类。
"""
import sys, json, pathlib
import numpy as np

sys.path.insert(0, ".")
import bm_real as R
import bm_core as C

OUT = pathlib.Path("out/coating_voigt_pilot")
PROD = dict(n_warmup=3000, n_samples=1000, num_chains=3, seed=0,
            progress=False, target_accept=0.9, max_tree_depth=13,
            extra_fields=("num_steps",))


def rhat_col(x):
    """x: (m, n) → R̂（Gelman-Rubin, split-free）"""
    m, n = x.shape
    if m < 2:
        return float("nan")
    bm = x.mean(1)
    W = x.var(1, ddof=1).mean()
    B = n * bm.var(ddof=1)
    if W <= 0:
        return float("inf")
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def main(n_take=10):
    man = json.load(open(OUT / "sample_manifest.json", encoding="utf-8"))
    pmap = {x["name"]: x for x in man["picked"]}
    rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    bad = [r for r in rows if r["voigt_prod"].get("rhat", 9) > 1.05]
    # 按 cell 轮转取，避免全落在同一格
    seen, take = {}, []
    for r in bad:
        c = tuple(r["cell"])
        if seen.get(c, 0) < 1:
            seen[c] = seen.get(c, 0) + 1
            take.append(r)
    for r in bad:
        if len(take) >= n_take:
            break
        if r not in take:
            take.append(r)
    take = take[:n_take]

    out = []
    for r in take:
        item = pmap[r["name"]]
        f, Z = R.load_spectrum(item["path"])
        f = np.asarray(f, float); Z = np.asarray(Z, complex)
        sv, mc = C.fit(f, Z, K=2, alpha=0.5, sigma=None, zeta_fixed=1.0,
                       init_values=None, **PROD)
        s = mc.get_samples(group_by_chain=True)
        ln = np.asarray(s["ln_tau"])                      # (m, n, K)
        K = ln.shape[-1]
        rh_n = max(rhat_col(ln[:, :, j]) for j in range(K))
        lns = np.sort(ln, axis=-1)
        rh_s = max(rhat_col(lns[:, :, j]) for j in range(K))
        # 每链中位（排序后）→ 聚类数
        med = np.median(lns, axis=1)                      # (m, K)
        modes = []
        for c in range(med.shape[0]):
            if not any(np.max(np.abs(med[c] - mm)) < 0.3 for mm in modes):
                modes.append(med[c])
        rec = dict(cell=list(r["cell"]), name=r["name"],
                   rhat_naive=round(float(rh_n), 3),
                   rhat_sorted=round(float(rh_s), 3),
                   n_modes=len(modes),
                   chain_med=[np.round(m, 2).tolist() for m in med],
                   ratio=r.get("ratio"))
        out.append(rec)
        print(f"cell={tuple(r['cell'])}  R̂ 原始 {rh_n:6.2f} → 排序后 {rh_s:6.2f}  "
              f"模式数 {len(modes)}  ratio={r.get('ratio')}")
        for c, m in enumerate(med):
            print(f"      chain{c} ln_tau 中位 {np.round(m,2).tolist()}")
        sys.stdout.flush()

    json.dump(out, open(OUT / "mode_diag.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    a = np.array([x["rhat_naive"] for x in out])
    b = np.array([x["rhat_sorted"] for x in out])
    print(f"\n=== 汇总 n={len(out)} ===")
    print(f"排序前 R̂ 中位 {np.median(a):.2f}；排序后中位 {np.median(b):.2f}")
    print(f"排序后仍 >1.05 的 {int((b > 1.05).sum())}/{len(b)}"
          f"（= 标签互换之外还有真多峰/不可辨识）")
    print(f"至少 2 个模式的 {sum(1 for x in out if x['n_modes'] >= 2)}/{len(out)}")
    print("写入", OUT / "mode_diag.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
