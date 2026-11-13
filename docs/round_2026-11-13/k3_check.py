"""诊断（事后）：K=2 的「分量逃到窗口外」是否因为 K 不够。

同一批逃逸样本，改用 K=3（其它口径与生产一致：自由 σ、3 链、warmup 3000）。
只作探索性诊断，不进任何判定。
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

diag = json.load(open(OUT / "mode_diag.json", encoding="utf-8"))
# 取「有分量逃到窗口外」的：max 链中位 ln_tau > 7
esc = [d for d in diag if max(max(c) for c in d["chain_med"]) > 7.0]
print(f"mode_diag 里逃逸样本 {len(esc)}/{len(diag)}，取前 4 条做 K=3 对照\n")

man = json.load(open(OUT / "sample_manifest.json", encoding="utf-8"))
pmap = {x["name"]: x for x in man["picked"]}
rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]
rmap = {r["name"]: r for r in rows}


def rhat_col(x):
    m, n = x.shape
    bm = x.mean(1); W = x.var(1, ddof=1).mean(); B = n * bm.var(ddof=1)
    return float(np.sqrt(((n - 1) / n * W + B / n) / W)) if W > 0 else float("inf")


out = []
for d in esc[:4]:
    r = rmap[d["name"]]
    f, Z = R.load_spectrum(pmap[d["name"]]["path"])
    f = np.asarray(f, float); Z = np.asarray(Z, complex)
    floor = r["floor"]
    print(f"cell={tuple(d['cell'])}  K=2: R̂={d['rhat_naive']:.1f} ratio={d['ratio']:.2f}")
    for K in (3,):
        sv, mc = C.fit(f, Z, K=K, alpha=0.5, sigma=None, zeta_fixed=1.0,
                       init_values=None, **PROD)
        s = mc.get_samples(group_by_chain=True)
        ln = np.asarray(s["ln_tau"])
        rh = max(rhat_col(ln[:, :, j]) for j in range(K))
        rh_s = max(rhat_col(np.sort(ln, axis=-1)[:, :, j]) for j in range(K))
        div = int(np.sum(np.asarray(mc.get_extra_fields()["diverging"])))
        med = np.median(ln, axis=1)
        pv = dict(R_inf=float(np.median(sv["R_inf"])), W=float(np.median(sv["W"])),
                  pi=np.median(sv["pi"], axis=0), tau=np.median(sv["tau"], axis=0),
                  phi=np.median(sv["phi"], axis=0))
        if "zeta" in sv:
            pv["zeta"] = np.median(sv["zeta"], axis=0)
        Zm = C.model_z(f, pv, include_L=False)
        dd = (Zm - Z) / np.abs(Z)
        rms = float(np.sqrt(np.mean(np.concatenate([dd.real, dd.imag]) ** 2))) * 100
        nesc = int(np.sum(np.median(ln, axis=1) > 7.0))
        print(f"          K={K}: R̂={rh:6.2f} (排序后 {rh_s:5.2f}) div={div:3d} "
              f"ratio={rms/floor:5.2f}  逃逸分量数(链中位>7)={nesc}")
        for c in range(med.shape[0]):
            print(f"               chain{c} ln_tau {np.round(med[c],2).tolist()}")
        out.append(dict(cell=d["cell"], name=d["name"], K=K,
                        rhat=round(rh, 3), rhat_sorted=round(rh_s, 3),
                        div=div, ratio=round(rms / floor, 3), n_escape=nesc))
    print()
    sys.stdout.flush()

json.dump(out, open(OUT / "k3_check.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("写入", OUT / "k3_check.json")
