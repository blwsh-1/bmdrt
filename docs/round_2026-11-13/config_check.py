"""对照：旧口径(2链/w2000/td10/ta0.98) vs 生产口径(3链/w3000/td13/ta0.9)。

只作口径诊断，不改任何判据、不写 results.jsonl。
"""
import sys, json, time, pathlib
import numpy as np

sys.path.insert(0, ".")
import bm_real as R
import bm_core as C

OUT = pathlib.Path("out/coating_voigt_pilot")
man = json.load(open(OUT / "sample_manifest.json", encoding="utf-8"))
pmap = {x["name"]: x for x in man["picked"]}

# 生产默认（bm_real.KW），逐字照抄
PROD = dict(n_warmup=3000, n_samples=1000, num_chains=3, seed=0,
            progress=False, target_accept=0.9, max_tree_depth=13,
            extra_fields=("num_steps",))
# 上一轮 pilot 用的（bm_select.BUDGETS["standard"] + ta 覆盖）
MINE = dict(n_warmup=2000, n_samples=1000, num_chains=2, seed=0,
            progress=False, target_accept=0.98, max_tree_depth=10,
            extra_fields=("num_steps",))

from numpyro.diagnostics import summary as nsum


def one(f, Z, kw):
    t0 = time.time()
    sv, mc = C.fit(f, Z, K=2, alpha=0.5, sigma=None, zeta_fixed=1.0,
                   init_values=None, **kw)
    sec = time.time() - t0
    st = nsum(mc.get_samples(group_by_chain=True), prob=0.9)
    rhat = max(float(np.max(np.asarray(v["r_hat"]))) for v in st.values())
    ess = min(float(np.min(np.asarray(v["n_eff"]))) for v in st.values())
    div = int(np.sum(np.asarray(mc.get_extra_fields()["diverging"])))
    pv = dict(R_inf=float(np.median(sv["R_inf"])), W=float(np.median(sv["W"])),
              pi=np.median(sv["pi"], axis=0), tau=np.median(sv["tau"], axis=0),
              phi=np.median(sv["phi"], axis=0))
    if "zeta" in sv:
        pv["zeta"] = np.median(sv["zeta"], axis=0)
    Zm = C.model_z(f, pv, include_L=False)
    d = (Zm - Z) / np.abs(Z)
    rms = float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2))) * 100
    return dict(rhat=rhat, ess=ess, div=div, sec=sec, rms=rms,
                s_re=float(np.median(sv["s_re"])), s_im=float(np.median(sv["s_im"])))


rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
bad = [r for r in rows if r["voigt_prod"].get("rhat", 9) > 5]
print(f"旧口径 R̂>5 的 {len(bad)}/{len(rows)} 条，取前 4 条做对照\n")

res = []
for r in bad[:4]:
    item = pmap[r["name"]]
    f, Z = R.load_spectrum(item["path"])
    f = np.asarray(f, float); Z = np.asarray(Z, complex)
    floor = r["floor"]
    meanZ = float(np.mean(np.abs(Z)))
    o = r["voigt_prod"]
    print(f"cell={tuple(r['cell'])}  旧: R̂={o['rhat']:6.2f} ESS={o['ess']:5.0f} "
          f"div={o['div']:3d} ratio={r['ratio']:5.2f}  ({o['sec']:.0f}s)")
    p = one(f, Z, PROD)
    print(f"                    生产: R̂={p['rhat']:6.2f} ESS={p['ess']:5.0f} "
          f"div={p['div']:3d} ratio={p['rms']/floor:5.2f}  ({p['sec']:.0f}s)  "
          f"σ/名义={p['s_re']/(0.02*meanZ):.2f}×")
    res.append(dict(cell=list(r["cell"]), old=dict(rhat=o["rhat"], ess=o["ess"], div=o["div"],
                                                   ratio=r["ratio"], sec=o["sec"]),
                    prod=p))
    print()

json.dump(res, open(OUT / "config_check.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("写入", OUT / "config_check.json")
