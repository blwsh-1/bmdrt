"""腐蚀涂层 / 钝化 合成谱的拟合效果对照（只读，不改生产代码）。

每族取 3 条（按 |Z|max 的四分位取点），同一噪声口径（逐点 2%·|Z|）下比较：
  · 噪声底   = 数据 vs 真值拓扑正演（合成数据有真值，可精确给出）
  · nested   = bm_top.z_nested(branch="capacitive")
  · voigt    = 生产默认族，K 取 bmdrt 在该谱上实际选的值

真值只用于①算噪声底 ②事后核对参数；**初值完全由数据统计量给出**，不读真值。
"""
import sys, json, re, time, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "D:/eis训练数据/bmdrt")
import bm_top as T
import bm_real as R
import nested_fit as N
from bm_select import BUDGETS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = "D:/eis训练数据/测试数据"
OUT = "D:/eis训练数据/bmdrt/out/_user_fit_nested"
FAMILIES = [
    dict(name="涂层", sub="涂层/Coating_Generated EIS data and Nyquist plots/Generated EIS data and Nyquist plots",
         layers=2, voigt_K=2, keys=["Rs", "Qa", "na", "R1", "Qb", "nb", "R2", "Qc", "nc"]),
    dict(name="钝化", sub="钝化/Passivating_Generated EIS data and Nyquist plots/Generated EIS data and Nyquist plots",
         layers=1, voigt_K=1, keys=["Qa", "na", "Qb", "nb", "R1", "Rs"]),
]


def rmsrel(Z, Zm):
    d = (Zm - Z) / np.abs(Z)
    return float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2))) * 100


def acf1(x):
    x = np.asarray(x, float) - np.mean(x)
    d = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / d) if d > 0 else 0.0


def build_truth(fam, vals):
    v = dict(zip(fam["keys"], vals))
    if fam["layers"] >= 2:
        return dict(Rs=v["Rs"], R1=v["R1"], Q1=v["Qa"], n1=v["na"],
                    R2=v["R2"], Q2=v["Qb"], n2=v["nb"], Q_b=v["Qc"], n_b=v["nc"])
    return dict(Rs=v["Rs"], R1=v["R1"], Q1=v["Qa"], n1=v["na"],
                Q_b=v["Qb"], n_b=v["nb"])


def data_init(f, Z, layers):
    """只看数据：Rs=min|Z|，Q 用窗口中心幅值，R1/R2 用 zmax 量级。"""
    zmax = float(np.abs(Z).max())
    fr = np.sqrt(f.min() * f.max())
    zfr = float(np.interp(np.log10(fr), np.log10(f)[::-1], np.abs(Z)[::-1]))
    if layers >= 2:
        return dict(Rs=float(np.abs(Z).min()), R1=0.5 * zmax, Q1=zfr, n1=0.9,
                    R2=0.1 * zmax, Q2=zfr * 0.1, n2=0.9, Q_b=zfr * 0.01, n_b=0.6)
    return dict(Rs=float(np.abs(Z).min()), R1=0.5 * zmax, Q1=zfr, n1=0.8,
                Q_b=zfr, n_b=0.6)


KW = {k: v for k, v in BUDGETS["standard"].items()
      if k in ("n_warmup", "n_samples", "num_chains", "max_tree_depth")}
KW["target_accept"] = 0.98

results = []
for fam in FAMILIES:
    files = sorted(pathlib.Path(ROOT, fam["sub"]).glob("*.csv"))
    # 按 |Z|max 四分位取 3 条（不做全量，只看代表性）
    sized = []
    for p in files:
        try:
            f0, Z0 = R.load_spectrum(p)
            sized.append((float(np.abs(Z0).max()), p))
        except Exception:
            continue
    sized.sort()
    pick = [sized[int(len(sized) * q)][1] for q in (0.25, 0.5, 0.75)]
    print(f"\n{'='*78}\n{fam['name']}：共 {len(files)} 条，取 3 条（|Z|max 四分位）")
    for p in pick:
        vals = [float(x) for x in re.findall(r"[-+0-9.eE]+", p.stem[p.stem.rfind("[") + 1:p.stem.rfind("]")])]
        f, Z = R.load_spectrum(p)
        f = np.asarray(f, float); Z = np.asarray(Z, complex)
        zmax = float(np.abs(Z).max()); w_ref = 2 * np.pi * np.sqrt(f.min() * f.max())
        s = 0.02 * np.abs(Z); sigma = np.stack([s, s])

        truth = build_truth(fam, vals)
        for k in list(truth):
            if k.startswith("Q"):
                truth[k] = truth[k] / w_ref ** truth["n" + k[1:]]
        Zt = T.z_nested(f, truth, branch="capacitive", layers=fam["layers"])

        init = T.physical_to_unconstrained(data_init(f, Z, fam["layers"]), zmax=zmax,
                                           w_ref=w_ref, branch="capacitive",
                                           layers=fam["layers"], sigma=(1.0, 1.0),
                                           num_chains=KW["num_chains"])
        t0 = time.time()
        o = T.fit_top(f, Z, branch="capacitive", layers=fam["layers"], sigma=sigma,
                      sigma_from_data=True, init_values=init, progress=False,
                      seed=0, **KW)
        dt = time.time() - t0
        med = {k: float(np.median(o["s"][k])) for k in T.sampled_keys("capacitive", fam["layers"])}
        Zm = T.z_nested(f, med, branch="capacitive", layers=fam["layers"])

        vm = N.voigt_reference(f, Z, sigma, True, KW, 0, "逐点 2%", K=fam["voigt_K"])
        Zv = vm.pop("curve")

        rec = dict(family=fam["name"], file=p.name, layers=fam["layers"],
                   seconds=round(dt, 1), rhat_max=float(o["rhat_max"]),
                   ess_min=float(o["ess_min"]), divergences=int(o["divergences"]),
                   floor_rms=round(rmsrel(Z, Zt), 3),
                   nested_rms=round(rmsrel(Z, Zm), 3),
                   nested_acf=round(max(acf1((Zm - Z).real), acf1((Zm - Z).imag)), 3),
                   voigt_K=fam["voigt_K"], voigt_rms=round(vm["rms_rel"], 3),
                   voigt_acf=round(vm["acf"], 3),
                   truth={k: float(v) for k, v in truth.items()},
                   median=med,
                   curve=dict(f=f.tolist(), Zr=Z.real.tolist(), Zi=Z.imag.tolist(),
                              Mr=Zm.real.tolist(), Mi=Zm.imag.tolist(),
                              Vr=Zv.real.tolist(), Vi=Zv.imag.tolist()))
        results.append(rec)
        print(f"\n  {p.name[:56]}…  {dt:.1f}s  R̂={rec['rhat_max']:.3f} div={rec['divergences']} "
              f"ESS={rec['ess_min']:.0f}")
        print(f"    rms_rel: 噪声底 {rec['floor_rms']:6.2f}% | nested {rec['nested_rms']:6.2f}% "
              f"(acf {rec['nested_acf']:+.3f}) | voigt K={fam['voigt_K']} {rec['voigt_rms']:7.2f}% "
              f"(acf {rec['voigt_acf']:+.3f})")
        for k in med:
            tv = truth.get(k)
            print(f"      {k:5} {med[k]:11.4g}  truth {tv:11.4g}  ratio {med[k]/tv:7.3f}")

json.dump(results, open(f"{OUT}/family_fit_results.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\n结果已写入 {OUT}/family_fit_results.json")


# ------------------------------------------------------------------ 画图
def panel(famname, recs):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)
    cols = plt.cm.viridis(np.linspace(0.15, 0.8, len(recs)))

    ax = fig.add_subplot(gs[0, 0])
    for c, r in zip(cols, recs):
        cv = r["curve"]
        ax.plot(cv["Zr"], [-x for x in cv["Zi"]], "o", ms=3.2, mfc="none",
                mec=c, mew=0.8, label=f"数据 #{r['idx']}")
        ax.plot(cv["Mr"], [-x for x in cv["Mi"]], "-", color=c, lw=1.4,
                label=f"nested #{r['idx']}  {r['nested_rms']:.1f}%")
    ax.set_xlabel("Z' / Ω"); ax.set_ylabel("-Z'' / Ω")
    ax.set_title("① Nyquist：数据(点) vs nested 拟合(线)")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[0, 1])
    for c, r in zip(cols, recs):
        cv = r["curve"]; f = np.array(cv["f"])
        ax.loglog(f, np.hypot(cv["Zr"], cv["Zi"]), "o", ms=3.2, mfc="none", mec=c, mew=0.8)
        ax.loglog(f, np.hypot(cv["Mr"], cv["Mi"]), "-", color=c, lw=1.4)
        ax.loglog(f, np.hypot(cv["Vr"], cv["Vi"]), "--", color=c, lw=1.0, alpha=.75)
    ax.set_xlabel("f / Hz"); ax.set_ylabel("|Z| / Ω")
    ax.set_title("② Bode 幅值：实线 nested，虚线 voigt")
    ax.grid(alpha=.3, which="both")

    ax = fig.add_subplot(gs[1, 0])
    for c, r in zip(cols, recs):
        cv = r["curve"]; f = np.array(cv["f"])
        Zo = np.array(cv["Zr"]) + 1j * np.array(cv["Zi"])
        Zm = np.array(cv["Mr"]) + 1j * np.array(cv["Mi"])
        Zv = np.array(cv["Vr"]) + 1j * np.array(cv["Vi"])
        ax.semilogx(f, np.abs(Zm - Zo) / np.abs(Zo) * 100, "-", color=c, lw=1.3,
                    label=f"nested #{r['idx']}")
        ax.semilogx(f, np.abs(Zv - Zo) / np.abs(Zo) * 100, "--", color=c, lw=1.0, alpha=.75)
    ax.set_xlabel("f / Hz"); ax.set_ylabel("|ΔZ|/|Z| / %")
    ax.set_title("③ 相对残差：实线 nested，虚线 voigt")
    ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")

    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(recs)); w = 0.26
    ax.bar(x - w, [r["floor_rms"] for r in recs], w, label="噪声底(真值正演)", color="#9e9e9e")
    ax.bar(x, [r["nested_rms"] for r in recs], w, label="nested capacitive", color="#c0392b")
    ax.bar(x + w, [r["voigt_rms"] for r in recs], w, label="voigt", color="#2980b9")
    for i, r in enumerate(recs):
        for dx, v in ((-w, r["floor_rms"]), (0, r["nested_rms"]), (w, r["voigt_rms"])):
            ax.text(i + dx, v * 1.06, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels([f"#{r['idx']}" for r in recs])
    ax.set_ylabel("rms_rel / %（对数轴）"); ax.set_title("④ 残差 vs 噪声底")
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y", which="both")

    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    lines = [f"{famname}：{len(recs)} 条合成谱", ""]
    for r in recs:
        lines.append(f"#{r['idx']}  nested {r['nested_rms']:.2f}% / 底 {r['floor_rms']:.2f}%"
                     f" = {r['nested_rms']/r['floor_rms']:.2f}×")
        lines.append(f"      voigt {r['voigt_rms']:.2f}% = "
                     f"{r['voigt_rms']/r['floor_rms']:.1f}× 底")
    lines += ["", "nested = Rs+(CPE ∥ (R+(CPE ∥ (R+CPE))))",
              "研究性接口，未过 §13/§14 验收线；", "参数不得当物理量或真值。"]
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=8.5, family="monospace")

    fig.suptitle(f"{famname} 拟合效果：nested capacitive vs 生产默认 voigt（逐点 2% 噪声口径）",
                 fontsize=12)
    return fig


for fam in FAMILIES:
    recs = [r for r in results if r["family"] == fam["name"]]
    for i, r in enumerate(recs, 1):
        r["idx"] = i
    fig = panel(fam["name"], recs)
    path = f"{OUT}/fig_{'coating' if fam['name']=='涂层' else 'passivation'}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("图:", path)

print("\n=== 汇总 ===")
for r in results:
    print(f"{r['family']} #{r['idx']}  底 {r['floor_rms']:6.2f}%  "
          f"nested {r['nested_rms']:6.2f}%  voigt {r['voigt_rms']:7.2f}%  "
          f"R̂ {r['rhat_max']:.3f}  div {r['divergences']}")
