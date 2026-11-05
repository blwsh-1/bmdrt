"""P2 只读分析器：2x3（K=1/2 x none/Ws/Wo）统一网格对照。

不修改 bmdrt 任何代码 / 参数。只读取已落盘的 run.log / JSON / NPZ，
用与 bm_real 完全相同的逐频点似然（loglik_matrix）重算 PSIS-LOO / Pareto-k，
再用 az.compare 把 6 个模型放在同一张比较表里。

运行：
  D:/autoeis_env/Scripts/python.exe analyze_grid.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# 归档到仓库 docs/cases/ 后也能自动定位仓库根；工作副本回退到绝对路径。
REPO = next((q for q in [ROOT, *ROOT.parents] if (q / "bm_core.py").exists()),
            Path(r"D:/eis训练数据/bmdrt"))
sys.path.insert(0, str(REPO))
import bm_core as B  # noqa: E402
import bm_real as R  # noqa: E402

NAMES = ["G_K1_none", "G_K2_none", "G_K1_Ws", "G_K2_Ws",
         "G_K1_Wo", "G_K2_Wo"]
LABEL = {
    "G_K1_none": "K=1 / none", "G_K2_none": "K=2 / none",
    "G_K1_Ws": "K=1 / Ws", "G_K2_Ws": "K=2 / Ws",
    "G_K1_Wo": "K=1 / Wo", "G_K2_Wo": "K=2 / Wo",
}


def q(a, p):
    return float(np.percentile(np.asarray(a, float), p))


def stats(a):
    a = np.asarray(a, float)
    return dict(q05=q(a, 5), q50=q(a, 50), q95=q(a, 95),
                mean=float(a.mean()), sd=float(a.std(ddof=1)))


def cv(a):
    a = np.asarray(a, float)
    return a.std(ddof=1) / a.mean()


def hn_component(f, w, tau, phi, zeta):
    """与 bm_core._hn_z 等价的向量化版本，形状 (n_f, n_sample)。"""
    x = np.maximum(2 * np.pi * np.asarray(f, float)[:, None]
                   * np.asarray(tau, float)[None, :], 1e-300)
    ph = np.asarray(phi, float)[None, :]
    mag = np.exp(ph * np.log(x))
    ang = ph * (np.pi / 2)
    denom = 1.0 + mag * (np.cos(ang) + 1j * np.sin(ang))
    return np.asarray(w, float)[None, :] / denom ** np.asarray(zeta, float)[None, :]


def load(name):
    # 案例包自包含：results/G_* 已有副本时优先读副本。
    d = REPO / "out/_grid" / name / name
    local = ROOT / "results" / name
    if (local / f"real_{name}_samples.npz").exists():
        d = local
    js_p = d / f"real_{name}.json"
    npz_p = d / f"real_{name}_samples.npz"
    log_p = d / "run.log"
    js = json.loads(js_p.read_text(encoding="utf-8"))
    z = np.load(npz_p, allow_pickle=True)
    s = {k: np.asarray(z[k], float) for k in z.files
         if k not in ("f", "Zre", "Zim")}
    f = np.asarray(z["f"], float)
    Z = np.asarray(z["Zre"], float) + 1j * np.asarray(z["Zim"], float)
    K = int(js["K"])
    if "zeta" not in s:
        s["zeta"] = np.ones((len(s["R_inf"]), K))
    txt = log_p.read_text(encoding="utf-8", errors="replace")
    mt = re.search(r"平滑趋势\s*([\d.]+)σ\(实\)\s*/\s*([\d.]+)σ\(虚\)", txt)
    trend = (float(mt.group(1)), float(mt.group(2))) if mt else (np.nan, np.nan)
    mj = re.search(r"残差有系统性弯曲\*\*：趋势\s*[\d.]+σ 位于白噪声零分布第\s*([\d.]+)\s*百分位", txt)
    trend_pct = float(mj.group(1)) if mj else np.nan
    mpc = re.search(r"PPC 覆盖率\s*([\d.]+)\(实\)\s*/\s*([\d.]+)\(虚\)", txt)
    ppc = (float(mpc.group(1)), float(mpc.group(2))) if mpc else (np.nan, np.nan)
    mm = B.match_components(s)
    o = np.argsort(-np.median(mm["f0"], axis=0))
    al = {nm: mm[nm][:, o] for nm in ("tau", "w", "phi", "zeta", "pi", "f0")}
    return dict(name=name, js=js, s=s, f=f, Z=Z, K=K, al=al,
                trend=trend, trend_pct=trend_pct, ppc=ppc,
                js_p=js_p, npz_p=npz_p, log_p=log_p)


def main():
    models = {n: load(n) for n in NAMES}

    # ---------- 1. 统一逐点似然 -> PSIS-LOO / Pareto-k -> az.compare ----------
    idatas, loose = {}, {}
    pk_rows = []
    for n, m in models.items():
        ll = R.loglik_matrix(m["s"], m["f"], m["Z"],
                             include_L=False, n_max=800)
        # loo_pointwise 只返回 idata / elpd / se / pk>0.7 占比 / 逐点 elpd，
        # 不返回逐点 pareto_k；从同一个 idata 再跑一次 az.loo 取逐点 k。
        idata, elpd, se, pk_frac, pt = R.loo_pointwise(ll)
        idatas[LABEL[n]] = idata
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            rr = az.loo(idata, pointwise=True)
        pk = np.asarray(rr.pareto_k.values).ravel()
        loose[n] = dict(elpd=float(rr.elpd_loo), se=float(rr.se),
                        pk_frac=float(np.mean(pk > 0.7)),
                        ll=ll, pt=pt, pk=pk)
        for j, (fi, v) in enumerate(zip(m["f"], pk)):
            pk_rows.append(dict(model=LABEL[n], f_Hz=float(fi),
                                pareto_k=float(v)))
    pd.DataFrame(pk_rows).to_csv(ROOT / "pareto_k.csv", index=False,
                                 float_format="%.4f")

    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        cmp = az.compare(idatas, ic="loo", method="stacking", scale="log")

    # ---------- 2. 六模型统一比较表 ----------
    rows = []
    for n, m in models.items():
        js, s, al = m["js"], m["s"], m["al"]
        lab = LABEL[n]
        cm = cmp.loc[lab]
        pk = loose[n]["pk"]
        d = dict(
            model=lab, K=js["K"], diffusion=js.get("diffusion") or "none",
            k_selection_status=js.get("k_selection_status"),
            grade=js.get("sample_grade"),
            rms_re=js.get("rms_re"), rms_im=js.get("rms_im"),
            sigma_re=js.get("sigma_re"), sigma_im=js.get("sigma_im"),
            acf=js.get("acf"),
            trend_sigma_re=m["trend"][0], trend_sigma_im=m["trend"][1],
            trend_pct=m["trend_pct"],
            PPC_re=m["ppc"][0], PPC_im=m["ppc"][1],
            divergences=js.get("divergences"), tree_cap=js.get("tree_cap"),
            p_peak_out_max=js.get("p_peak_out_max"),
            elpd_loo_orig=js.get("elpd_loo"),
            elpd_loo_recalc=loose[n]["elpd"], elpd_se=loose[n]["se"],
            elpd_diff=float(cm["elpd_diff"]), dse=float(cm["dse"]),
            weight=float(cm["weight"]), rank=int(cm["rank"]),
            pareto_k_max=float(pk.max()),
            pareto_k_gt07=float(np.mean(pk > 0.7)),
            n_pareto_gt07=int((pk > 0.7).sum()),
            t_fit_s=js.get("t_fit_s"),
        )
        # 扩散参数
        for key, out in (("Aw", "Aw"), ("Rw", "Rw"), ("Rwo", "Rwo"),
                         ("tauD", "tauD")):
            if key in s:
                d[out + "_q50"] = q(s[key], 50)
                d[out + "_q05"] = q(s[key], 5)
                d[out + "_q95"] = q(s[key], 95)
        if "tauD" in s:
            d["f_D_Hz"] = float(np.median(1.0 / (2 * np.pi * s["tauD"])))
        if "Rwo" in s and "tauD" in s:
            d["A_W_equiv"] = float(np.median(s["Rwo"] / np.sqrt(s["tauD"])))
            d["C_D_equiv"] = float(np.median(s["tauD"] / s["Rwo"]))
        if "Aw" in s:
            d["A_W_equiv"] = q(s["Aw"], 50)
        d["p_diff_small"] = js.get("p_diff_small")
        d["diffusion_resolution"] = json.dumps(
            js.get("diffusion_resolution"), ensure_ascii=False)[:400]
        # 慢分量渐近幅值组合 B2 = w2 * tau2^(-phi2)（仅 K=2）
        if m["K"] >= 2:
            j = m["K"] - 1
            w2, t2, p2 = al["w"][:, j], al["tau"][:, j], al["phi"][:, j]
            B2 = w2 * t2 ** (-p2)
            d.update(w2_cv=cv(w2), tau2_cv=cv(t2), phi2_cv=cv(p2),
                     B2_cv=cv(B2), B2_q50=q(B2, 50),
                     B2_q05=q(B2, 5), B2_q95=q(B2, 95),
                     f0_2_q50=q(al["f0"][:, j], 50),
                     w2_q50=q(w2, 50), tau2_q50=q(t2, 50),
                     phi2_q50=q(p2, 50))
        rows.append(d)
    grid = pd.DataFrame(rows)
    grid.to_csv(ROOT / "grid_comparison.csv", index=False, float_format="%.6g")

    # ---------- 3. 后验摘要（含扩散与 B2） ----------
    prows = []
    for n, m in models.items():
        s, al = m["s"], m["al"]
        for nm in ("R_inf", "W", "s_re", "s_im", "Aw", "Rw", "Rwo", "tauD"):
            if nm in s:
                prows.append(dict(model=LABEL[n], group="global",
                                  param=nm, **stats(s[nm])))
        for j in range(m["K"]):
            for nm in ("f0", "tau", "w", "phi", "zeta"):
                prows.append(dict(model=LABEL[n], group=f"comp{j+1}",
                                  param=nm, **stats(al[nm][:, j])))
            b = al["w"][:, j] * al["tau"][:, j] ** (-al["phi"][:, j])
            prows.append(dict(model=LABEL[n], group=f"comp{j+1}",
                              param="B=w*tau^-phi", **stats(b)))
    pd.DataFrame(prows).to_csv(ROOT / "grid_posterior_summary.csv",
                               index=False, float_format="%.6g")

    # ---------- 4. 逐频点总阻抗预测 + 模型间差异 ----------
    Zmed = {}
    for n, m in models.items():
        s, f = m["s"], m["f"]
        Zall = np.zeros((f.size, len(s["R_inf"])), complex)
        for k in range(m["K"]):
            Zall += hn_component(f, s["w"][:, k], s["tau"][:, k],
                                 s["phi"][:, k], s["zeta"][:, k])
        Zall += s["R_inf"][None, :]
        if "Aw" in s:
            Zall += s["Aw"][None, :] / np.sqrt(1j * 2 * np.pi * f[:, None])
        if "Rwo" in s and "tauD" in s:
            import bm_diff as D
            for i in range(len(s["Rwo"])):
                Zall[:, i] += D.z_wopen(f, float(s["Rwo"][i]),
                                        float(s["tauD"][i]))
        Zmed[n] = np.median(Zall, axis=1)
    pred = []
    for i, fi in enumerate(models["G_K1_none"]["f"]):
        row = dict(f_Hz=float(fi),
                   Zre_obs=float(models["G_K1_none"]["Z"][i].real),
                   Zim_obs=float(models["G_K1_none"]["Z"][i].imag))
        for n in NAMES:
            z = Zmed[n][i]
            row[n + "_Zre"] = float(z.real)
            row[n + "_Zim"] = float(z.imag)
            row[n + "_absZ"] = float(abs(z))
        for dif in ("none", "Ws", "Wo"):
            a, b = Zmed[f"G_K1_{dif}"][i], Zmed[f"G_K2_{dif}"][i]
            row[f"K1_vs_K2_{dif}_relDiff"] = float(abs(a - b) / max(abs(a), 1e-300))
        k2 = [Zmed[f"G_K2_{d}"][i] for d in ("none", "Ws", "Wo")]
        row["K2_family_max_relDiff"] = float(
            max(abs(x - y) for x in k2 for y in k2)
            / max(np.mean([abs(x) for x in k2]), 1e-300))
        pred.append(row)
    pd.DataFrame(pred).to_csv(ROOT / "predictive_comparison.csv",
                              index=False, float_format="%.6g")

    # ---------- 5. 各分量复阻抗贡献 + C_k(f)（只报告，不作门禁） ----------
    crows = []
    for n, m in models.items():
        f, s, al = m["f"], m["s"], m["al"]
        Ztot = np.zeros((f.size, len(s["R_inf"])), complex)
        Zc = []
        for k in range(m["K"]):
            zc = hn_component(f, al["w"][:, k], al["tau"][:, k],
                              al["phi"][:, k], al["zeta"][:, k])
            Zc.append(zc)
            Ztot += zc
        Ztot += s["R_inf"][None, :]
        for i, fi in enumerate(f):
            at = abs(Ztot[i])
            for k, zc in enumerate(Zc):
                re, im = zc[i].real, zc[i].imag
                crows.append(dict(
                    model=LABEL[n], comp=f"comp{k+1}", f_Hz=float(fi),
                    Zk_re_q50=q(re, 50), Zk_re_q05=q(re, 5), Zk_re_q95=q(re, 95),
                    Zk_im_q50=q(im, 50), Zk_im_q05=q(im, 5), Zk_im_q95=q(im, 95),
                    absZk_q50=q(np.abs(zc[i]), 50),
                    Ck_absZ_ratio=q(np.abs(zc[i]) / np.maximum(at, 1e-300), 50),
                    Zmodel_Re_q50=q(Ztot[i].real, 50),
                    Zmodel_Im_q50=q(Ztot[i].imag, 50),
                ))
    pd.DataFrame(crows).to_csv(ROOT / "component_contribution_grid.csv",
                               index=False, float_format="%.6g")

    # ---------- 6. 复制结果，写 grid_summary.json ----------
    for n, m in models.items():
        dst = ROOT / "results" / n
        dst.mkdir(parents=True, exist_ok=True)
        for p in (m["js_p"], m["npz_p"], m["log_p"]):
            if p.resolve() != (dst / p.name).resolve():
                shutil.copy2(p, dst / p.name)
    summary = dict(
        case="Blank_0.1KBr_pH10_EIS_PtCu_1.2V / P2 2x3 统一网格",
        created="2026-11-05",
        note="只读分析；未修改 bmdrt 代码 / 参数 / 先验 / 种子；未提交仓库",
        design=dict(
            grid=LABEL,
            fixed="同一 50 点输入、config.json 原样（quick=false, seed=0, "
                  "zeta=1.0, sigma_floor=0.0）、full 3000/1000/3 链、"
                  "逐频点联合似然 Normal(Z',s_re)+Normal(Z'',s_im)",
            krange="每组用 --krange K,K 强制单 K，消除 K 选择过程差异",
            loo="从 NPZ 后验样本用 bm_real.loglik_matrix(n_max=800) 重算逐点似然，"
                "az.loo(pointwise=True) 取 Pareto-k，az.compare 统一比较 6 模型",
        ),
        compare={lab: dict(elpd_diff=float(cmp.loc[lab, "elpd_diff"]),
                           dse=float(cmp.loc[lab, "dse"]),
                           weight=float(cmp.loc[lab, "weight"]),
                           rank=int(cmp.loc[lab, "rank"]))
                 for lab in LABEL.values()},
        pareto_k={LABEL[n]: dict(max=float(loose[n]["pk"].max()),
                                frac_gt07=loose[n]["pk_frac"],
                                n_gt07=int((loose[n]["pk"] > 0.7).sum()))
                  for n in NAMES},
        models={r["model"]: r for r in rows},
    )
    (ROOT / "grid_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")

    # ---------- 7. 控制台摘要 ----------
    print("=== az.compare（6 模型统一，elpd 越大越好） ===")
    print(cmp[["rank", "elpd_loo", "p_loo", "elpd_diff", "dse", "weight"]]
          .sort_values("rank").to_string())
    print("\n=== 统一比较表（节选） ===")
    cols = ["model", "rms_re", "rms_im", "acf", "trend_sigma_im",
            "elpd_loo_recalc", "elpd_diff", "dse", "weight",
            "pareto_k_max", "pareto_k_gt07", "p_peak_out_max"]
    print(grid[cols].to_string(index=False))
    print("\n=== 扩散参数 / 慢分量 B2 ===")
    cols2 = ["model", "Aw_q50", "Rwo_q50", "tauD_q50", "f_D_Hz",
             "A_W_equiv", "p_diff_small", "w2_cv", "tau2_cv", "B2_cv",
             "B2_q50", "f0_2_q50"]
    print(grid[[c for c in cols2 if c in grid.columns]].to_string(index=False))
    print("\n=== 预测差异汇总 ===")
    pp = pd.DataFrame(pred)
    print(pp[["K1_vs_K2_none_relDiff", "K1_vs_K2_Ws_relDiff",
              "K1_vs_K2_Wo_relDiff", "K2_family_max_relDiff"]]
          .describe().loc[["mean", "max"]].to_string())
    print("\nDONE ->", ROOT)


if __name__ == "__main__":
    main()
