"""只读分析器：从已落盘的 4 组 BM-DRT 结果生成可复现诊断材料。

本脚本不修改 D:/eis训练数据/bmdrt 下任何文件，只读取 JSON/NPZ/日志，
所有产物写入本案例目录。依赖 bm_core.match_components 只为保持与
bm_real 完全相同的「跨样本分量匹配 + 峰位出窗」口径。

运行：
  D:/autoeis_env/Scripts/python.exe analyze.py
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# 归档到仓库 docs/cases/ 后也能自动定位仓库根；工作副本回退到绝对路径。
REPO = next((q for q in [ROOT, *ROOT.parents] if (q / "bm_core.py").exists()),
            Path(r"D:/eis训练数据/bmdrt"))
sys.path.insert(0, str(REPO))
import bm_core as B  # noqa: E402  (只读 import，用于口径一致)

MODELS = {
    "K1_none": dict(
        dir=REPO / "out/Blank_0.1KBr_pH10_EIS_PtCu_1.2V",
        stem="real_Blank_0.1KBr_pH10_EIS_PtCu_1.2V",
        cmd="bmdrt.py auto inbox/Blank_0.1KBr_pH10_EIS_PtCu_1.2V.txt"
            "  (config.json 原样；DRT 局部扫描 K=1,2)",
        label="原运行 K=1",
    ),
    "K1_cut1Hz": dict(
        dir=REPO / "out/_diag_cut1Hz/Blank_cut1Hz",
        stem="real_Blank_cut1Hz",
        cmd="bmdrt.py auto out/_diag_inputs/Blank_cut1Hz.txt --no-move"
            " --out out/_diag_cut1Hz --krange 1,2",
        label="低频截断 >=1 Hz",
    ),
    "K2_none": dict(
        dir=REPO / "out/_diag_K2/Blank_K2",
        stem="real_Blank_K2",
        cmd="bmdrt.py auto out/_diag_inputs/Blank_K2.txt --no-move"
            " --out out/_diag_K2 --krange 2,2",
        label="强制 K=2（不加扩散）",
    ),
    "K2_Ws": dict(
        dir=REPO / "out/_diag_Ws/Blank_full_Ws",
        stem="real_Blank_full_Ws",
        cmd="bmdrt.py auto out/_diag_inputs/Blank_full_Ws.txt --no-move"
            " --out out/_diag_Ws --krange 1,2 --diffusion Ws",
        label="K=2 + Ws",
    ),
}


ORIGINAL_DIRS = {k: c["dir"] for k, c in MODELS.items()}


def q(a, p):
    return float(np.percentile(np.asarray(a, float), p))


def stats(a):
    a = np.asarray(a, float)
    return dict(q05=q(a, 5), q50=q(a, 50), q95=q(a, 95),
                mean=float(a.mean()), sd=float(a.std(ddof=1)))


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def hn_component(f, w, tau, phi, zeta):
    """向量化 Z_j(f, sample) = w / (1 + (j*2*pi*f*tau)^phi)^zeta，形状 (n_f, n_s)。

    与 bm_core._hn_z 逐点等价，只是把「样本」维一起广播，便于取后验分位数。
    """
    x = np.maximum(2 * np.pi * np.asarray(f, float)[:, None]
                   * np.asarray(tau, float)[None, :], 1e-300)
    ph = np.asarray(phi, float)[None, :]
    mag = np.exp(ph * np.log(x))
    ang = ph * (np.pi / 2)
    denom = 1.0 + mag * (np.cos(ang) + 1j * np.sin(ang))
    return np.asarray(w, float)[None, :] / denom ** np.asarray(zeta, float)[None, :]


def load_model(key, cfg):
    d, stem = cfg["dir"], cfg["stem"]
    js_p, npz_p, log_p = d / f"{stem}.json", d / f"{stem}_samples.npz", d / "run.log"
    js = json.loads(js_p.read_text(encoding="utf-8"))
    z = np.load(npz_p, allow_pickle=True)
    s = {k: np.asarray(z[k], float) for k in z.files if k not in ("f", "Zre", "Zim")}
    f = np.asarray(z["f"], float)
    Zre, Zim = np.asarray(z["Zre"], float), np.asarray(z["Zim"], float)
    K = int(js["K"])
    if "zeta" not in s:
        s["zeta"] = np.ones((len(s["R_inf"]), K))
    mm = B.match_components(s)
    o = np.argsort(-np.median(mm["f0"], axis=0))
    al = {}
    for nm in ("tau", "w", "phi", "zeta", "pi", "f0"):
        al[nm] = mm[nm][:, o]
    # 日志里的「xσ(实)/yσ(虚)」是相对残差零分布标准化的值，
    # 与 JSON 里的 trend_re/trend_im（未标准化）不是同一个量，必须分开存。
    txt = log_p.read_text(encoding="utf-8", errors="replace")
    mt = re.search(r"平滑趋势\s*([\d.]+)σ\(实\)\s*/\s*([\d.]+)σ\(虚\)", txt)
    trend_sigma = (float(mt.group(1)), float(mt.group(2))) if mt else (np.nan, np.nan)
    mp = re.search(r"零分布第\s*([\d.]+)\s*百分位", txt)
    trend_pct = float(mp.group(1)) if mp else np.nan
    mj = re.search(r"残差有系统性弯曲\*\*：趋势\s*[\d.]+σ 位于白噪声零分布第\s*([\d.]+)\s*百分位", txt)
    trend_pct_judge = float(mj.group(1)) if mj else np.nan
    return dict(key=key, cfg=cfg, js=js, s=s, f=f, Zre=Zre, Zim=Zim,
               K=K, al=al, js_p=js_p, npz_p=npz_p, log_p=log_p,
               trend_sigma=trend_sigma, trend_pct=trend_pct,
               trend_pct_judge=trend_pct_judge)


def main():
    # 案例包自包含：results/ 已有副本时优先读它，这样即使仓库 out/ 被清理也能重算。
    for key, c in MODELS.items():
        local = ROOT / "results" / key
        if (local / f"{c['stem']}_samples.npz").exists():
            c["dir"] = local
    models = {k: load_model(k, c) for k, c in MODELS.items()}

    # ---------- 1. 后验摘要（长表） ----------
    rows = []
    for key, m in models.items():
        s, al, f = m["s"], m["al"], m["f"]
        glob = {"R_inf": s["R_inf"], "W": s["W"],
                "s_re": s["s_re"], "s_im": s["s_im"]}
        for nm in ("Rw", "Aw", "Rwo", "tauD"):
            if nm in s:
                glob[nm] = s[nm]
        for nm, a in glob.items():
            rows.append(dict(model=key, comp="global", param=nm, **stats(a)))
        for j in range(m["K"]):
            for nm in ("f0", "tau", "w", "phi", "zeta", "pi"):
                rows.append(dict(model=key, comp=f"comp{j+1}", param=nm,
                                 **stats(al[nm][:, j])))
            pz = al["phi"][:, j] * al["zeta"][:, j]
            rows.append(dict(model=key, comp=f"comp{j+1}", param="phi*zeta",
                             **stats(pz)))
        # 每个分量的峰位出窗概率（与 bm_real 的 p_tau_out 同口径）
        for j in range(m["K"]):
            f0 = al["f0"][:, j]
            p_out = float(np.mean((f0 < f.min()) | (f0 > f.max())))
            rows.append(dict(model=key, comp=f"comp{j+1}",
                             param="P(f0 out of window)",
                             q05=np.nan, q50=p_out, q95=np.nan,
                             mean=np.nan, sd=np.nan))
    pd.DataFrame(rows).to_csv(ROOT / "posterior_summary.csv",
                              index=False, float_format="%.6g")

    # ---------- 2. 相关系数对照 ----------
    crows = []
    for key, m in models.items():
        s, al = m["s"], m["al"]
        lr = np.log(np.maximum(s["R_inf"], 1e-300))
        for j in range(m["K"]):
            lw = np.log(np.maximum(al["w"][:, j], 1e-300))
            lt = np.log(np.maximum(al["tau"][:, j], 1e-300))
            ph = al["phi"][:, j]
            pairs = {
                "ln R_inf ~ ln w_%d" % (j + 1): corr(lr, lw),
                "ln w_%d ~ ln tau_%d" % (j + 1, j + 1): corr(lw, lt),
                "ln R_inf ~ ln tau_%d" % (j + 1): corr(lr, lt),
                "ln w_%d ~ phi_%d" % (j + 1, j + 1): corr(lw, ph),
                "ln R_inf ~ phi_%d" % (j + 1): corr(lr, ph),
            }
            for p, v in pairs.items():
                crows.append(dict(model=key, pair=p, rho=round(v, 4)))
    pd.DataFrame(crows).to_csv(ROOT / "correlation_comparison.csv",
                               index=False, float_format="%.4f")

    # ---------- 3. 峰位出窗概率稳定性 ----------
    srows = []
    rng = np.random.default_rng(0)
    for key in ("K1_none", "K1_cut1Hz", "K2_none", "K2_Ws"):
        m = models[key]
        f, al = m["f"], m["al"]
        for j in range(m["K"]):
            f0 = al["f0"][:, j]
            out = (f0 < f.min()) | (f0 > f.max())
            n = len(out)
            # 假设 chain-major (3 链 x 1000 draws)；仅作参考
            seg = [float(out[i * n // 3:(i + 1) * n // 3].mean()) for i in range(3)]
            # 不依赖顺序假设：200 次随机重采 1/3 样本
            boot = [float(out[rng.choice(n, n // 3, replace=False)].mean())
                    for _ in range(200)]
            srows.append(dict(
                model=key, comp=f"comp{j+1}",
                f_min=float(f.min()), f_max=float(f.max()),
                f0_q05=q(f0, 5), f0_q50=q(f0, 50), f0_q95=q(f0, 95),
                p_out_all=float(out.mean()),
                p_out_seg1=seg[0], p_out_seg2=seg[1], p_out_seg3=seg[2],
                p_out_boot_q05=q(boot, 5), p_out_boot_q50=q(boot, 50),
                p_out_boot_q95=q(boot, 95),
                p_out_json=(m["js"].get("components") or [{}])[j].get(
                    "p_peak_outside", np.nan),
            ))
    pd.DataFrame(srows).to_csv(ROOT / "peak_out_stability.csv",
                               index=False, float_format="%.6g")

    # ---------- 4. 慢分量的逐频点阻抗贡献（K=2 两组） ----------
    contrib = []
    for key in ("K2_none", "K2_Ws"):
        m = models[key]
        f, al, s = m["f"], m["al"], m["s"]
        j = m["K"] - 1  # 按 f0 降序，最后一个 = 最慢分量
        Zj = hn_component(f, al["w"][:, j], al["tau"][:, j],
                          al["phi"][:, j], al["zeta"][:, j])  # (n_f, n_s)
        Zall = np.zeros_like(Zj)
        for k in range(m["K"]):
            Zall += hn_component(f, al["w"][:, k], al["tau"][:, k],
                                 al["phi"][:, k], al["zeta"][:, k])
        Ztot = s["R_inf"][None, :] + Zall
        # Ws 元件是 Aw / sqrt(j*2*pi*f)；Rw 只是采样参数，
        # 不能拿 Rw=0.072 Ω 说「Ws 幅值可忽略」。
        Zw = (s["Aw"][None, :] / np.sqrt(1j * 2 * np.pi * f[:, None])
              if "Aw" in s else None)
        for i, fi in enumerate(f):
            re, im = Zj[i].real, Zj[i].imag
            tz_re, tz_im = Ztot[i].real, Ztot[i].imag
            d = dict(
                model=key, f_Hz=float(fi),
                Zre_obs=float(m["Zre"][i]), Zim_obs=float(m["Zim"][i]),
                comp2_re_q05=q(re, 5), comp2_re_q50=q(re, 50), comp2_re_q95=q(re, 95),
                comp2_im_q05=q(im, 5), comp2_im_q50=q(im, 50), comp2_im_q95=q(im, 95),
                frac_of_model_Re=float(np.median(re / np.maximum(np.abs(tz_re), 1e-300))),
                abs_frac_of_model_Im=float(np.median(
                    np.abs(im) / np.maximum(np.abs(tz_im), 1e-300))),
            )
            if Zw is not None:
                d.update(Ws_re_q50=float(np.median(Zw[i].real)),
                         Ws_im_q50=float(np.median(Zw[i].imag)))
            contrib.append(d)
    pd.DataFrame(contrib).to_csv(ROOT / "component_contribution.csv",
                                 index=False, float_format="%.6g")

    # ---------- 5. 复制输入与结果，固定基线 ----------
    (ROOT / "inputs").mkdir(exist_ok=True)
    for src, dst in (
        (REPO / "archive/Blank_0.1KBr_pH10_EIS_PtCu_1.2V.txt",
         ROOT / "inputs/original_50pt.txt"),
        (REPO / "out/_diag_inputs/Blank_cut1Hz.txt",
         ROOT / "inputs/cut1Hz_40pt.txt"),
    ):
        if src.exists():
            shutil.copy2(src, dst)
        elif not dst.exists():
            raise FileNotFoundError(f"输入副本缺失且源已清理：{dst}")
    for key, m in models.items():
        dst = ROOT / "results" / key
        dst.mkdir(parents=True, exist_ok=True)
        for p in (m["js_p"], m["npz_p"], m["log_p"]):
            if p.resolve() != (dst / p.name).resolve():
                shutil.copy2(p, dst / p.name)

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    cfg = json.loads((REPO / "config.json").read_text(encoding="utf-8"))
    baseline = dict(
        case="Blank_0.1KBr_pH10_EIS_PtCu_1.2V",
        created="2026-11-05",
        note="只读基线：本轮未修改任何 bmdrt 代码 / 参数 / 先验 / 种子",
        code=dict(repo=str(REPO), head=head, working_tree_clean=(dirty == ""),
                  dirty_files=dirty.splitlines(),
                  remote="https://github.com/blwsh-1/dmdrt.git"),
        env=dict(python=r"D:/autoeis_env/Scripts/python.exe",
                 numpy=np.__version__, pandas=pd.__version__),
        config_snapshot={k: cfg.get(k) for k in ("fit", "kk", "diag", "drt")},
        prior_and_parameterization=dict(
            phi_prior="Uniform(0.5, 1.0) —— 下界 0.5 是硬边界",
            zeta="zeta_fixed=1.0 钉死，不是采样参数，故 phi*zeta == phi",
            omega="2*pi*f",
            note="慢分量 phi 后验 q05 ≈ 0.501 紧贴先验下界；"
                 "phi≈0.5 不能单独当作扩散过程的独立证据"),
        inputs={p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted((ROOT / "inputs").glob("*.txt"))},
        models={},
    )
    for key, m in models.items():
        js = m["js"]
        baseline["models"][key] = dict(
            label=m["cfg"]["label"], command=m["cfg"]["cmd"],
            source_dir=str(ORIGINAL_DIRS[key]),
            K=js["K"], k_selection_status=js.get("k_selection_status"),
            sample_grade=js.get("sample_grade"),
            f_min=js.get("f_min"), f_max=js.get("f_max"), n=js.get("n"),
            include_L=js.get("include_L"), diffusion=js.get("diffusion"),
            diffusion_params=js.get("diffusion_params"),
            sigma_floor=js.get("sigma_floor"),
            rms_re=js.get("rms_re"), rms_im=js.get("rms_im"),
            sigma_re=js.get("sigma_re"), sigma_im=js.get("sigma_im"),
            acf=js.get("acf"),
            trend_re_raw=js.get("trend_re"), trend_im_raw=js.get("trend_im"),
            trend_sigma_re=m["trend_sigma"][0],
            trend_sigma_im=m["trend_sigma"][1],
            trend_null_percentile_rough=m["trend_pct"],
            trend_null_percentile_judgement=m["trend_pct_judge"],
            p_diff_small=js.get("p_diff_small"),
            elpd_loo=js.get("elpd_loo"), elpd_diff=js.get("elpd_diff"),
            dse=js.get("dse"),
            divergences=js.get("divergences"), tree_cap=js.get("tree_cap"),
            p_peak_out_max=js.get("p_peak_out_max"),
            k_selection=js.get("k_selection"),
            warnings=js.get("warnings"),
            files=[p.name for p in (m["js_p"], m["npz_p"], m["log_p"])],
        )
    (ROOT / "baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")

    # ---------- 6. 控制台摘要（供撰写 README） ----------
    print("HEAD", head, "dirty:", repr(dirty))
    for key, m in models.items():
        js = m["js"]
        print(f"\n[{key}] K={js['K']} status={js.get('k_selection_status')} "
              f"diffusion={js.get('diffusion')} "
              f"rms={js.get('rms_re'):.4f}/{js.get('rms_im'):.4f} "
              f"acf={js.get('acf'):.3f} "
              f"trend={js.get('trend_re'):.2f}/{js.get('trend_im'):.2f}σ "
              f"p_peak_out_max={js.get('p_peak_out_max')}")
        for c in js.get("components") or []:
            print("   comp f0=%.4g Hz  w=%.4g  phi=%.4f  p_peak_out=%.3f"
                  % (c["f0"], c["w"], c["phi"], c["p_peak_outside"]))
    print("\n--- correlation ---")
    print(pd.DataFrame(crows).to_string(index=False))
    print("\n--- peak-out stability ---")
    print(pd.DataFrame(srows).to_string(index=False))
    print("\n--- posterior (phi/zeta/phi*zeta, slow comp) ---")
    ps = pd.DataFrame(rows)
    print(ps[(ps.param.isin(["phi", "zeta", "phi*zeta", "f0", "w", "tau"]))
             & (ps.comp == "comp2")].to_string(index=False))
    print("\n--- contribution (representative f, closest grid point) ---")
    cc = pd.DataFrame(contrib)
    for model in ("K2_none", "K2_Ws"):
        sub = cc[cc.model == model]
        picks = [sub.f_Hz.iloc[int(np.argmin(np.abs(sub.f_Hz.values - x)))]
                 for x in (0.1, 0.2, 0.5, 1.0, 4.64, 100, 1000, 10000)]
        print(f"[{model}]")
        print(sub[sub.f_Hz.isin(picks)].to_string(index=False))
    print("\nDONE ->", ROOT)


if __name__ == "__main__":
    main()
