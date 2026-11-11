"""AZ31-4Li_100 上七个电路的残差结构对比（含文献 Shahri 2024 Table 1）。

回答一个具体问题：**「残差一样大」到底是幅值相近，还是连随频率变化的形状和
正负号也一致？** 这两件事指向完全不同的诊断方向：

* 幅值相近 + 形状不同 → 各模型各自在不同频段失配，问题在模型表达能力；
* 幅值相近 + 形状/符号一致 → 存在一个**任何常规电路都吸收不掉的公共结构**，
  它更像数据侧的系统性贡献（仪器/夹具/非稳态/分布式效应）。

模型
----
  VOIGT_K1   本项目生产管线在该谱上选中的模型（K=1，不含 L，后验中位数）
  NESTED_mcmc 本项目嵌套拓扑 `parallel/L=2` 的 MCMC 后验中位谱（口径不同，只作对照）
  NESTED_LS   同一阶梯（C4）用**与外部电路完全相同**的相对加权 LS 拟合（幅值可比）
  EXT1       外部软件 `adsorption_2RQ`：Rs+(Rct||CPE1)+(CPEdl||(Rct2||(Rads+jωLads)))
  EXT2       外部软件 `adsorption_RQ` ：Rs+(CPEdl||(Rct||(Rads+jωLads)))
  LIT_C1     文献 Shahri 2024 Table 1 Model 1：`R(Q(R(QR(LR))))`（最内层 R+L 串联）
  LIT_C2     文献 Table 1 Model 2：三层嵌套，最内层 `R4∥L`
  LIT_C3     文献 Table 1 Model 3：`(R1∥QPE1)` 串联 + 两层嵌套
  LINKK      Lin-KK 无模型拟合，给出「因果 LTI 模型」的残差地板

C1 与 C4 是**同一个阻抗函数**（R-L 星三角，见 `NESTED_TOPOLOGY_PROJECT.md` §9.4.30.2），
而 C4 就是我们的 `parallel/L=2`；所以 `LIT_C1` 与 `NESTED_LS` 的预测谱必须重合
（脚本会打印两者 `max|ΔZ|/|Z|` 作独立核对）。真正另立拓扑的只有 C2、C3。

外部两个电路先由 `ext_circuit_repro.py` 复现（Rwp 与参数与截图差 <0.5%），
本脚本直接复用其结果，不再重复拟合。文献三个电路的阻抗函数不复制，直接复用
`evidence/nested_topology/m4_four_circuits.py` 里的 `_z_c1/_z_c2/_z_c3`（单一来源），
但**优化器与 NESTED_LS 相同**（DE 全局 + LS 精修、相对加权），保证幅值可比。

跑法
----
    python docs/round_2026-11-11/resid_structure_cmp.py
    python docs/round_2026-11-11/resid_structure_cmp.py --specs AZ31-4Li_100 --no-plot
    python docs/round_2026-11-11/resid_structure_cmp.py --no-lit      # 只跑原来六个
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import bm_core as B                    # noqa: E402
import bm_experiments as E             # noqa: E402
import bm_kk as KK                     # noqa: E402
import bm_real as R                    # noqa: E402
import bm_top as T                     # noqa: E402
from ext_circuit_repro import predict as ext_predict   # noqa: E402

SPEC_DIR = os.path.join("out", "_m6_samples", "z_csv")
PROD_DIR = os.path.join("out", "m6", "b3b")
NESTED_JSON = os.path.join("out", "m4_real_nested_az31.json")
EXT_JSON = os.path.join("out", "ext_circuit_repro.json")
OUT_JSON = os.path.join("out", "resid_structure_cmp.json")
OUT_PNG = os.path.join("out", "resid_structure_cmp.png")

BANDS = ((0.01, 1.0, "low"), (1.0, 100.0, "mid"), (100.0, 1e5, "high"))


# ---------------------------------------------------------------------------
# 四个模型的预测
# ---------------------------------------------------------------------------

def _voigt_median_spectrum(spec, n_draw=400):
    """后验预测：等距抽 `n_draw` 个样本算 Z(f)，中心取逐点中位数、带取 5/95 分位。

    `bm_core.model_z` 只吃单个样本（`float(p["R_inf"])`），所以这里必须逐个算。
    """
    p = os.path.join(PROD_DIR, spec, "seed0", spec, f"real_{spec}_samples.npz")
    d = np.load(p, allow_pickle=True)
    f = d["f"]
    n = d["R_inf"].size
    idx = np.linspace(0, n - 1, min(n_draw, n)).astype(int)
    Zr = np.empty((idx.size, f.size))
    Zi = np.empty((idx.size, f.size))
    for row, k in enumerate(idx):
        par = dict(R_inf=float(d["R_inf"][k]), W=float(d["W"][k]),
                   pi=np.asarray(d["pi"][k], float), tau=np.asarray(d["tau"][k], float),
                   phi=np.asarray(d["phi"][k], float), zeta=np.asarray(d["zeta"][k], float))
        z = B.model_z(f, par, include_L=False)
        Zr[row], Zi[row] = z.real, z.imag
    Z = np.median(Zr, axis=0) + 1j * np.median(Zi, axis=0)
    return f, Z, dict(re=(np.percentile(Zr, 5, axis=0), np.percentile(Zr, 95, axis=0)),
                      im=(np.percentile(Zi, 5, axis=0), np.percentile(Zi, 95, axis=0)))


def _nested_params(spec):
    d = json.load(open(NESTED_JSON, encoding="utf-8"))
    for r in d["rows"]:
        if r["label"] == "parallel/L=2":
            return r["params"], r
    raise KeyError("parallel/L=2 不在 nested 结果里")


def _z_c4(f, x, w_ref):
    """项目嵌套拓扑 `parallel/L=2` 的等价梯级（= Table 1 的 C4，见 §9.4.30）：

        Rs + (Q1 || (R1 + (Q2 || (R2 + (R3 || jωL)))))

    `x` 为 LS 向量：前 5 个是 log10（Rs/R1/R2/R3/A1），n1 线性，再 A2 log、n2 线性、L log。
    """
    w = 2.0 * np.pi * np.asarray(f, float)
    Rs, R1, R2, R3, A1 = 10.0 ** x[0], 10.0 ** x[1], 10.0 ** x[2], 10.0 ** x[3], 10.0 ** x[4]
    n1, A2, n2, L = x[5], 10.0 ** x[6], x[7], 10.0 ** x[8]

    def cpe(A, n):
        return A * (w_ref / w) ** n * np.exp(-1j * n * np.pi / 2.0)

    def par(*zs):
        return 1.0 / sum(1.0 / z for z in zs)

    return Rs + par(cpe(A1, n1), R1 + par(cpe(A2, n2), R2 + par(R3, 1j * w * L)))


_C4_BOUNDS = [(-1, 4), (0, 5), (0, 5), (0, 5), (-3, 5), (0.05, 1.0),
              (-3, 5), (0.05, 1.0), (-2, 7)]


def fit_c4_ls(f, Z, seed=3):
    """与外部两个电路**完全相同**的口径（相对加权 LS、DE 全局 + LS 精修），
    只换拓扑。这样幅值才可比——MCMC 后验中位谱不是最优拟合，不能拿来比 Rwp。
    """
    from scipy.optimize import differential_evolution, least_squares
    w_ref = 2.0 * np.pi * np.sqrt(f.min() * f.max())
    wa = np.abs(Z)

    def resid(x):
        d = _z_c4(f, x, w_ref) - Z
        return np.concatenate([d.real / wa, d.imag / wa])

    def cost(x):
        return float(np.sqrt(np.mean(resid(x) ** 2)))

    lo = [b[0] for b in _C4_BOUNDS]
    hi = [b[1] for b in _C4_BOUNDS]
    de = differential_evolution(cost, _C4_BOUNDS, seed=seed, tol=1e-10,
                                maxiter=1200, popsize=35, polish=False)
    r = least_squares(resid, de.x, bounds=(lo, hi), max_nfev=30000,
                      ftol=1e-14, xtol=1e-14, gtol=1e-14)
    return _z_c4(f, r.x, w_ref), dict(x=r.x.tolist())


# ---------------------------------------------------------------------------
# 文献电路（Shahri 2024 Table 1）
# ---------------------------------------------------------------------------

_LIT_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                        "evidence", "nested_topology")


def _lit_module():
    """§17 的四个电路实现只写一份，这里直接复用，不复制阻抗函数。"""
    if _LIT_DIR not in sys.path:
        sys.path.insert(0, _LIT_DIR)
    import m4_four_circuits as F4
    return F4


def fit_lit(name, f, Z, seed=3, maxiter=800, popsize=20):
    """文献电路的 LS 拟合：**与 `fit_c4_ls` 同一优化器与同一权重**（DE + LS 精修、
    相对加权）。换优化器会让 C2/C3 多 3 个参数的比较失真，所以这里不省。

    参数化：`n*` 线性（支撑 [0.5,1]），其余 log10；边界 `zmax·10^±6`，`L` 取
    `10^-6..10^8 H`。返回 (Z_fit, meta)。
    """
    from scipy.optimize import differential_evolution, least_squares

    F4 = _lit_module()
    spec = F4.CIRCUITS[name]
    keys, fn = spec["keys"], spec["fn"]
    w_ref = 2.0 * np.pi * np.sqrt(f.min() * f.max())
    zmax = float(np.max(np.abs(Z)))
    wa = np.abs(Z)
    lo, hi = [], []
    for k in keys:
        if k.startswith("n"):
            lo.append(0.5), hi.append(1.0)
        elif k == "L":
            lo.append(-6.0), hi.append(8.0)
        else:
            lo.append(float(np.log10(zmax)) - 6.0), hi.append(float(np.log10(zmax)) + 6.0)
    bounds = list(zip(lo, hi))

    def decode(x):
        return {k: (float(x[i]) if k.startswith("n") else float(10.0 ** x[i]))
                for i, k in enumerate(keys)}

    def resid(x):
        d = fn(f, decode(x), w_ref) - Z
        return np.concatenate([d.real / wa, d.imag / wa])

    def cost(x):
        return float(np.sqrt(np.mean(resid(x) ** 2)))

    de = differential_evolution(cost, bounds, seed=seed, tol=1e-10,
                                maxiter=maxiter, popsize=popsize, polish=False)
    r = least_squares(resid, de.x, bounds=(lo, hi), max_nfev=30000,
                      ftol=1e-14, xtol=1e-14, gtol=1e-14)
    x = r.x
    return fn(f, decode(x), w_ref), dict(params=decode(x), n_params=len(keys),
                                         de_cost=float(de.fun), ls_cost=float(cost(x)))


def build_models(spec, with_lit=True, lit_iters=800):
    f, Z = R.load_spectrum(os.path.join(SPEC_DIR, spec + ".csv"))
    models = {}

    fv, Zv, band = _voigt_median_spectrum(spec)
    assert np.allclose(fv, f), "后验谱的频点与 csv 不一致"
    models["VOIGT_K1"] = dict(Z=Zv, band=band, note="生产管线选中 K=1（无 L），后验逐点中位数")

    pn, rn = _nested_params(spec)
    models["NESTED_mcmc"] = dict(
        Z=T.z_nested(f, pn, branch="parallel", layers=2), band=None,
        note=f"嵌套 parallel/L=2，quick 档 MCMC 后验逐点中位数、acf={rn['acf']:.3f}"
             "（**不是最优拟合**，不能用来比 Rwp）")

    Zc, _ = fit_c4_ls(f, Z)
    models["NESTED_LS"] = dict(Z=Zc, band=None,
                               note="同一阶梯（C4）用与外部电路相同的相对加权 LS 拟合")

    ext = json.load(open(EXT_JSON, encoding="utf-8"))[spec]
    for name in ("EXT1", "EXT2"):
        rec = ext[name]
        models[name] = dict(Z=ext_predict(f, Z, name, rec["params"]), band=None,
                            note=f"外部软件复现，Rwp {rec['rwp']:.3f}%")

    # 文献四条（Shahri 2024 Table 1）。C1 ≡ C4 ≡ 我们的 `parallel/L=2`（R-L 星三角，
    # §9.4.30.2），所以只跑 C1 作独立核对，不重复跑 C4；C2/C3 是真正另立的两条拓扑。
    if with_lit:
        lit_note = {
            "LIT_C1": "文献 Table 1 Model 1（R(Q(R(QR(LR))))），最内层 R+L 串联"
                      "；与 C4/NESTED_LS 是同一个阻抗函数（星三角），此处作恒等核对",
            "LIT_C2": "文献 Table 1 Model 2：三层嵌套，最内层 R4∥L（比 C4 多一层、多 3 参数）",
            "LIT_C3": "文献 Table 1 Model 3：(R1∥QPE1) 串联 + 两层嵌套（多 3 参数，接法不同）",
        }
        for nm, cir in (("LIT_C1", "C1"), ("LIT_C2", "C2"), ("LIT_C3", "C3")):
            Zl, meta = fit_lit(cir, f, Z, maxiter=lit_iters)
            models[nm] = dict(Z=Zl, band=None, note=f"{lit_note[nm]}；LS {meta['n_params']} 参数",
                              params=meta["params"], n_params=meta["n_params"])

    # 无模型对照：Lin-KK（一串 RC 的分布）。它按构造满足 Kramers-Kronig，
    # 所以它的残差就是**这份数据自身违反 KK 的那一部分**——任何因果 LTI 模型
    # （含所有等效电路）都不可能比它更好，因此它给出「残差地板」。
    kk = KK.lin_kk(f, Z, c=0.85, max_M=50)
    models["LINKK"] = dict(Z=np.asarray(kk["Z_fit"], complex), band=None,
                           note=f"Lin-KK 无模型拟合（M={kk.get('M')}、μ={kk.get('mu'):.3f}），"
                                "按构造因果，给出因果 LTI 模型的残差地板")
    return f, Z, models


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def _sign_changes(x):
    s = np.sign(x)
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1])) if s.size else 0


def describe(f, Z, Zm):
    r = Zm - Z
    w = np.abs(Z)
    q = np.abs(r) / w
    band = {}
    for a, b, tag in BANDS:
        m = (f >= a) & (f < b)
        band[tag] = dict(n=int(m.sum()),
                         rms_rel=float(np.sqrt(np.mean(q[m] ** 2)) * 100) if m.any() else None)
    return dict(
        rwp=float(np.sqrt(np.mean(q ** 2)) * 100),
        rmse=float(np.sqrt(np.mean(r.real ** 2 + r.imag ** 2) / 2)),
        abs_max=float(np.max(np.abs(r))),
        rel_max=float(np.max(q) * 100),
        rel_max_at=float(f[int(np.argmax(q))]),
        rms_re=float(np.sqrt(np.mean(r.real ** 2))),
        rms_im=float(np.sqrt(np.mean(r.imag ** 2))),
        acf_re=float(E.acf_max(r.real)), acf_im=float(E.acf_max(r.imag)),
        acf=float(max(E.acf_max(r.real), E.acf_max(r.imag))),
        sign_chg_re=_sign_changes(r.real), sign_chg_im=_sign_changes(r.imag),
        frac_pos_re=float(np.mean(r.real > 0)), frac_pos_im=float(np.mean(r.imag > 0)),
        bands=band,
        resid_re=r.real.tolist(), resid_im=r.imag.tolist(),
        resid_rel_re=(r.real / w).tolist(), resid_rel_im=(r.imag / w).tolist(),
    )


def pair_stats(a, b, mask=None):
    """形状是否相同的三个独立读数。

    `corr`   原始残差的 Pearson（幅值差异会同时压低它）
    `r2_scale` 只允许一个标量缩放 `b ≈ k·a`，报最优 k 与 R²（把幅值差异整段剥掉，
              只留形状；R²→1 才是「同一结构不同尺度」）
    `sign_agree` 同号比例（形状无关幅值的最粗读数）
    """
    out = {}
    m = slice(None) if mask is None else mask
    for tag in ("re", "im"):
        x = np.asarray(a[f"resid_{tag}"], float)[m]
        y = np.asarray(b[f"resid_{tag}"], float)[m]
        k = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else float("nan")
        ss_res = float(np.sum((y - k * x) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out[tag] = dict(
            corr=float(np.corrcoef(x, y)[0, 1]) if x.size > 2 else float("nan"),
            k=float(k),
            r2_scale=float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            sign_agree=float(np.mean(np.sign(x) == np.sign(y))),
            rms_ratio=float(np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x ** 2))),
        )
    return out


def consensus_stats(recs, names):
    """把各模型的残差各自归一化到单位 RMS 后取平均，看谁贴着这条公共形状。

    归一化是关键：它把「幅值差」整段剥掉，只问**形状**是不是同一条。
    """
    out = {}
    for tag in ("re", "im"):
        prof = []
        for n in names:
            x = np.asarray(recs[n][f"resid_{tag}"], float)
            prof.append(x / np.sqrt(np.mean(x ** 2)))
        P = np.array(prof)
        cons = P.mean(axis=0)
        for n, p in zip(names, P):
            r2 = 1 - float(np.sum((p - cons) ** 2)) / float(np.sum((p - p.mean()) ** 2))
            out.setdefault(n, {})[tag] = dict(r2_vs_consensus=r2,
                                              corr_vs_consensus=float(np.corrcoef(p, cons)[0, 1]))
    return out


# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------

def plot(f, models, Z, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"VOIGT_K1": ("#9467bd", "o"), "NESTED_mcmc": ("#8c564b", "D"),
             "NESTED_LS": ("#1f77b4", "s"), "EXT1": ("#d62728", "^"),
             "EXT2": ("#ff7f0e", "v"), "LINKK": ("#2ca02c", "P"),
             "LIT_C1": ("#17becf", "x"), "LIT_C2": ("#bcbd22", "*"),
             "LIT_C3": ("#e377c2", "<")}

    fig, ax = plt.subplots(2, 2, figsize=(14.5, 9.0))

    for name, m in models.items():
        c, mk = style[name]
        r = m["Z"] - Z
        q = np.abs(r) / np.abs(Z)
        rwp = float(np.sqrt(np.mean(q ** 2)) * 100)
        lab = f"{name} (Rwp {rwp:.2f}%)"
        ax[0, 0].semilogx(f, r.real, mk + "-", color=c, ms=3, lw=1.1, label=lab)
        ax[0, 1].semilogx(f, r.imag, mk + "-", color=c, ms=3, lw=1.1, label=name)
        ax[1, 0].loglog(f, np.maximum(q * 100, 1e-3), mk + "-", color=c, ms=3, lw=1.1, label=name)

    ax[0, 0].axhline(0, color="k", lw=0.6)
    ax[0, 0].set_title("Re residual (Ω) vs f")
    ax[0, 0].set_ylabel("Ω")
    ax[0, 0].legend(fontsize=7.5)

    ax[0, 1].axhline(0, color="k", lw=0.6)
    ax[0, 1].set_title("Im residual (Ω) vs f")
    ax[0, 1].set_ylabel("Ω")

    ax[1, 0].axhline(2.0, color="gray", ls="--", lw=0.8, label="2% level")
    ax[1, 0].set_title("|ΔZ| / |Z| (%)")
    ax[1, 0].set_ylabel("%")

    # 环区 Nyquist 放大：看数据在 0.03–0.3 Hz 有什么局部特征被所有模型抹平
    msk = (f >= 0.008) & (f <= 2.0)
    ax[1, 1].plot(Z.real[msk], -Z.imag[msk], "ko", ms=4, label="data")
    for name, m in models.items():
        c, _ = style[name]
        ax[1, 1].plot(m["Z"].real[msk], -m["Z"].imag[msk], "-", color=c, lw=1.4, label=name)
    ax[1, 1].set_title("Nyquist zoom 0.01–2 Hz (low-freq loop)")
    ax[1, 1].set_xlabel("Z' (Ω)")
    ax[1, 1].set_ylabel("-Z'' (Ω)")
    ax[1, 1].legend(fontsize=7.5)

    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlabel("f (Hz)")
    fig.suptitle(f"AZ31-4Li_100: residual structure across {len(models)} fits "
                 "(incl. literature Table 1 C1–C3) + Lin-KK floor", fontsize=12)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"→ {path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=["AZ31-4Li_100"])
    ap.add_argument("--json", default=OUT_JSON)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-lit", action="store_true",
                    help="跳过文献 Table 1 的 C1/C2/C3（只跑原来六个）")
    ap.add_argument("--lit-iters", type=int, default=800,
                    help="文献电路的 DE 迭代上限（C2/C3 12 参数，太小会落到局部解）")
    a = ap.parse_args(argv)

    out = {}
    for spec in a.specs:
        f, Z, models = build_models(spec, with_lit=not a.no_lit, lit_iters=a.lit_iters)
        print(f"\n=== {spec}  n={f.size} ===")
        # 恒等核对：C1 与 C4(我们的 nested) 是同一个阻抗函数 → 预测谱必须重合。
        if "LIT_C1" in models and "NESTED_LS" in models:
            d = np.abs(models["LIT_C1"]["Z"] - models["NESTED_LS"]["Z"]) / np.abs(Z)
            print(f"  恒等核对 LIT_C1 vs NESTED_LS: max|ΔZ|/|Z| = {d.max():.3e}"
                  f"（中位 {np.median(d):.3e}）→ 同一条阻抗函数，差异只是优化器收敛")
        rec = dict(n=int(f.size), f=f.tolist(), models={})
        for name, m in models.items():
            met = describe(f, Z, m["Z"])
            met["note"] = m["note"]
            if m.get("n_params"):
                met["n_params"] = m["n_params"]
                met["params"] = m["params"]
            m["rwp"] = met["rwp"]
            rec["models"][name] = met
            print(f"  {name:<10} Rwp {met['rwp']:6.2f}%  rms_re {met['rms_re']:7.2f}"
                  f" rms_im {met['rms_im']:7.2f}  acf {met['acf']:.3f}"
                  f"  rel_max {met['rel_max']:5.1f}% @ {met['rel_max_at']:.3g} Hz"
                  f"  符号变化 re/im {met['sign_chg_re']}/{met['sign_chg_im']}")
            print(f"             频段 rms%: " + "  ".join(
                f"{k} {v['rms_rel']:.1f}" for k, v in met["bands"].items()))

        names = list(rec["models"])
        pw = []
        print("\n  两两残差结构（r2_scale = 只允许一个缩放后的形状重合度；sign = 同号比例）")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a1, b1 = rec["models"][names[i]], rec["models"][names[j]]
                st = pair_stats(a1, b1)
                bandw = {}
                for lo_, hi_, tag in BANDS:
                    m = (np.asarray(rec["f"]) >= lo_) & (np.asarray(rec["f"]) < hi_)
                    bandw[tag] = pair_stats(a1, b1, mask=m)
                pw.append(dict(a=names[i], b=names[j], **st, by_band=bandw))
                print(f"    {names[i]:<10} vs {names[j]:<10}"
                      f"  Re corr {st['re']['corr']:+.3f} R² {st['re']['r2_scale']:+.3f}"
                      f" 同号 {st['re']['sign_agree']*100:4.0f}%  |"
                      f"  Im corr {st['im']['corr']:+.3f} R² {st['im']['r2_scale']:+.3f}"
                      f" 同号 {st['im']['sign_agree']*100:4.0f}%")
                print("              分段同号%: " + "  ".join(
                    f"{t} Re {bandw[t]['re']['sign_agree']*100:3.0f}"
                    f"/Im {bandw[t]['im']['sign_agree']*100:3.0f}"
                    for _, _, t in BANDS))
        rec["pairwise"] = pw

        # 共识只在「同一 LS 口径、且都能形成低频感抗环」的三个模型间取。
        # NESTED_mcmc 是 MCMC 后验中位谱，口径不同，不进共识（但仍列出来对照）。
        good = [n for n in names if n in ("NESTED_LS", "EXT1", "EXT2")]
        cons = consensus_stats(rec["models"], good)
        rec["consensus"] = cons
        print(f"\n  公共形状检验（各残差归一化到单位 RMS 后与三模型共识比较，{'+'.join(good)}）")
        for n in good:
            print(f"    {n:<10} Re R²(共识) {cons[n]['re']['r2_vs_consensus']:+.3f}"
                  f"   Im R²(共识) {cons[n]['im']['r2_vs_consensus']:+.3f}")

        print("\n  残差极值位置（Ω @ Hz）")
        for n in names:
            m = rec["models"][n]
            i = int(np.argmax(np.abs(m["resid_re"])))
            j = int(np.argmax(np.abs(m["resid_im"])))
            print(f"    {n:<10} Re {m['resid_re'][i]:+8.1f} @ {rec['f'][i]:<9.4g}"
                  f"  Im {m['resid_im'][j]:+8.1f} @ {rec['f'][j]:<9.4g}")
        out[spec] = rec

        if not a.no_plot:
            plot(f, models, Z, OUT_PNG)

        # 残差逐点表（低频端最关键的 8 个点）
        print("\n  低频端逐点残差（Ω）:")
        print("    f(Hz)      |Z|     " + "".join(f"{k:>14}" for k in names))
        for k in range(8):
            row = f"    {f[k]:<9.4g} {np.abs(Z[k]):7.1f}  "
            row += "".join(f"{rec['models'][n]['resid_re'][k]:+7.1f}"
                           f"{rec['models'][n]['resid_im'][k]:+7.1f}" for n in names)
            print(row)

    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\n→ {a.json}")


if __name__ == "__main__":
    main()
