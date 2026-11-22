"""噪声尺度探针（只读）：用跨实验重复谱检验 sigma 的尺度形式。

问题
----
`free` 口径假设 sigma 与频率无关（中心 0.01*zmax）；`perpoint` 口径假设
sigma_j = 2%*|Z_j|，且 Re/Im **共用同一个 |Z|**。两者在宽动态范围谱上差 3 个量级。
本脚本不拟合电路、不碰生产代码，只用重复测量的离散度问三件事：

  Q1 离散度到底是噪声，还是样品漂移？（共模 Rs 分离前后对比）
  Q2 尺度是常数（b）、正比于 |Z|（a|Z|）、还是混合 sqrt(a^2|Z|^2+b^2)？
  Q3 尺度该用 |Z| 模长，还是各通道自己的分量（|Re Z| / |Im Z|）？

方法（四条硬约束）
------------------
1. **Im/Re 分开**：两通道各自估标准差，不合成复模长。
2. **共模分离**：跨实验重复含 Rs 漂移，整体平移会伪装成"常数噪声"。故每组做
   `raw`（原始）与 `decom`（先减各自 Rs 偏移，只动实部）两套。
3. **不确定度如实报**：n-1 自由度只有 2~3，单点 s 的相对标准误约 41%~50%，
   SSE 本身不是判据，须给 bootstrap CI；b 撞下界即标记 degenerate。
4. **尺度候选同台竞争**：M2 分别用 |Z| 与分量做自变量，AIC 直接比，不允许
   先验地假定哪个对。

模型
----
M1 const : s = b
M2 rel   : s = a*x        (x = |Z| 或 |Re Z| 或 |Im Z|)
M3 mix   : s = sqrt(a^2*x^2 + b^2)

三者在 log 空间最小化。依据：s ~ sigma*sqrt(chi2_nu/nu) ⇒ Var(log s) ~ 1/(2nu)
与 s 无关，故 log 空间等权最小二乘近似 GLS。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/noise_scale_probe.py
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import bm_real as R  # noqa: E402

SPEC_DIR = os.path.join("EISmeasurements", "csv_3e")
OUT = os.path.join("out", "noise_scale_probe")
PAT = re.compile(r"(3E-\d)_(HER|OER|full_electrolysis_cell)_exp(\d)_(\d+)mA\.csv$")

PRIMARY = ("3E-3", "full_electrolysis_cell", "0100")   # n=4
CONTROL = ("3E-2", "full_electrolysis_cell", "0100")   # n=3，Rs 漂移最小
N_HF = 5
FLOOR = 1e-9


# ---------------------------------------------------------------- 数据
def discover():
    g = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(SPEC_DIR, "*.csv"))):
        m = PAT.search(os.path.basename(p))
        if m:
            g[(m.group(1), m.group(2), m.group(4))].append((int(m.group(3)), p))
    return {k: sorted(v) for k, v in g.items()}


def load_group(items):
    fs, Zs, exps = [], [], []
    for e, p in items:
        f, Z = R.load_spectrum(p)
        fs.append(np.asarray(f, float))
        Zs.append(np.asarray(Z, complex))
        exps.append(e)
    if len({f.size for f in fs}) != 1 or not all(
            np.allclose(f, fs[0], rtol=1e-9, atol=0) for f in fs[1:]):
        raise SystemExit("频率网格不一致，本脚本不做重采样")
    order = np.argsort(exps)
    return fs[0], [Zs[i] for i in order], [exps[i] for i in order]


def decompose(Zs):
    A = np.stack(Zs)
    rs = A.real[:, -N_HF:].mean(axis=1)
    return (A.real - (rs - rs.mean())[:, None]) + 1j * A.imag, rs


def affine_align(Zs):
    """把每条谱做最优复仿射 c*Z+d 对齐到逐点中位谱。

    只减 Rs 常数偏移(`decom`)仍剩整体缩放；复仿射把两者都吸收，因此
    `affine` 后的离散度是**噪声下界**（4 个自由度会把真实噪声也吃掉一部分），
    `raw` 是上界。两者之间就是"重复谱能给出的噪声区间"。
    """
    A = np.stack(Zs)
    ref = np.median(A, axis=0)
    out, coef = [], []
    for z in A:
        zc, yc = z - z.mean(), ref - ref.mean()
        c = complex(np.vdot(zc, yc) / np.vdot(zc, zc))
        d = complex(ref.mean() - c * z.mean())
        out.append(c * z + d)
        coef.append(dict(c=float(abs(c)), d=float(abs(d))))
    return out, coef


def disp(arr):
    A = np.stack(arr)
    return A.mean(axis=0), A.real.std(axis=0, ddof=1), A.imag.std(axis=0, ddof=1)


# ---------------------------------------------------------------- 拟合
def _aic(rss, n, k):
    return float(n * np.log(max(rss, 1e-300) / n) + 2 * k)


def fit_m1(s):
    ok = np.isfinite(s) & (s > 0)
    y = np.log(s[ok])
    b = float(np.exp(y.mean()))
    return dict(kind="M1", x="const", b=b, rss=float(((y - np.log(b)) ** 2).sum()),
                n=int(ok.sum()), par=dict(b=b))


def fit_m2(x, s):
    ok = np.isfinite(s) & (s > 0) & np.isfinite(x) & (x > FLOOR)
    y, lx = np.log(s[ok]), np.log(x[ok])
    la = float((y - lx).mean())
    a = float(np.exp(la))
    return dict(kind="M2", a=a, rss=float(((y - la - lx) ** 2).sum()),
                n=int(ok.sum()), par=dict(a=a))


def fit_m3(x, s):
    """多初值 + 有界，避免 LM 沿边界退化。"""
    ok = np.isfinite(s) & (s > 0) & np.isfinite(x) & (x > FLOOR)
    xs, ys = x[ok], np.log(s[ok])
    n = int(ok.sum())
    smed = float(np.median(s[ok]))
    best = None
    for a0 in (0.005, 0.02, 0.15):
        for f0 in (1e-3, 0.5, 3.0):
            p0 = np.log([a0, max(f0 * smed, 1e-10)])

            def res(p):
                return np.log(np.hypot(np.exp(p[0]) * xs, np.exp(p[1]))) - ys

            sol = least_squares(res, p0, method="trf", max_nfev=8000,
                                bounds=(np.log([1e-7, 1e-13]), np.log([5.0, 1e4 * smed])))
            rss = float((sol.fun ** 2).sum())
            if best is None or rss < best[0]:
                a, b = (float(v) for v in np.exp(sol.x))
                best = (rss, a, b)
    rss, a, b = best
    degenerate = "b_floor" if b <= 1.1e-12 else ("a_floor" if a <= 1.1e-7 else None)
    return dict(kind="M3", a=a, b=b, rss=rss, n=n, degenerate=degenerate,
                par=dict(a=a, b=b))


def boot_a(x, s, n_boot=600, seed=0):
    """残差 bootstrap：给 M2 的 a 一个 95% CI（尺度形式选对时最有意义）。"""
    ok = np.isfinite(s) & (s > 0) & np.isfinite(x) & (x > FLOOR)
    y, lx = np.log(s[ok]), np.log(x[ok])
    la = float((y - lx).mean())
    resid = y - la - lx
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        j = rng.integers(0, y.size, y.size)
        out.append(float(np.exp((y[j] - lx[j]).mean())))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def compare(absZ, scales, s):
    """对一个通道的 s：M1 + 每个尺度上的 M2/M3 同台比 AIC。"""
    cands = [fit_m1(s)]
    for name, x in scales.items():
        m2 = fit_m2(x, s)
        m2["x"] = name
        m3 = fit_m3(x, s)
        m3["x"] = name
        cands += [m2, m3]
    for c in cands:
        c["aic"] = _aic(c["rss"], c["n"], len(c["par"]))
    best = min(cands, key=lambda c: c["aic"])
    for c in cands:
        c["daic"] = float(c["aic"] - best["aic"])
    return cands, best


def drift_test(rs, Zs):
    A = np.stack(Zs)
    e = np.arange(1, A.shape[0] + 1, dtype=float)
    rates = np.array([np.polyfit(e, A[:, j].real, 1)[0] / abs(A[:, j].mean())
                      for j in range(A.shape[1])])
    rho, pval = spearmanr(np.arange(len(rs)), rs)
    return dict(rs=[float(v) for v in rs],
                rs_cv=float(np.std(rs, ddof=1) / np.mean(rs)),
                rho=float(rho), p=float(pval),
                rate_med=float(np.median(rates)),
                rate_p10=float(np.percentile(rates, 10)),
                rate_p90=float(np.percentile(rates, 90)))


# ---------------------------------------------------------------- 主流程
def analyse(key, items, n_boot=600):
    f, Zs, exps = load_group(items)
    A2, rs = decompose(Zs)
    A3, coef = affine_align(Zs)
    rec = dict(key="/".join(key), exps=exps, n=len(Zs), n_freq=int(f.size),
               zmax=float(np.abs(np.stack(Zs)).max()),
               drift=drift_test(rs, Zs), affine_coef=coef, forms={})
    for form, arr in (("raw", Zs), ("decom", list(A2)), ("affine", list(A3))):
        Zbar, s_re, s_im = disp(arr)
        absZ = np.abs(Zbar)
        box = dict(z_span=[float(absZ.min()), float(absZ.max())],
                   rel={}, best={}, table={})
        for ch, s, comp in (("re", s_re, np.abs(Zbar.real)),
                            ("im", s_im, np.maximum(np.abs(Zbar.imag), FLOOR))):
            scales = {"|Z|": absZ, ("|ReZ|" if ch == "re" else "|ImZ|"): comp}
            cands, best = compare(absZ, scales, s)
            cands.sort(key=lambda c: c["aic"])
            box[ch] = dict(cands=cands, best=best["kind"] + "@" + best["x"],
                           s_med=float(np.median(s)),
                           rel_Z=float(np.median(s / absZ)),
                           rel_comp=float(np.median(s / comp)))
            box[best["kind"] + "@" + best["x"] + "_ci"] = (
                boot_a(scales[best["x"]], s) if best["kind"] == "M2" else None)
        rec["forms"][form] = box
    return rec


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    groups = discover()
    detail = {}
    print("=" * 78)
    print("A. 详例 n=4（3E-3 full cell 0100mA）与对照 n=3（3E-2 full cell 0100mA）")
    print("=" * 78)
    for key in (PRIMARY, CONTROL):
        rec = analyse(key, groups[key])
        detail["/".join(key)] = rec
        d = rec["drift"]
        print("\n### %s   n=%d  pts=%d  zmax=%.3g" % (rec["key"], rec["n"],
                                                    rec["n_freq"], rec["zmax"]))
        print("  Rs %s  CV %.1f%%  rho=%.2f (p=%.3f)" % (
            np.round(d["rs"], 4), 100 * d["rs_cv"], d["rho"], d["p"]))
        print("  逐点相对漂移率 中位 %+.1f%%  [%+.1f%%, %+.1f%%]" % (
            100 * d["rate_med"], 100 * d["rate_p10"], 100 * d["rate_p90"]))
        for form, box in rec["forms"].items():
            for ch in ("re", "im"):
                b = box[ch]
                print("  %-5s %s  s_med %.3g  s/|Z| %.2f%%  s/|comp| %.2f%%  -> %s" % (
                    form, ch, b["s_med"], 100 * b["rel_Z"], 100 * b["rel_comp"],
                    b["best"]))
                for c in b["cands"]:
                    tag = c["kind"] + ("@" + c["x"] if c["x"] != "const" else "")
                    par = ("b=%.3g" % c["b"]) if c["kind"] == "M1" else (
                        ("a=%.3f%%" % (100 * c["a"])) if c["kind"] == "M2"
                        else "a=%.3f%%,b=%.3g" % (100 * c["a"], c["b"]))
                    print("        %-14s dAIC %+7.1f   %s" % (tag, c["daic"], par))

    print("\n" + "=" * 78)
    print("B. 全部 n>=4 组：最优尺度形式与 a 是否跨组一致（decom）")
    print("=" * 78)
    sweep = []
    for key, items in sorted(groups.items()):
        if len(items) < 4:
            continue
        rec = analyse(key, items, n_boot=0)
        for ch in ("re", "im"):
            b = rec["forms"]["decom"][ch]
            sweep.append(dict(group="/".join(key), ch=ch, rs_cv=rec["drift"]["rs_cv"],
                              best=b["best"], rel_comp=b["rel_comp"],
                              a=(b["cands"][0]["par"].get("a", None)),
                              b=(b["cands"][0]["par"].get("b", None))))
        print("  %-38s RsCV %5.1f%%  re: %-14s a=%6.2f%% | im: %-14s a=%6.2f%%" % (
            "/".join(key), 100 * rec["drift"]["rs_cv"],
            rec["forms"]["decom"]["re"]["best"],
            100 * rec["forms"]["decom"]["re"]["rel_comp"],
            rec["forms"]["decom"]["im"]["best"],
            100 * rec["forms"]["decom"]["im"]["rel_comp"]))
    for ch in ("re", "im"):
        v = [r["rel_comp"] for r in sweep if r["ch"] == ch]
        print("  -> %s 的分量相对离散度跨 %d 组：中位 %.2f%%  IQR %.2f~%.2f%%" % (
            ch, len(v), 100 * np.median(v), 100 * np.percentile(v, 25),
            100 * np.percentile(v, 75)))

    with open(os.path.join(OUT, "probe.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(detail=detail, sweep=sweep), fh, ensure_ascii=False,
                  indent=1, allow_nan=False)
    print("\n-> %s" % os.path.join(OUT, "probe.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
