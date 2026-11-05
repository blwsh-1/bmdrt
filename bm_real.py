# -*- coding: utf-8 -*-
"""真实 EIS 数据主程序：DRT 只做预判断，K 由证据选，不由 DRT 定。

    python bm_real.py data.csv                # 默认：预判断 → K 扫描 → 选 K → 出图
    python bm_real.py data.csv --K 2          # 只跑 K=2（跳过选择）
    python bm_real.py data.csv --krange 1,4   # 指定扫描的 K 范围
    python bm_real.py data.csv --dry          # 只做 DRT 预判断（秒级）
    python bm_real.py data.csv --L            # 显式开电感（默认自动判定）
    python bm_real.py data.csv --zeta free    # 放开 ζ（默认钉死 1.0）

分工（这是本文件唯一要守住的东西）
----------------------------------
  ① 预处理 + 传统 DRT（drt.py 的 Tikhonov 反演）
     只用原始 Z 数据，不涉及 HN 参数。产出三样**仅供参考**的东西：
       · 有几个峰、峰位在哪（→ 给 K 一个**候选范围**，不是定值）
       · 峰面积/峰位（→ 给 NUTS 一个**初值**，先验范围不动）
       · 边界堆积、窗口外过程、峰重叠程度（→ 决定要不要报警）
     它**不**决定 K，**不**当额外观测，**不**进似然。

  ② BM-DRT 贝叶斯拟合
     似然只用原始复阻抗 (Z', Z'')，一次都不碰 DRT 的输出。
     K 由 ③ 的模型比较选出；DRT 只影响初值落点。

  ③ 模型比较与诊断
     对 K ∈ {K_drt−1, K_drt, K_drt+1} 各拟合一次，用 PSIS-LOO（elpd）
     比较，同时报 rms/σ、残差 acf、碎片/窗口外分量数、R̂/ESS/divergences。
     选择规则写在 `select_k()` 里，是显式的、可复核的。

为什么不能"DRT 定 K 再让贝叶斯验证"
------------------------------------
`examples/example_caseF.csv`（真值 2 个 ZARC + 电感）就是这个坑的实证：
DRT 在 λ=17 下把两个相距 3 倍的峰**合并成一个**，K_drt=1；
若照 K_drt 拟合，得到 rms/σ=0.95、PPC 覆盖 0.90——**所有常规指标都"合格"**，
只有残差 acf=0.70 暴露了它漏了一个过程。
也就是说：DRT 定 K + 常规拟合指标，会把"欠拟合"判成"通过"。
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np

import bm_core as B
import bm_experiments as E
import drt as DRT
from bm_experiments import acf_max
from drt import prescreen

try:                                    # 扩散诊断是可选的；缺了只是少一段报告
    import bm_diff as DIFF
except Exception:                       # pragma: no cover
    DIFF = None

try:                                    # K-K 事前门禁；缺了只是少一道体检，不阻断拟合
    from bm_kk import kk_check as _kk_check
    from bm_kk import format_report as _kk_format_report
except Exception:                       # pragma: no cover
    _kk_check = None
    _kk_format_report = None

try:                                    # 全参数相关矩阵（"别只看误差"那一层）
    import bm_corr as CORR
except Exception:                       # pragma: no cover
    CORR = None

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

KW = dict(n_warmup=3000, n_samples=1000, num_chains=3, seed=0,
          progress=False, target_accept=0.9, max_tree_depth=13,
          extra_fields=("num_steps",))
K_MAX = 5
OUT = Path("figures")

LOG2PI = float(np.log(2 * np.pi))
NULL_CACHE = None    # 同一份 f 的零分布阈值只算一次


# --------------------------------------------------------------------------
# 读数据
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    """列名归一：去掉单位符号与空白，保留字母数字和 | ' 。"""
    s = str(s).strip().lower()
    s = s.replace("Ω", "").replace("ω", "").replace("ohm", "")
    return re.sub(r"[^a-z0-9|']", "", s)


def _pick(names: list[str], cands: tuple[str, ...], contains: tuple[str, ...] = ()):
    """按优先级找列号；先精确匹配，再子串匹配。"""
    for c in cands:
        if c in names:
            return names.index(c)
    for c in contains:
        for i, nm in enumerate(names):
            if c in nm:
                return i
    return None


def load_spectrum(path, fscale: float | None = None, zscale: float = 1.0):
    """读 CSV/TSV/空白分隔的 EIS 数据，返回 (f, Z)。"""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"找不到文件：{path}")
    raw = p.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    rows = [ln for ln in raw
            if ln.strip() and not ln.lstrip().startswith(("#", "%", "!"))]

    best, best_n = None, 0
    for d in (",", "\t", ";", None):
        cnt = [len(ln.split(d)) for ln in rows[:20]]
        if min(cnt) >= 2 and max(cnt) == min(cnt) and cnt[0] > best_n:
            best, best_n = d, cnt[0]
    if best is None:
        raise SystemExit("无法识别列分隔符（试过 , \\t ; 空白）")

    def split(ln):
        return [x.strip() for x in ln.split(best)]

    head = split(rows[0])
    has_head = any(not re.fullmatch(r"[-+0-9.eE]+", h or "x") for h in head)
    names = [_norm(h) for h in head] if has_head else []
    body = rows[1:] if has_head else rows

    data = []
    for ln in body:
        try:
            data.append([float(x) for x in split(ln)])
        except ValueError:
            continue
    A = np.array(data, float)
    if A.ndim != 2 or A.shape[0] < 5:
        raise SystemExit(f"有效数据行太少（{A.shape[0] if A.ndim == 2 else 0} 行）")

    if names:
        i_f = _pick(names, ("f", "freq", "frequency", "hz", "fhz"), ("freq",))
        i_re = _pick(names, ("zre", "zreal", "re", "real", "z'", "zr", "z1"),
                     ("real", "zre"))
        i_im = _pick(names, ("zim", "zimag", "im", "imag", "z''", "zi", "z2"),
                     ("imag", "zim"))
        i_mag = _pick(names, ("|z|", "zmag", "zabs", "absz", "mag", "amplitude"),
                      ("|z|", "mag"))
        i_ph = _pick(names, ("phase", "theta", "angle", "ang", "phz"), ("phase",))
    else:
        i_f, i_re, i_im, i_mag, i_ph = 0, 1, 2, None, None
        if A.shape[1] < 3:
            raise SystemExit("无表头时需要 3 列：f, Zre, Zim")

    if i_f is None:
        raise SystemExit(f"找不到频率列，表头是：{head}")
    f = A[:, i_f].astype(float)

    if i_re is not None and i_im is not None:
        Z = A[:, i_re] + 1j * A[:, i_im]
        form = "Zre/Zim"
    elif i_mag is not None and i_ph is not None:
        mag = A[:, i_mag].astype(float)
        ph = A[:, i_ph].astype(float)
        rad = any("rad" in h for h in names)
        if not rad and np.nanmax(np.abs(ph)) > 7.0:
            ph = np.radians(ph)
        Z = mag * np.exp(1j * ph)
        form = "|Z|/phase" + ("(rad)" if rad else "(deg)")
    else:
        raise SystemExit(f"找不到 (Zre,Zim) 或 (|Z|,phase) 组合，表头是：{head}")

    if fscale is None:
        h = "".join(names)
        fscale = 1e3 if "khz" in h else (1e6 if "mhz" in h else 1.0)
    f = f * fscale
    Z = Z * zscale

    ok = np.isfinite(f) & np.isfinite(Z) & (f > 0)
    f, Z = f[ok], Z[ok]
    o = np.argsort(f)
    f, Z = f[o], Z[o]
    fu, inv, cnt = np.unique(f, return_inverse=True, return_counts=True)
    if np.any(cnt > 1):
        Zr = np.zeros(fu.size, complex)
        for i in range(fu.size):
            Zr[i] = Z[inv == i].mean()
        print(f"  注意：{int(np.sum(cnt > 1))} 个重复频率点已取平均")
        f, Z = fu, Zr

    print(f"  读入 {f.size} 点  {f.min():.3g} – {f.max():.3g} Hz  "
          f"({np.log10(f.max() / f.min()):.2f} decade)  格式 {form}"
          f"  f×{fscale:g}  Z×{zscale:g}")
    return f, Z


def detect_L(f, Z) -> bool:
    """高频端 Im Z > 0 说明有感抗尾巴（−Im Z < 0）。取最高 3 点投票。"""
    hi = np.argsort(f)[-3:]
    return bool(np.mean(Z.imag[hi] > 0) >= 0.5)


# --------------------------------------------------------------------------
# 逐样本正演、残差诊断、逐点对数似然（供 LOO 用）
# --------------------------------------------------------------------------

def _params(s, i, include_L):
    """转调 bm_core.sample_params。

    以前这里是手写的一份，加扩散项后必须跟着改，而调用点有三处（z_band、
    残差统计、PPC）。漏一处不会报错，只会让图上的曲线和报告的 rms 悄悄对不上，
    所以统一收到 core 里去。
    """
    return B.sample_params(s, i, include_L)


def _diff_line(r):
    """扩散项摘要 + **按方向区分**的可辨识性提示。

    为什么必须分方向：Wo 的两个渐近区定住的是**完全不同**的组合。

      f_D ≫ f_max（窗口全在低频侧）→ 低频电容区
          Z → R_W/3 + 1/(jωC_D)，C_D = τ_D/R_W
          只有 C_D 可辨识；R_W/3 是常数实部，被 R∞ 吸收；R_W 与 τ_D 单独都定不住。

      f_D ≪ f_min（窗口全在高频侧）→ Warburg 尾
          Z → A_W/√(jω)，A_W = R_W/√τ_D
          只有 A_W 可辨识；R_W 与 τ_D 沿 lnR_W − ½lnτ_D = lnA_W 这条线完全简并。

      f_min ≲ f_D ≲ f_max → 转折在窗口内，数据**有机会**约束转折位置
          （窗口外时 τ_D 跨几个量级都不变代价）。但"后验区间有限"这件事
          本身说明不了太多：它也可能是先验支撑（τ_D 的 LogNormal σ=2.0）
          与其他参数的联合约束一起压出来的，不等于数据单独定住了 τ_D。
          实测 corr(lnR_W,lnτ_D)=+0.999，定得最死的是 C_D=τ_D/R_W（±0.3%），
          而 R_W 与 τ_D 沿 R_W∝τ_D 方向仍一起滑动 ±9%。所以**只引用 C_D**，
          引用 R_W 时必须同时说明它与 τ_D 的相关性。

    以前这里把两种情况合并成一句"f_D 在窗口外 → 退化为 R_W/3+1/(jωC_D)"，
    在 f_D ≪ f_min 时是**反的**：那时窗口里根本没有电容区，看到的是 Warburg 尾。
    注意**不能靠 corr(lnR_W,lnτ_D) 区分这两种情况**——两边都是 +1
    （一边固定 τ_D−lnR_W，一边固定 lnR_W−½lnτ_D），方向只能由 f_D 与窗口的
    相对位置定。
    """
    s = r["s"]
    out = []
    f_lo = float(r.get("f_lo") or 0.0)
    f_hi = float(r.get("f_hi") or 1e300)

    if "Aw" in s:
        a = np.asarray(s["Aw"])
        out.append(f"A_W = {np.median(a):.4g} Ω·s^-0.5 "
                   f"[{np.percentile(a, 5):.4g}, {np.percentile(a, 95):.4g}]")

    if "Rwo" in s and "tauD" in s:
        rw = np.asarray(s["Rwo"])
        td = np.asarray(s["tauD"])
        mrw, mtd = float(np.median(rw)), float(np.median(td))
        fD = 1.0 / (2 * np.pi * mtd)
        a_impl = mrw / np.sqrt(mtd)          # A_W = R_W/√τ_D
        c_d = mtd / mrw                      # C_D = τ_D/R_W
        out.append(f"R_W = {mrw:.4g} Ω "
                   f"[{np.percentile(rw, 5):.4g}, {np.percentile(rw, 95):.4g}]")
        out.append(f"τ_D = {mtd:.4g} s → f_D = {fD:.3g} Hz")
        if fD > f_hi * 10:
            out.append(f"** f_D ≫ f_max：窗口全在低频侧，Wo 退化为 "
                       f"R_W/3 + 1/(jωC_D)，只有 C_D=τ_D/R_W={c_d:.3g} F 可辨识；"
                       f"R_W/3 被 R∞ 吸收，R_W 与 τ_D 单独都不可信 **")
        elif fD < f_lo / 10:
            out.append(f"** f_D ≪ f_min：窗口全在高频侧，Wo 退化为 Warburg 尾，"
                       f"只有 A_W=R_W/√τ_D={a_impl:.3g} 可辨识；"
                       f"R_W 与 τ_D 沿 lnR_W−½lnτ_D=lnA_W 完全简并 **")
        elif f_lo <= fD <= f_hi:
            # 措辞要留余地：有限后验区间也可能是"先验 + 其他元件"一起压出来的，
            # 不等于"两个参数各自被数据识别了"。B2 实测就是 C_D 定得很死、
            # R_W 与 τ_D 仍强相关（见下一行）。
            out.append(f"f_D 落在窗口内 → 数据**有机会**约束转折位置"
                       f"（窗口外时 τ_D 跨几个量级都不变代价）；"
                       f"但能否**分别**识别 R_W 与 τ_D，还取决于转折区的采样密度、"
                       f"噪声水平及与其他元件的相关性。等效 A_W={a_impl:.3g}、"
                       f"C_D={c_d:.3g} F")
        else:
            side = "上" if fD > f_hi else "下"
            out.append(f"f_D 在窗口{side}沿 1 dec 内，转折刚出窗口 → "
                       f"可辨识性介于两渐近区之间（等效 A_W={a_impl:.3g}、"
                       f"C_D={c_d:.3g} F）")
        # 简并结构：**只看 corr 不够**。corr 接近 ±1 只说明有脊线，还得说清
        # 脊线有多宽、以及"被定住的是哪个组合"。实测（B2 数据，f_D 在窗口内）
        # corr=+0.999、脊线 ±9%，但 C_D 被定到 ±0.3%——所以"R_W 与 τ_D 可分"
        # 这个说法是错的，正确的说法是"只有 C_D 定得很死，两者沿
        # R_W∝τ_D 方向仍一起滑动"。
        lr, lt = np.log(rw), np.log(td)
        if np.std(lr) > 1e-12 and np.std(lt) > 1e-12:
            combos = {"C_D=τ_D/R_W": lt - lr, "A_W=R_W/√τ_D": lr - 0.5 * lt,
                      "R_W·τ_D": lr + lt}
            nm, v = min(combos.items(), key=lambda kv: float(np.std(kv[1])))
            c = float(np.corrcoef(lr, lt)[0, 1])
            w2 = np.linalg.eigvalsh(np.cov(np.vstack([lr, lt])))
            ridge = (np.exp(np.sqrt(max(w2[-1], 0.0))) - 1) * 100
            out.append(f"corr(lnR_W,lnτ_D)={c:+.3f}；定得最死的是 {nm}"
                       f"（±{(np.exp(float(np.std(v))) - 1) * 100:.1f}%）")
            if abs(c) > 0.85:
                out.append(f"垂直于该组合仍有 ±{ridge:.1f}% 的脊线"
                           f"（R_W 与 τ_D 同向滑动，单独引用任一参数要带这句）")

    return "；".join(out) if out else "（无参数）"
def z_band(s, f, include_L: bool, n: int = 300):
    """逐样本正演 Z(f)，返回实部/虚部 (5, 50, 95) 分位。

    逐样本算再取分位，**不能**先平均参数再算曲线（非线性）。
    """
    m = len(s["R_inf"])
    idx = np.linspace(0, m - 1, min(n, m)).astype(int)
    Zs = np.empty((idx.size, f.size), complex)
    for a, i in enumerate(idx):
        Zs[a] = B.model_z(f, _params(s, i, include_L), include_L=include_L)
    q = (5, 50, 95)
    return np.percentile(Zs.real, q, 0), np.percentile(Zs.imag, q, 0)


def resid_bins(res, f, nb: int = 3):
    """按 log f 把残差切段，看均值/rms 是否随频率漂移。

    为什么需要它：残差 acf 的滞后阶数在**对数分布**的频率点上不对应固定
    的频率间隔（0.1→0.12 Hz 和 1000→1200 Hz 的相对间隔一样、绝对差三个
    数量级），所以 acf 只能当辅助指标。分段统计看的是"哪一段系统性地偏"，
    与点的疏密无关。
    """
    lf = np.log10(f)
    edges = np.quantile(lf, np.linspace(0, 1, nb + 1))
    out = []
    for i in range(nb):
        hi_ok = lf <= edges[i + 1] if i == nb - 1 else lf < edges[i + 1]
        m = (lf >= edges[i]) & hi_ok
        if m.sum() < 2:
            continue
        out.append(dict(
            n=int(m.sum()), f_lo=float(10 ** edges[i]), f_hi=float(10 ** edges[i + 1]),
            mean_re=float(res.real[m].mean()), mean_im=float(res.imag[m].mean()),
            rms_re=float(np.sqrt(np.mean(res.real[m] ** 2))),
            rms_im=float(np.sqrt(np.mean(res.imag[m] ** 2)))))
    return out


def resid_trend(res, f, deg: int = 2):
    """残差随 log f 的**平滑趋势**幅度（多项式拟合后取最大绝对偏差）。

    与 rms 的区别：rms 把"随机噪声"和"系统性弯曲"加在一起，
    趋势项只留后者。趋势 ≫ σ 说明模型形状不对（缺项或 K 不对），
    趋势 ≪ σ 才说明残差基本是白噪声。
    """
    x = np.log10(f)
    x = (x - x.mean()) / (x.std() or 1.0)
    out = {}
    for nm, y in (("re", res.real), ("im", res.imag)):
        c = np.polyfit(x, y, min(deg, max(1, y.size - 1)))
        out[nm] = float(np.max(np.abs(np.polyval(c, x) - y.mean())))
    return out


def loglik_matrix(s, f, Z, include_L, n_max: int = 800):
    """逐样本、**逐频率点**的对数似然，(n_draw, n_freq)。

    每个频率点 j 是一个**联合观测**：同一频率的实部与虚部来自同一次复阻抗
    测量，必须一起留出、一起计入——

        ll[a, j] = log p(Z'_j | θ_a) + log p(Z''_j | θ_a)

    不能摊成 2n 个"独立"观测。那样 LOO 每次只留出实部，同一频率的虚部还
    留在训练集里，等于用半个点预测另半个点；拟合越好、实虚部越相关，泄漏
    越多，elpd 被系统性高估——而且不同 K 的高估程度不同，模型比较直接失效。

    似然只用原始 Z，DRT 的输出一个字节都没进来。
    """
    m = len(s["R_inf"])
    idx = np.linspace(0, m - 1, min(n_max, m)).astype(int)
    ll = np.empty((idx.size, f.size))
    for a, i in enumerate(idx):
        z = B.model_z(f, _params(s, i, include_L), include_L=include_L)
        sr = float(s["s_re"][i]) if "s_re" in s else 1e-3
        si = float(s["s_im"][i]) if "s_im" in s else 1e-3
        lr = -0.5 * ((Z.real - z.real) / sr) ** 2 - np.log(sr) - 0.5 * LOG2PI
        li = -0.5 * ((Z.imag - z.imag) / si) ** 2 - np.log(si) - 0.5 * LOG2PI
        ll[a] = lr + li                      # 同一频率的实部+虚部一起算
    return ll

def loo_pointwise(ll):
    """PSIS-LOO（arviz）。返回 (idata, elpd, se, Pareto-k>0.7 占比, 逐点 elpd)。

    逐点 elpd 是**配对比较**的基础：两个模型在同一批频率点上求差，
    差值的不确定度才算得对。

    arviz 0.23 的 `loo` 要求同时存在 posterior 组（用来数 draw），
    塞一个占位变量即可；真正参与计算的是 log_likelihood。
    """
    import arviz as az
    idata = az.from_dict(
        posterior={"dummy": np.zeros((1, ll.shape[0], 1))},
        log_likelihood={"obs": ll[None, :, :]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = az.loo(idata, pointwise=True)
    pt = np.asarray(r.loo_i.values).ravel()
    pk = np.asarray(r.pareto_k.values).ravel()
    return idata, float(r.elpd_loo), float(r.se), float(np.mean(pk > 0.7)), pt


def compare_models(res_list):
    """配对比较：elpd_diff 与 dse 都基于**逐点差**（arviz.compare）。

    为什么不能只比 elpd 的绝对值：两个模型的 elpd 各有 ±5 的不确定度，
    差 3 分根本分不出高下。arviz 的 `dse` 是逐点差的标准误
    `sqrt(n·Var(逐点差))`，它通常远小于两个模型各自 se 的平方和开方
    ——因为两个模型在同一批点上犯的错高度相关。

    返回 DataFrame（按 elpd 降序），并把 elpd_diff/dse/weight 写回每个结果。
    """
    import arviz as az
    pack = {f"K={r['K']}": r["idata"] for r in res_list}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = az.compare(pack, ic="loo", method="stacking", scale="log")
    for r in res_list:
        key = f"K={r['K']}"
        r["elpd_diff"] = float(df.loc[key, "elpd_diff"])
        r["dse"] = float(df.loc[key, "dse"])
        r["weight"] = float(df.loc[key, "weight"])
        r["rank"] = int(df.loc[key, "rank"])
    return df

# ---- 初值：把 DRT 的峰列表调成恰好 K 个 ----
def null_thresholds(f, n_rep: int = 2000, deg: int = 2, seed: int = 0):
    """用蒙特卡洛给出 acf 与趋势统计量的**零分布 95% 分位**。

    为什么不能直接用 1.96/√n：那是单个滞后阶数上的近似区间，而
      · 我们取的是滞后 1–5 的**最大值**（多重比较，阈值必须抬高）；
      · 趋势统计量是二次多项式拟合的**最大偏差**，纯噪声下它本来就能
        达到 1.2σ 左右（边缘杠杆高），用 1.0σ 当线会天天误报；
      · 频率是对数分布、点距不均，解析式也不成立。
    直接对同一组 f 抽白噪声、算同样的统计量，取 95% 分位，最省事也最诚实。
    """
    rng = np.random.default_rng(seed)
    x = np.log10(f)
    x = (x - x.mean()) / (x.std() or 1.0)
    n = f.size
    acfs, trs = np.empty(n_rep), np.empty(n_rep)
    for i in range(n_rep):
        y = rng.standard_normal(n)
        c = np.polyfit(x, y, min(deg, max(1, n - 1)))
        trs[i] = np.max(np.abs(np.polyval(c, x) - y.mean()))
        yy = y - y.mean()
        d = float(np.dot(yy, yy))
        acfs[i] = max(abs(float(np.dot(yy[:-L], yy[L:])) / d)
                      for L in (1, 2, 3, 4, 5) if L < n)
    # 返回**样本**而不是几个分位点：调用方可以直接算"观测值落在第几百分位"，
    # 比二值判据诚实——1.15σ 在 n=31 时本来就是 93 百分位，不该报警。
    return dict(acf=acfs, trend=trs, n_rep=n_rep,
                acf95=float(np.percentile(acfs, 95)),
                trend95=float(np.percentile(trs, 95)),
                acf99=float(np.percentile(acfs, 99)),
                trend99=float(np.percentile(trs, 99)))


def adjust_peaks(pk, K, dsplit: float = 0.5):
    """把 DRT 峰列表调成恰好 K 个，作为**初值候选**。

    峰数不一致是常态（DRT 的正则化强度、峰重叠、窗口边界都会影响峰数），
    所以不能因为"峰数对不上"就放弃播种：
      · K 比峰数多 → 把面积最大的峰在 ln τ 上对称**劈开**（默认各偏 0.25
        decade），面积对半分。这正好对应"DRT 把两个近邻峰合并了"的情形。
      · K 比峰数少 → 把 ln τ 上**最近的一对**合并（面积相加、位置按面积加权）。

    这只是给 NUTS 一个更好的起点，**不改变先验范围**——后验仍可自由跑到
    别处。劈开/合并哪一侧，由 LOO 在多个 K 之间比较，不是这里定的。
    """
    comps = [dict(f=float(p["f_peak_Hz"]), R=float(p["R_ohm"])) for p in pk]
    if not comps:
        return None
    comps.sort(key=lambda c: -c["R"])
    while len(comps) < K:
        c = comps[0]                       # 面积最大者
        hi = dict(f=c["f"] * 10 ** (dsplit / 2), R=c["R"] / 2)
        lo = dict(f=c["f"] * 10 ** (-dsplit / 2), R=c["R"] / 2)
        comps[0:1] = [hi, lo]
        comps.sort(key=lambda c: -c["R"])
    while len(comps) > K:
        by_tau = sorted(comps, key=lambda c: -np.log(c["f"]))
        lt = np.array([np.log(c["f"]) for c in by_tau])
        i = int(np.argmin(np.diff(lt)))
        a, b = by_tau[i], by_tau[i + 1]
        R = a["R"] + b["R"]
        f = (a["f"] * a["R"] + b["f"] * b["R"]) / R if R > 0 else a["f"]
        merged = dict(f=f, R=R)
        comps = [c for c in comps if c is not a and c is not b] + [merged]
        comps.sort(key=lambda c: -c["R"])
    return comps


def init_values(pk, res, K, zeta_fixed=1.0):
    """DRT 峰 → 模型初值（约束空间）。K 与峰数不一致时先 adjust_peaks。"""
    comps = adjust_peaks(pk, K)
    if not comps:
        return None
    R = np.clip(np.array([c["R"] for c in comps], float), 1e-2, None)
    f0 = np.array([c["f"] for c in comps], float)
    W = float(R.sum())
    pi = R / W
    d = {"R_inf": float(res["R_inf"]), "W": W,
         "ln_tau": np.log(1.0 / (2 * np.pi * f0)),
         "phi": np.full(K, 0.85)}
    if zeta_fixed is None:
        d["zeta"] = np.ones(K)
    if K > 1:
        v, rem = [], 1.0
        for k in range(K - 1):
            vk = float(np.clip(pi[k] / rem, 1e-3, 1 - 1e-3))
            v.append(vk)
            rem *= (1 - vk)
        d["v"] = np.array(v)
    return d


# --------------------------------------------------------------------------
# 单个 K 的拟合
# --------------------------------------------------------------------------

def fit_one(f, Z, K, include_L, zeta, init=None, seed=0, kw=None, diffusion=None,
            diag=None):
    """跑一次 NUTS，返回全部指标（不打印、不画图）。

    diag：事后诊断阈值（corr_strong / corr_medium / diff_resolution_tol），
    由 config.json 的 diag 段传入；缺省时用模块内置默认值。
    """
    diag = diag or {}
    kw = dict(KW) if kw is None else dict(kw)
    kw["seed"] = seed
    t0 = time.time()
    s, mc = B.fit(f, Z, K=K, alpha=0.5, sigma=None, include_L=include_L,
                  zeta_fixed=(None if zeta == "free" else float(zeta)),
                  init_values=init, diffusion=diffusion, **kw)
    dt = time.time() - t0
    ch = mc.get_samples(group_by_chain=True)

    ndiv = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    tds = E.tree_depth_stats(mc, kw["max_tree_depth"])
    rows = E.full_param_diag(ch)
    zd = E.z_chain_diag(ch, f)

    # ---- 分量（跨样本匹配 + 按峰位排序）----
    mm = B.match_components(s)
    o = np.argsort(-np.median(mm["f0"], axis=0))
    f0 = np.median(mm["f0"], axis=0)[o]
    w = np.median(mm["w"], axis=0)[o]
    ph = np.median(mm["phi"], axis=0)[o]
    f0lo = np.percentile(mm["f0"], 5, axis=0)[o]
    f0hi = np.percentile(mm["f0"], 95, axis=0)[o]
    wlo = np.percentile(mm["w"], 5, axis=0)[o]
    whi = np.percentile(mm["w"], 95, axis=0)[o]
    # ---- 两个**不同**的"窗口外概率"，必须分开存 ----
    # f0 = 1/(2πτ) 是**参数特征频率**，不是 DRT 实际峰位（ζ≠1 时差 2%~25%）。
    # 以前只存一个 p_out、还叫"P(窗口外)"，等于把两个口径混成一个字段。
    #   p_tau_out  : P(f_τ ∉ [f_lo, f_hi])      —— 参数口径
    #   p_peak_out : P(f_peak ∉ [f_lo, f_hi])   —— 物理峰位口径（模型选择用这个）
    f0_s = mm["f0"][:, o]
    p_tau_out = np.mean((f0_s < f.min()) | (f0_s > f.max()), axis=0)

    # ---- 扩散项幅值是否可忽略：P(幅值 < ε·|Z|max | Z) ----
    # **不能用"90% 区间跨过 0"判定**：采样的是 Rw（LogNormal 先验），支撑严格
    # 大于零，区间永远不可能跨过 0；"上界只有 1e-3·zmax 量级"更是把一个后验
    # 概率说成了确定结论。正确口径是直接算概率：
    #     P(R_w < ε·|Z|max | Z),   ε = 1e-3
    # 它回答的是"这个扩散项的幅值相对整条谱是否可忽略"，**不是**"没有扩散"
    # 的概率——数据可能真有扩散，只是幅值小到这条谱分辨不出来。
    zmax_obs = float(np.max(np.abs(Z)))
    p_diff_small, diff_amp_key = float("nan"), None
    for _k in ("Rw", "Rwo"):
        if _k in s:
            diff_amp_key = _k
            p_diff_small = float(np.mean(np.asarray(s[_k], float) < 1e-3 * zmax_obs))
            break

    # ---- HN ↔ Warburg 退化指纹（逐分量）----
    # 为什么单列一项：单个 HN 能在窗口内把半无限 Warburg 逼近到 1e-6，
    # 代价是 φζ=1/2、R=A_W√τ、峰位跑到窗口外 8 个量级。这种分量在
    # "峰位/面积"表里看着完全正常（就是一个低频大峰），只有把 φζ、
    # 峰位偏移、lnR-lnτ 简并度放在一起看才能认出来。
    sig = []
    if DIFF is not None:
        mw, mt = mm["w"][:, o], mm["tau"][:, o]
        mp, mz = mm["phi"][:, o], mm["zeta"][:, o]
        for k in range(K):
            sk = DIFF.hn_warburg_signature(mw[:, k], mt[:, k], mp[:, k], mz[:, k],
                                           float(f.min()), float(f.max()))
            sk["level"], sk["why"] = DIFF.looks_like_warburg(sk)
            sig.append(sk)

    # ---- 模型选择用哪个口径：**物理峰位** ----
    # 理由：硬筛选的目的是剔除"峰跑到窗口外、面积不可信"的分量，而"峰"在
    # 物理上指 DRT 峰，不是参数 τ。φζ>1 时峰位没有物理意义（解析 γ 有负值），
    # 那些分量退回参数口径，并在报告里标 nonphysical。
    p_peak_out = np.full(K, np.nan)
    for k, sk in enumerate(sig):
        p_peak_out[k] = sk.get("p_peak_out", np.nan)

    # **展示字段**与**模型选择字段**必须分开：
    #   p_out          —— 兼容旧接口/给人看的统一出口，峰位不可用时退回 τ
    #   p_peak_out_max —— 只给 select_k 用，峰位不可用时是 NaN，**绝不回退**
    # 早先 select_k 直接读 p_out，等于在峰位诊断不可用时悄悄换成 τ 口径
    # 完成硬筛选，与文档写的"降级不是等价"矛盾。
    p_out = np.where(np.isfinite(p_peak_out), p_peak_out, p_tau_out)
    p_out_caliber = np.where(np.isfinite(p_peak_out), "peak", "tau")
    p_peak_out_max = (float(np.nanmax(p_peak_out))
                      if np.isfinite(p_peak_out).any() else float("nan"))
    p_tau_out_max = float(np.nanmax(p_tau_out)) if K else float("nan")

    # ---- 拟合优度 ----
    band = z_band(s, f, include_L)
    mr, mi = band[0][1], band[1][1]
    res = Z - (mr + 1j * mi)
    rms_re = float(np.sqrt(np.mean(res.real ** 2)))
    rms_im = float(np.sqrt(np.mean(res.imag ** 2)))
    acf = max(acf_max(res.real), acf_max(res.imag))
    pp = B.posterior_predictive_check(s, f, Z, include_L=include_L,
                                      n_rep=300, level=0.90)
    s_re = float(np.median(s["s_re"])) if "s_re" in s else np.nan
    s_im = float(np.median(s["s_im"])) if "s_im" in s else np.nan
    bins = resid_bins(res, f)
    trend = resid_trend(res, f)
    null = NULL_CACHE if NULL_CACHE is not None else null_thresholds(f)
    globals()["NULL_CACHE"] = null

    # ---- 结构证据 ----
    tot = float(np.sum(w)) or 1.0
    n_frag = int(np.sum(w < 0.02 * tot))
    frac_neg = float(np.mean((s["phi"] * s["zeta"]) > 1.0))

    # ---- 全参数相关矩阵（简并只在联合分布里可见，边际区间看不出来）----
    # 用**已对齐**的 mm 取列：混合模型的编号在样本间会跳，不先对齐
    # 会把两个不同的峰配成一对，得到假相关。
    pc = (CORR.param_corr(s, include_L=include_L, mm=mm,
                          strong=float(diag.get("corr_strong", CORR.STRONG)),
                          medium=float(diag.get("corr_medium", CORR.MEDIUM)))
          if CORR is not None else None)

    # ---- 扩散项可分辨性：Ws 与 Wo 在 ωτ_D≫1 时同形，先判能不能分 ----
    dres = (DIFF.diffusion_resolution(
                f, s, diffusion,
                tol=float(diag.get("diff_resolution_tol", 0.02)))
            if DIFF is not None and hasattr(DIFF, "diffusion_resolution") else None)

    # ---- 模型比较 ----
    ll = loglik_matrix(s, f, Z, include_L)
    try:
        idata, elpd, elpd_se, bad_k, _ = loo_pointwise(ll)
    except Exception as e:
        idata, elpd, elpd_se, bad_k = None, float("nan"), float("nan"), float("nan")
        print(f"    （LOO 失败：{type(e).__name__}: {e}）")

    r = dict(
        K=K, s=s, mc=mc, rows=rows, zd=zd, ndiv=ndiv, tds=tds, t=dt,
        n_draw=kw["num_chains"] * kw["n_samples"],
        band=band, f0=f0, w=w, phi=ph, f0lo=f0lo, f0hi=f0hi, wlo=wlo, whi=whi,
        p_out=p_out, p_out_max=float(np.nanmax(p_out)),
        p_peak_out_max=p_peak_out_max, p_tau_out_max=p_tau_out_max,
        p_tau_out=p_tau_out, p_peak_out=p_peak_out, p_out_caliber=p_out_caliber,
        rms_re=rms_re, rms_im=rms_im, s_re=s_re, s_im=s_im, acf=acf, pp=pp,
        bins=bins, trend=trend, null=null,
        R_inf=float(np.median(s["R_inf"])), W=float(np.median(s["W"])),
        L=(float(np.median(s["L"])) if include_L else None),
        n_frag=n_frag, frac_neg=frac_neg,
        degeneracy=sig,
        n_warburg_like=int(sum(1 for s in sig if s["level"] != "no")),
        diffusion=diffusion, param_corr=pc, diffusion_resolution=dres,
        f_lo=float(f.min()), f_hi=float(f.max()), n_pts=int(f.size),
        zmax_obs=zmax_obs, p_diff_small=p_diff_small, diff_amp_key=diff_amp_key,
        idata=idata, elpd=elpd, elpd_se=elpd_se, bad_k=bad_k,
        elpd_diff=0.0, dse=float("nan"), weight=float("nan"), rank=1,
        rhat_max=max(x["rhat"] for x in rows),
        tail_min=min(x["tail"] for x in rows),
        bulk_min=min(x["bulk"] for x in rows),
        seeded=init is not None,
    )
    lvl, bad_hard, bad_loose = grade(r)
    r["grade"], r["grade_bad"], r["grade_loose"] = lvl, bad_hard, bad_loose
    r["n_out"] = int(np.sum(p_out > 0.5))
    return r


# --------------------------------------------------------------------------
# K 选择
# --------------------------------------------------------------------------

def grade(r):
    """采样质量分级：2 = 严格通过，1 = 可用但有瑕疵，0 = 不可用。

    为什么分级而不是单一布尔：真实数据上严格标准经常一条都过不了，
    全判"不合格"会让选择退化成"按 elpd 硬选"，反而更糟。分级的作用是
    **优先在严格通过的里选**，没有才降级，并且明确告诉用户降级了。

    严格：R̂<1.01、bulk ESS>400、tail ESS>400、0 divergence、触顶<1%
    可用：R̂<1.05、bulk ESS>100、tail ESS>100、divergence<0.5%、触顶<5%

    这些是工程上的实用阈值（Vehtari 2021 的推荐量级），不是数学定理。
    """
    def fails(rt, bu, ta, dv, cp):
        out = []
        if r["rhat_max"] >= rt:
            out.append(f"R̂={r['rhat_max']:.3f}")
        if r["bulk_min"] <= bu:
            out.append(f"bulk={r['bulk_min']:.0f}")
        if r["tail_min"] <= ta:
            out.append(f"tail={r['tail_min']:.0f}")
        if r["ndiv"] > dv:
            out.append(f"div={r['ndiv']}")
        if r["tds"]["frac_cap"] >= cp:
            out.append(f"触顶={r['tds']['frac_cap'] * 100:.0f}%")
        return out

    hard = fails(1.01, 400, 400, 0, 0.01)
    loose = fails(1.05, 100, 100, 0.005 * r["n_draw"], 0.05)
    return (2 if not hard else (1 if not loose else 0)), hard, loose


def select_k(res_list):
    """在多个 K 之间选择。四条规则，全部可复核。

    1) **硬性剔除**（任一触发即淘汰）：
       · 采样分级 0（R̂/ESS/divergence/触顶 任一越线）
       · **物理峰位口径** `p_peak_out_max > 0.5`；峰位诊断不可用（NaN）时
         也淘汰，**绝不退回 τ 口径**（见下）
       · `frac_neg > bm_diff.NONPHYS_TOL`（φζ>1 的样本占比超容忍上限）
       · LOO 近似不可靠（Pareto-k>0.7 超 10%）、LOO 不可用
    2) 剩下的按 **elpd_loo 配对比较**排序，取最好。
    3) **简约优先**：若某个更小的 K 与最好者满足 `deficit < max(2, dse)`
       （差值落在不确定度内），取更小的 K。
    4) 若全部被剔除，仍返回一个 dict（下游 report/plot/save_out 需要它），
       但打上 `selection_status` 并在 note 首条写明「不输出最优 K 的结论」，
       **绝不把「相对最好」包装成「最优」**：
         ok         有 K 过完整硬筛选
         degraded   只降到「可用」档（分级≥1 但被物理/峰位/LOO 规则剔除）
         rejected   连「可用」都没有，结果只是相对排序、无物理意义

    为什么硬筛选**只认 p_peak_out_max、不认 p_out**：`p_out` 是给人看的
    兼容字段，峰位不可用时会退回 τ 口径；如果拿它做硬筛选，模型选择就
    不知道自己在用另一套口径，等于悄悄换了判据。峰位诊断不可用本身
    就是一个结论（这个模型没法用物理峰位评估），应当淘汰而不是绕过。

    注意：这里的「最好」只在**扫过的 K 范围**内成立，不是全局最优。
    """
    note, ok = [], []
    tol = getattr(DIFF, "NONPHYS_TOL", 0.01) if DIFF is not None else 0.01
    for r in res_list:
        hard = []                                    # 直接剔除的理由
        if r["grade"] == 0:
            hard += r["grade_loose"]

        ppm = r.get("p_peak_out_max", float("nan"))
        if not np.isfinite(ppm):
            hard.append("物理峰位诊断不可用（φζ>1 样本过多）；"
                        "不能用 τ 口径代替完成硬筛选")
        elif ppm > 0.5:
            hard.append(f"{int(np.nansum(np.asarray(r['p_peak_out']) > 0.5))} "
                        f"个分量 P(峰位出窗)>0.5")

        fn = r.get("frac_neg", float("nan"))
        if np.isfinite(fn) and fn > tol:
            hard.append(f"{fn * 100:.1f}% 样本 φζ>1（容忍上限 {tol * 100:.0f}%，"
                        f"解析 DRT 出现负值）")

        if np.isfinite(r["bad_k"]) and r["bad_k"] > 0.10:
            hard.append(f"Pareto-k>0.7 占 {r['bad_k'] * 100:.0f}%")
        if not np.isfinite(r["elpd"]):
            hard.append("LOO 不可用")
        if hard:
            note.append(f"K={r['K']} **剔除**：" + "；".join(hard))
        elif r["grade"] == 1:
            note.append(f"K={r['K']} 未过严格线，降级为可用："
                        + "；".join(r["grade_bad"]))
            ok.append(r)                             # 仍可参与比较
        else:
            ok.append(r)

    usable = [r for r in res_list if r["grade"] >= 1]
    pool = ok or usable or list(res_list)
    status = "ok" if ok else ("degraded" if usable else "rejected")
    if not ok:
        note.append("严格标准下没有 K 全过（分级≥1 但被物理有效性/峰位/LOO "
                    "规则剔除），已降级到「可用」档；下面的选择只是相对最好，"
                    "引用时必须说明它没过硬筛选。"
                    if usable else
                    "**没有任何 K 达到可用**，下面的选择只是相对最好，不能当结论。")
    if status == "rejected":
        note.insert(0, "**不输出最优 K 的结论**：所有候选 K 都被硬性剔除或采样不可用；"
                       "下面的 elpd 排名仅是相对排序、无物理意义。请回查 K-K 门禁、"
                       "噪声水平与模型基底，不要引用这个 K。")

    pool = sorted(pool, key=lambda r: (-r["elpd"] if np.isfinite(r["elpd"]) else 1e18))
    best = pool[0]
    # 注意：arviz 的 elpd_diff 是**正数表示更差**（deficit），且以全局最好者为基准。
    # 用 abs 与之解耦，无论 arviz 将来改符号约定都不会错。
    def deficit(r):
        return abs(r["elpd_diff"]) if np.isfinite(r["elpd_diff"]) else np.inf

    close = [r for r in pool
             if r["K"] < best["K"] and deficit(r) < max(2.0, r["dse"])]
    if close:
        small = min(close, key=lambda r: r["K"])
        note.append(f"K={best['K']} 与 K={small['K']} 的 elpd 差 "
                    f"{deficit(small):.1f} ± {small['dse']:.1f}，"
                    f"小于 max(2, dse) → 按简约优先取 K={small['K']}")
        best = small
    best["selection_status"] = status
    return best, note


# --------------------------------------------------------------------------
# 打印
# --------------------------------------------------------------------------

def report(res, include_L, zeta, f):
    r = res
    K = r["K"]
    # 拟合前的 K-K 门禁结论要跟着结果一起出现，免得只看日志尾部就漏掉。
    kk = r.get("kk")
    if kk:
        print(f"\n=== ⓪ K-K 事前门禁：{kk.get('status')} ===")
        print(f"  {kk.get('reason', '')}")
    st = r.get("selection_status")
    if st in ("degraded", "rejected"):
        print(f"  ** 模型选择状态 {st}：本结果不是「最优 K」的结论 **")
    tag = {2: "严格通过", 1: "可用（有瑕疵）", 0: "**不可用**"}[r["grade"]]
    print(f"\n=== 拟合（K={K}, ζ={'free' if zeta == 'free' else zeta}"
          f"{', +L' if include_L else ''}, 播种={'是' if r['seeded'] else '否'}, "
          f"{r['t']:.0f}s） ===")
    print(f"  divergences {r['ndiv']}/{r['n_draw']}   "
          f"树深度触顶 {r['tds']['frac_cap'] * 100:.1f}%   采样质量 {tag}")
    if r["grade"] < 2 and r["grade_bad"]:
        print(f"    未过严格线：{'；'.join(r['grade_bad'])}")
    if r["grade"] == 0:
        print(f"    不可用原因：{'；'.join(r['grade_loose'])}")
    extra = (f"   落后最好者 {abs(r['elpd_diff']):.1f} ± {r['dse']:.1f}"
             if r["rank"] > 0 else "")
    print(f"  elpd_loo {r['elpd']:.1f} ± {r['elpd_se']:.1f}{extra}")
    if r["frac_neg"] > 0:
        _tol = (getattr(DIFF, "NONPHYS_TOL", 0.01) * 100) if DIFF is not None else 1.0
        _verdict = ("**已超容忍上限，按规则淘汰**" if r["frac_neg"] * 100 > _tol
                    else f"在 {_tol:.0f}% 容忍范围内")
        print(f"  ** {r['frac_neg'] * 100:.1f}% 的（分量×样本）满足 φζ>1"
              f"（{_verdict}），其解析 DRT 出现负值，"
              f"DRT 曲线只作可视化，不可读峰面积。")

    print("\n=== 收敛诊断（rank-normalized split R̂ / bulk & tail ESS） ===")
    E.print_param_diag(r["rows"])
    print(f"  Z(f) 跨链：实部 R̂={r['zd']['max_rhat_re']:.3f} "
          f"bulk={r['zd']['min_bulk_re']:.0f} | 虚部 "
          f"R̂={r['zd']['max_rhat_im']:.3f} bulk={r['zd']['min_bulk_im']:.0f}")

    print("\n=== 后验分量（按峰位从高到低；w 是低频电阻贡献） ===")
    # f_τ 与 f_peak 是两个东西：ζ=1 时相同，ζ≠1 时差 2%~25%。
    # 表里只放 f_τ（参数），峰位放在下面的退化检查块里（那是物理量）。
    print(f"  {'#':<3}{'f_τ / Hz':>11}{'90% 区间':>22}{'P(τ出窗)':>9}{'P(峰出窗)':>9}"
          f"{'w / Ω':>10}{'90% 区间':>18}{'φ':>7}")
    for i in range(K):
        rng_f = "[{:.3g}, {:.3g}]".format(r["f0lo"][i], r["f0hi"][i])
        rng_w = "[{:.3g}, {:.3g}]".format(r["wlo"][i], r["whi"][i])
        pk = r["p_peak_out"][i]
        pks = f"{pk:.2f}" if np.isfinite(pk) else "  — "
        print(f"  {i + 1:<3}{r['f0'][i]:>11.4g}{rng_f:>22}{r['p_tau_out'][i]:>9.2f}{pks:>9}"
              f"{r['w'][i]:>10.3f}{rng_w:>18}{r['phi'][i]:>7.3f}")
    print(f"  Σw={r['w'].sum():.4g} Ω   R_inf={r['R_inf']:.4g} Ω"
          + (f"   L={r['L'] * 1e6:.3g} µH" if include_L else ""))
    print("  f_τ=1/(2πτ) 是参数特征频率；P(峰出窗) 用 DRT 实际峰位，"
          "**模型选择的硬筛选只认它**。φζ>1 时峰位无意义 → 标 — 并按"
          "「峰位诊断不可用」淘汰，**不退回 τ 口径充数**。")
    if r.get("diffusion"):
        print("  扩散项：" + _diff_line(r))
        if np.isfinite(r.get("p_diff_small", np.nan)):
            print(f"  P({r['diff_amp_key']} < 1e-3·|Z|max) = {r['p_diff_small']:.2f}"
                  f"  ← 后验概率，衡量「幅值相对整条谱是否可忽略」；"
                  f"**不等于「没有扩散」**")

    # ---- 退化检查 ----
    sig = r.get("degeneracy") or []
    if sig:
        print("\n=== HN ↔ Warburg 退化检查（峰位/简并度为主，φζ 为辅）===")
        print("  位置按 **DRT 实际峰位** 报——判「弛豫在不在窗口内」看峰；"
              "τ 只用来判「参数是否顶在先验边界」。")
        shifted = []
        nonphys = []
        for i, sk in enumerate(sig):
            if not np.isfinite(sk["f_peak"]):
                loc = f"峰位不适用（{sk['frac_nonphysical'] * 100:.0f}% 样本 φζ>1）"
                fpk = "  —  "
            else:
                loc = ("窗口内" if sk["side"] == "in"
                       else f"窗口{'下' if sk['side'] == 'below' else '上'}方 "
                            f"{sk['outside_dec']:.1f} dec")
                fpk = f"{sk['f_peak']:.3g}"
            print(f"  #{i + 1}  φζ={sk['pz']:.3f} [{sk['pz_lo']:.3f}, {sk['pz_hi']:.3f}]"
                  f"  f_peak={fpk} Hz（{loc}）  f_τ={sk['f_tau']:.3g} Hz"
                  f"  P(出窗)={sk['p_peak_out']:.2f}"
                  f"  corr(lnR,lnτ)={sk['corr_lnR_lntau']:+.2f}  "
                  f"脊线比={sk['ridge_ratio']:.2f}  贴τ边界 {sk['at_bound'] * 100:.0f}%"
                  f"  → {sk['level']}")
            if sk["level"] != "no":
                print(f"      {'；'.join(sk['why'])}")
            if sk.get("nonphysical"):
                nonphys.append(i + 1)
            elif abs(sk["peak_shift_dec"]) > 0.1:
                shifted.append((i + 1, sk))
        # ζ≠1 时峰位与 1/(2πτ) 能差到 0.1 dec 以上，两个口径会给出不同结论
        for i, sk in shifted:
            print(f"  ** #{i}（φ={sk['phi']:.2f}, ζ={sk['zeta']:.2f}）的峰位与 1/(2πτ) "
                  f"差 {sk['peak_shift_dec']:+.2f} dec：用 f_τ 代替峰位判断会得出"
                  f"不同结论（τ 口径 {sk['tau_outside_dec']:.1f} dec / "
                  f"{sk['tau_side']}），本表以峰为准 **")
        for i in nonphys:
            sk = sig[i - 1]
            tol = getattr(DIFF, "NONPHYS_TOL", 0.01) * 100
            print(f"  ** #{i} 有 {sk['frac_nonphysical'] * 100:.1f}% 的样本 φζ>1"
                  f"（容忍上限 {tol:.0f}%）：解析 DRT 出现负值，"
                  f"**这个分量不是非负弛豫分布**，峰位/面积都不可引用；"
                  f"该模型按物理有效性规则不合格（模型选择会硬淘汰） **")
        ns = sum(1 for x in sig if x["level"] == "strong")
        nw = sum(1 for x in sig if x["level"] == "weak")
        if ns or nw:
            tag = (f"{ns} 强 / {nw} 弱" if nw else f"{ns} 个")
            tail = ("** 已有显式扩散项，仍有分量被挤到窗口外并沿脊线简并——"
                    "扩散项没吃干净，或数据里不止一种扩散 **" if r.get("diffusion")
                    else "** 先加 --diffusion Ws 做对照，再判断 K **")
            print(f"  {tag}分量符合「在代替扩散尾」。{tail}")
        else:
            print("  没有分量符合扩散尾指纹。")

    # ---- 扩散项可分辨性：这是模型比较的**前提**，不是事后补充 ----
    dres = r.get("diffusion_resolution")
    if dres and dres.get("level") in ("resolvable", "indistinguishable"):
        print("\n=== 扩散项可分辨性（Ws/Wo 能不能分） ===")
        print(f"  {dres['note']}")
        if dres["level"] == "indistinguishable":
            print("  ** 本频段两个模型同形：elpd 差不能当「有限长扩散」的证据，"
                  "只能报「不可分」**")

    # ---- 参数相关：边际区间看不出的简并，只能从这里看 ----
    pc = r.get("param_corr")
    if pc:
        print("\n=== 参数相关矩阵（简并只在联合分布里可见） ===")
        print(CORR.format_report(pc) if CORR is not None else "")
        print("  完整矩阵：python bm_corr.py <out>/real_*.json 同目录的 "
              "*_samples.npz --matrix")

    print("\n=== 拟合优度（没有真值，地板只能用后验 σ） ===")
    print(f"  rms  实部 {r['rms_re']:.4g} Ω / σ {r['s_re']:.4g} = "
          f"{r['rms_re'] / r['s_re']:.2f}   虚部 {r['rms_im']:.4g} Ω / σ "
          f"{r['s_im']:.4g} = {r['rms_im'] / r['s_im']:.2f}")
    print(f"  残差 acf(1–5) max = {r['acf']:.3f}（辅助指标；白噪声参考 "
          f"{1.96 / np.sqrt(f.size):.2f}，但频率是对数分布，滞后≠固定频距）")
    zr = r["trend"]["re"] / max(r["s_re"], 1e-12)
    zi = r["trend"]["im"] / max(r["s_im"], 1e-12)
    nl = r.get("null") or {}
    ptr = (float(np.mean(nl["trend"] < max(zr, zi))) * 100 if "trend" in nl else np.nan)
    pacf = (float(np.mean(nl["acf"] < r["acf"])) * 100 if "acf" in nl else np.nan)
    print(f"  平滑趋势 {zr:.2f}σ(实) / {zi:.2f}σ(虚) → 零分布第 {ptr:.0f} 百分位"
          f"（>99 才算形状不对）")
    print(f"  残差 acf {r['acf']:.3f} → 零分布第 {pacf:.0f} 百分位"
          f"（辅助；对数频率下滞后≠固定频距）")
    print(f"  PPC 覆盖率 {r['pp']['cover_re']:.3f}(实) / "
          f"{r['pp']['cover_im']:.3f}(虚)   目标 0.90")

    print("\n  残差分段（按 log f 三分，看是否系统性偏）")
    print(f"    {'频段 / Hz':>20}{'n':>4}{'mean Re':>10}{'mean Im':>10}"
          f"{'rms Re':>9}{'rms Im':>9}")
    for b in r["bins"]:
        rng = f"{b['f_lo']:.3g}–{b['f_hi']:.3g}"
        print(f"    {rng:>20}{b['n']:>4}{b['mean_re']:>10.3g}"
              f"{b['mean_im']:>10.3g}{b['rms_re']:>9.3g}{b['rms_im']:>9.3g}")


def judge(res, f, include_L):
    """把诊断翻成可执行的建议。真实数据没有真值，只能靠这些证据。"""
    r = res
    K, msg = r["K"], []
    # KK 门禁是事前证据，排在所有事后诊断之前。
    kk = r.get("kk")
    if kk and kk.get("status") in ("warn", "fail", "unavailable"):
        msg.append(f"**KK 门禁 {kk['status']}**：{kk.get('reason', '')}。"
                   f"引用参数前先看分频段残差与 attribution；"
                   f"似然/LOO 检验不了数据本身是否自洽。")
    st = r.get("selection_status")
    if st == "rejected":
        msg.insert(0, "**模型选择被拒绝**：没有任何 K 通过硬性筛选，"
                      "本次不给出「最优 K」，elpd 排名仅相对、无物理意义。")
    elif st == "degraded":
        msg.append("模型选择降级：只在「可用」档里做相对比较，"
                   "引用 K 时须说明它没过硬筛选。")

    # 参数简并：边际区间看着都窄，但两个参数其实只定出一个组合。
    pc = r.get("param_corr") or {}
    if pc.get("strong"):
        who = "；".join(f"{p['a']}↔{p['b']}(ρ={p['rho']:+.2f})"
                        for p in pc["strong"][:3])
        msg.append(f"**参数强相关**：{who}。这些参数只能定出一个组合，"
                   f"单独引用其中一个会被读成「已经定住」，实际没有。")

    # 扩散项：先确认能不能分，再谈比较结果。
    dres = r.get("diffusion_resolution") or {}
    if dres.get("level") == "indistinguishable":
        msg.append("**Ws/Wo 在本频段不可分**：" + dres.get("note", "")
                   + " 不要用 elpd 差宣称「存在有限长扩散」。")
    elif dres.get("level") == "resolvable":
        msg.append("扩散项可分辨：" + dres.get("note", ""))

    q_re = r["rms_re"] / max(r["s_re"], 1e-12)
    q_im = r["rms_im"] / max(r["s_im"], 1e-12)

    if max(q_re, q_im) > 2.0:
        msg.append(f"**拟合不住**：rms/σ = {q_re:.1f}(实)/{q_im:.1f}(虚) > 2。"
                   f"先查是否缺项（Warburg / 电感 / 多孔扩散），而不是加 K。")
    elif max(q_re, q_im) < 0.6:
        msg.append(f"rms/σ = {q_re:.2f}/{q_im:.2f} < 0.6，残差比模型自估噪声还小——"
                   f"σ 可能被高估，别把 σ 当真实误差棒用。")

    # 趋势：先按**各自**的 σ 标准化（实部/虚部噪声不同，不能混用），
    # 再在零分布里查百分位。二值阈值在 n=31 时太容易误报。
    nl = r.get("null") or {}
    zr = r["trend"]["re"] / max(r["s_re"], 1e-12)
    zi = r["trend"]["im"] / max(r["s_im"], 1e-12)
    z = max(zr, zi)
    p_tr = float(np.mean(nl["trend"] < z)) if "trend" in nl else float("nan")
    p_acf = float(np.mean(nl["acf"] < r["acf"])) if "acf" in nl else float("nan")
    r["p_trend"], r["p_acf"] = p_tr, p_acf
    if p_tr > 0.99:
        msg.append(f"**残差有系统性弯曲**：趋势 {z:.2f}σ 位于白噪声零分布第 "
                   f"{p_tr * 100:.1f} 百分位（实 {zr:.2f}σ / 虚 {zi:.2f}σ）。"
                   f"这比 acf 直接——形状不对，不是随机误差。看分段表是哪段偏。")
    elif p_tr > 0.95:
        msg.append(f"趋势 {z:.2f}σ 位于零分布第 {p_tr * 100:.0f} 百分位，"
                   f"刚过 95% 线但不极端，先当提示、别据此改模型。")
    elif p_acf > 0.99:
        msg.append(f"残差 acf={r['acf']:.2f}（零分布第 {p_acf * 100:.0f} 百分位）偏高，"
                   f"但趋势只有 {z:.2f}σ。频率是对数分布，acf 只当辅助，"
                   f"以分段表和趋势为准。")

    # 这里用**峰位口径**（条件概率）给提示，与 select_k 的硬筛选同一口径。
    # 峰位诊断不可用（NaN）时不再退回 τ 口径糊过去，而是直接说明原因。
    hi_out = [i for i in range(K) if np.isfinite(r["p_peak_out"][i])
              and r["p_peak_out"][i] > 0.5]
    mid_out = [i for i in range(K) if np.isfinite(r["p_peak_out"][i])
               and 0.05 < r["p_peak_out"][i] <= 0.5]
    nan_out = [i for i in range(K) if not np.isfinite(r["p_peak_out"][i])]
    if hi_out:
        who = ", ".join(f"#{i + 1} P={r['p_peak_out'][i]:.2f}" for i in hi_out)
        msg.append(f"**{len(hi_out)} 个分量大概率在窗口外**（{who}，峰位口径）："
                   "这是「用窗口外分量吸收缺项」的典型症状，面积不可信。")
    if mid_out:
        who = ", ".join(f"#{i + 1} P={r['p_peak_out'][i]:.2f}" for i in mid_out)
        msg.append(f"{len(mid_out)} 个分量跨在窗口边界上（{who}）：峰位本身不可信，"
                   f"只当「这里有个未分辨的过程」看。")
    if nan_out:
        who = ", ".join(f"#{i + 1}" for i in nan_out)
        msg.append(f"{who} 的物理峰位诊断不可用（φζ>1 样本过多）："
                   f"**峰位口径失效，不能改用 τ 口径充数**——"
                   f"这几条分量按物理有效性规则应被淘汰。")

    tot = float(np.sum(r["w"])) or 1.0
    frag = [i for i in range(K) if r["w"][i] < 0.02 * tot]
    if frag:
        who = ", ".join(f"{r['w'][i]:.2f} Ω" for i in frag)
        msg.append(f"{len(frag)} 个分量是碎片（面积 < 总面积 2%：{who}）。多半是 K 给多了。")

    cov = min(r["pp"]["cover_re"], r["pp"]["cover_im"])
    if cov < 0.80:
        msg.append(f"PPC 覆盖率只有 {cov:.3f}（目标 ~0.90）：区间太窄，"
                   f"通常是 σ 先验偏小或后验被少数点主导。")
    elif cov > 0.97:
        msg.append(f"PPC 覆盖率 {cov:.3f} 明显高于 0.90：σ 偏大，区间过宽，"
                   f"后验可能比实际更不确定。")

    if r["grade"] == 0:
        msg.append(f"**采样不可用**：{'；'.join(r['grade_loose'])}。这个 K 的数字先别引用。")
    elif r["grade"] == 1:
        msg.append(f"采样只到「可用」档：{'；'.join(r['grade_bad'])}。"
                   f"提高 warmup / max_tree_depth 或 --ta 0.95 后重跑再看。")
    if r["frac_neg"] > 0:
        _tol = (getattr(DIFF, "NONPHYS_TOL", 0.01) * 100) if DIFF is not None else 1.0
        msg.append(f"{r['frac_neg'] * 100:.1f}% 的样本 φζ>1（容忍 {_tol:.0f}%），"
                   f"解析 DRT 有负值；要定量 γ 就对后验样本逐个跑 Tikhonov 反演。")
    if include_L:
        msg.append("已启用 L：高频感抗由 L 承担，不再挤进 R_inf。")
    return msg


# --------------------------------------------------------------------------
# 出图
# --------------------------------------------------------------------------

def plot_real(name, f, Z, res, pres, pk, pk_all, include_L, path):
    s, band = res["s"], res["band"]
    (br, mr, tr), (bi, mi, ti) = band
    grid, gmed, glo, ghi = B.drt_band(s)
    s_re, s_im = res["s_re"], res["s_im"]

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))

    a = ax[0, 0]
    a.plot(Z.real, -Z.imag, "o", ms=4.5, mfc="none", mec="#1f4e79", mew=1.1,
           label="data")
    a.plot(mr, -mi, "-", lw=1.6, color="#c0392b", label="posterior median")
    a.fill(np.concatenate([br, br[::-1]]),
           np.concatenate([-bi, (-bi)[::-1]]),
           color="#c0392b", alpha=0.13, lw=0, label="90% band")
    a.set_xlabel(r"Re $Z$ / $\Omega$"); a.set_ylabel(r"$-$Im $Z$ / $\Omega$")
    a.set_title("Nyquist"); a.legend(fontsize=7); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.loglog(f, np.abs(Z), "o", ms=4.5, mfc="none", mec="#1f4e79", mew=1.1)
    a.loglog(f, np.hypot(mr, mi), "-", lw=1.6, color="#c0392b")
    a.fill_between(f, np.hypot(br, bi), np.hypot(tr, ti), color="#c0392b",
                   alpha=0.13, lw=0)
    # 对数轴默认把"2×10¹、4×10¹"这些次刻度标签画出来，主刻度反被淹没
    a.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
    a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.yaxis.set_minor_formatter(mticker.NullFormatter())
    a.set_xlabel("f / Hz"); a.set_ylabel(r"$|Z|$ / $\Omega$")
    a.set_title("Bode: magnitude"); a.grid(alpha=0.3, which="both")

    a = ax[0, 2]
    a.semilogx(f, np.degrees(np.angle(Z)), "o", ms=4.5, mfc="none",
               mec="#1f4e79", mew=1.1)
    a.semilogx(f, np.degrees(np.angle(mr + 1j * mi)), "-", lw=1.6,
               color="#c0392b")
    a.set_xlabel("f / Hz"); a.set_ylabel("phase / deg")
    a.set_title("Bode: phase"); a.grid(alpha=0.3, which="both")

    a = ax[1, 0]
    xf = 1.0 / (2 * np.pi * np.exp(grid))
    a.semilogx(1.0 / (2 * np.pi * pres["tau"]), pres["gamma"], "--", lw=1.4,
               color="#555555", alpha=0.85, label="pre-screen (data only)")
    a.semilogx(xf, gmed, "-", lw=1.6, color="#8e44ad",
               label="posterior (HN closed form)")
    a.fill_between(xf, glo, ghi, color="#8e44ad", alpha=0.15, lw=0)
    for p in pk:
        a.plot([p["f_peak_Hz"]], [p["gamma_max"]], "v", ms=6, color="#555555")
    for p in pk_all:
        if p not in pk:
            a.plot([p["f_peak_Hz"]], [p["gamma_max"]], "x", ms=5, mew=1.2,
                   color="#999999")
    a.axvspan(f.min(), f.max(), color="#f1c40f", alpha=0.07, lw=0)
    # 竖排长文本会顺着文字方向往下压到峰上，所以只留 #编号，数值放右上角
    for rank, i in enumerate(range(res["K"])):
        a.axvline(res["f0"][i], color="k", ls=":", lw=0.9, alpha=0.55)
        a.text(res["f0"][i], 0.985, f"#{rank + 1}",
               transform=a.get_xaxis_transform(), fontsize=6.8,
               va="top", ha="center", color="k", alpha=0.75)
    a.text(0.985, 0.94,
           "\n".join(f"#{k + 1} {res['f0'][k]:.4g} Hz  {res['w'][k]:.3g} Ω"
                     for k in range(res["K"])),
           transform=a.transAxes, ha="right", va="top", fontsize=6.8,
           family="monospace",
           bbox=dict(fc="white", ec="#bbbbbb", lw=0.6, alpha=0.9, pad=2.4))
    # ζ 自由时 φζ>1 的样本 γ 会出负值。**不能**把下界钉在 -5% 最大值，
    # 那正好把负的 5% 带裁掉，等于把"这不是物理分布"这件事藏起来。
    gmax = max(gmed.max(), 1e-9)
    if glo.min() < -0.02 * gmax:
        a.axhline(0, color="k", lw=0.8, ls="-", alpha=0.6)
        a.set_ylim(bottom=1.15 * glo.min())
        a.text(0.985, 0.70,
               f"5% band dips to {glo.min():.3g} < 0\n"
               f"= φζ>1 samples, γ is not a physical DRT",
               transform=a.transAxes, fontsize=7.0, color="#c0392b",
               va="top", ha="right",
               bbox=dict(fc="white", ec="#c0392b", lw=0.6, alpha=0.9, pad=2.2))
    else:
        a.set_ylim(bottom=-0.05 * gmax)
    a.set_xlabel("f / Hz"); a.set_ylabel(r"$\gamma$ / $\Omega$")
    # 用 --diffusion 时，这条紫线只画 HN 分量的解析 DRT，**不含扩散项**。
    # 扩散项的 DRT 是 A_W√τ/π，∫γ dlnτ 发散，根本没法画在这张图上；
    # 不标注的话，读图的人会以为紫线是"整个拟合阻抗的完整 DRT"。
    # 注：第二行放进 title 而不是 axes 上方的 text——text 在 y=1.015 会和
    # title 叠在一起（试过，两行字压成一团）。
    _ttl = (f"DRT: pre-screen {len(pk_all)} raw / {len(pk)} used "
            f"(yellow = measured band)")
    if res.get("diffusion"):
        _ttl += (f"\npurple = HN components only; {res['diffusion']} contribution "
                 f"NOT included (its γ diverges)")
    a.set_title(_ttl, fontsize=9.2)
    a.legend(fontsize=7, loc="upper left")
    a.grid(alpha=0.3, which="both")

    rr, ri = Z.real - mr, Z.imag - mi
    for k, (rv, sg, nm) in enumerate(((rr, s_re, "Re"), (ri, s_im, "Im"))):
        a = ax[1, 1 + k]
        a.semilogx(f, rv, "o-", ms=4, lw=0.8, color="#2471a3")
        a.axhline(0, color="k", lw=0.8)
        for sgn in (1, -1):
            a.axhline(sgn * sg, color="#c0392b", ls=":", lw=1)
        # 分段均值：方块。位置按频段几何中心，与点疏密无关
        for b in res["bins"]:
            fc = float(np.sqrt(b["f_lo"] * b["f_hi"]))
            v = b["mean_re"] if nm == "Re" else b["mean_im"]
            a.plot([fc], [v], "s", ms=6, mfc="none", mec="#e67e22", mew=1.4,
                   zorder=5, label="bin mean" if b is res["bins"][0] else None)
        a.set_xlabel("f / Hz"); a.set_ylabel(f"residual {nm} / $\\Omega$")
        a.set_title(f"residual {nm} (dotted = posterior $\\sigma$={sg:.3g})")
        a.grid(alpha=0.3, which="both")

    fig.suptitle(f"{name}   K={res['K']}   {f.size} pts, "
                 f"{f.min():.3g}–{f.max():.3g} Hz"
                 f"{'   +L' if include_L else ''}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def prescreen_stage(f, Z):
    """① 预处理 + 传统 DRT。只产出候选与初值，不产出结论。"""
    t0 = time.time()
    pres, pk, pk_all = prescreen(f, Z)
    dt = time.time() - t0
    print(f"\n=== ① DRT 预判断（{dt * 1000:.0f} ms, λ={pres['lam']:.3g}） ===")
    print(f"  原始 {len(pk_all)} 峰 → 频段内且 ≥2% 面积 {len(pk)} 个"
          f"（这是**候选**，不是 K 的定值）")
    for p in pk:
        print(f"    [用] f={p['f_peak_Hz']:10.4g} Hz  R={p['R_ohm']:7.3f} Ω  "
              f"峰宽={p['width_dec']:.2f} decade")
    for p in pk_all:
        if p not in pk:
            why = "窗口外" if not (f.min() <= p["f_peak_Hz"] <= f.max()) else "面积过小"
            print(f"    [弃] f={p['f_peak_Hz']:10.4g} Hz  R={p['R_ohm']:7.3f} Ω  ({why})")
    print(f"  R_inf(DRT)={pres['R_inf']:.4g} Ω   ΣR(峰)={pres['R_total']:.4g} Ω")
    edge = max(pres["edge_low_frac"], pres["edge_high_frac"])
    if edge > 0.5:
        print(f"  ** 端部堆积：γ 在网格边界仍为峰高的 {edge * 100:.0f}%，"
              f"窗口外还有未分辨的弛豫 → 峰数可能被**低估**，K 范围要往上放。")
    return pres, pk, pk_all, dt


def _write_kk_reject(path, f, Z, kk, kk_mode):
    """gate 模式且 K-K 判 fail：跳过整条拟合链，只落一份可机读的拒绝记录。

    必须写 `real_<stem>.json`——`bmdrt.py` 的批量流程靠它判「流程跑完」，
    不写会被记成「跑完但没结果」而误报 [FAIL]。`fit_performed=false`
    就是批量流程区分「正常拒绝」与「真的拟合成功」的契约字段。
    """
    OUT.mkdir(exist_ok=True)
    stem = Path(path).stem
    js = OUT / f"real_{stem}.json"
    rec = dict(
        file=stem, n=int(f.size), f_min=float(f.min()), f_max=float(f.max()),
        fit_performed=False, kk=kk, kk_mode=kk_mode,
        k_selection_status="rejected_by_kk",
        reason=kk.get("reason", ""),
        K=None, sample_grade=None, components=[],
        warnings=["**KK 门禁拒绝**：数据不自洽，本次未拟合，不输出最优 K。"],
        k_selection=["**不输出最优 K 的结论**：拟合前 K-K 门禁判 fail。"],
    )
    js.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=float),
                  encoding="utf-8")
    return js


def save_out(stem, f, Z, res, pres, include_L, zeta, tips, K_sel_note):
    OUT.mkdir(exist_ok=True)
    js = OUT / f"real_{stem}.json"
    summary = dict(
        file=stem, n=int(f.size), f_min=float(f.min()), f_max=float(f.max()),
        K=int(res["K"]), include_L=bool(include_L),
        zeta=("free" if zeta == "free" else float(zeta)),
        lam_drt=float(pres["lam"]), R_inf_drt=float(pres["R_inf"]),
        t_fit_s=float(res["t"]), divergences=int(res["ndiv"]),
        tree_cap=float(res["tds"]["frac_cap"]),
        sample_grade=int(res["grade"]), grade_fail=res["grade_bad"],
        elpd_loo=res["elpd"], elpd_se=res["elpd_se"],
        elpd_diff=res["elpd_diff"], dse=res["dse"],
        rms_re=res["rms_re"], rms_im=res["rms_im"],
        sigma_re=res["s_re"], sigma_im=res["s_im"],
        acf=float(res["acf"]), trend_re=res["trend"]["re"],
        trend_im=res["trend"]["im"],
        cover_re=float(res["pp"]["cover_re"]),
        cover_im=float(res["pp"]["cover_im"]),
        R_inf=res["R_inf"], W=res["W"],
        L_uH=(res["L"] * 1e6 if include_L else None),
        components=[dict(f0=float(res["f0"][i]), f0_lo=float(res["f0lo"][i]),
                         f0_hi=float(res["f0hi"][i]), w=float(res["w"][i]),
                         w_lo=float(res["wlo"][i]), w_hi=float(res["whi"][i]),
                         phi=float(res["phi"][i]),
                         p_outside=float(res["p_out"][i]),
                         p_tau_outside=float(res["p_tau_out"][i]),
                         p_peak_outside=(float(res["p_peak_out"][i])
                                         if np.isfinite(res["p_peak_out"][i]) else None),
                         p_outside_caliber=str(res["p_out_caliber"][i]))
                    for i in range(res["K"])],
        resid_bins=res["bins"],
        diag={r["name"]: {k: r[k] for k in ("rhat", "bulk", "tail")}
              for r in res["rows"]},
        k_selection=K_sel_note, warnings=tips,
        # 事前门禁 / 模型选择状态 / 后处理诊断：全部「有则写、缺则跳过」，
        # 保证旧结果文件与未启用新模块的调用方仍能读。
        fit_performed=True,
        kk=res.get("kk"),
        k_selection_status=res.get("selection_status"),
        param_corr=res.get("param_corr"),
        diffusion_resolution=res.get("diffusion_resolution"),
        # 扩散项：写进 json 才能让 status / 事后复核不重跑就看得到
        diffusion=res.get("diffusion"),
        n_warburg_like=res.get("n_warburg_like", 0),
        p_diff_small=res.get("p_diff_small"),
        zmax_obs=res.get("zmax_obs"),
        # 模型选择实际用到的字段，单独记一份，免得事后只能从 p_out 反推：
        #   p_peak_out_max 是**唯一**参与硬筛选的窗口外判据（NaN = 不可用 → 淘汰）
        #   p_tau_out_max  只作对照，**不参与选择**
        #   frac_neg       与 bm_diff.NONPHYS_TOL 比较，超限即淘汰
        p_peak_out_max=(float(res["p_peak_out_max"])
                        if np.isfinite(res.get("p_peak_out_max", float("nan")))
                        else None),
        p_tau_out_max=(float(res["p_tau_out_max"])
                       if np.isfinite(res.get("p_tau_out_max", float("nan")))
                       else None),
        frac_neg=res.get("frac_neg"),
        nonphys_tol=getattr(DIFF, "NONPHYS_TOL", None) if DIFF is not None else None,
        degeneracy=[{k: s[k] for k in ("phi", "zeta", "pz", "pz_lo", "pz_hi",
                                      "f_tau", "f_peak", "peak_shift_dec",
                                      "tau_outside_dec", "tau_side",
                                      "outside_dec", "side", "at_bound",
                                      "p_tau_out", "p_peak_out",
                                      "frac_nonphysical", "nonphysical",
                                      "peak_usable",
                                      "corr_lnR_lntau", "ridge_ratio",
                                      "a_impl", "level")} for s in res.get("degeneracy", [])],
        diffusion_params={k: (float(np.median(v)) if isinstance(v, np.ndarray) else None)
                          for k, v in res["s"].items()
                          if k in ("Rw", "Aw", "Rwo", "tauD")},
    )
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                             default=float), encoding="utf-8")
    np.savez_compressed(OUT / f"real_{stem}_samples.npz",
                        f=f, Zre=Z.real, Zim=Z.imag,
                        **{k: v for k, v in res["s"].items()
                           if isinstance(v, np.ndarray)})
    return js


def run(path, K_force=None, krange=None, wide=False, kmax=K_MAX,
        include_L=None, zeta="1.0", dry=False, quick=False, ta=None,
        diffusion=None, kk_mode="gate", kk_opts=None, diag=None):
    """kk_mode: gate(fail 即拒绝定 K) | warn(只报告) | off(不跑门禁)。
    kk_opts 是传给 kk_check 的数值参数（c/max_M/res_tol/auto_L/...），
    不在代码里写死，由 config.json 统一管。
    diag 同理：事后诊断阈值（相关矩阵 / 扩散可分辨性），来自 config.json。
    """
    diag = diag or {}
    print(f"\n=== 读数据 {path} ===")
    f, Z = load_spectrum(path)

    if diffusion:
        print(f"  扩散项：{diffusion}（对比基线时用 --diffusion none）")

    if include_L is None:
        include_L = detect_L(f, Z)
        print(f"  电感项：{'需要（高频 −Im Z < 0）' if include_L else '不需要'}"
              f"（--L / --no-L 可覆盖）")

    # ---- ⓪ K-K 事前门禁：读数据之后、DRT 预判之前 ------------------
    # PSIS-LOO 只能在「似然假设成立」的前提下比较模型，检验不了假设本身。
    # 所以这一步必须在任何拟合之前，且 fail 时直接拒绝定 K。
    kk = None
    if kk_mode != "off" and _kk_check is not None:
        try:
            kk = _kk_check(f, Z, include_L=include_L, **(kk_opts or {}))
        except Exception as e:             # 门禁绝不能让主流程崩
            kk = {"status": "unavailable",
                  "reason": f"K-K 门禁异常，已跳过：{type(e).__name__}: {e}",
                  "attribution": [], "n": int(f.size),
                  "f_min": float(f.min()), "f_max": float(f.max())}
        print("\n=== ⓪ K-K 事前门禁（拟合前） ===")
        print(_kk_format_report(kk) if _kk_format_report is not None
              else f"  {kk['status']}: {kk['reason']}")
        if kk_mode == "gate" and kk.get("status") == "fail":
            print("\n  ** 数据不自洽，拒绝定 K；本次不做拟合 **")
            js = _write_kk_reject(path, f, Z, kk, kk_mode)
            print(f"  结果：{js}")
            return None

    pres, pk, pk_all, _ = prescreen_stage(f, Z)
    K_drt = max(1, min(len(pk), kmax))

    if dry:
        lo, hi = max(1, K_drt - 1), min(kmax, K_drt + 1)
        print(f"\n（--dry 结束）建议下一步：python bm_real.py {path} --krange {lo},{hi}")
        print(f"  若要确认局部扫描没漏掉更好的 K：--wide（扫 1..{kmax}）")
        return None

    kw = dict(KW)
    if quick:
        kw.update(n_warmup=800, n_samples=400, num_chains=2)
        print("\n  ** --quick：只做粗筛。elpd 与诊断都不可当结论，选定 K 后要重跑 **")
    if ta is not None:
        kw["target_accept"] = float(ta)
        print(f"  target_accept = {kw['target_accept']}")

    # ---- ② K 的候选范围 ----
    if K_force:
        ks = [int(K_force)]
        print(f"\n=== ② 固定 K={ks[0]} 拟合 ===")
        print("  ** 这是「给定 K 条件下」的结果，不是模型选择。"
              "要选 K 请去掉 --K，用 --krange / --wide **")
    else:
        if wide:
            lo, hi = 1, kmax
        elif krange:
            lo, hi = (int(x) for x in str(krange).split(","))
        else:
            lo, hi = max(1, K_drt - 1), min(kmax, K_drt + 1)
        ks = list(range(lo, hi + 1))
        print(f"\n=== ② 拟合 K ∈ {ks}（DRT 只给候选，K 由配对 LOO 选） ===")
        print(f"  K 上界 {kmax} 是人为设定，选出的 K 只在这个范围内最优；"
              f"--wide 扫 1..{kmax}，--krange a,b 自定义")

    res_list = []
    for K in ks:
        init = init_values(pk, pres, K) if pk else None
        print(f"\n  --- K={K} 拟合中（播种={'是' if init else '否'}）---")
        r = fit_one(f, Z, K, include_L, zeta, init=init, kw=kw,
                    diffusion=diffusion, diag=diag)
        print(f"      {r['t']:.0f}s  R̂max={r['rhat_max']:.3f} "
              f"bulk={r['bulk_min']:.0f} tail={r['tail_min']:.0f} "
              f"div={r['ndiv']} 触顶={r['tds']['frac_cap'] * 100:.1f}% "
              f"→ 分级 {r['grade']}")
        _ppm = r["p_peak_out_max"]
        _ppms = f"{_ppm:.2f}" if np.isfinite(_ppm) else "—（不可用）"
        print(f"      elpd={r['elpd']:.1f}  "
              f"rms/σ={r['rms_re'] / r['s_re']:.2f}/{r['rms_im'] / r['s_im']:.2f}  "
              f"acf={r['acf']:.2f}  P(峰位出窗)max={_ppms}"
              f"  φζ>1 占比={r['frac_neg'] * 100:.1f}%")
        res_list.append(r)

    if len(res_list) > 1:
        df = compare_models(res_list)
        print("\n=== ③ 配对比较（az.compare / PSIS-LOO，逐点相减） ===")
        cols = [c for c in ("rank", "elpd_loo", "elpd_diff", "dse", "weight", "p_loo")
                if c in df.columns]
        print(df[cols].to_string(float_format=lambda v: f"{v:.1f}"))
        best, note = select_k(res_list)
        print("\n  规则：① 硬性剔除（分级0 / P(峰位出窗)>0.5 / 峰位诊断不可用 /"
              " φζ>1 超 1% / Pareto-k>0.7 超 10%）"
              "② elpd 最大 ③ 差 < max(2, dse) 取更小 K")
        for t in note:
            print(f"    · {t}")
        st = best.get("selection_status", "ok")
        if st == "rejected":
            print("  → **没有可信的 K**：下面的 elpd 排序仅相对、无物理意义，"
                  "不输出最优 K 结论")
        else:
            extra = "，且仅来自「可用」降级档" if st == "degraded" else ""
            print(f"  → **K={best['K']}**（只在 {ks[0]}..{ks[-1]} 内最优{extra}）")
    else:
        best, note = res_list[0], []
        best["selection_status"] = "not_compared"
        print("  （只跑了一个 K，未做模型选择）")

    if kk is not None:
        best["kk"] = kk

    report(best, include_L, zeta, f)
    tips = judge(best, f, include_L)
    print("\n=== 判读建议 ===")
    for t in (tips or ["没有触发任何警告：收敛、残差、覆盖、峰位都正常。"]):
        print(f"  · {t}")

    OUT.mkdir(exist_ok=True)
    stem = Path(path).stem
    fig_path = OUT / f"real_{stem}.png"
    plot_real(stem, f, Z, best, pres, pk, pk_all, include_L, fig_path)

    # 等效电路图：画的是**这次选中的模型**，参数取后验中位数。
    # 画失败不影响主结果，所以单独兜住异常。
    circ_path = OUT / f"real_{stem}_circuit.png"
    try:
        from bm_circuit import draw as draw_circuit
        draw_circuit(best, include_L, circ_path, name=stem)
        circ_note = f"\n  电路：{circ_path}"
    except Exception as e:
        circ_note = f"\n  （电路图没画出来：{type(e).__name__}: {e}）"

    js = save_out(stem, f, Z, best, pres, include_L, zeta, tips, note)
    print(f"\n  图：{fig_path}{circ_note}\n"
          f"  结果：{js}  +  {OUT / f'real_{stem}_samples.npz'}")
    return best


def main():
    ap = argparse.ArgumentParser(
        description="真实 EIS：DRT 预判断 → K 扫描 + 配对 LOO 选择 → 诊断出图")
    ap.add_argument("csv", help="数据文件（CSV/TSV/空白分隔）")
    ap.add_argument("--K", type=int, default=None,
                    help="只跑这个 K（给定 K 的拟合，不做模型选择）")
    ap.add_argument("--krange", default=None, help="扫描范围，如 1,4")
    ap.add_argument("--wide", action="store_true",
                    help="扫 1..kmax，用于复核 DRT 局部扫描没漏掉更好的 K")
    ap.add_argument("--kmax", type=int, default=K_MAX, help=f"K 上界（默认 {K_MAX}）")
    ap.add_argument("--quick", action="store_true",
                    help="粗筛：更短的链，只用于快速看 K 的量级，不可当结论")
    ap.add_argument("--ta", type=float, default=None,
                    help="target_accept（默认 0.9；难采时可试 0.95）")
    ap.add_argument("--L", dest="L", action="store_true", default=None,
                    help="启用串联电感")
    ap.add_argument("--no-L", dest="L", action="store_false", help="强制不加电感")
    ap.add_argument("--zeta", default="1.0",
                    help="ζ 固定值，或 free 让它自由（默认 1.0 = 纯 ZARC）")
    ap.add_argument("--dry", action="store_true", help="只做 DRT 预判断")
    ap.add_argument("--diffusion", default="none",
                    choices=["none", "Ws", "Wo", "both"],
                    help="加扩散项：none(默认) / Ws=半无限 / Wo=有限长阻挡 / both。"
                         "注意 Ws 与 Wo 的高频尾完全相同，both 只适合做"
                         "'是否真的存在有限长度'的对照，不适合当最终模型")
    ap.add_argument("--kk", default="gate", choices=["gate", "warn", "off"],
                    help="拟合前 Lin-KK 门禁：gate=判 fail 就拒绝定 K（默认）/ "
                         "warn=只报告仍拟合 / off=完全跳过。"
                         "数值阈值走 config.json 的 kk 段")
    a = ap.parse_args()
    if a.zeta == "free":
        print("** --zeta free：HN 的解析 DRT 在 ζφ>1 时会出负值，"
              "后验曲线只作可视化；峰位/面积的解释要先做可辨识性检查 **")
    diff = None if a.diffusion == "none" else a.diffusion
    run(a.csv, K_force=a.K, krange=a.krange, wide=a.wide, kmax=a.kmax,
        include_L=a.L, zeta=a.zeta, dry=a.dry, quick=a.quick, ta=a.ta,
        diffusion=diff, kk_mode=a.kk)


if __name__ == "__main__":
    main()
