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
     2026-11-06 起分为两层：K 只由「采样/物理硬门禁 + 合格候选的配对 LOO」决定；
     峰位出窗、窗口内贡献、参数简并、残差结构、LOO 稳健性由
     `component_review()` 在拟合后单独报告，不再淘汰 K。

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
import bm_grade as GRADE
import drt as DRT
from bm_experiments import acf_max
from drt import prescreen

try:                                    # 扩散诊断是可选的；缺了只是少一段报告
    import bm_diff as DIFF
except Exception:                       # pragma: no cover
    DIFF = None

try:                                    # Ws 数据侧判据（端点走向 / 份额）；缺了只少证据
    import bm_ws as WS
except Exception:                       # pragma: no cover
    WS = None

try:                                    # K-K 事前门禁；缺了只是少一道体检，不阻断拟合
    from bm_kk import kk_check as _kk_check
    from bm_kk import format_report as _kk_format_report
    from bm_kk import estimate_L as _kk_estimate_L
    from bm_kk import lin_kk as _kk_lin_kk
except Exception:                       # pragma: no cover
    _kk_check = None
    _kk_format_report = None
    _kk_estimate_L = None
    _kk_lin_kk = None

try:                                    # 全参数相关矩阵（"别只看误差"那一层）
    import bm_corr as CORR
except Exception:                       # pragma: no cover
    CORR = None

try:                                    # 吸附 RL 元件（M4：corner 的 τ/Rad 粗估）
    import bm_ad as AD
except Exception:                       # pragma: no cover
    AD = None

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

KW = dict(n_warmup=3000, n_samples=1000, num_chains=3, seed=0,
          progress=False, target_accept=0.9, max_tree_depth=13,
          extra_fields=("num_steps",))
# 上面的 seed=0 是**默认值**，不是硬编码：`run(seed=)` / CLI `--seed` 可以覆盖，
# 优先级见 `_resolve_seed()`。改 seed 只换随机流，不换模型、不换初值策略。
K_MAX = 5
OUT = Path("figures")

LOG2PI = float(np.log(2 * np.pi))
NULL_CACHE: dict = {}    # f 网格字节 -> 零分布阈值（按网格分键，跨谱不串）


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
        # P2-8（二轮审查）：单位只看**相位列自己**的名字，且判不出来时
        # 报错而不是猜。旧实现两个坑：
        #   ① `any("rad" in h for h in names)` 扫全部列名——任何一列含
        #      "rad"（如 "radius"）就把整个文件当弧度；
        #   ② 近纯阻谱（相位 ≤7°、度制）不触发转换，度被当弧度。
        ph_name = names[i_ph] if names else ""
        if "rad" in ph_name:
            rad = True
        elif "deg" in ph_name or "°" in ph_name:
            rad = False
        else:
            # 表头判不出来：只有一条硬规则可用——弧度制 |phase| ≤ π，
            # 超过 π 只能是度。其余情况（max|phase| ≤ π）度/弧度都可能，
            # 宁可停下游手，也不猜（猜错会把 3° 变成 172°）。
            mx = float(np.nanmax(np.abs(ph)))
            if mx > np.pi:
                rad = False          # 弧度不可能超过 π → 必是度
            else:
                raise SystemExit(
                    f"相位列 {ph_name!r} 单位判不出来（表头无 deg/rad，"
                    f"max|phase|={mx:.3g} ≤ π，度/弧度皆可能）："
                    f"请把表头改成 phase_deg / phase_rad 之一")
        if not rad:
            # 度制一律转弧度。旧实现的 ">7 才转"启发式会把近纯阻谱的
            # 3° 当成 3 rad（172°）——正是 P2-8 的第二个坑。
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


L_SHARE_MIN = 0.01          # 感抗在 f_max 处占 |Z| 的最小比例（判据②）


def detect_L(f, Z) -> bool:
    """是否需要串联电感 L。两条判据，任一成立即加。

    ① **电感压过容性**：最高 3 点 Im Z > 0 过半。这是最干净的感抗尾，
       原判据，保留。

    ② **电感只部分抵消容性**：Im Z 仍为负，但 Lin-KK 自由拟合的串联
       电感 `L_fit` 在 f_max 处的感抗 ωL 已占 |Z| 的 ≥ `L_SHARE_MIN`。
       实测 Blank_0.1KBr_pH10_EIS_PtCu_1.2V：最高 3 点 Im Z 全负
       （-2.38/-2.87/-3.41），旧判据判「不需要」；但 L_fit=9.06 μH，
       ωL@1e4 = 0.57 Ω 占 |Z| 的 2.8%。加 L 后配对 PSIS-LOO
       Δ=+24.18（dse=4.89，4.95×dse），逐点增益集中在 3.9e3–1e4 Hz
       （+1.3 ~ +2.9）——这类「部分补偿」不能漏。

    为什么用 Lin-KK 的 `L_fit` 而不是 `Im Z/ω` 的中位数：后者只在
    Im Z > 0 的点上取（`estimate_L` 的取法），对判据②天然失效；
    `L_fit` 是 Lin-KK 拟合向量里的自由元素，Im 全负时仍可估。两条
    独立路径互验：本谱 L_fit=9.06 μH vs 加 L 后后验中位数 9.17 μH。
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    hi = np.argsort(f)[-3:]
    if bool(np.mean(Z.imag[hi] > 0) >= 0.5):
        return True
    if _kk_lin_kk is None:              # 没有 KK 模块：只能按判据①
        return False
    try:
        g = _kk_lin_kk(f, Z)
    except Exception:                   # noqa: BLE001
        return False
    L_fit = g.get("L_fit")
    if L_fit is None or not np.isfinite(L_fit) or L_fit <= 0:
        return False
    imax = int(np.argmax(f))
    share = 2 * np.pi * f[imax] * float(L_fit) / max(abs(Z[imax]), 1e-300)
    return bool(share >= L_SHARE_MIN)


# --------------------------------------------------------------------------
# DRT 去电感数据通道
# --------------------------------------------------------------------------
# **只影响 DRT 预判，不改 BM-DRT 的似然。** 两者必须分开：
#   · DRT 是 Tikhonov 非负反演，基底是 Σ γ(τ)/(1+jωτ)，**表示不了 jωL**。
#     有感性尾时它会在高频网格边缘长出一个伪峰去扛，把 K 候选撑大、
#     把峰位挤偏。
#   · BM-DRT 的似然必须用**原始 Z**，否则等于把预处理结论又喂回模型，
#     后验不再是"给定观测"的后验。
# 所以：Z_DRT = Z_raw − jωL 只喂 prescreen；fit_one 仍然收 Z_raw。
#
# 不能无条件扣。扣多了会在高频造出容性伪像（Im Z 变负），扣的位置不对
# 会把真实的容性高频弧一起削掉。下面四道检查都过才扣。

def _high_band_resid(res, f, Z, frac: float = 2.0 / 3.0) -> float:
    """DRT 残差在**高频段**的 RMS（按 log f 上 1/3 切）。

    用 DRT 自己的残差而不是另造一个平滑基准：DRT 就是下游消费者，
    "扣 L 有没有用"应该由它的拟合能力回答，不该由我们猜。

    **`drt()` 返回的 `resid` 已经是相对量**（`(Z_fit−Z)/|Z|`，见 drt.py:140），
    这里不能再除一次 `|Z|`。之前多除一次，在 |Z|≈0.03 Ω 的 2E 谱上把残差
    放大成 1917%，纯属自造。
    """
    lf = np.log10(np.asarray(f, float))
    m = lf >= np.quantile(lf, frac)
    if int(m.sum()) < 3:
        return float("nan")
    r = np.asarray(res["resid"], complex)[m]
    return float(np.sqrt(np.mean(np.abs(r) ** 2)))


def deinduce_for_drt(f, Z, L_est=None, n_hi: int = 5, min_share: float = 5e-3,
                     tol_stab: float = 0.35, min_gain: float = 0.20,
                     tol_neg: float = 0.02):
    """决定 DRT 预判用原始 `Z` 还是去电感后的 `Z_DRT = Z − jωL`。

    参数
    ----
    L_est    : 先验电感估计（H）。None 时内部用 `bm_kk.estimate_L` 再算一次。
    n_hi     : 用最高的几个点估 L。
    min_share: 感抗在 f_max 处占 |Z| 的最小比例。低于它扣了也白扣，
               直接跳过（避免为一个 0.1% 的尾巴冒过扣的风险）。
    tol_stab : `Im Z/ω` 在最高若干点上的**离散度**上限，取
               `(p90−p10)/median`。超过就说明高频尾不是一个干净的 ωL
               （混了容性弧或噪声），此时扣 L 不可信。
    min_gain : 要求去电感后高频段 DRT 相对残差至少降低这个比例才采用。
    tol_neg  : 扣除后最高频若干点的 `Im Z` 允许为负的幅度（占该点 |Z|）。
               超过说明 L 估大了、造出了容性伪像。

    返回 dict（全部可 JSON 序列化，`Z` 字段除外）：
        used / L / reason / checks / resid_high_raw / resid_high_drt
        edge_high_raw / edge_high_drt / share_max
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    out = dict(used=False, L=None, reason="", share_max=float("nan"),
               resid_high_raw=float("nan"), resid_high_drt=float("nan"),
               edge_high_raw=float("nan"), edge_high_drt=float("nan"),
               checks={})
    chk = out["checks"]

    if L_est is None and _kk_estimate_L is not None:
        try:
            L_est = _kk_estimate_L(f, Z, n_hi=n_hi)
        except Exception:                   # noqa: BLE001
            L_est = None
    if L_est is None or not np.isfinite(L_est) or L_est <= 0:
        out["reason"] = "无感抗点（高频 Im Z 不为正），DRT 直接用原始 Z"
        chk["has_L"] = False
        return out
    chk["has_L"] = True

    # ---- ① 稳定性：高频 Im Z/ω 是不是一个常数 ----
    hi = np.argsort(f)[-min(n_hi, f.size):]
    w = 2 * np.pi * f[hi]
    pos = Z.imag[hi] > 0
    if int(pos.sum()) < 3:
        out["reason"] = (f"高频只有 {int(pos.sum())} 个感抗点，不足以判稳定性，"
                         f"DRT 用原始 Z")
        chk["stable"] = False
        return out
    x = Z.imag[hi][pos] / w[pos]
    med = float(np.median(x))
    spread = float((np.percentile(x, 90) - np.percentile(x, 10)) / med) if med > 0 else np.inf
    chk.update(stable=bool(spread <= tol_stab), spread=spread, n_pos=int(pos.sum()))

    # ---- ② 占比：扣了到底有没有影响 ----
    share = float(w.max() * med / max(abs(Z[np.argmax(f)]), 1e-300))
    out["share_max"] = share
    chk["share_ok"] = bool(share >= min_share)

    if not chk["stable"]:
        out["reason"] = (f"高频 Im Z/ω 离散度 {spread:.0%} > {tol_stab:.0%}，"
                         f"不像干净的串联电感（可能混了容性弧或噪声）→ "
                         f"**保留原始 Z**，只标记电感干扰")
        return out
    if not chk["share_ok"]:
        out["reason"] = (f"感抗在 f_max 处仅占 |Z| 的 {share:.2%} < {min_share:.2%}，"
                         f"扣除无实质影响 → DRT 直接用原始 Z")
        return out

    # ---- ③ 过扣检查：扣完**最高频** Im Z 不能明显变负 ----
    # 只查最高 3 点，不能查整个 top-5：Im Z/ω 在**高频段下沿**本来就会偏低
    # （容性/弛豫贡献还没衰减完），那里扣完必然剩负值，那是物理不是过扣。
    # 过扣只可能出现在最高频——那里 ωL 占优，L 估大了就直接把 Im Z 压穿。
    # 实测 2EMC1-1_0005mA：top-5 最小 -4.78%（误报），top-3 是 0.00/+1.13/+1.49%。
    Zs = Z - 1j * (2 * np.pi * f) * med
    hi3 = np.argsort(f)[-min(3, f.size):]
    worst = float(np.min(Zs.imag[hi3] / np.maximum(np.abs(Z[hi3]), 1e-300)))
    chk["over_correct"] = bool(worst < -tol_neg)
    chk["worst_resid"] = worst
    if chk["over_correct"]:
        out["reason"] = (f"扣除后最高频 Im Z 降到 {worst:.2%}·|Z|（超过 -{tol_neg:.0%}），"
                         f"L 明显估大 → **保留原始 Z**")
        return out

    # ---- ④ 收益：DRT 自己的高频残差有没有变好 ----
    try:
        res_raw, _, _ = prescreen(f, Z)
        res_drt, _, _ = prescreen(f, Zs)
    except Exception as e:                  # noqa: BLE001
        out["reason"] = f"DRT 试算失败（{type(e).__name__}: {e}）→ 保留原始 Z"
        return out
    r_raw = _high_band_resid(res_raw, f, Z)
    r_drt = _high_band_resid(res_drt, f, Zs)
    e_raw = float(res_raw.get("edge_high_frac", float("nan")))
    e_drt = float(res_drt.get("edge_high_frac", float("nan")))
    out.update(resid_high_raw=r_raw, resid_high_drt=r_drt,
               edge_high_raw=e_raw, edge_high_drt=e_drt)
    gain = (r_raw - r_drt) / r_raw if np.isfinite(r_raw) and r_raw > 0 else 0.0
    chk["gain"] = float(gain)
    chk["gain_ok"] = bool(gain >= min_gain)

    if not chk["gain_ok"]:
        out["reason"] = (f"去电感后 DRT 高频残差只降 {gain:.0%}（< {min_gain:.0%}），"
                         f"收益不足 → DRT 用原始 Z")
        return out

    out.update(used=True, L=float(med),
               reason=(f"扣 jωL（L={med:.3g} H，ImZ/ω 离散度 {spread:.0%}，"
                       f"f_max 处占 |Z| {share:.1%}）后 DRT 高频残差 "
                       f"{r_raw:.2%}→{r_drt:.2%}（降 {gain:.0%}）"))
    return out


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


def _ad_line(r):
    """吸附 RL 摘要 + **按拐点位置分区间**的可辨识性降级说明（立项 §2.3）。

    与 `_diff_line` 同一套写法：窗口外两侧定住的是完全不同的组合，
    不能一句"可辨识"蒙过去。

      f_c ≳ f_max（τ 小）→ Z_ad ≈ jω·(R_ad·τ_ad)，与串联 L **完全简并**：
          只有 L_eff = R_ad·τ_ad 可信，R_ad 与 τ_ad 各自都定不住。
      f_min ≲ f_c ≲ f_max → 完整 RL 半圆在窗内，R_ad 与 τ_ad 都可辨识。
      f_c ≲ f_min（τ 大）→ Z_ad ≈ R_ad（纯实部），被 R_inf 吸收，环不可见。

    另加 M3 标定发现的简并警示：corr(R_ad, R_inf) 强负时（环宽且居中），
    常数项与欧姆电阻沿补偿方向滑动，只能报"存在低频感性弛豫"。
    """
    s = r.get("s") or {}
    if "Rad" not in s or "tau_ad" not in s:
        return "（无参数）"
    out = []
    ra = np.asarray(s["Rad"], float)
    ta = np.asarray(s["tau_ad"], float)
    mra, mta = float(np.median(ra)), float(np.median(ta))
    f_c = 1.0 / (2 * np.pi * mta) if mta > 0 else float("nan")
    f_lo, f_hi = float(r.get("f_lo") or 0.0), float(r.get("f_hi") or 1e300)

    out.append(f"R_ad = {mra:.4g} Ω "
               f"[{np.percentile(ra, 5):.4g}, {np.percentile(ra, 95):.4g}]")
    out.append(f"τ_ad = {mta:.4g} s → f_c = {f_c:.3g} Hz")
    out.append(f"L_ad = R_ad·τ_ad = {mra * mta:.4g} H（派生量，只报告不采样）")

    # 拐点出窗概率（fit_one 已算），以及它对应的降级措辞
    p_out = r.get("f_c_out_prob")
    if f_c > f_hi * 10:
        out.append(f"** f_c ≫ f_max：Z_ad ≈ jωL_eff，与串联电感完全简并——"
                   f"只有 L_eff = {mra * mta:.4g} H 可信，"
                   f"R_ad 与 τ_ad 单独都不可信 **")
    elif f_c < f_lo / 10:
        out.append(f"** f_c ≪ f_min：Z_ad ≈ R_ad（纯实部），已被 R_inf 吸收——"
                   f"环在窗内不可见，R_ad 的绝对值没有意义 **")
    elif np.isfinite(p_out) and p_out > 0.5:
        out.append(f"** P(f_c > f_max) = {p_out:.2f} > 0.5：拐点大概率在窗口上方，"
                   f"按「只有 L_eff 可信」处理 **")
    elif f_lo <= f_c <= f_hi:
        out.append("f_c 落在窗口内 → R_ad 与 τ_ad 都可辨识"
                   "（完整 RL 半圆在窗内）")
    else:
        out.append(f"f_c={f_c:.3g} Hz 在窗口边缘 1 dec 内 → 可辨识性介于"
                   f"两渐近区之间，引用 R_ad 要带不确定度")

    # 幅值相对噪声（v1.1 口径：按 σ̂，不是 |Z|max）
    p_small = r.get("p_ad_small")
    if np.isfinite(p_small):
        out.append(f"P(R_ad < 3σ̂) = {p_small:.2f}"
                   f"{'（幅值与噪声同量级，环可能不显著）' if p_small > 0.3 else ''}")

    # R_ad–R_inf 简并（M3 标定）：从 param_corr 或样本直接算
    rho = None
    pc = r.get("param_corr") or {}
    for item in list(pc.get("strong") or []) + list(pc.get("medium") or []):
        if {item.get("a"), item.get("b")} == {"R_ad", "R_inf"} \
                and _finite(item.get("rho")):
            rho = float(item["rho"])
            break
    if rho is None and "R_inf" in s and ra.size >= 30:
        ri = np.asarray(s["R_inf"], float)
        if np.std(ra) > 0 and np.std(ri) > 0:
            rho = float(np.corrcoef(np.log(np.maximum(ra, 1e-300)),
                                    np.log(np.maximum(ri, 1e-300)))[0, 1])
    if rho is not None:
        out.append(f"corr(R_ad, R_inf) = {rho:+.3f}"
                   + ("（**强负：常数项与欧姆电阻简并，只报「存在低频感性弛豫」，"
                      "不报 R_ad 单独值**）" if rho <= -0.8 else ""))
    return "；".join(out)


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


def loglik_matrix(s, f, Z, include_L, n_max: int = 800, sigma=None):
    """逐样本、**逐频率点**的对数似然，(n_draw, n_freq)。

    sigma：固定噪声路径（`bm_core.fit(sigma=(s_re, s_im))`，σ 不进采样）下
    **必须**由调用方显式传入的噪声标准差，标量或 (s_re, s_im) 元组均可。
    缺省时优先用样本里的 `s_re`/`s_im`（主流程 σ 进采样，样本里一定有）；
    两者都没有时**直接 raise**，绝不回退到任何默认值——回退会把错误口径
    写进 elpd / Pareto-k / 模型比较且不报错（一轮审查 P1-1 曾回退 1e-3，
    二轮审查 P1-1 证实 0.2 同样失真且回归测试因锚值撞上回退值而失效）。

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
    # 解析调用方给的 σ：标量 / (s_re, s_im) / None（走样本或默认）
    if sigma is not None:
        if np.ndim(sigma) == 0:
            _sr = _si = float(sigma)
        else:
            _sr, _si = float(sigma[0]), float(sigma[1])
    else:
        _sr = _si = None
    ll = np.empty((idx.size, f.size))
    for a, i in enumerate(idx):
        z = B.model_z(f, _params(s, i, include_L), include_L=include_L)
        if _sr is not None:                       # 调用方显式指定，最可信
            sr, si = _sr, _si
        elif "s_re" in s and "s_im" in s:         # 主流程：σ 进采样
            sr = float(s["s_re"][i])
            si = float(s["s_im"][i])
        else:
            # 固定噪声（bm_core.fit(sigma=...)）路径：σ 不在样本里，调用方
            # 必须显式传入。静默回退默认值会把错误口径写进 elpd / Pareto-k /
            # select_k 且无任何报错（审查 P1-1，二轮）——宁可拒绝计算，
            # 也不给一个看似合理的数。
            raise ValueError(
                "loglik_matrix：样本里没有 s_re/s_im（固定 σ 路径），且调用方"
                "未传 sigma。请在调用时显式传入本次拟合实际使用的 σ"
                "（标量或 (s_re, s_im) 元组）。")
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
    return (idata, float(r.elpd_loo), float(r.se),
            float(np.mean(pk > 0.7)), pt, pk)


def compare_models(res_list):
    """配对比较：elpd_diff 与 dse 都基于**逐点差**（arviz.compare）。

    为什么不能只比 elpd 的绝对值：两个模型的 elpd 各有 ±5 的不确定度，
    差 3 分根本分不出高下。arviz 的 `dse` 是逐点差的标准误
    `sqrt(n·Var(逐点差))`，它通常远小于两个模型各自 se 的平方和开方
    ——因为两个模型在同一批点上犯的错高度相关。

    返回 DataFrame（按 elpd 降序），并把 elpd_diff/dse/weight 写回每个结果。

    **先剔除 LOO 不可用的候选**（idata=None 或 elpd 非有限）：arviz 的
    `compare` 在 ELPD 计算里遇到 None 会抛异常，把整条谱连同已算好的 K
    一起报废（审查 P1-2）。被剔除的候选标 `compare_note` 并置 NaN，
    由 `select_k` 的「LOO 不可用」分支统一处理，而不是连坐。
    """
    import arviz as az
    ok = [r for r in res_list
          if r.get("idata") is not None and np.isfinite(r.get("elpd", np.nan))]
    ok_ids = {id(r) for r in ok}

    def _cmp_key(r):
        # P3（二轮审查）：key 原来只含 K —— 同 K 不同 diffusion/adsorption
        # 的候选会在 pack 里互相覆盖。当前主链路 K 唯一，属公开 API 风险。
        k = f"K={r['K']}"
        if r.get("diffusion"):
            k += f"+{r['diffusion']}"
        if r.get("adsorption") == "RL":
            k += "+ad"
        return k

    for r in res_list:
        if id(r) in ok_ids:
            continue
        # P2-1（二轮审查）：未参与比较的候选写 None，不写 NaN/0 冒充真值。
        r["elpd_diff"] = None
        r["dse"] = None
        r["weight"] = None
        r["rank"] = None
        r["comparison_performed"] = False
        r["compare_note"] = "LOO 不可用，未参与 az.compare"
    if not ok:                                    # 全部 LOO 失败：返回空表
        import pandas as pd
        return pd.DataFrame()
    pack = {_cmp_key(r): r["idata"] for r in ok}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = az.compare(pack, ic="loo", method="stacking", scale="log")
    for r in ok:
        key = _cmp_key(r)
        r["elpd_diff"] = float(df.loc[key, "elpd_diff"])
        r["dse"] = float(df.loc[key, "dse"])
        r["weight"] = float(df.loc[key, "weight"])
        r["rank"] = int(df.loc[key, "rank"])
        r["comparison_performed"] = True
    return df

# ---- 采用补跑家族结果时的字段保留（审查 P2-5，二轮） ----
def _adopt_family_result(best, cand, note_cand, note_prev):
    """Ws / ad 补跑家族被采用时，**只替换模型结果字段**，K 选择状态与 note
    必须继承 none 家族的完整 K 扫描。

    旧实现 `best, note = cand, note_cand` 整体替换：补跑只固定 K 跑了
    一个模型，其 `selection_status` 是 "not_compared"、elpd_diff/rank 是
    未比较默认值——证据卡 `K_STATUS_CLASS`（bm_evidence.py）据此把一条
    完整扫描过的谱误判成「只拟合了一个 K」，none 家族的判读理由也全部丢失。
    """
    saved_status = best.get("selection_status")
    best = cand
    if saved_status is not None:
        best["selection_status"] = saved_status
    best["comparison_performed"] = False
    best["k_compare_note"] = ("K 选择与跨 K 配对比较在 none 家族完成；本结果"
                              "是固定 K 补跑，elpd_diff/rank 不适用（None）。"
                              "与 none 家族的配对差见 auto_mechanism/ad_selection")
    return best, list(note_cand) + list(note_prev)


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


def init_values(pk, res, K, zeta_fixed=1.0, ad_init=None):
    """DRT 峰 → 模型初值（约束空间）。K 与峰数不一致时先 adjust_peaks。

    ad_init：`bm_ad.inv_from_lowfreq` 的返回（M4 接线）。只有显式传入且
    `init_reliable=True` 才播种 Rad/tau_ad —— 不可靠时宁可从先验中位数
    起步（§7.3：初值只影响采样效率，不影响后验）。
    """
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
    if ad_init and ad_init.get("init_reliable"):
        _ra, _ta = ad_init.get("Rad"), ad_init.get("tau_ad")
        if _ra is not None and _ta is not None and _ra > 0 and _ta > 0:
            d["Rad"] = float(_ra)
            d["tau_ad"] = float(_ta)
    return d


# --------------------------------------------------------------------------
# 单个 K 的拟合
# --------------------------------------------------------------------------

def _resolve_seed(seed, kw):
    """决定这一次拟合实际用哪个 sampler seed。

    优先级（**显式 > 隐式 > 模块默认**）：

    1. `seed` 实参非 `None` → 用它（调用方明确要求）；
    2. 否则 `kw["seed"]` 存在 → 用它（配置/上层传入）；
    3. 否则 `KW["seed"]`（= 0，与改动前逐位一致）。

    为什么要有这个函数：改动前 `fit_one` 写的是 `kw["seed"] = seed`（`seed` 默认 0），
    于是「往 kw 里塞 seed」会被**静默覆盖**成 0——参数收到了却不生效，比报错更难查。
    现在把解析规则抽出来单测，不让它藏在函数体中间。
    """
    if seed is not None:
        return int(seed)
    if kw is not None and kw.get("seed") is not None:
        return int(kw["seed"])
    return int(KW["seed"])


def fit_one(f, Z, K, include_L, zeta, init=None, seed=None, kw=None, diffusion=None,
            diag=None, sigma_floor=0.0, adsorption=None):
    """跑一次 NUTS，返回全部指标（不打印、不画图）。

    seed：sampler 随机流种子。`None`（默认）= 沿用 `kw["seed"]`，`kw` 里也没有
    就用 `KW["seed"]`(=0)。显式传值则覆盖 `kw`——规则见 `_resolve_seed()`。

    diag：事后诊断阈值（corr_strong / corr_medium / diff_resolution_tol），
    由 config.json 的 diag 段传入；缺省时用模块内置默认值。

    sigma_floor：噪声地板（相对 zmax）。0.0 = 关闭，行为与改动前一致。
    adsorption：None = 基础模型（逐位不变）；"RL" = 采样吸附元件（M2/M4）。
    """
    diag = diag or {}
    kw = dict(KW) if kw is None else dict(kw)
    seed_used = _resolve_seed(seed, kw)
    kw["seed"] = seed_used
    t0 = time.time()
    s, mc = B.fit(f, Z, K=K, alpha=0.5, sigma=None, include_L=include_L,
                  zeta_fixed=(None if zeta == "free" else float(zeta)),
                  init_values=init, diffusion=diffusion,
                  sigma_floor=float(sigma_floor or 0.0),
                  adsorption=adsorption, **kw)
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
    #   p_peak_out : P(f_peak ∉ [f_lo, f_hi])   —— 物理峰位口径（现只用于组分审查）
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

    # ---- 吸附 RL 派生量（M4；未采样时全 NaN，默认路径零影响）----
    p_ad_small, f_c_out_prob, L_ad_med = float("nan"), float("nan"), float("nan")
    if adsorption == "RL" and "Rad" in s and "tau_ad" in s:
        _ra = np.asarray(s["Rad"], float)
        _ta = np.asarray(s["tau_ad"], float)
        # σ̂ 取该模型自采的虚部噪声（v1.1 口径：按噪声尺度，不是 |Z|max；
        # none 的残差 std 在这里拿不到，自采 σ 是同一量纲的自然替代）
        _sig = float(np.median(s["s_im"])) if "s_im" in s else float("nan")
        if np.isfinite(_sig) and _sig > 0:
            p_ad_small = float(np.mean(_ra < 3.0 * _sig))
        f_c_out_prob = float(np.mean(1.0 / (2 * np.pi * _ta) > float(f.max())))
        L_ad_med = float(np.median(_ra * _ta))       # L_ad = R_ad·τ_ad，派生量

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

    # ---- 多分量拼 45° 段：逐分量判据**看不到**的路径 ----
    # 实测 4382（真值 Rs_Ws）：两分量 φζ=0.502/0.501，但峰位都在窗内、
    # 无 R-τ 简并，looks_like_warburg() 判 no。这时模型是用「多个 φ≈0.5
    # 的分量拼出 45° 段」来逼近扩散尾，逐分量判据的前提不成立。
    # 补一条查**合成阻抗斜率**的弱识别，只提示、不改 K 选择。
    # P2-9（二轮审查）：「检查没跑成」用 not_evaluable，不用 "no"——
    # "no" 是阴性结论（检查跑了、没发现），失败伪装成阴性会误导判读。
    wmul = dict(level="not_evaluable", why=["多分量扩散检查未运行（前置条件不满足）"],
                band=None)
    if DIFF is not None and K >= 2:
        try:
            n_s = len(s["R_inf"])
            ii = np.linspace(0, n_s - 1, min(100, n_s)).astype(int)
            Zm = np.stack([B.model_z(f, _params(s, i, include_L), include_L=include_L)
                           for i in ii])
            band = DIFF.warburg_band_slope(f, Zm)
            lvl, why = DIFF.warburg_multicomponent(sig, band)
            wmul = dict(level=lvl, why=list(why), band=band)
        except Exception as e:              # noqa: BLE001  诊断不该让主流程崩
            wmul = dict(level="not_evaluable",
                        why=[f"多分量扩散检查失败：{type(e).__name__}"],
                        band=None)

    # ---- 模型选择用哪个口径：**物理峰位** ----
    # 理由：硬筛选的目的是剔除"峰跑到窗口外、面积不可信"的分量，而"峰"在
    # 物理上指 DRT 峰，不是参数 τ。φζ>1 时峰位没有物理意义（解析 γ 有负值），
    # 那些分量退回参数口径，并在报告里标 nonphysical。
    p_peak_out = np.full(K, np.nan)
    for k, sk in enumerate(sig):
        p_peak_out[k] = sk.get("p_peak_out", np.nan)

    # **展示字段**与**模型选择字段**必须分开：
    #   p_out          —— 兼容旧接口/给人看的统一出口，峰位不可用时退回 τ
    #   p_peak_out_max —— 只给 component_review 用，不再参与 K 硬筛选； NaN 记为 uncertain，**绝不回退**
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
    # 按 f 网格分键：watch/auto 在同一进程连续处理不同网格的谱时，不能把
    # 第一条谱的零分布阈值套到第二条上（审查 P2-1）。
    # 外部仍可能把 NULL_CACHE 重置成 None（bmdrt.py / test_ecm.py），
    # 这里兜住：非 dict 就重建，避免 'NoneType' has no attribute 'get'。
    _cache = NULL_CACHE
    if not isinstance(_cache, dict):
        _cache = {}
        globals()["NULL_CACHE"] = _cache
    _nkey = f.tobytes()
    null = _cache.get(_nkey)
    if null is None:
        null = null_thresholds(f)
        _cache[_nkey] = null

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
    _loo_error = None
    try:
        idata, elpd, elpd_se, bad_k, pt, pk = loo_pointwise(ll)
    except Exception as e:
        idata, elpd, elpd_se, bad_k, pt, pk = (None, float("nan"),
                                               float("nan"), float("nan"),
                                               None, None)
        # P3（二轮审查）：失败原因只 print 不落盘，事后只见 elpd=NaN
        # 无从追查。写进结果 dict，随 JSON 落盘。
        _loo_error = f"{type(e).__name__}: {e}"
        print(f"    （LOO 失败：{_loo_error}）")

    r = dict(
        K=K, s=s, mc=mc, rows=rows, zd=zd, ndiv=ndiv, tds=tds, t=dt,
        loo_error=_loo_error,
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
        warburg_multi=wmul,
        diffusion=diffusion, param_corr=pc, diffusion_resolution=dres,
        adsorption=adsorption, p_ad_small=p_ad_small,
        f_c_out_prob=f_c_out_prob, L_ad=L_ad_med,
        f_lo=float(f.min()), f_hi=float(f.max()), n_pts=int(f.size),
        zmax_obs=zmax_obs, p_diff_small=p_diff_small, diff_amp_key=diff_amp_key,
        idata=idata, elpd=elpd, elpd_se=elpd_se, bad_k=bad_k,
        pointwise_elpd=pt, pareto_k=pk,
        elpd_diff=None, dse=None, weight=None, rank=None,
        comparison_performed=False,
        rhat_max=max(x["rhat"] for x in rows),
        tail_min=min(x["tail"] for x in rows),
        bulk_min=min(x["bulk"] for x in rows),
        seeded=init is not None,
        # 实际生效的采样配置（不是命令行请求值）：B3a/B3b 的证据就靠这两个字段。
        # seed 取 `seed_used`（真传给 mcmc.run 的那个），不取外层入参。
        seed=int(seed_used),
        sampling=dict(n_warmup=int(kw.get("n_warmup", 0)),
                      n_samples=int(kw.get("n_samples", 0)),
                      num_chains=int(kw.get("num_chains", 0)),
                      target_accept=float(kw.get("target_accept", float("nan"))),
                      max_tree_depth=int(kw.get("max_tree_depth", 0))),
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


def select_k(res_list, exploratory: bool = False):
    """在多个 K 之间选择。四条规则，全部可复核。

    exploratory=True（quick 档）时，最终状态一律改写成 `exploratory`：
    quick 只有 800 warmup / 400 采样 / 2 链，而严格线要求 `tail ESS>400`、
    `触顶<1%` —— **这两套设置不在同一个采样预算尺度上**。此时出现
    `rejected` 只说明采样不够，不说明模型被统计证据否定。用单独的状态名
    把这两件事分开，避免把"预算不足"读成"模型不合格"。

    状态取值：exploratory | converged | degraded | rejected

    1) **硬性剔除**（任一触发即淘汰）——只管「这个候选能否可靠参与比较」：
       · 采样分级 0（R̂/ESS/divergence/触顶 任一越线）
       · `frac_neg > bm_diff.NONPHYS_TOL`（φζ>1 的样本占比超容忍上限）
       · LOO 近似不可靠（Pareto-k>0.7 超 10%）、LOO 不可用
       **注意：`P(峰位出窗)` 已从硬筛选中彻底移出**（2026-11-06 审查意见）。
       峰位出窗是「组分的物理可辨识性」问题，不是「模型预测能力」问题：
       一个分量峰位在窗口外，仍可能对窗口内阻抗有显著贡献。
       它改由拟合后的 `component_review()` 单独报告，不淘汰 K。
    2) 剩下的按 **elpd_loo 配对比较**排序，取最好。
    3) **简约优先**：若某个更小的 K 与最好者满足 `deficit < max(2, dse)`
       （差值落在不确定度内），取更小的 K。
    4) 若全部被剔除，仍返回一个 dict（下游 report/plot/save_out 需要它），
       但打上 `selection_status` 并在 note 首条写明「不输出最优 K 的结论」，
       **绝不把「相对最好」包装成「最优」**：
         converged  有 K 过完整硬筛选（旧结果文件里写作 "ok"）
         degraded   只降到「可用」档（分级≥1 但被物理/LOO 规则剔除）
         rejected   连「可用」都没有，结果只是相对排序、无物理意义
         exploratory quick 档专属，见函数开头说明

    峰位口径说明：`p_out` 是兼容展示字段（峰位不可用会退回 τ 口径），
    `p_peak_out` 才是物理峰位口径。两者现在都**只用于组分审查**，
    不再参与 K 的硬筛选；峰位诊断不可用（NaN）在组分状态里记为 `uncertain`。

    注意：这里的「最好」只在**扫过的 K 范围**内成立，不是全局最优。
    """
    note, ok = [], []
    tol = getattr(DIFF, "NONPHYS_TOL", 0.01) if DIFF is not None else 0.01
    for r in res_list:
        hard = []                                    # 直接剔除的理由
        if r["grade"] == 0:
            hard += r["grade_loose"]

        # 峰位出窗不再淘汰 K；只记录，交给 component_review() 做组分审查。
        ppm = r.get("p_peak_out_max", float("nan"))
        r["peak_out_for_review"] = (float(ppm) if np.isfinite(ppm) else None)

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
    # 四档命名按审查意见统一：converged / degraded / rejected / exploratory。
    # 旧结果文件里的 "ok" 视为 converged 的别名，读的时候不用改。
    status = "converged" if ok else ("degraded" if usable else "rejected")
    if not ok:
        note.append("严格标准下没有 K 全过（分级≥1 但被物理有效性/LOO "
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
        # P2-1（二轮审查）：未比较的候选 elpd_diff=None（不再伪装成 0.0）。
        _ed = r.get("elpd_diff")
        return abs(_ed) if _ed is not None and np.isfinite(_ed) else np.inf

    close = [r for r in pool
             if r["K"] < best["K"] and r.get("comparison_performed")
             and deficit(r) < max(2.0, r["dse"])]
    if close:
        small = min(close, key=lambda r: r["K"])
        note.append(f"K={best['K']} 与 K={small['K']} 的 elpd 差 "
                    f"{deficit(small):.1f} ± {small['dse']:.1f}，"
                    f"小于 max(2, dse) → 按简约优先取 K={small['K']}")
        best = small
    if exploratory:
        # quick 档：采样预算与严格线不匹配，把"预算不足"和"模型被否定"分开。
        note.insert(0, "**exploratory（quick 档）**：本次采样预算（800/400/2 链）"
                       "撑不起严格收敛线，所以下面的采样分级与 elpd 排名都不能"
                       "当作模型选择结论。要判 K，请用正式档重跑。")
        status = "exploratory"
    best["selection_status"] = status
    return best, note


# --------------------------------------------------------------------------
# 扩散家族的外层选择（独立于 select_k）
# --------------------------------------------------------------------------
# 设计边界（2026-11-06）：
#   · `select_k()` 只做「给定 diffusion 配置下的 K 选择」，语义与规则不动；
#   · 「要不要加 W」是**外层**决策，由下面的 `diffusion_gate` +
#     `select_mechanism` 完成，输出 none / Ws / uncertain；
#   · 默认只比 none vs Ws；Wo 保留手动对照（频窗不足时 Ws/Wo 不可分辨）。

def _finite(x):
    """None / NaN / inf 都返回 False；避免 np.isfinite(None) 抛 TypeError。"""
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _paired_loo_diff(a, b):
    """两个模型逐点 LOO 的配对差（b − a），用于跨扩散家族比较。

    口径与 `pointwise_loo.py` 一致：Δ = Σd_i、dse = √(N·Var(d_i))。
    **不能**直接拿 `select_k()` 里的 `elpd_diff/dse`：那是 az.compare
    相对全局最优的口径，跨家族时基准不同，且不是配对差。
    """
    pa, pb = a.get("pointwise_elpd"), b.get("pointwise_elpd")
    if pa is None or pb is None:
        return None
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    if pa.shape != pb.shape or pa.size < 2:
        return None
    if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
        return None
    d = pb - pa
    return dict(delta=float(np.sum(d)),
                dse=float(np.sqrt(d.size * np.var(d, ddof=1))),
                n=int(d.size))


def _drt_evidence(pres, pk):
    """把 DRT 预判里与低频弛豫有关的量摘出来，作为可复核证据。

    **只记录，不参与决策**：低频宽峰既可能是窗口外扩散，也可能是阻截电极
    或未闭合弧；算法能说的是「这里有一个多宽的峰、占多少面积、端部有没有
    堆积」，不能直接说「是 Warburg」。
    """
    if not pres or not pk:
        return None
    rt = float(pres.get("R_total") or 0.0)
    peaks = [dict(f_peak_hz=float(p["f_peak_Hz"]),
                  r_ohm=float(p["R_ohm"]),
                  width_dec=float(p["width_dec"]),
                  area_frac=(float(p["R_ohm"]) / rt if rt > 0 else None))
             for p in pk]
    low = min(peaks, key=lambda p: p["f_peak_hz"]) if peaks else None
    edge = pres.get("edge_low_frac")
    return dict(n_peaks=len(peaks), lowest_freq_peak=low,
                edge_low_frac=(float(edge) if edge is not None else None),
                note="低频峰宽/面积/端部堆积是候选证据，不能单独判 Warburg")


def _none_needs_ws(res, acf_thr: float = 0.8, low_snr_thr: float = 1.5):
    """拟合后门控：none 的残差是否还有「加 W 能救」的系统性。

    比 45° 指纹可靠，且**零额外拟合成本**——用的都是 none 已经算出来的量：
      · 残差 lag-1 自相关（`acf`）：平滑系统性偏差 → 高；
      · 最低频段 rms/σ：低频偏离超过噪声 1.5 倍 → 值得补跑 Ws。

    标定（117 条 2E）：acf>0.8 占 3%、低频 rms/σ>1.5 占 25%，组合 28%。
    目标谱（无 W）acf=0.38、低频 1.05 → 跳过；模拟 Ws 谱 acf=0.86 → 触发。
    """
    acf = abs(float(res.get("acf") or 0.0))
    # 兼容两种来源：`fit_one` 内部用 s_re/s_im，落盘 JSON 用 sigma_re/sigma_im。
    s_re = float(res.get("s_re", res.get("sigma_re", 0.0)) or 0.0)
    s_im = float(res.get("s_im", res.get("sigma_im", 0.0)) or 0.0)
    # 同样兼容：内部叫 bins，落盘 JSON 叫 resid_bins。
    bins = res.get("bins") or res.get("resid_bins") or []
    low = 0.0
    if bins and s_re > 0 and s_im > 0:
        b = bins[0]
        low = max(float(b.get("rms_re", 0.0)) / s_re,
                  float(b.get("rms_im", 0.0)) / s_im)
    return (acf > acf_thr or low > low_snr_thr,
            dict(acf=acf, low_rms_over_sigma=low,
                 acf_thr=acf_thr, low_snr_thr=low_snr_thr))


def diffusion_gate(kk, none_best, pres=None, pk=None, f=None, Z=None):
    """门控：只决定「要不要补跑 Ws」，不做模型比较、不改 K。

    证据顺序（弱 → 强）：
      1. KK 低频尾 45° 候选（`bm_kk.diffusion_tail_candidate`，必要不充分）；
      2. **数据侧端点走向**（`bm_ws.ws_diagnostics`）：`to_inductive` 直接登记成
         感抗尾（三类扩散尾都要求 `-Im>0`，符号不符即否证）；`to_cap` /
         `to_plateau` 只记录家族，不改门控结论；
      3. none 拟合后的 `n_warburg_like`（HN 分量是否已在替扩散）。

    `n_warburg_like > 0` 时**不进入**比较：这是「简并」警告，不是加 W 的
    理由——加 Ws 只会和宽 HN 抢同一段自由度，Δelpd 通常很小甚至为负。
    自动层应输出 uncertain，建议人工对照。

    `f` / `Z` 缺省时跳过第 2 条，行为与 2026-11-09 之前逐位一致。
    """
    tail = ((kk or {}).get("tail") or {})
    nw = int(none_best.get("n_warburg_like", 0) or 0)
    ws_diag = None
    if WS is not None and f is not None and Z is not None:
        try:
            ws_diag = WS.ws_diagnostics(f, Z)
        except Exception:                   # noqa: BLE001
            ws_diag = None
    g = dict(enter=False, status="no_candidate", reason="",
             kk_tail=dict(is_candidate=bool(tail.get("is_candidate")),
                          slope_ny=tail.get("slope_ny"),
                          slope_z=tail.get("slope_z")),
             drt=_drt_evidence(pres, pk),
             ws_diag=ws_diag,
             n_warburg_like=nw)
    if nw > 0:
        g.update(status="degenerate",
                 reason=f"none 的 HN 已有 {nw} 个分量呈 Warburg 退化指纹；"
                        f"加 Ws 与宽 HN 不可分辨，不自动切换，建议人工对照")
        return g
    if ws_diag and ws_diag.get("term") == "to_inductive":
        g.update(status="inductive_tail",
                 reason="低频 `-Im<0`（感抗尾，数据侧判据）：三类扩散尾都要求"
                        "`-Im>0` 且按 f^(-1/2)~f^(-1) 增长，符号不符即否证，"
                        "不用再看 45° 段长；该特征走吸附支路（R∥L）")
        return g
    if not tail.get("is_candidate"):
        g.update(status="no_candidate",
                 reason="KK 低频尾不是 45° 候选（必要不充分），不触发 Ws")
        return g
    g.update(enter=True, status="candidate",
             reason="KK 低频尾 45° 候选，进入 none vs Ws 比较")
    return g


def select_mechanism(none_res, ws_res, gate, exploratory=False, share=None):
    """外层选择器（公开入口）：none vs Ws。

    只包一层 `bm_grade.annotate`——给结果贴**证据资格档**（A/B/C），
    不改任何判据、不改 `status` / `reason` / `guards` / `choice` / `delta`。
    选择规则全在 `_select_mechanism_impl()` 里；要改规则去改那里，别在这里改。
    """
    return GRADE.annotate(_select_mechanism_impl(none_res, ws_res, gate,
                                                 exploratory=exploratory,
                                                 share=share))


def _select_mechanism_impl(none_res, ws_res, gate, exploratory=False, share=None):
    """外层选择器：none vs Ws。**不修改 `select_k()` 的语义**。

    规则（取严，全部可复核）：
      · 切换门槛 `Δ ≥ max(2, 1.5·dse)`；
      · 物理护栏（Ws 采样 / Pareto-k / 幅值不可辨识 / HN 简并）触发 → 不切换；
      · `Δ < 0`（Ws 更差）→ none；
      · `0 ≤ Δ < 门槛` → uncertain（**合法输出**，不硬选）。

    护栏：
      · Ws 采样分级 0 / Pareto-k>0.7 超 10% / LOO 不可用；
      · `p_diff_small > 0.5`（扩散幅值大量 <1e-3·|Z|max，不可辨识）。

    `diffusion_resolution.resolvable=False` **只记录、不降级**：它判的是
    Ws 与 Wo 的可分辨性，不是「加不加 W」。默认只选 Ws 时用它一票否决，
    会把大量本来能加扩散的谱打成 uncertain；要判 Wo 时才需要这条。

    `share`（`bm_ws.ws_share_low` 的结果）里有两个量，分工不同：

      · `cv_aw = sd(A_W)/median(A_W)` **参与决策**（guard
        `ws_amp_unidentified`）：超过 `bm_ws.CV_MAX` 说明扩散幅值没被数据
        定住，加 Ws 也说不清 `A_W`，不切换；
      · `share_med`（`Re Z_Ws(f_min)` 占份额）**只报表**：它随窗口下限
        漂移（同一条 `A_W=2` 的谱 0.01 Hz 窗 0.161、0.1 Hz 窗 0.057），
        而且真 Ws（`A_W=0.5`、0.01 Hz 窗）只有 0.048 —— 5% 阈值会误杀。

    标定见 `evidence/ws_crit/L3_CALIB.md`。`cv` 只挡"定不住"，不挡
    "元件选错"：真 Wo 用 Ws 拟合时 `cv` 只有 0.010–0.038，那种情况归
    `Δ` 与 `diffusion_resolution.resolvable`。
    """
    out = dict(choice="none", status="not_tested", reason="", gate=gate,
               delta=None, dse=None, threshold=None, guards={}, note=[],
               share=share)
    if ws_res is None:
        out.update(status="fit_missing", reason="Ws 未拟合，保持 none")
        return out

    guards = {
        "ws_grade0": ws_res.get("grade") == 0,
        "ws_bad_k": float(ws_res.get("bad_k") or 0) > 0.10,
        "ws_loo_unavailable": not _finite(ws_res.get("elpd")),
        "diff_amp_unidentifiable": (_finite(ws_res.get("p_diff_small"))
                                    and float(ws_res["p_diff_small"]) > 0.5),
        # 简并：none 的 HN 已经在替扩散。即使 Δ 显著也不自动切，
        # 因为这是「Ws 与宽 HN 抢同一段自由度」的结构问题，应人工对照。
        "hn_degenerate": gate.get("status") == "degenerate",
        # L3 简约闸（2026-11-09 标定）：A_W 后验相对不确定度超线 → 幅值没被
        # 数据定住。与 diff_amp_unidentifiable 覆盖不同侧面：那条比的是
        # 「相对 |Z|max 的大小」，这条比的是「后验收得多紧」。
        "ws_amp_unidentified": bool(share and share.get("identifiable") is False),
    }
    out["guards"] = guards
    dres = ws_res.get("diffusion_resolution") or {}
    out["ws_resolvable"] = dres.get("resolvable")
    if share and np.isfinite(share.get("cv_aw", float("nan"))):
        out["note"].append(
            f"A_W = {share['A_W']:.3g} ± {share['A_W_sd']:.2g}"
            f"（cv={share['cv_aw']:.3f}，判据线 {WS.CV_MAX:.2f} → "
            f"{'可辨识' if share['identifiable'] else '不可辨识'}；"
            f"等效 SNR≈{share['snr_med']:.1f}）"
            if np.isfinite(share.get("snr_med", float("nan")))
            else f"A_W = {share['A_W']:.3g} ± {share['A_W_sd']:.2g}"
                 f"（cv={share['cv_aw']:.3f}，判据线 {WS.CV_MAX:.2f}）")
    if share and np.isfinite(share.get("share_med", float("nan"))):
        out["note"].append(
            f"份额诊断（只报表，不作闸门）：Ws 在 f_min={share['f_min']:.3g} Hz "
            f"的 Re 份额中位 {share['share_med']:.3f}（90% CI "
            f"[{share['share_lo']:.3f}, {share['share_hi']:.3f}]）。"
            f"该量随窗口下限漂移，且真 Ws 在弱幅值窄窗可低至 0.048 → "
            f"5% 阈值会误杀，故不参与决策")
    if dres and dres.get("resolvable") is False:
        out["note"].append("Ws/Wo 在当前频窗不可分辨：可以加 Ws，但不能据此"
                           "宣称有限长扩散（Wo）；要判 Wo 需更宽频窗")

    cmp = _paired_loo_diff(none_res, ws_res)
    if cmp is None:
        out.update(status="comparison_incomplete",
                   reason="两个家族没有可配对的逐点 LOO，保持 none")
        return out
    out.update(delta=cmp["delta"], dse=cmp["dse"], n=cmp["n"])
    thr = max(2.0, 1.5 * cmp["dse"])
    out["threshold"] = thr

    if guards.get("hn_degenerate"):
        out.update(choice="none", status="degenerate",
                   reason="none 的 HN 已呈 Warburg 退化指纹，Ws 与宽 HN 不可分辨；"
                          "即使 Δ 显著也建议人工对照，不自动切换")
        return out
    if any(guards.values()):
        bad = [k for k, v in guards.items() if v]
        why = []
        if guards.get("diff_amp_unidentifiable"):
            why.append("Ws 幅值大量 <1e-3·|Z|max（谱图无 45° 拖尾 / 扩散被抑制），"
                       "强行加 W 只会让 Aw 不可辨识、参数失去物理意义")
        if guards.get("ws_amp_unidentified"):
            cv = float(share.get("cv_aw")) if share else float("nan")
            why.append(f"A_W 后验相对不确定度 cv={cv:.2f} > "
                       f"bm_ws.CV_MAX={WS.CV_MAX:.2f}（≈SNR {0.40 / max(cv, 1e-9):.1f}）："
                       f"扩散幅值没被当前窗口/噪声定住，加了也说不清 A_W")
        if guards.get("ws_grade0") or guards.get("ws_bad_k") \
                or guards.get("ws_loo_unavailable"):
            why.append("Ws 采样/LOO 不健康")
        out.update(choice="none", status="guard_degraded",
                   reason="不加 W：" + "；".join(why or ["未过物理/采样护栏"])
                          + f"（guards: {'、'.join(bad)}）")
        return out
    if cmp["delta"] < 0:
        out.update(choice="none", status="ws_worse",
                   reason=f"Δelpd(Ws−none)={cmp['delta']:.2f} < 0，保持 none")
        return out
    if cmp["delta"] >= thr:
        out.update(choice="Ws", status="switched",
                   reason=f"Δelpd(Ws−none)={cmp['delta']:.2f} ≥ "
                          f"max(2, 1.5·dse)={thr:.2f}（dse={cmp['dse']:.2f}），"
                          f"切换到 Ws")
        # P3（二轮审查）：exploratory 入参此前从未使用——quick 档的采样
        # 预算（800/400）不足以支撑正式的机制切换结论，与吸附支路
        # （select_k 的 exploratory 标注）不对称。补上。
        if exploratory:
            out["note"].append(
                "**quick 档**：本切换基于 quick 预算的 LOO，需正式档复核后"
                "才能作为机制结论引用（exploratory）")
        return out
    out.update(choice="none", status="uncertain",
               reason=f"Δelpd(Ws−none)={cmp['delta']:.2f} 为正但未达 "
                      f"{thr:.2f}（dse={cmp['dse']:.2f}）；证据不足，"
                      f"保持 none，报告保留手动对照建议")
    return out


# --------------------------------------------------------------------------
# 吸附 RL：门控 + 外层选择器（M4，立项 §6；语义与扩散的 gate/select 同构，
# select_k() 一行不动）
# --------------------------------------------------------------------------

def detect_ad(f, Z, res_none, frac_lo: float = 0.40,
              acf_thr: float = 0.8, low_snr_thr: float = 1.5) -> dict:
    """拟合后门控（零额外拟合成本）：none 的残差 + 原始数据形状。

    四条件（§6.1）+ 三道假阳性闸（§6.1.1，R7）：
      · sign_inductive：最低频段残差虚部均值 > 0（观测比拟合更"感性"）。
        残差口径 = 观测 − 拟合（与 resid_bins 一致），bins 取最低段。
      · systematic：acf > 0.8 或低频 rms/σ > 1.5 —— 随机噪声翻不了符号。
      · 闸①内部极小：最低 frac_lo 频点内 −Im 先降后升，极小点两侧各 ≥2 点
        （主判据是**离散点数** k_left/k_right，d_low 只作辅助记录）。
        **完全单调上翘一律不算**（Drift/气泡/仪器低频噪声的共同形状）。
        峰在窗内但贴边（k_left=1）→ `edge_peak`：只开放候选诊断，不自动选择。
      · 闸②半圆完整性：ΔRe（band 内实部跨度）与 2|Im|_peak 同量级（0.3–3×），
        环两臂要回到同一实轴基线；只翘不复位的排除。
      · 闸③可复现性：本函数拿不到重复谱 → 不判否，只在 reason 记
        "置信度降级"。有重复数据时由调用方补判。

    返回全部可 JSON 序列化：enter（是否做候选诊断拟合）、strong（是否具备
    自动模型比较资格——两者**不可合并**）、status、checks、corner（峰位/Rad
    粗估，复用 bm_ad.inv_from_lowfreq）、reason。
    只看证据不做决策——是否换模型由 select_adsorption 的配对 LOO 定。
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    order = np.argsort(f)
    f, Z = f[order], Z[order]
    out = dict(enter=False, strong=False, status="no_candidate", reason="",
               checks={}, corner=None, frac_lo=frac_lo)
    chk = out["checks"]

    # ---- 条件 1/2：符号 + 系统性（用 none 已算好的 bins/acf，零成本）----
    bins = res_none.get("bins") or res_none.get("resid_bins") or []
    s_im = float(res_none.get("s_im", res_none.get("sigma_im", 0.0)) or 0.0)
    acf = abs(float(res_none.get("acf") or 0.0))
    sign_ind, low_snr = False, float("nan")
    if bins:
        b = bins[0]                          # resid_bins 按 log f 从低到高排
        mean_im = float(b.get("mean_im", 0.0))
        sign_ind = mean_im > 0
        if s_im > 0:
            low_snr = float(b.get("rms_im", 0.0)) / s_im
    systematic = (acf > acf_thr) or (np.isfinite(low_snr) and low_snr > low_snr_thr)
    chk.update(sign_inductive=bool(sign_ind), low_mean_im=(float(bins[0].get("mean_im", float("nan"))) if bins else None),
               systematic=bool(systematic), acf=acf,
               low_rms_over_sigma=(float(low_snr) if np.isfinite(low_snr) else None))
    if not sign_ind:
        out.update(status="capacitive_tail",
                   reason="低频虚部残差为负（更容性）→ 交给扩散路径，不是吸附候选")
        return out
    if not systematic:
        out.update(status="noise_only",
                   reason=f"低频感性偏移未过系统性判据（acf={acf:.2f} ≤ {acf_thr}，"
                          f"低频 rms/σ={low_snr:.2f} ≤ {low_snr_thr}）——更像噪声翻转")
        return out

    # ---- 闸①/②：原始数据形状（最低 frac_lo 频点）----
    # 主判据是**离散点数** k_left / k_right（与 interior = 2<=k<=n_band-3 逐条
    # 一致）；d_low 只作辅助记录，不依赖"对数均匀采样"假设（AD_GATE_SPEC §4）。
    lf = np.log10(f)
    m = lf <= float(np.quantile(lf, frac_lo))
    n_band = int(m.sum())
    neg_im = -Z.imag[m]
    interior = half = False
    k_left = k_right = -1
    d_low = float("nan")
    if n_band >= 5:
        k = int(np.argmin(neg_im))
        k_left = int(np.count_nonzero(f[m] < f[m][k]))
        k_right = int(n_band - 1 - k)
        d_low = float(np.log10(f[m][k] / f[m][0]))
        interior = bool(k_left >= 2 and k_right >= 2)   # 两侧各 ≥2 点
        if k_left >= 1:            # 峰不在最低频点 → 粗估可用（含 edge_peak）
            im_pk = float(Z.imag[m][k])            # 极小点的 Im（感性 → 正）
            re_band = Z.real[m]
            d_re = float(np.max(re_band) - np.min(re_band))
            amp = 2.0 * abs(im_pk)
            half = bool(amp > 0 and 0.3 <= d_re / amp <= 3.0)
            # R_ad 粗估走 §7.3 恒等式（峰值 Im_max = R/2）；τ 由峰位反算。
            # 完整反演（含可靠性标记）见 bm_ad.inv_from_lowfreq，采样初值
            # 接线在 M6 一起做，这里只给门控报告用。
            out["corner"] = dict(f_peak=float(f[m][k]),
                                 Rad=2.0 * max(im_pk, 0.0),
                                 tau_ad=AD.tau_from_peak(float(f[m][k])) if AD else None,
                                 im_peak=im_pk, d_re=d_re,
                                 re_over_im_ratio=float(d_re / amp) if amp > 0 else None,
                                 n_band=n_band, k=int(k),
                                 k_left=k_left, k_right=k_right, d_low=d_low)
    chk.update(interior_min=interior, half_circle=half,
               k_left=(k_left if k_left >= 0 else None),
               k_right=(k_right if k_right >= 0 else None),
               d_low=(d_low if np.isfinite(d_low) else None))

    if interior and half:
        out.update(enter=True, strong=True, status="strong_candidate",
                   reason="符号+系统性+内部极小+半圆完整：进入 none vs ad 比较"
                          "（可复现性无重复谱可查，置信度记为单谱级）")
        return out
    if interior and not half:
        # 状态名不再被 weak_candidate 覆盖；补跑只是**候选诊断拟合**，
        # 不是模型比较（审查意见 §1 / AD_GATE_SPEC §3）。
        out.update(enter=True, strong=False, status="partial_arc",
                   reason="闸②：有感抗极小但 ΔRe 与 2|Im|_peak 不同量级——"
                          "只翘不复位，不是闭合半圆。保留候选诊断拟合"
                          "（enter=True），不进模型比较（strong=False）")
        return out
    if k_left == 1:
        # 峰在窗口内但贴边：低频侧只有 1 个采样点，无法确认完整极值。
        # 只开放诊断（enter=True），**不**降低自动选择门槛（strong 恒 False）。
        out.update(enter=True, strong=False, status="edge_peak",
                   reason="闸①：峰在窗口内但贴边（k_left=1，低频侧仅 1 个采样点），"
                          "无法确认完整极值——保留候选诊断拟合（enter=True），"
                          "不自动选择 ad（strong=False）")
        return out
    if k_left == 0:
        why = ("极小落在最低频点（k_left=0）——峰可能在窗口外（拐点低于 f_min，"
               "被 R_inf 吸收），或纯单调上翘")
    elif k_right < 2:
        why = f"峰靠近 band 上沿（k_right={k_right} < 2），低频段以单调上升为主"
    else:
        why = "band 内无可识别转折"
    out.update(status="monotonic_rise", enter=False, strong=False,
               reason=f"闸①：{why}——按 Drift/漂移处理，不算吸附（R7）")
    return out


def select_adsorption(none_res, ad_res, gate, exploratory: bool = False) -> dict:
    """外层选择器（公开入口）：none vs ad(RL)。

    与 `select_mechanism` 同一分工：只包一层 `bm_grade.annotate` 贴资格档，
    选择规则全在 `_select_adsorption_impl()` 里。
    """
    return GRADE.annotate(_select_adsorption_impl(none_res, ad_res, gate,
                                                  exploratory=exploratory))


def _select_adsorption_impl(none_res, ad_res, gate,
                            exploratory: bool = False) -> dict:
    """外层选择器：none vs ad(RL)。**不修改 select_k() 语义**。

    对齐 select_mechanism 的口径（§6.2）：
      · **先拦后比**（R9）：ad 采样不健康（分级 0 / Pareto-k>0.7 超 10% /
        LOO 不可用）直接 guard_degraded，失败结果绝不送去比 LOO；
      · 阻断型物理护栏（两条）：P(R_ad < 3σ̂) > 0.5（幅值与噪声同量级，
        v1.1 按 σ 不按 |Z|max）、P(f_c > f_max) > 0.5（退化为串联 L，
        已被 include_L 建模）；
      · **降级不阻断**：corr(R_ad, R_inf) ≤ −0.8（M3 标定的简并区）只把
        R_ad 的报告口径降级为"存在低频感性弛豫"，不阻止切换——Δ 显著时
        拦回 none 等于丢掉更好的模型（实测 Δelpd=+121、acf 0.89→0.49）；
      · gate.strong=False 不进比较（三闸有任一未全过 = 假阳性风险未排除）；
      · 切换门槛 Δ ≥ max(2, 1.5·dse)，Δ 为配对差（ad − none，正 = ad 更好）；
      · 0 ≤ Δ < 门槛 → uncertain（合法输出，保持 none）；Δ < 0 → ad_worse。
    """
    out = dict(choice="none", status="not_tested", reason="", gate=gate,
               delta=None, dse=None, threshold=None, guards={}, note=[])
    if not gate or not gate.get("strong"):
        g_stat = (gate or {}).get("status", "missing")
        out.update(status="gate_not_strong",
                   reason=f"门控 {g_stat}（三闸未全过），不进比较，保持 none")
        return out
    if ad_res is None:
        out.update(status="fit_missing", reason="ad 未拟合，保持 none")
        return out

    guards = {
        "ad_grade0": ad_res.get("grade") == 0,
        "ad_bad_k": float(ad_res.get("bad_k") or 0) > 0.10,
        "ad_loo_unavailable": not _finite(ad_res.get("elpd")),
        "amp_unidentifiable": (_finite(ad_res.get("p_ad_small"))
                               and float(ad_res["p_ad_small"]) > 0.5),
        "corner_out_of_window": (_finite(ad_res.get("f_c_out_prob"))
                                 and float(ad_res["f_c_out_prob"]) > 0.5),
    }
    out["guards"] = {k: bool(v) for k, v in guards.items()}
    # Rad–R_inf 简并（M3 标定）**不是阻断条件**：实测某谱 Δelpd=+121、
    # acf 0.89→0.49，环明显为真，若因"R_ad 绝对值不可单独引用"就拦回 none，
    # 等于为了报告口径丢掉一个显著更好的模型。§6.2 的护栏清单本来也没列它，
    # M3 的原话是「只报存在低频感性弛豫，不报 R_ad 单独值」= 改报告口径。
    degenerate = _rad_rinf_degenerate(ad_res)
    out["rad_rinf_degenerate"] = bool(degenerate)

    if any(guards.values()):
        bad = [k for k, v in guards.items() if v]
        why = []
        if guards["ad_grade0"] or guards["ad_bad_k"] \
                or guards["ad_loo_unavailable"]:
            why.append("ad 采样/LOO 不健康（先拦后比：失败结果不参与配对比较）")
        if guards["amp_unidentifiable"]:
            why.append("P(R_ad < 3σ̂) > 0.5：环幅值与噪声同量级，报了也没意义")
        if guards["corner_out_of_window"]:
            why.append("P(f_c > f_max) > 0.5：拐点在窗口上方，Z_ad 与串联 L "
                       "简并——要报就报 L_eff，不要报 R_ad/τ_ad 单独值")
        out.update(choice="none", status="guard_degraded",
                   reason="不加 ad：" + "；".join(why) + f"（guards: {'、'.join(bad)}）")
        return out

    cmp = _paired_loo_diff(none_res, ad_res)     # Δ = ad − none，正 = ad 更好
    if cmp is None:
        out.update(status="comparison_incomplete",
                   reason="两个家族没有可配对的逐点 LOO，保持 none")
        return out
    out.update(delta=cmp["delta"], dse=cmp["dse"], n=cmp["n"])
    thr = max(2.0, 1.5 * cmp["dse"])
    out["threshold"] = thr

    if exploratory:
        out["note"].append("quick 档采样预算撑不起严格结论，下面的比较只筛方向")

    if cmp["delta"] < 0:
        out.update(choice="none", status="ad_worse",
                   reason=f"Δelpd(ad−none)={cmp['delta']:.2f} < 0，保持 none")
        return out
    if cmp["delta"] >= thr:
        out.update(choice="ad", status="switched",
                   reason=f"Δelpd(ad−none)={cmp['delta']:.2f} ≥ "
                          f"max(2, 1.5·dse)={thr:.2f}（dse={cmp['dse']:.2f}），"
                          f"切换到吸附 RL" + ("；quick 档需正式档复核" if exploratory else ""))
        if degenerate:
            out["note"].append(
                "**R_ad 与 R_inf 强负相关（简并区）：模型切换成立（Δ 显著），"
                "但 R_ad 的绝对值不能单独引用**——常数项与欧姆电阻沿补偿方向"
                "滑动。报告只说「存在低频感性弛豫」，要报幅值就报 L_ad=R_ad·τ_ad "
                "或带上相关性说明（见 _ad_line / M3 标定）。")
        return out
    out.update(choice="none", status="uncertain",
               reason=f"Δelpd(ad−none)={cmp['delta']:.2f} 为正但未达 {thr:.2f}"
                      f"（dse={cmp['dse']:.2f}）；证据不足，保持 none")
    return out


def _rad_rinf_degenerate(res, thr: float = -0.8) -> bool:
    """corr(R_ad, R_inf) 强负 → 简并。两条口径都查：
    param_corr 的 strong 列表（{a,b,rho}），以及后验样本直接算
    （新列可能不在 strong 阈值里，但 −0.8 一定在 medium 以上）。"""
    pc = res.get("param_corr") or {}
    for item in list(pc.get("strong") or []) + list(pc.get("medium") or []):
        pair = {item.get("a"), item.get("b")}
        if pair == {"R_ad", "R_inf"} and _finite(item.get("rho")) \
                and float(item["rho"]) <= thr:
            return True
    s = res.get("s")
    if s is not None and "Rad" in s and "R_inf" in s:
        a, b = np.asarray(s["Rad"], float), np.asarray(s["R_inf"], float)
        if a.size >= 30 and np.std(a) > 0 and np.std(b) > 0:
            rho = float(np.corrcoef(np.log(np.maximum(a, 1e-300)),
                                    np.log(np.maximum(b, 1e-300)))[0, 1])
            return rho <= thr
    return False


def component_review(res, f):
    """拟合后的**组分可辨识性审查**（与 K 选择分离，不改变 K）。

    审查意见 2026-11-06：`P(峰位出窗)` 不再是 K 的硬剔除条件。
    K 由合格候选之间的配对 LOO 决定；峰位、窗口内贡献、参数简并、
    残差结构、LOO 稳健性在这里单独报告，分别产出「预测选择」与
    「组分/物理可辨识性」两类结论。

    返回 dict（全部可 JSON 序列化），由 save_out 落盘、run 打印、judge 引用。
    """
    K = int(res["K"])
    p = np.asarray(res.get("p_peak_out", []), float)
    states = {}
    cgrades = {}
    for i in range(K):
        v = p[i] if i < p.size else float("nan")
        key = f"component_{i + 1}"
        if not np.isfinite(v):
            states[key] = "uncertain"
        elif v > 0.5:
            states[key] = "out_of_window"
        elif v > 0.05:
            states[key] = "uncertain"
        else:
            states[key] = "in_window"
        # 资格档（只加字段，不改 states）：out_of_window 是 **C** —— 参数定得住、
        # 预测算得出，但本窗没覆盖该特征频率，无权说该过程存在与否；
        # 原来两种 uncertain（后验缺失 = A / 0.05<p≤0.5 = B）在这里分开。
        cgrades[key] = GRADE.component_grade(
            states[key], (float(v) if np.isfinite(v) else None))

    # 窗口内阻抗贡献：后验中位数参数在 f_min 与几何中心处的分量贡献。
    contrib = []
    try:
        sm = res["s"]
        mm = B.match_components(sm)
        o = np.argsort(-np.median(mm["f0"], axis=0))
        w = np.median(mm["w"][:, o], axis=0)
        tau = np.median(mm["tau"][:, o], axis=0)
        phi = np.median(mm["phi"][:, o], axis=0)
        zeta = np.median(mm["zeta"][:, o], axis=0)

        def zcomp(freqs):
            x = np.maximum(2 * np.pi * np.asarray(freqs, float)[:, None]
                           * tau[None, :], 1e-300)
            mag = np.exp(phi[None, :] * np.log(x))
            ang = phi[None, :] * (np.pi / 2)
            den = 1.0 + mag * (np.cos(ang) + 1j * np.sin(ang))
            return w[None, :] / den ** zeta[None, :]

        f_lo, f_geo = float(np.min(f)), float(np.sqrt(np.min(f) * np.max(f)))
        z_lo, z_geo = zcomp([f_lo])[0], zcomp([f_geo])[0]
        tot_lo = float(res["R_inf"]) + z_lo.sum()
        tot_geo = float(res["R_inf"]) + z_geo.sum()
        for i in range(K):
            contrib.append(dict(
                component=i + 1, state=states[f"component_{i + 1}"],
                p_peak_out=(float(p[i]) if i < p.size and np.isfinite(p[i])
                            else None),
                f0_hz=float(res["f0"][i]), w_ohm=float(res["w"][i]),
                zk_re_at_fmin=float(z_lo[i].real),
                zk_im_at_fmin=float(z_lo[i].imag),
                frac_abs_at_fmin=float(abs(z_lo[i]) / max(abs(tot_lo), 1e-300)),
                frac_im_at_fmin=float(z_lo[i].imag / max(abs(tot_lo.imag), 1e-300)),
                frac_abs_at_geom=float(abs(z_geo[i]) / max(abs(tot_geo), 1e-300)),
            ))
    except Exception as e:
        contrib.append(dict(error=f"{type(e).__name__}: {e}"))

    pc = res.get("param_corr") or {}
    deg = res.get("degeneracy") or []
    bounds = [float(d.get("at_bound", 0)) for d in deg]
    strong = [dict(a=x["a"], b=x["b"], rho=float(x["rho"]))
              for x in (pc.get("strong") or [])]
    param = dict(
        strong_pairs=strong, n_strong=len(strong),
        max_at_bound=(max(bounds) if bounds else None),
        level=("degenerate" if (strong or (bounds and max(bounds) > 0.10))
               else "identified"),
    )

    nl = res.get("null") or {}
    zr = float(res["trend"]["re"] / max(res["s_re"], 1e-12))
    zi = float(res["trend"]["im"] / max(res["s_im"], 1e-12))
    p_tr = (float(np.mean(nl["trend"] < max(zr, zi))) if "trend" in nl else None)
    p_acf = (float(np.mean(nl["acf"] < res["acf"])) if "acf" in nl else None)
    residual = dict(
        acf=float(res["acf"]), trend_re_sigma=zr, trend_im_sigma=zi,
        trend_percentile=p_tr, acf_percentile=p_acf,
        ppc_re=float(res["pp"]["cover_re"]),
        ppc_im=float(res["pp"]["cover_im"]),
        systematic=bool((p_tr is not None and p_tr > 0.99)
                        or (p_acf is not None and p_acf > 0.99)),
    )

    pk, pt = res.get("pareto_k"), res.get("pointwise_elpd")
    if pk is not None and pt is not None:
        pk, pt = np.asarray(pk, float), np.asarray(pt, float)
        bad = pk > 0.7
        kept = pt[~bad]
        loo = dict(
            available=True,
            pareto_k_max=float(np.nanmax(pk)),
            n_gt_07=int(bad.sum()), frac_gt_07=float(bad.mean()),
            high_influence_freqs=[float(x) for x in np.asarray(f)[bad]],
            elpd_all=float(pt.sum()),
            elpd_excl_high_k=(float(kept.sum()) if kept.size else None),
            elpd_shift=(float(kept.sum() - pt.sum()) if kept.size else None),
            note="删除高 Pareto-k 点仅是敏感性检查，不是精确 LOO / moment matching",
        )
    else:
        loo = dict(available=False)

    return dict(
        component_identifiability=states,
        # 与 states 一一对应的资格档（A/B/C/None）。**C 档原来只活在文字里**：
        # 峰位出窗的分量会被人读成「没有这个分量」，其实是「本窗无权评价」。
        component_grade=cgrades,
        window_contribution=contrib,
        parameter_identifiability=param,
        residual_review=residual,
        loo_robustness=loo,
        separation_note=("K 选择由合格候选之间的配对 LOO 决定；峰位出窗只进入"
                         "组分可辨识性审查，不再淘汰 K，也不放宽采样硬门禁。"),
    )


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
    elif st == "exploratory":
        print("  ** 探索性结果（quick 档）：采样预算不足，K 与 elpd 都不是"
              "模型选择结论，要引用请用正式档重跑 **")
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
    extra = ""
    _ed, _rk = r.get("elpd_diff"), r.get("rank")
    if _rk is not None and np.isfinite(_rk) and _rk > 0 \
            and _ed is not None and np.isfinite(_ed):
        extra = f"   落后最好者 {abs(_ed):.1f} ± {r['dse']:.1f}"
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
          "**模型选择的硬筛选优先看它**。φζ>1 时峰位无意义 → 标 —，"
          "退回参数口径 P(τ出窗)（p_tau_out）作降级判据并标注 nonphysical"
          "——峰位**不再单独淘汰 K**（P(峰位出窗) 已移到组分可辨识性审查，"
          "见 ③b；2026-11-10 起）。")
    if r.get("diffusion"):
        print("  扩散项：" + _diff_line(r))
        if np.isfinite(r.get("p_diff_small", np.nan)):
            print(f"  P({r['diff_amp_key']} < 1e-3·|Z|max) = {r['p_diff_small']:.2f}"
                  f"  ← 后验概率，衡量「幅值相对整条谱是否可忽略」；"
                  f"**不等于「没有扩散」**")
    if r.get("adsorption") == "RL" and "Rad" in r["s"]:
        print("  吸附项：" + _ad_line(r))

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
    elif st == "exploratory":
        msg.insert(0, "**探索性结果（quick 档）**：采样预算撑不起严格收敛线，"
                      "K 与 elpd 都不是模型选择结论；引用前须用正式档重跑。")
    elif st == "degraded":
        msg.append("模型选择降级：只在「可用」档里做相对比较，"
                   "引用 K 时须说明它没过硬筛选。")

    cr = r.get("component_review") or {}
    cid = cr.get("component_identifiability") or {}
    oow = [k for k, v in cid.items() if v == "out_of_window"]
    unc = [k for k, v in cid.items() if v == "uncertain"]
    if oow or unc:
        parts = []
        if oow:
            parts.append("、".join(oow) + " 峰位大概率在窗口外")
        if unc:
            parts.append("、".join(unc) + " 峰位状态不确定")
        msg.append("**组分可辨识性受限**（不影响上面的 K 选择）：" + "；".join(parts)
                   + "。请结合窗口内阻抗贡献、参数简并与扩散对照判断，"
                     "不要把该分量解释成已确定的独立物理弛豫过程。")

    # 参数简并：边际区间看着都窄，但两个参数其实只定出一个组合。
    pc = r.get("param_corr") or {}
    if pc.get("strong"):
        who = "；".join(f"{p['a']}↔{p['b']}(ρ={p['rho']:+.2f})"
                        for p in pc["strong"][:3])
        msg.append(f"**参数强相关**：{who}。这些参数只能定出一个组合，"
                   f"单独引用其中一个会被读成「已经定住」，实际没有。")

    # 多分量拼 45° 段：逐分量判据漏掉的那条路径（如真值是纯 Warburg 的谱）。
    wm = r.get("warburg_multi") or {}
    if wm.get("level") == "weak":
        msg.append("**可能是扩散尾被拆成多个分量**：" + "；".join(wm.get("why") or [])
                   + "。弱识别，不改 K；若要确认，用 --diffusion Ws 跑一次对比。")

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

    # 峰位口径现在只用于**组分审查**，不参与 select_k 的 K 硬筛选。
    # 出窗 = 峰形/特征位置未被频率窗口约束，不等于窗口内阻抗贡献无效。
    hi_out = [i for i in range(K) if np.isfinite(r["p_peak_out"][i])
              and r["p_peak_out"][i] > 0.5]
    mid_out = [i for i in range(K) if np.isfinite(r["p_peak_out"][i])
               and 0.05 < r["p_peak_out"][i] <= 0.5]
    nan_out = [i for i in range(K) if not np.isfinite(r["p_peak_out"][i])]
    if hi_out:
        who = ", ".join(f"#{i + 1} P={r['p_peak_out'][i]:.2f}" for i in hi_out)
        msg.append(f"**{len(hi_out)} 个分量大概率在窗口外**（{who}，峰位口径）："
                   "峰形与特征位置未被当前频率窗口约束。**这不淘汰 K**，"
                   "但要核对它在实测频段内的阻抗贡献（见组分审查），"
                   "不能据此断言已识别出独立物理过程。")
    if mid_out:
        who = ", ".join(f"#{i + 1} P={r['p_peak_out'][i]:.2f}" for i in mid_out)
        msg.append(f"{len(mid_out)} 个分量跨在窗口边界上（{who}，组分状态 uncertain）："
                   f"峰位本身不可信，只当「这里有个未分辨的过程」看。")
    if nan_out:
        who = ", ".join(f"#{i + 1}" for i in nan_out)
        msg.append(f"{who} 的物理峰位诊断不可用（φζ>1 样本过多，组分状态 uncertain）："
                   f"**峰位口径失效，不能改用 τ 口径充数**；"
                   f"φζ>1 超限仍由物理有效性规则处理，但峰位本身不再淘汰 K。")

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

def prescreen_stage(f, Z, deind=None):
    """① 预处理 + 传统 DRT。只产出候选与初值，不产出结论。

    Z 是**交给 DRT 的输入**，可能已经过 `deinduce_for_drt` 扣掉 jωL；
    BM-DRT 的似然另外收原始 Z。deind 只用于打印这一层做了什么。
    """
    t0 = time.time()
    pres, pk, pk_all = prescreen(f, Z)
    dt = time.time() - t0
    print(f"\n=== ① DRT 预判断（{dt * 1000:.0f} ms, λ={pres['lam']:.3g}） ===")
    if deind is not None:
        tag = "**已扣 jωL**" if deind.get("used") else "原始 Z"
        print(f"  DRT 输入：{tag} —— {deind.get('reason', '')}")
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


def _json_safe(o):
    """递归把 NaN/±Inf 换成 None，其余原样（审查 P2-4，二轮）。

    `json.dumps` 默认 `allow_nan=True` 会把 NaN/Infinity 写成字面量——
    那不是合法 JSON，严格解析器（`json.loads(..., parse_constant=抛错)`）
    直接报错，事后难排查。配合 `allow_nan=False` 使用：漏网的 NaN 会
    raise 而不是静默落盘。
    """
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)) and not np.isfinite(float(o)):
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    return o


def _write_kk_reject(path, f, Z, kk, kk_mode, seed=None):
    """gate 模式且 K-K 判 fail：跳过整条拟合链，只落一份可机读的拒绝记录。

    必须写 `real_<stem>.json`——`bmdrt.py` 的批量流程靠它判「流程跑完」，
    不写会被记成「跑完但没结果」而误报 [FAIL]。`fit_performed=false`
    就是批量流程区分「正常拒绝」与「真的拟合成功」的契约字段。

    `sampling_performed=false` + `seed=null` 是给 B3b 用的：KK 拒绝谱**没有采样**，
    不能被当成「某个 seed 下跑过了」。把 seed 记成 0 会让它看起来像一次拟合。
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
        sampling_performed=False,
        seed=(None if seed is None else int(seed)),
        warnings=["**KK 门禁拒绝**：数据不自洽，本次未拟合，不输出最优 K。"],
        k_selection=["**不输出最优 K 的结论**：拟合前 K-K 门禁判 fail。"],
    )
    js.write_text(json.dumps(_json_safe(rec), ensure_ascii=False, indent=2,
                             default=float, allow_nan=False),
                  encoding="utf-8")
    return js


def save_out(stem, f, Z, res, pres, include_L, zeta, tips, K_sel_note):
    OUT.mkdir(exist_ok=True)
    js = OUT / f"real_{stem}.json"
    # P3（二轮审查）：结果按 stem 命名，同名谱/重跑会静默覆盖——至少让
    # 覆盖在输出里可见。（相对 CWD 的 OUT 行为不动：主链路已显式
    # resolve，直接调 run() 的外部脚本自行负责。）
    if js.exists():
        print(f"  [注意] {js} 已存在，本次运行将覆盖它（stem 同名复跑）")
    summary = dict(
        file=stem, n=int(f.size), f_min=float(f.min()), f_max=float(f.max()),
        K=int(res["K"]), include_L=bool(include_L),
        zeta=("free" if zeta == "free" else float(zeta)),
        # seed + sampling：留给 B3b 的随机性诊断。写的是**实际生效值**，
        # 不是命令行请求值（`--seed` 被吃掉时这两项会露馅）。
        seed=int(res.get("seed", KW["seed"])),
        sampling=res.get("sampling"),
        lam_drt=float(pres["lam"]), R_inf_drt=float(pres["R_inf"]),
        t_fit_s=float(res["t"]), divergences=int(res["ndiv"]),
        tree_cap=float(res["tds"]["frac_cap"]),
        sample_grade=int(res["grade"]), grade_fail=res["grade_bad"],
        elpd_loo=res["elpd"], elpd_se=res["elpd_se"],
        elpd_diff=res["elpd_diff"], dse=res["dse"],
        loo_error=res.get("loo_error"),
        comparison_performed=bool(res.get("comparison_performed")),
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
        component_review=res.get("component_review"),
        param_corr=res.get("param_corr"),
        diffusion_resolution=res.get("diffusion_resolution"),
        auto_mechanism=res.get("auto_mechanism"),
        # 吸附 RL（M4）：门控证据 + 选择结论 + 派生可辨识量，全部有则写。
        # 与 auto_mechanism 并列——事后复核"这条谱有没有判过吸附、为什么"。
        ad_gate=res.get("ad_gate"),
        ad_selection=res.get("ad_selection"),
        adsorption=res.get("adsorption"),
        p_ad_small=(res.get("p_ad_small")
                    if _finite(res.get("p_ad_small")) else None),
        f_c_out_prob=(res.get("f_c_out_prob")
                      if _finite(res.get("f_c_out_prob")) else None),
        L_ad_uH=(res.get("L_ad") * 1e6 if _finite(res.get("L_ad")) else None),
        # DRT 去电感通道的决策（used/L/四道检查/扣前后残差）+ 噪声地板
        drt_deinduce=res.get("drt_deinduce"),
        sigma_floor=res.get("sigma_floor"),
        # 扩散项：写进 json 才能让 status / 事后复核不重跑就看得到
        diffusion=res.get("diffusion"),
        n_warburg_like=res.get("n_warburg_like", 0),
        warburg_multi=res.get("warburg_multi"),
        p_diff_small=res.get("p_diff_small"),
        zmax_obs=res.get("zmax_obs"),
        # 模型选择实际用到的字段，单独记一份，免得事后只能从 p_out 反推：
        #   p_peak_out_max 现在只用于组分可辨识性审查，不再参与 K 硬筛选
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
    js.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2,
                             default=float, allow_nan=False),
                  encoding="utf-8")
    np.savez_compressed(OUT / f"real_{stem}_samples.npz",
                        f=f, Zre=Z.real, Zim=Z.imag,
                        **{k: v for k, v in res["s"].items()
                           if isinstance(v, np.ndarray)})
    return js


def run(path, K_force=None, krange=None, wide=False, kmax=K_MAX,
        include_L=None, zeta="1.0", dry=False, quick=False, ta=None,
        diffusion=None, kk_mode="gate", kk_opts=None, diag=None,
        drt_opts=None, sigma_floor=0.0, auto_mechanism=False,
        auto_mechanism_full=False, adsorption=None, auto_adsorption=False,
        seed=None):
    """kk_mode: gate(fail 即拒绝定 K) | warn(只报告) | off(不跑门禁)。
    kk_opts 是传给 kk_check 的数值参数（c/max_M/res_tol/auto_L/...），
    不在代码里写死，由 config.json 统一管。
    diag 同理：事后诊断阈值（相关矩阵 / 扩散可分辨性），来自 config.json。
    drt_opts 是 DRT 去电感通道的阈值（min_share/tol_stab/min_gain/tol_neg）。
    sigma_floor 是噪声地板（相对 zmax），0.0 = 关闭。
    adsorption：None(默认，逐位不变) / "RL" 手动加吸附元件。
    auto_adsorption：True 时在 none 家族跑完后过 detect_ad 门控，三闸全过才
    补跑 (K, adsorption='RL') 并配对比较（M4，§6.3 一期屏蔽 both，R8）。
    seed：sampler 随机流。`None`（默认）= 不动 `KW["seed"]`(=0)，旧命令逐位不变；
    传整数则本次所有拟合（none 家族 + 补跑的 Ws / ad）都用这个 seed。
    """
    drt_opts = drt_opts or {}
    sigma_floor = float(sigma_floor or 0.0)
    diag = diag or {}
    print(f"\n=== 读数据 {path} ===")
    f, Z = load_spectrum(path)

    if diffusion:
        print(f"  扩散项：{diffusion}（对比基线时用 --diffusion none）")
    if adsorption or auto_adsorption:
        print(f"  吸附项：{adsorption or '自动判定'}"
              f"{'（与扩散互斥：一期屏蔽 both，见立项 §6.3）' if diffusion else ''}"
              f"（关闭用 --adsorption none）")
    if diffusion and adsorption:
        # 立项 §6.3/R8：R_ad 与 R_W 在极低频不可分辨，同时放开会让后验
        # 退化成脊线、R̂ 爆表。一期不做组合，直接拒绝而不是静默降级。
        raise SystemExit("一期不支持 --diffusion 与 --adsorption 同时开启"
                         "（R_ad 与 R_W 在低频段不可分辨，见 ADSORPTION_PROJECT.md §6.3）")

    if include_L is None:
        include_L = detect_L(f, Z)
        print(f"  电感项：{'需要（高频 −Im Z < 0）' if include_L else '不需要'}"
              f"（--L / --no-L 可覆盖）")

    # ---- ⓪ K-K 事前门禁：读数据之后、DRT 预判之前 ------------------
    # PSIS-LOO 只能在「似然假设成立」的前提下比较模型，检验不了假设本身。
    # 所以这一步必须在任何拟合之前，且 fail 时直接拒绝定 K。
    # P2-2（二轮审查）：门禁**不能静默 fail-open**——异常/依赖缺失导致的
    # "unavailable" 过去照常拟合，读起来却像门禁通过。现在：
    #   ① 恒写 `kk_performed`（True=门禁真的跑了 / False=没跑成）；
    #   ② gate 模式遇 unavailable 打显著警告；
    #   ③ config kk 段 `strict: true` 时，gate 模式下 unavailable 也拒绝
    #      （默认 false = 保持旧行为，但警告与字段照写）。
    kk = None
    if kk_mode != "off" and _kk_check is not None:
        _strict = bool((kk_opts or {}).get("strict", False))
        _kk_kw = {k: v for k, v in (kk_opts or {}).items() if k != "strict"}
        try:
            kk = _kk_check(f, Z, include_L=include_L, **_kk_kw)
            kk["kk_performed"] = True
        except Exception as e:             # 门禁绝不能让主流程崩
            kk = {"status": "unavailable",
                  "reason": f"K-K 门禁异常，已跳过：{type(e).__name__}: {e}",
                  "attribution": [], "n": int(f.size),
                  "kk_performed": False,
                  "f_min": float(f.min()), "f_max": float(f.max())}
        print("\n=== ⓪ K-K 事前门禁（拟合前） ===")
        print(_kk_format_report(kk) if _kk_format_report is not None
              else f"  {kk['status']}: {kk['reason']}")
        if kk.get("status") == "unavailable":
            print("  ** 注意：本次 K-K 门禁没有跑成（异常/依赖缺失），"
                  f"kk_performed=false —— 数据自洽性未经验证，"
                  f"{'strict 模式：拒绝拟合' if (kk_mode == 'gate' and _strict) else '本次仍继续拟合（fail-open，已如实披露）'} **")
        if kk_mode == "gate" and kk.get("status") in ("fail", "unavailable") \
                and (kk.get("status") == "fail" or _strict):
            print("\n  ** 数据不自洽（或门禁不可用且 strict），拒绝定 K；本次不做拟合 **")
            js = _write_kk_reject(path, f, Z, kk, kk_mode, seed=seed)
            print(f"  结果：{js}")
            return None

    # ---- ⓪b DRT 去电感数据通道 -------------------------------------
    # 只决定**交给 DRT 的那份数据**。BM-DRT 的似然在下面仍然收原始 Z。
    deind = None
    Z_drt = Z
    if drt_opts.get("deinduce", True) and _kk_estimate_L is not None:
        try:
            deind = deinduce_for_drt(
                f, Z, L_est=(kk or {}).get("L_est"),
                n_hi=int(drt_opts.get("n_hi", 5)),
                min_share=float(drt_opts.get("min_share", 5e-3)),
                tol_stab=float(drt_opts.get("tol_stab", 0.35)),
                min_gain=float(drt_opts.get("min_gain", 0.20)),
                tol_neg=float(drt_opts.get("tol_neg", 0.02)))
            if deind.get("used"):
                Z_drt = Z - 1j * (2 * np.pi * f) * float(deind["L"])
        except Exception as e:              # noqa: BLE001
            deind = dict(used=False, L=None, checks={},
                         reason=f"去电感通道异常，已跳过：{type(e).__name__}: {e}")

    pres, pk, pk_all, _ = prescreen_stage(f, Z_drt, deind=deind)
    K_drt = max(1, min(len(pk), kmax))

    if dry:
        lo, hi = max(1, K_drt - 1), min(kmax, K_drt + 1)
        print(f"\n（--dry 结束）建议下一步：python bm_real.py {path} --krange {lo},{hi}")
        print(f"  若要确认局部扫描没漏掉更好的 K：--wide（扫 1..{kmax}）")
        return None

    kw = dict(KW)
    # seed 必须在 quick / ta 之前落进 kw：`_fit_ks` 把 kw 整包传给 fit_one，
    # 再由 `_resolve_seed()` 从这里取。`None` = 保持 KW 的 0（旧行为逐位不变）。
    if seed is not None:
        kw["seed"] = int(seed)
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

    def _fit_ks(diffusion_use, tag="", k_list=None, adsorption_use=None,
                ad_init=None):
        """给定扩散/吸附家族，跑 K 扫描 + select_k。select_k 的语义不动。

        `k_list` 非空时只跑这些 K（跨家族补跑 Ws / ad 用：固定 none 选出的
        K，不重扫，成本约 +1 次拟合）。
        `ad_init` 是 `bm_ad.inv_from_lowfreq` 的粗估，仅在吸附家族且
        init_reliable 时并入播种（见 init_values）。
        """
        res_list = []
        for K in (list(k_list) if k_list else ks):
            init = init_values(pk, pres, K, ad_init=ad_init) if pk else None
            print(f"\n  --- K={K} 拟合中（播种={'是' if init else '否'}"
                  f"{('，' + tag) if tag else ''}）---")
            r = fit_one(f, Z, K, include_L, zeta, init=init, kw=kw,
                        seed=kw.get("seed", KW["seed"]),
                        diffusion=diffusion_use, diag=diag,
                        sigma_floor=sigma_floor, adsorption=adsorption_use)
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
            print(f"\n=== ③ 配对比较（az.compare / PSIS-LOO，逐点相减"
                  f"{('，' + tag) if tag else ''}） ===")
            cols = [c for c in ("rank", "elpd_loo", "elpd_diff", "dse", "weight", "p_loo")
                    if c in df.columns]
            print(df[cols].to_string(float_format=lambda v: f"{v:.1f}"))
            best, note = select_k(res_list, exploratory=bool(quick))
            print("\n  规则：① 采样/物理硬门禁（分级0 / φζ>1 超 1% /"
                  " Pareto-k>0.7 超 10% / LOO 不可用）"
                  " ② 合格候选按配对 LOO 排序 ③ 差 < max(2, dse) 取更小 K"
                  " ④ P(峰位出窗) 不再淘汰 K，移到组分可辨识性审查")
            for t in note:
                print(f"    · {t}")
            st = best.get("selection_status", "converged")
            if st == "rejected":
                print("  → **没有可信的 K**：下面的 elpd 排序仅相对、无物理意义，"
                      "不输出最优 K 结论")
            elif st == "exploratory":
                print(f"  → 探索性结果 K={best['K']}（quick 档，**不是模型选择结论**；"
                      f"正式档重跑后再引用）")
            else:
                extra = "，且仅来自「可用」降级档" if st == "degraded" else ""
                print(f"  → **K={best['K']}**（只在 {ks[0]}..{ks[-1]} 内最优{extra}）")
        else:
            best, note = res_list[0], []
            best["selection_status"] = "not_compared"
            print("  （只跑了一个 K，未做模型选择）")
        return best, note, res_list

    # 吸附粗估（§7.3）：手动 RL 或自动判定都用它做播种。粗估失败不致命，
    # 退回先验中位数起步（只影响采样效率，不影响后验）。
    ad_init = None
    if (adsorption == "RL" or auto_adsorption) and AD is not None:
        try:
            ad_init = AD.inv_from_lowfreq(f, Z)
            if not ad_init.get("init_reliable"):
                print(f"  吸附粗估不可靠（interior={ad_init['interior']}, "
                      f"low_half={ad_init['low_half']}, "
                      f"prom={ad_init['prominence']:.2f}）→ 从先验起步")
        except Exception as e:                   # noqa: BLE001
            ad_init = None
            print(f"  吸附粗估失败（{type(e).__name__}: {e}）→ 从先验起步")

    best, note, res_list = _fit_ks(diffusion, adsorption_use=adsorption,
                                   ad_init=ad_init)

    # ---- ③c 外层机制选择：none vs Ws（独立于 select_k）----
    # 只有默认 none 且显式 --auto-mechanism 才触发；手动 --diffusion Ws/Wo
    # 时保持原行为，不叠加自动层。
    auto_mech = None
    if auto_mechanism and diffusion is None:
        gate = diffusion_gate(kk, best, pres=pres, pk=pk, f=f, Z=Z)
        need, ev = _none_needs_ws(best)
        gate_enter = gate.get("status") == "candidate"
        print("\n=== ③c 机制比较（none vs Ws，独立于 select_k） ===")
        print(f"  门控证据：{gate['status']} — {gate['reason']}")
        wd = gate.get("ws_diag") or {}
        if wd:
            _bd = wd.get("band_dec", float("nan"))
            _bds = f"{_bd:.2f}" if np.isfinite(_bd) else "—"
            print(f"  数据侧端点：{wd.get('term')}（family={wd.get('family')}，"
                  f"verdict={wd.get('verdict')}，45° 段 {_bds} dec）")
        print(f"  none 残差：acf={ev['acf']:.2f}，低频 rms/σ="
              f"{ev['low_rms_over_sigma']:.2f}（阈值 {ev['acf_thr']:.1f} / "
              f"{ev['low_snr_thr']:.1f}）")
        print(f"  触发：残差{'有系统性' if need else '无系统性'}"
              f"{'；低频尾是 45° 候选（门控）' if gate_enter else ''}"
              f"{'；感抗尾 → 不补跑' if gate.get('status') == 'inductive_tail' and not auto_mechanism_full else ''}")
        if gate.get("status") == "inductive_tail" and not auto_mechanism_full:
            # L1' 的价值就在这里：数据侧已否证「窗内容性扩散尾」，补跑 Ws
            # 只会让 Aw 不可辨识（实测 18/18 真实谱都被 diff_amp_unidentifiable
            # 挡回），白花一次拟合。`--auto-mechanism-full` 时不短路，保留验证路径。
            auto_mech = dict(choice="none", status="skipped_inductive",
                             reason="低频 -Im<0（感抗尾，数据侧判据）：三类扩散尾"
                                    "都要求 -Im>0 且按 f^(-1/2)~f^(-1) 增长，"
                                    "符号不符即否证；不进入 none vs Ws 比较，"
                                    "该特征走吸附支路（R∥L）",
                             gate=gate, residual=ev, delta=None, dse=None,
                             threshold=None, guards={}, note=[])
            auto_mech = GRADE.annotate(auto_mech)   # 资格档 B（数据侧否证）
            print(f"  → 跳过 Ws：{auto_mech['reason']}")
        elif not need and not gate_enter and not auto_mechanism_full:
            auto_mech = dict(choice="none", status="skipped_by_residual",
                             reason="none 已拟合到噪声水平（残差无系统性），"
                                    "补跑 Ws 不可能显著改善，跳过",
                             gate=gate, residual=ev, delta=None, dse=None,
                             threshold=None, guards={}, note=[])
            auto_mech = GRADE.annotate(auto_mech)   # 资格档 B（残差已到噪声）
            print(f"  → 跳过 Ws：{auto_mech['reason']}")
        else:
            k_fixed = int(best["K"])
            why = ("强制全量" if auto_mechanism_full else
                   ("低频尾 45° 候选（门控）" if gate_enter and not need
                    else "残差有系统性"))
            print(f"  补跑 Ws：固定 K={k_fixed}（{why}；不重扫 K，"
                  f"成本约 +1 次拟合）")
            best_ws, note_ws, _ = _fit_ks("Ws", tag=f"Ws/K={k_fixed}",
                                          k_list=[k_fixed])
            share = None
            if WS is not None and isinstance(best_ws.get("s"), dict):
                try:
                    share = WS.ws_share_low(best_ws["s"], float(np.min(f)))
                except Exception:            # noqa: BLE001
                    share = None
            auto_mech = select_mechanism(best, best_ws, gate,
                                         exploratory=bool(quick), share=share)
            auto_mech["residual"] = ev
            print(f"\n  → 机制选择：{auto_mech['status']} "
                  f"[资格档 {auto_mech.get('grade') or '-'}] — "
                  f"{auto_mech['reason']}")
            for t in auto_mech.get("note", []):
                print(f"    · {t}")
            if auto_mech["choice"] == "Ws":
                print("  → 采用 Ws 家族结果（K 固定为 none 选出的 K）")
                best, note = _adopt_family_result(best, best_ws, note_ws, note)
    best["auto_mechanism"] = auto_mech

    # ---- ③d 吸附判定：none vs ad(RL)（M4，独立于 select_k，§6.3 一期屏蔽 both）----
    ad_gate, ad_sel = None, None
    if auto_adsorption and adsorption is None and diffusion is None:
        print("\n=== ③d 吸附判定（none vs ad(RL)，独立于 select_k） ===")
        if auto_mech and auto_mech.get("choice") == "Ws":
            # 机制已判扩散：一期不做 Ws+ad 组合（R_ad 与 R_W 低频不可分辨，R8）
            ad_sel = dict(choice="none", status="ws_taken",
                          reason="机制选择已判定 Ws（容性尾）；一期屏蔽 Ws+ad 组合，"
                                 "不再评估吸附环",
                          gate=None, delta=None, dse=None, threshold=None,
                          guards={}, note=[])
            ad_sel = GRADE.annotate(ad_sel)     # 资格档 A（代码范围限制，非数据结论）
            print(f"  → 跳过吸附判定：{ad_sel['reason']}")
        else:
            ad_gate = detect_ad(f, Z, best)
            print(f"  门控：{ad_gate['status']} — {ad_gate['reason']}")
            if not ad_gate.get("enter"):
                ad_sel = dict(choice="none", status="skipped_by_gate",
                              reason=ad_gate["reason"], gate=ad_gate,
                              delta=None, dse=None, threshold=None,
                              guards={}, note=[])
                ad_sel = GRADE.annotate(ad_sel)   # 资格档按 gate.status 定（A/B）
                print(f"  → 不补跑 ad：{ad_gate['reason']}")
            else:
                k_fixed = int(best["K"])
                print(f"  补跑 ad(RL)：固定 K={k_fixed}（不重扫 K，"
                      f"成本约 +1 次拟合）")
                best_ad, note_ad, _ = _fit_ks(
                    None, tag=f"ad/K={k_fixed}", k_list=[k_fixed],
                    adsorption_use="RL", ad_init=ad_init)
                ad_sel = select_adsorption(best, best_ad, ad_gate,
                                           exploratory=bool(quick))
                print(f"\n  → 吸附选择：{ad_sel['status']} "
                      f"[资格档 {ad_sel.get('grade') or '-'}] — {ad_sel['reason']}")
                for t in ad_sel.get("note", []):
                    print(f"    · {t}")
                if ad_sel["choice"] == "ad":
                    print("  → 采用 ad(RL) 家族结果（K 固定为 none 选出的 K）")
                    best, note = _adopt_family_result(best, best_ad, note_ad, note)
    best["ad_gate"] = ad_gate
    best["ad_selection"] = ad_sel

    if kk is not None:
        best["kk"] = kk
    # DRT 去电感通道的决策 + 噪声地板：必须落盘，否则事后无法复核
    # "这次 DRT 用的是原始 Z 还是扣过 L 的 Z"、"σ 有没有被地板截住"。
    best["drt_deinduce"] = deind
    best["sigma_floor"] = sigma_floor

    report(best, include_L, zeta, f)
    best["component_review"] = component_review(best, f)
    cr = best["component_review"]
    print("\n=== ③b 组分可辨识性审查（与 K 选择分离，不改变 K） ===")
    for c in cr["window_contribution"]:
        if "error" in c:
            print(f"    （贡献计算失败：{c['error']}）")
            continue
        po = "—" if c["p_peak_out"] is None else f"{c['p_peak_out']:.2f}"
        print(f"    #{c['component']} f0={c['f0_hz']:.3g} Hz  P出窗={po}  "
              f"状态={c['state']}  |Z|占比@fmin={c['frac_abs_at_fmin']:.1%}  "
              f"Im占比@fmin={c['frac_im_at_fmin']:+.1%}")
    pi = cr["parameter_identifiability"]
    print(f"    参数可辨识性：{pi['level']}（强相关 {pi['n_strong']} 对）")
    rr = cr["residual_review"]
    print(f"    残差系统性：{rr['systematic']}（acf={rr['acf']:.2f}，"
          f"趋势 {rr['trend_im_sigma']:.2f}σ）")
    lo = cr["loo_robustness"]
    if lo.get("available"):
        sh = lo.get("elpd_shift")
        shift = "—" if sh is None else f"{sh:+.2f}"
        print(f"    LOO 稳健性：Pareto-k max={lo['pareto_k_max']:.2f}，"
              f">0.7 共 {lo['n_gt_07']} 点，去点后 Δelpd={shift}")
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
    ap.add_argument("--sigma-floor", type=float, default=None,
                    help="噪声地板（相对 zmax），如 1e-4。默认 0 = 关闭。"
                         "只对无噪/近无噪数据有意义；开启后 σ 不会低于该值，"
                         "用来切断 σ→0 的漏斗。真实数据请按仪器精度标定，"
                         "不要为了让 R̂ 好看而调")
    ap.add_argument("--no-deinduce", action="store_true",
                    help="关闭 DRT 去电感通道，DRT 直接用原始 Z（对照用）")
    ap.add_argument("--adsorption", default="none", choices=["none", "RL"],
                    help="加吸附 RL 元件（低频感性环）：none(默认，逐位不变) / RL。"
                         "与 --diffusion 互斥（一期屏蔽组合，见 ADSORPTION_PROJECT.md §6.3）")
    ap.add_argument("--auto-adsorption", action="store_true",
                    help="none 家族跑完后过 detect_ad 门控，三闸全过才补跑 ad(RL) "
                         "并配对比较（不修改 select_k 语义）")
    a = ap.parse_args()
    if a.zeta == "free":
        print("** --zeta free：HN 的解析 DRT 在 ζφ>1 时会出负值，"
              "后验曲线只作可视化；峰位/面积的解释要先做可辨识性检查 **")
    diff = None if a.diffusion == "none" else a.diffusion
    ads = None if a.adsorption == "none" else a.adsorption
    run(a.csv, K_force=a.K, krange=a.krange, wide=a.wide, kmax=a.kmax,
        include_L=a.L, zeta=a.zeta, dry=a.dry, quick=a.quick, ta=a.ta,
        diffusion=diff, kk_mode=a.kk,
        drt_opts={"deinduce": not a.no_deinduce},
        sigma_floor=(a.sigma_floor or 0.0),
        adsorption=ads, auto_adsorption=a.auto_adsorption)


if __name__ == "__main__":
    main()
