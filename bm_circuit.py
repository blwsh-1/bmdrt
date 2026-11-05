"""等效电路图（模式 A：BM-DRT 对应的电路）。

**这个模块只画 BM-DRT 的模型本身**，也就是

    Z(ω) = R∞ + jωL + Σ_k  w_k / [1 + (jωτ_k)^φ_k]^ζ_k

它**不是**通用 ECM 拓扑绘图器。BM-DRT 的 K 个分量在数学上是 K 个串联的阻抗项，
这不等于"程序识别出了 K 个 R∥CPE 的真实电路拓扑"。要做任意串并联网络
（Warburg / 有限长 Warburg / Gerischer / 多支路），得另写一个读拓扑描述
（节点 + 元件列表）的渲染器，那是模式 B，本文件不承担。

## 参数怎么来的（每个符号都能复现拟合）

模型分量 `w/[1+(jωτ)^φ]^ζ` 与 `R∥CPE` 在 ζ=1 时完全等价：

    Z_{R∥CPE} = R / [1 + R·Q(jω)^φ]
    ⇒  R_k = w_k,   Q_k = τ_k^{φ_k} / w_k,   τ_k = (R_k Q_k)^{1/φ_k}

φ_k=1 时 CPE 退化为理想电容，`C_k = Q_k = τ_k/R_k`。
Q 的量纲是 `S·s^φ`（不是法拉），所以图上只在 φ=1 时才写成 C。

`τ_k` 取的是**模型里实际使用的那个 τ**（由 `bm_core.match_components` 的跨样本
匹配结果取中位数），不是从峰位反推的。注意 ζ≠1 时 DRT 峰**不在** `1/(2πτ)`：
实测 ζ=1.31 时峰位偏 −8.4%，所以图上把 `f_τ = 1/(2πτ)` 和 `f_peak` 分开标。

## 元件库：只画模型里真有的东西

| 元件 | BM-DRT 支持 | 本图是否画 |
|---|---|---|
| R（串联 R∞ / 并联 R_k） | 是 | 是 |
| L（串联电感） | 是（`include_L`） | 是，仅在启用时 |
| CPE（R_k ∥ CPE_k） | 是（ζ=1 时精确等价） | 是 |
| C（理想电容） | 是（CPE 的 φ=1 特例） | 是，φ≈1 时改画电容、标 C |
| HN（ζ≠1） | 是（唯象） | 是，虚线框 |
| 半无限 Warburg（Ws） | 是（`diffusion="Ws"`） | 是，三道斜线 |
| 有限长阻挡 Warburg（Wo） | 是（`diffusion="Wo"`） | 是，三道斜线 + 端墙 |
| 有限长透射 Warburg（tanh） | **否**（`bm_diff` 有实现，未接入采样器） | **否** |
| Gerischer / G | **否** | **否** |

最后两种在 `bm_core.make_model` 里根本不存在，所以不画——图必须跟着方程走，
否则"图"和"拟合方程"不一致，那是比画得难看严重得多的错。

扩散项只有在 `res["diffusion"]` 真的非空、且 `res["s"]` 里真有对应参数时才画。
`diffusion` 是拟合时的开关，不是绘图时的开关：拿一个没加扩散项的旧结果来画，
不会凭空多出一个 Warburg。

## 视觉约定

- 元件符号深灰，参数只用一种强调色（深蓝），不再用红色（红色在诊断图里表示拟合曲线）。
- CPE 用"电容 + 斜线"，并在脚注里解释。
- ζ≠1 画成主线上的虚线框 `Z_HN`，框里只写 `Z_HN`，**不展开成 R/C 网络**。理由：
  一般 HN 阻抗不能由单个 R∥CPE 元件精确表示（那是 ζ=1 的特例）。它可以通过分布式
  弛豫或适当的分布网络来表征，但不应直接解释为有限个理想电路元件组成的唯一拓扑。
  所以图上只声明"这里有一个 HN 阻抗项"，不假装画出了它的内部结构。
- 宽度随 K 自适应，K 越大每个分量的单元越窄，但总宽控制在 ~10 英寸以内。
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyBboxPatch

import bm_core as B
import bm_diff as D

INK = "#2b2b2b"      # 元件
ACC = "#1f4e79"      # 参数（唯一强调色）
MUTE = "#6b6b6b"     # 辅助文字


# --------------------------------------------------------------------------
# 元件符号
# --------------------------------------------------------------------------

def _wire(ax, x0, x1, y=0.0, lw=1.6):
    ax.plot([x0, x1], [y, y], color=INK, lw=lw, solid_capstyle="round", zorder=2)


def _dot(ax, x, y=0.0, r=0.04):
    ax.add_patch(plt.Circle((x, y), r, color=INK, zorder=4))


def _resistor(ax, x0, x1, y=0.0, amp=0.16, n=6):
    step = (x1 - x0) / n
    xs, ys = [x0], [y]
    for i in range(n):
        xs.append(x0 + step * (i + 0.5))
        ys.append(y + (amp if i % 2 == 0 else -amp))
    xs.append(x1)
    ys.append(y)
    ax.plot(xs, ys, color=INK, lw=1.6, solid_joinstyle="miter", zorder=3)


def _inductor(ax, x0, x1, y=0.0, n=3, amp=0.18):
    w = (x1 - x0) / n
    for i in range(n):
        ax.add_patch(Arc((x0 + w * (i + 0.5), y), w, 2 * amp,
                         theta1=0, theta2=180, color=INK, lw=1.6, zorder=3))


def _warburg(ax, x0, x1, y=0.0, blocked=False, n=3, half=0.20):
    """扩散阻抗：三道 45° 斜线 + 两端引线。

    为什么不用锯齿：`_resistor` 已经是锯齿，再画一个锯齿只能靠标注区分，
    而这张图的规矩是"符号本身要能读出来"。三道斜线是 Warburg 族在
    教材里最常见的画法，也不会和电阻/电感混。

    blocked=True 时在右端加一道竖墙 —— 那正是 Wo 与 Ws 的**唯一**区别
    （x=L 处无通量 vs 浓度固定），所以用符号把边界条件画出来，而不是只靠标签。
    """
    span = x1 - x0
    d = span * 0.11
    xa = x0 + span * 0.16
    for i in range(n):
        xi = xa + i * span * 0.19
        ax.plot([xi, xi + d], [y - half, y + half], color=INK, lw=1.9, zorder=3)
    last = xa + (n - 1) * span * 0.19 + d
    _wire(ax, x0, xa, y)
    _wire(ax, last, x1, y)
    if blocked:
        xw = last + 0.09
        ax.plot([xw, xw], [y - half, y + half], color=INK, lw=2.4, zorder=4)
        _wire(ax, xw, x1, y)


def _cpe(ax, x0, x1, y=0.0, gap=0.05, half=0.21, slash=True):
    """恒相位元件：电容 + 一道斜线（IEC 风格）。φ=1 时就是理想电容，不画斜线。"""
    xm = 0.5 * (x0 + x1)
    _wire(ax, x0, xm - gap, y)
    _wire(ax, xm + gap, x1, y)
    for sx in (xm - gap, xm + gap):
        ax.plot([sx, sx], [y - half, y + half], color=INK, lw=2.1, zorder=3)
    if slash:
        ax.plot([xm - 0.15, xm + 0.15], [y - 0.19, y + 0.25],
                color=INK, lw=1.0, zorder=4)


def _hn_box(ax, x0, x1, y0, y1, zeta):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.03,rounding_size=0.09",
        fc="#f7f7f7", ec=INK, ls=(0, (4, 2)), lw=1.1, zorder=3))
    ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1),
            "Z$_{HN}$", ha="center", va="center",
            fontsize=9.0, color=INK, zorder=4)


# --------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------

def _fmt(v: float, unit: str = "", nd: int = 3) -> str:
    """3 位有效数字；太小或太大就用科学计数。"""
    if v == 0:
        return f"0 {unit}".strip()
    a = abs(v)
    if 1e-2 <= a < 1e4:
        return f"{v:.{nd}g} {unit}".strip()
    m, e = f"{v:.{nd - 1}e}".split("e")
    return f"{m}×10$^{{{int(e)}}}$ {unit}".strip()


def component_params(res: dict) -> list[dict]:
    """每个分量的 (R, τ, φ, ζ, Q, f_τ, f_peak)。

    τ 用 `bm_core.match_components` 的匹配结果取中位数——和 `fit_one` 里
    算 f0 用的是同一套排序（按 f0 降序），所以 τ = 1/(2π f0) 与这里一致。
    ζ≠1 时另外算出 DRT 的真实峰位 f_peak，因为那时 f_peak ≠ 1/(2πτ)。
    """
    s = res["s"]
    K = int(res["K"])
    try:
        mm = B.match_components(s)
        o = np.argsort(-np.median(mm["f0"], axis=0))
        tau = np.median(mm["tau"][:, o], axis=0)
        phi = np.median(mm["phi"][:, o], axis=0)
        zeta = np.median(mm["zeta"][:, o], axis=0)
    except Exception:                       # 兜底：用已排好序的 phi + 反推 tau
        tau = 1.0 / (2 * np.pi * np.asarray(res["f0"], float))
        phi = np.asarray(res["phi"], float)
        zeta = np.ones(K)

    out = []
    for k in range(K):
        R = float(res["w"][k])
        Q = float(tau[k] ** phi[k] / R)
        f_tau = 1.0 / (2 * np.pi * tau[k])
        # 峰位统一走 bm_diff.hn_peak_ln_tau（两级网格 + 与暴力解核对到 4e-4）。
        # 这里曾经自己写一份单级网格，步长不同 → 图上标的 f_pk 与 bm_real
        # 报告里的数字对不上，排查了半天才发现是两处各算一遍。
        f_pk = float(np.exp(-D.hn_peak_ln_tau(np.array([tau[k]]),
                                              np.array([phi[k]]),
                                              np.array([zeta[k]]))[0]) / (2 * np.pi))
        out.append(dict(R=R, tau=float(tau[k]), phi=float(phi[k]),
                        zeta=float(zeta[k]), Q=Q, f_tau=float(f_tau),
                        f_pk=f_pk))
    return out


# --------------------------------------------------------------------------
# 主图
# --------------------------------------------------------------------------

def draw(res: dict, include_L: bool, path, name: str = "") -> str:
    K = int(res["K"])
    R_inf = float(res["R_inf"])
    L = res.get("L")
    pars = component_params(res)

    # ---- 自适应横向排布：K 大就压窄单元，总宽别超过 ~10 英寸 ----
    pad = 0.34
    W_L, W_R = 0.95, 1.15
    W_P = float(np.clip(6.3 / K, 1.32, 2.10))
    fs = 7.8 if K <= 3 else 7.1

    # 扩散项：只有拟合时真的加了才画（图跟着方程走，不跟着选项走）
    s = res.get("s") or {}
    diff = str(res.get("diffusion") or "none")
    show_ws = diff in ("Ws", "both") and "Aw" in s
    show_wo = diff in ("Wo", "both") and "Rwo" in s and "tauD" in s
    W_W = 1.30

    x = pad
    pos = {"L": None, "R": None, "P": [], "W": []}
    if include_L and L is not None:
        pos["L"] = (x, x + W_L)
        x += W_L
    pos["R"] = (x, x + W_R)
    x += W_R
    for _ in range(K):
        pos["P"].append((x, x + W_P))
        x += W_P
    for _ in range(int(show_ws) + int(show_wo)):
        pos["W"].append((x, x + W_W))
        x += W_W
    x_end = x + pad

    rail = 0.44
    fig, ax = plt.subplots(figsize=(1.02 * x_end, 3.05))
    ax.set_xlim(0, x_end)
    ax.set_ylim(-1.95, 0.90)
    ax.axis("off")

    # ---- 串联臂 ----
    _dot(ax, pad * 0.5)
    if pos["L"]:
        x0, x1 = pos["L"]
        _wire(ax, pad * 0.5, x0 + 0.10)
        _inductor(ax, x0 + 0.10, x1 - 0.10)
        _wire(ax, x1 - 0.10, x1)
        ax.text(0.5 * (x0 + x1), -0.32,
                f"L = {_fmt(L * 1e6, 'µH')}", ha="center", va="top",
                fontsize=fs, color=ACC)

    x0, x1 = pos["R"]
    _wire(ax, x0, x0 + 0.16)
    _resistor(ax, x0 + 0.16, x1 - 0.16)
    _wire(ax, x1 - 0.16, x1)
    ax.text(0.5 * (x0 + x1), -0.32, f"R∞ = {_fmt(R_inf, 'Ω')}",
            ha="center", va="top", fontsize=fs, color=ACC)

    # ---- 每个分量 ----
    for k, (xa, xb) in enumerate(pos["P"]):
        p = pars[k]
        prev = pos["R"][1] if k == 0 else pos["P"][k - 1][1]
        _wire(ax, prev, xa)
        xm = 0.5 * (xa + xb)
        hn = abs(p["zeta"] - 1.0) > 1e-6

        if hn:
            # HN 是二端元件，串在主干上，不画成并联支路
            _wire(ax, xa, xa + 0.20)
            _hn_box(ax, xa + 0.20, xb - 0.20, -0.32, 0.32, p["zeta"])
            _wire(ax, xb - 0.20, xb)
            ax.text(xm, 0.38, f"ζ = {p['zeta']:.2f}", ha="center", va="bottom",
                    fontsize=fs, color=ACC)
            ax.text(xm, -0.50, f"R$_{k + 1}$ = {_fmt(p['R'], 'Ω')}",
                    ha="center", va="top", fontsize=fs, color=ACC)
            ax.text(xm, -0.78, f"φ$_{k + 1}$ = {p['phi']:.3f}",
                    ha="center", va="top", fontsize=fs, color=ACC)
        else:
            ax.plot([xa, xa], [-rail, rail], color=INK, lw=1.6, zorder=2)
            ax.plot([xb, xb], [-rail, rail], color=INK, lw=1.6, zorder=2)
            for yy in (rail, -rail):
                _wire(ax, xa + 0.02, xa + 0.28, yy)
                _wire(ax, xb - 0.28, xb - 0.02, yy)
            _resistor(ax, xa + 0.28, xb - 0.28, rail)
            _cpe(ax, xa + 0.28, xb - 0.28, -rail,
                 slash=abs(p["phi"] - 1.0) >= 1e-3)

            # 参数紧贴元件：R 在电阻上方，Q/φ 在 CPE 下方
            ax.text(xm, rail + 0.24, f"R$_{k + 1}$ = {_fmt(p['R'], 'Ω')}",
                    ha="center", va="bottom", fontsize=fs, color=ACC)
            if abs(p["phi"] - 1.0) < 1e-3:
                ax.text(xm, -rail - 0.34,
                        f"C$_{k + 1}$ = {_fmt(p['Q'], 'F')}",
                        ha="center", va="top", fontsize=fs, color=ACC)
            else:
                ax.text(xm, -rail - 0.34,
                        f"Q$_{k + 1}$ = {_fmt(p['Q'])}",
                        ha="center", va="top", fontsize=fs, color=ACC)
                ax.text(xm, -rail - 0.60, f"φ$_{k + 1}$ = {p['phi']:.3f}",
                        ha="center", va="top", fontsize=fs, color=ACC)

        # 底部：τ 与频率。ζ≠1 时 f_τ 与真实峰位分开写。
        if hn:
            ax.text(xm, -1.30, f"τ$_{k + 1}$ = {_fmt(p['tau'], 's')}\n"
                               f"f$_τ$ = {_fmt(p['f_tau'], 'Hz')}\n"
                               f"f$_{{pk}}$ = {_fmt(p['f_pk'], 'Hz')}",
                    ha="center", va="top", fontsize=fs - 0.4, color=ACC,
                    linespacing=1.4)
        else:
            ax.text(xm, -1.30, f"τ$_{k + 1}$ = {_fmt(p['tau'], 's')}\n"
                               f"f$_{k + 1}$ = {_fmt(p['f_tau'], 'Hz')}",
                    ha="center", va="top", fontsize=fs - 0.4, color=ACC,
                    linespacing=1.4)

    # ---- 扩散项：串在主干上，与 K 个 HN 项同级 ----
    prev = pos["P"][-1][1]
    for j, (xa, xb) in enumerate(pos["W"]):
        _wire(ax, prev, xa)
        xm = 0.5 * (xa + xb)
        if show_ws and j == 0:
            _warburg(ax, xa, xb, 0.0, blocked=False)
            a = np.asarray(s["Aw"], float)
            ax.text(xm, 0.34, "Z$_{Ws}$", ha="center", va="bottom",
                    fontsize=fs + 0.4, color=INK)
            ax.text(xm, -0.36, "A$_W$ = " + _fmt(float(np.median(a)),
                                                  "Ω·s$^{-1/2}$"),
                    ha="center", va="top", fontsize=fs, color=ACC)
            ax.text(xm, -0.62,
                    f"[{_fmt(float(np.percentile(a, 5)))}, "
                    f"{_fmt(float(np.percentile(a, 95)))}]",
                    ha="center", va="top", fontsize=fs - 1.0, color=MUTE)
        else:
            _warburg(ax, xa, xb, 0.0, blocked=True)
            rw = np.asarray(s["Rwo"], float)
            td = np.asarray(s["tauD"], float)
            ax.text(xm, 0.34, "Z$_{Wo}$", ha="center", va="bottom",
                    fontsize=fs + 0.4, color=INK)
            ax.text(xm, -0.36, "R$_W$ = " + _fmt(float(np.median(rw)), "Ω"),
                    ha="center", va="top", fontsize=fs, color=ACC)
            ax.text(xm, -0.62, "τ$_D$ = " + _fmt(float(np.median(td)), "s"),
                    ha="center", va="top", fontsize=fs, color=ACC)
            fD = 1.0 / (2 * np.pi * float(np.median(td)))
            ax.text(xm, -0.88, "f$_D$ = " + _fmt(fD, "Hz"),
                    ha="center", va="top", fontsize=fs, color=ACC)
        prev = xb

    _wire(ax, prev, x_end - pad * 0.5)
    _dot(ax, x_end - pad * 0.5)

    # ---- 标题 ----
    head = "Equivalent circuit (posterior median)"
    if name:
        head = f"{name} — " + head
    any_hn = any(abs(p["zeta"] - 1.0) > 1e-6 for p in pars)
    tag = ""
    if show_ws or show_wo:
        tag = "    + " + "+".join(n for n, f in (("Ws", show_ws), ("Wo", show_wo)) if f)
    ax.set_title(f"{head}    K = {K}"
                 + ("    ζ free" if any_hn else "") + tag,
                 fontsize=10.5, pad=12)

    # ---- 脚注：只留必要说明，长公式不进图 ----
    notes = ["Q has units S·s$^φ$;  C = Q only when φ = 1."]
    notes.append("CPE (capacitor + slash): Z = 1/[Q(jω)$^φ$].")
    if any_hn:
        notes.append("Z$_{HN}$: a general HN impedance is not exactly one R∥CPE "
                     "(that holds only at ζ = 1); it describes distributed relaxation "
                     "with no unique finite ideal-element topology, so the box is not "
                     "expanded and no Q is quoted.")
    if show_ws:
        notes.append("Z$_{Ws}$ = A$_W$(jω)$^{-1/2}$: semi-infinite diffusion "
                     "(3 slashes = distributed, not a lumped element).")
    if show_wo:
        # 两个渐近区定住的是**不同**的组合，脚注不能只写一半——写"只剩 C_D"
        # 在 f_D ≪ f_min 时是反的（那时看到的是 Warburg 尾，不是电容区）。
        notes.append("Z$_{Wo}$ = R$_W$·coth√(jωτ$_D$)/√(jωτ$_D$): finite-length, "
                     "blocking wall at x = L (drawn as the end bar). Low-frequency "
                     "limit R$_W$/3 + 1/(jωC$_D$): if f$_D$ ≫ f$_{max}$ only "
                     "C$_D$ = τ$_D$/R$_W$ is identifiable; if f$_D$ ≪ f$_{min}$ only "
                     "A$_W$ = R$_W$/√τ$_D$ is. With f$_D$ inside the window, "
                     "C$_D$ is pinned (±0.3% in the B2 test) but R$_W$ and τ$_D$ "
                     "still slide together along R$_W$ ∝ τ$_D$.")
    # 脚注按画布宽度折行：一行太长会让 tight_layout 把坐标轴压扁
    import textwrap
    wrap = max(56, int(x_end * 13.5))
    note_txt = "\n".join(textwrap.fill(n, wrap) for n in notes)
    ax.text(0.5, -0.03, note_txt, transform=ax.transAxes,
            ha="center", va="top", fontsize=6.9, color=MUTE, linespacing=1.6)

    # 不用 tight_layout：它会把超出坐标轴的文字算进去，反过来把电路压窄。
    # 直接固定边距，超出的部分交给 bbox_inches="tight" 去扩展画布。
    fig.subplots_adjust(left=0.004, right=0.996, top=0.84, bottom=0.05)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)
