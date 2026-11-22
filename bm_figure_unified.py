"""统一拟合图：一条谱 + 任意多个候选曲线 → 同一张 6 子图。

为什么单独一层：现有两套绘图层互不兼容——`bm_real.plot_real` 绑死 voigt 的
DRT 预览/PPC/分段残差（6 子图，但只认 voigt 的中间量），`nested_fit.write_plot`
只认 nested 的三联图。`bmdrt compare` 要把 voigt 与 nested 画在同一张图上并列，
必须有一层只依赖 `(f, Z, curves)` 的画法。

本层**不改**那两个入口：`bmdrt auto` 仍用 `bm_real.plot_real`、`nested_fit.py`
仍用自己的三联图。合并它们会改变两条已有 CLI 的输出，属于另一个决定。

用法：
    from bm_figure_unified import plot_fit
    plot_fit("x.png", f, Z, [dict(label="voigt K=2", Zc=Zc, color="#c0392b")],
             title="...", meta="rms=1.8%  acf=0.90")
"""
from __future__ import annotations

import numpy as np

#: 候选曲线配色（数据一律黑，候选按顺序取色）。
#: 12 色（2026-11-20）：三族 × 三种扩散 + 各自基线最多 12 个候选，
#: 原来 6 色会循环复用 ⇒ 同图两条不同候选同色，只能靠图例文字区分。
#: 刻意避开 `#7f8c8d`（2% 参考线）与 `#1f2d3d`（实测点）。
COLORS = ("#c0392b", "#2e86c1", "#27ae60", "#8e44ad", "#d68910", "#16a085",
          "#c2185b", "#2c3e50", "#6d4c41", "#3f51b5", "#00838f", "#827717")


def _setup_cn():
    """中文字体：有就用中文标签，没有就退回英文，避免满屏方框。"""
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    cn = any(x.name in ("Microsoft YaHei", "SimHei")
             for x in font_manager.fontManager.ttflist)
    if cn:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                           "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    return (lambda zh, en: zh) if cn else (lambda zh, en: en)


def _resid_label(c: dict) -> str:
    """残差子图的图例：`label` 有 rms 就带上，没有（None/缺失）就不加，
    避免打出 `rms=nan%` 这种没有信息量的后缀。"""
    r = c.get("rms")
    return c["label"] if r is None else f"{c['label']}  rms={r:.2f}%"


def plot_fit(path, f, Z, curves, *, title: str = "", meta: str = "",
             dpi: int = 140, ref_pct: float = 2.0,
             sigma_re=None, sigma_im=None):
    """画 2×3：Nyquist / Bode 幅值 / Bode 相位 / 相对残差 / Re 残差 / Im 残差。

    `curves` 每项：`dict(label=str, Zc=complex ndarray, color=str|None,
    style="-"|"--", lw=float)`。空列表 ⇒ 只画数据。

    `sigma_re` / `sigma_im`（可选）：在残差实部/虚部子图各画一条 **±σ 点线带**
    （红色点线，与 voigt 的 `bm_real.plot_real` 残差图同风格）。标量或逐点
    数组都行；`None` 就不画。`nested_fit` 把逐点 `2%·|Z|` 传进来，所以带会随
    频率收窄/放宽——那是"逐点相对 σ"的真实形状，不是常数带。

    **多候选自适应**（2026-11-20）：三族 × 三种扩散最多 12 条曲线，固定
    9.4 in 高 + 单列图例会溢出画布、压住标题与坐标轴。这里按曲线数把
    图例切两列、字号缩小，并按 `meta` 行数加高画布——清单一行不少，
    只是不再挤扁子图。
    """
    import matplotlib.pyplot as plt

    L = _setup_cn()
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    curves = [dict(c) for c in (curves or [])]
    for i, c in enumerate(curves):
        c.setdefault("color", COLORS[i % len(COLORS)])
        c.setdefault("style", "-")
        c.setdefault("lw", 1.7)
        c["Zc"] = np.asarray(c["Zc"], complex)

    n = len(curves)
    ncol = 2 if n > 6 else 1
    lsz = 6.4 if n > 6 else (7.0 if n > 3 else 7.5)
    # 画布高度随底部清单增长：fontsize 8.2 × linespacing 1.5 ≈ 0.175 in/行。
    # 之前按 0.135 估，27 行清单会压住子图的横轴标签（实测截图确认）。
    n_meta = (meta.count("\n") + 1) if meta else 0
    meta_in = max(0.0, n_meta * 0.175 + 0.35) if meta else 0.0
    fig_h = 9.4 + max(0.0, meta_in - 1.5)

    fig, ax = plt.subplots(2, 3, figsize=(16.8, fig_h))
    a = ax[0, 0]
    a.plot(Z.real, -Z.imag, "o", ms=4.6, mfc="none", mec="#1f2d3d", mew=1.0,
           label=L("实测", "data"), zorder=3)
    for c in curves:
        a.plot(c["Zc"].real, -c["Zc"].imag, c["style"], lw=c["lw"],
               color=c["color"], label=c["label"], zorder=2)
    a.set_xlabel(L("Z′ / Ω", "Z' / ohm"))
    a.set_ylabel(L("−Z″ / Ω", "-Z'' / ohm"))
    a.set_title(L("① Nyquist", "(1) Nyquist"), fontsize=11)
    a.grid(alpha=.3, ls=":")
    a.legend(fontsize=lsz, frameon=True, framealpha=.85, ncol=ncol,
             loc="best", borderpad=.3, labelspacing=.28, columnspacing=.8)
    # 不用 `set_aspect("equal")`：`adjustable="datalim"` 会把坐标轴撑到数据范围之外
    # （腐蚀谱纵轴被拉到 -2e5…5e5，实际只有 0…3e5），肉眼上像是数据跑飞了。

    a = ax[0, 1]
    a.loglog(f, np.abs(Z), "o", ms=4.2, mfc="none", mec="#1f2d3d", mew=1.0,
             label=L("实测", "data"))
    for c in curves:
        a.loglog(f, np.abs(c["Zc"]), c["style"], lw=c["lw"], color=c["color"],
                 label=c["label"])
    a.set_xlabel(L("频率 / Hz", "f / Hz"))
    a.set_ylabel(L("|Z| / Ω", "|Z| / ohm"))
    a.set_title(L("② Bode 幅值", "(2) Bode |Z|"), fontsize=11)
    a.grid(alpha=.3, ls=":", which="both")

    a = ax[0, 2]
    a.semilogx(f, np.degrees(np.angle(Z)), "o", ms=4.2, mfc="none",
               mec="#1f2d3d", mew=1.0, label=L("实测", "data"))
    for c in curves:
        a.semilogx(f, np.degrees(np.angle(c["Zc"])), c["style"], lw=c["lw"],
                   color=c["color"], label=c["label"])
    a.axhline(0, color="k", lw=.6, alpha=.4)
    a.set_xlabel(L("频率 / Hz", "f / Hz"))
    a.set_ylabel(L("相位 / °", "phase / deg"))
    a.set_title(L("③ Bode 相位", "(3) Bode phase"), fontsize=11)
    a.grid(alpha=.3, ls=":", which="both")

    a = ax[1, 0]
    for c in curves:
        rel = np.abs(c["Zc"] - Z) / np.abs(Z)
        a.loglog(f, np.maximum(rel, 1e-6), c["style"], lw=c["lw"],
                 color=c["color"], label=_resid_label(c))
    a.axhline(ref_pct / 100.0, color="#7f8c8d", lw=1.0, ls="--",
              label=L(f"{ref_pct:g}% 参考", f"{ref_pct:g}% ref"))
    a.set_xlabel(L("频率 / Hz", "f / Hz"))
    a.set_ylabel(L("|Z_model−Z| / |Z|", "rel. residual"))
    a.set_title(L("④ 相对残差", "(4) relative residual"), fontsize=11)
    a.grid(alpha=.3, ls=":", which="both")
    a.legend(fontsize=lsz if n <= 6 else 5.9, frameon=True, framealpha=.85,
             ncol=ncol, loc="best", borderpad=.3, labelspacing=.26,
             columnspacing=.8)

    for j, (comp, name) in enumerate((("real", L("实部", "Re")),
                                      ("imag", L("虚部", "Im")))):
        a = ax[1, 1 + j]
        res = (Z.real if comp == "real" else Z.imag)
        a.semilogx(f, np.zeros_like(f), color="k", lw=.6, alpha=.4)
        for c in curves:
            r = (c["Zc"].real if comp == "real" else c["Zc"].imag) - res
            a.semilogx(f, r, c["style"], lw=c["lw"], color=c["color"],
                       label=c["label"])
        # ±σ 点线带（voigt 风格）：标量或逐点数组都广播到 f 长度。
        sg = sigma_re if comp == "real" else sigma_im
        if sg is not None:
            sg = np.broadcast_to(np.asarray(sg, float), f.shape)
            a.semilogx(f, +sg, ":", color="#c0392b", lw=1.1, alpha=.65,
                       label=L("±σ", "±σ"))
            a.semilogx(f, -sg, ":", color="#c0392b", lw=1.1, alpha=.65)
        a.set_xlabel(L("频率 / Hz", "f / Hz"))
        a.set_ylabel(L(f"残差 {name} / Ω", f"residual {name} / ohm"))
        a.set_title(L(f"{'⑤' if j == 0 else '⑥'} 残差 {name}",
                      f"({'5' if j == 0 else '6'}) residual {name}"),
                    fontsize=11)
        a.grid(alpha=.3, ls=":", which="both")
        # 候选多时不再重复 12 条图例：①（两列图例）+ ④ + 底部清单已能定身份，
        # ⑤⑥ 的小面板放图例会盖住残差曲线。
        if n <= 6:
            a.legend(fontsize=8, frameon=False)

    sup = title or L("拟合结果", "fit")
    fig.suptitle(sup, fontsize=13, y=0.985)
    if meta:
        fig.text(0.5, 0.006, meta, ha="center", va="bottom", fontsize=8.2,
                 color="#34495e", linespacing=1.5)
        fig.tight_layout(rect=(0, meta_in / fig_h, 1, 0.955))
    else:
        fig.tight_layout(rect=(0, 0.02, 1, 0.955))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
