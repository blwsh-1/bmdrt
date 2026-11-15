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

#: 候选曲线配色（数据一律黑，候选按顺序取色）
COLORS = ("#c0392b", "#2e86c1", "#27ae60", "#8e44ad", "#d68910", "#16a085")


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


def plot_fit(path, f, Z, curves, *, title: str = "", meta: str = "",
             dpi: int = 140, ref_pct: float = 2.0):
    """画 2×3：Nyquist / Bode 幅值 / Bode 相位 / 相对残差 / Re 残差 / Im 残差。

    `curves` 每项：`dict(label=str, Zc=complex ndarray, color=str|None,
    style="-"|"--", lw=float)`。空列表 ⇒ 只画数据。
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

    fig, ax = plt.subplots(2, 3, figsize=(16.8, 9.4))
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
    a.legend(fontsize=7.5, frameon=False, loc="best")
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
                 color=c["color"], label=f"{c['label']}  rms={c.get('rms', float('nan')):.2f}%")
    a.axhline(ref_pct / 100.0, color="#7f8c8d", lw=1.0, ls="--",
              label=L(f"{ref_pct:g}% 参考", f"{ref_pct:g}% ref"))
    a.set_xlabel(L("频率 / Hz", "f / Hz"))
    a.set_ylabel(L("|Z_model−Z| / |Z|", "rel. residual"))
    a.set_title(L("④ 相对残差", "(4) relative residual"), fontsize=11)
    a.grid(alpha=.3, ls=":", which="both")
    a.legend(fontsize=8, frameon=False)

    for j, (comp, name) in enumerate((("real", L("实部", "Re")),
                                      ("imag", L("虚部", "Im")))):
        a = ax[1, 1 + j]
        res = (Z.real if comp == "real" else Z.imag)
        a.semilogx(f, np.zeros_like(f), color="k", lw=.6, alpha=.4)
        for c in curves:
            r = (c["Zc"].real if comp == "real" else c["Zc"].imag) - res
            a.semilogx(f, r, c["style"], lw=c["lw"], color=c["color"],
                       label=c["label"])
        a.set_xlabel(L("频率 / Hz", "f / Hz"))
        a.set_ylabel(L(f"残差 {name} / Ω", f"residual {name} / ohm"))
        a.set_title(L(f"{'⑤' if j == 0 else '⑥'} 残差 {name}",
                      f"({'5' if j == 0 else '6'}) residual {name}"),
                    fontsize=11)
        a.grid(alpha=.3, ls=":", which="both")
        if j == 1:
            a.legend(fontsize=8, frameon=False)

    sup = title or L("拟合结果", "fit")
    fig.suptitle(sup, fontsize=13, y=0.985)
    if meta:
        n_lines = meta.count("\n") + 1
        fig.text(0.5, 0.010, meta, ha="center", va="bottom", fontsize=8.2,
                 color="#34495e", linespacing=1.5)
        fig.tight_layout(rect=(0, 0.018 + 0.020 * n_lines, 1, 0.955))
    else:
        fig.tight_layout(rect=(0, 0.02, 1, 0.955))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
