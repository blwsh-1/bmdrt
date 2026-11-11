"""bm_figure.py —— 出版级静态图（matplotlib），与交互面板共用同一份数据。

分工（设计文档 §5）
-------------------
* `bm_dash.py`  → 交互、看数、排查，不追求印刷精度；
* `bm_figure.py`→ 固定 mm 尺寸、矢量、字体可嵌入，用于投稿与汇报。

两者**都只读** `bm_result.AnalysisResult`，同一组数组、同一套分位，
所以「图上的曲线」和「导出的 CSV」不可能对不上——这是可复现性的前提，
不是靠人工核对来保证的。

风格预设只约束**排版语言**（字体/线宽/留白/配色），
**不是**期刊合规声明：真正投稿还要按目标刊当期要求核对字号、线宽、
色彩模式和分辨率。

用法
----
    python bm_figure.py <sample_dir> --theme acs --panels nyquist,drt,residual
    python bm_figure.py <sample_dir> --all            # png/svg/pdf/tiff 全出
    python bm_figure.py <sample_dir> --spec spec.json # 用保存的图形配置重画
    python bm_figure.py --dump-spec                   # 打印默认 spec
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager                  # noqa: E402
from matplotlib.ticker import (LogLocator, NullFormatter,       # noqa: E402
                               FuncFormatter)

import bm_result as R                                # noqa: E402

MM = 1.0 / 25.4                                      # mm → inch

# --------------------------------------------------------------------------
# 风格预设：只描述排版语言
# --------------------------------------------------------------------------
THEMES = {
    "minimal": dict(
        font="DejaVu Sans", font_size=9.0, axis_lw=0.8, tick_w=0.8,
        tick_len=3.0, grid=False, minor=True,
        data="#333333", fit="#1f6fb4", band="#1f6fb4", pred="#8c8c8c",
        drt="#7b3fa0", resid_re="#1f6fb4", resid_im="#d1495b",
        legend_frame=False, spine_top=False, spine_right=False,
    ),
    "nature": dict(                                     # 简洁、留白足
        font="Arial", font_size=7.0, axis_lw=0.6, tick_w=0.6,
        tick_len=2.5, grid=False, minor=True,
        data="#222222", fit="#0b6fa4", band="#0b6fa4", pred="#9a9a9a",
        drt="#6a3d9a", resid_re="#0b6fa4", resid_im="#b2182b",
        legend_frame=False, spine_top=False, spine_right=False,
    ),
    "acs": dict(                                        # 电化学/材料双栏
        font="Arial", font_size=8.0, axis_lw=0.9, tick_w=0.9,
        tick_len=3.0, grid=False, minor=True,
        data="#1a1a1a", fit="#0055a4", band="#0055a4", pred="#8f8f8f",
        drt="#7a3b8f", resid_re="#0055a4", resid_im="#c0392b",
        legend_frame=False, spine_top=False, spine_right=False,
    ),
    "angew": dict(                                      # 强调层次
        font="Arial", font_size=8.0, axis_lw=1.0, tick_w=1.0,
        tick_len=3.4, grid=True, minor=False,
        data="#111111", fit="#c0392b", band="#c0392b", pred="#999999",
        drt="#1f4e79", resid_re="#c0392b", resid_im="#1f4e79",
        legend_frame=True, spine_top=True, spine_right=True,
    ),
    "presentation": dict(                               # 组会/课堂
        font="Arial", font_size=13.0, axis_lw=1.4, tick_w=1.4,
        tick_len=5.0, grid=True, minor=False,
        data="#111111", fit="#d94801", band="#d94801", pred="#888888",
        drt="#5b2c83", resid_re="#d94801", resid_im="#2171b5",
        legend_frame=True, spine_top=True, spine_right=True,
    ),
}

LAYOUTS = {
    "single": (1, 1),
    "pair": (1, 2),
    "triple": (1, 3),
    "quad": (2, 2),
    "strip": (3, 1),
}

PANELS = ("nyquist", "drt", "residual", "bode")

DEFAULT_SPEC = dict(
    spec_version=R.SPEC_VERSION,
    theme="acs",
    font=dict(family=None, size=None),               # None = 跟主题走
    canvas=dict(width_mm=180.0, height_mm=110.0),
    layout="quad",
    panels=["nyquist", "drt", "residual", "bode"],
    dpi=600,
    labels="en",                                     # en = 投稿；zh = 中文汇报
    resid_mode="normalized",                         # normalized | absolute
    show_predictive=False,
    show_legend=True,
    formats=["png"],
    titles=None,                                     # None = 不写标题（期刊习惯）
    allow_font_fallback=False,
    # 逐系列覆盖颜色与线宽。None = 跟主题走。
    # 只放开**颜色与线宽**，不放开标记形状：实验数据永远空心圆点、
    # 拟合永远实线、Re/Im 永远不同标记——颜色可以换，编码规则不许换，
    # 否则同一条谱的两张图会被读成两种数据。
    style=dict(data=None, fit=None, band=None, drt=None,
               resid_re=None, resid_im=None, lw_data=None, lw_fit=None),
)

STYLE_COLORS = ("data", "fit", "band", "drt", "resid_re", "resid_im")
STYLE_LW = ("lw_data", "lw_fit")
_HEX = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


class SpecError(ValueError):
    pass


def _check_block(block, allowed, path):
    """嵌套块也逐键校验。`{"font": {"famliy": "Arial"}}` 必须报错——
    静默接受会让人以为字体改了，实际还是默认字体，投出去才发现。"""
    if block is None:
        return
    if not isinstance(block, dict):
        raise SpecError(f"{path} 必须是对象，收到 {type(block).__name__}")
    unknown = set(block) - set(allowed)
    if unknown:
        raise SpecError(f"{path} 里有未知键 {sorted(unknown)}；允许 {sorted(allowed)}")


def validate_spec(spec: dict) -> dict:
    """校验并补默认值。**未知键直接报错**——静默忽略会让人以为设置生效了。

    顶层与 font / canvas / style / titles 各块**逐层**校验，
    不做 `dict.update` 蒙混过关。
    """
    if not isinstance(spec, dict):
        raise SpecError("spec 必须是 dict")
    unknown = set(spec) - set(DEFAULT_SPEC)
    if unknown:
        raise SpecError(f"未知配置键：{sorted(unknown)}；"
                        f"允许的是 {sorted(DEFAULT_SPEC)}")
    out = dict(DEFAULT_SPEC)
    blocks = ("font", "canvas", "style")
    out.update({k: v for k, v in spec.items() if k not in blocks})
    for blk in blocks:
        _check_block(spec.get(blk), DEFAULT_SPEC[blk], blk)
        merged = dict(DEFAULT_SPEC[blk])
        merged.update(spec.get(blk) or {})
        out[blk] = merged

    # style 只认颜色与线宽：值必须是 #rgb/#rrggbb，否则报错不猜
    for k in STYLE_COLORS:
        v = out["style"].get(k)
        if v is not None and not _HEX.match(str(v)):
            raise SpecError(f"style.{k} 必须是 #rrggbb 形式的颜色，收到 {v!r}")
    for k in STYLE_LW:
        v = out["style"].get(k)
        if v is not None:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise SpecError(f"style.{k} 必须是数字，收到 {v!r}") from None
            if not (0.1 <= fv <= 8.0):
                raise SpecError(f"style.{k} 超出 0.1–8.0 pt：{fv}")
            out["style"][k] = fv

    if out["titles"] is not None:
        _check_block(out["titles"], PANELS, "titles")

    if out["theme"] not in THEMES:
        raise SpecError(f"未知主题 {out['theme']!r}；可选 {sorted(THEMES)}")
    if out["layout"] not in LAYOUTS:
        raise SpecError(f"未知布局 {out['layout']!r}；可选 {sorted(LAYOUTS)}")
    bad = [p for p in out["panels"] if p not in PANELS]
    if bad:
        raise SpecError(f"未知面板 {bad}；可选 {list(PANELS)}")
    if not out["panels"]:
        raise SpecError("panels 不能为空")
    rows, cols = LAYOUTS[out["layout"]]
    if rows * cols < len(out["panels"]):
        raise SpecError(f"布局 {out['layout']}({rows}×{cols}) 放不下 "
                        f"{len(out['panels'])} 个面板")
    if out["labels"] not in ("en", "zh"):
        raise SpecError("labels 只能是 'en' 或 'zh'")
    if out["resid_mode"] not in ("normalized", "absolute"):
        raise SpecError("resid_mode 只能是 'normalized' 或 'absolute'")
    for k in ("width_mm", "height_mm"):
        if not (10.0 <= float(out["canvas"][k]) <= 500.0):
            raise SpecError(f"canvas.{k} 超出 10–500 mm：{out['canvas'][k]}")
    if int(out["dpi"]) < 72:
        raise SpecError("dpi 至少 72")
    return out


def check_fonts(spec: dict) -> list:
    """返回缺失字体清单（不静默替换）。"""
    th = THEMES[spec["theme"]]
    want = [spec["font"]["family"] or th["font"]]
    if spec["labels"] == "zh":
        want.append("SimHei")                        # 中文标签至少要一个
    missing = []
    for fam in want:
        try:
            font_manager.findfont(fam, fallback_to_default=False)
        except Exception:
            missing.append(fam)
    return missing


def _rc(spec: dict):
    th = THEMES[spec["theme"]]
    fam = spec["font"]["family"] or th["font"]
    size = float(spec["font"]["size"] or th["font_size"])
    if spec["labels"] == "zh" and fam in ("Arial", "Helvetica"):
        fam = "SimHei"
    matplotlib.rcParams.update({
        "font.family": fam,
        "font.size": size,
        "axes.labelsize": size,
        "xtick.labelsize": size - 0.5,
        "ytick.labelsize": size - 0.5,
        "legend.fontsize": size - 0.8,
        "axes.linewidth": th["axis_lw"],
        "xtick.major.width": th["tick_w"],
        "ytick.major.width": th["tick_w"],
        "xtick.major.size": th["tick_len"],
        "ytick.major.size": th["tick_len"],
        "xtick.minor.width": th["tick_w"] * 0.7,
        "ytick.minor.width": th["tick_w"] * 0.7,
        "xtick.minor.size": th["tick_len"] * 0.55,
        "ytick.minor.size": th["tick_len"] * 0.55,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": th["spine_top"],
        "axes.spines.right": th["spine_right"],
        "legend.frameon": th["legend_frame"],
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#999999",
        "legend.handlelength": 1.6,
        "figure.dpi": 120,
        "savefig.dpi": int(spec["dpi"]),
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,                          # TrueType 可编辑
        "ps.fonttype": 42,
        "svg.fonttype": "none",                      # 文本保留为文本
    })
    return th


# --------------------------------------------------------------------------
# 标签
# --------------------------------------------------------------------------
_LAB = {
    "en": dict(f="f / Hz", zre=r"$Z'$ / $\Omega$", zim=r"$-Z''$ / $\Omega$",
               gam=r"$\gamma$ / $\Omega$", nres=r"residual / $\sigma$",
               res=r"residual / $\Omega$", mag=r"$|Z|$ / $\Omega$",
               ph="phase / deg", tau=r"$\tau$ / s",
               data="data", fit="posterior median",
               band="90% CI (marginal)",
               pred="90% predictive", zero="0"),
    "zh": dict(f="频率 f / Hz", zre=r"实部 $Z'$ / $\Omega$",
               zim=r"虚部 $-Z''$ / $\Omega$", gam=r"$\gamma$ / $\Omega$",
               nres=r"残差 / $\sigma$", res=r"残差 / $\Omega$",
               mag=r"$|Z|$ / $\Omega$", ph="相位 / deg", tau=r"$\tau$ / s",
               data="实验数据", fit="后验中位", band="90% 区间（边际）",
               pred="90% 预测区间", zero="0"),
}


def _lab(spec):
    return _LAB[spec["labels"]]


def _logx(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _sty(ax, spec, th):
    if th["grid"]:
        ax.grid(alpha=0.25, which="major", lw=0.5)
    if th["minor"]:
        ax.minorticks_on()
        ax.tick_params(which="minor", length=th["tick_len"] * 0.55,
                       width=th["tick_w"] * 0.7)


# --------------------------------------------------------------------------
# 面板
# --------------------------------------------------------------------------
def panel_nyquist(ax, res, spec, th):
    L = _lab(spec)
    c = res["curves"]
    zr, zi = np.array(c["zre"]), np.array(c["zim"])
    fr, fi = np.array(c["fre"]), np.array(c["fim"])
    # 实验数据：空心圆点（全项目固定编码）
    ax.plot(zr, -zi, "o", ms=3.6, mfc="none", mec=th["data"],
            mew=0.9, label=L["data"], zorder=3)
    ax.plot(fr, -fi, "-", lw=float(th.get("lw_fit", 1.3)), color=th["fit"],
            label=L["fit"], zorder=4)
    if res.get("has_band"):
        # 区间的 Re/Im 是**分别**取的边缘分位，复平面上是矩形包络，不是一条带。
        # 早先按 re_lo→re_hi→im_hi 连成封闭多边形，那圈面积没有概率含义，
        # 在拟合很差的谱上会变成横贯全图的色带。这里画逐点十字误差棒。
        ax.errorbar(fr, -fi,
                    xerr=[fr - np.array(c["re_lo"]),
                          np.array(c["re_hi"]) - fr],
                    yerr=[np.array(c["im_hi"]) - fi, fi - np.array(c["im_lo"])],
                    fmt="none", ecolor=th["band"], elinewidth=0.6,
                    alpha=0.45, capsize=0, zorder=2, label=L["band"])
    if spec["show_predictive"] and res.get("has_pred"):
        ax.plot(c["pre_lo"], -np.array(c["pim_lo"]), ":", lw=0.8,
                color=th["pred"], label=L["pred"])
        ax.plot(c["pre_hi"], -np.array(c["pim_hi"]), ":", lw=0.8,
                color=th["pred"])
    ax.set_xlabel(L["zre"])
    ax.set_ylabel(L["zim"])
    ax.set_aspect("equal", adjustable="datalim")
    _sty(ax, spec, th)


def panel_drt(ax, res, spec, th):
    L = _lab(spec)
    d = res.get("drt")
    if not d:
        ax.text(0.5, 0.5, "no posterior DRT", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#999999")
        return
    f, med = np.array(d["f"]), np.array(d["med"])
    ax.fill_between(f, d["lo"], d["hi"], color=th["drt"], alpha=0.15, lw=0)
    ax.plot(f, med, "-", lw=float(th.get("lw_fit", 1.3)), color=th["drt"])
    _logx(ax)
    ax.set_xlabel(L["f"])
    ax.set_ylabel(L["gam"])
    ymax = float(np.nanmax(med)) if np.isfinite(np.nanmax(med)) else 1.0
    for comp in res.get("components") or []:
        f0 = comp.get("f0")
        if not f0 or not np.isfinite(f0):
            continue
        if not (f[0] <= f0 <= f[-1]):
            continue
        ax.axvline(f0, color="#666666", ls=":", lw=0.7, alpha=0.8)
        ax.annotate(f"#{comp['idx']}", xy=(f0, ymax * 0.94),
                    xytext=(1.5, 0), textcoords="offset points",
                    fontsize=6.5, color="#555555", va="top")
    _sty(ax, spec, th)


def panel_residual(ax, res, spec, th):
    L = _lab(spec)
    c = res["curves"]
    f = np.array(c["f"])
    norm = spec["resid_mode"] == "normalized"
    re_key, im_key = ("nre", "nim") if norm else ("rre", "rim")
    # 标准化残差可能只有一半有效（σ 缺失或为 0）。两种量纲不能画在一根轴上，
    # 所以只要有一半缺，整幅退回绝对残差，并在图上写明原因——
    # 悄悄混画会让 "1σ 线" 失去意义。
    half_missing = norm and not (c.get(re_key) and c.get(im_key))
    if half_missing:
        re_key, im_key, norm = "rre", "rim", False
    ylab = L["nres"] if norm else L["res"]
    ax.axhline(0, color="#333333", lw=0.8, zorder=1)
    if norm:
        for s in (1, -1):
            ax.axhline(s, color="#bbbbbb", ls="--", lw=0.6, zorder=1)
        ax.text(f[0], 1.0, " 1σ", fontsize=6.0, color="#999999",
                va="bottom", ha="left")
    else:
        ax.text(0.995, 0.02, "normalized unavailable",
                transform=ax.transAxes, fontsize=5.8, color="#999999",
                ha="right", va="bottom")
    mre, mim = "o", "s"
    ax.plot(f, c[re_key], mre, ms=2.8, mfc="none", mec=th["resid_re"],
            mew=0.8, label="Re", zorder=3)
    ax.plot(f, c[im_key], mim, ms=2.6, mfc="none", mec=th["resid_im"],
            mew=0.8, label="Im", zorder=3)
    _logx(ax)
    ax.set_xlabel(L["f"])
    ax.set_ylabel(ylab)
    ax.legend(loc="upper left", ncol=2, handletextpad=0.3,
              columnspacing=0.8, borderpad=0.25)
    _sty(ax, spec, th)


def panel_bode(ax, res, spec, th):
    L = _lab(spec)
    c = res["curves"]
    f = np.array(c["f"])
    zo = np.array(c["zre"]) + 1j * np.array(c["zim"])
    zf = np.array(c["fre"]) + 1j * np.array(c["fim"])
    ax.loglog(f, np.abs(zo), "o", ms=3.2, mfc="none", mec=th["data"],
              mew=0.8, label=L["data"])
    ax.loglog(f, np.abs(zf), "-", lw=1.2, color=th["fit"], label=L["fit"])
    ax.set_xlabel(L["f"])
    ax.set_ylabel(L["mag"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    _sty(ax, spec, th)
    ax2 = ax.twinx()
    ax2.semilogx(f, np.degrees(np.angle(zo)), "o", ms=3.0, mfc="none",
                 mec=th["resid_im"], mew=0.7)
    ax2.semilogx(f, np.degrees(np.angle(zf)), "--", lw=1.0,
                 color=th["resid_im"])
    ax2.set_ylabel(L["ph"], color=th["resid_im"])
    ax2.tick_params(axis="y", colors=th["resid_im"],
                    width=th["tick_w"], length=th["tick_len"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(th["axis_lw"])
    return ax2


_PANEL_FN = dict(nyquist=panel_nyquist, drt=panel_drt,
                 residual=panel_residual, bode=panel_bode)


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------
def _apply_style(th: dict, style: dict) -> dict:
    """把 style 覆盖叠到主题上；band 没单独给颜色就跟随 fit。"""
    th = dict(th)
    for k in STYLE_COLORS:
        v = (style or {}).get(k)
        if v:
            th[k] = v
    if (style or {}).get("fit") and not (style or {}).get("band"):
        th["band"] = th["fit"]              # 区间与拟合线必须同色，否则读成两组数据
    for k in STYLE_LW:
        v = (style or {}).get(k)
        if v:
            th[k] = float(v)
    return th


def draw(res: dict, spec: dict):
    """按 spec 画一张图，返回 (fig, spec)。**不落盘**，便于测试复用。"""
    spec = validate_spec(spec)
    missing = check_fonts(spec)
    if missing and not spec["allow_font_fallback"]:
        raise SpecError(f"字体不可用：{missing}。装字体，或显式设 "
                        f"allow_font_fallback=true（会替换成默认字体，"
                        f"导出的图与预览不一致）")
    th = _apply_style(_rc(spec), spec["style"])
    rows, cols = LAYOUTS[spec["layout"]]
    w = float(spec["canvas"]["width_mm"]) * MM
    h = float(spec["canvas"]["height_mm"]) * MM
    panels = spec["panels"]
    fig, axes = plt.subplots(rows, cols, figsize=(w, h), layout="constrained",
                             squeeze=False)
    flat = axes.ravel()
    for i, ax in enumerate(flat):
        if i >= len(panels):
            ax.set_visible(False)
            continue
        name = panels[i]
        extra = _PANEL_FN[name](ax, res, spec, th)
        ttl = (spec["titles"] or {}).get(name)
        if ttl:
            ax.set_title(ttl, fontsize=float(spec["font"]["size"] or
                                             th["font_size"]) + 0.5)
        if name == "bode" and extra is not None and spec["show_legend"]:
            pass                                     # 双轴图例会压住数据，不画
        elif spec["show_legend"] and name in ("nyquist", "drt"):
            ax.legend(loc="best", handletextpad=0.4, borderpad=0.3,
                      labelspacing=0.25)
    return fig, spec


def save(fig, path: Path, fmt: str, dpi: int):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(format=fmt, facecolor="white", bbox_inches=None)
    if fmt in ("png", "tif", "tiff", "jpg"):
        kw["dpi"] = dpi
    if fmt == "tiff":
        # 不压缩时 180×110 mm @600 dpi 是 43 MB，投审系统未必吃得下
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(path, **kw)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="出版级 EIS 图（同源于 bm_result）")
    ap.add_argument("sample", nargs="?", help="样品输出目录或 real_*.json")
    ap.add_argument("--theme", default=None, choices=sorted(THEMES))
    ap.add_argument("--panels", default=None,
                    help="逗号分隔：nyquist,drt,residual,bode")
    ap.add_argument("--layout", default=None, choices=sorted(LAYOUTS))
    ap.add_argument("--resid-mode", default=None,
                    choices=("normalized", "absolute"))
    ap.add_argument("--labels", default=None, choices=("en", "zh"))
    ap.add_argument("--width-mm", type=float, default=None)
    ap.add_argument("--height-mm", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--format", default="png",
                    help="png,svg,pdf,tiff（逗号分隔）")
    ap.add_argument("--all", action="store_true",
                    help="png+svg+pdf+tiff 全出")
    ap.add_argument("--out", default=None, help="输出目录（默认 results/<stem>/figures）")
    ap.add_argument("--spec", default=None, help="读取保存的 PlotSpec JSON")
    ap.add_argument("--color", action="append", default=None, metavar="KEY=#hex",
                    help="逐系列覆盖配色，如 fit=#c0392b（可重复）；"
                         f"可选键 {STYLE_COLORS + STYLE_LW}")
    ap.add_argument("--dump-spec", action="store_true")
    ap.add_argument("--show-predictive", action="store_true")
    a = ap.parse_args(argv)

    if a.dump_spec:
        print(json.dumps(DEFAULT_SPEC, ensure_ascii=False, indent=2))
        return 0

    spec = dict(DEFAULT_SPEC)
    if a.spec:
        spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    if a.color:
        sty = dict(spec.get("style") or {})
        for item in a.color:
            if "=" not in item:
                print(f"[spec 错误] --color 需要 KEY=#hex，收到 {item!r}",
                      file=sys.stderr)
                return 2
            k, v = item.split("=", 1)
            k, v = k.strip(), v.strip()
            if k not in STYLE_COLORS + STYLE_LW:
                print(f"[spec 错误] 未知配色键 {k!r}；"
                      f"可选 {STYLE_COLORS + STYLE_LW}", file=sys.stderr)
                return 2
            sty[k] = v
        spec["style"] = sty
    for key, val in (("theme", a.theme), ("layout", a.layout),
                     ("labels", a.labels), ("resid_mode", a.resid_mode),
                     ("dpi", a.dpi)):
        if val is not None:
            spec[key] = val
    if a.panels:
        spec["panels"] = [s.strip() for s in a.panels.split(",") if s.strip()]
    if a.width_mm:
        spec["canvas"] = dict(spec.get("canvas") or {},
                              width_mm=a.width_mm)
    if a.height_mm:
        spec["canvas"] = dict(spec.get("canvas") or {},
                              height_mm=a.height_mm)
    if a.show_predictive:
        spec["show_predictive"] = True

    if not a.sample:
        print("需要 sample 参数", file=sys.stderr)
        return 2
    res = R.AnalysisResult.load(a.sample).payload()
    try:
        fig, spec = draw(res, spec)
    except SpecError as exc:
        print(f"[spec 错误] {exc}", file=sys.stderr)
        return 2

    out = Path(a.out) if a.out else Path("results") / res["name"] / "figures"
    fmts = ["png", "svg", "pdf", "tiff"] if a.all else \
        [s.strip() for s in a.format.split(",") if s.strip()]
    stem = "figure_" + "_".join(spec["panels"])
    made = []
    for fmt in fmts:
        fmt = {"tif": "tiff"}.get(fmt, fmt)
        made.append(save(fig, out / f"{stem}.{fmt}", fmt, int(spec["dpi"])))
    plt.close(fig)
    # 保存实际生效的 spec：换一条谱可以原样复现同一版式
    spec_path = out / "figure_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    for p in made:
        print(f"  {p}  ({p.stat().st_size / 1024:.1f} KB)")
    print(f"  {spec_path}")
    # 结果包封口：figures/ 是刚写进去的，bundle() 那会儿的 manifest 里没有它们。
    # 不重扫一遍，"每个文件都有 SHA256" 就是空话。
    mf = out.parent / "manifest.json"
    if mf.exists():
        mp = R.finalize_bundle(out.parent)
        man = json.loads(mp.read_text(encoding="utf-8"))
        print(f"  {mp}  (已封包，{len(man['files'])} 个文件，"
              f"figures {man['counts']['figures']} 个)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
