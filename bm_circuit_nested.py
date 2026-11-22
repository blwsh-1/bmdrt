"""nested 拓扑的等效电路图（只画 `bm_top.z_nested` 这一个拓扑族）。

为什么单独一个文件
------------------
`bm_circuit.py` 明确只负责 **BM-DRT 的串联骨架**（`R∞ + jωL + Σ HN`），并在
文档里写了"不是通用 ECM 拓扑绘图器"。nested 是**梯形网络**，不是串联链：

    layers=2:  Z = Rs + [ CPE1 ∥ ( R1 + [ CPE2 ∥ ( R2 + Z_b ) ] ) ]
    layers=1:  Z = Rs + [ CPE1 ∥ ( R1 + Z_b ) ]

    Z_b = R_ad ∥ L_ad        (branch=parallel，默认)
        = R_L + jωL_ad       (branch=series)
        = CPE_b              (branch=capacitive)

所以要另画一个。符号（导线/电阻/电感/CPE/节点）与配色**复用**
`bm_circuit` 的原语，不复制一套：同一张图里两个文件画出来的电阻必须长得一样。

图与方程必须一一对应
--------------------
`elements()` 是**单一来源**：`draw_nested()` 的所有参数标签都从它生成，而
`--selftest` 断言"图上出现的参数集合（去掉派生量 `L_ad`）== `bm_top.sampled_keys(branch, layers)`"。
这句话的意思是：**拟合采了哪些参数，图上就有哪些；图上有的，都能拿回复现拟合**。
多画一个元件（比如给 layers=1 画 R2）会让这条断言直接红。

CPE 的参数化要特别说明
----------------------
`bm_top.z_cpe` 用的是**窗口内幅值** `Q = |Z(ω_ref)|`（单位 Ω），不是 CPE 系数
`Y`（单位 S·s^n）。所以图上标的是 `Q = … Ω` 和 `n`，**不写 F**：

    Z_CPE = Q·(ω_ref/ω)^n · exp(−j nπ/2)

想让读者算电容得自己换算 `Y = 1/(Q·ω_ref^n)`，这对 n≠1 根本不是法拉。
脚注里写了这一句，避免被当成 C。

状态
----
产物恒带 `admission.state="provisional"`——电路图画得再对也不改变 §13 `0/3`、
§14 `4/14` 的事实，参数不得当物理量。
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 符号原语与配色复用 bm_circuit（同一个包里，符号必须完全一致）
from bm_circuit import (ACC, INK, MUTE, _cpe, _dot, _inductor,
                        _resistor, _wire)

VALID_BRANCHES = ("parallel", "series", "capacitive")

#: 版面常数（单位与 bm_circuit 一致：1 数据单位 ≈ 1 英寸）
PAD = 0.34
W_RS, W_R1, W_R2 = 0.95, 1.05, 1.05
#: 顶层串联电感段宽度（include_L=True 才出现）
W_L = 0.72
W_BRANCH = {"parallel": 1.70, "series": 1.90, "capacitive": 1.05}
GAP = 0.26
Y_TOP = 1.10          # CPE1 所在的外层回路
Y_BOT = -1.00         # CPE2 所在的内层回路
Y_SUB = 0.36          # R_ad ∥ L_ad 的两条子横线
FS = 7.8


def _check(branch: str, layers: int) -> None:
    if branch not in VALID_BRANCHES:
        raise ValueError(f"branch 只能是 {VALID_BRANCHES}，收到 {branch!r}")
    if layers not in (1, 2):
        raise ValueError(f"layers 只能是 1/2，收到 {layers!r}")


def _num(v: float, unit: str = "", nd: int = 3) -> str:
    """数值格式化。

    不用 `bm_circuit._fmt`：它走 `%.3g`，2990 会写成 `2.99e+03`——那是 G 格式的
    指数写法，和同一张图上 `2.69×10⁴` 的写法混在一起很难读。
    这里 [1e-2, 1e4) 用定点全写（2990 → `2990`），超出才进科学计数。
    """
    if v == 0:
        return f"0 {unit}".strip()
    a = abs(v)
    if 1e-2 <= a < 1e4:
        return f"{v:.4g} {unit}".strip()
    m, e = f"{v:.{nd - 1}e}".split("e")
    return f"{m}×10$^{{{int(e)}}}$ {unit}".strip()


def _num_H(L: float) -> str:
    """电感按量级换单位：13.8 kH 写成 `1.38×10⁴ H` 比 `1.38×10¹⁰ µH` 好读。"""
    a = abs(L)
    if a >= 1.0:
        return _num(L, "H")
    if a >= 1e-3:
        return _num(L * 1e3, "mH")
    if a >= 1e-6:
        return _num(L * 1e6, "µH")
    return _num(L * 1e9, "nH")


def elements(branch: str, layers: int, include_L: bool = False) -> list[dict]:
    """图上出现的元件，以及每个元件绑定的参数名。

    **单一来源**：`draw_nested` 的标签由它生成，自检拿它和
    `bm_top.sampled_keys` 对拍。`L_ad` 永远标 `derived=True`——它不进采样器
    （`parallel` 由 `R_ad·tau_ad` 算出，`series` 由 `R_L·tau_L` 算出）。

    `include_L`（2026-11-20）：顶层串联 `jωL` 是**采样参数**，必须画出来；
    否则图与 `z_nested` 的方程不一致（图上参数≠采样参数）。
    """
    _check(branch, layers)
    out = [dict(id="Rs",   kind="resistor", params=("Rs",))]
    if include_L:
        out.append(dict(id="L", kind="inductor", params=("L",)))
    out += [
        dict(id="CPE1", kind="cpe",      params=("Q1", "n1")),
        dict(id="R1",   kind="resistor", params=("R1",)),
    ]
    if layers >= 2:
        out += [
            dict(id="CPE2", kind="cpe",      params=("Q2", "n2")),
            dict(id="R2",   kind="resistor", params=("R2",)),
        ]
    if branch == "parallel":
        out += [dict(id="R_ad", kind="resistor", params=("R_ad",)),
                dict(id="L_ad", kind="inductor", params=("tau_ad",),
                     derived=("L_ad",))]
    elif branch == "series":
        out += [dict(id="R_L",  kind="resistor", params=("R_L",)),
                dict(id="L_ad", kind="inductor", params=("tau_L",),
                     derived=("L_ad",))]
    else:
        out += [dict(id="CPE_b", kind="cpe", params=("Q_b", "n_b"))]
    return out


def equation(branch: str, layers: int, include_L: bool = False) -> str:
    """与 `z_nested` 逐项对应的拓扑式（mathtext）。"""
    _check(branch, layers)
    zb = {"parallel": r"R_{ad}\parallel L_{ad}",
          "series": r"R_L + j\omega L_{ad}",
          "capacitive": r"\mathrm{CPE}_b"}[branch]
    lead = r"R_s + j\omega L + " if include_L else r"R_s + "
    if layers >= 2:
        return (r"$Z = " + lead + r"\left[\mathrm{CPE}_1 \parallel \left(R_1 + "
                r"\left[\mathrm{CPE}_2 \parallel (R_2 + " + zb + r")\right]"
                r"\right)\right]$")
    return (r"$Z = " + lead + r"\left[\mathrm{CPE}_1 \parallel \left(R_1 + "
            + zb + r"\right)\right]$")


def layout(branch: str, layers: int, include_L: bool = False) -> dict:
    """横向坐标：串联元件在主干上，两个 CPE 分别走上下两条回路。

    节点顺序固定为 `Rs → A → R1 → B →(R2)→ C → Z_b → D`，`A` 与 `D` 之间
    挂 CPE1，`B` 与 `D` 之间挂 CPE2（layers=2）。这与 `z_nested` 的嵌套关系
    一一对应：CPE1 跨整个 `z_mid`，CPE2 只跨内层 `z_par2`。

    `include_L`（2026-11-20）：在 `Rs` 之后插入串联 `L` 段，其余节点整体右移，
    所以 `xA` 之后的所有坐标都跟着 `ls` 顺延。
    """
    _check(branch, layers)
    x = PAD
    rs = (x, x + W_RS)
    x = rs[1]
    if include_L:
        ls = (x + GAP, x + GAP + W_L)
        x = ls[1]
    else:
        ls = None
    xA = x + GAP
    r1 = (xA + GAP, xA + GAP + W_R1)
    xB = r1[1] + GAP
    if layers >= 2:
        r2 = (xB + GAP, xB + GAP + W_R2)
        xC = r2[1] + GAP
    else:
        r2, xC = None, xB
    br = (xC + GAP, xC + GAP + W_BRANCH[branch])
    xD = br[1] + GAP
    return dict(rs=rs, ls=ls, xA=xA, r1=r1, xB=xB, r2=r2, xC=xC, br=br, xD=xD,
                x_in=PAD * 0.5, x_out=xD + 0.40, x_end=xD + 0.62)


def derived_line(p: dict, branch: str) -> str:
    """派生量一行（电感与转折频率）。取不到就返回空串，不猜。"""
    try:
        if branch == "parallel":
            R, tau = float(p["R_ad"]), float(p["tau_ad"])
            L = R * tau
            return (f"L$_{{ad}}$ = R$_{{ad}}$·τ$_{{ad}}$ = {_num_H(L)} "
                    f"(derived)    τ$_{{ad}}$ = {_num(tau, 's')} → "
                    f"f$_{{ad}}$ = {_num(1.0 / (2 * np.pi * tau), 'Hz')}")
        if branch == "series":
            L = float(p["L_ad"]) if p.get("L_ad") else \
                float(p["R_L"]) * float(p["tau_L"])
            f = float(p["R_L"]) / (2 * np.pi * L) if L else None
            return (f"L$_{{ad}}$ = R$_L$·τ$_L$ = {_num_H(L)} (derived)    "
                    + (f"f$_{{ad}}$ = {_num(f, 'Hz')}" if f else ""))
    except (KeyError, TypeError, ValueError):
        return ""
    return ""


def draw_nested(p: dict, *, branch: str, layers: int, f_ref: float, path,
                name: str = "", admission_state: str = "provisional",
                include_L: bool = False) -> str:
    """画 nested 拓扑。`p` 是**后验中位数**字典（键见 `bm_top.sampled_keys`）。

    `f_ref` 是 CPE 的归一化频率（`z_nested` 默认 `√(f_min·f_max)`），
    必须与拟合时一致，否则图上的 `Q` 数值对不上同一套方程。

    `include_L`（2026-11-20）：`z_nested(include_L=True)` 时顶层多一个串联 `L`，
    图上必须画出同一个元件——**图上参数集合 == 采样参数集合**是这张图的
    硬约束（T10），不能只改方程不改图。
    """
    _check(branch, layers)
    L = layout(branch, layers, include_L)
    els = {e["id"]: e for e in elements(branch, layers, include_L)}

    # 只要求**采样参数**到齐；`L_ad` 是派生量（R·τ），由 derived_line 现算，
    # 缺了不算错——它在 `z_nested` 里也只有 series 才可能出现在 p 里。
    need = [k for e in els.values() for k in e["params"]]
    missing = [k for k in need if k not in p]
    if missing:
        raise KeyError(f"画图缺少参数 {missing}（按 {branch}/L={layers} 需要 "
                       f"{sorted(set(need))}）")
    if f_ref <= 0:
        raise ValueError(f"f_ref 必须为正，收到 {f_ref!r}")

    fig, ax = plt.subplots(figsize=(1.05 * L["x_end"], 3.75))
    ax.set_xlim(0, L["x_end"])
    ax.set_ylim(-1.78, 1.80)
    ax.axis("off")

    def label(xm, y, txt, *, va="bottom", fs=FS):
        ax.text(xm, y, txt, ha="center", va=va, fontsize=fs, color=ACC)

    # ---- 输入 → Rs → 节点 A ----
    _dot(ax, L["x_in"])
    _wire(ax, L["x_in"], L["rs"][0])
    x0, x1 = L["rs"]
    _wire(ax, x0, x0 + 0.10)
    _resistor(ax, x0 + 0.10, x1 - 0.10)
    _wire(ax, x1 - 0.10, x1)
    label(0.5 * (x0 + x1), -0.28, f"R$_s$ = {_num(p['Rs'], 'Ω')}", va="top")
    if L.get("ls"):
        lx0, lx1 = L["ls"]
        _wire(ax, x1, lx0)
        _inductor(ax, lx0, lx1)
        label(0.5 * (lx0 + lx1), -0.28, f"L = {_num_H(float(p['L']))}",
              va="top")
        _wire(ax, lx1, L["xA"])
    else:
        _wire(ax, x1, L["xA"])
    _dot(ax, L["xA"])

    # ---- 主干：R1（layers=2 时再接 R2） ----
    a, b = L["r1"]
    _wire(ax, L["xA"], a)
    _wire(ax, a, a + 0.10)
    _resistor(ax, a + 0.10, b - 0.10)
    _wire(ax, b - 0.10, b)
    label(0.5 * (a + b), 0.22, f"R$_1$ = {_num(p['R1'], 'Ω')}")
    if layers >= 2:
        _wire(ax, b, L["r2"][0])
        _dot(ax, L["xB"])
        c, d = L["r2"]
        _wire(ax, c, c + 0.10)
        _resistor(ax, c + 0.10, d - 0.10)
        _wire(ax, d - 0.10, d)
        label(0.5 * (c + d), 0.22, f"R$_2$ = {_num(p['R2'], 'Ω')}")
        _wire(ax, d, L["br"][0])
    else:
        _wire(ax, b, L["br"][0])

    # ---- 支路 Z_b ----
    b0, b1 = L["br"]
    xm = 0.5 * (b0 + b1)
    if branch == "parallel":
        for xx in (b0, b1):
            ax.plot([xx, xx], [-Y_SUB, Y_SUB], color=INK, lw=1.6, zorder=2)
        _dot(ax, b0)                          # 主干 × 并联支路 = 4 向焊接点
        _dot(ax, b1)
        _wire(ax, b0 + 0.02, b0 + 0.18, Y_SUB)
        _resistor(ax, b0 + 0.18, b1 - 0.18, Y_SUB)
        _wire(ax, b1 - 0.18, b1 - 0.02, Y_SUB)
        _wire(ax, b0 + 0.02, b0 + 0.18, -Y_SUB)
        _inductor(ax, b0 + 0.18, b1 - 0.18, -Y_SUB)
        _wire(ax, b1 - 0.18, b1 - 0.02, -Y_SUB)
        label(xm, Y_SUB + 0.20, f"R$_{{ad}}$ = {_num(p['R_ad'], 'Ω')}")
        tau = float(p["tau_ad"])
        Lad = float(p["R_ad"]) * tau
        label(xm, -Y_SUB - 0.20, f"L$_{{ad}}$ = {_num_H(Lad)}",
              va="top")
    elif branch == "series":
        w = 0.5 * (b1 - b0) - 0.10
        _wire(ax, b0, b0 + 0.08)
        _resistor(ax, b0 + 0.08, b0 + 0.08 + w * 0.82)
        label(b0 + 0.08 + w * 0.41, 0.22,
              f"R$_L$ = {_num(p['R_L'], 'Ω')}")
        mid = b0 + 0.08 + w * 0.82 + 0.16
        _wire(ax, b0 + 0.08 + w * 0.82, mid)
        _inductor(ax, mid, mid + w * 0.82)
        _wire(ax, mid + w * 0.82, b1)
        Lad = float(p["L_ad"]) if p.get("L_ad") else float(p["R_L"]) * float(p["tau_L"])
        label(mid + w * 0.41, -0.26, f"L$_{{ad}}$ = {_num_H(Lad)}",
              va="top")
    else:
        _wire(ax, b0, b0 + 0.10)
        _cpe(ax, b0 + 0.10, b1 - 0.10, 0.0,
             slash=abs(float(p["n_b"]) - 1.0) >= 1e-3)
        _wire(ax, b1 - 0.10, b1)
        label(xm, 0.30, f"Q$_b$ = {_num(p['Q_b'], 'Ω')},  "
                        f"n$_b$ = {float(p['n_b']):.3f}")

    # 支路输出 → 节点 D：主干必须连续，这里断掉整个梯形网络就开路了。
    # （2026-11-16 用户看图指出 AZ31-4Li_150 的 R∥L 块右侧像"断点"，
    #   核查确认是漏画了这条主干连线，不是符号风格问题。）
    _wire(ax, b1, L["xD"])

    # ---- 外层 CPE1：A → D（跨整个 z_mid） ----
    xa_d = L["xA"]
    xd = L["xD"]
    c1 = (0.5 * (xa_d + xd) - 0.44, 0.5 * (xa_d + xd) + 0.44)
    ax.plot([xa_d, xa_d], [0.0, Y_TOP], color=INK, lw=1.6, zorder=2)
    _wire(ax, xa_d, c1[0], Y_TOP)
    _cpe(ax, c1[0], c1[1], Y_TOP,
         slash=abs(float(p["n1"]) - 1.0) >= 1e-3)
    _wire(ax, c1[1], xd, Y_TOP)
    ax.text(0.5 * (c1[0] + c1[1]), Y_TOP + 0.24,
            f"CPE$_1$:  Q$_1$ = {_num(p['Q1'], 'Ω')},  "
            f"n$_1$ = {float(p['n1']):.3f}",
            ha="center", va="bottom", fontsize=FS, color=ACC)

    # ---- 内层 CPE2：B → D（只跨 z_par2） ----
    if layers >= 2:
        ax.plot([L["xB"], L["xB"]], [0.0, Y_BOT], color=INK, lw=1.6, zorder=2)
        c2 = (0.5 * (L["xB"] + xd) - 0.40, 0.5 * (L["xB"] + xd) + 0.40)
        _wire(ax, L["xB"], c2[0], Y_BOT)
        _cpe(ax, c2[0], c2[1], Y_BOT,
             slash=abs(float(p["n2"]) - 1.0) >= 1e-3)
        _wire(ax, c2[1], xd, Y_BOT)
        ax.text(0.5 * (c2[0] + c2[1]), Y_BOT - 0.24,
                f"CPE$_2$:  Q$_2$ = {_num(p['Q2'], 'Ω')},  "
                f"n$_2$ = {float(p['n2']):.3f}",
                ha="center", va="top", fontsize=FS, color=ACC)

    # ---- 节点 D：CPE1/(CPE2)/主干 的汇合点 ----
    _dot(ax, xd)
    ax.plot([xd, xd], [Y_BOT if layers >= 2 else 0.0, Y_TOP],
            color=INK, lw=1.6, zorder=2)
    _wire(ax, xd, L["x_out"])
    _dot(ax, L["x_out"])

    # ---- 标题 ----
    head = "Equivalent circuit — nested ladder (posterior median)"
    if name:
        head = f"{name} — " + head
    tag = {"parallel": "   R∥L branch", "series": "   R+L branch",
           "capacitive": "   CPE branch"}[branch]
    ax.set_title(f"{head}    layers = {layers}{tag}", fontsize=10.5, pad=12)

    # ---- 脚注：方程、Q 的量纲、派生量、准入状态 ----
    # mathtext 的行内 `$...$` 不能被折行截断（$ 一旦跨行，整段退回字面量），
    # 所以方程这一行单独放、不 wrap；其余散文才折行。
    import textwrap
    eq = equation(branch, layers, include_L)
    prose = [
        "CPE is parametrised by its window amplitude Q = |Z(f_ref)| = "
        "|Z(√(f_min·f_max))|, not by a capacitance: "
        "Z = Q·(ω_ref/ω)^n·exp(−jnπ/2).  f_ref = " + _num(f_ref, "Hz") + ".",
    ]
    d = derived_line(p, branch)
    if d:
        prose.append(d)
    if admission_state:
        prose.append("admission = " + admission_state +
                     ": §13 0/3, §14 4/14 — symbols are the fitted "
                     "parameters, not physical values.")
    wrap = max(72, int(L["x_end"] * 15.0))
    lines = [eq] + [ln for t in prose for ln in textwrap.wrap(t, wrap)]
    note_txt = "\n".join(lines)
    ax.text(0.5, -0.02, note_txt, transform=ax.transAxes,
            ha="center", va="top", fontsize=6.9, color=MUTE, linespacing=1.6)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.86, bottom=0.06)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile
    from pathlib import Path

    import bm_top as T

    ok = total = 0
    bad = []

    def c(nm, cond, extra=""):
        nonlocal ok, total
        total += 1
        print(f"  [{'ok' if cond else 'FAIL'}] {nm}" + (f"  {extra}" if extra else ""))
        if cond:
            ok += 1
        else:
            bad.append(nm)

    combos = [(b, l) for b in VALID_BRANCHES for l in (1, 2)]
    print("1) 图上参数集合 == 采样参数集合（拟合采什么，图上画什么）")
    for br, ly in combos:
        got = {k for e in elements(br, ly) for k in e["params"]}
        got |= {k for e in elements(br, ly) for k in e.get("derived", ())}
        # 容性支路没有电感，`L_ad` 不该出现（`z_nested` 也只有另外两支才派生它）
        want = set(T.sampled_keys(br, ly))
        if br != "capacitive":
            want |= {"L_ad"}
        c(f"{br}/L={ly}: 参数集合一致", got == want,
          f"多 {sorted(got - want)} 少 {sorted(want - got)}")

    print("2) layers=1 不许出现 R2 / CPE2（z_nested 里根本不连接）")
    for br in VALID_BRANCHES:
        ids = {e["id"] for e in elements(br, 1)}
        c(f"{br}/L=1 无 R2/CPE2", not ({"R2", "CPE2"} & ids))

    print("3) 三种支路的元件类型正确")
    c("parallel 有 R_ad + 电感",
      {"R_ad", "L_ad"} <= {e["id"] for e in elements("parallel", 2)}
      and [e for e in elements("parallel", 2) if e["id"] == "L_ad"][0]["kind"]
      == "inductor")
    c("series 有 R_L + 电感",
      {"R_L", "L_ad"} <= {e["id"] for e in elements("series", 2)})
    c("capacitive 有 CPE_b 且没有电感",
      {"CPE_b"} <= {e["id"] for e in elements("capacitive", 2)}
      and "L_ad" not in {e["id"] for e in elements("capacitive", 2)})

    print("4) 非法拓扑/层数显式报错")
    try:
        elements("wrong", 2)
        c("未知 branch 报错", False)
    except ValueError:
        c("未知 branch 报错", True)
    try:
        elements("parallel", 3)
        c("layers=3 报错", False)
    except ValueError:
        c("layers=3 报错", True)
    try:
        draw_nested({}, branch="parallel", layers=2, f_ref=100.0, path="x.png")
        c("缺参数时报错（不静默画半张图）", False)
    except KeyError as e:
        c("缺参数时报错（不静默画半张图）", "Rs" in str(e))

    print("5) 版面坐标单调：Rs < A < R1 < B < (R2) < C < Z_b < D")
    for br, ly in combos:
        g = layout(br, ly)
        xs = [g["rs"][1], g["xA"], g["r1"][0], g["r1"][1], g["xB"]]
        if ly >= 2:
            xs += [g["r2"][0], g["r2"][1], g["xC"]]
        xs += [g["br"][0], g["br"][1], g["xD"]]
        c(f"{br}/L={ly}: 横坐标单调递增", all(a < b for a, b in zip(xs, xs[1:])))

    print("6) 真正落盘：6 种组合都能出图，且尺寸非零")
    p_full = dict(Rs=12.0, R1=180.0, Q1=95.0, n1=0.86, R2=42.0, Q2=18.0,
                  n2=0.74, R_ad=26.0, tau_ad=1.4e-3, R_L=26.0, tau_L=1.4e-3,
                  Q_b=30.0, n_b=0.91, L_ad=0.0364)
    with tempfile.TemporaryDirectory() as td:
        for br, ly in combos:
            pth = Path(td) / f"{br}_{ly}.png"
            draw_nested(p_full, branch=br, layers=ly, f_ref=31.6,
                        path=pth, name="selftest")
            c(f"{br}/L={ly}: 写出 {pth.stat().st_size // 1024 if pth.exists() else 0} KB",
              pth.exists() and pth.stat().st_size > 20_000)

    print("7) 方程与 z_nested 的分支一致（不含错支路的名字）")
    names = {"parallel": "R_{ad}", "series": "R_L", "capacitive": "CPE}_b"}
    for br in VALID_BRANCHES:
        eq = equation(br, 2)
        others = [v for k, v in names.items() if k != br]
        c(f"{br}: 方程只含自己的支路", names[br] in eq
          and not any(o in eq for o in others))

    print("8) 主干连通回归：支路输出 b1 → 节点 D 必须有主干导线（用户发现的断点）")
    orig_wire = _wire
    for br, ly in combos:
        g = layout(br, ly)
        segs = []

        def rec(ax_, x0_, x1_, y=0.0, **kw):
            segs.append((float(x0_), float(x1_), float(y)))

        globals()["_wire"] = rec
        try:
            with tempfile.TemporaryDirectory() as td:
                draw_nested(p_full, branch=br, layers=ly, f_ref=31.6,
                            path=f"{td}/x.png")
        finally:
            globals()["_wire"] = orig_wire
        b1, xd = float(g["br"][1]), float(g["xD"])
        hit = any(abs(a - b1) < 1e-6 and abs(b - xd) < 1e-6 and abs(y) < 1e-6
                  for a, b, y in segs)
        c(f"{br}/L={ly}: 存在 b1→D 主干连线（y=0）", hit)

    print(f"\n自检 {ok}/{total}")
    if bad:
        print("失败：")
        for n in bad:
            print("  -", n)
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
