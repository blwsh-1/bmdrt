"""等效电路图（adr：吸附弛豫三支路并联）。

只画 `bm_top.z_adr` 这一个拓扑：

    Z = Rs + [ CPE ∥ Rct ∥ Z_ad ]，Z_ad = R_ad·jωτ_ad/(1+jωτ_ad)（R∥L）

与 `bm_circuit`（voigt 串联）和 `bm_circuit_nested`（两层嵌套）并列的第三个
绘图器。符号/配色复用 `bm_circuit` 原语（同一张图里电阻长得一样）。

图与方程一一对应（同 `bm_circuit_nested` 的纪律）：
- `elements()` 是单一来源，`draw_adr` 的参数标签全从它生成；
- 自检断言"图上参数集合 == `bm_top.ADR_KEYS`"——拟合采什么图上画什么；
- CPE 按窗口幅值 Q=|Z(ω_ref)|（Ω）标注、不写 F（n≠1 时不是法拉）；
- 派生量 L_ad=R_ad·τ_ad 现算，按量级换单位。
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bm_circuit import ACC, INK, MUTE, _cpe, _dot, _inductor, _resistor, _wire

#: 版面常数（单位与 bm_circuit 一致：1 数据单位 ≈ 1 英寸）
PAD = 0.34
W_RS = 0.95
W_BR = 2.30          # 三支路并联块的宽度（R∥L 需要较宽）
GAP = 0.30
Y_TOP = 1.05          # CPE 支路
Y_BOT = -0.90         # R∥L 支路块中心（尽量居中，避免 L_ad 标签压到底部脚注）
Y_SUB = 0.32          # R∥L 的两条子横线间距
FS = 7.8


def _num(v: float, unit: str = "", nd: int = 3) -> str:
    """数值格式化：不用 %.3g（2990 → `2.99e+03` 与 ×10ⁿ 混写难读）。"""
    if v == 0:
        return f"0 {unit}".strip()
    a = abs(v)
    if 1e-2 <= a < 1e4:
        return f"{v:.4g} {unit}".strip()
    m, e = f"{v:.{nd - 1}e}".split("e")
    return f"{m}×10$^{{{int(e)}}}$ {unit}".strip()


def _num_H(L: float) -> str:
    """电感按量级换单位：1.38×10⁴ H 比 1.38×10¹⁰ µH 好读。"""
    a = abs(L)
    if a >= 1.0:
        return _num(L, "H")
    if a >= 1e-3:
        return _num(L * 1e3, "mH")
    if a >= 1e-6:
        return _num(L * 1e6, "µH")
    return _num(L * 1e9, "nH")


def elements() -> list[dict]:
    """图上出现的元件，以及每个元件绑定的参数名（**单一来源**）。

    与 `bm_top.ADR_KEYS = (Rs, Q, n, Rct, R_ad, tau_ad)` 对拍：
    `L_ad` 是派生量（R_ad·τ_ad），不进采样器，标 `derived`。
    """
    return [
        dict(id="Rs",  kind="resistor", params=("Rs",)),
        dict(id="CPE", kind="cpe",      params=("Q", "n")),
        dict(id="Rct", kind="resistor", params=("Rct",)),
        dict(id="R_ad", kind="resistor", params=("R_ad",)),
        dict(id="L_ad", kind="inductor", params=("tau_ad",), derived=("L_ad",)),
    ]


def equation() -> str:
    """与 `z_adr` 逐项对应的拓扑式（mathtext）。"""
    return (r"$Z = R_s + \left[\mathrm{CPE} \parallel R_{ct} \parallel "
            r"\left(R_{ad} \parallel L_{ad}\right)\right]$")


def layout() -> dict:
    x = PAD
    rs = (x, x + W_RS)
    xA = rs[1] + GAP
    xB = xA + GAP + W_BR
    return dict(rs=rs, xA=xA, xB=xB, x_in=PAD * 0.5, x_out=xB + 0.40,
                x_end=xB + 0.62)


def draw_adr(p: dict, *, f_ref: float, path, name: str = "",
             admission_state: str = "provisional") -> str:
    """画 adr 拓扑。`p` 是后验中位数（键见 `bm_top.ADR_KEYS`）。

    `f_ref` 是 CPE 的归一化频率，必须与拟合时一致（`z_adr` 默认 √(f_min·f_max)）。
    """
    need = [k for e in elements() for k in e["params"]]
    missing = [k for k in need if k not in p]
    if missing:
        raise KeyError(f"画图缺少参数 {missing}（adr 需要 {sorted(set(need))}）")
    if f_ref <= 0:
        raise ValueError(f"f_ref 必须为正，收到 {f_ref!r}")

    L = layout()
    fig, ax = plt.subplots(figsize=(1.05 * L["x_end"], 3.75))
    ax.set_xlim(0, L["x_end"])
    ax.set_ylim(-1.85, 1.75)
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
    label(0.5 * (x0 + x1), -0.30, f"R$_s$ = {_num(p['Rs'], 'Ω')}", va="top")
    _wire(ax, x1, L["xA"])
    _dot(ax, L["xA"])

    # ---- 节点 A 竖直线（三支路并联的左端）----
    # 竖直线要覆盖到 R∥L 的电感子线（Y_BOT - Y_SUB），否则电感支路两端悬空。
    ax.plot([L["xA"], L["xA"]], [Y_BOT - Y_SUB, Y_TOP], color=INK, lw=1.6, zorder=2)
    ax.plot([L["xB"], L["xB"]], [Y_BOT - Y_SUB, Y_TOP], color=INK, lw=1.6, zorder=2)
    _dot(ax, L["xB"])

    # ---- 支路 1：CPE（上）----
    xa, xb = L["xA"], L["xB"]
    _wire(ax, xa, xa + 0.08, Y_TOP)
    _cpe(ax, xa + 0.08, xb - 0.08, Y_TOP)
    _wire(ax, xb - 0.08, xb, Y_TOP)
    label(0.5 * (xa + xb), Y_TOP + 0.28,
          f"CPE:  Q = {_num(p['Q'], 'Ω')},  n = {float(p['n']):.3f}")

    # ---- 支路 2：Rct（中）----
    _wire(ax, xa, xa + 0.10, 0.0)
    _resistor(ax, xa + 0.10, xb - 0.10, 0.0)
    _wire(ax, xb - 0.10, xb, 0.0)
    # Rct 标签放主干上方（y=+0.30），避开下支路 R_ad（y≈-0.53）避免重叠
    label(0.5 * (xa + xb), 0.30, f"R$_{{ct}}$ = {_num(p['Rct'], 'Ω')}")

    # ---- 支路 3：R∥L（下）----
    _wire(ax, xa, xa + 0.08, Y_BOT + Y_SUB)
    _resistor(ax, xa + 0.10, xb - 0.10, Y_BOT + Y_SUB)
    _wire(ax, xb - 0.08, xb, Y_BOT + Y_SUB)
    _wire(ax, xa, xa + 0.08, Y_BOT - Y_SUB)
    _inductor(ax, xa + 0.10, xb - 0.10, Y_BOT - Y_SUB)
    _wire(ax, xb - 0.08, xb, Y_BOT - Y_SUB)
    label(0.5 * (xa + xb), Y_BOT + Y_SUB + 0.28, f"R$_{{ad}}$ = {_num(p['R_ad'], 'Ω')}")
    tau = float(p["tau_ad"])
    Lad = float(p["R_ad"]) * tau
    label(0.5 * (xa + xb), Y_BOT - Y_SUB - 0.30, f"L$_{{ad}}$ = {_num_H(Lad)}",
          va="top")

    # ---- 输出端子 ----
    _wire(ax, L["xB"], L["x_out"])
    _dot(ax, L["x_out"])

    # ---- 脚注：方程、Q 的量纲、派生量、准入状态 ----
    # mathtext 的行内 `$...$` 不能被折行截断（$ 一旦跨行，整段退回字面量），
    # 所以方程这一行单独放、不 wrap；其余散文才折行。
    import textwrap
    eq = equation()
    prose = [
        f"CPE is parametrised by its window amplitude Q = |Z(f_ref)|, not by a "
        f"capacitance: Z = Q(ω_ref/ω)^n·exp(−jnπ/2).  f_ref = {_num(f_ref, 'Hz')}.",
        f"L$_{{ad}}$ = R$_{{ad}}$·τ$_{{ad}}$ = {_num_H(Lad)} (derived)    "
        f"τ$_{{ad}}$ = {_num(tau, 's')} → "
        f"f$_{{ad}}$ = {_num(1.0 / (2 * np.pi * tau), 'Hz')}",
    ]
    if admission_state:
        prose.append(f"admission = {admission_state}: §13 0/3, §14 4/14 — "
                     f"symbols are the fitted parameters, not physical values.")
    wrap = max(72, int(L["x_end"] * 15.0))
    lines = [eq] + [ln for t in prose for ln in textwrap.wrap(t, wrap)]
    note_txt = "\n".join(lines)
    ax.text(0.5, -0.02, note_txt, transform=ax.transAxes,
            ha="center", va="top", fontsize=6.9, color=MUTE, linespacing=1.6)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.86, bottom=0.06)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _selftest() -> int:
    ok, total, bad = 0, 0, []

    def c(name, cond, msg=""):
        nonlocal ok, total
        total += 1
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {msg}" if msg else ""))
        if cond:
            ok += 1
        else:
            bad.append(name)

    import bm_top as T
    print("adr 电路图自检")

    got = {k for e in elements() for k in e["params"]}
    c("图上参数集合 == bm_top.ADR_KEYS", got == set(T.ADR_KEYS), f"{got}")
    c("派生量只标 L_ad", [e["id"] for e in elements() if e.get("derived")] == ["L_ad"])

    eq = equation()
    c("方程含三支路并联", "parallel" in eq and "R_{ct}" in eq)

    # 参数缺一个要报错，不静默
    p = {k: 1.0 for k in T.ADR_KEYS}
    del p["Rct"]
    try:
        draw_adr(p, f_ref=1.0, path="out/_adr_selftest.png")
        c("缺参数被拒", False)
    except KeyError:
        c("缺参数被拒", True)

    # 合法参数能画出来
    p = dict(Rs=100.0, Q=120.0, n=0.79, Rct=130.0, R_ad=3000.0, tau_ad=4.6)
    out = draw_adr(p, f_ref=31.6, path="out/_adr_selftest.png", name="selftest")
    import os
    c("产出 png 存在", os.path.exists(out))
    if os.path.exists(out):
        os.remove(out)

    print(f"\n自检 {ok}/{total} 通过" if not bad else f"\nFAIL {len(bad)}/{total}: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
