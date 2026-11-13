"""涂层族 Voigt pilot 汇总图（读 results.jsonl / mode_diag.json / k3_check.json，不跑拟合）。"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = pathlib.Path("D:/eis训练数据/bmdrt/out/coating_voigt_pilot")
RATIO_STRUCT, RATIO_RUNAWAY, RHAT_MAX = 1.5, 5.0, 1.05
rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
rows.sort(key=lambda r: (tuple(r["cell"]), r["name"]))


def ok(d):
    return d.get("rhat", 9) <= RHAT_MAX and d.get("div", 1) == 0


C_NEST, C_PROD, C_FIX2, C_BAD = "#2e7d32", "#c62828", "#1565c0", "#546e7a"


def fig_overview():
    fig = plt.figure(figsize=(18, 9.6))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.26)

    # ① 三种口径的 ratio 分布（只有收敛的才进）
    ax = fig.add_subplot(gs[0, 0])
    data, labs, cols = [], [], []
    data.append([r["nested_ratio"] for r in rows]); labs.append("nested\n(真值拓扑)"); cols.append(C_NEST)
    p = [r["ratio"] for r in rows if ok(r["voigt_prod"])]
    data.append(p); labs.append(f"voigt\n自由σ (n={len(p)})"); cols.append(C_PROD)
    q = [r["fix2_ratio"] for r in rows if ok(r["voigt_fix2"])]
    data.append(q); labs.append(f"voigt\n固定2%σ (n={len(q)})"); cols.append(C_FIX2)
    bp = ax.boxplot(data, widths=.55, patch_artist=True, showfliers=True,
                    flierprops=dict(ms=4, mfc="none", mec="gray"))
    for b, c in zip(bp["boxes"], cols):
        b.set(facecolor=c, alpha=.35, edgecolor=c)
    for med in bp["medians"]:
        med.set(color="k", lw=1.6)
    ax.axhline(1.0, color="k", ls="-", lw=.9)
    ax.axhline(RATIO_STRUCT, color="k", ls="--", lw=1)
    ax.text(3.45, RATIO_STRUCT * 1.05, "1.5", ha="right", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("rms_rel / 噪声底")
    ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_title("① 幅度：只有真值拓扑稳定贴底", fontsize=10.5)
    ax.grid(alpha=.3, axis="y")

    # ② 残差结构 acf（与幅度正交的判据）
    ax = fig.add_subplot(gs[0, 1])
    d2, l2, c2 = [], [], []
    d2.append([r["nested"]["acf"] for r in rows]); l2.append("nested"); c2.append(C_NEST)
    d2.append([r["voigt_prod"]["acf"] for r in rows if ok(r["voigt_prod"])])
    l2.append("voigt 自由σ"); c2.append(C_PROD)
    d2.append([r["voigt_fix2"]["acf"] for r in rows if ok(r["voigt_fix2"])])
    l2.append("voigt 固定σ"); c2.append(C_FIX2)
    bp = ax.boxplot(d2, widths=.55, patch_artist=True, showfliers=True,
                    flierprops=dict(ms=4, mfc="none", mec="gray"))
    for b, c in zip(bp["boxes"], c2):
        b.set(facecolor=c, alpha=.35, edgecolor=c)
    for med in bp["medians"]:
        med.set(color="k", lw=1.6)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.text(3.45, .52, "0.50 检出线", ha="right", fontsize=8)
    ax.set_ylabel("acf_max（滞后 1–5）")
    ax.set_xticklabels(l2, fontsize=8.5)
    ax.set_title("② 结构：固定σ 把幅度压到 1.0，残差仍有结构", fontsize=10.5)
    ax.grid(alpha=.3, axis="y")

    # ③ 配对：幅度 vs 结构（同一条谱，两口径都收敛）
    ax = fig.add_subplot(gs[0, 2])
    both = [r for r in rows if ok(r["voigt_prod"]) and ok(r["voigt_fix2"])]
    for r in both:
        ax.plot(r["nested_ratio"], r["nested"]["acf"], "o", ms=7, mfc=C_NEST, mec="k", mew=.4)
        ax.plot(r["fix2_ratio"], r["voigt_fix2"]["acf"], "s", ms=7, mfc=C_FIX2, mec="k", mew=.4)
        ax.plot(r["ratio"], r["voigt_prod"]["acf"], "^", ms=7, mfc=C_PROD, mec="k", mew=.4)
        ax.annotate("", xy=(r["fix2_ratio"], r["voigt_fix2"]["acf"]),
                    xytext=(r["nested_ratio"], r["nested"]["acf"]),
                    arrowprops=dict(arrowstyle="->", lw=.6, color="gray", alpha=.6))
    ax.axvline(1.0, color="k", lw=.9); ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlim(.7, 12)
    ax.set_xlabel("rms_rel / 噪声底"); ax.set_ylabel("acf_max")
    ax.set_title(f"③ 配对 n={len(both)}：箭头末端仍在结构区", fontsize=10.5)
    ax.grid(alpha=.3, which="both")
    ax.legend(handles=[plt.Line2D([], [], marker="o", mfc=C_NEST, mec="k", ls="", label="nested"),
                       plt.Line2D([], [], marker="s", mfc=C_FIX2, mec="k", ls="", label="voigt 固定σ"),
                       plt.Line2D([], [], marker="^", mfc=C_PROD, mec="k", ls="", label="voigt 自由σ")],
              fontsize=7.5, loc="upper right")

    # ④ R-hat 分布
    ax = fig.add_subplot(gs[1, 0])
    rr = np.array([r["voigt_prod"].get("rhat") for r in rows], float)
    ax.hist(np.log10(rr), bins=20, color=C_BAD, alpha=.85)
    ax.axvline(np.log10(RHAT_MAX), color="r", ls="--", lw=1.4)
    ax.text(np.log10(RHAT_MAX), ax.get_ylim()[1] * .92, " R-hat=1.05", color="r", fontsize=8)
    ax.set_xlabel("log10 R-hat_max（voigt 自由σ）"); ax.set_ylabel("条数")
    ax.set_title(f"④ 收敛性：{int((rr>RHAT_MAX).sum())}/{len(rr)} 条 R-hat>1.05", fontsize=10.5)
    ax.grid(alpha=.3, axis="y")

    # ⑤ 逃逸诊断：每链的 ln_tau 中位（K=2）
    ax = fig.add_subplot(gs[1, 1])
    diag = json.load(open(OUT / "mode_diag.json", encoding="utf-8"))
    l1 = np.log(1 / (2 * np.pi * 0.001) * 10 ** 1.5)     # 先验上界
    for i, d in enumerate(diag):
        for c, m in enumerate(d["chain_med"]):
            ax.plot([i, i], [min(m), max(m)], color="gray", lw=.8, zorder=1)
        for c, m in enumerate(d["chain_med"]):
            ax.plot([i] * len(m), m, "o", ms=5,
                    mfc=[C_PROD, C_FIX2, "#f9a825"][c % 3], mec="k", mew=.4, zorder=3)
    ax.axhline(7.0, color="r", ls="--", lw=1.2)
    ax.set_ylim(-11.5, 9.6)
    ax.text(.02, 7.35, "窗口外（τ > 1/f_min）", color="r", fontsize=8, va="bottom",
            bbox=dict(fc="w", ec="none", alpha=.85))
    ax.set_xlabel("样本（mode_diag 的 10 条）"); ax.set_ylabel("ln τ 各链中位")
    ax.set_title("⑤ 一个分量逃到窗口外，三链互换槽位", fontsize=10.5)
    ax.grid(alpha=.3, axis="y")

    # ⑥ 结论
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    n = len(rows)
    n_ok = sum(1 for r in rows if ok(r["voigt_prod"]))
    L = [f"涂层族 60 条（12 格 × 5，随机分层）", "",
         f"nested 真值拓扑   ratio 中位 {np.median([r['nested_ratio'] for r in rows]):.2f}"
         f"  max {max(r['nested_ratio'] for r in rows):.2f}",
         f"                  acf 中位 {np.median([r['nested']['acf'] for r in rows]):.2f}"
         f"  >0.5 的 {sum(1 for r in rows if r['nested']['acf']>0.5)}/60",
         "",
         f"voigt 自由σ       收敛 {n_ok}/60 = {n_ok/n:.0%}",
         f"                  收敛者 ratio 中位 {np.median(p):.2f}",
         f"                  acf 中位 {np.median([r['voigt_prod']['acf'] for r in rows if ok(r['voigt_prod'])]):.2f}",
         "",
         f"voigt 固定2%σ     收敛 {len(q)}/60 = {len(q)/n:.0%}",
         f"                  收敛者 ratio 中位 {np.median(q):.2f}",
         f"                  acf 中位 {np.median([r['voigt_fix2']['acf'] for r in rows if ok(r['voigt_fix2'])]):.2f}",
         "",
         "R-hat 爆表 ≠ 采样器坏：三条链把", "同一后验的不同标签实现分开拟合",
         "（K=3 对照：排序后 R-hat 107 → 1.00）"]
    ax.text(0, 1, "\n".join(L), va="top", fontsize=8.8, linespacing=1.65)

    fig.suptitle("涂层族 60 条：生产 Voigt K=2 的失配发生在幅度与结构两层", fontsize=13)
    return fig


if __name__ == "__main__":
    fig = fig_overview()
    p = OUT / "fig_overview.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("图:", p, "| n =", len(rows))
