"""report_2e.py —— 由 batch_2e 的汇总表生成一份可复核的交付说明。

只读 summary_all.csv / manifest.csv / 原始 CSV，不重新拟合。
用法： python report_2e.py [quick|full]
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CSVDIR = ROOT / "EISmeasurements" / "csv_2e"


def _f(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def med(vals):
    v = [x for x in (_f(a) for a in vals) if x is not None]
    return float(np.median(v)) if v else float("nan")


def main(mode: str) -> int:
    root = ROOT / "out" / ("2E_quick" if mode == "quick" else "2E_full")
    tab = root / "summary_all.csv"
    if not tab.exists():
        print("先跑 batch_2e.py --report")
        return 1
    with tab.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with (CSVDIR / "manifest.csv").open(encoding="utf-8-sig", newline="") as fh:
        man = {r["stem"]: r for r in csv.DictReader(fh)}

    ok = [r for r in rows if r["K"] not in ("", None)]
    fig = [r for r in rows if r["fit_png"]]
    cir = [r for r in rows if r["circuit_png"]]
    cur = [r for r in rows if r["curve_csv"]]
    grades = Counter(r["grade"] for r in ok)
    ks = Counter(r["K"] for r in ok)
    n_out = sum(1 for r in ok if (_f(r["p_peak_out_max"]) or 0) > 0.5)
    n_nan = sum(1 for r in ok if r["p_peak_out_max"] == "NaN")
    n_neg = sum(1 for r in ok if (_f(r["frac_neg"]) or 0) > 0.01)
    n_L = sum(1 for r in ok if str(r["include_L"]).lower() == "true")
    t_sum = sum(_f(r["t_fit_s"]) or 0 for r in ok)

    # ---- 原始数据 survey（与拟合无关，解释结果性质）----
    # 判据写死在这里，报告里引用的是同一批数，不是手抄的结论：
    #   ind     : 频率最高的 3 点里 Im Z>0 占多数（感抗尾）
    #   ripple  : 频率最低 5 点 Re 的 (max-min)/|mean|
    #   jump    : 频率最高 10 点内相邻 |Z| 相对跳变 > 20%
    #   neg     : 谱里出现负 Re 点
    survey = []
    for st, m in man.items():
        A = np.loadtxt(CSVDIR / m["csv"], delimiter=",", skiprows=1)
        f, Z = A[:, 0], A[:, 1] + 1j * A[:, 2]
        o = np.argsort(f)
        f, Z = f[o], Z[o]
        za = np.abs(Z)
        ph = np.degrees(np.angle(Z))
        rz = np.abs(np.diff(za)) / za[:-1]
        x5 = Z.real[:5]
        survey.append(dict(j=float(m["j_mA_cm2"]),
                         ind=np.mean(Z.imag[-3:] > 0) >= 0.5,
                         ph_lo=ph[0], ph_hi=ph[-1],
                         zmax=za.max(), dr=za.max() / za.min(),
                         ripple=float((x5.max() - x5.min()) / abs(x5.mean())),
                         jump=int(np.max(rz[-10:]) > 0.20),
                         neg=int(np.sum(Z.real < 0))))
    n_ind = sum(x["ind"] for x in survey)
    n_outlier = sum(1 for x in survey if x["jump"] or x["neg"])
    js = sorted({x["j"] for x in survey})
    ph_by_j = {j: float(np.median([x["ph_lo"] for x in survey if x["j"] == j]))
               for j in js}
    other_ph = [v for j, v in ph_by_j.items() if j != 5.0]

    L = []
    A = L.append
    A("# 2E 实测 EIS 批量拟合交付说明\n")
    A(f"- 数据来源：`EISmeasurements/2E*.xlsx` 共 13 个文件，每个 9 个电流密度"
      " sheet（5 / 50 / 100 / 150 / 200 / 350 / 500 / 670 / 750 mA cm⁻²）= **117 条谱**，"
      "每条 56 点、1 Hz – 100 kHz（5 decade）。\n"
      "- 转换：`convert_2e.py` → `EISmeasurements/csv_2e/*.csv`（`f_Hz,Zre,Zim`）。"
      "Gamry 每个 sheet 前 97 行是仪器元数据，阻抗谱在 `ZCURVE` 段；数字存成文本，"
      "已显式转 float。**虚部符号**用同行 `Zmod`/`Zphz` 自检：117 条全部是"
      "真 Im Z（容性为负、高频感抗为正），无需翻号；相位最大误差 0.00022°、"
      "|Z| 最大相对误差 5.4e-6。\n"
      f"- 采样档：**{mode}**。"
      + ("`n_warmup=800, n_samples=400, num_chains=2`（bm_real 的粗筛档）。"
         "目的是在合理时间内给 117 条谱各出一套完整图；"
         "**elpd / 采样分级 / R̂ / ESS / divergence 不能当结论**，"
         "用 `python batch_2e.py --full --jobs 3` 全量重跑后才可引用。\n"
         if mode == "quick" else
         "`n_warmup=3000, n_samples=1000, num_chains=3`，与 `evidence/b2x` 的四组"
         "NUTS 对照同设置。\n")
      + "\n## 1. 交付内容\n")
    A(f"| 项目 | 数量 | 位置 |\n|---|---|---|\n"
      f"| 2×3 诊断图（Nyquist / Bode \\|Z\\| / Bode 相位 / DRT / 残差 Re / 残差 Im）"
      f" | {len(fig)} | `figures/2E/*_fit.png` |\n"
      f"| 等效电路图（参数 = 后验中位数） | {len(cir)} | `figures/2E/*_circuit.png` |\n"
      f"| 拟合后的逐点数据 | {len(cur)} | `figures/2E/curves/*_fit.csv` |\n"
      f"| 指标总表 | {len(rows)} 行 | `out/2E_{mode}/summary_all.csv` |\n"
      "| 图片索引页 | 1 | `figures/2E/index.html`（浏览器打开，按文件 × 电流密度排列） |\n"
      f"| 打包 | 1 | `2E_figures_{mode}.zip` |\n")
    A("\n每张诊断图里**蓝色空圈 = 实测点、红线 = 后验中位拟合、淡红带 = 90% 区间**；"
      "DRT 面板灰虚线是只用数据做的 Tikhonov 预判断、紫线是拟合分量的解析 DRT。"
      "`curves/*.csv` 每点给出观测值、拟合中位数、90% 上下界与残差，"
      "可直接重画 Bode/Nyquist。\n")

    A("\n## 2. 批量运行结果\n")
    A(f"- 有结果 JSON：{len(ok)} / {len(rows)}；诊断图 {len(fig)}、电路图 {len(cir)}、"
      f"拟合曲线 {len(cur)}。\n"
      f"- 累计拟合墙钟时间 {t_sum/60:.1f} 分钟（各 worker 并行，不是总等待时间）。\n"
      f"- 采样分级分布：{dict(sorted(grades.items(), key=lambda x: str(x[0])))}"
      "（2 = 严格通过，1 = 可用，0 = 不可用）。\n"
      f"- 选中的 K 分布：{dict(sorted(ks.items(), key=lambda x: str(x[0])))}。\n"
      f"- `P(峰位出窗)>0.5` 的谱：{n_out}；峰位诊断不可用（NaN）：{n_nan}；"
      f"`φζ>1` 超 1% 容忍：{n_neg}；自动启用电感 L：{n_L}。\n"
      f"- `acf` 中位 {med([r['acf'] for r in ok]):.2f}、"
      f"`rms_re/σ_re` 中位 {med([ (float(r['rms_re'])/float(r['sigma_re'])) if _f(r['rms_re']) and _f(r['sigma_re']) and float(r['sigma_re'])>0 else None for r in ok]):.2f}。\n")

    A("\n### 按电流密度（中位值）\n")
    # 整个表格必须一次 append：逐行 append 会被 join 插入空行，Markdown 表格会断掉
    byj = defaultdict(list)
    for r in ok:
        byj[r["j_mA_cm2"]].append(r)
    tab = ["| j / mA cm⁻² | n | ph(1 Hz) / ° | \\|Z\\|max / Ω | K | 分级 | acf | P(峰出窗)>0.5 |",
           "|---|---|---|---|---|---|---|---|"]
    for j in sorted(byj, key=float):
        g = byj[j]
        ph = [x["ph_lo"] for x in survey if f"{x['j']:g}" == j]
        zm = [x["zmax"] for x in survey if f"{x['j']:g}" == j]
        tab.append(
            f"| {j} | {len(g)} | {np.median(ph):.1f} | {np.median(zm):.4g} | "
            f"{med([x['K'] for x in g]):.0f} | {med([x['grade'] for x in g]):.0f} | "
            f"{med([x['acf'] for x in g]):.2f} | "
            f"{sum(1 for x in g if (_f(x['p_peak_out_max']) or 0) > 0.5)}/{len(g)} |")
    A("\n".join(tab) + "\n")

    A("\n## 3. 这批数据本身的性质（解释结果，不是拟合失败）\n")
    A(f"- 高频 3 点 `Im Z>0`（感抗尾）：**{n_ind}/117** 条 → 程序自动加 L。"
      "但按此前真实数据的经验，L 常是**装置/引线**而非元件，裁剪高频段后若 L 剧烈变化就应只当拟合装置。\n"
      f"- 低频相位：5 mA 组中位 {ph_by_j.get(5.0, float('nan')):.1f}°（明显扩散/电容尾），"
      f"其余 8 组的中位落在 {min(other_ph):.1f}° ~ {max(other_ph):.1f}°，"
      "说明 **1 Hz 处还没走到直流极限**。\n"
      f"- |Z| 最大值的中位 {np.median([x['zmax'] for x in survey]):.3f} Ω，"
      f"动态范围（|Z|max/|Z|min）中位 {np.median([x['dr'] for x in survey]):.1f} 倍；"
      f"低频 5 点 Re 的相对起伏 `(max−min)/|mean|` 中位 "
      f"{100*np.median([x['ripple'] for x in survey]):.0f}%，噪声与系统偏差相对信号不小。\n"
      "- 低频端没有走平 ⇒ DRT 的主峰落在窗口下方（全量试跑的 5 mA 谱峰位 0.04 Hz，"
      "比 f_min=1 Hz 低 1.4 decade），有限 K 的 HN 都要靠把 τ 推到边界去补这条尾巴。"
      f"**所以 5 mA 组 13 条全部 `P(峰位出窗)>0.5` 是数据频率窗口决定的，不是 K 选错**；"
      f"全批共 {n_out}/117 条出窗，其余电流密度组多为 0~1 条。\n"
      f"- 按「高频 10 点内相邻 |Z| 跳变 >20% 或出现负 Re 点」判据，{n_outlier}/117 条有高频离群点，"
      "这是原始谱里的离群点，转换时**没有删除**（保持数据原样），报告里应连同残差图一起看。\n")

    A("\n## 4. 使用限制（必须随图一起说明）\n")
    A("1. " + ("**本批是粗筛采样**。" if mode == "quick" else "本批是全量采样，但")
      + "elpd_loo 只能在同一批同样设置下相对比较；"
      "分级 0 的模型其 K、参数、elpd 都不可引用。\n"
      "2. 频率窗口 1 Hz – 100 kHz 没覆盖直流极限，" 
      "峰位在窗口外的分量面积不可信，不能据此给「弛豫过程数」或极化电阻的定值。\n"
      "3. 基础模型默认 `ζ=1`（ZARC 基底）、`diffusion=none`；"
      "低频 −45° 尾若要显式建模，需要单独跑 `--diffusion Ws/Wo` 对照，本批没做。\n"
      "4. 所有图内文字是英文（matplotlib 默认字体无 CJK 字形）；"
      "中文说明在 `index.html` 和本文件里。\n")

    A("\n## 5. 复现\n")
    A("```bash\n"
      "python convert_2e.py                       # xlsx -> 117 CSV（含符号自检）\n"
      f"python batch_2e.py --{mode} --jobs 3        # 批量拟合\n"
      f"python batch_2e.py --report --{mode}       # 平铺图片 / 曲线 / 索引 / zip\n"
      f"python report_2e.py {mode}                 # 生成本文件\n"
      "python batch_2e.py --full --jobs 3 --every # 每个源文件抽 1 条做全量严格复跑\n"
      "```\n")
    A("单次运行的完整控制台输出在 `out/2E_%s/w*/<stem>/run.log`，"
      "子任务日志在 `out/2E_%s/logs/`。断点续跑直接重跑同一条命令即可"
      "（已有 `real_<stem>.json` 会跳过）。\n" % (mode, mode))

    out = ROOT / "figures" / ("2E" if mode == "quick" else "2E_full") / "REPORT_2E.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"写出 {out}  ({len(L)} 段, {len(rows)} 条谱)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "quick"))
