# -*- coding: utf-8 -*-
"""吸附电感 M6 抽样验证：数据盘点 / 分层抽样 / 结果汇总 / 置信上界。

三个子命令：
  python evidence/ad_m6/sampling_check.py inventory          # 盘点 .z 含环谱
  python evidence/ad_m6/sampling_check.py prepare           # 转 csv + 生成抽样清单
  python evidence/ad_m6/sampling_check.py summarize         # 读 out/_m6_samples 结果汇总

数据按仓库约定不入库：`D:/EIS 数据/EIS/*.z` 为 ZView 原始导出（外部），
2E 谱在 `EISmeasurements/csv_2e/`（被 .gitignore 忽略）。结果 json 在
`out/_m6_samples/`（同样不入库），因此 summarize 只在本地跑过拟合后才有输出。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import numpy as np

ZDIR = r"D:\EIS 数据\EIS"
OUT = "out/_m6_samples"
MANIFEST = "EISmeasurements/csv_2e/manifest.csv"
SEED = 20261107
REAL_PICK = ["AZ31-12Li_0", "AZ31-4Li_50", "AZ31-4Li_100", "AZ31-4Li_150"]
CONTROL_MUST = ["2EMC1-2_0750mA", "2EP-2_exp1_0005mA", "2EMC1-1_0750mA"]


def read_z(path):
    """ZView .z：跳过前 11 行，逗号分隔，列 0=f / 4=Z' / 5=Z''。"""
    rows = []
    with open(path, encoding="utf-8-sig", errors="replace") as fp:
        lines = fp.readlines()
    for ln in lines[11:]:
        p = ln.strip().split(",")
        if len(p) >= 6:
            try:
                rows.append([float(p[0]), float(p[4]), float(p[5])])
            except ValueError:
                continue
    a = np.array(rows)
    return a[:, 0], a[:, 1] + 1j * a[:, 2]


def inventory():
    print("%-22s %4s %12s %6s %12s %10s %10s" %
          ("file", "n", "f范围(Hz)", "Im>0", "Im>0频窗", "Im峰位", "Im峰值"))
    for p in sorted(glob.glob(os.path.join(ZDIR, "*.z"))):
        f, Z = read_z(p)
        pos = Z.imag > 0
        i = int(np.argmax(Z.imag))
        win = ("%.3g-%.3g" % (f[pos].min(), f[pos].max())) if pos.any() else "-"
        print("%-22s %4d %.2g-%.2g %6d %12s %10.4g %10.1f" %
              (os.path.basename(p), len(f), f.min(), f.max(), pos.sum(),
               win, f[i], Z.imag[i]))


def prepare():
    os.makedirs(os.path.join(OUT, "z_csv"), exist_ok=True)
    for p in sorted(glob.glob(os.path.join(ZDIR, "*.z"))):
        f, Z = read_z(p)
        stem = os.path.basename(p).replace(".z", "")
        with open(os.path.join(OUT, "z_csv", stem + ".csv"), "w",
                  encoding="utf-8", newline="") as fp:
            fp.write("f_Hz,Zre,Zim\n")
            for fi, zi in zip(f, Z):
                fp.write("%.10g,%.10g,%.10g\n" % (fi, zi.real, zi.imag))

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    strata = {}
    for r in rows:
        m = re.match(r"(2E[A-Za-z0-9-]+?)_", r["stem"])
        strata.setdefault(m.group(1) if m else r["stem"].split("_")[0], []).append(r)
    rng = random.Random(SEED)
    picked = []
    for key in sorted(strata):                 # 7 个系列各 1 条（中间电流段）
        g = sorted(strata[key], key=lambda r: float(r["j_mA_cm2"]))
        picked.append(rng.choice(g[len(g) // 3: 2 * len(g) // 3 + 1]))
    for must in CONTROL_MUST:                 # 补 3 条点名谱
        cand = next((r for r in rows if r["stem"] == must), None)
        if cand:
            picked.append(cand)
    seen, sample = set(), []
    for r in picked:
        if r["stem"] not in seen:
            seen.add(r["stem"]); sample.append(r)
    sample.sort(key=lambda r: r["stem"])

    man = {"seed": SEED,
           "real_stems": REAL_PICK,
           "real": [os.path.join(OUT, "z_csv", s + ".csv") for s in REAL_PICK],
           "control_stems": [r["stem"] for r in sample],
           "control": [os.path.join("EISmeasurements", "csv_2e", r["csv"])
                       for r in sample],
           "series_covered": sorted(strata)}
    with open(os.path.join(OUT, "sample_manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(man, fp, ensure_ascii=False, indent=1)
    print("真实含环:", REAL_PICK)
    print("对照谱 (%d 条，覆盖 %d 系列):" % (len(sample), len(strata)))
    for r in sample:
        print("  %-28s j=%-6s zmax=%s" % (r["stem"], r["j_mA_cm2"], r["zmax"]))
    print("清单:", os.path.join(OUT, "sample_manifest.json"))


def edges():
    """边界峰分析：**与 detect_ad() 完全一致的频段筛选与峰位定义**。

    第三轮审查指出旧版用「全频段 argmax(Z.imag)」，而门控用的是
    「最低 frac_lo 频段内 argmin(-Im)」，二者不是等价回放。本版改为：

        lf = log10(f); m = lf <= quantile(lf, 0.40)
        f_band, Z_band = f[m], Z[m]
        k = argmin(-Z_band.imag)                 # = band 内 Im 最大
        f_peak = f_band[k]
        k_left = count_nonzero(f_band < f_peak)  # 主判据（离散点数）
        k_right = len(f_band) - 1 - k

    **主判据是离散点数 k_left / k_right**，与 detect_ad 的
    `interior = 2 <= k <= n_band-3` 逐条一致；`d_low` 只作辅助记录与
    跨数据集比较，不依赖「对数均匀采样」假设，也不声称与点数等价。
    """
    FRAC_LO = 0.40
    print("%-26s %9s %10s %8s %6s %6s %7s %-18s" %
          ("stem", "f_min", "f_peak", "n_band", "k_left", "k_rgt", "d_low", "门控等价形状状态"))
    allrows = []
    for p in sorted(glob.glob(os.path.join(ZDIR, "*.z"))):
        f, Z = read_z(p)
        allrows.append((os.path.basename(p).replace(".z", ""), f, Z))
    man_path = os.path.join(OUT, "sample_manifest.json")
    if os.path.exists(man_path):
        man = json.load(open(man_path, encoding="utf-8"))
        for stem in man["control_stems"]:
            a = np.loadtxt(os.path.join("EISmeasurements", "csv_2e", stem + ".csv"),
                           delimiter=",", skiprows=1)
            allrows.append((stem, a[:, 0], a[:, 1] + 1j * a[:, 2]))

    cnt = {"edge_peak": 0, "monotonic": 0, "interior": 0, "right_edge": 0}
    for stem, f, Z in allrows:
        o = np.argsort(f); f, Z = f[o], Z[o]
        lf = np.log10(f)
        m = lf <= float(np.quantile(lf, FRAC_LO))
        f_band, Z_band = f[m], Z[m]
        n_band = len(f_band)
        k = int(np.argmin(-Z_band.imag))
        f_pk = float(f_band[k])
        k_left = int(np.count_nonzero(f_band < f_pk))
        k_right = n_band - 1 - k
        d_low = float(np.log10(f_pk / f_band[0]))
        # 注意：detect_ad 还要求右侧 ≥2 点（k <= n_band-3），这里一并检查
        interior = bool(2 <= k <= n_band - 3)
        if not stem.startswith("AZ31"):
            state = "n/a(容性对照)"
        elif interior:
            state = "interior(可确认)"; cnt["interior"] += 1
        elif k_left == 1:
            state = "edge_peak(贴边)"; cnt["edge_peak"] += 1
        elif k_left == 0:
            state = "outside/单调"; cnt["monotonic"] += 1
        else:
            state = "right_edge(右侧不足)"; cnt["right_edge"] += 1
        print("%-26s %9.3g %10.4g %8d %6d %6d %7.3f %-18s" %
              (stem[:26], f_band[0], f_pk, n_band, k_left, k_right, d_low, state))
    print("\n主判据 = 离散点数（k_left/k_right），与 detect_ad 的 interior 判定逐条一致。")
    print("  d_low = log10(f_peak/f_min) 仅作辅助记录 / 跨数据集比较；")
    print("  仅当频率近似对数均匀时才有 k_left ≈ d_low·dens，不写成『等价』。")
    print("  AZ31: interior %d、edge_peak %d、k_left=0 %d、右侧不足 %d" %
          (cnt["interior"], cnt["edge_peak"], cnt["monotonic"], cnt["right_edge"]))
    print("  新规则只把 k_left=1 从 enter=False 改为 enter=True/strong=False，")
    print("  interior 判定路径一字不动 → strong 集合不变（控制流逻辑不变性）。")


def _load(root):
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*", "real_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        stem = os.path.basename(p)[5:-5]
        g, s = d.get("ad_gate") or {}, d.get("ad_selection") or {}
        out.append((stem, d.get("K"), d.get("elpd_loo"),
                    d.get("rms_im"), d.get("acf"),
                    g.get("status", "-"), g.get("enter", "-"),
                    s.get("choice", "-"), s.get("status", "-"),
                    s.get("delta"), s.get("dse"), s.get("threshold")))
    return out


def clopper_upper(n, x=0, alpha=0.05):
    """零误报时单侧 (1-alpha) 上界：1 - alpha^(1/n)。"""
    if n == 0:
        return float("nan")
    return 1 - alpha ** (1.0 / n) if x == 0 else float("nan")


def summarize():
    for tag, root in [("真实含环", os.path.join(OUT, "fit_real")),
                      ("2E 对照", os.path.join(OUT, "fit_control"))]:
        rows = _load(root)
        print("=" * 108)
        print("%s (%d 条)" % (tag, len(rows)))
        print("%-24s %4s %10s %9s %6s %-20s %-6s %-22s" %
              ("stem", "K", "elpd", "rms_im", "acf", "gate", "enter", "selection"))
        for r in rows:
            fmt = lambda v, spec: (spec % v) if isinstance(v, (int, float)) else str(v)
            print("%-24s %4s %10s %9s %6s %-20s %-6s %-22s" %
                  (r[0][:24], str(r[1]), fmt(r[2], "%.1f"), fmt(r[3], "%.3f"),
                   fmt(r[4], "%.3f"), r[5], str(r[6]), "%s/%s" % (r[7], r[8])))
        picked = [r for r in rows if r[1] is not None]
        switched = [r for r in picked if r[7] == "ad"]
        print("  最终切 ad: %d / %d（有效拟合）" % (len(switched), len(picked)))
        if tag == "2E 对照":
            n = len(picked)
            print("  零误报 95%% 单侧上界: %.1f%%（本轮有效 n=%d）"
                  % (100 * clopper_upper(n, len(switched)), n))
            print("  要达到上界 ≤5% 需 n≥59：" + "、".join(
                "n=%d → %.1f%%" % (nn, 100 * clopper_upper(nn))
                for nn in (30, 59, 117)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("inventory", "prepare", "summarize", "edges"))
    a = ap.parse_args()
    {"inventory": inventory, "prepare": prepare, "summarize": summarize,
     "edges": edges}[a.cmd]()
