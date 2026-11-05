"""离线回放：用现有已落盘结果检验 K_SELECTION_AMBIGUOUS 的判定规格。

不修改 bmdrt 任何代码，不重跑 NUTS。只解析 run.log / JSON，
按「预先登记」的规则判定每条谱会落到哪个状态，用于检查是否误伤。

运行：
  D:/autoeis_env/Scripts/python.exe replay_ambiguous.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

# ---- 预先登记的阈值（改代码前先固定，避免事后调参）----
TH = dict(
    p_peak_out=0.50,      # 沿用现有硬筛选阈值，不放宽
    elpd_abs=2.0,         # 「差 < max(2, dse) 取更小 K」的同一常数
    elpd_k=1.0,           # 显著更优要求 elpd_diff > max(elpd_abs, elpd_k*dse)
    rhat=1.01, ess=400.0, div=0, tree_cap=0.01, grade_min=1,
)

# 有 K 扫描的谱（其余单候选 / 无 K 比较，不参与触发）
CASES = {
    "K1_none": dict(log="results/K1_none/run.log",
                    js="results/K1_none/real_Blank_0.1KBr_pH10_EIS_PtCu_1.2V.json"),
    "K1_cut1Hz": dict(log="results/K1_cut1Hz/run.log",
                      js="results/K1_cut1Hz/real_Blank_cut1Hz.json"),
    "K2_Ws": dict(log="results/K2_Ws/run.log",
                  js="results/K2_Ws/real_Blank_full_Ws.json"),
}


def parse_log(p: Path):
    txt = p.read_text(encoding="utf-8", errors="replace")
    cands, cur = {}, None
    for ln in txt.splitlines():
        m = re.search(r"--- K=(\d+) 拟合中", ln)
        if m:
            cur = int(m.group(1))
            cands[cur] = {}
            continue
        if cur is None:
            continue
        m = re.search(r"R̂max=([\d.]+)\s+bulk=([\d.]+)\s+tail=([\d.]+)\s+"
                      r"div=(\d+)\s+触顶=([\d.]+)%\s+→ 分级 (\d)", ln)
        if m:
            cands[cur].update(rhat=float(m.group(1)), bulk=float(m.group(2)),
                              tail=float(m.group(3)), div=int(m.group(4)),
                              cap=float(m.group(5)) / 100.0,
                              grade=int(m.group(6)))
            continue
        m = re.search(r"elpd=(-?[\d.]+)\s+rms/σ=([\d.]+)/([\d.]+)\s+"
                      r"acf=([\d.]+)\s+P\(峰位出窗\)max=([\d.]+)", ln)
        if m:
            cands[cur].update(elpd=float(m.group(1)), rms_re=float(m.group(2)),
                              rms_im=float(m.group(3)), acf=float(m.group(4)),
                              p_out=float(m.group(5)))
    # ③ 配对比较
    pair = {}
    for ln in txt.splitlines():
        m = re.match(r"\s*K=(\d+)\s+\d+\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s", ln)
        if m:
            pair[int(m.group(1))] = dict(elpd=float(m.group(2)),
                                         elpd_diff=float(m.group(3)),
                                         dse=float(m.group(4)))
    return cands, pair


def decide(cands: dict, pair: dict, kk_status: str):
    """按新规则判定；返回 (status, best_K, reason)。"""
    if kk_status not in ("ok",):
        return "rejected_by_kk", None, f"KK status={kk_status}"
    bad = {}
    for K, c in cands.items():
        why = []
        if c.get("grade", 0) < TH["grade_min"]:
            why.append(f"分级={c.get('grade')}")
        if c.get("div", 0) != TH["div"]:
            why.append(f"div={c.get('div')}")
        if c.get("cap", 1) >= TH["tree_cap"]:
            why.append(f"触顶={c.get('cap'):.1%}")
        if c.get("rhat", 9) > TH["rhat"]:
            why.append(f"R̂={c.get('rhat')}")
        if min(c.get("bulk", 0), c.get("tail", 0)) < TH["ess"]:
            why.append(f"ESS={c.get('bulk')}/{c.get('tail')}")
        if why:
            bad[K] = "; ".join(why)
    ok = {K: c for K, c in cands.items() if K not in bad}
    if not ok:
        return "rejected", None, "所有候选采样不合格：" + json.dumps(bad, ensure_ascii=False)

    best = max(ok, key=lambda K: ok[K]["elpd"])
    small = min(ok)
    if len(ok) == 1:
        return "not_compared", best, "只有一个可接受候选"

    diff = pair.get(small, {}).get("elpd_diff", abs(ok[best]["elpd"] - ok[small]["elpd"]))
    dse = pair.get(small, {}).get("dse", float("nan"))
    thr = max(TH["elpd_abs"], TH["elpd_k"] * (dse if dse == dse else 0.0))
    significant = diff > thr
    out_max = ok[best].get("p_out", 0.0)

    if best > small and significant and out_max > TH["p_peak_out"]:
        return ("ambiguous", best,
                f"K={best} 的 LOO 优于 K={small}（Δ={diff:.1f} > {thr:.1f}），"
                f"但其分量 P(峰出窗)={out_max:.2f} > {TH['p_peak_out']}")
    if not significant:
        return "converged", small, f"Δ={diff:.1f} ≤ {thr:.1f}，按简约优先取 K={small}"
    return "converged", best, f"Δ={diff:.1f} > {thr:.1f} 且峰位在窗内，取 K={best}"


def main():
    rows = []
    for name, cfg in CASES.items():
        cands, pair = parse_log(ROOT / cfg["log"])
        js = json.loads((ROOT / cfg["js"]).read_text(encoding="utf-8"))
        st, best, why = decide(cands, pair, js.get("kk", {}).get("status", "ok"))
        rows.append(dict(
            case=name, kk=js.get("kk", {}).get("status"),
            old_status=js.get("k_selection_status"), old_K=js.get("K"),
            new_status=st, new_K_reference=best,
            cands=", ".join(
                f"K={K}: elpd={c['elpd']:.1f}, P_out={c['p_out']:.2f}, "
                f"grade={c['grade']}" for K, c in sorted(cands.items())),
            reason=why,
        ))
    # 单候选组（网格 + 强制 K 基线）只登记，不触发
    for name in ("K2_none",):
        js_p = next((ROOT / "results" / name).glob("real_*.json"))
        js = json.loads(js_p.read_text(encoding="utf-8"))
        rows.append(dict(case=name, kk=js.get("kk", {}).get("status"),
                         old_status=js.get("k_selection_status"), old_K=js.get("K"),
                         new_status="not_compared", new_K_reference=js.get("K"),
                         cands=f"K={js.get('K')}（单候选）",
                         reason="单候选，不涉及 K 比较"))
    for name in ("G_K1_none", "G_K2_none", "G_K1_Ws", "G_K2_Ws",
                 "G_K1_Wo", "G_K2_Wo"):
        d = ROOT / "results" / name
        js_p = next(d.glob("real_*.json"))
        js = json.loads(js_p.read_text(encoding="utf-8"))
        rows.append(dict(case=name, kk=js.get("kk", {}).get("status"),
                         old_status=js.get("k_selection_status"), old_K=js.get("K"),
                         new_status="not_compared", new_K_reference=js.get("K"),
                         cands=f"K={js.get('K')}（--krange K,K）",
                         reason="强制单 K，不涉及 K 比较"))

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "replay_ambiguous.csv", index=False, float_format="%.4g")
    print("=== 预先登记的阈值 ===")
    print(TH)
    print("\n=== 离线回放 ===")
    print(df[["case", "old_status", "old_K", "new_status",
              "new_K_reference"]].to_string(index=False))
    print("\n=== 触发理由 ===")
    for _, r in df.iterrows():
        if r.new_status == "ambiguous" or r.old_status != r.new_status:
            print(f"- {r.case}: {r.old_status} → {r.new_status}")
            print(f"    {r.cands}")
            print(f"    {r.reason}")
    print("\nDONE ->", ROOT / "replay_ambiguous.csv")


if __name__ == "__main__":
    main()
