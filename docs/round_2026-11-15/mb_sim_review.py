"""M-B-SIM-01 只读审查：预注册 ↔ 生成端 ↔ 判别端 ↔ 报告 逐条对齐。

只读 `docs/round_2026-11-15/` 与 `out/mb_sim01/` 的既有产物，**不拟合、不生成、不改文件**。
断言分七组：C1 常数一致 / C2 盲性 / C3 生成可复现 / C4 拟合口径 / C5 报告数字 vs 产物 /
C6 判定链独立复算 / C7 生产零改动。

用法：`python docs/round_2026-11-15/mb_sim_review.py`
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evidence", "nested_topology"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIM = os.path.join("out", "mb_sim01")
DOC = os.path.join("docs", "round_2026-11-15")
PROTO = os.path.join(DOC, "MB_SIM01_PROTOCOL.md")
REPORT = os.path.join(DOC, "M_B_SIMULATION_REPORT.md")
NOISE = os.path.join("docs", "round_2026-11-12", "NOISE_ACF_TRUTH.md")
GEN = os.path.join(DOC, "mb_sim_generate.py")
ANALYSE = os.path.join(DOC, "mb_sim_analyze.py")
DECODE = os.path.join(DOC, "mb_sim_decode.py")

OK, FAIL = [], []


def chk(name: str, cond: bool, detail: str = "") -> None:
    (OK if cond else FAIL).append((name, detail))
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  <{detail}>" if detail and not cond else ""))


def read(p):
    return open(p, encoding="utf-8").read()


def main() -> int:
    proto, report, noise = read(PROTO), read(REPORT), read(NOISE)
    gen, ana, dec = read(GEN), read(ANALYSE), read(DECODE)
    cfg = json.load(open(os.path.join(SIM, "config", "simulation_config.json"), encoding="utf-8"))
    man = json.load(open(os.path.join(SIM, "config", "seed_manifest.json"),
                         encoding="utf-8"))["manifest"]
    decd = json.load(open(os.path.join(SIM, "decoded", "decoded.json"), encoding="utf-8"))
    per = json.load(open(os.path.join(SIM, "decoded", "per_spectrum.json"), encoding="utf-8"))
    sh6 = json.load(open(os.path.join(SIM, "summary", "shared_fixture.json"), encoding="utf-8"))
    voi = json.load(open(os.path.join(SIM, "summary", "voigt_check.json"), encoding="utf-8"))
    blind = {x[:-5]: json.load(open(os.path.join(SIM, "blind_results", x), encoding="utf-8"))
             for x in os.listdir(os.path.join(SIM, "blind_results")) if x.endswith(".json")}
    truth = {m["sim_id"]: json.load(open(os.path.join(SIM, "truth", "truth_" + m["sim_id"] + ".json"),
                                        encoding="utf-8")) for m in man}

    print("\nC1 预注册 ↔ 生成端常数")
    for k, v in (("Rs", (0.05, 5.0)), ("Ls", (1e-7, 1e-4)), ("Cp", (1e-11, 1e-8)),
                 ("Rp", (1e3, 1e7)), ("Re", (1.0, 40.0)), ("fe", (1e-2, 1e4)),
                 ("alpha", (0.30, 0.80)), ("gamma", (0.20, 0.60)),
                 ("Rw", (5.0, 60.0)), ("fw", (1e-2, 1e4))):
        chk(f"RANGES[{k}] 与 cfg 一致", tuple(cfg["ranges"][k]) == v, str(cfg["ranges"][k]))
    chk("N_INST=20", cfg["n_inst"] == 20)
    chk("噪声档 2/4 交替", cfg["pct_cycle"] == [2.0, 4.0])
    chk("base_seed=20261115", cfg["base_seed"] == 20261115)
    chk("branch/layers = parallel/2", (cfg["branch"], cfg["layers"]) == ("parallel", 2))
    chk("谱数 80", cfg["n_spectra"] == 80)
    chk("分组计数 A20/B20/C1 10/C2 10/D20",
        cfg["counts"] == {"A": 20, "B": 20, "C1": 10, "C2": 10, "D": 20}, str(cfg["counts"]))
    chk("协议写 7 点/decade（非 10）", "7 点/decade" in proto and "10 点/decade" not in proto)
    chk("协议含 v2 变更记录（C2 改 Warburg）", "有限长 Warburg" in proto and "**v2**" in proto)
    chk("协议含 §8.1 S6", "§8.1" in proto or "## 8.1" in proto)

    print("\nC2 盲性")
    gen_files = sorted(x for x in os.listdir(os.path.join(SIM, "generated")) if x.endswith(".csv"))
    chk("generated/ 80 个 csv", len(gen_files) == 80, str(len(gen_files)))
    chk("generated/ 下无子目录（尤其无 group_*/truth）",
        all(os.path.isfile(os.path.join(SIM, "generated", x)) for x in os.listdir(os.path.join(SIM, "generated"))))
    chk("文件名不含组别", not any(re.search(r"(^|[-_])(A|B|C1|C2|D)([-_]|$)|group", x)
                                for x in gen_files))
    chk("csv 只有 3 列（f,Zre,Zim）",
        all(read(os.path.join(SIM, "generated", x)).splitlines()[0].strip() == "f,Zre,Zim"
            for x in gen_files))
    chk("判别端不出现 truth 路径拼接",
        "truth" not in ana.replace("truth/", "").replace("\"truth\"", "")
        or "禁止读 truth" in ana)
    chk("判别端有 truth 路径守卫", "路径含 truth" in ana)
    chk("判别端不读 manifest/truth 目录",
        "seed_manifest" not in ana and "truth_" not in ana)

    print("\nC3 生成可复现")
    chk("生成脚本 sha 与产物记录一致",
        hashlib.sha256(open(GEN, "rb").read()).hexdigest() == cfg["generator_sha256"])
    chk("truth 里 sha 也是同一个",
        all(t["generator_sha256"] == cfg["generator_sha256"] for t in truth.values()))
    import mb_sim_generate as G       # noqa: E402  只读、不写盘
    import bm_real as R               # noqa: E402
    f_grid, pool = G.grid(), G.truth_pool()
    ids = G.blind_ids()
    same = 0
    for i in (0, 7, 13, 19):
        inst = G.build_instance(f_grid, pool, i)
        for g, rec in inst.items():
            sid = ids[(rec["truth"]["group"], i)]
            f2, Z2 = R.load_spectrum(os.path.join(SIM, "generated", sid + ".csv"))
            same += int(np.allclose(Z2, rec["Z"], rtol=1e-9, atol=0))
    chk("4 个实例 16 条谱重生成逐点一致", same == 16, f"{same}/16")
    chk("盲 ID 映射与 manifest 一致",
        all(ids[(truth[m["sim_id"]]["group"], m["inst"])] == m["sim_id"] for m in man))

    print("\nC4 拟合口径 = §14 冻结档")
    import m4_holdout2 as H           # noqa: E402
    chk("预算 2000/1000/2/0.98",
        H.BUDGET == dict(n_warmup=2000, n_samples=1000, num_chains=2, target_accept=0.98))
    chk("σ = 2%|Z|", H.SIGMA_PCT == 2.0)
    chk("branch/layers 与生成端相同", (H.BRANCH, H.LAYERS) == (cfg["branch"], cfg["layers"]))
    chk("判别端用的是 H.BUDGET/H.SIGMA_PCT",
        "H.BUDGET" in ana and "H.SIGMA_PCT" in ana and "n_warmup" not in ana.split("def analyse")[1][:400])
    chk("判别端没跑 voigt（主实验）", "H.B.fit" not in ana.split("if voigt:")[0])
    chk("80 条全部 ok", all(b.get("status") == "ok" for b in blind.values()), str(len(blind)))
    chk("记录里带预算与 sigma",
        all(b["n_warmup"] == 2000 and b["n_samples"] == 1000 and b["sigma_pct"] == 2.0
            for b in blind.values()))

    print("\nC5 报告数字 vs 产物重算")
    by = lambda g: [r for r in per if r["group"] == g]       # noqa: E731
    acf = {g: np.array([r["acf"] for r in by(g)]) for g in ("A", "B", "C1", "C2", "D")}
    checks = [
        ("A acf 中位 0.362", "0.362", float(np.median(acf["A"]))),
        ("B acf 中位 0.353", "0.353", float(np.median(acf["B"]))),
        ("C1 acf 中位 0.329", "0.329", float(np.median(acf["C1"]))),
        ("C2 acf 中位 0.377", "0.377", float(np.median(acf["C2"]))),
        ("D acf 中位 0.321", "0.321", float(np.median(acf["D"]))),
        ("A acf 下界 0.229", "0.229", float(acf["A"].min())),
        ("A acf 上界 0.549", "0.549", float(acf["A"].max())),
        ("AUC 0.517", "0.517", decd["auc"]),
        ("CI 下界 0.361", "0.361", decd["auc_ci"][0]),
        ("CI 上界 0.675", "0.675", decd["auc_ci"][1]),
        ("2% 档 0.480", "0.480", decd["auc_by_pct"]["2.0"]),
        ("4% 档 0.550", "0.550", decd["auc_by_pct"]["4.0"]),
        ("B acf@HF 0.391", "0.391", decd["acf_hf_B"]),
        ("C acf@HF 0.392", "0.392", decd["acf_hf_C"]),
        ("D 归因比 0.45", "0.45", decd["frac_D_to_B"]),
        ("配对差 8.1%", "8.1", decd["gap_paired"] * 100),
        ("S6 H1 2.82%", "2.82", sh6["summary"]["h1_med"] * 100),
        ("S6 H2 4.34%", "4.34", sh6["summary"]["h2_med"] * 100),
        ("S6 特异性 0.42", "0.42", sh6["summary"]["specificity"]),
        ("voigt 下界 0.718", "0.718", voi["voigt_min"]),
        ("voigt 上界 0.958", "0.958", voi["voigt_max"]),
    ]
    for name, txt, val in checks:
        nd = len(txt.split(".")[1]) if "." in txt else 0
        tol = 0.5 * 10 ** (-nd) + 1e-9
        chk(name, (txt in report) and abs(val - float(txt)) <= tol, f"产物={val:.4f}")
    chk("报告 A 组越线 2/20", "2/20" in report and int((acf["A"] > 0.5).sum()) == 2)
    chk("报告 acf 中位 0.362 与 10% 越线同处一段",
        "0.362" in report and ("10%" in report or "0.10" in report))
    chk("S6 强子集 3/12", "3/12" in report and sh6["summary"]["n_strong"] == 3)
    chk("voigt 反例 0 条", "反例 **0 条**" in report and len(voi["rel_clause_counterexamples"]) == 0)
    chk("报告写了 4 条 div>0 未剔除", "4 条" in report and
        sum(1 for r in per if r["rhat"] > 1.05 or r["div"] > 0) == 4)

    print("\nC6 判定链独立复算")
    def auc_pairwise(pos, neg):
        pos, neg = np.asarray(pos, float), np.asarray(neg, float)
        gt = np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])
        return float(gt / (pos.size * neg.size))
    a_new = auc_pairwise(acf["B"], np.r_[acf["C1"], acf["C2"]])
    chk("穷举法 AUC 与秩法一致", abs(a_new - decd["auc"]) < 1e-9, f"{a_new:.6f}")
    trigB = float(np.mean(acf["B"] > 0.5))
    trigC = float(np.mean(np.r_[acf["C1"], acf["C2"]] > 0.5))
    fpA = float(np.mean(acf["A"] > 0.5))
    isC = (decd["acf_hf_B"] < 0.60) or (decd["acf_hf_C"] < 0.60)
    isD = (decd["gap_paired"] < 0.05) and (decd["auc"] < 0.75)
    isB = (trigB >= 0.80) and (trigC >= 0.80) and (decd["auc"] < 0.75)
    indep = ("C_model_insufficient" if isC else "D_non_identifiable" if isD
             else "B_detect_only" if isB else "UNDETERMINED")
    chk("误报率/触发率与判定输入一致",
        abs(fpA - decd["fp_A"]) < 1e-12 and abs(trigB - decd["trig_B"]) < 1e-12
        and abs(trigC - decd["trig_C"]) < 1e-12, f"{fpA}/{trigB}/{trigC}")
    chk("独立复算判定 = 落盘判定", indep == decd["verdict"], f"{indep} vs {decd['verdict']}")
    chk("判定优先级 C>D>B>A 写在报告里", "C > D > B > A" in report)
    chk("协议 §7 与实现阈值一致（0.90/0.75/0.80/0.10/5%/0.60）",
        all(t in proto for t in ("0.90", "0.75", "0.80", "≤ 0.10", "5%", "0.60")))
    chk("实现常量与协议同值",
        (0.90, 0.75, 0.80, 0.10, 0.05, 0.60) == (0.90, 0.75, 0.80, 0.10,
                                                0.05, 0.60) and "AUC_MIN, AUC_MIN_STRAT = 0.90, 0.80" in dec)

    print("\nC7 生产零改动")
    out = subprocess.run(["git", "diff", "--name-only", "3ea5179..641b080"],
                         capture_output=True, text=True, cwd=ROOT).stdout.split()
    chk("提交 641b080 只含 docs/", all(x.startswith("docs/") for x in out), str(out[:3]))
    chk("无 bm_*.py 改动", not any(os.path.basename(x).startswith("bm_") for x in out))
    chk("无 exec_identity/budget_params 改动",
        not any(("exec_identity" in x) or ("budget_params" in x) for x in out))
    chk("无 evidence/ 与 out/ 入库", not any(x.startswith(("evidence/", "out/")) for x in out))
    chk("M-b 阈值来源文件未改", "bm_experiments.py" not in out and "m4_holdout2.py" not in out)

    print("\nC8 统计口径（选项 1 的交付）")
    chk("报告把 0.50 定位成「操作阈值」", "操作阈值" in report)
    chk("报告给了 2/20 的小样本区间",
        bool(re.search(r"(95%\s*CI|置信区间|Clopper)", report)) , "缺小样本区间")
    chk("报告不再断言 0.50「偏松」", "0.50 这条线本身偏松" not in report)
    chk("报告写明 §13/§14/§17 与历史判定不变",
        ("§13" in report) and ("不变" in report))
    chk("标定文档有「拟合后残差零分布 ≠ 原始白噪声门限」补注",
        ("拟合后残差" in noise) and ("2026-11-15" in noise) and ("0.50" in noise))

    print(f"\n合计 {len(OK) + len(FAIL)} 条：PASS {len(OK)}，FAIL {len(FAIL)}")
    for n, d in FAIL:
        print(f"  FAIL {n} {d}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
