# -*- coding: utf-8 -*-
"""M6 E2 驱动：真实谱基线 B1 / B2 + B3a 设置敏感性（主臂）。

按 M6_VALIDATION_PLAN.md v0.2 §4.1 / §6 / §6.2 / §7 冻结口径执行。

    python evidence/m6/run_e2.py --out out/m6/e2 --dry-run
    python evidence/m6/run_e2.py --out out/m6/e2
    python evidence/m6/run_e2.py --out out/m6/e2 --summarize-only

矩阵（不改生产代码、只调现有 CLI）：
    B1  8 条  Blank 6 + AZ31-4Li_100/150   默认档 3000/1000/3
    B2 16 条  其余 AZ31                    --quick（exploratory，仅筛选）
    B3a 3 条 × 2 设置  Blank 主谱 / AZ31-4Li_100 / AZ31-12Li_0
                        ta=0.9（B1 档）与 --ta 0.99

按「不擅自扩实验矩阵」：本脚本**只跑 §6 冻结命令里的主臂 `--wide`**。
§3.1 / G7 的副臂（默认 `krange=null`，DRT 决定 K 范围）**未执行**，
在 e2_summary.json 里登记为 `secondary_arm=not_run`、`G7=pending`。
每条都加 `--no-move`（config.batch.move_done=True 的工程保护，不改变任何拟合数值）。
E2 独立重跑，不复用已封存的 B3b 结果。

产物（不入库）：
    out/m6/e2/b1/<stem>/real_<stem>.json ...
    out/m6/e2/b2/<stem>/...
    out/m6/e2/b3a/<setting>/<stem>/...
    out/m6/e2/run_progress.jsonl
    out/m6/e2/summary_m6.csv       §7 固定字段，B1+B2 共 24 行
    out/m6/e2/b3a_compare.csv      G6a 四个量，逐谱 × 2 设置
    out/m6/e2/e2_summary.json      矩阵元数据 + 未执行项登记
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BLANK = [
    "Blank_0.1KBr_pH10_EIS_PtCu_1.2V",
    "Blank_0.1KBr_pH10_EIS_PtCu_1.2V_1",
    "Blank_0.1KBr_pH10_EIS_PtCu_1.2V_2",
    "Blank_0.1KBr_pH10_EIS_PtCu_1.2V_3",
    "Blank_0.1KBr_pH10_EIS_PtCu_1.2V_4",
    "Blank_0.1KBr_pH10_EIS_PtCu_1.2V_5",
]
B1_AZ31 = ["AZ31-4Li_100", "AZ31-4Li_150"]
B2_AZ31 = [
    f"AZ31-{li}Li_{t}"
    for li in ("4", "8", "12")
    for t in ("0", "5", "10", "50", "100", "150")
    if f"AZ31-{li}Li_{t}" not in B1_AZ31
]
B3A = [BLANK[0], "AZ31-4Li_100", "AZ31-12Li_0"]
SETTINGS = [("ta090", []), ("ta099", ["--ta", "0.99"])]

INVENTORY = ROOT / "evidence/m6/data_inventory.csv"


def blank_path(stem: str) -> str:
    return f"archive/{stem}.txt"


def az31_path(stem: str) -> str:
    return f"out/_m6_samples/z_csv/{stem}.csv"


def build_jobs() -> list[dict]:
    jobs = []
    for stem in BLANK:
        jobs.append(dict(job_id=f"b1:{stem}", batch="B1", stem=stem,
                         path=blank_path(stem), out_rel="b1",
                         outdir=None, extra=[], quick=False, setting="ta090"))
    for stem in B1_AZ31:
        jobs.append(dict(job_id=f"b1:{stem}", batch="B1", stem=stem,
                         path=az31_path(stem), out_rel="b1",
                         outdir=None, extra=[], quick=False, setting="ta090"))
    for stem in B2_AZ31:
        jobs.append(dict(job_id=f"b2:{stem}", batch="B2", stem=stem,
                         path=az31_path(stem), out_rel="b2",
                         outdir=None, extra=["--quick"], quick=True,
                         setting="ta090"))
    for stem in B3A:
        path = blank_path(stem) if stem.startswith("Blank") else az31_path(stem)
        for tag, extra in SETTINGS:
            jobs.append(dict(job_id=f"b3a:{tag}:{stem}", batch="B3a",
                             stem=stem, path=path,
                             out_rel=f"b3a/{tag}", outdir=None, extra=extra,
                             quick=False, setting=tag))
    return jobs


def load_inventory() -> dict[str, dict]:
    with INVENTORY.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {r["stem"]: r for r in rows}


def job_outdir(out: Path, j: dict) -> Path:
    return out / j["out_rel"]


def result_path(out: Path, j: dict) -> Path:
    return job_outdir(out, j) / j["stem"] / f"real_{j['stem']}.json"


def progress_rows(out: Path) -> list[dict]:
    p = out / "run_progress.jsonl"
    if not p.exists():
        return []
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:                                    # noqa: BLE001
            pass
    return rows


def done(out: Path, j: dict, prog: list[dict]) -> bool:
    if not any(r.get("job_id") == j["job_id"] and r.get("returncode") == 0
               for r in prog):
        return False
    return result_path(out, j).exists()


def cmd_for(out: Path, j: dict) -> list[str]:
    od = job_outdir(out, j)
    cmd = [sys.executable, "-u", str(ROOT / "bmdrt.py"), "auto", j["path"],
           "--wide", "--kk", "gate", "--no-move", "--out", str(od)]
    cmd += list(j["extra"])
    return cmd


def precheck(jobs: list[dict]) -> int:
    bad = 0
    for j in jobs:
        p = ROOT / j["path"]
        if not p.exists():
            print(f"[precheck] 缺输入文件：{j['path']}")
            bad += 1
    inv = load_inventory()
    for j in jobs:
        if j["stem"] not in inv:
            print(f"[precheck] E1 清单缺 stem：{j['stem']}")
            bad += 1
    print(f"[precheck] {len(jobs)} 个 job，输入文件与 E1 清单核对，问题 {bad} 处")
    return bad


def run_jobs(out: Path, jobs: list[dict], only=None, limit=None,
             force=False, dry=False) -> int:
    out.mkdir(parents=True, exist_ok=True)
    if precheck(jobs):
        return 2
    jobs = list(jobs)
    if only:
        want = {x.strip() for x in only.split(",") if x.strip()}
        jobs = [j for j in jobs if j["job_id"] in want
                or j["batch"] in want or j["stem"] in want]
    if limit:
        jobs = jobs[:limit]
    prog = progress_rows(out)
    print(f"[run_e2] 计划 {len(jobs)} 个 job（全矩阵 {len(build_jobs())} 个）"
          f"  out={out}", flush=True)
    n_ok = n_skip = n_bad = 0
    for i, j in enumerate(jobs, 1):
        od = job_outdir(out, j)
        od.mkdir(parents=True, exist_ok=True)
        if not force and done(out, j, prog):
            print(f"[{i}/{len(jobs)}] 跳过 {j['job_id']}（已成功）", flush=True)
            n_skip += 1
            continue
        cmd = cmd_for(out, j)
        print(f"[{i}/{len(jobs)}] {j['job_id']} → {' '.join(cmd[1:])}",
              flush=True)
        if dry:
            continue
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), check=False)
        row = dict(job_id=j["job_id"], batch=j["batch"], stem=j["stem"],
                   setting=j["setting"], returncode=r.returncode,
                   elapsed_s=round(time.time() - t0, 2),
                   result_dir=str((od / j["stem"]).relative_to(ROOT)))
        with (out / "run_progress.jsonl").open("a", encoding="utf-8",
                                               newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        prog.append(row)
        if r.returncode == 0:
            n_ok += 1
        else:
            n_bad += 1
            print(f"    [WARN] {j['job_id']} 返回 {r.returncode}，已记录，继续",
                  flush=True)
    print(f"[run_e2] 完成：成功 {n_ok} / 跳过 {n_skip} / 失败 {n_bad}",
          flush=True)
    return 1 if n_bad else 0


def _s(x) -> str:
    return "" if x is None else str(x)


def _diag_extreme(j: dict, key: str, fn) -> object:
    vals = [v.get(key) for v in (j.get("diag") or {}).values()
            if isinstance(v, dict) and v.get(key) is not None]
    return fn(vals) if vals else ""


def _comp_fields(j: dict, K: int) -> dict:
    out = {}
    wc = ((j.get("component_review") or {}).get("window_contribution") or [])
    by_n = {int(c.get("component", -1)): c for c in wc
            if isinstance(c, dict)}
    for n in range(1, 6):
        c = by_n.get(n, {})
        out[f"component_{n}_state"] = _s(c.get("state"))
        out[f"component_{n}_frac_abs_at_fmin"] = _s(c.get("frac_abs_at_fmin"))
    return out


def summary_row(out: Path, j: dict, inv: dict) -> dict:
    rp = result_path(out, j)
    i = inv.get(j["stem"], {})
    base = dict(
        stem=j["stem"], batch=j["batch"], setting=j["setting"],
        sha256=i.get("sha256", ""), n_pts=i.get("n_pts", ""),
        f_min=i.get("f_min_hz", ""), f_max=i.get("f_max_hz", ""),
        kk_status="", kk_reason="", drt_npeaks="", k_primary="",
        k_selection_status="", grade="", elpd_loo="", dse="",
        pareto_k_bad_frac="", component_review_status="not_available",
        ad_status="not_applicable", ad_choice="not_applicable",
        evaluable="not_applicable", truth="unknown",
        sample_id=i.get("sample_id", ""),
        sample_id_confidence=i.get("sample_id_confidence", ""),
        group_id=i.get("group_id", ""),
        group_id_confidence=i.get("group_id_confidence", ""),
        experiment_id=i.get("exp_id", ""),
        independence_status=i.get("independence_status", ""),
        evidence_type="real_data",
        seed="", n_warmup="", n_samples="", num_chains="",
        target_accept="", quick=str(j["quick"]).lower(),
        wide="true", k_range="1..5", kk_mode="gate", no_move="true",
        elapsed_s="", returncode="", result_json=str(rp.relative_to(ROOT)))
    base.update(_comp_fields({}, 5))
    if not rp.exists():
        return base
    d = json.loads(rp.read_text(encoding="utf-8"))
    kk = d.get("kk") or {}
    cr = d.get("component_review") or {}
    if d.get("fit_performed") is False:
        cr_status = "not_applicable"
    elif cr:
        cr_status = "present"
    else:
        cr_status = "not_available"
    smp = d.get("sampling") or {}
    base.update(
        kk_status=_s(kk.get("status")), kk_reason=_s(kk.get("reason")),
        k_primary=_s(d.get("K")),
        k_selection_status=_s(d.get("k_selection_status")),
        grade=_s(d.get("sample_grade")),
        elpd_loo=_s(d.get("elpd_loo")), dse=_s(d.get("dse")),
        pareto_k_bad_frac=_s(d.get("pareto_k_bad_frac")),
        component_review_status=cr_status,
        seed=_s(d.get("seed")),
        n_warmup=_s(smp.get("n_warmup")), n_samples=_s(smp.get("n_samples")),
        num_chains=_s(smp.get("num_chains")),
        target_accept=_s(smp.get("target_accept")),
        rhat_max=_diag_extreme(d, "rhat", max),
        bulk_min=_diag_extreme(d, "bulk", min),
        tail_min=_diag_extreme(d, "tail", min),
        divergences=_s(d.get("divergences")),
        grade_fail=";".join(str(x) for x in (d.get("grade_fail") or [])),
        rms_re=_s(d.get("rms_re")), rms_im=_s(d.get("rms_im")),
        acf=_s(d.get("acf")), n_components=len(d.get("components") or []),
        warnings_n=len(d.get("warnings") or []))
    base.update(_comp_fields(d, int(d.get("K") or 0)))
    pr = progress_rows(out)
    for r in pr:
        if r.get("job_id") == j["job_id"]:
            base["elapsed_s"] = r.get("elapsed_s", "")
            base["returncode"] = r.get("returncode", "")
    return base


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"[summarize] 无行可写：{path.name}")
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[summarize] 写入 {path}（{len(rows)} 行）", flush=True)


def b3a_compare(out: Path, jobs: list[dict]) -> list[dict]:
    """G6a 四个量：K 翻转 / 组分状态集合变化 / 采样分级变化 / 后验位置与宽度。"""
    by_stem: dict[str, dict] = {}
    for j in jobs:
        if j["batch"] != "B3a":
            continue
        rp = result_path(out, j)
        rec = dict(setting=j["setting"], K="", k_selection_status="",
                   sample_grade="", comp_states="", R_inf="", W="",
                   R_inf_lo="", R_inf_hi="", W_lo="", W_hi="",
                   main_w="", main_w_width="", main_f0="", main_f0_width="",
                   rhat_max="", divergences="", available=False)
        if rp.exists():
            d = json.loads(rp.read_text(encoding="utf-8"))
            wc = ((d.get("component_review") or {})
                  .get("window_contribution") or [])
            comps = d.get("components") or []
            main = max(comps, key=lambda c: c.get("w", 0)) if comps else {}
            def _width(lo, hi):
                try:
                    return float(hi) - float(lo)
                except (TypeError, ValueError):
                    return ""
            rec.update(
                K=_s(d.get("K")),
                k_selection_status=_s(d.get("k_selection_status")),
                sample_grade=_s(d.get("sample_grade")),
                comp_states=";".join(sorted(
                    _s(c.get("state")) for c in wc if isinstance(c, dict))),
                R_inf=_s(d.get("R_inf")), W=_s(d.get("W")),
                main_w=_s(main.get("w")),
                main_w_width=_width(main.get("w_lo"), main.get("w_hi")),
                main_f0=_s(main.get("f0")),
                main_f0_width=_width(main.get("f0_lo"), main.get("f0_hi")),
                rhat_max=_diag_extreme(d, "rhat", max),
                divergences=_s(d.get("divergences")),
                available=True)
        by_stem.setdefault(j["stem"], {})[j["setting"]] = rec
    rows = []
    for stem, pair in sorted(by_stem.items()):
        a, b = pair.get("ta090", {}), pair.get("ta099", {})
        four = [a.get("K") != b.get("K"),
                a.get("comp_states") != b.get("comp_states"),
                a.get("sample_grade") != b.get("sample_grade"),
                any(a.get(k) != b.get(k) for k in
                    ("R_inf", "W", "main_w", "main_w_width",
                     "main_f0", "main_f0_width"))]
        rows.append(dict(
            stem=stem,
            ta090_K=a.get("K", ""), ta099_K=b.get("K", ""),
            K_flipped=str(bool(a.get("available") and b.get("available")
                               and four[0])).lower(),
            ta090_comp_states=a.get("comp_states", ""),
            ta099_comp_states=b.get("comp_states", ""),
            comp_states_changed=str(bool(a.get("available") and b.get("available")
                                         and four[1])).lower(),
            ta090_grade=a.get("sample_grade", ""),
            ta099_grade=b.get("sample_grade", ""),
            grade_changed=str(bool(a.get("available") and b.get("available")
                                   and four[2])).lower(),
            ta090_R_inf=a.get("R_inf", ""), ta099_R_inf=b.get("R_inf", ""),
            ta090_W=a.get("W", ""), ta099_W=b.get("W", ""),
            ta090_main_w_width=a.get("main_w_width", ""),
            ta099_main_w_width=b.get("main_w_width", ""),
            ta090_main_f0_width=a.get("main_f0_width", ""),
            ta099_main_f0_width=b.get("main_f0_width", ""),
            posterior_changed=str(bool(a.get("available") and b.get("available")
                                       and four[3])).lower(),
            setting_sensitive=(str(any(four)).lower()
                               if a.get("available") and b.get("available")
                               else "not_available"),
            ta090_available=str(a.get("available")).lower(),
            ta099_available=str(b.get("available")).lower()))
    return rows


def summarize(out: Path) -> int:
    jobs = build_jobs()
    inv = load_inventory()
    out.mkdir(parents=True, exist_ok=True)
    base_jobs = [j for j in jobs if j["batch"] in ("B1", "B2")]
    rows = [summary_row(out, j, inv) for j in base_jobs]
    _write_csv(out / "summary_m6.csv", rows)
    comp = b3a_compare(out, jobs)
    _write_csv(out / "b3a_compare.csv", comp)
    pr = progress_rows(out)
    meta = dict(
        plan="M6_VALIDATION_PLAN.md v0.2 §4.1 / §6 / §6.2 / §7",
        primary_arm="--wide --kk gate (K=1..5)",
        secondary_arm="not_run",
        G7="pending: 副臂未执行，主臂/副臂 K 差异条数无法给出",
        no_move="true (工程保护；不改变任何拟合数值)",
        b3b_reuse="none (B3b 已封存为独立证据，E2 独立重跑)",
        matrix=dict(B1=len(BLANK) + len(B1_AZ31), B2=len(B2_AZ31),
                    B3a_spectra=len(B3A), B3a_settings=len(SETTINGS),
                    total_jobs=len(jobs)),
        n_progress_rows=len(pr),
        n_failed=sum(1 for r in pr if r.get("returncode") != 0),
        n_summary_rows=len(rows), n_b3a_rows=len(comp),
        notes=[
            "B2 为 --quick / exploratory，仅探查与筛选，不进正式验收（§4.1）",
            "G3 / G3b / G6a 不设通过线，只分列记录（§4.1）",
            "G4 按拟合状态条件适用：KK 拒绝记 not_applicable（§4.1）",
            "drt_npeaks / pareto_k_bad_frac 在当前结果 JSON 无有效值，留空",
            "ad_status / ad_choice / evaluable 属 E3/E4 吸附线，E2 记 not_applicable",
            "experiment_id 直接取自 E1 清单：对 AZ31 谱它是文件名里的浸泡时间标记，"
            "不是 2E 的 exp1/2/3，仅作文件名所见记录",
            "B3a 的 ta=0.9 与 B1 中同名谱在同配置下重复执行，可作内部一致性对照",
        ],
        jobs=[dict(job_id=j["job_id"], batch=j["batch"], stem=j["stem"],
                   setting=j["setting"], path=j["path"],
                   out_rel=j["out_rel"], extra=j["extra"], quick=j["quick"])
              for j in jobs])
    (out / "e2_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(f"[summarize] 写入 e2_summary.json；副臂未跑，G7=pending", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/m6/e2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--only", default=None,
                    help="只跑指定 job / batch / stem，逗号分隔")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    if a.summarize_only:
        return summarize(out)
    rc = run_jobs(out, build_jobs(), only=a.only, limit=a.limit,
                  force=a.force, dry=a.dry_run)
    if not a.dry_run:
        summarize(out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
