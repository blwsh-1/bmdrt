# -*- coding: utf-8 -*-
"""B3b 15 组驱动：3 谱 × 5 seed，逐组独立子进程，仅 seed 变化。

    python evidence/m6/b3b_run.py --out out/m6/b3b
    python evidence/m6/b3b_run.py --out out/m6/b3b --dry-run
    python evidence/m6/b3b_run.py --out out/m6/b3b --summarize-only

产物（不入库）：
    out/m6/b3b/<谱>/seed<N>/case_meta.json
    out/m6/b3b/<谱>/seed<N>/<stem>/init_fingerprints.json
    out/m6/b3b/<谱>/seed<N>/<stem>/real_<stem>.json / *_samples.npz
    out/m6/b3b/run_progress.jsonl
    out/m6/b3b/init_fingerprint_verify.json
    out/m6/b3b/b3b_gate.csv          收敛门禁（单独一表）
    out/m6/b3b/b3b_posterior.csv    后验读数（单独一表）
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import b3b_fingerprint as F                             # noqa: E402

SPECTRA = [
    ("Blank_main", "archive/Blank_0.1KBr_pH10_EIS_PtCu_1.2V.txt"),
    ("AZ31-4Li_100", "out/_m6_samples/z_csv/AZ31-4Li_100.csv"),
    ("AZ31-12Li_0", "out/_m6_samples/z_csv/AZ31-12Li_0.csv"),
]
SEEDS = (0, 1, 2, 3, 4)


def case_dir(out: Path, name: str, seed: int) -> Path:
    return out / name / f"seed{seed}"


def done(out: Path, name: str, seed: int) -> bool:
    p = case_dir(out, name, seed) / "case_meta.json"
    if not p.exists():
        return False
    try:
        return int(json.loads(p.read_text(encoding="utf-8")).get("returncode")) == 0
    except Exception:                                    # noqa: BLE001
        return False


def run_cases(out: Path, only=None, limit=None, force=False, dry=False) -> int:
    out.mkdir(parents=True, exist_ok=True)
    prog = out / "run_progress.jsonl"
    jobs = [(n, p, s) for n, p in SPECTRA for s in SEEDS]
    if only:
        want = {x.strip() for x in only.split(",") if x.strip()}
        jobs = [j for j in jobs if f"{j[0]}:{j[2]}" in want]
    if limit:
        jobs = jobs[:limit]

    print(f"[b3b_run] 计划 {len(jobs)} 组（全矩阵 {len(SPECTRA) * len(SEEDS)} 组）"
          f"  out={out}")
    n_ok = n_skip = n_bad = 0
    for i, (name, rel, seed) in enumerate(jobs, 1):
        cd = case_dir(out, name, seed)
        if not force and done(out, name, seed):
            print(f"[{i}/{len(jobs)}] 跳过 {name} seed={seed}（已成功）")
            n_skip += 1
            continue
        cmd = [sys.executable, "-u",
               str(Path(__file__).resolve().parent / "b3b_case.py"),
               "--spectrum", name, "--path", rel, "--seed", str(seed),
               "--out", str(cd)]
        print(f"[{i}/{len(jobs)}] {name} seed={seed} → {' '.join(cmd[1:])}",
              flush=True)
        if dry:
            continue
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), check=False)
        row = dict(spectrum=name, seed=seed, returncode=r.returncode,
                   elapsed_s=round(time.time() - t0, 2),
                   case_dir=str(cd.relative_to(ROOT)))
        with prog.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if r.returncode == 0:
            n_ok += 1
        else:
            n_bad += 1
            print(f"    [WARN] 该组返回 {r.returncode}，已记录，继续下一组",
                  flush=True)
    print(f"[b3b_run] 完成：成功 {n_ok} / 跳过 {n_skip} / 失败 {n_bad}")
    return 1 if n_bad else 0


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return ""


def summarize(out: Path) -> int:
    cases, gate_rows, post_rows = [], [], []
    for name, rel in SPECTRA:
        stem = Path(rel).stem
        for seed in SEEDS:
            cd = case_dir(out, name, seed)
            meta_p = cd / "case_meta.json"
            res_p = cd / stem / f"real_{stem}.json"
            fp_p = cd / stem / "init_fingerprints.json"
            if not meta_p.exists():
                print(f"[summarize] 缺 case_meta：{cd}")
                continue
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if fp_p.exists():
                d = json.loads(fp_p.read_text(encoding="utf-8"))
                cases.append(dict(spectrum=name, seed=seed,
                                  by_K=d.get("by_K", [])))
            if not res_p.exists():
                print(f"[summarize] 缺结果 JSON：{res_p}")
                continue
            j = json.loads(res_p.read_text(encoding="utf-8"))
            diag = j.get("diag") or {}
            rhat = max([v.get("rhat") for v in diag.values()
                        if isinstance(v, dict) and v.get("rhat") is not None]
                       or [float("nan")])
            bulk = min([v.get("bulk") for v in diag.values()
                        if isinstance(v, dict) and v.get("bulk") is not None]
                       or [float("nan")])
            tail = min([v.get("tail") for v in diag.values()
                        if isinstance(v, dict) and v.get("tail") is not None]
                       or [float("nan")])
            comps = j.get("components") or []
            main = max(comps, key=lambda c: c.get("w", 0)) if comps else {}
            base = dict(spectrum=name, seed=seed, file=j.get("file"),
                        rc=meta.get("returncode"),
                        elapsed_s=meta.get("elapsed_s"))
            gate_rows.append(dict(
                **base, K=j.get("K"), sample_grade=j.get("sample_grade"),
                rhat_max=rhat, bulk_min=bulk, tail_min=tail,
                divergences=j.get("divergences"), tree_cap=j.get("tree_cap"),
                k_selection_status=j.get("k_selection_status"),
                fit_performed=j.get("fit_performed"),
                grade_fail=";".join(j.get("grade_fail") or []),
                warnings_n=len(j.get("warnings") or [])))
            post_rows.append(dict(
                **base, K=j.get("K"), elpd_loo=j.get("elpd_loo"),
                R_inf=j.get("R_inf"), W=j.get("W"),
                n_comp=len(comps),
                main_f0=main.get("f0"),
                main_f0_width=(_num(main.get("f0_hi")) - _num(main.get("f0_lo"))
                               if main else ""),
                main_w=main.get("w"),
                main_w_width=(_num(main.get("w_hi")) - _num(main.get("w_lo"))
                              if main else ""),
                main_phi=main.get("phi"),
                rms_re=j.get("rms_re"), rms_im=j.get("rms_im"),
                sigma_re=j.get("sigma_re"), sigma_im=j.get("sigma_im")))

    def _write_csv(path: Path, rows: list[dict]):
        if not rows:
            print(f"[summarize] 无行可写：{path.name}")
            return
        keys = list(rows[0].keys())
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[summarize] 写入 {path}（{len(rows)} 行）")

    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "b3b_gate.csv", gate_rows)
    _write_csv(out / "b3b_posterior.csv", post_rows)

    ver = F.verify(cases)
    (out / "init_fingerprint_verify.json").write_text(
        json.dumps(ver, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(f"[summarize] 初值指纹核验：level={ver['level']} "
          f"检查数={ver['n_checks']}")
    print(f"            {ver['conclusion']}")
    (out / "b3b_summary.json").write_text(json.dumps(dict(
        spectra=[n for n, _ in SPECTRA], seeds=list(SEEDS),
        n_cases_expected=len(SPECTRA) * len(SEEDS),
        n_gate_rows=len(gate_rows), n_posterior_rows=len(post_rows),
        init_fingerprint=ver), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/m6/b3b")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--only", default=None,
                    help="只跑指定组，逗号分隔 name:seed，如 Blank_main:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="已成功的组也重跑")
    a = ap.parse_args()
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    if a.summarize_only:
        return summarize(out)
    rc = run_cases(out, only=a.only, limit=a.limit, force=a.force,
                   dry=a.dry_run)
    if not a.dry_run:
        summarize(out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
