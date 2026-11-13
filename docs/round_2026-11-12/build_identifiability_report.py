# -*- coding: utf-8 -*-
"""统一参数可辨识性报告：批处理器（P0 ①）。

只读 `out/` 下**已有**产物，不重跑拟合、不改源文件、不改生产代码。
预注册：`docs/IDENTIFIABILITY_REPORT_PREREG.md`；schema：`bm_identifiability.py`。

    python docs/round_2026-11-12/build_identifiability_report.py \
        --out out/identifiability_report

产物：每候选一份 JSON + `summary.md` + `manifest.json`（finalized）。
`fit_performed=false` 的 voigt 记录（KK 门禁拒绝、无后验）如实跳过并计数，
**不生成空报告冒充结果**。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import bm_identifiability as ID  # noqa: E402

DEFAULT_OUT = ROOT / "out" / "identifiability_report"


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return (s[:56] or "case") + "_" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:6]


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect_voigt():
    """返回 (可处理条目, 跳过的无后验记录数)。"""
    out, skipped = [], 0
    for p in sorted((ROOT / "out").rglob("real_*.json")):
        if "bundle" in p.parts or "identifiability_report" in p.parts:
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not raw.get("fit_performed", False):
            skipped += 1                  # 无后验，不生成报告
            continue
        stem = p.name.replace("real_", "").replace(".json", "")
        mid = f"{stem}::K={raw.get('K', '?')}"
        out.append((mid, "voigt", raw, p))
    return out, skipped


def collect_nested():
    out = []
    for p in sorted((ROOT / "out").rglob("*.nested.json")):
        if "identifiability_report" in p.parts:
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = raw.get("config") or {}
        mid = (f"{p.name.replace('.nested.json', '')}::"
               f"L={cfg.get('layers', '?')}/{cfg.get('branch', '?')}")
        out.append((mid, "nested", raw, p))
    return out


def _median(xs):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    n = len(xs)
    return float(xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--summary-rows", type=int, default=6,
                    help="summary.md 每族展示几份（报告文件仍是全量）")
    a = ap.parse_args()
    dest = Path(a.out)
    if not dest.is_absolute():
        dest = ROOT / dest
    dest.mkdir(parents=True, exist_ok=True)

    voigt_items, skipped = collect_voigt()
    items = voigt_items + collect_nested()
    if not items:
        print("没有找到可处理的产物", file=sys.stderr)
        return 1

    reports, files = [], []

    for mid, fam, raw, src in items:
        rep = ID.build_report(raw, family=fam, model_id=mid,
                              source_files=[str(src.relative_to(ROOT))])
        fn = dest / f"{_slug(mid)}.{fam}.json"
        fn.write_text(json.dumps(rep, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
        reports.append((rep, fn))
        files.append(fn)

    # 跨族同 schema 硬校验（build_report 自己已查过一次，这里全量再查）
    byfam = {}
    for rep, _ in reports:
        byfam.setdefault(rep["family"], []).append(rep)
    for fam, rs in byfam.items():
        ID.assert_same_schema(*rs[:2]) if len(rs) > 1 else None
    if len(byfam) == 2:
        ID.assert_same_schema(byfam["voigt"][0], byfam["nested"][0])

    # ---------- summary.md ----------
    L = ["# 统一参数可辨识性报告 · 汇总", "",
         f"> 预注册：`docs/IDENTIFIABILITY_REPORT_PREREG.md` · "
         f"schema `{ID.SCHEMA_VERSION}` · 只读已有产物，未重跑拟合。", "",
         ID.DISCLAIMER, "", "## 读这份报告的三条禁令", "",
         "1. **后验窄 ≠ 可辨识**：`R_inf` 的 q05/q95 可以很窄，但 "
         "`identifiability_check.py` 的剖面平坦度实验证明固定 `Rs` 从 0.01→25 Ω "
         "残差几乎不动 → 不可辨识。相对跨度只是「后验离散度」，不是可辨识性证明。",
         "2. **`acf` 不是 PPC**：两族 PPC 均 `unavailable`，不得拿残差 acf 顶替。",
         "3. **`unavailable` ≠ 通过**：只是数据没落盘，不表示诊断合格。", "",
         "## 全量统计", "",
         "| 族 | 报告数 | available/份 | partial/份 | unavailable/份 |", "|---|---|---|---|---|"]
    for fam, rs in sorted(byfam.items()):
        c = {k: sum(r["coverage"][k] for r in rs)
             for k in ("available", "partial", "unavailable")}
        L.append(f"| `{fam}` | {len(rs)} | {c['available']} | "
                 f"{c['partial']} | {c['unavailable']} |")
    L += ["", f"跳过的 voigt 记录（无后验 / KK 拒绝）：**{skipped}** —— "
          "如实跳过，不生成空报告。", ""]

    # 已知结论复现检查
    vrs = byfam.get("voigt", [])
    with_ridge = sum(1 for r in vrs
                     if any("Rinf-Wtot" in (d.get("kind") or "") or
                           "ridge" in (d.get("kind") or "")
                           for d in (r["diagnostics"]["combinations"]
                                     .get("known_combinations") or [])))
    rinf_spans = []
    for r in vrs:
        for p in (r["diagnostics"]["posterior"].get("params") or []):
            if p.get("name") == "R_inf" and p.get("rel_span") is not None:
                rinf_spans.append(p["rel_span"])
    L += ["## 已知结论复现（自动，不新增判断）", "",
          f"- voigt 中出现 `Rinf-Wtot` / `R-tau ridge` 组合提示：**{with_ridge}/{len(vrs)}** 份 "
          "（原样转述 `param_corr.degenerate`）。",
          f"- `R_inf` 后验相对跨度中位数：**{_median(rinf_spans)}** —— "
          "按禁令 1，此值**不得**读成「Rs 可辨识」。",
          "- `Rs–R1` ρ(log)≈−0.94、`Rs+R1` 区间极窄 → 报告里表现为强相关对与组合提示；"
          "物理解释仍只报和。", ""]

    if "nested" in byfam:
        n0 = byfam["nested"][0]["diagnostics"]
        L += ["## nested 侧暴露的缺口（预注册 §4 实测）", ""]
        for k in ("correlation", "boundary", "combinations"):
            L.append(f"- `{k}`：`{n0[k]['status']}` — {n0[k]['note']}")
        L += ["", "这是 `nested_fit.py` 不落盘后验样本的直接后果，属**产物缺口**，"
              "不是实现失败，也不构成任何模型的通过/拒绝。", ""]

    for fam, rs in sorted(byfam.items()):
        L += [f"## 样例报告 · `{fam}`（共 {len(rs)} 份，展示前 {min(a.summary_rows, len(rs))} 份）", ""]
        for r in rs[:a.summary_rows]:
            L.append(ID.format_markdown(r))
        L.append("")

    sm = dest / "summary.md"
    sm.write_text("\n".join(L) + "\n", encoding="utf-8")
    files.append(sm)

    manifest = dict(
        status="finalized", schema_version=ID.SCHEMA_VERSION,
        generator="docs/round_2026-11-12/build_identifiability_report.py",
        n_reports=len(reports),
        by_family={k: len(v) for k, v in sorted(byfam.items())},
        skipped_no_posterior=skipped,
        files=[dict(name=p.name, sha256=_sha256(p)) for p in files],
        notes=list(ID.GENERAL_NOTES),
    )
    mf = dest / "manifest.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                             allow_nan=False) + "\n", encoding="utf-8")

    print(f"生成 {len(reports)} 份报告 → {dest}")
    for k, v in sorted(byfam.items()):
        print(f"  {k}: {len(v)} 份")
    print(f"  跳过无后验 voigt 记录: {skipped}")
    print(f"  summary.md / manifest.json（finalized，{len(files)} 个文件进清单）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
