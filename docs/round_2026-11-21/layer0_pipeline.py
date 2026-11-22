"""层 0 两阶段固化入口：一条命令重排 21 条（编排器，不改判据、不改任何数字）。

两阶段（顺序不可颠倒）
----------------------
  阶段 1  std_recheck.py      quick 采样不过者 → standard 预算复核（带 MCMC，断点续跑）
  阶段 2  layer0_summary.py   合并 results.json + std_recheck.json → LAYER0_SUMMARY.md

为什么需要固化
--------------
`LAYER0_GATE_AUDIT.md` §3：缺口不在判据在自动化——`batch_fit_21.py` 跑完不会提示
"你的 best 有 67% 采样不过"，必须靠人记得分两阶段跑。本入口把两步串成一条命令，
并加三道守卫：

  G1 前置：results.json / std_recheck.json / progress.jsonl 存在、可解析、条数自洽（21）。
  G2 部分跑保护：阶段 1 若带 --limit/--new-only（覆盖 < 21 条），**拒绝阶段 2**，
     并在阶段 1 结束后用 progress.jsonl **重建 std_recheck.json**（抵消 std_recheck
     在部分模式下把 rows 从 21 条重写成 N 条的副作用）。
     两层都要防：layer0_summary 会整份覆盖 LAYER0_SUMMARY.md（15/6 → 大部分
     missing_standard）；而 std_recheck 自身会把 std_recheck.json 也截成 N 条，
     下一轮全量的 G1 就会因条数不自洽而红。
  G3 后置：汇总表生成后重读，逐条数必须 = results 条数，non_evaluable 数必须 =
     std_recheck 的 structurally_unidentifiable 数，否则报错退出。

用法
----
    python docs/round_2026-11-21/layer0_pipeline.py              # 全量两阶段（幂等）
    python docs/round_2026-11-21/layer0_pipeline.py --dry        # 只打印计划，不写
    python docs/round_2026-11-21/layer0_pipeline.py --check      # 只读核对现有产物
    python docs/round_2026-11-21/layer0_pipeline.py --limit 2    # 部分复核（不写汇总表）

边界
----
- 不改 std_recheck.py / layer0_summary.py 一个字节；它们是被文档引用的证据脚本。
- 不改判据：standard R̂<=1.05、minor_div 算通过、structurally_unidentifiable 不可用
  等判定全在两个原脚本里，本入口只负责编排与校验。
- 断点续跑沿用 progress.jsonl：已完成的自动跳过 ⇒ 全量运行通常是 no-op（幂等）。
  想强制重跑某条须先删其 progress 行，见 STD_RECHECK.md。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import redirect_stdout
from io import StringIO

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import layer0_summary as SUM          # noqa: E402
import std_recheck as REC             # noqa: E402

RES = os.path.join(REPO, "out", "batch21", "results.json")
STDOUT = os.path.join(REPO, REC.OUT, "std_recheck.json")
PROG = os.path.join(REPO, REC.OUT, "progress.jsonl")
SUMMARY = os.path.join(REPO, SUM.OUTMD)


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def _load_results():
    d = json.load(open(RES, encoding="utf-8"))
    return d["rows"] if isinstance(d, dict) else d


def _precheck() -> int:
    """G1：三份输入产物存在、可解析、条数自洽。"""
    for p in (RES, STDOUT, PROG, SUMMARY):
        if not os.path.isfile(p):
            print("[FAIL] G1 缺文件:", os.path.relpath(p, REPO))
            return 1
    try:
        rows = _load_results()
        std = json.load(open(STDOUT, encoding="utf-8"))
        prog = [json.loads(l) for l in open(PROG, encoding="utf-8") if l.strip()]
    except Exception as ex:                    # noqa: BLE001
        print("[FAIL] G1 解析失败:", type(ex).__name__, str(ex)[:120])
        return 1
    n_exp = sum(1 for r in rows if (r.get("best") or {}).get("rhat") is not None)
    n_std = len(std.get("rows", []))
    if not (n_exp == n_std == len(prog)):
        print("[FAIL] G1 条数不自洽: results %d / std_recheck %d / progress %d"
              % (n_exp, n_std, len(prog)))
        return 1
    print("[ok] G1 前置：results/std/progress 均为 %d 条，budget=%s"
          % (n_exp, std.get("budget")))
    return 0


def _postcheck(rows) -> int:
    """G3：汇总表逐条数、判定分布与 std_recheck 一致。"""
    std = json.load(open(STDOUT, encoding="utf-8"))["rows"]
    n_nv = sum(1 for r in std if r["verdict"] == "structurally_unidentifiable")
    allowed = {"budget_limited", "budget_limited_minor_div",
               "structurally_unidentifiable"}
    unknown = sorted({r["verdict"] for r in std
                      if r["verdict"] not in allowed
                      and not r["verdict"].startswith("failed")})
    if unknown:
        print("[FAIL] G3 出现未知 verdict:", unknown)
        return 1
    text = open(SUMMARY, encoding="utf-8").read()
    # 只看「## 1. 逐条」到「## 2.」之间的表，否则按类表的 3 行会被重复计入。
    # 逐条表的层 0 列只有两种取值（见 layer0_summary.py）：pass_standard / non_evaluable。
    seg = text.split("## 1. 逐条")[-1].split("## 2.")[0]
    detail = [l for l in seg.splitlines()
              if l.startswith("| ") and ("pass_standard" in l or "non_evaluable" in l)]
    n_rows = len(detail)
    n_sum_nv = sum(1 for l in detail if "non_evaluable" in l)
    total_bad = 0
    if n_rows != len(rows):
        print("[FAIL] G3 汇总表逐条数 %d != results %d" % (n_rows, len(rows)))
        total_bad += 1
    if n_sum_nv != n_nv:
        print("[FAIL] G3 汇总表 non_evaluable %d != std_recheck %d" % (n_sum_nv, n_nv))
        total_bad += 1
    if total_bad:
        return 1
    print("[ok] G3 后置：汇总表 %d 条，non_evaluable %d 条，verdict 取值合法"
          % (n_rows, n_nv))
    return 0


def cmd_check(argv=None) -> int:
    """只读核对现有产物，不跑 MCMC、不写任何文件。"""
    rows = _load_results()
    rc = _precheck()
    if rc:
        return rc
    text = open(SUMMARY, encoding="utf-8").read()
    need = "**合计**"
    if need not in text:
        print("[FAIL] check 汇总表缺合计行")
        return 1
    std = json.load(open(STDOUT, encoding="utf-8"))["rows"]
    n_nv = sum(1 for r in std if r["verdict"] == "structurally_unidentifiable")
    print("[ok] check 汇总表合计行存在；std non_evaluable %d 条" % n_nv)
    return _postcheck(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry", action="store_true", help="只打印计划，不写")
    ap.add_argument("--check", action="store_true", help="只读核对现有产物")
    ap.add_argument("--limit", type=int, default=0, help="阶段 1 最多新跑几条（触发部分模式）")
    ap.add_argument("--new-only", type=int, default=0, help="阶段 1 本次新跑上限（触发部分模式）")
    ap.add_argument("--all", action="store_true", help="阶段 1 复核全部（默认即全部）")
    ap.add_argument("--budget", default="standard", help="阶段 1 采样档，默认 standard")
    a = ap.parse_args(argv)

    if a.check:
        return cmd_check()

    rows = _load_results()
    rc = _precheck()
    if rc:
        return rc
    partial = bool(a.limit or a.new_only)
    stage1 = ["--budget", a.budget]
    if not partial:
        stage1.append("--all")
    if a.limit:
        stage1 += ["--limit", str(a.limit)]
    if a.new_only:
        stage1 += ["--new-only", str(a.new_only)]

    print("阶段 1: std_recheck.main(%s)   [MCMC，断点续跑]" % " ".join(stage1))
    if partial:
        print("  ⚠ 部分模式 → 阶段 2 将跳过（防完整汇总表被部分结果覆盖）")
    else:
        print("阶段 2: layer0_summary.main()  [只读合并 → 覆盖 LAYER0_SUMMARY.md]")
    if a.dry:
        print("[dry] 未写任何文件")
        return 0

    before = {p: _sha(p) for p in (STDOUT, SUMMARY)}
    buf = StringIO()
    with redirect_stdout(buf):
        r1 = REC.main(stage1)
    # 阶段 1 的表格只把末 6 行摘要透出，避免淹没输出
    tail = [l for l in buf.getvalue().splitlines() if l.strip()][-6:]
    print("  阶段 1 rc=%d：" % r1)
    for l in tail:
        print("    " + l)
    if r1:
        print("[FAIL] 阶段 1 非零退出，中止（不写汇总表）")
        return r1
    if partial:
        # std_recheck 在 --limit 下把 std_recheck.json 的 rows 截成 N 条；
        # progress.jsonl 始终是全量 append-only，用它重建可抵消这个副作用。
        prog = [json.loads(l) for l in open(PROG, encoding="utf-8") if l.strip()]
        cur = json.load(open(STDOUT, encoding="utf-8"))
        n_cur = len(cur.get("rows", []))
        if n_cur != len(prog):
            cur["rows"] = prog
            # 与 std_recheck.py 的原生写法一致：不指定 newline（Windows 下写 CRLF），
            # 且不补末尾换行——否则全量运行时会看到纯格式造成的字节差异。
            with open(STDOUT, "w", encoding="utf-8") as fh:
                json.dump(cur, fh, ensure_ascii=False, indent=1, allow_nan=False)
            print("[fix] 部分模式副作用已抵消：std_recheck.json rows 从 %d 重建为 %d"
                  % (n_cur, len(prog)))
        else:
            print("[ok] std_recheck.json rows 仍为 %d 条" % len(prog))
        print("[skip] 部分模式：阶段 2 未执行，汇总表保持原样")
        return 0

    buf2 = StringIO()
    with redirect_stdout(buf2):
        r2 = SUM.main()
    if r2:
        print("[FAIL] 阶段 2 非零退出:", r2)
        return r2
    rc = _postcheck(rows)
    changed = [os.path.relpath(p, REPO) for p in (STDOUT, SUMMARY)
               if before[p] != _sha(p)]
    print("  产物变化:", ", ".join(changed) if changed else "无（幂等）")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
