# -*- coding: utf-8 -*-
"""b3b_precheck.py —— B3b 的前置检查：这三条谱的初值到底是不是被钉住的？

为什么需要这一步：`bm_core.fit()` 只有一条随机流 `PRNGKey(seed)`，**初始化和 NUTS
共用它**。`init_values` 存在时走确定性 `init_to_value`（初值与 seed 无关）；为 None
时 NumPyro 的默认初始化策略也消耗这条流，初值可能随 seed 变。所以 B3b 能被解释成
"纯轨迹随机流敏感性"还是"整条链路的 seed 敏感性"，取决于 `seeded`。

判定口径（与 `bm_real.fit_one()` 完全一致）：
    seeded = init_values(pk, pres, K) is not None
本脚本对每个 K 都算一遍——因为 `--wide` 会扫 K=1..5，只要有一个 K 没播种，
那次拟合的初值就可能随 seed 变。

只读：不拟合、不采样、不写结果 json，只读谱文件与 config.json。
依赖 `bm_real` 的内部函数（`_kk_check` / `_kk_estimate_L` / `prescreen_stage` /
`init_values`）——商用路径变了这里要跟着改，所以输出里记了 bm_real 的关键行号锚点。

用法：
    python evidence/m6/b3b_precheck.py                 # 默认 3 条（B3b 目标谱）
    python evidence/m6/b3b_precheck.py --kmax 5 --out evidence/m6/b3b_precheck.txt
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import bm_real as R                                    # noqa: E402

# B3b 的三条目标谱（M6 §4.1 B3b 行）。Blank 主谱取无编号的那条。
DEFAULT = [
    ("Blank_main", "archive/Blank_0.1KBr_pH10_EIS_PtCu_1.2V.txt"),
    ("AZ31-4Li_100", "out/_m6_samples/z_csv/AZ31-4Li_100.csv"),
    ("AZ31-12Li_0", "out/_m6_samples/z_csv/AZ31-12Li_0.csv"),
]

# 与 bmdrt.py 的 KK_OPT_KEYS 同源；config 的 kk 段里还混着 mode/_说明，不能整包透传
KK_OPT_KEYS = ("c", "max_M", "res_tol", "auto_L", "auto_tail", "auto_cap")


def _cfg():
    c = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    kk = {k: c["kk"][k] for k in KK_OPT_KEYS if k in (c.get("kk") or {})}
    drt = {k: v for k, v in (c.get("drt") or {}).items() if k != "_说明"}
    return kk, drt


def check_one(path, kk_opts, drt_opts, kmax, quiet=True):
    """复现 run() 的 ⓪→① 段（KK → 去电感 → DRT 预判），再逐个 K 判 seeded。"""
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    with ctx:
        f, Z = R.load_spectrum(path)
        include_L = R.detect_L(f, Z)
        kk = (R._kk_check(f, Z, include_L=include_L, **kk_opts)
              if R._kk_check is not None
              else {"status": "unavailable", "reason": "无 kk_check"})
        deind, Z_drt = None, Z
        if drt_opts.get("deinduce", True) and R._kk_estimate_L is not None:
            try:
                deind = R.deinduce_for_drt(
                    f, Z, L_est=(kk or {}).get("L_est"),
                    n_hi=int(drt_opts.get("n_hi", 5)),
                    min_share=float(drt_opts.get("min_share", 5e-3)),
                    tol_stab=float(drt_opts.get("tol_stab", 0.35)),
                    min_gain=float(drt_opts.get("min_gain", 0.20)),
                    tol_neg=float(drt_opts.get("tol_neg", 0.02)))
                if deind.get("used"):
                    Z_drt = Z - 1j * (2 * 3.141592653589793 * f) * float(deind["L"])
            except Exception as e:                     # noqa: BLE001
                deind = {"used": False,
                         "reason": f"去电感通道异常，已跳过：{type(e).__name__}: {e}"}
        pres, pk, _pk_all, _ = R.prescreen_stage(f, Z_drt, deind=deind)
        flags = {}
        for K in range(1, kmax + 1):
            flags[K] = R.init_values(pk, pres, K) is not None

    kk_status = (kk or {}).get("status", "unavailable")
    # KK 判 fail 时 run() 直接拒绝拟合（_write_kk_reject），seeded 不适用
    applicable = kk_status != "fail"
    return dict(
        n_pts=int(f.size),
        f_min=float(f.min()), f_max=float(f.max()),
        kk_status=kk_status,
        include_L=bool(include_L),
        deind_used=bool(deind and deind.get("used")),
        n_peaks=int(len(pk)) if pk is not None else 0,
        seeded={int(k): bool(v) for k, v in flags.items()},
        seeded_applicable=applicable,
        verdict=("not_applicable" if not applicable
                 else ("all_seeded" if all(flags.values()) else
                       ("none_seeded" if not any(flags.values()) else "partial"))),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int, default=5, help="K 上界（--wide 用 5）")
    ap.add_argument("--out", default="evidence/m6/b3b_precheck.txt")
    ap.add_argument("--show-drt", action="store_true", help="把 DRT 预判打印出来")
    a = ap.parse_args(argv)

    kk_opts, drt_opts = _cfg()
    rows = []
    for name, rel in DEFAULT:
        p = ROOT / rel
        if not p.exists():
            rows.append((name, rel, None))
            continue
        rows.append((name, rel, check_one(p, kk_opts, drt_opts, a.kmax,
                                          quiet=not a.show_drt)))

    lines = [
        "# B3b 前置检查：seeded 状态（只读，未拟合）",
        "# seeded=true  → init_to_value 固定初值，B3b 可解释为「轨迹随机流敏感性」",
        "# seeded=false → NumPyro 默认初始化也吃这条随机流，初值可能随 seed 变",
        "# 判据口径：seeded = bm_real.init_values(pk, pres, K) is not None",
        f"# K 范围 1..{a.kmax}；kk_opts={kk_opts}；deinduce={drt_opts.get('deinduce', True)}",
        "",
        "%-14s %6s %-10s %5s %5s %6s %-12s %s"
        % ("spectrum", "pts", "KK", "incL", "deind", "n_peaks",
           "seeded(K=1..%d)" % a.kmax, "verdict"),
    ]
    bad = 0
    for name, rel, r in rows:
        if r is None:
            lines.append("%-14s  MISSING  %s" % (name, rel))
            bad += 1
            continue
        flags = "".join("T" if r["seeded"][k] else "F"
                        for k in range(1, a.kmax + 1))
        lines.append("%-14s %6d %-10s %5s %5s %6d %-12s %s"
                     % (name, r["n_pts"], r["kk_status"], r["include_L"],
                        r["deind_used"], r["n_peaks"], flags, r["verdict"]))

    verdicts = [r["verdict"] for _, _, r in rows if r]
    lines += ["", "总判定：" + ("全部 all_seeded → B3b 可按「初值固定」解释"
                              if verdicts and all(v == "all_seeded" for v in verdicts)
                              else "存在 partial / none_seeded → B3b 只能按「当前初始化策略下的随机流敏感性」解释"),
              "注：本检查只回答「初值是否被钉住」，不回答「结果是否稳定」；"
              "后者是 B3b 正式实验的事，样本量远不止一次极短档。"]

    txt = "\n".join(lines) + "\n"
    print(txt)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt, encoding="utf-8", newline="\n")
    print(f"已写入 {a.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
