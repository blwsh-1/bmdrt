"""cmd_status —— 看 out/ 里所有跑过的结果，不用重跑。

这是"往目录里加一个文件就多一个子命令"的样板：
只要文件名叫 cmd_<名字>.py，里面有一个 `run(argv) -> int`，
`python bmdrt.py <名字>` 就能用，bmdrt.py 本身一行都不用改。

用法：
    python bmdrt.py status          # 汇总所有历史结果
    python bmdrt.py status --bad    # 只看需要人看一眼的
    python bmdrt.py status --write  # 顺便重建 out/summary.md 与 summary.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _is_info(w: str) -> bool:
    """和 bmdrt._is_info 同义：说明性内容不算警告。"""
    return w.startswith("说明") or w.startswith("已启用 L")


def _verdict(d: dict) -> str:
    # KK 门禁拒绝：流程正常完成、但没拟合、不输出 K。
    # 必须排在所有拟合指标之前判——这种记录里 grade/K 都是 None，
    # 继续往下走会被误读成「采样不可用」。
    if d.get("fit_performed") is False or d.get("kk_rejected"):
        return "KK 拒绝"
    g = d.get("sample_grade", d.get("grade"))
    if g == 0:
        return "采样不可用"
    if g == 1:
        return "仅可用档"
    comps = d.get("components") or []
    if any(c.get("p_outside", 0) > 0.5 for c in comps):
        return "有分量出窗口"
    if d.get("n_out"):
        return "有分量出窗口"
    if [w for w in (d.get("warnings") or []) if not _is_info(w)]:
        return "看警告"
    return "OK"


def _pad(s, n: int) -> str:
    """按显示宽度补空格（中文占两格）。"""
    import unicodedata
    s = str(s)
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(0, n - w)


def _from_meta(mp: Path) -> dict | None:
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return None


def _from_result(js: Path) -> dict | None:
    """老结果没有 meta.json，就从 real_*.json 反推一条记录。"""
    try:
        d = json.loads(js.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {"file": d.get("file", js.stem), "stem": js.stem, "ok": True,
            "skip": False, "err": None, "outdir": str(js.parent),
            # 老结果没有 meta.json 时，KK 拒绝只能从这里读出来；不传下去
            # 就会被 _verdict 当成「看警告」。
            "fit_performed": d.get("fit_performed", True),
            "kk_rejected": bool(d.get("fit_performed") is False),
            "kk_status": (d.get("kk") or {}).get("status"),
            "t_s": d.get("t_fit_s", 0.0), "K": d.get("K"),
            "grade": d.get("sample_grade"), "elpd": d.get("elpd_loo"),
            "elpd_diff": d.get("elpd_diff"), "dse": d.get("dse"),
            "rms_re": d.get("rms_re"), "rms_im": d.get("rms_im"),
            "sig_re": d.get("sigma_re"), "sig_im": d.get("sigma_im"),
            "cover": min(d.get("cover_re", 1.0), d.get("cover_im", 1.0)),
            "n_comp": len(d.get("components") or []),
            "n_out": sum(1 for c in (d.get("components") or [])
                         if c.get("p_outside", 0) > 0.5),
            "warnings": d.get("warnings") or [],
            "ksel": d.get("k_selection") or []}


def run(argv: list[str]) -> int:
    only_bad = "--bad" in argv
    rebuild = "--write" in argv
    out = ROOT / "out"
    dirs = sorted(p for p in out.iterdir() if p.is_dir()) if out.exists() else []
    if not dirs:
        print(f"[status] {out} 里还没有结果")
        return 0

    rows, recs = [], []
    for d in dirs:
        meta = d / "meta.json"
        if meta.exists():
            r = _from_meta(meta)
            t = datetime.fromtimestamp(meta.stat().st_mtime).strftime("%m-%d %H:%M")
        else:
            js = sorted(d.glob("real_*.json"))
            if not js:
                continue
            r = _from_result(js[0])
            t = datetime.fromtimestamp(js[0].stat().st_mtime).strftime("%m-%d %H:%M")
        if r is None:
            rows.append((d.name, "?", "-", "-", "-", "读不出来", "-"))
            continue
        recs.append(r)
        v = "跳过" if r.get("skip") else ("失败" if not r.get("ok") else _verdict(r))
        el = f"{r['elpd']:.1f}" if r.get("elpd") is not None else "-"
        K = r.get("K") if r.get("K") is not None else "-"
        g = r.get("grade") if r.get("grade") is not None else "-"
        rows.append((str(r.get("file", d.name)), K, g, el,
                     f"{r.get('t_s', 0):.0f}s", v, t))

    if only_bad:
        rows = [x for x in rows if x[5] not in ("OK", "跳过")]

    print(_pad("文件", 28) + f"{'K':>3}{'分级':>4}{'elpd':>8}{'耗时':>7}  "
          + _pad("判定", 16) + "时间")
    for name, K, g, el, ts, v, t in rows:
        print(_pad(name, 28) + f"{str(K):>3}{str(g):>4}{el:>8}{ts:>7}  "
              + _pad(v, 16) + t)

    n = len(rows)
    bad = [x for x in rows if x[5] not in ("OK", "跳过")]
    kinds = "、".join(sorted({x[5] for x in bad})) or "无"
    print(f"\n共 {n} 个结果，其中 {len(bad)} 个需要看一眼（{kinds}）")

    if rebuild:
        import bmdrt as M
        cfg = M.resolve(M.load_cfg())
        md = M.write_summary(recs, cfg)
        print(f"已重建：{md}")
    return 0
