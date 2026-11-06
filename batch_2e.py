"""batch_2e.py —— 把 convert_2e.py 产出的 117 条谱批量跑完，并交付图片/数据。

为什么需要它
------------
`bmdrt.py auto` 是单入口，但一次传 117 个文件是**串行**的，其中任何一个
崩了还会把后面的排队拖住；多个进程共用一个 out 根又会互相覆盖
`summary.json`。这里做三件事：

1. 按 worker 轮询分任务，每个 worker 一个独立 out 根（`w0/ w1/ …`），
   互不覆盖，可断点续跑（已有 `real_<stem>.json` 就跳过）。
2. 用 `ThreadPoolExecutor` 只负责等子进程，CPU 全给 NUTS；线程数即并发数。
   本机 8 核 / 可用内存有限，默认 3 并发。
3. 全部跑完后从 `meta.json` + `real_*.json` + `*_samples.npz` 重建汇总：
   - `summary_all.csv`          117 行指标总表
   - `figures/2E/*_fit.png`      **2×3 诊断图**（Nyquist / Bode |Z| / Bode 相位 /
                                 DRT / 残差 Re / 残差 Im；原始数据与拟合叠加）
   - `figures/2E/*_circuit.png`  **等效电路图**（参数 = 后验中位数）
   - `figures/2E/curves/*.csv`   **拟合后的数据**：逐点观测 / 后验中位拟合 /
                                 90% 带 / 残差
   - `figures/2E/index.html`     按文件 × 电流密度排好的图片索引
   - `2E_figures.zip`            上面全部内容打包

采样设置（**必须在报告里说清**）
--------------------------------
默认 `--quick`：`n_warmup=800, n_samples=400, num_chains=2`（bm_real.py 的
粗筛档）。目的是在合理时间内为 **117 条谱各出一套完整的图**。粗筛档的
`elpd_loo`、采样分级、R̂ / ESS / divergence **不能当结论**，程序自己也会打印
"选定 K 后要重跑"。要严格证据用 `--full`（3000 / 1000 / 3 链，与
`evidence/b2x` 的四组 NUTS 对照同设置）；`--only` 可以只重跑一部分。

用法：
    python batch_2e.py --quick --jobs 3          # 全量 117 条，粗筛
    python batch_2e.py --full  --jobs 3 --only 2EMC1-1   # 只重跑某个源文件的 9 条
    python batch_2e.py --report --quick          # 只重建汇总/图片/zip，不重跑
    python batch_2e.py --full  --jobs 3 --every 9        # 每个源文件抽 1 条严格复跑
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CSVDIR = ROOT / "EISmeasurements" / "csv_2e"
MANIFEST = CSVDIR / "manifest.csv"
PY = sys.executable


# --------------------------------------------------------------------------
# 任务分派
# --------------------------------------------------------------------------
def tasks_from_manifest(only: str | None, every: int | None) -> list[dict]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if only:
        rows = [r for r in rows if only.lower() in r["stem"].lower()
                or only.lower() in r["src"].lower()]
    if every:                       # 每个源文件取第 0 条（9 个 sheet 一组）
        seen, keep = set(), []
        for r in rows:
            if r["src"] not in seen:
                seen.add(r["src"])
                keep.append(r)
        rows = keep
    return rows


def out_root(mode: str) -> Path:
    return ROOT / "out" / ("2E_quick" if mode == "quick" else "2E_full")


def result_dir(root: Path, worker: int, stem: str) -> Path:
    # bmdrt.process_one 把结果放在 <--out>/<stem>/
    return root / f"w{worker}" / stem


def _is_kk_reject(js: Path) -> bool:
    """结果 json 存在但 `fit_performed=false` = K-K 门禁正常拒绝，不是失败。

    这个判断必须在这里做一次：拒绝记录**不会**有诊断图，若只看
    `js.exists() and png.exists()` 就会把它记成 `fail(rc=0)`，
    让整批退出码变红，而实际什么都没坏。
    """
    try:
        return json.loads(js.read_text(encoding="utf-8")).get("fit_performed") is False
    except Exception:
        return False


def _run_one(args) -> dict:
    root, worker, row, mode, timeout, auto_mech, auto_full = args
    stem = row["stem"]
    d = result_dir(root, worker, stem)
    js = d / f"real_{stem}.json"
    png = d / f"real_{stem}.png"
    if js.exists() and (png.exists() or _is_kk_reject(js)):
        # KK 拒绝也算法「已处理完」：它本来就不该有 png，不认这一点会每次重跑。
        return dict(stem=stem, status="skip", t=0.0, worker=worker)

    d.parent.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    log = root / "logs" / f"{stem}.log"
    cmd = [PY, str(ROOT / "bmdrt.py"), "auto",
           str(CSVDIR / row["csv"]),
           "--out", str(root / f"w{worker}"),
           "--no-move"]
    if mode == "quick":
        cmd.append("--quick")
    if auto_full:
        cmd.append("--auto-mechanism-full")
    elif auto_mech:
        cmd.append("--auto-mechanism")
    env = os.environ.copy()
    # 每个 NUTS 进程限单线程 BLAS，避免 3 个进程各起 8 条线程互相踩
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    t0 = time.time()
    try:
        with log.open("w", encoding="utf-8") as fh:
            fh.write(f"# {' '.join(cmd)}\n")
            fh.flush()
            cp = subprocess.run(cmd, cwd=str(ROOT), stdout=fh,
                                stderr=subprocess.STDOUT, env=env,
                                timeout=timeout)
        rc = cp.returncode
    except subprocess.TimeoutExpired:
        rc = -9
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n[TIMEOUT] {timeout} s\n")
    if js.exists() and png.exists():
        status = "ok"
    elif js.exists() and _is_kk_reject(js):
        status = "kk_reject"          # 正常拒绝：不算失败，不计入成功拟合
    else:
        status = f"fail(rc={rc})"
    return dict(stem=stem, status=status, t=time.time() - t0, worker=worker)


def run_batch(mode: str, jobs: int, only: str | None, every: int | None,
              timeout: int, auto_mech: bool = False,
              auto_full: bool = False) -> int:
    rows = tasks_from_manifest(only, every)
    if not rows:
        print("[batch_2e] 没有匹配的任务")
        return 1
    root = out_root(mode)
    root.mkdir(parents=True, exist_ok=True)
    groups = [[(root, j, r, mode, timeout, auto_mech, auto_full)
               for r in rows[j::jobs]] for j in range(jobs)]
    print(f"[batch_2e] mode={mode}  {len(rows)} 条谱  {jobs} 并发  out={root}")
    t0 = time.time()
    done, bad, skipped, kk_rej = 0, [], 0, 0
    with open(root / "progress.log", "a", encoding="utf-8") as pf:
        def worker_fn(group):
            return [_run_one(a) for a in group]
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(worker_fn, g) for g in groups]
            for fu in as_completed(futs):
                for res in fu.result():
                    done += 1
                    if res["status"] == "skip":
                        skipped += 1
                    elif res["status"] == "kk_reject":
                        kk_rej += 1
                    elif res["status"] != "ok":
                        bad.append((res["stem"], res["status"]))
                    line = (f"[{done:3d}/{len(rows)}] {res['stem']:<28} "
                            f"{res['status']:<12} {res['t']:6.1f}s "
                            f"w{res['worker']}  累计 {time.time()-t0:.0f}s")
                    print(line, flush=True)
                    pf.write(line + "\n")
                    pf.flush()
    print(f"[batch_2e] 完成：{done - len(bad) - skipped - kk_rej} 新跑 / "
          f"{kk_rej} KK 拒绝（未拟合）/ {skipped} 已存在跳过 / "
          f"{len(bad)} 失败，总耗时 {time.time()-t0:.0f}s")
    for s, why in bad:
        print(f"    失败 {s}: {why}（看 out/2E_*/logs/{s}.log）")
    return 0 if not bad else 1


# --------------------------------------------------------------------------
# 汇总 / 图片 / 拟合曲线
# --------------------------------------------------------------------------
FIELDS = ["stem", "src", "sheet", "j_mA_cm2", "sampling", "fit_performed",
          "kk_status", "K", "grade",
          "elpd_loo", "rms_re", "rms_im", "sigma_re", "sigma_im", "acf",
          "p_peak_out_max", "p_tau_out_max", "frac_neg", "divergences",
          "rhat_max", "ess_bulk_min", "include_L", "n_outside", "t_fit_s",
          "judge", "reasons", "n_warn",
          "fit_png", "circuit_png", "curve_csv", "err"]


def _collect(root: Path, mode: str) -> list[dict]:
    import bm_real as R            # 只为 z_band 重建拟合曲线；不需要 numpyro
    rows = tasks_from_manifest(None, None)
    meta = {r["stem"]: r for r in rows}
    figdir = ROOT / "figures" / ("2E" if mode == "quick" else "2E_full")
    curvedir = figdir / "curves"
    figdir.mkdir(parents=True, exist_ok=True)
    curvedir.mkdir(parents=True, exist_ok=True)

    out = []
    for w in sorted(root.glob("w*")):
        for d in sorted(w.iterdir()):
            if not d.is_dir():
                continue
            stem = d.name
            m = meta.get(stem, {})
            rec = {k: "" for k in FIELDS}
            rec.update(stem=stem, src=m.get("src", ""), sheet=m.get("sheet", ""),
                       j_mA_cm2=m.get("j_mA_cm2", ""), sampling=mode)
            js = d / f"real_{stem}.json"
            mt = d / "meta.json"
            if mt.exists():
                try:
                    x = json.loads(mt.read_text(encoding="utf-8"))
                    rec["err"] = x.get("err") or ""
                    rec["t_fit_s"] = x.get("t_s", "")
                except Exception:
                    pass
            if not js.exists():
                out.append(rec)
                continue
            try:
                j = json.loads(js.read_text(encoding="utf-8"))
            except Exception as e:
                rec["err"] = f"读 json 失败：{e}"
                out.append(rec)
                continue
            # ---- K-K 门禁拒绝：保留成一行，但不参与任何拟合统计 ----
            fp = j.get("fit_performed", True)
            rec["fit_performed"] = fp
            rec["kk_status"] = (j.get("kk") or {}).get("status") or ""
            if not fp:
                rec["judge"] = "KK 拒绝（未拟合）"
                rec["reasons"] = (j.get("kk") or {}).get("reason", "") or "数据不自洽"
                out.append(rec)
                continue

            dg = j.get("diag") or {}
            rh = [v["rhat"] for v in dg.values()
                  if isinstance(v, dict) and np.isfinite(v.get("rhat", np.nan))]
            eb = [v["bulk"] for v in dg.values()
                  if isinstance(v, dict) and np.isfinite(v.get("bulk", np.nan))]
            ppm = j.get("p_peak_out_max")
            ppm = float(ppm) if ppm is not None else float("nan")
            rec.update(
                K=j.get("K"), grade=j.get("sample_grade"),
                elpd_loo=j.get("elpd_loo"), rms_re=j.get("rms_re"),
                rms_im=j.get("rms_im"), sigma_re=j.get("sigma_re"),
                sigma_im=j.get("sigma_im"), acf=j.get("acf"),
                p_peak_out_max=(ppm if np.isfinite(ppm) else "NaN"),
                p_tau_out_max=j.get("p_tau_out_max"),
                frac_neg=j.get("frac_neg"), divergences=j.get("divergences"),
                rhat_max=max(rh) if rh else "",
                ess_bulk_min=min(eb) if eb else "",
                include_L=j.get("include_L"),
                n_outside=sum(1 for c in j.get("components", [])
                              if (c.get("p_outside") or 0) > 0.5),
                n_warn=len(j.get("warnings") or []))
            why = []
            if rec["grade"] != 2:
                why.append(f"采样分级={rec['grade']}")
            if not np.isfinite(ppm):
                why.append("峰位诊断不可用")
            elif ppm > 0.5:
                why.append(f"P(峰出窗)={ppm:.2f}")
            if (j.get("frac_neg") or 0) > (j.get("nonphys_tol") or 0.01):
                why.append("非物理样本超限")
            rec["reasons"] = "；".join(why)
            rec["judge"] = "OK" if not why else "需复核"

            # ---- 图片：复制到平铺目录，文件名带 stem ----
            for src, key in ((f"real_{stem}.png", "fit_png"),
                             (f"real_{stem}_circuit.png", "circuit_png")):
                p = d / src
                if p.exists():
                    dst = figdir / (f"{stem}_fit.png" if key == "fit_png"
                                    else f"{stem}_circuit.png")
                    shutil.copy2(p, dst)
                    rec[key] = str(dst.relative_to(ROOT)).replace("\\", "/")

            # ---- 拟合后的数据：观测 / 中位拟合 / 90% 带 / 残差 ----
            npzs = list(d.glob("*_samples.npz"))
            if npzs:
                try:
                    z = np.load(npzs[0], allow_pickle=True)
                    s = {k: z[k] for k in z.keys()}
                    f = np.asarray(s["f"], float)
                    Z = np.asarray(s["Zre"], float) + 1j * np.asarray(s["Zim"], float)
                    br, bi = R.z_band(s, f, bool(j.get("include_L")))
                    med = br[1] + 1j * bi[1]
                    cp = curvedir / f"{stem}_fit.csv"
                    with cp.open("w", newline="", encoding="utf-8") as fh:
                        wcsv = csv.writer(fh)
                        wcsv.writerow(["f_Hz", "Zre_obs", "Zim_obs",
                                       "Zre_fit_med", "Zim_fit_med",
                                       "Zre_fit_lo90", "Zre_fit_hi90",
                                       "Zim_fit_lo90", "Zim_fit_hi90",
                                       "resid_Re", "resid_Im"])
                        for i in range(f.size):
                            wcsv.writerow([f"{f[i]:.10g}",
                                           f"{Z.real[i]:.10g}", f"{Z.imag[i]:.10g}",
                                           f"{med.real[i]:.10g}", f"{med.imag[i]:.10g}",
                                           f"{br[0][i]:.10g}", f"{br[2][i]:.10g}",
                                           f"{bi[0][i]:.10g}", f"{bi[2][i]:.10g}",
                                           f"{Z.real[i]-med.real[i]:.10g}",
                                           f"{Z.imag[i]-med.imag[i]:.10g}"])
                    rec["curve_csv"] = str(cp.relative_to(ROOT)).replace("\\", "/")
                except Exception as e:
                    rec["err"] = (rec["err"] + f" 曲线重建失败：{e}").strip()
            out.append(rec)
    return out


def _index_html(recs: list[dict], figdir: Path, mode: str) -> None:
    """按源文件 × 电流密度分组的图片索引页（本地打开即可看全部图）。"""
    by_src: dict[str, list[dict]] = {}
    for r in recs:
        by_src.setdefault(r["src"] or "unknown", []).append(r)
    for v in by_src.values():
        v.sort(key=lambda x: float(x["j_mA_cm2"] or 0))

    def cell(r):
        fit = Path(r["fit_png"]).name if r["fit_png"] else ""
        cir = Path(r["circuit_png"]).name if r["circuit_png"] else ""
        a = (f'<a href="{html.escape(fit)}" target="_blank"><img src="{html.escape(fit)}"></a>'
             if fit else '<div class="miss">无诊断图</div>')
        b = (f'<a href="{html.escape(cir)}" target="_blank"><img src="{html.escape(cir)}"></a>'
             if cir else '<div class="miss">无电路图</div>')
        g = r["grade"] if r["grade"] != "" else "—"
        return (f'<div class="card"><div class="j">{html.escape(r["j_mA_cm2"])} mA cm<sup>-2</sup>'
                f' <span class="meta">K={r["K"] or "—"} / 分级 {g} / {r["judge"]}</span></div>'
                f'<div class="imgs">{a}{b}</div>'
                f'<div class="stem">{html.escape(r["stem"])}</div></div>')

    parts = [
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
        f"<title>2E 批量拟合图片索引（{mode}）</title>",
        "<style>",
        "body{font:14px/1.5 system-ui,'Microsoft YaHei',sans-serif;margin:24px;background:#fff;color:#222}",
        "h1{font-size:21px} h2{font-size:16px;margin-top:32px;border-bottom:2px solid #1f4e79;padding-bottom:5px;color:#1f4e79}",
        ".note{background:#fff8e1;border-left:4px solid #f0ad4e;padding:10px 14px;margin:14px 0}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:16px}",
        ".card{border:1px solid #ddd;border-radius:8px;padding:10px;background:#fafafa}",
        ".j{font-weight:600;color:#1f4e79;margin-bottom:6px}.meta{font-weight:400;color:#666;font-size:12px;margin-left:8px}",
        ".imgs{display:flex;gap:6px}.imgs img{width:50%;height:auto;border:1px solid #eee;background:#fff}",
        ".stem{font-size:11px;color:#888;margin-top:6px;word-break:break-all}",
        ".miss{padding:30px;background:#eee;color:#999;width:50%;text-align:center}",
        "a{text-decoration:none}</style></head><body>",
        f"<h1>2E 批量拟合图片索引 · {len(recs)} 条谱 / {len(by_src)} 个文件</h1>",
        "<div class='note'><b>每张卡片左</b> = 2×3 诊断图（Nyquist / Bode |Z| / Bode 相位 / "
        "DRT / 残差 Re / 残差 Im，蓝色空圈是实测点、红线是后验中位拟合）；"
        "<b>右</b> = 等效电路图（参数取后验中位数）。点击看大图。<br>",
        f"采样档：<b>{mode}</b>。quick = 800 warmup / 400 样本 / 2 链，"
        "只用于批量出图与相对比较，<b>elpd、采样分级不可当结论</b>；"
        "full = 3000 / 1000 / 3 链。<br>",
        "拟合后的逐点数据在 <code>curves/</code>；指标总表是 <code>summary_all.csv</code>。</div>",
    ]
    for src, rs in by_src.items():
        parts.append(f"<h2>{html.escape(src)} · {len(rs)} 条</h2><div class='grid'>")
        parts += [cell(r) for r in rs]
        parts.append("</div>")
    parts.append("</body></html>")
    (figdir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def make_report(mode: str) -> int:
    root = out_root(mode)
    if not root.exists():
        print(f"[batch_2e] 没有 {root}，先跑批量")
        return 1
    recs = _collect(root, mode)
    recs.sort(key=lambda r: (r["src"], float(r["j_mA_cm2"] or 0)))
    figdir = ROOT / "figures" / ("2E" if mode == "quick" else "2E_full")
    with (root / "summary_all.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(recs)
    shutil.copy2(root / "summary_all.csv", figdir / "summary_all.csv")
    shutil.copy2(MANIFEST, figdir / "manifest.csv")
    _index_html(recs, figdir, mode)

    n_fit = sum(1 for r in recs if r["fit_png"])
    n_cir = sum(1 for r in recs if r["circuit_png"])
    n_cur = sum(1 for r in recs if r["curve_csv"])
    grade2 = sum(1 for r in recs if r["grade"] == 2)
    n_kk = sum(1 for r in recs if r["fit_performed"] is False)
    print(f"[batch_2e] 汇总 {len(recs)} 条：诊断图 {n_fit} / 电路图 {n_cir} / "
          f"拟合曲线 {n_cur}；严格分级 2 的有 {grade2} 条"
          + (f"；KK 门禁拒绝 {n_kk} 条（未拟合）" if n_kk else ""))
    print(f"  图片目录：{figdir}")
    print(f"  图片索引：{figdir/'index.html'}")

    # 打包（图片 + 曲线 + 总表 + 索引 + 转换说明）
    zp = ROOT.parent / (f"2E_figures_{mode}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(figdir.rglob("*")):
            if p.is_file():
                z.write(p, Path("2E") / p.relative_to(figdir))
        z.write(CSVDIR / "README.md", Path("2E") / "CONVERSION_README.md")
    print(f"  打包：{zp}  ({zp.stat().st_size/1e6:.2f} MB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="粗筛采样（默认）")
    ap.add_argument("--full", action="store_true", help="全量采样 3000/1000/3")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only", default=None, help="只跑 stem/src 含此串的谱")
    ap.add_argument("--every", action="store_true",
                    help="每个源文件只取 1 条（用于严格采样抽检）")
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--auto-mechanism", action="store_true",
                    help="none 残差仍有系统性时才补跑 Ws（默认关）")
    ap.add_argument("--auto-mechanism-full", action="store_true",
                    help="每条谱都补跑 Ws，用配对 LOO 决定（验证用，成本 +50%%）")
    ap.add_argument("--report", action="store_true", help="只重建汇总与图片")
    a = ap.parse_args()
    mode = "full" if a.full else "quick"
    if a.report:
        return make_report(mode)
    timeout = a.timeout or (1500 if mode == "full" else 600)
    return run_batch(mode, a.jobs, a.only, a.every, timeout,
                     auto_mech=bool(a.auto_mechanism),
                     auto_full=bool(a.auto_mechanism_full))


if __name__ == "__main__":
    raise SystemExit(main())
