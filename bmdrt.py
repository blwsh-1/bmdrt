#!/usr/bin/env python
"""bmdrt —— BM-DRT 的唯一入口。

设计目标（按"以后不再人工点某个文件"来定的）：

1. **一个入口**。拟合、批处理、监听、自检、benchmark 全从这里进。
   使用者不需要知道 `bm_real.py` / `bm_check.py` 叫什么，也不需要记得参数。
2. **无参数也能跑**。`python bmdrt.py` 等价于"把 inbox/ 里的文件全跑一遍"。
   命令行参数只用来**临时覆盖** config.json 的默认值。
3. **自主运行的契约**：
      一个输入文件 → 一个独立输出目录 → 一份完整日志 → 汇总表里一行。
   任何一个文件失败都不中断整批；退出码反映整批成败（0 = 全成功）。
4. **可增删**。任何 `cmd_<名字>.py` 里写一个 `run(argv)` 函数，就自动多一个子命令，
   不用改这个文件。以后加功能只往这个目录里加文件。

用法：
    python bmdrt.py                      # = auto：处理 inbox/
    python bmdrt.py auto <文件/目录...>   # 指定输入
    python bmdrt.py watch                # 常驻监听 inbox/
    python bmdrt.py check [--fit]        # 数学自检（--fit 连拟合一起跑）
    python bmdrt.py diff                 # 扩散项回归（B1/B2/C1/Wo/B2-ident/DRT）
    python bmdrt.py diff --fast          # 只跑 B1/B2/C1，比全量少一半（B2 最贵）
    python bmdrt.py bench                # 合成 benchmark
    python bmdrt.py list                 # 看 inbox 里有什么
    python bmdrt.py config               # 打印当前生效配置
    python bmdrt.py help                 # 这个帮助

覆盖项（对 auto / watch 有效）：
    --krange 1,4   --wide   --kmax 6   --quick   --ta 0.95   --seed 1
    --L / --no-L   --zeta 1.0|free   --diffusion none|Ws|Wo|both
    --kk gate|warn|off   --auto-mechanism / --auto-mechanism-full
    --no-move   --out DIR   --inbox DIR   --interval 30
    --ref-rwp    # 额外落一份参考性 Rwp（独立 LS + 逐点相对权重重拟合）
                 # **只读对照**：不进 kk 门禁、不参与选 K、不改参数推断

    --seed 默认 0（= bm_real.KW 的默认值，旧命令逐位不变）。它换的是 NUTS 的
    随机流：同 seed 重跑逐位相同，不同 seed 才可能改变采样诊断与后验读数。
    **初始化策略不变，但初值不一定不变**——DRT 播种成功（结果里 seeded=true，
    走 init_to_value）时初值是确定的，只有轨迹随机流变；未播种（seeded=false）
    时 NumPyro 的默认初始化策略也消耗这条随机流，初值可能随 seed 改变。
    模型与采样配置（warmup/samples/链数/ta/树深）不受 --seed 影响。
    实际生效的 seed 与采样配置会写进结果 json 的 `seed` / `sampling` 两个
    字段——看那里，不要相信命令行。

    --kk 默认 gate：拟合前先过 Lin-KK，判 fail 就写 fit_performed=false
    的结果 json 并拒绝定 K（不是运行失败）。数值阈值在 config.json 的 kk 段。

    --auto-mechanism 默认关：先按 none 跑 K；满足下面任一条件才在同一个 K 上
    补跑一次 Ws，用逐点配对 LOO 决定是否切换（Δ ≥ max(2, 1.5·dse)）：
      · none 的残差仍有系统性（acf>0.8 或最低频段 rms/σ>1.5）；
      · 低频尾是 45° 候选（`bm_kk` 事前形状判据 `is_candidate`）。
    两条都不成立就跳过，保持 none。
    --auto-mechanism-full 强制每条谱都补跑 Ws（验证用，成本约 +50%）。
    数据侧端点走向（`bm_ws`：still_45/to_cap/to_plateau/to_inductive）随门控
    一起打印并落盘：`to_inductive`（低频 -Im<0）直接登记为感抗尾、不补跑；
    其余档只记录家族，不改门控结论。
    select_k 语义不变。

    --diffusion 默认 none（与改动前逐位一致）。Ws=半无限 Warburg，
    Wo=有限长阻挡 Warburg。两者高频尾相同，both 只作对照，别当最终模型。
    设计依据见 DIFFUSION_PLAN.md。

    diff 的耗时**对硬件敏感**，别把某个秒数当固定值：
    本机 --fast ≈18 s / 全量 ≈39 s；另一台机器实测 --fast ≈33.5 s。
    按 2~4 倍估预算。逐项：B1 6.4 / B2 9.5 / C1 1.7 / Wo 11.4 /
    B2-ident 6.7 / DRT 0.1 s（本机），B2 的多起点优化是 --fast 的大头。

目录约定（以后增删都在这个目录里做）：
    bmdrt.py            入口，**只有这一个**
    config.json         默认参数（改这里就不必记命令行）
    bm_core.py          模型（HN 基底 + stick-breaking）、NUTS、诊断原语
    bm_experiments.py   R̂ / bulk & tail ESS / 树深触顶 / acf
    drt.py              传统 DRT 预判断（Tikhonov + 峰识别）
    bm_real.py          主流程：读数据 → KK 门禁 → 预判断 → K 扫描 → 配对 LOO 选 K → 出图
    bm_kk.py            拟合前 Lin-KK 分级门禁（ok/warn/fail/unavailable）
    bm_corr.py          全参数后验相关矩阵 + 90% 区间（可对 npz 单独重算）
    bm_circuit.py       等效电路图（把选中的模型画成电路，参数标在元件上）
    bm_diff.py          扩散元件（W_inf/W_open/W_short）+ HN↔Warburg 退化诊断
                        + Ws/Wo 可分辨性（本频段能不能分）
    bm_check.py         数学自检（18 项）
    bm_bench.py         合成 benchmark
    bm_ident.py         可辨识性分析（bm_check 的峰位判据用它）
    cmd_diff.py         扩散项回归测试（常数 σ 最小二乘，非 MCMC）
    cmd_status.py       汇总 out/ 里的结果表
    cmd_<名字>.py       可选：文件里写 run(argv) 就自动多一个子命令
    docs/               设计与验证文档（总索引 docs/INDEX.md）
    inbox/              丢文件进来，无参数运行就会处理
    out/                每个文件一个目录 + summary.md / summary.json
    archive/            跑完的输入移到这里
    rejected/           没过门槛（点太少、频段太窄、读不进来）的输入
    examples/           6 个样例，含两种列格式、一个无表头文件、一个含 Warburg

运行产物（每个输入一份，互不覆盖）：
    out/<名>/real_<名>.png          2×3 诊断图（数据 + 拟合 + 残差 + DRT）
    out/<名>/real_<名>_circuit.png  等效电路图（参数 = 后验中位数）
    out/<名>/real_<名>.json         全部指标（汇总表就是从它抽的）
    out/<名>/real_<名>_samples.npz  后验样本
    out/<名>/run.log                这一次的完整控制台输出
"""

from __future__ import annotations

import contextlib
import importlib
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.json"

DEFAULTS = {
    "paths": {"inbox": "inbox", "out": "out", "archive": "archive",
              "rejected": "rejected"},
    "fit": {"krange": None, "wide": False, "kmax": 5, "quick": False,
            "ta": None, "include_L": None, "zeta": "1.0",
            "sigma_floor": 0.0, "seed": None, "auto_mechanism": False,
            "auto_mechanism_full": False,
            "adsorption": None, "auto_adsorption": False},
    "kk": {"mode": "gate", "c": 0.85, "max_M": 50, "res_tol": 0.05,
           "auto_L": True, "auto_tail": True, "auto_cap": True},
    "drt": {"deinduce": True, "n_hi": 5, "min_share": 5e-3,
            "tol_stab": 0.35, "min_gain": 0.20, "tol_neg": 0.02},
    "diag": {"corr_strong": 0.90, "corr_medium": 0.70,
             "diff_resolution_tol": 0.02},
    "batch": {"move_done": True, "skip_done": True, "min_points": 8,
              "min_decades": 1.0, "ext": [".csv", ".tsv", ".txt", ".dat"]},
    "watch": {"interval_s": 30},
}

# 只有这些键会传给 bm_kk.kk_check()；配置里多写别的键不会让门禁崩。
# （`strict` 不在此列：它由 bm_real 的 gate 逻辑消费，不进 kk_check。）
KK_OPT_KEYS = ("c", "max_M", "res_tol", "auto_L", "auto_tail", "auto_cap")


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_cfg() -> dict:
    if CONFIG.exists():
        try:
            raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        except Exception as e:                       # 配置坏了不能静默
            print(f"[bmdrt] config.json 解析失败（{e}），改用内置默认值")
            raw = {}
    else:
        raw = {}
    return _deep_merge(DEFAULTS, raw)


def resolve(cfg: dict) -> dict:
    """把配置里的相对路径落到脚本所在目录，避免"在哪儿启动"影响结果位置。"""
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = (p if p.is_absolute() else ROOT / p)
    for k in ("inbox", "out", "archive", "rejected"):
        cfg["paths"][k].mkdir(parents=True, exist_ok=True)
    return cfg


# --------------------------------------------------------------------------
# 输出分流：控制台 + 文件
# --------------------------------------------------------------------------

class Tee:
    """同时写控制台和日志文件。stdout 被重定向到调度器时也照写文件。"""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False


# --------------------------------------------------------------------------
# 单个文件的处理
# --------------------------------------------------------------------------

def _quick_probe(path: Path, cfg: dict) -> tuple:
    """拟合前的门槛：点数、频段宽度。返回 (拒绝理由, 说明)；理由为 None 表示放行。"""
    import io
    import numpy as np
    import bm_real as R
    try:
        with contextlib.redirect_stdout(io.StringIO()):   # 别让这次试探性读取污染日志
            f, Z = R.load_spectrum(str(path))
    except SystemExit as e:            # load_spectrum 对坏数据用 sys.exit 报错
        return f"读不进来（{e}）", ""
    except Exception as e:
        return f"读不进来（{type(e).__name__}: {e}）", ""
    n = f.size
    dec = float(np.log10(f.max() / f.min())) if f.min() > 0 else 0.0
    info = f"{n} 点，{f.min():.3g}–{f.max():.3g} Hz（{dec:.2f} decade）"
    if n < cfg["batch"]["min_points"]:
        return f"只有 {n} 个点，少于 min_points={cfg['batch']['min_points']}", info
    if dec < cfg["batch"]["min_decades"]:
        return f"频段只有 {dec:.2f} decade，少于 {cfg['batch']['min_decades']}", info
    return None, info


def process_one(path: Path, cfg: dict) -> dict:
    """跑一个文件，返回记录。异常不外抛——批处理里一个坏文件不能拖垮整批。"""
    import bm_real as R

    fit = cfg["fit"]
    stem = path.stem
    outdir = cfg["paths"]["out"] / stem
    outdir.mkdir(parents=True, exist_ok=True)

    rec = {"file": path.name, "stem": stem, "ok": False, "skip": False, "err": None,
           "outdir": str(outdir), "t_s": 0.0, "K": None, "grade": None,
           "warnings": [], "note": "", "kk_rejected": False, "kk_status": None}

    t0 = time.time()
    log_path = outdir / "run.log"
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"# {datetime.now().isoformat(timespec='seconds')}  {path}\n")
        fh.flush()
        with contextlib.redirect_stdout(Tee(sys.stdout, fh)):
            reason, info = _quick_probe(path, cfg)
            rec["note"] = info
            if reason:
                # 门槛没过 = **拒收**，不是失败。拒收不该让整批退出码变红。
                print(f"\n=== 跳过 {path.name}：{reason} ===")
                rec["err"] = reason
                rec["skip"] = True
                rec["t_s"] = time.time() - t0
                fh.write(f"\n[SKIP] {reason}\n")
                _write_meta(rec, outdir)
                return rec

            R.OUT = outdir                       # 图 / json / npz 都落到这个目录
            R.NULL_CACHE = {}                    # 清空零分布缓存（按 f 分键，换谱自动重算）
            try:
                # 注意：run() 返回的是选中的结果 dict，不是文件路径。
                # 结果文件名由 save_out 决定，这里是它的约定，别改一处忘另一处。
                kkcfg = cfg.get("kk") or {}
                kk_opts = {k: kkcfg[k] for k in KK_OPT_KEYS if k in kkcfg}
                if "strict" in kkcfg:
                    # strict 由 bm_real 的 gate 逻辑消费（unavailable 是否也
                    # 拒绝），不传给 bm_kk.kk_check —— 单独带过去（P2-2）。
                    kk_opts["strict"] = bool(kkcfg["strict"])
                dcfg = cfg.get("diag") or {}
                drtcfg = cfg.get("drt") or {}
                R.run(str(path), K_force=None,
                      krange=fit["krange"], wide=fit["wide"],
                      kmax=int(fit["kmax"]), include_L=fit["include_L"],
                      zeta=str(fit["zeta"]), quick=fit["quick"],
                      ta=fit["ta"],
                      diffusion=(None if str(fit.get("diffusion", "none")).lower()
                                 in ("none", "", "null") else str(fit["diffusion"])),
                      kk_mode=str(kkcfg.get("mode", "gate")),
                      kk_opts=kk_opts, diag=dcfg,
                      drt_opts=drtcfg,
                      sigma_floor=float(fit.get("sigma_floor", 0.0) or 0.0),
                      seed=(None if fit.get("seed") is None
                            else int(fit["seed"])),
                      auto_mechanism=bool(fit.get("auto_mechanism", False)),
                      auto_mechanism_full=bool(fit.get("auto_mechanism_full", False)),
                      adsorption=(None if str(fit.get("adsorption") or "none").lower()
                                  in ("none", "", "null")
                                  else str(fit.get("adsorption"))),
                      auto_adsorption=bool(fit.get("auto_adsorption", False)),
                      # 参考性 Rwp：唯读对照，不参与任何判定（默认 False）
                      ref_rwp=bool(fit.get("ref_rwp", False)))
                js = outdir / f"real_{stem}.json"
                rec["ok"] = js.exists()
                rec["json"] = str(js) if js.exists() else None
                if js.exists():
                    # KK 拒绝 = 流程正常完成，不是崩溃，也不是一次成功拟合。
                    try:
                        d = json.loads(js.read_text(encoding="utf-8"))
                        rec["kk_rejected"] = bool(d.get("fit_performed") is False)
                        rec["kk_status"] = (d.get("kk") or {}).get("status")
                        if rec["kk_rejected"]:
                            print("\n=== KK 门禁拒绝：数据不自洽，未拟合（流程正常完成）===")
                            fh.write("\n[KK-REJECT] 数据不自洽，未拟合\n")
                    except Exception as e:
                        # 读不出来 = 结果不可信，必须记失败；否则 KK 拒绝谱会被
                        # 记成「拟合成功」并进 archive（审查 P2-10）。
                        rec["ok"] = False
                        rec["err"] = (f"结果 json 读取失败："
                                      f"{type(e).__name__}: {e}")
                        print(f"\n=== 结果 json 读取失败：{js} ===")
                else:
                    rec["err"] = "跑完了但没写出结果 json"
            except KeyboardInterrupt:            # Ctrl-C 就该停，不吞
                raise
            except SystemExit as e:
                # bm_real 里有些分支用 sys.exit 报错（如"有效数据行太少"）。
                # 那是一个文件的问题，不能让整批死在这儿。
                rec["err"] = f"中止：{e}"
                print(f"\n!!! {path.name} 中止：{e}")
                fh.write(f"\n[FAIL] {rec['err']}\n")
            except Exception as e:
                rec["err"] = f"{type(e).__name__}: {e}"
                print(f"\n!!! {path.name} 处理失败：{rec['err']}")
                traceback.print_exc()
                fh.write(f"\n[FAIL] {rec['err']}\n")

    rec["t_s"] = time.time() - t0
    _enrich(rec, outdir, stem)
    _write_meta(rec, outdir)
    return rec


def _write_meta(rec: dict, outdir: Path):
    """每个输入都留一份 meta.json（含被拒收的），这样汇总可以事后重建。"""
    try:
        (outdir / "meta.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _enrich(rec: dict, outdir: Path, stem: str) -> dict:
    """把结果 json 里的关键数字抽进汇总记录。没有 json 就保持 None。"""
    js = Path(rec.get("json")) if rec.get("json") else None
    if not (js and js.exists()):
        return rec
    try:
        d = json.loads(js.read_text(encoding="utf-8"))
        fp = d.get("fit_performed", True)
        rec.update(K=d.get("K"), grade=d.get("sample_grade"),
                   elpd=d.get("elpd_loo"), elpd_diff=d.get("elpd_diff"),
                   dse=d.get("dse"), rms_re=d.get("rms_re"),
                   rms_im=d.get("rms_im"), sig_re=d.get("sigma_re"),
                   sig_im=d.get("sigma_im"), acf=d.get("acf"),
                   cover=(min(d.get("cover_re", 1.0), d.get("cover_im", 1.0))
                          if fp else None),
                   ndiv=d.get("divergences"),
                   n_comp=len(d.get("components", [])),
                   n_out=sum(1 for c in d.get("components", [])
                             if c.get("p_outside", 0) > 0.5),
                   warnings=d.get("warnings") or [],
                   ksel=d.get("k_selection") or [],
                   kk_status=(d.get("kk") or {}).get("status"),
                   k_selection_status=d.get("k_selection_status"))
        rec["fig"] = str(outdir / f"real_{stem}.png")
    except Exception as e:
        rec["err"] = rec["err"] or f"读结果 json 失败：{e}"
    return rec


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------

def _is_info(w: str) -> bool:
    """judge() 返回的 tips 里混着"说明性"内容（例如"已启用 L"），
    那不是警告，不该把一次完全正常的拟合标成「看警告」。"""
    return w.startswith("说明") or w.startswith("已启用 L")


def _verdict(r: dict) -> str:
    if r.get("skip"):
        return "跳过"
    if not r["ok"]:
        return "失败"
    if r.get("kk_rejected"):
        return "KK 拒绝"
    if r.get("grade") == 0:
        return "采样不可用"
    if r.get("grade") == 1:
        return "仅可用档"
    if r.get("n_out"):
        return "有分量出窗口"
    if [w for w in (r.get("warnings") or []) if not _is_info(w)]:
        return "看警告"
    return "OK"


def write_summary(recs: list[dict], cfg: dict) -> Path:
    out = cfg["paths"]["out"]
    js = out / "summary.json"
    js.write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                              "config": {k: v for k, v in cfg.items() if k != "paths"},
                              "runs": recs},
                             ensure_ascii=False, indent=2), encoding="utf-8")

    md = out / "summary.md"
    n_fit = sum(1 for r in recs if r["ok"] and not r.get("kk_rejected"))
    n_kk = sum(1 for r in recs if r.get("kk_rejected"))
    lines = [f"# bmdrt 本次批次汇总", "",
             f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　"
             f"共 {len(recs)} 个文件：拟合成功 {n_fit}，KK 拒绝 {n_kk}，"
             f"跳过 {sum(1 for r in recs if r.get('skip'))}，"
             f"失败 {sum(1 for r in recs if not r['ok'] and not r.get('skip'))}",
             "",
             "> 「KK 拒绝」= 拟合前 Lin-KK 判不自洽，流程正常完成但不输出 K；"
             "不计入拟合成功，也不算运行失败。", "",
             "> 全量状态（含以前跑过的）用 `python bmdrt.py status`；"
             "本文件只反映**最近一次**批处理。", ""]
    lines += ["| 文件 | K | 分级 | elpd | Δelpd±dse | rms/σ | PPC | 分量 | 耗时 | 判定 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        if not r["ok"]:
            lines.append(f"| {r['file']} | – | – | – | – | – | – | – | "
                         f"{r['t_s']:.0f}s | **{_verdict(r)}**：{r['err'] or ''} |")
            continue
        if r.get("kk_rejected"):
            lines.append(f"| {r['file']} | – | – | – | – | – | – | – | "
                         f"{r['t_s']:.0f}s | **KK 拒绝**（未拟合） |")
            continue
        q = ""
        if r.get("rms_re") is not None and r.get("sig_re"):
            q = f"{r['rms_re'] / r['sig_re']:.2f}/{r['rms_im'] / r['sig_im']:.2f}"
        dd = (f"{abs(r['elpd_diff']):.1f} ± {r['dse']:.1f}"
              if r.get("elpd_diff") is not None else "")
        el = f"{r['elpd']:.1f}" if r.get("elpd") is not None else "–"
        cv = f"{r['cover']:.3f}" if r.get("cover") is not None else "–"
        lines.append(
            f"| {r['file']} | {r.get('K')} | {r.get('grade')} | "
            f"{el} | {dd} | {q} | {cv} | {r.get('n_comp')} | "
            f"{r['t_s']:.0f}s | {_verdict(r)} |")

    flagged = [r for r in recs if _verdict(r) not in ("OK", "跳过")]
    if flagged:
        lines += ["", "## 需要人看一眼的", ""]
        for r in flagged:
            lines.append(f"### {r['file']}　→ {_verdict(r)}")
            if not r["ok"]:
                lines.append(f"- 错误：`{r['err']}`")
                lines.append(f"- 日志：`{Path(r['outdir']) / 'run.log'}`")
            elif r.get("kk_rejected"):
                lines.append(f"- KK 门禁状态 `{r.get('kk_status')}`："
                             f"数据不自洽，**未拟合、不输出 K**。")
                for t in (r.get("ksel") or []):
                    lines.append(f"- {t}")
                lines.append(f"- 日志：`{Path(r['outdir']) / 'run.log'}`")
            else:
                lines.append(f"- 选出的 K = {r.get('K')}，分级 {r.get('grade')}")
                for t in (r.get("ksel") or []):
                    lines.append(f"- 选择依据：{t}")
                for t in (r.get("warnings") or []):
                    lines.append(f"- 警告：{t}")
            lines.append("")

    skipped = [r for r in recs if r.get("skip")]
    if skipped:
        lines += ["", "## 被门槛拒收（未拟合）", ""]
        for r in skipped:
            lines.append(f"- `{r['file']}`：{r['err']}")

    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


# --------------------------------------------------------------------------
# 输入发现
# --------------------------------------------------------------------------

def scan_inputs(targets: list[str], cfg: dict) -> list[Path]:
    ext = {e.lower() for e in cfg["batch"]["ext"]}
    if not targets:
        targets = [str(cfg["paths"]["inbox"])]
    found: list[Path] = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            for q in sorted(p.iterdir()):
                if q.is_file() and q.suffix.lower() in ext and not q.name.startswith("."):
                    found.append(q)
        elif p.is_file():
            found.append(p)
        else:
            print(f"[bmdrt] 找不到 {t}，跳过")
    # 去重但保序
    seen, out = set(), []
    for q in found:
        k = q.resolve()
        if k not in seen:
            seen.add(k)
            out.append(q)
    return out


def _done_before(path: Path, cfg: dict) -> bool:
    js = cfg["paths"]["out"] / path.stem / f"real_{path.stem}.json"
    try:
        return js.exists() and js.stat().st_mtime > path.stat().st_mtime
    except OSError:
        return False


def _move_to(p: Path, sub: str, cfg: dict) -> Path:
    """把输入移到 archive/ 或 rejected/，重名就加后缀——自主运行时不能因为重名崩掉。"""
    d = cfg["paths"][sub]
    dst = d / p.name
    i = 1
    while dst.exists():
        dst = d / f"{p.stem}_{i}{p.suffix}"
        i += 1
    shutil.move(str(p), str(dst))
    return dst


# --------------------------------------------------------------------------
# 子命令
# --------------------------------------------------------------------------

def _apply_cli(cfg: dict, opt: dict):
    fit = cfg["fit"]
    for k in ("krange", "wide", "kmax", "quick", "ta", "zeta", "diffusion",
              "auto_mechanism", "auto_mechanism_full", "auto_adsorption",
              "seed"):
        if opt.get(k) is not None:
            fit[k] = opt[k]
    # --adsorption：与 --diffusion 同处理，"none"/""/"null" 归一成 None。
    # **必须归一到规范大写 "RL"**：下游 make_model / fit_one 的判据是
    # `adsorption == "RL"`（大小写敏感），直接 .lower() 会让开关静默失效
    # —— 与 --diffusion 不同，那边的取值本身就是小写（ws/wo）。
    if opt.get("adsorption") is not None:
        _ad = str(opt["adsorption"]).strip().lower()
        fit["adsorption"] = {"rl": "RL"}.get(_ad)
    # --sigma-floor 之前不在拷贝列表里，命令行传参被静默忽略（审查 P2-3）。
    if opt.get("sigma_floor") is not None:
        fit["sigma_floor"] = float(opt["sigma_floor"])
    if fit.get("auto_mechanism_full"):
        fit["auto_mechanism"] = True          # full 隐含 auto
    # --no-deinduce：关闭 DRT 去电感通道（README 宣传过但解析器里没有）
    if opt.get("no_deinduce"):
        cfg.setdefault("drt", {})["deinduce"] = False
    # --ref-rwp：只打开参考指标，其余一切照旧（默认 False，输出逐位不变）
    if opt.get("ref_rwp"):
        fit["ref_rwp"] = True
    if opt.get("kk") is not None:
        cfg.setdefault("kk", {})["mode"] = str(opt["kk"])
    if opt.get("L") is True:
        fit["include_L"] = True
    if opt.get("L") is False:
        fit["include_L"] = False
    if opt.get("no_move"):
        cfg["batch"]["move_done"] = False
    if opt.get("out"):
        cfg["paths"]["out"] = Path(opt["out"]).resolve()
        cfg["paths"]["out"].mkdir(parents=True, exist_ok=True)
    if opt.get("inbox"):
        cfg["paths"]["inbox"] = Path(opt["inbox"]).resolve()
    return cfg


def cmd_auto(paths: list[str], opt: dict) -> int:
    cfg = _apply_cli(resolve(load_cfg()), opt)
    files = scan_inputs(paths, cfg)
    if not files:
        print(f"[bmdrt] {cfg['paths']['inbox']} 里没有可处理的文件"
              f"（扩展名 {cfg['batch']['ext']}）")
        return 0

    print(f"[bmdrt] 待处理 {len(files)} 个文件")
    print(f"  配置：K 范围 {'1..' + str(cfg['fit']['kmax']) if cfg['fit']['wide'] else (cfg['fit']['krange'] or '按 DRT 自动')}"
          f"　quick={cfg['fit']['quick']}　ta={cfg['fit']['ta'] or '默认 0.9'}"
          f"　seed={('默认 0' if cfg['fit'].get('seed') is None else cfg['fit']['seed'])}"
          f"　L={cfg['fit']['include_L']}　ζ={cfg['fit']['zeta']}"
          f"　扩散={cfg['fit'].get('diffusion', 'none')}"
          f"　自动机制={cfg['fit'].get('auto_mechanism', False)}"
          f"{'（全量）' if cfg['fit'].get('auto_mechanism_full') else ''}"
          f"　KK={(cfg.get('kk') or {}).get('mode', 'gate')}")

    recs = []
    # skip_done 只在"没点名、扫 inbox"时才生效。用户明确点了一个文件，
    # 就是要重跑，不该因为目录里有旧结果而被悄悄跳过。
    use_skip = (not paths) and cfg["batch"]["skip_done"]
    for i, p in enumerate(files, 1):
        if use_skip and _done_before(p, cfg):
            print(f"\n[{i}/{len(files)}] 跳过 {p.name}（已有更新的结果）")
            continue
        print(f"\n{'=' * 70}\n[{i}/{len(files)}] {p.name}\n{'=' * 70}")
        r = process_one(p, cfg)
        recs.append(r)
        in_inbox = p.parent.resolve() == cfg["paths"]["inbox"].resolve()
        if in_inbox and cfg["batch"]["move_done"]:
            # 跑成功的进 archive，被门槛拒收的进 rejected——
            # 两者都要移走，否则 watch 模式会每轮重试同一个文件。
            sub = "rejected" if r.get("skip") else ("archive" if r["ok"] else None)
            if sub:
                _move_to(p, sub, cfg)
                print(f"  → 已移入 {sub}/")

    if not recs:
        print("\n[bmdrt] 没有实际运行的任务")
        return 0
    md = write_summary(recs, cfg)
    bad = [r for r in recs if not r["ok"] and not r.get("skip")]
    skip = [r for r in recs if r.get("skip")]
    kk_rej = [r for r in recs if r.get("kk_rejected")]
    print(f"\n{'=' * 70}")
    print(f"[bmdrt] 完成 {len(recs)} 个：拟合成功 "
          f"{len(recs) - len(bad) - len(skip) - len(kk_rej)}，"
          f"KK 拒绝 {len(kk_rej)}，跳过 {len(skip)}，失败 {len(bad)}")
    print(f"  汇总：{md}")
    print(f"  明细：{cfg['paths']['out'] / 'summary.json'}")
    for r in bad:
        print(f"  失败：{r['file']} —— {r['err']}")
    for r in skip:
        print(f"  跳过：{r['file']} —— {r['err']}")
    return 0 if not bad else 1


def cmd_watch(paths: list[str], opt: dict) -> int:
    cfg = _apply_cli(resolve(load_cfg()), opt)
    iv = float(opt.get("interval") or cfg["watch"]["interval_s"])
    inbox = cfg["paths"]["inbox"]
    print(f"[bmdrt] 监听 {inbox}，每 {iv:.0f}s 扫一次（Ctrl-C 停止）")
    print(f"  结果写 {cfg['paths']['out']}；成功的移入 archive/，被拒收的移入 rejected/")
    n_ok = n_bad = n_skip = n_kk = 0
    recs: list[dict] = []
    try:
        while True:
            files = scan_inputs(paths, cfg)
            todo = [p for p in files
                    if not (cfg["batch"]["skip_done"] and _done_before(p, cfg))]
            for p in todo:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 发现 {p.name}")
                r = process_one(p, cfg)
                recs.append(r)
                # KK 拒绝：流程完成但没拟合，既不算成功拟合也不算失败。
                if r.get("kk_rejected"):
                    n_kk += 1
                else:
                    n_ok += int(r["ok"])
                n_skip += int(bool(r.get("skip")))
                n_bad += int(not r["ok"] and not r.get("skip"))
                if cfg["batch"]["move_done"] and p.parent.resolve() == inbox.resolve():
                    sub = "rejected" if r.get("skip") else ("archive" if r["ok"] else None)
                    if sub:
                        _move_to(p, sub, cfg)
                        print(f"  → 已移入 {sub}/")
            time.sleep(iv)
    except KeyboardInterrupt:
        print(f"\n[bmdrt] 停止。本次拟合成功 {n_ok}，KK 拒绝 {n_kk}，"
              f"跳过 {n_skip}，失败 {n_bad}")
        # 退出前补写汇总（docstring 承诺过），失败有就返回非零，
        # 别让 Ctrl-C 把失败掩盖成成功（审查 P2-10）。
        if recs:
            md = write_summary(recs, cfg)
            print(f"  汇总：{md}")
        return 0 if n_bad == 0 else 1


def _pad(s, n: int, left: bool = True) -> str:
    """按**显示宽度**补空格：中文占两格，否则表格会歪。"""
    import unicodedata
    s = str(s)
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    fill = " " * max(0, n - w)
    return s + fill if left else fill + s


def cmd_list(paths: list[str], opt: dict) -> int:
    cfg = resolve(load_cfg())
    files = scan_inputs(paths, cfg)
    if not files:
        print(f"[bmdrt] {cfg['paths']['inbox']} 是空的")
        return 0
    print(_pad("文件", 34) + _pad("大小", 9, False) + "  修改时间            已有结果")
    for p in files:
        st = p.stat()
        done = (cfg["paths"]["out"] / p.stem / f"real_{p.stem}.json").exists()
        ts = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        print(_pad(p.name, 34) + _pad(f"{st.st_size / 1024:.1f}K", 9, False)
              + f"  {ts}  {'是' if done else ''}")
    return 0


def cmd_config(paths: list[str], opt: dict) -> int:
    cfg = _apply_cli(resolve(load_cfg()), opt)
    print(json.dumps(cfg, ensure_ascii=False, indent=2,
                     default=lambda o: str(o)))
    print(f"\n  入口：{Path(__file__).resolve()}")
    print(f"  目录：{ROOT}")
    return 0


def cmd_check(paths: list[str], opt: dict) -> int:
    import bm_check as C
    C.part_a()
    if opt.get("fit"):
        C.part_b(fast=bool(opt.get("fast")))
    C.summary()
    # 有任一 FAIL 就返回非零，否则自动化按退出码门控会永远绿灯（审查 P2-2）。
    nfail = sum(1 for _, v, _ in C._rows if v == C.NG)
    return 0 if nfail == 0 else 1


def cmd_bench(paths: list[str], opt: dict) -> int:
    import bm_bench as B
    return int(B.main() or 0)


BUILTIN = {"auto": cmd_auto, "watch": cmd_watch, "list": cmd_list,
           "config": cmd_config, "check": cmd_check, "bench": cmd_bench}


def discover() -> dict:
    """任何 cmd_<名字>.py 里的 run(argv) 都自动变成一个子命令。"""
    out = {}
    for p in sorted(ROOT.glob("cmd_*.py")):
        name = p.stem[4:]
        try:
            mod = importlib.import_module(p.stem)
        except Exception as e:
            print(f"[bmdrt] 跳过 {p.name}：{type(e).__name__}: {e}")
            continue
        if callable(getattr(mod, "run", None)):
            out[name] = mod.run
    return out


# --------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------

def parse_opts(argv: list[str]) -> tuple[list[str], dict]:
    """手写小解析器：没有子命令时，所有非选项参数都当成输入路径。

    审查 P1-2（二轮）：未知开关**必须报错**，不允许"照收进 opt、再被
    白名单拷贝悄悄丢弃"。旧行为的实际后果：`--multi-model x.csv`（合法
    开关属于别的入口）会把 `x.csv` 吞成开关取值，`paths=[]`，主命令
    改扫整个 inbox；`--sigam-floor 0.1`（拼错）静默失效。
    """
    # 白名单（post-alias 键名）：
    _VAL_KEYS = {"krange", "kmax", "ta", "zeta", "out", "inbox", "interval",
                 "kk", "adsorption", "sigma_floor", "seed", "diffusion"}
    _KK_MODES = {"gate", "warn", "off"}
    _AD_VALS = {"rl", "none", "null", ""}
    _DIFF_VALS = {"none", "Ws", "Wo", "both"}
    opt = {"krange": None, "wide": None, "kmax": None, "quick": None,
           "ta": None, "zeta": None, "L": None, "no_move": False,
           "out": None, "inbox": None, "interval": None, "kk": None,
           "auto_mechanism": None, "auto_mechanism_full": None,
           "adsorption": None, "auto_adsorption": None,
           "sigma_floor": None, "no_deinduce": False,
           "seed": None, "diffusion": None,
           "ref_rwp": False,
           "fit": False, "fast": False}
    paths, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help", "help"):
            print(__doc__)
            sys.exit(0)
        elif a == "--wide":
            opt["wide"] = True
        elif a == "--quick":
            opt["quick"] = True
        elif a == "--no-move":
            opt["no_move"] = True
        elif a == "--no-deinduce":
            opt["no_deinduce"] = True
        elif a == "--ref-rwp":
            # 参考性 Rwp：只读对照，不进 kk 门禁 / 不参与选 K / 不改参数推断。
            opt["ref_rwp"] = True
        elif a == "--auto-mechanism":
            opt["auto_mechanism"] = True
        elif a == "--auto-mechanism-full":
            opt["auto_mechanism_full"] = True
        elif a == "--auto-adsorption":
            opt["auto_adsorption"] = True
        elif a == "--fit":
            opt["fit"] = True
        elif a == "--fast":
            opt["fast"] = True
        elif a == "--L":
            opt["L"] = True
        elif a == "--no-L":
            opt["L"] = False
        elif a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            k = _alias(k)
            if k not in _VAL_KEYS:
                sys.exit(f"[bmdrt] 未知开关 --{a[2:].split('=', 1)[0]}=…"
                         f"（合法取值开关：{', '.join(sorted(_VAL_KEYS))}）")
            opt[k] = v
        elif a.startswith("--"):
            k = _alias(a[2:])
            if k not in _VAL_KEYS:
                # 布尔旗标在上面已逐个处理；走到这里的就是未知开关。
                sys.exit(f"[bmdrt] 未知开关 {a}（合法取值开关："
                         f"{', '.join(sorted(_VAL_KEYS))}；"
                         f"布尔旗标见 --help）")
            i += 1
            if i >= len(argv):
                sys.exit(f"[bmdrt] {a} 缺少取值")
            opt[k] = argv[i]
        else:
            paths.append(a)
        i += 1
    for k in ("kmax", "seed"):
        if opt[k] is not None:
            try:
                opt[k] = int(opt[k])
            except (TypeError, ValueError):
                # --seed 必须是整数：静默接受再丢掉（本文件以前的通病）比报错危险。
                sys.exit(f"[bmdrt] --{k} 需要整数，收到 {opt[k]!r}")
    for k in ("ta",):
        if opt[k] is not None:
            opt[k] = float(opt[k])
    # 枚举值校验（审查 P1-2 二轮）：--kk foo 以前会静默变成第四种状态。
    if opt["kk"] is not None and opt["kk"] not in _KK_MODES:
        sys.exit(f"[bmdrt] --kk 只接受 {'/'.join(sorted(_KK_MODES))}，"
                 f"收到 {opt['kk']!r}")
    if opt["adsorption"] is not None:
        _ad = str(opt["adsorption"]).strip().lower()
        if _ad not in _AD_VALS:
            sys.exit(f"[bmdrt] --adsorption 只接受 rl/none，收到 "
                     f"{opt['adsorption']!r}")
        opt["adsorption"] = _ad
    if opt["diffusion"] is not None and opt["diffusion"] not in _DIFF_VALS:
        sys.exit(f"[bmdrt] --diffusion 只接受 {'/'.join(sorted(_DIFF_VALS))}，"
                 f"收到 {opt['diffusion']!r}")
    return paths, opt


def _alias(k: str) -> str:
    return {"k": "kmax", "sigma-floor": "sigma_floor"}.get(k, k)


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        return cmd_auto([], {})
    cmd = argv[0]
    rest = argv[1:]

    if cmd in BUILTIN:
        paths, opt = parse_opts(rest)
        return BUILTIN[cmd](paths, opt)

    extra = discover()
    if cmd in extra:
        return int(extra[cmd](rest) or 0)

    # 不是子命令 → 全当输入路径（`python bmdrt.py data.csv` 直接可用）
    paths, opt = parse_opts(argv)
    return cmd_auto(paths, opt)


if __name__ == "__main__":
    sys.exit(main())
