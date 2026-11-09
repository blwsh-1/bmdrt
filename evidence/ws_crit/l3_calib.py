"""L3 闸门标定：给「份额」找一个窗口不变量。

为什么必须重做
--------------
`bm_ws.ws_share_low` 的分母是 `Re Z_Ws(f_min) + R_p`，而 `Re Z_Ws ∝ f_min^(-1/2)`
→ 同一个真值 `A_W=2`，窗口 0.01 Hz 给 0.161、0.1 Hz 给 0.057（`L123_REPORT` §4）。
5% 阈值在窄窗会把真 Ws 判成"可忽略"，所以它只能报、不能判。

本脚本对比**三个候选口径**，看谁能跨窗口稳定地分开「真 Ws」与「错元件」：

  A `share_low`  = Re Z_Ws(f_min) / (Re Z_Ws(f_min) + Σw)      [旧口径，已知漂移]
  B `snr_ws`     = Re Z_Ws(f_min) / σ_im(f_min)                [§3.1 同源]
  C `cv_aw`      = sd(A_W) / |median(A_W)|                     [后验可辨识性]

数据类（固定 `R∞=10`、`R_p=30`、`f0=300 Hz`、`n=0.85`、1% 相对噪声）：

  ws   R∞ + ZARC + Ws(A_W)          → 真 Ws，**应当**切
  wo   R∞ + ZARC + Wo(R_W, f_D)     → 真 Wo，用 Ws 拟合是错元件，**不该**切
  hn   R∞ + ZARC + ZARC(93, 1e-4)   → K+1 的宽弧，**不该**切（与 Ws 同形）
  none R∞ + ZARC                    → 无扩散，**不该**切

窗口下沿扫 {0.01, 0.03, 0.1, 0.3} Hz（上沿固定 1e3），检验跨窗口稳定性。

用法
----
    python evidence/ws_crit/l3_calib.py            # 跑全部（约 30 条拟合）
    python evidence/ws_crit/l3_calib.py --dry      # 只生成数据与真值表
    python evidence/ws_crit/l3_calib.py --only ws,0.01,2
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "out" / "_l3"
PY = sys.executable

FS = np.logspace(-2, 3, 51)          # 0.01–1000 Hz, 51 点
R_INF, RP, F0, NP = 10.0, 30.0, 300.0, 0.85
SIG_REL = 0.01


def zarc(f, R, f0, n=1.0):
    """ZARC：R / (1 + (j f/f0)^n)，低频 → R。"""
    return R / (1.0 + (1j * np.asarray(f, float) / f0) ** n)


def zws(f, A):
    """半无限 Warburg：A / sqrt(jω)。"""
    return A / np.sqrt(1j * 2 * np.pi * np.asarray(f, float))


def _cothz(z):
    z = np.asarray(z, complex)
    out = np.empty_like(z)
    small = np.abs(z) < 1e-6
    out[~small] = 1.0 / np.tanh(z[~small])
    zz = z[small]
    out[small] = 1.0 / zz + zz / 3.0 - zz ** 3 / 45.0
    return out


def zwo(f, Rw, fD):
    """有限长阻挡 Warburg：Rw·coth(z)/z，z=sqrt(jωτ)，τ=1/(2π fD)。"""
    w = 2 * np.pi * np.asarray(f, float)
    tau = 1.0 / (2 * np.pi * fD)
    z = np.sqrt(1j * w * tau)
    return Rw * _cothz(z) / z


def build(kind, a_w=2.0, f_min=0.01, seed=7):
    """返回 (f, Z_true, meta)。噪声按 σ_rel·|Z| 加，与 §3.1 曲面同口径。"""
    f = FS[FS >= f_min - 1e-12].copy()
    if kind == "ws":
        Zt = R_INF + zarc(f, RP, F0, NP) + zws(f, a_w)
        meta = dict(kind=kind, a_w=a_w)
    elif kind == "wo":
        # 转折在窗内、窗下各放一组：f_D=0.03 与 f_D=1e-4
        Zt = R_INF + zarc(f, RP, F0, NP) + zwo(f, 60.0, 0.03)
        meta = dict(kind=kind, Rw=60.0, fD=0.03)
    elif kind == "wo_below":
        Zt = R_INF + zarc(f, RP, F0, NP) + zwo(f, 60.0, 1e-4)
        meta = dict(kind=kind, Rw=60.0, fD=1e-4)
    elif kind == "hn":
        # §9.2 的不可区分参数：R2=93.28, f2=1e-4, n2=0.518
        Zt = R_INF + zarc(f, RP, F0, NP) + zarc(f, 93.28, 1e-4, 0.518)
        meta = dict(kind=kind, R2=93.28, f2=1e-4, n2=0.518)
    elif kind == "none":
        Zt = R_INF + zarc(f, RP, F0, NP)
        meta = dict(kind=kind)
    else:
        raise ValueError(kind)

    rng = np.random.default_rng(seed)
    s = SIG_REL * np.abs(Zt)
    Z = Zt + rng.normal(0, s) + 1j * rng.normal(0, s)
    meta.update(f_min=float(f.min()), f_max=float(f.max()), n=int(f.size),
                sig_rel=SIG_REL,
                # 真值侧的信噪比（§3.1 定义）：|Z_W(f_min)| / (σ_rel·|Z(f_min)|)
                snr_true=float(abs(zws(f.min(), a_w)) / s[0]) if kind == "ws" else None,
                re_ws_true=float(zws(f.min(), a_w).real) if kind == "ws" else None,
                Z_true_re=float(Zt[0].real), Z_true_im=float(Zt[0].imag))
    return f, Z, meta


def write_csv(path, f, Z):
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("f,Re,Im\n")
        for a, b in zip(f, Z):
            fh.write(f"{a:.10g},{b.real:.10g},{b.imag:.10g}\n")


def metrics(npz_path, f_min):
    """从后验样本算三个候选口径。"""
    d = np.load(npz_path, allow_pickle=True)
    if "Aw" not in d.files:
        return dict(ok=False, note="样本里没有 Aw（未跑 Ws）")

    A = np.asarray(d["Aw"], float).ravel()
    w = np.asarray(d["w"], float)
    if w.ndim == 1:
        w = w[:, None]
    n = min(A.size, w.shape[0])
    A, w = A[:n], w[:n]
    good = np.isfinite(A) & (A > 0) & np.isfinite(w).all(axis=1)
    A, w = A[good], w[good]
    if A.size < 50:
        return dict(ok=False, note=f"有效样本只有 {A.size}")

    re_ws = A / np.sqrt(4 * np.pi * f_min)
    rp = w.sum(axis=1)
    share = re_ws / np.maximum(re_ws + rp, 1e-300)

    # σ_im：后验中位的噪声尺度（模型里它是全局标量，同方差）
    sig = np.median(np.asarray(d["s_im"], float)) if "s_im" in d.files else np.nan
    snr = re_ws / sig if np.isfinite(sig) and sig > 0 else np.full_like(re_ws, np.nan)

    med = float(np.median(A))
    cv = float(A.std() / med) if med > 0 else np.nan

    return dict(
        ok=True,
        A_W=med, A_W_sd=float(A.std()),
        cv_aw=cv,
        share_med=float(np.median(share)),
        share_lo=float(np.percentile(share, 5)),
        share_hi=float(np.percentile(share, 95)),
        p_below_5pct=float(np.mean(share < 0.05)),
        snr_med=float(np.nanmedian(snr)),
        snr_lo=float(np.nanpercentile(snr, 5)),
        sig_im=float(sig),
        rp=float(np.median(rp)),
        n=int(A.size),
    )


def run_one(tag, kind, a_w, f_min, dry=False, quick=True):
    f, Z, meta = build(kind, a_w=a_w, f_min=f_min)
    csv = OUT / "data" / f"{tag}.csv"
    write_csv(csv, f, Z)
    rec = dict(tag=tag, spec_min=f_min, **meta)
    if dry:
        return rec

    odir = OUT / "fit" / tag
    cmd = [PY, "bmdrt.py", "auto", str(csv), "--krange", "1,1",
           "--diffusion", "Ws", "--kk", "off", "--no-move",
           "--out", str(odir)]
    if quick:
        cmd.insert(-2, "--quick")
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    rec["rc"] = p.returncode
    if p.returncode != 0:
        rec["err"] = (p.stderr or p.stdout)[-400:]
        return rec

    npzs = sorted(odir.rglob("*_samples.npz"))
    js = sorted(odir.rglob("real_*.json"))
    if not npzs:
        rec["err"] = "没有 npz"
        return rec
    rec.update(metrics(npzs[0], f_min))
    if js:
        try:
            j = json.loads(js[0].read_text(encoding="utf-8"))
            rec["diffusion_params"] = j.get("diffusion_params")
            rec["rms_rel"] = (j.get("residual") or {}).get("rms_rel")
            rec["elpd_loo"] = j.get("elpd_loo")
        except Exception as e:                                     # pragma: no cover
            rec["json_err"] = str(e)[:120]
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只生成数据与真值表")
    ap.add_argument("--only", default="", help="过滤 tag 子串（逗号分隔都可）")
    ap.add_argument("--out", default=str(HERE / "l3_calib.json"))
    args = ap.parse_args(argv)

    cases = []
    # 真 Ws：扫幅度 × 窗口
    for a_w in (0.5, 1.0, 2.0, 4.0):
        for f_min in (0.01, 0.03, 0.1, 0.3):
            cases.append((f"ws_a{a_w:g}_f{f_min:g}", "ws", a_w, f_min))
    # 错元件 / 宽弧 / 无扩散：每个窗口一条
    for kind in ("wo", "wo_below", "hn", "none"):
        for f_min in (0.01, 0.03, 0.1, 0.3):
            cases.append((f"{kind}_f{f_min:g}", kind, 2.0, f_min))

    filt = [s for s in args.only.split(",") if s]
    if filt:
        cases = [c for c in cases if all(s in c[0] for s in filt)]

    rows = []
    old = {}
    if not args.dry and Path(args.out).exists():
        try:
            prev = json.loads(Path(args.out).read_text(encoding="utf-8"))
            old = {r["tag"]: r for r in prev.get("rows", [])}
        except Exception:                                          # pragma: no cover
            old = {}
    for i, (tag, kind, a_w, f_min) in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {tag}", flush=True)
        rec = run_one(tag, kind, a_w, f_min, dry=args.dry)
        old[tag] = rec
        rows.append(rec)
        if not rec.get("ok"):
            print(f"    ! {rec.get('err') or rec.get('note')}", flush=True)
        elif not args.dry:
            print(f"    A_W={rec['A_W']:.3f} cv={rec['cv_aw']:.4f} "
                  f"share={rec['share_med']:.3f} snr={rec['snr_med']:.1f}", flush=True)
        # 增量落盘：全量 cases 的顺序 + 已有结果，中断也不丢
        Path(args.out).write_text(
            json.dumps(dict(quick=True, sig_rel=SIG_REL, rows=rows),
                       ensure_ascii=False, indent=1, default=float),
            encoding="utf-8")
    print(f"\n写入 {args.out}（{len(rows)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
