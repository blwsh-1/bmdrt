"""drt.py 命令行模式的兼容读取器（只依赖 numpy）。

历史遗留：`drt.py` 的 ``__main__`` 分支写的是
``from fit_store import load_spectrum_csv``，但该模块没有随代码保留，
导致 ``python drt.py data.csv`` 直接 ImportError。这里补一个最小实现，
让 DRT 模块可以独立跑，不必为此拉起 jax / numpyro 那套贝叶斯栈。

主流程 `bmdrt.py auto` 用的是 `bm_real.load_spectrum()`，与本文件无关；
两者的解析口径保持一致（分隔符、表头、|Z|/phase、符号、去重）。

    f, Z = load_spectrum_csv("data.csv")   # f: 升序 Hz, Z: 复数（容性 Im<0）
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_NUM = re.compile(r"[-+0-9.eE]+")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.strip().lower())


def _pick(names, exact, contains):
    for i, n in enumerate(names):
        if n in exact:
            return i
    for i, n in enumerate(names):
        if any(c in n for c in contains):
            return i
    return None


def load_spectrum_csv(path, fscale: float | None = None, zscale: float = 1.0):
    """读 CSV / TSV / 空白分隔的 EIS 数据，返回 ``(f, Z)``。

    * 带表头：识别 ``f, Zre, Zim`` 或 ``|Z|, phase`` 两组列；
    * 无表头：要求 3 列，顺序固定 ``f, Zre, Zim``；
    * 相位超过 7 时按「度」转弧度；表头含 kHz/MHz 时自动换算频率；
    * 过滤非有限值与 ``f<=0``，按频率升序，重复频率点取平均。
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"找不到文件：{path}")
    raw = p.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    rows = [ln for ln in raw
            if ln.strip() and not ln.lstrip().startswith(("#", "%", "!"))]
    if not rows:
        raise SystemExit("文件里没有数据行")

    best, best_n = None, 0
    for d in (",", "\t", ";", None):
        cnt = [len(ln.split(d)) for ln in rows[:20]]
        if min(cnt) >= 2 and max(cnt) == min(cnt) and cnt[0] > best_n:
            best, best_n = d, cnt[0]
    if best is None:
        raise SystemExit("无法识别列分隔符（试过 , \\t ; 空白）")

    def split(ln):
        return [x.strip() for x in ln.split(best)]

    head = split(rows[0])
    has_head = any(not _NUM.fullmatch(h or "x") for h in head)
    names = [_norm(h) for h in head] if has_head else []
    body = rows[1:] if has_head else rows

    data = []
    for ln in body:
        try:
            data.append([float(x) for x in split(ln)])
        except ValueError:
            continue
    A = np.asarray(data, float)
    if A.ndim != 2 or A.shape[0] < 5:
        raise SystemExit(f"有效数据行太少（{A.shape[0] if A.ndim == 2 else 0} 行）")

    if names:
        i_f = _pick(names, ("f", "freq", "frequency", "hz", "fhz"), ("freq",))
        i_re = _pick(names, ("zre", "zreal", "re", "real", "zr", "z1"), ("real", "zre"))
        i_im = _pick(names, ("zim", "zimag", "im", "imag", "zi", "z2"), ("imag", "zim"))
        i_mag = _pick(names, ("z", "zmag", "zabs", "absz", "mag", "amplitude"), ("mag",))
        i_ph = _pick(names, ("phase", "theta", "angle", "ang", "phz"), ("phase",))
    else:
        i_f, i_re, i_im, i_mag, i_ph = 0, 1, 2, None, None
        if A.shape[1] < 3:
            raise SystemExit("无表头时需要 3 列：f, Zre, Zim")

    if i_f is None or i_f >= A.shape[1]:
        raise SystemExit(f"找不到频率列，表头是：{head}")
    f = A[:, i_f].astype(float)

    if i_re is not None and i_im is not None \
            and i_re < A.shape[1] and i_im < A.shape[1]:
        Z = A[:, i_re] + 1j * A[:, i_im]
    elif i_mag is not None and i_ph is not None \
            and i_mag < A.shape[1] and i_ph < A.shape[1]:
        mag, ph = A[:, i_mag], A[:, i_ph].astype(float)
        if not any("rad" in h for h in names) and np.nanmax(np.abs(ph)) > 7.0:
            ph = np.radians(ph)
        Z = mag * np.exp(1j * ph)
    else:
        raise SystemExit(f"找不到 (Zre,Zim) 或 (|Z|,phase) 组合，表头是：{head}")

    if fscale is None:
        h = "".join(names)
        fscale = 1e3 if "khz" in h else (1e6 if "mhz" in h else 1.0)
    f, Z = f * fscale, Z * zscale

    ok = np.isfinite(f) & np.isfinite(Z) & (f > 0)
    f, Z = f[ok], Z[ok]
    if f.size < 5:
        raise SystemExit("过滤非有限值后剩余点数不足 5")
    order = np.argsort(f)
    f, Z = f[order], Z[order]
    fu, inv, cnt = np.unique(f, return_inverse=True, return_counts=True)
    if np.any(cnt > 1):
        Zm = np.zeros(fu.size, complex)
        for i in range(fu.size):
            Zm[i] = Z[inv == i].mean()
        f, Z = fu, Zm
    return f, Z


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("用法：python fit_store.py <data.csv>")
    f, Z = load_spectrum_csv(sys.argv[1])
    print(f"{f.size} 点  {f.min():.4g} – {f.max():.4g} Hz  "
          f"({np.log10(f.max() / f.min()):.2f} decade)")
