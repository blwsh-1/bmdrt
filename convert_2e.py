"""convert_2e.py —— EISmeasurements/ 里 13 个 "2E" Gamry xlsx → bmdrt 能吃的三列 CSV。

为什么要单独一层转换
--------------------
Gamry 的 .xlsx 不是数据表，是**实验记录**：每个 sheet 前面 97 行是仪器元数据
（EXPLAIN / TAG / OCVCURVE 的 40 行开路电压稳定曲线 …），真正的阻抗谱在
`ZCURVE` 段，表头长这样（B 列起，A 列全空）：

    Pt | Time | Freq | Zreal | Zimag | Zsig | Zmod | Zphz | Idc | Vdc | IERange
    #  | s    | Hz   | ohm   | ohm   | V    | ohm  | deg  | A   | V   | #

13 个文件 × 9 个 sheet（电流密度 5/50/100/150/200/350/500/670/750 mA cm-2）
= 117 条谱，每条 56 点、100 kHz → 1 Hz。**单元格里的数字多半存成文本**，
必须显式 float()，不能交给 pandas 猜 dtype。

两个必须处理的口径问题
----------------------
1. **虚部符号**。EIS 导出有 "真 Im Z"（容性为负）和 "-Z''"（容性为正）两派。
   这里用同一行自带的 `Zmod` / `Zphz` 做自检：分别按
   `atan2(+Zimag, Zreal)` 与 `atan2(-Zimag, Zreal)` 反算相位，谁和 `Zphz`
   一致就用谁的约定；若需要翻号，写 CSV 时统一翻成 **真 Im Z**，因为
   `bm_real.load_spectrum` 用 `Z = Zre + 1j*Zim`，`detect_L()` 又用
   `Im Z > 0` 判高频感抗。翻错会让 Nyquist 上下颠倒、L 判反。
2. **单位**。Freq 列单位 Hz、Z 列单位 ohm，都不乘缩放因子（fscale=1,
   zscale=1）。电流密度只在文件名/manifest 里，不参与拟合。

产物
----
    EISmeasurements/csv_2e/<stem>.csv      f_Hz,Zre,Zim 三列，无表头歧义
    EISmeasurements/csv_2e/manifest.csv    117 行的对照表（来源、电流、点数、频段、自检）
    EISmeasurements/csv_2e/README.md       转换规则与校验结果

用法：
    python convert_2e.py                # 只转换，不拟合
    python convert_2e.py --check        # 只报告，不写文件
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "EISmeasurements"
DST = SRC / "csv_2e"

# 文件名里的空格/特殊字符一律换成下划线，避免 Windows 路径与命令行转义问题
def _slug(name: str) -> str:
    s = name.replace(" EIS data", "").strip()
    s = re.sub(r"\s*experiment\s*(\d+)", r"_exp\1", s, flags=re.I)
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _num(v) -> float | None:
    """Gamry 把数字存成文本；'...........' 这类占位符要当缺失跳过。"""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def read_sheet(ws) -> dict:
    """从单个 sheet 里抽出 ZCURVE 数据段 + 符号自检结果。"""
    rows = list(ws.iter_rows(values_only=True))
    hdr_i = None
    for i, row in enumerate(rows):
        if len(row) > 5 and str(row[1]).strip() == "Pt" \
                and str(row[3]).strip() == "Freq":
            hdr_i = i
            break
    if hdr_i is None:
        raise ValueError(f"sheet {ws.title!r} 里找不到 ZCURVE 表头（Pt/Freq）")

    col = {str(v).strip(): j for j, v in enumerate(rows[hdr_i]) if v is not None}
    need = ("Freq", "Zreal", "Zimag", "Zmod", "Zphz")
    missing = [k for k in need if k not in col]
    if missing:
        raise ValueError(f"缺少列 {missing}，表头={rows[hdr_i][:12]}")

    rec = []
    for row in rows[hdr_i + 1:]:
        if len(row) <= max(col.values()):
            continue
        f = _num(row[col["Freq"]])
        zr = _num(row[col["Zreal"]])
        zi = _num(row[col["Zimag"]])
        zm = _num(row[col["Zmod"]])
        zp = _num(row[col["Zphz"]])
        # 单位行 "#/s/Hz/ohm/..." 和空行都在这里被跳过
        if None in (f, zr, zi, zm, zp) or f <= 0:
            continue
        rec.append((f, zr, zi, zm, zp))
    if len(rec) < 8:
        raise ValueError(f"只有 {len(rec)} 行有效数据")

    # ---- 虚部符号自检：用仪器自己给的 |Z| 和相位反算 ----
    err_same = err_flip = err_mod = 0.0
    for f, zr, zi, zm, zp in rec:
        err_same += abs(math.degrees(math.atan2(zi, zr)) - zp)
        err_flip += abs(math.degrees(math.atan2(-zi, zr)) - zp)
        err_mod += abs(math.hypot(zr, zi) - zm) / max(zm, 1e-12)
    n = len(rec)
    err_same, err_flip, err_mod = err_same / n, err_flip / n, err_mod / n
    # 只有翻号优势超过明确裕度才翻：高频近纯阻时两个误差都≈0，纯大小比较
    # 会由噪声随机决定翻不翻，可能把感抗/容抗尾翻反，直接影响 detect_L
    # （审查 P2-8）。
    flip = (err_same - err_flip) >= 5.0
    phase_err = min(err_same, err_flip)
    if phase_err > 1.0 or err_mod > 0.01:
        raise ValueError(f"符号/幅值自检失败：相位差 {phase_err:.2f}°、"
                         f"|Z| 相对差 {err_mod:.2%}")

    data = [(f, zr, -zi if flip else zi) for f, zr, zi, zm, zp in rec]
    return dict(data=data, flip=flip, phase_err=phase_err, mod_err=err_mod,
                n=n, sheet=ws.title)


def main() -> int:
    check_only = "--check" in sys.argv
    files = sorted(SRC.glob("2E*.xlsx"))
    if len(files) != 13:
        print(f"[warn] 顶层找到 {len(files)} 个 2E*.xlsx，预期 13 个")
    if not check_only:
        DST.mkdir(parents=True, exist_ok=True)

    manifest, problems = [], []
    flips = 0
    for p in files:
        wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
        base = _slug(p.stem)
        for ws in wb.worksheets:
            m = re.search(r"(\d+(?:\.\d+)?)\s*mA", ws.title, re.I)
            j = float(m.group(1)) if m else float("nan")
            tag = f"{int(j):04d}mA" if j == j else "unknown"
            stem = f"{base}_{tag}"
            try:
                r = read_sheet(ws)
            except Exception as e:
                problems.append((p.name, ws.title, str(e)))
                continue
            data = r["data"]
            f = [x[0] for x in data]
            zr = [x[1] for x in data]
            zi = [x[2] for x in data]
            if r["flip"]:
                flips += 1
            out = DST / f"{stem}.csv"
            if not check_only:
                with out.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["f_Hz", "Zre", "Zim"])
                    for a, b, c in data:
                        w.writerow([f"{a:.10g}", f"{b:.10g}", f"{c:.10g}"])
            manifest.append(dict(
                stem=stem, src=p.name, sheet=ws.title, j_mA_cm2=f"{j:g}",
                n=len(data), f_min=f"{min(f):.6g}", f_max=f"{max(f):.6g}",
                decades=f"{math.log10(max(f) / min(f)):.2f}",
                zre_min=f"{min(zr):.4g}", zre_max=f"{max(zr):.4g}",
                zim_min=f"{min(zi):.4g}", zim_max=f"{max(zi):.4g}",
                zmax=f"{max(math.hypot(a, b) for a, b in zip(zr, zi)):.4g}",
                imag_flipped=int(r["flip"]),
                phase_err_deg=f"{r['phase_err']:.3g}",
                mod_rel_err=f"{r['mod_err']:.2e}",
                csv=out.name))
        wb.close()

    print(f"[convert_2e] {len(files)} 个文件 → {len(manifest)} 条谱"
          f"（每条 {min(int(x['n']) for x in manifest)}–"
          f"{max(int(x['n']) for x in manifest)} 点）")
    print(f"  虚部需翻号的 sheet：{flips} / {len(manifest)}"
          f"（{'0 说明原表就是真 Im Z' if flips == 0 else '已统一翻成真 Im Z'}）")
    print(f"  相位自检最大误差：{max(float(x['phase_err_deg']) for x in manifest):.4g}°"
          f"；|Z| 最大相对误差：{max(float(x['mod_rel_err']) for x in manifest):.2e}")
    if problems:
        print(f"  ** {len(problems)} 个 sheet 转换失败 **")
        for x in problems:
            print("   ", x)
    if check_only:
        return 0

    keys = list(manifest[0])
    with (DST / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(manifest)

    (DST / "README.md").write_text(
        "# 2E 数据格式转换说明\n\n"
        f"- 源：`EISmeasurements/2E*.xlsx` 共 {len(files)} 个，每个 9 个 sheet"
        "（电流密度 5…750 mA cm-2）。\n"
        f"- 产物：{len(manifest)} 个 CSV，三列 `f_Hz,Zre,Zim`，"
        "直接给 `bmdrt.py auto` 用。\n"
        "- Gamry 每个 sheet 的前 97 行是仪器元数据，阻抗谱在 `ZCURVE` 段；"
        "数字存成文本，已显式转 float。\n"
        "- **虚部符号**：用同行 `Zmod`/`Zphz` 自检，统一写成真 Im Z"
        "（容性为负、高频感抗为正）。`imag_flipped=1` 表示原表是 -Z'' 约定。\n"
        f"- 自检：相位最大误差 "
        f"{max(float(x['phase_err_deg']) for x in manifest):.4g}°，"
        f"|Z| 最大相对误差 "
        f"{max(float(x['mod_rel_err']) for x in manifest):.2e}。\n"
        "- 单位：Hz / Ω，无缩放；面积未归一化（AREA=1 cm2，仪器已按此给出）。\n"
        "- `manifest.csv` 是每个 stem 与源文件 / sheet / 电流密度的对照表。\n"
        "- 复现：`python convert_2e.py`\n",
        encoding="utf-8")
    print(f"  写出：{DST/'manifest.csv'}、{DST/'README.md'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
