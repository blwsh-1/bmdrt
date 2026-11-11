"""convert_3e.py —— EISmeasurements/3E data/ 的 23 个 xlsx → 三列 CSV。

和 convert_2e.py 的关系
----------------------
产物口径**完全对齐** `EISmeasurements/csv_2e/`：

    EISmeasurements/csv_3e/<stem>.csv     f_Hz,Zre,Zim 三列（真 Im Z）
    EISmeasurements/csv_3e/manifest.csv   前 17 列与 csv_2e/manifest.csv 同名同序
    EISmeasurements/csv_3e/README.md      转换规则与校验结果

**现状注记（审查 P2-12，二轮）**：`csv_3e/manifest.csv` 在本项目内**没有
下游消费者**——批处理入口（`bmdrt.py` / `bm_real.run`）直接读三列 CSV，
不经过 manifest。「与 2E 同口径所以可直接复用」目前只是格式对齐，
不是已接通的管道；要接 manifest 需要单独立项。manifest 新增 `skipped`
列：异常行（短行 / 数值解析失败）计数，不再静默丢弃，f_min/f_max
缩水可据此察觉。

但**源表格式和 2E 完全不同**，不能复用 convert_2e.read_sheet()：

* 2E：Gamry 导出，每个 sheet 前 97 行仪器元数据，阻抗谱在 `ZCURVE`
  段（Pt/Time/Freq/Zreal/Zimag/Zsig/Zmod/Zphz/...），13 文件 × 9 sheet。
* 3E：已经是整理好的 9 列简表，第 1 行就是表头，无元数据：

      A Frequency (Hz) | B -Phase (°) | C Time (s) | D Z (Ω) | E Z' (Ω)
      F -Z'' (Ω) | G Index | H Current (DC) (A) | I Potential (DC) (V)

  23 文件 × 6 sheet（电流密度 ±5/50/100/150/200/300 mA cm-2，HER 为负、
  OER 与 full electrolysis cell 为正），每条 67 点、100 kHz → 0.1 Hz。
  部分文件把 Ω 写成了 `?`（早期 ANSI 导出丢字符），只影响表头文字。

符号口径（和 2E 一样重要）
------------------------
`bm_real.load_spectrum` 用 `Z = Zre + 1j*Zim`，`detect_L()` 以 `Im Z > 0`
判高频感抗，所以 CSV 必须写成**真 Im Z**（容性为负、高频感抗为正）。
3E 表头写的是 `-Z''` / `-Phase`，都是取过负号的量，因此

    Im Z = -F 列       φ = -B 列

不翻号会把 Nyquist 上下颠倒、L 判反。

与 2E 的判定方式有一处必要差别：2E 的 `ZCURVE` 列名不带符号提示，
只能靠相位列比大小选约定（`err_same - err_flip >= 5°` 才翻）；3E 的
列名**明确写了 `-Z''`**，符号是已知量，所以这里以列名为准，相位自检
降级为一致性校验——按两种约定分别算相位残差，若所选约定的残差明显
（≥5°）大于另一约定，说明列名与数据矛盾，直接报错。这一步不能省：
`3E-1 HER -300 mA cm-2` 整谱相位只有 ±10°，平均残差 4.32°，照搬 2E
的 5° 裕度会把它误判成"不用翻号"，高频感抗就翻反了。另用
`|Z| = hypot(Zre, Zim)` 做与符号无关的幅值自检。

单位
----
Freq：Hz；Z：Ω；均不缩放（fscale=1, zscale=1），面积未归一化。
电流密度只在文件名 / manifest 里，不参与拟合；Idc(A) / Vdc(V) 取列均值
记在 manifest 末尾，仅供核对工况。

实现说明
--------
当前 Python 环境没有 openpyxl，且 pip 源不可用；xlsx 本质是 zip+XML，
所以这里用标准库 `zipfile` + `xml.etree.ElementTree` 直接读，零第三方依赖，
在任何 Python 3 上都能复现。数字在 XML 里都是文本，统一显式 float()。

用法：
    python convert_3e.py            # 只转换，不拟合
    python convert_3e.py --check    # 只报告，不写文件
"""

from __future__ import annotations

import csv
import math
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "EISmeasurements" / "3E data"
DST = ROOT / "EISmeasurements" / "csv_3e"

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

# 2E manifest 的列（同名同序，方便 csv.DictReader 的下游脚本直接复用）
BASE_KEYS = ("stem", "src", "sheet", "j_mA_cm2", "n", "skipped", "f_min",
             "f_max",
             "decades", "zre_min", "zre_max", "zim_min", "zim_max", "zmax",
             "imag_flipped", "phase_err_deg", "mod_rel_err", "csv")
EXTRA_KEYS = ("electrode", "exp", "idc_A_mean", "vdc_V_mean")


# --------------------------------------------------------------------------
# 极简 xlsx 读取（标准库，只读值，不读样式/公式）
# --------------------------------------------------------------------------
def _col_index(ref: str) -> int:
    """'AB12' -> 27（0 基）。"""
    s = re.match(r"([A-Z]+)", ref).group(1)
    n = 0
    for ch in s:
        n = n * 26 + ord(ch) - 64
    return n - 1


def _sheet_paths(z: zipfile.ZipFile):
    names = set(z.namelist())
    rels = {}
    if "xl/_rels/workbook.xml.rels" in names:
        for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")):
            rels[r.get("Id")] = r.get("Target")
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    out = []
    for sh in wb.find(NS + "sheets"):
        tgt = rels.get(sh.get(RNS + "id"), "worksheets/sheet1.xml")
        tgt = tgt.lstrip("/")
        path = tgt if tgt.startswith("xl/") else "xl/" + tgt
        if path not in names:  # 极少数导出把 target 写成 xl/worksheets/...
            cand = [n for n in names if n.endswith(tgt.split("/")[-1])]
            if not cand:
                raise ValueError(f"找不到 sheet XML：{path}")
            path = cand[0]
        out.append((sh.get("name"), path))
    return out


def _shared_strings(z: zipfile.ZipFile) -> list[str]:
    p = "xl/sharedStrings.xml"
    if p not in z.namelist():
        return []
    return ["".join(t.text or "" for t in si.iter(NS + "t"))
            for si in ET.fromstring(z.read(p)).findall(NS + "si")]


def read_xlsx(path: Path):
    """-> [(sheet_name, rows)]，rows 是等宽的 list[list[str|None]]。"""
    with zipfile.ZipFile(path) as z:
        ss = _shared_strings(z)
        out = []
        for name, sp in _sheet_paths(z):
            root = ET.fromstring(z.read(sp))
            rows = []
            for row in root.find(NS + "sheetData"):
                cells = {}
                for c in row.findall(NS + "c"):
                    v = c.find(NS + "v")
                    t = c.get("t")
                    if t == "s" and v is not None:
                        val = ss[int(v.text)]
                    elif t == "inlineStr":
                        val = "".join(x.text or "" for x in c.iter(NS + "t"))
                    elif v is not None:
                        val = v.text
                    else:
                        val = None
                    cells[_col_index(c.get("r"))] = val
                width = max(cells) + 1 if cells else 0
                rows.append([cells.get(k) for k in range(width)])
            out.append((name, rows))
        return out


# --------------------------------------------------------------------------
# 表头 / 数值 / 文件名
# --------------------------------------------------------------------------
def _header_kind(v) -> tuple[str, int] | None:
    """-> (语义, 符号)：符号 -1 表示列名带负号（-Z'' / -Phase）。

    Ω 在部分文件里退化成 '?'，所以先把单位与非 ASCII 字符都剔掉再匹配。
    """
    h = re.sub(r"[^a-z0-9()'\-]+", "", str(v or "").strip().lower())
    if h.startswith("frequency"):
        return "freq", 1
    if h.startswith("time"):
        return "time", 1
    if h.startswith("index"):
        return "index", 1
    if h.startswith("current"):
        return "idc", 1
    if h.startswith("potential"):
        return "vdc", 1
    if h.startswith("-phase"):
        return "phase", -1
    if h.startswith("phase"):
        return "phase", 1
    if h.startswith("z()"):           # "Z (Ω)" = |Z|
        return "mod", 1
    if h.startswith("z'()"):          # "Z' (Ω)" = Zre
        return "re", 1
    if h.startswith("-z''"):          # "-Z'' (Ω)" = -Im Z
        return "im", -1
    if h.startswith("z''"):           # "Z'' (Ω)" = Im Z
        return "im", 1
    return None


def _num(v) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _slug(name: str) -> str:
    """与 convert_2e._slug 完全相同，保证 2E/3E 命名风格一致。"""
    s = name.replace(" EIS data", "").strip()
    s = re.sub(r"\s*experiment\s*(\d+)", r"_exp\1", s, flags=re.I)
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------
# 单条谱
# --------------------------------------------------------------------------
def read_sheet(sheet_name: str, rows: list[list]) -> dict:
    """从 3E 的 9 列简表里抽出阻抗谱 + 符号/幅值自检结果。"""
    if not rows:
        raise ValueError("空 sheet")

    col, sign = {}, {}
    for j, v in enumerate(rows[0]):
        k = _header_kind(v)
        if k:
            kind, s = k
            col.setdefault(kind, j)
            sign[kind] = s
    need = ("freq", "phase", "mod", "re", "im")
    missing = [k for k in need if k not in col]
    if missing:
        raise ValueError(f"缺列 {missing}，表头={rows[0][:9]}")
    last = max(col.values())

    rec = []
    n_skip = 0                  # P2-12（二轮审查）：异常行计数，不再静默丢弃
    for row in rows[1:]:
        if len(row) <= last:
            n_skip += 1
            continue
        f = _num(row[col["freq"]])
        zr = _num(row[col["re"]])
        zi_raw = _num(row[col["im"]])
        zm = _num(row[col["mod"]])
        ph_raw = _num(row[col["phase"]])
        idc = _num(row[col["idc"]]) if "idc" in col else None
        vdc = _num(row[col["vdc"]]) if "vdc" in col else None
        if None in (f, zr, zi_raw, zm, ph_raw) or f <= 0 or zm <= 0:
            n_skip += 1
            continue
        # 列名带 '-' 时，列值是 -φ / -Im Z，还原成真值
        ph = sign["phase"] * ph_raw
        rec.append((f, zr, zi_raw, zm, ph, idc, vdc))
    if len(rec) < 8:
        raise ValueError(f"只有 {len(rec)} 行有效数据")

    # ---- 符号以列名为准，相位自检做一致性校验 ----
    # err_same：假设虚部列就是 Im Z；err_flip：假设虚部列是 -Im Z。
    # φ 参考来自仪器相位列（已按列名还原成真 φ），与虚部列相互独立。
    err_same = err_flip = err_mod = 0.0
    for f, zr, zi_raw, zm, ph, *_ in rec:
        err_same += abs(_wrap180(math.degrees(math.atan2(zi_raw, zr)) - ph))
        err_flip += abs(_wrap180(math.degrees(math.atan2(-zi_raw, zr)) - ph))
        err_mod += abs(math.hypot(zr, zi_raw) - zm) / max(zm, 1e-12)
    n = len(rec)
    err_same, err_flip, err_mod = err_same / n, err_flip / n, err_mod / n
    flip = sign["im"] < 0                     # 列名 "-Z''" → 需要翻号
    phase_err = err_flip if flip else err_same
    other_err = err_same if flip else err_flip
    # 小相位谱两种残差可能都 <5°（如 3E-1 HER -300 只有 4.32°），
    # 此时无法靠相位分辨，以列名为准即可；只有数据**明确反对**列名才报错。
    if phase_err - other_err >= 5.0:
        raise ValueError(f"列名约定(flip={flip})与相位自检矛盾："
                         f"所选残差 {phase_err:.2f}° > 另一约定 {other_err:.2f}°")
    if phase_err > 1.0 or err_mod > 0.01:
        raise ValueError(f"符号/幅值自检失败：相位差 {phase_err:.2f}°、"
                         f"|Z| 相对差 {err_mod:.2%}")

    data = [(f, zr, -zi_raw if flip else zi_raw) for f, zr, zi_raw, *_ in rec]
    # 物理 sanity（宽松）：最高频应为引线感抗 Im>0，且谱内应有容性点 Im<0。
    # 不要求最低频一定容性——大电流下低频常出现弱感抗弧，属真实数据。
    hi = max(data, key=lambda x: x[0])
    zim_all = [x[2] for x in data]
    physical = hi[2] > 0 and min(zim_all) < 0

    idcs = [x[5] for x in rec if x[5] is not None]
    vdcs = [x[6] for x in rec if x[6] is not None]
    return dict(data=data, flip=flip, phase_err=phase_err, mod_err=err_mod,
                n=n, sheet=sheet_name, physical=physical, skipped=n_skip,
                idc=sum(idcs) / len(idcs) if idcs else float("nan"),
                vdc=sum(vdcs) / len(vdcs) if vdcs else float("nan"))


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> int:
    check_only = "--check" in sys.argv
    files = sorted(SRC.glob("*.xlsx"))
    if len(files) != 23:
        print(f"[warn] `3E data/` 找到 {len(files)} 个 xlsx，预期 23 个")
    if not files:
        print(f"[error] 源目录为空：{SRC}")
        return 1
    if not check_only:
        DST.mkdir(parents=True, exist_ok=True)

    manifest, problems, warnings = [], [], []
    flips = 0
    stems = set()
    for p in files:
        base = _slug(p.stem)
        low = p.stem.lower()
        electrode = ("HER" if re.search(r"\bher\b", low) else
                     "OER" if re.search(r"\boer\b", low) else
                     "full_cell" if "full" in low else "unknown")
        m_exp = re.search(r"experiment\s*(\d+)", p.stem, re.I)
        exp = m_exp.group(1) if m_exp else ""
        for sheet_name, rows in read_xlsx(p):
            m = re.search(r"(-?\d+(?:\.\d+)?)\s*mA", sheet_name, re.I)
            j = float(m.group(1)) if m else float("nan")
            tag = f"{int(abs(j)):04d}mA" if j == j else "unknown"
            stem = f"{base}_{tag}"
            if stem in stems:
                problems.append((p.name, sheet_name, f"stem 重复：{stem}"))
                continue
            stems.add(stem)
            try:
                r = read_sheet(sheet_name, rows)
            except Exception as e:
                problems.append((p.name, sheet_name, str(e)))
                continue
            if not r["physical"]:
                warnings.append((stem, "高频 Im Z ≤0 或全谱无容性点，请人工看一眼"))
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
            row = dict(
                stem=stem, src=p.name, sheet=sheet_name, j_mA_cm2=f"{j:g}",
                n=len(data), f_min=f"{min(f):.6g}", f_max=f"{max(f):.6g}",
                decades=f"{math.log10(max(f) / min(f)):.2f}",
                zre_min=f"{min(zr):.4g}", zre_max=f"{max(zr):.4g}",
                zim_min=f"{min(zi):.4g}", zim_max=f"{max(zi):.4g}",
                zmax=f"{max(math.hypot(a, b) for a, b in zip(zr, zi)):.4g}",
                imag_flipped=int(r["flip"]),
                phase_err_deg=f"{r['phase_err']:.3g}",
                mod_rel_err=f"{r['mod_err']:.2e}",
                skipped=int(r.get("skipped", 0)),
                csv=out.name,
                electrode=electrode, exp=exp,
                idc_A_mean=f"{r['idc']:.6g}", vdc_V_mean=f"{r['vdc']:.6g}")
            manifest.append(row)

    def _mx(key):
        return max(float(x[key]) for x in manifest)

    print(f"[convert_3e] {len(files)} 个文件 → {len(manifest)} 条谱"
          f"（每条 {min(int(x['n']) for x in manifest)}–"
          f"{max(int(x['n']) for x in manifest)} 点，"
          f"{min(float(x['f_min']) for x in manifest):g}–"
          f"{max(float(x['f_max']) for x in manifest):g} Hz）")
    print(f"  虚部需翻号的 sheet：{flips} / {len(manifest)}"
          f"（源表是 -Z'' 约定，已统一翻成真 Im Z）")
    print(f"  相位自检最大误差：{_mx('phase_err_deg'):.4g}°"
          f"；|Z| 最大相对误差：{_mx('mod_rel_err'):.2e}")
    if warnings:
        print(f"  {len(warnings)} 条物理符号提示：")
        for x in warnings[:10]:
            print("   ", x)
    if problems:
        print(f"  ** {len(problems)} 个 sheet 转换失败 **")
        for x in problems:
            print("   ", x)
    if check_only:
        # P2-11（二轮审查）：预检发现 problems 必须红灯，与正常模式同口径
        # ——恒 return 0 让预检门禁形同虚设。
        return 1 if problems else 0

    keys = list(BASE_KEYS) + list(EXTRA_KEYS)
    with (DST / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(manifest)

    n_her = sum(x["electrode"] == "HER" for x in manifest)
    n_oer = sum(x["electrode"] == "OER" for x in manifest)
    n_full = sum(x["electrode"] == "full_cell" for x in manifest)
    z_lo = min(float(x["zmax"]) for x in manifest)
    z_hi = max(float(x["zmax"]) for x in manifest)
    (DST / "README.md").write_text(
        "# 3E 数据格式转换说明\n\n"
        f"- 源：`EISmeasurements/3E data/*.xlsx` 共 {len(files)} 个，"
        "每个 6 个 sheet（电流密度 ±5/50/100/150/200/300 mA cm-2；"
        "HER 为负、OER 与 full electrolysis cell 为正）。\n"
        f"- 产物：{len(manifest)} 个 CSV，三列 `f_Hz,Zre,Zim`，"
        "直接给 `bmdrt.py auto` 用；"
        f"每条 {min(int(x['n']) for x in manifest)} 点、"
        "100 kHz → 0.1 Hz（6 个数量级）。\n"
        f"- 电极分布：HER {n_her} 条、OER {n_oer} 条、"
        f"full cell {n_full} 条。\n\n"
        "## 与 2E 的差别\n\n"
        "- 2E 是 Gamry 原始导出：每个 sheet 前 97 行仪器元数据，"
        "阻抗谱在 `ZCURVE` 段；3E 已是整理好的 9 列简表（第 1 行即表头）：\n"
        "  `Frequency(Hz) | -Phase(°) | Time(s) | Z(Ω) | Z'(Ω) | "
        "-Z''(Ω) | Index | Idc(A) | Vdc(V)`。\n"
        "- 部分 3E 文件的 Ω 在表头退化为 `?`（早期 ANSI 导出），"
        "只影响单位文字，数值列不受影响。\n"
        f"- 3E 的 |Z| 峰值范围 {z_lo:.3g}–{z_hi:.3g} Ω，明显大于 2E"
        "（约 0.02–0.2 Ω），属三电极半电池 / 全电解池与两电极体系的"
        "差别，未做任何缩放。\n"
        "- 本脚本用标准库 `zipfile`+`ElementTree` 读 xlsx，零第三方依赖"
        "（当前环境无 openpyxl，pip 源不可用），复现不需要装包。\n\n"
        "## 符号与自检\n\n"
        "- **虚部符号**：列名 `-Z''` / `-Phase` 都是取过负号的量，"
        "真 `Im Z = -F 列`、`φ = -B 列`，统一写成**真 Im Z**"
        "（容性为负、高频感抗为正），与 `csv_2e` 完全一致。\n"
        "- 判定口径与 2E 略有差别：2E 靠相位列比大小选约定（差值 ≥5° 才翻），"
        "3E 列名已明确写出 `-Z''`，故以列名为准，相位自检只做一致性校验"
        "（若所选约定残差明显大于另一约定则报错）。原因是 "
        "`3E-1 HER -300 mA cm-2` 整谱相位仅 ±10°、残差 4.32°，"
        "照搬 2E 的 5° 裕度会把高频感抗翻反。\n"
        f"- 自检结果：{flips}/{len(manifest)} 个 sheet 需要翻号；"
        f"相位最大误差 {_mx('phase_err_deg'):.4g}°，"
        f"|Z| 最大相对误差 {_mx('mod_rel_err'):.2e}。\n"
        "- 单位：Hz / Ω，无缩放（fscale=1, zscale=1），面积未归一化；"
        "电流密度只在文件名 / manifest 里，不参与拟合。\n"
        "- `manifest.csv` 前 17 列与 `csv_2e/manifest.csv` 同名同序；"
        "末尾 4 列（`electrode`, `exp`, `idc_A_mean`, `vdc_V_mean`）"
        "是 3E 特有工况，按列名读取的下游脚本不受影响。\n"
        "- 复现：`python convert_3e.py`（`--check` 只报告不写文件）\n",
        encoding="utf-8")
    print(f"  写出：{DST/'manifest.csv'}、{DST/'README.md'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
