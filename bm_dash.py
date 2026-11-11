"""bm_dash.py —— 单文件交互式分析面板（自包含 HTML，零运行时依赖）。

为什么不用 Plotly
-----------------
1. 这台机器（以及组里其他机器）是离线/内网环境，plotly 的 bundle 要么走 CDN
   打不开，要么内联 3 MB 且版本锁定；而我们要内联的只是一个 payload（~200 KB）。
2. 面板要**导出 SVG 原件**给论文排版用。Plotly 的 SVG 是渲染结果拼出来的，
   图层、裁剪、字体属性都带着 plotly 的痕迹；自己画 SVG 则每个元素都是
   我们写的，序列化出来就是干净的矢量文件。
3. 出版图那条线（`bm_figure.py`）本来就要用 matplotlib；交互侧再引一个
   渲染引擎，等于三套坐标轴实现，迟早对不齐。
4. 交互只需要「缩放 / 悬停读数 / 拖拽排序 / 切换显隐」，不需要三维、
   不需要 30 种图型。手写 400 行比引入 3 MB 更可控。

纪律
----
* **只读** `AnalysisResult.payload()`，不在这里重算任何物理量。
* 判定文案全部来自 payload（`verdict` / `checks`），前端不自己造「通过」。
* 布局与主题是**可导出的 JSON**（`localStorage` 只是缓存），
  换一条谱可以套同一套版式；导出 PDF 用 `bm_figure.py` 读同一份 spec。

用法
----
    python bm_dash.py <sample_dir>                 # 写 results/<stem>/dashboard.html
    python bm_dash.py <sample_dir> --bundle        # 同时写 data/ metadata/ manifest
    python bm_dash.py <sample_dir> --open          # 写完后用系统浏览器打开
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

import bm_result as R

# --------------------------------------------------------------------------
# 主题预设：只描述排版语言，不宣称期刊合规
# --------------------------------------------------------------------------
THEMES = {
    "minimal": dict(label="Minimal", plotFont="Inter, Helvetica, Arial",
                    fontSize=11, axisLw=1.0, tickLen=4, grid=False,
                    data="#333333", fit="#1f6fb4", band="#1f6fb4",
                    pred="#8c8c8c", drt="#7b3fa0", reC="#1f6fb4",
                    imC="#d1495b", bg="#ffffff", fg="#1c1c1c", sub="#6b6b6b",
                    line="#e4e4e4"),
    "nature": dict(label="Nature-like", plotFont="Arial, Helvetica",
                   fontSize=10, axisLw=0.9, tickLen=3.5, grid=False,
                   data="#222222", fit="#0b6fa4", band="#0b6fa4",
                   pred="#9a9a9a", drt="#6a3d9a", reC="#0b6fa4",
                   imC="#b2182b", bg="#ffffff", fg="#1c1c1c", sub="#6b6b6b",
                   line="#e6e6e6"),
    "acs": dict(label="ACS-like", plotFont="Arial, Helvetica",
                fontSize=10, axisLw=1.0, tickLen=4, grid=False,
                data="#1a1a1a", fit="#0055a4", band="#0055a4",
                pred="#8f8f8f", drt="#7a3b8f", reC="#0055a4",
                imC="#c0392b", bg="#ffffff", fg="#1c1c1c", sub="#6b6b6b",
                line="#e6e6e6"),
    "angew": dict(label="Angewandte-like", plotFont="Arial, Helvetica",
                  fontSize=10, axisLw=1.15, tickLen=4.5, grid=True,
                  data="#111111", fit="#c0392b", band="#c0392b",
                  pred="#999999", drt="#1f4e79", reC="#c0392b",
                  imC="#1f4e79", bg="#ffffff", fg="#1c1c1c", sub="#6b6b6b",
                  line="#ededed"),
    "presentation": dict(label="Presentation", plotFont="Arial, Helvetica",
                         fontSize=15, axisLw=1.6, tickLen=6, grid=True,
                         data="#111111", fit="#d94801", band="#d94801",
                         pred="#888888", drt="#5b2c83", reC="#d94801",
                         imC="#2171b5", bg="#ffffff", fg="#1c1c1c",
                         sub="#6b6b6b", line="#ededed"),
}

# 默认布局：Nyquist 占最大面积（第一眼判拟合好坏的就是它），
# DRT 与之同高空一级对照；残差与判定并排；表格类到底部通栏。
DEFAULT_PANELS = [
    # Nyquist 跨两行：它是第一眼判「拟合好不好」的图，等比例坐标下必须有
    # 近方的画布，否则要么字太小、要么四周留白。
    dict(id="fit", span=6, rows=2, visible=True),
    dict(id="drt", span=6, rows=1, visible=True),
    dict(id="resid", span=6, rows=1, visible=True),
    dict(id="notes", span=12, rows=1, visible=True),
    dict(id="model", span=12, rows=1, visible=True),
]

# 默认布局也导出成模块常量：面板生成、契约测试、以及"另存为模板"都读它，
# 免得默认值散在 main() 里被改一处、漏一处。
DEFAULT_LAYOUT = dict(theme="minimal", panels=DEFAULT_PANELS,
                      version=R.SPEC_VERSION)

PANEL_META = {
    "fit": dict(title="阻抗拟合", sub="Nyquist · 数据 / 后验中位 / 区间", detail="Bode"),
    "drt": dict(title="DRT 分布", sub="后验 γ(τ) 与峰位", detail="峰参数"),
    "resid": dict(title="残差诊断", sub="标准化残差 · ±1σ / ±2σ", detail="ACF / 分段"),
    "notes": dict(title="判定与警告", sub="§14 判据逐条 · 数据自洽性", detail=None),
    "model": dict(title="模型与参数", sub="K 候选 · 参数后验 · 相关性", detail="相关矩阵"),
}


# --------------------------------------------------------------------------
def build(payload: dict, title: str, layout: dict, theme: str) -> str:
    html = _TEMPLATE
    repl = {
        "__TITLE__": _esc(title),
        "__DATA__": json.dumps(payload, ensure_ascii=False,
                               separators=(",", ":"), default=float),
        "__LAYOUT__": json.dumps(layout, ensure_ascii=False),
        "__THEMES__": json.dumps(THEMES, ensure_ascii=False),
        "__PANELMETA__": json.dumps(PANEL_META, ensure_ascii=False),
        "__THEME__": theme,
        "__VER__": R.SPEC_VERSION,
    }
    for k, v in repl.items():
        html = html.replace(k, v)
    return html


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="交互式 EIS 分析面板")
    ap.add_argument("sample", help="样品输出目录或 real_*.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--theme", default="minimal", choices=sorted(THEMES))
    ap.add_argument("--layout", default=None,
                    help="布局 JSON 文件（bm_dash 导出的那份）")
    ap.add_argument("--bundle", action="store_true",
                    help="同时写出 data/ metadata/ manifest.json")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)

    res = R.AnalysisResult.load(a.sample)
    payload = res.payload()
    root = Path(a.out) if a.out else Path("results") / payload["name"]
    root.mkdir(parents=True, exist_ok=True)

    layout = dict(theme=a.theme, panels=[dict(p) for p in DEFAULT_PANELS])
    if a.layout:
        saved = json.loads(Path(a.layout).read_text(encoding="utf-8"))
        layout = dict(theme=saved.get("theme", a.theme),
                      panels=saved.get("panels") or layout["panels"],
                      version=saved.get("version", R.SPEC_VERSION))

    html = build(payload, payload["name"], layout, layout["theme"])
    out = root / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"  {out}  ({out.stat().st_size / 1024:.0f} KB)")

    if a.bundle:
        b = res.bundle(root)
        print(f"  {b}{'/manifest.json'}  ("
              f"{len(list((b / 'data').glob('*')))} 个数据文件)")

    if a.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


# --------------------------------------------------------------------------
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · BM-DRT</title>
<style>
:root{
  --fg:#1c1c1c; --sub:#6b6b6b; --line:#e4e4e4; --bg:#ffffff; --panel:#ffffff;
  --accent:#1f6fb4; --ok:#2e7d32; --warn:#b26a00; --bad:#c0392b;
  --okbg:#eaf5ea; --warnbg:#fdf3e3; --badbg:#fbeae8;
  --radius:10px; --gap:16px;
  --ui:"Inter","Segoe UI","Microsoft YaHei",system-ui,sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--fg);
  font-family:var(--ui);font-size:14px;line-height:1.5}
body{padding:0 0 40px}
a{color:var(--accent)}
.wrap{max-width:1560px;margin:0 auto;padding:0 20px}

/* ---------- 顶栏 ---------- */
header{position:sticky;top:0;z-index:30;background:rgba(255,255,255,.92);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.hd{display:flex;align-items:center;gap:16px;padding:10px 0}
.hd h1{font-size:17px;margin:0;font-weight:650;letter-spacing:.2px}
.hd .sub{color:var(--sub);font-size:12px}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-left:auto;align-items:center}
.chip{font-size:12px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);
  background:#fafafa;white-space:nowrap}
.chip.ok{color:var(--ok);border-color:#bfe0bf;background:var(--okbg)}
.chip.warn{color:var(--warn);border-color:#f0d4a8;background:var(--warnbg)}
.chip.bad{color:var(--bad);border-color:#f0c0ba;background:var(--badbg)}
.tools{display:flex;gap:8px;align-items:center}
select,button{font:inherit;font-size:12px;padding:5px 9px;border-radius:8px;
  border:1px solid var(--line);background:#fff;color:var(--fg);cursor:pointer}
button:hover,select:hover{border-color:#c9c9c9;background:#fafafa}
button.primary{background:var(--fg);color:#fff;border-color:var(--fg)}
button.primary:hover{background:#000}

/* ---------- KPI ---------- */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));
  gap:0;border:1px solid var(--line);border-radius:var(--radius);margin:16px 0;
  overflow:hidden;background:var(--panel)}
.kpi{padding:11px 14px;border-right:1px solid var(--line)}
.kpi:last-child{border-right:0}
.kpi .k{font-size:11px;color:var(--sub);letter-spacing:.3px;text-transform:uppercase}
.kpi .v{font-size:19px;font-weight:600;font-variant-numeric:tabular-nums;
  margin-top:2px}
.kpi .t{font-size:11px;color:var(--sub);margin-top:1px}
.kpi.fail .v{color:var(--bad)} .kpi.pass .v{color:var(--ok)}
.kpi.warn .v{color:var(--warn)}

/* ---------- 网格 ---------- */
main.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:var(--gap);
  align-items:start}
.card{grid-column:span 7;border:1px solid var(--line);border-radius:var(--radius);
  background:var(--panel);overflow:hidden;transition:box-shadow .15s;
  display:flex;flex-direction:column}
.card:hover{box-shadow:0 2px 10px rgba(0,0,0,.05)}
.card.dragging{opacity:.45;outline:2px dashed var(--accent)}
.card .ch{display:flex;align-items:center;gap:8px;padding:9px 12px;
  border-bottom:1px solid var(--line);cursor:grab}
.card .ch .drag{color:#bbb;font-size:13px;letter-spacing:-2px;cursor:grab}
.card .ch h2{font-size:13.5px;margin:0;font-weight:600;white-space:nowrap}
.card .ch .s{font-size:11.5px;color:var(--sub);white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;min-width:0;flex:1}
.card .ch .acts{margin-left:auto;display:flex;gap:4px;opacity:.35;
  transition:opacity .15s;flex:0 0 auto}
.card .ch button{white-space:nowrap}
.card:hover .ch .acts{opacity:1}
.card .ch button{padding:2px 7px;font-size:11px;border-radius:6px}
.card .cb{padding:6px 10px 2px;flex:1;min-height:0}
.card .readout{font-size:11.5px;color:var(--sub);font-family:ui-monospace,
  "Cascadia Mono",Consolas,monospace;padding:4px 12px 6px;height:26px;
  font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis}
svg{display:block;width:100%;touch-action:none}
svg text{font-family:var(--plotfont);fill:var(--fg)}

/* ---------- 表格 ---------- */
table.t{width:100%;border-collapse:collapse;font-size:12.5px;
  font-variant-numeric:tabular-nums}
table.t th{text-align:left;font-weight:600;color:var(--sub);font-size:11px;
  padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
table.t td{padding:5px 8px;border-bottom:1px solid #f2f2f2;vertical-align:top}
table.t tr:last-child td{border-bottom:0}
table.t td.num{text-align:right;font-family:ui-monospace,Consolas,monospace}
tr.sel td{background:#f4f8fd;font-weight:600}
.pill{font-size:10.5px;padding:1px 7px;border-radius:999px;border:1px solid var(--line);
  white-space:nowrap}
.pill.rejected{color:var(--bad);background:var(--badbg);border-color:#f0c0ba}
.pill.selected,.pill.converged{color:var(--ok);background:var(--okbg);border-color:#bfe0bf}
.pill.degraded{color:var(--warn);background:var(--warnbg);border-color:#f0d4a8}
.pill.noted,.pill.not_compared{color:var(--sub);background:#fafafa}

/* ---------- 判定卡 ---------- */
.checks{display:flex;flex-direction:column;gap:6px;margin-bottom:8px}
.chk{display:flex;align-items:center;gap:8px;font-size:12.5px;
  padding:5px 9px;border-radius:7px;background:#fafafa}
.chk .dot{width:8px;height:8px;border-radius:50%;flex:0 0 auto}
.chk.pass .dot{background:var(--ok)} .chk.fail .dot{background:var(--bad)}
.chk.na .dot{background:#bbb}
.chk .val{margin-left:auto;font-variant-numeric:tabular-nums;
  font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--sub)}
.notes-list{list-style:none;margin:0;padding:0;font-size:12px}
.notes-list li{padding:6px 0 6px 20px;border-bottom:1px solid #f4f4f4;position:relative}
.notes-list li:last-child{border-bottom:0}
.notes-list li:before{content:"!";position:absolute;left:4px;top:6px;width:13px;
  height:13px;line-height:13px;text-align:center;border-radius:50%;
  background:var(--warnbg);color:var(--warn);font-size:10px;font-weight:700}
.notes-list li.info:before{content:"i";background:#eef4fb;color:var(--accent)}

/* ---------- 三栏（model） ---------- */
.cols{display:grid;grid-template-columns:1fr 1.15fr 1fr;gap:16px}
@media(max-width:1100px){.cols{grid-template-columns:1fr}}
.cols h3{font-size:12px;margin:2px 0 7px;color:var(--sub);font-weight:600;
  letter-spacing:.3px;text-transform:uppercase}
.corr{display:grid;gap:1px;font-size:10.5px}
.corr .c{aspect-ratio:1;display:flex;align-items:center;justify-content:center;
  border-radius:2px;color:#fff;font-variant-numeric:tabular-nums}
.corr .nm{color:var(--sub);font-size:9.5px;text-align:right;padding-right:4px;
  white-space:nowrap}

/* ---------- 抽屉 ---------- */
.drawer{position:fixed;top:0;right:0;height:100%;width:min(760px,92vw);
  background:#fff;border-left:1px solid var(--line);z-index:50;
  transform:translateX(102%);transition:transform .22s ease;
  display:flex;flex-direction:column;box-shadow:-8px 0 30px rgba(0,0,0,.08)}
.drawer.open{transform:translateX(0)}
.drawer .dh{display:flex;align-items:center;gap:10px;padding:12px 16px;
  border-bottom:1px solid var(--line)}
.drawer .dh h2{font-size:14px;margin:0}
.drawer .db{padding:14px 16px;overflow:auto}
.mask{position:fixed;inset:0;background:rgba(0,0,0,.12);z-index:40;
  opacity:0;pointer-events:none;transition:opacity .2s}
.mask.open{opacity:1;pointer-events:auto}

/* ---------- 浮层 ---------- */
.tip{position:fixed;pointer-events:none;background:rgba(20,20,20,.9);
  color:#fff;font-size:11.5px;padding:6px 9px;border-radius:7px;z-index:60;
  font-family:ui-monospace,Consolas,monospace;opacity:0;transition:opacity .1s;
  white-space:pre;line-height:1.45}
.tip.on{opacity:1}
.hint{font-size:11.5px;color:var(--sub);margin:2px 0 10px}
kbd{font-family:ui-monospace,Consolas,monospace;font-size:11px;background:#f4f4f4;
  border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;
  padding:0 4px}
footer{margin-top:26px;color:var(--sub);font-size:11.5px;border-top:1px solid var(--line);
  padding-top:10px}
footer code{font-family:ui-monospace,Consolas,monospace;font-size:11px;
  background:#f7f7f7;padding:1px 5px;border-radius:4px}
</style>
</head>
<body>
<header><div class="wrap hd">
  <div>
    <h1 id="ttl">__TITLE__</h1>
    <div class="sub" id="sub"></div>
  </div>
  <div class="chips" id="chips"></div>
  <div class="tools">
    <select id="theme" title="排版风格预设（不是期刊合规声明）"></select>
    <button id="btn-exp">导出…</button>
    <button id="btn-reset" title="恢复默认布局">重置</button>
  </div>
</div></header>

<div class="wrap">
  <section class="kpis" id="kpis"></section>
  <div class="hint">
    滚轮缩放 · 拖动平移 · 双击复位 · 点击固定读数 ·
    <kbd>Shift</kbd>+滚轮只缩横轴 · 拖卡片标题可重排
  </div>
  <main class="grid" id="grid"></main>
  <footer id="foot"></footer>
</div>

<div class="mask" id="mask"></div>
<aside class="drawer" id="drawer">
  <div class="dh"><h2 id="dt"></h2><div style="margin-left:auto">
    <button id="dclose">关闭</button></div></div>
  <div class="db" id="db"></div>
</aside>
<div class="tip" id="tip"></div>

<script>
"use strict";
const D = __DATA__;
const THEMES = __THEMES__;
const META = __PANELMETA__;
const DEFAULT_LAYOUT = __LAYOUT__;
const SPEC_VER = "__VER__";
const LS_KEY = "bmdrt.dash.layout." + D.name;

let LAYOUT = JSON.parse(JSON.stringify(DEFAULT_LAYOUT));
let TH = THEMES[LAYOUT.theme] || THEMES.minimal;
// 卡片内高 = 行高×行数 + 间距；绘图区要再减掉卡头(37)、读数行(26)与内边距(10)。
// 之前漏了读数行，svg 比容器高 15px，轴标题被挤到卡片外面压住读数。
const ROW = 300, GAP = 16, HDR = 37, RO = 26, PADV = 10;
let PIN = null;                 // 固定的频点索引
let ZOOM = {};                  // panelId -> {x0,x1,y0,y1}
let OPEN = null;                // 抽屉里正在看的面板

/* ---------------- 小工具 ---------------- */
const $ = s => document.querySelector(s);
const el = (t, a = {}) => { const e = document.createElement(t);
  for (const k in a) e.setAttribute(k, a[k]); return e; };
// SVG 必须建在 SVG 命名空间里。用 document.createElement("svg") 得到的是
// HTML 元素，往它 innerHTML 塞 <line>/<text> 会被当普通文字排成一行——不是图。
const SVGNS = "http://www.w3.org/2000/svg";
const svgEl = (t, a = {}) => { const e = document.createElementNS(SVGNS, t);
  for (const k in a) e.setAttribute(k, a[k]); return e; };
const fmt = (v, d = 3) => {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1).replace("e", "e");
  return (+v.toFixed(d)).toString();
};
const sci = v => (v === null || v === undefined || !isFinite(v)) ? "—"
  : (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0))
    ? v.toExponential(1) : (+v.toPrecision(3)).toString();
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
  .replace(/>/g, "&gt;");
// 结果 JSON 里的提示是 markdown 风格（**加粗**）。直接 innerHTML 会显示星号，
// 转义后只放开 **…** 这一种标记。
const md = s => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");

const SUP = {"-":"⁻","0":"⁰","1":"¹","2":"²","3":"³","4":"⁴",
  "5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"};
function logLabel(v) {
  const e = Math.log10(v);
  if (Math.abs(e - Math.round(e)) < 1e-9) {
    const n = String(Math.round(e));
    return "10" + n.split("").map(ch => SUP[ch] || ch).join("");
  }
  return fmt(v, 3);
}
function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag)
    .find(s => (hi - lo) / s <= n * 1.35) || 10 * mag;
  const out = []; let v = Math.ceil(lo / step) * step;
  for (; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
function logTicks(lo, hi) {
  const a = Math.ceil(Math.log10(lo)), b = Math.floor(Math.log10(hi)), out = [];
  for (let e = a; e <= b; e++) if (Math.pow(10, e) >= lo && Math.pow(10, e) <= hi)
    out.push(Math.pow(10, e));
  if (out.length < 2) {                       // 跨度不足一个量级时补 2/5
    for (const m of [2, 5]) { const v = m * Math.pow(10, Math.floor(Math.log10(lo)));
      if (v > lo && v < hi) out.push(v); }
    out.sort((x, y) => x - y);
  }
  return out;
}

/* ---------------- 比例尺 ---------------- */
function scale(dom, rng, log) {
  if (log) {
    const a = Math.log10(dom[0]), b = Math.log10(dom[1]);
    return v => rng[0] + (Math.log10(v) - a) / (b - a) * (rng[1] - rng[0]);
  }
  return v => rng[0] + (v - dom[0]) / (dom[1] - dom[0]) * (rng[1] - rng[0]);
}
function invScale(dom, rng, log) {
  if (log) {
    const a = Math.log10(dom[0]), b = Math.log10(dom[1]);
    return p => Math.pow(10, a + (p - rng[0]) / (rng[1] - rng[0]) * (b - a));
  }
  return p => dom[0] + (p - rng[0]) / (rng[1] - rng[0]) * (dom[1] - dom[0]);
}

/* ---------------- 画布骨架 ---------------- */
function frame(svg, w, h, o) {
  const pad = { l: o.padL || 52, r: o.padR || 14, t: 12, b: o.padB || 40 };
  const rngX = [pad.l, w - pad.r], rngY = [h - pad.b, pad.t];
  const x = scale(o.xdom, rngX, o.xlog), y = scale(o.ydom, rngY, o.ylog);
  const xi = invScale(o.xdom, rngX, o.xlog), yi = invScale(o.ydom, rngY, o.ylog);
  const F = [];
  // 网格
  const xa = o.xlog ? logTicks(o.xdom[0], o.xdom[1]) : niceTicks(o.xdom[0], o.xdom[1]);
  const ya = o.ylog ? logTicks(o.ydom[0], o.ydom[1]) : niceTicks(o.ydom[0], o.ydom[1]);
  if (TH.grid) {
    xa.forEach(v => F.push(`<line x1="${x(v).toFixed(1)}" y1="${rngY[1]}" x2="${x(v).toFixed(1)}" y2="${rngY[0]}" stroke="${TH.line}" stroke-width="1"/>`));
    ya.forEach(v => F.push(`<line x1="${rngX[0]}" y1="${y(v).toFixed(1)}" x2="${rngX[1]}" y2="${y(v).toFixed(1)}" stroke="${TH.line}" stroke-width="1"/>`));
  }
  // 轴
  F.push(`<line x1="${rngX[0]}" y1="${rngY[0]}" x2="${rngX[1]}" y2="${rngY[0]}" stroke="${TH.fg}" stroke-width="${TH.axisLw}"/>`);
  F.push(`<line x1="${rngX[0]}" y1="${rngY[1]}" x2="${rngX[0]}" y2="${rngY[0]}" stroke="${TH.fg}" stroke-width="${TH.axisLw}"/>`);
  const fs = TH.fontSize - 1.5, tk = TH.tickLen;
  xa.forEach(v => {
    const p = x(v);
    F.push(`<line x1="${p.toFixed(1)}" y1="${rngY[0]}" x2="${p.toFixed(1)}" y2="${rngY[0] + tk}" stroke="${TH.fg}" stroke-width="${TH.axisLw * 0.8}"/>`);
    F.push(`<text x="${p.toFixed(1)}" y="${rngY[0] + tk + fs + 2}" font-size="${fs}" text-anchor="middle" fill="${TH.sub}">${o.xlog ? logLabel(v) : fmt(v, 3)}</text>`);
  });
  ya.forEach(v => {
    const p = y(v);
    F.push(`<line x1="${rngX[0] - tk}" y1="${p.toFixed(1)}" x2="${rngX[0]}" y2="${p.toFixed(1)}" stroke="${TH.fg}" stroke-width="${TH.axisLw * 0.8}"/>`);
    F.push(`<text x="${rngX[0] - tk - 5}" y="${(p + fs * 0.35).toFixed(1)}" font-size="${fs}" text-anchor="end" fill="${TH.sub}">${o.ylog ? logLabel(v) : fmt(v, 3)}</text>`);
  });
  // 轴标题
  F.push(`<text x="${((rngX[0] + rngX[1]) / 2).toFixed(1)}" y="${h - 6}" font-size="${fs + 1}" text-anchor="middle" fill="${TH.fg}">${o.xlabel || ""}</text>`);
  F.push(`<text transform="translate(13,${((rngY[0] + rngY[1]) / 2).toFixed(1)}) rotate(-90)" font-size="${fs + 1}" text-anchor="middle" fill="${TH.fg}">${o.ylabel || ""}</text>`);
  return { F, x, y, xi, yi, rngX, rngY, fs };
}
const path = (pts, x, y) => pts.map((p, i) =>
  (i ? "L" : "M") + x(p[0]).toFixed(2) + " " + y(p[1]).toFixed(2)).join(" ");
function bandPath(lo, hi, f, x, y) {
  const up = lo.map((v, i) => [f[i], v]);
  const dn = hi.map((v, i) => [f[i], v]).reverse();
  return path(up, x, y) + " L" + dn.map(p =>
    x(p[0]).toFixed(2) + " " + y(p[1]).toFixed(2)).join(" L") + " Z";
}

/* ---------------- 数据域 ---------------- */
function domain(panel, w, h) {
  const c = D.curves, z = ZOOM[panel];
  if (z) return z;
  if (panel === "fit") {
    const xs = c.zre.concat(c.fre);
    const ys = c.zim.concat(c.fim).map(v => -v);   // Nyquist 纵轴 = −Z″
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    const dx = (x1 - x0) * 0.06 || 0.1, dy = (y1 - y0) * 0.06 || 0.1;
    x0 -= dx; x1 += dx; y0 -= dy; y1 += dy;
    // Nyquist 必须等比例（1 Ω 横 = 1 Ω 纵），否则圆弧会被压成椭圆。
    // 等比例 + 宽画布 = 四周留白，所以改**扩数据域**去贴合画布比例，
    // 而不是把图缩到中间（matplotlib 的 adjustable='datalim' 同理）。
    const bw = Math.max(80, (w || 600) - 66), bh = Math.max(80, (h || 400) - 52);
    const tgt = bw / bh, xspan = x1 - x0, yspan = y1 - y0;
    if (xspan / yspan < tgt) {
      const need = yspan * tgt, cx = (x0 + x1) / 2;
      x0 = cx - need / 2; x1 = cx + need / 2;
    } else {
      const need = xspan / tgt, cy = (y0 + y1) / 2;
      y0 = cy - need / 2; y1 = cy + need / 2;
    }
    return { x0, x1, y0, y1 };
  }
  if (panel === "drt") {
    if (!D.drt) return null;
    const g = D.drt.med.concat(D.drt.hi);
    return { x0: Math.min(...D.drt.f), x1: Math.max(...D.drt.f),
             y0: Math.min(0, ...D.drt.lo) * 1.05 || -1e-3,
             y1: Math.max(...g) * 1.08 };
  }
  const yk = (c.nre && D.curves.nre.length) ? "nre" : "rre";
  const ys = (c[yk] || []).concat(c[yk === "nre" ? "nim" : "rim"] || []);
  const m = Math.max(1.2, ...ys.map(Math.abs)) * 1.12;
  return { x0: Math.min(...c.f), x1: Math.max(...c.f), y0: -m, y1: m };
}

/* ---------------- 面板渲染 ---------------- */
function renderFit(w, h) {
  const c = D.curves, o = domain("fit", w, h);
  const fr = frame(null, w, h, { xdom: [o.x0, o.x1], ydom: [o.y0, o.y1],
    xlabel: "Z′ / Ω", ylabel: "−Z″ / Ω" });
  const { x, y, F, rngX, rngY } = fr;
  // 区间的 Re/Im 是**分别**取的边缘分位，复平面上是个矩形包络，
  // 不是一条可以连成多边形的"带"。早先按 re_lo→re_hi→im_hi 连成封闭路径，
  // 在拟合很差的谱上会画出一条横贯全图的色带（几何上没意义）。
  // 这里改成逐点画十字误差棒，长度就是各自的 90% 边际区间。
  if (D.has_band) {
    c.f.forEach((v, i) => {
      const ymid = y(-c.fim[i]);                     // 该频点的中位虚部
      F.push(`<line x1="${x(c.re_lo[i]).toFixed(1)}" y1="${ymid.toFixed(1)}" x2="${x(c.re_hi[i]).toFixed(1)}" y2="${ymid.toFixed(1)}" stroke="${TH.band}" stroke-opacity="0.4" stroke-width="1"/>`);
      F.push(`<line x1="${x(c.fre[i]).toFixed(1)}" y1="${y(-c.im_lo[i]).toFixed(1)}" x2="${x(c.fre[i]).toFixed(1)}" y2="${y(-c.im_hi[i]).toFixed(1)}" stroke="${TH.band}" stroke-opacity="0.4" stroke-width="1"/>`);
    });
  }
  F.push(`<path d="${path(c.fre.map((v, i) => [v, -c.fim[i]]), x, y)}" fill="none" stroke="${TH.fit}" stroke-width="1.6"/>`);
  c.zre.forEach((v, i) => F.push(
    `<circle cx="${x(v).toFixed(1)}" cy="${y(-c.zim[i]).toFixed(1)}" r="3" fill="none" stroke="${TH.data}" stroke-width="1"/>`));
  // 图例
  const lx = rngX[0] + 10, ly = rngY[1] + 12, fs = TH.fontSize - 1.5;
  F.push(`<circle cx="${lx + 4}" cy="${ly}" r="3" fill="none" stroke="${TH.data}" stroke-width="1"/>`);
  F.push(`<text x="${lx + 14}" y="${ly + 3.5}" font-size="${fs}" fill="${TH.sub}">data</text>`);
  F.push(`<line x1="${lx + 52}" y1="${ly}" x2="${lx + 70}" y2="${ly}" stroke="${TH.fit}" stroke-width="1.6"/>`);
  F.push(`<text x="${lx + 75}" y="${ly + 3.5}" font-size="${fs}" fill="${TH.sub}">median</text>`);
  if (D.has_band) {
    F.push(`<rect x="${lx + 118}" y="${ly - 5}" width="16" height="10" fill="${TH.band}" fill-opacity="0.18"/>`);
    F.push(`<text x="${lx + 139}" y="${ly + 3.5}" font-size="${fs}" fill="${TH.sub}">90% 边际区间</text>`);
  }
  const pin = PIN !== null ? PIN : null;
  let mark = "";
  if (pin !== null) {
    const px = x(c.zre[pin]), py = y(-c.zim[pin]);
    mark = `<line x1="${px}" y1="${rngY[1]}" x2="${px}" y2="${rngY[0]}" stroke="${TH.fg}" stroke-opacity=".25" stroke-dasharray="3 3"/>
      <circle cx="${px}" cy="${py}" r="5.5" fill="none" stroke="${TH.fit}" stroke-width="1.6"/>`;
  }
  return { body: F.join("") + mark, fr, kind: "fit" };
}

function renderDrt(w, h) {
  const d = D.drt;
  if (!d) return { body: `<text x="${w / 2}" y="${h / 2}" text-anchor="middle" font-size="${TH.fontSize}" fill="${TH.sub}">无后验 DRT（缺 tau/pi）</text>`, skip: true };
  const o = domain("drt");
  const fr = frame(null, w, h, { xdom: [o.x0, o.x1], ydom: [o.y0, o.y1], xlog: true,
    xlabel: "f / Hz", ylabel: "γ / Ω" });
  const { x, y, F, rngY } = fr;
  F.push(`<path d="${bandPath(d.lo, d.hi, d.f, x, y)}" fill="${TH.drt}" fill-opacity="0.15" stroke="none"/>`);
  F.push(`<path d="${path(d.f.map((v, i) => [v, d.med[i]]), x, y)}" fill="none" stroke="${TH.drt}" stroke-width="1.7"/>`);
  const ymax = o.y1;
  (D.components || []).forEach(cm => {
    if (!cm.f0 || cm.f0 < o.x0 || cm.f0 > o.x1) return;
    const px = x(cm.f0);
    F.push(`<line x1="${px.toFixed(1)}" y1="${rngY[1]}" x2="${px.toFixed(1)}" y2="${rngY[0]}" stroke="${TH.sub}" stroke-opacity=".55" stroke-dasharray="2 3"/>`);
    F.push(`<text x="${(px + 3).toFixed(1)}" y="${rngY[1] + 12}" font-size="${TH.fontSize - 2.5}" fill="${TH.sub}">#${cm.idx}</text>`);
  });
  const pin = PIN;
  let mark = "";
  if (pin !== null && d.f[pin] !== undefined) {
    const px = x(d.f[pin]);
    mark = `<line x1="${px}" y1="${rngY[1]}" x2="${px}" y2="${rngY[0]}" stroke="${TH.fg}" stroke-opacity=".2" stroke-dasharray="3 3"/>`;
  }
  return { body: F.join("") + mark, fr, kind: "drt" };
}

function renderResid(w, h) {
  const c = D.curves;
  const useNorm = !!(c.nre && D.curves.nre.length);
  const o = domain("resid");
  const fr = frame(null, w, h, { xdom: [o.x0, o.x1], ydom: [o.y0, o.y1], xlog: true,
    xlabel: "f / Hz", ylabel: useNorm ? "残差 / σ" : "残差 / Ω" });
  const { x, y, F, rngX, rngY } = fr;
  const fs = TH.fontSize - 1.5;
  if (useNorm) {
    [-2, -1, 1, 2].forEach(s => F.push(
      `<line x1="${rngX[0]}" y1="${y(s).toFixed(1)}" x2="${rngX[1]}" y2="${y(s).toFixed(1)}" stroke="${TH.sub}" stroke-opacity="${Math.abs(s) === 1 ? .35 : .18}" stroke-dasharray="${Math.abs(s) === 1 ? '4 3' : '2 4'}"/>`));
  }
  F.push(`<line x1="${rngX[0]}" y1="${y(0).toFixed(1)}" x2="${rngX[1]}" y2="${y(0).toFixed(1)}" stroke="${TH.fg}" stroke-opacity=".55"/>`);
  const re = useNorm ? c.nre : c.rre, im = useNorm ? c.nim : c.rim;
  c.f.forEach((v, i) => {
    F.push(`<circle cx="${x(v).toFixed(1)}" cy="${y(re[i]).toFixed(1)}" r="2.6" fill="none" stroke="${TH.reC}" stroke-width="1.2"/>`);
    F.push(`<rect x="${(x(v) - 2.4).toFixed(1)}" y="${(y(im[i]) - 2.4).toFixed(1)}" width="4.8" height="4.8" fill="none" stroke="${TH.imC}" stroke-width="1.1"/>`);
  });
  const lx = rngX[0] + 8, ly = rngY[1] + 12;
  F.push(`<circle cx="${lx + 3}" cy="${ly}" r="2.6" fill="none" stroke="${TH.reC}" stroke-width="1.2"/><text x="${lx + 10}" y="${ly + 3.5}" font-size="${fs}" fill="${TH.sub}">Re</text>`);
  F.push(`<rect x="${lx + 33}" y="${ly - 2.4}" width="4.8" height="4.8" fill="none" stroke="${TH.imC}" stroke-width="1.1"/><text x="${lx + 42}" y="${ly + 3.5}" font-size="${fs}" fill="${TH.sub}">Im</text>`);
  let mark = "";
  if (PIN !== null) {
    const px = x(c.f[PIN]);
    mark = `<line x1="${px}" y1="${rngY[1]}" x2="${px}" y2="${rngY[0]}" stroke="${TH.fg}" stroke-opacity=".25" stroke-dasharray="3 3"/>
      <circle cx="${px}" cy="${y(re[PIN]).toFixed(1)}" r="5" fill="none" stroke="${TH.reC}" stroke-width="1.5"/>
      <rect x="${(px - 4.5).toFixed(1)}" y="${(y(im[PIN]) - 4.5).toFixed(1)}" width="9" height="9" fill="none" stroke="${TH.imC}" stroke-width="1.4"/>`;
  }
  return { body: F.join("") + mark, fr, kind: "resid" };
}

function renderBode(w, h) {
  const c = D.curves;
  const mag = c.f.map((_, i) => Math.hypot(c.zre[i], c.zim[i]));
  const magf = c.f.map((_, i) => Math.hypot(c.fre[i], c.fim[i]));
  const ph = c.f.map((_, i) => Math.atan2(c.zim[i], c.zre[i]) * 180 / Math.PI);
  const phf = c.f.map((_, i) => Math.atan2(c.fim[i], c.fre[i]) * 180 / Math.PI);
  const all = mag.concat(magf);
  const o = { x0: Math.min(...c.f), x1: Math.max(...c.f),
              y0: Math.min(...all) * 0.85, y1: Math.max(...all) * 1.15 };
  const fr = frame(null, w, h, { xdom: [o.x0, o.x1], ydom: [o.y0, o.y1],
    xlog: true, ylog: true, xlabel: "f / Hz", ylabel: "|Z| / Ω", padR: 46 });
  const { x, y, F, rngX, rngY } = fr;
  const p0 = Math.min(...ph.concat(phf)), p1 = Math.max(...ph.concat(phf));
  const yP = scale([p0 - 6, p1 + 6], [rngY[0], rngY[1]], false);
  niceTicks(p0 - 6, p1 + 6, 5).forEach(v => {
    F.push(`<text x="${rngX[1] + 6}" y="${(yP(v) + 3.5).toFixed(1)}" font-size="${TH.fontSize - 1.5}" fill="${TH.imC}">${fmt(v, 0)}</text>`);
  });
  F.push(`<text transform="translate(${w - 6},${(rngY[0] + rngY[1]) / 2}) rotate(-90)" font-size="${TH.fontSize - 0.5}" text-anchor="middle" fill="${TH.imC}">phase / deg</text>`);
  F.push(`<path d="${path(c.f.map((v, i) => [v, mag[i]]), x, y)}" fill="none" stroke="${TH.fit}" stroke-width="1.5"/>`);
  F.push(`<path d="${path(c.f.map((v, i) => [v, ph[i]]), x, yP)}" fill="none" stroke="${TH.imC}" stroke-width="1.2" stroke-dasharray="4 3"/>`);
  c.f.forEach((v, i) => F.push(
    `<circle cx="${x(v).toFixed(1)}" cy="${y(mag[i]).toFixed(1)}" r="2.6" fill="none" stroke="${TH.data}" stroke-width="1"/>`));
  return { body: F.join(""), fr, kind: "bode" };
}

/* ---------------- 卡片 ---------------- */
function buildCards() {
  const grid = $("#grid");
  grid.innerHTML = "";
  LAYOUT.panels.forEach((p, idx) => {
    const m = META[p.id] || { title: p.id, sub: "" };
    const card = el("section", { class: "card", "data-id": p.id, draggable: "true" });
    card.style.gridColumn = `span ${p.span}`;
    card.style.gridRow = `span ${p.rows || 1}`;
    card.style.display = p.visible ? "" : "none";
    if (["fit", "drt", "resid"].includes(p.id)) {
      const rows = p.rows || 1;
      card.style.height = (rows * ROW + (rows - 1) * GAP) + "px";
    }
    card.innerHTML = `
      <div class="ch">
        <span class="drag">⋮⋮</span>
        <h2>${m.title}</h2><span class="s">${m.sub || ""}</span>
        <div class="acts">
          ${m.detail ? `<button data-act="detail">${m.detail}</button>` : ""}
          <button data-act="span">宽</button>
          <button data-act="svg">SVG</button>
          <button data-act="hide">隐藏</button>
        </div>
      </div>
      <div class="cb"></div><div class="readout"></div>`;
    grid.appendChild(card);
    card.addEventListener("dragstart", e => {
      e.dataTransfer.setData("text/plain", String(idx));
      card.classList.add("dragging");
    });
    card.addEventListener("dragend", () => card.classList.remove("dragging"));
    card.addEventListener("dragover", e => e.preventDefault());
    card.addEventListener("drop", e => {
      e.preventDefault();
      const from = +e.dataTransfer.getData("text/plain");
      if (from === idx) return;
      const moved = LAYOUT.panels.splice(from, 1)[0];
      LAYOUT.panels.splice(idx, 0, moved);
      saveLayout(); buildCards(); drawAll();
    });
    card.querySelector(".acts").addEventListener("click", e => {
      const act = e.target.dataset.act; if (!act) return;
      if (act === "hide") { p.visible = false; saveLayout(); buildCards(); drawAll(); }
      if (act === "span") { p.span = p.span >= 12 ? 4 : p.span + 2;
        saveLayout(); buildCards(); drawAll(); }
      if (act === "detail") openDrawer(p.id);
      if (act === "svg") exportPanelSvg(p.id, m.title);
    });
  });
  // 「已隐藏」恢复条
  const hid = LAYOUT.panels.filter(p => !p.visible);
  if (hid.length) {
    const bar = el("div");
    bar.style.cssText = "grid-column:span 12;font-size:12px;color:var(--sub)";
    bar.innerHTML = "已隐藏：" + hid.map(p =>
      `<button data-show="${p.id}" style="margin-left:6px">${(META[p.id] || {}).title}</button>`).join("");
    bar.addEventListener("click", e => {
      const id = e.target.dataset.show; if (!id) return;
      LAYOUT.panels.find(q => q.id === id).visible = true;
      saveLayout(); buildCards(); drawAll();
    });
    grid.appendChild(bar);
  }
}

/* ---------------- 绘制 ---------------- */
function drawCard(id) {
  const card = document.querySelector(`.card[data-id="${id}"]`);
  if (!card || card.style.display === "none") return;
  const box = card.querySelector(".cb");
  if (id === "model") return drawModel(box);
  if (id === "notes") return drawNotes(box);
  const w = Math.max(320, box.clientWidth || 520);
  const rows = (LAYOUT.panels.find(q => q.id === id) || {}).rows || 1;
  const h = Math.max(180, rows * ROW + (rows - 1) * GAP - HDR - RO - PADV);
  let r;
  try {
    r = id === "fit" ? renderFit(w, h) : id === "drt" ? renderDrt(w, h) : renderResid(w, h);
  } catch (err) {
    box.innerHTML = `<div style="color:#c0392b;font-size:12px">渲染失败：${err.message}</div>`;
    return;
  }
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h,
    "data-panel": id, style: `height:${h}px;width:100%` });
  svg.innerHTML = r.body;
  box.innerHTML = ""; box.appendChild(svg);
  if (!r.skip) bindInteract(svg, id, r.fr, w, h);
  else card.querySelector(".readout").textContent = "";
}

function drawAll() {
  LAYOUT.panels.forEach(p => { if (p.visible) drawCard(p.id); });
  renderReadout();
}

/* ---------------- 交互：缩放 / 平移 / 悬停 ---------------- */
function bindInteract(svg, id, fr, w, h) {
  const { xi, yi, rngX, rngY } = fr;
  let drag = null;
  svg.addEventListener("wheel", e => {
    e.preventDefault();
    const rect = svg.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (w / rect.width);
    const py = (e.clientY - rect.top) * (h / rect.height);
    const k = e.deltaY > 0 ? 1.18 : 1 / 1.18;
    const o = domain(id); if (!o) return;
    const xAt = xi(clamp(px, rngX[0], rngX[1]));
    const yAt = yi(clamp(py, rngY[1], rngY[0]));
    const nx0 = xAt + (o.x0 - xAt) * (e.shiftKey ? 1 : k);
    const nx1 = xAt + (o.x1 - xAt) * (e.shiftKey ? 1 : k);
    const ny0 = yAt + (o.y0 - yAt) * k, ny1 = yAt + (o.y1 - yAt) * k;
    ZOOM[id] = { x0: nx0, x1: nx1, y0: ny0, y1: ny1 };
    drawCard(id);
  }, { passive: false });
  svg.addEventListener("pointerdown", e => {
    drag = { x: e.clientX, y: e.clientY, o: { ...domain(id) } };
    svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener("pointermove", e => {
    const rect = svg.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (w / rect.width);
    const py = (e.clientY - rect.top) * (h / rect.height);
    if (drag) {
      const o = drag.o, dx = (e.clientX - drag.x) * (w / rect.width),
        dy = (e.clientY - drag.y) * (h / rect.height);
      const x0 = xi(rngX[0] - dx), x1 = xi(rngX[1] - dx);
      const y0 = yi(rngY[0] - dy), y1 = yi(rngY[1] - dy);
      ZOOM[id] = { x0, x1, y0, y1 }; drawCard(id); return;
    }
    hoverAt(id, fr, px, py, w, h, e.clientX, e.clientY);
  });
  svg.addEventListener("pointerup", () => { drag = null; });
  svg.addEventListener("pointerleave", () => { $("#tip").classList.remove("on"); });
  svg.addEventListener("dblclick", () => { delete ZOOM[id]; drawCard(id); });
  svg.addEventListener("click", () => { PIN = PIN === hoverIdx ? null : hoverIdx; drawAll(); });
}

let hoverIdx = null;
function hoverAt(id, fr, px, py, w, h, cx, cy) {
  const c = D.curves;
  const { x, y } = fr;
  let best = null, bd = 1e9;
  if (id === "drt") {
    if (!D.drt) return;
    D.drt.f.forEach((v, i) => { const d = Math.hypot(x(v) - px, y(D.drt.med[i]) - py);
      if (d < bd) { bd = d; best = i; } });
  } else {
    c.f.forEach((v, i) => {
      const yy = id === "fit" ? y(-c.zim[i])
        : y((D.curves.nre && D.curves.nre.length) ? c.nre[i] : c.rre[i]);
      const d = Math.hypot(x(v) - px, yy - py);
      if (d < bd) { bd = d; best = i; }
    });
  }
  if (best === null || bd > 90) { $("#tip").classList.remove("on"); return; }
  hoverIdx = best;
  const tip = $("#tip");
  tip.classList.add("on");
  tip.style.left = Math.min(cx + 14, innerWidth - 250) + "px";
  tip.style.top = (cy + 16) + "px";
  tip.textContent = id === "drt"
    ? `f = ${fmt(D.drt.f[best], 4)} Hz\nγ = ${fmt(D.drt.med[best], 3)} Ω\n90%: ${fmt(D.drt.lo[best], 3)} – ${fmt(D.drt.hi[best], 3)}`
    : `f = ${fmt(c.f[best], 4)} Hz\nZ_obs = ${fmt(c.zre[best], 4)} ${c.zim[best] >= 0 ? "+" : "−"} j${fmt(Math.abs(c.zim[best]), 4)}\nZ_fit = ${fmt(c.fre[best], 4)} ${c.fim[best] >= 0 ? "+" : "−"} j${fmt(Math.abs(c.fim[best]), 4)}\nresid = ${fmt(c.rre[best], 3)} / ${fmt(c.rim[best], 3)} Ω\nσ = ${fmt(c.sre[best], 3)} / ${fmt(c.sim[best], 3)} Ω`
    + ((D.curves.nre && D.curves.nre.length) ? `\nnorm = ${fmt(c.nre[best], 2)} / ${fmt(c.nim[best], 2)} σ` : "");
  renderReadout(best);
}

function renderReadout(i = PIN) {
  LAYOUT.panels.forEach(p => {
    const card = document.querySelector(`.card[data-id="${p.id}"]`);
    if (!card) return;
    const out = card.querySelector(".readout");
    if (!out) return;
    if (i === null || i === undefined) { out.textContent = "悬停查看该频点的观测/拟合/残差，点击固定"; return; }
    const c = D.curves;
    if (p.id === "fit" || p.id === "resid") {
      out.textContent = `f ${fmt(c.f[i], 4)} Hz · obs ${fmt(c.zre[i], 3)}${c.zim[i] >= 0 ? "+" : "−"}j${fmt(Math.abs(c.zim[i]), 3)} · fit ${fmt(c.fre[i], 3)}${c.fim[i] >= 0 ? "+" : "−"}j${fmt(Math.abs(c.fim[i]), 3)} · resid ${fmt(c.rre[i], 3)}/${fmt(c.rim[i], 3)} Ω` +
        ((c.nre && c.nre.length) ? ` · ${fmt(c.nre[i], 2)}/${fmt(c.nim[i], 2)} σ` : "");
    } else if (p.id === "drt" && D.drt) {
      out.textContent = `f ${fmt(D.drt.f[i], 4)} Hz · γ ${fmt(D.drt.med[i], 3)} Ω · 90% ${fmt(D.drt.lo[i], 3)}–${fmt(D.drt.hi[i], 3)}`;
    } else { out.textContent = ""; }
  });
}

/* ---------------- 判定卡 ---------------- */
function drawNotes(box) {
  const chips = D.checks.map(c => {
    const icon = c.state === "pass" ? "✓" : c.state === "fail" ? "✕" : "–";
    return `<div class="chk ${c.state}" style="flex:1 1 200px">
      <span class="dot"></span><span>${c.name}</span>
      <span class="val">${icon} ${c.value === null ? "不可用" : fmt(c.value, 3)}</span></div>`;
  }).join("");
  const kk = (D.kk && D.kk.status) ? `<div class="chk ${D.kk.status === "ok" ? "pass" : "fail"}"
      style="flex:1 1 200px" title="${md(D.kk.reason || "")}">
      <span class="dot"></span><span>Lin-KK 门禁</span>
      <span class="val">${D.kk.status}</span></div>` : "";
  const warn = (D.warnings || []).map(w => `<li>${md(w)}</li>`).join("");
  const layer = (D.layer_notes || []).map(w => `<li class="info">${md(w)}</li>`).join("");
  const strong = ((D.corr && D.corr.strong) || []).map(x =>
    `<li><b>${esc(x.a)} ↔ ${esc(x.b)}</b> ρ=${fmt(x.rho, 2)}</li>`).join("");
  const v = D.verdict;
  const lv = v.level === "ok" ? "ok" : v.level === "bad" ? "bad" : "warn";
  box.innerHTML = `
    <div class="checks" style="flex-direction:row;flex-wrap:wrap;gap:8px">${chips}${kk}</div>
    <div style="font-size:11px;color:var(--sub);margin:8px 2px 10px">
      <span class="chip ${lv}" style="font-size:11px">${v.label}</span>
      阈值：acf ≤ ${D.thresholds.acf} · R̂ ≤ ${D.thresholds.rhat} · div = ${D.thresholds.div}
      —— 这三项是结构与采样判据，与「K 的采样分级」不是同一件事：
      分级 2 只说明采样达标，不代表残差结构过关。
    </div>
    <div class="cols" style="grid-template-columns:1.35fr 1fr">
      <div><h3>警告与限制</h3><ul class="notes-list">${warn || '<li class="info">无</li>'}</ul></div>
      <div><h3>口径备注</h3><ul class="notes-list">${layer}${strong}</ul></div>
    </div>`;
}

/* ---------------- 模型与参数 ---------------- */
function drawModel(box) {
  const candRows = (D.candidates || []).map(c => {
    const cls = c.selected ? "sel" : "";
    const st = c.state || "";
    const ZH = { rejected: "剔除", degraded: "降级可用", selected: "选中",
      converged: "通过", not_compared: "未比较", exploratory: "探索性",
      noted: "已记录" };
    return `<tr class="${cls}"><td class="num">${c.k}</td>
      <td><span class="pill ${st}">${ZH[st] || st}</span></td>
      <td>${c.selected ? "选中" : ""}</td>
      <td style="color:var(--sub);font-size:11.5px">${md(c.detail || "")}</td></tr>`;
  }).join("") || `<tr><td colspan="4" style="color:var(--sub)">无候选记录</td></tr>`;

  // param_corr.summary 里的分位**已经是原尺度**（ln 参数已换回），
  // 不能再取 exp——这是本轮踩到的坑：页面上会把 R_inf 显示成 1.0000。
  const pRows = (D.params || []).map(p => `<tr><td>${esc(p.name)}</td>
    <td class="num">${sci(p.q50)}</td>
    <td class="num" style="color:var(--sub)">${sci(p.q05)}</td>
    <td class="num" style="color:var(--sub)">${sci(p.q95)}</td>
    <td style="color:var(--sub);font-size:11px">${p.kind === "fixed" ? "固定" : (p.log ? "log 采样" : "")}</td></tr>`).join("")
    || `<tr><td colspan="5" style="color:var(--sub)">无参数记录</td></tr>`;

  const corr = D.corr;
  let corrHtml = `<div style="color:var(--sub);font-size:12px">无后验样本：不画相关矩阵（不伪造）</div>`;
  if (corr && corr.names && corr.matrix) {
    const n = corr.names.length, cell = n > 6 ? 17 : 21;
    let g = `<div class="corr" style="grid-template-columns:${cell * 3.4}px repeat(${n},${cell}px)">`;
    g += `<div></div>` + `<div></div>`.repeat(n);   // 表头留空：行标签+悬浮提示已够
    corr.matrix.forEach((row, i) => {
      g += `<div class="nm">${esc(corr.names[i])}</div>`;
      row.forEach((v, j) => {
        const a = Math.min(1, Math.abs(v));
        const col = v >= 0 ? `rgba(31,111,180,${a.toFixed(2)})` : `rgba(192,57,43,${a.toFixed(2)})`;
        g += `<div class="c" style="background:${i === j ? "#e2e2e2" : col}"
               title="${esc(corr.names[i])} ↔ ${esc(corr.names[j])}：ρ=${fmt(v, 2)}">${i === j || a < 0.45 ? "" : fmt(v, 2).replace("0.", ".")}</div>`;
      });
    });
    g += `</div>`;
    const strong = (corr.strong || []).map(x =>
      `<li><b>${esc(x.a)} ↔ ${esc(x.b)}</b> ρ=${fmt(x.rho, 2)}</li>`).join("");
    const med = (corr.medium || []).map(x =>
      `<li class="info"><b>${esc(x.a)} ↔ ${esc(x.b)}</b> ρ=${fmt(x.rho, 2)}</li>`).join("");
    corrHtml = g + `<div style="font-size:11px;color:var(--sub);margin-top:6px">
      行=列 同序（${n} 个参数）；点格子看 ρ 值。"宽"→ 详情 看完整矩阵与解释</div>
      <ul class="notes-list" style="margin-top:6px">${strong}${med}</ul>`;
  }

  const el2 = `<div style="font-size:11.5px;color:var(--sub);margin-bottom:6px">
    ELPD-LOO ${D.elpd === null ? "—" : fmt(D.elpd, 1)}
    ${(D.comparison && D.elpd_diff !== null)
      ? ` · Δ ${fmt(D.elpd_diff, 1)} ± ${fmt(D.dse, 1)}` : " · 未做跨 K 比较"}
    ${D.candidates_source === "note_text" ? " · 候选表由选择日志文本解析" : ""}
  </div>`;

  box.innerHTML = `<div class="cols">
    <div><h3>K 候选</h3>${el2}
      <table class="t"><thead><tr><th>K</th><th>状态</th><th></th><th>依据</th></tr></thead>
      <tbody>${candRows}</tbody></table>
      <div style="font-size:11px;color:var(--sub);margin-top:8px">
        只有扫过的 K 才在表里；状态 <span class="pill degraded">degraded</span>
        表示采样只到「可用」档，不当作严格通过</div></div>
    <div><h3>参数后验</h3>
      <table class="t"><thead><tr><th>参数</th><th>中位</th><th>5%</th><th>95%</th><th></th></tr></thead>
      <tbody>${pRows}</tbody></table>
      <div style="font-size:11px;color:var(--sub);margin-top:8px">
        分位已是原尺度（ln 参数已换回）；区间是逐参数边际分位，
        不是联合可信域，也不能据此判断两个参数「都不确定」</div></div>
    <div><h3>参数相关性</h3>${corrHtml}</div>
  </div>`;
}

/* ---------------- 抽屉 ---------------- */
function openDrawer(id) {
  OPEN = id;
  const m = META[id] || {};
  $("#dt").textContent = (m.title || id) + " · 详情";
  const box = $("#db");
  if (id === "fit") {
    const w = Math.min(720, box.clientWidth || 700), h = 340;
    const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h });
    svg.innerHTML = renderBode(w, h).body;
    box.innerHTML = `<div class="hint">Bode 幅值（实线）与相位（虚线，右轴）。相位用 −Z″ 口径之外的原始 atan2，注意符号约定。</div>`;
    box.appendChild(svg);
  } else if (id === "resid") {
    box.innerHTML = residDetail();
  } else if (id === "drt") {
    box.innerHTML = drtDetail();
  } else if (id === "model") {
    const w = Math.min(700, box.clientWidth || 680), h = 380;
    const c = D.corr;
    if (!c) { box.innerHTML = `<div style="color:var(--sub)">无后验样本，不画相关矩阵</div>`; }
    else {
      const pairs = (c.strong || []).map(x =>
        `<li><b>${esc(x.a)} ↔ ${esc(x.b)}</b> ρ=${fmt(x.rho, 2)} · ${md(x.why || "")}</li>`)
        .concat((c.medium || []).map(x =>
          `<li class="info"><b>${esc(x.a)} ↔ ${esc(x.b)}</b> ρ=${fmt(x.rho, 2)} · ${md(x.why || "")}</li>`))
        .join("");
      box.innerHTML = `<div class="hint">相关矩阵来自后验样本；|ρ|≥0.9 的两个参数只能引用组合量，
        单独引用任意一个都会被读成「已经定住」，实际没有。</div>`
        + `<div style="text-align:center">${corrSvg(c, Math.min(660, w), h)}</div>`
        + `<h3 style="font-size:12px;color:var(--sub);margin:14px 0 6px">强/中等相关</h3>
           <ul class="notes-list">${pairs || '<li class="info">没有 |ρ|≥0.7 的参数对</li>'}</ul>`;
    }
  } else {
    box.innerHTML = `<pre style="white-space:pre-wrap;font-size:12px">${(D.selection_notes || []).join("\n") || "无"}</pre>`;
  }
  $("#drawer").classList.add("open"); $("#mask").classList.add("open");
}
function closeDrawer() { $("#drawer").classList.remove("open"); $("#mask").classList.remove("open"); }

function residDetail() {
  const c = D.curves;
  const bins = (D.components || []).length;
  let rows = "";
  const seg = [[0, 1], [1, 10], [10, 100], [100, 1e5]];
  seg.forEach(([a, b]) => {
    const idx = c.f.map((v, i) => [v, i]).filter(([v]) => v >= a && v < b).map(([, i]) => i);
    if (!idx.length) return;
    const rr = Math.sqrt(idx.reduce((s, i) => s + c.rre[i] ** 2, 0) / idx.length);
    const ri = Math.sqrt(idx.reduce((s, i) => s + c.rim[i] ** 2, 0) / idx.length);
    const zz = Math.sqrt(idx.reduce((s, i) => s + (c.zre[i] ** 2 + c.zim[i] ** 2), 0) / idx.length);
    rows += `<tr><td>${a}–${b} Hz</td><td class="num">${idx.length}</td>
      <td class="num">${fmt(rr, 4)}</td><td class="num">${fmt(ri, 4)}</td>
      <td class="num">${fmt(100 * Math.hypot(rr, ri) / zz, 2)}%</td></tr>`;
  });
  const m = (D.curves.nre && D.curves.nre.length) ? D.curves.nre.concat(D.curves.nim) : [];
  const n1 = m.filter(v => Math.abs(v) > 1).length, n2 = m.filter(v => Math.abs(v) > 2).length;
  return `<div class="hint">标准化残差用本次拟合的逐点 σ（中位），不套 2% 观测噪声。</div>
    <table class="t"><thead><tr><th>频段</th><th>点数</th><th>RMS Re / Ω</th><th>RMS Im / Ω</th><th>相对 |Z|</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <table class="t" style="margin-top:14px"><thead><tr><th>指标</th><th>值</th><th>说明</th></tr></thead><tbody>
      <tr><td>acf（相邻残差自相关）</td><td class="num">${fmt(D.acf, 3)}</td><td style="color:var(--sub)">阈值 ≤ ${D.thresholds.acf}；对数分布频点，仅作辅助</td></tr>
      <tr><td>|norm| &gt; 1σ</td><td class="num">${n1}</td><td style="color:var(--sub)">共 ${m.length} 个分量点</td></tr>
      <tr><td>|norm| &gt; 2σ</td><td class="num">${n2}</td><td style="color:var(--sub)">单独离群点</td></tr>
      <tr><td>PPC 覆盖率 Re / Im</td><td class="num">${fmt(D.cover.re, 2)} / ${fmt(D.cover.im, 2)}</td><td style="color:var(--sub)">后验预测落在 90% 区间的比例</td></tr>
    </tbody></table>`;
}

function drtDetail() {
  if (!D.drt) return `<div style="color:var(--sub)">无后验 DRT</div>`;
  const rows = (D.components || []).map(c => `<tr><td>#${c.idx}</td>
    <td class="num">${fmt(c.f0, 4)}</td><td class="num">${fmt(c.f0_lo, 4)}–${fmt(c.f0_hi, 4)}</td>
    <td class="num">${fmt(c.w, 3)}</td><td class="num">${fmt(c.phi, 3)}</td>
    <td class="num">${c.p_out === null ? "—" : fmt(c.p_out, 2)}</td></tr>`).join("");
  return `<div class="hint">γ(τ) 由各分量 HN 的解析 DRT 逐样本相加后取分位；φζ&gt;1 的样本 γ 会出负值，此时曲线只作可视化。</div>
    <table class="t"><thead><tr><th>分量</th><th>f₀ / Hz</th><th>90% 区间</th><th>w / Ω</th><th>φ</th><th>P(峰位出窗)</th></tr></thead>
    <tbody>${rows || '<tr><td colspan="6">无分量</td></tr>'}</tbody></table>`;
}

function corrSvg(c, w, h) {
  const n = c.names.length, pad = 100, top = 84;
  const cell = Math.min((w - pad - 20) / n, (h - top - 26) / n);
  const F = [`<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">`];
  F.push(`<text x="${pad}" y="18" font-size="12" fill="${TH.sub}">参数相关性（后验 ρ）</text>`);
  c.names.forEach((nm, j) => {
    const px = (pad + j * cell + cell / 2).toFixed(1);
    F.push(`<text x="${px}" y="${top - 10}" font-size="10" fill="${TH.sub}" text-anchor="start" transform="rotate(-60 ${px} ${top - 10})">${esc(nm)}</text>`);
  });
  c.matrix.forEach((row, i) => {
    F.push(`<text x="${pad - 6}" y="${(top + i * cell + cell / 2 + 3.5).toFixed(1)}" font-size="10" fill="${TH.sub}" text-anchor="end">${c.names[i]}</text>`);
    row.forEach((v, j) => {
      const a = Math.min(1, Math.abs(v));
      const col = i === j ? "#cccccc" : (v >= 0 ? `rgba(31,111,180,${a})` : `rgba(192,57,43,${a})`);
      F.push(`<rect x="${(pad + j * cell).toFixed(1)}" y="${(top + i * cell).toFixed(1)}" width="${cell - 1}" height="${cell - 1}" fill="${col}"/>`);
      if (i !== j && a > 0.45) F.push(`<text x="${(pad + j * cell + cell / 2).toFixed(1)}" y="${(top + i * cell + cell / 2 + 3).toFixed(1)}" font-size="9" fill="#fff" text-anchor="middle">${fmt(v, 2).replace("0.", ".")}</text>`);
    });
  });
  F.push("</svg>");
  return F.join("");
}

/* ---------------- 导出 ---------------- */
function exportPanelSvg(id, name) {
  const card = document.querySelector(`.card[data-id="${id}"]`);
  const svg = card && card.querySelector("svg");
  if (!svg) return;
  const clone = svg.cloneNode(true);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", svg.getAttribute("width") + "px");
  clone.setAttribute("height", svg.getAttribute("height") + "px");
  clone.style.background = "#fff";
  clone.insertBefore(svgEl("rect", { width: "100%", height: "100%", fill: "#fff" }),
                     clone.firstChild);
  const src = new XMLSerializer().serializeToString(clone);
  download(`${name || id}.svg`, src, "image/svg+xml");
}
function download(name, text, mime) {
  const b = new Blob([text], { type: mime || "text/plain;charset=utf-8" });
  const a = el("a", { href: URL.createObjectURL(b), download: name });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
function specForFigure() {
  const map = { fit: "nyquist", drt: "drt", resid: "residual" };
  const panels = LAYOUT.panels.filter(p => p.visible && map[p.id]).map(p => map[p.id]);
  const n = panels.length;
  return { spec_version: SPEC_VER, theme: LAYOUT.theme,
    canvas: { width_mm: n >= 4 ? 180 : n === 3 ? 180 : 90, height_mm: n >= 4 ? 110 : 70 },
    layout: n >= 4 ? "quad" : n === 3 ? "triple" : n === 2 ? "pair" : "single",
    panels, labels: "en", dpi: 600, formats: ["pdf", "svg", "tiff"] };
}
function openExport() {
  const spec = specForFigure();
  const cmd = `python bm_figure.py "<sample_dir>" --spec figure_spec.json --all`;
  $("#dt").textContent = "导出";
  $("#db").innerHTML = `
    <div class="hint">图形文件由 <code>bm_figure.py</code> 用**同一份数据**重画，
    保证矢量图与 CSV 数值一致；这里的 spec 就是它的输入。</div>
    <h3 style="font-size:12px;color:var(--sub)">PlotSpec（当前面板选择）</h3>
    <pre style="font-size:11.5px;background:#fafafa;padding:10px;border-radius:8px;overflow:auto">${JSON.stringify(spec, null, 2)}</pre>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin:10px 0">
      <button id="ex-spec" class="primary">下载 figure_spec.json</button>
      <button id="ex-layout">下载布局 JSON</button>
      <button id="ex-data">下载数据 JSON</button>
      <button id="ex-cmd">复制命令</button>
    </div>
    <div class="hint">命令行：<code>${cmd.replace(/</g, "&lt;")}</code></div>
    <h3 style="font-size:12px;color:var(--sub);margin-top:14px">单面板矢量图</h3>
    <div class="hint">直接导出当前渲染的 SVG（含缩放与轴），可直接进 Illustrator / Inkscape 编辑。</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      ${LAYOUT.panels.filter(p => p.visible && META[p.id] && ["fit", "drt", "resid"].includes(p.id)).map(p => `<button data-svg="${p.id}">${META[p.id].title} SVG</button>`).join("")}
    </div>`;
  $("#drawer").classList.add("open"); $("#mask").classList.add("open");
  $("#ex-spec").onclick = () => download("figure_spec.json", JSON.stringify(spec, null, 2), "application/json");
  $("#ex-layout").onclick = () => download("dashboard_layout.json", JSON.stringify(LAYOUT, null, 2), "application/json");
  $("#ex-data").onclick = () => download("payload.json", JSON.stringify(D, null, 2), "application/json");
  $("#ex-cmd").onclick = () => { navigator.clipboard && navigator.clipboard.writeText(cmd); };
  $("#db").addEventListener("click", e => {
    const id = e.target.dataset.svg;
    if (id) exportPanelSvg(id, (META[id] || {}).title);
  });
}

/* ---------------- 布局持久化 ---------------- */
function saveLayout() {
  try { localStorage.setItem(LS_KEY, JSON.stringify(LAYOUT)); } catch (e) {}
}
function loadLayout() {
  try {
    const s = localStorage.getItem(LS_KEY);
    if (!s) return;
    const o = JSON.parse(s);
    if (o && o.panels && o.panels.length) {
      LAYOUT = o;
      // 与预置面板对齐：只保留已知 id，顺序按保存的来
      LAYOUT.panels = LAYOUT.panels.filter(p => META[p.id]);
      Object.keys(META).forEach(id => {
        if (!LAYOUT.panels.find(p => p.id === id))
          LAYOUT.panels.push({ id, span: 6, visible: false });
      });
    }
  } catch (e) {}
}

/* ---------------- 顶栏与 KPI ---------------- */
function applyTheme(name) {
  TH = THEMES[name] || THEMES.minimal;
  const r = document.documentElement.style;
  r.setProperty("--plotfont", TH.plotFont);
  r.setProperty("--accent", TH.fit);
  drawAll();
}

function renderChrome() {
  $("#ttl").textContent = D.name;
  $("#sub").textContent = `${D.n} 点 · ${sci(D.f_min)}–${sci(D.f_max)} Hz · K=${D.K}`
    + (D.include_L ? " · +L" : "");
  const v = D.verdict;
  const lv = v.level === "ok" ? "ok" : v.level === "bad" ? "bad" : "warn";
  $("#chips").innerHTML =
    `<span class="chip ${lv}">${v.label}</span>
     <span class="chip">K=${D.K}${D.include_L ? " +L" : ""}</span>
     <span class="chip">Rwp ${D.rwp === null ? "—" : fmt(D.rwp, 2) + "%"}</span>
     <span class="chip">acf ${fmt(D.acf, 2)}</span>`;
  const k = D.checks.find(c => c.key === "rhat"), dv = D.checks.find(c => c.key === "div");
  const cell = (label, val, hint, cls) =>
    `<div class="kpi ${cls || ""}"><div class="k">${label}</div>
      <div class="v">${val}</div><div class="t">${hint || ""}</div></div>`;
  $("#kpis").innerHTML =
    cell("Rwp", D.rwp === null ? "—" : fmt(D.rwp, 2) + "%", "相对加权残差（不除 √2）",
      D.rwp === null ? "" : (D.rwp < 2 ? "pass" : D.rwp < 10 ? "warn" : "fail")) +
    cell("残差 acf", fmt(D.acf, 2), `阈值 ≤ ${D.thresholds.acf}`,
      D.acf === null ? "" : (D.acf <= D.thresholds.acf ? "pass" : "fail")) +
    cell("max R̂", k && k.value !== null ? fmt(k.value, 3) : "—",
      `阈值 ≤ ${D.thresholds.rhat}`,
      !k || k.value === null ? "" : (k.state === "pass" ? "pass" : "fail")) +
    cell("divergence", dv && dv.value !== null ? fmt(dv.value, 0) : "—",
      "须为 0", !dv || dv.value === null ? "" : (dv.state === "pass" ? "pass" : "fail")) +
    cell("PPC 覆盖 Re/Im", `${fmt(D.cover.re, 2)}/${fmt(D.cover.im, 2)}`,
      "90% 区间内的观测占比") +
    cell("采样分级", D.verdict.grade === null ? "—" : D.verdict.grade,
      D.verdict.sampling || "", D.verdict.grade === 2 ? "pass" : "warn");

  $("#foot").innerHTML =
    `数据源：<code>${D.name}</code> · 生成器 bm_dash.py · spec ${SPEC_VER}<br>
     Rwp 定义：${D.rwp_def}<br>
     ${D.band_note}<br>
     ${D.sigma_note}<br>
     判定说明：采样分级（grade）与结构判据（acf / R̂ / div）分开陈述；
     <b>indistinguishable 不等于存在唯一最优模型</b>；缺失后验时不给相关矩阵。`;
}

/* ---------------- 启动 ---------------- */
function boot() {
  const sel = $("#theme");
  Object.keys(THEMES).forEach(k => {
    const o = el("option", { value: k }); o.textContent = THEMES[k].label;
    if (k === LAYOUT.theme) o.selected = true; sel.appendChild(o);
  });
  sel.onchange = () => { LAYOUT.theme = sel.value; saveLayout(); applyTheme(sel.value); };
  $("#btn-reset").onclick = () => {
    LAYOUT = JSON.parse(JSON.stringify(DEFAULT_LAYOUT));
    ZOOM = {}; PIN = null; delete LAYOUT.version;
    saveLayout(); buildCards(); applyTheme(LAYOUT.theme);
  };
  $("#btn-exp").onclick = openExport;
  $("#dclose").onclick = closeDrawer;
  $("#mask").onclick = closeDrawer;
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeDrawer(); });
  renderChrome();
  applyTheme(LAYOUT.theme);
  const mDeep = /detail=([a-z]+)/.exec(location.hash || "");
  if (mDeep && META[mDeep[1]]) openDrawer(mDeep[1]);
  let t = null;
  addEventListener("resize", () => { clearTimeout(t); t = setTimeout(drawAll, 120); });
}
loadLayout();
boot();
buildCards();
drawAll();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
