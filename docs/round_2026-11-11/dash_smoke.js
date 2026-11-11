// 目标：在 node 里用一个极简 DOM 桩把 dashboard 的脚本整体跑一遍。
// 两类检查：
//   ① 渲染冒烟——四个面板都产出非空 SVG，不含 NaN/undefined；
//   ② 交互契约——跨图频点映射、分页、Lin-KK 四态、Rwp 不配色。
// 不是视觉回归（那用无头 Edge 截图），只抓硬错误与口径回归。
const fs = require("fs");
const vm = require("vm");

const REG = {};                       // 按选择器缓存，$("#grid") 才是同一个节点
function mkEl(tag) {
  const e = {
    tagName: tag, style: {}, dataset: {}, children: [],
    textContent: "", className: "", clientWidth: 640, clientHeight: 300,
    setAttribute(k, v) { this[k] = v; },
    getAttribute(k) { return this[k]; },
    appendChild(c) { this.children.push(c); return c; },
    insertBefore(c) { this.children.unshift(c); return c; },
    addEventListener() {}, removeEventListener() {}, remove() {},
    querySelector() { return mkEl("div"); },
    querySelectorAll() { return []; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 640, height: 300 }; },
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    // offsetHeight 给左栏 sticky 定位用（syncHdr 读它算 --hdrtop）
    offsetHeight: 96, offsetWidth: 640,
    cloneNode() { return mkEl(tag); },
    setPointerCapture() {}, focus() {}, click() {},
  };
  // innerHTML="" 必须真的清空子节点，否则 buildCards 反复调用会把卡片越堆越多，
  // "这一页有几张卡"这类断言就白写了。
  let html = "";
  Object.defineProperty(e, "innerHTML", {
    get() { return html; },
    set(v) { html = v; if (v === "") e.children.length = 0; },
  });
  return e;
}
function node(sel) {
  if (!REG[sel]) REG[sel] = mkEl(sel.startsWith("#") ? "div" : "section");
  return REG[sel];
}

global.document = {
  querySelector: node,
  querySelectorAll: () => [],
  createElement: t => mkEl(t),
  createElementNS: (ns, t) => mkEl(t),
  documentElement: { style: { setProperty() {} } },
  addEventListener() {},
  body: mkEl("body"),
};
global.localStorage = {
  _m: {},
  getItem(k) { return this._m[k] || null; },
  setItem(k, v) { this._m[k] = v; },
};
global.innerWidth = 1440;
global.location = { hash: "", href: "file:///x.html" };
global.addEventListener = () => {};

const src = fs.readFileSync(process.argv[2], "utf8");
vm.runInThisContext(src, { filename: "dashboard.js" });

let bad = 0;
const pass = (n, extra) => console.log(`PASS  ${n}${extra ? "  " + extra : ""}`);
const fail = (n, why) => { console.log(`FAIL  ${n}  ${why}`); bad++; };

/* ---------- ① 渲染冒烟 ---------- */
const cases = [
  ["renderFit", () => renderFit(640, 400), "<circle"],
  ["renderDrt", () => renderDrt(520, 268), "<path"],
  ["renderResid", () => renderResid(640, 268), "<circle"],
  ["renderBode", () => renderBode(700, 340), "<circle"],
  ["renderReIm", () => renderReIm(700, 340), "<circle"],
];
for (const [name, fn, must] of cases) {
  try {
    const r = fn();
    const body = r && r.body ? r.body : "";
    const ok = body.length > 200 && body.includes(must) &&
      !/NaN|undefined|null/.test(body.replace(/fill-opacity="0\.\d+"/g, ""));
    ok ? pass(name, `${body.length} chars`)
       : fail(name, `must=${must}? ${body.includes(must)} len=${body.length}`);
  } catch (e) { fail(name, e.message); }
}

/* ---------- ② 跨图频点映射（评审 P1） ---------- */
// DRT 网格点远多于实测频点，且两组网格不重合——这正是原来下标串味的场景
try {
  const n0 = D.curves.f.length;
  const drtN = 400;
  const g = Array.from({ length: drtN }, (_, i) => Math.pow(10, -1 + 6 * i / (drtN - 1)));
  D.drt = { f: g, med: g.map(() => 0.01), lo: g.map(() => 0.008), hi: g.map(() => 0.012) };
  const probe = g[137];                       // 远大于 n0 的下标
  PIN = null;
  togglePin(probe);
  if (PIN !== probe) throw new Error("togglePin 没记住频率值");
  const ci = pinIdx("fit"), di = pinIdx("drt");
  if (!(ci >= 0 && ci < n0)) throw new Error(`Nyquist 下标越界 ${ci}/${n0}`);
  if (!(di >= 0 && di < drtN)) throw new Error(`DRT 下标越界 ${di}/${drtN}`);
  const err = Math.abs(D.curves.f[ci] - probe);
  let bestErr = Infinity;
  for (const v of D.curves.f) bestErr = Math.min(bestErr, Math.abs(v - probe));
  if (err > bestErr + 1e-12) throw new Error("没有映射到最近实测频点");
  // 三个面板都要能在 PIN 来自 DRT 网格时正常出图（原来会读到 c.zre[137] = undefined）
  for (const [nm, fn] of [["fit", () => renderFit(640, 400)],
                          ["resid", () => renderResid(640, 268)],
                          ["drt", () => renderDrt(520, 268)]]) {
    const body = fn().body;
    if (/undefined|NaN/.test(body)) throw new Error(`${nm} 出图带 undefined/NaN`);
  }
  renderReadout(probe);                       // 读数不能抛
  togglePin(probe);                           // 再点同一个频点应取消固定
  if (PIN !== null) throw new Error("第二次点击没有取消固定");
  pass("freq-map", `probe=${probe.toFixed(3)}Hz → 实测 idx ${ci}`);
} catch (e) { fail("freq-map", e.message); }

/* ---------- ③ Lin-KK 四态 ---------- */
try {
  const want = { ok: "pass", warn: "warn", fail: "fail", unavailable: "na" };
  const txt = { ok: "通过", warn: "警告", fail: "未通过", unavailable: "不可评价" };
  for (const [st, cls] of Object.entries(want)) {
    const keep = D.kk;
    D.kk = { status: st, reason: "测试" };
    renderSide();                            // 判定块现在在左控制台，不在卡片里
    const h = node("#side").innerHTML;
    if (!h.includes(`chk ${cls}`)) throw new Error(`${st} 没映射到 ${cls}`);
    if (!h.includes(txt[st])) throw new Error(`${st} 文案缺 ${txt[st]}`);
    D.kk = keep;
  }
  pass("kk-state", "ok/warn/fail/unavailable → pass/warn/fail/na");
} catch (e) { fail("kk-state", e.message); }

/* ---------- ④ Rwp 不配色 ---------- */
try {
  renderChrome();
  const k = node("#kpis").innerHTML;
  const cell = k.split('<div class="kpi').find(s => s.includes("Rwp"));
  if (!cell) throw new Error("KPI 里没有 Rwp");
  if (/kpi (pass|warn|fail)/.test(cell)) throw new Error("Rwp 仍在用红黄绿");
  pass("rwp-neutral", "无预注册阈值 → 不上色");
} catch (e) { fail("rwp-neutral", e.message); }

/* ---------- ⑤ 分页 ---------- */
try {
  setPage("ov");
  const nOv = node("#grid").children.length;
  setPage("md");
  const nMd = node("#grid").children.length;
  if (nOv !== 3) throw new Error(`分析页应 3 张图卡，实际 ${nOv}`);
  if (nMd !== 1) throw new Error(`数据表页应只有 model 一张卡，实际 ${nMd}`);
  if (node("#kpis").style.display !== "none") throw new Error("表格页没收起指标条");
  setPage("ov");
  // 主图卡必须带投影 tab，否则"同一份 Z(f) 三种画法"这个入口就丢了
  const fitCard = node("#grid").children.find(c => c["data-id"] === "fit");
  if (!fitCard || !fitCard.innerHTML.includes('data-mtab="bode"'))
    throw new Error("主图缺 Bode / Re&Im 投影 tab");
  pass("pages", `ov ${nOv} 图卡 / md ${nMd} 表卡`);
} catch (e) { fail("pages", e.message); }

/* ---------- ⑥ 表格面板与 spec ---------- */
try {
  const box2 = mkEl("div");
  drawModel(box2);
  const h = box2.innerHTML;
  if (!(h.includes("K 候选") && h.includes("参数后验") && h.length > 800))
    throw new Error("drawModel 内容不完整");
  const s = specForFigure();
  if (!s.style || !s.style.fit) throw new Error("spec 没带配色（图与面板会不一致）");
  pass("text+spec", `${JSON.stringify(s.panels)} layout=${s.layout}`);
} catch (e) { fail("text+spec", e.message); }

/* ---------- ⑦ 导出抽屉（配色输入 + spec 预览） ---------- */
try {
  openExport();
  const h = node("#db").innerHTML;
  if (!h.includes('data-c="fit"')) throw new Error("没有配色输入");
  if (!h.includes("figure_spec.json")) throw new Error("没有 spec 预览/下载");
  pass("export-drawer", "5 个配色槽 + spec");
} catch (e) { fail("export-drawer", e.message); }

console.log(bad ? `\n${bad} 项失败` : "\n全部通过");process.exit(bad ? 1 : 0);
