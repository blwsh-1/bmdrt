// 目标：在 node 里用一个极简 DOM 桩把 dashboard 的脚本整体跑一遍，
// 确认 boot() 不抛异常、四个面板都能产出非空 SVG。
// 这不是视觉回归，只是抓「未定义变量 / 数学 NaN / 空图」这类硬错误。
const fs = require("fs");
const vm = require("vm");

function mkEl(tag) {
  const e = {
    tagName: tag, style: {}, dataset: {}, children: [], innerHTML: "",
    textContent: "", className: "", clientWidth: 640, clientHeight: 300,
    setAttribute(k, v) { this[k] = v; },
    getAttribute(k) { return this[k]; },
    appendChild(c) { this.children.push(c); return c; },
    insertBefore(c) { this.children.unshift(c); return c; },
    addEventListener() {}, removeEventListener() {}, remove() {},
    querySelector() { return mkEl("div"); },
    querySelectorAll() { return []; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 640, height: 300 }; },
    classList: { add() {}, remove() {}, contains() { return false; } },
    cloneNode() { return mkEl(tag); },
    setPointerCapture() {}, focus() {}, click() {},
  };
  return e;
}

global.document = {
  querySelector: () => mkEl("div"),
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
global.location = { hash: '', href: 'file:///x.html' };
global.addEventListener = () => {};

const src = fs.readFileSync(process.argv[2], "utf8");
vm.runInThisContext(src, { filename: "dashboard.js" });

const cases = [
  ["renderFit", () => renderFit(640, 400), "<circle"],
  ["renderDrt", () => renderDrt(520, 268), "<path"],
  ["renderResid", () => renderResid(640, 268), "<circle"],
  ["renderBode", () => renderBode(700, 340), "<circle"],
];
let bad = 0;
for (const [name, fn, must] of cases) {
  try {
    const r = fn();
    const body = r && r.body ? r.body : "";
    const ok = body.length > 200 && body.includes(must) &&
      !/NaN|undefined|null/.test(body.replace(/fill-opacity="0\.\d+"/g, ""));
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}  ${body.length} chars` +
      (ok ? "" : `  (包含 ${must}? ${body.includes(must)})`));
    if (!ok) bad++;
  } catch (e) {
    console.log(`FAIL  ${name}  ${e.message}`);
    bad++;
  }
}
// 表格/文本面板
try {
  const box = mkEl("div");
  drawNotes(box);
  console.log(`PASS  drawNotes  ${box.innerHTML.length} chars`);
  const box2 = mkEl("div");
  drawModel(box2);
  const h = box2.innerHTML;
  const ok = h.includes("K 候选") && h.includes("参数后验") && h.length > 800;
  console.log(`${ok ? "PASS" : "FAIL"}  drawModel  ${h.length} chars`);
  if (!ok) bad++;
} catch (e) {
  console.log(`FAIL  text panels  ${e.message}`);
  bad++;
}
// 抽屉与 spec
try {
  const s = specForFigure();
  console.log(`PASS  specForFigure  ${JSON.stringify(s.panels)} layout=${s.layout}`);
} catch (e) {
  console.log(`FAIL  specForFigure  ${e.message}`);
  bad++;
}
console.log(bad ? `\n${bad} 项失败` : "\n全部通过");
process.exit(bad ? 1 : 0);
