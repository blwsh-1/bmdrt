# -*- coding: utf-8 -*-
"""③ 第 3 步「正式代码审查」静态可复现核对（只读）。

审查对象固定为**远端提交 8015721 的实际 blob**，不是本地工作树。
本脚本：
  · 不 import 探针 / 产出层 / 生产模块（纯 ast + 文本 + `git cat-file`）；
  · 不跑拟合、不写产物、不改任何文件、不构造任何身份号；
  · 只做「远端字节 == 本地字节」与机械断言，结论供
    `BUDGET_CHAIN_PROBE_REVIEW.md` 引用。

    python docs/round_2026-11-15/budget_chain_probe_review.py
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
import warnings
from pathlib import Path

# nested_fit.py:42 既有注释里的 \S 在 ast.parse 时会报 SyntaxWarning；与审查无关。
warnings.filterwarnings("ignore", category=SyntaxWarning)

ROOT = Path(__file__).resolve().parents[2]
SUB = Path("docs/round_2026-11-15")
PROBE = SUB / "budget_chain_probe.py"
REPORT = SUB / "BUDGET_CHAIN_PROBE_REPORT.md"
BASE = "8015721ddfb8157edc4712723294d0d7d085ac13"
PARENT = "a4a4c595b95186d32fdbe7f3d7c7775087b3a1e6"


def git(*args: str) -> bytes:
    return subprocess.run(("git", *args), cwd=str(ROOT),
                          capture_output=True, check=True).stdout


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def text(rel: Path) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def parse(rel: Path) -> ast.Module:
    return ast.parse(text(rel), filename=str(rel))


def self_defined(tree: ast.Module) -> set[str]:
    return {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def assign_const(tree: ast.Module, name: str):
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    return None


class Ledger:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str]] = []

    def ck(self, group: str, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((group, name, bool(ok), detail))

    def show(self) -> int:
        for g, n, ok, d in self.rows:
            print(f"  [{'PASS' if ok else 'FAIL'}] {g:<5} {n}" + (f"  — {d}" if d else ""))
        bad = [r for r in self.rows if not r[2]]
        print(f"\n  机械断言 {len(self.rows)} 条，FAIL {len(bad)} 条")
        return 0 if not bad else 1


def main() -> int:
    L = Ledger()
    pt = parse(PROBE)
    rt = text(REPORT)
    bt = parse(Path("budget_params.py"))

    # ---------------------------------------------------------------- A 基线
    head = git("rev-parse", "origin/main").decode().strip()
    # 审查报告自身会推进 main，故不能断言「当前 main == BASE」：
    # 审查对象是 BASE 这个提交对象及其 blob，只要求它是当前 main 的祖先。
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", BASE, "origin/main"),
        cwd=str(ROOT), capture_output=True).returncode == 0
    L.ck("A", "8015721 是 origin/main 的祖先（审查报告提交后仍成立）",
         ancestor, f"origin/main={head[:8]}")
    L.ck("A", "8015721 提交对象可读",
         git("cat-file", "-t", BASE).decode().strip() == "commit")
    parent = git("rev-parse", f"{BASE}^").decode().strip()
    L.ck("A", "父提交 = a4a4c59（第 2 步产出层）", parent == PARENT, parent[:8])
    changed = sorted(git("diff", "--name-only", PARENT, BASE).decode().split())
    expect = sorted([PROBE.as_posix(), REPORT.as_posix(), "docs/INDEX.md"])
    L.ck("A", "提交仅 3 文件、零生产改动", changed == expect, ", ".join(changed))
    for rel in (PROBE, REPORT, Path("budget_params.py")):
        oid = git("ls-tree", BASE, "--", rel.as_posix()).decode().split()[2]
        blob = git("cat-file", "-p", oid)
        L.ck("A", f"8015721 的 blob 字节对齐 {rel.name}",
             sha(blob) == sha((ROOT / rel).read_bytes()), oid[:8])
    # INDEX 会被本审查报告自身追加条目 ⇒ 不能与本地字节比；改核 BASE 相对父提交的单行 diff。
    idx_old, idx_new = (git("show", f"{PARENT}:docs/INDEX.md"),
                        git("show", f"{BASE}:docs/INDEX.md"))
    L.ck("A", "8015721 的 INDEX 相对父提交仅追加第 3 步条目",
         idx_old != idx_new and b"budget_chain_probe.py" in idx_new
         and b"budget_chain_probe.py" not in idx_old,
         f"parent={len(idx_old)}B, base={len(idx_new)}B")

    # ------------------------------------------------------------ B 独立性
    bad: list[tuple[str, int]] = []
    for n in ast.walk(pt):
        names: list[str] = []
        if isinstance(n, ast.Import):
            names = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            names = [n.module or ""]
        elif isinstance(n, ast.Name):
            names = [n.id]
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            names = [n.value.id]
        if any(x.split(".")[0] == "budget_params" for x in names):
            bad.append((type(n).__name__, getattr(n, "lineno", -1)))
    L.ck("B", "探针 AST 零产出层标识符引用", not bad, str(bad))

    dyn = [(getattr(n, "lineno", -1), getattr(n.func, "id", ""))
           for n in ast.walk(pt) if isinstance(n, ast.Call)
           and isinstance(n.func, ast.Name)
           and n.func.id in {"__import__", "import_module"}]
    L.ck("B", "探针无动态导入（字符串形式）", not dyn, str(dyn))

    api = {n.name for n in bt.body if isinstance(n, ast.FunctionDef)}
    called = {n.func.id for n in ast.walk(pt)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    overlap = (api & called) - self_defined(pt)
    L.ck("B", "探针未调用产出层公开函数", not overlap,
         f"产出层公开函数 {len(api)} 个，交集 {sorted(overlap)}")

    prod_imports: list[str] = []
    for p in sorted((ROOT).glob("*.py")):
        if p.name.startswith("test_"):
            continue
        try:
            t = ast.parse(p.read_text(encoding="utf-8"), filename=p.name)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for n in ast.walk(t):
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                mods = [n.module or ""]
            else:
                continue
            if any(m.split(".")[0] == "budget_params" for m in mods):
                prod_imports.append(p.name)
    L.ck("B", "顶层生产/工具 .py 零产出层 import（探针模块级 import 链不污染）",
         not prod_imports, str(prod_imports))

    ei = parse(Path("exec_identity.py"))
    L.ck("B", "既有身份层不 import 产出层",
         all(not (isinstance(n, ast.Import) and
                  any(a.name.split(".")[0] == "budget_params" for a in n.names))
             for n in ast.walk(ei)))
    L.ck("B", "探针五键与产出层 BUDGET_KEYS 静态对齐",
         tuple(assign_const(bt, "BUDGET_KEYS") or ()) == tuple(assign_const(pt, "KEYS") or ()),
         str(assign_const(pt, "KEYS")))
    L.ck("B", "sys.modules 差集护栏存在",
         "before = set(sys.modules)" in text(PROBE) and "tainted" in text(PROBE))
    L.ck("B", "辅助核验 import 的是 exec_identity 而非产出层",
         "import exec_identity as E" in text(PROBE))

    # -------------------------------------------------------------- C 埋点
    wraps_node = next((n for n in ast.walk(pt) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "WRAPS"
                               for t in n.targets)), None)
    wraps_items = (wraps_node.value.elts if wraps_node is not None
                   and isinstance(wraps_node.value, (ast.Tuple, ast.List)) else [])
    wrap_names = [e.elts[1].value for e in wraps_items
                  if isinstance(e, (ast.Tuple, ast.List)) and len(e.elts) > 1
                  and isinstance(e.elts[1], ast.Constant)]
    L.ck("C", "WRAPS 恰 5 个生产函数站点",
         len(wraps_items) == 5 and len(wrap_names) == 5, ", ".join(wrap_names))
    sel, real, nf = (text(Path("bm_select.py")), text(Path("bm_real.py")),
                     text(Path("nested_fit.py")))
    L.ck("C", "生产确以模块属性调用（wrapper 拦得到）",
         "R.fit_one(" in sel and "B.fit(" in real
         and "T.fit_top(" in sel and "T.fit_top(" in nf)
    core, top = parse(Path("bm_core.py")), parse(Path("bm_top.py"))

    def mcmc_kw(tree: ast.Module) -> set[str]:
        out: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "MCMC":
                out |= {k.arg for k in n.keywords if k.arg}
        return out

    L.ck("C", "bm_core.fit 构造点全用关键字实参",
         {"num_warmup", "num_samples", "num_chains", "progress_bar"} <= mcmc_kw(core),
         str(sorted(mcmc_kw(core))))
    L.ck("C", "bm_top.fit_top 构造点全用关键字实参",
         {"num_warmup", "num_samples", "num_chains", "progress_bar"} <= mcmc_kw(top),
         str(sorted(mcmc_kw(top))))
    L.ck("C", "类级 hook 覆盖 MCMC.__init__ / MCMC.run / NUTS / HMC",
         all(s in text(PROBE) for s in
             ('MCMC.__init__ =', 'MCMC.run =', 'NUTS.__init__ =', 'HMC.__init__ =')))
    L.ck("C", "类钩用 functools.wraps 保住签名",
         text(PROBE).count("functools.wraps") >= 4)
    L.ck("C", "close() 还原类属性与模块函数",
         'self._keep_cls["init"]' in text(PROBE) and "self._keep_mod.items()" in text(PROBE))
    L.ck("C", "报告明账 C4/C7(b) 等价展开",
         "等价展开" in rt and "未调用" in rt)

    # ---------------------------------------------------------- D 注入边界
    micro = assign_const(pt, "MICRO") or {}
    L.ck("D", "MICRO 恰两项且值为 8",
         set(micro) == {"num_warmup", "num_samples"} and set(micro.values()) == {8},
         str(micro))
    init_fn = next((n for n in ast.walk(pt)
                    if isinstance(n, ast.FunctionDef) and n.name == "mcmc_init"), None)
    mutating: list[str] = []
    if init_fn:
        for n in ast.walk(init_fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr in {"update", "setdefault", "pop", "clear", "__setitem__"}:
                mutating.append(f"{n.func.attr}@{n.lineno}")
            if isinstance(n, (ast.Assign, ast.AugAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, ast.Subscript):
                        mutating.append(f"subscript@{getattr(t, 'lineno', -1)}")
    upd_lines = [n.lineno for n in ast.walk(init_fn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "update"] \
        if init_fn else []
    L.ck("D", "类边界对 kw 的唯一改动是 eff.update(MICRO)",
         len(upd_lines) == 1 and mutating == [f"update@{upd_lines[0]}"],
         f"mutating={mutating}, update_lines={upd_lines}")
    L.ck("D", "ctor_passed 在注入前赋值（取证用注入前值）",
         text(PROBE).index("outer.ctor_passed =") < text(PROBE).index("eff.update(MICRO)"))
    L.ck("D", "内核 hook 直接调原函数且不改 kw",
         "return orig(obj, model, **kw)" in text(PROBE))
    L.ck("D", "finish() 断言注入差异 ⊆ 两项 + 未被注入三项交叉",
         'diff <= {"n_warmup", "n_samples"}' in text(PROBE)
         and 'for k in ("num_chains", "target_accept", "max_tree_depth")' in text(PROBE))
    L.ck("D", "C8 的转发层篡改只挂 bm_core.fit、只改 n_warmup",
         'mod is BC and name == "fit"' in text(PROBE)
         and 'lambda kw: {**kw, "n_warmup": 16}' in text(PROBE))

    # ------------------------------------------------- E 八条 case ↔ 报告
    cases = sorted(n.name for n in pt.body
                   if isinstance(n, ast.FunctionDef) and n.name.startswith("case_C"))
    L.ck("E", "恰 8 个 case_C* 函数", len(cases) == 8, ", ".join(cases))
    L.ck("E", "main() 汇总 8 条 case + 1 条辅助核验",
         sum(1 for l in text(PROBE).splitlines() if 'results.append((' in l) == 9)
    L.ck("E", "退出码由全部结果决定",
         "ok = all(v for _, v in results)" in text(PROBE)
         and "return 0 if ok else 1" in text(PROBE))
    rows = [l for l in rt.splitlines()
            if l.startswith("| C") and len(l) > 3 and l[3].isdigit()]
    L.ck("E", "报告 §3 表格恰 8 行 case", len(rows) == 8, str(len(rows)))
    for i, c in enumerate(cases, 1):
        L.ck("E", f"{c} 在报告 §3 有对应行", f"| C{i} |" in rt)
    for token in ("600 / 1200 / 1 / 0.9 / 10", "2000 / 1000 / 2 / 0.9 / 10",
                  "**0.95** / **13**", "**0.98**", "KeyError('max_tree_depth')",
                  "记 **2000**", "**16**"):
        L.ck("E", f"报告含关键实测 {token}", token in rt)
    L.ck("E", "报告明账辅助核验不是传递链证据",
         "辅助核验（不是传递链证据）" in rt and "不构造任何号" in rt)
    L.ck("E", "报告保留 upstream_origin_captured=False 结论",
         "`upstream_origin_captured` **保持 `False`**" in rt)

    print("=" * 96)
    print("③ 第 3 步正式代码审查：静态可复现核对（对象 = 远端 8015721 的 blob）")
    print("=" * 96)
    rc = L.show()

    # ------------------------------------------------ 审查发现（静态证据，不算 FAIL）
    print("\n审查发现（文档/实现对应性，均为只读证据，不修改被审对象）")
    print(f"  · F1 探针定义的方法 = {sorted(n.name for n in ast.walk(pt) if isinstance(n, ast.FunctionDef) and n.name in {'reset','close','begin','finish'})}")
    print(f"    报告 §2 写「每条 case 都 reset()、close()」；实际无 reset 方法，"
          f"close() 调用次数 = {text(PROBE).count('p.close()')}（main 的 finally 一次）")
    empty_marks = sum(1 for l in text(PROBE).splitlines()
                      if "p.mark(" in l and ("{}," in l or "{})" in l))
    print(f"  · F2 ①–③ 层标注：探针 {empty_marks} 处 p.mark 传空 dict；"
          "③ 覆盖三行是照抄 bm_real.py:2771/2774 的等价复现，非生产入口埋点")
    print(f"  · F3 报告 §3 含扩大化措辞「五项总是被生产 fit 显式传给」= "
          f"{'五项**总是**被生产 fit 显式传给' in rt}；K3 未纳入 = "
          f"{'m35_cond_n2.py:116' in text(Path('sampler_capture.py'))}")
    print(f"  · F4 探针全文出现 budget_hash_payload 次数 = "
          f"{text(PROBE).count('budget_hash_payload')}（报告 §0 引用的是第 2 步结论，"
          "本轮只调 E._budget_payload）")
    print(f"  · F5 chain_method 未登记 RUN_ID_CONFIG_KEYS = "
          f"{'\"chain_method\"' not in text(Path('exec_identity.py'))}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
