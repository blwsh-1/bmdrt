# -*- coding: utf-8 -*-
"""只读审计：为 `EXECUTION_IDENTITY_CONTRACT.md` 的三个入口算**静态导入闭包**。

为什么需要这个脚本：契约里 `declared_source_set` 的分母不能靠人手列目录
（F2：入口脚本同字节、库模块一份副本都没有）。这里用 AST 从入口出发求闭包，
把「模块级导入」「函数体内导入（含 try/except 内的延迟导入）」分开报，
因为后者正是 `sys.modules` 在拟合前看不到的那一类。

不写任何文件、不改 `evidence/`、不 import 被审计的项目模块（只解析源码）。
    python docs/round_2026-11-13/audit_source_profiles.py
    python docs/round_2026-11-13/audit_source_profiles.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: 三个入口。名字对着契约 §3.3.4 的 Profile A/B/C。
PROFILES = {
    "A": {"label": "bm_real / Voigt 单族（fit_one）", "entries": ["bm_real"]},
    "B": {"label": "nested / bm_top（fit_top）", "entries": ["nested_fit"]},
    "C": {"label": "跨族比较 bm_select（fit_voigt + fit_nested）",
          "entries": ["bm_select"]},
}

DYNAMIC = ("importlib", "__import__", "exec(", "eval(")
RESOURCE_CALLS = ("open", "loadtxt", "genfromtxt", "read_csv", "json.load",
                  "np.load", "imread")


def _is_project(mod: str) -> bool:
    if not mod or mod.startswith("."):
        return False
    top = mod.split(".")[0]
    return (ROOT / f"{top}.py").is_file() or (ROOT / top / "__init__.py").is_file()


def _scan(mod_name: str) -> dict:
    """解析单个模块，返回导入（按位置分层）与动态/资源线索。"""
    path = ROOT / f"{mod_name}.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    imports, func_imports = [], []
    dynamic, resources = [], []

    def visit(node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                visit(child, depth + 1)
                continue
            if isinstance(child, ast.Import):
                for a in child.names:
                    bucket = func_imports if depth else imports
                    bucket.append({"module": a.name, "line": child.lineno})
            elif isinstance(child, ast.ImportFrom):
                bucket = func_imports if depth else imports
                bucket.append({"module": child.module or "", "line": child.lineno,
                               "names": [a.name for a in child.names]})
            elif isinstance(child, ast.Call):
                fn = child.func
                name = getattr(fn, "attr", None) or getattr(fn, "id", None)
                label = f"{name}()" if name else ""
                if label in ("import_module()", "__import__()") or label in DYNAMIC:
                    dynamic.append({"call": label, "line": child.lineno})
                if label and any(r in label for r in RESOURCE_CALLS):
                    seg = ast.get_source_segment(src, child) or ""
                    if "__file__" in seg or ".json" in seg or ".npz" in seg:
                        resources.append({"call": label, "line": child.lineno,
                                          "snippet": seg[:90].replace("\n", " ")})
            visit(child, depth)

    visit(tree, 0)
    return {"module": mod_name, "path": path.relative_to(ROOT).as_posix(),
            "lines": src.count("\n") + 1,
            "imports": imports, "function_imports": func_imports,
            "dynamic": dynamic, "resources": resources}


def closure(entries: list[str]) -> dict:
    seen: dict[str, dict] = {}
    order: list[str] = []
    deferred: dict[str, list[str]] = {}

    def walk(mod: str) -> None:
        if mod in seen:
            return
        info = _scan(mod)
        seen[mod] = info
        order.append(mod)
        proj = sorted({i["module"] for i in info["imports"] if _is_project(i["module"])}
                      | {i["module"] for i in info["function_imports"]
                         if _is_project(i["module"])})
        late = sorted({i["module"] for i in info["function_imports"]
                       if _is_project(i["module"])})
        if late:
            deferred[mod] = late
        for sub in proj:
            walk(sub)

    for e in entries:
        walk(e)

    third = sorted({i["module"].split(".")[0]
                    for m in order
                    for i in seen[m]["imports"] + seen[m]["function_imports"]
                    if i["module"] and not _is_project(i["module"])
                    and not i["module"].startswith("__")})
    return {"profile": {"entries": entries,
                        "label": PROFILES[[k for k, v in PROFILES.items()
                                           if v["entries"] == entries][0]]["label"]},
            "modules": [{"module": m, "path": seen[m]["path"],
                         "lines": seen[m]["lines"],
                         "module_level_imports": len(seen[m]["imports"]),
                         "function_body_imports": len(seen[m]["function_imports"])}
                        for m in order],
            "deferred_project_imports": deferred,
            "dynamic_calls": {m: seen[m]["dynamic"] for m in order if seen[m]["dynamic"]},
            "resource_refs": {m: seen[m]["resources"] for m in order if seen[m]["resources"]},
            "third_party": third}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    out = {k: closure(v["entries"]) for k, v in PROFILES.items()}
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0

    for k, d in out.items():
        mods = d["modules"]
        print(f"=== Profile {k}：{d['profile']['label']} ===")
        print(f"  项目模块 {len(mods)} 个，合计 {sum(m['lines'] for m in mods)} 行")
        for m in mods:
            print(f"    {m['module']:<20} {m['lines']:>5} 行  "
                  f"模块级 {m['module_level_imports']:>2} / 函数内 {m['function_body_imports']:>2}")
        if d["deferred_project_imports"]:
            print("  ** 函数体内导入的项目模块（拟合前不可见）:")
            for m, subs in sorted(d["deferred_project_imports"].items()):
                print(f"    {m} → {', '.join(subs)}")
        if d["dynamic_calls"]:
            print("  ** 动态导入/求值:")
            for m, calls in sorted(d["dynamic_calls"].items()):
                print(f"    {m}: " + ", ".join(f"{c['call']}@{c['line']}" for c in calls))
        if d["resource_refs"]:
            print("  ** 资源文件引用（需人工判断是否影响数值）:")
            for m, refs in sorted(d["resource_refs"].items()):
                for r in refs:
                    print(f"    {m}:{r['line']} {r['call']} {r['snippet']}")
        print(f"  第三方（进 env_sha256，不进 declared 集）: {', '.join(d['third_party'])}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
