# -*- coding: utf-8 -*-
"""test_repo_integrity.py —— 仓库完整性：生产依赖不许留在版本控制之外。

起因（2026-11-22）
----------------
`sampling_hook.py` 被 `bm_core` / `bm_top` / `sampler_capture` 直接 import，
却**没有被 git 跟踪**。本机工作区能跑（文件在磁盘上），一旦别人 clone 或换机
检出，三个生产模块当场 ImportError —— 而**所有测试都在本机跑、都能过**：
没有一条测试会问"这个 import 的东西进版本控制了吗"。

这类漏洞的特征是"测试全绿 + 仓库已坏"，靠跑拟合永远发现不了。所以单独立一个
文件守它，而不是塞进入口对齐或拟合测试里。

只回答三个问题，不检查风格、不碰拟合：
1. 生产模块 import 的本地模块，是否都被 git 跟踪；
2. 顶层 `.py` 是否都被 git 跟踪（抓"没人 import 的孤立脚本"，第 1 条看不出来）；
3. 守卫自己是不是恒真（抽取逻辑坏掉时会"永远通过"）。

**git 不可用时判 `skip` 并明说"未验证"**，不当成通过 —— 与
`test_stage0_freeze` 的 D5d 同一条纪律："查不了 ≠ 没问题"。

用法：python test_repo_integrity.py
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#: 顶层里不算"生产模块"的：测试文件自成一组，由第 2 条整体覆盖。
_NOT_PROD_PREFIX = ("test_",)


def _tracked_py() -> set | None:
    """`git ls-files *.py` 的结果；git 不可用返回 None（**不要**返回空集）。"""
    try:
        r = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    if r.returncode != 0:
        return None
    return {ln.strip().replace("\\", "/") for ln in r.stdout.splitlines() if ln.strip()}


def _local_imports(path: Path) -> set:
    """抽出该文件 import 的**顶层名字**（`a.b` 取 `a`，相对 import 忽略）。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return set()
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def case_prod_imports_tracked():
    """生产模块 import 的本地模块必须在 git 里（否则 clone 即 ImportError）。"""
    tracked = _tracked_py()
    assert tracked is not None, "git 不可用，无法验证（未验证 ≠ 通过）"
    local = {p.stem for p in ROOT.glob("*.py")}
    bad = []
    for p in sorted(ROOT.glob("*.py")):
        if p.name.startswith(_NOT_PROD_PREFIX):
            continue
        for m in sorted(_local_imports(p) & local):
            if f"{m}.py" not in tracked:
                bad.append(f"{p.name} -> {m}.py")
    assert not bad, "生产模块 import 了未进版本控制的本地模块：" + "；".join(bad)
    return f"{len(local)} 个本地模块的被引用关系全部落在 git 内"


def case_top_level_py_tracked():
    """顶层 `.py` 全部要在 git 里 —— 抓"没人 import 的孤立脚本"。"""
    tracked = _tracked_py()
    assert tracked is not None, "git 不可用，无法验证（未验证 ≠ 通过）"
    missing = sorted(p.name for p in ROOT.glob("*.py") if p.name not in tracked)
    assert not missing, "顶层脚本未进版本控制：" + "、".join(missing)
    return "顶层 .py 全部已跟踪"


def case_guard_not_vacuous():
    """守卫不能恒真：抽取逻辑必须能在已知文件上抓到已知依赖。"""
    mods = _local_imports(ROOT / "bm_core.py")
    assert "sampling_hook" in mods, (
        "从 bm_core.py 抽不到 sampling_hook，说明 import 抽取失效、"
        f"第 1 条会恒真。抽到的是：{sorted(mods)}")
    tracked = _tracked_py()
    assert tracked is not None, "git 不可用"
    assert "__not_a_real_module__.py" not in tracked, "跟踪集合判定异常"
    return "抽取与判定都有效（非恒真）"


CASES = [
    ("R1 生产 import 都已跟踪", case_prod_imports_tracked),
    ("R2 顶层 .py 全部已跟踪", case_top_level_py_tracked),
    ("R3 守卫非恒真", case_guard_not_vacuous),
]


def main() -> int:
    ok, bad = 0, []
    for name, fn in CASES:
        try:
            msg = fn()
            ok += 1
            print(f"[ok] {name}：{msg}")
        except AssertionError as exc:
            bad.append(name)
            print(f"[FAIL] {name}：{exc}")
    print(f"\n{ok}/{len(CASES)} 通过" + (f"；未通过：{', '.join(bad)}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
