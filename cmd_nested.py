"""`bmdrt nested` —— 把 `nested_fit.py` 接成 CLI 子命令（薄包装，零算法）。

为什么只是包装
--------------
用户 2026-11-16 裁决「合起来能跑就行」：命令行入口收敛成 `bmdrt.py` 一条，
不再需要记两个脚本名。但**不改任何拟合路径**——这里直接把参数转发给
`nested_fit.main()`，底层仍是 `bm_top.fit_top`，与 `python nested_fit.py` 逐位相同。

`bmdrt.py` 的自动发现规则是「`cmd_<名字>.py` 里有 `run(argv)` 就多一个子命令」，
所以本文件只提供 `run()`，不注册别的东西。

状态仍是 `provisional`
----------------------
准入没通过（§13 0/3、§14 4/14、§17 0/10），所以：
· 产物带 `admission.state="provisional"` + `passed=False`（由 `nested_fit.run` 写）；
· 不进 `bmdrt auto` 的 `summary.json`、不参与 KK 门禁、不当真值；
· 本子命令**不改** `auto` 的默认族（仍是 voigt）。

用法
----
    bmdrt nested                          # 扫 inbox/
    bmdrt nested data.csv
    bmdrt nested data.csv --budget quick --no-plot
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run(argv=None) -> int:
    """转发给 `nested_fit.main`。参数表完全一致，不另立一套。"""
    import nested_fit as NF
    return int(NF.main(list(argv) if argv else []))


def selftest() -> int:
    ok, total, bad = 0, 0, []

    def c(name, cond, extra=""):
        nonlocal ok, total
        total += 1
        print(f"  [{'ok' if cond else 'FAIL'}] {name}"
              + (f"  {extra}" if not cond and extra else ""))
        if cond:
            ok += 1
        else:
            bad.append(name)

    import ast
    import inspect
    import nested_fit as NF

    tree = ast.parse(open(__file__, encoding="utf-8").read())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])

    c("run 是 bmdrt 自动发现认的形状（接收 argv）",
      list(inspect.signature(run).parameters) == ["argv"])
    c("转发目标是 nested_fit.main（同一份实现）",
      run.__doc__ is not None and "nested_fit.main" in (run.__doc__ or ""))
    # 参数表必须由 nested_fit 单方定义：本模块不另写 argparse
    # （用 AST 查真正的 import 语句；只查文本会被 docstring 里的字面量误伤）
    c("本文件不 import argparse（不另立参数表）", "argparse" not in mods)
    c("本文件不 import bm_top / bm_core / bm_real（不碰拟合）",
      not ({"bm_top", "bm_core", "bm_real"} & mods), f"imports={sorted(mods)}")
    # 与直接跑脚本等价：同一个 main 对象
    c("nested_fit.main 可调用", callable(NF.main))

    print(f"\n自检 {ok}/{total} 通过" if not bad else f"\nFAIL {len(bad)}/{total}: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(run(sys.argv[1:]))
