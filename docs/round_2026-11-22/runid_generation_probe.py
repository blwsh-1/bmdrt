"""找出 records.jsonl 为什么随 REGISTRY_VERSION 变化。

已知：`--check` 在 version=0.7.0 时 records.jsonl 哈希一致（bf22f64b），
version=0.8.0 时变成 283bf0ee。但每条记录的顶层键里没有 registry_version。

本脚本在内存里跑两次 build()，逐字段定位差异。
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "docs/round_2026-11-13")

import bm_select  # noqa: E402
import circuit_registry as R  # noqa: E402
import map_stage0_records as M  # noqa: E402

budgets = {k: {kk: int(vv) if isinstance(vv, int) else float(vv)
               for kk, vv in v.items() if kk in R.BUDGET_PARAM_KEYS}
           for k, v in bm_select.BUDGETS.items()}


def build_at(ver):
    R.REGISTRY_VERSION = ver
    M._HASHED_FILES.clear()
    return M.build(budgets)


a = build_at("0.7.0")
b = build_at("0.8.0")


def diff(x, y, path=""):
    out = []
    if type(x) is not type(y):
        out.append((path, type(x).__name__, type(y).__name__))
        return out
    if isinstance(x, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                out.append((path + "/" + k, "<missing>" if k not in x else "present",
                            "<missing>" if k not in y else "present"))
            else:
                out += diff(x[k], y[k], path + "/" + k)
    elif isinstance(x, list):
        if len(x) != len(y):
            out.append((path, "len %d" % len(x), "len %d" % len(y)))
        for i, (u, v) in enumerate(zip(x, y)):
            out += diff(u, v, path + "[%d]" % i)
    else:
        if x != y:
            out.append((path, x, y))
    return out


print("记录数：0.7.0=%d  0.8.0=%d" % (len(a), len(b)))
d = diff(a, b)
print("差异字段数：%d" % len(d))
for p, u, v in d[:12]:
    print("  %s\n    0.7.0=%r\n    0.8.0=%r" % (p, u, v))

# 定位到具体是哪类路径
import collections
kinds = collections.Counter()
for p, _, _ in d:
    # 去掉 [i] 索引，保留结构
    s = p.replace("[0]", "[]").replace("[1]", "[]")
    kinds[s] += 1
print("\n按路径归并（前 8）：")
for k, c in kinds.most_common(8):
    print("  %-60s x%d" % (k, c))
