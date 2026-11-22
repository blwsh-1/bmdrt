"""A/B 证据：加诊断字段前后，拟合产物的数值必须逐字节一致。

做法不是改文件再跑，而是运行时把 `dynamic_range_diag` 换成一个不碰谱的桩，
两次运行走**完全相同的候选集与预算**，只差 meta 里那个键。比较：

  observations.jsonl / fits/*.json / plan.json / screening.json 的 sha256

任何一份不同，就说明诊断泄漏进了拟合路径。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/ab_diag.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import circuit_runner as RU  # noqa: E402

SRC = os.path.join("EISmeasurements", "csv_3e",
                   "3E-3_full_electrolysis_cell_exp1_0100mA.csv")
BASE = os.path.join("out", "ab_diag")
ARTIFACTS = ("plan.json", "screening.json", "observations.jsonl")


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def norm_obs(p: str) -> str:
    """观测行规范化：剔除 `elapsed_s`（运行耗时，受 jax 编译缓存影响）。

    耗时不是拟合数值，把它算进 A/B 会让"第二次更快"伪装成"结果变了"。
    剔除后仍逐键比较，任何真实数值差异都会暴露。
    """
    rows = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            r.pop("elapsed_s", None)
            ob = r.get("observation")
            if isinstance(ob, dict) and isinstance(ob.get("notes"), list):
                ob["notes"] = [n for n in ob["notes"]
                               if not str(n).startswith("elapsed_s=")]
            rows.append(r)
    return json.dumps(rows, sort_keys=True, ensure_ascii=False)


def digest_dir(d: str) -> dict:
    out = {}
    for name in ARTIFACTS:
        p = os.path.join(d, name)
        if not os.path.exists(p):
            out[name] = None
        elif name == "observations.jsonl":
            out[name] = hashlib.sha256(norm_obs(p).encode()).hexdigest()
        else:
            out[name] = sha(p)
    fdir = os.path.join(d, "fits")
    for fn in sorted(os.listdir(fdir)) if os.path.isdir(fdir) else []:
        out["fits/" + fn] = sha(os.path.join(fdir, fn))
    return out


def main() -> int:
    shutil.rmtree(BASE, ignore_errors=True)
    da, db = os.path.join(BASE, "A"), os.path.join(BASE, "B")

    mA = RU.run(SRC, max_candidates=1, include_L=True, out_dir=da)
    orig = RU.dynamic_range_diag
    RU.dynamic_range_diag = lambda Z: {"stub": True}   # 等价于"没有这个字段"
    try:
        mB = RU.run(SRC, max_candidates=1, include_L=True, out_dir=db)
    finally:
        RU.dynamic_range_diag = orig

    # 空比较不算证据：必须真的拟合过至少一个候选
    obs = os.path.join(da, "observations.jsonl")
    if not os.path.exists(obs) or os.path.getsize(obs) == 0:
        print("ERROR: A 组没有产生任何观测行，A/B 无效（候选可能全被预检排除）")
        print("A.rows =", mA["rows"])
        return 2

    ha, hb = digest_dir(da), digest_dir(db)
    print("%-34s %-12s %-12s %s" % ("artifact", "with diag", "stub", "equal"))
    print("-" * 76)
    same = True
    for k in ha:
        eq = ha[k] == hb.get(k)
        same &= eq
        print("%-34s %-12s %-12s %s" % (k, (ha[k] or "")[:10],
                                        (hb.get(k) or "")[:10], eq))

    print("-" * 76)
    print("拟合产物逐字节一致:", same)
    print("A.meta.noise_scale_diag =", json.dumps(mA["noise_scale_diag"],
                                                ensure_ascii=False))
    print("B.meta.noise_scale_diag =", json.dumps(mB["noise_scale_diag"],
                                                ensure_ascii=False))
    keys_a = set(mA) - {"noise_scale_diag"}
    keys_b = set(mB) - {"noise_scale_diag"}
    print("除新键外 meta 键集合相同:", keys_a == keys_b)
    print("候选 id 相同:", [r.get("candidate_id") for r in mA["rows"]]
          == [r.get("candidate_id") for r in mB["rows"]])
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
