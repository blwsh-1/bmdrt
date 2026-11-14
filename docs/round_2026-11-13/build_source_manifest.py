# -*- coding: utf-8 -*-
"""生成 `source_manifest.json`（执行身份契约 v1.2 的声明集，**只写清单**）。

分工：
  · 本脚本 = **声明**：哪些文件属于哪个 Profile、扮演什么角色、有没有资格降级；
  · runner（未授权）= **取值**：运行点算每份文件的 `sha256`、三层摘要、补丁包。

所以清单里**不存内容哈希**（存了就会随一次编辑变陈旧，而陈旧值最容易被当成事实读）。
`digest_preview` 只记录"按**当前文件内容**算出来是多少"，是审阅用的快照，不是身份。
它刻意**不含提交号**：带了提交号的预览会在下一次不碰这些文件的提交之后变成一句假话，
而`--check` 又会因为"文本不一致"逼人重新生成——一个纯展示字段不该有这个权力。

模块清单由 `audit_source_profiles.py` 的 AST 闭包算出，不手列目录——
`nested_fit.py:211` 的函数体内 `import bm_core` 就是手列目录会漏掉的那类。

    python docs/round_2026-11-13/build_source_manifest.py            # 写文件
    python docs/round_2026-11-13/build_source_manifest.py --check     # 只比对不写
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "docs" / "round_2026-11-13") not in sys.path:
    sys.path.insert(0, str(ROOT / "docs" / "round_2026-11-13"))

import exec_identity as E                                    # noqa: E402
from audit_source_profiles import PROFILES, closure           # noqa: E402

OUT = ROOT / "source_manifest.json"

#: 静态闭包**上方**的两个文件：闭包从 `bm_real`/`nested_fit`/`bm_select` 出发，
#: 看不到"把它们调起来"的那一层。但真实进程入口决定传给 fit 的实参，
#: `config.json` 决定这些实参的默认值——两条都会改数字，所以必须显式补进声明集。
SUPPLEMENT = [
    {"path": "bmdrt.py", "role": "entry",
     "reason": "真实进程入口：解析命令行 + 合并 config.json 后再调 fit；"
               "在 AST 闭包‘上方’，闭包看不到它，但它决定传给 fit 的实参"},
    {"path": "config.json", "role": "config",
     "reason": "显式清单：数值默认值的来源（fit/kk/diag 段）；非 Python 文件，"
               "静态导入闭包必然漏掉"},
]

#: 降级**候选**（全部 `applied: false`）。契约 §3.1.3 要求降级必须过验证，而验证要
#: 真实运行（关掉功能看 `runtime_loaded_modules`，或改无关常量看数值是否逐位相同）。
#: 在跑出来之前，清单里一律 `numeric` / `workflow_scope=true`：
#: 误降会**静默合并**两次不同代码的运行，误升只会多换号。
PENDING_DEMOTIONS = [
    {"paths": ["bm_circuit.py"], "profiles": ["A", "B"],
     "proposed_compute_role": "plot_only",
     "hypothesis": "只画图；`plot=False` 时 `bm_real` 不会在函数体内导入它",
     "applied": False,
     "required_test": "一次 `plot=False` 的真实运行：`runtime_loaded_modules` 不含 "
                      "`bm_circuit.py`，且产物数值字段与 `plot=True` 逐位相同"},
    {"paths": ["bm_ref_rwp.py"], "profiles": ["A"],
     "proposed_compute_role": "reference_only",
     "hypothesis": "只加 `Rwp_ref` 对照指标，不改参数推断",
     "applied": False,
     "required_test": "一次 `ref_rwp=False` 的真实运行：不加载该模块，"
                      "且 `params`/`metrics`/`elpd` 与开启时逐位相同"},
    {"paths": ["bm_grade.py", "bm_experiments.py"], "profiles": ["A", "B"],
     "proposed_compute_role": "decision",
     "hypothesis": "事后诊断/评分/报告，不改后验",
     "applied": False,
     "required_test": "改一个不相关常量后重跑：`params`/`metrics`/`elpd` 逐位相同"},
    {"paths": ["bm_select.py"], "profiles": ["B"],
     "proposed_compute_role": "budget_source",
     "hypothesis": "Profile B 里只读 `BUDGETS` 常量，数值效应已由 `budget_params` 捕获",
     "applied": False,
     "required_test": "改 `select_across_families` 一行后重跑 nested："
                      "`params`/`metrics`/`elpd` 逐位相同（且 `budget_params` 不变）"},
    {"paths": ["bm_kk.py"], "profiles": ["A", "B"],
     "proposed_compute_role": "decision",
     "hypothesis": "只产 KK 门禁状态；`kk_opts.auto_L` 不回写 `include_L`",
     "applied": False,
     "required_test": "`kk_mode=off/warn` 两次真实运行的 `params`/`metrics` 逐位相同，"
                      "且不加载该模块"},
    {"paths": ["config.json"], "profiles": ["A", "B", "C"],
     "proposed_compute_role": "decision",
     "hypothesis": "只有 fit/kk/diag 段影响数值，paths 段只影响产物落点",
     "applied": False,
     "required_test": "拆分需按**段**降级，而 `compute_role` 是**文件**粒度 ⇒ "
                      "在文件级身份里无法拆；若要拆，须先把 config 段拆成独立文件"},
]

#: 明确不进声明集的东西（写出来，免得下一个人以为漏了）。
EXCLUDED = [
    {"pattern": "requirements.txt", "reason": "环境意图，属 `env` 块（`lock_digest`），"
                                             "不是被执行的代码"},
    {"pattern": "out/ evidence/ archive/ __pycache__/ docs/",
     "reason": "产物、证据、存档、缓存、文档；改它们不改数字"},
    {"pattern": "数据文件（.csv/.txt/.z 等）", "reason": "属 `data` 块；重复计算会让"
                                                       "‘改数据’和‘改代码’混成同一种变化"},
    {"pattern": "已安装的第三方包", "reason": "进 `env_sha256`（包名+版本），不进 declared 集"},
]


def build() -> dict:
    profiles: dict[str, dict] = {}
    for key, spec in PROFILES.items():
        cls = closure(spec["entries"])
        items = []
        for mod in cls["modules"]:
            path = mod["path"]
            items.append({"path": path,
                          "role": "entry" if mod["module"] in spec["entries"] else "module"})
        for sup in SUPPLEMENT:
            items.append(dict(sup))
        # 同一路径只留一份（`bmdrt.py` 不在任何闭包里，这里只是防将来重复）
        dedup, seen = [], set()
        for it in items:
            if it["path"] in seen:
                continue
            seen.add(it["path"])
            dedup.append(it)
        dedup.sort(key=lambda d: d["path"])

        with_sha = [dict(it, sha256=E.source_file_sha256(ROOT / it["path"])) for it in dedup]
        E.check_declared_set(with_sha)
        profiles[key] = {
            "label": spec["label"],
            "entries": list(spec["entries"]),
            "process_entry": "bmdrt.py",
            "closure_source": "audit_source_profiles.closure (AST，含函数体内导入)",
            "n_files": len(dedup),
            "declared_source_set": dedup,
            "digest_preview": {
                "note": "审阅快照，不是身份；身份由 runner 在运行点重算（§3.1.1）",
                "declared": E.declared_source_tree_sha256(with_sha),
                "numeric": E.numeric_source_tree_sha256(with_sha),
                "workflow": E.workflow_source_tree_sha256(with_sha),
            },
        }
    return {
        "manifest_version": "1",
        "identity_contract_version": E.IDENTITY_CONTRACT_VERSION,
        "generated_by": "docs/round_2026-11-13/build_source_manifest.py",
        "note": "声明集只列**文件与角色**，不存内容哈希；降级须 why_not_numeric + evidence。",
        "defaults": {
            "compute_role": "numeric",
            "workflow_scope": True,
            "why": "误降会静默合并两次不同代码的运行；误升只会多换号（看得见）",
        },
        "profiles": profiles,
        "pending_demotions": PENDING_DEMOTIONS,
        "excluded": EXCLUDED,
    }


def render(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=2) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="只比对，不写文件")
    a = ap.parse_args(argv)

    doc = build()
    text = render(doc)
    if a.check:
        cur = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        if cur != text:
            print("source_manifest.json 与源码不一致（请重新生成）")
            print(f"  期望字节 {len(text)}，盘上 {len(cur)}")
            return 1
        print("source_manifest.json 与源码一致")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"写入 {OUT.relative_to(ROOT).as_posix()}（{len(text)} 字节）")
    for k, v in doc["profiles"].items():
        print(f"  Profile {k}: {v['n_files']} 文件  declared={v['digest_preview']['declared'][:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
