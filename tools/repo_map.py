"""仓库代码地图：把根目录的模块按「层 / 产品可用性 / 职责」分类，并统计依赖与耦合。

纯静态（AST + 文本计数），**不 import 任何项目模块、不跑拟合、不写产物数据**。
输出 `docs/code_map.json`；`--check` 只校验分类表覆盖了磁盘上所有模块（新文件漏分类会报错）。

用法
----
    python tools/repo_map.py            # 生成 docs/code_map.json + 打印摘要
    python tools/repo_map.py --check    # 只校验，不写文件（CI 用）
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 层定义（顺序即依赖方向：上层可依赖下层，下层不得依赖上层）
LAYERS = {
    "L0": "数值内核：纯数学/模型定义，无文件 IO",
    "L1": "数据与 IO：读谱、转换、批处理、参考实现",
    "L2": "分析与诊断：门禁、可辨识性、相关矩阵、合成验收",
    "L3": "结果与呈现：唯一结果对象、静态图、交互面板",
    "L4": "编排与入口：CLI、研究接口、跨族选择、命令行子命令",
    "L5": "执行身份与审计：预算/采样/补丁包的契约与捕获",
    "L6": "测试：反例矩阵与回归",
}

# module -> (层, 职责, 产品候选, 稳定性)
#   产品候选：core=产品核心 / opt=可选 / no=仅研究或审计
#   稳定性：frozen=已冻结口径不许改语义 / stable=可复用 / research=研究性
ROLE = {
    # ---- L0 ----
    "bm_core.py": ("L0", "贝叶斯混合（stick-breaking + HN）voigt 族模型与 `fit`", "core", "frozen"),
    "bm_top.py": ("L0", "嵌套并联拓扑 `z_nested` 正演 + `fit_top` 独立采样路径", "core", "frozen"),
    "bm_diff.py": ("L0", "Warburg 族扩散元件与 HN↔Warburg 退化诊断", "core", "stable"),
    "bm_ad.py": ("L0", "低频吸附电感元件 `Z_ad`", "opt", "stable"),
    "bm_ws.py": ("L0", "Ws 判据候选量（生产模块，被 `bm_real` 消费）", "core", "frozen"),
    "drt.py": ("L0", "DRT 粗解（只报过程数/时间常数/电阻）", "opt", "stable"),
    "bm_circuit.py": ("L0", "等效电路图绘制（模式 A）", "no", "stable"),
    "circuit_registry.py": ("L0", "候选电路注册表（schema 承载能力）", "opt", "research"),
    "circuit_runner.py": ("L4", "registry 驱动的候选调度：预检→逐候选拟合→观测落盘（研究接口，不改默认族、不接 auto）", "opt", "research"),
    "circuit_evaluator.py": ("L2", "候选观测→evidence 状态的推导层（规则 E1–E7，争议规则显式标 disputed）", "opt", "research"),
    "bm_figure_unified.py": ("L3", "统一拟合图：一条谱 + 任意多条候选曲线 → 同一张 6 子图（只依赖 f/Z/curves）", "opt", "stable"),
    "cmd_compare.py": ("L4", "bmdrt compare：多族候选自动并列 → 统一状态 + 统一图（默认 standard / 逐点 σ）", "opt", "research"),
    "cmd_nested.py": ("L4", "bmdrt nested：nested_fit 的薄包装（零算法，参数表复用 nested_fit.main）", "opt", "research"),
    "admission.py": ("L2", "临时准入状态：provisional 常量 + 逐条差距 + 有/无证据范围（**恒 passed=False，不是开关**）", "opt", "research"),
    # ---- L1 ----
    "bm_real.py": ("L1", "真实数据主程序：读谱 / Lin-KK 门禁 / 选 K / 报告（**职责过载，需拆分**）", "core", "frozen"),
    "bm_ref_rwp.py": ("L1", "参考性 Rwp（独立 LS 重拟合，只作对照）", "opt", "frozen"),
    "fit_store.py": ("L1", "drt 命令行模式的兼容读取器", "no", "stable"),
    "convert_2e.py": ("L1", "Gamry 2E xlsx → 三列 CSV", "opt", "stable"),
    "convert_3e.py": ("L1", "Gamry 3E xlsx → 三列 CSV", "opt", "stable"),
    "batch_2e.py": ("L1", "2E 批处理调度 + 交付", "no", "research"),
    "report_2e.py": ("L1", "2E 汇总表 → 可复核交付说明", "no", "research"),
    # ---- L2 ----
    "bm_kk.py": ("L2", "拟合前 Lin-KK 门禁（gate/warn/off）", "core", "frozen"),
    "bm_identifiability.py": ("L2", "统一可辨识性报告（六类诊断 / 三态）", "opt", "stable"),
    "bm_ident.py": ("L2", "峰位偏差归因（代码错 vs 估计量性质）", "no", "research"),
    "bm_corr.py": ("L2", "全参数后验相关矩阵 + 置信区间", "opt", "stable"),
    "bm_bench.py": ("L2", "合成 benchmark（难度递增案例）", "opt", "stable"),
    "bm_check.py": ("L2", "合成数据端到端验收 PASS/FAIL", "opt", "stable"),
    "bm_evidence.py": ("L2", "证据卡生成器（L3 唯一输出形态，只读消费 JSON）", "opt", "stable"),
    "bm_grade.py": ("L2", "证据资格分档 A/B/C（只贴标签）", "opt", "stable"),
    "bm_zhit.py": ("L2", "Z-HIT 自洽检验一阶形式（离线原型，**已否决不接**）", "no", "research"),
    "bm_top_gen.py": ("L2", "nested 拓扑独立生成器（合成侧）", "no", "research"),
    "bm_experiments.py": ("L2", "全部实验脚本分发（`acf_max` 等指标也在这里）", "opt", "frozen"),
    "spectrum_metrics.py": ("L2", "两入口共用的残差指标/逐点 σ/ELPD 可比性判定/预算归一（纯函数）", "opt", "stable"),
    # ---- L3 ----
    "bm_result.py": ("L3", "`AnalysisResult` 唯一结果来源与导出包", "core", "stable"),
    "bm_figure.py": ("L3", "出版级静态图（与面板共用同一份数据）", "core", "stable"),
    "bm_dash.py": ("L3", "自包含 HTML 交互面板（零运行时依赖）", "core", "stable"),
    # ---- L4 ----
    "bmdrt.py": ("L4", "唯一 CLI 入口（`cmd_*.py` 自动注册子命令）", "core", "stable"),
    "nested_fit.py": ("L4", "nested 独立研究接口（**未接主流程**）", "opt", "research"),
    "bm_select.py": ("L4", "M4 跨族选择器骨架（voigt vs nested）", "opt", "research"),
    "cmd_diff.py": ("L4", "子命令：扩散项数学与可辨识性回归", "no", "stable"),
    "cmd_status.py": ("L4", "子命令：查看 out/ 里跑过的结果", "opt", "stable"),
    "sweep_sigma_floor.py": ("L4", "σ 噪声地板敏感性扫描", "no", "research"),
    # ---- L5 ----
    "exec_identity.py": ("L5", "执行身份契约 v1.5（纯函数：`run_id`/`workflow_id`）", "no", "frozen"),
    "capture_contract.py": ("L5", "采集契约：路径规范化 / harness 归属 / 事件归属", "no", "frozen"),
    "sampler_provenance.py": ("L5", "采样参数来源三态（请求/解析/观测）", "no", "frozen"),
    "sampler_capture.py": ("L5", "采样参数捕获层（两处生产构造点）", "no", "frozen"),
    "budget_params.py": ("L5", "`budget_params` 产出层（五项预算纯函数）", "no", "frozen"),
    "budget_capture_adapter.py": ("L5", "捕获事件 → 六键的拒绝式适配层", "no", "frozen"),
    "patch_bundle.py": ("L5", "补丁包契约（纯函数层）", "no", "frozen"),
    "patch_bundle_io.py": ("L5", "补丁包 I/O（采集 / 独立重建）", "no", "frozen"),
}

TEST_NOTE = {
    "test_exec_identity.py": "身份 v1.4 反例矩阵 T/W/S/Y/M/X/C 七组",
    "test_identity_p0.py": "身份 P0 四项反例 + v1.5 版本升级",
    "test_budget_params.py": "budget_params v1.4 产出层",
    "test_capture_contract.py": "采集契约 C-1/C-2/C-3",
    "test_sampler_provenance.py": "采样来源 15 格 + 接口护栏",
    "test_sampler_capture.py": "捕获层 28 格",
    "test_patch_bundle.py": "补丁包 15 格 + 真实 Git 往返 7 格",
    "test_contract.py": "二轮审查四组契约回归",
    "test_circuit_registry.py": "registry schema 契约",
    "test_ecm.py": "AutoECM 数据端到端回归",
    "test_ad_gate.py": "`detect_ad` 边界 + 不变性",
    "test_ad_gate_synth.py": "吸附门控合成矩阵",
    "test_bm_ad_e2e.py": "吸附端到端接线",
    "test_bm_ad_sampler.py": "吸附接入采样器 + 合成标定",
    "test_bm_ad_select.py": "`detect_ad` 三闸 + 选择器",
    "test_drt_signed.py": "有符号 DRT 方案 A",
    "test_audit_findings.py": "审查缺陷可复现",
    "test_include_L.py": "`include_L` 对照（单变量）",
    "test_seed.py": "sampler seed 传递链",
    "test_entry_parity.py": "两入口口径合并 T1–T6 + 差分（不跑 MCMC）",
}


def scan(mods: list[str]) -> dict:
    local = {os.path.splitext(m)[0] for m in mods}
    out = {}
    for m in mods:
        src = open(os.path.join(ROOT, m), encoding="utf-8", errors="ignore").read()
        try:
            tree = ast.parse(src)
            doc = (ast.get_docstring(tree) or "").strip().splitlines()
        except SyntaxError:
            tree, doc = None, []
        deps, has_main = set(), False
        if tree is not None:
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    for al in n.names:
                        r = al.name.split(".")[0]
                        if r in local and r != os.path.splitext(m)[0]:
                            deps.add(r)
                elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                    r = n.module.split(".")[0]
                    if r in local and r != os.path.splitext(m)[0]:
                        deps.add(r)
                elif isinstance(n, ast.If) and "__name__" in ast.dump(n.test):
                    has_main = True
        out[m] = dict(module=os.path.splitext(m)[0], lines=src.count("\n") + 1,
                      doc=(doc[0][:100] if doc else ""), deps=sorted(deps),
                      has_main=has_main,
                      uses_out=('"out"' in src or "'out'" in src),
                      sys_path_hack="sys.path.insert" in src,
                      hardcwd=("os.getcwd" in src))
    rev = collections.Counter()
    for v in out.values():
        for d in v["deps"]:
            rev[d] += 1
    for m, v in out.items():
        v["imported_by"] = rev[os.path.splitext(m)[0]]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    mods = sorted(x for x in os.listdir(ROOT) if x.endswith(".py"))
    data = scan(mods)
    tests = [m for m in mods if m.startswith("test_")]
    unclassified = [m for m in mods if m not in ROLE and m not in tests]

    if a.check:
        bad = []
        if unclassified:
            bad.append(f"未分类模块：{unclassified}")
        missing = [m for m in ROLE if m not in mods]
        if missing:
            bad.append(f"分类表里有磁盘上不存在的模块：{missing}")
        md = os.path.join(ROOT, "docs", "CODE_MAP.md")
        if not os.path.exists(md):
            bad.append("缺 docs/CODE_MAP.md")
        else:
            txt = open(md, encoding="utf-8").read()
            absent = [m for m in mods if m not in txt]
            if absent:
                bad.append(f"CODE_MAP.md 漏列模块：{absent}")
        print("分类覆盖：" + ("OK" if not bad else "FAIL"))
        for b in bad:
            print("  " + b)
        return 0 if not bad else 1

    rows = []
    for m in mods:
        v = data[m]
        if m in tests:
            layer, role, prod, stab = "L6", TEST_NOTE.get(m, "回归测试"), "no", "stable"
        elif m in ROLE:
            layer, role, prod, stab = ROLE[m]
        else:
            layer, role, prod, stab = "??", "**未分类**", "no", "?"
        rows.append(dict(file=m, layer=layer, role=role, product=prod, stability=stab, **v))

    # 层间依赖违例：L0 不该依赖 L1+，L1 不该依赖 L2+
    lv = {v: int(k[1]) for k, v in [(k, k) for k in LAYERS]}
    layer_of = {r["module"]: r["layer"] for r in rows}
    viol = []
    for r in rows:
        li = lv.get(r["layer"], 9)
        for d in r["deps"]:
            lj = lv.get(layer_of.get(d, "??"), 9)
            if lj > li:
                viol.append(dict(src=r["module"], src_layer=r["layer"],
                                 dep=d, dep_layer=layer_of.get(d, "??")))

    pack = dict(
        generated_by="tools/repo_map.py",
        n_modules=len(mods), n_tests=len(tests),
        total_lines=sum(v["lines"] for v in data.values()),
        layers={k: dict(desc=v, modules=[r["file"] for r in rows if r["layer"] == k])
                for k, v in LAYERS.items()},
        product_core=[r["file"] for r in rows if r["product"] == "core"],
        product_optional=[r["file"] for r in rows if r["product"] == "opt"],
        not_for_product=[r["file"] for r in rows if r["product"] == "no"],
        coupling=dict(
            uses_out_dir=[r["file"] for r in rows if r["uses_out"]],
            sys_path_hack=[r["file"] for r in rows if r["sys_path_hack"]],
            entry_points=[r["file"] for r in rows if r["has_main"]],
            most_depended=sorted(((r["module"], r["imported_by"]) for r in rows),
                                 key=lambda t: -t[1])[:10]),
        layer_violations=viol,
        modules=rows)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    p = os.path.join(ROOT, "docs", "code_map.json")
    json.dump(pack, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"模块 {len(mods)}（测试 {len(tests)}），总计 {pack['total_lines']} 行")
    for k in LAYERS:
        n = len(pack["layers"][k]["modules"])
        print(f"  {k} {LAYERS[k]:<40} {n:>3} 个")
    print(f"产品核心 {len(pack['product_core'])} / 可选 {len(pack['product_optional'])}"
          f" / 不进产品 {len(pack['not_for_product'])}")
    print(f"耦合：硬编码 out/ {len(pack['coupling']['uses_out_dir'])} 个，"
          f"sys.path hack {len(pack['coupling']['sys_path_hack'])} 个，"
          f"入口 {len(pack['coupling']['entry_points'])} 个")
    if viol:
        print(f"层间反向依赖 {len(viol)} 处：", viol[:6])
    if unclassified:
        print("未分类：", unclassified)
    print("已写", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
