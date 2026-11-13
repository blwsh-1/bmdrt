"""阶段 0 证据快照：把历史产物复制到 evidence/ 并生成可校验清单。

只读复制，不重算、不改原始产物。每个条目录 sha256，源文件与副本双记，
用来证明副本与 out/ 里的原始结果字节一致。
"""
import hashlib
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COATING = ("R1-[P2,R3-[P4,R5-P6]]-[150.0, 100926299.76068324, 0.9, "
           "309325.2729362051, 7919355.501653096, 0.95, 100000.0, 10000.0, 0.5]")
PASSIV = ("[P1,P2-R3]-R4-[1002185.2194945764, 0.782449807, 1873214.9711620852, "
          "0.5002444313837133, 9277752.99835045, 141.24023283906809]")

# (目标相对 evidence/ 的路径, 源相对仓库根的路径)
PLAN = [
    # --- 涂层：同一拓扑，只换最内层元件类型 / 只换预算 ---
    ("stage0_three_spectra/coating__nested_parallel_standard.nested.json",
     f"out/nested_fit_cap_control/{COATING}.nested.json"),
    ("stage0_three_spectra/coating__nested_capacitive_standard.nested.json",
     f"out/nested_fit_cap_standard/{COATING}.nested.json"),
    ("stage0_three_spectra/coating__nested_capacitive_quick.nested.json",
     f"out/nested_fit_cap_smoke/{COATING}.nested.json"),
    ("stage0_three_spectra/coating__nested_capacitive_standard_1113.nested.json",
     f"out/_user_fit_nested/{COATING}.nested.json"),
    # --- 钝化：同一候选，只换初始化 ---
    ("stage0_three_spectra/passivation__nested_capacitive_standard_default_init.nested.json",
     f"out/nested_fit_cap_pass/{PASSIV}.nested.json"),
    ("stage0_three_spectra/passivation__nested_capacitive_standard_1113.nested.json",
     f"out/_user_fit_nested/{PASSIV}.nested.json"),
    ("stage0_three_spectra/passivation__diag_truth_init.json",
     "out/_user_fit_nested/diag_passivation_init.json"),
    # --- 六条谱两族对照（nested 走 data_init，voigt 走逐点 2%） ---
    ("stage0_three_spectra/six_runs__family_fit_results.json",
     "out/_user_fit_nested/family_fit_results.json"),
    # --- AZ31：一个被结构排除，一个跑得动但不过线 ---
    ("stage0_three_spectra/az31__m4_real_az31.json", "out/m4_real_az31.json"),
    ("stage0_three_spectra/az31__m4_real_nested_az31.json", "out/m4_real_nested_az31.json"),
    # --- 产生上述产物的脚本 ---
    ("stage0_three_spectra/scripts/fit_family_compare.py",
     "out/_user_fit_nested/fit_family_compare.py"),
    ("stage0_three_spectra/scripts/diag_passivation_init.py",
     "out/_user_fit_nested/diag_passivation_init.py"),
    # --- 60 条涂层 pilot：逐条记录（含四列口径 + 敏感性 + 参数恢复） ---
    ("coating_voigt_pilot/results.jsonl", "out/coating_voigt_pilot/results.jsonl"),
    ("coating_voigt_pilot/summary.json", "out/coating_voigt_pilot/summary.json"),
    ("coating_voigt_pilot/sample_manifest.json", "out/coating_voigt_pilot/sample_manifest.json"),
    ("coating_voigt_pilot/mode_diag.json", "out/coating_voigt_pilot/mode_diag.json"),
    ("coating_voigt_pilot/k3_check.json", "out/coating_voigt_pilot/k3_check.json"),
    ("coating_voigt_pilot/config_check.json", "out/coating_voigt_pilot/config_check.json"),
]

OUT_DIR = os.path.join(ROOT, "evidence")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    records = []
    for dst_rel, src_rel in PLAN:
        src = os.path.join(ROOT, src_rel.replace("/", os.sep))
        dst = os.path.join(OUT_DIR, dst_rel.replace("/", os.sep))
        if not os.path.isfile(src):
            raise SystemExit(f"缺源文件: {src_rel}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        h_src, h_dst = sha256(src), sha256(dst)
        assert h_src == h_dst, (dst_rel, h_src, h_dst)
        records.append({
            "evidence_path": dst_rel,
            "source_path": src_rel,
            "source_basename": os.path.basename(src_rel),
            "bytes": os.path.getsize(dst),
            "sha256": h_dst,
            "sha256_matches_source": True,
        })
        print(f"ok  {dst_rel:<78} {os.path.getsize(dst):>7} B  {h_dst[:16]}")

    manifest = {
        "purpose": "阶段 0 证据快照：供 schema 承载能力审计，只读复制，不重算",
        "repo_commit_at_snapshot": os.popen(
            "git -C " + ROOT.replace(os.sep, "/") + " rev-parse HEAD").read().strip(),
        "n_files": len(records),
        "total_bytes": sum(r["bytes"] for r in records),
        "files": records,
        "notes": [
            "副本与源文件字节一致，逐条断言过；sha256 双方都记，可脱离 out/ 独立校验。",
            "未包含任何 .png：图像由 CSV/JSON 重绘，不入库（.gitignore 全局忽略 *.png）。",
            "文件名以角色前缀重命名，原始文件名（含真值参数）见 source_basename。",
        ],
    }
    out = os.path.join(OUT_DIR, "stage0_three_spectra", "MANIFEST.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"\n{len(records)} 个文件 / {manifest['total_bytes']/1024:.0f} KiB → {out}")


if __name__ == "__main__":
    main()
