# -*- coding: utf-8 -*-
"""M3.5 后验样本清单：`m35_*.samples.npz` 的身份、形状与校验和（§20.24）。

为什么需要它：样本现在**随版本库归档**，但"文件在不在"不等于"文件是不是当年那份"。
JSON 里只记了条件化诊断那 3 个样本的 sha256；其余 14 个（`§20.16` 首跑、`§20.17`
正式跑、`§20.19` 采样器敏感性）没有对应字段。本脚本给全部样本一份统一清单，
并提供 `--check` 用于日后发现漂移。

用法
----
    python evidence/nested_topology/m35_samples_manifest.py            # 重写清单
    python evidence/nested_topology/m35_samples_manifest.py --check    # 与清单核对

`--check` 的判定：文件名集合必须一致、逐个 sha256 一致、字节数一致、结构元数据
（`keys` / `shapes` / `n_draws`）一致，且清单自身的 `n_files` / `total_bytes` 与
`files` 条目自洽；任一不符返回 1。只比 sha256 是不够的——清单若是被手工改过，
头部统计和结构字段会先失真而 sha256 仍然"对得上"（§20.25）。
不比较修改时间（git checkout 会改 mtime，那与"内容是否同一份"无关）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "m35_samples_manifest.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: Path) -> dict:
    """一个 npz 的身份 + 可供核对的结构信息（不读全量统计，读元数据即可）。"""
    with np.load(path) as z:
        keys = sorted(z.files)
        shapes = {k: list(np.asarray(z[k]).shape) for k in keys}
        n_draws = int(np.asarray(z["R1"]).size) if "R1" in z.files else None
    return {"sha256": sha256_of(path), "bytes": path.stat().st_size,
            "keys": keys, "shapes": shapes, "n_draws": n_draws}


def build() -> dict:
    files = sorted(HERE.glob("m35_*.samples.npz"))
    entries = {p.name: describe(p) for p in files}
    return {"note": "M3.5 后验样本清单（§20.24）；样本随版本库归档，本清单用于发现漂移",
            "n_files": len(entries),
            "total_bytes": sum(e["bytes"] for e in entries.values()),
            "files": entries}


def check(manifest_path: Path = MANIFEST) -> int:
    if not manifest_path.exists():
        print(f"!! 缺清单 {manifest_path.name}，先不带 --check 跑一次")
        return 2
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    old = doc["files"]
    new = build()["files"]
    common = set(old) & set(new)
    missing = sorted(set(old) - set(new))
    extra = sorted(set(new) - set(old))
    bad = sorted(k for k in common if old[k]["sha256"] != new[k]["sha256"])
    size = sorted(k for k in common if old[k]["bytes"] != new[k]["bytes"])
    # 结构元数据：sha256 已经能保证内容同一，这里额外核对"清单里记的结构描述"
    # 是否与文件实际一致——防的是清单被手工改动后仍显示通过
    meta_keys = ("keys", "shapes", "n_draws")
    meta = sorted(k for k in common
                  if any(old[k].get(f) != new[k][f] for f in meta_keys))
    # 清单头部自洽：n_files / total_bytes 必须与 files 条目对得上
    head = []
    if doc.get("n_files") != len(old):
        head.append(f"n_files={doc.get('n_files')} ≠ 条目数 {len(old)}")
    tot = sum(e["bytes"] for e in old.values())
    if doc.get("total_bytes") != tot:
        head.append(f"total_bytes={doc.get('total_bytes')} ≠ 条目求和 {tot}")
    ok = not (missing or extra or bad or size or meta or head)
    print(f"清单核对：清单 {len(old)} 个 / 磁盘 {len(new)} 个 → {'一致' if ok else '不一致'}"
          f"（sha256 + 字节数 + keys/shapes/n_draws + 头部统计）")
    for tag, items in (("清单里有、磁盘没有", missing), ("磁盘多出", extra),
                       ("sha256 不符", bad), ("字节数不符", size),
                       ("结构元数据不符", meta), ("清单头部不自洽", head)):
        if items:
            print(f"  {tag}：{', '.join(items) if isinstance(items, list) else items}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M3.5 样本清单（§20.24）")
    ap.add_argument("--check", action="store_true", help="只核对，不重写")
    ap.add_argument("--json", default=str(MANIFEST))
    args = ap.parse_args(argv)
    path = Path(args.json)
    if args.check:
        return check(path)
    data = build()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"清单写入 {path.name}：{data['n_files']} 个文件、{data['total_bytes'] / 1e6:.2f} MB")
    for name, e in data["files"].items():
        print(f"  {e['sha256'][:12]}  {e['bytes']:>7} B  {e['n_draws']:>5} draws  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
