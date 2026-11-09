# -*- coding: utf-8 -*-
"""B3b 单组入口：装运行期指纹 hook → 跑一次 `bmdrt.py auto` → 落指纹证据。

由 `b3b_run.py` 为每组（谱 × seed）起一个独立子进程调用，保证 JAX/NumPyro
状态与编译缓存不跨组串味，且 hook 只作用于这一组。

固定项（与 M6 §4.1 B3b / §10 ② 一致）：
    --wide --kk gate --no-move；采样档取 KW 默认（3000/1000/3，ta=0.9）
唯一变化项：--seed N

用法：
    python evidence/m6/b3b_case.py --spectrum Blank_main \
        --path archive/Blank_0.1KBr_pH10_EIS_PtCu_1.2V.txt \
        --seed 0 --out out/m6/b3b/Blank_main/seed0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import b3b_fingerprint as F                             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectrum", required=True)
    ap.add_argument("--path", required=True, help="相对 bmdrt/ 的谱路径")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="",
                    help="仅用于接线自检的附加 CLI 参数（如 --quick）；"
                         "正式 15 组矩阵不使用，避免改变固定变量")
    a = ap.parse_args()

    spectrum_path = ROOT / a.path
    if not spectrum_path.exists():
        print(f"[b3b_case] 谱不存在：{spectrum_path}", file=sys.stderr)
        return 2

    outdir = Path(a.out)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    stem = spectrum_path.stem
    case_dir = outdir / stem

    status = F.install()
    F.begin_case(a.spectrum, a.path, a.seed)
    argv = ["bmdrt.py", "auto", str(spectrum_path),
            "--wide", "--kk", "gate", "--seed", str(a.seed),
            "--no-move", "--out", str(outdir)]
    if a.extra.strip():
        argv += a.extra.strip().split()

    t0 = time.time()
    rc, err = None, None
    old_argv = sys.argv
    try:
        import bmdrt
        sys.argv = argv
        rc = bmdrt.main()
    except SystemExit as e:                              # noqa: BLE001
        rc = int(e.code or 0)
    except Exception:                                    # noqa: BLE001
        err = traceback.format_exc()
        rc = 1
    finally:
        sys.argv = old_argv
    dt = time.time() - t0

    fp_path = F.write_case(case_dir, extra=dict(
        argv=argv, returncode=rc, elapsed_s=round(dt, 3),
        python=sys.executable, error=err))
    meta = dict(spectrum=a.spectrum, path=a.path, seed=a.seed,
                argv=argv, returncode=rc, elapsed_s=round(dt, 3),
                python=sys.executable, hook_status=status,
                fingerprint_file=str(fp_path.relative_to(ROOT)),
                error=err)
    (outdir / "case_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")

    d = json.loads(fp_path.read_text(encoding="utf-8"))
    print(f"[b3b_case] {a.spectrum} seed={a.seed} rc={rc} "
          f"K 记录数={d['n_k_records']} level={d['level']} "
          f"耗时={dt:.1f}s → {fp_path}")
    if err:
        print(err, file=sys.stderr)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
