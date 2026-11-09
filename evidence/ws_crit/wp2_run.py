"""WP2 补救重跑用的薄包装：先把采样配置按预注册抬上去，再走正常的 bmdrt 流程。

只用于 `grade=0` 的臂（`WS_DECISION_PROJECT.md` WP2「允许的补救（仅一次）」）：
`max_tree_depth 13→15` + `n_warmup 3000→6000`，其余一律透传，本脚本不写死
任何模型或数据设置。

    python evidence/ws_crit/wp2_run.py --depth 15 --warmup 6000 -- \
        auto examples/example_caseE.csv --krange 2,2 --diffusion Ws \
        --zeta 1.0 --no-L --ta 0.99 --out out/_wp2_caseE/k2_Ws

注意：`bm_real.KW` 必须在 `bmdrt` 真正取用之前改好。`process_one` 里的
`import bm_real as R` 命中 `sys.modules`，所以这里先 import 再改即可生效。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bm_real as R          # noqa: E402
import bmdrt                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    if a.warmup:
        R.KW["n_warmup"] = int(a.warmup)
    if a.depth:
        R.KW["max_tree_depth"] = int(a.depth)

    rest = [x for x in a.rest if x != "--"]
    sys.argv = ["bmdrt.py"] + rest
    print(f"[wp2] 采样配置覆盖：n_warmup={R.KW['n_warmup']} "
          f"max_tree_depth={R.KW['max_tree_depth']} "
          f"n_samples={R.KW['n_samples']} chains={R.KW['num_chains']}")
    return bmdrt.main()


if __name__ == "__main__":
    sys.exit(main())
