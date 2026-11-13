# 检出力审计的可复现清单

对应产物：`out/power_audit/{audit.json,summary.md}`、`docs/round_2026-11-13/POWER_AUDIT_FINDINGS.md`
脚本：`docs/round_2026-11-13/power_audit.py`

目的：让「只 clone 本仓库、没有本地 `out/`」的人也能判定这份审计**是不是跑出来的、跑的哪一版**。
（本文件由审查意见「可复现性」一项要求补，2026-11-13。）

## 1. 上游依赖（精确到 commit）

| 依赖 | 版本 | 用途 |
|---|---|---|
| 本仓库 `main` @ **`0fc77d2`** | `docs/round_2026-11-12/truth_recovery.py` | 正演 `forward()`、`acf_max()`、`keys_for()`、`z_cpe()` — 审计直接 import |
| 同上 | `docs/round_2026-11-12/truth_dataset.py` | 168 条真值谱的生成器（重建数据集的唯一途径） |
| 同上 | `bm_top.py` @ `9ff937b` | 容性支路正演（数据生成侧）；审计不 import，交叉校验用 |

`power_audit.py` 用 `sys.path.insert(0, ROOT/"docs"/"round_2026-11-12")` 后 `import truth_recovery as TR`，
**没有第二份正演实现**。若上游 `truth_recovery.forward` 改动，本审计结果即失效。

## 2. 输入（已入库，可用 sha256 核对）

| 文件 | 字节 | sha256 |
|---|---|---|
| `out/truth_recovery/results.json` | 1014622 | `116ee49c7d65d28e290659269a4f8d3ded86bf5f67c7ac48937bd3ebec6d1b43` |
| `out/truth_dataset/truth_manifest.json` | 140289 | `5abbe4d3fc3eee33e31af6d0d355b1a6881d3e9f0a4ea969f5710e8298a0d13f` |
| `out/truth_dataset/*.csv`（168 张） | 900 KB | **不入库**；逐张 sha256 记在 manifest 的 `csv_sha256` 字段 |

168 张 csv 不入库的理由：`truth_dataset.py` 是确定性的（seed 由 `md5(key)[:8]` 决定，
不用内置 `hash`），重建后逐条 sha256 与本 manifest 一致；把 168 个 csv 塞进 git 只增加噪声。
`results.json` 入库的理由：它需要 12.6 min 拟合才能重算，是审计的真正输入。

**已实测（2026-11-13，`--out out/_regen_check`）**：重建 168 条 →
**168/168 条 csv 的 sha256 与 manifest 里的 `csv_sha256` 及库内文件三方一致**；
`truth_manifest.json` 解析后**逐键相同**，但字节不同（多 4741 个 CRLF）——
因为上游脚本用 `Path.write_text`，Windows 上按 `os.linesep` 写；库内副本已归一为 LF。
⇒ **恢复以内容（sha256 / JSON 逐键）为准，不以字节哈希为准**；上游脚本按原样保留不动
（改了就不再是产生 `results.json` 的那一版，provenance 会断）。

## 3. 运行环境

```
Python 3.10.7   (D:/autoeis_env)
numpy  2.2.6
scipy  1.15.3
```
仅用 numpy + scipy（无 jax / numpyro）→ 与生产拟合环境解耦，clone 后装这两个包即可跑。

## 4. 随机性

| 位置 | 种子 | 说明 |
|---|---|---|
| 数据集真值参数、噪声实现 | `md5("<family>\|<difficulty>\|<pct>\|<seed>")[:8]` | 稳定，跨进程一致 |
| A1 符号随机化零分布 | `np.random.default_rng(20261113)` | 逐谱顺序固定（按 `results.json` 顺序） |
| clustered bootstrap | `np.random.default_rng(20261113)` | 同上 |
| `--selftest` 蒙特卡洛 | `np.random.default_rng(20261113)` | — |

⇒ 同一份输入 + 同一 `--boot`，`audit.json` 应逐键一致（本轮验证：两次运行内容相同）。

## 5. 复跑命令

```bash
# 0) 只需要审计（输入已入库）
python docs/round_2026-11-13/power_audit.py --selftest        # 7 项对拍，退出码 0
python docs/round_2026-11-13/power_audit.py --boot 2000       # ~4 s，写 out/power_audit/

# 1) 若要重建数据集（可选）
python docs/round_2026-11-12/truth_dataset.py                 # 168 csv + manifest，确定性

# 2) 若要重建 results.json（12.6 min，99.4% 收敛）
python docs/round_2026-11-12/truth_recovery.py --selftest     # 先对拍
python docs/round_2026-11-12/truth_recovery.py                # 全量 168 × 9 × 3 起点
```

`power_audit.py` 的第一件事是 `repro` 自检：用 `results.json` 里存的参数重建残差，
比对 `rwp_pct` 与 `acf`（容差 1e-6），覆盖 **168 谱 / 1503 个 (谱, 候选) 对**；
任一条不一致就 `SystemExit`，不写产物。⇒ 「残差是从存盘参数算出来的」这件事由脚本自己保证。

## 6. 输出（本版产物）

| 文件 | 字节 | sha256 |
|---|---|---|
| `out/power_audit/audit.json` | 180303 | `2fbf30beed0c73f87e7b174e76e53d8b86f037ff295eba96c3a94a669609bfdb` |
| `out/power_audit/summary.md` | 13974 | `3d65e3200b34d1be7e8f12d3b0a2b0cea52c8bab57bbe1f608317e3ecd655aad` |

`audit.json` 顶层键：`generated_by / repro{n_spectra,n_pairs,mismatch} / boot /
a0_units{S,S_units,by_kind,…} / a1_null{summary,per_spectrum} / a2_power{rows,bins,boundary} /
a4_labels / a4_dab`。

## 7. 已知的不可复现点（诚实登记）

1. **`results.json` 本身**：LS 多起点在 `least_squares` 内部是确定性的，但依赖 scipy 版本与
   BLAS；跨平台重跑可能在某些谱上落到不同局部极小（本轮未验证跨平台一致性）。
   因此审计的「输入」以本文件登记的 sha256 为准，不以「重跑应当一样」为假设。
2. **`--boot` 只影响零分布/区间**：`--boot 200` 与 `--boot 2000` 的 `p` 值会差，
   判据不使用 `p`，故不影响任何结论；引用数字时须写明 `B=2000`。
3. **168 张 csv 不在库内**：clone 后不重建就直接跑 `power_audit.py` 会因缺文件失败，
   这是设计选择（见 §2），不是脚本缺陷。
4. **字节级 vs 内容级**：库内的 `results.json` / `truth_manifest.json` 是 LF 版；
   用上游脚本重跑会在 Windows 上写出 CRLF 版，字节哈希不同、内容相同（已实测，见 §2）。
   比对请用「JSON 逐键 / csv sha256」，不要用字节哈希。
