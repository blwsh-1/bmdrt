# 噪声口径（likelihood）：两个默认值为什么不统一

**日期**：2026-11-23　**状态**：生产入口新增 `--likelihood free|perpoint`（默认 `free`，逐位不变）
**改动**：`bm_real.run` / `bmdrt.py --likelihood` / `circuit_runner.build_candidates` / `plan.json`
**判据**：无改动。M-a/M-b/M-c、A1–A4、`acf≤0.50` 全部不动。

---

## 1. 结论：口径不由模型定，由"跟谁比"定

| 场景 | 口径 | 理由 |
|---|---|---|
| `bmdrt auto`（生产单谱、入台账） | **free**（默认） | 历史 239 条 stage0 记录、既有全部 json 都是 free；换默认 = 新旧不可比 |
| `bmdrt compare` / §13 / §14 / 任何跨族对照 | **perpoint**（默认） | `circuit_runner` 恒 perpoint；voigt 与 nested 必须同口径才可比 |
| `ref_rwp` 参考指标 | 固定 `2%·max\|Z\|`（既有实现） | 独立 LS，唯一用途是与外部工具对表 |

**两个默认值不同是有意的，不要"统一"**。统一任何一边都要付代价：
统一到 free ⇒ §13/§14 的 4/14、0/3 全部失效必须重跑，且跨族 ELPD 本来就不成立；
统一到 perpoint ⇒ 239 条历史记录与新生结果不可比，且 free 是"σ 由数据自报"的唯一通道。

## 2. 为什么跨族对照必须 perpoint

1. **同噪声**：`σ_j = 2%·|Z_j|` 由同一输入 `Z` 算出，所有候选拿到**逐位相同**的数组
   ⇒ 满足"候选间同噪声"。free 下每个候选的 σ 是各自采样的后验，**跨候选的
   ELPD/rms 比较本身就不成立**（不是数值差，是对象不同）。
2. **与历史一致**：`circuit_runner.fit_candidate` 的 perpoint 分支自 2026-11 起就是
   §13/§14 的实际口径（`sigma_from_data=True`）。跑 free 等于换尺子。
3. **代价要认**：perpoint 下 `elpd_validity.sigma_leak=True`（σ 含被留出点自身）
   ⇒ **`elpd_loo` 不得当样本外预测证据**。这是已知代价，不是新问题；M4 的族选择
   本来就不用 ELPD（`elpd` 只作谱内 K 比较）。

## 3. 为什么生产端保持 free

- free 把 σ 当被采样位点，是"这条谱自己报告噪声多大"。生产端没有外部给定的 2%
  依据——2% 是从验收口径借来的，借来的常数不该悄悄变成生产默认。
- `sigma_floor` 只在 free 下有意义；perpoint 与 floor 互斥（`bm_top._check_sigma`
  显式报错）。生产端要地板就必须 free。
- 逐位不变：不传 `--likelihood` 时 `sigma=None`，与 2026-11-23 之前**完全同路径**。

## 4. 修掉的一个真缺陷：候选登记曾把 free 写成 perpoint

`circuit_runner.build_candidates` 原先 7 处硬编码 `likelihood_profile="perpoint-2pct"`，
于是 `run(likelihood="free")` 产出的候选：

- `plan.json` / `describe()` 里 **自述为 perpoint**；
- `candidate_id` 与真正 perpoint 跑的候选**完全相同**（`likelihood` 进 id 哈希）。

`cache_key` 靠 `|perpoint` 后缀区分，缓存没串，但**事后读产物分不清哪条是 free**。
现改为按 `likelihood` 参数登记：`free` → `"free"`，`perpoint` → `"perpoint-2pct"`（历史名不变）。

**影响**：perpoint 路径的 `candidate_id` 逐位不变（既有产物全部有效）；
**free 路径的 `candidate_id` 变了**（此前借的是 perpoint 的名字），free 的历史缓存
会失效并重算。这是"修正错误登记"的必然代价，不是回归。

`plan.json` 同时新增 `likelihood` 字段：看计划就该知道这一批是在哪个 σ 下比的。

## 5. 口径不能跨读的三件事

- `rms` / `ELPD` / `acf` 三者**都随口径变**：perpoint 更偏爱小 `|Z|` 的高频点，
  同一配置 `Rwp` 会降（实测 M2 9.816% → 8.613%）。这个降幅**不是模型更好**，
  是权重变了。
- `acf` 会跟着变（实测同谱 0.747 → 0.640），但 **0.50 阈值本身与口径无关**
  （它标定在"残差序列的白噪声分位"上）。acf 下降 ≠ 残差更白。
- `--ref-rwp` 是第三把尺子（`2%·max|Z|` + 独立 LS），三个数**互不可比**。

## 6. 复现

```bash
# 生产（free，默认）
python bmdrt.py auto 谱.csv
# 生产但显式 perpoint（与验收口径对齐；结果 json 里 likelihood=perpoint）
python bmdrt.py auto 谱.csv --likelihood perpoint
# 跨族对照（perpoint 默认）
python bmdrt.py compare 谱.csv --ks 1,2 --layers 1,2
```

`--likelihood` 接受 `per-point` / `per_point` / `fixed`（归一成 `perpoint`）、
`sampled`（归一成 `free`），落盘一律规范名；非法值直接退出。
