# E2 之后的后续（P1–P3）执行记录

生成时间：2026-11-09 14:32

## P1 L1 幅度判据换 `log(-Im)`：**已应用并通过回归**

- `bm_kk.py` sha256：`b47f69d85595` → `230114002295`
- 自动回滚：**否**

### 代码版本溯源（E2 全程在补丁前运行）

E2 的 30 个 job 全部跑在补丁**应用之前**。补丁应用后，在 E2 用到的 31 条谱上，`diffusion_tail_candidate` 有 10 条的输出发生变化：

| 谱 | 字段 | 补丁前 | 补丁后 |
|---|---|---|---|
| `e2:Blank_0.1KBr_pH10_EIS_PtCu_1.2V` | `slope_im` | `None` | `-0.11025721854845034` |
| `e2:Blank_0.1KBr_pH10_EIS_PtCu_1.2V_1` | `slope_im` | `None` | `-0.11025721854845034` |
| `e2:Blank_0.1KBr_pH10_EIS_PtCu_1.2V_2` | `slope_im` | `None` | `-0.11025721854845034` |
| `e2:Blank_0.1KBr_pH10_EIS_PtCu_1.2V_3` | `slope_im` | `None` | `-0.11025721854845034` |
| `e2:Blank_0.1KBr_pH10_EIS_PtCu_1.2V_4` | `slope_im` | `None` | `-0.11025721854845034` |
| `e2:Blank_0.1KBr_pH10_EIS_PtCu_1.2V_5` | `slope_im` | `None` | `-0.11025721854845034` |
| `ex:example_caseE` | `amp_ok` | `0.0` | `1.0` |
| `ex:example_caseE` | `is_candidate` | `0.0` | `1.0` |
| `ex:example_caseE` | `f_cut` | `None` | `0.42986623` |
| `ex:example_caseE` | `slope_im` | `None` | `-0.49236674395527624` |
| `ex:example_caseE_warburg` | `amp_ok` | `0.0` | `1.0` |
| `ex:example_caseE_warburg` | `is_candidate` | `0.0` | `1.0` |
| `ex:example_caseE_warburg` | `f_cut` | `None` | `0.857696` |
| `ex:example_caseE_warburg` | `slope_im` | `None` | `-0.447624445337463` |
| `ex:example_caseF_wo` | `amp_ok` | `1.0` | `0.0` |
| `ex:example_caseF_wo` | `f_cut` | `0.857696` | `None` |
| `ex:example_caseF_wo` | `slope_im` | `None` | `-0.9747170438159376` |
| `ex:example_caseG_wo_only` | `amp_ok` | `1.0` | `0.0` |
| `ex:example_caseG_wo_only` | `f_cut` | `0.857696` | `None` |
| `ex:example_caseG_wo_only` | `slope_im` | `None` | `-0.9645818173281693` |

回归（同一组命令，应用前/后各跑一次）：

| 检查 | 应用前 rc | 应用后 rc | 耗时 |
|---|---|---|---|
| `bm_check nofit` | 0 | 0 | 4.9 s |
| `bm_check fast` | 0 | 0 | 20.3 s |

逐字段快照见 `p1_before.json` / `p1_after.json` / `p1_provenance.json`。

## P2 幅度带宽是否按端点走向分设：**不改**

结论：V1/V2 与 V0 在 9 类 × 20 条上**逐行相同**；`wo_in` 被挡在**相位判据**（最低 25% 段的 Nyquist 斜率中位数 −17.0，0/20 落在 45°±0.35），不是带宽。放宽带宽什么也换不来 —— L1 天生只能做 Ws，Wo 归 `bm_wscrit` 的家族。**生产代码不改**。

## P3 `to_inductive` 是否接到吸附支路做预筛：**不改**

结论：18 条真实谱 `to_inductive` 18/18、`init_reliable` 12/18，加预筛会挡掉 **0/18**（零收益）；合成侧唯一分歧是 `wshort_in` 10/20 误放行，但那条路径不选吸附支路、seed 用不上。**不建议改门控**。

---

## 附：14:36 WP4 注释修正（只改 docstring，行为不变）

上表的 sha `230114002295` 是**补丁本体**刚应用时的状态。随后按
`WS_DECISION_PROJECT.md` WP4 把 `diffusion_tail_candidate` 的 docstring 按真值
重写（删掉「`example_caseE` 无扩散元件 / 被幅度判据挡掉」的错误叙述，补
`-Im <= 0` 早退分支与 `slope_im` 字段说明），**不动任何逻辑**：

- 当前 `bm_kk.py` sha256 = `d10b4c07d73a0884216d61141c3dfdd4f7eda60725a1de6bccb6b8a3b48096f8`
- 校验：31 条谱六字段与 `p1_after.json` **0 处差异**（行为未变）
- 回归：`bm_check.py nofit` 18/18、`fast` 18 通过 / 0 失败 / 7 跳过、
  `bmdrt.py check` 18/18、`test_audit_findings` 4/4、`test_bm_ad_select` 10/10、
  `test_drt_signed` 6/6
- 补丁仍**未 `git commit`**（HEAD `88f6c8d`）；撤销用 `git checkout -- bm_kk.py`
