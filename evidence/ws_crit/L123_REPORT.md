# L1' / L2 / L3 接入报告（2026-11-09）

> 范围：把 Ws 判据的**数据侧端点走向**（L1'）与**候选触发**（L2）接进生产链路，
> 并把**份额诊断**（L3）挂进选择器但**不接决策闸门**。
> 不改 `select_k()` 语义、不改预注册判据、不动 `G8b=pending`。

---

## 1. 先回答「完不完善」

| 层 | 判据本身 | 生产接口 | 结论 |
|---|---|---|---|
| **L1'** 数据侧端点走向 | 完善：7 项自检、9 类 × 20 条 × 2 窗基准 360 条、18 条真实谱登记 | 原先只在 `evidence/ws_crit/`（影子），生产无对应函数 | **已接入**（新建 `bm_ws.py`） |
| **L2** 触发 `diffusion_gate` | 完善：`is_candidate` 已有 P1 回归；端点走向可提前否证 | 已存在，但 `is_candidate` 只打印不决策 | **已接入**（触发条件 + 感抗尾短路） |
| **L3** 简约闸门 | **不完善**：`rpc_share` 只对 Wo 有意义；`ws_share_low` 随窗口下限漂移（0.161 vs 0.057），5% 阈值未标定 | 无 | **只挂诊断，不接闸门**（见 §4） |

> 「L3 不完善」不是实现问题，是**标定问题**：同一条真 `A_W=2` 的谱，0.01 Hz 窗给
> 0.161、0.1 Hz 窗只给 0.057，5% 阈值在窄窗上只剩 14% 余量；错元件（Wo 数据用
> Ws 拟合）是 0.011。分离度不足以支撑一个通用闸门。

---

## 2. 改了什么

| 文件 | 改动 | 是否改行为 |
|---|---|---|
| `bm_ws.py`（新，643 行） | 由 `evidence/ws_crit/bm_wscrit.py` 派生，**只改文件头定位**（加「来源与验证」段）；判据一行未动 | 否 |
| `bm_real.py` | ① 可选 `import bm_ws as WS`；② `diffusion_gate(..., f=None, Z=None)` 增 `ws_diag` 字段；③ 新增 `status="inductive_tail"`；④ 触发条件加 `gate.status=="candidate"`；⑤ 感抗尾短路（`status="skipped_inductive"`）；⑥ `select_mechanism(..., share=None)` 只写 `note` | **是**（仅在 `--auto-mechanism` 下，且方向为「少跑一次拟合」） |
| `bmdrt.py` | `--auto-mechanism` 帮助文本按新触发条件重写 | 否（仅文档） |

**默认路径逐位不变**：`--auto-mechanism` 默认关，`diffusion_gate` 只在它的分支里被调用；
`diffusion_tail_candidate`、`select_k`、`select_mechanism` 的决策规则均未改。

---

## 3. 验证

### 3.1 迁移对拍（`relocate_check.py` → `relocate_check.json`）

`bm_ws` vs `bm_wscrit`：**31 条谱 × 9 个诊断字段 + 8 组 npz × 2 个份额函数，
不一致 0 处**（数值容差 1e-12、nan 对齐）。

### 3.2 L1' 分档（`l123_check.py` → `l123_check.json`）

| 谱 | `is_candidate` | `gate.status` | `enter` | `term` | `family` | `verdict` |
|---|---|---|---|---|---|---|
| `AZ31-4Li_100` | False | `inductive_tail` | False | `to_inductive` | `inductive_tail` | reject |
| `example_caseE` | True | `candidate` | True | `still_45` | `Ws_or_wide_CPE` | candidate |
| `example_caseE_warburg` | True | `candidate` | True | `still_45` | `Ws_or_wide_CPE` | candidate |
| `example_caseF_wo`（真 Wo） | False | `no_candidate` | False | `to_cap` | `Wo_like` | candidate |
| `example_caseE` + `n_warburg_like=1` | True | `degenerate` | False | `still_45` | `Ws_or_wide_CPE` | candidate |

两个关键读数：

- **`caseF_wo` 不会被 L1' 带偏**：数据侧 `verdict=candidate`（`Wo_like`），但 `is_candidate=False`，
  门控仍 `no_candidate` —— L1' 只记录家族，不改门控结论；
- **简并优先**：`n_warburg_like>0` 时即使 `is_candidate=True` 也给 `degenerate`（原逻辑保持）。

### 3.3 L2 一致性（31 条谱）

`verdict == "candidate"` 与 `gate.enter` **不一致 0 条**（`inductive_tail` 除外）。
其中 3 条 `enter=True`：`example_caseE`、`example_caseE_warburg`、`example_caseF_wo`
（后者由 `is_candidate` 强制为 True 的对照构造，实际运行不会）。

### 3.4 实机跑（quick 档，`--krange 1,1`）

| 谱 | 拟合次数 | 门控 | 结果 |
|---|---|---|---|
| `example_caseE` | 2 | `candidate` / `still_45`（45° 段 1.63 dec） | `switched`，Δelpd=131.93 ≥ 25.83（与接入前逐位相同） |
| `AZ31-4Li_100` | **1**（接入前 2） | `inductive_tail` / `to_inductive` | `skipped_inductive`，保持 none |

AZ31 那条此前会白跑一次 Ws、再被 `diff_amp_unidentifiable` 挡回；现在**提前短路**，
结论不变、成本少一次拟合。`--auto-mechanism-full` 时仍强制补跑（验证路径保留）。

### 3.5 回归

`bm_check.py nofit` 18/18、`fast` 18 通过 / 0 失败 / 7 跳过、`bmdrt.py check` 18/18、
`test_audit_findings` 4/4、`test_bm_ad_select` 10/10、`test_drt_signed` 6/6、
`python bm_ws.py` 自检 7/7。

---

## 4. L3 为什么只报不判

`ws_share_low` 实测（`l123_check.json`）：

| 组 | `f_min` | `A_W` | `share` 中位 [90% CI] |
|---|---|---|---|
| `caseE`（真 Ws） | 0.01 Hz | 2.04 | **0.161** [0.157, 0.165] |
| `caseE_warburg`（同一真值，窄窗） | 0.1 Hz | 2.04 | **0.057** [0.053, 0.061] |
| `caseF_wo` 用 Ws 拟合（错元件） | 0.1 Hz | 13.65 | **0.011** [0.004, 0.085] |

- 真 Ws 与错元件在**同一窗口**（0.1 Hz）上是 0.057 vs 0.011，比值 5×，但真 Ws 距
  5% 阈值只有 0.7 个百分点；
- 份额随 `f_min` 系统漂移，`5%` 不是窗口不变量；
- `rpc_share`（分母是极化电阻）对 `Ws` 无意义（`Rw` 是窗口归一化幅值），只适用于 `Wo`，
  而当前 `select_mechanism` 只做 none vs Ws。

→ 所以 `select_mechanism(share=...)` 只把数写进 `note`，**不参与 `choice`**。
要接闸门，先做「份额 × 窗口」的标定（把 `f_min` 归一化，或改用 `Re Z_Ws(f_min)/σ_im`
的信噪比口径——与 §3.1 的 `SNR_im` 同源）。

> **2026-11-09 后续（P7）**：这项标定已做完，结论是**份额不能用**——
> 真 `Ws`、`A_W=0.5`、0.01 Hz 窗的份额只有 **0.048**，5% 阈值直接误杀。
> 闸门改用后验相对不确定度 `cv(A_W) ≤ 0.15`（`bm_ws.CV_MAX`），见
> `L3_CALIB.md`。本节保留为「为什么当时只报不判」的历史记录。

---

## 5. 复现与回退

```bash
PY=D:/autoeis_env/Scripts/python.exe
$PY bm_ws.py                                  # 判据自检 7/7
$PY evidence/ws_crit/relocate_check.py        # 迁移对拍（应 0 差异）
$PY evidence/ws_crit/l123_check.py            # L1'/L2/L3 分档与份额
$PY bmdrt.py auto <谱> --krange 1,1 --quick --auto-mechanism --out out/_l123/x
```

回退：`git revert` 对应提交即可。`bm_ws.py` 是纯新增文件；`bm_real.py` 的六处改动
都集中在 `diffusion_gate` / 触发分支 / `select_mechanism` 的 `share` 参数，
`git checkout -- bm_real.py bmdrt.py` 可整体撤销（不影响 P1 的 `bm_kk.py`）。

---

## 6. 边界（写死）

- L1' 只输出**端点走向与家族**，`still_45` **不等于**「确认 Ws」（`Ws ≡ 窗外 R||CPE(0.5)` 同形未解）；
- `inductive_tail` 只否证**窗内可见的电容性扩散尾**，不等于断言「不存在扩散」
  （吸附环与扩散尾共存时会把后者掩盖）；
- 合成谱的 `truth` 由构造给出，不替代真实谱验收；真实谱仍无扩散真值，
  不得据此算真假阳性、不改 `G8b`；
- `skipped_inductive` 是**新增状态值**，只在 `--auto-mechanism` 分支出现，
  不得在报告里读成「没有扩散」。
