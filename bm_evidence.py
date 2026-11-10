# -*- coding: utf-8 -*-
"""证据卡生成器（L3 的唯一输出形态）：只读消费既有 JSON，不重算、不裁决、不排序。

对应 `THREE_LAYER_SCOPE.md` §4。与 `bm_grade.py` 的分工：

  · `bm_grade`：给**一条结论**贴资格档（A/B/C）+ 一句「这条推不出什么」；
  · 本模块：把结论拆成**一条条证据**，每条证据自带 `not`（它不能推出什么）、
    `state`（拿到没拿到）与「缺它还得补什么」。

为什么 `not` 要下沉到每条证据
-----------------------------
选择器级的 `not` 只能写一句「这条推不出机制」。但真正被读错的是**单项证据**：
`A_W>0` 被读成「扩散被证实」、`斜率≠−0.5` 被读成「没有 Ws」、`f_D 在窗外`
被读成「不存在」。所以 §4.1 要求每条证据自带一句 `not`，本模块按表给出。

硬约束（改这个文件前先看）
--------------------------
1. **只读**：入口先 `copy.deepcopy`，再只读；`fingerprint()` 供对拍，
   生成前后输入一个字节都不能变（`evidence/ws_crit/evidence_card_check.py` 会查）。
2. **不重算**：不改 `status` / `choice` / `delta` / `grade`；`grade` 缺失时
   用 `bm_grade` 的同一张表补，并标 `grade_source="recomputed"`，不另立判据。
3. **不排序、不打分、不单选**：卡是清单；`multi_model` 表按输入顺序列 arm，
   没有 `best` / `rank` / `score` / `confidence` 这些键（自检会查）。
4. **每条证据都有 `not`**：缺一条就是 bug（自检会查）。
5. **`next_evidence_needed` 必填且非空**：说不清还缺什么的结论，
   说明证据链没想清楚，不该出现在卡上。

多模型并列（开关）
------------------
`config.json` 的 `evidence.multi_model` 默认 **false**（测试阶段不输出成堆结果）。
打开后（或命令行 `--multi-model`）多给一张表：每个 arm 一行，
`Δ±dse` + 塌陷/出窗标记，**不排序、不单选**。单臂没有配对样本时 Δ 记为
`None`（不补算、不假装有）。

用法
----
    import bm_evidence as EV

    doc = EV.build_document([run_json_dict], labels=["blank"])
    print(EV.render_markdown(doc))

    python bm_evidence.py out/xxx/real_xxx.json --md
    python bm_evidence.py out/_wp2_blank/*/summary.json --multi-model --md

自检：`python bm_evidence.py --selfcheck`
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import bm_grade as G

ROOT = Path(__file__).resolve().parent
CV_MAX_FALLBACK = 0.15

# --------------------------------------------------------------------------
# 证据项台账：一条证据一行，`not_` 是硬要求（没写 = 自检失败）
#   source : 该证据从哪个字段来（写死在代码事实里，便于复核）
#   reads  : 这条证据**可以**读成什么
#   not_   : 这条证据**不能**推出什么（防越界的主手段）
#   need   : 这条证据缺失/不可用时，还差什么
# --------------------------------------------------------------------------
ITEM = {
    # ---------------- 扩散（Ws / Wo / K+1） ----------------
    "gate_shape": dict(
        source="auto_mechanism.gate.status",
        reads="门控看到的低频尾形状（45° 候选 / 感抗尾 / 无候选 / HN 已退化）",
        not_="门控跳过或判感抗尾，只否证「本窗内可见的电容性扩散尾」；"
             "不等于全谱没有扩散，也不等于扩散不存在",
        need=["更宽频窗（f_min 下移）后重跑门控"],
    ),
    "lowf_slope": dict(
        source="auto_mechanism.gate.ws_diag（bm_ws.slope_im）",
        reads="最低频段 log(−Im)–log f 的斜率、线性度与 45° 段长",
        not_="斜率 ≠ −0.5 不能推出「没有 Ws」：弧尾共存时真 Ws 的斜率会被拖到 "
             "−0.41…−0.49（WP2 §3.5.1）",
        need=["把 f_min 下移 1 decade，让 45° 段变长"],
    ),
    "aw_amp": dict(
        source="auto_mechanism.share（bm_ws.ws_share_low）",
        reads="A_W 后验中位数、标准误与相对不确定度 cv",
        not_="「A_W > 0」不等于扩散存在；「A_W 小 / cv 大」也不等于不存在——"
             "只说明幅值没被本窗+本噪声定住",
        need=["更长采样或更低噪声，把 cv(A_W) 压到判据线以下"],
    ),
    "freq_window": dict(
        source="diffusion_resolution（bm_diff）",
        reads="扩散特征频率 f_D 落在窗内还是窗外（窗内/窗下/窗上，由数值比较定）",
        not_="f_D 在窗外不能推出「没有该过程」；只能说「本窗无权评价」（C 档）",
        need=["补测使 f_D 进窗：转折在窗下就下移 f_min，在窗上就上移 f_max"],
    ),
    "paired_loo": dict(
        source="auto_mechanism.delta / dse / threshold",
        reads="两模型逐点配对 LOO 的 Δelpd ± dse 与判决门槛",
        not_="Δ 过线只说明「本窗预测更好」，不能推出「该机制为真」",
        need=["机制唯一性证据：独立重复实验 / 正交表征 / 真值"],
    ),
    "param_corr": dict(
        source="param_corr.strong / medium",
        reads="参数相关矩阵里的强相关对（|ρ| ≥ 0.9）",
        not_="两个参数强相关不等于两个机制等价；只说明边际分布不能单独引用",
        need=["独立实验或正交表征，打破参数简并"],
    ),
    "degeneracy": dict(
        source="degeneracy / n_warburg_like",
        reads="HN 分量是否已呈 Warburg 退化指纹（ζ→1、峰位、implied A）",
        not_="HN 与 Ws 同形不等于「扩散就是原因」；同形只说明这段自由度已被占",
        need=["用温度/厚度/浓度系列把 HN 与扩散分开"],
    ),
    "residual": dict(
        source="component_review.residual_review",
        reads="残差 RMS、ACF、趋势是否已到噪声水平",
        not_="拟合到噪声水平不等于机制正确：多个模型可以同时拟合到噪声",
        need=["重新拟合以产出残差诊断"],
    ),
    "kk": dict(
        source="kk",
        reads="Lin-KK 是否通过（数据自洽性）",
        not_="KK 通过只说数据满足因果/线性，不说明任何模型或机制正确",
        need=["重测或检查端部处理"],
    ),
    "cross_model": dict(
        source="（缺）多模型并列",
        reads="两两模型在同一谱上的预测差 max|ΔZ|/σ（包络内/外）",
        not_="预测重叠不能推出「机制等价」；只说明本窗+本噪声不可分（B 档）",
        need=["同一谱上的 Ws/Wo/K+1 配对 LOO 与逐点预测差（multi_model 表）"],
    ),
    # ---------------- 吸附（R ∥ L） ----------------
    "ad_gate_shape": dict(
        source="ad_gate.status / checks",
        reads="吸附门控的三闸结果（内部极小 / 半圆完整性 / 可复现性）",
        not_="门控没进不等于「没有吸附」；只说明本窗内没看到完整的感性环形状",
        need=["更宽频窗或更低 f_min，覆盖吸附环的完整半圆"],
    ),
    "ad_amp": dict(
        source="p_ad_small",
        reads="P(R_ad < 3σ̂)：吸附电阻是否落在噪声量级",
        not_="R_ad 小不等于没有吸附：幅值不可辨识只说明定不住，不说明不存在",
        need=["更低噪声 / 更长采样，把 R_ad 从噪声里拉出来"],
    ),
    "ad_corner": dict(
        source="f_c_out_prob / L_ad_uH",
        reads="吸附拐点频率 f_c 是否在窗内（出窗即退化为串联 L）",
        not_="f_c 在窗外不等于没有吸附；串联 L 只是本窗内的等效描述",
        need=["把 f_min 下移，让 f_c 进窗"],
    ),
    "ad_paired_loo": dict(
        source="ad_selection.delta / dse",
        reads="none vs R∥L 的逐点配对 LOO Δelpd ± dse",
        not_="Δ 过线只说明本窗预测更好，不能推出吸附机制为真",
        need=["独立实验（浓度 / 电位系列）验证吸附"],
    ),
    # ---------------- 组分可辨识性 ----------------
    "component_peak": dict(
        source="component_review.window_contribution",
        reads="该分量峰位后验出窗概率 p_peak_out 与 f0",
        not_="峰位在窗外不能推出「没有这个分量」；只能说本窗没覆盖它的特征频率（C 档）",
        need=["把 f_min 下移，使 f0 进窗"],
    ),
    "component_contrib": dict(
        source="component_review.window_contribution",
        reads="该分量在 f_min / 几何中心处对总阻抗的贡献（幅值与虚部占比）",
        not_="窗内有贡献不等于该分量是独立的物理过程；贡献大也可能只是拟合自由度",
        need=["独立频段或正交表征确认该分量的物理归属"],
    ),
    "component_param": dict(
        source="component_review.parameter_identifiability",
        reads="该分量的 ln τ / ln w 与其他参数的强相关对",
        not_="强相关不等于该分量不存在；只说明这些参数只能定出一个组合",
        need=["独立实验打破 τ–w 简并"],
    ),
    "component_degeneracy": dict(
        source="degeneracy",
        reads="该分量的退化指纹（at_bound、φ 后验、implied A）",
        not_="退化指纹不等于该分量就是 Warburg；只说明它和扩散同形",
        need=["温度 / 厚度系列区分 HN 与扩散"],
    ),
    # ---------------- K 选择（L2 结论） ----------------
    "k_loo": dict(
        source="elpd_diff / dse（arviz：正数=更差）",
        reads="K 的配对 LOO 差与 dse",
        not_="K 选出来只说明本窗预测最好，不说明有 K 个物理过程",
        need=["独立实验或正交表征确认过程数"],
    ),
    "k_hard": dict(
        source="sample_grade / grade_fail / frac_neg / pareto_k",
        reads="采样分级、硬筛理由（Pareto-k、φζ>1 占比、LOO 可用性）",
        not_="采样通过不等于模型正确；它只说明这次拟合可信",
        need=["提高采样预算或重跑"],
    ),
    "k_residual": dict(
        source="component_review.residual_review / cover_re / cover_im",
        reads="PPC 覆盖率、ACF 与残差趋势",
        not_="残差白不等于机制正确；多个 K 可以同样白",
        need=["补残差诊断产物（重跑）"],
    ),
    "k_components": dict(
        source="component_review.component_identifiability",
        reads="各分量峰位是否在窗内（in / uncertain / out）",
        not_="分量状态是「本窗可辨识性」，不是「物理过程存在性」",
        need=["更宽频窗或独立实验"],
    ),
}

# 选择器未运行时按 A 处理；这里给「为什么」的一句话，写进 grade_reason。
# K 选择状态 → 档（K 本身是 L2 模型比较，不是 L1 质量）
K_STATUS_CLASS = {
    "converged": G.GRADE_C,       # 有结论，但缺机制唯一性证据
    "degraded": G.GRADE_B,        # 只到「可用」档，相对排序
    "rejected": G.GRADE_A,        # 全被剔除，结果无物理意义
    "exploratory": G.GRADE_A,     # 预算不足，不能当结论
    "not_compared": G.GRADE_A,    # 只拟合了一个 K
    "rejected_by_kk": G.GRADE_A,  # L1 门禁拦下
}

# 吸附门控状态 → 档（门控只到「本窗不支持」，属数据边界 B）
AD_GATE_CLASS = {
    "strong_candidate": G.GRADE_A,   # 有资格比较却没跑 → 算法侧缺项
    "partial_arc": G.GRADE_B,
    "edge_peak": G.GRADE_B,
    "no_candidate": G.GRADE_B,
    "capacitive_tail": G.GRADE_B,
    "noise_only": G.GRADE_B,
    "monotonic_rise": G.GRADE_B,
}

# 档位自带的最低补充要求（保证 next_evidence_needed 非空）
GRADE_NEED = {
    G.GRADE_A: "先修/重跑：A 档是工程问题，重跑一次可能就变（不是数据结论）",
    G.GRADE_B: "并列保留候选模型，不要单选；要分开需更宽频窗或独立重复实验",
    G.GRADE_C: "机制唯一性证据：独立重复实验 / 正交表征 / 真值",
    None: "该卡不落在 A/B/C 限制内；若要做机制结论，仍需独立实验",
}

_GRADE_SRC = {
    "field": "字段（选择器已写）",
    "recomputed": "按 bm_grade 同一张表补算（原 JSON 无 grade 字段）",
    "derived": "按状态表派生（该结论没有选择器级 grade）",
}


# --------------------------------------------------------------------------
# 小工具：数值清洗 / 取字段 / 指纹
# --------------------------------------------------------------------------
def _finite(v):
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _num(v):
    return float(v) if _finite(v) else None


def _get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _clean(v):
    """递归把 nan/inf/numpy 标量转成 JSON 友好的值。"""
    try:
        import numpy as np                      # 只在有 numpy 时用
        if isinstance(v, np.generic):
            v = v.item()
    except Exception:                            # noqa: BLE001
        pass
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def fingerprint(obj) -> str:
    """稳定指纹：只用于证明「生成前后输入没变」。"""
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=repr)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cv_max():
    try:
        import bm_ws
        return float(getattr(bm_ws, "CV_MAX", CV_MAX_FALLBACK))
    except Exception:                            # noqa: BLE001
        return CV_MAX_FALLBACK


def _grade_of(sel):
    """选择器 dict → (grade, reason, source)。缺 grade 时用 bm_grade 补，不另判。"""
    if not isinstance(sel, dict):
        d = G.annotate({"status": "not_tested"})
        return d["grade"], d["grade_reason"], "derived"
    if "grade" in sel:
        return sel.get("grade"), sel.get("grade_reason"), "field"
    d = G.annotate(dict(sel))                   # annotate 会改传入的 dict → 传副本
    return d.get("grade"), d.get("grade_reason"), "recomputed"


def _ev(item, value=None, state="read", note=None, extra_need=None):
    """构造一条证据。state: read / unusable / missing。"""
    spec = ITEM[item]
    need = []
    if state != "read":
        need += list(spec.get("need") or [])
    if extra_need:
        need += list(extra_need)
    out = dict(item=item, state=state, value=_clean(value),
               source=spec["source"], reads_as=spec["reads"])
    out["not"] = spec["not_"]       # `not` 是关键字，只能这样写
    if note:
        out["note"] = note
    if need:
        out["need"] = list(dict.fromkeys(need))
    return out


def _card(claim_id, claim, grade, reason, src, evidence, notes=None):
    needs = []
    for e in evidence:
        needs += list(e.get("need") or [])
    gneed = GRADE_NEED.get(grade) if grade else GRADE_NEED[None]
    if gneed:
        needs.append(gneed)
    d = dict(
        claim_id=claim_id, claim=claim,
        grade=grade, grade_reason=reason, grade_source=src,
        evidence=evidence,
        next_evidence_needed=list(dict.fromkeys(needs)),
        notes=list(notes or []),
    )
    d["not"] = G.reads_as(grade)
    return d


# --------------------------------------------------------------------------
# 各张卡
# --------------------------------------------------------------------------
def _diffusion_card(run):
    am = run.get("auto_mechanism")
    dres = run.get("diffusion_resolution") or {}
    ev, notes = [], []

    if isinstance(am, dict):
        gate = am.get("gate") or {}
        st = gate.get("status")
        ev.append(_ev("gate_shape", dict(status=st, enter=gate.get("enter")),
                      state=("read" if st else "missing"),
                      note=gate.get("reason")))
        wd = gate.get("ws_diag") or {}
        if wd:
            usable = bool(wd.get("im_usable"))
            ev.append(_ev(
                "lowf_slope",
                dict(slope_im=_num(wd.get("slope_im")), r2=_num(wd.get("r2_im")),
                     band_dec=_num(wd.get("band_dec")), term=wd.get("term")),
                state=("read" if usable else "unusable"),
                note=None if usable else "斜率不可用（-Im 在低频端 ≤0 或线性度不足）"))
        else:
            ev.append(_ev("lowf_slope", None, "missing"))

        share = am.get("share") or {}
        cv = _num(share.get("cv_aw"))
        if cv is not None:
            ev.append(_ev(
                "aw_amp",
                dict(A_W=_num(share.get("A_W")), sd=_num(share.get("A_W_sd")),
                     cv=cv, identifiable=share.get("identifiable")),
                state="read",
                extra_need=([f"cv(A_W)={cv:.2f} > {_cv_max():.2f}：幅值没被定住，"
                             f"重跑不会变，需要更宽窗或更低噪声"]
                            if cv > _cv_max() else None)))
        else:
            dp_aw = _num((run.get("diffusion_params") or {}).get("Aw"))
            aw = _num(share.get("A_W"))
            if aw is None:
                aw = dp_aw
            if aw is not None:
                ev.append(_ev("aw_amp", dict(A_W=aw),
                              state="unusable",
                              note="只有 A_W 点估计，没有 sd/cv（该 run 的 share 未算 cv_aw）"))
            else:
                ev.append(_ev("aw_amp", None, "missing"))

        delta, dse = _num(am.get("delta")), _num(am.get("dse"))
        ev.append(_ev(
            "paired_loo",
            dict(delta=delta, dse=dse, threshold=_num(am.get("threshold")),
                 choice=am.get("choice"), status=am.get("status"), n=am.get("n")),
            state=("read" if delta is not None else "unusable"),
            note=None if delta is not None else "没有可配对的逐点 LOO"))
        grade, reason, src = _grade_of(am)
    else:
        for it in ("gate_shape", "lowf_slope", "aw_amp", "paired_loo"):
            ev.append(_ev(it, None, "missing"))
        grade, reason, src = _grade_of(None)
        notes.append("该谱未运行 --auto-mechanism / --auto-mechanism-full："
                     "这些「缺」是没跑，不是数据说没有")

    lvl = dres.get("level")
    if lvl in (None, "no_diffusion") or not dres:
        note = "未启用扩散项，没有 τ_D 后验"
        if isinstance(am, dict) and "ws_resolvable" in am:
            note = ("本 run 落盘的是所选模型（none），Ws 臂的 τ_D 后验没保存；"
                    f"auto_mechanism.ws_resolvable={am.get('ws_resolvable')}。"
                    "要看 f_D 需要把 Ws 臂结果单独落盘")
        ev.append(_ev("freq_window", None, "missing", note=note))
    elif lvl == "unknown":
        ev.append(_ev("freq_window", dict(level=lvl), "unusable",
                      note="模型里没有 tauD 后验（配置缺项）"))
    else:
        # `bm_diff.diffusion_resolution.below_fD` 是 `bool(f.min() < f_D)`：
        # **True 表示窗口里有点低于 f_D（转折可能进窗）**，不是「f_D 在窗下」。
        # 名字容易读反，这里一律用 f_D / f_min / f_max 的数值直接判位置。
        fD = _num(dres.get("f_D"))
        fmin = _num(dres.get("f_min"))
        fmax = _num(dres.get("f_max"))
        pos, extra_need = "unknown", None
        if fD is not None and fmin is not None:
            if fD < fmin:
                pos = "below_fmin"
                dec = math.log10(fmin / fD) if fD > 0 else float("nan")
                extra_need = [f"f_D={fD:.3g} Hz 在窗口下界 {fmin:.3g} Hz 之下 "
                              f"{dec:.2f} decade：当前窗口看不到转折，需要把 f_min 下移"]
            elif fmax is not None and fD > fmax:
                pos = "above_fmax"
                extra_need = [f"f_D={fD:.3g} Hz 高于窗口上界 {fmax:.3g} Hz："
                              "本窗只看到低频渐近段，看不到转折"]
            else:
                pos = "in_window"
        ev.append(_ev(
            "freq_window",
            dict(level=lvl, f_D=fD, f_min=fmin, f_max=fmax,
                 window_position=pos,
                 max_reldiff=_num(dres.get("max_reldiff")),
                 tauD_max_resolvable=_num(dres.get("tauD_max_resolvable"))),
            state="read", extra_need=extra_need))

    pc = run.get("param_corr") or {}
    strong = pc.get("strong") or []
    if pc:
        ev.append(_ev(
            "param_corr",
            dict(n_strong=pc.get("n_strong", len(strong)),
                 top=[dict(a=x.get("a"), b=x.get("b"), rho=_num(x.get("rho")))
                      for x in strong[:5]],
                 level=_get(run, "component_review", "parameter_identifiability",
                            "level")),
            state="read",
            note=pc.get("note")))
    else:
        ev.append(_ev("param_corr", None, "missing"))

    deg = run.get("degeneracy") or []
    nw = int(run.get("n_warburg_like") or 0)
    if deg or "n_warburg_like" in run:
        ev.append(_ev(
            "degeneracy",
            dict(n_warburg_like=nw,
                 levels=[d.get("level") for d in deg],
                 at_bound_max=max([_num(d.get("at_bound")) or 0.0 for d in deg],
                                  default=0.0)),
            state="read",
            note=None))
    else:
        ev.append(_ev("degeneracy", None, "missing"))

    rr = _get(run, "component_review", "residual_review")
    if rr:
        ev.append(_ev("residual",
                      dict(acf=_num(rr.get("acf")),
                           trend_re_sigma=_num(rr.get("trend_re_sigma")),
                           trend_im_sigma=_num(rr.get("trend_im_sigma")),
                           systematic=rr.get("systematic")),
                      state="read"))
    elif _finite(run.get("rms_re")):
        ev.append(_ev("residual", dict(rms_re=_num(run.get("rms_re")),
                                       rms_im=_num(run.get("rms_im")),
                                       acf=_num(run.get("acf"))),
                      state="read", note="无残差审查块，只有 RMS/ACF"))
    else:
        ev.append(_ev("residual", None, "missing"))

    kk = run.get("kk")
    if isinstance(kk, dict):
        ev.append(_ev("kk", dict(status=kk.get("status"), mu=_num(kk.get("mu")),
                                 M=kk.get("M")),
                      state="read", note=kk.get("reason")))
    else:
        ev.append(_ev("kk", None, "missing"))

    # 单臂没有跨模型预测差：如实标缺，不补算
    ev.append(_ev("cross_model", None, "missing",
                  note="单臂 JSON 只有自己；要跨模型差跑 multi_model"))

    return _card("diffusion", "本谱低频尾需要 ω^(-1/2) 扩散项（Ws）",
                 grade, reason, src, ev, notes)


def _adsorption_card(run):
    sel = run.get("ad_selection")
    gate = run.get("ad_gate") or {}
    notes, ev = [], []

    st = gate.get("status")
    if st:
        ev.append(_ev("ad_gate_shape",
                      dict(status=st, strong=gate.get("strong"),
                           enter=gate.get("enter")),
                      state="read", note=gate.get("reason")))
    else:
        ev.append(_ev("ad_gate_shape", None, "missing"))

    p_ad = _num(run.get("p_ad_small"))
    ev.append(_ev("ad_amp", dict(p_ad_small=p_ad),
                  state=("read" if p_ad is not None else "missing"),
                  extra_need=(["R_ad 落在 3σ̂ 以内：幅值不可辨识，不等于没有吸附"]
                              if p_ad is not None and p_ad > 0.5 else None)))

    fc = _num(run.get("f_c_out_prob"))
    ev.append(_ev("ad_corner",
                  dict(f_c_out_prob=fc, L_ad_uH=_num(run.get("L_ad_uH"))),
                  state=("read" if fc is not None else "missing"),
                  extra_need=(["f_c 大概率在窗外：本窗只看到串联 L 的等效行为"]
                              if fc is not None and fc > 0.5 else None)))

    if isinstance(sel, dict):
        delta, dse = _num(sel.get("delta")), _num(sel.get("dse"))
        ev.append(_ev("ad_paired_loo",
                      dict(delta=delta, dse=dse, choice=sel.get("choice"),
                           status=sel.get("status"), n=sel.get("n")),
                      state=("read" if delta is not None else "unusable")))
        grade, reason, src = _grade_of(sel)
    elif st:
        grade = AD_GATE_CLASS.get(st, G.GRADE_A)
        if st == "strong_candidate":
            reason = ("由 ad_gate.status=strong_candidate 派生 A：门控已给比较资格，"
                      "但 ad_selection 缺失（算法侧缺项，不是数据结论）")
        else:
            reason = (f"由 ad_gate.status={st} 派生 {grade}：门控未进入模型比较，"
                      f"只说明本窗没有可比较的吸附候选，不是「没有吸附」")
        src = "derived"
        ev.append(_ev("ad_paired_loo", None, "missing",
                      note="门控未进入比较，没有 Δelpd"))
        notes.append("ad_selection 为空是门控没进，不是选择器说「没吸附」")
    else:
        grade, reason, src = _grade_of(None)
        ev.append(_ev("ad_paired_loo", None, "missing"))
        notes.append("该谱未运行 --auto-adsorption，吸附支路未检验")

    return _card("adsorption", "低频感性特征需要并联吸附支路（R ∥ L）",
                 grade, reason, src, ev, notes)


def _component_cards(run):
    cr = run.get("component_review") or {}
    states = cr.get("component_identifiability") or {}
    cg = cr.get("component_grade") or {}
    contrib = {int(c.get("component", 0)): c
               for c in (cr.get("window_contribution") or [])
               if isinstance(c, dict) and c.get("component")}
    pc = run.get("param_corr") or {}
    deg = run.get("degeneracy") or []
    cards = []

    for key in sorted(states, key=lambda s: int(s.split("_")[-1])
                      if s.split("_")[-1].isdigit() else 99):
        state = states[key]
        if state == "in_window":
            continue
        idx = int(key.split("_")[-1]) if key.split("_")[-1].isdigit() else None
        c = contrib.get(idx) or {}
        p = _num(c.get("p_peak_out"))
        grade = cg.get(key) if key in cg else G.component_grade(state, p)

        ev = []
        ev.append(_ev("component_peak",
                      dict(state=state, p_peak_out=p, f0_hz=_num(c.get("f0_hz")),
                           w_ohm=_num(c.get("w_ohm"))),
                      state="read",
                      extra_need=(["峰位在窗外：本窗没覆盖它的特征频率（C 档）"]
                                  if state == "out_of_window" else None)))
        ev.append(_ev("component_contrib",
                      dict(frac_abs_at_fmin=_num(c.get("frac_abs_at_fmin")),
                           frac_im_at_fmin=_num(c.get("frac_im_at_fmin")),
                           frac_abs_at_geom=_num(c.get("frac_abs_at_geom"))),
                      state="read"))

        pairs = []
        if idx is not None:
            want = (f"ln τ_{idx}", f"ln w_{idx}")
            pairs = [dict(a=x.get("a"), b=x.get("b"), rho=_num(x.get("rho")))
                     for x in (pc.get("strong") or [])
                     if x.get("a") in want or x.get("b") in want]
        ev.append(_ev("component_param", dict(pairs=pairs, n=len(pairs)),
                      state=("read" if pc else "missing"),
                      extra_need=(["τ–w 强相关：两个参数只能定出一个组合"]
                                  if pairs else None)))

        d = deg[idx - 1] if (idx and 0 < idx <= len(deg)) else {}
        ev.append(_ev("component_degeneracy",
                      dict(level=d.get("level"), at_bound=_num(d.get("at_bound")),
                           phi=_num(d.get("phi")), a_impl=_num(d.get("a_impl"))),
                      state=("read" if d else "missing")))

        if state == "out_of_window":
            reason = (f"峰位出窗（P={p:.2f}）判 C：参数定得住、预测算得出，"
                      f"但本窗没覆盖该特征频率，无权说该过程存在与否")
        elif p is None:
            reason = "后验缺失判 A：拿不到 p_peak_out，属计算问题"
        else:
            reason = (f"峰位状态 uncertain（0.05 < P={p:.2f} ≤ 0.5）判 B："
                      f"数据只给了「说不清」，不能单选")
        cards.append(_card(
            f"component_{idx}", f"component_{idx} 对应一个独立物理弛豫过程",
            grade, reason, ("field" if key in cg else "derived"), ev,
            ["组分审查不改变 K；这条卡只说明该分量不被本窗约束，"
             "不是「分量不存在」"]))
    return cards


def _k_card(run):
    K = run.get("K")
    st = run.get("k_selection_status")
    grade = K_STATUS_CLASS.get(st, G.GRADE_A)
    reason = (f"由 k_selection_status={st} 派生 {grade}："
              + (f"K={K} 是 L2 模型比较结论，缺「过程数=K」的唯一性证据"
                 if st == "converged" else
                 "该状态不是可引用的 K 结论"))
    ev = []
    ed, dse = _num(run.get("elpd_diff")), _num(run.get("dse"))
    ev.append(_ev("k_loo", dict(K=K, elpd_diff=ed, dse=dse),
                  state=("read" if ed is not None else "missing"),
                  note="elpd_diff 为 arviz 口径：正数表示相对最好者更差"))
    ev.append(_ev("k_hard",
                  dict(sample_grade=run.get("sample_grade"),
                       grade_fail=run.get("grade_fail"),
                       frac_neg=_num(run.get("frac_neg")),
                       pareto_k_max=_num(_get(run, "component_review",
                                              "loo_robustness", "pareto_k_max"))),
                  state=("read" if run.get("sample_grade") is not None else "missing")))
    rr = _get(run, "component_review", "residual_review")
    ev.append(_ev("k_residual",
                  dict(cover_re=_num(run.get("cover_re")),
                       cover_im=_num(run.get("cover_im")),
                       acf=_num(run.get("acf")),
                       systematic=(rr or {}).get("systematic")),
                  state=("read" if rr or _finite(run.get("cover_re")) else "missing")))
    states = _get(run, "component_review", "component_identifiability") or {}
    ev.append(_ev("k_components",
                  dict(states=states,
                       n_out=sum(1 for v in states.values() if v == "out_of_window")),
                  state=("read" if states else "missing")))
    claim = (f"K={K} 个 RC 分量是本频窗内的模型选择结果" if K is not None
             else "K 未知（本次没有拟合结果）")
    return _card("k_selection", claim,
                 grade, reason, "derived", ev,
                 ["K 是模型选择结论，不是物理过程数；引用 K 时要同时引用组分审查"])


def cards_for_run(run):
    """一个 run JSON → 证据卡列表（只读；按 claim_id 固定顺序，不排序）。"""
    src = copy.deepcopy(run)                    # 硬约束 1：只读副本
    cards = [_diffusion_card(src), _adsorption_card(src)]
    cards += _component_cards(src)
    cards.append(_k_card(src))
    return cards


# --------------------------------------------------------------------------
# 多模型并列（开关默认关）
# --------------------------------------------------------------------------
def _arm_row(label, run):
    am = run.get("auto_mechanism") or {}
    share = am.get("share") or {}
    dp = run.get("diffusion_params") or {}
    dres = run.get("diffusion_resolution") or {}
    cr = run.get("component_review") or {}
    cv_max = _cv_max()

    flags = []
    pds = _num(run.get("p_diff_small"))
    if pds is not None and pds > 0.5:
        flags.append(f"幅值落到噪声（P={pds:.2f}）")
    cv = _num(share.get("cv_aw"))
    if cv is not None and cv > cv_max:
        flags.append(f"幅值未定住（cv={cv:.2f} > {cv_max:.2f}）")
    # 只标记「转折出窗」；f_D 落在窗内不是异常，也不等于数据里有扩散。
    # 判位置用数值比较，不用 `below_fD`（它的语义是 f_min < f_D，易读反）。
    fD, fmin, fmax = (_num(dres.get("f_D")), _num(dres.get("f_min")),
                      _num(dres.get("f_max")))
    if fD is not None and fmin is not None:
        if fD < fmin:
            flags.append(f"转折在窗外：f_D={fD:.3g} Hz < f_min={fmin:.3g} Hz")
        elif fmax is not None and fD > fmax:
            flags.append(f"转折在窗外：f_D={fD:.3g} Hz > f_max={fmax:.3g} Hz")
    nw = int(run.get("n_warburg_like") or 0)
    if nw > 0:
        flags.append(f"HN 已呈 Warburg 指纹（{nw} 个）")
    out = [k for k, v in (cr.get("component_identifiability") or {}).items()
           if v == "out_of_window"]
    if out:
        flags.append("峰位出窗：" + "、".join(out))

    return dict(
        label=label, K=run.get("K"), include_L=run.get("include_L"),
        diffusion=run.get("diffusion"),
        elpd=_num(run.get("elpd_loo")) if run.get("elpd_loo") is not None
        else _num(run.get("elpd")),
        elpd_se=_num(run.get("elpd_se")),
        delta=_num(am.get("delta")), dse=_num(am.get("dse")),
        threshold=_num(am.get("threshold")),
        choice=am.get("choice"), status=am.get("status"),
        grade=am.get("grade") if "grade" in am else None,
        A_W=(_num(share.get("A_W"))
             if share.get("A_W") is not None else _num(dp.get("Aw"))),
        cv_aw=cv, flags=flags,
    )


def multi_model_table(items):
    """items: [(label, run)] → 并列表。按输入顺序，不排序、不打分、不单选。"""
    rows = [_arm_row(lbl, run) for lbl, run in items]
    aws = [r["A_W"] for r in rows if _finite(r["A_W"])]
    spread = None
    if len(aws) >= 2 and min(aws) > 0:
        spread = dict(min=min(aws), max=max(aws), ratio=max(aws) / min(aws))
    return dict(
        kind="l2_multi_model",
        note="按输入顺序列出，不排序、不打分、不单选；每行的 Δ 只相对它自己的 "
             "none 基线，跨臂不做配对检验（那需要 WP2 式的逐点样本）",
        rows=rows,
        amp_spread=spread,
        amp_spread_note=("跨臂 A_W 范围只作描述：哪个 arm 塌到小值由人看，"
                         "程序不判「哪个对」") if spread else None,
    )


# --------------------------------------------------------------------------
# 文档 / 载入 / 渲染
# --------------------------------------------------------------------------
def build_document(runs, labels=None, multi_model=False):
    if labels is None:
        labels = [f"run{i + 1}" for i in range(len(runs))]
    if len(labels) != len(runs):
        raise ValueError("labels 与 runs 数量不一致")
    blocks = []
    for lbl, run in zip(labels, runs):
        blocks.append(dict(label=lbl,
                           file=run.get("file"),
                           path=run.get("_source_path"),
                           cards=cards_for_run(run)))
    doc = dict(
        schema="evidence_card/v1",
        generated=datetime.now().isoformat(timespec="seconds"),
        note="证据卡不排序、不打分、不单选；每条证据自带「这条推不出什么」",
        runs=blocks,
    )
    if multi_model:
        doc["multi_model"] = multi_model_table(list(zip(labels, runs)))
    return doc


def load_run(path):
    """读 bm_real 结果 JSON 或 bmdrt 批处理 summary（summary 会顺着 json 字段取全量）。"""
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(d, dict) and isinstance(d.get("runs"), list) and d["runs"]:
        row = dict(d["runs"][0])
        jp = row.get("json")
        if jp and Path(jp).exists():
            full = json.loads(Path(jp).read_text(encoding="utf-8"))
            merged = dict(row)
            merged.update(full)
            return merged
        return row
    return d


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        parts = [f"{k}={_fmt(x)}" for k, x in v.items() if x is not None]
        return "，".join(parts) if parts else "—"
    if isinstance(v, (list, tuple)):
        return "；".join(_fmt(x) for x in v) if v else "—"
    return str(v)


def _md_value(e):
    """证据值的一行摘要（不展开 list-of-dict）。"""
    v = e.get("value")
    if v is None:
        return "未采到"
    if isinstance(v, dict):
        keep = []
        for k, x in v.items():
            if isinstance(x, (list, dict)):
                if k == "top" and x:
                    keep.append("top=" + "、".join(
                        f"{p.get('a')}↔{p.get('b')} ρ={_fmt(p.get('rho'))}"
                        for p in x[:3]))
                elif k == "pairs" and x:
                    keep.append("、".join(
                        f"{p.get('a')}↔{p.get('b')} ρ={_fmt(p.get('rho'))}"
                        for p in x[:3]))
                elif k == "states" and x:
                    keep.append("、".join(f"{a}:{b}" for a, b in x.items()))
                continue
            if x is not None:
                keep.append(f"{k}={_fmt(x)}")
        return "，".join(keep) if keep else "—"
    return _fmt(v)


def render_markdown(doc):
    L = [f"# 证据卡（只读生成，{doc.get('schema')}）", ""]
    L.append("> " + doc.get("note", ""))
    L.append("> 资格档：**A** 算法无法判断（重跑/修代码）｜**B** 数据无法判断"
             "（并列，不得单选）｜**C** 物理证据不足（只报相容）。**不是分数**。")
    L.append("")

    for blk in doc.get("runs", []):
        L += [f"## {blk.get('label')}", ""]
        if blk.get("path") or blk.get("file"):
            L.append(f"来源：`{blk.get('path') or blk.get('file')}`")
            L.append("")
        cards = blk.get("cards", [])
        if cards:
            L.append("| # | 卡 | 资格档 |")
            L.append("|---|---|---|")
            for i, c in enumerate(cards, 1):
                L.append(f"| {i} | {c['claim_id']} | {c.get('grade') or '—'} |")
            L.append("")
        for i, card in enumerate(cards, 1):
            g = card.get("grade")
            L.append(f"### 卡 {i}｜{card['claim']}")
            L.append("")
            L.append(f"- **资格档：{g if g else '—（无限制）'}**"
                     f" — {G.explain(g) if g else '该结论不落在 A/B/C 三档内'}")
            L.append(f"- 判据：{card.get('grade_reason') or '—'}")
            L.append(f"- 判据来源：{_GRADE_SRC.get(card.get('grade_source'), card.get('grade_source'))}")
            L.append(f"- **整卡推不出**：{card.get('not')}")
            for n in card.get("notes") or []:
                L.append(f"- 备注：{n}")
            L.append("")

            got = [e for e in card.get("evidence", []) if e.get("state") == "read"]
            miss = [e for e in card.get("evidence", []) if e.get("state") != "read"]
            if got:
                L.append("已采到的证据：")
                L.append("")
                for e in got:
                    L.append(f"- **`{e['item']}`** {_md_value(e)}")
                    L.append(f"  - 可以读成：{e['reads_as']}")
                    L.append(f"  - **推不出**：{e['not']}")
                    L.append(f"  - 来源：`{e['source']}`")
                    if e.get("note"):
                        L.append(f"  - 备注：{e['note']}")
                L.append("")
            if miss:
                L.append("未采到 / 不可用：")
                L.append("")
                for e in miss:
                    why = "不可用" if e.get("state") == "unusable" else "缺"
                    val = _md_value(e) if e.get("value") is not None else ""
                    tail = f"；备注：{e['note']}" if e.get("note") else ""
                    L.append(f"- `{e['item']}`（{why}）"
                             f"{(' ' + val) if val else ''}{tail} — "
                             f"**推不出**：{e['not']}")
                L.append("")
            L.append("还缺什么（next_evidence_needed）：")
            L.append("")
            for x in card.get("next_evidence_needed") or []:
                L.append(f"- {x}")
            L.append("")

    mm = doc.get("multi_model")
    if mm:
        L += ["## L2 多模型并列（开关已打开）", ""]
        L.append("> " + mm.get("note", ""))
        L.append("")
        L.append("| arm | K | 扩散 | elpd ± se | Δ vs none | dse | 档 | A_W | 塌陷/出窗标记 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in mm.get("rows", []):
            L.append("| {l} | {k} | {d} | {e} | {dl} | {dse} | {g} | {aw} | {f} |".format(
                l=r.get("label"), k=_fmt(r.get("K")), d=r.get("diffusion") or "none",
                e=(f"{_fmt(r.get('elpd'))} ± {_fmt(r.get('elpd_se'))}"
                   if r.get("elpd") is not None else "—"),
                dl=_fmt(r.get("delta")), dse=_fmt(r.get("dse")),
                g=r.get("grade") or "—", aw=_fmt(r.get("A_W")),
                f="；".join(r.get("flags") or []) or "—"))
        L.append("")
        sp = mm.get("amp_spread")
        if sp:
            L.append(f"跨臂 A_W 范围：{sp['min']:.4g} – {sp['max']:.4g}"
                     f"（{sp['ratio']:.1f}×）。{mm.get('amp_spread_note')}")
            L.append("")
    return "\n".join(L).rstrip() + "\n"


# --------------------------------------------------------------------------
# 自检（只查本模块的硬约束；端到端对拍见 evidence/ws_crit/evidence_card_check.py）
# --------------------------------------------------------------------------
def _selfcheck():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        assert cond, msg
        ok += 1

    # 台账完整性
    for k, spec in ITEM.items():
        chk(spec.get("not_") and len(spec["not_"]) > 6, f"{k} 缺 not")
        chk(spec.get("reads") and spec.get("source"), f"{k} 缺 reads/source")
    chk(len(ITEM) >= 20, f"证据项只有 {len(ITEM)} 条")

    run = dict(
        file="synthetic", K=2, include_L=True, k_selection_status="converged",
        elpd_diff=0.0, dse=0.0, sample_grade=2, grade_fail=[], frac_neg=0.0,
        cover_re=0.9, cover_im=0.9, acf=0.1,
        p_diff_small=0.9, p_ad_small=None, f_c_out_prob=None,
        n_warburg_like=0,
        param_corr=dict(strong=[dict(a="ln τ_2", b="ln w_2", rho=0.94)],
                        n_strong=1, note=""),
        degeneracy=[dict(level="no", at_bound=0.0, phi=0.51, a_impl=6.7)],
        diffusion_resolution=dict(level="indistinguishable", f_D=0.071,
                                  f_min=0.1, below_fD=0.1466, max_reldiff=0.01,
                                  tauD_max_resolvable=1.59),
        auto_mechanism=dict(
            choice="none", status="guard_degraded", delta=0.41, dse=1.36,
            threshold=2.04, n=50, grade="B",
            grade_reason="由 guards: diff_amp_unidentifiable 判 B",
            gate=dict(status="candidate", enter=True, reason="候选",
                      ws_diag=dict(slope_im=-0.11, r2_im=0.9, im_usable=False,
                                   band_dec=0.2, term="to_cap")),
            share=dict(A_W=0.78, A_W_sd=0.19, cv_aw=0.24, identifiable=False)),
        component_review=dict(
            component_identifiability={"component_1": "in_window",
                                       "component_2": "out_of_window"},
            window_contribution=[
                dict(component=1, state="in_window", p_peak_out=0.0, f0_hz=358.0,
                     w_ohm=34.6, frac_abs_at_fmin=0.6, frac_im_at_fmin=-0.05,
                     frac_abs_at_geom=0.63),
                dict(component=2, state="out_of_window", p_peak_out=0.99,
                     f0_hz=0.071, w_ohm=10.0, frac_abs_at_fmin=0.086,
                     frac_im_at_fmin=-0.95, frac_abs_at_geom=0.009)],
            parameter_identifiability=dict(strong_pairs=[], n_strong=0,
                                           max_at_bound=0.0, level="degenerate"),
            residual_review=dict(acf=0.38, trend_re_sigma=1.0, trend_im_sigma=1.09,
                                 systematic=True),
            loo_robustness=dict(available=True, pareto_k_max=0.75),
        ),
    )
    fp = fingerprint(run)
    cards = cards_for_run(run)
    chk(fingerprint(run) == fp, "生成过程改了输入 dict")
    chk(len(cards) == 4, f"应出 4 张卡（扩散/吸附/组分2/K），实为 {len(cards)}")

    for c in cards:
        chk(c["next_evidence_needed"], f"{c['claim_id']} 的 next 为空")
        chk(isinstance(c["grade"], (str, type(None))), "grade 只能是 str/None")
        for e in c["evidence"]:
            chk(e["not"], f"{c['claim_id']}/{e['item']} 缺 not")
        chk("score" not in json.dumps(c).lower(), "不允许出现 score")
        chk("confidence" not in json.dumps(c, ensure_ascii=False).lower(),
            "不允许出现 confidence")

    dcard = cards[0]
    chk(dcard["grade"] == "B", "扩散卡应沿用 auto_mechanism.grade=B")
    chk(dcard["grade_source"] == "field", "有 grade 字段应标 field")
    chk(any(e["item"] == "freq_window" and e["state"] == "read"
            for e in dcard["evidence"]), "freq_window 应可读")
    comp2 = [c for c in cards if c["claim_id"] == "component_2"][0]
    chk(comp2["grade"] == "C", "出窗分量应为 C")

    # 未跑门控 → A，且必须说明「是没跑，不是没有」
    bare = dict(file="bare", K=1, k_selection_status="not_compared")
    bcards = cards_for_run(bare)
    chk(bcards[0]["grade"] == "A", "未跑机制门控应为 A")
    chk(any("未运行" in n for n in bcards[0]["notes"]), "未跑门控应写明")

    # JSON 必须能严格序列化（allow_nan=False）
    json.dumps(build_document([run], ["s"]), ensure_ascii=False, allow_nan=False)

    # 多模型：默认关；开了也不许出现选择/排序字段
    doc_off = build_document([run], ["s"])
    chk("multi_model" not in doc_off, "默认必须关掉多模型")
    doc_on = build_document([run, run], ["a", "b"], multi_model=True)
    chk("multi_model" in doc_on and len(doc_on["multi_model"]["rows"]) == 2,
        "开开关后应有两个 arm")
    blob = json.dumps(doc_on, ensure_ascii=False)
    for bad in ('"best"', '"rank"', '"selected"', '"recommend"'):
        chk(bad not in blob, f"多模型表不得出现 {bad}")

    print(f"bm_evidence 自检 {ok}/{ok} 通过")
    return ok


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _load_cfg():
    p = ROOT / "config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("evidence") or {}
    except Exception:                            # noqa: BLE001
        return {}


def main(argv=None):
    ap = argparse.ArgumentParser(description="证据卡生成器（只读）")
    ap.add_argument("inputs", nargs="*", help="run JSON 或 bmdrt summary.json")
    ap.add_argument("--labels", help="逗号分隔的 arm 名（默认用文件名）")
    ap.add_argument("--multi-model", dest="multi_model", action="store_true",
                    default=None, help="打开 L2 多模型并列表")
    ap.add_argument("--no-multi-model", dest="multi_model", action="store_false",
                    help="强制关闭（覆盖 config）")
    ap.add_argument("--md", action="store_true", help="输出 Markdown（默认）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--out", help="写到文件而不是 stdout")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args(argv)

    if a.selfcheck:
        _selfcheck()
        return 0
    if not a.inputs:
        ap.error("需要至少一个输入文件")

    runs, labels = [], []
    for p in a.inputs:
        try:
            r = load_run(p)
            if isinstance(r, dict):
                r["_source_path"] = str(Path(p))
            runs.append(r)
        except Exception as e:                   # noqa: BLE001
            print(f"[bm_evidence] 跳过 {p}：{type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        labels.append(Path(p).parent.name if Path(p).name.endswith("summary.json")
                      else Path(p).stem.replace("real_", ""))
    if not runs:
        return 2
    if a.labels:
        labels = [s.strip() for s in a.labels.split(",")]
        if len(labels) != len(runs):
            print(f"[bm_evidence] --labels 有 {len(labels)} 个，输入 {len(runs)} 个",
                  file=sys.stderr)
            return 2

    cfg = _load_cfg()
    mm = a.multi_model if a.multi_model is not None \
        else bool(cfg.get("multi_model", False))
    doc = build_document(runs, labels, multi_model=mm)
    text = json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) \
        if a.json else render_markdown(doc)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"已写 {a.out}（{len(runs)} 个 run，"
              f"多模型{'开' if mm else '关'}）")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
