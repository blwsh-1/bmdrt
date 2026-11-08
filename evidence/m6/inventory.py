# -*- coding: utf-8 -*-
"""M6 E1：数据指纹与纳入清单（只读，不拟合）。

对应 `M6_VALIDATION_PLAN.md` §6.0。**只做四件事**：算指纹、量形状、记来源、
标置信。不做剔除、不做拟合、不改原始文件。

为什么单独一个脚本而不是塞进 `sampling_check.py`
------------------------------------------------
`sampling_check.py` 的职责是"抽样验证"（选 4+10 条、跑 summarize），它的输出依赖
抽样 seed；E1 要的是**全量、与抽样无关的身份表**。两者混在一起会让"清单"随抽样
策略漂移，而清单必须可独立复核。

三条边界（写进代码里，免得结果被当成别的东西）
----------------------------------------------
1. **文件名推断 ≠ 实验记录**。`sample_id_confidence` / `group_id_confidence` 默认
   `filename`，`independence_status` 默认 `inferred`。**E1 不产出 `confirmed`**——
   那是人工核对实验记录后才能填的值，脚本无权升级。
2. **不判"有没有环"**。本脚本只记 `n_im_pos`（Im>0 的点数）与 `im_at_fmin` 这类
   **原始符号事实**，不调用 `detect_ad`、不跑 DRT、不下"无环"结论。用门控自己的
   判据去证明"这批谱无环"是循环论证（`M6_VALIDATION_PLAN.md` §4.3.2b）。
3. **异常值只记录不剔除**。`n_nonfinite` / `n_dup_freq` / `n_nonmonotonic` /
   `n_re_leq0` 全部如实计数，纳不纳入由已冻结的 §2.2 规则决定，不由本脚本决定。

用法：
    python evidence/m6/inventory.py --selftest     # 先自检
    python evidence/m6/inventory.py                # 盘点，写 csv + log
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # bmdrt/
ZDIR = r"D:\EIS 数据\EIS"                              # 外部原始导出，仓外
ZCSV = os.path.join(ROOT, "out", "_m6_samples", "z_csv")
ARCHIVE = os.path.join(ROOT, "archive")
E2DIR = os.path.join(ROOT, "EISmeasurements", "csv_2e")

OUT_CSV = os.path.join(HERE, "data_inventory.csv")
OUT_LOG = os.path.join(HERE, "inventory_log.txt")

FIELDS = [
    "dataset", "role", "stem", "path", "sha256", "bytes",
    "n_pts", "f_min_hz", "f_max_hz", "decades",
    "n_nonfinite", "n_dup_freq", "n_nonmonotonic", "n_re_leq0",     "n_im_pos", "n_pts_lf", "n_im_pos_lf", "im_max_lf", "im_at_lf_min",
    "lf_interior_min",
    "f_im_pos_min_hz", "f_im_pos_max_hz", "im_at_fmin",
    "j_mA_cm2", "sample_id", "sample_id_confidence",
    "group_id", "group_id_confidence", "exp_id",
    "independence_status", "in_scope", "notes",
]


# --------------------------------------------------------------------------
# 读入：三种格式，统一返回 (f, Z)
# --------------------------------------------------------------------------
def read_z(path):
    """ZView `.z`：跳前 11 行，逗号分隔，列 0=f / 4=Z' / 5=Z''。

    与 `sampling_check.read_z()` 保持一致的解析规则（E1 要核对的就是这条转换链，
    解析口径若不同，核对就没有意义）。
    """
    rows = []
    with open(path, encoding="utf-8-sig", errors="replace") as fp:
        for ln in fp.readlines()[11:]:
            p = ln.strip().split(",")
            if len(p) >= 6:
                try:
                    rows.append([float(p[0]), float(p[4]), float(p[5])])
                except ValueError:
                    continue
    a = np.asarray(rows, dtype=float)
    if a.size == 0:
        return np.array([]), np.array([], dtype=complex)
    return a[:, 0], a[:, 1] + 1j * a[:, 2]


def read_csv3(path):
    """`f_Hz,Zre,Zim` 三列（.z 转换产物与 2E csv 同格式）。"""
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return a[:, 0], a[:, 1] + 1j * a[:, 2]


def read_blank_txt(path):
    """Blank 的 ZView 文本导出：TSV，表头带 BOM，行尾 CRLF，列 0/4/5。"""
    rows = []
    with open(path, encoding="utf-8-sig", errors="replace") as fp:
        for ln in fp:
            p = ln.rstrip("\r\n").split("\t")
            if len(p) >= 6:
                try:
                    rows.append([float(p[0]), float(p[4]), float(p[5])])
                except ValueError:
                    continue
    a = np.asarray(rows, dtype=float)
    if a.size == 0:
        return np.array([]), np.array([], dtype=complex)
    return a[:, 0], a[:, 1] + 1j * a[:, 2]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            b = fp.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# 形状度量：只记录事实，不判定好坏
# --------------------------------------------------------------------------
def metrics(f, Z):
    """形状度量。**分全频段与低频段两套 Im 符号指标**。

    为什么必须分开：全频段 `Im>0` 几乎在所有谱上都成立（高频感抗），它反映的是
    电感，不是吸附环。要检验"这批谱无环"只能看**低频段**（最低 40% 对数频宽，
    与 `detect_ad` 的 `frac_lo=0.40` 同一划分；这里只借用频段划分，不借用它的判据）。
    """
    m = dict(n_pts=0, f_min_hz="", f_max_hz="", decades="",
             n_nonfinite=0, n_dup_freq=0, n_nonmonotonic=0, n_re_leq0=0,
             n_im_pos=0, n_pts_lf=0, n_im_pos_lf=0, im_max_lf="", im_at_lf_min="",
             lf_interior_min="",
             f_im_pos_min_hz="", f_im_pos_max_hz="", im_at_fmin="")
    if f.size == 0:
        return m
    m["n_pts"] = int(f.size)
    fin = np.isfinite(f) & np.isfinite(Z.real) & np.isfinite(Z.imag)
    m["n_nonfinite"] = int((~fin).sum())
    fv, Zv = f[fin], Z[fin]
    if fv.size == 0:
        return m

    m["f_min_hz"] = "%.6g" % float(fv.min())
    m["f_max_hz"] = "%.6g" % float(fv.max())
    if fv.min() > 0 and fv.max() > 0:
        m["decades"] = "%.3f" % float(np.log10(fv.max() / fv.min()))

    # 重复频点：先按 f 排序后数相邻相等（原始顺序不假设有序）
    fs = np.sort(fv)
    m["n_dup_freq"] = int(np.count_nonzero(np.diff(fs) == 0))
    # 原始顺序是否非单调（降序按"递减"也算单调，这里只看是否单调）
    d = np.diff(fv)
    m["n_nonmonotonic"] = int(np.count_nonzero(d * d[0] < 0)) if d.size else 0

    m["n_re_leq0"] = int(np.count_nonzero(Zv.real <= 0))
    pos = Zv.imag > 0
    m["n_im_pos"] = int(pos.sum())
    if pos.any():
        m["f_im_pos_min_hz"] = "%.6g" % float(fv[pos].min())
        m["f_im_pos_max_hz"] = "%.6g" % float(fv[pos].max())
    i_low = int(np.argmin(fv))
    m["im_at_fmin"] = "%.6g" % float(Zv.imag[i_low])

    # ---- 低频段（最低 40% 对数频宽）----
    o = np.argsort(fv)
    fl, Zl = fv[o], Zv[o]
    lf = np.log10(np.clip(fl, 1e-30, None))
    sel = lf <= float(np.quantile(lf, 0.40))
    fb, Zb = fl[sel], Zl[sel]
    if fb.size:
        m["n_pts_lf"] = int(fb.size)
        m["n_im_pos_lf"] = int(np.count_nonzero(Zb.imag > 0))
        m["im_max_lf"] = "%.6g" % float(Zb.imag.max())
        m["im_at_lf_min"] = "%.6g" % float(Zb.imag[0])       # 最低频点（已升序）
        # 低频段是否存在"内部极小"（先降后升）：峰位不在两端各 2 点以内
        k = int(np.argmin(-Zb.imag))                          # Im 最大处
        k_left = k
        k_right = Zb.size - 1 - k
        m["lf_interior_min"] = "yes" if (2 <= k_left and 2 <= k_right) else "no"
    return m


# --------------------------------------------------------------------------
# 身份推断：只做字符串拆解，每条都留 note 说明推断依据
# --------------------------------------------------------------------------
def identity_2e(stem):
    """`2EMC1-3_exp2_0100mA` → sample=2EMC1 / group=2EMC1-3 / exp=exp2。

    **不合并 `2EMC1-1` 与 `2EMC1-2`**：连字符后那一位是"第几次实验"还是"第几个电极"，
    文件名判不出来，所以 group 保留到 `2EMC1-1` 一层，sample 只到 `2EMC1`。
    `2EP-1` / `2EP-2` 同理会拆成 sample=2EP、group=2EP-1，这与 2EMC1-1 的 `-1`
    语义未必相同，notes 里如实标注。
    """
    core = re.sub(r"_\d+mA$", "", stem)          # 先剥电流段，再剥 exp
    exp = ""
    m = re.match(r"^(.*)_(exp\d+)$", core)
    if m:
        core, exp = m.group(1), m.group(2)
    m = re.match(r"^(2E[A-Za-z]*\d*)", core)
    if m:
        sample = m.group(1)
    else:
        sample = core.split("_")[0]
    return dict(sample_id=sample, group_id=core, exp_id=exp,
                notes=("group=文件名去 exp/电流段；sample=前缀；"
                       "`-N` 语义（第几次实验 or 第几个电极）未确认"))


def identity_az31(stem):
    """`AZ31-4Li_100` → sample=AZ31-4Li / group=AZ31-4Li / exp=100(浸泡时间)。"""
    m = re.match(r"^(AZ31-\d+Li)_(\d+)$", stem)
    if m:
        return dict(sample_id=m.group(1), group_id=m.group(1), exp_id=m.group(2),
                    notes="后缀为浸泡时间(min)，非独立实验重复")
    return dict(sample_id=stem, group_id=stem, exp_id="",
                notes="不符合 AZ31-<n>Li_<t> 命名，人工核对")


def identity_blank(stem):
    """`Blank_..._1.2V` / `..._1.2V_3` → 同一样品的重复测量。"""
    m = re.match(r"^(Blank_.*?1\.2V)(?:_(\d+))?$", stem)
    if m:
        return dict(sample_id=m.group(1), group_id=m.group(1),
                    exp_id=(m.group(2) or "0"),
                    notes="后缀数字为重复次数，文件名推断" if m.group(2)
                          else "主谱")
    return dict(sample_id=stem, group_id=stem, exp_id="", notes="命名不符，人工核对")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def collect(log):
    rows = []

    def add(dataset, role, stem, path, f, Z, ident, in_scope, extra_note="",
            sha=None, nbytes=None):
        m = metrics(f, Z)
        note = ident.get("notes", "")
        if extra_note:
            note = (note + "；" + extra_note) if note else extra_note
        rows.append(dict(
            dataset=dataset, role=role, stem=stem,
            path=os.path.relpath(path, ROOT).replace("\\", "/")
            if path.startswith(ROOT) else path.replace("\\", "/"),
            sha256=sha if sha is not None else sha256_file(path),
            bytes=nbytes if nbytes is not None else os.path.getsize(path),
            j_mA_cm2="",
            sample_id=ident["sample_id"], sample_id_confidence="filename",
            group_id=ident["group_id"], group_id_confidence="filename",
            exp_id=ident["exp_id"], independence_status="inferred",
            in_scope=in_scope, notes=note,
            **m))

    # ---- 1. AZ31 原始 .z（仓外，只读）----
    zs = sorted(p for p in os.listdir(ZDIR) if p.lower().endswith(".z")) \
        if os.path.isdir(ZDIR) else []
    if not zs:
        log("!! 找不到 .z 目录：%s" % ZDIR)
    for name in zs:
        p = os.path.join(ZDIR, name)
        f, Z = read_z(p)
        add("az31_raw", "real_ring_series", os.path.splitext(name)[0], p, f, Z,
            identity_az31(os.path.splitext(name)[0]), "yes")

    # ---- 2. AZ31 转换后 csv（E2 的实际输入）----
    if not os.path.isdir(ZCSV):
        log("!! 缺 %s —— 必须先跑 sampling_check.py prepare" % ZCSV)
        return rows, False
    zc = sorted(p for p in os.listdir(ZCSV) if p.endswith(".csv"))
    for name in zc:
        p = os.path.join(ZCSV, name)
        f, Z = read_csv3(p)
        add("az31_csv", "model_input", os.path.splitext(name)[0], p, f, Z,
            identity_az31(os.path.splitext(name)[0]), "yes",
            extra_note="由 .z 转换，供 bmdrt.py auto 读取")

    # ---- 3. AZ31 同源 txt（核对，不纳入）----
    for name in sorted(p for p in os.listdir(ARCHIVE)
                       if p.startswith("AZ31-") and p.endswith(".txt")):
        p = os.path.join(ARCHIVE, name)
        f, Z = read_blank_txt(p)
        add("az31_txt", "cross_check", os.path.splitext(name)[0], p, f, Z,
            identity_az31(os.path.splitext(name)[0]), "no",
            extra_note="与 .z 同源的文本版，仅作核对，不重复计入")

    # ---- 4. Blank 对照 ----
    for name in sorted(p for p in os.listdir(ARCHIVE)
                       if p.startswith("Blank_") and p.endswith(".txt")):
        p = os.path.join(ARCHIVE, name)
        f, Z = read_blank_txt(p)
        add("blank", "negative_control", os.path.splitext(name)[0], p, f, Z,
            identity_blank(os.path.splitext(name)[0]), "yes")

    # ---- 5. 2E 117 条 ----
    man_path = os.path.join(E2DIR, "manifest.csv")
    jmap = {}
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8-sig") as fp:
            for r in csv.DictReader(fp):
                jmap[r["stem"]] = r.get("j_mA_cm2", "")
    e2 = sorted(p for p in os.listdir(E2DIR)
                if p.endswith(".csv") and p != "manifest.csv")
    for name in e2:
        p = os.path.join(E2DIR, name)
        stem = os.path.splitext(name)[0]
        f, Z = read_csv3(p)
        add("2e", "production_2e", stem, p, f, Z, identity_2e(stem), "yes")
        rows[-1]["j_mA_cm2"] = jmap.get(stem, "")

    # ---- 6. archive 里的 2E 副本（核对，不纳入）----
    for name in sorted(p for p in os.listdir(ARCHIVE)
                       if p.startswith("2E") and p.endswith(".csv")):
        p = os.path.join(ARCHIVE, name)
        stem = os.path.splitext(name)[0]
        f, Z = read_csv3(p)
        base = re.sub(r"_\d+$", "", stem)
        add("2e_archive", "cross_check", stem, p, f, Z, identity_2e(base), "no",
            extra_note="archive 副本，与 csv_2e 去重核对，不重复计入")
        rows[-1]["j_mA_cm2"] = jmap.get(base, "")
        rows[-1]["_base"] = base                     # 去重用的归一名（不写进 CSV）
    return rows, True


def cross_checks(rows, log):
    """一致性核对：.z ↔ 转换 csv、csv_2e ↔ archive 副本。"""
    raw = {r["stem"]: r for r in rows if r["dataset"] == "az31_raw"}
    cv = {r["stem"]: r for r in rows if r["dataset"] == "az31_csv"}
    mism = []
    for stem, a in raw.items():
        b = cv.get(stem)
        if b is None:
            mism.append((stem, "转换产物缺失"))
            continue
        if a["n_pts"] != b["n_pts"]:
            mism.append((stem, "点数 %s→%s" % (a["n_pts"], b["n_pts"])))
        if a["f_min_hz"] != b["f_min_hz"] or a["f_max_hz"] != b["f_max_hz"]:
            mism.append((stem, "频段 %s-%s → %s-%s" % (
                a["f_min_hz"], a["f_max_hz"], b["f_min_hz"], b["f_max_hz"])))
    log("一致性：.z ↔ z_csv 共 %d 对，不一致 %d 对" % (len(raw), len(mism)))
    for s, why in mism:
        log("   ! %-24s %s" % (s, why))

    prod = {r["stem"]: r for r in rows if r["dataset"] == "2e"}
    dup = {r["stem"]: r for r in rows if r["dataset"] == "2e_archive"}
    same = diff = 0
    for stem, a in dup.items():
        base = a.get("_base", stem)
        b = prod.get(base)
        if b is None:
            log("   · archive 副本 %s（归一为 %s）在 csv_2e 无同名，需人工判断"
                % (stem, base))
            continue
        if a["sha256"] == b["sha256"]:
            same += 1
        else:
            diff += 1
            log("   · 同名不同内容：%s → %s（点数 %s/%s）"
                % (stem, base, a["n_pts"], b["n_pts"]))
    log("一致性：2E archive 副本 %d 条，字节相同 %d，不同 %d（以 csv_2e 为准）"
        % (len(dup), same, diff))
    return len(mism) == 0


def summarize(rows, log):
    from collections import Counter
    c = Counter((r["dataset"], r["in_scope"]) for r in rows)
    log("")
    log("汇总（dataset, in_scope, 条数）：")
    for (d, s), n in sorted(c.items()):
        log("   %-12s %-4s %3d" % (d, s, n))
    scope = [r for r in rows if r["in_scope"] == "yes"]
    log("纳入清单合计 %d 条（AZ31 %d 含 raw+csv 两份指纹 / Blank %d / 2E %d）"
        % (len(scope),
           sum(1 for r in scope if r["dataset"].startswith("az31")),
           sum(1 for r in scope if r["dataset"] == "blank"),
           sum(1 for r in scope if r["dataset"] == "2e")))

    # 低频段才是"有没有环"的相关指标；全频段 Im>0 主要反映高频感抗
    lf_clean = [r for r in scope if str(r["n_im_pos_lf"]) == "0"]
    interior = [r for r in scope if r.get("lf_interior_min") == "yes"]
    log("")
    log("原始符号事实（不判定有无环，也不调用 detect_ad）：")
    log("   全频段含 Im>0 的谱：%d/%d —— 主要由高频感抗贡献，不能用来判环"
        % (sum(1 for r in scope if str(r["n_im_pos"]) not in ("0", "")), len(scope)))
    log("   低频段 Im 全程 ≤0 的谱：%d/%d（§4.3.2b 条件 1 的候选）"
        % (len(lf_clean), len(scope)))
    log("   低频段存在内部极小（先降后升，条件 2 的候选）：%d/%d"
        % (len(interior), len(scope)))
    for ds in ("2e", "blank", "az31_csv"):
        g = [r for r in scope if r["dataset"] == ds]
        if not g:
            continue
        log("     %-9s 低频无 Im>0 %3d/%3d（低频段点数 %s-%s），内部极小 %3d/%3d，"
            "最低频 Im>0 %3d/%3d"
            % (ds,
               sum(1 for r in g if str(r["n_im_pos_lf"]) == "0"), len(g),
               min(r["n_pts_lf"] for r in g), max(r["n_pts_lf"] for r in g),
               sum(1 for r in g if r.get("lf_interior_min") == "yes"), len(g),
               sum(1 for r in g if r["im_at_lf_min"] not in ("", None)
                   and float(r["im_at_lf_min"]) > 0), len(g)))
    log("")
    log("解读边界（重要，别把符号事实当成结论）：")
    log("   · 上表的 Im>0 是**原始符号**，不是'有感抗峰'。低阻谱（|Z| 在 mΩ 量级）")
    log("     的 Zre/Zim 在零附近抖动，Im 变号可以纯粹是噪声。")
    log("     已知反例：2EMC1-2_0750mA 是 AD_M6_SAMPLING §3.2 记录的噪声谱（KK 拒绝，")
    log("     中段残差 27.6%），它的 Zre 在 -0.012~0.049 之间跳、Zim 在 ±0.01 跳，")
    log("     最低频 Im=+0.0109 完全是噪声，不是环。")
    log("   · 所以 E1 既不能确认'无环'，也不能确认'有环'。它能证明的只有一件事：")
    log("     '117 条无环'这个项目标注**没有经过数据层面检验**，因此 §4.3.2b 默认")
    log("     truth=unknown 是必要的，而不是过度保守。")
    log("   · 要判环需要信噪比 / 形状 / DRT 证据（§4.3.2b 的四条），且不得复用")
    log("     detect_ad 自身判据。那属 E3，不在 E1。")
    c2 = Counter(r["independence_status"] for r in rows if r["in_scope"] == "yes")
    log("independence_status 分布：%s" % dict(c2))
    if not c2.get("confirmed"):
        log("   → 无 confirmed：§4.3.3 第二层（推广上界）状态维持 pending。")


def write_csv(rows):
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------
def selftest():
    ok = 0
    fail = []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    import tempfile
    f = np.array([100.0, 10.0, 1.0])
    Z = np.array([1 + 0.5j, 2 - 0.1j, 3 + 0.2j])
    m = metrics(f, Z)
    chk("n_pts", m["n_pts"] == 3)
    chk("decades=2", m["decades"] == "2.000")
    chk("n_im_pos=2", m["n_im_pos"] == 2)
    chk("im_at_fmin 取最低频 1Hz 的 +0.2", m["im_at_fmin"] == "0.2")
    chk("f_im_pos_min=1", m["f_im_pos_min_hz"] == "1")
    chk("降序不算非单调", m["n_nonmonotonic"] == 0)

    chk("dup freq", metrics(np.array([1.0, 1.0, 2.0]), np.ones(3, complex))["n_dup_freq"] == 1)
    chk("非单调", metrics(np.array([1.0, 5.0, 2.0]), np.ones(3, complex))["n_nonmonotonic"] >= 1)
    nanZ = np.array([1 + 1j, np.nan + 0j, 3 + 0j])
    chk("nonfinite 计数", metrics(np.array([1.0, 2.0, 3.0]), nanZ)["n_nonfinite"] == 1)
    chk("空数组不炸", metrics(np.array([]), np.array([], complex))["n_pts"] == 0)
    chk("Re<=0 计数", metrics(np.array([1.0, 2.0]), np.array([-1 + 0j, 2 + 0j]))["n_re_leq0"] == 1)

    # 低频段指标：11 点降频，最低 5 点为低频段；中间出现 Im 内部极大=环状拐点
    f11 = 10.0 ** np.arange(10, -1, -1)
    im11 = np.array([0.5] * 6 + [-0.3, -0.4, -0.1, -0.35, -0.5])
    m11 = metrics(f11, 1 + 1j * im11)
    chk("低频段内部极小=yes", m11["lf_interior_min"] == "yes")
    chk("低频段点数=5", m11["n_pts_lf"] == 5)
    chk("低频段无 Im>0", m11["n_im_pos_lf"] == 0)
    chk("im_at_lf_min 取最低频", m11["im_at_lf_min"] == "-0.5")
    chk("im_max_lf=-0.1", m11["im_max_lf"] == "-0.1")
    m12 = metrics(f11, 1 + 1j * np.array([0.5] * 6 + [-0.3, -0.4, -0.5, -0.35, -0.6]))
    chk("峰在边缘不算内部极小", m12["lf_interior_min"] == "no")

    i = identity_2e("2EMC1-3_exp2_0100mA")
    chk("2E sample", i["sample_id"] == "2EMC1")
    chk("2E group", i["group_id"] == "2EMC1-3")
    chk("2E exp", i["exp_id"] == "exp2")
    i = identity_2e("2EMC1-1_0005mA")
    chk("2E 无 exp", i["exp_id"] == "" and i["group_id"] == "2EMC1-1")
    i = identity_2e("2EP-1_exp1_0005mA")
    chk("2EP sample", i["sample_id"] == "2EP" and i["group_id"] == "2EP-1")
    i = identity_az31("AZ31-4Li_100")
    chk("AZ31 sample", i["sample_id"] == "AZ31-4Li" and i["exp_id"] == "100")
    i = identity_blank("Blank_0.1KBr_pH10_EIS_PtCu_1.2V")
    chk("Blank 主谱", i["exp_id"] == "0")
    i = identity_blank("Blank_0.1KBr_pH10_EIS_PtCu_1.2V_3")
    chk("Blank 重复", i["exp_id"] == "3")

    with tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as fp:
        fp.write(b"abc" * 1000)
        tmp = fp.name
    chk("sha256 正确", sha256_file(tmp) == hashlib.sha256(b"abc" * 1000).hexdigest())
    chk("bytes 正确", os.path.getsize(tmp) == 3000)
    os.unlink(tmp)

    print("selftest: %d 通过, %d 失败" % (ok, len(fail)))
    if fail:
        print("  失败项：" + "；".join(fail))
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--log", default=OUT_LOG)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    os.makedirs(HERE, exist_ok=True)
    lines = []

    def log(s=""):
        lines.append(s)
        print(s)

    t0 = time.time()
    log("M6 E1 数据指纹与纳入清单")
    log("时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    log("仓库：%s" % ROOT)
    log("外部原始：%s" % ZDIR)
    log("脚本：evidence/m6/inventory.py（只读；不拟合；不剔除）")
    log("=" * 78)

    rows, have_csv = collect(log)
    if not have_csv:
        log("中止：转换产物缺失，先跑 `python evidence/ad_m6/sampling_check.py prepare`")
        open(a.log, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
        return 2

    good = cross_checks(rows, log)
    summarize(rows, log)

    # 聚合指纹：重跑 prepare 后可直接对比，判断转换链是否改变（幂等性核验）
    log("")
    log("聚合指纹（各 dataset 的 sha256 排序后串联再 sha256）：")
    for ds in ("az31_raw", "az31_csv", "blank", "2e"):
        hs = sorted(r["sha256"] for r in rows if r["dataset"] == ds)
        if hs:
            agg = hashlib.sha256("".join(hs).encode()).hexdigest()
            log("   %-10s n=%-4d %s" % (ds, len(hs), agg))
    write_csv(rows)
    log("")
    log("写出：%s（%d 行 × %d 列）" % (a.out, len(rows), len(FIELDS)))
    log("耗时：%.1fs" % (time.time() - t0))
    if not good:
        log("注意：存在 .z ↔ csv 点数/频段不一致，见上方 ! 行（不影响清单，但要查 prepare）")
    open(a.log, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("\n日志：" + a.log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
