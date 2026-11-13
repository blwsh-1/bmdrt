"""第二段 24 配置设计：格点构造 + 三组分配 + 平衡核验（只算，不拟合）。

为 `STAGE2_SPEC.md` §2（精度反推）与 §3（24 配置分层）提供可复算数字。
本脚本**不生成数据、不拟合、不读任何结果文件**；它只把设计写死成一张可核对的表。

设计变量（配置 = 格点的取值组合）：
  family   {L1, L2}                    嵌套层数（真值支路固定为容性 CPE）
  spacing  {separated, overlap, out_of_window, near_degenerate}
  n_b      {0.60, 0.75, 0.90}          支路 CPE 指数
单元级属性（不进配置因子，理由见 STAGE2_SPEC §3.4）：
  strength {strong 3.0, weak 0.15}     R_b / R_L
特征频率 f_vis（支路主导上界）：|Z_b(f_vis)| = R_L
  ⇒ f_vis = f_b·(R_b/R_L)^(1/n_b)，设计按 f_vis 摆位、由它反解 f_b，
    使「窗内/窗外」与「支路强弱」解耦（同一 f_vis 下强弱不改变位置）。
"""
import math

import numpy as np

F_LO, F_HI, N_F = 0.1, 1e5, 60
F = np.logspace(math.log10(F_LO), math.log10(F_HI), N_F)

RS = 150.0
R1_BASE, R2_BASE = 3e4, 1e5
R_DEGEN = 6e4
L1_F1_SCALE = 0.8            # 让 L1/L2 不再共享同一 f1（避免真值向量跨族重合）

SPACING = {                  # (f1, f2, f_vis, n1, n2)
    "separated":       (3e3, 3e1, 1.0,  0.88, 0.82),
    "overlap":         (4e1, 1.8e1, 6.0, 0.85, 0.83),
    "out_of_window":   (2e4, 5e1, 1e-2, 0.90, 0.80),
    "near_degenerate": (12.0, 10.8, 10.0, 0.80, 0.80),
}
SPACING_ORDER = ("separated", "overlap", "out_of_window", "near_degenerate")
STRENGTH = {"strong": 3.0, "weak": 0.15}
NB_LEVELS = (0.60, 0.75, 0.90)
GROUPS = ("dev", "cal", "hold")
GROUP_ZH = {"dev": "开发", "cal": "校准", "hold": "留出"}

# 支路强弱按单元指定：每个 spacing 跨族各 1 强 1 弱，每族 2 强 2 弱，
# 每组因覆盖全部 8 单元而恰好 4 强 4 弱。
CELL_STRENGTH = {
    ("L1", "separated"): "strong", ("L1", "overlap"): "weak",
    ("L1", "out_of_window"): "strong", ("L1", "near_degenerate"): "weak",
    ("L2", "separated"): "weak", ("L2", "overlap"): "strong",
    ("L2", "out_of_window"): "weak", ("L2", "near_degenerate"): "strong",
}
CELLS = [(fam, sp) for fam in ("L1", "L2") for sp in SPACING_ORDER]

# 各拓扑真正使用的参数（L1 不消费 R2/Q2/n2，比较独立性时不得计入）
USED = {"L1": ("Rs", "R1", "Q1", "n1", "Q_b", "n_b"),
        "L2": ("Rs", "R1", "Q1", "n1", "R2", "Q2", "n2", "Q_b", "n_b")}


def z_cpe(f, Q, n):
    return Q / (1j * 2 * np.pi * np.asarray(f, dtype=float)) ** n


def cpe_q(R, f0, n):
    """Q_data 口径：|Z_CPE(f0)| = R。"""
    return R * (2 * np.pi * f0) ** n


def make_cfg(fam, sp, n_b, strength):
    f1, f2, fvis, n1, n2 = SPACING[sp]
    L = 1 if fam == "L1" else 2
    if L == 1:
        f1 *= L1_F1_SCALE
    R1, R2 = (R_DEGEN, R_DEGEN) if sp == "near_degenerate" else (R1_BASE, R2_BASE)
    R_L = R1 if L == 1 else R2
    Rb = STRENGTH[strength] * R_L
    fb = fvis * (R_L / Rb) ** (1.0 / n_b)
    p = dict(Rs=RS, R1=R1, R2=R2,
             Q1=cpe_q(R1, f1, n1), n1=n1,
             Q2=cpe_q(R2, f2, n2), n2=n2,
             Q_b=cpe_q(Rb, fb, n_b), n_b=n_b)
    got = float(abs(z_cpe(np.array([fvis]), p["Q_b"], n_b)[0]))
    assert abs(got / R_L - 1) < 1e-9, (sp, n_b, strength, got / R_L)
    return dict(fam=fam, sp=sp, n_b=n_b, strength=strength, L=L,
                f1=f1, f2=f2, fvis=fvis, fb=fb, Rb=Rb, R_L=R_L,
                d_f1=abs(math.log10(f1 / fvis)),
                d_f2=(abs(math.log10(f2 / fvis)) if L == 2 else None),
                d_win=math.log10(fvis / F_LO), p=p)


ROWS = []
for i, (fam, sp) in enumerate(CELLS):
    st = CELL_STRENGTH[(fam, sp)]
    for gi, g in enumerate(GROUPS):
        c = make_cfg(fam, sp, NB_LEVELS[(i + gi) % 3], st)
        c.update(cell=i + 1, group=g, cid=f"c{i+1:02d}-{g}")
        ROWS.append(c)


def ci_table(s=0.330, p=0.625):
    """三种尺度并列：正态近似 / t 分布（主口径）/ 伯努利尺度敏感性。

    `p=0.625` 时单配置 sd = sqrt(p(1-p)) = 0.484（本轮实测命中率均值下的伯努利 sd），
    `s=0.5` 是 0/1 结果的**严格上界**（p=0.5 时最大）。两者都不是"配置间实测 sd"。
    """
    from scipy import stats
    out = []
    for n_tot in (24, 42, 57, 63, 126, 135):
        n_h = n_tot // 3
        tq = float(stats.t.ppf(0.975, n_h - 1))
        out.append((n_tot, n_h,
                    1.96 * s / math.sqrt(n_h),          # 正态（实测 sd）
                    tq * s / math.sqrt(n_h),            # t（实测 sd，主口径）
                    tq * math.sqrt(p * (1 - p) / n_h),  # 伯努利尺度敏感性 p=0.625
                    tq * 0.5 / math.sqrt(n_h)))         # 伯努利严格上界 s=0.5
    return out


def n_for_halfwidth(h, s=0.330):
    from scipy import stats
    n = 2
    while float(stats.t.ppf(0.975, n - 1)) * s / math.sqrt(n) > h:
        n += 1
    return n


# ------------------------------------------------- 设计矩阵诊断（§3.3 / §3.6）
def _onehot(vals, levels):
    return np.array([[1.0 if v == lv else 0.0 for lv in levels] for v in vals])


def _orth(D):
    """中心化后的列空间 → 标准正交基（即该因子的对比空间）。"""
    C = D - D.mean(axis=0, keepdims=True)
    Q, R = np.linalg.qr(C)
    r = int(np.linalg.matrix_rank(C, tol=1e-9))
    return Q[:, :r]


def _rank(A):
    return int(np.linalg.matrix_rank(A, tol=1e-9))


def _proj(B, A):
    """A 在 B 列空间上的投影。"""
    return B @ np.linalg.lstsq(B, A, rcond=None)[0]


def _block_r2(A, B):
    """A 的列空间被 B 解释的比例（Frobenius 意义）。"""
    den = float(np.sum(A * A))
    if den == 0:
        return 0.0
    return float(np.sum(_proj(B, A) ** 2) / den)


def _vif(M):
    out = []
    for j in range(M.shape[1]):
        others = np.delete(M, j, axis=1)
        out.append(1.0 / max(1e-12, 1.0 - _block_r2(M[:, [j]], others)))
    return out


def design_diagnostics():
    fam = [c["fam"] for c in ROWS]
    sp = [c["sp"] for c in ROWS]
    nb = [c["n_b"] for c in ROWS]
    st = [c["strength"] for c in ROWS]
    unit = [c["cell"] - 1 for c in ROWS]

    Bf, Bs, Bn, Bt = (_orth(_onehot(fam, ("L1", "L2"))),
                      _orth(_onehot(sp, SPACING_ORDER)),
                      _orth(_onehot(nb, NB_LEVELS)),
                      _orth(_onehot(st, ("strong", "weak"))))
    blocks = {"family": Bf, "spacing": Bs, "n_b": Bn, "strength": Bt}
    M = np.column_stack([np.ones(len(ROWS)), Bf, Bs, Bn, Bt])
    Mn = M / np.linalg.norm(M, axis=0, keepdims=True)

    print("### D1｜主效应设计矩阵\n")
    print(f"- 观测数 24，模型列 = 截距 + family(1) + spacing(3) + n_b(2) + strength(1) = **{M.shape[1]}**")
    print(f"- **秩 = {_rank(M)} / {M.shape[1]}**（满列秩即主效应可估的前提）")
    print(f"- 列标准化后**条件数 = {np.linalg.cond(Mn):.6f}**（1.0 = 完全正交，无数值共线）")
    print(f"- 各列 VIF = {[round(v, 6) for v in _vif(M)]}（全部 1.0 即无共线）\n")
    print("| 列 | 1 | family | spacing1 | spacing2 | spacing3 | n_b1 | n_b2 | strength |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("| VIF | " + " | ".join(f"{v:.4f}" for v in _vif(M)) + " |")

    print("\n### D2｜主效应两两混叠（R²，块对块）\n")
    print("| | family | spacing | n_b | strength |")
    print("|---|---|---|---|---|")
    for a in blocks:
        row = [f"{_block_r2(blocks[a], blocks[b]):.3g}" for b in blocks]
        print(f"| {a} | " + " | ".join(row) + " |")
    print("\n（0 表示两块的列空间正交 ⇒ 该主效应不受对方牵连）")

    # ---------------- 单元层：8 个 (family, spacing) 单元
    u_fam = [CELLS[i][0] for i in range(8)]
    u_sp = [CELLS[i][1] for i in range(8)]
    u_st = [CELL_STRENGTH[CELLS[i]] for i in range(8)]
    uf, us, ut = (_orth(_onehot(u_fam, ("L1", "L2"))),
                  _orth(_onehot(u_sp, SPACING_ORDER)),
                  _orth(_onehot(u_st, ("strong", "weak"))))
    U = np.column_stack([np.ones(8), uf, us, ut])
    Du = _onehot([str(i) for i in range(8)], [str(i) for i in range(8)])
    FS = Du - _proj(np.column_stack([np.ones(8), uf, us]), Du)   # 去掉 1+family+spacing
    FS_basis = np.linalg.qr(FS)[0][:, :3]
    print("\n### D3｜单元层（8 个单元）：strength 落在 family×spacing 空间里\n")
    print(f"- 单元层模型 [截距, family, spacing] 秩 = {_rank(np.column_stack([np.ones(8), uf, us]))} / 8")
    print(f"- 余下 family×spacing 交互空间维数 = **{_rank(FS_basis)}**（= 8 − 5）")
    tvec = ut[:, 0]
    resid = tvec - _proj(FS_basis, tvec[:, None])[:, 0]
    print(f"- strength 列投影到该空间后的残差范数 = {np.linalg.norm(resid):.2e}"
          f" ⇒ **strength 完全落在 family×spacing 交互空间内**")
    print("- 结论：strength 的 1 df **占用了** family×spacing 的 3 df 中的 1 df；"
          "该交互只剩 **2 df** 可估。")
    print("- 因此 strength 主效应**只在「该特定 family×spacing 对比为零」的假设下可解释**，"
          "不能当作与 family/spacing 无关的独立效应。")
    FS_rest = FS_basis - _proj(tvec[:, None], FS_basis)
    FS_rest = np.linalg.qr(FS_rest)[0][:, :2]
    print(f"- 剩余 2 个可估交互方向：{FS_rest.shape[1]} 维（秩 {_rank(FS_rest)}）")

    print("\n### D4｜单元内（每个单元 3 个 n_b）：自由度账本\n")
    Du24 = _onehot([str(u) for u in unit], [str(i) for i in range(8)])
    X0 = np.column_stack([Du24, Bn])
    r0 = _rank(X0)
    print(f"- 单元固定效应(8) + n_b(2) 秩 = {r0}；单元内剩余 df = 24 − 8 − 2 = **{24 - 8 - 2}**")
    famv = np.where(np.array(fam) == "L1", 1.0, -1.0)[:, None]
    nf = Bn * famv
    ns = np.column_stack([Bn * Bs[:, [j]] for j in range(Bs.shape[1])])
    nt = Bn * ut[np.array(unit)][:, :1]
    nfs = np.column_stack([Bn * FS_rest[np.array(unit)][:, [j]] for j in range(2)])
    print(f"- family×n_b：{_rank(nf)} df；spacing×n_b：{_rank(ns)} df；"
          f"strength×n_b：{_rank(nt)} df；(family×spacing 自由 2 维)×n_b：{_rank(nfs)} df")
    allX = np.column_stack([Du24, Bn, nf, ns, nt, nfs])
    print(f"- 全模型秩 = {_rank(allX)} / 24 ⇒ **单元内空间饱和**（残差 df = {24 - _rank(allX)}）")
    print("- 饱和的后果：不得在单元内拟合并报告残差方差；"
          "任何「n_b 交互」的检验都必须预先指定，不能事后从饱和模型里挑。")

    print("\n### D5｜编码 / 参考水平依赖性（用固定合成响应验证）\n")
    y = np.random.default_rng(20261113).normal(size=24)[:, None]
    Xa = M
    Xb = np.column_stack([
        np.ones(24),
        _onehot(fam, ("L1", "L2"))[:, 1:],
        _onehot(sp, SPACING_ORDER)[:, 1:],
        _onehot(nb, NB_LEVELS)[:, 1:],
        _onehot(st, ("strong", "weak"))[:, 1:]])
    pa = Xa @ np.linalg.lstsq(Xa, y, rcond=None)[0]
    pb = Xb @ np.linalg.lstsq(Xb, y, rcond=None)[0]
    print(f"- 求和编码 vs 处理编码：拟合值最大差 = {np.max(np.abs(pa - pb)):.2e}"
          f"（列空间相同 ⇒ 预测不变）")
    print(f"- 两种编码下 family 的边际均值差 = "
          f"{float(y[np.array(fam) == 'L1'].mean() - y[np.array(fam) == 'L2'].mean()):+.6f}（唯一值，与参考水平无关）")
    print("- 因 n_b 在每个单元内恰好各出现一次、family/spacing/strength 在单元间平衡，"
          "**主效应估计不依赖参考水平**；不平衡的是 24 个配置的响应值，不是设计。")

    print("\n### D6｜交互项：完全混叠 vs 仅「没自由度」\n")
    print("判据：把交互块投影掉主效应模型 M 后看残差秩。残差秩 = 0 ⇒ **完全混叠**；"
          "残差秩 > 0 但全模型放不下 ⇒ 仅是**自由度不足**。\n")
    print("| 交互块 | 名义 df | **单独**加入时的残差秩 | 判定 |")
    print("|---|---|---|---|")
    famc = np.where(np.array(fam) == "L1", 1.0, -1.0)[:, None]
    stc = np.where(np.array(st) == "strong", 1.0, -1.0)[:, None]
    blocks_i = {
        "strength×family": Bt * famc,
        "strength×spacing": np.column_stack([Bt * Bs[:, [j]] for j in range(Bs.shape[1])]),
        "family×spacing": FS_basis[np.array(unit)],
        "n_b×family": Bn * famc,
        "n_b×spacing": np.column_stack([Bn * Bs[:, [j]] for j in range(Bs.shape[1])]),
        "n_b×strength": Bn * stc,
        "n_b×family×spacing": np.column_stack(
            [Bn * FS_basis[np.array(unit)][:, [j]] for j in range(3)]),
    }
    for name, Bl in blocks_i.items():
        R = Bl - _proj(M, Bl)
        rr = _rank(R)
        nom = Bl.shape[1]
        verdict = ("**完全混叠**（0 df）" if rr == 0
                   else ("可估（主效应后仍满秩）" if rr == nom else f"部分混叠，实得 {rr}/{nom} df"))
        print(f"| {name} | {nom} | {rr} | {verdict} |")
    print("\n- 含 strength 的交互全部落进 family×spacing 的 3 维空间，"
          "与 strength 主效应争夺同一批自由度 ⇒ **0 df**，不是「样本不够」。")
    print("- 含 n_b 的交互落在单元内空间，与主效应正交 ⇒ 单独加入都可估；"
          "但**联合**加入时 2+6+2+6 = 16 > 单元内剩余 14 df，"
          "`(family×spacing)×n_b` 实际只有 **4 df**（F×S 仅 2 维自由）。")
    print("- 因此 n_b 侧的结论必须**预先指定到某一个交互**，不能事后从饱和模型里挑最大的那个。")



def fmt(v, nd=3):
    if v is None:
        return "—"
    if v != 0 and (abs(v) >= 1e4 or abs(v) < 3e-3):
        m, e = f"{v:.0e}".split("e")
        return f"{m}e{int(e):+d}"
    return f"{v:.{nd}g}"


def main():
    print("## 表 A｜配置总数 ↔ 留出集 95%CI 半宽（三尺度并列）\n")
    print("| 总配置 | 留出 | 正态近似（s=0.330） | **t 分布（主口径）** | "
          "伯努利尺度 p=0.625（s=0.484） | 伯努利严格上界（s=0.5） |")
    print("|---|---|---|---|---|---|")
    for r in ci_table():
        print(f"| {r[0]} | {r[1]} | ±{r[2]:.3f} | **±{r[3]:.3f}** | ±{r[4]:.3f} | ±{r[5]:.3f} |")
    print(f"\n留出集要 ±0.10（t + 实测 s 口径）需 N_hold = {n_for_halfwidth(0.10)}"
          f" ⇒ N_total ≈ {3*n_for_halfwidth(0.10)}\n")

    print("## 表 B｜24 配置\n")
    print("| id | family | spacing | n_b | strength | R_b/R_L | f1 (Hz) | f2 (Hz) | "
          "f_vis (Hz) | f_b (Hz) | log10(f1/f_vis) | log10(f2/f_vis) | log10(f_vis/f_min) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in ROWS:
        print(f"| {c['cid']} | {c['fam']} | {c['sp']} | {c['n_b']:.2f} | {c['strength']} | "
              f"{c['Rb']/c['R_L']:.2f} | {fmt(c['f1'])} | {fmt(c['f2'])} | "
              f"{fmt(c['fvis'])} | {fmt(c['fb'])} | {c['d_f1']:.2f} | {fmt(c['d_f2'],2)} | "
              f"{c['d_win']:+.2f} |")

    print("\n## 表 C｜三组平衡核验\n")
    print("| 组 | n | L1/L2 | separated | overlap | out_of_window | near_degenerate | "
          "n_b 0.60/0.75/0.90 | strong/weak |")
    print("|---|---|---|---|---|---|---|---|---|")
    for g in GROUPS:
        sub = [c for c in ROWS if c["group"] == g]
        n1 = sum(1 for c in sub if c["fam"] == "L1")
        spc = [sum(1 for c in sub if c["sp"] == s) for s in SPACING_ORDER]
        nbc = [sum(1 for c in sub if c["n_b"] == k) for k in NB_LEVELS]
        stc = [sum(1 for c in sub if c["strength"] == k) for k in STRENGTH]
        print(f"| {GROUP_ZH[g]} | {len(sub)} | {n1}/{len(sub)-n1} | "
              + " | ".join(str(x) for x in spc) + " | "
              + "/".join(str(x) for x in nbc) + " | "
              + f"{stc[0]}/{stc[1]} |")

    print("\n## 表 D｜留出集独立性\n")
    used = lambda c: [c["p"][k] for k in USED[c["fam"]]]
    hold = [c for c in ROWS if c["group"] == "hold"]
    others = [c for c in ROWS if c["group"] != "hold"]
    ident = 0
    intra = []
    for h in hold:
        for o in others:
            if h["fam"] != o["fam"]:
                continue
            d = [abs(math.log(a / b)) for a, b in zip(used(h), used(o))]
            ident += int(max(d) == 0.0)
            if h["cell"] == o["cell"]:
                intra.append(max(d))
    allpairs = 0
    for i in range(len(ROWS)):
        for j in range(i + 1, len(ROWS)):
            a, b = ROWS[i], ROWS[j]
            if a["fam"] != b["fam"]:
                continue
            d = [abs(math.log(x / y)) for x, y in zip(used(a), used(b))]
            allpairs += int(max(d) == 0.0)
    print(f"- 同族、非留出 vs 留出、真值向量完全相同的对数：**{ident}**")
    print(f"- 全部同族配置对中真值向量完全相同的对数：**{allpairs}**")
    print(f"- 同一单元内（留出 vs 开发/校准）最大单参数 |ln(比值)|："
          f"最小 **{min(intra):.3f}**、最大 **{max(intra):.3f}**")
    print(f"- 同单元内唯一存在差异的参数：`Q_b`、`n_b`（其余 {len(USED['L2'])-2} 个共享）")

    print("\n## 表 E｜spacing 的判定量（可复算）\n")
    print("| spacing | f_vis (Hz) | f_vis/f_min | f1/f_vis | f2/f_vis | 预期标签 |")
    print("|---|---|---|---|---|---|")
    for sp in SPACING_ORDER:
        c = next(x for x in ROWS if x["sp"] == sp and x["fam"] == "L2")
        print(f"| {sp} | {fmt(c['fvis'])} | {c['fvis']/F_LO:.4g} | "
              f"{c['d_f1']:.2f} dec | {c['d_f2']:.2f} dec | "
              f"{ {'separated':'identifiable_by_design','overlap':'weakly_identifiable','out_of_window':'non_identifiable_by_window','near_degenerate':'non_identifiable_by_degeneracy'}[sp] } |")

    print("\n## 附录 D｜设计矩阵诊断（回答 §3.3 的「主效应可估」是否成立）\n")
    design_diagnostics()


if __name__ == "__main__":
    main()
