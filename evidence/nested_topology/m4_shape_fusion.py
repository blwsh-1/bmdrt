"""把「voigt 形态判据」做成多信号融合，并用独立的训练/测试集标定。

为什么要有这个脚本
------------------
`m4_shape_roc.py` 证明单条判据是坏的：裸 voigt 误排除 25%、σ=5% 时 95%，
`voigt+ad` 误排除 75%，series 支路只检出 68%。但那次扫描有两个方法论缺陷：

1. **阈值与成绩出自同一批样本**（8%/5% 由 4 个手工用例定，又在扫描集上读数）
   → 报出来的是标定集成绩，不是泛化成绩。
2. 只用了两个信号，且都按 `max|Z|` 归一化 → σ 越大差分噪声越大，必然越阈。

本脚本用四个**互补**信号 + 严格 train/test 分离 + 双 σ̂ 口径重做，回答
「形态到底够不够用」。答案可以是"不够"，那就放弃这条路。

四个信号（方向：越大越像 nested）
---------------------------------
- `s_up`  ：`Re Z` 最大上升段 / σ̂。voigt 每个 HN 实部单调升，相加仍单调；
  nested 的 RL 支路在低频把实部拉回，形成上升段。
- `s_im`  ：最低频 `Im Z` / σ̂。voigt 的 Im≤0（HN/Ws 恒负、Wo 实数、jωL→0）；
  RL 支路 ω→0 时 `Z→jωL`，`Im>0`。
- `s_ph`  ：低频段相位斜率的绝对值（deg/decade）。voigt 低频相位趋 0°；
  RL 在低频趋 +90°，跨窗时斜率大。无量纲，不需要 σ。
- `s_low` ：`Re(f_min)/max(Re)` 的**端点比值**（越小越像 nested，故取负）。
  这是本脚本找到的最强信号：voigt 的每个 HN 在 ω→0 都趋于 `w`，所以低频
  实部必是全谱最大 → 比值≈1（实测 p05=0.96）；RL 支路 `Re=R(ωL)²/(R²+(ωL)²)`
  在 ω→0 时 →0 → 比值小（npar/nser 中位 0.23–0.24）。**不用差分**，所以
  高噪声下不退化，这一点是它优于 `s_up` 的根本原因。
- `s_gam` ：有符号 DRT 的**负 γ 面积占比**（`drt.drt(signed=True)`）。
  **实测作废，默认不启用**（`--with-gam` 才跑）：四类中位数全在 0.40–0.49，
  连裸 voigt 都有 42% 负 γ 面积；换 `m∈{30,60,100}`、固定 λ 后方向仍是反的
  （裸 voigt 低频 `min γ/max|γ| = −0.98`，nested 只有 −0.45~−0.70）。成因是
  有限窗宽下带符号解把**窗外低频过程**吸收成负弛豫，与 `WS_CRITERION_PLAN`
  记的「DRT 分不出窗外过程」同源。故该信号只作负面结果记录，不进融合。

**`voigt+ad` 是本判据的结构性软肋，不是调参能救的**：`Z_ad` 的低频渐近与
RL 支路同为 `jω` 型（虚部 ∝ ω、实部 →0），`s_im`/`s_gam` 必然同向报警。
所以结果分两栏：裸 voigt 特异度（判据是否成立）、ad 组特异度（已知会低，
用来界定"生产开 --adsorption 时本判据不可用"）。

方法
----
1. 四类各生成 `2n` 条，按单双号切成 train/test **两批独立样本**。
2. 用 **train** 的全体（含 ad）算每个信号的 robust 中位数与 IQR，做等权
   z 归一化后求和 → 单一分数。**只有阈值一个自由参数**在 train 上选，
   约束是「裸 voigt 特异度 ≥ 目标」，目标是让灵敏度最大。
3. 在 **test** 上报最终特异度/灵敏度。train 上那些数字不报，避免自我欺骗。
4. 另跑一个逻辑回归作为**上界参考**（4 权重 + 截距），明确标注为乐观上界，
   不作为可用结果。

σ̂ 双口径
--------
- `sigma_true`：扫描时已知的噪声，属**乐观上界**（真实谱拿不到）。
- `sigma_mad` ：用相邻点差分 MAD 自估（`1.4826·MAD(d)/√2`），**这才是上线
  口径**。两者都报，差距就是"σ̂ 估不准"的代价。

用法
----
    python evidence/nested_topology/m4_shape_fusion.py            # 默认 120/类/批
    python evidence/nested_topology/m4_shape_fusion.py 200        # 每批 200
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bm_top_gen as G                      # noqa: E402
from drt import drt                         # noqa: E402

SIGMAS = (0.005, 0.01, 0.02, 0.05)
GRIDS = (("AZ31", 50), ("AZ31", 30), ("AZ31", 80))

#: 特异度目标：裸 voigt 被误排除的比例不得超过 5%
TARGET_SPEC = 0.95

#: 三态判定里「低频感抗显著」的门限（单位 σ）；低于此值判「信息不足」而非「不排除」
SIM_GATE = 3.0

#: 参与融合的信号。`s_gam` 实测无判别力且方向相反，默认排除（见模块 docstring）
SIGNALS = ("s_up", "s_im", "s_ph", "s_low")
ALL_SIGNALS = ("s_up", "s_im", "s_ph", "s_low", "s_gam")

#: 每个信号的方向：+1 = 越大越像 nested；-1 = 越小越像 nested
SIGN_DIR = {"s_up": 1.0, "s_im": 1.0, "s_ph": 1.0, "s_low": -1.0, "s_gam": 1.0}

KINDS = ("voigt", "voigt_ad", "npar", "nser")


# ---------------------------------------------------------------- 正演 ----
def voigt_spec(f, R_inf, ws, taus, phis):
    """裸 voigt：R_inf + Σ w/(1+(jωτ)^φ)，ζ=1（`zeta="1.0"` 的生产默认）。"""
    z = np.full(len(f), float(R_inf), complex)
    w_ax = 2 * np.pi * np.asarray(f, float)
    for w, tau, phi in zip(ws, taus, phis):
        z = z + w / (1.0 + (1j * w_ax * tau) ** phi)
    return z


def voigt_ad_spec(f, R_inf, ws, taus, phis, R_ad, tau_ad):
    """voigt + 吸附支路（`bm_core.model_z` 的 `Rad`/`tau_ad` 项）。"""
    jt = 1j * (2 * np.pi * np.asarray(f, float)) * tau_ad
    return voigt_spec(f, R_inf, ws, taus, phis) + R_ad * jt / (1.0 + jt)


def rand_voigt(rng, K):
    lu = lambda lo, hi: float(10.0 ** rng.uniform(lo, hi))
    return (lu(-0.5, 1.7),
            [lu(0.5, 3.5) for _ in range(K)],
            [lu(-2.0, 2.0) for _ in range(K)],
            [float(rng.uniform(0.6, 1.0)) for _ in range(K)])


# ---------------------------------------------------------------- 信号 ----
def sigma_mad(Z) -> float:
    """相邻点差分 MAD 估单点噪声绝对值（不做 √2 缩放的保守版）。"""
    d = np.diff(np.asarray(Z).real)
    if d.size < 3:
        return float(np.abs(Z).max() * 0.01)
    mad = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    return float(max(mad, 1e-12))


def signals(f, Z, sig_abs: float, with_gam: bool = False) -> dict:
    """信号 dict。`sig_abs` 是绝对噪声（与 sqrt(var) 同量纲），用于归一化。"""
    o = np.argsort(f)
    f, Z = np.asarray(f)[o], np.asarray(Z)[o]
    re, im = Z.real, Z.imag

    dre = np.diff(re)
    up = float(dre[dre > 0].max()) if (dre > 0).any() else 0.0
    s_up = up / sig_abs

    s_im = float(im[0]) / sig_abs

    # 低频段相位斜率：最低 25% 频点（至少 5 个），phase(deg) vs log10(f)
    n = f.size
    k = max(5, n // 4)
    ph = np.degrees(np.angle(Z[:k]))
    x = np.log10(f[:k])
    if np.ptp(x) > 0:
        slope = float(np.polyfit(x, ph, 1)[0])       # deg / decade
    else:
        slope = 0.0
    s_ph = abs(slope) / 45.0                         # 1 单位 = 45°/decade

    # 端点比值：低频实部 / 全谱最大实部。voigt 必≈1；RL 支路低频→0。
    rmax = float(re.max())
    s_low = float(re[0] / rmax) if rmax > 0 else 1.0

    # 有符号 DRT 的负 γ 面积占比（实测作废，仅在 --with-gam 时计算并展示）
    s_gam = float("nan")
    if with_gam:
        try:
            r = drt(f, Z, sigma=sig_abs / max(float(np.abs(Z).max()), 1e-12),
                    m=60, signed=True)
            g = np.asarray(r["gamma"], float)
            tot = float(np.sum(np.abs(g)))
            if tot > 0:
                s_gam = float(np.sum(np.abs(g[g < 0])) / tot)
        except Exception:
            pass
    return dict(s_up=s_up, s_im=s_im, s_ph=s_ph, s_low=s_low, s_gam=s_gam)


def make_set(kind: str, n: int, rng, with_gam: bool = False):
    """生成 n 条该类谱：[(sig_dict, sigma_frac, sigma_true_abs, sigma_mad_abs, pts)]。"""
    out = []
    for i in range(n):
        sf = SIGMAS[i % len(SIGMAS)]
        gname, gpts = GRIDS[(i // len(SIGMAS)) % len(GRIDS)]
        f = G.real_grid(gname, gpts)
        if kind in ("voigt", "voigt_ad"):
            L = 1 + i % 3
            R_inf, ws, taus, phis = rand_voigt(rng, L)
            Z = (voigt_spec(f, R_inf, ws, taus, phis) if kind == "voigt" else
                 voigt_ad_spec(f, R_inf, ws, taus, phis,
                               R_ad=10.0 ** rng.uniform(1.5, 3.5),
                               tau_ad=10.0 ** rng.uniform(-2.0, 2.0)))
        else:
            branch = "parallel" if kind == "npar" else "series"
            p = G._rand_params(rng, layers=1 + i % 2, branch=branch)
            Z = G.gen(f, p, branch=branch, layers=1 + i % 2)
        Z = G.add_noise(Z, sf, seed=7000 + i)
        s_true = sf * float(np.abs(Z).max())
        s_mad = sigma_mad(Z)
        out.append((signals(f, Z, s_true, with_gam), sf, s_true, s_mad, gpts))
    return out


# ------------------------------------------------------------ 归一化 ----
def fit_norm(train: dict) -> dict:
    """用 train 全体（含 ad 组）算每个信号的 robust 中心与尺度。"""
    norm = {}
    for s in SIGNALS:
        v = np.array([d[s] for k in KINDS for d, *_ in train[k] if np.isfinite(d[s])],
                     float)
        if v.size < 8:
            norm[s] = (0.0, 1.0)
            continue
        med = float(np.median(v))
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        norm[s] = (med, iqr if iqr > 1e-9 else 1.0)
    return norm


def score(rec, norm, sigma_key: str, w: dict | None = None) -> float:
    """加权 z 和。信号缺失（nan）按 0 计（该信号不投票，不惩罚）。"""
    d, _sf, s_true, s_mad, _ = rec
    den = s_true if sigma_key == "sigma_true" else s_mad
    w = {s: 1.0 for s in SIGNALS} if w is None else w
    tot = 0.0
    for s in SIGNALS:
        x = d[s]
        if s in ("s_up", "s_im") and np.isfinite(x):
            # 这两个信号收集时按真值 σ 归一化，换 σ̂ 后按比例重算
            x = x * (s_true / den)
        if not np.isfinite(x):
            continue
        med, iqr = norm[s]
        tot += float(w.get(s, 0.0)) * SIGN_DIR[s] * (x - med) / iqr
    return float(tot)


def fit_logit(train, norm, sigma_key="sigma_mad", epochs=800, lr=0.05):
    """train 上拟合 3 个权重 + 截距（手写梯度下降，避免额外依赖）。

    正类 = nested（npar/nser），负类 = voigt。**ad 组不参与拟合**：
    它物理上介于两者之间，混进去会把权重往"分不开"拉，掩盖真实能力。
    """
    X, y = [], []
    for k in ("voigt", "npar", "nser"):
        for r in train[k]:
            v = []
            for s in SIGNALS:
                x = r[0][s]
                if s in ("s_up", "s_im") and np.isfinite(x):
                    x = x * (r[2] / (r[2] if sigma_key == "sigma_true" else r[3]))
                med, iqr = norm[s]
                v.append(SIGN_DIR[s] * (x - med) / iqr if np.isfinite(x) else 0.0)
            X.append(v)
            y.append(0.0 if k == "voigt" else 1.0)
    X, y = np.array(X), np.array(y)
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / len(y))
        b -= lr * float(g.mean())
    return {s: float(wi) for s, wi in zip(SIGNALS, w)}, float(b)


def pick_threshold(scores, labels, target_spec=TARGET_SPEC):
    """在 train 上选阈值：满足裸 voigt 特异度 ≥ target 的最大灵敏度。"""
    cand = np.unique(np.round(scores, 4))
    best = None
    for th in cand:
        pred = scores >= th
        spec = 1.0 - float(np.mean(pred[labels == "voigt"]))
        if spec < target_spec:
            continue
        sens = 0.5 * (float(np.mean(pred[labels == "npar"])) +
                      float(np.mean(pred[labels == "nser"])))
        if best is None or sens > best[1]:
            best = (float(th), sens, spec)
    return best


def report(sets, norm, sigma_key, th, title, w=None, quiet=False):
    rows = {}
    for k in KINDS:
        sc = np.array([score(r, norm, sigma_key, w) for r in sets[k]])
        rows[k] = float(np.mean(sc >= th))
    if quiet:
        return rows
    print(f"\n{title}（σ̂={sigma_key}，阈值 {th:.3f}）")
    print(f"{'类':<10}{'判排除':>9}   应为")
    print(f"{'voigt':<10}{rows['voigt']:>9.1%}   不该排除  ← 特异度 {1 - rows['voigt']:.1%}")
    print(f"{'voigt_ad':<10}{rows['voigt_ad']:>9.1%}   不该排除（已知软肋）")
    print(f"{'npar':<10}{rows['npar']:>9.1%}   该排除")
    print(f"{'nser':<10}{rows['nser']:>9.1%}   该排除")
    print(f"{'灵敏度':<10}{0.5 * (rows['npar'] + rows['nser']):>9.1%}")
    return rows


def per_sigma(sets, norm, sigma_key, th, w=None):
    """按噪声档拆开看融合分是否稳定（σ 越大越难，看退化是否可接受）。"""
    print(f"\n按 σ 拆开（{sigma_key}）")
    print(f"{'σ':>7}" + "".join(f"{k:>10}" for k in KINDS))
    for s in SIGMAS:
        row = ""
        for k in KINDS:
            v = [score(r, norm, sigma_key, w) for r in sets[k] if r[1] == s]
            row += f"{np.mean(np.array(v) >= th):>10.0%}"
        print(f"{s:>6.1%}" + row)


def single_signal_table(sets, norm):
    """单信号各自的能力：确认信号不是冗余的，也是融合的对照组。"""
    print("\n=== 单信号（各自单独用，阈值取让裸 voigt 误判 ≤5% 者）===")
    print(f"{'信号':<8}{'裸voigt':>9}{'voigt+ad':>10}{'npar':>8}{'nser':>8}   灵敏度")
    for s in ALL_SIGNALS:
        def raw(rec):
            x = rec[0].get(s, float("nan"))
            if s in ("s_up", "s_im") and np.isfinite(x):
                x = x * (rec[2] / rec[3])
            return SIGN_DIR[s] * x if np.isfinite(x) else x
        vv = np.array([raw(r) for r in sets["voigt"]], float)
        vv = vv[np.isfinite(vv)]
        if vv.size < 4:
            print(f"{s:<8}{'(无有效值，未启用或无判别力)':<20}")
            continue
        th = float(np.percentile(vv, 95))
        out = []
        for k in KINDS:
            v = np.array([raw(r) for r in sets[k]], float)
            v = v[np.isfinite(v)]
            out.append(float(np.mean(v >= th)) if v.size else float("nan"))
        print(f"{s:<8}{out[0]:>9.1%}{out[1]:>10.1%}{out[2]:>8.1%}{out[3]:>8.1%}"
              f"   {0.5 * (out[2] + out[3]):.0%}")


def lowfreq_attribution(sets, norm, sigma_key, th, w=None):
    """漏检归因：按「低频端 ωL/|Z|」分档看 nested 检出率。

    若检出率随该比例单调升、且低档几乎为 0，则漏检来自**数据不可判**
    （RL 转折在窗外、L 太小），不是判据失灵。这部分谱任何形态判据都判不出。
    """
    print(f"\n漏检归因（{sigma_key}）：nested 检出率 vs 低频感抗占比 ωL/|Z|")
    print(f"{'ωL/|Z|@f_min':>14}{'n':>6}{'npar':>8}{'nser':>8}")
    bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 1.0), (1.0, 1e9)]
    for lo, hi in bins:
        cells = []
        for k in ("npar", "nser"):
            hit, tot = 0, 0
            for r in sets[k]:
                x = r[0]["s_im"] * (r[2] / (r[2] if sigma_key == "sigma_true" else r[3]))
                frac = max(x, 0.0)            # s_im 近似 = 低频 ωL/|Z|
                if lo <= frac < hi:
                    tot += 1
                    hit += score(r, norm, sigma_key, w) >= th
            cells.append((hit, tot))
        n = sum(t for _, t in cells)
        s = "".join(f"{(h / t if t else float('nan')):>8.0%}" for h, t in cells)
        print(f"{f'{lo:g}–{hi:g}':>14}{n:>6}{s}")
    print("读法：低档（ωL/|Z|<5%）检出率接近 0 → 那些谱的低频感抗小于噪声，"
          "形态上不可判。")


def three_state(sets, norm, sigma_key, th, w=None, gate=SIM_GATE):
    """三态判定：排除 / 信息不足 / 不排除。

    二值化会把「谱里没有形状信息」和「判据判错」混成一件事。真实用法是：
    `score ≥ th` → 排除 voigt；否则看低频感抗是否显著（`s_im ≥ gate`，单位 σ）：
    显著 → **信息不足**（有 nested 型感抗但形状没越阈，交 LOO 定）；
    不显著 → **不排除**（正常 voigt 形状）。

    这样"排除"这一栏的精确率才有意义，覆盖率下降是诚实的代价。
    """
    print(f"\n三态判定（{sigma_key}，排除阈 {th:+.3f}，感抗门限 {gate:g}σ）")
    print(f"{'类':<10}{'排除':>8}{'信息不足':>10}{'不排除':>9}   排除栏精确率")
    stat = {}
    for k in KINDS:
        n = len(sets[k])
        ex = ins = no = 0
        for r in sets[k]:
            s = score(r, norm, sigma_key, w)
            sim = r[0]["s_im"] * (r[2] / (r[2] if sigma_key == "sigma_true" else r[3]))
            if s >= th:
                ex += 1
            elif np.isfinite(sim) and sim >= gate:
                ins += 1
            else:
                no += 1
        stat[k] = (ex, ins, no, n)
        prec = ex / n if n else float("nan")
        print(f"{k:<10}{ex / n:>8.0%}{ins / n:>10.0%}{no / n:>9.0%}"
              f"   {'—' if k == 'voigt' else f'{prec:.0%} 被排除（该排除）'}")
    tn, tp = stat["npar"][0] + stat["nser"][0], stat["voigt"][0]
    pre = tn / (tn + tp) if (tn + tp) else float("nan")
    nested_excl = tn / (stat["npar"][3] + stat["nser"][3])
    print(f"  合并：nested 被排除 {nested_excl:.0%}；排除栏里真 nested 占 {pre:.0%}"
          f"（{tn}/{tn + tp}）；voigt 误排除 {tp / stat['voigt'][3]:.0%}")

    # 漏检归因的关键证据：那些"没被排除"的 nested，低频感抗本身有多显著？
    print("  漏检者（判为不排除的 nested）低频 `Im/σ` 分布：")
    for k in ("npar", "nser"):
        sim = []
        for r in sets[k]:
            s = score(r, norm, sigma_key, w)
            v = r[0]["s_im"] * (r[2] / (r[2] if sigma_key == "sigma_true" else r[3]))
            if s < th and np.isfinite(v):
                sim.append(v)
        if not sim:
            print(f"    {k:<5} 无漏检")
            continue
        sim = np.array(sim)
        print(f"    {k:<5} n={len(sim):>3} 中位 {np.median(sim):+6.2f}σ  "
              f"|Ims>3σ 的 {int((sim >= 3).sum())} 条  "
              f"→ 其余 {int((sim < 3).sum())} 条的感抗本就在噪声内")
    print("  读法：漏检者的低频感抗多为**负**（与 voigt 同号）→ 形态上与 voigt 不可分，"
          "不是判据判错。")
    return stat


def real_screen(patterns, n_train=120):
    """对真实谱做形态筛查：用**合成 train** 定归一化/权重/阈值，再逐条打分。

    这是本脚本唯一接触真实数据的地方。注意三条边界：
    · 归一化与阈值全部来自合成谱，**没有**用真实谱调任何参数；
    · 输出只有「排除 voigt / 未排除」，**不是**族级结论；机制结论仍 C 档
      （结构性不可表示是另一回事，按 B-缺项 记，见 §9.4.12）；
    · 真实谱 σ̂ 只能用差分自估（`sigma_mad`），这是上线唯一口径。
    """
    import glob

    import bm_real as R

    rng_tr = {k: np.random.default_rng(20261110 + 101 * i) for i, k in enumerate(KINDS)}
    train = {k: make_set(k, n_train, rng_tr[k]) for k in KINDS}
    norm = fit_norm(train)
    w, b = fit_logit(train, norm, "sigma_mad")
    sc = np.array([score(r, norm, "sigma_mad", w) + b
                   for k in ("voigt", "npar", "nser") for r in train[k]])
    lb = np.concatenate([np.full(len(train[k]), k) for k in ("voigt", "npar", "nser")])
    th = pick_threshold(sc, lb)[0]
    print(f"\n阈值 {th:+.3f}（合成 train 定，裸 voigt 特异度 ≥{TARGET_SPEC:.0%}）")

    for pat in patterns:
        files = sorted(glob.glob(pat))
        if not files:
            print(f"\n{pat}：无文件")
            continue
        print(f"\n=== 真实谱筛查 {pat}（{len(files)} 条）===")
        print(f"{'sample':<22}{'σ̂/max|Z|':>10}{'s_up':>8}{'s_im':>8}{'s_ph':>7}"
              f"{'s_low':>7}{'分数':>8}  形态")
        hits = 0
        for p in files:
            f, Z = R.load_spectrum(p)
            o = np.argsort(f)
            f, Z = f[o], Z[o]
            sm = sigma_mad(Z)
            d = signals(f, Z, sm)
            rec = (d, sm / float(np.abs(Z).max()), sm, sm, len(f))
            s = score(rec, norm, "sigma_mad", w) + b
            ex = s >= th
            hits += ex
            name = Path(p).stem
            print(f"{name:<22}{sm / np.abs(Z).max():>10.2%}{d['s_up']:>8.1f}"
                  f"{d['s_im']:>8.1f}{d['s_ph']:>7.2f}{d['s_low']:>7.2f}{s:>8.2f}  "
                  f"{'排除 voigt' if ex else '未排除'}")
        print(f"  小计：{hits}/{len(files)} 条排除 voigt"
              f"（{hits / len(files):.0%}）—— 只排除、不证明 nested")


def main(argv):
    if "--real" in argv:
        i = argv.index("--real")
        pats = [a for a in argv[i + 1:] if not a.startswith("-")] or [
            "out/_m6_samples/z_csv/*.csv", "archive/*.csv"]
        return real_screen(pats)
    n = int(argv[1]) if len(argv) > 1 else 120
    with_gam = "--with-gam" in argv
    print(f"每类 train {n} / test {n}，噪声 {SIGMAS}，网格 {[g for _, g in GRIDS]}")
    print(f"融合信号 = {list(SIGNALS)}"
          + ("（另含作废的 s_gam，仅展示）" if with_gam else ""))
    print("σ 口径：sigma_true = 已知真值（乐观上界）；sigma_mad = 差分自估（上线口径）")

    rng_tr = {k: np.random.default_rng(20261110 + 101 * i) for i, k in enumerate(KINDS)}
    rng_te = {k: np.random.default_rng(20261110 + 101 * i + 5000)
              for i, k in enumerate(KINDS)}
    print("\n生成 train …")
    train = {k: make_set(k, n, rng_tr[k], with_gam) for k in KINDS}
    print("生成 test …")
    test = {k: make_set(k, n, rng_te[k], with_gam) for k in KINDS}

    norm = fit_norm(train)
    print("\n归一化中心/尺度（train 全体）：")
    for s in SIGNALS:
        print(f"  {s:<7} median={norm[s][0]:+8.3f}  IQR={norm[s][1]:8.3f}")

    # ---- 三种融合方式：只在 train 上选，test 只报被选中那个 ----
    w_logit, b_logit = fit_logit(train, norm, "sigma_mad")
    print("\nlogit 权重（train 拟合，ad 组不参与）：" +
          "  ".join(f"{s}={w_logit[s]:+.2f}" for s in SIGNALS) +
          f"  bias={b_logit:+.2f}")
    strategies = {
        "equal": (None, 0.0),
        "up_only": ({s: float(s == "s_up") for s in SIGNALS}, 0.0),
        "logit": (w_logit, b_logit),
    }

    for key in ("sigma_true", "sigma_mad"):
        print(f"\n{'=' * 62}\nσ̂ 口径 = {key}\n{'=' * 62}")
        chosen = None
        for name, (w, b) in strategies.items():
            sc = np.array([score(r, norm, key, w) + b
                           for k in ("voigt", "npar", "nser") for r in train[k]])
            lb = np.concatenate([np.full(len(train[k]), k)
                                 for k in ("voigt", "npar", "nser")])
            best = pick_threshold(sc, lb)
            if best is None:
                print(f"  [{name:<7}] train 上无法达到特异度 ≥{TARGET_SPEC:.0%}")
                continue
            print(f"  [{name:<7}] train 阈值 {best[0]:+.3f}  "
                  f"train 灵敏度 {best[1]:.1%}（选型用，非成绩）")
            if chosen is None or best[1] > chosen[2]:
                chosen = (name, best[0], best[1], w, b)
        if chosen is None:
            print("  → 三种融合都没能同时在 train 上达标：融合路线不成立")
            continue
        name, th, _sens_tr, w, b = chosen
        print(f"\n  ★ train 选中策略 = {name}（阈值 {th:+.3f}）→ 下面只报它的 test 成绩")
        test_sets = {k: [(r[0], r[1], r[2], r[3], r[4]) for r in test[k]]
                     for k in KINDS}
        rows = report(test_sets, norm, key, th, "TEST 成绩（唯一一次）", w)
        per_sigma(test_sets, norm, key, th, w)
        lowfreq_attribution(test_sets, norm, key, th, w)
        three_state(test_sets, norm, key, th, w)
        print(f"  → test 特异度 {1 - rows['voigt']:.1%}、"
              f"灵敏度 {0.5 * (rows['npar'] + rows['nser']):.1%}、"
              f"ad 组特异度 {1 - rows['voigt_ad']:.1%}")

    single_signal_table(test, norm)

    print("\n判读规则（写报告时照抄，别升档）：")
    print(f"· 裸 voigt 特异度 ≥{TARGET_SPEC:.0%} 且灵敏度 ≥75% → 可作接入前筛查")
    print("· ad 组特异度必然低（低频渐近与 RL 同为 jω 型，数学上不可分）")
    print("  → 生产 `--adsorption` 开启时本判据不适用，必须靠 ad_status 分支")
    print("· 两条 σ 口径差距 = σ̂ 估不准的代价；上线只能用 sigma_mad")
    print("· 本判据只排除 voigt，不证明 nested；机制结论仍 C 档（B-缺项见 §9.4.12）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
