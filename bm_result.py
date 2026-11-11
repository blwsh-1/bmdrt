"""bm_result.py —— 一次分析结果的**唯一来源**（AnalysisResult）与导出包。

为什么单独加这一层
------------------
以前 `plot_real()` 直接读内存里的 `res`/`pres` 画图，`save_out()` 另写一份
JSON，事后写 CSV 又要再拼一次；三处各算一遍残差，只要有一处口径不同
（σ 用后验还是用 2%、区间是参数传播还是后验预测），图上曲线与表里数字
就会对不上，而且**不会报错**。

这里把「算一次」固化：`AnalysisResult` 只从 `real_*.json` + `*_samples.npz`
重建全部派生量（拟合曲线、两个区间的分位、残差、标准化残差、DRT），
交互面板（`bm_dash.py`）、出版图（`bm_figure.py`）、CSV 包（`bundle()`）
三条出口都只读这一份。

三条口径承诺（对应设计文档 §5.3）
--------------------------------
1. **区间分两类，字段名不同**，绝不混称：
   · `band`  = 参数后验传播得到的**响应区间**（逐样本算 Z(f) 再取分位）；
   · `pred`  = **后验预测区间**（响应 ⊕ 该样本自己的噪声 σ）。
2. **标准化残差用拟合自己的 σ**（后验 σ 的中位数），不套 2% 观测噪声。
3. **Re/Im 分别取分位**，得到的是复平面上的矩形包络，不是联合可信域；
   `payload()["band_note"]` 明写这一点。

注意：区间用「逐样本算曲线再取分位」，不是「先平均参数再算曲线」——
后者在非线性模型上会系统性偏窄（`bm_core.drt_band` 的同一理由）。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import bm_core as B

try:                                    # Rwp 定义只留一处（bm_ref_rwp）
    import bm_ref_rwp as REF
except Exception:                       # pragma: no cover
    REF = None

# 分位与抽样上限：**固定**，否则同一条谱两次导出给出不同区间
BAND_Q = (5.0, 95.0)
MAX_DRAWS = 400
PRED_SEED = 0
SPEC_VERSION = "1.0"

# §14 三判据（预注册值，不在这里放宽）
THRESHOLDS = dict(acf=0.50, rhat=1.05, div=0)

_STATUS_ALIAS = {"ok": "converged"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_one(directory: Path, patterns) -> Path | None:
    for pat in patterns:
        hits = sorted(directory.glob(pat))
        if hits:
            return hits[0]
    return None


def _finite(v, default=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


@dataclass
class AnalysisResult:
    """一条谱的一次分析：原始观测 + 后验 + 派生量 + 判定。"""

    sample_dir: Path
    json_path: Path
    npz_path: Path | None
    raw: dict

    f: np.ndarray
    z_obs: np.ndarray
    z_fit: np.ndarray
    band: dict                                   # re_lo/re_hi/im_lo/im_hi
    pred: dict | None                            # 同上，含噪声
    sigma: dict                                  # re/im（拟合的 σ 中位数）
    resid: dict                                  # re/im
    norm: dict                                   # re/im（除以自身 σ）
    rwp_pct: float | None
    drt: dict | None
    band_q: tuple = BAND_Q
    n_draws: int = 0
    notes: list = field(default_factory=list)

    # ---------------- 构造 ----------------

    @classmethod
    def load(cls, path) -> "AnalysisResult":
        """`path` 可以是样品输出目录，也可以是 `real_*.json` 本身。"""
        p = Path(path)
        if p.is_file():
            jp, d = p, p.parent
        else:
            d = p
            # 先认 `real_*.json`，再认导出包里的 `metadata/source_result.json`。
            # 不能直接 glob `*.json`——`results/<stem>/` 里第一个 json 是
            # manifest.json，读它会报「未拟合」。
            jp = _find_one(d, ("real_*.json",
                               "metadata/source_result.json",
                               "source_result.json"))
            if jp is None:
                raise FileNotFoundError(f"{d} 下没有 real_*.json")
        raw = json.loads(jp.read_text(encoding="utf-8"))
        npz = _find_one(d, ("*_samples.npz", "*samples*.npz",
                            "data/posterior_samples.npz"))

        notes = []
        if not raw.get("fit_performed", False):
            raise ValueError(
                f"{jp.name} 是「未拟合」记录（fit_performed=false，"
                f"常见于 KK 门禁拒绝）：没有后验，不能出图")

        f = z_obs = None
        sigma = {"re": _finite(raw.get("sigma_re")), "im": _finite(raw.get("sigma_im"))}
        band = pred = None
        n_draws = 0
        if npz is not None:
            s = {k: np.load(npz)[k] for k in np.load(npz).files}
            f = np.asarray(s["f"], float)
            z_obs = np.asarray(s["Zre"], float) + 1j * np.asarray(s["Zim"], float)
            n_draws = int(np.asarray(s["R_inf"]).size)
            include_L = bool(raw.get("include_L", False))
            idx = np.linspace(0, n_draws - 1,
                              min(MAX_DRAWS, n_draws)).astype(int)
            draws = np.empty((idx.size, f.size), complex)
            s_re = np.empty((idx.size, f.size))
            s_im = np.empty((idx.size, f.size))
            for a, i in enumerate(idx):
                draws[a] = B.model_z(f, B.sample_params(s, int(i), include_L),
                                     include_L)
                s_re[a] = np.broadcast_to(s["s_re"][i], f.shape)
                s_im[a] = np.broadcast_to(s["s_im"][i], f.shape)
            q_lo, q_hi = BAND_Q
            band = dict(
                re_lo=np.percentile(draws.real, q_lo, axis=0),
                re_hi=np.percentile(draws.real, q_hi, axis=0),
                im_lo=np.percentile(draws.imag, q_lo, axis=0),
                im_hi=np.percentile(draws.imag, q_hi, axis=0),
            )
            # 后验预测：同一样本的响应 + 它自己的噪声（σ 也随样本变）
            rng = np.random.default_rng(PRED_SEED)
            noisy = draws + rng.normal(0.0, 1.0, draws.shape) * s_re \
                + 1j * rng.normal(0.0, 1.0, draws.shape) * s_im
            pred = dict(
                re_lo=np.percentile(noisy.real, q_lo, axis=0),
                re_hi=np.percentile(noisy.real, q_hi, axis=0),
                im_lo=np.percentile(noisy.imag, q_lo, axis=0),
                im_hi=np.percentile(noisy.imag, q_hi, axis=0),
            )
            z_fit = np.median(draws.real, axis=0) + 1j * np.median(draws.imag, axis=0)
            sig_re = np.median(s_re, axis=0)
            sig_im = np.median(s_im, axis=0)
            sigma = {"re": sig_re, "im": sig_im}
        else:
            # 没有 npz：只能用 JSON 里的中位参数重建曲线（无区间）
            notes.append("缺少后验样本 npz：只给中位曲线，不给区间，"
                         "也不画参数相关图（不能伪造）")
            f, z_obs = cls._from_csv(d, raw, notes)
            z_fit = cls._z_from_json(f, raw)
            sigma = {"re": np.full(f.size, _finite(raw.get("sigma_re"), np.nan)),
                     "im": np.full(f.size, _finite(raw.get("sigma_im"), np.nan))}

        resid = {"re": z_obs.real - z_fit.real, "im": z_obs.imag - z_fit.imag}
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = {"re": resid["re"] / sigma["re"], "im": resid["im"] / sigma["im"]}
        if not np.all(np.isfinite(norm["re"])):
            notes.append("σ 缺失或为 0：标准化残差不可用（置空，不用 2% 顶替）")
            norm = {"re": None, "im": None}

        rwp_pct = None
        if REF is not None:
            try:
                rwp_pct = float(REF.rwp(f, z_obs, z_fit))
            except Exception:
                rwp_pct = None

        drt = None
        if npz is not None:
            try:
                g = B.drt_band(s)
                drt = dict(ln_tau=g[0], med=g[1], lo=g[2], hi=g[3])
            except Exception as exc:            # 缺 tau 的旧谱
                notes.append(f"后验 DRT 不可用：{type(exc).__name__}: {exc}")

        return cls(sample_dir=d, json_path=jp, npz_path=npz, raw=raw,
                   f=np.asarray(f, float), z_obs=np.asarray(z_obs, complex),
                   z_fit=np.asarray(z_fit, complex), band=band, pred=pred,
                   sigma=sigma, resid=resid, norm=norm, rwp_pct=rwp_pct,
                   drt=drt, n_draws=n_draws, notes=notes)

    @staticmethod
    def _from_csv(d: Path, raw: dict, notes: list):
        """无 npz 时从原始 csv 读观测（只读，不改任何数据）。"""
        try:
            import bm_experiments as E
            hits = list(d.glob("*.csv")) or list(
                (d / "z_csv").glob("*.csv"))
            if not hits:
                raise FileNotFoundError("同目录没有 csv，也没有 z_csv/")
            f, Z = E.load_spectrum(str(hits[0]))
            notes.append(f"观测数据取自 {hits[0].name}（不是 npz）")
            return np.asarray(f, float), np.asarray(Z, complex)
        except Exception as exc:
            raise FileNotFoundError(
                f"既无后验 npz 也读不到观测数据：{exc}") from exc

    @staticmethod
    def _z_from_json(f, raw):
        """用 JSON 里的中位参数重建 Z(f)。只支持基础模型 + 串联 L。"""
        comps = raw.get("components") or []
        if not comps:
            raise ValueError("JSON 既无 components 也无可重建的参数")
        tau = np.array([1.0 / (2 * np.pi * c["f0"]) for c in comps], float)
        # f0 是峰频不是 τ 的位置；用 w/φ 还原的 HN 峰位与之差一个 φ 因子，
        # 但这是唯一可得的近似，故只在无 npz 的降级路径上使用并明确标注。
        p = dict(R_inf=float(raw["R_inf"]), W=1.0,
                 pi=np.array([c["w"] for c in comps], float),
                 tau=tau, phi=np.array([c["phi"] for c in comps], float),
                 zeta=np.ones(len(comps)))
        if raw.get("L_uH"):
            p["L"] = float(raw["L_uH"]) * 1e-6
        return B.model_z(f, p, bool(raw.get("include_L", False)))

    # ---------------- 判定（口径诚实） ----------------

    @property
    def status(self) -> str:
        st = self.raw.get("k_selection_status")
        return _STATUS_ALIAS.get(st, st) or "unknown"

    @property
    def grade(self) -> int | None:
        g = self.raw.get("sample_grade")
        return None if g is None else int(g)

    @property
    def verdict(self) -> dict:
        """总判定：**采样档与结构判据分开陈述**，不合并成一句「通过」。

        `sample_grade=2` 只表示采样过了严格线（R̂/ESS/div/触顶），
        它**不表示**残差的频率结构也过线（M-b `acf≤0.50` 是另一回事）。
        旧代码把两者压成一个「通过」，正是这一轮要修的口径问题。
        """
        st, g = self.status, self.grade
        if st == "rejected":
            samp, s_level = "无可用 K", "bad"
        elif st == "exploratory":
            samp, s_level = "探索性（quick 档）", "warn"
        elif g == 2:
            samp, s_level = "采样达标（分级 2）", "ok"
        elif g == 1:
            samp, s_level = "采样降级（分级 1，未过严格线）", "warn"
        elif g == 0:
            samp, s_level = "采样不合格（分级 0）", "bad"
        else:
            samp, s_level = st or "未判定", "warn"

        ck = [c for c in self.checks if c["state"] != "na"]
        n_pass = sum(c["state"] == "pass" for c in ck)
        n_all = len(ck)
        if n_all and n_pass < n_all:
            struct = f"结构判据 {n_pass}/{n_all}"
            level = s_level if s_level == "bad" else "warn"
        elif n_all:
            struct, level = f"结构判据 {n_pass}/{n_all}", s_level
        else:
            struct, level = "结构判据不可用", "warn"
        label = f"{samp} · {struct}"
        return dict(status=st, grade=g, label=label, level=level,
                    sampling=samp, sampling_level=s_level, structure=struct,
                    structure_pass=n_pass, structure_total=n_all,
                    tier=self.raw.get("selection_tier"))

    @property
    def checks(self) -> list:
        """§14 三判据逐条，用预注册阈值，不改口径。"""
        acf = _finite(self.raw.get("acf"))
        rmax = None
        diag = self.raw.get("diag") or {}
        if diag:
            vals = [_finite(v.get("rhat")) for v in diag.values()]
            vals = [v for v in vals if v is not None]
            rmax = max(vals) if vals else None
        ndiv = self.raw.get("divergences")
        out = []

        def add(key, name, val, thr, cmp_):
            if val is None:
                out.append(dict(key=key, name=name, value=None, threshold=thr,
                                state="na", text=f"{name} 不可用"))
                return
            ok = val <= thr if cmp_ == "<=" else val == thr
            out.append(dict(key=key, name=name, value=float(val), threshold=thr,
                            state="pass" if ok else "fail",
                            text=f"{name}={val:.3g}"
                                 f"{'≤' if cmp_ == '<=' else '='}{thr}"))

        add("acf", "残差 acf", acf, THRESHOLDS["acf"], "<=")
        add("rhat", "max R̂", rmax, THRESHOLDS["rhat"], "<=")
        add("div", "divergence", None if ndiv is None else float(ndiv),
            THRESHOLDS["div"], "==")
        return out

    def candidates(self) -> tuple:
        """K 候选表。优先用结构化记录，其次从选择日志文本解析并**标注来源**。"""
        struct = self.raw.get("k_candidates")
        if isinstance(struct, list) and struct:
            return struct, "structured"
        rows, seen = [], {}
        _rank = {"rejected": 3, "degraded": 2, "selected": 1, "noted": 0}
        for line in (self.raw.get("k_selection") or []):
            k = _parse_k(line)
            if k is None:
                continue
            # 同一 K 可能出现在多行日志里（先「剔除」后「只进可用档」），
            # 只留最严重的那一条，别在表里给同一 K 两行互相矛盾的结论
            old = seen.get(k["k"])
            if old is None or _rank.get(k["state"], 0) > _rank.get(old["state"], 0):
                seen[k["k"]] = k
        rows = [seen[key] for key in sorted(seen)]
        sel = self.raw.get("K")
        if sel is not None and int(sel) not in seen:
            rows.append(dict(k=int(sel), state=self.status, detail="",
                             selected=True))
        for r in rows:
            r["selected"] = (r["k"] == sel)
        return rows, "note_text"

    def param_rows(self) -> list:
        pc = self.raw.get("param_corr") or {}
        rows = []
        for s in (pc.get("summary") or []):
            rows.append(dict(name=s.get("name"), q05=s.get("q05"),
                             q50=s.get("q50"), q95=s.get("q95"),
                             sd=s.get("sd"), log=bool(s.get("log")),
                             kind="sampled"))
        for v in (pc.get("fixed") or []):
            rows.append(dict(name=v.get("name"), q05=v.get("value"),
                             q50=v.get("value"), q95=v.get("value"),
                             sd=0.0, log=False, kind="fixed"))
        return rows

    def correlation(self):
        pc = self.raw.get("param_corr") or {}
        names, mat = pc.get("names"), pc.get("corr")
        if not names or not mat:
            return None
        return dict(names=list(names), matrix=mat,
                    strong=pc.get("strong") or [],
                    medium=pc.get("medium") or [],
                    degenerate=pc.get("degenerate") or [])

    # ---------------- 出口 1：数据包 ----------------

    def bundle(self, root, *, figures_from=None) -> Path:
        """写 `data/ metadata/ manifest.json`（figures/ 由 bm_figure 填）。"""
        root = Path(root)
        (root / "data").mkdir(parents=True, exist_ok=True)
        (root / "metadata").mkdir(parents=True, exist_ok=True)
        (root / "figures").mkdir(parents=True, exist_ok=True)
        stem = self.raw.get("file") or self.sample_dir.name

        def wcsv(name, header, cols):
            p = root / "data" / name
            with open(p, "w", encoding="utf-8", newline="") as fh:
                fh.write(",".join(header) + "\n")
                for i in range(len(cols[0])):
                    fh.write(",".join("" if v is None else f"{v:.12g}"
                                      for v in (c[i] for c in cols)) + "\n")
            return p

        f = self.f
        wcsv("input_data.csv", ["frequency_Hz", "Zre_obs", "Zim_obs"],
             [f, self.z_obs.real, self.z_obs.imag])

        if self.band is not None:
            wcsv("fitted_impedance.csv",
                 ["frequency_Hz", "Zre_observed", "Zim_observed",
                  "Zre_fitted_median", "Zim_fitted_median",
                  "Zre_lower90", "Zre_upper90", "Zim_lower90", "Zim_upper90",
                  "Zre_pred_lower90", "Zre_pred_upper90",
                  "Zim_pred_lower90", "Zim_pred_upper90",
                  "residual_re", "residual_im", "sigma_re", "sigma_im",
                  "normalized_residual_re", "normalized_residual_im"],
                 [f, self.z_obs.real, self.z_obs.imag,
                  self.z_fit.real, self.z_fit.imag,
                  self.band["re_lo"], self.band["re_hi"],
                  self.band["im_lo"], self.band["im_hi"],
                  self.pred["re_lo"], self.pred["re_hi"],
                  self.pred["im_lo"], self.pred["im_hi"],
                  self.resid["re"], self.resid["im"],
                  self.sigma["re"], self.sigma["im"],
                  self.norm["re"], self.norm["im"]])
        else:
            wcsv("fitted_impedance.csv",
                 ["frequency_Hz", "Zre_observed", "Zim_observed",
                  "Zre_fitted_median", "Zim_fitted_median",
                  "residual_re", "residual_im", "sigma_re", "sigma_im"],
                 [f, self.z_obs.real, self.z_obs.imag,
                  self.z_fit.real, self.z_fit.imag,
                  self.resid["re"], self.resid["im"],
                  self.sigma["re"], self.sigma["im"]])

        sig_re = (self.norm["re"] is not None)
        wcsv("residuals.csv",
             ["frequency_Hz", "residual_re", "residual_im", "sigma_re", "sigma_im",
              "normalized_residual_re", "normalized_residual_im"],
             [f, self.resid["re"], self.resid["im"], self.sigma["re"],
              self.sigma["im"],
              self.norm["re"] if sig_re else [None] * f.size,
              self.norm["im"] if sig_re else [None] * f.size])

        if self.drt is not None:
            fq = 1.0 / (2 * np.pi * np.exp(self.drt["ln_tau"]))
            wcsv("drt_distribution.csv",
                 ["frequency_Hz", "tau_s", "gamma_median", "gamma_lower90",
                  "gamma_upper90"],
                 [fq, np.exp(self.drt["ln_tau"]), self.drt["med"],
                  self.drt["lo"], self.drt["hi"]])

        prows = self.param_rows()
        p = root / "data" / "parameter_summary.csv"
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write("parameter,kind,q05,q50,q95,sd,log_scale\n")
            for r in prows:
                fh.write(",".join([str(r["name"]), r["kind"]]
                                  + ["" if r[k] is None else f"{r[k]:.12g}"
                                     for k in ("q05", "q50", "q95", "sd")]
                                  + [str(int(r["log"]))]) + "\n")

        # 源结果 JSON 也带走一份：导出包要能**自己**重新出图与再生成面板，
        # 否则"可复现"只在本机的 out/ 还在时成立。
        shutil.copyfile(self.json_path, root / "metadata" / "source_result.json")

        # 后验样本：原样带走（参数名/维度/链信息见 fit_config.json）
        npz_out = root / "data" / "posterior_samples.npz"
        if self.npz_path is not None:
            shutil.copyfile(self.npz_path, npz_out)

        meta_pc = self.raw.get("param_corr") or {}
        (root / "metadata" / "fit_config.json").write_text(json.dumps({
            "spec_version": SPEC_VERSION,
            "source_json": str(self.json_path),
            "source_npz": str(self.npz_path) if self.npz_path else None,
            "file": stem, "n_points": int(self.f.size),
            "f_min_Hz": float(self.f.min()), "f_max_Hz": float(self.f.max()),
            "K": self.raw.get("K"), "include_L": self.raw.get("include_L"),
            "zeta": self.raw.get("zeta"),
            "sigma_floor": self.raw.get("sigma_floor"),
            "seed": self.raw.get("seed"), "sampling": self.raw.get("sampling"),
            "sigma_model": "per-point posterior sigma (fit's own), NOT 2%",
            "band_quantiles": list(BAND_Q),
            "band_kind": "posterior-propagated response (marginal Re/Im)",
            "pred_kind": "posterior predictive incl. fitted noise",
            "pred_seed": PRED_SEED,
            "posterior_draws": self.n_draws,
            "posterior_draws_used": min(MAX_DRAWS, self.n_draws) if self.n_draws else 0,
            "param_names": meta_pc.get("names"),
            "param_fixed": [v.get("name") for v in (meta_pc.get("fixed") or [])],
            "thresholds": THRESHOLDS,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        cands, cand_src = self.candidates()
        (root / "metadata" / "model_comparison.json").write_text(json.dumps({
            "selected_K": self.raw.get("K"),
            "status": self.status,
            "selection_tier": self.raw.get("selection_tier"),
            "candidates_source": cand_src,
            "candidates": cands,
            "elpd_loo": self.raw.get("elpd_loo"),
            "elpd_se": self.raw.get("elpd_se"),
            "elpd_diff": self.raw.get("elpd_diff"),
            "dse": self.raw.get("dse"),
            "comparison_performed": self.raw.get("comparison_performed"),
            "notes": self.raw.get("k_selection"),
            "caveat": ("配对 LOO 只在扫过的 K 范围内成立，"
                       "不是全局最优；indistinguishable 不等于有最优模型"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        (root / "metadata" / "analysis_report.json").write_text(json.dumps({
            "spec_version": SPEC_VERSION,
            "file": stem,
            "verdict": self.verdict,
            "checks": self.checks,
            "rwp_pct": self.rwp_pct,
            "acf": self.raw.get("acf"),
            "cover_re": self.raw.get("cover_re"),
            "cover_im": self.raw.get("cover_im"),
            "rms_re": self.raw.get("rms_re"), "rms_im": self.raw.get("rms_im"),
            "kk": self.raw.get("kk"),
            "components": self.raw.get("components"),
            "component_review": self.raw.get("component_review"),
            "degeneracy": self.raw.get("degeneracy"),
            "diffusion_resolution": self.raw.get("diffusion_resolution"),
            "warnings": self.raw.get("warnings"),
            "layer_notes": self.notes,
            "band_note": ("Re/Im 分别取 5/95% 分位 = 复平面矩形包络，"
                          "不是联合可信域；区间为参数后验传播，非预测区间"),
        }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

        files = []
        for pth in sorted(root.rglob("*")):
            if pth.is_file() and pth.name != "manifest.json":
                files.append(dict(path=str(pth.relative_to(root)).replace("\\", "/"),
                                  bytes=pth.stat().st_size,
                                  sha256=_sha256(pth)))
        manifest = dict(
            spec_version=SPEC_VERSION,
            generator="bm_result.AnalysisResult.bundle",
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            sample=stem,
            source=dict(
                json=dict(path=str(self.json_path),
                          sha256=_sha256(self.json_path)),
                npz=(dict(path=str(self.npz_path),
                          sha256=_sha256(self.npz_path))
                     if self.npz_path else None),
            ),
            files=files,
            reproduce=dict(
                figure="python bm_figure.py <sample_dir> --format pdf,svg,tiff",
                dashboard="python bm_dash.py <sample_dir>",
                note="data/ 与 figures/ 同源于本 manifest 记录的源文件",
            ),
        )
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return root

    # ---------------- 出口 2：交互面板载荷 ----------------

    def payload(self) -> dict:
        cands, cand_src = self.candidates()
        corr = self.correlation()
        comps = []
        for i, c in enumerate(self.raw.get("components") or []):
            comps.append(dict(idx=i + 1, f0=c.get("f0"), f0_lo=c.get("f0_lo"),
                              f0_hi=c.get("f0_hi"), w=c.get("w"),
                              w_lo=c.get("w_lo"), w_hi=c.get("w_hi"),
                              phi=c.get("phi"),
                              p_out=c.get("p_peak_outside")))
        d = dict(
            spec_version=SPEC_VERSION,
            name=self.raw.get("file") or self.sample_dir.name,
            n=int(self.f.size),
            f_min=float(self.f.min()), f_max=float(self.f.max()),
            K=self.raw.get("K"), include_L=bool(self.raw.get("include_L")),
            include_L_note=str(self.raw.get("include_L")),
            verdict=self.verdict,
            checks=self.checks,
            thresholds=THRESHOLDS,
            rwp=self.rwp_pct,
            rwp_def="100·sqrt(mean(|ΔZ|²/|Z|²))，每频点等权，实虚联合，不除 √2",
            acf=_finite(self.raw.get("acf")),
            rms=dict(re=_finite(self.raw.get("rms_re")),
                     im=_finite(self.raw.get("rms_im"))),
            cover=dict(re=_finite(self.raw.get("cover_re")),
                       im=_finite(self.raw.get("cover_im"))),
            elpd=_finite(self.raw.get("elpd_loo")),
            elpd_se=_finite(self.raw.get("elpd_se")),
            elpd_diff=_finite(self.raw.get("elpd_diff")),
            comparison=bool(self.raw.get("comparison_performed")),
            dse=_finite(self.raw.get("dse")),
            kk=self.raw.get("kk"),
            components=comps,
            candidates=cands,
            candidates_source=cand_src,
            params=self.param_rows(),
            corr=corr,
            warnings=self.raw.get("warnings") or [],
            layer_notes=self.notes,
            selection_notes=self.raw.get("k_selection") or [],
            band_note=("阴影/误差棒 = 参数后验传播的 5–95% **边际**区间"
                       "（Re、Im 各自取分位，复平面上是矩形包络，不是联合可信域）；"
                       "预测区间另算（含拟合噪声），不在同一层"),
            sigma_note="标准化残差除以本次拟合的逐点 σ 中位数，不套 2% 观测噪声",
            has_band=self.band is not None,
            has_pred=self.pred is not None,
        )
        d["curves"] = dict(
            f=self.f.tolist(),
            zre=self.z_obs.real.tolist(), zim=self.z_obs.imag.tolist(),
            fre=self.z_fit.real.tolist(), fim=self.z_fit.imag.tolist(),
            sre=np.asarray(self.sigma["re"]).tolist(),
            sim=np.asarray(self.sigma["im"]).tolist(),
            rre=self.resid["re"].tolist(), rim=self.resid["im"].tolist(),
        )
        if self.band is not None:
            d["curves"].update({
                "re_lo": self.band["re_lo"].tolist(),
                "re_hi": self.band["re_hi"].tolist(),
                "im_lo": self.band["im_lo"].tolist(),
                "im_hi": self.band["im_hi"].tolist(),
            })
        if self.pred is not None:
            d["curves"].update({
                "pre_lo": self.pred["re_lo"].tolist(),
                "pre_hi": self.pred["re_hi"].tolist(),
                "pim_lo": self.pred["im_lo"].tolist(),
                "pim_hi": self.pred["im_hi"].tolist(),
            })
        if self.norm["re"] is not None:
            d["curves"].update({"nre": self.norm["re"].tolist(),
                                "nim": self.norm["im"].tolist()})
        if self.drt is not None:
            d["drt"] = dict(
                f=(1.0 / (2 * np.pi * np.exp(self.drt["ln_tau"]))).tolist(),
                med=self.drt["med"].tolist(), lo=self.drt["lo"].tolist(),
                hi=self.drt["hi"].tolist())
        return d


def _parse_k(text: str):
    """从 `select_k()` 的中文日志行解析候选状态；解析不了就返回 None。"""
    import re
    m = re.search(r"K\s*=\s*(\d+)", text)
    if not m:
        return None
    k = int(m.group(1))
    if "剔除" in text:
        state = "rejected"
    elif "只进「可用」档" in text or "降级" in text:
        state = "degraded"
    elif "选中" in text:
        state = "selected"
    else:
        state = "noted"
    detail = text.split("：", 1)[1] if "：" in text else ""
    return dict(k=k, state=state, detail=detail, selected=False)


def load_any(path) -> AnalysisResult:
    return AnalysisResult.load(path)


if __name__ == "__main__":                   # 手工核对一个样品
    import sys
    r = AnalysisResult.load(sys.argv[1])
    print(f"{r.raw.get('file')}  n={r.f.size}  K={r.raw.get('K')}  "
          f"Rwp={r.rwp_pct:.3f}%  acf={r.raw.get('acf')}")
    print("verdict:", r.verdict)
    for c in r.checks:
        print("  ", c["state"], c["text"])
    if len(sys.argv) > 2:
        print("bundle ->", r.bundle(sys.argv[2]))
