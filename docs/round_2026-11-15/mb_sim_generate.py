"""M-B-SIM-01 生成端：按 `MB_SIM01_PROTOCOL.md` §2/§3/§4 生成盲测模拟谱。

只写两处，物理分离：
  out/mb_sim01/generated/SIM-xxxx.csv   仅 f,Zre,Zim（盲）
  out/mb_sim01/truth/*.json             标签 + 真值参数 + 种子（解盲端才读）

用法
----
    python docs/round_2026-11-15/mb_sim_generate.py --selftest
    python docs/round_2026-11-15/mb_sim_generate.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evidence", "nested_topology"))

import bm_real as R      # noqa: E402
import bm_top as T       # noqa: E402

# ---- 冻结常数（与 PROTOCOL §3/§4 逐条对齐）---------------------------------
BASE_SEED = 20261115
GRID_SRC = os.path.join("out", "_m6_samples", "z_csv", "AZ31-8Li_50.csv")
POOL_SRC = os.path.join("out", "m4_holdout2_ess.json")
OUT_ROOT = os.path.join("out", "mb_sim01")
N_INST = 20                     # 每组实例数；C 组前 10 = C1，后 10 = C2
PCT_CYCLE = (2.0, 4.0)          # i 偶数 2%，奇数 4%
BRANCH, LAYERS = "parallel", 2

RANGES = dict(
    # H1 仪器/夹具（B、D）
    Rs=(0.05, 5.0),             # Ω
    Ls=(1e-7, 1e-4),            # H
    Cp=(1e-11, 1e-8),           # F
    Rp=(1e3, 1e7),              # Ω
    # H2 样品侧（C、D）
    Re=(1.0, 40.0),             # Ω
    fe=(1e-2, 1e4),             # Hz（v2：峰值可落在窗口上端附近）
    alpha=(0.30, 0.80),
    gamma=(0.20, 0.60),
    Rw=(5.0, 60.0),             # Ω，有限长 Warburg 的低频平台
    fw=(1e-2, 1e4),             # Hz，特征频率
)
GROUPS = ("A", "B", "C1", "C2", "D")


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def grid() -> np.ndarray:
    f, _ = R.load_spectrum(GRID_SRC)
    if f.size != 50:
        raise SystemExit(f"栅格点数 {f.size} != 50，口径变了，停")
    return np.asarray(f, float)


def truth_pool() -> list[dict]:
    d = json.load(open(POOL_SRC, encoding="utf-8"))
    return [dict(r["rows"][0]["params"]) for r in d["rows"]]


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def draw_theta(rng, pool):
    """从 14 条真实谱的拟合中位参数池抽一条 + 量级抖动（PROTOCOL §2）。"""
    base = dict(pool[int(rng.integers(len(pool)))])
    th = {}
    for k, v in base.items():
        if k.startswith("n"):
            th[k] = float(np.clip(v + rng.normal(0, 0.03), 0.30, 1.00))
        else:
            th[k] = float(v * np.exp(rng.normal(0, 0.15)))
    return th


def draw_phi(rng):
    return dict(Rs=_logu(rng, *RANGES["Rs"]), Ls=_logu(rng, *RANGES["Ls"]),
                Cp=_logu(rng, *RANGES["Cp"]), Rp=_logu(rng, *RANGES["Rp"]))


def draw_psi(rng, kind: str):
    if kind == "C1":
        return dict(kind="C1", Re=_logu(rng, *RANGES["Re"]), fe=_logu(rng, *RANGES["fe"]),
                    alpha=float(rng.uniform(*RANGES["alpha"])),
                    gamma=float(rng.uniform(*RANGES["gamma"])))
    return dict(kind="C2", Rw=_logu(rng, *RANGES["Rw"]), fw=_logu(rng, *RANGES["fw"]))


def z_parasitic(f, Zs, phi):
    """PROTOCOL §2 的 B 组变换：Rs + jωLs + 1/(1/Zs + 1/Rp + jωCp)。"""
    w = 2.0 * np.pi * np.asarray(f, float)
    y = 1.0 / Zs + 1.0 / float(phi["Rp"]) + 1j * w * float(phi["Cp"])
    return float(phi["Rs"]) + 1j * w * float(phi["Ls"]) + 1.0 / y


def z_extra(f, psi):
    w = 2.0 * np.pi * np.asarray(f, float)
    if psi["kind"] == "C1":
        x = 1j * w * (1.0 / (2.0 * np.pi * float(psi["fe"])))
        return float(psi["Re"]) / (1.0 + x ** float(psi["alpha"])) ** float(psi["gamma"])
    # 有限长 Warburg（tanh 型分布式响应）：低频 → Rw，高频 → Rw/sqrt(jωτ)
    tau = 1.0 / (2.0 * np.pi * float(psi["fw"]))
    z = np.sqrt(1j * w * tau)
    return float(psi["Rw"]) * np.tanh(z) / z


def add_noise(rng, Z_clean, pct):
    sg = pct / 100.0 * np.abs(Z_clean)
    return Z_clean + sg * (rng.normal(size=Z_clean.size) + 1j * rng.normal(size=Z_clean.size))


def build_instance(f, pool, i: int) -> dict:
    """生成实例 i 的 A/B/C/D 四条（同一 θ、同一 ε）。"""
    ss = lambda k: np.random.default_rng(  # noqa: E731
        np.random.SeedSequence([BASE_SEED, i, k]))
    r_theta, r_noise = ss(1), ss(2)
    r_phi, r_psi = ss(3), ss(4)
    theta = draw_theta(r_theta, pool)
    phi = draw_phi(r_phi)
    kind = "C1" if i < N_INST // 2 else "C2"
    psi = draw_psi(r_psi, kind)
    pct = PCT_CYCLE[i % 2]
    zs = T.z_nested(f, theta, branch=BRANCH, layers=LAYERS)
    ze = z_extra(f, psi)

    spec = {}
    # A：理想基线
    Za = zs
    ea = add_noise(r_noise, Za, pct)
    spec["A"] = dict(Z_clean=Za, Z=ea, phi=None, psi=None)
    # B：只有寄生
    Zb = z_parasitic(f, zs, phi)
    spec["B"] = dict(Z_clean=Zb, Z=add_noise(r_noise, Zb, pct), phi=phi, psi=None)
    # C：只有样品侧额外结构
    Zc = zs + ze
    spec["C"] = dict(Z_clean=Zc, Z=add_noise(r_noise, Zc, pct), phi=None, psi=psi)
    # D：两者都有，寄生作用在 (Z_s + Z_extra) 上
    Zd = z_parasitic(f, zs + ze, phi)
    spec["D"] = dict(Z_clean=Zd, Z=add_noise(r_noise, Zd, pct), phi=phi, psi=psi)

    out = {}
    for g in ("A", "B", "C", "D"):
        out[g] = dict(Z=spec[g]["Z"], truth=dict(
            theta=theta, phi=spec[g]["phi"], psi=spec[g]["psi"], pct=pct,
            inst=i, group=g if g != "C" else kind,
            seed_streams=dict(theta=[BASE_SEED, i, 1], noise=[BASE_SEED, i, 2],
                              parasitic=[BASE_SEED, i, 3], extra=[BASE_SEED, i, 4])))
    return out


def write_csv(path, f, Z):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("f,Zre,Zim\n")
        for fj, zj in zip(f, Z):
            fh.write(f"{fj:.10g},{zj.real:.12g},{zj.imag:.12g}\n")


def blind_ids() -> dict:
    """固定置换：把 (group,inst) 映射到 SIM 编号（PROTOCOL §5）。"""
    keys = []
    for i in range(N_INST):
        keys += [("A", i), ("B", i), ("D", i),
                 ("C1" if i < N_INST // 2 else "C2", i)]
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 9999]))
    perm = rng.permutation(len(keys))
    return {keys[j]: f"SIM-{n + 1:04d}" for n, j in enumerate(perm)}


def selftest(f, pool) -> int:
    ok, total, bad = 0, 0, []
    def c(name, cond):
        nonlocal ok, total
        total += 1
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")
        if cond:
            ok += 1
        else:
            bad.append(name)

    c("栅格 50 点 / 0.01–1e5", f.size == 50 and abs(f[0] - 0.01) < 1e-9 and abs(f[-1] - 1e5) < 1e-3)
    c("栅格等比 7 点/decade", abs(np.log10(f[1] / f[0]) - 1.0 / 7.0) < 2e-3)
    c("真值池 14 条", len(pool) == 14)

    Zs = T.z_nested(f, pool[0], branch=BRANCH, layers=LAYERS)
    c("正演有限且 |Z|>0", np.all(np.isfinite(Zs)) and np.all(np.abs(Zs) > 0))

    phi0 = dict(Rs=0.0, Ls=0.0, Cp=0.0, Rp=1e30)
    c("寄生参数全零 ⇒ 恒等", np.allclose(z_parasitic(f, Zs, phi0), Zs, rtol=1e-12))

    # 纯串联 L 的解析极限：Rs=0, Cp=0, Rp=inf ⇒ Z = Zs + jωL
    ph = dict(Rs=0.0, Ls=3e-5, Cp=0.0, Rp=1e30)
    c("纯 Ls ⇒ Zs + jωL", np.allclose(z_parasitic(f, Zs, ph), Zs + 1j * 2 * np.pi * f * 3e-5,
                                      rtol=1e-10))

    # 有限长 Warburg：低频 → Rw，高频按 |z|^-1 衰减（z = sqrt(jωτ)）
    psi2 = dict(kind="C2", Rw=30.0, fw=1e4)
    z2 = z_extra(f, psi2)
    c("C2 低频 → Rw", abs(z2[0] - 30.0) / 30.0 < 1e-3)
    zz = np.sqrt(1j * 2 * np.pi * f[-1] / (2 * np.pi * 1e4))
    c("C2 高频 → Rw/sqrt(jωτ) (5% 渐近)", abs(z2[-1] - 30.0 / zz) / abs(z2[-1]) < 0.05)
    c("C2 高频衰减", abs(z2[-1]) < abs(z2[0]))

    # C1：低频 → Re，高频按 (ωτ)^{-αγ} 衰减
    # 宽分布尾收敛慢：取 fe 远高于窗口底、αγ 较大时才逼近 Re
    psi1 = dict(kind="C1", Re=20.0, fe=1e3, alpha=0.8, gamma=0.6)
    z1 = z_extra(f, psi1)
    c("C1 低频 → Re", abs(z1[0] - 20.0) / 20.0 < 5e-3)
    c("C1 高频衰减", abs(z1[-1]) < abs(z1[0]))

    # 噪声尺度：rms(|eps|)/|Z| ≈ pct/sqrt(1)（复分量各自 sd=pct·|Z|）
    rng = np.random.default_rng(0)
    e = add_noise(rng, Zs, 2.0) - Zs
    c("噪声逐分量 sd = pct·|Z|",
      abs(float(np.std((e / np.abs(Zs)).real)) - 0.02) < 2e-3
      and abs(float(np.std((e / np.abs(Zs)).imag)) - 0.02) < 2e-3)

    # CSV 往返：写-读-逐位比较
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.csv")
        write_csv(p, f, Zs)
        f2, Z2 = R.load_spectrum(p)
    c("CSV 往返：频率逐位相同", np.array_equal(f, f2))
    c("CSV 往返：Z 相对误差 < 1e-10", np.allclose(Z2, Zs, rtol=1e-10, atol=0))

    # 配对性：同一 i 的 A/B/C/D 必须共用 θ
    bi = build_instance(f, pool, 3)
    c("配对：A/B/C/D 的 θ 相同",
      bi["A"]["truth"]["theta"] == bi["B"]["truth"]["theta"] == bi["D"]["truth"]["theta"])
    c("配对：噪声档一致", len({bi[g]["truth"]["pct"] for g in bi}) == 1)
    c("C 组按 i 分 C1/C2", bi["C"]["truth"]["group"] == "C1"
      and build_instance(f, pool, 12)["C"]["truth"]["group"] == "C2")

    print(f"\n自检 {ok}/{total} 通过" if not bad else f"\nFAIL {len(bad)}/{total}: {bad}")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT_ROOT)
    a = ap.parse_args(argv)

    f = grid()
    pool = truth_pool()
    if a.selftest:
        return selftest(f, pool)

    gen_dir = os.path.join(a.out, "generated")
    tr_dir = os.path.join(a.out, "truth")
    cfg_dir = os.path.join(a.out, "config")
    for d in (gen_dir, tr_dir, cfg_dir):
        os.makedirs(d, exist_ok=True)

    ids = blind_ids()
    manifest = []
    for i in range(N_INST):
        inst = build_instance(f, pool, i)
        for g, rec in inst.items():
            sid = ids[(rec["truth"]["group"], i)]
            write_csv(os.path.join(gen_dir, sid + ".csv"), f, rec["Z"])
            t = dict(sim_id=sid, **rec["truth"],
                     generator=os.path.basename(__file__),
                     generator_sha256=_sha(__file__),
                     protocol="MB_SIM01_PROTOCOL.md §2–§4")
            json.dump(t, open(os.path.join(tr_dir, "truth_" + sid + ".json"), "w",
                              encoding="utf-8"), ensure_ascii=False, indent=1)
            manifest.append(dict(sim_id=sid, group=t["group"], inst=i,
                                 pct=t["pct"], file=sid + ".csv"))

    cfg = dict(base_seed=BASE_SEED, n_inst=N_INST, pct_cycle=list(PCT_CYCLE),
               branch=BRANCH, layers=LAYERS, ranges=RANGES,
               grid=dict(n=int(f.size), fmin=float(f[0]), fmax=float(f[-1])),
               grid_src=GRID_SRC, pool_src=POOL_SRC,
               generator_sha256=_sha(__file__),
               n_spectra=len(manifest),
               counts={g: sum(1 for m in manifest if m["group"] == g) for g in GROUPS})
    json.dump(cfg, open(os.path.join(cfg_dir, "simulation_config.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(dict(manifest=manifest), open(os.path.join(cfg_dir, "seed_manifest.json"),
                                            "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"生成 {len(manifest)} 条：", cfg["counts"])
    print("盲目录：", gen_dir, " 真值目录：", tr_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
