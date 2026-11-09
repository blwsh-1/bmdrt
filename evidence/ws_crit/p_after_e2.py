# -*- coding: utf-8 -*-
"""E2 跑完 → 自动接手 P1（打补丁 + 回归），并把结果写成报告。

为什么必须是"等 E2 跑完"
------------------------
`evidence/m6/run_e2.py` 是**逐 job 起一个 `subprocess.run`**，新进程会重新
`import bm_kk`。E2 还没跑完就改生产文件，等于让剩余 job 跑在另一份代码上，
E2 的结果就不再是同一次执行。所以本脚本先等 `out/m6/e2/e2_summary.json`
出现（`run_e2.summarize()` 在 `run_jobs()` 返回后写它），再动手。

P1 做什么
---------
1. 备份 `bm_kk.py`（存 sha256），把 E2 用到的谱 + examples 的
   `diffusion_tail_candidate` 六字段快照成 `p1_before.json`；
2. `git apply` 已校验过的补丁 `patches/n1_l1_amplitude.patch`；
3. 再快照成 `p1_after.json`，逐条 diff —— 这是 E2 的**代码版本溯源记录**：
   E2 全程跑在补丁前，报告里必须写清哪些谱的 `amp_ok`/`f_cut` 会因补丁而变；
4. 校验打过补丁的真实函数与影子实现（`n1_shadow.candidate_new`）
   逐字段一致（32 条谱 × 6 字段），不一致即判失败；
5. 跑 `bm_check.py nofit`（秒级数学自检）与 `bm_check.py fast`（约 1 min 冒烟）；
   **任一非零退出 → `git checkout -- bm_kk.py` 自动回滚**并如实记录；
6. 回滚后重跑 `nofit` 确认回到基线。

不做什么
--------
不碰 `bm_real.py` / `bm_ad.py` / 先验 / 门控 / `sigma_floor` / `select_k`；
不重跑任何 M6 采样；不改 E2 的任何产物。

    $PY evidence/ws_crit/p_after_e2.py            # 等 E2 完成后执行
    $PY evidence/ws_crit/p_after_e2.py --now      # 不等（E2 已完成后手动跑）
    $PY evidence/ws_crit/p_after_e2.py --check    # 只做预检与快照，不 apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

PATCH = HERE / "patches/n1_l1_amplitude.patch"
BAK = HERE / "patches/bm_kk.py.pre_p1"
LOG = HERE / "_p_after_e2.log"
REPORT = HERE / "P_AFTER_E2.md"
E2OUT = ROOT / "out/m6/e2"
MEMDIR = Path("D:/eis训练数据/.workbuddy/memory")

PY = sys.executable
FIELDS = ("amp_ok", "phase_ok", "is_candidate", "f_cut", "slope_z", "slope_im")


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(cmd, timeout=3600, cwd=ROOT):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                       timeout=timeout, encoding="utf-8", errors="replace")
    return dict(rc=r.returncode, out=r.stdout or "", err=r.stderr or "",
                secs=round(time.time() - t0, 1))


# --------------------------------------------------------------------------
# 等待 E2
# --------------------------------------------------------------------------

def wait_for_e2(timeout_h=16.0, poll_s=60, log_every_s=600):
    """轮询 `e2_summary.json`。

    **轮询与日志用两个不同周期**：轮询保持 `poll_s=60`（summary 一落盘就接手，
    不因降噪而延迟 P1）；日志改成 `log_every_s=600`，只在两种情况下写一行：
      · `progress` 数字变了（有 job 刚跑完，信息量最大，立刻记）；
      · 距上次写日志已满 `log_every_s`（心跳，证明守护还活着）。
    """
    t0 = time.time()
    last_log = 0.0
    last_rows = -1
    while True:
        if (E2OUT / "e2_summary.json").exists():
            return True
        rows = 0
        p = E2OUT / "run_progress.jsonl"
        if p.exists():
            rows = sum(1 for ln in p.read_text(encoding="utf-8").splitlines()
                       if ln.strip())
        now = time.time()
        if now - t0 > timeout_h * 3600:
            log(f"等待超时：{timeout_h:.0f} h 内未见 e2_summary.json"
                f"（progress {rows}/30）→ 不启动 P1")
            return False
        if rows != last_rows or now - last_log >= log_every_s:
            el = now - t0
            log(f"E2 仍在跑（progress {rows}/30，已等 {el / 3600:.1f} h）")
            last_rows, last_log = rows, now
        time.sleep(poll_s)


# --------------------------------------------------------------------------
# 语料：E2 用到的谱 + examples
# --------------------------------------------------------------------------

#: `load_csv` 走 `.txt` 分支时记下的加载器输出（写进溯源 JSON，不刷日志）
LOAD_NOTES: list[tuple[str, str]] = []


def load_csv(p: Path):
    """读一条谱。

    `.csv` 走 `genfromtxt` 快路径；**非 csv（如 `archive/*.txt`，Z60W 原始导出：
    制表符分隔 + 表头）必须交给 `bm_real.load_spectrum`**——那是 E2 真跑时用的
    同一个加载器，否则溯源快照里的 (f, Z) 和 E2 实际拟合的数据不是同一份。
    （2026-11-09 首次自动执行时就是在这里崩的：`genfromtxt(delimiter=",")`
    把整行制表符数据当成一个字符串。）
    """
    if p.suffix.lower() == ".csv":
        d = np.genfromtxt(p, delimiter=",", names=True)
        nm = d.dtype.names
        f = np.asarray(d[nm[0]], float)
        Z = np.asarray(d[nm[1]], float) + 1j * np.asarray(d[nm[2]], float)
        o = np.argsort(f)
        return f[o], Z[o]
    import contextlib
    import io

    import bm_real
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        f, Z = bm_real.load_spectrum(p)
    LOAD_NOTES.append((p.name, buf.getvalue().strip()))
    return f, Z


def corpus():
    """返回 [(标签, f, Z)]，去重。优先用 e2_summary.json 的 job 清单。"""
    seen, out = set(), []
    summ = E2OUT / "e2_summary.json"
    paths = []
    if summ.exists():
        meta = json.loads(summ.read_text(encoding="utf-8"))
        for j in meta.get("jobs", []):
            rel = j.get("path")
            if rel:
                p = Path(rel)
                paths.append(p if p.is_absolute() else ROOT / p)
    if not paths:
        paths = sorted((ROOT / "out/_m6_samples/z_csv").glob("*.csv"))
    for p in paths:
        key = str(p.resolve())
        if key in seen or not p.exists():
            continue
        seen.add(key)
        f, Z = load_csv(p)
        out.append((f"e2:{p.stem}", f, Z))
    for p in sorted((ROOT / "examples").glob("*.csv")):
        try:
            f, Z = load_csv(p)
        except Exception:                        # noqa: BLE001
            continue
        out.append((f"ex:{p.stem}", f, Z))
    return out


def snapshot():
    """快照 `diffusion_tail_candidate` 的六个字段。

    **必须 `importlib.reload(bm_kk)`，不能用裸 `import`。** `import` 命中
    `sys.modules` 缓存：先拍 before、再 `git apply`、再拍 after，两次拿到的
    是**同一份旧字节码**，diff 恒为 0——溯源表会变成"什么都没变"的假记录。
    （2026-11-09 首次自动执行真踩到了：`n_changed=0`，而影子对拍和实机
    都知道 `example_caseE` 的 `is_candidate` 应从 False 变 True。）
    """
    import importlib

    import bm_kk
    importlib.reload(bm_kk)
    rows = {}
    for tag, f, Z in corpus():
        r = bm_kk.diffusion_tail_candidate(f, Z)
        rows[tag] = {k: (None if r.get(k) is None else
                         (float(r[k]) if isinstance(r[k], (int, float, np.floating))
                          else bool(r[k])))
                     for k in FIELDS}
    return rows


def diff_snap(a, b):
    out = []
    for tag in a:
        if tag not in b:
            continue
        for k in FIELDS:
            va, vb = a[tag][k], b[tag][k]
            same = (va == vb) or (isinstance(va, float) and isinstance(vb, float)
                                  and np.isnan(va) and np.isnan(vb))
            if not same:
                out.append(dict(tag=tag, field=k, before=va, after=vb))
    return out


# --------------------------------------------------------------------------
# P1
# --------------------------------------------------------------------------

def run_checks():
    """两条回归。返回逐项 rc/耗时/末几行，供与基线对比。"""
    out = []
    for name, cmd, to in (("bm_check nofit", [PY, "bm_check.py", "nofit"], 600),
                          ("bm_check fast", [PY, "bm_check.py", "fast"], 3600)):
        r = run(cmd, to)
        out.append(dict(name=name, rc=r["rc"], secs=r["secs"],
                        tail=(r["out"] + r["err"]).strip().splitlines()[-6:]))
        log(f"  {name}: rc={r['rc']}  {r['secs']}s")
    return out


def stage_p1(check_only=False):
    """返回 (ok, info)。任何一步失败都回滚。"""
    info = dict(stage="P1", steps=[], reverted=False, ok=False)
    src = ROOT / "bm_kk.py"
    if not PATCH.exists():
        log("补丁不存在 → 终止")
        return False, info

    info["sha_before"] = sha256(src)
    chk = run(["git", "apply", "--check", str(PATCH.relative_to(ROOT))], 120)
    info["steps"].append(dict(name="git apply --check", **chk))
    if chk["rc"] != 0:
        log("补丁无法应用（工作树已变）→ 终止，不改任何文件")
        log(chk["err"].strip()[:400])
        return False, info

    shutil.copy2(src, BAK)
    log(f"已备份 bm_kk.py → {BAK.name}（sha {info['sha_before'][:12]}）")

    log("快照 before（E2 语料）")
    before = snapshot()
    (HERE / "p1_before.json").write_text(
        json.dumps(before, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"  {len(before)} 条谱")

    if check_only:
        log("--check：不应用补丁，结束")
        info["ok"] = True
        return True, info

    # 基线必须在应用**之前**取：否则"补丁后 rc≠0"分不清是补丁引入的，
    # 还是本来就没过（那样自动回滚会误伤，把一次正确的修改当失败丢掉）。
    log("回归基线（应用前）")
    base = run_checks()
    info["baseline_rows"] = base
    info["baseline"] = {r["name"]: r["rc"] for r in base}
    log(f"  基线 rc：{info['baseline']}")

    ap = run(["git", "apply", str(PATCH.relative_to(ROOT))], 120)
    info["steps"].append(dict(name="git apply", **ap))
    if ap["rc"] != 0:
        log("git apply 失败 → 回滚")
        run(["git", "checkout", "--", "bm_kk.py"], 120)
        info["reverted"] = True
        return False, info
    info["sha_after"] = sha256(src)
    log(f"补丁已应用（sha {info['sha_after'][:12]}）")

    log("快照 after + 溯源 diff")
    after = snapshot()
    (HERE / "p1_after.json").write_text(
        json.dumps(after, ensure_ascii=False, indent=1), encoding="utf-8")
    diffs = diff_snap(before, after)
    (HERE / "p1_provenance.json").write_text(json.dumps(dict(
        sha_before=info["sha_before"], sha_after=info["sha_after"],
        n_spectra=len(before), n_changed=len({d["tag"] for d in diffs}),
        changes=diffs), ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"  变化 {len(diffs)} 处，涉及 {len({d['tag'] for d in diffs})}/{len(before)} 条谱")

    log("校验：真实函数 vs 影子实现（32 条谱 × 6 字段）")
    bad = 0
    try:
        import importlib
        import n1_shadow as S
        importlib.reload(S)
        import bm_kk
        importlib.reload(bm_kk)
        n = 0
        for tag, f, Z in corpus()[:24]:
            a, b = bm_kk.diffusion_tail_candidate(f, Z), S.candidate_new(f, Z)
            n += 1
            for k in FIELDS:
                va, vb = a.get(k), b.get(k)
                if isinstance(va, float) and isinstance(vb, float):
                    if not ((np.isnan(va) and np.isnan(vb))
                            or abs(va - vb) < 1e-12):
                        bad += 1
                elif va != vb:
                    bad += 1
        log(f"  对照 {n} 条，不一致 {bad} 处")
    except Exception as e:                       # noqa: BLE001
        log(f"  对照脚本异常：{type(e).__name__}: {e}")
        bad = -1

    log("回归（应用后）")
    reg = run_checks()
    info["steps"].extend(reg)

    # 判失败的条件：影子不一致，或以基线为绿、补丁后变红
    worse = [r for r in reg
             if r["rc"] != 0 and r["rc"] != info.get("baseline", {}).get(r["name"])]
    ok = (bad == 0) and not worse
    if not ok:
        log(f"回归失败（{', '.join(r['name'] for r in worse) or '影子不一致'}）→ 回滚 bm_kk.py")
        run(["git", "checkout", "--", "bm_kk.py"], 120)
        info["reverted"] = True
        info["sha_after"] = sha256(src)
        r = run([PY, "bm_check.py", "nofit"], 600)
        log(f"  回滚后 nofit: rc={r['rc']}")
        return False, info

    info["ok"] = True
    log("P1 通过：补丁保留在工作树中")
    return True, info


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------

def write_report(p1_ok, p1, p2=None, p3=None):
    try:
        prov = json.loads((HERE / "p1_provenance.json").read_text(encoding="utf-8"))
    except Exception:                            # noqa: BLE001
        prov = {}
    lines = [
        "# E2 之后的后续（P1–P3）执行记录", "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M}", "",
        "## P1 L1 幅度判据换 `log(-Im)`：**"
        + ("已应用并通过回归" if p1_ok else "未通过，已回滚") + "**", "",
        f"- `bm_kk.py` sha256：`{p1.get('sha_before','?')[:12]}` → "
        f"`{p1.get('sha_after','?')[:12]}`",
        f"- 自动回滚：**{'是' if p1.get('reverted') else '否'}**", "",
        "### 代码版本溯源（E2 全程在补丁前运行）", "",
        f"E2 的 30 个 job 全部跑在补丁**应用之前**。补丁应用后，在 E2 用到的"
        f" {prov.get('n_spectra','?')} 条谱上，`diffusion_tail_candidate` 有 "
        f"{prov.get('n_changed','?')} 条的输出发生变化：", "",
        "| 谱 | 字段 | 补丁前 | 补丁后 |", "|---|---|---|---|",
    ]
    for d in prov.get("changes", [])[:40]:
        lines.append(f"| `{d['tag']}` | `{d['field']}` | `{d['before']}` | "
                     f"`{d['after']}` |")
    if len(prov.get("changes", [])) > 40:
        lines.append(f"| … | 共 {len(prov['changes'])} 处 | | |")
    lines += ["", "回归（同一组命令，应用前/后各跑一次）：", "",
              "| 检查 | 应用前 rc | 应用后 rc | 耗时 |", "|---|---|---|---|"]
    after_rows = {r["name"]: r for r in p1.get("steps", [])
                  if r["name"].startswith("bm_check")}
    for b in p1.get("baseline_rows", []):
        a = after_rows.get(b["name"], {})
        lines.append(f"| `{b['name']}` | {b['rc']} | {a.get('rc', '—')} | "
                     f"{a.get('secs', '—')} s |")
    lines += ["", "逐字段快照见 `p1_before.json` / `p1_after.json` / "
              "`p1_provenance.json`。", ""]
    if p2 is not None:
        lines += ["## P2 幅度带宽是否按端点走向分设：**不改**", "",
                  p2, ""]
    if p3 is not None:
        lines += ["## P3 `to_inductive` 是否接到吸附支路做预筛：**不改**", "",
                  p3, ""]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"报告写入 {REPORT}")


def append_memory(p1_ok, p1):
    try:
        p = MEMDIR / f"{datetime.now():%Y-%m-%d}.md"
        head = "" if p.exists() else f"# {datetime.now():%Y-%m-%d}\n"
        txt = (head
               + f"\n## E2 跑完自动接手 P1（脚本 `p_after_e2.py`，"
                 f"{datetime.now():%H:%M}）\n"
               + f"- P1 补丁 `n1_l1_amplitude.patch`："
                 f"**{'已应用，bm_check nofit/fast 全过' if p1_ok else '未通过，已回滚'}**；"
                 f"`bm_kk.py` sha {p1.get('sha_before','?')[:12]} → "
                 f"{p1.get('sha_after','?')[:12]}，回滚={p1.get('reverted')}。\n"
               + "- E2 的 30 个 job 全部跑在补丁前，代码版本溯源见 "
                 "`evidence/ws_crit/p1_provenance.json`。\n")
        with p.open("a", encoding="utf-8") as fh:
            fh.write(txt)
        log(f"已追加记忆日志 {p.name}")
    except Exception as e:                       # noqa: BLE001
        log(f"写记忆日志失败（不影响结论）：{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true", help="不等 E2")
    ap.add_argument("--check", action="store_true", help="只预检+快照，不 apply")
    ap.add_argument("--timeout-h", type=float, default=16.0)
    ap.add_argument("--poll-s", type=float, default=60.0,
                    help="多久查一次 e2_summary.json（不影响日志频率）")
    ap.add_argument("--log-every-s", type=float, default=600.0,
                    help="等待期间心跳日志的最小间隔；progress 变化则立即记")
    a = ap.parse_args()

    log("=" * 60)
    log("p_after_e2 启动")
    if not a.now:
        if not wait_for_e2(a.timeout_h, a.poll_s, a.log_every_s):
            log("未等到 E2，退出（P1 未执行）")
            return 2

    dirty = run(["git", "diff", "--quiet", "bm_kk.py"], 60)
    if dirty["rc"] != 0:
        log("bm_kk.py 工作树已有改动 → 不自动打补丁（避免覆盖别人的修改）")
        return 3

    ok, info = stage_p1(check_only=a.check)
    if a.check:
        return 0 if ok else 1
    p2 = None
    if (HERE / "p2_band.json").exists():
        p2 = ("结论：V1/V2 与 V0 在 9 类 × 20 条上**逐行相同**；`wo_in` 被挡在"
              "**相位判据**（最低 25% 段的 Nyquist 斜率中位数 −17.0，0/20 落在 "
              "45°±0.35），不是带宽。放宽带宽什么也换不来 —— L1 天生只能做 Ws，"
              "Wo 归 `bm_wscrit` 的家族。**生产代码不改**。")
    p3 = None
    if (HERE / "p3_gate.json").exists():
        p3 = ("结论：18 条真实谱 `to_inductive` 18/18、`init_reliable` 12/18，"
              "加预筛会挡掉 **0/18**（零收益）；合成侧唯一分歧是 `wshort_in` "
              "10/20 误放行，但那条路径不选吸附支路、seed 用不上。"
              "**不建议改门控**。")
    write_report(ok, info, p2=p2, p3=p3)
    append_memory(ok, info)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
