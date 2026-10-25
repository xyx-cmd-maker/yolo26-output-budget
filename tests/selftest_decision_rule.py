"""
tests/selftest_decision_rule.py
===============================
判据 v2 实施脚本（`scripts/run_decision_rule.py`）的**判别力自测**。

═══════════════════════════════════════════════════════════════════════════
为什么必须有它（P16：测试的价值 = 它能否失败）
═══════════════════════════════════════════════════════════════════════════
判据脚本的价值全在**能不能给出正确的 A/B/C 判定**。若它无论输入什么都报同一个结论，
那它就是死的。本自测构造两类**已知答案**的合成世界：

  world "flat"   —— 所有图完全一样（nᵢ 相等、tp_prefix 相同）
                    → 均匀与水位**同分配** → `Δworst_abs ≈ 0` → 应判 **C**
  world "skewed" —— 图分两族：稀疏族（n 小、检出率高）与密集族（n 大、检出率低）
                    且**最坏图是密集族** → 水位法把配额按 nᵢ 比例倾斜 → 应判 **A**

若两个世界报出**同一个判定** → 脚本无判别力 → 自测失败。

═══════════════════════════════════════════════════════════════════════════
★ 同时验证一条已证的单调性
═══════════════════════════════════════════════════════════════════════════
`scripts/check_plan_distinctness.py` 已证：容量口径下**水位法的 worst 单调 ≥ 均匀**。
本自测在合成世界上再查一次：**`Δworst_abs` 必须 ≥ −1e−9**（水位不可能更差）。
若出现负值 → 实现有 bug。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable if Path(sys.executable).name.startswith("python") else r"G:\conda-envs\yolo26train\python.exe"
SCRIPT = PROJECT_ROOT / "scripts" / "run_decision_rule.py"


def make_prefix(frac: float, p: int) -> list[int]:
    """构造 tp_prefix：p 个预测，总 TP = round(frac*p)，均匀分布在前缀上。

    用"累积后取整"保证单调不减且末值 = round(frac*p)。
    """
    total = int(round(frac * p))
    out, acc = [], 0.0
    for k in range(1, p + 1):
        acc += total / p
        out.append(int(round(acc)))
    # 修正末值
    if out and out[-1] != total:
        out = [min(v, total) for v in out]
        out[-1] = total
    return out


def load_real_n(split: str = "val") -> np.ndarray:
    """真实 VisDrone 逐图 GT 数（**用真实分布造世界**，比手编数字可靠）。"""
    import yaml
    cfg = yaml.safe_load((PROJECT_ROOT / "data/processed/visdrone/visdrone.yaml")
                         .read_text(encoding="utf-8"))
    d = Path(cfg["path"]) / cfg[split]
    parts = list(d.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return np.array([sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
                     for f in sorted(Path(*parts).glob("*.txt"))], dtype=float)


def build_world(name: str) -> dict:
    """构造合成世界。

    ★★ 第 36 轮的两次失败（务必保留，防回退）：
      ① 初版把**密集图的检出率设得与稀疏图接近** → 均匀下最坏图是稀疏图、水位下换成密集图
         → 结论含混，三个世界全判 C。
      ② 第二版把**稀疏图检出率设得更高**，但仍失败——因为我的世界**倾斜度不够**：
         `mean/max = 105/200 = 0.525`，而真实 VisDrone 是 `70.7/317 = 0.223`（val）、
         `46.7/461 = 0.101`（test）。倾斜度不足 → 水位的增益空间被压小。
      → **最终做法：直接用真实 VisDrone 的逐图 GT 数**，只在其上叠加一族检出率模型。
        这样世界的倾斜度与现实一致，判据的可达范围也由
        `scripts/calibrate_decision_threshold.py` 独立算出（val 上确界 37.3 pt、test 44.1 pt）。
    """
    if name == "homogeneous":
        # ★ 应判 **C**：**所有图 nᵢ 相等且检出率相等** → 均匀与水位**逐图同分配**
        #   ⚠️ 第 36 轮第三次踩坑：只让"检出率相等"是**不够的**——只要 `nᵢ` 不均匀，
        #      最坏图就是最密的图，其召回 ≈ hit·qᵢ/nᵢ **依赖分配**，Δ 仍会很大（实测 +23.6 pt）。
        #      **真正的分配无关对照世界必须 nᵢ 全等。**
        recs = []
        for i in range(200):
            ni = 60
            total_tp = int(round(0.75 * ni))
            pref, acc = [], 0.0
            for k in range(1, ni + 1):
                acc += total_tp / ni
                pref.append(min(int(round(acc)), total_tp))
            recs.append({"name": f"h{i}.jpg", "n_gt": ni, "n_pred_at_run": ni,
                         "tp_prefix": pref})
        return {"per_image": recs}

    n = load_real_n("val")
    recs = []

    if name == "uniform_hit":
        # 真实 nᵢ 分布 + **均匀检出率**。注意它**不是**分配无关的对照
        # （因为 nᵢ 不均匀）——故对它的断言只是"Δ 应为正"，不是"Δ≈0"。
        hit = np.full(len(n), 0.75)
    elif name == "density_decay":
        # 检出率随密度下降（检测器的一般规律）
        hit = np.clip(0.95 - 0.25 * np.log10(np.maximum(n, 1.0)), 0.0, 1.0)
    elif name == "extreme_decay":
        # 更强的密度依赖 → 应给出更大的 Δ（有序性断言用）
        hit = np.clip(1.00 - 0.40 * np.log10(np.maximum(n, 1.0)), 0.0, 1.0)
    else:
        raise ValueError(name)

    for ni, hi in zip(n, hit):
        ni = int(ni)
        if ni <= 0:
            continue
        p = ni  # 假设检测器能输出足够多的候选（pᵢ = nᵢ）
        total_tp = int(round(hi * p))
        # tp_prefix 单调，末值 = total_tp（在配额 = nᵢ 时达到 hit·nᵢ）
        pref, acc = [], 0.0
        for k in range(1, p + 1):
            acc += total_tp / p
            pref.append(min(int(round(acc)), total_tp))
        recs.append({"name": f"i{len(recs)}.jpg", "n_gt": ni, "n_pred_at_run": p,
                     "tp_prefix": pref})
    return {"per_image": recs}


def run_world(name: str) -> tuple[str, float, str]:
    """把合成世界写成 budget_scan 的 JSON 格式，跑判据脚本，返回 (stdout, exit, verdict line)。"""
    tmp = Path(tempfile.mkdtemp(prefix="drule_"))
    d = tmp / "runs" / "budget"
    d.mkdir(parents=True)
    (d / "syn_val_o2m.json").write_text(json.dumps(build_world(name)), encoding="utf-8")

    # 判据脚本硬编码读 PROJECT_ROOT/runs/budget —— 用 --exp 指向合成文件会找不到，
    # 故这里**直接 import 它的函数**跑，避免污染项目目录。
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("rdr", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    recs = mod.load_recs(d / "syn_val_o2m.json")
    res = mod.analyse(recs, 0.5, f"syn:{name}")
    return name, res["dworst"], res["verdict"]


def main() -> int:
    print("=" * 92)
    print("判据 v2 判别力自测（世界基于**真实 VisDrone 标签分布**）")
    print("=" * 92)
    out = {}
    for w in ("homogeneous", "uniform_hit", "density_decay", "extreme_decay"):
        name, dw, verdict = run_world(w)
        out[name] = (dw, verdict)
        print(f"\n  → world `{name}`：Δworst_abs = {dw:+.4f} pt ，判定 = {verdict}")

    print("\n" + "=" * 92)
    print("★ 断言")
    print("=" * 92)
    fails = []

    # ① 单调性：Δworst_abs 不得为负（水位法 worst 单调 ≥ 均匀——但**仅在整数水位法**下；
    #   第 36 轮实测：连续松弛会给出负数，那是实现的错，不是方法的错）
    neg = [k for k, (dw, _) in out.items() if dw < -1e-9]
    print(f"  ① 单调性（Δworst_abs ≥ 0，整数水位法）：{'✅' if not neg else '❌ ' + str(neg)}")
    for k in neg:
        fails.append(f"world {k}：Δworst_abs = {out[k][0]:+.4f} 为**负**，违反已证单调性")

    # ② homogeneous 世界（nᵢ 全等 + 检出率全等）→ 应判 C（分配无关）
    ok_c = out["homogeneous"][0] < 5.0
    print(f"  ② world `homogeneous` 判 C（Δ < 5 pt）：{'✅' if ok_c else '❌'}  "
          f"(实际 {out['homogeneous'][0]:+.4f} pt)")
    if not ok_c:
        fails.append(f"homogeneous 的 Δ={out['homogeneous'][0]:+.4f} 应 <5（nᵢ 与检出率全等 → 分配无关）")

    # ③ 密度衰减明显 → 应判 A
    ok_a = out["density_decay"][0] >= 10.0
    print(f"  ③ world `density_decay` 判 A（Δ ≥ 10 pt）：{'✅' if ok_a else '❌'}  "
          f"(实际 {out['density_decay'][0]:+.4f} pt)")
    if not ok_a:
        fails.append(f"density_decay 的 Δ={out['density_decay'][0]:+.4f} 应 ≥10")

    # ④ ★ 判别力：不得所有世界同一判定
    verdicts = {v[1] for v in out.values()}
    distinct = len(verdicts) >= 2
    print(f"  ④ 判别力（≥2 种判定）：{'✅' if distinct else '❌'}  {sorted(verdicts)}")
    if not distinct:
        fails.append("所有世界报了同一判定 → 脚本无判别力")

    # ⑤ ★ 饱和效应：`extreme_decay` 把最密图的检出率压到 0 → worst = 0 对**两种分配都成立**
    #    → Δ 必然为 0，且应判 C。
    #    这不是缺陷，而是**有意义的行为**：水位法无法改善"根本检不出"的图。
    #    （第 36 轮：`1.00 − 0.40·log10(317) = 0` 正好归零。）
    de = out["extreme_decay"][0]
    ok_sat = de < 5.0
    print(f"  ⑤ 饱和效应（extreme_decay 的 Δ < 5 pt，因最密图检出率归零）："
          f"{'✅' if ok_sat else '❌'}  (实际 {de:+.4f} pt)")
    if not ok_sat:
        fails.append(f"extreme_decay 的 Δ={de:+.4f} 应 <5（最密图检出率归零 → 分配无从改善）")

    print("\n" + "=" * 92)
    if fails:
        print(f"❌ 自测失败（{len(fails)} 条）：")
        for f in fails:
            print(f"   · {f}")
        return 1
    print("✅ 全部 5 项断言通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
