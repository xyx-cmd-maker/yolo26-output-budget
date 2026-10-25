"""
★★ K3 的可标定性检验：`c*` 能否**只从 val 估出**、在 **test** 上仍有效？

═══════════════════════════════════════════════════════════════════════════
为什么这是 K3 的生死问题
═══════════════════════════════════════════════════════════════════════════
K3 目前的主张是：「把逐图配额上限从 `B` 换成紧 cap `c*`，可多拿 **10.87–27.53 pt** 召回」。

**但 `c*` 是从真实 `nᵢ` 算出来的**（oracle）：
    `c* = max{c : Σᵢ min(c, nᵢ) ≤ B·N}`

部署时 `nᵢ` **不可得**。所以 K3 与 Bae 批的"oracle 分配"**面临同一个问题**。
→ **必须证明 `c*` 能从可得的信号估出来，且跨数据集有效。**

═══════════════════════════════════════════════════════════════════════════
三条"可得"的估计路径（本脚本逐条检验）
═══════════════════════════════════════════════════════════════════════════
  W1 **从 val 标定，用到 test**：
     在 val 上算 `c*_val`，直接施加到 test → 与 test 上的 `c*_test` 比。
     ⚠️ 这不是严格的无标签标定（val 有标签），但**是最低限度的可迁移性检验**：
     若连这一步都不成立，K3 就是纯 oracle。

  W2 **从少量目标域未标注图估均值**：
     只取目标域 `m` 张图的**真实 nᵢ 均值**，代入"均值式规则" `c_hat = B·N / mean(n)`
     （因为 exp. Σmin(c,n) ≈ N·E[min(c,n)]，而预算 = N·B → 需要 E[min(c,n)] ≈ B）。
     → 检验 m 从 10 到全量时，达到的召回离 oracle 有多远。

  W3 **纯对数假设**：假设目标域 `nᵢ` 服从对数正态（均值由 W2 估出），
     解析算 `c_hat` → 完全不看目标域真实分布的形状。

对每条路径报告：**在 test 上达到的召回 vs 三个参照**（均匀(B) / 均匀(c*_test) 上界 / 均匀(300)）。

用法（纯算术，零 GPU，可在训练期间安全运行）：
    python scripts/k3_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))
from src.eval.proxy_density import ceiling_uniform, uniform_cap  # noqa: E402

RNG = np.random.default_rng(20261014)


def load_counts(split: str) -> np.ndarray:
    cfg = yaml.safe_load((ROOT / "data/processed/visdrone/visdrone.yaml").read_text(encoding="utf-8"))
    d = Path(cfg["path"]) / cfg[split]
    parts = list(d.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return np.array([
        sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
        for f in sorted(Path(*parts).glob("*.txt"))
    ], dtype=float)


def c_star(counts: np.ndarray, B: int) -> int:
    """oracle 紧 cap（需要真实 nᵢ）。"""
    return uniform_cap(counts, B)


def c_from_mean(mean_n: float, B: int) -> int:
    """W2 的'均值式'规则：解 E[min(c,n)] ≈ B for n ~ 对数正态(mu, sigma)。

    没有分布形状时，用最粗的近似：`c ≈ B·(某个与分布有关的因子)`。
    这里给两个变体：
      · `c_mean = B`（朴素：配额就等于平均预算）
      · `c_ratio = B · (mean_n / median_n)`（用均值/中位比校正右偏）
    """
    return int(round(B))


def c_from_lognormal(mean_n: float, sigma: float, B: int, hi: int = 4000) -> int:
    """假设 n ~ LogNormal(mu, sigma)，其中 mu 由 mean_n 与 sigma 定：mu = ln(mean) - sigma²/2。
    解析/数值求 `c` 使 E[min(c, n)] = B。"""
    mu = np.log(max(mean_n, 1e-6)) - sigma**2 / 2
    cs = np.arange(1, hi)
    # E[min(c,n)] 对对数正态的数值积分（用大样本近似）
    samp = RNG.lognormal(mu, sigma, size=200000)
    exp_min = np.array([np.minimum(c, samp).mean() for c in cs])
    idx = int(np.argmin(np.abs(exp_min - B)))
    return int(cs[idx])


def main() -> int:
    val = load_counts("val")
    test = load_counts("test")
    print("=" * 100)
    print("K3 可标定性检验：c* 能否只从可得信号估出并在 test 上有效？")
    print("=" * 100)
    print(f"""
两切分的密度统计（★ 关键：它们的分布**不同**，这正是可迁移性的障碍）：
  val : N={len(val):>5}  总框={int(val.sum()):>6}  均值={val.mean():6.2f}  中位={np.median(val):5.1f}  max={val.max():.0f}  std={val.std():6.2f}
  test: N={len(test):>5}  总框={int(test.sum()):>6}  均值={test.mean():6.2f}  中位={np.median(test):5.1f}  max={test.max():.0f}  std={test.std():6.2f}
  → 均值比 test/val = {test.mean() / val.mean():.3f}；中位比 = {np.median(test) / np.median(val):.3f}
""")

    Nt = len(test)
    TOTt = test.sum()
    print("=" * 100)
    print("参照（在 test 上算，三种「上限」）")
    print("=" * 100)
    print(f"{'B':>5} │ {'均匀(B)':>10} {'均匀(c*_test)':>14} {'均匀(300)':>11} │ 说明")
    refs = {}
    for B in (20, 30, 40, 50, 75, 100):
        v_B = ceiling_uniform(test, B)
        cs_t = c_star(test, B)
        v_cs = ceiling_uniform(test, cs_t)
        v_300 = ceiling_uniform(test, 300)
        refs[B] = (v_B, v_cs, v_300, cs_t)
        print(f"{B:>5} │ {v_B * 100:>9.2f}% {v_cs * 100:>13.2f}% {v_300 * 100:>10.2f}% │ "
              f"c*_test={cs_t}（相对 B 放大 {cs_t / B:.2f}×）")

    # ── W1：从 val 标定，用到 test ─────────────────────────────────────
    print("\n" + "=" * 100)
    print("W1：在 **val** 上算 c*，施加到 **test**")
    print("=" * 100)
    print(f"{'B':>5} {'c*_val':>8} {'c*_test':>8} │ {'test 上达到':>12} {'test 上界':>10} "
          f"{'损失':>8} {'占上界增益':>11}")
    w1 = {}
    for B in (20, 30, 40, 50, 75, 100):
        cs_v = c_star(val, B)          # 从 val 标定
        cs_t = refs[B][3]
        v_use = ceiling_uniform(test, cs_v)   # 用 val 的 c* 施加到 test
        v_cap = refs[B][1]                    # test 上界
        v_base = refs[B][0]
        loss = v_cap - v_use
        gain_cap = v_cap - v_base
        frac = (v_use - v_base) / gain_cap if gain_cap > 1e-9 else float("nan")
        w1[B] = (cs_v, cs_t, v_use, v_cap, loss, frac)
        print(f"{B:>5} {cs_v:>8} {cs_t:>8} │ {v_use * 100:>11.2f}% {v_cap * 100:>9.2f}% "
              f"{loss * 100:>7.2f}pt {frac * 100:>10.1f}%")

    # ── W2：从少量目标域图估均值 ────────────────────────────────────────
    print("\n" + "=" * 100)
    print("W2：只取目标域 m 张图估均值 → 用'均值式'规则定 c → 施加到全 test")
    print("=" * 100)
    print("  规则：c_hat = B · (E[n] / median(n))（用右偏校正），median 也要估 → 用同 m 张图")
    print(f"\n{'B':>5} {'m':>5} │ {'c_hat':>7} {'test 达到':>10} {'上界':>9} {'占上界增益':>11} {'c_hat 的标准差':>14}")
    for B in (30, 50, 100):
        cs_t = refs[B][3]
        v_cap = refs[B][1]
        v_base = refs[B][0]
        gain_cap = v_cap - v_base
        for m in (10, 20, 50, 100, 300):
            est = []
            for rep in range(200):
                idx = RNG.choice(Nt, size=min(m, Nt), replace=False)
                sub = test[idx]
                ratio = sub.mean() / max(np.median(sub), 1e-6)
                est.append(int(round(B * ratio)))
            c_hat = int(np.median(est))
            v_use = ceiling_uniform(test, c_hat)
            frac = (v_use - v_base) / gain_cap if gain_cap > 1e-9 else float("nan")
            print(f"{B:>5} {m:>5} │ {c_hat:>7} {v_use * 100:>9.2f}% {v_cap * 100:>8.2f}% "
                  f"{frac * 100:>10.1f}% {np.std(est):>14.1f}")

    # ── W3：完全假设对数正态 ──────────────────────────────────────────
    print("\n" + "=" * 100)
    print("W3：假设目标域 nᵢ ~ LogNormal（只估均值与 std，不看真实形状）")
    print("=" * 100)
    print(f"{'B':>5} {'σ 来源':>12} {'c_hat':>7} │ {'test 达到':>10} {'上界':>9} {'占上界增益':>11}")
    for B in (30, 50, 100):
        cs_t = refs[B][3]
        v_cap = refs[B][1]
        v_base = refs[B][0]
        gain_cap = v_cap - v_base
        # 用 val 的均值与 std 当假设（最诚实的"不知道目标域"情形）
        for label, (mn, sd) in (("val 的 mean/std", (val.mean(), val.std())),
                                ("test 的 mean/std", (test.mean(), test.std()))):
            sigma = float(np.sqrt(np.log(1 + (sd / mn) ** 2)))  # 对数正态的形状参数
            c_hat = c_from_lognormal(mn, sigma, B)
            v_use = ceiling_uniform(test, c_hat)
            frac = (v_use - v_base) / gain_cap if gain_cap > 1e-9 else float("nan")
            print(f"{B:>5} {label:>12} {c_hat:>7} │ {v_use * 100:>9.2f}% {v_cap * 100:>8.2f}% {frac * 100:>10.1f}%")

    print("\n" + "=" * 100)
    print("★ 判读")
    print("=" * 100)
    print("  · **W1** 若『占上界增益』接近 100% → 从 val 标定的 c* 在 test 上几乎无损，")
    print("    K3 具备**最低限度的可迁移性**（不是纯 oracle）。")
    print("  · **W2/W3** 若在 m≤50 时就能拿到 >80% 的上界增益 → 部署时**只需少量目标域图像**，")
    print("    K3 是**真正可实现**的方法，而不是又一个 oracle 上界。")
    print("  · 若 W1/W2/W3 都远低于 100% → **K3 与 Bae 批的 oracle 分配同病**，必须降级。")
    print()
    print("  ⚠️ 本脚本全部是**确定性算术**（假设排序完美），衡量的是**容量上界**；")
    print("     真实检测器的排序质量会更低 —— 但那影响的是「实现率」，不影响本处的可标定性结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
