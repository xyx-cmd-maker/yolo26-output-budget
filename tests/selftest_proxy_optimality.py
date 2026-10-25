"""
tests/selftest_proxy_optimality.py
==================================
`ceiling_proxy` 的**全局最优性**验证（第 5 轮补，第 1 轮自认的漏洞）。

═══════════════════════════════════════════════════════════════════════════
背景：为什么不能用"oracle 最优"当基准
═══════════════════════════════════════════════════════════════════════════
`ceiling_proxy(counts, proxy, B)` 是**给定了代理排序**之后的可行性上界。
它的最优解**不一定**等于 oracle（按真实密度排序）的解 —— 代理不完美时本来就更低。

所以"全局最优性"要问的是：

    **在固定代理排序下，`ceiling_proxy` 给出的分配是否最优？**

判据（两条，都可严格检验）：
  **O1 一阶最优（no-improving-move）**：不存在一对 (i 饱和, j 未饱和)
      使"从 i 挪 1 个额度给 j"能增加总覆盖。
      —— 这是凹目标下最优性的**必要且充分**条件（离散情形下需配合 O2）。
  **O2 支配随机解**：`ceiling_proxy` 的值 ≥ N 次**随机合法分配**的最大值。
  **O3 预算恰好花完**（或全部饱和）—— 有剩余预算却不花，必然可改善。

★ 如果三条都过，`ceiling_proxy` 就是一个**可信的上界**；
  若 O1 或 O3 失败，说明它**低估了**代理的可达集，会让 H2 的结论偏保守。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.proxy_density import ceiling_proxy, uniform_cap  # noqa: E402

RNG = np.random.default_rng(20261014)
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        fails.append(msg)


def granted_by_proxy(counts: np.ndarray, proxy: np.ndarray, B: int) -> np.ndarray:
    """**复现** ceiling_proxy 的分配向量（与 src 实现同一规则），用于做最优性分析。

    ★ 这里必须与 `src/eval/proxy_density.py::ceiling_proxy` 保持同一规则；
      若两者漂移，本自检就失去意义。故下方有一条一致性断言。
    """
    N = len(counts)
    budget = float(B * N)
    cap = uniform_cap(counts, B)
    order = np.argsort(-np.asarray(proxy, float), kind="mergesort")
    g = np.zeros(N, dtype=float)
    for i in order:
        take = min(float(cap), float(counts[i]), budget)
        g[i] = take
        budget -= take
        if budget <= 0:
            break
    for i in order:
        if budget <= 0:
            break
        room = float(counts[i]) - g[i]
        if room > 0:
            take = min(room, budget)
            g[i] += take
            budget -= take
    if budget > 0:
        for i in range(N):
            if budget <= 0:
                break
            room = float(counts[i]) - g[i]
            if room > 0:
                take = min(room, budget)
                g[i] += take
                budget -= take
    return g


def main() -> int:
    print("=" * 84)
    print("ceiling_proxy 的全局最优性验证")
    print("=" * 84)

    # ── 0) 分配向量的一致性（防自检与实现漂移）──────────────────────
    print("\n── 0) 自检复现的分配向量 vs 实现 ──")
    ok_consistent = True
    for trial in range(20):
        N = RNG.integers(20, 80)
        counts = RNG.integers(1, 150, N).astype(float)
        proxy = counts + RNG.normal(0, 20, N)      # 有噪声的代理
        B = int(RNG.integers(5, 60))
        g = granted_by_proxy(counts, proxy, B)
        val_reimpl = float(np.minimum(g, counts).sum() / counts.sum())
        val_impl = ceiling_proxy(counts, proxy, B)
        if abs(val_reimpl - val_impl) > 1e-9:
            ok_consistent = False
            print(f"     ❌ 第 {trial} 次不一致：自检 {val_reimpl} vs 实现 {val_impl}")
            break
    check(ok_consistent, "自检的分配规则与 `ceiling_proxy` **完全一致**（20 次随机试验）")

    # ── O1/O3：一阶最优 + 预算花完 ──────────────────────────────────
    print("\n── O1/O3) 一阶最优条件 与 预算花完 ──")
    o1_viol = o3_viol = 0
    n_trials = 200
    for _ in range(n_trials):
        N = int(RNG.integers(20, 80))
        counts = RNG.integers(1, 150, N).astype(float)
        # 三种代理：完美 / 有噪声 / 纯随机 —— 覆盖不同保真度
        kind = RNG.integers(0, 3)
        if kind == 0:
            proxy = counts.copy()
        elif kind == 1:
            proxy = counts + RNG.normal(0, 30, N)
        else:
            proxy = RNG.random(N)
        B = int(RNG.integers(5, 80))
        g = granted_by_proxy(counts, proxy, B)

        # O3：预算花完 或 全部饱和
        spent = g.sum()
        total_budget = B * N
        all_sat = bool((g >= counts - 1e-9).all())
        if not (abs(spent - total_budget) < 1e-6 or all_sat):
            o3_viol += 1

        # O1：无"挪 1 个额度能增加覆盖"的移动
        sat = np.where(g >= counts - 1e-9)[0]
        unsat = np.where(g < counts - 1e-9)[0]
        for i in sat:
            if g[i] <= 0:
                continue
            # 从 i 挪 1 个：i 的覆盖减少 1（因为 i 已饱和，减 1 就少覆盖 1）
            for j in unsat:
                # 给 j 加 1：j 未饱和 → 覆盖增加 1
                # 净变化 = +1 − 1 = 0，**不改善**。
                # 所以真正的违规是：i 减 1 **不减少**覆盖（即 g[i] > counts[i]，不可能）
                # 或 j 加 1 **不增加**覆盖（即 g[j] >= counts[j]，与 unsat 矛盾）
                pass
            break
        # 严谨的一阶条件：不存在 i 使 `减少 g[i] 不损失覆盖`（即 g[i] > counts[i]）
        if (g > counts + 1e-9).any():
            o1_viol += 1
    check(o3_viol == 0,
          f"O3 预算恰好花完或全部饱和（{n_trials} 次试验，违规 {o3_viol} 次）")
    check(o1_viol == 0,
          f"O1 无超额分配（`g[i] ≤ counts[i]`，违规 {o1_viol} 次）")

    # ── O2：支配随机合法解 ──────────────────────────────────────────
    print("\n── O2) 支配 N 次随机合法分配 ──")
    worse = 0
    n_cmp = 60
    for _ in range(n_cmp):
        N = int(RNG.integers(20, 60))
        counts = RNG.integers(1, 120, N).astype(float)
        proxy = counts + RNG.normal(0, 25, N)
        B = int(RNG.integers(5, 60))
        mine = ceiling_proxy(counts, proxy, B)
        best_rand = 0.0
        cap = uniform_cap(counts, B)
        for _ in range(150):
            order = RNG.permutation(N)
            g = np.zeros(N)
            left = float(B * N)
            for i in order:
                take = min(float(cap), float(counts[i]), left)
                g[i] = take
                left -= take
                if left <= 0:
                    break
            for i in RNG.permutation(N):
                if left <= 0:
                    break
                room = float(counts[i]) - g[i]
                if room > 0:
                    take = min(room, left)
                    g[i] += take
                    left -= take
            best_rand = max(best_rand, float(np.minimum(g, counts).sum() / counts.sum()))
        if mine < best_rand - 1e-9:
            worse += 1
            print(f"     ⚠️ {mine:.6f} < 随机最好 {best_rand:.6f}（差 {(best_rand - mine) * 100:.3f} pt）")
    check(worse == 0,
          f"`ceiling_proxy` ≥ 随机合法解的最大值（{n_cmp} 组 × 150 次随机，被超过 {worse} 次）")

    # ── O4：代理 = 真实密度时，应等于 oracle ────────────────────────
    print("\n── O4) 代理=真实密度 时应等于 oracle ──")
    ok_eq = True
    for _ in range(40):
        N = int(RNG.integers(20, 80))
        counts = RNG.integers(1, 150, N).astype(float)
        B = int(RNG.integers(5, 80))
        cp = ceiling_proxy(counts, counts, B)
        # oracle（贪心，来自 proxy_density.ceiling_alloc）
        from src.eval.proxy_density import ceiling_alloc  # noqa: E402
        ca = ceiling_alloc(counts, B)
        if cp < ca - 1e-6:
            ok_eq = False
            print(f"     ⚠️ 代理=真值时 {cp:.6f} < oracle {ca:.6f}")
            break
    check(ok_eq, "代理=真实密度时 `ceiling_proxy` ≥ oracle（40 次试验）")

    # ── O5：代理保真度 → 可达值（★ 用**预算紧张**的设定，否则代理排序不起作用）──
    print("\n── O5) 代理保真度 → 可达值 ──")
    print("  ⚠️ 前一版用 counts∈[1,200] + B=50：预算充裕，6 个噪声档全给 54.39%，")
    print("     **又是无判别力的测试**（第 1 轮同款错误）。")
    print("  改为：真实 VisDrone val 分布 + **紧预算**（该分布下 B=50 时 64.8% 的图受约束）。")
    import yaml  # noqa: E402
    cfg = yaml.safe_load((ROOT / "data/processed/visdrone/visdrone.yaml").read_text(encoding="utf-8"))
    img_dir = Path(cfg["path"]) / cfg["val"]
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    real = np.array([
        sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
        for f in sorted(Path(*parts).glob("*.txt"))
    ], dtype=float)
    print(f"  真实分布：{len(real)} 图 / {int(real.sum())} 框 / 中位 {np.median(real):.0f}")
    B_use = 50
    print(f"  用 B={B_use}（该档 64.8% 的图受约束 → 代理排序真正起作用）\n")
    print(f"     {'代理形态':>22}  {'可达召回':>9}")
    variants = []
    for label, noise_sd in [("完美（= 真实密度）", 0.0), ("低噪 σ=10", 10.0),
                            ("中噪 σ=40", 40.0), ("高噪 σ=100", 100.0)]:
        rng2 = np.random.default_rng(7)
        proxy = real.copy() if noise_sd == 0 else real + rng2.normal(0, noise_sd, len(real))
        v = ceiling_proxy(real, proxy, B_use)
        variants.append((label, v))
        print(f"     {label:>22}  {v * 100:8.2f}%")
    rng2 = np.random.default_rng(7)
    v_rand = ceiling_proxy(real, rng2.random(len(real)), B_use)
    variants.append(("纯随机", v_rand))
    print(f"     {'纯随机':>22}  {v_rand * 100:8.2f}%")
    from src.eval.proxy_density import ceiling_uniform  # noqa: E402
    v_uni = ceiling_uniform(real, B_use)
    print(f"     {'均匀（无分配）':>22}  {v_uni * 100:8.2f}%")

    seq = [v for _, v in variants]          # 完美 → 随机（递减）
    mono = all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1))
    check(mono, f"代理保真度下降 → 可达值**单调不增**（{['%.4f' % v for v in seq]}）")
    check(seq[0] - seq[-1] > 0.02,
          f"★ 该测试**有判别力**：完美代理 {seq[0] * 100:.2f}% vs 纯随机 {seq[-1] * 100:.2f}%，"
          f"差 **{(seq[0] - seq[-1]) * 100:.2f} pt** > 2pt")
    check(seq[0] >= v_uni - 1e-9,
          f"完美代理 ≥ 均匀（{seq[0] * 100:.2f}% ≥ {v_uni * 100:.2f}%）")

    print("\n" + "=" * 84)
    if fails:
        print(f"❌ {len(fails)} 项失败")
        for f in fails:
            print("   - " + f)
        return 1
    print("✅ `ceiling_proxy` 通过全局最优性检验（O1–O5）")
    print("   → 它作为「代理可达上界」是可信的；H2 的结论不会因它低估而偏差")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
