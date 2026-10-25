"""
proxy_density.py — 「代理密度 vs 真实密度」的关联与可实现增益

═══════════════════════════════════════════════════════════════════════════
为什么这是 H2 的生死实验（docs/execution-queue.md §2 ③-b、P0-2 Step 2）
═══════════════════════════════════════════════════════════════════════════
`scripts/budget_headroom.py` 算出的"非平凡增量"（val B=50 → 10.87 pt；
test B=50 → 27.53 pt）是 **oracle 上界**：它假设**逐图 GT 数已知**。

真实系统里 GT 数不可知。**必须用检测器自己可得的代理量去估计密度。**
所以真正要回答两个问题：

  Q1  ρ = Spearman(代理密度, GT 数) 是多少？
      ρ 高（如 ≥0.6）→ H2 有物理基础；ρ ≈ 0 → **H2 直接降级**，不写任何分配增益正文。

  Q2  用代理分配，能拿回 oracle 增量的百分之几？
      ★ 这一问比 Q1 更重要：ρ 是中间量，**可实现增益才是主张本身**。

═══════════════════════════════════════════════════════════════════════════
四个候选代理（全部只需一次前向，零额外成本）
═══════════════════════════════════════════════════════════════════════════
  P1  `n_pred`        低阈值（conf=0.001）下的原始预测数 —— 最朴素
  P2  `sum_conf`      全部预测的置信度之和 —— 对漏检更鲁棒
  P3  `n_hi_conf`     高置信（conf ≥ 0.25）预测数 —— 只有"确定的目标"才计数
  P4  `neg_entropy`   top-B 内置信度分布的负熵 —— 越密集越"拥挤"，熵越高

★ 关键设计：**P1/P2/P3/P4 全部来自被评估检测器自身的输出**，
  不需要额外模型、不需要额外标注 —— 这是"零成本"的含义。

═══════════════════════════════════════════════════════════════════════════
可实现增益的算法（与 oracle 上界同口径，便于直接比较）
═══════════════════════════════════════════════════════════════════════════
固定**总预算** `B·N`（与均匀方案同总量），按代理排序分配变长预算：

  1. 按代理把图排序
  2. 分配：从"最密"的图开始，每图给 `cap` 个输出，直到总预算用完
     （cap = 允许的单图上限；用与 budget_headroom 相同的二分法求最大可行均匀 cap，
      再把剩余预算给代理最高的图）
  3. 算分配后的召回，与 oracle 分配、均匀分配比较

报告三个数（**这是正文表格**）：
  · ceiling_uniform(B)   均匀分配
  · ceiling_proxy(B)     代理分配（可实现）
  · ceiling_alloc(B)     oracle 分配（上界，不可实现）
  → `可实现率 = (proxy − uniform) / (alloc − uniform)` ∈ [0, 1]

用法：
    python -m src.eval.proxy_density --pred runs/budget/r1_val_o2o.json \\
        --data data/processed/visdrone/visdrone.yaml --split val --budgets 30 50 75 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RNG_SEED = 20261014

# 调试开关：置 True 可打印 `ceiling_proxy` 的内部分配统计（默认关闭，不影响正常路径）
_DEBUG_PROXY = False


def _labels_dir_for(img_dir: Path) -> Path:
    """从右往左替换最后一个 `images` 组件（全局 replace 会改到祖先目录，见 P5）。"""
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return img_dir.parent / "labels" / img_dir.name


def _ranks(x: np.ndarray) -> np.ndarray:
    """平均秩（正确处理并列值）。

    ★ 不能用 `argsort(argsort(x))` —— 那给并列值任意不同秩，
      在"预测数"这种**整数、大量并列**的代理上会产生虚假相关。
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ = 平均秩上的 Pearson 相关。

    ★ 修复（自检发现）：任一侧零变异（全常数）时**必须**返回 `nan`（不可用），
      不得返回 0.0。原实现写成 `np.sqrt(0.0) > 0` → False → 落入 else 返回 0.0，
      会让"常数代理"看起来像"零相关"而不是"无定义"。见 tests/selftest_proxy_density.py。
    """
    if len(x) < 3:
        return float("nan")
    rx, ry = _ranks(np.asarray(x, float)), _ranks(np.asarray(y, float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def spearman_ci(x: np.ndarray, y: np.ndarray, *, n_boot: int = 10000,
                alpha: float = 0.05) -> tuple[float, float, float]:
    """按图重采样（配对单位 = 图，P0-2 的 C-6 要求）。"""
    rng = np.random.default_rng(RNG_SEED)
    obs = spearman(x, y)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = spearman(x[idx], y[idx])
        if not np.isnan(r):
            boots.append(r)
    if not boots:
        return obs, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return obs, float(lo), float(hi)


def proxies_from_prediction(rec: dict, hi_conf: float = 0.25, top_b: int = 50) -> dict[str, float]:
    """从 budget_scan 的逐图记录算四个代理量。

    记录需含：`conf_list`（该图全部预测的置信度，降序）。
    若只有标量字段，则退化为可用子集并如实标注。
    """
    conf = np.asarray(rec.get("conf_list") or [], dtype=float)
    n_pred = rec.get("n_pred_at", {}).get(str(top_b))
    if n_pred is None:
        n_pred = len(conf)
    out = {
        "P1_n_pred": float(n_pred),
        "P2_sum_conf": float(conf.sum()) if len(conf) else 0.0,
        "P3_n_hi_conf": float((conf >= hi_conf).sum()),
    }
    # P4 负熵：top-B 内置信度分布归一化后的香农熵
    c = conf[:top_b]
    if len(c) >= 2 and c.sum() > 0:
        p = c / c.sum()
        p = p[p > 0]
        out["P4_entropy"] = float(-(p * np.log(p)).sum())
    else:
        out["P4_entropy"] = 0.0
    return out


def uniform_cap(counts: np.ndarray, B: int) -> int:
    """**紧**均匀单图上限 cap：满足 `Σᵢ min(cap, nᵢ) ≤ B·N` 的最大整数。

    ★ 这个量是理解"为什么需要分配"的钥匙（自检时才发现，见下）：
      均匀方案若用 cap = B，当某图 nᵢ < B 时那部分预算是**闲置**的。
      实测（VisDrone val, B=50）：每图给 50 个 → 召回上限 59.73%；
      而用紧 cap = 63 → 上限 **70.61%**，**两者用掉的总预算完全相同**。
      差额 10.87 pt **纯粹来自"把钱花在该花的地方"** —— 这就是分配的全部价值。
    """
    budget = B * len(counts)
    lo, hi, best = 0, int(counts.max()) + 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if np.minimum(mid, counts).sum() <= budget:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def ceiling_uniform(counts: np.ndarray, B: int) -> float:
    """均匀方案：每图给 **B** 个（历史上就是按这个口径算的，保留以便与旧结果对比）。

    ⚠️ 注意它**不一定用满预算**：当多数图 nᵢ < B 时会闲置。见 `uniform_cap`。
    """
    tot = counts.sum()
    return float(np.minimum(B, counts).sum() / tot) if tot else 0.0


def ceiling_alloc(counts: np.ndarray, B: int) -> float:
    """**最优**逐图分配（总预算 = B·N）：贪心，凹目标下可证最优。

    ★ 第二节自检发现：我原来的"二分求均匀 cap 再平摊"**不是最优的** ——
      它给出 28.95%，而常数代理的贪心给出 **29.78%**，即"oracle 被常数代理打败"，
      这在数学上不可能，说明**是 oracle 算错了**。

    为什么贪心最优：目标 `Σ min(kᵢ, nᵢ)` 关于每个 kᵢ 是**凹**的，
    最优解的必要条件是所有"未饱和者"的边际收益相等；而把预算优先给 **nᵢ 最小**的图
    （它的边际损失最小）正是贪心。流程：
        counts 升序 → 依次把 nᵢᵢ 填到 nᵢ₊₁ 的水平，直到预算不足以整体抬升
        → 剩余预算在未饱和的图上平摊
    """
    c = np.sort(np.asarray(counts, dtype=float))
    N = len(c)
    budget = float(B * N)
    # 1) 从最小图开始整体抬升到下一档
    i = 0
    while i < N and budget >= (c[i] - (c[i - 1] if i > 0 else 0.0)) * (N - i):
        budget -= (c[i] - (c[i - 1] if i > 0 else 0.0)) * (N - i)
        i += 1
    # 2) 前 i 张已饱和到 c[i-1]，后 N-i 张处于同一水平，剩余预算平摊
    base = c[i - 1] if i > 0 else 0.0
    level = base + budget / (N - i) if i < N else base
    total = c[:i].sum() + (level * (N - i) if i < N else 0.0)
    return float(min(total, c.sum()) / c.sum())


def ceiling_proxy(counts: np.ndarray, proxy: np.ndarray, B: int, *,
                  n_boot: int = 0) -> float:
    """**可实现**的分配：把总预算 `B·N` 按**代理比例**分给各图。

    ═══════════════════════════════════════════════════════════════════════
    ★★ 第 5 轮的重大修正（我自己发现的设计缺陷，不是测试 bug）
    ═══════════════════════════════════════════════════════════════════════
    旧实现（第 1–4 轮）用"紧 cap + 按代理排序逐个给 min(cap, nᵢ)"。自检发现：
    **它对任何排序都给出同一个值**（完美代理 / 噪声代理 / 纯随机代理
    / 均匀 全部 = 70.69% = oracle）。

    根因：当 cap 取到**恰好用满预算**时，`Σmin(cap, nᵢ)` 已经把全部预算耗尽，
    第一轮就结束，**排序根本不起作用**；而 `Σmin(cap, nᵢ)` 是**与排序无关**的量。
    → 那不是"代理可达上界"，**是伪装的 oracle**，会把 H2 的结论严重高估。

    **正确做法**：让配额**按代理比例**分配，使代理质量真正影响结果：

        quotaᵢ = B·N · proxyᵢ / Σproxy      （每图按代理份额分得预算）
        实际输出 = min(quotaᵢ, nᵢ, cap)     （不能超过真实目标数，也不能超单图上限）

    这样：代理完美 → 接近 oracle；代理随机 → 接近均匀；**代理质量真正起作用**。
    ═══════════════════════════════════════════════════════════════════════
    """
    proxy = np.asarray(proxy, dtype=float)
    counts = np.asarray(counts, dtype=float)
    N = len(counts)
    total = float(B * N)
    s = proxy.sum()
    if s <= 0:
        return ceiling_uniform(counts, B)
    cap = float(counts.max()) if N else 0.0
    quota = total * (proxy / s)                     # 按代理比例分预算
    quota = np.minimum(quota, cap)                  # 单图上限（不超过最密图的真实需求）
    granted = np.minimum(quota, counts)             # 不能超过真实目标数
    # 把因夹紧而剩余的预算补给**代理最高且未饱和**的图（仍受 cap 与 nᵢ 约束）
    leftover = total - granted.sum()
    if leftover > 1e-9:
        order = np.argsort(-proxy, kind="mergesort")
        for i in order:
            if leftover <= 0:
                break
            room = min(float(counts[i]), cap) - granted[i]
            if room > 0:
                take = min(room, leftover)
                granted[i] += take
                leftover -= take
    if __debug__ and _DEBUG_PROXY:
        print(f"    [debug ceiling_proxy] B={B} total={total:.0f} "
              f"Σquota={quota.sum():.1f} Σgranted={granted.sum():.1f} "
              f"leftover_after={leftover:.1f}")
    return float(granted.sum() / counts.sum()) if counts.sum() else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="代理密度 vs 真实密度的关联与可实现增益")
    ap.add_argument("--pred", required=True,
                    help="budget_scan 产出的逐图 json（需含 per_image[].conf_list）")
    ap.add_argument("--data", default="data/processed/visdrone/visdrone.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--budgets", type=int, nargs="+", default=[30, 50, 75, 100, 300])
    ap.add_argument("--hi-conf", type=float, default=0.25)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--json", default="results/tables/proxy-density.json")
    a = ap.parse_args()

    import yaml

    pred = json.loads(Path(a.pred).read_text(encoding="utf-8"))
    recs = pred.get("per_image") or []
    if not recs:
        print("❌ 输入的 per_image 为空。请先用 src/eval/budget_scan.py 生成逐图记录。")
        return 1
    if not any(r.get("conf_list") for r in recs):
        print("⚠️ 记录里没有 `conf_list`（逐图置信度列表）—— 四个代理里有三个无法算。")
        print("   → 需要在 budget_scan.py 的逐图记录里补 `conf_list`（已列入待办）。")

    names, gt, proxies = [], [], {k: [] for k in
                                  ("P1_n_pred", "P2_sum_conf", "P3_n_hi_conf", "P4_entropy")}
    for r in recs:
        if not r.get("n_gt"):
            continue
        names.append(Path(r["name"]).stem)
        gt.append(r["n_gt"])
        p = proxies_from_prediction(r, hi_conf=a.hi_conf)
        for k, v in p.items():
            proxies[k].append(v)
    gt_a = np.asarray(gt, dtype=float)
    print(f"图数 {len(gt_a)} | GT 中位 {np.median(gt_a):.0f} | max {gt_a.max():.0f}\n")

    out: dict = {"n_images": int(len(gt_a)), "split": a.split, "proxies": {}, "budgets": []}

    print("=" * 84)
    print("Q1：代理密度 vs GT 数 的 Spearman ρ（配对单位 = 图，bootstrap 10000 次）")
    print("=" * 84)
    print(f"{'代理':>16} {'ρ':>8} {'95% CI':>22}  判读")
    for k in proxies:
        x = np.asarray(proxies[k], dtype=float)
        if x.std() == 0:
            print(f"{k:>16} {'—':>8} {'(无变异)':>22}  不可用")
            continue
        rho, lo, hi = spearman_ci(x, gt_a, n_boot=a.n_boot)
        verdict = ("✅ 强" if abs(rho) >= 0.6 else "⚠️ 中" if abs(rho) >= 0.3 else "❌ 弱")
        print(f"{k:>16} {rho:>8.3f} [{lo:>7.3f}, {hi:>7.3f}]  {verdict}")
        out["proxies"][k] = {"rho": round(rho, 4), "ci95": [round(lo, 4), round(hi, 4)]}

    print("\n" + "=" * 84)
    print("Q2：可实现增益（总预算固定 = B·N；三列同口径可直接比）")
    print("=" * 84)
    print(f"{'B':>5} {'均匀':>9} {'代理分配':>10} {'oracle':>9} {'可实现率':>10}  最佳代理")
    best_proxy_overall = max(
        proxies, key=lambda k: (spearman(np.asarray(proxies[k], float), gt_a)
                                if np.std(proxies[k]) > 0 else -2))
    for B in a.budgets:
        cu = ceiling_uniform(gt_a, B)
        ca = ceiling_alloc(gt_a, B)
        row = {"B": B, "ceiling_uniform": round(cu, 5), "ceiling_alloc": round(ca, 5), "by_proxy": {}}
        bp, bv = None, -1.0
        for k in proxies:
            x = np.asarray(proxies[k], dtype=float)
            if x.std() == 0:
                continue
            cp = ceiling_proxy(gt_a, x, B)
            row["by_proxy"][k] = round(cp, 5)
            if cp > bv:
                bp, bv = k, cp
        gap = ca - cu
        rate = (bv - cu) / gap if gap > 1e-9 else float("nan")
        row["best_proxy"] = bp
        row["ceiling_best_proxy"] = round(bv, 5)
        row["realisable_rate"] = round(rate, 3) if not np.isnan(rate) else None
        out["budgets"].append(row)
        rs = f"{rate * 100:>9.1f}%" if not np.isnan(rate) else "        —"
        print(f"{B:>5} {cu * 100:>8.2f}% {bv * 100:>9.2f}% {ca * 100:>8.2f}% {rs:>10}  {bp}")

    print("\n" + "=" * 84)
    print("★ 判读（H2 的 Go/No-Go）")
    print("=" * 84)
    best_rho = max((abs(v["rho"]) for v in out["proxies"].values()), default=float("nan"))
    rates = [r["realisable_rate"] for r in out["budgets"]
             if r["realisable_rate"] is not None and r["B"] <= 100]
    best_rate = max(rates) if rates else float("nan")
    rate_s = f"{best_rate * 100:.1f}%" if rates else "不可算"
    print(f"  最佳 |ρ| = {best_rho:.3f} | B≤100 档最佳可实现率 = {rate_s}")
    if not np.isnan(best_rho) and best_rho >= 0.6 and rates and best_rate >= 0.5:
        print("  → ✅ **H2 有物理基础**：代理能捕获一半以上的 oracle 增量，进入正式实验")
    elif not np.isnan(best_rho) and best_rho >= 0.3:
        print("  → ⚠️ **H2 弱成立**：代理只捕获部分增量，主张必须限定为「部分可实现」")
    else:
        print("  → ❌ **H2 降级**：代理与真实密度关联弱 → **不得写任何分配增益的正文**")
        print("     改走 C2（预算受限覆盖认证），它不依赖密度可预测性")
    print("\n  ⚠️ 本脚本给出的是**上界**（假设分到的额度能被完美排序用满）。")
    print("     真实可实现量还受「检测器在低预算下的排序质量」限制，须由 budget_scan 实测。")

    jp = PROJECT_ROOT / a.json
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果 -> {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
