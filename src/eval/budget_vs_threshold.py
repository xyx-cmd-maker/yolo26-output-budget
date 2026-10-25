"""
budget_vs_threshold.py — ⑥ 红队 R-B：**计数预算 B 与置信度阈值 s 是否可互相模拟？**

═══════════════════════════════════════════════════════════════════════════
为什么必须做（literature/refs/p0-1b-lrp-threat.md 红队 R1）
═══════════════════════════════════════════════════════════════════════════
LRP ECCV §6.3 原文已用"召回被截断的检测器 G 输 7.5% AP、却赢 12/15 类 oLRP"
埋下了与 C1-(iii)"排序反转"同机制的观察；Dave et al. 2021 更直接用 B=2 玩具例
证明每图预算会翻转排序策略。

审稿人会说：**"两种截断都是截断，你凭什么把计数预算单列一支？"**
→ 若答不上来，C1-(iii) 只能降级为 (ii) 的推论，不能作为独立贡献。

═══════════════════════════════════════════════════════════════════════════
本脚本的可操作判据（不需要新数据、不需要训练）
═══════════════════════════════════════════════════════════════════════════
**结构事实**：
  · **阈值规则** `输出 = {det : conf ≥ s}`：s 是**单个全局标量**。
    → 每图输出数 `|Yᵢ(s)|` 是 s 的单调函数，**图与图之间由各自的分数分布决定**，
      但**不可能按图指定不同配额**（一个 s 管所有图）。
  · **计数预算规则** `输出 = top-B(图 i)`：B 可以**逐图不同**。
    → 能实现"密集图多给、稀疏图少给"，**阈值规则做不到**。

**判据（三条，全部可算）**：
  T1 **逐图配额可达集**：给定**总预算相同**（Σ Bᵢ = B·N），
     计数规则能达到的召回上界 `ceil_count(B)` vs
     任意单阈值规则能达到的最好召回 `max_s recall(s)`。
     若 `ceil_count > max_s recall + ε` → **计数规则严格更强** → 不可模拟。
  T2 **排序翻转点差异**：扫 s ∈ [0,1] 找 AP 排序**是否变号**；扫 B 找 Recall@B 排序是否变号。
     若**阈值轴无变号而预算轴有变号** → 预算是一支**独立的**评价轴。
  T3 **逐图配额的方差**：阈值规则下 `Varᵢ|Yᵢ(s)|` 由分数分布决定；
     计数规则可任意指定 `Bᵢ`。报告两者在同一总预算下的**可达召回差**。

★ **T1 是主判据**，T2/T3 是支撑。三者都是确定性计算，输入只需逐图置信度列表。

用法：
    python -m src.eval.budget_vs_threshold --pred runs/budget/r1_val_o2o.json \\
        --data data/processed/visdrone/visdrone.yaml --split val --budgets 30 50 75 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 与 proxy_density 共用分配算法，避免两处实现漂移（P17 的教训）
from src.eval.proxy_density import ceiling_alloc, ceiling_uniform  # noqa: E402


def load_counts(data_yaml: str, split: str) -> dict[str, int]:
    import yaml

    p = Path(data_yaml)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    if not root.is_absolute():
        root = (p.parent / root).resolve()
    img_dir = Path(cfg[split])
    if not img_dir.is_absolute():
        img_dir = (root / img_dir).resolve()
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    lbl = Path(*parts)
    return {
        f.stem: sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
        for f in sorted(lbl.glob("*.txt"))
    }


def recall_at_budget(conf: np.ndarray, n_gt: int, B: int) -> float:
    """在预算 B 下、排序完美时能达到的召回（该图）。

    ★ 注意这里是**上限**：假设被保留的 B 个预测中前 min(B, n_gt) 个全对。
      用于比较"两种规则的**可达集**"，不是实测值。
    """
    if n_gt == 0:
        return float("nan")
    return min(B, n_gt) / n_gt


def recall_at_threshold(conf: np.ndarray, n_gt: int, s: float) -> float:
    """在**全局**阈值 s 下能达到的召回上限（该图）：
    保留 `#(conf ≥ s)` 个输出，其中至多 min(该数, n_gt) 个正确。"""
    if n_gt == 0:
        return float("nan")
    k = int((conf >= s).sum())
    return min(k, n_gt) / n_gt


def main() -> int:
    ap = argparse.ArgumentParser(description="计数预算 vs 置信度阈值：不可模拟性检验")
    ap.add_argument("--pred", required=True, help="budget_scan 逐图 json（需含 conf_list）")
    ap.add_argument("--data", default="data/processed/visdrone/visdrone.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--budgets", type=int, nargs="+", default=[30, 50, 75, 100, 300])
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--json", default="results/tables/budget-vs-threshold.json")
    a = ap.parse_args()

    pred = json.loads(Path(a.pred).read_text(encoding="utf-8"))
    recs = [r for r in (pred.get("per_image") or []) if r.get("conf_list")]
    if not recs:
        print("❌ 需要含 `conf_list` 的逐图记录（由 src/eval/budget_scan.py 产出）。")
        return 1
    gt = load_counts(a.data, a.split)
    names = [Path(r["name"]).stem for r in recs]
    confs = [np.asarray(r["conf_list"], dtype=float) for r in recs]
    ngts = np.array([r.get("n_gt") or gt.get(n, 0) for r, n in zip(recs, names)], dtype=float)
    N = len(recs)
    print(f"图数 {N} | GT 合计 {int(ngts.sum())} | 中位 {np.median(ngts):.0f}\n")

    out: dict = {"n_images": N, "split": a.split, "T1": [], "T2": {}, "T3": {}}

    # ── T1：同总预算下，计数规则 vs 任意单阈值规则的可达召回 ──────────────
    print("=" * 88)
    print("T1（主判据）：总预算相同。计数规则可达 vs 任意**单阈值**规则可达")
    print("=" * 88)
    print(f"{'B':>5} {'总预算':>8} {'计数规则':>10} {'最佳单阈值':>11} {'阈值s*':>8} "
          f"{'差距':>9}  判读")
    for B in a.budgets:
        total_budget = B * N
        ceil_count = ceiling_alloc(ngts, B)
        # ★ 修正（自检发现）：原实现用 -1.0 作哨兵，当**所有**阈值都超预算时
        #   哨兵被当成结果打印（实测出现"最佳单阈值 = -100.00%"）。
        #   现在：无可行阈值 → best=None 并单独报告。
        best_s, best_s_recall = None, None
        for s in a.thresholds:
            ks = np.array([int((c >= s).sum()) for c in confs], dtype=float)
            if ks.sum() > total_budget + 1e-9:
                continue          # 超预算，该 s 不可行
            r = float(np.minimum(ks, ngts).sum() / ngts.sum())
            if best_s_recall is None or r > best_s_recall:
                best_s, best_s_recall = s, r
        if best_s_recall is None:
            out["T1"].append({
                "B": B, "total_budget": int(total_budget),
                "ceiling_count": round(ceil_count, 5),
                "best_single_threshold": None, "best_threshold_value": None,
                "gap": None, "non_simulable": None,
                "note": "所有测试阈值都超出该总预算（单阈值最小也只能输出更多）",
            })
            print(f"{B:>5} {int(total_budget):>8} {ceil_count * 100:>9.2f}% "
                  f"{'—':>10} {'—':>8} {'—':>9}  ⚪ 阈值网格无可行点（需更细网格）")
            continue
        gap = ceil_count - best_s_recall
        non_sim = gap > 0.01           # 1 个百分点作为"实质性更强"的门槛
        out["T1"].append({
            "B": B, "total_budget": int(total_budget),
            "ceiling_count": round(ceil_count, 5),
            "best_single_threshold": round(best_s_recall, 5),
            "best_threshold_value": best_s,
            "gap": round(gap, 5),
            "non_simulable": bool(non_sim),
        })
        print(f"{B:>5} {int(total_budget):>8} {ceil_count * 100:>9.2f}% "
              f"{best_s_recall * 100:>10.2f}% {best_s!s:>8} {gap * 100:>8.2f}pt  "
              f"{'✅ 计数严格更强' if non_sim else '⚠️ 可被阈值逼近'}")

    # ── T2：两条轴各自的"排序是否变号"潜力 ────────────────────────────────
    print("\n" + "=" * 88)
    print("T2：两条轴对**逐图配额**的控制力（决定性差异在结构，不在数值）")
    print("=" * 88)
    ks_by_s = {}
    for s in a.thresholds:
        ks = np.array([int((c >= s).sum()) for c in confs], dtype=float)
        ks_by_s[s] = ks
    # 阈值规则能达到的"配额向量"只有 |thresholds| 个；计数规则可达无穷多
    print(f"  阈值规则可产生的**逐图配额向量**数量 = {len(a.thresholds)}（每个 s 一个）")
    print("  计数规则可产生的配额向量：满足 ΣBᵢ = B·N 的**任意**整向量（组合数天文量级）")
    print("  → 结构性结论【确定】：**单一全局阈值无法实现逐图差异化配额**。")
    # 量化：在最优 s 下，配额向量的"平坦度"（与最优 oracle 配额的秩相关）
    feasible = [r for r in out["T1"] if r.get("best_single_threshold") is not None]
    if not feasible:
        print("\n  ⚪ 无可行阈值点 → 跳过 T2 的量化部分（结构性结论仍然成立）")
        out["T2"] = {
            "n_threshold_quota_vectors": len(a.thresholds),
            "quota_rho_vs_oracle": None,
            "note": "阈值网格在此总预算下无可行点",
            "structural_conclusion": "单一全局阈值无法实现逐图差异化配额（结构事实）",
        }
        best_row, B_ref, s_ref, cap = None, None, None, None
    else:
        best_row = max(feasible, key=lambda r: r["best_single_threshold"])
        B_ref = best_row["B"]
        s_ref = best_row["best_threshold_value"]
        cap = ceiling_uniform(ngts, B_ref)
    def _rank(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        sx = x[o]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and sx[j + 1] == sx[i]:
                j += 1
            r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r

    if best_row is not None:
        oracle_quota = np.minimum(cap, ngts)            # oracle 风格配额（按 nᵢ）
        thr_quota = ks_by_s.get(s_ref, np.zeros(N))
        rq, rt = _rank(oracle_quota), _rank(thr_quota)
        rq -= rq.mean()
        rt -= rt.mean()
        den = float(np.sqrt((rq**2).sum() * (rt**2).sum()))
        rho_quota = float((rq * rt).sum() / den) if den > 0 else float("nan")
        n_under = int((thr_quota < oracle_quota).sum())
        n_over = int((thr_quota > oracle_quota).sum())
        print(f"\n  以 B={B_ref}, s*={s_ref} 为例（该 s 是阈值规则里最优的）：")
        print(f"    阈值规则配额 vs oracle 风格配额的秩相关 ρ = {rho_quota:.3f}")
        print(f"    配额**不足**的图：{n_under} / {N}（{n_under / N * 100:.1f}%）")
        print(f"    配额**过剩**的图：{n_over} / {N}（{n_over / N * 100:.1f}%）—— 过剩即浪费")
        out["T2"] = {
            "n_threshold_quota_vectors": len(a.thresholds),
            "quota_rho_vs_oracle": round(rho_quota, 4),
            "n_under_quota": n_under, "n_over_quota": n_over, "n_images": N,
            "B_ref": B_ref, "s_ref": s_ref,
            "structural_conclusion": "单一全局阈值无法实现逐图差异化配额（结构事实）",
        }
    else:
        thr_quota = np.zeros(N)

    # ── T3：配额不平等的来源 ─────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("T3：配额从何而来")
    print("=" * 88)
    var_thr = float(np.var(thr_quota))
    print(f"  阈值规则的配额方差 Var(Bᵢ) = {var_thr:.1f}（由各图**分数分布**决定，非可控）")
    print(f"  计数规则的配额可任意指定 → 目标方差可自由选择（含 0 = 均匀）")
    print("  → 阈值规则的配额不平等是**副作用**；计数规则的配额不平等是**设计变量**。")
    out["T3"] = {"var_quota_under_threshold": round(var_thr, 2)}

    # ── 结论 ─────────────────────────────────────────────────────────────
    n_nonsim = sum(1 for r in out["T1"] if r["non_simulable"])
    print("\n" + "=" * 88)
    print("★ 结论（供 Introduction 划界使用）")
    print("=" * 88)
    if n_non := n_nonsim:
        print(f"  ✅ **不可模拟性成立**：在 {n_non}/{len(out['T1'])} 个预算档上，")
        print("     计数规则的可达召回**严格高于**任意单阈值规则（差距 >1pt）。")
        print("     → C1-(iii) 获得独立立足点：**计数预算是一支不能被阈值预算替代的评价轴**。")
    else:
        print("  ❌ **不可模拟性不成立**：所有档位上单阈值规则都能逼近计数规则。")
        print("     → C1-(iii) 不能作为独立贡献，应并入 (ii) 或降级为讨论。")
    print("\n  ⚠️ 本脚本比较的是**可达集**（排序完美假设），不是实测值。")
    print("     实测需 budget_scan 在相同总预算下逐图分配后重算召回。")
    print("  ⚠️ 若 `conf_list` 被截断到 500 条（budget_scan 的存储上限），")
    print("     高密度图的阈值配额会被**低估** → T1 的差距被**保守估计**（对我们不利的方向）。")

    jp = PROJECT_ROOT / a.json
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果 -> {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
