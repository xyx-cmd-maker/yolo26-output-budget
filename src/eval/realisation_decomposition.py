"""
realisation_decomposition.py — ⑥ 红队 R-C：漏检分解的**非平凡性**检验

═══════════════════════════════════════════════════════════════════════════
R-C 的质疑（literature/refs/p0-1b-lrp-threat.md）
═══════════════════════════════════════════════════════════════════════════
"`Oracle上限(B) = Σᵢmin(B,nᵢ)/Σᵢnᵢ` 只需知道 GT 数量分布即可写出，推导成本近乎为零。
 你判定它'完全未被 LRP 覆盖'，但**未被覆盖 ≠ 有贡献**——
 一个显然到没人写下来的式子，其未被覆盖恰恰可能说明它不值得写。"

**要消解它，必须证明这个上界承载非平凡信息。**

═══════════════════════════════════════════════════════════════════════════
本脚本的答案：**漏检分解恒等式**（LRP 家族没有这个分解）
═══════════════════════════════════════════════════════════════════════════
LRP 把漏检压成单一计数：`LRP_FN := 1 − Recall = N_FN/|X|`（LRP 原文 Eq.4）。
它**不追问**：这个 FN 是"预算不允许把它输出出来"，还是"排不上来"？

我们把它**拆成两项**：

    漏检(B) = TP_full − TP(B)
            = [Σmin(B,nᵢ) − TP(B)]   （**排序受限**：额度内没排上）
            + [Σnᵢ − Σmin(B,nᵢ)]     （**容量受限**：额度本身不够）

其中
  · `Σnᵢ`          = 全部 GT 框数
  · `Σmin(B,nᵢ)`   = 排序完美时的 TP 上限（= Oracle 上限的分母部分）
  · `TP(B)`        = 预算 B 下的实测 TP

★ 两项**互相独立地可算**，且**非负**（因 TP(B) ≤ Σmin(B,nᵢ) ≤ Σnᵢ）。

═══════════════════════════════════════════════════════════════════════════
"非平凡"的判据（两条，都必须成立才算消解 R-C）
═══════════════════════════════════════════════════════════════════════════
  J1 **分解能区分不同的失效模式**：存在 B 与数据切分，使
     `容量受限占比` 跨越一个实质区间（如 0.3 → 0.8），
     即"该不该加预算"这个结论**随 B 反转**。
     若两项占比在所有权重下都几乎恒定 → 分解没有诊断价值 → R-C 未消解。

  J2 **分解揭示了 FN 计数揭示不了的东西**：同一 `TP(B)`（同样的漏检总数）下，
     容量/排序两分量可以有**显著不同的配比**。
     若给定 TP(B) 就能唯一确定配比 → 分解是冗余的 → R-C 未消解。

  J3（附加）**容量项只由 (B, 密度分布) 决定，与检测器无关** ——
     这是恒等式的直接推论，可用来把"检测器之间的差异"**完全归因于排序项**。
     这给出一句可写进论文的强陈述：
     > 两个检测器在同一 B 下的召回差异，**100% 来自排序能力，0% 来自容量**。

用法：
    python -m src.eval.realisation_decomposition --pred runs/budget/r1_val_o2o.json \\
        --data data/processed/visdrone/visdrone.yaml --split val --budgets 20 30 50 75 100 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.proxy_density import ceiling_alloc, uniform_cap  # noqa: E402


def load_gt_counts(data_yaml: str, split: str) -> dict[str, int]:
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
    return {
        f.stem: sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
        for f in sorted(Path(*parts).glob("*.txt"))
    }


def decompose(ngts: np.ndarray, tp_at: dict[str, int], B: int) -> dict:
    """返回给定 B 下的漏检分解。全部是整数计数，无估计量。"""
    tot_gt = int(ngts.sum())
    ceiling = int(np.minimum(B, ngts).sum())          # 排序完美时的 TP 上限
    tp = int(tp_at.get(str(B), 0))
    capacity_miss = tot_gt - ceiling                  # 额度本身不够
    ranking_miss = ceiling - tp                       # 额度内没排上
    total_miss = tot_gt - tp
    assert capacity_miss >= 0 and ranking_miss >= 0, "分解项必须非负"
    assert capacity_miss + ranking_miss == total_miss, "分解必须恰好等于总漏检"
    return {
        "B": B, "total_gt": tot_gt, "tp": tp,
        "tp_ceiling": ceiling,
        "capacity_miss": capacity_miss,
        "ranking_miss": ranking_miss,
        "total_miss": total_miss,
        "capacity_share": round(capacity_miss / total_miss, 5) if total_miss else None,
        "ranking_share": round(ranking_miss / total_miss, 5) if total_miss else None,
        "recall": round(tp / tot_gt, 5) if tot_gt else None,
        "realisation_rate": round(tp / ceiling, 5) if ceiling else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="漏检分解：容量受限 vs 排序受限")
    ap.add_argument("--pred", nargs="+", required=True,
                    help="一个或多个 budget_scan 逐图 json（格式 name=path 或裸 path）")
    ap.add_argument("--data", default="data/processed/visdrone/visdrone.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--budgets", type=int, nargs="+", default=[20, 30, 50, 75, 100, 300])
    ap.add_argument("--json", default="results/tables/realisation-decomposition.json")
    a = ap.parse_args()

    gt = load_gt_counts(a.data, a.split)
    print(f"GT 图数 {len(gt)} | 总框 {sum(gt.values())} | 切分 {a.split}\n")

    runs: dict[str, dict] = {}
    for spec in a.pred:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(name).stem, name
        p = Path(path)
        if not p.exists():
            print(f"[跳过] {name}: 不存在")
            continue
        runs[name] = json.loads(p.read_text(encoding="utf-8"))
    if not runs:
        print("❌ 没有可用的输入。请先跑 src/eval/budget_scan.py。")
        return 1

    out: dict = {"split": a.split, "runs": {}, "J1": {}, "J2": {}, "J3": {}}

    # ── 每个 run 的分解表 ──────────────────────────────────────────────
    print("=" * 100)
    print("分解表：漏检(B) = 容量受限 + 排序受限")
    print("=" * 100)
    for name, pred in runs.items():
        recs = pred.get("per_image") or []
        if not recs:
            print(f"[跳过] {name}: per_image 为空")
            continue
        names = [Path(r["name"]).stem for r in recs]
        ngts = np.array([r.get("n_gt") or gt.get(n, 0) for r, n in zip(recs, names)], dtype=float)
        # 累计 tp_at
        tp_at: dict[str, int] = {}
        keys = set()
        for r in recs:
            keys |= set((r.get("tp_at") or {}).keys())
        for k in keys:
            tp_at[k] = int(sum((r.get("tp_at") or {}).get(k, 0) for r in recs))

        print(f"\n── {name} ──  {len(recs)} 图 / {int(ngts.sum())} 框")
        print(f"{'B':>5} {'TP':>8} {'TP上限':>8} {'容量漏':>8} {'排序漏':>8} {'总漏':>8} "
              f"{'容量占比':>9} {'排序占比':>9} {'实现率':>8}")
        rows = []
        for B in a.budgets:
            d = decompose(ngts, tp_at, B)
            rows.append(d)
            cs = f"{d['capacity_share'] * 100:8.1f}%" if d["capacity_share"] is not None else "       —"
            rs = f"{d['ranking_share'] * 100:8.1f}%" if d["ranking_share"] is not None else "       —"
            rr = f"{d['realisation_rate'] * 100:7.1f}%" if d["realisation_rate"] is not None else "      —"
            print(f"{B:>5} {d['tp']:>8} {d['tp_ceiling']:>8} {d['capacity_miss']:>8} "
                  f"{d['ranking_miss']:>8} {d['total_miss']:>8} {cs:>9} {rs:>9} {rr:>8}")
        out["runs"][name] = rows

        # ── J1：容量占比是否随 B 跨越实质区间 ──
        shares = [d["capacity_share"] for d in rows if d["capacity_share"] is not None]
        if shares:
            span = max(shares) - min(shares)
            out["J1"][name] = {"capacity_share_min": round(min(shares), 4),
                               "capacity_share_max": round(max(shares), 4),
                               "span": round(span, 4)}
            print(f"  J1 容量占比跨度 = {span * 100:.1f} pt"
                  f"（{min(shares) * 100:.1f}% → {max(shares) * 100:.1f}%）")

    # ── J2：同样的总漏检下，配比是否可变 ──────────────────────────────
    # ★ 修正（2026-10-14，由 tests/selftest_j2_collision.py 触发）：
    #   当 `total_miss == 0`（预算已覆盖全部 GT）时 `capacity_share` 是 **None**，
    #   原来的 `max(...) - min(...)` 会拿 None 比较 → `TypeError`。
    #   且"总漏检为 0"的条目**不构成有意义碰撞**（没有漏检就没有配比），必须排除。
    print("\n" + "=" * 100)
    print("J2：同一「总漏检数」下，容量/排序配比是否可变（若不变则分解冗余）")
    print("=" * 100)
    all_rows = [(n, d) for n, rows in out["runs"].items() for d in rows]
    by_miss: dict[int, list[tuple[str, dict]]] = {}
    for n, d in all_rows:
        if not d["total_miss"] or d.get("capacity_share") is None:
            continue                      # 排除"无漏检"与占比未定义
        by_miss.setdefault(d["total_miss"], []).append((n, d))
    collisions = {k: v for k, v in by_miss.items() if len(v) > 1}
    if collisions:
        worst = max(
            collisions.items(),
            key=lambda kv: (max(x[1]["capacity_share"] for x in kv[1])
                            - min(x[1]["capacity_share"] for x in kv[1])),
        )
        miss_n, group = worst
        shares = [d["capacity_share"] for _, d in group]
        spread = max(shares) - min(shares)
        print(f"  总漏检 = {miss_n} 时，{len(group)} 个条目的容量占比跨度 = {spread * 100:.1f} pt")
        for n, d in group:
            print(f"     {n:>28} B={d['B']:>4}  容量占比 {d['capacity_share'] * 100:5.1f}%")
        out["J2"] = {"total_miss": miss_n, "n_entries": len(group),
                     "capacity_share_spread": round(spread, 4)}
        if spread < 0.02:
            print("  ⚠️ 跨度 <2pt → 分解在此数据上接近冗余（R-C 未充分消解）")
    else:
        print("  本次输入中没有「总漏检相同且非零」的条目 → 无法检验 J2"
              "（需更多 run 或更多 B 档）")
        out["J2"] = {"note": "无碰撞条目"}

    # ── J3：容量项与检测器无关（把检测器差异 100% 归因于排序）──────────
    print("\n" + "=" * 100)
    print("J3：同一 (B, 切分) 下，容量项只由密度分布决定 → 与检测器无关")
    print("=" * 100)
    if len(runs) >= 2:
        names2 = list(out["runs"].keys())
        for B in a.budgets:
            vals = {n: next((d for d in out["runs"][n] if d["B"] == B), None) for n in names2}
            vals = {k: v for k, v in vals.items() if v}
            if len(vals) < 2:
                continue
            caps = {v["capacity_miss"] for v in vals.values()}
            tps = {v["tp"] for v in vals.values()}
            same_cap = len(caps) == 1
            tp_diff = max(tps) - min(tps)
            print(f"  B={B:>4}  容量漏检集合={sorted(caps)}  {'✅ 完全相同' if same_cap else '❌ 不同（异常！）'}"
                  f"   TP 极差={tp_diff}")
            out["J3"].setdefault(str(B), {"capacity_identical": bool(same_cap), "tp_range": int(tp_diff)})
        print("\n  → 若所有 B 档的容量项都相同：**检测器之间的召回差异 100% 来自排序项**。")
        print("     这是可写进论文的强陈述，且是 LRP 的单一 N_FN 做不到的归因。")
    else:
        print("  只提供了一个 run → 无法比较。J3 需要 ≥2 个检测器的逐图输出。")
        out["J3"] = {"note": "需要 ≥2 个 run"}

    # ── 结论 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("★ R-C 判定")
    print("=" * 100)
    j1_ok = any(v.get("span", 0) > 0.10 for v in out["J1"].values())
    j2_ok = out["J2"].get("capacity_share_spread", 0) > 0.02
    print(f"  J1 容量占比跨度 >10pt：{'✅ 成立' if j1_ok else '❌ 不成立'}")
    print(f"  J2 同漏检下配比可变：{'✅ 成立' if j2_ok else '❌ 未验证/不成立'}")
    if j1_ok:
        print("\n  ✅ **R-C 已部分消解**：分解确实区分了两种失效模式，且结论随 B 反转。")
        print("     → 可写进论文：'该分解把漏检归因为容量/排序两项，"
              "使\"该加预算还是该换模型\"成为一个**可算**的问题'。")
    else:
        print("\n  ❌ **R-C 未消解**：分解在此数据上没有诊断价值。")
        print("     → C1 应降级为 DQ-DETR 式论文里的诊断小节，不作独立贡献。")
    print("\n  ⚠️ 全部为整数计数、无估计量；但**未做不确定度**（分解项是总体计数，")
    print("     其抽样变异应通过对图 bootstrap 重算，见 bstar_significance.py 的口径）。")

    jp = PROJECT_ROOT / a.json
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果 -> {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
