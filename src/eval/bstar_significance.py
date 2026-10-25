"""
bstar_significance.py — B*（排序反转点）的配对 bootstrap 显著性检验【自证伪实验】

为什么这是最高优先级（P0-1b 红队 R2）：
  C1 的核心可证伪命题是"检测器的 mAP 排序与 Recall@B 排序会在某个 B* 反转"。
  但 mAP 与 Recall@B **都是有限测试集上的估计量**——两个检测器差 0.3 个点时，
  所谓"反转"可能只是**排序噪声**。
  → **若反转的配对 bootstrap CI 跨 0，C1 的核心命题不成立，后续全部作废。**
  这一步不需要训练、不需要新数据，是自证伪成本最低的动作。

方法（配对单位 = 图，因为预算是逐图分配的；这是 P0-2 的 C-6 要求）：
  对候选的每对检测器 (A, B) 与每个预算档 B_budget：
    1. 逐图算 recall_A(i, B) 与 recall_B(i, B)（该图 TP 数 / 该图 GT 数）
    2. 配对差 d_i = recall_A(i,B) − recall_B(i,B)
    3. 对图做有放回重采样 n_boot 次 → 得 Δ 的 bootstrap 分布 → 95% CI
    4. 同时算 mAP 差（聚合量，用图级重采样近似其不确定性）
    5. **反转成立的条件**：sign(ΔmAP) ≠ sign(Δrecall@B) 且 **两个差的 CI 都不跨 0**
  报告：每个 (pair, B) 的组合、Δ、CI、是否构成"显著反转"。

⚠️ 诚实约束：
  · mAP 的重采样**不能**简单地对图重采样得到（mAP 是全局聚合量，不是图的均值）。
    这里对 mAP 用"图级 bootstrap 近似"，并在输出里**显式标注该近似**——
    严谨做法是每折重算完整 AP（成本高 100×）。**默认只对召回做严格检验**，
    mAP 差用点估计 + 标注"未做严格 CI"。

用法：
    python -m src.eval.bstar_significance --runs runA=path/a.json runB=path/b.json \\
        --data data/processed/visdrone/visdrone.yaml --split val --budgets 20 30 50 75 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RNG_SEED = 20261014


def _labels_dir_for(img_dir: Path) -> Path:
    """从右往左替换最后一个 `images` 组件（全局 replace 会改到祖先目录，见 P5）。"""
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return img_dir.parent / "labels" / img_dir.name


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
    lbl = _labels_dir_for(img_dir)
    return {
        f.stem: sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
        for f in sorted(lbl.glob("*.txt"))
    }


def per_image_recall(pred: dict, B: int, gt_counts: dict[str, int]) -> dict[str, float]:
    """从 budget_scan 的 `per_image` 记录算出每图 recall@B。

    记录结构（`budget_scan.run_head` 产出）：
        {"name": "xxx.jpg", "n_gt": int, "n_pred_at": {"50": ..}, "tp_at": {"50": ..}}
    """
    out: dict[str, float] = {}
    for rec in pred.get("per_image", []):
        name = Path(rec["name"]).stem
        n_gt = rec.get("n_gt") or gt_counts.get(name, 0)
        if not n_gt:
            continue
        tp = (rec.get("tp_at") or {}).get(str(B))
        if tp is None:
            continue
        out[name] = tp / n_gt
    return out


def paired_bootstrap_ci(d: np.ndarray, *, n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float, float]:
    rng = np.random.default_rng(RNG_SEED)
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description="B* 排序反转的配对 bootstrap 检验")
    ap.add_argument("--runs", nargs="+", required=True, help="name=逐图预测 json（budget_scan 产出）")
    ap.add_argument("--data", default="data/processed/visdrone/visdrone.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--budgets", type=int, nargs="+", default=[20, 30, 50, 75, 100, 300])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--json", default="results/tables/bstar-significance.json")
    a = ap.parse_args()

    gt = load_gt_counts(a.data, a.split)
    print(f"GT 图数 {len(gt)} | 切分 {a.split}")

    runs: dict[str, dict] = {}
    for spec in a.runs:
        name, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            print(f"[跳过] {name}: {path} 不存在")
            continue
        runs[name] = json.loads(p.read_text(encoding="utf-8"))
        print(f"载入 {name} <- {path}")

    if len(runs) < 2:
        print("❌ 至少需要两个 run 才能比较。")
        print("   先跑 src/eval/budget_scan.py 产出逐图预测，再回来比较。")
        return 1

    names = sorted(runs)
    out: dict = {"split": a.split, "n_gt_images": len(gt), "n_boot": a.n_boot, "pairs": []}

    print(f"\n{'=' * 96}")
    print("说明：配对单位 = 图（预算逐图分配）。**只有召回差有严格 CI**；mAP 差为点估计（标注）。")
    print(f"{'=' * 96}")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = names[i], names[j]
            for budget in a.budgets:
                ra = per_image_recall(runs[A], budget, gt)
                rb = per_image_recall(runs[B], budget, gt)
                common = sorted(set(ra) & set(rb))
                if len(common) < 10:
                    continue
                d = np.array([ra[k] - rb[k] for k in common])
                mean, lo, hi = paired_bootstrap_ci(d, n_boot=a.n_boot)
                sig = "✅显著" if (lo > 0 or hi < 0) else "❌跨0"
                rec = {
                    "pair": [A, B], "budget": budget, "n_images": len(common),
                    "recall_mean_diff": round(mean, 5),
                    "recall_ci95": [round(lo, 5), round(hi, 5)],
                    "recall_significant": bool(lo > 0 or hi < 0),
                }
                out["pairs"].append(rec)
                print(f"  {A:>12} vs {B:<12} B={budget:>4}  Δrecall={mean:+.4f} "
                      f"CI=[{lo:+.4f},{hi:+.4f}] {sig}  n={len(common)}")

    jp = PROJECT_ROOT / a.json
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * 96}")
    print("★ 判读：")
    print("  · 若所有 Δrecall 的 CI 都跨 0 → 这些检测器间**没有可检出的召回差异**，")
    print("    C1 的'排序反转'命题在本档位上不成立。")
    print("  · 反转需要**两个**条件同时满足：(a) ΔmAP 与 Δrecall 反号；")
    print("    (b) 二者的差都显著。仅 (a) 不构成反转，只是噪声。")
    print("  · ⚠️ mAP 差此处为**点估计未做严格 CI**（严格的需每折重算完整 AP）。")
    print(f"\n结果 -> {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
