"""
从 COCO annotations 提取**逐图目标数分布**，并与 VisDrone 做对照。

═══════════════════════════════════════════════════════════════════════════
为什么这对我们至关重要（零训练成本）
═══════════════════════════════════════════════════════════════════════════
本项目的**容量项**是：
    `ceiling(B) = Σ min(B, nᵢ) / Σ nᵢ`
它**只依赖逐图目标数 `nᵢ` 的分布** —— 不需要图片、不需要训练、不需要检测器。

**→ 因此"跨数据集对照"可以零训练完成**：把 COCO 的 `nᵢ` 分布代进去即可。

用途：
  1. 回答审稿人"零和性/公平性增益是否只是 VisDrone 密度分布的产物"（红队 R2）
  2. 给出"预算利用率"与"饱和点"在**不同密度分布**下的可比数字
  3. 为论文提供第二个数据集（**维度**：标签分布，而非 perf）

产出：`results/tables/n-gt-distributions.json`
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
ZIP = ROOT / "data" / "raw" / "coco_annotations_trainval2017.zip"
OUT = ROOT / "results" / "tables" / "n-gt-distributions.json"


def visdrone_counts(split: str) -> np.ndarray:
    import yaml
    cfg = yaml.safe_load((ROOT / "data/processed/visdrone/visdrone.yaml").read_text(encoding="utf-8"))
    d = Path(cfg["path"]) / cfg[split]
    parts = list(d.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return np.array([sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
                     for f in sorted(Path(*parts).glob("*.txt"))], dtype=float)


def coco_counts(inner_name: str) -> np.ndarray:
    print(f"  从 zip 读 {inner_name} …", flush=True)
    with zipfile.ZipFile(ZIP) as z:
        with z.open(inner_name) as f:
            obj = json.load(io.TextIOWrapper(f, encoding="utf-8"))
    per = {}
    for a in obj["annotations"]:
        per[a["image_id"]] = per.get(a["image_id"], 0) + 1
    counts = [per.get(im["id"], 0) for im in obj["images"]]
    return np.array(counts, dtype=float)


def stats(n: np.ndarray) -> dict:
    n = n[n > 0] if (n > 0).any() else n
    return {
        "N_images": int(len(n)),
        "sum": int(n.sum()),
        "mean": round(float(n.mean()), 3),
        "median": float(np.median(n)),
        "p90": float(np.percentile(n, 90)),
        "p99": float(np.percentile(n, 99)),
        "max": int(n.max()),
        "min": int(n.min()),
        "skew_ratio_mean_over_max": round(float(n.mean() / max(n.max(), 1)), 4),
        "frac_over_100": round(float((n > 100).mean()), 4),
        "frac_over_300": round(float((n > 300).mean()), 4),
    }


def utilisation(n: np.ndarray, B: int) -> dict:
    """预算利用率 `util(B) = Σmin(B,nᵢ)/(B·N)` 与天花板 `ceiling(B) = Σmin(B,nᵢ)/Σnᵢ`。"""
    spent = float(np.minimum(B, n).sum())
    return {
        "B": B,
        "ceiling": round(spent / float(n.sum()), 5),
        "utilisation": round(spent / (B * len(n)), 5),
        "images_capped": int((n > B).sum()),
        "frac_capped": round(float((n > B).mean()), 4),
    }


def main() -> int:
    print("=" * 96)
    print("逐图目标数分布：VisDrone vs COCO（零训练跨数据集对照）")
    print("=" * 96)

    data: dict[str, dict] = {}

    for split in ("val", "test"):
        n = visdrone_counts(split)
        data[f"visdrone_{split}"] = {"stats": stats(n), "util": [utilisation(n, B) for B in
                                                                (10, 20, 30, 50, 100, 300)]}
        print(f"\n  visdrone_{split}: {data[f'visdrone_{split}']['stats']}")

    if not ZIP.exists():
        print(f"\n⚠️ 缺 {ZIP} —— 跳过 COCO")
        return 1

    for tag, inner in (("coco_val2017", "annotations/instances_val2017.json"),
                       ("coco_train2017", "annotations/instances_train2017.json")):
        try:
            n = coco_counts(inner)
            data[tag] = {"stats": stats(n), "util": [utilisation(n, B) for B in
                                                     (10, 20, 30, 50, 100, 300)]}
            print(f"\n  {tag}: {data[tag]['stats']}")
        except KeyError as e:
            print(f"\n  ❌ {tag}：zip 里没有 {inner}（{e}）")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {OUT.relative_to(ROOT)}")

    # ── 对照表：天花板与利用率 ──────────────────────────────────
    print("\n" + "=" * 96)
    print("★ 跨数据集对照：ceiling(B)（总量能覆盖多少）与 util(B)（预算用得多满）")
    print("=" * 96)
    keys = [k for k in data]
    print(f"  {'B':>5} │ " + " │ ".join(f"{k[:18]:>18}" for k in keys))
    print("  " + "-" * (7 + 21 * len(keys)))
    for i, B in enumerate((10, 20, 30, 50, 100, 300)):
        cells = []
        for k in keys:
            u = data[k]["util"][i]
            cells.append(f"{u['ceiling'] * 100:>7.2f}%/{u['utilisation'] * 100:>6.1f}%")
        print(f"  {B:>5} │ " + " │ ".join(f"{c:>18}" for c in cells))
    print("\n  （每格：天花板 ceiling / 利用率 util）")

    print("\n★ 判读：")
    print("  · 若 COCO 的 `mean/max` 明显高于 VisDrone → COCO **更均匀** →")
    print("    则「分配」的收益空间更小 → 我们的 +8 pt 是**密集倾斜分布**特有的")
    print("  · 若两者量级相近 → 分配问题在 COCO 上同样存在 → **跨数据集稳健**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
