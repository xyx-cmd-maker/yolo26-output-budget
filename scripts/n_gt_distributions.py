"""
★ 三数据集「逐图目标数分布」对照 —— 支撑论文的**边界条件**骨架

═══════════════════════════════════════════════════════════════════════════
为什么这是论文最强的一块
═══════════════════════════════════════════════════════════════════════════
论文的核心是一条**可操作判据**：

    只有当 `util(B) = Σmin(B,nᵢ)/(B·N)` 显著低于 1（预算真的在咬合）
    **且** `max nᵢ / mean nᵢ` 较大（分布高度倾斜）时，
    才该考虑按 `nᵢ` 分配输出预算；否则均匀配额已经足够。

**容量项只依赖 `nᵢ` 分布** → 三个数据集的对照**零训练**即可完成。

三个数据集覆盖两个域 × 两种密度：
  · **VisDrone**（空中 / 密集）    mean 70.7、max 317
  · **COCO**    （地面 / 稀疏）    mean  7.4、max  63
  · **CrowdHuman**（地面 / 稠密人群）← 本轮新增，补上"地面密集"这一格

产出：`results/tables/n-gt-distributions.json`（合并）+ 控制台对照表
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
ZIP = ROOT / "data" / "raw" / "coco_annotations_trainval2017.zip"
CH = ROOT / "data" / "raw" / "crowdhuman_annotation_train.odgt"
OUT = ROOT / "results" / "tables" / "n-gt-distributions.json"

BUDGETS = (10, 20, 30, 50, 100, 300, 500)


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
    with zipfile.ZipFile(ZIP) as z:
        with z.open(inner_name) as f:
            obj = json.load(io.TextIOWrapper(f, encoding="utf-8"))
    per: dict[int, int] = {}
    for a in obj["annotations"]:
        per[a["image_id"]] = per.get(a["image_id"], 0) + 1
    return np.array([per.get(im["id"], 0) for im in obj["images"]], dtype=float)


def crowdhuman_counts() -> np.ndarray:
    """CrowdHuman 的 `annotation_train.odgt` 是 **JSONL**（每行一个图的 JSON）。

    每行的 `gtboxes` 里含 `tag`：
      · `tag == "person"` → 计入（可见人体框）
      · `tag == "mask"` / `"ignore"` → **不计入**（被标为忽略区域）
    ★ 计数与 YOLO 的 `nᵢ` 口径对齐：**只数会进入评测的实例**。
    """
    counts = []
    with CH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gts = obj.get("gtboxes") or []
            n = sum(1 for g in gts if str(g.get("tag", "")).lower() in ("person", "person_", ""))
            counts.append(n)
    return np.array(counts, dtype=float)


def stats(n: np.ndarray) -> dict:
    m = n > 0
    nn = n[m] if m.any() else n
    _edges = np.unique(np.round(np.logspace(0, 3, 49)).astype(int))
    return {
        "N_images": int(len(n)),
        "N_images_with_gt": int(m.sum()),
        "sum": int(n.sum()),
        "mean": round(float(nn.mean()), 3),
        "median": float(np.median(nn)),
        "p90": round(float(np.percentile(nn, 90)), 2),
        "p99": round(float(np.percentile(nn, 99)), 2),
        "max": int(nn.max()),
        "min": int(nn.min()),
        "skew_ratio_mean_over_max": round(float(nn.mean() / max(nn.max(), 1)), 4),
        # ★ 分箱直方图：让下游（论文 Figure 1）能**无损重建 CDF**，
        #   而不必重新下载/解析原始标注。对数分箱覆盖 1…1000。
        "hist_edges": [int(e) for e in _edges],
        "hist_counts": [int(c) for c in np.histogram(nn, bins=np.append(_edges, _edges[-1] + 1))[0]],
    }


def utilisation(n: np.ndarray, B: int) -> dict:
    m = n > 0
    nn = n[m]
    spent = float(np.minimum(B, nn).sum())
    return {
        "B": B,
        "ceiling": round(spent / float(nn.sum()), 5),
        "utilisation": round(spent / (B * len(nn)), 5),
        "frac_capped": round(float((nn > B).mean()), 4),
    }


def main() -> int:
    print("=" * 104)
    print("三个数据集的逐图目标数分布（容量项 → 零训练对照）")
    print("=" * 104)

    data: dict[str, dict] = {}
    jobs = [
        ("visdrone_val", lambda: visdrone_counts("val")),
        ("visdrone_test", lambda: visdrone_counts("test")),
    ]
    if CH.exists():
        jobs.append(("crowdhuman_train", crowdhuman_counts))
    else:
        print(f"⚠️ 缺 {CH.name} —— 跳过 CrowdHuman")
    if ZIP.exists():
        jobs.append(("coco_val2017", lambda: coco_counts("annotations/instances_val2017.json")))
        jobs.append(("coco_train2017", lambda: coco_counts("annotations/instances_train2017.json")))
    else:
        print(f"⚠️ 缺 {ZIP.name} —— 跳过 COCO")

    for tag, fn in jobs:
        print(f"\n  读 {tag} …", flush=True)
        n = fn()
        data[tag] = {"stats": stats(n), "util": [utilisation(n, B) for B in BUDGETS]}
        s = data[tag]["stats"]
        print(f"    N={s['N_images']}  有GT={s['N_images_with_gt']}  Σn={s['sum']}  "
              f"mean={s['mean']}  median={s['median']}  max={s['max']}  mean/max={s['skew_ratio_mean_over_max']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {OUT.relative_to(ROOT)}")

    # ── 主对照表 ────────────────────────────────────────────────
    keys = list(data)
    print("\n" + "=" * 104)
    print("★ 天花板 ceiling(B) —— 总量能覆盖多少目标（分配问题的「存在性」）")
    print("=" * 104)
    print(f"  {'B':>5} │ " + " │ ".join(f"{k[:20]:>20}" for k in keys))
    print("  " + "-" * (7 + 23 * len(keys)))
    for i, B in enumerate(BUDGETS):
        print(f"  {B:>5} │ " + " │ ".join(f"{data[k]['util'][i]['ceiling'] * 100:>19.2f}%" for k in keys))

    print("\n" + "=" * 104)
    print("★ 预算利用率 util(B) —— 预算是否真的在咬合（分配问题的「必要性」）")
    print("=" * 104)
    print(f"  {'B':>5} │ " + " │ ".join(f"{k[:20]:>20}" for k in keys))
    print("  " + "-" * (7 + 23 * len(keys)))
    for i, B in enumerate(BUDGETS):
        print(f"  {B:>5} │ " + " │ ".join(f"{data[k]['util'][i]['utilisation'] * 100:>19.1f}%" for k in keys))

    print("\n" + "=" * 104)
    print("★ 被预算卡住的图占比 frac(nᵢ > B)")
    print("=" * 104)
    print(f"  {'B':>5} │ " + " │ ".join(f"{k[:20]:>20}" for k in keys))
    print("  " + "-" * (7 + 23 * len(keys)))
    for i, B in enumerate(BUDGETS):
        print(f"  {B:>5} │ " + " │ ".join(f"{data[k]['util'][i]['frac_capped'] * 100:>19.1f}%" for k in keys))

    print("\n" + "=" * 104)
    print("★ 这些量能说什么、不能说什么（第 43 轮撤回后修正，见 P39）")
    print("=" * 104)
    print("""  **能说**：某一预算 B 下，**有多少比例的图被预算卡住**（`frac(nᵢ>B)`），
          以及**总量上还能覆盖多少目标**（`ceiling(B)`）。
          → 这界住了"**有没有空间可分配**"。

  **不能说**：**分配是否有效**。本脚本输出的量全部只是**标签分布**的函数，
          而"重分配能否改善最坏图召回"取决于**检测器在最坏图上的表现**。
          实测（VisDrone val，双头，12 个可比档位）：
            r(Δworst, util)   = **−0.309**（o2m）/ −0.043（o2o）
            r(Δworst, frac>B) = **−0.270**（o2m）/ −0.010（o2o）
          → **与"预算咬合就有空间"的直觉方向相反**。

  **正确判据**见 `src/eval/allocation_regime.py`：
          看**最坏图的配额是否已触及它物理能输出的上限 `min(nᵢ, pᵢ)`**。
          触及 ⇒ 增益为 0；未触及 ⇒ 才有空间。

  **三个数据集的定位（仍然成立的观察，但不再是"边界条件"）**：
    · COCO（mean 7.4）      → 预算在几乎每张图上都松弛（`frac(nᵢ>100)` = 0.0%）
    · CrowdHuman（mean 23.5）→ 中间（2.1%）
    · VisDrone（mean 70.7）  → 咬合最强（19.5%）
    → **极值不预测**：CrowdHuman 的 `max`（391）> VisDrone（317），但 `frac` 只有 2.1% vs 19.5%。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
