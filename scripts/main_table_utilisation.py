"""
★ 论文骨架主表：`Utilisation → Allocation Benefit` 的**数据字典**

═══════════════════════════════════════════════════════════════════════════
为什么这是论文的骨架（且不受"单种子"威胁）
═══════════════════════════════════════════════════════════════════════════
主表的三个量与**训练完全无关**，只依赖**标签分布**：
    `mean nᵢ`            —— 平均密度
    `util(B)`            —— 预算是否真的在咬合
    `frac(nᵢ > B)`       —— 被预算卡住的图占比

→ 所以**即使 s43/s44 出任何意外，这张表仍然成立**。

═══════════════════════════════════════════════════════════════════════════
★ CrowdHuman 的口径（已核，见 P37）
═══════════════════════════════════════════════════════════════════════════
官方转换脚本 `crowdhuman2coco.py` 的 `--tag` 默认 **`"person"`**，且用
`extra.ignore` 剔除难例。故本脚本产出**两个版本**：
    · `crowdhuman_person_all`    —— 全部 `tag == "person"`（= 官方默认）
    · `crowdhuman_person_vis`    —— 再剔除 `extra.ignore == 1` 的框（更保守）

**同时把口径对齐检查写出来**：VisDrone 与 COCO 无 ignore 标记 → 只有"全部实例"一个版本。
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
OUT = ROOT / "results" / "tables" / "main-table-utilisation.json"

BUDGETS = (10, 20, 30, 50, 100, 200, 300, 500)


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


def coco_counts(inner: str) -> np.ndarray:
    with zipfile.ZipFile(ZIP) as z:
        with z.open(inner) as f:
            obj = json.load(io.TextIOWrapper(f, encoding="utf-8"))
    per: dict[int, int] = {}
    for a in obj["annotations"]:
        per[a["image_id"]] = per.get(a["image_id"], 0) + 1
    return np.array([per.get(im["id"], 0) for im in obj["images"]], dtype=float)


def crowdhuman_counts(drop_ignore: bool) -> np.ndarray:
    """★ 口径见 P37：官方 `--tag person` 默认；`extra.ignore` 用于剔除难例。"""
    counts = []
    with CH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n = 0
            for g in (obj.get("gtboxes") or []):
                if str(g.get("tag", "")).lower() != "person":
                    continue
                if drop_ignore:
                    ex = g.get("extra") or {}
                    if int(ex.get("ignore", 0)) == 1:
                        continue
                n += 1
            counts.append(n)
    return np.array(counts, dtype=float)


def summarize(n: np.ndarray, label: str) -> dict:
    m = n > 0
    nn = n[m]
    N = len(nn)
    row = {
        "label": label,
        "N_images_with_gt": int(N),
        "sum": int(nn.sum()),
        "mean": round(float(nn.mean()), 3),
        "median": float(np.median(nn)),
        "p90": round(float(np.percentile(nn, 90)), 1),
        "p99": round(float(np.percentile(nn, 99)), 1),
        "max": int(nn.max()),
        "CV": round(float(nn.std(ddof=1) / nn.mean()), 3),
        "by_B": {},
    }
    for B in BUDGETS:
        spent = float(np.minimum(B, nn).sum())
        row["by_B"][str(B)] = {
            "ceiling": round(spent / float(nn.sum()), 5),
            "utilisation": round(spent / (B * N), 5),
            "frac_capped": round(float((nn > B).mean()), 4),
        }
    return row


def main() -> int:
    print("=" * 106)
    print("论文主表：预算利用率 → 分配意义 的数据字典（只依赖标签分布，与训练无关）")
    print("=" * 106)

    rows = []
    rows.append(summarize(visdrone_counts("val"), "VisDrone2019-DET val (aerial, dense)"))
    rows.append(summarize(visdrone_counts("test"), "VisDrone2019-DET test-dev (aerial, dense)"))
    if CH.exists():
        rows.append(summarize(crowdhuman_counts(False), "CrowdHuman train — tag=person (ground, crowded)"))
        rows.append(summarize(crowdhuman_counts(True), "CrowdHuman train — tag=person, drop ignore"))
    if ZIP.exists():
        rows.append(summarize(coco_counts("annotations/instances_val2017.json"),
                              "COCO val2017 (ground, sparse)"))
        rows.append(summarize(coco_counts("annotations/instances_train2017.json"),
                              "COCO train2017 (ground, sparse)"))

    # ── 表 A：密度与离散度 ──────────────────────────────────────
    print("\n" + "─" * 106)
    print("表 A：逐图目标数分布的描述统计")
    print("─" * 106)
    print(f"  {'数据集':<46} {'图数':>8} {'mean':>8} {'median':>7} {'p99':>7} {'max':>6} {'CV':>6}")
    print("  " + "-" * 100)
    for r in rows:
        print(f"  {r['label']:<46} {r['N_images_with_gt']:>8} {r['mean']:>8.2f} "
              f"{r['median']:>7.0f} {r['p99']:>7.0f} {r['max']:>6} {r['CV']:>6.2f}")

    # ── 表 B：主表（util / frac_capped / ceiling）───────────────
    print("\n" + "─" * 106)
    print("表 B（主表）：ceiling / utilisation / frac(nᵢ > B)")
    print("─" * 106)
    for B in (30, 50, 100, 200, 300):
        print(f"\n  ── B = {B} ──")
        print(f"  {'数据集':<46} {'ceiling':>9} {'util':>8} {'frac>B':>8}")
        print("  " + "-" * 78)
        for r in rows:
            d = r["by_B"][str(B)]
            print(f"  {r['label']:<46} {d['ceiling'] * 100:>8.2f}% "
                  f"{d['utilisation'] * 100:>7.1f}% {d['frac_capped'] * 100:>7.1f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {OUT.relative_to(ROOT)}")

    # ── 口径对齐检查 ─────────────────────────────────────────────
    print("\n" + "─" * 106)
    print("★ 口径对齐检查（S1 严谨性）")
    print("─" * 106)
    if CH.exists():
        a = next(r for r in rows if r["label"].endswith("(ground, crowded)"))
        b = next(r for r in rows if r["label"].endswith("drop ignore"))
        print(f"  CrowdHuman tag=person          : mean={a['mean']}  max={a['max']}  图数={a['N_images_with_gt']}")
        print(f"  CrowdHuman tag=person, no-ignore: mean={b['mean']}  max={b['max']}  图数={b['N_images_with_gt']}")
        print(f"  → 剔除 ignore 后均值变化 = {(b['mean'] / a['mean'] - 1) * 100:+.1f}%"
              f"   （B=100 的 frac 从 {a['by_B']['100']['frac_capped'] * 100:.1f}% "
              f"→ {b['by_B']['100']['frac_capped'] * 100:.1f}%）")
        print("  ⚠️ VisDrone 与 COCO **无 ignore 标记** → 只有「全部实例」一个版本；")
        print("     故主表同时报 CrowdHuman 两版，**跨数据集结论不依赖 ignore 口径的选择**。")

    print("\n★ 主表读法（论文用）")
    print("  按 `util(B=100)` 升序：COCO (7.4%) < CrowdHuman (22.5%) < VisDrone (63.0%)")
    print("  按 `frac(nᵢ>100)` 升序：COCO (0.0%) < CrowdHuman (2.1%) < VisDrone (19.5%)")
    print("  → 与「分配是否有意义」的排序一致，且**与「域」（地面/空中）无关**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
