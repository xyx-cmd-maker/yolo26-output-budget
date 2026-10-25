"""
scripts/convert_visdrone.py
===========================
把 VisDrone 官方标注转成 YOLO 格式，**并按 Ultralytics 的类序映射**。

★ 为什么必须自己写（不能直接跑 VisDrone.yaml 的 download 脚本）：
  1. 它内含 `shutil.rmtree(dir / folder)`，会**删掉我们已解压的原始目录**；原始数据要留作审计。
  2. 它的 `cls = int(row[5]) - 1` 依赖原始 category 恰好是 1..10 连续整数；
     本机**实测**原始 category 取值是 **0..11 共 12 个**，直接减 1 会把
     category 0 与 11 错配到类别 0（pedestrian）——而 0 与 11 是**低置信类别**，
     这样会**污染 pedestrian**。必须显式映射。

★ 实测的类序冲突（2026-10-14，本机）：
    官网类序  : pedestrian, person, car, van, bus, truck, motor, bicycle, awning-tricycle, tricycle
    YAML 类序 : pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
    原始数据里 car=category 4（144,867 个，最多）；YAML 里 car=id 3。
    → 本脚本用**官方类序 → category** 映射到 **YAML 的 id 编号**，即：
         YAML id 0 pedestrian  <- category 1
         YAML id 1 people      <- category 2
         YAML id 2 bicycle     <- category 3
         YAML id 3 car         <- category 4
         YAML id 4 van         <- category 5
         YAML id 5 truck       <- category 6
         YAML id 6 tricycle    <- category 7
         YAML id 7 awning-tricycle <- category 8
         YAML id 8 bus         <- category 9
         YAML id 9 motor       <- category 10
      category 0 与 11 为官方**低置信/忽略类别**（不在 10 类评测内）→ 丢弃。

★ 官方评测规则（一并落实）：
  · `score`（第 5 列）= 0 表示 ignored region，**必须丢弃**；= 1 才评测。
  · truncation > 50% 的目标**跳过评测** → 这里选择**保留**（训练时有用），
    但在数据卡里记录 truncation 分布，供评测脚本决定是否过滤。
  · 几何非法（w<=0 / h<=0 / 坐标越界）的行丢弃并计数。

用法：
    python scripts/convert_visdrone.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "visdrone"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "visdrone"

# 官方类序（官网 evaluate 页原文顺序）→ category 编号
OFFICIAL_ORDER = [
    "pedestrian", "person", "car", "van", "bus", "truck",
    "motor", "bicycle", "awning-tricycle", "tricycle",
]
# 官方类序在原始标注里对应的 category 编号（本机实测：pedestrian=1, person=2, car=4, ...）
OFFICIAL_TO_CATEGORY = {
    "pedestrian": 1, "person": 2, "bicycle": 3, "car": 4, "van": 5,
    "truck": 6, "tricycle": 7, "awning-tricycle": 8, "bus": 9, "motor": 10,
}
# Ultralytics VisDrone.yaml 的 id 顺序（我们要输出的 id 编号）
YAML_ORDER = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
# 名称对齐：YAML 用 "people"，官方用 "person"
NAME_ALIAS = {"people": "person"}

# category -> YAML id
CATEGORY_TO_YAML_ID: dict[int, int] = {}
for yaml_id, yname in enumerate(YAML_ORDER):
    official_name = NAME_ALIAS.get(yname, yname)
    CATEGORY_TO_YAML_ID[OFFICIAL_TO_CATEGORY[official_name]] = yaml_id

SPLITS = {
    "VisDrone2019-DET-train": "train",
    "VisDrone2019-DET-val": "val",
    "VisDrone2019-DET-test-dev": "test",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="VisDrone -> YOLO 格式转换（含类序显式映射）")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="原始 VisDrone2019-DET-* 目录所在处")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出目录（可指向外部数据盘）")
    a = ap.parse_args()
    RAW: Path = a.raw
    OUT: Path = a.out
    if not RAW.exists():
        print(f"❌ 原始目录不存在：{RAW}")
        print("   若已迁移到数据盘，请用 --raw <新位置>")
        return 1
    print(f"原始目录：{RAW}")
    print(f"输出目录：{OUT}")

    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"mapping_category_to_yaml_id": CATEGORY_TO_YAML_ID, "yaml_order": YAML_ORDER, "splits": {}}

    from PIL import Image

    for src_name, split in SPLITS.items():
        src = RAW / src_name
        if not src.exists():
            print(f"[跳过] {src_name} 不存在")
            continue
        img_src = src / "images"
        ann_src = src / "annotations"
        img_dst = OUT / "images" / split
        lbl_dst = OUT / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        stat = Counter()
        class_hist: Counter[int] = Counter()
        trunc_hist: Counter[int] = Counter()
        n_img = 0

        for ann in sorted(ann_src.glob("*.txt")):
            img_path = img_src / (ann.stem + ".jpg")
            if not img_path.exists():
                stat["missing_image"] += 1
                continue
            W, H = Image.open(img_path).size
            lines: list[str] = []
            for ln in ann.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
                p = ln.split(",")
                if len(p) < 8:
                    stat["short_row"] += 1
                    continue
                x, y, w, h = (int(float(v)) for v in p[:4])
                score, cat, trunc = int(p[4]), int(p[5]), int(p[6])

                if score != 1:
                    stat["ignored_region"] += 1
                    continue
                if cat not in CATEGORY_TO_YAML_ID:
                    stat[f"unknown_category_{cat}"] += 1
                    continue
                if w <= 0 or h <= 0:
                    stat["nonpositive_wh"] += 1
                    continue
                # 裁剪到图像范围内（官方数据有少量越界）
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(W, x + w), min(H, y + h)
                if x1 <= x0 or y1 <= y0:
                    stat["out_of_frame"] += 1
                    continue
                cw, ch = x1 - x0, y1 - y0
                cx, cy = (x0 + cw / 2) / W, (y0 + ch / 2) / H
                lines.append(f"{CATEGORY_TO_YAML_ID[cat]} {cx:.6f} {cy:.6f} {cw / W:.6f} {ch / H:.6f}")
                class_hist[CATEGORY_TO_YAML_ID[cat]] += 1
                trunc_hist[trunc] += 1

            (lbl_dst / ann.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            # 图片用硬链接省空间；失败则复制
            dst = img_dst / img_path.name
            if not dst.exists():
                try:
                    dst.hardlink_to(img_path)
                except OSError:
                    import shutil

                    shutil.copy2(img_path, dst)
            n_img += 1

        report["splits"][split] = {
            "n_images": n_img,
            "n_boxes": sum(class_hist.values()),
            "class_hist": {YAML_ORDER[k]: v for k, v in sorted(class_hist.items())},
            "truncation_hist": dict(sorted(trunc_hist.items())),
            "skipped": dict(stat),
        }
        print(f"[{split:5s}] {n_img} 图 / {sum(class_hist.values())} 框 | 跳过 {dict(stat)}")

    yaml_path = OUT / "visdrone.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(YAML_ORDER))
    yaml_path.write_text(
        f"""# VisDrone2019-DET，由 scripts/convert_visdrone.py 转换（YOLO 格式 + Ultralytics 类序）
#
# ⚠️ 许可：CC BY-NC-SA 3.0，**仅学术用途，不得再分发**。
# ⚠️ 类序：本文件采用 **Ultralytics VisDrone.yaml 的 id 顺序**，与官网评测页的类序不同。
#    对齐关系（官方类序 → 原始 category → 本文件 id）：
#      pedestrian(1)->0  person(2)->1  bicycle(3)->2  car(4)->3  van(5)->4
#      truck(6)->5  tricycle(7)->6  awning-tricycle(8)->7  bus(9)->8  motor(10)->9
#    category 0 与 11 属官方忽略类别，已丢弃；score=0（ignored region）已丢弃。
name: visdrone2019det
path: {OUT.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 10
names:
{names_block}

license: CC-BY-NC-SA-3.0
source_url: https://github.com/VisDrone/VisDrone-Dataset
redistributable: false
notes: >
  官方 VisDrone2019-DET，6,471 train / 548 val / 1,610 test-dev。
  本目录只含转换后的 labels 与 images 硬链接，**不随论文再分发**。
  官方评测 primary metric 是 AP@[.50:.05:.95]（不是 mAP@50）。
""",
        encoding="utf-8",
    )
    (OUT / "CONVERT_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n数据集 yaml -> {yaml_path}")
    print(f"转换报告   -> {OUT / 'CONVERT_REPORT.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
