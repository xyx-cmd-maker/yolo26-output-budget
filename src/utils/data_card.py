"""
yolo26-detection / src/utils/data_card.py
=========================================
数据集完整性 + 许可 + 泄漏体检。

为什么必须有这个文件：
    AGENTS.md §3 规定数据许可与切分泄漏是「发出去就收不回」的错误。
    人眼查几万张图的标注是不现实的，所以把它变成一条命令。

检查项（按严重度排序，全部**已实现**）：
  E1  路径缺失 / split 扫到 0 张图 / 图像与标签数量严重不匹配
  E2  标签越界（归一化坐标不在 [0,1]）或 w/h <= 0
  E3  类别 id 超出 nc 范围
  E4  ★ 重复图像：跨切分（真泄漏）与切分内部（削弱有效样本量）
  E5  ★ 近重复图像（感知哈希汉明距离 <= 阈值）—— 弱泄漏
  E6  空标签比例异常高（可能是转换脚本静默丢框）
  E7  极端长宽比 / 极小框占比（判断 P2 头是否需要）
  E8  nc 与 names 长度不一致；names 有重复名
  W1  许可字段缺失（无法商用/无法再分发）
  W2  类别分布极度不均衡
  W3  path 用了相对路径（会随启动目录漂移，见下）

★ 关于 path 的实测事实（2026-10-14，ultralytics 8.4.177）：
    `check_det_dataset` 的解析顺序是
        Path(extract_dir or data["path"] or yaml所在目录)
        if not path.exists() and not path.is_absolute(): path = DATASETS_DIR / path
    也就是说：**相对 path 若恰好存在于 CWD 下会被直接采用**，否则才落到
    `DATASETS_DIR`（本机实测 = `C:\\Users\\本手\\Documents\\深度学习\\datasets`）。
    → 相对路径会随启动目录漂移。**本项目规定 path 必须写绝对路径**（W3）。

用法：
    python -m src.utils.data_card --yaml src/configs/datasets/<name>.yaml
    python -m src.utils.data_card --yaml ... --json results/tables/<name>-datacard.json
    python -m src.utils.data_card --yaml ... --no-phash        # 跳过近重复检查（快）
    python -m src.utils.data_card --yaml ... --phash-threshold 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

__all__ = ["audit_dataset", "write_data_card", "phash"]

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 近重复判定阈值：64 位 dHash 的汉明距离。<=4 基本可判定为同一图的重编码/缩放。
PHASH_THRESHOLD = 6


def _load_yaml(p: Path) -> dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except ImportError:  # 退化到极简解析
        data: dict[str, Any] = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
        return data


def _imgs_in(split_path: Path) -> list[Path]:
    if not split_path.exists():
        return []
    return [p for p in split_path.rglob("*") if p.suffix.lower() in IMG_EXT]


def _label_for(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img.with_suffix(".txt")


def _file_hash(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def phash(p: Path, size: int = 8) -> int | None:
    """dHash：转灰度 → 缩到 (size+1)×size → 比较水平相邻像素 → 64 位整数。

    不依赖 numpy/PIL 之外的任何东西；PIL 缺失时返回 None（降级而非报错）。
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(p) as im:
            im = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
            px = list(im.getdata())
    except Exception:
        return None
    bits = 0
    idx = 0
    for r in range(size):
        for c in range(size):
            left = px[r * (size + 1) + c]
            right = px[r * (size + 1) + c + 1]
            bits |= (1 if left > right else 0) << idx
            idx += 1
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def audit_dataset(
    yaml_path: Path,
    *,
    hash_images: bool = True,
    phash_images: bool = False,
    phash_threshold: int = PHASH_THRESHOLD,
    max_hash: int = 0,
) -> dict[str, Any]:
    """对一份 YOLO 格式数据集做体检，返回结构化报告。

    Args:
        yaml_path: 数据集 yaml 路径。
        hash_images: 内容哈希（E4）。大库会慢，但这是唯一可靠的完全重复检测方式。
        phash_images: 感知哈希（E5，近重复）。更慢；大库建议抽样或跳过。
        phash_threshold: 汉明距离阈值，<= 该值判为近重复。
        max_hash: 限制哈希图像数；0 = 全部。仅用于快速冒烟。
    """
    yaml_path = Path(yaml_path).resolve()
    cfg = _load_yaml(yaml_path)
    raw_path = cfg.get("path")
    root = Path(raw_path) if raw_path else None
    if root is not None and not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    report: dict[str, Any] = {
        "yaml": str(yaml_path),
        "root": str(root) if root else None,
        "path_is_absolute": bool(raw_path) and Path(str(raw_path)).is_absolute(),
        "nc": cfg.get("nc"),
        "names": cfg.get("names"),
        "license": cfg.get("license"),
        "source_url": cfg.get("source_url"),
        "redistributable": cfg.get("redistributable"),
        "errors": [],
        "warnings": [],
        "splits": {},
    }

    # ---- W1 许可与溯源 ----
    if not report["license"]:
        report["warnings"].append("W1 缺 license 字段：无法证明可再分发/可商用")
    if not report["source_url"]:
        report["warnings"].append("W1 缺 source_url 字段：来源不可追溯")
    if report["redistributable"] is None:
        report["warnings"].append("W1 缺 redistributable 字段：再分发边界未声明")

    # ---- W3 path 必须绝对 ----
    if root is None:
        report["warnings"].append(
            "W3 yaml 未写 path：ultralytics 会退回到 yaml 所在目录，data_card 与训练可能不一致"
        )
    elif not report["path_is_absolute"]:
        report["warnings"].append(
            f"W3 path 是相对路径（{raw_path}）：ultralytics 8.4.177 会先按 CWD 试解析、"
            f"失败才落到 DATASETS_DIR —— 启动目录一变数据根就变。本项目规定写绝对路径。"
        )

    # ---- E8 nc 与 names ----
    names = report["names"]
    if isinstance(names, dict):
        n_names = len(names)
        vals = [str(v) for v in names.values()]
        if len(set(vals)) != len(vals):
            dups = [k for k, c in Counter(vals).items() if c > 1]
            report["errors"].append(f"E8 names 有重复类别名：{dups}")
    elif isinstance(names, list):
        n_names = len(names)
        if len(set(names)) != len(names):
            report["errors"].append("E8 names 列表有重复类别名")
    else:
        n_names = None
        report["errors"].append("E8 names 字段缺失或类型不对")
    if n_names is not None and report["nc"] is not None and int(report["nc"]) != n_names:
        report["errors"].append(f"E8 nc={report['nc']} 与 names 长度 {n_names} 不一致")
    if n_names is not None:
        report["n_classes"] = n_names

    # ---- 逐 split ----
    hashes: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))  # hash -> split -> [name]
    phashes: list[tuple[int, str, str]] = []  # (phash, split, name)
    n_hashed = 0

    for split in ("train", "val", "test"):
        raw = cfg.get(split)
        if not raw:
            continue
        sp = Path(raw)
        if not sp.is_absolute():
            base = root if root is not None else yaml_path.parent
            sp = (base / raw).resolve()
        imgs = _imgs_in(sp)

        s: dict[str, Any] = {"path": str(sp), "exists": sp.exists(), "n_images": len(imgs)}
        if not sp.exists():
            report["errors"].append(f"E1 {split} 路径不存在：{sp}")
            report["splits"][split] = s
            continue
        if not imgs:
            report["errors"].append(
                f"E1 {split} 路径存在但扫到 0 张图像（扩展名白名单：{sorted(IMG_EXT)}）。"
                f"若该字段指向图片清单 .txt 或目录层级写错，训练会静默用错数据。"
            )
            report["splits"][split] = s
            continue

        cls_counter: Counter[int] = Counter()
        n_boxes = 0
        n_missing_label = 0
        n_empty_label = 0
        n_bad_box = 0
        n_bad_cls = 0
        areas: list[float] = []
        ars: list[float] = []

        for img in imgs:
            # ★ 修 E4：哈希必须在"标签是否为空"之前做，否则背景图完全不参与泄漏检查
            if hash_images and (not max_hash or n_hashed < max_hash):
                try:
                    hashes[_file_hash(img)][split].append(img.name)
                    n_hashed += 1
                except OSError:
                    pass
            if phash_images and (not max_hash or len(phashes) < max_hash):
                ph = phash(img)
                if ph is not None:
                    phashes.append((ph, split, img.name))

            lbl = _label_for(img)
            if not lbl.exists():
                n_missing_label += 1
                continue
            lines = [ln.strip() for ln in lbl.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
            if not lines:
                n_empty_label += 1
                continue
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    n_bad_box += 1
                    continue
                try:
                    cid = int(float(parts[0]))
                    x, y, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    n_bad_box += 1
                    continue
                if report["nc"] is not None and not (0 <= cid < int(report["nc"])):
                    n_bad_cls += 1
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    n_bad_box += 1
                cls_counter[cid] += 1
                n_boxes += 1
                areas.append(w * h)
                ars.append(max(w / h, h / w) if h > 0 and w > 0 else 0.0)

        n_no_label = n_missing_label + n_empty_label
        s.update(
            n_boxes=n_boxes,
            n_missing_label_file=n_missing_label,
            n_empty_label_file=n_empty_label,
            no_label_ratio=round(n_no_label / len(imgs), 4),
            n_bad_box=n_bad_box,
            n_bad_cls=n_bad_cls,
            class_hist={int(k): v for k, v in sorted(cls_counter.items())},
            box_area_median=round(sorted(areas)[len(areas) // 2], 6) if areas else None,
            tiny_box_ratio=round(sum(1 for a in areas if a < 0.001) / max(len(areas), 1), 4) if areas else None,
            aspect_ratio_p95=round(sorted(ars)[int(len(ars) * 0.95)], 3) if ars else None,
        )

        if n_bad_box:
            report["errors"].append(f"E2 {split} 有 {n_bad_box} 个越界/非法框")
        if n_bad_cls:
            report["errors"].append(f"E3 {split} 有 {n_bad_cls} 个类别 id 超出 nc={report['nc']}")
        # ★ 修 E1：标签缺失比例过高是真问题（标注没准备好），必须报 error 而非沉默
        if n_missing_label / len(imgs) > 0.02:
            report["errors"].append(
                f"E1 {split} 有 {n_missing_label}/{len(imgs)} 张图找不到标签文件（{n_missing_label/len(imgs):.1%}）"
            )
        if n_empty_label / len(imgs) > 0.2:
            report["warnings"].append(
                f"E6 {split} 空标签文件比例 {n_empty_label/len(imgs):.1%} 偏高：确认这些是真实背景图，"
                f"而不是转换脚本静默丢框（见 AGENTS.md 技术债 D3）"
            )
        if n_boxes == 0:
            report["errors"].append(f"E1 {split} 一张框都没有：要么标注没转成功，要么路径指错")
        report["splits"][split] = s

    # ---- E4 完全重复 ----
    cross = [{"hash": h[:16], "splits": dict(v)} for h, v in hashes.items() if len(v) > 1]
    intra: list[dict[str, Any]] = []
    for h, v in hashes.items():
        for sp, names in v.items():
            if len(names) > 1:
                intra.append({"hash": h[:16], "split": sp, "names": names[:5], "n": len(names)})
    report["cross_split_duplicates"] = cross
    report["intra_split_duplicates"] = intra
    report["n_hashed_images"] = n_hashed
    if cross:
        report["errors"].append(
            f"E4 发现 {len(cross)} 组图像在不同切分间完全相同 —— 数据泄漏，必须先去重再训练"
        )
    if intra:
        report["warnings"].append(
            f"E4 发现 {len(intra)} 组图像在同一 split 内完全重复 —— 不会泄漏，但会虚高该 split 的权重"
        )

    # ---- E5 近重复 ----
    if phash_images:
        near: list[dict[str, Any]] = []
        n = len(phashes)
        for i in range(n):
            hi, si, ni = phashes[i]
            for j in range(i + 1, n):
                hj, sj, nj = phashes[j]
                d = _hamming(hi, hj)
                if d <= phash_threshold:
                    near.append({"a": f"{si}/{ni}", "b": f"{sj}/{nj}", "distance": d, "cross_split": si != sj})
        report["near_duplicates"] = near[:200]
        report["n_near_duplicates"] = len(near)
        n_cross_near = sum(1 for x in near if x["cross_split"])
        if n_cross_near:
            report["errors"].append(
                f"E5 发现 {n_cross_near} 对近重复图像跨越不同切分（汉明距离 <= {phash_threshold}）—— 弱泄漏"
            )
        elif near:
            report["warnings"].append(f"E5 发现 {len(near)} 对近重复图像，但都在同一 split 内")

    # ---- W2 类别不均衡 ----
    all_hist: Counter[int] = Counter()
    for s in report["splits"].values():
        all_hist.update(s.get("class_hist", {}))
    if all_hist:
        report["class_hist_total"] = {int(k): v for k, v in sorted(all_hist.items())}
        cmin, cmax = min(all_hist.values()), max(all_hist.values())
        report["class_imbalance_ratio"] = round(cmax / max(cmin, 1), 2)
        if cmin == 0 or cmax / max(cmin, 1) > 20:
            report["warnings"].append(
                f"W2 类别极不均衡（最多/最少 = {cmax}/{cmin}）："
                f"考虑分层切分、重采样或按类报 AP，不要只报总 mAP"
            )

    return report


def write_data_card(report: dict[str, Any], out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO 数据集体检")
    ap.add_argument("--yaml", required=True, type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--no-hash", action="store_true", help="跳过内容哈希（快，但查不出 E4）")
    ap.add_argument("--phash", action="store_true", help="启用 E5 近重复检查（慢）")
    ap.add_argument("--phash-threshold", type=int, default=PHASH_THRESHOLD)
    ap.add_argument("--max-hash", type=int, default=0)
    a = ap.parse_args()

    rep = audit_dataset(
        a.yaml,
        hash_images=not a.no_hash,
        phash_images=a.phash,
        phash_threshold=a.phash_threshold,
        max_hash=a.max_hash,
    )
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if a.json:
        print("已写入", write_data_card(rep, a.json))
    if rep["errors"]:
        print(f"\n❌ 有 {len(rep['errors'])} 条 error。按 AGENTS.md §3，**不得**用这个数据集训练。")
        return 1
    print("\n✅ 无 error。可以进入训练流程。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
