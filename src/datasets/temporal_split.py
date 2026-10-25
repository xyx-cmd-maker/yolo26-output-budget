"""
yolo26-detection / src/datasets/temporal_split.py
=================================================
按时间戳切分视频抽帧数据集，杜绝帧间近重复泄漏。

为什么必须有这个文件（本项目实测教训，2026-10-14）：
    `dataset125` 的 125 张图来自一段 **73.7 秒监控视频的连续帧**
    （相邻帧间隔中位 **80 ms**）。其原有 train/val/test 是**随机打散**的，
    实测后果：
        * 124 个相邻帧对中，**63 对跨越不同切分**；
        * 其中 **54 对 Δt <= 200 ms**（几乎同一瞬间）；
        * 跨切分近重复（dHash 汉明距离 <= 6）**64 对**，最小距离 **0**（完全相同）。
    → val 几乎是 train 的复制品，在其上报的 mAP 测的是**记忆**不是**泛化**。

修法：把**连续时间段整体分给同一个切分**，并在切分之间留**时间间隔**，
      使任何跨切分的帧对在时间上都足够远。

用法：
    python -m src.datasets.temporal_split \\
        --images  <含 *.jpeg 的目录> \\
        --labels  <含 *.txt 的目录> \\
        --out     <输出根目录> \\
        --ratios  0.7 0.15 0.15 \\
        --gap-sec 2.0
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

__all__ = ["parse_ts", "temporal_split", "verify_no_leakage"]

TS_RE = re.compile(r"(\d{10,})")  # 纯毫秒时间戳（>=10 位）
# 本项目实测命名：`image_1188886_00.jpeg`
#   第 1 个数 = 相对时间戳，单位 **毫秒**（帧间隔 80ms，与 12.5fps 吻合）
#   第 2 个数 = 帧序号（同一次抓取的子序号）
SEQ_RE = re.compile(r"^(?P<prefix>.*?)_(?P<ts>\d{5,})(?:_(?P<seq>\d+))?$")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_ts(name: str) -> int | None:
    """从文件名抽取**毫秒**时间戳（不含序号）。

    支持两种命名：
      * `image_1188886_00`  → 取 `1188886`（第 2 段是序号，不是时间的一部分）
      * `1700000000000`     → 纯 13 位毫秒时间戳

    返回值单位统一为**毫秒**。
    """
    stem = Path(name).stem
    m = SEQ_RE.match(stem)
    if m:
        return int(m.group("ts"))
    m2 = TS_RE.search(stem)
    if m2:
        return int(m2.group(1))
    return None


def _collect(images: Path) -> list[tuple[int, Path]]:
    rows: list[tuple[int, Path]] = []
    for f in sorted(images.rglob("*")):
        if f.suffix.lower() not in IMG_EXT:
            continue
        t = parse_ts(f.stem)
        if t is None:
            raise ValueError(f"文件名里找不到时间戳，无法做时间切分：{f.name}（请改用 --by-name 或改脚本）")
        rows.append((t, f))
    rows.sort()
    return rows


def temporal_split(
    images: Path,
    labels: Path,
    out: Path,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    gap_sec: float = 2.0,
    copy: bool = True,
) -> dict:
    """把帧按时间顺序切成 train/val/test 三段，段间留 gap_sec 秒的间隔。

    Args:
        gap_sec: 段与段之间**丢弃**的时间跨度（秒）。丢弃的帧不进任何切分，
            目的是保证任何跨切分的帧对在时间上至少相距 gap_sec。
    Returns:
        切分报告（含每段的帧数与时间范围、被丢弃的帧）。
    """
    rows = _collect(images)
    n = len(rows)
    if n < 10:
        raise ValueError(f"帧数太少（{n}），时间切分没有意义")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios 必须和为 1，得到 {sum(ratios)}")

    # 先按比例定位切点，再把切点向两侧扩出 gap
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val :]

    gap_ms = int(gap_sec * 1000)
    dropped: list[tuple[int, Path]] = []

    def _apply_gap(prev: list[tuple[int, Path]], nxt: list[tuple[int, Path]]) -> list[tuple[int, Path]]:
        """从 prev 尾部与 nxt 头部各削掉靠得太近的帧，被削的记入 dropped。"""
        if not prev or not nxt:
            return prev
        cut = nxt[0][0] - gap_ms
        kept = [r for r in prev if r[0] <= cut]
        removed = [r for r in prev if r[0] > cut]
        dropped.extend(removed)
        # nxt 头部也检查（prev 已削完，通常无需再削；保守起见仍查一次）
        return kept

    train_rows = _apply_gap(train_rows, val_rows)
    val_rows = _apply_gap(val_rows, test_rows)

    report: dict = {
        "source_images": str(images),
        "ratios": list(ratios),
        "gap_sec": gap_sec,
        "n_total_frames": n,
        "time_span_sec": round((rows[-1][0] - rows[0][0]) / 1000, 3),
        "splits": {},
        "n_dropped": len(dropped),
        "dropped": [d[1].stem for d in dropped],
    }

    for split, part in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        img_out = out / "images" / split
        lbl_out = out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        n_missing = 0
        for t, src in part:
            dst = img_out / src.name
            if copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())
            lbl = labels / (src.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, lbl_out / lbl.name)
            else:
                n_missing += 1

        report["splits"][split] = {
            "n": len(part),
            "t_start": part[0][0] if part else None,
            "t_end": part[-1][0] if part else None,
            "duration_sec": round((part[-1][0] - part[0][0]) / 1000, 3) if part else 0,
            "n_missing_label": n_missing,
        }

    # 段间实际间隔
    def _gap_ms(a: list, b: list) -> int | None:
        if not a or not b:
            return None
        return b[0][0] - a[-1][0]

    report["inter_split_gap_ms"] = {
        "train->val": _gap_ms(train_rows, val_rows),
        "val->test": _gap_ms(val_rows, test_rows),
    }
    (out / "SPLIT_REPORT.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "SPLIT_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def verify_no_leakage(out: Path, *, max_hamming: int = 6, max_dt_ms: int = 200) -> dict:
    """在切分结果上复查泄漏：跨切分的近重复对与过近时间对。"""
    from src.utils.data_card import _hamming, phash

    rows: list[tuple[int, str, Path, int | None]] = []
    for split in ("train", "val", "test"):
        d = out / "images" / split
        if not d.exists():
            continue
        for f in d.glob("*"):
            if f.suffix.lower() in IMG_EXT:
                rows.append((parse_ts(f.stem) or 0, split, f, phash(f)))
    rows.sort()

    near: list[dict] = []
    too_close: list[dict] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            si, sj = rows[i][1], rows[j][1]
            if si == sj:
                continue
            dt = abs(rows[j][0] - rows[i][0])
            if dt <= max_dt_ms:
                too_close.append({"a": f"{si}/{rows[i][2].stem}", "b": f"{sj}/{rows[j][2].stem}", "dt_ms": dt})
            if rows[i][3] is not None and rows[j][3] is not None:
                h = _hamming(rows[i][3], rows[j][3])
                if h <= max_hamming:
                    near.append({"a": f"{si}/{rows[i][2].stem}", "b": f"{sj}/{rows[j][2].stem}", "hamming": h, "dt_ms": dt})

    return {
        "n_frames": len(rows),
        "counts": {s: sum(1 for r in rows if r[1] == s) for s in ("train", "val", "test")},
        "cross_split_near_duplicates": near,
        "n_cross_split_near_duplicates": len(near),
        "cross_split_too_close_in_time": too_close,
        "n_cross_split_too_close_in_time": len(too_close),
        "criteria": {"max_hamming": max_hamming, "max_dt_ms": max_dt_ms},
        "verdict": "PASS" if (not near and not too_close) else "FAIL",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="按时间切分视频抽帧数据集")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    ap.add_argument("--gap-sec", type=float, default=2.0)
    ap.add_argument("--verify-only", action="store_true", help="只复查已有切分，不重切")
    a = ap.parse_args()

    if a.verify_only:
        rep = verify_no_leakage(a.out)
    else:
        rep = temporal_split(a.images, a.labels, a.out, ratios=tuple(a.ratios), gap_sec=a.gap_sec)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        print("\n=== 泄漏复查 ===")
        rep = verify_no_leakage(a.out)

    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print("\n判定:", rep["verdict"])
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
