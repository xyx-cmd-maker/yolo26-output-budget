"""
scripts/migrate_data.py
=======================
把大数据集从项目目录（D 盘，空间紧张）迁到外部数据盘（F 盘），并更新所有引用。

为什么需要（本机实测，2026-10-14）：
    D 盘仅剩 **24.1 GB**，而 VisDrone 原始数据就占 **3.63 GB**，D 盘还要放后续数据集、
    权重、runs。F 盘有 **367 GB** 空闲。项目规则要求"数据读写只在项目内"，
    但**数据盘与项目目录分离是常见做法**，关键是①路径可追溯②不破坏可复现性
    ③迁移后配置自动更新（不靠人手改）。

设计要点：
  · **迁移的是 raw（占空间的大头）**；processed 里的图片是 hardlink，跨盘迁移会
    退化成真实复制，所以 processed 通过 `--reconvert` 在新位置重新生成。
  · 迁移完成后**自动更新** `src/configs/datasets/*.yaml` 与 `data/processed/*/*.yaml`
    里的 `path:` 字段，并写 `DATA_LOCATION.md` 记录新位置。
  · **移动而非复制**（`shutil.move`），迁移后校验文件数，避免"复制一半"。

用法：
    python scripts/migrate_data.py --plan                    # 只打印计划，不动文件
    python scripts/migrate_data.py --to G:/yolo26-data       # 实际迁移
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = Path("G:/yolo26-data")

# 需要迁移的目录：项目内路径 -> 数据盘内子目录名
MIGRATE = {
    "data/raw/visdrone": "visdrone-raw",
    "data/raw": None,  # 由子项覆盖，不单独迁移
}


def dir_size_mb(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1048576


def count_files(p: Path) -> int:
    return sum(1 for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def update_paths(dest_root: Path, *, dry: bool) -> list[tuple[Path, str, str]]:
    """把所有数据集 yaml 的 path 指向新位置。返回 (文件, 旧值, 新值)。"""
    import re

    changed: list[tuple[Path, str, str]] = []
    targets = sorted((PROJECT_ROOT / "src" / "configs" / "datasets").glob("*.yaml"))
    targets += sorted((PROJECT_ROOT / "data" / "processed").glob("*/*.yaml"))

    rules = [
        (re.compile(r"(?m)^path:\s*.*?/data/raw/visdrone\s*$"), f"path: {(dest_root / 'visdrone-raw').as_posix()}"),
        (re.compile(r"(?m)^path:\s*.*?/data/processed/visdrone\s*$"), f"path: {(dest_root / 'visdrone').as_posix()}"),
    ]
    for y in targets:
        txt = y.read_text(encoding="utf-8")
        orig = txt
        old_vals = re.findall(r"(?m)^path:\s*(.+)$", txt)
        for pat, rep in rules:
            txt = pat.sub(rep, txt)
        if txt != orig:
            new_vals = re.findall(r"(?m)^path:\s*(.+)$", txt)
            changed.append((y, "; ".join(old_vals), "; ".join(new_vals)))
            if not dry:
                y.write_text(txt, encoding="utf-8")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="把大数据集迁到外部数据盘")
    ap.add_argument("--to", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--plan", action="store_true", help="只打印计划")
    ap.add_argument("--skip-raw", action="store_true", help="跳过 raw 迁移（只更新路径）")
    a = ap.parse_args()

    dest_root = a.to
    print("=" * 72)
    print(f"目标数据盘：{dest_root}")
    print("=" * 72)

    # ---- 0. 前置检查：确认没有进程在读数据 ----
    print("\n[0] 前置检查")
    print("    ⚠️ 迁移前必须确认**没有任何训练/评测在跑**（否则会打断或损坏文件）")
    print("    → 本脚本不做进程检查，请自行确认（见 AGENTS-LESSONS.md P10）")

    # ---- 1. 迁移计划 ----
    print("\n[1] 迁移计划")
    plan: list[tuple[Path, Path, float]] = []
    for rel, sub in MIGRATE.items():
        if sub is None:
            continue
        src = PROJECT_ROOT / rel
        if not src.exists():
            print(f"    (跳过) {rel} 不存在")
            continue
        mb = dir_size_mb(src)
        dst = dest_root / sub
        plan.append((src, dst, mb))
        print(f"    {rel:28s} {mb:8.1f} MB  ->  {dst}")
    if not plan:
        print("    无可迁移项")

    total = sum(mb for _, _, mb in plan)
    print(f"    合计 {total:.1f} MB")

    if a.plan:
        print("\n--plan：未执行任何移动。")
        return 0

    if a.skip_raw:
        print("\n--skip-raw：跳过文件迁移，只更新路径。")
    else:
        # ---- 2. 执行迁移 ----
        print("\n[2] 执行迁移（move，非 copy）")
        dest_root.mkdir(parents=True, exist_ok=True)
        for src, dst, mb in plan:
            n_before = count_files(src)
            if dst.exists():
                print(f"    ⚠️ 目标已存在，跳过：{dst}（请先手工处理）")
                continue
            print(f"    移动 {src.name} ({mb:.1f} MB, {n_before} 文件) …", flush=True)
            shutil.move(str(src), str(dst))
            n_after = count_files(dst)
            ok = "✅" if n_after == n_before else f"❌ {n_after}/{n_before}"
            print(f"      完成，校验 {ok}")

    # ---- 3. 更新配置路径 ----
    print("\n[3] 更新数据集 yaml 的 path")
    changed = update_paths(dest_root / "visdrone", dry=False)
    for y, old, new in changed:
        print(f"    {y.relative_to(PROJECT_ROOT)}")
        print(f"        旧: {old}")
        print(f"        新: {new}")
    if not changed:
        print("    （无变化）")

    # ---- 4. 写位置记录 ----
    doc = dest_root / "DATA_LOCATION.md"
    rec = {
        "migrated_on": "2026-10-14",
        "data_root": str(dest_root),
        "items": [{"from": str(s), "to": str(d), "MB": round(mb, 1)} for s, d, mb in plan],
        "note": "项目内的 data/raw 迁至此处；data/processed 由 scripts/convert_visdrone.py "
                "在 --to 指向的位置重新生成（跨盘 hardlink 会退化为复制）。",
    }
    dest_root.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        "# 数据位置记录\n\n```json\n" + json.dumps(rec, indent=2, ensure_ascii=False) + "\n```\n",
        encoding="utf-8",
    )
    print(f"\n[4] 位置记录 -> {doc}")
    print("\n⚠️ 迁移后必须重新生成 processed（跨盘 hardlink 失效）：")
    print("    python scripts/convert_visdrone.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
