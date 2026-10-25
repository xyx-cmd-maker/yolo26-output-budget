"""
分块跑 budget_scan 并合并 per_image —— 绕过 VRAM 限制的正式解法。

═══════════════════════════════════════════════════════════════════════════
为什么需要（第 40 轮后实测）
═══════════════════════════════════════════════════════════════════════════
`--limit 548 --budgets ... 500` 在本机 **CUDA OOM**：
    torch.OutOfMemoryError: Tried to allocate 2.51 GiB（总容量 7.96 GiB）
`--limit 150 --budgets ... 300` 则 26 秒完成、无 OOM。

**但判据 v2 需要全量 548 图**：`worst` 是**尾部统计**，
子集的 GT 最大密度（186）低于全量（317）→ **子集的最坏图不代表全量的最坏图**。

═══════════════════════════════════════════════════════════════════════════
做法
═══════════════════════════════════════════════════════════════════════════
`budget_scan` 的 `--limit` 取"**排序后的前 N 张**"，因此第 k 块的起止是已知的。
但 `budget_scan` 不支持 `--offset`，所以**由本脚本承担分块**：
  1. 枚举当前 split 的全部图片（按 `budget_scan.load_split` 的同一排序）
  2. 按块切分，每块写一个**临时数据 yaml**，指向该块的图片
  3. 对每块调用 `--limit <块大小>`（因为 yaml 只含该块的图 → 顺序即全局顺序的连续段）
  4. 合并 `per_image`（按 name 去重并校验连续性）
  5. 校验：合并后的 `per_image` 条数 == 全量图数；且每条的 `n_gt` 与直接从标签算出的一致

**不修改 `budget_scan`**（它是已验证的工具）；本脚本只做调度与合并。

用法：
    python scripts/budget_scan_chunked.py --split val --head o2m --chunk 150
    python scripts/budget_scan_chunked.py --split test --head o2m --chunk 150
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
DEFAULT_WEIGHTS = (PROJECT_ROOT / "runs" / "train"
                   / "r1_yolo26n_visdrone_640__s42" / "weights" / "best.pt")
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "visdrone" / "visdrone.yaml"
BUDGETS = [10, 20, 30, 50, 100, 300]


def list_images(data_yaml: Path, split: str) -> list[Path]:
    """与 `budget_scan.load_split` **同一排序**（sorted + rglob）——必须一致，否则块边界错位。"""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    d = Path(cfg[split])
    if not d.is_absolute():
        d = (root / d).resolve()
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(f for f in d.rglob("*") if f.suffix.lower() in exts)


def gt_counts(data_yaml: Path, split: str, imgs: list[Path]) -> dict[str, int]:
    """直接从标签文件算逐图 GT 数（用于**独立**校验合并结果）。"""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    d = Path(cfg[split])
    if not d.is_absolute():
        d = (root / d).resolve()
    parts = list(d.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    lbl = Path(*parts)
    out = {}
    for im in imgs:
        lb = lbl / (im.stem + ".txt")
        n = 0
        if lb.exists():
            n = sum(1 for ln in lb.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
        out[im.name] = n
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="分块跑 budget_scan 并合并")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--split", default="val")
    ap.add_argument("--head", default="o2m", choices=["o2m", "o2o"])
    ap.add_argument("--exp", default="r1")
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    data_yaml = Path(a.data)
    imgs = list_images(data_yaml, a.split)
    N = len(imgs)
    n_chunks = (N + a.chunk - 1) // a.chunk
    print("=" * 92)
    print(f"分块 budget_scan：split={a.split}  N={N}  chunk={a.chunk}  → {n_chunks} 块")
    print(f"  head={a.head}  budgets={a.budgets}  max_budget_run={max(a.budgets)}")
    print("=" * 92)

    tmp = Path(tempfile.mkdtemp(prefix="bscan_"))
    merged: dict[str, dict] = {}
    curves = []
    try:
        for k in range(n_chunks):
            seg = imgs[k * a.chunk:(k + 1) * a.chunk]
            # 写一个只含该块图片的临时数据目录结构：用软链不稳，改为写"图片清单 yaml"
            # budget_scan 的 load_split 从 yaml 的 path/split 解析目录，故改为：
            # 建一个临时目录，把该块的图**硬链**进去（同盘硬链，零拷贝）。
            seg_dir = tmp / f"seg{k}" / "images"
            seg_dir.mkdir(parents=True, exist_ok=True)
            for im in seg:
                try:
                    (seg_dir / im.name).hardlink_to(im)
                except OSError:
                    shutil.copy2(im, seg_dir / im.name)
            # 标签目录用符号方式：把 labels 也硬链
            cfg0 = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
            root0 = Path(cfg0["path"])
            if not root0.is_absolute():
                root0 = (data_yaml.parent / root0).resolve()
            d0 = Path(cfg0[a.split])
            if not d0.is_absolute():
                d0 = (root0 / d0).resolve()
            p0 = list(d0.parts)
            for i in range(len(p0) - 1, -1, -1):
                if p0[i] == "images":
                    p0[i] = "labels"
                    break
            lbl0 = Path(*p0)
            seg_lbl = seg_dir.parent / "labels"
            seg_lbl.mkdir(parents=True, exist_ok=True)
            for im in seg:
                lb = lbl0 / (im.stem + ".txt")
                if lb.exists():
                    try:
                        (seg_lbl / lb.name).hardlink_to(lb)
                    except OSError:
                        shutil.copy2(lb, seg_lbl / lb.name)

            seg_yaml = tmp / f"seg{k}.yaml"
            # ★★ P70：临时 yaml 的 split 键**必须用请求的那个 split**，不能硬编码 train/val。
            #   此前写死 `{"train": "images", "val": "images"}`，于是 `--split test`
            #   一路走到 `budget_scan.load_split` 的 `cfg[split]` → **KeyError: 'test'**。
            #   它在 val 上一直能用**纯属巧合**（val 恰在硬编码里）。
            #   同时保留 train/val 作为别名，兼容任何按固定键读取的下游代码。
            seg_cfg = {"path": str(seg_dir.parent), "names": cfg0.get("names")}
            for _k in ("train", "val", "test"):
                seg_cfg[_k] = "images"
            seg_yaml.write_text(yaml.safe_dump(seg_cfg, allow_unicode=True), encoding="utf-8")

            print(f"\n  [{k + 1}/{n_chunks}] {len(seg)} 图 …", flush=True)
            cmd = [PY, "-m", "src.eval.budget_scan",
                   "--weights", a.weights, "--data", str(seg_yaml), "--split", a.split,
                   "--exp", f"{a.exp}_seg{k}", "--head", a.head,
                   "--budgets", *[str(b) for b in a.budgets], "--imgsz", str(a.imgsz)]
            r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=7200)
            if r.returncode != 0:
                print(f"     ❌ 块 {k} 失败：\n{(r.stderr or '')[-900:]}")
                return 1
            jp = PROJECT_ROOT / "runs" / "budget" / f"{a.exp}_seg{k}_{a.split}_{a.head}.json"
            if not jp.exists():
                print(f"     ❌ 块 {k} 无输出 {jp}")
                return 1
            obj = json.loads(jp.read_text(encoding="utf-8"))
            for rec in obj.get("per_image", []):
                merged[rec["name"]] = rec
            curves.append({"seg": k, "curve": obj.get("curve"), "gt_density": obj.get("gt_density"),
                           "infer_seconds": obj.get("infer_seconds")})
            print(f"     ✅ 累计 per_image {len(merged)} 条", flush=True)
    finally:
        pass

    # ── 独立性校验：合并结果的 n_gt 必须与直接从标签算出的一致 ──
    ref = gt_counts(data_yaml, a.split, imgs)
    bad = [(nm, merged[nm]["n_gt"], ref[nm]) for nm in merged if nm in ref and merged[nm]["n_gt"] != ref[nm]]
    missing = [p.name for p in imgs if p.name not in merged]

    print("\n" + "=" * 92)
    print("合并结果与独立性校验")
    print("=" * 92)
    print(f"  全量图数           : {N}")
    print(f"  合并 per_image 条数 : {len(merged)}")
    print(f"  缺失               : {len(missing)}{'  ' + str(missing[:5]) if missing else ''}")
    print(f"  n_gt 不一致        : {len(bad)}{'  ' + str(bad[:3]) if bad else ''}")
    ok = (len(merged) == N) and not bad
    print(f"  ★ 校验：{'✅ 通过' if ok else '❌ 未通过'}")

    out = PROJECT_ROOT / "runs" / "budget" / f"{a.exp}_{a.split}_{a.head}.json"
    payload = {
        "weights": a.weights, "data": a.data, "split": a.split, "head": a.head,
        "chunked": True, "chunk_size": a.chunk, "n_chunks": n_chunks,
        "budgets": a.budgets, "n_images": len(merged),
        "curves_per_segment": curves,
        "per_image": list(merged.values()),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  合并输出 -> {out}")
    if not ok:
        print("\n  ❌ 校验未通过 —— **不要把该文件用于判据**")
        return 1
    print("\n  ✅ 可用于判据：")
    print(f"     python scripts/run_decision_rule.py --exp {a.exp} --splits {a.split} --heads {a.head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
