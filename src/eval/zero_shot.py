"""
yolo26-detection / src/eval/zero_shot.py
========================================
零点基线（zero-shot）：用官方预训练权重，**不训练**，直接在我们的 val 上评测。

为什么单独一个入口（修 B3）：
    `src/train/run.py` 的路径必然是「训练 → 取 best.pt → 评测」，
    没有 best.pt 就走不通。而零点基线根本不该训练——它是"预训练先验在目标域
    还剩多少"的上界，是判断"值不值得训"的第一块砖。

用法：
    # 冒烟
    python -m src.eval.zero_shot --data coco8 --model yolo26n.pt --tier smoke

    # 正式：多尺度 × 多头 × 多权重
    python -m src.eval.zero_shot --data tt100k --tier paper \
        --models yolo26n.pt yolo26s.pt --imgsz 640 --seeds 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.console import setup_console  # noqa: E402

setup_console()  # 必须在任何 print 之前

from src.utils.repro import dump_fingerprint, env_fingerprint, seed_everything  # noqa: E402
from src.utils.results_io import TIERS, ResultRecord, write_record  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def eval_one(
    *,
    exp_id: str,
    data: str,
    ds_id: str,
    n_images: int | None,
    model_path: str,
    seed: int,
    tier: str,
    imgsz: int,
    batch: int,
    device: str,
    nms_free: bool,
    split: str,
    val_args: dict,
) -> ResultRecord:
    from ultralytics import YOLO

    seed_everything(seed, deterministic=True)

    run_name = f"{exp_id}__{Path(model_path).stem}__{'e2e' if nms_free else 'nms'}__s{seed}"
    out_dir = PROJECT_ROOT / "runs" / "zeroshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_fingerprint(out_dir / f"{run_name}__env.json")

    yolo = YOLO(model_path)  # 只加载，不 train
    t0 = time.time()
    metrics = yolo.val(
        data=data,
        imgsz=imgsz,
        batch=batch,
        device=device,
        split=split,
        nms=False if nms_free else None,
        project=str(out_dir),
        name=run_name,
        exist_ok=True,
        **val_args,
    )
    elapsed = time.time() - t0

    clean = {
        k.replace("metrics/", "").replace("(B)", ""): float(v)
        for k, v in metrics.results_dict.items()
        if isinstance(v, (int, float))
    }

    try:
        from ultralytics.utils.torch_utils import get_flops

        flops = float(get_flops(yolo.model, imgsz=imgsz))
    except Exception:
        flops = None
    try:
        import torch

        peak = round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else None
    except ImportError:
        peak = None

    # 权重哈希：零点基线的可追溯性全部压在这个文件上
    import hashlib

    wp = Path(model_path)
    if not wp.is_absolute():
        # ultralytics 会把裸文件名下到 CWD；从那里找
        cand = Path.cwd() / wp
        wp = cand if cand.exists() else wp
    w_hash = None
    if wp.exists():
        h = hashlib.sha256()
        with wp.open("rb") as f:
            while blk := f.read(1 << 20):
                h.update(blk)
        w_hash = h.hexdigest()

    return ResultRecord(
        exp_id=exp_id,
        dataset=ds_id,
        tier=tier,
        seed=seed,
        metrics=clean,
        n_images=n_images,
        imgsz=imgsz,
        conf=val_args.get("conf"),
        iou=val_args.get("iou"),
        max_det=val_args.get("max_det"),
        batch=batch,
        optimizer="NONE_zero_shot",
        split=split,
        weights=str(wp),
        weights_sha256=w_hash,
        eval_seconds=round(elapsed, 3),
        notes=f"zero-shot; nms_free={nms_free}",
        fingerprint=env_fingerprint(),
        extra={
            "speed_ms_per_image": getattr(metrics, "speed", {}) or {},
            "n_params": sum(p.numel() for p in yolo.model.parameters()),
            "gflops": flops,
            "peak_gpu_mem_gb": peak,
            "nms_free": nms_free,
            "no_training": True,
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="零点基线评测（不训练）")
    ap.add_argument("--exp", default="e01_zero_shot")
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", default=["yolo26n.pt"])
    ap.add_argument("--tier", default="smoke", choices=TIERS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--split", default="val")
    ap.add_argument("--nms-free", action="store_true", help="用 one-to-one 头（无 NMS）评测")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--max-det", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from src.train.run import count_split_images, dataset_id, resolve_data

    data = resolve_data(a.data)
    ds_id = dataset_id(data)
    n_imgs = count_split_images(data, a.split)
    val_args: dict = {}
    for k, v in (("conf", a.conf), ("iou", a.iou), ("max_det", a.max_det)):
        if v is not None:
            val_args[k] = v

    print(json.dumps({**vars(a), "resolved_data": data}, indent=2, ensure_ascii=False, default=str))
    if a.dry_run:
        print("--dry-run：未执行。")
        return 0

    failed: list[str] = []
    for mp in a.models:
        for s in a.seeds:
            try:
                rec = eval_one(
                    exp_id=a.exp,
                    data=data,
                    ds_id=ds_id,
                    n_images=n_imgs,
                    model_path=mp,
                    seed=int(s),
                    tier=a.tier,
                    imgsz=a.imgsz,
                    batch=a.batch,
                    device=a.device,
                    nms_free=a.nms_free,
                    split=a.split,
                    val_args=val_args,
                )
                p = write_record(rec, overwrite=a.overwrite)
                print(f"✅ {mp} seed={s} -> {p}  {rec.primary_metric}={rec.primary_value:.4f}")
            except FileExistsError as e:
                print(f"❌ 记录已存在，跳过：{e}")
                failed.append(f"{mp}#s{s}: FileExistsError")
            except Exception as e:  # noqa: BLE001
                print(f"❌ {mp} seed={s} 失败：{type(e).__name__}: {e}")
                failed.append(f"{mp}#s{s}: {type(e).__name__}: {e}")

    if failed:
        print("\n失败清单（不得选择性报告）：")
        for f in failed:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
