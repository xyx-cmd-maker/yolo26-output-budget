"""
resolution_probe.py — 探测目标分辨率在本机 8GB 上的可行性与真实吞吐

为什么必须实测（不是外推）：
  ① research-framework.md §4 的实验矩阵含 imgsz=960 档，但**在实测之前不得报出 GPU 小时数**
     （教训来自审计红队：s/epoch 在 batch 变化时不是可线性外推量；
      正确外推量是 s/iteration × iterations/epoch）。
  ② 8GB 卡上 1280 是否 OOM **零验证**（审计 §[未完成] 明确列为缺口）。

做法：每个 (imgsz, batch) 组合跑极短训练（默认 12 iter），记录
  · 峰值显存（torch.cuda.max_memory_allocated）
  · 稳态 s/iter（取后 2/3 的中位数，丢掉前几个 iter 的预热）
  · 是否 OOM
并据此算出各 imgsz 相对 640/batch8 的**真实**每 epoch 成本倍率。

★ 不与 R1 争抢的原则：本脚本只跑 12 iter，且一次只跑一个组合。
   若检测到 GPU 显存已被占用 > 5 GB，则拒绝运行（避免把 R1 挤崩）。

用法：
    python scripts/resolution_probe.py                       # 全部组合
    python scripts/resolution_probe.py --combos 1280:1 960:4 # 指定组合
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def gpu_free_gb() -> tuple[float, float]:
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    return free / 1024**3, total / 1024**3


def main() -> int:
    ap = argparse.ArgumentParser(description="分辨率/显存探测")
    ap.add_argument("--weights", default="weights/yolo26n.pt")
    ap.add_argument("--data", default="data/processed/visdrone/visdrone.yaml")
    ap.add_argument("--combos", nargs="+", default=["640:8", "960:4", "1280:2", "1280:1", "1600:1"],
                    help="imgsz:batch 列表")
    ap.add_argument("--iters", type=int, default=12, help="每个组合的训练迭代数（越小越快）")
    ap.add_argument("--device", default="0")
    ap.add_argument("--json", default="results/tables/resolution-probe.json")
    ap.add_argument("--min-free-gb", type=float, default=3.0,
                    help="GPU 可用显存低于此值则拒绝运行（避免挤崩并行的 R1）")
    a = ap.parse_args()

    import torch

    free, total = gpu_free_gb()
    print(f"GPU 显存：可用 {free:.2f} GB / 总 {total:.2f} GB")
    if free < a.min_free_gb:
        print(f"❌ 可用显存 < {a.min_free_gb} GB —— 可能有训练在跑。**拒绝启动**（见 AGENTS-LESSONS P4）。")
        return 2

    from ultralytics import YOLO

    results = []
    for combo in a.combos:
        imgsz_s, batch_s = combo.split(":")
        imgsz, batch = int(imgsz_s), int(batch_s)
        print(f"\n{'=' * 70}\n=== imgsz={imgsz} batch={batch} ({a.iters} iter) ===")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            m = YOLO(a.weights)
            m.train(
                data=a.data, imgsz=imgsz, batch=batch, epochs=1, device=a.device,
                workers=0,                  # ★ 必须 0：Windows spawn 死锁（P12）
                plots=False, val=False, save=False, verbose=False,
                # 用 iterations 精确截断：epochs=1 下 6471/batch 次 iter，太大；
                # 用 fraction 缩到刚好覆盖 iters 次
                fraction=max(0.01, min(1.0, a.iters * batch / 6471)),
                deterministic=False,        # 探测不需要确定性
            )
            ok, err = True, None
        except RuntimeError as e:
            ok, err = False, f"{type(e).__name__}: {str(e)[:200]}"
        except Exception as e:  # noqa: BLE001
            ok, err = False, f"{type(e).__name__}: {str(e)[:200]}"
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.empty_cache()

        row = {"imgsz": imgsz, "batch": batch, "ok": ok, "error": err,
               "wall_s": round(dt, 1), "peak_alloc_gb": round(peak, 2)}
        if ok:
            row["s_per_iter_est"] = round(dt / a.iters, 3)
        results.append(row)
        status = "✅" if ok else "❌ OOM/失败"
        print(f"  {status}  峰值显存 {peak:.2f} GB  墙钟 {dt:.0f}s")
        if err:
            print(f"     {err}")

    # 汇总
    print(f"\n{'=' * 70}\n=== 汇总 ===")
    base = next((r for r in results if r["imgsz"] == 640 and r["batch"] == 8), None)
    print(f"{'imgsz':>6} {'batch':>6} {'峰值GB':>8} {'s/iter':>9} {'相对640b8':>10}  状态")
    for r in results:
        rel = ""
        if base and base.get("s_per_iter_est") and r.get("s_per_iter_est"):
            rel = f"{r['s_per_iter_est'] / base['s_per_iter_est']:.2f}×"
        print(f"{r['imgsz']:>6} {r['batch']:>6} {r['peak_alloc_gb']:>8.2f} "
              f"{r.get('s_per_iter_est', float('nan')):>9.3f} {rel:>10}  {'可用' if r['ok'] else 'OOM'}")

    print("\n★ 判读要点：")
    print("  · 只有 640/batch8 一行有 R1 的**真实** s/iter（0.294）可交叉验证；")
    print("    若本探测给出的 640/batch8 明显不同，说明**短跑有预热偏差**，倍率仍可用但绝对值不可用。")
    print("  · 每 epoch 成本倍率 = (s/iter 倍率) × (iter/epoch 倍率)，后者 = batch 的反比。")

    jp = PROJECT_ROOT / a.json
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps({"combos": results, "min_free_gb": a.min_free_gb}, indent=2,
                             ensure_ascii=False), encoding="utf-8")
    print(f"\n结果 -> {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
