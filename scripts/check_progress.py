"""稳健地读训练进度（避免 PowerShell 字符串排序的坑，见 P25）"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")


def main() -> int:
    ap = argparse.ArgumentParser(description="读 runs/train/*/results.csv 报告进度")
    ap.add_argument("--run", default="r1_yolo26n_visdrone_640__s42")
    ap.add_argument("--tail", type=int, default=5)
    a = ap.parse_args()

    f = PROJECT / "runs" / "train" / a.run / "results.csv"
    if not f.exists():
        print(f"❌ 不存在：{f}")
        return 1
    with f.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("（尚无完成的 epoch）")
        return 0

    def col(name: str) -> list[float]:
        out = []
        for r in rows:
            v = (r.get(name) or "").strip()
            if v:
                try:
                    out.append(float(v))
                except ValueError:
                    pass
        return out

    m50 = col("metrics/mAP50(B)")
    m95 = col("metrics/mAP50-95(B)")
    times = col("time")
    n = len(rows)
    print(f"{a.run}")
    print(f"  已完成 epoch: {n}/100")
    if m50:
        bi = max(range(len(m50)), key=lambda i: m50[i])
        print(f"  最佳 mAP50    = {max(m50):.4f}  (epoch {bi + 1})")
        print(f"  末轮 mAP50    = {m50[-1]:.4f}")
    if m95:
        print(f"  最佳 mAP50-95 = {max(m95):.4f}")
        print(f"  末轮 mAP50-95 = {m95[-1]:.4f}")
    if len(times) >= 2:
        per = (times[-1] - times[0]) / max(len(times) - 1, 1)
        print(f"  {per:.0f} s/epoch  →  剩余约 {per * (100 - n) / 3600:.1f} 小时")
    print(f"\n  最近 {a.tail} 个 epoch：")
    for i in range(max(0, n - a.tail), n):
        print(f"    ep {i + 1:>3}: mAP50={m50[i]:.5f}  mAP50-95={m95[i]:.5f}"
              if i < len(m50) and i < len(m95) else f"    ep {i + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ── 手稿合规核验（第 54 轮新增：字数与数字溯源）────────────────────
import subprocess as _sp
for _s in ("scripts/count_manuscript_words.py",
           "scripts/verify_manuscript_numbers_against_source.py"):
    _p = _sp.run([sys.executable, str(ROOT / _s)], capture_output=True, text=True)
    _last = [l for l in _p.stdout.strip().split("\n") if l.strip()][-1:]
    print(f"  [{_s.split('/')[-1]}] {_last[0].strip() if _last else '(无输出)'}")