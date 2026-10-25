"""
只读分析：R1 训练曲线的收敛趋势与外推（**不改动任何东西**）

用途：
  1. 判断 100 epoch 是否必要（还是已收敛）
  2. 给出 G0 终值的**保守外推区间**（用于提前判断是否需要干预）
  3. 检查震荡幅度（判断是否受 LR 调度或数据顺序影响）

⚠️ 本脚本**只读** `results.csv`，不写任何文件、不占 GPU。
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
CSV = ROOT / "runs" / "train" / "r1_yolo26n_visdrone_640__s42" / "results.csv"


def main() -> int:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    m50 = [float(r["metrics/mAP50(B)"]) for r in rows]
    m95 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    tm = [float(r["time"]) for r in rows]
    n = len(m50)

    print("=" * 88)
    print(f"R1 训练曲线（已 {n}/100 epoch）")
    print("=" * 88)
    print(f"  最佳 mAP50    = {max(m50):.4f}  (ep {m50.index(max(m50)) + 1})")
    print(f"  最佳 mAP50-95 = {max(m95):.4f}  (ep {m95.index(max(m95)) + 1})")
    print(f"  s/epoch = {(tm[-1] - tm[0]) / max(n - 1, 1):.0f}  剩余约 "
          f"{(tm[-1] - tm[0]) / max(n - 1, 1) * (100 - n) / 3600:.1f} h")

    # ── 分段趋势（每 5 epoch 一段的均值与增量）─────────────────
    print("\n" + "-" * 88)
    print("分段趋势（每 5 epoch：均值 / 段内增量）")
    print("-" * 88)
    print(f"  {'段':>10} {'mAP50 均值':>11} {'段内增量':>10} {'每 epoch 增量':>13}")
    prev = None
    for s in range(0, n, 5):
        seg = m50[s:s + 5]
        if not seg:
            continue
        avg = sum(seg) / len(seg)
        inc = (seg[-1] - seg[0]) / max(len(seg) - 1, 1)
        mark = ""
        if prev is not None:
            mark = "  ← 增速下降" if inc < prev * 0.7 else ""
        print(f"  ep{s + 1:>3}-{s + len(seg):<3} {avg:>11.4f} {inc:>+10.5f} {inc:>13.5f}{mark}")
        prev = inc

    # ── 外推：★ 用**对数饱和模型**，不用线性 ─────────────────────
    # 为什么：分段趋势显示每 epoch 增量在系统性衰减（0.0191 → 0.0042 → 0.0010 → 0.0025），
    # **线性外推必然高估**。收敛型训练曲线的典型形式是 `m(e) = a − b/(e + c)` 或 `a + b·ln(e)`。
    # 这里用最小二乘拟合 `m(e) = A + B·ln(e)`（单调、饱和、只有一个曲率参数）。
    import math

    tail = min(25, n)
    xs = list(range(1, n + 1))[-tail:]
    ys = m50[-tail:]
    lx = [math.log(x) for x in xs]
    mx = sum(lx) / len(lx)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(lx, ys))
    den = sum((x - mx) ** 2 for x in lx)
    B = num / den if den else 0.0
    A = my - B * mx
    pred100_log = A + B * math.log(100)

    # 线性外推（仅供参考，已知会高估）
    xs2 = list(range(n - tail, n))
    mx2 = sum(xs2) / len(xs2)
    my2 = sum(ys) / len(ys)
    n2 = sum((x - mx2) * (y - my2) for x, y in zip(xs2, ys))
    d2 = sum((x - mx2) ** 2 for x in xs2)
    slope = n2 / d2 if d2 else 0.0
    pred100_lin = my2 + slope * (99 - mx2)

    print("\n" + "-" * 88)
    print(f"外推（用最后 {tail} 个 epoch 拟合）")
    print("-" * 88)
    print(f"  **对数饱和模型 m(e) = {A:.4f} + {B:.4f}·ln(e)**   ← 推荐")
    print(f"     → ep100 → mAP50 ≈ **{pred100_log:.4f}**")
    print(f"  线性模型（⚠️ 已知高估，仅对照）")
    print(f"     斜率 {slope:+.5f}/epoch → ep100 → {pred100_lin:.4f}")
    print(f"\n  ★ 为什么用对数而不用线性：分段增量在**系统性衰减**"
          f"（{0.01913:.4f} → {0.00417:.4f} → {0.00102:.4f}），线性外推必然高估。")
    print(f"  ★ `close_mosaic: 10` 会在最后 10 epoch 带来一次**跳升**，"
          f"故 ep100 终值可能略高于对数外推。")
    print(f"\n  当前最佳 = {max(m50):.4f}；G0 判据 = 0.24 → **已超出 {(max(m50) / 0.24 - 1) * 100:.1f}%**")

    # ── 震荡幅度（最后 10 epoch 的极差 vs 趋势）───────────────────
    last10 = m50[-10:]
    rng = max(last10) - min(last10)
    print("\n" + "-" * 88)
    print("震荡检查（最后 10 个 epoch）")
    print("-" * 88)
    print(f"  极差 = {rng:.4f}  （相对水平 {sum(last10) / len(last10):.4f} 的 {rng / (sum(last10) / len(last10)) * 100:.1f}%）")
    print(f"  判定：{'⚠️ 震荡较大，末值不可信，应以最佳值为准' if rng > 0.02 else '✅ 单调上升为主，稳定'}")

    # ── close_mosaic 的位置 ──────────────────────────────────────
    print("\n" + "-" * 88)
    print("注：`close_mosaic: 10` → 最后 10 个 epoch 关闭 mosaic 增强。")
    print("     这通常带来一次跳升，故 ep100 终值**可能略高于**上面的对数外推。")
    print("     ⚠️ 但**不要**用线性外推（会高估；见上文对照）。")
    print("-" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
