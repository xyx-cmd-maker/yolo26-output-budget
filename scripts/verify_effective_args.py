"""
scripts/verify_effective_args.py
===============================
**从 checkpoint 读回"实际生效"的训练参数**，而不是相信配置文件。

═══════════════════════════════════════════════════════════════════════════
为什么需要（教训 **P31**，2026-10-15 实测）
═══════════════════════════════════════════════════════════════════════════
我把 `save_period: 10` 写进 r1 的 expcfg（23:04:10），而 **R1 进程在 ~23:02 已启动**
→ 加载的是旧配置 → 从 `last.pt` 读回的 `train_args.save_period` 是 **-1**（默认禁用）
→ **`epoch10.pt` / `epoch20.pt` 从未产生**，我此前"最坏损失降到 10 epoch"的说法**不成立**。

**根因**：**"改了配置文件" ≠ "运行中的进程用了新配置"。**
而我在报告里把"改了配置"当成了"参数已生效"。

**本脚本的作用**：把"参数是否生效"变成**可核验的一步**——
直接读 checkpoint 里的 `train_args`（Ultralytics 会把 `vars(self.args)` 存进去）。

用法：
    python scripts/verify_effective_args.py runs/train/<run_name>
    python scripts/verify_effective_args.py runs/train/<run_name> --expect save_period=10 workers=0
退出码：0 = 全部符合预期；1 = 有参数与预期不符。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

# 本项目**必须**生效的关键参数（依据见 AGENTS-LESSONS P12 / P21 / P31）
CRITICAL = {
    "workers": 0,        # P12：Windows 上 workers>0 会 spawn 死锁
    "save_period": 10,   # P31：周期性 checkpoint，崩溃时限制损失
}


def load_train_args(ckpt: Path) -> dict:
    import torch

    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    return obj.get("train_args") or {}, obj.get("epoch")


def main() -> int:
    ap = argparse.ArgumentParser(description="从 checkpoint 读回实际生效的参数")
    ap.add_argument("run_dir", type=Path, help="runs/train/<run_name> 或直接给 .pt")
    ap.add_argument("--ckpt", default="last.pt", help="用哪个 checkpoint（默认 last.pt）")
    ap.add_argument("--expect", nargs="*", default=None,
                    help="额外期望，形如 key=value（值会按 JSON 解析，失败则当字符串）")
    a = ap.parse_args()

    ckpt = a.run_dir if a.run_dir.suffix == ".pt" else a.run_dir / "weights" / a.ckpt
    if not ckpt.exists():
        print(f"❌ 找不到 checkpoint：{ckpt}")
        return 1

    args, epoch = load_train_args(ckpt)
    print("=" * 78)
    print(f"checkpoint: {ckpt}")
    print(f"记录时的 epoch: {epoch}   （Ultralytics 从 0 计数，故对应第 {None if epoch is None else epoch + 1} 个 epoch）")
    print("=" * 78)

    expect = dict(CRITICAL)
    if a.expect:
        for kv in a.expect:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            try:
                expect[k] = json.loads(v)
            except json.JSONDecodeError:
                expect[k] = v

    print(f"\n{'参数':<18} {'实测值':<14} {'期望值':<14} 判定")
    print("-" * 78)
    bad = 0
    for k, want in sorted(expect.items()):
        got = args.get(k, "<缺失>")
        ok = got == want
        if not ok:
            bad += 1
        print(f"{k:<18} {str(got):<14} {str(want):<14} {'✅' if ok else '❌ 不符'}")

    print(f"\n（共核 {len(expect)} 项，{bad} 项不符）")
    if bad:
        print("""
❌ **有参数未按预期生效**。最常见原因：
   · **该参数是在进程启动之后才写进配置文件的** → 运行中的进程读的是旧配置（P31）
   → 处置：不要用"改了配置"当作"已生效"；重启该次运行，或改用代码层强制（run.py 的兜底）。
""")
        return 1
    print("\n✅ 全部符合预期")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
