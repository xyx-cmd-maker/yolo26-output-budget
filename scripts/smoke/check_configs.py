"""
scripts/smoke/check_configs.py
=============================
强制手段：把 `src/configs/experiments/*.yaml` 逐个用 `run.py --dry-run` 验一遍。

为什么存在（AGENTS-LESSONS.md P3）：
    实测教训——我用**合成夹具的配置**做冒烟，通过了；换成**真实 expcfg** 后
    立刻崩在 `TypeError: Model.train() got multiple values for keyword argument`。
    夹具不含那个键，所以回归没被抓住。**校验必须针对真实配置。**

退出码解析（run.py --dry-run 的约定）：
    0 = 配置可执行（注意：dry-run 也会因"记录已存在"返回 1，见下）
    1 = 记录已存在（**不算配置错误**，视为 PASS）
    2 = train_args 与显式传参撞名（**配置错误**）
    其它非 0 = 失败

退出码：0 = 所有配置通过；1 = 有配置不可用
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = Path(r"G:\conda-envs\yolo26train\python.exe")
CFG_DIR = PROJECT_ROOT / "src" / "configs" / "experiments"

# 视为 PASS 的退出码及其含义
PASS_CODES = {0: "配置可执行", 1: "记录已存在（dry-run 前置检查触发，非配置错误）"}
# 视为 FAIL 的退出码及其含义
FAIL_CODES = {2: "train_args 与显式传参撞名（AGENTS-LESSONS P3）"}


def check_one(cfg: Path) -> tuple[bool, str]:
    exe = str(PY) if PY.exists() else sys.executable
    cmd = [exe, "-m", "src.train.run", "--expcfg", str(cfg), "--dry-run"]
    env_patch = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    import os

    env = {**os.environ, **env_patch}
    try:
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8", env=env, timeout=180)
    except subprocess.TimeoutExpired:
        return False, "超时（180s）"

    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode in PASS_CODES:
        return True, PASS_CODES[r.returncode]
    if r.returncode in FAIL_CODES:
        return False, FAIL_CODES[r.returncode]
    tail = "\n".join(out.strip().splitlines()[-6:])
    return False, f"退出码 {r.returncode}\n{tail}"


def main() -> int:
    if not CFG_DIR.exists():
        print(f"❌ 配置目录不存在：{CFG_DIR}")
        return 1
    cfgs = sorted(p for p in CFG_DIR.glob("*.yaml") if not p.name.startswith("_"))
    if not cfgs:
        print(f"（{CFG_DIR} 下没有 .yaml，跳过）")
        return 0

    print(f"校验 {len(cfgs)} 个实验配置（{CFG_DIR.relative_to(PROJECT_ROOT)}）\n")
    bad: list[tuple[Path, str]] = []
    for cfg in cfgs:
        ok, msg = check_one(cfg)
        print(f"  {'✅' if ok else '❌'} {cfg.name} —— {msg}")
        if not ok:
            bad.append((cfg, msg))

    print()
    if bad:
        print(f"❌ {len(bad)} 个配置不可用：")
        for cfg, msg in bad:
            print(f"   - {cfg.name}: {msg}")
        return 1
    print("✅ 全部配置通过 dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
