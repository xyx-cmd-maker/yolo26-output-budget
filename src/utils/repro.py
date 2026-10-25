"""
yolo26-detection / src/utils/repro.py
=====================================
随机性与环境指纹的可复现控制。

设计原则（对应 AGENTS.md §2 数字纪律）：
  * 任何一次训练/评测都必须能回答：哪个种子、哪份代码、哪个环境、哪块硬件。
  * 因此这里提供 seed_everything() 与 env_fingerprint()，训练入口强制调用。

不做什么：不封装 ultralytics（它自己有 seed 参数）；只补齐它不管的部分。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["seed_everything", "env_fingerprint", "code_fingerprint", "git_rev"]


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """固定 python / numpy / torch / cuda 的随机源。

    Args:
        seed: 随机种子。必须显式传入，不设默认值——默认值会让"忘了设种子"变成静默事故。
        deterministic: 是否强制 cuDNN 确定性算法。开了会变慢，但能消除
            "同种子不同结果"的最后一类来源。论文级实验建议 True。

    Returns:
        传入的 seed，便于调用处 `seed = seed_everything(0)` 链式书写。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # 某些算子（如部分 upsample/interpolate 反传）没有确定性实现，
            # 打开下面这行会让它们直接报错而不是静默不确定——这是我们要的。
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=False)
            except Exception:
                # 老版本 torch 没有 use_deterministic_algorithms；不静默吞掉，打印出来。
                print("[repro] 警告：torch.use_deterministic_algorithms 不可用，确定性未完全保证", file=sys.stderr)
    except ImportError:
        pass

    return seed


def git_rev(repo: Path | None = None) -> dict[str, str]:
    """取当前 git 版本。仓库没初始化或没有 git 时返回明确标记，不抛异常。

    ★ 修 B4：原写法在**没有仓库**时也返回 dirty="clean"，把"没有版本控制"
      伪装成"工作区干净"。现在用 NO_GIT 显式区分，避免可追溯性字段说谎。
    """
    cwd = str(repo or Path(__file__).resolve().parents[2])

    def _run(args: list[str]) -> str:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=False
            )
            return out.stdout.strip() if out.returncode == 0 else ""
        except (FileNotFoundError, subprocess.SubprocessError):
            return ""

    commit = _run(["rev-parse", "HEAD"])
    if not commit:
        return {"commit": "NO_GIT", "branch": "NO_GIT", "dirty": "NO_GIT"}
    return {
        "commit": commit,
        "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]) or "NO_GIT",
        "dirty": "dirty" if _run(["status", "--porcelain"]) else "clean",
    }


def code_fingerprint(paths: list[Path] | None = None) -> str:
    """对源码文件集合做内容哈希（sha256，取前 16 位）。

    用途：论文里的数字必须能指回一份确定的代码。提交号可能被 rebase 掉，
    内容哈希不会。
    """
    root = Path(__file__).resolve().parents[1]  # -> src/
    targets = paths or sorted(p for p in root.rglob("*.py"))
    h = hashlib.sha256()
    for p in targets:
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def env_fingerprint() -> dict[str, object]:
    """采集环境指纹。写进每个结果文件，确保数字可追溯到确定的环境。"""
    fp: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "code_hash": code_fingerprint(),
        "git": git_rev(),
    }

    for mod, attr in (("torch", "__version__"), ("torchvision", "__version__"), ("ultralytics", "__version__")):
        try:
            m = __import__(mod)
            fp[mod] = getattr(m, attr, "unknown")
        except ImportError:
            fp[mod] = None

    try:
        import torch

        if torch.cuda.is_available():
            fp["cuda"] = torch.version.cuda
            fp["gpu"] = torch.cuda.get_device_name(0)
            fp["gpu_capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
            fp["gpu_total_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            fp["cudnn"] = torch.backends.cudnn.version()
            fp["arch_list"] = torch.cuda.get_arch_list()
        else:
            fp["cuda"] = None
            fp["gpu"] = "CPU_ONLY"
    except ImportError:
        pass

    return fp


def dump_fingerprint(out_path: Path) -> Path:
    """把环境指纹写到 JSON。训练入口调用一次即可。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(env_fingerprint(), indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    print(json.dumps(env_fingerprint(), indent=2, ensure_ascii=False))
