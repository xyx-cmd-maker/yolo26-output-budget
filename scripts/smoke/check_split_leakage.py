"""
scripts/smoke/check_split_leakage.py
===================================
强制手段：对所有数据集 yaml 做帧间泄漏复查，`verdict != PASS` 即失败。

为什么存在（AGENTS-PROJECT.md §5.1）：
    本项目实测——`dataset125` 的 125 帧来自 **73.7 秒连续视频**（相邻帧间隔中位 80ms），
    原随机切分导致 **124 个相邻帧对中 63 对跨切分**、**54 对 Δt≤200ms**、
    跨切分近重复（dHash ≤6）**64 对，最小距离 0**。
    在这样的切分上报的 mAP 测的是**记忆**不是**泛化**。

判据（三选一，按 yaml 内容自动选择）：
    1. yaml 的 path 下有 `SPLIT_REPORT.json` → 用 `temporal_split.verify_no_leakage()` 复查；
    2. yaml 声明了 `challenge: temporal`（或 `notes` 提到抽帧/视频）但**没有** SPLIT_REPORT → **FAIL**（未做时间切分）；
    3. 普通图像数据集 → 用 `data_card` 的 E4/E5 检查（跨切分完全重复 + 感知哈希近重复）。

退出码：0 = 全部 PASS；1 = 有数据集泄漏或未做时间切分
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CFG_DIR = PROJECT_ROOT / "src" / "configs" / "datasets"
SKIP = {"_TEMPLATE.yaml"}


def _load_yaml(p: Path) -> dict:
    import yaml

    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def check_one(cfg_path: Path) -> tuple[bool, str]:
    cfg = _load_yaml(cfg_path)
    root = Path(str(cfg.get("path", "")))
    if not root.is_absolute():
        root = (cfg_path.parent / root).resolve()

    blob = (str(cfg.get("notes", "")) + " " + str(cfg.get("challenge", ""))).lower()
    looks_temporal = any(k in blob for k in ("抽帧", "视频", "video", "frame", "temporal", "序列"))

    report = root / "SPLIT_REPORT.json"
    if report.exists():
        from src.datasets.temporal_split import verify_no_leakage

        rep = verify_no_leakage(root)
        ok = rep["verdict"] == "PASS"
        msg = (
            f"temporal_split 复查 {rep['verdict']}"
            f" | 帧数 {rep['counts']}"
            f" | 跨切分近重复 {rep['n_cross_split_near_duplicates']}"
            f" | 过近时间对 {rep['n_cross_split_too_close_in_time']}"
        )
        if not ok:
            ex = rep["cross_split_near_duplicates"][:3]
            msg += f"\n       例：{ex}"
        return ok, msg

    if looks_temporal:
        return False, (
            "yaml 表明这是抽帧/视频数据集，但 path 下**没有 SPLIT_REPORT.json** —— "
            "说明未做时间切分。请先跑：\n"
            f"         python -m src.datasets.temporal_split --images <images> --labels <labels> --out {root}"
        )

    # 普通图像数据集：用 data_card 的 E4/E5
    from src.utils.data_card import audit_dataset

    rep = audit_dataset(cfg_path, hash_images=True, phash_images=True)
    n_cross = len(rep.get("cross_split_duplicates") or [])
    n_near_cross = sum(1 for x in (rep.get("near_duplicates") or []) if x.get("cross_split"))
    ok = n_cross == 0 and n_near_cross == 0
    msg = f"data_card E4/E5 | 跨切分完全重复 {n_cross} | 跨切分近重复 {n_near_cross}"
    if not ok:
        msg += "  ← 存在跨切分泄漏"
    return ok, msg


def main() -> int:
    if not CFG_DIR.exists():
        print(f"❌ 配置目录不存在：{CFG_DIR}")
        return 1
    cfgs = sorted(p for p in CFG_DIR.glob("*.yaml") if p.name not in SKIP)
    if not cfgs:
        print("（没有数据集 yaml，跳过）")
        return 0

    print(f"泄漏复查 {len(cfgs)} 个数据集配置\n")
    bad = []
    for cfg in cfgs:
        try:
            ok, msg = check_one(cfg)
        except Exception as e:  # noqa: BLE001 —— 检查脚本自身出错也要报出来
            ok, msg = False, f"检查过程异常：{type(e).__name__}: {e}"
        print(f"  {'✅' if ok else '❌'} {cfg.name} —— {msg}")
        if not ok:
            bad.append(cfg.name)

    print()
    if bad:
        print(f"❌ {len(bad)} 个数据集未通过泄漏复查：{bad}")
        print("   按 AGENTS.md §1.2 红线 4，未通过者**不得用于训练**。")
        return 1
    print("✅ 全部数据集无跨切分泄漏")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
