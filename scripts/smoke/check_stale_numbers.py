"""
scripts/smoke/check_stale_numbers.py
====================================
机器校验器：扫描项目里是否残留**已作废的数字与口径**。

为什么需要（教训 P13 / P2 的强制手段）：
    我曾在多处写下 "YOLO26n @640 batch8 ≈ 20 s/epoch" 并据此做了 12 倍错误的算力外推。
    事后虽然改了主文档，但**同类过期数字会散落在别处**（子代理报告、旧笔记、图表脚本）。
    靠人记住"哪些数字已作废"不可靠 → 必须让机器扫。

判据：**已作废的数字必须与"作废标记"同现，或根本不出现**。
      允许 `~~20 s/epoch~~`、`20 s（已作废）`、`原值 ... 20 秒` 这类**带留痕**的写法
      （留痕是好事：它记录了修正历史），但**不允许**裸用。

退出码：0 = 干净；1 = 发现裸用的过期数字。

用法：
    python scripts/smoke/check_stale_numbers.py
    python scripts/smoke/check_stale_numbers.py --root .        # 指定扫描根
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── 已作废的口径（每项：标识、正则、说明）──────────────────────────────
# 每条正则会匹配"疑似的裸用"；随后再用 RETIRED_MARKERS 判断是否带留痕。
STALE = {
    "旧的 20 s/epoch 吞吐": {
        # ★ 2026-10-14 修正：原正则 `20\s*(?:秒|s)\s*/\s*epoch` 要求单位后**紧跟** /epoch，
        #   漏掉了"20 秒每 epoch""约 20 秒 1 个 epoch"这类写法（阳性对照实测抓到该漏洞）。
        #   现在覆盖 4 种写法：20秒/epoch、20 s/epoch、20秒每epoch、20秒跑完一个epoch。
        "pattern": re.compile(
            r"20\s*(?:秒|s|sec(?:ond)?s?)\s*(?:/|每|per)?\s*(?:epoch|轮|个\s*epoch|跑完)"
        ),
        "why": "来自 imgsz=64 合成夹具冒烟，与 640px/VisDrone 实测（238 s/epoch）差 12 倍。见 P13。",
    },
    "旧的自建数据集 test 规模（18 图 / 40 框）": {
        "pattern": re.compile(r"test\s*仅\s*18\s*图|18\s*图\s*/\s*40\s*框|test 18 图"),
        "why": "那是已作废的 dataset125。现行 VisDrone test-dev = 1,610 图 / 75,102 框。",
    },
    "workers ≤ 2 的旧资源纪律": {
        "pattern": re.compile(r"workers\s*[≤<]\s*2|workers\s*不超过\s*2"),
        "why": "实测 workers=2 在 Windows 上会 spawn 死锁；现行规则是 **一律 workers: 0**。见 P12。",
    },
    "旧的 3 类口径（交通标志/行人/信号灯）": {
        "pattern": re.compile(r"本项目\s*3\s*类|3\s*类、目标为.*交通标志"),
        "why": "现行数据集 VisDrone 为 10 类。见 src/configs/datasets/visdrone.yaml。",
    },
    "旧的 1.33 GB 峰值显存": {
        "pattern": re.compile(r"1\.33\s*GB"),
        "why": "那是 imgsz=64 冒烟值；640/batch8 实测约 3.46 GB。见 P13。",
    },
}

# 出现这些标记 → 视为**带留痕**，不算裸用
RETIRED_MARKERS = re.compile(
    r"作废|已废|旧的|原值|修正|错\s*12\s*倍|~~|已更新|P1[23]|实测|不再|历史",
)

# ── 假阳性豁免（**必须逐条给出理由**，不得图省事批量加）─────────────────
# 每条：正则 + 为什么它是**合法**的（在说别的东西，不是在用旧口径描述当前状态）
FALSE_POSITIVE_EXEMPT = [
    {
        "pattern": re.compile(r"test\s*18\s*图|val\s*仅\s*7\s*图"),
        "why": "`dataset125` **时间切分后**的正确描述（val 7 图 / test 18 图）。"
               "该数据集确实只有 5 个场景，是 `tier=pilot` 的**理由**，不是过期数字。"
               "见 docs/decisions.md D5.1、src/configs/datasets/roadside125.yaml。",
    },
    {
        "pattern": re.compile(r"workers\s*[≤<]\s*2|workers\s*不超过\s*2"),
        "why": "出现在**解释为何改为 0 的注释/教训**里（原文即含『旧规则』『实测会死锁』）。"
               "这是修正留痕，不是现行纪律。见 AGENTS-LESSONS P12。",
    },
    {
        "pattern": re.compile(r"dataset125|roadside125|自建集|pilot"),
        "why": "自建集**仍然存在且仍作 pilot 使用**（管线验证、零点基线、泄漏案例）。"
               "凡在讲该数据集自身规模的表述都是合法的，只要**不冒称它是当前正式数据集**。",
    },
]

# 扫描范围与豁免
SCAN_SUFFIX = {".md", ".py", ".yaml", ".yml", ".json"}
SKIP_DIRS = {".git", ".probe", "runs", "data", "weights", "__pycache__", "node_modules"}
# 豁免：本校验器自身（它必须写出这些模式才能匹配它们）
#   ★ `selftest_checkers.py` 也必须豁免：它是**阳性对照的载体**，
#     按设计**必须**包含已作废的数字作为测试输入。见 AGENTS-LESSONS P15。
EXEMPT = {
    "scripts/smoke/check_stale_numbers.py",
    "scripts/smoke/selftest_checkers.py",
    "scripts/smoke/check_no_mojibake.py",   # 同样含乱码样本字面量
    "AGENTS-LESSONS.md",                    # 教训表必须引用错误原值
}


def scan(root: Path) -> list[tuple[Path, int, str, str, str]]:
    hits = []
    files = [p for p in root.rglob("*")
             if p.is_file() and p.suffix.lower() in SCAN_SUFFIX
             and not any(part in SKIP_DIRS for part in p.parts)]
    for p in files:
        rel = p.relative_to(root).as_posix()
        if rel in EXEMPT:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for label, spec in STALE.items():
                if spec["pattern"].search(line):
                    # 先看豁免（合法语境），再看留痕，最后才判裸用
                    if any(fp["pattern"].search(line) for fp in FALSE_POSITIVE_EXEMPT):
                        verdict = "豁免"
                    elif RETIRED_MARKERS.search(line):
                        verdict = "留痕OK"
                    else:
                        verdict = "❌裸用"
                    hits.append((p, lineno, label, spec["why"], verdict))
    return hits, len(files)


def main() -> int:
    ap = argparse.ArgumentParser(description="扫描残留的已作废数字与口径")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    a = ap.parse_args()

    print(f"扫描根目录：{a.root}")
    print(f"作废口径：{'、'.join(STALE)}")
    print("判据：命中且**同行无留痕标记**才算裸用（留痕是有益的修正历史）\n")

    hits, n_files = scan(a.root)
    print(f"共扫描 {n_files} 个文件")
    if n_files == 0:
        print("❌ 扫描到 0 个文件 —— 视为**未验证**（见 AGENTS-LESSONS P15）")
        return 1

    bare = [h for h in hits if h[4] == "❌裸用"]
    traced = [h for h in hits if h[4] == "留痕OK"]
    exempt = [h for h in hits if h[4] == "豁免"]

    if traced:
        print(f"\n✅ {len(traced)} 处带留痕（正常，记录了修正历史）：")
        for p, ln, label, _, _ in traced[:6]:
            print(f"   {p.relative_to(a.root)}:{ln}  [{label}]")
        if len(traced) > 6:
            print(f"   …另有 {len(traced) - 6} 处")

    if exempt:
        print(f"\n✅ {len(exempt)} 处豁免（合法语境，理由见脚本 FALSE_POSITIVE_EXEMPT）：")
        for p, ln, label, _, _ in exempt[:6]:
            print(f"   {p.relative_to(a.root)}:{ln}  [{label}]")
        if len(exempt) > 6:
            print(f"   …另有 {len(exempt) - 6} 处")

    if bare:
        print(f"\n❌ 发现 {len(bare)} 处**裸用**已作废口径：\n")
        for p, ln, label, why, _ in bare:
            print(f"   {p.relative_to(a.root)}:{ln}")
            print(f"      口径：{label}")
            print(f"      原因：{why}")
        print("\n修法：① 改正为现行值；② 或加留痕标记（~~旧值~~ /（已作废）/ 原值）；")
        print("      ③ 若是合法语境（在讲另一个对象），把理由加进 FALSE_POSITIVE_EXEMPT 并写明依据。")
        return 1

    print("\n✅ 未发现裸用的已作废口径")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
