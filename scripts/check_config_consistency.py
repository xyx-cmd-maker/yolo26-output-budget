#!/usr/bin/env python
"""scripts/check_config_consistency.py —— JSON 结果文件的配置一致性检查（P131）

═══ 为什么需要 ═══════════════════════════════════════════════════════════
实测抓到：`decision-rule-r1_s43-val.json` 的 `n_boot` 还是 2000，
而其余结果文件都是 10000 —— 因为当初为了省时间只给 s42 跑了 10000。
这类不一致不会报错、不会崩溃，只是让三个种子的 CI 精度不同，
却让 Methods/图注里的"统一 10,000"变成不可为真的句子。

═══ 机制 ═════════════════════════════════════════════════════════════════
对 `results/tables/*.json` 做深度遍历，抽出配置字段，与声明的期望值比对。
`EXEMPT` 里可以按文件名登记合理的例外并写明理由。

用法：python scripts/check_config_consistency.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"

# 期望值：字段 -> 允许的取值集合
EXPECT = {
    "n_boot": {10000},
    "rng_seed": {20261014},
    "alpha": {0.05},
    "margin_pt": {1.0},
}

# 按 (文件名, 字段) 登记例外并写明理由
EXEMPT = {
    ("model-structure.json", "seed"): "训练种子（42/43/44），不是 RNG 种子",
}


def walk(o, path=""):
    """深度遍历，yield (字段名, 值, 路径)。"""
    if isinstance(o, dict):
        for k, v in o.items():
            if k in EXPECT and not isinstance(v, (dict, list)):
                yield k, v, path
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(o, list):
        for i, v in enumerate(o[:5]):
            yield from walk(v, f"{path}[{i}]")


def main() -> int:
    print("=" * 80)
    print("JSON 结果文件的配置一致性检查")
    print("=" * 80)

    bad = 0
    total = 0
    for f in sorted(TAB.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        hits = list(walk(d))
        if not hits:
            continue
        by_key: dict[str, set] = {}
        for k, v, _ in hits:
            by_key.setdefault(k, set()).add(v)
        for k, vals in sorted(by_key.items()):
            if (f.name, k) in EXEMPT:
                print(f"  ⊘  {f.name:<34} {k:<9} = {sorted(map(str, vals))}  "
                      f"（例外：{EXEMPT[(f.name, k)]}）")
                continue
            total += 1
            unexpected = {v for v in vals if v not in EXPECT[k]}
            if unexpected:
                bad += 1
                print(f"  ❌ {f.name:<34} {k:<9} = {sorted(map(str, vals))}  "
                      f"期望 {sorted(EXPECT[k])}")
            else:
                print(f"  ✅ {f.name:<34} {k:<9} = {sorted(map(str, vals))}")

    print()
    if bad:
        print(f"  ❌ {bad} 处配置与期望不一致 —— 必须重跑对应产物或修正期望")
        return 1
    print(f"  ✅ 全部 {total} 处配置字段一致"
          f"（n_boot={sorted(EXPECT['n_boot'])}, rng_seed={sorted(EXPECT['rng_seed'])}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
