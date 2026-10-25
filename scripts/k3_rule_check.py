"""
K3 的实操规则检验（c 无关的保守规则）

诊断发现（.probe/diag_cstar_sensitivity.py）：
  · B=30：c* = 39，c* 处斜率 0.989 pt/c → **估计误差会直接放大**（±10% → ±3.9 pt）
  · B=50：c* = 462 = max nᵢ，c* 处斜率 0.001 pt/c → **平台机制**
    （预算已能覆盖 93.3% 的框，超过 300 之后加 c 完全不再增加召回）

→ 所以"用简单分布摘要估 c*"在松预算档**天然失败**，因为它需要的是**尾部**（max nᵢ），
  而不是均值/中位。

★ 本脚本检验一条**保守且 c 无关**的实用规则：
      **总是把 cap 设在数据集最大目标数量级**（例如 300，即上游默认），
      而不是设在 B。
  理由：配额只在"该图 nᵢ < cap"时才浪费；把 cap 设大**不会**浪费预算
  （未用尽的额度会按规则回填），但把 cap 设在 B 会**人为掐掉**密集图的配额。

若该规则在所有预算档都接近 c* 上界，则 K3 的实操建议极其简单：
**不要把逐图配额上限设成 B；设成一个与数据集最大目标数同量级的常数。**

用法：python scripts/k3_rule_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))
from src.eval.proxy_density import ceiling_uniform, uniform_cap  # noqa: E402




def main() -> int:
    def load(split):
        cfg = yaml.safe_load((ROOT / "data/processed/visdrone/visdrone.yaml").read_text(encoding="utf-8"))
        d = Path(cfg["path"]) / cfg[split]
        p = list(d.parts)
        for i in range(len(p) - 1, -1, -1):
            if p[i] == "images":
                p[i] = "labels"
                break
        return np.array([sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
                         for f in sorted(Path(*p).glob("*.txt"))], dtype=float)


    RULES = [
        ("cap = B", lambda B, mx: B),
        ("cap = 2B", lambda B, mx: 2 * B),
        ("cap = 4B", lambda B, mx: 4 * B),
        ("cap = 100", lambda B, mx: 100),
        ("cap = 300（上游默认）", lambda B, mx: 300),
        ("cap = max nᵢ", lambda B, mx: int(mx)),
        ("cap = c*（oracle）", lambda B, mx: None),   # 特判
    ]

    print("=" * 100)
    print("K3 实操规则检验：哪条'不需要知道 nᵢ'的规则最接近 c* 上界？")
    print("=" * 100)

    for split in ("val", "test"):
        counts = load(split)
        N, TOT, MX = len(counts), counts.sum(), counts.max()
        print(f"\n{'#' * 100}\n### {split}：N={N}  TOT={int(TOT)}  max nᵢ={MX:.0f}  均值={counts.mean():.2f}")
        hdr = f"{'B':>5} │"
        for name, _ in RULES:
            hdr += f" {name:>18}"
        print(hdr)
        print("-" * 100)
        summary: dict[str, list[float]] = {name: [] for name, _ in RULES}
        for B in (20, 30, 40, 50, 75, 100):
            cstar = uniform_cap(counts, B)
            v_cap = ceiling_uniform(counts, cstar)
            v_base = ceiling_uniform(counts, B)
            gain = v_cap - v_base
            row = f"{B:>5} │"
            for name, fn in RULES:
                # ★ 注意：`fn` 是 lambda，永不为 None —— 必须按**名字**特判 oracle 规则
                c = cstar if name.startswith("cap = c*") else fn(B, MX)
                v = ceiling_uniform(counts, c)
                frac = (v - v_base) / gain if gain > 1e-9 else float("nan")
                summary[name].append(frac)
                mark = "✅" if frac >= 0.95 else ("⚠️" if frac >= 0.5 else "❌")
                row += f" {frac * 100:>10.1f}%{mark:>7}"
            print(row)
        print(f"\n  各规则的**最差档位**表现（最差 = 该规则在 6 个 B 档里最差的那个）:")
        for name, _ in RULES:
            vals = [v for v in summary[name] if not np.isnan(v)]
            if not vals:
                continue
            print(f"    {name:>22}  最差 {min(vals) * 100:6.1f}%   平均 {np.mean(vals) * 100:6.1f}%")

    print("\n" + "=" * 100)
    print("★ 判读")
    print("=" * 100)
    print("  · 若 `cap = max nᵢ` 或 `cap = 300` 在所有档位都 ≥95% → **K3 的实操建议是 c 无关的**：")
    print("    「**不要把逐图配额上限设成 B；设成一个与数据集最大目标数同量级的常数**」，")
    print("    而该常数**无需精确标定**、也**不需要知道逐图 nᵢ**。")
    print("  · 若是 → K3 从「需要标定一个标量」升级为「**一个 c 无关的部署修正**」，")
    print("    这比原来的表述**更强**（更简单、更鲁棒、更可迁移）。")



if __name__ == "__main__":
    raise SystemExit(main())
