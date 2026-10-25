"""
tests/selftest_j2_collision.py
==============================
**J2 的可触发自检**：显式构造「总漏检相同、容量/排序配比不同」的一对世界。

═══════════════════════════════════════════════════════════════════════════
为什么需要这个单独的自检（第 3 轮自认的漏洞）
═══════════════════════════════════════════════════════════════════════════
第 3 轮里我给 `realisation_decomposition.py` 设计了 J1/J2/J3 三条判据，
但 J2 **在自检里根本没被触发** —— 两个合成世界没产生"总漏检相同"的碰撞条目。
这正是 P16 的变体：**看起来有测试，实际没跑**。

═══════════════════════════════════════════════════════════════════════════
J2 的正确表述（第 4 轮重写）
═══════════════════════════════════════════════════════════════════════════
★ 关键认识：固定数据与 B 时
      `容量漏 = Σnᵢ − Σmin(B,nᵢ)`  **与检测器完全无关**
   所以"同一 TP(B)"**不可能**由同一 (数据, B) 上的两个检测器产生。

因此 J2 的正确问题是：

    **两个模型报告了相同的漏检数时，它们的失效模式是否相同？**

  若"不同模型 + 相同漏检数"可以对应**迥异的容量/排序配比** → FN 无法诊断根因
  → 分解**揭示了 FN 计数揭示不了的东西** → R-C 消解。

═══════════════════════════════════════════════════════════════════════════
本自检构造什么
═══════════════════════════════════════════════════════════════════════════
在世界 **W_cap**（预算紧、排序完美）里取 B₁，在 **W_rank**（预算宽、排序差）里取 B₂，
令两者的**总漏检数尽量相等**，然后断言：
  · 容量占比**差异 > 20 pt**（否则 J2 不成立）
  · 总漏检数**差异 ≤ 1%**（否则不构成"碰撞"）
两个条件同时满足，J2 才算**被验证**。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RNG = np.random.default_rng(20261014)


def load_ngts(data_yaml: str, split: str, n_img: int | None = None) -> np.ndarray:
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    img_dir = Path(cfg["path"]) / cfg[split]
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    lbl = Path(*parts)
    vals = []
    for f in sorted(lbl.glob("*.txt")):
        vals.append(sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip()))
    a = np.array(vals, dtype=float)
    return a[:n_img] if n_img else a


def make_run(ngts: np.ndarray, budgets: list[int], quality: float,
             tp_overrides: dict[int, int] | None = None) -> dict:
    """造逐图记录。

    `quality` 控制排序能力（1.0 = 完美）。
    `tp_overrides`：`{B: 目标总 TP}` —— 对指定 B 档**按比例缩放**使总 TP 命中目标。

    ★ 为什么需要 override（第 4 轮的教训）：
      单一 `quality` 无法同时命中两个不同 B 档的 TP 目标 —— 而 J2 的碰撞对**天生**
      是"两个不同 B 档"（如 B₁=20 与 B₂=150，容量漏分别是 6356 与 94）。
      靠网格搜 quality 找不到，是因为我在用 1 个自由度去解 2 个约束。
      → 直接按目标构造，把碰撞**设计出来**，而不是碰运气搜出来。
    """
    tp_overrides = tp_overrides or {}
    per_image = []
    for i, k in enumerate(ngts):
        tp = {}
        for B in budgets:
            cap = min(B, int(k))
            got = max(0, min(int(round(cap * quality)), cap))
            tp[str(B)] = got
        per_image.append({
            "name": f"img_{i:05d}.jpg", "n_gt": int(k),
            "n_pred_at": {str(b): min(b, int(k)) for b in budgets},
            "tp_at": tp,
            "conf_list": [round(float(c), 6) for c in
                          np.sort(RNG.random(min(2 * int(k) + 10, 200)))[::-1]],
        })

    # 按比例缩放指定 B 档，使总 TP **精确**命中目标
    # ★ 教训（第 4 轮）：只做"缩放 + 取整"会让总和**差几个框**（实测目标 729、实得 726），
    #   于是两个世界的总漏检不再相等 → 端到端报"无碰撞条目"。
    #   → 必须**把余数补足**，使总和严格等于 target。
    for B, target in tp_overrides.items():
        key = str(B)
        caps = [min(B, r["n_gt"]) for r in per_image]
        total_cap = sum(caps)
        target = max(0, min(int(target), total_cap))
        cur = sum(r["tp_at"][key] for r in per_image)
        if cur <= 0:
            # 从零开始按容量比例分配
            vals = [int(round(c * target / total_cap)) if total_cap else 0 for c in caps]
        else:
            scale = target / cur
            vals = [int(round(r["tp_at"][key] * scale)) for r in per_image]
        vals = [max(0, min(v, c)) for v, c in zip(vals, caps)]
        # 补/减余数（按容量从大到小调整，确定性的）
        diff = target - sum(vals)
        order = sorted(range(len(vals)), key=lambda i: -caps[i])
        k = 0
        while diff != 0 and k < 10 * len(vals):
            i = order[k % len(order)]
            if diff > 0 and vals[i] < caps[i]:
                vals[i] += 1
                diff -= 1
            elif diff < 0 and vals[i] > 0:
                vals[i] -= 1
                diff += 1
            k += 1
        for i, r in enumerate(per_image):
            r["tp_at"][key] = vals[i]
        got = sum(vals)
        assert got == target, f"override 未精确命中：目标 {target} 实得 {got}"
    return {"n_images": len(ngts), "per_image": per_image}


def main() -> int:
    data_yaml = str(ROOT / "data/processed/visdrone/visdrone.yaml")
    if not Path(data_yaml).exists():
        print(f"❌ 找不到 {data_yaml}")
        return 1
    ngts = load_ngts(data_yaml, "val", n_img=150)
    tot_gt = int(ngts.sum())

    # 用一组合适的 B 档，使 W_cap 与 W_rank 的漏检数有可能碰撞
    budgets = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300]
    print(f"数据：{len(ngts)} 图 / {tot_gt} 框")

    # ── 解析地找碰撞对（先算，不依赖外部脚本）────────────────────────
    print("\n=== 解析搜索：找「总漏检数相近、容量占比迥异」的 (B₁, B₂) 对 ===")
    print(f"{'B':>5} {'容量漏':>8} {'上限':>8}")
    rows = []
    for B in budgets:
        ceiling = int(np.minimum(B, ngts).sum())
        rows.append({"B": B, "ceiling": ceiling, "capacity_miss": tot_gt - ceiling})
    for r in rows:
        print(f"{r['B']:>5} {r['capacity_miss']:>8} {r['ceiling']:>8}")

    # ── 直接构造碰撞对（不再网格搜索）────────────────────────────────
    # ★ 第 4 轮的关键认识：J2 的碰撞对**天生**是"两个不同 B 档"：
    #     `容量漏(B) = Σnᵢ − Σmin(B,nᵢ)` 只依赖 (数据, B)，与检测器无关。
    #   所以要制造"总漏检相同、容量占比迥异"，必须让两个世界用**不同的 B**：
    #     · W_cap 用**小 B**（容量漏大）＋ 排序完美 → 漏检几乎全是容量
    #     · W_rank 用**大 B**（容量漏小）＋ 排序差   → 漏检几乎全是排序
    #   并用 `tp_overrides` 把两者的**总 TP 精确对齐**，从而总漏检严格相等。
    ceilings = {B: int(np.minimum(B, ngts).sum()) for B in budgets}
    cap_miss = {B: tot_gt - ceilings[B] for B in budgets}
    print("\n各 B 档的容量漏（只依赖数据与 B，与检测器无关）：")
    for B in budgets:
        print(f"   B={B:>4}  容量漏={cap_miss[B]:>7}  上限={ceilings[B]:>7}")

    # ── 穷举所有 (小B, 大B) 组合，选出**端到端必定命中**的那一对 ─────────
    # ★ 教训（第 4 轮，我在这里栽了 4 次）：不要"碰运气式构造"。
    #   前几版分别试过：网格搜 q、按最小漏检差选、按占比跨度选、取 min/max B ——
    #   全都因为**选择策略与实际碰撞点不对齐**而失败。
    #   根因：`realisation_decomposition.py` 的 J2 取的是"**跨度最大**的碰撞组"，
    #   所以我要构造的必须是**全局跨度最大**的那一对，而不是我随手挑的一对。
    #   → 正确做法：**穷举 + 用与端到端相同的目标函数排序**，取第一名。
    ceilings_map = {B: int(np.minimum(B, ngts).sum()) for B in budgets}
    cap_miss_map = {B: tot_gt - ceilings_map[B] for B in budgets}

    cands = []
    for B1 in budgets:                       # W_cap 用的小 B（容量漏大）
        for B2 in budgets:                   # W_rank 用的大 B（容量漏小）
            if B2 <= B1:
                continue
            if cap_miss_map[B1] <= 0 or ceilings_map[B2] <= 0:
                continue
            miss_target = cap_miss_map[B1]                  # 两世界共同的总漏检
            tp_target = tot_gt - miss_target                # = ceilings_map[B1]
            if tp_target > ceilings_map[B2]:
                continue                                    # 大 B 上也拿不到这么多 TP → 不可行
            if miss_target <= 0:
                continue
            s_cap = cap_miss_map[B1] / miss_target           # = 1.0（排序完美）
            s_rank = cap_miss_map[B2] / miss_target
            cands.append((abs(s_cap - s_rank), B1, B2, miss_target, tp_target, s_cap, s_rank))
    if not cands:
        print("\n❌ 没有可行的 (B1, B2) 组合")
        return 1
    # ★ 用与端到端**相同的目标函数**（跨度最大）排序 → 保证端到端会选中同一对
    cands.sort(reverse=True, key=lambda t: t[0])
    expected_spread, B1, B2, miss_target, tp_target, s_cap, s_rank = cands[0]
    print(f"\n可行组合 {len(cands)} 个；按**跨度最大**（与端到端同目标）选第一名：")
    print(f"  W_cap : B={B1:>4}（quality=1.00）→ 总漏检 {miss_target}，容量占比 {s_cap * 100:5.1f}%")
    print(f"  W_rank: B={B2:>4}（tp_override 对齐）→ 总漏检 {miss_target}，容量占比 {s_rank * 100:5.1f}%")
    print(f"  → 预期跨度 **{expected_spread * 100:.1f} pt**")

    # ── 用真实脚本跑一遍（端到端）────────────────────────────────────
    # W_cap ：quality=1.0 → 排序完美；在小 B₁ 上漏检 = 容量漏 c1（100% 容量）
    # W_rank：quality 低，且 **用 tp_override 把 B₂ 档的总 TP 对齐到 ceilings[B1]**
    #         → 总漏检也 = c1，但其中只有 c2 来自容量、其余来自排序
    tmp = Path(tempfile.mkdtemp(prefix="j2_"))
    w_cap = make_run(ngts, budgets, quality=1.0)
    w_rank = make_run(ngts, budgets, quality=0.5,
                      tp_overrides={B2: tp_target})
    p1, p2 = tmp / "cap.json", tmp / "rank.json"
    p1.write_text(json.dumps(w_cap, ensure_ascii=False), encoding="utf-8")
    p2.write_text(json.dumps(w_rank, ensure_ascii=False), encoding="utf-8")
    out_json = tmp / "out.json"
    proc = subprocess.run(
        [sys.executable, "-m", "src.eval.realisation_decomposition",
         "--pred", f"cap={p1}", f"rank={p2}", "--split", "val",
         "--json", str(out_json),
         "--budgets", *[str(b) for b in budgets]],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if not out_json.exists():
        print("❌ 脚本未产出结果\n" + ((proc.stdout or "") + (proc.stderr or ""))[-1500:])
        return 1
    res = json.loads(out_json.read_text(encoding="utf-8"))

    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("  ✅ " if cond else "  ❌ ") + msg)
        if not cond:
            fails.append(msg)

    print("\n=== 端到端结果（来自 realisation_decomposition.py）===")
    j2 = res.get("J2", {})
    print(f"  J2 = {j2}")
    print(f"  J1 = {res.get('J1')}")

    print("\n=== 断言 ===")
    check("capacity_share_spread" in j2,
          "★ J2 **被触发**了（找到了碰撞条目）—— 这正是第 3 轮缺失的部分")
    if "capacity_share_spread" in j2:
        got_spread = j2["capacity_share_spread"]
        check(got_spread > 0.20,
              f"★ 同一总漏检下容量占比跨度 {got_spread * 100:.1f} pt > 20pt"
              f" → **FN 无法诊断失效模式，分解可以** → J2 成立")
        check(j2.get("n_entries", 0) >= 2,
              f"碰撞组含 ≥2 个条目（实得 {j2.get('n_entries')}）")
        check(abs(got_spread - expected_spread) < 0.02,
              f"构造预期与端到端一致"
              f"（预期 {expected_spread * 100:.1f} pt vs 端到端 {got_spread * 100:.1f} pt）")

    print("\n" + "=" * 78)
    if fails:
        print(f"❌ {len(fails)} 项失败")
        for f in fails:
            print("   - " + f)
        return 1
    print("✅ J2 现已**被验证**（第 3 轮的『未验证』漏洞已消除）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
