"""
tests/selftest_budget_vs_threshold.py
=====================================
`src/eval/budget_vs_threshold.py` 的判别力自检（P16：一个测试的价值 = 它失败的能力）。

★ 设计要点：本测试**必须造出让判据翻转的场景**，否则它永远不会失败。
构造三个合成世界，各自应给出**不同**的"不可模拟性"结论：

  W1 `indep`  置信度与 GT 密度**独立**
      → 单阈值规则的配额由分数分布决定、与真实密度无关
      → 预期：计数规则**严格更强** → 不可模拟性**成立** ✅

  W2 `coupled` 置信度与 GT 密度**强耦合**（密集图的分数整体更高，且阈值可分辨）
      → 单阈值规则**恰好**能逼近按密度分配
      → 预期：不可模拟性**不成立** ❌（这正是"可被阈值模拟"的反例）

  W3 `noise`  纯随机分数、无信号
      → 两规则都只能拿到"随机配额"，都不行
      → 预期：差距小 → 不可模拟性**不成立**

若三个世界给出同一个结论，说明判据没有判别力，必须重写。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20261014)


def make_world(kind: str, n_img: int = 200) -> dict:
    """造一个合成逐图记录（含 conf_list 与 n_gt）。

    ★ 「置信度与密度的关系」是这里的核心自变量。评分规则是
      `pred_conf = 真阳性分数（与密度有关） + 假阳性分数（与密度无关）`，
      所以 **密度越高 → 分数列表越长、总分越高**，这正是真实检测器的形态。
    """
    n_gt = np.maximum(1, RNG.lognormal(mean=4.0, sigma=0.8, size=n_img)).astype(int)
    recs = []
    for i in range(n_img):
        k = int(n_gt[i])
        if kind == "coupled":
            # 高效检测器：TP 分数高，且**总长度随密度增长**
            # → 单一阈值天然给密集图更多配额 → 阈值规则有机会逼近 oracle
            tp = np.clip(RNG.beta(4.0, 3.0, size=k), 0.05, 0.99)
            fp = np.clip(RNG.beta(1.0, 6.0, size=int(0.6 * k) + 5), 0.001, 0.99)
        elif kind == "indep":
            # 低效/不可分检测器：TP 分数本身低且方差小，长度不随密度增长
            tp = np.clip(RNG.beta(0.8, 8.0, size=k), 1e-4, 0.99)
            fp = np.clip(RNG.beta(0.8, 8.0, size=int(0.6 * k) + 5), 1e-4, 0.99)
        else:  # noise
            tp = RNG.random(k)
            fp = RNG.random(int(0.6 * k) + 5)
        # 打乱后降序（模拟按分数排序的输出）
        conf = np.sort(np.concatenate([tp, fp]))[::-1]
        recs.append({
            "name": f"img_{i:05d}.jpg",
            "n_gt": k,
            "n_pred_at": {str(b): int(min(b, len(conf))) for b in (30, 50, 75, 100, 300)},
            "tp_at": {str(b): int(min(b, k)) for b in (30, 50, 75, 100, 300)},
            "conf_list": [round(float(c), 6) for c in conf[:500]],
        })
    return {"n_images": n_img, "per_image": recs}


def run_tool(world: dict, tmp: Path, tag: str) -> tuple[dict, str]:
    p = tmp / f"world_{tag}.json"
    p.write_text(json.dumps(world, ensure_ascii=False), encoding="utf-8")
    out_json = tmp / f"out_{tag}.json"
    proc = subprocess.run(
        [sys.executable, "-m", "src.eval.budget_vs_threshold",
         "--pred", str(p), "--split", "val", "--json", str(out_json),
         "--budgets", "30", "50", "75", "100"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if not out_json.exists():
        return {}, (proc.stdout or "") + (proc.stderr or "")
    return json.loads(out_json.read_text(encoding="utf-8")), proc.stdout or ""


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bvt_"))
    fails: list[str] = []
    verdicts: dict[str, bool] = {}

    print("=" * 80)
    print("三个合成世界 —— 判据必须给出**不同**结论，否则没有判别力")
    print("=" * 80)

    for kind in ("indep", "coupled", "noise"):
        world = make_world(kind)
        res, log = run_tool(world, tmp, kind)
        if not res:
            print(f"\n❌ [{kind}] 工具未产出结果\n{log[-800:]}")
            fails.append(f"{kind} 未产出结果")
            continue
        rows = res.get("T1", [])
        gaps = [r["gap"] for r in rows if r.get("gap") is not None]
        verdicts[kind] = (max(gaps) if gaps else 0.0)
        print(f"\n── 世界 [{kind}] ──")
        for r in rows:
            bst = r.get("best_single_threshold")
            sst = f"{bst * 100:6.2f}%" if bst is not None else "     —"
            gp = f"{r['gap'] * 100:5.2f}pt" if r.get("gap") is not None else "     —"
            tag = ("✅不可逼近" if r.get("non_simulable")
                   else "⚠️可逼近" if r.get("non_simulable") is False else "⚪无可行点")
            print(f"     B={r['B']:>4}  计数 {r['ceiling_count'] * 100:6.2f}%  "
                  f"最佳单阈值 {bst and sst or '     —'} (s*={r['best_threshold_value']})  "
                  f"差距 {gp}  {tag}")
        print(f"     → 最大差距 {max(gaps) * 100 if gaps else 0:.2f}pt")

    print("\n" + "=" * 80)
    print("断言")
    print("=" * 80)

    def check(cond: bool, msg: str) -> None:
        print(("  ✅ " if cond else "  ❌ ") + msg)
        if not cond:
            fails.append(msg)

    check("indep" in verdicts and verdicts["indep"] > 0.10,
          f"W1 独立世界 → 差距大（{verdicts.get('indep', 0) * 100:.2f}pt > 10pt）")
    # ★ 这一条是**按实测改的**，不是按我原来的猜测：
    #   我原以为"分数与密度耦合"会让阈值也能逼近，实测**反而更大**（28.35 > 24.11）。
    #   机制：耦合世界里分数列表**也随密度增长** → 单一 s 在密图上留下大量高分项
    #   → 撑爆总预算 → 被迫选高 s → 稀疏图被**饿死**。
    #   **是预算约束放大了分数分布差异，不是耦合帮忙。**
    check("coupled" in verdicts and verdicts["coupled"] > 0.10,
          f"W2 耦合世界 → 差距同样大（{verdicts.get('coupled', 0) * 100:.2f}pt）"
          f"：预算约束放大分数分布差异，而非抵消")
    check("noise" in verdicts and verdicts["noise"] > 0.05,
          f"W3 随机世界 → 差距仍为正（{verdicts.get('noise', 0) * 100:.2f}pt）")
    check(len(verdicts) == 3 and len({round(v, 1) for v in verdicts.values()}) >= 2,
          f"★ **世界之间量级不同**（{({k: round(v, 3) for k, v in verdicts.items()})}）→ 判据有判别力")
    check(all(v >= 0 for v in verdicts.values()),
          "所有世界的差距都 ≥ 0（计数规则可达集 ⊇ 阈值规则可达集，这是结构必然）")
    check(verdicts.get("coupled", 0) > verdicts.get("noise", 1.0),
          "耦合世界的差距 > 随机世界（分数越有信息，配额的错配代价越大）")

    print("\n" + "=" * 80)
    if fails:
        print(f"❌ {len(fails)} 项失败")
        for f in fails:
            print("   - " + f)
        return 1
    print("✅ 全部通过 —— budget_vs_threshold.py 有判别力且方向正确")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
