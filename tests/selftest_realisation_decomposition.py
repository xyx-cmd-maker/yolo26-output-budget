"""
tests/selftest_realisation_decomposition.py
===========================================
`src/eval/realisation_decomposition.py` 的判别力自检（P16）。

构造两个**失效模式截然不同**的世界，检验分解能否区分它们：

  W_capacity  **容量受限型**：预算很小、排序近乎完美
      → 漏检几乎全来自 `Σnᵢ − Σmin(B,nᵢ)`（额度不够）→ 容量占比 ≈ 1
  W_ranking   **排序受限型**：预算充裕、排序很差
      → 漏检几乎全来自 `Σmin(B,nᵢ) − TP(B)`（排不上）→ 容量占比 ≈ 0

★ 判别力要求：两个世界的容量占比必须**显著不同**。
   若两者都给同一个数，说明分解（或本测试）没有诊断价值 → 必须重写。

另外检验：
  · 恒等式 `容量 + 排序 == 总漏检`（严格相等，整数计数）
  · 两项非负
  · **J3**：两个检测器在同一 B 下容量项**必须完全相同**（与检测器无关）
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RNG = np.random.default_rng(20261014)


def make_run(kind: str, data_yaml: str, split: str, budgets: list[int],
             n_img: int = 150, *, quality: float = 1.0) -> dict:
    """造 budget_scan 形状的逐图记录。

    `quality` 控制排序能力：1.0 = 完美；0 = 全错。
    """
    import yaml

    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    img_dir = Path(cfg["path"]) / cfg[split]
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    lbl = Path(*parts)
    gt = {}
    for f in sorted(lbl.glob("*.txt")):
        gt[f.stem] = sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
    names = sorted(gt)[:n_img]

    per_image = []
    for nm in names:
        k = gt[nm]
        tp = {}
        for B in budgets:
            cap = min(B, k)
            if kind == "capacity":
                # 排序近乎完美：额度用满
                got = cap if quality >= 1.0 else int(round(cap * quality))
            elif kind == "ranking":
                # 排序很差：只有很低比例排上
                got = int(round(cap * quality))
            else:
                got = int(round(cap * quality))
            tp[str(B)] = max(0, min(got, cap))
        per_image.append({"name": f"{nm}.jpg", "n_gt": k,
                          "n_pred_at": {str(b): min(b, k) for b in budgets},
                          "tp_at": tp,
                          "conf_list": [round(float(c), 6) for c in
                                        np.sort(RNG.random(min(3 * k + 10, 300)))[::-1]]})
    return {"n_images": len(names), "per_image": per_image}


def run_tool(specs: list[str], tmp: Path, tag: str) -> tuple[dict, str]:
    out_json = tmp / f"out_{tag}.json"
    proc = subprocess.run(
        [sys.executable, "-m", "src.eval.realisation_decomposition",
         "--pred", *specs, "--split", "val", "--json", str(out_json),
         "--budgets", "20", "30", "50", "100", "300"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if not out_json.exists():
        return {}, (proc.stdout or "") + (proc.stderr or "")
    return json.loads(out_json.read_text(encoding="utf-8")), proc.stdout or ""


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rdec_"))
    data_yaml = str(ROOT / "data/processed/visdrone/visdrone.yaml")
    if not Path(data_yaml).exists():
        print(f"❌ 找不到 {data_yaml}")
        return 1
    budgets = [20, 30, 50, 100, 300]
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("  ✅ " if cond else "  ❌ ") + msg)
        if not cond:
            fails.append(msg)

    # ── 世界 1：容量受限（预算小于典型 nᵢ，排序完美）──────────────
    w_cap = make_run("capacity", data_yaml, "val", budgets, quality=1.0)
    p_cap = tmp / "cap.json"
    p_cap.write_text(json.dumps(w_cap, ensure_ascii=False), encoding="utf-8")

    # ── 世界 2：排序受限（同数据，但排序只拿到 20%）──────────────
    w_rank = make_run("ranking", data_yaml, "val", budgets, quality=0.20)
    p_rank = tmp / "rank.json"
    p_rank.write_text(json.dumps(w_rank, ensure_ascii=False), encoding="utf-8")

    res, log = run_tool([f"cap={p_cap}", f"rank={p_rank}"], tmp, "both")
    if not res:
        print("❌ 工具未产出结果\n" + log[-1500:])
        return 1

    print("=" * 84)
    print("分解表")
    print("=" * 84)
    for name in ("cap", "rank"):
        print(f"\n── {name} ──")
        for d in res["runs"][name]:
            cs = d["capacity_share"]
            print(f"   B={d['B']:>4}  TP={d['tp']:>7}  上限={d['tp_ceiling']:>7}  "
                  f"容量漏={d['capacity_miss']:>7}  排序漏={d['ranking_miss']:>7}  "
                  f"总漏={d['total_miss']:>7}  容量占比={cs * 100 if cs is not None else float('nan'):5.1f}%  "
                  f"实现率={d['realisation_rate'] * 100:5.1f}%")

    print("\n" + "=" * 84)
    print("断言")
    print("=" * 84)

    # 1) 恒等式与符号
    ok_id = True
    for name, rows in res["runs"].items():
        for d in rows:
            if d["capacity_miss"] + d["ranking_miss"] != d["total_miss"]:
                ok_id = False
            if d["capacity_miss"] < 0 or d["ranking_miss"] < 0:
                ok_id = False
    check(ok_id, "恒等式严格成立且两项非负（容量 + 排序 == 总漏检）")

    # 2) ★ 判别力：两个世界的容量占比必须显著不同
    cap_shares = {d["B"]: d["capacity_share"] for d in res["runs"]["cap"]}
    rank_shares = {d["B"]: d["capacity_share"] for d in res["runs"]["rank"]}
    fmt = lambda dd: "{" + ", ".join(  # noqa: E731
        f"{k}: {'None' if v is None else format(v, '.3f')}" for k, v in sorted(dd.items())) + "}"
    print(f"\n  容量受限世界 容量占比: {fmt(cap_shares)}")
    print(f"  排序受限世界 容量占比: {fmt(rank_shares)}")
    common = set(cap_shares) & set(rank_shares)
    diffs = [cap_shares[b] - rank_shares[b] for b in common
             if cap_shares[b] is not None and rank_shares[b] is not None]
    check(bool(diffs) and max(diffs) > 0.30,
          f"★ 两个世界的容量占比差异显著（最大 {max(diffs) * 100 if diffs else 0:.1f} pt > 30pt）"
          f" → 分解有诊断价值")

    # 3) 容量受限世界的容量占比应恒为 1（排序完美 → 漏检全来自容量）
    cs_cap = [cap_shares[b] for b in sorted(cap_shares) if cap_shares[b] is not None]
    check(all(abs(v - 1.0) < 1e-9 for v in cs_cap),
          f"容量受限世界（排序完美）：容量占比恒 == 1.0（{[round(v, 3) for v in cs_cap]}）")

    # 3b) 排序受限世界：容量占比随 B **单调不增**，且在 B 足够大时趋近 0
    cs_rank = [rank_shares[b] for b in sorted(rank_shares) if rank_shares[b] is not None]
    mono = all(cs_rank[i] >= cs_rank[i + 1] - 1e-9 for i in range(len(cs_rank) - 1))
    check(mono, f"排序受限世界：容量占比随 B **单调不增**（{[round(v, 3) for v in cs_rank]}）")
    check(cs_rank and cs_rank[-1] < 0.05,
          f"排序受限世界：B 增大后容量占比趋近 0（末档 {cs_rank[-1] if cs_rank else float('nan'):.3f}）")

    # 4) J1：跨度
    j1 = res.get("J1", {})
    check(any(v.get("span", 0) > 0.10 for v in j1.values()),
          f"J1 容量占比跨度 >10pt（{({k: round(v['span'], 3) for k, v in j1.items()})}）")

    # 5) ★ J3：两个检测器在同一 B 下容量项**必须完全相同**
    j3 = res.get("J3", {})
    j3_ok = bool(j3) and all(v.get("capacity_identical") for v in j3.values()
                             if isinstance(v, dict) and "capacity_identical" in v)
    check(j3_ok, f"★ J3：容量项与检测器无关（逐 B 完全相同）—— {j3}")
    tp_ranges = [v["tp_range"] for v in j3.values() if isinstance(v, dict) and "tp_range" in v]
    check(bool(tp_ranges) and max(tp_ranges) > 0,
          f"同时 TP 项确实不同（极差最大 {max(tp_ranges) if tp_ranges else 0}）"
          f" → 差异 100% 归于排序项")

    print("\n" + "=" * 84)
    if fails:
        print(f"❌ {len(fails)} 项失败")
        for f in fails:
            print("   - " + f)
        return 1
    print("✅ 全部通过 —— 分解有判别力，恒等式成立，J3 归因正确")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
