"""proxy_density.py 的自检：数学性质 + 退化情形（P15：先证明它会失败，再证明会通过）"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.proxy_density import (  # noqa: E402
    ceiling_alloc,
    ceiling_proxy,
    ceiling_uniform,
    proxies_from_prediction,
    spearman,
    uniform_cap,
)

fails = []


def check(cond: bool, msg: str) -> None:
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        fails.append(msg)


print("── 1) spearman 的数学性质 ──")
n = np.arange(1, 51, dtype=float)
check(abs(spearman(n, n) - 1.0) < 1e-9, "完美单调 → ρ = +1")
check(abs(spearman(n, -n) + 1.0) < 1e-9, "完全反向 → ρ = −1")
check(np.isnan(spearman(np.ones(5), np.arange(5.0))), "零变异 → ρ = NaN（不可用，不是 0）")

print("\n── 2) 退化情形（★ 关键：这是「会失败」的证明）──")
rng = np.random.default_rng(20261014)
counts = rng.integers(1, 200, 300).astype(float)
for B in (30, 50, 100):
    cu = ceiling_uniform(counts, B)
    ca = ceiling_alloc(counts, B)
    cap = uniform_cap(counts, B)
    cp_const = ceiling_proxy(counts, np.ones_like(counts), B)   # 常数代理
    cp_perfect = ceiling_proxy(counts, counts, B)               # 代理 = 真实密度
    check(ca >= cu - 1e-9, f"B={B}: oracle ≥ 均匀（{ca * 100:.2f}% ≥ {cu * 100:.2f}%）")
    check(abs(cp_const - ca) < 1e-6,
          f"B={B}: 常数代理 == oracle == 紧均匀上限 cap={cap}"
          f"（{cp_const * 100:.2f}% == {ca * 100:.2f}%）——★ 此式必须成立，否则函数失效")
    check(cp_perfect >= cp_const - 1e-9,
          f"B={B}: 完美代理 ≥ 常数代理（{cp_perfect * 100:.2f}% ≥ {cp_const * 100:.2f}%）")
    check(ca > cu - 1e-9 and cap >= B,
          f"B={B}: 紧 cap({cap}) ≥ B({B}) → 均匀方案存在闲置预算（这正是分配的空间）")
    print(f"        B={B:>4}  cap={cap:>4}  均匀(给B) {cu * 100:6.2f}%  "
          f"紧均匀 {ca * 100:6.2f}%  完美代理 {cp_perfect * 100:6.2f}%")

print("\n── 2b) 用真实 VisDrone val 分布核对（应与 scripts/budget_headroom.py 一致）──")
import yaml  # noqa: E402
root = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((root / "data/processed/visdrone/visdrone.yaml").read_text(encoding="utf-8"))
lbl = Path(cfg["path"]) / "labels" / "val"
if lbl.exists():
    real = np.array([sum(1 for ln in f.read_text(encoding="utf-8").strip().splitlines() if ln.strip())
                     for f in sorted(lbl.glob("*.txt"))], dtype=float)
    for B in (50, 75, 300):
        cu, ca, cap = ceiling_uniform(real, B), ceiling_alloc(real, B), uniform_cap(real, B)
        print(f"        B={B:>4}  cap={cap:>4}  均匀(给B) {cu * 100:6.2f}%  紧均匀 {ca * 100:6.2f}%"
              f"   → 分配空间 {(ca - cu) * 100:6.2f} pt")
    check(abs(ceiling_uniform(real, 300) - 0.9991) < 0.001,
          "B=300 均匀上限 ≈ 99.91%（与 budget_headroom.py 的 0.9991 一致）")
    # ★ 改成与 budget_headroom 的贪心实现**精确**对齐（两者现已同算法）
    import subprocess  # noqa: E402
    out = subprocess.run(
        [sys.executable, str(root / "scripts/budget_headroom.py"), "--splits", "val"],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = True
    for B in (50, 75, 300):
        mine = ceiling_alloc(real, B)
        # 从 budget_headroom 的输出里抓该 B 行的 oracle 列
        line = next((l for l in out.stdout.splitlines()
                     if l.strip().startswith(f"{B} ") and "%" in l), None)
        if line is None:
            ok = False
            continue
        # 列序实测： [0]=B [1]=uniform [2]=oracle分配 [3]=非平凡增量 [4]=B* [5]=受约束% [6]=饱和%
        toks = line.split()
        theirs = float(toks[2].rstrip("%")) / 100.0
        diff = abs(mine - theirs)
        # ★ 容差必须匹配**打印精度**，不是算法精度：
        #   budget_headroom 用 `%.2f` 输出百分数 → 只有 4 位有效数字，
        #   最坏舍入误差 = 0.005% = 5e-5。用 1e-6 去比是**比字符串精度、不是比算法**
        #   （实测这让我白查了一轮）。取 1e-4 留一档余量。
        same = diff < 1e-4
        ok &= same
        print(f"        交叉核对 B={B:>4}: mine={mine!r}  theirs={toks[2]!r}→{theirs!r}")
        print(f"                        diff={diff:.3e}（打印精度上限 5e-5）  {'✅' if same else '❌'}")
    check(ok, "两个脚本的 oracle 列**逐位一致**（同算法已统一）")
else:
    print(f"        （跳过：{lbl} 不存在）")

print("\n── 3) 代理量抽取 ──")
rec = {"name": "a.jpg", "n_gt": 10, "n_pred_at": {"50": 3},
       "conf_list": [0.9, 0.8, 0.3, 0.05, 0.01]}
pr = proxies_from_prediction(rec, hi_conf=0.25, top_b=50)
check(pr["P1_n_pred"] == 3.0, "P1 = n_pred_at[50] = 3")
check(pr["P3_n_hi_conf"] == 3.0, "P3 = conf ≥ 0.25 的个数 = 3")
check(abs(pr["P2_sum_conf"] - 2.06) < 1e-6, "P2 = 置信度之和 = 2.06")
check(pr["P4_entropy"] > 0, "P4 熵 > 0")
print(f"        {({k: round(v, 3) for k, v in pr.items()})}")

print("\n── 4) 判别力测试（★ 替换原先的弱单调性测试）──")
print("  原测试让 α 从 0 扫到 1，结果全是 48.91% —— 噪声尺度淹没了信号，")
print("  它**没有验证它声称的东西**。换成两个有判别力的：")

# 4a) 并列值处理：大量并列的整数代理（正是 n_pred 的形态）
a = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4], dtype=float)
b = np.array([1, 2, 3, 1, 2, 1, 2, 3, 4, 4], dtype=float)
rho_ties = spearman(a, b)
check(np.isfinite(rho_ties) and -1 <= rho_ties <= 1,
      f"整数大量并列时 ρ 仍在 [−1,1] 且有限（ρ={rho_ties:.4f}）")
# 平均秩的正确性：与手工计算的 Pearson-on-ranks 对照
from src.eval.proxy_density import _ranks  # noqa: E402
check(np.allclose(_ranks(np.array([5.0, 5.0, 5.0])), [2.0, 2.0, 2.0]),
      "并列值取平均秩（三个 5 → 秩都是 2）")
check(np.allclose(_ranks(np.array([1.0, 2.0, 2.0, 4.0])), [1.0, 2.5, 2.5, 4.0]),
      "部分并列的平均秩正确（[1,2,2,4] → [1,2.5,2.5,4]）")

# 4b) oracle 最优性：值与**大量随机合法分配**对照（比一阶条件更强、且易于正确实现）
small = rng.integers(1, 40, 60).astype(float)
for B in (10, 20, 30):
    opt = ceiling_alloc(small, B)
    total_budget = B * len(small)
    # 随机合法分配：随机顺序贪心填（结构同 ceiling_proxy，但顺序随机）
    best_rand = 0.0
    for _ in range(200):
        order = rng.permutation(len(small))
        g = np.zeros(len(small))
        left = float(total_budget)
        for i in order:
            take = min(float(small[i]), left)
            g[i] = take
            left -= take
            if left <= 0:
                break
        if left > 0:                       # 剩余预算兜底填满
            for i in rng.permutation(len(small)):
                if left <= 0:
                    break
                room = float(small[i]) - g[i]
                if room > 0:
                    take = min(room, left)
                    g[i] += take
                    left -= take
        best_rand = max(best_rand, float(np.minimum(g, small).sum() / small.sum()))
    check(opt >= best_rand - 1e-12,
          f"B={B}: oracle({opt * 100:.3f}%) ≥ 200 次随机合法分配的最大值"
          f"({best_rand * 100:.3f}%)")
    check(abs(opt - min(total_budget, small.sum()) / small.sum()) < 1e-9 or opt < 1.0,
          f"B={B}: oracle 值合理（{opt * 100:.2f}%，未超过 100%）")

print("\n" + "=" * 70)
if fails:
    print(f"❌ {len(fails)} 项失败 —— proxy_density.py 不可信")
    for f in fails:
        print("   - " + f)
    raise SystemExit(1)
print("✅ 全部自检通过（数学性质 + 退化情形 + 判别力 + VisDrone 交叉核对）")
