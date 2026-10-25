"""
scripts/run_decision_rule.py
============================
★ 判据 v2 的实施脚本（**分配线是留是砍**）

规格：`docs/prereg-decision-rule-v2-absolute.md`（第 36 轮预注册，在真实数据产出之前）
输入：`runs/budget/{exp}_{split}_{o2m,o2o}.json`（由 `src/eval/budget_scan.py` 产出）
输出：逐 split × head 的 `Δworst_abs` 与判定 A/B/C

═══════════════════════════════════════════════════════════════════════════
★ 关键修正（第 36 轮，推翻我第 24 轮的"取舍"叙述）
═══════════════════════════════════════════════════════════════════════════
**花费必须写成 `spendᵢ = min(配额ᵢ, nᵢ, pᵢ)`**（`pᵢ` = 检测器在该图的预测数）。
我第 24 轮用 `hit_rate = x/(x+κ(1−x))` 建模，**它隐含"给了配额就必须花掉"**——
这违反"检测器给不出那么多预测时可以不花"的事实 → **那版的 `Δmicro < 0` 是模型 artifice**。
`scripts/check_plan_distinctness.py` 实测：容量口径下 `Δmicro` **精确为 0**。

**→ 真实取舍不是"总量"，而是「稀疏图的召回」**：
   水位法按 `nᵢ` 比例给配额 → 密集图拿得多、**稀疏图拿得少** → 稀疏图召回下降。
   故本脚本同时报告 `macro_recall`（逐图平均）与稀疏图的召回。

═══════════════════════════════════════════════════════════════════════════
三种分配（同一总花费 S = ρ·Σnᵢ 下比较）
═══════════════════════════════════════════════════════════════════════════
  P1 均匀   ：qᵢ = c（同一配额），二分最大 c 使 Σ min(c, nᵢ, pᵢ) ≤ S
  P2 水位   ：qᵢ = min(ρ·nᵢ, pᵢ)，ρ = S/Σnᵢ
  P3 自适应 ：qᵢ = min(ρ'·nᵢ, pᵢ)，二分 ρ' 使 Σ min(qᵢ, nᵢ, pᵢ) 尽量接近 S

TPᵢ(q) = tp_prefix[min(q, len)-1]（前缀和，`tp_prefix[0]` = 取 1 个预测时的 TP）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ★ 判据阈值（预注册，不得事后更改）
THRESH_A = 10.0   # pt
THRESH_B = 5.0    # pt


def tp_at(prefix: list[int], q: int) -> int:
    """取前 q 个预测（按置信度降序）时的真阳性数。`q<=0` → 0。"""
    if q <= 0 or not prefix:
        return 0
    k = min(q, len(prefix))
    return int(prefix[k - 1])


def spend_at(n: int, p: int, q: float) -> float:
    """★ 真实花费 = min(配额, GT 数, 检测器输出数)。"""
    return float(min(q, n, p))


def evaluate(recs: list[dict], q: np.ndarray) -> dict:
    """给定逐图配额 q，算 worst / micro / macro / 花费。

    ★★ 第 54 轮修（P48）：**TP 必须在截断到 `capᵢ = min(nᵢ,pᵢ)` 的配额上取。**

    原实现的 bug：`tp` 用**原始配额** `q` 取，而 `spend` 用 `min(q, cap)`。
    当 `q > cap` 时，`tp_prefix[q]` 会取到**超过检测器物理输出上限**的 TP。
    实测（o2m）：`B = 50` 时 `q > cap` 的图数为 0（无影响）；
    **`B = 100` 时 548/548 张图全部 `q > cap`** → micro 由 59.003% 虚高到 64.669%（**+5.7 pt**），
    worst 由 25.000% 虚高到 28.333%（**+3.3 pt**）。
    而 `analyse()` 报的是 `max(worst_P2, worst_P3)`，第三个臂（自适应）在大预算下不受约束，
    **可能单独决定头条 Δ** —— 而手稿只描述了两种分配。

    修法：先把配额截到 `cap`，再用它取 TP 与花费 —— **同一个量的两个派生值必须同源**。
    """
    n = np.array([r["n_gt"] for r in recs], dtype=float)
    p = np.array([r["n_pred_at_run"] for r in recs], dtype=float)
    cap = np.minimum(n, p)
    q_eff = np.minimum(np.asarray(q, dtype=float), cap)      # ★ 唯一的有效配额
    tp = np.array([tp_at(r["tp_prefix"], int(round(qi))) for r, qi in zip(recs, q_eff)], dtype=float)
    spend = q_eff.copy()
    m = n > 0
    rec = np.where(m, tp / np.maximum(n, 1), np.nan)
    return {
        "worst": float(np.nanmin(rec)),
        "micro": float(tp.sum() / n.sum()),
        "macro": float(np.nanmean(rec)),
        "spend": float(spend.sum()),
        "q_mean": float(np.mean(q_eff)),
        "q_min": float(np.min(q_eff)),
        "q_max": float(np.max(q_eff)),
    }


def _per_image_recall(recs: list[dict], q: np.ndarray) -> np.ndarray:
    """逐图召回（供分位数报告用）。★ 同样先截到 `cap`（P48）。"""
    n = np.array([r["n_gt"] for r in recs], dtype=float)
    p = np.array([r["n_pred_at_run"] for r in recs], dtype=float)
    q_eff = np.minimum(np.asarray(q, dtype=float), np.minimum(n, p))
    tp = np.array([tp_at(r["tp_prefix"], int(round(qi))) for r, qi in zip(recs, q_eff)], dtype=float)
    n = np.array([r["n_gt"] for r in recs], dtype=float)
    return tp / np.maximum(n, 1)


def plan_uniform(recs: list[dict], S: float) -> np.ndarray:
    n = np.array([r["n_gt"] for r in recs], dtype=float)
    p = np.array([r["n_pred_at_run"] for r in recs], dtype=float)
    cap = np.minimum(n, p)
    lo, hi = 0.0, float(cap.max()) if cap.size else 0.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.minimum(mid, cap).sum() <= S:
            lo = mid
        else:
            hi = mid
    # ★ 预测数必须为整数：向上取整（见 plan_waterfill 的说明）
    return np.minimum(np.ceil(lo), cap)


# ══════════════════════════════════════════════════════════════════════
# ★★ 等花费分配（第 49 轮从 boundary_sweep_fixed.py 上移到此，**唯一定义处**）
# ══════════════════════════════════════════════════════════════════════
# 为什么必须上移（教训 P42）：我曾在 `boundary_sweep_fixed.py` 里另写一份
# "均匀 + 补足差额"的实现，而本文件的 `plan_uniform` 仍是旧版 ——
# **同一算法的两份实现静默漂移**：均匀方案超支 1.82%（`spend/S = 1.0182`），
# 于是手稿 Table 2 的 Δmicro/Δmacro 来自旧版、判据实验来自新版，
# 两处数字**对不上**（Δmicro +1.1017 vs +1.0810）。
# → 现在**只有这一处定义**，`boundary_sweep_fixed.py` 改为 import。
def _cap_of(recs: list[dict]):
    n = np.array([r["n_gt"] for r in recs], dtype=float)
    p = np.array([r["n_pred_at_run"] for r in recs], dtype=float)
    return n, p, np.minimum(n, p)


def plan_uniform_eq(recs: list[dict], S: float) -> np.ndarray:
    """均匀配额 `ceil(c)` 截到 `cap`，`c` 二分使总花费 ≤ S（取整）。"""
    _, _, cap = _cap_of(recs)
    if cap.size == 0:
        return np.zeros(0)
    lo, hi = 0.0, float(cap.max())
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.minimum(np.ceil(mid), cap).sum() <= S:
            lo = mid
        else:
            hi = mid
    return np.minimum(np.ceil(lo), cap)


def plan_waterfill_eq(recs: list[dict], S: float) -> np.ndarray:
    """★ **整数**水位，参数是**总预算 S**（不是 ρ）：`min(⌈ρ·nᵢ⌉, capᵢ)`。

    ⚠️ 早期版本把 ρ 当参数传（`plan_waterfill(recs, B)`），在 `B > mean(n)` 时
       每图配额会超过 `B` → 总花费远超预算 → 比较无效（**P39 错误①**）。
    """
    n, _, cap = _cap_of(recs)
    if cap.size == 0:
        return np.zeros(0)
    lo, hi = 0.0, 64.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.minimum(np.ceil(mid * n), cap).sum() <= S:
            lo = mid
        else:
            hi = mid
    return np.minimum(np.ceil(lo * n), cap)


_PREFIX_CACHE: dict[int, np.ndarray] = {}
_PREFIX_CACHE_MAX = 8


def prefix_matrix(recs: list[dict]) -> np.ndarray:
    """把所有 `tp_prefix` 补零成 **2D 矩阵** `T[i, k] = TPᵢ(k+1)`，便于向量化取数。

    ★ 第 49 轮（P44）：`topup` 与 bootstrap 需要**每轮**对数百张候选图取 `tp_at`，
      逐项 Python 调用实测 15–20 ms/次 → 一万次重抽样不可接受。
      补零成矩阵后一次向量索引即可。
    ⚠️ 缓存**有界**（上限 8 条）：bootstrap 每轮都新建 list，若按 `id()` 无界缓存，
      1 万轮会积压约 48 MB，且 `id()` 被回收复用后有取到**过期矩阵**的风险。
    """
    key = id(recs)
    hit = _PREFIX_CACHE.get(key)
    if hit is not None and hit.shape[0] == len(recs):
        return hit
    w = max((len(r["tp_prefix"]) for r in recs), default=1)
    T = np.zeros((len(recs), w), dtype=np.int64)
    for i, r in enumerate(recs):
        pre = r["tp_prefix"]
        if pre:
            T[i, :len(pre)] = pre
            T[i, len(pre):] = pre[-1]      # ★ 超出预测数后 TP 不再增长（tp_at 的语义）
    if len(_PREFIX_CACHE) >= _PREFIX_CACHE_MAX:
        _PREFIX_CACHE.clear()
    _PREFIX_CACHE[key] = T
    return T


def tp_at_matrix(T: np.ndarray, rows: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """向量化 `tp_at`：取 `T[rows[j], ks[j]-1]`。`rows` 与 `ks` 等长。"""
    r = np.asarray(rows, dtype=np.int64)
    k = np.clip(np.asarray(ks, dtype=np.int64) - 1, 0, T.shape[1] - 1)
    return T[r, k]


def topup(recs: list[dict], q: np.ndarray, S: float) -> np.ndarray:
    """把剩余预算按**边际 TP 增益**贪心补上，使花费尽量贴近 `S`。

    ★ 为什么需要：整数配额下二分常常欠一点。若两种分配欠得不一样多，
      比较就掺进了"谁花得多"的偏差（违反判据的 N1）。
    ★★ 第 49 轮向量化（P44）：查表用 2D 矩阵，一轮补满剩余额度。
    """
    _, _, cap = _cap_of(recs)
    T = prefix_matrix(recs)
    q = np.minimum(q, cap).astype(np.int64)
    for _ in range(40):
        deficit = S - float(np.minimum(q, cap).sum())
        if deficit <= 1e-9:
            break
        room = cap - q
        cand = np.where(room >= 1)[0]
        if cand.size == 0:
            break
        tp_now = tp_at_matrix(T, cand, np.maximum(q[cand], 1))
        tp_next = tp_at_matrix(T, cand, q[cand] + 1)
        gain = tp_next - tp_now
        order = cand[np.argsort(-gain, kind="stable")]
        take = min(int(deficit), order.size)
        q[order[:take]] += 1
    return q.astype(float)


def plan_waterfill(recs: list[dict], rho: float) -> np.ndarray:
    """★★ **整数**水位法（不是连续松弛）。

    ═══════════════════════════════════════════════════════════════════════
    为什么必须取整（第 36 轮实测踩到，务必保留）
    ═══════════════════════════════════════════════════════════════════════
    连续松弛 `qᵢ = ρ·nᵢ` 会给出**小数配额**。对 `nᵢ = 1` 的图，ρ=0.5 → `qᵢ = 0.5`
    → **一个预测都输出不了**（`int(round(0.5)) = 0`）→ 该图召回 **0**
    → **`worst = 0`** → 水位法看起来灾难性地差（实测 Δ = −9.46 pt），
    而**这是实现的错，不是方法的错**。

    正确做法是**整数水位法**（第 20/22 轮已证明，见
    `docs/proof-waterfill-optimality.md` 与 `docs/verify-int-waterfill-and-citation-discipline.md`）：

        `max{ ρ : Σᵢ ⌈ρ·nᵢ⌉ ≤ S }` ，取 `qᵢ = ⌈ρ·nᵢ⌉`

    该式在 **55,005 个组合上精确最优、0 反例**（`scripts/waterfill_dp_verify.py`）。
    **连续松弛只是它在 nᵢ 较大时的近似**；本项目 min nᵢ = 1，近似完全失效。
    """
    n = np.array([r["n_gt"] for r in recs], dtype=float)
    p = np.array([r["n_pred_at_run"] for r in recs], dtype=float)
    return np.minimum(np.ceil(rho * n), p)


def plan_adaptive(recs: list[dict], S: float) -> tuple[np.ndarray, float]:
    """二分 ρ' 使整数水位法的总花费尽量接近 S（上界仍受 pᵢ 限制）。"""
    n = np.array([r["n_gt"] for r in recs], dtype=float)
    p = np.array([r["n_pred_at_run"] for r in recs], dtype=float)
    lo, hi = 0.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.minimum(np.ceil(mid * n), p).sum() <= S:
            lo = mid
        else:
            hi = mid
    return np.minimum(np.ceil(lo * n), p), lo


def load_recs(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    recs = obj.get("per_image") or []
    missing = [r["name"] for r in recs if "tp_prefix" not in r or "n_pred_at_run" not in r]
    if missing:
        print(f"  ❌ {path.name}：{len(missing)} 条记录缺 `tp_prefix`/`n_pred_at_run`")
        print("     → 需用**新版** budget_scan 重跑该 split（第 36 轮新增的两个字段）")
    return [r for r in recs if "tp_prefix" in r and "n_pred_at_run" in r]


def _verdict(dworst: float) -> str:
    """判据 v2 的三档判定（预注册阈值，不得事后更改）。"""
    if dworst >= THRESH_A:
        return "A：分配线成立"
    if dworst >= THRESH_B:
        return "B：降级为主线一节"
    return "C：放弃分配线"


def d_worst(recs: list[dict], q_a: np.ndarray, q_b: np.ndarray) -> float:
    """两套配额之间的 **Δworst（百分点）**。与 `evaluate` 同源，不另算。"""
    return (evaluate(recs, q_b)["worst"] - evaluate(recs, q_a)["worst"]) * 100


def paired_bootstrap_dworst(recs: list[dict], rho: float, S: float,
                            n_boot: int = 10000, seed: int = 20261014) -> dict:
    """★ `Δworst_abs` 的**配对 bootstrap 置信区间**（配对单位 = 图）。

    为什么必须做：`worst` 是**极值统计**，对"哪张图恰好最差"极敏感。
    单个点估计（如 +8.013 pt）在审稿人眼里等于"没有不确定度"。
    配对 bootstrap 在**同一重抽样的图集**上同时算两种分配，从而消除图集差异带来的方差。

    `seed` 用项目预注册的 `RNG_SEED = 20261014`（`src/eval/stats.py`）。

    ★★ 正确性要点（2026-10-15 修，**P36**）：
      **绝不能用「全局索引」去取「子样本长度」的数组。**
      原实现写成
          `tp = [tp_at(recs[i]['tp_prefix'], int(round(q[i]))) for i in idx]`
      其中 `idx` 是**全局**索引、而 `q` 是**子样本**长度的配额向量 →
      `q[i]` 取到的是「第 i 个子样本位置」的配额，而不是「全局第 i 张图」的配额
      → **配对完全错位** → 曾报出 `Δ` 均值 **−0.565 pt**（真值 **+8.76 pt**），**符号都反了**。
      修法：**统一在子样本自己的记录与配额上算**（`zip(sub, q)`），不跨索引空间。
    """
    rng = np.random.default_rng(seed)
    N = len(recs)
    n = np.array([r["n_gt"] for r in recs], dtype=float)

    def worst_on(sub: list[dict], q: np.ndarray) -> float:
        """★ 在**子样本自己的**记录与配额上算 worst（不跨索引空间）。"""
        nn = np.array([r["n_gt"] for r in sub], dtype=float)
        tp = np.array([tp_at(r["tp_prefix"], int(round(qi))) for r, qi in zip(sub, q)], dtype=float)
        m = nn > 0
        return float((tp[m] / nn[m]).min()) if m.any() else np.nan

    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        sub = [recs[i] for i in idx]
        s_sub = rho * float(n[idx].sum())
        # ★★ 第 49 轮（P43）：必须与 `analyse()` **同一套**分配与同一个 Δ 定义，
        #    否则点估计与 CI 算的不是同一个量（实测报出点估计落在自己 CI 之外）。
        q_u = topup(sub, plan_uniform_eq(sub, s_sub), s_sub)
        q_w = topup(sub, plan_waterfill_eq(sub, s_sub), s_sub)
        q_a, _ = plan_adaptive(sub, s_sub)
        best = q_w if evaluate(sub, q_w)["worst"] >= evaluate(sub, q_a)["worst"] else q_a
        d[b] = d_worst(sub, q_u, best)
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {
        "mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
        "p_gt0": float((d > 0).mean()), "n_boot": int(len(d)), "seed": seed,
    }


def analyse(recs: list[dict], rho: float, label: str,
            n_boot: int = 0, quiet: bool = False) -> dict:
    n = np.array([r["n_gt"] for r in recs], dtype=float)
    p = np.array([r["n_pred_at_run"] for r in recs], dtype=float)
    TOT = float(n.sum())
    S = rho * TOT

    if not quiet:
        print(f"\n{'=' * 100}\n### {label}   N={len(recs)}  Σnᵢ={TOT:.0f}  Σpᵢ={p.sum():.0f}"
              f"  S(ρ={rho})={S:.0f}\n{'=' * 100}")
        if p.sum() < S:
            print(f"  ⚠️ **预算花不完**：Σpᵢ={p.sum():.0f} < S={S:.0f}")

    # ★★ 第 49 轮修正：三种方案统一走 `*_eq` + `topup` —— 等花费分配（唯一定义处）。
    #    旧版 P1 用 `plan_uniform`（超支 1.82%）、P2 用 `plan_waterfill`（不 topup），
    #    与 `boundary_sweep_fixed.py` 的实现**静默漂移**，导致
    #    Table 2 的 Δmicro/Δmacro 与判据实验对不上（P42）。
    q1 = topup(recs, plan_uniform_eq(recs, S), S)
    q2 = topup(recs, plan_waterfill_eq(recs, S), S)
    q3, rho3 = plan_adaptive(recs, S)
    r1, r2, r3 = evaluate(recs, q1), evaluate(recs, q2), evaluate(recs, q3)

    if not quiet:
        print(f"\n  {'方案':<12} {'worst':>9} {'micro':>9} {'macro':>9} {'Σspend':>10} "
              f"{'spend/S':>9} {'q 范围':>16}")
        print("  " + "-" * 96)
        for tag, r, extra in (("P1 均匀", r1, ""), ("P2 水位", r2, ""),
                              ("P3 自适应", r3, f"  ρ'={rho3:.4f}")):
            print(f"  {tag:<12} {r['worst'] * 100:>8.3f}% {r['micro'] * 100:>8.3f}% "
                  f"{r['macro'] * 100:>8.3f}% {r['spend']:>10.0f} {r['spend'] / S:>8.4f} "
                  f"[{r['q_min']:.1f}, {r['q_max']:.1f}]{extra}")

    # ★ 判据量
    best_w = max(r2["worst"], r3["worst"])
    dworst = (best_w - r1["worst"]) * 100
    dmicro = (r2["micro"] - r1["micro"]) * 100
    dmacro = (r2["macro"] - r1["macro"]) * 100
    rel_err = max(abs(r2["spend"] - S), abs(r3["spend"] - S)) / S

    boot = None
    if n_boot and n_boot > 0:
        boot = paired_bootstrap_dworst(recs, rho, S, n_boot=n_boot)

    if quiet:
        return {"label": label, "dworst": dworst, "dmicro": dmicro, "dmacro": dmacro,
                "rel_err": rel_err, "verdict": _verdict(dworst), "ok": None,
                "p1": r1, "p2": r2, "p3": r3, "boot": boot}

    print(f"\n  ★ Δworst_abs（水位/自适应 较优者 − 均匀）= **{dworst:+.3f} pt**")
    print(f"     Δmicro = {dmicro:+.3f} pt      Δmacro（逐图）= {dmacro:+.3f} pt")
    print(f"     N1 花费相对误差 = {rel_err * 100:.2f}%  "
          f"{'✅' if rel_err <= 0.01 else '❌ 超过 1%，同花费比较不成立'}")
    print(f"     N2 严格更优 = {'✅' if dworst > 0 else '❌ 平局，无发现'}")
    print(f"     N3 Δmicro = {dmicro:+.3f} pt  "
          f"{'⚠️ 超过 −5 pt，摘要须写成取舍' if dmicro < -5 else '✅ 未超 −5 pt'}")

    if boot:
        print(f"\n  ★★ 配对 bootstrap（配对单位 = 图，{boot['n_boot']} 次，seed={boot['seed']}）")
        print(f"     Δworst_abs 的 95% CI = **[{boot['lo']:+.3f}, {boot['hi']:+.3f}] pt**"
              f"   均值 {boot['mean']:+.3f}")
        print(f"     P(Δ > 0) = **{boot['p_gt0'] * 100:.2f}%**")
        if boot["lo"] <= 0:
            print("     ⚠️ **CI 跨 0** → 不能声称「分配改善最坏图召回」是一个稳健效应")
        print(f"     ★ 统计判读：CI 上限 {boot['hi']:+.3f} pt "
              f"{'**仍小于 A 阈值 10 pt** → 阴性结论对抽样不确定性稳健' if boot['hi'] < THRESH_A else '**超过 A 阈值** → 阴性结论不稳健，须重估'}")

    # ★ 真实取舍所在：水位法把配额从**稀疏图**抽走 → 它们的召回下降。
    #   故必须按 nᵢ 分位报告逐图召回的分布，而不是只报 worst / micro / macro。
    print("\n  ★ 逐图召回按 nᵢ 分位（看清「谁被牺牲了」）")
    print(f"    {'nᵢ 分位':<16} {'图数':>6} {'nᵢ 范围':>14} │ {'P1 均匀':>10} {'P2 水位':>10} "
          f"{'Δ pt':>9}")
    print("    " + "-" * 76)
    rec1 = _per_image_recall(recs, q1)
    rec2 = _per_image_recall(recs, q2)
    n_arr = np.array([r["n_gt"] for r in recs], dtype=float)
    qs = [(0, 5, "最稀疏 5%"), (5, 25, "5–25%"), (25, 50, "25–50%"),
          (50, 75, "50–75%"), (75, 100, "最密 25%")]
    # ★ 必须用**稳定排序**（P49）：`n_i` 有大量并列值，默认 quicksort 会让并列元素
    #   跨越分带边界 → 分带成员不确定 → 同一份数据在不同脚本里给出不同分量。
    order = np.argsort(n_arr, kind="stable")
    for lo, hi, lab in qs:
        i0 = int(lo / 100 * len(order))
        i1 = max(i0 + 1, int(hi / 100 * len(order)))
        idx = order[i0:i1]
        a1, a2 = float(np.mean(rec1[idx])), float(np.mean(rec2[idx]))
        print(f"    {lab:<16} {len(idx):>6} [{n_arr[idx].min():>5.0f},{n_arr[idx].max():>6.0f}] │ "
              f"{a1 * 100:>9.3f}% {a2 * 100:>9.3f}% {(a2 - a1) * 100:>+8.3f}")

    verdict = ("A：分配线成立" if dworst >= THRESH_A else
               "B：降级为主线一节" if dworst >= THRESH_B else "C：放弃分配线")
    ok = (rel_err <= 0.01) and (dworst > 0)
    print(f"\n  ★ **判定：{verdict}**（Δworst_abs={dworst:+.3f} pt，"
          f"阈值 A≥{THRESH_A} / B≥{THRESH_B}）")
    if not ok:
        print("     ⚠️ 附加必要条件未全满足 → 即使 Δworst_abs 达标也须降级")
    # ★ 必须把 `boot` 一起返回：早期版本这里漏了，导致 CLI 的 `--json` 输出**只有 bootstrap
    #   的控制台打印、而没有落盘**，下游核验脚本因此读不到 CI（P38）。
    return {"label": label, "dworst": dworst, "dmicro": dmicro, "dmacro": dmacro,
            "rel_err": rel_err, "verdict": verdict, "ok": ok,
            "p1": r1, "p2": r2, "p3": r3, "boot": boot}


def main() -> int:
    ap = argparse.ArgumentParser(description="判据 v2：分配线是留是砍")
    ap.add_argument("--exp", default="r1")
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--heads", nargs="+", default=["o2m", "o2o"])
    ap.add_argument("--rho", type=float, default=0.5,
                    help="S/Σnᵢ；预注册值为 0.5")
    ap.add_argument("--rho-sweep", nargs="+", type=float, default=None,
                    help="★ 敏感性：同时报这些 ρ（**纪律要求：不得只报最好的那个**）")
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="配对 bootstrap 次数（配对单位 = 图）；0 = 跳过")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out_dir = PROJECT_ROOT / "runs" / "budget"

    # ── ★ ρ 敏感性（零 GPU：从已落盘的 tp_prefix 离线算任意 ρ）─────────
    if a.rho_sweep:
        print("=" * 100)
        print("★ ρ 敏感性（全部 ρ 一并报告 —— 纪律要求：不得只报最好的那个）")
        print("=" * 100)
        for split in a.splits:
            for head in a.heads:
                p = out_dir / f"{a.exp}_{split}_{head}.json"
                if not p.exists():
                    continue
                recs = load_recs(p)
                if not recs:
                    continue
                print(f"\n### {split} / {head}")
                print(f"  {'ρ':>6} {'worst均匀':>11} {'worst水位':>11} {'Δworst_abs':>12} "
                      f"{'Δmicro':>10} {'判定':>22}")
                print("  " + "-" * 82)
                for rho in a.rho_sweep:
                    r = analyse(recs, rho, f"{split}/{head}@ρ={rho}", quiet=True)
                    print(f"  {rho:>6.2f} {r['p1']['worst'] * 100:>10.3f}% "
                          f"{r['p2']['worst'] * 100:>10.3f}% "
                          f"{r['dworst']:>+11.3f}p {r['dmicro']:>+9.3f}p {r['verdict']:>22}")
                print("\n  ★ 判读：若 Δworst_abs 在各 ρ 下**符号一致**且量级同阶 → 结论对 ρ 稳健；"
                      "\n     若跨阈值（A/B/C）→ **必须在论文中如实报告该敏感性**。")
        print()

    results = []
    for split in a.splits:
        for head in a.heads:
            p = out_dir / f"{a.exp}_{split}_{head}.json"
            if not p.exists():
                print(f"\n⚠️ 缺 {p.name} —— 跳过（需先跑 budget_scan）")
                continue
            recs = load_recs(p)
            if not recs:
                continue
            results.append(analyse(recs, a.rho, f"{split} / {head}", n_boot=a.n_boot))

    if not results:
        print("\n❌ 没有可用输入。先跑**分块**扫描（`--limit 548` 会 CUDA OOM）：")
        print("   python scripts/budget_scan_chunked.py --split val --head o2m")
        print("   python scripts/budget_scan_chunked.py --split val --head o2o")
        return 1

    print(f"\n{'=' * 100}\n★ 汇总（判据 v2）\n{'=' * 100}")
    print(f"  {'输入':<20} {'Δworst_abs':>12} {'Δmicro':>10} {'Δmacro':>10} {'判定':>22}")
    print("  " + "-" * 80)
    for r in results:
        print(f"  {r['label']:<20} {r['dworst']:>+11.3f}p {r['dmicro']:>+9.3f}p "
              f"{r['dmacro']:>+9.3f}p {r['verdict']:>22}")
    worst = min(results, key=lambda x: x["dworst"])
    print(f"\n  ★ **保守判定（取最差输入）：{worst['verdict']}**"
          f"（{worst['label']}，Δworst_abs={worst['dworst']:+.3f} pt）")

    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  已写入 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
