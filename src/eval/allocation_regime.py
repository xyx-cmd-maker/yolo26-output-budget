"""
src/eval/allocation_regime.py
=============================
★★ **分配机制的正式判据** —— "谁在卡住最坏图"

═══════════════════════════════════════════════════════════════════════════
本模块解决什么问题
═══════════════════════════════════════════════════════════════════════════
给定逐图的 `(nᵢ, pᵢ, tp_prefixᵢ)` 与一个总预算 `S`，**在跑实验之前**回答：

    在这个预算下，把配额按 `nᵢ` 比例重新分配，**会不会**改善最坏图召回？

**为什么需要它**：我此前用**标签侧的量**（`util(B)`、`frac(nᵢ>B)`）去预测这件事，
**方向都搞反了**（`r = −0.31`；见 AGENTS-LESSONS **P39**）。
根因是**范畴错误**：那些量只描述**标签分布**，而"分配能否改善 `worst`"取决于
**检测器在最坏那张图上的表现**。

═══════════════════════════════════════════════════════════════════════════
判据（全部逐图可验，不依赖任何数据集层面的量）
═══════════════════════════════════════════════════════════════════════════
对任一分配 `q`，定义

    `capᵢ = min(nᵢ, pᵢ)`       该图**物理上能输出**的最大框数
    `spendᵢ = min(qᵢ, capᵢ)`    该图实际花掉的配额
    `recallᵢ = TPᵢ(spendᵢ) / nᵢ`

令 `i* = argmin recallᵢ`（最坏图）。**三种 regime**：

  | regime | 条件 | 对 Δworst 的预测 |
  |---|---|---|
  | **`DETECTOR_BOUND`** | 两种分配的**最坏图都已 `spend ≥ cap`** | **≈ 0** —— 最坏图已到物理上限，任何分配改不动 |
  | **`BUDGET_BOUND`** | 至少一方的最坏图 `spend < cap` | **可能 > 0** —— 最坏图被配额卡住，重分配有空间 |
  | **`MIXED`** | 两种分配的 `spend` 相对差 > 1% | **不可比**（违反同花费前提） |

  ★ `nᵢ ≤ 2` 的图被整数取整成 0 配额的情况**不作为独立 regime**，而是作为
    **`starved_units` 计数 + `degenerate` 标志**报告（见下）。

**判据的核心**：`Δworst ≈ 0` 的充要条件是**两种分配的最坏图都已 `spend ≥ cap`**。
这不需要拟合，也不需要任何数据集层面的量（如 `util(B)`）。

═══════════════════════════════════════════════════════════════════════════
⚠️ `starved_units`：一个**必须单独报告**的退化情形
═══════════════════════════════════════════════════════════════════════════
整数配额的算术是 `⌈ρ·nᵢ⌉`。当 `ρ < 1`（等价于**总预算小于图的张数**，
即平均每图不到 1 个框）时，`nᵢ = 1` 的图会得到 `⌈ρ⌉ = 1`，
而 `nᵢ = 1` 且 `ρ` 更小时可能落到 0 —— 此时该图召回为 **0**，
**水位法反而比均匀法更差**（实测 Δ 为负，见 `docs/finding-alloc-mechanism-who-binds-worst-image.md` §3）。

**为什么它不是独立 regime**：该情形只在 `B ≲ 1` 出现，而那时**预算小到连"每图一个"都给不起**，
`topup()` 也无处可补 —— 所以它描述的是**分配本身退化**，不是"另一种绑定机制"。
**处置**：`starved_units > 0` 时 `Diagnosis.degenerate = True`，
**调用方应把该档位标注为"分配退化"而非据此推断机制。**


═══════════════════════════════════════════════════════════════════════════
用法
═══════════════════════════════════════════════════════════════════════════
    from src.eval.allocation_regime import diagnose, plan_uniform_eq, plan_waterfill_eq
    d = diagnose(recs, S)
    print(d.regime, d.verdict, d.gain_pt)

`recs` 的每项形如 `{"n_gt": int, "n_pred_at_run": int, "tp_prefix": list[int]}`，
即 `src/eval/budget_scan.py` 落盘的 `per_image` 格式。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ── regime 常量 ──────────────────────────────────────────────────────
DETECTOR_BOUND = "DETECTOR_BOUND"
BUDGET_BOUND = "BUDGET_BOUND"
STARVED = "STARVED"
MIXED = "MIXED"

# ── 增益判定的点数阈值（与预注册判据一致，见 docs/prereg-decision-rule-v2-absolute.md）──
THRESH_MEANINGFUL_PT = 5.0


# ═══════════════════════════════════════════════════════════════════════
# ★★ 第 54 轮（P47/P48）：**分配与取数函数改为从唯一实现导入，本文件不再自行定义。**
# ═══════════════════════════════════════════════════════════════════════
# 此前本文件与 `scripts/run_decision_rule.py` **各有一份** `tp_at` /
# `plan_uniform_eq` / `plan_waterfill_eq` / `topup`，而且 **`topup` 的填充规则不同**：
#   · 本文件（旧）：每次迭代只补 **1 个单位**给边际增益最大的图
#   · run_decision_rule（新）：向量化，**一轮补满**剩余额度
# → 两者对同一条件给出**不同的 Δ**（B=15 时 6.695 vs 6.379），
#   于是手稿的 Table 4（来自 `diagnose`）与 Table 2（来自 `analyse`）**互相对不上**。
# **这是 P42「同一算法两份实现静默漂移」的复发 —— 我上一次只修了一半。**
#
# 现在**唯一的定义处是 `scripts/run_decision_rule.py`**；本文件全部改为导入。
# 若该脚本不可导入（例如在仅有 `src/` 的部署里），才回退到本地实现并**显式警告**。
def _import_single_source():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "run_decision_rule.py"
    if not path.exists():
        return None
    name = "_rdr_single_source"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_RDR = _import_single_source()

if _RDR is not None:
    tp_at = _RDR.tp_at
    prefix_matrix = _RDR.prefix_matrix
    tp_at_matrix = _RDR.tp_at_matrix
    plan_uniform_eq = _RDR.plan_uniform_eq
    plan_waterfill_eq = _RDR.plan_waterfill_eq
    topup = _RDR.topup
    evaluate = _RDR.evaluate
    _cap_of = _RDR._cap_of
else:  # pragma: no cover - 仅在脱离项目根时触发
    import warnings

    warnings.warn(
        "allocation_regime: 找不到 scripts/run_decision_rule.py，"
        "正在使用本文件的**回退实现**；其结果可能与判据脚本不一致。",
        RuntimeWarning, stacklevel=2,
    )

    def tp_at(prefix, q):  # type: ignore[misc]
        if q <= 0 or not prefix:
            return 0
        return int(prefix[min(int(q), len(prefix)) - 1])

    def _cap_of(recs):  # type: ignore[misc]
        n = np.array([r["n_gt"] for r in recs], dtype=float)
        p = np.array([r["n_pred_at_run"] for r in recs], dtype=float)
        return n, p, np.minimum(n, p)


def cap_of(recs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 `(n, p, cap)`，其中 `cap = min(n, p)`。

    ⚠️ **第 54 轮的诚实记录**：本文件此前把 `cap` 描述为"物理输出上限"，
      但实测 **`pᵢ < nᵢ` 在全部 2,192 条记录上从未发生**（四个数据集划分，
      `Σmin(nᵢ,pᵢ) = Σnᵢ = 38,759`）→ **`capᵢ` 恒等于 `nᵢ`，`pᵢ` 那一半是惰性的**。
      这使"判据"退化为标签侧谓词。详见 `docs/finding-CORE-CLAIM-FALSIFIED.md`。
    """
    return _cap_of(recs)


def spend_of(recs: list[dict], q: np.ndarray) -> np.ndarray:
    """实际花掉的配额 `min(qᵢ, capᵢ)` —— **不是** `qᵢ` 本身。"""
    _, _, cap = cap_of(recs)
    return np.minimum(q, cap)


def recall_of(recs: list[dict], q: np.ndarray) -> np.ndarray:
    """逐图召回 `TPᵢ(spendᵢ)/nᵢ`（`nᵢ = 0` 的图返回 NaN）。"""
    n, _, _ = cap_of(recs)
    sp = spend_of(recs, q)
    tp = np.array([tp_at(r["tp_prefix"], int(s)) for r, s in zip(recs, sp)], dtype=float)
    out = np.full(len(recs), np.nan)
    m = n > 0
    out[m] = tp[m] / n[m]
    return out


@dataclass
class Diagnosis:
    """`diagnose()` 的返回。字段全部可直接进论文表格。"""

    regime: str
    verdict: str
    S: float
    spend_total: float
    spend_over_S: float
    n_units: int
    sum_n: float
    sum_cap: float
    # 最坏图（两种分配各自）
    worst_idx_uniform: int = -1
    worst_recall_uniform: float = float("nan")
    worst_idx_waterfill: int = -1
    worst_recall_waterfill: float = float("nan")
    gain_pt: float = float("nan")
    # 触顶信息
    uniform_worst_at_cap: bool = False
    waterfill_worst_at_cap: bool = False
    starved_units: int = 0
    single_pred_units: int = 0
    degenerate: bool = False
    notes: list[str] = field(default_factory=list)


def diagnose(recs: list[dict], S: float) -> Diagnosis:
    """★ **主判据**：给定逐图记录与总预算，判断 regime 与预测增益方向。

    步骤：
      1. 造两种同花费分配（均匀 / 整数水位），都 `topup` 到 `S`（保证可比）
      2. 核 N1：两者 `spend/S` 的相对差 > 1% → 标 `MIXED` 并提示不可比
      3. 找各自的最坏图，检查它是否 `spend ≥ cap`（触顶）
      4. 按 §判据 表定 regime
    """
    n, p, cap = cap_of(recs)
    N = len(recs)
    if N == 0:
        return Diagnosis(MIXED, "空输入", S, 0.0, float("nan"), 0, 0.0, 0.0)

    q_u = topup(recs, plan_uniform_eq(recs, S), S)
    q_w = topup(recs, plan_waterfill_eq(recs, S), S)
    sp_u, sp_w = spend_of(recs, q_u), spend_of(recs, q_w)
    rec_u, rec_w = recall_of(recs, q_u), recall_of(recs, q_w)

    i_u = int(np.nanargmin(rec_u)) if np.isfinite(rec_u).any() else -1
    i_w = int(np.nanargmin(rec_w)) if np.isfinite(rec_w).any() else -1
    gain = float((rec_w[i_w] - rec_u[i_u]) * 100) if i_u >= 0 and i_w >= 0 else float("nan")

    at_cap_u = bool(i_u >= 0 and sp_u[i_u] >= cap[i_u] - 1e-9)
    at_cap_w = bool(i_w >= 0 and sp_w[i_w] >= cap[i_w] - 1e-9)

    # ★★ 退化判定的**两个**来源（都比较"最坏图是否只剩 1 个预测可赌"）：
    #   (a) 配额为 0 —— `nᵢ` 很小的图被整数取整成 0（`B ≲ 1` 时）
    #   (b) **配额恰为 1** —— 此时最坏图的召回 ∈ {0, 1/nᵢ}，
    #       完全由"**头号预测对不对**"决定，是**单预测噪声**而非机制。
    #       实测来源：VisDrone-val `B = 2` 时，水位的最坏图配额 = 1、其 top-1 是假阳
    #       → 召回 0 → 出现 −0.631 pt 的"负增益"，与机制无关。
    zero_mask = (n <= 4) & ((sp_u <= 0) | (sp_w <= 0))
    one_mask = ((sp_u <= 1) | (sp_w <= 1)) & (n >= 4)     # 配额 1 且 nᵢ 不至于太小
    starved = int(zero_mask.sum())
    single_pred = int(one_mask.sum())

    # ── regime 判定 ────────────────────────────────────────────────
    notes: list[str] = []
    spend_u, spend_w = float(sp_u.sum()), float(sp_w.sum())
    rel = abs(spend_u - spend_w) / max(spend_w, 1e-9)

    if rel > 0.01:
        regime, verdict = MIXED, "N1 未过：两种分配花费不对等，**不可比**"
        notes.append(f"spend 相对差 {rel * 100:.2f}% > 1%")
    elif at_cap_u and at_cap_w:
        regime, verdict = DETECTOR_BOUND, "预测增益 **≈ 0**：两种分配的最坏图均已触顶 `min(n,p)`"
    elif at_cap_u and not at_cap_w:
        regime, verdict = BUDGET_BOUND, "预测增益 **> 0**：均匀的最坏图触顶、水位的不触顶"
    else:
        regime, verdict = BUDGET_BOUND, "预测增益 **> 0 可能**：至少一方的最坏图未触顶"
        if at_cap_w and not at_cap_u:
            notes.append("水位的最坏图触顶而均匀的未触顶 —— 增益符号需实测")

    degenerate = (starved > 0) or (single_pred > 0)
    if starved > 0:
        notes.insert(0, f"⚠️ 分配退化(a)：{starved} 张 nᵢ≤4 的图有一侧配额为 0"
                        f"（预算不足以每图一个）")
    if single_pred > 0:
        notes.insert(0, f"⚠️ 分配退化(b)：{single_pred} 张图有一侧配额恰为 1 → "
                        f"最坏图召回由**头号预测对错**决定（单预测噪声，非机制）")
    if degenerate:
        notes.append("→ **该档位不应据以推断机制**；请提高预算后重测")

    return Diagnosis(
        regime=regime, verdict=verdict, S=S, spend_total=spend_w,
        spend_over_S=spend_w / max(S, 1e-9), n_units=N,
        sum_n=float(n.sum()), sum_cap=float(cap.sum()),
        worst_idx_uniform=i_u,
        worst_recall_uniform=float(rec_u[i_u]) if i_u >= 0 else float("nan"),
        worst_idx_waterfill=i_w,
        worst_recall_waterfill=float(rec_w[i_w]) if i_w >= 0 else float("nan"),
        gain_pt=gain,
        uniform_worst_at_cap=at_cap_u, waterfill_worst_at_cap=at_cap_w,
        starved_units=starved, single_pred_units=single_pred,
        degenerate=degenerate, notes=notes,
    )


def main() -> int:
    """命令行：对一个已落盘的 `per_image` JSON 跑一个预算区间的判据。"""
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="分配机制判据（谁卡住最坏图）")
    ap.add_argument("json", type=Path, help="budget_scan 落盘的 per_image JSON")
    ap.add_argument("--budgets", type=int, nargs="+", default=[10, 25, 50, 70, 100, 300])
    a = ap.parse_args()

    obj = json.loads(a.json.read_text(encoding="utf-8"))
    recs = obj.get("per_image") or []
    if not recs:
        print("❌ 该 JSON 没有 per_image")
        return 1
    N = len(recs)
    print("=" * 104)
    print(f"{a.json.name}   N={N}")
    print("=" * 104)
    print(f"  {'S':>8} {'regime':>16} {'spend/S':>9} │ {'最坏图(均)':>12} {'到顶':>5} │ "
          f"{'最坏图(水)':>12} {'到顶':>5} │ {'Δworst 实测':>12} {'判据预测':>10}")
    print("  " + "-" * 100)
    for B in a.budgets:
        d = diagnose(recs, float(B * N))
        pred = {DETECTOR_BOUND: "≈ 0", BUDGET_BOUND: "> 0", STARVED: "< 0", MIXED: "不可比"}[d.regime]
        print(f"  {B * N:>8.0f} {d.regime:>16} {d.spend_over_S:>9.4f} │ "
              f"{d.worst_recall_uniform * 100:>11.3f}% {'是' if d.uniform_worst_at_cap else '否':>5} │ "
              f"{d.worst_recall_waterfill * 100:>11.3f}% {'是' if d.waterfill_worst_at_cap else '否':>5} │ "
              f"{d.gain_pt:>+11.3f}p {pred:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
