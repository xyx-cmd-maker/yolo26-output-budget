"""
yolo26-detection / src/eval/stats.py
====================================
AGENTS.md §0.4 要求的统计检验的**唯一实现**。

为什么必须有这个文件（修 B10）：
    §0.4 规定了「配对 bootstrap 或 Wilcoxon + Holm 校正 + 效应量 CI + 等价检验」，
    但此前整个仓库没有任何代码实现这些。没有实现的规定 = 没规定。

设计约束：
  * **只用标准库 + numpy**。scipy 不是 ultralytics 的依赖，未必装了；
    本模块在 scipy 缺失时降级到精确/置换实现，并在结果里标注用了哪条路径。
  * n 通常很小（3–5 个种子）。**小样本下不要相信正态近似**：
    bootstrap 用 BCa 之外的 percentile 法并显式报告 n；
    Wilcoxon 用精确分布（n<=25 时可精确枚举）。
  * 所有函数返回结构化 dict，字段名与论文表格一一对应，避免转录错误。

术语（与 AGENTS.md §0.6 一致）：
  * 配对：同一 (dataset, imgsz, seed) 下方法 A 与 B 的对比 —— 这才是本项目的主分析单元。
  * 效应量：ΔmAP = mean(A) − mean(B)；同时给配对差的中位数与 95% CI。
  * 等价检验 TOST：要主张"无差异"时必须做，并**预先**声明边界 margin。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

__all__ = [
    "paired_bootstrap",
    "wilcoxon_signed_rank",
    "holm_bonferroni",
    "tost_equivalence",
    "compare_methods",
    "summarize",
]

RNG_SEED = 20261014  # 固定，保证 bootstrap 结果可复现

# ★ 预注册的等价边界（见 docs/decisions.md D1）。
#   定义：ΔmAP50-95 的 95% CI 完全落在 (-0.010, +0.010) 内时判等价。
#   依据：检测文献普遍把 <1 点 mAP 视为种子噪声量级；且本项目目标为交通标志/行人/
#   信号灯等安全相关目标，不应把边界放宽到 2-3 点。
#   ⚠️ 修改此值必须在 docs/decisions.md 留痕并说明理由，禁止事后挑选。
#   ⚠️ 2026-10-14 修订：本值口径为 **mAP50-95 绝对差**；数据集已由自建
#      dataset125（3 类）换为 VisDrone2019-DET（10 类，test-dev 1,610 图 / 75,102 框）。
#      数值不变，理由与"规模不足导致 TOST 无功效"的旧推论已更新（见 D1 第 3 条）。
EQUIVALENCE_MARGIN = 0.010
EQUIVALENCE_MARGIN_SOURCE = (
    "预注册（docs/decisions.md D1，2026-10-14）：检测文献中 <1 点 mAP 属种子噪声量级；"
    "本项目目标为安全相关目标，边界不宜放宽"
)

# ★★ 召回版等价边界（见 docs/decisions.md D1b，2026-10-14 新增）。
#   为什么必须与上面分开：本项目的预算实验主因变量是**逐图召回**，而 mAP 与召回
#   **不是同一个量纲**——DQ-DETR 自己的 Table 6 就显示"收紧预算 AP 反升"，二者可反向。
#   用 mAP 的 margin 去判定召回等价 = 量纲错误，会被统计审稿人直接指出。
#   取值与 D1 同为数 0.010 仅为叙说一致；**两者必须分别引用，不得互相替代**。
EQUIVALENCE_MARGIN_RECALL = 0.010
EQUIVALENCE_MARGIN_RECALL_SOURCE = (
    "预注册（docs/decisions.md D1b，2026-10-14）：召回与 mAP 量纲不同，"
    "mAP 的等价边界不可用于召回；取同值 0.010 以保持跨指标叙说一致"
)


# --------------------------------------------------------------------------
# 1. 配对 bootstrap
# --------------------------------------------------------------------------
def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """配对 bootstrap：估计 Δ = A − B 的均值与 (1−alpha) 百分位 CI，以及双侧 p 值。

    p 值的定义：在"Δ 的真实分布关于 0 对称"的零假设下，观测到 |Δ| 这么极端的概率。
    实现方式是对**去中心化后的配对差**重采样（即零假设下的 bootstrap），
    这比用正态近似更适合 n=3~5。

    Args:
        a, b: 配对的两个方法在**同一种子序列**下的主指标值。
        n_boot: 重采样次数。论文里 >=10000。
        alpha: 显著性水平。
        seed: RNG 种子（固定以保证可复现）。

    Returns:
        dict：n、均值差、中位数差、CI、p、以及重采样后 Δ=0 的占比参考值。
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"配对样本长度必须相同：len(a)={len(a_arr)}, len(b)={len(b_arr)}")
    n = len(a_arr)
    if n < 2:
        raise ValueError(f"配对 bootstrap 至少需要 2 对样本，得到 {n}")

    d = a_arr - b_arr
    obs = float(d.mean())
    rng = np.random.default_rng(seed)

    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)  # 观测分布下的 Δ 抽样分布

    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))

    # 零假设下的 bootstrap：把 d 去中心化后重采样
    d0 = d - obs
    boot0 = d0[idx].mean(axis=1)
    p = float((np.abs(boot0) >= abs(obs)).mean())
    p = max(p, 1.0 / n_boot)  # 避免报 0（n_boot 次里一次都没抽到不等于概率为 0）

    return {
        "test": "paired_bootstrap",
        "n_pairs": n,
        "n_boot": n_boot,
        "delta_mean": round(obs, 5),
        "delta_median": round(float(np.median(d)), 5),
        "delta_std": round(float(d.std(ddof=1)) if n > 1 else 0.0, 5),
        "ci_low": round(lo, 5),
        "ci_high": round(hi, 5),
        "ci_level": 1 - alpha,
        "p_value": round(p, 5),
        "per_seed_deltas": [round(float(x), 5) for x in d],
        "note": f"percentile bootstrap，n={n}；小样本下 CI 偏窄，论文中须同时报 n 与逐种子值",
    }


# --------------------------------------------------------------------------
# 2. Wilcoxon 符号秩检验
# --------------------------------------------------------------------------
def _wilcoxon_exact_p(d: np.ndarray, w_obs: float) -> float | None:
    """n<=25 时精确枚举 2^n 种符号组合，返回双侧精确 p。

    （n>25 时返回 None，调用方改用正态近似。）
    """
    n = len(d)
    if n > 25:
        return None
    absd = np.abs(d)
    # 去掉零差
    nz = absd > 0
    absd = absd[nz]
    n = len(absd)
    if n == 0:
        return 1.0
    if n > 25:
        return None

    ranks = _average_ranks(absd)
    total = 0
    count = 0
    for mask in range(1 << n):
        s = 0.0
        for i in range(n):
            if mask >> i & 1:
                s += ranks[i]
        if abs(s - ranks.sum() / 2) >= abs(w_obs - ranks.sum() / 2) - 1e-12:
            count += 1
        total += 1
    return count / total


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """平均秩（处理并列）。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Wilcoxon 符号秩检验（配对）。n<=25 走精确枚举，否则正态近似 + 连续性校正。

    不依赖 scipy。
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("配对样本长度必须相同")
    d = a_arr - b_arr
    n_all = len(d)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {
            "test": "wilcoxon_signed_rank",
            "n_pairs": n_all,
            "n_nonzero": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "all_zero",
            "note": "所有配对差为 0，检验无信息",
        }

    ranks = _average_ranks(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    stat = min(w_plus, w_minus)

    exact_p = _wilcoxon_exact_p(d, stat)
    if exact_p is not None:
        return {
            "test": "wilcoxon_signed_rank",
            "n_pairs": n_all,
            "n_nonzero": n,
            "statistic": round(stat, 3),
            "w_plus": round(w_plus, 3),
            "w_minus": round(w_minus, 3),
            "p_value": round(exact_p, 5),
            "method": "exact_enumeration",
            "note": f"n={n} <= 25，精确分布；n 很小时最小可能 p 受限于 2^-n",
        }

    mu = n * (n + 1) / 4
    # 并列校正
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = float(((counts**3 - counts).sum()) / 48.0)
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie_term)
    if sigma == 0:
        return {
            "test": "wilcoxon_signed_rank",
            "n_pairs": n_all,
            "n_nonzero": n,
            "statistic": round(stat, 3),
            "p_value": 1.0,
            "method": "degenerate",
            "note": "秩和为常数，无法计算 p",
        }
    z = (stat - mu + 0.5) / sigma  # 连续性校正
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {
        "test": "wilcoxon_signed_rank",
        "n_pairs": n_all,
        "n_nonzero": n,
        "statistic": round(stat, 3),
        "z": round(z, 4),
        "p_value": round(min(max(p, 0.0), 1.0), 5),
        "method": "normal_approximation_with_tie_correction",
        "note": f"n={n} > 25，正态近似 + 连续性校正",
    }


# --------------------------------------------------------------------------
# 3. Holm–Bonferroni 多重比较校正
# --------------------------------------------------------------------------
def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, Any]:
    """Holm–Bonferroni 逐步下降法。比 Bonferroni 有更高功效且同样控制 FWER。

    Args:
        p_values: {比较名: p 值}
        alpha: 族错误率上限

    Returns:
        dict：每个比较的校正后阈值、是否拒绝、以及 step-down 的停止位置。
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    results: dict[str, Any] = {}
    rejected_all = True
    for i, (name, p) in enumerate(items):
        thr = alpha / (m - i)
        reject = rejected_all and (p <= thr)
        if not reject:
            rejected_all = False
        results[name] = {
            "p_raw": round(float(p), 6),
            "rank": i + 1,
            "holm_threshold": round(thr, 6),
            "reject": bool(reject),
        }
    return {
        "test": "holm_bonferroni",
        "alpha": alpha,
        "n_comparisons": m,
        "results": results,
        "note": "逐步下降：一旦某步不拒绝，其后一律不拒绝（即使个别 p 很小）",
    }


# --------------------------------------------------------------------------
# 4. TOST 等价检验
# --------------------------------------------------------------------------
def _t_ci(d: np.ndarray, alpha: float) -> tuple[float, float]:
    """配对差的 (1-alpha) t 置信区间。n 很小时用 t 分布临界值（查表 + 插值）。"""
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    tcrit = _t_critical(n - 1, alpha)
    return mean - tcrit * se, mean + tcrit * se


_T_TABLE = {
    # df: t_{0.975}
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}


def _t_critical(df: int, alpha: float) -> float:
    """t 分布双侧临界值。alpha=0.05 查表插值；其他 alpha 用正态近似兜底。"""
    if abs(alpha - 0.05) > 1e-9:
        # 粗略兜底：α=0.10 → z=1.645，α=0.01 → z=2.576
        z = {0.10: 1.645, 0.05: 1.960, 0.01: 2.576}.get(round(alpha, 2), 1.960)
        return z * 1.1  # 小样本保守放大
    if df in _T_TABLE:
        return _T_TABLE[df]
    keys = sorted(_T_TABLE)
    if df < keys[0]:
        return _T_TABLE[keys[0]]
    if df > keys[-1]:
        return 1.960
    lo = max(k for k in keys if k < df)
    hi = min(k for k in keys if k > df)
    w = (df - lo) / (hi - lo)
    return _T_TABLE[lo] * (1 - w) + _T_TABLE[hi] * w


def tost_equivalence(
    a: Sequence[float],
    b: Sequence[float],
    *,
    margin: float,
    alpha: float = 0.05,
    margin_source: str = "",
) -> dict[str, Any]:
    """TOST 等价检验。**只有在预先声明了 margin 之后才允许用本函数主张"无差异"。**

    Args:
        margin: 等价边界（绝对单位，与指标同单位）。必须**事先**定，不能看着结果定。
        margin_source: margin 的依据（文献、领域惯例、或"我们声明的最小有意义差异"）。
            留空会给出一条 warning —— 事后挑 margin 是 p-hacking 的一种。

    Returns:
        dict：两个单侧检验的 p、合并 p、是否可宣称等价、以及 CI。
    """
    if margin <= 0:
        raise ValueError("margin 必须为正")
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("配对样本长度必须相同")
    d = a_arr - b_arr
    n = len(d)
    if n < 2:
        raise ValueError(f"TOST 至少需要 2 对样本，得到 {n}")

    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    if se == 0:
        # 所有差完全相同：若在界内则等价成立
        equiv = abs(mean) < margin
        return {
            "test": "tost_equivalence",
            "n_pairs": n,
            "margin": margin,
            "margin_source": margin_source or "[未声明]",
            "delta_mean": round(mean, 5),
            "p_lower": 0.0 if equiv else 1.0,
            "p_upper": 0.0 if equiv else 1.0,
            "p_tost": 0.0 if equiv else 1.0,
            "equivalent": bool(equiv),
            "note": "配对差方差为 0",
        }

    df = n - 1
    t_lower = (mean - (-margin)) / se   # H0: Δ <= -margin
    t_upper = (mean - margin) / se      # H0: Δ >= +margin

    def _sf(t: float) -> float:
        """t 分布上尾概率，用不完全 beta 的连分数近似。"""
        return _t_sf(t, df)

    p_lower = _sf(t_lower)               # 拒绝 Δ<=-margin 的证据
    p_upper = 1 - _sf(t_upper)           # 拒绝 Δ>=+margin 的证据
    p_tost = max(p_lower, p_upper)
    lo, hi = _t_ci(d, alpha)

    return {
        "test": "tost_equivalence",
        "n_pairs": n,
        "margin": margin,
        "margin_source": margin_source or "[未声明 —— 事后挑 margin 会被审稿人认定 p-hacking]",
        "delta_mean": round(mean, 5),
        "delta_ci_low": round(lo, 5),
        "delta_ci_high": round(hi, 5),
        "p_lower": round(p_lower, 5),
        "p_upper": round(p_upper, 5),
        "p_tost": round(p_tost, 5),
        "alpha": alpha,
        "equivalent": bool(p_tost < alpha),
        "note": (
            f"在 |Δ| < {margin} 的边界下{'可以主张等价' if p_tost < alpha else '**不能**主张等价'}"
            f"（n={n}，功效极低时 TOST 很难通过，须同时报 CI）"
        ),
    }


def _t_sf(t: float, df: int) -> float:
    """t 分布上尾概率 P(T > t)。用正则化不完全 beta 函数。"""
    x = df / (df + t * t)
    ib = _betainc_reg(df / 2.0, 0.5, x)
    return 0.5 * ib if t > 0 else 1 - 0.5 * ib


def _betainc_reg(a: float, b: float, x: float) -> float:
    """正则化不完全 beta 函数 I_x(a,b)，连分数展开（Numerical Recipes 6.4）。"""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _beta_cf(a, b, x) / a
    return 1 - front * _beta_cf(b, a, 1 - x) / b


def _beta_cf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-12) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


# --------------------------------------------------------------------------
# 5. 组合入口：一次比较两个方法
# --------------------------------------------------------------------------
def compare_methods(
    results_a: dict[int, float],
    results_b: dict[int, float],
    *,
    name_a: str = "A",
    name_b: str = "B",
    equivalence_margin: float | None = None,
    margin_source: str = "",
    n_boot: int = 10000,
) -> dict[str, Any]:
    """主分析入口：给定两个方法在**同一种子集合**上的主指标，输出一张完整对比卡。

    Args:
        results_a, results_b: {seed: 主指标值}。**种子集合必须完全一致**，否则报错
            —— 不配对的比较不是配对检验，混用会低估方差。
        equivalence_margin: 若要主张"无差异"，必须给；不给就只报"无显著差异"
            （这是两件不同的事，见 AGENTS.md §0.1）。
    """
    seeds_a, seeds_b = set(results_a), set(results_b)
    if seeds_a != seeds_b:
        raise ValueError(
            f"配对检验要求种子集合一致。A 独有 {sorted(seeds_a - seeds_b)}，"
            f"B 独有 {sorted(seeds_b - seeds_a)}。缺种子必须补齐，不得丢弃配对。"
        )
    seeds = sorted(seeds_a)
    if len(seeds) < 2:
        raise ValueError(f"配对检验至少需要 2 个共同种子，得到 {len(seeds)}")
    a = [results_a[s] for s in seeds]
    b = [results_b[s] for s in seeds]

    out: dict[str, Any] = {
        "comparison": f"{name_a} vs {name_b}",
        "seeds": seeds,
        "n_pairs": len(seeds),
        "mean_a": round(float(np.mean(a)), 5),
        "mean_b": round(float(np.mean(b)), 5),
        "std_a": round(float(np.std(a, ddof=1)), 5),
        "std_b": round(float(np.std(b, ddof=1)), 5),
        "bootstrap": paired_bootstrap(a, b, n_boot=n_boot),
        "wilcoxon": wilcoxon_signed_rank(a, b),
    }
    if equivalence_margin is not None:
        out["tost"] = tost_equivalence(a, b, margin=equivalence_margin, margin_source=margin_source)
    else:
        out["tost"] = None
        out["equivalence_note"] = (
            "未提供 margin，因此**只能**说『未检出显著差异』，"
            "不得说『两者无差异』（AGENTS.md §0.1）"
        )
    return out


def summarize(comparisons: dict[str, float], alpha: float = 0.05) -> dict[str, Any]:
    """对一组比较的 p 值做 Holm 校正，并给出可直接写进论文的措辞提示。"""
    holm = holm_bonferroni(comparisons, alpha=alpha)
    tips = []
    for name, r in holm["results"].items():
        tips.append(
            f"{name}: p={r['p_raw']:.4f}, Holm 阈值={r['holm_threshold']:.4f} → "
            f"{'显著' if r['reject'] else '不显著（不得写成无差异）'}"
        )
    holm["wording_tips"] = tips
    return holm


if __name__ == "__main__":
    # 自检：用可验证的构造案例确认实现正确
    import json

    print("=== 自检 1：完全相同的两组 → Δ=0，p≈1 ===")
    x = [0.50, 0.52, 0.48]
    print(json.dumps(paired_bootstrap(x, x), indent=2, ensure_ascii=False))

    print("\n=== 自检 2：A 一致优于 B → Δ>0，p 小 ===")
    a = [0.60, 0.62, 0.58, 0.61]
    b = [0.50, 0.51, 0.49, 0.50]
    print(json.dumps(paired_bootstrap(a, b), indent=2, ensure_ascii=False))
    print(json.dumps(wilcoxon_signed_rank(a, b), indent=2, ensure_ascii=False))

    print("\n=== 自检 3：Holm 校正 ===")
    print(json.dumps(holm_bonferroni({"X vs Y": 0.01, "X vs Z": 0.04, "X vs W": 0.20}), indent=2, ensure_ascii=False))

    print("\n=== 自检 4：TOST ===")
    print(json.dumps(tost_equivalence(a, b, margin=0.05, margin_source="自检构造"), indent=2, ensure_ascii=False))

    print("\n=== 自检 5：种子集合不一致必须报错 ===")
    try:
        compare_methods({0: 0.5, 1: 0.6}, {0: 0.5, 2: 0.6})
    except ValueError as e:
        print("正确抛出：", e)
