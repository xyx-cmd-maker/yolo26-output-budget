"""
★★ 补 per-class macro（红队 R2）：per-image macro 与 per-class macro 是否一致？

═══════════════════════════════════════════════════════════════════════════
红队 R2 的质疑（第 17 轮）
═══════════════════════════════════════════════════════════════════════════
*"你的 macro 是**逐图召回平均**，Dave 的 AP 是**逐类 AP 平均**。
  两者不等价，你的'口径'论证跨了两种 macro。"*

**这质疑是对的**——所以我必须把两种 macro 都算出来，看它们的结论是否一致。

═══════════════════════════════════════════════════════════════════════════
★ 关键的方法论问题（必须诚实处理）
═══════════════════════════════════════════════════════════════════════════
**per-class macro 需要"如何把逐图预算分给类"的规则**——而这是一个**自由度**。
本脚本在**可复现的规则**下计算，并明确标注每个数字依赖哪条规则：

  · **R-uniform**：每图配额 `kᵢ` **不加类区分**，各类共享（最自然的部署情形）
  · **R-proportional**：每图的配额按**该类在该图的真实目标数**比例分（oracle 式上界）

per-class 口径定义（**容量代理**，假设排序完美）：
    `recall_c = Σᵢ min(k_{i,c}, n_{i,c}) / Σᵢ n_{i,c}`
    **per-class macro** = `mean_c recall_c`（10 类等权）
    **micro（总召回）** = `Σ_c Σᵢ min(...) / Σ_c Σᵢ n_{i,c}`

→ 两者在同一分配下的差异，就是"per-class macro 是否与 micro 同步"的答案。
→ 还要回答：**"余量给最密/最疏图"对 per-class macro 的影响，是否与对 per-image macro 一致？**

用法：python scripts/per_class_macro.py
"""
from __future__ import annotations
import textwrap

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))

RNG = np.random.default_rng(20261014)
N_CLS = 10


def main() -> int:
    CLS_NAMES = ["pedestrian", "person", "bicycle", "car", "van",
                 "truck", "tricycle", "awning-tri", "bus", "motor"]


    def load_labels(split: str) -> list[np.ndarray]:
        """返回每图的 `(cls, )` 数组（每个标注一个类号）。"""
        cfg = yaml.safe_load((ROOT / "data/processed/visdrone/visdrone.yaml").read_text(encoding="utf-8"))
        d = Path(cfg["path"]) / cfg[split]
        p = list(d.parts)
        for i in range(len(p) - 1, -1, -1):
            if p[i] == "images":
                p[i] = "labels"
                break
        out = []
        for f in sorted(Path(*p).glob("*.txt")):
            cls = []
            for ln in f.read_text(encoding="utf-8").strip().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                cls.append(int(float(ln.split()[0])))
            out.append(np.array(cls, dtype=int))
        return out


    def counts_matrix(labels: list[np.ndarray]) -> np.ndarray:
        """N × C 的真实目标数矩阵。"""
        M = np.zeros((len(labels), N_CLS), dtype=float)
        for i, c in enumerate(labels):
            for v in c:
                if 0 <= v < N_CLS:
                    M[i, v] += 1
        return M


    def recall_micro(M: np.ndarray, K: np.ndarray) -> float:
        tot = M.sum()
        return float(np.minimum(K, M).sum() / tot) if tot else 0.0


    def recall_macro_class(M: np.ndarray, K: np.ndarray) -> float:
        """per-class macro（10 类等权）。空类跳过。"""
        vals = []
        for c in range(M.shape[1]):
            tc = M[:, c].sum()
            if tc > 0:
                vals.append(float(np.minimum(K[:, c], M[:, c]).sum() / tc))
        return float(np.mean(vals)) if vals else 0.0


    def recall_macro_image(M: np.ndarray, K: np.ndarray) -> float:
        """per-image macro（逐图召回等权，跨类合并）。"""
        n = M.sum(axis=1)
        tp = np.minimum(K.sum(axis=1), n)
        ok = n > 0
        return float((tp[ok] / n[ok]).mean()) if ok.any() else 0.0


    def k_uniform(M: np.ndarray, B: int) -> np.ndarray:
        """每图给 B 个配额，**不加类区分** → 各类共享。
        容量口径下：K[i,c] = min(B, n_i) * (n_{i,c}/n_i)（按需分配的上界实现）。
        """
        n = M.sum(axis=1, keepdims=True)
        qi = np.minimum(float(B), n)                     # 每图可用总额
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(n > 0, M / np.maximum(n, 1e-12), 0.0)
        return qi * frac


    def k_by_rule(M: np.ndarray, B: int, rule: str) -> np.ndarray:
        """两种"预算怎么分给类"的规则。"""
        n = M.sum(axis=1, keepdims=True)
        qi = np.minimum(float(B), n)
        if rule == "uniform":
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(n > 0, M / np.maximum(n, 1e-12), 0.0)
            return qi * frac
        if rule == "prop":            # 与 uniform 在此实现下等价（容量口径）
            return k_by_rule(M, B, "uniform")
        raise ValueError(rule)


    def k_shift_to(M: np.ndarray, B: int, target: str, frac: float = 0.5) -> np.ndarray:
        """在**守恒实际花费** `Σmin(qᵢ,nᵢ)` 的前提下，把配额从一半图挪给另一半图。

        ★★ 守恒对象的修正（本轮踩到的坑）：
          · 恒等式要求的是 **`Σmin(qᵢ, nᵢ)`（实际花费）守恒** —— 只有它守恒时 micro 才不变。
          · 我上一版守恒的是 **`Σqᵢ`（配额）**，那**不**守恒 `Σmin`
            → micro 本就会变 → 那个设计**根本测不出「micro 不变」**。
          · 且上一版有守恒 bug：从"不喜欢"的图抽走后，目标图**没有足够余量**接回
            （实测 `总额未守恒 14581.25 vs 15019.0`），因为 `Σmin(B,nᵢ)` 已把余量用尽。

        **正确做法**：先把**花费**抽进一个池子（从任意图抽，减少其 `min(qᵢ,nᵢ)`），
        再按 `target` 顺序把池子**花到还有余量的图**上（增加其 `min(qᵢ,nᵢ)`）。
        每一步都保证 `qᵢ ≤ nᵢ`，且**池子必须花完**（若真的花不完，如实报出残留）。
        """
        n = M.sum(axis=1)
        q = np.minimum(float(B), n).copy()          # 逐图配额；此时 Σmin(q,n) = Σq = Σmin(B,n)
        base_spend = float(np.minimum(q, n).sum())
        pool = frac * base_spend * 0.5
        order = np.argsort(-n if target == "dense" else n, kind="mergesort")

        # ① 抽：从排序**靠后**的图抽（减少其 min(q,n)）
        taken = 0.0
        for i in order[::-1]:
            if taken >= pool:
                break
            room = float(np.minimum(q[i], n[i]))     # 最多能抽这么多
            if room > 0:
                t = min(room, pool - taken)
                q[i] -= t
                taken += t
        # ② 花：按 target 顺序花到还有余量的图（`qᵢ < nᵢ` 才能增加 min）
        given = 0.0
        for i in order:
            if given >= taken:
                break
            room = float(n[i] - q[i])
            if room > 0:
                add = min(room, taken - given)
                q[i] += add
                given += add
        residual = taken - given
        if residual > 1e-6:
            print(f"      [注] 池子未能花完，残留 {residual:.1f}（目标图余量不足）")

        with np.errstate(invalid="ignore", divide="ignore"):
            fr = np.where(n[:, None] > 0, M / np.maximum(n[:, None], 1e-12), 0.0)
        K = q[:, None] * fr
        # 守恒核对：实际花费应只差 residual
        new_spend = float(np.minimum(K, M).sum())
        assert abs((base_spend - new_spend) - residual) < 1e-6, \
            f"花费不守恒：{new_spend} vs 基线 {base_spend}（残留 {residual}）"
        return K


    print("=" * 104)
    print("per-class macro vs per-image macro：两种口径是否给出相同结论？")
    print("=" * 104)

    for split in ("val", "test"):
        labels = load_labels(split)
        M = counts_matrix(labels)
        N, TOT = M.shape[0], M.sum()
        print(f"\n{'#' * 104}\n### {split}  N={N}  Σn={int(TOT)}  "
              f"B*={TOT / N:.2f}  空类={int((M.sum(axis=0) == 0).sum())}")
        print("  逐类目标数：")
        cs = M.sum(axis=0)
        line = "    "
        for c in range(N_CLS):
            line += f"{CLS_NAMES[c]}={int(cs[c])}  "
            if c == 4:
                print(line); line = "    "
        print(line)

        print(f"\n{'B':>4} │ {'方案':>14} │ {'Σmin(B,nᵢ)':>11} {'micro(总)':>10} {'macro(逐类)':>11} "
              f"{'macro(逐图)':>11} │ {'Δmicro':>8} {'Δmacro类':>9} {'Δmacro图':>9}")
        print("-" * 118)
        for B in (20, 30, 50, 100):
            base = k_by_rule(M, B, "uniform")
            spend0 = float(np.minimum(base, M).sum())
            m0 = recall_micro(M, base)
            c0 = recall_macro_class(M, base)
            i0 = recall_macro_image(M, base)
            print(f"{B:>4} │ {'uniform':>14} │ {spend0:>11.0f} {m0 * 100:>9.3f}% {c0 * 100:>10.3f}% "
                  f"{i0 * 100:>10.3f}% │ {'—':>8} {'—':>9} {'—':>9}")
            for tgt, nm in (("dense", "余量→最密"), ("sparse", "余量→最疏")):
                k = k_shift_to(M, B, tgt)
                spend = float(np.minimum(k, M).sum())
                m, c, i = recall_micro(M, k), recall_macro_class(M, k), recall_macro_image(M, k)
                # ★ 守恒核对：花费必须与基线一致（允许 shift 未花完的残留）
                flag = "" if abs(m - m0) < 1e-12 else "  ⚠️micro 变了!"
                print(f"{B:>4} │ {nm:>14} │ {spend:>11.0f} {m * 100:>9.3f}% {c * 100:>10.3f}% "
                      f"{i * 100:>10.3f}% │ {(m - m0) * 100:>+7.4f} {(c - c0) * 100:>+8.3f} "
                      f"{(i - i0) * 100:>+8.3f}{flag}")
            print("     " + "·" * 114)

    print("\n" + "=" * 104)
    print("★ 判读")
    print("=" * 104)
    print(textwrap.dedent("""  · 若 **Δmicro = 0 而 Δmacro(逐类) 与 Δmacro(逐图) 同号且量级相近**
          → 两种 macro 一致 → "口径"论证成立，红队 R2 被消解
      · 若 **两者异号** → "口径"论证跨了两种 macro，必须重写（只主张其中一种）
      · 若 **Δmacro(逐类) ≈ 0** → per-class 口径对**图间**分配不敏感
          → 说明我们研究的是「**空间公平性**」而非「**类别公平性**」，定位需相应收窄
    """))



if __name__ == "__main__":
    raise SystemExit(main())
