"""
★★ 补齐第三个口径：`worst`（最坏图召回）—— 修掉第 17 轮的伪影

═══════════════════════════════════════════════════════════════════════════
第 17 轮的伪影（必须修正的原因）
═══════════════════════════════════════════════════════════════════════════
`alloc_permuted` 从"配额充足"的图里抽走额度时，**可能把某张图的配额抽到 0**
→ 该图 `recall = 0` → `worst` 必然为 0 → 所有变体的 `worst` 都显示 −9.5 ~ −31.5。
**那不是发现，是构造方式的产物。**

**修正**：加 **`qᵢ ≥ q_floor`** 约束（本脚本取 `q_floor = 1`，即"每张图至少给 1 个检测"）。
这在部署上是**自然的**（没有哪个系统会对某张图完全不给输出）。

═══════════════════════════════════════════════════════════════════════════
本脚本做两件事
═══════════════════════════════════════════════════════════════════════════
  1. **四个口径同时报告**：micro / macro(逐图) / macro(逐类) / **worst(最坏图)**
  2. 在 **`Σmin` 守恒 + `qᵢ ≥ 1`** 双约束下重做"余量给最密/最疏"的对照
     → 回答："把额度从稀疏图挪给密集图，最坏图召回会怎样？"

用法：python scripts/worst_metric_fixed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"D:\Deepseek Harness\yolo\yolo26-detection")
sys.path.insert(0, str(ROOT))

N_CLS = 10


def main() -> int:
    Q_FLOOR = 1.0     # ★ 每图配额下限（修伪影的关键）


    def load_labels(split: str) -> list[np.ndarray]:
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
                if ln:
                    cls.append(int(float(ln.split()[0])))
            out.append(np.array(cls, dtype=int))
        return out


    def counts_matrix(labels) -> np.ndarray:
        M = np.zeros((len(labels), N_CLS), dtype=float)
        for i, c in enumerate(labels):
            for v in c:
                if 0 <= v < N_CLS:
                    M[i, v] += 1
        return M


    def m_micro(M, K):
        t = M.sum()
        return float(np.minimum(K, M).sum() / t) if t else 0.0


    def m_macro_class(M, K):
        v = [float(np.minimum(K[:, c], M[:, c]).sum() / M[:, c].sum())
             for c in range(M.shape[1]) if M[:, c].sum() > 0]
        return float(np.mean(v)) if v else 0.0


    def m_macro_image(M, K):
        n = M.sum(axis=1)
        tp = np.minimum(K.sum(axis=1), n)
        ok = n > 0
        return float((tp[ok] / n[ok]).mean()) if ok.any() else 0.0


    def m_worst(M, K):
        n = M.sum(axis=1)
        ok = n > 0
        if not ok.any():
            return 0.0
        return float((np.minimum(K.sum(axis=1), n)[ok] / n[ok]).min())


    def build_K(M, q):
        """给定逐图配额 q，按图内真实需求比例分到各类（容量口径）。"""
        n = M.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            fr = np.where(n > 0, M / np.maximum(n, 1e-12), 0.0)
        return q[:, None] * fr


    def base_q(M, B):
        return np.minimum(float(B), M.sum(axis=1)).copy()


    def shift(M, q0, target, pool_frac=0.5):
        """在 **Σmin 守恒 + qᵢ ≥ Q_FLOOR** 下重分配。

        ① 抽：从"不喜欢"的一端抽，但**每图的 min(qᵢ,nᵢ) 不得低于 Q_FLOOR**
        ② 花：按 target 顺序花到还有余量的图（`qᵢ < nᵢ`）
        """
        n = M.sum(axis=1)
        q = q0.copy()
        spend0 = float(np.minimum(q, n).sum())
        pool = pool_frac * spend0 * 0.5
        order = np.argsort(-n if target == "dense" else n, kind="mergesort")

        taken = 0.0
        for i in order[::-1]:
            if taken >= pool:
                break
            cur = float(np.minimum(q[i], n[i]))
            # ★ 保留下限：最多抽到 Q_FLOOR 为止
            room = cur - Q_FLOOR
            if room > 0:
                t = min(room, pool - taken)
                q[i] -= t
                taken += t

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
        new = float(np.minimum(build_K(M, q), M).sum())
        assert abs((spend0 - new) - residual) < 1e-6, f"花费不守恒 {new} vs {spend0}"
        assert (q >= Q_FLOOR - 1e-9).all(), f"违反配额下限：min q = {q.min()}"
        return q, residual


    print("=" * 112)
    print(f"四口径对照（含 worst），配额下限 qᵢ ≥ {Q_FLOOR:g}（修第 17 轮伪影）")
    print("=" * 112)

    for split in ("val", "test"):
        M = counts_matrix(load_labels(split))
        N, TOT = M.shape[0], M.sum()
        print(f"\n{'#' * 112}\n### {split}  N={N}  Σn={int(TOT)}  B*={TOT / N:.2f}")
        print(f"{'B':>4} │ {'方案':>12} │ {'Σmin':>8} {'残余':>5} │ {'micro':>8} {'macro类':>8} "
              f"{'macro图':>8} {'worst':>8} │ {'Δmicro':>8} {'Δmac类':>8} {'Δmac图':>8} {'Δworst':>8}")
        print("-" * 112)
        for B in (20, 30, 50, 100):
            q0 = base_q(M, B)
            K0 = build_K(M, q0)
            s0 = float(np.minimum(K0, M).sum())
            v0 = (m_micro(M, K0), m_macro_class(M, K0), m_macro_image(M, K0), m_worst(M, K0))
            print(f"{B:>4} │ {'uniform':>12} │ {s0:>8.0f} {'—':>5} │ "
                  f"{v0[0] * 100:>7.3f}% {v0[1] * 100:>7.3f}% {v0[2] * 100:>7.3f}% {v0[3] * 100:>7.3f}% │ "
                  f"{'—':>8} {'—':>8} {'—':>8} {'—':>8}")
            for tgt, nm in (("dense", "→最密"), ("sparse", "→最疏")):
                q, res = shift(M, q0, tgt)
                K = build_K(M, q)
                s = float(np.minimum(K, M).sum())
                v = (m_micro(M, K), m_macro_class(M, K), m_macro_image(M, K), m_worst(M, K))
                print(f"{B:>4} │ {nm:>12} │ {s:>8.0f} {res:>5.0f} │ "
                      f"{v[0] * 100:>7.3f}% {v[1] * 100:>7.3f}% {v[2] * 100:>7.3f}% {v[3] * 100:>7.3f}% │ "
                      f"{(v[0] - v0[0]) * 100:>+7.4f} {(v[1] - v0[1]) * 100:>+7.3f} "
                      f"{(v[2] - v0[2]) * 100:>+7.3f} {(v[3] - v0[3]) * 100:>+7.3f}")
            print("     " + "·" * 108)

    print("\n" + "=" * 112)
    print("★ 判读")
    print("=" * 112)
    print(f"""  · 若 `Δworst` **不再恒为负且量大** → 第 17 轮的伪影已修，worst 列可用
      · 关键比较：**四个口径对同一操作的敏感度排序**
          micro ≈ 0 ｜ macro(逐类) 最小 ｜ macro(逐图) 与 worst 最大？
        这个排序就是"预算分配到底影响哪种公平性"的答案。
      · 配额下限 qᵢ ≥ {Q_FLOOR:g} 的代价：可从"不喜欢"的图抽走的量变少
        → 若某档 `残余` 不为 0，说明**该约束确实咬合**，需如实报告。
    """)



if __name__ == "__main__":
    raise SystemExit(main())
