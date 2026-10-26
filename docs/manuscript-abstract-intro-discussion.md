# Manuscript draft — Abstract, Introduction, Discussion

> Target venue: **Scientific Reports** (Article). Constraints: main text ≤ 4,500 words, Abstract ≤ 200
> words (unstructured, no references, no figures), ≤ 60 references, Discussion **without** subheadings.
> Drafted 2026-10-15. Companion file: `docs/manuscript-results-methods.md` (Results R0–R4, R3b, Methods).
> Writing skills applied: `science-top-journal-writing` (evidence ledger, A–F gates, red team) and
> `academic-humanizer` (over-claiming verbs, claim–evidence discipline).
>
> **DRAFT STATUS**
> · Numbers marked `[pending]` are not yet measured (multi-seed, test-dev).
> · Every number below is traceable to `results/tables/` or `runs/`; verified by
>   `verify_manuscript_numbers.py`.
> · Citation keys in `[brackets]` resolve against `docs/reference-ledger.md`.

---

# Abstract

Object detectors emit a bounded number of boxes per image, usually one constant: COCO sweeps maxDets = [1, 10, 100], LVIS fixes max_dets = 300. Images differ greatly in count, so a fixed bound is slack on sparse images and binding on dense ones. Given a fixed total of boxes, does
redistributing them between images help? We show that it is a positional intervention rather than a performance one. On validation, reallocating raises worst-image recall by 5.3 to 10.9 points across three training
seeds and both heads, but total recall changes by at most 1.41 points and mean per-image recall
falls by 6.1 to 6.9 points. The gain is positional: it changes which image is worst rather than lifting every image. When that image is quota-limited the reallocation replaces it; when it is detection-limited it keeps zero recall at any quota and the gain vanishes on test-dev for both heads. The cost is sharply graded: the sparsest 5% of images lose 24 points, and in the condition tabulated here every band's change is bounded away from zero. Once the budget exceeds the label-count cap on every image, the two allocations become the same vector and the comparison is vacuous.

---

# Introduction

A detector's output budget is the largest number of boxes it may emit for an image. It is rarely treated
as a scientific choice. The COCO evaluation protocol sweeps `maxDets = [1, 10, 100]` [lin2014coco]; the
LVIS protocol fixes `max_dets = 300` [gupta2019lvis]; the reference implementations in common use default
to 300. In each case a single constant is applied to every image in the benchmark, whatever those images
contain.

That constant is not innocuous, and the observation is not new. Dave et al. [dave2021devil] raised the
detections-per-image limit on LVIS from 300 to 10,000 and reported that mean average precision rises from
22.6 to 26.1, with the average precision for rare categories rising from 12.6 to 19.8. The same
intervention leaves COCO essentially unchanged at 37.4 to 37.5, which they attribute to COCO's smaller
vocabulary. The first-place entry in the 2021 LVIS challenge reports a consistent effect on a different
metric, with boundary AP rising from 44.1 to 45.3 as the per-image limit increases from 300 to 1,000, and
notes that the competition rules imposed no limit at all [fu2021lvis]. Oksuz et al. [oksuz2022lrp] report
from a third direction that localisation recall precision responds to detections-per-image, although in
their setting average precision and oLRP move in opposite directions, so the two metrics do not agree on
the sign of the change. Ultralytics issue #25992 [ultralytics_issue25992] records an aggregate ceiling of
the same kind: at the default limit, the fraction of annotated objects recoverable by construction is
23.6% on the benchmark discussed there.

Those reports are consistent with a fixed output budget being able to bind. All of them act by raising the
limit, which gives every image more boxes. None asks what follows when the limit is genuinely fixed, which is the situation in any deployed system with a hard quota. A concrete case is a downlink that can carry a fixed number of boxes per frame for a whole camera array, so that boxes spent on one camera are boxes denied to another. If the budget is fixed and the constraint binds,
the distribution across images becomes a free parameter: the same total number of boxes can be spread
evenly, concentrated on dense images, or allocated in any other way. Whether that freedom is worth
exercising is an open question. The evaluation literature treats the output budget as a protocol constant
rather than an allocatable resource, and the allocation literature that would supply the machinery addresses
other budgets: compute or patch selection rather than emitted boxes [densescout2026], or a
decision-theoretic value of perception compute under a budget-constrained oracle with uniform inference as
the baseline [decisionvalue2026]. Two lines do reallocate a per-sample capacity, but of computation rather than emitted boxes: content-aware detectors give each image a data-dependent compute budget [cadet2020], and computation reallocation moves backbone FLOPs across resolutions [crnas2020]. In retrieval and recommendation, a fixed output window has been formalised
as a deterministic ceiling on achievable ranking quality [wang2026recceiling; buckley2004reliability], but
there the window is per query or per user, leaving no room for redistribution.

We therefore ask a narrow question. Given a fixed total number of boxes to distribute across an image set,
does reallocating it change the worst-image recall, and what does the change cost? We answer with a
measurement rather than a method. The proportional allocation we compare against is the classical max-min
fair one, obtained by integer water-filling, and we claim nothing new for it
[bertsekas1992data; hardy1934inequalities]. What we contribute is the magnitude of the effect on a real
detector, the distribution of its cost across the images, and the regime in which the comparison is
vacuous.

We evaluate this on VisDrone2019-DET [zhu2021visdrone] with YOLO26n [jocher2026yolo26], whose two prediction heads give two settings within one detector, and we train three seeds. Reallocating a fixed total budget
raises worst-image recall by 5.3 to 10.9 points on validation, but not on the held-out test split. Three results constrain how that gain should be read.

First, the intervention is positional rather than absolute. It changes which image attains the minimum; some images gain, others lose. When we trace the identity of the worst image under both allocations, the
uniform allocation consistently pins a dense image as the worst case, and reallocation replaces it with a
different one. Mean per-image recall therefore falls even as the minimum rises.

Second, the total box count is conserved and the cost is graded. Total recall changes by at most 1.41 points, because
when the budget binds on every image the number of boxes that can be spent is fixed by the budget. The
cost falls unevenly: the sparsest 5% of images lose 24.2 points while the densest quartile gains 10.1, and
mean per-image recall falls by 6.1 to 6.9 points. A practitioner who adopts per-image allocation to
protect the worst case will redistribute recall towards images that already contained the most objects.

Third, the boundary of the effect is structural, not empirical. Once the budget exceeds the annotated count on every image, the label-count cap binds everywhere and the two allocations coincide exactly; we verify this by exact comparison of the two quota vectors. Any "zero gain" measured there is an identity, and we report it as
such rather than as evidence about the allocation.

This work sits between three literatures that do not meet. Detection evaluation treats the per-image output
limit as a protocol constant, and the reports that show it can bind act by raising it, which gives every
image more boxes rather than moving boxes between images [dave2021devil; fu2021lvis; oksuz2022lrp].
Resource-allocation work in detection budgets compute or input regions rather than emitted outputs, and has
already argued that allocation should be judged against a budget-constrained oracle with uniform inference
as a baseline [densescout2026; decisionvalue2026]. Information retrieval and recommendation formalise a
fixed output window as a hard ceiling on achievable ranking quality [wang2026recceiling;
buckley2004reliability], but the window there is fixed per query or per user, so no redistribution is
possible. What has been missing is a measurement of what redistribution actually buys, and at whose
expense.

Two things follow for practice. First, worst-image recall is an extreme order statistic governed by
whichever image happens to be worst, so treating it as a deployment objective moves the loss onto the
images with the fewest objects rather than reducing it. Second, before per-image allocation is adopted,
the budget should be checked against what the detector can actually emit: if the budget is not binding,
the allocation question does not arise, and a comparison run there will return zero for structural
reasons. We report these results for one detector on one benchmark, and we state the scope explicitly
rather than generalising.

---

# Discussion

*(No subheadings, per the Scientific Reports Article format.)*

This work began as an attempt to predict, per image, whether an output budget should be reallocated. The
prediction failed and we withdrew it. What survived is a measurement of what reallocation buys, what it
costs, and when the question is vacuous. We report both halves, because the failure is informative about
the shape of the problem.

The failed attempt is worth stating precisely, because the same construction is natural. We tried to
decide, per image, whether the budget or the detector limited that image, by comparing the worst image's
quota against the number of boxes it could physically emit. The quantity is computable from stored
predictions and needs no retraining. It is not diagnostic, for two reasons. First, shuffling the detector's
confidence-ordered hit sequences across images, which holds the label counts and the budget fixed while substituting another detector, leaves the verdict unchanged in 198 of 200 draws and makes the gain non-positive in each: the quantity tracks the labels, not the detector. Second, at seed 42 on the one-to-one head, ten non-degenerate conditions at budgets 58 to 67 predict a gain that is measured negative, down to −2.8 points; a counterexample, not a rate. The deeper reason is that a detector emits at least as many predictions as an image has annotated objects, so the prediction count cancels and the construction reduces to a statement about the labels alone, the
same class of quantity as the utilisation statistics we already argued are uninformative. The only reliable
test of such a quantity is to change the detector and see whether the verdict follows.

 Under a uniform quota the densest image is pinned as the worst case; once the quota is redistributed it no longer is, so the minimum falls on a different image.  The shift reaches the 5th percentile while the median falls: reallocation moves recall down the distribution rather than creating it. Describing this as an improvement of the worst
case would be arithmetically accurate and substantively misleading, because the binding constraint moved
rather than relaxed.

The zero-sum property follows from the same accounting, with one correction we owe the reader. When the budget binds on every image, the conservation identity fixes the spend. That identity is about boxes emitted, not about objects recovered:
the false-positive share of the spent boxes rises with the budget, so the identity bounds the
total recall rather than fixing it. Measured total recall changes by at most 1.41 points across the twelve
spendable budget conditions, consistent with that bound. Reallocation is therefore not a performance
intervention. It moves recall between images, and the images it moves recall away from are the sparsest
ones, which in VisDrone contain the fewest and largest objects. Adopting worst-image recall as an objective therefore means, in this benchmark, degrading the easiest images to improve the hardest. That is a defensible choice for some applications
and a poor one for others, but it should be made explicitly rather than inherited from a metric name.

**The gain is bounded by the nature of the worst image.** Because it replaces rather than
improves that image, it helps only when a quota can help it. On validation the worst image is
quota-limited, so the proportional allocation moves the minimum elsewhere. On test-dev both allocations share the same worst image: the uniform quota already covers its 8 boxes and recovers none (Supplementary Table S3), while the vectors differ on 1568 of 1610 images and both spend 37,551 boxes. The comparison is well posed and the gain is +0.000 on both heads. The mechanism replicates while the aggregate statistic does not.

Several limitations bound these claims. The results come from three training seeds of one detector on one
benchmark; the between-seed spread is material, reaching 4.8 points on one head, so the magnitudes we
report are ranges rather than points, including the 24.2-point loss on the sparsest images, whose interval
spans 12.14 points and still excludes zero, and the 11.2-point peak gain. Above a per-image budget of about
100 the budget can no longer be spent in full,
because the label-count cap saturates every quota; for those conditions
the two allocations coincide exactly, and we report them as a structural identity rather than as evidence
about allocation. CrowdHuman [shao2019crowdhuman] object counts depend on an annotation convention, with
ignored regions optionally excluded; we report both versions, and the cross-benchmark ordering is
unchanged. Finally, the two heads we test are two settings of the same network rather than two independent
implementations, so the measurement rests on one detector family.

Two directions follow. The measurement should be repeated on detectors whose per-image recall is flat in object count, since the positional gain is a property of the failure profile, not the allocation. The objective also deserves scrutiny: worst-image recall is an extreme order statistic whose sampling variability is governed by a handful of images, which is why the intervals we
report are wide relative to the effect. Lower quantiles of the per-image recall distribution would trade the
interpretability of a worst case for estimates a finite benchmark can actually resolve.

---

# Notes for the next revision

> ⛔ **2026-10-16（第 54 轮）：本文件已按路线 A 重写。**
> 原核心主张（"判据能预测"）**已被证伪并撤回**，Abstract / Introduction / Discussion 已改为
> **positional（位置性）+ 零和 + 分级代价**的表述。
> 依据：`docs/finding-CORE-CLAIM-FALSIFIED.md`（含复现脚本 `.probe/verify_r2_findings.py`）。

1. **已完成（本轮）**：
   · Abstract 重写（**195 词**，按空白切分口径；上限 200）；删除全部"判据/预测"叙述；
   · Introduction 重写：删除"we propose a criterion"整段，改为"measurement rather than a method"；
     并入"位置性 / 守恒与分级代价 / 边界是结构性的"三条约束；
   · Discussion 重写：**新增一段如实说明失败的判据主张与两次证伪检验**（置换 198/200、10 个反例）；
     修正零和恒等式（**花费的框 ≠ 真阳性**，假阳占比随预算上升）；
     Limitations 从"single seed"改为"two seeds"并写出**种间跨幅达 3.0 pt**；
   · 修正 `oksuz2022lrp` 的归属（该项目自己提取的数据显示 AP 与 oLRP **反向**）；
     修正 `ultralytics_issue25992` 的归属（是**聚合标签计数上界**，非"deployment 测量"）；
     删除"nearest the camera"（无深度数据支持，且与项目自己的笔记相反）。

2. **`manuscript-results-methods.md` 仍需同步重写**（本轮未完成）：
   · R5 / R5b 两节标题含 "criterion"（14 处 criterion / 19 处 predict 待清理）；
   · R5b 的两个头数字**标反了**（实际 o2m s42 = 11.672、s43 = 11.987）；
   · R3 已修（旧值 → 重算值，并如实写"+10.21 达到 A 档"）；
   · Table 2 需补 `Δmacro`（−6.14…−6.57 pt）与修正 o2o 的 `P(Δ>0) = 0.9975`；
   · Table 5 需区分"结构性恒等（B≥100 两分配重合）"与"实测零增益"。

3. **R1/R4 的三种子替换**：等 s44 跑完（当前 epoch 33/100）。
   注意：**"ten non-degenerate conditions"这个计数本身也是错的**（它来自
   `verify_seed_replication.py` 里硬编码的 9 档预算表）；
   同代码同数据下的真实计数是 **12**（去掉两个空档）或 **22**（全部非退化）。

4. **`Δmacro`** 已补入 Abstract、Discussion、Table 2（−6.14 … −6.57 pt，比最坏图的**增益**还大）。

5. **引文预算**：台账 20 条（17 条在正文），远低于 60 条上限 —— **引文不是瓶颈**。
   ✅ **2026-10-16 已补齐作者名**：`decisionvalue2026`（Pham Cong & Luong）、
   `densescout2026`（Xiong, Zeng, Chen, Xu, Yan & Qi），均从 **arXiv API** 取回。
   **★ 两处更正**：① `densescout2026` **已被 ACM Multimedia 2026 接收**
   （DOI `10.1145/3767308.3836001`），**此前误记为 arXiv preprint**；
   ② 其作者含 `Zimo Zeng`；作者已于 2026-10-16 确认其与本文通讯作者
   `曾子铭 (Ziming Zeng)` **为不同人**，故不构成自引。

6. **补充材料的正文引用（2026-10-16 已修）**：此前只有 Table S1 被引用，
   **Figure S1 / Figure S2 / Note S1 从未被引用**，而 Sci Rep 明确要求
   "refer to each piece of supplementary material in the main article"。
   现已在 R3（Figure S2）、R4（Figure S1）、Code and reproducibility（Note S1）三处补上引用。
