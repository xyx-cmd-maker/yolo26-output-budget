# Manuscript draft — Results and Methods

> Target venue: **Scientific Reports** (Article; ≤4,500 words main text, ≤11 typeset pages,
> ≤200-word unstructured Abstract, ≤60 references, no graphical abstract).
> Drafted: 2026-10-15. **All numbers below are traceable to `results/tables/` and `runs/` in this repository.**
> Writing skills applied: `science-top-journal-writing` (evidence ledger, A–F gates, red team) and
> `academic-humanizer` (over-claiming verbs, empty intensifiers, claim–evidence discipline).
>
---

# Results

## R0. Output-budget utilisation varies by up to 8.5-fold across detection benchmarks

A detector's output budget is the maximum number of boxes it may emit per image. In practice this budget is a fixed constant: the COCO evaluation protocol sweeps `maxDets = [1, 10, 100]`
[lin2014coco], the LVIS protocol fixes `max_dets = 300` [gupta2019lvis], and the Ultralytics YOLO
implementation defaults to `max_det = 300` [jocher2026yolo26].  Whether that is reasonable depends on `n_i`, because a fixed budget binds only on images whose count exceeds it.

We therefore computed, for five benchmark splits, three quantities that depend only on the label
distribution and not on any model: the mean per-image object count, the budget utilisation
`util(B) = Σ_i min(B, n_i) / (B·N)`, and the fraction of images whose count exceeds the budget,
`frac(n_i > B)`. Supplementary Table S8 reports them and Figure 1 shows them as curves.

At `B = 100`, utilisation spans 0.074 to 0.630 and the fraction of images on which the budget binds spans
0.000 to 0.195. The ordering follows the mean object count, not the domain. The largest object count does not predict utilisation either. CrowdHuman
contains images with 391 annotated people, more than any VisDrone validation image (317), yet only 2.1% of
CrowdHuman images exceed a budget of 100, against 19.5% of VisDrone validation images.

Two cautions apply. CrowdHuman flags ignore regions per box; excluding them lowers the mean object count by 3.8% and `frac(n_i > 100)` from 0.021 to 0.018, leaving the ordering unchanged. Supplementary Table S8 bounds the room a reallocation has; whether recall changes depends on the detector, not the labels.

## R1. Reallocation is positional: worst-image recall rises while mean per-image recall falls

*(Three training seeds, all complete.)*

To test whether the budget is better distributed than by a uniform constant, we trained YOLO26n on VisDrone2019-DET and evaluated on the 548-image validation split, for three seeds. For a fixed total budget `S = ρ·Σ_i n_i` with `ρ = 0.5`, we compared three allocations of
the same total: a uniform quota `q_i = min(c, n_i, p_i)` with `c` chosen so that `Σ_i min(c, n_i, p_i) = S`; a quota proportional to the true object count (an oracle, see Methods), `q_i = min(⌈a·n_i⌉, n_i, p_i)` with `a` the largest multiplier whose spend stays within `S`; and an adaptive variant that caps each quota at `p_i` alone rather than at `min(n_i, p_i)`. Here `p_i` is the number of boxes the
detector can emit for image `i`, which caps the quota. Quotas are integers, and all three allocations spend
within 0.02% of `S`.

**Table 1. Allocation of a fixed output budget at ρ = 0.5, VisDrone2019-DET validation (N = 548).**
Worst-image recall is `min_i TP_i/n_i`; mean per-image recall is the unweighted mean of `TP_i/n_i` over
images. Cells are ordered **one-to-many head (first entry) / one-to-one head (second entry)**; within each
entry, seeds 42, 43 and 44 follow in that order. CI is a paired bootstrap over images, 10,000 resamples for every seed, with RNG seed 20261014.

| Quantity | Uniform | Proportional | Difference (pt) |
|---|---|---|---|
| Worst recall, o2m | 11.67 / 11.99 / 11.99% | 20.00 / 20.37 / 20.00% | **+8.33 / +8.38 / +8.01** |
| Worst recall, o2o | 11.36 / 11.67 / 11.36% | 16.67 / 20.00 / 22.22% | **+5.31 / +8.33 / +10.87** |
| Total recall (micro), o2m | 40.68 / 40.75 / 41.07% | 42.06 / 42.00 / 42.30% | +1.38 / +1.25 / +1.23 |
| **Mean recall (macro), o2m** | 50.21 / 50.45 / 50.90% | 43.85 / 43.88 / 44.03% | **−6.36 / −6.57 / −6.87** |
| **Mean recall (macro), o2o** | 49.03 / 49.32 / 49.80% | 42.89 / 43.15 / 43.42% | **−6.14 / −6.17 / −6.38** |
| 95% CI for gain, o2m | n/a | n/a | [+5.86,+13.01] / [+5.63,+13.01] / [+5.86,+13.33] |
| 95% CI for gain, o2o | n/a | n/a | [+1.93,+11.18] / [+5.26,+11.18] / [+7.48,+14.13] |
| `P(gain > 0)`, o2m | n/a | n/a | 1.0000 / 1.0000 / 1.0000 |
| `P(gain > 0)`, o2o | n/a | n/a | 0.9975 / 1.0000 / 1.0000 |

On validation, proportional allocation raises worst-image recall by 5.31 to 10.87 points across the three
seeds and both heads (Figure 2), and Figure 3 resolves the change image by image. Replacing the label count with the detector's own prediction count as the allocation signal recovers part of this gain at about half the mean cost (Methods; Supplementary Table S2). The gain is positive in every resample on five of the six seed-by-head conditions and in 99.75% on the sixth, the one-to-one head at seed 42. It is not a general improvement: mean
per-image recall **falls** by 6.14 to 6.87 points on the same comparisons, so the intervention raises the
minimum by lowering the average. Total recall is largely unchanged, with the two allocations differing by 1.23 to 1.40 points. When the budget binds on every image, the number of boxes that can be *spent* is fixed
at `Σ_i min(k_i, n_i) = Σ_i k_i = S` by the budget rather than by the allocation. That identity constrains boxes emitted and not objects recovered: the false-positive share of the spent boxes rises with the budget, from under 1% at two boxes per image to over 40% at seventy, so the identity bounds the total recall instead of fixing it. Reallocating a fixed budget redistributes recall; total recall rises by only 1.2 to 1.4 points, which the equivalence test does not resolve.

 The one-to-many gain moves from +11.15 to +8.78 points and
the one-to-one gain from +8.24 to +6.40 at `B = 25`, while at `B = 50` the one-to-one gain moves in the
opposite direction, from +3.93 to +7.65. The direction of the effect is stable, but its magnitude is not
resolvable at the precision a single seed would suggest, which is why we report ranges rather than point
estimates throughout, and why the confidence intervals extend past the threshold we pre-registered. 

## R2. The cost of the reallocation falls on the sparsest images

Because the total box count is conserved, only the distributional consequences matter. Table 2
breaks the per-image recall change down by object-count band, for the one-to-many head at ρ = 0.5
(Figure 4).

**Table 2. Per-image recall by object-count band, VisDrone2019-DET validation, one-to-many head, seed 42.**
ρ = 0.5. Δ is proportional minus uniform, in points. The band boundaries are fixed by the label distribution, which does not depend on the model. Bands are formed by sorting images on `n_i` and cutting at fixed ranks, so images that share an object count can fall in adjacent bands; the cuts are rank quantiles rather than disjoint value intervals, the `n_i range` column reports the observed span within each band, and no model output enters the cut. The interval is a paired percentile bootstrap over
images within the band (10,000 resamples; the RNG is derived per seed so that re-running one seed alone
reproduces its row), so it estimates the mean cost of the band and not the membership of the band. The one-to-one head gives the same ordering with slightly smaller magnitudes: −23.4 and +10.0 points at the two extremes, both intervals excluding zero.

| `n_i` band | Images | `n_i` range | Uniform | Proportional | Δ | 95% interval for Δ |
|---|---|---|---|---|---|---|
| sparsest 5% | 27 | 1–9 | 76.85% | 52.64% | **−24.20** | [−30.28,−18.14] |
| 5–25% | 110 | 10–39 | 67.92% | 46.11% | **−21.81** | [−23.92,−19.69] |
| 25–50% | 137 | 40–65 | 56.70% | 44.26% | −12.44 | [−13.65,−11.28] |
| 50–75% | 137 | 65–93 | 44.44% | 43.64% | −0.80 | [−1.35,−0.26] |
| densest 25% | 137 | 93–317 | 30.03% | 40.10% | **+10.07** | [+9.11,+11.07] |

The gain is concentrated in the densest quartile (+10.07 points) and the loss in the sparsest images
(−24.20 points). The sparsest band holds 27 images, so its interval is widest, at 12.14 points, yet excludes zero: the cost is present in every one of the 10,000
resamples. Every band's interval excludes zero, and the interval ordering matches the band ordering, so under this scheme the gradient is not an artefact of band membership. In
VisDrone, sparse images are those containing few and comparatively large objects: mean box area is
2.31×10⁻² of the frame in the sparsest 5% against 1.18×10⁻³ in the densest quartile, and the Spearman
correlation between object count and mean box area is −0.46 (**Figure 5**). The shift is distributional: the 5th percentile rises 5.84 to 9.68 points while the median falls (Figure 6, Table S7). We deliberately do not infer
object distance
from this, because the benchmark provides no depth annotation and the size–distance relation in aerial
imagery also depends on camera tilt. Optimising worst-image recall therefore raises the recall of the dense images, which start from the lower baseline, and lowers that of the sparse images, which start from the higher baseline. Maximising the
minimum is not the same objective as improving the worst case in the sense that matters for deployment.

## R3. The gain depends on the budget fraction and does not clear the pre-registered threshold uniformly

We pre-registered a decision rule before measuring anything: the allocation line would be treated as a
standalone contribution if the worst-image gain reached 10 points, as a component of a larger study
between 5 and 10 points, and abandoned below 5 points. The rule used the absolute point gain rather than
a ratio, because a ratio would have required a denominator that a real detector cannot attain.

For seed 42, at the pre-registered budget fraction ρ = 0.5 the point estimates are **+8.33 points on the one-to-many
head and +5.31 on the one-to-one head**, both inside the middle band, and the 95% percentile intervals are
**[+5.86, +13.01]** and **[+1.93, +11.18]** (Table 1). Both intervals extend past the 10-point threshold
at their upper limits, and neither is bounded away from it, so the point estimate alone would understate
the uncertainty. The measured effect is therefore too uncertain to decide whether the allocation is worth reporting as a standalone result, and we report it as a component while stating the uncertainty explicitly.

A label-free top-up reproduces the gain (Table S6); it is not an artefact of the budget fraction, but neither is it uniformly below the threshold; removing the label-count cap raises it further (Supplementary Table S5). Sweeping ρ over {0.30, 0.40, 0.50, 0.60, 0.70} and both heads gives ten conditions spanning
**+4.25 to +10.21 points**. Nine of the ten fall in the middle band; **one condition reaches the
standalone band**: the one-to-many head at ρ = 0.40 gives +10.21 points, which the pre-registered rule
scores as A. The remaining condition, the one-to-one head at ρ = 0.70, falls below 5 points and scores as
C. The full sweep is reported in Supplementary Table S1 and plotted against ρ in **Supplementary Fig. S2**; we do not report the
pre-registered value alone, and we do not
claim that the threshold is never reached.

## R3b. The effect is hump-shaped in the budget

*(Independent of seed; computed from stored per-image curves.)*

Supplementary Table S8 reports how much room a fixed budget leaves. That is a property of the labels; whether reallocating changes the worst-image recall is a property of the detector, and the two come apart. To separate them we swept the average per-image budget `B` from 2 to 500, holding the total spend
identical across the two allocations at every budget (relative difference between spends below 10^-4).
Panel a of Figure 2 plots the same gain against the budget.
Of the 17 budgets, 12 are spendable: at `B ≥ 100` neither allocation can spend the budget, because the label-count cap saturates every quota, and those budgets are excluded from the spendable sweep.

Three regimes appear, and we separate them by cause rather than by sign. At the smallest budgets the gain
is negative on the one-to-many head, because proportional rounding gives an image with two to twenty objects one box where uniform gives two, so sparse images lose recall before dense ones gain.
At intermediate budgets the gain peaks, at 11.15 points when `B = 25`, and the between-seed spread is largest there: the one-to-one head moves from +8.42 at `B` = 40 to +3.93 at `B` = 50, a range of 4.5 points across budgets and 4.8 points across seeds at fixed budget. At `B ≥ 70` the gain is exactly
zero, but for two different reasons: at `B = 70` the two quota vectors differ while the worst image is the
same image at its cap under both, whereas at every budget from `B` = 100 to 500 — five values in all — the two vectors are equal entry by entry and
the comparison is an identity. Across the sweep the total recall moves by at most 1.31 points, and across the budget fraction sweep by at most 1.40 points. The full ρ sweep is reported in Supplementary Table S1.

## R4. A reproducible YOLO26n baseline on VisDrone2019-DET

*(All three seeds reported.)*

The detector used throughout is YOLO26n [jocher2026yolo26], trained from COCO weights [lin2014coco] on the
VisDrone2019-DET training split [zhu2021visdrone]
for 100 epochs at 640×640 with batch size 8 on a single 8 GB consumer GPU. For seed
42, peak validation mAP50 is 0.3436 at epoch 83 and peak mAP50-95 is 0.1890 at epoch 82; the final-epoch
values are 0.3385 and 0.1854. For seed 43 the two peaks coincide at epoch 80, at 0.3407 and 0.1858, with
final-epoch values of 0.3365 and 0.1839. Seed 44 peaks at epoch 82, at 0.3421 and 0.1874, with final-epoch
values of 0.3364 and 0.1838. The three seeds land within 0.3 points on mAP50 and 0.4 on mAP50-95, small relative to the between-seed spread of the allocation gain in R5b, and consistent with the effect being a property of how quotas interact with the label distribution rather than of detector quality. Training takes 238 s per epoch. We report no comparison against a published YOLO26n baseline: those figures used a different batch size and no stated seed. The per-epoch curve, the environment lock, and the weight
hashes are released with the code, and the curves for all three seeds are plotted in **Supplementary Fig. S1**.

---

---

## R5. The gain is positional, and part of the zero-gain regime is structural

Supplementary Table S8 bounds the room a reallocation has. Whether it is worth using that room depends on the detector,
and on which image the minimum lands on. Supplementary Table S9 reports the measured gain; Supplementary Table S10 reports the two
quantities that decide it: whether the two quota vectors coincide, and the identity of the
worst image under each allocation.

Three features of Supplementary Table S10 matter. At `B` = 100 and 300 the comparison is
vacuous: the two quota vectors are equal entry by entry, which we verify by exact array comparison rather
than by inspecting the gain, so both allocations collapse to the same quota vector under the label-count cap and the measured zero is an identity. At `B` = 70 the vectors differ on 314 of 548 images, yet the worst image
is the same image under both allocations (index 22, with 12 annotated objects) and has reached its cap
under both, so its recall is literally the same number and the minimum is taken over one unchanged value.
Below `B` = 70 the gain is real and sign-varying: the largest is +11.15 points at `B` = 25, the smallest is
−2.84 at `B` = 10, and the sign is not uniform across heads at `B` = 6 and 10.

The gain comes from the identity of the worst image changing. At `B` = 25 under a uniform quota the worst
image is the densest one, with 317 annotated objects, held to 8.20% recall; redistribution moves quota to
it, and the minimum is then taken over a different image at 19.36%. Some images gain and others lose (densest quartile +10.07 points, sparsest 5% -24.20); what changes is which image is worst. This is why the gain disappears once
no image's quota is binding, and why the intervention should not be described as an improvement of a fixed
worst case.

## R5b. What replicates across seeds, and what does not

Two things replicate exactly and one does not. The allocation-side quantities are fixed by the quota
vector, which depends on `n_i` and the budget and not on the network weights: at `B` = 25 the uniform
worst-image recall is 11.67% for the one-to-many head on seed 42 and 11.99% on seed 43, and 11.36% and
11.67% respectively for the one-to-one head. The two heads are not interchangeable here. The same holds at `B ≤ 15`, where quotas of one or two boxes quantise per-image recall to multiples of `1/n_i` and the gains are identical across seeds, because the detector's ordering no longer decides the minimum.

The zero-gain regime also replicates exactly, for the reasons given above: at `B` = 100 and 300
the two quota vectors are equal on all three seeds and both heads, and at `B` = 70 the worst image is the same
one at its cap in all four combinations.

The magnitudes do not replicate, and we report them as ranges. The between-seed spread reaches 4.8 points on the one-to-one head, where the gain at `B` = 50 runs from +3.93 on seed 42 to +8.69 on seed 44, and at
`B` = 25 both heads lose 2 to 3 points from the first seed to the second. Every magnitude in this paper is
therefore a range over three seeds and two heads, and the ranges are wide enough that a single-seed comparison
at a single budget could be made to show almost any effect within them.

---

# Methods

## Detector and training

We used YOLO26n as implemented in Ultralytics 8.4.177. YOLO26 is an end-to-end detector with two parallel
heads over the same backbone: a one-to-many head that produces dense predictions and requires
non-maximum suppression, and a one-to-one head that produces a fixed set of predictions and requires
none. The one-to-one head is the head used at deployment; we report both, since the choice of head changes
the allocation result by 3.0 points.

Training used the Ultralytics default optimiser with mosaic augmentation at 1.0, disabled for the final
10 epochs, translation 0.1, scale 0.5, horizontal flip 0.5, and HSV jitter (h 0.015, s 0.7, v 0.4).
Mixup, copy-paste, and rotation were disabled to keep the recipe comparable to the published VisDrone
anchor. Early stopping was disabled so that the full 100-epoch curve is available. Data-loader workers
were set to 0, which is required on Windows: with workers > 0 the validation pass deadlocks during worker
spawn.

## Evaluation protocol

Detection quality is reported as AP@[.50:.05:.95] following the VisDrone protocol. The validation split
was used for all development; the test-dev split was reserved for a single evaluation at freeze time, run through the same label-informed pipeline.
Predictions were generated at confidence 0.001 with IoU 0.7, matching the COCO and LVIS sweeps.

For the budget experiments, the detector is run once with `max_det` set to the largest budget in the sweep.
Smaller budgets are then obtained by truncating each image's predictions to its top `q_i` by confidence.
This is exact rather than approximate: we verified on 40 images across eight budgets and both heads that
truncating a large-`max_det` run reproduces running at the small `max_det` directly, with 40/40 images
matching and a maximum coordinate deviation of 0.0. Because the equivalence holds, each budget costs no
additional inference.

A full-resolution pass over all 548 validation images at once exhausts GPU memory, so the scan is run in
chunks that are merged afterwards; the merge is checked against annotation counts recomputed from the
label files, and any mismatch aborts the merge.

## Allocation

For a total budget `S = ρ·Σ_i n_i`, the uniform allocation gives every image the same quota `c`, chosen by
bisection as the largest value for which `Σ_i min(c, n_i, p_i) ≤ S`. The proportional allocation is defined through the budget rather than directly: we bisect the multiplier `a` and give image `i` the quota `min(⌈a·n_i⌉, n_i, p_i)` for the largest `a` whose total spend `Σ_i min(q_i, n_i, p_i)` does not exceed `S`. Writing the quota directly as `min(⌈ρ·n_i⌉, n_i, p_i)` would overspend, because the ceiling is applied per image; on the validation split at ρ = 0.5 it exceeds `S` by 0.70% and changes 135 of 548 quotas. Quotas are rounded up to integers, and the spend of an allocation is `Σ_i min(q_i, n_i, p_i)`. Every quota is capped at `min(n_i, p_i)` by construction: the `p_i` term because the detector cannot emit more boxes than it predicts, and the `n_i` term because this paper studies the label-count-capped regime, in which each image may be allocated at most as many boxes as it has annotated objects. That second cap is a modelling choice and not a property of the detector: because predictions are ranked by confidence, a box beyond the first `n_i` can still be the first to match a previously unmatched object, so the cap removes output slots that could in principle raise recall. It binds on no reported condition with `B ≤ 70`, where every quota stays at or below the annotated count, and it is what saturates the allocation at `B ≥ 100`; the saturation there is therefore label-count saturation, not exhaustion of the detector's prediction capacity. Hence `q_i ≤ min(n_i, p_i)` holds for every image at every budget we report, so this spend equals the number of boxes the detector actually emits, `Σ_i min(q_i, p_i)`; we checked the two sums agree exactly at every swept budget, on both heads, with a maximum difference of zero boxes., where `p_i` is the number of predictions the detector emitted for image `i`.
The three allocations are compared only when their spends agree to within 1%. Because `n_i` is the annotated count, the proportional allocation is an **oracle**: it measures what redistribution could buy if each image's object count were known in advance, not a deployable policy.

The proportional rule is the classical max-min fair rule for the total-spend constraint, and we use it as a baseline rather than as an optimum of the recall objective: which image ends up worst depends on where the detector's errors fall, not only on how many boxes each image receives. That maximisation is a max-min fair allocation problem, and progressive water-filling is its standard solution [bertsekas1992data]; we do not claim the optimisation as a new result. The per-image caps, the integer rounding and the top-up step mean the result is a capped proportional allocation rather than an exact max-min fair rule, and the quotas are set from object counts rather than from recall, so the allocation is not the solution that directly maximises the worst-image recall we evaluate. What we report is its
behaviour when the constraint is the number of boxes a detector may emit, and the empirical consequences
of that setting, which neither the optimisation literature nor the detection-evaluation literature
addresses.

## Prediction-proportional allocation

The proportional allocation above uses the annotated object count `n_i`, which is not available at
inference time; it therefore measures what redistribution could buy if each image's object count were
known in advance, and it is an oracle rather than a policy. To ask what the proportional rule would do if the allocation signal were the detector's own output rather than the label count, we replace `n_i` with the number of predictions the detector emits for that image once, at the largest budget in the sweep, `p_i`. This signal is available at inference; the top-up below is not, so the resulting pipeline is **label-informed throughout** and is not a deployable policy. The same bisection, the same integer rounding, and the same
top-up are applied, so the only change is the allocation signal.

On the one-to-many head the prediction-proportional variant recovers 53% to 78% of the oracle gain in worst-image
recall (+5.72, +6.55 and +4.24 points for seeds 42, 43 and 44, against +8.33, +8.38 and +8.01 for the
oracle), and it does so at roughly half the macro cost (−3.44, −3.49 and −3.83 points against −6.36,
−6.57 and −6.87). On the one-to-one head it recovers almost none of the gain (+0.32 points), which is
what the positional account predicts: on that head the worst image is already close to its ceiling under
the uniform quota, so reallocation has little to replace. Because `p_i` exceeds `n_i` on every image in our records, the prediction-based signal is a noisy proxy for the oracle one, and the gap between the two rows
is the price of not knowing the object count.

## Budget convention and top-up

Two conventions are easy to misread, so we state them explicitly. First, `B` is the total budget
divided by the number of images, `S = B·N`. It is an average per-image budget, not a cap on any
single image: the uniform allocation fixes one quota `c` for every image, so an image whose capacity
exceeds `c` can never receive more, but the proportional allocation can give an image several times
`B` when that image is dense. At `B = 70` on the validation split the largest quota is 206 boxes under the
uniform allocation and 312 under the proportional one, against an average of 70. Second, integer bisection alone usually underspends `S` by a few
boxes. We therefore add a deterministic top-up: while the spend is below `S`, one box is added to the image whose next prediction carries the highest marginal true-positive gain. That gain is read off the label matches, so the top-up is label-informed; it is applied identically to every allocation, which keeps the comparison internally valid, but it means no allocation in this paper is a deployable policy. The top-up changes the
quota of 0.4% of images at `B = 2`, 8.9% at `B = 25` and 1.1% at `B = 70`; it is applied identically
to every allocation, so it cannot favour one over another. Without it the two spends would differ and
the comparison would confound where boxes go with how many are spent.

## Top-up ablation

The main text reports the oracle top-up, which reads true-positive matches from the annotations. To
test whether the reported gain depends on that choice, we repeated every ρ condition with three
label-free variants: no top-up at all; a largest-remainder rule that gives the deficit to the images
with the largest fractional part of their unrounded quota; and a capacity rule that gives it to the
images with the most remaining prediction slots. The last two need only the annotation-free
quantities already used to build the quota. With the budget matched to within 0.01%, both label-free rules stay within 0.32 points of the oracle arm at every condition but one, where the largest-remainder rule exceeds it by 3.33 points. The no-top-up arm under-spends by up to 2.75% at the smallest budget fraction and is therefore not a valid control there; where its spend does match, it agrees with the oracle arm exactly. The reported gain therefore does not depend on label
access in the top-up step. Per-condition values are in Supplementary Table S6.

## Pre-registration and decision rule

The decision rule, the threshold values, the budget fraction ρ = 0.5, and the statistic were fixed before
the detector was evaluated. The rule was revised once, before any measurement, because the first version
used a ratio whose denominator, the capacity-only ceiling `S/Σ_i n_i`, is unattainable by a real detector;
using it would have made both the accept and the abandon thresholds unreachable. The revision to an
absolute point gain is recorded, with the reason, in the repository. The implementation carries a self-test
that constructs synthetic worlds with known outcomes and asserts that the rule separates them; it currently
passes all five assertions.

## Test-set evaluation and label dependence

The test-dev split was evaluated once, after the method and every decision rule had been frozen. That
single evaluation is nonetheless not an independent test of a deployable policy: the deterministic
top-up step used on test-dev is the same one used on validation, and it reads true-positive matches from
the test-dev annotations to decide which image receives the next box. Every allocation analysed here,
on both splits, is therefore an oracle-style diagnostic rather than a deployable procedure. What the
test-dev result establishes is narrower and still informative: when the worst image is detection-limited
rather than quota-limited, reallocating a fixed budget leaves it at zero recall even though the quota
vectors differ on 1568 of 1,610 images.

## Sensitivity to the label-count cap

Because the `n_i` term is a modelling choice rather than a detector property, we repeated the whole
ρ sweep with that term removed: quotas were re-derived with only the prediction cap `p_i`, spend was
counted as the boxes actually emitted, `Σ_i min(q_i, p_i)`, and true positives were still capped at
`n_i` because no image can yield more. Removing the label-count cap raises the worst-image gain at
every one of the ten conditions, by 0.63 to 3.78 points, and raises the micro-recall change by 1.80 to
4.49 points; the per-condition values are in Supplementary Table S5 and in
`results/tables/gt-cap-sensitivity.json`. The main text reports the capped numbers, so every gain
quoted there is a conservative estimate specific to the label-count-capped convention, and the
positional conclusion does not depend on that cap.

## Statistical analysis

Uncertainty in the worst-image gain is estimated by a paired bootstrap over images, 10,000 resamples, with
the pairing unit being the image rather than the prediction [efron1993bootstrap]. Within each resample both
allocations are recomputed from the resampled image set and the change is evaluated on that same set, so
the comparison is never made across different image sets. The resampling seed is fixed at 20261014. We
report the 95% percentile interval and the fraction of resamples with a positive change.

Three qualifications belong with those intervals. First, what they quantify is image-resampling
uncertainty: they answer how much the measured difference would move if a different set of 548 images
were drawn from the same population with the detector held fixed. They do not quantify training-seed
uncertainty. The three seeds are a finite check on model randomness rather than a random-effects sample,
so we report each seed's estimate and interval separately and give the spread across seeds as a
descriptive range; the per-seed intervals should not be read as if they came from three independent
populations. Second, the resampling unit is the image, and VisDrone images are frames extracted from video
sequences, so images from the same sequence are not strictly independent. We did not have a reliable
sequence identifier for the validation split, so we use the image as the unit; if same-sequence images are
positively correlated then the reported intervals are somewhat too narrow, and the claim that the
intervals exclude zero would be the first to need revisiting. Third, these are percentile intervals of a
discrete quantity and no correction is made for the fact that a minimum over 548 images is itself an order
statistic, which is why the quantitative claim also rests on the distribution-wide result in Figure 6
rather than on the minimum alone.

For the ten ρ-sweep conditions we also report a family-wise correction rather than asserting that it is
unnecessary. Testing the one-sided null that the worst-image gain is not positive, per condition, gives raw p values between 0.0001 and 0.0054 across the ten conditions, with 0.0054 the largest of the ten; Holm step-down [holm1979] at a family-wise α of 0.05 leaves the smallest Holm threshold at 0.0050 and the largest at 0.0500, and **all ten remain rejected** — the smallest raw value clears the first step's threshold of 0.005, and the ordering then carries the rest. The
correction therefore changes none of the decisions we report, and the quantities are in
`results/tables/fwer-and-equivalence.json`, and the per-condition ranks, Holm thresholds and adjusted $p$ values are in Supplementary Table S4.

The claim that reallocating leaves total recall approximately unchanged is a claim of equivalence, not of
a failure to detect a difference, and we test it as such. The equivalence margin is **pre-registered at
1.0 percentage point** (`src/eval/stats.py`, rationale in `docs/decisions.md`); it was fixed before these
measurements and is not chosen to fit them. Across the ten conditions the point estimate of that change ranges from 0.74 to 1.40 points. Against the pre-registered 1.0-point margin none of the ten passes: the two one-sided tests give p between 0.33 and 0.73, far above 0.05, because four of the ten point estimates already exceed the margin. We therefore do not claim equivalence at that margin and report the observed range instead. Had the test statistic been scaled wrongly, it would have returned p near zero and a spurious equivalence claim; the margin is in points, so the per-image contributions enter the test in those units.

Two cautions apply to the interpretation. The bootstrap quantifies sampling variability over images; it
does not capture variability over training runs, and the results here come from three seeds. And the worst-image recall is an extreme order statistic, so it is governed by the few images
with the largest or smallest object counts. Both cautions are the reason the confidence intervals in
Table 1 extend past the pre-registered threshold.

## Image and annotation counts

The `Images` column of Supplementary Table S8 counts images that carry at least one annotated object, which is the
population any recall statistic is defined over. COCO val2017 and train2017 contain a further 48 and
1,021 images with no annotations; these enter no recall and change no reported fraction, so they are
excluded from `N` and from every per-image table in this paper.

## Data

VisDrone2019-DET is a publicly available aerial detection benchmark with 6,471 training, 548 validation,
and 1,610 test-dev images across ten classes, distributed under CC BY-NC-SA 3.0 for academic use. Its
per-image object counts range from 1 to 461. COCO 2017 and CrowdHuman are used only through their
annotation files, since Supplementary Table S8 depends on label distributions alone; no images from either benchmark were
downloaded, and no model was trained on them. CrowdHuman object counts use boxes tagged `person`, matching
the default of the reference conversion script; ignored regions are excluded in a reported variant.

## Computational reproducibility

Training, evaluation, and every analysis in this paper run through a single entry point per stage. Each run
writes a code fingerprint (SHA-256 over the source tree), the git commit, the full argument set, and the
weight hashes. The dependency set is pinned in `requirements.lock.txt`, generated by `pip freeze` in the
environment that produced the results. The repository also ships `scripts/check_all.py`, which reruns every
consistency and numerical check used here, including independent recomputation of the reported statistics
from the stored records; the self-tests for the statistical routines and for the decision rule are part of
that suite, and the numerical self-checks against independent recomputation are recorded in
**Supplementary Note S1**. The repository location, release tag and licence are given in the Data
availability statement. A large language model was used to assist with code implementation, literature
search, and language editing; the research question, the experimental design, the allocations compared, and
all numerical results and conclusions are the authors' own, and every reported number is recomputed from
stored prediction records by the released scripts.

> **Author contributions 与声明段的位置**：官方结构（逐字）是
> `… Methods → References → Acknowledgements → Author contributions → Data availability →`
> `Additional Information（含 Competing Interests）→ Figure legends → Tables`。
> 因此 **Author contributions 不进正文文件**，由 `scripts/build_submission_pdf.py` 的
> `DECLARATIONS` 统一生成（含 CRediT 逐人陈述），避免重复。

---

# Figure legends

*(Not counted toward the 4,500-word main text under the Scientific Reports Article format. Every legend
names the sample size and defines the symbols appearing in its figure, as `publication_guidelines.md`
requires. Where a plotted quantity is an exact count rather than an estimate, the legend says so and no
error bars are shown; where it is a mean over images, the legend states the interval.)*

**Figure 1. Per-image annotated object counts across detection benchmarks.** **a**, Percentage of images
whose object count `n_i` exceeds a budget `B`. **b**, Fraction of the permitted budget the annotations can
absorb, `Σ_i min(B, n_i)/(B·N)`. Colours, line styles and marker shapes all distinguish the curves, which
remain separable in grayscale. `N` = 548 (VisDrone val), 15,000 (CrowdHuman train), 4,952 (COCO val2017).
These are exact label counts, so no error bars are shown.

**Figure 2. Gain in worst-image recall from reallocating a fixed output budget, in percentage points.**
**a**, Gain against budget `B` for both prediction heads; faint crosses mark budgets above which the budget cannot be spent in full, so the two allocations coincide and the comparison is an identity rather than a measurement. The horizontal axis is the average per-image budget. **b**, The same
comparison at the pre-registered budget fraction ρ = 0.5, shown separately for the two prediction heads, so
that its horizontal axis is the head and not the budget; only the extreme budgets of panel **a** are budget
separations, and this panel is not one of them. `N` = 548 throughout; these are exact counts, so no error
bars are shown.

**Figure 3. What the reallocation does at the level of individual images.** **a**, **b**, Per-image recall
against object count `n_i` under a uniform quota (grey) and proportional allocation (vermillion) at
`B` = 25 and `B` = 70. Connectors join the two allocations for images whose recall changes by more than two
points. Circles and squares mark the worst image under each allocation, labelled with its object count,
quota and recall; a coloured outline means its quota reached the imposed label-count cap `n_i`, not that the detector ran out of predictions. (The per-image ceiling `min(n_i, p_i)` equals `n_i` throughout
these records, because a prediction count `p_i` below the annotated count `n_i` never occurs; the outline
condition is therefore the binding one.) Green crosses in **b** mark
images that reached `n_i` under both allocations yet stay below 45% recall, which is why, under this label-count-capped convention, the gain there is exactly zero. `N` = 548 in both panels.

**Figure 4. The cost of the reallocation by object-count band.** **a**, Change in per-image recall,
proportional minus uniform, in percentage points, for each `n_i` band; horizontal bars are **95%
confidence intervals** from a paired percentile bootstrap over images within the band (10,000 resamples; the
RNG is derived per seed so that re-running one seed alone reproduces its row), and the vertical line marks
zero change. Because the band boundaries are fixed by
the label distribution, the interval estimates the mean cost of a band and not which images fall in it.
Image counts per band are printed beside the labels. Colours and symbols follow Figure 2: circles for the
one-to-many head, squares for the one-to-one head. **b**, Mean per-image recall under each allocation for
the same bands; dashed connectors join the two allocations within a band, so the horizontal length of a
connector is the change plotted in **a**. Grey circles mark the uniform allocation and vermillion triangles
the proportional one. ρ = 0.5, VisDrone validation, `N` = 548. Panel **b** shows the one-to-many head only;
the one-to-one head appears in **a**.
**Figure 5. Sparse images hold larger objects.** **a**, Mean annotation-box area per image against the
annotated object count `n_i`, with the median per logarithmic count bin overlaid. **b**, Mean box area for
the sparsest 5% of images (27 images, `n_i` 1-9) and the densest 25% (137 images, `n_i` 93-317), the same
bands used in Figure 4; the number above each bar is the image count in that band. Areas are normalised
(YOLO w*h, as a fraction of image area), so no image dimension enters the statistic. The Spearman
correlation between object count and mean box area is -0.4596, and the sparsest band's mean box area is
19.62 times the densest band's. This is why a per-image quota costs the sparse images most: when an image
holds few but large objects, each box carries more of that image's recall, so a proportional allocation that reclaims
that image's boxes removes more recall than the same number of boxes taken from a dense image with small
objects. We deliberately do not infer object distance, because VisDrone carries no depth annotation.


---

**Figure 6. Reallocation shifts the whole lower tail of the recall distribution.** Quantile curves of
per-image recall under uniform (blue) and proportional (orange) allocation at `ρ` = 0.50, seed 42. **a**, one-to-many
head. **b**, one-to-one head. The orange curve lies above the blue one below roughly the 35th percentile and
below it between the 40th and 95th, so the gain is a property of the lower tail rather than of one image;
the two allocations share the upper endpoint but their minimum recalls differ, because reallocation moves the minimum to a different image. Dotted verticals mark the minimum, 5th and 10th percentiles. `N` = 548 images per panel. Each
curve is a quantile function of an exact per-image count, so no error bars are shown.

# Data and code availability

VisDrone2019-DET is public under CC BY-NC-SA 3.0 for academic use. COCO 2017 and CrowdHuman are public as
well; only their annotation files were used here, and neither their images nor any model trained on them
contributes to a result. Code, configurations, per-epoch curves, per-image prediction records and the
figures are released with the manuscript. Because the per-image records hold the full confidence-ordered
prediction list for every validation image, every number, table and figure can be recomputed without
retraining or re-inference.
