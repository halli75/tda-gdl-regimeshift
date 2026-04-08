# Response to Reviewer — Minor Revision (Round 3)

**Paper**: Walk-Forward Evaluation of Hybrid Econometric–Graph Models for Financial Regime Detection  
**Submitted revision**: main_ver9.tex

We thank the reviewer for the careful reading and the upgrade to Minor Revision. The verdict and the positive assessment of the core contributions are encouraging. Below we address each remaining concern in turn. All five have been resolved; the paper has been updated accordingly.

---

## Concern 1 — H₁ null result: inference too strong

**Reviewer**: "The current text says 'H₁ cycle structure does not add discriminative power' — but the data actually say 'we cannot tell whether H₁ adds discriminative power at this scale.' These are different claims."

**Response**: We agree entirely. With permutation importance standard deviations of 0.006–0.009 and means of −0.008 to −0.007 across the three H₁ features, the values are statistically indistinguishable from zero at n=48 test events. We have revised all five locations in the paper where H₁ results appear (abstract, H₁ paragraph in Section 7, table caption for Table 7, conclusion, and limitations section) to use the correct inferential language:

> "H₁ permutation importances are statistically indistinguishable from zero at n=48 test events (mean: −0.008, std: 0.009); we cannot exclude small effects from the point estimates alone, as the test set is underpowered to detect sub-0.01 importance differences."

The abstract now reads: "all three H₁ permutation importances statistically indistinguishable from zero at n=48 test events (mean ≈ −0.008, std ≈ 0.009); the result is consistent with H₀-only sufficiency at this embedding scale, though limited statistical power (n=48) prevents ruling out small effects."

We retain the experiment as an informative negative finding. The limitations section has been updated to acknowledge statistical power as the binding constraint rather than treating H₁ extension as a straightforward improvement.

---

## Concern 2 — τ=2 result underexplored (single seed)

**Reviewer**: "You're implicitly claiming a 3% PR-AUC improvement over production based on one seed — that's a stronger claim than the evidence warrants. Either run the ensemble or add the caveat prominently."

**Response**: We ran a 3-seed ensemble at τ=2 (embed_dim=8, embed_tau=2, n_ensemble=3). The result is:

- **τ=2, 3-seed ensemble**: PR-AUC = **0.100**, 95% CI [0.086, 0.126]
- **τ=1, production (3-seed)**: PR-AUC = 0.102

The CIs fully overlap. The single-seed result (0.132, CI [0.107, 0.176]) was high-variance noise — a regression to the mean when properly ensembled, consistent with the std=0.002 we observe at τ=1. The 3-seed τ=2 result is within 0.001 of τ=1 production.

Table 6 (embedding sensitivity) now shows both the 1-seed and 3-seed τ=2 rows side by side. The sensitivity paragraph has been revised to state:

> "A 3-seed ensemble at τ=2 yields PR-AUC = 0.100 (95% CI [0.086, 0.126]), with CIs fully overlapping the production τ=1 result. The single-seed τ=2 result was high-variance noise; ensemble evaluation reveals no advantage from sparser temporal sampling at m=8."

Contribution 2 has been updated to remove the "best single-model result" claim. The future directions item for τ=2 investigation has been removed since the ensemble experiment resolves the question.

---

## Concern 3 — Ensemble threshold proxy unresolved

**Reviewer**: "It's an easy fix in the codebase — save predictions_val.csv during training and rerun the ensemble threshold selection."

**Response**: Done. We wrote `compute_gcn_val_predictions.py`, which retrains gcn_fusion (single seed) and saves predictions on the val split to `predictions_val.csv` (2,296 rows, feature-window aligned). `compute_hybrid_detector.py` was updated to use this file directly as the val frame for threshold selection, replacing the EGARCH proxy entirely.

The ensemble now uses blended val scores (50% actual GCN + 50% EGARCH) for threshold grid-search. The PR-AUC of 0.118 is unchanged (threshold-independent); the proxy caveat has been removed from Section 7. The JSON output field `gcn_val_source` confirms: `"predictions_val.csv (actual GCN val predictions, feature-window aligned)"`.

---

## Concern 4 — Gap trajectory has no confidence intervals

**Reviewer**: "Adding block bootstrap CIs on each gap value would either strengthen the claim significantly or reveal it's noisier than it looks."

**Response**: We added block-bootstrap 95% CIs (B=5,000, block=64) for the Round 6 gap. The results reveal it is noisier than the point estimate alone suggests:

- **Round 6 gap** (walk-forward): −0.055, **95% CI [−0.134, 0.000]**  
- **Round 5 gap** (historical, predictions not retained): −0.157 (point estimate only)  
- **Round 3 gap** (single-split, historical): −0.215 (point estimate only)

The CI for Round 6 is wide, driven by the small number of test events (n=194). The point estimate is on the boundary of conventional significance — the CI reaches zero on the upper end — which is an honest representation of the uncertainty at this sample size. We state this explicitly in the gap subsection and note that the historical rounds cannot be bootstrapped because the predictions are no longer retained.

The gap subsection now reads: "Block-bootstrap 95% CI for the Round 6 gap: [−0.134, 0.000] (B=5,000, block=64); the wide CI reflects the small number of test events (n=194) and constitutes an honest uncertainty bound on the walk-forward improvement claim."

---

## Concern 5 — ECE = 0.320 not addressed

**Reviewer**: "Temperature scaling on the GCN logits is literally five lines of code. Either add it as an experiment or remove it from future directions."

**Response**: We applied temperature scaling. Using val GCN scores from `predictions_val.csv`, we grid-searched T ∈ [0.1, 10.0] (200 points) to minimize val ECE. Results:

- **T\*** = 0.80 (val ECE: 0.172 → 0.168)
- **Test ECE**: 0.320 → **0.302** (−5.5%)
- **Test PR-AUC**: 0.102 → 0.102 (unchanged; monotonic transform)

The modest improvement (−0.018 ECE) is noted in the calibration section with the interpretation that the miscalibration is structural rather than a simple overconfidence offset correctable by a single scalar. This is consistent with the score compression observed in gcn-graph (ECE=0.423) and the tabular features partially correcting it in gcn-fusion.

"Calibration correction (Platt scaling, isotonic regression)" has been removed from future directions. The temperature scaling result is reported as a completed experiment in Section 5.3. Future directions now target `gcn-graph` (ECE=0.198) as the remaining calibration improvement opportunity.

---

## Summary of Changes

| Concern | Action Taken | Key Numbers |
|---|---|---|
| H₁ inference | Language softened in 5 locations | "indistinguishable from zero at n=48" |
| τ=2 validation | 3-seed ensemble run | PR-AUC=0.100, CI [0.086, 0.126]; noise confirmed |
| Ensemble proxy | predictions_val.csv generated; hybrid rerun | Proxy removed; actual GCN val scores used |
| Gap CIs | Block bootstrap (B=5000) | Round 6: [−0.134, 0.000] |
| ECE calibration | Temperature scaling applied | T*=0.80; ECE 0.320→0.302; PR-AUC unchanged |
