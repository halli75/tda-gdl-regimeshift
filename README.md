# Walk-Forward Evaluation of Hybrid Econometric–Graph Models for Volatility Regime Detection

**Arnav Gowda** — University of Maryland, College Park
*ICAIF 2026 (Preprint)*

---

## Abstract

We introduce a walk-forward evaluation framework for financial regime-shift detection and apply it to a systematic comparison of hybrid econometric–graph models against classical baselines. The framework reduces the validation-to-test generalization gap from −0.215 under single-split evaluation to −0.055, providing direct evidence that standard protocols overstate out-of-sample performance in nonstationary markets.

Applying the framework to a 13-symbol, daily-frequency cross-asset dataset (2007–2026), we report a five-way comparison on a held-out 4-year test split (2022–2026): EGARCH(1,1) achieves the highest PR-AUC (0.119) and best probability calibration (ECE 0.101), significantly outperforming a GCN on Brier score (DM = −10.37, p < 0.001); two additional econometric baselines — HAR-RV (PR-AUC 0.094) and MS-GARCH (PR-AUC 0.092) — confirm that classical models are strong competitors on probability ranking. The GCN's advantage is exclusive to event-level detection: event F1 = 0.395 vs. 0.155 for EGARCH, reflecting superior operating-threshold calibration. A simple EGARCH-GCN ensemble (PR-AUC 0.118, event F1 0.482) combines both strengths.

A H₁ persistent homology feasibility pilot (Ripser on a 3-symbol subset) finds no PR-AUC improvement over H₀ alone (+0.010, overlapping CIs), with all three H₁ permutation importances statistically indistinguishable from zero at n=48 test events (mean ≈ −0.008, std ≈ 0.009); the result is consistent with H₀-only sufficiency at this scale but should be interpreted as a computational feasibility assessment rather than a definitive test of H₁'s discriminative value at full scale. Label nonstationarity across splits is documented (KS test, p = 0.047), with the residual generalization gap attributed to the 2022–2026 rate cycle.

**Keywords:** financial regime detection, walk-forward validation, topological data analysis, graph convolutional networks, volatility forecasting, persistent homology

---

## Key Results

Test split: 2022–2026 (held-out 4 years) | Dataset: 13-symbol daily cross-asset (2007–2026)

| Model | PR-AUC | Event F1 | ECE |
|---|---|---|---|
| EGARCH(1,1) | **0.119** | 0.155 | **0.101** |
| HAR-RV | 0.094 | — | — |
| MS-GARCH | 0.092 | — | — |
| GCN-fusion (3-seed) | 0.102 | **0.395** | 0.302 |
| EGARCH-GCN ensemble | 0.118 | **0.482** | — |

Walk-forward validation gap (Round 6): −0.055, 95% CI [−0.134, 0.000] (B=5,000, block=64)

---

## Repository Structure

```
tda-gdl-regimeshift/
├── paper/
│   ├── main.tex / main.pdf / main.bbl   # manuscript (ICAIF 2026)
│   ├── figures/                          # all paper figures (PDF)
│   ├── references.bib
│   ├── response_letter_r2.md             # point-by-point reviewer response
│   └── revisions/                        # reproducibility scripts + result JSONs
│       ├── compute_m5_egarch_baseline.py
│       ├── compute_hybrid_detector.py
│       ├── compute_h1_tda_features.py
│       └── ...                           # see Reproducing Results section
├── src/tda_gdl_regime/                   # core library
│   ├── run_pipeline.py                   # main entry point
│   ├── walk_forward.py                   # expanding-window CV
│   ├── tda_features.py                   # TDA: delay embedding, persistence, Betti curves
│   ├── gdl_models.py                     # GCN-graph and GCN-fusion architectures
│   ├── models.py                         # vol-threshold, RF, MLP baselines
│   ├── feature_engineering.py            # classical + topological feature assembly
│   ├── labels.py                         # volatility-threshold label generation
│   ├── evaluation.py                     # PR-AUC, event F1, bootstrap CIs
│   └── ...
├── configs/
│   ├── default.yaml                      # base configuration
│   └── autoresearch_*.yaml               # experiment-specific overrides
├── data/
│   ├── example_prices.csv               # minimal smoke-test input (timestamp, mid_price)
│   └── cache/                            # gitignored; auto-fetched via yfinance
├── research/
│   └── bootstrap_rerun/                  # canonical final run artifacts
│       ├── bootstrap_final_config.yaml   # exact config used for paper runs
│       └── final/                        # predictions.csv, metrics.json, figures
├── outputs_autoresearch_baseline/        # smoke-test run results
├── outputs_autoresearch_daily/           # full multi-symbol daily run
├── outputs_autoresearch_walkforward/     # 5-fold walk-forward run
├── tests/                                # pytest suite (6 files)
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/halli75/tda-gdl-regimeshift.git
cd tda-gdl-regimeshift
pip install -r requirements.txt
```

**Data cache:** `data/cache/` is gitignored. On first run the pipeline will auto-fetch daily OHLCV via yfinance. To use pre-downloaded files, place them at `data/cache/<SYMBOL>_1d_2007-01-02_end.csv`.

---

## Quick Start

```bash
# Smoke test on bundled example data
PYTHONPATH=src python -m tda_gdl_regime.run_pipeline \
  --config configs/default.yaml \
  --evaluation-mode search

# View results
cat outputs/search/run_summary.md
ls outputs/search/tables/
```

---

## Pipeline Overview

The pipeline runs end-to-end from raw prices to evaluation artifacts:

1. **Data loading** — Load daily OHLCV from local CSV cache or yfinance; compute log returns; pool 13 symbols
2. **Label generation** — Binary regime-shift labels: transition into high-volatility regime (forward realized vol ≥ rolling 90th-percentile threshold); ~8% positive rate; events merged within 5-bar gaps
3. **Feature engineering** — Classical features (realized vol, bipower variance, skewness, kurtosis, autocorrelation, semivariance) + TDA features (H₀ delay-embedded persistence summaries, Betti curves, persistence images via Ripser)
4. **Models** — vol-threshold → RF (classical / topology / combined) → MLP → GCN-graph → GCN-fusion; threshold selected on validation split by event F1
5. **Evaluation** — PR-AUC (threshold-independent), event F1/recall, mean lead time, false alarms/day; block-bootstrap 95% CIs (B=5,000, block=64)

**Evaluation modes** (set via `--evaluation-mode`):
- `search` — single train/val/test split with threshold optimization on val
- `walk_forward` — expanding-window cross-validation (5 folds)

---

## Configuration

Key parameters from `configs/default.yaml`:

| Section | Parameter | Default |
|---|---|---|
| Data | symbols | SPY, QQQ, VXX (+ 10 more for full run) |
| Data | interval | 1d |
| Features | window_bars | 60 |
| Features | embed_dim | 8 |
| Features | embed_tau | 1 |
| Features | graph_knn_k | 8 |
| Labels | lookahead_bars | 30 |
| Labels | threshold_quantile | 0.9 |
| Evaluation | train/val/test split | 60% / 20% / 20% |
| Evaluation | early_warning_bars | 15 |
| Evaluation | bootstrap_samples | 5,000 |

---

## Reproducing Paper Results

All revision scripts run from the repo root:

```bash
PYTHONPATH=src python paper/revisions/<script>.py
```

| Script | Purpose | Key Output |
|---|---|---|
| `compute_m5_egarch_baseline.py` | EGARCH(1,1) and vol-threshold baselines | PR-AUC 0.119 |
| `compute_ms_garch_baseline.py` | Markov-switching GARCH baseline | PR-AUC 0.092 |
| `compute_har_rv_baseline.py` | HAR-RV (Corsi 2009) baseline | PR-AUC 0.094 |
| `compute_gcn_val_predictions.py` | GCN validation predictions for ensemble threshold selection | `predictions_val.csv` |
| `compute_hybrid_detector.py` | EGARCH-GCN ensemble (gated + 50/50) | PR-AUC 0.118, F1 0.482 |
| `compute_h1_tda_features.py` | H₁ Ripser feasibility pilot on SPY/GLD/DBC | Permutation importances ≈ 0 |
| `compute_m7_embedding_sensitivity.py` | Embedding dimension (m) and lag (τ) sweep | Table 6 in paper |
| `compute_tau2_ensemble.py` | τ=2 3-seed ensemble validation | PR-AUC 0.100 (noise confirmed) |
| `compute_gap_cis.py` | Block bootstrap gap CIs (B=5,000, block=64) | CI [−0.134, 0.000] |
| `compute_temperature_scaling.py` | Post-hoc ECE calibration via temperature scaling | T*=0.80; ECE 0.320→0.302 |
| `compute_m3_permutation_importance.py` | Feature importance ranking | Table 7 in paper |
| `compute_m2_knn_sensitivity.py` | k-NN graph parameter ablation (k=4,6,8,10) | Table 5 in paper |
| `compute_m4_lookahead_sensitivity.py` | Label lookahead window sweep (10–50 bars) | Appendix |
| `compute_seed_variance.py` | 3-seed ensemble reproducibility | std ≈ 0.002 across seeds |
| `compute_calibration.py` | ECE/MCE reliability diagrams | Figure 1 in paper |

Results are saved as JSON files in `paper/revisions/` and figures in `paper/figures/`.

---

## Tests

```bash
python -m pytest tests/ -v
```

| Test file | Coverage |
|---|---|
| `test_pipeline.py` | Full end-to-end smoke test |
| `test_labels.py` | Volatility threshold labels, event merging |
| `test_splitting.py` | Train/val/test splits, purge/embargo buffers |
| `test_walk_forward.py` | Expanding-window fold generation |
| `test_gdl_calibration.py` | GCN training, focal loss, calibration |
| `test_autoresearch_logic.py` | Config mutation and model ranking logic |

---

## Paper

The manuscript is in `paper/main.tex`. To compile:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

A compiled PDF is included at `paper/main.pdf`.

### Citation

```bibtex
@inproceedings{gowda2026walkforward,
  title     = {Walk-Forward Evaluation of Hybrid Econometric--Graph Models
               for Volatility Regime Detection: When Classical Baselines Win
               and Where Learned Geometry Adds Value},
  author    = {Arnav Gowda},
  booktitle = {Proceedings of the ACM International Conference on AI in Finance (ICAIF)},
  year      = {2026}
}
```
