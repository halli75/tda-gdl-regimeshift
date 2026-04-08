# Topological Features for Regime-Shift Detection in Minute-Level ETF Markets

## Abstract

This paper studies whether topological summaries of sliding-window return embeddings improve binary regime-shift detection in minute-level ETF data. The empirical task is early warning for future volatility shocks rather than trading strategy optimization. We frame each trailing return window as a delay-embedded point cloud, compute persistent-homology summaries and persistence-image features, and compare these against classical realized-volatility statistics. The main comparison is between classical-only, topology-only, and combined classifiers evaluated on chronological splits and event-centric metrics such as PR-AUC, event F1, detection rate, and lead time. The working hypothesis is that topological features capture nonlinear structure missed by classical volatility features and therefore improve shift-event detection when fused with a lightweight supervised model.

## 1. Introduction

Regime changes in liquid markets often appear as abrupt transitions in volatility, serial dependence, and microstructure behavior. Most practical monitoring systems still rely on rolling volatility or parametric state-space models, which can react late when the underlying dynamics change quickly. This project tests whether topological data analysis offers a useful complementary signal for these transitions.

The paper focuses on a narrow, defensible claim: topological features extracted from trailing return windows can improve event-based regime-shift detection in minute-level ETF data. The scope is intentionally detection-first. Any downstream trading analysis is excluded from the main results so the evaluation burden stays tied to classification quality and early warning.

## 2. Research Question and Hypothesis

The central research question is:

Can persistent topological features from high-frequency return windows improve regime-shift detection beyond classical volatility and return statistics?

The tested hypotheses are:

- `H0`: topological features do not improve out-of-sample regime-shift detection relative to classical statistics alone.
- `H1`: topological features improve out-of-sample regime-shift detection, especially on event-level metrics and early-warning lead time.

## 3. Data

The target empirical dataset is minute-level U.S. ETF data for `SPY`, `QQQ`, and `VXX`, processed asset-by-asset and pooled for model training. The implementation in this workspace already supports the experiment contract on local CSV inputs and produces frozen artifacts and dataset manifests for reproducibility. The bundled sample file is only a smoke-test input and should be replaced with real market data for paper results.

Key data handling rules:

- use one provider consistently per study horizon
- work on 1-minute log returns
- preserve chronological order by asset
- keep market-calendar filtering explicit in config
- persist dataset metadata for every run

## 4. Label Definition

The task is binary `stable` versus `shift-event` detection. A positive label is assigned when future realized volatility over a fixed lookahead window exceeds a rolling historical threshold and represents an upward transition from a non-extreme prior state. The current implementation uses:

- trailing volatility window: 30 bars
- lookahead volatility window: 30 bars
- threshold: rolling 90th percentile of past realized volatility
- threshold lookback: 7,800 bars by default
- event consolidation: merge nearby positives to avoid fragmented event counts

This definition keeps the labeling tied to observable future turbulence rather than latent regimes that require a separate hidden-state model.

## 5. Features

### 5.1 Classical Features

Each trailing 60-bar return window is summarized with a compact classical feature set:

- realized volatility
- bipower variation
- skewness
- kurtosis
- lag-1 autocorrelation
- upside and downside semivariance
- cumulative return
- trend slope
- sign-flip rate
- mean absolute return

### 5.2 Topological Features

The same return window is delay-embedded and converted into a point cloud. The current implementation extracts:

- persistence summary statistics
- Betti-0 curve values on a fixed radius grid
- persistence-image bins

These features are designed to remain lightweight enough for repeated automated experiments while still representing multiscale geometric structure.

## 6. Models and Baselines

The benchmark suite is intentionally simple:

- rolling-volatility threshold baseline
- Random Forest on classical features only
- Random Forest on topology features only
- Random Forest on combined classical and topology features
- shallow MLP on combined features

All models use chronological train, validation, and test partitions. Validation data are used to choose the probability threshold that converts scores into event predictions.

## 7. Evaluation Protocol

The main metrics are:

- PR-AUC
- ROC-AUC
- sample-level F1
- event-level precision, recall, and F1
- detected-event rate
- mean lead bars relative to event onset
- false alarms per trading day

The reported results should be pooled and per-symbol, with bootstrap confidence intervals for the primary ranking metrics.

## 8. Reproducibility Workflow

This repository is structured so it can be driven by an `autoresearch`-style loop:

- the experiment configuration is declarative
- the pipeline emits machine-readable metrics and manifests
- comparison tables and figures are regenerated from saved artifacts
- the research loop can propose config changes without rewriting the codebase

That makes it feasible to automate ablations, threshold sweeps, and paper-table regeneration in a controlled way.

## 9. Current Status and Next Results to Insert

Before this manuscript is publish-ready, the following sections still need real empirical content inserted from actual ETF runs:

1. dataset provenance and study horizon
2. quantitative comparison tables on held-out data
3. per-asset robustness analysis
4. ablation showing whether topology adds value over classical features
5. citations and bibliography

## 10. Limitations

The current implementation uses only lightweight topological summaries and does not yet include higher-dimensional persistence, order-book data, or a formal econometric baseline such as Markov switching or GARCH. Those can be added later, but the initial workshop-grade claim should be won or lost on the narrower question of whether topological summaries improve event detection on minute-level ETF returns.
