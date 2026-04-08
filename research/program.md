# Research Program

You are operating an autonomous but constrained research loop for the `tda-gdl-regime` project.

## Objective

Improve out-of-sample regime-shift detection on intraday ETF data, with the hybrid `gcn_fusion` TDA+GDL model as the optimization target, and keep the manuscript synchronized with the strongest reproducible evidence.

## Allowed Mutations

- YAML config files in `configs/`
- analysis or reporting code that consumes pipeline artifacts
- manuscript sections in `paper/`
- research planning files in `research/`

## Disallowed Mutations

- changing the dataset schema without updating manifests
- deleting prior experiment artifacts
- editing unrelated workspace directories
- claiming empirical results in the paper that are not backed by saved outputs

## Required Loop

1. Read the latest validation-state summary and comparison tables.
2. Propose one bounded experiment that changes only a small number of config values, prioritizing hyperparameters and representation choices that can improve `gcn_fusion`.
3. Run the pipeline in `search` mode only.
4. Record the hypothesis, config delta, validation metrics, and keep/discard decision.
5. Touch the final test split only in an explicit finalization run after search is complete.
6. Update the manuscript only if the result is reproducible and materially improves the evidence.

## Primary Decision Rule

Prefer experiments that improve:

- PR-AUC
- event F1
- mean early-warning lead time
- false alarms per trading day

Do not optimize on final-test metrics during the search loop, and reject candidates with materially negative mean lead bars by default.
