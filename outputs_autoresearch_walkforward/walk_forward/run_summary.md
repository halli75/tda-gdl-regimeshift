# Run Summary

- Best model: `vol_threshold`
- Evaluation split: `walk_forward`
- PR-AUC: `0.0682`
- Event F1: `0.4040`
- Mean lead bars: `4.792723311546841`
- False alarms/day: `0.6`
- Feature samples: `11835`
- Positive labels: `956`

## Walk-Forward Summary (5 folds)

| Fold | Val Period | Events | vol_threshold F1 | rf_topology F1 | rf_combined F1 |
|------|-----------|--------|------|------|------|
| 0 | 2016-07-14–2017-10-09 | 22 | 0.367 | 0.108 | 0.000 |
| 1 | 2017-07-12–2018-10-05 | 50 | 0.572 | 0.131 | 0.000 |
| 2 | 2018-07-10–2019-10-04 | 46 | 0.497 | 0.069 | 0.131 |
| 3 | 2019-07-09–2020-10-01 | 40 | 0.279 | 0.176 | 0.143 |
| 4 | 2020-07-06–2021-09-29 | 13 | 0.304 | 0.000 | 0.000 |

- Mean vol_threshold event F1: `0.4040 ± 0.1132`
- Mean rf_topology event F1: `0.0969 ± 0.0597`
- Mean rf_combined event F1: `0.0548 ± 0.0672`
- Mean gcn_graph event F1: `0.0000 ± 0.0000`
- Mean gcn_fusion event F1: `0.3933 ± 0.2772`