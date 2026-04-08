from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataFileConfig:
    symbol: str
    path: str


@dataclass
class DataConfig:
    provider: str = "local_csv"
    files: list[DataFileConfig] = field(
        default_factory=lambda: [DataFileConfig(symbol="SPY", path="data/example_prices.csv")]
    )
    symbols: list[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "^VIX", "TLT", "GLD",
        "EEM", "HYG", "LQD", "DBC", "UUP", "IEF", "XLF", "XLE",
    ])
    timestamp_col: str = "timestamp"
    price_col: str = "mid_price"
    symbol_col: str | None = None
    regular_hours_only: bool = False
    returns_mode: str = "log"
    interval: str = "1m"
    period: str = "7d"
    start: str | None = None
    end: str | None = None
    chunk_days: int | None = None
    auto_adjust: bool = False
    prepost: bool = False
    force_refresh: bool = False
    cache_dir: str = "data/cache"


@dataclass
class LabelConfig:
    lookahead_bars: int = 30
    volatility_window_bars: int = 30
    threshold_quantile: float = 0.9
    threshold_lookback_bars: int = 7800
    min_history_bars: int = 390
    event_merge_gap: int = 5
    min_event_span: int = 1
    positive_transition_only: bool = True
    vix_confirmation_col: str | None = None
    vix_confirmation_threshold: float | None = None


@dataclass
class FeatureConfig:
    window_bars: int = 60
    stride_bars: int = 5
    embed_dim: int = 8
    embed_tau: int = 1
    graph_knn_k: int = 8
    betti_radii: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    )
    persistence_image_bins: int = 8
    topology_feature_sets: list[str] = field(default_factory=lambda: ["summary", "betti", "image"])
    classical_features: list[str] = field(
        default_factory=lambda: [
            "realized_volatility",
            "bipower_variation",
            "skewness",
            "kurtosis",
            "lag1_autocorr",
            "downside_semivariance",
            "upside_semivariance",
            "cumulative_return",
            "trend_slope",
            "sign_flip_rate",
            "mean_abs_return",
        ]
    )
    include_symbol_one_hot: bool = True
    enable_vxx_tailored: bool = False
    vxx_tailored_short_horizon: int = 10
    vxx_tailored_radii: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    )
    vxx_tailored_topology_feature_sets: list[str] = field(default_factory=lambda: ["summary", "betti"])


@dataclass
class ModelConfig:
    enabled: list[str] = field(
        default_factory=lambda: [
            "vol_threshold",
            "rf_classical",
            "rf_topology",
            "rf_combined",
            "mlp_combined",
            "gcn_graph",
            "gcn_fusion",
        ]
    )
    random_state: int = 42
    rf_estimators: int = 300
    rf_max_depth: int | None = None
    mlp_hidden_sizes: list[int] = field(default_factory=lambda: [64, 32])
    gdl_hidden_dim: int = 32
    gdl_dropout: float = 0.1
    gdl_epochs: int = 20
    gdl_patience: int = 5
    gdl_batch_size: int = 64
    gdl_learning_rate: float = 0.001
    gdl_weight_decay: float = 0.0001
    gdl_balance_classes: bool = True
    gdl_focal_gamma: float | None = None
    gdl_n_ensemble: int = 1
    gdl_use_cuda: bool = True
    enable_symbol_offset_calibration: bool = False
    symbol_offset_fit_split: str = "validation"
    symbol_offset_grid: list[float] = field(
        default_factory=lambda: [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    )
    probability_threshold_grid: list[float] = field(
        default_factory=lambda: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    )


@dataclass
class EvaluationConfig:
    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2
    purge_bars: int | None = None
    embargo_bars: int | None = None
    min_validation_events: int = 5
    min_test_events: int = 5
    early_warning_bars: int = 15
    bars_per_day: int = 390
    bootstrap_samples: int = 100
    bootstrap_block_size: int = 390
    selection_metric: str = "event_f1"
    max_false_alarms_per_day: float | None = 0.10
    max_positive_rate: float | None = 0.5
    promotion_confirmation_runs: int = 2
    evaluation_mode: str = "search"
    walk_forward_min_train_years: int = 10
    walk_forward_val_years: int = 1


@dataclass
class OutputConfig:
    root_dir: str = "outputs"
    metrics_filename: str = "metrics.json"
    predictions_filename: str = "predictions.csv"
    event_table_filename: str = "event_table.csv"
    model_table_filename: str = "model_comparison.csv"
    per_symbol_table_filename: str = "per_symbol_metrics.csv"
    short_event_table_filename: str = "short_event_metrics.csv"
    manifest_filename: str = "run_manifest.json"
    split_summary_filename: str = "split_summary.json"
    walkforward_fold_summary_filename: str = "walkforward_fold_summary.json"
    summary_filename: str = "run_summary.md"
    figure_dir: str = "figures"
    table_dir: str = "tables"


@dataclass
class ResearchConfig:
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ResearchConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        data_raw = raw.get("data", {})
        files = [DataFileConfig(**item) for item in data_raw.get("files", [])]
        if files:
            data_raw = {**data_raw, "files": files}
        config = cls(
            data=DataConfig(**data_raw),
            labels=LabelConfig(**raw.get("labels", {})),
            features=FeatureConfig(**raw.get("features", {})),
            models=ModelConfig(**raw.get("models", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            outputs=OutputConfig(**raw.get("outputs", {})),
        )
        total = config.evaluation.train_frac + config.evaluation.val_frac + config.evaluation.test_frac
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Evaluation split fractions must sum to 1.0")
        return config
