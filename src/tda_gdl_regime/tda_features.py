from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist


def delay_embedding(series: np.ndarray, m: int, tau: int) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    width = (m - 1) * tau
    if len(values) <= width:
        raise ValueError("Series too short for the requested delay embedding")
    embedding = np.empty((len(values) - width, m), dtype=float)
    for column in range(m):
        embedding[:, column] = values[column * tau : column * tau + embedding.shape[0]]
    return embedding


def compute_persistence_diagram(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((0, 2), dtype=float)
    distances = pdist(points, metric="euclidean")
    dendrogram = linkage(distances, method="single")
    deaths = dendrogram[:, 2]
    births = np.zeros_like(deaths)
    return np.column_stack([births, deaths])


def persistence_summary(diagram: np.ndarray) -> np.ndarray:
    if len(diagram) == 0:
        return np.zeros(3, dtype=float)
    lifetimes = diagram[:, 1] - diagram[:, 0]
    return np.array(
        [
            float(lifetimes.mean()),
            float(lifetimes.max()),
            float(lifetimes.sum()),
        ],
        dtype=float,
    )


def betti_curve(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.ones(len(radii), dtype=float)
    distances = pdist(points, metric="euclidean")
    dendrogram = linkage(distances, method="single")
    merge_distances = np.sort(dendrogram[:, 2])
    if merge_distances.size == 0:
        return np.ones(len(radii), dtype=float)
    max_radius = float(np.max(radii)) if radii.size else 0.0
    scale = merge_distances[-1] if max_radius <= 1.0 else 1.0
    actual_radii = radii * scale
    n_points = len(points)
    counts = []
    for radius in actual_radii:
        merges = np.searchsorted(merge_distances, radius, side="right")
        counts.append(n_points - merges)
    return np.asarray(counts, dtype=float)


def persistence_image(diagram: np.ndarray, bins: int) -> np.ndarray:
    if len(diagram) == 0:
        return np.zeros(bins * bins, dtype=float)
    births = diagram[:, 0]
    persistence = diagram[:, 1] - diagram[:, 0]
    birth_scale = births.max() if births.max() > 0 else 1.0
    persistence_scale = persistence.max() if persistence.max() > 0 else 1.0
    histogram, _, _ = np.histogram2d(
        births / birth_scale,
        persistence / persistence_scale,
        bins=bins,
        range=((0.0, 1.0), (0.0, 1.0)),
        weights=persistence,
    )
    return histogram.astype(float).ravel()


def topology_feature_vector(
    series: np.ndarray,
    embed_dim: int,
    embed_tau: int,
    radii: list[float],
    image_bins: int,
    feature_sets: list[str],
) -> tuple[np.ndarray, list[str]]:
    embedding = delay_embedding(series, embed_dim, embed_tau)
    normalized = (embedding - embedding.mean(axis=0)) / (embedding.std(axis=0) + 1e-8)
    diagram = compute_persistence_diagram(normalized)
    features: list[np.ndarray] = []
    names: list[str] = []
    if "summary" in feature_sets:
        features.append(persistence_summary(diagram))
        names.extend(["top_summary_mean_lifetime", "top_summary_max_lifetime", "top_summary_total_persistence"])
    if "betti" in feature_sets:
        curve = betti_curve(normalized, np.asarray(radii, dtype=float))
        features.append(curve)
        names.extend([f"top_betti_{idx}" for idx in range(len(curve))])
    if "image" in feature_sets:
        image = persistence_image(diagram, image_bins)
        features.append(image)
        names.extend([f"top_image_{idx}" for idx in range(len(image))])
    if not features:
        return np.zeros(0, dtype=float), []
    return np.concatenate(features), names
