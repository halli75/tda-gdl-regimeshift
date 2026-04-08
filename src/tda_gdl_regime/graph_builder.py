"""
graph_builder.py
================

This module constructs simple k‑nearest neighbour graphs from point clouds and
computes summary statistics that capture aspects of their local connectivity.
Graph neural networks often rely on adjacency information, but here we
approximate the graph structure with basic numerical features.  By avoiding
heavy dependencies such as ``networkx`` we keep the pipeline lightweight and
portable.

Functions:
    build_knn_graph(points, k): construct a kNN adjacency matrix.
    graph_summary(adj): compute summary statistics from an adjacency matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

def build_knn_graph(points: np.ndarray, k: int) -> np.ndarray:
    """Build a symmetric k‑nearest neighbour adjacency matrix.

    For each node, find its ``k`` nearest neighbours (excluding itself) based on
    Euclidean distance.  Create an unweighted graph where an edge exists
    between two nodes if either node is in the other's neighbour set.  The
    resulting adjacency matrix is symmetric and has zeros on the diagonal.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n_nodes, dim) containing node coordinates.
    k : int
        Number of nearest neighbours for each node.

    Returns
    -------
    np.ndarray of shape (n_nodes, n_nodes)
        Symmetric adjacency matrix with entries in {0, 1}.
    """
    n = points.shape[0]
    if k >= n:
        raise ValueError("k must be less than the number of nodes")
    # Compute pairwise distances
    dist = cdist(points, points, metric="euclidean")
    # For each node, find indices of k smallest distances (excluding self)
    # Use argsort on each row
    knn_idx = np.argsort(dist, axis=1)[:, 1 : k + 1]
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in knn_idx[i]:
            adj[i, j] = 1
            adj[j, i] = 1  # ensure symmetry
    return adj

def graph_summary(adj: np.ndarray) -> np.ndarray:
    """Compute basic summary statistics of an adjacency matrix.

    We extract features capturing the distribution of degrees in the graph.

    Parameters
    ----------
    adj : np.ndarray
        Binary adjacency matrix of shape (n_nodes, n_nodes).

    Returns
    -------
    np.ndarray of shape (3,)
        Vector containing (mean degree, max degree, degree variance).
    """
    # Degree of each node (exclude diagonal, which is zero by construction)
    degrees = adj.sum(axis=1)
    mean_deg = degrees.mean() if degrees.size > 0 else 0.0
    max_deg = degrees.max() if degrees.size > 0 else 0.0
    var_deg = degrees.var() if degrees.size > 0 else 0.0
    return np.array([mean_deg, max_deg, var_deg])