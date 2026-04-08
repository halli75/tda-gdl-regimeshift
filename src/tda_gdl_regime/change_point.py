"""
change_point.py
================

This module implements a simple online change‑point detection mechanism based on
the cumulative sum (CUSUM) of log‑likelihood ratios.  Given a stream of
predicted probabilities for a target regime (e.g. ``turbulent``), we
cumulatively accumulate evidence in favour of the regime and signal an alarm
whenever the accumulated sum crosses a predefined threshold.  The drift
parameter ``k`` controls sensitivity: higher ``k`` results in fewer false
alarms but longer detection delays.

Functions:
    cusum_detection(probs, target_idx, k, h): return list of alarm indices.
"""

from __future__ import annotations

import numpy as np

def cusum_detection(probs: np.ndarray, target_idx: int = 2, k: float = 0.0, h: float = 5.0) -> list[int]:
    """Detect change points using a probability CUSUM procedure.

    Parameters
    ----------
    probs : np.ndarray of shape (n_samples, n_classes)
        Sequence of predicted class probabilities.  ``probs[t, target_idx]``
        should be the probability of the ``target_idx`` regime at time ``t``.
    target_idx : int, optional
        Index of the target regime (default 2 corresponds to ``turbulent`` in
        our 3‑class setting: calm=0, volatile=1, turbulent=2).
    k : float, optional
        Reference value (drift) for the CUSUM statistic.  A positive ``k``
        reduces sensitivity by penalising small log‑likelihood ratios.  Set
        ``k`` to the expected mean of the log‑likelihood ratio under the null
        hypothesis (no change).
    h : float, optional
        Threshold at which to signal an alarm.  Larger values lead to fewer
        detections.

    Returns
    -------
    list[int]
        Indices ``t`` at which change points are detected.
    """
    alarms: list[int] = []
    s = 0.0
    for t, p_vec in enumerate(probs):
        p = p_vec[target_idx]
        # Avoid divide by zero; clamp probabilities
        p = np.clip(p, 1e-8, 1 - 1e-8)
        llr = np.log(p / (1.0 - p))
        s = max(0.0, s + llr - k)
        if s > h:
            alarms.append(t)
            s = 0.0  # reset after detection
    return alarms