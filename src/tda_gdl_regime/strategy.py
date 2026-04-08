"""
strategy.py
===========

This module defines a simple regime‑adaptive micro‑trading strategy.  The
strategy examines predicted regimes at each time step and chooses between
mean‑reversion, momentum or flat exposure.  Execution cost is modelled as a
proportional transaction cost applied to changes in position.  This is not a
production trading system but a toy example illustrating how to integrate
machine learning predictions into trading logic.

Functions:
    regime_based_strategy(prices, regimes, costs): simulate trading returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

def regime_based_strategy(prices: np.ndarray, regimes: np.ndarray, costs: float = 0.001) -> pd.DataFrame:
    """Run a simple trading strategy based on regime classification.

    For each time step, the strategy selects a position according to the
    predicted regime:

    * Calm (0): mean‑reversion strategy.  Compute z‑score of price deviation
      from a rolling mean (window=20).  If z > 1, go short; if z < -1, go
      long; else remain neutral.
    * Volatile (1): stay flat (zero position).
    * Turbulent (2): momentum strategy.  If price_t > price_{t-1}, go long;
      else go short.

    Transaction costs are proportional to the absolute change in position.

    Parameters
    ----------
    prices : np.ndarray of shape (n_samples,)
        Price series aligned with regime predictions.
    regimes : np.ndarray of shape (n_samples,)
        Integer regime labels (0: calm, 1: volatile, 2: turbulent).
    costs : float, optional
        Transaction cost (in decimal units) per unit of position change.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'price', 'regime', 'position', 'pnl',
        'cum_pnl'.  'pnl' is the per‑period profit after costs, and
        'cum_pnl' is the cumulative profit.
    """
    n = len(prices)
    prices = np.asarray(prices)
    regimes = np.asarray(regimes)
    positions = np.zeros(n)
    pnl = np.zeros(n)
    # Precompute rolling mean and std for z‑score
    window = 20
    # To avoid NaNs at the beginning, extend by replicating first value
    extended = np.concatenate([np.full(window - 1, prices[0]), prices])
    rolling_mean = pd.Series(extended).rolling(window=window).mean().to_numpy()[window - 1 :]
    rolling_std = pd.Series(extended).rolling(window=window).std().to_numpy()[window - 1 :]
    for t in range(1, n):
        reg = regimes[t]
        prev_pos = positions[t - 1]
        pos = prev_pos
        if reg == 0:  # calm: mean‑reversion
            # compute z‑score of deviation
            if rolling_std[t] > 0:
                z = (prices[t] - rolling_mean[t]) / rolling_std[t]
            else:
                z = 0.0
            if z > 1.0:
                pos = -1.0
            elif z < -1.0:
                pos = 1.0
            else:
                pos = 0.0
        elif reg == 1:  # volatile: flat
            pos = 0.0
        elif reg == 2:  # turbulent: momentum
            if prices[t] > prices[t - 1]:
                pos = 1.0
            elif prices[t] < prices[t - 1]:
                pos = -1.0
            else:
                pos = 0.0
        else:
            pos = 0.0
        positions[t] = pos
        # Calculate return from t-1 to t
        ret = (prices[t] - prices[t - 1]) / prices[t - 1]
        # Position change cost
        trade_cost = costs * abs(pos - prev_pos)
        pnl[t] = prev_pos * ret - trade_cost
    cum_pnl = np.cumsum(pnl)
    return pd.DataFrame({
        'price': prices,
        'regime': regimes,
        'position': positions,
        'pnl': pnl,
        'cum_pnl': cum_pnl,
    })