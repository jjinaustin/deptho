"""Performance and risk metrics.

Everything here is plain NumPy/pandas so it runs anywhere. Functions take a
``pd.Series`` of simple per-period returns indexed by timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "infer_periods_per_year",
    "equity_curve",
    "drawdown",
    "max_drawdown",
    "sharpe",
    "sortino",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_alpha",
    "summary",
]

_SECONDS_PER_YEAR = 365.0 * 24 * 3600  # crypto trades 24/7


def infer_periods_per_year(index: pd.Index, trading_days: float | None = None) -> float:
    """Infer annualisation factor from the median bar spacing.

    Crypto markets are 24/7, so by default a year is 365 calendar days. Pass
    ``trading_days=252`` for equities (6.5h sessions are handled by the caller
    supplying ``periods_per_year`` directly).
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return 365.0
    step = np.median(np.diff(index.as_unit("ns").asi8)) / 1e9  # seconds
    if step <= 0:
        return 365.0
    if trading_days is not None and step >= 86400 * 0.9:
        return trading_days * 86400 / step
    return _SECONDS_PER_YEAR / step


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1.0 + returns.fillna(0.0)).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    """Underwater curve: fractional distance below the running equity peak (<= 0)."""
    eq = equity_curve(returns)
    peak = np.maximum.accumulate(np.maximum(eq.to_numpy(), 1.0))
    return pd.Series(eq.to_numpy() / peak - 1.0, index=returns.index, name="drawdown")


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown(returns).min()) if len(returns) else 0.0


def sharpe(returns: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    r = returns.dropna() - rf / periods_per_year
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def _downside_dev(r: np.ndarray) -> float:
    neg = np.minimum(r, 0.0)
    return float(np.sqrt(np.mean(neg * neg))) if len(r) else float("nan")


def sortino(returns: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    r = (returns.dropna() - rf / periods_per_year).to_numpy()
    dd = _downside_dev(r)
    if not np.isfinite(dd) or dd == 0:
        return float("nan")
    return float(r.mean() / dd * np.sqrt(periods_per_year))


def rolling_sharpe(returns: pd.Series, window: int, periods_per_year: float) -> pd.Series:
    roll = returns.rolling(window, min_periods=max(2, window // 2))
    sd = roll.std().replace(0.0, np.nan)
    return (roll.mean() / sd * np.sqrt(periods_per_year)).rename("rolling_sharpe")


def rolling_sortino(returns: pd.Series, window: int, periods_per_year: float) -> pd.Series:
    neg_sq = np.minimum(returns, 0.0) ** 2
    mp = max(2, window // 2)
    dd = np.sqrt(neg_sq.rolling(window, min_periods=mp).mean()).replace(0.0, np.nan)
    mean = returns.rolling(window, min_periods=mp).mean()
    return (mean / dd * np.sqrt(periods_per_year)).rename("rolling_sortino")


def rolling_alpha(
    returns: pd.Series, benchmark: pd.Series, window: int, periods_per_year: float
) -> pd.DataFrame:
    """Rolling OLS of strategy on benchmark: annualised alpha and beta.

    A falling alpha line is the "alpha decay" view: the edge left over after
    stripping out market exposure, and whether it is fading over time.
    """
    df = pd.concat([returns.rename("y"), benchmark.rename("x")], axis=1).dropna()
    mp = max(3, window // 2)
    roll = df.rolling(window, min_periods=mp)
    cov = roll.cov().unstack()[("x", "y")]
    var_x = df["x"].rolling(window, min_periods=mp).var().replace(0.0, np.nan)
    beta = cov / var_x
    alpha = df["y"].rolling(window, min_periods=mp).mean() - beta * df["x"].rolling(
        window, min_periods=mp
    ).mean()
    return pd.DataFrame({"alpha": alpha * periods_per_year, "beta": beta})


def summary(returns: pd.Series, periods_per_year: float) -> dict[str, float]:
    r = returns.dropna()
    eq = equity_curve(r)
    total = float(eq.iloc[-1] - 1.0) if len(eq) else 0.0
    years = len(r) / periods_per_year if periods_per_year else float("nan")
    cagr = float((1 + total) ** (1 / years) - 1) if years and years > 0 and total > -1 else float("nan")
    mdd = max_drawdown(r)
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else float("nan")
    hit = float((r > 0).sum() / max(1, (r != 0).sum()))
    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe(r, periods_per_year),
        "sortino": sortino(r, periods_per_year),
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 and np.isfinite(cagr) else float("nan"),
        "hit_rate": hit,
        "periods": float(len(r)),
    }
