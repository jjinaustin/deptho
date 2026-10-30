"""Cost attribution: funding/carry drag, fees and execution slippage.

``funding`` is deliberately generic. For crypto perpetuals it is the periodic
funding rate; for equities it is a borrow fee or financing rate. The sign
convention is the exchange one: a positive rate means longs pay shorts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import pro_counterpart

__all__ = ["pnl_from_positions", "funding_cost", "slippage_bps", "simulate_taker_fills", "apply_depth_bands",
           "FILL_MODELS"]


def _ns(index: pd.Index) -> np.ndarray:
    """Timestamps as int64 nanoseconds, whatever unit/timezone pandas chose."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.as_unit("ns").asi8


def _as_series(x, name: str) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.rename(name)
    if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
        return x.iloc[:, 0].rename(name)
    raise TypeError(f"{name} must be a pandas Series indexed by timestamp")


def funding_cost(positions: pd.Series, funding: pd.Series) -> pd.Series:
    """Per-bar funding cost as a fraction of equity (positive = you paid).

    Each funding event is charged against the position held at that instant
    (the last bar at or before the event) and booked on that bar.
    """
    positions = _as_series(positions, "position").sort_index()
    funding = _as_series(funding, "funding").sort_index().dropna()
    out = pd.Series(0.0, index=positions.index, name="funding")
    if funding.empty or positions.empty:
        return out
    funding = funding[(funding.index >= positions.index[0]) & (funding.index <= positions.index[-1])]
    if funding.empty:
        return out
    loc = np.searchsorted(_ns(positions.index), _ns(funding.index), side="right") - 1
    held = positions.to_numpy()[loc]
    np.add.at(out.values, loc, held * funding.to_numpy())
    return out


def pnl_from_positions(
    prices: pd.Series,
    positions: pd.Series,
    funding: pd.Series | None = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Decompose strategy returns into gross, fees, slippage and funding.

    ``positions`` is the exposure as a fraction of equity (1.0 = 1x long,
    -2.0 = 2x short) decided at the close of each bar and held over the next.
    Returns a DataFrame of per-bar fractional returns; ``net`` is what you keep.
    """
    prices = _as_series(prices, "price").sort_index()
    positions = _as_series(positions, "position").reindex(prices.index).ffill().fillna(0.0)
    held = positions.shift(1).fillna(0.0)
    gross = held * prices.pct_change().fillna(0.0)
    turnover = positions.diff().abs().fillna(positions.abs())
    fees = turnover * fee_bps / 1e4
    slip = turnover * slippage_bps / 1e4
    fund = funding_cost(held, funding) if funding is not None else pd.Series(0.0, index=prices.index)
    out = pd.DataFrame(
        {
            "position": held,
            "gross": gross,
            "fees": fees,
            "slippage": slip,
            "funding": fund,
        }
    )
    out["net"] = out["gross"] - out["fees"] - out["slippage"] - out["funding"]
    return out


FILL_MODELS = ("auto", "trade_prints", "depth_bands", "book")


def resolve_fill_model(fill_model: str, depth) -> str:
    """``auto`` keeps the original behaviour: trade prints, plus depth bands when given."""
    if fill_model not in FILL_MODELS:
        raise ValueError(f"fill_model must be one of {FILL_MODELS}, got {fill_model!r}")
    if fill_model == "book":
        raise NotImplementedError(
            "fill_model='book' (level-by-level fills from a reconstructed L2 book) needs "
            "deptho_pro.BookBuilder, which is coming next. Use 'trade_prints' or 'depth_bands'.")
    if fill_model == "auto":
        return "depth_bands" if depth is not None and len(depth) else "trade_prints"
    if fill_model == "depth_bands" and (depth is None or not len(depth)):
        raise ValueError("fill_model='depth_bands' needs depth= (deptho.datasets.read_bookdepth)")
    return fill_model


@pro_counterpart("Deptho Pro: deptho_pro.sync(...).replay(orders) replays against months of trades, memory-mapped")
def simulate_taker_fills(
    trades: pd.DataFrame,
    orders: pd.DataFrame,
    depth: pd.DataFrame | None = None,
    fill_model: str = "auto",
) -> pd.DataFrame:
    """Trade-print execution replay: estimate what market orders would have paid.

    ``trades``: tick trades with ``price``, ``qty``, ``side`` (aggressor +1/-1),
    DatetimeIndex (see :func:`deptho.datasets.read_aggtrades`).
    ``orders``: DatetimeIndex, ``side`` (+1/-1) and ``qty`` in base units.

    Fill models:

    * ``"trade_prints"``: each order fills at the VWAP of the following trade
      prints whose aggressor was on the same side, until its quantity is
      covered. It captures the spread and short-term drift, but **does not
      model order-book depth or queue position**, and assumes your order would
      have got the prices other takers got.
    * ``"depth_bands"``: trade prints, plus a cost from walking Binance's
      aggregated depth bands (``depth``, from
      :func:`deptho.datasets.read_bookdepth`: cumulative resting notional
      within +/-0.2%..5% of mid, linear in between). Coarse: not level by
      level, and no queue position.
    * ``"auto"`` (default): ``depth_bands`` if ``depth`` is given, else ``trade_prints``.
    * ``"book"``: reserved for level-by-level fills from a reconstructed order book (not available yet).

    ``ref_price`` is the last print before the order. With depth bands,
    ``depth`` is the resting size (base units) inside the tightest band and
    ``depth_band_bps`` that component of the cost.
    """
    model = resolve_fill_model(fill_model, depth)
    trades = trades.sort_index()
    t_ns = _ns(trades.index)
    o_ns = _ns(orders.index)
    prev = np.searchsorted(t_ns, o_ns, side="left") - 1
    ref = trades["price"].to_numpy()[np.clip(prev, 0, None)]
    ref = np.where(prev >= 0, ref, np.nan)

    fill = np.full(len(orders), np.nan)
    sides = orders["side"].to_numpy()
    qtys = orders["qty"].to_numpy(dtype=float)
    for s in (1, -1):
        m = trades["side"].to_numpy() == s
        ts, px, q = t_ns[m], trades["price"].to_numpy()[m], trades["qty"].to_numpy()[m]
        if len(ts) == 0:
            continue
        cq = np.r_[0.0, np.cumsum(q)]
        cpq = np.r_[0.0, np.cumsum(px * q)]
        sel = np.flatnonzero(sides == s)
        start = np.searchsorted(ts, o_ns[sel], side="left")
        end = np.searchsorted(cq, cq[start] + qtys[sel], side="left")
        ok = end < len(cq)
        end = np.minimum(end, len(cq) - 1)
        filled = cq[end] - cq[start]
        vwap = np.where(filled > 0, (cpq[end] - cpq[start]) / np.where(filled > 0, filled, 1), np.nan)
        fill[sel] = np.where(ok, vwap, np.nan)

    out = pd.DataFrame({"side": sides, "qty": qtys, "price": fill, "ref_price": ref}, index=orders.index)
    return apply_depth_bands(out, depth if model == "depth_bands" else None)


def apply_depth_bands(fills: pd.DataFrame, depth: pd.DataFrame | None) -> pd.DataFrame:
    """Add the depth-band walk cost to trade-print fills (``side``, ``qty``, ``price``,
    ``ref_price``, DatetimeIndex) and drop orders that could not be filled."""
    out = fills.copy()
    if depth is not None and len(depth):
        o_ns = _ns(out.index)
        sides = out["side"].to_numpy()
        qtys = out["qty"].to_numpy(dtype=float)
        ref = out["ref_price"].to_numpy(dtype=float)
        snap = np.clip(np.searchsorted(_ns(depth.index), o_ns, side="right") - 1, 0, None)
        band = np.full(len(out), np.nan)
        tight = np.full(len(out), np.nan)
        for s in (1, -1):
            lvls = sorted(c for c in depth.columns if c * s > 0 and np.isfinite(c))
            lvls = lvls if s > 0 else lvls[::-1]
            if not lvls:
                continue
            pct = np.r_[0.0, np.abs(lvls)] / 100.0                      # price offset knots
            cum = np.c_[np.zeros(len(depth)), depth[lvls].to_numpy()]  # cumulative notional
            for i in np.flatnonzero(sides == s):
                c = np.maximum.accumulate(np.nan_to_num(cum[snap[i]]))
                band[i] = _walk_bps(qtys[i] * ref[i], c, pct)
                tight[i] = c[1] / ref[i]
        out["depth_band_bps"] = band
        out["depth"] = tight
        out["price"] = out["price"] * (1 + out["side"] * np.nan_to_num(band) / 1e4)
    return out.dropna(subset=["price", "ref_price"])


def _walk_bps(notional: float, cum: np.ndarray, pct: np.ndarray) -> float:
    """Average price offset (bps) to take ``notional`` from a cumulative depth-band curve."""
    if not np.isfinite(notional) or notional <= 0 or cum[-1] <= 0:
        return np.nan
    n = min(notional, cum[-1])
    # Marginal offset as a function of cumulative notional is the inverse curve;
    # integrate it with trapezoids up to n.
    k = np.searchsorted(cum, n, side="left")
    q = np.r_[cum[:k], n]
    p = np.r_[pct[:k], np.interp(n, cum, pct)]
    avg = np.trapezoid(p, q) / n if hasattr(np, "trapezoid") else np.trapz(p, q) / n
    return float(avg * 1e4)


_SIDE_MAP = {"buy": 1, "b": 1, "long": 1, "bid": 1, "sell": -1, "s": -1, "short": -1, "ask": -1}


def slippage_bps(fills: pd.DataFrame) -> pd.DataFrame:
    """Implementation shortfall per fill in basis points (positive = cost).

    Required columns: ``side`` (+1/-1 or buy/sell), ``price`` (fill price),
    ``ref_price`` (arrival price or mid at decision time). Optional ``qty`` and
    ``depth`` (resting size available on the touched side when the order was
    sent) add a participation column, ``qty / depth``.
    """
    missing = {"side", "price", "ref_price"} - set(fills.columns)
    if missing:
        raise ValueError(f"fills is missing columns: {sorted(missing)}")
    side = fills["side"]
    if not pd.api.types.is_numeric_dtype(side):
        side = side.astype(str).str.lower().map(_SIDE_MAP)
    side = side.astype(float)
    out = fills.copy()
    out["side"] = side
    out["slippage_bps"] = side * (fills["price"] - fills["ref_price"]) / fills["ref_price"] * 1e4
    if {"qty", "depth"} <= set(fills.columns):
        out["participation"] = fills["qty"].abs() / fills["depth"].replace(0, np.nan)
    return out
