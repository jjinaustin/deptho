"""BTCUSDT perpetual: what do funding and execution actually cost a trend strategy?

    python -m deptho.demo                       # fetch from data.binance.vision, run, open report
    python -m deptho.demo --symbol ETHUSDT --day 2026-09-30
    python -m deptho.demo --data DIR            # use archive files already on disk
    python -m deptho.demo --synthetic           # offline: schema-identical fake data

Data is fetched by your machine straight from Binance's public archive and
cached in ~/.cache/deptho/binance. Nothing is sent anywhere else.

Inputs (Binance USD-M futures archive format):
  *-1m-*.zip           1-minute klines        -> strategy + PnL
  *-fundingRate-*.zip  8-hourly funding rates -> funding drag
  *-aggTrades-*.zip    one day of tick trades -> execution replay
  *-bookDepth-*.zip    one day of depth bands -> trade-print replay + depth-band cost
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import webbrowser
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import costs
from . import datasets as bn
from .dashboard import Report, tearsheet


def default_day() -> date:
    """Last day of the previous month: every monthly funding file exists by then."""
    today = datetime.now(timezone.utc).date()
    return today.replace(day=1) - timedelta(days=1)


def cache_dir(symbol: str) -> Path:
    base = os.environ.get("DEPTHO_DATA") or Path.home() / ".cache" / "deptho" / "binance"
    return Path(base) / symbol


def fetch(symbol: str, day: date, kline_days: int, dest: Path, log=print) -> Path:
    """Download the archive files the demo needs. Tick/book files are optional."""
    span = bn.days(day, kline_days)
    log(f"fetching {symbol}: {kline_days} days of 1m klines + funding, tick day {day} -> {dest}")
    for i, d in enumerate(span, 1):
        bn.download("klines", symbol, d, dest)
        if i % 15 == 0 or i == len(span):
            log(f"  klines {i}/{len(span)}")
    for month in sorted({d.replace(day=1) for d in span}):
        bn.download("fundingRate", symbol, month, dest)
    for kind in ("aggTrades", "bookDepth"):
        try:
            bn.download(kind, symbol, day, dest)
            log(f"  {kind} {day}")
        except urllib.error.HTTPError as e:
            log(f"  {kind} {day}: not available ({e.code}); continuing without it")
    return dest


def load_dir(data: Path):
    def cat(pattern, reader):
        parts = [reader(p) for p in sorted(data.glob(pattern))]
        return pd.concat(parts).sort_index() if parts else None

    klines = cat("*-1m-*.zip", bn.read_klines)
    funding = cat("*-fundingRate-*.zip", bn.read_funding)
    if klines is None or funding is None:
        raise FileNotFoundError(f"{data}: need *-1m-*.zip klines and *-fundingRate-*.zip files")
    return klines, funding, cat("*-aggTrades-*.zip", bn.read_aggtrades), cat("*-bookDepth-*.zip", bn.read_bookdepth)


def strategy(close: pd.Series, fast: int = 12, slow: int = 48, target_vol: float = 0.6) -> pd.Series:
    """Hourly EMA trend, sized to a target annualised volatility, capped at 2x."""
    # Right-labelled bars: the value stamped 11:00 is the 11:00 close, so the
    # forward-fill below only uses information available at each timestamp.
    h = close.resample("1h", label="right", closed="right").last().dropna()
    signal = np.sign(h.ewm(span=fast).mean() - h.ewm(span=slow).mean())
    realised = h.pct_change().rolling(72).std() * np.sqrt(24 * 365)
    pos = (signal * (target_vol / realised)).clip(-2, 2).fillna(0.0)
    return pos.reindex(close.index, method="ffill").fillna(0.0)


def execution_study(trades: pd.DataFrame, depth: pd.DataFrame | None, n: int = 600, seed: int = 1) -> pd.DataFrame:
    """Replay market orders of 0.05-25 units at random times through the tick day."""
    rng = np.random.default_rng(seed)
    lo, hi = trades.index[0].value, trades.index[-1].value
    when = pd.to_datetime(np.sort(rng.integers(lo, hi - 600 * 10**9, n)), utc=True)
    orders = pd.DataFrame({
        "side": rng.choice([1, -1], n),
        "qty": np.round(np.exp(rng.uniform(np.log(0.05), np.log(25), n)), 3),
    }, index=when)
    return costs.simulate_taker_fills(trades, orders, depth)


def run(data: Path, symbol: str = "BTCUSDT", fee_bps: float = 5.0) -> Report:
    klines, funding, trades, depth = load_dir(data)
    # Klines are stamped at their open time; a close is only known at open + 1m.
    close_1m = klines["close"].shift(freq="1min")
    close = close_1m.resample("5min", label="right", closed="right").last().dropna()
    pos = strategy(close)

    fills = execution_study(trades, depth) if trades is not None and len(trades) else None
    # Charge the strategy the median execution cost from the trade-print replay.
    slip = float(costs.slippage_bps(fills)["slippage_bps"].median()) if fills is not None and len(fills) else 1.0

    pnl = costs.pnl_from_positions(close, pos, funding=funding, fee_bps=fee_bps, slippage_bps=max(slip, 0.0))
    from ._synthetic import SYNTHETIC_MARKER, SYNTHETIC_TAG

    tag = SYNTHETIC_TAG if (data / SYNTHETIC_MARKER).exists() else None
    return tearsheet(pnl=pnl, benchmark=close.pct_change(), fills=fills, data_tag=tag,
                     fills_label="Simulated execution cost (trade-print replay)",
                     title=f"{symbol} perp - EMA trend, {len(klines) // 1440} days")


def main(argv=None, default_data: Path | None = None) -> Path:
    ap = argparse.ArgumentParser(prog="python -m deptho.demo", description=__doc__.split("\n")[0])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--day", type=date.fromisoformat, default=None, help="tick-data day (default: end of last month)")
    ap.add_argument("--days", type=int, default=90, help="days of 1m klines")
    ap.add_argument("--data", type=Path, default=default_data, help="directory of archive files (skip download)")
    ap.add_argument("--synthetic", action="store_true", help="generate fake data in Binance's schemas (offline)")
    ap.add_argument("--fee-bps", type=float, default=5.0, help="taker fee; Binance VIP0 USD-M = 5 bps")
    ap.add_argument("--out", type=Path, default=Path("deptho_report.html"))
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)

    day = args.day or default_day()
    if args.data is not None and any(args.data.glob("*-1m-*.zip")):
        data = args.data
    elif args.synthetic:
        from ._synthetic import make

        data = cache_dir(args.symbol + "-synthetic")
        if not any(data.glob("*-1m-*.zip")):
            make(data, args.symbol, day, args.days)
    else:
        data = cache_dir(args.symbol)
        try:
            fetch(args.symbol, day, args.days, data)
        except urllib.error.URLError as e:
            sys.exit(f"could not reach data.binance.vision ({e}). Try --synthetic for an offline demo.")

    report = run(data, args.symbol, args.fee_bps)
    print(report.table().to_string(float_format=lambda x: f"{x:,.4f}"))
    path = report.save(args.out).resolve()
    print(f"\nreport -> {path}")
    if not args.no_open:
        webbrowser.open(path.as_uri())
    return path


if __name__ == "__main__":
    main()
