"""Readers and a downloader for the public Binance Data archive (data.binance.vision).

USDⓈ-M futures files used here:

    futures/um/daily/aggTrades/{SYM}/{SYM}-aggTrades-{YYYY-MM-DD}.zip
    futures/um/daily/bookDepth/{SYM}/{SYM}-bookDepth-{YYYY-MM-DD}.zip
    futures/um/daily/klines/{SYM}/1m/{SYM}-1m-{YYYY-MM-DD}.zip
    futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{YYYY-MM}.zip

Older files have no header row; newer ones do. The readers handle both.
"""
from __future__ import annotations

import io
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import pro_counterpart

BASE_URL = "https://data.binance.vision/data/futures/um"

AGGTRADES_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
                  "transact_time", "is_buyer_maker"]
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
BOOKDEPTH_COLS = ["timestamp", "percentage", "depth", "notional"]


def _read(path: str | Path, cols: list[str], usecols: list[str] | None = None) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as zf:
            raw = zf.read(zf.namelist()[0])
        src = io.BytesIO(raw)
        first = raw[:200].split(b"\n", 1)[0]
    else:
        src = p
        with open(p, "rb") as fh:
            first = fh.readline()
    has_header = not first[:1].isdigit()
    return pd.read_csv(src, header=0 if has_header else None, names=cols, usecols=usecols)


def _ms_index(s: pd.Series) -> pd.DatetimeIndex:
    """Epoch timestamps in milliseconds (13 digits) or microseconds (16 digits; Binance moved some
    archive files to microseconds in 2025), decided per value, so a file or a join of files that
    mixes the two still lands on the right times."""
    v = s.to_numpy(dtype="int64")
    ns = np.where(v > 10**14, v * 1_000, v * 1_000_000)
    return pd.to_datetime(ns, unit="ns", utc=True)


@pro_counterpart("Deptho Pro: deptho_pro.sync joins months of these daily files into one memory-mapped dataset")
def read_aggtrades(path) -> pd.DataFrame:
    """Tick trades. ``side`` is the aggressor: +1 buyer-initiated, -1 seller-initiated."""
    from .guard import record_daily_load

    t0 = time.perf_counter()
    df = _read(path, AGGTRADES_COLS, ["price", "quantity", "transact_time", "is_buyer_maker"])
    record_daily_load(path, time.perf_counter() - t0)
    maker = df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"])
    out = pd.DataFrame({
        "price": df["price"].astype("float64").to_numpy(),
        "qty": df["quantity"].astype("float64").to_numpy(),
        "side": np.where(maker, -1, 1).astype("int8"),
    }, index=_ms_index(df["transact_time"]))
    out.index.name = "ts"
    return out


TARDIS_TRADES_COLS = ["exchange", "symbol", "timestamp", "local_timestamp", "id", "side", "price", "amount"]


def read_tardis_trades(path) -> pd.DataFrame:
    """Tick trades from a Tardis.dev ``trades`` CSV (``.csv`` or ``.csv.gz``), in the same shape
    as :func:`read_aggtrades`: ``price``, ``qty``, ``side`` (aggressor, +1 buy / -1 sell),
    UTC DatetimeIndex from the exchange ``timestamp`` (microseconds).

    The file is yours: Deptho reads it locally and never downloads from Tardis."""
    df = pd.read_csv(path, usecols=["timestamp", "side", "price", "amount"])
    missing = {"timestamp", "side", "price", "amount"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: not a Tardis trades file (missing {sorted(missing)})")
    side = df["side"].astype(str).str.lower()
    out = pd.DataFrame({
        "price": df["price"].astype("float64").to_numpy(),
        "qty": df["amount"].astype("float64").to_numpy(),
        "side": np.where(side == "sell", -1, 1).astype("int8"),
    }, index=pd.to_datetime(df["timestamp"].astype("int64"), unit="us", utc=True))
    if not side.isin(["buy", "sell"]).all():
        raise ValueError(f"{path}: side must be 'buy' or 'sell' (the aggressor); found {sorted(set(side) - {'buy', 'sell'})[:3]}")
    out.index.name = "ts"
    return out.sort_index(kind="stable")


def read_klines(path) -> pd.DataFrame:
    df = _read(path, KLINE_COLS)
    out = df[["open", "high", "low", "close", "volume"]].astype("float64")
    out.index = _ms_index(df["open_time"])
    out.index.name = "ts"
    return out


def read_funding(path) -> pd.Series:
    """Funding rate per event (fraction, e.g. 0.0001 = 1 bp), positive = longs pay."""
    df = _read(path, FUNDING_COLS)
    s = pd.Series(df["last_funding_rate"].astype("float64").to_numpy(), index=_ms_index(df["calc_time"]),
                  name="funding")
    s.index.name = "ts"
    return s


def read_bookdepth(path) -> pd.DataFrame:
    """Snapshots of cumulative resting notional within +/-N% of mid (wide format by percentage)."""
    df = _read(path, BOOKDEPTH_COLS)
    if pd.api.types.is_numeric_dtype(df["timestamp"]):       # epoch ms or us, like the other files
        df["ts"] = _ms_index(df["timestamp"])
    else:
        df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    wide = df.pivot_table(index="ts", columns="percentage", values="notional", aggfunc="last")
    wide.columns = [float(c) for c in wide.columns]
    return wide


def _url(kind: str, symbol: str, day: date, interval: str = "1m") -> str:
    d = day.isoformat()
    if kind == "aggTrades":
        return f"{BASE_URL}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{d}.zip"
    if kind == "bookDepth":
        return f"{BASE_URL}/daily/bookDepth/{symbol}/{symbol}-bookDepth-{d}.zip"
    if kind == "klines":
        return f"{BASE_URL}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{d}.zip"
    if kind == "fundingRate":
        return f"{BASE_URL}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{day:%Y-%m}.zip"
    raise ValueError(kind)


_announced: set[str] = set()


def download(kind: str, symbol: str, day: date, dest: str | Path, interval: str = "1m", log=print) -> Path:
    """Download one archive file (skips if already present). Returns the local path.

    This is the only network call in Deptho. It runs only when you invoke a
    download (``python -m deptho.demo``, ``deptho_pro.sync``), and only
    contacts the exchange's public archive, never a Deptho server.
    """
    url = _url(kind, symbol, day, interval)
    out = Path(dest) / url.rsplit("/", 1)[1]
    if not out.exists():
        host = urllib.parse.urlsplit(url).hostname
        if host not in _announced:
            _announced.add(host)
            log(f"Downloading from {host} (Binance public data archive)")
        out.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "deptho"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out.write_bytes(r.read())
    return out


def days(end: date, n: int) -> list[date]:
    return [end - timedelta(days=i) for i in range(n - 1, -1, -1)]
