"""Deptho: one-line interactive tearsheets for systematic strategies.

    import deptho as dp
    pnl = dp.pnl_from_positions(prices, positions, funding=funding, fee_bps=4)
    dp.tearsheet(pnl=pnl, benchmark=prices.pct_change()).show()
"""
from . import metrics
from .config import config
from . import datasets
from .costs import funding_cost, pnl_from_positions, simulate_taker_fills, slippage_bps
from .io import MemoryWallError, load

__version__ = "0.1.1"


def __getattr__(name):
    # The dashboard pulls in plotly; load it on first use so data paths
    # (load, costs, the Pro engine's replay) don't pay for a charting library.
    if name in ("tearsheet", "Report", "dashboard"):
        import importlib

        dashboard = importlib.import_module(".dashboard", __name__)
        return dashboard if name == "dashboard" else getattr(dashboard, name)
    raise AttributeError(f"module 'deptho' has no attribute {name!r}")
__all__ = [
    "config", "load", "MemoryWallError", "metrics", "tearsheet", "Report",
    "pnl_from_positions", "funding_cost", "slippage_bps", "simulate_taker_fills", "datasets",
]
