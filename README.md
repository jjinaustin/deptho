# deptho

Deptho is the tick-level reality check for any backtest. Bring signals from any backtester or custom code, and Deptho replays them against months of real trades from Binance's public archive, entirely on a local machine. The free library shows funding and fees by position; Deptho Pro handles tick data bigger than memory.

```bash
pip install deptho
python -m deptho.demo               # 90 days of BTCUSDT perp data from Binance's archive, a sample strategy, a report
python -m deptho.demo --synthetic   # same, offline
```

## A tearsheet that shows what costs took

```python
import deptho as dp

pnl = dp.pnl_from_positions(prices, positions, funding=funding_rates, fee_bps=4)
dp.tearsheet(pnl=pnl, benchmark=prices.pct_change(), title="BTC perp momentum").show()
```

- Net vs gross equity, so you see exactly what costs took
- Cumulative cost drag broken out into funding, fees and slippage
- Funding charged to the position actually held at each funding time
- Underwater (drawdown) curve, rolling Sharpe and Sortino, rolling alpha against a benchmark
- Slippage distribution and slippage vs order size (pass `fills=`)
- `report.save("report.html")` writes one self-contained HTML file that works offline
- `report.fill_gap_card("card")` writes one shareable image (SVG, plus PNG with `pip install "deptho[png]"`): the bar-close return, the return with slippage measured by replaying the orders, and the gap between them in points

Cost attribution is lookahead-safe by default: a position decided at a bar's close earns from the next bar.

## Inputs

| argument | shape |
|---|---|
| `prices` | `pd.Series` of mark/close prices, DatetimeIndex |
| `positions` | `pd.Series` exposure as a fraction of equity (1.0 = 1x long), decided at each bar's close |
| `funding` | `pd.Series` of funding rates at funding timestamps (positive = longs pay) |
| `fills` | `pd.DataFrame` with `side`, `price`, `ref_price`, optionally `qty`, `depth` |

Tick trades come from Binance's public archive (`dp.datasets.read_aggtrades`) or from Tardis.dev `trades` files you've downloaded (`dp.datasets.read_tardis_trades`), in the same shape either way.

## Large files

`dp.load(path)` reads CSV or Parquet with pandas. It first measures a small sample, and if the full file won't fit in memory it raises `dp.MemoryWallError` with an estimate, instead of letting pandas exhaust RAM and kill your kernel.

## Privacy

Deptho runs entirely on your machine and collects no telemetry. The only network call is a data download you start yourself, which goes to Binance's public archive and prints the host first.

## Deptho Pro

Pro is for when the free tools run out: replaying orders against every trade across months of tick data.

- Sync months of Binance USD-M perpetual trades into one local, memory-mapped dataset that grows a day at a time and repairs itself after an interruption.
- Replay a table of orders (time, side, quantity) from any backtester against every trade print, lookahead-safe by default. Fills are modeled from trade prints; order book depth is coming.
- Sweep many order sets over the same dataset in tens of MB of memory.
- License checked offline. Nothing is uploaded.

Benchmarks, method and pricing: [deptho.io](https://deptho.io). Turn off in-library Pro mentions with `dp.config.upgrade_prompts = False` or `DEPTHO_NO_PROMPTS=1`.

## Links

- Website and benchmarks: https://deptho.io
- Guides: https://deptho.io/fix/
- Deptho Pro: https://deptho.io/pro.html

License: Apache-2.0. The card's PNG uses the bundled Space Grotesk and JetBrains Mono fonts (SIL Open Font License 1.1, in `deptho/fonts/`). Market data comes from Binance's public archive (data.binance.vision); Deptho is not affiliated with Binance.
