import re
import numpy as np
import pandas as pd
import pytest

import deptho as dp
from deptho import costs, guard, metrics


def _idx(n, freq="1h"):
    return pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")


def test_drawdown_and_summary():
    r = pd.Series([0.1, -0.5, 0.2, 0.0], index=_idx(4, "1D"))
    dd = metrics.drawdown(r)
    assert dd.iloc[0] == 0
    assert dd.iloc[1] == pytest.approx(-0.5)
    assert dd.iloc[2] == pytest.approx(1.1 * 0.5 * 1.2 / 1.1 - 1)
    s = metrics.summary(r, 365)
    assert s["max_drawdown"] == pytest.approx(-0.5)
    assert s["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)


def test_periods_per_year_inference():
    assert metrics.infer_periods_per_year(_idx(10, "1D")) == pytest.approx(365)
    assert metrics.infer_periods_per_year(_idx(10, "5min")) == pytest.approx(365 * 288)
    # pandas>=2 non-ns units must not change the answer
    ms = pd.DatetimeIndex(_idx(10, "1h")).as_unit("ms")
    assert metrics.infer_periods_per_year(ms) == pytest.approx(365 * 24)


def test_rolling_matches_full_sample_on_last_window():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 500), index=_idx(500))
    rs = metrics.rolling_sharpe(r, 100, 8760)
    assert rs.iloc[-1] == pytest.approx(metrics.sharpe(r.iloc[-100:], 8760))
    so = metrics.rolling_sortino(r, 100, 8760)
    assert so.iloc[-1] == pytest.approx(metrics.sortino(r.iloc[-100:], 8760))


def test_rolling_alpha_recovers_known_beta():
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0, 0.01, 2000), index=_idx(2000))
    y = 0.0002 + 1.5 * b + rng.normal(0, 1e-4, 2000)
    ra = metrics.rolling_alpha(y, b, 500, 8760)
    assert ra["beta"].iloc[-1] == pytest.approx(1.5, abs=0.01)
    assert ra["alpha"].iloc[-1] == pytest.approx(0.0002 * 8760, rel=0.05)


def test_funding_charged_to_position_held_at_event():
    idx = _idx(24)
    pos = pd.Series(0.0, index=idx)
    pos.iloc[5:] = 2.0  # 2x long from 05:00
    fund = pd.Series([0.0001, 0.0002, -0.0001], index=pd.DatetimeIndex(
        ["2026-01-01 00:00", "2026-01-01 08:00", "2026-01-01 16:30"], tz="UTC").as_unit("ms"))
    fc = costs.funding_cost(pos, fund)
    assert fc.iloc[0] == 0.0                       # flat at 00:00
    assert fc.iloc[8] == pytest.approx(2 * 0.0002)  # long pays positive funding
    assert fc.iloc[16] == pytest.approx(2 * -0.0001)  # 16:30 booked on 16:00 bar; long receives
    assert fc.sum() == pytest.approx(2 * 0.0001)


def test_pnl_decomposition_is_additive_and_lagged():
    idx = _idx(5)
    px = pd.Series([100, 110, 99, 99, 108.9], index=idx, dtype=float)
    pos = pd.Series([1, 1, -1, -1, 0], index=idx, dtype=float)
    pnl = costs.pnl_from_positions(px, pos, fee_bps=10)
    assert pnl["gross"].iloc[1] == pytest.approx(0.10)    # held long over 100->110
    assert pnl["gross"].iloc[2] == pytest.approx(-0.10)   # still long (decided at bar 1) over 110->99
    assert pnl["gross"].iloc[4] == pytest.approx(-0.10)   # short over 99->108.9
    assert pnl["fees"].tolist() == pytest.approx([1e-3, 0, 2e-3, 0, 1e-3])
    np.testing.assert_allclose(pnl["net"], pnl["gross"] - pnl["fees"] - pnl["slippage"] - pnl["funding"])


def test_slippage_sign_convention():
    f = pd.DataFrame({"side": ["buy", "sell"], "price": [101.0, 99.0], "ref_price": [100.0, 100.0]})
    s = costs.slippage_bps(f)
    assert s["slippage_bps"].tolist() == pytest.approx([100.0, 100.0])  # both paid 1%


def test_taker_replay_vwap_and_book_walk():
    t = pd.DatetimeIndex(pd.to_datetime([1, 2, 3, 4, 5], unit="s", utc=True))
    trades = pd.DataFrame({"price": [100.0, 100.1, 100.2, 99.9, 100.3],
                           "qty": [1.0, 1.0, 1.0, 1.0, 1.0], "side": [1, 1, 1, -1, 1]}, index=t)
    orders = pd.DataFrame({"side": [1], "qty": [2.0]}, index=pd.to_datetime([1.5], unit="s", utc=True))
    fills = costs.simulate_taker_fills(trades, orders)
    assert fills["ref_price"].iloc[0] == 100.0
    assert fills["price"].iloc[0] == pytest.approx(100.15)  # VWAP of next two buys
    # Depth bands: $10k within +1%, linear from 0 -> taking $5k averages 0.25% = 25bps.
    depth = pd.DataFrame({1.0: [10_000.0], -1.0: [10_000.0]}, index=t[:1])
    orders = pd.DataFrame({"side": [1], "qty": [50.0]}, index=pd.to_datetime([1.5], unit="s", utc=True))
    trades2 = trades.assign(qty=100.0)
    fills = costs.simulate_taker_fills(trades2, orders, depth)
    assert fills["depth_band_bps"].iloc[0] == pytest.approx(25.0)
    tp = costs.simulate_taker_fills(trades2, orders, depth, fill_model="trade_prints")
    assert "depth_band_bps" not in tp and tp["price"].iloc[0] == pytest.approx(100.1)  # next buy print
    with pytest.raises(ValueError):
        costs.simulate_taker_fills(trades2, orders, fill_model="depth_bands")
    with pytest.raises(NotImplementedError, match="BookBuilder"):
        costs.simulate_taker_fills(trades2, orders, fill_model="book")


def test_tearsheet_builds_with_all_panels(tmp_path):
    rng = np.random.default_rng(2)
    idx = _idx(3000, "5min")
    px = pd.Series(60000 * np.exp(np.cumsum(rng.normal(0, 1e-3, 3000))), index=idx)
    pos = np.sign(px.diff(12)).fillna(0)
    fund = pd.Series(1e-4, index=pd.date_range(idx[0], idx[-1], freq="8h"))
    pnl = dp.pnl_from_positions(px, pos, funding=fund, fee_bps=5)
    fills = pd.DataFrame({"side": [1, -1] * 50, "price": 100 + rng.normal(0, 0.1, 100), "ref_price": 100.0,
                          "qty": rng.uniform(1, 5, 100), "depth": 100.0})
    rep = dp.tearsheet(pnl=pnl, benchmark=px.pct_change(), fills=fills, title="t")
    assert {"sharpe", "funding_paid", "slippage_bps_mean"} <= set(rep.metrics)
    assert len(rep.figure.layout.annotations) >= 7
    out = rep.save(tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert "dp-narrow" in html and len(html) < 8_000_000
    assert all(len(tr.x) <= 3000 for tr in rep.figure.data if tr.type == "scatter")
    dd = next(tr for tr in rep.figure.data if tr.name == "Drawdown")
    assert len(dd.x) <= 3000 * 5 / 60 + 2                  # drawn hourly, not per bar


def test_kpi_tiles_short_labels_with_footnote(tmp_path):
    r = pd.Series(np.random.default_rng(4).normal(0, 1e-3, 500), index=_idx(500, "5min"))
    fills = pd.DataFrame({"side": [1, -1] * 5, "price": 100.0, "ref_price": 100.0, "qty": 1.0})
    rep = dp.tearsheet(r, fills=fills, fills_label="Simulated execution cost (trade-print replay)")
    assert all(len(t[0]) <= 18 for t in rep._tiles), [t[0] for t in rep._tiles]
    html = rep._tiles_html()
    assert "Mean replay cost" in html and 'title="Average simulated execution cost' in html
    assert "text-overflow:ellipsis" in html                  # one-line labels keep tiles the same height


def test_returns_only_tearsheet():
    r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 400), index=_idx(400, "1D"))
    rep = dp.tearsheet(r)
    assert np.isfinite(rep.metrics["sharpe"])


def test_guard_measures_and_notifies(tmp_path, capsys, monkeypatch):
    p = tmp_path / "ticks.csv"
    n = 50_000
    pd.DataFrame({"price": np.linspace(1, 2, n), "qty": 1.0, "t": np.arange(n)}).to_csv(p, index=False)
    a = guard.assess(p)
    assert a.est_memory_bytes > 0 and a.sample_rows > 0
    monkeypatch.setattr(dp.config, "large_file_bytes", 1)  # force "large"
    monkeypatch.setattr(dp.config, "upgrade_prompts", True)
    df = dp.load(p)
    assert len(df) == n
    assert "memory bottleneck" in capsys.readouterr().err
    p2 = tmp_path / "other.csv"
    p.rename(p2)
    dp.load(p2)  # same trigger, different file: still once per session
    assert capsys.readouterr().err == ""


def test_guard_respects_opt_out(tmp_path, capsys, monkeypatch):
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": range(10)}).to_csv(p, index=False)
    monkeypatch.setattr(dp.config, "large_file_bytes", 1)
    monkeypatch.setattr(dp.config, "upgrade_prompts", False)
    dp.load(p)
    assert capsys.readouterr().err == ""


def test_memory_wall_raises_instead_of_crashing(tmp_path, monkeypatch):
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": range(1000)}).to_csv(p, index=False)
    monkeypatch.setattr(guard, "available_memory", lambda: 10)
    monkeypatch.setattr(dp.config, "upgrade_prompts", False)
    with pytest.raises(dp.MemoryWallError):
        dp.load(p)
    assert len(dp.load(p, force=True)) == 1000


def test_guard_samples_inside_zip(tmp_path):
    import zipfile

    n = 200_000
    csv = pd.DataFrame({"price": np.linspace(1, 2, n), "qty": 1.0}).to_csv(index=False)
    z = tmp_path / "t.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("t.csv", csv)
    a = guard.assess(z)
    assert a.sample_rows > 0
    # Two float64 columns: ~16 bytes/row in the frame, regardless of zip compression.
    assert n * 16 < a.est_memory_bytes < 4 * n * 16  # incl. parse peak factor


def test_demo_runs_offline_on_synthetic_data(tmp_path, monkeypatch, capsys):
    from deptho import _synthetic, demo

    data = tmp_path / "d"
    _synthetic.make(data, kline_days=10, trades_per_day=20_000)
    out = demo.main(["--data", str(data), "--no-open", "--out", str(tmp_path / "r.html"),
                     "--card", str(tmp_path / "card")])
    assert out.exists()
    assert (tmp_path / "card.svg").exists() or "card   -> not written" in capsys.readouterr().out
    k, f, t, d = demo.load_dir(data)
    assert len(k) == 10 * 1440 and len(f) == 30 and t is not None and d is not None
    # The strategy is charged the mean replay cost, not the median (which hides the tail), and the
    # report says so. Fills costing 1, 1, 1 and 9 bps: mean 3, median 1.
    idx = t.index[[100, 200, 300, 400]]
    fills = pd.DataFrame({"side": [1, 1, -1, 1], "qty": 1.0, "ref_price": 100.0,
                          "price": [100.01, 100.01, 99.99, 100.09]}, index=idx)
    monkeypatch.setattr(demo, "execution_study", lambda trades, depth: fills)
    rep = demo.run(data)
    close = k["close"].shift(freq="1min").resample("5min", label="right", closed="right").last().dropna()
    expect = demo.strategy(close).diff().abs().fillna(0.0)
    assert rep.pnl["slippage"].sum() == pytest.approx(3e-4 * expect.sum(), rel=0.05)
    assert "Slippage (mean replay cost, 3.00 bps)" in rep.to_html(include_plotlyjs=False)
    assert "Mean replay cost" in rep._tiles_html()


def test_demo_strategy_has_no_lookahead():
    from deptho import demo

    rng = np.random.default_rng(5)
    idx = pd.date_range("2026-01-01", periods=5000, freq="5min", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 5000))), index=idx)
    pos = demo.strategy(close)
    # Perturbing a future price must not change any earlier position.
    bumped = close.copy()
    bumped.iloc[3006:] *= 1.5  # mid-hour, so a left-labelled hourly bar would leak it
    pd.testing.assert_series_equal(pos.iloc[:3006], demo.strategy(bumped).iloc[:3006])


@pytest.fixture()
def fresh_prompts(monkeypatch):
    monkeypatch.setattr(guard, "_shown", set())
    monkeypatch.setattr(guard, "_daily", {})
    monkeypatch.setattr(dp.config, "upgrade_prompts", True)
    monkeypatch.setattr(dp.config, "large_file_bytes", 10**12)


def _daily_files(d, n=3):
    from deptho import _synthetic

    paths = []
    for i in range(n):
        day = pd.Timestamp("2026-09-28") + pd.Timedelta(days=i)
        _synthetic.make(d / str(i), tick_day=day.date(), kline_days=1, trades_per_day=2_000, seed=i)
        paths.append(next((d / str(i)).glob("*aggTrades*.zip")))
    return paths


def test_daily_files_prompt_measured_and_once(tmp_path, capsys, fresh_prompts):
    files = _daily_files(tmp_path)
    dp.load(files[0])
    assert capsys.readouterr().err == ""                 # one day: no prompt
    dp.load(files[1])
    err = capsys.readouterr().err
    assert "one growing dataset" in err and "Measured load here:" in err and "for 2 daily BTCUSDT aggTrades" in err
    dp.load(files[2])
    assert capsys.readouterr().err == ""                 # once per session


def test_daily_files_prompt_via_glob_and_reader(tmp_path, capsys, fresh_prompts):
    files = _daily_files(tmp_path)
    for f in files:
        f.rename(tmp_path / f.name)
    df = dp.load(str(tmp_path / "BTCUSDT-aggTrades-*.zip"))
    assert len(df) == 6_000
    assert "3 daily BTCUSDT" in capsys.readouterr().err
    guard._shown.clear()
    guard._daily.clear()
    from deptho import datasets as bn
    for f in sorted(tmp_path.glob("*aggTrades*.zip"))[:2]:
        bn.read_aggtrades(f)
    assert "2 daily BTCUSDT" in capsys.readouterr().err


def test_prompts_respect_opt_out_and_env(tmp_path, capsys, fresh_prompts, monkeypatch):
    files = _daily_files(tmp_path, 2)
    monkeypatch.setattr(dp.config, "upgrade_prompts", False)
    for f in files:
        dp.load(f)
    assert capsys.readouterr().err == ""
    monkeypatch.setenv("DEPTHO_NO_PROMPTS", "1")
    assert dp.config.__class__().upgrade_prompts is False


def test_non_binance_names_never_trigger(tmp_path, capsys, fresh_prompts):
    for i in range(3):
        p = tmp_path / f"trades_{i}.csv"
        pd.DataFrame({"a": [1]}).to_csv(p, index=False)
        dp.load(p)
    assert capsys.readouterr().err == ""


def test_saved_report_embeds_metrics(tmp_path):
    import json
    import re

    r = pd.Series(np.random.default_rng(5).normal(0, 1e-3, 500), index=_idx(500, "5min"))
    rep = dp.tearsheet(r, title="m")
    html = rep.save(tmp_path / "r.html").read_text(encoding="utf-8")
    data = json.loads(re.search(r'id="deptho-metrics">(.*?)</script>', html).group(1))
    assert data["title"] == "m" and abs(data["metrics"]["total_return"] - rep.metrics["total_return"]) < 1e-12
    assert data["start"].startswith(str(r.index[0].date()))


def test_pro_pointers_in_docstrings_follow_upgrade_prompts(monkeypatch):
    import inspect

    pointed = [dp.load, dp.simulate_taker_fills, dp.datasets.read_aggtrades]
    monkeypatch.setattr(dp.config, "upgrade_prompts", True)
    for f in pointed:
        last = inspect.getdoc(f).splitlines()[-1]
        assert last.startswith("Deptho Pro: ") and last.endswith("(https://deptho.io/go/docstring)."), (f.__name__, last)
    assert "Deptho Pro" not in (dp.pnl_from_positions.__doc__ or "")       # no counterpart, no pointer
    monkeypatch.setattr(dp.config, "upgrade_prompts", False)
    for f in pointed:
        assert "Pro" not in f.__doc__ and "deptho_pro" not in f.__doc__, f.__name__


def test_report_footer_is_a_plain_link_with_off_switches(monkeypatch):
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300),
                  index=pd.date_range("2026-01-01", periods=300, freq="h"))
    rep = dp.tearsheet(r)
    monkeypatch.setattr(dp.config, "upgrade_prompts", True)
    monkeypatch.setattr(dp.config, "report_footer", True)
    html = rep.to_html(include_plotlyjs=False)
    foot = re.search(r'<div class="dp-footer".*?</div>', html).group(0)
    assert re.findall(r'(?:href|src)="([^"]+)"', foot) == ["https://deptho.io/go/report-footer"]  # one link
    assert "Made with" in foot and "<img" not in foot and "<script" not in foot
    monkeypatch.setattr(dp.config, "report_footer", False)
    assert "dp-footer" not in rep.to_html(include_plotlyjs=False)
    monkeypatch.setattr(dp.config, "report_footer", True)
    monkeypatch.setattr(dp.config, "upgrade_prompts", False)
    assert "dp-footer" not in rep.to_html(include_plotlyjs=False)
    monkeypatch.setenv("DEPTHO_NO_FOOTER", "1")
    assert dp.config.__class__().report_footer is False


def test_in_library_links_go_through_source_pages():
    from deptho import guard
    from deptho.config import DEFAULT_UPGRADE_URL, via

    assert via("banner", DEFAULT_UPGRADE_URL) == "https://deptho.io/go/banner"
    assert via("report-footer") == "https://deptho.io/go/report-footer"
    assert via("banner", "https://example.com/pro") == "https://example.com/pro"     # custom DEPTHO_UPGRADE_URL kept
    with pytest.raises(ValueError):
        via("newsletter")
    a = guard.LoadAssessment(path="t.csv", file_bytes=10**9, est_memory_bytes=4 * 10**9, est_seconds=30,
                             available_bytes=10**9, sample_rows=1)
    assert "https://deptho.io/go/banner" in guard.notice_text(a)                # terminal
    assert 'href="https://deptho.io/go/banner"' in guard._notice_html(a)        # notebook


def _tardis_from_aggtrades(src, dest):
    """The same trades in Tardis's `trades` layout (timestamps in microseconds, aggressor as text)."""
    from deptho import datasets as bn

    raw = bn._read(src, bn.AGGTRADES_COLS)
    maker = raw["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"])
    pd.DataFrame({
        "exchange": "binance-futures", "symbol": "btcusdt",
        "timestamp": raw["transact_time"].astype("int64") * 1000,
        "local_timestamp": raw["transact_time"].astype("int64") * 1000 + 1500,
        "id": raw["agg_trade_id"], "side": np.where(maker, "sell", "buy"),
        "price": raw["price"], "amount": raw["quantity"],
    }).to_csv(dest, index=False)
    return dest


def test_read_tardis_trades_matches_aggtrades(tmp_path):
    from datetime import date
    from deptho import datasets as bn
    from deptho._synthetic import make

    make(tmp_path, tick_day=date(2026, 9, 30), kline_days=1, trades_per_day=5_000)
    agg = next(tmp_path.glob("*aggTrades*.zip"))
    gz = _tardis_from_aggtrades(agg, tmp_path / "binance-futures_trades_2026-09-30_BTCUSDT.csv.gz")
    a, t = bn.read_aggtrades(agg), bn.read_tardis_trades(gz)
    assert list(t.columns) == list(a.columns) == ["price", "qty", "side"]
    assert str(t.index.tz) == "UTC" and t.index.name == "ts" and t["side"].dtype == np.int8
    ns = lambda f: f.set_axis(f.index.as_unit("ns"))      # index resolution differs by source and pandas version
    pd.testing.assert_frame_equal(ns(t), ns(a), check_freq=False)
    # Same replay either way.
    orders = pd.DataFrame({"side": [1, -1, 1], "qty": [0.5, 2.0, 0.1]}, index=a.index[[10, 200, 3000]])
    pd.testing.assert_frame_equal(dp.simulate_taker_fills(t, orders), dp.simulate_taker_fills(a, orders))
    bad = tmp_path / "bad.csv"
    pd.read_csv(gz).assign(side="unknown").to_csv(bad, index=False)
    with pytest.raises(ValueError, match="aggressor"):
        bn.read_tardis_trades(bad)


def test_fill_gap_card(tmp_path, monkeypatch):
    idx = pd.date_range("2026-07-01", periods=1500, freq="1h", tz="UTC")
    px = pd.Series(60000 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0002, 0.004, len(idx)))), index=idx)
    pos = np.sign(px.diff(24)).fillna(0.0)
    fund = pd.Series(0.0001, index=pd.date_range("2026-07-01", periods=200, freq="8h", tz="UTC"))
    pnl = dp.pnl_from_positions(px, pos, funding=fund, fee_bps=5, slippage_bps=3.0)
    rep = dp.tearsheet(pnl=pnl, title="BTCUSDT perp")
    monkeypatch.setattr(dp.config, "upgrade_prompts", True)
    monkeypatch.setattr(dp.config, "report_footer", True)
    out = rep.fill_gap_card(tmp_path / "card.svg")
    assert out["svg"] == tmp_path / "card.svg"
    svg = out["svg"].read_text(encoding="utf-8")
    bar_close = (1 + pnl["net"] + pnl["slippage"]).prod() - 1
    replayed = (1 + pnl["net"]).prod() - 1
    shown = lambda x: f"{x * 100:+.1f}%".replace("-", "\u2212")
    assert shown(bar_close) in svg and shown(replayed) in svg and replayed < bar_close
    assert f"Both include fees {shown(-pnl['fees'].sum())} and funding {shown(-pnl['funding'].sum())}" in svg
    assert "Slippage" not in svg and "Gap from replayed fills" in svg       # slippage stays in the report
    assert "BTCUSDT perp · 2026-06-30 \u2013 2026-09-01" in svg and "Synthetic" not in svg  # 1st close 07-01 00:00       # own case, en dash
    assert "matched the backtest" not in svg                                    # only when the gap is 0.0
    assert re.findall(r'(?:href|src)="([^"]+)"', svg) == ["https://deptho.io/go/card"]   # one link, nothing loaded
    assert "<image" not in svg and "<script" not in svg and "Made with Deptho" in svg
    try:
        import PIL  # noqa: F401
        assert out["png"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    except ImportError:
        assert out["png"] is None
    monkeypatch.setattr(dp.config, "report_footer", False)                  # same switch as the report footer
    plain = rep.fill_gap_card(tmp_path / "plain")["svg"].read_text(encoding="utf-8")
    assert "Made with" not in plain and "href" not in plain
    from deptho._synthetic import SYNTHETIC_TAG

    demo = dp.tearsheet(pnl=pnl, title="BTCUSDT perp", data_tag=SYNTHETIC_TAG).fill_gap_card(tmp_path / "demo")
    assert "BTCUSDT perp · 2026-06-30 \u2013 2026-09-01 · Synthetic demo data" in demo["svg"].read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="no slippage"):
        dp.tearsheet(pnl=dp.pnl_from_positions(px, pos, fee_bps=5)).fill_gap_card(tmp_path / "x")
    with pytest.raises(ValueError, match="pnl="):
        dp.tearsheet(pnl["net"]).fill_gap_card(tmp_path / "y")


def _card_rows(svg: str) -> dict:
    texts = re.findall(r">([^<>]+)</text>", svg)
    return {texts[i]: texts[i + 1] for i in range(len(texts) - 1)}


def test_fill_gap_card_gap_is_the_difference_as_shown():
    """The gap row is replayed minus backtest, as the two rows display them, for any returns."""
    from decimal import Decimal

    from deptho import card

    num = lambda t: Decimal(t.replace("\u2212", "-").rstrip("%").removesuffix(" pts"))
    rng = np.random.default_rng(7)
    cases = [(0.36403, 0.36257), (-0.03957, -0.11171), (0.00049, -0.00049), (0.1005, 0.0995),
             (0.12345, 0.12345), (-0.0004, 0.0004), (2.5, -0.99)]
    cases += [tuple(rng.normal(0, 0.5, 2)) for _ in range(500)]
    cases += [(b, b - abs(rng.normal(0, 0.002))) for b in rng.normal(0, 0.3, 300)]      # small gaps
    for b, r in cases:
        n = card.GapNumbers("BTCUSDT", "2026-07-01 \u2013 2026-09-30", float(b), float(r), 0.08, -0.007, 0.001)
        rows = _card_rows(card.svg(n))
        back, rep, gap = (rows[k] for k in ("Backtest (fills at bar close)", "Replayed against trades",
                                            "Gap from replayed fills"))
        assert num(gap) == num(rep) - num(back), (b, r, back, rep, gap)
        assert gap.endswith(" pts") and "-" not in back + rep + gap                # real minus signs
        assert gap == "0.0 pts" or gap[0] in "+\u2212"


def test_fill_gap_card_png_uses_the_bundled_fonts(tmp_path, monkeypatch):
    """The PNG is drawn with the bundled Space Grotesk and JetBrains Mono, whose OFL licences ship with them."""
    pytest.importorskip("PIL")
    from pathlib import Path

    from PIL import ImageFont

    from deptho import card

    for font, licence in (("SpaceGrotesk-Variable.ttf", "OFL-SpaceGrotesk.txt"),
                          ("JetBrainsMono-Regular.ttf", "OFL-JetBrainsMono.txt")):
        assert (card.FONTS / font).stat().st_size > 50_000
        text = (card.FONTS / licence).read_text(encoding="utf-8")
        assert "SIL Open Font License, Version 1.1" in text and "Copyright 2020 The" in text
    used = []
    real = ImageFont.truetype
    monkeypatch.setattr(ImageFont, "truetype", lambda f, *a, **k: used.append(Path(f).name) or real(f, *a, **k))
    monkeypatch.setattr(ImageFont, "load_default", lambda *a, **k: pytest.fail("fell back to Pillow's font"))
    n = card.GapNumbers("BTCUSDT", "2026-07-01 \u2013 2026-09-30", 0.36, 0.35, 0.08, -0.007, 0.001, synthetic=True)
    out = card.png(n, tmp_path / "c.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert set(used) == {"SpaceGrotesk-Variable.ttf", "JetBrainsMono-Regular.ttf"}


def test_fill_gap_card_header_fits_and_says_when_fills_matched(tmp_path):
    """Header: subject in its own case, en dash, long subjects cut with an ellipsis so it never overflows.
    A gap that shows as 0.0 pts gets one quiet line saying the fills matched."""
    from deptho import card

    period = "2026-07-01 \u2013 2026-09-30"
    for subject in ("BTCUSDT", "BTCUSDT perp - EMA trend, 90 days", "a" * 300, "word " * 40, "1000PEPEUSDT momentum v2"):
        for synthetic in (False, True):
            n = card.GapNumbers(subject, period, 0.1, 0.05, 0.02, 0.001, 0.04, synthetic=synthetic)
            h = card.header(n)
            tail = " · " + period + (" · Synthetic demo data" if synthetic else "")
            assert len(h) <= card.HEADER_CHARS and h.endswith(tail)
            shown, full = h[:-len(tail)], " ".join(subject.split()).replace(" - ", " \u2013 ")
            assert shown == full or (shown.endswith(card.ELLIPSIS) and full.startswith(shown[:-1]))
            assert (shown != full) == (len(full) + len(tail) > card.HEADER_CHARS)
    assert card.header(card.GapNumbers("BTCUSDT perp - EMA", period, 0, 0, 0, 0, 1)).startswith("BTCUSDT perp \u2013 EMA")
    try:
        from PIL import ImageFont

        mono = ImageFont.truetype(str(card.FONTS / "JetBrainsMono-Regular.ttf"), card.HEADER_PX)
        widest = card.header(card.GapNumbers("W" * 300, period, 0, 0, 0, 0, 1, synthetic=True))
        width = sum(mono.getlength(ch) + card.HEADER_SPACING for ch in widest)
        assert 88 + width <= card.W - 88, width                       # inside the card's text column
    except ImportError:
        pass

    same = card.GapNumbers("BTCUSDT", period, -0.36012, -0.35989, 0.05, 0.001, 0.0002)
    assert card.gap(same) == "0.0 pts" and card.MATCHED in card.svg(same)
    apart = card.GapNumbers("BTCUSDT", period, -0.360, -0.362, 0.05, 0.001, 0.002)
    assert card.gap(apart) == "\u22120.2 pts" and card.MATCHED not in card.svg(apart)
    assert card.MATCHED == "Replayed fills matched the backtest at these order sizes."
    pytest.importorskip("PIL")
    assert card.png(same, tmp_path / "same.png").exists()


def test_demo_fetch_names_missing_files_and_loads_only_its_window(tmp_path, monkeypatch):
    """A day missing from the archive is skipped and named, not reported as 'could not reach';
    a missing funding month stops with its URL; old days left in the cache stay out of the run."""
    import shutil
    import urllib.error
    from datetime import date, timedelta
    from pathlib import Path

    from deptho import _synthetic, demo
    from deptho import datasets as bn

    src = tmp_path / "src"
    _synthetic.make(src, tick_day=date(2026, 9, 30), kline_days=40, trades_per_day=5_000)
    names = {p.name for p in src.iterdir()}
    gone = {date(2026, 9, 25)}

    def fake(kind, symbol, day, dest, interval="1m", log=print):
        url = bn._url(kind, symbol, day, interval)
        name = url.rsplit("/", 1)[1]
        if day in gone or name not in names:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        Path(dest).mkdir(parents=True, exist_ok=True)
        return Path(shutil.copy(src / name, Path(dest) / name))

    monkeypatch.setattr(bn, "download", fake)
    logs = []
    cache = tmp_path / "cache"
    demo.fetch("BTCUSDT", date(2026, 9, 30), 30, cache, log=logs.append)
    text = "\n".join(logs)
    assert "klines 2026-09-25: not in the archive (404) https://data.binance.vision/" in text
    assert "continuing without 1 kline day(s): 2026-09-25" in text

    # Old days from an earlier run, still in the cache, don't leak into this run's window.
    for p in src.glob("*-1m-2026-08-2*.zip"):
        shutil.copy(p, cache / p.name)
    start, end = date(2026, 9, 1), date(2026, 9, 30)
    k, f, t, d = demo.load_dir(cache, start, end)
    assert k.index.min() >= pd.Timestamp(start, tz="UTC") and k.index.max() < pd.Timestamp(end + timedelta(days=1), tz="UTC")
    assert len(demo.load_dir(cache)[0]) > len(k)                       # without a window: everything

    gone.update(date(2026, 9, d) for d in range(10, 15))                 # too many missing days
    with pytest.raises(demo.ArchiveFileMissing, match="5 of 30 kline days|6 of 30 kline days"):
        demo.fetch("BTCUSDT", date(2026, 9, 30), 30, tmp_path / "c2", log=logs.append)
    gone.clear(); names.discard("BTCUSDT-fundingRate-2026-09.zip")
    with pytest.raises(demo.ArchiveFileMissing, match="funding file for 2026-09 isn't in the archive yet "
                                                      "\\(https://data.binance.vision/.*2026-09.zip\\)"):
        demo.fetch("BTCUSDT", date(2026, 9, 30), 30, tmp_path / "c3", log=logs.append)


def _zip_csv(path, text):
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(path.with_suffix(".csv").name, text)
    return path


@pytest.mark.parametrize("header", [True, False])
def test_archive_readers_take_milliseconds_and_microseconds(tmp_path, header):
    """Every reader lands on the same times whether a file stamps ms (13 digits) or us (16)."""
    from deptho import datasets as bn

    t = pd.Timestamp("2026-10-31 12:34:56.789", tz="UTC")
    ms, us = t.value // 10**6, t.value // 10**3
    files = {
        "aggTrades": ("agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker",
                      "1,70000.5,0.01,1,1,{ts},false", bn.read_aggtrades),
        "klines": ("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
                   "taker_buy_quote_volume,ignore", "{ts},1,2,0.5,1.5,10,{ts},15,3,4,6,0", bn.read_klines),
        "fundingRate": ("calc_time,funding_interval_hours,last_funding_rate", "{ts},8,0.0001", bn.read_funding),
        "bookDepth": ("timestamp,percentage,depth,notional", "{ts},-1.00,10,700000", bn.read_bookdepth),
    }
    for kind, (head, row, reader) in files.items():
        got = []
        for unit, ts in (("ms", ms), ("us", us)):
            text = (head + "\n" if header else "") + row.format(ts=ts) + "\n"
            out = reader(_zip_csv(tmp_path / f"BTCUSDT-{kind}-{unit}-{header}.zip", text))
            got.append(out.index[0])
        assert got[0] == got[1] == t, (kind, got)
    # A join of files in both units sorts correctly (per-value detection, not per file).
    mixed = bn._ms_index(pd.Series([ms, us + 1000]))
    assert list(mixed) == [t, t + pd.Timedelta("1ms")]
    # bookDepth's usual text timestamps still work.
    text = "timestamp,percentage,depth,notional\n2026-10-31 12:34:56,-1.00,10,700000\n"
    assert bn.read_bookdepth(_zip_csv(tmp_path / "bd-text.zip", text)).index[0] == pd.Timestamp("2026-10-31 12:34:56", tz="UTC")


def test_replay_drops_orders_across_a_gap_and_at_the_end_of_the_tape():
    """Orders whose fill would need prints far after them (a gap in the tape, or the end of the
    day) are dropped, not filled at far-off prices; max_wait=None keeps the old behaviour."""
    day1 = pd.date_range("2026-09-30 23:50", periods=600, freq="1s", tz="UTC")
    day2 = pd.date_range("2026-10-31 00:00", periods=600, freq="1s", tz="UTC")
    trades = pd.DataFrame({"price": np.r_[np.full(600, 60000.0), np.full(600, 70000.0)],
                           "qty": 0.5, "side": np.tile([1, -1], 600).astype("int8")}, index=day1.append(day2))
    orders = pd.DataFrame({"side": [1, 1, 1, -1], "qty": [1.0, 1.0, 1.0, 400.0]},
                          index=pd.DatetimeIndex([pd.Timestamp(x, tz="UTC") for x in (
                              "2026-09-30 23:55:00",      # inside the tape: fills
                              "2026-10-15 12:00:00",      # in the gap between the days
                              "2026-09-30 23:59:58",      # its fill would cross the gap
                              "2026-10-31 00:01:00")]))   # more than the rest of the tape holds
    loose = dp.simulate_taker_fills(trades, orders)
    assert len(loose) == 3 and loose["price"].max() == 70000.0                 # old behaviour: fills across the gap
    strict = dp.simulate_taker_fills(trades, orders, max_wait="5min")
    assert list(strict.index) == [orders.index[0]]                             # only the order inside the tape
    assert dp.slippage_bps(strict)["slippage_bps"].abs().max() < 1


def test_demo_replays_one_tick_day_even_with_older_days_cached(tmp_path):
    """A cache holding an earlier run's tick day inside the window: the replay uses this run's day
    only, so slippage stays in basis points, not a month's price move."""
    from datetime import date

    from deptho import _synthetic, costs, demo

    data = tmp_path / "cache"
    for tick, seed in ((date(2026, 9, 30), 1), (date(2026, 10, 31), 2)):
        _synthetic.make(tmp_path / str(tick), tick_day=tick, kline_days=40, trades_per_day=30_000, seed=seed)
        data.mkdir(exist_ok=True)
        for p in (tmp_path / str(tick)).glob("*.zip"):
            if not (data / p.name).exists():
                p.rename(data / p.name)
    k, f, t, d = demo.load_dir(data, date(2026, 9, 22), date(2026, 10, 31))
    assert {str(x) for x in t.index.date} == {"2026-10-31"} and {str(x) for x in d.index.date} == {"2026-10-31"}
    assert {str(x) for x in demo.load_dir(data)[2].index.date} == {"2026-10-31"}   # no window: latest day
    s = costs.slippage_bps(demo.execution_study(t, d))["slippage_bps"]
    assert len(s) > 300 and s.abs().quantile(0.99) < 25, s.describe()     # thin test tape: some orders dropped


def test_report_numbers_never_show_minus_zero_and_titles_use_en_dashes(tmp_path):
    from deptho import _synthetic, demo
    from deptho.dashboard import _share

    assert [_share(x) for x in (-0.0004, 0.0004, -0.003, 0.25, -0.256, float("nan"))] == \
        ["0%", "0%", "-0.3%", "25%", "-26%", "n/a"]
    data = tmp_path / "d"
    _synthetic.make(data, kline_days=10, trades_per_day=20_000)
    rep = demo.run(data)
    assert rep.title.startswith("BTCUSDT perp – EMA trend") and " - " not in rep.title
    assert "-0%" not in rep._tiles_html()


def test_card_when_replay_measured_no_cost_and_demo_never_leaves_a_stale_card(tmp_path, monkeypatch, capsys):
    """Joseph's real run: the replay measured a mean cost below zero, so nothing was charged. That
    is a result (card: 0.0 pts, fills matched), not a missing measurement (still refused). The demo
    replaces any older card at its path, and a card it can't write is impossible to miss."""
    from deptho import _synthetic, card, costs, demo

    idx = pd.date_range("2026-07-01", periods=500, freq="1h", tz="UTC")
    px = pd.Series(60000 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.004, len(idx)))), index=idx)
    pos = np.sign(px.diff(24)).fillna(0.0)
    pnl = dp.pnl_from_positions(px, pos, fee_bps=5, slippage_bps=0.0)          # measured -0.32 bps -> charged 0
    fills = pd.DataFrame({"side": [1, -1, 1], "qty": 1.0, "ref_price": 100.0, "price": [99.99, 100.01, 100.0]},
                         index=idx[[10, 20, 30]])
    assert costs.slippage_bps(fills)["slippage_bps"].mean() < 0
    svg = dp.tearsheet(pnl=pnl, fills=fills).fill_gap_card(tmp_path / "measured")["svg"].read_text(encoding="utf-8")
    assert "0.0 pts" in svg and card.MATCHED in svg
    assert dp.tearsheet(pnl=pnl).fill_gap_card(tmp_path / "flag", measured=True)["svg"].exists()
    with pytest.raises(ValueError, match="no slippage"):                         # never measured: refused
        dp.tearsheet(pnl=pnl).fill_gap_card(tmp_path / "never")

    # The demo, end to end, with a replay whose mean cost is below zero and an old card in place.
    data = tmp_path / "d"
    _synthetic.make(data, kline_days=10, trades_per_day=20_000)
    old_svg, old_png = tmp_path / "btc-card.svg", tmp_path / "btc-card.png"
    old_svg.write_text("<svg>OLD -72.2%</svg>"); old_png.write_bytes(b"OLD")
    neg = lambda trades, depth: pd.DataFrame({"side": [1, -1], "qty": 1.0, "ref_price": 100.0, "price": [99.99, 100.01]},
                                             index=trades.index[[100, 200]])
    monkeypatch.setattr(demo, "execution_study", neg)
    args = ["--data", str(data), "--no-open", "--out", str(tmp_path / "r.html"), "--card", str(tmp_path / "btc-card")]
    demo.main(args)
    new = old_svg.read_text(encoding="utf-8")
    assert "OLD" not in new and "0.0 pts" in new and card.MATCHED in new
    pytest.importorskip("PIL")
    assert old_png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    # Without Pillow, an older PNG doesn't survive beside the new SVG.
    old_png.write_bytes(b"OLD")
    monkeypatch.setattr(card, "png", lambda n, path: None)
    demo.main(args)
    assert old_svg.exists() and not old_png.exists() and "SVG only" in capsys.readouterr().out

    # A card that can't be written: the old one is gone, a banner says so last, and the exit code is 1.
    old_svg.write_text("<svg>OLD</svg>")
    monkeypatch.setattr(dp.Report, "fill_gap_card", lambda self, *a, **k: (_ for _ in ()).throw(ValueError("why")))
    with pytest.raises(SystemExit) as exit_:
        demo.main(args)
    err = capsys.readouterr().err
    assert exit_.value.code == 1 and "FILL GAP CARD NOT WRITTEN" in err and "why" in err and "btc-card.svg" in err
    assert not old_svg.exists() and not old_png.exists()


def test_fill_gap_card_period_shows_the_last_day_included():
    """Bars are stamped at their close: 5-minute closes through 2026-11-01 00:00 are data through
    2026-10-31 (Joseph's real card showed 2026-11-01)."""
    from deptho import card

    d = "\u2013"
    five = pd.date_range("2026-08-03 00:05", "2026-11-01 00:00", freq="5min", tz="UTC")
    assert card.period(five) == f"2026-08-03 {d} 2026-10-31"
    first_at_midnight = pd.date_range("2026-08-03 00:00", "2026-10-31 23:55", freq="5min", tz="UTC")
    assert card.period(first_at_midnight) == f"2026-08-02 {d} 2026-10-31"         # that bar closed Aug 2
    hourly = pd.date_range("2026-07-01 01:00", "2026-09-01 11:00", freq="1h", tz="UTC")
    assert card.period(hourly) == f"2026-07-01 {d} 2026-09-01"
    daily = pd.date_range("2026-08-01", "2026-10-31", freq="1D", tz="UTC")
    assert card.period(daily) == f"2026-08-01 {d} 2026-10-31"                      # daily bars keep their dates
    pnl = dp.pnl_from_positions(pd.Series(100.0, index=five), pd.Series(0.0, index=five), fee_bps=5)
    assert card.gap_numbers(pnl, "BTCUSDT", measured=True).period == f"2026-08-03 {d} 2026-10-31"
