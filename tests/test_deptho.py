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
    assert "Exec cost (sim)" in html and 'title="Average simulated execution cost' in html
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


def test_demo_runs_offline_on_synthetic_data(tmp_path, monkeypatch):
    from deptho import _synthetic, demo

    data = tmp_path / "d"
    _synthetic.make(data, kline_days=10, trades_per_day=20_000)
    out = demo.main(["--data", str(data), "--no-open", "--out", str(tmp_path / "r.html")])
    assert out.exists()
    k, f, t, d = demo.load_dir(data)
    assert len(k) == 10 * 1440 and len(f) == 30 and t is not None and d is not None


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
        assert last.startswith("Deptho Pro: ") and last.endswith("(https://deptho.io/pro.html?utm_source=docstring)."), \
            (f.__name__, last)
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
    assert re.findall(r'(?:href|src)="([^"]+)"', foot) == ["https://deptho.io/?utm_source=report-footer"]  # one link
    assert "Made with" in foot and "<img" not in foot and "<script" not in foot
    monkeypatch.setattr(dp.config, "report_footer", False)
    assert "dp-footer" not in rep.to_html(include_plotlyjs=False)
    monkeypatch.setattr(dp.config, "report_footer", True)
    monkeypatch.setattr(dp.config, "upgrade_prompts", False)
    assert "dp-footer" not in rep.to_html(include_plotlyjs=False)
    monkeypatch.setenv("DEPTHO_NO_FOOTER", "1")
    assert dp.config.__class__().report_footer is False


def test_in_library_links_are_tagged_and_static():
    from deptho import guard
    from deptho.config import tagged

    assert tagged("https://deptho.io", "banner") == "https://deptho.io/?utm_source=banner"
    assert tagged("https://deptho.io/pro.html#pricing", "banner") == "https://deptho.io/pro.html?utm_source=banner#pricing"
    assert tagged("https://x.io/p?a=1", "banner") == "https://x.io/p?a=1&utm_source=banner"
    assert tagged("https://x.io/p?utm_source=mine", "banner") == "https://x.io/p?utm_source=mine"   # custom URL kept
    a = guard.LoadAssessment(path="t.csv", file_bytes=10**9, est_memory_bytes=4 * 10**9, est_seconds=30,
                             available_bytes=10**9, sample_rows=1)
    assert "https://deptho.io/pro.html?utm_source=banner" in guard.notice_text(a)            # terminal
    assert 'href="https://deptho.io/pro.html?utm_source=banner"' in guard._notice_html(a)    # notebook
