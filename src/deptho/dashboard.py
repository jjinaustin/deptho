"""The interactive tearsheet: one call in, a Deptho-themed Plotly dashboard out."""
from __future__ import annotations

import html
import json
import tempfile
import textwrap
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import costs, metrics
from .config import HOMEPAGE, config, tagged
from ._theme import FONT_DISPLAY, FONT_MONO, c, lockup_html, rgba


@dataclass
class Report:
    title: str
    figure: go.Figure
    metrics: dict
    returns: pd.Series
    pnl: pd.DataFrame | None = None
    figure_narrow: go.Figure | None = None
    fills: pd.DataFrame | None = None
    data_tag: str | None = None
    _tiles: list = field(default_factory=list)

    # ---- output -------------------------------------------------------
    def table(self) -> pd.DataFrame:
        return pd.Series(self.metrics, name=self.title).to_frame()

    def _tiles_html(self) -> str:
        label = (f"font-family:{FONT_MONO};font-size:12px;font-weight:500;letter-spacing:0.1em;"
                 f"text-transform:uppercase;color:{c('ink-muted')}")
        value = (f"font-family:{FONT_MONO};font-size:22px;font-weight:500;font-variant-numeric:tabular-nums;"
                 f"color:{c('ink')};margin-top:6px")
        card = f"flex:1 1 150px;min-width:0;background:{c('layer-1')};border-radius:16px;padding:14px 18px"
        one_line = "white-space:nowrap;overflow:hidden;text-overflow:ellipsis"   # equal tile heights
        cells, notes = [], []
        for name, val, *note in self._tiles:
            tip = f' title="{html.escape(note[0])}"' if note else ""
            mark = "*" * (len(notes) + 1) if note else ""
            if note:
                notes.append(f"{mark} {html.escape(note[0])}")
            cells.append(f'<div style="{card}"{tip}><div style="{label};{one_line}">{html.escape(name)}{mark}</div>'
                         f'<div style="{value};{one_line}">{html.escape(val)}</div></div>')
        foot = "".join(f'<div style="font-size:13px;color:{c("ink-muted")};margin-top:10px">{n}</div>' for n in notes)
        tag = f'<span style="{label};margin-left:auto">{html.escape(self.data_tag)}</span>' if self.data_tag else ""
        outer = f"font-family:{FONT_DISPLAY};background:{c('abyss')};color:{c('ink')};padding:24px 24px 8px"
        return (
            f'<div style="{outer}">'
            f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:16px">{lockup_html()}{tag}</div>'
            f'<div style="font-size:22px;font-weight:600;letter-spacing:-0.02em;margin-bottom:16px">'
            f'{html.escape(self.title)}</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:12px">{"".join(cells)}</div>{foot}</div>'
        )

    def _footer_html(self) -> str:
        """'Made with Deptho': a plain link, nothing fetched. Off with dp.config.report_footer = False,
        DEPTHO_NO_FOOTER=1, or upgrade prompts off."""
        if not config.show_footer:
            return ""
        style = (f"font-family:{FONT_MONO};font-size:12px;color:{c('ink-muted')};background:{c('abyss')};"
                 "padding:0 24px 20px;text-align:right")
        return (f'<div class="dp-footer" style="{style}">Made with <a href="{html.escape(tagged(HOMEPAGE, "report-footer"))}" target="_blank" '
                f'rel="noopener" style="color:{c("ink-muted")};text-decoration:underline">Deptho</a></div>')

    def to_html(self, include_plotlyjs: bool | str = True) -> str:
        cfg = {"displaylogo": False, "responsive": True}
        wide = self.figure.to_html(full_html=False, include_plotlyjs=include_plotlyjs, config=cfg)
        narrow = ""
        if self.figure_narrow is not None:
            # Phones get a stacked single-column layout; CSS picks which one shows.
            narrow = self.figure_narrow.to_html(full_html=False, include_plotlyjs=False, config=cfg)
        css = (".dp-narrow{display:none}"
               "@media (max-width:700px){.dp-wide{display:none}.dp-narrow{display:block}}") if narrow else ""
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<meta name='color-scheme' content='dark'>"
            f"<title>{html.escape(self.title)} - Deptho</title><style>{css}</style></head>"
            f"<body style='margin:0;background:{c('abyss')}'>{self._tiles_html()}"
            f"<div style='background:{c('abyss')};padding:0 12px 24px'>"
            f"<div class='dp-wide'>{wide}</div><div class='dp-narrow'>{narrow}</div></div>"
            f"{self._footer_html()}{self._metrics_json()}</body></html>"
        )

    def _metrics_json(self) -> str:
        """The headline metrics, machine-readable, so tools can quote a saved report without re-running it."""
        def clean(v):
            v = float(v)
            return v if np.isfinite(v) else None

        data = {"title": self.title, "metrics": {k: clean(v) for k, v in self.metrics.items()
                                                 if isinstance(v, (int, float, np.floating, np.integer))}}
        if self.returns is not None and len(self.returns):
            data["start"], data["end"] = str(self.returns.index[0]), str(self.returns.index[-1])
        payload = json.dumps(data).replace("</", "<\\/")
        return f'<script type="application/json" id="deptho-metrics">{payload}</script>'

    def save(self, path: str | Path) -> Path:
        """Write a self-contained HTML file (works offline, shareable with LPs)."""
        p = Path(path)
        p.write_text(self.to_html(include_plotlyjs=True), encoding="utf-8")
        return p

    def show(self):
        if _in_notebook():
            self._ipython_display_()
        else:
            p = self.save(Path(tempfile.gettempdir()) / "deptho_report.html")
            webbrowser.open(p.as_uri())
            return p

    def _ipython_display_(self):
        from IPython.display import HTML, display

        display(HTML(self._tiles_html()))
        self.figure.show()
        if config.show_footer:
            display(HTML(self._footer_html()))


def _in_notebook() -> bool:
    from .guard import _in_notebook as f

    return f()


# ---------------------------------------------------------------------------

def _pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:+.{digits}f}%"


def _num(x: float, digits: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def _decimate(s: pd.Series, max_points: int = 3000) -> tuple[pd.Index, np.ndarray]:
    """Min/max decimation for display: each bucket keeps its extremes, in time
    order, so spikes and drawdown troughs survive while the payload stays small."""
    n = len(s)
    if n <= max_points:
        return s.index, s.to_numpy()
    buckets = max_points // 2
    edges = np.linspace(0, n, buckets + 1, dtype=int)
    v = s.to_numpy()
    keep = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = v[a:b]
        if np.isnan(seg).all():
            keep.append(a)
            continue
        i, j = a + int(np.nanargmin(seg)), a + int(np.nanargmax(seg))
        keep.extend(sorted({i, j}))
    keep = np.asarray(keep)
    return s.index[keep], v[keep]


def _auto_window(n: int) -> int:
    return int(min(max(20, n // 10), 5000))


def tearsheet(
    returns: pd.Series | None = None,
    *,
    pnl: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    fills: pd.DataFrame | None = None,
    periods_per_year: float | None = None,
    window: int | None = None,
    title: str = "Strategy",
    fills_label: str = "Execution cost",
    data_tag: str | None = None,
) -> Report:
    """Build the dashboard.

    Pass either ``returns`` (net per-period returns) or ``pnl`` (the output of
    :func:`deptho.costs.pnl_from_positions`, which unlocks the cost-drag
    panel). ``fills`` adds the execution-cost panels (fill vs arrival price;
    label them with ``fills_label``, e.g. "Simulated execution cost (trade-print
    replay)" for replayed fills); ``benchmark`` (per-period
    returns of e.g. BTC buy-and-hold) turns the alpha panel into true alpha.
    ``data_tag`` (e.g. "SYNTHETIC DATA · DEMO") is stamped top-right on every chart.
    """
    if pnl is not None:
        rets = pnl["net"]
    elif returns is not None:
        rets = returns
    else:
        raise ValueError("pass returns= or pnl=")
    rets = rets.astype(float).sort_index()
    ppy = periods_per_year or metrics.infer_periods_per_year(rets.index)
    win = window or _auto_window(len(rets))
    stats = metrics.summary(rets, ppy)

    slip = costs.slippage_bps(fills) if fills is not None else None
    has_slip = slip is not None and len(slip) > 0

    titles = [
        "Equity (cumulative return)",
        "Cumulative cost drag (% of equity, below 0 = received)" if pnl is not None else "Rolling return (annualised)",
        "Underwater (drawdown from peak)",
        f"Rolling Sharpe / Sortino ({win} bars)",
        "Alpha decay: rolling alpha vs benchmark (annualised)" if benchmark is not None
        else "Edge decay: rolling mean return (annualised)",
    ]
    if has_slip:
        titles += [f"{fills_label} vs arrival price (bps, + = cost)",
                   f"{fills_label} vs depth-band participation" if "participation" in slip
                   else f"{fills_label} vs order size"]

    def build(cols: int) -> go.Figure:
        n = len(titles)
        if cols == 2:
            cells = [(1, 1)] + [(2 + i // 2, 1 + i % 2) for i in range(n - 1)]
            rows = cells[-1][0]
            specs = [[{"colspan": 2}, None]] + [[{}, {}] for _ in range(rows - 1)]
            heights = [0.34] + [0.22] * (rows - 1)
            vspace = 0.07
        else:
            cells = [(i + 1, 1) for i in range(n)]
            rows = n
            specs = [[{}] for _ in range(n)]
            heights = [1.0] * n
            vspace = 0.065
        shown = titles
        if cols == 1:   # phone width: wrap long titles, and the data tag goes on its own line under each
            tag = (f"<br><span style=\"font-family:{FONT_MONO};font-size:10px;color:{c('ink-muted')}\">"
                   f"{data_tag}</span>") if data_tag else ""
            shown = ["<br>".join(textwrap.wrap(t, 40)) + tag for t in titles]
        fig = make_subplots(rows=rows, cols=cols, specs=specs, row_heights=heights, subplot_titles=shown,
                            vertical_spacing=vspace, horizontal_spacing=0.07)
        # Subplot titles are the only annotations so far: style them before others are added.
        fig.update_annotations(font=dict(color=c("ink"), size=13, family=FONT_DISPLAY))
        # Left-align panel titles over their plot, leaving the top-right for the data tag.
        domains = [fig.layout[k].domain for k in fig.layout if k.startswith("xaxis")]
        for ann in fig.layout.annotations:
            d = min(domains, key=lambda d: abs((d[0] + d[1]) / 2 - ann.x))
            ann.update(x=d[0], xanchor="left", align="left")
        cell = dict(zip(["equity", "cost", "dd", "roll", "alpha", "hist", "scatter"], cells))

        def line(x, y, name, color, panel, dash=None, fill=None, fillcolor=None, width=2, legend=True,
                 fmt=".2%"):
            row, col = cell[panel]
            x, y = _decimate(pd.Series(np.asarray(y, dtype=float), index=x))
            fig.add_trace(go.Scatter(
                x=x, y=y, name=name, mode="lines", showlegend=legend,
                legend=f"legend{_suffix(cells.index((row, col)))}" if legend else None,
                line=dict(color=color, width=width, dash=dash), fill=fill, fillcolor=fillcolor,
                hovertemplate=f"{name}: %{{y:{fmt}}}<extra></extra>",
            ), row=row, col=col)

        # 1. Equity
        eq_net = metrics.equity_curve(rets) - 1
        if pnl is not None and (pnl[["fees", "slippage", "funding"]].abs().to_numpy().sum() > 0):
            line(eq_net.index, metrics.equity_curve(pnl["gross"]) - 1, "Gross (before costs)", c("sky"), "equity",
                 dash="dot")
        if benchmark is not None:
            b = benchmark.reindex(rets.index).fillna(0.0)
            line(b.index, metrics.equity_curve(b) - 1, "Benchmark", c("sky"), "equity", width=1.5)
        line(eq_net.index, eq_net, "Net", c("signal"), "equity", fill="tozeroy", fillcolor=rgba("signal", 0.15))

        # 2. Cost drag or rolling return
        if pnl is not None:
            # Every cost is a loss colour; dash style tells them apart.
            for col, name, dash in (("funding", "Funding", None), ("fees", "Fees", "dash"),
                                    ("slippage", "Slippage", "dot")):
                if pnl[col].abs().sum() > 0:
                    line(pnl.index, pnl[col].cumsum(), name, c("loss"), "cost", dash=dash)
        else:
            line(rets.index, rets.rolling(win, min_periods=2).mean() * ppy, "Rolling return", c("signal"), "cost",
                 legend=False)

        # 3. Underwater
        dd = metrics.drawdown(rets)            # computed at full resolution...
        if isinstance(dd.index, pd.DatetimeIndex) and len(dd) > 1 and (dd.index[-1] - dd.index[0]) > pd.Timedelta("7D"):
            dd = dd.resample("1h").last().dropna()  # ...drawn hourly (the max-drawdown tile uses full data)
        line(dd.index, dd, "Drawdown", c("series-2"), "dd", fill="tozeroy", fillcolor=c("drawdown-fill"), width=1,
             legend=False)

        # 4. Rolling Sharpe / Sortino
        line(rets.index, metrics.rolling_sharpe(rets, win, ppy), "Rolling Sharpe", c("signal"), "roll", fmt=".2f")
        line(rets.index, metrics.rolling_sortino(rets, win, ppy), "Rolling Sortino", c("sky"), "roll", fmt=".2f")

        # 5. Alpha decay
        if benchmark is not None:
            ra = metrics.rolling_alpha(rets, benchmark, win, ppy)
            line(ra.index, ra["alpha"], "Rolling alpha", c("signal"), "alpha", legend=False)
        else:
            line(rets.index, rets.rolling(win, min_periods=2).mean() * ppy, "Rolling mean", c("signal"), "alpha",
                 legend=False)

        # 6. Execution cost per fill
        if has_slip:
            s = slip["slippage_bps"].dropna()
            fig.add_trace(go.Histogram(x=s, name="Fills", marker=dict(color=c("loss"), line=dict(color=c("layer-1"), width=1)),
                                       showlegend=False, hovertemplate="%{x} bps: %{y} fills<extra></extra>"), **_rc(cell["hist"]))
            for q, lbl, pos in ((s.mean(), "mean", "top left"), (s.quantile(0.95), "p95", "top right")):
                fig.add_vline(x=q, line=dict(color=c("ink-muted"), width=1, dash="dash"), **_rc(cell["hist"]),
                              annotation_text=f"{lbl} {q:.1f}", annotation_position=pos,
                              annotation_font=dict(color=c("ink-muted"), family=FONT_MONO, size=11))
            fig.update_xaxes(title_text="cost vs arrival price (bps)", **_rc(cell["hist"]))
            fig.update_yaxes(title_text="fills", **_rc(cell["hist"]))
            xcol = "participation" if "participation" in slip else ("qty" if "qty" in slip else None)
            if xcol:
                fig.add_trace(go.Scattergl(
                    x=slip[xcol].abs(), y=slip["slippage_bps"], mode="markers", showlegend=False,
                    marker=dict(color=c("loss"), size=8, opacity=0.6, line=dict(color=c("layer-1"), width=1)),
                    hovertemplate=f"{xcol}: %{{x:.3g}}<br>cost: %{{y:.2f}} bps<extra></extra>",
                ), **_rc(cell["scatter"]))
                fig.update_xaxes(title_text="order size / resting size in tightest depth band" if xcol == "participation"
                                 else "order size", **_rc(cell["scatter"]))
                fig.update_yaxes(title_text="cost vs arrival price (bps)", **_rc(cell["scatter"]))

        _style(fig, rows)
        _place_legends(fig)
        if data_tag and cols == 2:
            _tag_panels(fig, cells, data_tag)
        for panel, f in (("equity", ".0%"), ("cost", ".1%"), ("dd", ".0%"), ("alpha", ".0%")):
            fig.update_yaxes(tickformat=f, **_rc(cell[panel]))
        if cols == 1:
            fig.update_layout(height=260 * rows + 80, margin=dict(l=48, r=12, t=40, b=40))
        return fig

    fig = build(2)

    # KPI tiles
    tiles = [("Net return", _pct(stats["total_return"])), ("Sharpe", _num(stats["sharpe"])),
             ("Sortino", _num(stats["sortino"])), ("Max drawdown", _pct(stats["max_drawdown"]))]
    if pnl is not None:
        gross = float(metrics.equity_curve(pnl["gross"]).iloc[-1] - 1)
        fund = float(pnl["funding"].sum())
        stats["funding_paid"] = fund
        stats["fees_paid"] = float(pnl["fees"].sum())
        stats["gross_return"] = gross
        stats["funding_drag_of_gross"] = fund / abs(gross) if gross else float("nan")
        tiles.append(("Funding paid", _pct(fund, 2)))
        tiles.append(("Funding / |gross|", "n/a" if not np.isfinite(stats["funding_drag_of_gross"])
                      else f"{stats['funding_drag_of_gross'] * 100:.0f}%"))
    if has_slip:
        stats["slippage_bps_mean"] = float(slip["slippage_bps"].mean())
        short = "Exec cost (sim)" if "simulated" in fills_label.lower() else "Exec cost"
        tiles.append((short, f"{stats['slippage_bps_mean']:.2f} bps",
                      f"Average {fills_label.lower()} per fill, in bps vs arrival price (+ = cost)."))

    rep = Report(title=title, figure=fig, figure_narrow=build(1), metrics=stats, returns=rets, pnl=pnl, fills=slip,
                 data_tag=data_tag)
    rep._tiles = tiles
    return rep


def _suffix(i: int) -> str:
    """Plotly names the first axis/legend ``xaxis``/``legend`` and later ones ``xaxis2``..."""
    return "" if i == 0 else str(i + 1)


def _rc(rc: tuple[int, int]) -> dict:
    return {"row": rc[0], "col": rc[1]}


def _place_legends(fig: go.Figure) -> None:
    """One legend per panel, pinned inside its top-left corner, so a colour
    only ever has to be decoded against the series beside it."""
    used = {tr.legend or "legend" for tr in fig.data if tr.showlegend is not False}
    for key in used:
        suffix = key.replace("legend", "")
        xa = fig.layout[f"xaxis{suffix}"]
        ya = fig.layout[f"yaxis{suffix}"]
        fig.update_layout({key: dict(
            x=xa.domain[0] + 0.005, y=ya.domain[1] - 0.005, xanchor="left", yanchor="top",
            orientation="h", bgcolor=rgba("layer-1", 0.85), font=dict(color=c("ink-muted"), size=11),
            xref="paper", yref="paper",
        )})


def _tag_panels(fig: go.Figure, cells: list, tag: str) -> None:
    """Data-source tag (label style) at the top-right of every chart."""
    for i in range(len(cells)):
        sfx = _suffix(i)
        fig.add_annotation(text=tag, xref=f"x{sfx} domain", yref=f"y{sfx} domain", x=1, y=1, xanchor="right",
                           yanchor="bottom", yshift=4, showarrow=False,
                           font=dict(family=FONT_MONO, size=10, color=c("ink-muted")))


def template() -> go.layout.Template:
    """The Deptho Plotly template, built from the design tokens."""
    axis = dict(gridcolor=c("contour"), gridwidth=1, showline=False, zerolinecolor=c("contour"),
                tickfont=dict(family=FONT_MONO, size=12, color=c("ink-muted")),
                title=dict(font=dict(family=FONT_DISPLAY, size=12, color=c("ink-muted"))))
    return go.layout.Template(layout=dict(
        paper_bgcolor=c("abyss"), plot_bgcolor=c("layer-1"),
        font=dict(family=FONT_DISPLAY, color=c("ink-muted"), size=12),
        colorway=[c("signal"), c("sky"), c("series-2"), c("ink-muted")],
        xaxis=axis, yaxis=axis,
        hoverlabel=dict(bgcolor=c("layer-3"), bordercolor=c("contour"),
                        font=dict(color=c("ink"), family=FONT_MONO, size=12)),
        legend=dict(bgcolor=rgba("layer-1", 0.85), font=dict(color=c("ink-muted"))),
    ))


def _register_template() -> None:
    import plotly.io as pio

    pio.templates["deptho"] = template()


_register_template()


def _style(fig: go.Figure, rows: int) -> None:
    fig.update_layout(template="deptho", height=300 + 260 * rows, margin=dict(l=56, r=24, t=48, b=40),
                      hovermode="x unified", bargap=0.05)
    fig.update_xaxes(showspikes=True, spikecolor=c("ink-muted"), spikethickness=1, spikedash="dot", spikemode="across")
