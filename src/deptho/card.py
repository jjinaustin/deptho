"""The fill gap card: one shareable image of what replaying a backtest's orders against trades did
to its return. Written as SVG (no dependency) and, when Pillow is installed, PNG. Nothing is fetched
or sent: the numbers come from the user's own report.

    report = dp.tearsheet(pnl=pnl, ...)        # pnl with slippage measured by replaying the orders
    report.fill_gap_card("btc-momentum")       # -> btc-momentum.svg, btc-momentum.png
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from ._theme import FONT_DISPLAY, FONT_MONO, c, mark_svg
from .config import BRAND, config, via

W, H = 1200, 630            # the usual social-card size


@dataclass
class GapNumbers:
    subject: str            # symbol or strategy name
    period: str             # "2026-07-01 – 2026-09-30"
    bar_close: float        # compounded return without execution cost
    replayed: float         # compounded net return, slippage included
    fees: float             # summed per-bar costs, fractions of equity (positive = paid)
    funding: float
    slippage: float         # in the report, not on the card
    synthetic: bool = False  # the demo's synthetic data: said on the card


def gap_numbers(pnl: pd.DataFrame, subject: str, synthetic: bool = False, measured: bool = False) -> GapNumbers:
    """``measured``: slippage was measured by replaying the orders, even if it came out at or below
    zero (then nothing was charged, both numbers match and the card says the fills matched).
    Without it, a report with no slippage is refused: it was never measured."""
    need = {"net", "fees", "funding", "slippage"}
    if pnl is None or not need <= set(pnl.columns):
        raise ValueError("the fill gap card needs a report built from pnl= (dp.pnl_from_positions)")
    slip = float(pnl["slippage"].sum())
    if slip == 0 and not measured:
        raise ValueError("this report has no slippage, so both numbers would match. Measure it by replaying the "
                         "orders (dp.simulate_taker_fills, dp.slippage_bps), pass slippage_bps= to "
                         "dp.pnl_from_positions and fills= to dp.tearsheet")
    return GapNumbers(
        subject=subject, period=period(pnl.index),
        bar_close=float((1 + pnl["net"] + pnl["slippage"]).prod() - 1),
        replayed=float((1 + pnl["net"]).prod() - 1),
        fees=float(pnl["fees"].sum()), funding=float(pnl["funding"].sum()), slippage=slip,
        synthetic=synthetic,
    )


def period(index: pd.Index) -> str:
    """First and last day the bars cover. Bars are stamped at their close, so an intraday bar
    closing at midnight ends the day before: 5-minute closes through 2026-11-01 00:00 are data
    through 2026-10-31. Daily or coarser bars keep their own dates."""
    idx = pd.DatetimeIndex(index).sort_values()
    first, last = idx[0], idx[-1]
    if len(idx) > 1 and pd.Series(idx).diff().median() < pd.Timedelta("1D"):
        tick = pd.Timedelta(1, "ns")
        first, last = first - tick, last - tick
    return f"{first:%Y-%m-%d} {EN_DASH} {last:%Y-%m-%d}"


MINUS = "−"            # a real minus sign; both bundled fonts have it (and the two below)
EN_DASH = "–"
ELLIPSIS = "…"


def _signed(d: Decimal, unit: str) -> str:
    if d == 0:
        return f"0.0{unit}"
    return f"{d:+.1f}{unit}".replace("-", MINUS)


def _shown(x: float) -> Decimal:
    """A fraction as the percentage the card shows, to one decimal."""
    return Decimal(f"{x * 100:.1f}")


def pct(x: float) -> str:
    return _signed(_shown(x), "%")


def cost(x: float) -> str:
    """A cost as its effect on return: paid -> '−1.2%', received -> '+0.3%'."""
    return pct(-x)


def gap(n: GapNumbers) -> str:
    """Replayed minus backtest, in points, from the two numbers as shown (so it always adds up)."""
    return _signed(_shown(n.replayed) - _shown(n.bar_close), " pts")


HEADER_PX, HEADER_SPACING = 24, 2
# JetBrains Mono is 0.6 em per character; keep 16 px spare for fallback monospace fonts in the SVG.
HEADER_CHARS = int((W - 176 - 16) // (0.6 * HEADER_PX + HEADER_SPACING))


def header(n: GapNumbers) -> str:
    """Subject · period [· Synthetic demo data], in the subject's own case. A subject too long for
    one line is cut with an ellipsis; the period and the synthetic label are always shown whole."""
    tail = [n.period] + (["Synthetic demo data"] if n.synthetic else [])
    subject = " ".join(n.subject.split()).replace(" - ", f" {EN_DASH} ")
    room = HEADER_CHARS - len(" · ".join(tail)) - len(" · ")
    if len(subject) > room:
        subject = subject[:max(room - 1, 1)].rstrip(" ,;:·-" + EN_DASH) + ELLIPSIS
    return " · ".join([subject] + tail)                   # single spaces: SVG collapses runs anyway


def includes_line(n: GapNumbers) -> str:
    return f"Both include fees {cost(n.fees)} and funding {cost(n.funding)}"


MATCHED = "Replayed fills matched the backtest at these order sizes."


def matched_line(n: GapNumbers) -> str | None:
    """One quiet line when the gap shows as 0.0 pts, so a zero reads as a result, not a fault."""
    return MATCHED if gap(n) == "0.0 pts" else None


ROWS = (("Backtest (fills at bar close)", 196), ("Replayed against trades", 296), ("Gap from replayed fills", 396))


def svg(n: GapNumbers) -> str:
    e = html.escape
    mark = ""
    if config.show_footer:
        mark = (f'<a href="{e(via("card"))}"><g transform="translate({W - 330},{H - 70})">'
                f'<g transform="scale(0.5)">{mark_svg(64)}</g>'
                f'<text x="44" y="23" font-family="{FONT_MONO}" font-size="20" fill="{c("ink-muted")}">'
                f'Made with {e(BRAND)}</text></g></a>')
    matched = ""
    if matched_line(n):
        matched = (f'<text x="88" y="542" font-family="{FONT_MONO}" font-size="22" fill="{c("ink-muted")}">'
                   f'{e(matched_line(n))}</text>\n')
    values = ((pct(n.bar_close), "ink"), (pct(n.replayed), "ink"), (gap(n), "signal"))
    rows = "\n".join(
        f'<text x="88" y="{y}" font-family="{FONT_DISPLAY}" font-size="30" fill="{c("ink-muted")}">{e(label)}</text>\n'
        f'<text x="{W - 88}" y="{y + 4}" text-anchor="end" font-family="{FONT_DISPLAY}" font-size="64" '
        f'font-weight="600" fill="{c(colour)}">{e(value)}</text>'
        for (label, y), (value, colour) in zip(ROWS, values))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect width="{W}" height="{H}" fill="{c('abyss')}"/>
<rect x="40" y="40" width="{W - 80}" height="{H - 80}" rx="24" fill="{c('layer-1')}"/>
<text x="88" y="112" font-family="{FONT_MONO}" font-size="{HEADER_PX}" letter-spacing="{HEADER_SPACING}" fill="{c('ink-muted')}">{e(header(n))}</text>
{rows}
<line x1="88" y1="446" x2="{W - 88}" y2="446" stroke="{c('contour')}" stroke-width="2"/>
<text x="88" y="500" font-family="{FONT_MONO}" font-size="24" fill="{c('ink-muted')}">{e(includes_line(n))}</text>
{matched}{mark}
</svg>
"""


FONTS = Path(__file__).parent / "fonts"


def png(n: GapNumbers, path: Path) -> Path | None:
    """The same card as PNG, if Pillow is installed (pip install "deptho[png]"); else None.
    Drawn with the bundled Space Grotesk and JetBrains Mono (deptho/fonts, SIL OFL 1.1)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    def fallback(size: int):
        try:
            return ImageFont.load_default(size=size)       # Pillow >= 10.1, scalable
        except TypeError:
            return ImageFont.load_default()

    def display(size: int, weight: int):
        try:
            f = ImageFont.truetype(str(FONTS / "SpaceGrotesk-Variable.ttf"), size)
            f.set_variation_by_axes([weight])
            return f
        except (OSError, ValueError):                      # no FreeType variation support: default weight
            try:
                return ImageFont.truetype(str(FONTS / "SpaceGrotesk-Variable.ttf"), size)
            except OSError:
                return fallback(size)

    def mono(size: int):
        try:
            return ImageFont.truetype(str(FONTS / "JetBrainsMono-Regular.ttf"), size)
        except OSError:
            return fallback(size)

    img = Image.new("RGB", (W, H), c("abyss"))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((40, 40, W - 40, H - 40), radius=24, fill=c("layer-1"))
    head, label_font, value_font = mono(HEADER_PX), display(30, 400), display(64, 600)
    x = 88                                                  # letter-spacing as in the SVG
    for ch in header(n):
        d.text((x, 112), ch, font=head, fill=c("ink-muted"), anchor="ls")
        x += head.getlength(ch) + HEADER_SPACING
    values = ((pct(n.bar_close), "ink"), (pct(n.replayed), "ink"), (gap(n), "signal"))
    for (label, y), (value, colour) in zip(ROWS, values):
        d.text((88, y), label, font=label_font, fill=c("ink-muted"), anchor="ls")
        d.text((W - 88, y + 4), value, font=value_font, fill=c(colour), anchor="rs")
    d.line((88, 446, W - 88, 446), fill=c("contour"), width=2)
    d.text((88, 500), includes_line(n), font=mono(24), fill=c("ink-muted"), anchor="ls")
    if matched_line(n):
        d.text((88, 542), matched_line(n), font=mono(22), fill=c("ink-muted"), anchor="ls")
    if config.show_footer:                                  # the mark at 32 px, then the words
        x, y, k = W - 330, H - 70, 0.5
        d.rounded_rectangle((x + 4 * k, y + 4 * k, x + 60 * k, y + 60 * k), radius=16 * k, outline=c("ink"), width=3)
        d.rounded_rectangle((x + 17 * k, y + 19 * k, x + 53 * k, y + 55 * k), radius=10 * k, outline=c("sky"), width=3)
        d.rounded_rectangle((x + 30 * k, y + 34 * k, x + 46 * k, y + 50 * k), radius=4 * k, fill=c("signal"))
        d.text((x + 44, y + 23), f"Made with {BRAND}", font=mono(20), fill=c("ink-muted"), anchor="ls")
    img.save(path)
    return path


def paths(path: str | Path) -> tuple[Path, Path]:
    """``card`` / ``card.svg`` / ``card.png`` -> (card.svg, card.png)."""
    base = Path(path)
    base = base.with_suffix("") if base.suffix.lower() in (".svg", ".png") else base
    return base.with_name(base.name + ".svg"), base.with_name(base.name + ".png")


def remove(path: str | Path) -> list[Path]:
    """Delete an earlier card at ``path`` (both files), so a stale image can't outlive a new run."""
    gone = []
    for p in paths(path):
        if p.exists():
            p.unlink()
            gone.append(p)
    return gone


def write(n: GapNumbers, path: str | Path) -> dict:
    out_svg, out_png = paths(path)
    out_svg.write_text(svg(n), encoding="utf-8")
    made = png(n, out_png)
    if made is None and out_png.exists():                  # no Pillow: don't leave an older PNG beside the new SVG
        out_png.unlink()
    return {"svg": out_svg, "png": made}
