"""Deptho design tokens: the single Python source of colors and type.

The site defines the same tokens as CSS custom properties in site/style.css;
a test keeps the two in sync and blocks raw hex colors anywhere else.
Dark only for now.
"""
from __future__ import annotations

COLORS = {
    "abyss": "#061018",          # page ground
    "layer-1": "#0A1B27",        # panels, code blocks, report cards
    "layer-2": "#0C2130",        # shallowest data column; hover on layer-1
    "layer-3": "#0F2738",        # badges, secondary buttons, mid data column
    "layer-4": "#13304A",        # deepest step: the hardest condition
    "contour": "#1B4560",        # contour motif, hairline borders; never text
    "ink": "#E6F1F5",            # primary text
    "ink-muted": "#8AA4B2",      # secondary text, captions, table headers
    "ink-faint": "#6F8C9B",      # failed cells on layer-2 only
    "sky": "#9CC3D9",            # code strings, second series, mark middle ring
    "signal": "#4FD1C5",         # the one accent
    "on-signal": "#04141A",      # text on a signal fill
    "series-2": "#7AA7FF",       # drawdown line, third series
    "drawdown-fill": "#1E4A6A",  # underwater area fill
    "loss": "#F2994A",           # losses, costs, fees
    "focus": "#9CC3D9",          # focus ring
}

# Single-quoted family names so the stacks can sit inside double-quoted HTML style
# attributes (report header, notebook banner) as well as Plotly font settings.
FONT_DISPLAY = "'Space Grotesk', 'Helvetica Neue', Arial, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace"


def c(token: str) -> str:
    return COLORS[token]


def rgba(token: str, alpha: float) -> str:
    h = COLORS[token].lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def mark_svg(size: int = 28) -> str:
    """The default Deptho mark (for dark grounds), inline, built from tokens."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 64 64" '
        f'aria-hidden="true" focusable="false"><g fill="none" stroke-width="5">'
        f'<rect x="4" y="4" width="56" height="56" rx="16" stroke="{c("ink")}"/>'
        f'<rect x="17" y="19" width="36" height="36" rx="10" stroke="{c("sky")}"/></g>'
        f'<rect x="30" y="34" width="16" height="16" rx="4" fill="{c("signal")}"/></svg>'
    )


def lockup_html(mark: int = 28, text_px: int = 22) -> str:
    """Mark + 'Deptho' wordmark, inline styles only (for offline report and notebook output)."""
    from .config import BRAND

    return (
        f'<span style="display:inline-flex;align-items:center;gap:12px">{mark_svg(mark)}'
        f'<span style="font-family:{FONT_DISPLAY};font-weight:600;font-size:{text_px}px;letter-spacing:-0.02em;'
        f'color:{c("ink")}">{BRAND}</span></span>'
    )
