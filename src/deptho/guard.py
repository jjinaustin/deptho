"""Memory-wall detection for the free (pandas) loading path.

Estimates are measured, not guessed: we parse a small head sample of the
actual file with pandas on the user's machine, time it, measure its
in-memory footprint, and extrapolate linearly by bytes.
"""
from __future__ import annotations

import html
import io
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import config, tagged

_SAMPLE_BYTES = 8 * 1024 * 1024
# pandas' peak RSS while parsing is ~2x the finished DataFrame (parser buffers
# plus the result). Measured: 30M-row / 2 GB tick CSV, frame 1.4 GB, peak 2.8 GB.
_PARSE_PEAK = 2.0
_shown: set[str] = set()


@dataclass
class LoadAssessment:
    path: str
    file_bytes: int
    est_memory_bytes: int
    est_seconds: float
    available_bytes: int | None
    sample_rows: int

    @property
    def is_large(self) -> bool:
        return self.file_bytes >= config.large_file_bytes

    @property
    def exceeds_memory(self) -> bool:
        if self.available_bytes is None:
            return False
        return self.est_memory_bytes > self.available_bytes * config.memory_fraction

    @property
    def is_bottleneck(self) -> bool:
        return self.is_large or self.exceeds_memory

    @property
    def will_crash(self) -> bool:
        return self.available_bytes is not None and self.est_memory_bytes > self.available_bytes * 0.9


def available_memory() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def _sample_csv(path: Path, file_bytes: int) -> tuple[float, int, int, float]:
    """Parse the head of a CSV; return (seconds, mem_bytes, rows, bytes_parsed).

    ``file_bytes`` is the uncompressed size; zipped CSVs are sampled in-stream.
    """
    if path.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as zf:
            with zf.open(zf.infolist()[0]) as fh:
                head = fh.read(min(_SAMPLE_BYTES, file_bytes))
    else:
        with open(path, "rb") as fh:
            head = fh.read(min(_SAMPLE_BYTES, file_bytes))
    if len(head) < file_bytes:
        head = head[: head.rfind(b"\n") + 1] or head
    t0 = time.perf_counter()
    df = pd.read_csv(io.BytesIO(head))
    dt = time.perf_counter() - t0
    return dt, int(df.memory_usage(deep=True).sum()), len(df), float(len(head))


def _sample_parquet(path: Path, file_bytes: int) -> tuple[float, int, int, float]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    t0 = time.perf_counter()
    df = pf.read_row_group(0).to_pandas()
    dt = time.perf_counter() - t0
    # Treat the sampled row group as its row-share of the file's bytes.
    frac = len(df) / max(1, pf.metadata.num_rows)
    return dt, int(df.memory_usage(deep=True).sum()), len(df), max(1.0, file_bytes * frac)


def assess(path: str | os.PathLike) -> LoadAssessment:
    p = Path(path)
    size = p.stat().st_size
    suffix = p.suffix.lower()
    raw = size
    try:
        if suffix == ".zip":
            import zipfile

            with zipfile.ZipFile(p) as zf:
                raw = zf.infolist()[0].file_size
        if suffix in {".parquet", ".pq"}:
            secs, mem, rows, parsed = _sample_parquet(p, size)
            scale = size / max(parsed, 1.0)
        else:
            secs, mem, rows, parsed = _sample_csv(p, raw)
            scale = raw / max(parsed, 1.0)
    except Exception:
        # Rule of thumb when sampling fails: numeric CSV frames are ~0.75x the text size.
        secs, mem, rows, scale = raw / 100e6, int(raw * 0.75), 0, 1.0
    return LoadAssessment(
        path=str(p),
        file_bytes=size,
        est_memory_bytes=int(mem * scale * _PARSE_PEAK),
        est_seconds=float(secs * scale),
        available_bytes=available_memory(),
        sample_rows=rows,
    )


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_secs(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ in {"ZMQInteractiveShell", "Shell"}
    except Exception:
        return False


_HIDE = "dp.config.upgrade_prompts = False"


def notice_text(a: LoadAssessment) -> str:
    name = Path(a.path).name
    lines = [f"Deptho: pandas memory bottleneck detected for {name} ({_fmt_bytes(a.file_bytes)})."]
    mem = f"~{_fmt_bytes(a.est_memory_bytes)} of RAM"
    if a.available_bytes:
        mem += f" (you have {_fmt_bytes(a.available_bytes)} free)"
    lines.append(f"  Loading it with pandas peaks at {mem} and takes ~{_fmt_secs(a.est_seconds)}"
                 " (estimated from a sample of this file).")
    if a.will_crash:
        lines.append("  That is more memory than this machine has available; the load would likely crash.")
    lines.append(
        "  Deptho Pro converts the file once to a columnar binary store and memory-maps it,"
        " so later loads use little RAM, all on this machine."
    )
    lines.append(f"  Upgrade: {tagged(config.upgrade_url, 'banner')}   (hide: {_HIDE})")
    return "\n".join(lines)


def _card(title: str, detail: str, body: str, extra: str = "") -> str:
    """Notebook banner. Brings its own layer-1 background, so it reads the same on
    light and dark notebook themes. Styled entirely from deptho._theme tokens."""
    from ._theme import FONT_DISPLAY, FONT_MONO, c, mark_svg

    url = html.escape(tagged(config.upgrade_url, "banner"))
    code = f"font-family:{FONT_MONO};font-size:0.9em;color:{c('ink')}"
    body = body.replace("<code>", f'<code style="{code}">')
    return f"""
<div class="dp-banner" role="note" style="position:relative;box-sizing:border-box;font-family:{FONT_DISPLAY};
 background:{c('layer-1')};color:{c('ink')};border:1px solid {c('contour')};border-radius:16px;
 padding:18px 20px;margin:8px 0;max-width:820px;display:flex;gap:14px;align-items:flex-start;line-height:1.45">
  <div style="flex:0 0 22px;padding-top:2px">{mark_svg(22)}</div>
  <div style="flex:1;min-width:0;padding-right:28px">
    <div style="font-size:16px;font-weight:600;color:{c('ink')}">{title}</div>
    <div style="font-family:{FONT_MONO};font-size:13px;color:{c('ink-muted')};margin-top:4px;
     font-variant-numeric:tabular-nums">{detail}</div>{extra}
    <div style="font-size:15px;color:{c('ink-muted')};margin-top:8px">{body}</div>
    <div style="margin-top:12px"><a href="{url}" target="_blank" rel="noopener"
     style="color:{c('signal')};font-weight:600;text-decoration:none">See Deptho Pro &rarr;</a></div>
    <div style="font-family:{FONT_MONO};font-size:12px;color:{c('ink-muted')};margin-top:10px">Turn off: {_HIDE}</div>
  </div>
  <button type="button" aria-label="Dismiss" title="Dismiss"
   onclick="this.closest('.dp-banner').remove()"
   style="position:absolute;top:10px;right:10px;width:32px;height:32px;border:0;border-radius:10px;cursor:pointer;
   background:transparent;color:{c('ink-muted')};font-size:20px;line-height:32px;padding:0">&times;</button>
</div>"""


def _notice_html(a: LoadAssessment) -> str:
    name = html.escape(Path(a.path).name)
    avail = f" &middot; {_fmt_bytes(a.available_bytes)} free" if a.available_bytes else ""
    from ._theme import c

    crash = (f"<div style='margin-top:6px;color:{c('loss')}'>More than this machine has available: "
             "the load would likely crash.</div>" if a.will_crash else "")
    return _card(
        f"&#9888; Pandas memory bottleneck detected: {name}",
        f"{_fmt_bytes(a.file_bytes)} on disk &middot; pandas peaks at ~{_fmt_bytes(a.est_memory_bytes)} RAM{avail}"
        f" &middot; parse ~{_fmt_secs(a.est_seconds)} (estimated from a sample of this file)",
        "Deptho Pro converts the file once to a columnar binary store and memory-maps it: later loads "
        "use little RAM. Nothing leaves this machine.",
        crash,
    )


def _show(trigger: str, text: str, html_card: str) -> bool:
    """Show one prompt per trigger per session; never with Pro active or prompts off."""
    from ._pro import pro

    if not config.upgrade_prompts or trigger in _shown or pro.available:
        return False
    _shown.add(trigger)
    if _in_notebook():
        from IPython.display import HTML, display

        display(HTML(html_card))
    else:
        print(text, file=sys.stderr)
    return True


def notify(a: LoadAssessment) -> bool:
    """Memory-wall prompt. Returns True if shown."""
    return _show("memory_wall", notice_text(a), _notice_html(a))


# --- Daily exchange files loaded one at a time --------------------------------

DAILY_FILE = re.compile(r"^(?P<sym>[A-Z0-9]+)-(?P<kind>aggTrades|trades)-(?P<day>\d{4}-\d{2}-\d{2})\.(zip|csv)$")
_daily: dict[tuple[str, str], dict[str, float]] = {}


def record_daily_load(path: str | os.PathLike, seconds: float) -> bool:
    """Note a measured load of a daily Binance trade file. From the second
    distinct day of the same symbol and kind, prompt once (``daily_files``)."""
    m = DAILY_FILE.match(Path(path).name)
    if not m:
        return False
    days = _daily.setdefault((m["sym"], m["kind"]), {})
    days[m["day"]] = seconds
    if _deferring:
        return False
    return _prompt_daily(m["sym"], m["kind"])


_deferring = 0


class defer_daily_prompt:
    """Hold the daily-files prompt until a multi-file load finishes, so it
    reports the whole load (all files, total measured time) once."""

    def __enter__(self):
        global _deferring
        _deferring += 1

    def __exit__(self, *exc):
        global _deferring
        _deferring -= 1
        if not _deferring and exc[0] is None:
            for sym, kind in list(_daily):
                if _prompt_daily(sym, kind):
                    break


def _prompt_daily(sym: str, kind: str) -> bool:
    days = _daily.get((sym, kind), {})
    if len(days) < 2:
        return False
    m = {"sym": sym, "kind": kind}
    n, total = len(days), sum(days.values())
    text = (f"Deptho: loading daily files one at a time? deptho_pro.sync keeps them as one growing "
            f"dataset. Measured load here: {total:.1f} s for {n} daily {m['sym']} {m['kind']} files.\n"
            f"  Upgrade: {tagged(config.upgrade_url, 'banner')}   (hide: {_HIDE})")
    card = _card(
        "Loading daily files one at a time?",
        f"Measured load here: {total:.1f} s for {n} daily {html.escape(m['sym'])} {m['kind']} files",
        "<code>deptho_pro.sync</code> keeps them as one growing, memory-mapped dataset: each new day is "
        "converted once and appended, and the whole range opens without re-parsing.",
    )
    return _show("daily_files", text, card)
