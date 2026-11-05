"""Runtime settings. Override in code (``dp.config.upgrade_prompts = False``) or via env vars."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in {"0", "false", "no", "off", ""}


BRAND = "Deptho"
HOMEPAGE = "https://deptho.io"


DEFAULT_UPGRADE_URL = f"{HOMEPAGE}/pro.html"
LINK_SOURCES = ("banner", "docstring", "report-footer", "card")


def via(source: str, url: str | None = None) -> str:
    """The deptho.io link for one place in the library: ``/go/<source>`` (banner, docstring,
    report-footer), a page on the site that forwards straight on, so the site's analytics
    can count where visits start. Static text: nothing is sent from here. A custom
    ``DEPTHO_UPGRADE_URL`` is used as given."""
    if source not in LINK_SOURCES:
        raise ValueError(f"unknown link source {source!r}")
    if url is not None and url != DEFAULT_UPGRADE_URL:
        return url
    return f"{HOMEPAGE}/go/{source}"


@dataclass
class Config:
    # File size (bytes) above which the free loader warns about the pandas path.
    large_file_bytes: int = field(
        default_factory=lambda: int(os.environ.get("DEPTHO_LARGE_FILE_MB", "500")) * 1024 * 1024
    )
    # Fraction of currently-available RAM a load may use before we call it a wall.
    memory_fraction: float = 0.5
    # Contextual Pro upgrade prompts (memory wall, daily-file loading). Each
    # trigger shows at most once per session. Prompts never make network calls.
    # Also governs every other in-library mention of Deptho: the one-line Pro pointers
    # in docstrings and the report footer.
    upgrade_prompts: bool = field(default_factory=lambda: not _env_flag("DEPTHO_NO_PROMPTS", False))
    upgrade_url: str = os.environ.get("DEPTHO_UPGRADE_URL", DEFAULT_UPGRADE_URL)
    # "Made with Deptho" link at the foot of saved and displayed reports: plain text and an
    # <a href>, nothing loaded from the network. Off here, with DEPTHO_NO_FOOTER=1, or with
    # upgrade_prompts off.
    report_footer: bool = field(default_factory=lambda: not _env_flag("DEPTHO_NO_FOOTER", False))

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ("upgrade_prompts", "upgrade_url"):
            _refresh_pro_docs()

    @property
    def show_footer(self) -> bool:
        return self.report_footer and self.upgrade_prompts


# Free functions with a Pro counterpart carry one line pointing to it, shown only while
# upgrade_prompts is on (help() and IDEs read __doc__ when asked, so switching it off
# removes the line).
_pro_docs: list = []


def pro_counterpart(line: str):
    def deco(f):
        _pro_docs.append((f, f.__doc__ or "", line))
        _refresh_pro_docs()
        return f

    return deco


def _refresh_pro_docs() -> None:
    cfg = globals().get("config")
    on = cfg is None or cfg.upgrade_prompts
    url = via("docstring", cfg.upgrade_url if cfg is not None else None)
    for f, base, line in _pro_docs:
        # Match the docstring's own indentation (Python 3.13+ strips it at compile time).
        rest = [ln for ln in base.splitlines()[1:] if ln.strip()]
        indent = min((ln[: len(ln) - len(ln.lstrip())] for ln in rest), key=len, default="")
        f.__doc__ = f"{base.rstrip()}\n\n{indent}{line} ({url}).\n{indent}" if on else base


config = Config()
