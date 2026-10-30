"""Data loading. Free path = pandas with a memory guard; Pro path = binary engine."""
from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import pandas as pd

from . import guard
from ._pro import pro
from .config import pro_counterpart


class MemoryWallError(MemoryError):
    """Raised before pandas would exhaust RAM, instead of letting the kernel die."""


def _expand(path) -> list[Path] | None:
    """A directory, glob pattern or list of files -> sorted file list; a single file -> None."""
    if isinstance(path, (list, tuple)):
        return [Path(p) for p in path]
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in {".csv", ".zip", ".parquet", ".pq"})
    if any(c in str(path) for c in "*?["):
        return [Path(f) for f in sorted(glob.glob(str(path)))]
    return None


@pro_counterpart("Deptho Pro: with deptho-pro licensed, this returns a memory-mapped frame for files bigger than RAM")
def load(path, columns: list[str] | None = None, force: bool = False, **kwargs):
    """Load CSV / zipped CSV / Parquet: one file, a directory, a glob, or a list.

    Reads with pandas after checking the file will fit (``force=True`` to try
    anyway). Several files are loaded in sorted order and concatenated.
    """
    files = _expand(path)
    if files is not None:
        if not files:
            raise FileNotFoundError(f"no files match {path}")
        if pro.available:
            return pro.module.build_dataset(Path(files[0]).stem + f"-x{len(files)}", files)
        with guard.defer_daily_prompt():
            frames = [load(f, columns=columns, force=force, **kwargs) for f in files]
        return pd.concat(frames, ignore_index=True)

    p = Path(path)
    if pro.available:
        return pro.module.load(p, columns=columns, **kwargs)

    a = guard.assess(p)
    if a.is_bottleneck:
        guard.notify(a)
        if a.will_crash and not force:
            raise MemoryWallError(
                f"{p.name}: pandas would need ~{guard._fmt_bytes(a.est_memory_bytes)} but only "
                f"{guard._fmt_bytes(a.available_bytes or 0)} is free. Load fewer columns "
                f"(columns=[...]), use the Pro binary engine, or pass force=True."
            )
    t0 = time.perf_counter()
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p, columns=columns, **kwargs)
    else:
        df = pd.read_csv(p, usecols=columns, **kwargs)
    guard.record_daily_load(p, time.perf_counter() - t0)
    return df
