"""Terminal progress for the multimodal training lifecycle."""
import os
import sys

from tqdm import tqdm


def progress_enabled():
    return (
        sys.stderr.isatty()
        and os.environ.get("TERM") != "dumb"
        and os.environ.get("TQDM_DISABLE", "").lower() not in {"1", "true", "yes", "on"}
    )


def progress_bar(description, *, total, unit, leave=True):
    return tqdm(
        desc=description, total=total, unit=unit, leave=leave,
        file=sys.stderr, disable=not progress_enabled(), dynamic_ncols=True,
        bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}|{postfix}" if total is not None else None,
    )


def batch_count(loader):
    try:
        return len(loader)
    except TypeError:
        return None
