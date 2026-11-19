"""Refresh curves for workers started before live output was added."""
import argparse
import fcntl
from pathlib import Path
import time

from export_multimodal import APP

import torch

from multimodal.live_artifacts import publish_curves
from multimodal.run_state import file_hash, read_json


def active(run):
    lock = run / ".lock"
    if not lock.is_file():
        return False
    with lock.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def refresh(root, plots, published):
    count = 0
    for path in root.glob("*/executions/*/.lock"):
        run = path.parent
        if not active(run):
            continue
        count += 1
        manifest = read_json(run.parents[1] / "manifest.json")
        info = manifest["identity"]
        if info["model"] not in ("M1", "M2", "M3"):
            continue
        attempts = sorted((run / "attempts").glob("*"), key=lambda p: p.stat().st_mtime_ns)
        if not attempts:
            continue
        attempt = attempts[-1]
        if (attempt / "live").is_dir():
            continue
        snapshot = max((attempt / "resume").glob("epoch-*.pt"), default=None)
        if snapshot is None or snapshot in published:
            continue
        try:
            if file_hash(snapshot) != snapshot.stem.rsplit("-", 1)[-1]:
                raise ValueError("Epoch checkpoint checksum mismatch")
            state = torch.load(snapshot, map_location="cpu", weights_only=True, mmap=True)
        except FileNotFoundError:
            continue  # Writer replaced the previous epoch while it was being read.
        label = f"{info['model']}_seed{info['seed']}_fold{info['fold']}"
        directory = plots / label / run.name
        publish_curves(directory, state["history"], state["best_epoch"],
                       model=info["model"], seed=info["seed"], fold=info["fold"],
                       execution_id=run.name, training=manifest["resolved_config"]["settings"]["training"])
        published.add(snapshot)
        print(f"{label} · epoch {len(state['history'])} · {directory}", flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    outputs = APP / "outputs"
    published = set()
    idle = 0
    while True:
        count = sum(refresh(root, outputs / "multimodal_live_figures", published)
                    for root in outputs.glob("multimodal_runs*"))
        idle = 0 if count else idle + 1
        if args.once or idle >= 12:
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
