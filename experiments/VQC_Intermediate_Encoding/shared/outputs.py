"""Helpers for keeping each experiment execution in a separate results folder."""
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"


def create_run_directory(num_layers=None, encoding_method=None):
    """Create a unique directory for one complete execution."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    parts = ["run"]
    if num_layers is not None:
        parts.append(f"layers_{num_layers}")
    if encoding_method:
        parts.append(str(encoding_method).lower())
    parts.append(timestamp)
    run_dir = RESULTS_ROOT / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
