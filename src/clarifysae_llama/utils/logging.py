from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from clarifysae_llama.utils.io import append_jsonl


def log_run(path: str | Path, payload: dict) -> None:
    row = dict(payload)
    row['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
    append_jsonl(path, row)
