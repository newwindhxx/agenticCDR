from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from .config import configured_path


class DownloadError(RuntimeError):
    pass


def download_one(url: str, destination: Path, expected_bytes: int | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        size = destination.stat().st_size
        if expected_bytes is None or size == expected_bytes:
            print(f"Already complete: {destination} ({size:,} bytes)")
            return destination
        raise DownloadError(
            f"Existing final file has unexpected size: {destination} ({size:,} != {expected_bytes:,})"
        )

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
            mode = "wb"
        else:
            mode = "ab" if offset else "wb"
        content_length = int(response.headers.get("content-length", 0))
        total = expected_bytes or (offset + content_length if content_length else None)
        with partial.open(mode) as handle, tqdm(
            total=total,
            initial=offset,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))
            handle.flush()
            os.fsync(handle.fileno())

    actual = partial.stat().st_size
    if expected_bytes is not None and actual != expected_bytes:
        raise DownloadError(
            f"Incomplete download: {partial} ({actual:,} != {expected_bytes:,}); rerun to resume"
        )
    os.replace(partial, destination)
    return destination


def download_all(config: dict[str, Any]) -> list[Path]:
    raw_dir = configured_path(config, "raw_dir")
    outputs: list[Path] = []
    for entry in config.get("downloads", []):
        outputs.append(
            download_one(
                str(entry["url"]),
                raw_dir / str(entry["name"]),
                int(entry["expected_bytes"]) if entry.get("expected_bytes") else None,
            )
        )
    if not outputs:
        raise DownloadError("No downloads configured")
    return outputs
