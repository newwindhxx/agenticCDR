from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from .config import configured_path


class DownloadError(RuntimeError):
    pass


def _verify_and_promote(
    partial: Path, destination: Path, expected_bytes: int | None
) -> Path:
    if not partial.exists():
        raise DownloadError(f"Downloader did not create the expected file: {partial}")
    actual = partial.stat().st_size
    if expected_bytes is not None and actual != expected_bytes:
        raise DownloadError(
            f"Incomplete download: {partial} ({actual:,} != {expected_bytes:,}); rerun to resume"
        )
    os.replace(partial, destination)
    return destination


def _download_with_aria2(
    url: str,
    partial: Path,
    connections: int,
) -> None:
    aria2 = shutil.which("aria2c")
    if aria2 is None:
        raise DownloadError(
            "aria2c is not installed; install it with apt-get install aria2 "
            "or run with --connections 1"
        )
    command = [
        aria2,
        "--continue=true",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        "--min-split-size=4M",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--max-tries=10",
        "--retry-wait=5",
        "--connect-timeout=30",
        "--timeout=120",
        "--summary-interval=5",
        "--console-log-level=warn",
        f"--dir={partial.parent}",
        f"--out={partial.name}",
        url,
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise DownloadError(
            f"aria2 download stopped with exit code {exc.returncode}; rerun the same command to resume"
        ) from exc


def _download_with_requests(url: str, partial: Path, expected_bytes: int | None) -> None:
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
            desc=partial.name.removesuffix(".part"),
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))
            handle.flush()
            os.fsync(handle.fileno())


def download_one(
    url: str,
    destination: Path,
    expected_bytes: int | None = None,
    connections: int = 16,
) -> Path:
    if not 1 <= connections <= 16:
        raise ValueError("connections must be between 1 and 16")
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
    if connections > 1:
        print(
            f"Downloading {destination.name} with aria2 ({connections} connections); "
            f"existing .part progress will be resumed."
        )
        _download_with_aria2(url, partial, connections)
    else:
        _download_with_requests(url, partial, expected_bytes)
    return _verify_and_promote(partial, destination, expected_bytes)


def download_all(config: dict[str, Any], connections: int = 16) -> list[Path]:
    raw_dir = configured_path(config, "raw_dir")
    outputs: list[Path] = []
    for entry in config.get("downloads", []):
        outputs.append(
            download_one(
                str(entry["url"]),
                raw_dir / str(entry["name"]),
                int(entry["expected_bytes"]) if entry.get("expected_bytes") else None,
                connections=connections,
            )
        )
    if not outputs:
        raise DownloadError("No downloads configured")
    return outputs
