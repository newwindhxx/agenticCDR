from __future__ import annotations

import subprocess

import pytest

from agentic_cdr.downloader import download_one


def test_aria2_resumes_partial_and_promotes_file(tmp_path, monkeypatch):
    destination = tmp_path / "sample.jsonl.gz"
    partial = tmp_path / "sample.jsonl.gz.part"
    partial.write_bytes(b"old")

    monkeypatch.setattr("agentic_cdr.downloader.shutil.which", lambda _: "/usr/bin/aria2c")

    def fake_run(command, check):
        assert check is True
        assert "--continue=true" in command
        assert "--max-connection-per-server=8" in command
        assert f"--out={partial.name}" in command
        partial.write_bytes(b"done")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("agentic_cdr.downloader.subprocess.run", fake_run)
    result = download_one(
        "https://example.test/sample.jsonl.gz",
        destination,
        expected_bytes=4,
        connections=8,
    )

    assert result == destination
    assert destination.read_bytes() == b"done"
    assert not partial.exists()


@pytest.mark.parametrize("connections", [0, 17])
def test_connection_count_is_bounded(tmp_path, connections):
    with pytest.raises(ValueError):
        download_one(
            "https://example.test/sample.jsonl.gz",
            tmp_path / "sample.jsonl.gz",
            connections=connections,
        )
