import subprocess

import pytest

from turbomlx.exceptions import UnsupportedRuntimeVersionError
from turbomlx.mlx_runtime import availability


def test_mlx_runtime_available_returns_false_when_probe_fails(monkeypatch):
    availability.mlx_runtime_available.cache_clear()

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=1)

    monkeypatch.setattr(availability.subprocess, "run", fake_run)
    assert availability.mlx_runtime_available() is False


def test_mlx_runtime_available_returns_true_when_probe_succeeds(monkeypatch):
    availability.mlx_runtime_available.cache_clear()

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0)

    monkeypatch.setattr(availability.subprocess, "run", fake_run)
    assert availability.mlx_runtime_available() is True


def test_runtime_version_guard_accepts_tested_minor(monkeypatch):
    monkeypatch.setattr(
        availability.metadata,
        "version",
        lambda name: {"mlx": "0.31.2", "mlx-lm": "0.31.3"}[name],
    )
    assert availability.ensure_supported_runtime_versions() == {"mlx": "0.31.2", "mlx-lm": "0.31.3"}


def test_runtime_version_guard_rejects_out_of_range_versions(monkeypatch):
    monkeypatch.setattr(
        availability.metadata,
        "version",
        lambda name: {"mlx": "0.30.9", "mlx-lm": "0.31.3"}[name],
    )
    with pytest.raises(UnsupportedRuntimeVersionError):
        availability.ensure_supported_runtime_versions()


def test_runtime_version_guard_reports_independent_mlx_lm_floor(monkeypatch):
    monkeypatch.setattr(
        availability.metadata,
        "version",
        lambda name: {"mlx": "0.31.2", "mlx-lm": "0.31.0"}[name],
    )
    with pytest.raises(UnsupportedRuntimeVersionError, match="mlx-lm==0.31.0"):
        availability.ensure_supported_runtime_versions()
