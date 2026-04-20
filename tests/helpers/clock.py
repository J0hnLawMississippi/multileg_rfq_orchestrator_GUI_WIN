from __future__ import annotations


def freeze_time(monkeypatch, module, epoch_seconds: float = 1700000000.0):
    monkeypatch.setattr(module.time, "time", lambda: epoch_seconds)


def freeze_monotonic(monkeypatch, module, value: float = 1000.0):
    monkeypatch.setattr(module.time, "monotonic", lambda: value)
