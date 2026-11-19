"""Crash-safe helpers for experiment metadata, logs, and checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch


def _prepare_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sync_directory(directory: Path) -> None:
    """Best-effort directory fsync so a rename survives a machine failure."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write(path: str | Path, writer: Any) -> Path:
    destination = Path(path)
    _prepare_parent(destination)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        _sync_directory(destination.parent)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return destination


def atomic_json_dump(
    value: Any,
    path: str | Path,
    *,
    indent: int | None = 2,
) -> Path:
    """Serialize JSON to a sibling temporary file and atomically replace ``path``."""

    def write(handle: Any) -> None:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=True,
        )
        if indent is not None:
            encoded += "\n"
        handle.write(encoded.encode("utf-8"))

    return _atomic_write(path, write)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_torch_save(value: Any, path: str | Path) -> Path:
    """Atomically save a PyTorch-serializable checkpoint."""

    return _atomic_write(path, lambda handle: torch.save(value, handle))


def load_torch_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = True,
) -> Any:
    """Load a checkpoint, defaulting to PyTorch's safer weights-only unpickler."""

    return torch.load(
        Path(path),
        map_location=map_location,
        weights_only=weights_only,
    )


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Atomically replace ``path`` with newline-delimited JSON records."""

    def write(handle: Any) -> None:
        for record in records:
            encoded = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if "\n" in encoded or "\r" in encoded:
                raise ValueError("serialized JSONL records must occupy exactly one line")
            handle.write(encoded.encode("utf-8") + b"\n")

    return _atomic_write(path, write)


def append_jsonl(record: Mapping[str, Any], path: str | Path, *, fsync: bool = True) -> Path:
    """Append one JSONL record using a single POSIX ``O_APPEND`` write."""

    destination = Path(path)
    _prepare_parent(destination)
    encoded = json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"short JSONL append: wrote {written} of {len(encoded)} bytes")
        if fsync:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def iter_jsonl(path: str | Path) -> Iterator[Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc


def read_jsonl(path: str | Path) -> list[Any]:
    return list(iter_jsonl(path))


__all__ = [
    "append_jsonl",
    "atomic_json_dump",
    "atomic_torch_save",
    "iter_jsonl",
    "load_json",
    "load_torch_checkpoint",
    "read_jsonl",
    "write_jsonl",
]
