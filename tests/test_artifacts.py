from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from llm_compression.artifacts import (
    append_jsonl,
    atomic_json_dump,
    atomic_torch_save,
    load_json,
    load_torch_checkpoint,
    read_jsonl,
    write_jsonl,
)


class ArtifactTests(unittest.TestCase):
    def test_atomic_json_creates_parent_and_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "manifest.json"
            atomic_json_dump({"version": 1}, path)
            self.assertEqual(load_json(path), {"version": 1})
            atomic_json_dump({"version": 2, "name": "probe"}, path)
            self.assertEqual(load_json(path)["version"], 2)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_torch_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            value = {"step": 4, "weight": torch.arange(3)}
            atomic_torch_save(value, path)
            loaded = load_torch_checkpoint(path)
            self.assertEqual(loaded["step"], 4)
            self.assertTrue(torch.equal(loaded["weight"], value["weight"]))

    def test_jsonl_replace_append_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            write_jsonl([{"step": 1}, {"step": 2}], path)
            append_jsonl({"step": 3, "loss": 0.25}, path)
            self.assertEqual(
                read_jsonl(path),
                [{"step": 1}, {"step": 2}, {"loss": 0.25, "step": 3}],
            )

    def test_json_rejects_nonstandard_nan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            with self.assertRaises(ValueError):
                atomic_json_dump({"loss": float("nan")}, path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
