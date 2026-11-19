import json
import tempfile
import unittest
from pathlib import Path

import torch
try:
    from safetensors.torch import save_file

    from llm_compression.data import (
        cached_batch,
        load_boundary_cache,
        prepare_token_shards,
    )

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


@unittest.skipUnless(HAS_SAFETENSORS, "safetensors is not installed")
class BoundaryBatchTests(unittest.TestCase):
    def test_completed_token_cache_rejects_changed_data_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "complete": True,
                "dataset": "dataset",
                "dataset_config": "subset",
                "dataset_revision": "data-rev",
                "dataset_split": "train",
                "tokenizer": "tokenizer",
                "tokenizer_revision": "model-rev",
                "sequence_length": 512,
                "seed": 7,
                "splits": {"probe": {"num_sequences": 2, "shards": []}},
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            common = {
                "output_dir": root,
                "tokenizer_name": "tokenizer",
                "tokenizer_revision": "model-rev",
                "dataset_name": "dataset",
                "dataset_config": "subset",
                "dataset_split": "train",
                "text_column": "text",
                "split_sequences": {"probe": 2},
                "sequence_length": 512,
                "shard_sequences": 2,
                "seed": 7,
                "dataset_revision": "data-rev",
            }
            self.assertEqual(prepare_token_shards(**common), root / "manifest.json")
            with self.assertRaisesRegex(ValueError, "does not match"):
                prepare_token_shards(**{**common, "seed": 8})
            with self.assertRaisesRegex(ValueError, "split counts"):
                prepare_token_shards(
                    **{**common, "split_sequences": {"probe": 3}}
                )

    def test_batch_can_span_shards_smaller_than_batch_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = []
            for shard_index in range(2):
                start = shard_index * 2
                ids = torch.arange(start, start + 2).view(2, 1)
                path = root / f"shard-{shard_index}.safetensors"
                save_file(
                    {
                        "input_ids": ids,
                        "attention_mask": torch.ones_like(ids),
                        "h_000": ids.float().unsqueeze(-1),
                    },
                    path,
                )
                shards.append({"path": path.name, "num_sequences": 2})
            manifest = {
                "num_layers": 1,
                "splits": {"probe": {"num_sequences": 4, "shards": shards}},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            cache = load_boundary_cache(manifest_path, "probe")
            batch = cached_batch(
                cache,
                boundaries=(0,),
                batch_size=7,
                seed=3,
                step=2,
                device="cpu",
                return_global_indices=True,
            )
            self.assertEqual(batch["input_ids"].shape, (7, 1))
            torch.testing.assert_close(
                batch["input_ids"].squeeze(1), batch["__global_indices__"]
            )
            torch.testing.assert_close(
                batch["h_000"].squeeze(-1).long(), batch["input_ids"]
            )


if __name__ == "__main__":
    unittest.main()
