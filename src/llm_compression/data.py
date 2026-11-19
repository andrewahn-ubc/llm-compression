"""Reproducible token shards and compact teacher-boundary caches."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset

from .modeling import load_causal_lm, load_tokenizer, teacher_boundaries


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _check_manifest_fields(
    path: Path,
    manifest: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    mismatches = {
        key: {"cached": manifest.get(key), "requested": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Completed data cache {path} does not match this experiment: {mismatches}. "
            "Use a separate paths.data_cache_dir for the new data specification."
        )


def prepare_token_shards(
    *,
    output_dir: str | Path,
    tokenizer_name: str,
    tokenizer_revision: str | None,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    text_column: str,
    split_sequences: Mapping[str, int],
    sequence_length: int,
    shard_sequences: int,
    seed: int,
    dataset_revision: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Stream text once and materialize disjoint fixed-length token splits.

    Documents are separated by EOS, concatenated, then cut into non-overlapping
    sequences exactly as specified in the proposal.  Split order is recorded in
    the manifest, making the slices disjoint and reproducible.
    """

    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("complete"):
            _check_manifest_fields(
                manifest_path,
                existing,
                {
                    "dataset": dataset_name,
                    "dataset_config": dataset_config,
                    "dataset_revision": dataset_revision,
                    "dataset_split": dataset_split,
                    "tokenizer": tokenizer_name,
                    "tokenizer_revision": tokenizer_revision,
                    "sequence_length": sequence_length,
                    "seed": seed,
                },
            )
            cached_counts = {
                name: int(info["num_sequences"])
                for name, info in existing.get("splits", {}).items()
            }
            requested_counts = {
                name: int(count) for name, count in split_sequences.items()
            }
            if cached_counts != requested_counts:
                raise ValueError(
                    f"Completed data cache {manifest_path} has split counts "
                    f"{cached_counts}, requested {requested_counts}; use a separate "
                    "paths.data_cache_dir"
                )
            return manifest_path

    from datasets import load_dataset

    tokenizer = load_tokenizer(
        tokenizer_name,
        revision=tokenizer_revision,
        local_files_only=local_files_only,
    )
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
        streaming=True,
        revision=dataset_revision,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    requested = sum(int(value) for value in split_sequences.values())
    if requested <= 0:
        raise ValueError("split_sequences must request at least one sequence")
    split_items = [(name, int(count)) for name, count in split_sequences.items()]
    if any(count <= 0 for _, count in split_items):
        raise ValueError("Every split must request a positive sequence count")
    for split_name, _ in split_items:
        (output / split_name).mkdir(parents=True, exist_ok=True)
    split_manifest: dict[str, Any] = {
        name: {"num_sequences": count, "shards": []} for name, count in split_items
    }
    split_index = 0
    split_written = 0
    shard_index = 0
    pending: list[list[int]] = []

    def flush_pending() -> None:
        nonlocal shard_index
        if not pending:
            return
        split_name = split_items[split_index][0]
        ids = torch.tensor(pending, dtype=torch.long)
        mask = torch.ones_like(ids)
        shard_path = output / split_name / f"shard-{shard_index:05d}.safetensors"
        save_file({"input_ids": ids, "attention_mask": mask}, shard_path)
        split_manifest[split_name]["shards"].append(str(shard_path.relative_to(output)))
        pending.clear()
        shard_index += 1

    generated = 0
    token_buffer: list[int] = []
    eos = tokenizer.eos_token_id
    for example in dataset:
        text = example.get(text_column)
        if not isinstance(text, str) or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_buffer.extend(ids)
        if eos is not None:
            token_buffer.append(eos)
        while len(token_buffer) >= sequence_length and generated < requested:
            pending.append(token_buffer[:sequence_length])
            del token_buffer[:sequence_length]
            generated += 1
            split_written += 1
            if len(pending) >= shard_sequences:
                flush_pending()
            if split_written == split_items[split_index][1]:
                flush_pending()
                split_index += 1
                split_written = 0
                shard_index = 0
                if split_index == len(split_items):
                    break
        if generated >= requested:
            break
    if generated != requested:
        raise RuntimeError(f"Dataset ended after {generated} sequences; requested {requested}")

    manifest = {
        "version": 1,
        "complete": True,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
        "dataset_split": dataset_split,
        "tokenizer": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "sequence_length": sequence_length,
        "seed": seed,
        "splits": split_manifest,
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def prepare_evaluation_shards(
    *,
    output_dir: str | Path,
    tokenizer_name: str,
    tokenizer_revision: str | None,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    text_column: str,
    sequence_length: int,
    shard_sequences: int,
    dataset_revision: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Materialize every complete sequence from an evaluation split."""

    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("complete"):
            _check_manifest_fields(
                manifest_path,
                existing,
                {
                    "dataset": dataset_name,
                    "dataset_config": dataset_config,
                    "dataset_revision": dataset_revision,
                    "dataset_split": dataset_split,
                    "tokenizer": tokenizer_name,
                    "tokenizer_revision": tokenizer_revision,
                    "sequence_length": sequence_length,
                },
            )
            return manifest_path
    from datasets import load_dataset
    tokenizer = load_tokenizer(
        tokenizer_name,
        revision=tokenizer_revision,
        local_files_only=local_files_only,
    )
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
        revision=dataset_revision,
    )
    split_dir = output / dataset_split
    split_dir.mkdir(parents=True, exist_ok=True)
    token_buffer: list[int] = []
    pending: list[list[int]] = []
    shard_names: list[str] = []
    count = 0
    shard_index = 0
    eos = tokenizer.eos_token_id

    def flush() -> None:
        nonlocal shard_index
        if not pending:
            return
        ids = torch.tensor(pending, dtype=torch.long)
        shard_path = split_dir / f"shard-{shard_index:05d}.safetensors"
        save_file({"input_ids": ids, "attention_mask": torch.ones_like(ids)}, shard_path)
        shard_names.append(str(shard_path.relative_to(output)))
        pending.clear()
        shard_index += 1

    for example in dataset:
        text = example.get(text_column)
        if not isinstance(text, str) or not text:
            continue
        token_buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if eos is not None:
            token_buffer.append(eos)
        while len(token_buffer) >= sequence_length:
            pending.append(token_buffer[:sequence_length])
            del token_buffer[:sequence_length]
            count += 1
            if len(pending) >= shard_sequences:
                flush()
    flush()
    if count == 0:
        raise RuntimeError("Evaluation split produced no complete token sequences")
    _write_json(
        manifest_path,
        {
            "version": 1,
            "complete": True,
            "dataset": dataset_name,
            "dataset_config": dataset_config,
            "dataset_revision": dataset_revision,
            "dataset_split": dataset_split,
            "tokenizer": tokenizer_name,
            "tokenizer_revision": tokenizer_revision,
            "sequence_length": sequence_length,
            "splits": {
                dataset_split: {"num_sequences": count, "shards": shard_names}
            },
        },
    )
    return manifest_path


class TokenShardDataset(Dataset):
    """Map-style view over safetensors token shards with a one-shard LRU."""

    def __init__(self, manifest_path: str | Path, split: str, max_open_shards: int = 1):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        try:
            split_info = manifest["splits"][split]
        except KeyError as exc:
            raise KeyError(f"Unknown data split {split!r}") from exc
        self.paths = [self.root / value for value in split_info["shards"]]
        self.lengths: list[int] = []
        for path in self.paths:
            with safe_open(path, framework="pt", device="cpu") as handle:
                self.lengths.append(int(handle.get_slice("input_ids").get_shape()[0]))
        self.offsets: list[int] = []
        total = 0
        for length in self.lengths:
            self.offsets.append(total)
            total += length
        self.total = total
        self.max_open_shards = max_open_shards
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return self.total

    def _location(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.total
        if not 0 <= index < self.total:
            raise IndexError(index)
        # Shard counts are small; this avoids an additional dependency.
        for shard_index in range(len(self.offsets) - 1, -1, -1):
            if index >= self.offsets[shard_index]:
                return shard_index, index - self.offsets[shard_index]
        raise IndexError(index)

    def _shard(self, index: int) -> dict[str, torch.Tensor]:
        if index not in self._cache:
            self._cache[index] = load_file(self.paths[index], device="cpu")
            self._cache.move_to_end(index)
            while len(self._cache) > self.max_open_shards:
                self._cache.popitem(last=False)
        return self._cache[index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index, local = self._location(index)
        shard = self._shard(shard_index)
        return {key: value[local] for key, value in shard.items()}


def deterministic_batch(
    dataset: Dataset,
    *,
    batch_size: int,
    seed: int,
    step: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Sample a resume-stable batch without storing sampler state."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(step) * 1_000_003)
    indices = torch.randint(0, len(dataset), (batch_size,), generator=generator).tolist()
    examples = [dataset[index] for index in indices]
    keys = examples[0].keys()
    return {key: torch.stack([item[key] for item in examples]).to(device) for key in keys}


@dataclass(frozen=True)
class BoundaryCacheInfo:
    root: Path
    split: str
    paths: tuple[Path, ...]
    lengths: tuple[int, ...]
    num_layers: int

    @property
    def num_sequences(self) -> int:
        return sum(self.lengths)


def load_boundary_cache(manifest_path: str | Path, split: str) -> BoundaryCacheInfo:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    info = manifest["splits"][split]
    paths = tuple(path.parent / item["path"] for item in info["shards"])
    lengths = tuple(int(item["num_sequences"]) for item in info["shards"])
    return BoundaryCacheInfo(path.parent, split, paths, lengths, int(manifest["num_layers"]))


def cached_batch(
    cache: BoundaryCacheInfo,
    *,
    boundaries: Sequence[int],
    batch_size: int,
    seed: int,
    step: int,
    device: torch.device | str,
    return_global_indices: bool = False,
) -> dict[str, torch.Tensor]:
    """Read a resume-stable random batch across all boundary-cache shards."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(step) * 1_000_003)
    if cache.num_sequences <= 0:
        raise ValueError("Boundary cache is empty")
    global_indices = torch.randint(
        0, cache.num_sequences, (batch_size,), generator=generator
    )
    keys = ["input_ids", "attention_mask", *(f"h_{index:03d}" for index in boundaries)]
    cpu_values: dict[str, torch.Tensor] = {}
    offset = 0
    for path, length in zip(cache.paths, cache.lengths, strict=True):
        positions = torch.nonzero(
            (global_indices >= offset) & (global_indices < offset + length),
            as_tuple=False,
        ).flatten()
        if positions.numel():
            local_indices = global_indices[positions] - offset
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in keys:
                    selected = handle.get_tensor(key)[local_indices]
                    if key not in cpu_values:
                        cpu_values[key] = torch.empty(
                            (batch_size, *selected.shape[1:]), dtype=selected.dtype
                        )
                    cpu_values[key][positions] = selected
        offset += length
    values = {key: value.to(device) for key, value in cpu_values.items()}
    if return_global_indices:
        values["__global_indices__"] = global_indices
    return values


def iter_cached_batches(
    cache: BoundaryCacheInfo,
    *,
    boundaries: Sequence[int],
    batch_size: int,
    device: torch.device | str,
) -> Iterator[dict[str, torch.Tensor]]:
    keys = ["input_ids", "attention_mask", *(f"h_{index:03d}" for index in boundaries)]
    for path in cache.paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            length = int(handle.get_slice("input_ids").get_shape()[0])
            for offset in range(0, length, batch_size):
                stop = min(offset + batch_size, length)
                yield {key: handle.get_tensor(key)[offset:stop].to(device) for key in keys}


def cache_teacher_boundaries(
    *,
    model_name_or_path: str,
    model_revision: str | None,
    dtype: str,
    token_manifest: str | Path,
    splits: Sequence[str],
    output_dir: str | Path,
    batch_size: int,
    device: str,
    local_files_only: bool,
    attn_implementation: str | None = None,
    cache_dtype: str = "fp16",
) -> Path:
    """Cache raw inputs to all decoder blocks for the calibration splits."""

    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("complete"):
            if existing.get("model") != model_name_or_path:
                raise ValueError(
                    "Teacher cache belongs to a different model: "
                    f"{existing.get('model')!r} != {model_name_or_path!r}"
                )
            if existing.get("model_revision") != model_revision:
                raise ValueError("Teacher cache belongs to a different model revision")
            if existing.get("cache_dtype") != cache_dtype:
                raise ValueError("Teacher cache uses a different cache_dtype")
            if existing.get("token_manifest") != str(Path(token_manifest).resolve()):
                raise ValueError("Teacher cache belongs to a different token manifest")
            if set(existing.get("splits", {})) != set(splits):
                raise ValueError("Teacher cache contains a different set of data splits")
            if int(existing.get("num_boundaries", 0)) != int(existing["num_layers"]) + 1:
                raise ValueError(
                    "Teacher cache is incomplete or predates the h_0..h_L format; rebuild it"
                )
            return manifest_path

    model = load_causal_lm(
        model_name_or_path,
        revision=model_revision,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    model.eval().requires_grad_(False)
    num_layers = len(model.model.layers)
    target_dtype = torch.float16 if cache_dtype in {"fp16", "float16"} else torch.bfloat16
    split_manifest: dict[str, Any] = {}
    for split in splits:
        dataset = TokenShardDataset(token_manifest, split)
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        for shard_index, offset in enumerate(range(0, len(dataset), batch_size)):
            examples = [dataset[index] for index in range(offset, min(offset + batch_size, len(dataset)))]
            batch = {
                key: torch.stack([example[key] for example in examples]).to(device)
                for key in examples[0]
            }
            boundaries = teacher_boundaries(model, batch["input_ids"], batch["attention_mask"])
            tensors = {
                "input_ids": batch["input_ids"].cpu(),
                "attention_mask": batch["attention_mask"].cpu(),
            }
            for index, hidden in enumerate(boundaries):
                tensors[f"h_{index:03d}"] = hidden.to(dtype=target_dtype, device="cpu")
            shard_path = split_dir / f"shard-{shard_index:05d}.safetensors"
            save_file(tensors, shard_path)
            entries.append(
                {
                    "path": str(shard_path.relative_to(output)),
                    "num_sequences": len(examples),
                }
            )
        split_manifest[split] = {"num_sequences": len(dataset), "shards": entries}

    manifest = {
        "version": 1,
        "complete": True,
        "model": model_name_or_path,
        "model_revision": model_revision,
        "num_layers": num_layers,
        "num_boundaries": num_layers + 1,
        "cache_dtype": cache_dtype,
        "token_manifest": str(Path(token_manifest).resolve()),
        "splits": split_manifest,
    }
    _write_json(manifest_path, manifest)
    return manifest_path
