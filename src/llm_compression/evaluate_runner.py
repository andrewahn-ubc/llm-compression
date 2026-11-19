"""End-to-end evaluation orchestration for compressed checkpoints."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from .data import TokenShardDataset
from .evaluation import (
    benchmark_model,
    build_lm_eval_command,
    default_lm_eval_configs,
    evaluate_perplexity,
    generate_records,
    parameter_counts,
    summarize_proposal_benchmarks,
    write_generation_records,
    write_lm_eval_config,
)
from .modeling import load_causal_lm, load_tokenizer


def _dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _batches(dataset: TokenShardDataset, batch_size: int, device: str) -> Iterator[dict[str, torch.Tensor]]:
    for offset in range(0, len(dataset), batch_size):
        examples = [dataset[index] for index in range(offset, min(offset + batch_size, len(dataset)))]
        yield {
            key: torch.stack([example[key] for example in examples]).to(device)
            for key in examples[0]
        }


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME"),
    }
    if torch.cuda.is_available():
        record.update(
            {
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return record


def run_evaluation(
    *,
    model_name_or_path: str | Path,
    tokenizer_name_or_path: str,
    model_revision: str | None,
    dtype: str,
    device: str,
    local_files_only: bool,
    attn_implementation: str | None,
    output_dir: str | Path,
    component: str,
    wikitext_manifest: str | Path | None,
    wikitext_split: str,
    perplexity_batch_size: int,
    harness_batch_size: str | int,
    harness_limit: int | float | None,
    teacher_harness_results: str | Path | None,
    prompts: Sequence[str],
    max_new_tokens: int,
    speed_warmup: int,
    speed_repetitions: int,
) -> Path:
    """Run one evaluation shard or all evaluation components."""

    allowed = {
        "all",
        "perplexity",
        "harness",
        "harness_commonsense",
        "harness_mmlu",
        "generation",
        "speed",
        "summarize",
    }
    if component not in allowed:
        raise ValueError(f"component must be one of {sorted(allowed)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _dump(output / "environment.json", environment_record())

    needs_model = component in {"all", "perplexity", "generation", "speed"}
    model = None
    tokenizer = None
    if needs_model:
        model = load_causal_lm(
            str(model_name_or_path),
            revision=model_revision,
            dtype=dtype,
            device=device,
            local_files_only=local_files_only,
            attn_implementation=attn_implementation,
        )
        model.eval()
        _dump(output / "parameters.json", parameter_counts(model))

    if component in {"all", "perplexity"}:
        if wikitext_manifest is None:
            raise ValueError("WikiText token manifest is required for perplexity")
        dataset = TokenShardDataset(wikitext_manifest, wikitext_split)
        assert model is not None
        perplexity = evaluate_perplexity(
            model,
            _batches(dataset, perplexity_batch_size, device),
            device,
        )
        _dump(
            output / "perplexity.json",
            {
                "dataset": "wikitext-2-raw-v1",
                "split": "test",
                "perplexity": perplexity,
                "num_sequences": len(dataset),
            },
        )

    if component in {"all", "harness", "harness_commonsense", "harness_mmlu"}:
        configs = default_lm_eval_configs(
            model_name_or_path,
            output / "lm_eval",
            tokenizer_name_or_path=tokenizer_name_or_path,
            revision=model_revision,
            device=device,
            batch_size=harness_batch_size,
            dtype=dtype,
            limit=harness_limit,
        )
        selected = list(enumerate(configs))
        if component == "harness_commonsense":
            selected = selected[:1]
        elif component == "harness_mmlu":
            selected = selected[1:]
        for index, config in selected:
            write_lm_eval_config(config, output / "lm_eval" / f"invocation-{index}.json")
            command = build_lm_eval_command(config)
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"lm-eval failed with exit code {completed.returncode}: {command}")

    if component in {"all", "generation", "speed"}:
        assert model is not None
        tokenizer = load_tokenizer(
            tokenizer_name_or_path,
            revision=model_revision,
            local_files_only=local_files_only,
        )

    if component in {"all", "generation"}:
        assert model is not None and tokenizer is not None
        records = generate_records(
            model,
            tokenizer,
            prompts,
            device,
            generation_kwargs={
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
            },
        )
        write_generation_records(records, output / "generations.jsonl")

    if component in {"all", "speed"}:
        assert model is not None and tokenizer is not None
        prompt = prompts[0] if prompts else "Explain layer merging in one sentence."
        batch = tokenizer(prompt, return_tensors="pt")
        forward = benchmark_model(
            model,
            batch,
            device,
            warmup=speed_warmup,
            repetitions=speed_repetitions,
        )
        generation = benchmark_model(
            model,
            batch,
            device,
            warmup=speed_warmup,
            repetitions=speed_repetitions,
            generation_kwargs={
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
            },
        )
        _dump(output / "speed.json", {"forward": forward, "generation": generation})

    if component in {"all", "summarize"}:
        harness_dir = output / "lm_eval"
        if harness_dir.exists():
            teacher = teacher_harness_results if teacher_harness_results else None
            summary = summarize_proposal_benchmarks(
                harness_dir, teacher_results=teacher
            )
            _dump(output / "accuracy_summary.json", summary)

    return output
