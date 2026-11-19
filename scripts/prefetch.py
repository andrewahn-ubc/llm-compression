#!/usr/bin/env python3
"""Populate model and bounded dataset caches on an Alliance login node."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from llm_compression.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    default_cache = os.environ.get("LLMC_CACHE_ROOT")
    if default_cache is None:
        scratch = os.environ.get("SCRATCH")
        default_cache = str(Path(scratch) / "llm-compression" / "cache") if scratch else str(
            REPO_ROOT / ".local" / "cache"
        )
    parser = argparse.ArgumentParser(
        description="Download the model snapshot and run bounded data preparation online."
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "main.yaml")
    parser.add_argument("--cache-root", type=Path, default=Path(default_cache))
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--skip-model", action="store_true", help="Do not download model files")
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Do not prefetch WikiText and lm-eval task datasets",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def printable(command: list[str]) -> str:
    return shlex.join(command)


def evaluation_tasks(config: dict[str, Any]) -> list[str]:
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return []
    tasks: list[str] = []
    for key in ("zero_shot_tasks", "five_shot_tasks"):
        values = evaluation.get(key, [])
        if isinstance(values, list):
            tasks.extend(item for item in values if isinstance(item, str) and item)
    return list(dict.fromkeys(tasks))


def prefetch_evaluation_data(config: dict[str, Any], cache_dir: Path) -> None:
    """Materialize evaluation datasets without running model inference.

    lm-eval APIs occasionally move between releases, so this is deliberately
    best-effort. Warnings are visible on the login node, before an offline job is
    submitted.
    """

    evaluation = config.get("evaluation", {})
    perplexity = evaluation.get("perplexity", {}) if isinstance(evaluation, dict) else {}
    if isinstance(perplexity, dict):
        dataset_name = perplexity.get("dataset_name")
        dataset_config = perplexity.get("dataset_config")
        split = perplexity.get("split", "test")
        if isinstance(dataset_name, str) and dataset_name:
            try:
                from datasets import load_dataset

                print(f"evaluation dataset: {dataset_name}/{dataset_config or 'default'} [{split}]")
                load_dataset(
                    dataset_name,
                    dataset_config,
                    split=split,
                    cache_dir=str(cache_dir),
                )
            except Exception as exc:  # noqa: BLE001 - login-node preflight must continue
                print(f"warning: could not prefetch perplexity dataset: {exc}", file=sys.stderr)

    tasks = evaluation_tasks(config)
    if not tasks:
        return
    print(f"lm-eval datasets: {', '.join(tasks)}")
    try:
        from lm_eval.tasks import TaskManager, get_task_dict

        manager = TaskManager()
        try:
            get_task_dict(tasks, task_manager=manager)
        except TypeError:
            # Compatibility with older 0.4.x releases where TaskManager was positional.
            get_task_dict(tasks, manager)
    except Exception as exc:  # noqa: BLE001 - see docstring
        print(f"warning: could not prefetch one or more lm-eval tasks: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    config = load_config(config_path, args.overrides).to_dict()

    model = config.get("model", {})
    model_id = model.get("name_or_path")
    revision = model.get("revision")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model.name_or_path must be present in the config")

    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(cache_root / "huggingface"),
            "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
            "HF_DATASETS_CACHE": str(cache_root / "huggingface" / "datasets"),
            "TRANSFORMERS_CACHE": str(cache_root / "huggingface" / "hub"),
            "HF_HUB_OFFLINE": "0",
            "HF_DATASETS_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    for key in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
    ):
        os.environ[key] = env[key]

    print(f"cache root: {cache_root}")
    if not args.skip_model:
        print(f"model snapshot: {model_id} (revision={revision or 'default'})")
        if not args.dry_run:
            cache_root.mkdir(parents=True, exist_ok=True)
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=env["HF_HUB_CACHE"],
                repo_type="model",
            )

    tasks = evaluation_tasks(config)
    if not args.skip_evaluation:
        evaluation = config.get("evaluation", {})
        perplexity = evaluation.get("perplexity", {}) if isinstance(evaluation, dict) else {}
        if isinstance(perplexity, dict) and perplexity.get("dataset_name"):
            print(
                "evaluation prefetch: "
                f"{perplexity.get('dataset_name')}/{perplexity.get('dataset_config', 'default')}"
            )
        if tasks:
            print(f"evaluation task prefetch: {', '.join(tasks)}")
        if not args.dry_run:
            prefetch_evaluation_data(config, Path(env["HF_DATASETS_CACHE"]))

    command = [
        sys.executable,
        "-m",
        "llm_compression.cli",
        "prepare-data",
        "--config",
        str(config_path),
        "--set",
        "runtime.offline=false",
        "--set",
        "model.local_files_only=false",
        "--set",
        f"paths.cache_dir={cache_root}",
        "--set",
        f"paths.data_cache_dir={cache_root / 'data'}",
    ]
    for override in args.overrides:
        command.extend(("--set", override))

    print(f"data preparation: {printable(command)}")
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
