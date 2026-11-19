"""Command-line entry point for local and Slurm experiment stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .baseline_runner import construct_baseline
from .artifacts import atomic_json_dump
from .config import ConfigError, ExperimentConfig, load_config
from .data import (
    cache_teacher_boundaries,
    prepare_evaluation_shards,
    prepare_token_shards,
)
from .evaluate_runner import run_evaluation
from .lora import DEFAULT_QWEN_TARGET_MODULES, run_lora_recovery
from .modeling import load_causal_lm
from .training import run_stage1, run_stage2, run_stage3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe-guided Qwen3 depth-compression experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = (
        "prepare-data",
        "prepare-eval-data",
        "cache-teacher",
        "stage1",
        "stage2",
        "stage3",
        "baseline",
        "lora-recovery",
        "evaluate",
        "plan",
    )
    for command in commands:
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Override a dotted YAML value; repeatable",
        )
        if command == "baseline":
            child.add_argument(
                "--method",
                choices=("original", "naive_drop", "drop_control", "swm", "xmerge"),
            )
    return parser


def _device(cfg: ExperimentConfig) -> str:
    requested = cfg.runtime.device
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def _run_dir(cfg: ExperimentConfig) -> Path:
    return cfg.run_dir.resolve()


def _data_root(cfg: ExperimentConfig) -> Path:
    configured = cfg.paths.data_cache_dir
    return Path(configured).resolve() if configured else _run_dir(cfg) / "data"


def _cache_root(cfg: ExperimentConfig) -> Path:
    configured = cfg.paths.cache_dir
    return Path(configured).resolve() if configured else _run_dir(cfg) / "cache"


def _model_cache_key(cfg: ExperimentConfig) -> str:
    identity = f"{cfg.model.name_or_path}@{cfg.model.revision or 'default'}"
    readable = cfg.model.name_or_path.replace("/", "--").replace(" ", "-")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{digest}"


def _teacher_cache_root(cfg: ExperimentConfig) -> Path:
    return _cache_root(cfg) / "teacher_boundaries" / _model_cache_key(cfg)


def _token_data_root(cfg: ExperimentConfig) -> Path:
    return _data_root(cfg) / _model_cache_key(cfg)


def _runtime(cfg: ExperimentConfig) -> dict[str, Any]:
    return cfg.runtime.to_dict()


def _configure_runtime(cfg: ExperimentConfig) -> None:
    """Apply the reproducibility policy before a command touches CUDA."""

    if not cfg.runtime.deterministic:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _stage_extra(stage) -> dict[str, Any]:
    return stage.to_dict()


def _evaluation(cfg: ExperimentConfig) -> dict[str, Any]:
    return dict(cfg.evaluation)


def _prepare_data(cfg: ExperimentConfig) -> Path:
    data = cfg.data
    return prepare_token_shards(
        output_dir=_token_data_root(cfg) / "fineweb_edu",
        tokenizer_name=cfg.model.name_or_path,
        tokenizer_revision=cfg.model.revision,
        dataset_name=data.dataset_name,
        dataset_config=data.dataset_config,
        dataset_split=data.split,
        text_column=data.text_column,
        split_sequences=data.split_sequences,
        sequence_length=data.sequence_length,
        shard_sequences=int(data.extra.get("shard_sequences", 256)),
        seed=data.seed if data.seed is not None else cfg.seed,
        dataset_revision=data.extra.get("dataset_revision"),
        local_files_only=cfg.model.local_files_only,
    )


def _prepare_eval_data(cfg: ExperimentConfig) -> Path:
    evaluation = _evaluation(cfg)
    perplexity = dict(evaluation.get("perplexity", {}))
    return prepare_evaluation_shards(
        output_dir=_token_data_root(cfg) / "wikitext2",
        tokenizer_name=cfg.model.name_or_path,
        tokenizer_revision=cfg.model.revision,
        dataset_name=str(perplexity.get("dataset_name", "wikitext")),
        dataset_config=perplexity.get("dataset_config", "wikitext-2-raw-v1"),
        dataset_split=str(perplexity.get("split", "test")),
        text_column=str(perplexity.get("text_column", "text")),
        sequence_length=int(perplexity.get("sequence_length", 512)),
        shard_sequences=int(perplexity.get("shard_sequences", 128)),
        dataset_revision=perplexity.get("revision"),
        local_files_only=cfg.model.local_files_only,
    )


def _cache_teacher(cfg: ExperimentConfig) -> Path:
    return cache_teacher_boundaries(
        model_name_or_path=cfg.model.name_or_path,
        model_revision=cfg.model.revision,
        dtype=cfg.model.dtype,
        token_manifest=_token_data_root(cfg) / "fineweb_edu" / "manifest.json",
        splits=("probe_train", "probe_validation"),
        output_dir=_teacher_cache_root(cfg),
        batch_size=int(cfg.data.extra.get("teacher_cache_batch_size", 4)),
        device=_device(cfg),
        local_files_only=cfg.model.local_files_only,
        attn_implementation=cfg.model.attn_implementation,
        cache_dtype=str(cfg.data.extra.get("teacher_cache_dtype", "fp16")),
    )


def _stage1(cfg: ExperimentConfig) -> Path:
    stage = cfg.stage1
    values = _stage_extra(stage)
    stage_root = _run_dir(cfg) / "stage1"
    target = stage.target_removed_layers
    thresholds = [stage.threshold]
    if target is not None:
        thresholds = [float(value) for value in values.get("threshold_grid", thresholds)]
        if stage.threshold not in thresholds:
            thresholds.append(stage.threshold)

    expected_settings = {
        "probe_steps": stage.probe_steps,
        "max_group_size": stage.max_group_size,
        "protected_prefix": stage.protected_prefix_layers,
        "protected_suffix": stage.protected_suffix_layers,
        "learning_rate": stage.learning_rate,
        "batch_size": stage.batch_size,
        "input_mode": str(values.get("input_mode", "mixed")),
        "mixed_lambda": float(values.get("mixed_lambda", 0.5)),
        "initialization": stage.initialization,
        "scan_direction": stage.scan_direction,
        "training_dtype": str(values.get("training_dtype", "float32")),
        "seed": cfg.seed,
    }

    def settings_match(manifest: Mapping[str, Any]) -> bool:
        if manifest.get("model") != cfg.model.name_or_path:
            return False
        if manifest.get("model_revision") != cfg.model.revision:
            return False
        recorded = manifest.get("settings", {})
        return all(recorded.get(key) == value for key, value in expected_settings.items())

    final_path = stage_root / "layout.json"
    if final_path.exists():
        finished = json.loads(final_path.read_text(encoding="utf-8"))
        recorded_target = finished.get("threshold_selection", {}).get(
            "target_removed_layers"
        )
        if (target is None or recorded_target == target) and settings_match(finished):
            return final_path
        raise RuntimeError(
            f"Existing Stage 1 result at {final_path} belongs to another experiment; "
            "use a unique run_name"
        )

    manifests: list[tuple[Path, dict[str, Any]]] = []
    max_new_thresholds = int(_runtime(cfg).get("max_thresholds_this_job", 0))
    thresholds_run = 0
    exact_match = False
    for threshold in thresholds:
        output = (
            stage_root
            if target is None
            else stage_root / "threshold_search" / f"threshold-{threshold:g}"
        )
        path = output / "layout.json"
        if not path.exists():
            if max_new_thresholds and thresholds_run >= max_new_thresholds:
                continue
            path = run_stage1(
                model_name_or_path=cfg.model.name_or_path,
                model_revision=cfg.model.revision,
                dtype=cfg.model.dtype,
                device=_device(cfg),
                local_files_only=cfg.model.local_files_only,
                attn_implementation=cfg.model.attn_implementation,
                cache_manifest=_teacher_cache_root(cfg) / "manifest.json",
                train_split="probe_train",
                validation_split="probe_validation",
                output_dir=output,
                threshold=threshold,
                probe_steps=stage.probe_steps,
                max_group_size=stage.max_group_size,
                protected_prefix=stage.protected_prefix_layers,
                protected_suffix=stage.protected_suffix_layers,
                learning_rate=stage.learning_rate,
                batch_size=stage.batch_size,
                input_mode=str(values.get("input_mode", "mixed")),
                mixed_lambda=float(values.get("mixed_lambda", 0.5)),
                initialization=stage.initialization,
                scan_direction=stage.scan_direction,
                training_dtype=str(values.get("training_dtype", "float32")),
                seed=cfg.seed,
            )
            thresholds_run += 1
        manifest = json.loads(path.read_text(encoding="utf-8"))
        settings = manifest.get("settings", {})
        if float(settings.get("threshold", float("nan"))) != threshold or not settings_match(
            manifest
        ):
            raise RuntimeError(
                f"Existing Stage 1 result at {path} does not match this configuration; "
                "use a new run_name"
            )
        manifests.append((path, manifest))
        if target is not None and int(manifest["removed_layers"]) == target:
            exact_match = True
            break

    if target is None:
        return manifests[0][0]
    completed_thresholds = {float(item[1]["settings"]["threshold"]) for item in manifests}
    search_complete = exact_match or all(value in completed_thresholds for value in thresholds)
    progress_path = stage_root / "progress.json"
    atomic_json_dump(
        {
            "complete": search_complete,
            "target_removed_layers": target,
            "completed_thresholds": sorted(completed_thresholds),
            "pending_thresholds": [
                value for value in thresholds if value not in completed_thresholds
            ],
            "candidates": [
                {
                    "path": str(path.resolve()),
                    "threshold": manifest["settings"]["threshold"],
                    "removed_layers": manifest["removed_layers"],
                }
                for path, manifest in manifests
            ],
        },
        progress_path,
    )
    if not search_complete:
        return progress_path

    _selected_path, selected = min(
        manifests,
        key=lambda item: (
            abs(int(item[1]["removed_layers"]) - target),
            abs(float(item[1]["settings"]["threshold"]) - stage.threshold),
        ),
    )
    selected["threshold_selection"] = {
        "target_removed_layers": target,
        "selected_threshold": selected["settings"]["threshold"],
        "selected_removed_layers": selected["removed_layers"],
        "exact_budget_match": int(selected["removed_layers"]) == target,
        "candidates": [
            {
                "path": str(path.resolve()),
                "threshold": manifest["settings"]["threshold"],
                "removed_layers": manifest["removed_layers"],
            }
            for path, manifest in manifests
        ],
    }
    atomic_json_dump(selected, final_path)
    return final_path


def _layout_path(cfg: ExperimentConfig) -> Path:
    runtime = _runtime(cfg)
    configured = runtime.get("layout_path")
    return Path(configured).resolve() if configured else _run_dir(cfg) / "stage1" / "layout.json"


def _stage2(cfg: ExperimentConfig) -> Path:
    stage = cfg.stage2
    values = _stage_extra(stage)
    runtime = _runtime(cfg)
    if stage.lambda_schedule == "fixed" and math.isclose(stage.lambda_start, 1.0):
        policy = "teacher_only"
    else:
        policy = stage.lambda_schedule
    output_override = runtime.get("stage2_output_dir")
    output = Path(output_override).resolve() if output_override else _run_dir(cfg) / "stage2"
    return run_stage2(
        model_name_or_path=cfg.model.name_or_path,
        model_revision=cfg.model.revision,
        dtype=cfg.model.dtype,
        device=_device(cfg),
        local_files_only=cfg.model.local_files_only,
        attn_implementation=cfg.model.attn_implementation,
        layout_path=_layout_path(cfg),
        student_name_or_path=runtime.get(
            "stage2_student_path", runtime.get("student_path")
        ),
        token_manifest=_token_data_root(cfg) / "fineweb_edu" / "manifest.json",
        train_split="local_train",
        validation_split="local_validation",
        output_dir=output,
        steps_per_group=stage.steps_per_group,
        learning_rate=stage.learning_rate,
        batch_size=stage.batch_size,
        lambda_policy=policy,
        lambda_fixed=stage.lambda_start,
        lambda_end=stage.lambda_end,
        eval_every=stage.eval_every,
        validation_batches=int(stage.extra.get("validation_batches", 4)),
        early_stopping_patience=stage.patience,
        max_groups_this_job=int(runtime.get("max_groups_this_job", 0)),
        max_steps_this_job=int(runtime.get("stage2_max_steps_this_job", 0)),
        training_dtype=str(values.get("training_dtype", "float32")),
        seed=cfg.seed,
    )


def _stage3(cfg: ExperimentConfig) -> Path:
    stage = cfg.stage3
    runtime = _runtime(cfg)
    if not stage.enabled:
        output_override = runtime.get("stage3_output_dir")
        output = Path(output_override).resolve() if output_override else _run_dir(cfg) / "stage3"
        path = output / "progress.json"
        atomic_json_dump(
            {"complete": True, "skipped": True, "reason": "stage3.enabled=false"},
            path,
        )
        return path
    stage2_dir = Path(runtime.get("stage2_output_dir", _run_dir(cfg) / "stage2"))
    stage2_progress = stage2_dir / "progress.json"
    if stage2_progress.exists() and not bool(
        json.loads(stage2_progress.read_text(encoding="utf-8")).get("complete")
    ):
        if not bool(runtime.get("allow_incomplete_stage2", False)):
            raise RuntimeError("Stage 2 is incomplete; submit more Stage 2 chunks before Stage 3")
    student_path = Path(
        runtime.get(
            "stage3_student_path",
            runtime.get("student_path", stage2_dir / "model"),
        )
    )
    if (stage2_dir / "model").exists():
        student_path = stage2_dir / "model"
    output_override = runtime.get("stage3_output_dir")
    output = Path(output_override).resolve() if output_override else _run_dir(cfg) / "stage3"
    return run_stage3(
        teacher_name_or_path=cfg.model.name_or_path,
        teacher_revision=cfg.model.revision,
        student_name_or_path=student_path,
        dtype=cfg.model.dtype,
        device=_device(cfg),
        local_files_only=cfg.model.local_files_only,
        attn_implementation=cfg.model.attn_implementation,
        layout_path=_layout_path(cfg),
        token_manifest=_token_data_root(cfg) / "fineweb_edu" / "manifest.json",
        train_split="global_train",
        validation_split="global_validation",
        output_dir=output,
        total_steps=stage.steps,
        max_steps_this_job=int(runtime.get("max_steps_this_job", 0)),
        micro_batch_size=stage.batch_size,
        gradient_accumulation=stage.gradient_accumulation_steps,
        learning_rate=stage.learning_rate,
        warmup_steps=stage.warmup_steps,
        temperature=stage.temperature,
        optimizer_name="adam8bit" if stage.use_8bit_optimizer else "adam",
        train_scope=stage.train_scope,
        gradient_checkpointing=stage.gradient_checkpointing,
        eval_every=stage.eval_every,
        validation_batches=int(stage.extra.get("validation_batches", 16)),
        early_stopping_patience=int(stage.extra.get("patience", 6)),
        seed=cfg.seed,
    )


def _lora_recovery(cfg: ExperimentConfig) -> Path:
    """Run the adapted SWM-style low-rank recovery baseline."""

    runtime = _runtime(cfg)
    values = dict(cfg.extra.get("lora", {}))
    raw_student = runtime.get("lora_student_path", runtime.get("student_path"))
    if not raw_student:
        raise ConfigError(
            "lora-recovery requires runtime.lora_student_path (or runtime.student_path)"
        )
    output_override = runtime.get("lora_output_dir")
    output = Path(output_override).resolve() if output_override else _run_dir(cfg) / "lora"
    raw_targets = values.get("target_modules", DEFAULT_QWEN_TARGET_MODULES)
    if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes)):
        raise ConfigError("lora.target_modules must be a sequence of projection names")
    objective = str(values.get("objective", "causal_lm"))
    return run_lora_recovery(
        student_name_or_path=str(raw_student),
        token_manifest=_token_data_root(cfg) / "fineweb_edu" / "manifest.json",
        output_dir=output,
        train_split=str(values.get("train_split", "global_train")),
        dtype=cfg.model.dtype,
        device=_device(cfg),
        local_files_only=cfg.model.local_files_only,
        attn_implementation=cfg.model.attn_implementation,
        total_steps=int(values.get("total_steps", 1000)),
        max_steps_this_job=int(runtime.get("lora_max_steps_this_job", 250)),
        micro_batch_size=int(values.get("micro_batch_size", 1)),
        gradient_accumulation=int(values.get("gradient_accumulation", 8)),
        learning_rate=float(values.get("learning_rate", values.get("lr", 1e-4))),
        weight_decay=float(values.get("weight_decay", 0.0)),
        warmup_steps=int(values.get("warmup_steps", 50)),
        rank=int(values.get("rank", 8)),
        alpha=float(values.get("alpha", 16.0)),
        dropout=float(values.get("dropout", 0.0)),
        target_modules=[str(value) for value in raw_targets],
        objective=objective,
        teacher_name_or_path=(cfg.model.name_or_path if objective == "logit_kd" else None),
        teacher_revision=cfg.model.revision,
        temperature=float(values.get("temperature", 1.0)),
        gradient_checkpointing=bool(values.get("gradient_checkpointing", True)),
        max_grad_norm=float(values.get("max_grad_norm", 1.0)),
        seed=cfg.seed,
        safety_seconds=int(values.get("safety_seconds", 240)),
    )


def _baseline(cfg: ExperimentConfig, method_argument: str | None) -> Path:
    runtime = _runtime(cfg)
    baseline = dict(cfg.extra.get("baseline", {}))
    method = method_argument or baseline.get("method") or runtime.get("baseline_method")
    if method is None:
        raise ConfigError("baseline requires --method or --set baseline.method=...")
    teacher = load_causal_lm(
        cfg.model.name_or_path,
        revision=cfg.model.revision,
        dtype=cfg.model.dtype,
        device=_device(cfg),
        local_files_only=cfg.model.local_files_only,
        attn_implementation=cfg.model.attn_implementation,
    )
    teacher.eval().requires_grad_(False)
    return construct_baseline(
        teacher=teacher,
        method_layout_path=_layout_path(cfg),
        method=str(method),
        output_dir=_run_dir(cfg) / "baselines" / str(method),
        cache_manifest=_teacher_cache_root(cfg) / "manifest.json",
        cache_split="probe_validation",
        swm_similarity_threshold=float(baseline.get("swm_similarity_threshold", 0.999)),
        validation_batch_size=int(baseline.get("validation_batch_size", 4)),
        validation_batches=int(baseline.get("validation_batches", 2)),
        xmerge_pair_direction=str(baseline.get("xmerge_pair_direction", "right")),
        device=_device(cfg),
    )


def _evaluation_model(cfg: ExperimentConfig) -> Path | str:
    runtime = _runtime(cfg)
    if runtime.get("model_path"):
        raw = str(runtime["model_path"])
        if raw == "__teacher__":
            return cfg.model.name_or_path
        if raw == "__stage2__":
            stage2_root = Path(runtime.get("stage2_output_dir", _run_dir(cfg) / "stage2"))
            return (stage2_root / "model").resolve()
        candidate = Path(raw).expanduser()
        if candidate.is_absolute() or raw.startswith(".") or candidate.exists():
            return candidate.resolve()
        return raw
    stage3_root = Path(runtime.get("stage3_output_dir", _run_dir(cfg) / "stage3"))
    stage3 = stage3_root / "model"
    if cfg.stage3.enabled and stage3.exists():
        return stage3
    stage2_root = Path(runtime.get("stage2_output_dir", _run_dir(cfg) / "stage2"))
    stage2 = stage2_root / "model"
    if stage2.exists():
        return stage2
    return cfg.model.name_or_path


def _evaluate(cfg: ExperimentConfig) -> Path:
    values = _evaluation(cfg)
    runtime = _runtime(cfg)
    required_progress = runtime.get("require_complete_progress")
    if required_progress:
        required_path = Path(str(required_progress)).expanduser().resolve()
        if not required_path.exists() or not bool(
            json.loads(required_path.read_text(encoding="utf-8")).get("complete")
        ):
            raise RuntimeError(f"Required training stage is incomplete: {required_path}")
    stage3_root = Path(runtime.get("stage3_output_dir", _run_dir(cfg) / "stage3"))
    stage3_progress = stage3_root / "progress.json"
    stage3_model = stage3_root / "model"
    stage2_root = Path(runtime.get("stage2_output_dir", _run_dir(cfg) / "stage2"))
    stage2_model = stage2_root / "model"
    if runtime.get("model_path") == "__stage2__":
        stage2_progress = stage2_root / "progress.json"
        if not stage2_model.exists() or not stage2_progress.exists() or not bool(
            json.loads(stage2_progress.read_text(encoding="utf-8")).get("complete")
        ):
            raise RuntimeError(
                "Stage 2 is incomplete; cannot evaluate runtime.model_path=__stage2__"
            )
    elif not runtime.get("model_path") and cfg.stage3.enabled:
        if not stage3_model.exists() or not stage3_progress.exists():
            raise RuntimeError(
                "Stage 3 output is missing; complete Stage 3 or explicitly disable it "
                "before evaluating the Stage 2 checkpoint"
            )
        progress = json.loads(stage3_progress.read_text(encoding="utf-8"))
        if not bool(progress.get("complete")) and not bool(
            runtime.get("allow_incomplete_stage3", False)
        ):
            raise RuntimeError(
                "Stage 3 is incomplete; submit additional Stage 3 chunks before evaluation"
            )
    elif not runtime.get("model_path"):
        stage2_progress = stage2_root / "progress.json"
        if not stage2_model.exists() or not stage2_progress.exists():
            raise RuntimeError(
                "Stage 2 output is missing; teacher evaluation must use "
                "runtime.model_path=__teacher__"
            )
        stage2_complete = stage2_progress.exists() and bool(
            json.loads(stage2_progress.read_text(encoding="utf-8")).get("complete")
        )
        if not stage2_complete and not bool(runtime.get("allow_incomplete_stage2", False)):
            raise RuntimeError(
                "Stage 2 is incomplete; submit additional Stage 2 chunks before evaluation"
            )
    component = str(values.get("component", "all"))
    prompts = values.get(
        "generation_prompts",
        [
            "Explain why the sky appears blue in two sentences.",
            "Write a short paragraph comparing compression and pruning.",
            "Solve: If 3x + 5 = 20, what is x? Explain briefly.",
        ],
    )
    if not isinstance(prompts, Sequence) or isinstance(prompts, (str, bytes)):
        raise ConfigError("evaluation.generation_prompts must be a sequence")
    perplexity = dict(values.get("perplexity", {}))
    output_subdir = str(values.get("output_subdir", "student"))
    if not output_subdir or Path(output_subdir).is_absolute() or ".." in Path(output_subdir).parts:
        raise ConfigError("evaluation.output_subdir must be a safe relative directory name")
    output_dir = _run_dir(cfg) / "evaluation" / output_subdir
    teacher_results = values.get("teacher_harness_results")
    if teacher_results is None:
        automatic_teacher = _run_dir(cfg) / "evaluation" / "teacher" / "lm_eval"
        if automatic_teacher.exists() and output_subdir != "teacher":
            teacher_results = automatic_teacher
    return run_evaluation(
        model_name_or_path=_evaluation_model(cfg),
        tokenizer_name_or_path=cfg.model.name_or_path,
        model_revision=cfg.model.revision,
        dtype=cfg.model.dtype,
        device=_device(cfg),
        local_files_only=cfg.model.local_files_only,
        attn_implementation=cfg.model.attn_implementation,
        output_dir=output_dir,
        component=component,
        wikitext_manifest=_token_data_root(cfg) / "wikitext2" / "manifest.json",
        wikitext_split=str(perplexity.get("split", "test")),
        perplexity_batch_size=int(perplexity.get("batch_size", 2)),
        harness_batch_size=values.get("batch_size", "auto"),
        harness_limit=values.get("limit"),
        teacher_harness_results=teacher_results,
        prompts=[str(prompt) for prompt in prompts],
        max_new_tokens=int(values.get("max_new_tokens", 64)),
        speed_warmup=int(values.get("speed_warmup", 3)),
        speed_repetitions=int(values.get("speed_repetitions", 10)),
    )


def _plan(cfg: ExperimentConfig) -> Path:
    """Write the full required experiment matrix without launching jobs."""

    stage1 = _stage_extra(cfg.stage1)
    thresholds = list(stage1.get("threshold_grid", [cfg.stage1.threshold]))
    compression_targets = list(stage1.get("target_removed_fractions", []))
    plan = {
        "run_name": cfg.run_name,
        "primary_model": cfg.model.name_or_path,
        "threshold_search": [
            {"threshold": value, "probe_steps": cfg.stage1.probe_steps} for value in thresholds
        ],
        "compression_targets_context_default": compression_targets,
        "main_stages": ["stage1", "stage2", "stage3"],
        "baselines": ["original", "naive_drop", "swm", "xmerge", "drop_control"],
        "ablations": {
            "selector": ["trained_probe", "untrained_merge", "swm_global_similarity"],
            "stage2_lambda": ["fixed_0.5", "linear_1_to_0.5", "teacher_only"],
            "stage3": ["after_stage2", "after_stage3"],
            "probe_steps": [50, 300],
        },
        "evaluation": [
            "wikitext2_perplexity",
            "seven_task_zero_shot",
            "mmlu_5shot",
            "generation",
            "parameter_count",
            "speed",
        ],
        "relative_accuracy_formula": "student_macro_accuracy / teacher_macro_accuracy * 100",
    }
    path = _run_dir(cfg) / "experiment_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    _configure_runtime(cfg)
    dispatch = {
        "prepare-data": _prepare_data,
        "prepare-eval-data": _prepare_eval_data,
        "cache-teacher": _cache_teacher,
        "stage1": _stage1,
        "stage2": _stage2,
        "stage3": _stage3,
        "lora-recovery": _lora_recovery,
        "evaluate": _evaluate,
        "plan": _plan,
    }
    if args.command == "baseline":
        result = _baseline(cfg, args.method)
    else:
        result = dispatch[args.command](cfg)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
