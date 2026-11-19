"""Model-agnostic evaluation and reporting helpers.

The module has no dependency on Transformers or lm-evaluation-harness at
import time.  This keeps unit tests and CPU smoke runs light while still
producing command lines and result summaries for the cluster jobs.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


COMMONSENSE_TASKS: tuple[str, ...] = (
    "boolq",
    "piqa",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
)


def _shifted_loss_sum_and_count(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    if logits.ndim != 3:
        raise ValueError(f"Expected logits shaped [batch, sequence, vocab], got {logits.shape}")
    if input_ids.ndim != 2:
        raise ValueError(f"Expected input_ids shaped [batch, sequence], got {input_ids.shape}")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(
            f"Logit leading dimensions {tuple(logits.shape[:2])} do not match "
            f"input_ids {tuple(input_ids.shape)}"
        )
    if input_ids.shape[1] < 2:
        raise ValueError("Causal perplexity requires sequences containing at least two tokens")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")

    shifted_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    valid = targets.ne(-100)
    if attention_mask is not None:
        mask = attention_mask.to(device=targets.device, dtype=torch.bool)
        # Requiring a valid source-target transition handles both ordinary
        # right padding and left padding (whose first content token has no
        # preceding content token and should not contribute to perplexity).
        valid &= mask[:, :-1] & mask[:, 1:]
    token_count = int(valid.sum().item())
    if token_count == 0:
        raise ValueError("No valid next-token targets remain after masking")

    flat_logits = shifted_logits.reshape(-1, shifted_logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_valid = valid.reshape(-1)
    loss_sum = F.cross_entropy(
        flat_logits[flat_valid].float(),
        flat_targets[flat_valid],
        reduction="sum",
    )
    return loss_sum, token_count


def shifted_cross_entropy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean causal next-token loss with the canonical one-token shift.

    ``logits[:, t]`` is scored against ``input_ids[:, t + 1]``.  Padding and
    ``-100`` labels are excluded.  The helper never scores a token against the
    logit at the same position, preventing the common unshifted-perplexity bug.
    """

    loss_sum, token_count = _shifted_loss_sum_and_count(logits, input_ids, attention_mask)
    return loss_sum / token_count


def _normalize_batch(batch: Any, device: torch.device) -> tuple[dict[str, Any], torch.Tensor, Any]:
    if isinstance(batch, torch.Tensor):
        raw: dict[str, Any] = {"input_ids": batch}
    elif isinstance(batch, Mapping):
        raw = dict(batch)
    elif isinstance(batch, (tuple, list)) and 1 <= len(batch) <= 2:
        raw = {"input_ids": batch[0]}
        if len(batch) == 2:
            raw["attention_mask"] = batch[1]
    else:
        raise TypeError("Each batch must be a tensor, mapping, or (input_ids, attention_mask)")

    if "input_ids" not in raw or not isinstance(raw["input_ids"], torch.Tensor):
        raise TypeError("Each evaluation batch must contain tensor input_ids")
    moved = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in raw.items()
    }
    input_ids = moved["input_ids"]
    attention_mask = moved.get("attention_mask")
    # Labels are deliberately removed: this evaluator owns the causal shift
    # and token weighting rather than trusting model-specific loss behavior.
    model_inputs = {key: value for key, value in moved.items() if key != "labels"}
    return model_inputs, input_ids, attention_mask


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, Mapping) and isinstance(output.get("logits"), torch.Tensor):
        return output["logits"]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Model output does not expose a logits tensor")


def _infer_device(model: nn.Module, device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def evaluate_perplexity(
    model: nn.Module,
    batches: Iterable[Any],
    device: str | torch.device | None = None,
) -> float:
    """Evaluate token-weighted causal perplexity over an iterable of batches.

    The model must already reside on ``device`` (important for sharded models);
    batches are moved there.  Loss sums and valid-token counts are accumulated
    across batches before exponentiating, so a short final batch is not
    overweighted.  The model's original train/eval state is restored.
    """

    target_device = _infer_device(model, device)
    was_training = model.training
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                model_inputs, input_ids, attention_mask = _normalize_batch(batch, target_device)
                model_inputs.setdefault("use_cache", False)
                logits = _extract_logits(model(**model_inputs))
                loss_sum, token_count = _shifted_loss_sum_and_count(
                    logits, input_ids, attention_mask
                )
                total_loss += float(loss_sum.detach().cpu())
                total_tokens += token_count
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("Perplexity evaluation received no batches with valid tokens")
    mean_nll = total_loss / total_tokens
    try:
        return math.exp(mean_nll)
    except OverflowError:
        return math.inf


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count unique model parameters (optionally only trainable parameters)."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return total, trainable, and frozen parameter counts for result records."""

    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _synchronize(device: torch.device) -> None:
    """Synchronize an accelerator before reading the wall clock."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _extract_sequences(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    sequences = getattr(output, "sequences", None)
    if isinstance(sequences, torch.Tensor):
        return sequences
    raise TypeError("generate() did not return a tensor or an object with .sequences")


def _timed_token_count(
    output: Any,
    model_inputs: Mapping[str, Any],
    generated: bool,
) -> int:
    input_ids = model_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        return 0
    if generated:
        sequences = _extract_sequences(output)
        new_length = max(0, sequences.shape[-1] - input_ids.shape[-1])
        return int(sequences.shape[0] * new_length)
    attention_mask = model_inputs.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        return int(attention_mask.sum().item())
    return int(input_ids.numel())


def benchmark_model(
    model: nn.Module,
    batch: Any,
    device: str | torch.device | None = None,
    *,
    warmup: int = 3,
    repetitions: int = 10,
    generation_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Benchmark forward or generation latency with correct CUDA barriers.

    Inputs are transferred before warmup and are therefore outside the timed
    region.  Every measured repetition synchronizes CUDA immediately before
    starting and immediately after the model call before stopping the clock.
    Supplying ``generation_kwargs`` benchmarks ``model.generate``; otherwise a
    normal forward is timed.
    """

    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")

    target_device = _infer_device(model, device)
    model_inputs, _, _ = _normalize_batch(batch, target_device)
    generated = generation_kwargs is not None
    kwargs = dict(generation_kwargs or {})
    was_training = model.training
    model.eval()

    def invoke() -> Any:
        if generated:
            return model.generate(**model_inputs, **kwargs)
        return model(**model_inputs)

    latencies: list[float] = []
    token_counts: list[int] = []
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                invoke()
            if target_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(target_device)
            for _ in range(repetitions):
                _synchronize(target_device)
                start = time.perf_counter()
                output = invoke()
                _synchronize(target_device)
                latencies.append(time.perf_counter() - start)
                token_counts.append(_timed_token_count(output, model_inputs, generated))
    finally:
        model.train(was_training)

    ordered = sorted(latencies)
    p90_index = max(0, math.ceil(0.90 * len(ordered)) - 1)
    total_seconds = sum(latencies)
    total_tokens = sum(token_counts)
    result = {
        "mode": "generate" if generated else "forward",
        "device": str(target_device),
        "warmup": warmup,
        "repetitions": repetitions,
        "latencies_seconds": latencies,
        "mean_seconds": statistics.fmean(latencies),
        "median_seconds": statistics.median(latencies),
        "p90_seconds": ordered[p90_index],
        "tokens_per_second": total_tokens / total_seconds if total_seconds > 0 else math.inf,
        "tokens_timed": total_tokens,
    }
    if target_device.type == "cuda":
        result["peak_gpu_memory_bytes"] = int(
            torch.cuda.max_memory_allocated(target_device)
        )
    return result


def generate_records(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    device: str | torch.device | None = None,
    *,
    generation_kwargs: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate one auditable JSON-ready record per free-form prompt."""

    target_device = _infer_device(model, device)
    kwargs = dict(generation_kwargs or {})
    was_training = model.training
    model.eval()
    records: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for prompt_index, prompt in enumerate(prompts):
                if not isinstance(prompt, str):
                    raise TypeError("Every generation prompt must be a string")
                encoded = tokenizer(prompt, return_tensors="pt")
                if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                    raise TypeError("Tokenizer must return a mapping containing input_ids")
                model_inputs = {
                    key: value.to(target_device) if isinstance(value, torch.Tensor) else value
                    for key, value in encoded.items()
                }
                input_ids = model_inputs["input_ids"]
                _synchronize(target_device)
                start = time.perf_counter()
                sequences = _extract_sequences(model.generate(**model_inputs, **kwargs))
                _synchronize(target_device)
                latency = time.perf_counter() - start
                prompt_width = int(input_ids.shape[-1])
                continuation = sequences[0, prompt_width:]
                token_ids = [int(item) for item in continuation.detach().cpu().tolist()]
                generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                records.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "generated_text": generated_text,
                        "generated_token_ids": token_ids,
                        "prompt_tokens": int(input_ids.numel()),
                        "generated_tokens": len(token_ids),
                        "latency_seconds": latency,
                        "generation_kwargs": kwargs,
                    }
                )
    finally:
        model.train(was_training)
    return records


def write_generation_records(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write generation records as JSON Lines and return the destination path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
    return destination


@dataclass(frozen=True)
class LmEvalConfig:
    """A serializable description of one lm-evaluation-harness invocation."""

    tasks: tuple[str, ...]
    model_args: str | Mapping[str, Any]
    num_fewshot: int = 0
    model: str = "hf"
    batch_size: str | int = "auto"
    device: str = "cuda:0"
    output_path: str | Path | None = None
    limit: int | float | None = None
    log_samples: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("LmEvalConfig.tasks cannot be empty")
        if self.num_fewshot < 0:
            raise ValueError("num_fewshot must be non-negative")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.output_path is not None:
            result["output_path"] = str(self.output_path)
        if isinstance(self.model_args, Mapping):
            result["model_args"] = dict(self.model_args)
        return result


# Acronym-capitalized alias for callers that prefer it.
LMEvalConfig = LmEvalConfig


def _serialize_model_args(model_args: str | Mapping[str, Any]) -> str:
    if isinstance(model_args, str):
        if not model_args:
            raise ValueError("model_args string cannot be empty")
        return model_args
    pieces: list[str] = []
    for key, value in model_args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = str(value)
        if "," in key or "," in rendered:
            raise ValueError("lm-eval model-argument keys and values cannot contain commas")
        pieces.append(f"{key}={rendered}")
    if not pieces:
        raise ValueError("model_args mapping cannot be empty")
    return ",".join(pieces)


def build_lm_eval_command(
    config: LmEvalConfig | Mapping[str, Any],
    executable: str = "lm_eval",
) -> list[str]:
    """Build a shell-safe argv list for lm-evaluation-harness."""

    if not isinstance(config, LmEvalConfig):
        values = dict(config)
        if "tasks" in values:
            values["tasks"] = (
                (values["tasks"],)
                if isinstance(values["tasks"], str)
                else tuple(values["tasks"])
            )
        if "extra_args" in values:
            values["extra_args"] = (
                (values["extra_args"],)
                if isinstance(values["extra_args"], str)
                else tuple(values["extra_args"])
            )
        config = LmEvalConfig(**values)
    command = [
        executable,
        "--model",
        config.model,
        "--model_args",
        _serialize_model_args(config.model_args),
        "--tasks",
        ",".join(config.tasks),
        "--num_fewshot",
        str(config.num_fewshot),
        "--batch_size",
        str(config.batch_size),
        "--device",
        config.device,
    ]
    if config.output_path is not None:
        command.extend(("--output_path", str(config.output_path)))
    if config.limit is not None:
        command.extend(("--limit", str(config.limit)))
    if config.log_samples:
        command.append("--log_samples")
    command.extend(config.extra_args)
    return command


def default_lm_eval_configs(
    model_name_or_path: str | Path,
    output_dir: str | Path,
    *,
    tokenizer_name_or_path: str | Path | None = None,
    revision: str | None = None,
    device: str = "cuda:0",
    batch_size: str | int = "auto",
    dtype: str | None = None,
    limit: int | float | None = None,
) -> tuple[LmEvalConfig, LmEvalConfig]:
    """Return the proposal's seven-task zero-shot run and 5-shot MMLU run."""

    model_args: dict[str, Any] = {"pretrained": str(model_name_or_path)}
    if tokenizer_name_or_path is not None:
        model_args["tokenizer"] = str(tokenizer_name_or_path)
    if revision is not None:
        model_args["revision"] = revision
    if dtype is not None:
        model_args["dtype"] = dtype
    output = Path(output_dir)
    return (
        LmEvalConfig(
            tasks=COMMONSENSE_TASKS,
            model_args=model_args,
            num_fewshot=0,
            batch_size=batch_size,
            device=device,
            output_path=output / "commonsense.json",
            limit=limit,
        ),
        LmEvalConfig(
            tasks=("mmlu",),
            model_args=model_args,
            num_fewshot=5,
            batch_size=batch_size,
            device=device,
            output_path=output / "mmlu.json",
            limit=limit,
        ),
    )


def write_lm_eval_config(config: LmEvalConfig, path: str | Path) -> Path:
    """Persist the exact lm-eval invocation configuration as JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
    return destination


_KNOWN_METRICS: dict[str, tuple[str, ...]] = {
    "boolq": ("acc,none", "acc"),
    "piqa": ("acc_norm,none", "acc_norm", "acc,none", "acc"),
    "hellaswag": ("acc_norm,none", "acc_norm", "acc,none", "acc"),
    "winogrande": ("acc,none", "acc"),
    "arc_easy": ("acc_norm,none", "acc_norm", "acc,none", "acc"),
    "arc_challenge": ("acc_norm,none", "acc_norm", "acc,none", "acc"),
    "openbookqa": ("acc_norm,none", "acc_norm", "acc,none", "acc"),
    "mmlu": ("acc,none", "acc"),
}


def relative_accuracy(student_accuracy: float, teacher_accuracy: float) -> float:
    """Return accuracy retention in percent: ``student / teacher * 100``.

    Inputs must use the same scale (both fractions or both percentages).  The
    result is always a percentage; for example, ``0.60 / 0.75 * 100 == 80``.
    """

    student = float(student_accuracy)
    teacher = float(teacher_accuracy)
    if not math.isfinite(student) or not math.isfinite(teacher):
        raise ValueError("Accuracies must be finite")
    if teacher == 0:
        raise ZeroDivisionError("Teacher accuracy is zero; relative accuracy is undefined")
    return student / teacher * 100.0


def _load_result_payloads(source: Any) -> list[Mapping[str, Any]]:
    if isinstance(source, Mapping):
        return [source]
    if isinstance(source, (str, Path)):
        path = Path(source)
        paths = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        if not paths:
            raise FileNotFoundError(f"No JSON result files found under {path}")
        payloads = []
        for item in paths:
            payload = json.loads(item.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping) and (
                "results" in payload or "groups" in payload or "task_accuracies" in payload
            ):
                payloads.append(payload)
        if not payloads:
            raise ValueError(f"No lm-eval result payloads found under {path}")
        return payloads
    if isinstance(source, Sequence):
        payloads: list[Mapping[str, Any]] = []
        for item in source:
            payloads.extend(_load_result_payloads(item))
        return payloads
    raise TypeError("lm-eval results must be mappings, paths, or a sequence of either")


def _choose_accuracy(
    task: str,
    metrics: Mapping[str, Any],
    preferences: Mapping[str, Sequence[str] | str] | None,
) -> tuple[str, float] | None:
    configured = preferences.get(task) if preferences else None
    if isinstance(configured, str):
        candidates = (configured,)
    elif configured is not None:
        candidates = tuple(configured)
    else:
        candidates = _KNOWN_METRICS.get(
            task,
            ("acc_norm,none", "acc,none", "acc_norm", "acc", "exact_match,none"),
        )
    for metric in candidates:
        value = metrics.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return metric, float(value)
    return None


def _extract_task_accuracies(
    source: Any,
    tasks: Sequence[str] | None,
    metric_preferences: Mapping[str, Sequence[str] | str] | None,
) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    metrics_used: dict[str, str] = {}
    requested = set(tasks) if tasks is not None else None
    for payload in _load_result_payloads(source):
        if "task_accuracies" in payload:
            sections = (payload["task_accuracies"],)
        else:
            result_section = payload.get("results", {})
            group_section = payload.get("groups", {})
            # With no explicit task filter, including both a group aggregate
            # (for example ``mmlu``) and all of its result subtasks would count
            # the same benchmark twice.  Groups are consulted automatically
            # only when results are absent; an explicitly requested group is
            # still found in the second section below.
            if requested is None and isinstance(result_section, Mapping) and result_section:
                sections = (result_section,)
            else:
                sections = (result_section, group_section)
        for section in sections:
            if not isinstance(section, Mapping):
                continue
            for task, metric_values in section.items():
                if requested is not None and task not in requested:
                    continue
                if section is sections[0] and "task_accuracies" in payload:
                    chosen = ("preaggregated", float(metric_values))
                elif isinstance(metric_values, Mapping):
                    chosen = _choose_accuracy(str(task), metric_values, metric_preferences)
                else:
                    chosen = None
                if chosen is None:
                    continue
                metric, value = chosen
                if task in values and not math.isclose(values[str(task)], value, abs_tol=1e-12):
                    raise ValueError(f"Conflicting lm-eval results for task {task!r}")
                values[str(task)] = value
                metrics_used[str(task)] = metric
    if requested is not None:
        missing = [task for task in tasks or () if task not in values]
        if missing:
            raise KeyError(f"Requested tasks are missing accuracy results: {missing}")
    if not values:
        raise ValueError("No supported accuracy metrics found in lm-eval results")
    return values, metrics_used


def aggregate_lm_eval_results(
    results: Any,
    teacher_results: Any | None = None,
    *,
    tasks: Sequence[str] | None = None,
    metric_preferences: Mapping[str, Sequence[str] | str] | None = None,
) -> dict[str, Any]:
    """Aggregate lm-eval JSON files and optionally compute teacher retention.

    Accuracies remain on lm-eval's native scale (normally fractions in
    ``[0, 1]``).  The macro average gives every requested task equal weight.
    When teacher results are supplied, ``relative_accuracy_pct`` is defined as
    the macro student accuracy divided by the macro teacher accuracy times 100;
    per-task percentages use the same explicit formula.
    """

    student, metrics_used = _extract_task_accuracies(results, tasks, metric_preferences)
    summary: dict[str, Any] = {
        "task_accuracies": student,
        "task_metrics": metrics_used,
        "average_accuracy": statistics.fmean(student.values()),
        "num_tasks": len(student),
    }
    if teacher_results is None:
        return summary

    comparison_tasks = tuple(student) if tasks is None else tuple(tasks)
    teacher, teacher_metrics = _extract_task_accuracies(
        teacher_results, comparison_tasks, metric_preferences
    )
    common = [task for task in comparison_tasks if task in student and task in teacher]
    if not common:
        raise ValueError("Student and teacher lm-eval results have no tasks in common")
    student_average = statistics.fmean(student[task] for task in common)
    teacher_average = statistics.fmean(teacher[task] for task in common)
    summary.update(
        {
            "teacher_task_accuracies": teacher,
            "teacher_task_metrics": teacher_metrics,
            "teacher_average_accuracy": teacher_average,
            "relative_task_accuracy_pct": {
                task: relative_accuracy(student[task], teacher[task]) for task in common
            },
            "relative_accuracy_pct": relative_accuracy(student_average, teacher_average),
            "relative_accuracy_formula": "student_macro_accuracy / teacher_macro_accuracy * 100",
        }
    )
    return summary


def summarize_proposal_benchmarks(
    results: Any,
    teacher_results: Any | None = None,
) -> dict[str, Any]:
    """Report the proposal's seven-task macro and MMLU as separate metrics."""

    commonsense = aggregate_lm_eval_results(
        results,
        teacher_results=teacher_results,
        tasks=COMMONSENSE_TASKS,
    )
    mmlu = aggregate_lm_eval_results(
        results,
        teacher_results=teacher_results,
        tasks=("mmlu",),
    )
    return {
        "commonsense_zero_shot": commonsense,
        "mmlu_five_shot": mmlu,
        "relative_accuracy_formula": "student_macro_accuracy / teacher_macro_accuracy * 100",
    }


# Short aliases convenient in job-building code.
lm_eval_command = build_lm_eval_command
lm_eval_configs = default_lm_eval_configs


__all__ = [
    "COMMONSENSE_TASKS",
    "LMEvalConfig",
    "LmEvalConfig",
    "aggregate_lm_eval_results",
    "benchmark_model",
    "build_lm_eval_command",
    "count_parameters",
    "default_lm_eval_configs",
    "evaluate_perplexity",
    "generate_records",
    "lm_eval_command",
    "lm_eval_configs",
    "parameter_counts",
    "relative_accuracy",
    "shifted_cross_entropy",
    "summarize_proposal_benchmarks",
    "write_generation_records",
    "write_lm_eval_config",
]
