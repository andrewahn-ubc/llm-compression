"""The three compression stages and resumable training utilities."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import (
    TokenShardDataset,
    cached_batch,
    deterministic_batch,
    iter_cached_batches,
    load_boundary_cache,
)
from .modeling import (
    block_forward,
    clone_block,
    get_backbone,
    get_layers,
    load_block_state,
    load_causal_lm,
    resolve_torch_dtype,
    set_layers,
    state_dict_to_cpu,
    teacher_boundaries,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _span_key(start: int, end: int) -> str:
    return f"{start:03d}-{end:03d}"


def _model_dtype(model: nn.Module) -> torch.dtype:
    return next(model.parameters()).dtype


class StopRequested:
    """Cooperative Slurm signal/time-limit handler."""

    def __init__(self, safety_seconds: int = 240):
        self.requested = False
        self.safety_seconds = safety_seconds
        self._previous = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, self._handle)

    def _handle(self, _signum, _frame) -> None:
        self.requested = True

    def should_stop(self) -> bool:
        end = os.environ.get("SLURM_JOB_END_TIME")
        return self.requested or (end is not None and int(end) - int(time.time()) <= self.safety_seconds)

    def close(self) -> None:
        signal.signal(signal.SIGUSR1, self._previous)


def prefix_hidden(
    model: nn.Module,
    blocks: Sequence[nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run a compressed prefix without gradients and return its boundary state."""

    hidden = get_backbone(model).embed_tokens(input_ids)
    for block in blocks:
        hidden = hidden.to(_model_dtype(block))
        hidden = block_forward(model, block, hidden, attention_mask)
    return hidden


@torch.no_grad()
def cache_prefix_inputs(
    *,
    model: nn.Module,
    prefix_blocks: Sequence[nn.Module],
    cache,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Materialize the frozen compressed-prefix output for a probe split.

    Stage 1 reuses the same student input for every optimization step and every
    attempted group size at a scan position.  Computing it once turns the
    mixed-input probe into two single-block forwards per step instead of
    repeatedly running an ever-longer prefix.
    """

    chunks: list[torch.Tensor] = []
    dtype = _model_dtype(model)
    for batch in iter_cached_batches(
        cache, boundaries=(), batch_size=batch_size, device=device
    ):
        hidden = prefix_hidden(
            model,
            prefix_blocks,
            batch["input_ids"],
            batch["attention_mask"],
        )
        chunks.append(hidden.to(device="cpu", dtype=dtype))
    if not chunks:
        raise ValueError("Cannot build a prefix cache from an empty boundary cache")
    return torch.cat(chunks, dim=0)


def relative_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    teacher_input: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    numerator = (prediction.float() - target.float()).square().sum()
    denominator = (target.float() - teacher_input.float()).square().sum().clamp_min(eps)
    return numerator / denominator


def _swm_merge(blocks: Sequence[nn.Module]) -> nn.Module:
    """SWM Eq. 4: theta_0 + sum(theta_k - theta_0)."""

    if not blocks:
        raise ValueError("Cannot merge an empty block sequence")
    merged = clone_block(blocks[0])
    states = [block.state_dict() for block in blocks]
    with torch.no_grad():
        result = merged.state_dict()
        for key, first in states[0].items():
            if not (first.is_floating_point() or first.is_complex()):
                result[key].copy_(first)
                continue
            accumulation_dtype = (
                torch.float32
                if first.dtype in (torch.float16, torch.bfloat16)
                else first.dtype
            )
            base = first.to(accumulation_dtype)
            value = base.clone()
            for other in states[1:]:
                value.add_(other[key].to(accumulation_dtype) - base)
            result[key].copy_(value.to(first.dtype))
    return merged


def _initialize_candidate(
    layers: Sequence[nn.Module], start: int, end: int, initialization: str
) -> nn.Module:
    if initialization == "swm":
        return _swm_merge(layers[start:end])
    if initialization == "copy_first":
        return clone_block(layers[start])
    if initialization == "copy_last":
        return clone_block(layers[end - 1])
    raise ValueError("initialization must be swm, copy_first, or copy_last")


def _mixed_block_loss(
    model: nn.Module,
    block: nn.Module,
    teacher_input: torch.Tensor,
    student_input: torch.Tensor,
    target: torch.Tensor,
    attention_mask: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_prediction = block_forward(model, block, teacher_input, attention_mask)
    teacher_error = relative_error(teacher_prediction, target, teacher_input)
    if lam >= 1.0:
        student_error = teacher_error.detach()
        return teacher_error, teacher_error, student_error
    student_prediction = block_forward(model, block, student_input, attention_mask)
    student_error = relative_error(student_prediction, target, teacher_input)
    return lam * teacher_error + (1.0 - lam) * student_error, teacher_error, student_error


@torch.no_grad()
def evaluate_probe(
    *,
    model: nn.Module,
    block: nn.Module,
    cache,
    start: int,
    end: int,
    prefix_blocks: Sequence[nn.Module],
    batch_size: int,
    lam: float,
    device: str,
    student_inputs: torch.Tensor | None = None,
) -> dict[str, float]:
    block.eval()
    teacher_num = teacher_den = student_num = 0.0
    dtype = _model_dtype(block)
    cursor = 0
    for batch in iter_cached_batches(
        cache, boundaries=(start, end), batch_size=batch_size, device=device
    ):
        teacher_input = batch[f"h_{start:03d}"].to(dtype)
        target = batch[f"h_{end:03d}"].to(dtype)
        teacher_prediction = block_forward(model, block, teacher_input, batch["attention_mask"])
        teacher_num += float((teacher_prediction.float() - target.float()).square().sum())
        teacher_den += float((target.float() - teacher_input.float()).square().sum())
        if lam < 1.0:
            if student_inputs is None:
                student_input = prefix_hidden(
                    model, prefix_blocks, batch["input_ids"], batch["attention_mask"]
                )
            else:
                count = batch["input_ids"].shape[0]
                student_input = student_inputs[cursor : cursor + count].to(
                    device=device, dtype=dtype
                )
            student_input = student_input.to(dtype)
            student_prediction = block_forward(model, block, student_input, batch["attention_mask"])
            student_num += float((student_prediction.float() - target.float()).square().sum())
        cursor += batch["input_ids"].shape[0]
    teacher_error = teacher_num / max(teacher_den, 1e-12)
    student_error = student_num / max(teacher_den, 1e-12) if lam < 1.0 else teacher_error
    return {
        "combined": lam * teacher_error + (1.0 - lam) * student_error,
        "teacher": teacher_error,
        "student": student_error,
    }


def run_stage1(
    *,
    model_name_or_path: str,
    model_revision: str | None,
    dtype: str,
    device: str,
    local_files_only: bool,
    attn_implementation: str | None,
    cache_manifest: str | Path,
    train_split: str,
    validation_split: str,
    output_dir: str | Path,
    threshold: float,
    probe_steps: int,
    max_group_size: int,
    protected_prefix: int,
    protected_suffix: int,
    learning_rate: float,
    batch_size: int,
    input_mode: str,
    mixed_lambda: float,
    initialization: str,
    scan_direction: str,
    training_dtype: str,
    seed: int,
) -> Path:
    """Greedy train-then-decide grouping from the proposal."""

    if input_mode not in {"teacher", "mixed"}:
        raise ValueError("stage1 input_mode must be 'teacher' or 'mixed'")
    if scan_direction != "front_to_back":
        raise NotImplementedError(
            "The proposed method is front-to-back; back-to-front is only used by the SWM baseline"
        )
    seed_everything(seed)
    output = Path(output_dir)
    checkpoint_dir = output / "probes"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = load_causal_lm(
        model_name_or_path,
        revision=model_revision,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    model.eval().requires_grad_(False)
    layers = get_layers(model)
    num_layers = len(layers)
    if protected_prefix + protected_suffix >= num_layers:
        raise ValueError("Protected layers leave no candidates")
    train_cache = load_boundary_cache(cache_manifest, train_split)
    valid_cache = load_boundary_cache(cache_manifest, validation_split)
    if train_cache.num_layers != num_layers:
        raise ValueError("Teacher cache and model layer counts differ")

    spans: list[dict[str, int]] = []
    block_checkpoints: dict[str, str] = {}
    decisions: list[dict[str, Any]] = []
    prefix_blocks: list[nn.Module] = []
    for index in range(protected_prefix):
        spans.append({"start": index, "end": index + 1})
        prefix_blocks.append(layers[index])

    scan_stop = num_layers - protected_suffix
    start = protected_prefix
    dtype_torch = resolve_torch_dtype(training_dtype)
    if isinstance(dtype_torch, str):
        raise ValueError("Stage 1 training_dtype must be an explicit floating dtype")
    while start < scan_stop:
        if start + 1 >= scan_stop:
            spans.append({"start": start, "end": start + 1})
            prefix_blocks.append(layers[start])
            start += 1
            continue

        end = start + 2
        lam = 1.0 if input_mode == "teacher" else mixed_lambda
        train_student_inputs = None
        valid_student_inputs = None
        if lam < 1.0:
            train_student_inputs = cache_prefix_inputs(
                model=model,
                prefix_blocks=prefix_blocks,
                cache=train_cache,
                batch_size=batch_size,
                device=device,
            )
            valid_student_inputs = cache_prefix_inputs(
                model=model,
                prefix_blocks=prefix_blocks,
                cache=valid_cache,
                batch_size=batch_size,
                device=device,
            )
        candidate = _initialize_candidate(layers, start, end, initialization).to(
            device=device, dtype=dtype_torch
        )
        candidate.requires_grad_(True)
        best_state: dict[str, torch.Tensor] | None = None
        best_end: int | None = None
        while end <= scan_stop and end - start <= max_group_size:
            # The trained selector deliberately warm-starts a growing group,
            # as specified in the proposal.  The zero-step ablation instead
            # measures the untrained merge for *each complete candidate span*;
            # carrying a size-2 merge into a size-3/4 score would omit the
            # newly added teacher block and would not be an untrained-merge
            # comparison.
            if probe_steps == 0:
                candidate = _initialize_candidate(
                    layers, start, end, initialization
                ).to(device=device, dtype=dtype_torch)
                candidate.requires_grad_(True)
            candidate.train()
            optimizer = torch.optim.Adam(candidate.parameters(), lr=learning_rate)
            for step in range(probe_steps):
                batch = cached_batch(
                    train_cache,
                    boundaries=(start, end),
                    batch_size=batch_size,
                    seed=seed + start * 10_000 + end * 100,
                    step=step,
                    device=device,
                    return_global_indices=lam < 1.0,
                )
                teacher_input = batch[f"h_{start:03d}"].to(dtype_torch)
                target = batch[f"h_{end:03d}"].to(dtype_torch)
                with torch.no_grad():
                    if lam >= 1.0:
                        student_input = teacher_input
                    else:
                        assert train_student_inputs is not None
                        global_indices = batch.pop("__global_indices__")
                        student_input = train_student_inputs[global_indices].to(
                            device=device, dtype=dtype_torch
                        )
                loss, _, _ = _mixed_block_loss(
                    model,
                    candidate,
                    teacher_input,
                    student_input,
                    target,
                    batch["attention_mask"],
                    lam,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
                optimizer.step()

            errors = evaluate_probe(
                model=model,
                block=candidate,
                cache=valid_cache,
                start=start,
                end=end,
                prefix_blocks=prefix_blocks,
                batch_size=batch_size,
                lam=lam,
                device=device,
                student_inputs=valid_student_inputs,
            )
            accepted = errors["combined"] <= threshold
            candidate_path = checkpoint_dir / f"candidate-{_span_key(start, end)}.pt"
            _torch_save(candidate_path, state_dict_to_cpu(candidate))
            decisions.append(
                {
                    "start": start,
                    "end": end,
                    "size": end - start,
                    "probe_steps": probe_steps,
                    "threshold": threshold,
                    "input_mode": input_mode,
                    "lambda": lam,
                    "accepted": accepted,
                    "errors": errors,
                    "checkpoint": str(candidate_path.resolve()),
                }
            )
            if not accepted:
                break
            best_state = state_dict_to_cpu(candidate)
            best_end = end
            end += 1

        if best_state is None or best_end is None:
            spans.append({"start": start, "end": start + 1})
            prefix_blocks.append(layers[start])
            start += 1
        else:
            selected = clone_block(layers[start]).to(device=device, dtype=dtype_torch)
            selected.load_state_dict(best_state)
            selected.eval().requires_grad_(False)
            selected_path = checkpoint_dir / f"selected-{_span_key(start, best_end)}.pt"
            _torch_save(selected_path, best_state)
            spans.append({"start": start, "end": best_end})
            block_checkpoints[_span_key(start, best_end)] = str(selected_path.resolve())
            prefix_blocks.append(selected)
            start = best_end

    for index in range(scan_stop, num_layers):
        spans.append({"start": index, "end": index + 1})

    removed = sum(item["end"] - item["start"] - 1 for item in spans)
    manifest = {
        "version": 1,
        "method": "probe_guided",
        "model": model_name_or_path,
        "model_revision": model_revision,
        "num_layers": num_layers,
        "spans": spans,
        "protected_layers": [*range(protected_prefix), *range(scan_stop, num_layers)],
        "removed_layers": removed,
        "compressed_layers": len(spans),
        "compression_fraction": removed / num_layers,
        "block_checkpoints": block_checkpoints,
        "decisions": decisions,
        "settings": {
            "threshold": threshold,
            "probe_steps": probe_steps,
            "max_group_size": max_group_size,
            "protected_prefix": protected_prefix,
            "protected_suffix": protected_suffix,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "input_mode": input_mode,
            "mixed_lambda": mixed_lambda,
            "initialization": initialization,
            "scan_direction": scan_direction,
            "training_dtype": training_dtype,
            "seed": seed,
        },
    }
    path = output / "layout.json"
    _json_dump(path, manifest)
    return path


def load_layout_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    spans = manifest["spans"]
    cursor = 0
    for span in spans:
        if span["start"] != cursor or span["end"] <= span["start"]:
            raise ValueError("Layout spans must form a contiguous partition")
        cursor = span["end"]
    if cursor != manifest["num_layers"]:
        raise ValueError("Layout does not cover all original layers")
    return manifest


def materialize_student(
    teacher: nn.Module,
    manifest: Mapping[str, Any],
    *,
    mode: str = "merged",
) -> nn.Module:
    """Clone a teacher and replace every original span with one block."""

    if mode not in {"merged", "drop", "swm"}:
        raise ValueError("mode must be merged, drop, or swm")
    student = copy.deepcopy(teacher)
    original = get_layers(teacher)
    new_layers: list[nn.Module] = []
    checkpoints = manifest.get("block_checkpoints", {})
    for span in manifest["spans"]:
        start, end = int(span["start"]), int(span["end"])
        if end == start + 1 or mode == "drop":
            block = clone_block(original[start])
        elif mode == "swm":
            block = _swm_merge(original[start:end])
        else:
            block = clone_block(original[start])
            key = _span_key(start, end)
            if key not in checkpoints:
                raise KeyError(f"No trained block checkpoint for merged span {key}")
            load_block_state(block, checkpoints[key])
        new_layers.append(block)
    set_layers(student, new_layers)
    return student


@torch.no_grad()
def _validate_local_group(
    *,
    teacher: nn.Module,
    student: nn.Module,
    dataset: TokenShardDataset,
    group_index: int,
    start: int,
    end: int,
    block: nn.Module,
    batch_size: int,
    device: str,
    max_batches: int,
) -> dict[str, float]:
    teacher_num = teacher_den = student_num = 0.0
    dtype = _model_dtype(block)
    for batch_index, offset in enumerate(range(0, len(dataset), batch_size)):
        if batch_index >= max_batches:
            break
        examples = [dataset[index] for index in range(offset, min(offset + batch_size, len(dataset)))]
        batch = {
            key: torch.stack([example[key] for example in examples]).to(device)
            for key in examples[0]
        }
        boundaries = teacher_boundaries(teacher, batch["input_ids"], batch["attention_mask"])
        teacher_input = boundaries[start].to(dtype)
        target = boundaries[end].to(dtype)
        student_input = prefix_hidden(
            student, list(get_layers(student)[:group_index]), batch["input_ids"], batch["attention_mask"]
        ).to(dtype)
        pred_teacher = block_forward(student, block, teacher_input, batch["attention_mask"])
        pred_student = block_forward(student, block, student_input, batch["attention_mask"])
        teacher_num += float((pred_teacher.float() - target.float()).square().sum())
        student_num += float((pred_student.float() - target.float()).square().sum())
        teacher_den += float((target.float() - teacher_input.float()).square().sum())
    return {
        "teacher": teacher_num / max(teacher_den, 1e-12),
        "student": student_num / max(teacher_den, 1e-12),
    }


def _lambda_value(policy: str, step: int, steps: int, fixed: float, end: float) -> float:
    if policy == "fixed":
        return fixed
    if policy == "teacher_only":
        return 1.0
    if policy == "linear":
        fraction = min(max(step / max(steps - 1, 1), 0.0), 1.0)
        return 1.0 + fraction * (end - 1.0)
    if policy == "cosine":
        fraction = min(max(step / max(steps - 1, 1), 0.0), 1.0)
        weight = 0.5 * (1.0 - math.cos(math.pi * fraction))
        return 1.0 + weight * (end - 1.0)
    raise ValueError(f"Unknown lambda policy {policy!r}")


def run_stage2(
    *,
    model_name_or_path: str,
    model_revision: str | None,
    dtype: str,
    device: str,
    local_files_only: bool,
    attn_implementation: str | None,
    layout_path: str | Path,
    student_name_or_path: str | Path | None,
    token_manifest: str | Path,
    train_split: str,
    validation_split: str,
    output_dir: str | Path,
    steps_per_group: int,
    learning_rate: float,
    batch_size: int,
    lambda_policy: str,
    lambda_fixed: float,
    lambda_end: float,
    eval_every: int,
    validation_batches: int,
    early_stopping_patience: int,
    max_groups_this_job: int,
    max_steps_this_job: int,
    training_dtype: str,
    seed: int,
) -> Path:
    """Train merged blocks in network order, exactly one block at a time."""

    seed_everything(seed)
    output = Path(output_dir)
    model_dir = output / "model"
    progress_path = output / "progress.json"
    working_path = output / "working.pt"
    manifest = load_layout_manifest(layout_path)
    teacher = load_causal_lm(
        model_name_or_path,
        revision=model_revision,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    teacher.eval().requires_grad_(False)
    if progress_path.exists() and model_dir.exists():
        student = load_causal_lm(
            str(model_dir),
            dtype=dtype,
            device=device,
            local_files_only=True,
            attn_implementation=attn_implementation,
        )
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        if student_name_or_path is None:
            student = materialize_student(teacher, manifest).to(device)
        else:
            student = load_causal_lm(
                str(student_name_or_path),
                dtype=dtype,
                device=device,
                local_files_only=local_files_only,
                attn_implementation=attn_implementation,
            )
            if len(get_layers(student)) != len(manifest["spans"]):
                raise ValueError(
                    "Starting student depth does not match the supplied layout"
                )
        progress = {"completed_groups": [], "current_group": None, "step": 0, "history": []}

    train_data = TokenShardDataset(token_manifest, train_split)
    valid_data = TokenShardDataset(token_manifest, validation_split)
    merged_indices = [
        index
        for index, span in enumerate(manifest["spans"])
        if int(span["end"]) > int(span["start"]) + 1
    ]
    if not merged_indices:
        model_dir.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(model_dir, safe_serialization=True)
        progress["complete"] = True
        _json_dump(progress_path, progress)
        return progress_path
    completed = set(int(value) for value in progress["completed_groups"])
    local_dtype = resolve_torch_dtype(training_dtype)
    if isinstance(local_dtype, str):
        raise ValueError("Stage 2 training_dtype must be an explicit floating dtype")
    groups_done_this_job = 0
    steps_done_this_job = 0
    stopper = StopRequested()
    try:
        for group_index in merged_indices:
            if group_index in completed:
                continue
            if max_groups_this_job and groups_done_this_job >= max_groups_this_job:
                break
            span = manifest["spans"][group_index]
            start, end = int(span["start"]), int(span["end"])
            block = get_layers(student)[group_index].to(dtype=local_dtype)
            student.requires_grad_(False)
            block.requires_grad_(True)
            block.train()
            optimizer = torch.optim.Adam(block.parameters(), lr=learning_rate)
            start_step = 0
            best_metric = math.inf
            best_state = state_dict_to_cpu(block)
            patience = 0
            if (
                working_path.exists()
                and progress.get("current_group") == group_index
                and int(progress.get("step", 0)) < steps_per_group
            ):
                state = torch.load(working_path, map_location="cpu", weights_only=False)
                block.load_state_dict(state["block"])
                optimizer.load_state_dict(state["optimizer"])
                start_step = int(state["step"])
                best_metric = float(state.get("best_metric", math.inf))
                best_state = state.get("best_block", best_state)
                patience = int(state.get("patience", 0))

            for step in range(start_step, steps_per_group):
                if stopper.should_stop() or (
                    max_steps_this_job and steps_done_this_job >= max_steps_this_job
                ):
                    break
                batch = deterministic_batch(
                    train_data,
                    batch_size=batch_size,
                    seed=seed + group_index * 100_000,
                    step=step,
                    device=device,
                )
                with torch.no_grad():
                    boundaries = teacher_boundaries(
                        teacher, batch["input_ids"], batch["attention_mask"]
                    )
                    teacher_input = boundaries[start].to(local_dtype)
                    target = boundaries[end].to(local_dtype)
                    student_input = prefix_hidden(
                        student,
                        list(get_layers(student)[:group_index]),
                        batch["input_ids"],
                        batch["attention_mask"],
                    ).to(local_dtype)
                lam = _lambda_value(
                    lambda_policy, step, steps_per_group, lambda_fixed, lambda_end
                )
                loss, teacher_error, student_error = _mixed_block_loss(
                    student,
                    block,
                    teacher_input,
                    student_input,
                    target,
                    batch["attention_mask"],
                    lam,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
                optimizer.step()
                steps_done_this_job += 1
                progress.update({"current_group": group_index, "step": step + 1})

                if (step + 1) % eval_every == 0 or step + 1 == steps_per_group:
                    metrics = _validate_local_group(
                        teacher=teacher,
                        student=student,
                        dataset=valid_data,
                        group_index=group_index,
                        start=start,
                        end=end,
                        block=block,
                        batch_size=batch_size,
                        device=device,
                        max_batches=validation_batches,
                    )
                    metric = lam * metrics["teacher"] + (1.0 - lam) * metrics["student"]
                    progress["history"].append(
                        {
                            "group_index": group_index,
                            "span": [start, end],
                            "step": step + 1,
                            "lambda": lam,
                            **metrics,
                        }
                    )
                    if metric < best_metric:
                        best_metric = metric
                        best_state = state_dict_to_cpu(block)
                        patience = 0
                    else:
                        patience += 1
                    if patience >= early_stopping_patience:
                        progress["step"] = steps_per_group
                        break

                _torch_save(
                    working_path,
                    {
                        "block": state_dict_to_cpu(block),
                        "optimizer": optimizer.state_dict(),
                        "step": step + 1,
                        "best_metric": best_metric,
                        "best_block": best_state,
                        "patience": patience,
                    },
                )
                _json_dump(progress_path, progress)

            finished_group = int(progress.get("step", 0)) >= steps_per_group or patience >= early_stopping_patience
            if finished_group:
                block.load_state_dict(best_state)
                block.eval().requires_grad_(False)
                completed.add(group_index)
                progress["completed_groups"] = sorted(completed)
                progress.update({"current_group": None, "step": 0})
                groups_done_this_job += 1
            model_dir.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(model_dir, safe_serialization=True)
            _json_dump(progress_path, progress)
            if stopper.should_stop() or (
                max_steps_this_job and steps_done_this_job >= max_steps_this_job
            ):
                break
    finally:
        stopper.close()

    progress["complete"] = len(completed) == len(merged_indices)
    _json_dump(progress_path, progress)
    return progress_path


def token_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student = student_logits[:, :-1].float() / temperature
    teacher = teacher_logits[:, :-1].float() / temperature
    mask = attention_mask[:, 1:].float()
    per_token = F.kl_div(
        F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1), reduction="none"
    ).sum(dim=-1)
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0) * temperature**2


def _make_optimizer(name: str, parameters: Iterable[nn.Parameter], learning_rate: float):
    params = [parameter for parameter in parameters if parameter.requires_grad]
    if name == "adam8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:  # pragma: no cover - cluster-only optional dependency
            raise RuntimeError("optimizer=adam8bit requires the bitsandbytes extra") from exc
        return bnb.optim.Adam8bit(params, lr=learning_rate)
    if name == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate)
    raise ValueError(f"Unknown optimizer {name!r}")


@torch.no_grad()
def _validate_global_kd(
    *,
    teacher: nn.Module,
    student: nn.Module,
    dataset: TokenShardDataset,
    batch_size: int,
    temperature: float,
    device: str,
    max_batches: int,
) -> float:
    """Return token-weighted held-out logit KL for Stage 3 stopping."""

    was_training = student.training
    student.eval()
    weighted_loss = 0.0
    token_count = 0
    try:
        for batch_index, offset in enumerate(range(0, len(dataset), batch_size)):
            if batch_index >= max_batches:
                break
            examples = [
                dataset[index]
                for index in range(offset, min(offset + batch_size, len(dataset)))
            ]
            batch = {
                key: torch.stack([example[key] for example in examples]).to(device)
                for key in examples[0]
            }
            teacher_logits = teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
            student_logits = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
            count = int(batch["attention_mask"][:, 1:].sum().item())
            value = token_kl(
                student_logits,
                teacher_logits,
                batch["attention_mask"],
                temperature,
            )
            weighted_loss += float(value) * count
            token_count += count
            del teacher_logits, student_logits
    finally:
        student.train(was_training)
    if token_count == 0:
        raise ValueError("Stage 3 validation split contains no scored tokens")
    return weighted_loss / token_count


def run_stage3(
    *,
    teacher_name_or_path: str,
    teacher_revision: str | None,
    student_name_or_path: str | Path,
    dtype: str,
    device: str,
    local_files_only: bool,
    attn_implementation: str | None,
    layout_path: str | Path,
    token_manifest: str | Path,
    train_split: str,
    validation_split: str,
    output_dir: str | Path,
    total_steps: int,
    max_steps_this_job: int,
    micro_batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    warmup_steps: int,
    temperature: float,
    optimizer_name: str,
    train_scope: str,
    gradient_checkpointing: bool,
    eval_every: int,
    validation_batches: int,
    early_stopping_patience: int,
    seed: int,
) -> Path:
    """Chunkable global logit distillation with exact optimizer-step resume."""

    seed_everything(seed)
    output = Path(output_dir)
    model_dir = output / "model"
    state_path = output / "state.pt"
    best_path = output / "best_student.pt"
    source_student = model_dir if state_path.exists() and model_dir.exists() else Path(student_name_or_path)
    teacher = load_causal_lm(
        teacher_name_or_path,
        revision=teacher_revision,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    teacher.eval().requires_grad_(False)
    student = load_causal_lm(
        str(source_student),
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    original_use_cache = bool(getattr(student.config, "use_cache", True))
    original_gradient_checkpointing = bool(
        getattr(student, "is_gradient_checkpointing", False)
    )
    student.config.use_cache = False
    enabled_checkpointing_here = False
    enabled_input_grads_here = False
    if (
        gradient_checkpointing
        and not original_gradient_checkpointing
        and hasattr(student, "gradient_checkpointing_enable")
    ):
        try:
            # Non-reentrant checkpointing does not require the frozen embedding
            # output to have requires_grad=True, which is essential for the
            # merged-block-only Stage 3 setting.
            student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:  # pragma: no cover - older supported model wrappers
            student.gradient_checkpointing_enable()
            if train_scope == "merged" and hasattr(student, "enable_input_require_grads"):
                student.enable_input_require_grads()
                enabled_input_grads_here = True
        enabled_checkpointing_here = True
    student.requires_grad_(train_scope == "all")
    manifest = load_layout_manifest(layout_path)
    if train_scope == "merged":
        student.requires_grad_(False)
        for index, span in enumerate(manifest["spans"]):
            if int(span["end"]) > int(span["start"]) + 1:
                get_layers(student)[index].requires_grad_(True)
    elif train_scope != "all":
        raise ValueError("stage3 train_scope must be 'all' or 'merged'")

    optimizer = _make_optimizer(optimizer_name, student.parameters(), learning_rate)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        return max((total_steps - step) / max(total_steps - warmup_steps, 1), 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    global_step = 0
    best_metric = math.inf
    patience = 0
    history: list[dict[str, Any]] = []
    stopped_early = False
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if "student" in state:
            student.load_state_dict(state["student"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = int(state["global_step"])
        best_metric = float(state.get("best_metric", math.inf))
        patience = int(state.get("patience", 0))
        history = list(state.get("history", []))
        stopped_early = bool(state.get("stopped_early", False))

    dataset = TokenShardDataset(token_manifest, train_split)
    validation_data = TokenShardDataset(token_manifest, validation_split)
    stopper = StopRequested()
    steps_this_job = 0
    student.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        while global_step < total_steps and not stopped_early:
            if stopper.should_stop() or (
                max_steps_this_job and steps_this_job >= max_steps_this_job
            ):
                break
            accumulated = 0.0
            for accumulation_index in range(gradient_accumulation):
                micro_step = global_step * gradient_accumulation + accumulation_index
                batch = deterministic_batch(
                    dataset,
                    batch_size=micro_batch_size,
                    seed=seed,
                    step=micro_step,
                    device=device,
                )
                with torch.no_grad():
                    teacher_logits = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    ).logits
                student_logits = student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
                loss = token_kl(
                    student_logits, teacher_logits, batch["attention_mask"], temperature
                ) / gradient_accumulation
                loss.backward()
                accumulated += float(loss.detach())
                del teacher_logits, student_logits
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in student.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            steps_this_job += 1

            if global_step % eval_every == 0 or global_step == total_steps:
                metric = _validate_global_kd(
                    teacher=teacher,
                    student=student,
                    dataset=validation_data,
                    batch_size=micro_batch_size,
                    temperature=temperature,
                    device=device,
                    max_batches=validation_batches,
                )
                history.append({"step": global_step, "heldout_kl": metric})
                if metric < best_metric:
                    best_metric = metric
                    patience = 0
                    _torch_save(best_path, state_dict_to_cpu(student))
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        stopped_early = True

        complete = global_step >= total_steps or stopped_early
        if complete and best_path.exists():
            student.load_state_dict(
                torch.load(best_path, map_location="cpu", weights_only=True), strict=True
            )

        if enabled_input_grads_here and hasattr(student, "disable_input_require_grads"):
            student.disable_input_require_grads()
        if enabled_checkpointing_here and hasattr(student, "gradient_checkpointing_disable"):
            student.gradient_checkpointing_disable()
        student.config.use_cache = original_use_cache
        model_dir.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(model_dir, safe_serialization=True)
        _torch_save(
            state_path,
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "student": state_dict_to_cpu(student),
                "global_step": global_step,
                "total_steps": total_steps,
                "best_metric": best_metric,
                "patience": patience,
                "history": history,
                "stopped_early": stopped_early,
            },
        )
        _json_dump(
            output / "progress.json",
            {
                "global_step": global_step,
                "total_steps": total_steps,
                "complete": complete,
                "stopped_early": stopped_early,
                "best_heldout_kl": best_metric,
                "history": history,
                "train_scope": train_scope,
                "temperature": temperature,
            },
        )
    finally:
        stopper.close()
    return output / "progress.json"
