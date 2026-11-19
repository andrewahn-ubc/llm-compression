"""Small, dependency-free LoRA recovery for compressed Qwen models.

The implementation intentionally avoids PEFT so the SWM baseline can be run
with the project's core environment.  Adapter tensors stay in FP32 even when
the frozen model is BF16/FP16, and a completed run is merged back into ordinary
``nn.Linear`` modules before ``save_pretrained``.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .modeling import load_causal_lm


DEFAULT_QWEN_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class _StopRequested:
    """Cooperatively stop before Slurm's hard time limit."""

    def __init__(self, safety_seconds: int) -> None:
        self.requested = False
        self.safety_seconds = safety_seconds
        self._previous = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, self._handle)

    def _handle(self, _signum, _frame) -> None:
        self.requested = True

    def should_stop(self) -> bool:
        end = os.environ.get("SLURM_JOB_END_TIME")
        return self.requested or (
            end is not None and int(end) - int(time.time()) <= self.safety_seconds
        )

    def close(self) -> None:
        signal.signal(signal.SIGUSR1, self._previous)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _token_data_api():
    """Import shard support only when recovery training is requested."""

    try:
        module = importlib.import_module(f"{__package__}.data")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".", 1)[0] == "safetensors":
            raise RuntimeError(
                "LoRA recovery training requires safetensors; install the project's "
                "core dependencies before reading token shards"
            ) from exc
        raise
    return module.TokenShardDataset, module.deterministic_batch


def _load_causal_model(*args, **kwargs):
    try:
        return load_causal_lm(*args, **kwargs)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".", 1)[0] == "transformers":
            raise RuntimeError(
                "LoRA recovery training requires transformers; install the project's "
                "core dependencies before loading Qwen"
            ) from exc
        raise


class LoRALinear(nn.Module):
    """A frozen linear layer with an FP32 low-rank residual branch."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        factory = {"device": base.weight.device, "dtype": torch.float32}
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        adapter_input = self.dropout(inputs.float())
        update = F.linear(F.linear(adapter_input, self.lora_A), self.lora_B)
        return base_output + update.to(base_output.dtype) * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold the adapter into the base weight and return the plain layer."""

        update = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        self.base.weight.add_(update.to(self.base.weight.dtype))
        return self.base


def _normalized_targets(target_modules: Sequence[str]) -> tuple[str, ...]:
    targets = tuple(dict.fromkeys(str(value).strip() for value in target_modules))
    if not targets or any(not value or "." in value for value in targets):
        raise ValueError("target_modules must contain non-empty linear-layer leaf names")
    return targets


def inject_qwen_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_modules: Sequence[str] = DEFAULT_QWEN_TARGET_MODULES,
) -> tuple[str, ...]:
    """Freeze ``model`` and replace selected Qwen projection layers with LoRA.

    The returned names are stable, fully-qualified module paths and are also
    used as keys in adapter checkpoints.
    """

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type is not None and model_type != "qwen3":
        raise TypeError(f"LoRA recovery expects Qwen3, got model_type={model_type!r}")
    targets = frozenset(_normalized_targets(target_modules))
    model.requires_grad_(False)
    matched: list[str] = []
    # Snapshot before mutation so modules inside newly inserted wrappers are not
    # revisited during the same injection pass.
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            child_name,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
        matched.append(name)
    if not matched:
        requested = ", ".join(sorted(targets))
        raise ValueError(f"No Qwen linear modules matched target_modules: {requested}")
    return tuple(matched)


def iter_lora_modules(model: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for _, module in iter_lora_modules(model):
        parameters.extend((module.lora_A, module.lora_B))
    if not parameters:
        raise ValueError("Model has no injected LoRA modules")
    return parameters


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a CPU/FP32 adapter-only state dictionary."""

    state: dict[str, torch.Tensor] = {}
    for name, module in iter_lora_modules(model):
        state[f"{name}.lora_A"] = module.lora_A.detach().float().cpu()
        state[f"{name}.lora_B"] = module.lora_B.detach().float().cpu()
    if not state:
        raise ValueError("Model has no injected LoRA modules")
    return state


@torch.no_grad()
def load_lora_state_dict(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
) -> None:
    expected: dict[str, nn.Parameter] = {}
    for name, module in iter_lora_modules(model):
        expected[f"{name}.lora_A"] = module.lora_A
        expected[f"{name}.lora_B"] = module.lora_B
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if strict and (missing or unexpected):
        raise ValueError(
            f"Incompatible LoRA checkpoint; missing={missing}, unexpected={unexpected}"
        )
    for key, parameter in expected.items():
        if key not in state:
            continue
        value = state[key]
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(
                f"LoRA tensor {key} has shape {tuple(value.shape)}, "
                f"expected {tuple(parameter.shape)}"
            )
        parameter.copy_(value.to(device=parameter.device, dtype=torch.float32))


def merge_qwen_lora(model: nn.Module) -> tuple[str, ...]:
    """Merge every adapter in place, restoring ordinary ``nn.Linear`` modules."""

    merged: list[str] = []

    def visit(module: nn.Module, prefix: str) -> None:
        for child_name, child in list(module.named_children()):
            name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merge())
                merged.append(name)
            else:
                visit(child, name)

    visit(model, "")
    return tuple(merged)


def _json_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _token_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student = student_logits[:, :-1].float() / temperature
    teacher = teacher_logits[:, :-1].float() / temperature
    mask = attention_mask[:, 1:].float()
    per_token = F.kl_div(
        F.log_softmax(student, dim=-1),
        F.softmax(teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0) * temperature**2


def _validate_run_arguments(
    *,
    total_steps: int,
    max_steps_this_job: int,
    micro_batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    warmup_steps: int,
    temperature: float,
    max_grad_norm: float,
    objective: str,
    teacher_name_or_path: str | None,
) -> None:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if max_steps_this_job < 0:
        raise ValueError("max_steps_this_job cannot be negative")
    if micro_batch_size <= 0 or gradient_accumulation <= 0:
        raise ValueError("micro_batch_size and gradient_accumulation must be positive")
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if max_grad_norm <= 0 or not math.isfinite(max_grad_norm):
        raise ValueError("max_grad_norm must be finite and positive")
    if objective not in {"causal_lm", "logit_kd"}:
        raise ValueError("objective must be 'causal_lm' or 'logit_kd'")
    if objective == "logit_kd" and not teacher_name_or_path:
        raise ValueError("teacher_name_or_path is required for objective='logit_kd'")


def run_lora_recovery(
    *,
    student_name_or_path: str | Path,
    token_manifest: str | Path,
    output_dir: str | Path,
    student_revision: str | None = None,
    train_split: str = "global_train",
    dtype: str = "bfloat16",
    device: str = "cuda",
    local_files_only: bool = True,
    attn_implementation: str | None = None,
    total_steps: int = 300,
    max_steps_this_job: int = 0,
    micro_batch_size: int = 1,
    gradient_accumulation: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    warmup_steps: int = 10,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Sequence[str] = DEFAULT_QWEN_TARGET_MODULES,
    objective: str = "causal_lm",
    teacher_name_or_path: str | None = None,
    teacher_revision: str | None = None,
    temperature: float = 1.0,
    gradient_checkpointing: bool = True,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    safety_seconds: int = 240,
) -> Path:
    """Run one resumable LoRA-recovery chunk and return ``progress.json``.

    ``max_steps_this_job=0`` means no per-invocation cap.  A Slurm ``SIGUSR1``
    or approaching ``SLURM_JOB_END_TIME`` stops after the current optimizer
    step and writes an exact-resume checkpoint.  On completion, adapters are
    merged and a normal Transformers model is written to ``output_dir/model``.
    """

    _validate_run_arguments(
        total_steps=total_steps,
        max_steps_this_job=max_steps_this_job,
        micro_batch_size=micro_batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        temperature=temperature,
        max_grad_norm=max_grad_norm,
        objective=objective,
        teacher_name_or_path=teacher_name_or_path,
    )
    if weight_decay < 0 or not math.isfinite(weight_decay):
        raise ValueError("weight_decay must be finite and non-negative")
    targets = _normalized_targets(target_modules)
    TokenShardDataset, deterministic_batch = _token_data_api()
    _seed_everything(seed)
    output = Path(output_dir)
    state_path = output / "state.pt"
    adapter_path = output / "adapter.pt"
    progress_path = output / "progress.json"
    model_dir = output / "model"

    run_config: dict[str, Any] = {
        "version": 1,
        "student_name_or_path": str(student_name_or_path),
        "student_revision": student_revision,
        "token_manifest": str(token_manifest),
        "train_split": train_split,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "total_steps": total_steps,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation": gradient_accumulation,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": list(targets),
        "objective": objective,
        "teacher_name_or_path": teacher_name_or_path,
        "teacher_revision": teacher_revision,
        "temperature": temperature,
        "gradient_checkpointing": gradient_checkpointing,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
    }
    resume_state: Mapping[str, Any] | None = None
    if state_path.exists():
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        if resume_state.get("run_config") != run_config:
            raise ValueError(
                "LoRA recovery settings differ from the existing checkpoint; "
                "use a new output_dir or restore the original settings"
            )

    student = _load_causal_model(
        str(student_name_or_path),
        revision=student_revision,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    original_use_cache = getattr(student.config, "use_cache", True)
    original_gradient_checkpointing = bool(
        getattr(student, "is_gradient_checkpointing", False)
    )
    student.config.use_cache = False
    matched_modules = inject_qwen_lora(
        student,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=targets,
    )
    enabled_checkpointing_here = False
    enabled_input_grads_here = False
    if (
        gradient_checkpointing
        and not original_gradient_checkpointing
        and hasattr(student, "gradient_checkpointing_enable")
    ):
        try:
            student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:  # pragma: no cover - older supported model wrappers
            student.gradient_checkpointing_enable()
        enabled_checkpointing_here = True
    if gradient_checkpointing:
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
            enabled_input_grads_here = True

    trainable = lora_parameters(student)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        return max((total_steps - step) / max(total_steps - warmup_steps, 1), 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    global_step = 0
    loss_history: list[dict[str, float | int]] = []
    if resume_state is not None:
        load_lora_state_dict(student, resume_state["adapter_state"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        global_step = int(resume_state["global_step"])
        loss_history = list(resume_state.get("loss_history", []))

    teacher: nn.Module | None = None
    if objective == "logit_kd":
        assert teacher_name_or_path is not None
        teacher = _load_causal_model(
            teacher_name_or_path,
            revision=teacher_revision,
            dtype=dtype,
            device=device,
            local_files_only=local_files_only,
            attn_implementation=attn_implementation,
        )
        teacher.eval().requires_grad_(False)
        teacher.config.use_cache = False
    if resume_state is not None:
        _restore_rng_state(resume_state["rng_state"])

    dataset = TokenShardDataset(token_manifest, train_split)
    stopper = _StopRequested(safety_seconds=safety_seconds)
    steps_this_job = 0
    student.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        while global_step < total_steps:
            if stopper.should_stop() or (
                max_steps_this_job and steps_this_job >= max_steps_this_job
            ):
                break
            accumulated_loss = 0.0
            for accumulation_index in range(gradient_accumulation):
                micro_step = global_step * gradient_accumulation + accumulation_index
                batch = deterministic_batch(
                    dataset,
                    batch_size=micro_batch_size,
                    seed=seed,
                    step=micro_step,
                    device=device,
                )
                if objective == "causal_lm":
                    labels = batch["input_ids"].clone()
                    labels.masked_fill_(batch["attention_mask"] == 0, -100)
                    loss = student(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=labels,
                        use_cache=False,
                    ).loss
                    if loss is None:
                        raise RuntimeError("Student model did not return a causal-LM loss")
                else:
                    assert teacher is not None
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
                    loss = _token_kl(
                        student_logits,
                        teacher_logits,
                        batch["attention_mask"],
                        temperature,
                    )
                scaled_loss = loss / gradient_accumulation
                scaled_loss.backward()
                accumulated_loss += float(scaled_loss.detach())
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            steps_this_job += 1
            loss_history.append({"step": global_step, "loss": accumulated_loss})
    finally:
        stopper.close()

    complete = global_step >= total_steps
    adapter_state = lora_state_dict(student)
    checkpoint = {
        "run_config": run_config,
        "adapter_state": adapter_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "loss_history": loss_history,
        "rng_state": _rng_state(),
    }
    _torch_save(state_path, checkpoint)
    _torch_save(
        adapter_path,
        {
            "format": "llm-compression-lora-v1",
            "config": {
                "rank": rank,
                "alpha": alpha,
                "dropout": dropout,
                "target_modules": list(targets),
                "matched_modules": list(matched_modules),
            },
            "state_dict": adapter_state,
        },
    )
    if complete:
        merged = merge_qwen_lora(student)
        if tuple(merged) != tuple(matched_modules):
            raise RuntimeError("Merged LoRA module set does not match the injected module set")
        if enabled_input_grads_here and hasattr(student, "disable_input_require_grads"):
            student.disable_input_require_grads()
        if enabled_checkpointing_here:
            if not hasattr(student, "gradient_checkpointing_disable"):
                raise RuntimeError(
                    "Model enabled gradient checkpointing but cannot restore it before saving"
                )
            student.gradient_checkpointing_disable()
        student.config.use_cache = original_use_cache
        student.eval()
        model_dir.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(model_dir, safe_serialization=True)

    _json_save(
        progress_path,
        {
            "global_step": global_step,
            "total_steps": total_steps,
            "steps_this_job": steps_this_job,
            "complete": complete,
            "objective": objective,
            "adapter_rank": rank,
            "adapter_alpha": alpha,
            "matched_modules": list(matched_modules),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "last_loss": loss_history[-1]["loss"] if loss_history else None,
            "model_path": str(model_dir.resolve()) if complete else None,
            "state_path": str(state_path.resolve()),
            "adapter_path": str(adapter_path.resolve()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )
    return progress_path
