"""Typed, forward-compatible YAML configuration loading.

Known experiment settings are validated and exposed as dataclasses. Unknown keys are
retained in each section's ``extra`` mapping and by :attr:`ExperimentConfig.raw`, which
lets new experiment knobs be introduced without invalidating old launch code.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class ConfigError(ValueError):
    """Raised when a configuration value is missing, malformed, or inconsistent."""


_MISSING = object()


def _as_mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping, got {type(value).__name__}")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError(f"{path} keys must be strings, got {key!r}")
        result[key] = copy.deepcopy(item)
    return result


def _take(
    values: dict[str, Any],
    key: str,
    default: Any,
    *,
    aliases: Sequence[str] = (),
) -> Any:
    present = [candidate for candidate in (key, *aliases) if candidate in values]
    if len(present) > 1:
        raise ConfigError(f"use only one of {', '.join(present)}")
    if not present:
        return copy.deepcopy(default)
    return values.pop(present[0])


def _string(value: Any, path: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    return value


def _integer(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    allow_none: bool = False,
) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path} must be at least {minimum}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{path} must be finite")
    if minimum is not None:
        invalid = result <= minimum if minimum_exclusive else result < minimum
        if invalid:
            relation = "greater than" if minimum_exclusive else "at least"
            raise ConfigError(f"{path} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ConfigError(f"{path} must be at most {maximum}")
    return result


def _choice(value: Any, path: str, choices: set[str]) -> str:
    result = _string(value, path)
    assert result is not None
    if result not in choices:
        allowed = ", ".join(sorted(choices))
        raise ConfigError(f"{path} must be one of: {allowed}")
    return result


def _path_string(value: Any, path: str, *, allow_none: bool = False) -> str | None:
    if isinstance(value, Path):
        value = str(value)
    return _string(value, path, allow_none=allow_none)


def _merge_extra(extra: Mapping[str, Any], known: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(extra))
    for key, value in known.items():
        result[key] = copy.deepcopy(value)
    return result


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str = "Qwen/Qwen3-0.6B"
    revision: str | None = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    attn_implementation: str | None = None
    local_files_only: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "ModelConfig":
        values = _as_mapping(raw, "model")
        name = _string(
            _take(values, "name_or_path", cls.name_or_path, aliases=("name", "model_id")),
            "model.name_or_path",
        )
        revision = _string(
            _take(values, "revision", cls.revision), "model.revision", allow_none=True
        )
        dtype = _choice(
            _take(values, "dtype", cls.dtype),
            "model.dtype",
            {"auto", "float32", "float16", "bfloat16", "fp32", "fp16", "bf16"},
        )
        trust_remote_code = _boolean(
            _take(values, "trust_remote_code", cls.trust_remote_code),
            "model.trust_remote_code",
        )
        attn = _string(
            _take(values, "attn_implementation", cls.attn_implementation),
            "model.attn_implementation",
            allow_none=True,
        )
        local_files_only = _boolean(
            _take(values, "local_files_only", cls.local_files_only),
            "model.local_files_only",
        )
        assert name is not None
        return cls(name, revision, dtype, trust_remote_code, attn, local_files_only, values)

    def to_dict(self) -> dict[str, Any]:
        return _merge_extra(
            self.extra,
            {
                "name_or_path": self.name_or_path,
                "revision": self.revision,
                "dtype": self.dtype,
                "trust_remote_code": self.trust_remote_code,
                "attn_implementation": self.attn_implementation,
                "local_files_only": self.local_files_only,
            },
        )


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str | None = "sample-10BT"
    split: str = "train"
    text_column: str = "text"
    sequence_length: int = 512
    probe_train_sequences: int = 128
    probe_validation_sequences: int = 32
    local_train_sequences: int = 1024
    local_validation_sequences: int = 128
    global_train_sequences: int = 4096
    global_validation_sequences: int = 256
    seed: int | None = None
    num_proc: int = 1
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def split_sequences(self) -> dict[str, int]:
        """Canonical disjoint token-shard split sizes used by the data pipeline."""

        return {
            "probe_train": self.probe_train_sequences,
            "probe_validation": self.probe_validation_sequences,
            "local_train": self.local_train_sequences,
            "local_validation": self.local_validation_sequences,
            "global_train": self.global_train_sequences,
            "global_validation": self.global_validation_sequences,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "DataConfig":
        values = _as_mapping(raw, "data")
        dataset_name = _string(
            _take(values, "dataset_name", cls.dataset_name, aliases=("dataset",)),
            "data.dataset_name",
        )
        dataset_config = _string(
            _take(values, "dataset_config", cls.dataset_config, aliases=("subset",)),
            "data.dataset_config",
            allow_none=True,
        )
        split = _string(_take(values, "split", cls.split), "data.split")
        text_column = _string(
            _take(values, "text_column", cls.text_column), "data.text_column"
        )
        sequence_length = _integer(
            _take(values, "sequence_length", cls.sequence_length, aliases=("seq_length",)),
            "data.sequence_length",
            minimum=2,
        )
        count_fields = {
            "probe_train_sequences": cls.probe_train_sequences,
            "probe_validation_sequences": cls.probe_validation_sequences,
            "local_train_sequences": cls.local_train_sequences,
            "local_validation_sequences": cls.local_validation_sequences,
            "global_train_sequences": cls.global_train_sequences,
            "global_validation_sequences": cls.global_validation_sequences,
        }
        counts: dict[str, int] = {}
        for key, default in count_fields.items():
            parsed = _integer(_take(values, key, default), f"data.{key}", minimum=1)
            assert parsed is not None
            counts[key] = parsed
        seed = _integer(
            _take(values, "seed", cls.seed), "data.seed", minimum=0, allow_none=True
        )
        num_proc = _integer(_take(values, "num_proc", cls.num_proc), "data.num_proc", minimum=1)
        assert dataset_name is not None
        assert split is not None
        assert text_column is not None
        assert sequence_length is not None
        assert num_proc is not None
        return cls(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            text_column=text_column,
            sequence_length=sequence_length,
            seed=seed,
            num_proc=num_proc,
            extra=values,
            **counts,
        )

    def to_dict(self) -> dict[str, Any]:
        return _merge_extra(
            self.extra,
            {
                "dataset_name": self.dataset_name,
                "dataset_config": self.dataset_config,
                "split": self.split,
                "text_column": self.text_column,
                "sequence_length": self.sequence_length,
                "probe_train_sequences": self.probe_train_sequences,
                "probe_validation_sequences": self.probe_validation_sequences,
                "local_train_sequences": self.local_train_sequences,
                "local_validation_sequences": self.local_validation_sequences,
                "global_train_sequences": self.global_train_sequences,
                "global_validation_sequences": self.global_validation_sequences,
                "seed": self.seed,
                "num_proc": self.num_proc,
            },
        )


@dataclass(frozen=True)
class Stage1Config:
    enabled: bool = True
    threshold: float = 0.15
    probe_steps: int = 50
    long_probe_steps: int = 300
    max_group_size: int = 4
    batch_size: int = 16
    learning_rate: float = 1e-5
    protected_prefix_layers: int = 2
    protected_suffix_layers: int = 1
    initialization: str = "swm"
    scan_direction: str = "front_to_back"
    target_removed_layers: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "Stage1Config":
        values = _as_mapping(raw, "stages.stage1")
        enabled = _boolean(_take(values, "enabled", cls.enabled), "stages.stage1.enabled")
        threshold = _number(
            _take(values, "threshold", cls.threshold),
            "stages.stage1.threshold",
            minimum=0.0,
        )
        probe_steps = _integer(
            _take(values, "probe_steps", cls.probe_steps),
            "stages.stage1.probe_steps",
            minimum=0,
        )
        long_steps = _integer(
            _take(values, "long_probe_steps", cls.long_probe_steps),
            "stages.stage1.long_probe_steps",
            minimum=1,
        )
        max_group_size = _integer(
            _take(values, "max_group_size", cls.max_group_size),
            "stages.stage1.max_group_size",
            minimum=2,
        )
        batch_size = _integer(
            _take(values, "batch_size", cls.batch_size),
            "stages.stage1.batch_size",
            minimum=1,
        )
        learning_rate = _number(
            _take(values, "learning_rate", cls.learning_rate, aliases=("lr",)),
            "stages.stage1.learning_rate",
            minimum=0.0,
            minimum_exclusive=True,
        )
        protected_prefix = _integer(
            _take(values, "protected_prefix_layers", cls.protected_prefix_layers),
            "stages.stage1.protected_prefix_layers",
            minimum=0,
        )
        protected_suffix = _integer(
            _take(values, "protected_suffix_layers", cls.protected_suffix_layers),
            "stages.stage1.protected_suffix_layers",
            minimum=0,
        )
        initialization = _choice(
            _take(values, "initialization", cls.initialization, aliases=("init",)),
            "stages.stage1.initialization",
            {"swm", "copy_first", "copy_last"},
        )
        scan_direction = _choice(
            _take(values, "scan_direction", cls.scan_direction, aliases=("direction",)),
            "stages.stage1.scan_direction",
            {"front_to_back", "back_to_front"},
        )
        target_removed = _integer(
            _take(values, "target_removed_layers", cls.target_removed_layers),
            "stages.stage1.target_removed_layers",
            minimum=0,
            allow_none=True,
        )
        assert probe_steps is not None
        assert long_steps is not None
        assert max_group_size is not None
        assert batch_size is not None
        assert protected_prefix is not None
        assert protected_suffix is not None
        return cls(
            enabled,
            threshold,
            probe_steps,
            long_steps,
            max_group_size,
            batch_size,
            learning_rate,
            protected_prefix,
            protected_suffix,
            initialization,
            scan_direction,
            target_removed,
            values,
        )

    def to_dict(self) -> dict[str, Any]:
        return _merge_extra(
            self.extra,
            {
                "enabled": self.enabled,
                "threshold": self.threshold,
                "probe_steps": self.probe_steps,
                "long_probe_steps": self.long_probe_steps,
                "max_group_size": self.max_group_size,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "protected_prefix_layers": self.protected_prefix_layers,
                "protected_suffix_layers": self.protected_suffix_layers,
                "initialization": self.initialization,
                "scan_direction": self.scan_direction,
                "target_removed_layers": self.target_removed_layers,
            },
        )


@dataclass(frozen=True)
class Stage2Config:
    enabled: bool = True
    steps_per_group: int = 300
    batch_size: int = 16
    learning_rate: float = 1e-5
    lambda_start: float = 1.0
    lambda_end: float = 0.5
    lambda_schedule: str = "linear"
    eval_every: int = 25
    patience: int = 4
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "Stage2Config":
        values = _as_mapping(raw, "stages.stage2")
        enabled = _boolean(_take(values, "enabled", cls.enabled), "stages.stage2.enabled")
        steps = _integer(
            _take(values, "steps_per_group", cls.steps_per_group, aliases=("steps",)),
            "stages.stage2.steps_per_group",
            minimum=1,
        )
        batch_size = _integer(
            _take(values, "batch_size", cls.batch_size),
            "stages.stage2.batch_size",
            minimum=1,
        )
        learning_rate = _number(
            _take(values, "learning_rate", cls.learning_rate, aliases=("lr",)),
            "stages.stage2.learning_rate",
            minimum=0.0,
            minimum_exclusive=True,
        )
        lambda_start = _number(
            _take(values, "lambda_start", cls.lambda_start),
            "stages.stage2.lambda_start",
            minimum=0.0,
            maximum=1.0,
        )
        lambda_end = _number(
            _take(values, "lambda_end", cls.lambda_end),
            "stages.stage2.lambda_end",
            minimum=0.0,
            maximum=1.0,
        )
        schedule = _choice(
            _take(values, "lambda_schedule", cls.lambda_schedule),
            "stages.stage2.lambda_schedule",
            {"fixed", "linear", "cosine", "teacher_only"},
        )
        eval_every = _integer(
            _take(values, "eval_every", cls.eval_every),
            "stages.stage2.eval_every",
            minimum=1,
        )
        patience = _integer(
            _take(values, "patience", cls.patience),
            "stages.stage2.patience",
            minimum=1,
        )
        if schedule == "fixed" and lambda_start != lambda_end:
            raise ConfigError(
                "stages.stage2 fixed lambda schedule requires lambda_start == lambda_end"
            )
        assert steps is not None
        assert batch_size is not None
        assert eval_every is not None
        assert patience is not None
        return cls(
            enabled,
            steps,
            batch_size,
            learning_rate,
            lambda_start,
            lambda_end,
            schedule,
            eval_every,
            patience,
            values,
        )

    def to_dict(self) -> dict[str, Any]:
        return _merge_extra(
            self.extra,
            {
                "enabled": self.enabled,
                "steps_per_group": self.steps_per_group,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "lambda_start": self.lambda_start,
                "lambda_end": self.lambda_end,
                "lambda_schedule": self.lambda_schedule,
                "eval_every": self.eval_every,
                "patience": self.patience,
            },
        )


@dataclass(frozen=True)
class Stage3Config:
    enabled: bool = True
    steps: int = 1000
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    temperature: float = 1.0
    train_scope: str = "merged"
    gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = False
    eval_every: int = 50
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "Stage3Config":
        values = _as_mapping(raw, "stages.stage3")
        enabled = _boolean(_take(values, "enabled", cls.enabled), "stages.stage3.enabled")
        steps = _integer(
            _take(values, "steps", cls.steps), "stages.stage3.steps", minimum=1
        )
        batch_size = _integer(
            _take(values, "batch_size", cls.batch_size),
            "stages.stage3.batch_size",
            minimum=1,
        )
        accumulation = _integer(
            _take(values, "gradient_accumulation_steps", cls.gradient_accumulation_steps),
            "stages.stage3.gradient_accumulation_steps",
            minimum=1,
        )
        learning_rate = _number(
            _take(values, "learning_rate", cls.learning_rate, aliases=("lr",)),
            "stages.stage3.learning_rate",
            minimum=0.0,
            minimum_exclusive=True,
        )
        warmup_steps = _integer(
            _take(values, "warmup_steps", cls.warmup_steps),
            "stages.stage3.warmup_steps",
            minimum=0,
        )
        temperature = _number(
            _take(values, "temperature", cls.temperature),
            "stages.stage3.temperature",
            minimum=0.0,
            minimum_exclusive=True,
        )
        train_scope = _choice(
            _take(values, "train_scope", cls.train_scope),
            "stages.stage3.train_scope",
            {"all", "merged", "merged_only"},
        )
        if train_scope == "merged_only":
            train_scope = "merged"
        gradient_checkpointing = _boolean(
            _take(values, "gradient_checkpointing", cls.gradient_checkpointing),
            "stages.stage3.gradient_checkpointing",
        )
        use_8bit_optimizer = _boolean(
            _take(values, "use_8bit_optimizer", cls.use_8bit_optimizer),
            "stages.stage3.use_8bit_optimizer",
        )
        eval_every = _integer(
            _take(values, "eval_every", cls.eval_every),
            "stages.stage3.eval_every",
            minimum=1,
        )
        assert steps is not None
        assert batch_size is not None
        assert accumulation is not None
        assert warmup_steps is not None
        assert eval_every is not None
        if warmup_steps > steps:
            raise ConfigError("stages.stage3.warmup_steps cannot exceed stages.stage3.steps")
        return cls(
            enabled,
            steps,
            batch_size,
            accumulation,
            learning_rate,
            warmup_steps,
            temperature,
            train_scope,
            gradient_checkpointing,
            use_8bit_optimizer,
            eval_every,
            values,
        )

    def to_dict(self) -> dict[str, Any]:
        return _merge_extra(
            self.extra,
            {
                "enabled": self.enabled,
                "steps": self.steps,
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "learning_rate": self.learning_rate,
                "warmup_steps": self.warmup_steps,
                "temperature": self.temperature,
                "train_scope": self.train_scope,
                "gradient_checkpointing": self.gradient_checkpointing,
                "use_8bit_optimizer": self.use_8bit_optimizer,
                "eval_every": self.eval_every,
            },
        )


@dataclass(frozen=True)
class StagesConfig:
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "StagesConfig":
        values = _as_mapping(raw, "stages")
        stage1 = Stage1Config.from_dict(values.pop("stage1", None))
        stage2 = Stage2Config.from_dict(values.pop("stage2", None))
        stage3 = Stage3Config.from_dict(values.pop("stage3", None))
        return cls(stage1, stage2, stage3, values)

    def to_dict(self) -> dict[str, Any]:
        return _merge_extra(
            self.extra,
            {
                "stage1": self.stage1.to_dict(),
                "stage2": self.stage2.to_dict(),
                "stage3": self.stage3.to_dict(),
            },
        )


@dataclass(frozen=True)
class PathsConfig:
    output_dir: str = "outputs"
    cache_dir: str | None = None
    data_cache_dir: str | None = None
    checkpoint_dir: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "PathsConfig":
        values = _as_mapping(raw, "paths")
        output_dir = _path_string(
            _take(values, "output_dir", cls.output_dir), "paths.output_dir"
        )
        cache_dir = _path_string(
            _take(values, "cache_dir", cls.cache_dir), "paths.cache_dir", allow_none=True
        )
        data_cache_dir = _path_string(
            _take(values, "data_cache_dir", cls.data_cache_dir),
            "paths.data_cache_dir",
            allow_none=True,
        )
        checkpoint_dir = _path_string(
            _take(values, "checkpoint_dir", cls.checkpoint_dir),
            "paths.checkpoint_dir",
            allow_none=True,
        )
        assert output_dir is not None
        return cls(output_dir, cache_dir, data_cache_dir, checkpoint_dir, values)

    def to_dict(self) -> dict[str, Any]:
        return _merge_extra(
            self.extra,
            {
                "output_dir": self.output_dir,
                "cache_dir": self.cache_dir,
                "data_cache_dir": self.data_cache_dir,
                "checkpoint_dir": self.checkpoint_dir,
            },
        )


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    seed: int | None = None
    num_workers: int = 2
    deterministic: bool = True
    log_every: int = 10
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "RuntimeConfig":
        values = _as_mapping(raw, "runtime")
        device = _string(_take(values, "device", cls.device), "runtime.device")
        seed = _integer(
            _take(values, "seed", cls.seed), "runtime.seed", minimum=0, allow_none=True
        )
        workers = _integer(
            _take(values, "num_workers", cls.num_workers), "runtime.num_workers", minimum=0
        )
        deterministic = _boolean(
            _take(values, "deterministic", cls.deterministic), "runtime.deterministic"
        )
        log_every = _integer(
            _take(values, "log_every", cls.log_every), "runtime.log_every", minimum=1
        )
        assert device is not None
        assert workers is not None
        assert log_every is not None
        return cls(device, seed, workers, deterministic, log_every, values)

    def to_dict(self) -> dict[str, Any]:
        return _merge_extra(
            self.extra,
            {
                "device": self.device,
                "seed": self.seed,
                "num_workers": self.num_workers,
                "deterministic": self.deterministic,
                "log_every": self.log_every,
            },
        )


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    seed: int
    model: ModelConfig
    data: DataConfig
    stages: StagesConfig
    paths: PathsConfig
    runtime: RuntimeConfig
    evaluation: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)
    _raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    source_path: Path | None = field(default=None, repr=False, compare=False)

    @property
    def stage1(self) -> Stage1Config:
        return self.stages.stage1

    @property
    def stage2(self) -> Stage2Config:
        return self.stages.stage2

    @property
    def stage3(self) -> Stage3Config:
        return self.stages.stage3

    @property
    def output_dir(self) -> Path:
        return Path(self.paths.output_dir)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    @property
    def raw(self) -> dict[str, Any]:
        """Return a defensive copy of the resolved input mapping."""

        return copy.deepcopy(dict(self._raw))

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        source_path: str | Path | None = None,
    ) -> "ExperimentConfig":
        resolved = _as_mapping(raw, "config")
        values = copy.deepcopy(resolved)

        experiment = _as_mapping(values.pop("experiment", None), "experiment")
        for key in ("run_name", "seed"):
            if key in values and key in experiment:
                raise ConfigError(f"{key} is set both at top level and under experiment")
        run_name_default = Path(source_path).stem if source_path is not None else "experiment"
        if "run_name" in values:
            run_name_raw = values.pop("run_name")
        elif "run_name" in experiment:
            run_name_raw = experiment.pop("run_name")
        else:
            run_name_raw = experiment.pop("name", run_name_default)
        if "seed" in values:
            seed_raw = values.pop("seed")
        else:
            seed_raw = experiment.pop("seed", 42)
        run_name = _string(
            run_name_raw,
            "run_name",
        )
        seed = _integer(seed_raw, "seed", minimum=0)
        if experiment:
            values["experiment"] = experiment

        model = ModelConfig.from_dict(values.pop("model", None))
        data = DataConfig.from_dict(values.pop("data", None))

        stages_raw = _as_mapping(values.pop("stages", None), "stages")
        for stage_name in ("stage1", "stage2", "stage3"):
            if stage_name in values:
                if stage_name in stages_raw:
                    top_level_stage = _as_mapping(values.pop(stage_name), stage_name)
                    nested_stage = _as_mapping(stages_raw[stage_name], f"stages.{stage_name}")
                    top_level_stage.update(nested_stage)
                    stages_raw[stage_name] = top_level_stage
                else:
                    stages_raw[stage_name] = values.pop(stage_name)
        stages = StagesConfig.from_dict(stages_raw)

        paths_raw = _as_mapping(values.pop("paths", None), "paths")
        if "output_dir" in values:
            if "output_dir" in paths_raw:
                raise ConfigError("output_dir is set both at top level and under paths")
            paths_raw["output_dir"] = values.pop("output_dir")
        paths = PathsConfig.from_dict(paths_raw)
        runtime = RuntimeConfig.from_dict(values.pop("runtime", None))
        evaluation = _as_mapping(values.pop("evaluation", None), "evaluation")

        assert run_name is not None
        assert seed is not None
        return cls(
            run_name=run_name,
            seed=seed,
            model=model,
            data=data,
            stages=stages,
            paths=paths,
            runtime=runtime,
            evaluation=evaluation,
            extra=values,
            _raw=resolved,
            source_path=Path(source_path) if source_path is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable canonical mapping, retaining all unknown keys."""

        return _merge_extra(
            self.extra,
            {
                "run_name": self.run_name,
                "seed": self.seed,
                "model": self.model.to_dict(),
                "data": self.data.to_dict(),
                "stages": self.stages.to_dict(),
                "paths": self.paths.to_dict(),
                "runtime": self.runtime.to_dict(),
                "evaluation": copy.deepcopy(dict(self.evaluation)),
            },
        )


def _set_override(mapping: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ConfigError(f"override must have KEY=VALUE form: {expression!r}")
    dotted_key, encoded_value = expression.split("=", 1)
    parts = dotted_key.split(".")
    if any(not part for part in parts):
        raise ConfigError(f"override has an invalid key: {expression!r}")
    try:
        value = yaml.safe_load(encoded_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML value in override {expression!r}: {exc}") from exc

    cursor = mapping
    for part in parts[:-1]:
        current = cursor.get(part, _MISSING)
        if current is _MISSING:
            current = {}
            cursor[part] = current
        if not isinstance(current, dict):
            raise ConfigError(
                f"cannot set {dotted_key!r}: {part!r} is not a mapping"
            )
        cursor = current
    cursor[parts[-1]] = value


def load_config(
    path: str | Path,
    overrides: Sequence[str] | None = None,
) -> ExperimentConfig:
    """Load YAML from ``path``, apply dotted ``KEY=VALUE`` overrides, and validate it."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"could not read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if loaded is None:
        loaded = {}
    resolved = _as_mapping(loaded, "config")
    for expression in overrides or ():
        _set_override(resolved, expression)
    return ExperimentConfig.from_dict(resolved, source_path=config_path)


__all__ = [
    "ConfigError",
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "PathsConfig",
    "RuntimeConfig",
    "Stage1Config",
    "Stage2Config",
    "Stage3Config",
    "StagesConfig",
    "load_config",
]
