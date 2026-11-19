from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import llm_compression.evaluation as evaluation
from llm_compression.evaluation import (
    COMMONSENSE_TASKS,
    LmEvalConfig,
    aggregate_lm_eval_results,
    benchmark_model,
    build_lm_eval_command,
    count_parameters,
    default_lm_eval_configs,
    evaluate_perplexity,
    generate_records,
    parameter_counts,
    relative_accuracy,
    shifted_cross_entropy,
    summarize_proposal_benchmarks,
    write_generation_records,
)


class _UniformLM(nn.Module):
    def __init__(self, vocabulary: int = 5) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocabulary = vocabulary

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> SimpleNamespace:
        logits = torch.zeros(
            *input_ids.shape,
            self.vocabulary,
            device=input_ids.device,
        ) + self.anchor
        return SimpleNamespace(logits=logits)


class _GeneratingLM(_UniformLM):
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        suffix = torch.tensor([[3, 4]], device=input_ids.device).expand(input_ids.shape[0], -1)
        return torch.cat((input_ids, suffix), dim=-1)


class _Tokenizer:
    def __call__(self, prompt: str, return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        width = max(1, len(prompt.split()))
        return {
            "input_ids": torch.arange(1, width + 1).unsqueeze(0),
            "attention_mask": torch.ones(1, width, dtype=torch.long),
        }

    def decode(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return " ".join(f"tok-{token_id}" for token_id in token_ids)


def test_shifted_cross_entropy_matches_manual_causal_shift_and_mask() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [-1.0, 3.0, 0.0],
                [0.5, 0.5, 0.5],
                [9.0, -9.0, -9.0],
            ]
        ]
    )
    input_ids = torch.tensor([[2, 0, 1, 2]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    actual = shifted_cross_entropy(logits, input_ids, attention_mask)
    expected = F.cross_entropy(logits[0, :2].float(), torch.tensor([0, 1]))

    assert actual.item() == pytest.approx(expected.item())


def test_shifted_cross_entropy_excludes_left_padding_transition() -> None:
    logits = torch.zeros(1, 4, 4)
    ids = torch.tensor([[0, 0, 2, 3]])
    mask = torch.tensor([[0, 0, 1, 1]])
    loss = shifted_cross_entropy(logits, ids, mask)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(4.0)).item())


def test_shifted_cross_entropy_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match="No valid"):
        shifted_cross_entropy(
            torch.zeros(1, 3, 4),
            torch.tensor([[1, 2, 3]]),
            torch.zeros(1, 3),
        )


def test_evaluate_perplexity_is_token_weighted_and_restores_training_state() -> None:
    model = _UniformLM(vocabulary=5)
    model.train()
    batches = [
        {
            "input_ids": torch.tensor([[0, 1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        },
        {
            "input_ids": torch.tensor([[4, 3, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        },
    ]

    perplexity = evaluate_perplexity(model, batches, "cpu")

    assert perplexity == pytest.approx(5.0)
    assert model.training


def test_parameter_counts_report_total_trainable_and_frozen() -> None:
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1, bias=False))
    model[0].bias.requires_grad_(False)

    assert count_parameters(model) == 10
    assert count_parameters(model, trainable_only=True) == 8
    assert parameter_counts(model) == {"total": 10, "trainable": 8, "frozen": 2}


def test_benchmark_uses_barriers_around_every_measured_call(monkeypatch) -> None:
    barriers: list[str] = []
    monkeypatch.setattr(evaluation, "_synchronize", lambda device: barriers.append(str(device)))
    model = _UniformLM()
    model.train()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    result = benchmark_model(model, batch, "cpu", warmup=1, repetitions=2)

    assert barriers == ["cpu", "cpu", "cpu", "cpu"]
    assert result["mode"] == "forward"
    assert result["repetitions"] == 2
    assert result["tokens_timed"] == 6
    assert result["mean_seconds"] >= 0
    assert model.training


def test_generation_benchmark_counts_only_new_tokens() -> None:
    model = _GeneratingLM()
    result = benchmark_model(
        model,
        {"input_ids": torch.tensor([[1, 2, 3]])},
        "cpu",
        warmup=0,
        repetitions=3,
        generation_kwargs={"max_new_tokens": 2},
    )
    assert result["mode"] == "generate"
    assert result["tokens_timed"] == 6


def test_generation_records_are_auditable_and_jsonl_serializable(tmp_path) -> None:
    records = generate_records(
        _GeneratingLM(),
        _Tokenizer(),
        ["one two", "three"],
        "cpu",
        generation_kwargs={"max_new_tokens": 2, "do_sample": False},
    )

    assert records[0]["prompt"] == "one two"
    assert records[0]["generated_token_ids"] == [3, 4]
    assert records[0]["generated_text"] == "tok-3 tok-4"
    assert records[0]["prompt_tokens"] == 2
    assert records[0]["generated_tokens"] == 2
    destination = write_generation_records(records, tmp_path / "generations.jsonl")
    assert destination.read_text().count("\n") == 2


def test_lm_eval_command_is_shell_safe_argv_and_keeps_shot_count() -> None:
    config = LmEvalConfig(
        tasks=("boolq", "piqa"),
        model_args={"pretrained": "/models/student with spaces", "dtype": "bfloat16"},
        num_fewshot=0,
        batch_size=4,
        device="cuda:0",
        output_path="results/out.json",
        log_samples=True,
    )

    command = build_lm_eval_command(config)

    assert command[0] == "lm_eval"
    assert command[command.index("--tasks") + 1] == "boolq,piqa"
    assert command[command.index("--num_fewshot") + 1] == "0"
    assert command[command.index("--model_args") + 1] == (
        "pretrained=/models/student with spaces,dtype=bfloat16"
    )
    assert "--log_samples" in command


def test_default_lm_eval_configs_separate_zero_shot_and_five_shot_runs(tmp_path) -> None:
    commonsense, mmlu = default_lm_eval_configs(
        "student", tmp_path, revision="immutable-commit"
    )
    assert commonsense.tasks == COMMONSENSE_TASKS
    assert commonsense.num_fewshot == 0
    assert commonsense.model_args["revision"] == "immutable-commit"
    assert mmlu.tasks == ("mmlu",)
    assert mmlu.num_fewshot == 5
    assert mmlu.model_args["revision"] == "immutable-commit"


def test_relative_accuracy_is_student_over_teacher_times_100() -> None:
    assert relative_accuracy(0.60, 0.75) == pytest.approx(80.0)
    assert relative_accuracy(60, 75) == pytest.approx(80.0)
    with pytest.raises(ZeroDivisionError):
        relative_accuracy(0.5, 0.0)


def test_aggregate_lm_eval_uses_task_metrics_and_macro_retention() -> None:
    student = {
        "results": {
            "boolq": {"acc,none": 0.60, "acc_stderr,none": 0.01},
            "piqa": {"acc,none": 0.40, "acc_norm,none": 0.50},
        }
    }
    teacher = {
        "results": {
            "boolq": {"acc,none": 0.75},
            "piqa": {"acc,none": 0.45, "acc_norm,none": 0.50},
        }
    }

    summary = aggregate_lm_eval_results(
        student,
        teacher,
        tasks=("boolq", "piqa"),
    )

    assert summary["task_accuracies"] == {"boolq": 0.60, "piqa": 0.50}
    assert summary["average_accuracy"] == pytest.approx(0.55)
    assert summary["teacher_average_accuracy"] == pytest.approx(0.625)
    assert summary["relative_task_accuracy_pct"] == pytest.approx(
        {"boolq": 80.0, "piqa": 100.0}
    )
    assert summary["relative_accuracy_pct"] == pytest.approx(88.0)
    assert summary["relative_accuracy_formula"] == (
        "student_macro_accuracy / teacher_macro_accuracy * 100"
    )


def test_aggregate_lm_eval_can_read_separate_json_files(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"results": {"boolq": {"acc,none": 0.7}}}')
    second.write_text('{"results": {"piqa": {"acc_norm,none": 0.5}}}')

    summary = aggregate_lm_eval_results([first, second], tasks=("boolq", "piqa"))

    assert summary["num_tasks"] == 2
    assert summary["average_accuracy"] == pytest.approx(0.6)


def test_proposal_summary_keeps_commonsense_and_mmlu_separate() -> None:
    student_results = {
        "results": {
            **{task: {"acc,none": 0.5} for task in COMMONSENSE_TASKS},
            "mmlu_abstract_algebra": {"acc,none": 0.2},
        },
        "groups": {"mmlu": {"acc,none": 0.4}},
    }
    teacher_results = {
        "results": {
            **{task: {"acc,none": 0.625} for task in COMMONSENSE_TASKS},
            "mmlu_abstract_algebra": {"acc,none": 0.5},
        },
        "groups": {"mmlu": {"acc,none": 0.5}},
    }

    summary = summarize_proposal_benchmarks(student_results, teacher_results)

    commonsense = summary["commonsense_zero_shot"]
    mmlu = summary["mmlu_five_shot"]
    assert commonsense["num_tasks"] == 7
    assert commonsense["relative_accuracy_pct"] == pytest.approx(80.0)
    assert mmlu["task_accuracies"] == {"mmlu": 0.4}
    assert mmlu["relative_accuracy_pct"] == pytest.approx(80.0)
