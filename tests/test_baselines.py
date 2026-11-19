from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llm_compression.baselines import (
    average_blocks,
    build_student,
    get_layers,
    make_naive_drop_layout,
    make_xmerge_layout,
    select_xmerge_pairs,
    set_layers,
    swm_merge_blocks,
)
from llm_compression.layout import Layout, Span


class _Attention(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx


class _ScalarBlock(nn.Module):
    def __init__(self, value: float, layer_idx: int = -1) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value], dtype=torch.float32))
        self.register_buffer("integer_marker", torch.tensor(7, dtype=torch.int64))
        self.self_attn = _Attention(layer_idx)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * self.weight


class _Backbone(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_ScalarBlock(value, index) for index, value in enumerate(values)]
        )
        self.config = SimpleNamespace(num_hidden_layers=len(values))


class _ToyLM(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.model = _Backbone(values)
        self.config = SimpleNamespace(num_hidden_layers=len(values))


class _NeoXToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gpt_neox = nn.Module()
        self.gpt_neox.layers = nn.ModuleList([_ScalarBlock(1), _ScalarBlock(2)])
        self.config = SimpleNamespace(num_hidden_layers=2)


def _weights(model: nn.Module) -> list[float]:
    return [float(block.weight.item()) for block in get_layers(model)]


def test_model_agnostic_layer_access_set_and_reindex() -> None:
    model = _NeoXToy()
    assert len(get_layers(model)) == 2

    set_layers(model, [_ScalarBlock(4), _ScalarBlock(5), _ScalarBlock(6)])

    assert _weights(model) == [4, 5, 6]
    assert model.config.num_hidden_layers == 3
    assert [block.self_attn.layer_idx for block in get_layers(model)] == [0, 1, 2]


def test_explicit_layer_path_supports_nonstandard_model() -> None:
    model = nn.Module()
    model.stack = nn.Module()
    model.stack.units = nn.ModuleList([_ScalarBlock(1), _ScalarBlock(2)])

    assert len(get_layers(model, "stack.units")) == 2
    set_layers(model, [_ScalarBlock(3)], "stack.units")
    assert get_layers(model, "stack.units")[0].weight.item() == 3


def test_average_and_swm_arithmetic_are_distinct_and_exact() -> None:
    blocks = [_ScalarBlock(1), _ScalarBlock(3), _ScalarBlock(7)]

    averaged = average_blocks(blocks)
    swm = swm_merge_blocks(blocks)

    assert averaged.weight.item() == pytest.approx(11 / 3)
    # Eq. 4: 1 + (3 - 1) + (7 - 1) = 9, not the arithmetic mean.
    assert swm.weight.item() == pytest.approx(9)
    assert swm.integer_marker.item() == 7
    swm.weight.data.fill_(100)
    assert blocks[0].weight.item() == 1


def test_swm_two_block_window_equals_second_block_parameters() -> None:
    merged = swm_merge_blocks([_ScalarBlock(-2), _ScalarBlock(5)])
    assert merged.weight.item() == pytest.approx(5)


def test_nonfloating_state_must_agree_for_weight_arithmetic() -> None:
    left, right = _ScalarBlock(1), _ScalarBlock(2)
    right.integer_marker.fill_(8)
    with pytest.raises(ValueError, match="Non-floating"):
        swm_merge_blocks([left, right])
    with pytest.raises(ValueError, match="Non-floating"):
        average_blocks([left, right])


def test_build_student_swm_and_drop_have_identical_depth_and_spans() -> None:
    teacher = _ToyLM([1, 2, 4, 8, 16, 32])
    layout = Layout(
        num_layers=6,
        spans=(Span(0, 1), Span(1, 4), Span(4, 5), Span(5, 6)),
        protected_layers=frozenset({0, 5}),
    )

    merged = build_student(teacher, layout, mode="merged")
    drop_layout = make_naive_drop_layout(layout)
    dropped = build_student(teacher, drop_layout, mode="drop")

    assert _weights(merged) == pytest.approx([1, 10, 16, 32])
    assert _weights(dropped) == pytest.approx([1, 2, 16, 32])
    assert len(get_layers(merged)) == len(get_layers(dropped)) == layout.compressed_num_layers
    assert layout.removed_layers == drop_layout.removed_layers == 2
    assert teacher.config.num_hidden_layers == 6
    assert len(get_layers(teacher)) == 6
    assert merged.config.num_hidden_layers == dropped.config.num_hidden_layers == 4
    assert [block.self_attn.layer_idx for block in get_layers(merged)] == [0, 1, 2, 3]


def test_build_student_loads_trained_merged_state(tmp_path) -> None:
    teacher = _ToyLM([1, 2, 4, 8])
    layout = Layout(4, (Span(0, 1), Span(1, 3), Span(3, 4)))
    trained = _ScalarBlock(42)
    checkpoint = tmp_path / "block.pt"
    torch.save({"block_state_dict": trained.state_dict()}, checkpoint)

    student = build_student(
        teacher,
        layout,
        merged_state_paths={(1, 3): checkpoint},
        mode="merged",
    )

    assert _weights(student) == pytest.approx([1, 42, 8])


def test_build_student_rejects_layout_for_different_teacher_depth() -> None:
    with pytest.raises(ValueError, match="Layout describes"):
        build_student(_ToyLM([1, 2]), Layout.identity(3))


def test_xmerge_cross_axis_selection_uses_smallest_max_z_score() -> None:
    # RM and BI agree here, making layers 2 then 4 the lowest combined scores.
    rm = [10, 9, -10, 8, -9, 7, 6, 5]
    bi = [9, 8, -8, 7, -7, 6, 5, 4]
    pairs = select_xmerge_pairs(
        8,
        2,
        protected_prefix=1,
        protected_suffix=1,
        rm_scores=rm,
        bi_scores=bi,
        pair_direction="right",
    )

    assert [pair.removed_index for pair in pairs] == [2, 4]
    assert [(pair.span.start, pair.span.end) for pair in pairs] == [(2, 4), (4, 6)]
    layout = make_xmerge_layout(
        8,
        2,
        protected_prefix=1,
        protected_suffix=1,
        rm_scores=rm,
        bi_scores=bi,
    )
    assert layout.removed_layers == 2
    assert all(span.length == 2 for span in layout.merged_spans)
    assert layout.protected_layers == frozenset({0, 7})


def test_xmerge_structural_layout_has_exact_requested_removal() -> None:
    layout = make_xmerge_layout(7, 3)
    assert layout.removed_layers == 3
    assert layout.compressed_num_layers == 4
    assert [(span.start, span.end) for span in layout.merged_spans] == [
        (0, 2),
        (2, 4),
        (4, 6),
    ]


def test_xmerge_zero_removal_is_identity() -> None:
    layout = make_xmerge_layout(4, 0)
    assert layout == Layout.identity(4)


def test_xmerge_rejects_more_pairs_than_fit() -> None:
    with pytest.raises(ValueError, match="capacity"):
        make_xmerge_layout(8, 3, protected_prefix=2, protected_suffix=1)
