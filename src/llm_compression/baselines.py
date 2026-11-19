"""Construction helpers for depth-compression baselines.

The functions in this module intentionally operate on ordinary PyTorch
modules rather than importing a particular Transformers model class.  Qwen,
Llama/Mistral, GPT-2, and GPT-NeoX style layer locations are recognized; an
explicit dotted path can be supplied for another architecture.

``Layout`` spans are half-open original-layer intervals.  A singleton span is
copied unchanged and a longer span becomes exactly one student block, so every
materialization of a given layout has the same depth and removal count.
"""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .layout import Layout, Span


_COMMON_LAYER_PATHS = (
    "model.layers",  # Qwen, Llama, Mistral, Gemma
    "gpt_neox.layers",  # Pythia / GPT-NeoX
    "transformer.h",  # GPT-2 and many GPT-style models
    "model.decoder.layers",  # OPT and encoder-decoder backbones
    "decoder.layers",
    "transformer.blocks",
    "layers",
    "blocks",
    "module.model.layers",  # a simply wrapped model
)


def _resolve_dotted(root: object, path: str) -> object:
    current = root
    for component in path.split("."):
        if not component:
            raise ValueError(f"Invalid empty component in layer path {path!r}")
        try:
            current = getattr(current, component)
        except AttributeError as exc:
            raise AttributeError(
                f"Could not resolve decoder-layer path {path!r}: "
                f"{type(current).__name__} has no attribute {component!r}"
            ) from exc
    return current


def _find_layer_path(model: nn.Module, layer_path: str | None = None) -> str:
    if layer_path is not None:
        value = _resolve_dotted(model, layer_path)
        if not isinstance(value, nn.ModuleList):
            raise TypeError(
                f"Layer path {layer_path!r} resolved to {type(value).__name__}, "
                "not torch.nn.ModuleList"
            )
        return layer_path

    for candidate in _COMMON_LAYER_PATHS:
        try:
            value = _resolve_dotted(model, candidate)
        except AttributeError:
            continue
        if isinstance(value, nn.ModuleList):
            return candidate

    configured_depth = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    possible: list[str] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.ModuleList) or not len(module):
            continue
        if not all(isinstance(item, nn.Module) for item in module):
            continue
        if configured_depth is None or len(module) == int(configured_depth):
            possible.append(name)
    if len(possible) == 1:
        return possible[0]
    if possible:
        raise TypeError(
            "Decoder block container is ambiguous; pass layer_path explicitly. "
            f"Candidates: {possible}"
        )
    raise TypeError(
        "Could not locate a decoder-block ModuleList. Pass its dotted path via layer_path."
    )


def _resolve_owner(model: nn.Module, layer_path: str) -> tuple[object, str]:
    components = layer_path.split(".")
    owner_path = ".".join(components[:-1])
    owner = _resolve_dotted(model, owner_path) if owner_path else model
    return owner, components[-1]


def get_layers(model: nn.Module, layer_path: str | None = None) -> nn.ModuleList:
    """Return a model's decoder blocks without assuming a model class.

    Args:
        model: A causal language model or its backbone.
        layer_path: Optional dotted path to an ``nn.ModuleList``.  When omitted,
            common Hugging Face paths are tried before a conservative structural
            fallback.
    """

    path = _find_layer_path(model, layer_path)
    layers = _resolve_dotted(model, path)
    assert isinstance(layers, nn.ModuleList)  # established by _find_layer_path
    return layers


def _set_depth_on_config(config: object, depth: int) -> None:
    if config is None:
        return
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = depth
    if hasattr(config, "n_layer"):
        config.n_layer = depth
    layer_types = getattr(config, "layer_types", None)
    if isinstance(layer_types, list) and len(layer_types) != depth:
        # Hybrid-attention models carry one entry per decoder block.  Preserve
        # the existing pattern rather than leaving a stale-length list.
        if layer_types:
            config.layer_types = [layer_types[min(i, len(layer_types) - 1)] for i in range(depth)]


def reindex_layers(
    model_or_layers: nn.Module | Sequence[nn.Module],
    layer_path: str | None = None,
) -> None:
    """Renumber per-block cache indices after changing model depth.

    The function recognizes the common ``layer_idx`` field on a block and on
    its ``self_attn``, ``attn``, or ``attention`` child.  Missing fields are
    harmless.  When a full model is passed, depth metadata in its config and
    backbone config is also updated.
    """

    if isinstance(model_or_layers, nn.ModuleList) or not isinstance(model_or_layers, nn.Module):
        layers = list(model_or_layers)
        model: nn.Module | None = None
        path: str | None = None
    else:
        model = model_or_layers
        path = _find_layer_path(model, layer_path)
        layers = list(get_layers(model, path))

    for index, layer in enumerate(layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = index
        for name in ("self_attn", "attn", "attention"):
            attention = getattr(layer, name, None)
            if attention is not None and hasattr(attention, "layer_idx"):
                attention.layer_idx = index

    if model is not None:
        _set_depth_on_config(getattr(model, "config", None), len(layers))
        assert path is not None
        owner, _ = _resolve_owner(model, path)
        _set_depth_on_config(getattr(owner, "config", None), len(layers))
        first_component = path.split(".")[0]
        backbone = getattr(model, first_component, None)
        _set_depth_on_config(getattr(backbone, "config", None), len(layers))


def set_layers(
    model: nn.Module,
    layers: Iterable[nn.Module],
    layer_path: str | None = None,
) -> nn.Module:
    """Replace decoder blocks, update cache indices/config, and return ``model``."""

    path = _find_layer_path(model, layer_path)
    owner, attribute = _resolve_owner(model, path)
    replacement = nn.ModuleList(list(layers))
    setattr(owner, attribute, replacement)
    reindex_layers(model, path)
    return model


def average_blocks(blocks: Sequence[nn.Module]) -> nn.Module:
    """Return a uniform parameter-average ablation initialization.

    The arithmetic is exact and intentionally simple.  For every floating or
    complex state entry :math:`x` (parameters *and* floating buffers) in ``m``
    structurally identical blocks, the returned entry is

    .. math::

        x_{merge} = (x_0 + x_1 + ... + x_{m-1}) / m.

    Accumulation is performed in float32 for fp16/bf16 inputs and in the
    original dtype otherwise, then cast back to the first block's dtype and
    device.  Non-floating buffers are not arithmetically meaningful: they must
    agree exactly and are copied from the first block.  Python attributes also
    come from a deep copy of the first block and are subsequently reindexed by
    :func:`set_layers` when installed in a model.

    This is an explicit ablation helper, *not* the SWM rule.  The published SWM
    arithmetic is implemented by :func:`swm_merge_blocks` below.
    """

    if not blocks:
        raise ValueError("average_blocks requires at least one block")
    if not all(isinstance(block, nn.Module) for block in blocks):
        raise TypeError("Every item passed to average_blocks must be an nn.Module")

    states = [block.state_dict() for block in blocks]
    reference_keys = tuple(states[0])
    for index, state in enumerate(states[1:], start=1):
        if tuple(state) != reference_keys:
            raise ValueError(f"Block {index} has a different state_dict structure")

    averaged: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key in reference_keys:
            values = [state[key] for state in states]
            reference = values[0]
            if any(value.shape != reference.shape for value in values[1:]):
                raise ValueError(f"State entry {key!r} has incompatible shapes")
            if any(value.dtype != reference.dtype for value in values[1:]):
                raise ValueError(f"State entry {key!r} has incompatible dtypes")

            if reference.is_floating_point() or reference.is_complex():
                if reference.dtype in (torch.float16, torch.bfloat16):
                    accumulation_dtype = torch.float32
                else:
                    accumulation_dtype = reference.dtype
                accumulator = torch.zeros_like(reference, dtype=accumulation_dtype)
                for value in values:
                    accumulator.add_(value.to(device=reference.device, dtype=accumulation_dtype))
                averaged[key] = accumulator.div_(len(values)).to(dtype=reference.dtype)
            else:
                if any(
                    not torch.equal(value.to(device=reference.device), reference)
                    for value in values[1:]
                ):
                    raise ValueError(
                        f"Non-floating state entry {key!r} differs between blocks; "
                        "weight arithmetic is undefined"
                    )
                averaged[key] = reference.detach().clone()

    merged = copy.deepcopy(blocks[0])
    merged.load_state_dict(averaged, strict=True)
    return merged


def swm_merge_blocks(blocks: Sequence[nn.Module]) -> nn.Module:
    r"""Apply Sliding-Window Merging's parameter arithmetic exactly.

    For a window of ``m`` blocks with the first block as base, SWM Eq. 4 is

    .. math::

        \theta^* = \theta_0 + \sum_{k=1}^{m-1}(\theta_k-\theta_0)
                 = \sum_{k=0}^{m-1}\theta_k - (m-1)\theta_0.

    Thus a two-block merge initializes to the second block; a three-block
    merge is ``theta_1 + theta_2 - theta_0``.  This must not be replaced by a
    uniform mean.  The equation is applied entrywise to every floating or
    complex state entry, with fp16/bf16 accumulation promoted to float32.
    Non-floating buffers must agree and are copied from the base block.
    """

    if not blocks:
        raise ValueError("swm_merge_blocks requires at least one block")
    if not all(isinstance(block, nn.Module) for block in blocks):
        raise TypeError("Every item passed to swm_merge_blocks must be an nn.Module")

    states = [block.state_dict() for block in blocks]
    reference_keys = tuple(states[0])
    for index, state in enumerate(states[1:], start=1):
        if tuple(state) != reference_keys:
            raise ValueError(f"Block {index} has a different state_dict structure")

    merged_state: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key in reference_keys:
            values = [state[key] for state in states]
            reference = values[0]
            if any(value.shape != reference.shape for value in values[1:]):
                raise ValueError(f"State entry {key!r} has incompatible shapes")
            if any(value.dtype != reference.dtype for value in values[1:]):
                raise ValueError(f"State entry {key!r} has incompatible dtypes")

            if reference.is_floating_point() or reference.is_complex():
                accumulation_dtype = (
                    torch.float32
                    if reference.dtype in (torch.float16, torch.bfloat16)
                    else reference.dtype
                )
                base = reference.to(dtype=accumulation_dtype)
                accumulator = base.clone()
                for value in values[1:]:
                    accumulator.add_(
                        value.to(device=reference.device, dtype=accumulation_dtype) - base
                    )
                merged_state[key] = accumulator.to(dtype=reference.dtype)
            else:
                if any(
                    not torch.equal(value.to(device=reference.device), reference)
                    for value in values[1:]
                ):
                    raise ValueError(
                        f"Non-floating state entry {key!r} differs between blocks; "
                        "SWM arithmetic is undefined"
                    )
                merged_state[key] = reference.detach().clone()

    merged = copy.deepcopy(blocks[0])
    merged.load_state_dict(merged_state, strict=True)
    return merged


def _span_key(span: Span) -> tuple[int, int]:
    return int(span.start), int(span.end)


def _checkpoint_for_span(
    span: Span,
    merged_index: int,
    merged_state_paths: Mapping[Any, Any] | Sequence[Any] | None,
) -> Any | None:
    if merged_state_paths is None:
        return None
    if not isinstance(merged_state_paths, Mapping):
        try:
            return merged_state_paths[merged_index]
        except IndexError as exc:
            raise ValueError("Not enough merged_state_paths for the layout's merged spans") from exc

    candidates: tuple[Any, ...] = (
        span,
        _span_key(span),
        f"{span.start}:{span.end}",
        f"{span.start}-{span.end}",
        merged_index,
    )
    for candidate in candidates:
        try:
            if candidate in merged_state_paths:
                return merged_state_paths[candidate]
        except TypeError:
            # A non-hashable Span implementation is still addressable via tuple/string.
            continue
    return None


def _load_state(source: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(source, nn.Module):
        return source.state_dict()
    if isinstance(source, (str, Path)):
        try:
            source = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - for older supported PyTorch releases
            source = torch.load(source, map_location="cpu")
    if not isinstance(source, Mapping):
        raise TypeError("A merged block checkpoint must be a module, mapping, or filesystem path")
    for key in ("block_state_dict", "state_dict", "model_state_dict", "model"):
        nested = source.get(key)
        if isinstance(nested, Mapping):
            source = nested
            break
    if not all(isinstance(value, torch.Tensor) for value in source.values()):
        raise TypeError("Merged block state dict contains non-tensor values")
    return source  # type: ignore[return-value]


def build_student(
    teacher: nn.Module,
    layout: Layout,
    merged_state_paths: Mapping[Any, Any] | Sequence[Any] | None = None,
    mode: str = "merged",
    *,
    layer_path: str | None = None,
) -> nn.Module:
    """Materialize an independent student at the layout's exact compression.

    ``mode="merged"`` uses :func:`swm_merge_blocks` for every non-singleton span,
    optionally replacing that initialization with a trained state from
    ``merged_state_paths``.  A sequence of paths/states is aligned with
    ``layout.merged_spans``; a mapping may use ``Span``, ``(start, end)``,
    ``"start:end"``, or merged-span index keys.

    ``mode="drop"`` is the equal-compression keep-one/drop control: for every
    span it keeps the original block at ``span.start`` and drops the remaining
    ``span.length - 1`` blocks.  Singleton handling and span boundaries are
    identical in both modes, so depth/removal count cannot drift between the
    learned merge and its control.
    """

    if mode not in {"merged", "drop"}:
        raise ValueError("mode must be either 'merged' or 'drop'")

    student = copy.deepcopy(teacher)
    path = _find_layer_path(student, layer_path)
    original_layers = list(get_layers(student, path))
    if int(layout.num_layers) != len(original_layers):
        raise ValueError(
            f"Layout describes {layout.num_layers} original layers, "
            f"but teacher exposes {len(original_layers)}"
        )

    new_layers: list[nn.Module] = []
    merged_index = 0
    for span in layout.spans:
        start, end = _span_key(span)
        if end - start == 1 or mode == "drop":
            block = original_layers[start]
        else:
            block = swm_merge_blocks(original_layers[start:end])
            source = _checkpoint_for_span(span, merged_index, merged_state_paths)
            if source is not None:
                block.load_state_dict(_load_state(source), strict=True)
            merged_index += 1
        new_layers.append(block)

    if mode == "merged" and merged_state_paths is not None and not isinstance(
        merged_state_paths, Mapping
    ):
        if len(merged_state_paths) != merged_index:
            raise ValueError(
                f"Expected {merged_index} merged states, received {len(merged_state_paths)}"
            )

    set_layers(student, new_layers, path)
    return student


def make_naive_drop_layout(layout: Layout) -> Layout:
    """Clone ``layout`` for the equal-span, equal-removal drop control.

    Layout records topology, not how each resulting block is initialized.  The
    returned layout therefore deliberately has the same spans as the merge;
    pass it to :func:`build_student` with ``mode="drop"`` to keep each span's
    first original block.  A fresh object makes accidental metadata mutation
    between baseline records less likely.
    """

    result = Layout(
        num_layers=int(layout.num_layers),
        spans=tuple(Span(int(span.start), int(span.end)) for span in layout.spans),
        protected_layers=frozenset(int(index) for index in layout.protected_layers),
    )
    layout.assert_equal_removal(result)
    return result


@dataclass(frozen=True)
class XMergePair:
    """One fixed-pair XMerge choice before local reconstruction."""

    span: Span
    removed_index: int
    survivor_index: int
    selection_score: float


def _z_scores(values: Sequence[float]) -> list[float]:
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("XMerge RM/BI scores must be finite")
    if len(numeric) < 2:
        return [0.0 for _ in numeric]
    mean = statistics.fmean(numeric)
    variance = statistics.fmean((value - mean) ** 2 for value in numeric)
    if variance == 0:
        return [0.0 for _ in numeric]
    scale = math.sqrt(variance)
    return [(value - mean) / scale for value in numeric]


def select_xmerge_pairs(
    num_layers: int,
    removed_count: int,
    protected_prefix: int = 0,
    protected_suffix: int = 0,
    *,
    rm_scores: Sequence[float] | None = None,
    bi_scores: Sequence[float] | None = None,
    pair_direction: str = "right",
) -> tuple[XMergePair, ...]:
    """Select XMerge removal candidates and adjacent surviving partners.

    With RM and BI supplied, selection follows XMerge's cross-axis rule:
    standardize each score over layers, compute
    ``s_l = max(z_RM_l, z_BI_l)``, and consider removal candidates in ascending
    ``s_l`` order (smallest first).  A candidate is accepted only if it can be
    paired with an unused, unprotected adjacent survivor.

    ``pair_direction`` may be ``"right"`` or ``"left"`` (with boundary
    fallback), or ``"auto"`` to keep the adjacent layer with larger combined
    score.  This direction is topology metadata only; the XMerge local
    reconstruction trainer decides the actual initialization/training details.
    If scores are omitted, layer order supplies a deterministic structural
    smoke-test layout; experiment runs should always pass measured RM and BI.
    """

    values = (num_layers, removed_count, protected_prefix, protected_suffix)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("Layer counts and protected widths must be integers")
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    if removed_count < 0 or protected_prefix < 0 or protected_suffix < 0:
        raise ValueError("removed_count and protected widths must be non-negative")
    if protected_prefix + protected_suffix > num_layers:
        raise ValueError("Protected prefix and suffix overlap")
    if pair_direction not in {"left", "right", "auto"}:
        raise ValueError("pair_direction must be 'left', 'right', or 'auto'")
    if (rm_scores is None) != (bi_scores is None):
        raise ValueError("rm_scores and bi_scores must be supplied together")
    if rm_scores is not None:
        assert bi_scores is not None
        if len(rm_scores) != num_layers or len(bi_scores) != num_layers:
            raise ValueError("RM and BI must each contain one value per original layer")

    first = protected_prefix
    stop = num_layers - protected_suffix
    capacity = (stop - first) // 2
    if removed_count > capacity:
        raise ValueError(
            f"Cannot place {removed_count} disjoint pairs in {stop - first} "
            f"unprotected layers (capacity {capacity})"
        )
    if removed_count == 0:
        return ()

    if rm_scores is None:
        combined = [float(index) for index in range(num_layers)]
    else:
        rm_z = _z_scores(rm_scores)
        assert bi_scores is not None
        bi_z = _z_scores(bi_scores)
        combined = [max(rm, bi) for rm, bi in zip(rm_z, bi_z, strict=True)]
    candidate_order = sorted(range(first, stop), key=lambda index: (combined[index], index))

    used: set[int] = set()
    selected: list[XMergePair] = []
    for candidate in candidate_order:
        if candidate in used:
            continue
        available = [
            neighbor
            for neighbor in (candidate - 1, candidate + 1)
            if first <= neighbor < stop and neighbor not in used
        ]
        if not available:
            continue
        if pair_direction == "right":
            neighbor = candidate + 1 if candidate + 1 in available else available[0]
        elif pair_direction == "left":
            neighbor = candidate - 1 if candidate - 1 in available else available[-1]
        else:
            # Retain the higher-scoring neighbor; ties prefer the right side.
            neighbor = max(available, key=lambda index: (combined[index], index))
        start = min(candidate, neighbor)
        selected.append(
            XMergePair(
                span=Span(start, start + 2),
                removed_index=candidate,
                survivor_index=neighbor,
                selection_score=combined[candidate],
            )
        )
        used.update((candidate, neighbor))
        if len(selected) == removed_count:
            break
    if len(selected) != removed_count:
        raise ValueError(
            f"Could select only {len(selected)} non-overlapping adjacent XMerge pairs; "
            f"requested {removed_count}"
        )
    # Preserve ranking/acceptance order here; layout construction sorts by
    # physical layer position independently.
    return tuple(selected)


def make_xmerge_layout(
    num_layers: int,
    removed_count: int,
    protected_prefix: int = 0,
    protected_suffix: int = 0,
    *,
    rm_scores: Sequence[float] | None = None,
    bi_scores: Sequence[float] | None = None,
    pair_direction: str = "right",
) -> Layout:
    """Create XMerge's disjoint fixed-pair local-reconstruction layout.

    Published XMerge selection does not apply this project's protected-boundary
    mask, hence the zero defaults.  Set ``protected_prefix=2`` and
    ``protected_suffix=1`` for a boundary-matched control.  Every selected pair
    becomes one block and therefore removes exactly one layer.
    """

    pairs = select_xmerge_pairs(
        num_layers,
        removed_count,
        protected_prefix,
        protected_suffix,
        rm_scores=rm_scores,
        bi_scores=bi_scores,
        pair_direction=pair_direction,
    )
    starts = {pair.span.start for pair in pairs}
    spans: list[Span] = []
    cursor = 0
    while cursor < num_layers:
        if cursor in starts:
            spans.append(Span(cursor, cursor + 2))
            cursor += 2
        else:
            spans.append(Span(cursor, cursor + 1))
            cursor += 1
    first = protected_prefix
    stop = num_layers - protected_suffix
    protected = frozenset(range(first)) | frozenset(range(stop, num_layers))
    layout = Layout(num_layers=num_layers, spans=tuple(spans), protected_layers=protected)
    layout.assert_equal_removal(removed_count)
    return layout


# Descriptive aliases used in manifests and notebooks.
swm_weight_arithmetic_init = swm_merge_blocks
make_drop_control_layout = make_naive_drop_layout


__all__ = [
    "average_blocks",
    "build_student",
    "get_layers",
    "make_drop_control_layout",
    "make_naive_drop_layout",
    "make_xmerge_layout",
    "reindex_layers",
    "select_xmerge_pairs",
    "set_layers",
    "swm_merge_blocks",
    "swm_weight_arithmetic_init",
    "XMergePair",
]
