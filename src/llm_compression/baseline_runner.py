"""Construction of the proposal's equal-depth baseline checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .baselines import (
    build_student,
    make_xmerge_layout,
    select_xmerge_pairs,
)
from .data import iter_cached_batches, load_boundary_cache
from .layout import Layout, Span
from .modeling import get_layers, state_dict_to_cpu
from .training import load_layout_manifest


def _dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def xmerge_activation_scores(cache_manifest: str | Path, split: str) -> tuple[list[float], list[float]]:
    """Compute XMerge relative-magnitude and block-influence axes."""

    cache = load_boundary_cache(cache_manifest, split)
    rm_num = [0.0] * cache.num_layers
    bi_num = [0.0] * cache.num_layers
    counts = [0] * cache.num_layers
    for layer in range(cache.num_layers):
        for batch in iter_cached_batches(
            cache, boundaries=(layer, layer + 1), batch_size=8, device="cpu"
        ):
            before = batch[f"h_{layer:03d}"].float()
            after = batch[f"h_{layer + 1:03d}"].float()
            delta_norm = (after - before).norm(dim=-1)
            before_norm = before.norm(dim=-1).clamp_min(1e-12)
            rm_num[layer] += float((delta_norm / before_norm).sum())
            bi_num[layer] += float((1.0 - F.cosine_similarity(before, after, dim=-1)).sum())
            counts[layer] += before.shape[0] * before.shape[1]
    rm = [rm_num[index] / max(counts[index], 1) for index in range(cache.num_layers)]
    bi = [bi_num[index] / max(counts[index], 1) for index in range(cache.num_layers)]
    return rm, bi


@torch.no_grad()
def _global_hidden_cosine(
    teacher,
    student,
    cache,
    *,
    batch_size: int,
    max_batches: int,
    device: str,
) -> float:
    numerator = denominator_t = denominator_s = 0.0
    final_key = f"h_{cache.num_layers:03d}"
    final_norm = getattr(teacher.model, "norm", None)
    if final_norm is None:
        raise TypeError("SWM selection requires the teacher backbone's final norm")
    norm_dtype = next(final_norm.parameters()).dtype
    for batch_index, batch in enumerate(
        iter_cached_batches(
            cache,
            boundaries=(cache.num_layers,),
            batch_size=batch_size,
            device=device,
        )
    ):
        if batch_index >= max_batches:
            break
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "use_cache": False,
        }
        # The raw final boundary is already in the proposal's teacher cache;
        # applying the teacher's final norm reproduces the backbone output and
        # avoids an otherwise repeated full teacher forward for every SWM
        # candidate window.
        teacher_hidden = final_norm(batch[final_key].to(norm_dtype)).float()
        student_hidden = student.model(**kwargs).last_hidden_state.float()
        numerator += float((teacher_hidden * student_hidden).sum())
        denominator_t += float(teacher_hidden.square().sum())
        denominator_s += float(student_hidden.square().sum())
    return numerator / max((denominator_t * denominator_s) ** 0.5, 1e-12)


def select_swm_layout(
    teacher,
    *,
    removed_count: int,
    cache_manifest: str | Path,
    cache_split: str,
    similarity_threshold: float,
    protected_prefix: int,
    protected_suffix: int,
    batch_size: int,
    max_validation_batches: int,
    device: str,
) -> tuple[Layout, list[dict[str, Any]]]:
    """Back-to-front SWM window scan with its global-hidden cosine test.

    The threshold is normally chosen by a validation grid under the exact
    removal constraint.  If a supplied threshold cannot reach that budget,
    the best failed pair is forced and explicitly logged; this preserves the
    proposal's equal-depth comparison without hiding the threshold mismatch.
    """

    num_layers = len(get_layers(teacher))
    protected = frozenset(range(protected_prefix)) | frozenset(
        range(num_layers - protected_suffix, num_layers)
    )
    spans = [Span(index, index + 1) for index in range(num_layers)]
    cache = load_boundary_cache(cache_manifest, cache_split)
    decisions: list[dict[str, Any]] = []

    while sum(span.removed for span in spans) < removed_count:
        best: tuple[float, int, Layout] | None = None
        # Published SWM scans from the deepest eligible window downward.
        for right in range(len(spans) - protected_suffix - 1, protected_prefix - 1, -1):
            if right <= 0:
                continue
            left = right - 1
            if spans[left].start in protected or spans[right].end - 1 in protected:
                continue
            candidate_spans = list(spans)
            merged = Span(candidate_spans[left].start, candidate_spans[right].end)
            candidate_spans[left : right + 1] = [merged]
            candidate = Layout(num_layers, tuple(candidate_spans), protected)
            if candidate.removed_layers > removed_count:
                continue
            candidate_model = build_student(teacher, candidate, mode="merged").to(device)
            candidate_model.eval()
            similarity = _global_hidden_cosine(
                teacher,
                candidate_model,
                cache,
                batch_size=batch_size,
                max_batches=max_validation_batches,
                device=device,
            )
            del candidate_model
            if best is None or similarity > best[0]:
                best = (similarity, left, candidate)
            if similarity >= similarity_threshold:
                # First passing window in a deepest-to-shallowest scan.
                best = (similarity, left, candidate)
                break
        if best is None:
            raise RuntimeError("No eligible SWM merge remains for the requested budget")
        similarity, left, candidate = best
        forced = similarity < similarity_threshold
        decisions.append(
            {
                "span": candidate.spans[left].to_manifest(),
                "global_hidden_cosine": similarity,
                "threshold": similarity_threshold,
                "forced_to_match_budget": forced,
            }
        )
        spans = list(candidate.spans)
    layout = Layout(num_layers, tuple(spans), protected)
    layout.assert_equal_removal(removed_count)
    return layout, decisions


def _compatibility_manifest(
    *,
    layout: Layout,
    method: str,
    checkpoints: Mapping[str, str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = layout.to_manifest()
    result.update(
        {
            "version": 1,
            "method": method,
            "block_checkpoints": dict(checkpoints),
            "compression_fraction": layout.compression_fraction,
        }
    )
    if extra:
        result.update(extra)
    return result


def construct_baseline(
    *,
    teacher,
    method_layout_path: str | Path,
    method: str,
    output_dir: str | Path,
    cache_manifest: str | Path,
    cache_split: str,
    swm_similarity_threshold: float = 0.999,
    validation_batch_size: int = 4,
    validation_batches: int = 2,
    xmerge_pair_direction: str = "right",
    device: str = "cuda",
) -> Path:
    """Build original, naive-drop, SWM, XMerge, or trained-drop-control input."""

    source_manifest = load_layout_manifest(method_layout_path)
    source_layout = Layout.from_manifest(source_manifest)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if method == "original":
        manifest = Layout.identity(source_layout.num_layers).to_manifest()
        manifest.update({"method": "original", "model_path": source_manifest.get("model")})
        path = output / "layout.json"
        _dump(path, manifest)
        return path

    removed_count = source_layout.removed_layers
    checkpoints: dict[str, str] = {}
    extra: dict[str, Any] = {"matched_removed_layers": removed_count}

    if method in {"naive_drop", "drop_control"}:
        layout = source_layout
        student = build_student(teacher, layout, mode="drop")
        if method == "drop_control":
            block_dir = output / "blocks"
            for span in layout.merged_spans:
                block_path = block_dir / f"keep-{span.start:03d}-{span.end:03d}.pt"
                _save(block_path, state_dict_to_cpu(get_layers(teacher)[span.start]))
                checkpoints[f"{span.start:03d}-{span.end:03d}"] = str(block_path.resolve())
    elif method == "swm":
        layout, decisions = select_swm_layout(
            teacher,
            removed_count=removed_count,
            cache_manifest=cache_manifest,
            cache_split=cache_split,
            similarity_threshold=swm_similarity_threshold,
            protected_prefix=2,
            protected_suffix=1,
            batch_size=validation_batch_size,
            max_validation_batches=validation_batches,
            device=device,
        )
        student = build_student(teacher, layout, mode="merged")
        block_dir = output / "blocks"
        for student_index, span in enumerate(layout.spans):
            if not span.is_merged:
                continue
            state = state_dict_to_cpu(get_layers(student)[student_index])
            block_path = block_dir / f"swm-{span.start:03d}-{span.end:03d}.pt"
            _save(block_path, state)
            checkpoints[f"{span.start:03d}-{span.end:03d}"] = str(block_path.resolve())
        extra.update({"selection": "swm_global_hidden_cosine", "decisions": decisions})
    elif method == "xmerge":
        rm, bi = xmerge_activation_scores(cache_manifest, cache_split)
        pairs = select_xmerge_pairs(
            source_layout.num_layers,
            removed_count,
            rm_scores=rm,
            bi_scores=bi,
            pair_direction=xmerge_pair_direction,
        )
        layout = make_xmerge_layout(
            source_layout.num_layers,
            removed_count,
            rm_scores=rm,
            bi_scores=bi,
            pair_direction=xmerge_pair_direction,
        )
        block_dir = output / "blocks"
        merged_states: dict[tuple[int, int], Mapping[str, torch.Tensor]] = {}
        for pair in pairs:
            state = state_dict_to_cpu(get_layers(teacher)[pair.survivor_index])
            block_path = block_dir / f"survivor-{pair.span.start:03d}-{pair.span.end:03d}.pt"
            _save(block_path, state)
            checkpoints[f"{pair.span.start:03d}-{pair.span.end:03d}"] = str(block_path.resolve())
            merged_states[(pair.span.start, pair.span.end)] = state
        student = build_student(teacher, layout, merged_states, mode="merged")
        extra.update(
            {
                "selection": "xmerge_cross_axis",
                "pairs": [
                    {
                        "span": pair.span.to_manifest(),
                        "removed_index": pair.removed_index,
                        "survivor_index": pair.survivor_index,
                        "selection_score": pair.selection_score,
                    }
                    for pair in pairs
                ],
                "rm_scores": rm,
                "bi_scores": bi,
            }
        )
    else:
        raise ValueError("method must be original, naive_drop, drop_control, swm, or xmerge")

    source_layout.assert_equal_removal(layout)
    model_dir = output / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(model_dir, safe_serialization=True)
    manifest = _compatibility_manifest(
        layout=layout,
        method=method,
        checkpoints=checkpoints,
        extra={**extra, "model_path": str(model_dir.resolve())},
    )
    path = output / "layout.json"
    _dump(path, manifest)
    return path
