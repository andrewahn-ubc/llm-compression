"""Serializable source-to-student layer layouts.

Layer indices are zero-based. A :class:`Span` is half-open (``[start, end)``) and
corresponds to exactly one student block. Consequently, a span of length ``k`` removes
``k - 1`` layers. A valid layout is a gap-free partition of all teacher layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


class LayoutError(ValueError):
    """Raised when a span or layout violates a structural invariant."""


def _index(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LayoutError(f"{name} must be an integer")
    if value < minimum:
        raise LayoutError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, order=True)
class Span:
    """Contiguous teacher layers represented by one student block."""

    start: int
    end: int

    def __post_init__(self) -> None:
        _index(self.start, "span.start")
        _index(self.end, "span.end", minimum=1)
        if self.end <= self.start:
            raise LayoutError("span.end must be greater than span.start")

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def removed(self) -> int:
        return self.length - 1

    @property
    def is_merged(self) -> bool:
        return self.length > 1

    def __contains__(self, layer: object) -> bool:
        return isinstance(layer, int) and not isinstance(layer, bool) and self.start <= layer < self.end

    def to_manifest(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_manifest(cls, raw: Mapping[str, Any] | Sequence[int]) -> "Span":
        if isinstance(raw, Mapping):
            if "start" not in raw or "end" not in raw:
                raise LayoutError("span manifest requires start and end")
            return cls(_index(raw["start"], "span.start"), _index(raw["end"], "span.end", minimum=1))
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 2:
            return cls(_index(raw[0], "span.start"), _index(raw[1], "span.end", minimum=1))
        raise LayoutError("span manifest must be a {start, end} mapping or a two-item sequence")


@dataclass(frozen=True)
class Layout:
    """A complete partition of a teacher model's layers into student blocks."""

    num_layers: int
    spans: tuple[Span, ...]
    protected_layers: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        num_layers = _index(self.num_layers, "num_layers", minimum=1)
        try:
            spans = tuple(
                span if isinstance(span, Span) else Span.from_manifest(span)  # type: ignore[arg-type]
                for span in self.spans
            )
        except TypeError as exc:
            raise LayoutError("spans must be an iterable of Span values") from exc
        if not spans:
            raise LayoutError("a layout must contain at least one span")
        try:
            protected = frozenset(
                _index(layer, "protected layer") for layer in self.protected_layers
            )
        except TypeError as exc:
            raise LayoutError("protected_layers must be an iterable of indices") from exc

        cursor = 0
        for position, span in enumerate(spans):
            if span.start != cursor:
                relation = "gap" if span.start > cursor else "overlap or out-of-order span"
                raise LayoutError(
                    f"layout has a {relation} before span {position}: expected start {cursor}, "
                    f"got {span.start}"
                )
            if span.end > num_layers:
                raise LayoutError(
                    f"span {position} ends at {span.end}, beyond num_layers={num_layers}"
                )
            cursor = span.end
        if cursor != num_layers:
            raise LayoutError(
                f"layout does not cover all layers: final end is {cursor}, expected {num_layers}"
            )

        for layer in protected:
            if layer >= num_layers:
                raise LayoutError(
                    f"protected layer {layer} is outside [0, {num_layers})"
                )
            owner = next(span for span in spans if layer in span)
            if owner.length != 1:
                raise LayoutError(
                    f"protected layer {layer} must remain a singleton, but is in "
                    f"[{owner.start}, {owner.end})"
                )

        object.__setattr__(self, "num_layers", num_layers)
        object.__setattr__(self, "spans", spans)
        object.__setattr__(self, "protected_layers", protected)

    @property
    def merged_spans(self) -> tuple[Span, ...]:
        return tuple(span for span in self.spans if span.is_merged)

    @property
    def removed_layers(self) -> int:
        return sum(span.removed for span in self.spans)

    @property
    def compressed_num_layers(self) -> int:
        return len(self.spans)

    @property
    def compression_fraction(self) -> float:
        return self.removed_layers / self.num_layers

    def span_for_layer(self, layer: int) -> Span:
        layer = _index(layer, "layer")
        if layer >= self.num_layers:
            raise LayoutError(f"layer {layer} is outside [0, {self.num_layers})")
        return next(span for span in self.spans if layer in span)

    def assert_equal_removal(self, other: "Layout | int") -> None:
        """Raise if ``other`` does not remove exactly the same number of layers."""

        expected = other.removed_layers if isinstance(other, Layout) else _index(other, "removed_layers")
        if self.removed_layers != expected:
            raise LayoutError(
                f"unequal compression: layout removes {self.removed_layers} layers, "
                f"expected {expected}"
            )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "num_layers": self.num_layers,
            "spans": [span.to_manifest() for span in self.spans],
            "protected_layers": sorted(self.protected_layers),
            "compressed_num_layers": self.compressed_num_layers,
            "removed_layers": self.removed_layers,
        }

    @classmethod
    def from_manifest(cls, raw: Mapping[str, Any]) -> "Layout":
        if not isinstance(raw, Mapping):
            raise LayoutError("layout manifest must be a mapping")
        version = raw.get("format_version", 1)
        if version != 1:
            raise LayoutError(f"unsupported layout manifest format_version {version!r}")
        if "num_layers" not in raw or "spans" not in raw:
            raise LayoutError("layout manifest requires num_layers and spans")
        spans_raw = raw["spans"]
        if not isinstance(spans_raw, Sequence) or isinstance(spans_raw, (str, bytes)):
            raise LayoutError("layout manifest spans must be a sequence")
        protected_raw = raw.get("protected_layers", ())
        if not isinstance(protected_raw, Sequence) or isinstance(protected_raw, (str, bytes)):
            raise LayoutError("layout manifest protected_layers must be a sequence")
        layout = cls(
            num_layers=_index(raw["num_layers"], "num_layers", minimum=1),
            spans=tuple(Span.from_manifest(span) for span in spans_raw),
            protected_layers=frozenset(
                _index(layer, "protected layer") for layer in protected_raw
            ),
        )
        if "compressed_num_layers" in raw:
            declared = _index(raw["compressed_num_layers"], "compressed_num_layers", minimum=1)
            if declared != layout.compressed_num_layers:
                raise LayoutError(
                    "manifest compressed_num_layers does not match its spans: "
                    f"{declared} != {layout.compressed_num_layers}"
                )
        if "removed_layers" in raw:
            declared = _index(raw["removed_layers"], "removed_layers")
            layout.assert_equal_removal(declared)
        return layout

    @classmethod
    def identity(
        cls,
        num_layers: int,
        *,
        protected_layers: Iterable[int] = (),
    ) -> "Layout":
        num_layers = _index(num_layers, "num_layers", minimum=1)
        return cls(
            num_layers,
            tuple(Span(index, index + 1) for index in range(num_layers)),
            frozenset(protected_layers),
        )

    @classmethod
    def from_spans(
        cls,
        num_layers: int,
        spans: Iterable[Span | Sequence[int] | Mapping[str, Any]],
        *,
        protected_prefix_layers: int = 0,
        protected_suffix_layers: int = 0,
        protected_layers: Iterable[int] = (),
    ) -> "Layout":
        """Build a layout and derive protected prefix/suffix layer indices."""

        num_layers = _index(num_layers, "num_layers", minimum=1)
        prefix = _index(protected_prefix_layers, "protected_prefix_layers")
        suffix = _index(protected_suffix_layers, "protected_suffix_layers")
        if prefix + suffix > num_layers:
            raise LayoutError("protected prefix and suffix exceed num_layers")
        protected = set(protected_layers)
        protected.update(range(prefix))
        protected.update(range(num_layers - suffix, num_layers))
        normalized = tuple(
            span if isinstance(span, Span) else Span.from_manifest(span) for span in spans
        )
        return cls(num_layers, normalized, frozenset(protected))


def assert_equal_removal(layouts: Iterable[Layout]) -> int:
    """Validate equal compression across layouts and return the removal count."""

    items = tuple(layouts)
    if not items:
        raise LayoutError("at least one layout is required")
    expected = items[0].removed_layers
    for layout in items[1:]:
        layout.assert_equal_removal(expected)
    return expected


__all__ = ["Layout", "LayoutError", "Span", "assert_equal_removal"]
