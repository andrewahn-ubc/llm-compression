from __future__ import annotations

import unittest

from llm_compression.layout import Layout, LayoutError, Span, assert_equal_removal


class SpanTests(unittest.TestCase):
    def test_half_open_properties(self) -> None:
        span = Span(2, 6)
        self.assertEqual(span.length, 4)
        self.assertEqual(span.removed, 3)
        self.assertTrue(span.is_merged)
        self.assertIn(2, span)
        self.assertIn(5, span)
        self.assertNotIn(6, span)

    def test_invalid_span(self) -> None:
        with self.assertRaises(LayoutError):
            Span(2, 2)
        with self.assertRaises(LayoutError):
            Span(True, 2)


class LayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = Layout.from_spans(
            8,
            [Span(0, 1), Span(1, 2), Span(2, 5), Span(5, 7), Span(7, 8)],
            protected_prefix_layers=2,
            protected_suffix_layers=1,
        )

    def test_counts_lookup_and_round_trip(self) -> None:
        self.assertEqual(self.layout.compressed_num_layers, 5)
        self.assertEqual(self.layout.removed_layers, 3)
        self.assertEqual(self.layout.merged_spans, (Span(2, 5), Span(5, 7)))
        self.assertEqual(self.layout.span_for_layer(6), Span(5, 7))
        self.assertAlmostEqual(self.layout.compression_fraction, 3 / 8)
        self.assertEqual(Layout.from_manifest(self.layout.to_manifest()), self.layout)

    def test_partition_and_protection_invariants(self) -> None:
        with self.assertRaisesRegex(LayoutError, "gap"):
            Layout(4, (Span(0, 1), Span(2, 4)))
        with self.assertRaisesRegex(LayoutError, "overlap"):
            Layout(4, (Span(0, 3), Span(2, 4)))
        with self.assertRaisesRegex(LayoutError, "protected layer 0"):
            Layout(4, (Span(0, 2), Span(2, 3), Span(3, 4)), frozenset({0}))
        with self.assertRaisesRegex(LayoutError, "does not cover"):
            Layout(4, (Span(0, 3),))

    def test_equal_removal_invariant(self) -> None:
        equal = Layout.from_spans(
            8,
            [Span(0, 1), Span(1, 2), Span(2, 4), Span(4, 7), Span(7, 8)],
            protected_prefix_layers=2,
            protected_suffix_layers=1,
        )
        self.layout.assert_equal_removal(equal)
        self.assertEqual(assert_equal_removal([self.layout, equal]), 3)
        with self.assertRaisesRegex(LayoutError, "unequal compression"):
            self.layout.assert_equal_removal(Layout.identity(8))

    def test_manifest_rejects_tampered_derived_counts(self) -> None:
        manifest = self.layout.to_manifest()
        manifest["removed_layers"] = 2
        with self.assertRaisesRegex(LayoutError, "unequal compression"):
            Layout.from_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
