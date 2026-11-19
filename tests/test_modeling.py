import unittest

import torch
try:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from llm_compression.modeling import block_forward, teacher_boundaries

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers is not installed")
class QwenModelingTests(unittest.TestCase):
    def setUp(self):
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
        )
        self.model = Qwen3ForCausalLM(config).eval()

    def test_boundary_cache_includes_raw_final_boundary(self):
        input_ids = torch.randint(0, 64, (2, 12))
        attention_mask = torch.ones_like(input_ids)
        boundaries = teacher_boundaries(self.model, input_ids, attention_mask)
        self.assertEqual(len(boundaries), 4)
        self.assertTrue(all(value.shape == (2, 12, 32) for value in boundaries))

    def test_isolated_block_matches_full_forward_boundary(self):
        input_ids = torch.randint(0, 64, (2, 12))
        attention_mask = torch.ones_like(input_ids)
        boundaries = teacher_boundaries(self.model, input_ids, attention_mask)
        actual = block_forward(
            self.model,
            self.model.model.layers[0],
            boundaries[0],
            attention_mask,
        )
        torch.testing.assert_close(actual, boundaries[1])


if __name__ == "__main__":
    unittest.main()
