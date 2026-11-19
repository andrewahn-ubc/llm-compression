from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

try:
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover - exercised by minimal developer environments
    save_file = None

from llm_compression.lora import (
    LoRALinear,
    _load_causal_model,
    _token_data_api,
    inject_qwen_lora,
    iter_lora_modules,
    load_lora_state_dict,
    lora_state_dict,
    merge_qwen_lora,
    run_lora_recovery,
)


class TinyProjectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.q_proj = nn.Linear(4, 3, bias=False)
        self.other = nn.Linear(3, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.other(torch.tanh(self.q_proj(inputs)))


class TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", use_cache=True)
        self.embed = nn.Embedding(11, 6)
        self.q_proj = nn.Linear(6, 6, bias=False)
        self.lm_head = nn.Linear(6, 11, bias=False)
        self.is_gradient_checkpointing = False
        self.input_grads_enabled = False

    def gradient_checkpointing_enable(self, **_kwargs) -> None:
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.is_gradient_checkpointing = False

    def enable_input_require_grads(self) -> None:
        self.input_grads_enabled = True

    def disable_input_require_grads(self) -> None:
        self.input_grads_enabled = False

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        hidden = torch.tanh(self.q_proj(self.embed(input_ids)))
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)

    def save_pretrained(self, path: str | Path, **_kwargs) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination / "model.pt")
        (destination / "saved_state.json").write_text(
            json.dumps(
                {
                    "use_cache": self.config.use_cache,
                    "gradient_checkpointing": self.is_gradient_checkpointing,
                    "input_grads_enabled": self.input_grads_enabled,
                    "training": self.training,
                }
            ),
            encoding="utf-8",
        )


class LoRALinearTests(unittest.TestCase):
    def test_zero_initialization_preserves_output_and_adapters_are_fp32(self) -> None:
        torch.manual_seed(0)
        base = nn.Linear(4, 3, dtype=torch.float64)
        inputs = torch.randn(2, 4, dtype=torch.float64)
        expected = base(inputs)
        layer = LoRALinear(base, rank=2, alpha=4.0)

        torch.testing.assert_close(layer(inputs), expected)
        self.assertEqual(layer.lora_A.dtype, torch.float32)
        self.assertEqual(layer.lora_B.dtype, torch.float32)
        self.assertFalse(layer.base.weight.requires_grad)
        self.assertTrue(layer.lora_A.requires_grad)
        self.assertTrue(layer.lora_B.requires_grad)

    def test_inject_state_roundtrip_and_merge_preserve_adapted_output(self) -> None:
        torch.manual_seed(1)
        original = TinyProjectionModel()
        source = copy.deepcopy(original)
        target = copy.deepcopy(original)
        names = inject_qwen_lora(
            source,
            rank=2,
            alpha=2.0,
            target_modules=("q_proj",),
        )
        inject_qwen_lora(target, rank=2, alpha=2.0, target_modules=("q_proj",))
        self.assertEqual(names, ("q_proj",))
        module = dict(iter_lora_modules(source))["q_proj"]
        with torch.no_grad():
            module.lora_A.fill_(0.25)
            module.lora_B.fill_(0.5)
        load_lora_state_dict(target, lora_state_dict(source))

        inputs = torch.randn(3, 4)
        source.eval()
        target.eval()
        adapted = source(inputs)
        torch.testing.assert_close(target(inputs), adapted)
        self.assertFalse(torch.allclose(original(inputs), adapted))

        merged = merge_qwen_lora(source)
        self.assertEqual(merged, ("q_proj",))
        self.assertFalse(any(isinstance(item, LoRALinear) for item in source.modules()))
        torch.testing.assert_close(source(inputs), adapted, rtol=1e-5, atol=1e-6)

    def test_optional_dependency_errors_are_actionable(self) -> None:
        missing_safetensors = ModuleNotFoundError(
            "No module named 'safetensors'", name="safetensors"
        )
        with patch(
            "llm_compression.lora.importlib.import_module",
            side_effect=missing_safetensors,
        ), self.assertRaisesRegex(RuntimeError, "requires safetensors"):
            _token_data_api()

        missing_transformers = ModuleNotFoundError(
            "No module named 'transformers'", name="transformers"
        )
        with patch(
            "llm_compression.lora.load_causal_lm",
            side_effect=missing_transformers,
        ), self.assertRaisesRegex(RuntimeError, "requires transformers"):
            _load_causal_model("tiny")


@unittest.skipIf(save_file is None, "safetensors is not installed")
class LoRARecoveryTests(unittest.TestCase):
    @staticmethod
    def _token_manifest(root: Path) -> Path:
        split_dir = root / "global_train"
        split_dir.mkdir(parents=True)
        ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
            ],
            dtype=torch.long,
        )
        shard = split_dir / "shard-00000.safetensors"
        save_file({"input_ids": ids, "attention_mask": torch.ones_like(ids)}, shard)
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "splits": {
                        "global_train": {
                            "num_sequences": len(ids),
                            "shards": [str(shard.relative_to(root))],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_chunked_resume_matches_uninterrupted_and_saves_merged_model(self) -> None:
        torch.manual_seed(12)
        template = TinyCausalLM()
        loaded: list[tuple[TinyCausalLM, dict]] = []

        def loader(*_args, **kwargs):
            model = copy.deepcopy(template)
            loaded.append((model, kwargs))
            return model

        with tempfile.TemporaryDirectory() as temporary, patch(
            "llm_compression.lora.load_causal_lm", side_effect=loader
        ):
            root = Path(temporary)
            manifest = self._token_manifest(root / "tokens")
            common = {
                "student_name_or_path": "tiny",
                "token_manifest": manifest,
                "train_split": "global_train",
                "dtype": "float32",
                "device": "cpu",
                "local_files_only": True,
                "attn_implementation": "sdpa",
                "total_steps": 2,
                "micro_batch_size": 2,
                "gradient_accumulation": 2,
                "learning_rate": 1e-2,
                "warmup_steps": 0,
                "rank": 2,
                "alpha": 4.0,
                "dropout": 0.2,
                "target_modules": ("q_proj",),
                "gradient_checkpointing": True,
                "seed": 7,
            }

            chunked = root / "chunked"
            first_progress = run_lora_recovery(
                **common,
                output_dir=chunked,
                max_steps_this_job=1,
            )
            first = json.loads(first_progress.read_text(encoding="utf-8"))
            self.assertEqual(first["global_step"], 1)
            self.assertFalse(first["complete"])

            second_progress = run_lora_recovery(
                **common,
                output_dir=chunked,
                max_steps_this_job=1,
            )
            second = json.loads(second_progress.read_text(encoding="utf-8"))
            self.assertEqual(second["global_step"], 2)
            self.assertTrue(second["complete"])
            self.assertTrue((chunked / "model" / "model.pt").is_file())
            saved_state = json.loads(
                (chunked / "model" / "saved_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                saved_state,
                {
                    "use_cache": True,
                    "gradient_checkpointing": False,
                    "input_grads_enabled": False,
                    "training": False,
                },
            )

            uninterrupted = root / "uninterrupted"
            run_lora_recovery(
                **common,
                output_dir=uninterrupted,
                max_steps_this_job=0,
            )
            chunked_state = torch.load(
                chunked / "adapter.pt", map_location="cpu", weights_only=False
            )["state_dict"]
            full_state = torch.load(
                uninterrupted / "adapter.pt", map_location="cpu", weights_only=False
            )["state_dict"]
            self.assertEqual(set(chunked_state), set(full_state))
            for key in chunked_state:
                torch.testing.assert_close(chunked_state[key], full_state[key])
            self.assertTrue(loaded)
            self.assertTrue(
                all(kwargs["attn_implementation"] == "sdpa" for _, kwargs in loaded)
            )


if __name__ == "__main__":
    unittest.main()
