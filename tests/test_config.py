from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_compression.config import ConfigError, ExperimentConfig, load_config


class ConfigTests(unittest.TestCase):
    def test_load_overrides_aliases_and_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.yaml"
            path.write_text(
                """
run_name: qwen-pilot
seed: 7
model:
  name: Qwen/Qwen3-0.6B
  revision: abc123
  custom_model_knob: retained
data:
  dataset: example/data
  subset: sample
  seq_length: 256
stage1:
  threshold: 0.2
  lr: 2.0e-5
stages:
  stage2:
    lambda_start: 0.5
    lambda_end: 0.5
    lambda_schedule: fixed
paths:
  output_dir: scratch/results
evaluation:
  tasks: [boolq, piqa]
future_section:
  new_knob: true
""".lstrip(),
                encoding="utf-8",
            )
            config = load_config(
                path,
                overrides=["stages.stage1.probe_steps=12", "model.dtype=float16"],
            )

        self.assertEqual(config.run_name, "qwen-pilot")
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.model.name_or_path, "Qwen/Qwen3-0.6B")
        self.assertEqual(config.model.dtype, "float16")
        self.assertEqual(config.data.sequence_length, 256)
        self.assertEqual(config.stage1.probe_steps, 12)
        self.assertAlmostEqual(config.stage1.learning_rate, 2e-5)
        self.assertEqual(config.stage2.lambda_schedule, "fixed")
        self.assertEqual(config.run_dir, Path("scratch/results/qwen-pilot"))
        serialized = config.to_dict()
        self.assertEqual(serialized["model"]["custom_model_knob"], "retained")
        self.assertTrue(serialized["future_section"]["new_knob"])
        self.assertEqual(config.raw["stages"]["stage1"]["probe_steps"], 12)

    def test_default_name_comes_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "narval.yaml"
            path.write_text("{}\n", encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.run_name, "narval")

    def test_direct_mapping_and_top_level_output_alias(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "experiment": {"name": "direct", "seed": 11, "tag": "pilot"},
                "output_dir": "out",
                "stage3": {"steps": 10, "warmup_steps": 2},
            }
        )
        self.assertEqual(config.run_name, "direct")
        self.assertEqual(config.seed, 11)
        self.assertEqual(config.output_dir, Path("out"))
        self.assertEqual(config.stage3.steps, 10)
        self.assertEqual(config.to_dict()["experiment"]["tag"], "pilot")

    def test_rejects_bad_known_values_and_override_paths(self) -> None:
        self.assertEqual(
            ExperimentConfig.from_dict({"stage1": {"probe_steps": 0}}).stage1.probe_steps,
            0,
        )
        with self.assertRaisesRegex(ConfigError, "max_group_size"):
            ExperimentConfig.from_dict({"stage1": {"max_group_size": 1}})
        with self.assertRaisesRegex(ConfigError, "warmup_steps"):
            ExperimentConfig.from_dict(
                {"stage3": {"steps": 10, "warmup_steps": 11}}
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("model: scalar\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "model must be a mapping"):
                load_config(path)
            path.write_text("model: {dtype: float32}\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "KEY=VALUE"):
                load_config(path, overrides=["model.dtype"])


if __name__ == "__main__":
    unittest.main()
