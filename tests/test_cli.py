from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import safetensors  # noqa: F401
except ImportError:  # pragma: no cover - minimal developer environment
    raise unittest.SkipTest("safetensors is not installed")

from llm_compression import cli
from llm_compression.config import ExperimentConfig


class CliStageRoutingTests(unittest.TestCase):
    @staticmethod
    def _config(
        output: Path,
        *,
        stage3_enabled: bool = True,
        runtime_extra: dict[str, object] | None = None,
    ) -> ExperimentConfig:
        runtime: dict[str, object] = {"device": "cpu"}
        runtime.update(runtime_extra or {})
        return ExperimentConfig.from_dict(
            {
                "run_name": "routing-test",
                "model": {
                    "name_or_path": "Qwen/Qwen3-0.6B",
                    "local_files_only": True,
                },
                "runtime": runtime,
                "paths": {"output_dir": str(output)},
                "stage3": {"enabled": stage3_enabled},
            }
        )

    def test_enabled_stage3_dispatches_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            expected = Path(temporary) / "stage3" / "progress.json"
            cfg = self._config(Path(temporary))
            with patch.object(cli, "run_stage3", return_value=expected) as mocked:
                result = cli._stage3(cfg)
            self.assertEqual(result, expected)
            mocked.assert_called_once()

    def test_stage2_fallback_refuses_incomplete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg = self._config(Path(temporary), stage3_enabled=False)
            stage2 = cfg.run_dir / "stage2"
            (stage2 / "model").mkdir(parents=True)
            (stage2 / "progress.json").write_text(
                json.dumps({"complete": False}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "Stage 2 is incomplete"):
                cli._evaluate(cfg)

    def test_enabled_stage3_never_silently_falls_back_to_stage2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg = self._config(Path(temporary), stage3_enabled=True)
            stage2 = cfg.run_dir / "stage2"
            (stage2 / "model").mkdir(parents=True)
            (stage2 / "progress.json").write_text(
                json.dumps({"complete": True}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "Stage 3 output is missing"):
                cli._evaluate(cfg)

    def test_explicit_stage2_evaluation_uses_retained_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg = self._config(
                Path(temporary),
                runtime_extra={"model_path": "__stage2__"},
            )
            stage2 = cfg.run_dir / "stage2"
            (stage2 / "model").mkdir(parents=True)
            (stage2 / "progress.json").write_text(
                json.dumps({"complete": True}), encoding="utf-8"
            )
            expected = cfg.run_dir / "evaluation" / "student"
            with patch.object(cli, "run_evaluation", return_value=expected) as mocked:
                result = cli._evaluate(cfg)
            self.assertEqual(result, expected)
            self.assertEqual(
                mocked.call_args.kwargs["model_name_or_path"],
                (stage2 / "model").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
