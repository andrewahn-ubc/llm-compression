# AGENTS.md

Guidance for AI coding agents (Codex, Claude Code, Cursor, Copilot, Gemini CLI, etc.) working in this repository. Keep agent-specific instruction files (`CLAUDE.md`, `GEMINI.md`, …) out of the repo so this stays the single source of truth.

## Project

Probe-guided depth compression for decoder-only LMs (primary: Qwen3-0.6B, 28 blocks; stretch: Qwen3-1.7B). Contiguous runs of transformer blocks are merged into single blocks, then the student is recovered by local reconstruction and global distillation. The **AI 360 Project Proposal** ([docs/Project Proposal.md](docs/Project%20Proposal.md), a full transcription of the original PDF) is the authoritative spec; "Project idea ai360" supplies only background and the 20%/35% compression targets. Experiments run on Alliance Canada clusters (Narval A100 / Fir H100) through Slurm.

## Commands

```bash
pip install -e '.[dev]'          # add ,eval for lm-eval and ,memory-efficient for bitsandbytes (8-bit Adam)

python -m pytest                                   # full suite
python -m pytest tests/test_layout.py              # one file
python -m pytest tests/test_cli.py -k stage3       # one test by name
ruff check src tests                               # lint (line length 100, py310 target)

# Local verification as documented in README
python -m compileall -q src tests
bash -n scripts/*.sh scripts/slurm/*.sbatch
```

The tests are `unittest.TestCase` classes run under pytest. They skip themselves when an optional dependency (transformers, safetensors) is missing. Model tests build a tiny random `Qwen3Config` (3 layers, hidden size 32), so they need no downloads or GPU. The CLI tests patch `run_stage*` and `run_evaluation` on the `cli` module.

CLI (`llm-compression` or `python -m llm_compression.cli`):

```bash
llm-compression <command> --config configs/main.yaml [--set dotted.key=value ...]
# commands: prepare-data prepare-eval-data cache-teacher stage1 stage2 stage3
#           baseline [--method original|naive_drop|drop_control|swm|xmerge]
#           lora-recovery evaluate plan
```

`plan` only writes `experiment_plan.json` and is safe to run anywhere. On the cluster, always use the launchers with `--dry-run` first. `submit_pipeline.sh` and `submit_baseline_pipeline.sh` submit jobs unless `--dry-run` is passed. `submit_ablation_sweep.sh` is a dry run unless `--submit` is passed. Never submit jobs without the user's explicit go-ahead. The default Slurm account is `def-krallen`.

## Architecture

**Pipeline and data flow.** `prepare-data` → `prepare-eval-data` → `cache-teacher` → `stage1` → `stage2` → `stage3` → `evaluate`. Each step reads the previous step's files from disk:
- `data.py` streams FineWeb-Edu into disjoint token-shard splits (`probe_*`, `local_*`, `global_*`) with manifests. `cache-teacher` stores the teacher's hidden states at every layer boundary for the probe splits.
- **Stage 1** (`training.run_stage1`) scans front to back with greedy grouping. For each candidate span it applies an SWM-initialized merge, trains a short probe (50 steps) on cached boundaries, and accepts the span if the held-out relative error is under the threshold. `cli._stage1` runs a threshold grid (one threshold per Slurm job via `runtime.max_thresholds_this_job`) and picks the threshold that reaches `stage1.target_removed_layers` exactly, or the closest one. The result is written to `stage1/layout.json`.
- **Stage 2** (`run_stage2`) reconstructs each merged group in network order. Only the merged block is trainable, and it is trained on a teacher/student input mixture with λ scheduled 1.0→0.5. Output: `stage2/model` + `progress.json`.
- **Stage 3** (`run_stage3`) does whole-model token-level logit KL distillation. Output: `stage3/model` + `progress.json`.
- **Evaluation** (`evaluate_runner.py`, `evaluation.py`) is split into components selected with `evaluation.component`: perplexity, commonsense harness, MMLU harness, generation, speed, and summary. Harness tasks run through lm-eval. Relative accuracy = 100 × student seven-task macro accuracy / teacher macro accuracy, with MMLU reported separately.

**Layout is the central contract** (`layout.py`). A `Layout` is a gap-free partition of the teacher's layers into half-open `Span`s; each span becomes one student block, and a span of length k removes k−1 layers. Protected layers (the first 2 and last 1) must stay singletons. Baselines (`baselines.py` and `baseline_runner.py`: naive drop, SWM, XMerge, drop control) consume the main run's realized Stage 1 layout and must remove the same number of blocks (`assert_equal_removal`). `baselines.py` has its own model-agnostic layer helpers (`get_layers`, `reindex_layers`) that discover the layer path. `modeling.py` has Qwen3-specific versions of the same helpers.

**Config system** (`config.py`). Each runnable YAML (`configs/main.yaml`, `baselines.yaml`, `ablations.yaml`) is self-contained. Files under `configs/model|data|runtime|cluster/` are reference snippets for humans, not includes. `--set` values are parsed as YAML and applied before validation. Typed dataclasses validate the known keys and keep unknown keys in `.extra`. The `runtime` section in particular acts as a free-form channel through which the shell launchers plumb per-job paths and limits (`layout_path`, `stage2_output_dir`, `stage3_output_dir`, `student_path`, `model_path=__stage2__`, `max_groups_this_job`, `max_steps_this_job`, `chunk_index`, `require_complete_progress`, …), which `cli.py` reads with `.get()`. When adding a launcher option, wire it through as a `runtime.*` override instead of a new CLI flag.

**Paths and caches.** Run artifacts go in `{paths.output_dir}/{run_name}/`. Token and teacher-boundary caches are keyed by model+revision (`cli._model_cache_key`), so runs can share them. Every model, depth target, seed, and ablation cell needs a **unique `run_name`**. Stage 1 raises an error if an existing `layout.json` was produced with different settings.

**Slurm and resumability.** Each job is a 1-hour, single-GPU allocation. `scripts/slurm/run_{narval,fir}.sbatch` call `scripts/run_stage.sh`, which loads modules (`StdEnv/2023`, gcc, `arrow/23.0.1`, `python/3.11`), forces Hugging Face offline mode, injects the scratch `paths.*` overrides, and forwards Slurm's `USR1` (sent 300 s before the limit) to Python. A 240 s watchdog kills the process if it hasn't exited by then. In `training.py` and `lora.py`, a `StopRequested` handler catches `USR1` so that Stages 2 and 3 and the LoRA recovery checkpoint and resume (`runtime.resume=auto`). A Stage 1 threshold scan cannot resume mid-scan and must rerun from the start. Downstream stages check `progress.json["complete"]` and refuse to run on incomplete input. Long stages are split into chunked jobs chained with `afterok`. With default chunk counts the main pipeline dry run is 51 jobs. Write artifacts with the atomic helpers in `artifacts.py`.

**Environment constraints.** Compute nodes have no network. All models and datasets, including lm-eval task data, must be prefetched on a login node with `scripts/prefetch.sh`, separately on each cluster because Narval and Fir don't share filesystems. PyArrow comes from the Alliance `arrow` module, not pip. `transformers` is pinned to `>=4.57,<4.58` because the code reaches into Qwen3 internals (`model.model.layers`, decoder-layer forward signatures). Model configs keep `revision: null` for now; pin an immutable HF commit before producing final results.

See `README.md` for launcher examples (10-block target, 1.7B model, baseline and ablation suites) and `docs/HPC.md` for cluster, storage, and monitoring details.
