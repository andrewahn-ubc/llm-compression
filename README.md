# Probe-guided depth compression experiments

This repository implements the experiment plan in **AI 360 Project Proposal** as the
authoritative specification. **Project idea ai360** is used only for background and the
suggested 20%/35% compression operating points. The primary run is Qwen3-0.6B; Qwen3-1.7B
is an optional stretch run.

The implementation covers the full experimental chain:

| Experiment component | Implementation |
| --- | --- |
| Data | Disjoint FineWeb-Edu `sample-10BT` slices, 512-token sequences, 32,047,104 tokens total |
| Stage 1 | Front-to-back greedy groups, first 2/last 1 protected, size 2–4, SWM initialization, 50-step mixed-input probes, held-out threshold search |
| Stage 2 | Network-order local reconstruction, only the merged block trainable, scheduled 1.0→0.5 mixture by default, held-out early stopping |
| Stage 3 | Token-level logit KL at temperature 1, 3,000 steps, gradient checkpointing, 8-bit Adam, held-out KL early stopping |
| Baselines | Teacher, same-span naive drop, SWM, XMerge reconstruction, and trained drop control; matched recovery variants are available |
| Ablations | Trained/untrained/SWM selection, 50/300 probe steps, fixed/scheduled/teacher-only mixture, and evaluation after Stage 2/Stage 3 |
| Evaluation | WikiText-2 perplexity, seven zero-shot tasks, 5-shot MMLU, generations, parameters, latency/throughput, and peak GPU memory |

## Quick start on Narval or Fir

Narval and Fir have separate filesystems, so run setup and prefetch on whichever cluster you
choose. On its login node:

```bash
cd /path/to/llm-compression

export LLMC_VENV=/path/to/llmc-venv
./scripts/bootstrap_env.sh --venv "$LLMC_VENV"

export LLMC_CACHE_ROOT="$SCRATCH/llm-compression/cache"
./scripts/prefetch.sh \
  --config configs/main.yaml \
  --cache-root "$LLMC_CACHE_ROOT"
```

Always inspect a dry run first. The default account is `def-krallen`, which is the usual
Slurm name for the requested `krallen` allocation; confirm the exact Group Name in CCDB and
override `--account` if it differs.

```bash
# Narval / A100
./scripts/submit_pipeline.sh \
  --cluster narval \
  --account def-krallen \
  --config configs/main.yaml \
  --dry-run

# Fir / H100: only the cluster selection changes
./scripts/submit_pipeline.sh \
  --cluster fir \
  --account def-krallen \
  --config configs/main.yaml \
  --dry-run
```

Remove `--dry-run` to submit. Every GPU job requests `01:00:00` and receives a pre-timeout
signal five minutes before the limit. Stages 2, 3, and LoRA recovery checkpoint and resume;
Stage 1 is split one threshold value per job.

The main config targets six removed blocks (about 20% of Qwen3-0.6B's 28 blocks). Run the
ten-block context target separately so artifacts cannot collide:

```bash
./scripts/submit_pipeline.sh \
  --cluster narval \
  --account def-krallen \
  --set stage1.target_removed_layers=10 \
  --set run_name=qwen3_0_6b_remove_10 \
  --dry-run
```

Stage 1 records whether the threshold grid reached the requested depth exactly. All downstream
baselines consume that realized layout and enforce the same number of removed blocks.

## Running on a local GPU

The same launchers run on a workstation GPU (for example an RTX 4090) with `--cluster local`.
They build the same job graph as on Slurm, with the same flags, chunking, and `runtime.*`
overrides, and run its jobs one after another in the foreground on this machine. `--account`
is accepted and ignored. Jobs go through `scripts/local/run_local.sh`, the local counterpart
of the `.sbatch` wrappers.

```bash
pip install -e '.[dev,eval,memory-efficient]'    # bootstrap_env.sh is Alliance-only
./scripts/prefetch.sh --config configs/main.yaml # caches go to .local/cache by default

./scripts/submit_pipeline.sh --cluster local --config configs/main.yaml --dry-run
./scripts/submit_pipeline.sh --cluster local --config configs/main.yaml

# A single step, the local equivalent of `sbatch scripts/slurm/run_narval.sbatch ...`
./scripts/local/run_local.sh stage1 configs/main.yaml --set run_name=qwen3_0_6b_probe_merge
```

`submit_baseline_pipeline.sh` and `submit_ablation_sweep.sh` accept `--cluster local` in the
same way. `--after JOBID` is Slurm-only, so run the main pipeline locally before the baselines.

- **Paths.** Without `$SCRATCH`, caches, outputs, and checkpoints live under `.local/`.
  Override them with `LLMC_WORK_ROOT`, `LLMC_CACHE_ROOT`, or `LLMC_OUTPUT_ROOT`. Each job
  logs to `logs/local/llmc-<label>.log`. Choose a GPU with `CUDA_VISIBLE_DEVICES`.
- **Failures and resuming.** The first failed job stops the launcher. Every job that
  succeeds is recorded in `<output root>/.local_jobs/`, keyed by its exact command line and
  the config file's contents, so re-running the same launcher command skips finished jobs.
  Set `LLMC_LOCAL_RERUN=1` to run everything again.
- **Ctrl-C.** The first Ctrl-C asks the running job to checkpoint, the same path as Slurm's
  pre-timeout `USR1`. Stages 2 and 3 and LoRA recovery resume from that checkpoint when you
  re-run the command. A second Ctrl-C kills the job immediately.
- **Memory.** A 24 GB card has less memory than an A100 (40 GB) or H100 (80 GB). If a stage
  runs out of memory, lower its batch size with `--set` (for example
  `--set stage2.batch_size=8`). Note that this changes the experiment compared with a
  cluster run.

## Baselines and ablations

Run baselines only after the main run has produced `stage1/layout.json`. The full baseline
matrix includes raw methods, published-style XMerge local reconstruction, adapted SWM LoRA,
matched Stage 2+3 recovery, and teacher-relative evaluation:

```bash
./scripts/submit_baseline_pipeline.sh \
  --cluster narval \
  --account def-krallen \
  --main-run qwen3_0_6b_probe_merge \
  --suite-name qwen3_0_6b_baselines_r1 \
  --recovery all \
  --evaluate all \
  --dry-run
```

The SWM LoRA branch uses causal-LM recovery on the proposal's disjoint FineWeb-Edu training
slice. This is an explicit dataset-matched adaptation: the original SWM recovery corpus is
outside the proposal's data plan. The launcher also runs the same Stage 2+3 recovery on SWM,
XMerge, naive drop, and the drop control so selection and recovery effects can be separated.

The ablation sweep is a dry run unless `--submit` is supplied:

```bash
./scripts/submit_ablation_sweep.sh \
  --cluster narval \
  --account def-krallen \
  --prefix qwen3_0_6b_ablation_r1

# After reviewing every generated command:
./scripts/submit_ablation_sweep.sh \
  --cluster narval \
  --account def-krallen \
  --prefix qwen3_0_6b_ablation_r1 \
  --submit
```

Every end-to-end pipeline evaluates the exact retained Stage 2 checkpoint into
`evaluation/after_stage2`, then evaluates the recovered Stage 3 checkpoint into
`evaluation/student`; the comparison does not retrain Stage 2. The SWM global-similarity
selector comparison is produced by the baseline launcher with
`--recovery same`. The 300-step probe is intentionally included, but unlike the other training
stages it cannot resume within one threshold scan; if it exceeds 1:00 in practice, lower the
calibration slice for that feasibility ablation or give it a separately approved longer limit.

## Outputs and reproducibility

Compute jobs run offline and default to these scratch paths:

```text
$SCRATCH/llm-compression/cache       model, datasets, token shards, teacher boundaries
$SCRATCH/llm-compression/outputs     layouts, checkpoints, progress, metrics
$SCRATCH/llm-compression/checkpoints signal markers
```

Use a unique `run_name` for every model, depth target, seed, and ablation cell. Teacher-boundary
and token caches are namespaced by model/revision, while training artifacts are isolated by run.
The final accuracy summary reports the seven-task macro separately from MMLU and defines
relative accuracy as:

```text
100 × student seven-task macro accuracy / teacher seven-task macro accuracy
```

Qwen3-1.7B can be selected with:

```bash
./scripts/submit_pipeline.sh \
  --cluster fir \
  --account def-krallen \
  --set model.name_or_path=Qwen/Qwen3-1.7B \
  --set run_name=qwen3_1_7b_remove_6 \
  --dry-run
```

Prefetch it under the same overrides before submission. Start with smaller local/global batch
sizes if the 1.7B memory profile requires it.

For scheduler details, cache placement, monitoring, resume behavior, and official Alliance
references, see [docs/HPC.md](docs/HPC.md).

## Local verification

```bash
python -m compileall -q src tests
python -m pytest
bash -n scripts/*.sh scripts/lib/*.sh scripts/local/*.sh scripts/slurm/*.sbatch
```

No cluster job is submitted by repository setup or by either dry-run command.
