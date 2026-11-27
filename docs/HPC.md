# Running the experiments on Narval or Fir

The launchers use one full GPU for `01:00:00`, ask Slurm to send `USR1` five minutes
before the limit, and run with network access disabled. The same experiment YAML can
therefore be used on either cluster; only the Slurm wrapper changes.

To run the same experiments on a workstation GPU instead, pass `--cluster local`; see
"Running on a local GPU" in the [README](../README.md#running-on-a-local-gpu).

## Cluster choice

| Cluster | Requested GPU | GPU memory | Launcher | Requested CPU/RAM |
| --- | --- | ---: | --- | ---: |
| Narval | A100 | 40 GB | `scripts/slurm/run_narval.sbatch` | 8 CPU / 96 GB |
| Fir | H100 | 80 GB | `scripts/slurm/run_fir.sbatch` | 8 CPU / 128 GB |

These requests fit within the per-GPU resource bundles documented on the official
[Narval](https://docs.alliancecan.ca/wiki/Narval/en) and
[Fir](https://docs.alliancecan.ca/wiki/Fir) pages. Fir access must be requested in
CCDB before first use. Alliance recommends the typed
`--gpus-per-node=a100:1` / `--gpus-per-node=h100:1` syntax used here; see
[Using GPUs with Slurm](https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm).

Every job has a one-hour wall-clock request. Short requests can also fit
backfill opportunities more readily; see the Alliance
[job scheduling policies](https://docs.alliancecan.ca/wiki/Job_scheduling_policies)
and [job-submission best practices](https://docs.alliancecan.ca/wiki/Best_practices_for_job_submission).

## Account name

The scripts default to `def-krallen`, interpreting the requested `krallen` allocation
as its default-allocation Slurm group. Slurm needs the exact **Group Name** shown for
the allocation in CCDB; that may instead be an `rrg-...` or another cluster-enabled
account. Verify it before a long submission and override it explicitly if needed:

```bash
./scripts/submit_pipeline.sh \
  --cluster narval \
  --account def-krallen \
  --dry-run
```

Passing `--account` to `sbatch` overrides the default embedded in the wrapper. The
[allocations and compute scheduling](https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling)
page explains how allocation groups map to scheduler access.

## One-time setup on each cluster

Narval and Fir do not share filesystems. Clone or synchronize the repository and
populate the caches separately on each cluster, as described in
[Migrating between clusters](https://docs.alliancecan.ca/wiki/Migrating_between_clusters).

On a login node:

```bash
cd /path/to/llm-compression

export LLMC_VENV=/path/on/this/cluster/to/llmc-venv
./scripts/bootstrap_env.sh --venv "$LLMC_VENV"

export LLMC_CACHE_ROOT="$SCRATCH/llm-compression/cache"
./scripts/prefetch.sh \
  --config configs/main.yaml \
  --cache-root "$LLMC_CACHE_ROOT"
```

The bootstrap script loads `StdEnv/2023`, `gcc`, `arrow/23.0.1`, and `python/3.11` before it
creates the virtualenv. Alliance supplies PyArrow through the Arrow module and places a
deliberately non-installable PyArrow placeholder in its wheelhouse, so the module must
also be loaded by every compute job. Both the bootstrap and runtime scripts do this.
The pinned module is available on both Narval and Fir and satisfies the PyArrow 21+
requirement of the resolved Datasets package. To use a later compatible module:

```bash
module spider arrow
export LLMC_ARROW_MODULE=arrow/X.Y.Z
./scripts/bootstrap_env.sh --venv "$LLMC_VENV"
```

When `$SCRATCH` is defined, bootstrap also puts the virtualenv application data and pip
cache under scratch instead of the quota-limited home directory. It creates the virtualenv
with `--no-download` and installs from the Alliance wheelhouse with `--no-index`. It asks
for PyTorch 2.5.1 or newer because that is the minimum documented for H100 support on Fir.
See [Arrow](https://docs.alliancecan.ca/wiki/Arrow),
[standard software environments](https://docs.alliancecan.ca/wiki/Standard_software_environments),
[Python environments](https://docs.alliancecan.ca/wiki/Python), and
[PyTorch](https://docs.alliancecan.ca/wiki/PyTorch).

`prefetch.sh` reloads the same compiler, Arrow, and Python modules used to create the
environment, then runs `prefetch.py`. This wrapper is necessary because module changes made
inside `bootstrap_env.sh` do not persist in the parent login shell. It downloads the
model/tokenizer snapshot, WikiText-2 test data, and the
lm-eval datasets for BoolQ, PIQA, HellaSwag, WinoGrande, ARC, OpenBookQA, and MMLU. It
also invokes the bounded FineWeb `prepare-data` command while login-node network
access is available. lm-eval materialization is best-effort because its task API can
change between 0.4.x releases: treat any warning from that step as a preflight failure
and resolve it before submitting an offline evaluation. Compute jobs set the Hugging
Face, Transformers, Datasets, and evaluation libraries to offline mode. This is
required on Narval compute nodes and makes Fir runs reproducible as well. The
[Hugging Face guidance](https://docs.alliancecan.ca/wiki/Huggingface) covers cache
placement and offline operation.

The model configs intentionally leave `revision: null` during initial development.
After the first successful prefetch, record and set an immutable Hugging Face commit
revision before producing final paper results.

## Submit the complete dependency chain

First inspect the generated commands:

```bash
./scripts/submit_pipeline.sh \
  --cluster narval \
  --account def-krallen \
  --config configs/main.yaml \
  --dry-run
```

Then remove `--dry-run`. For Fir, change only `--cluster fir`:

```bash
./scripts/submit_pipeline.sh \
  --cluster fir \
  --account def-krallen \
  --config configs/main.yaml
```

The dependency chain is:

```text
prepare-data -> prepare-eval-data -> cache-teacher
             -> stage1 (5 jobs x at most one threshold scan)
             -> stage2 (14 one-group resumable jobs)
             -> stage3 (12 jobs x at most 250 steps)
             -> teacher evaluation (5 component jobs)
             -> retained Stage 2 evaluation (6 component jobs, including summary)
             -> Stage 3 student evaluation (6 component jobs, including summary)
```

Jobs are linked with `afterok`, so a failed stage stops its dependants. Stage 2 has
one job per possible merge group; once all selected groups are complete, later chunks
exit quickly. Stage 3's 12 chunks cover the configured 3,000 steps. Evaluation is
split into perplexity, commonsense harness, MMLU harness, generation, speed, and final
summary jobs so the full task suite is not forced into one monolithic allocation. The exact
retained Stage 2 model is evaluated before the Stage 3 model, without retraining Stage 2. The
teacher is evaluated separately so both summaries include the required seven-task relative
accuracy. The default dry-run contains 51 jobs. Adjust the chunk counts without editing a
script:

```bash
./scripts/submit_pipeline.sh \
  --cluster narval \
  --stage1-chunks 5 \
  --stage2-chunks 10 \
  --stage3-chunks 16
```

Use repeated `--set` arguments for an auditable variant. They are forwarded to every
stage and applied after the YAML:

```bash
./scripts/submit_pipeline.sh \
  --cluster narval \
  --config configs/main.yaml \
  --set stage1.target_removed_layers=6 \
  --set run_name=qwen3_0_6b_remove_6
```

The standalone YAML targets removing 6 of Qwen3-0.6B's 28 blocks (about 20%) and
records whether its threshold search found an exact budget match. Submit the
approximately 35% target as a separate run so its artifacts cannot collide:

```bash
./scripts/submit_pipeline.sh \
  --cluster narval \
  --config configs/main.yaml \
  --set stage1.target_removed_layers=10 \
  --set run_name=qwen3_0_6b_remove_10
```

The runnable experiment files are:

- `configs/main.yaml`: proposal method, targeting 6 removed blocks; use the
  override above for the 10-block target.
- `configs/baselines.yaml`: teacher, naive drop, SWM, XMerge, and trained
  original-block control.
- `configs/ablations.yaml`: selection score, mixed/teacher probe inputs,
  50/300-step probe, Stage 2 loss, and with/without-Stage-3 axes.

The files under `configs/model`, `configs/data`, `configs/runtime`, and
`configs/cluster` are readable override references, not include files. Runnable YAMLs
are deliberately self-contained.

After a main run has produced its Stage 1 layout, preview the complete equal-depth
baseline matrix with:

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

`--recovery all` includes XMerge's teacher-input local reconstruction, an adapted
SWM causal-LM LoRA recovery, and matched Stage 2+3 recovery for naive drop, SWM,
XMerge, and the original-block drop control. The SWM LoRA run uses the proposal's
FineWeb-Edu split instead of SWM's original recovery corpus; this is recorded as a
dataset-matched adaptation rather than represented as an exact published-data
replication. Use `--recovery xmerge-local`, `swm-lora`, or `same` to run only one
comparison family.

The ablation launcher is dry-run by default and covers trained/untrained selection,
50/300-step probes, Stage 2 lambda policies, and explicit evaluation after Stage 2:

```bash
./scripts/submit_ablation_sweep.sh \
  --cluster narval \
  --account def-krallen \
  --prefix qwen3_0_6b_ablation_r1
```

Review all generated commands, then add `--submit`. The SWM global-similarity
selector cell comes from the baseline launcher with matched recovery. The 300-step
probe cannot resume inside a threshold scan and is the one cell that may need a
separately approved longer limit if cluster timing shows it cannot finish in 1:00.

To run a single command directly, create the log directory first and use the matching
wrapper:

```bash
mkdir -p logs/slurm
sbatch --account=def-krallen \
  scripts/slurm/run_narval.sbatch \
  baseline configs/main.yaml \
  --set baseline.method=swm
```

The command above intentionally uses the completed main run because equal-compression
baselines consume its selected Stage 1 layout. The baseline and ablation sweep launchers
described in the repository README add unique output paths and the appropriate recovery jobs.

## Checkpoint and resume behavior

All launchers request `--signal=B:USR1@300`. `run_stage.sh` forwards that signal to
the Python process, which is configured to write a resumable checkpoint. A four-minute
watchdog then terminates a process that did not stop on its own, leaving time for Slurm
to flush logs. Stage 2 persists progress while training each group, Stage 3 persists at
each 250-step chunk or cooperative stop, and every pipeline invocation sets
`runtime.resume=auto`.

Each Stage 1 job performs at most one threshold scan and is not resumable in the middle of
that scan. If it reaches the wall-time limit, rerun that threshold job from the beginning.
Completed thresholds are reused, and an exact budget match ends the search early. Stage 2
resumes from its per-step state; Stage 3 resumes at the latest chunk/signal checkpoint and
uses held-out KL for early stopping. Evaluation refuses to use an incomplete Stage 3. An
unfinished evaluation component can be rerun with the same config and
`evaluation.component` override. Alliance's
[running jobs](https://docs.alliancecan.ca/wiki/Running_jobs) and
[monitoring jobs](https://docs.alliancecan.ca/wiki/Monitoring_jobs) pages document
`squeue`, `sacct`, and `seff`.

Useful checks are:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,Timelimit,AllocTRES,ExitCode
seff JOB_ID
tail -f logs/slurm/llmc-stage3-1-JOB_ID.out
```

## Storage choices

By default the launcher writes caches, checkpoints, and outputs beneath
`$SCRATCH/llm-compression`. Scratch is high-performance, is not backed up, and files
are subject to the published 60-day purge policy. Copy irreplaceable final metrics,
manifests, and compact checkpoints to project storage. Do not place the only copy of
an active checkpoint in `$SLURM_TMPDIR`: node-local storage disappears when the job
ends.

See [storage and file management](https://docs.alliancecan.ca/wiki/Storage_and_file_management),
the [scratch purge policy](https://docs.alliancecan.ca/wiki/Scratch_purging_policy),
and [node-local storage](https://docs.alliancecan.ca/wiki/Using_node-local_storage).
