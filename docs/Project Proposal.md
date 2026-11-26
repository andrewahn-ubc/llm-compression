# AI 360: Deep Learning — Group Project Proposal

## Probe-Guided Layer Merging for Depth Compression of LLMs

**Group:** Jinho and Derek and Andrew

---

## 1 Group Composition

- Member 1: Andrew Ahn
- Member 2: Jinho Park
- Member 3: Derek Wang

## 2 Introduction and Problem Definition

Large language models (LLMs) are expensive to run because every token passes through a deep stack
of transformer blocks. Prior work shows that many consecutive blocks change the hidden state only
slightly, so removing or merging them can reduce depth with limited quality loss. The result is a
standard, shallower transformer that runs faster on ordinary hardware. This is a deep learning
problem because the replacement for a group of blocks is itself a neural network that must be
trained to reproduce the original model's internal representations.

**Research question.** Can a single trained transformer block replace a group of consecutive
blocks, and does choosing groups by how well a briefly trained replacement reproduces them (rather
than by redundancy scores) give better accuracy at the same compression?

**Scope.** We compress a pretrained Qwen3-0.6B, and Qwen3-1.7B if resources allow. The number of
removed blocks is set by Stage 1's threshold, chosen on held-out data, and all baselines remove the
same number of blocks. We aim to show that our method retains more accuracy than SWM and XMerge at
equal compression, and to measure which components of the method matter.

## 3 Related Work and Background

ShortGPT [1] and Gromov et al. [2] show that many LLM layers are redundant and can be removed. LaCo
[3] and Sliding-Window Merging (SWM) [4] merge consecutive layers by weight arithmetic, growing each
group while the merged model's final hidden state stays similar to the original, then recover with
LoRA. LLM-Streamline [5], ReplaceMe [6], and Puzzle [7] instead replace removed layers with modules
trained to match the original activations (local distillation). XMerge [8], the most relevant work,
merges adjacent pairs by training one surviving block to reproduce the original two-block output,
and evaluates on Qwen3-0.6B; it selects pairs with redundancy scores computed before any training.
Minitron [10] recovers pruned models with logit distillation, and Wagner et al. [11] find local
reconstruction can match full fine-tuning at far lower cost.

**Our contribution.** Prior methods decide what to merge before any training, by untrained weight
merging (SWM, LaCo) or by redundancy scores of the original blocks (XMerge, ShortGPT, ReplaceMe).
Our method is original in three ways:

1. **Train-then-decide selection.** A group is accepted only if a briefly trained replacement block
   reproduces its output on held-out data. Groups are chosen by whether a trained block can actually
   replace them, not by how redundant the original blocks appear.
2. **Variable-size groups replaced by full transformer blocks.** Groups grow greedily while the probe
   succeeds, so one block can replace two or more consecutive blocks. XMerge merges only fixed
   pairs, LLM-Streamline [5] replaces a single span with a lightweight module, and ReplaceMe uses
   linear maps.
3. **Recovery designed for stitching.** Each merged block is trained on both the original model's
   inputs and the compressed model's actual inputs, then the full model is trained with global logit
   distillation. Among the layer-merging methods we reviewed, none combines local reconstruction
   with global output-level distillation as part of the method: SWM uses LoRA fine-tuning, XMerge
   uses local reconstruction only, and CoMe [9] uses layer-wise feature alignment only.

## 4 Datasets

- **FineWeb-Edu** [14] (<https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>, `sample-10BT`):
  filtered educational web text. Separate slices (about 30 to 40M tokens in total) are used for
  probe training, local distillation, and global distillation, plus a held-out slice for probe
  errors, threshold selection, and early stopping.
- **WikiText-2** [13]: the test split is used only to report perplexity, so that no training data
  comes from the evaluation domain.
- **Benchmarks** via lm-evaluation-harness [16]: BoolQ, PIQA, HellaSwag, WinoGrande, ARC-easy,
  ARC-challenge, and OpenBookQA (zero-shot, matching SWM), plus MMLU [15] (5-shot). These are used
  only for evaluation.

**Preprocessing.** Text is tokenized with the Qwen3 tokenizer, concatenated, and split into
512-token sequences. We cache the original model's hidden states at every block boundary for the
calibration set (about 3 to 4 GB). No augmentation is needed.

**Licensing.** FineWeb-Edu is ODC-By and WikiText is CC BY-SA. Qwen3 models are Apache 2.0. The
training data is plain text, since the original model provides all targets, so no labels are
required.

## 5 Available Hardware and Computational Resources

We will use Google Colab (NVIDIA T4, 16 GB). *[EDIT: list any additional GPUs, e.g., Colab Pro A100
or a university cluster.]* Probes and local distillation train one block at a time and need little
memory; global distillation is the most expensive step. We estimate 20 to 50 GPU-hours in total.

> The bracketed EDIT note is an unfilled placeholder in the submitted PDF. The implementation in this
> repository actually targets the Alliance Canada Narval (A100) and Fir (H100) clusters; see
> `docs/HPC.md`.

## 6 Proposed Model and Methodology

**Model and Starting Point.** We use the pretrained Qwen3-0.6B [12] (28 decoder blocks, hidden size
1024), a modern decoder-only transformer, and Qwen3-1.7B if resources allow. We do not train from
scratch. The original model stays frozen as the teacher, and the student is the same model with
groups of consecutive blocks replaced by single trained blocks.

**Proposed Approach.** Let $h_k$ be the original model's hidden state after block $k$. A merged block
$B$ that replaces blocks $i$ to $j$ is scored with a relative error, which measures the fraction of
the group's change it fails to reproduce:

$$
e(\hat h) = \frac{\sum_{n,t} \lVert \hat h_{n,t} - h_{j,n,t} \rVert_2^2}{\sum_{n,t} \lVert h_{j,n,t} - h_{i-1,n,t} \rVert_2^2},
$$

where $n$ indexes sequences and $t$ indexes tokens. $B$ is trained on both the teacher's input
$h_{i-1}$ and the student's actual input $h^S$ (the output of the compressed model up to block
$i-1$):

$$
\mathcal{L}_{\text{local}} = \lambda \, e\big(B(h_{i-1})\big) + (1 - \lambda) \, e\big(B(h^S)\big).
$$

The method has three stages:

1. **Greedy grouping with probes.** Scanning front to back from the first unprotected block (the
   first two and last blocks are protected), we open a group of two blocks, initialize the merged
   block with SWM weight arithmetic, and train it for about 50 steps with $\mathcal{L}_{\text{local}}$.
   If the held-out error is below a threshold $T$, the group grows by one block (warm-starting from
   the current block), up to 4 blocks; otherwise the group closes at its last accepted size.
2. **Mixed local distillation.** Accepted blocks are trained further with $\mathcal{L}_{\text{local}}$,
   in network order, with $\lambda$ fixed at 0.5 or scheduled from 1.0 to 0.5. Only $B$ is updated.
   Unmerged blocks keep their original weights.
3. **Global distillation.** The stitched student is trained end to end to match the teacher's
   next-token distribution:

   $$
   \mathcal{L}_{\text{KD}} = \frac{\tau^2}{N} \sum_{n,t} \mathrm{KL}\big(\mathrm{softmax}(z^T_{n,t}/\tau) \,\big\Vert\, \mathrm{softmax}(z^S_{n,t}/\tau)\big), \qquad \tau = 1.
   $$

   ($z^T$, $z^S$: teacher and student logits; $N$: number of token positions.) All student
   parameters are trained. If GPU memory is insufficient, only merged blocks are trained.

**Optimization.** Adam is used throughout. Local stages use learning rate $10^{-5}$ and batch size
16. Stage 3 uses learning rate $10^{-5}$ to $5 \times 10^{-5}$ with warmup for 1,000 to 3,000 steps,
with fp16 and loss scaling (the T4 lacks native bf16), gradient checkpointing, and an 8-bit
optimizer.

**Inference.** The output is a standard shallower transformer with no added modules, used exactly
like the original model.

### Baselines and Evaluation

- **Baselines** (same number of removed blocks): the original model (upper bound), naive removal of
  the same blocks, SWM [4], and XMerge [8], each run as published on our calibration data. A control
  keeps one original block per group and deletes the rest, then applies our Stages 2 and 3, testing
  whether merging beats dropping under equal training.
- **Metrics:** WikiText-2 perplexity, average and relative accuracy on the seven zero-shot tasks,
  MMLU accuracy, a short free-form generation check, parameter count, inference speed.
- **Ablations:**
  1. Group selection by trained probe error vs. untrained merge error, to test whether training
     before deciding helps, plus SWM's global similarity test if resources allow.
  2. $\lambda$ fixed at 0.5 vs. scheduled from 1.0 to 0.5, plus $\lambda = 1.0$ (teacher input only)
     if resources allow.
  3. Accuracy after Stage 2 vs. after Stage 3, to measure what each recovery stage contributes.
  4. 50-step vs. 300-step probes, to test whether short probes rank groups correctly.
- **Expected results:** we expect higher accuracy retention than SWM and XMerge at equal
  compression, with the gap growing at higher compression where larger groups are needed. If
  probe-based selection gives no gain over redundancy scores, as XMerge's ablation suggests is
  possible, we will report this as a finding.

## 7 Timeline and Deliverables

| Time Period | Milestone / Activities | Expected Deliverable |
| --- | --- | --- |
| Week 1 | Shared formats; data pipeline; caching | Finalized plan and data pipeline |
| Week 2 | All stages on Qwen3-0.6B, small settings | **Milestone:** first end-to-end run |
| Week 3 | Threshold selection; main runs; baselines | Initial method and baseline results |
| Week 4 | Ablations; Qwen3-1.7B if possible | Complete results tables |
| Week 5 | Final analysis, report, and slides | Final report and slides |

The Week 2 milestone tests feasibility. If probes cannot merge groups larger than pairs on
Qwen3-0.6B, we will fall back to fixed pairs and focus on the selection and mixed-input comparisons.

## 8 Division of Work

Work is divided by method stage, with supporting tasks to balance the load. Each member develops
against a stand-in input (a hand-picked grouping for Stage 2, a naively pruned model for Stage 3)
until real outputs are available. In Week 1 the team fixes shared grouping and checkpoint formats,
and we integrate through a shared repository and weekly meetings.

| Team Member | Primary Responsibilities | Expected Contributions |
| --- | --- | --- |
| Andrew, Jinho, Derek | Literature review and problem formulation | Related work and project design |
| Andrew | Stage 2; data pipeline and caching | Stage 2 code; data pipeline |
| Jinho | Stage 1; shared local-distillation code | Stage 1 code; probe ablations |
| Derek | Stage 3; evaluation; SWM and XMerge baselines | Stage 3 code; metrics and visualization |

## References

1. X. Men et al. ShortGPT: Layers in large language models are more redundant than you expect.
   arXiv:2403.03853, 2024.
2. A. Gromov, K. Tirumala, H. Shapourian, P. Glorioso, D. A. Roberts. The unreasonable
   ineffectiveness of the deeper layers. In *ICLR*, 2025.
3. Y. Yang, Z. Cao, H. Zhao. LaCo: Large language model pruning via layer collapse.
   arXiv:2402.11187, 2024.
4. X. Ding, Y. Zhu, Y. Zhang, C. Xie. Sliding-window merging for compacting patch-redundant layers
   in LLMs. In *AAAI*, 2026. arXiv:2502.19159.
5. X. Chen et al. Streamlining redundant layers to compress large language models. In *ICLR*, 2025.
6. D. Shopkhoev et al. ReplaceMe: Network simplification via depth pruning and transformer block
   linearization. arXiv:2505.02819, 2025.
7. A. Bercovich et al. Puzzle: Distillation-based NAS for inference-optimized LLMs. In *ICML*, 2025.
8. J. Hu, S. Ramachandran. XMerge: Cross-axis selection and reconstructive layer merging for LLM
   depth compression. In *NeurIPS*, 2026. arXiv:2609.02083.
9. F. Wang, L. Shen, L. Ding, C. Xue, Y. Liu, C. Ding. Layer as puzzle pieces: Compressing large
   language models through layer concatenation. In *NeurIPS*, 2025.
10. S. Muralidharan et al. Compact language models via pruning and knowledge distillation. In
    *NeurIPS*, 2024.
11. M. Wagner, C. Roux, M. Zimmer, S. Pokutta. A free lunch in LLM compression: Revisiting
    retraining after pruning. arXiv:2510.14444, 2025.
12. A. Yang et al. Qwen3 technical report. arXiv:2505.09388, 2025.
13. S. Merity, C. Xiong, J. Bradbury, R. Socher. Pointer sentinel mixture models. In *ICLR*, 2017.
14. G. Penedo et al. The FineWeb datasets: Decanting the web for the finest text data at scale. In
    *NeurIPS Datasets and Benchmarks*, 2024.
15. D. Hendrycks et al. Measuring massive multitask language understanding. In *ICLR*, 2021.
16. L. Gao et al. A framework for few-shot language model evaluation (lm-evaluation-harness).
    Zenodo, 2023.
