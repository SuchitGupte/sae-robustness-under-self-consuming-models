# Self-Consuming Generative Loops for Small LMs

Empirical counterpart to the SAE-degradation analysis. A model generates text, the next
generation is fine-tuned on that text, and the process repeats. A **real lineage** control is
trained on fresh real data for the same number of steps and the same token budget, so any
degradation can be attributed to self-consumption rather than to fine-tuning itself.

In the theory, one loop step is `Σ^(t+1) = Σ^(t) + σ²I`. Here `σ²` is not injected by hand —
it is whatever distortion sampling + retraining introduces. The real lineage is the `σ² = 0`
reference.

Every model is fully fine-tuned (no adapters), so nothing artificially damps the
generation-to-generation drift the experiment is measuring. Checkpoints are staged to a temp
directory and pushed to the Hub; **no weights are kept on local disk**.

## Install

```bash
pip install "transformers>=4.42" datasets accelerate huggingface_hub torch
pip install zstandard       # only if you stream the Pile
huggingface-cli login       # needed to push, and to access gated Gemma repos
```

## Usage

One lineage per invocation. Run both to get the comparison:

```bash
# self-consuming lineage
python self_consume.py --model p160m --generations 4 --lineage synthetic --real-percent 0 \
  --repo suchitg/generations

# real-data control, same model, same budget
python self_consume.py --model p160m --generations 4 --lineage real \
  --repo suchitg/generations
```

Both invocations build the data pool from the same corpus and seed, so the held-out eval
blocks are identical across them and the perplexities are directly comparable.

```bash
# 30% real data mixed into every synthetic generation
python self_consume.py --model p160m --generations 4 --lineage synthetic --real-percent 30

# smoke test: exercises every code path in ~a minute
python self_consume.py --model p70m --generations 1 \
  --blocks-per-gen 20 --metric-blocks 4 --eval-blocks 8 --repo suchitg/generations-smoke

# large model spread over 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python self_consume.py --model g9b --generations 4 \
  --lineage synthetic --device-map auto --grad-ckpt --batch-size 1 --grad-accum 16

# resume from generation 3 (same --model, --real-percent and --lineage as the original run)
python self_consume.py --model p160m --generations 8 --resume-gen 3 --lineage synthetic
```

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--model` | `p160m` | `p70m`, `p160m`, `g2b`, `g9b`, or any HF causal-LM id |
| `--generations` | `5` | number of loop steps `T` |
| `--lineage` | `synthetic` | `synthetic` (self-consuming, optionally mixed) or `real` (control) |
| `--real-percent` | `0` | % of each generation's training blocks drawn from real data; ignored when `--lineage real` (forced to 100) |
| `--resume-gen` | `0` | checkpoint to start from; `0` = base model, `N` = load `<tag>/<lineage>/genN` from `--repo` |
| `--repo` | `suchitg/generations` | HF repo for checkpoints and logs |
| `--tmp-dir` | *(scratch path)* | staging dir for uploads; needs ~1× checkpoint size free (≈18 GB for g9b) |
| `--out-dir` | *(scratch path)* | where logs and metrics files are written |
| `--run-name` | `<tag>-<lineage>-<YYYYMMDD-HHMMSS>` | names the log and metrics files |
| `--device-map` | – | `auto` splits the layers over the visible GPUs |
| `--grad-ckpt`, `--fp32` | off, off (bf16) | memory / precision knobs |
| `--blocks-per-gen` | `2000` | training blocks per generation (blocks × `--seq-len` = tokens per step) |
| `--seq-len` | `512` | block length, used for training, generation and scoring |
| `--eval-blocks` | `100` | fixed held-out real blocks used for perplexity |
| `--metric-blocks` | `64` | generated blocks used for output entropy and n-gram diversity |
| `--max-ngram` | `4` | reports distinct-1 … distinct-n |
| `--temperature`, `--top-p` | `1.0`, `1.0` | sampling for synthetic data; lowering these makes collapse faster |
| `--dataset`, `--dataset-config`, `--split`, `--text-column` | wikitext-103 | real corpus (streamed) |
| `--epochs`, `--lr`, `--batch-size`, `--grad-accum`, `--seed` | `1`, `1e-5`, `4`, `4`, `0` | training |

A "block" is one chunk of `--seq-len` tokens. Mixing is exact, not stochastic:
`n_real = round(real_percent/100 × blocks_per_gen)`, the rest sampled from the previous
generation. Total tokens per step is identical across lineages and mixing ratios, so the only
variable is data provenance.

## Choice of corpus

The default (wikitext-103) is fine for smoke tests but is the wrong corpus for the real
experiment. Pythia was pretrained on the Pile, so gen 0 samples Pile-like text; if "real data"
is Wikipedia, the real lineage becomes domain adaptation rather than a `σ² = 0` control, and
the gap you measure mixes degradation with domain shift. Use a corpus matched to pretraining:

```bash
--dataset monology/pile-uncopyrighted --dataset-config default    # Pythia (needs zstandard)
--dataset HuggingFaceFW/fineweb-edu   --dataset-config sample-10BT  # Gemma (no matched corpus exists)
```

Gemma's pretraining mix is undisclosed, so there is no matched option — either use each
model's best-guess corpus, or run everything on one corpus so cross-model degradation rates
stay comparable. Put the corpus in `--run-name` if you compare across corpora.

## Multi-GPU

`CUDA_VISIBLE_DEVICES` alone only chooses *which* GPUs the process sees — the model still
lands on one. Add `--device-map auto` and the layers are distributed across them; the log
prints the resulting map. This is layer-wise parallelism, so it buys memory, not speed.

Total memory for full AdamW fine-tuning (params + grads + optimizer, bf16 weights), summed
across all visible GPUs:

| model | weights + grads + optimizer |
|---|---|
| p70m | 1 GB |
| p160m | 3 GB |
| g2b | 42 GB |
| g9b | 147 GB |

On 4×40GB: the Pythia models fit on one card, `g2b` across two, `g9b` needs all four with
`--device-map auto --grad-ckpt`. `g9b` is tight — keep `--batch-size 1` and lower `--seq-len`
if it OOMs. Point `--tmp-dir` at node-local scratch rather than NFS for `g9b`, or the 18 GB
checkpoint crosses the network twice per push.

## Logging and metrics

Every invocation is stamped with the date and time it was called:

```
<out-dir>/logs/<run_name>.log            # console transcript, every line timestamped
<out-dir>/metrics/<run_name>.json/.csv   # one row per checkpoint, gen0 .. genT
```

Each checkpoint is scored right after it is trained and the metrics file is rewritten (and
re-uploaded) at that point, so it stays complete and readable even if the run is interrupted.
With `--generations 4` the file holds gen0 (the base model, unmodified) through gen4:

| field | measured on | meaning |
|---|---|---|
| `perplexity`, `real_loss` | fixed held-out **real** blocks | how well the checkpoint still models real data |
| `output_entropy_nats` / `_bits` | the checkpoint's **own samples** | mean predictive entropy per token; falls as the model sharpens onto its own modes |
| `distinct_1` … `distinct_4` | the checkpoint's **own samples** | unique n-grams / total n-grams; falls as generations start repeating themselves |

Each row also carries `run_name`, `date`, `time`, `timestamp`, `model`, `lineage`,
`generation` and `real_percent`, so rows from separate runs concatenate cleanly. Entropy and
diversity use the same sampling settings as the loop itself.

## Repo layout

```
suchitg/generations/
├── p160m-mix0/synthetic/gen1/ ... gen4/     # weights, tokenizer, metrics.json
├── p160m-mix0/real/gen1/      ... gen4/
├── p160m-mix30/synthetic/gen1/ ... gen4/
└── logs/<run_name>.json|.csv|.log
```

The tag is `<model>-mix<pct>`, with no timestamp — that is what makes `--resume-gen` able to
find a previous run, and it also means re-running the same model and mix **overwrites** those
checkpoints in place. Load any checkpoint with:

```python
AutoModelForCausalLM.from_pretrained("suchitg/generations",
                                     subfolder="p160m-mix0/synthetic/gen3")
```

## Notes

- Gemma-2 is loaded with eager attention (recommended for that architecture) on both fresh and
  resumed runs. Gemma repos are gated — accept the license on the model page first.
- Synthetic data is sampled unconditionally from BOS, so degradation is not confounded by
  real-data prompts leaking into the synthetic corpus.
- The real pool is streamed and handed out in disjoint slices, so no generation trains on the
  same real blocks twice (it wraps around, with a reset, if the pool is exhausted).
- Optimizer states are freed after each generation's training call, before sampling, so
  training and generation memory never coexist.
- Entropy is computed in chunks over sequence positions, since Gemma's 256k-wide vocabulary
  makes a full-sequence softmax expensive. Lower `--metric-blocks` if scoring is slow.
- Sampling dominates wall-clock; raise `--gen-batch-size` before anything else.
- Check `--tmp-dir` is empty after a run: that confirms the staging cleanup fired.

## Suggested next step

For SAE analysis, stream activations from each pushed checkpoint inside the SAE training loop
rather than dumping them here — SAE training needs 10M–1B tokens of activations, which is
hundreds of GB per checkpoint if materialized. The checkpoints plus the corpus are enough to
regenerate any layer or hook point you want. Then track SAE reconstruction error and the
firing-threshold sparsity index `Ψ^(t)` against `t`, alongside the perplexity, entropy and
diversity columns. The theory predicts `Ψ^(t)` degrades on synthetic data but stays bounded
away from zero when evaluated on real data.