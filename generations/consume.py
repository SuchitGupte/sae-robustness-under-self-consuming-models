#!/usr/bin/env python
"""
Self-consuming generative loop for small LMs (Pythia / Gemma-2).
 
Two lineages:
  synthetic : gen_t is fine-tuned on text sampled from gen_{t-1}  (+ optional real data mix)
  real      : gen_t is fine-tuned on fresh real data only         (control lineage)
 
Each generation is saved to a Hugging Face repo under <lineage>/gen<t>/, and every
checkpoint is scored for perplexity, output entropy and n-gram diversity. All scores
are appended to a single timestamped run log.
"""
 
import argparse, csv, gc, json, logging, math, sys
from datetime import datetime
from pathlib import Path
 
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)
import tempfile
 
MODELS = {
    "p70m":  "EleutherAI/pythia-70m-deduped",
    "p160m": "EleutherAI/pythia-160m-deduped",
    "g2b":  "google/gemma-2-2b",
    "g9b":  "google/gemma-2-9b",
}

log = logging.getLogger("selfconsume")
def now():
    return datetime.now()
 
def setup_logging(run_name, log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{run_name}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(path)])
    return path

"""
Methods to help load model, tokenizer and data
"""

def load_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path, args, subfolder=None):
    kw = dict(torch_dtype=torch.bfloat16 if args.bf16 else torch.float32)
    if args.is_gemma:
        kw["attn_implementation"] = "eager"     # recommended for Gemma-2
    if subfolder:
        kw["subfolder"] = subfolder
    if args.device_map:                          # split layers over the visible GPUs
        kw["device_map"] = args.device_map
        model = AutoModelForCausalLM.from_pretrained(path, **kw)
        log.info(f"[init] device map: {getattr(model, 'hf_device_map', 'n/a')}")
        return model
    return AutoModelForCausalLM.from_pretrained(path, **kw).to(args.device)


class RealPool:
    """Streams the real corpus once into fixed-length token blocks, hands out disjoint slices."""
 
    def __init__(self, tok, args, n_blocks):
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
        ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
        blocks, buf = [], []
        for ex in ds:
            text = (ex[args.text_column] or "").strip()
            if not text:
                continue
            buf.extend(tok(text).input_ids + [tok.eos_token_id])
            while len(buf) >= args.seq_len and len(blocks) < n_blocks:
                blocks.append(buf[:args.seq_len])
                buf = buf[args.seq_len:]
            if len(blocks) >= n_blocks:
                break
        self.blocks, self.cursor = blocks, 0
        log.info(f"[data] real pool: {len(blocks)} blocks of {args.seq_len} tokens")
 
    def take(self, n):
        if n <= 0:
            return []
        if self.cursor + n > len(self.blocks):      # wrap around if the pool runs dry
            self.cursor = 0
        out = self.blocks[self.cursor:self.cursor + n]
        self.cursor += n
        return out
    

@torch.no_grad()
def generate_blocks(model, tok, n_blocks, args):
    """Sample continuations, then pack into seq_len blocks like the real pool."""
    model.eval()
    bos = tok.bos_token_id or tok.eos_token_id
    blocks, buf = [], []
    while len(blocks) < n_blocks:
        prompt = torch.full((args.gen_batch_size, 1), bos, device=model.device)
        out = model.generate(prompt, do_sample=True, temperature=args.temperature,
                             top_p=args.top_p, max_new_tokens=args.seq_len,
                             pad_token_id=tok.pad_token_id)
        for row in out[:, 1:].cpu().tolist():
            if tok.eos_token_id in row:                 # trim trailing pad, keep one EOS
                row = row[:row.index(tok.eos_token_id) + 1]
            buf.extend(row)
            while len(buf) >= args.seq_len and len(blocks) < n_blocks:
                blocks.append(buf[:args.seq_len])
                buf = buf[args.seq_len:]
    log.info(f"[gen]  sampled {len(blocks)} blocks")
    return blocks[:n_blocks]

"""
Finetuning trainer
"""
 
def finetune(model, blocks, tok, args, out_dir):
    model.train()
    if args.grad_ckpt:
        model.config.use_cache = False
 
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / "trainer"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            bf16=args.bf16,
            gradient_checkpointing=args.grad_ckpt,
            logging_steps=20,
            save_strategy="no",
            report_to=[],
            seed=args.seed,
        ),
        train_dataset=Dataset.from_dict({"input_ids": blocks}),
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
 
    del trainer                      # free optimizer states before sampling
    gc.collect()
    torch.cuda.empty_cache()
 
    model.config.use_cache = True
    return model


"""
Metric computation and logging
"""
 
@torch.no_grad()
def real_loss(model, blocks, args):
    """Mean cross-entropy (nats/token) on held-out real data -> perplexity."""
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(blocks), args.batch_size):
        ids = torch.tensor(blocks[i:i + args.batch_size], device=model.device)
        total += model(input_ids=ids, labels=ids).loss.item() * len(ids)
        n += len(ids)
    return total / max(n, 1)


@torch.no_grad()
def output_entropy(model, blocks, args):
    """Mean predictive entropy (nats/token) of the model over its own generated text."""
    model.eval()
    total, n = 0.0, 0
    for b in blocks:
        ids = torch.tensor([b], device=model.device)
        logits = model(input_ids=ids).logits[0]
        for s in range(0, logits.shape[0], 128):        # chunked: vocab can be 256k wide
            lp = torch.log_softmax(logits[s:s + 128].float(), dim=-1)
            total += float(-(lp.exp() * lp).sum(-1).sum())
            n += lp.shape[0]
        del logits
    return total / max(n, 1)

def ngram_diversity(blocks, max_n=4):
    """distinct-n: unique n-grams / total n-grams over the generated corpus."""
    out = {}
    for n in range(1, max_n + 1):
        grams, total = set(), 0
        for b in blocks:
            for i in range(len(b) - n + 1):
                grams.add(tuple(b[i:i + n]))
                total += 1
        out[f"distinct_{n}"] = len(grams) / max(total, 1)
    return out

def score_checkpoint(model, tok, eval_blocks, args, lineage, gen):
    """Perplexity (real data) + output entropy and n-gram diversity (model's own samples)."""
    t0 = now()
    loss = real_loss(model, eval_blocks, args)
    samples = generate_blocks(model, tok, args.metric_blocks, args)
    rec = {
        "run_name": args.run_name,
        "date": t0.strftime("%Y-%m-%d"),
        "time": t0.strftime("%H:%M:%S"),
        "timestamp": t0.isoformat(timespec="seconds"),
        "model": MODELS.get(args.model, args.model),
        "lineage": lineage,
        "generation": gen,
        "real_percent": args.real_percent if lineage == "synthetic" else 100.0,
        "real_loss": round(loss, 6),
        "perplexity": round(math.exp(min(loss, 50)), 4),
        "output_entropy_nats": round(output_entropy(model, samples, args), 6),
    }
    rec["output_entropy_bits"] = round(rec["output_entropy_nats"] / math.log(2), 6)
    rec.update({k: round(v, 6) for k, v in ngram_diversity(samples, args.max_ngram).items()})
    log.info(f"[score] {lineage} gen{gen} | ppl={rec['perplexity']:.2f} "
             f"| entropy={rec['output_entropy_nats']:.3f} nats | distinct-1..n=" +
             "/".join(f"{rec[f'distinct_{n}']:.3f}" for n in range(1, args.max_ngram + 1)))
    return rec


class MetricsLog:
    """One cumulative file per run: every checkpoint of every lineage, in call order."""
 
    def __init__(self, out_dir, run_name, started):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = out_dir / f"{run_name}.json"
        self.csv_path = out_dir / f"{run_name}.csv"
        self.run_name, self.started, self.records = run_name, started, []
 
    def add(self, rec):
        self.records.append(rec)
        self.json_path.write_text(json.dumps({
            "run_name": self.run_name,
            "run_started_date": self.started.strftime("%Y-%m-%d"),
            "run_started_time": self.started.strftime("%H:%M:%S"),
            "checkpoints": self.records,
        }, indent=2))
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.records[0].keys()))
            w.writeheader()
            w.writerows(self.records)
 
    def push(self, args):
        if args.repo:
            api = HfApi()
            for p in (self.json_path, self.csv_path):
                api.upload_file(repo_id=args.repo, path_or_fileobj=str(p),
                                path_in_repo=f"logs/{p.name}", repo_type="model")


 

"""
Lineage loop: Need to verify after pasting main
"""

def save_and_push(model, tok, args, lineage, t, rec):
    """Serialize to a temp dir, upload, leave nothing behind on local disk."""
    if not args.repo:
        log.warning("[push] no --repo set; checkpoint discarded")
        return
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp:
        tmp = Path(tmp)
        model.save_pretrained(tmp, safe_serialization=True)
        tok.save_pretrained(tmp)
        (tmp / "metrics.json").write_text(json.dumps(rec, indent=2))
        HfApi().upload_folder(repo_id=args.repo, folder_path=str(tmp),
                              path_in_repo=f"{args.tag}/{lineage}/gen{t}", repo_type="model")
        log.info(f"[push] {args.repo}/{args.tag}/{lineage}/gen{t}")
 
 
def run_lineage(lineage, real_frac, args, tok, pool, eval_blocks, mlog):
    """real_frac = fraction of each generation's training blocks drawn from real data."""
    out_dir = Path(args.out_dir) / lineage
    out_dir.mkdir(parents=True, exist_ok=True)
    base = MODELS.get(args.model, args.model)
 
    # start from scratch, or resume a checkpoint already in the repo
    if args.resume_gen > 0:
        src = args.repo or str(Path(args.out_dir) / lineage)
        model = load_model(src, args, subfolder=f"{args.tag}/{lineage}/gen{args.resume_gen}")
        log.info(f"[init] resumed {lineage} at gen{args.resume_gen}")
    else:
        model = load_model(base, args)
 
    # score the starting point too, so the log covers gen0..genT
    mlog.add(score_checkpoint(model, tok, eval_blocks, args, lineage, args.resume_gen))
    mlog.push(args)
 
    n_real = round(real_frac * args.blocks_per_gen)
    n_synth = args.blocks_per_gen - n_real
 
    for t in range(args.resume_gen + 1, args.generations + 1):
        log.info(f"=== {lineage} | gen {t}/{args.generations} "
                 f"| {n_real} real + {n_synth} synthetic blocks ===")
 
        blocks = pool.take(n_real)
        if n_synth:
            blocks += generate_blocks(model, tok, n_synth, args)
 
        model = finetune(model, blocks, tok, args, out_dir)
 
        rec = score_checkpoint(model, tok, eval_blocks, args, lineage, t)
        mlog.add(rec)
        save_and_push(model, tok, args, lineage, t, rec)
        mlog.push(args)
 
    del model
    gc.collect()
    torch.cuda.empty_cache()


"""
Main Configuration
"""

def main():
    p = argparse.ArgumentParser(description="Self-consuming fine-tuning loop")
    p.add_argument("--model", default="p160m", help=f"one of {list(MODELS)} or any HF causal-LM id")
    p.add_argument("--generations", type=int, default=5, help="number of loop steps T")
    p.add_argument("--resume-gen", type=int, default=0, help="checkpoint to start from; 0 = base model, N = load <lineage>/genN")

    p.add_argument("--repo", default=None, help="HF repo; defaults to suchitg/generations-<model>")
    p.add_argument("--real-percent", type=float, default=0.0, help="%% real data mixed into the synthetic lineage (0=pure self-consuming)")
    p.add_argument("--lineage", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--run-name", default=None, help="default: <model>-mix<pct>-<date>-<time>")
    p.add_argument("--tmp-dir", default='/research/nfs_khalili_17/gupte.31/selfconsuming/sae-robustness-under-self-consuming-models/generations/tmp', help="scratch dir for staging uploads; /tmp is often too small for 9B")
 
    p.add_argument("--dataset", default="monology/pile-uncopyrighted")
    p.add_argument("--dataset-config", default="default")
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--blocks-per-gen", type=int, default=2000, help="training blocks per generation")
    p.add_argument("--eval-blocks", type=int, default=100, help="held-out real blocks for perplexity")
    p.add_argument("--metric-blocks", type=int, default=64, help="generated blocks used for output entropy and n-gram diversity")
    p.add_argument("--max-ngram", type=int, default=4)
 
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--device-map", default=None, help="'auto' splits the layers over the visible GPUs (see CUDA_VISIBLE_DEVICES)")
 
    p.add_argument("--gen-batch-size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
 
    p.add_argument("--out-dir", default="/research/nfs_khalili_17/gupte.31/selfconsuming/sae-robustness-under-self-consuming-models/generations/runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp32", dest="bf16", action="store_false")
    
    args = p.parse_args()
 
    started = now()
    args.tag = f"mix{args.real_percent:g}"
    args.is_gemma = "gemma" in MODELS.get(args.model, args.model)
    args.repo = args.repo or f"suchitg/generations-{args.model.replace('/', '-')}"
    args.run_name = args.run_name or \
        f"{args.model.replace('/', '-')}-{args.tag}-{args.lineage}-{started:%Y%m%d-%H%M%S}"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # Setup logging
    log_path = setup_logging(args.run_name, Path(args.out_dir) / "logs")
    log.info(f"[run] {args.run_name} started {started:%Y-%m-%d %H:%M:%S} "
             f"| {torch.cuda.device_count()} visible GPU(s)")
    log.info(f"[run] args: {json.dumps(vars(args))}")
    mlog = MetricsLog(Path(args.out_dir) / "metrics", args.run_name, started)

    # Load tokenizer
    tok = load_tokenizer(MODELS.get(args.model, args.model))
    real_frac = 1.0 if args.lineage == "real" else args.real_percent / 100.0
    lineages = [(args.lineage, real_frac)]

    # Define dataset requirements
    needed = args.eval_blocks + args.generations * args.blocks_per_gen * len(lineages)
    pool = RealPool(tok, args, needed)
    eval_blocks = pool.take(args.eval_blocks)

    # Create huggingFace repo 
    if args.repo:
        HfApi().create_repo(args.repo, repo_type="model", exist_ok=True)
 
    for name, frac in lineages:
        run_lineage(name, frac, args, tok, pool, eval_blocks, mlog)

    # Upload to HuggingFace
    if args.repo:
        HfApi().upload_file(repo_id=args.repo, path_or_fileobj=str(log_path),
                            path_in_repo=f"logs/{log_path.name}", repo_type="model")
    log.info(f"[run] finished {now():%Y-%m-%d %H:%M:%S} "
             f"({(now() - started).total_seconds() / 60:.1f} min) -> {mlog.json_path}")
 
 
if __name__ == "__main__":
    main()