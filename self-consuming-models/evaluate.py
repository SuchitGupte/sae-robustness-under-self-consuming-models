"""
evaluate.py — Load each generation's checkpoint and compute collapse metrics.

Metrics computed per generation:
  - Held-out perplexity      (on frozen real-text reference set)
  - Self-BLEU                (output similarity to each other)
  - Type-Token Ratio (TTR)   (per-sample average)
  - Unigram entropy          (token distribution entropy)
  - KL divergence            (drift from generation 1's token distribution)
  - N-gram repetition        (4-gram, re-computed for consistency)
  - Unique token ratio       (global, re-computed for consistency)

Usage
-----
    # Uses manifest.json to find all checkpoints automatically
    python evaluate.py --checkpoint_dir ./checkpoints --logs_dir ./loop_logs

    # Override reference corpus (default: pulls WikiText-2 from HuggingFace)
    python evaluate.py --checkpoint_dir ./checkpoints --reference_corpus path/to/ref.txt

    # Skip slow metrics
    python evaluate.py --checkpoint_dir ./checkpoints --no_selfbleu

Requirements
------------
    pip install transformers torch datasets nltk numpy tqdm
"""

import argparse
import json
import logging
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evaluate.log"),
    ],
)
logger = logging.getLogger(__name__)

def load_reference_sentences(path: str | None, n: int = 200) -> list[str]:
    """
    Load n sentences from a reference corpus of real human text.
    If path is None, downloads WikiText-2 validation split from HuggingFace.
    If path points to a .txt file, reads lines from it.
    """
    if path is not None:
        logger.info(f"Loading reference corpus from: {path}")
        lines = Path(path).read_text().splitlines()
        lines = [l.strip() for l in lines if len(l.strip()) > 30]
        assert len(lines) >= n, (
            f"Reference file has only {len(lines)} usable lines; need at least {n}"
        )
        sentences = lines[:n]
    else:
        logger.info("Downloading WikiText-2 validation split as reference corpus...")
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            sentences = [
                row["text"].strip()
                for row in ds
                if len(row["text"].strip()) > 30
            ][:n]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load WikiText-2: {e}\n"
                "Pass --reference_corpus path/to/file.txt instead."
            )

    assert len(sentences) == n, f"Expected {n} sentences, got {len(sentences)}"
    logger.info(f"Reference corpus ready: {n} sentences")
    return sentences


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load tokenizer + model from a checkpoint directory."""
    logger.info(f"  Loading checkpoint: {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Metric implementations
# ──────────────────────────────────────────────────────────────────────────────

def compute_perplexity(
    model,
    tokenizer,
    sentences: list[str],
    device: torch.device,
    max_length: int = 512,
) -> float:
    """
    Compute mean perplexity over `sentences` using the model's NLL loss.
    Higher = model has drifted further from real language.
    """
    nlls = []
    for sent in tqdm(sentences, desc="    Perplexity", leave=False):
        enc = tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        # Need at least 2 tokens to compute NLL
        if enc["input_ids"].shape[1] < 2:
            continue

        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])

        # out.loss is mean NLL per token; convert to total NLL for this sentence
        nlls.append(out.loss.item() * enc["input_ids"].shape[1])

    assert len(nlls) > 0, "No valid sentences for perplexity computation"
    mean_nll = sum(nlls) / sum(
        min(tokenizer(s, return_tensors="pt")["input_ids"].shape[1], max_length)
        for s in sentences
        if tokenizer(s, return_tensors="pt")["input_ids"].shape[1] >= 2
    )
    return math.exp(mean_nll)


def compute_selfbleu(texts: list[str], n: int = 4, sample_size: int = 200) -> float:
    """
    Self-BLEU: for each sentence, compute BLEU against all other sentences.
    Higher = outputs are more similar to each other = more collapse.

    Uses a random subsample if len(texts) > sample_size for speed.
    """
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except ImportError:
        raise ImportError("Run: pip install nltk")

    smoother = SmoothingFunction().method1
    tokenized = [nltk.word_tokenize(t.lower()) for t in texts]

    # Subsample for speed on large sets
    rng = np.random.default_rng(42)
    if len(tokenized) > sample_size:
        indices = rng.choice(len(tokenized), sample_size, replace=False).tolist()
        tokenized = [tokenized[i] for i in indices]

    scores = []
    for i, hyp in enumerate(tqdm(tokenized, desc="    Self-BLEU", leave=False)):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        if not refs or not hyp:
            continue
        score = sentence_bleu(refs, hyp, smoothing_function=smoother)
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def compute_ttr(texts: list[str]) -> float:
    """
    Average per-sample Type-Token Ratio.
    TTR = unique_words / total_words, averaged across all samples.
    Lower = less vocabulary diversity per sample.
    """
    ratios = []
    for text in texts:
        tokens = text.lower().split()
        if not tokens:
            continue
        ratios.append(len(set(tokens)) / len(tokens))
    return float(np.mean(ratios)) if ratios else 0.0


def compute_unigram_entropy(texts: list[str]) -> float:
    """
    Shannon entropy of the global token frequency distribution.
    Lower = distribution is more peaked / less diverse.
    """
    from collections import Counter
    all_tokens = " ".join(texts).lower().split()
    assert len(all_tokens) > 0, "No tokens found for entropy computation"

    counts = Counter(all_tokens)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return float(entropy)


def compute_kl_divergence(
    current_texts: list[str],
    reference_texts: list[str],
) -> float:
    """
    KL divergence of current generation's unigram distribution
    from the reference distribution (generation 1).
    KL(current || reference). Higher = more drift from gen 1.
    """
    from collections import Counter

    def token_dist(texts):
        counts = Counter(" ".join(texts).lower().split())
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    p = token_dist(current_texts)   # current
    q = token_dist(reference_texts) # reference (gen 1)

    # Vocabulary union; unseen tokens get a small smoothing mass
    vocab = set(p) | set(q)
    eps = 1e-10
    kl = sum(
        p.get(w, eps) * math.log((p.get(w, eps)) / (q.get(w, eps) + eps) + eps)
        for w in vocab
    )
    return float(kl)


def compute_ngram_repetition(texts: list[str], n: int = 4) -> float:
    """Fraction of n-grams that are repeated across all texts."""
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        all_ngrams.extend(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    if not all_ngrams:
        return 0.0
    return 1.0 - len(set(all_ngrams)) / len(all_ngrams)


def compute_unique_token_ratio(texts: list[str]) -> float:
    """Global unique tokens / total tokens."""
    all_tokens = " ".join(texts).split()
    return len(set(all_tokens)) / max(len(all_tokens), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Per-generation evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_generation(
    gen_entry: dict,
    reference_sentences: list[str],
    device: torch.device,
    gen1_texts: list[str] | None
) -> dict:
    """Run all metrics for a single generation entry from the manifest."""
    gen_idx = gen_entry["generation"]
    ckpt_path = gen_entry["checkpoint_dir"]
    records_path = gen_entry["training_records"]

    logger.info(f"\n{'─' * 56}")
    logger.info(f"Generation {gen_idx}")
    logger.info(f"{'─' * 56}")

    # ── Load generated texts used to train this checkpoint ──
    assert records_path is not None and Path(records_path).exists(), \
        f"Records not found: {records_path}"
    with open(records_path) as f:
        records = [json.loads(line) for line in f]
    texts = [r["full_text"] for r in records]
    assert len(texts) > 0, f"No texts found in {records_path}"
    logger.info(f"  Loaded {len(texts)} training samples from records")

    # ── Load model ──
    model, tokenizer = load_checkpoint(ckpt_path, device)

    results: dict = {"generation": gen_idx}

    # 1. Held-out perplexity
    logger.info("  Computing held-out perplexity...")
    ppl = compute_perplexity(model, tokenizer, reference_sentences, device)
    results["perplexity"] = round(ppl, 4)
    logger.info(f"  Perplexity:          {ppl:.4f}")

    # Free model from GPU before CPU-bound metrics
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Self-BLEU
    logger.info("  Computing Self-BLEU...")
    sbleu = compute_selfbleu(texts)
    results["self_bleu"] = round(sbleu, 6)
    logger.info(f"  Self-BLEU:           {sbleu:.6f}")

    # 3. Type-Token Ratio
    ttr = compute_ttr(texts)
    results["ttr"] = round(ttr, 4)
    logger.info(f"  TTR:                 {ttr:.4f}")

    # 4. Unigram entropy
    entropy = compute_unigram_entropy(texts)
    results["unigram_entropy"] = round(entropy, 4)
    logger.info(f"  Unigram entropy:     {entropy:.4f}")

    # 5. KL divergence from generation 1
    if gen1_texts is not None and gen_idx > 1:
        kl = compute_kl_divergence(texts, gen1_texts)
        results["kl_from_gen1"] = round(kl, 6)
        logger.info(f"  KL from gen 1:       {kl:.6f}")
    else:
        results["kl_from_gen1"] = 0.0  # gen 1 is the reference; divergence = 0

    # 6. N-gram repetition (re-computed for full consistency)
    ngram_rep = compute_ngram_repetition(texts)
    results["ngram_repetition"] = round(ngram_rep, 4)
    logger.info(f"  N-gram repetition:   {ngram_rep:.4f}")

    # 7. Unique token ratio (re-computed)
    utr = compute_unique_token_ratio(texts)
    results["unique_token_ratio"] = round(utr, 4)
    logger.info(f"  Unique token ratio:  {utr:.4f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def print_summary_table(all_results: list[dict]) -> None:
    """Print a compact aligned table of all metrics across generations."""
    headers = [
        "Gen", "Perplexity", "Self-BLEU↑", "TTR", "Entropy", "KL↑", "N-gram Rep↑", "UTR↓"
    ]

    col_w = 13
    header_row = "  ".join(h.ljust(col_w) for h in headers)
    separator = "─" * len(header_row)

    print(f"\n{'=' * len(header_row)}")
    print("EVALUATION SUMMARY  (↑ = higher means more collapse,  ↓ = lower means more collapse)")
    print(separator)
    print(header_row)
    print(separator)

    for r in all_results:
        row = [
            str(r["generation"]).ljust(col_w),
            f"{r['perplexity']:.2f}".ljust(col_w),
            f"{r['ttr']:.4f}".ljust(col_w),
            f"{r['unigram_entropy']:.4f}".ljust(col_w),
            f"{r['kl_from_gen1']:.4f}".ljust(col_w),
            f"{r['ngram_repetition']:.4f}".ljust(col_w),
            f"{r['unique_token_ratio']:.4f}".ljust(col_w),
        ]

        sb = r.get("self_bleu")
        row.insert(2, (f"{sb:.4f}" if sb is not None else "skipped").ljust(col_w))
        print("  ".join(row))

    print(separator)


def save_results(all_results: list[dict], checkpoint_dir: str) -> None:
    out_path = Path(checkpoint_dir) / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n📊 Full results saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate self-consuming loop checkpoints for model collapse",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints",
        help="Root checkpoint directory containing manifest.json",
    )
    parser.add_argument(
        "--reference_corpus", default=None,
        help="Path to a .txt file of real sentences (one per line). "
             "Defaults to WikiText-2 validation split.",
    )
    parser.add_argument(
        "--num_reference_sentences", type=int, default=200,
        help="Number of reference sentences to use for perplexity",
    )
    parser.add_argument(
        "--generations", type=int, nargs="+", default=None,
        help="Evaluate only these generation indices (e.g. --generations 1 3 5). "
             "Default: all generations in manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load manifest ──
    manifest_path = Path(args.checkpoint_dir) / "manifest.json"
    assert manifest_path.exists(), (
        f"manifest.json not found in {args.checkpoint_dir}. "
        "Run the self-consuming loop first."
    )
    with open(manifest_path) as f:
        manifest = json.load(f)

    # gen_000 is the base model snapshot with no generated texts — skip it
    generations = [g for g in manifest["generations"] if g["generation"] > 0]
    if args.generations:
        generations = [g for g in generations if g["generation"] in args.generations]
        assert generations, "None of the requested generation indices found in manifest."

    logger.info(
        f"Evaluating {len(generations)} generation(s) from "
        f"{manifest['model_name']}"
    )

    # ── Load reference corpus once ──
    reference_sentences = load_reference_sentences(
        args.reference_corpus, n=args.num_reference_sentences
    )

    # ── Load gen-1 texts once (KL reference) ──
    gen1_entry = next((g for g in manifest["generations"] if g["generation"] == 1), None)
    gen1_texts: list[str] | None = None
    if gen1_entry:
        with open(gen1_entry["training_records"]) as f:
            gen1_texts = [json.loads(l)["full_text"] for l in f]

    # ── Evaluate each generation ──
    all_results: list[dict] = []
    for entry in generations:
        result = evaluate_generation(
            gen_entry=entry,
            reference_sentences=reference_sentences,
            device=device,
            gen1_texts=gen1_texts
        )
        all_results.append(result)

    # ── Print and save ──
    print_summary_table(all_results)
    save_results(all_results, args.checkpoint_dir)


if __name__ == "__main__":
    main()