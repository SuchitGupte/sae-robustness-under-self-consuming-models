"""
sanity.py — Quality and collapse-detection checks for each loop generation.
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from config import LoopConfig

logger = logging.getLogger(__name__)


class SanityChecker:
    """
    Runs a suite of checks after each generation and accumulates results
    so cross-generation trends (e.g. diversity collapse) can be detected.
    """

    def __init__(self, cfg: LoopConfig):
        self.cfg = cfg
        self.history: list[dict] = []


    def check_model_loaded(self, model, tokenizer) -> None:
        """Verify model and tokenizer loaded correctly."""
        assert model is not None, "Model failed to load"
        assert tokenizer is not None, "Tokenizer failed to load"
        assert tokenizer.pad_token is not None, "Tokenizer missing pad token"

        vocab_size = tokenizer.vocab_size
        assert vocab_size > 1000, f"Suspiciously small vocab: {vocab_size}"

        logger.info(f"Model sanity check passed | Vocab size: {vocab_size}")

    def check_generation_quality(self, texts: list[str], generation: int) -> dict:
        """
        Run multiple quality checks on the generated texts and return a
        results dict. Warnings and errors are logged; errors are also stored
        in results['errors'] for the caller to act on.
        """
        results: dict = {
            "generation": generation,
            "num_samples": len(texts),
            "warnings": [],
            "errors": [],
        }

        # Check 1 — Minimum sample count
        if len(texts) < 5:
            results["errors"].append(f"Too few samples generated: {len(texts)}")

        # Check 2 — Empty / very short outputs
        short = [t for t in texts if len(t.split()) < 10]
        if len(short) > len(texts) * 0.2:
            results["warnings"].append(
                f"{len(short)}/{len(texts)} samples are very short (<10 words)"
            )

        # Check 3 — Lexical diversity (unique tokens / total tokens)
        all_tokens = " ".join(texts).split()
        unique_ratio = len(set(all_tokens)) / max(len(all_tokens), 1)
        results["unique_token_ratio"] = round(unique_ratio, 4)
        if unique_ratio < self.cfg.diversity_threshold:
            results["warnings"].append(
                f"Low lexical diversity: {unique_ratio:.3f} "
                f"(threshold: {self.cfg.diversity_threshold})"
            )

        # Check 4 — Repeated 4-gram ratio
        ngram_repeats = self._ngram_repetition(texts, n=4)
        results["ngram_repetition"] = round(ngram_repeats, 4)
        if ngram_repeats > self.cfg.repetition_threshold:
            results["warnings"].append(
                f"High n-gram repetition: {ngram_repeats:.3f} "
                f"(threshold: {self.cfg.repetition_threshold})"
            )

        # Check 5 — Exact duplicate detection
        hashes = [hashlib.md5(t.encode()).hexdigest() for t in texts]
        dup_count = len(hashes) - len(set(hashes))
        results["exact_duplicates"] = dup_count
        if dup_count > len(texts) * 0.1:
            results["warnings"].append(
                f"High exact duplicates: {dup_count}/{len(texts)}"
            )

        # Check 6 — Cross-generation collapse detection
        if self.history:
            prev_ratio = self.history[-1].get("unique_token_ratio", 1.0)
            if unique_ratio < prev_ratio * 0.7:
                results["warnings"].append(
                    "Diversity dropped >30% from last generation — possible collapse"
                )

        self.history.append(results)
        self._log_results(results)
        return results

    def check_checkpoint_saved(self, checkpoint_path: str) -> None:
        """Verify checkpoint files exist and are non-empty."""
        path = Path(checkpoint_path)
        assert path.exists(), f"Checkpoint dir not found: {checkpoint_path}"

        files = list(path.iterdir())
        assert len(files) > 0, f"Checkpoint dir is empty: {checkpoint_path}"

        names = [f.name for f in files]
        assert any("config" in n for n in names), (
            f"No config file in checkpoint: {names}"
        )

        total_size_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
        assert total_size_mb > 1, (
            f"Checkpoint suspiciously small: {total_size_mb:.1f} MB"
        )

        logger.info(
            f"Checkpoint saved | Path: {checkpoint_path} | "
            f"Files: {len(files)} | Size: {total_size_mb:.1f} MB"
        )

    def check_training_loss(self, loss: float, generation: int) -> None:
        """Flag abnormal training loss values."""
        if np.isnan(loss):
            raise ValueError(f"Training loss is NaN at generation {generation}!")
        if np.isinf(loss):
            raise ValueError(f"Training loss is Inf at generation {generation}!")
        if loss > 20:
            logger.warning(f"Very high training loss: {loss:.4f}")
        elif loss < 0.001:
            logger.warning(
                f"Suspiciously low loss (possible overfitting): {loss:.4f}"
            )
        else:
            logger.info(f"Training loss OK: {loss:.4f}")

    def save_report(self, logs_dir: str) -> None:
        """Persist the full sanity-check history to a JSON file."""
        path = Path(logs_dir) / "sanity_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Sanity report saved to {path}")

    @staticmethod
    def _ngram_repetition(texts: list[str], n: int = 4) -> float:
        """Return the fraction of n-grams that are repeated across all texts."""
        all_ngrams: list[tuple] = []
        for text in texts:
            tokens = text.split()
            all_ngrams.extend(
                tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
            )
        if not all_ngrams:
            return 0.0
        return 1.0 - len(set(all_ngrams)) / len(all_ngrams)

    @staticmethod
    def _log_results(results: dict) -> None:
        gen = results["generation"]
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Generation {gen} — Quality Report")
        logger.info(f"  Samples         : {results['num_samples']}")
        logger.info(f"  Unique tok ratio: {results.get('unique_token_ratio', 'N/A')}")
        logger.info(f"  N-gram repeat   : {results.get('ngram_repetition', 'N/A')}")
        logger.info(f"  Exact duplicates: {results.get('exact_duplicates', 'N/A')}")
        for w in results["warnings"]:
            logger.warning(f"Warning: {w}")
        for e in results["errors"]:
            logger.error(f"Error: {e}")
        if not results["warnings"] and not results["errors"]:
            logger.info("All checks passed")
        logger.info(f"{'─' * 50}\n")