"""
dataset.py — PyTorch Dataset wrapper for synthetic text used in fine-tuning,
             plus a helper to stream prompt seeds from The Pile.
"""

import logging
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def load_pile_prompts(
    dataset_name: str = "EleutherAI/pile",
    subset: str = "all",
    num_prompts: int = 1000,
    min_chars: int = 40,
    max_chars: int = 120,
    seed: int = 42,
) -> list[str]:
    """
    Stream The Pile and return a pool of short text snippets to use as prompts.

    Each snippet is the first sentence/phrase of a document, trimmed to a
    natural word boundary within [min_chars, max_chars].  Requires
    `datasets` ≥ 2.x and HuggingFace access to EleutherAI/pile.
    """
    from datasets import load_dataset

    logger.info(
        f"Streaming {num_prompts} prompts from {dataset_name} "
        f"(subset={subset!r}, min_chars={min_chars}, max_chars={max_chars})"
    )

    load_kwargs: dict = dict(split="train", streaming=True, trust_remote_code=True)
    if subset:
        load_kwargs["name"] = subset

    dataset = load_dataset(dataset_name, **load_kwargs)

    prompts: list[str] = []
    for example in dataset:
        text: str = (example.get("text") or "").strip()
        if len(text) < min_chars * 2:
            continue

        # Find a word boundary near max_chars
        candidate = text[: max_chars * 2]
        cut = candidate.rfind(" ", min_chars, max_chars)
        if cut == -1:
            cut = max_chars
        snippet = text[:cut].strip()

        if len(snippet) >= min_chars:
            prompts.append(snippet)

        if len(prompts) >= num_prompts:
            break

    logger.info(f"Loaded {len(prompts)} Pile prompts")
    return prompts


class SyntheticTextDataset(Dataset):
    """
    Tokenizes a list of strings and exposes them as (input_ids, attention_mask, labels)
    triples suitable for causal language model training.
    """

    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Sanity check: non-empty text list
        assert len(texts) > 0, "Cannot build dataset from empty text list"
        assert max_length > 0, f"max_length must be positive, got {max_length}"

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            # Labels are a copy of input_ids; the Trainer handles loss masking.
            "labels": self.encodings["input_ids"][idx].clone(),
        }