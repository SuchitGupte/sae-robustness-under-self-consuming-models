"""
dataset.py — PyTorch Dataset wrapper for synthetic text used in fine-tuning.
"""

import torch
from torch.utils.data import Dataset


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