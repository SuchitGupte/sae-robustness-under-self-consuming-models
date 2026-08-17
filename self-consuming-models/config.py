"""
config.py — Configuration dataclass for the self-consuming loop.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoopConfig:
    model_name: str = "google/gemma-2-2b"
    seed_prompts: list = field(default_factory=lambda: [
        "The history of artificial intelligence began",
        "Climate change affects global ecosystems by",
        "The principles of quantum mechanics state that",
        "Modern democracy is built on the foundation of",
        "The human brain processes information through",
    ])
    num_generations: int = 5           # Number of self-consuming loop iterations
    samples_per_generation: int = 50   # Synthetic samples to generate per loop
    max_new_tokens: int = 200          # Tokens per generated sample
    min_new_tokens: int = 50
    generation_batch_size: int = 8     # Prompts to generate in parallel
    train_epochs: int = 1              # Fine-tuning epochs per generation
    train_batch_size: int = 2
    learning_rate: float = 2e-5
    max_seq_length: int = 512
    checkpoint_dir: str = "checkpoints"
    logs_dir: str = "loop_logs"
    load_in_4bit: bool = True          # Use QLoRA-style 4-bit quantization
    device: str = "auto"
    # Collapse detection thresholds
    diversity_threshold: float = 0.3   # Unique token ratio below this → warning
    repetition_threshold: float = 0.5  # Repeated n-gram ratio above this → warning
    perplexity_spike_factor: float = 3.0  # PPL increased by this factor → warning

    # ── Pile prompt settings ──────────────────────────────────
    use_pile_prompts: bool = False
    pile_dataset_name: str = "monology/pile-uncopyrighted"
    pile_subset: str = ""       # empty → no subset (monology/pile-uncopyrighted has none)
    pile_prompt_pool_size: int = 1000  # Prompts to cache at loop start
    pile_prompt_min_chars: int = 40    # Min chars for a valid prompt snippet
    pile_prompt_max_chars: int = 120   # Max chars to keep as prompt prefix

    # ── HuggingFace Hub settings ──────────────────────────────
    hf_repo_id: Optional[str] = None   # e.g. "username/self-consuming-gemma"
    hf_token: Optional[str] = None     # HF write token (or set HF_TOKEN env var)
    hf_push_each_gen: bool = True      # Push after every generation
    hf_only: bool = False              # Use HF Hub as sole checkpoint store:
                                       #   load next gen from Hub, not local dir
    hf_delete_local_after_push: bool = False  # Delete local ckpt after Hub push
