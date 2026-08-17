import argparse
import logging
import warnings

from config import LoopConfig
from loop import SelfConsumingLoop

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("self_consuming_loop.log"),
    ],
)
warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-Consuming Loop for Gemma-2-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name", default="google/gemma-2-2b",
        help="HuggingFace model ID or local path for the base model",
    )
    parser.add_argument(
        "--num_generations", type=int, default=10,
        help="Number of self-consuming loop iterations",
    )
    parser.add_argument(
        "--samples_per_generation", type=int, default=500,
        help="Synthetic samples to generate per iteration",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=200,
        help="Max tokens to generate per sample",
    )
    parser.add_argument(
        "--generation_batch_size", type=int, default=16,
        help="Number of prompts to generate in parallel (reduce if OOM)",
    )
    parser.add_argument(
        "--train_epochs", type=int, default=1,
        help="Fine-tuning epochs per generation",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5,
        help="Fine-tuning learning rate",
    )
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints",
        help="Root directory for saving generation checkpoints",
    )
    parser.add_argument(
        "--logs_dir", default="loop_logs",
        help="Directory for synthetic data and sanity reports",
    )
    parser.add_argument(
        "--resume_from", default=None,
        help=(
            "Resume from a local checkpoint dir OR a HuggingFace Hub path. "
            "Local:  checkpoints/gen_007  "
            "HF Hub: username/self-consuming-gemma@gen_007  "
            "The generation number is auto-detected from the path/branch name."
        ),
    )
    parser.add_argument(
        "--resume_generation", type=int, default=None,
        help=(
            "Override auto-detected start generation (last gen already done). "
            "Loop runs from resume_generation+1 to num_generations."
        ),
    )

    # ── Pile prompts ──────────────────────────────────────────
    parser.add_argument(
        "--use_pile_prompts", action="store_true",
        help="Draw seed prompts from The Pile instead of static strings",
    )
    parser.add_argument(
        "--pile_subset", default="",
        help="Pile subset/config name passed to datasets.load_dataset. Leave empty for datasets with no subsets (e.g. monology/pile-uncopyrighted).",
    )
    parser.add_argument(
        "--pile_prompt_pool_size", type=int, default=1000,
        help="Number of Pile prompts to cache at loop start",
    )

    # ── HuggingFace Hub ───────────────────────────────────────
    parser.add_argument(
        "--hf_repo_id", default=None,
        help="HF Hub repo ID to push checkpoints to, e.g. 'username/self-consuming-gemma'",
    )
    parser.add_argument(
        "--no_hf_push", action="store_true",
        help="Disable automatic Hub push even when --hf_repo_id is set",
    )
    parser.add_argument(
        "--hf_only", action="store_true",
        help=(
            "Use HF Hub as the sole checkpoint store. "
            "Each generation loads its starting weights from the previously "
            "pushed Hub branch instead of a local directory. "
            "On first run, auto-resumes from the latest gen_NNN branch found on Hub."
        ),
    )
    parser.add_argument(
        "--hf_delete_local", action="store_true",
        help="Delete the local checkpoint directory after each successful Hub push.",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    cfg = LoopConfig(
        model_name=args.model_name,
        num_generations=args.num_generations,
        samples_per_generation=args.samples_per_generation,
        max_new_tokens=args.max_new_tokens,
        generation_batch_size=args.generation_batch_size,
        train_epochs=args.train_epochs,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        logs_dir=args.logs_dir,
        # Pile
        use_pile_prompts=args.use_pile_prompts,
        pile_subset=args.pile_subset,
        pile_prompt_pool_size=args.pile_prompt_pool_size,
        # HuggingFace Hub
        hf_repo_id=args.hf_repo_id,
        hf_push_each_gen=not args.no_hf_push,
        hf_only=args.hf_only,
        hf_delete_local_after_push=args.hf_delete_local,
    )

    loop = SelfConsumingLoop(cfg)
    loss_history = loop.run(
        start_model_path=args.resume_from,
        start_generation=args.resume_generation,
    )

    print("\nFinal loss trajectory:")
    for i, loss in enumerate(loss_history, start=1):
        print(f"  Generation {i:>2}: {loss:.4f}")


if __name__ == "__main__":
    main()