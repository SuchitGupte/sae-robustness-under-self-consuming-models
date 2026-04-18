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
        help="Path to a checkpoint directory to resume training from",
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
    )

    loop = SelfConsumingLoop(cfg)
    loss_history = loop.run(start_model_path=args.resume_from)

    print("\nFinal loss trajectory:")
    for i, loss in enumerate(loss_history, start=1):
        print(f"  Generation {i:>2}: {loss:.4f}")


if __name__ == "__main__":
    main()