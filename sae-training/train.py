"""
Train SAEs on a single layer of a self-consuming model checkpoint.

Usage
─────
    python train.py --gen_label gen_000 --layer 0
    python train.py --gen_label gen_000 --layer 0 --device 1 --no_wandb
"""

import argparse
import gc
import logging

import torch as t
import sae_lens
from sae_lens import (
    LanguageModelSAERunnerConfig,
    SAETrainingRunner,
    JumpReLUTrainingSAEConfig,
    LoggingConfig,
    upload_saes_to_huggingface,
    SAE,
)
from transformers import AutoModelForCausalLM

logging.getLogger("sae_lens").setLevel(logging.CRITICAL)

CHECKPOINT_ROOT = "/research/nfs_khalili_17/gupte.31/selfconsuming/sae-robustness-under-self-consuming-models/self-consuming-models/checkpoints"
SAE_OUTPUT_ROOT = "/research/nfs_khalili_17/gupte.31/selfconsuming/sae-robustness-under-self-consuming-models/sae-training/saes"
HF_REPO_ID      = "suchitg/selfconsuming-saes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_label", type=str, required=True)
    parser.add_argument("--layer",     type=int, required=True)
    parser.add_argument("--device",    type=int, default=0)
    parser.add_argument("--no_wandb",  action="store_true")
    parser.add_argument("--wandb_project", type=str, default="selfconsuming-saes")
    parser.add_argument("--training_tokens", type=int, default=int(4e7))
    parser.add_argument("--hf_repo_id", type=str, default=HF_REPO_ID)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = t.device(f"cuda:{args.device}" if t.cuda.is_available() else "cpu")
    model_path = f"{CHECKPOINT_ROOT}/{args.gen_label}"

    print(f"Device     : {device}")
    print(f"Generation : {args.gen_label}")
    print(f"Model path : {model_path}")
    print(f"Layer      : {args.layer}")
    print()
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=t.bfloat16)
    model = sae_lens.HookedSAETransformer.from_pretrained("gemma-2-2b", hf_model=hf_model, device=device)
    del hf_model

    # ── Gemma 2 2B model dimensions ──────────────────────────────
    # d_model = 2304, 26 layers, MLP intermediate = 9216 (post-RMSNorm)
    # attention concat output = num_heads * head_dim = 8 * 256 = 2048

    LAYER = args.layer  # change per run (0–25)
    SITE  = "resid_post"  # "resid_post" | "hook_mlp_out" | "hook_z" (attn pre-linear)

    D_IN_BY_SITE = {
        "resid_post":  2304,   # residual stream
        "hook_mlp_out": 2304,  # MLP output (after RMSNorm)
        "hook_z":       2048,  # attn heads concat (8 heads × 256)
    }
    HOOK_BY_SITE = {
        "resid_post":   f"blocks.{LAYER}.hook_resid_post",
        "hook_mlp_out": f"blocks.{LAYER}.hook_mlp_out",
        "hook_z":       f"blocks.{LAYER}.attn.hook_z",
    }

    D_IN   = D_IN_BY_SITE[SITE]
    D_SAE  = 16_384   # 16K width (paper also trains 32K, 65K, 131K, 1M)

    # ── Exact Gemma Scope hyperparameters (§3.2 of the paper) ────
    # lr = 7e-5, cosine warmup from 0.1*lr over 1000 steps
    # Adam: β1=0, β2=0.999, ε=1e-8
    # batch_size = 4096
    # λ (l0_coefficient) swept per desired L0; paper uses λ ∈ {6e-4, 1e-3, 2e-3, ...}
    # bandwidth ε = 0.001 (when inputs normalized to unit MSN)
    # sparsity warmup: 0 → λ over first 10,000 steps
    # 16K SAEs trained for 4B tokens; all other widths for 8B tokens
    # normalize_activations = "expected_average_only_in" (unit mean-squared norm)
    # init threshold θ = 0.001, Wdec = He-uniform (unit norm cols), Wenc = Wdec^T

    # TRAINING_TOKENS = 4_000_000_000   # 4B for 16K-width; use 8B for wider SAEs
    TRAINING_TOKENS = args.training_tokens
    BATCH_SIZE      = 4096
    TOTAL_STEPS     = TRAINING_TOKENS // BATCH_SIZE   # ~976,562

    print(f"{'='*60}")
    print(f"  {args.gen_label} | layer {args.layer} ")
    print(f"{'='*60}")

    cfg = LanguageModelSAERunnerConfig(
        # ── SAE architecture ─────────────────────────────────────
        sae=JumpReLUTrainingSAEConfig(
            d_in=D_IN,
            d_sae=D_SAE,
            apply_b_dec_to_input=True,        # pre-encoder bias (folded in after training)
            normalize_activations="expected_average_only_in",  # unit MSN normalization

            # JumpReLU-specific (paper §3.2)
            l0_coefficient=1e-3,              # λ; sweep this to hit desired L0
            l0_warm_up_steps=10_000,          # linear warmup 0 → λ over 10k steps
            jumprelu_init_threshold=0.001,    # θ₀ = 0.001
            jumprelu_bandwidth=0.001,         # ε = 0.001 (unit-norm input assumption)
        ),

        # ── Model + data ─────────────────────────────────────────
        model_name="gemma-2-2b",
        model_class_name="HookedTransformer",
        hook_name=HOOK_BY_SITE[SITE],
        dataset_path="monology/pile-uncopyrighted",     # Gemma 1 pretraining distribution
        dataset_trust_remote_code=True,
        is_dataset_tokenized=False,
        streaming=True,
        context_size=1024,
        prepend_bos=True,

        # ── Optimizer (exact paper values) ───────────────────────
        lr=7e-5,
        lr_warm_up_steps=1_000,              # cosine warmup from 0.1*lr → lr
        adam_beta1=0.0,                      # β1 = 0 (paper §3.2)
        adam_beta2=0.999,
        # ε = 1e-8 is the SAELens default

        # ── Training volume ──────────────────────────────────────
        train_batch_size_tokens=BATCH_SIZE,
        training_tokens=TRAINING_TOKENS,
        store_batch_size_prompts=32,
        n_batches_in_buffer=64,

        # ── Logging ──────────────────────────────────────────────
        logger=LoggingConfig(
            run_name=f"{args.gen_label}.{HOOK_BY_SITE[SITE]}.4e7_tokens",
            log_to_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_log_frequency=30,
            eval_every_n_wandb_logs=20,
        ),
        n_eval_batches=2,
        eval_batch_size_prompts=4,

        # ── Misc ─────────────────────────────────────────────────
        device=str(device),
        dtype="bfloat16",   # paper uses 32-bit precision throughout
        seed=42,
        n_checkpoints=0,
        checkpoint_path=f"{SAE_OUTPUT_ROOT}/{args.gen_label}/{HOOK_BY_SITE[SITE]}",
        act_store_device="cpu",
    )

    t.set_grad_enabled(True)
    runner = SAETrainingRunner(cfg, override_model=model)
    sae = runner.run()

    sae_id = f"{args.gen_label}/{HOOK_BY_SITE[SITE]}"
    upload_saes_to_huggingface({sae_id: sae}, hf_repo_id=args.hf_repo_id)
    print(f"  Uploaded → {args.hf_repo_id}/{sae_id}")

    del runner, sae
    t.cuda.empty_cache()
    gc.collect()

    print("\nDone.")
