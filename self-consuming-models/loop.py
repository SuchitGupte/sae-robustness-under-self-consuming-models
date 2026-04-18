"""
loop.py — Core self-consuming loop: load → generate → fine-tune → checkpoint → repeat.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

from config import LoopConfig
from dataset import SyntheticTextDataset
from sanity import SanityChecker

logger = logging.getLogger(__name__)


class SelfConsumingLoop:
    """
    Orchestrates the full self-consuming loop:
      1. Load the current model (base or previous checkpoint)
      2. Generate synthetic text using the model
      3. Fine-tune the model on that synthetic text
      4. Save a checkpoint
      5. Repeat for num_generations iterations
    """

    def __init__(self, cfg: LoopConfig):
        self.cfg = cfg
        self.checker = SanityChecker(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.logs_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Using device: {self.device}")

    # ──────────────────────────────────────────────────────────
    # Step 1 — Load model
    # ──────────────────────────────────────────────────────────

    def load_model(self, model_path: str):
        """Load tokenizer and model from a HuggingFace path or local checkpoint."""
        logger.info(f"Loading model from: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        load_kwargs: dict = dict(
            device_map=self.cfg.device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="eager",
        )

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        model.eval()

        self.checker.check_model_loaded(model, tokenizer)
        return model, tokenizer

    # ──────────────────────────────────────────────────────────
    # Step 2 — Generate synthetic data
    # ──────────────────────────────────────────────────────────

    def generate_synthetic_data(
        self,
        model,
        tokenizer,
        generation_idx: int,
    ) -> list[str]:
        """Generate `samples_per_generation` texts from the current model."""
        logger.info(
            f"Generating {self.cfg.samples_per_generation} samples "
            f"for generation {generation_idx}..."
        )
        model.eval()

        gen_config = GenerationConfig(
            max_new_tokens=self.cfg.max_new_tokens,
            min_new_tokens=self.cfg.min_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Cycle seed prompts to fill the requested sample count
        prompts = (
            self.cfg.seed_prompts
            * (self.cfg.samples_per_generation // len(self.cfg.seed_prompts) + 1)
        )[: self.cfg.samples_per_generation]

        texts: list[str] = []
        records: list[dict] = []   # rich per-sample records for SAE use

        # Left-padding is required for batched causal LM generation
        tokenizer.padding_side = "left"
        bs = self.cfg.generation_batch_size

        for batch_start in tqdm(
            range(0, len(prompts), bs),
            desc=f"Gen {generation_idx} — generating",
        ):
            batch_prompts = prompts[batch_start : batch_start + bs]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True,
            ).to(self.device)
            # input_length is the padded sequence length (same for all in batch)
            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=gen_config)

            for j, prompt in enumerate(batch_prompts):
                idx = batch_start + j
                # Real prompt tokens (strip left-padding)
                prompt_ids     = inputs["input_ids"][j][inputs["attention_mask"][j].bool()].tolist()
                # Everything the model generated after the (padded) input
                completion_ids = output_ids[j][input_length:].tolist()

                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                full_text = f"{prompt} {completion_text.strip()}"
                texts.append(full_text)

                records.append({
                    "sample_id":              f"gen{generation_idx:03d}_{idx:04d}",
                    "generation":             generation_idx,
                    "prompt":                 prompt,
                    "completion":             completion_text.strip(),
                    "full_text":              full_text,
                    "prompt_token_ids":       prompt_ids,
                    "completion_token_ids":   completion_ids,
                    "full_token_ids":         prompt_ids + completion_ids,
                    "prompt_len":             len(prompt_ids),
                    "completion_len":         len(completion_ids),
                    "total_len":              len(prompt_ids) + len(completion_ids),
                })

        # Restore right-padding for training
        tokenizer.padding_side = "right"

        # Sanity-check generated data quality
        self.checker.check_generation_quality(texts, generation_idx)

        # ── Persist flat text list (backwards-compatible) ──
        out_path = Path(self.cfg.logs_dir) / f"gen_{generation_idx:03d}_data.json"
        with open(out_path, "w") as f:
            json.dump(texts, f, indent=2)

        # ── Persist rich records (SAE-ready) to logs dir ──
        rich_path = Path(self.cfg.logs_dir) / f"gen_{generation_idx:03d}_records.jsonl"
        with open(rich_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        logger.info(
            f"Saved synthetic data → {out_path} "
            f"| SAE records → {rich_path}"
        )
        # Store records on self so fine_tune() can co-locate them with the checkpoint
        self._last_records = records
        return texts

    # ──────────────────────────────────────────────────────────
    # Step 3 — Fine-tune & checkpoint
    # ──────────────────────────────────────────────────────────

    def fine_tune(
        self,
        model,
        tokenizer,
        texts: list[str],
        generation_idx: int,
    ) -> float:
        """Fine-tune `model` on `texts`, save a checkpoint, and return the final loss."""
        logger.info(
            f"Fine-tuning on {len(texts)} samples (generation {generation_idx})..."
        )

        dataset = SyntheticTextDataset(texts, tokenizer, self.cfg.max_seq_length)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        ckpt_path = os.path.join(self.cfg.checkpoint_dir, f"gen_{generation_idx:03d}")

        training_args = TrainingArguments(
            output_dir=ckpt_path,
            num_train_epochs=self.cfg.train_epochs,
            per_device_train_batch_size=self.cfg.train_batch_size,
            learning_rate=self.cfg.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            bf16=torch.cuda.is_available(),
            report_to="none",
            seed=42 + generation_idx,
            dataloader_drop_last=False,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        train_result = trainer.train()
        final_loss = train_result.training_loss

        # Sanity-check the loss value
        self.checker.check_training_loss(final_loss, generation_idx)

        # Persist model + tokenizer
        trainer.save_model(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)

        # Verify the checkpoint was written correctly
        self.checker.check_checkpoint_saved(ckpt_path)

        # Write loop metadata alongside the checkpoint
        records = getattr(self, "_last_records", [])
        meta = {
            "generation":        generation_idx,
            "timestamp":         datetime.now().isoformat(),
            "num_samples":       len(texts),
            "training_loss":     final_loss,
            "epochs":            self.cfg.train_epochs,
            "checkpoint_path":   ckpt_path,
            "data_files": {
                "texts_json":    str(Path(self.cfg.logs_dir) / f"gen_{generation_idx:03d}_data.json"),
                "records_jsonl": str(Path(self.cfg.logs_dir) / f"gen_{generation_idx:03d}_records.jsonl"),
                "ckpt_records_jsonl": str(Path(ckpt_path) / "training_records.jsonl"),
            },
        }
        with open(os.path.join(ckpt_path, "loop_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Co-locate a copy of the rich records inside the checkpoint folder
        # so the checkpoint is fully self-contained for SAE experiments
        ckpt_records_path = Path(ckpt_path) / "training_records.jsonl"
        with open(ckpt_records_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        logger.info(f"  SAE records co-located → {ckpt_records_path}")

        logger.info(f"Checkpoint saved at: {ckpt_path} | Loss: {final_loss:.4f}")
        return final_loss

    # ──────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────

    def run(self, start_model_path: Optional[str] = None) -> list[float]:
        """Run the full self-consuming loop and return the per-generation loss history."""
        model_path = start_model_path or self.cfg.model_name
        loss_history: list[float] = []

        logger.info("=" * 60)
        logger.info("Starting Self-Consuming Loop")
        logger.info(f"  Base model      : {self.cfg.model_name}")
        logger.info(f"  Generations     : {self.cfg.num_generations}")
        logger.info(f"  Samples/gen     : {self.cfg.samples_per_generation}")
        logger.info(f"  Checkpoint dir  : {self.cfg.checkpoint_dir}")
        logger.info("=" * 60)

        # Save M0 (base model) as gen_000 before any training starts
        # Always use self.cfg.model_name (not model_path) so --resume_from
        # doesn't accidentally overwrite gen_000 with a later checkpoint.
        m0_path = os.path.join(self.cfg.checkpoint_dir, "gen_000")
        if not Path(m0_path).exists():
            logger.info(f"Saving base model (M0) → {m0_path}")
            _model, _tok = self.load_model(self.cfg.model_name)
            _model.save_pretrained(m0_path)
            _tok.save_pretrained(m0_path)
            del _model, _tok
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logger.info(f"Base model checkpoint already exists, skipping: {m0_path}")

        for gen in range(1, self.cfg.num_generations + 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"GENERATION {gen} / {self.cfg.num_generations}")
            logger.info(f"{'=' * 60}")

            model, tokenizer = self.load_model(model_path)
            texts = self.generate_synthetic_data(model, tokenizer, gen)
            loss = self.fine_tune(model, tokenizer, texts, gen)
            loss_history.append(loss)

            # Point to the freshly saved checkpoint for the next iteration
            model_path = os.path.join(self.cfg.checkpoint_dir, f"gen_{gen:03d}")

            # Free GPU memory between generations
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Final summary ──
        logger.info("\n" + "=" * 60)
        logger.info("SELF-CONSUMING LOOP COMPLETE")
        logger.info(f"  Loss trajectory: {[round(l, 4) for l in loss_history]}")
        if len(loss_history) > 1:
            delta = loss_history[-1] - loss_history[0]
            direction = "↑ increased" if delta > 0 else "↓ decreased"
            logger.info(f"  Total loss shift: {delta:+.4f} ({direction})")
        logger.info("=" * 60)

        self.checker.save_report(self.cfg.logs_dir)
        self._save_dataset_manifest(loss_history, m0_path)
        return loss_history

    def _save_dataset_manifest(self, loss_history: list[float], m0_path: str) -> None:
        """
        Write a single manifest.json at the checkpoint root that lists every
        generation's checkpoint path, data files, and loss — making it easy
        to load all SAE training data in one shot later.
        """
        # gen_000 = base model, no synthetic data
        entries = [{
            "generation":       0,
            "training_loss":    None,
            "checkpoint_dir":   m0_path,
            "training_records": None,
            "loop_meta":        None,
            "texts_json":       None,
            "records_jsonl":    None,
        }]

        for gen in range(1, len(loss_history) + 1):
            ckpt_path = os.path.join(self.cfg.checkpoint_dir, f"gen_{gen:03d}")
            entries.append({
                "generation":        gen,
                "training_loss":     loss_history[gen - 1],
                "checkpoint_dir":    ckpt_path,
                "training_records":  str(Path(ckpt_path) / "training_records.jsonl"),
                "loop_meta":         str(Path(ckpt_path) / "loop_meta.json"),
                "texts_json":        str(Path(self.cfg.logs_dir) / f"gen_{gen:03d}_data.json"),
                "records_jsonl":     str(Path(self.cfg.logs_dir) / f"gen_{gen:03d}_records.jsonl"),
            })

        manifest = {
            "created_at":      datetime.now().isoformat(),
            "model_name":      self.cfg.model_name,
            "num_generations": len(loss_history),
            "loss_trajectory": loss_history,
            "generations":     entries,
        }

        manifest_path = Path(self.cfg.checkpoint_dir) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"📦 Dataset manifest saved → {manifest_path}")