"""
loop.py — Core self-consuming loop: load → generate → fine-tune → checkpoint → repeat.
"""

import json
import logging
import os
import re
import random
import shutil
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
from dataset import SyntheticTextDataset, load_pile_prompts
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

        # Pre-load Pile prompt pool once so every generation draws from it
        self._pile_prompts: list[str] = []
        if cfg.use_pile_prompts:
            self._pile_prompts = load_pile_prompts(
                dataset_name=cfg.pile_dataset_name,
                subset=cfg.pile_subset,
                num_prompts=cfg.pile_prompt_pool_size,
                min_chars=cfg.pile_prompt_min_chars,
                max_chars=cfg.pile_prompt_max_chars,
            )
            if not self._pile_prompts:
                logger.warning(
                    "Pile prompt pool is empty — falling back to seed_prompts. "
                    "Check your HF access token and dataset name."
                )

    # ──────────────────────────────────────────────────────────
    # Step 1 — Load model
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_resume_path(path: str) -> tuple[str, Optional[str], int]:
        """
        Parse a resume path into (model_path, revision, start_generation).

        Supported formats
        -----------------
        "checkpoints/gen_007"          → local path, resumes after gen 7
        "username/model@gen_007"       → HF Hub branch gen_007, resumes after gen 7
        "username/model@some-branch"   → HF Hub named branch, start_gen inferred if possible
        "username/model"               → HF Hub main branch, start_gen = 0
        """
        revision: Optional[str] = None

        # Split @revision only when the left side is not an existing local path
        if "@" in path and not os.path.exists(path):
            path, revision = path.rsplit("@", 1)

        # Auto-detect generation number from revision or the last path component
        gen_source = revision if revision else os.path.basename(path.rstrip("/"))
        m = re.search(r"gen[_-](\d+)", gen_source, re.IGNORECASE)
        start_gen = int(m.group(1)) if m else 0

        return path, revision, start_gen

    def load_model(self, model_path: str):
        """Load tokenizer and model from a HuggingFace path or local checkpoint.

        Accepts an optional ``@revision`` suffix for HF Hub branches, e.g.
        ``"username/model@gen_007"`` loads the gen_007 branch.
        """
        revision: Optional[str] = None
        if "@" in model_path and not os.path.exists(model_path):
            model_path, revision = model_path.rsplit("@", 1)

        logger.info(
            f"Loading model from: {model_path}"
            + (f"  (revision: {revision})" if revision else "")
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        load_kwargs: dict = dict(
            device_map=self.cfg.device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="eager",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=revision, **load_kwargs
        )
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

        # Build prompt list for this generation
        n = self.cfg.samples_per_generation
        if self.cfg.use_pile_prompts and self._pile_prompts:
            # Sample with replacement so we always get exactly n prompts
            prompts = random.choices(self._pile_prompts, k=n)
        else:
            # Cycle the static seed prompts
            pool = self.cfg.seed_prompts
            prompts = (pool * (n // len(pool) + 1))[:n]

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
            # With device_map="auto" the model is sharded across GPUs; move inputs
            # to whichever device holds the embedding layer (first parameter).
            input_device = next(model.parameters()).device
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True,
            ).to(input_device)
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
    # Step 4 — HuggingFace Hub helpers
    # ──────────────────────────────────────────────────────────

    def _latest_hub_generation(self) -> int:
        """Return the highest gen_NNN branch already on HF Hub, or 0."""
        if not self.cfg.hf_repo_id:
            return 0
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.cfg.hf_token or None)
            refs = api.list_repo_refs(self.cfg.hf_repo_id, repo_type="model")
            gen_nums = []
            for branch in refs.branches:
                m = re.match(r"gen[_-](\d+)$", branch.name, re.IGNORECASE)
                if m:
                    gen_nums.append(int(m.group(1)))
            latest = max(gen_nums) if gen_nums else 0
            logger.info(f"HF Hub latest generation found: {latest}")
            return latest
        except Exception as exc:
            logger.warning(f"Could not query HF Hub for existing generations: {exc}")
            return 0

    def push_checkpoint_to_hub(self, ckpt_path: str, generation_idx: int) -> None:
        """Upload a local checkpoint directory to HF Hub on a per-generation branch."""
        if not self.cfg.hf_repo_id:
            return

        token = self.cfg.hf_token or None  # None → uses cached huggingface-cli login

        try:
            from huggingface_hub import HfApi
        except ImportError:
            logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
            return

        api = HfApi(token=token)
        repo_id = self.cfg.hf_repo_id
        branch = f"gen_{generation_idx:03d}"

        try:
            api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
        except Exception as exc:
            logger.error(f"Could not create/access HF repo {repo_id!r}: {exc}")
            return

        try:
            api.create_branch(repo_id=repo_id, branch=branch, exist_ok=True)
        except Exception as exc:
            logger.warning(f"Could not create branch {branch!r}: {exc}")

        try:
            api.upload_folder(
                folder_path=ckpt_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Self-consuming loop: generation {generation_idx}",
                revision=branch,
            )
            logger.info(
                f"Pushed gen_{generation_idx:03d} → "
                f"https://huggingface.co/{repo_id}/tree/{branch}"
            )
        except Exception as exc:
            logger.error(f"HF Hub upload failed for gen_{generation_idx:03d}: {exc}")

    # ──────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────

    def run(
        self,
        start_model_path: Optional[str] = None,
        start_generation: Optional[int] = None,
    ) -> list[float]:
        """Run the self-consuming loop, optionally resuming from a checkpoint.

        Parameters
        ----------
        start_model_path:
            Local checkpoint dir **or** HF Hub ``repo_id[@revision]``.
            Examples::

                "checkpoints/gen_007"
                "username/self-consuming-gemma@gen_007"
                "username/self-consuming-gemma"   # loads main branch
        start_generation:
            Generation index already completed.  The loop will run from
            ``start_generation + 1`` to ``num_generations``.
            When *None*, inferred from the path (gen_NNN pattern) or 0.
        """
        # ── Resolve resume path and starting generation ──────────
        #
        # Priority order:
        #  1. Explicit start_model_path (user-supplied)
        #  2. hf_only=True → query HF Hub for the latest pushed gen
        #  3. Fall back to the configured base model (fresh start)
        if start_model_path:
            resolved_path, revision, detected_gen = self._parse_resume_path(
                start_model_path
            )
            if start_generation is None:
                start_generation = detected_gen
            model_path = (
                f"{resolved_path}@{revision}" if revision else resolved_path
            )
        elif self.cfg.hf_only and self.cfg.hf_repo_id:
            latest = self._latest_hub_generation()
            if latest > 0:
                if start_generation is None:
                    start_generation = latest
                model_path = f"{self.cfg.hf_repo_id}@gen_{latest:03d}"
                logger.info(
                    f"hf_only: resuming from HF Hub gen_{latest:03d}"
                )
            else:
                model_path = self.cfg.model_name
        else:
            model_path = self.cfg.model_name

        if start_generation is None:
            start_generation = 0

        loss_history: list[float] = []

        logger.info("=" * 60)
        logger.info("Starting Self-Consuming Loop")
        logger.info(f"  Base model      : {self.cfg.model_name}")
        logger.info(f"  Generations     : {self.cfg.num_generations}")
        logger.info(f"  Samples/gen     : {self.cfg.samples_per_generation}")
        logger.info(f"  Checkpoint dir  : {self.cfg.checkpoint_dir}")
        if start_generation > 0:
            logger.info(f"  Resuming from   : gen_{start_generation:03d}  ({model_path})")
        logger.info("=" * 60)

        # ── gen_000 (base model) ──────────────────────────────────
        m0_path = os.path.join(self.cfg.checkpoint_dir, "gen_000")
        if start_generation == 0:
            # Always use self.cfg.model_name (not resume path) so gen_000
            # is always the original base model, not a later checkpoint.
            if not Path(m0_path).exists():
                logger.info(f"Saving base model (M0) → {m0_path}")
                _model, _tok = self.load_model(self.cfg.model_name)
                _model.save_pretrained(m0_path)
                _tok.save_pretrained(m0_path)
                del _model, _tok
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                logger.info(f"gen_000 already exists, skipping: {m0_path}")
            if self.cfg.hf_push_each_gen:
                self.push_checkpoint_to_hub(m0_path, 0)
            if self.cfg.hf_delete_local_after_push and self.cfg.hf_push_each_gen:
                shutil.rmtree(m0_path, ignore_errors=True)
                logger.info(f"Deleted local gen_000: {m0_path}")
        else:
            logger.info(
                f"Resuming after gen_{start_generation:03d} — skipping gen_000 setup"
            )
            if not Path(m0_path).exists():
                logger.warning(
                    f"gen_000 not found locally at {m0_path}. "
                    "That is expected when the base model lives only on HF Hub."
                )

        # ── Main loop ─────────────────────────────────────────────
        for gen in range(start_generation + 1, self.cfg.num_generations + 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"GENERATION {gen} / {self.cfg.num_generations}")
            logger.info(f"{'=' * 60}")

            model, tokenizer = self.load_model(model_path)
            texts = self.generate_synthetic_data(model, tokenizer, gen)
            loss = self.fine_tune(model, tokenizer, texts, gen)
            loss_history.append(loss)

            local_ckpt = os.path.join(self.cfg.checkpoint_dir, f"gen_{gen:03d}")

            if self.cfg.hf_push_each_gen:
                self.push_checkpoint_to_hub(local_ckpt, gen)

            if self.cfg.hf_only and self.cfg.hf_repo_id:
                # Next generation loads straight from the Hub branch
                model_path = f"{self.cfg.hf_repo_id}@gen_{gen:03d}"
                logger.info(f"hf_only: next gen will load from {model_path}")
            else:
                model_path = local_ckpt

            if self.cfg.hf_delete_local_after_push and self.cfg.hf_push_each_gen:
                shutil.rmtree(local_ckpt, ignore_errors=True)
                logger.info(f"Deleted local checkpoint: {local_ckpt}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Final summary ──────────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("SELF-CONSUMING LOOP COMPLETE")
        logger.info(f"  Loss trajectory: {[round(l, 4) for l in loss_history]}")
        if len(loss_history) > 1:
            delta = loss_history[-1] - loss_history[0]
            direction = "↑ increased" if delta > 0 else "↓ decreased"
            logger.info(f"  Total loss shift: {delta:+.4f} ({direction})")
        logger.info("=" * 60)

        self.checker.save_report(self.cfg.logs_dir)
        self._save_dataset_manifest(loss_history, m0_path, start_generation)
        return loss_history

    def _save_dataset_manifest(
        self,
        loss_history: list[float],
        m0_path: str,
        start_generation: int = 0,
    ) -> None:
        """
        Write a single manifest.json at the checkpoint root that lists every
        generation's checkpoint path, data files, and loss.

        ``start_generation`` is the last generation that was already done
        before this run (0 = fresh start).  The manifest covers only the
        generations produced by *this* run.
        """
        entries = [{
            "generation":       0,
            "training_loss":    None,
            "checkpoint_dir":   m0_path,
            "training_records": None,
            "loop_meta":        None,
            "texts_json":       None,
            "records_jsonl":    None,
        }]

        for i, loss in enumerate(loss_history):
            gen = start_generation + i + 1
            ckpt_path = os.path.join(self.cfg.checkpoint_dir, f"gen_{gen:03d}")
            entries.append({
                "generation":        gen,
                "training_loss":     loss,
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