"""
Step-JEPA Adapter for finetune.py

This script adapts the existing finetune.py to support Step-JEPA by:
1. Using the data format we prepared (question → all steps concatenated)
2. Finding step boundaries (\n\n separators)
3. Modifying the attention mask to isolate Step 2
4. Using existing RepresentationTrainer infrastructure

Usage:
    python finetune_step_jepa_adapted.py \
        --train_file ./gsm8k_step_jepa.jsonl \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --output_dir ./checkpoints_adapted \
        --step_jepa \
        --lora
"""

import sys
import os
import torch

# Add parent directory to use finetune.py modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from finetune import (
    setup_model_and_tokenizer,
    load_and_prepare_dataset,
    RepresentationTrainer,
    ProfilerFLOPCallback
)
from transformers import TrainingArguments, DataCollatorForLanguageModeling
import argparse
import shutil


class StepJEPARepresentationTrainer(RepresentationTrainer):
    """
    Extended RepresentationTrainer that supports Step-JEPA masking.
    
    When step_jepa=True:
    - Finds \n\n separators to identify step boundaries
    - Isolates Step 2 in the attention mask
    - Extracts embeddings at predictor token and Step 2 end
    
    When step_jepa=False:
    - Uses original LLM-JEPA logic (Text ↔ Code)
    """
    
    def __init__(self, *args, **kwargs):
        # Extract Step-JEPA specific parameters before calling super().__init__
        self.step_jepa = kwargs.pop('step_jepa', False)
        self.step_jepa_predictors = kwargs.pop('step_jepa_predictors', 1)  # K predictor tokens after Step 1
        
        # Pass all other parameters to parent, including last_token, jepa_ratio
        super().__init__(*args, **kwargs)
        
        if self.step_jepa:
            print(f"✅ Step-JEPA mode enabled: Will isolate Step 2, K={self.step_jepa_predictors} predictor tokens")
        else:
            print("ℹ️  Using base RepresentationTrainer (additive_mask logic from finetune.py)")
    
    def _find_step_boundaries(self, input_ids, processing_class):
        """
        Find positions of \n\n markers that represent step boundaries.
        
        In step-based reasoning, \n\n serves as THE representation point for each step.
        We extract embeddings AT the \n\n position (not after it).
        
        Returns:
            step1_end: Position of \n\n after Step 1 (last token if \n\n spans multiple tokens)
            step2_end: Position of \n\n after Step 2 (last token if \n\n spans multiple tokens)
            Returns None, None if can't find boundaries
            
        Note: Predictor tokens will be inserted RIGHT AFTER step1_end.
        """
        # Tokenize the separator - this is THE representation marker for each step
        sep_tokens = processing_class.encode("\n\n", add_special_tokens=False)
        
        # Convert to list for searching
        ids_list = input_ids.tolist()
        
        # Find first \n\n (THE representation point of Step 1)
        step1_end = None
        for i in range(len(ids_list) - len(sep_tokens) + 1):
            if ids_list[i:i+len(sep_tokens)] == sep_tokens:
                # Position of last token in \n\n sequence
                # This is THE embedding extraction point for Step 1
                step1_end = i + len(sep_tokens) - 1
                break
        
        if step1_end is None:
            return None, None
        
        # Find second \n\n (THE representation point of Step 2)
        step2_end = None
        for i in range(step1_end + len(sep_tokens), len(ids_list) - len(sep_tokens) + 1):
            if ids_list[i:i+len(sep_tokens)] == sep_tokens:
                # Position of last token in \n\n sequence
                # This is THE embedding extraction point for Step 2
                step2_end = i + len(sep_tokens) - 1
                break
        
        if step2_end is None:
            return None, None
        
        return step1_end, step2_end
    
    def build_with_additive_mask(self, inputs):
        """
        Override parent's method to support Step-JEPA masking.
        
        Follows the original finetune.py pattern:
        1. Insert K predictor tokens after Step 1
        2. DOUBLE the batch
        3. First half: Modified tokens + Normal causal mask
        4. Second half: Modified tokens + Step-JEPA isolation mask
        
        If step_jepa is False, use parent's implementation (default additive_mask from finetune.py).
        If step_jepa is True, apply Step-JEPA attention masking.
        """
        if not self.step_jepa:
            # Use parent's default additive_mask logic from finetune.py
            return super().build_with_additive_mask(inputs)
        
        # Apply jepa_ratio dropout (same as original)
        if self.jepa_ratio > 0.0:
            if torch.rand(1).item() > self.jepa_ratio:
                return {
                    "input_ids": inputs["input_ids"],
                    "labels": inputs["labels"],
                    "attention_mask": inputs["attention_mask"],
                }, True  # skip_jepa=True
        
        # Step-JEPA logic
        batch_size = inputs["input_ids"].shape[0]
        seq_length = inputs["input_ids"].shape[-1]
        device = inputs["input_ids"].device
        
        # Find step boundaries for each example
        step1_end_positions = []
        step2_end_positions = []
        
        for i in range(batch_size):
            step1_end, step2_end = self._find_step_boundaries(
                inputs["input_ids"][i],
                self.processing_class
            )
            
            if step1_end is None or step2_end is None:
                # Fall back to sequence middle if can't find boundaries
                last_token = self._last_token_index(
                    inputs["input_ids"][i:i+1],
                    inputs["labels"][i:i+1],
                    inputs["attention_mask"][i:i+1]
                )[0].item()
                step1_end = last_token // 3
                step2_end = (last_token * 2) // 3
            
            step1_end_positions.append(step1_end)
            step2_end_positions.append(step2_end)
        
        step1_end_pos = torch.tensor(step1_end_positions, device=device)
        step2_end_pos = torch.tensor(step2_end_positions, device=device)
        
        # Insert K predictor tokens after Step 1 for each example
        # Create new input_ids with predictor tokens inserted
        new_input_ids = []
        for i in range(batch_size):
            step1_end = step1_end_pos[i].item()
            # Insert K predictor tokens after step1_end
            predictor_ids = [self.processing_class.convert_tokens_to_ids(f"<|predictor_{j+1}|>") 
                            for j in range(self.step_jepa_predictors)]
            
            new_seq = torch.cat([
                inputs["input_ids"][i, :step1_end+1],
                torch.tensor(predictor_ids, device=device),
                inputs["input_ids"][i, step1_end+1:]
            ])
            new_input_ids.append(new_seq[:seq_length])  # Truncate to original length
        
        new_input_ids = torch.stack(new_input_ids)
        
        # DOUBLE THE BATCH (following original finetune.py pattern)
        # First half: normal causal mask
        # Second half: Step-JEPA isolation mask
        doubled_input_ids = torch.cat([new_input_ids, new_input_ids], dim=0)
        doubled_labels = torch.cat([inputs["labels"], inputs["labels"]], dim=0)
        
        # Create attention masks for DOUBLED batch
        mask = torch.full((batch_size * 2, 1, seq_length, seq_length), float('-inf')).to(device)
        
        for i in range(batch_size):
            step1_end = step1_end_pos[i].item()
            step2_end = step2_end_pos[i].item()
            
            # Calculate positions after inserting K predictor tokens
            predictor_start = step1_end + 1
            predictor_end = predictor_start + self.step_jepa_predictors - 1
            step2_start = predictor_end + 1
            step2_end_adjusted = step2_end + self.step_jepa_predictors
            
            # Find actual sequence length (non-padding)
            last_token = self._last_token_index(
                inputs["input_ids"][i:i+1],
                inputs["labels"][i:i+1],
                inputs["attention_mask"][i:i+1]
            )[0].item()
            seq_len_actual = last_token + 1 + self.step_jepa_predictors  # Adjust for inserted tokens
            
            # FIRST HALF (index i): Normal causal mask for entire sequence
            mask[i, 0, :seq_len_actual, :seq_len_actual] = self._build_additive_mask(seq_len_actual)
            
            # SECOND HALF (index i + batch_size): Step-JEPA isolation mask
            # - Everything before Step 2: normal causal
            mask[i + batch_size, 0, :step2_start, :step2_start] = self._build_additive_mask(step2_start)
            # - Step 2: isolated (can only see itself)
            mask[i + batch_size, 0, step2_start:step2_end_adjusted+1, step2_start:step2_end_adjusted+1] = \
                self._build_additive_mask(step2_end_adjusted - step2_start + 1)
            # - Step 3+: normal causal (can see everything)
            if step2_end_adjusted + 1 < seq_len_actual:
                mask[i + batch_size, 0, step2_end_adjusted+1:seq_len_actual, :seq_len_actual] = \
                    self._build_additive_mask(seq_len_actual)[step2_end_adjusted+1:seq_len_actual, :seq_len_actual]
        
        # Store positions for later use in compute_loss
        # These are for the SECOND HALF of the doubled batch
        self._step1_end_pos = step1_end_pos
        self._step2_end_pos = step2_end_pos + self.step_jepa_predictors  # Adjusted for inserted tokens
        self._predictor_pos = step1_end_pos + self.step_jepa_predictors  # Last predictor token
        
        return {
            "input_ids": doubled_input_ids,      # Shape: (batch_size * 2, seq_len)
            "labels": doubled_labels,            # Shape: (batch_size * 2, seq_len)
            "attention_mask": mask,              # Shape: (batch_size * 2, 1, seq_len, seq_len)
        }, False
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override to extract embeddings at correct positions for Step-JEPA.
        
        Follows the original finetune.py pattern:
        - forward_results contains outputs from DOUBLED batch (if JEPA is not skipped)
        - LM loss is averaged over BOTH halves (normal + JEPA masked)
        - JEPA embeddings extracted from SECOND HALF only
        - Respects jepa_ratio for selective JEPA loss
        """
        if not self.step_jepa or not self.additive_mask:
            # Use parent's implementation
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        
        # Step-JEPA specific loss computation
        forward_results = self.forward(model, inputs)
        
        main_outputs = forward_results['main_outputs']
        lm_loss = main_outputs.loss
        
        # Check if JEPA was skipped (due to jepa_ratio)
        user_hidden_states = forward_results['user_hidden_states']
        
        if user_hidden_states is None:
            # JEPA was skipped, only use LM loss
            total_loss = lm_loss
            
            if self.debug >= 5 and torch.cuda.current_device() == 0:
                print(f"llm_loss: {lm_loss.float()}, jepa_loss: skipped (jepa_ratio)")
            
            return (total_loss, main_outputs) if return_outputs else total_loss
        
        # JEPA was not skipped, compute JEPA loss
        # Extract hidden states from the forward pass
        # hidden_states shape: (batch_size * 2, seq_len, hidden_dim)
        hidden_states = main_outputs.hidden_states[-1]
        total_batch_size = hidden_states.shape[0]
        original_batch_size = total_batch_size // 2  # Recover original batch size
        
        # Extract embeddings from SECOND HALF only (indices batch_size to batch_size*2)
        # This is where the Step-JEPA isolation mask was applied
        jepa_hidden_states = hidden_states[original_batch_size:total_batch_size, :, :]
        
        # Extract View 1: Last predictor token's embedding
        view1_embeddings = jepa_hidden_states[range(original_batch_size), self._predictor_pos, :]
        
        # Extract View 2: Step 2's \n\n embedding
        view2_embeddings = jepa_hidden_states[range(original_batch_size), self._step2_end_pos, :]
        
        # Compute JEPA loss
        if self.jepa_l2:
            jepa_loss = torch.linalg.norm(view1_embeddings - view2_embeddings, ord=2, dim=-1).mean()
        elif self.jepa_mse:
            jepa_loss = torch.mean((view1_embeddings - view2_embeddings) ** 2)
        elif self.infonce:
            import torch.nn.functional as F
            v1_norm = F.normalize(view1_embeddings, p=2, dim=1)
            v2_norm = F.normalize(view2_embeddings, p=2, dim=1)
            cosine_sim = torch.mm(v1_norm, v2_norm.T)
            infonce_logit = cosine_sim / 0.07
            infonce_label = torch.arange(cosine_sim.size(0), device=cosine_sim.device)
            jepa_loss = F.cross_entropy(infonce_logit, infonce_label)
        else:
            import torch.nn.functional as F
            cosine_similarity = F.cosine_similarity(view1_embeddings, view2_embeddings, dim=-1)
            jepa_loss = 1.0 - torch.mean(cosine_similarity)
        
        total_loss = self.gamma * lm_loss + self.lbd * jepa_loss
        
        # Log losses like original finetune.py (debug level 5)
        if self.debug >= 5 and torch.cuda.current_device() == 0:
            print(f"llm_loss: {lm_loss.float()}, jepa_loss: {jepa_loss.float()}")
        
        return (total_loss, main_outputs) if return_outputs else total_loss


def main():
    parser = argparse.ArgumentParser(description="Step-JEPA Training (Adapted from finetune.py)")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--eval_file", type=str, help="Path to evaluation JSONL file")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name/path")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_step_jepa_adapted", help="Output directory")
    
    # Training arguments
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluation steps")
    
    # LoRA arguments
    parser.add_argument("--lora", action="store_true", help="Enable LoRA")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    
    # Step-JEPA arguments
    parser.add_argument("--step_jepa", action="store_true", help="Enable Step-JEPA mode (isolate Step 2)")
    parser.add_argument("--regular", action="store_true", help="Use regular trainer without JEPA")
    parser.add_argument("--predictors", type=int, default=1, help="Number of K predictor tokens after Step 1")
    parser.add_argument("--lbd", type=float, default=0.1, help="Lambda for JEPA loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma for LM loss")
    parser.add_argument("--last_token", type=int, default=-1, help="Index of last token for embedding extraction")
    parser.add_argument("--jepa_l2", action="store_true", help="Use L2 norm as JEPA loss")
    parser.add_argument("--jepa_mse", action="store_true", help="Use MSE as JEPA loss")
    parser.add_argument("--infonce", action="store_true", help="Use InfoNCE loss")
    parser.add_argument("--jepa_ratio", type=float, default=-1.0, help="Random JEPA loss dropout ratio")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--finetune_seed", type=int, default=42, help="Finetune random seed (alias for seed)")
    parser.add_argument("--debug", type=int, default=0, help="Debug level")
    
    args = parser.parse_args()
    
    # Use finetune_seed if provided (matches finetune.py)
    if args.finetune_seed != 42:
        args.seed = args.finetune_seed
    
    # Validate
    if args.regular and args.step_jepa:
        parser.error("Cannot use both --regular and --step_jepa. Choose one.")
    
    if not args.step_jepa and not args.regular and args.predictors == 1:
        # Default predictors to 1 for Step-JEPA, keep for backwards compatibility
        print("⚠️  Warning: Neither --step_jepa nor --regular specified.")
        print("   Will use base RepresentationTrainer from finetune.py.")
        print("   Add --step_jepa for Step-JEPA or --regular for no JEPA.")
    
    # Print config
    print("="*80)
    print("Step-JEPA Training (Adapted from finetune.py)")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Mode: {'Regular (No JEPA)' if args.regular else 'Step-JEPA' if args.step_jepa else 'Base RepresentationTrainer'}")
    if not args.regular:
        print(f"Lambda (JEPA): {args.lbd}")
        print(f"Gamma (LM): {args.gamma}")
        print(f"Last token: {args.last_token}")
        print(f"Predictors (K): {args.predictors}")
        if args.step_jepa:
            print(f"Step-JEPA: Isolate Step 2, K={args.predictors} tokens after Step 1")
    print(f"LoRA: {args.lora}")
    print(f"Seed: {args.seed}")
    print("="*80)
    
    # Setup distributed training
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        print(f"Running with torchrun: world_size={world_size}, local_rank={local_rank}")
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    # Setup model and tokenizer
    print("\n1. Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        use_lora=args.lora,
        lora_rank=args.lora_rank,
        debug=args.debug,
        seed=args.seed
    )
    
    # Add predictor tokens if Step-JEPA
    if args.step_jepa:
        # Add K predictor tokens (where K = args.predictors)
        special_tokens = [f"<|predictor_{i+1}|>" for i in range(args.predictors)]
        new_tokens = [token for token in special_tokens if token not in tokenizer.vocab]
        
        if new_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            model.resize_token_embeddings(len(tokenizer))
            print(f"Added {len(new_tokens)} predictor tokens: {new_tokens}")
    
    # Load dataset
    print("\n2. Loading and preparing dataset...")
    train_dataset = load_and_prepare_dataset(
        args.train_file,
        tokenizer,
        args.model_name,
        args.max_length,
        debug=args.debug,
        predictors=args.predictors,
        regular=args.regular  # Use regular mode if specified
    )
    
    eval_dataset = None
    if args.eval_file:
        eval_dataset = load_and_prepare_dataset(
            args.eval_file,
            tokenizer,
            args.model_name,
            args.max_length,
            debug=args.debug,
            predictors=args.predictors,
            regular=args.regular
        )
    
    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval samples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    save_steps = len(train_dataset) // (world_size * args.batch_size * args.grad_accum)
    output_dir = os.path.abspath(args.output_dir)
    
    if torch.cuda.current_device() == 0:
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.eval_steps,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl" if world_size > 1 else None,
        report_to="none",
        remove_unused_columns=False,
        tf32=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    
    # Initialize Step-JEPA trainer
    print("\n3. Initializing trainer...")
    
    if args.regular:
        # Use regular Trainer (no JEPA)
        from transformers import Trainer
        print("Using regular Trainer (no JEPA)")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
        # Use StepJEPARepresentationTrainer
        print(f"Using StepJEPARepresentationTrainer (step_jepa={args.step_jepa}, K={args.predictors})")
        trainer = StepJEPARepresentationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            step_jepa=args.step_jepa,
            step_jepa_predictors=args.predictors if args.step_jepa else 1,  # K for Step-JEPA
            lbd=args.lbd,
            gamma=args.gamma,
            last_token=args.last_token,
            debug=args.debug if args.debug > 0 else 5,  # Default to 5 for loss logging
            additive_mask=True,  # Use additive mask for efficiency
            jepa_l2=args.jepa_l2,
            jepa_mse=args.jepa_mse,
            infonce=args.infonce,
            jepa_ratio=args.jepa_ratio,
        )
    
    # Start training
    print("\n4. Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    # Save final model
    print("\n5. Saving final model...")
    if torch.cuda.current_device() == 0:
        if args.lora:
            model = model.merge_and_unload()
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            trainer.save_model()
            trainer.save_state()
            tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Training completed! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

