"""
Training script for Step-JEPA

Uses the custom StepJEPATrainer with proper attention masking.
"""

import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse

# Add parent directory to path to import trainer
sys.path.insert(0, os.path.dirname(__file__))
from step_jepa_trainer import StepJEPATrainer


def setup_model_and_tokenizer(model_name, use_lora=True, lora_rank=16):
    """Setup model and tokenizer with predictor token"""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add predictor token
    special_tokens = ["<|predictor|>"]
    new_tokens = [token for token in special_tokens if token not in tokenizer.vocab]
    
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        print(f"Added {len(new_tokens)} new special tokens: {new_tokens}")
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if int(os.environ.get('WORLD_SIZE', 1)) == 1 else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    
    # Resize embeddings if we added tokens
    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))
    
    # Setup LoRA if requested
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        print("="*80)
        model.print_trainable_parameters()
        print("="*80)
    
    return model, tokenizer


def load_and_prepare_dataset(data_file, tokenizer, max_length=2048):
    """Load dataset and tokenize"""
    
    dataset = load_dataset('json', data_files=data_file)['train']
    print(f"Loaded {len(dataset)} examples from {data_file}")
    
    def tokenize_function(examples):
        """Tokenize conversations with proper label masking"""
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        
        for messages in examples['messages']:
            # Apply chat template
            formatted_chat = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            
            # Tokenize
            tokenized = tokenizer(
                formatted_chat,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
            
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            
            # Create labels: mask system and user, keep assistant
            labels = create_masked_labels(messages, tokenizer, input_ids, attention_mask)
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
        
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list,
        }
    
    def create_masked_labels(messages, tokenizer, input_ids, attention_mask):
        """Create labels with system/user masked (-100)"""
        labels = [-100] * len(input_ids)
        
        # Mask padding
        for i, mask in enumerate(attention_mask):
            if mask == 0:
                labels[i] = -100
        
        # Find and unmask assistant response
        for msg in messages:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                assistant_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)
                
                # Find assistant response in sequence
                decoded_assistant = [tokenizer.decode([token]) for token in assistant_tokens]
                decoded_input = [tokenizer.decode([token]) for token in input_ids]
                
                for i in range(len(input_ids) - len(assistant_tokens) + 1):
                    if attention_mask[i] == 1 and decoded_input[i:i+len(assistant_tokens)] == decoded_assistant:
                        # Unmask assistant tokens
                        for j in range(i, min(i + len(assistant_tokens), len(input_ids))):
                            if attention_mask[j] == 1:
                                labels[j] = input_ids[j]
                        break
        
        return labels
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Step-JEPA model")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default=None)
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    
    # Training arguments
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--eval_steps", type=int, default=50)
    
    # LoRA arguments
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    
    # Step-JEPA arguments
    parser.add_argument("--lbd", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--jepa_loss_type", type=str, default="cosine", choices=["cosine", "l2", "mse"])
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", type=int, default=0)
    
    args = parser.parse_args()
    
    # Print config
    print("="*80)
    print("Step-JEPA Training Configuration")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Lambda (JEPA): {args.lbd}")
    print(f"Gamma (LM): {args.gamma}")
    print(f"JEPA loss type: {args.jepa_loss_type}")
    print(f"LoRA: {args.lora}")
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
        lora_rank=args.lora_rank
    )
    
    # Load dataset
    print("\n2. Loading and preparing dataset...")
    train_dataset = load_and_prepare_dataset(
        args.train_file,
        tokenizer,
        args.max_length
    )
    
    eval_dataset = None
    if args.eval_file:
        eval_dataset = load_and_prepare_dataset(
            args.eval_file,
            tokenizer,
            args.max_length
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
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
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
    print("\n3. Initializing Step-JEPA trainer...")
    trainer = StepJEPATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        lbd=args.lbd,
        gamma=args.gamma,
        debug=args.debug if args.debug > 0 else 5,  # Default to 5 for loss logging
        jepa_loss_type=args.jepa_loss_type,
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
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        else:
            trainer.save_model()
            trainer.save_state()
            tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n✅ Training completed! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

