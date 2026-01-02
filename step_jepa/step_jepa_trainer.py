"""
Step-JEPA Custom Trainer

This implements the correct Step-JEPA architecture:
- Question in user message
- ALL reasoning steps in assistant message (Step1\n\nStep2\n\nStep3...)
- Two forward passes:
  1. Normal NTP loss on full sequence
  2. Custom attention: Step2 isolated, extract embeddings for JEPA loss
- <|predictor|> token inserted after Step1
- JEPA aligns: predictor token embedding <-> Step2 end embedding
"""

import torch
import torch.nn.functional as F
from transformers import Trainer


class StepJEPATrainer(Trainer):
    """
    Custom trainer for Step-JEPA with proper attention masking.
    
    Key features:
    - Finds Step1 and Step2 boundaries in tokenized sequence
    - Inserts <|predictor|> after Step1
    - Forward pass 1: Normal causal (NTP loss)
    - Forward pass 2: Custom mask (Step2 isolated), extract embeddings
    - JEPA loss aligns embeddings
    """
    
    def __init__(self, *args, **kwargs):
        # Extract custom parameters
        self.lbd = kwargs.pop('lbd', 0.1)
        self.gamma = kwargs.pop('gamma', 1.0)
        self.debug = kwargs.pop('debug', 0)
        self.jepa_loss_type = kwargs.pop('jepa_loss_type', 'cosine')
        super().__init__(*args, **kwargs)
        
        # Cache for step separator token
        self.step_separator = "\n\n"
        self.predictor_token_id = None
    
    def _find_step_boundaries(self, input_ids, tokenizer):
        """
        Find token positions where Step1 and Step2 end.
        
        Returns:
            step1_end_pos: Position after first \n\n (where predictor goes)
            step2_end_pos: Position after second \n\n (where Step2 ends)
            
        Returns None, None if can't find boundaries.
        """
        # Tokenize the separator
        sep_tokens = tokenizer.encode(self.step_separator, add_special_tokens=False)
        
        # Convert to list for easier searching
        ids_list = input_ids.tolist()
        
        # Find first occurrence of separator (after Step1)
        step1_end = None
        for i in range(len(ids_list) - len(sep_tokens) + 1):
            if ids_list[i:i+len(sep_tokens)] == sep_tokens:
                step1_end = i + len(sep_tokens) - 1  # Position of last separator token
                break
        
        if step1_end is None:
            return None, None
        
        # Find second occurrence (after Step2)
        step2_end = None
        for i in range(step1_end + len(sep_tokens), len(ids_list) - len(sep_tokens) + 1):
            if ids_list[i:i+len(sep_tokens)] == sep_tokens:
                step2_end = i + len(sep_tokens) - 1
                break
        
        if step2_end is None:
            return None, None
        
        return step1_end, step2_end
    
    def _insert_predictor_token(self, input_ids, step1_end_pos, predictor_token_id):
        """
        Insert predictor token after Step1.
        
        Args:
            input_ids: [batch_size, seq_len]
            step1_end_pos: tensor of positions for each batch
            predictor_token_id: ID of predictor token
            
        Returns:
            new_input_ids: [batch_size, seq_len + 1] with predictor inserted
            new_step2_end_pos: adjusted Step2 end positions (+1)
        """
        batch_size = input_ids.shape[0]
        new_input_ids = []
        
        for i in range(batch_size):
            pos = step1_end_pos[i].item()
            # Insert predictor after step1_end_pos
            new_seq = torch.cat([
                input_ids[i, :pos+1],
                torch.tensor([predictor_token_id], device=input_ids.device),
                input_ids[i, pos+1:]
            ])
            new_input_ids.append(new_seq)
        
        return torch.stack(new_input_ids)
    
    def _build_step2_isolation_mask(self, seq_len, step1_end_pos, step2_end_pos, predictor_pos, device):
        """
        Create attention mask that isolates Step2 tokens.
        
        Attention pattern:
        - [0:predictor_pos]: Normal causal (System, User, Step1, predictor)
        - [predictor_pos+1:step2_end_pos+1]: Isolated (Step2 can only see itself)
        - [step2_end_pos+1:]: Normal causal (Step3+ can see everything including Step2)
        
        Returns:
            mask: [batch_size, 1, seq_len, seq_len] additive attention mask
                  (0 = attend, -inf = blocked)
        """
        batch_size = step1_end_pos.shape[0]
        
        # Start with causal mask (lower triangular)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        
        # Expand for batch
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).clone()
        
        # For each example in batch, isolate Step2
        for i in range(batch_size):
            pred_pos = predictor_pos[i].item()
            step2_start = pred_pos + 1
            step2_end = step2_end_pos[i].item()
            
            # Step2 tokens can only attend to themselves
            # Block attention from Step2 to everything before it
            mask[i, 0, step2_start:step2_end+1, :step2_start] = float('-inf')
            
            # Step3+ has normal causal attention (can see everything including Step2)
            # No additional masking needed - the base causal mask handles it
        
        return mask
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with two forward passes:
        1. Normal forward: Standard NTP loss
        2. Custom masked forward: Extract embeddings for JEPA loss
        """
        tokenizer = self.tokenizer
        
        # Initialize predictor token ID (cache it)
        if self.predictor_token_id is None:
            self.predictor_token_id = tokenizer.convert_tokens_to_ids("<|predictor|>")
        
        batch_size = inputs["input_ids"].shape[0]
        
        # Find step boundaries for each example in batch
        step1_end_positions = []
        step2_end_positions = []
        
        for i in range(batch_size):
            step1_end, step2_end = self._find_step_boundaries(
                inputs["input_ids"][i],
                tokenizer
            )
            if step1_end is None or step2_end is None:
                # Skip this example if can't find boundaries
                if self.debug >= 1:
                    print(f"Warning: Could not find step boundaries for example {i}")
                step1_end_positions.append(0)
                step2_end_positions.append(0)
            else:
                step1_end_positions.append(step1_end)
                step2_end_positions.append(step2_end)
        
        step1_end_pos = torch.tensor(step1_end_positions, device=inputs["input_ids"].device)
        step2_end_pos = torch.tensor(step2_end_positions, device=inputs["input_ids"].device)
        
        # ===== Forward Pass 1: Normal NTP Loss =====
        with torch.set_grad_enabled(True):
            outputs_ntp = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
        
        lm_loss = outputs_ntp.loss
        
        # ===== Forward Pass 2: JEPA Embeddings with Custom Mask =====
        
        # Insert predictor tokens
        input_ids_with_pred = self._insert_predictor_token(
            inputs["input_ids"],
            step1_end_pos,
            self.predictor_token_id
        )
        
        # Predictor is right after step1_end
        predictor_positions = step1_end_pos + 1
        
        # Adjust step2_end positions (+1 because we inserted a token)
        step2_end_pos_adjusted = step2_end_pos + 1
        
        # Build custom attention mask
        seq_len_with_pred = input_ids_with_pred.shape[1]
        custom_mask = self._build_step2_isolation_mask(
            seq_len_with_pred,
            step1_end_pos,
            step2_end_pos_adjusted,
            predictor_positions,
            inputs["input_ids"].device
        )
        
        # Forward with custom mask
        with torch.set_grad_enabled(True):
            outputs_jepa = model(
                input_ids=input_ids_with_pred,
                attention_mask=custom_mask,
                output_hidden_states=True
            )
        
        # Extract embeddings
        hidden_states = outputs_jepa.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        
        # View 1: Embedding at predictor token
        view1_embeddings = hidden_states[range(batch_size), predictor_positions, :]
        
        # View 2: Embedding at Step2 end
        view2_embeddings = hidden_states[range(batch_size), step2_end_pos_adjusted, :]
        
        # Compute JEPA loss
        if self.jepa_loss_type == 'cosine':
            cosine_sim = F.cosine_similarity(view1_embeddings, view2_embeddings, dim=-1)
            jepa_loss = 1.0 - torch.mean(cosine_sim)
        elif self.jepa_loss_type == 'l2':
            jepa_loss = torch.linalg.norm(view1_embeddings - view2_embeddings, ord=2, dim=-1).mean()
        elif self.jepa_loss_type == 'mse':
            jepa_loss = torch.mean((view1_embeddings - view2_embeddings) ** 2)
        else:
            raise ValueError(f"Unknown jepa_loss_type: {self.jepa_loss_type}")
        
        # Total loss
        total_loss = self.gamma * lm_loss + self.lbd * jepa_loss
        
        if self.debug >= 1 and torch.cuda.current_device() == 0:
            print(f"LM loss: {lm_loss.item():.4f}, JEPA loss: {jepa_loss.item():.4f}, Total: {total_loss.item():.4f}")
        
        return (total_loss, outputs_ntp) if return_outputs else total_loss

