import os
import time
import math
import pickle
from contextlib import nullcontext
from functools import partial
import copy # FLOW CHANGE: For deepcopying model for EMA

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.nn import functional as F # FLOW CHANGE: Added for F.mse_loss

from model import GPTConfig, GPT # Assuming model.py contains the modified GPT class

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out_flow_match' 
eval_interval = 2000
log_interval = 1
eval_iters = 100 
eval_only = False 
skip_val_loss = False 
always_save_checkpoint = True 
never_save_checkpoint = False
init_from = 'scratch' 
# wandb logging
wandb_log = False 
wandb_project = 'flow_match_gpt2' 
wandb_run_name = 'gpt2-flow' 
# csv logging
csv_log = False
# data
dataset = 'openwebtext' 
gradient_accumulation_steps = 5 * 8 
batch_size = 12 
block_size = 1024
MASK_TOKEN_ID = -1 

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 
bias = False 
init_std = 0.02
# adamw optimizer
learning_rate = 3e-4 
max_iters = 600000 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.999 # FLOW CHANGE: Changed beta2 to 0.999 as per EDM2 suggestion
grad_clip = 1.0 
adam_eps = 1e-8 

# learning rate decay settings
decay_lr = True 
warmup_iters = 2000 
lr_decay_iters = 600000 
min_lr = 3e-5 

# mup settings
mup_enabled = False 
mup_disable_attention_scaling = False 
mup_disable_hidden_lr_scaling = False 
mup_width_multiplier = 1.0 
mup_input_alpha = 1.0 
mup_output_alpha = 1.0 
mup_enable_coord_check_logging = False 
# Depth scaling settings
depth_alpha_enabled = False 
depth_multiplier = 1.0
depth_alpha_exp = 1.0

# FLOW CHANGE: EMA settings
ema_decay = 0.999 # Typical EMA decay rate
ema_start_iter = 2000 # Iteration to start EMA updates
ema_update_every = 10 # How often to update EMA (in iterations)

# seed
seed = 1337
# DDP settings
backend = 'nccl' 
# system
device = 'cuda' 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
compile = True 
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # Can be uncommented if used
config = {k: globals()[k] for k in config_keys} 
# -----------------------------------------------------------------------------

assert not (never_save_checkpoint and always_save_checkpoint)

# FLOW CHANGE: EMA Model Class
class ModelEMA:
    """
    Model Exponential Moving Average.
    Keeps a copy of the model weights and updates them with EMA.
    """
    def __init__(self, model, decay, device=None):
        self.model = copy.deepcopy(model) # Create a shadow model
        self.model.requires_grad_(False) # EMA model doesn't need gradients
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.model.to(self.device)
        self.num_updates = 0

    def update(self, model_online):
        self.num_updates += 1
        # Correct EMA decay factor, especially for early updates
        # Karras et al. (EDM2) often use a dynamic decay based on number of updates
        # For simplicity here, sticking to a fixed decay but this can be refined.
        # A common refinement: decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        with torch.no_grad():
            for ema_param, online_param in zip(self.model.parameters(), model_online.parameters()):
                if online_param.device != ema_param.device: # Ensure devices match, DDP might cause this
                    online_param_data = online_param.data.to(ema_param.device)
                else:
                    online_param_data = online_param.data
                ema_param.data.mul_(self.decay).add_(online_param_data, alpha=1 - self.decay)
            
            # Also update buffers (e.g., running means in BatchNorm, though GPT typically doesn't use many)
            for ema_buffer, online_buffer in zip(self.model.buffers(), model_online.buffers()):
                if online_buffer.device != ema_buffer.device:
                     online_buffer_data = online_buffer.data.to(ema_buffer.device)
                else:
                    online_buffer_data = online_buffer.data
                ema_buffer.data.copy_(online_buffer_data)


    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 
    seed_offset = ddp_rank 
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size 
print(f"X_0 tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
device_type = 'cuda' if 'cuda' in device else 'cpu' 
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)
train_data_map = None
val_data_map = None
if os.path.exists(os.path.join(data_dir, 'train.bin')):
    train_data_map = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
if os.path.exists(os.path.join(data_dir, 'val.bin')):
    val_data_map = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split, current_MASK_TOKEN_ID):
    if split == 'train':
        data = train_data_map
    else:
        data = val_data_map
    
    if data is None:
        raise FileNotFoundError(f"Data file for split '{split}' not found in {data_dir}")

    ix = torch.randint(len(data) - block_size, (batch_size,))
    X_0_ids = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])

    epsilon_time = 3e-2 
    
    time_t_batch = torch.rand(batch_size, device=device) * (1.0 - 2 * epsilon_time) + epsilon_time
    time_s_batch = torch.rand(batch_size, device=device) * (1.0 - time_t_batch - epsilon_time) + time_t_batch + epsilon_time
    time_s_batch = torch.clamp(time_s_batch, min=time_t_batch + epsilon_time, max=1.0 - epsilon_time)

    # MOHIT
    def make_masks(tokens: torch.Tensor, s: torch.Tensor,t: torch.Tensor, mask_id: int):
        B, L = tokens.shape
        device = tokens.device

        # Expand keep-probabilities to (B, L)
        keep_s = s.view(B, 1).expand(B, L)   # probability 1
        keep_t = t.view(B, 1).expand(B, L)

        # Uniform[0,1) noise (optionally from a given RNG)
        U_s = torch.rand(B, L, device=device)
        U_t = torch.rand(B, L, device=device)

        M_s = (U_s < keep_s).int()           # 1 = keep, 0 = mask
        M_t = (U_t < keep_t).int()

        # Apply the masks → masked views
        X_s = tokens.clone()
        X_t = tokens.clone()
        X_s[M_s == 0] = mask_id
        X_t[M_t == 0] = mask_id

        return M_s, M_t, X_s, X_t

    M_s, M_t, X_s, X_t = make_masks(X_0_ids, time_t_batch, time_s_batch, current_MASK_TOKEN_ID)
    return X_0_ids, X_s, X_t, M_s, M_t, time_s_batch, time_t_batch


iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, init_std=init_std, 
                  mup_enabled=mup_enabled,
                  mup_disable_attention_scaling=mup_disable_attention_scaling,
                  mup_disable_hidden_lr_scaling=mup_disable_hidden_lr_scaling,
                  mup_width_multiplier=mup_width_multiplier, mup_input_alpha=mup_input_alpha,
                  mup_output_alpha=mup_output_alpha,
                  depth_alpha_enabled=depth_alpha_enabled, depth_alpha_exp=depth_alpha_exp, depth_multiplier=depth_multiplier) 

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if model_args.get('vocab_size') is None:
    raise ValueError("Model vocab_size is not determined. Cannot set MASK_TOKEN_ID.")
MASK_TOKEN_ID = model_args['vocab_size'] - 1 
print(f"Using MASK_TOKEN_ID: {MASK_TOKEN_ID} (derived from vocab_size: {model_args['vocab_size']})")
_get_batch_partial = partial(get_batch, current_MASK_TOKEN_ID=MASK_TOKEN_ID)


if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size 
model.to(device)

# FLOW CHANGE: Initialize EMA model
# Ensure model is fully on device before deepcopying for EMA
model_ema = None
if master_process: # EMA is usually managed by the master process
    model_ema = ModelEMA(model, decay=ema_decay, device=device)
    print(f"Initialized EMA model with decay {ema_decay}")
    if init_from == 'resume' and 'model_ema' in checkpoint:
        print("Loading EMA model weights from checkpoint.")
        model_ema.load_state_dict(checkpoint['model_ema'])


scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), adam_eps, device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None 

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) 

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model # unwrap DDP container if needed
# For eval, we might want to use the EMA model.
# Decide if raw_model_for_eval should be raw_model or model_ema.model
# Typically, EMA is used for eval.
# raw_model_for_eval = model_ema.model if model_ema is not None else raw_model (Handled inside estimate_loss)

@torch.no_grad()
def estimate_loss():
    out = {}
    # FLOW CHANGE: Determine which model to use for evaluation (EMA or online)
    eval_model_ref = model_ema.model if model_ema is not None and iter_num >= ema_start_iter else raw_model
    eval_model_ref.eval() # Set the chosen model to eval mode

    splits = ['train'] if skip_val_loss else ['train', 'val']
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # (B, T), (B, T), (B, T), (B, T), (B, 1), (B, T)
            X_0_ids, X_s, X_t, M_s, M_t, t_batch, s_batch = _get_batch_partial(split)
            
            with ctx:
                predicted_flow, _ = eval_model_ref(X_s, t_batch, s_batch) # Use eval_model_ref

                # Use wte from the same model for consistency in embeddings
                E_0 = eval_model_ref.transformer.wte(X_0) # (B, T, K)
                E_t_input_for_target = eval_model_ref.transformer.wte(X_t_in) # (B, T, K)
                
                time_s_reshaped = s_batch.view(-1, 1, 1) # (B, 1, 1)
                time_t_reshaped = t_batch.view(-1, 1, 1) # (B, 1, 1)
                delta_t = time_s_reshaped - time_t_reshaped + 1e-8 
                
                target_flow_velocities = (E_0 - E_t_input_for_target) / delta_t
                loss_calc_mask = (M_s == 1) & (M_t == 0)  # (B, T)
                
                current_loss = torch.tensor(0.0, device=device)
                num_active_positions = loss_calc_mask.sum()

                if num_active_positions > 0:
                    gap = (predicted_flow - target_flow_velocities) ** 2
                    mse_per_token = gap.mean(dim=2) # Mean over embedding dimension (B, T)
                    masked_mse = mse_per_token * loss_calc_mask 
                    current_loss = masked_mse.sum() / (num_active_positions)
                losses[k] = current_loss.item()
        out[split] = losses.mean().item() 
    if skip_val_loss:
        out['val'] = -1.0 
    
    # Ensure the original training model is set back to train mode if it was changed
    # raw_model.train() # This is implicitly handled if estimate_loss is called from master and model.train() is called later
    eval_model_ref.train() # Set back to train mode (though EMA model isn't trained directly)
                           # This ensures that if eval_model_ref was raw_model, it's set to train.
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)

if master_process:
    if wandb_log:
        import wandb
        wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    if csv_log:
        from csv_logging import CSVLogWrapper
        def log_func_stub(log_dict_arg): pass 
        csv_logger = CSVLogWrapper(log_func_stub, config=config, out_dir=out_dir)


X_0, X_t_in, M_t, M_s, t_batch, s_batch = _get_batch_partial('train') 
t0 = time.time()
local_iter_num = 0 
running_mfu = -1.0
coord_check_dict = None 

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group.get('lr_scale', 1.0)

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss() # Uses EMA model for eval if ready
        if np.isnan(losses['train']): 
            print(f"WARNING: NaN loss detected at iter {iter_num}. Stopping training.")
            break 
        print(f"step {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}") 
        log_dict = {
            "iter": iter_num,
            "train/loss": losses['train'], # This is train loss using EMA model
            "val/loss": losses['val'],   # Val loss using EMA model
            "lr": lr,
            "mfu": running_mfu*100, 
        }
        if mup_enable_coord_check_logging and coord_check_dict is not None:
            for key_cc in list(coord_check_dict.keys()): 
                 if key_cc == 'lm_head': continue 
                 log_dict[key_cc + '_act_abs_mean'] = np.mean(coord_check_dict[key_cc]) if coord_check_dict[key_cc] else 0

        if wandb_log:
            wandb_run.log(log_dict)
        if csv_log:
            csv_logger.log(log_dict) 

        if (not never_save_checkpoint) and (losses['val'] < best_val_loss or always_save_checkpoint):
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(), # Save online model
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                # FLOW CHANGE: Save EMA model state if it exists
                if model_ema is not None:
                    checkpoint['model_ema'] = model_ema.state_dict()
                print(f"saving checkpoint to {os.path.join(out_dir, 'ckpt.pt')}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    if mup_enable_coord_check_logging and iter_num % eval_interval == 0: 
        if coord_check_dict is None: 
            coord_check_dict = {
                'token_embedding': [], 'attn': [], 'mlp': [], 'last_layer': []
            }
            coord_check_handles = []
            for module_name, module in raw_model.named_modules(): 
                if module_name == 'transformer.wte':
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='token_embedding')))
                elif module_name.endswith('.attn'):
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='attn')))
                elif module_name.endswith('.mlp'):
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='mlp')))
                elif module_name == f'transformer.h.{model_args["n_layer"] - 1}': # Use model_args n_layer
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='last_layer')))
        else: 
            for key_cc in coord_check_dict: coord_check_dict[key_cc] = []


    # Ensure the main model is in training mode for the forward-backward pass
    model.train() # This sets the DDP-wrapped model or the plain model to train mode

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            predicted_flow, _ = model(X_t_in, t_batch, s_batch) 

            E_0 = raw_model.transformer.wte(X_0)
            E_t_input_for_target = raw_model.transformer.wte(X_t_in)

            time_s_reshaped = s_batch.view(-1, 1, 1)
            time_t_reshaped = t_batch.view(-1, 1, 1)
            delta_t = time_s_reshaped - time_t_reshaped + 1e-8

            target_flow_velocities = (E_0 - E_t_input_for_target) / delta_t
            loss_calc_mask = (M_s == 1) & (M_t == 0) 
            
            current_batch_loss = torch.tensor(0.0, device=device, requires_grad=True) 
            num_active_positions = loss_calc_mask.sum()

            if num_active_positions > 0:
                mse_per_element = F.mse_loss(predicted_flow, target_flow_velocities, reduction='none')
                masked_mse = mse_per_element * loss_calc_mask.unsqueeze(-1)
                current_batch_loss = masked_mse.sum() / (num_active_positions + 1e-8)
            
            loss = current_batch_loss / gradient_accumulation_steps 
        
        X_0, X_t_in, M_t, M_s, t_batch, s_batch = _get_batch_partial('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # Clip gradients of the online model
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # FLOW CHANGE: Update EMA model
    if model_ema is not None and iter_num >= ema_start_iter and iter_num % ema_update_every == 0:
        if master_process: # Only master process should update the single EMA model copy
            model_ema.update(raw_model) # Update EMA with the unwrapped online model


    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps 
        if local_iter_num >= 5: 
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.6f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%") 
    iter_num += 1
    local_iter_num += 1

    if mup_enable_coord_check_logging and iter_num % eval_interval == 0 : 
        if 'coord_check_handles' in locals(): 
            for handle in coord_check_handles:
                handle.remove()
            coord_check_handles = [] 

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

print("--- Flow Matching Training Finished ---")


# **Key Changes Made and Explanations:**

# 1.  **`get_batch(split, current_MASK_TOKEN_ID)` Function:**
#     * It now takes `current_MASK_TOKEN_ID` as an argument.
#     * Loads `X_0_ids` (original token sequences).
#     * Generates `time_t_batch` and `time_s_batch` ensuring `0 < t < s < 1` and `s-t` is not excessively small using `epsilon_time`.
#     * Generates boolean masks `M_t_batch` and `M_s_batch` (1 for unmasked, 0 for masked).
#         * The number of unmasked tokens at time `t` is proportional to `t`.
#         * Crucially, it ensures that tokens unmasked at time `t` are a subset of those unmasked at time `s` (`M_t <= M_s`), which is necessary for the loss condition `(M_s == 1) & (M_t == 0)` to represent tokens that *become* unmasked.
#     * Creates `idx_t_input` by taking `X_0_ids` and replacing tokens where `M_t_batch == 0` with the `MASK_TOKEN_ID`. This `idx_t_input` is what the model will receive as input for time `t`.
#     * Returns `(X_0_ids, idx_t_input, M_t_batch.long(), M_s_batch.long(), time_t_batch, time_s_batch)`. Masks are converted to `long` for convenience, though `bool` is also fine for PyTorch indexing.

# 2.  **`MASK_TOKEN_ID` Handling:**
#     * A global `MASK_TOKEN_ID` is introduced, initialized to a placeholder.
#     * After the model configuration `model_args` (especially `vocab_size`) is finalized (either from scratch, resume, or GPT-2 pretrained), `MASK_TOKEN_ID` is set to `model_args['vocab_size'] - 1`.
#         * **Caution:** This assumes that the token ID `vocab_size - 1` is suitable for masking (e.g., it's a padding token or an unused token whose embedding can be learned/adapted for masking). If using a pretrained GPT-2 model, `vocab_size` is 50257, so `MASK_TOKEN_ID` would be 50256, which is `<|endoftext|>`. Using a special token like this as a mask might have unintended consequences. True masked language modeling often involves adding a dedicated `[MASK]` token to the vocabulary and embedding matrix.
#     * The `get_batch` function is then partially applied with this determined `MASK_TOKEN_ID`.

# 3.  **Training Loop (`while True`):**
#     * **Data Fetching:** Unpacks the new batch structure: `X_0, X_t_in, M_t, M_s, t_batch, s_batch = _get_batch_partial('train')`.
#     * **Target Calculation:**
#         * `E_0 = raw_model.transformer.wte(X_0)`: Embeddings of the original clean tokens.
#         * `E_t_input_for_target = raw_model.transformer.wte(X_t_in)`: Embeddings of the actual input given to the model at time `t`.
#         * `delta_t = time_s_reshaped - time_t_reshaped + 1e-8`: Time difference, with epsilon for stability.
#         * `target_flow_velocities = (E_0 - E_t_input_for_target) / delta_t`: This is the target vector field your model aims to predict.
#     * **Model Forward Pass:** `predicted_flow, _ = model(X_t_in, t_batch, s_batch)`. The model takes the (potentially masked) input `X_t_in` and the time parameters.
#     * **Loss Mask:** `loss_calc_mask = (M_s == 1) & (M_t == 0)` identifies token positions that were masked at `t` but are unmasked at `s`.
#     * **MSE Loss Calculation:**
#         * `mse_per_element = F.mse_loss(predicted_flow, target_flow_velocities, reduction='none')` calculates element-wise squared errors.
#         * `masked_mse = mse_per_element * loss_calc_mask.unsqueeze(-1)` applies the loss mask (zeros out errors for non-target positions).
#         * `current_batch_loss = masked_mse.sum() / (num_active_positions + 1e-8)`: The sum of squared errors for the relevant components, normalized by the number of active token positions. This gives an average MSE per active token position.
#         * If `num_active_positions == 0`, the loss is set to `0.0`.
#     * The final `loss` is scaled by `gradient_accumulation_steps`.

# 4.  **`estimate_loss()` Function:**
#     * Updated to use the new `_get_batch_partial` and perform the same target and loss calculation logic as in the main training loop.

# 5.  **Configuration and Logging:**
#     * Changed default `out_dir` and `wandb_project` to reflect flow matching.
#     * Adjusted default `learning_rate`, `min_lr`, and `beta2` as common starting points for similar generative tasks.
#     * Removed `lm_head` specific coordinate check logging as the `lm_head` is no longer part of the model.

# 6.  **Hyperparameter Suggestions:**
#     * A new printout at the end of the script provides detailed suggestions for hyperparameters specifically relevant to training flow-matching models, covering time sampling, masking, learning rates, etc.

# Remember to have your modified `model.py` (with the `GPTConfig` and `GPT` class adapted for flow matching as per previous discussions) in the same directory or accessible in your `PYTHONPATH`. You'll also need to prepare your dataset (`openwebtext` or other) in the `data/{dataset}` directory, specifically the `train.bin` and `val.bin` files containing tokenized `X_0` sequenc