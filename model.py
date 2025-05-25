"""
time_conditioned_gpt2_vector.py
================================
A **time‑conditioned GPT backbone that outputs a token‑aligned *vector field* `v_θ`**
instead of language‑model logits.  The final hidden states after the Transformer
stack are returned directly and have shape **`(batch, seq_len, n_embd)`** — the
same *token grid* as the input.

You can attach *any* loss externally (for instance the score‑matching objective
of masked diffusion LMs):
```python
v_theta = model(input_ids, time_s, time_t)          # (B, T, D)
loss    = custom_loss_fn(v_theta, target_ids, extras...)
loss.backward()
```
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# ─────────────────────────────────────────────────────────────────────────────
# 1. Building blocks (unchanged from your original GPT) ───────────────────────
# ─────────────────────────────────────────────────────────────────────────────
class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch doesn’t expose `bias=False`)."""
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """Full‑attention (bidirectional) block. Identical to original except causal mask removed."""
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.qkv   = nn.Linear(cfg.n_embd, 3*cfg.n_embd, bias=cfg.bias)
        self.proj  = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.proj_drop = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.flash   = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B,T,self.n_head,self.head_dim).transpose(1,2)  # (B,h,T,hd)
        k = k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        scale = 1.0 / math.sqrt(self.head_dim)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=False, scale=scale)
        else:
            att = (q @ k.transpose(-2,-1)) * scale
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y   = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.proj_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.fc   = nn.Linear(cfg.n_embd, 4*cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4*cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self,x):
        return self.drop(self.proj(F.gelu(self.fc(x))))

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.att = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MLP(cfg)
        self.res_scale = 1/(cfg.depth_multiplier ** cfg.depth_alpha_exp) if cfg.depth_alpha_enabled else 1.0
    def forward(self,x):
        x = x + self.res_scale * self.att(self.ln1(x))
        x = x + self.res_scale * self.mlp(self.ln2(x))
        return x

# ─────────────────────────────────────────────────────────────────────────────
# 2. Configuration dataclass ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GPTConfig:
    block_size : int  = 1024
    vocab_size : int  = 50304
    n_layer    : int  = 12
    n_head     : int  = 12
    n_embd     : int  = 768
    dropout    : float= 0.0
    bias       : bool = True
    init_std   : float= 0.02
    # muP / depth scaling (kept)
    mup_enabled                    : bool  = False
    mup_disable_attention_scaling  : bool  = False
    mup_disable_hidden_lr_scaling  : bool  = False
    mup_width_multiplier           : float = 1.0
    mup_input_alpha                : float = 1.0
    mup_output_alpha               : float = 1.0
    depth_alpha_enabled            : bool  = False
    depth_multiplier               : float = 1.0
    depth_alpha_exp                : float = 1.0
    # NEW: dimension of hidden layer in time‑embedding MLP (None→n_embd)
    time_dim : int | None = None

# ─────────────────────────────────────────────────────────────────────────────
# 3. GPT backbone with time conditioning, outputting v_theta ──────────────────
# ─────────────────────────────────────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self, cfg:GPTConfig):
        super().__init__()
        self.cfg = cfg
        # Embeddings
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        # Time embedding
        hid = cfg.n_embd if cfg.time_dim is None else cfg.time_dim
        self.time_mlp = nn.Sequential(nn.Linear(2,hid), nn.SiLU(), nn.Linear(hid,cfg.n_embd))
        # Blocks
        self.drop = nn.Dropout(cfg.dropout)
        self.h    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, cfg.bias)
        # Init
        self.apply(self._init_weights)
        print(f"parameters: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")

    # weight init identical to original
    def _init_weights(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            torch.nn.init.normal_(m.weight,mean=0.0,std=self.cfg.init_std)
            if isinstance(m,nn.Linear) and m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    # ── forward ────────────────────────────────────────────────────────────
    def forward(self, idx:torch.Tensor, time_s:torch.Tensor|None=None, time_t:torch.Tensor|None=None):
        """Return v_theta with shape `(B, T, n_embd)`.
        `time_s` / `time_t` can be None to disable conditioning.
        """
        B,T = idx.shape
        if T>self.cfg.block_size:
            raise ValueError(f"seq length {T} > block_size {self.cfg.block_size}")
        device = idx.device
        pos = torch.arange(T, device=device)
        x = self.wte(idx) + self.wpe(pos)              # (B,T,D)
        # Time conditioning
        if time_s is not None and time_t is not None:
            tvec = torch.stack([time_s,time_t],dim=-1).to(device)  # (B,2)
            temb = self.time_mlp(tvec).unsqueeze(1)                # (B,1,D)
            x = x + temb
        x = self.drop(x * self.cfg.mup_input_alpha if self.cfg.mup_enabled else x)
        for blk in self.h:
            x = blk(x)
        v_theta = self.ln_f(x)  # (B,T,D)
        return v_theta

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, adam_eps, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        if self.config.mup_enabled and not self.config.mup_disable_hidden_lr_scaling:
            emb_params  = []
            hidden_ln_params = []
            hidden_weight_params = []
            hidden_bias_params = []
            final_ln_params = []
            for n, p in param_dict.items():
                if n in ('transformer.wte.weight', 'transformer.wpe.weight'):
                    emb_params.append(p)
                elif '.ln_' in n and not '.ln_f.' in n:
                    hidden_ln_params.append(p)
                elif n.endswith('c_attn.weight') or n.endswith('c_fc.weight') or n.endswith('c_proj.weight'):
                    hidden_weight_params.append(p)
                elif n.endswith('c_attn.bias') or n.endswith('c_fc.bias') or n.endswith('c_proj.bias'):
                    hidden_bias_params.append(p)
                elif '.ln_f.' in n:
                    final_ln_params.append(p)
                else:
                    raise Exception(f'Unhandled parameter {n}')
            depth_lr_scaling = (self.config.depth_multiplier ** (self.config.depth_alpha_exp - 1))
            width_lr_scaling = (1 / self.config.mup_width_multiplier)
            if self.config.depth_alpha_enabled:
                ### Begin CompleteP code ###
                adam_eps *= (1 / self.config.mup_width_multiplier) * (self.config.depth_multiplier ** (-1 * self.config.depth_alpha_exp))
                optim_groups = [
                    {
                        'params': emb_params,
                        'weight_decay': weight_decay,
                        'lr_scale': 1.0,
                    },
                    {
                        'params': hidden_ln_params,
                        'weight_decay': 0.0,
                        'lr_scale': depth_lr_scaling,
                    },
                    {
                        'params': hidden_weight_params,
                        'weight_decay': weight_decay / width_lr_scaling,
                        'lr_scale': width_lr_scaling * depth_lr_scaling,
                    },
                    {
                        'params': hidden_bias_params,
                        'weight_decay': 0.0,
                        'lr_scale': depth_lr_scaling,
                    },
                    {
                        'params': final_ln_params,
                        'weight_decay': 0.0,
                        'lr_scale': 1.0,
                    },
                ]
                ### End CompleteP code ###
            else:
                ### Begin muP code ###
                optim_groups = [
                    {
                        'params': emb_params,
                        'weight_decay': weight_decay,
                        'lr_scale': 1.0,
                    },
                    {
                        'params': hidden_ln_params,
                        'weight_decay': 0.0,
                        'lr_scale': 1.0,
                    },
                    {
                        'params': hidden_weight_params,
                        'weight_decay': weight_decay,
                        'lr_scale': width_lr_scaling,
                    },
                    {
                        'params': hidden_bias_params,
                        'weight_decay': 0.0,
                        'lr_scale': 1.0,
                    },
                    {
                        'params': final_ln_params,
                        'weight_decay': 0.0,
                        'lr_scale': 1.0,
                    },
                ]
                ### End muP code ###
        else:
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=adam_eps, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx