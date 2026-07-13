"""Modern small-LM architecture for the Pashto SLM (v2).

Upgrades the nanoGPT/GPT-2 model in `model.py` to the 2024-2025 SLM consensus
used by SmolLM2, MobileLLM, TinyLlama and the Kazakh SozKZ model:

  * RoPE   -- rotary positional embeddings (no learned `wpe`); better + longer ctx
  * RMSNorm instead of LayerNorm
  * SwiGLU MLP instead of GELU
  * GQA    -- grouped-query attention (n_kv_head < n_head) -> smaller KV, faster
  * tied embeddings, bias-free linears
  * depth-over-width friendly (MobileLLM finding for sub-billion models)

Kept as a SEPARATE file so the original `model.py` and its checkpoints are
untouched. Same GPTConfig-style interface + .generate() so the training loop is
unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModernGPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    n_kv_head: int = 4          # GQA: n_head must be divisible by n_kv_head
    dropout: float = 0.0
    bias: bool = False          # modern LMs drop biases
    rope_theta: float = 10000.0
    mlp_multiple_of: int = 256  # round SwiGLU hidden to this


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # compute in fp32 for stability, cast back
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt)) * self.weight


def _build_rope(head_dim, max_seq, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)                 # [T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)          # [T, hd]
    return emb.cos(), emb.sin()                      # each [T, hd]


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    # x: [B, H, T, hd]; cos/sin: [T, hd]
    T = x.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.hd = config.n_embd // config.n_head
        out = (config.n_head + 2 * config.n_kv_head) * self.hd
        self.c_attn = nn.Linear(config.n_embd, out, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(
            [self.n_head * self.hd, self.n_kv_head * self.hd,
             self.n_kv_head * self.hd], dim=-1)
        q = q.view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if self.n_kv_head != self.n_head:            # expand KV for GQA
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(2 * (4 * config.n_embd) / 3)    # 8/3 * n_embd
        m = config.mlp_multiple_of
        hidden = m * ((hidden + m - 1) // m)
        self.w_gate = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w_down = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n1 = RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.n2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class ModernGPT(nn.Module):
    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight        # tied

        cos, sin = _build_rope(config.n_embd // config.n_head,
                               config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("w_down.weight") or pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"seq {T} > block {self.config.block_size}"
        x = self.drop(self.wte(idx))
        cos, sin = self.rope_cos.to(x.device), self.rope_sin.to(x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
            return logits, loss
        return self.lm_head(x[:, [-1], :]), None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx
