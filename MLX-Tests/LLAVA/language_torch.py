import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TextConfig:
    model_type: str
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 32000
    num_key_value_heads: int = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")

class Attention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        head_dim = dim // n_heads
        self.scale = (head_dim ** -0.5)

        self.q_proj = nn.Linear(dim, n_heads * head_dim)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim)
        self.o_proj = nn.Linear(n_heads * head_dim, dim)

    def forward(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, -1).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, -1).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, -1).permute(0, 2, 1, 3)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=-2)
            v = torch.cat([cache[1], v], dim=-2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(context), (k, v)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, mask=None, cache=None):
        x = self.norm1(x)
        x, new_cache = self.self_attn(x, mask, cache)
        x = x + self.mlp(self.norm2(x))
        return x, new_cache

class Llama(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, inputs, cache=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(inputs)
        mask = None
        for layer in self.layers:
            inputs_embeds, cache = layer(inputs_embeds, mask, cache)
        return self.norm(inputs_embeds), cache

class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.model = Llama(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, inputs, cache=None, inputs_embeds=None):
        out, cache = self.model(inputs, cache, inputs_embeds)
        return self.lm_head(out), cache
