import inspect
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class VisionConfig:
    model_type: str
    num_hidden_layers: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    image_size: int = 336
    patch_size: int = 14
    projection_dim: int = 768
    vocab_size: int = 32000
    num_channels: int = 3
    layer_norm_eps: float = 1e-5

    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})

class Attention(nn.Module):
    def __init__(self, dims, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim ** -0.5

        self.qkv_proj = nn.Linear(dims, 3 * dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, -1 // self.num_heads).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        attn = (queries @ keys.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)

        out = (attn @ values).transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        return self.fc2(x)

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config.hidden_size, config.num_attention_heads)
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, mask=None):
        y = self.layernorm1(x)
        y = self.self_attn(y, mask=mask)
        x = x + y
        y = self.layernorm2(x)
        y = self.mlp(y)
        return x + y

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.position_embedding = nn.Parameter(torch.zeros((config.image_size // config.patch_size) ** 2 + 1, config.hidden_size))
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embedding(x)  # [B, hidden_size, grid, grid]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_size]
        cls_tokens = self.class_embedding.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, hidden_size]
        x += self.position_embedding
        return x

class ClipVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, output_hidden_states=False):
        x = self.embeddings(x)
        x = self.encoder(x)
        return self.layernorm(x[:, 0]), x

class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model_type != 'clip_vision_model':
            raise ValueError(f'Unsupported model type: {config.model_type}')
        self.vision_model = ClipVisionModel(config)

    def forward(self, x, output_hidden_states=False):
        return self.vision_model(x, output_hidden_states=output_hidden_states)

