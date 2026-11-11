"""
model_192.py — BitNet Transformer con MQA y PLE para contexto de 192 tokens.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        scale = w.abs().mean().clamp(min=1e-8)
        w_ternary = torch.clamp(torch.round(w / scale), -1, 1)
        w_quant = w_ternary * scale
        w_ste = w + (w_quant - w).detach()
        return F.linear(x, w_ste, self.bias)


class MQA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads=1, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.q_proj = BitLinear(embed_dim, embed_dim)
        self.k_proj = BitLinear(embed_dim, self.kv_dim)
        self.v_proj = BitLinear(embed_dim, self.kv_dim)
        self.out_proj = BitLinear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.num_kv_heads != self.num_heads:
            k = k.expand(B, self.num_heads, T, self.head_dim)
            v = v.expand(B, self.num_heads, T, self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = BitLinear(embed_dim, ff_dim)
        self.fc2 = BitLinear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class PLE(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, ple_hidden_dim):
        super().__init__()
        self.num_layers = num_layers
        self.ple_hidden_dim = ple_hidden_dim
        self.embed_tokens_per_layer = nn.Embedding(vocab_size, num_layers * ple_hidden_dim)
        self.per_layer_model_projection = BitLinear(embed_dim, num_layers * ple_hidden_dim, bias=False)
        self.scale = math.sqrt(ple_hidden_dim)

    def forward(self, token_ids, base_embeds):
        B, T = token_ids.shape
        ple_table = self.embed_tokens_per_layer(token_ids)
        ple_proj = self.per_layer_model_projection(base_embeds)
        combined = (ple_table + ple_proj) * self.scale
        return combined.view(B, T, self.num_layers, self.ple_hidden_dim)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads, ff_dim, ple_hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MQA(embed_dim, num_heads, num_kv_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.ple_to_residual = BitLinear(ple_hidden_dim, embed_dim, bias=False)
        self.ple_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, ple_slice, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        ple_out = self.ple_norm(self.ple_to_residual(ple_slice))
        x = x + ple_out
        return x


class BitNetTransformer(nn.Module):
    def __init__(self, vocab_size=2000, embed_dim=192, num_heads=4, num_kv_heads=1,
                 num_layers=6, ff_dim=384, max_seq_len=192,
                 ple_hidden_dim=320, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.ple_hidden_dim = ple_hidden_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.ple = PLE(vocab_size, embed_dim, num_layers, ple_hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, num_kv_heads, ff_dim, ple_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(embed_dim)
        self.lm_head = BitLinear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, mask=None):
        B, T = idx.shape
        if mask is None:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1)
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.token_embed(idx) + self.pos_embed(pos)
        x = self.dropout(x)
        ple_slices = self.ple(idx, x)
        for i, block in enumerate(self.blocks):
            x = block(x, ple_slices[:, :, i, :], mask)
        return self.lm_head(self.norm_final(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
