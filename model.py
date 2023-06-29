from collections import OrderedDict

import math
import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, n, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n)) if bias else None

    def forward(self, x):
        return nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias)

class MultiheadAttention(nn.Module):

    def __init__(self, n_embd, n_head, dropout, bias):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias = bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias = bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, x):
        B, N, D = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        k = k.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                               dropout_p = self.dropout if self.training else 0,
                                                               is_causal = True)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.resid_dropout(self.c_proj(out))
        return out

def MLP(d_embedding, dropout, bias):

    return nn.Sequential(OrderedDict([
        ("c_fc", nn.Linear(d_embedding, 4 * d_embedding, bias = bias)),
        ("act", nn.GELU(approximate = "tanh")),
        ("c_proj", nn.Linear(4 * d_embedding, d_embedding, bias = bias)),
        ("dropout", nn.Dropout(p = dropout))
    ]))

class Block(nn.Module):

    def __init__(self, d_embedding, n_head, dropout, bias):
        super().__init__()
        self.ln_1 = LayerNorm(d_embedding, bias)
        self.attn = MultiheadAttention(d_embedding, n_head, dropout, bias)
        self.ln_2 = LayerNorm(d_embedding, bias=bias)
        self.mlp = MLP(d_embedding, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_head, dropout, bias, n_layer, device):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd), # codificación de símbolos
            wpe = nn.Embedding(block_size, n_embd), # codificación posicional
            drop = nn.Dropout(p = dropout),
            h = nn.ModuleList([Block(n_embd, n_head, dropout, bias) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_embd, bias)
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias = False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))


    def forward(self, x):
        _, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=self.device).unsqueeze(0)
        tok_emb = self.transformer.wte(x) # representación de los símbolos
        pos_emb = self.transformer.wpe(pos) # representación de las posiciones

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
          x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
