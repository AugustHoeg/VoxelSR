import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, is_causal: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=self.is_causal,
        )
        return self.proj(x.transpose(1, 2).contiguous().view(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, is_causal: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout=dropout, is_causal=is_causal)
        self.ln2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VQTransformer3D(nn.Module):
    """GPT-style autoregressive transformer over VQ codebook indices.

    Sequence length is inferred at runtime — positional embeddings are
    learned and allocated up to max_seq_len (set this >= the longest
    sequence you will encounter across all encoder configurations).

    Set is_causal=False to switch to bidirectional attention (e.g. MaskGIT).
    The SOS token occupies index num_embeddings (one past the codebook).
    """

    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        seq_len: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        is_causal: bool = True,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.seq_len = seq_len

        # +1 for the SOS token (index = num_embeddings)
        self.tok_emb = nn.Embedding(num_embeddings + 1, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout, is_causal=is_causal)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_embeddings, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Plain forward pass: embed idx directly, run transformer blocks, project to logits.

        Teacher-forced (autoregressive) inputs should be prepared by the caller —
        see ModelTransformerVQ which prepends the SOS token before calling this.

        Args:
            idx: (B, T) token indices
        Returns:
            logits: (B, T, num_embeddings)
        """
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))  # (B, T, num_embeddings)

if __name__ == '__main__':
    # Quick sanity check: 64-token sequences (e.g. 4^3 from a 64^3 patch + 16x encoder)
    model = VQTransformer3D(num_embeddings=512, embed_dim=512, depth=12, num_heads=8, seq_len=64)
    idx = torch.randint(0, 512, (2, 64))
    logits = model(idx)
    assert logits.shape == (2, 64, 512), logits.shape
    print("GPT3D OK — logits:", logits.shape)
