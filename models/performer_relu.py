import torch
import torch.nn as nn
# Include necessary imports
from models.vit import PatchEmbedding, PositionalEncoding


def relu_kernel_feature_map(x):
    return torch.nn.functional.relu(x)

class PerformerAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(PerformerAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size, seq_length, embed_dim = query.size()

        # Linear projections
        Q = self.query_proj(query)  # [batch_size, seq_length, embed_dim]
        K = self.key_proj(key)
        V = self.value_proj(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Apply kernel feature map
        Q = relu_kernel_feature_map(Q)  # [batch_size, seq_length, num_heads, head_dim]
        K = relu_kernel_feature_map(K)

        # Compute attention (linear complexity)
        KV = torch.einsum('blhd,blhe->bhde', K, V)  # [batch_size, num_heads, head_dim, head_dim]
        QKV = torch.einsum('blhd,bhde->blhe', Q, KV)  # [batch_size, seq_length, num_heads, head_dim]

        # Reshape and project out
        QKV = QKV.contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out_proj(QKV)

        return output

class PerformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(PerformerEncoderLayer, self).__init__()
        self.self_attn = PerformerAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src):
        # Self-attention
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src
    
class PerformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(PerformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            PerformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src
    

class PerformerReLUTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_heads=8,
        hidden_dim=512,
        num_layers=6,
        dropout=0.1
    ):
        super(PerformerReLUTransformer, self).__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional Encoding
        self.pos_embed = PositionalEncoding(embed_dim=embed_dim, max_len=num_patches+1)
        self.dropout = nn.Dropout(dropout)

        # Performer Encoder
        self.encoder = PerformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Classification Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, num_patches + 1, embed_dim]

        x = self.pos_embed(x)
        x = self.dropout(x)

        x = self.encoder(x)  # [batch_size, num_patches + 1, embed_dim]

        cls_output = x[:, 0]  # [batch_size, embed_dim]
        logits = self.mlp_head(cls_output)  # [batch_size, num_classes]
        return logits
    

if __name__ == "__main__":
    # Test the model
    pass


