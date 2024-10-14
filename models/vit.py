import torch
import torch.nn as nn
import math

# Define Patch Enbedding Layer

### EXPLAIN ###
'''
- The PatchEmbedding layer converts the image into a sequence of patch embeddings.
- Uses a convolutional layer with kernel and stride equal to the patch size to extract patches.
- The output is reshaped to [batch_size, num_patches, embed_dim].
'''

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super(PatchEmbedding, self).__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.num_patches = num_patches

    def forward(self, x):
        # x shape: [batch_size, in_channels, img_size, img_size]
        x = self.proj(x)  # [batch_size, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        return x
    

### Implement the Positional Encoding ###

'''
- The PositionalEncoding adds positional information to the patch embeddings.
- Uses sine and cosine functions of different frequencies.
- The positional encoding is added to the input embeddings.

'''

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() *
            (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x
    
### Build the Transformer Encoder ###

'''
- Uses PyTorch’s nn.TransformerEncoder and nn.TransformerEncoderLayer.
- Configurable parameters like embedding dimension, number of heads, hidden dimension, and number of layers.
- Applies the Transformer encoder layers to the input sequence.
'''

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=512, num_layers=6, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Added this parameter
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x
    
### Assemble the ViT Model ###

'''
- Patch Embedding: Converts images to patch embeddings.
- Class Token: A learnable embedding representing the whole image.
- Positional Encoding: Added to embeddings to retain positional information.
- Transformer Encoder: Processes the sequence of embeddings.
- Classification Head: Outputs class logits for the input images.

'''



class VisionTransformer(nn.Module):
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
        super(VisionTransformer, self).__init__()

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

        # Transformer Encoder
        self.encoder = TransformerEncoder(
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
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_heads=8,
        hidden_dim=512,
        num_layers=6,
        dropout=0.1
    )

    # Test with a random input tensor
    x = torch.randn(16, 3, 32, 32)  # [batch_size, channels, height, width]
    logits = model(x)
    print(logits.shape)  # Expected output: [16, 10]