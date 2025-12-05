import torch 
import torch.nn as nn


class PatchEmbedder(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim=embed_dim
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, image):
        patches = self.conv(image)
        patches = patches.flatten(2, 3).permute(0, 2, 1)
        return patches
    

class LinearProjection(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_embedder = PatchEmbedder(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))    
        num_patches = (image_size ** 2) // (patch_size ** 2)
        self.positional_embeddings = nn.Parameter(
            torch.normal(mean=0, std=0.02, size=(1, num_patches + 1, embed_dim))
        )

    def forward(self, image):
        patches = self.patch_embedder(image)
        cls_tokens = self.cls_token.repeat((image.size(0), 1, 1))
        patches = torch.cat([cls_tokens, patches], dim=1)
        return patches + self.positional_embeddings
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.head_dim = self.embed_dim // self.num_heads
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** (-0.5)
        self.attn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        Q = Q.view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        scores = Q @ K.mT
        attn = torch.softmax(scores * self.scale, dim=-1)
        context = self.attn_dropout(attn @ V) 
        context = context.permute(0, 2, 1, 3).contiguous().view(x.shape[0], -1, self.embed_dim)
        return context


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim=3072, dropout_rate=0.0):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.act(self.in_proj(x)))
        x = self.dropout(self.act(self.out_proj(x)))
        return x
    

class TransofmerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim=3072):
        super().__init__()
        self.pre_ln = nn.LayerNorm(embed_dim)
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim, mlp_hidden_dim)
        self.post_ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = residual + self.pre_ln(self.mhsa(x))
        residual = x
        x = residual + self.mlp(self.post_ln(x))
        return x
    

class ViT(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        mlp_hidden_size=512,
        num_transformer_blocks=12,
        num_heads=12,
        n_classes=10
    ):
        super().__init__()
        self.projection = LinearProjection(image_size, patch_size, in_channels, embed_dim)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_transformer_blocks):
            block = TransofmerEncoderBlock(embed_dim, num_heads, mlp_hidden_size)
            self.transformer_blocks.append(block)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_classes)
        )

    def forward(self, image):
        patches = self.projection(image)
        for block in self.transformer_blocks:
            patches = block(patches)
        cls_token = patches[:, 0, :]
        logits = self.classifier(cls_token)
        return logits   
