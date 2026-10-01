import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbed(nn.Module):
    def __init__(self, patch=16, dim=384):
        super().__init__()

        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, x):   # x: (B, 3, H, W)
        x = self.proj(x)    # (B, dim, H/p, W/p)

        x = x.permute(0, 2, 3, 1)   # (B, H/p, W/p, dim)
        x = x.reshape(x.shape[0], -1, x.shape[3])   # (B, N, dim)

        return x

class Attention(nn.Module):
    def __init__(self, dim=384, heads=6):
        super().__init__()

        self.h = heads
        self.dh = dim // heads

        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)

        self.keep_attn = False
        self.attn = None

    def forward(self, x):   # x: (B, N, d)
        B, N, d = x.shape

        qkv = self.qkv(x)   # (B, N, 3d)

        qkv = qkv.reshape(B, N, 3, self.h, self.dh)    # (B, N, 3, h, dh)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, h, N, dh)
        q, k, v = qkv.unbind(dim=0) # (B, h, N, dh)

        if self.keep_attn:
            a = (q @ k.transpose(-2,-1)) * self.dh ** -0.5  # (B, h, N, N)
            a = a.softmax(dim=-1)   # (B, h, N, N)
            self.attn = a
            y = a @ v   # (B, h, N, dh)
        else:
            y = F.scaled_dot_product_attention(q, k, v) # (B, h, N, dh)

        y = y.transpose(1, 2)  # (B, N, h ,dh)
        y = y.reshape(B, N, d)

        return self.proj(y)

class Mlp(nn.Module):
    def __init__(self, dim=384):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim*4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim*4, dim)

    def forward(self, x):   # x: (B, N, d)
        x = self.fc1(x) # (B, N, 4d)
        x = self.act(x) # (B, N, 4d)
        x = self.fc2(x) # (B, N, d)

        return x

class Block(nn.Module):
    def __init__(self, dim=384, heads=6, drop_path=0.0):
        super().__init__()

        self.dp = drop_path

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

    def _drop(self, y): # y: branch output (B, N, d)
        if not self.training or self.dp == 0:
            return y
        
        keep = torch.rand(y.shape[0], 1, 1, device=y.device) < (1-self.dp)
        return y * keep / (1-self.dp)


    def forward(self, x):   # x: (B, N, d)
        x = x + self._drop(self.attn(self.norm1(x)))    # (B, N, d)
        x = x + self._drop(self.mlp(self.norm2(x)))     # (B, N, d)
        return x

class ViT(nn.Module):
    def __init__(self, img=224, patch=16, dim=384, depth=12, heads=6, num_classes=1000, drop_path=0.0):
        super().__init__()

        self.patch_embed = PatchEmbed(patch, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1+(img//patch)**2, dim)*0.02)

        drop_path_rates = torch.linspace(0, drop_path, depth)
        self.blocks = nn.ModuleList([Block(dim, heads, drop_path_rates[i]) for i in range(depth)])

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)

    def forward_features(self, x):  # (B, 3, H, W)
        x = self.patch_embed(x)     # (B, N, d)
        cls = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, d)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, d)

        x = x + self.pos_embed          # (B, N+1, d)

        for block in self.blocks:
            x = block(x)                # (B, N+1, d)

        x = self.norm(x)                # (B, N+1, d)

        return x

    def forward(self, x):   # (B, 3, H, W)
        x = self.forward_features(x)    # (B, N+1, d)
        x = self.head(x[:,0,:])         # (B, num_classes)
        return x

def resize_pos_embed(pos_embed, new_grid):
    d = pos_embed.shape[2]

    cls_pos = pos_embed[:, :1]
    patch_pos = pos_embed[:, 1:]

    g = math.isqrt(patch_pos.shape[1])

    patch_pos = patch_pos.reshape(1, g, g, d)
    patch_pos = patch_pos.permute(0, 3, 1, 2)

    patch_pos = F.interpolate(patch_pos, size=(new_grid,new_grid), mode='bicubic', align_corners=False)

    patch_pos = patch_pos.permute(0, 2, 3, 1)
    patch_pos = patch_pos.reshape(1, new_grid*new_grid, d)

    pos_embed = torch.cat([cls_pos, patch_pos], dim=1)

    return pos_embed

@torch.no_grad()
def attention_rollout(model, x):    # (B, 3, H, W) -> (B, g, g)
    # Switch every block to the explicit path so attention gets stored
    for block in model.blocks:
        block.attn.keep_attn = True

    model(x)    # fills block.attn.attn: (B, h, N, N) per block

    # Per-layer
    roll = None
    for block in model.blocks:
        a = block.attn.attn.mean(dim=1)
        eye = torch.eye(a.shape[-1], device=a.device)
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(dim=-1, keepdim=True)
        roll = a if roll is None else a @ roll

    # CLS row, drop the CLS -> CLS entry, back onto the patch grid
    cls_row = roll[:, 0, 1:]    # (B, N-1)
    g = math.isqrt(cls_row.shape[1])
    heat = cls_row.reshape(cls_row.shape[0], g, g)

    # Restore the fast path and free the stored matrices
    for block in model.blocks:
        block.attn.keep_attn = False
        block.attn.attn = None

    return heat



