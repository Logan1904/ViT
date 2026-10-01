import torch
import timm
import torch.nn as nn

from vit import PatchEmbed, Attention, Block, ViT, resize_pos_embed, attention_rollout

m = timm.create_model("vit_small_patch16_224", pretrained=False)

pe = PatchEmbed()
x = torch.randn(2, 3, 224, 224)
print(pe(x).shape)                       # expect (2, 196, 384)
pe.load_state_dict(m.patch_embed.state_dict())
torch.testing.assert_close(pe(x), m.patch_embed(x))

blk = m.blocks[0]
attn = Attention(384, 6)
attn.load_state_dict(blk.attn.state_dict())   # q_norm/k_norm are Identity, so no extra keys
x = torch.randn(2, 197, 384)
torch.testing.assert_close(attn(x), blk.attn(x))
attn.keep_attn = True
torch.testing.assert_close(attn(x), blk.attn(x))
print(attn.attn.shape, attn.attn.sum(-1)[0, 0, :3])   # expect (2, 6, 197, 197), rows sum to 1

blk = Block(384, 6)
blk.load_state_dict(m.blocks[0].state_dict())
x = torch.randn(2, 197, 384)
m.eval(); blk.eval()
torch.testing.assert_close(blk(x), m.blocks[0](x))

ref = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=10).eval()
mine = ViT(num_classes=10).eval()
mine.load_state_dict(ref.state_dict(), strict=True)
x = torch.randn(2, 3, 224, 224)
torch.testing.assert_close(mine(x), ref(x), atol=1e-4, rtol=1e-4)

from timm.layers import resample_abs_pos_embed
pe = torch.randn(1, 197, 384)
out = resize_pos_embed(pe, 24)
print(out.shape)                                           # (1, 577, 384)
torch.testing.assert_close(out[:, :1], pe[:, :1])          # CLS untouched
ref = resample_abs_pos_embed(pe, (24, 24), num_prefix_tokens=1, antialias=False)
torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

model = ViT()
model.pos_embed = nn.Parameter(resize_pos_embed(model.pos_embed.data, 24))
print(model(torch.randn(1, 3, 384, 384)).shape)            # (1, 1000)

model = ViT().eval()
heat = attention_rollout(model, torch.randn(2, 3, 224, 224))
print(heat.shape, heat.min() >= 0)                    # (2, 14, 14), True
print(model.blocks[0].attn.keep_attn, model.blocks[0].attn.attn)   # False, None