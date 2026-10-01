import timm, torch, torch.nn.functional as F, torchvision
import matplotlib.pyplot as plt
from torchvision import transforms as T
from vit import ViT, attention_rollout

RUN = "lr3e-04_decay0.7_s0"                 # the fine-tuned run to inspect
IDX = [0, 500, 1500, 3000]               # fixed test images, same for every plot

ref = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
before = ViT(); before.load_state_dict(ref.state_dict())            # pretrained, 1000-class head
after = ViT(num_classes=37); after.load_state_dict(torch.load(f"runs/{RUN}.pt", map_location="cpu"))
before.eval(); after.eval()                                         # rollout ignores the head, so head size doesn't matter

tf = T.Compose([T.Resize(248), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
ds = torchvision.datasets.OxfordIIITPet(root="data", split="test", transform=tf, download=True)
x = torch.stack([ds[i][0] for i in IDX])                            # (4, 3, 224, 224)

def overlay_maps(model):
    heat = attention_rollout(model, x)                              # (4, 14, 14)
    heat = F.interpolate(heat[:, None], size=224, mode="bilinear")[:, 0]
    return heat / heat.amax(dim=(1, 2), keepdim=True)               # per-image [0, 1] for display

imgs = (x * 0.5 + 0.5).permute(0, 2, 3, 1)                          # undo Normalize for display
fig, ax = plt.subplots(3, len(IDX), figsize=(3 * len(IDX), 9))
for row, (title, maps) in enumerate([("image", None), ("pretrained", overlay_maps(before)), ("fine-tuned", overlay_maps(after))]):
    for c in range(len(IDX)):
        ax[row, c].imshow(imgs[c])
        if maps is not None:
            ax[row, c].imshow(maps[c], cmap="jet", alpha=0.5)
        ax[row, c].set_axis_off()
    ax[row, 0].set_title(title, loc="left")
plt.savefig(f"runs/rollout_{RUN}.png", dpi=150, bbox_inches="tight")