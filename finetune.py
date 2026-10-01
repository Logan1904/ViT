import timm
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T
import math
import time
import json
from pathlib import Path
import argparse

from vit import ViT

def main():

    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--decay", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    LR, DECAY = args.lr, args.decay
    torch.manual_seed(args.seed)
    RUN = f"lr{LR:.0e}_decay{DECAY}_s{args.seed}"

    NUM_CLASSES = 37
    EPOCHS, WARMUP_EPOCHS = 10, 1
    WD = 0.05
    OUT = Path("runs"); OUT.mkdir(exist_ok=True)

    # Load pretrained ViT weights
    ref = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True)

    # Imbue weights into our custom built ViT
    model = ViT(num_classes=1000, drop_path=0.1)
    model.load_state_dict(ref.state_dict(), strict=True)

    # Replace the head with a linear layer
    model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)

    # Send model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Load configs used for ref pretraining
    cfg = ref.pretrained_cfg
    mean, std = cfg["mean"], cfg["std"]

    # Train and test transforms
    train_tf = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    test_tf = T.Compose([
        T.Resize(248),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    # Load data
    train_ds = torchvision.datasets.OxfordIIITPet(root="./", split="trainval", transform=train_tf, download=True)
    test_ds  = torchvision.datasets.OxfordIIITPet(root="./", split="test",     transform=test_tf,  download=True)

    # Create dataloaders
    train_dl = DataLoader(train_ds, batch_size=64,  shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    test_dl  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    def param_groups(model, lr, wd, decay):
        """Build AdamW parameter groups with layer-wise lr decay and selective weight decay.

        Inputs:  model ViT, lr float (top lr, used by norm/head), wd float, decay float (1.0 = flat lr)
        Outputs: list[dict] of optimiser param groups
        """
        L = len(model.blocks)
        groups = {}
        for name, p in model.named_parameters():
            # Depth of this parameter: 0 = input embeddings, 1..L = blocks, L+1 = final norm/head
            if name.startswith(("patch_embed", "cls_token", "pos_embed")):
                lid = 0
            elif name.startswith("blocks."):
                lid = int(name.split(".")[1]) + 1          # "blocks.7.attn.qkv.weight" -> 8
            else:
                lid = L + 1

            # Only decay weight matrices; biases, LN params and the learnt tokens are exempt
            use_wd = p.ndim > 1 and name not in ("cls_token", "pos_embed")

            key = (lid, use_wd)
            if key not in groups:
                groups[key] = {"params": [], "weight_decay": wd if use_wd else 0.0,
                            "lr": lr * decay ** (L + 1 - lid)}   # deeper from the top -> smaller lr
            groups[key]["params"].append(p)
        return list(groups.values())

    # Optimizer
    optimizer = torch.optim.AdamW(param_groups(model, LR, WD, DECAY), betas=(0.9, 0.999))

    # LR schedule
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    # Calculate LR factor, warmup and cosine scheduling
    def lr_factor(step):
        """Multiplier on each group's base lr: linear warmup to 1, then cosine to 0."""
        if step < warmup_steps:
            return (step + 1) / warmup_steps                                  # rises linearly from ~0 to 1
        progress = (step - warmup_steps) / (total_steps - warmup_steps)   # fraction of the post-warmup phase done, in [0, 1]
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    # Training Loop
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    print(amp_dtype)

    @torch.no_grad()
    def evaluate(model, dl):
        model.eval()
        correct = total = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.numel()
        return correct / total

    acc0 = evaluate(model, test_dl)
    print(f"epoch 0  test acc {acc0:.3f}")   # sanity: expect ~0.027

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()                               # per iteration, after the optimiser step
        acc = evaluate(model, test_dl)
        history.append(acc)
        print(f"epoch {epoch}  loss {loss.item():.3f}  test acc {acc:.3f}  ({time.time() - t0:.0f}s)")

    # Accuracy per epoch, index 0 = before training, so curves start from the same point
    json.dump({"run": RUN, "lr": LR, "decay": DECAY, "acc": [acc0] + history},
            open(OUT / f"{RUN}.json", "w"), indent=2)
    torch.save(model.state_dict(), OUT / f"{RUN}.pt")

if __name__ == "__main__":
    main()