"""train.py - model কে data দিয়ে শেখাও (improved version).

CPU তে চলবে। বড় করতে MAX_ITERS বাড়াও।
"""
import math
import os

import torch

from mini_gpt import MiniGPT

# ---- settings ----
DATA_FILE = os.environ.get("DATA_FILE", "data.txt")
OUT_FILE = os.environ.get("OUT_FILE", "model.pt")
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", 256))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
N_EMBD = int(os.environ.get("N_EMBD", 512))
N_HEAD = int(os.environ.get("N_HEAD", 8))
N_LAYER = int(os.environ.get("N_LAYER", 8))
DROPOUT = 0.1
MAX_ITERS = int(os.environ.get("MAX_ITERS", 20000))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 1000))
LEARNING_RATE = 3e-4
WARMUP_ITERS = 200
GRAD_CLIP = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- data ----
text = open(DATA_FILE, encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
vocab_size = len(chars)
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(10):
            xb, yb = batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def get_lr(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    progress = (it - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    torch.manual_seed(1337)
    model = MiniGPT(vocab_size, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={DEVICE}, vocab={vocab_size}, params={n_params/1e6:.2f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

    for it in range(MAX_ITERS):
        lr = get_lr(it)
        for g in optim.param_groups:
            g["lr"] = lr

        xb, yb = batch("train")
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()

        if it % EVAL_EVERY == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model)
            print(f"iter {it:5d}: lr {lr:.2e}, train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")

    torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos,
                "config": dict(vocab_size=vocab_size, block_size=BLOCK_SIZE, n_embd=N_EMBD,
                               n_head=N_HEAD, n_layer=N_LAYER)}, OUT_FILE)
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
