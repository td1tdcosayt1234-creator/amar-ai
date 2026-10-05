"""generate.py - train করা model দিয়ে text বানাও (streaming + sampling controls).

Usage:  python generate.py "prompt" [temperature] [max_tokens] [ckpt] [top_k] [top_p]
Example: python generate.py "আমি একটি" 0.8 200 model_bangla.pt 40 0.9
"""
import sys
import time

import torch
import torch.nn.functional as F

from mini_gpt import MiniGPT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def sample_next(model, idx, temperature, top_k, top_p):
    idx_cond = idx[:, -model.block_size:]
    logits, _ = model(idx_cond)
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
    if top_p is not None:
        sl, si = torch.sort(logits, descending=True)
        cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
        rem = cp > top_p
        rem[:, 1:] = rem[:, :-1].clone()
        rem[:, 0] = False
        logits[0, si[0][rem[0]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def main():
    ckpt_path = sys.argv[4] if len(sys.argv) > 4 else "model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    stoi, itos = ckpt["stoi"], ckpt["itos"]

    model = MiniGPT(**cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt = sys.argv[1] if len(sys.argv) > 1 else ""
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    top_k = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    top_p = float(sys.argv[6]) if len(sys.argv) > 6 else 0.9

    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
    sys.stdout.write(prompt)
    sys.stdout.flush()
    with torch.no_grad():
        for _ in range(max_new):
            nxt = sample_next(model, idx, temperature, top_k, top_p)
            idx = torch.cat((idx, nxt), dim=1)
            ch = itos[nxt.item()]
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(0.01)
    print()


if __name__ == "__main__":
    main()
