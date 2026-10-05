"""generate.py - train করা model দিয়ে text বানাও।

Usage:  python generate.py "পরে লেখা শুরু" [temperature] [max_tokens]
Example: python generate.py "আমি একটি" 0.8 200
"""
import sys

import torch

from mini_gpt import MiniGPT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHECKPOINT = "model.pt"


def main():
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    stoi, itos = ckpt["stoi"], ckpt["itos"]

    model = MiniGPT(**cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt = sys.argv[1] if len(sys.argv) > 1 else ""
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
    out = model.generate(idx, max_new, temperature=temperature, top_k=40, top_p=0.9)[0].tolist()
    print("".join(itos[i] for i in out))


if __name__ == "__main__":
    main()
