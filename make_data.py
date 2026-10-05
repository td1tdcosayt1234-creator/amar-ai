"""Build a larger data.txt (~10000 lines) from the seed corpus."""
import random

random.seed(7)

seed_lines = [l.strip() for l in open("data.txt", encoding="utf-8").read().splitlines() if l.strip()]

out = []
while len(out) < 10000:
    line = random.choice(seed_lines)
    if random.random() < 0.3:
        line = line + " " + random.choice(seed_lines)[:40]
    out.append(line)

with open("data.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")

print("lines:", len(out))
