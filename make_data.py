"""Build diverse data.txt for a given subject.

Usage: python make_data.py coding
       python make_data.py bangla
       python make_data.py english
"""
import random
import sys

random.seed(42)

SUBJECTS = {
    "coding": [
        "print('hello world')",
        "def add(a, b):\n    return a + b",
        "for i in range(10):\n    print(i)",
        "class Model:\n    def __init__(self):\n        pass",
        "import torch\nimport torch.nn as nn",
        "x = torch.tensor([1, 2, 3])",
        "model = MiniGPT()\nmodel.eval()\noutput = model.generate(idx, 100)",
        "loss = F.cross_entropy(logits, targets)",
        "python train.py  # start training",
        "def main():\n    pass\n\nif __name__ == '__main__':\n    main()",
        "git add . && git commit -m 'update'",
        "result = [x**2 for x in range(5)]",
        "try:\n    f = open('data.txt')\nexcept FileNotFoundError:\n    print('missing')",
        "# machine learning with python",
        "while True:\n    break",
    ],
    "bangla": [
        "আমি বাংলায় গান গাই।",
        "আজকের আকাশ নীল।",
        "নদী বয়ে যায় পাড় ছুঁয়ে।",
        "শিশুরা মাঠে খেলছে সারাদিন।",
        "শিক্ষক পড়াচ্ছেন নতুন কবিতা।",
        "ভাত খিচুড়ি রান্না হয়েছে।",
        "বৃষ্টি পড়ছে রিমঝিম।",
        "গাছে গাছে পাখি গান গায়।",
        "ভোরের আলো চোখে লাগে।",
        "বন্ধুরা মিলে গল্প করে।",
        "ফুল ফুটেছে বাগানে।",
        "চাঁদ উঠেছে রাতের আকাশে।",
        "বাজার থেকে আনা হয়েছে শাকসবজি।",
        "প্রভাতফেরি বের হয় সকালে।",
        "কবিতা লেখা যায় মেঘ দেখে।",
    ],
    "english": [
        "The quick brown fox jumps over the lazy dog.",
        "I love programming and building things.",
        "The sun rises in the east.",
        "She sells sea shells by the sea shore.",
        "A journey of a thousand miles begins with a single step.",
        "Machine learning models learn from data.",
        "The cat sat on the mat.",
        "Practice makes perfect.",
        "Knowledge is power.",
        "The rain in Spain stays mainly in the plain.",
        "To be or not to be, that is the question.",
        "Every cloud has a silver lining.",
        "Actions speak louder than words.",
        "The early bird catches the worm.",
        "Where there is a will, there is a way.",
    ],
}


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "bangla"
    if subject not in SUBJECTS:
        print(f"unknown subject: {subject}. choose from {list(SUBJECTS)}")
        sys.exit(1)
    base = SUBJECTS[subject]
    lines = []
    for i in range(8000):
        a = random.choice(base)
        if random.random() < 0.25:
            b = random.choice(base)
            a = a + " " + b
        lines.append(a)
    with open(f"data_{subject}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote data_{subject}.txt, lines: {len(lines)}")


if __name__ == "__main__":
    main()
