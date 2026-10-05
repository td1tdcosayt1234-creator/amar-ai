"""Build a diverse, varied data.txt (~12000 lines) in Bengali + English."""
import itertools
import random

random.seed(42)

subjects_bn = ["আমি", "তুমি", "সে", "আমরা", "তোমরা", "তারা", "গাছ", "নদী", "সূর্য", "চাঁদ", "পাখি", "মানুষ", "ছাত্র", "শিক্ষক", "ডাক্তার", "শিশু"]
verbs_bn = ["যাই", "আসি", "খায়", "খেলি", "লিখি", "পড়ি", "গান গাই", "ঘুমাই", "জিতে যায়", "পড়ে", "বলে"]
places_bn = ["স্কুলে", "বাজারে", "মাঠে", "ঘরে", "অফিসে", "নদীতে", "পাহাড়ে", "বাগানে", "রাস্তায়"]
adjs_bn = ["নীল", "সবুজ", "লাল", "বড়", "ছোট", "সুন্দর", "আলো", "শান্ত", "বড়ো", "মিষ্টি"]

subjects_en = ["I", "You", "We", "They", "The boy", "The girl", "The teacher", "The river", "The sun", "The dog", "The cat", "My friend", "The student"]
verbs_en = ["go to", "play in", "write", "read", "see", "build", "learn from", "enjoy", "drink from", "run through"]
places_en = ["the park", "the market", "the school", "the home", "the office", "the field", "the river", "the mountain", "the garden", "the street"]

templates_bn = [
    "{s} {v} {p}।",
    "আজ {p} {a} আবহাওয়া।",
    "{s} মনে করে {a} কথা।",
    "প্রতিদিন {s} {v} {p}।",
    "{p} এর মধ্যে {s} থাকে।",
]
templates_en = [
    "{s} {v} {p} today.",
    "The weather is {a} in {p}.",
    "{s} often {v} {p}.",
    "Every morning {s} {v} {p}.",
    "There is a {a} light over {p}.",
]
adjs_en = ["blue", "green", "red", "big", "small", "beautiful", "bright", "quiet", "warm", "sweet"]

lines = []
for s, v, p, a in itertools.product(subjects_bn, verbs_bn, places_bn, adjs_bn[:4]):
    lines.append(random.choice(templates_bn).format(s=s, v=v, p=p, a=a))
for s, v, p, a in itertools.product(subjects_en, verbs_en, places_en, adjs_en[:4]):
    lines.append(random.choice(templates_en).format(s=s, v=v, p=p, a=a))

extra = [
    "বৈচিত্র্যশীল data দিয়ে model ভালো হয়।", "Machine learning is fun.",
    "Transformer নেটওয়ার্ক শক্তিশালী।", "Neural networks learn patterns.",
    "Dataset, model, এবং training গুরুত্বপূর্ণ।", "Training takes time on CPU.",
    "পাহাড়ের সৌন্দর্য বর্ণনাতীত।", "Coding opens new doors.",
    "ভোরের আলো সুন্দর লাগে।", "Practice makes progress.",
]
lines.extend(extra * 20)
random.shuffle(lines)
with open("data.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines[:12000]) + "\n")
print("lines:", min(len(lines), 12000))
