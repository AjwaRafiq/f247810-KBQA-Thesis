import json
import os
import random

# ── paths ──────────────────────────────────────────────────────────────────
RAW_TRAIN  = "/home/f247810/BertT5-LLaMA-KBQA/data/webqsp/WebQSP.train.json"
RAW_TEST   = "/home/f247810/BertT5-LLaMA-KBQA/data/webqsp/WebQSP.test.json"
OUT_DIR    = "/home/f247810/KBQA_WebQSP_improve/data/processed"
SIZE_FILE  = "/home/f247810/KBQA_WebQSP_improve/results/dataset_sizes.txt"

TRAIN_RATIO = 0.90
SEED        = 42

# ── helpers ────────────────────────────────────────────────────────────────
def extract_sample(q):
    """
    Extract one clean record from a WebQSP question entry.
    Uses first parse that has Answers and an InferentialChain.
    Falls back to first parse if none qualifies.
    """
    question_id   = q["QuestionId"]
    question_text = q["ProcessedQuestion"].strip()

    chosen_parse = None
    for p in q["Parses"]:
        if p.get("Answers") and p.get("InferentialChain"):
            chosen_parse = p
            break
    if chosen_parse is None:
        chosen_parse = q["Parses"][0]

    sparql          = chosen_parse.get("Sparql", "").strip()
    inferential     = chosen_parse.get("InferentialChain") or []
    topic_mid       = chosen_parse.get("TopicEntityMid", "")
    topic_name      = chosen_parse.get("TopicEntityName", "")

    # all answer entity names  (keep MID as fallback)
    answers = []
    for a in chosen_parse.get("Answers", []):
        name = a.get("EntityName") or a.get("AnswerArgument", "")
        if name:
            answers.append(name)

    # constraints
    constraints = []
    for c in chosen_parse.get("Constraints", []):
        constraints.append({
            "predicate" : c.get("NodePredicate", ""),
            "entity"    : c.get("EntityName", ""),
            "operator"  : c.get("Operator", "")
        })

    # all valid parses (for richer Llama supervision)
    all_parses = []
    for p in q["Parses"]:
        if p.get("Answers") and p.get("InferentialChain") and p.get("Sparql"):
            all_parses.append({
                "sparql"          : p["Sparql"].strip(),
                "inferential_chain": p["InferentialChain"],
                "answers"         : [
                    a.get("EntityName") or a.get("AnswerArgument", "")
                    for a in p["Answers"]
                ]
            })

    return {
        "question_id"       : question_id,
        "question"          : question_text,
        "sparql"            : sparql,
        "inferential_chain" : inferential,
        "topic_entity_mid"  : topic_mid,
        "topic_entity_name" : topic_name,
        "answers"           : answers,
        "constraints"       : constraints,
        "all_parses"        : all_parses
    }

def load_and_extract(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    questions = raw["Questions"]
    samples = []
    skipped = 0
    for q in questions:
        if not q.get("Parses"):
            skipped += 1
            continue
        samples.append(extract_sample(q))
    return samples, skipped

# ── load ───────────────────────────────────────────────────────────────────
print("Loading train file ...")
train_all, train_skipped = load_and_extract(RAW_TRAIN)
print(f"  Loaded  : {len(train_all)}  |  Skipped (no parses): {train_skipped}")

print("Loading test file ...")
test_data, test_skipped = load_and_extract(RAW_TEST)
print(f"  Loaded  : {len(test_data)}  |  Skipped (no parses): {test_skipped}")

# ── train / dev split ──────────────────────────────────────────────────────
random.seed(SEED)
indices = list(range(len(train_all)))
random.shuffle(indices)

split_point = int(len(indices) * TRAIN_RATIO)
train_idx   = indices[:split_point]
dev_idx     = indices[split_point:]

train_data = [train_all[i] for i in sorted(train_idx)]
dev_data   = [train_all[i] for i in sorted(dev_idx)]

# ── verify no leakage ──────────────────────────────────────────────────────
train_ids = set(s["question_id"] for s in train_data)
dev_ids   = set(s["question_id"] for s in dev_data)
test_ids  = set(s["question_id"] for s in test_data)

assert len(train_ids & dev_ids)  == 0, "LEAKAGE: train ∩ dev"
assert len(train_ids & test_ids) == 0, "LEAKAGE: train ∩ test"
assert len(dev_ids   & test_ids) == 0, "LEAKAGE: dev ∩ test"
print("Data leakage check PASSED — no overlap between train / dev / test")

# ── save ───────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

for name, data in [("train", train_data), ("dev", dev_data), ("test", test_data)]:
    out_path = os.path.join(OUT_DIR, f"{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {name:5s} → {out_path}")

# ── dataset sizes report ───────────────────────────────────────────────────
os.makedirs(os.path.dirname(SIZE_FILE), exist_ok=True)

lines = [
    "=" * 50,
    "DATASET SIZES — KBQA_WebQSP_improve",
    "=" * 50,
    f"Source train file : {RAW_TRAIN}",
    f"Source test  file : {RAW_TEST}",
    f"Train/dev split   : {int(TRAIN_RATIO*100)} / {int((1-TRAIN_RATIO)*100)}  (seed={SEED})",
    "-" * 50,
    f"Train samples     : {len(train_data)}",
    f"Dev   samples     : {len(dev_data)}",
    f"Test  samples     : {len(test_data)}",
    "-" * 50,
    f"Total (train+dev) : {len(train_data)+len(dev_data)}",
    f"Total overall     : {len(train_data)+len(dev_data)+len(test_data)}",
    "-" * 50,
    "Leakage check     : PASSED",
    f"Train skipped     : {train_skipped}",
    f"Test  skipped     : {test_skipped}",
    "=" * 50,
]

with open(SIZE_FILE, "w") as f:
    f.write("\n".join(lines) + "\n")

print()
for l in lines:
    print(l)

# ── quick sanity check on saved files ─────────────────────────────────────
print()
print("Sanity check on saved files:")
for name in ["train", "dev", "test"]:
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path) as f:
        d = json.load(f)
    has_q   = all("question"          in s for s in d)
    has_lf  = all("sparql"            in s for s in d)
    has_ans = all("answers"           in s for s in d)
    has_inf = all("inferential_chain" in s for s in d)
    print(f"  {name:5s} | count={len(d):4d} | has_question={has_q} | has_sparql={has_lf} | has_answers={has_ans} | has_chain={has_inf}")

print()
print("Step 1 COMPLETE.")
